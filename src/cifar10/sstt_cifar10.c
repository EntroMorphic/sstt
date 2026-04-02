/*
 * sstt_cifar10.c — SSTT Bytepacked Cascade on CIFAR-10
 *
 * Tests whether the ternary cascade architecture generalizes beyond
 * 28x28 grayscale (MNIST/Fashion) to 32x32 natural color images.
 *
 * Adaptation from sstt_bytecascade.c:
 *   - CIFAR-10 binary format loader (5 train batches + 1 test batch)
 *   - RGB -> grayscale conversion (Y = 0.299R + 0.587G + 0.114B)
 *   - 32x32 image dimensions (10 blocks/row instead of 9)
 *   - Same pipeline: quantize -> encode -> vote -> dot -> kNN
 *   - Same thresholds (85/170) — no tuning for CIFAR
 *
 * This is the simplest faithful test: change the data, keep the method.
 *
 * Build: make sstt_cifar10
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N         50000
#define TEST_N          10000
#define IMG_W           32
#define IMG_H           32
#define PIXELS          1024    /* 32*32 */
#define PADDED          1024    /* 1024 % 32 == 0, AVX2-safe */
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    10      /* 10 blocks of 3 = 30 pixels, 2 unused */
#define N_BLOCKS        320     /* 32 * 10 */
#define N_BVALS         27
#define SIG_PAD         320     /* 320 % 32 == 0 */
#define BYTE_VALS       256

#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13
#define BG_JOINT        20

#define H_TRANS_PER_ROW 9       /* BLKS_PER_ROW - 1 */
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)  /* 288 */
#define V_TRANS_PER_COL 31      /* IMG_H - 1 */
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)  /* 310 */
#define TRANS_PAD       320

#define IG_SCALE        16
#define MAX_K           1000

static const char *data_dir = "data-cifar10/";

static const char *class_names[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  Data arrays
 * ================================================================ */

static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test;
static int8_t  *vgrad_train, *vgrad_test;

static uint8_t *px_sigs_tr, *px_sigs_te;
static uint8_t *hg_sigs_tr, *hg_sigs_te;
static uint8_t *vg_sigs_tr, *vg_sigs_te;
static uint8_t *ht_sigs_tr, *ht_sigs_te;
static uint8_t *vt_sigs_tr, *vt_sigs_te;
static uint8_t *joint_sigs_tr, *joint_sigs_te;

static uint32_t *joint_hot;
static uint16_t ig_weights[N_BLOCKS];
static uint8_t  nbr_table[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;

/* ================================================================
 *  CIFAR-10 data loading
 *
 *  Binary format: each record is 3073 bytes
 *    byte 0: label (0-9)
 *    bytes 1-1024: red channel (32x32 row-major)
 *    bytes 1025-2048: green channel
 *    bytes 2049-3072: blue channel
 * ================================================================ */

static void load_cifar_batch(const char *path, uint8_t *imgs, uint8_t *labels, int offset) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", path); exit(1); }
    uint8_t record[3073];
    for (int i = 0; i < 10000; i++) {
        if (fread(record, 1, 3073, f) != 3073) {
            fprintf(stderr, "ERROR: Short read in %s at image %d\n", path, i);
            fclose(f); exit(1);
        }
        labels[offset + i] = record[0];
        const uint8_t *r = record + 1;
        const uint8_t *g = record + 1 + 1024;
        const uint8_t *b = record + 1 + 2048;
        uint8_t *dst = imgs + (size_t)(offset + i) * PIXELS;
        for (int p = 0; p < 1024; p++)
            dst[p] = (uint8_t)((77 * (int)r[p] + 150 * (int)g[p] + 29 * (int)b[p]) >> 8);
    }
    fclose(f);
}

static void load_data(void) {
    raw_train_img = malloc((size_t)TRAIN_N * PIXELS);
    train_labels  = malloc(TRAIN_N);
    raw_test_img  = malloc((size_t)TEST_N * PIXELS);
    test_labels   = malloc(TEST_N);

    char path[512];
    for (int batch = 1; batch <= 5; batch++) {
        snprintf(path, sizeof(path), "%sdata_batch_%d.bin", data_dir, batch);
        load_cifar_batch(path, raw_train_img, train_labels, (batch - 1) * 10000);
        printf("  Loaded %s\n", path);
    }
    snprintf(path, sizeof(path), "%stest_batch.bin", data_dir);
    load_cifar_batch(path, raw_test_img, test_labels, 0);
    printf("  Loaded %s\n", path);

    /* Verify label distribution */
    int counts[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) counts[train_labels[i]]++;
    printf("  Train distribution:");
    for (int c = 0; c < N_CLASSES; c++) printf(" %s:%d", class_names[c], counts[c]);
    printf("\n");
}

/* ================================================================
 *  Feature computation (identical to bytecascade, constants differ)
 * ================================================================ */

static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

static void quantize_avx2(const uint8_t *src, int8_t *dst, int n_images) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));
    const __m256i one  = _mm256_set1_epi8(1);
    for (int img = 0; img < n_images; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED;
        for (int i = 0; i < PIXELS; i += 32) {
            __m256i px  = _mm256_loadu_si256((const __m256i *)(s + i));
            __m256i spx = _mm256_xor_si256(px, bias);
            __m256i pos = _mm256_cmpgt_epi8(spx, thi);
            __m256i neg = _mm256_cmpgt_epi8(tlo, spx);
            __m256i p   = _mm256_and_si256(pos, one);
            __m256i nm  = _mm256_and_si256(neg, one);
            _mm256_storeu_si256((__m256i *)(d + i), _mm256_sub_epi8(p, nm));
        }
    }
}

static void compute_gradients(const int8_t *tern, int8_t *hg, int8_t *vg, int n) {
    for (int img = 0; img < n; img++) {
        const int8_t *t = tern + (size_t)img * PADDED;
        int8_t *h = hg  + (size_t)img * PADDED;
        int8_t *v = vg  + (size_t)img * PADDED;
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W - 1; x++)
                h[y * IMG_W + x] = clamp_trit(t[y * IMG_W + x + 1] - t[y * IMG_W + x]);
            h[y * IMG_W + IMG_W - 1] = 0;
        }
        for (int y = 0; y < IMG_H - 1; y++)
            for (int x = 0; x < IMG_W; x++)
                v[y * IMG_W + x] = clamp_trit(t[(y + 1) * IMG_W + x] - t[y * IMG_W + x]);
        memset(v + (IMG_H - 1) * IMG_W, 0, IMG_W);
    }
}

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

static void compute_block_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int base = y * IMG_W + s * 3;
                sig[y * BLKS_PER_ROW + s] =
                    block_encode(img[base], img[base + 1], img[base + 2]);
            }
    }
}

static inline uint8_t trans_enc(uint8_t a, uint8_t b) {
    int8_t a0 = (a / 9) - 1, a1 = ((a / 3) % 3) - 1, a2 = (a % 3) - 1;
    int8_t b0 = (b / 9) - 1, b1 = ((b / 3) % 3) - 1, b2 = (b % 3) - 1;
    return block_encode(clamp_trit(b0 - a0), clamp_trit(b1 - a1), clamp_trit(b2 - a2));
}

static void compute_transitions(const uint8_t *bsig, int stride,
                                 uint8_t *ht, uint8_t *vt, int n) {
    for (int i = 0; i < n; i++) {
        const uint8_t *s = bsig + (size_t)i * stride;
        uint8_t *h = ht + (size_t)i * TRANS_PAD;
        uint8_t *v = vt + (size_t)i * TRANS_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int ss = 0; ss < H_TRANS_PER_ROW; ss++)
                h[y * H_TRANS_PER_ROW + ss] =
                    trans_enc(s[y * BLKS_PER_ROW + ss], s[y * BLKS_PER_ROW + ss + 1]);
        memset(h + N_HTRANS, 0xFF, TRANS_PAD - N_HTRANS);
        for (int y = 0; y < V_TRANS_PER_COL; y++)
            for (int ss = 0; ss < BLKS_PER_ROW; ss++)
                v[y * BLKS_PER_ROW + ss] =
                    trans_enc(s[y * BLKS_PER_ROW + ss], s[(y + 1) * BLKS_PER_ROW + ss]);
        memset(v + N_VTRANS, 0xFF, TRANS_PAD - N_VTRANS);
    }
}

static uint8_t encode_d(uint8_t px_bv, uint8_t hg_bv, uint8_t vg_bv,
                         uint8_t ht_bv, uint8_t vt_bv) {
    int ps = ((px_bv / 9) - 1) + (((px_bv / 3) % 3) - 1) + ((px_bv % 3) - 1);
    int hs = ((hg_bv / 9) - 1) + (((hg_bv / 3) % 3) - 1) + ((hg_bv % 3) - 1);
    int vs = ((vg_bv / 9) - 1) + (((vg_bv / 3) % 3) - 1) + ((vg_bv % 3) - 1);
    uint8_t pc = ps < 0 ? 0 : ps == 0 ? 1 : ps < 3 ? 2 : 3;
    uint8_t hc = hs < 0 ? 0 : hs == 0 ? 1 : hs < 3 ? 2 : 3;
    uint8_t vc = vs < 0 ? 0 : vs == 0 ? 1 : vs < 3 ? 2 : 3;
    uint8_t ht_a = (ht_bv != BG_TRANS) ? 1 : 0;
    uint8_t vt_a = (vt_bv != BG_TRANS) ? 1 : 0;
    return pc | (hc << 2) | (vc << 4) | (ht_a << 6) | (vt_a << 7);
}

static void compute_joint_sigs(uint8_t *out, int n,
                                const uint8_t *px_s, const uint8_t *hg_s,
                                const uint8_t *vg_s, const uint8_t *ht_s,
                                const uint8_t *vt_s) {
    for (int i = 0; i < n; i++) {
        const uint8_t *px = px_s + (size_t)i * SIG_PAD;
        const uint8_t *hg = hg_s + (size_t)i * SIG_PAD;
        const uint8_t *vg = vg_s + (size_t)i * SIG_PAD;
        const uint8_t *ht = ht_s + (size_t)i * TRANS_PAD;
        const uint8_t *vt = vt_s + (size_t)i * TRANS_PAD;
        uint8_t *os = out + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int k = y * BLKS_PER_ROW + s;
                uint8_t ht_bv = (s > 0) ? ht[y * H_TRANS_PER_ROW + (s - 1)] : BG_TRANS;
                uint8_t vt_bv = (y > 0) ? vt[(y - 1) * BLKS_PER_ROW + s]     : BG_TRANS;
                os[k] = encode_d(px[k], hg[k], vg[k], ht_bv, vt_bv);
            }
    }
}

/* ================================================================
 *  Hot map, IG, index, dot, top-K, kNN — identical to bytecascade
 * ================================================================ */

static void build_hot_map(void) {
    joint_hot = aligned_alloc(32, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);
    memset(joint_hot, 0, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            joint_hot[(size_t)k * BYTE_VALS * CLS_PAD + (size_t)sig[k] * CLS_PAD + lbl]++;
    }
}

static int classify_bayesian(const uint8_t *sig, uint8_t bg) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sig[k];
        if (bv == bg) continue;
        const uint32_t *h = joint_hot + (size_t)k * BYTE_VALS * CLS_PAD + (size_t)bv * CLS_PAD;
        double mx = 0;
        for (int c = 0; c < N_CLASSES; c++) {
            acc[c] *= (h[c] + 0.5);
            if (acc[c] > mx) mx = acc[c];
        }
        if (mx > 1e10)
            for (int c = 0; c < N_CLASSES; c++) acc[c] /= mx;
    }
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
    return best;
}

static void compute_ig_weights(uint8_t bg) {
    int class_counts[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) class_counts[train_labels[i]]++;
    double h_class = 0.0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)class_counts[c] / TRAIN_N;
        if (p > 0) h_class -= p * log2(p);
    }
    double raw_ig[N_BLOCKS], max_ig = 0.0;
    for (int k = 0; k < N_BLOCKS; k++) {
        double h_cond = 0.0;
        for (int v = 0; v < BYTE_VALS; v++) {
            if ((uint8_t)v == bg) continue;
            const uint32_t *h = joint_hot +
                (size_t)k * BYTE_VALS * CLS_PAD + (size_t)v * CLS_PAD;
            int vt = 0;
            for (int c = 0; c < N_CLASSES; c++) vt += (int)h[c];
            if (vt == 0) continue;
            double pv = (double)vt / TRAIN_N;
            double hv = 0.0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)h[c] / vt;
                if (pc > 0) hv -= pc * log2(pc);
            }
            h_cond += pv * hv;
        }
        raw_ig[k] = h_class - h_cond;
        if (raw_ig[k] > max_ig) max_ig = raw_ig[k];
    }
    for (int k = 0; k < N_BLOCKS; k++) {
        ig_weights[k] = max_ig > 0
            ? (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5) : 1;
        if (ig_weights[k] == 0) ig_weights[k] = 1;
    }
}

static void build_neighbor_table(void) {
    for (int v = 0; v < BYTE_VALS; v++)
        for (int bit = 0; bit < 8; bit++)
            nbr_table[v][bit] = (uint8_t)(v ^ (1 << bit));
}

static void build_index(uint8_t bg) {
    memset(idx_sz, 0, sizeof(idx_sz));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != bg) idx_sz[k][sig[k]]++;
    }
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < BYTE_VALS; v++) {
            idx_off[k][v] = total;
            total += idx_sz[k][v];
        }
    idx_pool = malloc((size_t)total * sizeof(uint32_t));
    if (!idx_pool) { fprintf(stderr, "ERROR: idx_pool malloc (%u entries)\n", total); exit(1); }
    uint32_t (*wpos)[BYTE_VALS] = malloc((size_t)N_BLOCKS * BYTE_VALS * sizeof(uint32_t));
    memcpy(wpos, idx_off, (size_t)N_BLOCKS * BYTE_VALS * sizeof(uint32_t));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != bg) idx_pool[wpos[k][sig[k]]++] = (uint32_t)i;
    }
    free(wpos);
    printf("  Index: %u entries (%.1f MB)\n", total, (double)total * 4 / (1024 * 1024));
}

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < PADDED; i += 32) {
        __m256i va = _mm256_load_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_load_si256((const __m256i *)(b + i));
        acc = _mm256_add_epi8(acc, _mm256_sign_epi8(va, vb));
    }
    __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc, 1));
    __m256i sum16 = _mm256_add_epi16(lo16, hi16);
    __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum32),
                              _mm256_extracti128_si256(sum32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

typedef struct { uint32_t id; uint32_t votes; int32_t dot; } candidate_t;

static int cmp_votes_desc(const void *a, const void *b) {
    return (int)((const candidate_t *)b)->votes - (int)((const candidate_t *)a)->votes;
}
static int cmp_dots_desc(const void *a, const void *b) {
    int32_t da = ((const candidate_t *)a)->dot;
    int32_t db = ((const candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

static int select_top_k(const uint32_t *votes, int n, candidate_t *out, int k) {
    uint32_t actual_max = 0;
    for (int i = 0; i < n; i++)
        if (votes[i] > actual_max) actual_max = votes[i];
    if (actual_max == 0) return 0;
    int *hist = calloc((size_t)(actual_max + 1), sizeof(int));
    for (int i = 0; i < n; i++)
        if (votes[i] > 0) hist[votes[i]]++;
    int cum = 0, thr;
    for (thr = (int)actual_max; thr >= 1; thr--) {
        cum += hist[thr];
        if (cum >= k) break;
    }
    if (thr < 1) thr = 1;
    free(hist);
    int nc = 0;
    for (int i = 0; i < n && nc < k; i++) {
        if (votes[i] >= (uint32_t)thr) {
            out[nc].id = (uint32_t)i;
            out[nc].votes = votes[i];
            out[nc].dot = 0;
            nc++;
        }
    }
    qsort(out, (size_t)nc, sizeof(candidate_t), cmp_votes_desc);
    return nc;
}

static int knn_vote(const candidate_t *cands, int nc, int k) {
    int votes[N_CLASSES] = {0};
    if (k > nc) k = nc;
    for (int i = 0; i < k; i++) votes[train_labels[cands[i].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
    return best;
}

/* ================================================================
 *  Error analysis with class names
 * ================================================================ */

static void error_analysis(const uint8_t *preds) {
    int conf[N_CLASSES][N_CLASSES];
    memset(conf, 0, sizeof(conf));
    for (int i = 0; i < TEST_N; i++) conf[test_labels[i]][preds[i]]++;

    printf("\n  Confusion Matrix (rows=actual, cols=predicted):\n");
    printf("  %12s", "");
    for (int c = 0; c < N_CLASSES; c++) printf(" %5d", c);
    printf("  | Recall\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("  %2d %-8s", r, class_names[r]);
        int rt = 0;
        for (int c = 0; c < N_CLASSES; c++) { printf(" %5d", conf[r][c]); rt += conf[r][c]; }
        printf("  | %5.1f%%\n", rt > 0 ? 100.0 * conf[r][r] / rt : 0.0);
    }

    typedef struct { int a, b, count; } pair_t;
    pair_t pairs[45]; int np = 0;
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = a + 1; b < N_CLASSES; b++)
            pairs[np++] = (pair_t){a, b, conf[a][b] + conf[b][a]};
    for (int i = 0; i < np - 1; i++)
        for (int j = i + 1; j < np; j++)
            if (pairs[j].count > pairs[i].count) {
                pair_t t = pairs[i]; pairs[i] = pairs[j]; pairs[j] = t;
            }
    printf("\n  Top-5 confused pairs:\n");
    for (int i = 0; i < 5 && i < np; i++)
        printf("    %s<->%s: %d\n",
               class_names[pairs[i].a], class_names[pairs[i].b], pairs[i].count);
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(int argc, char **argv) {
    double t0 = now_sec();

    if (argc > 1) {
        data_dir = argv[1];
        size_t len = strlen(data_dir);
        if (len > 0 && data_dir[len - 1] != '/') {
            char *buf = malloc(len + 2);
            memcpy(buf, data_dir, len);
            buf[len] = '/'; buf[len + 1] = '\0';
            data_dir = buf;
        }
    }
    printf("=== SSTT Bytepacked Cascade (CIFAR-10) ===\n\n");
    printf("  Architecture: identical to MNIST bytecascade\n");
    printf("  Adaptation: RGB->grayscale, 32x32, same thresholds (85/170)\n");
    printf("  Baseline: random = 10%%, brute kNN ~35-40%%\n\n");

    /* --- Load and featurize --- */
    printf("Loading CIFAR-10...\n");
    load_data();

    printf("\nComputing features...\n");
    tern_train  = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test   = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    hgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);

    quantize_avx2(raw_train_img, tern_train, TRAIN_N);
    quantize_avx2(raw_test_img,  tern_test,  TEST_N);

    /* Quick trit distribution check */
    {
        long tn[3] = {0};
        for (int i = 0; i < TRAIN_N; i++) {
            const int8_t *t = tern_train + (size_t)i * PADDED;
            for (int p = 0; p < PIXELS; p++) tn[t[p]+1]++;
        }
        long total = (long)TRAIN_N * PIXELS;
        printf("  Trit distribution: -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
               100.0*tn[0]/total, 100.0*tn[1]/total, 100.0*tn[2]/total);
    }

    compute_gradients(tern_train, hgrad_train, vgrad_train, TRAIN_N);
    compute_gradients(tern_test,  hgrad_test,  vgrad_test,  TEST_N);

    px_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    px_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    hg_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    hg_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    vg_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    vg_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    compute_block_sigs(tern_train,  px_sigs_tr, TRAIN_N);
    compute_block_sigs(tern_test,   px_sigs_te, TEST_N);
    compute_block_sigs(hgrad_train, hg_sigs_tr, TRAIN_N);
    compute_block_sigs(hgrad_test,  hg_sigs_te, TEST_N);
    compute_block_sigs(vgrad_train, vg_sigs_tr, TRAIN_N);
    compute_block_sigs(vgrad_test,  vg_sigs_te, TEST_N);

    ht_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    ht_sigs_te = aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    vt_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    vt_sigs_te = aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    compute_transitions(px_sigs_tr, SIG_PAD, ht_sigs_tr, vt_sigs_tr, TRAIN_N);
    compute_transitions(px_sigs_te, SIG_PAD, ht_sigs_te, vt_sigs_te, TEST_N);

    joint_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    joint_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    compute_joint_sigs(joint_sigs_tr, TRAIN_N,
                       px_sigs_tr, hg_sigs_tr, vg_sigs_tr, ht_sigs_tr, vt_sigs_tr);
    compute_joint_sigs(joint_sigs_te, TEST_N,
                       px_sigs_te, hg_sigs_te, vg_sigs_te, ht_sigs_te, vt_sigs_te);

    /* Auto-detect background */
    long val_counts[BYTE_VALS] = {0};
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) val_counts[sig[k]]++;
    }
    uint8_t bg = BG_JOINT;
    long max_count = 0;
    for (int v = 0; v < BYTE_VALS; v++)
        if (val_counts[v] > max_count) { max_count = val_counts[v]; bg = (uint8_t)v; }
    printf("  Background: value %d (%.1f%% of positions)\n",
           bg, 100.0 * max_count / ((long)TRAIN_N * N_BLOCKS));

    int n_used = 0;
    for (int v = 0; v < BYTE_VALS; v++) if (val_counts[v] > 0) n_used++;
    printf("  Unique byte values: %d / 256\n", n_used);

    printf("  Building hot map...\n");
    build_hot_map();
    printf("  Computing IG weights...\n");
    compute_ig_weights(bg);

    int ig_min = IG_SCALE, ig_max = 0;
    double ig_sum = 0;
    for (int k = 0; k < N_BLOCKS; k++) {
        if (ig_weights[k] < ig_min) ig_min = ig_weights[k];
        if (ig_weights[k] > ig_max) ig_max = ig_weights[k];
        ig_sum += ig_weights[k];
    }
    printf("  IG weights: min=%d max=%d avg=%.1f\n", ig_min, ig_max, ig_sum / N_BLOCKS);

    printf("  Building inverted index...\n");
    build_index(bg);
    build_neighbor_table();
    printf("  Built (%.2f sec)\n\n", now_sec() - t0);

    /* --- Test 1: Bayesian hot map --- */
    printf("--- Test 1: Bytepacked Bayesian Hot Map ---\n");
    {
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *sig = joint_sigs_te + (size_t)i * SIG_PAD;
            if (classify_bayesian(sig, bg) == test_labels[i]) correct++;
        }
        printf("  Bayesian: %.2f%% (%d/%d)\n\n", 100.0 * correct / TEST_N, correct, TEST_N);
    }

    /* --- Test 2: Full cascade --- */
    printf("--- Test 2: Full Bytepacked Cascade ---\n\n");

    int k_vals[]   = {50, 100, 200, 500, 1000};
    int knn_vals[] = {1, 3, 5, 7};
    int n_k = 5, n_knn = 4;

    int correct_vote = 0;
    int correct[5][4];
    memset(correct, 0, sizeof(correct));

    uint8_t *best_preds = calloc(TEST_N, 1);
    double best_acc = 0;
    int best_ki = 0, best_knni = 0;

    uint32_t *votes    = calloc(TRAIN_N, sizeof(uint32_t));
    candidate_t *cands = malloc(MAX_K * sizeof(candidate_t));

    double t_cas = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        memset(votes, 0, TRAIN_N * sizeof(uint32_t));
        const uint8_t *sig = joint_sigs_te + (size_t)i * SIG_PAD;

        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sig[k];
            if (bv == bg) continue;
            uint16_t w      = ig_weights[k];
            uint16_t w_half = w > 1 ? w / 2 : 1;
            {
                uint32_t off = idx_off[k][bv];
                uint16_t sz  = idx_sz[k][bv];
                const uint32_t *ids = idx_pool + off;
                for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w;
            }
            for (int nb = 0; nb < 8; nb++) {
                uint8_t nv = nbr_table[bv][nb];
                if (nv == bg) continue;
                uint32_t noff = idx_off[k][nv];
                uint16_t nsz  = idx_sz[k][nv];
                const uint32_t *nids = idx_pool + noff;
                for (uint16_t j = 0; j < nsz; j++) votes[nids[j]] += w_half;
            }
        }

        {
            uint32_t best_id = 0, best_v = 0;
            for (int j = 0; j < TRAIN_N; j++)
                if (votes[j] > best_v) { best_v = votes[j]; best_id = (uint32_t)j; }
            if (train_labels[best_id] == test_labels[i]) correct_vote++;
        }

        int max_nc = k_vals[n_k - 1];
        int nc = select_top_k(votes, TRAIN_N, cands, max_nc);
        const int8_t *query = tern_test + (size_t)i * PADDED;
        for (int j = 0; j < nc; j++)
            cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        for (int ki = 0; ki < n_k; ki++) {
            int eff = nc < k_vals[ki] ? nc : k_vals[ki];
            for (int knni = 0; knni < n_knn; knni++) {
                int pred = knn_vote(cands, eff, knn_vals[knni]);
                if (pred == test_labels[i]) correct[ki][knni]++;
            }
        }

        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  Cascade: %d/%d (vote %.2f%%)\r",
                    i + 1, TEST_N, 100.0 * correct_vote / (i + 1));
    }
    fprintf(stderr, "\n");
    double t_cas_end = now_sec();

    printf("  Multi-probe IG vote-only: %.2f%%\n\n", 100.0 * correct_vote / TEST_N);

    printf("  %-6s", "K\\kNN");
    for (int knni = 0; knni < n_knn; knni++) printf("  k=%-5d", knn_vals[knni]);
    printf("\n  ------");
    for (int knni = 0; knni < n_knn; knni++) (void)knni, printf("  -------");
    printf("\n");
    for (int ki = 0; ki < n_k; ki++) {
        printf("  %-6d", k_vals[ki]);
        for (int knni = 0; knni < n_knn; knni++) {
            double acc = 100.0 * correct[ki][knni] / TEST_N;
            printf("  %5.2f%%", acc);
            if (acc > best_acc) {
                best_acc = acc; best_ki = ki; best_knni = knni;
            }
        }
        printf("\n");
    }
    printf("\n  Best: K=%d k-NN=%d -> %.2f%%  (%.2f sec)\n",
           k_vals[best_ki], knn_vals[best_knni], best_acc, t_cas_end - t_cas);

    /* Re-run best config for error analysis */
    for (int i = 0; i < TEST_N; i++) {
        memset(votes, 0, TRAIN_N * sizeof(uint32_t));
        const uint8_t *sig = joint_sigs_te + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sig[k];
            if (bv == bg) continue;
            uint16_t w      = ig_weights[k];
            uint16_t w_half = w > 1 ? w / 2 : 1;
            { uint32_t off = idx_off[k][bv]; uint16_t sz = idx_sz[k][bv];
              const uint32_t *ids = idx_pool + off;
              for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w; }
            for (int nb = 0; nb < 8; nb++) {
                uint8_t nv = nbr_table[bv][nb];
                if (nv == bg) continue;
                uint32_t noff = idx_off[k][nv]; uint16_t nsz = idx_sz[k][nv];
                const uint32_t *nids = idx_pool + noff;
                for (uint16_t j = 0; j < nsz; j++) votes[nids[j]] += w_half;
            }
        }
        int eff_k = k_vals[best_ki];
        int nc = select_top_k(votes, TRAIN_N, cands, eff_k);
        const int8_t *query = tern_test + (size_t)i * PADDED;
        for (int j = 0; j < nc; j++)
            cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);
        best_preds[i] = (uint8_t)knn_vote(cands, nc, knn_vals[best_knni]);
    }

    printf("\n--- Error Analysis (K=%d k=%d) ---",
           k_vals[best_ki], knn_vals[best_knni]);
    error_analysis(best_preds);

    printf("\n=== SUMMARY ===\n");
    printf("  Random baseline:    10.00%%\n");
    printf("  Bayesian hot map:   see Test 1\n");
    printf("  Best cascade:       %.2f%%  (K=%d k=%d)\n",
           best_acc, k_vals[best_ki], knn_vals[best_knni]);
    printf("  Brute kNN baseline: ~35-40%% (literature)\n");
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    free(votes); free(cands); free(best_preds);
    free(joint_hot); free(idx_pool);
    free(joint_sigs_tr); free(joint_sigs_te);
    free(px_sigs_tr); free(px_sigs_te);
    free(hg_sigs_tr); free(hg_sigs_te);
    free(vg_sigs_tr); free(vg_sigs_te);
    free(ht_sigs_tr); free(ht_sigs_te);
    free(vt_sigs_tr); free(vt_sigs_te);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);
    return 0;
}
