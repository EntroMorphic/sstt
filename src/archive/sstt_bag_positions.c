/*
 * sstt_bag_positions.c — Bagging Over Block Positions (Experiment 1)
 *
 * Random-forest-inspired ensemble for SSTT cascade classification.
 * Creates M random subsets of the 252 block positions and runs
 * independent IG-weighted votes per bag.  Candidates are merged
 * using appearance-count weighting, then ranked with multi-channel
 * dot product and classified with k=3 majority vote.
 *
 * Based on sstt_bytecascade.c — all infrastructure (data loading,
 * quantization, gradients, block signatures, transitions, Encoding D,
 * hot map, IG weights, inverted index, neighbor table, AVX2 ternary
 * dot product, top-K selection, kNN vote, error analysis) is identical.
 *
 * Build: gcc -O3 -mavx2 -lm -o sstt_bag_positions sstt_bag_positions.c
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N         60000
#define TEST_N          10000
#define IMG_W           28
#define IMG_H           28
#define PIXELS          784
#define PADDED          800
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    9
#define N_BLOCKS        252
#define N_BVALS         27
#define SIG_PAD         256
#define BYTE_VALS       256

#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13
#define BG_JOINT        20   /* Encoding D background: all-bg channels, no transitions */

#define H_TRANS_PER_ROW 8
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       256

#define IG_SCALE        16
#define MAX_K           1000

/* Bagging parameters */
#define MAX_BAGS        15
#define BAG_SEED        42
#define BAG_TOP_K       50      /* per-bag candidate count */
#define MERGE_TOP_K     200     /* merged candidate count */

static const char *data_dir = "data/";

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

/* Bytepacked hot map [N_BLOCKS][BYTE_VALS][CLS_PAD] — heap */
static uint32_t *joint_hot;

/* IG weights, neighbor table, inverted index */
static uint16_t ig_weights[N_BLOCKS];
static uint8_t  nbr_table[BYTE_VALS][8];   /* 8 bit-flip neighbors */
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;

/* Bag position masks: bag_mask[m][k] = 1 if position k is used in bag m */
static uint8_t bag_mask[MAX_BAGS][N_BLOCKS];

/* ================================================================
 *  Data loading
 * ================================================================ */

static uint8_t *load_idx(const char *path, uint32_t *count,
                         uint32_t *rows_out, uint32_t *cols_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", path); exit(1); }
    uint32_t magic, n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) { fclose(f); exit(1); }
    magic = __builtin_bswap32(magic); n = __builtin_bswap32(n); *count = n;
    int ndim = magic & 0xFF; size_t item_size = 1;
    if (ndim >= 3) {
        uint32_t rows, cols;
        if (fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1) { fclose(f); exit(1); }
        rows = __builtin_bswap32(rows); cols = __builtin_bswap32(cols);
        if (rows_out) *rows_out = rows; if (cols_out) *cols_out = cols;
        item_size = (size_t)rows * cols;
    } else { if (rows_out) *rows_out = 0; if (cols_out) *cols_out = 0; }
    size_t total = (size_t)n * item_size;
    uint8_t *data = malloc(total);
    if (!data || fread(data, 1, total, f) != total) { fclose(f); exit(1); }
    fclose(f); return data;
}

static void load_data(void) {
    uint32_t n, r, c; char path[256];
    snprintf(path, sizeof(path), "%strain-images-idx3-ubyte", data_dir);
    raw_train_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%strain-labels-idx1-ubyte", data_dir);
    train_labels = load_idx(path, &n, NULL, NULL);
    snprintf(path, sizeof(path), "%st10k-images-idx3-ubyte", data_dir);
    raw_test_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%st10k-labels-idx1-ubyte", data_dir);
    test_labels = load_idx(path, &n, NULL, NULL);
}

/* ================================================================
 *  Feature computation
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
        int i;
        for (i = 0; i + 32 <= PIXELS; i += 32) {
            __m256i px  = _mm256_loadu_si256((const __m256i *)(s + i));
            __m256i spx = _mm256_xor_si256(px, bias);
            __m256i pos = _mm256_cmpgt_epi8(spx, thi);
            __m256i neg = _mm256_cmpgt_epi8(tlo, spx);
            __m256i p   = _mm256_and_si256(pos, one);
            __m256i nm  = _mm256_and_si256(neg, one);
            _mm256_storeu_si256((__m256i *)(d + i), _mm256_sub_epi8(p, nm));
        }
        for (; i < PIXELS; i++)
            d[i] = s[i] > 170 ? 1 : s[i] < 85 ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
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
        memset(h + PIXELS, 0, PADDED - PIXELS);
        for (int y = 0; y < IMG_H - 1; y++)
            for (int x = 0; x < IMG_W; x++)
                v[y * IMG_W + x] = clamp_trit(t[(y + 1) * IMG_W + x] - t[y * IMG_W + x]);
        memset(v + (IMG_H - 1) * IMG_W, 0, IMG_W);
        memset(v + PIXELS, 0, PADDED - PIXELS);
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
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
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

/* Encoding D: majority trit x 3 channels + 2 transition activity bits */
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
        memset(os + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

/* ================================================================
 *  Hot map (Bayesian baseline)
 * ================================================================ */

static void build_hot_map(void) {
    joint_hot = aligned_alloc(32, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);
    if (!joint_hot) { fprintf(stderr, "ERROR: joint_hot alloc\n"); exit(1); }
    memset(joint_hot, 0, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            joint_hot[(size_t)k * BYTE_VALS * CLS_PAD + (size_t)sig[k] * CLS_PAD + lbl]++;
    }
}

/* ================================================================
 *  IG weights (from bytepacked hot map)
 * ================================================================ */

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

/* ================================================================
 *  Multi-probe neighbor table (bit-flip in 8-bit space)
 * ================================================================ */

static void build_neighbor_table(void) {
    for (int v = 0; v < BYTE_VALS; v++)
        for (int bit = 0; bit < 8; bit++)
            nbr_table[v][bit] = (uint8_t)(v ^ (1 << bit));
}

/* ================================================================
 *  Inverted index
 * ================================================================ */

static void build_index(uint8_t bg) {
    /* Pass 1: count bucket sizes (skip background) */
    memset(idx_sz, 0, sizeof(idx_sz));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != bg) idx_sz[k][sig[k]]++;
    }

    /* Compute offsets */
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < BYTE_VALS; v++) {
            idx_off[k][v] = total;
            total += idx_sz[k][v];
        }

    idx_pool = malloc((size_t)total * sizeof(uint32_t));
    if (!idx_pool) { fprintf(stderr, "ERROR: idx_pool malloc (%u entries)\n", total); exit(1); }

    /* wpos on heap: 252 x 256 x 4 = 258 KB */
    uint32_t (*wpos)[BYTE_VALS] = malloc((size_t)N_BLOCKS * BYTE_VALS * sizeof(uint32_t));
    if (!wpos) { fprintf(stderr, "ERROR: wpos malloc\n"); exit(1); }
    memcpy(wpos, idx_off, (size_t)N_BLOCKS * BYTE_VALS * sizeof(uint32_t));

    /* Pass 2: fill */
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != bg) idx_pool[wpos[k][sig[k]]++] = (uint32_t)i;
    }
    free(wpos);
    printf("  Index: %u entries (%.1f MB)\n", total, (double)total * 4 / (1024 * 1024));
}

/* ================================================================
 *  AVX2 ternary dot product (from sstt_v2.c)
 * ================================================================ */

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

/* ================================================================
 *  Top-K + k-NN (from sstt_v2.c)
 * ================================================================ */

typedef struct { uint32_t id; uint32_t votes; int32_t dot; } candidate_t;

/* Extended candidate for bag merging */
typedef struct {
    uint32_t id;
    uint32_t total_votes;   /* sum of votes across all bags */
    uint16_t appearance;    /* number of bags this candidate appeared in */
    int32_t  dot;           /* multi-channel dot product score */
} merged_candidate_t;

static int cmp_votes_desc(const void *a, const void *b) {
    return (int)((const candidate_t *)b)->votes - (int)((const candidate_t *)a)->votes;
}
static int cmp_dots_desc(const void *a, const void *b) {
    int32_t da = ((const candidate_t *)a)->dot;
    int32_t db = ((const candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

static int cmp_merged_desc(const void *a, const void *b) {
    const merged_candidate_t *ma = (const merged_candidate_t *)a;
    const merged_candidate_t *mb = (const merged_candidate_t *)b;
    /* Sort by appearance count DESC, then total votes DESC */
    if (mb->appearance != ma->appearance)
        return (int)mb->appearance - (int)ma->appearance;
    if (mb->total_votes != ma->total_votes)
        return (mb->total_votes > ma->total_votes) ? 1 : -1;
    return 0;
}

static int cmp_merged_dot_desc(const void *a, const void *b) {
    int32_t da = ((const merged_candidate_t *)a)->dot;
    int32_t db = ((const merged_candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

static int select_top_k(const uint32_t *votes, int n, candidate_t *out, int k) {
    uint32_t actual_max = 0;
    for (int i = 0; i < n; i++)
        if (votes[i] > actual_max) actual_max = votes[i];
    if (actual_max == 0) return 0;

    int *hist = calloc((size_t)(actual_max + 1), sizeof(int));
    if (!hist) { fprintf(stderr, "ERROR: hist malloc\n"); exit(1); }
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

static int knn_vote_cands(const candidate_t *cands, int nc, int k) {
    int votes[N_CLASSES] = {0};
    if (k > nc) k = nc;
    for (int i = 0; i < k; i++) votes[train_labels[cands[i].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
    return best;
}

static int knn_vote_merged(const merged_candidate_t *cands, int nc, int k) {
    int votes[N_CLASSES] = {0};
    if (k > nc) k = nc;
    for (int i = 0; i < k; i++) votes[train_labels[cands[i].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
    return best;
}

/* ================================================================
 *  Error analysis
 * ================================================================ */

static void error_analysis(const uint8_t *preds) {
    int conf[N_CLASSES][N_CLASSES];
    memset(conf, 0, sizeof(conf));
    for (int i = 0; i < TEST_N; i++) conf[test_labels[i]][preds[i]]++;

    printf("\n  Confusion Matrix (rows=actual, cols=predicted):\n");
    printf("       ");
    for (int c = 0; c < N_CLASSES; c++) printf(" %4d", c);
    printf("   | Recall\n  -----");
    for (int c = 0; c < N_CLASSES; c++) (void)c, printf("-----");
    printf("---+-------\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("    %d: ", r);
        int rt = 0;
        for (int c = 0; c < N_CLASSES; c++) { printf(" %4d", conf[r][c]); rt += conf[r][c]; }
        printf("   | %5.1f%%\n", rt > 0 ? 100.0 * conf[r][r] / rt : 0.0);
    }
    printf("\n  Precision: ");
    for (int c = 0; c < N_CLASSES; c++) {
        int ct = 0;
        for (int r = 0; r < N_CLASSES; r++) ct += conf[r][c];
        printf("%d:%.1f%% ", c, ct > 0 ? 100.0 * conf[c][c] / ct : 0.0);
    }
    printf("\n");

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
        printf("    %d\xE2\x86\x94%d: %d (%d\xE2\x86\x92%d: %d, %d\xE2\x86\x92%d: %d)\n",
               pairs[i].a, pairs[i].b, pairs[i].count,
               pairs[i].a, pairs[i].b, conf[pairs[i].a][pairs[i].b],
               pairs[i].b, pairs[i].a, conf[pairs[i].b][pairs[i].a]);
}

/* ================================================================
 *  LCG PRNG — deterministic, no external dependency
 * ================================================================ */

static uint32_t lcg_state;

static void lcg_seed(uint32_t s) { lcg_state = s; }

static uint32_t lcg_next(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

/* ================================================================
 *  Bag mask generation (Fisher-Yates shuffle)
 * ================================================================ */

static void generate_bag_masks(int n_bags, double bag_frac, uint32_t seed) {
    int bag_pos = (int)(bag_frac * N_BLOCKS);
    if (bag_pos < 1) bag_pos = 1;
    if (bag_pos > N_BLOCKS) bag_pos = N_BLOCKS;

    lcg_seed(seed);
    int perm[N_BLOCKS];

    for (int m = 0; m < n_bags; m++) {
        /* Initialize identity permutation */
        for (int k = 0; k < N_BLOCKS; k++) perm[k] = k;

        /* Fisher-Yates shuffle */
        for (int k = N_BLOCKS - 1; k > 0; k--) {
            int j = (int)(lcg_next() % (uint32_t)(k + 1));
            int tmp = perm[k]; perm[k] = perm[j]; perm[j] = tmp;
        }

        /* Take first bag_pos positions */
        memset(bag_mask[m], 0, N_BLOCKS);
        for (int p = 0; p < bag_pos; p++)
            bag_mask[m][perm[p]] = 1;
    }
}

/* ================================================================
 *  Multi-probe vote for a single bag (masked positions)
 * ================================================================ */

static void vote_single_bag(const uint8_t *sig, uint8_t bg,
                             const uint8_t *mask, uint32_t *votes) {
    for (int k = 0; k < N_BLOCKS; k++) {
        if (!mask[k]) continue;
        uint8_t bv = sig[k];
        if (bv == bg) continue;
        uint16_t w      = ig_weights[k];
        uint16_t w_half = w > 1 ? w / 2 : 1;

        /* Exact match */
        {
            uint32_t off = idx_off[k][bv];
            uint16_t sz  = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off;
            for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w;
        }
        /* 8 bit-flip neighbors at half weight */
        for (int nb = 0; nb < 8; nb++) {
            uint8_t nv = nbr_table[bv][nb];
            if (nv == bg) continue;
            uint32_t noff = idx_off[k][nv];
            uint16_t nsz  = idx_sz[k][nv];
            const uint32_t *nids = idx_pool + noff;
            for (uint16_t j = 0; j < nsz; j++) votes[nids[j]] += w_half;
        }
    }
}

/* Full vote (all positions, no mask) — for baseline */
static void vote_full(const uint8_t *sig, uint8_t bg, uint32_t *votes) {
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sig[k];
        if (bv == bg) continue;
        uint16_t w      = ig_weights[k];
        uint16_t w_half = w > 1 ? w / 2 : 1;

        /* Exact match */
        {
            uint32_t off = idx_off[k][bv];
            uint16_t sz  = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off;
            for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w;
        }
        /* 8 bit-flip neighbors at half weight */
        for (int nb = 0; nb < 8; nb++) {
            uint8_t nv = nbr_table[bv][nb];
            if (nv == bg) continue;
            uint32_t noff = idx_off[k][nv];
            uint16_t nsz  = idx_sz[k][nv];
            const uint32_t *nids = idx_pool + noff;
            for (uint16_t j = 0; j < nsz; j++) votes[nids[j]] += w_half;
        }
    }
}

/* ================================================================
 *  Multi-channel dot product ranking (px * 256 + vg * 192)
 * ================================================================ */

static inline int32_t multichannel_dot(int qi, uint32_t ti) {
    int32_t px_dot = ternary_dot(tern_test  + (size_t)qi * PADDED,
                                  tern_train + (size_t)ti * PADDED);
    int32_t vg_dot = ternary_dot(vgrad_test + (size_t)qi * PADDED,
                                  vgrad_train + (size_t)ti * PADDED);
    return px_dot * 256 + vg_dot * 192;
}

/* ================================================================
 *  Merge candidates from M bags using appearance-count weighting
 * ================================================================ */

static int merge_bag_candidates(candidate_t bag_cands[][BAG_TOP_K],
                                 int bag_nc[], int n_bags,
                                 merged_candidate_t *merged, int merge_k) {
    /*
     * Use a temporary hash-like array indexed by training image ID.
     * Since TRAIN_N = 60000, we can afford per-image arrays.
     */
    static uint16_t appear_count[TRAIN_N];
    static uint32_t total_votes_arr[TRAIN_N];
    static uint8_t  seen[TRAIN_N];
    memset(appear_count, 0, sizeof(appear_count));
    memset(total_votes_arr, 0, sizeof(total_votes_arr));
    memset(seen, 0, sizeof(seen));

    /* Collect unique candidate IDs */
    uint32_t *unique_ids = malloc((size_t)n_bags * BAG_TOP_K * sizeof(uint32_t));
    int n_unique = 0;

    for (int m = 0; m < n_bags; m++) {
        for (int j = 0; j < bag_nc[m]; j++) {
            uint32_t id = bag_cands[m][j].id;
            appear_count[id]++;
            total_votes_arr[id] += bag_cands[m][j].votes;
            if (!seen[id]) {
                seen[id] = 1;
                unique_ids[n_unique++] = id;
            }
        }
    }

    /* Build merged candidate array */
    int nm = n_unique < merge_k ? n_unique : merge_k;
    /* First put all into merged array, then sort and take top merge_k */
    merged_candidate_t *all_merged = malloc((size_t)n_unique * sizeof(merged_candidate_t));
    for (int j = 0; j < n_unique; j++) {
        uint32_t id = unique_ids[j];
        all_merged[j].id = id;
        all_merged[j].total_votes = total_votes_arr[id];
        all_merged[j].appearance = appear_count[id];
        all_merged[j].dot = 0;
    }

    qsort(all_merged, (size_t)n_unique, sizeof(merged_candidate_t), cmp_merged_desc);

    memcpy(merged, all_merged, (size_t)nm * sizeof(merged_candidate_t));
    free(all_merged);
    free(unique_ids);
    return nm;
}

/* ================================================================
 *  Run baseline (M=1, all positions) — should match sstt_bytecascade
 * ================================================================ */

static double run_baseline(uint8_t bg, uint8_t *preds_out) {
    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));
    candidate_t *cands = malloc(MAX_K * sizeof(candidate_t));
    int correct = 0;

    double t1 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        memset(votes, 0, TRAIN_N * sizeof(uint32_t));
        const uint8_t *sig = joint_sigs_te + (size_t)i * SIG_PAD;
        vote_full(sig, bg, votes);

        int nc = select_top_k(votes, TRAIN_N, cands, 200);
        const int8_t *query = tern_test + (size_t)i * PADDED;
        for (int j = 0; j < nc; j++)
            cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        int pred = knn_vote_cands(cands, nc, 3);
        if (preds_out) preds_out[i] = (uint8_t)pred;
        if (pred == test_labels[i]) correct++;

        if ((i + 1) % 2000 == 0)
            fprintf(stderr, "  Baseline: %d/%d (%.2f%%)\r",
                    i + 1, TEST_N, 100.0 * correct / (i + 1));
    }
    fprintf(stderr, "\n");
    double elapsed = now_sec() - t1;
    double acc = 100.0 * correct / TEST_N;
    printf("  Baseline (M=1, all pos): %.2f%%  (%.2f sec, %.0f us/query)\n",
           acc, elapsed, elapsed * 1e6 / TEST_N);
    free(votes); free(cands);
    return acc;
}

/* ================================================================
 *  Run bagged ensemble for a given (n_bags, bag_frac) config
 * ================================================================ */

typedef struct {
    int n_bags;
    double bag_frac;
    double accuracy;
    double elapsed;
    double mean_cands;
} config_result_t;

static config_result_t run_bagged(int n_bags, double bag_frac, uint8_t bg,
                                   uint8_t *preds_out) {
    config_result_t result;
    result.n_bags = n_bags;
    result.bag_frac = bag_frac;

    /* Generate masks for this configuration */
    generate_bag_masks(n_bags, bag_frac, BAG_SEED);

    int bag_pos = (int)(bag_frac * N_BLOCKS);
    printf("  Config M=%d R=%.1f (%d pos/bag):\n", n_bags, bag_frac, bag_pos);

    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));
    candidate_t (*bag_cands)[BAG_TOP_K] = malloc((size_t)n_bags * BAG_TOP_K * sizeof(candidate_t));
    int *bag_nc = malloc((size_t)n_bags * sizeof(int));
    merged_candidate_t *merged = malloc(MERGE_TOP_K * sizeof(merged_candidate_t));

    int correct = 0;
    long total_cands = 0;

    double t1 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *sig = joint_sigs_te + (size_t)i * SIG_PAD;

        /* Step a: Run M independent vote passes */
        for (int m = 0; m < n_bags; m++) {
            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            vote_single_bag(sig, bg, bag_mask[m], votes);
            bag_nc[m] = select_top_k(votes, TRAIN_N, bag_cands[m], BAG_TOP_K);
        }

        /* Step b-c: Merge candidates using appearance-count weighting */
        int nm = merge_bag_candidates(bag_cands, bag_nc, n_bags, merged, MERGE_TOP_K);
        total_cands += nm;

        /* Step d: Rank merged candidates using multi-channel dot product */
        for (int j = 0; j < nm; j++)
            merged[j].dot = multichannel_dot(i, merged[j].id);
        qsort(merged, (size_t)nm, sizeof(merged_candidate_t), cmp_merged_dot_desc);

        /* Step e: k=3 majority vote */
        int pred = knn_vote_merged(merged, nm, 3);
        if (preds_out) preds_out[i] = (uint8_t)pred;
        if (pred == test_labels[i]) correct++;

        if ((i + 1) % 2000 == 0)
            fprintf(stderr, "    M=%d R=%.1f: %d/%d (%.2f%%)\r",
                    n_bags, bag_frac, i + 1, TEST_N,
                    100.0 * correct / (i + 1));
    }
    fprintf(stderr, "\n");
    double elapsed = now_sec() - t1;

    result.accuracy = 100.0 * correct / TEST_N;
    result.elapsed = elapsed;
    result.mean_cands = (double)total_cands / TEST_N;

    printf("    Accuracy: %.2f%%  Candidates: %.1f (mean)  "
           "Time: %.2f sec (%.0f us/query)\n",
           result.accuracy, result.mean_cands, elapsed, elapsed * 1e6 / TEST_N);

    free(votes); free(bag_cands); free(bag_nc); free(merged);
    return result;
}

/* ================================================================
 *  Error mode analysis (Mode A/B/C)
 * ================================================================ */

static void error_mode_analysis(int n_bags, double bag_frac, uint8_t bg) {
    generate_bag_masks(n_bags, bag_frac, BAG_SEED);
    int bag_pos = (int)(bag_frac * N_BLOCKS);

    printf("\n--- Error Mode Analysis (M=%d R=%.1f, %d pos/bag) ---\n",
           n_bags, bag_frac, bag_pos);

    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));
    candidate_t (*bag_cands)[BAG_TOP_K] = malloc((size_t)n_bags * BAG_TOP_K * sizeof(candidate_t));
    int *bag_nc = malloc((size_t)n_bags * sizeof(int));
    merged_candidate_t *merged = malloc(MERGE_TOP_K * sizeof(merged_candidate_t));

    int mode_a = 0, mode_b = 0, mode_c = 0, n_correct = 0;
    int n_errors = 0;

    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *sig = joint_sigs_te + (size_t)i * SIG_PAD;
        int true_label = test_labels[i];

        /* Run M vote passes */
        for (int m = 0; m < n_bags; m++) {
            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            vote_single_bag(sig, bg, bag_mask[m], votes);
            bag_nc[m] = select_top_k(votes, TRAIN_N, bag_cands[m], BAG_TOP_K);
        }

        /* Merge */
        int nm = merge_bag_candidates(bag_cands, bag_nc, n_bags, merged, MERGE_TOP_K);

        /* Rank */
        for (int j = 0; j < nm; j++)
            merged[j].dot = multichannel_dot(i, merged[j].id);
        qsort(merged, (size_t)nm, sizeof(merged_candidate_t), cmp_merged_dot_desc);

        /* k=3 vote */
        int pred = knn_vote_merged(merged, nm, 3);

        if (pred == true_label) {
            n_correct++;
            continue;
        }

        n_errors++;

        /* Mode A: true label not in merged candidate set at all */
        int true_in_set = 0;
        for (int j = 0; j < nm; j++) {
            if (train_labels[merged[j].id] == true_label) {
                true_in_set = 1;
                break;
            }
        }
        if (!true_in_set) { mode_a++; continue; }

        /* Mode B: true label in set but wrong class wins dot ranking */
        /* Check: does the true class ever appear as rank-1? */
        int rank1_label = train_labels[merged[0].id];
        if (rank1_label != true_label) { mode_b++; continue; }

        /* Mode C: true label wins dot rank-1 but loses k=3 vote */
        mode_c++;
    }

    printf("  Total errors: %d / %d\n", n_errors, TEST_N);
    printf("  Mode A (true label not in merged top-K): %d (%.1f%% of errors)\n",
           mode_a, n_errors > 0 ? 100.0 * mode_a / n_errors : 0.0);
    printf("  Mode B (true label in top-K, wrong class wins dot rank): %d (%.1f%%)\n",
           mode_b, n_errors > 0 ? 100.0 * mode_b / n_errors : 0.0);
    printf("  Mode C (true label rank-1, loses k=3 vote): %d (%.1f%%)\n",
           mode_c, n_errors > 0 ? 100.0 * mode_c / n_errors : 0.0);
    printf("  Accuracy: %.2f%%\n", 100.0 * n_correct / TEST_N);

    free(votes); free(bag_cands); free(bag_nc); free(merged);
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
    const char *ds = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Bag Positions — RF-Inspired Ensemble (%s) ===\n\n", ds);

    /* --- Load and featurize (identical to sstt_bytecascade.c) --- */
    printf("Loading and building features...\n");
    load_data();

    tern_train  = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test   = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    hgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);

    quantize_avx2(raw_train_img, tern_train, TRAIN_N);
    quantize_avx2(raw_test_img,  tern_test,  TEST_N);
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

    /* ================================================================
     * Test 1: Baseline (M=1, all positions) — verify matches bytecascade
     * ================================================================ */
    printf("--- Test 1: Baseline (all positions, K=200, k=3) ---\n");
    uint8_t *baseline_preds = calloc(TEST_N, 1);
    double baseline_acc = run_baseline(bg, baseline_preds);
    printf("\n");

    /* ================================================================
     * Test 2: Parameter sweep — M={3,5,7,10} x R={0.5,0.6,0.7,0.8}
     * ================================================================ */
    printf("--- Test 2: Bagged Ensemble Parameter Sweep ---\n\n");

    int    m_vals[] = {3, 5, 7, 10};
    double r_vals[] = {0.5, 0.6, 0.7, 0.8};
    int n_m = 4, n_r = 4;

    config_result_t results[16];
    int n_results = 0;
    double best_acc = 0;
    int best_mi = 0, best_ri = 0;

    for (int mi = 0; mi < n_m; mi++) {
        for (int ri = 0; ri < n_r; ri++) {
            results[n_results] = run_bagged(m_vals[mi], r_vals[ri], bg, NULL);
            if (results[n_results].accuracy > best_acc) {
                best_acc = results[n_results].accuracy;
                best_mi = mi;
                best_ri = ri;
            }
            n_results++;
        }
    }

    /* Print parameter sweep table */
    printf("\n  Parameter Sweep Summary:\n\n");
    printf("  %-6s", "M\\R");
    for (int ri = 0; ri < n_r; ri++) printf("  R=%-5.1f  ", r_vals[ri]);
    printf("  | best\n  ------");
    for (int ri = 0; ri < n_r; ri++) (void)ri, printf("-----------");
    printf("--+------\n");
    for (int mi = 0; mi < n_m; mi++) {
        printf("  M=%-4d", m_vals[mi]);
        double row_best = 0;
        for (int ri = 0; ri < n_r; ri++) {
            int idx = mi * n_r + ri;
            printf("  %5.2f%%   ", results[idx].accuracy);
            if (results[idx].accuracy > row_best) row_best = results[idx].accuracy;
        }
        printf("  | %5.2f%%\n", row_best);
    }

    printf("\n  Best config: M=%d R=%.1f -> %.2f%% (%.2f sec)\n",
           m_vals[best_mi], r_vals[best_ri], best_acc,
           results[best_mi * n_r + best_ri].elapsed);
    printf("  Baseline:    M=1 all pos -> %.2f%%\n", baseline_acc);
    printf("  Delta:       %+.2f pp\n", best_acc - baseline_acc);

    /* Print timing and candidate table */
    printf("\n  Timing & Candidate Counts:\n\n");
    printf("  %-5s %-5s %8s %9s %12s\n", "M", "R", "Acc%", "Cands", "us/query");
    printf("  ----- ----- -------- --------- ------------\n");
    for (int i = 0; i < n_results; i++)
        printf("  %-5d %-5.1f %7.2f%% %8.1f  %11.0f\n",
               results[i].n_bags, results[i].bag_frac, results[i].accuracy,
               results[i].mean_cands, results[i].elapsed * 1e6 / TEST_N);

    /* ================================================================
     * Test 3: Error mode analysis on the best config
     * ================================================================ */
    printf("\n--- Test 3: Error Mode Analysis (best config) ---\n");
    error_mode_analysis(m_vals[best_mi], r_vals[best_ri], bg);

    /* Re-run best config to get predictions for confusion matrix */
    printf("\n--- Test 4: Full Error Analysis (best config M=%d R=%.1f) ---",
           m_vals[best_mi], r_vals[best_ri]);
    uint8_t *best_preds = calloc(TEST_N, 1);
    run_bagged(m_vals[best_mi], r_vals[best_ri], bg, best_preds);
    error_analysis(best_preds);

    /* ================================================================
     * Summary
     * ================================================================ */
    printf("\n=== SUMMARY ===\n");
    printf("  Baseline (M=1, all positions, K=200, k=3):   %.2f%%\n", baseline_acc);
    printf("  Best bagged (M=%d, R=%.1f, K_bag=%d, K_merge=%d, k=3): %.2f%%\n",
           m_vals[best_mi], r_vals[best_ri], BAG_TOP_K, MERGE_TOP_K, best_acc);
    printf("  Delta: %+.2f pp\n", best_acc - baseline_acc);
    printf("  Mean candidates (best): %.1f\n",
           results[best_mi * n_r + best_ri].mean_cands);
    printf("  3-chan reference: 96.12%%  (sstt_v2.c)\n");
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    /* Cleanup */
    free(baseline_preds); free(best_preds);
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
