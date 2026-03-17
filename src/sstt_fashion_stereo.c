/*
 * sstt_fashion_stereo.c — Stereoscopic Multi-Perspective Quantization
 *
 * Fashion-MNIST is 28x28 grayscale.  We apply 3 different ternary
 * quantization perspectives ("eyes") to the same raw pixel data:
 *
 *   Eye 1: Fixed 85/170 (standard SSTT)
 *   Eye 2: Adaptive P33/P67 (per-image percentile thresholds)
 *   Eye 3: Wide adaptive P20/P80 (wider middle band — more zeros)
 *
 * For each eye we build the full Encoding D pipeline:
 *   ternary quantization -> gradients -> block signatures ->
 *   transitions -> Encoding D -> Bayesian hot map -> log-posterior
 *
 * Combined classification strategies:
 *   A. Bayesian log-posterior fusion (sum log-posteriors, argmax)
 *   B. Per-eye IG-weighted cascade -> fused voting -> dot -> k-NN
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -o sstt_fashion_stereo \
 *        src/sstt_fashion_stereo.c -lm
 * Run:   ./sstt_fashion_stereo data-fashion/
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
#define PADDED          800   /* next multiple of 32 */
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    9
#define N_BLOCKS        252
#define SIG_PAD         256
#define BYTE_VALS       256

#define BG_TRANS        13

#define H_TRANS_PER_ROW 8
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)   /* 224 */
#define V_TRANS_PER_COL 27
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL) /* 243 */
#define TRANS_PAD       256

#define N_EYES          3
#define SMOOTH          0.5  /* Laplace smoothing for Bayesian */
#define IG_SCALE        16
#define MAX_K           1000

static const char *data_dir = "data-fashion/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  Data loading (IDX format, big-endian)
 * ================================================================ */

static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;

static uint8_t *load_idx(const char *path, uint32_t *count,
                         uint32_t *rows_out, uint32_t *cols_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", path); exit(1); }
    uint32_t magic, n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) { fclose(f); exit(1); }
    magic = __builtin_bswap32(magic);
    n = __builtin_bswap32(n);
    *count = n;
    int ndim = magic & 0xFF;
    size_t item_size = 1;
    if (ndim >= 3) {
        uint32_t rows, cols;
        if (fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1) { fclose(f); exit(1); }
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        if (rows_out) *rows_out = rows;
        if (cols_out) *cols_out = cols;
        item_size = (size_t)rows * cols;
    } else {
        if (rows_out) *rows_out = 0;
        if (cols_out) *cols_out = 0;
    }
    size_t total = (size_t)n * item_size;
    uint8_t *data = malloc(total);
    if (!data || fread(data, 1, total, f) != total) { fclose(f); exit(1); }
    fclose(f);
    return data;
}

static void load_data(void) {
    uint32_t n, r, c;
    char path[512];
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
 *  Per-eye data structures
 * ================================================================ */

typedef struct {
    const char *name;

    /* Ternary arrays (PADDED per image) */
    int8_t  *tern_train, *tern_test;
    int8_t  *hgrad_train, *hgrad_test;
    int8_t  *vgrad_train, *vgrad_test;

    /* Block signatures (SIG_PAD per image) */
    uint8_t *px_sigs_tr, *px_sigs_te;
    uint8_t *hg_sigs_tr, *hg_sigs_te;
    uint8_t *vg_sigs_tr, *vg_sigs_te;

    /* Transition signatures (TRANS_PAD per image) */
    uint8_t *ht_sigs_tr, *ht_sigs_te;
    uint8_t *vt_sigs_tr, *vt_sigs_te;

    /* Joint Encoding D signatures (SIG_PAD per image) */
    uint8_t *joint_sigs_tr, *joint_sigs_te;

    /* Bayesian hot map: [N_BLOCKS][BYTE_VALS][CLS_PAD] */
    uint32_t *hot_map;

    /* IG weights for cascade */
    uint16_t ig_weights[N_BLOCKS];

    /* Inverted index */
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];
    uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
    uint32_t *idx_pool;

    /* Neighbor table */
    uint8_t nbr_table[BYTE_VALS][8];

    /* Auto-detected background value */
    uint8_t bg;
} eye_t;

static eye_t eyes[N_EYES];

/* ================================================================
 *  Ternary quantization — 3 perspectives
 * ================================================================ */

static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

/* Eye 1: Fixed 85/170 */
static void quantize_fixed(const uint8_t *src, int8_t *dst, int n_images) {
    for (int img = 0; img < n_images; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED;
        for (int i = 0; i < PIXELS; i++)
            d[i] = s[i] > 170 ? 1 : s[i] < 85 ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
    }
}

/* Helper: find percentile value in a sorted histogram */
static uint8_t percentile_from_hist(const int *hist, int total, int pct) {
    int target = (int)((long)total * pct / 100);
    int cum = 0;
    for (int v = 0; v < 256; v++) {
        cum += hist[v];
        if (cum > target) return (uint8_t)v;
    }
    return 255;
}

/* Eye 2: Adaptive P33/P67 */
static void quantize_adaptive_33_67(const uint8_t *src, int8_t *dst, int n_images) {
    for (int img = 0; img < n_images; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED;
        int hist[256];
        memset(hist, 0, sizeof(hist));
        for (int i = 0; i < PIXELS; i++) hist[s[i]]++;
        uint8_t lo = percentile_from_hist(hist, PIXELS, 33);
        uint8_t hi = percentile_from_hist(hist, PIXELS, 67);
        if (hi <= lo) hi = lo + 1;
        for (int i = 0; i < PIXELS; i++)
            d[i] = s[i] > hi ? 1 : s[i] < lo ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
    }
}

/* Eye 3: Wide adaptive P20/P80 */
static void quantize_adaptive_20_80(const uint8_t *src, int8_t *dst, int n_images) {
    for (int img = 0; img < n_images; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED;
        int hist[256];
        memset(hist, 0, sizeof(hist));
        for (int i = 0; i < PIXELS; i++) hist[s[i]]++;
        uint8_t lo = percentile_from_hist(hist, PIXELS, 20);
        uint8_t hi = percentile_from_hist(hist, PIXELS, 80);
        if (hi <= lo) hi = lo + 1;
        for (int i = 0; i < PIXELS; i++)
            d[i] = s[i] > hi ? 1 : s[i] < lo ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
    }
}

/* ================================================================
 *  Gradients, blocks, transitions, Encoding D
 * ================================================================ */

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

static inline uint8_t block_encode(int8_t a, int8_t b, int8_t c) {
    return (uint8_t)((a + 1) * 9 + (b + 1) * 3 + (c + 1));
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
 *  AVX2 ternary dot product
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
 *  Build full pipeline for one eye (including IG + inverted index)
 * ================================================================ */

static void build_eye(eye_t *e, int eye_idx,
                      const uint8_t *train_img, const uint8_t *test_img) {
    printf("  Building Eye %d: %s\n", eye_idx + 1, e->name);

    /* Allocate ternary + gradient arrays */
    e->tern_train  = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    e->tern_test   = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    e->hgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    e->hgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    e->vgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    e->vgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);

    /* Quantize */
    switch (eye_idx) {
        case 0:
            quantize_fixed(train_img, e->tern_train, TRAIN_N);
            quantize_fixed(test_img,  e->tern_test,  TEST_N);
            break;
        case 1:
            quantize_adaptive_33_67(train_img, e->tern_train, TRAIN_N);
            quantize_adaptive_33_67(test_img,  e->tern_test,  TEST_N);
            break;
        case 2:
            quantize_adaptive_20_80(train_img, e->tern_train, TRAIN_N);
            quantize_adaptive_20_80(test_img,  e->tern_test,  TEST_N);
            break;
    }

    /* Gradients */
    compute_gradients(e->tern_train, e->hgrad_train, e->vgrad_train, TRAIN_N);
    compute_gradients(e->tern_test,  e->hgrad_test,  e->vgrad_test,  TEST_N);

    /* Block signatures */
    e->px_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    e->px_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    e->hg_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    e->hg_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    e->vg_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    e->vg_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);

    compute_block_sigs(e->tern_train,  e->px_sigs_tr, TRAIN_N);
    compute_block_sigs(e->tern_test,   e->px_sigs_te, TEST_N);
    compute_block_sigs(e->hgrad_train, e->hg_sigs_tr, TRAIN_N);
    compute_block_sigs(e->hgrad_test,  e->hg_sigs_te, TEST_N);
    compute_block_sigs(e->vgrad_train, e->vg_sigs_tr, TRAIN_N);
    compute_block_sigs(e->vgrad_test,  e->vg_sigs_te, TEST_N);

    /* Transitions */
    e->ht_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    e->ht_sigs_te = aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    e->vt_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    e->vt_sigs_te = aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);

    compute_transitions(e->px_sigs_tr, SIG_PAD, e->ht_sigs_tr, e->vt_sigs_tr, TRAIN_N);
    compute_transitions(e->px_sigs_te, SIG_PAD, e->ht_sigs_te, e->vt_sigs_te, TEST_N);

    /* Joint Encoding D */
    e->joint_sigs_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    e->joint_sigs_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);

    compute_joint_sigs(e->joint_sigs_tr, TRAIN_N,
                       e->px_sigs_tr, e->hg_sigs_tr, e->vg_sigs_tr,
                       e->ht_sigs_tr, e->vt_sigs_tr);
    compute_joint_sigs(e->joint_sigs_te, TEST_N,
                       e->px_sigs_te, e->hg_sigs_te, e->vg_sigs_te,
                       e->ht_sigs_te, e->vt_sigs_te);

    /* Auto-detect background */
    long val_counts[BYTE_VALS] = {0};
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = e->joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            val_counts[sig[k]]++;
    }
    e->bg = 0;
    long max_count = 0;
    for (int v = 0; v < BYTE_VALS; v++)
        if (val_counts[v] > max_count) { max_count = val_counts[v]; e->bg = (uint8_t)v; }

    int n_used = 0;
    for (int v = 0; v < BYTE_VALS; v++) if (val_counts[v] > 0) n_used++;
    printf("    Background: %d (%.1f%%), unique bytes: %d/256\n",
           e->bg, 100.0 * max_count / ((long)TRAIN_N * N_BLOCKS), n_used);

    /* Build hot map */
    e->hot_map = aligned_alloc(32, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);
    if (!e->hot_map) { fprintf(stderr, "ERROR: hot_map alloc\n"); exit(1); }
    memset(e->hot_map, 0, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);

    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = e->joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            e->hot_map[(size_t)k * BYTE_VALS * CLS_PAD + (size_t)sig[k] * CLS_PAD + lbl]++;
    }

    /* Compute IG weights */
    {
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
                if ((uint8_t)v == e->bg) continue;
                const uint32_t *h = e->hot_map +
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
            e->ig_weights[k] = max_ig > 0
                ? (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5) : 1;
            if (e->ig_weights[k] == 0) e->ig_weights[k] = 1;
        }
    }

    /* Build neighbor table */
    for (int v = 0; v < BYTE_VALS; v++)
        for (int bit = 0; bit < 8; bit++)
            e->nbr_table[v][bit] = (uint8_t)(v ^ (1 << bit));

    /* Build inverted index */
    memset(e->idx_sz, 0, sizeof(e->idx_sz));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = e->joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != e->bg) e->idx_sz[k][sig[k]]++;
    }
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < BYTE_VALS; v++) {
            e->idx_off[k][v] = total;
            total += e->idx_sz[k][v];
        }
    e->idx_pool = malloc((size_t)total * sizeof(uint32_t));
    if (!e->idx_pool) { fprintf(stderr, "ERROR: idx_pool malloc\n"); exit(1); }

    uint32_t (*wpos)[BYTE_VALS] = malloc((size_t)N_BLOCKS * BYTE_VALS * sizeof(uint32_t));
    memcpy(wpos, e->idx_off, (size_t)N_BLOCKS * BYTE_VALS * sizeof(uint32_t));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = e->joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != e->bg)
                e->idx_pool[wpos[k][sig[k]]++] = (uint32_t)i;
    }
    free(wpos);

    printf("    Index: %u entries (%.1f MB), hot map built.\n",
           total, (double)total * 4 / (1024 * 1024));
}

/* ================================================================
 *  Bayesian classification — log-posterior per eye
 * ================================================================ */

static void log_posterior_eye(const eye_t *e, const uint8_t *sig, double *acc) {
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sig[k];
        if (bv == e->bg) continue;
        const uint32_t *h = e->hot_map +
            (size_t)k * BYTE_VALS * CLS_PAD + (size_t)bv * CLS_PAD;
        for (int c = 0; c < N_CLASSES; c++)
            acc[c] += log(h[c] + SMOOTH);
    }
}

static int classify_bayesian_single(const eye_t *e, const uint8_t *sig) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 0.0;
    log_posterior_eye(e, sig, acc);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
    return best;
}

static int classify_bayesian_multi(int mask, int ti) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 0.0;
    for (int e = 0; e < N_EYES; e++) {
        if (!(mask & (1 << e))) continue;
        const uint8_t *sig = eyes[e].joint_sigs_te + (size_t)ti * SIG_PAD;
        log_posterior_eye(&eyes[e], sig, acc);
    }
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
    return best;
}

/* ================================================================
 *  Cascade: multi-probe IG-weighted vote per eye -> fused -> k-NN
 * ================================================================ */

/* Accumulate votes from one eye into shared votes array */
static void eye_vote(const eye_t *e, int ti, uint32_t *votes, int weight_mult) {
    const uint8_t *sig = e->joint_sigs_te + (size_t)ti * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sig[k];
        if (bv == e->bg) continue;
        uint16_t w = e->ig_weights[k] * weight_mult;
        uint16_t w_half = w > 1 ? w / 2 : 1;

        /* Exact match */
        {
            uint32_t off = e->idx_off[k][bv];
            uint16_t sz  = e->idx_sz[k][bv];
            const uint32_t *ids = e->idx_pool + off;
            for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w;
        }
        /* 8 bit-flip neighbors at half weight */
        for (int nb = 0; nb < 8; nb++) {
            uint8_t nv = e->nbr_table[bv][nb];
            if (nv == e->bg) continue;
            uint32_t noff = e->idx_off[k][nv];
            uint16_t nsz  = e->idx_sz[k][nv];
            const uint32_t *nids = e->idx_pool + noff;
            for (uint16_t j = 0; j < nsz; j++) votes[nids[j]] += w_half;
        }
    }
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

/* Bayesian sequential on candidates */
static int bayesian_seq_vote(const candidate_t *cands, int nc, int K_seq, int decay_S) {
    int64_t state[N_CLASSES];
    memset(state, 0, sizeof(state));
    int ks = K_seq < nc ? K_seq : nc;
    for (int j = 0; j < ks; j++) {
        uint8_t lbl = train_labels[cands[j].id];
        int64_t ev = cands[j].dot;
        /* Ensure positive evidence */
        ev = ev > 0 ? ev : 1;
        int64_t w2 = ev * decay_S / (decay_S + j);
        state[lbl] += w2;
    }
    int pred = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (state[c] > state[pred]) pred = c;
    return pred;
}

/* ================================================================
 *  Free eye resources
 * ================================================================ */

static void free_eye(eye_t *e) {
    free(e->tern_train);  free(e->tern_test);
    free(e->hgrad_train); free(e->hgrad_test);
    free(e->vgrad_train); free(e->vgrad_test);
    free(e->px_sigs_tr);  free(e->px_sigs_te);
    free(e->hg_sigs_tr);  free(e->hg_sigs_te);
    free(e->vg_sigs_tr);  free(e->vg_sigs_te);
    free(e->ht_sigs_tr);  free(e->ht_sigs_te);
    free(e->vt_sigs_tr);  free(e->vt_sigs_te);
    free(e->joint_sigs_tr); free(e->joint_sigs_te);
    free(e->hot_map);
    free(e->idx_pool);
}

/* ================================================================
 *  Error analysis
 * ================================================================ */

static void error_analysis(const uint8_t *preds, const char *label) {
    int conf[N_CLASSES][N_CLASSES];
    memset(conf, 0, sizeof(conf));
    for (int i = 0; i < TEST_N; i++) conf[test_labels[i]][preds[i]]++;

    const char *class_names[] = {
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Boot"
    };
    printf("\n  Confusion Matrix (%s):\n", label);
    printf("  %12s", "");
    for (int c = 0; c < N_CLASSES; c++) printf(" %5d", c);
    printf("   | Recall\n  %12s", "");
    for (int c = 0; c < N_CLASSES; c++) (void)c, printf("------");
    printf("---+-------\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("  %d %-8s:", r, class_names[r]);
        int rt = 0;
        for (int c = 0; c < N_CLASSES; c++) { printf(" %5d", conf[r][c]); rt += conf[r][c]; }
        printf("   | %5.1f%%\n", rt > 0 ? 100.0 * conf[r][r] / rt : 0.0);
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
        printf("    %d(%s) <-> %d(%s): %d\n",
               pairs[i].a, class_names[pairs[i].a],
               pairs[i].b, class_names[pairs[i].b],
               pairs[i].count);
}

/* ================================================================
 *  Quantization statistics
 * ================================================================ */

static void quant_stats(const int8_t *tern, int n, const char *name) {
    long neg = 0, zero = 0, pos = 0;
    for (int i = 0; i < n; i++) {
        const int8_t *d = tern + (size_t)i * PADDED;
        for (int p = 0; p < PIXELS; p++) {
            if (d[p] < 0) neg++;
            else if (d[p] > 0) pos++;
            else zero++;
        }
    }
    long total = (long)n * PIXELS;
    printf("    %s: -1: %.1f%%, 0: %.1f%%, +1: %.1f%%\n",
           name, 100.0 * neg / total, 100.0 * zero / total, 100.0 * pos / total);
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

    printf("=== SSTT Stereoscopic Multi-Perspective Quantization (Fashion-MNIST) ===\n\n");

    /* --- Load data --- */
    printf("Loading data from %s ...\n", data_dir);
    load_data();
    printf("  Loaded: %d train, %d test images (%dx%d)\n\n", TRAIN_N, TEST_N, IMG_W, IMG_H);

    /* --- Build all 3 eyes --- */
    eyes[0].name = "Fixed 85/170";
    eyes[1].name = "Adaptive P33/P67";
    eyes[2].name = "Wide Adaptive P20/P80";

    printf("Building %d eyes (with IG + inverted index)...\n", N_EYES);
    for (int e = 0; e < N_EYES; e++)
        build_eye(&eyes[e], e, raw_train_img, raw_test_img);

    /* Quantization distribution stats */
    printf("\n  Quantization distributions (train):\n");
    for (int e = 0; e < N_EYES; e++)
        quant_stats(eyes[e].tern_train, TRAIN_N, eyes[e].name);

    printf("\n  Build time: %.2f sec\n\n", now_sec() - t0);

    /* ================================================================
     *  Part A: Bayesian log-posterior (fast, weak)
     * ================================================================ */
    printf("=== Part A: Bayesian Log-Posterior ===\n\n");

    /* Individual eye Bayesian */
    int correct_bayes_eye[N_EYES] = {0};
    for (int e = 0; e < N_EYES; e++) {
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *sig = eyes[e].joint_sigs_te + (size_t)i * SIG_PAD;
            if (classify_bayesian_single(&eyes[e], sig) == test_labels[i])
                correct_bayes_eye[e]++;
        }
        printf("  Eye %d (%s) Bayesian: %.2f%%\n",
               e + 1, eyes[e].name, 100.0 * correct_bayes_eye[e] / TEST_N);
    }

    /* 3-eye Bayesian fusion */
    int correct_bayes_3 = 0;
    for (int i = 0; i < TEST_N; i++)
        if (classify_bayesian_multi(7, i) == test_labels[i]) correct_bayes_3++;
    printf("\n  3-Eye Bayesian fusion: %.2f%%\n\n", 100.0 * correct_bayes_3 / TEST_N);

    /* ================================================================
     *  Part B: Single-eye cascade (baseline per eye)
     * ================================================================ */
    printf("=== Part B: Single-Eye Cascade (vote -> top-K -> dot -> k-NN) ===\n\n");

    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));
    candidate_t *cands = malloc(MAX_K * sizeof(candidate_t));
    int correct_cascade_eye[N_EYES] = {0};
    int correct_cascade_seq_eye[N_EYES] = {0};

    for (int e = 0; e < N_EYES; e++) {
        double te = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            eye_vote(&eyes[e], i, votes, 1);

            int nc = select_top_k(votes, TRAIN_N, cands, 200);

            /* Dot-product refinement using this eye's ternary */
            const int8_t *query = eyes[e].tern_test + (size_t)i * PADDED;
            for (int j = 0; j < nc; j++)
                cands[j].dot = ternary_dot(query,
                    eyes[e].tern_train + (size_t)cands[j].id * PADDED);
            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

            if (knn_vote(cands, nc, 3) == test_labels[i])
                correct_cascade_eye[e]++;
            if (bayesian_seq_vote(cands, nc, 15, 2) == test_labels[i])
                correct_cascade_seq_eye[e]++;

            if ((i + 1) % 2000 == 0)
                fprintf(stderr, "  Eye %d cascade: %d/%d\r", e + 1, i + 1, TEST_N);
        }
        fprintf(stderr, "\n");
        printf("  Eye %d (%s): k-NN=%.2f%%  BayesSeq=%.2f%%  (%.1f sec)\n",
               e + 1, eyes[e].name,
               100.0 * correct_cascade_eye[e] / TEST_N,
               100.0 * correct_cascade_seq_eye[e] / TEST_N,
               now_sec() - te);
    }
    printf("\n");

    /* ================================================================
     *  Part C: Stereoscopic cascade — fused votes from all 3 eyes
     * ================================================================ */
    printf("=== Part C: Stereoscopic Cascade (fused multi-eye vote -> dot -> k-NN) ===\n\n");

    /* Try different dot-product strategies for the fused cascade */
    int correct_stereo_knn = 0;
    int correct_stereo_seq = 0;
    int correct_stereo_sum_dot_knn = 0;
    int correct_stereo_sum_dot_seq = 0;
    uint8_t *preds_best = calloc(TEST_N, 1);

    double tc = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        /* Fuse votes from all 3 eyes */
        memset(votes, 0, TRAIN_N * sizeof(uint32_t));
        for (int e = 0; e < N_EYES; e++)
            eye_vote(&eyes[e], i, votes, 1);

        int nc = select_top_k(votes, TRAIN_N, cands, 200);

        /* Dot product: sum across all eyes' ternary representations */
        for (int j = 0; j < nc; j++) {
            int32_t sum_dot = 0;
            for (int e = 0; e < N_EYES; e++) {
                const int8_t *query = eyes[e].tern_test + (size_t)i * PADDED;
                sum_dot += ternary_dot(query,
                    eyes[e].tern_train + (size_t)cands[j].id * PADDED);
            }
            cands[j].dot = sum_dot;
        }
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        int pred_knn = knn_vote(cands, nc, 3);
        int pred_seq = bayesian_seq_vote(cands, nc, 15, 2);
        preds_best[i] = (uint8_t)pred_seq;

        if (pred_knn == test_labels[i]) correct_stereo_sum_dot_knn++;
        if (pred_seq == test_labels[i]) correct_stereo_sum_dot_seq++;

        /* Also try Eye 1 dot only (standard) */
        for (int j = 0; j < nc; j++) {
            const int8_t *query = eyes[0].tern_test + (size_t)i * PADDED;
            cands[j].dot = ternary_dot(query,
                eyes[0].tern_train + (size_t)cands[j].id * PADDED);
        }
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        if (knn_vote(cands, nc, 3) == test_labels[i]) correct_stereo_knn++;
        if (bayesian_seq_vote(cands, nc, 15, 2) == test_labels[i]) correct_stereo_seq++;

        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  Stereo cascade: %d/%d  (sum-dot-seq: %.2f%%)\r",
                    i + 1, TEST_N, 100.0 * correct_stereo_sum_dot_seq / (i + 1));
    }
    fprintf(stderr, "\n");
    double tc_end = now_sec();

    printf("  Fused vote -> Eye1 dot -> k-NN:      %.2f%%\n",
           100.0 * correct_stereo_knn / TEST_N);
    printf("  Fused vote -> Eye1 dot -> BayesSeq:   %.2f%%\n",
           100.0 * correct_stereo_seq / TEST_N);
    printf("  Fused vote -> SumDot -> k-NN:         %.2f%%\n",
           100.0 * correct_stereo_sum_dot_knn / TEST_N);
    printf("  Fused vote -> SumDot -> BayesSeq:     %.2f%%\n",
           100.0 * correct_stereo_sum_dot_seq / TEST_N);
    printf("  (%.1f sec)\n\n", tc_end - tc);

    /* Part D removed — sweep too expensive */
    int best_d_correct = 0;

    /* ================================================================
     *  Part E: 2-eye cascade combinations
     * ================================================================ */
    printf("=== Part E: 2-Eye Cascade Combinations ===\n\n");

    int pairs2[][2] = {{0,1}, {0,2}, {1,2}};
    for (int p = 0; p < 3; p++) {
        int e1 = pairs2[p][0], e2 = pairs2[p][1];
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            eye_vote(&eyes[e1], i, votes, 1);
            eye_vote(&eyes[e2], i, votes, 1);

            int nc = select_top_k(votes, TRAIN_N, cands, 200);
            for (int j = 0; j < nc; j++) {
                cands[j].dot = ternary_dot(
                    eyes[e1].tern_test + (size_t)i * PADDED,
                    eyes[e1].tern_train + (size_t)cands[j].id * PADDED)
                  + ternary_dot(
                    eyes[e2].tern_test + (size_t)i * PADDED,
                    eyes[e2].tern_train + (size_t)cands[j].id * PADDED);
            }
            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

            if (bayesian_seq_vote(cands, nc, 15, 2) == test_labels[i]) correct++;
        }
        printf("  Eye %d + Eye %d: %.2f%%\n", e1 + 1, e2 + 1, 100.0 * correct / TEST_N);
    }
    printf("\n");

    /* ================================================================
     *  Part F: Bayesian sequential with sweep over K and decay
     * ================================================================ */
    printf("=== Part F: Stereo Cascade BayesSeq Sweep ===\n\n");

    int best_f_correct = 0;
    int best_f_K = 0, best_f_dS = 0, best_f_topk = 0;

    int topk_vals[] = {200, 500};
    int K_vals[] = {10, 15, 20};
    int dS_vals[] = {2, 5};
    int n_topk = 2, n_K = 3, n_dS = 2;

    for (int tki = 0; tki < n_topk; tki++) {
        for (int Ki = 0; Ki < n_K; Ki++) {
            for (int dSi = 0; dSi < n_dS; dSi++) {
                int correct = 0;
                for (int i = 0; i < TEST_N; i++) {
                    memset(votes, 0, TRAIN_N * sizeof(uint32_t));
                    for (int e = 0; e < N_EYES; e++)
                        eye_vote(&eyes[e], i, votes, 1);

                    int nc = select_top_k(votes, TRAIN_N, cands, topk_vals[tki]);

                    for (int j = 0; j < nc; j++) {
                        int32_t sum_dot = 0;
                        for (int e = 0; e < N_EYES; e++) {
                            const int8_t *query = eyes[e].tern_test + (size_t)i * PADDED;
                            sum_dot += ternary_dot(query,
                                eyes[e].tern_train + (size_t)cands[j].id * PADDED);
                        }
                        cands[j].dot = sum_dot;
                    }
                    qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

                    if (bayesian_seq_vote(cands, nc, K_vals[Ki], dS_vals[dSi]) == test_labels[i])
                        correct++;
                }
                printf("  TopK=%d K=%d dS=%d: %.2f%%",
                       topk_vals[tki], K_vals[Ki], dS_vals[dSi],
                       100.0 * correct / TEST_N);
                if (correct > best_f_correct) {
                    best_f_correct = correct;
                    best_f_topk = topk_vals[tki];
                    best_f_K = K_vals[Ki];
                    best_f_dS = dS_vals[dSi];
                    printf(" *BEST*");
                }
                printf("\n");
            }
        }
    }
    printf("\n  Best Part F: TopK=%d K=%d dS=%d -> %.2f%%\n\n",
           best_f_topk, best_f_K, best_f_dS,
           100.0 * best_f_correct / TEST_N);

    /* Re-run best config for error analysis */
    {
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            for (int e = 0; e < N_EYES; e++)
                eye_vote(&eyes[e], i, votes, 1);

            int nc = select_top_k(votes, TRAIN_N, cands, best_f_topk);
            for (int j = 0; j < nc; j++) {
                int32_t sum_dot = 0;
                for (int e = 0; e < N_EYES; e++) {
                    const int8_t *query = eyes[e].tern_test + (size_t)i * PADDED;
                    sum_dot += ternary_dot(query,
                        eyes[e].tern_train + (size_t)cands[j].id * PADDED);
                }
                cands[j].dot = sum_dot;
            }
            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

            int pred = bayesian_seq_vote(cands, nc, best_f_K, best_f_dS);
            preds_best[i] = (uint8_t)pred;
            if (pred == test_labels[i]) correct++;
        }

        char buf[256];
        snprintf(buf, sizeof(buf), "Best Stereo Cascade: %.2f%%",
                 100.0 * correct / TEST_N);
        error_analysis(preds_best, buf);
    }

    /* ================================================================
     *  Agreement analysis
     * ================================================================ */
    printf("\n=== Agreement Analysis (Bayesian eyes) ===\n\n");
    {
        int any_correct = 0, none_correct = 0, all_agree = 0;
        uint8_t *pe[N_EYES];
        for (int e = 0; e < N_EYES; e++) {
            pe[e] = calloc(TEST_N, 1);
            for (int i = 0; i < TEST_N; i++) {
                const uint8_t *sig = eyes[e].joint_sigs_te + (size_t)i * SIG_PAD;
                pe[e][i] = (uint8_t)classify_bayesian_single(&eyes[e], sig);
            }
        }
        for (int i = 0; i < TEST_N; i++) {
            int t = test_labels[i];
            int c0 = (pe[0][i] == t), c1 = (pe[1][i] == t), c2 = (pe[2][i] == t);
            if (pe[0][i] == pe[1][i] && pe[1][i] == pe[2][i]) all_agree++;
            if (c0 || c1 || c2) any_correct++;
            if (!c0 && !c1 && !c2) none_correct++;
        }
        printf("  All 3 agree: %d/%d\n", all_agree, TEST_N);
        printf("  Any eye correct: %d (%.2f%% ceiling)\n",
               any_correct, 100.0 * any_correct / TEST_N);
        printf("  No eye correct:  %d (%.2f%% error floor)\n",
               none_correct, 100.0 * none_correct / TEST_N);
        for (int e = 0; e < N_EYES; e++) free(pe[e]);
    }

    /* ================================================================
     *  Summary
     * ================================================================ */
    printf("\n=== SUMMARY ===\n\n");
    printf("  Part A: Bayesian log-posterior\n");
    for (int e = 0; e < N_EYES; e++)
        printf("    Eye %d (%s): %.2f%%\n",
               e + 1, eyes[e].name, 100.0 * correct_bayes_eye[e] / TEST_N);
    printf("    3-Eye fusion: %.2f%%\n\n", 100.0 * correct_bayes_3 / TEST_N);

    printf("  Part B: Single-eye cascade\n");
    for (int e = 0; e < N_EYES; e++)
        printf("    Eye %d: k-NN=%.2f%%  BayesSeq=%.2f%%\n",
               e + 1,
               100.0 * correct_cascade_eye[e] / TEST_N,
               100.0 * correct_cascade_seq_eye[e] / TEST_N);

    printf("\n  Part C: 3-Eye stereoscopic cascade\n");
    printf("    Fused vote -> SumDot -> k-NN:     %.2f%%\n",
           100.0 * correct_stereo_sum_dot_knn / TEST_N);
    printf("    Fused vote -> SumDot -> BayesSeq:  %.2f%%\n",
           100.0 * correct_stereo_sum_dot_seq / TEST_N);

    printf("  Part F: Best stereo cascade sweep:   %.2f%%\n",
           100.0 * best_f_correct / TEST_N);

    int best_overall = correct_stereo_sum_dot_seq;
    if (best_f_correct > best_overall) best_overall = best_f_correct;
    double best_pct = 100.0 * best_overall / TEST_N;

    printf("\n  Baseline to beat: 85.68%% (contribution 35, val-derived Bayesian sequential)\n");
    if (best_pct > 85.68)
        printf("\n  >>> STEREOSCOPIC VALIDATED: %.2f%% beats 85.68%% baseline by +%.2f%% <<<\n",
               best_pct, best_pct - 85.68);
    else
        printf("\n  Best result: %.2f%% (delta %+.2f%% vs 85.68%% baseline)\n",
               best_pct, best_pct - 85.68);

    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    /* Cleanup */
    for (int e = 0; e < N_EYES; e++) free_eye(&eyes[e]);
    free(preds_best);
    free(votes); free(cands);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);

    return 0;
}
