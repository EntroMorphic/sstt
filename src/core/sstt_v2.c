/*
 * sstt_v2.c — Second-Generation Ternary Geometric Classifier
 *
 * Improvements over sstt_geom.c (v1):
 *   1. 2D-aware blocks (no cross-row bleed)
 *   2. Multi-channel observers: pixel + h_gradient + v_gradient
 *   3. Information-gain weighted voting
 *   4. Multi-probe (Hamming-1 neighbors at half weight)
 *   5. k-NN majority vote (k=1,3,5,7)
 *   6. Error analysis (confusion matrix, top confused pairs)
 *
 * Build: make sstt_v2  (after: make mnist)
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- Config ---------- */
#define TRAIN_N     60000
#define TEST_N      10000
#define IMG_W       28
#define IMG_H       28
#define PIXELS      784         /* 28 x 28 */
#define PADDED      800         /* 25 x 32, clean AVX2 */
#define N_CLASSES   10
#define CLS_PAD     16          /* rounded for AVX2 */
#define PAD_ITERS   25          /* PADDED / 32 */

/* 2D-aware blocks: 3-pixel horizontal strips, 9 per row */
#define BLK_W       3
#define BLKS_PER_ROW 9          /* floor(28/3) = 9, drops pixel 27 */
#define N_BLOCKS2D  (BLKS_PER_ROW * IMG_H)  /* 252 */
#define N_BVALS     27          /* 3^3 */
#define SIG2D_PAD   256         /* 8 x 32, clean AVX2 */

/* Observer channels */
#define N_CHANNELS  3           /* pixel, h_gradient, v_gradient */

/* Information gain */
#define IG_SCALE    16

/* k-NN */
#define MAX_K_NN    7
#define MAX_K       1000        /* max candidates for refinement */

/* Background block values (to skip in hot map voting):
 * Pixel:    block_encode(-1,-1,-1) = 0  (all dark = MNIST background)
 * Gradient: block_encode( 0, 0, 0) = 13 (no change = flat region) */
#define BG_PIXEL    0
#define BG_GRAD     13

/* WHT (Walsh-Hadamard / Haar Wavelet Transform) */
#define WHT_DIM       32          /* 28×28 padded to 32×32 */
#define WHT_SIZE      1024        /* 32 × 32 */

/* ---------- xoshiro256++ PRNG ---------- */
static uint64_t rs[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng(void) {
    uint64_t r = rotl(rs[0] + rs[3], 23) + rs[0];
    uint64_t t = rs[1] << 17;
    rs[2] ^= rs[0]; rs[3] ^= rs[1];
    rs[1] ^= rs[2]; rs[0] ^= rs[3];
    rs[2] ^= t; rs[3] = rotl(rs[3], 45);
    return r;
}

static void seed_rng(uint64_t v) {
    for (int i = 0; i < 4; i++) {
        v += 0x9e3779b97f4a7c15ULL;
        uint64_t z = v;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rs[i] = z ^ (z >> 31);
    }
}

/* ---------- Data directory (configurable via argv[1]) ---------- */
static const char *data_dir = "data/";

/* ---------- Timing ---------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Global data ---------- */
static uint8_t *raw_train_img;     /* [TRAIN_N * PIXELS] */
static uint8_t *raw_test_img;      /* [TEST_N * PIXELS]  */
static uint8_t *train_labels;      /* [TRAIN_N] */
static uint8_t *test_labels;       /* [TEST_N]  */

/* Pixel ternary */
static int8_t *tern_train;        /* [TRAIN_N * PADDED], 32-byte aligned */
static int8_t *tern_test;         /* [TEST_N * PADDED],  32-byte aligned */

/* Gradient ternary (same PADDED layout for dot product compatibility) */
static int8_t *hgrad_train;       /* [TRAIN_N * PADDED] */
static int8_t *hgrad_test;        /* [TEST_N * PADDED] */
static int8_t *vgrad_train;       /* [TRAIN_N * PADDED] */
static int8_t *vgrad_test;        /* [TEST_N * PADDED] */

/* 2D block signatures per channel */
static uint8_t *px_train_sigs;    /* [TRAIN_N * SIG2D_PAD] */
static uint8_t *px_test_sigs;
static uint8_t *hg_train_sigs;
static uint8_t *hg_test_sigs;
static uint8_t *vg_train_sigs;
static uint8_t *vg_test_sigs;

/* Pointers for per-channel access */
static uint8_t *chan_train_sigs[N_CHANNELS];
static uint8_t *chan_test_sigs[N_CHANNELS];

/* Information gain weights per channel */
static uint16_t ig_weights[N_CHANNELS][N_BLOCKS2D];

/* Multi-probe neighbor table */
static uint8_t nbr_table[N_BVALS][6];
static uint8_t nbr_count[N_BVALS];

/* Hot maps per channel: uint32_t [N_BLOCKS2D][N_BVALS][CLS_PAD] */
static uint32_t px_hot[N_BLOCKS2D][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS2D][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS2D][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* Pointers to hot maps for per-channel access */
static uint32_t (*chan_hot[N_CHANNELS])[N_BVALS][CLS_PAD];

/* Inverted index per channel */
static uint32_t *chan_idx_pool[N_CHANNELS];
static uint32_t chan_idx_off[N_CHANNELS][N_BLOCKS2D][N_BVALS];
static uint16_t chan_idx_sz[N_CHANNELS][N_BLOCKS2D][N_BVALS];

/* WHT (Haar wavelet) spectral data */
static int16_t *wht_train;      /* [TRAIN_N * WHT_SIZE], 32-aligned */
static int16_t *wht_test;       /* [TEST_N * WHT_SIZE], 32-aligned */
static int16_t wht_proto[N_CLASSES][WHT_SIZE] __attribute__((aligned(32)));
static int16_t wht_hg_proto[N_CLASSES][WHT_SIZE] __attribute__((aligned(32)));
static int16_t wht_vg_proto[N_CLASSES][WHT_SIZE] __attribute__((aligned(32)));

/* ================================================================
 *  MNIST IDX Loader
 * ================================================================ */

static uint8_t *load_idx(const char *path, uint32_t *count,
                         uint32_t *rows_out, uint32_t *cols_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        fprintf(stderr, "  Run 'make mnist' or 'make fashion' to download data first.\n");
        exit(1);
    }

    uint32_t magic, n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) {
        fprintf(stderr, "ERROR: Failed to read header from %s\n", path);
        fclose(f); exit(1);
    }
    magic = __builtin_bswap32(magic);
    n = __builtin_bswap32(n);
    *count = n;

    int ndim = magic & 0xFF;
    size_t item_size = 1;

    if (ndim >= 3) {
        uint32_t rows, cols;
        if (fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1) {
            fprintf(stderr, "ERROR: Failed to read dims from %s\n", path);
            fclose(f); exit(1);
        }
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
    if (!data) { fprintf(stderr, "ERROR: malloc(%zu) failed\n", total); fclose(f); exit(1); }
    if (fread(data, 1, total, f) != total) {
        fprintf(stderr, "ERROR: Short read from %s\n", path);
        fclose(f); exit(1);
    }
    fclose(f);
    return data;
}

static void load_data(const char *dir) {
    uint32_t n, r, c;
    char path[256];
    snprintf(path, sizeof(path), "%strain-images-idx3-ubyte", dir);
    raw_train_img = load_idx(path, &n, &r, &c);
    if (n != TRAIN_N || r != 28 || c != 28) {
        fprintf(stderr, "ERROR: Unexpected training image dimensions\n"); exit(1);
    }
    snprintf(path, sizeof(path), "%strain-labels-idx1-ubyte", dir);
    train_labels = load_idx(path, &n, NULL, NULL);
    if (n != TRAIN_N) { fprintf(stderr, "ERROR: Wrong train label count\n"); exit(1); }
    snprintf(path, sizeof(path), "%st10k-images-idx3-ubyte", dir);
    raw_test_img = load_idx(path, &n, &r, &c);
    if (n != TEST_N || r != 28 || c != 28) {
        fprintf(stderr, "ERROR: Unexpected test image dimensions\n"); exit(1);
    }
    snprintf(path, sizeof(path), "%st10k-labels-idx1-ubyte", dir);
    test_labels = load_idx(path, &n, NULL, NULL);
    if (n != TEST_N) { fprintf(stderr, "ERROR: Wrong test label count\n"); exit(1); }
}

/* ================================================================
 *  AVX2 Ternary Quantization
 * ================================================================ */

static void quantize_one(const uint8_t *src, int8_t *dst) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));
    const __m256i one  = _mm256_set1_epi8(1);

    int i;
    for (i = 0; i + 32 <= PIXELS; i += 32) {
        __m256i px = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i spx = _mm256_xor_si256(px, bias);
        __m256i pos = _mm256_cmpgt_epi8(spx, thi);
        __m256i neg = _mm256_cmpgt_epi8(tlo, spx);
        __m256i p = _mm256_and_si256(pos, one);
        __m256i n = _mm256_and_si256(neg, one);
        __m256i result = _mm256_sub_epi8(p, n);
        _mm256_storeu_si256((__m256i *)(dst + i), result);
    }
    for (; i < PIXELS; i++) {
        if (src[i] > 170)     dst[i] =  1;
        else if (src[i] < 85) dst[i] = -1;
        else                   dst[i] =  0;
    }
    memset(dst + PIXELS, 0, PADDED - PIXELS);
}

static void quantize_all(void) {
    tern_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    if (!tern_train || !tern_test) {
        fprintf(stderr, "ERROR: aligned_alloc failed\n"); exit(1);
    }
    for (int i = 0; i < TRAIN_N; i++)
        quantize_one(raw_train_img + (size_t)i * PIXELS,
                     tern_train + (size_t)i * PADDED);
    for (int i = 0; i < TEST_N; i++)
        quantize_one(raw_test_img + (size_t)i * PIXELS,
                     tern_test + (size_t)i * PADDED);
}

/* ================================================================
 *  AVX2 Ternary Dot Product
 * ================================================================ */

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < PADDED; i += 32) {
        __m256i va = _mm256_load_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_load_si256((const __m256i *)(b + i));
        __m256i prod = _mm256_sign_epi8(va, vb);
        acc = _mm256_add_epi8(acc, prod);
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
 *  Gradient Computation
 *
 *  h_grad[y][x] = clamp(tern[y][x+1] - tern[y][x], {-1,0,+1})
 *  v_grad[y][x] = clamp(tern[y+1][x] - tern[y][x], {-1,0,+1})
 * ================================================================ */

static inline int8_t clamp_trit(int v) {
    if (v > 0) return 1;
    if (v < 0) return -1;
    return 0;
}

static void compute_gradients_one(const int8_t *tern, int8_t *h_grad, int8_t *v_grad) {
    /* Horizontal gradient: 28 rows × 27 usable columns (store in 28×28 padded) */
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++) {
            int diff = tern[y * IMG_W + x + 1] - tern[y * IMG_W + x];
            h_grad[y * IMG_W + x] = clamp_trit(diff);
        }
        h_grad[y * IMG_W + IMG_W - 1] = 0;  /* last column = 0 */
    }
    memset(h_grad + PIXELS, 0, PADDED - PIXELS);

    /* Vertical gradient: 27 usable rows × 28 columns */
    for (int y = 0; y < IMG_H - 1; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int diff = tern[(y + 1) * IMG_W + x] - tern[y * IMG_W + x];
            v_grad[y * IMG_W + x] = clamp_trit(diff);
        }
    }
    /* Last row = 0 */
    memset(v_grad + (IMG_H - 1) * IMG_W, 0, IMG_W);
    memset(v_grad + PIXELS, 0, PADDED - PIXELS);
}

static void compute_all_gradients(void) {
    hgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    if (!hgrad_train || !hgrad_test || !vgrad_train || !vgrad_test) {
        fprintf(stderr, "ERROR: gradient alloc failed\n"); exit(1);
    }

    for (int i = 0; i < TRAIN_N; i++)
        compute_gradients_one(tern_train + (size_t)i * PADDED,
                              hgrad_train + (size_t)i * PADDED,
                              vgrad_train + (size_t)i * PADDED);
    for (int i = 0; i < TEST_N; i++)
        compute_gradients_one(tern_test + (size_t)i * PADDED,
                              hgrad_test + (size_t)i * PADDED,
                              vgrad_test + (size_t)i * PADDED);
}

/* ================================================================
 *  2D Block Signatures
 *
 *  9 horizontal strips per row × 28 rows = 252 blocks.
 *  Each strip = 3 consecutive pixels in the SAME row.
 *  No cross-row contamination.
 * ================================================================ */

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

static void compute_2d_sigs(const int8_t *img, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *src = img + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG2D_PAD;
        for (int y = 0; y < IMG_H; y++) {
            const int8_t *row = src + y * IMG_W;
            for (int s = 0; s < BLKS_PER_ROW; s++)
                sig[y * BLKS_PER_ROW + s] = block_encode(
                    row[s * BLK_W], row[s * BLK_W + 1], row[s * BLK_W + 2]);
        }
        memset(sig + N_BLOCKS2D, 0xFF, SIG2D_PAD - N_BLOCKS2D);
    }
}

static void compute_all_sigs(void) {
    px_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG2D_PAD);
    px_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG2D_PAD);
    hg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG2D_PAD);
    hg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG2D_PAD);
    vg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG2D_PAD);
    vg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG2D_PAD);
    if (!px_train_sigs || !px_test_sigs || !hg_train_sigs ||
        !hg_test_sigs || !vg_train_sigs || !vg_test_sigs) {
        fprintf(stderr, "ERROR: sig alloc failed\n"); exit(1);
    }

    compute_2d_sigs(tern_train, px_train_sigs, TRAIN_N);
    compute_2d_sigs(tern_test,  px_test_sigs,  TEST_N);
    compute_2d_sigs(hgrad_train, hg_train_sigs, TRAIN_N);
    compute_2d_sigs(hgrad_test,  hg_test_sigs,  TEST_N);
    compute_2d_sigs(vgrad_train, vg_train_sigs, TRAIN_N);
    compute_2d_sigs(vgrad_test,  vg_test_sigs,  TEST_N);

    chan_train_sigs[0] = px_train_sigs;
    chan_train_sigs[1] = hg_train_sigs;
    chan_train_sigs[2] = vg_train_sigs;
    chan_test_sigs[0]  = px_test_sigs;
    chan_test_sigs[1]  = hg_test_sigs;
    chan_test_sigs[2]  = vg_test_sigs;
}

/* ================================================================
 *  Multi-Channel Hot Maps
 * ================================================================ */

static void build_hot_map_from(const uint8_t *sigs, int n,
                                uint32_t hot[][N_BVALS][CLS_PAD]) {
    memset(hot, 0, sizeof(uint32_t) * N_BLOCKS2D * N_BVALS * CLS_PAD);
    for (int i = 0; i < n; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = sigs + (size_t)i * SIG2D_PAD;
        for (int k = 0; k < N_BLOCKS2D; k++)
            hot[k][sig[k]][lbl]++;
    }
}

static void build_all_hot_maps(void) {
    build_hot_map_from(px_train_sigs, TRAIN_N, px_hot);
    build_hot_map_from(hg_train_sigs, TRAIN_N, hg_hot);
    build_hot_map_from(vg_train_sigs, TRAIN_N, vg_hot);

    chan_hot[0] = px_hot;
    chan_hot[1] = hg_hot;
    chan_hot[2] = vg_hot;
}

/* AVX2 single-channel hot map classification.
 * bg_val = background block value to skip (BG_PIXEL=0 or BG_GRAD=13). */
static inline int hot_classify_chan(const uint8_t *qsig,
                                     uint32_t hot[][N_BVALS][CLS_PAD],
                                     uint8_t bg_val) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    for (int k = 0; k < N_BLOCKS2D; k++) {
        uint8_t bv = qsig[k];
        if (bv == bg_val) continue;
        const __m256i *cc = (const __m256i *)hot[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(cc));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(cc + 1));
    }

    uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)cv, acc_lo);
    _mm256_store_si256((__m256i *)(cv + 8), acc_hi);

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (cv[c] > cv[best]) best = c;
    return best;
}

/* Multi-channel hot map classification: sum votes from all 3 channels */
static inline int hot_classify_multi(const uint8_t *px_sig,
                                      const uint8_t *hg_sig,
                                      const uint8_t *vg_sig) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    /* Pixel channel (skip BG_PIXEL=0: all dark) */
    for (int k = 0; k < N_BLOCKS2D; k++) {
        uint8_t bv = px_sig[k];
        if (bv == BG_PIXEL) continue;
        const __m256i *cc = (const __m256i *)px_hot[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(cc));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(cc + 1));
    }
    /* H-gradient channel (skip BG_GRAD=13: flat region) */
    for (int k = 0; k < N_BLOCKS2D; k++) {
        uint8_t bv = hg_sig[k];
        if (bv == BG_GRAD) continue;
        const __m256i *cc = (const __m256i *)hg_hot[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(cc));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(cc + 1));
    }
    /* V-gradient channel (skip BG_GRAD=13: flat region) */
    for (int k = 0; k < N_BLOCKS2D; k++) {
        uint8_t bv = vg_sig[k];
        if (bv == BG_GRAD) continue;
        const __m256i *cc = (const __m256i *)vg_hot[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(cc));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(cc + 1));
    }

    uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)cv, acc_lo);
    _mm256_store_si256((__m256i *)(cv + 8), acc_hi);

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (cv[c] > cv[best]) best = c;
    return best;
}

/* ================================================================
 *  Information Gain Weights
 *
 *  IG(block_k) = H(class) - H(class | block_k)
 *  H(class | block_k) = Σ_v P(v) × H(class | block_k = v)
 * ================================================================ */

static void compute_ig_weights_chan(const uint8_t *train_sigs, int ch) {
    /* Class prior distribution */
    int class_counts[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++)
        class_counts[train_labels[i]]++;

    double h_class = 0.0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)class_counts[c] / TRAIN_N;
        if (p > 0) h_class -= p * log2(p);
    }

    double raw_ig[N_BLOCKS2D];
    double max_ig = 0.0;

    for (int k = 0; k < N_BLOCKS2D; k++) {
        /* Count class distribution for each block value */
        int counts[N_BVALS][N_CLASSES];
        int val_total[N_BVALS];
        memset(counts, 0, sizeof(counts));
        memset(val_total, 0, sizeof(val_total));

        for (int i = 0; i < TRAIN_N; i++) {
            uint8_t v = train_sigs[(size_t)i * SIG2D_PAD + k];
            int lbl = train_labels[i];
            counts[v][lbl]++;
            val_total[v]++;
        }

        /* H(class | block_k) */
        double h_cond = 0.0;
        for (int v = 0; v < N_BVALS; v++) {
            if (val_total[v] == 0) continue;
            double pv = (double)val_total[v] / TRAIN_N;
            double hv = 0.0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)counts[v][c] / val_total[v];
                if (pc > 0) hv -= pc * log2(pc);
            }
            h_cond += pv * hv;
        }

        raw_ig[k] = h_class - h_cond;
        if (raw_ig[k] > max_ig) max_ig = raw_ig[k];
    }

    /* Normalize to [0, IG_SCALE] */
    for (int k = 0; k < N_BLOCKS2D; k++) {
        if (max_ig > 0)
            ig_weights[ch][k] = (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5);
        else
            ig_weights[ch][k] = 1;
        if (ig_weights[ch][k] == 0) ig_weights[ch][k] = 1;  /* minimum weight 1 */
    }
}

static void compute_all_ig_weights(void) {
    compute_ig_weights_chan(px_train_sigs, 0);
    compute_ig_weights_chan(hg_train_sigs, 1);
    compute_ig_weights_chan(vg_train_sigs, 2);
}

/* ================================================================
 *  Multi-Probe Neighbor Table
 *
 *  For each block value v (0-26), list Hamming-distance-1 neighbors:
 *  Change one of the 3 trits to each of the 2 other values.
 *  = 3 positions × 2 alternatives = 6 neighbors max.
 * ================================================================ */

static void build_neighbor_table(void) {
    int trits[3] = {-1, 0, 1};
    for (int v = 0; v < N_BVALS; v++) {
        /* Decode v to 3 trits */
        int t0 = (v / 9) - 1;
        int t1 = ((v / 3) % 3) - 1;
        int t2 = (v % 3) - 1;
        int orig[3] = {t0, t1, t2};
        int nc = 0;

        for (int pos = 0; pos < 3; pos++) {
            for (int alt = 0; alt < 3; alt++) {
                if (trits[alt] == orig[pos]) continue;
                int mod[3] = {orig[0], orig[1], orig[2]};
                mod[pos] = trits[alt];
                uint8_t nv = block_encode((int8_t)mod[0], (int8_t)mod[1], (int8_t)mod[2]);
                nbr_table[v][nc++] = nv;
            }
        }
        nbr_count[v] = (uint8_t)nc;
    }
}

/* ================================================================
 *  Inverted Index (per channel)
 * ================================================================ */

static void build_channel_index(int ch) {
    const uint8_t *sigs = chan_train_sigs[ch];
    /* Pass 1: count bucket sizes */
    memset(chan_idx_sz[ch], 0, sizeof(chan_idx_sz[ch]));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = sigs + (size_t)i * SIG2D_PAD;
        for (int k = 0; k < N_BLOCKS2D; k++)
            chan_idx_sz[ch][k][sig[k]]++;
    }

    /* Compute offsets */
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS2D; k++)
        for (int v = 0; v < N_BVALS; v++) {
            chan_idx_off[ch][k][v] = total;
            total += chan_idx_sz[ch][k][v];
        }

    /* Allocate flat pool */
    chan_idx_pool[ch] = malloc((size_t)total * sizeof(uint32_t));
    if (!chan_idx_pool[ch]) {
        fprintf(stderr, "ERROR: idx_pool malloc ch=%d\n", ch); exit(1);
    }

    /* Pass 2: fill buckets */
    uint32_t wpos[N_BLOCKS2D][N_BVALS];
    memcpy(wpos, chan_idx_off[ch], sizeof(wpos));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = sigs + (size_t)i * SIG2D_PAD;
        for (int k = 0; k < N_BLOCKS2D; k++)
            chan_idx_pool[ch][wpos[k][sig[k]]++] = (uint32_t)i;
    }
}

static void build_all_indices(void) {
    for (int ch = 0; ch < N_CHANNELS; ch++)
        build_channel_index(ch);
}

/* ================================================================
 *  Top-K Selection + k-NN
 * ================================================================ */

typedef struct {
    uint32_t id;
    uint32_t votes;
    int32_t dot;
} candidate_t;

static int cmp_votes_desc(const void *a, const void *b) {
    return (int)((const candidate_t *)b)->votes -
           (int)((const candidate_t *)a)->votes;
}

static int cmp_dots_desc(const void *a, const void *b) {
    int32_t da = ((const candidate_t *)a)->dot;
    int32_t db = ((const candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

/* Select top-K from vote array using counting sort */
static int select_top_k(const uint32_t *votes, int n,
                        candidate_t *out, int k, int max_vote) {
    int *hist = calloc((size_t)(max_vote + 1), sizeof(int));
    for (int i = 0; i < n; i++)
        if (votes[i] > 0) hist[votes[i]]++;

    int cum = 0, thr;
    for (thr = max_vote; thr >= 1; thr--) {
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

/* k-NN majority vote on top-k candidates (already sorted by dot desc) */
static int knn_vote(const candidate_t *cands, int nc, int k) {
    int votes[N_CLASSES] = {0};
    if (k > nc) k = nc;
    for (int i = 0; i < k; i++)
        votes[train_labels[cands[i].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
    return best;
}

/* ================================================================
 *  Error Analysis
 * ================================================================ */

static void error_analysis(const uint8_t *predictions) {
    int confusion[N_CLASSES][N_CLASSES];
    memset(confusion, 0, sizeof(confusion));

    for (int i = 0; i < TEST_N; i++)
        confusion[test_labels[i]][predictions[i]]++;

    printf("\n  Confusion Matrix (rows=actual, cols=predicted):\n");
    printf("       ");
    for (int c = 0; c < N_CLASSES; c++) printf(" %4d", c);
    printf("   | Recall\n");
    printf("  -----");
    for (int c = 0; c < N_CLASSES; c++) (void)c, printf("-----");
    printf("---+-------\n");

    for (int r = 0; r < N_CLASSES; r++) {
        printf("    %d: ", r);
        int row_total = 0;
        for (int c = 0; c < N_CLASSES; c++) {
            printf(" %4d", confusion[r][c]);
            row_total += confusion[r][c];
        }
        double recall = row_total > 0 ? 100.0 * confusion[r][r] / row_total : 0;
        printf("   | %5.1f%%\n", recall);
    }

    /* Per-class precision */
    printf("\n  Precision: ");
    for (int c = 0; c < N_CLASSES; c++) {
        int col_total = 0;
        for (int r = 0; r < N_CLASSES; r++) col_total += confusion[r][c];
        double prec = col_total > 0 ? 100.0 * confusion[c][c] / col_total : 0;
        printf("%d:%.1f%% ", c, prec);
    }
    printf("\n");

    /* Top-5 most confused pairs */
    typedef struct { int a, b; int count; } pair_t;
    pair_t pairs[45];  /* C(10,2) = 45 */
    int np = 0;
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = a + 1; b < N_CLASSES; b++) {
            pairs[np].a = a;
            pairs[np].b = b;
            pairs[np].count = confusion[a][b] + confusion[b][a];
            np++;
        }
    /* Sort by confusion count descending */
    for (int i = 0; i < np - 1; i++)
        for (int j = i + 1; j < np; j++)
            if (pairs[j].count > pairs[i].count) {
                pair_t tmp = pairs[i]; pairs[i] = pairs[j]; pairs[j] = tmp;
            }

    printf("\n  Top-5 confused pairs:\n");
    for (int i = 0; i < 5 && i < np; i++)
        printf("    %d↔%d: %d errors (%d→%d: %d, %d→%d: %d)\n",
               pairs[i].a, pairs[i].b, pairs[i].count,
               pairs[i].a, pairs[i].b, confusion[pairs[i].a][pairs[i].b],
               pairs[i].b, pairs[i].a, confusion[pairs[i].b][pairs[i].a]);
}

/* ================================================================
 *  2D Haar Wavelet Transform (in-place pyramid, 32×32)
 *
 *  The ternary FFT: pure add/subtract butterflies.
 *  After 5-level pyramid decomposition:
 *    top-left 1×1 = DC (global sum)
 *    top-left 2×2 = level-4 coarse
 *    top-left 4×4 = level-3 coarse (digit shape class)
 *    top-left 8×8 = level-2 coarse (detailed shape)
 *    top-left 16×16 = level-1 coarse (fine detail)
 *    full 32×32 = all coefficients
 *
 *  Value range: int8 {-1,0,+1} input → int16 [-1024,+1024] output.
 * ================================================================ */

/* In-place 1D Walsh-Hadamard Transform (full butterfly).
 * Unlike Haar (which only recurses on the first half), WHT processes
 * ALL elements at every stage → orthogonal: H*H^T = N*I.
 * This preserves dot products: <WHT(x), WHT(y)> = N * <x, y>. */
static void wht_1d_inplace(int16_t *data, int n) {
    for (int half = n / 2; half >= 1; half /= 2) {
        for (int i = 0; i < n; i += 2 * half) {
            for (int j = 0; j < half; j++) {
                int16_t a = data[i + j];
                int16_t b = data[i + j + half];
                data[i + j]        = a + b;
                data[i + j + half] = a - b;
            }
        }
    }
}

/* Full 2D WHT: 28×28 ternary → 32×32 int16 Walsh-Hadamard.
 * Row WHT + column WHT. Preserves 2D dot products: <WHT(X), WHT(Y)> = N² * <X, Y>. */
static void wht_2d_full(const int8_t *tern_img, int16_t *out) {
    memset(out, 0, WHT_SIZE * sizeof(int16_t));
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            out[y * WHT_DIM + x] = (int16_t)tern_img[y * IMG_W + x];
    /* 1D WHT on each row */
    for (int y = 0; y < WHT_DIM; y++)
        wht_1d_inplace(out + y * WHT_DIM, WHT_DIM);
    /* 1D WHT on each column */
    int16_t col[WHT_DIM];
    for (int x = 0; x < WHT_DIM; x++) {
        for (int y = 0; y < WHT_DIM; y++)
            col[y] = out[y * WHT_DIM + x];
        wht_1d_inplace(col, WHT_DIM);
        for (int y = 0; y < WHT_DIM; y++)
            out[y * WHT_DIM + x] = col[y];
    }
}

/* ================================================================
 *  WHT Bulk Computation and Prototypes
 * ================================================================ */

static void compute_all_wht(void) {
    wht_train = (int16_t *)aligned_alloc(32,
        (size_t)TRAIN_N * WHT_SIZE * sizeof(int16_t));
    wht_test = (int16_t *)aligned_alloc(32,
        (size_t)TEST_N * WHT_SIZE * sizeof(int16_t));
    if (!wht_train || !wht_test) {
        fprintf(stderr, "ERROR: WHT alloc failed (~137 MB)\n"); exit(1);
    }
    for (int i = 0; i < TRAIN_N; i++)
        wht_2d_full(tern_train + (size_t)i * PADDED,
                     wht_train + (size_t)i * WHT_SIZE);
    for (int i = 0; i < TEST_N; i++)
        wht_2d_full(tern_test + (size_t)i * PADDED,
                     wht_test + (size_t)i * WHT_SIZE);
}

/* Average WHT coefficients per class. With the true WHT (Parseval),
 * no centering needed — dot product rankings match pixel space exactly. */
static void build_wht_prototypes(const int16_t *wht_data, int n,
                                  int16_t proto[][WHT_SIZE]) {
    int32_t acc[N_CLASSES][WHT_SIZE];
    int class_n[N_CLASSES];
    memset(acc, 0, sizeof(acc));
    memset(class_n, 0, sizeof(class_n));

    for (int i = 0; i < n; i++) {
        int lbl = train_labels[i];
        class_n[lbl]++;
        const int16_t *w = wht_data + (size_t)i * WHT_SIZE;
        for (int j = 0; j < WHT_SIZE; j++)
            acc[lbl][j] += w[j];
    }
    for (int c = 0; c < N_CLASSES; c++)
        for (int j = 0; j < WHT_SIZE; j++)
            proto[c][j] = (int16_t)(acc[c][j] / class_n[c]);
}

/* Streaming: compute WHT + accumulate per-class, no per-image storage. */
static void build_wht_proto_from_ternary(const int8_t *tern_data, int n,
                                          int16_t proto[][WHT_SIZE]) {
    int32_t acc[N_CLASSES][WHT_SIZE];
    int class_n[N_CLASSES];
    memset(acc, 0, sizeof(acc));
    memset(class_n, 0, sizeof(class_n));

    int16_t tmp[WHT_SIZE];
    for (int i = 0; i < n; i++) {
        int lbl = train_labels[i];
        class_n[lbl]++;
        wht_2d_full(tern_data + (size_t)i * PADDED, tmp);
        for (int j = 0; j < WHT_SIZE; j++)
            acc[lbl][j] += tmp[j];
    }
    for (int c = 0; c < N_CLASSES; c++)
        for (int j = 0; j < WHT_SIZE; j++)
            proto[c][j] = (int16_t)(acc[c][j] / class_n[c]);
}

/* ================================================================
 *  AVX2 Spectral Dot Product
 *
 *  _mm256_madd_epi16: int16×int16 → int32 pairwise accumulation
 *  Processes 16 int16 values per instruction.
 * ================================================================ */

/* Dot product over top-left n×n sub-block of 32-wide WHT buffers.
 * n must be power of 2 in {1, 2, 4, 8, 16, 32}. */
static inline int32_t spectral_dot_subblock(const int16_t *a,
                                              const int16_t *b, int n) {
    if (n >= 16) {
        __m256i acc = _mm256_setzero_si256();
        for (int y = 0; y < n; y++) {
            const int16_t *ra = a + y * WHT_DIM;
            const int16_t *rb = b + y * WHT_DIM;
            for (int x = 0; x < n; x += 16) {
                __m256i va = _mm256_loadu_si256((const __m256i *)(ra + x));
                __m256i vb = _mm256_loadu_si256((const __m256i *)(rb + x));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
            }
        }
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        return _mm_cvtsi128_si32(s);
    } else {
        int32_t sum = 0;
        for (int y = 0; y < n; y++) {
            const int16_t *ra = a + y * WHT_DIM;
            const int16_t *rb = b + y * WHT_DIM;
            for (int x = 0; x < n; x++)
                sum += (int32_t)ra[x] * rb[x];
        }
        return sum;
    }
}

/* Classify by max dot product against 10 class prototypes at level n */
/* Classify query against prototypes by maximum dot product.
 * With true WHT (Parseval), no centering needed. */
static int wht_classify_proto(const int16_t *query,
                               const int16_t proto[][WHT_SIZE], int n) {
    int best = 0;
    int32_t best_dot = INT32_MIN;
    for (int c = 0; c < N_CLASSES; c++) {
        int32_t d = spectral_dot_subblock(query, proto[c], n);
        if (d > best_dot) { best_dot = d; best = c; }
    }
    return best;
}

/* ================================================================
 *  Test F: Walsh-Hadamard Transform (WHT) Spectral Classifier
 *
 *  True WHT (H*H^T = N*I) preserves dot products exactly:
 *    <WHT(x), WHT(y)> = N² × <x, y>
 *  So brute WHT k-NN = pixel-space brute k-NN.
 *
 *  F1: WHT prototype classifier (nearest centroid)
 *  F2: Brute WHT k-NN (all-pairs AVX2 int16 dot products)
 * ================================================================ */

static void test_wht_spectral(void) {
    double t_start = now_sec();

    /* --- F1: WHT prototype classifier --- */
    printf("  F1: WHT prototype classifier (nearest centroid)\n\n");
    {
        int correct = 0;
        double t0 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            if (wht_classify_proto(wht_test + (size_t)i * WHT_SIZE,
                                   wht_proto, WHT_DIM) == test_labels[i])
                correct++;
        }
        double t1 = now_sec();
        printf("  Pixel WHT proto:          %.2f%% (%.0f ns/q)\n",
               100.0 * correct / TEST_N, (t1 - t0) * 1e9 / TEST_N);
    }

    /* 3-channel prototypes (compute test gradient WHTs transiently) */
    {
        int16_t *wht_hg_test = (int16_t *)aligned_alloc(32,
            (size_t)TEST_N * WHT_SIZE * sizeof(int16_t));
        int16_t *wht_vg_test = (int16_t *)aligned_alloc(32,
            (size_t)TEST_N * WHT_SIZE * sizeof(int16_t));
        if (!wht_hg_test || !wht_vg_test) {
            fprintf(stderr, "ERROR: WHT grad test alloc\n"); exit(1);
        }
        for (int i = 0; i < TEST_N; i++) {
            wht_2d_full(hgrad_test + (size_t)i * PADDED,
                         wht_hg_test + (size_t)i * WHT_SIZE);
            wht_2d_full(vgrad_test + (size_t)i * PADDED,
                         wht_vg_test + (size_t)i * WHT_SIZE);
        }
        int correct = 0;
        double t0 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            const int16_t *qp = wht_test    + (size_t)i * WHT_SIZE;
            const int16_t *qh = wht_hg_test + (size_t)i * WHT_SIZE;
            const int16_t *qv = wht_vg_test + (size_t)i * WHT_SIZE;

            int best = 0;
            int64_t best_dot = INT64_MIN;
            for (int c = 0; c < N_CLASSES; c++) {
                int64_t d = (int64_t)spectral_dot_subblock(qp, wht_proto[c], WHT_DIM)
                          + (int64_t)spectral_dot_subblock(qh, wht_hg_proto[c], WHT_DIM)
                          + (int64_t)spectral_dot_subblock(qv, wht_vg_proto[c], WHT_DIM);
                if (d > best_dot) { best_dot = d; best = c; }
            }
            if (best == test_labels[i]) correct++;
        }
        double t1 = now_sec();
        printf("  3-channel WHT proto:      %.2f%% (%.0f ns/q)\n",
               100.0 * correct / TEST_N, (t1 - t0) * 1e9 / TEST_N);
        free(wht_hg_test);
        free(wht_vg_test);
    }
    printf("\n");

    /* --- F2: Brute WHT k-NN (AVX2 int16 dot product) ---
     * WHT is orthogonal (H*H^T = N*I), so WHT dot product = N² × pixel dot product.
     * Brute WHT k-NN should match pixel-space brute k-NN. */
    printf("  F2: Brute WHT k-NN (60K × 10K AVX2 int16 dots)\n\n");

    int knn_vals[] = {1, 3, 5, 7};
    int n_knn = 4;
    int correct_knn[4] = {0};
    candidate_t *cands = malloc(TRAIN_N * sizeof(candidate_t));

    double t_brute = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const int16_t *q = wht_test + (size_t)i * WHT_SIZE;
        for (int j = 0; j < TRAIN_N; j++) {
            const int16_t *tw = wht_train + (size_t)j * WHT_SIZE;
            cands[j].id = (uint32_t)j;
            cands[j].dot = spectral_dot_subblock(q, tw, WHT_DIM);
        }
        qsort(cands, TRAIN_N, sizeof(candidate_t), cmp_dots_desc);
        for (int ki = 0; ki < n_knn; ki++) {
            int pred = knn_vote(cands, TRAIN_N, knn_vals[ki]);
            if (pred == test_labels[i]) correct_knn[ki]++;
        }
        if ((i + 1) % 2000 == 0)
            fprintf(stderr, "  WHT brute: %d/%d\r", i + 1, TEST_N);
    }
    fprintf(stderr, "\n");
    double t_brute_end = now_sec();

    printf("  WHT brute k-NN results:\n");
    for (int ki = 0; ki < n_knn; ki++)
        printf("    k=%-2d: %.2f%%\n", knn_vals[ki],
               100.0 * correct_knn[ki] / TEST_N);
    printf("\n  Brute: %.2f sec (%.1f ms/query)\n",
           t_brute_end - t_brute,
           (t_brute_end - t_brute) * 1e3 / TEST_N);

    free(cands);
    printf("  Total Test F: %.2f sec\n", now_sec() - t_start);
}

/* ================================================================
 *  Test A: 2D-Block Pixel Hot Map (single channel)
 * ================================================================ */

static void test_hotmap_2d(void) {
    /* Classify all test images */
    int correct = 0;
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *qsig = px_test_sigs + (size_t)i * SIG2D_PAD;
        if (hot_classify_chan(qsig, px_hot, BG_PIXEL) == test_labels[i])
            correct++;
    }
    double t1 = now_sec();

    printf("  Pixel hot map (252 2D blocks): %.2f%% (%.0f ns/query)\n",
           100.0 * correct / TEST_N, (t1 - t0) * 1e9 / TEST_N);
    printf("  Size: %zu KB\n", sizeof(px_hot) / 1024);
}

/* ================================================================
 *  Test B: 3-Channel Hot Map
 * ================================================================ */

static void test_multichan_hotmap(void) {
    /* Per-channel accuracy */
    const char *names[3] = {"Pixel", "H-Grad", "V-Grad"};
    const uint8_t bg_vals[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *qsig = chan_test_sigs[ch] + (size_t)i * SIG2D_PAD;
            if (hot_classify_chan(qsig, chan_hot[ch], bg_vals[ch]) == test_labels[i])
                correct++;
        }
        printf("  %s channel:    %.2f%%\n", names[ch], 100.0 * correct / TEST_N);
    }

    /* Combined 3-channel */
    int correct = 0;
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *px = px_test_sigs + (size_t)i * SIG2D_PAD;
        const uint8_t *hg = hg_test_sigs + (size_t)i * SIG2D_PAD;
        const uint8_t *vg = vg_test_sigs + (size_t)i * SIG2D_PAD;
        if (hot_classify_multi(px, hg, vg) == test_labels[i])
            correct++;
    }
    double t1 = now_sec();
    printf("  Combined 3-channel: %.2f%% (%.0f ns/query)\n",
           100.0 * correct / TEST_N, (t1 - t0) * 1e9 / TEST_N);
    printf("  Total size: %zu KB (3 × %zu KB)\n",
           (sizeof(px_hot) + sizeof(hg_hot) + sizeof(vg_hot)) / 1024,
           sizeof(px_hot) / 1024);
}

/* ================================================================
 *  Test C: Information-Gain Weighted Voting
 * ================================================================ */

static void test_ig_voting(void) {
    /* Print IG weight stats per channel */
    const char *names[3] = {"Pixel", "H-Grad", "V-Grad"};
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        int mn = IG_SCALE, mx = 0;
        double sum = 0;
        for (int k = 0; k < N_BLOCKS2D; k++) {
            if (ig_weights[ch][k] < mn) mn = ig_weights[ch][k];
            if (ig_weights[ch][k] > mx) mx = ig_weights[ch][k];
            sum += ig_weights[ch][k];
        }
        printf("  %s IG: min=%d max=%d avg=%.1f\n",
               names[ch], mn, mx, sum / N_BLOCKS2D);
    }
    printf("\n");

    /* IG-weighted single-channel (pixel) vote-only classification */
    {
        int correct = 0;
        double t0 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *qsig = px_test_sigs + (size_t)i * SIG2D_PAD;
            uint32_t votes[TRAIN_N];
            memset(votes, 0, sizeof(votes));

            for (int k = 0; k < N_BLOCKS2D; k++) {
                uint8_t bv = qsig[k];
                if (bv == 0) continue;
                uint16_t w = ig_weights[0][k];
                uint32_t off = chan_idx_off[0][k][bv];
                uint16_t sz  = chan_idx_sz[0][k][bv];
                const uint32_t *ids = chan_idx_pool[0] + off;
                for (uint16_t j = 0; j < sz; j++)
                    votes[ids[j]] += w;
            }

            /* Vote-only: find max-voted training image */
            uint32_t best_id = 0;
            uint32_t best_v = 0;
            for (int j = 0; j < TRAIN_N; j++) {
                if (votes[j] > best_v) { best_v = votes[j]; best_id = (uint32_t)j; }
            }
            if (train_labels[best_id] == test_labels[i]) correct++;
        }
        double t1 = now_sec();
        printf("  IG-weighted pixel vote-only: %.2f%% (%.2f sec)\n",
               100.0 * correct / TEST_N, t1 - t0);
    }

    /* IG-weighted 3-channel vote-only */
    {
        int correct = 0;
        double t0 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            uint32_t votes[TRAIN_N];
            memset(votes, 0, sizeof(votes));

            for (int ch = 0; ch < N_CHANNELS; ch++) {
                const uint8_t *qsig = chan_test_sigs[ch] + (size_t)i * SIG2D_PAD;
                for (int k = 0; k < N_BLOCKS2D; k++) {
                    uint8_t bv = qsig[k];
                    if (bv == 0) continue;
                    uint16_t w = ig_weights[ch][k];
                    uint32_t off = chan_idx_off[ch][k][bv];
                    uint16_t sz  = chan_idx_sz[ch][k][bv];
                    const uint32_t *ids = chan_idx_pool[ch] + off;
                    for (uint16_t j = 0; j < sz; j++)
                        votes[ids[j]] += w;
                }
            }

            uint32_t best_id = 0;
            uint32_t best_v = 0;
            for (int j = 0; j < TRAIN_N; j++) {
                if (votes[j] > best_v) { best_v = votes[j]; best_id = (uint32_t)j; }
            }
            if (train_labels[best_id] == test_labels[i]) correct++;
        }
        double t1 = now_sec();
        printf("  IG-weighted 3-channel vote-only: %.2f%% (%.2f sec)\n",
               100.0 * correct / TEST_N, t1 - t0);
    }
}

/* ================================================================
 *  Test D: Multi-Probe Voting
 * ================================================================ */

static void test_multiprobe(void) {
    /* Multi-probe IG-weighted 3-channel vote-only */
    int correct_exact = 0;
    int correct_probe = 0;
    double t0 = now_sec();

    for (int i = 0; i < TEST_N; i++) {
        uint32_t votes_exact[TRAIN_N];
        uint32_t votes_probe[TRAIN_N];
        memset(votes_exact, 0, sizeof(votes_exact));
        memset(votes_probe, 0, sizeof(votes_probe));

        for (int ch = 0; ch < N_CHANNELS; ch++) {
            const uint8_t *qsig = chan_test_sigs[ch] + (size_t)i * SIG2D_PAD;
            for (int k = 0; k < N_BLOCKS2D; k++) {
                uint8_t bv = qsig[k];
                if (bv == 0) continue;
                uint16_t w = ig_weights[ch][k];
                uint16_t w_half = w / 2;
                if (w_half == 0) w_half = 1;

                /* Exact match */
                uint32_t off = chan_idx_off[ch][k][bv];
                uint16_t sz  = chan_idx_sz[ch][k][bv];
                const uint32_t *ids = chan_idx_pool[ch] + off;
                for (uint16_t j = 0; j < sz; j++) {
                    votes_exact[ids[j]] += w;
                    votes_probe[ids[j]] += w;
                }

                /* Hamming-1 neighbors at half weight */
                for (int n = 0; n < nbr_count[bv]; n++) {
                    uint8_t nv = nbr_table[bv][n];
                    uint32_t noff = chan_idx_off[ch][k][nv];
                    uint16_t nsz  = chan_idx_sz[ch][k][nv];
                    const uint32_t *nids = chan_idx_pool[ch] + noff;
                    for (uint16_t j = 0; j < nsz; j++)
                        votes_probe[nids[j]] += w_half;
                }
            }
        }

        /* Vote-only for exact */
        {
            uint32_t best_id = 0, best_v = 0;
            for (int j = 0; j < TRAIN_N; j++)
                if (votes_exact[j] > best_v) { best_v = votes_exact[j]; best_id = (uint32_t)j; }
            if (train_labels[best_id] == test_labels[i]) correct_exact++;
        }
        /* Vote-only for probe */
        {
            uint32_t best_id = 0, best_v = 0;
            for (int j = 0; j < TRAIN_N; j++)
                if (votes_probe[j] > best_v) { best_v = votes_probe[j]; best_id = (uint32_t)j; }
            if (train_labels[best_id] == test_labels[i]) correct_probe++;
        }

        if ((i + 1) % 2000 == 0)
            fprintf(stderr, "  Multi-probe: %d/%d\r", i + 1, TEST_N);
    }
    fprintf(stderr, "\n");
    double t1 = now_sec();

    printf("  Exact-only 3-chan IG vote: %.2f%%\n", 100.0 * correct_exact / TEST_N);
    printf("  Multi-probe 3-chan IG vote: %.2f%% (%.2f sec)\n",
           100.0 * correct_probe / TEST_N, t1 - t0);
}

/* ================================================================
 *  Test E: Full Cascade — Weighted Multi-Probe Vote → Top-K → k-NN
 * ================================================================ */

static void test_full_cascade(double *best_acc_out, uint8_t *best_preds_out) {
    int k_vals[] = {50, 100, 200, 500, 1000};
    int n_k = 5;
    int knn_vals[] = {1, 3, 5, 7};
    int n_knn = 4;

    /* Track best accuracy for each (K, knn) combination */
    int correct[5][4];
    memset(correct, 0, sizeof(correct));

    /* Also track vote-only */
    int correct_vote = 0;

    /* For error analysis: store predictions at best config */
    uint8_t *preds = calloc(TEST_N, 1);
    double best_acc = 0;
    int best_ki = 0, best_knni = 0;

    /* Max possible weighted vote for counting sort */
    int max_vote = 0;
    for (int ch = 0; ch < N_CHANNELS; ch++)
        for (int k = 0; k < N_BLOCKS2D; k++)
            max_vote += ig_weights[ch][k];
    /* Multi-probe can roughly double the vote total */
    max_vote *= 2;

    double t0 = now_sec();

    for (int i = 0; i < TEST_N; i++) {
        uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));

        /* Multi-probe weighted voting across all channels */
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            const uint8_t *qsig = chan_test_sigs[ch] + (size_t)i * SIG2D_PAD;
            for (int k = 0; k < N_BLOCKS2D; k++) {
                uint8_t bv = qsig[k];
                if (bv == 0) continue;
                uint16_t w = ig_weights[ch][k];
                uint16_t w_half = w / 2;
                if (w_half == 0) w_half = 1;

                /* Exact */
                uint32_t off = chan_idx_off[ch][k][bv];
                uint16_t sz  = chan_idx_sz[ch][k][bv];
                const uint32_t *ids = chan_idx_pool[ch] + off;
                for (uint16_t j = 0; j < sz; j++)
                    votes[ids[j]] += w;

                /* Neighbors */
                for (int nb = 0; nb < nbr_count[bv]; nb++) {
                    uint8_t nv = nbr_table[bv][nb];
                    uint32_t noff = chan_idx_off[ch][k][nv];
                    uint16_t nsz  = chan_idx_sz[ch][k][nv];
                    const uint32_t *nids = chan_idx_pool[ch] + noff;
                    for (uint16_t j = 0; j < nsz; j++)
                        votes[nids[j]] += w_half;
                }
            }
        }

        /* Vote-only classification */
        {
            uint32_t best_id = 0, best_v = 0;
            for (int j = 0; j < TRAIN_N; j++)
                if (votes[j] > best_v) { best_v = votes[j]; best_id = (uint32_t)j; }
            if (train_labels[best_id] == test_labels[i]) correct_vote++;
        }

        /* Top-K → dot product refinement → k-NN */
        candidate_t cands[MAX_K];
        int max_nc = k_vals[n_k - 1];  /* largest K */
        int nc = select_top_k(votes, TRAIN_N, cands, max_nc, max_vote);

        /* Compute dot products for all candidates (pixel channel) */
        const int8_t *query = tern_test + (size_t)i * PADDED;
        for (int j = 0; j < nc; j++)
            cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);

        /* Sort by dot descending */
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        /* Score each (K, knn) */
        for (int ki = 0; ki < n_k; ki++) {
            int eff_nc = nc < k_vals[ki] ? nc : k_vals[ki];
            for (int knni = 0; knni < n_knn; knni++) {
                int pred = knn_vote(cands, eff_nc, knn_vals[knni]);
                if (pred == test_labels[i]) correct[ki][knni]++;
            }
        }

        free(votes);

        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  Cascade: %d/%d (vote acc %.2f%%)\r",
                    i + 1, TEST_N, 100.0 * correct_vote / (i + 1));
    }
    fprintf(stderr, "\n");
    double t1 = now_sec();

    printf("  Multi-probe weighted vote-only: %.2f%%\n\n",
           100.0 * correct_vote / TEST_N);

    printf("  %-6s", "K\\kNN");
    for (int knni = 0; knni < n_knn; knni++)
        printf("  k=%-5d", knn_vals[knni]);
    printf("\n  ------");
    for (int knni = 0; knni < n_knn; knni++) (void)knni, printf("  -------");
    printf("\n");

    for (int ki = 0; ki < n_k; ki++) {
        printf("  %-6d", k_vals[ki]);
        for (int knni = 0; knni < n_knn; knni++) {
            double acc = 100.0 * correct[ki][knni] / TEST_N;
            printf("  %5.2f%%", acc);
            if (acc > best_acc) {
                best_acc = acc;
                best_ki = ki;
                best_knni = knni;
            }
        }
        printf("\n");
    }

    printf("\n  Best: K=%d, k-NN=%d → %.2f%% (%.2f sec total)\n",
           k_vals[best_ki], knn_vals[best_knni], best_acc, t1 - t0);

    /* Re-run best config to get predictions for error analysis */
    for (int i = 0; i < TEST_N; i++) {
        uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));

        for (int ch = 0; ch < N_CHANNELS; ch++) {
            const uint8_t *qsig = chan_test_sigs[ch] + (size_t)i * SIG2D_PAD;
            for (int k = 0; k < N_BLOCKS2D; k++) {
                uint8_t bv = qsig[k];
                if (bv == 0) continue;
                uint16_t w = ig_weights[ch][k];
                uint16_t w_half = w / 2;
                if (w_half == 0) w_half = 1;

                uint32_t off = chan_idx_off[ch][k][bv];
                uint16_t sz  = chan_idx_sz[ch][k][bv];
                const uint32_t *ids = chan_idx_pool[ch] + off;
                for (uint16_t j = 0; j < sz; j++)
                    votes[ids[j]] += w;

                for (int nb = 0; nb < nbr_count[bv]; nb++) {
                    uint8_t nv = nbr_table[bv][nb];
                    uint32_t noff = chan_idx_off[ch][k][nv];
                    uint16_t nsz  = chan_idx_sz[ch][k][nv];
                    const uint32_t *nids = chan_idx_pool[ch] + noff;
                    for (uint16_t j = 0; j < nsz; j++)
                        votes[nids[j]] += w_half;
                }
            }
        }

        candidate_t cands[MAX_K];
        int eff_k = k_vals[best_ki];
        int nc = select_top_k(votes, TRAIN_N, cands, eff_k, max_vote);

        const int8_t *query = tern_test + (size_t)i * PADDED;
        for (int j = 0; j < nc; j++)
            cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        preds[i] = (uint8_t)knn_vote(cands, nc, knn_vals[best_knni]);
        free(votes);
    }

    *best_acc_out = best_acc;
    memcpy(best_preds_out, preds, TEST_N);
    free(preds);
}

/* ================================================================
 *  Test G: Spectral Coarse-to-Fine k-NN
 *
 *  Stage 1: Prototype filter — keep top 3 classes by WHT prototype dot.
 *  Stage 2: Coarse spectral pruning — partial WHT dot (first C coeffs)
 *           over surviving-class training images, keep top K.
 *  Stage 3: Full dot product refinement on K survivors.
 *  Stage 4: k=3 majority vote.
 *
 *  Goal: 96%+ accuracy with sub-second wall time.
 * ================================================================ */

/* Partial dot product: first C coefficients in row-major order.
 * C must be a multiple of 16 for AVX2. */
static inline int32_t spectral_dot_partial(const int16_t *a,
                                            const int16_t *b, int C) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < C; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* Full dot product over all WHT_SIZE=1024 coefficients (flat layout). */
static inline int32_t spectral_dot_full_flat(const int16_t *a,
                                              const int16_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < WHT_SIZE; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

static void test_coarse_to_fine(void) {
    double t_total_start = now_sec();

    /* Brute force timing reference (use Test F timing if available, else estimate) */
    volatile int32_t sink = 0;
    double brute_time;
    {
        int sample = 50;
        double t0 = now_sec();
        for (int i = 0; i < sample; i++) {
            const int16_t *q = wht_test + (size_t)i * WHT_SIZE;
            for (int j = 0; j < TRAIN_N; j++)
                sink += spectral_dot_full_flat(q, wht_train + (size_t)j * WHT_SIZE);
        }
        double t1 = now_sec();
        brute_time = (t1 - t0) / sample * TEST_N;
        printf("  Brute force estimate: %.1f sec (from %d-sample benchmark)\n\n",
               brute_time, sample);
    }

    /*
     * Approach: use the vote-based cascade (91% filter accuracy) to select
     * top-K candidates, then refine with WHT full dot product + k=3 vote.
     * This combines the cascade's strong filtering with the spectral domain's
     * fast int16 dot products.
     */

    int K_vals[] = {50, 100, 200, 500};
    int n_K = 4;
    int knn_vals[] = {1, 3, 5};
    int n_knn = 3;

    /* Allocate vote buffer on heap */
    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    candidate_t *cands = malloc(TRAIN_N * sizeof(candidate_t));
    if (!votes || !cands) { fprintf(stderr, "ERROR: alloc\n"); exit(1); }

    /* --- G1: Vote → WHT dot (no cascade IG/multi-probe, just raw pixel vote) --- */
    printf("  --- G1: Pixel Vote → WHT k-NN (simple pipeline) ---\n\n");
    {
        int correct[4][3] = {{0}};
        double t0 = now_sec();

        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *qsig = px_test_sigs + (size_t)i * SIG2D_PAD;
            const int16_t *qwht = wht_test + (size_t)i * WHT_SIZE;

            /* Vote via pixel inverted index */
            memset(votes, 0, TRAIN_N * sizeof(uint16_t));
            for (int k = 0; k < N_BLOCKS2D; k++) {
                uint8_t bv = qsig[k];
                if (bv == 0) continue;
                uint32_t off = chan_idx_off[0][k][bv];
                uint16_t sz = chan_idx_sz[0][k][bv];
                const uint32_t *ids = chan_idx_pool[0] + off;
                for (uint16_t j = 0; j < sz; j++)
                    votes[ids[j]]++;
            }

            /* Select top-K by votes */
            /* Quick counting-sort selection */
            int nc = 0;
            {
                int hist[N_BLOCKS2D + 2];
                memset(hist, 0, sizeof(hist));
                for (int j = 0; j < TRAIN_N; j++)
                    hist[votes[j]]++;

                int max_K = K_vals[n_K - 1];
                int cum = 0, thr;
                for (thr = N_BLOCKS2D; thr >= 1; thr--) {
                    cum += hist[thr];
                    if (cum >= max_K) break;
                }
                if (thr < 1) thr = 1;

                for (int j = 0; j < TRAIN_N && nc < max_K; j++) {
                    if (votes[j] >= (uint16_t)thr) {
                        cands[nc].id = (uint32_t)j;
                        cands[nc].votes = votes[j];
                        nc++;
                    }
                }
            }

            /* WHT dot product refinement */
            for (int j = 0; j < nc; j++)
                cands[j].dot = spectral_dot_full_flat(qwht,
                    wht_train + (size_t)cands[j].id * WHT_SIZE);

            /* Sort by WHT dot */
            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

            /* Evaluate each K × k combination */
            for (int ki = 0; ki < n_K; ki++) {
                int K = K_vals[ki];
                int ek = nc < K ? nc : K;
                for (int kni = 0; kni < n_knn; kni++) {
                    int pred = knn_vote(cands, ek, knn_vals[kni]);
                    if (pred == test_labels[i]) correct[ki][kni]++;
                }
            }
        }
        double t1 = now_sec();

        printf("  %-6s | %-10s %-10s %-10s | %-8s\n",
               "K", "k=1", "k=3", "k=5", "WHT dots");
        printf("  -------+------------------------------------+---------\n");
        for (int ki = 0; ki < n_K; ki++) {
            printf("  %5d  |", K_vals[ki]);
            for (int kni = 0; kni < n_knn; kni++)
                printf(" %7.2f%%  ", 100.0 * correct[ki][kni] / TEST_N);
            printf("| %d\n", K_vals[ki] * TEST_N);
        }
        printf("\n  Time: %.2f sec (%.1fx vs brute)\n\n",
               t1 - t0, brute_time / (t1 - t0));
    }

    /* --- G2: Full 3-channel IG multi-probe vote → WHT k-NN --- */
    printf("  --- G2: 3-Chan IG Multi-Probe Vote → WHT k-NN ---\n\n");
    {
        int correct[4][3] = {{0}};
        double t0 = now_sec();

        for (int i = 0; i < TEST_N; i++) {
            const int16_t *qwht = wht_test + (size_t)i * WHT_SIZE;

            /* 3-channel IG-weighted multi-probe voting */
            memset(votes, 0, TRAIN_N * sizeof(uint16_t));
            const uint8_t *ch_sigs[N_CHANNELS] = {
                px_test_sigs + (size_t)i * SIG2D_PAD,
                hg_test_sigs + (size_t)i * SIG2D_PAD,
                vg_test_sigs + (size_t)i * SIG2D_PAD
            };
            uint8_t ch_bg[N_CHANNELS] = {0, BG_GRAD, BG_GRAD};

            for (int ch = 0; ch < N_CHANNELS; ch++) {
                const uint8_t *qsig = ch_sigs[ch];
                for (int k = 0; k < N_BLOCKS2D; k++) {
                    uint8_t bv = qsig[k];
                    if (bv == ch_bg[ch]) continue;
                    int w = ig_weights[ch][k];

                    /* Exact match */
                    uint32_t off = chan_idx_off[ch][k][bv];
                    uint16_t sz = chan_idx_sz[ch][k][bv];
                    const uint32_t *ids = chan_idx_pool[ch] + off;
                    for (uint16_t j = 0; j < sz; j++)
                        votes[ids[j]] += (uint16_t)w;

                    /* Hamming-1 neighbors at half weight */
                    int hw = w >> 1;
                    if (hw > 0) {
                        for (int ni = 0; ni < nbr_count[bv]; ni++) {
                            uint8_t nbv = nbr_table[bv][ni];
                            uint32_t noff = chan_idx_off[ch][k][nbv];
                            uint16_t nsz = chan_idx_sz[ch][k][nbv];
                            const uint32_t *nids = chan_idx_pool[ch] + noff;
                            for (uint16_t j = 0; j < nsz; j++)
                                votes[nids[j]] += (uint16_t)hw;
                        }
                    }
                }
            }

            /* Select top-K by votes */
            int nc = 0;
            {
                /* Find max vote for histogram sizing */
                uint16_t max_v = 0;
                for (int j = 0; j < TRAIN_N; j++)
                    if (votes[j] > max_v) max_v = votes[j];

                int max_K = K_vals[n_K - 1];
                /* Counting sort selection */
                int *hist = calloc((size_t)(max_v + 2), sizeof(int));
                for (int j = 0; j < TRAIN_N; j++)
                    hist[votes[j]]++;

                int cum = 0;
                int thr;
                for (thr = (int)max_v; thr >= 1; thr--) {
                    cum += hist[thr];
                    if (cum >= max_K) break;
                }
                if (thr < 1) thr = 1;
                free(hist);

                for (int j = 0; j < TRAIN_N && nc < max_K; j++) {
                    if (votes[j] >= (uint16_t)thr) {
                        cands[nc].id = (uint32_t)j;
                        cands[nc].votes = votes[j];
                        nc++;
                    }
                }
            }

            /* WHT dot product refinement */
            for (int j = 0; j < nc; j++)
                cands[j].dot = spectral_dot_full_flat(qwht,
                    wht_train + (size_t)cands[j].id * WHT_SIZE);

            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

            for (int ki = 0; ki < n_K; ki++) {
                int K = K_vals[ki];
                int ek = nc < K ? nc : K;
                for (int kni = 0; kni < n_knn; kni++) {
                    int pred = knn_vote(cands, ek, knn_vals[kni]);
                    if (pred == test_labels[i]) correct[ki][kni]++;
                }
            }

            if ((i + 1) % 2000 == 0)
                fprintf(stderr, "  G2: %d/%d\r", i + 1, TEST_N);
        }
        fprintf(stderr, "\n");
        double t1 = now_sec();

        printf("  %-6s | %-10s %-10s %-10s | %-8s\n",
               "K", "k=1", "k=3", "k=5", "WHT dots");
        printf("  -------+------------------------------------+---------\n");
        for (int ki = 0; ki < n_K; ki++) {
            printf("  %5d  |", K_vals[ki]);
            for (int kni = 0; kni < n_knn; kni++)
                printf(" %7.2f%%  ", 100.0 * correct[ki][kni] / TEST_N);
            printf("| %d\n", K_vals[ki] * TEST_N);
        }
        printf("\n  Time: %.2f sec (%.1fx vs brute)\n",
               t1 - t0, brute_time / (t1 - t0));
    }

    free(votes);
    free(cands);
    printf("  Total Test G: %.2f sec\n", now_sec() - t_total_start);
}

/* ================================================================
 *  main()
 * ================================================================ */

int main(int argc, char **argv) {
    seed_rng(42);
    double t0 = now_sec();

    /* Optional data directory from command line */
    if (argc > 1) {
        data_dir = argv[1];
        size_t len = strlen(data_dir);
        if (len > 0 && data_dir[len - 1] != '/') {
            char *buf = malloc(len + 2);
            memcpy(buf, data_dir, len);
            buf[len] = '/';
            buf[len + 1] = '\0';
            data_dir = buf;
        }
    }

    const char *dataset_name = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";

    printf("=== SSTT v2: Multi-Channel Geometric Classifier (%s) ===\n", dataset_name);
    puts("");
    puts("Improvements: 2D blocks, gradient observers, IG weights,");
    puts("              multi-probe, k-NN, error analysis.");
    puts("");

    /* Load data */
    printf("Loading %s...\n", dataset_name);
    load_data(data_dir);
    double t1 = now_sec();
    printf("  %d train + %d test images loaded (%.2f sec)\n\n", TRAIN_N, TEST_N, t1 - t0);

    /* Quantize to ternary */
    printf("Quantizing to ternary (AVX2)...\n");
    quantize_all();
    double t2 = now_sec();
    printf("  Done (%.2f sec)\n\n", t2 - t1);

    /* Compute gradients */
    printf("Computing gradient observers...\n");
    compute_all_gradients();
    double t3 = now_sec();
    printf("  Horizontal + vertical gradients for %d images (%.2f sec)\n\n",
           TRAIN_N + TEST_N, t3 - t2);

    /* Compute 2D block signatures for all channels */
    printf("Computing 2D block signatures (%d blocks/image × %d channels)...\n",
           N_BLOCKS2D, N_CHANNELS);
    compute_all_sigs();
    double t4 = now_sec();
    printf("  Done (%.2f sec)\n\n", t4 - t3);

    /* Build hot maps */
    printf("Building multi-channel hot maps...\n");
    build_all_hot_maps();
    double t5 = now_sec();
    printf("  3 × %zu KB = %zu KB total (%.4f sec)\n\n",
           sizeof(px_hot) / 1024,
           (sizeof(px_hot) + sizeof(hg_hot) + sizeof(vg_hot)) / 1024,
           t5 - t4);

    /* Compute IG weights */
    printf("Computing information gain weights...\n");
    compute_all_ig_weights();
    double t6 = now_sec();
    printf("  Done (%.2f sec)\n\n", t6 - t5);

    /* Build neighbor table */
    build_neighbor_table();

    /* Build inverted indices */
    printf("Building per-channel inverted indices...\n");
    build_all_indices();
    double t7 = now_sec();
    printf("  Done (%.2f sec)\n\n", t7 - t6);

    /* Compute WHT (Walsh-Hadamard) transforms */
    printf("Computing 2D Walsh-Hadamard transforms (32×32)...\n");
    compute_all_wht();
    double t8 = now_sec();
    printf("  %d train + %d test pixel WHTs (%.2f sec)\n",
           TRAIN_N, TEST_N, t8 - t7);
    printf("  Memory: train=%zu MB, test=%zu MB\n\n",
           (size_t)TRAIN_N * WHT_SIZE * sizeof(int16_t) / (1024 * 1024),
           (size_t)TEST_N * WHT_SIZE * sizeof(int16_t) / (1024 * 1024));

    /* Build WHT prototypes (true WHT preserves dot products — no centering needed) */
    printf("Building WHT prototypes...\n");
    build_wht_prototypes(wht_train, TRAIN_N, wht_proto);
    build_wht_proto_from_ternary(hgrad_train, TRAIN_N, wht_hg_proto);
    build_wht_proto_from_ternary(vgrad_train, TRAIN_N, wht_vg_proto);
    double t9 = now_sec();
    printf("  Pixel + H-Grad + V-Grad (%.2f sec)\n\n", t9 - t8);

    /* ============================================================ */

    /* Test A: 2D-Block Pixel Hot Map */
    puts("--- Test A: 2D-Block Pixel Hot Map ---");
    test_hotmap_2d();
    puts("");

    /* Test B: 3-Channel Hot Map */
    puts("--- Test B: 3-Channel Hot Map (Pixel + Gradients) ---");
    test_multichan_hotmap();
    puts("");

    /* Test C: Information-Gain Weighted Voting */
    puts("--- Test C: Information-Gain Weighted Voting ---");
    test_ig_voting();
    puts("");

    /* Test D: Multi-Probe Voting */
    puts("--- Test D: Multi-Probe Voting ---");
    puts("  Exact match at full weight + Hamming-1 neighbors at half weight.\n");
    test_multiprobe();
    puts("");

    /* Test E: Full Cascade */
    puts("--- Test E: Full Cascade (Multi-Probe + IG + 3-Chan + k-NN) ---");
    puts("  Weighted multi-probe multi-channel vote → top-K → dot → k-NN\n");
    double best_acc = 0;
    uint8_t *best_preds = calloc(TEST_N, 1);
    test_full_cascade(&best_acc, best_preds);

    /* Test F: WHT Spectral Classifier */
    puts("\n--- Test F: Walsh-Hadamard Spectral Classifier ---");
    puts("  Orthogonal: H*H^T=N*I. WHT k-NN = pixel k-NN.\n");
    test_wht_spectral();

    /* Test G: Spectral Coarse-to-Fine k-NN */
    puts("\n--- Test G: Spectral Coarse-to-Fine k-NN ---");
    puts("  Stage 1: prototype filter (top 3 classes)");
    puts("  Stage 2: coarse spectral pruning (partial WHT dot)");
    puts("  Stage 3: full dot refinement");
    puts("  Stage 4: k=3 majority vote\n");
    test_coarse_to_fine();

    /* Error analysis on best config */
    puts("\n--- Error Analysis (best cascade config) ---");
    error_analysis(best_preds);
    free(best_preds);

    /* Summary */
    puts("\n=== SUMMARY: v1 vs v2 ===");
    puts("  v1 baselines (from sstt_geom.c):");
    puts("    Hot map (261 1D blocks):     71.12%");
    puts("    Vote-only:                   84.43%");
    puts("    TST K2=50:                   94.67%");
    puts("    TST K2=200:                  95.47%");
    puts("    Brute 1-NN:                  95.87%");
    printf("\n  v2 best cascade: %.2f%%\n", best_acc);
    puts("    WHT spectral:                see Test F above");

    double total_time = now_sec() - t0;
    printf("\nTotal runtime: %.2f seconds.\n", total_time);

    /* Cleanup */
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(px_train_sigs); free(px_test_sigs);
    free(hg_train_sigs); free(hg_test_sigs);
    free(vg_train_sigs); free(vg_test_sigs);
    for (int ch = 0; ch < N_CHANNELS; ch++)
        free(chan_idx_pool[ch]);
    free(wht_train); free(wht_test);

    return 0;
}
