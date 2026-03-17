/*
 * sstt_hybrid.c — Confidence-Gated Hybrid Classifier
 *
 * Confidence-gated inference:
 *   Phase 1-3: Fused IG multi-probe hot map (fast, pure addressing)
 *   Phase 4:   Argmax with confidence margin
 *   Loop-back: Low confidence → cascade refinement (dot products)
 *
 * Build: make sstt_hybrid  (after: make mnist)
 *
 * Composites tested:
 *   A. Baseline 3-chan hot map (73%)
 *   B. Pre-baked IG hot map (expected ~83-86%)
 *   C. IG + multi-probe hot map (expected ~87-91%)
 *   D. Confidence-gated hybrid: C for easy images, cascade for hard ones
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
#define PIXELS      784
#define PADDED      800
#define N_CLASSES   10
#define CLS_PAD     16

/* 2D blocks */
#define BLK_W       3
#define BLKS_PER_ROW 9
#define N_BLOCKS    252
#define N_BVALS     27
#define SIG_PAD     256

#define BG_PIXEL    0
#define BG_GRAD     13

/* IG */
#define IG_SCALE    16

/* Multi-probe */
#define MAX_NEIGHBORS 6

/* Cascade */
#define MAX_K       500

static const char *data_dir = "data/";

/* ---------- Timing ---------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Data ---------- */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t *tern_train, *tern_test;
static int8_t *hgrad_train, *hgrad_test;
static int8_t *vgrad_train, *vgrad_test;

/* Signatures */
static uint8_t *px_train_sigs, *px_test_sigs;
static uint8_t *hg_train_sigs, *hg_test_sigs;
static uint8_t *vg_train_sigs, *vg_test_sigs;

/* Hot maps: [N_BLOCKS][N_BVALS][CLS_PAD] */
static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* IG-weighted hot maps (pre-baked) */
static uint32_t px_ig_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_ig_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_ig_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* IG weights per block per channel */
static uint16_t ig_weights[3][N_BLOCKS];

/* Neighbor table for multi-probe */
static uint8_t nbr_table[N_BVALS][MAX_NEIGHBORS];
static uint8_t nbr_count[N_BVALS];

/* Inverted index for cascade fallback */
static uint32_t *idx_pool[3];
static uint32_t idx_off[3][N_BLOCKS][N_BVALS];
static uint16_t idx_sz[3][N_BLOCKS][N_BVALS];

/* ================================================================
 *  Standard infrastructure (load, quantize, gradient, sigs)
 * ================================================================ */

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
        rows = __builtin_bswap32(rows); cols = __builtin_bswap32(cols);
        if (rows_out) *rows_out = rows; if (cols_out) *cols_out = cols;
        item_size = (size_t)rows * cols;
    } else { if (rows_out) *rows_out = 0; if (cols_out) *cols_out = 0; }
    size_t total = (size_t)n * item_size;
    uint8_t *data = malloc(total);
    if (!data || fread(data, 1, total, f) != total) { fclose(f); exit(1); }
    fclose(f);
    return data;
}

static void load_data(const char *dir) {
    uint32_t n, r, c; char path[256];
    snprintf(path, sizeof(path), "%strain-images-idx3-ubyte", dir);
    raw_train_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%strain-labels-idx1-ubyte", dir);
    train_labels = load_idx(path, &n, NULL, NULL);
    snprintf(path, sizeof(path), "%st10k-images-idx3-ubyte", dir);
    raw_test_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%st10k-labels-idx1-ubyte", dir);
    test_labels = load_idx(path, &n, NULL, NULL);
}

static void quantize_one(const uint8_t *src, int8_t *dst) {
    for (int i = 0; i < PIXELS; i++) {
        if (src[i] > 170) dst[i] = 1;
        else if (src[i] < 85) dst[i] = -1;
        else dst[i] = 0;
    }
    memset(dst + PIXELS, 0, PADDED - PIXELS);
}

static inline int8_t clamp_trit(int v) {
    return v > 0 ? 1 : v < 0 ? -1 : 0;
}

static void compute_gradients_one(const int8_t *t, int8_t *hg, int8_t *vg) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++)
            hg[y*IMG_W+x] = clamp_trit(t[y*IMG_W+x+1] - t[y*IMG_W+x]);
        hg[y*IMG_W+IMG_W-1] = 0;
    }
    memset(hg+PIXELS, 0, PADDED-PIXELS);
    for (int y = 0; y < IMG_H-1; y++)
        for (int x = 0; x < IMG_W; x++)
            vg[y*IMG_W+x] = clamp_trit(t[(y+1)*IMG_W+x] - t[y*IMG_W+x]);
    memset(vg+(IMG_H-1)*IMG_W, 0, IMG_W);
    memset(vg+PIXELS, 0, PADDED-PIXELS);
}

static void init_all(void) {
    tern_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    hgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    for (int i = 0; i < TRAIN_N; i++) {
        quantize_one(raw_train_img + (size_t)i*PIXELS, tern_train + (size_t)i*PADDED);
        compute_gradients_one(tern_train + (size_t)i*PADDED,
                              hgrad_train + (size_t)i*PADDED,
                              vgrad_train + (size_t)i*PADDED);
    }
    for (int i = 0; i < TEST_N; i++) {
        quantize_one(raw_test_img + (size_t)i*PIXELS, tern_test + (size_t)i*PADDED);
        compute_gradients_one(tern_test + (size_t)i*PADDED,
                              hgrad_test + (size_t)i*PADDED,
                              vgrad_test + (size_t)i*PADDED);
    }
}

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0+1)*9 + (t1+1)*3 + (t2+1));
}

static void compute_2d_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int base = y * IMG_W + s * BLK_W;
                sig[y * BLKS_PER_ROW + s] =
                    block_encode(img[base], img[base+1], img[base+2]);
            }
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

static void compute_all_sigs(void) {
    px_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    px_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    hg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    hg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    vg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    vg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    compute_2d_sigs(tern_train,  px_train_sigs, TRAIN_N);
    compute_2d_sigs(tern_test,   px_test_sigs,  TEST_N);
    compute_2d_sigs(hgrad_train, hg_train_sigs, TRAIN_N);
    compute_2d_sigs(hgrad_test,  hg_test_sigs,  TEST_N);
    compute_2d_sigs(vgrad_train, vg_train_sigs, TRAIN_N);
    compute_2d_sigs(vgrad_test,  vg_test_sigs,  TEST_N);
}

/* ================================================================
 *  Hot Map Building
 * ================================================================ */

static void build_hot_map(const uint8_t *sigs, uint32_t hot[][N_BVALS][CLS_PAD]) {
    memset(hot, 0, sizeof(uint32_t) * N_BLOCKS * N_BVALS * CLS_PAD);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            hot[k][sig[k]][lbl]++;
    }
}

/* ================================================================
 *  Information Gain Weights
 * ================================================================ */

static void compute_ig(const uint8_t *sigs, int ch) {
    /* Class prior entropy */
    int class_count[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) class_count[train_labels[i]]++;
    double H_prior = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)class_count[c] / TRAIN_N;
        if (p > 0) H_prior -= p * log2(p);
    }

    double raw_ig[N_BLOCKS];
    double max_ig = 0;

    for (int k = 0; k < N_BLOCKS; k++) {
        /* Count block value frequencies and per-class breakdowns */
        int bv_count[N_BVALS] = {0};
        int bv_class[N_BVALS][N_CLASSES];
        memset(bv_class, 0, sizeof(bv_class));
        for (int i = 0; i < TRAIN_N; i++) {
            uint8_t bv = sigs[i * SIG_PAD + k];
            bv_count[bv]++;
            bv_class[bv][train_labels[i]]++;
        }
        /* Conditional entropy H(Y|X=bv) weighted by P(X=bv) */
        double H_cond = 0;
        for (int bv = 0; bv < N_BVALS; bv++) {
            if (bv_count[bv] == 0) continue;
            double p_bv = (double)bv_count[bv] / TRAIN_N;
            double h = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double p = (double)bv_class[bv][c] / bv_count[bv];
                if (p > 0) h -= p * log2(p);
            }
            H_cond += p_bv * h;
        }
        raw_ig[k] = H_prior - H_cond;
        if (raw_ig[k] > max_ig) max_ig = raw_ig[k];
    }

    /* Debug: print IG stats */
    if (ch == 0) {
        printf("  IG debug ch=%d: H_prior=%.4f max_ig=%.6f ig[0]=%.6f ig[126]=%.6f ig[200]=%.6f\n",
               ch, H_prior, max_ig, raw_ig[0], raw_ig[126], raw_ig[200]);
    }

    /* Quantize to [1, IG_SCALE] */
    for (int k = 0; k < N_BLOCKS; k++) {
        if (max_ig > 0)
            ig_weights[ch][k] = (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5);
        else
            ig_weights[ch][k] = 1;
        if (ig_weights[ch][k] == 0) ig_weights[ch][k] = 1;
    }
}

/* ================================================================
 *  Pre-baked IG Hot Map: hot[k][bv][c] *= ig_weight[k]
 * ================================================================ */

static void prebake_ig(const uint32_t src[][N_BVALS][CLS_PAD],
                        uint32_t dst[][N_BVALS][CLS_PAD],
                        int ch) {
    for (int k = 0; k < N_BLOCKS; k++) {
        uint16_t w = ig_weights[ch][k];
        for (int bv = 0; bv < N_BVALS; bv++)
            for (int c = 0; c < CLS_PAD; c++)
                dst[k][bv][c] = src[k][bv][c] * w;
    }
}

/* ================================================================
 *  Neighbor Table (Hamming-1 in block-value space)
 * ================================================================ */

static void build_neighbor_table(void) {
    int trits[3] = {-1, 0, 1};
    for (int v = 0; v < N_BVALS; v++) {
        int t0 = (v / 9) - 1;
        int t1 = ((v / 3) % 3) - 1;
        int t2 = (v % 3) - 1;
        int orig[3] = {t0, t1, t2};
        int nc = 0;
        for (int pos = 0; pos < 3; pos++)
            for (int alt = 0; alt < 3; alt++) {
                if (trits[alt] == orig[pos]) continue;
                int mod[3] = {orig[0], orig[1], orig[2]};
                mod[pos] = trits[alt];
                nbr_table[v][nc++] = block_encode((int8_t)mod[0],
                    (int8_t)mod[1], (int8_t)mod[2]);
            }
        nbr_count[v] = (uint8_t)nc;
    }
}

/* ================================================================
 *  Inverted Index (for cascade fallback)
 * ================================================================ */

static void build_inverted_index(const uint8_t *sigs, int ch) {
    memset(idx_sz[ch], 0, sizeof(idx_sz[ch]));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            idx_sz[ch][k][sig[k]]++;
    }
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < N_BVALS; v++) {
            idx_off[ch][k][v] = total;
            total += idx_sz[ch][k][v];
        }
    idx_pool[ch] = malloc((size_t)total * sizeof(uint32_t));
    uint32_t wpos[N_BLOCKS][N_BVALS];
    memcpy(wpos, idx_off[ch], sizeof(idx_off[ch]));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            idx_pool[ch][wpos[k][sig[k]]++] = (uint32_t)i;
    }
}

/* ================================================================
 *  Ternary Dot Product (for cascade refinement)
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
 *  Classifiers
 * ================================================================ */

/* A: Baseline 3-channel hot map */
static int classify_baseline(int img_idx) {
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
    const uint8_t *ps = px_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *hs = hg_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *vs = vg_test_sigs + (size_t)img_idx * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv;
        bv = ps[k]; if (bv != BG_PIXEL) {
            lo = _mm256_add_epi32(lo, _mm256_load_si256((const __m256i *)px_hot[k][bv]));
            hi = _mm256_add_epi32(hi, _mm256_load_si256((const __m256i *)px_hot[k][bv]+1));
        }
        bv = hs[k]; if (bv != BG_GRAD) {
            lo = _mm256_add_epi32(lo, _mm256_load_si256((const __m256i *)hg_hot[k][bv]));
            hi = _mm256_add_epi32(hi, _mm256_load_si256((const __m256i *)hg_hot[k][bv]+1));
        }
        bv = vs[k]; if (bv != BG_GRAD) {
            lo = _mm256_add_epi32(lo, _mm256_load_si256((const __m256i *)vg_hot[k][bv]));
            hi = _mm256_add_epi32(hi, _mm256_load_si256((const __m256i *)vg_hot[k][bv]+1));
        }
    }
    uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)cv, lo);
    _mm256_store_si256((__m256i *)(cv+8), hi);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[best]) best = c;
    return best;
}

/* B: Pre-baked IG hot map */
static int classify_ig(int img_idx) {
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
    const uint8_t *ps = px_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *hs = hg_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *vs = vg_test_sigs + (size_t)img_idx * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv;
        bv = ps[k]; if (bv != BG_PIXEL) {
            lo = _mm256_add_epi32(lo, _mm256_load_si256((const __m256i *)px_ig_hot[k][bv]));
            hi = _mm256_add_epi32(hi, _mm256_load_si256((const __m256i *)px_ig_hot[k][bv]+1));
        }
        bv = hs[k]; if (bv != BG_GRAD) {
            lo = _mm256_add_epi32(lo, _mm256_load_si256((const __m256i *)hg_ig_hot[k][bv]));
            hi = _mm256_add_epi32(hi, _mm256_load_si256((const __m256i *)hg_ig_hot[k][bv]+1));
        }
        bv = vs[k]; if (bv != BG_GRAD) {
            lo = _mm256_add_epi32(lo, _mm256_load_si256((const __m256i *)vg_ig_hot[k][bv]));
            hi = _mm256_add_epi32(hi, _mm256_load_si256((const __m256i *)vg_ig_hot[k][bv]+1));
        }
    }
    uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)cv, lo);
    _mm256_store_si256((__m256i *)(cv+8), hi);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[best]) best = c;
    return best;
}

/* C: IG exact + raw neighbor multi-probe hot map.
 * Exact match: IG-weighted (strong, discriminative).
 * Neighbors: raw (unweighted) at 1/4 weight (soft matching without IG amplification). */
static int classify_ig_multiprobe(int img_idx, uint32_t *margin_out) {
    __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
    const uint8_t *sigs_arr[3] = {
        px_test_sigs + (size_t)img_idx * SIG_PAD,
        hg_test_sigs + (size_t)img_idx * SIG_PAD,
        vg_test_sigs + (size_t)img_idx * SIG_PAD
    };
    uint32_t (*ig_hots[3])[N_BVALS][CLS_PAD] = {px_ig_hot, hg_ig_hot, vg_ig_hot};
    uint32_t (*raw_hots[3])[N_BVALS][CLS_PAD] = {px_hot, hg_hot, vg_hot};
    uint8_t bgs[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};

    for (int ch = 0; ch < 3; ch++) {
        const uint8_t *sig = sigs_arr[ch];
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sig[k];
            if (bv == bgs[ch]) continue;

            /* Exact match: IG-weighted */
            lo = _mm256_add_epi32(lo, _mm256_load_si256((const __m256i *)ig_hots[ch][k][bv]));
            hi = _mm256_add_epi32(hi, _mm256_load_si256((const __m256i *)ig_hots[ch][k][bv]+1));

            /* Hamming-1 neighbors: raw at 1/4 weight */
            for (int ni = 0; ni < nbr_count[bv]; ni++) {
                uint8_t nbv = nbr_table[bv][ni];
                __m256i nlo = _mm256_srli_epi32(
                    _mm256_load_si256((const __m256i *)raw_hots[ch][k][nbv]), 2);
                __m256i nhi = _mm256_srli_epi32(
                    _mm256_load_si256((const __m256i *)raw_hots[ch][k][nbv]+1), 2);
                lo = _mm256_add_epi32(lo, nlo);
                hi = _mm256_add_epi32(hi, nhi);
            }
        }
    }

    uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)cv, lo);
    _mm256_store_si256((__m256i *)(cv+8), hi);

    /* Argmax + margin */
    int best = 0;
    uint32_t best_v = cv[0], second_v = 0;
    for (int c = 1; c < N_CLASSES; c++) {
        if (cv[c] > best_v) {
            second_v = best_v;
            best_v = cv[c];
            best = c;
        } else if (cv[c] > second_v) {
            second_v = cv[c];
        }
    }
    if (margin_out) *margin_out = best_v - second_v;
    return best;
}

/* D: Cascade refinement for low-confidence images */
typedef struct { uint32_t id; uint16_t votes; int32_t dot; } candidate_t;

static int cmp_dots_desc(const void *a, const void *b) {
    int32_t da = ((const candidate_t *)a)->dot;
    int32_t db = ((const candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

static int classify_cascade(int img_idx) {
    /* 3-channel IG-weighted multi-probe vote */
    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    const uint8_t *sigs[3] = {
        px_test_sigs + (size_t)img_idx * SIG_PAD,
        hg_test_sigs + (size_t)img_idx * SIG_PAD,
        vg_test_sigs + (size_t)img_idx * SIG_PAD
    };
    uint8_t bgs[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};

    for (int ch = 0; ch < 3; ch++) {
        const uint8_t *sig = sigs[ch];
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sig[k];
            if (bv == bgs[ch]) continue;
            uint16_t w = ig_weights[ch][k];

            /* Exact match */
            uint32_t off = idx_off[ch][k][bv];
            uint16_t sz = idx_sz[ch][k][bv];
            const uint32_t *ids = idx_pool[ch] + off;
            for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w;

            /* Multi-probe at half weight */
            uint16_t hw = w >> 1;
            if (hw > 0)
                for (int ni = 0; ni < nbr_count[bv]; ni++) {
                    uint8_t nbv = nbr_table[bv][ni];
                    uint32_t noff = idx_off[ch][k][nbv];
                    uint16_t nsz = idx_sz[ch][k][nbv];
                    const uint32_t *nids = idx_pool[ch] + noff;
                    for (uint16_t j = 0; j < nsz; j++) nids[j] < TRAIN_N ? (votes[nids[j]] += hw) : 0;
                }
        }
    }

    /* Top-K by votes */
    candidate_t cands[MAX_K];
    int nc = 0;
    uint16_t max_v = 0;
    for (int j = 0; j < TRAIN_N; j++)
        if (votes[j] > max_v) max_v = votes[j];

    int *hist = calloc((size_t)(max_v + 2), sizeof(int));
    for (int j = 0; j < TRAIN_N; j++) hist[votes[j]]++;
    int cum = 0, thr;
    for (thr = (int)max_v; thr >= 1; thr--) {
        cum += hist[thr];
        if (cum >= MAX_K) break;
    }
    if (thr < 1) thr = 1;
    free(hist);

    for (int j = 0; j < TRAIN_N && nc < MAX_K; j++)
        if (votes[j] >= (uint16_t)thr) {
            cands[nc].id = (uint32_t)j;
            cands[nc].votes = votes[j];
            nc++;
        }
    free(votes);

    /* Dot product refinement */
    const int8_t *query = tern_test + (size_t)img_idx * PADDED;
    for (int j = 0; j < nc; j++)
        cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);

    qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

    /* k=3 majority vote */
    int k = nc < 3 ? nc : 3;
    int class_votes[N_CLASSES] = {0};
    for (int j = 0; j < k; j++)
        class_votes[train_labels[cands[j].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (class_votes[c] > class_votes[best]) best = c;
    return best;
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(int argc, char **argv) {
    double t0 = now_sec();

    if (argc > 1) {
        data_dir = argv[1];
        size_t len = strlen(data_dir);
        if (len > 0 && data_dir[len-1] != '/') {
            char *buf = malloc(len + 2);
            memcpy(buf, data_dir, len);
            buf[len] = '/'; buf[len+1] = '\0';
            data_dir = buf;
        }
    }

    const char *dataset = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Hybrid Classifier (%s) ===\n\n", dataset);

    printf("Loading and preprocessing...\n");
    load_data(data_dir);
    init_all();
    compute_all_sigs();
    build_neighbor_table();
    double t1 = now_sec();
    printf("  Done (%.2f sec)\n\n", t1 - t0);

    /* Build hot maps */
    printf("Building hot maps...\n");
    build_hot_map(px_train_sigs, px_hot);
    build_hot_map(hg_train_sigs, hg_hot);
    build_hot_map(vg_train_sigs, vg_hot);

    /* Compute IG weights */
    compute_ig(px_train_sigs, 0);
    compute_ig(hg_train_sigs, 1);
    compute_ig(vg_train_sigs, 2);

    printf("  IG weights: px=[%d,%d] hg=[%d,%d] vg=[%d,%d]\n",
           ig_weights[0][0], ig_weights[0][N_BLOCKS/2],
           ig_weights[1][0], ig_weights[1][N_BLOCKS/2],
           ig_weights[2][0], ig_weights[2][N_BLOCKS/2]);

    /* Pre-bake IG into hot maps */
    prebake_ig(px_hot, px_ig_hot, 0);
    prebake_ig(hg_hot, hg_ig_hot, 1);
    prebake_ig(vg_hot, vg_ig_hot, 2);

    /* Build inverted indices for cascade fallback */
    build_inverted_index(px_train_sigs, 0);
    build_inverted_index(hg_train_sigs, 1);
    build_inverted_index(vg_train_sigs, 2);
    double t2 = now_sec();
    printf("  All structures built (%.2f sec)\n\n", t2 - t1);

    /* ============================================================
     * Test A: Baseline 3-channel hot map
     * ============================================================ */
    printf("--- A: Baseline 3-Channel Hot Map ---\n");
    int a_correct = 0;
    double ta0 = now_sec();
    for (int i = 0; i < TEST_N; i++)
        if (classify_baseline(i) == test_labels[i]) a_correct++;
    double ta1 = now_sec();
    printf("  Accuracy: %.2f%% (%.0f ns/query)\n\n",
           100.0 * a_correct / TEST_N, (ta1-ta0)*1e9/TEST_N);

    /* ============================================================
     * Test B: Pre-baked IG hot map
     * ============================================================ */
    printf("--- B: Pre-Baked IG Hot Map ---\n");
    int b_correct = 0;
    double tb0 = now_sec();
    for (int i = 0; i < TEST_N; i++)
        if (classify_ig(i) == test_labels[i]) b_correct++;
    double tb1 = now_sec();
    printf("  Accuracy: %.2f%% (%+.2f pp vs baseline, %.0f ns/query)\n\n",
           100.0 * b_correct / TEST_N,
           100.0 * (b_correct - a_correct) / TEST_N,
           (tb1-tb0)*1e9/TEST_N);

    /* ============================================================
     * Test C: IG + Multi-Probe Hot Map (with confidence margin)
     * ============================================================ */
    printf("--- C: IG + Multi-Probe Hot Map ---\n");
    int c_correct = 0;
    uint32_t *margins = malloc(TEST_N * sizeof(uint32_t));
    int *c_preds = malloc(TEST_N * sizeof(int));
    double tc0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        c_preds[i] = classify_ig_multiprobe(i, &margins[i]);
        if (c_preds[i] == test_labels[i]) c_correct++;
    }
    double tc1 = now_sec();
    printf("  Accuracy: %.2f%% (%+.2f pp vs baseline, %.0f ns/query)\n\n",
           100.0 * c_correct / TEST_N,
           100.0 * (c_correct - a_correct) / TEST_N,
           (tc1-tc0)*1e9/TEST_N);

    /* ============================================================
     * Test D: Confidence-Gated Hybrid
     * ============================================================ */
    printf("--- D: Confidence-Gated Hybrid ---\n");
    printf("  Hot map (IG+MP) for high-confidence, cascade for low.\n\n");

    /* Sweep confidence thresholds */
    /* Sort margins to find percentiles */
    uint32_t *sorted_margins = malloc(TEST_N * sizeof(uint32_t));
    memcpy(sorted_margins, margins, TEST_N * sizeof(uint32_t));
    /* Simple insertion sort on margins for percentile computation */
    for (int i = 1; i < TEST_N; i++) {
        uint32_t key = sorted_margins[i];
        int j = i - 1;
        while (j >= 0 && sorted_margins[j] > key) {
            sorted_margins[j+1] = sorted_margins[j];
            j--;
        }
        sorted_margins[j+1] = key;
    }

    double pcts[] = {0.05, 0.10, 0.15, 0.20, 0.30};
    int n_pcts = 5;

    printf("  %-8s | %-8s | %-10s | %-10s | %-12s | %-8s\n",
           "Fallback", "Thresh", "Hot Acc", "Cascade", "Hybrid Acc", "Time");
    printf("  ---------+----------+------------+------------+--------------+---------\n");

    for (int pi = 0; pi < n_pcts; pi++) {
        double pct = pcts[pi];
        /* Threshold: images below this margin get cascade refinement */
        int thr_idx = (int)(TEST_N * pct);
        uint32_t margin_thr = sorted_margins[thr_idx];

        int hot_correct = 0, cascade_correct = 0, total_cascade = 0;
        double td0 = now_sec();

        for (int i = 0; i < TEST_N; i++) {
            if (margins[i] <= margin_thr) {
                /* Low confidence → cascade */
                total_cascade++;
                if (classify_cascade(i) == test_labels[i]) cascade_correct++;
            } else {
                /* High confidence → use hot map prediction */
                if (c_preds[i] == test_labels[i]) hot_correct++;
            }
        }

        double td1 = now_sec();
        int total_correct = hot_correct + cascade_correct;
        double acc = 100.0 * total_correct / TEST_N;

        printf("  %5.0f%%   | %6u   | %7.2f%%   | %4d/%-4d  | %8.2f%%    | %.2fs\n",
               pct * 100, margin_thr,
               100.0 * hot_correct / (TEST_N - total_cascade),
               cascade_correct, total_cascade,
               acc, td1 - td0);
    }

    /* Pure cascade baseline */
    printf("\n  --- Pure Cascade (all images) ---\n");
    int d_all_correct = 0;
    double te0 = now_sec();
    for (int i = 0; i < TEST_N; i++)
        if (classify_cascade(i) == test_labels[i]) d_all_correct++;
    double te1 = now_sec();
    printf("  Accuracy: %.2f%% (%.2f sec)\n", 100.0 * d_all_correct / TEST_N, te1 - te0);

    /* Summary */
    printf("\n=== SUMMARY ===\n");
    printf("  A. Baseline hot map:      %6.2f%%\n", 100.0 * a_correct / TEST_N);
    printf("  B. Pre-baked IG:          %6.2f%% (%+.2f pp, free)\n",
           100.0 * b_correct / TEST_N, 100.0 * (b_correct - a_correct) / TEST_N);
    printf("  C. IG + multi-probe:      %6.2f%% (%+.2f pp)\n",
           100.0 * c_correct / TEST_N, 100.0 * (c_correct - a_correct) / TEST_N);
    printf("  D. Hybrid (best):         see table above\n");
    printf("  E. Pure cascade:          %6.2f%%\n", 100.0 * d_all_correct / TEST_N);
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    /* Cleanup */
    free(margins); free(c_preds); free(sorted_margins);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(px_train_sigs); free(px_test_sigs);
    free(hg_train_sigs); free(hg_test_sigs);
    free(vg_train_sigs); free(vg_test_sigs);
    for (int ch = 0; ch < 3; ch++) free(idx_pool[ch]);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);

    return 0;
}
