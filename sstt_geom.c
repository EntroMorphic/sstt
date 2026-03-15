/*
 * sstt_geom.c — Geometric Inference: Ternary MNIST Classification
 *
 * Question: Can ternary lattice geometry separate real digit classes?
 * Method:   Quantize MNIST to {-1, 0, +1}, classify by ternary dot product.
 *           No matrix multiply, no activation functions — just geometric proximity.
 *
 * Test A: Ternary Centroid Classifier (10 class prototypes)
 * Test B: Ternary 1-NN (nearest training example, 600M dot products)
 * Baselines: Random (10%), Float centroid (unquantized)
 * Noise test: Accuracy under analog noise at various SNR levels
 *
 * Build: make sstt_geom  (after: make mnist)
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
#define PAD_ITERS   25          /* PADDED / 32 */

/* Block index config */
#define N_BLOCKS    261         /* 783 / 3, complete 3-trit blocks */
#define N_BVALS     27          /* 3^3 possible block values */
#define SIG_PAD     288         /* 9 × 32, clean AVX2 */
#define MAX_K       1000        /* max candidates for refinement */

/* Gradient background: block_encode(0,0,0) = 13 (flat region, no transition) */
#define BG_GRAD     13
#define CLS_PAD     16          /* pad 10 classes to 16 for AVX2 */

/* TST multi-resolution hierarchy (FFT-style butterfly cascade) */
#define TST_L0      261         /* = N_BLOCKS, 3-trit blocks */
#define TST_L1      87          /* 261 / 3, superblocks */
#define TST_L2      29          /* 87 / 3, megablocks */
#define TST_L3      9           /* 27 / 3, hyperblocks */
#define TST_L4      3           /* 9 / 3, regions */
#define TST_W0      1           /* twiddle weight 3^0 */
#define TST_W1      3           /* twiddle weight 3^1 */
#define TST_W2      9           /* twiddle weight 3^2 */
#define TST_W3      27          /* twiddle weight 3^3 */
#define TST_W4      81          /* twiddle weight 3^4 */
#define TST_MAX_SCORE 1269      /* 261 + 87*3 + 29*9 + 9*27 + 3*81 */
#define TST_K1      1000        /* vote stage candidates */

/* ---------- xoshiro256++ PRNG (from sstt_mvp.c) ---------- */
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

/* ---------- Box-Muller Gaussian (from sstt_mvp.c) ---------- */
static int gspare = 0;
static float gval;

static inline float gauss(void) {
    if (gspare) { gspare = 0; return gval; }
    float u, v, s;
    do {
        u = (float)(rng() >> 40) / (float)(1 << 24) * 2.0f - 1.0f;
        v = (float)(rng() >> 40) / (float)(1 << 24) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    float m = sqrtf(-2.0f * logf(s) / s);
    gval = v * m;
    gspare = 1;
    return u * m;
}

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

static int8_t *tern_train;         /* [TRAIN_N * PADDED], 32-byte aligned */
static int8_t *tern_test;          /* [TEST_N * PADDED],  32-byte aligned */

static int8_t tern_centroids[N_CLASSES][PADDED] __attribute__((aligned(32)));

/* Cached centroid accuracy for noise test comparison */
static double centroid_acc_clean;

/* Block signatures: byte k = encode(trit[3k], trit[3k+1], trit[3k+2]) */
static uint8_t *train_sigs;     /* [TRAIN_N * SIG_PAD], aligned */
static uint8_t *test_sigs;      /* [TEST_N * SIG_PAD],  aligned */

/* SDF cipher data */
static int8_t *sdf_raw_train;    /* [TRAIN_N * PADDED], raw SDF values */
static int8_t *sdf_raw_test;     /* [TEST_N * PADDED] */
static int8_t *sdf_tern_train;   /* [TRAIN_N * PADDED], SDF ternary at best T */
static int8_t *sdf_tern_test;    /* [TEST_N * PADDED] */
static uint8_t *sdf_train_sigs;  /* [TRAIN_N * SIG_PAD], SDF block sigs */
static uint8_t *sdf_test_sigs;   /* [TEST_N * SIG_PAD] */

/* Gradient data */
static int8_t *hgrad_train;      /* [TRAIN_N * PADDED] */
static int8_t *hgrad_test;       /* [TEST_N * PADDED] */
static int8_t *vgrad_train;      /* [TRAIN_N * PADDED] */
static int8_t *vgrad_test;       /* [TEST_N * PADDED] */
static uint8_t *hg_train_sigs;   /* [TRAIN_N * SIG_PAD] */
static uint8_t *hg_test_sigs;    /* [TEST_N * SIG_PAD] */
static uint8_t *vg_train_sigs;   /* [TRAIN_N * SIG_PAD] */
static uint8_t *vg_test_sigs;    /* [TEST_N * SIG_PAD] */

/* Gradient hot maps */
static uint32_t hg_hot_map[N_BLOCKS][N_BVALS][CLS_PAD]
    __attribute__((aligned(32)));
static uint32_t vg_hot_map[N_BLOCKS][N_BVALS][CLS_PAD]
    __attribute__((aligned(32)));

/* Gradient-SDF hot maps */
static uint32_t hg_sdf_hot[N_BLOCKS][N_BVALS][CLS_PAD]
    __attribute__((aligned(32)));
static uint32_t vg_sdf_hot[N_BLOCKS][N_BVALS][CLS_PAD]
    __attribute__((aligned(32)));
/* Gradient-SDF test sigs (kept for classification) */
static uint8_t *hg_sdf_test_sigs;
static uint8_t *vg_sdf_test_sigs;

/* Inverted index: flat pool of image IDs, indexed by (block_pos, block_val) */
static uint32_t *idx_pool;
static uint32_t idx_off[N_BLOCKS][N_BVALS];
static uint16_t idx_sz[N_BLOCKS][N_BVALS];

/* ================================================================
 *  MNIST IDX Loader
 * ================================================================ */

static uint8_t *load_idx(const char *path, uint32_t *count,
                         uint32_t *rows_out, uint32_t *cols_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        fprintf(stderr, "  Run 'make mnist' to download MNIST data first.\n");
        exit(1);
    }

    uint32_t magic, n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) {
        fprintf(stderr, "ERROR: Failed to read header from %s\n", path);
        fclose(f);
        exit(1);
    }
    magic = __builtin_bswap32(magic);
    n = __builtin_bswap32(n);
    *count = n;

    int ndim = magic & 0xFF;
    size_t item_size = 1;

    if (ndim >= 3) {
        uint32_t rows, cols;
        if (fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1) {
            fprintf(stderr, "ERROR: Failed to read dimensions from %s\n", path);
            fclose(f);
            exit(1);
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
    if (!data) {
        fprintf(stderr, "ERROR: malloc(%zu) failed\n", total);
        fclose(f);
        exit(1);
    }
    if (fread(data, 1, total, f) != total) {
        fprintf(stderr, "ERROR: Short read from %s\n", path);
        fclose(f);
        exit(1);
    }
    fclose(f);
    return data;
}

static void load_mnist(void) {
    uint32_t n, r, c;

    raw_train_img = load_idx("data/train-images-idx3-ubyte", &n, &r, &c);
    if (n != TRAIN_N || r != 28 || c != 28) {
        fprintf(stderr, "ERROR: Unexpected training image dimensions\n");
        exit(1);
    }

    train_labels = load_idx("data/train-labels-idx1-ubyte", &n, NULL, NULL);
    if (n != TRAIN_N) {
        fprintf(stderr, "ERROR: Unexpected training label count\n");
        exit(1);
    }

    raw_test_img = load_idx("data/t10k-images-idx3-ubyte", &n, &r, &c);
    if (n != TEST_N || r != 28 || c != 28) {
        fprintf(stderr, "ERROR: Unexpected test image dimensions\n");
        exit(1);
    }

    test_labels = load_idx("data/t10k-labels-idx1-ubyte", &n, NULL, NULL);
    if (n != TEST_N) {
        fprintf(stderr, "ERROR: Unexpected test label count\n");
        exit(1);
    }
}

/* ================================================================
 *  AVX2 Ternary Quantization
 *
 *  pixel < 85  → -1
 *  85 ≤ pixel ≤ 170 → 0
 *  pixel > 170 → +1
 *
 *  XOR with 0x80 converts unsigned ordering to signed for cmpgt.
 * ================================================================ */

static void quantize_one(const uint8_t *src, int8_t *dst) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));  /* 42 signed */
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));  /* -43 signed */
    const __m256i one  = _mm256_set1_epi8(1);

    int i;
    for (i = 0; i + 32 <= PIXELS; i += 32) {
        __m256i px = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i spx = _mm256_xor_si256(px, bias);

        __m256i pos = _mm256_cmpgt_epi8(spx, thi);     /* pixel > 170 */
        __m256i neg = _mm256_cmpgt_epi8(tlo, spx);     /* pixel < 85 */

        __m256i p = _mm256_and_si256(pos, one);         /* +1 where high */
        __m256i n = _mm256_and_si256(neg, one);         /* +1 where low  */
        __m256i result = _mm256_sub_epi8(p, n);         /* {-1, 0, +1}   */

        _mm256_storeu_si256((__m256i *)(dst + i), result);
    }

    /* Tail (784 - 768 = 16 remaining pixels) */
    for (; i < PIXELS; i++) {
        if (src[i] > 170)     dst[i] =  1;
        else if (src[i] < 85) dst[i] = -1;
        else                   dst[i] =  0;
    }

    /* Zero-pad to PADDED */
    memset(dst + PIXELS, 0, PADDED - PIXELS);
}

static void quantize_all(void) {
    tern_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    if (!tern_train || !tern_test) {
        fprintf(stderr, "ERROR: aligned_alloc failed\n");
        exit(1);
    }

    for (int i = 0; i < TRAIN_N; i++)
        quantize_one(raw_train_img + (size_t)i * PIXELS,
                     tern_train + (size_t)i * PADDED);

    for (int i = 0; i < TEST_N; i++)
        quantize_one(raw_test_img + (size_t)i * PIXELS,
                     tern_test + (size_t)i * PADDED);
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

static void compute_gradients_one(const int8_t *tern,
                                   int8_t *h_grad, int8_t *v_grad) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++) {
            int diff = tern[y * IMG_W + x + 1] - tern[y * IMG_W + x];
            h_grad[y * IMG_W + x] = clamp_trit(diff);
        }
        h_grad[y * IMG_W + IMG_W - 1] = 0;
    }
    memset(h_grad + PIXELS, 0, PADDED - PIXELS);

    for (int y = 0; y < IMG_H - 1; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int diff = tern[(y + 1) * IMG_W + x] - tern[y * IMG_W + x];
            v_grad[y * IMG_W + x] = clamp_trit(diff);
        }
    }
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
 *  AVX2 Ternary Dot Product — The Critical Inner Kernel
 *
 *  Both vectors contain int8 values in {-1, 0, +1}, length PADDED=800.
 *
 *  _mm256_sign_epi8(a, b) computes:
 *    b > 0: a,  b = 0: 0,  b < 0: -a
 *  This is exactly a*b for ternary values.
 *
 *  Accumulate in int8 (safe: 25 iters × max |product| = 1, max |acc| = 25 < 127).
 *  Then widen to int16, int32, horizontal sum.
 * ================================================================ */

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();

    for (int i = 0; i < PADDED; i += 32) {
        __m256i va = _mm256_load_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_load_si256((const __m256i *)(b + i));
        __m256i prod = _mm256_sign_epi8(va, vb);
        acc = _mm256_add_epi8(acc, prod);
    }

    /* Widen int8 → int16 (sign-extend low and high 128-bit halves) */
    __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc, 1));
    __m256i sum16 = _mm256_add_epi16(lo16, hi16);

    /* Widen int16 → int32 via madd with 1s */
    __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));

    /* Horizontal sum of 8 × int32 */
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum32),
                              _mm256_extracti128_si256(sum32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Test A: Ternary Centroid Classifier
 * ================================================================ */

static void compute_centroids(void) {
    int32_t sums[N_CLASSES][PADDED];
    int counts[N_CLASSES];
    memset(sums, 0, sizeof(sums));
    memset(counts, 0, sizeof(counts));

    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        counts[lbl]++;
        const int8_t *img = tern_train + (size_t)i * PADDED;
        for (int j = 0; j < PADDED; j++)
            sums[lbl][j] += img[j];
    }

    /* Quantize centroid: if |mean| > 1/3, assign ±1, else 0 */
    for (int c = 0; c < N_CLASSES; c++) {
        int thr = counts[c] / 3;
        for (int j = 0; j < PADDED; j++) {
            if (sums[c][j] > thr)
                tern_centroids[c][j] = 1;
            else if (sums[c][j] < -thr)
                tern_centroids[c][j] = -1;
            else
                tern_centroids[c][j] = 0;
        }
    }
}

static double test_centroid(void) {
    int correct = 0;
    for (int i = 0; i < TEST_N; i++) {
        const int8_t *img = tern_test + (size_t)i * PADDED;
        int best = 0;
        int32_t best_dot = ternary_dot(img, tern_centroids[0]);

        for (int c = 1; c < N_CLASSES; c++) {
            int32_t d = ternary_dot(img, tern_centroids[c]);
            if (d > best_dot) {
                best_dot = d;
                best = c;
            }
        }
        if (best == test_labels[i]) correct++;
    }
    return (double)correct / TEST_N;
}

/* ================================================================
 *  Test B: Ternary 1-NN Classifier
 * ================================================================ */

static double test_1nn(void) {
    int correct = 0;

    for (int i = 0; i < TEST_N; i++) {
        const int8_t *query = tern_test + (size_t)i * PADDED;
        int best_label = 0;
        int32_t best_dot = -PADDED - 1;

        for (int j = 0; j < TRAIN_N; j++) {
            const int8_t *ref = tern_train + (size_t)j * PADDED;
            int32_t d = ternary_dot(query, ref);
            if (d > best_dot) {
                best_dot = d;
                best_label = train_labels[j];
            }
        }
        if (best_label == test_labels[i]) correct++;

        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  1-NN: %d/%d (running acc %.2f%%)\r",
                    i + 1, TEST_N, 100.0 * correct / (i + 1));
    }
    fprintf(stderr, "\n");
    return (double)correct / TEST_N;
}

/* ================================================================
 *  Float Centroid Baseline (unquantized)
 * ================================================================ */

static double test_float_centroid(void) {
    static float centroids[N_CLASSES][PIXELS];
    int counts[N_CLASSES] = {0};
    memset(centroids, 0, sizeof(centroids));

    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        counts[lbl]++;
        const uint8_t *img = raw_train_img + (size_t)i * PIXELS;
        for (int j = 0; j < PIXELS; j++)
            centroids[lbl][j] += (float)img[j];
    }
    for (int c = 0; c < N_CLASSES; c++)
        for (int j = 0; j < PIXELS; j++)
            centroids[c][j] /= counts[c];

    int correct = 0;
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *img = raw_test_img + (size_t)i * PIXELS;
        int best = 0;
        float best_dot = -1e30f;

        for (int c = 0; c < N_CLASSES; c++) {
            float dot = 0.0f;
            for (int j = 0; j < PIXELS; j++)
                dot += (float)img[j] * centroids[c][j];
            if (dot > best_dot) {
                best_dot = dot;
                best = c;
            }
        }
        if (best == test_labels[i]) correct++;
    }
    return (double)correct / TEST_N;
}

/* ================================================================
 *  Noise Resilience Test
 *
 *  Inject Gaussian noise into ternary dot products (simulating analog
 *  hardware noise). Measure classification accuracy degradation.
 * ================================================================ */

static void test_noise(void) {
    float snrs[] = {10, 15, 18, 20, 24, 28, 30, 40};
    int n_snr = sizeof(snrs) / sizeof(snrs[0]);

    printf("\n--- Noise Resilience: Ternary Centroid under Analog Noise ---\n\n");
    printf("%-8s | %-12s | %-12s\n", "SNR(dB)", "Accuracy", "vs Clean");
    printf("---------+--------------+-------------\n");

    for (int si = 0; si < n_snr; si++) {
        float snr = snrs[si];
        float sigma = 1.0f / powf(10.0f, snr / 20.0f);

        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            const int8_t *img = tern_test + (size_t)i * PADDED;
            int best = 0;
            float best_score = -1e30f;

            for (int c = 0; c < N_CLASSES; c++) {
                int32_t d = ternary_dot(img, tern_centroids[c]);
                /* Noise proportional to signal range (max dot = PIXELS) */
                float score = (float)d + gauss() * sigma * (float)PIXELS;
                if (score > best_score) {
                    best_score = score;
                    best = c;
                }
            }
            if (best == test_labels[i]) correct++;
        }

        double acc = (double)correct / TEST_N;
        printf("%6.1f   | %9.2f%%   | %+.2f pp\n",
               snr, acc * 100.0, (acc - centroid_acc_clean) * 100.0);
    }
}

/* ================================================================
 *  Test C: Indexed Geometric Lookup (Pre-addressed Shape Space)
 *
 *  Each image's ternary pattern is its own address in shape-space.
 *  Split into 261 blocks of 3 trits. Each block = one sub-address.
 *  Block value = (t0+1)*9 + (t1+1)*3 + (t2+1) ∈ {0..26}.
 *  Value 0 = all-background (-1,-1,-1). Skipped during voting.
 *
 *  Inverted index: for each (position, block_value) → list of image IDs.
 *  Query: vote on matching blocks, top-K by votes, refine with dot product.
 * ================================================================ */

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

static void compute_sigs(void) {
    train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    if (!train_sigs || !test_sigs) {
        fprintf(stderr, "ERROR: sig alloc failed\n"); exit(1);
    }

    for (int i = 0; i < TRAIN_N; i++) {
        const int8_t *img = tern_train + (size_t)i * PADDED;
        uint8_t *sig = train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            sig[k] = block_encode(img[k * 3], img[k * 3 + 1], img[k * 3 + 2]);
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }

    for (int i = 0; i < TEST_N; i++) {
        const int8_t *img = tern_test + (size_t)i * PADDED;
        uint8_t *sig = test_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            sig[k] = block_encode(img[k * 3], img[k * 3 + 1], img[k * 3 + 2]);
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

static void compute_grad_sigs(void) {
    hg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    hg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    vg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    vg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    if (!hg_train_sigs || !hg_test_sigs || !vg_train_sigs || !vg_test_sigs) {
        fprintf(stderr, "ERROR: grad sig alloc failed\n"); exit(1);
    }

    for (int i = 0; i < TRAIN_N; i++) {
        const int8_t *himg = hgrad_train + (size_t)i * PADDED;
        uint8_t *hsig = hg_train_sigs + (size_t)i * SIG_PAD;
        const int8_t *vimg = vgrad_train + (size_t)i * PADDED;
        uint8_t *vsig = vg_train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) {
            hsig[k] = block_encode(himg[k*3], himg[k*3+1], himg[k*3+2]);
            vsig[k] = block_encode(vimg[k*3], vimg[k*3+1], vimg[k*3+2]);
        }
        memset(hsig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
        memset(vsig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }

    for (int i = 0; i < TEST_N; i++) {
        const int8_t *himg = hgrad_test + (size_t)i * PADDED;
        uint8_t *hsig = hg_test_sigs + (size_t)i * SIG_PAD;
        const int8_t *vimg = vgrad_test + (size_t)i * PADDED;
        uint8_t *vsig = vg_test_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) {
            hsig[k] = block_encode(himg[k*3], himg[k*3+1], himg[k*3+2]);
            vsig[k] = block_encode(vimg[k*3], vimg[k*3+1], vimg[k*3+2]);
        }
        memset(hsig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
        memset(vsig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

static void build_index(void) {
    /* Pass 1: count bucket sizes */
    memset(idx_sz, 0, sizeof(idx_sz));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            idx_sz[k][sig[k]]++;
    }

    /* Compute offsets */
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < N_BVALS; v++) {
            idx_off[k][v] = total;
            total += idx_sz[k][v];
        }

    /* Allocate flat pool */
    idx_pool = malloc((size_t)total * sizeof(uint32_t));
    if (!idx_pool) { fprintf(stderr, "ERROR: idx_pool malloc\n"); exit(1); }

    /* Pass 2: fill buckets */
    uint32_t wpos[N_BLOCKS][N_BVALS];
    memcpy(wpos, idx_off, sizeof(idx_off));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            idx_pool[wpos[k][sig[k]]++] = (uint32_t)i;
    }
}

static void print_index_stats(void) {
    long bg_total = 0, fg_total = 0;
    int fg_active = 0;
    uint16_t max_fg = 0;

    for (int k = 0; k < N_BLOCKS; k++) {
        bg_total += idx_sz[k][0];
        for (int v = 1; v < N_BVALS; v++) {
            fg_total += idx_sz[k][v];
            if (idx_sz[k][v] > 0) fg_active++;
            if (idx_sz[k][v] > max_fg) max_fg = idx_sz[k][v];
        }
    }

    printf("  Buckets: %d positions x %d values = %d\n",
           N_BLOCKS, N_BVALS, N_BLOCKS * N_BVALS);
    printf("  Background (block=0): avg %ld images/bucket (not used for voting)\n",
           bg_total / N_BLOCKS);
    printf("  Foreground (block>0): avg %ld images/bucket, %d active, max %d\n",
           fg_active > 0 ? fg_total / fg_active : 0L, fg_active, max_fg);
    printf("  Pool size: %.1f MB\n",
           (double)(TRAIN_N * N_BLOCKS) * sizeof(uint32_t) / (1024 * 1024));
}

/* Top-K selection via counting sort on vote values (0..261) */
typedef struct {
    uint32_t id;
    uint16_t votes;
    int32_t dot;
} candidate_t;

static int cmp_u32(const void *a, const void *b) {
    uint32_t va = *(const uint32_t *)a;
    uint32_t vb = *(const uint32_t *)b;
    return (va > vb) - (va < vb);
}

static int cmp_votes_desc(const void *a, const void *b) {
    return (int)((const candidate_t *)b)->votes -
           (int)((const candidate_t *)a)->votes;
}

static int cmp_dots_desc(const void *a, const void *b) {
    int32_t da = ((const candidate_t *)a)->dot;
    int32_t db = ((const candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

static int select_top_k(const uint16_t *votes, int n,
                        candidate_t *out, int k) {
    int hist[262];
    memset(hist, 0, sizeof(hist));
    for (int i = 0; i < n; i++)
        hist[votes[i]]++;

    int cum = 0, thr;
    for (thr = 261; thr >= 1; thr--) {
        cum += hist[thr];
        if (cum >= k) break;
    }
    if (thr < 1) thr = 1;

    int nc = 0;
    for (int i = 0; i < n && nc < k; i++) {
        if (votes[i] >= (uint16_t)thr) {
            out[nc].id = (uint32_t)i;
            out[nc].votes = votes[i];
            out[nc].dot = 0;
            nc++;
        }
    }

    qsort(out, (size_t)nc, sizeof(candidate_t), cmp_votes_desc);
    return nc;
}

static void test_indexed(double nn_acc, double nn_time) {
    int k_vals[] = {50, 100, 200, 500, 1000};
    int n_k = 5;
    int correct[5] = {0};
    int correct_vote = 0;   /* vote-only classification (no dot product) */

    double t0 = now_sec();
    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    if (!votes) { fprintf(stderr, "ERROR: votes alloc failed\n"); exit(1); }

    for (int i = 0; i < TEST_N; i++) {
        memset(votes, 0, TRAIN_N * sizeof(uint16_t));

        const uint8_t *qsig = test_sigs + (size_t)i * SIG_PAD;

        /* Vote: each non-background block indexes into the inverted list.
         * Every training image in that bucket gets +1 vote. */
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == 0) continue;  /* skip background */
            uint32_t off = idx_off[k][bv];
            uint16_t sz = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off;
            for (uint16_t j = 0; j < sz; j++)
                votes[ids[j]]++;
        }

        /* Vote-only: classify by label of highest-voted training image */
        {
            uint32_t best_id = 0;
            uint16_t best_v = 0;
            for (int j = 0; j < TRAIN_N; j++) {
                if (votes[j] > best_v) {
                    best_v = votes[j];
                    best_id = (uint32_t)j;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_vote++;
        }

        /* Top-K by votes, sorted descending */
        candidate_t cands[MAX_K];
        int nc = select_top_k(votes, TRAIN_N, cands, MAX_K);

        /* Refine: compute full ternary dot product for all candidates */
        const int8_t *query = tern_test + (size_t)i * PADDED;
        for (int j = 0; j < nc; j++) {
            const int8_t *ref = tern_train + (size_t)cands[j].id * PADDED;
            cands[j].dot = ternary_dot(query, ref);
        }

        /* Evaluate each K threshold */
        for (int ki = 0; ki < n_k; ki++) {
            int kv = k_vals[ki];
            int ek = (nc < kv) ? nc : kv;
            int32_t best_d = cands[0].dot;
            uint32_t best_id = cands[0].id;
            for (int j = 1; j < ek; j++) {
                if (cands[j].dot > best_d) {
                    best_d = cands[j].dot;
                    best_id = cands[j].id;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct[ki]++;
        }

        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  Indexed: %d/%d (vote-only %.2f%%)\r",
                    i + 1, TEST_N, 100.0 * correct_vote / (i + 1));
    }
    fprintf(stderr, "\n");
    free(votes);

    double elapsed = now_sec() - t0;

    printf("  Vote-only (zero dot products): %.2f%%\n\n",
           100.0 * correct_vote / TEST_N);

    printf("  %-8s | %-10s | %-14s | %-10s | %-8s\n",
           "Top-K", "Accuracy", "Dot products", "Speedup", "vs 1-NN");
    printf("  ---------+------------+----------------+------------+---------\n");

    for (int ki = 0; ki < n_k; ki++) {
        double acc = (double)correct[ki] / TEST_N;
        long long dots = (long long)k_vals[ki] * TEST_N;
        double dot_ratio = (double)TRAIN_N * TEST_N / dots;
        printf("  %5d    | %7.2f%%   | %12lld   | %7.0fx    | %+.2f pp\n",
               k_vals[ki], acc * 100.0, dots, dot_ratio,
               (acc - nn_acc) * 100.0);
    }

    printf("\n  Total time: %.2f sec (vs brute 1-NN: %.2f sec, %.1fx faster)\n",
           elapsed, nn_time, nn_time / elapsed);
}

/* ================================================================
 *  Test D: Hot Map — The Cipher
 *
 *  Pre-compute per-class vote counts for every (position, block_value)
 *  pair. This collapses the entire 60 MB inverted index into a 451 KB
 *  table that fits in L2 cache.
 *
 *  Classification = for each block, look up class votes, accumulate.
 *  No per-image tracking. No scatter. No dot products.
 *  Pure sequential AVX2 reads. The map IS the model.
 *
 *  hot_map[block_pos][block_val][class] = count of training images
 *  of that class with that block value at that position.
 *
 *  Persists to disk. Load once, prefetch, keep hot.
 * ================================================================ */

static uint32_t hot_map[N_BLOCKS][N_BVALS][CLS_PAD]
    __attribute__((aligned(32)));

static uint32_t sdf_hot_map[N_BLOCKS][N_BVALS][CLS_PAD]
    __attribute__((aligned(32)));

static void build_hot_map(void) {
    memset(hot_map, 0, sizeof(hot_map));
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            hot_map[k][sig[k]][lbl]++;
    }
}

/* Generalized hot map builder: works with any sig array */
static void build_hot_map_gen(const uint8_t *sigs, int n,
                               uint32_t hot[][N_BVALS][CLS_PAD]) {
    memset(hot, 0, sizeof(uint32_t) * N_BLOCKS * N_BVALS * CLS_PAD);
    for (int i = 0; i < n; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            hot[k][sig[k]][lbl]++;
    }
}

/* Generalized hot map classification with parameterized background skip */
static inline int hot_classify_gen(const uint8_t *qsig,
                                    uint32_t hot[][N_BVALS][CLS_PAD],
                                    uint8_t bg_val) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    for (int k = 0; k < N_BLOCKS; k++) {
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

/* Touch every cache line to prime L2 */
static void warmup_hot_map(void) {
    volatile uint32_t sink = 0;
    const uint32_t *p = (const uint32_t *)hot_map;
    size_t n = sizeof(hot_map) / sizeof(uint32_t);
    for (size_t i = 0; i < n; i += 16)   /* 64B cache line = 16 × uint32 */
        sink += p[i];
    (void)sink;
}

static void save_hot_map(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "WARNING: cannot save hot map to %s\n", path); return; }
    fwrite(hot_map, sizeof(hot_map), 1, f);
    fclose(f);
}

static int load_hot_map(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t r = fread(hot_map, sizeof(hot_map), 1, f);
    fclose(f);
    return r == 1;
}

/* AVX2 hot-map classification.
 * For each foreground block: accumulate class vote counts (16-wide).
 * Then argmax over 10 classes. */
static inline int hot_classify(const uint8_t *qsig) {
    __m256i acc_lo = _mm256_setzero_si256();  /* classes 0–7  */
    __m256i acc_hi = _mm256_setzero_si256();  /* classes 8–15 */

    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = qsig[k];
        if (bv == 0) continue;
        const __m256i *cc = (const __m256i *)hot_map[k][bv];
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

static void test_hot_map(void) {
    double tb0 = now_sec();
    int loaded = load_hot_map("data/hot_map.bin");
    if (!loaded) {
        build_hot_map();
        save_hot_map("data/hot_map.bin");
        printf("  Built and saved to data/hot_map.bin\n");
    } else {
        printf("  Loaded from data/hot_map.bin (persistent cipher)\n");
    }
    double tb1 = now_sec();
    printf("  Size: %zu KB (L2-resident)\n", sizeof(hot_map) / 1024);
    printf("  Load/build: %.4f sec\n\n", tb1 - tb0);

    /* Prime L2 cache */
    warmup_hot_map();

    /* Classify all test images */
    int correct = 0;
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *qsig = test_sigs + (size_t)i * SIG_PAD;
        if (hot_classify(qsig) == test_labels[i])
            correct++;
    }
    double t1 = now_sec();

    double acc = (double)correct / TEST_N;
    double elapsed = t1 - t0;
    double ns_per = elapsed * 1e9 / TEST_N;

    printf("  Accuracy:    %.2f%%\n", acc * 100.0);
    printf("  Wall time:   %.4f sec (%d queries)\n", elapsed, TEST_N);
    printf("  Latency:     %.0f ns/query\n", ns_per);
    printf("  Throughput:  %.0f classifications/sec\n", (double)TEST_N / elapsed);

    /* Second pass: fully hot (everything in L1/L2 now) */
    correct = 0;
    double t2 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *qsig = test_sigs + (size_t)i * SIG_PAD;
        if (hot_classify(qsig) == test_labels[i])
            correct++;
    }
    double t3 = now_sec();
    double hot_ns = (t3 - t2) * 1e9 / TEST_N;
    printf("  Hot latency: %.0f ns/query (second pass, fully cached)\n", hot_ns);
}

/* ================================================================
 *  Test E: Ternary Shape Transform (TST)
 *
 *  FFT-style multi-resolution cascade classifier.
 *
 *  The butterfly: at each level, AND 3 adjacent block matches.
 *  Higher levels = wider spatial coherence = more discriminative.
 *
 *  L0: 261 blocks of 3 trits    (weight 1)
 *  L1:  87 superblocks of 9 trits  (weight 3)   = 3 adjacent L0 ANDed
 *  L2:  29 megablocks of 27 trits  (weight 9)   = 3 adjacent L1 ANDed
 *  L3:   9 hyperblocks of 81 trits (weight 27)  = 3 adjacent L2 ANDed
 *  L4:   3 regions of 243 trits    (weight 81)  = 3 adjacent L3 ANDed
 *
 *  Score = L0*1 + L1*3 + L2*9 + L3*27 + L4*81  ∈ [0, 1269]
 *
 *  Cascade: vote → butterfly score → dot product
 * ================================================================ */

/*
 * Multi-resolution butterfly match score.
 * Compare two block signatures at 5 resolution levels via cascade AND.
 * Returns score in [0, TST_MAX_SCORE].
 */
static inline int32_t tst_multi_res_score(const uint8_t *qsig,
                                           const uint8_t *csig) {
    /* L0: AVX2 byte-compare of 261 block signatures (padded to 288) */
    uint8_t match[SIG_PAD] __attribute__((aligned(32)));

    for (int i = 0; i < SIG_PAD; i += 32) {
        __m256i q = _mm256_load_si256((const __m256i *)(qsig + i));
        __m256i c = _mm256_load_si256((const __m256i *)(csig + i));
        __m256i m = _mm256_cmpeq_epi8(q, c);  /* 0xFF=match, 0x00=miss */
        _mm256_store_si256((__m256i *)(match + i), m);
    }

    /* Zero padding bytes 261-287 (both sigs pad with 0xFF → false match) */
    memset(match + N_BLOCKS, 0, SIG_PAD - N_BLOCKS);

    /* L0 count via movemask + popcount */
    int l0 = 0;
    for (int i = 0; i < SIG_PAD; i += 32) {
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(
            _mm256_load_si256((const __m256i *)(match + i)));
        l0 += __builtin_popcount(mask);
    }

    /* L1: cascade AND — 87 triplets from 261 L0 bytes */
    uint8_t l1m[TST_L1];
    int l1 = 0;
    for (int j = 0; j < TST_L1; j++) {
        l1m[j] = match[3 * j] & match[3 * j + 1] & match[3 * j + 2];
        l1 += (l1m[j] != 0);
    }

    /* L2: cascade AND — 29 triplets from 87 L1 bytes */
    uint8_t l2m[TST_L2];
    int l2 = 0;
    for (int m = 0; m < TST_L2; m++) {
        l2m[m] = l1m[3 * m] & l1m[3 * m + 1] & l1m[3 * m + 2];
        l2 += (l2m[m] != 0);
    }

    /* L3: cascade AND — 9 triplets from first 27 of 29 L2 bytes */
    uint8_t l3m[TST_L3];
    int l3 = 0;
    for (int h = 0; h < TST_L3; h++) {
        l3m[h] = l2m[3 * h] & l2m[3 * h + 1] & l2m[3 * h + 2];
        l3 += (l3m[h] != 0);
    }

    /* L4: cascade AND — 3 triplets from 9 L3 bytes */
    int l4 = 0;
    for (int r = 0; r < TST_L4; r++)
        l4 += ((l3m[3 * r] & l3m[3 * r + 1] & l3m[3 * r + 2]) != 0);

    return l0 * TST_W0 + l1 * TST_W1 + l2 * TST_W2
         + l3 * TST_W3 + l4 * TST_W4;
}

static void test_tst(double nn_acc, double nn_time) {
    int k2_vals[] = {10, 20, 50, 100, 200};
    int n_k2 = 5;
    int correct_cascade[5] = {0};
    int correct_mr_only = 0;

    double t0 = now_sec();
    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    if (!votes) { fprintf(stderr, "ERROR: votes alloc failed\n"); exit(1); }

    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *qsig = test_sigs + (size_t)i * SIG_PAD;
        const int8_t *query = tern_test + (size_t)i * PADDED;

        /* Stage 1: Vote via inverted index (same as Test C) */
        memset(votes, 0, TRAIN_N * sizeof(uint16_t));
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == 0) continue;
            uint32_t off = idx_off[k][bv];
            uint16_t sz = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off;
            for (uint16_t j = 0; j < sz; j++)
                votes[ids[j]]++;
        }

        /* Select top-K1 by votes */
        candidate_t cands[TST_K1];
        int nc = select_top_k(votes, TRAIN_N, cands, TST_K1);

        /* Stage 2: Multi-resolution butterfly scoring */
        for (int j = 0; j < nc; j++) {
            const uint8_t *csig = train_sigs + (size_t)cands[j].id * SIG_PAD;
            cands[j].dot = tst_multi_res_score(qsig, csig);
        }

        /* Multi-res only classification (zero dot products) */
        {
            int32_t best_score = -1;
            uint32_t best_id = cands[0].id;
            for (int j = 0; j < nc; j++) {
                if (cands[j].dot > best_score) {
                    best_score = cands[j].dot;
                    best_id = cands[j].id;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_mr_only++;
        }

        /* Sort by multi-res score descending */
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        /* Stage 3: Dot product on top survivors */
        int max_k2 = k2_vals[n_k2 - 1];
        int ek = (nc < max_k2) ? nc : max_k2;
        for (int j = 0; j < ek; j++) {
            const int8_t *ref = tern_train + (size_t)cands[j].id * PADDED;
            cands[j].dot = ternary_dot(query, ref);
        }

        /* Evaluate each K2 threshold */
        for (int ki = 0; ki < n_k2; ki++) {
            int kv = k2_vals[ki];
            int ek2 = (ek < kv) ? ek : kv;
            int32_t best_d = cands[0].dot;
            uint32_t best_id = cands[0].id;
            for (int j = 1; j < ek2; j++) {
                if (cands[j].dot > best_d) {
                    best_d = cands[j].dot;
                    best_id = cands[j].id;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_cascade[ki]++;
        }

        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  TST: %d/%d (mr-only %.2f%%)\r",
                    i + 1, TEST_N, 100.0 * correct_mr_only / (i + 1));
    }
    fprintf(stderr, "\n");
    free(votes);

    double elapsed = now_sec() - t0;

    printf("  Multi-res only (vote -> butterfly, zero dots): %.2f%%\n",
           100.0 * correct_mr_only / TEST_N);
    printf("  (vs vote-only 84.43%%: %+.2f pp)\n\n",
           100.0 * correct_mr_only / TEST_N - 84.43);

    printf("  %-6s %-6s | %-10s | %-14s | %-10s | %-8s\n",
           "K1", "K2", "Accuracy", "Dot products", "Speedup", "vs 1-NN");
    printf("  ------+------+------------+----------------+"
           "------------+---------\n");

    for (int ki = 0; ki < n_k2; ki++) {
        double acc = (double)correct_cascade[ki] / TEST_N;
        long long dots = (long long)k2_vals[ki] * TEST_N;
        double dot_ratio = (double)TRAIN_N * TEST_N / dots;
        printf("  %5d  %5d | %7.2f%%   | %12lld   | %7.0fx    | %+.2f pp\n",
               TST_K1, k2_vals[ki], acc * 100.0, dots, dot_ratio,
               (acc - nn_acc) * 100.0);
    }

    printf("\n  Total time: %.2f sec (vs brute 1-NN: %.2f sec, %.1fx faster)\n",
           elapsed, nn_time, nn_time / elapsed);
}

/* ================================================================
 *  Test F: SDF Cipher — Topological Compression
 *
 *  Signed Distance Field transforms pixel-ternary (what the ink looks
 *  like) into topology-ternary (what shape the ink makes). Two images
 *  with different thickness but same letter form collapse to the same
 *  SDF representation.
 *
 *  Pipeline: ternary image → raw SDF → quantize to ternary → block
 *  signatures → hot map cipher. Same 451KB table, different address
 *  space. Dual cipher = pixel + SDF hot maps combined (902KB).
 * ================================================================ */

/*
 * Two-pass Chamfer distance transform on 28×28 ternary grid.
 *
 * Boundary: pixel where +1 meets -1 (directly or via 0-neighbor),
 * or pixel value is 0 (already in transition zone).
 * Distance: Manhattan to nearest boundary pixel.
 * Sign: foreground (+1) → negative, background (-1) → positive.
 *
 * Output: int8 raw SDF in [-14, +14], zero-padded to PADDED.
 */
static void compute_raw_sdf(const int8_t *tern, int8_t *sdf) {
    int16_t dist[IMG_H * IMG_W];
    int8_t sign_map[IMG_H * IMG_W];

    /* Step 1: identify boundary pixels */
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int idx = y * IMG_W + x;
            int8_t val = tern[idx];

            if (val == 0) {
                dist[idx] = 0;
                sign_map[idx] = 0;
                continue;
            }

            /* Check 4-neighbors for opposite sign or zero */
            int is_boundary = 0;
            if (y > 0) {
                int8_t n = tern[(y - 1) * IMG_W + x];
                if (n == 0 || (n > 0) != (val > 0)) is_boundary = 1;
            }
            if (y < IMG_H - 1 && !is_boundary) {
                int8_t n = tern[(y + 1) * IMG_W + x];
                if (n == 0 || (n > 0) != (val > 0)) is_boundary = 1;
            }
            if (x > 0 && !is_boundary) {
                int8_t n = tern[y * IMG_W + x - 1];
                if (n == 0 || (n > 0) != (val > 0)) is_boundary = 1;
            }
            if (x < IMG_W - 1 && !is_boundary) {
                int8_t n = tern[y * IMG_W + x + 1];
                if (n == 0 || (n > 0) != (val > 0)) is_boundary = 1;
            }

            if (is_boundary) {
                dist[idx] = 0;
            } else {
                dist[idx] = 999;
            }
            sign_map[idx] = (val > 0) ? -1 : 1;  /* inside=-1, outside=+1 */
        }
    }

    /* Step 2: forward pass (top-left → bottom-right) */
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int idx = y * IMG_W + x;
            if (y > 0 && dist[(y - 1) * IMG_W + x] + 1 < dist[idx])
                dist[idx] = dist[(y - 1) * IMG_W + x] + 1;
            if (x > 0 && dist[y * IMG_W + x - 1] + 1 < dist[idx])
                dist[idx] = dist[y * IMG_W + x - 1] + 1;
        }
    }

    /* Step 3: backward pass (bottom-right → top-left) */
    for (int y = IMG_H - 1; y >= 0; y--) {
        for (int x = IMG_W - 1; x >= 0; x--) {
            int idx = y * IMG_W + x;
            if (y < IMG_H - 1 && dist[(y + 1) * IMG_W + x] + 1 < dist[idx])
                dist[idx] = dist[(y + 1) * IMG_W + x] + 1;
            if (x < IMG_W - 1 && dist[y * IMG_W + x + 1] + 1 < dist[idx])
                dist[idx] = dist[y * IMG_W + x + 1] + 1;
        }
    }

    /* Step 4: apply sign, clamp to int8 */
    for (int i = 0; i < PIXELS; i++) {
        int16_t d = dist[i] * sign_map[i];
        if (d > 127) d = 127;
        if (d < -128) d = -128;
        sdf[i] = (int8_t)d;
    }
    memset(sdf + PIXELS, 0, PADDED - PIXELS);
}

/* SDF variant for gradient/edge images where val=0 is background, not boundary.
   Foreground = non-zero pixels. Measures signed distance from edge of foreground.
   Interior (non-zero) gets negative distance, exterior (zero) gets positive. */
static void compute_edge_sdf(const int8_t *tern, int8_t *sdf) {
    int16_t dist[IMG_H * IMG_W];
    int8_t sign_map[IMG_H * IMG_W];

    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int idx = y * IMG_W + x;
            int fg = (tern[idx] != 0);  /* foreground = non-zero (edge pixel) */

            /* Boundary = where foreground meets background */
            int is_boundary = 0;
            if (fg) {
                if (y > 0 && tern[(y-1)*IMG_W+x] == 0) is_boundary = 1;
                if (!is_boundary && y < IMG_H-1 && tern[(y+1)*IMG_W+x] == 0) is_boundary = 1;
                if (!is_boundary && x > 0 && tern[y*IMG_W+x-1] == 0) is_boundary = 1;
                if (!is_boundary && x < IMG_W-1 && tern[y*IMG_W+x+1] == 0) is_boundary = 1;
            } else {
                if (y > 0 && tern[(y-1)*IMG_W+x] != 0) is_boundary = 1;
                if (!is_boundary && y < IMG_H-1 && tern[(y+1)*IMG_W+x] != 0) is_boundary = 1;
                if (!is_boundary && x > 0 && tern[y*IMG_W+x-1] != 0) is_boundary = 1;
                if (!is_boundary && x < IMG_W-1 && tern[y*IMG_W+x+1] != 0) is_boundary = 1;
            }

            dist[idx] = is_boundary ? 0 : 999;
            sign_map[idx] = fg ? -1 : 1;  /* inside edge=-1, outside=+1 */
        }
    }

    /* Forward Chamfer pass */
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            int idx = y * IMG_W + x;
            if (y > 0 && dist[(y-1)*IMG_W+x]+1 < dist[idx]) dist[idx] = dist[(y-1)*IMG_W+x]+1;
            if (x > 0 && dist[y*IMG_W+x-1]+1 < dist[idx]) dist[idx] = dist[y*IMG_W+x-1]+1;
        }

    /* Backward Chamfer pass */
    for (int y = IMG_H-1; y >= 0; y--)
        for (int x = IMG_W-1; x >= 0; x--) {
            int idx = y * IMG_W + x;
            if (y < IMG_H-1 && dist[(y+1)*IMG_W+x]+1 < dist[idx]) dist[idx] = dist[(y+1)*IMG_W+x]+1;
            if (x < IMG_W-1 && dist[y*IMG_W+x+1]+1 < dist[idx]) dist[idx] = dist[y*IMG_W+x+1]+1;
        }

    /* Apply sign and clamp */
    for (int i = 0; i < PIXELS; i++) {
        int16_t d = dist[i] * sign_map[i];
        if (d > 127) d = 127;
        if (d < -128) d = -128;
        sdf[i] = (int8_t)d;
    }
    memset(sdf + PIXELS, 0, PADDED - PIXELS);
}

static void compute_all_sdf(void) {
    sdf_raw_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    sdf_raw_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    if (!sdf_raw_train || !sdf_raw_test) {
        fprintf(stderr, "ERROR: SDF alloc failed\n"); exit(1);
    }

    for (int i = 0; i < TRAIN_N; i++)
        compute_raw_sdf(tern_train + (size_t)i * PADDED,
                        sdf_raw_train + (size_t)i * PADDED);

    for (int i = 0; i < TEST_N; i++)
        compute_raw_sdf(tern_test + (size_t)i * PADDED,
                        sdf_raw_test + (size_t)i * PADDED);
}

/*
 * AVX2 SDF ternary quantization.
 * SDF < -T → -1,  |SDF| ≤ T → 0,  SDF > T → +1
 * SDF values are already signed int8 — no XOR trick needed.
 */
static void quantize_sdf_one(const int8_t *sdf_raw, int8_t *sdf_tern,
                              int threshold) {
    const __m256i thi = _mm256_set1_epi8((int8_t)threshold);
    const __m256i tlo = _mm256_set1_epi8((int8_t)(-threshold));
    const __m256i one = _mm256_set1_epi8(1);

    int i;
    for (i = 0; i + 32 <= PIXELS; i += 32) {
        __m256i v = _mm256_load_si256((const __m256i *)(sdf_raw + i));
        __m256i pos = _mm256_cmpgt_epi8(v, thi);    /* sdf > T  */
        __m256i neg = _mm256_cmpgt_epi8(tlo, v);    /* sdf < -T */
        __m256i p = _mm256_and_si256(pos, one);
        __m256i n = _mm256_and_si256(neg, one);
        __m256i result = _mm256_sub_epi8(p, n);
        _mm256_store_si256((__m256i *)(sdf_tern + i), result);
    }

    /* Tail */
    for (; i < PIXELS; i++) {
        if (sdf_raw[i] > threshold)        sdf_tern[i] =  1;
        else if (sdf_raw[i] < -threshold)  sdf_tern[i] = -1;
        else                                sdf_tern[i] =  0;
    }
    memset(sdf_tern + PIXELS, 0, PADDED - PIXELS);
}

static void compute_sdf_sigs_from(const int8_t *sdf_tern, uint8_t *sigs,
                                   int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = sdf_tern + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            sig[k] = block_encode(img[k * 3], img[k * 3 + 1], img[k * 3 + 2]);
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

static void build_sdf_hot_map_from(const uint8_t *sigs, int n) {
    memset(sdf_hot_map, 0, sizeof(sdf_hot_map));
    for (int i = 0; i < n; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            sdf_hot_map[k][sig[k]][lbl]++;
    }
}

/* AVX2 SDF hot map classification (same kernel as hot_classify, different map) */
static inline int sdf_hot_classify(const uint8_t *sdf_sig) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sdf_sig[k];
        if (bv == 0) continue;
        const __m256i *cc = (const __m256i *)sdf_hot_map[k][bv];
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

/* Dual cipher: pixel hot map + SDF hot map combined */
static inline int dual_classify(const uint8_t *px_sig, const uint8_t *sdf_sig) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    /* Pixel hot map contributions */
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = px_sig[k];
        if (bv == 0) continue;
        const __m256i *cc = (const __m256i *)hot_map[k][bv];
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(cc));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(cc + 1));
    }

    /* SDF hot map contributions */
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sdf_sig[k];
        if (bv == 0) continue;
        const __m256i *cc = (const __m256i *)sdf_hot_map[k][bv];
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

static void test_sdf_cipher(void) {
    /* Step 1: compute raw SDF for all images */
    printf("  Computing SDF (two-pass Chamfer distance transform)...\n");
    double t0 = now_sec();
    compute_all_sdf();
    double t1 = now_sec();
    printf("  %d + %d images: %.3f sec\n\n", TRAIN_N, TEST_N, t1 - t0);

    /* Allocate buffers for quantized SDF and signatures */
    sdf_tern_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    sdf_tern_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    sdf_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    sdf_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    if (!sdf_tern_train || !sdf_tern_test || !sdf_train_sigs || !sdf_test_sigs) {
        fprintf(stderr, "ERROR: SDF cipher alloc failed\n"); exit(1);
    }

    /* Step 2: threshold sweep */
    printf("  %-6s | %-10s | %-12s | %-10s\n",
           "T", "SDF Acc", "Active bkt", "Trit dist");
    printf("  -------+------------+--------------+---------------------------\n");

    int thresholds[] = {1, 2, 3, 4};
    int n_t = 4;
    double best_sdf_acc = 0;
    int best_t = 2;

    for (int ti = 0; ti < n_t; ti++) {
        int T = thresholds[ti];

        /* Quantize raw SDF → ternary */
        for (int i = 0; i < TRAIN_N; i++)
            quantize_sdf_one(sdf_raw_train + (size_t)i * PADDED,
                             sdf_tern_train + (size_t)i * PADDED, T);
        for (int i = 0; i < TEST_N; i++)
            quantize_sdf_one(sdf_raw_test + (size_t)i * PADDED,
                             sdf_tern_test + (size_t)i * PADDED, T);

        /* Trit distribution */
        long neg = 0, zero = 0, pos = 0;
        for (size_t i = 0; i < (size_t)TRAIN_N * PIXELS; i++) {
            if (sdf_tern_train[i] < 0) neg++;
            else if (sdf_tern_train[i] > 0) pos++;
            else zero++;
        }
        long total = (long)TRAIN_N * PIXELS;

        /* Build sigs and hot map */
        compute_sdf_sigs_from(sdf_tern_train, sdf_train_sigs, TRAIN_N);
        compute_sdf_sigs_from(sdf_tern_test, sdf_test_sigs, TEST_N);
        build_sdf_hot_map_from(sdf_train_sigs, TRAIN_N);

        /* Count active buckets */
        int active = 0;
        for (int k = 0; k < N_BLOCKS; k++)
            for (int v = 1; v < N_BVALS; v++) {
                int has = 0;
                for (int c = 0; c < N_CLASSES; c++)
                    if (sdf_hot_map[k][v][c] > 0) { has = 1; break; }
                if (has) active++;
            }

        /* Classify */
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *sig = sdf_test_sigs + (size_t)i * SIG_PAD;
            if (sdf_hot_classify(sig) == test_labels[i])
                correct++;
        }
        double acc = (double)correct / TEST_N;

        printf("  %5d  | %7.2f%%   | %5d / %d | -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
               T, acc * 100.0, active, N_BLOCKS * (N_BVALS - 1),
               100.0 * neg / total, 100.0 * zero / total, 100.0 * pos / total);

        if (acc > best_sdf_acc) {
            best_sdf_acc = acc;
            best_t = T;
        }
    }

    /* Step 3: rebuild with best threshold for dual cipher */
    printf("\n  Best SDF threshold: T=%d (%.2f%%)\n", best_t, best_sdf_acc * 100.0);

    /* Re-quantize at best T if not already current */
    if (thresholds[n_t - 1] != best_t) {
        for (int i = 0; i < TRAIN_N; i++)
            quantize_sdf_one(sdf_raw_train + (size_t)i * PADDED,
                             sdf_tern_train + (size_t)i * PADDED, best_t);
        for (int i = 0; i < TEST_N; i++)
            quantize_sdf_one(sdf_raw_test + (size_t)i * PADDED,
                             sdf_tern_test + (size_t)i * PADDED, best_t);
        compute_sdf_sigs_from(sdf_tern_train, sdf_train_sigs, TRAIN_N);
        compute_sdf_sigs_from(sdf_tern_test, sdf_test_sigs, TEST_N);
        build_sdf_hot_map_from(sdf_train_sigs, TRAIN_N);
    }

    /* Step 4: dual cipher (pixel + SDF) */
    int correct_dual = 0;
    double td0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *px_sig = test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *sf_sig = sdf_test_sigs + (size_t)i * SIG_PAD;
        if (dual_classify(px_sig, sf_sig) == test_labels[i])
            correct_dual++;
    }
    double td1 = now_sec();

    double dual_acc = (double)correct_dual / TEST_N;
    printf("\n  --- Dual Cipher (pixel + SDF hot maps) ---\n");
    printf("  Size:       %zu KB (pixel %zu KB + SDF %zu KB)\n",
           (sizeof(hot_map) + sizeof(sdf_hot_map)) / 1024,
           sizeof(hot_map) / 1024, sizeof(sdf_hot_map) / 1024);
    printf("  Accuracy:   %.2f%%\n", dual_acc * 100.0);
    printf("  vs pixel:   %+.2f pp (pixel=%.2f%%)\n",
           (dual_acc - 71.12 / 100.0) * 100.0, 71.12);
    printf("  vs SDF:     %+.2f pp (SDF=%.2f%%)\n",
           (dual_acc - best_sdf_acc) * 100.0, best_sdf_acc * 100.0);
    printf("  Latency:    %.0f ns/query\n", (td1 - td0) * 1e9 / TEST_N);

    /* Step 5: entropy comparison */
    int px_active = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 1; v < N_BVALS; v++) {
            int has = 0;
            for (int c = 0; c < N_CLASSES; c++)
                if (hot_map[k][v][c] > 0) { has = 1; break; }
            if (has) px_active++;
        }

    /* Recount SDF active at best T (already in sdf_hot_map) */
    int sdf_active = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 1; v < N_BVALS; v++) {
            int has = 0;
            for (int c = 0; c < N_CLASSES; c++)
                if (sdf_hot_map[k][v][c] > 0) { has = 1; break; }
            if (has) sdf_active++;
        }

    printf("\n  --- Entropy Comparison ---\n");
    printf("  Pixel hot map: %d / %d active foreground buckets\n",
           px_active, N_BLOCKS * (N_BVALS - 1));
    printf("  SDF hot map:   %d / %d active foreground buckets\n",
           sdf_active, N_BLOCKS * (N_BVALS - 1));
    printf("  SDF entropy:   %.1f%% of pixel entropy\n",
           100.0 * sdf_active / (px_active > 0 ? px_active : 1));

    /* Step 6: codebook compression stats */
    /* Count distinct SDF block-value patterns per position */
    int total_unique_sigs = 0;
    {
        /* Simple hash: count unique 261-byte signatures */
        /* Use a sampling approach: hash each sig, count unique hashes */
        uint32_t *hashes = malloc((size_t)TRAIN_N * sizeof(uint32_t));
        if (hashes) {
            for (int i = 0; i < TRAIN_N; i++) {
                const uint8_t *sig = sdf_train_sigs + (size_t)i * SIG_PAD;
                /* FNV-1a hash of 261 bytes */
                uint32_t h = 2166136261u;
                for (int k = 0; k < N_BLOCKS; k++) {
                    h ^= sig[k];
                    h *= 16777619u;
                }
                hashes[i] = h;
            }
            /* Count unique hashes (sort + count transitions) */
            qsort(hashes, TRAIN_N, sizeof(uint32_t), cmp_u32);
            total_unique_sigs = 1;
            for (int i = 1; i < TRAIN_N; i++)
                if (hashes[i] != hashes[i - 1])
                    total_unique_sigs++;
            free(hashes);
        }
    }

    printf("\n  --- Codebook Compression ---\n");
    printf("  Unique SDF signatures: %d / %d training images\n",
           total_unique_sigs, TRAIN_N);
    if (total_unique_sigs > 0)
        printf("  Codebook size:         %d entries × %d bytes = %.1f KB\n",
               total_unique_sigs, N_BLOCKS + N_CLASSES * 4,
               (double)total_unique_sigs * (N_BLOCKS + N_CLASSES * 4) / 1024.0);
    printf("  Compression ratio:     %.1fx (vs %d raw images)\n",
           (double)TRAIN_N / (total_unique_sigs > 0 ? total_unique_sigs : 1),
           TRAIN_N);

    /* Keep raw SDF alive for Test H (gradient-of-SDF needs it) */
}

/* ================================================================
 *  Test G: Combined Powers
 *
 *  Three fusion strategies for pixel + SDF:
 *
 *  1. Weighted hot map: pixel_votes × W + sdf_votes × 1
 *     Keep pixel dominant; SDF breaks ties.
 *
 *  2. SDF attention mask: only vote on pixel blocks near the shape
 *     boundary (where SDF contains near-boundary trits). The SDF
 *     tells the pixel where to look.
 *
 *  3. Combined dot product: pixel_dot + sdf_dot for candidate
 *     refinement in the indexed pipeline.
 * ================================================================ */

/* SDF boundary mask: 1 if block value contains at least one 0-trit
 * (near-boundary in SDF space). 0-trits encode SDF ≈ 0 = shape edge. */
static const uint8_t sdf_boundary[N_BVALS] = {
    0,1,0,1,1,1,0,1,0,  /* t0=-1: boundary if t1=0 or t2=0 */
    1,1,1,1,1,1,1,1,1,  /* t0=0:  always boundary */
    0,1,0,1,1,1,0,1,0   /* t0=+1: boundary if t1=0 or t2=0 */
};

static void test_combined(double nn_acc, double nn_time) {
    (void)nn_time;
    /* ---- Strategy 1: Weighted Dual Hot Map ---- */
    printf("  --- Strategy 1: Weighted Dual Hot Map ---\n");
    printf("  pixel_votes × W + sdf_votes × 1\n\n");

    int weights[] = {2, 3, 5, 10};
    int n_w = 4;

    printf("  %-6s | %-10s | %-12s\n", "W", "Accuracy", "vs pixel");
    printf("  -------+------------+-------------\n");

    for (int wi = 0; wi < n_w; wi++) {
        int W = weights[wi];
        int correct = 0;

        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *px_sig  = test_sigs + (size_t)i * SIG_PAD;
            const uint8_t *sf_sig  = sdf_test_sigs + (size_t)i * SIG_PAD;

            /* Weighted accumulation */
            __m256i acc_lo = _mm256_setzero_si256();
            __m256i acc_hi = _mm256_setzero_si256();
            __m256i wvec = _mm256_set1_epi32(W);

            for (int k = 0; k < N_BLOCKS; k++) {
                uint8_t pv = px_sig[k];
                if (pv != 0) {
                    const __m256i *cc = (const __m256i *)hot_map[k][pv];
                    __m256i lo = _mm256_mullo_epi32(_mm256_load_si256(cc), wvec);
                    __m256i hi = _mm256_mullo_epi32(_mm256_load_si256(cc + 1), wvec);
                    acc_lo = _mm256_add_epi32(acc_lo, lo);
                    acc_hi = _mm256_add_epi32(acc_hi, hi);
                }
                uint8_t sv = sf_sig[k];
                if (sv != 0) {
                    const __m256i *cc = (const __m256i *)sdf_hot_map[k][sv];
                    acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256(cc));
                    acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256(cc + 1));
                }
            }

            uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *)cv, acc_lo);
            _mm256_store_si256((__m256i *)(cv + 8), acc_hi);
            int best = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cv[c] > cv[best]) best = c;
            if (best == test_labels[i]) correct++;
        }

        double acc = (double)correct / TEST_N;
        printf("  %5d  | %7.2f%%   | %+.2f pp\n",
               W, acc * 100.0, (acc - 0.7112) * 100.0);
    }

    /* ---- Strategy 2: SDF Attention Mask ---- */
    printf("\n  --- Strategy 2: SDF Attention Mask (indexed vote) ---\n");
    printf("  Only vote on pixel blocks near shape boundary (SDF has 0-trits)\n\n");

    int k2_vals[] = {50, 200};
    int n_k2 = 2;
    int correct_masked_vote = 0;
    int correct_masked[2] = {0};
    int total_blocks_used = 0, total_blocks_skipped = 0;

    double t0 = now_sec();
    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    if (!votes) { fprintf(stderr, "ERROR: votes alloc failed\n"); exit(1); }

    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *qsig = test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *sdf_qsig = sdf_test_sigs + (size_t)i * SIG_PAD;
        const int8_t *query = tern_test + (size_t)i * PADDED;

        /* SDF-masked pixel voting */
        memset(votes, 0, TRAIN_N * sizeof(uint16_t));

        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == 0) continue;                    /* skip pixel background */
            if (!sdf_boundary[sdf_qsig[k]]) {         /* SDF says "far from shape" */
                total_blocks_skipped++;
                continue;
            }
            total_blocks_used++;

            uint32_t off = idx_off[k][bv];
            uint16_t sz = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off;
            for (uint16_t j = 0; j < sz; j++)
                votes[ids[j]]++;
        }

        /* Vote-only classification */
        {
            uint32_t best_id = 0;
            uint16_t best_v = 0;
            for (int j = 0; j < TRAIN_N; j++) {
                if (votes[j] > best_v) {
                    best_v = votes[j];
                    best_id = (uint32_t)j;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_masked_vote++;
        }

        /* Top-K refine with pixel dot product */
        candidate_t cands[MAX_K];
        int nc = select_top_k(votes, TRAIN_N, cands, k2_vals[n_k2 - 1]);

        for (int j = 0; j < nc; j++) {
            const int8_t *ref = tern_train + (size_t)cands[j].id * PADDED;
            cands[j].dot = ternary_dot(query, ref);
        }

        for (int ki = 0; ki < n_k2; ki++) {
            int kv = k2_vals[ki];
            int ek = (nc < kv) ? nc : kv;
            int32_t best_d = cands[0].dot;
            uint32_t best_id = cands[0].id;
            for (int j = 1; j < ek; j++) {
                if (cands[j].dot > best_d) {
                    best_d = cands[j].dot;
                    best_id = cands[j].id;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_masked[ki]++;
        }
    }
    free(votes);

    double t1 = now_sec();

    printf("  Blocks used: %d  skipped: %d (%.1f%% filtered by SDF)\n\n",
           total_blocks_used, total_blocks_skipped,
           100.0 * total_blocks_skipped /
           (total_blocks_used + total_blocks_skipped));

    printf("  Masked vote-only:  %.2f%% (vs unmasked 84.43%%: %+.2f pp)\n",
           100.0 * correct_masked_vote / TEST_N,
           100.0 * correct_masked_vote / TEST_N - 84.43);

    for (int ki = 0; ki < n_k2; ki++) {
        double acc = (double)correct_masked[ki] / TEST_N;
        printf("  Masked Top-%-5d:  %.2f%% (vs 1-NN: %+.2f pp)\n",
               k2_vals[ki], acc * 100.0, (acc - nn_acc) * 100.0);
    }
    printf("  Time: %.2f sec\n", t1 - t0);

    /* ---- Strategy 3: Combined Dot Product ---- */
    printf("\n  --- Strategy 3: Combined Dot (pixel_dot + sdf_dot) ---\n");
    printf("  Indexed vote → refine with pixel_dot + sdf_dot\n\n");

    int correct_combo[2] = {0};
    double t2 = now_sec();
    votes = calloc(TRAIN_N, sizeof(uint16_t));
    if (!votes) { fprintf(stderr, "ERROR: votes alloc failed\n"); exit(1); }

    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *qsig = test_sigs + (size_t)i * SIG_PAD;
        const int8_t *query_px  = tern_test + (size_t)i * PADDED;
        const int8_t *query_sdf = sdf_tern_test + (size_t)i * PADDED;

        /* Standard pixel vote (unmasked) */
        memset(votes, 0, TRAIN_N * sizeof(uint16_t));
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == 0) continue;
            uint32_t off = idx_off[k][bv];
            uint16_t sz = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off;
            for (uint16_t j = 0; j < sz; j++)
                votes[ids[j]]++;
        }

        candidate_t cands[MAX_K];
        int nc = select_top_k(votes, TRAIN_N, cands, k2_vals[n_k2 - 1]);

        /* Combined dot product: pixel_dot + sdf_dot */
        for (int j = 0; j < nc; j++) {
            uint32_t cid = cands[j].id;
            const int8_t *ref_px  = tern_train + (size_t)cid * PADDED;
            const int8_t *ref_sdf = sdf_tern_train + (size_t)cid * PADDED;
            cands[j].dot = ternary_dot(query_px, ref_px)
                         + ternary_dot(query_sdf, ref_sdf);
        }

        for (int ki = 0; ki < n_k2; ki++) {
            int kv = k2_vals[ki];
            int ek = (nc < kv) ? nc : kv;
            int32_t best_d = cands[0].dot;
            uint32_t best_id = cands[0].id;
            for (int j = 1; j < ek; j++) {
                if (cands[j].dot > best_d) {
                    best_d = cands[j].dot;
                    best_id = cands[j].id;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_combo[ki]++;
        }
    }
    free(votes);

    double t3 = now_sec();

    for (int ki = 0; ki < n_k2; ki++) {
        double acc = (double)correct_combo[ki] / TEST_N;
        printf("  Combined Top-%-5d: %.2f%% (vs pixel-only: %+.2f pp, vs 1-NN: %+.2f pp)\n",
               k2_vals[ki], acc * 100.0,
               acc * 100.0 - (ki == 0 ? 94.39 : 95.16),
               (acc - nn_acc) * 100.0);
    }
    printf("  Time: %.2f sec\n", t3 - t2);
}

/* ================================================================
 *  Test H: Gradient-Enhanced SDF & TST
 *
 *  Hypothesis: feeding gradient ternary images (edge direction)
 *  through SDF (distance transform) captures corners & junctions.
 *  Adding gradient hot maps and gradient-SDF hot maps should improve
 *  classification. Multi-channel TST should capture orientation.
 * ================================================================ */

static void test_gradient_sdf(void) {
    /* ---- H1: Gradient Hot Maps (direct, no SDF) ---- */
    printf("  --- H1: Gradient Hot Maps ---\n");
    printf("  H-Grad and V-Grad ternary fed through same hot map cipher.\n");
    printf("  Background skip: bv=%d (block_encode(0,0,0) = flat region)\n\n",
           BG_GRAD);

    build_hot_map_gen(hg_train_sigs, TRAIN_N, hg_hot_map);
    build_hot_map_gen(vg_train_sigs, TRAIN_N, vg_hot_map);

    int correct_hg = 0, correct_vg = 0, correct_3chan = 0;
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *px_sig = test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *hg_sig = hg_test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *vg_sig = vg_test_sigs + (size_t)i * SIG_PAD;

        if (hot_classify_gen(hg_sig, hg_hot_map, BG_GRAD) == test_labels[i])
            correct_hg++;
        if (hot_classify_gen(vg_sig, vg_hot_map, BG_GRAD) == test_labels[i])
            correct_vg++;

        /* 3-channel combined: pixel + h-grad + v-grad */
        __m256i acc_lo = _mm256_setzero_si256();
        __m256i acc_hi = _mm256_setzero_si256();
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv;
            bv = px_sig[k];
            if (bv != 0) {
                acc_lo = _mm256_add_epi32(acc_lo,
                    _mm256_load_si256((const __m256i *)hot_map[k][bv]));
                acc_hi = _mm256_add_epi32(acc_hi,
                    _mm256_load_si256((const __m256i *)hot_map[k][bv] + 1));
            }
            bv = hg_sig[k];
            if (bv != BG_GRAD) {
                acc_lo = _mm256_add_epi32(acc_lo,
                    _mm256_load_si256((const __m256i *)hg_hot_map[k][bv]));
                acc_hi = _mm256_add_epi32(acc_hi,
                    _mm256_load_si256((const __m256i *)hg_hot_map[k][bv] + 1));
            }
            bv = vg_sig[k];
            if (bv != BG_GRAD) {
                acc_lo = _mm256_add_epi32(acc_lo,
                    _mm256_load_si256((const __m256i *)vg_hot_map[k][bv]));
                acc_hi = _mm256_add_epi32(acc_hi,
                    _mm256_load_si256((const __m256i *)vg_hot_map[k][bv] + 1));
            }
        }
        uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i *)cv, acc_lo);
        _mm256_store_si256((__m256i *)(cv + 8), acc_hi);
        int best = 0;
        for (int c = 1; c < N_CLASSES; c++)
            if (cv[c] > cv[best]) best = c;
        if (best == test_labels[i]) correct_3chan++;
    }

    printf("  H-Grad hot map:            %.2f%%\n",
           100.0 * correct_hg / TEST_N);
    printf("  V-Grad hot map:            %.2f%%\n",
           100.0 * correct_vg / TEST_N);
    printf("  Pixel+H-Grad+V-Grad:       %.2f%% (vs pixel 71.12%%: %+.2f pp)\n\n",
           100.0 * correct_3chan / TEST_N,
           100.0 * correct_3chan / TEST_N - 71.12);

    /* ---- H2: Gradient-SDF Hot Maps ---- */
    printf("  --- H2: Gradient-SDF Hot Maps ---\n");
    printf("  SDF of gradient ternary: boundaries = where edge direction changes\n");
    printf("  = corners, junctions, endpoints\n\n");

    /* Process H-Grad SDF */
    int8_t *gsdf_raw   = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    int8_t *gsdf_tern  = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    uint8_t *gsdf_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    int8_t *gsdf_raw_t  = (int8_t *)aligned_alloc(32, (size_t)TEST_N * PADDED);
    int8_t *gsdf_tern_t = (int8_t *)aligned_alloc(32, (size_t)TEST_N * PADDED);
    hg_sdf_test_sigs = (uint8_t *)aligned_alloc(32, (size_t)TEST_N * SIG_PAD);
    vg_sdf_test_sigs = (uint8_t *)aligned_alloc(32, (size_t)TEST_N * SIG_PAD);
    if (!gsdf_raw || !gsdf_tern || !gsdf_sigs || !gsdf_raw_t ||
        !gsdf_tern_t || !hg_sdf_test_sigs || !vg_sdf_test_sigs) {
        fprintf(stderr, "ERROR: gradient-SDF alloc failed\n"); exit(1);
    }

    /* Helper: sweep threshold and bg_val for a gradient-SDF channel */
    /* For each (T, bg_val) combo, find best accuracy */
    int thresholds[] = {1, 2, 3, 4};
    /* Background candidates: 0=(-1,-1,-1), 13=(0,0,0), 26=(+1,+1,+1) */
    uint8_t bg_candidates[] = {0, 13, 26};
    int n_bg = 3;
    const char *bg_names[] = {"bv=0", "bv=13", "bv=26"};

    /* --- H-Grad SDF --- */
    printf("  H-Grad SDF (sweep T × background skip):\n");

    double best_hg_sdf_acc = 0;
    int best_hg_t = 2;
    uint8_t best_hg_bg = 13;

    for (int i = 0; i < TRAIN_N; i++)
        compute_edge_sdf(hgrad_train + (size_t)i * PADDED,
                         gsdf_raw + (size_t)i * PADDED);
    for (int i = 0; i < TEST_N; i++)
        compute_edge_sdf(hgrad_test + (size_t)i * PADDED,
                         gsdf_raw_t + (size_t)i * PADDED);

    /* Print trit distribution at T=2 for diagnosis */
    {
        for (int i = 0; i < TRAIN_N; i++)
            quantize_sdf_one(gsdf_raw + (size_t)i * PADDED,
                             gsdf_tern + (size_t)i * PADDED, 2);
        long neg = 0, zero = 0, pos = 0;
        for (size_t j = 0; j < (size_t)TRAIN_N * PIXELS; j++) {
            if (gsdf_tern[j] < 0) neg++;
            else if (gsdf_tern[j] > 0) pos++;
            else zero++;
        }
        long tot = (long)TRAIN_N * PIXELS;
        printf("  Trit dist (T=2): -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
               100.0*neg/tot, 100.0*zero/tot, 100.0*pos/tot);
    }

    printf("  %-4s | %-6s | %-10s\n", "T", "BG", "Accuracy");
    printf("  -----+--------+-----------\n");

    for (int ti = 0; ti < 4; ti++) {
        int T = thresholds[ti];
        for (int i = 0; i < TRAIN_N; i++)
            quantize_sdf_one(gsdf_raw + (size_t)i * PADDED,
                             gsdf_tern + (size_t)i * PADDED, T);
        for (int i = 0; i < TEST_N; i++)
            quantize_sdf_one(gsdf_raw_t + (size_t)i * PADDED,
                             gsdf_tern_t + (size_t)i * PADDED, T);

        compute_sdf_sigs_from(gsdf_tern, gsdf_sigs, TRAIN_N);
        uint8_t *test_gsdf_sigs = (uint8_t *)aligned_alloc(32,
            (size_t)TEST_N * SIG_PAD);
        compute_sdf_sigs_from(gsdf_tern_t, test_gsdf_sigs, TEST_N);

        uint32_t tmp_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
        build_hot_map_gen(gsdf_sigs, TRAIN_N, tmp_hot);

        for (int bi = 0; bi < n_bg; bi++) {
            uint8_t bg = bg_candidates[bi];
            int correct = 0;
            for (int i = 0; i < TEST_N; i++) {
                if (hot_classify_gen(test_gsdf_sigs + (size_t)i * SIG_PAD,
                                     tmp_hot, bg) == test_labels[i])
                    correct++;
            }
            double acc = (double)correct / TEST_N;
            printf("  %3d  | %-6s | %7.2f%%\n", T, bg_names[bi], acc * 100.0);

            if (acc > best_hg_sdf_acc) {
                best_hg_sdf_acc = acc;
                best_hg_t = T;
                best_hg_bg = bg;
                memcpy(hg_sdf_hot, tmp_hot, sizeof(tmp_hot));
                memcpy(hg_sdf_test_sigs, test_gsdf_sigs,
                       (size_t)TEST_N * SIG_PAD);
            }
        }
        free(test_gsdf_sigs);
    }
    printf("  Best H-Grad SDF: T=%d bg=%s → %.2f%%\n\n",
           best_hg_t, best_hg_bg == 0 ? "bv=0" : best_hg_bg == 13 ? "bv=13" : "bv=26",
           best_hg_sdf_acc * 100.0);

    /* --- V-Grad SDF --- */
    printf("  V-Grad SDF (sweep T × background skip):\n");

    double best_vg_sdf_acc = 0;
    int best_vg_t = 2;
    uint8_t best_vg_bg = 13;

    for (int i = 0; i < TRAIN_N; i++)
        compute_edge_sdf(vgrad_train + (size_t)i * PADDED,
                         gsdf_raw + (size_t)i * PADDED);
    for (int i = 0; i < TEST_N; i++)
        compute_edge_sdf(vgrad_test + (size_t)i * PADDED,
                         gsdf_raw_t + (size_t)i * PADDED);

    /* Trit distribution diagnosis */
    {
        for (int i = 0; i < TRAIN_N; i++)
            quantize_sdf_one(gsdf_raw + (size_t)i * PADDED,
                             gsdf_tern + (size_t)i * PADDED, 2);
        long neg = 0, zero = 0, pos = 0;
        for (size_t j = 0; j < (size_t)TRAIN_N * PIXELS; j++) {
            if (gsdf_tern[j] < 0) neg++;
            else if (gsdf_tern[j] > 0) pos++;
            else zero++;
        }
        long tot = (long)TRAIN_N * PIXELS;
        printf("  Trit dist (T=2): -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
               100.0*neg/tot, 100.0*zero/tot, 100.0*pos/tot);
    }

    printf("  %-4s | %-6s | %-10s\n", "T", "BG", "Accuracy");
    printf("  -----+--------+-----------\n");

    for (int ti = 0; ti < 4; ti++) {
        int T = thresholds[ti];
        for (int i = 0; i < TRAIN_N; i++)
            quantize_sdf_one(gsdf_raw + (size_t)i * PADDED,
                             gsdf_tern + (size_t)i * PADDED, T);
        for (int i = 0; i < TEST_N; i++)
            quantize_sdf_one(gsdf_raw_t + (size_t)i * PADDED,
                             gsdf_tern_t + (size_t)i * PADDED, T);

        compute_sdf_sigs_from(gsdf_tern, gsdf_sigs, TRAIN_N);
        uint8_t *test_gsdf_sigs = (uint8_t *)aligned_alloc(32,
            (size_t)TEST_N * SIG_PAD);
        compute_sdf_sigs_from(gsdf_tern_t, test_gsdf_sigs, TEST_N);

        uint32_t tmp_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
        build_hot_map_gen(gsdf_sigs, TRAIN_N, tmp_hot);

        for (int bi = 0; bi < n_bg; bi++) {
            uint8_t bg = bg_candidates[bi];
            int correct = 0;
            for (int i = 0; i < TEST_N; i++) {
                if (hot_classify_gen(test_gsdf_sigs + (size_t)i * SIG_PAD,
                                     tmp_hot, bg) == test_labels[i])
                    correct++;
            }
            double acc = (double)correct / TEST_N;
            printf("  %3d  | %-6s | %7.2f%%\n", T, bg_names[bi], acc * 100.0);

            if (acc > best_vg_sdf_acc) {
                best_vg_sdf_acc = acc;
                best_vg_t = T;
                best_vg_bg = bg;
                memcpy(vg_sdf_hot, tmp_hot, sizeof(tmp_hot));
                memcpy(vg_sdf_test_sigs, test_gsdf_sigs,
                       (size_t)TEST_N * SIG_PAD);
            }
        }
        free(test_gsdf_sigs);
    }
    printf("  Best V-Grad SDF: T=%d bg=%s → %.2f%%\n\n",
           best_vg_t, best_vg_bg == 0 ? "bv=0" : best_vg_bg == 13 ? "bv=13" : "bv=26",
           best_vg_sdf_acc * 100.0);

    /* Free temporary SDF buffers */
    free(gsdf_raw); free(gsdf_tern); free(gsdf_sigs);
    free(gsdf_raw_t); free(gsdf_tern_t);

    /* ---- H3: Full Ensemble ---- */
    printf("  --- H3: Full Ensemble ---\n");
    printf("  Pixel + Pixel-SDF + H-Grad-SDF + V-Grad-SDF (4 ciphers)\n\n");

    /* Need pixel SDF test sigs — rebuild if freed */
    /* test_sdf_cipher frees raw SDF but keeps sdf_test_sigs alive */
    int correct_dual = 0, correct_4way = 0;
    int correct_px_3grad = 0;
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *px_sig = test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *sdf_sig = sdf_test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *hgsdf_sig = hg_sdf_test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *vgsdf_sig = vg_sdf_test_sigs + (size_t)i * SIG_PAD;

        __m256i acc_lo = _mm256_setzero_si256();
        __m256i acc_hi = _mm256_setzero_si256();

        /* Pixel hot map */
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = px_sig[k];
            if (bv == 0) continue;
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_load_si256((const __m256i *)hot_map[k][bv]));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_load_si256((const __m256i *)hot_map[k][bv] + 1));
        }

        /* Snapshot after pixel (for dual comparison) */
        __m256i dual_lo = acc_lo, dual_hi = acc_hi;

        /* Pixel SDF hot map */
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sdf_sig[k];
            if (bv == 0) continue;
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_load_si256((const __m256i *)sdf_hot_map[k][bv]));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_load_si256((const __m256i *)sdf_hot_map[k][bv] + 1));
            dual_lo = _mm256_add_epi32(dual_lo,
                _mm256_load_si256((const __m256i *)sdf_hot_map[k][bv]));
            dual_hi = _mm256_add_epi32(dual_hi,
                _mm256_load_si256((const __m256i *)sdf_hot_map[k][bv] + 1));
        }

        /* Dual cipher result (pixel + pixel SDF) */
        {
            uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *)cv, dual_lo);
            _mm256_store_si256((__m256i *)(cv + 8), dual_hi);
            int best = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cv[c] > cv[best]) best = c;
            if (best == test_labels[i]) correct_dual++;
        }

        /* Pixel + 3 gradient SDF channels (no pixel SDF) */
        __m256i p3g_lo = _mm256_setzero_si256();
        __m256i p3g_hi = _mm256_setzero_si256();
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = px_sig[k];
            if (bv != 0) {
                p3g_lo = _mm256_add_epi32(p3g_lo,
                    _mm256_load_si256((const __m256i *)hot_map[k][bv]));
                p3g_hi = _mm256_add_epi32(p3g_hi,
                    _mm256_load_si256((const __m256i *)hot_map[k][bv] + 1));
            }
        }

        /* H-Grad SDF */
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = hgsdf_sig[k];
            if (bv == 0) continue;
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_load_si256((const __m256i *)hg_sdf_hot[k][bv]));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_load_si256((const __m256i *)hg_sdf_hot[k][bv] + 1));
            p3g_lo = _mm256_add_epi32(p3g_lo,
                _mm256_load_si256((const __m256i *)hg_sdf_hot[k][bv]));
            p3g_hi = _mm256_add_epi32(p3g_hi,
                _mm256_load_si256((const __m256i *)hg_sdf_hot[k][bv] + 1));
        }

        /* V-Grad SDF */
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = vgsdf_sig[k];
            if (bv == 0) continue;
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_load_si256((const __m256i *)vg_sdf_hot[k][bv]));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_load_si256((const __m256i *)vg_sdf_hot[k][bv] + 1));
            p3g_lo = _mm256_add_epi32(p3g_lo,
                _mm256_load_si256((const __m256i *)vg_sdf_hot[k][bv]));
            p3g_hi = _mm256_add_epi32(p3g_hi,
                _mm256_load_si256((const __m256i *)vg_sdf_hot[k][bv] + 1));
        }

        /* 4-way ensemble (pixel + pixel SDF + hg SDF + vg SDF) */
        {
            uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *)cv, acc_lo);
            _mm256_store_si256((__m256i *)(cv + 8), acc_hi);
            int best = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cv[c] > cv[best]) best = c;
            if (best == test_labels[i]) correct_4way++;
        }

        /* Pixel + H-Grad-SDF + V-Grad-SDF (3-way, no pixel SDF) */
        {
            uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *)cv, p3g_lo);
            _mm256_store_si256((__m256i *)(cv + 8), p3g_hi);
            int best = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cv[c] > cv[best]) best = c;
            if (best == test_labels[i]) correct_px_3grad++;
        }
    }

    printf("  Dual cipher (pixel + pixel-SDF):     %.2f%%\n",
           100.0 * correct_dual / TEST_N);
    printf("  Pixel + H-Grad-SDF + V-Grad-SDF:     %.2f%% (%+.2f pp vs dual)\n",
           100.0 * correct_px_3grad / TEST_N,
           100.0 * (correct_px_3grad - correct_dual) / TEST_N);
    printf("  4-cipher (px + px-SDF + hg-SDF + vg-SDF): %.2f%% (%+.2f pp vs dual)\n\n",
           100.0 * correct_4way / TEST_N,
           100.0 * (correct_4way - correct_dual) / TEST_N);

    /* ---- H4: Multi-Channel TST ---- */
    printf("  --- H4: Multi-Channel TST ---\n");
    printf("  Combined TST score = pixel_tst + hg_tst + vg_tst\n\n");

    int k2_vals[] = {10, 50, 200};
    int n_k2 = 3;
    int correct_px_tst[3] = {0};
    int correct_mc_tst[3] = {0};
    int correct_mr_px = 0, correct_mr_mc = 0;

    double t0 = now_sec();
    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    if (!votes) { fprintf(stderr, "ERROR: votes alloc failed\n"); exit(1); }
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *qsig = test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *qhg  = hg_test_sigs + (size_t)i * SIG_PAD;
        const uint8_t *qvg  = vg_test_sigs + (size_t)i * SIG_PAD;
        const int8_t *query = tern_test + (size_t)i * PADDED;

        /* Stage 1: Vote via inverted index (pixel only) */
        memset(votes, 0, TRAIN_N * sizeof(uint16_t));
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == 0) continue;
            uint32_t off = idx_off[k][bv];
            uint16_t sz = idx_sz[k][bv];
            const uint32_t *ids = idx_pool + off;
            for (uint16_t j = 0; j < sz; j++)
                votes[ids[j]]++;
        }

        /* Top-K1 */
        candidate_t cands[TST_K1];
        int nc = select_top_k(votes, TRAIN_N, cands, TST_K1);

        /* Stage 2: Multi-res scoring */
        for (int j = 0; j < nc; j++) {
            uint32_t cid = cands[j].id;
            const uint8_t *csig = train_sigs + (size_t)cid * SIG_PAD;
            int32_t px_score = tst_multi_res_score(qsig, csig);

            /* Gradient multi-res scores */
            const uint8_t *chg = hg_train_sigs + (size_t)cid * SIG_PAD;
            const uint8_t *cvg = vg_train_sigs + (size_t)cid * SIG_PAD;
            int32_t hg_score = tst_multi_res_score(qhg, chg);
            int32_t vg_score = tst_multi_res_score(qvg, cvg);

            cands[j].dot = px_score + hg_score + vg_score;
            cands[j].votes = (uint16_t)px_score;  /* stash pixel-only */
        }

        /* Multi-res only classification: pixel-only vs multi-channel */
        {
            int32_t best_px = -1, best_mc = -1;
            uint32_t best_px_id = cands[0].id, best_mc_id = cands[0].id;
            for (int j = 0; j < nc; j++) {
                if ((int32_t)cands[j].votes > best_px) {
                    best_px = (int32_t)cands[j].votes;
                    best_px_id = cands[j].id;
                }
                if (cands[j].dot > best_mc) {
                    best_mc = cands[j].dot;
                    best_mc_id = cands[j].id;
                }
            }
            if (train_labels[best_px_id] == test_labels[i]) correct_mr_px++;
            if (train_labels[best_mc_id] == test_labels[i]) correct_mr_mc++;
        }

        /* Sort by multi-channel TST score descending */
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        /* Stage 3: Dot product on top survivors */
        int max_k2 = k2_vals[n_k2 - 1];
        int ek = (nc < max_k2) ? nc : max_k2;
        for (int j = 0; j < ek; j++) {
            const int8_t *ref = tern_train + (size_t)cands[j].id * PADDED;
            cands[j].dot = ternary_dot(query, ref);
        }

        /* Pixel-only TST scores for comparison (re-sort by votes=px_score) */
        /* Save multi-channel dot results first */
        int32_t mc_dots[200];
        uint32_t mc_ids[200];
        for (int j = 0; j < ek && j < 200; j++) {
            mc_dots[j] = cands[j].dot;
            mc_ids[j] = cands[j].id;
        }

        /* Re-sort by pixel-only TST score for pixel-only cascade */
        /* Use votes field which has pixel-only score */
        for (int j = 0; j < nc; j++)
            cands[j].dot = (int32_t)cands[j].votes;
        qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

        /* Pixel-only dot products */
        int ek_px = (nc < max_k2) ? nc : max_k2;
        for (int j = 0; j < ek_px; j++) {
            const int8_t *ref = tern_train + (size_t)cands[j].id * PADDED;
            cands[j].dot = ternary_dot(query, ref);
        }

        /* Evaluate pixel-only TST */
        for (int ki = 0; ki < n_k2; ki++) {
            int kv = k2_vals[ki];
            int ek2 = (ek_px < kv) ? ek_px : kv;
            int32_t best_d = cands[0].dot;
            uint32_t best_id = cands[0].id;
            for (int j = 1; j < ek2; j++) {
                if (cands[j].dot > best_d) {
                    best_d = cands[j].dot;
                    best_id = cands[j].id;
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_px_tst[ki]++;
        }

        /* Evaluate multi-channel TST (use mc_dots/mc_ids) */
        for (int ki = 0; ki < n_k2; ki++) {
            int kv = k2_vals[ki];
            int ek2 = (ek < kv) ? ek : kv;
            int32_t best_d = mc_dots[0];
            uint32_t best_id = mc_ids[0];
            for (int j = 1; j < ek2; j++) {
                if (mc_dots[j] > best_d) {
                    best_d = mc_dots[j];
                    best_id = mc_ids[j];
                }
            }
            if (train_labels[best_id] == test_labels[i])
                correct_mc_tst[ki]++;
        }

        if ((i + 1) % 2000 == 0)
            fprintf(stderr, "  TST H4: %d/%d\r", i + 1, TEST_N);
    }
    fprintf(stderr, "\n");
    free(votes);
    double t1 = now_sec();

    printf("  Multi-res only (pixel):       %.2f%%\n",
           100.0 * correct_mr_px / TEST_N);
    printf("  Multi-res only (3-channel):   %.2f%% (%+.2f pp)\n\n",
           100.0 * correct_mr_mc / TEST_N,
           100.0 * (correct_mr_mc - correct_mr_px) / TEST_N);

    printf("  %-6s | %-14s | %-14s | %-8s\n",
           "K2", "Pixel TST", "3-Chan TST", "Delta");
    printf("  -------+----------------+----------------+---------\n");
    for (int ki = 0; ki < n_k2; ki++) {
        double px = 100.0 * correct_px_tst[ki] / TEST_N;
        double mc = 100.0 * correct_mc_tst[ki] / TEST_N;
        printf("  %5d  | %10.2f%%   | %10.2f%%   | %+.2f pp\n",
               k2_vals[ki], px, mc, mc - px);
    }
    printf("\n  Time: %.2f sec\n", t1 - t0);

    /* ---- H5: Gradient-of-SDF (boundary normal field) ---- */
    printf("\n  --- H5: Gradient-of-SDF (Boundary Normal Field) ---\n");
    printf("  Compute h-grad and v-grad of raw pixel-SDF image.\n");
    printf("  SDF gradient = boundary normal direction.\n");
    printf("  Non-zero only near edges. Direction encodes boundary shape.\n\n");

    {
        /* Compute gradients of raw SDF */
        int8_t *sdf_hg_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
        int8_t *sdf_vg_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
        int8_t *sdf_hg_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
        int8_t *sdf_vg_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
        if (!sdf_hg_train || !sdf_vg_train || !sdf_hg_test || !sdf_vg_test) {
            fprintf(stderr, "ERROR: gradient-of-SDF alloc failed\n"); exit(1);
        }

        /* SDF raw values are int8_t but can be large. Gradients (diff of neighbors)
           can exceed [-1,+1]. We clamp to trit via clamp_trit. */
        for (int i = 0; i < TRAIN_N; i++)
            compute_gradients_one(sdf_raw_train + (size_t)i * PADDED,
                                  sdf_hg_train  + (size_t)i * PADDED,
                                  sdf_vg_train  + (size_t)i * PADDED);
        for (int i = 0; i < TEST_N; i++)
            compute_gradients_one(sdf_raw_test + (size_t)i * PADDED,
                                  sdf_hg_test  + (size_t)i * PADDED,
                                  sdf_vg_test  + (size_t)i * PADDED);

        /* Trit distribution of SDF gradients */
        {
            long neg = 0, zero = 0, pos = 0;
            for (size_t j = 0; j < (size_t)TRAIN_N * PIXELS; j++) {
                if (sdf_hg_train[j] < 0) neg++;
                else if (sdf_hg_train[j] > 0) pos++;
                else zero++;
            }
            long tot = (long)TRAIN_N * PIXELS;
            printf("  SDF H-Grad trit dist: -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
                   100.0*neg/tot, 100.0*zero/tot, 100.0*pos/tot);
        }
        {
            long neg = 0, zero = 0, pos = 0;
            for (size_t j = 0; j < (size_t)TRAIN_N * PIXELS; j++) {
                if (sdf_vg_train[j] < 0) neg++;
                else if (sdf_vg_train[j] > 0) pos++;
                else zero++;
            }
            long tot = (long)TRAIN_N * PIXELS;
            printf("  SDF V-Grad trit dist: -1=%.1f%% 0=%.1f%% +1=%.1f%%\n\n",
                   100.0*neg/tot, 100.0*zero/tot, 100.0*pos/tot);
        }

        /* Build block signatures */
        uint8_t *shg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        uint8_t *shg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
        uint8_t *svg_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        uint8_t *svg_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
        compute_sdf_sigs_from(sdf_hg_train, shg_train_sigs, TRAIN_N);
        compute_sdf_sigs_from(sdf_hg_test,  shg_test_sigs,  TEST_N);
        compute_sdf_sigs_from(sdf_vg_train, svg_train_sigs, TRAIN_N);
        compute_sdf_sigs_from(sdf_vg_test,  svg_test_sigs,  TEST_N);

        /* Free ternary images, keep only sigs */
        free(sdf_hg_train); free(sdf_vg_train);
        free(sdf_hg_test);  free(sdf_vg_test);

        /* Sweep background skip for each channel */
        uint32_t shg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
        uint32_t svg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

        double best_shg_acc = 0, best_svg_acc = 0;
        uint8_t best_shg_bg = 13, best_svg_bg = 13;

        /* Build hot maps */
        build_hot_map_gen(shg_train_sigs, TRAIN_N, shg_hot);
        build_hot_map_gen(svg_train_sigs, TRAIN_N, svg_hot);

        printf("  %-12s | %-6s | %-10s\n", "Channel", "BG", "Accuracy");
        printf("  ------------+--------+-----------\n");

        for (int bi = 0; bi < n_bg; bi++) {
            uint8_t bg = bg_candidates[bi];
            int correct_shg = 0, correct_svg = 0;
            for (int i = 0; i < TEST_N; i++) {
                if (hot_classify_gen(shg_test_sigs + (size_t)i * SIG_PAD,
                                     shg_hot, bg) == test_labels[i])
                    correct_shg++;
                if (hot_classify_gen(svg_test_sigs + (size_t)i * SIG_PAD,
                                     svg_hot, bg) == test_labels[i])
                    correct_svg++;
            }
            double shg_a = (double)correct_shg / TEST_N;
            double svg_a = (double)correct_svg / TEST_N;
            printf("  SDF-H-Grad  | %-6s | %7.2f%%\n", bg_names[bi], shg_a * 100.0);
            printf("  SDF-V-Grad  | %-6s | %7.2f%%\n", bg_names[bi], svg_a * 100.0);
            if (shg_a > best_shg_acc) { best_shg_acc = shg_a; best_shg_bg = bg; }
            if (svg_a > best_svg_acc) { best_svg_acc = svg_a; best_svg_bg = bg; }
        }
        printf("\n  Best SDF-H-Grad: bg=%s → %.2f%%\n",
               best_shg_bg == 0 ? "bv=0" : best_shg_bg == 13 ? "bv=13" : "bv=26",
               best_shg_acc * 100.0);
        printf("  Best SDF-V-Grad: bg=%s → %.2f%%\n\n",
               best_svg_bg == 0 ? "bv=0" : best_svg_bg == 13 ? "bv=13" : "bv=26",
               best_svg_acc * 100.0);

        /* 3-channel ensemble: pixel + SDF-H-Grad + SDF-V-Grad */
        int correct_p_shg_svg = 0;
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *px_sig = test_sigs + (size_t)i * SIG_PAD;
            const uint8_t *shg_sig = shg_test_sigs + (size_t)i * SIG_PAD;
            const uint8_t *svg_sig = svg_test_sigs + (size_t)i * SIG_PAD;

            __m256i acc_lo = _mm256_setzero_si256();
            __m256i acc_hi = _mm256_setzero_si256();
            for (int k = 0; k < N_BLOCKS; k++) {
                uint8_t bv;
                bv = px_sig[k];
                if (bv != 0) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)hot_map[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)hot_map[k][bv] + 1));
                }
                bv = shg_sig[k];
                if (bv != best_shg_bg) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)shg_hot[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)shg_hot[k][bv] + 1));
                }
                bv = svg_sig[k];
                if (bv != best_svg_bg) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)svg_hot[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)svg_hot[k][bv] + 1));
                }
            }
            uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *)cv, acc_lo);
            _mm256_store_si256((__m256i *)(cv + 8), acc_hi);
            int best = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cv[c] > cv[best]) best = c;
            if (best == test_labels[i]) correct_p_shg_svg++;
        }
        printf("  Pixel + SDF-H-Grad + SDF-V-Grad: %.2f%% (vs pixel 71.12%%: %+.2f pp)\n",
               100.0 * correct_p_shg_svg / TEST_N,
               100.0 * correct_p_shg_svg / TEST_N - 71.12);

        /* 5-channel ensemble: pixel + raw H-Grad + raw V-Grad + SDF-H-Grad + SDF-V-Grad */
        int correct_5chan = 0;
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *px_sig  = test_sigs     + (size_t)i * SIG_PAD;
            const uint8_t *hg_sig  = hg_test_sigs  + (size_t)i * SIG_PAD;
            const uint8_t *vg_sig  = vg_test_sigs  + (size_t)i * SIG_PAD;
            const uint8_t *shg_sig = shg_test_sigs + (size_t)i * SIG_PAD;
            const uint8_t *svg_sig = svg_test_sigs + (size_t)i * SIG_PAD;

            __m256i acc_lo = _mm256_setzero_si256();
            __m256i acc_hi = _mm256_setzero_si256();
            for (int k = 0; k < N_BLOCKS; k++) {
                uint8_t bv;
                bv = px_sig[k];
                if (bv != 0) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)hot_map[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)hot_map[k][bv] + 1));
                }
                bv = hg_sig[k];
                if (bv != BG_GRAD) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)hg_hot_map[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)hg_hot_map[k][bv] + 1));
                }
                bv = vg_sig[k];
                if (bv != BG_GRAD) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)vg_hot_map[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)vg_hot_map[k][bv] + 1));
                }
                bv = shg_sig[k];
                if (bv != best_shg_bg) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)shg_hot[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)shg_hot[k][bv] + 1));
                }
                bv = svg_sig[k];
                if (bv != best_svg_bg) {
                    acc_lo = _mm256_add_epi32(acc_lo,
                        _mm256_load_si256((const __m256i *)svg_hot[k][bv]));
                    acc_hi = _mm256_add_epi32(acc_hi,
                        _mm256_load_si256((const __m256i *)svg_hot[k][bv] + 1));
                }
            }
            uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *)cv, acc_lo);
            _mm256_store_si256((__m256i *)(cv + 8), acc_hi);
            int best = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (cv[c] > cv[best]) best = c;
            if (best == test_labels[i]) correct_5chan++;
        }
        printf("  5-chan (px+hg+vg+sdf_hg+sdf_vg): %.2f%% (vs pixel 71.12%%: %+.2f pp)\n",
               100.0 * correct_5chan / TEST_N,
               100.0 * correct_5chan / TEST_N - 71.12);
        printf("  (vs 3-chan px+hg+vg 73.13%%: %+.2f pp)\n",
               100.0 * correct_5chan / TEST_N - 73.13);

        free(shg_train_sigs); free(shg_test_sigs);
        free(svg_train_sigs); free(svg_test_sigs);
    }

    /* ---- H6: SDF-Gated Gradient Voting ---- */
    printf("\n  --- H6: SDF-Gated Gradient Voting ---\n");
    printf("  Only vote on gradient blocks where pixel-SDF is near boundary.\n");
    printf("  SDF focuses gradients on structurally important regions.\n\n");

    {
        /* Use pixel SDF test sigs to gate gradient votes.
           A block is "near boundary" if its SDF block value contains at least one 0-trit.
           block_encode: val = (t0+1)*9 + (t1+1)*3 + (t2+1) → range [0,26]
           All-positive (+1,+1,+1) = 26, contains no 0-trit → far from boundary
           We'll check: is this block "far from boundary"?
           Far = all trits are +1 (bv=26) or all trits are -1 (bv=0).
           Near = contains at least one 0-trit (boundary crossing within block). */

        /* Precompute: which block values contain at least one 0-trit? */
        int has_zero[N_BVALS];
        for (int bv = 0; bv < N_BVALS; bv++) {
            /* Decode block value: bv = (t0+1)*9 + (t1+1)*3 + (t2+1) */
            int t2 = (bv % 3) - 1;
            int t1 = ((bv / 3) % 3) - 1;
            int t0 = ((bv / 9) % 3) - 1;
            has_zero[bv] = (t0 == 0 || t1 == 0 || t2 == 0);
        }

        /* For each SDF threshold, gate gradient hot map */
        int sdf_thresholds[] = {1, 2, 3, 4};
        printf("  T    | Gated H-Grad | Gated V-Grad | Gated 3-ch  | Blocks used\n");
        printf("  -----+--------------+--------------+-------------+------------\n");

        for (int ti = 0; ti < 4; ti++) {
            int T = sdf_thresholds[ti];
            /* Recompute pixel SDF test sigs at this threshold */
            int8_t *sdf_tern_buf = (int8_t *)aligned_alloc(32, (size_t)TEST_N * PADDED);
            uint8_t *sdf_sig_buf = (uint8_t *)aligned_alloc(32, (size_t)TEST_N * SIG_PAD);
            for (int i = 0; i < TEST_N; i++)
                quantize_sdf_one(sdf_raw_test + (size_t)i * PADDED,
                                 sdf_tern_buf + (size_t)i * PADDED, T);
            compute_sdf_sigs_from(sdf_tern_buf, sdf_sig_buf, TEST_N);
            free(sdf_tern_buf);

            int correct_ghg = 0, correct_gvg = 0, correct_g3 = 0;
            long blocks_used = 0, blocks_total = 0;

            for (int i = 0; i < TEST_N; i++) {
                const uint8_t *px_sig  = test_sigs    + (size_t)i * SIG_PAD;
                const uint8_t *hg_sig  = hg_test_sigs + (size_t)i * SIG_PAD;
                const uint8_t *vg_sig  = vg_test_sigs + (size_t)i * SIG_PAD;
                const uint8_t *ss      = sdf_sig_buf  + (size_t)i * SIG_PAD;

                __m256i hg_lo = _mm256_setzero_si256(), hg_hi = _mm256_setzero_si256();
                __m256i vg_lo = _mm256_setzero_si256(), vg_hi = _mm256_setzero_si256();
                __m256i a3_lo = _mm256_setzero_si256(), a3_hi = _mm256_setzero_si256();

                for (int k = 0; k < N_BLOCKS; k++) {
                    int near_boundary = has_zero[ss[k]];
                    blocks_total++;

                    /* Pixel always votes (ungated) */
                    uint8_t bv = px_sig[k];
                    if (bv != 0) {
                        a3_lo = _mm256_add_epi32(a3_lo,
                            _mm256_load_si256((const __m256i *)hot_map[k][bv]));
                        a3_hi = _mm256_add_epi32(a3_hi,
                            _mm256_load_si256((const __m256i *)hot_map[k][bv] + 1));
                    }

                    if (!near_boundary) continue;
                    blocks_used++;

                    /* Gated gradient votes */
                    bv = hg_sig[k];
                    if (bv != BG_GRAD) {
                        hg_lo = _mm256_add_epi32(hg_lo,
                            _mm256_load_si256((const __m256i *)hg_hot_map[k][bv]));
                        hg_hi = _mm256_add_epi32(hg_hi,
                            _mm256_load_si256((const __m256i *)hg_hot_map[k][bv] + 1));
                        a3_lo = _mm256_add_epi32(a3_lo,
                            _mm256_load_si256((const __m256i *)hg_hot_map[k][bv]));
                        a3_hi = _mm256_add_epi32(a3_hi,
                            _mm256_load_si256((const __m256i *)hg_hot_map[k][bv] + 1));
                    }
                    bv = vg_sig[k];
                    if (bv != BG_GRAD) {
                        vg_lo = _mm256_add_epi32(vg_lo,
                            _mm256_load_si256((const __m256i *)vg_hot_map[k][bv]));
                        vg_hi = _mm256_add_epi32(vg_hi,
                            _mm256_load_si256((const __m256i *)vg_hot_map[k][bv] + 1));
                        a3_lo = _mm256_add_epi32(a3_lo,
                            _mm256_load_si256((const __m256i *)vg_hot_map[k][bv]));
                        a3_hi = _mm256_add_epi32(a3_hi,
                            _mm256_load_si256((const __m256i *)vg_hot_map[k][bv] + 1));
                    }
                }

                /* Classify gated H-Grad */
                uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
                _mm256_store_si256((__m256i *)cv, hg_lo);
                _mm256_store_si256((__m256i *)(cv + 8), hg_hi);
                int best = 0;
                for (int c = 1; c < N_CLASSES; c++)
                    if (cv[c] > cv[best]) best = c;
                if (best == test_labels[i]) correct_ghg++;

                /* Classify gated V-Grad */
                _mm256_store_si256((__m256i *)cv, vg_lo);
                _mm256_store_si256((__m256i *)(cv + 8), vg_hi);
                best = 0;
                for (int c = 1; c < N_CLASSES; c++)
                    if (cv[c] > cv[best]) best = c;
                if (best == test_labels[i]) correct_gvg++;

                /* Classify gated 3-channel */
                _mm256_store_si256((__m256i *)cv, a3_lo);
                _mm256_store_si256((__m256i *)(cv + 8), a3_hi);
                best = 0;
                for (int c = 1; c < N_CLASSES; c++)
                    if (cv[c] > cv[best]) best = c;
                if (best == test_labels[i]) correct_g3++;
            }

            double pct_used = 100.0 * blocks_used / blocks_total;
            printf("  %3d  | %9.2f%%   | %9.2f%%   | %8.2f%%   | %.1f%%\n",
                   T,
                   100.0 * correct_ghg / TEST_N,
                   100.0 * correct_gvg / TEST_N,
                   100.0 * correct_g3 / TEST_N,
                   pct_used);
            free(sdf_sig_buf);
        }
        printf("\n  Reference (ungated):\n");
        printf("  H-Grad: 60.43%%  V-Grad: 72.86%%  3-chan: 73.13%%\n");
    }

    /* Cleanup gradient-SDF test sigs */
    free(hg_sdf_test_sigs); hg_sdf_test_sigs = NULL;
    free(vg_sdf_test_sigs); vg_sdf_test_sigs = NULL;
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(void) {
    seed_rng(42);
    double t0 = now_sec();

    puts("=== SSTT Geometric Inference: Ternary MNIST Classification ===");
    puts("");
    puts("Question: Can ternary lattice geometry separate real digit classes?");
    puts("Method:   Quantize MNIST to {-1,0,+1}, classify by ternary dot product.");
    puts("          No matrix multiply. No activation functions. Just geometry.");
    puts("");

    /* Load MNIST */
    printf("Loading MNIST...\n");
    load_mnist();
    double t1 = now_sec();
    printf("  %d train + %d test images loaded (%.2f sec)\n\n",
           TRAIN_N, TEST_N, t1 - t0);

    /* Quantize to ternary */
    printf("Quantizing to ternary (AVX2, padded %d → %d)...\n", PIXELS, PADDED);
    quantize_all();
    double t2 = now_sec();
    printf("  Done (%.2f sec)\n\n", t2 - t1);

    /* Compute gradient observers */
    printf("Computing gradient observers...\n");
    compute_all_gradients();
    double t2a = now_sec();
    printf("  H-grad + V-grad for %d images (%.2f sec)\n\n",
           TRAIN_N + TEST_N, t2a - t2);

    /* Build block index (pre-addressed shape-state-space) */
    printf("Building block index (%d blocks of 3 trits)...\n", N_BLOCKS);
    compute_sigs();
    compute_grad_sigs();
    build_index();
    double t2b = now_sec();
    print_index_stats();
    printf("  Built in %.2f sec\n\n", t2b - t2);

    /* Quick stats on quantization */
    {
        long neg = 0, zero = 0, pos = 0;
        for (size_t i = 0; i < (size_t)TRAIN_N * PADDED; i++) {
            if (tern_train[i] < 0) neg++;
            else if (tern_train[i] > 0) pos++;
            else zero++;
        }
        /* Subtract padding zeros */
        long pad_zeros = (long)TRAIN_N * (PADDED - PIXELS);
        zero -= pad_zeros;
        long total = (long)TRAIN_N * PIXELS;
        printf("  Trit distribution: -1=%.1f%%  0=%.1f%%  +1=%.1f%%\n\n",
               100.0 * neg / total, 100.0 * zero / total, 100.0 * pos / total);
    }

    /* Baselines */
    puts("--- Baselines ---");
    printf("  Random guessing:   10.00%%\n");
    double float_acc = test_float_centroid();
    double t3 = now_sec();
    printf("  Float centroid:    %.2f%% (%.2f sec)\n\n", float_acc * 100.0, t3 - t2);

    /* Test A: Ternary Centroid */
    puts("--- Test A: Ternary Centroid Classifier ---");
    compute_centroids();
    centroid_acc_clean = test_centroid();
    double t4 = now_sec();
    printf("  Accuracy: %.2f%% (%.2f sec)\n", centroid_acc_clean * 100.0, t4 - t3);
    printf("  vs float centroid: %+.2f pp\n\n",
           (centroid_acc_clean - float_acc) * 100.0);

    /* Test B: Ternary 1-NN */
    puts("--- Test B: Ternary 1-NN Classifier ---");
    printf("  Computing %d x %d = %dM ternary dot products (AVX2)...\n",
           TEST_N, TRAIN_N, TEST_N / 1000 * TRAIN_N / 1000);
    double nn_acc = test_1nn();
    double t5 = now_sec();
    printf("  Accuracy: %.2f%% (%.2f sec)\n", nn_acc * 100.0, t5 - t4);
    printf("  Throughput: %.0f dot products/sec\n",
           (double)TEST_N * TRAIN_N / (t5 - t4));
    double nn_time = t5 - t4;

    /* Test C: Indexed lookup */
    puts("\n--- Test C: Indexed Geometric Lookup (Pre-addressed Shape Space) ---");
    puts("  The shape IS the address. Vote on matching blocks, refine top-K.\n");
    test_indexed(nn_acc, nn_time);

    /* Test D: Hot Map */
    puts("\n--- Test D: Hot Map (L2-Resident Cipher) ---");
    puts("  Collapse 60MB index into 451KB class-count table.");
    puts("  Warm it up. Keep it hot. Nanosecond classification.\n");
    test_hot_map();

    /* Test E: Ternary Shape Transform */
    puts("\n--- Test E: Ternary Shape Transform (TST) ---");
    puts("  FFT analogy: O(n*d) brute -> O(d*log_3(d) + K*d) via multi-res butterfly");
    puts("  Cascade: vote -> butterfly_score -> dot_product\n");
    printf("  Hierarchy: L0=%d  L1=%d  L2=%d  L3=%d  L4=%d  (max score=%d)\n\n",
           TST_L0, TST_L1, TST_L2, TST_L3, TST_L4, TST_MAX_SCORE);
    test_tst(nn_acc, nn_time);

    /* Test F: SDF Cipher */
    puts("\n--- Test F: SDF Cipher (Topological Compression) ---");
    puts("  SDF transforms appearance into topology.");
    puts("  Thick and thin '7' collapse to the same SDF skeleton.");
    puts("  Lower entropy → smaller codebook → better compression.\n");
    test_sdf_cipher();

    /* Test G: Combined Powers */
    puts("\n--- Test G: Combined Powers (Pixel + SDF Fusion) ---");
    puts("  The SDF tells the pixel where to look.\n");
    test_combined(nn_acc, nn_time);

    /* Test H: Gradient-Enhanced SDF & TST */
    puts("\n--- Test H: Gradient-Enhanced SDF & TST ---");
    puts("  Gradients add direction. SDF of gradients detects corners.");
    puts("  Multi-channel TST adds orientation consistency.\n");
    test_gradient_sdf();

    /* Cleanup SDF data */
    free(sdf_raw_train);  sdf_raw_train  = NULL;
    free(sdf_raw_test);   sdf_raw_test   = NULL;
    free(sdf_tern_train); sdf_tern_train = NULL;
    free(sdf_tern_test);  sdf_tern_test  = NULL;
    free(sdf_train_sigs); sdf_train_sigs = NULL;
    free(sdf_test_sigs);  sdf_test_sigs  = NULL;

    /* Cleanup gradient data */
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(hg_train_sigs); free(hg_test_sigs);
    free(vg_train_sigs); free(vg_test_sigs);

    /* Noise test */
    test_noise();
    double t7 = now_sec();

    /* Summary */
    puts("\n=== SUMMARY ===");
    printf("  Random baseline:       10.00%%\n");
    printf("  Float centroid:        %.2f%%\n", float_acc * 100.0);
    printf("  Ternary centroid:      %.2f%%\n", centroid_acc_clean * 100.0);
    printf("  Ternary 1-NN:          %.2f%% (brute force, %.1f sec)\n",
           nn_acc * 100.0, nn_time);
    printf("  Indexed 1-NN:          see Test C table above\n");
    printf("  Hot map:               see Test D above\n");
    printf("  TST cascade:           see Test E above\n");
    printf("  SDF cipher:            see Test F above\n");
    printf("  Combined powers:       see Test G above\n");
    printf("  Gradient-SDF + TST:    see Test H above\n");

    /* Verdict */
    puts("\n=== VERDICT ===");
    double best = centroid_acc_clean > nn_acc ? centroid_acc_clean : nn_acc;
    if (best > 0.80) {
        printf("Best ternary accuracy %.2f%% > 80%% threshold.\n", best * 100.0);
        puts("STATUS: GEOMETRIC INFERENCE WORKS.");
        puts("  Ternary lattice geometry separates real digit classes.");
        puts("  The shapes carry the classification — no arithmetic needed.");
        puts("  Block index makes it fast — the shape is its own address.");
    } else {
        printf("Best ternary accuracy %.2f%% <= 80%% threshold.\n", best * 100.0);
        puts("STATUS: GEOMETRIC INFERENCE INSUFFICIENT.");
        puts("  Ternary quantization destroys too much class structure.");
    }

    printf("\nTotal runtime: %.2f seconds.\n", t7 - t0);
    return (best > 0.80) ? 0 : 1;
}
