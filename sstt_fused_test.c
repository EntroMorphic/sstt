/*
 * sstt_fused_test.c — Test harness for the fused AVX2 assembly kernel.
 *
 * 1. Loads MNIST (or Fashion-MNIST via argv[1]).
 * 2. Builds 3-channel hot maps in C (pixel, h-grad, v-grad).
 * 3. Classifies all test images with both:
 *    a) C reference (hot_classify_3chan)
 *    b) ASM kernel (sstt_fused_classify)
 * 4. Asserts 100% agreement between C and ASM.
 * 5. Benchmarks both.
 *
 * Build: make sstt_fused_test  (after: make mnist)
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sstt_fused.h"

/* ---------- Config ---------- */
#define TRAIN_N     60000
#define TEST_N      10000
#define IMG_W       28
#define IMG_H       28
#define PIXELS      784
#define PADDED      800         /* 25 × 32 for AVX2 alignment */
#define N_CLASSES   10
#define CLS_PAD     16

/* 2D blocks: 3-pixel horizontal strips, 9 per row */
#define BLK_W       3
#define BLKS_PER_ROW 9
#define N_BLOCKS    252         /* 9 × 28 */
#define N_BVALS     27
#define SIG_PAD     256         /* 8 × 32 */

#define BG_PIXEL    0
#define BG_GRAD     13

/* ---------- Data directory ---------- */
static const char *data_dir = "data/";

/* ---------- Timing ---------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Global data ---------- */
static uint8_t *raw_train_img;
static uint8_t *raw_test_img;
static uint8_t *train_labels;
static uint8_t *test_labels;

static int8_t *tern_train;     /* [TRAIN_N * PADDED] */
static int8_t *tern_test;      /* [TEST_N * PADDED] */

/* Hot maps: uint32_t[N_BLOCKS][N_BVALS][CLS_PAD] */
static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* Block signatures for C reference */
static uint8_t *px_train_sigs;
static uint8_t *px_test_sigs;
static uint8_t *hg_train_sigs;
static uint8_t *hg_test_sigs;
static uint8_t *vg_train_sigs;
static uint8_t *vg_test_sigs;

/* Gradient images */
static int8_t *hgrad_train;
static int8_t *hgrad_test;
static int8_t *vgrad_train;
static int8_t *vgrad_test;

/* ================================================================
 *  IDX Loader (from sstt_geom.c)
 * ================================================================ */

static uint8_t *load_idx(const char *path, uint32_t *count,
                         uint32_t *rows_out, uint32_t *cols_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        fprintf(stderr, "  Run 'make mnist' or 'make fashion' first.\n");
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
    if (!data) { fprintf(stderr, "ERROR: malloc failed\n"); fclose(f); exit(1); }
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
    if (n != TRAIN_N || r != 28 || c != 28) { fprintf(stderr, "ERROR: bad train images\n"); exit(1); }
    snprintf(path, sizeof(path), "%strain-labels-idx1-ubyte", dir);
    train_labels = load_idx(path, &n, NULL, NULL);
    snprintf(path, sizeof(path), "%st10k-images-idx3-ubyte", dir);
    raw_test_img = load_idx(path, &n, &r, &c);
    if (n != TEST_N || r != 28 || c != 28) { fprintf(stderr, "ERROR: bad test images\n"); exit(1); }
    snprintf(path, sizeof(path), "%st10k-labels-idx1-ubyte", dir);
    test_labels = load_idx(path, &n, NULL, NULL);
}

/* ================================================================
 *  Quantization, Gradients, Block Signatures, Hot Maps — C reference
 * ================================================================ */

static void quantize_one(const uint8_t *src, int8_t *dst) {
    for (int i = 0; i < PIXELS; i++) {
        if (src[i] > 170)      dst[i] =  1;
        else if (src[i] < 85)  dst[i] = -1;
        else                    dst[i] =  0;
    }
    memset(dst + PIXELS, 0, PADDED - PIXELS);
}

static void quantize_all(void) {
    tern_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    for (int i = 0; i < TRAIN_N; i++)
        quantize_one(raw_train_img + (size_t)i * PIXELS, tern_train + (size_t)i * PADDED);
    for (int i = 0; i < TEST_N; i++)
        quantize_one(raw_test_img + (size_t)i * PIXELS, tern_test + (size_t)i * PADDED);
}

static inline int8_t clamp_trit(int v) {
    if (v > 0) return 1;
    if (v < 0) return -1;
    return 0;
}

static void compute_gradients_one(const int8_t *tern, int8_t *h_grad, int8_t *v_grad) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++) {
            int diff = tern[y * IMG_W + x + 1] - tern[y * IMG_W + x];
            h_grad[y * IMG_W + x] = clamp_trit(diff);
        }
        h_grad[y * IMG_W + IMG_W - 1] = 0;
    }
    memset(h_grad + PIXELS, 0, PADDED - PIXELS);
    for (int y = 0; y < IMG_H - 1; y++)
        for (int x = 0; x < IMG_W; x++) {
            int diff = tern[(y + 1) * IMG_W + x] - tern[y * IMG_W + x];
            v_grad[y * IMG_W + x] = clamp_trit(diff);
        }
    memset(v_grad + (IMG_H - 1) * IMG_W, 0, IMG_W);
    memset(v_grad + PIXELS, 0, PADDED - PIXELS);
}

static void compute_all_gradients(void) {
    hgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    for (int i = 0; i < TRAIN_N; i++)
        compute_gradients_one(tern_train + (size_t)i * PADDED,
                              hgrad_train + (size_t)i * PADDED,
                              vgrad_train + (size_t)i * PADDED);
    for (int i = 0; i < TEST_N; i++)
        compute_gradients_one(tern_test + (size_t)i * PADDED,
                              hgrad_test + (size_t)i * PADDED,
                              vgrad_test + (size_t)i * PADDED);
}

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

/* 2D block signatures: 9 horizontal 3-pixel strips per row, 28 rows = 252 blocks */
static void compute_2d_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int base = y * IMG_W + s * BLK_W;
                sig[y * BLKS_PER_ROW + s] =
                    block_encode(img[base], img[base + 1], img[base + 2]);
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
    compute_2d_sigs(tern_train, px_train_sigs, TRAIN_N);
    compute_2d_sigs(tern_test,  px_test_sigs,  TEST_N);
    compute_2d_sigs(hgrad_train, hg_train_sigs, TRAIN_N);
    compute_2d_sigs(hgrad_test,  hg_test_sigs,  TEST_N);
    compute_2d_sigs(vgrad_train, vg_train_sigs, TRAIN_N);
    compute_2d_sigs(vgrad_test,  vg_test_sigs,  TEST_N);
}

static void build_hot_map(const uint8_t *train_sigs,
                           uint32_t hot[][N_BVALS][CLS_PAD]) {
    memset(hot, 0, sizeof(uint32_t) * N_BLOCKS * N_BVALS * CLS_PAD);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            hot[k][sig[k]][lbl]++;
    }
}

/* ================================================================
 *  C Reference Classifier: 3-channel hot map (pixel + h-grad + v-grad)
 * ================================================================ */

static int c_classify_3chan(const uint8_t *px_sig,
                             const uint8_t *hg_sig,
                             const uint8_t *vg_sig) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv;
        bv = px_sig[k];
        if (bv != BG_PIXEL) {
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_load_si256((const __m256i *)px_hot[k][bv]));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_load_si256((const __m256i *)px_hot[k][bv] + 1));
        }
        bv = hg_sig[k];
        if (bv != BG_GRAD) {
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_load_si256((const __m256i *)hg_hot[k][bv]));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_load_si256((const __m256i *)hg_hot[k][bv] + 1));
        }
        bv = vg_sig[k];
        if (bv != BG_GRAD) {
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_load_si256((const __m256i *)vg_hot[k][bv]));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_load_si256((const __m256i *)vg_hot[k][bv] + 1));
        }
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
 *  Main: Train → Verify → Benchmark
 * ================================================================ */

int main(int argc, char **argv) {
    double t0 = now_sec();

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

    const char *dataset = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Fused Kernel Test (%s) ===\n\n", dataset);

    /* Load data */
    printf("Loading %s...\n", dataset);
    load_data(data_dir);
    double t1 = now_sec();
    printf("  Loaded (%.2f sec)\n\n", t1 - t0);

    /* Quantize + gradients + signatures + hot maps */
    printf("Building hot maps (training)...\n");
    quantize_all();
    compute_all_gradients();
    compute_all_sigs();
    build_hot_map(px_train_sigs, px_hot);
    build_hot_map(hg_train_sigs, hg_hot);
    build_hot_map(vg_train_sigs, vg_hot);
    double t2 = now_sec();
    printf("  3 × %zu KB hot maps built (%.2f sec)\n\n",
           sizeof(px_hot) / 1024, t2 - t1);

    /* ================================================================
     *  Phase 1: C reference classification
     * ================================================================ */
    printf("--- C Reference (3-chan hot map) ---\n");
    int *c_preds = malloc(TEST_N * sizeof(int));
    int c_correct = 0;
    double tc0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        c_preds[i] = c_classify_3chan(
            px_test_sigs + (size_t)i * SIG_PAD,
            hg_test_sigs + (size_t)i * SIG_PAD,
            vg_test_sigs + (size_t)i * SIG_PAD);
        if (c_preds[i] == test_labels[i]) c_correct++;
    }
    double tc1 = now_sec();
    printf("  Accuracy:  %.2f%% (%d/%d)\n", 100.0 * c_correct / TEST_N, c_correct, TEST_N);
    printf("  Time:      %.4f sec (%.0f ns/query)\n\n",
           tc1 - tc0, (tc1 - tc0) * 1e9 / TEST_N);

    /* ================================================================
     *  Phase 2: ASM fused kernel classification
     * ================================================================ */
    printf("--- ASM Fused Kernel (raw pixels → class) ---\n");

    /* Allocate padded pixel buffers (need 4 bytes readable past byte 783) */
    uint8_t *test_pixels_padded = calloc(TEST_N, PIXELS + 4);
    for (int i = 0; i < TEST_N; i++)
        memcpy(test_pixels_padded + (size_t)i * (PIXELS + 4),
               raw_test_img + (size_t)i * PIXELS, PIXELS);

    int *asm_preds = malloc(TEST_N * sizeof(int));
    int asm_correct = 0;
    double ta0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        asm_preds[i] = sstt_fused_classify(
            test_pixels_padded + (size_t)i * (PIXELS + 4),
            (const uint32_t *)px_hot,
            (const uint32_t *)hg_hot,
            (const uint32_t *)vg_hot);
        if (asm_preds[i] == test_labels[i]) asm_correct++;
    }
    double ta1 = now_sec();
    printf("  Accuracy:  %.2f%% (%d/%d)\n", 100.0 * asm_correct / TEST_N, asm_correct, TEST_N);
    printf("  Time:      %.4f sec (%.0f ns/query)\n\n",
           ta1 - ta0, (ta1 - ta0) * 1e9 / TEST_N);

    /* ================================================================
     *  Phase 3: Verification — assert 100% agreement
     * ================================================================ */
    printf("--- Verification ---\n");
    int mismatches = 0;
    for (int i = 0; i < TEST_N; i++) {
        if (c_preds[i] != asm_preds[i]) {
            if (mismatches < 20)
                printf("  MISMATCH image %d: C=%d ASM=%d (true=%d)\n",
                       i, c_preds[i], asm_preds[i], test_labels[i]);
            mismatches++;
        }
    }
    if (mismatches == 0)
        printf("  PASS: 100%% agreement (10000/10000)\n");
    else
        printf("  FAIL: %d mismatches out of %d\n", mismatches, TEST_N);

    /* ================================================================
     *  Phase 4: Throughput benchmark (10 passes, fully hot)
     * ================================================================ */
    printf("\n--- Throughput Benchmark (10 passes, hot cache) ---\n");
    int passes = 10;
    volatile int sink = 0;
    double tb0 = now_sec();
    for (int p = 0; p < passes; p++)
        for (int i = 0; i < TEST_N; i++)
            sink += sstt_fused_classify(
                test_pixels_padded + (size_t)i * (PIXELS + 4),
                (const uint32_t *)px_hot,
                (const uint32_t *)hg_hot,
                (const uint32_t *)vg_hot);
    double tb1 = now_sec();
    double ns_per = (tb1 - tb0) * 1e9 / (passes * TEST_N);
    printf("  %.0f ns/query (%.0f classifications/sec)\n",
           ns_per, 1e9 / ns_per);
    printf("  Hot maps: %zu KB in L-cache (Shared state)\n",
           (sizeof(px_hot) + sizeof(hg_hot) + sizeof(vg_hot)) / 1024);

    /* Summary */
    printf("\n=== SUMMARY ===\n");
    printf("  C reference:   %.2f%% in %.0f ns/query\n",
           100.0 * c_correct / TEST_N, (tc1 - tc0) * 1e9 / TEST_N);
    printf("  ASM fused:     %.2f%% in %.0f ns/query\n",
           100.0 * asm_correct / TEST_N, ns_per);
    printf("  Speedup:       %.1fx\n",
           (tc1 - tc0) * 1e9 / TEST_N / ns_per);
    printf("  Agreement:     %s (%d mismatches)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches);
    printf("  Model size:    %zu KB (3 hot maps, L-cache resident)\n",
           (sizeof(px_hot) + sizeof(hg_hot) + sizeof(vg_hot)) / 1024);
    printf("  Working set:   224 bytes stack (L1d)\n");
    printf("  Code size:     <1 KB (L1i)\n");
    printf("  Intermediates: 0 bytes (fused pipeline)\n");

    free(c_preds); free(asm_preds); free(test_pixels_padded);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test); free(vgrad_train); free(vgrad_test);
    free(px_train_sigs); free(px_test_sigs);
    free(hg_train_sigs); free(hg_test_sigs);
    free(vg_train_sigs); free(vg_test_sigs);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);

    return mismatches > 0 ? 1 : 0;
}
