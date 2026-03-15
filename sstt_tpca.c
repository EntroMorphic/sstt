/*
 * sstt_tpca.c — Ternary PCA: Eigenvector-Guided Feature Space
 *
 * Hypothesis: Quantizing the top PCA eigenvectors to ternary and using
 * them as the feature basis (instead of raw pixels) concentrates
 * discriminative signal into fewer dimensions, improving hot map accuracy.
 *
 * Pipeline:
 *   1. Compute pixel covariance matrix (784×784) from training data.
 *   2. Extract top-K eigenvectors via power iteration.
 *   3. Quantize eigenvectors to ternary {-1, 0, +1}.
 *   4. Project images: ternary dot product with each ternary eigenvector.
 *   5. Quantize projections to ternary (threshold on projection range).
 *   6. Block-encode projected features → hot map → classify.
 *
 * The key insight: PCA finds directions of maximum variance. Quantizing
 * those directions to ternary preserves the most discriminative axes
 * while staying in the integer-only domain.
 *
 * Build: make sstt_tpca  (after: make mnist)
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

/* PCA config */
#define MAX_PCS     252     /* max principal components (to match block count) */
#define POWER_ITERS 50      /* power iteration convergence */

/* Block config for projected features */
#define BLK_W_PC    3       /* 3 PCs per block */
#define N_BVALS     27      /* 3^3 */

/* Background skip for projected features:
 * block_encode(0,0,0) = 13 (all-zero projections = uninformative) */
#define BG_PC       13

static const char *data_dir = "data/";

/* ---------- Timing ---------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Data ---------- */
static uint8_t *raw_train_img;
static uint8_t *raw_test_img;
static uint8_t *train_labels;
static uint8_t *test_labels;

static int8_t *tern_train;     /* [TRAIN_N * PADDED] */
static int8_t *tern_test;

/* PCA eigenvectors: float[MAX_PCS][PIXELS] */
static float *eigvecs;         /* row-major: eigvecs[k*PIXELS + j] */
/* Ternary eigenvectors: int8_t[MAX_PCS][PADDED] */
static int8_t *tern_eigvecs;
/* Projections: int32_t[N][MAX_PCS] (ternary dot products) */
static int32_t *train_proj;
static int32_t *test_proj;
/* Ternary projections: int8_t[N][MAX_PCS] */
static int8_t *tern_train_proj;
static int8_t *tern_test_proj;

/* ================================================================
 *  IDX Loader
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
    if (!data) { fclose(f); exit(1); }
    if (fread(data, 1, total, f) != total) { fclose(f); exit(1); }
    fclose(f);
    return data;
}

static void load_data(const char *dir) {
    uint32_t n, r, c;
    char path[256];
    snprintf(path, sizeof(path), "%strain-images-idx3-ubyte", dir);
    raw_train_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%strain-labels-idx1-ubyte", dir);
    train_labels = load_idx(path, &n, NULL, NULL);
    snprintf(path, sizeof(path), "%st10k-images-idx3-ubyte", dir);
    raw_test_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%st10k-labels-idx1-ubyte", dir);
    test_labels = load_idx(path, &n, NULL, NULL);
}

/* ================================================================
 *  Ternary Quantization
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
        quantize_one(raw_train_img + (size_t)i * PIXELS,
                     tern_train + (size_t)i * PADDED);
    for (int i = 0; i < TEST_N; i++)
        quantize_one(raw_test_img + (size_t)i * PIXELS,
                     tern_test + (size_t)i * PADDED);
}

/* ================================================================
 *  Ternary Dot Product (from sstt_geom.c, scalar version)
 * ================================================================ */

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b, int len) {
    int32_t sum = 0;
    for (int i = 0; i < len; i++)
        sum += a[i] * b[i];
    return sum;
}

/* ================================================================
 *  PCA via Power Iteration on Ternary Data
 *
 *  We compute eigenvectors of the covariance of ternary images.
 *  Since ternary values are {-1, 0, +1}, the covariance is integer:
 *    C[i][j] = (1/N) * Σ_n tern[n][i] * tern[n][j]
 *  We don't subtract the mean (it's near zero for centered ternary).
 *
 *  Power iteration: start with random vector, repeatedly multiply
 *  by the data matrix (implicitly), normalize. Deflate after each.
 * ================================================================ */

static void compute_pca(int n_pcs) {
    eigvecs = calloc((size_t)n_pcs * PIXELS, sizeof(float));
    if (!eigvecs) { fprintf(stderr, "ERROR: eigvec alloc\n"); exit(1); }

    /* We'll work with float copies for numerical stability */
    float *mean = calloc(PIXELS, sizeof(float));

    /* Compute mean (for centering) */
    for (int i = 0; i < TRAIN_N; i++) {
        const int8_t *img = tern_train + (size_t)i * PADDED;
        for (int j = 0; j < PIXELS; j++)
            mean[j] += (float)img[j];
    }
    for (int j = 0; j < PIXELS; j++)
        mean[j] /= TRAIN_N;

    printf("  Mean range: [%.3f, %.3f]\n", mean[0], mean[PIXELS/2]);

    /* Power iteration for each PC */
    float *v = malloc(PIXELS * sizeof(float));
    float *Av = malloc(PIXELS * sizeof(float));

    /* Simple PRNG for initialization */
    unsigned seed = 42;

    for (int pc = 0; pc < n_pcs; pc++) {
        /* Random init */
        for (int j = 0; j < PIXELS; j++) {
            seed = seed * 1103515245 + 12345;
            v[j] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
        }

        for (int iter = 0; iter < POWER_ITERS; iter++) {
            /* Av = X^T X v  (implicitly, without forming the 784×784 matrix) */
            /* Step 1: u = X * v (project all training images onto v) */
            /* Step 2: Av = X^T * u (backproject) */
            memset(Av, 0, PIXELS * sizeof(float));

            for (int i = 0; i < TRAIN_N; i++) {
                const int8_t *img = tern_train + (size_t)i * PADDED;
                /* u_i = Σ_j (img[j] - mean[j]) * v[j] */
                float u = 0;
                for (int j = 0; j < PIXELS; j++)
                    u += ((float)img[j] - mean[j]) * v[j];
                /* Av += u_i * (img - mean) */
                for (int j = 0; j < PIXELS; j++)
                    Av[j] += u * ((float)img[j] - mean[j]);
            }

            /* Deflate: remove projections onto previous eigenvectors */
            for (int prev = 0; prev < pc; prev++) {
                float dot = 0;
                for (int j = 0; j < PIXELS; j++)
                    dot += Av[j] * eigvecs[prev * PIXELS + j];
                for (int j = 0; j < PIXELS; j++)
                    Av[j] -= dot * eigvecs[prev * PIXELS + j];
            }

            /* Normalize */
            float norm = 0;
            for (int j = 0; j < PIXELS; j++)
                norm += Av[j] * Av[j];
            norm = sqrtf(norm);
            if (norm < 1e-10f) break;
            for (int j = 0; j < PIXELS; j++)
                v[j] = Av[j] / norm;
        }

        /* Store eigenvector */
        memcpy(eigvecs + (size_t)pc * PIXELS, v, PIXELS * sizeof(float));

        /* Compute eigenvalue (Rayleigh quotient) */
        float lambda = 0;
        for (int i = 0; i < TRAIN_N; i++) {
            const int8_t *img = tern_train + (size_t)i * PADDED;
            float proj = 0;
            for (int j = 0; j < PIXELS; j++)
                proj += ((float)img[j] - mean[j]) * v[j];
            lambda += proj * proj;
        }
        lambda /= TRAIN_N;

        if (pc < 20 || pc % 50 == 0)
            printf("  PC %3d: eigenvalue = %.1f\n", pc, lambda);
    }

    free(mean);
    free(v);
    free(Av);
}

/* ================================================================
 *  Ternary Eigenvectors: quantize float eigenvectors to {-1, 0, +1}
 *
 *  Strategy: threshold at ±T where T captures the top/bottom N% of
 *  the eigenvector's magnitude. This preserves the most active
 *  components and zeros out the weak ones.
 * ================================================================ */

static void quantize_eigenvectors(int n_pcs, float threshold_pct) {
    tern_eigvecs = (int8_t *)aligned_alloc(32, (size_t)n_pcs * PADDED);
    memset(tern_eigvecs, 0, (size_t)n_pcs * PADDED);

    int total_pos = 0, total_neg = 0, total_zero = 0;

    for (int pc = 0; pc < n_pcs; pc++) {
        const float *ev = eigvecs + (size_t)pc * PIXELS;
        int8_t *tev = tern_eigvecs + (size_t)pc * PADDED;

        /* Find threshold: sort absolute values, pick percentile */
        float absvals[PIXELS];
        for (int j = 0; j < PIXELS; j++)
            absvals[j] = fabsf(ev[j]);

        /* Simple insertion sort (only 784 elements) */
        for (int i = 1; i < PIXELS; i++) {
            float key = absvals[i];
            int j = i - 1;
            while (j >= 0 && absvals[j] > key) {
                absvals[j + 1] = absvals[j];
                j--;
            }
            absvals[j + 1] = key;
        }

        /* Threshold: top (1-threshold_pct) fraction get ±1 */
        int thr_idx = (int)(PIXELS * threshold_pct);
        if (thr_idx >= PIXELS) thr_idx = PIXELS - 1;
        float T = absvals[thr_idx];

        for (int j = 0; j < PIXELS; j++) {
            if (ev[j] > T)       { tev[j] =  1; total_pos++; }
            else if (ev[j] < -T) { tev[j] = -1; total_neg++; }
            else                  { tev[j] =  0; total_zero++; }
        }
    }

    long total = (long)n_pcs * PIXELS;
    printf("  Ternary eigenvector distribution: -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
           100.0 * total_neg / total, 100.0 * total_zero / total,
           100.0 * total_pos / total);
}

/* ================================================================
 *  Project: ternary dot product of images with ternary eigenvectors
 * ================================================================ */

static void project_all(int n_pcs) {
    train_proj = malloc((size_t)TRAIN_N * n_pcs * sizeof(int32_t));
    test_proj  = malloc((size_t)TEST_N  * n_pcs * sizeof(int32_t));

    for (int i = 0; i < TRAIN_N; i++) {
        const int8_t *img = tern_train + (size_t)i * PADDED;
        int32_t *proj = train_proj + (size_t)i * n_pcs;
        for (int pc = 0; pc < n_pcs; pc++) {
            const int8_t *ev = tern_eigvecs + (size_t)pc * PADDED;
            proj[pc] = ternary_dot(img, ev, PIXELS);
        }
    }

    for (int i = 0; i < TEST_N; i++) {
        const int8_t *img = tern_test + (size_t)i * PADDED;
        int32_t *proj = test_proj + (size_t)i * n_pcs;
        for (int pc = 0; pc < n_pcs; pc++) {
            const int8_t *ev = tern_eigvecs + (size_t)pc * PADDED;
            proj[pc] = ternary_dot(img, ev, PIXELS);
        }
    }
}

/* ================================================================
 *  Quantize projections to ternary
 *
 *  Each PC projection has a different range. Threshold per-PC:
 *  projection > T_pc → +1, < -T_pc → -1, else 0.
 *  T_pc = mean(|projection|) across training set.
 * ================================================================ */

static void quantize_projections(int n_pcs) {
    tern_train_proj = (int8_t *)calloc(TRAIN_N, n_pcs);
    tern_test_proj  = (int8_t *)calloc(TEST_N,  n_pcs);

    /* Compute per-PC threshold from training data */
    float *thresholds = malloc(n_pcs * sizeof(float));

    for (int pc = 0; pc < n_pcs; pc++) {
        double sum_abs = 0;
        for (int i = 0; i < TRAIN_N; i++)
            sum_abs += abs(train_proj[i * n_pcs + pc]);
        thresholds[pc] = (float)(sum_abs / TRAIN_N);
    }

    /* Quantize */
    for (int i = 0; i < TRAIN_N; i++) {
        for (int pc = 0; pc < n_pcs; pc++) {
            int32_t p = train_proj[i * n_pcs + pc];
            float T = thresholds[pc];
            if ((float)p > T)        tern_train_proj[i * n_pcs + pc] =  1;
            else if ((float)p < -T)  tern_train_proj[i * n_pcs + pc] = -1;
            /* else 0 (already) */
        }
    }
    for (int i = 0; i < TEST_N; i++) {
        for (int pc = 0; pc < n_pcs; pc++) {
            int32_t p = test_proj[i * n_pcs + pc];
            float T = thresholds[pc];
            if ((float)p > T)        tern_test_proj[i * n_pcs + pc] =  1;
            else if ((float)p < -T)  tern_test_proj[i * n_pcs + pc] = -1;
        }
    }

    /* Stats */
    long pos = 0, neg = 0, zero = 0;
    for (long i = 0; i < (long)TRAIN_N * n_pcs; i++) {
        if (tern_train_proj[i] > 0) pos++;
        else if (tern_train_proj[i] < 0) neg++;
        else zero++;
    }
    long total = (long)TRAIN_N * n_pcs;
    printf("  Projection trit dist: -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
           100.0 * neg / total, 100.0 * zero / total, 100.0 * pos / total);

    free(thresholds);
}

/* ================================================================
 *  Block Encoding & Hot Map on Projected Features
 * ================================================================ */

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

/* Hot map for projected features: hot[n_blocks_pc][27][16] */
static void build_pc_hot_map(int n_pcs, uint32_t *hot) {
    int n_blocks = n_pcs / 3;
    size_t hot_size = (size_t)n_blocks * N_BVALS * CLS_PAD;
    memset(hot, 0, hot_size * sizeof(uint32_t));

    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const int8_t *proj = tern_train_proj + (size_t)i * n_pcs;
        for (int k = 0; k < n_blocks; k++) {
            int base = k * 3;
            uint8_t bv = block_encode(proj[base], proj[base+1], proj[base+2]);
            hot[(size_t)k * N_BVALS * CLS_PAD + (size_t)bv * CLS_PAD + lbl]++;
        }
    }
}

static int pc_hot_classify(const int8_t *proj, int n_pcs,
                            const uint32_t *hot) {
    int n_blocks = n_pcs / 3;
    uint32_t acc[CLS_PAD] = {0};

    for (int k = 0; k < n_blocks; k++) {
        int base = k * 3;
        uint8_t bv = block_encode(proj[base], proj[base+1], proj[base+2]);
        if (bv == BG_PC) continue;
        const uint32_t *votes = hot + (size_t)k * N_BVALS * CLS_PAD
                                    + (size_t)bv * CLS_PAD;
        for (int c = 0; c < CLS_PAD; c++)
            acc[c] += votes[c];
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
    return best;
}

/* ================================================================
 *  Pixel-space hot map baseline (2D blocks, for comparison)
 * ================================================================ */

#define PX_BLKS_PER_ROW 9
#define PX_N_BLOCKS     252

static void build_pixel_hot_map(uint32_t *hot) {
    size_t hot_size = (size_t)PX_N_BLOCKS * N_BVALS * CLS_PAD;
    memset(hot, 0, hot_size * sizeof(uint32_t));

    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const int8_t *img = tern_train + (size_t)i * PADDED;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < PX_BLKS_PER_ROW; s++) {
                int k = y * PX_BLKS_PER_ROW + s;
                int base = y * IMG_W + s * 3;
                uint8_t bv = block_encode(img[base], img[base+1], img[base+2]);
                hot[(size_t)k * N_BVALS * CLS_PAD + (size_t)bv * CLS_PAD + lbl]++;
            }
    }
}

static int pixel_hot_classify(const int8_t *img, const uint32_t *hot) {
    uint32_t acc[CLS_PAD] = {0};
    for (int y = 0; y < IMG_H; y++)
        for (int s = 0; s < PX_BLKS_PER_ROW; s++) {
            int k = y * PX_BLKS_PER_ROW + s;
            int base = y * IMG_W + s * 3;
            uint8_t bv = block_encode(img[base], img[base+1], img[base+2]);
            if (bv == 0) continue;
            const uint32_t *votes = hot + (size_t)k * N_BVALS * CLS_PAD
                                        + (size_t)bv * CLS_PAD;
            for (int c = 0; c < CLS_PAD; c++)
                acc[c] += votes[c];
        }
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
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
        if (len > 0 && data_dir[len - 1] != '/') {
            char *buf = malloc(len + 2);
            memcpy(buf, data_dir, len);
            buf[len] = '/';
            buf[len + 1] = '\0';
            data_dir = buf;
        }
    }

    const char *dataset = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Ternary PCA: Eigenvector-Guided Features (%s) ===\n\n", dataset);

    /* Load and quantize */
    printf("Loading %s...\n", dataset);
    load_data(data_dir);
    quantize_all();
    double t1 = now_sec();
    printf("  Loaded and quantized (%.2f sec)\n\n", t1 - t0);

    /* Pixel baseline */
    printf("--- Pixel Hot Map Baseline (252 2D blocks) ---\n");
    uint32_t *px_hot = aligned_alloc(32, (size_t)PX_N_BLOCKS * N_BVALS * CLS_PAD * sizeof(uint32_t));
    build_pixel_hot_map(px_hot);
    int px_correct = 0;
    for (int i = 0; i < TEST_N; i++)
        if (pixel_hot_classify(tern_test + (size_t)i * PADDED, px_hot) == test_labels[i])
            px_correct++;
    printf("  Pixel hot map: %.2f%%\n\n", 100.0 * px_correct / TEST_N);
    free(px_hot);

    /* PCA */
    int pc_counts[] = {27, 54, 84, 126, 252};
    int n_trials = 5;
    float thresholds[] = {0.5f, 0.33f, 0.67f};
    int n_thresholds = 3;

    for (int ti = 0; ti < n_trials; ti++) {
        int n_pcs = pc_counts[ti];
        int n_blocks = n_pcs / 3;

        printf("--- Ternary PCA: %d PCs (%d blocks of 3) ---\n", n_pcs, n_blocks);

        /* Compute PCA (or reuse if already computed enough) */
        double tp0 = now_sec();
        compute_pca(n_pcs);
        double tp1 = now_sec();
        printf("  PCA: %.2f sec (%d eigenvectors × %d power iterations)\n",
               tp1 - tp0, n_pcs, POWER_ITERS);

        /* Sweep eigenvector quantization thresholds */
        for (int thi = 0; thi < n_thresholds; thi++) {
            float evq_pct = thresholds[thi];
            printf("\n  Eigenvector quantization: top %.0f%% → ±1\n", (1.0f - evq_pct) * 100);

            quantize_eigenvectors(n_pcs, evq_pct);

            /* Project */
            project_all(n_pcs);

            /* Quantize projections */
            quantize_projections(n_pcs);

            /* Build hot map and classify */
            uint32_t *pc_hot = aligned_alloc(32,
                (size_t)n_blocks * N_BVALS * CLS_PAD * sizeof(uint32_t));
            build_pc_hot_map(n_pcs, pc_hot);

            int correct = 0;
            for (int i = 0; i < TEST_N; i++) {
                const int8_t *proj = tern_test_proj + (size_t)i * n_pcs;
                if (pc_hot_classify(proj, n_pcs, pc_hot) == test_labels[i])
                    correct++;
            }
            double acc = 100.0 * correct / TEST_N;
            printf("  %d-PC hot map: %.2f%% (vs pixel %.2f%%: %+.2f pp)\n",
                   n_pcs, acc, 100.0 * px_correct / TEST_N,
                   acc - 100.0 * px_correct / TEST_N);

            free(pc_hot);
            free(tern_train_proj); tern_train_proj = NULL;
            free(tern_test_proj);  tern_test_proj = NULL;
            free(train_proj); train_proj = NULL;
            free(test_proj);  test_proj = NULL;
            free(tern_eigvecs); tern_eigvecs = NULL;
        }

        free(eigvecs); eigvecs = NULL;
        printf("\n");
    }

    /* Cleanup */
    free(tern_train); free(tern_test);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);

    printf("Total runtime: %.2f seconds.\n", now_sec() - t0);
    return 0;
}
