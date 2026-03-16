/*
 * sstt_eigenseries.c — Eigenvalue-Ordered Bayesian Series
 *
 * The block-interleaved Bayesian (83.86%) processes blocks in spatial
 * order. This experiment reorders block processing by semantic importance:
 *
 * 1. Compute block signature covariance (252×252) from training data.
 * 2. Eigendecompose to find which block positions co-vary most.
 * 3. Reorder block processing: highest-eigenvalue-loading blocks first.
 * 4. Also test: IG-ordered (simpler), and early stopping (skip low-value blocks).
 *
 * The eigenvectors are the KEYS (semantic dimensions).
 * The eigenvalues are the VALUES (importance weights).
 * The block ordering is the ADDRESSING (traverse the hot map by semantic
 * importance, not spatial position).
 *
 * Build: make sstt_eigenseries  (after: make mnist)
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N     60000
#define TEST_N      10000
#define IMG_W       28
#define IMG_H       28
#define PIXELS      784
#define PADDED      800
#define N_CLASSES   10
#define CLS_PAD     16
#define BLK_W       3
#define BLKS_PER_ROW 9
#define N_BLOCKS    252
#define N_BVALS     27
#define SIG_PAD     256
#define BG_PIXEL    0
#define BG_GRAD     13
#define POWER_ITERS 30

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t *tern_train, *tern_test;
static int8_t *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;
static uint8_t *px_train_sigs, *px_test_sigs;
static uint8_t *hg_train_sigs, *hg_test_sigs;
static uint8_t *vg_train_sigs, *vg_test_sigs;
static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* --- Standard infrastructure (same as sstt_series.c) --- */

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
    for (int i = 0; i < PIXELS; i++)
        dst[i] = src[i] > 170 ? 1 : src[i] < 85 ? -1 : 0;
    memset(dst + PIXELS, 0, PADDED - PIXELS);
}

static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

static void compute_gradients_one(const int8_t *t, int8_t *hg, int8_t *vg) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W-1; x++)
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
        quantize_one(raw_train_img+(size_t)i*PIXELS, tern_train+(size_t)i*PADDED);
        compute_gradients_one(tern_train+(size_t)i*PADDED,
            hgrad_train+(size_t)i*PADDED, vgrad_train+(size_t)i*PADDED);
    }
    for (int i = 0; i < TEST_N; i++) {
        quantize_one(raw_test_img+(size_t)i*PIXELS, tern_test+(size_t)i*PADDED);
        compute_gradients_one(tern_test+(size_t)i*PADDED,
            hgrad_test+(size_t)i*PADDED, vgrad_test+(size_t)i*PADDED);
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
                int base = y*IMG_W + s*BLK_W;
                sig[y*BLKS_PER_ROW+s] = block_encode(img[base], img[base+1], img[base+2]);
            }
        memset(sig+N_BLOCKS, 0xFF, SIG_PAD-N_BLOCKS);
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
 *  Block Importance Metrics
 * ================================================================ */

/* Information Gain per block position (across all channels) */
static void compute_block_ig(double ig_out[N_BLOCKS]) {
    double class_prior[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) class_prior[train_labels[i]]++;
    double H_prior = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        class_prior[c] /= TRAIN_N;
        if (class_prior[c] > 0) H_prior -= class_prior[c] * log2(class_prior[c]);
    }

    /* Combined IG: sum of IG across all 3 channels per position */
    const uint8_t *all_sigs[3] = {px_train_sigs, hg_train_sigs, vg_train_sigs};
    uint8_t bgs[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};

    for (int k = 0; k < N_BLOCKS; k++) {
        double total_ig = 0;
        for (int ch = 0; ch < 3; ch++) {
            int bv_count[N_BVALS] = {0};
            int bv_class[N_BVALS][N_CLASSES];
            memset(bv_class, 0, sizeof(bv_class));
            for (int i = 0; i < TRAIN_N; i++) {
                uint8_t bv = all_sigs[ch][i * SIG_PAD + k];
                bv_count[bv]++;
                bv_class[bv][train_labels[i]]++;
            }
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
            total_ig += H_prior - H_cond;
        }
        ig_out[k] = total_ig;
    }
}

/* Block covariance eigenvector importance (power iteration on 252×252) */
static void compute_block_eigen_importance(double eigen_imp[N_BLOCKS]) {
    /* Represent each training image as a 252-dim vector where each
     * dimension is the block value (0-26) at that position.
     * Compute covariance of these vectors. Eigendecompose.
     * The first eigenvector's loadings tell us which positions
     * co-vary the most. */

    /* Mean block value per position */
    double mean[N_BLOCKS] = {0};
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = px_train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            mean[k] += sig[k];
    }
    for (int k = 0; k < N_BLOCKS; k++) mean[k] /= TRAIN_N;

    /* Power iteration for top eigenvector of covariance matrix.
     * C = (1/N) X^T X where X is (N × 252), each row centered.
     * We compute C*v implicitly: Cv = (1/N) X^T (X v) */
    double *v = malloc(N_BLOCKS * sizeof(double));
    double *Cv = malloc(N_BLOCKS * sizeof(double));

    /* Random init */
    unsigned seed = 42;
    for (int k = 0; k < N_BLOCKS; k++) {
        seed = seed * 1103515245 + 12345;
        v[k] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }

    for (int iter = 0; iter < POWER_ITERS; iter++) {
        memset(Cv, 0, N_BLOCKS * sizeof(double));
        for (int i = 0; i < TRAIN_N; i++) {
            const uint8_t *sig = px_train_sigs + (size_t)i * SIG_PAD;
            /* u = (x - mean)^T v */
            double u = 0;
            for (int k = 0; k < N_BLOCKS; k++)
                u += ((double)sig[k] - mean[k]) * v[k];
            /* Cv += u * (x - mean) */
            for (int k = 0; k < N_BLOCKS; k++)
                Cv[k] += u * ((double)sig[k] - mean[k]);
        }
        /* Normalize */
        double norm = 0;
        for (int k = 0; k < N_BLOCKS; k++) norm += Cv[k] * Cv[k];
        norm = sqrt(norm);
        if (norm < 1e-10) break;
        for (int k = 0; k < N_BLOCKS; k++) v[k] = Cv[k] / norm;
    }

    /* eigen_imp[k] = |v[k]| = how much position k loads on the top eigenvector */
    for (int k = 0; k < N_BLOCKS; k++)
        eigen_imp[k] = fabs(v[k]);

    free(v);
    free(Cv);
}

/* Combined importance: IG × eigen loading (product captures both signals) */
static void compute_combined_importance(const double ig[N_BLOCKS],
                                         const double eigen[N_BLOCKS],
                                         double combined[N_BLOCKS]) {
    for (int k = 0; k < N_BLOCKS; k++)
        combined[k] = ig[k] * eigen[k];
}

/* Sort indices by importance (descending) */
static void sort_by_importance(const double *imp, int *order, int n) {
    for (int i = 0; i < n; i++) order[i] = i;
    /* Insertion sort (only 252 elements) */
    for (int i = 1; i < n; i++) {
        int key = order[i];
        double key_val = imp[key];
        int j = i - 1;
        while (j >= 0 && imp[order[j]] < key_val) {
            order[j+1] = order[j];
            j--;
        }
        order[j+1] = key;
    }
}

/* ================================================================
 *  Block-Interleaved Bayesian with Custom Ordering + Early Stop
 * ================================================================ */

static int series_ordered(int img_idx, const int *order, int n_blocks) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;

    const uint8_t *sigs[3] = {
        px_test_sigs + (size_t)img_idx * SIG_PAD,
        hg_test_sigs + (size_t)img_idx * SIG_PAD,
        vg_test_sigs + (size_t)img_idx * SIG_PAD
    };
    uint32_t (*hots[3])[N_BVALS][CLS_PAD] = {px_hot, hg_hot, vg_hot};
    uint8_t bgs[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};

    for (int ki = 0; ki < n_blocks; ki++) {
        int k = order[ki];
        for (int ch = 0; ch < 3; ch++) {
            uint8_t bv = sigs[ch][k];
            if (bv == bgs[ch]) continue;
            const uint32_t *h = hots[ch][k][bv];
            double max_a = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                acc[c] *= ((double)h[c] + 0.5);
                if (acc[c] > max_a) max_a = acc[c];
            }
            if (max_a > 1e10)
                for (int c = 0; c < N_CLASSES; c++) acc[c] /= max_a;
        }
    }
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (acc[c] > acc[best]) best = c;
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
    printf("=== SSTT Eigen-Series: KV-Addressed Bayesian Pipeline (%s) ===\n\n", dataset);

    printf("Loading and preprocessing...\n");
    load_data(data_dir); init_all(); compute_all_sigs();
    build_hot_map(px_train_sigs, px_hot);
    build_hot_map(hg_train_sigs, hg_hot);
    build_hot_map(vg_train_sigs, vg_hot);
    double t1 = now_sec();
    printf("  Done (%.2f sec)\n\n", t1 - t0);

    /* Compute importance metrics */
    printf("Computing block importance metrics...\n");
    double ig[N_BLOCKS], eigen_imp[N_BLOCKS], combined[N_BLOCKS];
    compute_block_ig(ig);
    double te0 = now_sec();
    compute_block_eigen_importance(eigen_imp);
    double te1 = now_sec();
    compute_combined_importance(ig, eigen_imp, combined);
    printf("  IG computed, eigen importance (%.2f sec)\n", te1 - te0);

    /* Sort orderings */
    int spatial_order[N_BLOCKS], ig_order[N_BLOCKS];
    int eigen_order[N_BLOCKS], combined_order[N_BLOCKS];
    int reverse_spatial[N_BLOCKS];

    for (int k = 0; k < N_BLOCKS; k++) spatial_order[k] = k;
    for (int k = 0; k < N_BLOCKS; k++) reverse_spatial[k] = N_BLOCKS - 1 - k;
    sort_by_importance(ig, ig_order, N_BLOCKS);
    sort_by_importance(eigen_imp, eigen_order, N_BLOCKS);
    sort_by_importance(combined, combined_order, N_BLOCKS);

    /* Print top-10 blocks per ordering */
    printf("\n  Top-10 blocks by ordering:\n");
    printf("  %-5s | %-20s | %-20s | %-20s\n", "Rank", "IG", "Eigen", "Combined");
    printf("  ------+----------------------+----------------------+---------------------\n");
    for (int i = 0; i < 10; i++)
        printf("  %4d  | k=%3d (%.4f)     | k=%3d (%.4f)     | k=%3d (%.6f)\n",
               i, ig_order[i], ig[ig_order[i]],
               eigen_order[i], eigen_imp[eigen_order[i]],
               combined_order[i], combined[combined_order[i]]);

    /* ================================================================
     *  Test all orderings, with early stopping sweep
     * ================================================================ */
    printf("\n--- Ordering × Early Stopping ---\n\n");

    const char *order_names[] = {
        "Spatial (baseline)",
        "Reverse spatial",
        "IG-ordered",
        "Eigen-ordered",
        "Combined (IG×Eigen)"
    };
    int *orders[] = {spatial_order, reverse_spatial, ig_order, eigen_order, combined_order};
    int n_orders = 5;
    int stop_points[] = {50, 100, 150, 200, 252};
    int n_stops = 5;

    printf("  %-22s |", "Ordering");
    for (int si = 0; si < n_stops; si++) printf(" %4d blks ", stop_points[si]);
    printf("\n  -----------------------+");
    for (int si = 0; si < n_stops; si++) printf("----------");
    printf("\n");

    for (int oi = 0; oi < n_orders; oi++) {
        printf("  %-22s |", order_names[oi]);
        for (int si = 0; si < n_stops; si++) {
            int correct = 0;
            for (int i = 0; i < TEST_N; i++)
                if (series_ordered(i, orders[oi], stop_points[si]) == test_labels[i])
                    correct++;
            printf(" %6.2f%% ", 100.0 * correct / TEST_N);
        }
        printf("\n");
    }

    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(px_train_sigs); free(px_test_sigs);
    free(hg_train_sigs); free(hg_test_sigs);
    free(vg_train_sigs); free(vg_test_sigs);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);
    return 0;
}
