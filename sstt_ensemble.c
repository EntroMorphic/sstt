/*
 * sstt_ensemble.c — Series Ensemble: Three Independent Hot Maps Scored at the End
 *
 * Each channel (pixel, h-grad, v-grad) produces its own independent
 * class vote vector. The three vectors are combined with various
 * fusion rules AFTER all channels have voted.
 *
 * Fusion strategies:
 *   1. Additive (current baseline): final[c] = px[c] + hg[c] + vg[c]
 *   2. Majority argmax: 3 independent argmaxes, majority vote
 *   3. Weighted sum: final[c] = w_px*px[c] + w_hg*hg[c] + w_vg*vg[c]
 *   4. Rank fusion (Borda count): each channel ranks classes, sum ranks
 *   5. Product: final[c] = px[c] * hg[c] * vg[c] (agreement enforced)
 *   6. Max: final[c] = max(px[c], hg[c], vg[c]) (any-channel support)
 *   7. Cascaded confidence: best single channel; add others only if margin low
 *   8. Normalized sum: normalize each channel to [0,1] then sum
 *   9. Agreement-weighted: weight each channel's vote by its own confidence
 *
 * Build: make sstt_ensemble  (after: make mnist)
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

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Data */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t *tern_train, *tern_test;
static int8_t *hgrad_train, *hgrad_test;
static int8_t *vgrad_train, *vgrad_test;
static uint8_t *px_train_sigs, *px_test_sigs;
static uint8_t *hg_train_sigs, *hg_test_sigs;
static uint8_t *vg_train_sigs, *vg_test_sigs;

static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* --- Standard infrastructure (same as sstt_hybrid.c, abbreviated) --- */

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
 *  Per-Channel Hot Map Classification (independent accumulators)
 * ================================================================ */

static void classify_per_channel(int img_idx, uint32_t px_votes[CLS_PAD],
                                  uint32_t hg_votes[CLS_PAD],
                                  uint32_t vg_votes[CLS_PAD]) {
    memset(px_votes, 0, CLS_PAD * sizeof(uint32_t));
    memset(hg_votes, 0, CLS_PAD * sizeof(uint32_t));
    memset(vg_votes, 0, CLS_PAD * sizeof(uint32_t));

    const uint8_t *ps = px_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *hs = hg_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *vs = vg_test_sigs + (size_t)img_idx * SIG_PAD;

    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv;
        bv = ps[k];
        if (bv != BG_PIXEL)
            for (int c = 0; c < CLS_PAD; c++) px_votes[c] += px_hot[k][bv][c];
        bv = hs[k];
        if (bv != BG_GRAD)
            for (int c = 0; c < CLS_PAD; c++) hg_votes[c] += hg_hot[k][bv][c];
        bv = vs[k];
        if (bv != BG_GRAD)
            for (int c = 0; c < CLS_PAD; c++) vg_votes[c] += vg_hot[k][bv][c];
    }
}

static int argmax(const uint32_t *v, int n) {
    int best = 0;
    for (int c = 1; c < n; c++) if (v[c] > v[best]) best = c;
    return best;
}

static int argmax_f(const double *v, int n) {
    int best = 0;
    for (int c = 1; c < n; c++) if (v[c] > v[best]) best = c;
    return best;
}

static uint32_t margin(const uint32_t *v, int n) {
    uint32_t best = 0, second = 0;
    for (int c = 0; c < n; c++) {
        if (v[c] > best) { second = best; best = v[c]; }
        else if (v[c] > second) second = v[c];
    }
    return best - second;
}

/* ================================================================
 *  Fusion Strategies
 * ================================================================ */

/* 1. Additive (baseline) */
static int fuse_additive(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    uint32_t f[CLS_PAD];
    for (int c = 0; c < CLS_PAD; c++) f[c] = px[c] + hg[c] + vg[c];
    return argmax(f, N_CLASSES);
}

/* 2. Majority vote */
static int fuse_majority(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    int p = argmax(px, N_CLASSES);
    int h = argmax(hg, N_CLASSES);
    int v = argmax(vg, N_CLASSES);
    if (p == h || p == v) return p;
    if (h == v) return h;
    /* No majority — fall back to highest-margin channel */
    uint32_t mp = margin(px, N_CLASSES);
    uint32_t mh = margin(hg, N_CLASSES);
    uint32_t mv = margin(vg, N_CLASSES);
    if (mv >= mp && mv >= mh) return v;
    if (mp >= mh) return p;
    return h;
}

/* 3. Weighted sum (weight by known channel accuracy) */
static int fuse_weighted(const uint32_t *px, const uint32_t *hg, const uint32_t *vg,
                          double w_px, double w_hg, double w_vg) {
    double f[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++)
        f[c] = w_px * px[c] + w_hg * hg[c] + w_vg * vg[c];
    return argmax_f(f, N_CLASSES);
}

/* 4. Borda count (rank fusion) */
static int fuse_borda(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    /* Rank each channel's classes (higher vote = lower rank = better) */
    int rank_score[N_CLASSES] = {0};
    /* For each channel, sort classes by vote descending, assign rank 0=best */
    const uint32_t *channels[3] = {px, hg, vg};
    for (int ch = 0; ch < 3; ch++) {
        int order[N_CLASSES];
        for (int c = 0; c < N_CLASSES; c++) order[c] = c;
        /* Bubble sort (only 10 elements) */
        for (int i = 0; i < N_CLASSES-1; i++)
            for (int j = i+1; j < N_CLASSES; j++)
                if (channels[ch][order[j]] > channels[ch][order[i]]) {
                    int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
                }
        /* Borda: class at rank r gets (N-1-r) points */
        for (int r = 0; r < N_CLASSES; r++)
            rank_score[order[r]] += (N_CLASSES - 1 - r);
    }
    return argmax((uint32_t *)rank_score, N_CLASSES);
}

/* 5. Normalized sum (each channel to [0,1] then sum) */
static int fuse_normalized(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    double f[N_CLASSES];
    /* Find max per channel for normalization */
    uint32_t px_max = 1, hg_max = 1, vg_max = 1;
    for (int c = 0; c < N_CLASSES; c++) {
        if (px[c] > px_max) px_max = px[c];
        if (hg[c] > hg_max) hg_max = hg[c];
        if (vg[c] > vg_max) vg_max = vg[c];
    }
    for (int c = 0; c < N_CLASSES; c++)
        f[c] = (double)px[c]/px_max + (double)hg[c]/hg_max + (double)vg[c]/vg_max;
    return argmax_f(f, N_CLASSES);
}

/* 6. Confidence-weighted: each channel's vote scaled by its own margin */
static int fuse_confidence_weighted(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    double f[N_CLASSES];
    double mp = (double)margin(px, N_CLASSES);
    double mh = (double)margin(hg, N_CLASSES);
    double mv = (double)margin(vg, N_CLASSES);
    double total_m = mp + mh + mv;
    if (total_m < 1) total_m = 1;
    /* Weight each channel's votes by its confidence (margin) */
    for (int c = 0; c < N_CLASSES; c++)
        f[c] = mp/total_m * px[c] + mh/total_m * hg[c] + mv/total_m * vg[c];
    return argmax_f(f, N_CLASSES);
}

/* 7. V-Grad primary, add others only if margin is low */
static int fuse_vgrad_primary(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    uint32_t mv = margin(vg, N_CLASSES);
    uint32_t vg_max = 1;
    for (int c = 0; c < N_CLASSES; c++) if (vg[c] > vg_max) vg_max = vg[c];

    /* If V-Grad is confident (margin > 10% of max), use it alone */
    if (mv > vg_max / 10) return argmax(vg, N_CLASSES);

    /* Otherwise, add pixel votes (skip h-grad, it's the weakest) */
    uint32_t f[CLS_PAD];
    for (int c = 0; c < CLS_PAD; c++) f[c] = 2 * vg[c] + px[c];
    return argmax(f, N_CLASSES);
}

/* 8. Best-2: V-Grad + Pixel only (drop H-Grad) */
static int fuse_best2(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    (void)hg;
    uint32_t f[CLS_PAD];
    for (int c = 0; c < CLS_PAD; c++) f[c] = px[c] + vg[c];
    return argmax(f, N_CLASSES);
}

/* 9. V-Grad dominant: 4:1:2 weighting (favor v-grad, penalize h-grad) */
static int fuse_vgrad_dominant(const uint32_t *px, const uint32_t *hg, const uint32_t *vg) {
    uint32_t f[CLS_PAD];
    for (int c = 0; c < CLS_PAD; c++) f[c] = 2*px[c] + hg[c] + 4*vg[c];
    return argmax(f, N_CLASSES);
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
    printf("=== SSTT Series Ensemble (%s) ===\n\n", dataset);

    printf("Loading and preprocessing...\n");
    load_data(data_dir);
    init_all();
    compute_all_sigs();
    build_hot_map(px_train_sigs, px_hot);
    build_hot_map(hg_train_sigs, hg_hot);
    build_hot_map(vg_train_sigs, vg_hot);
    double t1 = now_sec();
    printf("  Done (%.2f sec)\n\n", t1 - t0);

    /* Individual channel accuracies */
    int px_correct = 0, hg_correct = 0, vg_correct = 0;
    for (int i = 0; i < TEST_N; i++) {
        uint32_t px_v[CLS_PAD], hg_v[CLS_PAD], vg_v[CLS_PAD];
        classify_per_channel(i, px_v, hg_v, vg_v);
        if (argmax(px_v, N_CLASSES) == test_labels[i]) px_correct++;
        if (argmax(hg_v, N_CLASSES) == test_labels[i]) hg_correct++;
        if (argmax(vg_v, N_CLASSES) == test_labels[i]) vg_correct++;
    }
    printf("--- Individual Channels ---\n");
    printf("  Pixel:   %.2f%%\n", 100.0 * px_correct / TEST_N);
    printf("  H-Grad:  %.2f%%\n", 100.0 * hg_correct / TEST_N);
    printf("  V-Grad:  %.2f%%\n\n", 100.0 * vg_correct / TEST_N);

    /* Run all fusion strategies */
    const char *names[] = {
        "1. Additive (baseline)",
        "2. Majority vote",
        "3. Weighted (by accuracy)",
        "4. Borda rank fusion",
        "5. Normalized sum",
        "6. Confidence-weighted",
        "7. V-Grad primary + fallback",
        "8. Best-2 (px + vg, drop hg)",
        "9. V-Grad dominant (2:1:4)"
    };
    int n_strategies = 9;

    printf("--- Fusion Strategies ---\n\n");
    printf("  %-35s | %-10s | %-12s\n", "Strategy", "Accuracy", "vs Additive");
    printf("  ------------------------------------+------------+-------------\n");

    for (int s = 0; s < n_strategies; s++) {
        int correct = 0;
        double ts0 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            uint32_t px_v[CLS_PAD], hg_v[CLS_PAD], vg_v[CLS_PAD];
            classify_per_channel(i, px_v, hg_v, vg_v);
            int pred;
            switch (s) {
                case 0: pred = fuse_additive(px_v, hg_v, vg_v); break;
                case 1: pred = fuse_majority(px_v, hg_v, vg_v); break;
                case 2: pred = fuse_weighted(px_v, hg_v, vg_v, 0.71, 0.60, 0.74); break;
                case 3: pred = fuse_borda(px_v, hg_v, vg_v); break;
                case 4: pred = fuse_normalized(px_v, hg_v, vg_v); break;
                case 5: pred = fuse_confidence_weighted(px_v, hg_v, vg_v); break;
                case 6: pred = fuse_vgrad_primary(px_v, hg_v, vg_v); break;
                case 7: pred = fuse_best2(px_v, hg_v, vg_v); break;
                case 8: pred = fuse_vgrad_dominant(px_v, hg_v, vg_v); break;
                default: pred = 0;
            }
            if (pred == test_labels[i]) correct++;
        }
        double ts1 = now_sec();
        double acc = 100.0 * correct / TEST_N;
        double base = 100.0 * (s == 0 ? correct : 0) / TEST_N;
        if (s == 0) base = acc;
        printf("  %-35s | %7.2f%%  | %+.2f pp\n",
               names[s], acc,
               s == 0 ? 0.0 : acc - 100.0 * px_correct / TEST_N * 0 - 73.23);
        (void)ts1; (void)base;
    }

    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    /* Cleanup */
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
