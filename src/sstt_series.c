/*
 * sstt_series.c — True Series Pipeline: Bayesian Channel Cascade
 *
 * Each channel updates a running posterior over classes.
 * Channel 1's output is Channel 2's prior. Channel 2's output is
 * Channel 3's prior. Evidence compounds — agreement amplifies,
 * disagreement dampens. Later channels make sharper decisions
 * because they have stronger priors.
 *
 * This is Bayesian inference with three independent likelihood terms:
 *   P(class | all) ∝ P(px|class) × P(vg|class) × P(hg|class) × P(class)
 *
 * In log space: log P = log L_px + log L_vg + log L_hg + log prior
 * In vote space: accumulate normalized vote products.
 *
 * Build: make sstt_series  (after: make mnist)
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

/* --- Standard infrastructure (abbreviated) --- */

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
 *  Per-channel vote extraction (raw uint32 vote vectors)
 * ================================================================ */

static void get_channel_votes(int img_idx,
                               uint32_t px_v[CLS_PAD],
                               uint32_t hg_v[CLS_PAD],
                               uint32_t vg_v[CLS_PAD]) {
    memset(px_v, 0, CLS_PAD * sizeof(uint32_t));
    memset(hg_v, 0, CLS_PAD * sizeof(uint32_t));
    memset(vg_v, 0, CLS_PAD * sizeof(uint32_t));
    const uint8_t *ps = px_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *hs = hg_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *vs = vg_test_sigs + (size_t)img_idx * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv;
        bv = ps[k]; if (bv != BG_PIXEL)
            for (int c = 0; c < CLS_PAD; c++) px_v[c] += px_hot[k][bv][c];
        bv = hs[k]; if (bv != BG_GRAD)
            for (int c = 0; c < CLS_PAD; c++) hg_v[c] += hg_hot[k][bv][c];
        bv = vs[k]; if (bv != BG_GRAD)
            for (int c = 0; c < CLS_PAD; c++) vg_v[c] += vg_hot[k][bv][c];
    }
}

/* Convert raw votes to probability distribution (with Laplace smoothing) */
static void votes_to_prob(const uint32_t *votes, double *prob, double temperature) {
    /* Softmax with temperature: P(c) = exp(votes[c]/T) / Σ exp(votes[c]/T) */
    double max_v = 0;
    for (int c = 0; c < N_CLASSES; c++)
        if (votes[c] > max_v) max_v = (double)votes[c];

    double sum = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        prob[c] = exp(((double)votes[c] - max_v) / temperature);
        sum += prob[c];
    }
    for (int c = 0; c < N_CLASSES; c++)
        prob[c] /= sum;
}

/* Normalize a probability vector (with floor to avoid zeros) */
static void normalize_prob(double *prob, double floor_val) {
    double sum = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        if (prob[c] < floor_val) prob[c] = floor_val;
        sum += prob[c];
    }
    for (int c = 0; c < N_CLASSES; c++) prob[c] /= sum;
}

static int argmax_d(const double *v, int n) {
    int best = 0;
    for (int c = 1; c < n; c++) if (v[c] > v[best]) best = c;
    return best;
}

/* ================================================================
 *  Series Pipeline Strategies
 * ================================================================ */

/* 1. Bayesian product (Naive Bayes): P(c|all) ∝ P(px|c) × P(vg|c) × P(hg|c)
 * Each channel's votes are converted to likelihoods, then multiplied. */
static int series_bayesian(const uint32_t *px, const uint32_t *hg,
                            const uint32_t *vg, double temp) {
    double p_px[N_CLASSES], p_hg[N_CLASSES], p_vg[N_CLASSES];
    votes_to_prob(px, p_px, temp);
    votes_to_prob(hg, p_hg, temp);
    votes_to_prob(vg, p_vg, temp);

    double posterior[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++)
        posterior[c] = p_px[c] * p_vg[c] * p_hg[c];
    return argmax_d(posterior, N_CLASSES);
}

/* 2. Sequential update: each channel updates the running posterior.
 * Channel order: ch[0] → ch[1] → ch[2]. Later channels see sharper priors. */
static int series_sequential(const uint32_t *votes[3], double temps[3]) {
    double posterior[N_CLASSES];
    /* Start with uniform prior */
    for (int c = 0; c < N_CLASSES; c++) posterior[c] = 1.0 / N_CLASSES;

    for (int ch = 0; ch < 3; ch++) {
        double likelihood[N_CLASSES];
        votes_to_prob(votes[ch], likelihood, temps[ch]);

        /* Bayesian update: posterior ∝ prior × likelihood */
        for (int c = 0; c < N_CLASSES; c++)
            posterior[c] *= likelihood[c];
        normalize_prob(posterior, 1e-10);
    }
    return argmax_d(posterior, N_CLASSES);
}

/* 3. Log-likelihood accumulation with progressive temperature.
 * Same as Bayesian but in log space (numerically stable).
 * Temperature decreases per stage → later channels make sharper decisions. */
static int series_log_progressive(const uint32_t *votes[3], double base_temp,
                                   double temp_decay) {
    double log_post[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) log_post[c] = 0;

    double temp = base_temp;
    for (int ch = 0; ch < 3; ch++) {
        double max_v = 0;
        for (int c = 0; c < N_CLASSES; c++)
            if (votes[ch][c] > max_v) max_v = (double)votes[ch][c];

        for (int c = 0; c < N_CLASSES; c++)
            log_post[c] += ((double)votes[ch][c] - max_v) / temp;

        temp *= temp_decay;  /* sharpen for next stage */
    }
    return argmax_d(log_post, N_CLASSES);
}

/* 4. Multiplicative gating: each channel gates the running accumulator.
 * Pure integer: acc[c] = (acc[c] * votes[c]) >> shift per stage. */
static int series_multiplicative_gate(const uint32_t *px, const uint32_t *hg,
                                       const uint32_t *vg) {
    /* Start with channel 1 votes */
    uint64_t acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = (uint64_t)vg[c] + 1;  /* +1 avoids zero */

    /* Gate with channel 2: multiply and renormalize */
    uint64_t max_acc = 1;
    for (int c = 0; c < N_CLASSES; c++) {
        acc[c] *= ((uint64_t)px[c] + 1);
        if (acc[c] > max_acc) max_acc = acc[c];
    }
    /* Renormalize to prevent overflow */
    for (int c = 0; c < N_CLASSES; c++)
        acc[c] = acc[c] * 10000 / max_acc;

    /* Gate with channel 3 */
    max_acc = 1;
    for (int c = 0; c < N_CLASSES; c++) {
        acc[c] *= ((uint64_t)hg[c] + 1);
        if (acc[c] > max_acc) max_acc = acc[c];
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
    return best;
}

/* 5. Block-interleaved series: process blocks in series, alternating channels.
 * Block k from ch0 → block k from ch1 → block k from ch2 → next block.
 * Each block's lookup is weighted by the current running confidence. */
static int series_block_interleaved(int img_idx) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;  /* uniform prior */

    const uint8_t *sigs[3] = {
        px_test_sigs + (size_t)img_idx * SIG_PAD,
        hg_test_sigs + (size_t)img_idx * SIG_PAD,
        vg_test_sigs + (size_t)img_idx * SIG_PAD
    };
    uint32_t (*hots[3])[N_BVALS][CLS_PAD] = {px_hot, hg_hot, vg_hot};
    uint8_t bgs[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};

    for (int k = 0; k < N_BLOCKS; k++) {
        for (int ch = 0; ch < 3; ch++) {
            uint8_t bv = sigs[ch][k];
            if (bv == bgs[ch]) continue;

            /* This block's likelihood: hot[k][bv][c] for each class */
            const uint32_t *h = hots[ch][k][bv];

            /* Bayesian update: acc[c] *= (h[c] + 1) */
            double max_a = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                acc[c] *= ((double)h[c] + 0.5);
                if (acc[c] > max_a) max_a = acc[c];
            }
            /* Renormalize to prevent overflow/underflow */
            if (max_a > 1e10) {
                for (int c = 0; c < N_CLASSES; c++) acc[c] /= max_a;
            }
        }
    }
    return argmax_d(acc, N_CLASSES);
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
    printf("=== SSTT Series Pipeline: Bayesian Channel Cascade (%s) ===\n\n", dataset);

    printf("Loading and preprocessing...\n");
    load_data(data_dir); init_all(); compute_all_sigs();
    build_hot_map(px_train_sigs, px_hot);
    build_hot_map(hg_train_sigs, hg_hot);
    build_hot_map(vg_train_sigs, vg_hot);
    double t1 = now_sec();
    printf("  Done (%.2f sec)\n\n", t1 - t0);

    /* --- Baseline: additive --- */
    int add_correct = 0;
    for (int i = 0; i < TEST_N; i++) {
        uint32_t px[CLS_PAD], hg[CLS_PAD], vg[CLS_PAD];
        get_channel_votes(i, px, hg, vg);
        uint32_t f[CLS_PAD];
        for (int c = 0; c < CLS_PAD; c++) f[c] = px[c] + hg[c] + vg[c];
        int best = 0;
        for (int c = 1; c < N_CLASSES; c++) if (f[c] > f[best]) best = c;
        if (best == test_labels[i]) add_correct++;
    }
    printf("  Additive baseline: %.2f%%\n\n", 100.0 * add_correct / TEST_N);

    /* --- Strategy 1: Bayesian product, sweep temperatures --- */
    printf("--- 1. Bayesian Product (Naive Bayes) ---\n");
    printf("  P(c|all) ∝ P(px|c) × P(vg|c) × P(hg|c)\n\n");
    double temps[] = {100, 500, 1000, 5000, 10000, 50000};
    int n_temps = 6;
    printf("  %-10s | %-10s\n", "Temp", "Accuracy");
    printf("  -----------+----------\n");
    for (int ti = 0; ti < n_temps; ti++) {
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            uint32_t px[CLS_PAD], hg[CLS_PAD], vg[CLS_PAD];
            get_channel_votes(i, px, hg, vg);
            if (series_bayesian(px, hg, vg, temps[ti]) == test_labels[i]) correct++;
        }
        printf("  %8.0f  | %7.2f%%\n", temps[ti], 100.0 * correct / TEST_N);
    }

    /* --- Strategy 2: Sequential update, sweep channel orderings --- */
    printf("\n--- 2. Sequential Bayesian Update (order matters) ---\n");
    printf("  Each channel updates the posterior. Later channels see sharper priors.\n\n");
    int orderings[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    const char *ch_names[] = {"px","hg","vg"};
    double seq_temp[3] = {5000, 5000, 5000};
    printf("  %-20s | %-10s\n", "Order", "Accuracy");
    printf("  ---------------------+----------\n");
    for (int oi = 0; oi < 6; oi++) {
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            uint32_t pvotes[3][CLS_PAD];
            get_channel_votes(i, pvotes[0], pvotes[1], pvotes[2]);
            const uint32_t *ordered[3] = {
                pvotes[orderings[oi][0]],
                pvotes[orderings[oi][1]],
                pvotes[orderings[oi][2]]
            };
            if (series_sequential(ordered, seq_temp) == test_labels[i]) correct++;
        }
        printf("  %s → %s → %s          | %7.2f%%\n",
               ch_names[orderings[oi][0]],
               ch_names[orderings[oi][1]],
               ch_names[orderings[oi][2]],
               100.0 * correct / TEST_N);
    }

    /* --- Strategy 3: Log-likelihood with progressive temperature --- */
    printf("\n--- 3. Log-Likelihood with Progressive Temperature ---\n");
    printf("  Temperature decreases per stage → later channels sharper.\n\n");
    double base_temps[] = {10000, 5000, 2000};
    double decays[] = {0.5, 0.3, 0.7};
    printf("  %-8s %-8s | %-10s\n", "Base T", "Decay", "Accuracy");
    printf("  ---------+---------+----------\n");
    for (int bi = 0; bi < 3; bi++) {
        for (int di = 0; di < 3; di++) {
            int correct = 0;
            for (int i = 0; i < TEST_N; i++) {
                uint32_t pvotes[3][CLS_PAD];
                get_channel_votes(i, pvotes[0], pvotes[1], pvotes[2]);
                /* Best order from strategy 2 (vg first) */
                const uint32_t *ordered[3] = {pvotes[2], pvotes[0], pvotes[1]};
                if (series_log_progressive(ordered, base_temps[bi], decays[di])
                    == test_labels[i]) correct++;
            }
            printf("  %7.0f  %5.1f    | %7.2f%%\n",
                   base_temps[bi], decays[di], 100.0 * correct / TEST_N);
        }
    }

    /* --- Strategy 4: Multiplicative gating (integer, no FP) --- */
    printf("\n--- 4. Multiplicative Gating (integer pipeline) ---\n");
    printf("  acc *= (votes + 1), renormalize per stage\n\n");
    {
        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            uint32_t px[CLS_PAD], hg[CLS_PAD], vg[CLS_PAD];
            get_channel_votes(i, px, hg, vg);
            if (series_multiplicative_gate(px, hg, vg) == test_labels[i]) correct++;
        }
        printf("  Accuracy: %.2f%%\n", 100.0 * correct / TEST_N);
    }

    /* --- Strategy 5: Block-interleaved Bayesian --- */
    printf("\n--- 5. Block-Interleaved Bayesian ---\n");
    printf("  Process one block at a time, all 3 channels, update posterior per block.\n");
    printf("  252 blocks × 3 channels = 756 sequential Bayesian updates.\n\n");
    {
        int correct = 0;
        double ts0 = now_sec();
        for (int i = 0; i < TEST_N; i++)
            if (series_block_interleaved(i) == test_labels[i]) correct++;
        double ts1 = now_sec();
        printf("  Accuracy: %.2f%% (%.0f ns/query)\n",
               100.0 * correct / TEST_N, (ts1-ts0)*1e9/TEST_N);
    }

    printf("\n=== SUMMARY ===\n");
    printf("  Additive baseline: %.2f%%\n", 100.0 * add_correct / TEST_N);
    printf("  See tables above for all series strategies.\n");
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
