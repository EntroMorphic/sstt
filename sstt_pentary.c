/*
 * sstt_pentary.c — Pentary Quantization: Breaking the Trit Bottleneck
 *
 * The trit {-1, 0, +1} collapses the gray zone (85-170) to zero.
 * The pentary {-2, -1, 0, +1, +2} splits it: trending-dark vs trending-light.
 * Faint features (thin bars, loop closures) that trits destroy are preserved.
 *
 * Block encoding: 3 pentary values → 5^3 = 125 block values (vs 27 for ternary).
 * Hot map: 252 × 125 × 16 = ~2MB per channel (vs 435KB). Still L2/L3 resident.
 * Bayesian series: same block-interleaved update, just wider address space.
 *
 * Also tests multi-scale blocks (1+2+3 pixel blocks simultaneously).
 *
 * Build: make sstt_pentary  (after: make mnist)
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
#define N_CLASSES   10
#define CLS_PAD     16

/* Pentary quantization levels */
#define PENT_LEVELS 5
#define PENT_NEG2   (-2)
#define PENT_NEG1   (-1)
#define PENT_ZERO   0
#define PENT_POS1   1
#define PENT_POS2   2

/* 2D blocks */
#define BLKS_PER_ROW 9
#define N_BLOCKS    252     /* 9 × 28 */
#define BLK_W       3

/* Block values */
#define TERN_BVALS  27      /* 3^3 */
#define PENT_BVALS  125     /* 5^3 */
#define SIG_PAD     256     /* padded for alignment */

/* Background values */
#define TERN_BG_PX  0       /* block_encode_tern(-1,-1,-1) */
#define TERN_BG_GR  13      /* block_encode_tern(0,0,0) */
/* For pentary: bg_pixel = pent_encode(-2,-2,-2) = 0 */
/* bg_grad = pent_encode(0,0,0) = 62 (middle of 125) */
#define PENT_BG_PX  0
#define PENT_BG_GR  62

/* 1-pixel blocks for multi-scale */
#define N_BLOCKS_1PX  784
#define BLOCKS_1PX_PAD 800

/* 2-pixel blocks for multi-scale */
#define N_BLOCKS_2PX  (13 * IMG_H)   /* 13 pairs per row × 28 rows = 364 */
#define PENT_2PX_BVALS 25  /* 5^2 */
#define BLOCKS_2PX_PAD 384

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Data ---------- */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;

/* Ternary images (for comparison baseline) */
static int8_t *tern_train, *tern_test;

/* Pentary images */
static int8_t *pent_train, *pent_test;

/* Gradient images (ternary and pentary) */
static int8_t *tern_hg_train, *tern_hg_test, *tern_vg_train, *tern_vg_test;
static int8_t *pent_hg_train, *pent_hg_test, *pent_vg_train, *pent_vg_test;

/* ---------- IDX loader ---------- */
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

/* ---------- Quantization ---------- */

/* Ternary: standard thresholds */
static inline int8_t quant_tern(uint8_t p) {
    if (p > 170) return 1;
    if (p < 85) return -1;
    return 0;
}

/* Pentary: 5 levels with configurable thresholds */
static int pent_t1 = 64, pent_t2 = 128, pent_t3 = 192, pent_t4 = 224;

static inline int8_t quant_pent(uint8_t p) {
    if (p >= pent_t4) return 2;
    if (p >= pent_t3) return 1;
    if (p >= pent_t2) return 0;
    if (p >= pent_t1) return -1;
    return -2;
}

static void quantize_all_tern(void) {
    tern_train = malloc((size_t)TRAIN_N * PIXELS);
    tern_test  = malloc((size_t)TEST_N  * PIXELS);
    for (int i = 0; i < TRAIN_N; i++)
        for (int j = 0; j < PIXELS; j++)
            tern_train[i * PIXELS + j] = quant_tern(raw_train_img[i * PIXELS + j]);
    for (int i = 0; i < TEST_N; i++)
        for (int j = 0; j < PIXELS; j++)
            tern_test[i * PIXELS + j] = quant_tern(raw_test_img[i * PIXELS + j]);
}

static void quantize_all_pent(void) {
    pent_train = malloc((size_t)TRAIN_N * PIXELS);
    pent_test  = malloc((size_t)TEST_N  * PIXELS);
    for (int i = 0; i < TRAIN_N; i++)
        for (int j = 0; j < PIXELS; j++)
            pent_train[i * PIXELS + j] = quant_pent(raw_train_img[i * PIXELS + j]);
    for (int i = 0; i < TEST_N; i++)
        for (int j = 0; j < PIXELS; j++)
            pent_test[i * PIXELS + j] = quant_pent(raw_test_img[i * PIXELS + j]);
}

/* ---------- Gradient (pentary clamp to {-2..+2}) ---------- */

static inline int8_t clamp_tern(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

static inline int8_t clamp_pent(int v) {
    if (v >= 2) return 2;
    if (v == 1) return 1;
    if (v == 0) return 0;
    if (v == -1) return -1;
    return -2;
}

static void compute_gradients(const int8_t *src, int8_t *hg, int8_t *vg,
                               int n, int8_t (*clamp_fn)(int)) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = src + (size_t)i * PIXELS;
        int8_t *h = hg + (size_t)i * PIXELS;
        int8_t *v = vg + (size_t)i * PIXELS;
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W - 1; x++)
                h[y * IMG_W + x] = clamp_fn(img[y * IMG_W + x + 1] - img[y * IMG_W + x]);
            h[y * IMG_W + IMG_W - 1] = 0;
        }
        for (int y = 0; y < IMG_H - 1; y++)
            for (int x = 0; x < IMG_W; x++)
                v[y * IMG_W + x] = clamp_fn(img[(y + 1) * IMG_W + x] - img[y * IMG_W + x]);
        memset(v + (IMG_H - 1) * IMG_W, 0, IMG_W);
    }
}

/* ---------- Block encoding ---------- */

/* Ternary: (t+1)*9 + (t+1)*3 + (t+1), range [0,26] */
static inline uint8_t block_enc_tern(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

/* Pentary: (p+2)*25 + (p+2)*5 + (p+2), range [0,124] */
static inline uint8_t block_enc_pent(int8_t p0, int8_t p1, int8_t p2) {
    return (uint8_t)((p0 + 2) * 25 + (p1 + 2) * 5 + (p2 + 2));
}

/* 2-pixel pentary block: (p+2)*5 + (p+2), range [0,24] */
static inline uint8_t block_enc_pent2(int8_t p0, int8_t p1) {
    return (uint8_t)((p0 + 2) * 5 + (p1 + 2));
}

/* 1-pixel pentary block: (p+2), range [0,4] */
static inline uint8_t block_enc_pent1(int8_t p0) {
    return (uint8_t)(p0 + 2);
}

/* ---------- Signature computation ---------- */

static void compute_sigs_3px(const int8_t *data, uint8_t *sigs, int n,
                              uint8_t (*enc)(int8_t, int8_t, int8_t)) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PIXELS;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int base = y * IMG_W + s * BLK_W;
                sig[y * BLKS_PER_ROW + s] = enc(img[base], img[base + 1], img[base + 2]);
            }
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

static void compute_sigs_1px(const int8_t *data, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PIXELS;
        uint8_t *sig = sigs + (size_t)i * BLOCKS_1PX_PAD;
        for (int j = 0; j < PIXELS; j++)
            sig[j] = block_enc_pent1(img[j]);
        memset(sig + PIXELS, 0xFF, BLOCKS_1PX_PAD - PIXELS);
    }
}

static void compute_sigs_2px(const int8_t *data, uint8_t *sigs, int n) {
    /* 2-pixel horizontal blocks: 13 pairs per row (pixels 0-1, 2-3, ..., 24-25, 26-27) */
    /* Actually: 14 non-overlapping pairs per row (0-1, 2-3, ..., 26-27) = 14 */
    /* Let's use 13 to avoid pixel 27 alignment issues: pairs at 0,2,4,...,24 = 13 */
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PIXELS;
        uint8_t *sig = sigs + (size_t)i * BLOCKS_2PX_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < 13; s++) {
                int base = y * IMG_W + s * 2;
                sig[y * 13 + s] = block_enc_pent2(img[base], img[base + 1]);
            }
        memset(sig + N_BLOCKS_2PX, 0xFF, BLOCKS_2PX_PAD - N_BLOCKS_2PX);
    }
}

/* ---------- Hot map building ---------- */

static void build_hot(const uint8_t *sigs, int n_pos, int n_bvals,
                       uint32_t *hot, int sig_stride) {
    memset(hot, 0, sizeof(uint32_t) * (size_t)n_pos * n_bvals * CLS_PAD);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = sigs + (size_t)i * sig_stride;
        for (int k = 0; k < n_pos; k++)
            hot[(size_t)k * n_bvals * CLS_PAD + (size_t)sig[k] * CLS_PAD + lbl]++;
    }
}

/* ---------- Bayesian update ---------- */

static inline void bayes_update(double *acc, const uint32_t *h, int n_cls) {
    double mx = 0;
    for (int c = 0; c < n_cls; c++) {
        acc[c] *= ((double)h[c] + 0.5);
        if (acc[c] > mx) mx = acc[c];
    }
    if (mx > 1e10)
        for (int c = 0; c < n_cls; c++) acc[c] /= mx;
}

/* ---------- Block-interleaved Bayesian classifier ---------- */

static int classify_bayesian(const uint8_t *sigs[], int n_ch,
                              const uint32_t *hots[], int n_pos[],
                              int n_bvals_arr[], int sig_strides[],
                              uint8_t bgs[]) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;

    /* Find max n_pos across channels for interleaving */
    int max_pos = 0;
    for (int ch = 0; ch < n_ch; ch++)
        if (n_pos[ch] > max_pos) max_pos = n_pos[ch];

    /* Interleave all channels at each position */
    for (int k = 0; k < max_pos; k++) {
        for (int ch = 0; ch < n_ch; ch++) {
            if (k >= n_pos[ch]) continue;
            uint8_t bv = sigs[ch][k];
            if (bv == bgs[ch]) continue;
            int nbv = n_bvals_arr[ch];
            const uint32_t *h = hots[ch] + (size_t)k * nbv * CLS_PAD + (size_t)bv * CLS_PAD;
            bayes_update(acc, h, N_CLASSES);
        }
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
            buf[len] = '/'; buf[len + 1] = '\0';
            data_dir = buf;
        }
    }

    const char *ds = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Pentary: Breaking the Trit Bottleneck (%s) ===\n\n", ds);

    printf("Loading...\n");
    load_data(data_dir);
    double t1 = now_sec();

    /* ============================================================
     *  Test 1: Ternary baseline (3-channel Bayesian, 27 block values)
     * ============================================================ */
    printf("--- Test 1: Ternary Baseline (3-chan Bayesian, 27 bvals) ---\n");
    quantize_all_tern();
    tern_hg_train = malloc((size_t)TRAIN_N * PIXELS);
    tern_hg_test  = malloc((size_t)TEST_N  * PIXELS);
    tern_vg_train = malloc((size_t)TRAIN_N * PIXELS);
    tern_vg_test  = malloc((size_t)TEST_N  * PIXELS);
    compute_gradients(tern_train, tern_hg_train, tern_vg_train, TRAIN_N, clamp_tern);
    compute_gradients(tern_test,  tern_hg_test,  tern_vg_test,  TEST_N,  clamp_tern);

    /* Build ternary sigs and hot maps */
    uint8_t *t_px_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    uint8_t *t_px_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    uint8_t *t_hg_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    uint8_t *t_hg_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    uint8_t *t_vg_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    uint8_t *t_vg_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    compute_sigs_3px(tern_train,    t_px_sigs_tr, TRAIN_N, block_enc_tern);
    compute_sigs_3px(tern_test,     t_px_sigs_te, TEST_N,  block_enc_tern);
    compute_sigs_3px(tern_hg_train, t_hg_sigs_tr, TRAIN_N, block_enc_tern);
    compute_sigs_3px(tern_hg_test,  t_hg_sigs_te, TEST_N,  block_enc_tern);
    compute_sigs_3px(tern_vg_train, t_vg_sigs_tr, TRAIN_N, block_enc_tern);
    compute_sigs_3px(tern_vg_test,  t_vg_sigs_te, TEST_N,  block_enc_tern);

    uint32_t *t_px_hot = aligned_alloc(32, (size_t)N_BLOCKS * TERN_BVALS * CLS_PAD * 4);
    uint32_t *t_hg_hot = aligned_alloc(32, (size_t)N_BLOCKS * TERN_BVALS * CLS_PAD * 4);
    uint32_t *t_vg_hot = aligned_alloc(32, (size_t)N_BLOCKS * TERN_BVALS * CLS_PAD * 4);
    build_hot(t_px_sigs_tr, N_BLOCKS, TERN_BVALS, t_px_hot, SIG_PAD);
    build_hot(t_hg_sigs_tr, N_BLOCKS, TERN_BVALS, t_hg_hot, SIG_PAD);
    build_hot(t_vg_sigs_tr, N_BLOCKS, TERN_BVALS, t_vg_hot, SIG_PAD);

    int tern_correct = 0;
    double tt0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        const uint8_t *sigs[3] = {
            t_px_sigs_te + (size_t)i * SIG_PAD,
            t_hg_sigs_te + (size_t)i * SIG_PAD,
            t_vg_sigs_te + (size_t)i * SIG_PAD
        };
        const uint32_t *hots[3] = {t_px_hot, t_hg_hot, t_vg_hot};
        int npos[3] = {N_BLOCKS, N_BLOCKS, N_BLOCKS};
        int nbvals[3] = {TERN_BVALS, TERN_BVALS, TERN_BVALS};
        int strides[3] = {SIG_PAD, SIG_PAD, SIG_PAD};
        uint8_t bgs[3] = {TERN_BG_PX, TERN_BG_GR, TERN_BG_GR};
        if (classify_bayesian(sigs, 3, hots, npos, nbvals, strides, bgs) == test_labels[i])
            tern_correct++;
    }
    double tt1 = now_sec();
    printf("  Accuracy: %.2f%% (%.0f ns/query)\n",
           100.0 * tern_correct / TEST_N, (tt1 - tt0) * 1e9 / TEST_N);
    printf("  Hot map size: %zu KB per channel\n\n",
           (size_t)N_BLOCKS * TERN_BVALS * CLS_PAD * 4 / 1024);

    /* ============================================================
     *  Test 2: Pentary 3-pixel blocks (3-chan Bayesian, 125 bvals)
     * ============================================================ */
    printf("--- Test 2: Pentary 3-Pixel Blocks (3-chan, 125 bvals) ---\n");

    /* Sweep pentary thresholds */
    int thresh_sets[][4] = {
        {64, 128, 192, 224},    /* Wide gray zone */
        {51, 128, 178, 230},    /* Balanced */
        {85, 128, 170, 220},    /* Split ternary zones in half */
        {40, 100, 180, 230},    /* Aggressive dark/light capture */
        {70, 128, 185, 215},    /* Narrow gray zone */
    };
    int n_thresh = 5;
    const char *thresh_names[] = {
        "64/128/192/224",
        "51/128/178/230",
        "85/128/170/220",
        "40/100/180/230",
        "70/128/185/215"
    };

    printf("  %-22s | %-10s | %-8s\n", "Thresholds", "Accuracy", "Size/ch");
    printf("  -----------------------+------------+---------\n");

    double best_pent_acc = 0;
    int best_thresh_idx = 0;

    for (int ti = 0; ti < n_thresh; ti++) {
        pent_t1 = thresh_sets[ti][0];
        pent_t2 = thresh_sets[ti][1];
        pent_t3 = thresh_sets[ti][2];
        pent_t4 = thresh_sets[ti][3];

        quantize_all_pent();
        pent_hg_train = malloc((size_t)TRAIN_N * PIXELS);
        pent_hg_test  = malloc((size_t)TEST_N  * PIXELS);
        pent_vg_train = malloc((size_t)TRAIN_N * PIXELS);
        pent_vg_test  = malloc((size_t)TEST_N  * PIXELS);
        compute_gradients(pent_train, pent_hg_train, pent_vg_train, TRAIN_N, clamp_pent);
        compute_gradients(pent_test,  pent_hg_test,  pent_vg_test,  TEST_N,  clamp_pent);

        uint8_t *p_px_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        uint8_t *p_px_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
        uint8_t *p_hg_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        uint8_t *p_hg_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
        uint8_t *p_vg_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        uint8_t *p_vg_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
        compute_sigs_3px(pent_train,    p_px_tr, TRAIN_N, block_enc_pent);
        compute_sigs_3px(pent_test,     p_px_te, TEST_N,  block_enc_pent);
        compute_sigs_3px(pent_hg_train, p_hg_tr, TRAIN_N, block_enc_pent);
        compute_sigs_3px(pent_hg_test,  p_hg_te, TEST_N,  block_enc_pent);
        compute_sigs_3px(pent_vg_train, p_vg_tr, TRAIN_N, block_enc_pent);
        compute_sigs_3px(pent_vg_test,  p_vg_te, TEST_N,  block_enc_pent);

        uint32_t *p_px_hot = aligned_alloc(32, (size_t)N_BLOCKS * PENT_BVALS * CLS_PAD * 4);
        uint32_t *p_hg_hot = aligned_alloc(32, (size_t)N_BLOCKS * PENT_BVALS * CLS_PAD * 4);
        uint32_t *p_vg_hot = aligned_alloc(32, (size_t)N_BLOCKS * PENT_BVALS * CLS_PAD * 4);
        build_hot(p_px_tr, N_BLOCKS, PENT_BVALS, p_px_hot, SIG_PAD);
        build_hot(p_hg_tr, N_BLOCKS, PENT_BVALS, p_hg_hot, SIG_PAD);
        build_hot(p_vg_tr, N_BLOCKS, PENT_BVALS, p_vg_hot, SIG_PAD);

        int correct = 0;
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *sigs[3] = {
                p_px_te + (size_t)i * SIG_PAD,
                p_hg_te + (size_t)i * SIG_PAD,
                p_vg_te + (size_t)i * SIG_PAD
            };
            const uint32_t *hots[3] = {p_px_hot, p_hg_hot, p_vg_hot};
            int npos[3] = {N_BLOCKS, N_BLOCKS, N_BLOCKS};
            int nbvals[3] = {PENT_BVALS, PENT_BVALS, PENT_BVALS};
            int strides[3] = {SIG_PAD, SIG_PAD, SIG_PAD};
            uint8_t bgs[3] = {PENT_BG_PX, PENT_BG_GR, PENT_BG_GR};
            if (classify_bayesian(sigs, 3, hots, npos, nbvals, strides, bgs) == test_labels[i])
                correct++;
        }

        double acc = 100.0 * correct / TEST_N;
        printf("  %-22s | %7.2f%%  | %4zu KB\n",
               thresh_names[ti], acc,
               (size_t)N_BLOCKS * PENT_BVALS * CLS_PAD * 4 / 1024);

        if (acc > best_pent_acc) {
            best_pent_acc = acc;
            best_thresh_idx = ti;
        }

        free(p_px_tr); free(p_px_te);
        free(p_hg_tr); free(p_hg_te);
        free(p_vg_tr); free(p_vg_te);
        free(p_px_hot); free(p_hg_hot); free(p_vg_hot);
        free(pent_hg_train); free(pent_hg_test);
        free(pent_vg_train); free(pent_vg_test);
    }

    printf("\n  Best pentary: %s → %.2f%% (vs ternary %.2f%%: %+.2f pp)\n\n",
           thresh_names[best_thresh_idx], best_pent_acc,
           100.0 * tern_correct / TEST_N,
           best_pent_acc - 100.0 * tern_correct / TEST_N);

    /* ============================================================
     *  Test 3: Multi-scale pentary (1+2+3 pixel blocks combined)
     * ============================================================ */
    printf("--- Test 3: Multi-Scale Pentary (1px + 2px + 3px blocks) ---\n");

    /* Use best threshold set */
    pent_t1 = thresh_sets[best_thresh_idx][0];
    pent_t2 = thresh_sets[best_thresh_idx][1];
    pent_t3 = thresh_sets[best_thresh_idx][2];
    pent_t4 = thresh_sets[best_thresh_idx][3];
    quantize_all_pent();

    /* 3px pentary sigs (pixel channel only for simplicity) */
    uint8_t *ms3_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    uint8_t *ms3_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    compute_sigs_3px(pent_train, ms3_tr, TRAIN_N, block_enc_pent);
    compute_sigs_3px(pent_test,  ms3_te, TEST_N,  block_enc_pent);
    uint32_t *ms3_hot = aligned_alloc(32, (size_t)N_BLOCKS * PENT_BVALS * CLS_PAD * 4);
    build_hot(ms3_tr, N_BLOCKS, PENT_BVALS, ms3_hot, SIG_PAD);

    /* 2px pentary sigs */
    uint8_t *ms2_tr = aligned_alloc(32, (size_t)TRAIN_N * BLOCKS_2PX_PAD);
    uint8_t *ms2_te = aligned_alloc(32, (size_t)TEST_N  * BLOCKS_2PX_PAD);
    compute_sigs_2px(pent_train, ms2_tr, TRAIN_N);
    compute_sigs_2px(pent_test,  ms2_te, TEST_N);
    uint32_t *ms2_hot = aligned_alloc(32, (size_t)N_BLOCKS_2PX * PENT_2PX_BVALS * CLS_PAD * 4);
    build_hot(ms2_tr, N_BLOCKS_2PX, PENT_2PX_BVALS, ms2_hot, BLOCKS_2PX_PAD);

    /* 1px pentary sigs */
    uint8_t *ms1_tr = aligned_alloc(32, (size_t)TRAIN_N * BLOCKS_1PX_PAD);
    uint8_t *ms1_te = aligned_alloc(32, (size_t)TEST_N  * BLOCKS_1PX_PAD);
    compute_sigs_1px(pent_train, ms1_tr, TRAIN_N);
    compute_sigs_1px(pent_test,  ms1_te, TEST_N);
    uint32_t *ms1_hot = aligned_alloc(32, (size_t)N_BLOCKS_1PX * PENT_LEVELS * CLS_PAD * 4);
    build_hot(ms1_tr, N_BLOCKS_1PX, PENT_LEVELS, ms1_hot, BLOCKS_1PX_PAD);

    /* Classify: single-scale and multi-scale */
    int correct_3px = 0, correct_2px = 0, correct_1px = 0, correct_ms = 0;

    for (int i = 0; i < TEST_N; i++) {
        /* 3px only */
        {
            const uint8_t *sigs[1] = {ms3_te + (size_t)i * SIG_PAD};
            const uint32_t *hots[1] = {ms3_hot};
            int npos[1] = {N_BLOCKS}; int nbv[1] = {PENT_BVALS};
            int str[1] = {SIG_PAD}; uint8_t bg[1] = {PENT_BG_PX};
            if (classify_bayesian(sigs, 1, hots, npos, nbv, str, bg) == test_labels[i])
                correct_3px++;
        }
        /* 2px only */
        {
            const uint8_t *sigs[1] = {ms2_te + (size_t)i * BLOCKS_2PX_PAD};
            const uint32_t *hots[1] = {ms2_hot};
            int npos[1] = {N_BLOCKS_2PX}; int nbv[1] = {PENT_2PX_BVALS};
            int str[1] = {BLOCKS_2PX_PAD};
            uint8_t bg[1] = {(uint8_t)((0 + 2) * 5 + (0 + 2))}; /* pent_encode2(0,0)=12 */
            if (classify_bayesian(sigs, 1, hots, npos, nbv, str, bg) == test_labels[i])
                correct_2px++;
        }
        /* 1px only */
        {
            const uint8_t *sigs[1] = {ms1_te + (size_t)i * BLOCKS_1PX_PAD};
            const uint32_t *hots[1] = {ms1_hot};
            int npos[1] = {N_BLOCKS_1PX}; int nbv[1] = {PENT_LEVELS};
            int str[1] = {BLOCKS_1PX_PAD};
            uint8_t bg[1] = {2}; /* pent_encode1(0)=2 (uncertain pixel) */
            if (classify_bayesian(sigs, 1, hots, npos, nbv, str, bg) == test_labels[i])
                correct_1px++;
        }
        /* Multi-scale: 1px + 2px + 3px combined */
        {
            const uint8_t *sigs[3] = {
                ms1_te + (size_t)i * BLOCKS_1PX_PAD,
                ms2_te + (size_t)i * BLOCKS_2PX_PAD,
                ms3_te + (size_t)i * SIG_PAD
            };
            const uint32_t *hots[3] = {ms1_hot, ms2_hot, ms3_hot};
            int npos[3] = {N_BLOCKS_1PX, N_BLOCKS_2PX, N_BLOCKS};
            int nbv[3] = {PENT_LEVELS, PENT_2PX_BVALS, PENT_BVALS};
            int str[3] = {BLOCKS_1PX_PAD, BLOCKS_2PX_PAD, SIG_PAD};
            uint8_t bg[3] = {2, 12, PENT_BG_PX};
            if (classify_bayesian(sigs, 3, hots, npos, nbv, str, bg) == test_labels[i])
                correct_ms++;
        }
    }

    printf("  1-pixel pentary:   %.2f%% (%d blocks × %d values = %zu KB)\n",
           100.0 * correct_1px / TEST_N, N_BLOCKS_1PX, PENT_LEVELS,
           (size_t)N_BLOCKS_1PX * PENT_LEVELS * CLS_PAD * 4 / 1024);
    printf("  2-pixel pentary:   %.2f%% (%d blocks × %d values = %zu KB)\n",
           100.0 * correct_2px / TEST_N, N_BLOCKS_2PX, PENT_2PX_BVALS,
           (size_t)N_BLOCKS_2PX * PENT_2PX_BVALS * CLS_PAD * 4 / 1024);
    printf("  3-pixel pentary:   %.2f%% (%d blocks × %d values = %zu KB)\n",
           100.0 * correct_3px / TEST_N, N_BLOCKS, PENT_BVALS,
           (size_t)N_BLOCKS * PENT_BVALS * CLS_PAD * 4 / 1024);
    printf("  Multi-scale (1+2+3): %.2f%%\n\n",
           100.0 * correct_ms / TEST_N);

    /* ============================================================
     *  Summary
     * ============================================================ */
    printf("=== SUMMARY ===\n");
    printf("  Ternary 3-chan Bayesian:     %.2f%%  (baseline)\n",
           100.0 * tern_correct / TEST_N);
    printf("  Best pentary 3-chan:         %.2f%%  (%+.2f pp)\n",
           best_pent_acc, best_pent_acc - 100.0 * tern_correct / TEST_N);
    printf("  Pentary multi-scale (1+2+3): %.2f%%  (%+.2f pp)\n",
           100.0 * correct_ms / TEST_N,
           100.0 * correct_ms / TEST_N - 100.0 * tern_correct / TEST_N);
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    /* Cleanup */
    free(tern_train); free(tern_test);
    free(pent_train); free(pent_test);
    free(tern_hg_train); free(tern_hg_test);
    free(tern_vg_train); free(tern_vg_test);
    free(t_px_sigs_tr); free(t_px_sigs_te);
    free(t_hg_sigs_tr); free(t_hg_sigs_te);
    free(t_vg_sigs_tr); free(t_vg_sigs_te);
    free(t_px_hot); free(t_hg_hot); free(t_vg_hot);
    free(ms3_tr); free(ms3_te); free(ms3_hot);
    free(ms2_tr); free(ms2_te); free(ms2_hot);
    free(ms1_tr); free(ms1_te); free(ms1_hot);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);

    return 0;
}
