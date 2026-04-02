/*
 * sstt_transitions.c — Quantized Inter-Block Transitions
 *
 * The space between block vectors: how the ternary pattern changes
 * from one spatial position to the next.
 *
 * For adjacent blocks (k, k+1) with trits (a0,a1,a2) and (b0,b1,b2):
 *   transition_trit[j] = clamp(b[j] - a[j])
 *   transition_value = block_encode(transition_trits)
 *
 * Two transition channels:
 *   H-trans: horizontal transitions (8 per row × 28 rows = 224)
 *   V-trans: vertical transitions (9 per strip × 27 rows = 243)
 *
 * These capture spatial grammar — the sequence of pattern changes
 * that define shape structure. Added to the Bayesian series pipeline.
 *
 * Build: make sstt_transitions  (after: make mnist)
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

/* Transition layout */
#define H_TRANS_PER_ROW 8       /* 9 strips - 1 = 8 horizontal transitions */
#define N_HTRANS    (H_TRANS_PER_ROW * IMG_H)   /* 224 */
#define V_TRANS_PER_COL 27      /* 28 rows - 1 = 27 vertical transitions */
#define N_VTRANS    (BLKS_PER_ROW * V_TRANS_PER_COL)  /* 243 */
#define HTRANS_PAD  256         /* align for AVX2 */
#define VTRANS_PAD  256
/* Background for transitions: block_encode(0,0,0) = 13 (no change) */
#define BG_TRANS    13

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

/* Block hot maps */
static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* Transition signatures and hot maps */
static uint8_t *ht_train_sigs, *ht_test_sigs;   /* horizontal transitions */
static uint8_t *vt_train_sigs, *vt_test_sigs;   /* vertical transitions */
static uint32_t ht_hot[N_HTRANS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vt_hot[N_VTRANS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* --- Standard infrastructure --- */

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

static void build_hot_map(const uint8_t *sigs, int n_pos,
                           uint32_t *hot, int sig_stride) {
    memset(hot, 0, sizeof(uint32_t) * (size_t)n_pos * N_BVALS * CLS_PAD);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = sigs + (size_t)i * sig_stride;
        for (int k = 0; k < n_pos; k++)
            hot[(size_t)k * N_BVALS * CLS_PAD + (size_t)sig[k] * CLS_PAD + lbl]++;
    }
}

/* ================================================================
 *  Transition Signatures
 *
 *  For each pair of adjacent blocks, decode both to trits,
 *  compute the trit-level difference, clamp, re-encode.
 *
 *  Horizontal: between strip s and s+1 in the same row
 *  Vertical: between row y and y+1 at the same strip
 * ================================================================ */

/* Decode block value to 3 trits */
static inline void block_decode(uint8_t bv, int8_t *t0, int8_t *t1, int8_t *t2) {
    *t0 = (int8_t)((bv / 9) - 1);
    *t1 = (int8_t)(((bv / 3) % 3) - 1);
    *t2 = (int8_t)((bv % 3) - 1);
}

/* Compute transition block value from two adjacent block values */
static inline uint8_t transition_encode(uint8_t bv_a, uint8_t bv_b) {
    int8_t a0, a1, a2, b0, b1, b2;
    block_decode(bv_a, &a0, &a1, &a2);
    block_decode(bv_b, &b0, &b1, &b2);
    int8_t d0 = clamp_trit(b0 - a0);
    int8_t d1 = clamp_trit(b1 - a1);
    int8_t d2 = clamp_trit(b2 - a2);
    return block_encode(d0, d1, d2);
}

static void compute_transition_sigs(const uint8_t *block_sigs, int sig_stride,
                                     uint8_t *ht_sigs, uint8_t *vt_sigs, int n) {
    for (int i = 0; i < n; i++) {
        const uint8_t *bsig = block_sigs + (size_t)i * sig_stride;
        uint8_t *ht = ht_sigs + (size_t)i * HTRANS_PAD;
        uint8_t *vt = vt_sigs + (size_t)i * VTRANS_PAD;

        /* Horizontal transitions: between adjacent strips in same row */
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < H_TRANS_PER_ROW; s++) {
                uint8_t bv_left  = bsig[y * BLKS_PER_ROW + s];
                uint8_t bv_right = bsig[y * BLKS_PER_ROW + s + 1];
                ht[y * H_TRANS_PER_ROW + s] = transition_encode(bv_left, bv_right);
            }
        memset(ht + N_HTRANS, 0xFF, HTRANS_PAD - N_HTRANS);

        /* Vertical transitions: between adjacent rows at same strip */
        for (int y = 0; y < V_TRANS_PER_COL; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                uint8_t bv_top = bsig[y * BLKS_PER_ROW + s];
                uint8_t bv_bot = bsig[(y + 1) * BLKS_PER_ROW + s];
                vt[y * BLKS_PER_ROW + s] = transition_encode(bv_top, bv_bot);
            }
        memset(vt + N_VTRANS, 0xFF, VTRANS_PAD - N_VTRANS);
    }
}

/* ================================================================
 *  Classifiers
 * ================================================================ */

static int argmax_d(const double *v, int n) {
    int best = 0;
    for (int c = 1; c < n; c++) if (v[c] > v[best]) best = c;
    return best;
}

/* Bayesian update from a hot map slice */
static inline void bayesian_update(double *acc, const uint32_t *h) {
    double max_a = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        acc[c] *= ((double)h[c] + 0.5);
        if (acc[c] > max_a) max_a = acc[c];
    }
    if (max_a > 1e10)
        for (int c = 0; c < N_CLASSES; c++) acc[c] /= max_a;
}

/* 3-channel Bayesian series (baseline from sstt_series.c) */
static int classify_3chan(int img_idx) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;
    const uint8_t *ps = px_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *hs = hg_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *vs = vg_test_sigs + (size_t)img_idx * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv;
        bv = ps[k]; if (bv != BG_PIXEL)
            bayesian_update(acc, px_hot[k][bv]);
        bv = hs[k]; if (bv != BG_GRAD)
            bayesian_update(acc, hg_hot[k][bv]);
        bv = vs[k]; if (bv != BG_GRAD)
            bayesian_update(acc, vg_hot[k][bv]);
    }
    return argmax_d(acc, N_CLASSES);
}

/* 5-channel Bayesian: 3 original + 2 transition channels */
static int classify_5chan(int img_idx) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;
    const uint8_t *ps = px_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *hs = hg_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *vs = vg_test_sigs + (size_t)img_idx * SIG_PAD;
    const uint8_t *hts = ht_test_sigs + (size_t)img_idx * HTRANS_PAD;
    const uint8_t *vts = vt_test_sigs + (size_t)img_idx * VTRANS_PAD;

    for (int y = 0; y < IMG_H; y++) {
        for (int s = 0; s < BLKS_PER_ROW; s++) {
            int k = y * BLKS_PER_ROW + s;
            uint8_t bv;

            /* Original 3 channels at position k */
            bv = ps[k]; if (bv != BG_PIXEL)
                bayesian_update(acc, px_hot[k][bv]);
            bv = hs[k]; if (bv != BG_GRAD)
                bayesian_update(acc, hg_hot[k][bv]);
            bv = vs[k]; if (bv != BG_GRAD)
                bayesian_update(acc, vg_hot[k][bv]);

            /* Horizontal transition: between strip s-1 and s */
            if (s > 0) {
                int ht_idx = y * H_TRANS_PER_ROW + (s - 1);
                bv = hts[ht_idx]; if (bv != BG_TRANS)
                    bayesian_update(acc,
                        &ht_hot[ht_idx][bv][0]);
            }

            /* Vertical transition: between row y-1 and y */
            if (y > 0) {
                int vt_idx = (y - 1) * BLKS_PER_ROW + s;
                bv = vts[vt_idx]; if (bv != BG_TRANS)
                    bayesian_update(acc,
                        &vt_hot[vt_idx][bv][0]);
            }
        }
    }
    return argmax_d(acc, N_CLASSES);
}

/* Transition-only: just the 2 transition channels */
static int classify_trans_only(int img_idx) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;
    const uint8_t *hts = ht_test_sigs + (size_t)img_idx * HTRANS_PAD;
    const uint8_t *vts = vt_test_sigs + (size_t)img_idx * VTRANS_PAD;

    for (int k = 0; k < N_HTRANS; k++) {
        uint8_t bv = hts[k]; if (bv != BG_TRANS)
            bayesian_update(acc, &ht_hot[k][bv][0]);
    }
    for (int k = 0; k < N_VTRANS; k++) {
        uint8_t bv = vts[k]; if (bv != BG_TRANS)
            bayesian_update(acc, &vt_hot[k][bv][0]);
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
    printf("=== SSTT Transitions: Quantized Inter-Block Space (%s) ===\n\n", dataset);

    printf("Loading and preprocessing...\n");
    load_data(data_dir); init_all(); compute_all_sigs();
    build_hot_map(px_train_sigs, N_BLOCKS, (uint32_t *)px_hot, SIG_PAD);
    build_hot_map(hg_train_sigs, N_BLOCKS, (uint32_t *)hg_hot, SIG_PAD);
    build_hot_map(vg_train_sigs, N_BLOCKS, (uint32_t *)vg_hot, SIG_PAD);

    /* Compute transition signatures */
    printf("Computing transition signatures...\n");
    ht_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * HTRANS_PAD);
    ht_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * HTRANS_PAD);
    vt_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * VTRANS_PAD);
    vt_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * VTRANS_PAD);
    compute_transition_sigs(px_train_sigs, SIG_PAD, ht_train_sigs, vt_train_sigs, TRAIN_N);
    compute_transition_sigs(px_test_sigs,  SIG_PAD, ht_test_sigs,  vt_test_sigs,  TEST_N);

    /* Build transition hot maps */
    build_hot_map(ht_train_sigs, N_HTRANS, (uint32_t *)ht_hot, HTRANS_PAD);
    build_hot_map(vt_train_sigs, N_VTRANS, (uint32_t *)vt_hot, VTRANS_PAD);

    double t1 = now_sec();

    /* Transition stats */
    {
        long bg_ht = 0, bg_vt = 0;
        for (long i = 0; i < (long)TRAIN_N * N_HTRANS; i++)
            if (ht_train_sigs[i / N_HTRANS * HTRANS_PAD + i % N_HTRANS] == BG_TRANS)
                bg_ht++;
        for (long i = 0; i < (long)TRAIN_N * N_VTRANS; i++)
            if (vt_train_sigs[i / N_VTRANS * VTRANS_PAD + i % N_VTRANS] == BG_TRANS)
                bg_vt++;
        printf("  H-transitions: %d positions, %.1f%% background (no change)\n",
               N_HTRANS, 100.0 * bg_ht / ((long)TRAIN_N * N_HTRANS));
        printf("  V-transitions: %d positions, %.1f%% background (no change)\n",
               N_VTRANS, 100.0 * bg_vt / ((long)TRAIN_N * N_VTRANS));
        printf("  Hot maps: H-trans %zu KB, V-trans %zu KB\n",
               sizeof(ht_hot)/1024, sizeof(vt_hot)/1024);
    }
    printf("  Built (%.2f sec)\n\n", t1 - t0);

    /* --- Test: 3-channel baseline (block-interleaved Bayesian) --- */
    printf("--- 3-Channel Bayesian (baseline) ---\n");
    {
        int correct = 0;
        double ts0 = now_sec();
        for (int i = 0; i < TEST_N; i++)
            if (classify_3chan(i) == test_labels[i]) correct++;
        double ts1 = now_sec();
        printf("  Accuracy: %.2f%% (%.0f ns/query)\n\n",
               100.0 * correct / TEST_N, (ts1-ts0)*1e9/TEST_N);
    }

    /* --- Test: Transition-only (just H-trans + V-trans) --- */
    printf("--- Transition-Only (H-trans + V-trans) ---\n");
    {
        int correct = 0;
        double ts0 = now_sec();
        for (int i = 0; i < TEST_N; i++)
            if (classify_trans_only(i) == test_labels[i]) correct++;
        double ts1 = now_sec();
        printf("  Accuracy: %.2f%% (%.0f ns/query)\n\n",
               100.0 * correct / TEST_N, (ts1-ts0)*1e9/TEST_N);
    }

    /* --- Test: 5-channel (3 original + 2 transitions) --- */
    printf("--- 5-Channel Bayesian (3 original + 2 transitions) ---\n");
    {
        int correct = 0;
        double ts0 = now_sec();
        for (int i = 0; i < TEST_N; i++)
            if (classify_5chan(i) == test_labels[i]) correct++;
        double ts1 = now_sec();
        printf("  Accuracy: %.2f%% (%.0f ns/query)\n\n",
               100.0 * correct / TEST_N, (ts1-ts0)*1e9/TEST_N);
    }

    printf("=== SUMMARY ===\n");
    printf("  Model size: original 3×%zu KB + transitions %zu KB + %zu KB = %zu KB total\n",
           sizeof(px_hot)/1024,
           sizeof(ht_hot)/1024, sizeof(vt_hot)/1024,
           (sizeof(px_hot)*3 + sizeof(ht_hot) + sizeof(vt_hot))/1024);
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(px_train_sigs); free(px_test_sigs);
    free(hg_train_sigs); free(hg_test_sigs);
    free(vg_train_sigs); free(vg_test_sigs);
    free(ht_train_sigs); free(ht_test_sigs);
    free(vt_train_sigs); free(vt_test_sigs);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);
    return 0;
}
