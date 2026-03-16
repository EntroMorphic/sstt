/*
 * sstt_softcascade.c — Soft-Prior Cascade: Information Perihelion
 *
 * The Transition Bayesian (84.69%, 2μs) tells the cascade where to look.
 * The cascade (96.12%, 300s) provides the precise approach.
 *
 * The Bayesian posterior BOOSTS cascade votes for likely classes
 * WITHOUT EXCLUDING any class. boost[c] >= 1 always.
 * The cascade can override the Bayesian when it's wrong (15% of the time),
 * but converges faster when it's right (85% of the time).
 *
 * Architecture:
 *   1. Transition Bayesian → posterior[10] (2μs, pure table lookup)
 *   2. Posterior → boost[c] = 1 + α × softmax(posterior, T)[c] × BOOST_SCALE
 *   3. Pre-compute img_boost[tid] = boost[label[tid]] for all 60K training images
 *   4. 3-channel IG multi-probe voting: votes[tid] += w × img_boost[tid]
 *   5. Top-K → dot product → k=3 majority vote
 *
 * Build: make sstt_softcascade  (after: make mnist)
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
#define N_CHANNELS  3

/* 2D blocks */
#define BLK_W       3
#define BLKS_PER_ROW 9
#define N_BLOCKS    252
#define N_BVALS     27
#define SIG_PAD     256
#define BG_PIXEL    0
#define BG_GRAD     13

/* Transitions (for Bayesian prior) */
#define H_TRANS_PER_ROW 8
#define N_HTRANS    (H_TRANS_PER_ROW * IMG_H)   /* 224 */
#define V_TRANS_PER_COL 27
#define N_VTRANS    (BLKS_PER_ROW * V_TRANS_PER_COL) /* 243 */
#define TRANS_PAD   256
#define BG_TRANS    13

/* IG */
#define IG_SCALE    16

/* Multi-probe */
#define MAX_NEIGHBORS 6

/* Cascade */
#define MAX_K       1000

/* Soft prior */
#define BOOST_SCALE 256

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Global data ---------- */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t *tern_train, *tern_test;
static int8_t *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;

/* Block signatures (3 channels) */
static uint8_t *chan_train_sigs[N_CHANNELS];
static uint8_t *chan_test_sigs[N_CHANNELS];

/* Hot maps for Bayesian prior (pixel + h-grad + v-grad blocks + transitions) */
static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint8_t *ht_train_sigs, *ht_test_sigs;
static uint8_t *vt_train_sigs, *vt_test_sigs;
static uint32_t ht_hot[N_HTRANS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vt_hot[N_VTRANS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* IG weights per channel */
static uint16_t ig_weights[N_CHANNELS][N_BLOCKS];

/* Neighbor table */
static uint8_t nbr_table[N_BVALS][MAX_NEIGHBORS];
static uint8_t nbr_count[N_BVALS];

/* Inverted index per channel */
static uint32_t *chan_idx_pool[N_CHANNELS];
static uint32_t chan_idx_off[N_CHANNELS][N_BLOCKS][N_BVALS];
static uint16_t chan_idx_sz[N_CHANNELS][N_BLOCKS][N_BVALS];

/* Per-channel background values (BUG FIX from sstt_v2 line 1297) */
static const uint8_t chan_bg[N_CHANNELS] = {BG_PIXEL, BG_GRAD, BG_GRAD};

/* ================================================================
 *  Standard infrastructure
 * ================================================================ */

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
        if (rows_out) *rows_out = rows;
        if (cols_out) *cols_out = cols;
        item_size = (size_t)rows * cols;
    } else {
        if (rows_out) *rows_out = 0;
        if (cols_out) *cols_out = 0;
    }
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
    for (int i = 0; i < PIXELS; i++)
        dst[i] = src[i] > 170 ? 1 : src[i] < 85 ? -1 : 0;
    memset(dst + PIXELS, 0, PADDED - PIXELS);
}

static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

static void grad_one(const int8_t *t, int8_t *hg, int8_t *vg) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++)
            hg[y * IMG_W + x] = clamp_trit(t[y * IMG_W + x + 1] - t[y * IMG_W + x]);
        hg[y * IMG_W + IMG_W - 1] = 0;
    }
    memset(hg + PIXELS, 0, PADDED - PIXELS);
    for (int y = 0; y < IMG_H - 1; y++)
        for (int x = 0; x < IMG_W; x++)
            vg[y * IMG_W + x] = clamp_trit(t[(y + 1) * IMG_W + x] - t[y * IMG_W + x]);
    memset(vg + (IMG_H - 1) * IMG_W, 0, IMG_W);
    memset(vg + PIXELS, 0, PADDED - PIXELS);
}

static void init_images(void) {
    tern_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    hgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    for (int i = 0; i < TRAIN_N; i++) {
        quantize_one(raw_train_img + (size_t)i * PIXELS, tern_train + (size_t)i * PADDED);
        grad_one(tern_train + (size_t)i * PADDED,
                 hgrad_train + (size_t)i * PADDED, vgrad_train + (size_t)i * PADDED);
    }
    for (int i = 0; i < TEST_N; i++) {
        quantize_one(raw_test_img + (size_t)i * PIXELS, tern_test + (size_t)i * PADDED);
        grad_one(tern_test + (size_t)i * PADDED,
                 hgrad_test + (size_t)i * PADDED, vgrad_test + (size_t)i * PADDED);
    }
}

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

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

static void init_sigs(void) {
    const int8_t *train_data[3] = {tern_train, hgrad_train, vgrad_train};
    const int8_t *test_data[3]  = {tern_test, hgrad_test, vgrad_test};
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        chan_train_sigs[ch] = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        chan_test_sigs[ch]  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
        compute_2d_sigs(train_data[ch], chan_train_sigs[ch], TRAIN_N);
        compute_2d_sigs(test_data[ch],  chan_test_sigs[ch],  TEST_N);
    }
}

static void build_hot(const uint8_t *sigs, int n_pos, uint32_t *hot, int stride) {
    memset(hot, 0, sizeof(uint32_t) * (size_t)n_pos * N_BVALS * CLS_PAD);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = sigs + (size_t)i * stride;
        for (int k = 0; k < n_pos; k++)
            hot[(size_t)k * N_BVALS * CLS_PAD + (size_t)sig[k] * CLS_PAD + lbl]++;
    }
}

/* ================================================================
 *  Transition signatures (for Bayesian prior)
 * ================================================================ */

static inline uint8_t trans_enc(uint8_t a, uint8_t b) {
    int8_t a0 = (a / 9) - 1, a1 = ((a / 3) % 3) - 1, a2 = (a % 3) - 1;
    int8_t b0 = (b / 9) - 1, b1 = ((b / 3) % 3) - 1, b2 = (b % 3) - 1;
    return block_encode(clamp_trit(b0 - a0), clamp_trit(b1 - a1), clamp_trit(b2 - a2));
}

static void compute_trans(const uint8_t *bsig, int stride,
                           uint8_t *ht, uint8_t *vt, int n) {
    for (int i = 0; i < n; i++) {
        const uint8_t *s = bsig + (size_t)i * stride;
        uint8_t *h = ht + (size_t)i * TRANS_PAD;
        uint8_t *v = vt + (size_t)i * TRANS_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int ss = 0; ss < H_TRANS_PER_ROW; ss++)
                h[y * H_TRANS_PER_ROW + ss] =
                    trans_enc(s[y * BLKS_PER_ROW + ss], s[y * BLKS_PER_ROW + ss + 1]);
        memset(h + N_HTRANS, 0xFF, TRANS_PAD - N_HTRANS);
        for (int y = 0; y < V_TRANS_PER_COL; y++)
            for (int ss = 0; ss < BLKS_PER_ROW; ss++)
                v[y * BLKS_PER_ROW + ss] =
                    trans_enc(s[y * BLKS_PER_ROW + ss], s[(y + 1) * BLKS_PER_ROW + ss]);
        memset(v + N_VTRANS, 0xFF, TRANS_PAD - N_VTRANS);
    }
}

static void init_transitions(void) {
    ht_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    ht_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    vt_train_sigs = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    vt_test_sigs  = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    compute_trans(chan_train_sigs[0], SIG_PAD, ht_train_sigs, vt_train_sigs, TRAIN_N);
    compute_trans(chan_test_sigs[0],  SIG_PAD, ht_test_sigs,  vt_test_sigs,  TEST_N);
    build_hot(ht_train_sigs, N_HTRANS, (uint32_t *)ht_hot, TRANS_PAD);
    build_hot(vt_train_sigs, N_VTRANS, (uint32_t *)vt_hot, TRANS_PAD);
}

/* ================================================================
 *  IG weights (from sstt_v2.c compute_ig_weights_chan)
 * ================================================================ */

static void compute_ig(int ch) {
    const uint8_t *sigs = chan_train_sigs[ch];
    int class_count[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) class_count[train_labels[i]]++;
    double H_prior = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)class_count[c] / TRAIN_N;
        if (p > 0) H_prior -= p * log2(p);
    }
    double raw_ig[N_BLOCKS], max_ig = 0;
    for (int k = 0; k < N_BLOCKS; k++) {
        int bvc[N_BVALS] = {0}, bvcc[N_BVALS][N_CLASSES];
        memset(bvcc, 0, sizeof(bvcc));
        for (int i = 0; i < TRAIN_N; i++) {
            uint8_t bv = sigs[i * SIG_PAD + k];
            bvc[bv]++;
            bvcc[bv][train_labels[i]]++;
        }
        double Hc = 0;
        for (int bv = 0; bv < N_BVALS; bv++) {
            if (bvc[bv] == 0) continue;
            double pbv = (double)bvc[bv] / TRAIN_N, h = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double p = (double)bvcc[bv][c] / bvc[bv];
                if (p > 0) h -= p * log2(p);
            }
            Hc += pbv * h;
        }
        raw_ig[k] = H_prior - Hc;
        if (raw_ig[k] > max_ig) max_ig = raw_ig[k];
    }
    for (int k = 0; k < N_BLOCKS; k++) {
        if (max_ig > 0)
            ig_weights[ch][k] = (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5);
        else
            ig_weights[ch][k] = 1;
        if (ig_weights[ch][k] == 0) ig_weights[ch][k] = 1;
    }
}

/* ================================================================
 *  Neighbor table (from sstt_v2.c build_neighbor_table)
 * ================================================================ */

static void build_neighbors(void) {
    int trits[3] = {-1, 0, 1};
    for (int v = 0; v < N_BVALS; v++) {
        int t0 = (v / 9) - 1, t1 = ((v / 3) % 3) - 1, t2 = (v % 3) - 1;
        int orig[3] = {t0, t1, t2};
        int nc = 0;
        for (int pos = 0; pos < 3; pos++)
            for (int alt = 0; alt < 3; alt++) {
                if (trits[alt] == orig[pos]) continue;
                int mod[3] = {orig[0], orig[1], orig[2]};
                mod[pos] = trits[alt];
                nbr_table[v][nc++] =
                    block_encode((int8_t)mod[0], (int8_t)mod[1], (int8_t)mod[2]);
            }
        nbr_count[v] = (uint8_t)nc;
    }
}

/* ================================================================
 *  Inverted index per channel (from sstt_v2.c)
 * ================================================================ */

static void build_index(int ch) {
    const uint8_t *sigs = chan_train_sigs[ch];
    memset(chan_idx_sz[ch], 0, sizeof(chan_idx_sz[ch]));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            chan_idx_sz[ch][k][sig[k]]++;
    }
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < N_BVALS; v++) {
            chan_idx_off[ch][k][v] = total;
            total += chan_idx_sz[ch][k][v];
        }
    chan_idx_pool[ch] = malloc((size_t)total * sizeof(uint32_t));
    uint32_t wp[N_BLOCKS][N_BVALS];
    memcpy(wp, chan_idx_off[ch], sizeof(chan_idx_off[ch]));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            chan_idx_pool[ch][wp[k][sig[k]]++] = (uint32_t)i;
    }
}

/* ================================================================
 *  AVX2 ternary dot product
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
 *  Bayesian prior: 5-channel block-interleaved
 *  (pixel + hgrad + vgrad blocks + h-trans + v-trans)
 * ================================================================ */

static inline void bayes_update(double *acc, const uint32_t *h) {
    double mx = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        acc[c] *= ((double)h[c] + 0.5);
        if (acc[c] > mx) mx = acc[c];
    }
    if (mx > 1e10)
        for (int c = 0; c < N_CLASSES; c++) acc[c] /= mx;
}

static void compute_posterior(int img_idx, double posterior[N_CLASSES]) {
    for (int c = 0; c < N_CLASSES; c++) posterior[c] = 1.0;

    /* 3 block channels (pixel, hgrad, vgrad) interleaved */
    const uint8_t *sigs[3] = {
        chan_test_sigs[0] + (size_t)img_idx * SIG_PAD,
        chan_test_sigs[1] + (size_t)img_idx * SIG_PAD,
        chan_test_sigs[2] + (size_t)img_idx * SIG_PAD
    };
    const uint32_t (*hots[3])[N_BVALS][CLS_PAD] = {px_hot, hg_hot, vg_hot};

    for (int k = 0; k < N_BLOCKS; k++) {
        for (int ch = 0; ch < 3; ch++) {
            uint8_t bv = sigs[ch][k];
            if (bv == chan_bg[ch]) continue;
            bayes_update(posterior, hots[ch][k][bv]);
        }
    }

    /* 2 transition channels */
    const uint8_t *ht = ht_test_sigs + (size_t)img_idx * TRANS_PAD;
    const uint8_t *vt = vt_test_sigs + (size_t)img_idx * TRANS_PAD;
    for (int k = 0; k < N_HTRANS; k++) {
        uint8_t bv = ht[k];
        if (bv != BG_TRANS) bayes_update(posterior, ht_hot[k][bv]);
    }
    for (int k = 0; k < N_VTRANS; k++) {
        uint8_t bv = vt[k];
        if (bv != BG_TRANS) bayes_update(posterior, vt_hot[k][bv]);
    }

    /* Normalize */
    double sum = 0;
    for (int c = 0; c < N_CLASSES; c++) sum += posterior[c];
    if (sum > 0)
        for (int c = 0; c < N_CLASSES; c++) posterior[c] /= sum;
    else
        for (int c = 0; c < N_CLASSES; c++) posterior[c] = 1.0 / N_CLASSES;
}

/* ================================================================
 *  Posterior → boost factors
 * ================================================================ */

static void compute_boost(const double *posterior, uint32_t boost[N_CLASSES],
                           double alpha, double temperature) {
    double scaled[N_CLASSES], sum = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        scaled[c] = pow(posterior[c] + 1e-30, 1.0 / temperature);
        sum += scaled[c];
    }
    for (int c = 0; c < N_CLASSES; c++)
        boost[c] = 1 + (uint32_t)(alpha * (scaled[c] / sum) * BOOST_SCALE);
}

/* ================================================================
 *  Top-K selection and k-NN
 * ================================================================ */

typedef struct { uint32_t id; uint32_t votes; int32_t dot; } candidate_t;

static int cmp_dots_desc(const void *a, const void *b) {
    int32_t da = ((const candidate_t *)a)->dot;
    int32_t db = ((const candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

static int select_top_k(const uint32_t *votes, candidate_t *cands, int K) {
    /* Find max vote */
    uint32_t max_v = 0;
    for (int j = 0; j < TRAIN_N; j++)
        if (votes[j] > max_v) max_v = votes[j];
    if (max_v == 0) return 0;

    /* Counting sort — cap histogram to prevent huge allocation */
    int hist_size = (int)max_v + 2;
    if (hist_size > 1000000) {
        /* Fallback: linear scan for large vote ranges */
        /* Find threshold via partial sort */
        int nc = 0;
        for (int j = 0; j < TRAIN_N && nc < K; j++)
            if (votes[j] > 0) {
                cands[nc].id = (uint32_t)j;
                cands[nc].votes = votes[j];
                nc++;
            }
        /* Sort by votes descending, keep top K */
        for (int a = 0; a < nc - 1; a++)
            for (int b = a + 1; b < nc; b++)
                if (cands[b].votes > cands[a].votes) {
                    candidate_t tmp = cands[a]; cands[a] = cands[b]; cands[b] = tmp;
                }
        return nc < K ? nc : K;
    }

    int *hist = calloc((size_t)hist_size, sizeof(int));
    for (int j = 0; j < TRAIN_N; j++) hist[votes[j]]++;
    int cum = 0, thr;
    for (thr = (int)max_v; thr >= 1; thr--) {
        cum += hist[thr];
        if (cum >= K) break;
    }
    if (thr < 1) thr = 1;
    free(hist);

    int nc = 0;
    for (int j = 0; j < TRAIN_N && nc < K; j++)
        if (votes[j] >= (uint32_t)thr) {
            cands[nc].id = (uint32_t)j;
            cands[nc].votes = votes[j];
            nc++;
        }
    return nc;
}

static int knn_vote(const candidate_t *cands, int nc, int k) {
    if (k > nc) k = nc;
    int cv[N_CLASSES] = {0};
    for (int i = 0; i < k; i++)
        cv[train_labels[cands[i].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (cv[c] > cv[best]) best = c;
    return best;
}

/* ================================================================
 *  Soft-Prior Cascade Classifier
 * ================================================================ */

static int softcascade_classify(int img_idx, const uint32_t *img_boost,
                                 int top_k, int knn_k) {
    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));

    /* 3-channel IG-weighted multi-probe voting with boost */
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const uint8_t *qsig = chan_test_sigs[ch] + (size_t)img_idx * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == chan_bg[ch]) continue;   /* per-channel background (BUG FIX) */

            uint16_t w = ig_weights[ch][k];
            uint16_t w_half = w / 2;
            if (w_half == 0) w_half = 1;

            /* Exact match — boosted */
            uint32_t off = chan_idx_off[ch][k][bv];
            uint16_t sz  = chan_idx_sz[ch][k][bv];
            const uint32_t *ids = chan_idx_pool[ch] + off;
            for (uint16_t j = 0; j < sz; j++)
                votes[ids[j]] += (uint32_t)w * img_boost[ids[j]];

            /* Hamming-1 neighbors — boosted */
            for (int nb = 0; nb < nbr_count[bv]; nb++) {
                uint8_t nv = nbr_table[bv][nb];
                uint32_t noff = chan_idx_off[ch][k][nv];
                uint16_t nsz  = chan_idx_sz[ch][k][nv];
                const uint32_t *nids = chan_idx_pool[ch] + noff;
                for (uint16_t j = 0; j < nsz; j++)
                    votes[nids[j]] += (uint32_t)w_half * img_boost[nids[j]];
            }
        }
    }

    /* Top-K selection */
    candidate_t cands[MAX_K];
    int nc = select_top_k(votes, cands, top_k);
    free(votes);
    if (nc == 0) return 0;

    /* Dot product refinement */
    const int8_t *query = tern_test + (size_t)img_idx * PADDED;
    for (int j = 0; j < nc; j++)
        cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);

    qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

    /* k-NN majority vote */
    return knn_vote(cands, nc, knn_k);
}

/* Pure cascade (alpha=0 baseline, no boost) */
static int pure_cascade_classify(int img_idx, int top_k, int knn_k) {
    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const uint8_t *qsig = chan_test_sigs[ch] + (size_t)img_idx * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == chan_bg[ch]) continue;

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
    int nc = select_top_k(votes, cands, top_k);
    free(votes);
    if (nc == 0) return 0;

    const int8_t *query = tern_test + (size_t)img_idx * PADDED;
    for (int j = 0; j < nc; j++)
        cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);
    qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);

    return knn_vote(cands, nc, knn_k);
}

/* ================================================================
 *  Main: Build → Baseline → Sweep → Report
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
    printf("=== SSTT Soft-Prior Cascade: Information Perihelion (%s) ===\n\n", ds);

    /* Build everything */
    printf("Building all structures...\n");
    load_data(data_dir);
    init_images();
    init_sigs();
    build_hot(chan_train_sigs[0], N_BLOCKS, (uint32_t *)px_hot, SIG_PAD);
    build_hot(chan_train_sigs[1], N_BLOCKS, (uint32_t *)hg_hot, SIG_PAD);
    build_hot(chan_train_sigs[2], N_BLOCKS, (uint32_t *)vg_hot, SIG_PAD);
    init_transitions();
    for (int ch = 0; ch < N_CHANNELS; ch++) compute_ig(ch);
    build_neighbors();
    for (int ch = 0; ch < N_CHANNELS; ch++) build_index(ch);
    double t1 = now_sec();

    printf("  IG weights: px min=%d max=%d, hg min=%d max=%d, vg min=%d max=%d\n",
           ig_weights[0][0], ig_weights[0][N_BLOCKS/2],
           ig_weights[1][0], ig_weights[1][N_BLOCKS/2],
           ig_weights[2][0], ig_weights[2][N_BLOCKS/2]);
    printf("  Built (%.2f sec)\n\n", t1 - t0);

    /* --- Pre-compute all posteriors --- */
    printf("Pre-computing Bayesian posteriors for all test images...\n");
    double (*posteriors)[N_CLASSES] = malloc(TEST_N * sizeof(double[N_CLASSES]));
    double tp0 = now_sec();
    int bayes_correct = 0;
    for (int i = 0; i < TEST_N; i++) {
        compute_posterior(i, posteriors[i]);
        int best = 0;
        for (int c = 1; c < N_CLASSES; c++)
            if (posteriors[i][c] > posteriors[i][best]) best = c;
        if (best == test_labels[i]) bayes_correct++;
    }
    double tp1 = now_sec();
    printf("  5-channel Bayesian: %.2f%% (%.4f sec, %.0f ns/query)\n\n",
           100.0 * bayes_correct / TEST_N, tp1 - tp0, (tp1 - tp0) * 1e9 / TEST_N);

    /* --- Baseline: pure cascade (alpha=0, with BG fix) --- */
    printf("--- Baseline: Pure Cascade (α=0, 3-chan IG multi-probe, BG fix) ---\n");
    int pure_correct = 0;
    double tc0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        if (pure_cascade_classify(i, 500, 3) == test_labels[i]) pure_correct++;
        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  Pure cascade: %d/%d (%.2f%%)\r",
                    i + 1, TEST_N, 100.0 * pure_correct / (i + 1));
    }
    fprintf(stderr, "\n");
    double tc1 = now_sec();
    printf("  Accuracy: %.2f%% (%.2f sec)\n\n", 100.0 * pure_correct / TEST_N, tc1 - tc0);

    /* --- Sweep: soft-prior cascade --- */
    printf("--- Soft-Prior Cascade Sweep ---\n");
    printf("  votes[tid] += w × boost[label[tid]]\n");
    printf("  boost[c] = 1 + α × softmax(posterior, T)[c] × %d\n\n", BOOST_SCALE);

    double alphas[] = {1, 2, 4, 8, 16};
    int n_alpha = 5;
    double temps[] = {1.0, 3.0, 10.0};
    int n_temp = 3;

    printf("  %-6s %-6s | %-10s | %-10s | %-12s\n",
           "α", "T", "Accuracy", "Time", "vs Pure");
    printf("  -------+------+------------+------------+-------------\n");

    /* Pre-allocate img_boost array (reused per config) */
    uint32_t *img_boost = malloc(TRAIN_N * sizeof(uint32_t));

    double best_acc = 100.0 * pure_correct / TEST_N;
    double best_alpha = 0, best_temp = 0;

    for (int ai = 0; ai < n_alpha; ai++) {
        for (int ti = 0; ti < n_temp; ti++) {
            double alpha = alphas[ai];
            double temp = temps[ti];

            int correct = 0;
            double ts0 = now_sec();

            for (int i = 0; i < TEST_N; i++) {
                /* Compute boost factors from pre-computed posterior */
                uint32_t boost[N_CLASSES];
                compute_boost(posteriors[i], boost, alpha, temp);

                /* Expand to per-training-image boost (label lookup) */
                for (int j = 0; j < TRAIN_N; j++)
                    img_boost[j] = boost[train_labels[j]];

                /* Classify with soft-prior cascade */
                if (softcascade_classify(i, img_boost, 500, 3) == test_labels[i])
                    correct++;

                if ((i + 1) % 2000 == 0)
                    fprintf(stderr, "  α=%.0f T=%.0f: %d/%d (%.2f%%)\r",
                            alpha, temp, i + 1, TEST_N, 100.0 * correct / (i + 1));
            }
            fprintf(stderr, "\n");
            double ts1 = now_sec();
            double acc = 100.0 * correct / TEST_N;

            printf("  %5.1f  %5.1f | %7.2f%%   | %7.2f s  | %+.2f pp\n",
                   alpha, temp, acc, ts1 - ts0,
                   acc - 100.0 * pure_correct / TEST_N);

            if (acc > best_acc) {
                best_acc = acc;
                best_alpha = alpha;
                best_temp = temp;
            }
        }
    }

    /* --- Summary --- */
    printf("\n=== SUMMARY ===\n");
    printf("  5-chan Bayesian (prior):  %.2f%%  (%.4f sec)\n",
           100.0 * bayes_correct / TEST_N, tp1 - tp0);
    printf("  Pure cascade (α=0):      %.2f%%  (%.2f sec)\n",
           100.0 * pure_correct / TEST_N, tc1 - tc0);
    if (best_alpha > 0)
        printf("  Best soft cascade:       %.2f%%  (α=%.1f T=%.1f)\n",
               best_acc, best_alpha, best_temp);
    else
        printf("  No improvement over pure cascade.\n");
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    /* Cleanup */
    free(posteriors);
    free(img_boost);
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        free(chan_train_sigs[ch]); free(chan_test_sigs[ch]);
        free(chan_idx_pool[ch]);
    }
    free(ht_train_sigs); free(ht_test_sigs);
    free(vt_train_sigs); free(vt_test_sigs);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);

    return 0;
}
