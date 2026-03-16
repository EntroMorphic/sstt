/*
 * sstt_tiled.c — Ternary K-Means Tile Routing Cascade
 *
 * Translates the routing idea:
 *   signature = tile_weights.sum(dim=0).sign()  # what the tile wants
 *   score     = input @ signature               # how well input matches
 *   route     = (score == scores.max())         # send to best match
 *
 * Architecture:
 *   Training:
 *     1. Ternary k-means: cluster 60K training images into N_TILES tiles
 *        prototype[t] = sign(sum of ternary images in tile t)
 *        similarity   = ternary_dot(image, prototype)
 *        5 EM iterations from class-mean initialization
 *
 *     2. Each tile builds its own 3-channel cascade:
 *        inverted index + IG weights from its ~6K assigned images
 *
 *   Inference:
 *     1. Routing: for each test image, score[t] = ternary_dot(test, proto[t])
 *        → route to argmax tile (hard) or top-2 tiles (soft)
 *     2. Accumulate IG-weighted multi-probe votes from routed tile(s)
 *     3. Top-K → ternary dot refinement → k-NN vote
 *
 * Tests:
 *   A. Global cascade baseline (all 60K images, confirms 96.12%)
 *   B. Hard routing: top-1 tile only
 *   C. Soft routing: merge votes from top-2 tiles
 *
 * Build: make sstt_tiled
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N       60000
#define TEST_N        10000
#define IMG_W         28
#define IMG_H         28
#define PIXELS        784
#define PADDED        800
#define N_CLASSES     10
#define CLS_PAD       16

#define BLKS_PER_ROW  9
#define N_BLOCKS      252
#define N_BVALS       27
#define SIG_PAD       256
#define BG_PIXEL      0
#define BG_GRAD       13

#define N_CHANNELS    3
#define IG_SCALE      16
#define MAX_K         500
#define N_TILES       10
#define EM_ITERS      5

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  Data arrays
 * ================================================================ */

static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test;
static int8_t  *vgrad_train, *vgrad_test;

static uint8_t *chan_train_sigs[N_CHANNELS];
static uint8_t *chan_test_sigs[N_CHANNELS];

/* K-means tile assignments */
static int8_t  tile_proto[N_TILES][PADDED];  /* ternary routing prototype */
static int     tile_assign[TRAIN_N];          /* tile id for each training image */
static int     tile_members[N_TILES][TRAIN_N];
static int     tile_n[N_TILES];

/* Per-tile per-channel inverted index */
static uint32_t tile_off[N_TILES][N_CHANNELS][N_BLOCKS][N_BVALS];
static uint16_t tile_sz [N_TILES][N_CHANNELS][N_BLOCKS][N_BVALS];
static uint32_t *tile_pool[N_TILES][N_CHANNELS];
static uint16_t tile_ig[N_TILES][N_CHANNELS][N_BLOCKS];

/* Global cascade index (for Test A baseline) */
static uint16_t global_ig[N_CHANNELS][N_BLOCKS];
static uint32_t global_off[N_CHANNELS][N_BLOCKS][N_BVALS];
static uint16_t global_sz [N_CHANNELS][N_BLOCKS][N_BVALS];
static uint32_t *global_pool[N_CHANNELS];

/* Neighbor table (shared: ternary Hamming-1) */
static uint8_t nbr_table[N_BVALS][6];
static uint8_t nbr_count[N_BVALS];

/* ================================================================
 *  Data loading
 * ================================================================ */

static uint8_t *load_idx(const char *path, uint32_t *count,
                         uint32_t *ro, uint32_t *co) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", path); exit(1); }
    uint32_t magic, n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) { fclose(f); exit(1); }
    magic = __builtin_bswap32(magic); n = __builtin_bswap32(n); *count = n;
    int ndim = magic & 0xFF; size_t item_size = 1;
    if (ndim >= 3) {
        uint32_t r, c;
        if (fread(&r, 4, 1, f) != 1 || fread(&c, 4, 1, f) != 1) { fclose(f); exit(1); }
        r = __builtin_bswap32(r); c = __builtin_bswap32(c);
        if (ro) *ro = r; if (co) *co = c;
        item_size = (size_t)r * c;
    } else { if (ro) *ro = 0; if (co) *co = 0; }
    size_t total = (size_t)n * item_size;
    uint8_t *data = malloc(total);
    if (!data || fread(data, 1, total, f) != total) { fclose(f); exit(1); }
    fclose(f); return data;
}

static void load_data(void) {
    uint32_t n, r, c; char path[256];
    snprintf(path, sizeof(path), "%strain-images-idx3-ubyte", data_dir);
    raw_train_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%strain-labels-idx1-ubyte", data_dir);
    train_labels = load_idx(path, &n, NULL, NULL);
    snprintf(path, sizeof(path), "%st10k-images-idx3-ubyte", data_dir);
    raw_test_img = load_idx(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%st10k-labels-idx1-ubyte", data_dir);
    test_labels = load_idx(path, &n, NULL, NULL);
}

/* ================================================================
 *  Feature computation
 * ================================================================ */

static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

static void quantize_avx2(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));
    const __m256i one  = _mm256_set1_epi8(1);
    for (int img = 0; img < n; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED;
        int i;
        for (i = 0; i + 32 <= PIXELS; i += 32) {
            __m256i px  = _mm256_loadu_si256((const __m256i *)(s + i));
            __m256i spx = _mm256_xor_si256(px, bias);
            __m256i pos = _mm256_cmpgt_epi8(spx, thi);
            __m256i neg = _mm256_cmpgt_epi8(tlo, spx);
            _mm256_storeu_si256((__m256i *)(d + i),
                _mm256_sub_epi8(_mm256_and_si256(pos, one),
                                _mm256_and_si256(neg, one)));
        }
        for (; i < PIXELS; i++)
            d[i] = s[i] > 170 ? 1 : s[i] < 85 ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
    }
}

static void compute_gradients(const int8_t *tern, int8_t *hg, int8_t *vg, int n) {
    for (int img = 0; img < n; img++) {
        const int8_t *t = tern + (size_t)img * PADDED;
        int8_t *h = hg + (size_t)img * PADDED;
        int8_t *v = vg + (size_t)img * PADDED;
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W - 1; x++)
                h[y * IMG_W + x] = clamp_trit(t[y * IMG_W + x + 1] - t[y * IMG_W + x]);
            h[y * IMG_W + IMG_W - 1] = 0;
        }
        memset(h + PIXELS, 0, PADDED - PIXELS);
        for (int y = 0; y < IMG_H - 1; y++)
            for (int x = 0; x < IMG_W; x++)
                v[y * IMG_W + x] = clamp_trit(t[(y + 1) * IMG_W + x] - t[y * IMG_W + x]);
        memset(v + (IMG_H - 1) * IMG_W, 0, IMG_W);
        memset(v + PIXELS, 0, PADDED - PIXELS);
    }
}

static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1));
}

static void compute_block_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int base = y * IMG_W + s * 3;
                sig[y * BLKS_PER_ROW + s] =
                    block_encode(img[base], img[base+1], img[base+2]);
            }
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

/* ================================================================
 *  AVX2 ternary dot product
 * ================================================================ */

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < PADDED; i += 32)
        acc = _mm256_add_epi8(acc,
            _mm256_sign_epi8(_mm256_load_si256((const __m256i *)(a + i)),
                             _mm256_load_si256((const __m256i *)(b + i))));
    __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc, 1));
    __m256i s32  = _mm256_madd_epi16(_mm256_add_epi16(lo16, hi16),
                                      _mm256_set1_epi16(1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(s32),
                              _mm256_extracti128_si256(s32, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Ternary K-Means
 *
 *  signature = tile_weights.sum(dim=0).sign()
 *  score     = input @ signature
 *  route     = argmax(scores)
 * ================================================================ */

static void kmeans_cluster(void) {
    /* Initialize: one prototype per class = sign(sum of class images) */
    printf("  K-means init from class means...\n");
    for (int t = 0; t < N_TILES; t++) {
        int32_t acc[PADDED] = {0};
        int cnt = 0;
        for (int i = 0; i < TRAIN_N; i++) {
            if (train_labels[i] != t) continue;
            const int8_t *img = tern_train + (size_t)i * PADDED;
            for (int p = 0; p < PIXELS; p++) acc[p] += img[p];
            cnt++;
        }
        /* sign(sum) → ternary prototype */
        for (int p = 0; p < PADDED; p++)
            tile_proto[t][p] = acc[p] > 0 ? 1 : acc[p] < 0 ? -1 : 0;
        (void)cnt;
    }

    for (int iter = 0; iter < EM_ITERS; iter++) {
        /* --- Assign each training image to nearest prototype --- */
        int reassigned = 0;
        for (int i = 0; i < TRAIN_N; i++) {
            const int8_t *img = tern_train + (size_t)i * PADDED;
            int best_t = 0;
            int32_t best_score = INT32_MIN;
            for (int t = 0; t < N_TILES; t++) {
                int32_t score = ternary_dot(img, tile_proto[t]);
                if (score > best_score) { best_score = score; best_t = t; }
            }
            if (tile_assign[i] != best_t) reassigned++;
            tile_assign[i] = best_t;
        }

        /* --- Update prototypes: sign(sum of assigned images) --- */
        int32_t (*acc)[PADDED] = calloc(N_TILES * PADDED, sizeof(int32_t));
        memset(tile_n, 0, sizeof(tile_n));
        for (int i = 0; i < TRAIN_N; i++) {
            int t = tile_assign[i];
            const int8_t *img = tern_train + (size_t)i * PADDED;
            for (int p = 0; p < PIXELS; p++) acc[t][p] += img[p];
            tile_n[t]++;
        }
        for (int t = 0; t < N_TILES; t++) {
            if (tile_n[t] == 0) {
                /* Empty tile: re-init from a random training image */
                int rand_i = (int)((double)TRAIN_N * rand() / RAND_MAX);
                const int8_t *img = tern_train + (size_t)rand_i * PADDED;
                memcpy(tile_proto[t], img, PADDED);
                tile_n[t] = 1;
                continue;
            }
            for (int p = 0; p < PADDED; p++)
                tile_proto[t][p] = acc[t][p] > 0 ? 1 : acc[t][p] < 0 ? -1 : 0;
        }
        free(acc);

        printf("  Iter %d: %d reassigned. Tile sizes:", iter + 1, reassigned);
        for (int t = 0; t < N_TILES; t++) printf(" %d", tile_n[t]);
        printf("\n");
        if (reassigned == 0) break;
    }

    /* Build member lists */
    memset(tile_n, 0, sizeof(tile_n));
    for (int i = 0; i < TRAIN_N; i++) {
        int t = tile_assign[i];
        tile_members[t][tile_n[t]++] = i;
    }

    /* Print class distribution per tile */
    printf("\n  Tile class distributions:\n");
    for (int t = 0; t < N_TILES; t++) {
        int cc[N_CLASSES] = {0};
        for (int i = 0; i < tile_n[t]; i++)
            cc[train_labels[tile_members[t][i]]]++;
        printf("  Tile %d (%d imgs):", t, tile_n[t]);
        for (int c = 0; c < N_CLASSES; c++) printf(" %d:%d", c, cc[c]);
        printf("\n");
    }
    printf("\n");
}

/* ================================================================
 *  Neighbor table (ternary Hamming-1)
 * ================================================================ */

static void build_neighbor_table(void) {
    int trits[3] = {-1, 0, 1};
    for (int v = 0; v < N_BVALS; v++) {
        int orig[3] = {(v/9)-1, ((v/3)%3)-1, (v%3)-1};
        int nc = 0;
        for (int pos = 0; pos < 3; pos++)
            for (int alt = 0; alt < 3; alt++) {
                if (trits[alt] == orig[pos]) continue;
                int mod[3] = {orig[0], orig[1], orig[2]};
                mod[pos] = trits[alt];
                nbr_table[v][nc++] = block_encode(
                    (int8_t)mod[0], (int8_t)mod[1], (int8_t)mod[2]);
            }
        nbr_count[v] = (uint8_t)nc;
    }
}

/* ================================================================
 *  Per-tile IG weights (computed from tile's subset of training images)
 * ================================================================ */

static void compute_tile_ig_ch(int t, int ch) {
    const int *members = tile_members[t];
    int nm = tile_n[t];
    const uint8_t *train_sigs = chan_train_sigs[ch];
    const uint8_t bg = (ch == 0) ? BG_PIXEL : BG_GRAD;

    int class_counts[N_CLASSES] = {0};
    for (int i = 0; i < nm; i++) class_counts[train_labels[members[i]]]++;
    double h_class = 0.0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)class_counts[c] / nm;
        if (p > 0) h_class -= p * log2(p);
    }

    double raw_ig[N_BLOCKS], max_ig = 0.0;
    for (int k = 0; k < N_BLOCKS; k++) {
        int counts[N_BVALS][N_CLASSES];
        int val_total[N_BVALS];
        memset(counts, 0, sizeof(counts)); memset(val_total, 0, sizeof(val_total));
        for (int i = 0; i < nm; i++) {
            int img = members[i];
            uint8_t v = train_sigs[(size_t)img * SIG_PAD + k];
            counts[v][train_labels[img]]++; val_total[v]++;
        }
        double h_cond = 0.0;
        for (int v = 0; v < N_BVALS; v++) {
            if (!val_total[v] || v == bg) continue;
            double pv = (double)val_total[v] / nm;
            double hv = 0.0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)counts[v][c] / val_total[v];
                if (pc > 0) hv -= pc * log2(pc);
            }
            h_cond += pv * hv;
        }
        raw_ig[k] = h_class - h_cond;
        if (raw_ig[k] > max_ig) max_ig = raw_ig[k];
    }
    for (int k = 0; k < N_BLOCKS; k++) {
        tile_ig[t][ch][k] = max_ig > 0 ?
            (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5) : 1;
        if (!tile_ig[t][ch][k]) tile_ig[t][ch][k] = 1;
    }
}

/* ================================================================
 *  Per-tile per-channel inverted index
 * ================================================================ */

static void build_tile_ch_index(int t, int ch) {
    const int *members = tile_members[t];
    int nm = tile_n[t];
    const uint8_t *train_sigs = chan_train_sigs[ch];
    const uint8_t bg = (ch == 0) ? BG_PIXEL : BG_GRAD;

    memset(tile_sz[t][ch], 0, sizeof(tile_sz[t][ch]));
    for (int i = 0; i < nm; i++) {
        int img = members[i];
        const uint8_t *sig = train_sigs + (size_t)img * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != bg) tile_sz[t][ch][k][sig[k]]++;
    }
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < N_BVALS; v++) {
            tile_off[t][ch][k][v] = total;
            total += tile_sz[t][ch][k][v];
        }
    tile_pool[t][ch] = malloc((size_t)total * sizeof(uint32_t));
    if (!tile_pool[t][ch]) { fprintf(stderr, "ERROR: tile pool\n"); exit(1); }

    /* wpos on stack: 252 × 27 × 4 = ~27KB — fine */
    uint32_t wpos[N_BLOCKS][N_BVALS];
    memcpy(wpos, tile_off[t][ch], sizeof(wpos));
    for (int i = 0; i < nm; i++) {
        int img = members[i];
        const uint8_t *sig = train_sigs + (size_t)img * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (sig[k] != bg)
                tile_pool[t][ch][wpos[k][sig[k]]++] = (uint32_t)img;
    }
}

/* ================================================================
 *  Global cascade index (same as sstt_v2.c, for Test A baseline)
 * ================================================================ */

static void build_global_index(void) {
    /* IG weights */
    int class_counts[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) class_counts[train_labels[i]]++;
    double h_class = 0.0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)class_counts[c] / TRAIN_N;
        if (p > 0) h_class -= p * log2(p);
    }

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const uint8_t *sigs = chan_train_sigs[ch];
        uint8_t bg = (ch == 0) ? BG_PIXEL : BG_GRAD;
        double raw_ig[N_BLOCKS], max_ig = 0.0;
        for (int k = 0; k < N_BLOCKS; k++) {
            int counts[N_BVALS][N_CLASSES], vt[N_BVALS];
            memset(counts, 0, sizeof(counts)); memset(vt, 0, sizeof(vt));
            for (int i = 0; i < TRAIN_N; i++) {
                uint8_t v = sigs[(size_t)i * SIG_PAD + k];
                counts[v][train_labels[i]]++; vt[v]++;
            }
            double h_cond = 0.0;
            for (int v = 0; v < N_BVALS; v++) {
                if (!vt[v] || v == bg) continue;
                double pv = (double)vt[v] / TRAIN_N, hv = 0.0;
                for (int c = 0; c < N_CLASSES; c++) {
                    double pc = (double)counts[v][c] / vt[v];
                    if (pc > 0) hv -= pc * log2(pc);
                }
                h_cond += pv * hv;
            }
            raw_ig[k] = h_class - h_cond;
            if (raw_ig[k] > max_ig) max_ig = raw_ig[k];
        }
        for (int k = 0; k < N_BLOCKS; k++) {
            global_ig[ch][k] = max_ig > 0 ?
                (uint16_t)(raw_ig[k] / max_ig * IG_SCALE + 0.5) : 1;
            if (!global_ig[ch][k]) global_ig[ch][k] = 1;
        }

        /* Inverted index */
        memset(global_sz[ch], 0, sizeof(global_sz[ch]));
        for (int i = 0; i < TRAIN_N; i++) {
            const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
            for (int k = 0; k < N_BLOCKS; k++)
                if (sig[k] != bg) global_sz[ch][k][sig[k]]++;
        }
        uint32_t total = 0;
        for (int k = 0; k < N_BLOCKS; k++)
            for (int v = 0; v < N_BVALS; v++) {
                global_off[ch][k][v] = total; total += global_sz[ch][k][v];
            }
        global_pool[ch] = malloc((size_t)total * sizeof(uint32_t));
        uint32_t wpos[N_BLOCKS][N_BVALS];
        memcpy(wpos, global_off[ch], sizeof(wpos));
        for (int i = 0; i < TRAIN_N; i++) {
            const uint8_t *sig = sigs + (size_t)i * SIG_PAD;
            for (int k = 0; k < N_BLOCKS; k++)
                if (sig[k] != bg)
                    global_pool[ch][wpos[k][sig[k]]++] = (uint32_t)i;
        }
    }
}

/* ================================================================
 *  Cascade machinery: top-K, dot product, k-NN
 * ================================================================ */

typedef struct { uint32_t id; uint32_t votes; int32_t dot; } candidate_t;

static int cmp_votes_desc(const void *a, const void *b) {
    return (int)((const candidate_t *)b)->votes - (int)((const candidate_t *)a)->votes;
}
static int cmp_dots_desc(const void *a, const void *b) {
    int32_t da = ((const candidate_t *)a)->dot;
    int32_t db = ((const candidate_t *)b)->dot;
    return (db > da) - (db < da);
}

static int select_top_k(const uint32_t *votes, int n, candidate_t *out, int k) {
    uint32_t mx = 0;
    for (int i = 0; i < n; i++) if (votes[i] > mx) mx = votes[i];
    if (!mx) return 0;
    int *hist = calloc((size_t)(mx + 1), sizeof(int));
    for (int i = 0; i < n; i++) if (votes[i]) hist[votes[i]]++;
    int cum = 0, thr;
    for (thr = (int)mx; thr >= 1; thr--) { cum += hist[thr]; if (cum >= k) break; }
    if (thr < 1) thr = 1;
    free(hist);
    int nc = 0;
    for (int i = 0; i < n && nc < k; i++)
        if (votes[i] >= (uint32_t)thr)
            out[nc++] = (candidate_t){(uint32_t)i, votes[i], 0};
    qsort(out, (size_t)nc, sizeof(candidate_t), cmp_votes_desc);
    return nc;
}

static int knn_vote(const candidate_t *cands, int nc, int k) {
    int votes[N_CLASSES] = {0};
    if (k > nc) k = nc;
    for (int i = 0; i < k; i++) votes[train_labels[cands[i].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (votes[c] > votes[best]) best = c;
    return best;
}

/* Accumulate IG-weighted multi-probe votes using given index */
static void accumulate_votes(uint32_t *votes,
                              int t_idx,     /* tile index, -1 for global */
                              int test_img) {
    const uint8_t bg_vals[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const uint8_t *qsig = chan_test_sigs[ch] + (size_t)test_img * SIG_PAD;
        const uint8_t bg = bg_vals[ch];
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = qsig[k];
            if (bv == bg) continue;

            uint16_t w, w_half;
            uint32_t off; uint16_t sz; const uint32_t *ids;

            if (t_idx < 0) {
                w = global_ig[ch][k];
                off = global_off[ch][k][bv]; sz = global_sz[ch][k][bv];
                ids = global_pool[ch] + off;
            } else {
                w = tile_ig[t_idx][ch][k];
                off = tile_off[t_idx][ch][k][bv]; sz = tile_sz[t_idx][ch][k][bv];
                ids = tile_pool[t_idx][ch] + off;
            }
            w_half = w > 1 ? w / 2 : 1;
            for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w;

            for (int nb = 0; nb < nbr_count[bv]; nb++) {
                uint8_t nv = nbr_table[bv][nb];
                if (nv == bg) continue;
                if (t_idx < 0) {
                    off = global_off[ch][k][nv]; sz = global_sz[ch][k][nv];
                    ids = global_pool[ch] + off;
                } else {
                    off = tile_off[t_idx][ch][k][nv]; sz = tile_sz[t_idx][ch][k][nv];
                    ids = tile_pool[t_idx][ch] + off;
                }
                for (uint16_t j = 0; j < sz; j++) votes[ids[j]] += w_half;
            }
        }
    }
}

/* ================================================================
 *  Error analysis
 * ================================================================ */

static void error_analysis(const uint8_t *preds) {
    int conf[N_CLASSES][N_CLASSES];
    memset(conf, 0, sizeof(conf));
    for (int i = 0; i < TEST_N; i++) conf[test_labels[i]][preds[i]]++;
    printf("  Confusion Matrix:\n       ");
    for (int c = 0; c < N_CLASSES; c++) printf(" %4d", c);
    printf("   | Recall\n  -----");
    for (int c = 0; c < N_CLASSES; c++) (void)c, printf("-----");
    printf("---+-------\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("    %d: ", r); int rt = 0;
        for (int c = 0; c < N_CLASSES; c++) { printf(" %4d", conf[r][c]); rt += conf[r][c]; }
        printf("   | %5.1f%%\n", rt > 0 ? 100.0 * conf[r][r] / rt : 0.0);
    }
    typedef struct { int a, b, count; } pair_t;
    pair_t pairs[45]; int np = 0;
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = a+1; b < N_CLASSES; b++)
            pairs[np++] = (pair_t){a, b, conf[a][b] + conf[b][a]};
    for (int i = 0; i < np-1; i++)
        for (int j = i+1; j < np; j++)
            if (pairs[j].count > pairs[i].count) { pair_t x=pairs[i]; pairs[i]=pairs[j]; pairs[j]=x; }
    printf("  Top-5 confused pairs:");
    for (int i = 0; i < 5 && i < np; i++)
        printf("  %d\xE2\x86\x94%d:%d", pairs[i].a, pairs[i].b, pairs[i].count);
    printf("\n");
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(int argc, char **argv) {
    double t0 = now_sec();
    srand(42);

    if (argc > 1) {
        data_dir = argv[1];
        size_t len = strlen(data_dir);
        if (len > 0 && data_dir[len-1] != '/') {
            char *buf = malloc(len + 2);
            memcpy(buf, data_dir, len); buf[len] = '/'; buf[len+1] = '\0';
            data_dir = buf;
        }
    }
    const char *ds = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Tiled Cascade — K-Means Routing (%s) ===\n\n", ds);

    /* --- Load and featurize --- */
    printf("Loading and building features...\n");
    load_data();

    tern_train  = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test   = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    hgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);

    quantize_avx2(raw_train_img, tern_train, TRAIN_N);
    quantize_avx2(raw_test_img,  tern_test,  TEST_N);
    compute_gradients(tern_train, hgrad_train, vgrad_train, TRAIN_N);
    compute_gradients(tern_test,  hgrad_test,  vgrad_test,  TEST_N);

    uint8_t *sigs[6];
    for (int i = 0; i < 6; i++)
        sigs[i] = aligned_alloc(32, (size_t)((i < 3 ? TRAIN_N : TEST_N)) * SIG_PAD);
    compute_block_sigs(tern_train,  sigs[0], TRAIN_N);
    compute_block_sigs(hgrad_train, sigs[1], TRAIN_N);
    compute_block_sigs(vgrad_train, sigs[2], TRAIN_N);
    compute_block_sigs(tern_test,   sigs[3], TEST_N);
    compute_block_sigs(hgrad_test,  sigs[4], TEST_N);
    compute_block_sigs(vgrad_test,  sigs[5], TEST_N);
    for (int ch = 0; ch < 3; ch++) {
        chan_train_sigs[ch] = sigs[ch];
        chan_test_sigs[ch]  = sigs[3 + ch];
    }

    build_neighbor_table();
    printf("  Features ready (%.2f sec)\n\n", now_sec() - t0);

    /* --- Ternary K-Means --- */
    printf("--- Ternary K-Means Clustering (%d tiles, %d EM iters) ---\n",
           N_TILES, EM_ITERS);
    memset(tile_assign, 0, sizeof(tile_assign));
    kmeans_cluster();

    /* --- Build tile indices --- */
    printf("Building tile indices...\n");
    double tb = now_sec();
    for (int t = 0; t < N_TILES; t++) {
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            compute_tile_ig_ch(t, ch);
            build_tile_ch_index(t, ch);
        }
        if ((t + 1) % 2 == 0)
            fprintf(stderr, "  Tile %d/%d built\r", t + 1, N_TILES);
    }
    fprintf(stderr, "\n");
    printf("  Tile indices built (%.2f sec)\n\n", now_sec() - tb);

    /* --- Build global index (for Test A baseline) --- */
    printf("Building global index...\n");
    double tg = now_sec();
    build_global_index();
    printf("  Global index built (%.2f sec)\n\n", now_sec() - tg);

    uint32_t *votes    = calloc(TRAIN_N, sizeof(uint32_t));
    candidate_t *cands = malloc(MAX_K * sizeof(candidate_t));
    uint8_t *best_preds = calloc(TEST_N, 1);

    int K = 200, KNN = 3;

    /* ----------------------------------------------------------------
     * Test A: Global cascade (baseline — all 60K images)
     * ---------------------------------------------------------------- */
    printf("--- Test A: Global Cascade (baseline) ---\n");
    {
        int correct = 0;
        double ta = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            accumulate_votes(votes, -1, i);
            int nc = select_top_k(votes, TRAIN_N, cands, K);
            const int8_t *q = tern_test + (size_t)i * PADDED;
            for (int j = 0; j < nc; j++)
                cands[j].dot = ternary_dot(q, tern_train + (size_t)cands[j].id * PADDED);
            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);
            int pred = knn_vote(cands, nc, KNN);
            best_preds[i] = (uint8_t)pred;
            if (pred == test_labels[i]) correct++;
            if ((i+1) % 2000 == 0)
                fprintf(stderr, "  Global: %d/%d\r", i+1, TEST_N);
        }
        fprintf(stderr, "\n");
        printf("  K=%d k=%d: %.2f%%  (%.2f sec)\n\n",
               K, KNN, 100.0 * correct / TEST_N, now_sec() - ta);
        error_analysis(best_preds);
        printf("\n");
    }

    /* ----------------------------------------------------------------
     * Test B: Hard routing — top-1 tile's cascade
     * ---------------------------------------------------------------- */
    printf("--- Test B: Hard Routing (top-1 tile) ---\n");
    {
        int correct = 0, routing_correct = 0;
        int route_hist[N_TILES] = {0};
        double tb2 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            const int8_t *q = tern_test + (size_t)i * PADDED;

            /* Route: argmax ternary_dot(test, proto[t]) */
            int best_t = 0;
            int32_t best_score = INT32_MIN;
            for (int t = 0; t < N_TILES; t++) {
                int32_t score = ternary_dot(q, tile_proto[t]);
                if (score > best_score) { best_score = score; best_t = t; }
            }
            route_hist[best_t]++;

            /* Check routing "correctness" by dominant class in tile */
            /* (informational only) */

            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            accumulate_votes(votes, best_t, i);

            int nc = select_top_k(votes, TRAIN_N, cands, K);
            for (int j = 0; j < nc; j++)
                cands[j].dot = ternary_dot(q, tern_train + (size_t)cands[j].id * PADDED);
            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);
            int pred = knn_vote(cands, nc, KNN);
            best_preds[i] = (uint8_t)pred;
            if (pred == test_labels[i]) correct++;
            if ((i+1) % 2000 == 0)
                fprintf(stderr, "  Hard: %d/%d\r", i+1, TEST_N);
        }
        fprintf(stderr, "\n");
        (void)routing_correct;
        printf("  K=%d k=%d: %.2f%%  (%.2f sec)\n",
               K, KNN, 100.0 * correct / TEST_N, now_sec() - tb2);
        printf("  Route distribution:");
        for (int t = 0; t < N_TILES; t++) printf(" t%d:%d", t, route_hist[t]);
        printf("\n\n");
        error_analysis(best_preds);
        printf("\n");
    }

    /* ----------------------------------------------------------------
     * Test C: Soft routing — merge votes from top-2 tiles
     * ---------------------------------------------------------------- */
    printf("--- Test C: Soft Routing (top-2 tiles, merged votes) ---\n");
    {
        int correct = 0;
        double tc = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            const int8_t *q = tern_test + (size_t)i * PADDED;

            /* Score all tiles, pick top-2 */
            int32_t scores[N_TILES];
            for (int t = 0; t < N_TILES; t++)
                scores[t] = ternary_dot(q, tile_proto[t]);
            int t1 = 0, t2 = 1;
            for (int t = 0; t < N_TILES; t++)
                if (scores[t] > scores[t1]) { t2 = t1; t1 = t; }
                else if (t != t1 && scores[t] > scores[t2]) t2 = t;

            memset(votes, 0, TRAIN_N * sizeof(uint32_t));
            accumulate_votes(votes, t1, i);
            accumulate_votes(votes, t2, i);

            int nc = select_top_k(votes, TRAIN_N, cands, K);
            for (int j = 0; j < nc; j++)
                cands[j].dot = ternary_dot(q, tern_train + (size_t)cands[j].id * PADDED);
            qsort(cands, (size_t)nc, sizeof(candidate_t), cmp_dots_desc);
            int pred = knn_vote(cands, nc, KNN);
            best_preds[i] = (uint8_t)pred;
            if (pred == test_labels[i]) correct++;
            if ((i+1) % 2000 == 0)
                fprintf(stderr, "  Soft: %d/%d\r", i+1, TEST_N);
        }
        fprintf(stderr, "\n");
        printf("  K=%d k=%d: %.2f%%  (%.2f sec)\n\n",
               K, KNN, 100.0 * correct / TEST_N, now_sec() - tc);
        error_analysis(best_preds);
        printf("\n");
    }

    printf("=== SUMMARY ===\n");
    printf("  See Test A/B/C above.\n");
    printf("  Global reference: 96.12%% (sstt_v2.c)\n");
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    for (int t = 0; t < N_TILES; t++)
        for (int ch = 0; ch < N_CHANNELS; ch++) free(tile_pool[t][ch]);
    for (int ch = 0; ch < N_CHANNELS; ch++) free(global_pool[ch]);
    free(votes); free(cands); free(best_preds);
    for (int i = 0; i < 6; i++) free(sigs[i]);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);
    return 0;
}
