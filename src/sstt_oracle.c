/*
 * sstt_oracle.c — Multi-Specialist Oracle with Confidence Routing
 *
 * Architecture:
 *   Primary specialist:    Bytepacked cascade (Encoding D, 96.28%)
 *   Secondary specialist:  3-channel ternary cascade (v2 architecture, 96.12%)
 *
 * Routing:
 *   k=3 unanimous (3-0) from primary → done, high confidence
 *   k=3 split    (2-1)  from primary → escalate, run secondary
 *
 * Teaching (candidate pool merge):
 *   Primary top-K candidates + Secondary top-K candidates →
 *   Deduplicated union, re-ranked by multi-channel dot (px=256, vg=192) →
 *   Final k=3 majority vote
 *
 * The secondary specialist uses a different feature representation
 * (independent ternary channels vs joint bytepacked) and finds
 * different training neighbors — its top-K candidates teach the
 * primary what it missed.
 *
 * Tests:
 *   A. Primary only (bytepacked cascade, baseline)
 *   B. Hard routing: unanimous → primary, split → merge
 *   C. Full ensemble: always run both, always merge
 *
 * Build: make sstt_oracle
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
#define N_BVALS       27          /* ternary: 3^3 */
#define SIG_PAD       256
#define BYTE_VALS     256
#define N_CHANNELS    3

#define BG_PIXEL      0
#define BG_GRAD       13
#define BG_TRANS      13
#define BG_JOINT      20

#define H_TRANS_PER_ROW 8
#define N_HTRANS      (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS      (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD     256

#define IG_SCALE      16
#define TOP_K         200          /* candidates per specialist */
#define TOP_K_MERGE   300          /* merged pool upper bound */

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 *  Data
 * ================================================================ */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels,  *test_labels;
static int8_t  *tern_train,    *tern_test;
static int8_t  *hgrad_train,   *hgrad_test;
static int8_t  *vgrad_train,   *vgrad_test;

/* Per-channel block signatures (ternary specialist) */
static uint8_t *chan_tr[N_CHANNELS], *chan_te[N_CHANNELS];

/* Bytepacked joint signatures (primary specialist) */
static uint8_t *joint_tr, *joint_te;
static uint8_t *ht_tr, *ht_te, *vt_tr, *vt_te;

/* Primary: bytepacked inverted index */
static uint16_t  p_ig[N_BLOCKS];
static uint8_t   p_nbr[BYTE_VALS][8];
static uint32_t  p_off[N_BLOCKS][BYTE_VALS];
static uint16_t  p_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *p_pool;
static uint8_t   p_bg;

/* Secondary: 3-channel ternary inverted index */
static uint16_t  s_ig[N_CHANNELS][N_BLOCKS];
static uint8_t   s_nbr[N_BVALS][6];
static uint8_t   s_nbr_cnt[N_BVALS];
static uint32_t  s_off[N_CHANNELS][N_BLOCKS][N_BVALS];
static uint16_t  s_sz [N_CHANNELS][N_BLOCKS][N_BVALS];
static uint32_t *s_pool[N_CHANNELS];

/* Pre-allocated top-K histogram */
static int   *g_hist     = NULL;
static size_t g_hist_cap = 0;

/* ================================================================
 *  Feature computation
 * ================================================================ */
static uint8_t *load_idx(const char *path, uint32_t *cnt,
                          uint32_t *ro, uint32_t *co) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERR: %s\n", path); exit(1); }
    uint32_t m, n;
    fread(&m, 4, 1, f); fread(&n, 4, 1, f);
    m = __builtin_bswap32(m); n = __builtin_bswap32(n); *cnt = n;
    size_t s = 1;
    if ((m & 0xFF) >= 3) {
        uint32_t r, c; fread(&r, 4, 1, f); fread(&c, 4, 1, f);
        r = __builtin_bswap32(r); c = __builtin_bswap32(c);
        if (ro) *ro = r; if (co) *co = c; s = (size_t)r * c;
    } else { if (ro) *ro = 0; if (co) *co = 0; }
    uint8_t *d = malloc((size_t)n * s);
    fread(d, 1, (size_t)n * s, f); fclose(f); return d;
}

static void load_data(void) {
    uint32_t n, r, c; char p[256];
    snprintf(p, sizeof(p), "%strain-images-idx3-ubyte", data_dir);
    raw_train_img = load_idx(p, &n, &r, &c);
    snprintf(p, sizeof(p), "%strain-labels-idx1-ubyte", data_dir);
    train_labels  = load_idx(p, &n, NULL, NULL);
    snprintf(p, sizeof(p), "%st10k-images-idx3-ubyte",  data_dir);
    raw_test_img  = load_idx(p, &n, &r, &c);
    snprintf(p, sizeof(p), "%st10k-labels-idx1-ubyte",  data_dir);
    test_labels   = load_idx(p, &n, NULL, NULL);
}

static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

static void quantize(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));
    const __m256i one  = _mm256_set1_epi8(1);
    for (int i = 0; i < n; i++) {
        const uint8_t *s = src + (size_t)i * PIXELS;
        int8_t *d = dst + (size_t)i * PADDED; int k;
        for (k = 0; k + 32 <= PIXELS; k += 32) {
            __m256i px = _mm256_loadu_si256((const __m256i *)(s + k));
            __m256i sp = _mm256_xor_si256(px, bias);
            _mm256_storeu_si256((__m256i *)(d + k),
                _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp, thi), one),
                                _mm256_and_si256(_mm256_cmpgt_epi8(tlo, sp), one)));
        }
        for (; k < PIXELS; k++) d[k] = s[k] > 170 ? 1 : s[k] < 85 ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
    }
}

static void gradients(const int8_t *t, int8_t *h, int8_t *v, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *ti = t + (size_t)i * PADDED;
        int8_t *hi = h + (size_t)i * PADDED;
        int8_t *vi = v + (size_t)i * PADDED;
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W - 1; x++)
                hi[y*IMG_W+x] = clamp_trit(ti[y*IMG_W+x+1] - ti[y*IMG_W+x]);
            hi[y*IMG_W+IMG_W-1] = 0;
        }
        memset(hi + PIXELS, 0, PADDED - PIXELS);
        for (int y = 0; y < IMG_H-1; y++)
            for (int x = 0; x < IMG_W; x++)
                vi[y*IMG_W+x] = clamp_trit(ti[(y+1)*IMG_W+x] - ti[y*IMG_W+x]);
        memset(vi + (IMG_H-1)*IMG_W, 0, IMG_W);
        memset(vi + PIXELS, 0, PADDED - PIXELS);
    }
}

static inline uint8_t benc(int8_t a, int8_t b, int8_t c) {
    return (uint8_t)((a+1)*9 + (b+1)*3 + (c+1));
}

static void block_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int b = y*IMG_W + s*3;
                sig[y*BLKS_PER_ROW+s] = benc(img[b], img[b+1], img[b+2]);
            }
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

static inline uint8_t tenc(uint8_t a, uint8_t b) {
    int8_t a0=(a/9)-1, a1=((a/3)%3)-1, a2=(a%3)-1;
    int8_t b0=(b/9)-1, b1=((b/3)%3)-1, b2=(b%3)-1;
    return benc(clamp_trit(b0-a0), clamp_trit(b1-a1), clamp_trit(b2-a2));
}

static void transitions(const uint8_t *bs, int str,
                         uint8_t *ht, uint8_t *vt, int n) {
    for (int i = 0; i < n; i++) {
        const uint8_t *s = bs + (size_t)i * str;
        uint8_t *h = ht + (size_t)i * TRANS_PAD;
        uint8_t *v = vt + (size_t)i * TRANS_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int ss = 0; ss < H_TRANS_PER_ROW; ss++)
                h[y*H_TRANS_PER_ROW+ss] =
                    tenc(s[y*BLKS_PER_ROW+ss], s[y*BLKS_PER_ROW+ss+1]);
        memset(h + N_HTRANS, 0xFF, TRANS_PAD - N_HTRANS);
        for (int y = 0; y < V_TRANS_PER_COL; y++)
            for (int ss = 0; ss < BLKS_PER_ROW; ss++)
                v[y*BLKS_PER_ROW+ss] =
                    tenc(s[y*BLKS_PER_ROW+ss], s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v + N_VTRANS, 0xFF, TRANS_PAD - N_VTRANS);
    }
}

static uint8_t enc_d(uint8_t px, uint8_t hg, uint8_t vg,
                      uint8_t ht, uint8_t vt) {
    int ps = ((px/9)-1)+(((px/3)%3)-1)+((px%3)-1);
    int hs = ((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1);
    int vs = ((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);
    uint8_t pc = ps<0?0:ps==0?1:ps<3?2:3;
    uint8_t hc = hs<0?0:hs==0?1:hs<3?2:3;
    uint8_t vc = vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);
}

static void joint_sigs(uint8_t *out, int n,
                        const uint8_t *px, const uint8_t *hg,
                        const uint8_t *vg, const uint8_t *ht,
                        const uint8_t *vt) {
    for (int i = 0; i < n; i++) {
        const uint8_t *pi=px+(size_t)i*SIG_PAD, *hi=hg+(size_t)i*SIG_PAD;
        const uint8_t *vi=vg+(size_t)i*SIG_PAD, *hti=ht+(size_t)i*TRANS_PAD;
        const uint8_t *vti=vt+(size_t)i*TRANS_PAD;
        uint8_t *oi = out + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int k = y*BLKS_PER_ROW+s;
                uint8_t htb = s>0 ? hti[y*H_TRANS_PER_ROW+(s-1)] : BG_TRANS;
                uint8_t vtb = y>0 ? vti[(y-1)*BLKS_PER_ROW+s]     : BG_TRANS;
                oi[k] = enc_d(pi[k], hi[k], vi[k], htb, vtb);
            }
        memset(oi + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

/* ================================================================
 *  IG computation (shared utility)
 * ================================================================ */
static void compute_ig_256(const uint8_t *sigs, int n_vals, uint8_t bg,
                             uint16_t *ig_out) {
    int cc[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) cc[train_labels[i]]++;
    double hc = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        double p = (double)cc[c]/TRAIN_N; if (p > 0) hc -= p*log2(p);
    }
    double raw[N_BLOCKS], mx = 0;
    for (int k = 0; k < N_BLOCKS; k++) {
        /* Per-block class counts from training signatures */
        int *cnt = calloc((size_t)n_vals * N_CLASSES, sizeof(int));
        int *vt  = calloc(n_vals, sizeof(int));
        for (int i = 0; i < TRAIN_N; i++) {
            int v = sigs[(size_t)i * SIG_PAD + k];
            cnt[v*N_CLASSES + train_labels[i]]++;
            vt[v]++;
        }
        double hcond = 0;
        for (int v = 0; v < n_vals; v++) {
            if (!vt[v] || v == bg) continue;
            double pv = (double)vt[v]/TRAIN_N, hv = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)cnt[v*N_CLASSES+c]/vt[v];
                if (pc > 0) hv -= pc*log2(pc);
            }
            hcond += pv * hv;
        }
        raw[k] = hc - hcond;
        if (raw[k] > mx) mx = raw[k];
        free(cnt); free(vt);
    }
    for (int k = 0; k < N_BLOCKS; k++) {
        ig_out[k] = mx > 0 ? (uint16_t)(raw[k]/mx*IG_SCALE + 0.5) : 1;
        if (!ig_out[k]) ig_out[k] = 1;
    }
}

/* ================================================================
 *  Primary index (bytepacked)
 * ================================================================ */
static void build_primary(void) {
    /* Detect background */
    long vc[BYTE_VALS] = {0};
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *s = joint_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) vc[s[k]]++;
    }
    p_bg = 0; long mc = 0;
    for (int v = 0; v < BYTE_VALS; v++)
        if (vc[v] > mc) { mc = vc[v]; p_bg = (uint8_t)v; }

    compute_ig_256(joint_tr, BYTE_VALS, p_bg, p_ig);

    /* Bit-flip neighbor table */
    for (int v = 0; v < BYTE_VALS; v++)
        for (int b = 0; b < 8; b++)
            p_nbr[v][b] = (uint8_t)(v ^ (1 << b));

    /* Inverted index */
    memset(p_sz, 0, sizeof(p_sz));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *s = joint_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (s[k] != p_bg) p_sz[k][s[k]]++;
    }
    uint32_t tot = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < BYTE_VALS; v++) { p_off[k][v] = tot; tot += p_sz[k][v]; }
    p_pool = malloc((size_t)tot * sizeof(uint32_t));
    uint32_t (*wp)[BYTE_VALS] = malloc((size_t)N_BLOCKS * BYTE_VALS * 4);
    memcpy(wp, p_off, (size_t)N_BLOCKS * BYTE_VALS * 4);
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *s = joint_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            if (s[k] != p_bg) p_pool[wp[k][s[k]]++] = (uint32_t)i;
    }
    free(wp);
    printf("  Primary index: %u entries (%.1f MB)\n", tot, (double)tot*4/1048576);
}

/* ================================================================
 *  Secondary index (3-channel ternary)
 * ================================================================ */
static void build_secondary(void) {
    const uint8_t bg_vals[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};

    /* Trit-flip neighbor table */
    int trits[3] = {-1, 0, 1};
    for (int v = 0; v < N_BVALS; v++) {
        int orig[3] = {(v/9)-1, ((v/3)%3)-1, (v%3)-1};
        int nc = 0;
        for (int pos = 0; pos < 3; pos++)
            for (int alt = 0; alt < 3; alt++) {
                if (trits[alt] == orig[pos]) continue;
                int m[3] = {orig[0], orig[1], orig[2]};
                m[pos] = trits[alt];
                s_nbr[v][nc++] = benc((int8_t)m[0], (int8_t)m[1], (int8_t)m[2]);
            }
        s_nbr_cnt[v] = (uint8_t)nc;
    }

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        compute_ig_256(chan_tr[ch], N_BVALS, bg_vals[ch], s_ig[ch]);

        memset(s_sz[ch], 0, sizeof(s_sz[ch]));
        for (int i = 0; i < TRAIN_N; i++) {
            const uint8_t *sig = chan_tr[ch] + (size_t)i * SIG_PAD;
            for (int k = 0; k < N_BLOCKS; k++)
                if (sig[k] != bg_vals[ch]) s_sz[ch][k][sig[k]]++;
        }
        uint32_t tot = 0;
        for (int k = 0; k < N_BLOCKS; k++)
            for (int v = 0; v < N_BVALS; v++) {
                s_off[ch][k][v] = tot; tot += s_sz[ch][k][v];
            }
        s_pool[ch] = malloc((size_t)tot * sizeof(uint32_t));
        /* wpos on stack: 252×27×4 = 27KB — fine */
        uint32_t wpos[N_BLOCKS][N_BVALS];
        memcpy(wpos, s_off[ch], sizeof(wpos));
        for (int i = 0; i < TRAIN_N; i++) {
            const uint8_t *sig = chan_tr[ch] + (size_t)i * SIG_PAD;
            for (int k = 0; k < N_BLOCKS; k++)
                if (sig[k] != bg_vals[ch])
                    s_pool[ch][wpos[k][sig[k]]++] = (uint32_t)i;
        }
        printf("  Secondary ch%d: %u entries\n", ch, tot);
    }
}

/* ================================================================
 *  AVX2 ternary dot
 * ================================================================ */
static inline int32_t tdot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < PADDED; i += 32)
        acc = _mm256_add_epi8(acc, _mm256_sign_epi8(
            _mm256_load_si256((const __m256i *)(a+i)),
            _mm256_load_si256((const __m256i *)(b+i))));
    __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc, 1));
    __m256i s32 = _mm256_madd_epi16(_mm256_add_epi16(lo, hi), _mm256_set1_epi16(1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(s32),
                              _mm256_extracti128_si256(s32, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* ================================================================
 *  Candidate struct and helpers
 * ================================================================ */
typedef struct { uint32_t id; uint32_t votes; int32_t dot_px, dot_hg, dot_vg; int64_t combined; } cand_t;

static int cmp_votes_d(const void *a, const void *b) {
    return (int)((const cand_t *)b)->votes - (int)((const cand_t *)a)->votes;
}
static int cmp_combined_d(const void *a, const void *b) {
    int64_t da = ((const cand_t *)a)->combined;
    int64_t db = ((const cand_t *)b)->combined;
    return (db > da) - (db < da);
}

static int select_top_k(const uint32_t *votes, int n, cand_t *out, int k) {
    uint32_t mx = 0;
    for (int i = 0; i < n; i++) if (votes[i] > mx) mx = votes[i];
    if (!mx) return 0;
    if ((size_t)(mx+1) > g_hist_cap) {
        g_hist_cap = (size_t)(mx+1) + 4096;
        free(g_hist); g_hist = malloc(g_hist_cap * sizeof(int));
    }
    memset(g_hist, 0, (mx+1) * sizeof(int));
    for (int i = 0; i < n; i++) if (votes[i]) g_hist[votes[i]]++;
    int cum = 0, thr;
    for (thr = (int)mx; thr >= 1; thr--) { cum += g_hist[thr]; if (cum >= k) break; }
    if (thr < 1) thr = 1;
    int nc = 0;
    for (int i = 0; i < n && nc < k; i++)
        if (votes[i] >= (uint32_t)thr)
            out[nc++] = (cand_t){(uint32_t)i, votes[i], 0, 0, 0, 0};
    qsort(out, (size_t)nc, sizeof(cand_t), cmp_votes_d);
    return nc;
}

static int knn_vote(const cand_t *c, int nc, int k) {
    int v[N_CLASSES] = {0};
    if (k > nc) k = nc;
    for (int i = 0; i < k; i++) v[train_labels[c[i].id]]++;
    int best = 0;
    for (int c2 = 1; c2 < N_CLASSES; c2++) if (v[c2] > v[best]) best = c2;
    return best;
}

/* Compute 3-channel dots for a candidate list */
static void compute_dots(cand_t *cands, int nc, int test_idx) {
    const int8_t *qp = tern_test   + (size_t)test_idx * PADDED;
    const int8_t *qh = hgrad_test  + (size_t)test_idx * PADDED;
    const int8_t *qv = vgrad_test  + (size_t)test_idx * PADDED;
    for (int j = 0; j < nc; j++) {
        uint32_t id = cands[j].id;
        cands[j].dot_px = tdot(qp, tern_train   + (size_t)id * PADDED);
        cands[j].dot_hg = tdot(qh, hgrad_train  + (size_t)id * PADDED);
        cands[j].dot_vg = tdot(qv, vgrad_train  + (size_t)id * PADDED);
        cands[j].combined = (int64_t)256 * cands[j].dot_px
                          + (int64_t)192 * cands[j].dot_vg;
    }
}

/* Merge two candidate lists — union by id, summing votes */
static int merge_pools(const cand_t *a, int na,
                        const cand_t *b, int nb,
                        cand_t *out, int max_out) {
    int n = na < max_out ? na : max_out;
    memcpy(out, a, (size_t)n * sizeof(cand_t));
    for (int j = 0; j < nb && n < max_out; j++) {
        int found = 0;
        for (int i = 0; i < na; i++) {
            if (out[i].id == b[j].id) {
                /* Already present: accumulate votes, keep larger dot */
                out[i].votes += b[j].votes;
                found = 1; break;
            }
        }
        if (!found) out[n++] = b[j];
    }
    return n;
}

/* ================================================================
 *  Vote accumulation helpers
 * ================================================================ */
static void vote_primary(uint32_t *votes, int test_img) {
    memset(votes, 0, TRAIN_N * sizeof(uint32_t));
    const uint8_t *sig = joint_te + (size_t)test_img * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sig[k]; if (bv == p_bg) continue;
        uint16_t w = p_ig[k], wh = w > 1 ? w/2 : 1;
        { uint32_t off=p_off[k][bv]; uint16_t sz=p_sz[k][bv];
          const uint32_t *ids=p_pool+off;
          for (uint16_t j=0; j<sz; j++) votes[ids[j]]+=w; }
        for (int nb=0; nb<8; nb++) {
            uint8_t nv=p_nbr[bv][nb]; if (nv==p_bg) continue;
            uint32_t noff=p_off[k][nv]; uint16_t nsz=p_sz[k][nv];
            const uint32_t *nids=p_pool+noff;
            for (uint16_t j=0; j<nsz; j++) votes[nids[j]]+=wh;
        }
    }
}

static void vote_secondary(uint32_t *votes, int test_img) {
    memset(votes, 0, TRAIN_N * sizeof(uint32_t));
    const uint8_t bg_vals[3] = {BG_PIXEL, BG_GRAD, BG_GRAD};
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const uint8_t *sig = chan_te[ch] + (size_t)test_img * SIG_PAD;
        uint8_t bg = bg_vals[ch];
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sig[k]; if (bv == bg) continue;
            uint16_t w = s_ig[ch][k], wh = w > 1 ? w/2 : 1;
            { uint32_t off=s_off[ch][k][bv]; uint16_t sz=s_sz[ch][k][bv];
              const uint32_t *ids=s_pool[ch]+off;
              for (uint16_t j=0; j<sz; j++) votes[ids[j]]+=w; }
            for (int nb=0; nb<s_nbr_cnt[bv]; nb++) {
                uint8_t nv=s_nbr[bv][nb]; if (nv==bg) continue;
                uint32_t noff=s_off[ch][k][nv]; uint16_t nsz=s_sz[ch][k][nv];
                const uint32_t *nids=s_pool[ch]+noff;
                for (uint16_t j=0; j<nsz; j++) votes[nids[j]]+=wh;
            }
        }
    }
}

/* ================================================================
 *  Error analysis
 * ================================================================ */
static void confusion_matrix(const uint8_t *preds, const char *label) {
    int conf[N_CLASSES][N_CLASSES]; memset(conf, 0, sizeof(conf));
    for (int i = 0; i < TEST_N; i++) conf[test_labels[i]][preds[i]]++;
    printf("  %s — confusion (rows=actual, cols=predicted):\n", label);
    printf("       ");
    for (int c = 0; c < N_CLASSES; c++) printf(" %4d", c);
    printf("   | Recall\n  -----");
    for (int c = 0; c < N_CLASSES; c++) (void)c, printf("-----");
    printf("---+-------\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("    %d: ", r); int rt=0;
        for (int c = 0; c < N_CLASSES; c++) { printf(" %4d", conf[r][c]); rt+=conf[r][c]; }
        printf("   | %5.1f%%\n", rt>0?100.0*conf[r][r]/rt:0.0);
    }
    typedef struct { int a,b,count; } pair_t;
    pair_t pairs[45]; int np=0;
    for (int a=0; a<N_CLASSES; a++) for (int b=a+1; b<N_CLASSES; b++)
        pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for (int i=0; i<np-1; i++) for (int j=i+1; j<np; j++)
        if (pairs[j].count>pairs[i].count){pair_t t=pairs[i];pairs[i]=pairs[j];pairs[j]=t;}
    printf("  Top-5 confused:");
    for (int i=0; i<5&&i<np; i++)
        printf("  %d\xe2\x86\x94%d:%d", pairs[i].a, pairs[i].b, pairs[i].count);
    printf("\n");
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc, char **argv) {
    double t0 = now_sec();
    if (argc > 1) {
        data_dir = argv[1];
        size_t l = strlen(data_dir);
        if (l && data_dir[l-1] != '/') {
            char *buf = malloc(l+2); memcpy(buf, data_dir, l);
            buf[l]='/'; buf[l+1]='\0'; data_dir=buf;
        }
    }
    const char *ds = strstr(data_dir,"fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Oracle — Multi-Specialist Routing (%s) ===\n\n", ds);

    load_data();
    tern_train  = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test   = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    hgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = aligned_alloc(32, (size_t)TEST_N  * PADDED);
    quantize(raw_train_img, tern_train, TRAIN_N);
    quantize(raw_test_img,  tern_test,  TEST_N);
    gradients(tern_train, hgrad_train, vgrad_train, TRAIN_N);
    gradients(tern_test,  hgrad_test,  vgrad_test,  TEST_N);

    /* Allocate and compute all signatures */
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        chan_tr[ch] = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        chan_te[ch] = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    }
    const int8_t *raw_ch_tr[3] = {tern_train, hgrad_train, vgrad_train};
    const int8_t *raw_ch_te[3] = {tern_test,  hgrad_test,  vgrad_test};
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        block_sigs(raw_ch_tr[ch], chan_tr[ch], TRAIN_N);
        block_sigs(raw_ch_te[ch], chan_te[ch], TEST_N);
    }

    ht_tr = aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    ht_te = aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    vt_tr = aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    vt_te = aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    transitions(chan_tr[0], SIG_PAD, ht_tr, vt_tr, TRAIN_N);
    transitions(chan_te[0], SIG_PAD, ht_te, vt_te, TEST_N);

    joint_tr = aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    joint_te = aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    joint_sigs(joint_tr, TRAIN_N, chan_tr[0], chan_tr[1], chan_tr[2], ht_tr, vt_tr);
    joint_sigs(joint_te, TEST_N,  chan_te[0], chan_te[1], chan_te[2], ht_te, vt_te);

    printf("Building indices...\n");
    build_primary();
    build_secondary();
    printf("  Setup: %.2f sec\n\n", now_sec() - t0);

    uint32_t *vp  = calloc(TRAIN_N, sizeof(uint32_t));  /* primary votes */
    uint32_t *vs  = calloc(TRAIN_N, sizeof(uint32_t));  /* secondary votes */
    cand_t   *cp  = malloc(TOP_K       * sizeof(cand_t));
    cand_t   *cs  = malloc(TOP_K       * sizeof(cand_t));
    cand_t   *cm  = malloc(TOP_K_MERGE * sizeof(cand_t));
    uint8_t  *pred_A = calloc(TEST_N, 1);
    uint8_t  *pred_B = calloc(TEST_N, 1);
    uint8_t  *pred_C = calloc(TEST_N, 1);

    int corr_A=0, corr_B=0, corr_C=0;
    int n_unanimous=0, n_escalated=0;
    int corr_B_unani=0, corr_B_escal=0;

    double t_A=0, t_B=0, t_C=0;
    double ts;

    printf("Running tests...\n");
    for (int i = 0; i < TEST_N; i++) {

        /* ---- Test A: primary only (baseline) ---- */
        ts = now_sec();
        vote_primary(vp, i);
        int nc_p = select_top_k(vp, TRAIN_N, cp, TOP_K);
        compute_dots(cp, nc_p, i);
        qsort(cp, (size_t)nc_p, sizeof(cand_t), cmp_combined_d);
        int pa = knn_vote(cp, nc_p, 3);
        pred_A[i] = (uint8_t)pa;
        if (pa == test_labels[i]) corr_A++;
        t_A += now_sec() - ts;

        /* Determine if primary vote was unanimous (3-0) */
        int ballot[N_CLASSES] = {0};
        int k3 = nc_p < 3 ? nc_p : 3;
        for (int j = 0; j < k3; j++) ballot[train_labels[cp[j].id]]++;
        int top_ballot = 0;
        for (int c = 0; c < N_CLASSES; c++) if (ballot[c] > top_ballot) top_ballot = ballot[c];
        int unanimous = (top_ballot == 3);

        /* ---- Test B: routed — unanimous → A result, split → merge ---- */
        ts = now_sec();
        if (unanimous) {
            pred_B[i] = (uint8_t)pa;
            if (pa == test_labels[i]) { corr_B++; corr_B_unani++; }
            n_unanimous++;
        } else {
            vote_secondary(vs, i);
            int nc_s = select_top_k(vs, TRAIN_N, cs, TOP_K);
            compute_dots(cs, nc_s, i);
            int nc_m = merge_pools(cp, nc_p, cs, nc_s, cm, TOP_K_MERGE);
            /* Re-score merged pool with multi-channel dot */
            for (int j = 0; j < nc_m; j++)
                cm[j].combined = (int64_t)256 * cm[j].dot_px
                                + (int64_t)192 * cm[j].dot_vg;
            qsort(cm, (size_t)nc_m, sizeof(cand_t), cmp_combined_d);
            int pb = knn_vote(cm, nc_m, 3);
            pred_B[i] = (uint8_t)pb;
            if (pb == test_labels[i]) { corr_B++; corr_B_escal++; }
            n_escalated++;
        }
        t_B += now_sec() - ts;

        /* ---- Test C: full ensemble — always merge both ---- */
        ts = now_sec();
        /* Secondary votes already computed for escalated images;
         * re-compute for unanimous ones */
        if (unanimous) {
            vote_secondary(vs, i);
            int nc_s = select_top_k(vs, TRAIN_N, cs, TOP_K);
            compute_dots(cs, nc_s, i);
            int nc_m = merge_pools(cp, nc_p, cs, nc_s, cm, TOP_K_MERGE);
            for (int j = 0; j < nc_m; j++)
                cm[j].combined = (int64_t)256 * cm[j].dot_px
                                + (int64_t)192 * cm[j].dot_vg;
            qsort(cm, (size_t)nc_m, sizeof(cand_t), cmp_combined_d);
            int pc = knn_vote(cm, nc_m, 3);
            pred_C[i] = (uint8_t)pc;
            if (pc == test_labels[i]) corr_C++;
        } else {
            /* Already have merged pool from Test B — reuse */
            int pc = knn_vote(cm, /* nc_m already valid */ TOP_K_MERGE, 3);
            /* Recompute properly */
            int nc_s2 = select_top_k(vs, TRAIN_N, cs, TOP_K);
            compute_dots(cs, nc_s2, i);
            int nc_m2 = merge_pools(cp, nc_p, cs, nc_s2, cm, TOP_K_MERGE);
            for (int j = 0; j < nc_m2; j++)
                cm[j].combined = (int64_t)256 * cm[j].dot_px
                                + (int64_t)192 * cm[j].dot_vg;
            qsort(cm, (size_t)nc_m2, sizeof(cand_t), cmp_combined_d);
            pc = knn_vote(cm, nc_m2, 3);
            pred_C[i] = (uint8_t)pc;
            if (pc == test_labels[i]) corr_C++;
        }
        t_C += now_sec() - ts;

        if ((i+1) % 2000 == 0)
            fprintf(stderr, "  %d/%d  A:%.2f%%  B:%.2f%%  C:%.2f%%\r",
                    i+1, TEST_N,
                    100.0*corr_A/(i+1), 100.0*corr_B/(i+1), 100.0*corr_C/(i+1));
    }
    fprintf(stderr, "\n");

    printf("\n=== RESULTS ===\n\n");

    printf("--- Test A: Primary Only (bytepacked cascade) ---\n");
    printf("  Accuracy: %.2f%%  (%d errors)  %.2f sec\n\n",
           100.0*corr_A/TEST_N, TEST_N-corr_A, t_A);
    confusion_matrix(pred_A, "Primary only");

    printf("\n--- Test B: Routed Oracle (unanimous→primary, split→merge) ---\n");
    printf("  Accuracy:   %.2f%%  (%d errors)  %.2f sec\n",
           100.0*corr_B/TEST_N, TEST_N-corr_B, t_B);
    printf("  Unanimous:  %d images (%.1f%%) → primary path: %.2f%%\n",
           n_unanimous, 100.0*n_unanimous/TEST_N,
           n_unanimous>0 ? 100.0*corr_B_unani/n_unanimous : 0.0);
    printf("  Escalated:  %d images (%.1f%%) → merge path:   %.2f%%\n\n",
           n_escalated, 100.0*n_escalated/TEST_N,
           n_escalated>0 ? 100.0*corr_B_escal/n_escalated : 0.0);
    confusion_matrix(pred_B, "Routed oracle");

    printf("\n--- Test C: Full Ensemble (always merge both) ---\n");
    printf("  Accuracy: %.2f%%  (%d errors)  %.2f sec\n\n",
           100.0*corr_C/TEST_N, TEST_N-corr_C, t_C);
    confusion_matrix(pred_C, "Full ensemble");

    printf("\n=== ORACLE SUMMARY ===\n");
    printf("  A. Primary only:          %.2f%%  (%.2f sec)\n",
           100.0*corr_A/TEST_N, t_A);
    printf("  B. Routed oracle:         %.2f%%  (%.2f sec, %d escalated = %.1f%%)\n",
           100.0*corr_B/TEST_N, t_B, n_escalated, 100.0*n_escalated/TEST_N);
    printf("  C. Full ensemble:         %.2f%%  (%.2f sec)\n",
           100.0*corr_C/TEST_N, t_C);
    printf("\n  Oracle vs primary delta:  %+.2f pp (routed)  %+.2f pp (ensemble)\n",
           100.0*(corr_B-corr_A)/TEST_N, 100.0*(corr_C-corr_A)/TEST_N);
    printf("\nTotal runtime: %.2f sec\n", now_sec()-t0);

    free(vp); free(vs); free(cp); free(cs); free(cm);
    free(pred_A); free(pred_B); free(pred_C);
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        free(chan_tr[ch]); free(chan_te[ch]); free(s_pool[ch]);
    }
    free(p_pool); free(joint_tr); free(joint_te);
    free(ht_tr); free(ht_te); free(vt_tr); free(vt_te);
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);
    return 0;
}
