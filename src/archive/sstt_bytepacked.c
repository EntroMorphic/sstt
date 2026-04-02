/*
 * sstt_bytepacked.c — Byte-Packed Joint Channel Classification
 *
 * 8 bits = 6 ternary agreement signals + 2 state bits.
 *
 * For each spatial position, 6 channels each produce a ternary value.
 * Each trit is collapsed to 1 bit: active (non-zero) = 1, inactive (zero) = 0.
 * 6 activity bits are packed with 2 state bits into a single byte address.
 * One hot map lookup per position replaces 6 independent lookups.
 *
 * The 6 channels:
 *   Bit 0: pixel block (non-background)
 *   Bit 1: h-grad block (non-background)
 *   Bit 2: v-grad block (non-background)
 *   Bit 3: pixel h-transition (non-background)
 *   Bit 4: pixel v-transition (non-background)
 *   Bit 5: pixel block polarity (positive = 1, non-positive = 0)
 *
 * State bits (6-7): encode the pixel block value category
 *   00 = all-negative (block value 0: (-1,-1,-1))
 *   01 = mixed with zeros (block values 1-12: contains 0 trits)
 *   10 = mixed with positives (block values 14-25: contains +1 trits)
 *   11 = all-positive (block value 26: (+1,+1,+1))
 *
 * Also tests richer encodings with more bits per channel.
 *
 * Build: make sstt_bytepacked  (after: make mnist)
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

#define BLKS_PER_ROW 9
#define N_BLOCKS    252
#define N_BVALS     27
#define SIG_PAD     256
#define BG_PIXEL    0
#define BG_GRAD     13

#define H_TRANS_PER_ROW 8
#define N_HTRANS    (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS    (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD   256
#define BG_TRANS    13

/* Byte-packed: 256 values per position */
#define BYTE_VALS   256

static const char *data_dir = "data/";

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Data ---------- */
static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t *tern_train, *tern_test;
static int8_t *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;

/* Block signatures per channel */
static uint8_t *px_sigs_tr, *px_sigs_te;
static uint8_t *hg_sigs_tr, *hg_sigs_te;
static uint8_t *vg_sigs_tr, *vg_sigs_te;

/* Transition signatures */
static uint8_t *ht_sigs_tr, *ht_sigs_te;
static uint8_t *vt_sigs_tr, *vt_sigs_te;

/* Independent hot maps (for baseline) */
static uint32_t px_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t hg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t vg_hot[N_BLOCKS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* Byte-packed hot map: 252 positions × 256 byte values × 16 classes
 * = 16,515,072 bytes ≈ 15.75 MB. Heap-allocated. */
static uint32_t *joint_hot;  /* [N_BLOCKS][BYTE_VALS][CLS_PAD] */
#define JOINT_HOT_BV_STRIDE  (BYTE_VALS * CLS_PAD)
#define JOINT_HOT_ENTRY(k, bv) (joint_hot + (size_t)(k) * JOINT_HOT_BV_STRIDE + (size_t)(bv) * CLS_PAD)

/* Byte-packed signatures */
static uint8_t *joint_sigs_tr, *joint_sigs_te;

/* === Richer encoding: 3 channels × 2 bits per trit = 6 bits, no state bits needed ===
 * Bit 0-1: pixel trit encoded as (trit+1) [0,1,2]
 * Bit 2-3: hgrad trit encoded as (trit+1) [0,1,2]
 * Bit 4-5: vgrad trit encoded as (trit+1) [0,1,2]
 * Bit 6-7: 00 (reserved)
 *
 * 3 trits at 2 bits each = 6 bits → 64 values (using bits 0-5)
 * With bits 6-7: 256 values, but only 64 are meaningful (3^3 = 27 unique patterns → 64 with 2-bit encoding)
 *
 * Actually: (trit+1) ∈ {0,1,2}, packed: val = px_enc | (hg_enc << 2) | (vg_enc << 4)
 * Range: 0-63 (6 bits). State bits 6-7 available for transition info. */
#define JOINT64_VALS 64
static uint32_t *joint64_hot;  /* [N_BLOCKS][64][CLS_PAD] */
static uint8_t *joint64_sigs_tr, *joint64_sigs_te;

/* With state bits: full 256 using transitions */
static uint32_t *joint256_hot;  /* [N_BLOCKS][256][CLS_PAD] */
static uint8_t *joint256_sigs_tr, *joint256_sigs_te;

/* ---------- Infrastructure ---------- */

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

static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

/* AVX2 ternary quantization (from sstt_geom.c) */
static void quantize_avx2(const uint8_t *src, int8_t *dst, int n_images) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));
    const __m256i one  = _mm256_set1_epi8(1);

    for (int img = 0; img < n_images; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED;
        int i;
        for (i = 0; i + 32 <= PIXELS; i += 32) {
            __m256i px = _mm256_loadu_si256((const __m256i *)(s + i));
            __m256i spx = _mm256_xor_si256(px, bias);
            __m256i pos = _mm256_cmpgt_epi8(spx, thi);
            __m256i neg = _mm256_cmpgt_epi8(tlo, spx);
            __m256i p = _mm256_and_si256(pos, one);
            __m256i n_ = _mm256_and_si256(neg, one);
            __m256i result = _mm256_sub_epi8(p, n_);
            _mm256_storeu_si256((__m256i *)(d + i), result);
        }
        for (; i < PIXELS; i++)
            d[i] = s[i] > 170 ? 1 : s[i] < 85 ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
    }
}

/* AVX2 gradient computation */
static void compute_gradients_avx2(const int8_t *tern, int8_t *hg, int8_t *vg,
                                    int n_images) {
    for (int img = 0; img < n_images; img++) {
        const int8_t *t = tern + (size_t)img * PADDED;
        int8_t *h = hg + (size_t)img * PADDED;
        int8_t *v = vg + (size_t)img * PADDED;

        /* H-gradient: process 28 bytes per row */
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W - 1; x++)
                h[y * IMG_W + x] = clamp_trit(t[y * IMG_W + x + 1] - t[y * IMG_W + x]);
            h[y * IMG_W + IMG_W - 1] = 0;
        }
        memset(h + PIXELS, 0, PADDED - PIXELS);

        /* V-gradient */
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
                sig[y * BLKS_PER_ROW + s] = block_encode(img[base], img[base + 1], img[base + 2]);
            }
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

static inline uint8_t trans_enc(uint8_t a, uint8_t b) {
    int8_t a0 = (a / 9) - 1, a1 = ((a / 3) % 3) - 1, a2 = (a % 3) - 1;
    int8_t b0 = (b / 9) - 1, b1 = ((b / 3) % 3) - 1, b2 = (b % 3) - 1;
    return block_encode(clamp_trit(b0 - a0), clamp_trit(b1 - a1), clamp_trit(b2 - a2));
}

static void compute_transitions(const uint8_t *bsig, int stride,
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

/* ================================================================
 *  Byte-Packed Joint Signatures
 *
 *  Encoding A (6-activity + 2-state):
 *    Bit 0: pixel block != BG_PIXEL
 *    Bit 1: hgrad block != BG_GRAD
 *    Bit 2: vgrad block != BG_GRAD
 *    Bit 3: h-transition != BG_TRANS (use nearest valid h-trans)
 *    Bit 4: v-transition != BG_TRANS (use nearest valid v-trans)
 *    Bit 5: pixel block has any +1 trit
 *    Bit 6-7: pixel block category (00=all-neg, 01=has-zero, 10=has-pos, 11=all-pos)
 *
 *  Encoding B (3-chan × 2 bits = 6 bits, bits 6-7 for transition):
 *    Bit 0-1: pixel trit-sum sign (00=neg, 01=zero, 10=pos, 11=mixed)
 *    Bit 2-3: hgrad trit-sum sign
 *    Bit 4-5: vgrad trit-sum sign
 *    Bit 6: h-trans active
 *    Bit 7: v-trans active
 *
 *  Encoding C (full 3-trit cross-channel: 3^3 = 27 × some state):
 *    For the FIRST trit of each channel's 3-trit block:
 *    px_t0 ∈ {-1,0,+1}, hg_t0, vg_t0 → 27 combinations → 5 bits
 *    Bits 5-7: px block value modulo 8 (spatial hash)
 *    Total: 256 values capturing cross-channel first-trit + spatial variety
 * ================================================================ */

/* Encoding B: most balanced information per bit */
static uint8_t encode_joint_b(uint8_t px_bv, uint8_t hg_bv, uint8_t vg_bv,
                                uint8_t ht_bv, uint8_t vt_bv) {
    /* Decode pixel block to trit sum category */
    int8_t px_t0 = (px_bv / 9) - 1;
    int8_t px_t1 = ((px_bv / 3) % 3) - 1;
    int8_t px_t2 = (px_bv % 3) - 1;
    int px_sum = px_t0 + px_t1 + px_t2;
    uint8_t px_cat = px_sum > 0 ? 2 : px_sum < 0 ? 0 : 1;

    int8_t hg_t0 = (hg_bv / 9) - 1;
    int8_t hg_t1 = ((hg_bv / 3) % 3) - 1;
    int8_t hg_t2 = (hg_bv % 3) - 1;
    int hg_sum = hg_t0 + hg_t1 + hg_t2;
    uint8_t hg_cat = hg_sum > 0 ? 2 : hg_sum < 0 ? 0 : 1;

    int8_t vg_t0 = (vg_bv / 9) - 1;
    int8_t vg_t1 = ((vg_bv / 3) % 3) - 1;
    int8_t vg_t2 = (vg_bv % 3) - 1;
    int vg_sum = vg_t0 + vg_t1 + vg_t2;
    uint8_t vg_cat = vg_sum > 0 ? 2 : vg_sum < 0 ? 0 : 1;

    uint8_t ht_active = (ht_bv != BG_TRANS) ? 1 : 0;
    uint8_t vt_active = (vt_bv != BG_TRANS) ? 1 : 0;

    return px_cat | (hg_cat << 2) | (vg_cat << 4) | (ht_active << 6) | (vt_active << 7);
}

/* Encoding C: cross-channel first-trit + spatial hash */
static uint8_t encode_joint_c(uint8_t px_bv, uint8_t hg_bv, uint8_t vg_bv) {
    /* First trit of each channel */
    int8_t px_t0 = (px_bv / 9) - 1;  /* {-1, 0, +1} */
    int8_t hg_t0 = (hg_bv / 9) - 1;
    int8_t vg_t0 = (vg_bv / 9) - 1;

    /* Cross-channel 3^3 encoding: 27 values in bits 0-4 */
    uint8_t cross = (uint8_t)((px_t0 + 1) * 9 + (hg_t0 + 1) * 3 + (vg_t0 + 1));

    /* Bits 5-7: pixel block value mod 8 for spatial variety */
    uint8_t spatial = px_bv & 0x07;

    return cross | (spatial << 5);
}

/* Encoding D: full 3-channel 2-bit-per-trit (the user's vision)
 * 3 trits from pixel block, encoded as 2 bits each:
 *   (trit+1) gives {0,1,2}, packed into 2 bits
 * But we need cross-channel info too. So use:
 *   Bits 0-1: majority trit of pixel block (sum: neg=0, zero=1, pos=2, strong=3)
 *   Bits 2-3: majority trit of hgrad block
 *   Bits 4-5: majority trit of vgrad block
 *   Bits 6-7: transition state (00=both inactive, 01=h active, 10=v active, 11=both)
 */
static uint8_t encode_joint_d(uint8_t px_bv, uint8_t hg_bv, uint8_t vg_bv,
                                uint8_t ht_bv, uint8_t vt_bv) {
    /* Decode and compute majority trit per channel */
    int8_t pt[3] = {(px_bv/9)-1, ((px_bv/3)%3)-1, (px_bv%3)-1};
    int8_t ht_arr[3] = {(hg_bv/9)-1, ((hg_bv/3)%3)-1, (hg_bv%3)-1};
    int8_t vt_arr[3] = {(vg_bv/9)-1, ((vg_bv/3)%3)-1, (vg_bv%3)-1};

    /* Majority: sum of 3 trits, quantized to 2 bits */
    int ps = pt[0] + pt[1] + pt[2];       /* range: -3 to +3 */
    int hs = ht_arr[0] + ht_arr[1] + ht_arr[2];
    int vs = vt_arr[0] + vt_arr[1] + vt_arr[2];

    /* Map sum to 2-bit code: -3..-1 → 0, 0 → 1, 1..2 → 2, 3 → 3 */
    uint8_t pc = ps < 0 ? 0 : ps == 0 ? 1 : ps < 3 ? 2 : 3;
    uint8_t hc = hs < 0 ? 0 : hs == 0 ? 1 : hs < 3 ? 2 : 3;
    uint8_t vc = vs < 0 ? 0 : vs == 0 ? 1 : vs < 3 ? 2 : 3;

    uint8_t ht_a = (ht_bv != BG_TRANS) ? 1 : 0;
    uint8_t vt_a = (vt_bv != BG_TRANS) ? 1 : 0;

    return pc | (hc << 2) | (vc << 4) | (ht_a << 6) | (vt_a << 7);
}

static void compute_joint_sigs(uint8_t *out, int n_img,
                                const uint8_t *px_sigs, const uint8_t *hg_sigs,
                                const uint8_t *vg_sigs, const uint8_t *ht_sigs,
                                const uint8_t *vt_sigs,
                                uint8_t (*enc_fn)(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t)) {
    for (int i = 0; i < n_img; i++) {
        const uint8_t *px = px_sigs + (size_t)i * SIG_PAD;
        const uint8_t *hg = hg_sigs + (size_t)i * SIG_PAD;
        const uint8_t *vg = vg_sigs + (size_t)i * SIG_PAD;
        const uint8_t *ht = ht_sigs + (size_t)i * TRANS_PAD;
        const uint8_t *vt = vt_sigs + (size_t)i * TRANS_PAD;
        uint8_t *out_sig = out + (size_t)i * SIG_PAD;

        for (int y = 0; y < IMG_H; y++) {
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int k = y * BLKS_PER_ROW + s;

                /* Get nearest h-trans and v-trans for this block position */
                uint8_t ht_bv = (s > 0) ? ht[y * H_TRANS_PER_ROW + (s - 1)] : BG_TRANS;
                uint8_t vt_bv = (y > 0) ? vt[(y - 1) * BLKS_PER_ROW + s] : BG_TRANS;

                out_sig[k] = enc_fn(px[k], hg[k], vg[k], ht_bv, vt_bv);
            }
        }
        memset(out_sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

/* Variant for encoding C (no transitions needed) */
static void compute_joint_sigs_c(uint8_t *out, int n_img,
                                  const uint8_t *px_sigs, const uint8_t *hg_sigs,
                                  const uint8_t *vg_sigs) {
    for (int i = 0; i < n_img; i++) {
        const uint8_t *px = px_sigs + (size_t)i * SIG_PAD;
        const uint8_t *hg = hg_sigs + (size_t)i * SIG_PAD;
        const uint8_t *vg = vg_sigs + (size_t)i * SIG_PAD;
        uint8_t *out_sig = out + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            out_sig[k] = encode_joint_c(px[k], hg[k], vg[k]);
        memset(out_sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}

/* ================================================================
 *  AVX2 Bayesian classifier for byte-packed signatures
 *
 *  The hot map is [N_BLOCKS][n_bvals][CLS_PAD] with n_bvals up to 256.
 *  For each block position, one lookup + Bayesian update.
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

static int classify_bayesian(const uint8_t *sig, const uint32_t *hot,
                              int n_pos, int n_bvals, uint8_t bg) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;

    for (int k = 0; k < n_pos; k++) {
        uint8_t bv = sig[k];
        if (bv == bg) continue;
        const uint32_t *h = hot + (size_t)k * n_bvals * CLS_PAD + (size_t)bv * CLS_PAD;
        bayes_update(acc, h);
    }

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
    return best;
}

/* AVX2 additive classifier (for comparison) */
static int classify_additive_avx2(const uint8_t *sig, const uint32_t *hot,
                                    int n_pos, int n_bvals, uint8_t bg) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    for (int k = 0; k < n_pos; k++) {
        uint8_t bv = sig[k];
        if (bv == bg) continue;
        const uint32_t *h = hot + (size_t)k * n_bvals * CLS_PAD + (size_t)bv * CLS_PAD;
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256((const __m256i *)h));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256((const __m256i *)h + 1));
    }

    uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)cv, acc_lo);
    _mm256_store_si256((__m256i *)(cv + 8), acc_hi);
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (cv[c] > cv[best]) best = c;
    return best;
}

/* Multi-channel Bayesian (independent channels, for baseline) */
static int classify_3chan_bayesian(int img_idx) {
    double acc[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) acc[c] = 1.0;

    const uint8_t *ps = px_sigs_te + (size_t)img_idx * SIG_PAD;
    const uint8_t *hs = hg_sigs_te + (size_t)img_idx * SIG_PAD;
    const uint8_t *vs = vg_sigs_te + (size_t)img_idx * SIG_PAD;

    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv;
        bv = ps[k]; if (bv != BG_PIXEL) bayes_update(acc, px_hot[k][bv]);
        bv = hs[k]; if (bv != BG_GRAD)  bayes_update(acc, hg_hot[k][bv]);
        bv = vs[k]; if (bv != BG_GRAD)  bayes_update(acc, vg_hot[k][bv]);
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
    printf("=== SSTT Byte-Packed Joint Channel (%s) ===\n\n", ds);

    /* Load and prepare */
    printf("Loading and building...\n");
    load_data(data_dir);

    tern_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    tern_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    hgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    hgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);
    vgrad_train = (int8_t *)aligned_alloc(32, (size_t)TRAIN_N * PADDED);
    vgrad_test  = (int8_t *)aligned_alloc(32, (size_t)TEST_N  * PADDED);

    quantize_avx2(raw_train_img, tern_train, TRAIN_N);
    quantize_avx2(raw_test_img,  tern_test,  TEST_N);
    compute_gradients_avx2(tern_train, hgrad_train, vgrad_train, TRAIN_N);
    compute_gradients_avx2(tern_test,  hgrad_test,  vgrad_test,  TEST_N);

    /* Block signatures */
    px_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    px_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    hg_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    hg_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    vg_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
    vg_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);
    compute_block_sigs(tern_train,  px_sigs_tr, TRAIN_N);
    compute_block_sigs(tern_test,   px_sigs_te, TEST_N);
    compute_block_sigs(hgrad_train, hg_sigs_tr, TRAIN_N);
    compute_block_sigs(hgrad_test,  hg_sigs_te, TEST_N);
    compute_block_sigs(vgrad_train, vg_sigs_tr, TRAIN_N);
    compute_block_sigs(vgrad_test,  vg_sigs_te, TEST_N);

    /* Independent hot maps */
    build_hot(px_sigs_tr, N_BLOCKS, N_BVALS, (uint32_t *)px_hot, SIG_PAD);
    build_hot(hg_sigs_tr, N_BLOCKS, N_BVALS, (uint32_t *)hg_hot, SIG_PAD);
    build_hot(vg_sigs_tr, N_BLOCKS, N_BVALS, (uint32_t *)vg_hot, SIG_PAD);

    /* Transition signatures */
    ht_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    ht_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    vt_sigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * TRANS_PAD);
    vt_sigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * TRANS_PAD);
    compute_transitions(px_sigs_tr, SIG_PAD, ht_sigs_tr, vt_sigs_tr, TRAIN_N);
    compute_transitions(px_sigs_te, SIG_PAD, ht_sigs_te, vt_sigs_te, TEST_N);

    double t1 = now_sec();
    printf("  Built (%.2f sec)\n\n", t1 - t0);

    /* --- Baseline: 3-chan independent Bayesian --- */
    printf("--- Baseline: 3-Channel Independent Bayesian ---\n");
    int base_correct = 0;
    double tb0 = now_sec();
    for (int i = 0; i < TEST_N; i++)
        if (classify_3chan_bayesian(i) == test_labels[i]) base_correct++;
    double tb1 = now_sec();
    printf("  Accuracy: %.2f%% (%.0f ns/query)\n\n",
           100.0 * base_correct / TEST_N, (tb1 - tb0) * 1e9 / TEST_N);

    /* === Test encodings === */
    struct {
        const char *name;
        int n_bvals;
        uint8_t bg;
    } tests[] = {
        {"B: 3×2bit trit-sum + 2 trans bits", BYTE_VALS, 0xFF}, /* no single bg */
        {"C: cross-channel first-trit + hash", BYTE_VALS, 0xFF},
        {"D: 3×2bit majority + 2 trans bits", BYTE_VALS, 0xFF},
    };
    int n_tests = 3;

    for (int ti = 0; ti < n_tests; ti++) {
        printf("--- Encoding %s ---\n", tests[ti].name);

        /* Compute joint signatures */
        uint8_t *jsigs_tr = (uint8_t *)aligned_alloc(32, (size_t)TRAIN_N * SIG_PAD);
        uint8_t *jsigs_te = (uint8_t *)aligned_alloc(32, (size_t)TEST_N  * SIG_PAD);

        if (ti == 0) { /* Encoding B */
            compute_joint_sigs(jsigs_tr, TRAIN_N, px_sigs_tr, hg_sigs_tr, vg_sigs_tr, ht_sigs_tr, vt_sigs_tr, encode_joint_b);
            compute_joint_sigs(jsigs_te, TEST_N,  px_sigs_te, hg_sigs_te, vg_sigs_te, ht_sigs_te, vt_sigs_te, encode_joint_b);
        } else if (ti == 1) { /* Encoding C */
            compute_joint_sigs_c(jsigs_tr, TRAIN_N, px_sigs_tr, hg_sigs_tr, vg_sigs_tr);
            compute_joint_sigs_c(jsigs_te, TEST_N,  px_sigs_te, hg_sigs_te, vg_sigs_te);
        } else { /* Encoding D */
            compute_joint_sigs(jsigs_tr, TRAIN_N, px_sigs_tr, hg_sigs_tr, vg_sigs_tr, ht_sigs_tr, vt_sigs_tr, encode_joint_d);
            compute_joint_sigs(jsigs_te, TEST_N,  px_sigs_te, hg_sigs_te, vg_sigs_te, ht_sigs_te, vt_sigs_te, encode_joint_d);
        }

        /* Count unique values used */
        int used[256] = {0};
        for (int i = 0; i < TRAIN_N; i++) {
            const uint8_t *sig = jsigs_tr + (size_t)i * SIG_PAD;
            for (int k = 0; k < N_BLOCKS; k++) used[sig[k]] = 1;
        }
        int n_used = 0;
        for (int v = 0; v < 256; v++) n_used += used[v];
        printf("  Unique byte values used: %d / 256\n", n_used);

        /* Build joint hot map */
        int nbv = tests[ti].n_bvals;
        uint32_t *jhot = aligned_alloc(32, (size_t)N_BLOCKS * nbv * CLS_PAD * 4);
        build_hot(jsigs_tr, N_BLOCKS, nbv, jhot, SIG_PAD);
        printf("  Hot map size: %zu MB\n",
               (size_t)N_BLOCKS * nbv * CLS_PAD * 4 / (1024 * 1024));

        /* Determine background (most common value across all images) */
        long val_counts[256] = {0};
        for (int i = 0; i < TRAIN_N; i++) {
            const uint8_t *sig = jsigs_tr + (size_t)i * SIG_PAD;
            for (int k = 0; k < N_BLOCKS; k++) val_counts[sig[k]]++;
        }
        uint8_t bg = 0;
        long max_count = 0;
        for (int v = 0; v < 256; v++)
            if (val_counts[v] > max_count) { max_count = val_counts[v]; bg = (uint8_t)v; }
        printf("  Auto-detected background: %d (%.1f%% of all entries)\n",
               bg, 100.0 * max_count / ((long)TRAIN_N * N_BLOCKS));

        /* Classify: additive */
        int add_correct = 0;
        double ta0 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *sig = jsigs_te + (size_t)i * SIG_PAD;
            if (classify_additive_avx2(sig, jhot, N_BLOCKS, nbv, bg) == test_labels[i])
                add_correct++;
        }
        double ta1 = now_sec();
        printf("  Additive:  %.2f%% (%.0f ns/query)\n",
               100.0 * add_correct / TEST_N, (ta1 - ta0) * 1e9 / TEST_N);

        /* Classify: Bayesian */
        int bay_correct = 0;
        double ty0 = now_sec();
        for (int i = 0; i < TEST_N; i++) {
            const uint8_t *sig = jsigs_te + (size_t)i * SIG_PAD;
            if (classify_bayesian(sig, jhot, N_BLOCKS, nbv, bg) == test_labels[i])
                bay_correct++;
        }
        double ty1 = now_sec();
        printf("  Bayesian:  %.2f%% (%.0f ns/query, %+.2f pp vs baseline)\n\n",
               100.0 * bay_correct / TEST_N, (ty1 - ty0) * 1e9 / TEST_N,
               100.0 * bay_correct / TEST_N - 100.0 * base_correct / TEST_N);

        free(jsigs_tr); free(jsigs_te); free(jhot);
    }

    /* Summary */
    printf("=== SUMMARY ===\n");
    printf("  3-chan independent Bayesian: %.2f%%  (3 lookups/position, %zu KB model)\n",
           100.0 * base_correct / TEST_N,
           (size_t)3 * N_BLOCKS * N_BVALS * CLS_PAD * 4 / 1024);
    printf("  Byte-packed:                see above  (1 lookup/position, ~16 MB model)\n");
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    /* Cleanup */
    free(tern_train); free(tern_test);
    free(hgrad_train); free(hgrad_test);
    free(vgrad_train); free(vgrad_test);
    free(px_sigs_tr); free(px_sigs_te);
    free(hg_sigs_tr); free(hg_sigs_te);
    free(vg_sigs_tr); free(vg_sigs_te);
    free(ht_sigs_tr); free(ht_sigs_te);
    free(vt_sigs_tr); free(vt_sigs_te);
    free(raw_train_img); free(raw_test_img);
    free(train_labels); free(test_labels);

    return 0;
}
