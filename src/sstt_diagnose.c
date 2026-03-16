/*
 * sstt_diagnose.c — Cascade Error Autopsy
 *
 * Runs the bytepacked cascade (our best method, 96.28%) and performs
 * a full post-mortem on every failure. For each wrong prediction:
 *
 *   1. FAILURE MODE: where did the cascade break down?
 *      - Mode A (Vote Miss):    correct-class images NOT in top-K candidates
 *      - Mode B (Dot Override): correct class IN top-K, but wrong class wins dot ranking
 *      - Mode C (kNN Dilution): correct class has best dot match, but outvoted by wrong class
 *
 *   2. VOTE MARGIN: how far was the correct class from being chosen?
 *
 *   3. VISUAL: side-by-side ASCII art of test image, top-1 wrong, best correct
 *
 *   4. CONFUSION PAIRS: deep stats per (true, predicted) pair
 *
 *   5. HARDNESS SCORE: estimate whether each error is fixable or irreducible
 *
 * Build: make sstt_diagnose
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N         60000
#define TEST_N          10000
#define IMG_W           28
#define IMG_H           28
#define PIXELS          784
#define PADDED          800
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    9
#define N_BLOCKS        252
#define N_BVALS         27
#define SIG_PAD         256
#define BYTE_VALS       256
#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13
#define BG_JOINT        20

#define H_TRANS_PER_ROW 8
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       256
#define IG_SCALE        16
#define TOP_K           200

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
static uint8_t *train_labels, *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;
static uint8_t *px_sigs_tr, *px_sigs_te;
static uint8_t *hg_sigs_tr, *hg_sigs_te;
static uint8_t *vg_sigs_tr, *vg_sigs_te;
static uint8_t *ht_sigs_tr, *ht_sigs_te;
static uint8_t *vt_sigs_tr, *vt_sigs_te;
static uint8_t *joint_sigs_tr, *joint_sigs_te;
static uint32_t *joint_hot;
static uint16_t ig_weights[N_BLOCKS];
static uint8_t  nbr_table[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz [N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;

/* ================================================================
 *  Error record
 * ================================================================ */
#define MAX_CANDS 200
typedef struct {
    int test_idx;
    int true_lbl;
    int pred_lbl;

    /* Failure mode */
    int mode;               /* A=0, B=1, C=2 */
    /*
     * A: correct class never reached top-K by votes
     * B: correct class in top-K, but wrong class wins dot ranking
     * C: correct class has best individual dot, but wrong class has more in top-k
     */

    /* Vote phase */
    uint32_t max_correct_vote;  /* best vote count for any correct-class candidate */
    uint32_t max_wrong_vote;    /* vote count of top-1 candidate (wrong) */
    int      correct_vote_rank; /* rank of best correct-class candidate by votes (0=not in topK) */

    /* Dot phase (for the top-K candidates) */
    int32_t  best_correct_dot;  /* dot of best correct-class candidate (-1 if absent) */
    int32_t  best_wrong_dot;    /* dot of top-1 wrong candidate */

    /* Top candidates */
    int      nc;
    uint32_t cand_id[MAX_CANDS];
    int32_t  cand_dot[MAX_CANDS];
    uint32_t cand_votes[MAX_CANDS];
} error_t;

static error_t errors[TEST_N];  /* worst case: all wrong */
static int n_errors = 0;
static uint8_t all_preds[TEST_N];

/* ================================================================
 *  Feature functions (from sstt_bytecascade.c)
 * ================================================================ */
static uint8_t *load_idx_file(const char *path, uint32_t *count,
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
    raw_train_img = load_idx_file(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%strain-labels-idx1-ubyte", data_dir);
    train_labels = load_idx_file(path, &n, NULL, NULL);
    snprintf(path, sizeof(path), "%st10k-images-idx3-ubyte", data_dir);
    raw_test_img = load_idx_file(path, &n, &r, &c);
    snprintf(path, sizeof(path), "%st10k-labels-idx1-ubyte", data_dir);
    test_labels = load_idx_file(path, &n, NULL, NULL);
}
static inline int8_t clamp_trit(int v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }
static void quantize_avx2(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i thi  = _mm256_set1_epi8((char)(170 ^ 0x80));
    const __m256i tlo  = _mm256_set1_epi8((char)(85  ^ 0x80));
    const __m256i one  = _mm256_set1_epi8(1);
    for (int img = 0; img < n; img++) {
        const uint8_t *s = src + (size_t)img * PIXELS;
        int8_t *d = dst + (size_t)img * PADDED; int i;
        for (i = 0; i + 32 <= PIXELS; i += 32) {
            __m256i px = _mm256_loadu_si256((const __m256i *)(s + i));
            __m256i spx = _mm256_xor_si256(px, bias);
            __m256i pos = _mm256_cmpgt_epi8(spx, thi);
            __m256i neg = _mm256_cmpgt_epi8(tlo, spx);
            _mm256_storeu_si256((__m256i *)(d + i),
                _mm256_sub_epi8(_mm256_and_si256(pos, one),
                                _mm256_and_si256(neg, one)));
        }
        for (; i < PIXELS; i++) d[i] = s[i] > 170 ? 1 : s[i] < 85 ? -1 : 0;
        memset(d + PIXELS, 0, PADDED - PIXELS);
    }
}
static void compute_gradients(const int8_t *t, int8_t *hg, int8_t *vg, int n) {
    for (int img = 0; img < n; img++) {
        const int8_t *ti = t + (size_t)img * PADDED;
        int8_t *h = hg + (size_t)img * PADDED;
        int8_t *v = vg + (size_t)img * PADDED;
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W - 1; x++)
                h[y*IMG_W+x] = clamp_trit(ti[y*IMG_W+x+1] - ti[y*IMG_W+x]);
            h[y*IMG_W+IMG_W-1] = 0;
        }
        memset(h + PIXELS, 0, PADDED - PIXELS);
        for (int y = 0; y < IMG_H-1; y++)
            for (int x = 0; x < IMG_W; x++)
                v[y*IMG_W+x] = clamp_trit(ti[(y+1)*IMG_W+x] - ti[y*IMG_W+x]);
        memset(v + (IMG_H-1)*IMG_W, 0, IMG_W);
        memset(v + PIXELS, 0, PADDED - PIXELS);
    }
}
static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0+1)*9 + (t1+1)*3 + (t2+1));
}
static void compute_block_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for (int i = 0; i < n; i++) {
        const int8_t *img = data + (size_t)i * PADDED;
        uint8_t *sig = sigs + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int base = y*IMG_W + s*3;
                sig[y*BLKS_PER_ROW+s] = block_encode(img[base], img[base+1], img[base+2]);
            }
        memset(sig + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}
static inline uint8_t trans_enc(uint8_t a, uint8_t b) {
    int8_t a0=(a/9)-1, a1=((a/3)%3)-1, a2=(a%3)-1;
    int8_t b0=(b/9)-1, b1=((b/3)%3)-1, b2=(b%3)-1;
    return block_encode(clamp_trit(b0-a0), clamp_trit(b1-a1), clamp_trit(b2-a2));
}
static void compute_transitions(const uint8_t *bsig, int stride,
                                 uint8_t *ht, uint8_t *vt, int n) {
    for (int i = 0; i < n; i++) {
        const uint8_t *s = bsig + (size_t)i * stride;
        uint8_t *h = ht + (size_t)i * TRANS_PAD;
        uint8_t *v = vt + (size_t)i * TRANS_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int ss = 0; ss < H_TRANS_PER_ROW; ss++)
                h[y*H_TRANS_PER_ROW+ss] = trans_enc(s[y*BLKS_PER_ROW+ss], s[y*BLKS_PER_ROW+ss+1]);
        memset(h + N_HTRANS, 0xFF, TRANS_PAD - N_HTRANS);
        for (int y = 0; y < V_TRANS_PER_COL; y++)
            for (int ss = 0; ss < BLKS_PER_ROW; ss++)
                v[y*BLKS_PER_ROW+ss] = trans_enc(s[y*BLKS_PER_ROW+ss], s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v + N_VTRANS, 0xFF, TRANS_PAD - N_VTRANS);
    }
}
static uint8_t encode_d(uint8_t px, uint8_t hg, uint8_t vg, uint8_t ht, uint8_t vt) {
    int ps=((px/9)-1)+(((px/3)%3)-1)+((px%3)-1);
    int hs=((hg/9)-1)+(((hg/3)%3)-1)+((hg%3)-1);
    int vs=((vg/9)-1)+(((vg/3)%3)-1)+((vg%3)-1);
    uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3;
    uint8_t hc=hs<0?0:hs==0?1:hs<3?2:3;
    uint8_t vc=vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht!=BG_TRANS)?1<<6:0)|((vt!=BG_TRANS)?1<<7:0);
}
static void compute_joint_sigs(uint8_t *out, int n,
    const uint8_t *px_s, const uint8_t *hg_s, const uint8_t *vg_s,
    const uint8_t *ht_s, const uint8_t *vt_s) {
    for (int i = 0; i < n; i++) {
        const uint8_t *px=px_s+(size_t)i*SIG_PAD, *hg=hg_s+(size_t)i*SIG_PAD;
        const uint8_t *vg=vg_s+(size_t)i*SIG_PAD, *ht=ht_s+(size_t)i*TRANS_PAD;
        const uint8_t *vt=vt_s+(size_t)i*TRANS_PAD;
        uint8_t *os = out + (size_t)i * SIG_PAD;
        for (int y = 0; y < IMG_H; y++)
            for (int s = 0; s < BLKS_PER_ROW; s++) {
                int k = y*BLKS_PER_ROW+s;
                uint8_t ht_bv = s>0 ? ht[y*H_TRANS_PER_ROW+(s-1)] : BG_TRANS;
                uint8_t vt_bv = y>0 ? vt[(y-1)*BLKS_PER_ROW+s]     : BG_TRANS;
                os[k] = encode_d(px[k], hg[k], vg[k], ht_bv, vt_bv);
            }
        memset(os + N_BLOCKS, 0xFF, SIG_PAD - N_BLOCKS);
    }
}
static void build_hot_map(uint8_t bg) {
    joint_hot = aligned_alloc(32, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);
    memset(joint_hot, 0, (size_t)N_BLOCKS * BYTE_VALS * CLS_PAD * 4);
    for (int i = 0; i < TRAIN_N; i++) {
        int lbl = train_labels[i];
        const uint8_t *sig = joint_sigs_tr + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++)
            joint_hot[(size_t)k*BYTE_VALS*CLS_PAD + (size_t)sig[k]*CLS_PAD + lbl]++;
    }
    (void)bg;
}
static void compute_ig_weights(uint8_t bg) {
    int cc[N_CLASSES]={0};
    for (int i = 0; i < TRAIN_N; i++) cc[train_labels[i]]++;
    double h_class=0.0;
    for (int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N; if(p>0) h_class-=p*log2(p);}
    double raw[N_BLOCKS], mx=0.0;
    for (int k=0;k<N_BLOCKS;k++) {
        double h_cond=0.0;
        for (int v=0;v<BYTE_VALS;v++) {
            if ((uint8_t)v==bg) continue;
            const uint32_t *h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
            int vt=0; for(int c=0;c<N_CLASSES;c++) vt+=h[c];
            if (!vt) continue;
            double pv=(double)vt/TRAIN_N, hv=0.0;
            for(int c=0;c<N_CLASSES;c++){double pc=(double)h[c]/vt; if(pc>0) hv-=pc*log2(pc);}
            h_cond+=pv*hv;
        }
        raw[k]=h_class-h_cond; if(raw[k]>mx) mx=raw[k];
    }
    for(int k=0;k<N_BLOCKS;k++){
        ig_weights[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;
        if(!ig_weights[k]) ig_weights[k]=1;
    }
}
static void build_neighbor_table(void) {
    for (int v=0;v<BYTE_VALS;v++)
        for (int bit=0;bit<8;bit++) nbr_table[v][bit]=(uint8_t)(v^(1<<bit));
}
static void build_index(uint8_t bg) {
    memset(idx_sz, 0, sizeof(idx_sz));
    for (int i=0;i<TRAIN_N;i++) {
        const uint8_t *sig=joint_sigs_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++) if(sig[k]!=bg) idx_sz[k][sig[k]]++;
    }
    uint32_t total=0;
    for(int k=0;k<N_BLOCKS;k++) for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=total;total+=idx_sz[k][v];}
    idx_pool=malloc((size_t)total*sizeof(uint32_t));
    uint32_t (*wpos)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    memcpy(wpos, idx_off, (size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    for(int i=0;i<TRAIN_N;i++){
        const uint8_t *sig=joint_sigs_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++) if(sig[k]!=bg) idx_pool[wpos[k][sig[k]]++]=(uint32_t)i;
    }
    free(wpos);
}
static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32)
        acc=_mm256_add_epi8(acc,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),
            _mm256_load_si256((const __m256i*)(b+i))));
    __m256i lo16=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi16=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo16,hi16),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s);
    return _mm_cvtsi128_si32(s);
}

typedef struct { uint32_t id; uint32_t votes; int32_t dot; } cand_t;
static int cmp_votes_d(const void *a,const void *b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_dots_d(const void *a,const void *b){int32_t da=((const cand_t*)a)->dot,db=((const cand_t*)b)->dot;return(db>da)-(db<da);}

static int select_top_k(const uint32_t *votes, int n, cand_t *out, int k) {
    uint32_t mx=0;
    for(int i=0;i<n;i++) if(votes[i]>mx) mx=votes[i];
    if(!mx) return 0;
    int *hist=calloc((size_t)(mx+1),sizeof(int));
    for(int i=0;i<n;i++) if(votes[i]) hist[votes[i]]++;
    int cum=0,thr;
    for(thr=(int)mx;thr>=1;thr--){cum+=hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1; free(hist);
    int nc=0;
    for(int i=0;i<n&&nc<k;i++)
        if(votes[i]>=(uint32_t)thr) out[nc++]=(cand_t){(uint32_t)i,votes[i],0};
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);
    return nc;
}

/* ================================================================
 *  ASCII art: display ternary image (28×28) with label
 * ================================================================ */
static void print_ternary_img(const int8_t *img, const char *label,
                               int col_offset) {
    /* Print label header */
    printf("%*s%s\n", col_offset, "", label);
    for (int y = 0; y < IMG_H; y++) {
        printf("%*s", col_offset, "");
        for (int x = 0; x < IMG_W; x++) {
            int8_t v = img[y * IMG_W + x];
            putchar(v > 0 ? '#' : v < 0 ? '.' : ' ');
        }
        putchar('\n');
    }
}

/* Print two images side by side */
static void print_side_by_side(const int8_t *imgA, const char *labelA,
                                const int8_t *imgB, const char *labelB,
                                const int8_t *imgC, const char *labelC) {
    int gap = 4;
    /* Header row */
    printf("  %-30s%*s%-30s%*s%-30s\n",
           labelA, gap, "", labelB, gap, "", labelC);
    printf("  %-30s%*s%-30s%*s%-30s\n",
           "------------------------------", gap, "",
           "------------------------------", gap, "",
           "------------------------------");
    for (int y = 0; y < IMG_H; y++) {
        /* Image A */
        printf("  ");
        for (int x = 0; x < IMG_W; x++) {
            int8_t v = imgA[y * IMG_W + x];
            putchar(v > 0 ? '#' : v < 0 ? '.' : ' ');
        }
        /* Gap + Image B */
        printf("%*s", gap, "");
        for (int x = 0; x < IMG_W; x++) {
            int8_t v = imgB ? imgB[y * IMG_W + x] : 0;
            putchar(imgB ? (v > 0 ? '#' : v < 0 ? '.' : ' ') : ' ');
        }
        /* Gap + Image C */
        printf("%*s", gap, "");
        for (int x = 0; x < IMG_W; x++) {
            int8_t v = imgC ? imgC[y * IMG_W + x] : 0;
            putchar(imgC ? (v > 0 ? '#' : v < 0 ? '.' : ' ') : ' ');
        }
        putchar('\n');
    }
    printf("\n");
}

/* ================================================================
 *  Main diagnostic run
 * ================================================================ */
static void run_cascade(uint8_t bg) {
    uint32_t *votes = calloc(TRAIN_N, sizeof(uint32_t));
    cand_t   *cands = malloc(TOP_K * sizeof(cand_t));

    int correct = 0;
    for (int i = 0; i < TEST_N; i++) {
        memset(votes, 0, TRAIN_N * sizeof(uint32_t));
        const uint8_t *sig = joint_sigs_te + (size_t)i * SIG_PAD;

        /* Bytepacked multi-probe voting */
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = sig[k];
            if (bv == bg) continue;
            uint16_t w = ig_weights[k];
            uint16_t wh = w > 1 ? w/2 : 1;
            { uint32_t off=idx_off[k][bv]; uint16_t sz=idx_sz[k][bv];
              const uint32_t *ids=idx_pool+off;
              for(uint16_t j=0;j<sz;j++) votes[ids[j]]+=w; }
            for(int nb=0;nb<8;nb++) {
                uint8_t nv=nbr_table[bv][nb]; if(nv==bg) continue;
                uint32_t noff=idx_off[k][nv]; uint16_t nsz=idx_sz[k][nv];
                const uint32_t *nids=idx_pool+noff;
                for(uint16_t j=0;j<nsz;j++) votes[nids[j]]+=wh;
            }
        }

        /* Collect max votes for each class */
        uint32_t max_class_vote[N_CLASSES] = {0};
        for (int j = 0; j < TRAIN_N; j++)
            if (votes[j] > max_class_vote[train_labels[j]])
                max_class_vote[train_labels[j]] = votes[j];

        /* Top-K */
        int nc = select_top_k(votes, TRAIN_N, cands, TOP_K);

        /* Dot product */
        const int8_t *q = tern_test + (size_t)i * PADDED;
        for (int j = 0; j < nc; j++)
            cands[j].dot = ternary_dot(q, tern_train + (size_t)cands[j].id * PADDED);
        qsort(cands, (size_t)nc, sizeof(cand_t), cmp_dots_d);

        /* k=3 vote */
        int class_votes[N_CLASSES] = {0};
        int k3 = nc < 3 ? nc : 3;
        for (int j = 0; j < k3; j++) class_votes[train_labels[cands[j].id]]++;
        int pred = 0;
        for (int c = 1; c < N_CLASSES; c++) if (class_votes[c] > class_votes[pred]) pred = c;

        all_preds[i] = (uint8_t)pred;
        if (pred == test_labels[i]) { correct++; continue; }

        /* --- Error: record full diagnostic --- */
        error_t *e = &errors[n_errors++];
        e->test_idx   = i;
        e->true_lbl   = test_labels[i];
        e->pred_lbl   = pred;
        e->nc         = nc < MAX_CANDS ? nc : MAX_CANDS;

        e->max_wrong_vote   = max_class_vote[pred];
        e->max_correct_vote = max_class_vote[e->true_lbl];

        /* Find correct-class rank in top-K by votes */
        e->correct_vote_rank = 0; /* 0 = not present */
        for (int j = 0; j < e->nc; j++) {
            e->cand_id[j]    = cands[j].id;
            e->cand_dot[j]   = cands[j].dot;
            e->cand_votes[j] = cands[j].votes;
        }

        /* Find best correct-class candidate in top-K */
        e->best_correct_dot = INT32_MIN;
        e->best_wrong_dot   = INT32_MIN;
        int correct_in_topk = 0;
        for (int j = 0; j < e->nc; j++) {
            int lbl = train_labels[cands[j].id];
            if (lbl == e->true_lbl) {
                correct_in_topk = 1;
                if (cands[j].dot > e->best_correct_dot) {
                    e->best_correct_dot = cands[j].dot;
                    if (!e->correct_vote_rank) e->correct_vote_rank = j + 1;
                }
            }
            if (lbl == e->pred_lbl && cands[j].dot > e->best_wrong_dot)
                e->best_wrong_dot = cands[j].dot;
        }

        /* Classify failure mode */
        if (!correct_in_topk) {
            e->mode = 0; /* A: vote miss */
        } else {
            /* Check if correct class had best individual dot */
            if (e->best_correct_dot >= e->best_wrong_dot)
                e->mode = 2; /* C: correct had better dot but lost kNN count */
            else
                e->mode = 1; /* B: wrong class won dot ranking */
        }

        if ((i+1) % 2000 == 0)
            fprintf(stderr, "  %d/%d (errors so far: %d)\r", i+1, TEST_N, n_errors);
    }
    fprintf(stderr, "\n");

    printf("Cascade accuracy: %.2f%%  (%d/%d correct, %d errors)\n\n",
           100.0 * correct / TEST_N, correct, TEST_N, n_errors);

    free(votes); free(cands);
}

/* ================================================================
 *  Report
 * ================================================================ */
static void print_report(void) {
    /* --- Failure mode summary --- */
    int mode_count[3] = {0};
    for (int i = 0; i < n_errors; i++) mode_count[errors[i].mode]++;

    printf("=== FAILURE MODE BREAKDOWN ===\n");
    printf("  Mode A — Vote Miss    (correct class not in top-%d): %d  (%.1f%%)\n",
           TOP_K, mode_count[0], 100.0*mode_count[0]/n_errors);
    printf("  Mode B — Dot Override (correct in top-K, wrong wins dot): %d  (%.1f%%)\n",
           mode_count[1], 100.0*mode_count[1]/n_errors);
    printf("  Mode C — kNN Dilution (correct has best dot, wrong has more): %d  (%.1f%%)\n\n",
           mode_count[2], 100.0*mode_count[2]/n_errors);

    /* --- Confusion pairs --- */
    int conf[N_CLASSES][N_CLASSES];
    memset(conf, 0, sizeof(conf));
    for (int i = 0; i < n_errors; i++) conf[errors[i].true_lbl][errors[i].pred_lbl]++;

    printf("=== CONFUSION MATRIX (%d errors) ===\n", n_errors);
    printf("       ");
    for (int c = 0; c < N_CLASSES; c++) printf(" %4d", c);
    printf("   | Recall\n  -----");
    for (int c = 0; c < N_CLASSES; c++) (void)c, printf("-----");
    printf("---+-------\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("    %d: ", r);
        int rt = 1000; /* 1000 test images per class in MNIST */
        for (int c = 0; c < N_CLASSES; c++) printf(" %4d", conf[r][c]);
        int wrong = 0;
        for (int c = 0; c < N_CLASSES; c++) if (c != r) wrong += conf[r][c];
        printf("   | %5.1f%%\n", 100.0 * (rt - wrong) / rt);
    }

    /* --- Per-pair mode breakdown --- */
    typedef struct { int a, b, count; } pair_t;
    pair_t pairs[90]; int np = 0;
    for (int a = 0; a < N_CLASSES; a++)
        for (int b = 0; b < N_CLASSES; b++) {
            if (a == b || conf[a][b] == 0) continue;
            pairs[np++] = (pair_t){a, b, conf[a][b]};
        }
    /* Sort by count */
    for (int i = 0; i < np-1; i++)
        for (int j = i+1; j < np; j++)
            if (pairs[j].count > pairs[i].count) {
                pair_t t = pairs[i]; pairs[i] = pairs[j]; pairs[j] = t;
            }

    printf("\n=== TOP CONFUSION PAIRS (with failure modes) ===\n");
    printf("  %-12s %5s  %5s %5s %5s  %8s  %8s\n",
           "True→Pred", "Count", "ModeA", "ModeB", "ModeC",
           "VoteGap%", "DotGap");
    printf("  %-12s %5s  %5s %5s %5s  %8s  %8s\n",
           "------------", "-----", "-----", "-----", "-----",
           "--------", "-------");
    int n_show = np < 15 ? np : 15;
    for (int pi = 0; pi < n_show; pi++) {
        int a = pairs[pi].a, b = pairs[pi].b;
        int ma=0, mb=0, mc=0;
        double vote_gap_sum = 0; int32_t dot_gap_sum = 0; int cnt = 0;
        for (int i = 0; i < n_errors; i++) {
            error_t *e = &errors[i];
            if (e->true_lbl != a || e->pred_lbl != b) continue;
            if (e->mode == 0) ma++;
            else if (e->mode == 1) mb++;
            else mc++;
            if (e->max_wrong_vote > 0)
                vote_gap_sum += 100.0*(double)e->max_correct_vote/e->max_wrong_vote;
            if (e->best_correct_dot != INT32_MIN)
                dot_gap_sum += e->best_correct_dot - e->best_wrong_dot;
            cnt++;
        }
        printf("  %d \xe2\x86\x92 %-7d %5d  %5d %5d %5d  %7.1f%%  %7d\n",
               a, b, pairs[pi].count, ma, mb, mc,
               cnt > 0 ? vote_gap_sum/cnt : 0.0,
               cnt > 0 ? dot_gap_sum/cnt : 0);
    }

    /* --- Vote gap analysis --- */
    printf("\n=== VOTE GAP DISTRIBUTION (all errors) ===\n");
    printf("  (correct_votes / wrong_votes — how close was the correct class?)\n");
    int buckets[5] = {0}; /* <20%, 20-40%, 40-60%, 60-80%, 80-100%+ */
    for (int i = 0; i < n_errors; i++) {
        error_t *e = &errors[i];
        if (!e->max_wrong_vote) { buckets[0]++; continue; }
        double pct = 100.0 * e->max_correct_vote / e->max_wrong_vote;
        int b = (int)(pct / 20.0);
        if (b >= 5) b = 4;
        buckets[b]++;
    }
    printf("  < 20%%: %d  |  20-40%%: %d  |  40-60%%: %d  |  60-80%%: %d  |  80-100%%+: %d\n\n",
           buckets[0], buckets[1], buckets[2], buckets[3], buckets[4]);

    /* --- Visual autopsy: show representative errors per top pair --- */
    printf("=== VISUAL AUTOPSY (top confusion pairs) ===\n\n");
    printf("  Legend: '#' = +1 (bright/foreground)  ' ' = 0 (neutral)  '.' = -1 (dark/background)\n\n");

    int pairs_shown = 0;
    for (int pi = 0; pi < np && pairs_shown < 8; pi++) {
        int a = pairs[pi].a, b = pairs[pi].b;
        if (pairs[pi].count < 3) continue;

        printf("--- True=%d Predicted=%d  (%d errors) ---\n", a, b, pairs[pi].count);

        /* Show mode breakdown for this pair */
        int ma=0,mb=0,mc=0;
        for (int i=0;i<n_errors;i++)
            if(errors[i].true_lbl==a&&errors[i].pred_lbl==b) {
                if(errors[i].mode==0) ma++;
                else if(errors[i].mode==1) mb++;
                else mc++;
            }
        printf("  Mode A(VoteMiss)=%d  Mode B(DotOverride)=%d  Mode C(kNNDilution)=%d\n\n",
               ma, mb, mc);

        int shown = 0;
        for (int i = 0; i < n_errors && shown < 3; i++) {
            error_t *e = &errors[i];
            if (e->true_lbl != a || e->pred_lbl != b) continue;

            char labelA[64], labelB[64], labelC[64];
            snprintf(labelA, sizeof(labelA), "Test #%d (true=%d)", e->test_idx, e->true_lbl);

            /* Find best wrong candidate by dot */
            int wrong_cand = -1;
            for (int j = 0; j < e->nc; j++)
                if ((int)train_labels[e->cand_id[j]] == e->pred_lbl) {
                    if (wrong_cand < 0 || e->cand_dot[j] > e->cand_dot[wrong_cand])
                        wrong_cand = j;
                }
            /* Find best correct candidate by dot */
            int correct_cand = -1;
            for (int j = 0; j < e->nc; j++)
                if ((int)train_labels[e->cand_id[j]] == e->true_lbl) {
                    if (correct_cand < 0 || e->cand_dot[j] > e->cand_dot[correct_cand])
                        correct_cand = j;
                }

            const int8_t *test_img = tern_test + (size_t)e->test_idx * PADDED;
            const int8_t *wrong_img = wrong_cand >= 0 ?
                tern_train + (size_t)e->cand_id[wrong_cand] * PADDED : NULL;
            const int8_t *correct_img = correct_cand >= 0 ?
                tern_train + (size_t)e->cand_id[correct_cand] * PADDED : NULL;

            if (wrong_cand >= 0)
                snprintf(labelB, sizeof(labelB),
                    "Top-%d (pred=%d, dot=%d, v=%u)",
                    e->pred_lbl, e->pred_lbl,
                    e->cand_dot[wrong_cand], e->cand_votes[wrong_cand]);
            else
                snprintf(labelB, sizeof(labelB), "pred=%d (not in top-K)", e->pred_lbl);

            if (correct_cand >= 0)
                snprintf(labelC, sizeof(labelC),
                    "Best correct (true=%d, dot=%d, v=%u)",
                    e->true_lbl,
                    e->cand_dot[correct_cand], e->cand_votes[correct_cand]);
            else
                snprintf(labelC, sizeof(labelC),
                    "true=%d (NOT in top-K) mode=A", e->true_lbl);

            printf("  Mode %c  |  correct_votes=%u (%.0f%% of wrong)  |  "
                   "best_correct_dot=%d  best_wrong_dot=%d\n",
                   'A' + e->mode,
                   e->max_correct_vote,
                   e->max_wrong_vote > 0 ? 100.0*e->max_correct_vote/e->max_wrong_vote : 0.0,
                   e->best_correct_dot == INT32_MIN ? -999 : e->best_correct_dot,
                   e->best_wrong_dot);

            print_side_by_side(test_img, labelA, wrong_img, labelB, correct_img, labelC);
            shown++;
        }
        pairs_shown++;
    }

    /* --- Mode A deep dive: what positions does the correct class miss on? --- */
    printf("=== MODE A DEEP DIVE: WHY DOES THE CORRECT CLASS MISS THE VOTE? ===\n");
    printf("  For Mode A errors, the correct class had max_vote = X vs wrong max_vote.\n");
    printf("  Vote gap for Mode A errors:\n");
    for (int i = 0; i < n_errors; i++) {
        error_t *e = &errors[i];
        if (e->mode != 0) continue;
        printf("    #%d: true=%d pred=%d  correct_votes=%u  wrong_votes=%u  (%.1f%%)\n",
               e->test_idx, e->true_lbl, e->pred_lbl,
               e->max_correct_vote, e->max_wrong_vote,
               e->max_wrong_vote > 0 ? 100.0*e->max_correct_vote/e->max_wrong_vote : 0.0);
    }

    /* --- Summary of fixability --- */
    printf("\n=== FIXABILITY ASSESSMENT ===\n");
    int fixable_estimate = 0;
    /* Mode B errors where correct was in top-K but lost dot: potentially fixable with better refinement */
    fixable_estimate += mode_count[1];
    /* Mode A errors where correct_vote > 50% of wrong: close miss, fixable with better encoding */
    for (int i = 0; i < n_errors; i++) {
        error_t *e = &errors[i];
        if (e->mode == 0 && e->max_wrong_vote > 0 &&
            (double)e->max_correct_vote / e->max_wrong_vote > 0.5)
            fixable_estimate++;
    }
    printf("  Total errors:                 %d\n", n_errors);
    printf("  Mode A (vote miss):           %d — encoding doesn't find right neighbors\n", mode_count[0]);
    printf("  Mode B (dot override):        %d — correct in top-K, wrong wins dot\n", mode_count[1]);
    printf("  Mode C (kNN dilution):        %d — correct has best dot, outnumbered\n", mode_count[2]);
    printf("  Estimated fixable:            ~%d  (Mode B + close Mode A misses)\n", fixable_estimate);
    printf("  Likely irreducible:           ~%d  (Mode A with <50%% vote ratio)\n",
           n_errors - fixable_estimate);
    printf("\n  Accuracy ceiling estimate: %.2f%%  (if all fixable errors resolved)\n",
           100.0 * (TEST_N - (n_errors - fixable_estimate)) / TEST_N);
}

int main(int argc, char **argv) {
    double t0 = now_sec();
    if (argc > 1) {
        data_dir = argv[1];
        size_t len = strlen(data_dir);
        if (len && data_dir[len-1] != '/') {
            char *buf = malloc(len+2); memcpy(buf,data_dir,len);
            buf[len]='/'; buf[len+1]='\0'; data_dir=buf;
        }
    }
    const char *ds = strstr(data_dir,"fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Cascade Autopsy (%s) ===\n\n", ds);

    load_data();
    tern_train  = aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    tern_test   = aligned_alloc(32,(size_t)TEST_N *PADDED);
    hgrad_train = aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    hgrad_test  = aligned_alloc(32,(size_t)TEST_N *PADDED);
    vgrad_train = aligned_alloc(32,(size_t)TRAIN_N*PADDED);
    vgrad_test  = aligned_alloc(32,(size_t)TEST_N *PADDED);
    quantize_avx2(raw_train_img,tern_train,TRAIN_N);
    quantize_avx2(raw_test_img, tern_test, TEST_N);
    compute_gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
    compute_gradients(tern_test, hgrad_test, vgrad_test, TEST_N);

    px_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    px_sigs_te=aligned_alloc(32,(size_t)TEST_N *SIG_PAD);
    hg_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    hg_sigs_te=aligned_alloc(32,(size_t)TEST_N *SIG_PAD);
    vg_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    vg_sigs_te=aligned_alloc(32,(size_t)TEST_N *SIG_PAD);
    compute_block_sigs(tern_train, px_sigs_tr,TRAIN_N);
    compute_block_sigs(tern_test,  px_sigs_te,TEST_N);
    compute_block_sigs(hgrad_train,hg_sigs_tr,TRAIN_N);
    compute_block_sigs(hgrad_test, hg_sigs_te,TEST_N);
    compute_block_sigs(vgrad_train,vg_sigs_tr,TRAIN_N);
    compute_block_sigs(vgrad_test, vg_sigs_te,TEST_N);

    ht_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);
    ht_sigs_te=aligned_alloc(32,(size_t)TEST_N *TRANS_PAD);
    vt_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);
    vt_sigs_te=aligned_alloc(32,(size_t)TEST_N *TRANS_PAD);
    compute_transitions(px_sigs_tr,SIG_PAD,ht_sigs_tr,vt_sigs_tr,TRAIN_N);
    compute_transitions(px_sigs_te,SIG_PAD,ht_sigs_te,vt_sigs_te,TEST_N);

    joint_sigs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    joint_sigs_te=aligned_alloc(32,(size_t)TEST_N *SIG_PAD);
    compute_joint_sigs(joint_sigs_tr,TRAIN_N,px_sigs_tr,hg_sigs_tr,vg_sigs_tr,ht_sigs_tr,vt_sigs_tr);
    compute_joint_sigs(joint_sigs_te,TEST_N, px_sigs_te,hg_sigs_te,vg_sigs_te,ht_sigs_te,vt_sigs_te);

    long vc[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=joint_sigs_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[s[k]]++;}
    uint8_t bg=0; long mc=0;
    for(int v=0;v<BYTE_VALS;v++) if(vc[v]>mc){mc=vc[v];bg=(uint8_t)v;}

    build_hot_map(bg);
    compute_ig_weights(bg);
    build_neighbor_table();
    build_index(bg);

    printf("Setup complete (%.2f sec). Running cascade...\n\n", now_sec()-t0);

    run_cascade(bg);
    print_report();

    printf("\nTotal runtime: %.2f sec\n", now_sec()-t0);
    return 0;
}
