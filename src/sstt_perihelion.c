/*
 * sstt_perihelion.c — Information Perihelion
 *
 * The Hybrid (2 sec) tells the Cascade (300 sec) where to look.
 * The Cascade doesn't recompute the orbit — just the final approach.
 *
 * Architecture:
 *   Stage 1: Transition Bayesian (2 μs/image) → class posterior + top-3 classes
 *   Stage 2: Class-filtered IG-ordered cascade → votes only on top-3 class
 *            training images, only top-K most informative blocks
 *   Stage 3: Dot product refinement → k=3 majority vote
 *
 * The Hybrid's posterior is the Cascade's prior.
 *
 * Build: make sstt_perihelion  (after: make mnist)
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
#define BG_TRANS    13

#define H_TRANS_PER_ROW 8
#define N_HTRANS    (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 27
#define N_VTRANS    (BLKS_PER_ROW * V_TRANS_PER_COL)
#define HTRANS_PAD  256
#define VTRANS_PAD  256

#define MAX_K       500

static const char *data_dir = "data/";
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* --- Data --- */
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
static uint8_t *px_ht_train, *px_ht_test, *px_vt_train, *px_vt_test;
static uint32_t px_ht_hot[N_HTRANS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));
static uint32_t px_vt_hot[N_VTRANS][N_BVALS][CLS_PAD] __attribute__((aligned(32)));

/* Per-class training indices (for class-filtered cascade) */
static int *class_idx[N_CLASSES];
static int class_count[N_CLASSES];

/* IG-ordered block indices */
static int ig_order[N_BLOCKS];

/* Inverted index per block per class (class-filtered) */
static uint32_t *idx_pool;
static uint32_t idx_off[N_BLOCKS][N_BVALS];
static uint16_t idx_sz[N_BLOCKS][N_BVALS];

/* === Infrastructure (compact) === */
static uint8_t *load_idx(const char *p,uint32_t *cnt,uint32_t *r,uint32_t *c){
    FILE *f=fopen(p,"rb");if(!f){fprintf(stderr,"ERROR: %s\n",p);exit(1);}
    uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}
    m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;
    int nd=m&0xFF;size_t is=1;
    if(nd>=3){uint32_t rr,cc;if(fread(&rr,4,1,f)!=1||fread(&cc,4,1,f)!=1){fclose(f);exit(1);}rr=__builtin_bswap32(rr);cc=__builtin_bswap32(cc);if(r)*r=rr;if(c)*c=cc;is=(size_t)rr*cc;}
    else{if(r)*r=0;if(c)*c=0;}
    size_t t=(size_t)n*is;uint8_t *d=malloc(t);if(!d||fread(d,1,t,f)!=t){fclose(f);exit(1);}
    fclose(f);return d;}
static void load_data(const char *dir){uint32_t n,r,c;char p[256];
    snprintf(p,256,"%strain-images-idx3-ubyte",dir);raw_train_img=load_idx(p,&n,&r,&c);
    snprintf(p,256,"%strain-labels-idx1-ubyte",dir);train_labels=load_idx(p,&n,NULL,NULL);
    snprintf(p,256,"%st10k-images-idx3-ubyte",dir);raw_test_img=load_idx(p,&n,&r,&c);
    snprintf(p,256,"%st10k-labels-idx1-ubyte",dir);test_labels=load_idx(p,&n,NULL,NULL);}
static void quantize_one(const uint8_t *s,int8_t *d){for(int i=0;i<PIXELS;i++)d[i]=s[i]>170?1:s[i]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}
static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}
static void grad_one(const int8_t *t,int8_t *hg,int8_t *vg){
    for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hg[y*IMG_W+x]=clamp_trit(t[y*IMG_W+x+1]-t[y*IMG_W+x]);hg[y*IMG_W+IMG_W-1]=0;}
    memset(hg+PIXELS,0,PADDED-PIXELS);
    for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vg[y*IMG_W+x]=clamp_trit(t[(y+1)*IMG_W+x]-t[y*IMG_W+x]);
    memset(vg+(IMG_H-1)*IMG_W,0,IMG_W);memset(vg+PIXELS,0,PADDED-PIXELS);}
static void init_all(void){
    tern_train=(int8_t*)aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_test=(int8_t*)aligned_alloc(32,(size_t)TEST_N*PADDED);
    hgrad_train=(int8_t*)aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgrad_test=(int8_t*)aligned_alloc(32,(size_t)TEST_N*PADDED);
    vgrad_train=(int8_t*)aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgrad_test=(int8_t*)aligned_alloc(32,(size_t)TEST_N*PADDED);
    for(int i=0;i<TRAIN_N;i++){quantize_one(raw_train_img+(size_t)i*PIXELS,tern_train+(size_t)i*PADDED);
        grad_one(tern_train+(size_t)i*PADDED,hgrad_train+(size_t)i*PADDED,vgrad_train+(size_t)i*PADDED);}
    for(int i=0;i<TEST_N;i++){quantize_one(raw_test_img+(size_t)i*PIXELS,tern_test+(size_t)i*PADDED);
        grad_one(tern_test+(size_t)i*PADDED,hgrad_test+(size_t)i*PADDED,vgrad_test+(size_t)i*PADDED);}}
static inline uint8_t block_encode(int8_t t0,int8_t t1,int8_t t2){return(uint8_t)((t0+1)*9+(t1+1)*3+(t2+1));}
static void compute_2d_sigs(const int8_t *data,uint8_t *sigs,int n){
    for(int i=0;i<n;i++){const int8_t *img=data+(size_t)i*PADDED;uint8_t *sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b=y*IMG_W+s*BLK_W;
            sig[y*BLKS_PER_ROW+s]=block_encode(img[b],img[b+1],img[b+2]);}
        memset(sig+N_BLOCKS,0xFF,SIG_PAD-N_BLOCKS);}}
static void compute_all_sigs(void){
    px_train_sigs=(uint8_t*)aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);px_test_sigs=(uint8_t*)aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_train_sigs=(uint8_t*)aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);hg_test_sigs=(uint8_t*)aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_train_sigs=(uint8_t*)aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);vg_test_sigs=(uint8_t*)aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    compute_2d_sigs(tern_train,px_train_sigs,TRAIN_N);compute_2d_sigs(tern_test,px_test_sigs,TEST_N);
    compute_2d_sigs(hgrad_train,hg_train_sigs,TRAIN_N);compute_2d_sigs(hgrad_test,hg_test_sigs,TEST_N);
    compute_2d_sigs(vgrad_train,vg_train_sigs,TRAIN_N);compute_2d_sigs(vgrad_test,vg_test_sigs,TEST_N);}
static void build_hot(const uint8_t *sigs,int npos,uint32_t *hot,int stride){
    memset(hot,0,sizeof(uint32_t)*(size_t)npos*N_BVALS*CLS_PAD);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t *s=sigs+(size_t)i*stride;
        for(int k=0;k<npos;k++)hot[(size_t)k*N_BVALS*CLS_PAD+(size_t)s[k]*CLS_PAD+l]++;}}
static inline uint8_t trans_enc(uint8_t a,uint8_t b){
    int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;
    return block_encode(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));}
static void compute_trans(const uint8_t *bsig,int stride,uint8_t *ht,uint8_t *vt,int n){
    for(int i=0;i<n;i++){const uint8_t *s=bsig+(size_t)i*stride;
        uint8_t *h=ht+(size_t)i*HTRANS_PAD,*v=vt+(size_t)i*VTRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)
            h[y*H_TRANS_PER_ROW+ss]=trans_enc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,HTRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)
            v[y*BLKS_PER_ROW+ss]=trans_enc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,VTRANS_PAD-N_VTRANS);}}

/* === Bayesian helpers === */
static inline void bayes(double *acc,const uint32_t *h){
    double mx=0;for(int c=0;c<N_CLASSES;c++){acc[c]*=((double)h[c]+0.5);if(acc[c]>mx)mx=acc[c];}
    if(mx>1e10)for(int c=0;c<N_CLASSES;c++)acc[c]/=mx;}

/* === Ternary dot product === */
static inline int32_t ternary_dot(const int8_t *a,const int8_t *b){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32){__m256i va=_mm256_load_si256((const __m256i*)(a+i));
        __m256i vb=_mm256_load_si256((const __m256i*)(b+i));
        acc=_mm256_add_epi8(acc,_mm256_sign_epi8(va,vb));}
    __m256i l=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i h=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s16=_mm256_add_epi16(l,h);
    __m256i s32=_mm256_madd_epi16(s16,_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);}

/* ================================================================
 *  Stage 1: Transition Bayesian → class posterior + top-N classes
 * ================================================================ */

static void bayesian_classify(int img_idx, double posterior[N_CLASSES]) {
    for (int c = 0; c < N_CLASSES; c++) posterior[c] = 1.0;
    const uint8_t *ht = px_ht_test + (size_t)img_idx * HTRANS_PAD;
    const uint8_t *vt = px_vt_test + (size_t)img_idx * VTRANS_PAD;
    for (int k = 0; k < N_HTRANS; k++) {
        uint8_t bv = ht[k]; if (bv != BG_TRANS)
            bayes(posterior, px_ht_hot[k][bv]);
    }
    for (int k = 0; k < N_VTRANS; k++) {
        uint8_t bv = vt[k]; if (bv != BG_TRANS)
            bayes(posterior, px_vt_hot[k][bv]);
    }
    /* Normalize */
    double sum = 0;
    for (int c = 0; c < N_CLASSES; c++) sum += posterior[c];
    for (int c = 0; c < N_CLASSES; c++) posterior[c] /= sum;
}

static void get_top_classes(const double *posterior, int *top, int n_top) {
    for (int i = 0; i < n_top; i++) top[i] = i;
    for (int s = 0; s < n_top; s++)
        for (int c = s + 1; c < N_CLASSES; c++)
            if (posterior[c] > posterior[top[s]]) {
                int tmp = top[s]; top[s] = top[c]; top[c] = tmp;
            }
}

/* ================================================================
 *  Stage 2: Class-Filtered IG-Ordered Cascade
 * ================================================================ */

typedef struct { uint32_t id; uint16_t votes; int32_t dot; } cand_t;

static int perihelion_classify(int img_idx, const int *top_classes, int n_top,
                                int n_blocks_vote, int n_dot) {
    /* Vote only on training images of the top-N classes */
    int total_candidates = 0;
    for (int ti = 0; ti < n_top; ti++)
        total_candidates += class_count[top_classes[ti]];

    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    const uint8_t *qsig = px_test_sigs + (size_t)img_idx * SIG_PAD;

    /* Vote using top n_blocks_vote IG-ordered blocks */
    for (int bi = 0; bi < n_blocks_vote && bi < N_BLOCKS; bi++) {
        int k = ig_order[bi];
        uint8_t bv = qsig[k];
        if (bv == BG_PIXEL) continue;
        uint32_t off = idx_off[k][bv];
        uint16_t sz = idx_sz[k][bv];
        const uint32_t *ids = idx_pool + off;
        for (uint16_t j = 0; j < sz; j++) {
            uint32_t tid = ids[j];
            /* Only vote for images in top-N classes */
            int lbl = train_labels[tid];
            int in_top = 0;
            for (int ti = 0; ti < n_top; ti++)
                if (lbl == top_classes[ti]) { in_top = 1; break; }
            if (in_top) votes[tid]++;
        }
    }

    /* Select top candidates by votes */
    cand_t cands[MAX_K];
    int nc = 0;
    uint16_t max_v = 0;
    for (int j = 0; j < TRAIN_N; j++)
        if (votes[j] > max_v) max_v = votes[j];

    if (max_v == 0) { free(votes); return top_classes[0]; }

    int *hist = calloc(max_v + 2, sizeof(int));
    for (int j = 0; j < TRAIN_N; j++) hist[votes[j]]++;
    int cum = 0, thr;
    for (thr = (int)max_v; thr >= 1; thr--) { cum += hist[thr]; if (cum >= n_dot) break; }
    if (thr < 1) thr = 1;
    free(hist);

    for (int j = 0; j < TRAIN_N && nc < n_dot; j++)
        if (votes[j] >= (uint16_t)thr) {
            cands[nc].id = (uint32_t)j;
            cands[nc].votes = votes[j];
            nc++;
        }
    free(votes);

    if (nc == 0) return top_classes[0];

    /* Dot product refinement */
    const int8_t *query = tern_test + (size_t)img_idx * PADDED;
    for (int j = 0; j < nc; j++)
        cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);

    /* Sort by dot */
    for (int a = 0; a < nc - 1; a++)
        for (int b = a + 1; b < nc; b++)
            if (cands[b].dot > cands[a].dot) { cand_t t = cands[a]; cands[a] = cands[b]; cands[b] = t; }

    /* k=3 majority vote */
    int kk = nc < 3 ? nc : 3;
    int cv[N_CLASSES] = {0};
    for (int j = 0; j < kk; j++) cv[train_labels[cands[j].id]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[best]) best = c;
    return best;
}

/* Full cascade (unfiltered, all blocks, for comparison) */
static int full_cascade(int img_idx) {
    uint16_t *votes = calloc(TRAIN_N, sizeof(uint16_t));
    const uint8_t *qsig = px_test_sigs + (size_t)img_idx * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = qsig[k]; if (bv == BG_PIXEL) continue;
        uint32_t off = idx_off[k][bv]; uint16_t sz = idx_sz[k][bv];
        const uint32_t *ids = idx_pool + off;
        for (uint16_t j = 0; j < sz; j++) votes[ids[j]]++;
    }
    cand_t cands[MAX_K]; int nc = 0;
    uint16_t max_v = 0;
    for (int j = 0; j < TRAIN_N; j++) if (votes[j] > max_v) max_v = votes[j];
    int *hist = calloc(max_v + 2, sizeof(int));
    for (int j = 0; j < TRAIN_N; j++) hist[votes[j]]++;
    int cum = 0, thr;
    for (thr = (int)max_v; thr >= 1; thr--) { cum += hist[thr]; if (cum >= MAX_K) break; }
    if (thr < 1) thr = 1; free(hist);
    for (int j = 0; j < TRAIN_N && nc < MAX_K; j++)
        if (votes[j] >= (uint16_t)thr) { cands[nc].id = (uint32_t)j; cands[nc].votes = votes[j]; nc++; }
    free(votes);
    const int8_t *query = tern_test + (size_t)img_idx * PADDED;
    for (int j = 0; j < nc; j++)
        cands[j].dot = ternary_dot(query, tern_train + (size_t)cands[j].id * PADDED);
    for (int a = 0; a < nc - 1; a++) for (int b = a + 1; b < nc; b++)
        if (cands[b].dot > cands[a].dot) { cand_t t = cands[a]; cands[a] = cands[b]; cands[b] = t; }
    int kk = nc < 3 ? nc : 3, cv[N_CLASSES] = {0};
    for (int j = 0; j < kk; j++) cv[train_labels[cands[j].id]]++;
    int best = 0; for (int c = 1; c < N_CLASSES; c++) if (cv[c] > cv[best]) best = c;
    return best;
}

/* === IG computation === */
static void compute_ig_order(void) {
    double class_prior[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) class_prior[train_labels[i]]++;
    double H = 0;
    for (int c = 0; c < N_CLASSES; c++) {
        class_prior[c] /= TRAIN_N;
        if (class_prior[c] > 0) H -= class_prior[c] * log2(class_prior[c]);
    }
    double ig[N_BLOCKS];
    for (int k = 0; k < N_BLOCKS; k++) {
        int bvc[N_BVALS] = {0}, bvcc[N_BVALS][N_CLASSES];
        memset(bvcc, 0, sizeof(bvcc));
        for (int i = 0; i < TRAIN_N; i++) {
            uint8_t bv = px_train_sigs[i * SIG_PAD + k];
            bvc[bv]++; bvcc[bv][train_labels[i]]++;
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
        ig[k] = H - Hc;
    }
    for (int k = 0; k < N_BLOCKS; k++) ig_order[k] = k;
    for (int i = 1; i < N_BLOCKS; i++) {
        int key = ig_order[i]; double kv = ig[key]; int j = i - 1;
        while (j >= 0 && ig[ig_order[j]] < kv) { ig_order[j + 1] = ig_order[j]; j--; }
        ig_order[j + 1] = key;
    }
}

/* === Build index === */
static void build_idx(void) {
    memset(idx_sz, 0, sizeof(idx_sz));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *s = px_train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) idx_sz[k][s[k]]++;
    }
    uint32_t total = 0;
    for (int k = 0; k < N_BLOCKS; k++)
        for (int v = 0; v < N_BVALS; v++) { idx_off[k][v] = total; total += idx_sz[k][v]; }
    idx_pool = malloc((size_t)total * sizeof(uint32_t));
    uint32_t wp[N_BLOCKS][N_BVALS]; memcpy(wp, idx_off, sizeof(idx_off));
    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *s = px_train_sigs + (size_t)i * SIG_PAD;
        for (int k = 0; k < N_BLOCKS; k++) idx_pool[wp[k][s[k]]++] = (uint32_t)i;
    }
}

/* ================================================================ */
int main(int argc, char **argv) {
    double t0 = now_sec();
    if (argc > 1) { data_dir = argv[1]; size_t l = strlen(data_dir);
        if (l > 0 && data_dir[l-1] != '/') { char *b = malloc(l + 2); memcpy(b, data_dir, l); b[l] = '/'; b[l+1] = 0; data_dir = b; }}
    const char *ds = strstr(data_dir, "fashion") ? "Fashion-MNIST" : "MNIST";
    printf("=== SSTT Perihelion: Hybrid-Guided Cascade (%s) ===\n\n", ds);

    printf("Building...\n");
    load_data(data_dir); init_all(); compute_all_sigs();
    build_hot(px_train_sigs, N_BLOCKS, (uint32_t*)px_hot, SIG_PAD);
    build_hot(hg_train_sigs, N_BLOCKS, (uint32_t*)hg_hot, SIG_PAD);
    build_hot(vg_train_sigs, N_BLOCKS, (uint32_t*)vg_hot, SIG_PAD);
    px_ht_train=(uint8_t*)aligned_alloc(32,(size_t)TRAIN_N*HTRANS_PAD);
    px_ht_test=(uint8_t*)aligned_alloc(32,(size_t)TEST_N*HTRANS_PAD);
    px_vt_train=(uint8_t*)aligned_alloc(32,(size_t)TRAIN_N*VTRANS_PAD);
    px_vt_test=(uint8_t*)aligned_alloc(32,(size_t)TEST_N*VTRANS_PAD);
    compute_trans(px_train_sigs,SIG_PAD,px_ht_train,px_vt_train,TRAIN_N);
    compute_trans(px_test_sigs,SIG_PAD,px_ht_test,px_vt_test,TEST_N);
    build_hot(px_ht_train,N_HTRANS,(uint32_t*)px_ht_hot,HTRANS_PAD);
    build_hot(px_vt_train,N_VTRANS,(uint32_t*)px_vt_hot,VTRANS_PAD);
    compute_ig_order();
    build_idx();

    /* Per-class training indices */
    memset(class_count, 0, sizeof(class_count));
    for (int i = 0; i < TRAIN_N; i++) class_count[train_labels[i]]++;
    int fill[N_CLASSES] = {0};
    for (int c = 0; c < N_CLASSES; c++) class_idx[c] = malloc(class_count[c] * sizeof(int));
    for (int i = 0; i < TRAIN_N; i++) { int c = train_labels[i]; class_idx[c][fill[c]++] = i; }

    double t1 = now_sec();
    printf("  Done (%.2f sec)\n\n", t1 - t0);

    /* --- Bayesian baseline --- */
    printf("--- Stage 1: Transition Bayesian (fast path) ---\n");
    int bayes_correct = 0;
    double tb0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        double post[N_CLASSES];
        bayesian_classify(i, post);
        int best = 0;
        for (int c = 1; c < N_CLASSES; c++) if (post[c] > post[best]) best = c;
        if (best == test_labels[i]) bayes_correct++;
    }
    double tb1 = now_sec();
    printf("  Accuracy: %.2f%% (%.2f sec)\n\n", 100.0 * bayes_correct / TEST_N, tb1 - tb0);

    /* --- Full cascade baseline --- */
    printf("--- Full Cascade (no guidance) ---\n");
    int full_correct = 0;
    double tf0 = now_sec();
    for (int i = 0; i < TEST_N; i++)
        if (full_cascade(i) == test_labels[i]) full_correct++;
    double tf1 = now_sec();
    printf("  Accuracy: %.2f%% (%.2f sec)\n\n", 100.0 * full_correct / TEST_N, tf1 - tf0);

    /* --- Perihelion: Bayesian-guided cascade --- */
    printf("--- Perihelion: Bayesian-Guided Cascade ---\n");
    printf("  Bayesian → top-N classes → IG-ordered vote → dot → k=3\n\n");

    int top_ns[] = {2, 3, 4, 5};
    int n_blks[] = {50, 100, 252};
    int n_dots[] = {50, 200, 500};

    printf("  %-5s %-6s %-6s | %-10s | %-10s | %-10s\n",
           "TopN", "Blks", "Dots", "Accuracy", "Time", "Speedup");
    printf("  ------+------+------+------------+------------+----------\n");

    for (int ti = 0; ti < 4; ti++) {
        for (int bi = 0; bi < 3; bi++) {
            for (int di = 0; di < 3; di++) {
                int correct = 0;
                double ts0 = now_sec();
                for (int i = 0; i < TEST_N; i++) {
                    double post[N_CLASSES];
                    bayesian_classify(i, post);
                    int top[N_CLASSES];
                    get_top_classes(post, top, top_ns[ti]);
                    if (perihelion_classify(i, top, top_ns[ti],
                            n_blks[bi], n_dots[di]) == test_labels[i])
                        correct++;
                }
                double ts1 = now_sec();
                double acc = 100.0 * correct / TEST_N;
                double speedup = (tf1 - tf0) / (ts1 - ts0);
                printf("  %4d  %5d  %5d | %7.2f%%   | %7.2f s  | %5.1fx\n",
                       top_ns[ti], n_blks[bi], n_dots[di],
                       acc, ts1 - ts0, speedup);
            }
        }
    }

    printf("\n=== SUMMARY ===\n");
    printf("  Bayesian (fast):   %.2f%% (%.2f sec)\n", 100.0 * bayes_correct / TEST_N, tb1 - tb0);
    printf("  Full cascade:      %.2f%% (%.2f sec)\n", 100.0 * full_correct / TEST_N, tf1 - tf0);
    printf("  Perihelion:        see table above\n");
    printf("\nTotal runtime: %.2f seconds.\n", now_sec() - t0);

    for (int c = 0; c < N_CLASSES; c++) free(class_idx[c]);
    free(idx_pool);
    return 0;
}
