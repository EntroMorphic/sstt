/*
 * sstt_cifar10_flat.c — CIFAR-10 with Flattened RGB
 *
 * Instead of converting to grayscale, interleave RGB per pixel:
 *   Row k: R[k,0] G[k,0] B[k,0] R[k,1] G[k,1] B[k,1] ... R[k,31] G[k,31] B[k,31]
 *
 * This creates a 32x96 image where:
 *   - Each 3x1 block captures (R_trit, G_trit, B_trit) = color encoding
 *   - H-gradients capture R->G, G->B, B->R(next pixel) = color transitions
 *   - V-gradients capture same-channel vertical edges (R->R, G->G, B->B)
 *   - The existing pipeline works unmodified on this representation
 *
 * Build: make sstt_cifar10_flat
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N         50000
#define TEST_N          10000
#define IMG_W           96      /* 32 pixels * 3 channels */
#define IMG_H           32
#define PIXELS          3072    /* 96 * 32 */
#define PADDED          3072    /* 3072 % 32 == 0 */
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    32      /* 96 / 3 = 32 blocks per row */
#define N_BLOCKS        1024    /* 32 * 32 */
#define SIG_PAD         1024    /* 1024 % 32 == 0 */
#define BYTE_VALS       256

#define BG_PIXEL        0
#define BG_GRAD         13
#define BG_TRANS        13

#define H_TRANS_PER_ROW 31      /* BLKS_PER_ROW - 1 */
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)  /* 992 */
#define V_TRANS_PER_COL 31      /* IMG_H - 1 */
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)  /* 992 */
#define TRANS_PAD       992     /* 992 % 32 == 0 */

#define IG_SCALE        16
#define MAX_K           1000

static const char *data_dir = "data-cifar10/";
static const char *class_names[] = {
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
};

static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train_img, *raw_test_img;
static uint8_t *train_labels, *test_labels;
static int8_t  *tern_train, *tern_test;
static int8_t  *hgrad_train, *hgrad_test;
static int8_t  *vgrad_train, *vgrad_test;

static uint8_t *px_tr, *px_te, *hg_tr, *hg_te, *vg_tr, *vg_te;
static uint8_t *ht_tr, *ht_te, *vt_tr, *vt_te;
static uint8_t *joint_tr, *joint_te;
static uint32_t *joint_hot;
static uint16_t ig_weights[N_BLOCKS];
static uint8_t  nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;

/* ================================================================
 *  CIFAR-10 loader: interleave RGB per pixel -> 32x96 image
 * ================================================================ */

static void load_cifar_batch(const char *path, uint8_t *imgs, uint8_t *labels, int offset) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", path); exit(1); }
    uint8_t rec[3073];
    for (int i = 0; i < 10000; i++) {
        if (fread(rec, 1, 3073, f) != 3073) { fclose(f); exit(1); }
        labels[offset + i] = rec[0];
        uint8_t *dst = imgs + (size_t)(offset + i) * PIXELS;
        const uint8_t *r = rec + 1;
        const uint8_t *g = rec + 1 + 1024;
        const uint8_t *b = rec + 1 + 2048;
        /* Interleave: for each row, R[y,x] G[y,x] B[y,x] */
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) {
                int src_idx = y * 32 + x;
                int dst_idx = y * 96 + x * 3;
                dst[dst_idx]     = r[src_idx];
                dst[dst_idx + 1] = g[src_idx];
                dst[dst_idx + 2] = b[src_idx];
            }
    }
    fclose(f);
}

static void load_data(void) {
    raw_train_img = malloc((size_t)TRAIN_N * PIXELS);
    raw_test_img  = malloc((size_t)TEST_N * PIXELS);
    train_labels  = malloc(TRAIN_N);
    test_labels   = malloc(TEST_N);
    char path[512];
    for (int batch = 1; batch <= 5; batch++) {
        snprintf(path, sizeof(path), "%sdata_batch_%d.bin", data_dir, batch);
        load_cifar_batch(path, raw_train_img, train_labels, (batch-1)*10000);
    }
    snprintf(path, sizeof(path), "%stest_batch.bin", data_dir);
    load_cifar_batch(path, raw_test_img, test_labels, 0);
    printf("  Loaded 50K train + 10K test (32x96 interleaved RGB)\n");
}

/* ================================================================
 *  Feature computation (identical logic, new dimensions)
 * ================================================================ */

static inline int8_t clamp_trit(int v){return v>0?1:v<0?-1:0;}

static void quantize(const uint8_t *src, int8_t *dst, int n) {
    const __m256i bias=_mm256_set1_epi8((char)0x80);
    const __m256i thi=_mm256_set1_epi8((char)(170^0x80));
    const __m256i tlo=_mm256_set1_epi8((char)(85^0x80));
    const __m256i one=_mm256_set1_epi8(1);
    for(int img=0;img<n;img++){
        const uint8_t *s=src+(size_t)img*PIXELS;
        int8_t *d=dst+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i+=32){
            __m256i px=_mm256_loadu_si256((const __m256i*)(s+i));
            __m256i sp=_mm256_xor_si256(px,bias);
            _mm256_storeu_si256((__m256i*)(d+i),
                _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),
                                _mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));
        }
    }
}

static void compute_gradients(const int8_t *t, int8_t *h, int8_t *v, int n) {
    for(int img=0;img<n;img++){
        const int8_t *ti=t+(size_t)img*PADDED;
        int8_t *hi=h+(size_t)img*PADDED, *vi=v+(size_t)img*PADDED;
        for(int y=0;y<IMG_H;y++){
            for(int x=0;x<IMG_W-1;x++) hi[y*IMG_W+x]=clamp_trit(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);
            hi[y*IMG_W+IMG_W-1]=0;
        }
        for(int y=0;y<IMG_H-1;y++)
            for(int x=0;x<IMG_W;x++) vi[y*IMG_W+x]=clamp_trit(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);
        memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);
    }
}

static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void block_sigs(const int8_t *data, uint8_t *sigs, int n) {
    for(int i=0;i<n;i++){
        const int8_t *img=data+(size_t)i*PADDED;
        uint8_t *sig=sigs+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int s=0;s<BLKS_PER_ROW;s++){
                int base=y*IMG_W+s*3;
                sig[y*BLKS_PER_ROW+s]=benc(img[base],img[base+1],img[base+2]);
            }
    }
}

static inline uint8_t tenc(uint8_t a,uint8_t b){
    int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1;
    int8_t b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;
    return benc(clamp_trit(b0-a0),clamp_trit(b1-a1),clamp_trit(b2-a2));
}
static void transitions(const uint8_t *bs, uint8_t *ht, uint8_t *vt, int n) {
    for(int i=0;i<n;i++){
        const uint8_t *s=bs+(size_t)i*SIG_PAD;
        uint8_t *h=ht+(size_t)i*TRANS_PAD, *v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int ss=0;ss<H_TRANS_PER_ROW;ss++)
                h[y*H_TRANS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)
            for(int ss=0;ss<BLKS_PER_ROW;ss++)
                v[y*BLKS_PER_ROW+ss]=tenc(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);
    }
}

/* Encoding D */
static uint8_t encode_d(uint8_t px_bv, uint8_t hg_bv, uint8_t vg_bv,
                         uint8_t ht_bv, uint8_t vt_bv) {
    int ps=((px_bv/9)-1)+(((px_bv/3)%3)-1)+((px_bv%3)-1);
    int hs=((hg_bv/9)-1)+(((hg_bv/3)%3)-1)+((hg_bv%3)-1);
    int vs=((vg_bv/9)-1)+(((vg_bv/3)%3)-1)+((vg_bv%3)-1);
    uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3;
    uint8_t hc=hs<0?0:hs==0?1:hs<3?2:3;
    uint8_t vc=vs<0?0:vs==0?1:vs<3?2:3;
    return pc|(hc<<2)|(vc<<4)|((ht_bv!=BG_TRANS)?1<<6:0)|((vt_bv!=BG_TRANS)?1<<7:0);
}
static void joint_sigs_fn(uint8_t *out, int n,
                           const uint8_t *px, const uint8_t *hg, const uint8_t *vg,
                           const uint8_t *ht, const uint8_t *vt) {
    for(int i=0;i<n;i++){
        const uint8_t *pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t *hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;
        uint8_t *oi=out+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int s=0;s<BLKS_PER_ROW;s++){
                int k=y*BLKS_PER_ROW+s;
                uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;
                uint8_t vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
                oi[k]=encode_d(pi[k],hi[k],vi[k],htb,vtb);
            }
    }
}

/* ================================================================
 *  Hot map, IG, index, dot, top-K, kNN
 * ================================================================ */

static void build_hot_map(void) {
    joint_hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(joint_hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){
        int lbl=train_labels[i];
        const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)
            joint_hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+lbl]++;
    }
}

static int classify_bayesian(const uint8_t *sig, uint8_t bg) {
    double acc[N_CLASSES];for(int c=0;c<N_CLASSES;c++)acc[c]=1.0;
    for(int k=0;k<N_BLOCKS;k++){
        uint8_t bv=sig[k];if(bv==bg)continue;
        const uint32_t *h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        double mx=0;
        for(int c=0;c<N_CLASSES;c++){acc[c]*=(h[c]+0.5);if(acc[c]>mx)mx=acc[c];}
        if(mx>1e10)for(int c=0;c<N_CLASSES;c++)acc[c]/=mx;
    }
    int best=0;for(int c=1;c<N_CLASSES;c++)if(acc[c]>acc[best])best=c;
    return best;
}

static void compute_ig(uint8_t bg) {
    int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
    double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
    double raw[N_BLOCKS],mx=0;
    for(int k=0;k<N_BLOCKS;k++){
        double hcond=0;
        for(int v=0;v<BYTE_VALS;v++){
            if((uint8_t)v==bg)continue;
            const uint32_t *h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
            int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
            double pv=(double)vt/TRAIN_N,hv=0;
            for(int c=0;c<N_CLASSES;c++){double pc=(double)h[c]/vt;if(pc>0)hv-=pc*log2(pc);}
            hcond+=pv*hv;
        }
        raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];
    }
    for(int k=0;k<N_BLOCKS;k++){
        ig_weights[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;
        if(!ig_weights[k])ig_weights[k]=1;
    }
}

static void build_index(uint8_t bg) {
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){
        const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg)idx_sz[k][sig[k]]++;
    }
    uint32_t tot=0;
    for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}
    idx_pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*sizeof(uint32_t));
    for(int i=0;i<TRAIN_N;i++){
        const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg)idx_pool[wp[k][sig[k]]++]=(uint32_t)i;
    }
    free(wp);
    printf("  Index: %u entries (%.1f MB)\n",tot,(double)tot*4/(1024*1024));
}

static inline int32_t tdot(const int8_t *a, const int8_t *b) {
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32)
        acc=_mm256_add_epi8(acc,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),
            _mm256_load_si256((const __m256i*)(b+i))));
    /* Need wider accumulation for 3072 elements (>127 possible) */
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);
    return _mm_cvtsi128_si32(s);
}

/* Safe dot for longer vectors: accumulate in int16 every 64 elements */
static inline int32_t tdot_safe(const int8_t *a, const int8_t *b) {
    int32_t total = 0;
    for (int chunk = 0; chunk < PADDED; chunk += 64) {
        __m256i acc = _mm256_setzero_si256();
        int end = chunk + 64; if (end > PADDED) end = PADDED;
        for (int i = chunk; i < end; i += 32)
            acc = _mm256_add_epi8(acc, _mm256_sign_epi8(
                _mm256_load_si256((const __m256i*)(a+i)),
                _mm256_load_si256((const __m256i*)(b+i))));
        __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
        __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc, 1));
        __m256i s32 = _mm256_madd_epi16(_mm256_add_epi16(lo, hi), _mm256_set1_epi16(1));
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(s32), _mm256_extracti128_si256(s32, 1));
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        total += _mm_cvtsi128_si32(s);
    }
    return total;
}

typedef struct{uint32_t id;uint32_t votes;int32_t dot;}cand_t;
static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_dot_d(const void*a,const void*b){int32_t da=((const cand_t*)a)->dot,db=((const cand_t*)b)->dot;return(db>da)-(db<da);}

static int select_topk(const uint32_t *votes, int n, cand_t *out, int k) {
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];if(!mx)return 0;
    int *hist=calloc((size_t)(mx+1),sizeof(int));
    for(int i=0;i<n;i++)if(votes[i])hist[votes[i]]++;
    int cum=0,thr;for(thr=(int)mx;thr>=1;thr--){cum+=hist[thr];if(cum>=k)break;}
    if(thr<1)thr=1;free(hist);
    int nc=0;for(int i=0;i<n&&nc<k;i++)
        if(votes[i]>=(uint32_t)thr){out[nc].id=(uint32_t)i;out[nc].votes=votes[i];out[nc].dot=0;nc++;}
    qsort(out,(size_t)nc,sizeof(cand_t),cmp_votes_d);
    return nc;
}
static int knn_vote(const cand_t *c,int nc,int k){
    int v[N_CLASSES]={0};if(k>nc)k=nc;
    for(int i=0;i<k;i++)v[train_labels[c[i].id]]++;
    int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;return best;
}

/* ================================================================ */

static void error_analysis(const uint8_t *preds) {
    int conf[N_CLASSES][N_CLASSES];memset(conf,0,sizeof(conf));
    for(int i=0;i<TEST_N;i++)conf[test_labels[i]][preds[i]]++;
    printf("\n  Per-class recall:\n");
    for(int r=0;r<N_CLASSES;r++){
        int rt=0;for(int c=0;c<N_CLASSES;c++)rt+=conf[r][c];
        printf("    %d %-10s %5.1f%%\n",r,class_names[r],rt>0?100.0*conf[r][r]/rt:0.0);
    }
    typedef struct{int a,b,count;}pair_t;
    pair_t pairs[45];int np=0;
    for(int a=0;a<N_CLASSES;a++)for(int b=a+1;b<N_CLASSES;b++)
        pairs[np++]=(pair_t){a,b,conf[a][b]+conf[b][a]};
    for(int i=0;i<np-1;i++)for(int j=i+1;j<np;j++)
        if(pairs[j].count>pairs[i].count){pair_t t=pairs[i];pairs[i]=pairs[j];pairs[j]=t;}
    printf("  Top-5 confused:\n");
    for(int i=0;i<5&&i<np;i++)
        printf("    %s<->%s: %d\n",class_names[pairs[i].a],class_names[pairs[i].b],pairs[i].count);
}

int main(int argc, char **argv) {
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b=malloc(l+2);memcpy(b,data_dir,l);b[l]='/';b[l+1]=0;data_dir=b;}}

    printf("=== SSTT CIFAR-10 Flattened RGB (32x96) ===\n\n");
    printf("  Each 3x1 block = (R_trit, G_trit, B_trit) = color encoding\n");
    printf("  H-gradients = R->G, G->B, B->R transitions\n");
    printf("  1024 blocks (4x MNIST), 3072 pixels for dot products\n\n");

    printf("Loading...\n");
    load_data();

    printf("\nComputing features...\n");
    tern_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vgrad_train=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vgrad_test=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quantize(raw_train_img,tern_train,TRAIN_N);quantize(raw_test_img,tern_test,TEST_N);
    compute_gradients(tern_train,hgrad_train,vgrad_train,TRAIN_N);
    compute_gradients(tern_test,hgrad_test,vgrad_test,TEST_N);

    /* Trit distribution */
    {
        long tn[3]={0};
        for(int i=0;i<TRAIN_N;i++){const int8_t *t=tern_train+(size_t)i*PADDED;
            for(int p=0;p<PIXELS;p++)tn[t[p]+1]++;}
        long total=(long)TRAIN_N*PIXELS;
        printf("  Trits: -1=%.1f%% 0=%.1f%% +1=%.1f%%\n",
               100.0*tn[0]/total,100.0*tn[1]/total,100.0*tn[2]/total);
    }

    px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);hg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);vg_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(tern_train,px_tr,TRAIN_N);block_sigs(tern_test,px_te,TEST_N);
    block_sigs(hgrad_train,hg_tr,TRAIN_N);block_sigs(hgrad_test,hg_te,TEST_N);
    block_sigs(vgrad_train,vg_tr,TRAIN_N);block_sigs(vgrad_test,vg_te,TEST_N);

    ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    transitions(px_tr,ht_tr,vt_tr,TRAIN_N);transitions(px_te,ht_te,vt_te,TEST_N);

    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs_fn(joint_tr,TRAIN_N,px_tr,hg_tr,vg_tr,ht_tr,vt_tr);
    joint_sigs_fn(joint_te,TEST_N,px_te,hg_te,vg_te,ht_te,vt_te);

    /* Background */
    long val_counts[BYTE_VALS]={0};
    for(int i=0;i<TRAIN_N;i++){const uint8_t *sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)val_counts[sig[k]]++;}
    uint8_t bg=0;long mc2=0;
    for(int v=0;v<BYTE_VALS;v++)if(val_counts[v]>mc2){mc2=val_counts[v];bg=(uint8_t)v;}
    printf("  BG=%d (%.1f%%)\n",bg,100.0*mc2/((long)TRAIN_N*N_BLOCKS));

    int n_used=0;for(int v=0;v<BYTE_VALS;v++)if(val_counts[v]>0)n_used++;
    printf("  Unique values: %d/256\n",n_used);

    build_hot_map();compute_ig(bg);

    int ig_min=IG_SCALE,ig_max=0;double ig_sum=0;
    for(int k=0;k<N_BLOCKS;k++){
        if(ig_weights[k]<ig_min)ig_min=ig_weights[k];
        if(ig_weights[k]>ig_max)ig_max=ig_weights[k];
        ig_sum+=ig_weights[k];
    }
    printf("  IG: min=%d max=%d avg=%.1f\n",ig_min,ig_max,ig_sum/N_BLOCKS);

    build_index(bg);
    for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* Test 1: Bayesian */
    printf("--- Bayesian Hot Map ---\n");
    {int correct=0;
     for(int i=0;i<TEST_N;i++)
         if(classify_bayesian(joint_te+(size_t)i*SIG_PAD,bg)==test_labels[i])correct++;
     printf("  Bayesian: %.2f%% (%d/%d)\n\n",100.0*correct/TEST_N,correct,TEST_N);
    }

    /* Test 2: Cascade — gradient dot only (learned from grad ablation) */
    printf("--- Cascade (gradient dot) ---\n");
    int k_vals[]={50,200,500,1000};int knn_vals[]={1,3,5,7};
    int nk=4,nknn=4;
    int correct[4][4];memset(correct,0,sizeof(correct));
    int vote_correct=0;

    uint32_t *votes=calloc(TRAIN_N,sizeof(uint32_t));
    cand_t *cands=malloc(MAX_K*sizeof(cand_t));
    uint8_t *best_preds=calloc(TEST_N,1);
    double best_acc=0;int best_ki=0,best_knni=0;

    double tc=now_sec();
    for(int i=0;i<TEST_N;i++){
        memset(votes,0,TRAIN_N*sizeof(uint32_t));
        const uint8_t *sig=joint_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=sig[k];if(bv==bg)continue;
            uint16_t w=ig_weights[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];
             for(uint16_t j=0;j<sz;j++)votes[idx_pool[off+j]]+=w;}
            for(int nb=0;nb<8;nb++){
                uint8_t nv=nbr[bv][nb];if(nv==bg)continue;
                uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];
                for(uint16_t j=0;j<nsz;j++)votes[idx_pool[noff+j]]+=wh;
            }
        }
        {uint32_t bid=0,bv2=0;for(int j=0;j<TRAIN_N;j++)
            if(votes[j]>bv2){bv2=votes[j];bid=(uint32_t)j;}
         if(train_labels[bid]==test_labels[i])vote_correct++;}

        int nc=select_topk(votes,TRAIN_N,cands,k_vals[nk-1]);
        /* Dot on h-grad + v-grad (not pixel — learned from ablation) */
        const int8_t *qh=hgrad_test+(size_t)i*PADDED;
        const int8_t *qv=vgrad_test+(size_t)i*PADDED;
        for(int j=0;j<nc;j++){
            uint32_t id=cands[j].id;
            cands[j].dot=tdot_safe(qh,hgrad_train+(size_t)id*PADDED)
                        +tdot_safe(qv,vgrad_train+(size_t)id*PADDED);
        }
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_dot_d);

        for(int ki=0;ki<nk;ki++){
            int eff=nc<k_vals[ki]?nc:k_vals[ki];
            for(int knni=0;knni<nknn;knni++){
                int pred=knn_vote(cands,eff,knn_vals[knni]);
                if(pred==test_labels[i])correct[ki][knni]++;
            }
        }
        if((i+1)%1000==0)fprintf(stderr,"  %d/%d (vote %.1f%%)\r",i+1,TEST_N,100.0*vote_correct/(i+1));
    }
    fprintf(stderr,"\n");

    printf("  Vote-only: %.2f%%\n",100.0*vote_correct/TEST_N);
    printf("  %-6s","K\\kNN");for(int ki2=0;ki2<nknn;ki2++)printf("  k=%-3d",knn_vals[ki2]);printf("\n");
    for(int ki=0;ki<nk;ki++){
        printf("  %-6d",k_vals[ki]);
        for(int knni=0;knni<nknn;knni++){
            double acc=100.0*correct[ki][knni]/TEST_N;printf("  %5.2f%%",acc);
            if(acc>best_acc){best_acc=acc;best_ki=ki;best_knni=knni;}
        }
        printf("\n");
    }
    printf("  Best: K=%d k=%d -> %.2f%% (%.1fs)\n",
           k_vals[best_ki],knn_vals[best_knni],best_acc,now_sec()-tc);

    /* Re-run best for error analysis */
    for(int i=0;i<TEST_N;i++){
        memset(votes,0,TRAIN_N*sizeof(uint32_t));
        const uint8_t *sig=joint_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){
            uint8_t bv=sig[k];if(bv==bg)continue;
            uint16_t w=ig_weights[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];
             for(uint16_t j=0;j<sz;j++)votes[idx_pool[off+j]]+=w;}
            for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg)continue;
                uint32_t noff=idx_off[k][nv];uint16_t nsz=idx_sz[k][nv];
                for(uint16_t j=0;j<nsz;j++)votes[idx_pool[noff+j]]+=wh;}
        }
        int nc=select_topk(votes,TRAIN_N,cands,k_vals[best_ki]);
        const int8_t *qh=hgrad_test+(size_t)i*PADDED,*qv=vgrad_test+(size_t)i*PADDED;
        for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
            cands[j].dot=tdot_safe(qh,hgrad_train+(size_t)id*PADDED)
                        +tdot_safe(qv,vgrad_train+(size_t)id*PADDED);}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_dot_d);
        best_preds[i]=(uint8_t)knn_vote(cands,nc,knn_vals[best_knni]);
    }
    error_analysis(best_preds);

    printf("\n=== SUMMARY ===\n");
    printf("  Bayesian:  see above\n");
    printf("  Cascade:   %.2f%% (K=%d k=%d)\n",best_acc,k_vals[best_ki],knn_vals[best_knni]);
    printf("  Grayscale baseline (sstt_cifar10): Bayesian 33.76%%, cascade 27.06%%\n");
    printf("\nTotal: %.1f sec\n",now_sec()-t0);

    free(votes);free(cands);free(best_preds);
    return 0;
}
