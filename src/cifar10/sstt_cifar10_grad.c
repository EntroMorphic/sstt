/*
 * sstt_cifar10_grad.c — CIFAR-10 Gradient Channel Ablation
 *
 * Tests whether the pixel channel is the problem on CIFAR-10.
 *
 * Four experiments:
 *   A: Grayscale full Encoding D (baseline from sstt_cifar10.c)
 *   B: Gradient-only encoding (pixel channel zeroed in Encoding D)
 *   C: Per-RGB-channel gradients (R,G,B each get h-grad + v-grad blocks)
 *   D: Dot product on gradient channels only (pixel dot = 0)
 *
 * Build: make sstt_cifar10_grad
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
#define IMG_W           32
#define IMG_H           32
#define PIXELS          1024
#define PADDED          1024
#define N_CLASSES       10
#define CLS_PAD         16

#define BLKS_PER_ROW    10
#define N_BLOCKS        320
#define SIG_PAD         320
#define BYTE_VALS       256

#define BG_GRAD         13
#define BG_TRANS        13

#define H_TRANS_PER_ROW 9
#define N_HTRANS        (H_TRANS_PER_ROW * IMG_H)
#define V_TRANS_PER_COL 31
#define N_VTRANS        (BLKS_PER_ROW * V_TRANS_PER_COL)
#define TRANS_PAD       320

#define IG_SCALE        16
#define MAX_K           1000

static const char *data_dir = "data-cifar10/";
static const char *class_names[] = {
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
};

static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *train_labels, *test_labels;

/* Per-channel raw images (R, G, B separately) */
static uint8_t *raw_r_tr, *raw_r_te, *raw_g_tr, *raw_g_te, *raw_b_tr, *raw_b_te;
/* Grayscale */
static uint8_t *raw_gray_tr, *raw_gray_te;

/* Ternary + gradients for grayscale */
static int8_t *tern_tr, *tern_te;
static int8_t *hg_tr, *hg_te, *vg_tr, *vg_te;

/* Ternary + gradients for R, G, B channels */
static int8_t *tern_r_tr, *tern_r_te, *tern_g_tr, *tern_g_te, *tern_b_tr, *tern_b_te;
static int8_t *hg_r_tr, *hg_r_te, *vg_r_tr, *vg_r_te;
static int8_t *hg_g_tr, *hg_g_te, *vg_g_tr, *vg_g_te;
static int8_t *hg_b_tr, *hg_b_te, *vg_b_tr, *vg_b_te;

/* ================================================================ */

static void load_cifar_batch(const char *path, int offset) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", path); exit(1); }
    uint8_t rec[3073];
    for (int i = 0; i < 10000; i++) {
        if (fread(rec, 1, 3073, f) != 3073) { fclose(f); exit(1); }
        int idx = offset + i;
        train_labels[idx] = rec[0];
        memcpy(raw_r_tr + (size_t)idx * PIXELS, rec + 1, 1024);
        memcpy(raw_g_tr + (size_t)idx * PIXELS, rec + 1 + 1024, 1024);
        memcpy(raw_b_tr + (size_t)idx * PIXELS, rec + 1 + 2048, 1024);
        uint8_t *dst = raw_gray_tr + (size_t)idx * PIXELS;
        for (int p = 0; p < 1024; p++)
            dst[p] = (uint8_t)((77*(int)rec[1+p] + 150*(int)rec[1+1024+p] + 29*(int)rec[1+2048+p]) >> 8);
    }
    fclose(f);
}

static void load_cifar_test(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", path); exit(1); }
    uint8_t rec[3073];
    for (int i = 0; i < 10000; i++) {
        if (fread(rec, 1, 3073, f) != 3073) { fclose(f); exit(1); }
        test_labels[i] = rec[0];
        memcpy(raw_r_te + (size_t)i * PIXELS, rec + 1, 1024);
        memcpy(raw_g_te + (size_t)i * PIXELS, rec + 1 + 1024, 1024);
        memcpy(raw_b_te + (size_t)i * PIXELS, rec + 1 + 2048, 1024);
        uint8_t *dst = raw_gray_te + (size_t)i * PIXELS;
        for (int p = 0; p < 1024; p++)
            dst[p] = (uint8_t)((77*(int)rec[1+p] + 150*(int)rec[1+1024+p] + 29*(int)rec[1+2048+p]) >> 8);
    }
    fclose(f);
}

static void load_data(void) {
    raw_r_tr = malloc((size_t)TRAIN_N*PIXELS); raw_r_te = malloc((size_t)TEST_N*PIXELS);
    raw_g_tr = malloc((size_t)TRAIN_N*PIXELS); raw_g_te = malloc((size_t)TEST_N*PIXELS);
    raw_b_tr = malloc((size_t)TRAIN_N*PIXELS); raw_b_te = malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr = malloc((size_t)TRAIN_N*PIXELS); raw_gray_te = malloc((size_t)TEST_N*PIXELS);
    train_labels = malloc(TRAIN_N); test_labels = malloc(TEST_N);

    char path[512];
    for (int b = 1; b <= 5; b++) {
        snprintf(path, sizeof(path), "%sdata_batch_%d.bin", data_dir, b);
        load_cifar_batch(path, (b-1)*10000);
    }
    snprintf(path, sizeof(path), "%stest_batch.bin", data_dir);
    load_cifar_test(path);
    printf("  Loaded 50K train + 10K test\n");
}

/* ================================================================
 *  Feature computation
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

static void gradients(const int8_t *t, int8_t *h, int8_t *v, int n) {
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

/* Encoding D with configurable pixel inclusion */
static void joint_sigs(uint8_t *out, int n,
                        const uint8_t *px, const uint8_t *hg, const uint8_t *vg,
                        const uint8_t *ht, const uint8_t *vt,
                        int include_pixel) {
    for(int i=0;i<n;i++){
        const uint8_t *pi=px+(size_t)i*SIG_PAD, *hi=hg+(size_t)i*SIG_PAD, *vi=vg+(size_t)i*SIG_PAD;
        const uint8_t *hti=ht+(size_t)i*TRANS_PAD, *vti=vt+(size_t)i*TRANS_PAD;
        uint8_t *oi=out+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)
            for(int s=0;s<BLKS_PER_ROW;s++){
                int k=y*BLKS_PER_ROW+s;
                uint8_t ht_bv=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS;
                uint8_t vt_bv=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;

                /* Pixel majority trit */
                uint8_t pc = 1; /* neutral default */
                if (include_pixel) {
                    int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);
                    pc=ps<0?0:ps==0?1:ps<3?2:3;
                }

                /* Gradient majority trits */
                int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
                int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
                uint8_t hc=hs<0?0:hs==0?1:hs<3?2:3;
                uint8_t vc=vs<0?0:vs==0?1:vs<3?2:3;
                uint8_t hta=(ht_bv!=BG_TRANS)?1:0;
                uint8_t vta=(vt_bv!=BG_TRANS)?1:0;
                oi[k]=pc|(hc<<2)|(vc<<4)|(hta<<6)|(vta<<7);
            }
    }
}

/* ================================================================
 *  Encoding for per-RGB-channel gradients
 *
 *  6 gradient block signatures (Rh, Rv, Gh, Gv, Bh, Bv) packed into
 *  a wider signature. We use a different approach: concatenate the 6
 *  gradient block signatures into 6 separate inverted indices and vote
 *  across all of them. Each has N_BLOCKS positions, 27 values.
 * ================================================================ */

#define N_GRAD_CHANNELS 6
#define RGB_N_BLOCKS    (N_BLOCKS * N_GRAD_CHANNELS)  /* 1920 */
#define RGB_SIG_PAD     1920  /* 1920 % 32 == 0 */
#define RGB_N_BVALS     27

static void rgb_grad_sigs(uint8_t *out, int n,
                           const uint8_t *rh, const uint8_t *rv,
                           const uint8_t *gh, const uint8_t *gv,
                           const uint8_t *bh, const uint8_t *bv) {
    const uint8_t *channels[] = {rh, rv, gh, gv, bh, bv};
    for (int i = 0; i < n; i++) {
        uint8_t *oi = out + (size_t)i * RGB_SIG_PAD;
        for (int ch = 0; ch < N_GRAD_CHANNELS; ch++) {
            const uint8_t *src = channels[ch] + (size_t)i * SIG_PAD;
            memcpy(oi + ch * N_BLOCKS, src, N_BLOCKS);
        }
    }
}

/* ================================================================
 *  Generic Bayesian + Cascade runner
 *  Works on any signature with configurable n_blocks, n_vals, sig_pad
 * ================================================================ */

static inline int32_t tdot(const int8_t *a, const int8_t *b) {
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PADDED;i+=32)
        acc=_mm256_add_epi8(acc,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),
            _mm256_load_si256((const __m256i*)(b+i))));
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);
    return _mm_cvtsi128_si32(s);
}

typedef struct{uint32_t id;uint32_t votes;int32_t dot;}cand_t;
static int cmp_votes_d(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmp_dot_d(const void*a,const void*b){int32_t da=((const cand_t*)a)->dot,db=((const cand_t*)b)->dot;return(db>da)-(db<da);}

static int select_topk(const uint32_t *votes, int n, cand_t *out, int k) {
    uint32_t mx=0;for(int i=0;i<n;i++)if(votes[i]>mx)mx=votes[i];
    if(!mx)return 0;
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
    int best=0;for(int c2=1;c2<N_CLASSES;c2++)if(v[c2]>v[best])best=c2;
    return best;
}

/* Run a full experiment: Bayesian + cascade with given signatures and dot channels */
static void run_experiment(const char *name,
                            const uint8_t *sigs_tr, const uint8_t *sigs_te,
                            int n_blocks, int n_vals, int sig_pad,
                            /* For dot product: which ternary arrays to dot on */
                            const int8_t **dot_tr, const int8_t **dot_te,
                            const int *dot_weights, int n_dot_channels) {
    printf("\n=== %s ===\n", name);

    /* Auto-detect background */
    long *val_counts = calloc(n_vals, sizeof(long));
    for(int i=0;i<TRAIN_N;i++){
        const uint8_t *sig=sigs_tr+(size_t)i*sig_pad;
        for(int k=0;k<n_blocks;k++) val_counts[sig[k]]++;
    }
    uint8_t bg=0;long mc=0;
    for(int v=0;v<n_vals;v++)if(val_counts[v]>mc){mc=val_counts[v];bg=(uint8_t)v;}
    printf("  BG=%d (%.1f%%)\n",bg,100.0*mc/((long)TRAIN_N*n_blocks));
    free(val_counts);

    /* Hot map */
    uint32_t *hot=aligned_alloc(32,(size_t)n_blocks*n_vals*CLS_PAD*4);
    memset(hot,0,(size_t)n_blocks*n_vals*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){
        int lbl=train_labels[i];
        const uint8_t *sig=sigs_tr+(size_t)i*sig_pad;
        for(int k=0;k<n_blocks;k++)
            hot[(size_t)k*n_vals*CLS_PAD+(size_t)sig[k]*CLS_PAD+lbl]++;
    }

    /* Bayesian */
    int bayesian_correct=0;
    for(int i=0;i<TEST_N;i++){
        const uint8_t *sig=sigs_te+(size_t)i*sig_pad;
        double acc[N_CLASSES];for(int c=0;c<N_CLASSES;c++)acc[c]=1.0;
        for(int k=0;k<n_blocks;k++){
            uint8_t bv=sig[k];if(bv==bg)continue;
            const uint32_t *h=hot+(size_t)k*n_vals*CLS_PAD+(size_t)bv*CLS_PAD;
            double mx2=0;
            for(int c=0;c<N_CLASSES;c++){acc[c]*=(h[c]+0.5);if(acc[c]>mx2)mx2=acc[c];}
            if(mx2>1e10)for(int c=0;c<N_CLASSES;c++)acc[c]/=mx2;
        }
        int pred=0;for(int c=1;c<N_CLASSES;c++)if(acc[c]>acc[pred])pred=c;
        if(pred==test_labels[i])bayesian_correct++;
    }
    printf("  Bayesian: %.2f%% (%d/%d)\n",100.0*bayesian_correct/TEST_N,bayesian_correct,TEST_N);

    /* IG weights */
    uint16_t *ig=malloc(n_blocks*2);
    {
        int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
        double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
        double *raw_ig=malloc(n_blocks*sizeof(double));double mig=0;
        for(int k=0;k<n_blocks;k++){
            double hcond=0;
            for(int v=0;v<n_vals;v++){
                if((uint8_t)v==bg)continue;
                const uint32_t *h=hot+(size_t)k*n_vals*CLS_PAD+(size_t)v*CLS_PAD;
                int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
                double pv=(double)vt/TRAIN_N,hv=0;
                for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}
                hcond+=pv*hv;
            }
            raw_ig[k]=hc-hcond;if(raw_ig[k]>mig)mig=raw_ig[k];
        }
        for(int k=0;k<n_blocks;k++){
            ig[k]=mig>0?(uint16_t)(raw_ig[k]/mig*IG_SCALE+0.5):1;
            if(!ig[k])ig[k]=1;
        }
        free(raw_ig);
    }

    /* Inverted index */
    uint16_t *isz=calloc((size_t)n_blocks*n_vals,sizeof(uint16_t));
    uint32_t *ioff=malloc((size_t)n_blocks*n_vals*sizeof(uint32_t));
    for(int i=0;i<TRAIN_N;i++){
        const uint8_t *sig=sigs_tr+(size_t)i*sig_pad;
        for(int k=0;k<n_blocks;k++)if(sig[k]!=bg)isz[k*n_vals+sig[k]]++;
    }
    uint32_t tot=0;
    for(int k=0;k<n_blocks;k++)for(int v=0;v<n_vals;v++){ioff[k*n_vals+v]=tot;tot+=isz[k*n_vals+v];}
    uint32_t *pool=malloc((size_t)tot*sizeof(uint32_t));
    uint32_t *wp=malloc((size_t)n_blocks*n_vals*sizeof(uint32_t));
    memcpy(wp,ioff,(size_t)n_blocks*n_vals*sizeof(uint32_t));
    for(int i=0;i<TRAIN_N;i++){
        const uint8_t *sig=sigs_tr+(size_t)i*sig_pad;
        for(int k=0;k<n_blocks;k++)if(sig[k]!=bg)pool[wp[k*n_vals+sig[k]]++]=(uint32_t)i;
    }
    free(wp);
    printf("  Index: %u entries (%.1f MB)\n",tot,(double)tot*4/(1024*1024));

    /* Neighbor table (8 bit-flips) — only for 256-val encoding */
    uint8_t nbr[BYTE_VALS][8];
    int use_probes = (n_vals == 256);
    if (use_probes)
        for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)nbr[v][b]=(uint8_t)(v^(1<<b));

    /* Cascade: vote → top-K → dot → kNN */
    uint32_t *votes=calloc(TRAIN_N,sizeof(uint32_t));
    cand_t *cands=malloc(MAX_K*sizeof(cand_t));
    int vote_correct=0;

    /* Test multiple K and kNN values */
    int k_vals[]={50,200,500};int knn_vals[]={1,3,7};
    int nk=3,nknn=3;
    int correct[3][3];memset(correct,0,sizeof(correct));

    double t1=now_sec();
    for(int i=0;i<TEST_N;i++){
        memset(votes,0,TRAIN_N*sizeof(uint32_t));
        const uint8_t *sig=sigs_te+(size_t)i*sig_pad;

        for(int k=0;k<n_blocks;k++){
            uint8_t bv=sig[k];if(bv==bg)continue;
            uint16_t w=ig[k];
            {uint32_t off=ioff[k*n_vals+bv];uint16_t sz=isz[k*n_vals+bv];
             for(uint16_t j=0;j<sz;j++)votes[pool[off+j]]+=w;}
            if(use_probes){
                uint16_t wh=w>1?w/2:1;
                for(int nb=0;nb<8;nb++){
                    uint8_t nv=nbr[bv][nb];if(nv==bg)continue;
                    uint32_t noff=ioff[k*n_vals+nv];uint16_t nsz=isz[k*n_vals+nv];
                    for(uint16_t j=0;j<nsz;j++)votes[pool[noff+j]]+=wh;
                }
            }
        }

        /* Vote-only */
        {uint32_t bid=0,bv2=0;
         for(int j=0;j<TRAIN_N;j++)if(votes[j]>bv2){bv2=votes[j];bid=(uint32_t)j;}
         if(train_labels[bid]==test_labels[i])vote_correct++;}

        /* Top-K → dot → kNN */
        int nc=select_topk(votes,TRAIN_N,cands,k_vals[nk-1]);

        /* Multi-channel dot product */
        for(int j=0;j<nc;j++){
            int32_t d=0;
            for(int ch=0;ch<n_dot_channels;ch++)
                d+=dot_weights[ch]*tdot(dot_te[ch]+(size_t)i*PADDED,
                                         dot_tr[ch]+(size_t)cands[j].id*PADDED);
            cands[j].dot=d;
        }
        qsort(cands,(size_t)nc,sizeof(cand_t),cmp_dot_d);

        for(int ki=0;ki<nk;ki++){
            int eff=nc<k_vals[ki]?nc:k_vals[ki];
            for(int knni=0;knni<nknn;knni++){
                int pred=knn_vote(cands,eff,knn_vals[knni]);
                if(pred==test_labels[i])correct[ki][knni]++;
            }
        }
        if((i+1)%2000==0)fprintf(stderr,"  %s: %d/%d\r",name,i+1,TEST_N);
    }
    fprintf(stderr,"\n");

    printf("  Vote-only: %.2f%%\n",100.0*vote_correct/TEST_N);
    printf("  %-6s","K\\kNN");
    for(int ki2=0;ki2<nknn;ki2++)printf("  k=%-3d",knn_vals[ki2]);printf("\n");
    double best_acc=0;int best_ki=0,best_knni=0;
    for(int ki=0;ki<nk;ki++){
        printf("  %-6d",k_vals[ki]);
        for(int knni=0;knni<nknn;knni++){
            double acc=100.0*correct[ki][knni]/TEST_N;printf("  %5.2f%%",acc);
            if(acc>best_acc){best_acc=acc;best_ki=ki;best_knni=knni;}
        }
        printf("\n");
    }
    printf("  Best: K=%d k=%d -> %.2f%% (%.1fs)\n",
           k_vals[best_ki],knn_vals[best_knni],best_acc,now_sec()-t1);

    free(votes);free(cands);free(hot);free(ig);free(isz);free(ioff);free(pool);
}

/* ================================================================ */

int main(int argc, char **argv) {
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b=malloc(l+2);memcpy(b,data_dir,l);b[l]='/';b[l+1]=0;data_dir=b;}}

    printf("=== SSTT CIFAR-10 Gradient Ablation ===\n\n");

    printf("Loading...\n");
    load_data();

    printf("\nComputing features...\n");
    /* Grayscale */
    tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quantize(raw_gray_tr,tern_tr,TRAIN_N);quantize(raw_gray_te,tern_te,TEST_N);
    gradients(tern_tr,hg_tr,vg_tr,TRAIN_N);gradients(tern_te,hg_te,vg_te,TEST_N);

    /* Per-channel RGB */
    tern_r_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_r_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    tern_g_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_g_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    tern_b_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_b_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quantize(raw_r_tr,tern_r_tr,TRAIN_N);quantize(raw_r_te,tern_r_te,TEST_N);
    quantize(raw_g_tr,tern_g_tr,TRAIN_N);quantize(raw_g_te,tern_g_te,TEST_N);
    quantize(raw_b_tr,tern_b_tr,TRAIN_N);quantize(raw_b_te,tern_b_te,TEST_N);

    hg_r_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg_r_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg_r_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg_r_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg_g_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg_g_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg_g_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg_g_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg_b_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg_b_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg_b_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg_b_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    gradients(tern_r_tr,hg_r_tr,vg_r_tr,TRAIN_N);gradients(tern_r_te,hg_r_te,vg_r_te,TEST_N);
    gradients(tern_g_tr,hg_g_tr,vg_g_tr,TRAIN_N);gradients(tern_g_te,hg_g_te,vg_g_te,TEST_N);
    gradients(tern_b_tr,hg_b_tr,vg_b_tr,TRAIN_N);gradients(tern_b_te,hg_b_te,vg_b_te,TEST_N);

    /* Block signatures */
    uint8_t *px_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    uint8_t *px_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hgs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    uint8_t *hgs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vgs_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    uint8_t *vgs_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(tern_tr,px_tr,TRAIN_N);block_sigs(tern_te,px_te,TEST_N);
    block_sigs(hg_tr,hgs_tr,TRAIN_N);block_sigs(hg_te,hgs_te,TEST_N);
    block_sigs(vg_tr,vgs_tr,TRAIN_N);block_sigs(vg_te,vgs_te,TEST_N);

    /* Transitions */
    uint8_t *ht_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);
    uint8_t *ht_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vt_tr=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD);
    uint8_t *vt_te=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    transitions(px_tr,ht_tr,vt_tr,TRAIN_N);transitions(px_te,ht_te,vt_te,TEST_N);

    /* RGB gradient block sigs */
    uint8_t *rh_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);uint8_t *rh_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *rv_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);uint8_t *rv_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *gh_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);uint8_t *gh_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *gv_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);uint8_t *gv_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *bh_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);uint8_t *bh_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *bv_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);uint8_t *bv_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    block_sigs(hg_r_tr,rh_tr,TRAIN_N);block_sigs(hg_r_te,rh_te,TEST_N);
    block_sigs(vg_r_tr,rv_tr,TRAIN_N);block_sigs(vg_r_te,rv_te,TEST_N);
    block_sigs(hg_g_tr,gh_tr,TRAIN_N);block_sigs(hg_g_te,gh_te,TEST_N);
    block_sigs(vg_g_tr,gv_tr,TRAIN_N);block_sigs(vg_g_te,gv_te,TEST_N);
    block_sigs(hg_b_tr,bh_tr,TRAIN_N);block_sigs(hg_b_te,bh_te,TEST_N);
    block_sigs(vg_b_tr,bv_tr,TRAIN_N);block_sigs(vg_b_te,bv_te,TEST_N);

    /* Joint sigs: full Encoding D and gradient-only */
    uint8_t *j_full_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    uint8_t *j_full_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *j_grad_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);
    uint8_t *j_grad_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    joint_sigs(j_full_tr,TRAIN_N,px_tr,hgs_tr,vgs_tr,ht_tr,vt_tr,1);
    joint_sigs(j_full_te,TEST_N,px_te,hgs_te,vgs_te,ht_te,vt_te,1);
    joint_sigs(j_grad_tr,TRAIN_N,px_tr,hgs_tr,vgs_tr,ht_tr,vt_tr,0);
    joint_sigs(j_grad_te,TEST_N,px_te,hgs_te,vgs_te,ht_te,vt_te,0);

    /* RGB gradient concatenated sigs */
    uint8_t *rgb_tr=aligned_alloc(32,(size_t)TRAIN_N*RGB_SIG_PAD);
    uint8_t *rgb_te=aligned_alloc(32,(size_t)TEST_N*RGB_SIG_PAD);
    rgb_grad_sigs(rgb_tr,TRAIN_N,rh_tr,rv_tr,gh_tr,gv_tr,bh_tr,bv_tr);
    rgb_grad_sigs(rgb_te,TEST_N,rh_te,rv_te,gh_te,gv_te,bh_te,bv_te);

    printf("  Features computed (%.1f sec)\n", now_sec()-t0);

    /* ================================================================
     * EXPERIMENT A: Full Encoding D, dot on pixel+vgrad (baseline)
     * ================================================================ */
    {
        const int8_t *dtr[]={tern_tr, vg_tr};
        const int8_t *dte[]={tern_te, vg_te};
        int dw[]={256, 192};
        run_experiment("A: Full Encoding D (pixel+grad), dot px+vg",
                       j_full_tr, j_full_te, N_BLOCKS, BYTE_VALS, SIG_PAD,
                       dtr, dte, dw, 2);
    }

    /* ================================================================
     * EXPERIMENT B: Gradient-only Encoding D, dot on hgrad+vgrad
     * ================================================================ */
    {
        const int8_t *dtr[]={hg_tr, vg_tr};
        const int8_t *dte[]={hg_te, vg_te};
        int dw[]={256, 256};
        run_experiment("B: Grad-only Encoding D, dot hg+vg",
                       j_grad_tr, j_grad_te, N_BLOCKS, BYTE_VALS, SIG_PAD,
                       dtr, dte, dw, 2);
    }

    /* ================================================================
     * EXPERIMENT C: Full Encoding D, dot on gradients only (no pixel)
     * ================================================================ */
    {
        const int8_t *dtr[]={hg_tr, vg_tr};
        const int8_t *dte[]={hg_te, vg_te};
        int dw[]={256, 256};
        run_experiment("C: Full Encoding D, dot hg+vg only (no px dot)",
                       j_full_tr, j_full_te, N_BLOCKS, BYTE_VALS, SIG_PAD,
                       dtr, dte, dw, 2);
    }

    /* ================================================================
     * EXPERIMENT D: RGB 6-channel gradients (27-val blocks, no probes)
     *              dot on all 6 RGB gradient channels
     * ================================================================ */
    {
        const int8_t *dtr[]={hg_r_tr, vg_r_tr, hg_g_tr, vg_g_tr, hg_b_tr, vg_b_tr};
        const int8_t *dte[]={hg_r_te, vg_r_te, hg_g_te, vg_g_te, hg_b_te, vg_b_te};
        int dw[]={256, 256, 256, 256, 256, 256};
        run_experiment("D: RGB 6-channel gradients (Rh,Rv,Gh,Gv,Bh,Bv)",
                       rgb_tr, rgb_te, RGB_N_BLOCKS, RGB_N_BVALS, RGB_SIG_PAD,
                       dtr, dte, dw, 6);
    }

    /* ================================================================
     * EXPERIMENT E: Full Encoding D, dot on ALL channels (px+hg+vg+Rhg+Rvg+Ghg+Gvg+Bhg+Bvg)
     * ================================================================ */
    {
        const int8_t *dtr[]={tern_tr, hg_tr, vg_tr, hg_r_tr, vg_r_tr, hg_g_tr, vg_g_tr, hg_b_tr, vg_b_tr};
        const int8_t *dte[]={tern_te, hg_te, vg_te, hg_r_te, vg_r_te, hg_g_te, vg_g_te, hg_b_te, vg_b_te};
        int dw[]={256, 192, 192, 128, 128, 128, 128, 128, 128};
        run_experiment("E: Full Encoding D, dot ALL 9 channels",
                       j_full_tr, j_full_te, N_BLOCKS, BYTE_VALS, SIG_PAD,
                       dtr, dte, dw, 9);
    }

    /* ================================================================
     * EXPERIMENT F: Full Encoding D vote + RGB 6-channel gradient dot
     * Best voting (A's 33.76% Bayesian) + best dot signal (D's gradients)
     * ================================================================ */
    {
        const int8_t *dtr[]={hg_r_tr, vg_r_tr, hg_g_tr, vg_g_tr, hg_b_tr, vg_b_tr};
        const int8_t *dte[]={hg_r_te, vg_r_te, hg_g_te, vg_g_te, hg_b_te, vg_b_te};
        int dw[]={256, 256, 256, 256, 256, 256};
        run_experiment("F: Full Encoding D vote + RGB 6-ch gradient dot",
                       j_full_tr, j_full_te, N_BLOCKS, BYTE_VALS, SIG_PAD,
                       dtr, dte, dw, 6);
    }

    /* ================================================================
     * EXPERIMENT G: Full Encoding D vote + grayscale hg+vg dot
     *              (C repeat with wider K/kNN sweep)
     * ================================================================ */
    {
        const int8_t *dtr[]={hg_tr, vg_tr};
        const int8_t *dte[]={hg_te, vg_te};
        int dw[]={256, 256};
        run_experiment("G: Full Encoding D vote + grayscale grad dot (=C)",
                       j_full_tr, j_full_te, N_BLOCKS, BYTE_VALS, SIG_PAD,
                       dtr, dte, dw, 2);
    }

    /* ================================================================
     * EXPERIMENT H: RGB 6-channel gradient vote + Full Encoding D + RGB gradient dot
     * Concatenate RGB grad sigs (1920 blocks) with Encoding D (320 blocks)
     * for a combined 2240-block signature, dot on RGB gradients
     * ================================================================ */
    {
        /* Build combined signatures: Encoding D (320) + RGB grads (1920) = 2240 */
        #define COMBO_N_BLOCKS 2240
        #define COMBO_SIG_PAD  2240  /* 2240 % 32 == 0 */
        uint8_t *combo_tr = aligned_alloc(32, (size_t)TRAIN_N * COMBO_SIG_PAD);
        uint8_t *combo_te = aligned_alloc(32, (size_t)TEST_N * COMBO_SIG_PAD);
        for (int i = 0; i < TRAIN_N; i++) {
            uint8_t *dst = combo_tr + (size_t)i * COMBO_SIG_PAD;
            memcpy(dst, j_full_tr + (size_t)i * SIG_PAD, N_BLOCKS);
            memcpy(dst + N_BLOCKS, rgb_tr + (size_t)i * RGB_SIG_PAD, RGB_N_BLOCKS);
        }
        for (int i = 0; i < TEST_N; i++) {
            uint8_t *dst = combo_te + (size_t)i * COMBO_SIG_PAD;
            memcpy(dst, j_full_te + (size_t)i * SIG_PAD, N_BLOCKS);
            memcpy(dst + N_BLOCKS, rgb_te + (size_t)i * RGB_SIG_PAD, RGB_N_BLOCKS);
        }
        /* First 320 blocks have 256 values (Encoding D), next 1920 have 27 values (grad blocks).
         * For the generic runner we need a single n_vals — use 256 and let the 27-val blocks
         * just have empty buckets for values 27-255. This wastes hot map memory but works. */
        const int8_t *dtr[]={hg_r_tr, vg_r_tr, hg_g_tr, vg_g_tr, hg_b_tr, vg_b_tr};
        const int8_t *dte[]={hg_r_te, vg_r_te, hg_g_te, vg_g_te, hg_b_te, vg_b_te};
        int dw[]={256, 256, 256, 256, 256, 256};
        run_experiment("H: Combined Encoding D + RGB grad vote, RGB grad dot",
                       combo_tr, combo_te, COMBO_N_BLOCKS, BYTE_VALS, COMBO_SIG_PAD,
                       dtr, dte, dw, 6);
        free(combo_tr); free(combo_te);
    }

    printf("\nTotal: %.1f sec\n", now_sec()-t0);

    /* Cleanup omitted for brevity — process exits */
    return 0;
}
