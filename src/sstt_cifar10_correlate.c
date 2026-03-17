/*
 * sstt_cifar10_correlate.c — What predicts success?
 *
 * For each test image, compute raw data features and compare distributions
 * between correctly classified and incorrectly classified images.
 *
 * Features measured:
 *   1. Mean brightness (per channel and overall)
 *   2. Contrast (stddev of pixel values)
 *   3. Edge density (fraction of non-zero ternary gradients)
 *   4. Color dominance (max channel - min channel mean)
 *   5. Spatial concentration (center vs border pixel energy)
 *   6. Block entropy (unique Encoding D values / total blocks)
 *   7. Divergence magnitude (|neg_divergence|)
 *   8. Background fraction (how many blocks match BG)
 *   9. Vote agreement (top-class count in top-200)
 *  10. Trit balance (fraction of zeros in ternary)
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_correlate src/sstt_cifar10_correlate.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 50000
#define TEST_N  10000
#define N_CLASSES 10
#define CLS_PAD 16
#define BYTE_VALS 256
#define BG_TRANS 13
#define IG_SCALE 16
#define TOP_K 200
#define IMG_W 96
#define IMG_H 32
#define PIXELS 3072
#define PADDED 3072
#define BLKS_PER_ROW 32
#define N_BLOCKS 1024
#define SIG_PAD 1024
#define H_TRANS_PER_ROW 31
#define N_HTRANS (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 31
#define N_VTRANS (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD 992
#define SPATIAL_W 32
#define SPATIAL_H 32
#define GRAY_PIXELS 1024
#define GRAY_PADDED 1024

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static uint8_t *raw_r_te,*raw_g_te,*raw_b_te; /* per-channel test */
static uint8_t *raw_gray_tr,*raw_gray_te;
static int8_t *tern_tr,*tern_te,*hg_te,*vg_te;
static int8_t *ghg_tr,*ghg_te,*gvg_tr,*gvg_te;
static int16_t *divneg_te;
static uint8_t *joint_tr,*joint_te;
static uint32_t *joint_hot;
static uint16_t ig_w[N_BLOCKS];
static uint8_t nbr[BYTE_VALS][8],bg_val;
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;

/* Minimal core functions */
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void qt(const uint8_t*s,int8_t*d,int n,int px,int pd){
    const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
        tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
    for(int i2=0;i2<n;i2++){const uint8_t*si=s+(size_t)i2*px;int8_t*di=d+(size_t)i2*pd;int i;
        for(i=0;i+32<=px;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));__m256i sp=_mm256_xor_si256(p,bi);
            _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}for(;i<px;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;memset(di+px,0,pd-px);}}
static void gr(const int8_t*t,int8_t*h,int8_t*v,int n,int w,int ht,int pd){
    for(int i2=0;i2<n;i2++){const int8_t*ti=t+(size_t)i2*pd;int8_t*hi=h+(size_t)i2*pd,*vi=v+(size_t)i2*pd;
        for(int y=0;y<ht;y++){for(int x=0;x<w-1;x++)hi[y*w+x]=ct(ti[y*w+x+1]-ti[y*w+x]);hi[y*w+w-1]=0;}
        for(int y=0;y<ht-1;y++)for(int x=0;x<w;x++)vi[y*w+x]=ct(ti[(y+1)*w+x]-ti[y*w+x]);memset(vi+(ht-1)*w,0,w);}}
static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void bsf(const int8_t*d,uint8_t*s,int n){for(int i=0;i<n;i++){const int8_t*im=d+(size_t)i*PADDED;
    uint8_t*si=s+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s2=0;s2<BLKS_PER_ROW;s2++){
        int b2=y*IMG_W+s2*3;si[y*BLKS_PER_ROW+s2]=be(im[b2],im[b2+1],im[b2+2]);}}}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,
    b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}
static void trf(const uint8_t*bsi,uint8_t*ht,uint8_t*vt,int n){
    for(int i=0;i<n;i++){const uint8_t*s=bsi+(size_t)i*SIG_PAD;uint8_t*h=ht+(size_t)i*TRANS_PAD,*v=vt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static void jsf(uint8_t*o,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){
    for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=o+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);
            int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}}
static void divf(const int8_t*hg,const int8_t*vg,int16_t*ns){
    int neg=0;for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0)neg+=d;}*ns=(int16_t)(neg<-32767?-32767:neg);}
typedef struct{uint32_t id;uint32_t votes;int64_t score;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc]=(cand_t){0};o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_gray_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    raw_r_te=malloc((size_t)TEST_N*GRAY_PIXELS);raw_g_te=malloc((size_t)TEST_N*GRAY_PIXELS);raw_b_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gd=raw_gray_tr+(size_t)idx*GRAY_PIXELS;
            for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        memcpy(raw_r_te+(size_t)i*GRAY_PIXELS,r,1024);memcpy(raw_g_te+(size_t)i*GRAY_PIXELS,g,1024);memcpy(raw_b_te+(size_t)i*GRAY_PIXELS,b3,1024);
        uint8_t*gd=raw_gray_te+(size_t)i*GRAY_PIXELS;
        for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

/* ================================================================
 *  Per-image feature computation
 * ================================================================ */
typedef struct {
    float brightness;      /* mean grayscale value */
    float contrast;        /* stddev of grayscale */
    float r_mean,g_mean,b_mean; /* per-channel means */
    float color_dom;       /* max channel mean - min channel mean */
    float edge_density;    /* fraction of non-zero h+v gradients */
    float center_energy;   /* fraction of foreground in center 16x16 vs border */
    float block_entropy;   /* unique Encoding D values / N_BLOCKS */
    float bg_fraction;     /* fraction of blocks matching background */
    float trit_zero_frac;  /* fraction of ternary values == 0 */
    float divergence_mag;  /* |neg_divergence| */
    int   vote_agreement;  /* max class count in top-200 */
    int   predicted;       /* what we predicted */
    int   correct;         /* 1 if correct, 0 if not */
} image_features_t;

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Feature-Accuracy Correlation ===\n\n");
    load_data();

    /* Build pipeline */
    tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    qt(raw_train,tern_tr,TRAIN_N,PIXELS,PADDED);qt(raw_test,tern_te,TEST_N,PIXELS,PADDED);
    {int8_t*hg_tr2=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vg_tr2=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
     gr(tern_tr,hg_tr2,vg_tr2,TRAIN_N,IMG_W,IMG_H,PADDED);free(hg_tr2);free(vg_tr2);}
    gr(tern_te,hg_te,vg_te,TEST_N,IMG_W,IMG_H,PADDED);
    {int8_t*gt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     ghg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);gvg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     qt(raw_gray_tr,gt_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);qt(raw_gray_te,gt_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
     {int8_t*ghg_tr2=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gvg_tr2=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);
      gr(gt_tr,ghg_tr2,gvg_tr2,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);free(ghg_tr2);free(gvg_tr2);}
     gr(gt_te,ghg_te,gvg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
     divneg_te=malloc((size_t)TEST_N*2);int16_t dummy;
     for(int i=0;i<TEST_N;i++)divf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,&divneg_te[i]);
     free(gt_tr);free(gt_te);}

    /* Block sigs + Encoding D */
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(tern_tr,pxt,TRAIN_N);bsf(tern_te,pxe,TEST_N);
    {int8_t*hgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);
     int8_t*hgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED),*vgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
     gr(tern_tr,hgc_tr,vgc_tr,TRAIN_N,IMG_W,IMG_H,PADDED);gr(tern_te,hgc_te,vgc_te,TEST_N,IMG_W,IMG_H,PADDED);
     bsf(hgc_tr,hst,TRAIN_N);bsf(hgc_te,hse,TEST_N);bsf(vgc_tr,vst,TRAIN_N);bsf(vgc_te,vse,TEST_N);
     uint8_t*htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
     uint8_t*vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
     trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte2,vte2,TEST_N);
     joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
     jsf(joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(joint_te,TEST_N,pxe,hse,vse,hte2,vte2);
     free(hgc_tr);free(vgc_tr);free(hgc_te);free(vgc_te);free(htt);free(hte2);free(vtt);free(vte2);}

    /* BG + hot map + IG + index */
    {long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}bg_val=0;long mc=0;
        for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg_val=(uint8_t)v;}}
    joint_hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(joint_hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)joint_hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw2[N_BLOCKS],mx=0;
     for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==bg_val)continue;
             const uint32_t*h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}
             hcond+=pv*hv;}raw2[k]=hc-hcond;if(raw2[k]>mx)mx=raw2[k];}
     for(int k=0;k<N_BLOCKS;k++){ig_w[k]=mx>0?(uint16_t)(raw2[k]/mx*IG_SCALE+0.5):1;if(!ig_w[k])ig_w[k]=1;}}
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)nbr[v][b3]=(uint8_t)(v^(1<<b3));
    memset(idx_sz,0,sizeof(idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){idx_off[k][v]=tot;tot+=idx_sz[k][v];}
    idx_pool=malloc((size_t)tot*4);
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=bg_val)idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * COMPUTE PER-IMAGE FEATURES + CLASSIFY
     * ================================================================ */
    image_features_t *feats = malloc(TEST_N * sizeof(image_features_t));
    uint32_t *vbuf = calloc(TRAIN_N, 4);

    printf("Computing per-image features...\n");
    for (int i = 0; i < TEST_N; i++) {
        image_features_t *f = &feats[i];
        const uint8_t *gray = raw_gray_te + (size_t)i * GRAY_PIXELS;
        const uint8_t *r = raw_r_te + (size_t)i * GRAY_PIXELS;
        const uint8_t *g = raw_g_te + (size_t)i * GRAY_PIXELS;
        const uint8_t *b = raw_b_te + (size_t)i * GRAY_PIXELS;
        const int8_t *tern = tern_te + (size_t)i * PADDED;
        const int8_t *hgi = hg_te + (size_t)i * PADDED;
        const int8_t *vgi = vg_te + (size_t)i * PADDED;
        const uint8_t *jsig = joint_te + (size_t)i * SIG_PAD;

        /* 1. Brightness + contrast */
        double sum = 0, sum2 = 0;
        for (int p = 0; p < 1024; p++) { sum += gray[p]; sum2 += (double)gray[p] * gray[p]; }
        f->brightness = (float)(sum / 1024);
        f->contrast = (float)sqrt(sum2 / 1024 - f->brightness * f->brightness);

        /* 2. Per-channel means */
        double rs = 0, gs = 0, bs = 0;
        for (int p = 0; p < 1024; p++) { rs += r[p]; gs += g[p]; bs += b[p]; }
        f->r_mean = (float)(rs / 1024); f->g_mean = (float)(gs / 1024); f->b_mean = (float)(bs / 1024);
        float mx_ch = f->r_mean > f->g_mean ? (f->r_mean > f->b_mean ? f->r_mean : f->b_mean) : (f->g_mean > f->b_mean ? f->g_mean : f->b_mean);
        float mn_ch = f->r_mean < f->g_mean ? (f->r_mean < f->b_mean ? f->r_mean : f->b_mean) : (f->g_mean < f->b_mean ? f->g_mean : f->b_mean);
        f->color_dom = mx_ch - mn_ch;

        /* 3. Edge density */
        int edge_count = 0;
        for (int p = 0; p < PIXELS; p++) { if (hgi[p] != 0) edge_count++; if (vgi[p] != 0) edge_count++; }
        f->edge_density = (float)edge_count / (2.0f * PIXELS);

        /* 4. Center energy (center 16x48 vs border in 32x96 flattened) */
        int center_fg = 0, border_fg = 0;
        for (int y = 0; y < IMG_H; y++) for (int x = 0; x < IMG_W; x++) {
            int is_center = (y >= 8 && y < 24 && x >= 24 && x < 72);
            if (tern[y * IMG_W + x] != 0) { if (is_center) center_fg++; else border_fg++; }
        }
        f->center_energy = (center_fg + border_fg > 0) ? (float)center_fg / (center_fg + border_fg) : 0.5f;

        /* 5. Block entropy */
        uint8_t seen[BYTE_VALS] = {0}; int unique = 0;
        for (int k = 0; k < N_BLOCKS; k++) if (!seen[jsig[k]]) { seen[jsig[k]] = 1; unique++; }
        f->block_entropy = (float)unique / N_BLOCKS;

        /* 6. BG fraction */
        int bg_count = 0;
        for (int k = 0; k < N_BLOCKS; k++) if (jsig[k] == bg_val) bg_count++;
        f->bg_fraction = (float)bg_count / N_BLOCKS;

        /* 7. Trit zero fraction */
        int zero_count = 0;
        for (int p = 0; p < PIXELS; p++) if (tern[p] == 0) zero_count++;
        f->trit_zero_frac = (float)zero_count / PIXELS;

        /* 8. Divergence magnitude */
        f->divergence_mag = (float)abs(divneg_te[i]);

        /* 9. Vote + classify (Bayesian) */
        memset(vbuf, 0, TRAIN_N * 4);
        for (int k = 0; k < N_BLOCKS; k++) {
            uint8_t bv = jsig[k]; if (bv == bg_val) continue;
            uint16_t w = ig_w[k], wh = w > 1 ? w / 2 : 1;
            { uint32_t off = idx_off[k][bv]; uint16_t sz = idx_sz[k][bv];
              for (uint16_t j = 0; j < sz; j++) vbuf[idx_pool[off + j]] += w; }
            for (int nb = 0; nb < 8; nb++) { uint8_t nv = nbr[bv][nb]; if (nv == bg_val) continue;
                uint32_t noff = idx_off[k][nv]; uint16_t nsz = idx_sz[k][nv];
                for (uint16_t j = 0; j < nsz; j++) vbuf[idx_pool[noff + j]] += wh; }
        }
        cand_t cands[TOP_K]; int nc = topk(vbuf, TRAIN_N, cands, TOP_K);
        int lc[N_CLASSES] = {0};
        for (int j = 0; j < nc; j++) lc[train_labels[cands[j].id]]++;
        int ma = 0; for (int c = 0; c < N_CLASSES; c++) if (lc[c] > ma) ma = lc[c];
        f->vote_agreement = ma;

        /* Bayesian prediction */
        double bp[N_CLASSES]; for (int c = 0; c < N_CLASSES; c++) bp[c] = 0.0;
        for (int k = 0; k < N_BLOCKS; k++) { uint8_t bv = jsig[k]; if (bv == bg_val) continue;
            const uint32_t *h = joint_hot + (size_t)k * BYTE_VALS * CLS_PAD + (size_t)bv * CLS_PAD;
            for (int c = 0; c < N_CLASSES; c++) bp[c] += log(h[c] + 0.5); }
        int pred = 0; for (int c = 1; c < N_CLASSES; c++) if (bp[c] > bp[pred]) pred = c;
        f->predicted = pred;
        f->correct = (pred == test_labels[i]) ? 1 : 0;

        if ((i + 1) % 2000 == 0) fprintf(stderr, "  %d/%d\r", i + 1, TEST_N);
    }
    fprintf(stderr, "\n");
    free(vbuf);

    /* ================================================================
     * ANALYSIS: Compare distributions
     * ================================================================ */
    int n_correct = 0, n_wrong = 0;
    for (int i = 0; i < TEST_N; i++) { if (feats[i].correct) n_correct++; else n_wrong++; }

    printf("\n=== FEATURE DISTRIBUTIONS: Correct (%d) vs Wrong (%d) ===\n\n", n_correct, n_wrong);

    /* Compute means for correct and wrong */
    #define N_FEATS 12
    const char *feat_names[] = {"brightness","contrast","R mean","G mean","B mean",
        "color dominance","edge density","center energy","block entropy","BG fraction",
        "trit zero frac","divergence mag"};

    double sum_c[N_FEATS] = {0}, sum_w[N_FEATS] = {0};
    double sum2_c[N_FEATS] = {0}, sum2_w[N_FEATS] = {0};
    double va_sum_c = 0, va_sum_w = 0;

    for (int i = 0; i < TEST_N; i++) {
        image_features_t *f = &feats[i];
        float vals[] = {f->brightness, f->contrast, f->r_mean, f->g_mean, f->b_mean,
            f->color_dom, f->edge_density, f->center_energy, f->block_entropy, f->bg_fraction,
            f->trit_zero_frac, f->divergence_mag};
        for (int fi = 0; fi < N_FEATS; fi++) {
            if (f->correct) { sum_c[fi] += vals[fi]; sum2_c[fi] += vals[fi] * vals[fi]; }
            else { sum_w[fi] += vals[fi]; sum2_w[fi] += vals[fi] * vals[fi]; }
        }
        if (f->correct) va_sum_c += f->vote_agreement; else va_sum_w += f->vote_agreement;
    }

    printf("  %-20s  Correct    Wrong      Delta    Discriminative?\n", "Feature");
    printf("  %-20s  -------    -----      -----    ---------------\n", "");
    for (int fi = 0; fi < N_FEATS; fi++) {
        double mc = sum_c[fi] / n_correct, mw = sum_w[fi] / n_wrong;
        double sc = sqrt(sum2_c[fi] / n_correct - mc * mc);
        double sw = sqrt(sum2_w[fi] / n_wrong - mw * mw);
        double pooled_sd = sqrt((sc * sc + sw * sw) / 2);
        double effect_size = pooled_sd > 0 ? fabs(mc - mw) / pooled_sd : 0;
        const char *disc = effect_size > 0.3 ? "*** STRONG" : effect_size > 0.15 ? "** moderate" : effect_size > 0.05 ? "* weak" : "  none";
        printf("  %-20s  %7.2f    %7.2f    %+6.2f   d=%.3f %s\n",
               feat_names[fi], mc, mw, mc - mw, effect_size, disc);
    }
    {double mc = va_sum_c / n_correct, mw = va_sum_w / n_wrong;
     printf("  %-20s  %7.1f    %7.1f    %+6.1f   (see curve)\n", "vote agreement", mc, mw, mc - mw);}

    /* ================================================================
     * PER-CLASS FEATURE COMPARISON
     * ================================================================ */
    printf("\n=== PER-CLASS: What distinguishes easy from hard? ===\n\n");
    for (int cls = 0; cls < N_CLASSES; cls++) {
        double br_c = 0, br_w = 0, con_c = 0, con_w = 0, ed_c = 0, ed_w = 0;
        double cd_c = 0, cd_w = 0, ce_c = 0, ce_w = 0, bg_c = 0, bg_w = 0;
        int nc2 = 0, nw = 0;
        for (int i = 0; i < TEST_N; i++) {
            if (test_labels[i] != cls) continue;
            image_features_t *f = &feats[i];
            if (f->correct) { br_c += f->brightness; con_c += f->contrast; ed_c += f->edge_density;
                cd_c += f->color_dom; ce_c += f->center_energy; bg_c += f->bg_fraction; nc2++; }
            else { br_w += f->brightness; con_w += f->contrast; ed_w += f->edge_density;
                cd_w += f->color_dom; ce_w += f->center_energy; bg_w += f->bg_fraction; nw++; }
        }
        if (nc2 == 0 || nw == 0) continue;
        printf("  %d %-10s (acc=%.0f%%): correct=%d wrong=%d\n", cls, cn[cls], 100.0 * nc2 / (nc2 + nw), nc2, nw);
        printf("    brightness:  correct=%.1f  wrong=%.1f  delta=%+.1f\n", br_c / nc2, br_w / nw, br_c / nc2 - br_w / nw);
        printf("    contrast:    correct=%.1f  wrong=%.1f  delta=%+.1f\n", con_c / nc2, con_w / nw, con_c / nc2 - con_w / nw);
        printf("    color dom:   correct=%.1f  wrong=%.1f  delta=%+.1f\n", cd_c / nc2, cd_w / nw, cd_c / nc2 - cd_w / nw);
        printf("    edge density:correct=%.3f  wrong=%.3f  delta=%+.3f\n", ed_c / nc2, ed_w / nw, ed_c / nc2 - ed_w / nw);
        printf("    center E:    correct=%.3f  wrong=%.3f  delta=%+.3f\n", ce_c / nc2, ce_w / nw, ce_c / nc2 - ce_w / nw);
        printf("    BG fraction: correct=%.3f  wrong=%.3f  delta=%+.3f\n\n", bg_c / nc2, bg_w / nw, bg_c / nc2 - bg_w / nw);
    }

    /* ================================================================
     * BINNED ACCURACY BY FEATURE VALUE
     * ================================================================ */
    printf("=== ACCURACY BY FEATURE BINS ===\n\n");

    /* Brightness bins */
    printf("  Brightness:\n");
    for (int bin = 0; bin < 5; bin++) {
        float lo = bin * 51.2f, hi = (bin + 1) * 51.2f;
        int count = 0, correct = 0;
        for (int i = 0; i < TEST_N; i++) if (feats[i].brightness >= lo && feats[i].brightness < hi) {
            count++; if (feats[i].correct) correct++; }
        if (count > 50) printf("    [%5.1f-%5.1f]: %4d images, %.1f%% accuracy\n", lo, hi, count, 100.0 * correct / count);
    }

    /* Contrast bins */
    printf("  Contrast:\n");
    for (int bin = 0; bin < 5; bin++) {
        float lo = bin * 20.0f, hi = (bin + 1) * 20.0f;
        int count = 0, correct = 0;
        for (int i = 0; i < TEST_N; i++) if (feats[i].contrast >= lo && feats[i].contrast < hi) {
            count++; if (feats[i].correct) correct++; }
        if (count > 50) printf("    [%5.1f-%5.1f]: %4d images, %.1f%% accuracy\n", lo, hi, count, 100.0 * correct / count);
    }

    /* Edge density bins */
    printf("  Edge density:\n");
    for (int bin = 0; bin < 5; bin++) {
        float lo = 0.3f + bin * 0.1f, hi = lo + 0.1f;
        int count = 0, correct = 0;
        for (int i = 0; i < TEST_N; i++) if (feats[i].edge_density >= lo && feats[i].edge_density < hi) {
            count++; if (feats[i].correct) correct++; }
        if (count > 50) printf("    [%.2f-%.2f]: %4d images, %.1f%% accuracy\n", lo, hi, count, 100.0 * correct / count);
    }

    /* Color dominance bins */
    printf("  Color dominance:\n");
    for (int bin = 0; bin < 5; bin++) {
        float lo = bin * 20.0f, hi = (bin + 1) * 20.0f;
        int count = 0, correct = 0;
        for (int i = 0; i < TEST_N; i++) if (feats[i].color_dom >= lo && feats[i].color_dom < hi) {
            count++; if (feats[i].correct) correct++; }
        if (count > 50) printf("    [%5.1f-%5.1f]: %4d images, %.1f%% accuracy\n", lo, hi, count, 100.0 * correct / count);
    }

    /* BG fraction bins */
    printf("  BG fraction:\n");
    for (int bin = 0; bin < 5; bin++) {
        float lo = bin * 0.1f, hi = (bin + 1) * 0.1f;
        int count = 0, correct = 0;
        for (int i = 0; i < TEST_N; i++) if (feats[i].bg_fraction >= lo && feats[i].bg_fraction < hi) {
            count++; if (feats[i].correct) correct++; }
        if (count > 50) printf("    [%.2f-%.2f]: %4d images, %.1f%% accuracy\n", lo, hi, count, 100.0 * correct / count);
    }

    printf("\nTotal: %.1f sec\n", now_sec() - t0);
    free(feats);
    return 0;
}
