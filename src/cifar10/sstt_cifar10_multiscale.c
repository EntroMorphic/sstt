/*
 * sstt_cifar10_multiscale.c — Multi-Scale Block Falsification
 *
 * Hypothesis: 3-pixel blocks see texture; 6-pixel blocks (via 2x downsample)
 * see object parts. The combination captures both texture and structure.
 *
 * Architecture:
 *   Scale 1: Original 32x96 flattened RGB → 32 blocks/row × 32 rows = 1024 blocks
 *   Scale 2: 2x downsampled 16x48 → 16 blocks/row × 16 rows = 256 blocks
 *   Scale 3: 4x downsampled 8x24 → 8 blocks/row × 8 rows = 64 blocks
 *
 * Each scale gets its own Encoding D, IG weights, inverted index.
 * Combined vote across all scales. MT4 dot on original + downsampled.
 *
 * If multi-scale beats single-scale: larger blocks capture complementary info.
 * If not: the representation ceiling is scale-independent.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_multiscale src/sstt_cifar10_multiscale.c -lm
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

/* Scale 1: 32x96 (original) */
#define S1_W 96
#define S1_H 32
#define S1_PX 3072
#define S1_PAD 3072
#define S1_BPR 32
#define S1_NB 1024
#define S1_SP 1024
#define S1_HTPR 31
#define S1_NHT (S1_HTPR*S1_H)
#define S1_VTPC 31
#define S1_NVT (S1_BPR*S1_VTPC)
#define S1_TP 992

/* Scale 2: 16x48 (2x down) */
#define S2_W 48
#define S2_H 16
#define S2_PX 768
#define S2_PAD 768   /* 768 % 32 == 0 */
#define S2_BPR 16
#define S2_NB 256
#define S2_SP 256
#define S2_HTPR 15
#define S2_NHT (S2_HTPR*S2_H)
#define S2_VTPC 15
#define S2_NVT (S2_BPR*S2_VTPC)
#define S2_TP 256

/* Scale 3: 8x24 (4x down) */
#define S3_W 24
#define S3_H 8
#define S3_PX 192
#define S3_PAD 192   /* 192 % 32 == 0 */
#define S3_BPR 8
#define S3_NB 64
#define S3_SP 64
#define S3_HTPR 7
#define S3_NHT (S3_HTPR*S3_H)
#define S3_VTPC 7
#define S3_NVT (S3_BPR*S3_VTPC)
#define S3_TP 64

/* Max across scales for combined arrays */
#define MAX_NB 1024
#define MAX_PAD 3072

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
/* Downsampled raw images */
static uint8_t *raw2_tr,*raw2_te; /* 16x48 */
static uint8_t *raw3_tr,*raw3_te; /* 8x24 */

/* Per-scale: ternary, gradients, sigs, Encoding D, hot map, IG, index */
typedef struct {
    int w,h,px,pad,bpr,nb,sp,htpr,nht,vtpc,nvt,tp;
    int8_t *tern_tr,*tern_te,*hg_tr,*hg_te,*vg_tr,*vg_te;
    uint8_t *px_tr,*px_te,*hgs_tr,*hgs_te,*vgs_tr,*vgs_te;
    uint8_t *ht_tr,*ht_te,*vt_tr,*vt_te;
    uint8_t *joint_tr,*joint_te;
    uint32_t *hot;
    uint16_t ig[MAX_NB];
    uint8_t bg;
    uint32_t *idx_off_flat; /* nb*256 */
    uint16_t *idx_sz_flat;
    uint32_t *idx_pool;
    uint32_t idx_total;
} scale_t;

static scale_t scales[3];
static uint8_t nbr[BYTE_VALS][8];

static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static void quant(const uint8_t*s,int8_t*d,int n,int px,int pd){
    const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
        tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
    for(int i2=0;i2<n;i2++){const uint8_t*si=s+(size_t)i2*px;int8_t*di=d+(size_t)i2*pd;int i;
        for(i=0;i+32<=px;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));__m256i sp=_mm256_xor_si256(p,bi);
            _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}for(;i<px;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;
        memset(di+px,0,pd-px);}}

static void grads(const int8_t*t,int8_t*h,int8_t*v,int n,int w,int ht,int pd){
    for(int i2=0;i2<n;i2++){const int8_t*ti=t+(size_t)i2*pd;int8_t*hi=h+(size_t)i2*pd,*vi=v+(size_t)i2*pd;
        for(int y=0;y<ht;y++){for(int x=0;x<w-1;x++)hi[y*w+x]=ct(ti[y*w+x+1]-ti[y*w+x]);hi[y*w+w-1]=0;}
        for(int y=0;y<ht-1;y++)for(int x=0;x<w;x++)vi[y*w+x]=ct(ti[(y+1)*w+x]-ti[y*w+x]);
        memset(vi+(ht-1)*w,0,w);}}

static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}

static void bsigs(const int8_t*d,uint8_t*s,int n,int w,int h,int bpr,int pad,int sp){
    for(int i=0;i<n;i++){const int8_t*im=d+(size_t)i*pad;uint8_t*si=s+(size_t)i*sp;
        for(int y=0;y<h;y++)for(int s2=0;s2<bpr;s2++){int b2=y*w+s2*3;
            si[y*bpr+s2]=be(im[b2],im[b2+1],im[b2+2]);}}}

static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,
    b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}

static void trans(const uint8_t*bs,uint8_t*ht,uint8_t*vt,int n,int bpr,int h,int htpr,int nht,int vtpc,int nvt,int sp,int tp){
    for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*sp;uint8_t*hh=ht+(size_t)i*tp,*v=vt+(size_t)i*tp;
        for(int y=0;y<h;y++)for(int ss=0;ss<htpr;ss++)hh[y*htpr+ss]=te2(s[y*bpr+ss],s[y*bpr+ss+1]);
        memset(hh+nht,0xFF,tp-nht);
        for(int y=0;y<vtpc;y++)for(int ss=0;ss<bpr;ss++)v[y*bpr+ss]=te2(s[y*bpr+ss],s[(y+1)*bpr+ss]);
        memset(v+nvt,0xFF,tp-nvt);}}

static void jsigs(uint8_t*o,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,
                  const uint8_t*ht,const uint8_t*vt,int bpr,int h,int htpr,int nb,int sp,int tp){
    for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*sp,*hi=hg+(size_t)i*sp,*vi=vg+(size_t)i*sp;
        const uint8_t*hti=ht+(size_t)i*tp,*vti=vt+(size_t)i*tp;uint8_t*oi=o+(size_t)i*sp;
        for(int y=0;y<h;y++)for(int s=0;s<bpr;s++){int k=y*bpr+s;
            uint8_t htb=(s>0)?hti[y*htpr+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*bpr+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);
            int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}}

static void bay_post(const uint8_t*sig,uint8_t bg,const uint32_t*hot,int nb,int sp,double*lp){
    for(int c=0;c<N_CLASSES;c++)lp[c]=0.0;
    for(int k=0;k<nb;k++){uint8_t bv=sig[k];if(bv==bg)continue;
        const uint32_t*h=hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)lp[c]+=log(h[c]+0.5);}
    double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(lp[c]>mx)mx=lp[c];
    for(int c=0;c<N_CLASSES;c++)lp[c]-=mx;}

static int32_t tdot(const int8_t*a,const int8_t*b,int len){int32_t tot=0;
    for(int ch=0;ch<len;ch+=64){__m256i ac=_mm256_setzero_si256();int e=ch+64;if(e>len)e=len;
        for(int i=ch;i<e;i+=32)ac=_mm256_add_epi8(ac,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
        __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(ac));
        __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(ac,1));
        __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
        __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
        s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);tot+=_mm_cvtsi128_si32(s);}return tot;}

typedef struct{uint32_t id;uint32_t votes;int64_t score;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmps(const void*a,const void*b){int64_t d=((const cand_t*)b)->score-((const cand_t*)a)->score;return(d>0)-(d<0);}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc]=(cand_t){0};o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

/* Build one scale's full pipeline */
static void build_scale(scale_t *sc, const uint8_t *raw_tr, const uint8_t *raw_te, int si) {
    int w=sc->w,h=sc->h,px=sc->px,pad=sc->pad,bpr=sc->bpr,nb=sc->nb,sp=sc->sp;
    int htpr=sc->htpr,nht=sc->nht,vtpc=sc->vtpc,nvt=sc->nvt,tp=sc->tp;

    sc->tern_tr=aligned_alloc(32,(size_t)TRAIN_N*pad);sc->tern_te=aligned_alloc(32,(size_t)TEST_N*pad);
    sc->hg_tr=aligned_alloc(32,(size_t)TRAIN_N*pad);sc->hg_te=aligned_alloc(32,(size_t)TEST_N*pad);
    sc->vg_tr=aligned_alloc(32,(size_t)TRAIN_N*pad);sc->vg_te=aligned_alloc(32,(size_t)TEST_N*pad);
    quant(raw_tr,sc->tern_tr,TRAIN_N,px,pad);quant(raw_te,sc->tern_te,TEST_N,px,pad);
    grads(sc->tern_tr,sc->hg_tr,sc->vg_tr,TRAIN_N,w,h,pad);
    grads(sc->tern_te,sc->hg_te,sc->vg_te,TEST_N,w,h,pad);

    sc->px_tr=aligned_alloc(32,(size_t)TRAIN_N*sp);sc->px_te=aligned_alloc(32,(size_t)TEST_N*sp);
    sc->hgs_tr=aligned_alloc(32,(size_t)TRAIN_N*sp);sc->hgs_te=aligned_alloc(32,(size_t)TEST_N*sp);
    sc->vgs_tr=aligned_alloc(32,(size_t)TRAIN_N*sp);sc->vgs_te=aligned_alloc(32,(size_t)TEST_N*sp);
    bsigs(sc->tern_tr,sc->px_tr,TRAIN_N,w,h,bpr,pad,sp);bsigs(sc->tern_te,sc->px_te,TEST_N,w,h,bpr,pad,sp);
    bsigs(sc->hg_tr,sc->hgs_tr,TRAIN_N,w,h,bpr,pad,sp);bsigs(sc->hg_te,sc->hgs_te,TEST_N,w,h,bpr,pad,sp);
    bsigs(sc->vg_tr,sc->vgs_tr,TRAIN_N,w,h,bpr,pad,sp);bsigs(sc->vg_te,sc->vgs_te,TEST_N,w,h,bpr,pad,sp);

    sc->ht_tr=aligned_alloc(32,(size_t)TRAIN_N*tp);sc->ht_te=aligned_alloc(32,(size_t)TEST_N*tp);
    sc->vt_tr=aligned_alloc(32,(size_t)TRAIN_N*tp);sc->vt_te=aligned_alloc(32,(size_t)TEST_N*tp);
    trans(sc->px_tr,sc->ht_tr,sc->vt_tr,TRAIN_N,bpr,h,htpr,nht,vtpc,nvt,sp,tp);
    trans(sc->px_te,sc->ht_te,sc->vt_te,TEST_N,bpr,h,htpr,nht,vtpc,nvt,sp,tp);

    sc->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*sp);sc->joint_te=aligned_alloc(32,(size_t)TEST_N*sp);
    jsigs(sc->joint_tr,TRAIN_N,sc->px_tr,sc->hgs_tr,sc->vgs_tr,sc->ht_tr,sc->vt_tr,bpr,h,htpr,nb,sp,tp);
    jsigs(sc->joint_te,TEST_N,sc->px_te,sc->hgs_te,sc->vgs_te,sc->ht_te,sc->vt_te,bpr,h,htpr,nb,sp,tp);

    /* Background */
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=sc->joint_tr+(size_t)i*sp;
        for(int k=0;k<nb;k++)vc[sig[k]]++;}
    sc->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];sc->bg=(uint8_t)v;}
    printf("    Scale %d (%dx%d): BG=%d (%.1f%%)\n",si,w,h,sc->bg,100.0*mc/((long)TRAIN_N*nb));

    /* Hot map + IG */
    sc->hot=aligned_alloc(32,(size_t)nb*BYTE_VALS*CLS_PAD*4);
    memset(sc->hot,0,(size_t)nb*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=sc->joint_tr+(size_t)i*sp;
        for(int k=0;k<nb;k++)sc->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw2[MAX_NB],mx=0;
     for(int k=0;k<nb;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==sc->bg)continue;
             const uint32_t*h=sc->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}
             hcond+=pv*hv;}raw2[k]=hc-hcond;if(raw2[k]>mx)mx=raw2[k];}
     for(int k=0;k<nb;k++){sc->ig[k]=mx>0?(uint16_t)(raw2[k]/mx*IG_SCALE+0.5):1;if(!sc->ig[k])sc->ig[k]=1;}}

    /* Index */
    sc->idx_off_flat=malloc((size_t)nb*BYTE_VALS*4);sc->idx_sz_flat=calloc((size_t)nb*BYTE_VALS,2);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=sc->joint_tr+(size_t)i*sp;
        for(int k=0;k<nb;k++)if(sig[k]!=sc->bg)sc->idx_sz_flat[k*BYTE_VALS+sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<nb;k++)for(int v=0;v<BYTE_VALS;v++){sc->idx_off_flat[k*BYTE_VALS+v]=tot;tot+=sc->idx_sz_flat[k*BYTE_VALS+v];}
    sc->idx_total=tot;sc->idx_pool=malloc((size_t)tot*4);
    uint32_t*wp=malloc((size_t)nb*BYTE_VALS*4);memcpy(wp,sc->idx_off_flat,(size_t)nb*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=sc->joint_tr+(size_t)i*sp;
        for(int k=0;k<nb;k++)if(sig[k]!=sc->bg)sc->idx_pool[wp[k*BYTE_VALS+sig[k]]++]=(uint32_t)i;}free(wp);
    printf("    Scale %d index: %u entries (%.1f MB)\n",si,tot,(double)tot*4/(1024*1024));
}

/* Vote using one scale */
static void vote_scale(const scale_t*sc,int img,uint32_t*votes){
    const uint8_t*sig=sc->joint_te+(size_t)img*sc->sp;int nb=sc->nb;uint8_t bg=sc->bg;
    for(int k=0;k<nb;k++){uint8_t bv=sig[k];if(bv==bg)continue;
        uint16_t w=sc->ig[k],wh=w>1?w/2:1;
        {uint32_t off=sc->idx_off_flat[k*BYTE_VALS+bv];uint16_t sz=sc->idx_sz_flat[k*BYTE_VALS+bv];
         for(uint16_t j=0;j<sz;j++)votes[sc->idx_pool[off+j]]+=w;}
        for(int n2=0;n2<8;n2++){uint8_t nv=nbr[bv][n2];if(nv==bg)continue;
            uint32_t noff=sc->idx_off_flat[k*BYTE_VALS+nv];uint16_t nsz=sc->idx_sz_flat[k*BYTE_VALS+nv];
            for(uint16_t j=0;j<nsz;j++)votes[sc->idx_pool[noff+j]]+=wh;}}}

/* 2x downsample: average 2x2 patches */
static void downsample2x(const uint8_t*src,uint8_t*dst,int n,int sw,int sh){
    int dw=sw/2,dh=sh/2;
    for(int img=0;img<n;img++){const uint8_t*s=src+(size_t)img*sw*sh;uint8_t*d=dst+(size_t)img*dw*dh;
        for(int y=0;y<dh;y++)for(int x=0;x<dw;x++)
            d[y*dw+x]=(uint8_t)(((int)s[2*y*sw+2*x]+(int)s[2*y*sw+2*x+1]+(int)s[(2*y+1)*sw+2*x]+(int)s[(2*y+1)*sw+2*x+1]+2)/4);}}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*S1_PX);raw_test=malloc((size_t)TEST_N*S1_PX);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*S1_PX;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*S1_PX;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Multi-Scale Blocks ===\n\n");
    load_data();

    /* Downsample */
    printf("Downsampling...\n");
    raw2_tr=malloc((size_t)TRAIN_N*S2_PX);raw2_te=malloc((size_t)TEST_N*S2_PX);
    raw3_tr=malloc((size_t)TRAIN_N*S3_PX);raw3_te=malloc((size_t)TEST_N*S3_PX);
    downsample2x(raw_train,raw2_tr,TRAIN_N,S1_W,S1_H);downsample2x(raw_test,raw2_te,TEST_N,S1_W,S1_H);
    downsample2x(raw2_tr,raw3_tr,TRAIN_N,S2_W,S2_H);downsample2x(raw2_te,raw3_te,TEST_N,S2_W,S2_H);

    /* Initialize scale structs */
    scales[0]=(scale_t){S1_W,S1_H,S1_PX,S1_PAD,S1_BPR,S1_NB,S1_SP,S1_HTPR,S1_NHT,S1_VTPC,S1_NVT,S1_TP};
    scales[1]=(scale_t){S2_W,S2_H,S2_PX,S2_PAD,S2_BPR,S2_NB,S2_SP,S2_HTPR,S2_NHT,S2_VTPC,S2_NVT,S2_TP};
    scales[2]=(scale_t){S3_W,S3_H,S3_PX,S3_PAD,S3_BPR,S3_NB,S3_SP,S3_HTPR,S3_NHT,S3_VTPC,S3_NVT,S3_TP};

    /* Build neighbor table */
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)nbr[v][b3]=(uint8_t)(v^(1<<b3));

    /* Build each scale */
    printf("Building 3 scales...\n");
    const uint8_t *raws_tr[]={raw_train,raw2_tr,raw3_tr};
    const uint8_t *raws_te[]={raw_test,raw2_te,raw3_te};
    for(int si=0;si<3;si++)build_scale(&scales[si],raws_tr[si],raws_te[si],si);
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * Test: single-scale vs multi-scale voting + dot ranking
     * ================================================================ */
    uint32_t *vbuf=calloc(TRAIN_N,4);

    /* Per-scale Bayesian + cascade accuracy */
    int bay_correct[3]={0},cas_correct[3]={0};

    /* Combined: vote across all 3 scales */
    int combined_bay=0, combined_cas=0;
    /* Combined: scales 1+2 only */
    int s12_cas=0;

    printf("Running multi-scale classification...\n");
    for(int i=0;i<TEST_N;i++){
        /* Per-scale Bayesian */
        for(int si=0;si<3;si++){
            double bp[N_CLASSES];
            bay_post(scales[si].joint_te+(size_t)i*scales[si].sp,scales[si].bg,scales[si].hot,scales[si].nb,scales[si].sp,bp);
            int best=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[best])best=c;
            if(best==test_labels[i])bay_correct[si]++;
        }

        /* Per-scale cascade (vote → dot → k=1) */
        for(int si=0;si<3;si++){
            memset(vbuf,0,TRAIN_N*4);vote_scale(&scales[si],i,vbuf);
            cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
            for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
                cands[j].score=tdot(scales[si].hg_te+(size_t)i*scales[si].pad,
                                     scales[si].hg_tr+(size_t)id*scales[si].pad,scales[si].pad)
                              +tdot(scales[si].vg_te+(size_t)i*scales[si].pad,
                                     scales[si].vg_tr+(size_t)id*scales[si].pad,scales[si].pad);}
            qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
            if(nc>0&&train_labels[cands[0].id]==test_labels[i])cas_correct[si]++;
        }

        /* Combined 3-scale vote → multi-scale dot → k=1 */
        memset(vbuf,0,TRAIN_N*4);
        int scale_weights[]={4,2,1}; /* weight finer scales more */
        for(int si=0;si<3;si++){
            /* Temporarily boost votes by scale weight */
            uint32_t *tmp=calloc(TRAIN_N,4);vote_scale(&scales[si],i,tmp);
            for(int j=0;j<TRAIN_N;j++)vbuf[j]+=tmp[j]*scale_weights[si];free(tmp);
        }
        {cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
         /* Multi-scale dot: sum gradient dots across all 3 scales */
         for(int j=0;j<nc;j++){uint32_t id=cands[j].id;int64_t d=0;
             for(int si=0;si<3;si++){
                 d+=(int64_t)scale_weights[si]*(
                     tdot(scales[si].hg_te+(size_t)i*scales[si].pad,scales[si].hg_tr+(size_t)id*scales[si].pad,scales[si].pad)
                    +tdot(scales[si].vg_te+(size_t)i*scales[si].pad,scales[si].vg_tr+(size_t)id*scales[si].pad,scales[si].pad));}
             /* Add Bayesian prior from scale 1 */
             double bp[N_CLASSES];bay_post(scales[0].joint_te+(size_t)i*scales[0].sp,scales[0].bg,scales[0].hot,scales[0].nb,scales[0].sp,bp);
             uint8_t lbl=train_labels[id];d+=(int64_t)(50*bp[lbl]*1000);
             cands[j].score=d;}
         qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
         if(nc>0&&train_labels[cands[0].id]==test_labels[i])combined_cas++;}

        /* Scales 1+2 only */
        memset(vbuf,0,TRAIN_N*4);
        for(int si=0;si<2;si++){uint32_t *tmp=calloc(TRAIN_N,4);vote_scale(&scales[si],i,tmp);
            for(int j=0;j<TRAIN_N;j++)vbuf[j]+=tmp[j]*scale_weights[si];free(tmp);}
        {cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
         for(int j=0;j<nc;j++){uint32_t id=cands[j].id;int64_t d=0;
             for(int si=0;si<2;si++){d+=(int64_t)scale_weights[si]*(
                 tdot(scales[si].hg_te+(size_t)i*scales[si].pad,scales[si].hg_tr+(size_t)id*scales[si].pad,scales[si].pad)
                +tdot(scales[si].vg_te+(size_t)i*scales[si].pad,scales[si].vg_tr+(size_t)id*scales[si].pad,scales[si].pad));}
             double bp[N_CLASSES];bay_post(scales[0].joint_te+(size_t)i*scales[0].sp,scales[0].bg,scales[0].hot,scales[0].nb,scales[0].sp,bp);
             uint8_t lbl=train_labels[id];d+=(int64_t)(50*bp[lbl]*1000);
             cands[j].score=d;}
         qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
         if(nc>0&&train_labels[cands[0].id]==test_labels[i])s12_cas++;}

        /* Combined Bayesian (sum log-posteriors across scales) */
        {double bp_sum[N_CLASSES];memset(bp_sum,0,sizeof(bp_sum));
         for(int si=0;si<3;si++){double bp[N_CLASSES];
             bay_post(scales[si].joint_te+(size_t)i*scales[si].sp,scales[si].bg,scales[si].hot,scales[si].nb,scales[si].sp,bp);
             for(int c=0;c<N_CLASSES;c++)bp_sum[c]+=bp[c]*scale_weights[si];}
         int best=0;for(int c=1;c<N_CLASSES;c++)if(bp_sum[c]>bp_sum[best])best=c;
         if(best==test_labels[i])combined_bay++;}

        if((i+1)%1000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(vbuf);

    printf("\n=== RESULTS (all k=1) ===\n\n");
    printf("  %-45s  Bayesian  Cascade\n","Scale");
    printf("  %-45s  %5.2f%%   %5.2f%%\n","Scale 1: 32x96 (3-pixel blocks)",100.0*bay_correct[0]/TEST_N,100.0*cas_correct[0]/TEST_N);
    printf("  %-45s  %5.2f%%   %5.2f%%\n","Scale 2: 16x48 (6-pixel effective)",100.0*bay_correct[1]/TEST_N,100.0*cas_correct[1]/TEST_N);
    printf("  %-45s  %5.2f%%   %5.2f%%\n","Scale 3: 8x24 (12-pixel effective)",100.0*bay_correct[2]/TEST_N,100.0*cas_correct[2]/TEST_N);
    printf("  %-45s  %5.2f%%   %5.2f%%\n","Combined S1+S2 (3+6 pixel)",100.0*combined_bay/TEST_N,100.0*s12_cas/TEST_N);
    printf("  %-45s  %5.2f%%   %5.2f%%\n","Combined S1+S2+S3 (3+6+12 pixel)",100.0*combined_bay/TEST_N,100.0*combined_cas/TEST_N);
    printf("\n  Previous best (MT4 4-plane, single scale): 42.05%%\n");
    printf("  Delta (multi-scale S1+S2 vs single-scale): %+.2fpp\n",100.0*s12_cas/TEST_N-42.05);

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
