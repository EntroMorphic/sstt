/*
 * sstt_cifar10_stereo_stack.c — Stereoscopic 3-Eye Voting + MT4 Dot + Topo + Combined Bayesian
 *
 * Architecture:
 *   1. THREE-EYE VOTING: Build inverted indices for all 3 eyes
 *      (Fixed 85/170, Adaptive P33/P67, Per-channel adaptive).
 *      Vote across all 3 simultaneously to retrieve top-200 candidates.
 *   2. MT4 DOT RANKING: For each candidate, compute MT4 (81-level) pixel+gradient
 *      dot using planes 0 (weight 27) and 3 (weight 1).
 *   3. TOPO FEATURES: Divergence + grid divergence (3x3) on grayscale 32x32.
 *   4. COMBINED BAYESIAN PRIOR: Sum log-posteriors from all 3 eyes' hot maps.
 *   5. SCORE: 512*mt4_dot + 200*div_sim + 200*gdiv_sim + 50*combined_bayesian_prior
 *   6. k=1 argmax.
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -o sstt_cifar10_stereo_stack src/sstt_cifar10_stereo_stack.c -lm
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
#define IMG_W   96
#define IMG_H   32
#define PIXELS  3072
#define PADDED  3072
#define N_CLASSES 10
#define CLS_PAD 16
#define BLKS_PER_ROW 32
#define N_BLOCKS 1024
#define SIG_PAD  1024
#define BYTE_VALS 256
#define BG_TRANS 13
#define H_TRANS_PER_ROW 31
#define N_HTRANS (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 31
#define N_VTRANS (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD 992
#define IG_SCALE 16
#define TOP_K 200
#define MAX_REGIONS 16
#define SPATIAL_W 32
#define SPATIAL_H 32
#define GRAY_PIXELS 1024
#define GRAY_PADDED 1024
#define N_EYES 3

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*raw_gray_tr,*raw_gray_te,*train_labels,*test_labels;

/* MT4 planes (coarsest=3, finest=0) */
static int8_t *t3_tr,*t3_te,*t0_tr,*t0_te;
/* MT4 gradients */
static int8_t *hg3_tr,*hg3_te,*vg3_tr,*vg3_te;
static int8_t *hg0_tr,*hg0_te,*vg0_tr,*vg0_te;
/* Grayscale topo */
static int8_t *ghg_tr,*ghg_te,*gvg_tr,*gvg_te;
static int16_t *divneg_tr,*divneg_te;
static int16_t *gdiv_tr,*gdiv_te;

static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}

/* ================================================================
 *  Quantization functions — 3 eyes
 * ================================================================ */

/* Eye 1: Fixed 85/170 */
static void quant_fixed(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}

/* Eye 2: Per-image adaptive P33/P67 */
static void quant_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}

/* Eye 3: Per-channel adaptive (R,G,B normalized independently) */
static void quant_perchannel(const uint8_t*s,int8_t*d,int n){
    uint8_t*cv=malloc(1024);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        uint8_t p33[3],p67[3];
        for(int ch=0;ch<3;ch++){int cnt=0;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++)cv[cnt++]=si[y*96+x*3+ch];
            qsort(cv,cnt,1,cmp_u8);p33[ch]=cv[cnt/3];p67[ch]=cv[2*cnt/3];
            if(p33[ch]==p67[ch]){p33[ch]=p33[ch]>0?p33[ch]-1:0;p67[ch]=p67[ch]<255?p67[ch]+1:255;}}
        for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int ch=0;ch<3;ch++){
            int idx=y*96+x*3+ch;di[idx]=si[idx]>p67[ch]?1:si[idx]<p33[ch]?-1:0;}}free(cv);}

/* ================================================================
 *  MT4 quantization (81-level, planes 0 and 3 only)
 * ================================================================ */
static void mt4_quant_03(const uint8_t *src,int n,int8_t *p3,int8_t *p0){
    for(int img=0;img<n;img++){const uint8_t*s=src+(size_t)img*PIXELS;
        int8_t *d3=p3+(size_t)img*PADDED,*d0=p0+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++){int bin=(int)s[i]*81/256;
            d3[i]=(int8_t)(bin/27-1);d0[i]=(int8_t)(bin%3-1);}}}

static void grads(const int8_t*t,int8_t*h,int8_t*v,int n,int w,int ht,int pd){
    for(int i2=0;i2<n;i2++){const int8_t*ti=t+(size_t)i2*pd;int8_t*hi=h+(size_t)i2*pd,*vi=v+(size_t)i2*pd;
        for(int y=0;y<ht;y++){for(int x=0;x<w-1;x++)hi[y*w+x]=ct(ti[y*w+x+1]-ti[y*w+x]);hi[y*w+w-1]=0;}
        for(int y=0;y<ht-1;y++)for(int x=0;x<w;x++)vi[y*w+x]=ct(ti[(y+1)*w+x]-ti[y*w+x]);memset(vi+(ht-1)*w,0,w);}}

/* ================================================================
 *  Eye infrastructure (block sigs, transitions, joint sigs, inverted index)
 * ================================================================ */
typedef struct {
    const char *name;
    uint8_t *joint_tr,*joint_te;
    uint32_t *hot;
    uint16_t ig[N_BLOCKS];
    uint8_t bg;
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];
    uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
    uint32_t *idx_pool;
    uint8_t nbr[BYTE_VALS][8];
} eye_t;

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

static void build_eye(eye_t *e, const int8_t *tern_tr, const int8_t *tern_te) {
    /* Compute gradients */
    int8_t *ehg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*ehg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *evg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*evg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    grads(tern_tr,ehg_tr,evg_tr,TRAIN_N,IMG_W,IMG_H,PADDED);
    grads(tern_te,ehg_te,evg_te,TEST_N,IMG_W,IMG_H,PADDED);

    /* Block sigs */
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(tern_tr,pxt,TRAIN_N);bsf(tern_te,pxe,TEST_N);
    bsf(ehg_tr,hst,TRAIN_N);bsf(ehg_te,hse,TEST_N);
    bsf(evg_tr,vst,TRAIN_N);bsf(evg_te,vse,TEST_N);
    free(ehg_tr);free(ehg_te);free(evg_tr);free(evg_te);

    /* Transitions */
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte,vte,TEST_N);

    /* Joint sigs */
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(e->joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(e->joint_te,TEST_N,pxe,hse,vse,hte,vte);
    free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte);free(vtt);free(vte);

    /* BG val */
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}

    /* Hot map */
    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}

    /* IG weights */
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw[N_BLOCKS],mx=0;
     for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==e->bg)continue;
             const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}
             hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];}
     for(int k=0;k<N_BLOCKS;k++){e->ig[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!e->ig[k])e->ig[k]=1;}}

    /* Neighbor table */
    for(int v=0;v<BYTE_VALS;v++)for(int b=0;b<8;b++)e->nbr[v][b]=(uint8_t)(v^(1<<b));

    /* Inverted index */
    memset(e->idx_sz,0,sizeof(e->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc((size_t)tot*4);
    uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,e->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);

    printf("    %-25s BG=%d (%.1f%%) idx_pool=%u\n",e->name,e->bg,100.0*mc/((long)TRAIN_N*N_BLOCKS),tot);
}

/* Vote using one eye into shared vote buffer */
static void vote_eye(const eye_t*e,int img,uint32_t*vbuf){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        uint16_t w=e->ig[k],wh=w>1?w/2:1;
        {uint32_t off=e->idx_off[k][bv];uint16_t sz=e->idx_sz[k][bv];
         for(uint16_t j=0;j<sz;j++)vbuf[e->idx_pool[off+j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=e->nbr[bv][nb];if(nv==e->bg)continue;
            uint32_t no=e->idx_off[k][nv];uint16_t ns=e->idx_sz[k][nv];
            for(uint16_t j=0;j<ns;j++)vbuf[e->idx_pool[no+j]]+=wh;}}}

/* Bayesian posterior from one eye */
static void bay_eye(const eye_t*e,int img,double*bp){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}

/* ================================================================
 *  Topo features on grayscale
 * ================================================================ */
static void quant_tern_gray(const uint8_t*s,int8_t*d,int n){
    for(int i=0;i<n;i++){const uint8_t*si=s+(size_t)i*GRAY_PIXELS;int8_t*di=d+(size_t)i*GRAY_PADDED;
        for(int j=0;j<GRAY_PIXELS;j++)di[j]=si[j]>170?1:si[j]<85?-1:0;}}

static void divf(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){
    int neg=0,ny=0,nc2=0;for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){neg+=d;ny+=y;nc2++;}}
    *ns=(int16_t)(neg<-32767?-32767:neg);*cy=nc2>0?(int16_t)(ny/nc2):-1;}

static void gdivf(const int8_t*hg,const int8_t*vg,int gr,int gc,int16_t*o){
    int nr=gr*gc,reg[MAX_REGIONS];memset(reg,0,nr*sizeof(int));
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){int ry=y*gr/SPATIAL_H,rx=x*gc/SPATIAL_W;
            if(ry>=gr)ry=gr-1;if(rx>=gc)rx=gc-1;reg[ry*gc+rx]+=d;}}
    for(int i=0;i<nr;i++)o[i]=(int16_t)(reg[i]<-32767?-32767:reg[i]);}

/* ================================================================
 *  Safe ternary dot with chunked accumulation
 * ================================================================ */
static int32_t tdot(const int8_t*a,const int8_t*b,int len){
    int32_t tot=0;for(int ch=0;ch<len;ch+=64){__m256i ac=_mm256_setzero_si256();int e=ch+64;if(e>len)e=len;
        for(int i=ch;i<e;i+=32)ac=_mm256_add_epi8(ac,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
        __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(ac));
        __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(ac,1));
        __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
        __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
        s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);tot+=_mm_cvtsi128_si32(s);}return tot;}

/* ================================================================
 *  Top-K extraction
 * ================================================================ */
typedef struct{uint32_t id;uint32_t votes;int64_t score;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int cmps(const void*a,const void*b){int64_t d=((const cand_t*)b)->score-((const cand_t*)a)->score;return(d>0)-(d<0);}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc]=(cand_t){0};o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

/* ================================================================
 *  Data loading
 * ================================================================ */
static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_gray_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
        for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gd=raw_gray_tr+(size_t)idx*GRAY_PIXELS;
            for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        uint8_t*gd=raw_gray_te+(size_t)i*GRAY_PIXELS;
        for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

/* ================================================================
 *  MAIN
 * ================================================================ */
int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Stereo Stack (3-Eye Vote + MT4 Dot + Topo + Bayesian) ===\n\n");
    load_data();printf("  Loaded data (%.1f sec)\n",now_sec()-t0);

    /* ---- Build 3 eyes ---- */
    eye_t eyes[N_EYES];
    const char *names[]={"Fixed 85/170","Adaptive P33/P67","Per-channel adaptive"};
    void (*quant_fns[])(const uint8_t*,int8_t*,int)={quant_fixed,quant_adaptive,quant_perchannel};

    printf("\nBuilding %d eyes...\n",N_EYES);
    for(int ei=0;ei<N_EYES;ei++){
        eyes[ei].name=names[ei];
        int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
        quant_fns[ei](raw_train,ttr,TRAIN_N);quant_fns[ei](raw_test,tte,TEST_N);
        build_eye(&eyes[ei],ttr,tte);
        free(ttr);free(tte);
    }
    printf("  Eyes built (%.1f sec)\n",now_sec()-t0);

    /* ---- MT4 planes 0 and 3 + gradients ---- */
    printf("\nBuilding MT4 planes + gradients...\n");
    t3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    mt4_quant_03(raw_train,TRAIN_N,t3_tr,t0_tr);mt4_quant_03(raw_test,TEST_N,t3_te,t0_te);
    hg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    grads(t3_tr,hg3_tr,vg3_tr,TRAIN_N,IMG_W,IMG_H,PADDED);grads(t3_te,hg3_te,vg3_te,TEST_N,IMG_W,IMG_H,PADDED);
    grads(t0_tr,hg0_tr,vg0_tr,TRAIN_N,IMG_W,IMG_H,PADDED);grads(t0_te,hg0_te,vg0_te,TEST_N,IMG_W,IMG_H,PADDED);
    printf("  MT4 built (%.1f sec)\n",now_sec()-t0);

    /* ---- Grayscale topo features ---- */
    printf("\nBuilding grayscale topo features...\n");
    int8_t *gt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    ghg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);ghg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    gvg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gvg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    quant_tern_gray(raw_gray_tr,gt_tr,TRAIN_N);quant_tern_gray(raw_gray_te,gt_te,TEST_N);
    grads(gt_tr,ghg_tr,gvg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
    grads(gt_te,ghg_te,gvg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);

    divneg_tr=malloc((size_t)TRAIN_N*2);divneg_te=malloc((size_t)TEST_N*2);
    int16_t *divcy_tr=malloc((size_t)TRAIN_N*2),*divcy_te=malloc((size_t)TEST_N*2);
    for(int i=0;i<TRAIN_N;i++)divf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,&divneg_tr[i],&divcy_tr[i]);
    for(int i=0;i<TEST_N;i++)divf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,&divneg_te[i],&divcy_te[i]);

    gdiv_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
    for(int i=0;i<TRAIN_N;i++)gdivf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,3,3,gdiv_tr+(size_t)i*MAX_REGIONS);
    for(int i=0;i<TEST_N;i++)gdivf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,3,3,gdiv_te+(size_t)i*MAX_REGIONS);
    free(gt_tr);free(gt_te);free(divcy_tr);free(divcy_te);
    printf("  Topo built (%.1f sec)\n",now_sec()-t0);

    /* ================================================================
     *  Precompute all candidates and features
     * ================================================================ */
    printf("\nPrecomputing candidates and features for %d test images...\n",TEST_N);
    uint32_t *vbuf=calloc(TRAIN_N,4);
    cand_t *cands=malloc(TOP_K*sizeof(cand_t));
    int nreg=9;

    /* Storage for precomputed data */
    int *nc_arr=malloc(TEST_N*sizeof(int));
    uint32_t *cand_ids=malloc((size_t)TEST_N*TOP_K*sizeof(uint32_t));
    int64_t *cand_scores=malloc((size_t)TEST_N*TOP_K*sizeof(int64_t));

    for(int i=0;i<TEST_N;i++){
        /* --- 1. THREE-EYE VOTING --- */
        memset(vbuf,0,TRAIN_N*4);
        for(int ei=0;ei<N_EYES;ei++)
            vote_eye(&eyes[ei],i,vbuf);

        int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
        nc_arr[i]=nc;

        /* --- 4. COMBINED BAYESIAN PRIOR --- */
        double bay[N_CLASSES];memset(bay,0,sizeof(bay));
        for(int ei=0;ei<N_EYES;ei++)
            bay_eye(&eyes[ei],i,bay);
        /* Normalize */
        double bmx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bay[c]>bmx)bmx=bay[c];
        for(int c=0;c<N_CLASSES;c++)bay[c]-=bmx;

        /* --- Per-candidate scoring --- */
        int16_t q_dn=divneg_te[i];
        const int16_t *q_gd=gdiv_te+(size_t)i*MAX_REGIONS;

        for(int j=0;j<nc;j++){
            uint32_t id=cands[j].id;
            cand_ids[(size_t)i*TOP_K+j]=id;

            /* 2. MT4 DOT RANKING: plane3 (weight 27) + plane0 (weight 1) pixel+gradient */
            int32_t mt4_dot= 27*(tdot(t3_te+(size_t)i*PADDED,t3_tr+(size_t)id*PADDED,PADDED)
                                +tdot(hg3_te+(size_t)i*PADDED,hg3_tr+(size_t)id*PADDED,PADDED)
                                +tdot(vg3_te+(size_t)i*PADDED,vg3_tr+(size_t)id*PADDED,PADDED))
                            +   (tdot(t0_te+(size_t)i*PADDED,t0_tr+(size_t)id*PADDED,PADDED)
                                +tdot(hg0_te+(size_t)i*PADDED,hg0_tr+(size_t)id*PADDED,PADDED)
                                +tdot(vg0_te+(size_t)i*PADDED,vg0_tr+(size_t)id*PADDED,PADDED));

            /* 3. TOPO: divergence similarity */
            int16_t cdn=divneg_tr[id];
            int32_t div_sim=-(int32_t)abs(q_dn-cdn);

            /* 3. TOPO: grid divergence similarity */
            const int16_t *cg=gdiv_tr+(size_t)id*MAX_REGIONS;
            int32_t gl=0;for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);
            int32_t gdiv_sim=-gl;

            /* 4. COMBINED BAYESIAN PRIOR for this candidate's label */
            uint8_t lbl=train_labels[id];
            int32_t bay_score=(int32_t)(bay[lbl]*1000);

            /* 5. SCORE */
            int64_t score=(int64_t)512*mt4_dot
                         +(int64_t)200*div_sim
                         +(int64_t)200*gdiv_sim
                         +(int64_t)50*bay_score;
            cand_scores[(size_t)i*TOP_K+j]=score;
            cands[j].score=score;
        }

        if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"  %d/%d done\n",TEST_N,TEST_N);

    /* ================================================================
     *  Results: k=1
     * ================================================================ */
    int correct=0;
    int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
    for(int i=0;i<TEST_N;i++){
        int nc=nc_arr[i];
        /* Find best scoring candidate */
        int64_t best_s=-1LL<<62;uint32_t best_id=0;
        for(int j=0;j<nc;j++){
            if(cand_scores[(size_t)i*TOP_K+j]>best_s){best_s=cand_scores[(size_t)i*TOP_K+j];best_id=cand_ids[(size_t)i*TOP_K+j];}
        }
        pt[test_labels[i]]++;
        if(nc>0&&train_labels[best_id]==test_labels[i]){correct++;pc[test_labels[i]]++;}
    }

    printf("\n=== RESULTS ===\n\n");
    printf("  Architecture: 3-Eye Vote -> MT4 Dot -> Topo -> Combined Bayesian\n");
    printf("  Weights: 512*mt4_dot + 200*div + 200*gdiv + 50*bayesian\n");
    printf("  k=1 accuracy: %.2f%% (%d/%d)\n\n",100.0*correct/TEST_N,correct,TEST_N);

    printf("  Per-class recall:\n");
    for(int c=0;c<N_CLASSES;c++)
        printf("    %d %-12s %4d/%-4d  %.2f%%\n",c,cn[c],pc[c],pt[c],100.0*pc[c]/pt[c]);

    /* kNN sweep using precomputed scores */
    printf("\n  kNN sweep:\n");
    for(int kk=1;kk<=7;kk+=2){
        int kc=0;
        for(int i=0;i<TEST_N;i++){
            int nc=nc_arr[i];
            /* Sort candidates by score */
            cand_t cs[TOP_K];
            for(int j=0;j<nc;j++){cs[j].id=cand_ids[(size_t)i*TOP_K+j];cs[j].score=cand_scores[(size_t)i*TOP_K+j];}
            qsort(cs,(size_t)nc,sizeof(cand_t),cmps);
            int v[N_CLASSES]={0};int ek=kk<nc?kk:nc;
            for(int j=0;j<ek;j++)v[train_labels[cs[j].id]]++;
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
            if(pred==test_labels[i])kc++;
        }
        printf("    k=%d: %.2f%% (%d/%d)\n",kk,100.0*kc/TEST_N,kc,TEST_N);
    }

    /* Baselines */
    printf("\n  Baselines:\n");
    printf("    Bayesian (single eye):      36.58%%\n");
    printf("    MT4 px+grad k=1 (stack):    42.05%%\n");
    printf("    Stereo 3-eye Bayesian:      40.20%%\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    free(vbuf);free(cands);free(nc_arr);free(cand_ids);free(cand_scores);
    return 0;
}
