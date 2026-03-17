/*
 * sstt_cifar10_rawdot.c — Raw Pixel Dot Product (MT-infinity limit)
 *
 * Tests the ceiling for Multi-Trit quantization improvement.
 * Uses raw uint8 pixel values for dot product ranking (no quantization).
 * Voting still uses ternary inverted index.
 *
 * If raw dot >> MT4 dot: more trits help, MT18 worth implementing.
 * If raw dot ≈ MT4 dot: quantization isn't the bottleneck, stop here.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_rawdot src/sstt_cifar10_rawdot.c -lm
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
#define SPATIAL_W 32
#define SPATIAL_H 32
#define GRAY_PIXELS 1024
#define GRAY_PADDED 1024
#define MAX_REGIONS 16

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*raw_gray_tr,*raw_gray_te,*train_labels,*test_labels;
static int8_t *t3_tr,*t3_te,*t0_tr,*t0_te;
static int8_t *hg3_tr,*hg3_te,*vg3_tr,*vg3_te,*hg0_tr,*hg0_te,*vg0_tr,*vg0_te;
static int8_t *tern_tr,*tern_te;
static int8_t *ghg_tr,*ghg_te,*gvg_tr,*gvg_te;
static int16_t *divneg_tr,*divneg_te,*divcy_tr,*divcy_te,*gdiv_tr,*gdiv_te;
static uint8_t *joint_tr,*joint_te;
static uint32_t *joint_hot;
static uint16_t ig_w[N_BLOCKS];
static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t bg_val;

/* Core functions — same as all other experiments */
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void mt4q(const uint8_t*s,int n,int8_t*p3,int8_t*p0){for(int img=0;img<n;img++){
    const uint8_t*si=s+(size_t)img*PIXELS;int8_t*d3=p3+(size_t)img*PADDED,*d0=p0+(size_t)img*PADDED;
    for(int i=0;i<PIXELS;i++){int bin=(int)si[i]*81/256;d3[i]=(int8_t)(bin/27-1);d0[i]=(int8_t)(bin%3-1);}}}
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
static void divf(const int8_t*hg,const int8_t*vg,int16_t*ns,int16_t*cy){
    int neg=0,ny=0,nc2=0;for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){neg+=d;ny+=y;nc2++;}}
    *ns=(int16_t)(neg<-32767?-32767:neg);*cy=nc2>0?(int16_t)(ny/nc2):-1;}
static void gdf(const int8_t*hg,const int8_t*vg,int grow,int gcol,int16_t*o){
    int nr=grow*gcol,reg[MAX_REGIONS];memset(reg,0,nr*sizeof(int));
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){int ry=y*grow/SPATIAL_H,rx=x*gcol/SPATIAL_W;
            if(ry>=grow)ry=grow-1;if(rx>=gcol)rx=gcol-1;reg[ry*gcol+rx]+=d;}}
    for(int i=0;i<nr;i++)o[i]=(int16_t)(reg[i]<-32767?-32767:reg[i]);}
static int32_t tdot(const int8_t*a,const int8_t*b,int len){int32_t tot=0;
    for(int ch=0;ch<len;ch+=64){__m256i ac=_mm256_setzero_si256();int e=ch+64;if(e>len)e=len;
        for(int i=ch;i<e;i+=32)ac=_mm256_add_epi8(ac,_mm256_sign_epi8(
            _mm256_load_si256((const __m256i*)(a+i)),_mm256_load_si256((const __m256i*)(b+i))));
        __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(ac));
        __m256i hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(ac,1));
        __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
        __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
        s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);tot+=_mm_cvtsi128_si32(s);}return tot;}
static void bay_post(const uint8_t*sig,double*lp){for(int c=0;c<N_CLASSES;c++)lp[c]=0.0;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg_val)continue;
        const uint32_t*h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)lp[c]+=log(h[c]+0.5);}
    double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(lp[c]>mx)mx=lp[c];
    for(int c=0;c<N_CLASSES;c++)lp[c]-=mx;}
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

/* ================================================================
 *  RAW uint8 dot product — the MT-infinity limit
 * ================================================================ */
static int64_t raw_dot(const uint8_t *a, const uint8_t *b, int len) {
    int64_t total = 0;
    /* Scalar — simple and correct. AVX2 optimization possible but not needed for this test. */
    for (int i = 0; i < len; i++)
        total += (int64_t)a[i] * b[i];
    return total;
}

/* Raw uint8 L2 distance (negative, for sorting — higher = more similar) */
static int64_t raw_neg_l2(const uint8_t *a, const uint8_t *b, int len) {
    int64_t total = 0;
    for (int i = 0; i < len; i++) {
        int d = (int)a[i] - (int)b[i];
        total -= (int64_t)d * d;
    }
    return total;
}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_gray_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gdd=raw_gray_tr+(size_t)idx*GRAY_PIXELS;
            for(int p2=0;p2<1024;p2++)gdd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        uint8_t*gdd=raw_gray_te+(size_t)i*GRAY_PIXELS;
        for(int p2=0;p2<1024;p2++)gdd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Raw Dot (MT-infinity ceiling) ===\n\n");
    load_data();

    /* Build everything for voting + topo */
    t3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    t0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);t0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    mt4q(raw_train,TRAIN_N,t3_tr,t0_tr);mt4q(raw_test,TEST_N,t3_te,t0_te);
    hg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg3_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg3_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg0_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg0_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    gr(t3_tr,hg3_tr,vg3_tr,TRAIN_N,IMG_W,IMG_H,PADDED);gr(t3_te,hg3_te,vg3_te,TEST_N,IMG_W,IMG_H,PADDED);
    gr(t0_tr,hg0_tr,vg0_tr,TRAIN_N,IMG_W,IMG_H,PADDED);gr(t0_te,hg0_te,vg0_te,TEST_N,IMG_W,IMG_H,PADDED);
    tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    qt(raw_train,tern_tr,TRAIN_N,PIXELS,PADDED);qt(raw_test,tern_te,TEST_N,PIXELS,PADDED);
    int8_t *hgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*hgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *vgc_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vgc_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    gr(tern_tr,hgc_tr,vgc_tr,TRAIN_N,IMG_W,IMG_H,PADDED);gr(tern_te,hgc_te,vgc_te,TEST_N,IMG_W,IMG_H,PADDED);
    {int8_t*gt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     ghg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);ghg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     gvg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gvg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     qt(raw_gray_tr,gt_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);qt(raw_gray_te,gt_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
     gr(gt_tr,ghg_tr,gvg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);gr(gt_te,ghg_te,gvg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
     divneg_tr=malloc((size_t)TRAIN_N*2);divneg_te=malloc((size_t)TEST_N*2);divcy_tr=malloc((size_t)TRAIN_N*2);divcy_te=malloc((size_t)TEST_N*2);
     for(int i=0;i<TRAIN_N;i++)divf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,&divneg_tr[i],&divcy_tr[i]);
     for(int i=0;i<TEST_N;i++)divf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,&divneg_te[i],&divcy_te[i]);
     gdiv_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
     for(int i=0;i<TRAIN_N;i++)gdf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,3,3,gdiv_tr+(size_t)i*MAX_REGIONS);
     for(int i=0;i<TEST_N;i++)gdf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,3,3,gdiv_te+(size_t)i*MAX_REGIONS);
     free(gt_tr);free(gt_te);}
    /* Voting infra — same as always */
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(tern_tr,pxt,TRAIN_N);bsf(tern_te,pxe,TEST_N);bsf(hgc_tr,hst,TRAIN_N);bsf(hgc_te,hse,TEST_N);
    bsf(vgc_tr,vst,TRAIN_N);bsf(vgc_te,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte2,vte2,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(joint_te,TEST_N,pxe,hse,vse,hte2,vte2);
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
     * Test 5 dot product variants: ternary, MT4, raw dot, raw L2, raw+topo+bayes
     * ================================================================ */
    printf("Running 5 dot variants + full stack...\n\n");
    uint32_t *vbuf=calloc(TRAIN_N,4);
    int nreg=9;

    int c_tern=0, c_mt4=0, c_rawdot=0, c_rawl2=0, c_raw_stack=0, c_mt4_stack=0;

    for(int i=0;i<TEST_N;i++){
        memset(vbuf,0,TRAIN_N*4);const uint8_t*sig=joint_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg_val)continue;
            uint16_t w=ig_w[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)vbuf[idx_pool[off+j]]+=w;}
            for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg_val)continue;
                uint32_t no=idx_off[k][nv];uint16_t ns=idx_sz[k][nv];for(uint16_t j=0;j<ns;j++)vbuf[idx_pool[no+j]]+=wh;}}

        cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);

        /* Bayesian + topo features (precompute once) */
        double bp[N_CLASSES];bay_post(sig,bp);
        int16_t q_dn=divneg_te[i],q_dc=divcy_te[i];
        const int16_t*q_gd=gdiv_te+(size_t)i*MAX_REGIONS;

        /* 1: Ternary pixel dot (3-level) */
        for(int j=0;j<nc;j++)cands[j].score=tdot(tern_te+(size_t)i*PADDED,tern_tr+(size_t)cands[j].id*PADDED,PADDED);
        qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
        if(nc>0&&train_labels[cands[0].id]==test_labels[i])c_tern++;

        /* 2: MT4 pixel+grad dot (81-level) */
        for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
            cands[j].score=27*(tdot(t3_te+(size_t)i*PADDED,t3_tr+(size_t)id*PADDED,PADDED)
                              +tdot(hg3_te+(size_t)i*PADDED,hg3_tr+(size_t)id*PADDED,PADDED)
                              +tdot(vg3_te+(size_t)i*PADDED,vg3_tr+(size_t)id*PADDED,PADDED))
                           +  (tdot(t0_te+(size_t)i*PADDED,t0_tr+(size_t)id*PADDED,PADDED)
                              +tdot(hg0_te+(size_t)i*PADDED,hg0_tr+(size_t)id*PADDED,PADDED)
                              +tdot(vg0_te+(size_t)i*PADDED,vg0_tr+(size_t)id*PADDED,PADDED));}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
        if(nc>0&&train_labels[cands[0].id]==test_labels[i])c_mt4++;

        /* 3: Raw uint8 dot product (MT-infinity) */
        for(int j=0;j<nc;j++)
            cands[j].score=raw_dot(raw_test+(size_t)i*PIXELS,raw_train+(size_t)cands[j].id*PIXELS,PIXELS);
        qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
        if(nc>0&&train_labels[cands[0].id]==test_labels[i])c_rawdot++;

        /* 4: Raw uint8 negative L2 distance */
        for(int j=0;j<nc;j++)
            cands[j].score=raw_neg_l2(raw_test+(size_t)i*PIXELS,raw_train+(size_t)cands[j].id*PIXELS,PIXELS);
        qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
        if(nc>0&&train_labels[cands[0].id]==test_labels[i])c_rawl2++;

        /* 5: Raw dot + topo + Bayesian (full stack with raw dot) */
        for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
            int64_t rd=raw_dot(raw_test+(size_t)i*PIXELS,raw_train+(size_t)id*PIXELS,PIXELS);
            int16_t cdn=divneg_tr[id],cdc=divcy_tr[id];
            int32_t ds=-(int32_t)abs(q_dn-cdn);if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;
            else if((q_dc<0)!=(cdc<0))ds-=10;
            const int16_t*cg=gdiv_tr+(size_t)id*MAX_REGIONS;int32_t gl=0;
            for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);
            uint8_t lbl=train_labels[id];
            cands[j].score=rd+(int64_t)200*ds+(int64_t)200*(-gl)+(int64_t)50*(int64_t)(bp[lbl]*1000);}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
        if(nc>0&&train_labels[cands[0].id]==test_labels[i])c_raw_stack++;

        /* 6: MT4 full stack (for comparison) */
        for(int j=0;j<nc;j++){uint32_t id=cands[j].id;
            int32_t mt4d=27*(tdot(t3_te+(size_t)i*PADDED,t3_tr+(size_t)id*PADDED,PADDED)
                             +tdot(hg3_te+(size_t)i*PADDED,hg3_tr+(size_t)id*PADDED,PADDED)
                             +tdot(vg3_te+(size_t)i*PADDED,vg3_tr+(size_t)id*PADDED,PADDED))
                         +  (tdot(t0_te+(size_t)i*PADDED,t0_tr+(size_t)id*PADDED,PADDED)
                             +tdot(hg0_te+(size_t)i*PADDED,hg0_tr+(size_t)id*PADDED,PADDED)
                             +tdot(vg0_te+(size_t)i*PADDED,vg0_tr+(size_t)id*PADDED,PADDED));
            int16_t cdn=divneg_tr[id],cdc=divcy_tr[id];
            int32_t ds=-(int32_t)abs(q_dn-cdn);if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;
            else if((q_dc<0)!=(cdc<0))ds-=10;
            const int16_t*cg=gdiv_tr+(size_t)id*MAX_REGIONS;int32_t gl=0;
            for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);
            uint8_t lbl=train_labels[id];
            cands[j].score=(int64_t)512*mt4d+(int64_t)200*ds+(int64_t)200*(-gl)+(int64_t)50*(int64_t)(bp[lbl]*1000);}
        qsort(cands,(size_t)nc,sizeof(cand_t),cmps);
        if(nc>0&&train_labels[cands[0].id]==test_labels[i])c_mt4_stack++;

        if((i+1)%1000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(vbuf);

    printf("=== RESULTS (all k=1) ===\n\n");
    printf("  %-40s  Accuracy\n","Method");
    printf("  %-40s  %5.2f%%\n","1. Ternary pixel dot (3-level)",100.0*c_tern/TEST_N);
    printf("  %-40s  %5.2f%%\n","2. MT4 pixel+grad dot (81-level)",100.0*c_mt4/TEST_N);
    printf("  %-40s  %5.2f%%\n","3. Raw uint8 dot (MT-infinity)",100.0*c_rawdot/TEST_N);
    printf("  %-40s  %5.2f%%\n","4. Raw uint8 L2 distance",100.0*c_rawl2/TEST_N);
    printf("  %-40s  %5.2f%%\n","5. Raw dot + topo + Bayesian",100.0*c_raw_stack/TEST_N);
    printf("  %-40s  %5.2f%%\n","6. MT4 + topo + Bayesian (current best)",100.0*c_mt4_stack/TEST_N);
    printf("\n  Delta (raw vs MT4, dot only): %+.2fpp\n",100.0*(c_rawdot-c_mt4)/TEST_N);
    printf("  Delta (raw vs MT4, full stack): %+.2fpp\n",100.0*(c_raw_stack-c_mt4_stack)/TEST_N);
    printf("\n  If raw >> MT4: more trits help, MT18 worth pursuing.\n");
    printf("  If raw ~= MT4: quantization not the bottleneck.\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
