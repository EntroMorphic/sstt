/*
 * sstt_cifar10_modec.c — Mode C Diagnostic: Why kNN Dilution Dominates
 *
 * The autopsy showed 58.9% of errors are Mode C: correct class in top-7
 * but outvoted. This diagnostic measures WHY:
 *
 *   1. Where does the correct class rank after scoring? (rank 1? 3? 7?)
 *   2. Do correct-class candidates have higher scores than wrong-class?
 *   3. Would score-weighted voting fix it?
 *   4. Would larger k help or hurt?
 *   5. What about Kalman spline: model per-class score curves?
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_modec src/sstt_cifar10_modec.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Same constants as cifar10_full */
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
#define BG_GRAD 13
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

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*raw_gray_tr,*raw_gray_te,*train_labels,*test_labels;
static int8_t *tern_tr,*tern_te,*hg_tr,*hg_te,*vg_tr,*vg_te;
static int8_t *ghg_tr,*ghg_te,*gvg_tr,*gvg_te;
static uint8_t *joint_tr,*joint_te;
static uint32_t *joint_hot;
static uint16_t ig_w[N_BLOCKS];
static uint8_t nbr[BYTE_VALS][8];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;
static uint8_t bg_val;
static int16_t *divneg_tr,*divneg_te,*divneg_cy_tr,*divneg_cy_te;
static int16_t *gdiv_tr,*gdiv_te;

/* Minimal implementations — same as cifar10_full */
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void quant(const uint8_t*s,int8_t*d,int n,int px,int pd){
    const __m256i bi=_mm256_set1_epi8((char)0x80),th=_mm256_set1_epi8((char)(170^0x80)),
                 tl=_mm256_set1_epi8((char)(85^0x80)),on=_mm256_set1_epi8(1);
    for(int i2=0;i2<n;i2++){const uint8_t*si=s+(size_t)i2*px;int8_t*di=d+(size_t)i2*pd;
        int i;for(i=0;i+32<=px;i+=32){__m256i p=_mm256_loadu_si256((const __m256i*)(si+i));
            __m256i sp=_mm256_xor_si256(p,bi);
            _mm256_storeu_si256((__m256i*)(di+i),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,th),on),
                _mm256_and_si256(_mm256_cmpgt_epi8(tl,sp),on)));}
        for(;i<px;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;memset(di+px,0,pd-px);}}
static void grads(const int8_t*t,int8_t*h,int8_t*v,int n,int w,int ht2,int pd){
    for(int i2=0;i2<n;i2++){const int8_t*ti=t+(size_t)i2*pd;int8_t*hi=h+(size_t)i2*pd,*vi=v+(size_t)i2*pd;
        for(int y=0;y<ht2;y++){for(int x=0;x<w-1;x++)hi[y*w+x]=ct(ti[y*w+x+1]-ti[y*w+x]);hi[y*w+w-1]=0;}
        for(int y=0;y<ht2-1;y++)for(int x=0;x<w;x++)vi[y*w+x]=ct(ti[(y+1)*w+x]-ti[y*w+x]);
        memset(vi+(ht2-1)*w,0,w);}}
static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void bsigs(const int8_t*d,uint8_t*s,int n){
    for(int i=0;i<n;i++){const int8_t*im=d+(size_t)i*PADDED;uint8_t*si=s+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s2=0;s2<BLKS_PER_ROW;s2++){int b2=y*IMG_W+s2*3;
            si[y*BLKS_PER_ROW+s2]=be(im[b2],im[b2+1],im[b2+2]);}}}
static inline uint8_t te(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,
    b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}
static void trans(const uint8_t*bs,uint8_t*ht2,uint8_t*vt2,int n){
    for(int i=0;i<n;i++){const uint8_t*s=bs+(size_t)i*SIG_PAD;uint8_t*h=ht2+(size_t)i*TRANS_PAD,*v=vt2+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static void jsigs(uint8_t*o,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht2,const uint8_t*vt2){
    for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht2+(size_t)i*TRANS_PAD,*vti=vt2+(size_t)i*TRANS_PAD;uint8_t*oi=o+(size_t)i*SIG_PAD;
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
static void gdiv(const int8_t*hg,const int8_t*vg,int gr,int gc,int16_t*o){
    int nr=gr*gc,reg[MAX_REGIONS];memset(reg,0,nr*sizeof(int));
    for(int y=0;y<SPATIAL_H;y++)for(int x=0;x<SPATIAL_W;x++){
        int dh=(int)hg[y*SPATIAL_W+x]-(x>0?(int)hg[y*SPATIAL_W+x-1]:0);
        int dv=(int)vg[y*SPATIAL_W+x]-(y>0?(int)vg[(y-1)*SPATIAL_W+x]:0);
        int d=dh+dv;if(d<0){int ry=y*gr/SPATIAL_H,rx=x*gc/SPATIAL_W;
            if(ry>=gr)ry=gr-1;if(rx>=gc)rx=gc-1;reg[ry*gc+rx]+=d;}}
    for(int i=0;i<nr;i++)o[i]=(int16_t)(reg[i]<-32767?-32767:reg[i]);}

static int32_t tdot(const int8_t*a,const int8_t*b,int len){
    int32_t tot=0;for(int ch=0;ch<len;ch+=64){__m256i ac=_mm256_setzero_si256();
        int e=ch+64;if(e>len)e=len;
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
static int *gh=NULL;static size_t ghc=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){
    uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghc){ghc=(size_t)(mx+1)+4096;free(gh);gh=malloc(ghc*sizeof(int));}
    memset(gh,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])gh[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=gh[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc]=(cand_t){0};o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*GRAY_PIXELS);raw_gray_te=malloc((size_t)TEST_N*GRAY_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){fread(rec,1,3073,f);int idx=(b2-1)*10000+i;
            train_labels[idx]=rec[0];uint8_t*d=raw_train+(size_t)idx*PIXELS;
            const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gd=raw_gray_tr+(size_t)idx*GRAY_PIXELS;
            for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){fread(rec,1,3073,f);
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;
        const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        uint8_t*gd=raw_gray_te+(size_t)i*GRAY_PIXELS;
        for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);
    printf("  Loaded\n");}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== Mode C Diagnostic: Where Does kNN Dilution Come From? ===\n\n");
    load_data();

    /* Features */
    tern_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);tern_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    quant(raw_train,tern_tr,TRAIN_N,PIXELS,PADDED);quant(raw_test,tern_te,TEST_N,PIXELS,PADDED);
    grads(tern_tr,hg_tr,vg_tr,TRAIN_N,IMG_W,IMG_H,PADDED);grads(tern_te,hg_te,vg_te,TEST_N,IMG_W,IMG_H,PADDED);
    ghg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);ghg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    gvg_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED);gvg_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
    {int8_t*gt_tr=aligned_alloc(32,(size_t)TRAIN_N*GRAY_PADDED),*gt_te=aligned_alloc(32,(size_t)TEST_N*GRAY_PADDED);
     quant(raw_gray_tr,gt_tr,TRAIN_N,GRAY_PIXELS,GRAY_PADDED);quant(raw_gray_te,gt_te,TEST_N,GRAY_PIXELS,GRAY_PADDED);
     grads(gt_tr,ghg_tr,gvg_tr,TRAIN_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);
     grads(gt_te,ghg_te,gvg_te,TEST_N,SPATIAL_W,SPATIAL_H,GRAY_PADDED);free(gt_tr);free(gt_te);}

    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsigs(tern_tr,pxt,TRAIN_N);bsigs(tern_te,pxe,TEST_N);
    bsigs(hg_tr,hst,TRAIN_N);bsigs(hg_te,hse,TEST_N);
    bsigs(vg_tr,vst,TRAIN_N);bsigs(vg_te,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte2=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trans(pxt,htt,vtt,TRAIN_N);trans(pxe,hte2,vte2,TEST_N);
    joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsigs(joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsigs(joint_te,TEST_N,pxe,hse,vse,hte2,vte2);

    divneg_tr=malloc((size_t)TRAIN_N*2);divneg_te=malloc((size_t)TEST_N*2);
    divneg_cy_tr=malloc((size_t)TRAIN_N*2);divneg_cy_te=malloc((size_t)TEST_N*2);
    for(int i=0;i<TRAIN_N;i++)divf(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,&divneg_tr[i],&divneg_cy_tr[i]);
    for(int i=0;i<TEST_N;i++)divf(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,&divneg_te[i],&divneg_cy_te[i]);
    gdiv_tr=malloc((size_t)TRAIN_N*MAX_REGIONS*2);gdiv_te=malloc((size_t)TEST_N*MAX_REGIONS*2);
    for(int i=0;i<TRAIN_N;i++)gdiv(ghg_tr+(size_t)i*GRAY_PADDED,gvg_tr+(size_t)i*GRAY_PADDED,3,3,gdiv_tr+(size_t)i*MAX_REGIONS);
    for(int i=0;i<TEST_N;i++)gdiv(ghg_te+(size_t)i*GRAY_PADDED,gvg_te+(size_t)i*GRAY_PADDED,3,3,gdiv_te+(size_t)i*MAX_REGIONS);

    /* Index */
    {long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}bg_val=0;long mc=0;
        for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];bg_val=(uint8_t)v;}}

    joint_hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(joint_hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)joint_hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw[N_BLOCKS],mx=0;
     for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==bg_val)continue;
             const uint32_t*h=joint_hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;
             double pv=(double)vt/TRAIN_N,hv=0;for(int c=0;c<N_CLASSES;c++){double pc=(double)h[c]/vt;if(pc>0)hv-=pc*log2(pc);}
             hcond+=pv*hv;}raw[k]=hc-hcond;if(raw[k]>mx)mx=raw[k];}
     for(int k=0;k<N_BLOCKS;k++){ig_w[k]=mx>0?(uint16_t)(raw[k]/mx*IG_SCALE+0.5):1;if(!ig_w[k])ig_w[k]=1;}}
    for(int v=0;v<BYTE_VALS;v++)for(int b2=0;b2<8;b2++)nbr[v][b2]=(uint8_t)(v^(1<<b2));
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
     * DETAILED MODE C ANALYSIS
     * ================================================================ */
    printf("Precomputing + analyzing...\n");
    uint32_t *vbuf=calloc(TRAIN_N,4);
    int nreg=9;

    /* Accumulators */
    int total_correct=0, mode_a=0, mode_b=0, mode_c=0;
    int correct_rank1_after_sort=0;  /* correct class is rank-1 after scoring */
    int correct_in_top3=0, correct_in_top7=0, correct_in_top15=0;

    /* For Mode C images: where does correct class rank? */
    int modec_correct_is_rank[TOP_K+1]; memset(modec_correct_is_rank,0,sizeof(modec_correct_is_rank));

    /* Score gap analysis: for Mode C, what's the score gap? */
    double modec_score_ratio_sum=0; int modec_count=0;

    /* kNN sweep for all k values */
    int knn_correct[51]; memset(knn_correct,0,sizeof(knn_correct)); /* k=1..50 */

    /* Score-weighted vote */
    int weighted_correct=0;

    /* Kalman accumulation with various S */
    int kalman_correct[6]; memset(kalman_correct,0,sizeof(kalman_correct));
    int kalman_S[]={1,2,5,10,20,50};

    /* Spline: per-class score area under curve */
    int spline_correct=0;

    for(int i=0;i<TEST_N;i++){
        /* Vote */
        memset(vbuf,0,TRAIN_N*4);const uint8_t*sig=joint_te+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==bg_val)continue;
            uint16_t w=ig_w[k],wh=w>1?w/2:1;
            {uint32_t off=idx_off[k][bv];uint16_t sz=idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)vbuf[idx_pool[off+j]]+=w;}
            for(int nb=0;nb<8;nb++){uint8_t nv=nbr[bv][nb];if(nv==bg_val)continue;
                uint32_t no=idx_off[k][nv];uint16_t ns=idx_sz[k][nv];for(uint16_t j=0;j<ns;j++)vbuf[idx_pool[no+j]]+=wh;}}

        cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
        int true_lbl=test_labels[i];

        /* Compute scores: gradient dot + topo */
        const int8_t *qh=hg_te+(size_t)i*PADDED,*qv=vg_te+(size_t)i*PADDED;
        int16_t q_dn=divneg_te[i],q_dc=divneg_cy_te[i];
        const int16_t*q_gd=gdiv_te+(size_t)i*MAX_REGIONS;
        for(int j=0;j<nc;j++){
            uint32_t id=cands[j].id;
            int32_t dh=tdot(qh,hg_tr+(size_t)id*PADDED,PADDED);
            int32_t dv=tdot(qv,vg_tr+(size_t)id*PADDED,PADDED);
            int16_t cdn=divneg_tr[id],cdc=divneg_cy_tr[id];
            int32_t ds=-(int32_t)abs(q_dn-cdn);
            if(q_dc>=0&&cdc>=0)ds-=(int32_t)abs(q_dc-cdc)*2;
            else if((q_dc<0)!=(cdc<0))ds-=10;
            const int16_t*cg=gdiv_tr+(size_t)id*MAX_REGIONS;
            int32_t gl=0;for(int r=0;r<nreg;r++)gl+=abs(q_gd[r]-cg[r]);
            cands[j].score=(int64_t)256*dh+(int64_t)256*dv+(int64_t)200*ds+(int64_t)200*(-gl);
        }
        qsort(cands,(size_t)nc,sizeof(cand_t),cmps);

        /* Check recall */
        int in_topk=0,first_correct_rank=-1;
        for(int j=0;j<nc;j++)if(train_labels[cands[j].id]==true_lbl){in_topk=1;if(first_correct_rank<0)first_correct_rank=j;break;}
        /* Actually scan all */
        for(int j=0;j<nc;j++)if(train_labels[cands[j].id]==true_lbl){in_topk=1;if(first_correct_rank<0)first_correct_rank=j;}

        if(first_correct_rank==0)correct_rank1_after_sort++;
        if(first_correct_rank>=0&&first_correct_rank<3)correct_in_top3++;
        if(first_correct_rank>=0&&first_correct_rank<7)correct_in_top7++;
        if(first_correct_rank>=0&&first_correct_rank<15)correct_in_top15++;

        /* Standard kNN for various k */
        for(int kk=1;kk<=50&&kk<=nc;kk++){
            int vv[N_CLASSES]={0};for(int j=0;j<kk;j++)vv[train_labels[cands[j].id]]++;
            int best=0;for(int c=1;c<N_CLASSES;c++)if(vv[c]>vv[best])best=c;
            if(best==true_lbl)knn_correct[kk]++;
        }

        /* Score-weighted vote (top-50) */
        {double sv[N_CLASSES];memset(sv,0,sizeof(sv));
         int kk=50<nc?50:nc;
         /* Shift scores to be positive */
         int64_t min_score=cands[0].score;for(int j=0;j<kk;j++)if(cands[j].score<min_score)min_score=cands[j].score;
         for(int j=0;j<kk;j++)sv[train_labels[cands[j].id]]+=(double)(cands[j].score-min_score+1);
         int best=0;for(int c=1;c<N_CLASSES;c++)if(sv[c]>sv[best])best=c;
         if(best==true_lbl)weighted_correct++;}

        /* Kalman accumulation with various S */
        for(int si=0;si<6;si++){
            int64_t state[N_CLASSES];memset(state,0,sizeof(state));
            int kk=30<nc?30:nc;int S=kalman_S[si];
            for(int j=0;j<kk;j++){
                uint8_t lbl=train_labels[cands[j].id];
                state[lbl]+=cands[j].score*S/(S+j);
            }
            int best=0;for(int c=1;c<N_CLASSES;c++)if(state[c]>state[best])best=c;
            if(best==true_lbl)kalman_correct[si]++;
        }

        /* Spline: per-class area under score curve (top-50) */
        {double area[N_CLASSES];memset(area,0,sizeof(area));
         int kk=50<nc?50:nc;
         for(int j=0;j<kk;j++){
             double w=1.0/(j+1); /* harmonic weight by rank */
             area[train_labels[cands[j].id]]+=w*(double)(cands[j].score>0?cands[j].score:0);
         }
         int best=0;for(int c=1;c<N_CLASSES;c++)if(area[c]>area[best])best=c;
         if(best==true_lbl)spline_correct++;}

        /* Mode classification (k=7) */
        {int vv[N_CLASSES]={0};int kk=7<nc?7:nc;for(int j=0;j<kk;j++)vv[train_labels[cands[j].id]]++;
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(vv[c]>vv[pred])pred=c;
         if(pred==true_lbl){total_correct++;}
         else if(!in_topk){mode_a++;}
         else{int in7=0;for(int j=0;j<kk;j++)if(train_labels[cands[j].id]==true_lbl){in7=1;break;}
              if(!in7)mode_b++;else{mode_c++;
                  if(first_correct_rank>=0&&first_correct_rank<TOP_K)modec_correct_is_rank[first_correct_rank]++;
                  if(nc>0&&cands[0].score!=0){modec_score_ratio_sum+=(first_correct_rank>=0&&first_correct_rank<nc)?
                      (double)cands[first_correct_rank].score/(double)cands[0].score:0;modec_count++;}}}}

        if((i+1)%2000==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    fprintf(stderr,"\n");free(vbuf);

    printf("\n=== AUTOPSY ===\n");
    int errors=TEST_N-total_correct;
    printf("  Accuracy (k=7): %.2f%% (%d errors)\n",100.0*total_correct/TEST_N,errors);
    printf("  Mode A (vote miss): %d (%.1f%%)\n",mode_a,100.0*mode_a/errors);
    printf("  Mode B (rank miss): %d (%.1f%%)\n",mode_b,100.0*mode_b/errors);
    printf("  Mode C (kNN dilution): %d (%.1f%%)\n\n",mode_c,100.0*mode_c/errors);

    printf("=== RANKING QUALITY ===\n");
    printf("  Correct class rank-1 after scoring: %.2f%%\n",100.0*correct_rank1_after_sort/TEST_N);
    printf("  Correct class in top-3:  %.2f%%\n",100.0*correct_in_top3/TEST_N);
    printf("  Correct class in top-7:  %.2f%%\n",100.0*correct_in_top7/TEST_N);
    printf("  Correct class in top-15: %.2f%%\n\n",100.0*correct_in_top15/TEST_N);

    printf("  For Mode C errors, first correct-class candidate rank:\n");
    for(int r=0;r<7;r++)printf("    rank %d: %d (%.1f%% of Mode C)\n",r,modec_correct_is_rank[r],100.0*modec_correct_is_rank[r]/mode_c);
    int rank7plus=0;for(int r=7;r<=TOP_K;r++)rank7plus+=modec_correct_is_rank[r];
    printf("    rank 7+: %d (should be 0)\n",rank7plus);
    if(modec_count>0)printf("  Avg score ratio (correct/top-1): %.4f\n",modec_score_ratio_sum/modec_count);

    printf("\n=== kNN SWEEP (k=1..20) ===\n");
    for(int k=1;k<=20;k++)printf("  k=%-3d %.2f%%\n",k,100.0*knn_correct[k]/TEST_N);

    printf("\n=== AGGREGATION METHODS ===\n");
    printf("  kNN k=7 (baseline):  %.2f%%\n",100.0*total_correct/TEST_N);
    printf("  Score-weighted (top-50): %.2f%%\n",100.0*weighted_correct/TEST_N);
    printf("  Spline (harmonic-weighted area): %.2f%%\n",100.0*spline_correct/TEST_N);
    for(int si=0;si<6;si++)
        printf("  Kalman S=%d K=30: %.2f%%\n",kalman_S[si],100.0*kalman_correct[si]/TEST_N);

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
