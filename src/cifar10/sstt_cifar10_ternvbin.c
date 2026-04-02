/*
 * sstt_cifar10_ternvbin.c — Ternary vs Binary Quantization
 *
 * Ternary: {-1, 0, +1} — 3 levels, 27 block values, "I don't know" middle
 * Binary:  {-1, +1}     — 2 levels, 8 block values, every pixel commits
 * Pentary: {-2,-1,0,+1,+2} — 5 levels, 125 block values, more resolution
 *
 * Tests: which quantization level is optimal for CIFAR-10?
 * Also: stereoscopic combinations (ternary eye + binary eye)
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_ternvbin src/sstt_cifar10_ternvbin.c -lm
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

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

/* ================================================================
 *  Quantization functions
 * ================================================================ */

/* Binary: fixed median 128 */
static void quant_binary_fixed(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>=128?1:-1;}}

/* Binary: adaptive median per image */
static void quant_binary_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);uint8_t med=sorted[PIXELS/2];
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>med?1:si[i]<med?-1:0;}free(sorted);}
    /* Note: median split can produce zeros when pixel==median */

/* Ternary: fixed 85/170 */
static void quant_ternary_fixed(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}

/* Ternary: adaptive P33/P67 */
static void quant_ternary_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}

/* Pentary: 5 levels {-2,-1,0,+1,+2} — adaptive quintiles */
static void quant_pentary_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p20=sorted[PIXELS/5],p40=sorted[2*PIXELS/5],p60=sorted[3*PIXELS/5],p80=sorted[4*PIXELS/5];
        for(int i=0;i<PIXELS;i++){
            if(si[i]>p80)di[i]=2;else if(si[i]>p60)di[i]=1;else if(si[i]>p40)di[i]=0;
            else if(si[i]>p20)di[i]=-1;else di[i]=-2;}}free(sorted);}

/* ================================================================
 *  Block encoding — must handle different value ranges
 * ================================================================ */

/* Standard ternary block: 27 values */
static inline uint8_t be3(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}

/* Binary block: 2^3 = 8 values (map -1→0, +1→1) */
static inline uint8_t be2(int8_t a,int8_t b,int8_t c){
    int va=(a>0)?1:0,vb=(b>0)?1:0,vc=(c>0)?1:0;return(uint8_t)(va*4+vb*2+vc);}

/* Pentary block: 5^3 = 125 values */
static inline uint8_t be5(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+2)*25+(b+2)*5+(c+2));}

/* ================================================================
 *  Generic pipeline
 * ================================================================ */

typedef struct{
    const char *name;
    uint8_t *joint_tr,*joint_te;
    uint32_t *hot;
    uint8_t bg;
    int n_bvals; /* 8 for binary, 27 for ternary, 125 for pentary, 256 for Encoding D */
} eye_t;

static void build_sigs(const int8_t*tern_tr,const int8_t*tern_te,
                        uint8_t*sig_tr,uint8_t*sig_te,
                        uint8_t(*enc)(int8_t,int8_t,int8_t)){
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=tern_tr+(size_t)i*PADDED;uint8_t*si=sig_tr+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=enc(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=tern_te+(size_t)i*PADDED;uint8_t*si=sig_te+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=enc(im[b2],im[b2+1],im[b2+2]);}}}

static void build_eye_raw(eye_t*e,const int8_t*ttr,const int8_t*tte,
                           uint8_t(*enc)(int8_t,int8_t,int8_t),int n_bvals){
    e->n_bvals=n_bvals;
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    build_sigs(ttr,tte,e->joint_tr,e->joint_te,enc);

    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}

    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
}

static void bay_logpost(const eye_t*e,int img,double*bp){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Ternary vs Binary vs Pentary ===\n\n");
    load_data();

    /* Build 6 eyes: 2 binary + 2 ternary + 2 pentary (fixed + adaptive each) */
    #define N_EYES 6
    eye_t eyes[N_EYES];
    const char *names[]={"Binary fixed 128","Binary adaptive median",
                         "Ternary fixed 85/170","Ternary adaptive P33/P67",
                         "Pentary adaptive quintile","Pentary fixed (51/102/153/204)"};

    printf("Building 6 eyes...\n");

    /* Binary fixed */
    {int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
     quant_binary_fixed(raw_train,ttr,TRAIN_N);quant_binary_fixed(raw_test,tte,TEST_N);
     eyes[0].name=names[0];build_eye_raw(&eyes[0],ttr,tte,be2,8);free(ttr);free(tte);}

    /* Binary adaptive */
    {int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
     quant_binary_adaptive(raw_train,ttr,TRAIN_N);quant_binary_adaptive(raw_test,tte,TEST_N);
     eyes[1].name=names[1];build_eye_raw(&eyes[1],ttr,tte,be2,8);free(ttr);free(tte);}

    /* Ternary fixed */
    {int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
     quant_ternary_fixed(raw_train,ttr,TRAIN_N);quant_ternary_fixed(raw_test,tte,TEST_N);
     eyes[2].name=names[2];build_eye_raw(&eyes[2],ttr,tte,be3,27);free(ttr);free(tte);}

    /* Ternary adaptive */
    {int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
     quant_ternary_adaptive(raw_train,ttr,TRAIN_N);quant_ternary_adaptive(raw_test,tte,TEST_N);
     eyes[3].name=names[3];build_eye_raw(&eyes[3],ttr,tte,be3,27);free(ttr);free(tte);}

    /* Pentary adaptive */
    {int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
     quant_pentary_adaptive(raw_train,ttr,TRAIN_N);quant_pentary_adaptive(raw_test,tte,TEST_N);
     eyes[4].name=names[4];build_eye_raw(&eyes[4],ttr,tte,be5,125);free(ttr);free(tte);}

    /* Pentary fixed (51/102/153/204) */
    {int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
     for(int img=0;img<TRAIN_N;img++){const uint8_t*si=raw_train+(size_t)img*PIXELS;int8_t*di=ttr+(size_t)img*PADDED;
         for(int i=0;i<PIXELS;i++){if(si[i]>204)di[i]=2;else if(si[i]>153)di[i]=1;else if(si[i]>102)di[i]=0;
             else if(si[i]>51)di[i]=-1;else di[i]=-2;}}
     for(int img=0;img<TEST_N;img++){const uint8_t*si=raw_test+(size_t)img*PIXELS;int8_t*di=tte+(size_t)img*PADDED;
         for(int i=0;i<PIXELS;i++){if(si[i]>204)di[i]=2;else if(si[i]>153)di[i]=1;else if(si[i]>102)di[i]=0;
             else if(si[i]>51)di[i]=-1;else di[i]=-2;}}
     eyes[5].name=names[5];build_eye_raw(&eyes[5],ttr,tte,be5,125);free(ttr);free(tte);}

    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * Individual eye accuracy
     * ================================================================ */
    printf("=== INDIVIDUAL EYE ACCURACY (Bayesian) ===\n\n");
    printf("  %-35s  Acc     BG%%   Vocab\n","Eye");
    for(int ei=0;ei<N_EYES;ei++){
        int correct=0;
        for(int i=0;i<TEST_N;i++){double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            bay_logpost(&eyes[ei],i,bp);double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i])correct++;}
        long bg_cnt=0;for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=eyes[ei].joint_tr+(size_t)i*SIG_PAD;
            for(int k=0;k<N_BLOCKS;k++)if(sig[k]==eyes[ei].bg)bg_cnt++;}
        printf("  %-35s  %.2f%%  %.1f%%  %d vals\n",names[ei],100.0*correct/TEST_N,
               100.0*bg_cnt/((long)TRAIN_N*N_BLOCKS),eyes[ei].n_bvals);
    }

    /* ================================================================
     * Cross-level stereo: binary + ternary, ternary + pentary, etc.
     * ================================================================ */
    printf("\n=== CROSS-LEVEL STEREO COMBINATIONS ===\n\n");

    /* Test interesting pairs */
    int pairs[][2]={{0,2},{0,3},{1,3},{2,3},{2,4},{3,4},{0,4},{1,4},
                    {0,5},{2,5},{3,5}};
    int npairs=11;
    printf("  %-50s  Accuracy\n","Combination");

    for(int pi=0;pi<npairs;pi++){
        int a=pairs[pi][0],b=pairs[pi][1];
        int correct=0;
        for(int i=0;i<TEST_N;i++){double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            bay_logpost(&eyes[a],i,bp);bay_logpost(&eyes[b],i,bp);
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i])correct++;}
        char label[128];snprintf(label,sizeof(label),"%s + %s",names[a],names[b]);
        printf("  %-50s  %.2f%%\n",label,100.0*correct/TEST_N);
    }

    /* Best triple from each level */
    printf("\n=== BEST TRIPLE: one from each level ===\n\n");
    {int best=0;int ba=-1,bb=-1,bc=-1;
     /* binary × ternary × pentary */
     int bins[]={0,1},terns[]={2,3},pents[]={4,5};
     for(int bi=0;bi<2;bi++)for(int ti=0;ti<2;ti++)for(int pi2=0;pi2<2;pi2++){
         int a=bins[bi],b=terns[ti],c=pents[pi2];
         int correct=0;
         for(int i=0;i<TEST_N;i++){double bp[N_CLASSES];memset(bp,0,sizeof(bp));
             bay_logpost(&eyes[a],i,bp);bay_logpost(&eyes[b],i,bp);bay_logpost(&eyes[c],i,bp);
             double mx=-1e30;for(int cc=0;cc<N_CLASSES;cc++)if(bp[cc]>mx)mx=bp[cc];
             int pred=0;for(int cc=1;cc<N_CLASSES;cc++)if(bp[cc]>bp[pred])pred=cc;
             if(pred==test_labels[i])correct++;}
         char label[256];snprintf(label,sizeof(label),"%s + %s + %s",names[a],names[b],names[c]);
         printf("  %-70s  %.2f%%\n",label,100.0*correct/TEST_N);
         if(correct>best){best=correct;ba=a;bb=b;bc=c;}}
     printf("\n  Best triple: %s + %s + %s = %.2f%%\n",names[ba],names[bb],names[bc],100.0*best/TEST_N);}

    /* All 6 eyes combined */
    printf("\n=== ALL 6 EYES COMBINED ===\n\n");
    {int correct=0;
     for(int i=0;i<TEST_N;i++){double bp[N_CLASSES];memset(bp,0,sizeof(bp));
         for(int ei=0;ei<N_EYES;ei++)bay_logpost(&eyes[ei],i,bp);
         double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         if(pred==test_labels[i])correct++;}
     printf("  All 6 eyes: %.2f%%\n",100.0*correct/TEST_N);}

    printf("\n  Reference: 3-eye ternary stereo = 41.18%%\n");
    printf("  Reference: stereo + MT4 stack    = 44.48%%\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
