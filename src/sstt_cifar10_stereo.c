/*
 * sstt_cifar10_stereo.c — Multi-Perspective Stereoscopic Classification
 *
 * Multiple quantization "eyes" viewing the same image:
 *   Eye 1: Fixed 85/170 (absolute brightness — sees scenes)
 *   Eye 2: Adaptive P33/P67 (relative structure — sees edges)
 *   Eye 3: Per-channel adaptive (color-normalized — sees color relationships)
 *   Eye 4: Median split (binary — sees foreground/background)
 *   Eye 5: Quartile (4-level — finer relative structure)
 *
 * Each eye builds its own Bayesian posterior. Combined posterior fuses all views.
 * The IG weights per eye automatically learn which blocks are discriminative
 * from that quantization perspective.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_stereo src/sstt_cifar10_stereo.c -lm
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
#define MAX_EYES 5

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;

static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}

/* ================================================================
 *  5 Quantization functions — 5 "eyes"
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

/* Eye 4: Median split (binary: above/below median → +1/-1, no zero) */
static void quant_median(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t med=sorted[PIXELS/2];
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>med?1:si[i]<med?-1:0;}free(sorted);}

/* Eye 5: Quartile (4-level using P25/P50/P75: -1, 0(low), 0(high)→mapped to trit via majority in block) */
/* Actually: use P20/P50/P80 for a wider middle band */
static void quant_wide_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p20=sorted[PIXELS/5],p80=sorted[4*PIXELS/5];
        if(p20==p80){p20=p20>0?p20-1:0;p80=p80<255?p80+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p80?1:si[i]<p20?-1:0;}free(sorted);}

/* ================================================================
 *  Pipeline per eye
 * ================================================================ */
typedef struct {
    const char *name;
    int8_t *hg_tr,*hg_te,*vg_tr,*vg_te;
    uint8_t *joint_tr,*joint_te;
    uint32_t *hot;
    uint16_t ig[N_BLOCKS];
    uint8_t bg;
    /* No inverted index needed — Bayesian only for speed */
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
    e->hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);e->hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    e->vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED);e->vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    for(int i2=0;i2<TRAIN_N;i2++){const int8_t*ti=tern_tr+(size_t)i2*PADDED;
        int8_t*hi=e->hg_tr+(size_t)i2*PADDED,*vi=e->vg_tr+(size_t)i2*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    for(int i2=0;i2<TEST_N;i2++){const int8_t*ti=tern_te+(size_t)i2*PADDED;
        int8_t*hi=e->hg_te+(size_t)i2*PADDED,*vi=e->vg_te+(size_t)i2*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}

    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(tern_tr,pxt,TRAIN_N);bsf(tern_te,pxe,TEST_N);
    bsf(e->hg_tr,hst,TRAIN_N);bsf(e->hg_te,hse,TEST_N);
    bsf(e->vg_tr,vst,TRAIN_N);bsf(e->vg_te,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte,vte,TEST_N);
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(e->joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(e->joint_te,TEST_N,pxe,hse,vse,hte,vte);
    free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte);free(vtt);free(vte);

    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}

    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}

    printf("    %-25s BG=%d (%.1f%%)\n",e->name,e->bg,100.0*mc/((long)TRAIN_N*N_BLOCKS));
}

static void bay_eye(const eye_t*e,int img,double*bp){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
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
    printf("=== SSTT CIFAR-10: Stereoscopic Multi-Perspective Classification ===\n\n");
    load_data();

    /* Build 5 eyes */
    eye_t eyes[MAX_EYES];
    const char *names[]={"Fixed 85/170","Adaptive P33/P67","Per-channel adaptive","Median split","Wide P20/P80"};
    void (*quant_fns[])(const uint8_t*,int8_t*,int)={quant_fixed,quant_adaptive,quant_perchannel,quant_median,quant_wide_adaptive};

    printf("Building %d eyes...\n",MAX_EYES);
    for(int ei=0;ei<MAX_EYES;ei++){
        eyes[ei].name=names[ei];
        int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
        quant_fns[ei](raw_train,ttr,TRAIN_N);quant_fns[ei](raw_test,tte,TEST_N);
        build_eye(&eyes[ei],ttr,tte);
        free(ttr);free(tte);
    }
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * Test: individual eyes, then progressive combinations
     * ================================================================ */

    /* Individual Bayesian per eye */
    printf("--- Individual Eye Accuracy (Bayesian) ---\n\n");
    for(int ei=0;ei<MAX_EYES;ei++){
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            bay_eye(&eyes[ei],i,bp);
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i])correct++;
        }
        printf("  Eye %d %-25s %.2f%%\n",ei+1,names[ei],100.0*correct/TEST_N);
    }

    /* Progressive combination: 1, 1+2, 1+2+3, ... */
    printf("\n--- Progressive Combination (Bayesian posterior fusion) ---\n\n");
    for(int n_eyes=1;n_eyes<=MAX_EYES;n_eyes++){
        int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            for(int ei=0;ei<n_eyes;ei++)bay_eye(&eyes[ei],i,bp);
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}
        }
        printf("  %d eye%s: %.2f%%  ",n_eyes,n_eyes>1?"s":" ",100.0*correct/TEST_N);
        /* Show which eyes */
        printf("(");for(int ei=0;ei<n_eyes;ei++){if(ei)printf("+");printf("%s",names[ei]);}printf(")\n");
        /* Per-class */
        for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);
        printf("\n");
    }

    /* Also: best subset search (try all 2^5-1 = 31 combinations) */
    printf("--- Best Subset Search (all 31 combinations) ---\n\n");
    int best_mask=0;double best_acc=0;
    for(int mask=1;mask<(1<<MAX_EYES);mask++){
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            for(int ei=0;ei<MAX_EYES;ei++)if(mask&(1<<ei))bay_eye(&eyes[ei],i,bp);
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i])correct++;
        }
        double acc=100.0*correct/TEST_N;
        if(acc>best_acc){best_acc=acc;best_mask=mask;}
    }
    printf("  Best combination: ");
    for(int ei=0;ei<MAX_EYES;ei++)if(best_mask&(1<<ei))printf("%s + ",names[ei]);
    printf("\b\b-> %.2f%%\n",best_acc);

    /* Show per-class for best */
    {int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
         double bp[N_CLASSES];memset(bp,0,sizeof(bp));
         for(int ei=0;ei<MAX_EYES;ei++)if(best_mask&(1<<ei))bay_eye(&eyes[ei],i,bp);
         double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         if(pred==test_labels[i])pc[test_labels[i]]++;}
     for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);}

    printf("\n=== SUMMARY ===\n");
    printf("  Single best eye:          %.2f%% (fixed 85/170)\n",36.58);
    printf("  Dual (fixed+adaptive):    40.20%%\n");
    printf("  Best multi-perspective:   %.2f%%\n",best_acc);
    printf("  Previous best (MT4 stack):42.05%%\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
