/*
 * sstt_cifar10_binary_dig.c — Atomic Analysis of Binary Split Failures
 *
 * For the 19% of images where machine/animal is wrong:
 * What do they look like? What features predict binary failure?
 * Can a dedicated binary classifier (2-class IG, 2-class hot map) do better?
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_binary_dig src/sstt_cifar10_binary_dig.c -lm
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

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint8_t *raw_train,*raw_test,*raw_gray_tr,*raw_gray_te,*train_labels,*test_labels;
static int is_machine(int c){return c==0||c==1||c==8||c==9;}

static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
/* Quantization functions */
static void quant_fixed(const uint8_t*s,int8_t*d,int n){for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
    for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}
static void quant_adaptive(const uint8_t*s,int8_t*d,int n){uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}
static void quant_perchannel(const uint8_t*s,int8_t*d,int n){uint8_t*cv=malloc(1024);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        uint8_t p33[3],p67[3];for(int ch=0;ch<3;ch++){int cnt=0;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++)cv[cnt++]=si[y*96+x*3+ch];
            qsort(cv,cnt,1,cmp_u8);p33[ch]=cv[cnt/3];p67[ch]=cv[2*cnt/3];
            if(p33[ch]==p67[ch]){p33[ch]=p33[ch]>0?p33[ch]-1:0;p67[ch]=p67[ch]<255?p67[ch]+1:255;}}
        for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int ch=0;ch<3;ch++){
            int idx=y*96+x*3+ch;di[idx]=si[idx]>p67[ch]?1:si[idx]<p33[ch]?-1:0;}}free(cv);}
/* Pipeline */
static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static void bsf(const int8_t*d,uint8_t*s,int n){for(int i=0;i<n;i++){const int8_t*im=d+(size_t)i*PADDED;uint8_t*si=s+(size_t)i*SIG_PAD;
    for(int y=0;y<IMG_H;y++)for(int s2=0;s2<BLKS_PER_ROW;s2++){int b2=y*IMG_W+s2*3;si[y*BLKS_PER_ROW+s2]=be(im[b2],im[b2+1],im[b2+2]);}}}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}
static void trf(const uint8_t*bsi,uint8_t*ht,uint8_t*vt,int n){for(int i=0;i<n;i++){const uint8_t*s=bsi+(size_t)i*SIG_PAD;uint8_t*h=ht+(size_t)i*TRANS_PAD,*v=vt+(size_t)i*TRANS_PAD;
    for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
    for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}}
static void jsf(uint8_t*o,int n,const uint8_t*px,const uint8_t*hg,const uint8_t*vg,const uint8_t*ht,const uint8_t*vt){
    for(int i=0;i<n;i++){const uint8_t*pi=px+(size_t)i*SIG_PAD,*hi=hg+(size_t)i*SIG_PAD,*vi=vg+(size_t)i*SIG_PAD;
        const uint8_t*hti=ht+(size_t)i*TRANS_PAD,*vti=vt+(size_t)i*TRANS_PAD;uint8_t*oi=o+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}}

typedef struct{uint8_t *joint_tr,*joint_te;uint32_t *hot;uint8_t bg;}eye_t;

static void build_eye(eye_t*e,const int8_t*ttr,const int8_t*tte){
    int8_t *hgr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*hge=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *vgr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vge=aligned_alloc(32,(size_t)TEST_N*PADDED);
    for(int i2=0;i2<TRAIN_N;i2++){const int8_t*ti=ttr+(size_t)i2*PADDED;int8_t*hi=hgr+(size_t)i2*PADDED,*vi=vgr+(size_t)i2*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    for(int i2=0;i2<TEST_N;i2++){const int8_t*ti=tte+(size_t)i2*PADDED;int8_t*hi=hge+(size_t)i2*PADDED,*vi=vge+(size_t)i2*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(ttr,pxt,TRAIN_N);bsf(tte,pxe,TEST_N);bsf(hgr,hst,TRAIN_N);bsf(hge,hse,TEST_N);bsf(vgr,vst,TRAIN_N);bsf(vge,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte,vte,TEST_N);
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(e->joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(e->joint_te,TEST_N,pxe,hse,vse,hte,vte);
    free(hgr);free(hge);free(vgr);free(vge);free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte);free(vtt);free(vte);
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    /* 10-class hot map */
    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}}

static void bay_logpost(const eye_t*e,int img,double*bp){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*1024);raw_gray_te=malloc((size_t)TEST_N*1024);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            uint8_t*gd=raw_gray_tr+(size_t)idx*1024;
            for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        uint8_t*gd=raw_gray_te+(size_t)i*1024;
        for(int p2=0;p2<1024;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== Binary Split Deep Dive: What Makes 5/5? ===\n\n");
    load_data();

    /* Build 3 eyes */
    eye_t eyes[3];
    void(*qfns[])(const uint8_t*,int8_t*,int)={quant_fixed,quant_adaptive,quant_perchannel};
    for(int ei=0;ei<3;ei++){int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
        qfns[ei](raw_train,ttr,TRAIN_N);qfns[ei](raw_test,tte,TEST_N);build_eye(&eyes[ei],ttr,tte);free(ttr);free(tte);}
    printf("  Built 3 eyes (%.1f sec)\n\n",now_sec()-t0);

    /* Classify every test image, record binary correctness + features */
    typedef struct{
        float brightness,contrast,r_mean,g_mean,b_mean,color_dom,edge_density;
        int true_class,true_binary; /* 0=machine,1=animal */
        int pred_binary;
        double machine_score,animal_score,margin;
        int binary_correct;
    } img_t;
    img_t *imgs=malloc(TEST_N*sizeof(img_t));

    for(int i=0;i<TEST_N;i++){
        img_t *im=&imgs[i];
        im->true_class=test_labels[i];
        im->true_binary=is_machine(test_labels[i])?0:1;

        /* 3-eye Bayesian */
        double bp[N_CLASSES];memset(bp,0,sizeof(bp));
        for(int ei=0;ei<3;ei++)bay_logpost(&eyes[ei],i,bp);
        double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
        for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;
        im->machine_score=0;im->animal_score=0;
        for(int c=0;c<N_CLASSES;c++){double p=exp(bp[c]);if(is_machine(c))im->machine_score+=p;else im->animal_score+=p;}
        im->pred_binary=(im->machine_score>im->animal_score)?0:1;
        im->binary_correct=(im->pred_binary==im->true_binary)?1:0;
        im->margin=fabs(im->machine_score-im->animal_score)/(im->machine_score+im->animal_score);

        /* Raw features */
        const uint8_t *gray=raw_gray_te+(size_t)i*1024;
        double sum=0,sum2=0;for(int p=0;p<1024;p++){sum+=gray[p];sum2+=(double)gray[p]*gray[p];}
        im->brightness=(float)(sum/1024);im->contrast=(float)sqrt(sum2/1024-im->brightness*im->brightness);

        const uint8_t *raw=raw_test+(size_t)i*PIXELS;
        double rs=0,gs=0,bs=0;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int di=y*96+x*3;rs+=raw[di];gs+=raw[di+1];bs+=raw[di+2];}
        im->r_mean=(float)(rs/1024);im->g_mean=(float)(gs/1024);im->b_mean=(float)(bs/1024);
        float mxc=im->r_mean>im->g_mean?(im->r_mean>im->b_mean?im->r_mean:im->b_mean):(im->g_mean>im->b_mean?im->g_mean:im->b_mean);
        float mnc=im->r_mean<im->g_mean?(im->r_mean<im->b_mean?im->r_mean:im->b_mean):(im->g_mean<im->b_mean?im->g_mean:im->b_mean);
        im->color_dom=mxc-mnc;

        const int8_t *tern=aligned_alloc(32,PADDED);
        for(int p=0;p<PIXELS;p++)((int8_t*)tern)[p]=raw[p]>170?1:raw[p]<85?-1:0;
        int edges=0;for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W-1;x++)
            if(ct(tern[y*IMG_W+x+1]-tern[y*IMG_W+x])!=0)edges++;
        im->edge_density=(float)edges/(IMG_H*(IMG_W-1));
        free((void*)tern);
    }

    /* ================================================================
     * REPORT 1: Per-class binary accuracy
     * ================================================================ */
    printf("=== PER-CLASS BINARY ACCURACY ===\n\n");
    for(int c=0;c<N_CLASSES;c++){
        int total=0,correct=0;
        for(int i=0;i<TEST_N;i++)if(imgs[i].true_class==c){total++;if(imgs[i].binary_correct)correct++;}
        printf("  %d %-12s %-8s  binary=%.1f%% (%d/%d)\n",c,cn[c],is_machine(c)?"MACHINE":"ANIMAL",100.0*correct/total,correct,total);
    }

    /* ================================================================
     * REPORT 2: Feature distributions for binary correct vs wrong
     * ================================================================ */
    printf("\n=== FEATURES: Binary Correct vs Wrong ===\n\n");
    {double bc[6]={0},bw[6]={0};int nc=0,nw=0;
     for(int i=0;i<TEST_N;i++){
         float f[]={imgs[i].brightness,imgs[i].contrast,imgs[i].color_dom,imgs[i].edge_density,imgs[i].r_mean-imgs[i].b_mean,(float)imgs[i].margin};
         if(imgs[i].binary_correct){for(int j=0;j<6;j++)bc[j]+=f[j];nc++;}
         else{for(int j=0;j<6;j++)bw[j]+=f[j];nw++;}}
     const char *fn[]={"brightness","contrast","color_dom","edge_density","R-B balance","margin"};
     printf("  %-15s  Correct(%d)  Wrong(%d)  Delta\n","Feature",nc,nw);
     for(int j=0;j<6;j++)printf("  %-15s  %7.2f     %7.2f   %+.2f\n",fn[j],bc[j]/nc,bw[j]/nw,bc[j]/nc-bw[j]/nw);}

    /* ================================================================
     * REPORT 3: Margin analysis — how confident are binary errors?
     * ================================================================ */
    printf("\n=== MARGIN ANALYSIS ===\n\n");
    printf("  Margin = |machine_score - animal_score| / total\n\n");
    int margin_bins[]={0,5,10,20,30,50,100};
    for(int bi=0;bi<6;bi++){
        float lo=(float)margin_bins[bi]/100,hi=(float)margin_bins[bi+1]/100;
        int count=0,correct=0;
        for(int i=0;i<TEST_N;i++)if(imgs[i].margin>=lo&&imgs[i].margin<hi){count++;if(imgs[i].binary_correct)correct++;}
        if(count>50)printf("  margin [%.2f-%.2f]: %4d images, binary %.1f%% correct\n",lo,hi,count,100.0*correct/count);
    }

    /* ================================================================
     * REPORT 4: Which classes are confused in binary errors?
     * ================================================================ */
    printf("\n=== BINARY ERROR PAIRS ===\n\n");
    printf("  Machines misclassified as animals:\n");
    {int conf[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++)if(!imgs[i].binary_correct&&imgs[i].true_binary==0)conf[imgs[i].true_class]++;
     for(int c=0;c<N_CLASSES;c++)if(is_machine(c)&&conf[c]>0)printf("    %-12s %d misrouted\n",cn[c],conf[c]);}
    printf("  Animals misclassified as machines:\n");
    {int conf[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++)if(!imgs[i].binary_correct&&imgs[i].true_binary==1)conf[imgs[i].true_class]++;
     for(int c=0;c<N_CLASSES;c++)if(!is_machine(c)&&conf[c]>0)printf("    %-12s %d misrouted\n",cn[c],conf[c]);}

    /* ================================================================
     * REPORT 5: Binary accuracy by brightness/contrast bins
     * ================================================================ */
    printf("\n=== BINARY ACCURACY BY BRIGHTNESS ===\n\n");
    for(int bi=0;bi<5;bi++){
        float lo=bi*51.2f,hi=(bi+1)*51.2f;int count=0,correct=0;
        for(int i=0;i<TEST_N;i++)if(imgs[i].brightness>=lo&&imgs[i].brightness<hi){count++;if(imgs[i].binary_correct)correct++;}
        if(count>50)printf("  [%5.1f-%5.1f]: %4d images, binary %.1f%%\n",lo,hi,count,100.0*correct/count);}

    printf("\n=== BINARY ACCURACY BY CONTRAST ===\n\n");
    for(int bi=0;bi<5;bi++){
        float lo=bi*20.0f,hi=(bi+1)*20.0f;int count=0,correct=0;
        for(int i=0;i<TEST_N;i++)if(imgs[i].contrast>=lo&&imgs[i].contrast<hi){count++;if(imgs[i].binary_correct)correct++;}
        if(count>50)printf("  [%5.1f-%5.1f]: %4d images, binary %.1f%%\n",lo,hi,count,100.0*correct/count);}

    printf("\n=== BINARY ACCURACY BY R-B BALANCE ===\n\n");
    printf("  (R-B > 0 = warm/red, R-B < 0 = cool/blue)\n");
    for(int bi=-2;bi<3;bi++){
        float lo=bi*20.0f,hi=(bi+1)*20.0f;int count=0,correct=0;
        for(int i=0;i<TEST_N;i++){float rb=imgs[i].r_mean-imgs[i].b_mean;
            if(rb>=lo&&rb<hi){count++;if(imgs[i].binary_correct)correct++;}}
        if(count>50)printf("  [%+5.1f to %+5.1f]: %4d images, binary %.1f%%\n",lo,hi,count,100.0*correct/count);}

    /* ================================================================
     * REPORT 6: Dedicated binary Bayesian (2-class hot map)
     * ================================================================ */
    printf("\n=== DEDICATED BINARY CLASSIFIER (2-class IG + hot map) ===\n\n");
    {
        /* Build binary hot maps per eye: count machine vs animal */
        int binary_correct=0;
        /* For each eye, build a 2-class hot map */
        for(int ei=0;ei<3;ei++){
            uint32_t *bhot=calloc((size_t)N_BLOCKS*BYTE_VALS*4,sizeof(uint32_t)); /* 2-class, padded to 4 */
            for(int i=0;i<TRAIN_N;i++){
                int bl=is_machine(train_labels[i])?0:1;
                const uint8_t*sig=eyes[ei].joint_tr+(size_t)i*SIG_PAD;
                for(int k=0;k<N_BLOCKS;k++)bhot[(size_t)k*BYTE_VALS*4+(size_t)sig[k]*4+bl]++;}
            /* Store for this eye */
            if(ei==0){/* Use eye 0's binary hot for initial test */
                int correct=0;
                for(int i=0;i<TEST_N;i++){
                    const uint8_t*sig=eyes[ei].joint_te+(size_t)i*SIG_PAD;
                    double bm=0,ba=0;
                    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==eyes[ei].bg)continue;
                        bm+=log(bhot[(size_t)k*BYTE_VALS*4+(size_t)bv*4+0]+0.5);
                        ba+=log(bhot[(size_t)k*BYTE_VALS*4+(size_t)bv*4+1]+0.5);}
                    int pred=(bm>ba)?0:1;int truth=is_machine(test_labels[i])?0:1;
                    if(pred==truth)correct++;}
                printf("  Eye 1 (fixed) binary-dedicated: %.2f%%\n",100.0*correct/TEST_N);}
            free(bhot);
        }

        /* 3-eye combined binary-dedicated */
        /* Build per-eye binary hot maps and combine */
        uint32_t *bhots[3];
        for(int ei=0;ei<3;ei++){
            bhots[ei]=calloc((size_t)N_BLOCKS*BYTE_VALS*4,sizeof(uint32_t));
            for(int i=0;i<TRAIN_N;i++){int bl=is_machine(train_labels[i])?0:1;
                const uint8_t*sig=eyes[ei].joint_tr+(size_t)i*SIG_PAD;
                for(int k=0;k<N_BLOCKS;k++)bhots[ei][(size_t)k*BYTE_VALS*4+(size_t)sig[k]*4+bl]++;}}

        int correct=0;
        for(int i=0;i<TEST_N;i++){
            double bm=0,ba=0;
            for(int ei=0;ei<3;ei++){
                const uint8_t*sig=eyes[ei].joint_te+(size_t)i*SIG_PAD;
                for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==eyes[ei].bg)continue;
                    bm+=log(bhots[ei][(size_t)k*BYTE_VALS*4+(size_t)bv*4+0]+0.5);
                    ba+=log(bhots[ei][(size_t)k*BYTE_VALS*4+(size_t)bv*4+1]+0.5);}}
            int pred=(bm>ba)?0:1;int truth=is_machine(test_labels[i])?0:1;
            if(pred==truth)correct++;}
        printf("  3-eye combined binary-dedicated: %.2f%%\n",100.0*correct/TEST_N);
        printf("  (vs 10-class collapsed: 81.04%%)\n");

        /* Per-class */
        printf("\n  Per-class binary-dedicated accuracy:\n");
        {int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
         for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
             double bm=0,ba=0;
             for(int ei=0;ei<3;ei++){const uint8_t*sig=eyes[ei].joint_te+(size_t)i*SIG_PAD;
                 for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==eyes[ei].bg)continue;
                     bm+=log(bhots[ei][(size_t)k*BYTE_VALS*4+(size_t)bv*4+0]+0.5);
                     ba+=log(bhots[ei][(size_t)k*BYTE_VALS*4+(size_t)bv*4+1]+0.5);}}
             int pred=(bm>ba)?0:1;int truth=is_machine(test_labels[i])?0:1;
             if(pred==truth)pc[test_labels[i]]++;}
         for(int c=0;c<N_CLASSES;c++)printf("    %d %-12s %-8s  %.1f%%\n",c,cn[c],is_machine(c)?"MACHINE":"ANIMAL",100.0*pc[c]/pt[c]);}

        for(int ei=0;ei<3;ei++)free(bhots[ei]);
    }

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    free(imgs);return 0;
}
