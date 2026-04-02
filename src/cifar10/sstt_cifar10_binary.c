/*
 * sstt_cifar10_binary.c — Binary Split: Machines vs Animals
 *
 * Machines: airplane(0), automobile(1), ship(8), truck(9)
 * Animals: bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)
 *
 * Tests: can the ternary system reliably distinguish these two groups?
 * If yes: hierarchical classification (binary split → within-group) may
 * beat flat 10-class classification.
 *
 * Also tests: within-group 4-class and 6-class accuracy.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_binary src/sstt_cifar10_binary.c -lm
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

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;

/* Machine classes: 0,1,8,9. Animal classes: 2,3,4,5,6,7 */
static int is_machine(int c){return c==0||c==1||c==8||c==9;}

static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}

/* Three quantization functions */
static void quant_fixed(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}
static void quant_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}
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

/* Pipeline */
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

typedef struct {
    uint8_t *joint_tr, *joint_te;
    uint32_t *hot;
    uint8_t bg;
} eye_t;

static void build_eye(eye_t *e, const int8_t *tern_tr, const int8_t *tern_te) {
    int8_t *hg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*hg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *vg_tr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vg_te=aligned_alloc(32,(size_t)TEST_N*PADDED);
    for(int i2=0;i2<TRAIN_N;i2++){const int8_t*ti=tern_tr+(size_t)i2*PADDED;int8_t*hi=hg_tr+(size_t)i2*PADDED,*vi=vg_tr+(size_t)i2*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    for(int i2=0;i2<TEST_N;i2++){const int8_t*ti=tern_te+(size_t)i2*PADDED;int8_t*hi=hg_te+(size_t)i2*PADDED,*vi=vg_te+(size_t)i2*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    bsf(tern_tr,pxt,TRAIN_N);bsf(tern_te,pxe,TEST_N);
    bsf(hg_tr,hst,TRAIN_N);bsf(hg_te,hse,TEST_N);bsf(vg_tr,vst,TRAIN_N);bsf(vg_te,vse,TEST_N);
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    trf(pxt,htt,vtt,TRAIN_N);trf(pxe,hte,vte,TEST_N);
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    jsf(e->joint_tr,TRAIN_N,pxt,hst,vst,htt,vtt);jsf(e->joint_te,TEST_N,pxe,hse,vse,hte,vte);
    free(hg_tr);free(hg_te);free(vg_tr);free(vg_te);free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);
    free(htt);free(hte);free(vtt);free(vte);
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    /* Full 10-class hot map — used for all classification tasks */
    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
}

/* Bayesian classification with arbitrary label mapping */
static void bay_logpost(const eye_t *e, int img, double *bp) {
    const uint8_t *sig = e->joint_te + (size_t)img * SIG_PAD;
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = sig[k]; if (bv == e->bg) continue;
        const uint32_t *h = e->hot + (size_t)k * BYTE_VALS * CLS_PAD + (size_t)bv * CLS_PAD;
        for (int c = 0; c < N_CLASSES; c++) bp[c] += log(h[c] + 0.5);
    }
}

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
    printf("=== SSTT CIFAR-10: Binary Split (Machines vs Animals) ===\n\n");
    printf("  Machines: airplane, automobile, ship, truck (4 classes)\n");
    printf("  Animals:  bird, cat, deer, dog, frog, horse (6 classes)\n\n");
    load_data();

    /* Build 3 stereoscopic eyes */
    printf("Building 3 eyes...\n");
    eye_t eyes[3];
    void (*qfns[])(const uint8_t*,int8_t*,int)={quant_fixed,quant_adaptive,quant_perchannel};
    const char *enames[]={"Fixed","Adaptive","Per-channel"};
    for(int ei=0;ei<3;ei++){
        int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
        qfns[ei](raw_train,ttr,TRAIN_N);qfns[ei](raw_test,tte,TEST_N);
        build_eye(&eyes[ei],ttr,tte);
        free(ttr);free(tte);
        printf("  %s: BG=%d\n",enames[ei],eyes[ei].bg);
    }
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * TEST 1: Binary classification (machine vs animal)
     * ================================================================ */
    printf("=== TEST 1: Binary — Machine vs Animal ===\n\n");
    {
        /* 3-eye stereo Bayesian, collapse to binary */
        int correct_binary=0;
        int conf[2][2]={{0}};  /* [actual][predicted] */
        for(int i=0;i<TEST_N;i++){
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            for(int ei=0;ei<3;ei++)bay_logpost(&eyes[ei],i,bp);
            /* Sum machine classes vs animal classes */
            double machine_score=0,animal_score=0;
            /* Normalize per-class first */
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;
            for(int c=0;c<N_CLASSES;c++){
                double p=exp(bp[c]);
                if(is_machine(c))machine_score+=p;else animal_score+=p;}
            int pred_binary=(machine_score>animal_score)?0:1;
            int true_binary=is_machine(test_labels[i])?0:1;
            conf[true_binary][pred_binary]++;
            if(pred_binary==true_binary)correct_binary++;
        }
        printf("  Binary accuracy: %.2f%% (%d/%d)\n",100.0*correct_binary/TEST_N,correct_binary,TEST_N);
        printf("  Confusion:  pred_machine  pred_animal\n");
        printf("    machine:     %5d        %5d     (recall %.1f%%)\n",conf[0][0],conf[0][1],100.0*conf[0][0]/(conf[0][0]+conf[0][1]));
        printf("    animal:      %5d        %5d     (recall %.1f%%)\n",conf[1][0],conf[1][1],100.0*conf[1][1]/(conf[1][0]+conf[1][1]));
    }

    /* ================================================================
     * TEST 2: Within-group classification
     * ================================================================ */
    printf("\n=== TEST 2: Within-Group Classification ===\n\n");

    /* Machines only: 4-class (airplane, automobile, ship, truck) */
    {
        int machine_classes[]={0,1,8,9};int nm=4;
        int correct=0,total=0;int pc[10]={0},pt[10]={0};
        for(int i=0;i<TEST_N;i++){
            if(!is_machine(test_labels[i]))continue;total++;pt[test_labels[i]]++;
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            for(int ei=0;ei<3;ei++)bay_logpost(&eyes[ei],i,bp);
            /* Argmax over machine classes only */
            int pred=machine_classes[0];
            for(int mi=1;mi<nm;mi++)if(bp[machine_classes[mi]]>bp[pred])pred=machine_classes[mi];
            if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}
        }
        printf("  Machines (4-class, %d images): %.2f%%\n",total,100.0*correct/total);
        for(int mi=0;mi<nm;mi++){int c=machine_classes[mi];
            printf("    %d %-12s %.1f%% (%d/%d)\n",c,cn[c],100.0*pc[c]/pt[c],pc[c],pt[c]);}
    }

    /* Animals only: 6-class */
    {
        int animal_classes[]={2,3,4,5,6,7};int na=6;
        int correct=0,total=0;int pc[10]={0},pt[10]={0};
        for(int i=0;i<TEST_N;i++){
            if(is_machine(test_labels[i]))continue;total++;pt[test_labels[i]]++;
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            for(int ei=0;ei<3;ei++)bay_logpost(&eyes[ei],i,bp);
            int pred=animal_classes[0];
            for(int ai=1;ai<na;ai++)if(bp[animal_classes[ai]]>bp[pred])pred=animal_classes[ai];
            if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}
        }
        printf("\n  Animals (6-class, %d images): %.2f%%\n",total,100.0*correct/total);
        for(int ai=0;ai<na;ai++){int c=animal_classes[ai];
            printf("    %d %-12s %.1f%% (%d/%d)\n",c,cn[c],100.0*pc[c]/pt[c],pc[c],pt[c]);}
    }

    /* ================================================================
     * TEST 3: Hierarchical — binary split then within-group
     * ================================================================ */
    printf("\n=== TEST 3: Hierarchical (binary → within-group) ===\n\n");
    {
        int machine_classes[]={0,1,8,9};int nm=4;
        int animal_classes[]={2,3,4,5,6,7};int na=6;
        int correct=0;int pc[10]={0},pt[10]={0};

        for(int i=0;i<TEST_N;i++){
            pt[test_labels[i]]++;
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            for(int ei=0;ei<3;ei++)bay_logpost(&eyes[ei],i,bp);
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            for(int c=0;c<N_CLASSES;c++)bp[c]-=mx;

            /* Binary split */
            double machine_score=0,animal_score=0;
            for(int c=0;c<N_CLASSES;c++){double p=exp(bp[c]);
                if(is_machine(c))machine_score+=p;else animal_score+=p;}

            int pred;
            if(machine_score>animal_score){
                /* Classify among machines only */
                pred=machine_classes[0];
                for(int mi=1;mi<nm;mi++)if(bp[machine_classes[mi]]>bp[pred])pred=machine_classes[mi];
            } else {
                /* Classify among animals only */
                pred=animal_classes[0];
                for(int ai=1;ai<na;ai++)if(bp[animal_classes[ai]]>bp[pred])pred=animal_classes[ai];
            }
            if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}
        }
        printf("  Hierarchical 10-class: %.2f%%\n",100.0*correct/TEST_N);
        printf("  (vs flat 3-eye Bayesian: 41.18%%)\n\n");
        for(int c=0;c<N_CLASSES;c++)printf("    %d %-12s %.1f%%\n",c,cn[c],100.0*pc[c]/(TEST_N/N_CLASSES));
    }

    /* ================================================================
     * TEST 4: Oracle binary — what if binary split were perfect?
     * ================================================================ */
    printf("\n=== TEST 4: Oracle — Perfect Binary Split ===\n\n");
    {
        int machine_classes[]={0,1,8,9};int nm=4;
        int animal_classes[]={2,3,4,5,6,7};int na=6;
        int correct=0;

        for(int i=0;i<TEST_N;i++){
            double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            for(int ei=0;ei<3;ei++)bay_logpost(&eyes[ei],i,bp);

            int pred;
            if(is_machine(test_labels[i])){
                pred=machine_classes[0];
                for(int mi=1;mi<nm;mi++)if(bp[machine_classes[mi]]>bp[pred])pred=machine_classes[mi];
            } else {
                pred=animal_classes[0];
                for(int ai=1;ai<na;ai++)if(bp[animal_classes[ai]]>bp[pred])pred=animal_classes[ai];
            }
            if(pred==test_labels[i])correct++;
        }
        printf("  Oracle hierarchical: %.2f%% (with perfect binary split)\n",100.0*correct/TEST_N);
        printf("  (ceiling for hierarchical approach)\n");
    }

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
