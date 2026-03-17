/*
 * sstt_cifar10_edgemask.c — Background Suppression via Edge Structure
 *
 * The problem: the system classifies by background (blue sky → airplane,
 * green → frog) instead of object shape. Background is noise.
 *
 * The fix: quantize based on GRADIENT MAGNITUDE, not pixel value.
 * Flat background → zero gradient → suppressed.
 * Object edges → high gradient → emphasized.
 *
 * Tests several background-suppressing quantization schemes:
 *   1. Gradient magnitude ternary (edge/no-edge/strong-edge)
 *   2. Edge-masked pixels (zero out low-gradient regions, quantize rest)
 *   3. Local contrast (pixel minus local mean → structure only)
 *   4. Gradient direction (which way edges point, ignoring magnitude)
 *   5. Stereoscopic: combine edge-aware eyes with standard eyes
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_edgemask src/sstt_cifar10_edgemask.c -lm
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

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

/* ================================================================
 *  Background-suppressing quantization schemes
 *  All operate on flattened RGB (32x96)
 * ================================================================ */

/* Eye 1: Standard ternary fixed (baseline) */
static void quant_standard(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}

/* Eye 2: Adaptive ternary (baseline) */
static void quant_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}

/* Eye 3: Edge-masked — zero out pixels with low local gradient */
static void quant_edgemask(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){
        const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        /* Compute gradient magnitude per pixel */
        for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
            int idx=y*IMG_W+x;
            int gx=(x<IMG_W-1)?(int)si[idx+1]-(int)si[idx]:0;
            int gy=(y<IMG_H-1)?(int)si[idx+IMG_W]-(int)si[idx]:0;
            int mag=abs(gx)+abs(gy); /* L1 gradient magnitude */
            /* If gradient is low → background → zero.
               If gradient is high → edge → quantize the pixel normally. */
            if(mag<30) di[idx]=0; /* suppress flat regions */
            else di[idx]=si[idx]>170?1:si[idx]<85?-1:0;
        }
    }
}

/* Eye 4: Local contrast — subtract 5x5 local mean, quantize residual */
static void quant_local_contrast(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){
        const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
            /* 5x5 local mean (clamped at borders) */
            int sum=0,cnt=0;
            for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){
                int ny=y+dy,nx=x+dx;
                if(ny>=0&&ny<IMG_H&&nx>=0&&nx<IMG_W){sum+=si[ny*IMG_W+nx];cnt++;}}
            int mean=sum/cnt;
            int residual=(int)si[y*IMG_W+x]-mean;
            /* Quantize residual: brighter than local mean → +1, darker → -1, similar → 0 */
            di[y*IMG_W+x]=(residual>15)?1:(residual<-15)?-1:0;
        }
    }
}

/* Eye 5: Gradient direction — encode which way edges point, ignore magnitude */
static void quant_gradient_dir(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){
        const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
            int idx=y*IMG_W+x;
            int gx=(x<IMG_W-1)?(int)si[idx+1]-(int)si[idx]:0;
            int gy=(y<IMG_H-1)?(int)si[idx+IMG_W]-(int)si[idx]:0;
            int mag=abs(gx)+abs(gy);
            if(mag<20){di[idx]=0;} /* no edge → zero */
            else{
                /* Dominant direction: horizontal vs vertical vs diagonal */
                if(abs(gx)>abs(gy)*2) di[idx]=(gx>0)?1:-1; /* horizontal edge */
                else if(abs(gy)>abs(gx)*2) di[idx]=(gy>0)?1:-1; /* vertical edge */
                else di[idx]=(gx>0)==(gy>0)?1:-1; /* diagonal */
            }
        }
    }
}

/* Eye 6: Gradient magnitude as the signal (not pixel value) */
static void quant_grad_magnitude(const uint8_t*s,int8_t*d,int n){
    uint8_t *mag_buf=malloc(PIXELS);
    uint8_t *sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){
        const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        /* Compute gradient magnitude as uint8 */
        for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
            int idx=y*IMG_W+x;
            int gx=(x<IMG_W-1)?abs((int)si[idx+1]-(int)si[idx]):0;
            int gy=(y<IMG_H-1)?abs((int)si[idx+IMG_W]-(int)si[idx]):0;
            int m=gx+gy;if(m>255)m=255;
            mag_buf[idx]=(uint8_t)m;}
        /* Adaptive quantize the magnitude image */
        memcpy(sorted,mag_buf,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=mag_buf[i]>p67?1:mag_buf[i]<p33?-1:0;
    }
    free(mag_buf);free(sorted);
}

/* ================================================================
 *  Pipeline
 * ================================================================ */
static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}

typedef struct{const char*name;uint8_t *joint_tr,*joint_te;uint32_t *hot;uint8_t bg;}eye_t;

static void build_eye(eye_t*e,const int8_t*ttr,const int8_t*tte){
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    /* Direct block sigs on the ternary data (no Encoding D — just raw blocks for simplicity) */
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=ttr+(size_t)i*PADDED;uint8_t*si=e->joint_tr+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=tte+(size_t)i*PADDED;uint8_t*si=e->joint_te+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    printf("    %-25s BG=%d (%.1f%%)\n",e->name,e->bg,100.0*mc/((long)TRAIN_N*N_BLOCKS));
}

static void bay(const eye_t*e,int img,double*bp){
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
    printf("=== SSTT CIFAR-10: Background Suppression via Edge Structure ===\n\n");
    load_data();

    #define N_EYES 6
    eye_t eyes[N_EYES];
    const char*names[]={"Standard fixed","Adaptive P33/P67","Edge-masked","Local contrast","Gradient direction","Gradient magnitude"};
    void(*qfns[N_EYES])(const uint8_t*,int8_t*,int)={quant_standard,quant_adaptive,quant_edgemask,quant_local_contrast,quant_gradient_dir,quant_grad_magnitude};

    printf("Building 6 eyes...\n");
    for(int ei=0;ei<N_EYES;ei++){
        int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
        qfns[ei](raw_train,ttr,TRAIN_N);qfns[ei](raw_test,tte,TEST_N);
        eyes[ei].name=names[ei];build_eye(&eyes[ei],ttr,tte);free(ttr);free(tte);}
    printf("  Built (%.1f sec)\n\n",now_sec()-t0);

    /* Individual accuracy */
    printf("=== INDIVIDUAL EYE ACCURACY ===\n\n");
    for(int ei=0;ei<N_EYES;ei++){
        int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            bay(&eyes[ei],i,bp);double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}}
        printf("  %-25s %.2f%%\n",names[ei],100.0*correct/TEST_N);
        for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);
        printf("\n");
    }

    /* Stereo: standard + each edge-aware eye */
    printf("=== STEREO: Standard + Edge-Aware Eye ===\n\n");
    for(int ei=2;ei<N_EYES;ei++){
        int correct=0;
        for(int i=0;i<TEST_N;i++){double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            bay(&eyes[0],i,bp);bay(&eyes[ei],i,bp);
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i])correct++;}
        printf("  Standard + %-25s %.2f%%\n",names[ei],100.0*correct/TEST_N);
    }

    /* Best 3-eye: standard + adaptive + best edge eye */
    printf("\n=== STEREO: Standard + Adaptive + Each Edge Eye ===\n\n");
    for(int ei=2;ei<N_EYES;ei++){
        int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;double bp[N_CLASSES];memset(bp,0,sizeof(bp));
            bay(&eyes[0],i,bp);bay(&eyes[1],i,bp);bay(&eyes[ei],i,bp);
            double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
            if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}}
        printf("  Std + Adp + %-22s %.2f%%\n",names[ei],100.0*correct/TEST_N);
        for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);
        printf("\n");
    }

    /* All 6 combined */
    printf("=== ALL 6 EYES ===\n\n");
    {int correct=0;int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
     for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;double bp[N_CLASSES];memset(bp,0,sizeof(bp));
         for(int ei=0;ei<N_EYES;ei++)bay(&eyes[ei],i,bp);
         double mx=-1e30;for(int c=0;c<N_CLASSES;c++)if(bp[c]>mx)mx=bp[c];
         int pred=0;for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[pred])pred=c;
         if(pred==test_labels[i]){correct++;pc[test_labels[i]]++;}}
     printf("  All 6 eyes: %.2f%%\n",100.0*correct/TEST_N);
     for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);}

    printf("\n  Reference: 3-eye ternary stereo (std+adp+perch) = 41.18%%\n");
    printf("  Reference: stereo + MT4 stack = 44.48%%\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
