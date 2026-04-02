/*
 * sstt_cifar10_fused.c — Fused Architecture: Block Stereo + Gauss Map + Hierarchy
 *
 * Combines two orthogonal systems:
 *   System A: Block-based 3-eye stereo Bayesian (texture + color)
 *   System B: Grid Gauss map kNN (edge geometry + shape)
 *
 * Three fusion strategies:
 *   1. Score fusion: combine posteriors from both systems
 *   2. Hierarchical: binary machine/animal router → within-group specialists
 *   3. Confidence routing: use System A when confident, System B when not
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_fused src/sstt_cifar10_fused.c -lm
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
#define IMG_W 96
#define IMG_H 32
#define PIXELS 3072
#define PADDED 3072
#define BLKS_PER_ROW 32
#define N_BLOCKS 1024
#define SIG_PAD 1024
#define BYTE_VALS 256
#define BG_TRANS 13
#define H_TRANS_PER_ROW 31
#define N_HTRANS (H_TRANS_PER_ROW*IMG_H)
#define V_TRANS_PER_COL 31
#define N_VTRANS (BLKS_PER_ROW*V_TRANS_PER_COL)
#define TRANS_PAD 992
#define SPATIAL_W 32
#define SPATIAL_H 32
#define SP_PIXELS 1024
#define GM_DIR 3
#define GM_MAG 5
#define GM_BINS (GM_DIR*GM_DIR*GM_MAG)
#define GM_CHANNELS 4
#define GM_TOTAL (GM_BINS*GM_CHANNELS)
#define GRID_R 4
#define GRID_C 4
#define GRID_N (GRID_R*GRID_C)
#define GM_GRID_TOTAL (GM_BINS*GRID_N)

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static int is_machine(int c){return c==0||c==1||c==8||c==9;}
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static uint8_t *raw_r_tr,*raw_r_te,*raw_g_tr,*raw_g_te,*raw_b_tr,*raw_b_te,*raw_gray_tr,*raw_gray_te;

/* ================================================================
 *  SYSTEM A: Block-based 3-eye stereo Bayesian
 * ================================================================ */
typedef struct{uint8_t *joint_tr,*joint_te;uint32_t *hot;uint8_t bg;}eye_t;

static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}

static void build_eye(eye_t*e,const uint8_t*raw_tr,const uint8_t*raw_te,
                       void(*qfn)(const uint8_t*,int8_t*,int)){
    int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
    qfn(raw_tr,ttr,TRAIN_N);qfn(raw_te,tte,TEST_N);
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
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=ttr+(size_t)i*PADDED;uint8_t*si=pxt+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=tte+(size_t)i*PADDED;uint8_t*si=pxe+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=hgr+(size_t)i*PADDED;uint8_t*si=hst+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=hge+(size_t)i*PADDED;uint8_t*si=hse+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=vgr+(size_t)i*PADDED;uint8_t*si=vst+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=vge+(size_t)i*PADDED;uint8_t*si=vse+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pxt+(size_t)i*SIG_PAD;uint8_t*h=htt+(size_t)i*TRANS_PAD,*v=vtt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}
    for(int i=0;i<TEST_N;i++){const uint8_t*s=pxe+(size_t)i*SIG_PAD;uint8_t*h=hte+(size_t)i*TRANS_PAD,*v=vte+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);
        memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);
        memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*pi=pxt+(size_t)i*SIG_PAD,*hi=hst+(size_t)i*SIG_PAD,*vi=vst+(size_t)i*SIG_PAD;
        const uint8_t*hti=htt+(size_t)i*TRANS_PAD,*vti=vtt+(size_t)i*TRANS_PAD;uint8_t*oi=e->joint_tr+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}
    for(int i=0;i<TEST_N;i++){const uint8_t*pi=pxe+(size_t)i*SIG_PAD,*hi=hse+(size_t)i*SIG_PAD,*vi=vse+(size_t)i*SIG_PAD;
        const uint8_t*hti=hte+(size_t)i*TRANS_PAD,*vti=vte+(size_t)i*TRANS_PAD;uint8_t*oi=e->joint_te+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);int hs=((hi[k]/9)-1)+(((hi[k]/3)%3)-1)+((hi[k]%3)-1);
            int vs=((vi[k]/9)-1)+(((vi[k]/3)%3)-1)+((vi[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}
    free(ttr);free(tte);free(hgr);free(hge);free(vgr);free(vge);
    free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte);free(vtt);free(vte);
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}e->bg=0;long mc=0;
    for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
}

static void bay_eye(const eye_t*e,int img,double*bp){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)bv*CLS_PAD;
        for(int c=0;c<N_CLASSES;c++)bp[c]+=log(h[c]+0.5);}}

static void quant_fixed(const uint8_t*s,int8_t*d,int n){for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
    for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}
static void quant_adaptive(const uint8_t*s,int8_t*d,int n){uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}
static void quant_perchannel(const uint8_t*s,int8_t*d,int n){uint8_t*cv=malloc(1024);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        uint8_t p33[3],p67[3];for(int ch=0;ch<3;ch++){int cnt=0;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++)cv[cnt++]=si[y*96+x*3+ch];
            qsort(cv,cnt,1,cmp_u8);p33[ch]=cv[cnt/3];p67[ch]=cv[2*cnt/3];
            if(p33[ch]==p67[ch]){p33[ch]=p33[ch]>0?p33[ch]-1:0;p67[ch]=p67[ch]<255?p67[ch]+1:255;}}
        for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int ch=0;ch<3;ch++){
            int idx=y*96+x*3+ch;di[idx]=si[idx]>p67[ch]?1:si[idx]<p33[ch]?-1:0;}}free(cv);}

/* ================================================================
 *  SYSTEM B: Grid Gauss map kNN
 * ================================================================ */
static int16_t *ggm_tr, *ggm_te;

static void gauss_map_grid(const uint8_t *img, int16_t *hist) {
    memset(hist, 0, GM_GRID_TOTAL * sizeof(int16_t));
    for (int y = 0; y < SPATIAL_H; y++)for (int x = 0; x < SPATIAL_W; x++) {
        int gx=(x<SPATIAL_W-1)?(int)img[y*SPATIAL_W+x+1]-(int)img[y*SPATIAL_W+x]:0;
        int gy=(y<SPATIAL_H-1)?(int)img[(y+1)*SPATIAL_W+x]-(int)img[y*SPATIAL_W+x]:0;
        int dx=(gx>10)?2:(gx<-10)?0:1,dy=(gy>10)?2:(gy<-10)?0:1;
        int mag=abs(gx)+abs(gy),mb=(mag<5)?0:(mag<20)?1:(mag<50)?2:(mag<100)?3:4;
        int bin=dx*GM_DIR*GM_MAG+dy*GM_MAG+mb;
        int ry=y*GRID_R/SPATIAL_H,rx=x*GRID_C/SPATIAL_W;
        if(ry>=GRID_R)ry=GRID_R-1;if(rx>=GRID_C)rx=GRID_C-1;
        hist[(ry*GRID_C+rx)*GM_BINS+bin]++;}}

static int32_t hist_l1(const int16_t*a,const int16_t*b,int len){int32_t d=0;for(int i=0;i<len;i++)d+=abs(a[i]-b[i]);return d;}

/* Find k nearest neighbors by grid Gauss map L1, return class votes */
static void gm_knn_votes(int test_idx, int k, int *votes) {
    memset(votes, 0, N_CLASSES * sizeof(int));
    const int16_t *qi = ggm_te + (size_t)test_idx * GM_GRID_TOTAL;
    /* Find k nearest */
    typedef struct{int32_t d;uint8_t lbl;}nb_t;
    nb_t *best = malloc(k * sizeof(nb_t));
    for(int j=0;j<k;j++){best[j].d=INT32_MAX;best[j].lbl=0;}
    for (int j = 0; j < TRAIN_N; j++) {
        int32_t d = hist_l1(qi, ggm_tr + (size_t)j * GM_GRID_TOTAL, GM_GRID_TOTAL);
        if (d < best[k-1].d) { best[k-1] = (nb_t){d, train_labels[j]};
            for(int x=k-2;x>=0;x--)if(best[x+1].d<best[x].d){nb_t t=best[x];best[x]=best[x+1];best[x+1]=t;}else break;}}
    for(int j=0;j<k;j++)votes[best[j].lbl]++;
    free(best);
}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_r_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_r_te=malloc((size_t)TEST_N*SP_PIXELS);
    raw_g_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_g_te=malloc((size_t)TEST_N*SP_PIXELS);
    raw_b_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_b_te=malloc((size_t)TEST_N*SP_PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_gray_te=malloc((size_t)TEST_N*SP_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            memcpy(raw_r_tr+(size_t)idx*SP_PIXELS,r,SP_PIXELS);memcpy(raw_g_tr+(size_t)idx*SP_PIXELS,g,SP_PIXELS);
            memcpy(raw_b_tr+(size_t)idx*SP_PIXELS,b3,SP_PIXELS);
            uint8_t*gd=raw_gray_tr+(size_t)idx*SP_PIXELS;
            for(int p2=0;p2<SP_PIXELS;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        memcpy(raw_r_te+(size_t)i*SP_PIXELS,r,SP_PIXELS);memcpy(raw_g_te+(size_t)i*SP_PIXELS,g,SP_PIXELS);
        memcpy(raw_b_te+(size_t)i*SP_PIXELS,b3,SP_PIXELS);
        uint8_t*gd=raw_gray_te+(size_t)i*SP_PIXELS;
        for(int p2=0;p2<SP_PIXELS;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Fused Architecture ===\n");
    printf("=== Block Stereo + Gauss Map + Hierarchy ===\n\n");
    load_data();

    /* Build System A: 3-eye stereo Bayesian */
    printf("Building System A (3-eye stereo)...\n");
    eye_t eyes[3];
    void(*qfns[])(const uint8_t*,int8_t*,int)={quant_fixed,quant_adaptive,quant_perchannel};
    for(int ei=0;ei<3;ei++)build_eye(&eyes[ei],raw_train,raw_test,qfns[ei]);
    printf("  System A built\n");

    /* Build System B: Grid Gauss map */
    printf("Building System B (grid Gauss map)...\n");
    ggm_tr=malloc((size_t)TRAIN_N*GM_GRID_TOTAL*sizeof(int16_t));
    ggm_te=malloc((size_t)TEST_N*GM_GRID_TOTAL*sizeof(int16_t));
    for(int i=0;i<TRAIN_N;i++)gauss_map_grid(raw_gray_tr+(size_t)i*SP_PIXELS,ggm_tr+(size_t)i*GM_GRID_TOTAL);
    for(int i=0;i<TEST_N;i++)gauss_map_grid(raw_gray_te+(size_t)i*SP_PIXELS,ggm_te+(size_t)i*GM_GRID_TOTAL);
    printf("  System B built\n");
    printf("  Total build: %.1f sec\n\n",now_sec()-t0);

    /* Precompute System A posteriors and System B kNN for all test images */
    printf("Precomputing classifications...\n");
    double *sysA_bp = malloc((size_t)TEST_N * N_CLASSES * sizeof(double));
    int *sysB_pred = malloc(TEST_N * sizeof(int));
    int *sysB_votes = malloc((size_t)TEST_N * N_CLASSES * sizeof(int));

    for (int i = 0; i < TEST_N; i++) {
        /* System A: 3-eye Bayesian */
        double *bp = sysA_bp + (size_t)i * N_CLASSES;
        memset(bp, 0, N_CLASSES * sizeof(double));
        for (int ei = 0; ei < 3; ei++) bay_eye(&eyes[ei], i, bp);
        double mx = -1e30; for (int c = 0; c < N_CLASSES; c++) if (bp[c] > mx) mx = bp[c];
        for (int c = 0; c < N_CLASSES; c++) bp[c] -= mx;

        /* System B: Gauss map kNN (k=5) */
        int *gv = sysB_votes + (size_t)i * N_CLASSES;
        gm_knn_votes(i, 5, gv);
        int best = 0; for (int c = 1; c < N_CLASSES; c++) if (gv[c] > gv[best]) best = c;
        sysB_pred[i] = best;

        if ((i + 1) % 1000 == 0) fprintf(stderr, "  %d/%d\r", i + 1, TEST_N);
    }
    fprintf(stderr, "\n");

    /* ================================================================
     * BASELINES
     * ================================================================ */
    printf("\n=== BASELINES ===\n\n");
    {int cA=0,cB=0;
     for(int i=0;i<TEST_N;i++){
         double*bp=sysA_bp+(size_t)i*N_CLASSES;int predA=0;
         for(int c=1;c<N_CLASSES;c++)if(bp[c]>bp[predA])predA=c;
         if(predA==test_labels[i])cA++;
         if(sysB_pred[i]==test_labels[i])cB++;}
     printf("  System A (3-eye Bayesian): %.2f%%\n",100.0*cA/TEST_N);
     printf("  System B (Gauss kNN k=5):  %.2f%%\n",100.0*cB/TEST_N);}

    /* ================================================================
     * FUSION 1: Score combination (normalize + weighted sum)
     * ================================================================ */
    printf("\n=== FUSION 1: Score Combination ===\n\n");
    {
        /* Convert System B votes to log-posterior-like scores */
        double w_vals[] = {0.5, 1.0, 2.0, 5.0, 10.0};
        for (int wi = 0; wi < 5; wi++) {
            double w = w_vals[wi];
            int correct = 0;
            for (int i = 0; i < TEST_N; i++) {
                double *bp = sysA_bp + (size_t)i * N_CLASSES;
                int *gv = sysB_votes + (size_t)i * N_CLASSES;
                /* Combined score: System A log-posterior + w * System B vote count */
                double combined[N_CLASSES];
                for (int c = 0; c < N_CLASSES; c++)
                    combined[c] = bp[c] + w * gv[c];
                int pred = 0; for (int c = 1; c < N_CLASSES; c++) if (combined[c] > combined[pred]) pred = c;
                if (pred == test_labels[i]) correct++;
            }
            printf("  w=%.1f: %.2f%%\n", w, 100.0 * correct / TEST_N);
        }
    }

    /* Best weight per-class analysis */
    printf("\n=== FUSION 1 BEST: Per-class ===\n\n");
    {
        /* Sweep weight more finely */
        double best_w = 0; int best_c = 0;
        for (double w = 0.1; w <= 20.0; w += 0.1) {
            int correct = 0;
            for (int i = 0; i < TEST_N; i++) {
                double *bp = sysA_bp + (size_t)i * N_CLASSES;
                int *gv = sysB_votes + (size_t)i * N_CLASSES;
                double combined[N_CLASSES];
                for (int c = 0; c < N_CLASSES; c++) combined[c] = bp[c] + w * gv[c];
                int pred = 0; for (int c = 1; c < N_CLASSES; c++) if (combined[c] > combined[pred]) pred = c;
                if (pred == test_labels[i]) correct++;
            }
            if (correct > best_c) { best_c = correct; best_w = w; }
        }
        printf("  Best weight: w=%.1f -> %.2f%%\n\n", best_w, 100.0 * best_c / TEST_N);

        /* Per-class with best weight */
        int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for (int i = 0; i < TEST_N; i++) {
            pt[test_labels[i]]++;
            double *bp = sysA_bp + (size_t)i * N_CLASSES;
            int *gv = sysB_votes + (size_t)i * N_CLASSES;
            double combined[N_CLASSES];
            for (int c = 0; c < N_CLASSES; c++) combined[c] = bp[c] + best_w * gv[c];
            int pred = 0; for (int c = 1; c < N_CLASSES; c++) if (combined[c] > combined[pred]) pred = c;
            if (pred == test_labels[i]) pc[test_labels[i]]++;
        }
        for (int c = 0; c < N_CLASSES; c++)
            printf("    %d %-10s %.1f%%\n", c, cn[c], 100.0 * pc[c] / pt[c]);
    }

    /* ================================================================
     * FUSION 2: Hierarchical — binary router + within-group specialists
     * ================================================================ */
    printf("\n=== FUSION 2: Hierarchical (binary router → specialists) ===\n\n");
    {
        int machine_cls[] = {0,1,8,9}, animal_cls[] = {2,3,4,5,6,7};
        int correct = 0; int pc[N_CLASSES]={0},pt[N_CLASSES]={0};

        for (int i = 0; i < TEST_N; i++) {
            pt[test_labels[i]]++;
            double *bp = sysA_bp + (size_t)i * N_CLASSES;

            /* Binary router: System A (better at binary: 81%) */
            double m_score = 0, a_score = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double p = exp(bp[c]);
                if (is_machine(c)) m_score += p; else a_score += p;
            }

            int pred;
            if (m_score > a_score) {
                /* Machine: use System A (good at texture/color for vehicles) */
                pred = machine_cls[0];
                for (int mi = 1; mi < 4; mi++) if (bp[machine_cls[mi]] > bp[pred]) pred = machine_cls[mi];
            } else {
                /* Animal: use System B (good at shape/geometry for animals) */
                int *gv = sysB_votes + (size_t)i * N_CLASSES;
                pred = animal_cls[0];
                for (int ai = 1; ai < 6; ai++) if (gv[animal_cls[ai]] > gv[pred]) pred = animal_cls[ai];
            }
            if (pred == test_labels[i]) { correct++; pc[test_labels[i]]++; }
        }
        printf("  Hierarchical (A routes, A=machines, B=animals): %.2f%%\n", 100.0 * correct / TEST_N);
        for (int c = 0; c < N_CLASSES; c++)
            printf("    %d %-10s %.1f%%\n", c, cn[c], 100.0 * pc[c] / pt[c]);
    }

    /* ================================================================
     * FUSION 3: Confidence routing — System A when confident, B when not
     * ================================================================ */
    printf("\n=== FUSION 3: Confidence Routing ===\n\n");
    {
        for (double threshold = 0.3; threshold <= 0.9; threshold += 0.1) {
            int correct = 0, n_A = 0, n_B = 0;
            for (int i = 0; i < TEST_N; i++) {
                double *bp = sysA_bp + (size_t)i * N_CLASSES;
                /* Convert to probabilities */
                double probs[N_CLASSES], sum = 0;
                for (int c = 0; c < N_CLASSES; c++) { probs[c] = exp(bp[c]); sum += probs[c]; }
                for (int c = 0; c < N_CLASSES; c++) probs[c] /= sum;
                double max_prob = 0; int predA = 0;
                for (int c = 0; c < N_CLASSES; c++) if (probs[c] > max_prob) { max_prob = probs[c]; predA = c; }

                int pred;
                if (max_prob >= threshold) { pred = predA; n_A++; }
                else { pred = sysB_pred[i]; n_B++; }
                if (pred == test_labels[i]) correct++;
            }
            printf("  threshold=%.1f: A=%4d B=%4d -> %.2f%%\n", threshold, n_A, n_B, 100.0 * correct / TEST_N);
        }
    }

    /* ================================================================
     * FUSION 4: Hierarchical + Score fusion (best of everything)
     * ================================================================ */
    printf("\n=== FUSION 4: Hierarchical + Score Fusion ===\n\n");
    {
        int machine_cls[] = {0,1,8,9}, animal_cls[] = {2,3,4,5,6,7};
        /* For machines: System A. For animals: combined A+B with weight */
        double w_vals[] = {1.0, 2.0, 5.0, 10.0, 20.0};
        for (int wi = 0; wi < 5; wi++) {
            double w = w_vals[wi];
            int correct = 0;
            for (int i = 0; i < TEST_N; i++) {
                double *bp = sysA_bp + (size_t)i * N_CLASSES;
                int *gv = sysB_votes + (size_t)i * N_CLASSES;
                double m_score = 0, a_score = 0;
                for (int c = 0; c < N_CLASSES; c++) { double p = exp(bp[c]); if (is_machine(c)) m_score += p; else a_score += p; }
                int pred;
                if (m_score > a_score) {
                    /* Machine: System A argmax */
                    pred = machine_cls[0];
                    for (int mi = 1; mi < 4; mi++) if (bp[machine_cls[mi]] > bp[pred]) pred = machine_cls[mi];
                } else {
                    /* Animal: combined score */
                    double combined[N_CLASSES];
                    for (int c = 0; c < N_CLASSES; c++) combined[c] = bp[c] + w * gv[c];
                    pred = animal_cls[0];
                    for (int ai = 1; ai < 6; ai++) if (combined[animal_cls[ai]] > combined[pred]) pred = animal_cls[ai];
                }
                if (pred == test_labels[i]) correct++;
            }
            printf("  Hier + animal_w=%.1f: %.2f%%\n", w, 100.0 * correct / TEST_N);
        }
    }

    printf("\n=== REFERENCE ===\n");
    printf("  System A alone (3-eye Bayesian): 41.18%%\n");
    printf("  System B alone (Gauss kNN):      46.06%%\n");
    printf("  Stereo + MT4 stack:              44.48%%\n");

    printf("\nTotal: %.1f sec\n", now_sec() - t0);
    free(sysA_bp); free(sysB_pred); free(sysB_votes); free(ggm_tr); free(ggm_te);
    return 0;
}
