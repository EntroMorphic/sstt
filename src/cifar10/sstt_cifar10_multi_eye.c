/*
 * sstt_cifar10_multi_eye.c — Multi-Threshold (Multi-Eye) Ensemble Retrieval
 *
 * Extends the 3-eye cascade pipeline from sstt_cifar10_cascade_gauss.c
 * with 4 additional quantization "eyes" for a total of 7.
 *
 * Eyes:
 *   1. Fixed (85/170)
 *   2. Adaptive (P33/P67 per-image)
 *   3. Per-channel adaptive (P33/P67 per channel)
 *   4. Wide-zero fixed (64/192)
 *   5. Narrow-zero fixed (96/160)
 *   6. Aggressive adaptive (P10/P90 per-image)
 *   7. Per-channel wide (P25/P75 per channel)
 *
 * Tests:
 *   1. Baseline 3-eye cascade (reproduce 50.18%)
 *   2. 5-eye cascade (eyes 1-5)
 *   3. 7-eye cascade (all 7 eyes)
 *   4. Ablation — recall and accuracy for 3, 5, 7 eyes
 *
 * Ranking: RGB 4x4 Gauss map (GC_FEAT = 2880, 4-channel)
 *
 * Build: gcc -O3 -mavx2 -march=native -Wall -Wextra -o build/sstt_cifar10_multi_eye src/sstt_cifar10_multi_eye.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 10000
#define TEST_N  200
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
#define IG_SCALE 16
#define TOP_K 200
#define SP_W 32
#define SP_H 32
#define SP_PX 1024

/* Gauss map parameters */
#define GM_DIR 3
#define GM_MAG 5
#define GM_BINS (GM_DIR*GM_DIR*GM_MAG) /* 45 */

/* RGB 4x4 Gauss map (the 50.18% config) */
#define GA_GRID 4
#define GA_FEAT (GA_GRID*GA_GRID*GM_BINS) /* 720 */
#define GC_CH 4
#define GC_FEAT (GC_CH*GA_GRID*GA_GRID*GM_BINS) /* 2880 */

#define MAX_EYES 7

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static uint8_t *raw_r_tr,*raw_r_te,*raw_g_tr,*raw_g_te,*raw_b_tr,*raw_b_te,*raw_gray_tr,*raw_gray_te;

/* ================================================================
 *  Block-based voting infrastructure
 * ================================================================ */
typedef struct{uint8_t *joint_tr,*joint_te;uint32_t *hot;uint16_t ig[N_BLOCKS];uint8_t bg;
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];uint16_t idx_sz[N_BLOCKS][BYTE_VALS];uint32_t *idx_pool;
    uint8_t nbr[BYTE_VALS][8];}eye_t;

static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}

static void build_eye(eye_t*e,const uint8_t*raw_tr2,const uint8_t*raw_te2,void(*qfn)(const uint8_t*,int8_t*,int)){
    int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
    qfn(raw_tr2,ttr,TRAIN_N);qfn(raw_te2,tte,TEST_N);
    int8_t *hgr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*hge=aligned_alloc(32,(size_t)TEST_N*PADDED);
    int8_t *vgr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*vge=aligned_alloc(32,(size_t)TEST_N*PADDED);
    for(int i=0;i<TRAIN_N;i++){const int8_t*ti=ttr+(size_t)i*PADDED;int8_t*hi=hgr+(size_t)i*PADDED,*vi=vgr+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    for(int i=0;i<TEST_N;i++){const int8_t*ti=tte+(size_t)i*PADDED;int8_t*hi=hge+(size_t)i*PADDED,*vi=vge+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}
        for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);}
    /* Block sigs, transitions, Encoding D */
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *hst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*hse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    uint8_t *vst=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*vse=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=ttr+(size_t)i*PADDED;uint8_t*si=pxt+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=tte+(size_t)i*PADDED;uint8_t*si=pxe+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=hgr+(size_t)i*PADDED;uint8_t*si=hst+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=hge+(size_t)i*PADDED;uint8_t*si=hse+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TRAIN_N;i++){const int8_t*im=vgr+(size_t)i*PADDED;uint8_t*si=vst+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){const int8_t*im=vge+(size_t)i*PADDED;uint8_t*si=vse+(size_t)i*SIG_PAD;for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    uint8_t *htt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*hte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    uint8_t *vtt=aligned_alloc(32,(size_t)TRAIN_N*TRANS_PAD),*vte=aligned_alloc(32,(size_t)TEST_N*TRANS_PAD);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*s=pxt+(size_t)i*SIG_PAD;uint8_t*h=htt+(size_t)i*TRANS_PAD,*v=vtt+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}
    for(int i=0;i<TEST_N;i++){const uint8_t*s=pxe+(size_t)i*SIG_PAD;uint8_t*h=hte+(size_t)i*TRANS_PAD,*v=vte+(size_t)i*TRANS_PAD;
        for(int y=0;y<IMG_H;y++)for(int ss=0;ss<H_TRANS_PER_ROW;ss++)h[y*H_TRANS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[y*BLKS_PER_ROW+ss+1]);memset(h+N_HTRANS,0xFF,TRANS_PAD-N_HTRANS);
        for(int y=0;y<V_TRANS_PER_COL;y++)for(int ss=0;ss<BLKS_PER_ROW;ss++)v[y*BLKS_PER_ROW+ss]=te2(s[y*BLKS_PER_ROW+ss],s[(y+1)*BLKS_PER_ROW+ss]);memset(v+N_VTRANS,0xFF,TRANS_PAD-N_VTRANS);}
    e->joint_tr=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD);e->joint_te=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*pi=pxt+(size_t)i*SIG_PAD,*hi2=hst+(size_t)i*SIG_PAD,*vi2=vst+(size_t)i*SIG_PAD;
        const uint8_t*hti=htt+(size_t)i*TRANS_PAD,*vti=vtt+(size_t)i*TRANS_PAD;uint8_t*oi=e->joint_tr+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);int hs=((hi2[k]/9)-1)+(((hi2[k]/3)%3)-1)+((hi2[k]%3)-1);
            int vs=((vi2[k]/9)-1)+(((vi2[k]/3)%3)-1)+((vi2[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}
    for(int i=0;i<TEST_N;i++){const uint8_t*pi=pxe+(size_t)i*SIG_PAD,*hi2=hse+(size_t)i*SIG_PAD,*vi2=vse+(size_t)i*SIG_PAD;
        const uint8_t*hti=hte+(size_t)i*TRANS_PAD,*vti=vte+(size_t)i*TRANS_PAD;uint8_t*oi=e->joint_te+(size_t)i*SIG_PAD;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int k=y*BLKS_PER_ROW+s;
            uint8_t htb=(s>0)?hti[y*H_TRANS_PER_ROW+(s-1)]:BG_TRANS,vtb=(y>0)?vti[(y-1)*BLKS_PER_ROW+s]:BG_TRANS;
            int ps=((pi[k]/9)-1)+(((pi[k]/3)%3)-1)+((pi[k]%3)-1);int hs=((hi2[k]/9)-1)+(((hi2[k]/3)%3)-1)+((hi2[k]%3)-1);
            int vs=((vi2[k]/9)-1)+(((vi2[k]/3)%3)-1)+((vi2[k]%3)-1);
            uint8_t pc=ps<0?0:ps==0?1:ps<3?2:3,hc=hs<0?0:hs==0?1:hs<3?2:3,vc=vs<0?0:vs==0?1:vs<3?2:3;
            oi[k]=pc|(hc<<2)|(vc<<4)|((htb!=BG_TRANS)?1<<6:0)|((vtb!=BG_TRANS)?1<<7:0);}}
    free(ttr);free(tte);free(hgr);free(hge);free(vgr);free(vge);free(pxt);free(pxe);free(hst);free(hse);free(vst);free(vse);free(htt);free(hte);free(vtt);free(vte);
    /* BG, hot map, IG, index */
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    e->hot=aligned_alloc(32,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);memset(e->hot,0,(size_t)N_BLOCKS*BYTE_VALS*CLS_PAD*4);
    for(int i=0;i<TRAIN_N;i++){int l=train_labels[i];const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;
        for(int k=0;k<N_BLOCKS;k++)e->hot[(size_t)k*BYTE_VALS*CLS_PAD+(size_t)sig[k]*CLS_PAD+l]++;}
    {int cc[N_CLASSES]={0};for(int i=0;i<TRAIN_N;i++)cc[train_labels[i]]++;
     double hc=0;for(int c=0;c<N_CLASSES;c++){double p=(double)cc[c]/TRAIN_N;if(p>0)hc-=p*log2(p);}
     double raw2[N_BLOCKS],mx=0;for(int k=0;k<N_BLOCKS;k++){double hcond=0;
         for(int v=0;v<BYTE_VALS;v++){if((uint8_t)v==e->bg)continue;const uint32_t*h=e->hot+(size_t)k*BYTE_VALS*CLS_PAD+(size_t)v*CLS_PAD;
             int vt=0;for(int c=0;c<N_CLASSES;c++)vt+=(int)h[c];if(!vt)continue;double pv=(double)vt/TRAIN_N,hv=0;
             for(int c=0;c<N_CLASSES;c++){double pc2=(double)h[c]/vt;if(pc2>0)hv-=pc2*log2(pc2);}hcond+=pv*hv;}
         raw2[k]=hc-hcond;if(raw2[k]>mx)mx=raw2[k];}
     for(int k=0;k<N_BLOCKS;k++){e->ig[k]=mx>0?(uint16_t)(raw2[k]/mx*IG_SCALE+0.5):1;if(!e->ig[k])e->ig[k]=1;}}
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)e->nbr[v][b3]=(uint8_t)(v^(1<<b3));
    memset(e->idx_sz,0,sizeof(e->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc((size_t)tot*4);uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,e->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
}

static void vote_eye(const eye_t*e,int img,uint32_t*votes){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        uint16_t w=e->ig[k],wh=w>1?w/2:1;
        {uint32_t off=e->idx_off[k][bv];uint16_t sz=e->idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)votes[e->idx_pool[off+j]]+=w;}
        for(int nb=0;nb<8;nb++){uint8_t nv=e->nbr[bv][nb];if(nv==e->bg)continue;
            uint32_t noff=e->idx_off[k][nv];uint16_t nsz=e->idx_sz[k][nv];for(uint16_t j=0;j<nsz;j++)votes[e->idx_pool[noff+j]]+=wh;}}}

/* ================================================================
 *  Quantization functions — original 3 eyes
 * ================================================================ */
static void quant_fixed(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}

static void quant_adaptive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];
        if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}
    free(sorted);}

static void quant_perchannel(const uint8_t*s,int8_t*d,int n){
    uint8_t*cv=malloc(1024);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        uint8_t p33[3],p67[3];
        for(int ch=0;ch<3;ch++){int cnt=0;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++)cv[cnt++]=si[y*96+x*3+ch];
            qsort(cv,cnt,1,cmp_u8);p33[ch]=cv[cnt/3];p67[ch]=cv[2*cnt/3];
            if(p33[ch]==p67[ch]){p33[ch]=p33[ch]>0?p33[ch]-1:0;p67[ch]=p67[ch]<255?p67[ch]+1:255;}}
        for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int ch=0;ch<3;ch++){
            int idx=y*96+x*3+ch;di[idx]=si[idx]>p67[ch]?1:si[idx]<p33[ch]?-1:0;}}
    free(cv);}

/* ================================================================
 *  Quantization functions — 4 new eyes
 * ================================================================ */

/* Eye 4: Wide-zero fixed (64/192) */
static void quant_fixed_64_192(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>192?1:si[i]<64?-1:0;}}

/* Eye 5: Narrow-zero fixed (96/160) */
static void quant_fixed_96_160(const uint8_t*s,int8_t*d,int n){
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>160?1:si[i]<96?-1:0;}}

/* Eye 6: Aggressive adaptive (P10/P90 per-image) */
static void quant_adaptive_aggressive(const uint8_t*s,int8_t*d,int n){
    uint8_t*sorted=malloc(PIXELS);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);
        uint8_t p10=sorted[PIXELS/10],p90=sorted[9*PIXELS/10];
        if(p10==p90){p10=p10>0?p10-1:0;p90=p90<255?p90+1:255;}
        for(int i=0;i<PIXELS;i++)di[i]=si[i]>p90?1:si[i]<p10?-1:0;}
    free(sorted);}

/* Eye 7: Per-channel wide (P25/P75 per channel) */
static void quant_perchannel_wide(const uint8_t*s,int8_t*d,int n){
    uint8_t*cv=malloc(1024);
    for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;
        uint8_t p25[3],p75[3];
        for(int ch=0;ch<3;ch++){int cnt=0;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++)cv[cnt++]=si[y*96+x*3+ch];
            qsort(cv,cnt,1,cmp_u8);p25[ch]=cv[cnt/4];p75[ch]=cv[3*cnt/4];
            if(p25[ch]==p75[ch]){p25[ch]=p25[ch]>0?p25[ch]-1:0;p75[ch]=p75[ch]<255?p75[ch]+1:255;}}
        for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int ch=0;ch<3;ch++){
            int idx=y*96+x*3+ch;di[idx]=si[idx]>p75[ch]?1:si[idx]<p25[ch]?-1:0;}}
    free(cv);}

/* ================================================================
 *  Candidate selection and Gauss map infrastructure
 * ================================================================ */
typedef struct{uint32_t id;uint32_t votes;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

static void gauss_map_grid_ch(const uint8_t *img, int16_t *hist, int gr, int gc, int w, int h) {
    int nbins = GM_BINS, nreg = gr * gc;
    memset(hist, 0, (size_t)nreg * nbins * sizeof(int16_t));
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        int gx = (x < w-1) ? (int)img[y*w+x+1] - (int)img[y*w+x] : 0;
        int gy = (y < h-1) ? (int)img[(y+1)*w+x] - (int)img[y*w+x] : 0;
        int dx = (gx>10)?2:(gx<-10)?0:1, dy = (gy>10)?2:(gy<-10)?0:1;
        int mag = abs(gx)+abs(gy), mb = (mag<5)?0:(mag<20)?1:(mag<50)?2:(mag<100)?3:4;
        int bin = dx*GM_DIR*GM_MAG + dy*GM_MAG + mb;
        int ry = y*gr/h, rx = x*gc/w;
        if(ry>=gr)ry=gr-1; if(rx>=gc)rx=gc-1;
        hist[(ry*gc+rx)*nbins + bin]++;
    }
}

static int32_t hist_l1(const int16_t*a,const int16_t*b,int len){int32_t d=0;for(int i=0;i<len;i++)d+=abs(a[i]-b[i]);return d;}

/* ================================================================
 *  Data loading
 * ================================================================ */
static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_r_tr=malloc((size_t)TRAIN_N*SP_PX);raw_r_te=malloc((size_t)TEST_N*SP_PX);
    raw_g_tr=malloc((size_t)TRAIN_N*SP_PX);raw_g_te=malloc((size_t)TEST_N*SP_PX);
    raw_b_tr=malloc((size_t)TRAIN_N*SP_PX);raw_b_te=malloc((size_t)TEST_N*SP_PX);
    raw_gray_tr=malloc((size_t)TRAIN_N*SP_PX);raw_gray_te=malloc((size_t)TEST_N*SP_PX);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);char p[512];uint8_t rec[3073];
    int train_idx=0;
    for(int b2=1;b2<=5 && train_idx < TRAIN_N;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb"); if(!f) continue;
        for(int i=0;i<10000 && train_idx < TRAIN_N;i++){if(fread(rec,1,3073,f)!=3073) break;
            int idx=train_idx++; train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            memcpy(raw_r_tr+(size_t)idx*SP_PX,r,SP_PX);memcpy(raw_g_tr+(size_t)idx*SP_PX,g,SP_PX);memcpy(raw_b_tr+(size_t)idx*SP_PX,b3,SP_PX);
            uint8_t*gd=raw_gray_tr+(size_t)idx*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb"); if(f){
        for(int i=0;i<TEST_N;i++){if(fread(rec,1,3073,f)!=3073) break;
            test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            memcpy(raw_r_te+(size_t)i*SP_PX,r,SP_PX);memcpy(raw_g_te+(size_t)i*SP_PX,g,SP_PX);memcpy(raw_b_te+(size_t)i*SP_PX,b3,SP_PX);
            uint8_t*gd=raw_gray_te+(size_t)i*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
}

/* ================================================================
 *  Cascade test helper: run N-eye vote → Gauss rank → kNN
 * ================================================================ */
static void run_cascade(const char *name, eye_t *eyes, int neyes,
                        const int16_t *gc_tr, const int16_t *gc_te,
                        int *out_recall, int out_correct[4]) {
    int kvals[]={1,3,5,7};
    int recall=0;
    int correct[4]={0};
    uint32_t *vbuf=calloc(TRAIN_N,4);

    for(int i=0;i<TEST_N;i++){
        memset(vbuf,0,TRAIN_N*4);
        for(int ei=0;ei<neyes;ei++)vote_eye(&eyes[ei],i,vbuf);
        cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);

        /* Check recall */
        for(int j=0;j<nc;j++)if(train_labels[cands[j].id]==test_labels[i]){recall++;break;}

        /* Gauss map L1 re-rank */
        const int16_t*qi=gc_te+(size_t)i*GC_FEAT;
        int32_t best_d[TOP_K];
        for(int j=0;j<nc;j++)best_d[j]=hist_l1(qi,gc_tr+(size_t)cands[j].id*GC_FEAT,GC_FEAT);

        /* Insertion sort by Gauss distance */
        for(int j=1;j<nc;j++){int32_t kd=best_d[j];cand_t kc=cands[j];int x=j-1;
            while(x>=0&&best_d[x]>kd){best_d[x+1]=best_d[x];cands[x+1]=cands[x];x--;}
            best_d[x+1]=kd;cands[x+1]=kc;}

        /* kNN vote */
        for(int ki=0;ki<4;ki++){
            int v[N_CLASSES]={0};int kk=kvals[ki]<nc?kvals[ki]:nc;
            for(int j=0;j<kk;j++)v[train_labels[cands[j].id]]++;
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
            if(pred==test_labels[i])correct[ki]++;}

        if((i+1)%1000==0)fprintf(stderr,"  %s: %d/%d\r",name,i+1,TEST_N);
    }
    fprintf(stderr,"\n");
    free(vbuf);

    *out_recall=recall;
    for(int ki=0;ki<4;ki++)out_correct[ki]=correct[ki];
}

/* ================================================================
 *  Main
 * ================================================================ */
int main(int argc,char**argv){
    setvbuf(stdout, NULL, _IONBF, 0); // Disable buffering
    printf("--- CIFAR-10 Multi-Eye: Minimal Diagnostic ---\n");
    if(argc>1)data_dir=argv[1];
    
    double t0=now_sec();
    load_data();
    printf("  [1] Data Loaded (Train: %d, Test: %d)\n", TRAIN_N, TEST_N);

    printf("  Data loaded (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * Compute RGB 4x4 Gauss maps (the 50.18% config)
     * ================================================================ */
    printf("Computing RGB 4x4 Gauss maps (GC_FEAT=%d)...\n",GC_FEAT);
    int16_t *gc_tr=malloc((size_t)TRAIN_N*GC_FEAT*2),*gc_te=malloc((size_t)TEST_N*GC_FEAT*2);
    {const uint8_t *ch_tr[]={raw_gray_tr,raw_r_tr,raw_g_tr,raw_b_tr};
     const uint8_t *ch_te[]={raw_gray_te,raw_r_te,raw_g_te,raw_b_te};
     for(int i=0;i<TRAIN_N;i++)for(int ch=0;ch<GC_CH;ch++)
         gauss_map_grid_ch(ch_tr[ch]+(size_t)i*SP_PX,gc_tr+(size_t)i*GC_FEAT+ch*GA_FEAT,GA_GRID,GA_GRID,SP_W,SP_H);
     for(int i=0;i<TEST_N;i++)for(int ch=0;ch<GC_CH;ch++)
         gauss_map_grid_ch(ch_te[ch]+(size_t)i*SP_PX,gc_te+(size_t)i*GC_FEAT+ch*GA_FEAT,GA_GRID,GA_GRID,SP_W,SP_H);}
    printf("  Gauss maps computed (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * Build all 7 eyes
     * ================================================================ */
    printf("Building 7 eyes...\n");
    eye_t *eyes = calloc(MAX_EYES, sizeof(eye_t));
    if (!eyes) { fprintf(stderr, "ERROR: eyes alloc\n"); exit(1); }
    const char *eye_names[MAX_EYES]={
        "Fixed 85/170",
        "Adaptive P33/P67",
        "Per-ch P33/P67",
        "Fixed 64/192",
        "Fixed 96/160",
        "Adaptive P10/P90",
        "Per-ch P25/P75"
    };
    void(*qfns[MAX_EYES])(const uint8_t*,int8_t*,int)={
        quant_fixed,
        quant_adaptive,
        quant_perchannel,
        quant_fixed_64_192,
        quant_fixed_96_160,
        quant_adaptive_aggressive,
        quant_perchannel_wide
    };
    for(int ei=0;ei<MAX_EYES;ei++){
        double et=now_sec();
        build_eye(&eyes[ei],raw_train,raw_test,qfns[ei]);
        printf("  Eye %d (%s) built (%.1f sec)\n",ei+1,eye_names[ei],now_sec()-et);
    }
    printf("  All eyes built (%.1f sec total)\n\n",now_sec()-t0);

    /* ================================================================
     * Test 1: Baseline 3-eye cascade (reproduce 50.18%)
     * ================================================================ */
    printf("=== TEST 1: Baseline 3-Eye Cascade (reproduce 50.18%%) ===\n");
    printf("  Eyes: 1-3 (Fixed, Adaptive, Per-channel)\n");
    printf("  Pipeline: 3-eye stereo vote -> top-200 -> RGB 4x4 Gauss L1 re-rank -> kNN\n\n");
    {
        int recall, correct[4];
        run_cascade("3-eye baseline",eyes,3,gc_tr,gc_te,&recall,correct);
        printf("  Recall (top-200): %.2f%%\n",100.0*recall/TEST_N);
        printf("  k=1: %.2f%%  k=3: %.2f%%  k=5: %.2f%%  k=7: %.2f%%\n\n",
               100.0*correct[0]/TEST_N,100.0*correct[1]/TEST_N,
               100.0*correct[2]/TEST_N,100.0*correct[3]/TEST_N);
    }

    /* ================================================================
     * Test 2: 5-eye cascade (eyes 1-5)
     * ================================================================ */
    printf("=== TEST 2: 5-Eye Cascade ===\n");
    printf("  Eyes: 1-5 (+ Wide-zero 64/192, Narrow-zero 96/160)\n");
    printf("  Pipeline: 5-eye stereo vote -> top-200 -> RGB 4x4 Gauss L1 re-rank -> kNN\n\n");
    {
        int recall, correct[4];
        run_cascade("5-eye",eyes,5,gc_tr,gc_te,&recall,correct);
        printf("  Recall (top-200): %.2f%%\n",100.0*recall/TEST_N);
        printf("  k=1: %.2f%%  k=3: %.2f%%  k=5: %.2f%%  k=7: %.2f%%\n\n",
               100.0*correct[0]/TEST_N,100.0*correct[1]/TEST_N,
               100.0*correct[2]/TEST_N,100.0*correct[3]/TEST_N);
    }

    /* ================================================================
     * Test 3: 7-eye cascade (all 7 eyes)
     * ================================================================ */
    printf("=== TEST 3: 7-Eye Cascade ===\n");
    printf("  Eyes: 1-7 (all)\n");
    printf("  Pipeline: 7-eye stereo vote -> top-200 -> RGB 4x4 Gauss L1 re-rank -> kNN\n\n");
    {
        int recall, correct[4];
        run_cascade("7-eye",eyes,7,gc_tr,gc_te,&recall,correct);
        printf("  Recall (top-200): %.2f%%\n",100.0*recall/TEST_N);
        printf("  k=1: %.2f%%  k=3: %.2f%%  k=5: %.2f%%  k=7: %.2f%%\n\n",
               100.0*correct[0]/TEST_N,100.0*correct[1]/TEST_N,
               100.0*correct[2]/TEST_N,100.0*correct[3]/TEST_N);
    }

    /* ================================================================
     * Test 4: Ablation summary
     * ================================================================ */
    printf("=== TEST 4: Ablation Summary ===\n\n");
    printf("  %-12s %8s  %8s  %8s  %8s  %8s\n","Eyes","Recall","k=1","k=3","k=5","k=7");
    printf("  %-12s %8s  %8s  %8s  %8s  %8s\n","----","------","---","---","---","---");
    {
        int eye_counts[]={3,5,7};
        const char *labels[]={"3-eye","5-eye","7-eye"};
        for(int ti=0;ti<3;ti++){
            int recall, correct[4];
            run_cascade(labels[ti],eyes,eye_counts[ti],gc_tr,gc_te,&recall,correct);
            printf("  %-12s %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%\n",
                   labels[ti],100.0*recall/TEST_N,
                   100.0*correct[0]/TEST_N,100.0*correct[1]/TEST_N,
                   100.0*correct[2]/TEST_N,100.0*correct[3]/TEST_N);
        }
    }

    /* ================================================================
     * Per-class breakdown for best config (7-eye, k=5)
     * ================================================================ */
    printf("\n=== PER-CLASS: 7-Eye Cascade k=5 ===\n\n");
    {
        int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        uint32_t *vbuf=calloc(TRAIN_N,4);
        for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
            memset(vbuf,0,TRAIN_N*4);
            for(int ei=0;ei<MAX_EYES;ei++)vote_eye(&eyes[ei],i,vbuf);
            cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
            const int16_t*qi=gc_te+(size_t)i*GC_FEAT;
            int32_t bd[TOP_K];for(int j=0;j<nc;j++)bd[j]=hist_l1(qi,gc_tr+(size_t)cands[j].id*GC_FEAT,GC_FEAT);
            for(int j=1;j<nc;j++){int32_t kd=bd[j];cand_t kc=cands[j];int x=j-1;
                while(x>=0&&bd[x]>kd){bd[x+1]=bd[x];cands[x+1]=cands[x];x--;}bd[x+1]=kd;cands[x+1]=kc;}
            int v[N_CLASSES]={0};int kk=5<nc?5:nc;for(int j=0;j<kk;j++)v[train_labels[cands[j].id]]++;
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
            if(pred==test_labels[i])pc[test_labels[i]]++;
            if((i+1)%2000==0)fprintf(stderr,"  per-class: %d/%d\r",i+1,TEST_N);}
        fprintf(stderr,"\n");
        free(vbuf);
        int total_correct=0;for(int c=0;c<N_CLASSES;c++)total_correct+=pc[c];
        printf("  7-eye cascade k=5: %.2f%%\n",100.0*total_correct/TEST_N);
        for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%% (%d/%d)\n",c,cn[c],100.0*pc[c]/pt[c],pc[c],pt[c]);
    }

    /* ================================================================
     * Reference
     * ================================================================ */
    printf("\n=== REFERENCE ===\n");
    printf("  Brute Gauss kNN (gray 4x4):  48.31%%\n");
    printf("  3-eye cascade (RGB 4x4 k=5): 50.18%%\n");
    printf("  Target:                       51.00%%+\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
