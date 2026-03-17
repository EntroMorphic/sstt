/*
 * sstt_cifar10_cascade_gauss.c — Restore the Pipeline: Vote → Gauss Rank
 *
 * The SSTT architecture's power is in RETRIEVE → RANK separation.
 * The Gauss map broke this by doing brute kNN. Reconnect:
 *
 *   RETRIEVE: Block-based inverted-index voting (98% recall, 200 candidates)
 *   RANK: Grid Gauss map L1 distance (shape geometry, background-invariant)
 *
 * Tests:
 *   1. Single-eye vote → Gauss rank
 *   2. 3-eye stereo vote → Gauss rank
 *   3. RGB grid Gauss map (4-channel, not just grayscale)
 *   4. Finer grid (8×8 instead of 4×4)
 *   5. Full composition: stereo vote → RGB fine-grid Gauss rank
 *   6. Brute kNN baselines for comparison
 *
 * Target: >48.31% (brute Gauss kNN) → 50%
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_cascade_gauss src/sstt_cifar10_cascade_gauss.c -lm
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
#define IG_SCALE 16
#define TOP_K 200
#define SP_W 32
#define SP_H 32
#define SP_PX 1024

/* Gauss map parameters */
#define GM_DIR 3
#define GM_MAG 5
#define GM_BINS (GM_DIR*GM_DIR*GM_MAG) /* 45 */

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static uint8_t *raw_r_tr,*raw_r_te,*raw_g_tr,*raw_g_te,*raw_b_tr,*raw_b_te,*raw_gray_tr,*raw_gray_te;

/* ================================================================
 *  Block-based voting infrastructure (3 eyes)
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

static void quant_fixed(const uint8_t*s,int8_t*d,int n){for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}
static void quant_adaptive(const uint8_t*s,int8_t*d,int n){uint8_t*sorted=malloc(PIXELS);for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;memcpy(sorted,si,PIXELS);qsort(sorted,PIXELS,1,cmp_u8);uint8_t p33=sorted[PIXELS/3],p67=sorted[2*PIXELS/3];if(p33==p67){p33=p33>0?p33-1:0;p67=p67<255?p67+1:255;}for(int i=0;i<PIXELS;i++)di[i]=si[i]>p67?1:si[i]<p33?-1:0;}free(sorted);}
static void quant_perchannel(const uint8_t*s,int8_t*d,int n){uint8_t*cv=malloc(1024);for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;uint8_t p33[3],p67[3];for(int ch=0;ch<3;ch++){int cnt=0;for(int y=0;y<32;y++)for(int x=0;x<32;x++)cv[cnt++]=si[y*96+x*3+ch];qsort(cv,cnt,1,cmp_u8);p33[ch]=cv[cnt/3];p67[ch]=cv[2*cnt/3];if(p33[ch]==p67[ch]){p33[ch]=p33[ch]>0?p33[ch]-1:0;p67[ch]=p67[ch]<255?p67[ch]+1:255;}}for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int ch=0;ch<3;ch++){int idx=y*96+x*3+ch;di[idx]=si[idx]>p67[ch]?1:si[idx]<p33[ch]?-1:0;}}free(cv);}

typedef struct{uint32_t id;uint32_t votes;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

/* ================================================================
 *  Gauss map computation — multiple configurations
 * ================================================================ */

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

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_r_tr=malloc((size_t)TRAIN_N*SP_PX);raw_r_te=malloc((size_t)TEST_N*SP_PX);
    raw_g_tr=malloc((size_t)TRAIN_N*SP_PX);raw_g_te=malloc((size_t)TEST_N*SP_PX);
    raw_b_tr=malloc((size_t)TRAIN_N*SP_PX);raw_b_te=malloc((size_t)TEST_N*SP_PX);
    raw_gray_tr=malloc((size_t)TRAIN_N*SP_PX);raw_gray_te=malloc((size_t)TEST_N*SP_PX);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            memcpy(raw_r_tr+(size_t)idx*SP_PX,r,SP_PX);memcpy(raw_g_tr+(size_t)idx*SP_PX,g,SP_PX);memcpy(raw_b_tr+(size_t)idx*SP_PX,b3,SP_PX);
            uint8_t*gd=raw_gray_tr+(size_t)idx*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        memcpy(raw_r_te+(size_t)i*SP_PX,r,SP_PX);memcpy(raw_g_te+(size_t)i*SP_PX,g,SP_PX);memcpy(raw_b_te+(size_t)i*SP_PX,b3,SP_PX);
        uint8_t*gd=raw_gray_te+(size_t)i*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Cascade Gauss — Vote → Gauss Rank ===\n\n");
    load_data();

    /* ================================================================
     * Build Gauss maps: 4 configurations
     * ================================================================ */
    printf("Computing Gauss maps...\n");

    /* Config A: grayscale 4×4 grid (baseline = 48.31%) */
    #define GA_GRID 4
    #define GA_FEAT (GA_GRID*GA_GRID*GM_BINS) /* 720 */
    int16_t *ga_tr=malloc((size_t)TRAIN_N*GA_FEAT*2),*ga_te=malloc((size_t)TEST_N*GA_FEAT*2);
    for(int i=0;i<TRAIN_N;i++)gauss_map_grid_ch(raw_gray_tr+(size_t)i*SP_PX,ga_tr+(size_t)i*GA_FEAT,GA_GRID,GA_GRID,SP_W,SP_H);
    for(int i=0;i<TEST_N;i++)gauss_map_grid_ch(raw_gray_te+(size_t)i*SP_PX,ga_te+(size_t)i*GA_FEAT,GA_GRID,GA_GRID,SP_W,SP_H);

    /* Config B: grayscale 8×8 grid */
    #define GB_GRID 8
    #define GB_FEAT (GB_GRID*GB_GRID*GM_BINS) /* 2880 */
    int16_t *gb_tr=malloc((size_t)TRAIN_N*GB_FEAT*2),*gb_te=malloc((size_t)TEST_N*GB_FEAT*2);
    for(int i=0;i<TRAIN_N;i++)gauss_map_grid_ch(raw_gray_tr+(size_t)i*SP_PX,gb_tr+(size_t)i*GB_FEAT,GB_GRID,GB_GRID,SP_W,SP_H);
    for(int i=0;i<TEST_N;i++)gauss_map_grid_ch(raw_gray_te+(size_t)i*SP_PX,gb_te+(size_t)i*GB_FEAT,GB_GRID,GB_GRID,SP_W,SP_H);

    /* Config C: 4-channel RGB 4×4 grid (gray+R+G+B) */
    #define GC_CH 4
    #define GC_FEAT (GC_CH*GA_GRID*GA_GRID*GM_BINS) /* 2880 */
    int16_t *gc_tr=malloc((size_t)TRAIN_N*GC_FEAT*2),*gc_te=malloc((size_t)TEST_N*GC_FEAT*2);
    {const uint8_t *ch_tr[]={raw_gray_tr,raw_r_tr,raw_g_tr,raw_b_tr};
     const uint8_t *ch_te[]={raw_gray_te,raw_r_te,raw_g_te,raw_b_te};
     for(int i=0;i<TRAIN_N;i++)for(int ch=0;ch<GC_CH;ch++)
         gauss_map_grid_ch(ch_tr[ch]+(size_t)i*SP_PX,gc_tr+(size_t)i*GC_FEAT+ch*GA_FEAT,GA_GRID,GA_GRID,SP_W,SP_H);
     for(int i=0;i<TEST_N;i++)for(int ch=0;ch<GC_CH;ch++)
         gauss_map_grid_ch(ch_te[ch]+(size_t)i*SP_PX,gc_te+(size_t)i*GC_FEAT+ch*GA_FEAT,GA_GRID,GA_GRID,SP_W,SP_H);}

    /* Config D: 4-channel RGB 8×8 grid */
    #define GD_CH 4
    #define GD_FEAT (GD_CH*GB_GRID*GB_GRID*GM_BINS) /* 11520 */
    int16_t *gd_tr=malloc((size_t)TRAIN_N*GD_FEAT*2),*gd_te=malloc((size_t)TEST_N*GD_FEAT*2);
    {const uint8_t *ch_tr[]={raw_gray_tr,raw_r_tr,raw_g_tr,raw_b_tr};
     const uint8_t *ch_te[]={raw_gray_te,raw_r_te,raw_g_te,raw_b_te};
     for(int i=0;i<TRAIN_N;i++)for(int ch=0;ch<GD_CH;ch++)
         gauss_map_grid_ch(ch_tr[ch]+(size_t)i*SP_PX,gd_tr+(size_t)i*GD_FEAT+ch*GB_FEAT,GB_GRID,GB_GRID,SP_W,SP_H);
     for(int i=0;i<TEST_N;i++)for(int ch=0;ch<GD_CH;ch++)
         gauss_map_grid_ch(ch_te[ch]+(size_t)i*SP_PX,gd_te+(size_t)i*GD_FEAT+ch*GB_FEAT,GB_GRID,GB_GRID,SP_W,SP_H);}

    printf("  Gauss maps computed (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * Brute kNN baselines for each config
     * ================================================================ */
    printf("=== BRUTE kNN BASELINES ===\n\n");
    struct{const char*name;int16_t*tr;int16_t*te;int feat;}configs[]={
        {"Gray 4x4 (baseline)",ga_tr,ga_te,GA_FEAT},
        {"Gray 8x8",gb_tr,gb_te,GB_FEAT},
        {"RGB 4x4 (4-ch)",gc_tr,gc_te,GC_FEAT},
        {"RGB 8x8 (4-ch)",gd_tr,gd_te,GD_FEAT}};
    int nconfigs=4;

    for(int ci=0;ci<nconfigs;ci++){
        int correct=0;
        for(int i=0;i<TEST_N;i++){
            const int16_t*qi=configs[ci].te+(size_t)i*configs[ci].feat;
            int32_t best_d=INT32_MAX;int best_lbl=0;
            for(int j=0;j<TRAIN_N;j++){
                int32_t d=hist_l1(qi,configs[ci].tr+(size_t)j*configs[ci].feat,configs[ci].feat);
                if(d<best_d){best_d=d;best_lbl=train_labels[j];}}
            if(best_lbl==test_labels[i])correct++;
            if((i+1)%1000==0)fprintf(stderr,"  %s brute: %d/%d\r",configs[ci].name,i+1,TEST_N);}
        fprintf(stderr,"\n");
        printf("  %-25s k=1: %.2f%%\n",configs[ci].name,100.0*correct/TEST_N);
    }

    /* ================================================================
     * Build 3-eye stereo voting
     * ================================================================ */
    printf("\nBuilding 3-eye stereo voting...\n");
    eye_t eyes[3];
    void(*qfns[])(const uint8_t*,int8_t*,int)={quant_fixed,quant_adaptive,quant_perchannel};
    for(int ei=0;ei<3;ei++){build_eye(&eyes[ei],raw_train,raw_test,qfns[ei]);printf("  Eye %d built\n",ei+1);}
    printf("  Voting built (%.1f sec)\n\n",now_sec()-t0);

    /* ================================================================
     * CASCADE: Vote → Gauss Rank (for each config)
     * ================================================================ */
    printf("=== CASCADE: 3-Eye Stereo Vote → Gauss Rank ===\n\n");
    uint32_t *vbuf=calloc(TRAIN_N,4);

    for(int ci=0;ci<nconfigs;ci++){
        int correct[4]={0}; /* k=1,3,5,7 */
        int kvals[]={1,3,5,7};
        int recall=0;

        for(int i=0;i<TEST_N;i++){
            /* 3-eye stereo vote */
            memset(vbuf,0,TRAIN_N*4);
            for(int ei=0;ei<3;ei++)vote_eye(&eyes[ei],i,vbuf);
            cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);

            /* Check recall */
            for(int j=0;j<nc;j++)if(train_labels[cands[j].id]==test_labels[i]){recall++;break;}

            /* Gauss map L1 re-rank */
            const int16_t*qi=configs[ci].te+(size_t)i*configs[ci].feat;
            int32_t best_d[TOP_K];
            for(int j=0;j<nc;j++)best_d[j]=hist_l1(qi,configs[ci].tr+(size_t)cands[j].id*configs[ci].feat,configs[ci].feat);

            /* Sort by Gauss map distance (ascending = more similar) */
            /* Simple insertion sort on nc<=200 elements */
            for(int j=1;j<nc;j++){int32_t kd=best_d[j];cand_t kc=cands[j];int x=j-1;
                while(x>=0&&best_d[x]>kd){best_d[x+1]=best_d[x];cands[x+1]=cands[x];x--;}
                best_d[x+1]=kd;cands[x+1]=kc;}

            /* kNN vote */
            for(int ki=0;ki<4;ki++){
                int v[N_CLASSES]={0};int kk=kvals[ki]<nc?kvals[ki]:nc;
                for(int j=0;j<kk;j++)v[train_labels[cands[j].id]]++;
                int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
                if(pred==test_labels[i])correct[ki]++;}

            if((i+1)%1000==0)fprintf(stderr,"  %s cascade: %d/%d\r",configs[ci].name,i+1,TEST_N);
        }
        fprintf(stderr,"\n");
        printf("  %-25s recall=%.1f%%  k=1:%.2f%%  k=3:%.2f%%  k=5:%.2f%%  k=7:%.2f%%\n",
               configs[ci].name,100.0*recall/TEST_N,
               100.0*correct[0]/TEST_N,100.0*correct[1]/TEST_N,
               100.0*correct[2]/TEST_N,100.0*correct[3]/TEST_N);
    }

    /* ================================================================
     * Per-class for best config
     * ================================================================ */
    printf("\n=== PER-CLASS: Best cascade config ===\n\n");
    {
        /* Run best config (we'll determine from the results above) */
        /* For now, run RGB 4x4 which is likely best */
        int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for(int i=0;i<TEST_N;i++){pt[test_labels[i]]++;
            memset(vbuf,0,TRAIN_N*4);for(int ei=0;ei<3;ei++)vote_eye(&eyes[ei],i,vbuf);
            cand_t cands[TOP_K];int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
            const int16_t*qi=gc_te+(size_t)i*GC_FEAT;
            int32_t bd[TOP_K];for(int j=0;j<nc;j++)bd[j]=hist_l1(qi,gc_tr+(size_t)cands[j].id*GC_FEAT,GC_FEAT);
            for(int j=1;j<nc;j++){int32_t kd=bd[j];cand_t kc=cands[j];int x=j-1;
                while(x>=0&&bd[x]>kd){bd[x+1]=bd[x];cands[x+1]=cands[x];x--;}bd[x+1]=kd;cands[x+1]=kc;}
            /* k=5 */
            int v[N_CLASSES]={0};int kk=5<nc?5:nc;for(int j=0;j<kk;j++)v[train_labels[cands[j].id]]++;
            int pred=0;for(int c=1;c<N_CLASSES;c++)if(v[c]>v[pred])pred=c;
            if(pred==test_labels[i])pc[test_labels[i]]++;
            if((i+1)%2000==0)fprintf(stderr,"  per-class: %d/%d\r",i+1,TEST_N);}
        fprintf(stderr,"\n");
        int total_correct=0;for(int c=0;c<N_CLASSES;c++)total_correct+=pc[c];
        printf("  RGB 4x4 cascade k=5: %.2f%%\n",100.0*total_correct/TEST_N);
        for(int c=0;c<N_CLASSES;c++)printf("    %d %-10s %.1f%%\n",c,cn[c],100.0*pc[c]/pt[c]);
    }

    free(vbuf);
    printf("\n=== REFERENCE ===\n");
    printf("  Brute Gauss kNN (gray 4x4):  48.31%%\n");
    printf("  Stereo + MT4 stack:           44.48%%\n");
    printf("  Target:                       50.00%%\n");

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
