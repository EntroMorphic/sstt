/*
 * sstt_cifar10_lagrangian.c — Discrete Lagrangian Vorticity (Curl)
 *
 * Eulerian approach (Current): Count edges in fixed boxes.
 * Lagrangian approach (This): Analyze field rotations (Curl) and flow.
 *
 * Curl = d(Vgrad)/dx - d(Hgrad)/dy
 *
 * A positive curl indicates counter-clockwise rotation (bottom-left corner?).
 * A negative curl indicates clockwise rotation.
 * High absolute curl = high-energy topological junction (corner/vertex).
 * Zero curl = laminar flow (straight edge) or flat region.
 *
 * Target: Identify corners and pose-invariant junctions to help animal classes.
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 5000
#define TEST_N  100
#define N_CLASSES 10
#define CLS_PAD 16
#define SP_W 32
#define SP_H 32
#define SP_PX 1024
#define IMG_W 96
#define IMG_H 32
#define PIXELS 3072
#define PADDED 3072
#define BLKS_PER_ROW 32
#define N_BLOCKS 1024
#define SIG_PAD 1024
#define BYTE_VALS 256
#define IG_SCALE 16
#define TOP_K 200

/* Curl parameters */
#define CURL_BINS 9 /* Curl values in {-4, -3, -2, -1, 0, 1, 2, 3, 4} */
#define GM_DIR 3
#define GM_MAG 5
#define GM_BINS (GM_DIR*GM_DIR*GM_MAG) /* 45 */
#define G4 4
#define G4_CURL_FEAT (G4*G4*CURL_BINS)
#define G4_GM_FEAT (G4*G4*GM_BINS)
#define G4_COMBO_FEAT (G4_CURL_FEAT + G4_GM_FEAT)

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static uint8_t *raw_r_tr,*raw_r_te,*raw_g_tr,*raw_g_te,*raw_b_tr,*raw_b_te,*raw_gray_tr,*raw_gray_te;

/* ================================================================
 *  Curl Computation
 * ================================================================ */

static void compute_gauss_curl_grid(const uint8_t *img, int16_t *hist, int gr, int gc) {
    int nbins_curl=CURL_BINS, nbins_gm=GM_BINS;
    int nreg=gr*gc;
    memset(hist,0,(size_t)nreg*(nbins_curl+nbins_gm)*sizeof(int16_t));

    /* 1. Compute ternary gradients */
    int8_t hg[SP_PX], vg[SP_PX];
    for(int y=0;y<SP_H;y++)for(int x=0;x<SP_W;x++){
        int px = img[y*SP_W+x];
        int t = (px > 170) ? 1 : (px < 85) ? -1 : 0;
        int tx = (x < SP_W-1) ? (int)img[y*SP_W+x+1] : px;
        int ty = (y < SP_H-1) ? (int)img[(y+1)*SP_W+x] : px;
        int t_nx = (tx > 170) ? 1 : (tx < 85) ? -1 : 0;
        int t_ny = (ty > 170) ? 1 : (ty < 85) ? -1 : 0;
        int gx = ct(t_nx - t);
        int gy = ct(t_ny - t);
        hg[y*SP_W+x] = (int8_t)gx;
        vg[y*SP_W+x] = (int8_t)gy;

        /* Gauss map part */
        int dx=(gx>0)?2:(gx<0)?0:1,dy=(gy>0)?2:(gy<0)?0:1;
        int mag=abs(gx)+abs(gy),mb=(mag==0)?0:(mag==1)?2:4; // simplified mag for ternary
        int bin_gm=dx*GM_DIR*GM_MAG+dy*GM_MAG+mb;
        int ry=y*gr/SP_H,rx=x*gc/SP_W;
        if(ry>=gr)ry=gr-1;if(rx>=gc)rx=gc-1;
        hist[(ry*gc+rx)*(nbins_curl+nbins_gm) + bin_gm]++;
    }

    /* 2. Compute Discrete Curl: d(V)/dx - d(H)/dy */
    for(int y=0;y<SP_H-1;y++)for(int x=0;x<SP_W-1;x++){
        int dv_dx = vg[y*SP_W+x+1] - vg[y*SP_W+x];
        int dh_dy = hg[(y+1)*SP_W+x] - hg[y*SP_W+x];
        int curl = dv_dx - dh_dy;
        int bin_curl = curl + 4;
        
        int ry=y*gr/SP_H, rx=x*gc/SP_W;
        if(ry>=gr)ry=gr-1; if(rx>=gc)rx=gc-1;
        hist[(ry*gc+rx)*(nbins_curl+nbins_gm) + nbins_gm + bin_curl]++;
    }
}

static int32_t hist_l1(const int16_t*a,const int16_t*b,int len){int32_t d=0;for(int i=0;i<len;i++)d+=abs(a[i]-b[i]);return d;}

/* ================================================================
 *  Block-based voting (3-eye stereo) - Simplified from curvature.c
 * ================================================================ */
typedef struct{uint8_t *joint_tr,*joint_te;uint32_t *hot;uint16_t ig[N_BLOCKS];uint8_t bg;
    uint32_t idx_off[N_BLOCKS][BYTE_VALS];uint16_t idx_sz[N_BLOCKS][BYTE_VALS];uint32_t *idx_pool;
    uint8_t nbr[BYTE_VALS][8];}eye_t;
static inline uint8_t be(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static inline uint8_t te2(uint8_t a,uint8_t b){int8_t a0=(a/9)-1,a1=((a/3)%3)-1,a2=(a%3)-1,b0=(b/9)-1,b1=((b/3)%3)-1,b2=(b%3)-1;return be(ct(b0-a0),ct(b1-a1),ct(b2-a2));}

static void build_eye(eye_t*e,void(*qfn)(const uint8_t*,int8_t*,int)){
    int8_t *ttr=aligned_alloc(32,(size_t)TRAIN_N*PADDED),*tte=aligned_alloc(32,(size_t)TEST_N*PADDED);
    qfn(raw_train,ttr,TRAIN_N);qfn(raw_test,tte,TEST_N);
    uint8_t *pxt=aligned_alloc(32,(size_t)TRAIN_N*SIG_PAD),*pxe=aligned_alloc(32,(size_t)TEST_N*SIG_PAD);
    for(int i=0;i<TRAIN_N;i++){uint8_t*si=pxt+(size_t)i*SIG_PAD;int8_t*im=ttr+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    for(int i=0;i<TEST_N;i++){uint8_t*si=pxe+(size_t)i*SIG_PAD;int8_t*im=tte+(size_t)i*PADDED;
        for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++){int b2=y*IMG_W+s*3;si[y*BLKS_PER_ROW+s]=be(im[b2],im[b2+1],im[b2+2]);}}
    e->joint_tr=pxt; e->joint_te=pxe;
    free(ttr); free(tte);
    /* Simplified indexing for Lagrangian test */
    long vc[BYTE_VALS]={0};for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)vc[sig[k]]++;}
    e->bg=0;long mc=0;for(int v=0;v<BYTE_VALS;v++)if(vc[v]>mc){mc=vc[v];e->bg=(uint8_t)v;}
    for(int v=0;v<BYTE_VALS;v++)for(int b3=0;b3<8;b3++)e->nbr[v][b3]=(uint8_t)(v^(1<<b3));
    memset(e->idx_sz,0,sizeof(e->idx_sz));
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_sz[k][sig[k]]++;}
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<BYTE_VALS;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc((size_t)tot*4);uint32_t(*wp)[BYTE_VALS]=malloc((size_t)N_BLOCKS*BYTE_VALS*4);memcpy(wp,e->idx_off,(size_t)N_BLOCKS*BYTE_VALS*4);
    for(int i=0;i<TRAIN_N;i++){const uint8_t*sig=e->joint_tr+(size_t)i*SIG_PAD;for(int k=0;k<N_BLOCKS;k++)if(sig[k]!=e->bg)e->idx_pool[wp[k][sig[k]]++]=(uint32_t)i;}free(wp);
    for(int k=0;k<N_BLOCKS;k++) e->ig[k]=1;
}

static void vote_eye(const eye_t*e,int img,uint32_t*votes){
    const uint8_t*sig=e->joint_te+(size_t)img*SIG_PAD;
    for(int k=0;k<N_BLOCKS;k++){uint8_t bv=sig[k];if(bv==e->bg)continue;
        uint32_t off=e->idx_off[k][bv];uint16_t sz=e->idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)votes[e->idx_pool[off+j]]++;}}

static void quant_fixed(const uint8_t*s,int8_t*d,int n){for(int img=0;img<n;img++){const uint8_t*si=s+(size_t)img*PIXELS;int8_t*di=d+(size_t)img*PADDED;for(int i=0;i<PIXELS;i++)di[i]=si[i]>170?1:si[i]<85?-1:0;}}
typedef struct{uint32_t id;uint32_t votes;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int *ghist=NULL;static size_t ghcap=0;
static int topk(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    if((size_t)(mx+1)>ghcap){ghcap=(size_t)(mx+1)+4096;free(ghist);ghist=malloc(ghcap*sizeof(int));}
    memset(ghist,0,(mx+1)*sizeof(int));for(int i=0;i<n;i++)if(v[i])ghist[v[i]]++;
    int cu=0,th;for(th=(int)mx;th>=1;th--){cu+=ghist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc].id=i;o[nc].votes=v[i];nc++;}
    qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS);raw_test=malloc((size_t)TEST_N*PIXELS);
    raw_r_tr=malloc((size_t)TRAIN_N*SP_PX);raw_r_te=malloc((size_t)TEST_N*SP_PX);
    raw_g_tr=malloc((size_t)TRAIN_N*SP_PX);raw_g_te=malloc((size_t)TEST_N*SP_PX);
    raw_b_tr=malloc((size_t)TRAIN_N*SP_PX);raw_b_te=malloc((size_t)TEST_N*SP_PX);
    raw_gray_tr=malloc((size_t)TRAIN_N*SP_PX);raw_gray_te=malloc((size_t)TEST_N*SP_PX);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");if(!f)continue;
        for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i; if(idx>=TRAIN_N) continue;
            train_labels[idx]=rec[0];
            uint8_t*d=raw_train+(size_t)idx*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
            memcpy(raw_r_tr+(size_t)idx*SP_PX,r,SP_PX);memcpy(raw_g_tr+(size_t)idx*SP_PX,g,SP_PX);memcpy(raw_b_tr+(size_t)idx*SP_PX,b3,SP_PX);
            uint8_t*gd=raw_gray_tr+(size_t)idx*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");if(!f)exit(1);
    for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        if(i>=TEST_N) continue;
        test_labels[i]=rec[0];uint8_t*d=raw_test+(size_t)i*PIXELS;const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++){int si=y*32+x,di=y*96+x*3;d[di]=r[si];d[di+1]=g[si];d[di+2]=b3[si];}
        memcpy(raw_r_te+(size_t)i*SP_PX,r,SP_PX);memcpy(raw_g_te+(size_t)i*SP_PX,g,SP_PX);memcpy(raw_b_te+(size_t)i*SP_PX,b3,SP_PX);
        uint8_t*gd=raw_gray_te+(size_t)i*SP_PX;for(int p2=0;p2<SP_PX;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1) data_dir=argv[1];
    printf("=== SSTT CIFAR-10: Discrete Lagrangian Vorticity (Curl) ===\n\n");
    load_data();

    printf("Computing Gauss+Curl maps...\n");
    const uint8_t *ch_tr[]={raw_gray_tr,raw_r_tr,raw_g_tr,raw_b_tr};
    const uint8_t *ch_te[]={raw_gray_te,raw_r_te,raw_g_te,raw_b_te};
    #define NCH 4
    #define FEAT_DIM (G4*G4*(GM_BINS+CURL_BINS))
    #define TOTAL_FEAT (NCH*FEAT_DIM)

    int16_t *fa_tr=aligned_alloc(32,(size_t)TRAIN_N*TOTAL_FEAT*2);
    int16_t *fa_te=aligned_alloc(32,(size_t)TEST_N*TOTAL_FEAT*2);
    if(!fa_tr || !fa_te){printf("alloc fail\n"); return 1;}

    for(int i=0;i<TRAIN_N;i++)for(int ch=0;ch<NCH;ch++)compute_gauss_curl_grid(ch_tr[ch]+(size_t)i*SP_PX,fa_tr+(size_t)i*TOTAL_FEAT+ch*FEAT_DIM,G4,G4);
    for(int i=0;i<TEST_N;i++)for(int ch=0;ch<NCH;ch++)compute_gauss_curl_grid(ch_te[ch]+(size_t)i*SP_PX,fa_te+(size_t)i*TOTAL_FEAT+ch*FEAT_DIM,G4,G4);

    printf("  Features computed (%.1f sec)\n\n",now_sec()-t0);

    printf("=== BRUTE kNN BASELINES (Gauss+Curl) ===\n");
    int correct=0;
    for(int i=0;i<TEST_N;i++){
        const int16_t*qi=fa_te+(size_t)i*TOTAL_FEAT;
        int32_t best_d=INT32_MAX; int best_l=0;
        for(int j=0;j<TRAIN_N;j++){
            int32_t d=hist_l1(qi,fa_tr+(size_t)j*TOTAL_FEAT,TOTAL_FEAT);
            if(d<best_d){best_d=d;best_l=train_labels[j];}
        }
        if(best_l==test_labels[i])correct++;
    }
    printf("  Gauss+Curl brute k=1: %.2f%%\n\n", 100.0*correct/TEST_N);

    printf("=== CASCADE: Texture Vote → Gauss+Curl Rank ===\n");
    eye_t eye; build_eye(&eye, quant_fixed);
    uint32_t *vbuf=calloc(TRAIN_N, 4);
    int correct_cascade=0;
    for(int i=0;i<TEST_N;i++){
        memset(vbuf,0,TRAIN_N*4); vote_eye(&eye, i, vbuf);
        cand_t cands[TOP_K]; int nc=topk(vbuf,TRAIN_N,cands,TOP_K);
        const int16_t*qi=fa_te+(size_t)i*TOTAL_FEAT;
        int32_t best_d=INT32_MAX; int best_l=0;
        for(int j=0;j<nc;j++){
            int32_t d=hist_l1(qi,fa_tr+(size_t)cands[j].id*TOTAL_FEAT,TOTAL_FEAT);
            if(d<best_d){best_d=d;best_l=train_labels[cands[j].id];}
        }
        if(best_l==test_labels[i])correct_cascade++;
    }
    printf("  Cascade (Texture → Gauss+Curl) k=1: %.2f%%\n", 100.0*correct_cascade/TEST_N);

    printf("\nTotal: %.1f sec\n",now_sec()-t0);
    return 0;
}
