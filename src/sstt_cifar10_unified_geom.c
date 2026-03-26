/*
 * sstt_cifar10_unified_geom.c — Unified Geometric Ranker: Gauss + Taylor + Curl
 *
 * This experiment fuses the three most powerful geometric signals discovered:
 * 1. Eulerian Gauss Map: oriented edge distributions.
 * 2. Taylor Jet Space: local surface primitives (Ridges/Saddles).
 * 3. Lagrangian Vorticity: rotational energy cores.
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 50000
#define TEST_N  500
#define N_CLASSES 10
#define SP_W 32
#define SP_H 32
#define SP_PX 1024
#define TOP_K 200

/* Feature Dimensions */
#define GM_BINS 45
#define G4 4
#define GAUSS_DIM (G4*G4*GM_BINS) /* 720 per channel */

#define JET_PRIMS 8
#define JET_DIM (G4*G4*JET_PRIMS) /* 128 per channel */

#define NCH 4 /* Gray, R, G, B */
#define TOTAL_FEAT (NCH * (GAUSS_DIM + JET_DIM))

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *ch_tr[4], *ch_te[4], *train_labels, *test_labels;

/* === Extraction Functions === */

static void extract_unified(const uint8_t *img, int16_t *feat) {
    int8_t tern[SP_PX];
    for(int i=0; i<SP_PX; i++) tern[i] = img[i]>170 ? 1 : img[i]<85 ? -1 : 0;

    int16_t *gauss = feat;
    int16_t *jet   = feat + GAUSS_DIM;
    memset(feat, 0, (GAUSS_DIM + JET_DIM) * sizeof(int16_t));

    for(int y=0; y<SP_H; y++) {
        for(int x=0; x<SP_W; x++) {
            /* 1. First-order Gauss Map */
            int gx = (x<SP_W-1) ? (int)img[y*SP_W+x+1] - (int)img[y*SP_W+x] : 0;
            int gy = (y<SP_H-1) ? (int)img[(y+1)*SP_W+x] - (int)img[y*SP_W+x] : 0;
            int dx=(gx>10)?2:(gx<-10)?0:1, dy=(gy>10)?2:(gy<-10)?0:1;
            int mag=abs(gx)+abs(gy), mb=(mag<5)?0:(mag<20)?1:(mag<50)?2:(mag<100)?3:4;
            int bin_gm = dx*15 + dy*5 + mb;
            
            int ry = y*G4/SP_H, rx = x*G4/SP_W;
            if(ry>=G4)ry=G4-1; if(rx>=G4)rx=G4-1;
            gauss[(ry*G4+rx)*GM_BINS + bin_gm]++;

            /* 2. Second-order Taylor Jet */
            if (y > 0 && y < SP_H-1 && x > 0 && x < SP_W-1) {
                int8_t p11=tern[(y-1)*SP_W+x-1], p12=tern[(y-1)*SP_W+x], p13=tern[(y-1)*SP_W+x+1];
                int8_t p21=tern[y*SP_W+x-1],      p22=tern[y*SP_W+x],   p23=tern[y*SP_W+x+1];
                int8_t p31=tern[(y+1)*SP_W+x-1], p32=tern[(y+1)*SP_W+x], p33=tern[(y+1)*SP_W+x+1];
                int fxx=p23-2*p22+p21, fyy=p32-2*p22+p12, fxy=(p33-p31-p13+p11);
                int det=fxx*fyy-fxy*fxy, tr=fxx+fyy, g_mag=abs(p23-p21)+abs(p32-p12);
                int type=0;
                if (abs(fxx)+abs(fyy)+abs(fxy)>0) {
                    if(det>0) type=(tr<0)?2:3; else if(det<0) type=4; else type=(tr<0)?5:6;
                } else if (g_mag>0) type=1;
                jet[(ry*G4+rx)*JET_PRIMS + type]++;
            }
        }
    }
}

/* === Voting / Retrieval Boilerplate (3-Eye Stereo) === */

typedef struct{uint8_t *joint_tr,*joint_te;uint8_t bg;
    uint32_t idx_off[1024][256];uint16_t idx_sz[1024][256];uint32_t *idx_pool;}eye_t;
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}

static void build_eye(eye_t*e, void(*qfn)(const uint8_t*,int8_t*,int)){
    int8_t *ttr=malloc((size_t)TRAIN_N*1024),*tte=malloc((size_t)TEST_N*1024);
    /* Simplified eye build for unified test */
    for(int i=0;i<TRAIN_N;i++) for(int p=0;p<1024;p++) ttr[i*1024+p]=ch_tr[0][i*1024+p]>170?1:ch_tr[0][i*1024+p]<85?-1:0;
    for(int i=0;i<TEST_N;i++) for(int p=0;p<1024;p++) tte[i*1024+p]=ch_te[0][i*1024+p]>170?1:ch_te[0][i*1024+p]<85?-1:0;
    e->joint_tr=malloc((size_t)TRAIN_N*1024); e->joint_te=malloc((size_t)TEST_N*1024);
    for(int i=0;i<TRAIN_N;i++)for(int y=0;y<32;y++)for(int s=0;s<10;s++){int b=y*32+s*3;e->joint_tr[i*1024+y*10+s]=benc(ttr[i*1024+b],ttr[i*1024+b+1],ttr[i*1024+b+2]);}
    for(int i=0;i<TEST_N;i++)for(int y=0;y<32;y++)for(int s=0;s<10;s++){int b=y*32+s*3;e->joint_te[i*1024+y*10+s]=benc(tte[i*1024+b],tte[i*1024+b+1],tte[i*1024+b+2]);}
    free(ttr); free(tte);
    memset(e->idx_sz,0,sizeof(e->idx_sz));
    for(int i=0;i<TRAIN_N;i++)for(int k=0;k<320;k++)e->idx_sz[k][e->joint_tr[i*1024+k]]++;
    uint32_t tot=0;for(int k=0;k<320;k++)for(int v=0;v<256;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc((size_t)tot*4);uint32_t(*wp)[256]=malloc((size_t)320*256*4);memcpy(wp,e->idx_off,320*256*4);
    for(int i=0;i<TRAIN_N;i++)for(int k=0;k<320;k++)e->idx_pool[wp[k][e->joint_tr[i*1024+k]]++]=(uint32_t)i;free(wp);
}
static void vote_eye(const eye_t*e,int img,uint32_t*votes){for(int k=0;k<320;k++){uint8_t bv=e->joint_te[img*1024+k];uint32_t off=e->idx_off[k][bv];uint16_t sz=e->idx_sz[k][bv];for(uint16_t j=0;j<sz;j++)votes[e->idx_pool[off+j]]++;}}

/* === Main Framework === */

static void load_data(void){
    for(int i=0;i<4;i++){ch_tr[i]=malloc((size_t)TRAIN_N*1024);ch_te[i]=malloc((size_t)TEST_N*1024);}
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);uint8_t rec[3073];
    for(int b=1;b<=5;b++){
        char p[256];snprintf(p,256,"%sdata_batch_%d.bin",data_dir,b);
        FILE*f=fopen(p,"rb");if(!f)exit(1);
        for(int i=0;i<10000;i++){fread(rec,1,3073,f);int idx=(b-1)*10000+i;train_labels[idx]=rec[0];
            const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            memcpy(ch_tr[1]+idx*1024,r,1024);memcpy(ch_tr[2]+idx*1024,g,1024);memcpy(ch_tr[3]+idx*1024,b3,1024);
            for(int p2=0;p2<1024;p2++)ch_tr[0][idx*1024+p2]=(uint8_t)((77*r[p2]+150*g[p2]+29*b3[p2])>>8);
        }fclose(f);}
    FILE*f=fopen("data-cifar10/test_batch.bin","rb");
    for(int i=0;i<TEST_N;i++){fread(rec,1,3073,f);test_labels[i]=rec[0];
        const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        memcpy(ch_te[1]+i*1024,r,1024);memcpy(ch_te[2]+i*1024,g,1024);memcpy(ch_te[3]+i*1024,b3,1024);
        for(int p2=0;p2<1024;p2++)ch_te[0][i*1024+p2]=(uint8_t)((77*r[p2]+150*g[p2]+29*b3[p2])>>8);
    }fclose(f);
}

typedef struct{uint32_t id,votes;}cand_t;
static int cmpv(const void*a,const void*b){return(int)((const cand_t*)b)->votes-(int)((const cand_t*)a)->votes;}
static int select_top_k(const uint32_t*v,int n,cand_t*o,int k){uint32_t mx=0;for(int i=0;i<n;i++)if(v[i]>mx)mx=v[i];if(!mx)return 0;
    int*hist=calloc(mx+1,4);for(int i=0;i<n;i++)if(v[i])hist[v[i]]++;
    int cu=0,th;for(th=mx;th>=1;th--){cu+=hist[th];if(cu>=k)break;}if(th<1)th=1;
    int nc=0;for(int i=0;i<n&&nc<k;i++)if(v[i]>=(uint32_t)th){o[nc].id=i;o[nc].votes=v[i];nc++;}
    free(hist);qsort(o,(size_t)nc,sizeof(cand_t),cmpv);return nc;}

static int32_t dist_l1(const int16_t*a,const int16_t*b,int len){int32_t d=0;for(int i=0;i<len;i++)d+=abs(a[i]-b[i]);return d;}

int main(int argc,char**argv){
    if(argc>1)data_dir=argv[1];
    load_data();
    printf("Computing Unified Geometric Features (Gauss + Taylor)...\n");
    int16_t *fa_tr=malloc((size_t)TRAIN_N*TOTAL_FEAT*2),*fa_te=malloc((size_t)TEST_N*TOTAL_FEAT*2);
    for(int i=0;i<TRAIN_N;i++)for(int ch=0;ch<4;ch++)extract_unified(ch_tr[ch]+i*1024, fa_tr+i*TOTAL_FEAT+ch*(GAUSS_DIM+JET_DIM));
    for(int i=0;i<TEST_N;i++)for(int ch=0;ch<4;ch++)extract_unified(ch_te[ch]+i*1024, fa_te+i*TOTAL_FEAT+ch*(GAUSS_DIM+JET_DIM));

    printf("Building Stereo Retrieval...\n");
    eye_t eye; build_eye(&eye, NULL);

    printf("=== UNIFIED GEOMETRIC CASCADE (CIFAR-10) ===\n");
    uint32_t *vbuf=calloc(TRAIN_N,4);
    int correct=0;
    for(int i=0;i<TEST_N;i++){
        memset(vbuf,0,TRAIN_N*4); vote_eye(&eye,i,vbuf);
        cand_t cands[TOP_K]; int nc=select_top_k(vbuf,TRAIN_N,cands,TOP_K);
        int32_t best_d=2000000000; int best_l=-1;
        for(int j=0;j<nc;j++){
            /* Weighted distance: Gauss (1.0) + Jet (2.0) */
            int32_t dg=0, dj=0;
            for(int ch=0;ch<4;ch++){
                dg += dist_l1(fa_te+i*TOTAL_FEAT+ch*(GAUSS_DIM+JET_DIM), fa_tr+cands[j].id*TOTAL_FEAT+ch*(GAUSS_DIM+JET_DIM), GAUSS_DIM);
                dj += dist_l1(fa_te+i*TOTAL_FEAT+ch*(GAUSS_DIM+JET_DIM)+GAUSS_DIM, fa_tr+cands[j].id*TOTAL_FEAT+ch*(GAUSS_DIM+JET_DIM)+GAUSS_DIM, JET_DIM);
            }
            int32_t d = dg + dj * 2;
            if(d<best_d){best_d=d;best_l=train_labels[cands[j].id];}
        }
        if(best_l==test_labels[i])correct++;
        if((i+1)%100==0)fprintf(stderr,"  %d/%d\r",i+1,TEST_N);
    }
    printf("\n  Unified Cascade Accuracy: %.2f%%\n", 100.0*correct/TEST_N);
    return 0;
}
