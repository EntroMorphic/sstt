/*
 * sstt_dual_hot_map.c — Dual Hot Map: Corrected Skip and Weighting
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    60000
#define TEST_N     10000
#define N_CLASSES  10
#define CLS_PAD    16
#define IMG_W      28
#define IMG_H      28
#define PIXELS     784
#define PADDED     800

#define BLKS_PER_ROW 9
#define N_BLOCKS   252
#define GM_BINS 45
#define G_DIM 4
#define TOTAL_BINS (G_DIM*G_DIM*GM_BINS)

static const char *data_dir = "data/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint8_t *raw_train_img, *raw_test_img, *train_labels, *test_labels;
static int8_t *tern_train, *tern_test, *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;

static uint32_t px_map[N_BLOCKS][27][CLS_PAD] __attribute__((aligned(32)));
static uint32_t gm_map[TOTAL_BINS][8][CLS_PAD] __attribute__((aligned(32)));

static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void quant_tern(const uint8_t*src,int8_t*dst,int n){const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;for(k=0;k+32<=PIXELS;k+=32){__m256i px=_mm256_loadu_si256((const __m256i*)(s+k));__m256i sp=_mm256_xor_si256(px,bias);_mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));}for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}memset(hi+PIXELS,0,PADDED-PIXELS);for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}
static inline int dc(int d){return d<-2?-2:d>2?2:d;}
static inline uint8_t qc(int c){return c==0?0:c==1?1:c==2?2:c<=4?3:c<=8?4:c<=16?5:6;}

static void get_px_sig(const int8_t *img, uint8_t *sig){for(int y=0;y<IMG_H;y++)for(int s=0;s<BLKS_PER_ROW;s++)sig[y*BLKS_PER_ROW+s]=benc(img[y*IMG_W+s*3],img[y*IMG_W+s*3+1],img[y*IMG_W+s*3+2]);}
static void get_gm_sig(const int8_t *h,const int8_t *v,uint8_t *sig){memset(sig,0,TOTAL_BINS);for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){int bin=(h[y*IMG_W+x]+1)*15+(v[y*IMG_W+x]+1)*5+(dc((h[y*IMG_W+x]-(x>0?h[y*IMG_W+x-1]:0))+(v[y*IMG_W+x]-(y>0?v[(y-1)*IMG_W+x]:0)))+2);sig[(y*G_DIM/IMG_H)*G_DIM*GM_BINS+(x*G_DIM/IMG_W)*GM_BINS+bin]++;}}

int main(int argc, char** argv) {
    if(argc > 1) data_dir = argv[1];
    load_data();
    tern_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); tern_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    hgrad_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); hgrad_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    vgrad_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); vgrad_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    quant_tern(raw_train_img, tern_train, TRAIN_N); quant_tern(raw_test_img,  tern_test,  TEST_N);
    gradients(tern_train, hgrad_train, vgrad_train, TRAIN_N); gradients(tern_test, hgrad_test, vgrad_test, TEST_N);
    
    memset(px_map, 0, sizeof(px_map)); memset(gm_map, 0, sizeof(gm_map));
    uint8_t psig[N_BLOCKS], gsig[TOTAL_BINS];
    for(int i=0; i<TRAIN_N; i++) {
        int l=train_labels[i]; get_px_sig(tern_train+(size_t)i*PADDED, psig); get_gm_sig(hgrad_train+(size_t)i*PADDED, vgrad_train+(size_t)i*PADDED, gsig);
        for(int k=0;k<N_BLOCKS;k++) px_map[k][psig[k]][l]++;
        for(int k=0;k<TOTAL_BINS;k++) gm_map[k][qc(gsig[k])][l]++;
    }
    
    int c_px=0, c_gm=0, c_dual=0;
    uint32_t cv_px[CLS_PAD], cv_gm[CLS_PAD], cv_dual[CLS_PAD];
    for(int i=0; i<TEST_N; i++) {
        get_px_sig(tern_test+(size_t)i*PADDED, psig); get_gm_sig(hgrad_test+(size_t)i*PADDED, vgrad_test+(size_t)i*PADDED, gsig);
        memset(cv_px, 0, sizeof(cv_px)); memset(cv_gm, 0, sizeof(cv_gm));
        for(int k=0;k<N_BLOCKS;k++) if(psig[k]!=0) for(int c=0;c<10;c++) cv_px[c]+=px_map[k][psig[k]][c];
        for(int k=0;k<TOTAL_BINS;k++) if(qc(gsig[k])!=0) for(int c=0;c<10;c++) cv_gm[c]+=gm_map[k][qc(gsig[k])][c];
        
        int b_px=0, b_gm=0, b_dual=0;
        for(int c=1;c<10;c++) {
            if(cv_px[c]>cv_px[b_px]) b_px=c;
            if(cv_gm[c]>cv_gm[b_gm]) b_gm=c;
            if(cv_px[c]+cv_gm[c]/2 > cv_px[b_dual]+cv_gm[b_dual]/2) b_dual=c;
        }
        if(b_px==test_labels[i]) c_px++;
        if(b_gm==test_labels[i]) c_gm++;
        if(b_dual==test_labels[i]) c_dual++;
    }
    printf("Result (N=%d):\n  Pixel Map: %.2f%%\n  Gauss Map: %.2f%%\n  Dual Map:  %.2f%%\n", TEST_N, 100.0*c_px/TEST_N, 100.0*c_gm/TEST_N, 100.0*c_dual/TEST_N);
    return 0;
}
