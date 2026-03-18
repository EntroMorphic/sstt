/*
 * sstt_delta_map.c — The Delta Map: Maps All The Way Down
 *
 * This experiment implements the "Ground-State Deviation" lookup.
 * 
 * Architecture:
 * 1. Ground States: 10 per-class Gauss map histograms (Mean/StdDev).
 * 2. Delta Map: A frequency table storing [prototype][bin][trit][target_class].
 * 
 * Training:
 * For each training image I with label L:
 *    For each prototype P from 0 to 9:
 *       Signature S_P = Quantize(I - Mean_P)
 *       For each bin B:
 *          DeltaMap[P][B][S_P(B)][L] += 1
 * 
 * Inference:
 * For query Q:
 *    Accumulator[0..9] = 0
 *    For each prototype P from 0 to 9:
 *       Signature S_P = Quantize(Q - Mean_P)
 *       For each bin B:
 *          Accumulator += DeltaMap[P][B][S_P(B)]
 *    Result = Argmax(Accumulator)
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

#define GM_HVALS 3
#define GM_VVALS 3
#define GM_DVALS 5
#define GM_BINS (GM_HVALS*GM_VVALS*GM_DVALS)
#define GM_PAD 48

/* Global data */
static const char *data_dir = "data/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint8_t *raw_train_img, *raw_test_img, *train_labels, *test_labels;
static int8_t *tern_train, *tern_test, *hgrad_train, *hgrad_test, *vgrad_train, *vgrad_test;
static int16_t *gm_train, *gm_test;

static int32_t gm_mean[N_CLASSES][GM_BINS];
static int16_t gm_std[N_CLASSES][GM_BINS];

/* Delta Map: [Prototype][Bin][Trit+1][Class] */
static uint32_t delta_map[N_CLASSES][GM_BINS][3][CLS_PAD] __attribute__((aligned(32)));

/* === Boilerplate === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test_img=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static void quant_tern(const uint8_t*src,int8_t*dst,int n){const __m256i bias=_mm256_set1_epi8((char)0x80),thi=_mm256_set1_epi8((char)(170^0x80)),tlo=_mm256_set1_epi8((char)(85^0x80)),one=_mm256_set1_epi8(1);for(int i=0;i<n;i++){const uint8_t*s=src+(size_t)i*PIXELS;int8_t*d=dst+(size_t)i*PADDED;int k;for(k=0;k+32<=PIXELS;k+=32){__m256i px=_mm256_loadu_si256((const __m256i*)(s+k));__m256i sp=_mm256_xor_si256(px,bias);_mm256_storeu_si256((__m256i*)(d+k),_mm256_sub_epi8(_mm256_and_si256(_mm256_cmpgt_epi8(sp,thi),one),_mm256_and_si256(_mm256_cmpgt_epi8(tlo,sp),one)));}for(;k<PIXELS;k++)d[k]=s[k]>170?1:s[k]<85?-1:0;memset(d+PIXELS,0,PADDED-PIXELS);}}
static void gradients(const int8_t*t,int8_t*h,int8_t*v,int n){for(int i=0;i<n;i++){const int8_t*ti=t+(size_t)i*PADDED;int8_t*hi=h+(size_t)i*PADDED;int8_t*vi=v+(size_t)i*PADDED;for(int y=0;y<IMG_H;y++){for(int x=0;x<IMG_W-1;x++)hi[y*IMG_W+x]=ct(ti[y*IMG_W+x+1]-ti[y*IMG_W+x]);hi[y*IMG_W+IMG_W-1]=0;}memset(hi+PIXELS,0,PADDED-PIXELS);for(int y=0;y<IMG_H-1;y++)for(int x=0;x<IMG_W;x++)vi[y*IMG_W+x]=ct(ti[(y+1)*IMG_W+x]-ti[y*IMG_W+x]);memset(vi+(IMG_H-1)*IMG_W,0,IMG_W);memset(vi+PIXELS,0,PADDED-PIXELS);}}

static inline int div_clamp(int d){return d<-2?-2:d>2?2:d;}
static void gauss_map_hist(const int8_t*hg,const int8_t*vg,int16_t*hist){
    memset(hist,0,GM_PAD*sizeof(int16_t));
    for(int y=0;y<IMG_H;y++)for(int x=0;x<IMG_W;x++){
        int h=hg[y*IMG_W+x],v=vg[y*IMG_W+x];
        int dh=h-(x>0?(int)hg[y*IMG_W+x-1]:0);int dv=v-(y>0?(int)vg[(y-1)*IMG_W+x]:0);
        int d=div_clamp(dh+dv);
        hist[(h+1)*GM_VVALS*GM_DVALS+(v+1)*GM_DVALS+(d+2)]++;
    }
}

/* === Core Logic === */

static void compute_ground_states(void) {
    int64_t sums[N_CLASSES][GM_BINS] = {0};
    int64_t sq_sums[N_CLASSES][GM_BINS] = {0};
    int counts[N_CLASSES] = {0};
    for(int i=0; i<TRAIN_N; i++){
        int c = train_labels[i]; counts[c]++;
        const int16_t *gm = gm_train + (size_t)i*GM_PAD;
        for(int b=0; b<GM_BINS; b++){
            sums[c][b] += gm[b];
            sq_sums[c][b] += (int64_t)gm[b]*gm[b];
        }
    }
    for(int c=0; c<N_CLASSES; c++){
        int n = counts[c]; if(!n) continue;
        for(int b=0; b<GM_BINS; b++){
            gm_mean[c][b] = (int32_t)(sums[c][b]/n);
            int64_t var = sq_sums[c][b]/n - (sums[c][b]/n)*(sums[c][b]/n);
            gm_std[c][b] = (int16_t)(var > 0 ? (int)sqrt((double)var) : 1);
        }
    }
}

static void build_delta_map(void) {
    memset(delta_map, 0, sizeof(delta_map));
    for(int i=0; i<TRAIN_N; i++) {
        int label = train_labels[i];
        const int16_t *gm = gm_train + (size_t)i*GM_PAD;
        for(int p=0; p<N_CLASSES; p++) {
            for(int b=0; b<GM_BINS; b++) {
                int32_t delta = (int32_t)gm[b] - gm_mean[p][b];
                int trit = delta > gm_std[p][b] ? 1 : delta < -gm_std[p][b] ? -1 : 0;
                delta_map[p][b][trit+1][label]++;
            }
        }
    }
}

static int delta_classify(const int16_t *q_gm) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    for(int p=0; p<N_CLASSES; p++) {
        for(int b=0; b<GM_BINS; b++) {
            int32_t delta = (int32_t)q_gm[b] - gm_mean[p][b];
            int trit = delta > gm_std[p][b] ? 1 : delta < -gm_std[p][b] ? -1 : 0;
            const uint32_t *row = delta_map[p][b][trit+1];
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_load_si256((const __m256i*)row));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_load_si256((const __m256i*)(row + 8)));
        }
    }
    uint32_t cv[CLS_PAD] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i*)cv, acc_lo);
    _mm256_store_si256((__m256i*)(cv + 8), acc_hi);
    int best = 0;
    for(int c=1; c<N_CLASSES; c++) if(cv[c] > cv[best]) best = c;
    return best;
}

int main(int argc, char** argv) {
    if(argc > 1) data_dir = argv[1];
    printf("=== SSTT Delta Map: Multi-Prototype Geometric Lookup ===\n\n");
    double t0 = now_sec();
    load_data();
    tern_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); tern_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    hgrad_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); hgrad_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    vgrad_train = aligned_alloc(32, (size_t)TRAIN_N*PADDED); vgrad_test = aligned_alloc(32, (size_t)TEST_N*PADDED);
    quant_tern(raw_train_img, tern_train, TRAIN_N); quant_tern(raw_test_img,  tern_test,  TEST_N);
    gradients(tern_train, hgrad_train, vgrad_train, TRAIN_N); gradients(tern_test, hgrad_test, vgrad_test, TEST_N);
    gm_train = aligned_alloc(32, (size_t)TRAIN_N * GM_PAD * 2); gm_test = aligned_alloc(32, (size_t)TEST_N * GM_PAD * 2);
    for(int i=0; i<TRAIN_N; i++) gauss_map_hist(hgrad_train+(size_t)i*PADDED, vgrad_train+(size_t)i*PADDED, gm_train+(size_t)i*GM_PAD);
    for(int i=0; i<TEST_N; i++) gauss_map_hist(hgrad_test+(size_t)i*PADDED, vgrad_test+(size_t)i*PADDED, gm_test+(size_t)i*GM_PAD);
    
    printf("  Computing ground states...\n"); compute_ground_states();
    printf("  Building Delta Map...\n"); build_delta_map();
    printf("  Classifying %d images...\n", TEST_N);
    int correct = 0; double tc0 = now_sec();
    for(int i=0; i<TEST_N; i++) if(delta_classify(gm_test + (size_t)i*GM_PAD) == test_labels[i]) correct++;
    double tc1 = now_sec();
    printf("\n--- Result: Delta Map Accuracy: %.2f%% (%.2f us/query) ---\n", 100.0 * correct / TEST_N, (tc1 - tc0) * 1e6 / TEST_N);
    return 0;
}
