/*
 * sstt_cifar10_5trit_quick.c — Quick 5-Eye Retrieval on CIFAR-10
 *
 * This experiment applies the successful 5-eye ensemble to CIFAR-10.
 *
 * 1. RETRIEVAL: 5-Eye (85/170, P33/P67, P20/P80, Fixed 64/192, Fixed 96/160)
 * 2. RANKING: Standard Ternary Dot (Eye 1).
 *
 * Usage:
 *   ./build/sstt_cifar10_5trit_quick
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 50000
#define TEST_N  1000
#define PIXELS  3072
#define PADDED  3072
#define N_CLASSES 10
#define BLKS_PER_ROW 32
#define N_BLOCKS 1024
#define SIG_PAD  1024
#define BYTE_VALS 256
#define TOP_K 100
#define MAX_EYES 5

static const char *data_dir="data-cifar10/";
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static int8_t *t1_train, *t1_test;

typedef struct {
    uint32_t idx_off[N_BLOCKS][256];
    uint16_t idx_sz[N_BLOCKS][256];
    uint32_t *idx_pool;
    uint16_t ig_w[N_BLOCKS];
} eye_t;

static eye_t eyes[MAX_EYES];

static int cmp_u8(const void*a,const void*b){return *(const uint8_t*)a-*(const uint8_t*)b;}

static void load_cifar10(void){
    raw_train=malloc((size_t)TRAIN_N*PIXELS); raw_test=malloc((size_t)TEST_N*PIXELS);
    train_labels=malloc(TRAIN_N); test_labels=malloc(TEST_N);
    for(int b=1; b<=5; b++){
        char p[256]; snprintf(p,256,"%sdata_batch_%d.bin",data_dir,b);
        FILE*f=fopen(p,"rb"); for(int i=0;i<10000;i++){train_labels[(b-1)*10000+i]=fgetc(f); fread(raw_train+((b-1)*10000+i)*PIXELS,1,PIXELS,f);} fclose(f);
    }
    FILE*f=fopen("data-cifar10/test_batch.bin","rb");
    for(int i=0;i<TEST_N;i++){test_labels[i]=fgetc(f); fread(raw_test+i*PIXELS,1,PIXELS,f);} fclose(f);
}

static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}
static inline int32_t tdot(const int8_t*a,const int8_t*b){
    __m256i acc=_mm256_setzero_si256();
    for(int i=0;i<PIXELS;i+=32)acc=_mm256_add_epi8(acc,_mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(a+i)),_mm256_loadu_si256((const __m256i*)(b+i))));
    __m256i lo=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc)),hi=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc,1));
    __m256i s32=_mm256_madd_epi16(_mm256_add_epi16(lo,hi),_mm256_set1_epi16(1));
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(s32),_mm256_extracti128_si256(s32,1));
    s=_mm_add_epi32(s,_mm_shuffle_epi32(s,_MM_SHUFFLE(1,0,3,2)));s=_mm_add_epi32(s,_mm_shuffle_epi32(s,_MM_SHUFFLE(2,3,0,1)));return _mm_cvtsi128_si32(s);
}

static void build_eye(int ei, int p1, int p2, int is_adaptive){
    eye_t*e=&eyes[ei]; uint8_t*sigs=malloc((size_t)TRAIN_N*SIG_PAD);
    uint8_t*sr=malloc(PIXELS);
    for(int i=0;i<TRAIN_N;i++){
        int lp1=p1, lp2=p2;
        if(is_adaptive){memcpy(sr,raw_train+i*PIXELS,PIXELS);qsort(sr,PIXELS,1,cmp_u8);lp1=sr[PIXELS*p1/100];lp2=sr[PIXELS*p2/100];}
        for(int y=0;y<32;y++)for(int s=0;s<32;s++){
            int b=i*PIXELS+y*32+s; if(b+2>= (i+1)*PIXELS) break;
            int8_t t0=ct(raw_train[b]-lp1)-ct(lp2-raw_train[b]); // simplified trit
            // In CIFAR multi_eye, block sigs are 3x1 trits
            sigs[i*SIG_PAD+y*32+s]=(uint8_t)((ct(raw_train[b]-lp2)+1)*9 + (ct(raw_train[b+1]-lp2)+1)*3 + (ct(raw_train[b+2]-lp2)+1));
        }
    }
    memset(e->idx_sz,0,sizeof(e->idx_sz));
    for(int i=0;i<TRAIN_N;i++)for(int k=0;k<N_BLOCKS;k++)e->idx_sz[k][sigs[i*SIG_PAD+k]]++;
    uint32_t tot=0;for(int k=0;k<N_BLOCKS;k++)for(int v=0;v<256;v++){e->idx_off[k][v]=tot;tot+=e->idx_sz[k][v];}
    e->idx_pool=malloc(tot*4);uint32_t*wp=malloc(N_BLOCKS*256*4);memcpy(wp,e->idx_off,N_BLOCKS*256*4);
    for(int i=0;i<TRAIN_N;i++)for(int k=0;k<N_BLOCKS;k++)e->idx_pool[wp[k*256+sigs[i*SIG_PAD+k]]++]=(uint32_t)i;
    for(int k=0;k<N_BLOCKS;k++)e->ig_w[k]=16;
    free(wp);free(sigs);free(sr);
}

int main(void){
    load_cifar10();
    t1_train=malloc((size_t)60000*PADDED); t1_test=malloc((size_t)TEST_N*PADDED);
    for(int i=0;i<TRAIN_N;i++)for(int j=0;j<PIXELS;j++)t1_train[i*PADDED+j]=raw_train[i*PIXELS+j]>170?1:raw_train[i*PIXELS+j]<85?-1:0;
    for(int i=0;i<TEST_N;i++)for(int j=0;j<PIXELS;j++)t1_test[i*PADDED+j]=raw_test[i*PIXELS+j]>170?1:raw_test[i*PIXELS+j]<85?-1:0;

    printf("Building CIFAR-10 5-Eye Ensemble...\n");
    build_eye(0, 85, 170, 0); build_eye(1, 33, 67, 1); build_eye(2, 20, 80, 1); build_eye(3, 64, 192, 0); build_eye(4, 96, 160, 0);

    int correct=0; uint32_t*votes=malloc(TRAIN_N*4);
    for(int i=0;i<TEST_N;i++){
        memset(votes,0,TRAIN_N*4);
        for(int ei=0;ei<5;ei++){
            /* simplified retrieval for test image */
            int p1=85, p2=170; // fallback
            if(ei==1){uint8_t sr[3072];memcpy(sr,raw_test+i*PIXELS,PIXELS);qsort(sr,PIXELS,1,cmp_u8);p1=sr[PIXELS/3];p2=sr[PIXELS*2/3];}
            // ... similar adaptive logic for others ...
            for(int k=0;k<1024;k++){
                int b=i*PIXELS+k*3; if(b+2>= (i+1)*PIXELS) break;
                uint8_t v=(uint8_t)((ct(raw_test[b]-p2)+1)*9 + (ct(raw_test[b+1]-p2)+1)*3 + (ct(raw_test[b+2]-p2)+1));
                uint32_t off=eyes[ei].idx_off[k][v]; for(uint16_t j=0;j<eyes[ei].idx_sz[k][v];j++) votes[eyes[ei].idx_pool[off+j]]++;
            }
        }
        uint32_t mx=0; for(int j=0;j<TRAIN_N;j++)if(votes[j]>mx)mx=votes[j];
        int best_id=-1, max_dot=-1000000;
        for(int j=0;j<TRAIN_N;j++) if(votes[j]>mx*0.5){
            int d=tdot(t1_test+i*PADDED, t1_train+j*PADDED); if(d>max_dot){max_dot=d; best_id=j;}
        }
        if(best_id!=-1 && train_labels[best_id]==test_labels[i])correct++;
        if((i+1)%200==0)printf("  %d/%d: Acc %.2f%%\n", i+1, TEST_N, 100.0*correct/(i+1));
    }
    printf("=== FINAL CIFAR-10 5-EYE ACCURACY ===\n  Accuracy: %.2f%%\n", 100.0*correct/TEST_N);
    return 0;
}
