/*
 * sstt_benchmark_cifar10.c — Competitive Benchmark: Gauss vs Lagrangian vs Combo
 *
 * This script benchmarks the three top-tier rankers for CIFAR-10:
 * 1. RGB Gauss Map (First-Order Geometry) - Baseline Best
 * 2. Structural Lagrangian Tracer (Topological Skeleton) - New Challenger
 * 3. Unified Combo (Gauss + Tracer) - The Potential Step-Change
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
#define N_CLASSES 10
#define SP_W 32
#define SP_H 32
#define SP_PX 1024
#define TOP_K 200

#define GM_DIR 3
#define GM_MAG 5
#define GM_BINS (GM_DIR*GM_DIR*GM_MAG)
#define G4 4
#define GAUSS_DIM (G4*G4*GM_BINS*4) // RGB + Gray
#define TRACE_DIM (G4*G4*4)

static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;
static uint8_t *ch_tr[4], *ch_te[4]; // Gray, R, G, B

/* ================================================================
 *  Feature Extractors
 * ================================================================ */

static void extract_gauss(const uint8_t *img, int16_t *hist) {
    for(int y=0;y<SP_H;y++)for(int x=0;x<SP_W;x++){
        int gx=(x<SP_W-1)?(int)img[y*SP_W+x+1]-(int)img[y*SP_W+x]:0;
        int gy=(y<SP_H-1)?(int)img[(y+1)*SP_W+x]-(int)img[y*SP_W+x]:0;
        int dx=(gx>10)?2:(gx<-10)?0:1,dy=(gy>10)?2:(gy<-10)?0:1;
        int mag=abs(gx)+abs(gy),mb=(mag<5)?0:(mag<20)?1:(mag<50)?2:(mag<100)?3:4;
        int bin=dx*GM_DIR*GM_MAG+dy*GM_MAG+mb;
        int ry=y*G4/SP_H,rx=x*G4/SP_W;
        hist[(ry*G4+rx)*GM_BINS+bin]++;
    }
}

static void extract_tracer(const uint8_t *img, float *feat) {
    int8_t hg[SP_PX], vg[SP_PX];
    for(int y=0;y<SP_H;y++)for(int x=0;x<SP_W;x++){
        int px = img[y*SP_W+x];
        int tx = (x < SP_W-1) ? (int)img[y*SP_W+x+1] : px;
        int ty = (y < SP_H-1) ? (int)img[(y+1)*SP_W+x] : px;
        hg[y*SP_W+x] = ct(((tx>170?1:tx<85?-1:0)) - ((px>170?1:px<85?-1:0)));
        vg[y*SP_W+x] = ct(((ty>170?1:ty<85?-1:0)) - ((px>170?1:px<85?-1:0)));
    }
    uint8_t visited[SP_PX] = {0};
    float glen[16]={0}, gcurv[16]={0}, gdens[16]={0}, gvort[16]={0}; int gpaths[16]={0};
    for(int y=1; y<SP_H-1; y++) for(int x=1; x<SP_W-1; x++) {
        int idx = y*SP_W+x;
        if(visited[idx] || (hg[idx]==0 && vg[idx]==0)) continue;
        int ridx = (y*G4/SP_H)*G4 + (x*G4/SP_W);
        gdens[ridx]++;
        int cy=y, cx=x, len=0; float curv=0; int pdx=0, pdy=0;
        while(len < 50) {
            int c_idx = cy*SP_W+cx;
            if(visited[c_idx]) break; visited[c_idx] = 1;
            int dx = -vg[c_idx], dy = hg[c_idx];
            if(dx==0 && dy==0) break;
            if(len>0 && (dx!=pdx || dy!=pdy)) curv += 1.0f;
            pdx=dx; pdy=dy; cx+=dx; cy+=dy;
            if(cx<0 || cx>=SP_W || cy<0 || cy>=SP_H) break;
            len++;
        }
        if(len > 1) { glen[ridx]+=len; gcurv[ridx]+=curv; gpaths[ridx]++; }
    }
    for(int y=0;y<SP_H-1;y++)for(int x=0;x<SP_W-1;x++){
        int curl = (vg[y*SP_W+x+1]-vg[y*SP_W+x]) - (hg[(y+1)*SP_W+x]-hg[y*SP_W+x]);
        gvort[(y*G4/SP_H)*G4 + (x*G4/SP_W)] += (float)abs(curl);
    }
    for(int i=0; i<16; i++) {
        feat[i*4+0] = gdens[i] / 64.0f;
        feat[i*4+1] = gpaths[i] ? glen[i]/gpaths[i]/10.0f : 0;
        feat[i*4+2] = gpaths[i] ? gcurv[i]/gpaths[i] : 0;
        feat[i*4+3] = gvort[i] / 32.0f;
    }
}

/* ================================================================
 *  Benchmark Framework
 * ================================================================ */

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*SP_PX);raw_test=malloc((size_t)TEST_N*SP_PX);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    for(int i=0;i<4;i++){ ch_tr[i]=malloc((size_t)TRAIN_N*SP_PX); ch_te[i]=malloc((size_t)TEST_N*SP_PX); }
    uint8_t rec[3073];
    for(int b=1;b<=5;b++){
        char p[256]; snprintf(p,256,"data-cifar10/data_batch_%d.bin",b);
        FILE*f=fopen(p,"rb"); if(!f) continue;
        for(int i=0;i<10000;i++){
            int idx = (b-1)*10000+i; if(idx>=TRAIN_N) break;
            fread(rec,1,3073,f); train_labels[idx]=rec[0];
            const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            memcpy(ch_tr[1]+(size_t)idx*SP_PX, r, SP_PX);
            memcpy(ch_tr[2]+(size_t)idx*SP_PX, g, SP_PX);
            memcpy(ch_tr[3]+(size_t)idx*SP_PX, b3, SP_PX);
            for(int p2=0;p2<SP_PX;p2++) ch_tr[0][idx*SP_PX+p2]=(uint8_t)((77*r[p2]+150*g[p2]+29*b3[p2])>>8);
        }
        fclose(f);
    }
    FILE*f=fopen("data-cifar10/test_batch.bin","rb");
    for(int i=0;i<TEST_N;i++){
        fread(rec,1,3073,f); test_labels[i]=rec[0];
        const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        memcpy(ch_te[1]+(size_t)i*SP_PX, r, SP_PX);
        memcpy(ch_te[2]+(size_t)i*SP_PX, g, SP_PX);
        memcpy(ch_te[3]+(size_t)i*SP_PX, b3, SP_PX);
        for(int p2=0;p2<SP_PX;p2++) ch_te[0][i*SP_PX+p2]=(uint8_t)((77*r[p2]+150*g[p2]+29*b3[p2])>>8);
    }
    fclose(f);
}

int main() {
    load_data();
    printf("Extracting Features (Gauss + Tracer)...\n");
    int16_t *gauss_tr = malloc((size_t)TRAIN_N * GAUSS_DIM * 2);
    int16_t *gauss_te = malloc((size_t)TEST_N * GAUSS_DIM * 2);
    float *trace_tr = malloc((size_t)TRAIN_N * TRACE_DIM * 4);
    float *trace_te = malloc((size_t)TEST_N * TRACE_DIM * 4);

    double t0 = now_sec();
    for(int i=0;i<TRAIN_N;i++){
        for(int ch=0;ch<4;ch++) extract_gauss(ch_tr[ch]+(size_t)i*SP_PX, gauss_tr+(size_t)i*GAUSS_DIM+ch*(G4*G4*GM_BINS));
        extract_tracer(ch_tr[0]+(size_t)i*SP_PX, trace_tr+(size_t)i*TRACE_DIM);
    }
    for(int i=0;i<TEST_N;i++){
        for(int ch=0;ch<4;ch++) extract_gauss(ch_te[ch]+(size_t)i*SP_PX, gauss_te+(size_t)i*GAUSS_DIM+ch*(G4*G4*GM_BINS));
        extract_tracer(ch_te[0]+(size_t)i*SP_PX, trace_te+(size_t)i*TRACE_DIM);
    }
    printf("Extraction complete in %.2f sec\n\n", now_sec()-t0);

    printf("=== BENCHMARK (k=1 Brute) ===\n");
    int c_gauss=0, c_trace=0, c_combo=0;
    double t_start = now_sec();
    for(int i=0; i<TEST_N; i++) {
        float b_gauss=1e30f, b_trace=1e30f, b_combo=1e30f;
        int l_gauss=-1, l_trace=-1, l_combo=-1;
        for(int j=0; j<TRAIN_N; j++) {
            float dg=0, dt=0;
            for(int k=0; k<GAUSS_DIM; k++) dg += (float)abs(gauss_te[i*GAUSS_DIM+k] - gauss_tr[j*GAUSS_DIM+k]);
            for(int k=0; k<TRACE_DIM; k++) dt += fabsf(trace_te[i*TRACE_DIM+k] - trace_tr[j*TRACE_DIM+k]);
            
            if(dg < b_gauss) { b_gauss=dg; l_gauss=train_labels[j]; }
            if(dt < b_trace) { b_trace=dt; l_trace=train_labels[j]; }
            
            /* Log-domain Combo: Gauss is primary, Tracer is precision support */
            /* We need to balance a distance of ~10000 (Gauss) with ~20 (Tracer) */
            float dc = logf(dg + 1.0f) + logf(dt + 1.0f) * 5.0f;
            if(dc < b_combo) { b_combo=dc; l_combo=train_labels[j]; }
        }
        if(l_gauss == test_labels[i]) c_gauss++;
        if(l_trace == test_labels[i]) c_trace++;
        if(l_combo == test_labels[i]) c_combo++;
        if((i+1)%100==0) fprintf(stderr, "  Benchmarking: %d/%d\r", i+1, TEST_N);
    }
    double t_end = now_sec();

    printf("\n\nResults (N=%d):\n", TEST_N);
    printf("  1. RGB Gauss Baseline:  %.2f%%\n", 100.0f*c_gauss/TEST_N);
    printf("  2. Lagrangian Tracer:   %.2f%%\n", 100.0f*c_trace/TEST_N);
    printf("  3. Unified Combo:       %.2f%%\n", 100.0f*c_combo/TEST_N);
    printf("\nTotal search time: %.2f sec (%.2f ms/query)\n", t_end-t_start, (t_end-t_start)*1000.0/TEST_N);
    
    return 0;
}
