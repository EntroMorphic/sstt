/*
 * sstt_cifar10_tracer.c — Discrete Structural Particle Tracing
 *
 * This experiment implements a true Lagrangian approach by tracing 
 * paths through the ternary gradient field.
 *
 * Algorithm:
 * 1. Identify "Seed" pixels (high magnitude ternary gradient).
 * 2. For each seed, follow the "Flow" (perpendicular to the gradient).
 * 3. Track path properties:
 *    - Length (how far does the edge go?)
 *    - Curvature (how much does the direction change?)
 *    - Closure (does it form a loop?)
 *
 * This should uniquely identify rounded edges (high closure) vs 
 * straight structural edges (low closure, long length).
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
#define SP_W 32
#define SP_H 32
#define SP_PX 1024
#define TOP_K 200

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;

/* ================================================================
 *  Tracer Logic
 * ================================================================ */

typedef struct {
    float avg_len;
    float closure_rate;
    float avg_curvature;
    int vortex_count;
} tracer_feat_t;

static void trace_image(const uint8_t *img, tracer_feat_t *feat) {
    int8_t hg[SP_PX], vg[SP_PX];
    for(int y=0;y<SP_H;y++)for(int x=0;x<SP_W;x++){
        int px = img[y*SP_W+x];
        int tx = (x < SP_W-1) ? (int)img[y*SP_W+x+1] : px;
        int ty = (y < SP_H-1) ? (int)img[(y+1)*SP_W+x] : px;
        hg[y*SP_W+x] = ct(((tx>170?1:tx<85?-1:0)) - ((px>170?1:px<85?-1:0)));
        vg[y*SP_W+x] = ct(((ty>170?1:ty<85?-1:0)) - ((px>170?1:px<85?-1:0)));
    }

    uint8_t visited[SP_PX] = {0};
    int total_len = 0, paths = 0, closures = 0, vortices = 0;
    float total_curv = 0;

    for(int y=1; y<SP_H-1; y++) for(int x=1; x<SP_W-1; x++) {
        int idx = y*SP_W+x;
        if(visited[idx]) continue;
        if(hg[idx] == 0 && vg[idx] == 0) continue;

        /* Trace path starting here */
        int cur_y = y, cur_x = x;
        int len = 0;
        int start_idx = idx;
        int prev_dx = 0, prev_dy = 0;

        while(len < 100) {
            int c_idx = cur_y*SP_W+cur_x;
            if(visited[c_idx] && len > 0) {
                if(c_idx == start_idx) closures++;
                break;
            }
            visited[c_idx] = 1;
            
            /* Flow direction is perpendicular to gradient */
            /* Grad = (hg, vg) -> Flow = (-vg, hg) */
            int dx = -vg[c_idx];
            int dy = hg[c_idx];
            
            if(dx == 0 && dy == 0) break;

            /* Track curvature: change in direction */
            if(len > 0) {
                if(dx != prev_dx || dy != prev_dy) total_curv += 1.0f;
            }
            prev_dx = dx; prev_dy = dy;

            cur_x += dx; cur_y += dy;
            if(cur_x < 0 || cur_x >= SP_W || cur_y < 0 || cur_y >= SP_H) break;
            len++;
        }
        if(len > 2) {
            total_len += len;
            paths++;
        }
    }

    /* Discrete Vorticity: count pixels with non-zero curl */
    for(int y=0;y<SP_H-1;y++)for(int x=0;x<SP_W-1;x++){
        int dv_dx = vg[y*SP_W+x+1] - vg[y*SP_W+x];
        int dh_dy = hg[(y+1)*SP_W+x] - hg[y*SP_W+x];
        if(abs(dv_dx - dh_dy) >= 2) vortices++;
    }

    feat->avg_len = paths ? (float)total_len / paths : 0;
    feat->closure_rate = paths ? (float)closures / paths : 0;
    feat->avg_curvature = total_len ? total_curv / total_len : 0;
    feat->vortex_count = vortices;
}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*SP_PX*3);raw_test=malloc((size_t)TEST_N*SP_PX*3);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    uint8_t rec[3073];
    FILE*f=fopen("data-cifar10/data_batch_1.bin","rb");
    for(int i=0;i<TRAIN_N;i++){fread(rec,1,3073,f);train_labels[i]=rec[0];
        const uint8_t*r=rec+1,*g=rec+1+1024,*b=rec+1+2048;
        uint8_t*gd=raw_train+(size_t)i*SP_PX;for(int p=0;p<SP_PX;p++)gd[p]=(uint8_t)((77*r[p]+150*g[p]+29*b[p])>>8);}
    fclose(f);
    f=fopen("data-cifar10/test_batch.bin","rb");
    for(int i=0;i<TEST_N;i++){fread(rec,1,3073,f);test_labels[i]=rec[0];
        const uint8_t*r=rec+1,*g=rec+1+1024,*b=rec+1+2048;
        uint8_t*gd=raw_test+(size_t)i*SP_PX;for(int p=0;p<SP_PX;p++)gd[p]=(uint8_t)((77*r[p]+150*g[p]+29*b[p])>>8);}
    fclose(f);
}

int main() {
    load_data();
    printf("Tracing paths...\n");
    tracer_feat_t *tr_feat = malloc(TRAIN_N * sizeof(tracer_feat_t));
    tracer_feat_t *te_feat = malloc(TEST_N * sizeof(tracer_feat_t));

    for(int i=0;i<TRAIN_N;i++) trace_image(raw_train+(size_t)i*SP_PX, &tr_feat[i]);
    for(int i=0;i<TEST_N;i++) trace_image(raw_test+(size_t)i*SP_PX, &te_feat[i]);

    printf("Evaluating kNN classification (Tracer only)...\n");
    int correct = 0;
    for(int i=0; i<TEST_N; i++) {
        float best_d = 1e30f;
        int best_l = -1;
        for(int j=0; j<TRAIN_N; j++) {
            /* Normalized L1 distance on tracer features */
            float d = fabsf(te_feat[i].avg_len - tr_feat[j].avg_len) / 5.0f
                    + fabsf(te_feat[i].closure_rate - tr_feat[j].closure_rate) * 10.0f
                    + fabsf(te_feat[i].avg_curvature - tr_feat[j].avg_curvature) * 5.0f
                    + (float)abs(te_feat[i].vortex_count - tr_feat[j].vortex_count) / 10.0f;
            if(d < best_d) {
                best_d = d;
                best_l = train_labels[j];
            }
        }
        if(best_l == test_labels[i]) correct++;
    }
    printf("  Tracer-only k=1: %.2f%%\n", 100.0f * correct / TEST_N);
    return 0;
}
