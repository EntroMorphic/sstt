/*
 * sstt_cifar10_grid_tracer.c — 4x4 Grid Structural Particle Tracing
 *
 * This experiment expands the Lagrangian tracer to a 4x4 grid.
 * Each of 16 regions tracks:
 * - Edge Density (active gradient pixels)
 * - Structural Length (avg path length)
 * - Structural Curvature (avg change in flow direction)
 * - Vortex Intensity (sum of absolute curl)
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 5000
#define TEST_N  200
#define N_CLASSES 10
#define SP_W 32
#define SP_H 32
#define SP_PX 1024
#define G4 4
#define FEAT_DIM (G4*G4*4)

static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train,*raw_test,*train_labels,*test_labels;

static void compute_grid_tracer(const uint8_t *img, float *feat) {
    int8_t hg[SP_PX], vg[SP_PX];
    for(int y=0;y<SP_H;y++)for(int x=0;x<SP_W;x++){
        int px = img[y*SP_W+x];
        int tx = (x < SP_W-1) ? (int)img[y*SP_W+x+1] : px;
        int ty = (y < SP_H-1) ? (int)img[(y+1)*SP_W+x] : px;
        hg[y*SP_W+x] = ct(((tx>170?1:tx<85?-1:0)) - ((px>170?1:px<85?-1:0)));
        vg[y*SP_W+x] = ct(((ty>170?1:ty<85?-1:0)) - ((px>170?1:px<85?-1:0)));
    }

    float grid_len[16]={0}, grid_curv[16]={0}, grid_vortex[16]={0}, grid_density[16]={0};
    int grid_paths[16]={0};

    uint8_t visited[SP_PX] = {0};
    for(int y=1; y<SP_H-1; y++) for(int x=1; x<SP_W-1; x++) {
        int idx = y*SP_W+x;
        if(visited[idx] || (hg[idx]==0 && vg[idx]==0)) continue;

        int ry=y*G4/SP_H, rx=x*G4/SP_W; int ridx = ry*G4+rx;
        grid_density[ridx]++;

        int cur_y = y, cur_x = x, len = 0;
        float curv = 0; int prev_dx=0, prev_dy=0;
        while(len < 50) {
            int c_idx = cur_y*SP_W+cur_x;
            if(visited[c_idx]) break;
            visited[c_idx] = 1;
            int dx = -vg[c_idx], dy = hg[c_idx];
            if(dx==0 && dy==0) break;
            if(len>0 && (dx!=prev_dx || dy!=prev_dy)) curv += 1.0f;
            prev_dx=dx; prev_dy=dy;
            cur_x += dx; cur_y += dy;
            if(cur_x<0 || cur_x>=SP_W || cur_y<0 || cur_y>=SP_H) break;
            len++;
        }
        if(len > 1) {
            grid_len[ridx] += (float)len;
            grid_curv[ridx] += curv;
            grid_paths[ridx]++;
        }
    }

    for(int y=0;y<SP_H-1;y++)for(int x=0;x<SP_W-1;x++){
        int curl = (vg[y*SP_W+x+1]-vg[y*SP_W+x]) - (hg[(y+1)*SP_W+x]-hg[y*SP_W+x]);
        int ridx = (y*G4/SP_H)*G4 + (x*G4/SP_W);
        grid_vortex[ridx] += (float)abs(curl);
    }

    for(int i=0; i<16; i++) {
        feat[i*4+0] = grid_density[i] / 64.0f;
        feat[i*4+1] = grid_paths[i] ? grid_len[i] / grid_paths[i] / 10.0f : 0;
        feat[i*4+2] = grid_paths[i] ? grid_curv[i] / grid_paths[i] : 0;
        feat[i*4+3] = grid_vortex[i] / 32.0f;
    }
}

static void load_data(void){
    raw_train=malloc((size_t)TRAIN_N*SP_PX);raw_test=malloc((size_t)TEST_N*SP_PX);
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
    printf("Computing Grid Tracer features...\n");
    float *tr_feat = malloc(TRAIN_N * FEAT_DIM * sizeof(float));
    float *te_feat = malloc(TEST_N * FEAT_DIM * sizeof(float));
    for(int i=0;i<TRAIN_N;i++) compute_grid_tracer(raw_train+(size_t)i*SP_PX, tr_feat + i*FEAT_DIM);
    for(int i=0;i<TEST_N;i++) compute_grid_tracer(raw_test+(size_t)i*SP_PX, te_feat + i*FEAT_DIM);

    printf("Evaluating kNN (Grid Tracer)...\n");
    int correct = 0;
    for(int i=0; i<TEST_N; i++) {
        float best_d = 1e30f; int best_l = -1;
        for(int j=0; j<TRAIN_N; j++) {
            float d = 0;
            for(int k=0; k<FEAT_DIM; k++) d += fabsf(te_feat[i*FEAT_DIM+k] - tr_feat[j*FEAT_DIM+k]);
            if(d < best_d) { best_d = d; best_l = train_labels[j]; }
        }
        if(best_l == test_labels[i]) correct++;
    }
    printf("  Grid Tracer k=1: %.2f%%\n", 100.0f * correct / TEST_N);
    return 0;
}
