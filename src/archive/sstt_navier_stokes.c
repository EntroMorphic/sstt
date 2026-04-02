/*
 * sstt_navier_stokes.c — Shape as Fluid Flow: Steady-State Streamlines
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    30000 
#define TEST_N     500
#define N_CLASSES  10
#define IMG_W      28
#define IMG_H      28
#define PIXELS     784
#define PADDED     800

#define ITER       30 /* More iterations for convergence */
#define G_DIM      4
#define N_BINS     8
#define FEAT_DIM   (G_DIM * G_DIM * N_BINS)

static const char *data_dir = "data/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static inline int8_t ct(int v){return v>0?1:v<0?-1:0;}

static uint8_t *raw_train, *raw_test, *train_labels, *test_labels;

/* === Boilerplate === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}

/* === Poisson Stream Function Solver === */

static void solve_stream_function(const uint8_t *img, int16_t *feat) {
    /* 1. Gradients from RAW 8-BIT */
    int16_t gx[PIXELS] = {0}, gy[PIXELS] = {0};
    for(int y=0; y<IMG_H-1; y++) {
        for(int x=0; x<IMG_W-1; x++) {
            gx[y*IMG_W+x] = (int16_t)img[y*IMG_W+x+1] - (int16_t)img[y*IMG_W+x];
            gy[y*IMG_W+x] = (int16_t)img[(y+1)*IMG_W+x] - (int16_t)img[y*IMG_W+x];
        }
    }

    /* 2. Vorticity from RAW Gradients */
    int16_t curl[PIXELS] = {0};
    for(int y=0; y<IMG_H-1; y++) {
        for(int x=0; x<IMG_W-1; x++) {
            int16_t dv_dx = gy[y*IMG_W+x+1] - gy[y*IMG_W+x];
            int16_t du_dy = gx[(y+1)*IMG_W+x] - gx[y*IMG_W+x];
            curl[y*IMG_W+x] = dv_dx - du_dy;
        }
    }

    /* 3. Poisson Solve (Jacobi) */
    int32_t psi[PIXELS] = {0};
    int32_t next_psi[PIXELS] = {0};
    for(int it=0; it<ITER; it++) {
        for(int y=1; y<IMG_H-1; y++) {
            for(int x=1; x<IMG_W-1; x++) {
                int32_t sum = psi[(y-1)*IMG_W+x] + psi[(y+1)*IMG_W+x] + 
                             psi[y*IMG_W+x-1] + psi[y*IMG_W+x+1];
                next_psi[y*IMG_W+x] = (sum + (int32_t)curl[y*IMG_W+x] * 16) / 4;
            }
        }
        memcpy(psi, next_psi, sizeof(psi));
    }

    /* 4. Histogram features */
    memset(feat, 0, FEAT_DIM * sizeof(int16_t));
    for(int y=0; y<IMG_H; y++) {
        for(int x=0; x<IMG_W; x++) {
            int bin = (psi[y*IMG_W+x] + 512) / 128;
            if(bin<0) bin=0; if(bin>=N_BINS) bin=N_BINS-1;
            int ry = y * G_DIM / IMG_H, rx = x * G_DIM / IMG_W;
            feat[(ry * G_DIM + rx) * N_BINS + bin]++;
        }
    }
}

static int32_t dist_l1(const int16_t *a, const int16_t *b, int len) {
    int32_t d = 0;
    for(int i=0; i<len; i++) d += abs(a[i] - b[i]);
    return d;
}

int main(int argc, char** argv) {
    if(argc > 1) data_dir = argv[1];
    load_data();
    printf("Computing Stream Function (Navier-Stokes) Features...\n");
    int16_t *tr_feat = malloc((size_t)TRAIN_N * FEAT_DIM * 2);
    int16_t *te_feat = malloc((size_t)TEST_N * FEAT_DIM * 2);
    double t0 = now_sec();
    for(int i=0; i<TRAIN_N; i++) solve_stream_function(raw_train + (size_t)i*PIXELS, tr_feat + i*FEAT_DIM);
    for(int i=0; i<TEST_N; i++)  solve_stream_function(raw_test + (size_t)i*PIXELS, te_feat + i*FEAT_DIM);
    printf("  Solve complete (%.2f sec)\n\n", now_sec()-t0);
    printf("=== Navier-Stokes Benchmark (k=1 Brute) ===\n");
    int correct = 0;
    for(int i=0; i<TEST_N; i++) {
        int32_t best_d = 2000000000; int best_l = -1;
        const int16_t *qi = te_feat + i*FEAT_DIM;
        for(int j=0; j<TRAIN_N; j++) {
            int32_t d = dist_l1(qi, tr_feat + j*FEAT_DIM, FEAT_DIM);
            if (d < best_d) { best_d = d; best_l = train_labels[j]; }
        }
        if (best_l == test_labels[i]) correct++;
        if ((i+1)%100 == 0) fprintf(stderr, "  %d/%d\r", i+1, TEST_N);
    }
    printf("\n  Accuracy: %.2f%%\n", 100.0 * correct / TEST_N);
    return 0;
}
