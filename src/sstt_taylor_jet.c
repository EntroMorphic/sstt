/*
 * sstt_taylor_jet.c — Geometric Jet Space: 2nd-Order Surface Classification
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    50000
#define TEST_N     500
#define N_CLASSES  10
#define IMG_W      32
#define IMG_H      32
#define PIXELS     1024

#define N_PRIMS 8
#define G_DIM 8
#define FEAT_DIM (G_DIM * G_DIM * N_PRIMS)

static const char *data_dir = "data-cifar10/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint8_t *raw_train, *raw_test, *train_labels, *test_labels;

/* === CIFAR-10 Loader === */
static void load_cifar(void) {
    raw_train = malloc((size_t)TRAIN_N * PIXELS);
    raw_test  = malloc((size_t)TEST_N * PIXELS);
    train_labels = malloc(TRAIN_N);
    test_labels  = malloc(TEST_N);
    uint8_t rec[3073];
    for(int b=1; b<=5; b++) {
        char p[256]; snprintf(p, 256, "%sdata_batch_%d.bin", data_dir, b);
        FILE *f = fopen(p, "rb"); if(!f) { printf("fail %s\n", p); exit(1); }
        for(int i=0; i<10000; i++) {
            fread(rec, 1, 3073, f);
            int idx = (b-1)*10000 + i;
            train_labels[idx] = rec[0];
            /* Grayscale conversion */
            for(int p2=0; p2<1024; p2++) 
                raw_train[idx*1024 + p2] = (uint8_t)((77*rec[p2+1] + 150*rec[p2+1025] + 29*rec[p2+2049])>>8);
        }
        fclose(f);
    }
    FILE *f = fopen("data-cifar10/test_batch.bin", "rb");
    for(int i=0; i<TEST_N; i++) {
        fread(rec, 1, 3073, f);
        test_labels[i] = rec[0];
        for(int p2=0; p2<1024; p2++)
            raw_test[i*1024 + p2] = (uint8_t)((77*rec[p2+1] + 150*rec[p2+1025] + 29*rec[p2+2049])>>8);
    }
    fclose(f);
}

/* === Taylor Jet Extraction === */
static void extract_jet_features(const uint8_t *img, int16_t *feat) {
    int8_t tern[PIXELS];
    for(int i=0; i<PIXELS; i++) tern[i] = img[i]>170 ? 1 : img[i]<85 ? -1 : 0;
    memset(feat, 0, FEAT_DIM * sizeof(int16_t));
    for(int y=1; y<IMG_H-1; y++) {
        for(int x=1; x<IMG_W-1; x++) {
            int8_t p11=tern[(y-1)*IMG_W+x-1], p12=tern[(y-1)*IMG_W+x], p13=tern[(y-1)*IMG_W+x+1];
            int8_t p21=tern[y*IMG_W+x-1],      p22=tern[y*IMG_W+x],   p23=tern[y*IMG_W+x+1];
            int8_t p31=tern[(y+1)*IMG_W+x-1], p32=tern[(y+1)*IMG_W+x], p33=tern[(y+1)*IMG_W+x+1];
            int fx=p23-p21, fy=p32-p12, fxx=p23-2*p22+p21, fyy=p32-2*p22+p12, fxy=(p33-p31-p13+p11);
            int det=fxx*fyy-fxy*fxy, tr=fxx+fyy, g_mag=abs(fx)+abs(fy);
            int type=0;
            if (abs(fxx)+abs(fyy)+abs(fxy)>0) {
                if(det>0) type=(tr<0)?2:3; else if(det<0) type=4; else type=(tr<0)?5:6;
            } else if (g_mag>0) type=1;
            int ry=y*G_DIM/IMG_H, rx=x*G_DIM/IMG_W;
            feat[(ry*G_DIM+rx)*N_PRIMS + type]++;
        }
    }
}

static int32_t dist_l1(const int16_t *a, const int16_t *b, int len) {
    int32_t d=0; for(int i=0;i<len;i++) d+=abs(a[i]-b[i]); return d;
}

int main(int argc, char** argv) {
    if(argc>1) data_dir=argv[1];
    load_cifar();
    printf("Extracting Taylor Jet (CIFAR-10 grayscale)...\n");
    int16_t *tr_feat = malloc((size_t)TRAIN_N * FEAT_DIM * 2);
    int16_t *te_feat = malloc((size_t)TEST_N * FEAT_DIM * 2);
    double t0 = now_sec();
    for(int i=0; i<TRAIN_N; i++) extract_jet_features(raw_train + (size_t)i*PIXELS, tr_feat + i*FEAT_DIM);
    for(int i=0; i<TEST_N; i++)  extract_jet_features(raw_test + (size_t)i*PIXELS, te_feat + i*FEAT_DIM);
    printf("  Extraction complete (%.2f sec)\n", now_sec()-t0);
    printf("=== Taylor Jet Benchmark (CIFAR-10 k=1 Brute) ===\n");
    int correct=0;
    for(int i=0; i<TEST_N; i++) {
        int32_t best_d=1000000; int best_l=-1;
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
