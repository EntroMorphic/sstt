/*
 * sstt_taylor_specialist.c — Taylor Jet Space Specialist: Cat vs Dog
 *
 * This experiment tests if local surface primitives (Ridges/Saddles) 
 * can distinguish between the two most similar CIFAR-10 classes.
 *
 * Cat (Class 3) vs Dog (Class 5)
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    50000
#define TEST_N     2000 /* Test on all available cats/dogs in test set */
#define SP_PX      1024
#define IMG_W      32
#define IMG_H      32

#define N_PRIMS 8
#define G_DIM 8
#define FEAT_DIM (G_DIM * G_DIM * N_PRIMS)

static const char *data_dir = "data-cifar10/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint8_t *raw_gray_tr, *raw_gray_te, *train_labels, *test_labels;

static void load_cifar(void) {
    raw_gray_tr = malloc((size_t)TRAIN_N * SP_PX);
    raw_gray_te = malloc((size_t)10000 * SP_PX);
    train_labels = malloc(TRAIN_N);
    test_labels  = malloc(10000);
    uint8_t rec[3073];
    for(int b=1; b<=5; b++) {
        char p[256]; snprintf(p, 256, "%sdata_batch_%d.bin", data_dir, b);
        FILE *f = fopen(p, "rb");
        for(int i=0; i<10000; i++) {
            if(fread(rec, 1, 3073, f)!=3073) break;
            int idx = (b-1)*10000 + i;
            train_labels[idx] = rec[0];
            for(int p2=0; p2<1024; p2++) 
                raw_gray_tr[idx*1024 + p2] = (uint8_t)((77*rec[p2+1] + 150*rec[p2+1025] + 29*rec[p2+2049])>>8);
        }
        fclose(f);
    }
    FILE *f = fopen("data-cifar10/test_batch.bin", "rb");
    for(int i=0; i<10000; i++) {
        if(fread(rec, 1, 3073, f)!=3073) break;
        test_labels[i] = rec[0];
        for(int p2=0; p2<1024; p2++)
            raw_gray_te[i*1024 + p2] = (uint8_t)((77*rec[p2+1] + 150*rec[p2+1025] + 29*rec[p2+2049])>>8);
    }
    fclose(f);
}

static void extract_jet(const uint8_t *img, int16_t *feat) {
    int8_t tern[SP_PX];
    for(int i=0; i<SP_PX; i++) tern[i] = img[i]>170 ? 1 : img[i]<85 ? -1 : 0;
    memset(feat, 0, FEAT_DIM * sizeof(int16_t));
    for(int y=1; y<IMG_H-1; y++) {
        for(int x=1; x<IMG_W-1; x++) {
            int8_t p11=tern[(y-1)*IMG_W+x-1], p12=tern[(y-1)*IMG_W+x], p13=tern[(y-1)*IMG_W+x+1];
            int8_t p21=tern[y*IMG_W+x-1],      p22=tern[y*IMG_W+x],   p23=tern[y*IMG_W+x+1];
            int8_t p31=tern[(y+1)*IMG_W+x-1], p32=tern[(y+1)*IMG_W+x], p33=tern[(y+1)*IMG_W+x+1];
            int fxx=p23-2*p22+p21, fyy=p32-2*p22+p12, fxy=(p33-p31-p13+p11);
            int det=fxx*fyy-fxy*fxy, tr=fxx+fyy, g_mag=abs(p23-p21)+abs(p32-p12);
            int type=0;
            if (abs(fxx)+abs(fyy)+abs(fxy)>0) {
                if(det>0) type=(tr<0)?2:3; else if(det<0) type=4; else type=(tr<0)?5:6;
            } else if (g_mag>0) type=1;
            int ry=y*G_DIM/IMG_H, rx=x*G_DIM/IMG_W;
            feat[(ry*G_DIM+rx)*N_PRIMS + type]++;
        }
    }
}

int main() {
    load_cifar();
    printf("Training Cat vs Dog Specialist (Jet Space)...\n");
    
    /* Filter only cats (3) and dogs (5) */
    int tr_cnt=0, te_cnt=0;
    for(int i=0; i<TRAIN_N; i++) if(train_labels[i]==3 || train_labels[i]==5) tr_cnt++;
    for(int i=0; i<10000; i++)   if(test_labels[i]==3  || test_labels[i]==5)  te_cnt++;

    int16_t *f_tr = malloc((size_t)tr_cnt * FEAT_DIM * 2);
    uint8_t *l_tr = malloc(tr_cnt);
    int curr=0;
    for(int i=0; i<TRAIN_N; i++) if(train_labels[i]==3 || train_labels[i]==5) {
        extract_jet(raw_gray_tr + i*1024, f_tr + curr*FEAT_DIM);
        l_tr[curr] = train_labels[i];
        curr++;
    }

    int ok=0;
    for(int i=0; i<10000; i++) if(test_labels[i]==3 || test_labels[i]==5) {
        int16_t q[FEAT_DIM]; extract_jet(raw_gray_te + i*1024, q);
        int32_t best_d=1000000; int best_l=-1;
        for(int j=0; j<tr_cnt; j++) {
            int32_t d=0; for(int k=0; k<FEAT_DIM; k++) d += abs(q[k] - f_tr[j*FEAT_DIM+k]);
            if(d < best_d) { best_d = d; best_l = l_tr[j]; }
        }
        if(best_l == test_labels[i]) ok++;
    }

    printf("Results:\n");
    printf("  Accuracy (Cat vs Dog): %.2f%%\n", 100.0 * ok / te_cnt);
    printf("  (Random Baseline: 50.00%%)\n");
    return 0;
}
