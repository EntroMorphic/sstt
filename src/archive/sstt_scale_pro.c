/*
 * sstt_scale_pro.c — High-Resolution Scaling: Eigenvalue-Weighted Knot Placement
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    30000
#define TEST_N     200
#define SRC_DIM    28
#define TGT_DIM    224
#define SCALE      (TGT_DIM / SRC_DIM)

#define N_CLASSES  10
#define CLS_PAD    16

#define BLKS_X     (TGT_DIM / 3) /* 74 */
#define BLKS_Y     TGT_DIM       /* 224 */
#define TOTAL_BLKS (BLKS_X * BLKS_Y) /* 16,576 */
#define N_KNOTS    512 /* Compressed model: ~3% of total blocks */

static const char *data_dir = "data/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint8_t *raw_train, *raw_test, *train_labels, *test_labels;

/* Compressed Model */
static uint32_t *knot_map; 
static int knot_indices[N_KNOTS];
static int block_to_knot[TOTAL_BLKS]; 

/* === Boilerplate === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline uint8_t benc(int8_t a, int8_t b, int8_t c) { return (uint8_t)((a + 1) * 9 + (b + 1) * 3 + (c + 1)); }
static void get_block_sigs(const uint8_t *img_224, uint8_t *sigs) {
    for (int y = 0; y < TGT_DIM; y++) {
        for (int x = 0; x < BLKS_X; x++) {
            int px = y * TGT_DIM + x * 3;
            int8_t t0 = img_224[px] > 170 ? 1 : img_224[px] < 85 ? -1 : 0;
            int8_t t1 = img_224[px+1] > 170 ? 1 : img_224[px+1] < 85 ? -1 : 0;
            int8_t t2 = img_224[px+2] > 170 ? 1 : img_224[px+2] < 85 ? -1 : 0;
            sigs[y * BLKS_X + x] = benc(t0, t1, t2);
        }
    }
}
static void resize_224(const uint8_t *src_28, uint8_t *dst_224) {
    for (int y = 0; y < TGT_DIM; y++) for (int x = 0; x < TGT_DIM; x++) dst_224[y * TGT_DIM + x] = src_28[(y / SCALE) * SRC_DIM + (x / SCALE)];
}

/* === Strategies === */
typedef struct { int bid; float var; } var_info_t;
static int cmp_var(const void *a, const void *b) { return (((var_info_t*)b)->var > ((var_info_t*)a)->var) ? 1 : -1; }

static void select_knots_variance(void) {
    float *variance = calloc(TOTAL_BLKS, sizeof(float)), *mean = calloc(TOTAL_BLKS, sizeof(float));
    uint8_t img[TGT_DIM * TGT_DIM], sigs[TOTAL_BLKS];
    for (int i = 0; i < 1000; i++) {
        resize_224(raw_train + (size_t)i * 784, img); get_block_sigs(img, sigs);
        for (int k = 0; k < TOTAL_BLKS; k++) {
            float val = (float)sigs[k], delta = val - mean[k];
            mean[k] += delta / (i + 1); variance[k] += delta * (val - mean[k]);
        }
    }
    var_info_t *infos = malloc(TOTAL_BLKS * sizeof(var_info_t));
    for (int k = 0; k < TOTAL_BLKS; k++) { infos[k].bid = k; infos[k].var = variance[k]; }
    qsort(infos, TOTAL_BLKS, sizeof(var_info_t), cmp_var);
    for (int i = 0; i < N_KNOTS; i++) knot_indices[i] = infos[i].bid;
    for (int k = 0; k < TOTAL_BLKS; k++) {
        int ky = k / BLKS_X, kx = k % BLKS_X, best_knot = 0, best_dist = 1000000;
        for (int j = 0; j < N_KNOTS; j++) {
            int jid = knot_indices[j], jy = jid / BLKS_X, jx = jid % BLKS_X;
            int dist = (ky-jy)*(ky-jy) + (kx-jx)*(kx-jx);
            if (dist < best_dist) { best_dist = dist; best_knot = j; }
            if (dist == 0) break;
        }
        block_to_knot[k] = best_knot;
    }
    free(variance); free(mean); free(infos);
}

static void select_knots_uniform(void) {
    for (int i = 0; i < N_KNOTS; i++) knot_indices[i] = (TOTAL_BLKS / N_KNOTS) * i;
    for (int k = 0; k < TOTAL_BLKS; k++) {
        int ky = k / BLKS_X, kx = k % BLKS_X, best_knot = 0, best_dist = 1000000;
        for (int j = 0; j < N_KNOTS; j++) {
            int jid = knot_indices[j], jy = jid / BLKS_X, jx = jid % BLKS_X;
            int dist = (ky-jy)*(ky-jy) + (kx-jx)*(kx-jx);
            if (dist < best_dist) { best_dist = dist; best_knot = j; }
            if (dist == 0) break;
        }
        block_to_knot[k] = best_knot;
    }
}

static void build_knot_map(void) {
    knot_map = calloc((size_t)N_KNOTS * 27 * CLS_PAD, 4);
    uint8_t img[TGT_DIM * TGT_DIM], sigs[TOTAL_BLKS];
    for (int i = 0; i < TRAIN_N; i++) {
        resize_224(raw_train + (size_t)i * 784, img); get_block_sigs(img, sigs);
        int lbl = train_labels[i];
        for (int k = 0; k < TOTAL_BLKS; k++) {
            int kid = block_to_knot[k];
            knot_map[(size_t)kid * 27 * CLS_PAD + sigs[k] * CLS_PAD + lbl]++;
        }
    }
}

static int classify(const uint8_t *sigs) {
    uint32_t acc[CLS_PAD] __attribute__((aligned(32))) = {0};
    for (int k = 0; k < TOTAL_BLKS; k++) {
        if (sigs[k] == 13) continue;
        int kid = block_to_knot[k];
        const uint32_t *row = &knot_map[(size_t)kid * 27 * CLS_PAD + sigs[k] * CLS_PAD];
        for (int c = 0; c < 10; c++) acc[c] += row[c];
    }
    int best = 0; for (int c = 1; c < 10; c++) if (acc[c] > acc[best]) best = c;
    return best;
}

int main(int argc, char** argv) {
    if(argc > 1) data_dir = argv[1];
    load_data();
    uint8_t img[TGT_DIM * TGT_DIM], sigs[TOTAL_BLKS];

    printf("Strategy 1: Uniform Knot Placement...\n");
    select_knots_uniform(); build_knot_map();
    int ok_u = 0; for (int i = 0; i < TEST_N; i++) {
        resize_224(raw_test + (size_t)i * 784, img); get_block_sigs(img, sigs);
        if (classify(sigs) == test_labels[i]) ok_u++;
    }
    free(knot_map);

    printf("Strategy 2: Variance-Guided Knot Placement...\n");
    select_knots_variance(); build_knot_map();
    int ok_v = 0; for (int i = 0; i < TEST_N; i++) {
        resize_224(raw_test + (size_t)i * 784, img); get_block_sigs(img, sigs);
        if (classify(sigs) == test_labels[i]) ok_v++;
    }

    printf("\n=== 224x224 Scaling (N_KNOTS=%d) ===\n", N_KNOTS);
    printf("  Uniform Accuracy:  %.2f%%\n", 100.0 * ok_u / TEST_N);
    printf("  Variance Accuracy: %.2f%%\n", 100.0 * ok_v / TEST_N);
    printf("  Improvement:       %+.2fpp\n", 100.0 * (ok_v - ok_u) / TEST_N);
    return 0;
}
