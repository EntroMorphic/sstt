/*
 * sstt_scale_hierarchical.c — Hierarchical Geometric Scaling (224x224)
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    20000 
#define TEST_N     1000
#define SRC_DIM    28
#define TGT_DIM    224
#define SCALE      8

#define N_CLASSES  10
#define CLS_PAD    16

#define L0_DIM     224
#define L0_BLKS_X  (L0_DIM / 3) 
#define L0_BLKS_Y  L0_DIM       
#define L0_TOTAL_BLKS (L0_BLKS_X * L0_BLKS_Y) 
#define L0_KNOTS_X 28
#define L0_KNOTS_Y 28
#define L0_N_KNOTS (L0_KNOTS_X * L0_KNOTS_Y) 

#define L1_DIM     56
#define L1_BLKS_X  (L1_DIM / 3) 
#define L1_BLKS_Y  L1_DIM       
#define L1_TOTAL_BLKS (L1_BLKS_X * L1_BLKS_Y) 
#define L1_KNOTS_X 14
#define L1_KNOTS_Y 14
#define L1_N_KNOTS (L1_KNOTS_X * L1_KNOTS_Y) 

static const char *data_dir = "data-fashion/";
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static uint8_t *raw_train, *raw_test, *train_labels, *test_labels;
static uint32_t *l0_map, *l1_map;
static int l0_block_to_knot[L0_TOTAL_BLKS];
static int l1_block_to_knot[L1_TOTAL_BLKS];

typedef struct { int bid; uint16_t ig; } blk_info_t;
static int cmp_blk(const void *a, const void *b) { return ((blk_info_t*)b)->ig - ((blk_info_t*)a)->ig; }
static blk_info_t l0_sorted[L0_TOTAL_BLKS];

/* === Boilerplate === */
static uint8_t *load_idx(const char*path,uint32_t*cnt,uint32_t*ro,uint32_t*co){FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"ERR:%s\n",path);exit(1);}uint32_t m,n;if(fread(&m,4,1,f)!=1||fread(&n,4,1,f)!=1){fclose(f);exit(1);}m=__builtin_bswap32(m);n=__builtin_bswap32(n);*cnt=n;size_t s=1;if((m&0xFF)>=3){uint32_t r,c;if(fread(&r,4,1,f)!=1||fread(&c,4,1,f)!=1){fclose(f);exit(1);}r=__builtin_bswap32(r);c=__builtin_bswap32(c);if(ro)*ro=r;if(co)*co=c;s=(size_t)r*c;}else{if(ro)*ro=0;if(co)*co=0;}size_t total=(size_t)n*s;uint8_t*d=malloc(total);if(!d||fread(d,1,total,f)!=total){fclose(f);exit(1);}fclose(f);return d;}
static void load_data(void){uint32_t n,r,c;char p[256];snprintf(p,sizeof(p),"%strain-images-idx3-ubyte",data_dir);raw_train=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%strain-labels-idx1-ubyte",data_dir);train_labels=load_idx(p,&n,NULL,NULL);snprintf(p,sizeof(p),"%st10k-images-idx3-ubyte",data_dir);raw_test=load_idx(p,&n,&r,&c);snprintf(p,sizeof(p),"%st10k-labels-idx1-ubyte",data_dir);test_labels=load_idx(p,&n,NULL,NULL);}
static inline uint8_t benc(int8_t a, int8_t b, int8_t c) { return (uint8_t)((a + 1) * 9 + (b + 1) * 3 + (c + 1)); }
static void resize_224(const uint8_t *src_28, uint8_t *dst_224) { for (int y = 0; y < TGT_DIM; y++) for (int x = 0; x < TGT_DIM; x++) dst_224[y * TGT_DIM + x] = src_28[(y / SCALE) * SRC_DIM + (x / SCALE)]; }

/* === Block Extraction === */
static void get_l0_sigs(const uint8_t *img_224, uint8_t *sigs) {
    for (int y = 0; y < L0_DIM; y++) {
        for (int x = 0; x < L0_BLKS_X; x++) {
            int px = y * L0_DIM + x * 3;
            int8_t t0 = img_224[px] > 170 ? 1 : img_224[px] < 85 ? -1 : 0;
            int8_t t1 = img_224[px+1] > 170 ? 1 : img_224[px+1] < 85 ? -1 : 0;
            int8_t t2 = img_224[px+2] > 170 ? 1 : img_224[px+2] < 85 ? -1 : 0;
            sigs[y * L0_BLKS_X + x] = benc(t0, t1, t2);
        }
    }
}
static void get_l1_sigs(const uint8_t *img_224, uint8_t *sigs) {
    uint8_t pooled[56*56];
    for(int y=0; y<56; y++) for(int x=0; x<56; x++) {
        int sum=0; for(int dy=0; dy<4; dy++) for(int dx=0; dx<4; dx++) sum += img_224[(y*4+dy)*224 + (x*4+dx)];
        pooled[y*56 + x] = (uint8_t)(sum / 16);
    }
    for (int y = 0; y < L1_DIM; y++) {
        for (int x = 0; x < L1_BLKS_X; x++) {
            int px = y * L1_DIM + x * 3;
            int8_t t0 = pooled[px] > 170 ? 1 : pooled[px] < 85 ? -1 : 0;
            int8_t t1 = pooled[px+1] > 170 ? 1 : pooled[px+1] < 85 ? -1 : 0;
            int8_t t2 = pooled[px+2] > 170 ? 1 : pooled[px+2] < 85 ? -1 : 0;
            sigs[y * L1_BLKS_X + x] = benc(t0, t1, t2);
        }
    }
}

/* === Mapping === */
static void init_mappings(void) {
    for (int k = 0; k < L0_TOTAL_BLKS; k++) {
        int ry = ( (k / L0_BLKS_X) * L0_KNOTS_Y) / L0_DIM;
        int rx = ( (k % L0_BLKS_X) * L0_KNOTS_X) / L0_BLKS_X;
        l0_block_to_knot[k] = ry * L0_KNOTS_X + rx;
    }
    for (int k = 0; k < L1_TOTAL_BLKS; k++) {
        int ry = ( (k / L1_BLKS_X) * L1_KNOTS_Y) / L1_DIM;
        int rx = ( (k % L1_BLKS_X) * L1_KNOTS_X) / L1_BLKS_X;
        l1_block_to_knot[k] = ry * L1_KNOTS_X + rx;
    }
}

static void build_models(void) {
    l0_map = calloc((size_t)L0_N_KNOTS * 27 * CLS_PAD, 4);
    l1_map = calloc((size_t)L1_N_KNOTS * 27 * CLS_PAD, 4);
    uint8_t img[TGT_DIM * TGT_DIM], sig0[L0_TOTAL_BLKS], sig1[L1_TOTAL_BLKS];
    printf("  Training hierarchical maps on %d images...\n", TRAIN_N);
    for (int i = 0; i < TRAIN_N; i++) {
        resize_224(raw_train + (size_t)i * 784, img); get_l0_sigs(img, sig0); get_l1_sigs(img, sig1);
        int lbl = train_labels[i];
        for (int k = 0; k < L0_TOTAL_BLKS; k++) l0_map[(size_t)l0_block_to_knot[k] * 27 * CLS_PAD + sig0[k] * CLS_PAD + lbl]++;
        for (int k = 0; k < L1_TOTAL_BLKS; k++) l1_map[(size_t)l1_block_to_knot[k] * 27 * CLS_PAD + sig1[k] * CLS_PAD + lbl]++;
    }
}

static void compute_ig(void) {
    int cc[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) cc[train_labels[i]]++;
    double hc = 0;
    for (int c = 0; c < N_CLASSES; c++) { double p = (double)cc[c] / TRAIN_N; if (p > 0) hc -= p * log2(p); }
    printf("  Computing IG weights for Level-0...\n");
    for (int k = 0; k < L0_TOTAL_BLKS; k++) {
        double hcond = 0; int vt[27] = {0}; int kid = l0_block_to_knot[k];
        for (int v = 0; v < 27; v++) for (int c = 0; c < N_CLASSES; c++) vt[v] += l0_map[(size_t)kid * 27 * 16 + v * 16 + c];
        for (int v = 0; v < 27; v++) {
            if (vt[v] == 0 || v == 13) continue;
            double pv = (double)vt[v] / TRAIN_N, hv = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                double pc = (double)l0_map[(size_t)kid * 27 * 16 + v * 16 + c] / vt[v];
                if (pc > 0) hv -= pc * log2(pc);
            }
            hcond += pv * hv;
        }
        l0_sorted[k].bid = k; l0_sorted[k].ig = (uint16_t)((hc - hcond) * 1000);
    }
    qsort(l0_sorted, L0_TOTAL_BLKS, sizeof(blk_info_t), cmp_blk);
}

static int classify(const uint8_t *img, int *p0, int *p1) {
    uint8_t sig0[L0_TOTAL_BLKS], sig1[L1_TOTAL_BLKS];
    get_l0_sigs(img, sig0); get_l1_sigs(img, sig1);
    uint32_t acc0[CLS_PAD] = {0}, acc1[CLS_PAD] = {0};
    for (int k = 0; k < L0_TOTAL_BLKS; k++) {
        if (sig0[k] == 13) continue;
        const uint32_t *row = &l0_map[(size_t)l0_block_to_knot[k] * 27 * CLS_PAD + sig0[k] * CLS_PAD];
        for (int c = 0; c < 10; c++) acc0[c] += row[c];
    }
    for (int k = 0; k < L1_TOTAL_BLKS; k++) {
        if (sig1[k] == 13) continue;
        const uint32_t *row = &l1_map[(size_t)l1_block_to_knot[k] * 27 * CLS_PAD + sig1[k] * CLS_PAD];
        for (int c = 0; c < 10; c++) acc1[c] += row[c];
    }
    int b0=0, b1=0, bt=0;
    for (int c = 0; c < 10; c++) {
        if (acc0[c] > acc0[b0]) b0 = c;
        if (acc1[c] > acc1[b1]) b1 = c;
        if (acc0[c] + acc1[c] * 8 > acc0[bt] + acc1[bt] * 8) bt = c;
    }
    *p0 = b0; *p1 = b1; return bt;
}

static int classify_adaptive(const uint8_t *img, long *lookups) {
    uint8_t sig0[L0_TOTAL_BLKS]; get_l0_sigs(img, sig0);
    uint32_t acc[CLS_PAD] = {0}; int exit_idx = L0_TOTAL_BLKS;
    for (int k = 0; k < L0_TOTAL_BLKS; k++) {
        int bid = l0_sorted[k].bid; if (sig0[bid] == 13) continue;
        const uint32_t *row = &l0_map[(size_t)l0_block_to_knot[bid] * 27 * CLS_PAD + sig0[bid] * CLS_PAD];
        for (int c = 0; c < 10; c++) acc[c] += row[c];
        if (k > 500 && k % 100 == 0) {
            int b1 = 0, b2 = 1; if (acc[b2] > acc[b1]) { b1 = 1; b2 = 0; }
            for (int c = 2; c < 10; c++) { if (acc[c] > acc[b1]) { b2 = b1; b1 = c; } else if (acc[c] > acc[b2]) b2 = c; }
            /* Lower threshold for Fashion dataset */
            if (acc[b1] > acc[b2] * 1.5 && k > 200) { exit_idx = k; break; }
        }
    }
    *lookups = exit_idx; int best = 0;
    for (int c = 1; c < 10; c++) if (acc[c] > acc[best]) best = c;
    return best;
}

int main(int argc, char** argv) {
    if(argc > 1) data_dir = argv[1];
    printf("=== SSTT 224x224 Hierarchical & Adaptive Scaling (%s) ===\n\n", data_dir);
    load_data(); init_mappings(); build_models(); compute_ig();
    uint8_t img[TGT_DIM * TGT_DIM]; int ok_hier=0, ok_adap=0; long total_lk=0;
    printf("  Classifying %d images...\n", TEST_N);
    double t0 = now_sec();
    for (int i = 0; i < TEST_N; i++) {
        resize_224(raw_test + (size_t)i * 784, img);
        int p0, p1; ok_hier += (classify(img, &p0, &p1) == test_labels[i]);
        long lk; ok_adap += (classify_adaptive(img, &lk) == test_labels[i]); total_lk += lk;
    }
    double t1 = now_sec();
    printf("\nResults:\n");
    printf("  Hierarchical Accuracy: %.2f%%\n", 100.0 * ok_hier / TEST_N);
    printf("  Adaptive Accuracy:     %.2f%%  (Avg Lookups: %ld, %.1f%%)\n", 100.0 * ok_adap / TEST_N, total_lk / TEST_N, 100.0 * total_lk / TEST_N / L0_TOTAL_BLKS);
    printf("  Latency:               %.2f ms/query\n", (t1 - t0) * 1000.0 / (TEST_N * 2));
    return 0;
}
