/*
 * sstt_cifar10_gauss.c — Gauss Map: Classify on Shape Geometry
 *
 * Map each pixel's gradient vector to a point on the unit sphere.
 * The distribution of points = the image's edge geometry fingerprint.
 * Background (flat) → north pole → suppressed. Only edges contribute.
 *
 * Per-channel RGB Gauss maps: 3 spheres capturing color-edge geometry.
 * Position-invariant. Brightness-invariant. Shape-sensitive.
 *
 * Build: gcc -O3 -mavx2 -march=native -o sstt_cifar10_gauss src/sstt_cifar10_gauss.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N 50000
#define TEST_N  10000
#define N_CLASSES 10
#define CLS_PAD 16
#define SPATIAL_W 32
#define SPATIAL_H 32
#define SP_PIXELS 1024

/* Gauss map bins: hgrad direction (3) × vgrad direction (3) × magnitude bucket (5) = 45 per channel */
#define GM_DIR 3    /* {negative, zero, positive} gradient direction */
#define GM_MAG 5    /* magnitude buckets: 0, 1, 2, 3, 4+ */
#define GM_BINS_PER_CH (GM_DIR * GM_DIR * GM_MAG)  /* 45 */
#define GM_CHANNELS 4  /* grayscale + R + G + B */
#define GM_TOTAL (GM_BINS_PER_CH * GM_CHANNELS)  /* 180 */

/* Also: grid-decomposed Gauss map (spatial structure) */
#define GRID_R 4
#define GRID_C 4
#define GRID_N (GRID_R * GRID_C)  /* 16 regions */
#define GM_GRID_TOTAL (GM_BINS_PER_CH * GRID_N)  /* 45 × 16 = 720 per channel */

static const char *data_dir="data-cifar10/";
static const char *cn[]={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint8_t *raw_r_tr,*raw_r_te,*raw_g_tr,*raw_g_te,*raw_b_tr,*raw_b_te,*raw_gray_tr,*raw_gray_te;
static uint8_t *train_labels,*test_labels;

/* ================================================================
 *  Gauss map computation
 * ================================================================ */

/* Compute Gauss map histogram for one channel image (32×32 uint8) */
static void gauss_map(const uint8_t *img, int16_t *hist) {
    memset(hist, 0, GM_BINS_PER_CH * sizeof(int16_t));
    for (int y = 0; y < SPATIAL_H; y++) {
        for (int x = 0; x < SPATIAL_W; x++) {
            /* Gradient */
            int gx = (x < SPATIAL_W - 1) ? (int)img[y * SPATIAL_W + x + 1] - (int)img[y * SPATIAL_W + x] : 0;
            int gy = (y < SPATIAL_H - 1) ? (int)img[(y + 1) * SPATIAL_W + x] - (int)img[y * SPATIAL_W + x] : 0;

            /* Direction: ternary {-1, 0, +1} */
            int dx = (gx > 10) ? 2 : (gx < -10) ? 0 : 1;  /* 0=neg, 1=zero, 2=pos */
            int dy = (gy > 10) ? 2 : (gy < -10) ? 0 : 1;

            /* Magnitude bucket */
            int mag = abs(gx) + abs(gy);
            int mb;
            if (mag < 5) mb = 0;       /* flat → essentially background */
            else if (mag < 20) mb = 1;  /* weak edge */
            else if (mag < 50) mb = 2;  /* moderate edge */
            else if (mag < 100) mb = 3; /* strong edge */
            else mb = 4;                /* very strong edge */

            int bin = dx * GM_DIR * GM_MAG + dy * GM_MAG + mb;
            hist[bin]++;
        }
    }
}

/* Grid-decomposed Gauss map: compute per-region */
static void gauss_map_grid(const uint8_t *img, int16_t *hist) {
    memset(hist, 0, GM_GRID_TOTAL * sizeof(int16_t));
    for (int y = 0; y < SPATIAL_H; y++) {
        for (int x = 0; x < SPATIAL_W; x++) {
            int gx = (x < SPATIAL_W - 1) ? (int)img[y * SPATIAL_W + x + 1] - (int)img[y * SPATIAL_W + x] : 0;
            int gy = (y < SPATIAL_H - 1) ? (int)img[(y + 1) * SPATIAL_W + x] - (int)img[y * SPATIAL_W + x] : 0;
            int dx = (gx > 10) ? 2 : (gx < -10) ? 0 : 1;
            int dy = (gy > 10) ? 2 : (gy < -10) ? 0 : 1;
            int mag = abs(gx) + abs(gy);
            int mb = (mag < 5) ? 0 : (mag < 20) ? 1 : (mag < 50) ? 2 : (mag < 100) ? 3 : 4;
            int bin = dx * GM_DIR * GM_MAG + dy * GM_MAG + mb;
            int ry = y * GRID_R / SPATIAL_H, rx = x * GRID_C / SPATIAL_W;
            if (ry >= GRID_R) ry = GRID_R - 1;
            if (rx >= GRID_C) rx = GRID_C - 1;
            int region = ry * GRID_C + rx;
            hist[region * GM_BINS_PER_CH + bin]++;
        }
    }
}

/* L1 distance between two histograms */
static int32_t hist_l1(const int16_t *a, const int16_t *b, int len) {
    int32_t d = 0;
    for (int i = 0; i < len; i++) d += abs(a[i] - b[i]);
    return d;
}

/* Chi-squared similarity (higher = more similar) */
static double hist_chi2_sim(const int16_t *a, const int16_t *b, int len) {
    double sim = 0;
    for (int i = 0; i < len; i++) {
        double sum = a[i] + b[i];
        if (sum > 0) {
            double diff = a[i] - b[i];
            sim -= diff * diff / sum;
        }
    }
    return sim;
}

static void load_data(void) {
    raw_r_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_r_te=malloc((size_t)TEST_N*SP_PIXELS);
    raw_g_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_g_te=malloc((size_t)TEST_N*SP_PIXELS);
    raw_b_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_b_te=malloc((size_t)TEST_N*SP_PIXELS);
    raw_gray_tr=malloc((size_t)TRAIN_N*SP_PIXELS);raw_gray_te=malloc((size_t)TEST_N*SP_PIXELS);
    train_labels=malloc(TRAIN_N);test_labels=malloc(TEST_N);
    char p[512];uint8_t rec[3073];
    for(int b2=1;b2<=5;b2++){snprintf(p,sizeof(p),"%sdata_batch_%d.bin",data_dir,b2);
        FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
            int idx=(b2-1)*10000+i;train_labels[idx]=rec[0];
            const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
            memcpy(raw_r_tr+(size_t)idx*SP_PIXELS,r,SP_PIXELS);
            memcpy(raw_g_tr+(size_t)idx*SP_PIXELS,g,SP_PIXELS);
            memcpy(raw_b_tr+(size_t)idx*SP_PIXELS,b3,SP_PIXELS);
            uint8_t*gd=raw_gray_tr+(size_t)idx*SP_PIXELS;
            for(int p2=0;p2<SP_PIXELS;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}
    snprintf(p,sizeof(p),"%stest_batch.bin",data_dir);
    FILE*f=fopen(p,"rb");for(int i=0;i<10000;i++){if(fread(rec,1,3073,f)!=3073){fclose(f);exit(1);}
        test_labels[i]=rec[0];const uint8_t*r=rec+1,*g=rec+1+1024,*b3=rec+1+2048;
        memcpy(raw_r_te+(size_t)i*SP_PIXELS,r,SP_PIXELS);
        memcpy(raw_g_te+(size_t)i*SP_PIXELS,g,SP_PIXELS);
        memcpy(raw_b_te+(size_t)i*SP_PIXELS,b3,SP_PIXELS);
        uint8_t*gd=raw_gray_te+(size_t)i*SP_PIXELS;
        for(int p2=0;p2<SP_PIXELS;p2++)gd[p2]=(uint8_t)((77*(int)r[p2]+150*(int)g[p2]+29*(int)b3[p2])>>8);}fclose(f);}

int main(int argc,char**argv){
    double t0=now_sec();
    if(argc>1){data_dir=argv[1];size_t l=strlen(data_dir);
        if(l&&data_dir[l-1]!='/'){char*b2=malloc(l+2);memcpy(b2,data_dir,l);b2[l]='/';b2[l+1]=0;data_dir=b2;}}
    printf("=== SSTT CIFAR-10: Gauss Map — Classify on Shape Geometry ===\n\n");
    load_data();

    /* ================================================================
     * Compute Gauss maps for all images
     * ================================================================ */
    printf("Computing Gauss maps...\n");

    /* Global Gauss maps: 4 channels × 45 bins = 180 per image */
    int16_t *gm_tr = malloc((size_t)TRAIN_N * GM_TOTAL * sizeof(int16_t));
    int16_t *gm_te = malloc((size_t)TEST_N * GM_TOTAL * sizeof(int16_t));

    const uint8_t *channels_tr[] = {raw_gray_tr, raw_r_tr, raw_g_tr, raw_b_tr};
    const uint8_t *channels_te[] = {raw_gray_te, raw_r_te, raw_g_te, raw_b_te};

    for (int i = 0; i < TRAIN_N; i++)
        for (int ch = 0; ch < GM_CHANNELS; ch++)
            gauss_map(channels_tr[ch] + (size_t)i * SP_PIXELS,
                     gm_tr + (size_t)i * GM_TOTAL + ch * GM_BINS_PER_CH);
    for (int i = 0; i < TEST_N; i++)
        for (int ch = 0; ch < GM_CHANNELS; ch++)
            gauss_map(channels_te[ch] + (size_t)i * SP_PIXELS,
                     gm_te + (size_t)i * GM_TOTAL + ch * GM_BINS_PER_CH);

    /* Grid Gauss maps: grayscale only, 4×4 grid × 45 bins = 720 per image */
    int16_t *ggm_tr = malloc((size_t)TRAIN_N * GM_GRID_TOTAL * sizeof(int16_t));
    int16_t *ggm_te = malloc((size_t)TEST_N * GM_GRID_TOTAL * sizeof(int16_t));
    for (int i = 0; i < TRAIN_N; i++)
        gauss_map_grid(raw_gray_tr + (size_t)i * SP_PIXELS, ggm_tr + (size_t)i * GM_GRID_TOTAL);
    for (int i = 0; i < TEST_N; i++)
        gauss_map_grid(raw_gray_te + (size_t)i * SP_PIXELS, ggm_te + (size_t)i * GM_GRID_TOTAL);

    /* Per-class mean Gauss maps */
    double class_mean[N_CLASSES][GM_TOTAL];
    memset(class_mean, 0, sizeof(class_mean));
    int class_count[N_CLASSES] = {0};
    for (int i = 0; i < TRAIN_N; i++) {
        int l = train_labels[i]; class_count[l]++;
        const int16_t *gm = gm_tr + (size_t)i * GM_TOTAL;
        for (int b = 0; b < GM_TOTAL; b++) class_mean[l][b] += gm[b];
    }
    for (int c = 0; c < N_CLASSES; c++)
        for (int b = 0; b < GM_TOTAL; b++) class_mean[c][b] /= class_count[c];

    printf("  Gauss maps computed (%.1f sec)\n\n", now_sec() - t0);

    /* ================================================================
     * Show inter-class Gauss map distances
     * ================================================================ */
    printf("=== INTER-CLASS GAUSS MAP DISTANCES ===\n\n");
    printf("  (L1 between class mean Gauss maps — lower = more similar)\n\n");
    int16_t class_mean_i16[N_CLASSES][GM_TOTAL];
    for (int c = 0; c < N_CLASSES; c++)
        for (int b = 0; b < GM_TOTAL; b++) class_mean_i16[c][b] = (int16_t)(class_mean[c][b] + 0.5);

    printf("  %12s", "");
    for (int c = 0; c < N_CLASSES; c++) printf(" %6.4s", cn[c]);
    printf("\n");
    for (int a = 0; a < N_CLASSES; a++) {
        printf("  %-10s", cn[a]);
        for (int b = 0; b < N_CLASSES; b++) {
            int32_t d = hist_l1(class_mean_i16[a], class_mean_i16[b], GM_TOTAL);
            printf(" %6d", d);
        }
        printf("\n");
    }

    /* ================================================================
     * Test 1: kNN on Gauss map histograms (global, 4-channel)
     * ================================================================ */
    printf("\n=== TEST 1: kNN on Global 4-Channel Gauss Map (L1, k=1) ===\n\n");
    {
        int correct = 0; int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for (int i = 0; i < TEST_N; i++) {
            pt[test_labels[i]]++;
            const int16_t *qi = gm_te + (size_t)i * GM_TOTAL;
            /* Brute kNN: find nearest training image by L1 */
            int32_t best_d = INT32_MAX; int best_lbl = 0;
            for (int j = 0; j < TRAIN_N; j++) {
                int32_t d = hist_l1(qi, gm_tr + (size_t)j * GM_TOTAL, GM_TOTAL);
                if (d < best_d) { best_d = d; best_lbl = train_labels[j]; }
            }
            if (best_lbl == test_labels[i]) { correct++; pc[test_labels[i]]++; }
            if ((i + 1) % 1000 == 0) fprintf(stderr, "  kNN-GM: %d/%d\r", i + 1, TEST_N);
        }
        fprintf(stderr, "\n");
        printf("  Gauss map kNN (L1, k=1): %.2f%%\n", 100.0 * correct / TEST_N);
        for (int c = 0; c < N_CLASSES; c++)
            printf("    %d %-10s %.1f%%\n", c, cn[c], 100.0 * pc[c] / pt[c]);
    }

    /* ================================================================
     * Test 2: kNN on Grid Gauss Map (spatial structure)
     * ================================================================ */
    printf("\n=== TEST 2: kNN on Grid Gauss Map (4x4 spatial, L1, k=1) ===\n\n");
    {
        int correct = 0; int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for (int i = 0; i < TEST_N; i++) {
            pt[test_labels[i]]++;
            const int16_t *qi = ggm_te + (size_t)i * GM_GRID_TOTAL;
            int32_t best_d = INT32_MAX; int best_lbl = 0;
            for (int j = 0; j < TRAIN_N; j++) {
                int32_t d = hist_l1(qi, ggm_tr + (size_t)j * GM_GRID_TOTAL, GM_GRID_TOTAL);
                if (d < best_d) { best_d = d; best_lbl = train_labels[j]; }
            }
            if (best_lbl == test_labels[i]) { correct++; pc[test_labels[i]]++; }
            if ((i + 1) % 1000 == 0) fprintf(stderr, "  kNN-GGM: %d/%d\r", i + 1, TEST_N);
        }
        fprintf(stderr, "\n");
        printf("  Grid Gauss map kNN (L1, k=1): %.2f%%\n", 100.0 * correct / TEST_N);
        for (int c = 0; c < N_CLASSES; c++)
            printf("    %d %-10s %.1f%%\n", c, cn[c], 100.0 * pc[c] / pt[c]);
    }

    /* ================================================================
     * Test 3: Nearest class mean (centroid classifier)
     * ================================================================ */
    printf("\n=== TEST 3: Nearest Class Mean (Gauss map centroid) ===\n\n");
    {
        int correct = 0; int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for (int i = 0; i < TEST_N; i++) {
            pt[test_labels[i]]++;
            const int16_t *qi = gm_te + (size_t)i * GM_TOTAL;
            int32_t best_d = INT32_MAX; int best_c = 0;
            for (int c = 0; c < N_CLASSES; c++) {
                int32_t d = hist_l1(qi, class_mean_i16[c], GM_TOTAL);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            if (best_c == test_labels[i]) { correct++; pc[test_labels[i]]++; }
        }
        printf("  Nearest class mean: %.2f%%\n", 100.0 * correct / TEST_N);
        for (int c = 0; c < N_CLASSES; c++)
            printf("    %d %-10s %.1f%%\n", c, cn[c], 100.0 * pc[c] / pt[c]);
    }

    /* ================================================================
     * Test 4: Combined Global + Grid Gauss map kNN
     * ================================================================ */
    printf("\n=== TEST 4: Combined Global + Grid Gauss Map kNN ===\n\n");
    {
        int correct = 0; int pc[N_CLASSES]={0},pt[N_CLASSES]={0};
        for (int i = 0; i < TEST_N; i++) {
            pt[test_labels[i]]++;
            const int16_t *qi_g = gm_te + (size_t)i * GM_TOTAL;
            const int16_t *qi_gg = ggm_te + (size_t)i * GM_GRID_TOTAL;
            int32_t best_d = INT32_MAX; int best_lbl = 0;
            for (int j = 0; j < TRAIN_N; j++) {
                int32_t d = hist_l1(qi_g, gm_tr + (size_t)j * GM_TOTAL, GM_TOTAL)
                          + hist_l1(qi_gg, ggm_tr + (size_t)j * GM_GRID_TOTAL, GM_GRID_TOTAL) / 4;
                if (d < best_d) { best_d = d; best_lbl = train_labels[j]; }
            }
            if (best_lbl == test_labels[i]) { correct++; pc[test_labels[i]]++; }
            if ((i + 1) % 1000 == 0) fprintf(stderr, "  kNN-combo: %d/%d\r", i + 1, TEST_N);
        }
        fprintf(stderr, "\n");
        printf("  Combined kNN: %.2f%%\n", 100.0 * correct / TEST_N);
        for (int c = 0; c < N_CLASSES; c++)
            printf("    %d %-10s %.1f%%\n", c, cn[c], 100.0 * pc[c] / pt[c]);
    }

    /* ================================================================
     * Test 5: Binary (machine vs animal) on Gauss map
     * ================================================================ */
    printf("\n=== TEST 5: Binary Machine/Animal on Gauss Map ===\n\n");
    {
        int is_machine[10] = {1,1,0,0,0,0,0,0,1,1};
        /* Class mean distance: machine centroid vs animal centroid */
        double machine_mean[GM_TOTAL] = {0}, animal_mean[GM_TOTAL] = {0};
        int mc = 0, ac = 0;
        for (int i = 0; i < TRAIN_N; i++) {
            const int16_t *gm = gm_tr + (size_t)i * GM_TOTAL;
            if (is_machine[train_labels[i]]) { for (int b = 0; b < GM_TOTAL; b++) machine_mean[b] += gm[b]; mc++; }
            else { for (int b = 0; b < GM_TOTAL; b++) animal_mean[b] += gm[b]; ac++; }
        }
        for (int b = 0; b < GM_TOTAL; b++) { machine_mean[b] /= mc; animal_mean[b] /= ac; }
        int16_t mm16[GM_TOTAL], am16[GM_TOTAL];
        for (int b = 0; b < GM_TOTAL; b++) { mm16[b] = (int16_t)(machine_mean[b] + 0.5); am16[b] = (int16_t)(animal_mean[b] + 0.5); }

        int correct = 0;
        int conf[2][2] = {{0}};
        for (int i = 0; i < TEST_N; i++) {
            const int16_t *qi = gm_te + (size_t)i * GM_TOTAL;
            int32_t dm = hist_l1(qi, mm16, GM_TOTAL);
            int32_t da = hist_l1(qi, am16, GM_TOTAL);
            int pred = (dm < da) ? 0 : 1; /* 0=machine, 1=animal */
            int truth = is_machine[test_labels[i]] ? 0 : 1;
            conf[truth][pred]++;
            if (pred == truth) correct++;
        }
        printf("  Gauss map binary: %.2f%%\n", 100.0 * correct / TEST_N);
        printf("  Confusion:  pred_machine  pred_animal\n");
        printf("    machine:     %5d        %5d\n", conf[0][0], conf[0][1]);
        printf("    animal:      %5d        %5d\n", conf[1][0], conf[1][1]);
    }

    printf("\n  Reference: block-based binary = 81.04%%\n");
    printf("  Reference: stereo + MT4 stack = 44.48%%\n");

    printf("\nTotal: %.1f sec\n", now_sec() - t0);
    free(gm_tr);free(gm_te);free(ggm_tr);free(ggm_te);
    return 0;
}
