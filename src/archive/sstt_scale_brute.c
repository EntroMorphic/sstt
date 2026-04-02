/*
 * sstt_scale_brute.c — Brute kNN Baseline at 224x224 Scale
 *
 * Establishes the hard accuracy ceiling for the 224x224 scaling framework.
 * Operates on the same 56x56 pooled, 1008-position block feature space used
 * by sstt_scale_224.c and sstt_scale_hierarchical.c.
 *
 * For each test image: compute its 1008 block signatures, then count matching
 * positions against all 60K training images. The training image with the most
 * matching positions is the 1-NN; k=3 majority vote gives k=3 NN.
 *
 * Also reports full hot map accuracy in the same run for direct comparison.
 *
 * Usage:
 *   ./build/sstt_scale_brute              # MNIST
 *   ./build/sstt_scale_brute data-fashion/ # Fashion-MNIST
 *
 * Contribution 61 — PRD experiment C.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAIN_N    60000
#define TEST_N     10000
#define N_CLASSES  10
#define CLS_PAD    16

/* 56x56 pooled representation: 18 blocks/row × 56 rows = 1008 positions */
#define P_DIM      56
#define BLKS_X     (P_DIM / 3)
#define BLKS_Y     P_DIM
#define TOTAL_BLKS (BLKS_X * BLKS_Y)   /* 1008 */

/* kNN top-K for majority vote */
#define KNN_K      3

static const char *data_dir = "data/";
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint8_t *raw_train, *raw_test, *train_labels, *test_labels;

/* Pre-computed block signatures for all training images: TRAIN_N × TOTAL_BLKS */
static uint8_t *train_sigs;    /* 60000 × 1008 = ~60 MB */

/* Hot map for same-run comparison */
static uint32_t hot_map[TOTAL_BLKS * 27 * CLS_PAD];

/* === Data loading === */
static uint8_t *load_idx(const char *path, uint32_t *cnt, uint32_t *ro, uint32_t *co) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERR: cannot open %s\n", path); exit(1); }
    uint32_t m, n;
    if (fread(&m, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) { fclose(f); exit(1); }
    m = __builtin_bswap32(m); n = __builtin_bswap32(n); *cnt = n;
    size_t s = 1;
    if ((m & 0xFF) >= 3) {
        uint32_t r, c;
        if (fread(&r, 4, 1, f) != 1 || fread(&c, 4, 1, f) != 1) { fclose(f); exit(1); }
        r = __builtin_bswap32(r); c = __builtin_bswap32(c);
        if (ro) *ro = r;
        if (co) *co = c;
        s = (size_t)r * c;
    } else { if (ro) *ro = 0; if (co) *co = 0; }
    size_t total = (size_t)n * s;
    uint8_t *d = malloc(total);
    if (!d || fread(d, 1, total, f) != total) { fclose(f); exit(1); }
    fclose(f); return d;
}

static void load_data(void) {
    uint32_t n, r, c; char p[256];
    snprintf(p, sizeof(p), "%strain-images-idx3-ubyte", data_dir);
    raw_train = load_idx(p, &n, &r, &c);
    snprintf(p, sizeof(p), "%strain-labels-idx1-ubyte", data_dir);
    train_labels = load_idx(p, &n, NULL, NULL);
    snprintf(p, sizeof(p), "%st10k-images-idx3-ubyte", data_dir);
    raw_test = load_idx(p, &n, &r, &c);
    snprintf(p, sizeof(p), "%st10k-labels-idx1-ubyte", data_dir);
    test_labels = load_idx(p, &n, NULL, NULL);
}

/* === Feature extraction === */
static void get_pooled_56(const uint8_t *src_28, uint8_t *dst_56) {
    for (int y = 0; y < 56; y++)
        for (int x = 0; x < 56; x++)
            dst_56[y * 56 + x] = src_28[(y / 2) * 28 + (x / 2)];
}

static inline uint8_t benc(int8_t a, int8_t b, int8_t c) {
    return (uint8_t)((a + 1) * 9 + (b + 1) * 3 + (c + 1));
}

static void get_block_sigs(const uint8_t *img_56, uint8_t *sigs) {
    for (int y = 0; y < 56; y++) {
        for (int x = 0; x < 18; x++) {
            int px = y * 56 + x * 3;
            int8_t t0 = img_56[px]   > 170 ? 1 : img_56[px]   < 85 ? -1 : 0;
            int8_t t1 = img_56[px+1] > 170 ? 1 : img_56[px+1] < 85 ? -1 : 0;
            int8_t t2 = img_56[px+2] > 170 ? 1 : img_56[px+2] < 85 ? -1 : 0;
            sigs[y * 18 + x] = benc(t0, t1, t2);
        }
    }
}

/* === Training: pre-compute signatures and build hot map === */
static void build_training_db(void) {
    train_sigs = malloc((size_t)TRAIN_N * TOTAL_BLKS);
    if (!train_sigs) { fprintf(stderr, "ERR: cannot allocate %zu MB for training signatures\n",
        (size_t)TRAIN_N * TOTAL_BLKS / 1000000); exit(1); }

    memset(hot_map, 0, sizeof(hot_map));
    uint8_t img_56[56 * 56];

    printf("  Pre-computing training signatures (%zu MB)...\n",
           (size_t)TRAIN_N * TOTAL_BLKS / 1000000);

    for (int i = 0; i < TRAIN_N; i++) {
        get_pooled_56(raw_train + (size_t)i * 784, img_56);
        uint8_t *sigs = train_sigs + (size_t)i * TOTAL_BLKS;
        get_block_sigs(img_56, sigs);
        int lbl = train_labels[i];
        for (int k = 0; k < TOTAL_BLKS; k++) {
            if (sigs[k] != 13)
                hot_map[(size_t)k * 27 * CLS_PAD + sigs[k] * CLS_PAD + lbl]++;
        }
    }
}

/* === Hot map classify (for comparison) === */
static int classify_hotmap(const uint8_t *sigs) {
    uint32_t acc[CLS_PAD] = {0};
    for (int k = 0; k < TOTAL_BLKS; k++) {
        if (sigs[k] == 13) continue;
        const uint32_t *row = &hot_map[(size_t)k * 27 * CLS_PAD + sigs[k] * CLS_PAD];
        for (int c = 0; c < 10; c++) acc[c] += row[c];
    }
    int best = 0;
    for (int c = 1; c < 10; c++) if (acc[c] > acc[best]) best = c;
    return best;
}

/* === Brute kNN classify ===
 * Computes Hamming similarity (matching positions) between query and every
 * training image. Returns k=1 and k=3 predictions.
 */
static void classify_brute(const uint8_t *q_sigs, int *pred1, int *pred3) {
    /* Top-KNN_K by matching positions */
    int   top_score[KNN_K]; int top_label[KNN_K];
    for (int k = 0; k < KNN_K; k++) { top_score[k] = -1; top_label[k] = 0; }

    for (int i = 0; i < TRAIN_N; i++) {
        const uint8_t *t = train_sigs + (size_t)i * TOTAL_BLKS;
        int score = 0;
        for (int k = 0; k < TOTAL_BLKS; k++)
            score += (q_sigs[k] == t[k] && q_sigs[k] != 13);

        /* Insert into top-K (sorted descending) */
        if (score > top_score[KNN_K - 1]) {
            top_score[KNN_K - 1] = score;
            top_label[KNN_K - 1] = train_labels[i];
            /* Bubble up */
            for (int j = KNN_K - 2; j >= 0; j--) {
                if (top_score[j + 1] > top_score[j]) {
                    int ts = top_score[j]; top_score[j] = top_score[j+1]; top_score[j+1] = ts;
                    int tl = top_label[j]; top_label[j] = top_label[j+1]; top_label[j+1] = tl;
                } else break;
            }
        }
    }

    *pred1 = top_label[0];

    /* k=3 majority vote */
    int votes[N_CLASSES] = {0};
    for (int k = 0; k < KNN_K; k++) votes[top_label[k]]++;
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (votes[c] > votes[best]) best = c;
    *pred3 = best;
}

int main(int argc, char **argv) {
    if (argc > 1) data_dir = argv[1];

    printf("=== SSTT 224x224 Brute kNN Baseline ===\n");
    printf("Data: %s  Train: %d  Test: %d\n\n", data_dir, TRAIN_N, TEST_N);

    load_data();
    build_training_db();

    int ok_brute1 = 0, ok_brute3 = 0, ok_hot = 0;
    int cls_brute3_ok[N_CLASSES] = {0}, cls_hot_ok[N_CLASSES] = {0};
    int cls_n[N_CLASSES] = {0};

    uint8_t img_56[56 * 56], q_sigs[TOTAL_BLKS];

    printf("  Running brute kNN on %d test images (this takes ~60s)...\n", TEST_N);
    double t0 = now_sec();

    for (int i = 0; i < TEST_N; i++) {
        int lbl = test_labels[i]; cls_n[lbl]++;
        get_pooled_56(raw_test + (size_t)i * 784, img_56);
        get_block_sigs(img_56, q_sigs);

        int p1, p3;
        classify_brute(q_sigs, &p1, &p3);
        if (p1 == lbl) ok_brute1++;
        if (p3 == lbl) { ok_brute3++; cls_brute3_ok[lbl]++; }

        int ph = classify_hotmap(q_sigs);
        if (ph == lbl) { ok_hot++; cls_hot_ok[lbl]++; }

        if ((i + 1) % 1000 == 0) {
            double elapsed = now_sec() - t0;
            printf("  ... %d/%d  (%.1fs elapsed, ~%.0fs remaining)\n",
                   i + 1, TEST_N, elapsed, elapsed / (i + 1) * (TEST_N - i - 1));
        }
    }

    double t1 = now_sec();
    double brute_time = t1 - t0;

    printf("\n=== Results ===\n");
    printf("  Brute 1-NN accuracy:      %.2f%%\n", 100.0 * ok_brute1 / TEST_N);
    printf("  Brute k=3 NN accuracy:    %.2f%%\n", 100.0 * ok_brute3 / TEST_N);
    printf("  Full hot map accuracy:    %.2f%%\n", 100.0 * ok_hot    / TEST_N);
    printf("\n");
    printf("  Retention (hot/brute k3): %.1f%%\n",
           100.0 * ok_hot / (ok_brute3 > 0 ? ok_brute3 : 1));
    printf("  Brute kNN runtime:        %.1fs total  (~%.1f ms/query)\n",
           brute_time, brute_time * 1000.0 / TEST_N);
    printf("  Training DB size:         %zu MB\n",
           (size_t)TRAIN_N * TOTAL_BLKS / 1000000);

    printf("\nPer-class (brute k=3 vs hot map):\n");
    printf("  %-10s  %8s  %8s\n", "Class", "Brute k3", "Hot Map");
    for (int c = 0; c < N_CLASSES; c++) {
        printf("  Class %-4d  %7.1f%%  %7.1f%%\n", c,
               cls_n[c] > 0 ? 100.0 * cls_brute3_ok[c] / cls_n[c] : 0.0,
               cls_n[c] > 0 ? 100.0 * cls_hot_ok[c]    / cls_n[c] : 0.0);
    }

    return 0;
}
