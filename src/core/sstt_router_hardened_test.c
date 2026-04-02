/*
 * sstt_router_hardened_test.c — Measured Proof of Shift Invariance
 *
 * This script runs the hardened router logic against a 
 * shifted version of the MNIST test set.
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sstt_fused.h"

#define TRAIN_N    60000
#define TEST_N     1000
#define N_CLASSES  10
#define PIXELS     784
#define PADDED     800
#define N_BLOCKS   252
#define SIG_PAD    256
#define BYTE_VALS  256
#define TOP_K      200
#define IG_SCALE   16

static const char *data_dir = "data/";
static uint8_t *raw_train, *raw_test, *train_labels, *test_labels;
static int8_t *tern_train;
static uint16_t ig_w[N_BLOCKS];
static uint32_t idx_off[N_BLOCKS][BYTE_VALS];
static uint16_t idx_sz[N_BLOCKS][BYTE_VALS];
static uint32_t *idx_pool;

/* Minimal boilerplate for the test */
static uint8_t *load_idx(const char*path,uint32_t*n){FILE*f=fopen(path,"rb");uint32_t m,cnt;fread(&m,4,1,f);fread(&cnt,4,1,f);cnt=__builtin_bswap32(cnt);if(n)*n=cnt;if((__builtin_bswap32(m)&0xFF)>=3){uint32_t r,c;fread(&r,4,1,f);fread(&c,4,1,f);cnt*=__builtin_bswap32(r)*__builtin_bswap32(c);}uint8_t*d=malloc(cnt);fread(d,1,cnt,f);fclose(f);return d;}
static inline uint8_t benc(int8_t a,int8_t b,int8_t c){return(uint8_t)((a+1)*9+(b+1)*3+(c+1));}

static void build_idx(void) {
    uint8_t *sigs = malloc(TRAIN_N * SIG_PAD);
    for(int i=0; i<TRAIN_N; i++) {
        for(int y=0; y<28; y++) for(int s=0; s<9; s++) {
            int b = i*PIXELS + y*28 + s*3;
            int8_t t0 = (raw_train[b]>170)?1:(raw_train[b]<85)?-1:0;
            int8_t t1 = (raw_train[b+1]>170)?1:(raw_train[b+1]<85)?-1:0;
            int8_t t2 = (raw_train[b+2]>170)?1:(raw_train[b+2]<85)?-1:0;
            sigs[i*SIG_PAD + y*9 + s] = benc(t0, t1, t2);
        }
    }
    memset(idx_sz, 0, sizeof(idx_sz));
    for(int i=0; i<TRAIN_N; i++) for(int k=0; k<252; k++) if(sigs[i*SIG_PAD+k]!=13) idx_sz[k][sigs[i*SIG_PAD+k]]++;
    uint32_t tot=0; for(int k=0; k<252; k++) for(int v=0; v<256; v++) { idx_off[k][v]=tot; tot+=idx_sz[k][v]; }
    idx_pool = malloc(tot*4); uint32_t *wp = malloc(252*256*4); memcpy(wp, idx_off, 252*256*4);
    for(int i=0; i<TRAIN_N; i++) for(int k=0; k<252; k++) if(sigs[i*SIG_PAD+k]!=13) idx_pool[wp[k*256+sigs[i*SIG_PAD+k]]++] = (uint32_t)i;
    for(int k=0; k<252; k++) ig_w[k]=16;
    free(sigs); free(wp);
}

static int classify_hardened(const uint8_t *img) {
    uint32_t *vbuf = calloc(TRAIN_N, 4);
    int8_t t[784]; for(int j=0; j<784; j++) t[j] = (img[j]>170)?1:(img[j]<85)?-1:0;
    
    /* 1. Primary Probe (Centered) */
    for(int y=0; y<28; y++) for(int s=0; s<9; s++) {
        uint8_t bv = benc(t[y*28+s*3], t[y*28+s*3+1], t[y*28+s*3+2]);
        if(bv == 13) continue;
        uint32_t off = idx_off[y*9+s][bv];
        for(uint16_t j=0; j<idx_sz[y*9+s][bv]; j++) vbuf[idx_pool[off+j]] += 16;
    }

    uint32_t mx = 0; for(int j=0; j<60000; j++) if(vbuf[j] > mx) mx = vbuf[j];

    /* 2. Jitter Rescue (If confidence is low) */
    if (mx < 100) {
        int dxs[] = {-1, 1, 0, 0, -1, 1, -1, 1};
        int dys[] = {0, 0, -1, 1, -1, -1, 1, 1};
        for (int off = 0; off < 8; off++) {
            for (int y = 1; y < 27; y++) {
                for (int s = 0; s < 9; s++) {
                    int b = (y + dys[off]) * 28 + s * 3 + dxs[off];
                    if (b < 0 || b + 2 >= 784) continue;
                    uint8_t bv = benc(t[b], t[b+1], t[b+2]);
                    if (bv == 13) continue;
                    uint32_t off2 = idx_off[y * 9 + s][bv];
                    for (uint16_t j = 0; j < idx_sz[y*9+s][bv]; j++) vbuf[idx_pool[off2+j]] += 8;
                }
            }
        }
    }

    int best_id = -1; uint32_t max_v = 0;
    for(int j=0; j<60000; j++) if(vbuf[j] > max_v) { max_v = vbuf[j]; best_id = j; }
    free(vbuf);
    return (best_id == -1) ? -1 : train_labels[best_id];
}

static void apply_shift(const uint8_t *src, uint8_t *dst, int dx, int dy) {
    memset(dst, 0, 784);
    for(int y=0; y<28; y++) {
        int ny = y + dy; if(ny < 0 || ny >= 28) continue;
        for(int x=0; x<28; x++) {
            int nx = x + dx; if(nx < 0 || nx >= 28) continue;
            dst[ny*28 + nx] = src[y*28 + x];
        }
    }
}

int main(void) {
    raw_train = load_idx("data/train-images-idx3-ubyte", NULL);
    raw_test = load_idx("data/t10k-images-idx3-ubyte", NULL);
    train_labels = load_idx("data/train-labels-idx1-ubyte", NULL);
    test_labels = load_idx("data/t10k-labels-idx1-ubyte", NULL);
    build_idx();

    printf("=== MEASURED PROOF OF HARDENING (1,000 images) ===\n");

    int clean_corr = 0;
    for(int i=0; i<TEST_N; i++) if(classify_hardened(raw_test+i*784) == test_labels[i]) clean_corr++;
    printf("  CLEAN ACCURACY:  %.2f%%\n", 100.0*clean_corr/TEST_N);

    int shift1_corr = 0;
    uint8_t shifted[784];
    for(int i=0; i<TEST_N; i++) {
        apply_shift(raw_test+i*784, shifted, 1, 1);
        if(classify_hardened(shifted) == test_labels[i]) shift1_corr++;
    }
    printf("  SHIFT (+1, +1):  %.2f%%\n", 100.0*shift1_corr/TEST_N);

    int shift2_corr = 0;
    for(int i=0; i<TEST_N; i++) {
        apply_shift(raw_test+i*784, shifted, 2, 2);
        if(classify_hardened(shifted) == test_labels[i]) shift2_corr++;
    }
    printf("  SHIFT (+2, +2):  %.2f%%\n", 100.0*shift2_corr/TEST_N);

    return 0;
}
