/*
 * sstt_fused.h — Fused AVX2 Ternary Classification Kernel
 *
 * Raw pixels → class label in one pass. Zero intermediate arrays.
 * The entire computation lives in L-cache:
 *   - Code: <1KB (L1i)
 *   - Working set: ~160 bytes stack (L1d)
 *   - Model: 3 hot maps ≈ 1.3MB (L2/L3, Shared state across cores)
 *
 * The 11 ternary primitives fused into a single streaming pipeline:
 *   quantize → gradient → block_encode → bg_gate → lookup → accumulate → argmax
 */

#ifndef SSTT_FUSED_H
#define SSTT_FUSED_H

#include <stdint.h>

/* 2D block layout (from sstt_v2): 9 strips/row × 28 rows = 252 blocks */
#define FUSED_IMG_W       28
#define FUSED_IMG_H       28
#define FUSED_PIXELS      784
#define FUSED_BLKS_PER_ROW 9
#define FUSED_N_BLOCKS    252       /* 9 × 28 */
#define FUSED_N_BVALS     27        /* 3^3 */
#define FUSED_CLS_PAD     16        /* 10 classes padded to 16 for AVX2 */
#define FUSED_N_CLASSES   10

/* Hot map: uint32_t[252][27][16], 32-byte aligned.
 * Each [k][bv] entry is 64 bytes = 1 cache line = 2 YMM loads.
 * hot[k][bv][c] = count of training images of class c with
 *                  block value bv at position k. */
#define FUSED_HOT_STRIDE_BV   (FUSED_CLS_PAD * sizeof(uint32_t))  /* 64 */
#define FUSED_HOT_STRIDE_BLK  (FUSED_N_BVALS * FUSED_HOT_STRIDE_BV)  /* 1728 */
#define FUSED_HOT_SIZE        (FUSED_N_BLOCKS * FUSED_HOT_STRIDE_BLK) /* 435456 */

/* Background block values to skip during voting */
#define FUSED_BG_PIXEL    0         /* block_encode(-1,-1,-1) */
#define FUSED_BG_GRAD     13        /* block_encode(0,0,0) */

/*
 * Classify one image from raw pixels. Returns class label [0-9].
 *
 * pixels:  784 uint8_t raw pixel bytes. Must have 4 bytes readable
 *          padding after byte 783 (for 32-byte aligned row loads).
 * px_hot:  Pixel channel hot map, uint32_t[252][27][16], 32-byte aligned.
 * hg_hot:  H-gradient channel hot map, same layout.
 * vg_hot:  V-gradient channel hot map, same layout.
 *
 * All computation happens in-register and on-stack (<256 bytes).
 * Hot maps are read-only — safe for concurrent multi-core use
 * under MESI Shared state.
 */
int sstt_fused_classify(const uint8_t *pixels,
                        const uint32_t *px_hot,
                        const uint32_t *hg_hot,
                        const uint32_t *vg_hot);

#endif /* SSTT_FUSED_H */
