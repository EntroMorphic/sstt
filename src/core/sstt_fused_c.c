/*
 * sstt_fused_c.c — Pure C reference implementation of the fused kernel.
 * Same interface as sstt_fused.S for A/B testing.
 */

#include <stdint.h>
#include <string.h>
#include "sstt_fused.h"

static inline int8_t quant(uint8_t p) {
    if (p > 170) return 1;
    if (p < 85) return -1;
    return 0;
}

static inline int8_t clamp_trit(int d) {
    if (d > 0) return 1;
    if (d < 0) return -1;
    return 0;
}

static inline uint8_t blk_enc(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0+1)*9 + (t1+1)*3 + (t2+1));
}

int sstt_fused_classify(const uint8_t *pixels,
                        const uint32_t *px_hot,
                        const uint32_t *hg_hot,
                        const uint32_t *vg_hot) {
    uint32_t acc[FUSED_CLS_PAD] = {0};
    int8_t cur[28], nxt[28], hg[28], vg[28];

    for (int row = 0; row < 28; row++) {
        const uint8_t *rp = pixels + row * 28;
        for (int x = 0; x < 28; x++) cur[x] = quant(rp[x]);

        if (row < 27) {
            const uint8_t *np = pixels + (row+1) * 28;
            for (int x = 0; x < 28; x++) nxt[x] = quant(np[x]);
        }

        for (int x = 0; x < 27; x++) hg[x] = clamp_trit(cur[x+1] - cur[x]);
        hg[27] = 0;

        if (row < 27)
            for (int x = 0; x < 28; x++) vg[x] = clamp_trit(nxt[x] - cur[x]);
        else
            memset(vg, 0, 28);

        for (int s = 0; s < 9; s++) {
            int b = s * 3;
            int k = row * 9 + s;
            size_t koff = (size_t)k * FUSED_N_BVALS * FUSED_CLS_PAD;

            uint8_t bv_px = blk_enc(cur[b], cur[b+1], cur[b+2]);
            if (bv_px != FUSED_BG_PIXEL)
                for (int c = 0; c < FUSED_CLS_PAD; c++)
                    acc[c] += px_hot[koff + (size_t)bv_px * FUSED_CLS_PAD + c];

            uint8_t bv_hg = blk_enc(hg[b], hg[b+1], hg[b+2]);
            if (bv_hg != FUSED_BG_GRAD)
                for (int c = 0; c < FUSED_CLS_PAD; c++)
                    acc[c] += hg_hot[koff + (size_t)bv_hg * FUSED_CLS_PAD + c];

            uint8_t bv_vg = blk_enc(vg[b], vg[b+1], vg[b+2]);
            if (bv_vg != FUSED_BG_GRAD)
                for (int c = 0; c < FUSED_CLS_PAD; c++)
                    acc[c] += vg_hot[koff + (size_t)bv_vg * FUSED_CLS_PAD + c];
        }
    }

    int best = 0;
    for (int c = 1; c < FUSED_N_CLASSES; c++)
        if (acc[c] > acc[best]) best = c;
    return best;
}
