# Contribution 5: Background-Value-Per-Channel Discrimination

## Discovery

When extending the hot map cipher from single-channel (pixel) to
multi-channel (pixel + gradients), the gradient channels initially
showed near-random accuracy. The bug: gradient hot maps were skipping
block value 0 (the pixel background), but the gradient background is
block value 13.

```
Pixel background:   block_encode(-1,-1,-1) = 0   (all dark = MNIST padding)
Gradient background: block_encode( 0, 0, 0) = 13  (no change = flat region)
```

The all-zeros gradient block (no transition) is the uninformative one,
not the all-negative block. Skipping bv=0 for gradients kept all the
uninformative flat-region votes while discarding informative edge blocks.

## Implementation

```c
/* sstt_v2.c:51-55 */
#define BG_PIXEL    0    /* block_encode(-1,-1,-1) */
#define BG_GRAD     13   /* block_encode( 0, 0, 0) */

/* sstt_v2.c:418-440 */
static inline int hot_classify_chan(const uint8_t *qsig,
                                     uint32_t hot[][N_BVALS][CLS_PAD],
                                     uint8_t bg_val) {
    for (int k = 0; k < N_BLOCKS2D; k++) {
        uint8_t bv = qsig[k];
        if (bv == bg_val) continue;  /* channel-specific background */
        /* ... accumulate ... */
    }
}
```

## Impact

Before fix (skip bv=0 for all channels):
- Pixel: 71.35%
- H-Grad: ~20% (near random)
- V-Grad: ~20% (near random)

After fix (skip bv=13 for gradients):
- Pixel: 71.35% (unchanged)
- H-Grad: 60.29%
- V-Grad: 73.95%
- Combined 3-channel: 73.23%

V-Grad alone (73.95%) actually outperforms Pixel alone (71.35%),
suggesting vertical transitions are more discriminative for digit
classification than raw pixel values.

## Why It Matters

This is a general principle for multi-modal ternary systems: each channel
has its own "uninformative" encoding, and the background skip must be
parameterized per channel. The optimization (skip background) is critical
for performance (saves ~60% of loads) but the background value itself is
domain-specific.

## Files

- `sstt_v2.c` lines 51-55: background value constants
- `sstt_v2.c` lines 416-440: parameterized hot map classifier
- `sstt_v2.c` lines 442-482: multi-channel hot map combining 3 channels
