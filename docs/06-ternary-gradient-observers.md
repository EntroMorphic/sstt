# Contribution 6: Ternary Gradient Observers

## Concept

Compute horizontal and vertical gradients directly in ternary space:

```
h_grad[y][x] = clamp(tern[y][x+1] - tern[y][x], {-1,0,+1})
v_grad[y][x] = clamp(tern[y+1][x] - tern[y][x], {-1,0,+1})
```

The result is itself ternary: +1 = transition from dark to light (or
zero to light), -1 = transition from light to dark, 0 = no change.

This creates three "observer channels" of the same image — pixel (what
the ink looks like), h-grad (horizontal edge transitions), and v-grad
(vertical edge transitions) — all in the same {-1, 0, +1} encoding,
all compatible with the same pipeline (block signatures → hot maps →
inverted indices → voting → dot products).

## Implementation

```c
/* sstt_v2.c:290-317 */
static void compute_gradients_one(const int8_t *tern,
                                   int8_t *h_grad, int8_t *v_grad) {
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W - 1; x++) {
            int diff = tern[y * IMG_W + x + 1] - tern[y * IMG_W + x];
            h_grad[y * IMG_W + x] = clamp_trit(diff);
        }
        h_grad[y * IMG_W + IMG_W - 1] = 0;  /* last column */
    }
    /* Similar for v_grad (row differences) */
}
```

## Key Properties

1. **Closed under ternary** — input is {-1,0,+1}, output is {-1,0,+1}.
   No precision expansion, no floating point, no normalization.

2. **Complementary information** — pixel says "what's here," gradient says
   "what changed." A horizontal stroke has strong v-grad signal but weak
   h-grad; a vertical stroke is the opposite. This directional information
   is invisible to the pixel channel alone.

3. **Different background** — pixel background is (-1,-1,-1) = block 0;
   gradient background is (0,0,0) = block 13. See contribution #5.

4. **V-Grad outperforms Pixel** — V-Grad alone achieves 73.95% vs
   Pixel's 71.35% in hot map classification. Vertical transitions are
   more discriminative for digit shape than raw pixel values.

## Impact Across Pipeline

| Stage | Pixel only | 3-Channel | Improvement |
|-------|-----------|-----------|-------------|
| Hot map | 71.35% | 73.23% | +1.88 pp |
| IG vote | 83.27% | 88.78% | +5.51 pp |
| Multi-probe | — | 91.16% | — |
| Full cascade | — | 96.04% | — |
| WHT proto | 56.83% | 62.88% | +6.05 pp |

The gradient channels provide the largest accuracy gains in the
intermediate pipeline stages (IG voting and prototypes), where the
directional information helps disambiguate digits that have similar
pixel density but different edge orientations (e.g., 3 vs 8).

## Connection to SDF (#4)

The SDF boundary detection step (4-neighbor sign-change check) is
fundamentally a gradient magnitude test: a pixel is on the boundary iff
its discrete gradient is nonzero. SDF captures *distance from edges*;
gradients capture *direction of edges*. Together they would provide
both scalar and vector information about shape boundaries.

## Files

- `sstt_v2.c` lines 283-336: gradient computation
- `sstt_v2.c` lines 104-108: gradient data globals
- `sstt_v2.c` lines 364-389: gradient block signatures
