# Contribution 15: Fashion-MNIST Generalization

## Motivation

MNIST digit classification is widely regarded as "too easy" — a linear
classifier reaches 92%, and even the simplest k-NN hits 96%. The question
is whether the SSTT geometric framework generalizes to harder data, or
whether its performance is an artifact of the simplicity of handwritten
digits.

Fashion-MNIST is a drop-in replacement: same 28x28 grayscale format,
same 60K/10K train/test split, same 10 classes. But the classes
(T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag,
ankle boot) have more texture variation, more intra-class diversity, and
less distinctive spatial structure than digits.

## Implementation

Zero algorithm changes. The only modification is the data directory:

```
./sstt_geom data/fashion/
```

The `argv[1]` parameter selects the data directory. All thresholds,
block sizes, cascade parameters, and pipeline stages remain identical.
This is a true generalization test: same code, different data.

## Full Results (sstt_geom on Fashion-MNIST)

| Test | Method | MNIST | Fashion | Delta |
|------|--------|-------|---------|-------|
| A | Ternary Centroid | 82.07% | — | — |
| B | Brute 1-NN | 95.87% | 76.86% | -19.01 pp |
| C | Indexed K=500 vote→refine | 96.00% | 81.05% | -14.95 pp |
| D | Hot map (pixel) | 71.35% | 49.39% | -21.96 pp |
| D | V-Grad hot map | 73.95% | 62.64% | -11.31 pp |
| C | Vote-only (no dot product) | 84.43% | 69.92% | -14.51 pp |
| F | SDF | ~95% | ~52% | ~-43 pp |

### Analysis by Method

**Brute 1-NN: 76.86%** — The geometric ceiling. Nearest-neighbor in
ternary space finds the closest training image, but with only 3 intensity
levels, clothing textures that differ in continuous grayscale may map to
identical ternary representations. The 19 pp drop from MNIST reflects the
loss of fine texture detail under ternary quantization.

**Indexed K=500: 81.05% (+4.19 pp over brute)** — The vote→refine
cascade *exceeds* brute force, just as it does on MNIST. With 500
candidates and k-NN refinement, the cascade actually outperforms the
all-pairs nearest neighbor by 4.19 pp. This means the vote-based
candidate selection is not just an approximation — it acts as a
regularizer, filtering out deceptive near-matches.

**Hot map (pixel): 49.39%** — Severe degradation. The hot map's block
encoding captures local 3-pixel spatial patterns. For digits, these
patterns are highly class-discriminative (a "1" has a distinctive column
pattern, "0" has distinctive edge blocks). For clothing, local texture
patches repeat across classes — a T-shirt collar and a pullover collar
produce similar block values.

**V-Grad: 62.64%** — Edge direction is more transferable than fill.
Vertical gradients capture silhouette structure (sleeve edges, trouser
legs, necklines) which varies less across instances than texture fill.
The 13 pp gap between V-Grad and pixel hot map on Fashion vs the 2.6 pp
gap on MNIST shows that gradient features degrade more gracefully.

**Vote-only: 69.92%** — Pure addressing without dot product refinement
reaches 70%. The inverted index maps each block value to specific
training images, and vote aggregation across 252 blocks provides enough
signal even for clothing.

**SDF: ~52%** — The signed distance field assumes binary topology
(foreground/background boundary structure). Clothing images often have
complex internal structure (stripes, patterns, collars) that violates
this assumption. SDF is the worst-performing method on Fashion-MNIST.

## Key Findings

1. **The vote→refine architecture generalizes.** On both MNIST and
   Fashion-MNIST, the cascade (vote stage → dot product refinement)
   *exceeds* brute-force 1-NN. The vote stage provides regularization,
   not just speedup.

2. **Hot map alone does not generalize.** The 22 pp drop (71→49%)
   shows that local block frequency statistics are insufficient when
   class-discriminative information is distributed across global
   structure rather than concentrated in local patches.

3. **Gradient features degrade gracefully.** The V-Grad channel's
   11 pp drop (74→63%) is roughly half the pixel channel's 22 pp drop.
   Edge orientation is a more robust feature than raw pixel value
   for cross-domain transfer.

4. **The SDF assumption is domain-specific.** Topology-based features
   assume clear foreground/background separation. This holds for
   handwritten digits but not for textured clothing.

## The Exit Code

`sstt_geom` exits with code 1 when accuracy falls below an 80%
threshold. On Fashion-MNIST, the hot map test returns 49.39%, triggering
exit code 1. This is a quality gate, not an error — it means the method
did not meet the accuracy threshold on this dataset.

## Files

- `sstt_geom.c`: Full pipeline (Tests A–H), `argv[1]` selects data dir
- `sstt_fused_test.c`: Fused kernel test, `argv[1]` selects data dir
- `Makefile`: `make fashion` target downloads Fashion-MNIST data
