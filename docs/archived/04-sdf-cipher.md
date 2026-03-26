# Contribution 4: SDF Cipher — Topological Compression

## Concept

The Signed Distance Field (SDF) transforms pixel-ternary (what the ink
looks like) into topology-ternary (what shape the ink makes). Two images
with different stroke thickness but the same letter form collapse to
nearly the same SDF representation.

The SDF is then quantized back to ternary and fed through the exact same
hot map cipher pipeline — yielding a second 451 KB cipher that encodes
*topological* rather than *appearance* information.

## Architecture

```
Pipeline:
  ternary image → raw SDF (int8, Chamfer distance) → threshold T
  → SDF ternary {-1,0,+1} → block signatures → SDF hot map

Boundary detection (first step of SDF):
  A pixel is on the boundary if any 4-neighbor has opposite sign or
  is zero. Boundary pixels get distance 0; all others get Manhattan
  distance to nearest boundary via two-pass Chamfer propagation.

Sign convention:
  Foreground (+1 ink) → negative SDF (inside)
  Background (-1)     → positive SDF (outside)
  Transition (0)      → zero SDF (on boundary)
```

## Implementation

```c
/* sstt_geom.c:1066-1140 */
static void compute_raw_sdf(const int8_t *tern, int8_t *sdf) {
    /* Step 1: boundary detection (4-neighbor sign-change) */
    /* Step 2: forward Chamfer pass (top-left → bottom-right) */
    /* Step 3: backward Chamfer pass (bottom-right → top-left) */
    /* Step 4: apply sign, clamp to int8 */
}

/* AVX2 SDF ternary quantization */
/* sstt_geom.c:1163-1187 */
static void quantize_sdf_one(const int8_t *sdf_raw, int8_t *sdf_tern,
                              int threshold) {
    /* SDF < -T → -1, |SDF| ≤ T → 0, SDF > T → +1 */
}
```

## Threshold Sweep Results

| T | SDF Accuracy | Trit Distribution |
|---|-------------|-------------------|
| 1 | ~65%        | -1=40% 0=19% +1=41% |
| 2 | ~68%        | -1=33% 0=26% +1=41% |
| 3 | ~66%        | -1=27% 0=30% +1=43% |
| 4 | ~63%        | -1=22% 0=32% +1=46% |

Best threshold T=2 balances discrimination (enough active buckets) with
noise reduction (smoothing out single-pixel variations).

## Dual Cipher

The pixel hot map and SDF hot map can be combined — just accumulate both
frequency vectors in the same AVX2 accumulators:

```c
/* sstt_geom.c:1234-1264 */
static inline int dual_classify(const uint8_t *px_sig,
                                 const uint8_t *sdf_sig) {
    /* Sum pixel hot map + SDF hot map votes */
}
```

- Pixel alone: 71.12%
- SDF alone:   ~68%
- Dual cipher: ~73% (+2pp over pixel)
- Total size:  902 KB (2 × 451 KB)

## What Makes It Novel

1. **Same cipher, different address space** — the hot map architecture is
   reused verbatim. Only the input encoding changes.
2. **Topological invariance** — thick and thin '7' have different pixel
   patterns but nearly identical SDF patterns.
3. **Lower entropy** — SDF has fewer unique signatures than pixel (the
   codebook compresses ~1.5x better), confirming it captures invariants.
4. **SDF boundary detection IS a gradient operation** — the 4-neighbor
   sign-change test is a discrete gradient magnitude check. This connects
   SDF to contribution #6 (ternary gradients).

## Files

- `sstt_geom.c` lines 1040-1462: SDF computation, quantization, cipher,
  dual classifier, entropy analysis, codebook compression
