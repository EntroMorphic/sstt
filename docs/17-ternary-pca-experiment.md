# Contribution 17: Ternary PCA — Why Eigenvectors Failed as Features

## Hypothesis

PCA eigenvectors of ternary image data, quantized to {-1, 0, +1},
should concentrate discriminative signal into fewer dimensions. Projecting
images onto ternary eigenvectors via ternary dot product, then quantizing
the projections and block-encoding them, should yield a more compact and
more discriminative feature space for hot map classification.

## Pipeline

```
1. Compute pixel covariance (784×784) from ternary training data
2. Extract top-K eigenvectors via power iteration (50 iterations each)
3. Quantize eigenvectors to ternary: top N% of |values| → ±1, rest → 0
4. Project images: ternary dot product with each ternary eigenvector
5. Quantize projections to ternary: |proj| > mean(|proj|) → ±1, else 0
6. Block-encode projected features (groups of 3 PCs → block value)
7. Hot map classify on projected features
```

The power iteration avoids forming the full 784x784 covariance matrix.
Instead, it implicitly multiplies via X^T X v using two passes over the
training data per iteration. Deflation removes previous eigenvectors to
extract successive components.

## Results

Pixel hot map baseline: **71.35%** (252 2D blocks, same code).

| PCs | Blocks | Threshold | Accuracy | vs Baseline |
|-----|--------|-----------|----------|-------------|
| 27  | 9      | top 50%   | 55.27%   | -16.08 pp   |
| 27  | 9      | top 67%   | 51.54%   | -19.81 pp   |
| 27  | 9      | top 33%   | 48.65%   | -22.70 pp   |
| 54  | 18     | top 50%   | 52.63%   | -18.72 pp   |
| 84  | 28     | top 50%   | 51.86%   | -19.49 pp   |
| 126 | 42     | top 50%   | 50.21%   | -21.14 pp   |
| 252 | 84     | top 50%   | 48.85%   | -22.50 pp   |

**Every configuration is 12–23 pp below the pixel baseline.** Adding
more PCs makes accuracy *worse*, not better.

The best result (55.27% with 27 PCs at 50% threshold) uses only 9
blocks — vs 252 blocks in the pixel baseline. Despite having 28x fewer
blocks, the loss is "only" 16 pp, suggesting the PCA projection does
capture some class structure. But it cannot compete with raw spatial
features.

## Root Cause Analysis

### 1. PCA Optimizes for Variance, Not Discrimination

PCA finds directions of maximum variance across all classes combined.
The first eigenvector captures the axis of greatest spread — which for
MNIST is roughly "how much ink is on the page." This separates empty
from full images, not 3s from 8s.

Discriminative methods (LDA, IG weighting) would concentrate on
between-class differences. PCA concentrates on total variance, which
is dominated by within-class variation and background.

### 2. Double Quantization Destroys Variance Structure

The PCA eigenvalue decomposition assumes linear algebra in continuous
space. Two quantization steps break this assumption:

- **Eigenvector quantization:** The top 50% of each eigenvector's
  components become ±1; the rest become 0. This collapses the fine
  gradient structure that makes eigenvectors orthogonal. After
  quantization, the ternary eigenvectors are no longer orthogonal.

- **Projection quantization:** Projections are further quantized to
  {-1, 0, +1} using per-PC thresholds. A projection of +47 and +3
  both become +1. This collapses the variance structure PCA was
  designed to preserve.

### 3. Projection Destroys Spatial Layout

This is the most critical failure. The hot map and block encoding
derive their power from **spatial addressing**: block k encodes position
(row, column) in the image. Block 0 always means "top-left corner,"
block 251 always means "bottom-right."

After PCA projection, the features are ordered by eigenvalue rank, not
by spatial position. Block 0 in the projected space means "projection
onto PC0, PC1, PC2" — these are global features with no spatial
locality. The hot map's position-dependent frequency counting, which
implicitly captures "this pattern occurs at this location," loses all
spatial information.

The hot map is fundamentally a spatial cipher. PCA produces a spectral
(variance-ranked) representation. These are incompatible.

### 4. The Eigenvalue Spectrum

```
PC0:    21.2
PC10:    3.1
PC50:    0.8
PC100:   0.3
PC250:   0.1
```

The rapid decay means the first few PCs dominate. But the hot map
cannot exploit this — it treats all block positions equally (or nearly
so, without IG weighting). The eigenvalue-weighted importance of PCs
is invisible to the block encoding.

## Conclusion

PCA is the wrong basis for the SSTT architecture. The system's power
comes from spatial addressing — the hot map's ability to say "at
position k, this block value predicts class c." PCA destroys spatial
addressing by reordering features by variance instead of position.

The correct research direction for dimensionality reduction in this
architecture is not to change the feature basis (PCA), but to compress
the spatial representation directly — e.g., eigenvalue-guided splines
over the position dimension (see contribution #18).

## Files

- `sstt_tpca.c`: Complete implementation (PCA + projection + hot map)
- `sstt_tpca.c` lines 185-274: Power iteration PCA
- `sstt_tpca.c` lines 284-326: Eigenvector ternary quantization
- `sstt_tpca.c` lines 332-353: Ternary dot product projection
- `sstt_tpca.c` lines 363-408: Projection quantization
- `sstt_tpca.c` lines 418-454: Block encoding + hot map on projected features
