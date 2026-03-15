# Contribution 18: Eigenvalue-Guided Weighted Splines — A Scaling Framework

## The Scaling Problem

The hot map is a lookup table indexed by (block_position, block_value,
class). For 28x28 MNIST images with 3-pixel horizontal strips:

```
252 blocks × 27 values × 16 classes × 4 bytes = 435 KB per channel
3 channels = 1.3 MB total
```

This fits in L2 cache. But at 224x224 (ImageNet resolution):

```
Block positions: 74 rows × 74 strips = 5,476 blocks per channel
3 channels × 5,476 × 27 × 16 × 4 bytes ≈ 57 MB per channel
Total: ~170 MB
```

This does not fit in any cache. The hot map's defining advantage — pure
table lookup with no per-image computation — becomes a liability because
the table outgrows the cache hierarchy.

## The Idea: Spline Compression of the Position Dimension

The hot map stores a 10-class frequency vector for each (position, value)
pair. Instead of storing all 5,476 position entries explicitly, fit
weighted B-splines across the position dimension:

```
hot_approx[k][bv][c] = Σ_j  w_j * B_j(k)
```

where B_j are B-spline basis functions centered at knot positions, and
w_j are learned weights. At training time, evaluate the splines at all
integer positions to produce the full hot map. At inference time, the
lookup table is still a flat array of integers — the spline is a
training-time compression step, not a runtime computation.

The key question is where to place the knots.

## Knot Placement: Eigenvalue Guidance

Compute the covariance of the block signature vectors across training
images. The eigenvalues of this covariance matrix indicate where the
block signatures vary most:

```
Σ = (1/N) Σ_i (sig_i - μ)(sig_i - μ)^T     [252 × 252 matrix]
```

Eigendecomposition of Σ reveals which linear combinations of block
positions carry the most variance. Regions of position space where the
top eigenvectors have large components correspond to positions where the
signatures vary most across training images — these are the positions
that need the most knots.

**Knot placement algorithm:**

1. Compute top-M eigenvectors of the block signature covariance.
2. For each position k, compute the weighted eigenvector magnitude:
   `e(k) = Σ_m λ_m * |v_m(k)|^2`
3. Place knots proportional to e(k) — more knots where eigenvalue-
   weighted variance is highest.
4. This concentrates spline resolution in the "interesting" parts of
   position space and compresses the "boring" parts (e.g., the always-
   background corners).

## Knot Weighting: Information Gain

The IG weights (contribution #7) measure how much each block position
discriminates between classes. Knots in high-IG regions should be
weighted more heavily in the spline fit:

```
Fitting objective: minimize Σ_k  ig(k) * ||hot[k] - spline(k)||^2
```

This biases the spline to perfectly reproduce the hot map at
discriminative positions, at the expense of less accuracy at
uninformative positions (which contribute little to classification
anyway).

## Why Eigenvalues and IG Are Complementary

- **Eigenvalue (variance):** "At position k, how much do block
  signatures vary across the training set?" High variance means the
  position is *informative* — it takes many different values depending
  on the input. Low variance means the position is nearly constant
  (e.g., always background).

- **IG (discrimination):** "At position k, how much does knowing the
  block value reduce uncertainty about the class label?" High IG means
  the position is *class-discriminative* — different classes produce
  different block values here.

A position can have high variance but low IG (many patterns, but they
don't predict class), or low variance but high IG (few patterns, but
they perfectly separate classes). The combination captures both:

- Eigenvalues guide **where to put knots** (don't waste resolution on
  constant regions).
- IG guides **how to weight the fit** (prioritize accuracy at
  discriminative positions).

## MNIST: Marginal Improvement

For 28x28 images, there are only 252 block positions. The hot map is
already 435 KB per channel — well within L2. Spline compression would
reduce this further but with minimal practical benefit:

```
252 positions → 100 knots → ~2.5× compression → ~170 KB per channel
```

The classification accuracy impact would be negligible because the
spline can nearly perfectly reconstruct a 252-element sequence with 100
knots, and the positions lost to compression are low-eigenvalue (always
background) and low-IG (not discriminative).

## Scaling: The Critical Application

At 224x224, the spline compression becomes essential:

```
5,476 positions → 200 eigenvalue-guided knots → ~27× compression
170 MB → ~6 MB total (3 channels)
```

With IG-weighted fitting, the 6 MB hot map would preserve nearly all of
the classification-relevant information while fitting in L3 cache
(typically 8–32 MB). Further compression to ~1000 knots (5×) would
target L2 residency at the cost of some accuracy at medium-variance
positions.

The inference path remains pure integer table lookup — the spline
evaluation happens once at training time to produce the compressed hot
map. The 200-knot approximation is still a flat array:

```
200 knots × 27 values × 16 classes × 4 bytes = 345 KB per channel
3 channels ≈ 1 MB total
```

This is comparable to the current MNIST hot map size, achieving
cache-residency parity at 64× the image resolution.

## Connection to Hilbert Spaces

The ternary dot product `<a, b> = Σ_i a_i * b_i` where a_i, b_i ∈
{-1, 0, +1} defines a valid inner product on the ternary lattice Z_3^n.
This inner product induces a norm and a metric, making the space of
ternary vectors a discrete inner product space.

The WHT provides an orthonormal eigenbasis for this space: if H is the
Hadamard matrix, then `H * H^T = N * I`, and the WHT coefficients are
the coordinates in this eigenbasis. The Parseval identity holds: the
ternary dot product in pixel space equals the spectral dot product in
WHT space (up to a scale factor).

Eigenvalue-guided splines are the natural interpolation framework in
this setting:

1. **The inner product** (ternary dot) defines the geometry.
2. **The eigenbasis** (WHT) provides the natural coordinates.
3. **The eigenvalues** (of the block signature covariance) identify the
   important subspaces.
4. **The splines** interpolate smoothly across the position dimension,
   with resolution guided by the eigenvalue structure.

This is analogous to kernel methods in reproducing kernel Hilbert spaces
(RKHS), where the kernel's eigenfunctions determine the natural basis
for function approximation. Here, the "kernel" is the ternary dot
product, the "eigenfunctions" are the WHT basis vectors, and the
splines provide the approximation framework.

## Files

- No implementation yet — this is a scaling framework design.
- Depends on: `sstt_v2.c` (IG weights), `sstt_geom.c` (hot map
  infrastructure), `sstt_v2.c` (WHT implementation)
