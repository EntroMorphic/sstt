# Contribution 16: Spectral Coarse-to-Fine Classification

## Concept

The cascade architecture (vote → top-K → dot product → k-NN) achieves
96%+ accuracy but requires an inverted index, IG weights, multi-probe
neighbors, and per-image dot products. The hypothesis: replace the
vote-based candidate selection with a spectral filter — use WHT prototype
dot products to cheaply prune classes, then partial spectral dot products
to prune candidates, then full dot products only on survivors.

The result: spectral-only filtering fails (71.95%), but hybrid
vote→spectral pipelines achieve 95.57–96.12% with significant speedups.

## The Original Spectral Hypothesis

The proposed pipeline:

```
Stage 1: WHT prototype filter    — top 3 classes by prototype dot
Stage 2: Coarse spectral prune   — partial WHT dot (first C coeffs)
Stage 3: Full dot refinement     — complete WHT dot on survivors
Stage 4: k=3 majority vote       — classify from top-3 neighbors
```

### Why It Failed: 71.95%

The prototype filter (Stage 1) is the bottleneck. WHT prototypes
(per-class mean spectral vectors) classify at only 56.83% (pixel) or
62.88% (3-channel). When used as a filter to select top-3 classes, the
true class is retained only ~57% of the time. Downstream stages cannot
recover from this early pruning error.

**Root cause: spectral energy distribution of ternary images.**

Ternary images have sharp edges: pixel values jump directly from -1 to
+1 with no smooth transition. In the frequency domain, sharp edges spread
energy across all frequencies. The WHT coefficients do not concentrate
discriminative information in the low-frequency components — unlike
natural images where low frequencies carry most of the variance.

This means the conventional spectral ordering (process low frequencies
first, prune by partial dot product) is wrong for ternary data. A
partial dot product using the first C coefficients captures neither the
dominant variance nor the discriminative structure.

## The Redesign: Vote-Based Candidate Selection

The key insight: the cascade's vote stage already provides 91% filter
accuracy (the true nearest neighbor is in the top-K candidates 91% of
the time). This is far better than any spectral filter for ternary data.

The redesign replaces the spectral prototype filter with vote-based
candidate selection, then uses WHT dot products for refinement:

```
Stage 1: Vote-based candidate selection  (inverted index)
Stage 2: Full WHT dot product refinement (int16 SIMD)
Stage 3: k=3 majority vote
```

## G1: Pixel Vote → WHT k=1

The simplest hybrid: raw pixel inverted index vote (no IG weights, no
multi-probe) selects top-K candidates, then WHT full dot product ranks
them.

```
K=500, k=1:  95.57%
Time:         8.7 sec
Speedup:      14.8× vs brute WHT
```

This is the minimal viable pipeline: a single-channel vote stage with
no weighting, followed by spectral refinement. The 95.57% result
demonstrates that the vote stage's filtering power, not the spectral
domain's discriminative ability, drives accuracy.

## G2: 3-Channel IG Multi-Probe Vote → WHT k=3

The full hybrid: 3-channel (pixel + h-grad + v-grad) IG-weighted
multi-probe voting selects top-K candidates, then WHT full dot + k=3
majority vote classifies.

```
K=500, k=3:  96.12%
Time:         68 sec
Speedup:      ~2× vs brute WHT
```

This matches the WHT brute k=3 result (96.12%) exactly — but with
fewer dot products (5M vs 600M). The vote stage selects candidates so
well that refining only 500 per query loses nothing compared to
exhaustive search.

## Full G1 Results Table

```
K      | k=1      k=3      k=5    | WHT dots
-------+---------------------------+---------
   50  | 94.66%   94.44%   94.14% | 500,000
  100  | 95.08%   95.11%   94.79% | 1,000,000
  200  | 95.30%   95.48%   95.29% | 2,000,000
  500  | 95.57%   95.70%   95.68% | 5,000,000
```

## Full G2 Results Table

```
K      | k=1      k=3      k=5    | WHT dots
-------+---------------------------+---------
   50  | 95.67%   95.71%   95.48% | 500,000
  100  | 95.82%   95.94%   95.84% | 1,000,000
  200  | 95.93%   96.09%   96.04% | 2,000,000
  500  | 96.03%   96.12%   96.12% | 5,000,000
```

## Key Insight

The cascade's vote stage (91% recall of the true nearest neighbor at
K=500) is far better than any spectral filter for ternary data. The
vote stage exploits *spatial block patterns* — the same patterns that
define the ternary representation. The spectral domain provides
efficient dot product computation (`_mm256_madd_epi16` on int16
coefficients) but not efficient filtering.

**Lesson:** Data-independent spectral ordering (low → high frequency)
does not match the discriminative ordering for ternary images. The
ternary lattice's discriminative structure is spread across all
frequencies because of the sharp edges inherent in {-1, 0, +1} data.
Spatial block patterns, which directly capture this structure, are the
right basis for candidate filtering.

## Files

- `sstt_v2.c` lines 1436-1717: Test G implementation
- `sstt_v2.c` lines 1447-1480: partial and full spectral dot products
- `sstt_v2.c` lines 1519-1598: G1 pixel vote → WHT pipeline
- `sstt_v2.c` lines 1600-1712: G2 3-channel IG multi-probe → WHT pipeline
