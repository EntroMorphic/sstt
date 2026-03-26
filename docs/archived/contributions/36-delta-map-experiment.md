# Contribution 36: Ground-State Delta Map — Honest Results

Tests the hypothesis from contribution 33 (Option 3): can the Gauss map
delta — how an image deviates from its class mean — replace or improve
the ranking step?

Code: `src/sstt_gauss_delta.c`

---

## Hypothesis

Each class has a "ground state" — the mean Gauss map histogram across
all training images of that class. Two images that deviate from the
class mean in the same way are structurally similar in a discriminative
sense. This delta signature might capture class-specific structure that
the raw dot product and divergence features miss.

Two modes tested:
- **Mode 0 (raw delta L1):** L1 distance between query and candidate
  deltas from the class-mean Gauss map
- **Mode 1 (quantized ternary match):** Quantize delta to {below, at,
  above} class mean, then match ternary signatures

---

## MNIST Results

Ground state L1 distances between class means:

| Pair | L1 |
|------|----|
| 3<->5 | 15 |
| 4<->9 | 22 |
| 3<->8 | 44 |
| 0<->6 | 61 |
| 2<->4 | 68 |
| 1<->7 | 110 |

The hardest pairs (3<->5, 4<->9) have the smallest ground-state
separation — consistent with the error analysis from contribution 23.

### Delta weight sweep (mode 0, raw L1)

| Weight | Accuracy | Delta |
|--------|----------|-------|
| 0 | 96.99% | baseline |
| 8 | 97.01% | +2 |
| 16 | 97.02% | +3 |
| 32 | 97.05% | +6 |
| 64 | 97.07% | +8 |
| 128 | 97.10% | +11 |

Monotonically improving — the delta carries real signal on MNIST.

### Full grid search

| Mode | Best accuracy | Delta vs baseline | Best params |
|------|--------------|-------------------|-------------|
| Raw delta L1 | **97.18%** | +19 errors fixed | d=200, c=50, g=100, gmd=32, S=30, dS=2, K=10 |
| Quantized ternary | 97.16% | +17 errors fixed | d=300, c=50, g=50, gmd=8, S=50, dS=2, K=10 |

Both modes achieve nearly identical results. The delta fixes ~19 errors
on MNIST.

---

## Fashion-MNIST Results

Ground state L1 distances between class means:

| Pair | L1 |
|------|----|
| 2<->4 | 62 |
| 0<->6 | 107 |
| 3<->5 | 129 |
| 3<->8 | 152 |
| 4<->9 | 161 |
| 1<->7 | 242 |

Fashion has much larger ground-state separation (classes are more
structurally distinct in the Gauss map space).

### Delta weight sweep (mode 0, raw L1)

| Weight | Accuracy | Delta |
|--------|----------|-------|
| 0 | 85.81% | baseline |
| 2 | 85.82% | +1 |
| 4 | 85.77% | -4 |
| 16 | 85.68% | -13 |
| 128 | 85.49% | -32 |

**The delta actively hurts Fashion in isolation.** Higher weights
make it worse. The signal is weak and trades off against existing
features.

### Full grid search

| Mode | Best accuracy | Delta vs baseline | Best params |
|------|--------------|-------------------|-------------|
| Raw delta L1 | **86.01%** | +20 errors fixed | d=100, c=25, g=200, gmd=16, S=50, dS=2, K=15 |
| Quantized ternary | 86.01% | +20 errors fixed | d=100, c=25, g=100, gmd=32, S=50, dS=2, K=15 |

The full grid search finds a configuration where the delta helps by
re-balancing other weights (reducing divergence weight from 200 to 100).
The delta's contribution is small (+20 errors) and comes at the cost
of reducing other feature weights.

---

## Assessment

### What the delta map does

1. **Carries real signal on MNIST** (+19 errors, monotonic weight sweep).
   The delta captures class-specific deviation structure that helps
   distinguish confusable pairs (3<->5, 4<->9).

2. **Hurts Fashion in isolation** but helps (+20 errors) when other
   weights are re-optimized. The delta doesn't add independent
   information on Fashion — it reshuffles the weight budget.

3. **Both modes (raw L1, quantized ternary) perform identically.**
   The quantization doesn't lose significant information.

### What the delta map does NOT do

1. **Does not replace the ranking step.** The "three table lookups"
   architectural endgame (contribution 33, Option 3) is not validated.
   The delta is a marginal feature addition, not a paradigm shift.

2. **Does not break through the 97% ceiling on MNIST.** 97.18% is
   below the topo9 best of 97.31% (which uses sequential processing
   instead of delta features). The delta map and sequential processing
   appear to target similar error modes.

3. **Does not generalize cleanly.** MNIST shows a clean monotonic
   signal; Fashion does not. The feature is dataset-dependent.

### Comparison to alternatives

| Method | MNIST | Fashion | Mechanism |
|--------|-------|---------|-----------|
| topo9 sequential | 97.31% | 85.81% | Bayesian candidate processing |
| Delta map (this) | 97.18% | 86.01% | Gauss map deviation |
| topo8 static | 97.27% | 84.80% | Grid divergence only |

The delta map is competitive with sequential processing on Fashion
(86.01% vs 85.81%) but worse on MNIST (97.18% vs 97.31%). They may
be complementary — testing their combination is a natural next step.

### Verdict on "maps all the way down"

**Negative result for the architectural claim.** The delta map is a
useful feature (+19-20 errors) but not a replacement for per-candidate
computation. The ranking step cannot be collapsed into a single table
lookup. The "maps all the way down" narrative from contribution 33
is not supported by these results.

This is documented as a partial negative result: the feature helps,
the architectural claim does not hold.

---

## Files

Code: `src/sstt_gauss_delta.c`
This document: `docs/36-delta-map-experiment.md`
Build: `make sstt_gauss_delta`
Run: `./sstt_gauss_delta` (MNIST) or `./sstt_gauss_delta data-fashion/` (Fashion)
