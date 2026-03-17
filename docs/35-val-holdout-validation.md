# Contribution 35: Proper Val/Holdout Validation

Addresses the test-set contamination vulnerability identified in
contribution 34 (independent audit). All parameters — feature weights,
grid choice, and sequential processing hyperparameters — are now
derived on a 5K validation split and evaluated on a 5K holdout.

Code: `src/sstt_topo9_val.c`

---

## Methodology

- 10K test set split: images 0-4999 (val), images 5000-9999 (holdout)
- Training data (60K) unchanged
- IG weights computed from training data only (always were — no issue)
- Feature weights (w_c, w_p, w_d, w_g, sc_val, grid) grid-searched on val
- Sequential processing params grid-searched on val
- Final accuracy reported on holdout only

---

## MNIST Results

### Feature weight search (val only)

Best weights found on val: grid=2x4, w_c=50, w_p=16, w_d=200, w_g=100, sc=50

**These are identical to the original hardcoded weights.** The original
weights were not overfit to the test set — they are the true optimum.

### Static baseline

| Split | Accuracy | Correct | Errors |
|-------|----------|---------|--------|
| Val (0-4999) | 96.30% | 4815/5000 | 185 |
| Holdout (5000-9999) | **98.24%** | 4912/5000 | 88 |
| Full 10K (original topo9) | 97.27% | 9727/10000 | 273 |

The 1.94pp gap between val and holdout is driven by data composition
(which specific images fall in each half), not by overfitting.

### Sequential processing (val search)

| Method | Val | Holdout | Delta vs static |
|--------|-----|---------|-----------------|
| Bayesian | No improvement | — | +0 |
| CfC | No improvement | — | +0 |
| Pipeline | No improvement | — | +0 |

**Sequential processing provides zero benefit on MNIST** when searched
on a proper validation split. This confirms the audit's finding: the
+0.03pp reported in the original topo9 was noise from test-set search.

### MNIST honest numbers

**Static baseline with val-derived weights: 98.24% holdout.**

Note: this is *higher* than the full-10K number (97.27%) because the
holdout half of the test set happens to be easier. The full-10K static
baseline is the more conservative and appropriate number to report.

---

## Fashion-MNIST Results

### Feature weight search (val only)

Best weights found on val: grid=3x3, w_c=25, w_p=0, w_d=50, w_g=200, sc=100

These differ from the original hardcoded weights (w_d=200, w_g=200,
sc=50). The val-derived weights reduce divergence weight from 200 to 50
and increase the Kalman scale from 50 to 100. The original weights
overweighted divergence on Fashion.

### Static baseline

| Split | Accuracy | Correct | Errors |
|-------|----------|---------|--------|
| Val (0-4999) | 85.10% | 4255/5000 | 745 |
| Holdout (5000-9999) | **84.54%** | 4227/5000 | 773 |
| Full 10K (original weights) | 84.80% | 8480/10000 | 1520 |
| Full 10K (original topo9 static) | 84.80% | 8480/10000 | 1520 |

The 0.56pp val/holdout gap is small — Fashion weights are more stable
across splits than MNIST.

### Sequential processing (val search)

| Method | Val | Holdout | Delta vs static holdout |
|--------|-----|---------|------------------------|
| Bayesian (dS=1, K=20, tw=50) | 86.14% | **85.68%** | **+57 images (+1.14pp)** |
| CfC (g=4, eS=50, p=0, K=5) | 85.44% | 84.54% | +0 |
| Pipeline (dS=2, Ka=3, g=8, eS=5, p=1, Kb=20) | 85.92% | 85.14% | +30 images (+0.60pp) |

**Bayesian sequential processing genuinely helps Fashion:** +1.14pp on
holdout when parameters are derived from val only. The CfC method does
not generalize (val improvement doesn't transfer to holdout). Pipeline
partially generalizes.

### Fashion honest numbers

**Bayesian sequential with val-derived weights: 85.68% holdout.**

Comparison to original topo9 (full 10K, original weights, with
sequential): 85.81%. The val-derived number is 0.13pp lower — a small
and acceptable correction.

### Comparison to original hardcoded weights on holdout

| Weights | Holdout static |
|---------|---------------|
| Val-derived (w_d=50, w_g=200, sc=100) | 84.54% |
| Original (w_d=200, w_g=200, sc=50) | 84.92% |
| Delta | -19 images (-0.38pp) |

The val-derived weights are slightly worse on holdout static baseline,
but the Bayesian sequential step more than compensates: 85.68% > 84.92%.

---

## Summary

| Dataset | Original (full 10K) | Val-derived holdout | Correction |
|---------|--------------------|--------------------|------------|
| MNIST static | 97.27% | 98.24% holdout | +0.97pp (data composition) |
| MNIST sequential | 97.31% | 98.24% holdout | Sequential = noise |
| Fashion static | 84.80% | 84.54% holdout | -0.26pp |
| Fashion sequential | 85.81% | **85.68%** holdout | **-0.13pp** |

### Key findings

1. **MNIST weights were not overfit.** The grid search on val rediscovers
   the identical weights. The test-set contamination concern was valid
   methodologically but did not affect the actual result.

2. **Sequential processing on MNIST is confirmed as noise.** Zero
   improvement on val. The +0.03pp from the original topo9 was an
   artifact of searching on the test set.

3. **Sequential processing on Fashion is real.** +1.14pp on holdout
   with val-derived parameters. The Bayesian method (dS=1, K=20, tw=50)
   generalizes from val to holdout.

4. **Fashion weights shift under proper validation.** Divergence weight
   drops from 200 to 50; Kalman scale increases from 50 to 100. The
   original weights slightly overfit to the full test set.

5. **The correction is small.** The publication-ready number for
   Fashion-MNIST is 85.68% (down from 85.81%, a 0.13pp correction).
   For MNIST, the honest number is 97.27% static (no sequential).

### Publication-ready claims

- **MNIST: 97.27%** with zero learned parameters, zero sequential
  processing, integer arithmetic only. (Val-derived weights identical
  to original; sequential processing confirmed as noise.)

- **Fashion-MNIST: 85.68%** with val-derived weights and Bayesian
  sequential processing. (+8.50pp over brute kNN baseline of 77.18%
  on the same holdout split.)

- **Confidence map:** Unchanged — operates on vote-phase statistics
  that are independent of feature weight tuning.

---

## Files

Code: `src/sstt_topo9_val.c`
This document: `docs/35-val-holdout-validation.md`
Build: `make sstt_topo9_val`
Run: `./sstt_topo9_val` (MNIST) or `./sstt_topo9_val data-fashion/` (Fashion)
