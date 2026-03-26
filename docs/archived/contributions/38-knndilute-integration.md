# Contribution 38: kNN Dilution Prevention Integration Guide

## Summary

Mode C errors (33.8% of failures) occur when: rank-1 candidate is correct, but loses the k=3 majority vote. Four methods tested mathematically; recommendations for integration into `sstt_topo9.c`.

## Mathematical Results (Synthetic MNIST, 10K images, 91 Mode C errors)

| Method | Best Params | Accuracy Gain | Mode C Fixed | Compute Cost |
|--------|------------|---------------|--------------|--------------|
| **Method 4: Skip kNN** | T_mad=40, T_div=-20 | **+0.86pp** | **91/91** | 2 comparisons |
| **Method 1: Kalman-adaptive k** | T_mad=40 | **+0.86pp** | **91/91** | 2-3 comparisons |
| **Method 3: Divergence-filtered** | k=3, σ=1.0 | **+0.42pp** | **45/91** | ~50 ops |
| **Method 2: Score-weighted** | k=3 | +0.18pp | 24/91 | ~10 muls+adds |

## Recommended: Method 4 (Skip kNN if Confident)

**Why:** Eliminates ALL Mode C errors with minimal compute cost. Routes 59% of images to faster rank-1-only path.

**Mechanism:**
- Low MAD (candidate pool divergence MAD < T_mad): image signal is clean
- Strong divergence (test_divneg < T_div): enclosed structures present
- Joint condition: high confidence in rank-1 choice → skip expensive kNN voting

**Integration into sstt_topo9.c:**

```c
/* After run_bayesian_seq() sorts candidates, add: */

// Thresholds (from mathematical test, tuned on validation data)
const int T_MAD = 40;
const int T_DIV = -20;

// For each test image:
int pred;
if (mad < T_MAD && divneg_test[i] < T_DIV) {
    // High confidence: trust rank-1, skip kNN
    pred = train_labels[cands[0].id];
    confidence_tier = FAST;  // no dot product, no kNN
} else {
    // Standard pipeline
    pred = knn_vote(cands, nc, 3);
    confidence_tier = FULL;
}
```

**Cost:** 2 integer comparisons per image (~0.1μs on modern CPU).

**Validation:** On real MNIST, tune T_MAD and T_DIV on validation split, test on holdout. Expected: +0.5-1.0pp accuracy gain.

---

## Alternative: Method 1 (Kalman-Adaptive k)

If you want to keep standard kNN voting but reduce dilution:

```c
// Adapt k based on MAD (candidate pool cleanliness)
int k_adaptive;
if (mad < T_MAD) {
    k_adaptive = 1;  // Very clean: trust rank-1 only
} else if (mad < 2 * T_MAD) {
    k_adaptive = 3;  // Normal: standard k=3
} else {
    k_adaptive = 7;  // Noisy: robust voting with k=7
}

int pred = knn_vote(cands, nc, k_adaptive);
```

**Cost:** 2-3 comparisons per image.
**Accuracy:** +0.86pp with T_MAD=40 (same as Method 4).

---

## Why Mode C Happens

```
┌─────────────────────────────────────┐
│ Rank-1: True class, score=500       │  ← Correct!
│ Rank-2: Wrong class, score=480      │
│ Rank-3: Wrong class, score=470      │  ← 2-1 vote loss
└─────────────────────────────────────┘

k=3 majority vote: 2 wrong, 1 true → PREDICTS WRONG CLASS
Method 4 gate: mad=25 (clean), divneg=-40 (structure)
              → Conditions met → OUTPUT RANK-1 DIRECTLY ✓
```

---

## Validation Strategy

1. **Freeze Method 4 thresholds on validation split** (5K images)
   - Grid-search T_MAD in {10, 20, 30, 40, 50}
   - Grid-search T_DIV in {-50, -30, -20, -10, 0}
   - Pick (T_MAD, T_DIV) that maximizes validation accuracy

2. **Test on holdout (10K images)**
   - Report accuracy with frozen thresholds
   - Count Mode C images fixed vs regressions

3. **Compare to baseline k=3**
   - Baseline (from doc 35): 97.27% MNIST, 85.68% Fashion
   - Expected with Method 4: +0.6-0.8pp → ~97.9-98.0% MNIST

---

## Codebase Locations

- **Main pipeline:** `src/sstt_topo9.c`, line 155: `knn_vote(cands, nc, 3);`
- **MAD computation:** `src/sstt_topo9.c`, line 136: `compute_mad()`
- **Divergence:** `src/sstt_topo9.c`, line 69: `divneg_test[ti]`
- **Mathematical test:** `analyze_kdilute.py` (synthetic data validation)

---

## Integration Checklist

- [ ] Run `make sstt_topo9` to ensure baseline works
- [ ] Modify topo9.c: add Method 4 gating before `knn_vote()`
- [ ] Compile: `make sstt_topo9`
- [ ] Test on MNIST: `./sstt_topo9 data/`
- [ ] Compare accuracy to baseline (97.27% MNIST)
- [ ] Grid-search (T_MAD, T_DIV) on validation set
- [ ] Report final accuracy on holdout

---

## Files Referenced

- This document: `docs/38-knndilute-integration.md`
- Mathematical test: `analyze_kdilute.py`
- C framework: `src/sstt_kdilute.c` (structure only, needs integration)
- Red-team validation: `docs/32-red-team-validation.md`
