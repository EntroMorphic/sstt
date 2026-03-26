# Method 4 Parameter Tuning on Real MNIST

## Problem

Initial integration of Method 4 (Skip kNN if Confident) with synthetic parameters:
- **T_MAD = 40, T_DIV = -20** (derived from synthetic MNIST in document 38)

**Result: Accuracy degradation from 97.27% → 96.91%** (+36 errors on holdout)

- Expected from synthetic: +0.6-0.8pp gain
- Actual on real MNIST: **-0.36pp loss**

## Root Cause

Synthetic parameters do not generalize to real MNIST:
- The distributions of `mad` (candidate pool MAD) and `divneg_test` on real MNIST differ from synthetic data
- Synthetic data was generated with specific statistical properties that don't match real image statistics
- Threshold values were optimized for synthetic distribution, not real data

## Solution: Real Data Tuning Framework

Created a tuning infrastructure in `src/sstt_topo9.c`:

### New Functions
1. **run_static_tuned()**: Version of `run_static()` with parameterized (T_MAD, T_DIV)
2. **tune_method4_split()**: Tests a parameter pair on a given image range

### Tuning Strategy
- **Validation split**: Images 0-4999 (5K from test set)
- **Holdout split**: Images 5000-9999 (5K from test set)
- **Grid search**: 5×5 = 25 parameter combinations
  - T_MAD ∈ {10, 20, 30, 40, 50}
  - T_DIV ∈ {-50, -30, -20, -10, 0}

### Process
1. Test all 25 combinations on validation split
2. Record validation accuracy for each
3. Select best parameter set based on validation
4. Report holdout accuracy with best parameters

## Current Status: **TUNING COMPLETE**

Tuning computation finished. Tested 25 parameter combinations on MNIST validation/holdout split.

## Tuning Results

### Full Grid Search (5×5 = 25 combinations)

**All parameter combinations reduce accuracy on the validation split (0-4999):**

| T_MAD | Best Val Acc | Best Hold Acc | Est. Full-Test |
|-------|------------|-------------|-----------------|
| 10 | 95.92% (204 E) | 98.04% (98 E) | **96.98%** |
| 20-50 | 95.94-95.98% (202-201 E) | 97.84% (108 E) | **96.90%** |
| **Baseline** | **96.30%** (185 E) | **98.24%** (88 E) | **97.27%** |

### Key Finding

**Method 4 reduces overall accuracy with ALL tested parameters.**
- Best estimate with T_MAD=10: 96.98% (vs baseline 97.27%)
- **Net loss: -0.29pp on estimated full test set**

The problem: Synthetic parameters derived from synthetic MNIST do NOT generalize to real MNIST. Real data has different distribution of `mad` and `divneg_test` values.

### Why It Failed

1. Mode C classification in synthetic data differs from real MNIST
2. Threshold values optimized for synthetic distribution don't work on real data
3. The gating condition helps some images (harder holdout subset) but hurts more (easier val subset)
4. Net effect across full test set is negative

## Decision: **REVERT METHOD 4**

Method 4 integration has been **disabled in `src/sstt_topo9.c`**:
- Synthetic parameters do not improve real MNIST
- Tuning showed no parameter combination beats the baseline
- Better to keep the simple, well-understood 97.27% baseline

### What This Means

- ✅ Commits 4dcc645 and c5d1967 remain in history (for documentation)
- ❌ Method 4 gating is **disabled** in current code (T_MAD=0 forces fallback to standard kNN)
- 📝 Problem and solution are documented for future reference

## Alternative Approaches

If Mode C error reduction is still desired:
- **Method 1: Kalman-Adaptive k** — Also showed +0.86pp on synthetic, needs real data tuning
- **Method 2: Score-weighted vote** — Simpler, lower compute cost, showed +0.18pp on synthetic
- **Method 3: Divergence-filtered** — More complex, showed +0.42pp on synthetic
- **Accept baseline** — 97.27% is solid, Mode C errors (33.8%) are manageable cost

## Files Modified
- `src/sstt_topo9.c`:
  - Added tuning framework (run_static_tuned, tune_method4_split)
  - Added grid search in main() for documentation
  - Disabled Method 4 gating in production code (T_MAD=0)

## Lessons Learned

1. **Synthetic ≠ Real**: Parameter tuning on synthetic data doesn't guarantee real-world improvement
2. **Cross-validation is critical**: Need proper val/holdout splits when tuning
3. **Publication caveat**: Methods that work mathematically may not work empirically due to distribution shifts
