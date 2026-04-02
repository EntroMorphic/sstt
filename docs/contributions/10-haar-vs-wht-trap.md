# Contribution 10: Haar ≠ WHT Non-Orthogonality Trap

## The Bug

The initial "WHT" implementation was actually a Haar wavelet pyramid
decomposition. It returned 11.35% accuracy for prototypes and 12.70%
for brute 1-NN — essentially random. After centering (subtracting global
mean), prototypes improved to 51.78% but 1-NN was still broken at 53.20%.

The root cause: the Haar wavelet pyramid is **not orthogonal**, so dot
products are not preserved.

## The Difference

**Haar wavelet pyramid** (what was implemented first):
```
Stage 1: process elements [0..n-1] → sums in [0..n/2-1], diffs in [n/2..n-1]
Stage 2: process elements [0..n/2-1] ONLY → recurse on first half
Stage 3: process elements [0..n/4-1] ONLY → recurse on first quarter
...
```

**Walsh-Hadamard Transform** (what should have been implemented):
```
Stage 1: butterflies on ALL n elements, stride n/2
Stage 2: butterflies on ALL n elements, stride n/4
Stage 3: butterflies on ALL n elements, stride n/8
...
```

The critical distinction: Haar recurses on the **first half only** at
each level. WHT processes **all elements** at every stage.

## Mathematical Proof of Non-Orthogonality

For N=4, the Haar pyramid matrix is:

```
W = [1  1  1  1]     (DC: sum of all)
    [1  1 -1 -1]     (low-freq diff)
    [1 -1  0  0]     (high-freq pair 1)
    [0  0  1 -1]     (high-freq pair 2)
```

Check W^T × W:
```
Column 1 · Column 2 = 1×1 + 1×1 + 1×(-1) + 1×(-1) = 0   ✓
Column 1 · Column 3 = 1×1 + 1×(-1) + 1×0 + 1×0 = 0       ✓
Column 2 · Column 3 = 1×1 + 1×(-1) + (-1)×0 + (-1)×0 = 0 ✓
BUT:
Column 1 · Column 1 = 1+1+1+1 = 4
Column 3 · Column 3 = 1+1+0+0 = 2    ← NOT the same!
```

W^T × W = diag(4, 4, 2, 2) ≠ c × I

The columns have **different norms**. This means the transform distorts
distances: elements encoded by short-norm columns are stretched relative
to elements encoded by long-norm columns.

In contrast, the WHT matrix has all columns with equal norm √N, giving
H^T × H = N × I (true orthogonality).

## The Centering Red Herring

Before discovering the Haar/WHT confusion, we tried centering (subtracting
the global mean WHT) to fix the DC dominance problem. This helped
prototypes (51.78%) but destroyed 1-NN (53.20%) because:

```
<q-m, c-m> = <q,c> - <q,m> - <m,c> + ||m||²
```

The term `<m,c>` varies across candidates, changing their ranking. So
centering is not just unnecessary with the true WHT — it's actively
harmful for nearest-neighbor search.

## The Fix

Replace the Haar pyramid with the true WHT butterfly:

```c
/* Haar (wrong): */
for (int size = n; size >= 2; size /= 2)
    process elements [0..size-1]          /* shrinking window */

/* WHT (correct): */
for (int half = n/2; half >= 1; half /= 2)
    for (int i = 0; i < n; i += 2*half)  /* ALL elements, every stage */
```

After the fix: brute WHT 1-NN = 95.87% (exact match with pixel 1-NN).

## Why This Matters

This trap is easy to fall into because:

1. Both transforms use only add/subtract butterflies
2. Both produce frequency-ordered coefficients (low → high)
3. Both are invertible
4. The Haar pyramid is often casually called a "wavelet transform" or
   even a "Walsh-Hadamard transform" in informal usage
5. For prototype classifiers (centroid matching), the non-orthogonality
   partially cancels out, masking the problem

The key diagnostic: **if your "WHT" brute 1-NN doesn't match pixel 1-NN
accuracy, your transform isn't orthogonal.**

## Files

- `sstt_v2.c` lines 769-784: correct WHT implementation
- Original Haar code (removed): `haar_1d_row`, `haar_1d_col`,
  `haar_2d_one_level`, `haar_2d_full` — all replaced
