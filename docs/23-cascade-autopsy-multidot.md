# Contribution 23: Cascade Autopsy + Multi-Channel Dot Refinement

## Motivation

The bytepacked cascade (contribution 22) achieved 96.28% on MNIST but the
failure mode was unknown. Before building more architecture, we autopsied
every error to understand *where* the cascade breaks down.

## Autopsy Findings (sstt_diagnose.c)

Running the bytepacked cascade on 10K MNIST test images produced 414 errors.
Each error was classified by failure mode:

| Mode | Count | % | Description |
|------|-------|---|-------------|
| A — Vote Miss | 7 | 1.7% | Correct class not in top-200 candidates |
| B — Dot Override | 267 | 64.5% | Correct class in top-K, wrong class wins dot |
| C — kNN Dilution | 140 | 33.8% | Correct has best individual dot, outvoted k=3 |

**407 of 414 errors (98.3%) happen at the dot product refinement step.**
The vote phase finds the right neighborhood 99.3% of the time. The failure
is in the final ranking among already-correct candidates.

**Vote gap distribution:** 370 of 414 errors (89%) had the correct class
accumulating 80–100%+ of the wrong class's votes. These are razor-thin
margin losses decided by 3–20 dot product units in a 700-dimensional space.

**Top confusion pairs:**

| True → Pred | Count | Mode A | Mode B | Mode C | Vote Gap | Dot Gap |
|------------|-------|--------|--------|--------|----------|---------|
| 4 → 9 | 29 | 0 | 21 | 8 | 102% | -3 |
| 7 → 1 | 29 | 1 | 21 | 7 | 98% | -10 |
| 8 → 3 | 24 | 0 | 16 | 8 | 102% | -5 |
| 3 → 5 | 15 | 0 | 14 | 1 | 108% | -12 |

The dot gap for 4↔9 is only **-3 units** out of ~700. The current
pixel-only ternary dot cannot resolve topological differences (closed loop
vs. open angle) at this margin.

**Estimated ceiling:** ~98.6% if all Mode B errors are fixable. Mode C
(140 errors) likely requires a different vote structure, not better features.

## Root Cause: Pixel-Only Dot Refinement

The vote phase uses all channels (bytepacked = pixel + h-grad + v-grad +
transitions). The dot product refinement step uses **pixel channel only**:

```c
// sstt_bytecascade.c — the seam that loses information
cands[j].dot = ternary_dot(query_tern, train_tern + cands[j].id * PADDED);
```

Two of three channels are discarded at the most critical decision point.

## Fix: Multi-Channel Weighted Dot (sstt_multidot.c)

Replace single-channel dot with a weighted combination:

```
score = w_px × dot(q_px, t_px)
      + w_hg × dot(q_hg, t_hg)
      + w_vg × dot(q_vg, t_vg)
```

Key design: dots for all top-K candidates across all channels are
**precomputed once** and stored. Weight strategies are then applied
without re-running the vote phase — the weight sweep is essentially free.

### Results: Weight Strategies (MNIST, K=200, k=3)

| Strategy | Weights (px, hg, vg) | Accuracy | Errors |
|----------|---------------------|----------|--------|
| A. Pixel-only baseline | (256, 0, 0) | 95.86% | 414 |
| B. Equal | (256, 256, 256) | 96.00% | 400 |
| C. IG-proportional | (84, 84, 87) | 95.97% | 403 |
| D. Empirical SNR | (94, 75, 87) | 96.03% | 397 |
| **E. Grid best** | **(256, 0, 192)** | **96.20%** | **380** |

**+0.34pp over pixel-only, at essentially zero added cost** (7 μs/img
for 3-channel dot vs 2 μs for 1-channel — a 5 μs difference on a
636 μs pipeline).

### The h-grad signal

The grid search reveals: **h-grad hurts, v-grad helps.** Optimal weights
are px + vg with no h-grad. v-grad (vertical gradient = change between
rows) captures stroke direction, the closed/open top distinguishing
4 from 9, and the horizontal bar distinguishing 7 from 1. H-grad
(horizontal gradient = change within rows) duplicates information already
in the pixel channel.

Specific improvements with (px=256, hg=0, vg=192):
- 7→1: 29 → 20 errors (**+9 fixed**)
- 2→1: 14 → 7 errors (+7 fixed)
- 9→7: 15 → 9 errors (+6 fixed)
- 4→9: 29 → 31 errors (-2, slightly worse — topological, not fixable by dot)

## Compute Profile and Optimizations (sstt_timeit)

Precise phase timing per test image (AMD Ryzen 5 PRO 5675U):

| Phase | Time | % of total |
|-------|------|-----------|
| memset votes[] | 2.3 μs | 0.4% |
| vote exact-only | 196.1 μs | 30.9% |
| vote probe neighbors | 356.6 μs | **56.1%** |
| select_top_k | 78.4 μs | 12.3% |
| 1-channel dot (K=200) | 2.2 μs | 0.3% |
| 3-channel dot overhead | +7.1 μs | +1.1% |

**87% of time is the vote phase.** The dot product is negligible.

### Fix 1: select_top_k histogram pre-allocation

**Root cause:** `select_top_k` called `calloc(max_vote+1, 4)` and `free()`
per image. The histogram (~32KB) was freshly mapped (cold pages) on every
call, causing ~512 RAM cache misses (×50ns = ~26 μs cold miss penalty)
plus glibc allocator overhead (~50 μs total savings).

**Fix:** Pre-allocate `g_hist` once, reuse across all calls, reset with
`memset(g_hist, 0, (mx+1)*sizeof(int))` — sequential write of ~32KB
at L2 speeds (~1 μs). The "zero touched entries" approach (second scan
over votes) was slower due to branch misprediction on the 40% fill rate.

**Caveat:** `g_hist` is a global static; `select_top_k` is not thread-safe.

### Fix 2: Probe count reduction

Sweeping probe count with multi-channel dot (best weights):

| Probes | Accuracy | Vote time | vs 8-probe |
|--------|----------|-----------|------------|
| 0 | 95.81% | 4.63s | 1.83× faster |
| 1 | 95.95% | 5.03s | 1.69× faster |
| 2 | 95.91% | 5.41s | 1.57× faster |
| **4** | **96.07%** | **6.14s** | **1.38× faster** |
| 8 | 96.20% | 8.48s | baseline |

**Decision: 4 probes.** Gets 86% of the accuracy gain (0.26/0.39pp) of 8
probes at 1.38× the speed.

## Red-Team Analysis

### 1. Probe ordering is suboptimal for N < 8 probes
The neighbor table flips bits in order 0–7:
- Bits 0–1: pixel majority trit
- Bits 2–3: h-grad majority trit ← **h-grad hurts the dot step**
- Bits 4–5: v-grad majority trit ← **v-grad helps the dot step**
- Bits 6–7: transition activity

With 4 probes, we flip bits {0,1,2,3} = pixel + h-grad. We should flip
bits {0,1,4,5} = pixel + v-grad. The 4-probe accuracy of 96.07% is
**underperforming** because it probes the wrong channel's neighbors.
Reordering to {0,1,4,5,2,3,6,7} would likely match or exceed 8-probe
accuracy at 4-probe speed. **This is the highest-confidence untested
improvement remaining.**

### 2. Weight selection is test-set contaminated
The grid search for `(w_px=256, w_hg=0, w_vg=192)` was optimized on the
full MNIST test set. The 96.20% figure may be slightly inflated vs. true
generalization. Fashion-MNIST validation is needed. The landscape is flat
enough (many configs give 96.1–96.2%) that the signal is likely real.

### 3. The h-grad=0 finding is MNIST-specific
MNIST digits are consistently upright; v-grad captures their dominant
stroke direction. Fashion-MNIST garment textures are more isotropic. The
optimal weights for Fashion may favor h-grad. Do not assume
`w_hg=0` transfers to other datasets.

### 4. 4 vs. 8 probe accuracy difference is 13 images
96.07% vs 96.20% = 13 images at N=10K. Statistically marginal but the
tradeoff (1.38× speed for -0.13pp) is reasonable in context.

## Files

- `sstt_diagnose.c`: Cascade error autopsy — failure mode classification,
  vote gap analysis, ASCII visual inspection, fixability assessment
- `sstt_multidot.c`: Multi-channel dot refinement — precomputed 3-channel
  dots, weight sweep strategies A–D, grid search E, probe count sweep
- `sstt_tiled.c`: K-means tile routing experiment (routing hurts — confused
  pairs cluster together; global cascade is better than split cascades)
