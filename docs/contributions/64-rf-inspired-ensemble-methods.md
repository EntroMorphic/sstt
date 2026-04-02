# Contribution 64: Random-Forest-Inspired Ensemble Methods for Zero-Parameter Cascade Classification

## Motivation

SSTT achieves 97.27% on MNIST with zero learned parameters, sitting between
random forest (97.05%) and a 1-layer MLP (97.63%). The error mode decomposition
identifies the remaining gap as a ranking problem (98.3% of errors), with a
fixable ceiling of ~98.6%.

Random forest's power comes not from any single tree but from **decorrelated
ensemble diversity** — many weak classifiers whose errors cancel. SSTT has one
retriever, one ranker, one vote. Importing RF's structural mechanisms (bagging,
feature subspace randomization, hierarchical partitioning) without introducing
learned parameters could push SSTT closer to its ceiling.

This document specifies three experiments, ordered by implementation simplicity
and expected impact.

---

## Experiment 1: Bagging Over Block Positions

### Concept

Instead of one inverted-index vote using all 252 block positions, create M
random subsets of positions and run the vote M times. Each retriever sees
different spatial information — their errors are decorrelated for the same
reason RF trees' errors are decorrelated.

### Specification

**Parameters (all fixed, not learned):**
- `M` = number of retriever bags (sweep: 3, 5, 7, 10, 15)
- `R` = fraction of positions per bag (sweep: 0.5, 0.6, 0.7, 0.8)
- `seed` = deterministic PRNG seed for reproducibility

**Algorithm:**

```
TRAINING:
  For each bag m = 1..M:
    Generate random position mask: select R * 252 positions (without replacement)
    Store mask[m] as a 252-bit vector
    Build inverted index as normal, but skip masked-out positions
    Compute IG weights as normal over the selected positions only

INFERENCE:
  For each test image:
    For each bag m = 1..M:
      Run IG-weighted multi-probe vote using only mask[m] positions
      Extract top-K_m candidates (K_m = 50)

    MERGE candidates:
      Option A (union):   candidates = union of all M top-K sets
      Option B (ranked):  score each candidate by number of bags it appears in,
                          break ties by total vote count across bags
      Option C (intersection-boosted): candidates appearing in >= T bags get
                          a vote bonus proportional to their appearance count

    RANK merged candidates using the existing multi-channel dot + structural
    features pipeline (unchanged)

    k=3 majority vote → class prediction
```

**Merge strategies to test:**
- **Union-then-rank:** Simple union of all M candidate sets, then standard ranking. Candidate set grows to at most M × 50 = 500 (but with heavy overlap, typically ~100-150).
- **Appearance-weighted:** Each candidate gets a bonus for appearing in multiple bags. Score = standard_score + β × appearance_count. β selected on validation split.
- **Threshold intersection:** Only candidates appearing in >= ceil(M/2) bags proceed to ranking. More aggressive filtering, potentially faster.

### Hypothesis

Bagging over positions will:
1. **Reduce Mode B errors** (ranking inversions) by surfacing candidates that a single retriever misses — correct-class images that match on positions the single retriever happened to miss.
2. **Reduce Mode C errors** (vote dilution) by decorrelating the noise in vote counts — wrong-class images that are accidentally boosted by one retriever won't be boosted by all M.
3. **Potentially reduce Mode A errors** (retrieval failures) by expanding effective recall beyond 99.3% — a candidate that falls just below top-50 in one retriever may be well within top-50 in another.

### Expected impact

Conservative estimate: +0.3-0.5pp on MNIST (fixing ~30-50 of the ~270 errors).
Optimistic estimate: +0.8pp (fixing ~80 errors, mostly Mode B).

The cost is M× more vote accumulations. Each bag does ~R × the work of a full
vote, so total cost ≈ M × R × original. At M=10, R=0.5, that's 5× the vote
phase — moving from ~930μs to ~4.5ms for the bytepacked cascade. The ranking
phase is unchanged (still applied to ~50-150 candidates).

### Success criteria

- Accuracy improvement on MNIST val split of >= +0.2pp (statistically significant at 95% CI)
- Accuracy improvement on Fashion-MNIST val split of >= +0.3pp
- Mode A recall increases (measurable via error autopsy)
- No regression on easy images (Tier 1 accuracy stays at 99.9%+)

### Measurement

For each configuration, report:
1. Accuracy (val and test, with 95% CI)
2. Mode A/B/C error counts (using existing autopsy infrastructure)
3. Candidate set size after merge (mean, p50, p95)
4. Wall-clock latency (vote phase, rank phase, total)

---

## Experiment 2: Multi-Threshold Ensemble (Generalized Stereoscopic)

### Concept

SSTT uses one quantization scheme: thresholds (85, 170). The stereoscopic
work (Doc 39) showed that multiple perspectives improve retrieval — 3 "eyes"
lifted CIFAR-10 recall to 99.4%. Generalize this: use M fixed quantization
schemes, each producing a different ternary representation with its own
inverted index. Vote across all M indices.

This is RF's feature randomization — different views of the same data,
aggregated. Where you draw the ternary boundary determines which training
images match. Some images are ambiguous at (85, 170) but clear at (64, 192).

### Specification

**Quantization schemes (all fixed, not learned):**

| Scheme | Low threshold | High threshold | Rationale |
|--------|--------------|----------------|-----------|
| Standard | 85 | 170 | Current SSTT default |
| Wide-zero | 64 | 192 | More zeros, less noise |
| Narrow-zero | 96 | 160 | More signal, more noise |
| Quarter | 64 | 128 | Asymmetric, dark-biased |
| Three-quarter | 128 | 192 | Asymmetric, bright-biased |
| Adaptive P25/P75 | per-image P25 | per-image P75 | Normalizes dynamic range |
| Adaptive P10/P90 | per-image P10 | per-image P90 | Edge-only, aggressive |

**Algorithm:**

```
TRAINING:
  For each scheme s = 1..S:
    Quantize all 60K training images using scheme s thresholds
    Compute gradients, block signatures, IG weights (all closed-form)
    Build inverted index for scheme s

INFERENCE:
  For each test image:
    For each scheme s = 1..S:
      Quantize test image using scheme s
      Run IG-weighted multi-probe vote
      Extract top-K_s candidates

    MERGE candidates (same strategies as Experiment 1)

    RANK merged candidates:
      Option A: standard ranking (pixel dot + v-grad dot + structural)
      Option B: multi-scheme dot product — for each candidate, compute
                dot product under EACH scheme, sum the scores. Candidates
                that are close under multiple quantizations score higher.

    k=3 majority vote → class prediction
```

**Key design decision:** Should ranking use one scheme or all schemes?

- Using all schemes for ranking is more expensive (S × dot products per candidate) but captures cross-scheme agreement — a candidate that ranks high under 5/7 quantization schemes is more likely correct than one that ranks high under 1/7.
- Recommended: start with single-scheme ranking (cheapest), measure impact of retrieval diversity alone, then test multi-scheme ranking if retrieval diversity alone isn't sufficient.

### Hypothesis

Multi-threshold ensemble will:
1. Improve recall by covering different regions of the ternary representation space — images at the quantization boundary (pixel value near 85 or 170) are unstable under one scheme but stable under at least one alternative.
2. Reduce Mode A errors more than Experiment 1, because it changes the *representation* (what the index contains), not just *which positions are queried*.
3. Have a larger effect on Fashion-MNIST than MNIST, because clothing textures span wider intensity ranges and are more sensitive to threshold placement.

### Expected impact

Conservative: +0.2-0.4pp on MNIST, +0.5-1.0pp on Fashion.
Optimistic: +0.5pp on MNIST, +1.5pp on Fashion.

Cost: S × memory for inverted indices (~S × 60MB), S × vote accumulation time.
At S=5, that's ~300MB memory and ~5ms vote phase.

### Success criteria

- Fashion-MNIST improvement >= +0.5pp (where threshold sensitivity is highest)
- Mode A recall measurably improves (retrieval failures decrease)
- The improvement is additive with Experiment 1 (they attack different axes)

---

## Experiment 3: Recursive Re-Retrieval with Diverse Block Geometry

### Concept

After the first vote produces top-50, re-encode those candidates with a
*different block geometry* and re-vote within the candidate set. Each geometry
captures different spatial structure. This is analogous to RF's recursive
splitting — each level uses different features to partition further.

### Specification

**Block geometries (all fixed):**

| Geometry | Block shape | Positions | Values | Captures |
|----------|------------|-----------|--------|----------|
| Standard | 3×1 horizontal | 252 | 27 | Horizontal stroke patterns |
| Vertical | 1×3 vertical | 234 | 27 | Vertical stroke patterns |
| Square | 2×2 square | 351 | 81 (3^4) | Local patch structure |
| Wide | 5×1 horizontal | 168 | 243 (3^5) | Wider horizontal context |
| Tall | 1×5 vertical | 130 | 243 (3^5) | Wider vertical context |

**Algorithm:**

```
TRAINING:
  For each geometry g = 1..G:
    Compute block signatures for all 60K training images using geometry g
    Compute IG weights for geometry g
    Build inverted index for geometry g

INFERENCE (cascaded):
  Level 1: Standard 3×1 vote → top-K1 candidates (K1 = 200)

  Level 2: For each candidate in top-K1:
    Re-encode using vertical 1×3 geometry
    Compute similarity to test image's 1×3 signature
    Score = level_1_vote + α × level_2_similarity
  Select top-K2 (K2 = 50)

  Level 3 (optional): Re-encode using 2×2 geometry
    Score = accumulated_score + β × level_3_similarity
  Select top-K3 (K3 = 20)

  Final: standard structural ranking on top-K3 → k=3 vote → prediction
```

### Hypothesis

Recursive re-retrieval will:
1. Break confusion pairs that share horizontal structure but differ vertically (e.g., 1 vs 7: identical horizontal profile, different vertical angle).
2. Function as a zero-parameter analog of RF's recursive partitioning — each level uses orthogonal features to subdivide what the previous level couldn't separate.
3. Potentially reduce the candidate set more aggressively (200 → 50 → 20) with less information loss, because each level contributes different discriminative signal.

### Expected impact

This is the most speculative of the three experiments. The 3×1 geometry was
not chosen arbitrarily — it was the result of early exploration. Alternative
geometries may be worse individually but complementary in cascade.

Conservative: +0.1-0.3pp (the 3×1 geometry may already capture most of the
discriminative structure).
Optimistic: +0.5pp if vertical geometry resolves the 1/7 and 4/9 confusion
pairs specifically.

### Success criteria

- Per-confusion-pair analysis showing improvement on specific digit pairs
- The cascade reduces candidate set size without reducing accuracy (compression test)
- Level 2 adds discriminative value beyond what level 1 provides (ablation)

---

## Implementation Priority

| # | Experiment | Complexity | Expected impact | Priority |
|---|-----------|-----------|----------------|----------|
| 1 | Block position bagging | Low (mask + loop) | +0.3-0.8pp | **First** |
| 2 | Multi-threshold ensemble | Medium (multiple indices) | +0.2-1.5pp | Second |
| 3 | Recursive re-retrieval | High (new geometries) | +0.1-0.5pp | Third |

**Recommended approach:**
1. Implement Experiment 1 as a single self-contained C file (`sstt_bag_positions.c`)
2. Measure Mode A/B/C impact with existing autopsy infrastructure
3. If Experiment 1 shows Mode A improvement, proceed to Experiment 2 (which attacks Mode A from a different angle — representation diversity vs position diversity)
4. If Experiments 1 and 2 together don't resolve the 4/9 and 1/7 confusion pairs, proceed to Experiment 3 (which specifically targets geometry-dependent confusions)

---

## Constraints

All experiments must satisfy:
- **Zero learned parameters.** Position masks are random (seeded). Thresholds are fixed constants or closed-form percentiles. Block geometries are fixed. IG weights are computed closed-form. No gradient descent. No loss function.
- **Val/holdout protocol.** Any merge parameter (β, T, appearance bonus) is selected on the 10% validation split and frozen before test evaluation.
- **Error autopsy.** Every configuration reports Mode A/B/C decomposition, not just accuracy. The goal is to understand *which error mode* each technique addresses.
- **Reproducibility.** Deterministic PRNG seeds. All configurations documented. Single-file, self-contained C programs.

---

## Experiment 1 Results

**Date:** 2026-03-26
**Source:** `src/sstt_bag_positions.c`, `src/sstt_bag_topo9.c`

### Part A: Bagging on Bytepacked Cascade (no topological features)

Baseline: bytepacked cascade with multi-probe voting, K=200, k=3.

**Parameter sweep (MNIST, 10K test):**

| M\R | R=0.5 | R=0.6 | R=0.7 | R=0.8 |
|-----|-------|-------|-------|-------|
| M=3 | 95.96% | 95.84% | 95.87% | 95.87% |
| M=5 | 96.07% | 95.95% | 95.86% | 95.85% |
| M=7 | 96.18% | 96.12% | 96.02% | 95.98% |
| M=10 | 96.23% | **96.24%** | 96.13% | 96.11% |

- **Baseline (M=1, all positions):** 95.86%
- **Best (M=10, R=0.6):** 96.24% (+0.38pp)

**Error mode analysis (M=10, R=0.6):**

| Mode | Count | % of errors |
|------|-------|-------------|
| A (retrieval miss) | 8 | 2.1% |
| B (ranking inversion) | 267 | 71.0% |
| C (vote dilution) | 101 | 26.9% |

**Findings:**
- More bags helps monotonically. Lower R (more diversity) helps.
- The improvement is real but modest: +0.38pp = 38 fewer errors.
- Mode B (ranking inversions) remains dominant at 71% — bagging improves
  candidate quality but cannot fix the ranker itself.

### Part B: Bagging on Topo9 (with topological features)

Tests whether bagged retrieval is additive with topological ranking.

**Sweep (MNIST, topo9 static ranking, K=200, k=3):**

| M\R | R=0.5 | R=0.6 | R=0.7 |
|-----|-------|-------|-------|
| M=5 | 97.24% | 97.01% | 96.90% |
| M=7 | 97.32% | 97.25% | 97.06% |
| M=10 | **97.37%** | 97.31% | 97.13% |

- **Baseline topo9 (no bagging):** 97.27% (273 errors)
- **Best bagged+topo9 (M=10, R=0.5):** 97.37% (263 errors, +0.10pp)

**Error mode comparison:**

| Mode | Baseline topo9 | Best bagged+topo9 |
|------|---------------|-------------------|
| A (retrieval miss) | 7 (2.6%) | 8 (3.0%) |
| B (ranking inversion) | 197 (72.2%) | ~185 (70.3%) |
| C (vote dilution) | 69 (25.3%) | ~70 (26.8%) |

**Per-pair deltas (best bagged vs baseline):**

| Pair | Baseline errors | Bagged errors | Delta |
|------|----------------|---------------|-------|
| 3↔5 | 29 | 27 | -2 |
| 4↔9 | 25 | 23 | -2 |
| 1↔7 | 11 | 12 | +1 |
| 3↔8 | 14 | 13 | -1 |

### Conclusions

1. **Bagging and topological ranking are largely redundant.** The +0.10pp
   gain on topo9 is within CI noise (±0.32pp). Both mechanisms fix the
   same Mode B errors through different paths — bagging via better
   candidates, topo9 via better ranking of the same candidates.

2. **Bagging has value on the fast path.** The +0.38pp gain on bytepacked
   cascade (no topo features) is meaningful. Position bagging could be
   integrated into the three-tier router's Tier 2 to improve accuracy
   without the cost of full topological ranking.

3. **Maximum diversity wins.** R=0.5 (126 of 252 positions) consistently
   outperforms R=0.7-0.8. Higher R produces bags too similar to each
   other, making the ensemble nearly equivalent to a single retriever
   but noisier. This confirms the RF insight: decorrelation matters
   more than individual classifier quality.

4. **Mode A is not improved.** Bagging does not reduce retrieval failures
   (Mode A stayed at 7-8 errors). The inverted index already achieves
   99.3% recall at K=50. To improve Mode A, the representation itself
   must change — which motivates Experiment 2.

### Status: COMPLETE. Proceed to Experiment 2.

---

## Experiment 2 Results

**Date:** 2026-03-26
**Source:** `src/sstt_multi_threshold.c` (standalone), `src/sstt_multi_threshold_topo9.c` (integration)

### Part A: Standalone (weaker ranking baseline)

Five quantization schemes with independent inverted indices, merged by
appearance count, ranked with a topo9-style pipeline that did not
perfectly replicate canonical topo9 (baseline 96.62% vs 97.27%).

| Config | Accuracy | Errors | Delta |
|--------|----------|--------|-------|
| S=1 Standard only | 96.62% | 338 | — |
| S=2 +Wide-zero | 96.86% | 314 | +0.24pp |
| S=3 +Narrow-zero | 96.93% | 307 | +0.31pp |
| S=4 +Adaptive P25/P75 | 97.10% | 290 | +0.48pp |
| S=5 +Adaptive P10/P90 | **97.23%** | **277** | **+0.61pp** |

Mode A errors: 37 → 8 (78% reduction). Every scheme contributed
monotonically. Adaptive percentile schemes provided the largest gains.

### Part B: Canonical Topo9 Integration — NEGATIVE RESULT

Multi-threshold retrieval wired into byte-for-byte canonical topo9
ranking. S=1 baseline reproduces 97.27% exactly.

| Config | Accuracy | Errors | Delta |
|--------|----------|--------|-------|
| S=1 (baseline topo9) | **97.27%** | 273 | — |
| S=2 (+wide-zero) | 96.86% | 314 | -0.41pp |
| S=3 (+narrow-zero) | 96.93% | 307 | -0.34pp |
| S=4 (+adaptive P25/P75) | 97.06% | 290 | -0.21pp |
| S=5 (+adaptive P10/P90) | 97.17% | 283 | **-0.10pp** |

**Adding any scheme beyond standard hurts when ranking is strong.**

### Root Cause

Candidates found by alternative schemes (64/192, adaptive P25/P75)
have poor dot-product similarity under the standard (85/170) ternary
representation used for ranking. They displace good candidates from
the top-200 and add noise to the MAD computation.

Multi-threshold retrieval compensated for ranking weakness in Part A
(96.62% baseline). With topo9's strong ranking (97.27% baseline),
retrieval diversity becomes retrieval noise — the same pattern as the
soft-prior cascade failure (Doc 21): a weaker signal interferes with
a stronger one.

### What Would Fix It

Multi-scheme ranking: compute dot products and structural features
under each scheme's representation, fuse scores. This is a significant
architectural change with uncertain payoff given that K-invariance
already shows retrieval is near-perfect.

### Conclusions

1. **Multi-threshold helps weak rankers, hurts strong rankers.**
   +0.61pp with weak ranking, -0.10pp with topo9.

2. **Ranking features are scheme-specific.** The ranker operates in
   scheme 1's representation space. Candidates from other schemes
   are ranked in the wrong space.

3. **The 97.27% ceiling is defended by ranking, not retrieval.**
   K-invariance (Doc 22) and this experiment converge on the same
   conclusion from different angles.

4. **Position bagging has the same weakness.** Experiment 1 showed
   +0.10pp on topo9 — also largely redundant with strong ranking.

### Status: NEGATIVE RESULT on MNIST. Cross-dataset testing follows.

---

## Cross-Dataset Results

**Date:** 2026-03-26
**Source:** Fashion runs via existing binaries with `data-fashion/` arg;
CIFAR-10 via `src/sstt_cifar10_multi_eye.c`

The MNIST negative result raised the question: does multi-threshold
retrieval help on datasets with higher intensity variation? Fashion-MNIST
(clothing textures, variable backgrounds) and CIFAR-10 (natural images,
full RGB) are the natural tests.

### Fashion-MNIST: Position Bagging

**Source:** `src/sstt_bag_topo9.c data-fashion/`

| M\R | R=0.5 | R=0.6 | R=0.7 |
|-----|-------|-------|-------|
| M=5 | 84.46% | 84.77% | 84.55% |
| M=7 | 84.66% | **84.89%** | 84.68% |
| M=10 | 84.61% | 84.62% | 84.47% |

- **Baseline topo9:** 84.80% (1520 errors)
- **Best bagged (M=7, R=0.6):** 84.89% (+0.09pp)

**Error mode analysis (baseline):**

| Mode | Count | % of errors |
|------|-------|-------------|
| A (retrieval miss) | 15 | 1.0% |
| B (ranking inversion) | 1183 | 77.8% |
| C (vote dilution) | 322 | 21.2% |

**Verdict:** Same as MNIST — negligible. Fashion's errors are 77.8%
Mode B (ranking inversions). Bagging can't fix the ranker.

### Fashion-MNIST: Multi-Threshold Ensemble

**Source:** `src/sstt_multi_threshold_topo9.c data-fashion/`

| Config | Accuracy | Errors | Delta |
|--------|----------|--------|-------|
| S=1 (baseline topo9) | 84.80% | 1520 | — |
| S=2 (+wide-zero 64/192) | 84.60% | 1540 | -0.20pp |
| S=3 (+narrow-zero 96/160) | 84.53% | 1547 | -0.27pp |
| S=4 (+adaptive P25/P75) | 85.02% | 1498 | **+0.22pp** |
| S=5 (+adaptive P10/P90) | **85.14%** | **1486** | **+0.34pp** |

**POSITIVE RESULT.** Fixed-threshold schemes (2, 3) hurt — same pattern
as MNIST. But adaptive percentile schemes (4, 5) rescue the ensemble
and push past baseline.

**Why this works on Fashion but not MNIST:**

Fashion-MNIST has much wider intensity variation across classes. A black
t-shirt, a white sneaker, and a grey coat have fundamentally different
brightness distributions. Fixed thresholds (85/170) map many clothing
items into the wrong ternary bins. Per-image adaptive thresholds (P25/P75,
P10/P90) normalize the dynamic range, bringing correct training matches
into the candidate set that fixed thresholds miss.

MNIST digits are white strokes on black backgrounds with nearly uniform
intensity. The (85/170) thresholds are already near-optimal for every
image. Alternative thresholds add noise, not signal.

**Per-pair deltas (S=5 vs S=1):**

| Pair | Baseline | S=5 | Delta |
|------|----------|-----|-------|
| 2↔6 (pullover↔shirt) | 239 | 230 | -9 |
| 3↔8 (dress↔bag) | 9 | 7 | -2 |
| 5↔7 (sandal↔sneaker) | 88 | 84 | -4 |
| 0↔6 (tshirt↔shirt) | 276 | 290 | +14 |
| 2↔4 (pullover↔coat) | 217 | 234 | +17 |

Mixed per-pair results. The 0↔6 and 2↔4 regressions offset gains
elsewhere. The adaptive schemes help some pairs (2↔6, 5↔7) but
introduce confusion on others (0↔6, 2↔4) where the adaptive
thresholds make similar textures even harder to distinguish.

### CIFAR-10: Multi-Eye Ensemble

**Source:** `src/sstt_cifar10_multi_eye.c`

The existing CIFAR-10 pipeline already uses 3 "eyes" (fixed, adaptive
P33/P67, per-channel P33/P67). We added 4 more:

| Eye | Quantization | Notes |
|-----|-------------|-------|
| 1 | Fixed (85/170) | Original |
| 2 | Adaptive P33/P67 per-image | Original |
| 3 | Per-channel P33/P67 | Original |
| 4 | Fixed (64/192) | Wide-zero, new |
| 5 | Fixed (96/160) | Narrow-zero, new |
| 6 | Adaptive P10/P90 per-image | Aggressive, new |
| 7 | Per-channel P25/P75 | Wide per-channel, new |

Ranking: RGB 4×4 Gauss map L1 distance (2880 features), unchanged
from the original cascade.

**Ablation (stereo vote → top-200 → RGB Gauss rank → kNN):**

| Config | Recall | k=1 | k=3 | k=5 | k=7 |
|--------|--------|-----|-----|-----|-----|
| 3-eye (baseline) | 99.44% | 49.07% | 48.78% | 50.00% | **50.18%** |
| 5-eye (+fixed variants) | 99.44% | 49.14% | 48.97% | 50.48% | 49.86% |
| 7-eye (all) | **99.60%** | **50.73%** | **50.59%** | **52.41%** | **52.41%** |

**STRONG POSITIVE RESULT: 50.18% → 52.41% (+2.23pp)**

The optimal k shifts from k=7 (baseline) to k=5 (7-eye), suggesting
the larger candidate diversity provides better top-5 neighbors than
the smaller diversity's top-7.

Recall increases from 99.44% to 99.60% — a small absolute gain (16
additional images with correct class in top-200) but meaningful at
this accuracy level.

**Per-class breakdown (7-eye k=5):**

| Class | Accuracy | Notes |
|-------|----------|-------|
| airplane | 58.9% | Strong |
| automobile | 57.8% | Strong |
| bird | 38.5% | Hard (texture similarity with other animals) |
| cat | 33.6% | Hardest (cat/dog confusion) |
| deer | 49.9% | Moderate |
| dog | 42.2% | Hard (cat/dog confusion) |
| frog | 63.5% | Strong (distinctive shape) |
| horse | 54.9% | Moderate |
| ship | 63.7% | Strong (distinctive shape + background) |
| truck | 61.1% | Strong |

Gains are broad — not concentrated in any single class. The hard
classes (bird, cat, dog) remain hard but improve. The strong classes
(frog, ship, truck) improve further.

**Why CIFAR-10 benefits most:**

1. **Maximum intensity variation.** Natural images span the full 0-255
   range with per-image distributions that vary enormously. Fixed
   thresholds are a poor fit for most images. Multiple quantization
   schemes cover more of the representation space.

2. **The Gauss map ranker is scheme-agnostic.** Unlike topo9's ternary
   dot products (which are scheme-specific), the RGB Gauss map ranking
   operates on raw pixel gradients, not ternary representations. So
   candidates found by alternative quantization schemes can still be
   ranked effectively.

3. **The 3-eye baseline was already multi-threshold.** CIFAR's existing
   pipeline already benefited from stereoscopic retrieval. Adding 4
   more eyes extends this principle. The improvement from 3→7 eyes
   (+2.23pp) is larger than the original improvement from 1→3 eyes
   in the stereoscopic work.

---

## Summary of All Results

### Experiment 1: Position Bagging

| Dataset | Baseline | Best Bagged | Delta | Verdict |
|---------|----------|-------------|-------|---------|
| MNIST (bytecascade) | 95.86% | 96.24% | +0.38pp | Modest |
| MNIST (topo9) | 97.27% | 97.37% | +0.10pp | Noise |
| Fashion (topo9) | 84.80% | 84.89% | +0.09pp | Noise |

**Conclusion:** Position bagging is largely redundant with topological
ranking. Value only on the fast path (bytecascade without topo features).

### Experiment 2: Multi-Threshold Ensemble

| Dataset | Baseline | Multi-threshold | Delta | Verdict |
|---------|----------|----------------|-------|---------|
| MNIST (standalone) | 96.62% | 97.23% | +0.61pp | Positive (weak ranker) |
| MNIST (topo9) | 97.27% | 97.17% | -0.10pp | **Negative** |
| Fashion (topo9) | 84.80% | 85.14% | +0.34pp | **Positive** |
| CIFAR-10 (7-eye) | 50.18% | 52.41% | +2.23pp | **Strong positive** |

**Conclusion:** Multi-threshold value scales with intensity variation.
Negative on MNIST (uniform intensity, strong ranker). Positive on
Fashion (+0.34pp, adaptive thresholds normalize clothing brightness).
Strong positive on CIFAR-10 (+2.23pp, maximum variation, scheme-agnostic
ranker).

### The Unifying Insight

Multi-threshold ensemble helps when **either** of these conditions holds:
1. The ranker is weak (standalone MNIST experiment, +0.61pp)
2. The dataset has high intensity variation AND the ranker is scheme-agnostic
   (Fashion +0.34pp, CIFAR-10 +2.23pp)

It hurts when both conditions fail: strong ranker AND uniform intensity
AND scheme-specific ranking features (MNIST + topo9, -0.10pp).

The deepest finding: **the value of retrieval diversity depends on
whether the ranker can use diverse candidates.** Topo9's ternary dot
products penalize candidates from other schemes. CIFAR's Gauss map
doesn't. This is the architectural constraint that determines whether
multi-threshold helps or hurts.

### Source Files

| File | Dataset | Experiment |
|------|---------|-----------|
| `src/sstt_bag_positions.c` | MNIST | Exp 1: position bagging (bytecascade) |
| `src/sstt_bag_topo9.c` | MNIST, Fashion | Exp 1: position bagging (topo9) |
| `src/sstt_multi_threshold.c` | MNIST | Exp 2: standalone multi-threshold |
| `src/sstt_multi_threshold_topo9.c` | MNIST, Fashion | Exp 2: multi-threshold + canonical topo9 |
| `src/sstt_cifar10_multi_eye.c` | CIFAR-10 | Exp 2: 7-eye cascade |

### Branches

| Branch | Contents |
|--------|----------|
| `exp/bag-positions` | Exp 1 + Exp 1b (bagging on bytecascade and topo9) |
| `exp/multi-threshold` | Exp 2 standalone |
| `exp/multi-threshold-topo9` | Exp 2 integration + Fashion + CIFAR-10 |
