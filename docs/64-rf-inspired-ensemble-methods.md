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
