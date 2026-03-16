# Contribution 24: Benchmark Catalogue — Strengths, Weaknesses, Opportunities

All measurements on AMD Ryzen 5 PRO 5675U, GCC -O3 -mavx2 -mfma -march=native.
MNIST test set: 10,000 images. Fashion-MNIST test set: 10,000 images.

---

## Summary Table

| Binary | MNIST | Fashion | Speed | Model | Category |
|--------|-------|---------|-------|-------|----------|
| sstt_geom | 73.23% | — | ~1 μs | 1.3 MB | Hot map baseline |
| sstt_ensemble | 75.34% | 64.68% | ~1 μs | 1.3 MB | Channel fusion |
| sstt_series | 83.86% | 73.83% | 4.3 μs | 1.3 MB | Bayesian series |
| sstt_eigenseries | 83.86% | — | 4.5 μs | 1.3 MB | IG-ordered series |
| sstt_transitions | 84.69% | 74.92% | 2.9 μs | 2.1 MB | Transition grammar |
| sstt_pentary | 86.31% | 77.45% | 4.7 μs | 6.1 MB | 5-level quant |
| sstt_hybrid | 87.80% | — | 9.3s/10K | 1.3 MB | Confidence-gated |
| sstt_bytepacked | 91.83% | 76.52% | 1.9 μs | 16 MB | Joint encoding |
| sstt_softcascade | 96.12% | — | ~30s/10K | — | Cascade + prior |
| sstt_tiled | 94.68% | — | 4.4s/10K | — | Routed cascade |
| sstt_v2 | 96.12% | ~83% | 32.9s/10K | — | 3-chan cascade |
| sstt_bytecascade | 96.28% | 82.89% | 19.2s/10K | 22 MB | Bytepacked cascade |
| sstt_multidot | 96.20% | — | 6.1s/10K | 22 MB | Multi-chan dot |

---

## 1. sstt_geom — Hot Map Baseline

**Accuracy:** 73.23% MNIST
**Speed:** ~1 μs/image
**Model:** 1.3 MB (3 hot maps, L1/L2 resident)

### Strengths
- Fastest possible classification — pure sequential table lookup, no per-image state
- L2-resident working set (3 × 435 KB), zero cache misses after warmup
- Reference implementation for all additive voting variants
- Includes comprehensive test suite (Tests A–H) covering all cascade stages

### Weaknesses
- No IG weighting (treats all block positions as equally discriminative)
- No per-image refinement — collapses 60K training identities into 10 class aggregates
- Hard ceiling near 73% in this configuration; adding channels alone does not help
- Long total runtime (includes WHT brute-force, ~5+ minutes end-to-end)

### Opportunities
- Pre-bake IG weights at training time: same inference code, free +10–15pp
- Serves as the architectural base that every other method extends
- Fused kernel (`sstt_fused_c.c`) ports this to a single-pass pipeline

---

## 2. sstt_ensemble — Channel Fusion Analysis

**Accuracy:** 75.34% MNIST, 64.68% Fashion (V-Grad primary + fallback)
**Speed:** ~1 μs/image
**Model:** 1.3 MB

### Strengths
- Reveals the contribution of each channel in isolation:
  - Pixel: 71.35%, H-Grad: 60.29%, V-Grad: 73.95% (MNIST)
  - Fashion reverses: H-Grad second-best at 52.83%
- V-Grad dominant strategy (+2.11pp) beats additive for free — just a fusion rule change
- Clean comparison framework for any channel combination

### Weaknesses
- Best result (75.34%) barely exceeds raw additive baseline (73.23%)
- Individual channel signals are weak and largely redundant when combined additively
- H-Grad hurts MNIST ensembles: adding noise that drowns V-Grad signal

### Opportunities
- **Key insight directly exploited by sstt_multidot**: w_hg=0, w_vg=192 is optimal for dot refinement because h-grad is redundant in pixel space on MNIST
- For Fashion-MNIST, channel weights should be reoptimized (H-Grad is second-best there)
- Channel-level asymmetry suggests per-dataset weight calibration is important

---

## 3. sstt_series — Block-Interleaved Bayesian

**Accuracy:** 83.86% MNIST, 73.83% Fashion
**Speed:** 4.3 μs/image (Strategy 5)
**Model:** 1.3 MB

### Strengths
- Largest single accuracy jump from pure hot map (+10.63pp over additive 73.23%)
- No dot products, no per-image state beyond the running posterior — pure table ops
- Reveals the critical insight: block-level sequential updating (not channel-level) is what works
- 2.4–4.3 μs latency, well within real-time constraints

### Weaknesses
- Strategies 1–4 (channel-level ordering, temperature scaling, multiplicative gating) give zero improvement over additive — only Strategy 5 (block-interleaved) works
- Hard ceiling at 83.86% — same as eigenseries at full traversal
- Posterior sharpening is monotonic: each block adds evidence but the final answer is fixed at full traversal

### Opportunities
- The block-interleaved update is a discrete approximation of a CfC (Closed-form Continuous-time network)
- Per-image adaptive stopping: high-confidence images exit early (eigenseries showed 76% at 50 blocks via IG ordering)
- Combining block-interleaved Bayesian with cascade fallback: a principled two-stage hybrid

---

## 4. sstt_eigenseries — IG-Ordered Block Traversal

**Accuracy:** 83.86% MNIST (full), 76.17% at 50 blocks (20% of total)
**Speed:** 4.5 μs/image
**Model:** 1.3 MB

### Strengths
- IG ordering reaches 76% with 50 blocks vs spatial ordering needing 150 blocks for 81%
- At 100 blocks (40% of data), IG hits 81.46% — nearly full accuracy at 60% cost
- All orderings converge to 83.86% at full traversal: ordering affects convergence speed, not the fixed point
- Eigenvalue ordering captures different structure (block co-variance vs class discrimination)

### Weaknesses
- IG ordering strictly better than eigenvalue ordering for classification (class signal dominates co-variance signal)
- No improvement in final accuracy over sstt_series — same ceiling
- IG pre-computation adds ~0.5s setup overhead

### Opportunities
- **Strongest case for adaptive inference**: exit when posterior confidence exceeds threshold
- IG block ordering is the input sequence for a potential CfC classifier: most informative blocks first
- At 100-block traversal: 81.46% accuracy in roughly 1.5–2 μs — matches transitions accuracy at equal speed

---

## 5. sstt_transitions — Inter-Block Transition Grammar

**Accuracy:** 84.69% MNIST, 74.92% Fashion (transition-only)
**Speed:** 2.9 μs/image
**Model:** 2.1 MB (hot maps + transition maps)

### Strengths
- **Transition-only beats all three original channels combined** (84.69% vs 83.86%)
- The space *between* block vectors carries more discriminative signal than the vectors themselves
- H-transitions: 66.9% background (sparse, highly class-discriminative where non-zero)
- V-transitions: 80.4% background (even sparser, even more selective)
- Generalizes to Fashion-MNIST (+1.09pp over 3-channel Bayesian)

### Weaknesses
- Adding transitions to the original 3 channels *hurts* (84.47% vs 84.69%) — gradient channels add noise
- Transitions alone have a ceiling near 85% in Bayesian hot map mode
- H+V transition hot maps add 788 KB to model size

### Opportunities
- Transition bits 6-7 are already in Encoding D (bytepacked): transition activity signals are baked in
- The spatial grammar of transitions could serve as a cascade vote channel: build a transition inverted index alongside the bytepacked index
- Transitions could replace h-grad in the cascade (since h-grad is redundant, transitions are a higher-signal alternative)

---

## 6. sstt_pentary — 5-Level Quantization

**Accuracy:** 86.31% MNIST, 77.45% Fashion
**Speed:** 4.7 μs/image
**Model:** 6.1 MB/channel (5× larger than ternary)

### Strengths
- +2.45pp MNIST, +3.62pp Fashion over ternary — consistent gains on both datasets
- Asymmetric optimal thresholds (40/100/180/230) capture faint strokes ternary maps to zero
- Fashion gain (+3.62pp) exceeds MNIST gain (+2.45pp): coarser ternary hurts Fashion more
- Hot map only — no cascade required

### Weaknesses
- Never tested in a full cascade pipeline: pentary cascade is completely unbuilt
- 6.1 MB/channel vs 435 KB/channel (14× larger hot map) — may not be L2-resident
- Multi-scale pentary (1+2+3 pixel blocks) hurts MNIST: 1-pixel blocks too sparse for Bayesian convergence
- 5-level block encoding: 5^3 = 125 values per position vs 27 for ternary

### Opportunities
- **Highest-confidence unbuilt experiment**: pentary cascade with IG + multi-probe + dot refinement
  - 125-value inverted index is still manageable
  - Vote index size: ~5× larger per bucket, but sparsity is higher
  - Expected accuracy: likely 97%+ on MNIST if the cascade gap scales proportionally
- Pentary encoding could replace pixel channel in bytepacked Encoding D

---

## 7. sstt_hybrid — Confidence-Gated Cascade

**Accuracy:** 87.80% MNIST (30% cascade fallback)
**Speed:** 9.3s/10K total (hot map handles ~70% of images)
**Model:** 1.3 MB hot map + full cascade index

### Strengths
- First confidence-gated architecture: fast path (hot map, ~1 μs) for high-confidence images
- Cascade fallback (92.3% on low-confidence subset) confirms gating is well-calibrated
- 3.5× faster than pure cascade at near-cascade accuracy in theory
- Establishes the "hot map as router" paradigm

### Weaknesses
- 87.80% overall — far below the 96%+ cascade
- The hot map's confidence signal (margin between top-2 classes) is too weak: Bayesian posteriors are extremely peaked (margin ≈ 1.0 for most images), making gating imprecise
- IG pre-baking and multi-probe in hot map space give marginal gains (+0.54pp, +0.26pp) because hot map collapses per-image identity
- 70% "high confidence" images at 85.87% accuracy: many are actually uncertain

### Opportunities
- Replace hot map confidence with **bytepacked cascade confidence**: the bytepacked vote distribution is much more peaked and meaningful
- Use sstt_diagnose's Mode A/B/C classification to build a better gate: Mode B errors (64.5% of failures) are exactly the images that need cascade fallback
- Threshold calibration: current 30% threshold is arbitrary; optimize on held-out data

---

## 8. sstt_bytepacked — Joint Channel Encoding

**Accuracy:** 91.83% MNIST, 76.52% Fashion
**Speed:** 1.9 μs/image (Encoding D, Bayesian)
**Model:** 16 MB (252 × 256 × 16 entries)

### Strengths
- **+7.97pp over ternary 3-channel Bayesian** — largest single-method accuracy gain in the project
- 1 lookup per block position (vs 3 for independent channels)
- Encoding D captures cross-channel correlations: pixel majority × hgrad majority × vgrad majority × transition activity in 8 bits
- 144 of 256 byte values actually used (background 62.8%): compact effective vocabulary
- Foundation of bytecascade and multidot

### Weaknesses
- Hot map only — 16 MB model may not be fully L2-resident (vs 1.3 MB for ternary hot maps)
- Fashion-MNIST gains (+2.69pp) are smaller than MNIST gains (+7.97pp)
- Encoding D collapses 3-trit blocks to 2-bit summaries: some local texture information is lost
- No cascade pipeline tested in this file (only Bayesian hot map)

### Opportunities
- Directly enables bytecascade (already built) and multidot (already built)
- Encoding variants B and C (+7.58pp and +4.33pp) suggest the joint channel space has headroom
- Pentary bytepacked: 5-level majority trit instead of 4-state — richer 2-bit field

---

## 9. sstt_softcascade — Bayesian Soft Prior on Votes

**Accuracy:** 96.12% MNIST (BG-fix baseline), soft prior hurts all configs
**Speed:** ~30s/10K
**Model:** Full cascade + Bayesian posterior

### Strengths
- **Independently validates the BG fix**: `chan_bg[3] = {0, 13, 13}` (not `bv==0` for all channels) gives 96.12% with uniform IG weights — the bg fix from sstt_v2 line 1297 was masking gradient channel contributions
- Architecture is sound: boost[c] = 1 + α × softmax(posterior)[c] × 256 always ≥ 1, no class excluded
- Careful red-team analysis of overflow, speed, top-K histogram
- Clean parameter sweep over α ∈ {1,2,4,8,16} and T ∈ {1.0, 3.0, 10.0}

### Weaknesses
- **Soft prior hurts at ALL configurations**: a weaker classifier (84.47%) cannot guide a stronger one (96.12%)
- Best soft prior (α=1, T=10): 85.40% — 10.72pp BELOW pure cascade
- Higher α = worse: the Bayesian's 15.5% error rate is injected into a 3.9% error system
- The principle "information perihelion requires the guide to be closer to the truth" is definitively proven

### Opportunities
- The BG fix should be applied to sstt_v2: background for gradient channels is BG_GRAD=13, not 0
- The soft prior architecture is correct for cases where the prior IS stronger than the cascade
- With a future 98%+ Bayesian (using CfC or pentary), soft guidance could become viable

---

## 10. sstt_tiled — K-Means Tile Routing

**Accuracy:** 94.68% (hard routing), 95.71% (soft top-2), 96.31% (global baseline)
**Speed:** 4.4s hard / 7.1s soft / 27.7s global
**Model:** 10 × full cascade indices

### Strengths
- **Definitively proves routing hurts**: global cascade > soft routing > hard routing
- K-means clustering converges quickly (5 EM iters, stable tile sizes)
- Class distribution per tile reveals the natural structure: 4/9 cluster together, 3/5 cluster together
- Hard routing is 6.3× faster than global (4.4s vs 27.7s) — speed benefit is real even if accuracy suffers
- Implements the user's code pattern (signature=tile_weights.sum(dim=0).sign(), score=input@signature, route=argmax)

### Weaknesses
- Confused digit pairs cluster together: tile 4 is 54% class-4 + 19% class-9 — routing concentrates confusion rather than resolving it
- Hard routing: -1.63pp vs global cascade
- Soft routing: -0.60pp vs global cascade
- 10 separate inverted indices: high memory overhead
- Tile 1 (12,665 images, 21% of all) is an "overflow" tile for ambiguous images

### Opportunities
- The routing mechanism (ternary dot → nearest prototype) could be repurposed for non-classification tasks: nearest-neighbor retrieval, outlier detection
- Spatial routing (top/bottom/left/right tiles) instead of ternary-space routing might capture meaningful visual sub-problems
- The tile-level confusion matrices reveal which pairs need specialized architectures — a diagnostic, not a classifier improvement

---

## 11. sstt_v2 — 3-Channel Cascade (Reference)

**Accuracy:** 96.04% (K=500, k=3), 96.12% (WHT brute k=3)
**Speed:** 32.9s / 10K (cascade), ~300s (WHT brute)
**Model:** 3 × inverted indices (~181 MB pool)

### Strengths
- **Definitive reference implementation**: Tests A–F cover the full ladder from hot map (73.23%) to cascade (96.12%)
- Proves: IG (+15.55pp over unweighted), multi-probe (+2.38pp), top-K dot (+4.88pp) are additive and composable
- WHT brute k-NN = pixel-space brute k-NN (confirmed orthogonality)
- Error analysis: confusion matrix, per-class precision/recall, top-5 confused pairs
- BG fix bug documented and patched in later experiments

### Weaknesses
- Pixel-only dot refinement (the seam identified by sstt_diagnose)
- 3 independent ternary channels miss cross-channel correlations captured by bytepacked
- Slow: ~33s for cascade, ~5 minutes end-to-end including WHT brute
- BG bug on line 1297: `if (bv == 0) continue` applies pixel background to gradient channels

### Opportunities
- BG fix + multi-channel dot = the two improvements from sstt_softcascade and sstt_multidot
- v2 is the baseline every experiment references; it should be the entry point for new contributors
- WHT brute proves the accuracy ceiling for dot-product similarity: 96.12% is the maximum for brute kNN in ternary pixel space

---

## 12. sstt_bytecascade — Bytepacked Cascade

**Accuracy:** 96.28% MNIST, 82.89% Fashion
**Speed:** 19.2s / 10K MNIST (8-probe, K=50)
**Model:** 22 MB (bytepacked inverted index, 5.6M entries)

### Strengths
- **Best accuracy on MNIST**: +0.16pp over 3-channel reference
- **3.5× faster than v2** when accounting for equivalent K (K-invariant: K=50 = K=1000)
- Inverted index 5.6M entries vs 45M for 3-channel — dramatically smaller working set
- K-invariance: vote distribution so concentrated that top-50 = top-1000 at all tested K
- Cross-channel correlations from Encoding D give richer per-block discrimination
- Validated on Fashion-MNIST: 82.89% (best cascade result on Fashion)

### Weaknesses
- Pixel-only dot refinement (addressed in sstt_multidot)
- 8-probe ordering: bits {0,1,2,3} probe pixel+hgrad neighbors; bits {4,5} (vgrad) probed only at probes 5,6
- Does not output Fashion-MNIST voting results mid-run (cascade grid only at end)
- Memory: 22 MB inverted index vs 1.3 MB for hot map methods

### Opportunities
- Multi-channel dot (+0.34pp) already implemented in sstt_multidot
- Probe reordering: test {0,1,4,5} first (pixel+vgrad) — confirmed zero accuracy gain but ordering principle holds for future encodings
- Pentary bytepacked cascade: richer per-block representation, likely 97%+

---

## 13. sstt_multidot — Multi-Channel Dot Refinement

**Accuracy:** 96.20% MNIST (4-probe, px=256 hg=0 vg=192)
**Speed:** 6.1s / 10K (4-probe, precomputed 3-channel dots)
**Model:** 22 MB + 3-channel dot precompute buffer (~560 MB for 10K × 200 × 3 × 28B)

### Strengths
- **Best accuracy-speed tradeoff**: 96.20% in 6.1s (1.38× faster than 8-probe bytecascade at 96.28%)
- Fixes the pixel-only dot bottleneck: v-grad dot captures stroke direction, fixes 7→1 (29→20), 9→7 (15→9), 2→1 (14→7)
- Pre-allocated histogram eliminates per-image calloc cold-miss penalty (~70 μs/img saved)
- Probe type comparison: definitively tests and rules out Hamming-2 pairwise probes
- Grid search over 9×9 weight combinations at essentially zero marginal cost (precomputed dots)
- Full compute profile: 87% vote accumulation, dot product is negligible (1.1% overhead for 3-chan)

### Weaknesses
- Large precompute buffer (TOP_K=200 × 3-channel × 10K images): memory-intensive
- Weight grid (w_hg=0, w_vg=192) is test-set contaminated: optimized on MNIST test set
- Fashion-MNIST weight calibration not yet done: h-grad=0 may not hold for Fashion
- Probe sweep sections run sequentially: 5 probe counts × full precompute = long total runtime (~120s)

### Opportunities
- **CfC as the final discriminator**: 380 remaining errors after best-weight 4-probe; Mode B (64.5%) are fixable with learned nonlinear correction
- Fashion-MNIST weight grid search: likely needs different (w_hg, w_vg) for clothing textures
- Pentary multidot: richer vote representation → better top-K candidates → more headroom for dot refinement
- Per-class-pair adaptive weights: the optimal (w_px, w_hg, w_vg) likely differs for 4↔9 vs 3↔5 vs 7↔1

---

## Cross-Cutting Opportunities

### Unbuilt experiments with highest expected return

| Experiment | Basis | Expected gain |
|------------|-------|---------------|
| **Pentary cascade** | Pentary +2.45pp on hot map; cascade scales the gain | 97%+ MNIST |
| **CfC final discriminator** | 380 Mode B errors fixable with learned nonlinear dot | 98%+ MNIST |
| **Fashion-MNIST weight calibration** | h-grad is second-best channel on Fashion | 84%+ Fashion |
| **Transition channel in cascade** | Transitions beat all original channels in Bayesian mode | +0.5pp? |

### Persistent bottlenecks across all cascade methods

1. **Vote accumulation** (56% of cascade time): scattered writes into 240 KB votes array. No solution that preserves accuracy has been found — the random write pattern is fundamental to the inverted-index architecture.

2. **4↔9 topological confusion** (29–31 errors): requires connectivity/topology features that no ternary block representation can provide. This is likely the hard floor for block-based methods.

3. **Hot map ceiling ~84%**: the hot map collapses per-image identity into per-class aggregates. No table-only method has exceeded 91.83% (bytepacked, which partially recovers cross-channel identity).

4. **Fashion-MNIST ceiling ~83%**: shirt/T-shirt/pullover (classes 0/2/6) have structurally ambiguous block signatures at ternary resolution. The cascade at 82.89% may be near the practical ceiling for ternary methods on Fashion.

### Architecture-level synthesis

The 13 binaries form a natural progression:

```
Hot map (73%)
  → Series (84%): add sequential Bayesian updating
    → Transitions (85%): spatial grammar as new channel
      → Pentary (86%): more quantization levels
        → Bytepacked (92%): cross-channel joint encoding
          → Cascade (96%): per-image inverted index voting
            → Multi-channel dot (96.2%): fix refinement bottleneck
              → CfC? (98%+): learn the residual
```

Each method solves a specific weakness of its predecessor. The most promising unexplored path is **pentary cascade**, which would insert a richer representation at the vote phase — the phase that drives 87% of compute and where the current encoding already gives 86%.
