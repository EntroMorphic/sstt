# SSTT Comprehensive Audit

**Date:** 2026-03-26
**Auditor:** Claude (Opus 4.6)
**Scope:** Full repository — 110+ C source files, 64 numbered docs, 4 paper drafts, Makefile, CI, tests, scripts

---

## 1. What This Project Is

SSTT is a three-year research notebook exploring a single question: **how far can you get on image classification with zero learned parameters?**

The answer: surprisingly far.

- **MNIST: 97.27%** — exceeds brute k=3 NN (96.12%) by +1.15pp using 1200x fewer comparisons
- **Fashion-MNIST: 85.68%** — exceeds brute 1-NN (76.86%) by +8.82pp with zero architectural changes
- **CIFAR-10: 50.18%** — 5x random, honestly presented as a boundary test, not a success

The system uses no gradient descent, no backpropagation, no floating-point arithmetic at inference. Classification is integer table lookups, AVX2 byte operations, and add/subtract. The entire model fits in L2 cache.

The codebase is 110+ self-contained C files (each a standalone executable), 64 numbered documentation contributions totaling 500KB+, and a Makefile with 60+ build targets. MIT licensed, CI via GitHub Actions with AVX2 detection.

---

## 2. The Architecture (What It Actually Does)

```
Raw image (28x28 uint8)
    |
    v  Ternary quantization: pixel < 85 -> -1, 85-169 -> 0, >= 170 -> +1
    |
    +-- Pixel channel (quantized intensity)
    +-- H-grad channel (horizontal finite difference, clamped to {-1,0,+1})
    +-- V-grad channel (vertical finite difference, clamped to {-1,0,+1})
    |
    v  Block encoding: 3x1 horizontal blocks at 252 overlapping positions
    |  Per block -> ternary signature (27 values) or bytepacked (256 values)
    |  Bytepacked fuses 3 channels + transition flags into 1 byte
    |
    v  Inverted-index retrieval
    |  For each (position, signature): look up training image IDs
    |  Add IG-weighted votes to 60K-element accumulator
    |  Multi-probe: also look up 6-8 Hamming-1 trit-flip neighbors at half weight
    |
    v  Top-K candidate extraction (K=50 suffices -- K-invariance)
    |
    v  Ranking: weighted multi-channel ternary dot product
    |  score = 256 * dot(pixel) + 192 * dot(v-grad)
    |  Computed via _mm256_sign_epi8 (32 ternary MACs per cycle)
    |
    v  Topological augmentation (topo9 config):
    |  + Gradient divergence in spatial grid cells (2x4 MNIST, 3x3 Fashion)
    |  + Enclosed-region centroid via flood-fill
    |  + Horizontal intensity profile
    |  + Kalman adaptive weighting (MAD-based per-query gain)
    |  + Bayesian sequential accumulation (CfC with decay S=2)
    |
    v  k=3 majority vote on top-3 scored candidates -> class prediction
```

**The adaptive router adds three tiers:**

- **Tier 1 (<1us):** Vote unanimity >= 198/200 -> return plurality instantly. Covers 10-15% of images at 99.9% accuracy.
- **Tier 2 (~1.5us):** Dual Hot Map (pixel + Gauss frequency tables). Covers 15-18%.
- **Tier 3 (~1ms):** Full pipeline for remaining queries.
- **Global:** 96.50% at 0.67ms average latency.

---

## 3. Complete Results Ledger

### MNIST (10,000-image standard test set)

| Method | Accuracy | Latency | Source |
|--------|----------|---------|--------|
| Brute 1-NN (pixel) | 95.87% | — | Baseline |
| Brute k=3 NN (WHT) | 96.12% | — | Baseline |
| Naive Bayes hot map (pixel only) | 73.23% | ~1.5us | Doc 02 |
| Bytepacked Bayesian hot map | 91.83% | 1.9us | Doc 22 |
| Pentary hot map (5-level) | 86.31% | 4.7us | Doc 24 |
| Dual Hot Map | 76.53% | ~1.5us | Doc 48 |
| IG-weighted vote only (no dots) | 88.78% | — | Paper draft Table 1 |
| Zero-weight cascade (K=50, k=3) | 96.04% | — | Doc 11 |
| Bytepacked cascade (Encoding D) | 96.28% | 930us | Doc 22 |
| Oracle v2 (unanimity-gated) | 96.44% | 1.1ms | Doc 25 |
| Three-Tier Router | 96.50% | 0.67ms | Doc 51 |
| Sequential Field Ranking (topo9) | **97.27%** | ~1ms | Doc 31 |
| Val/Holdout confirmed | **97.27%** | ~1ms | Doc 35 |

### Fashion-MNIST (10,000-image standard test set, zero architectural changes)

| Method | Accuracy | Notes | Source |
|--------|----------|-------|--------|
| Brute 1-NN (pixel) | 76.86% | Baseline | — |
| Bytepacked cascade | 82.89% | No tuning from MNIST | Doc 15 |
| Stereoscopic 3-eye | 86.12% | Multi-perspective fusion | Doc 39 |
| Val/Holdout confirmed | **85.68%** | Publication-ready | Doc 35 |
| Three-Tier Router | 83.42% | At 0.88ms | Doc 54 |
| Router Tier 1 (9.6% coverage) | **100.00%** | Zero false positives | Doc 54 |

### CIFAR-10 (10,000-image standard test set, 32x32 RGB)

| Method | Accuracy | Notes | Source |
|--------|----------|-------|--------|
| Grayscale ternary cascade | 26.51% | First attempt | Doc 37 |
| Flattened RGB Bayesian | 36.58% | Color blocks | Doc 37 |
| MT4 full stack | 42.05% | 81-level + topo + Bayesian | Doc 38 |
| Stereo + MT4 | 44.48% | 3-eye retrieval | Doc 41 |
| Grid Gauss map (gray, brute) | 46.06% | Shape geometry | Doc 42 |
| Grid Gauss map (RGB, brute) | 48.31% | 4x4 grid, 45 bins/channel | Doc 42 |
| Cascade + RGB Gauss (stereo vote) | **50.18%** | 5x random, best result | Doc 43 |

### 224x224 Scaling (requires baseline context)

| Method | MNIST-224 | Fashion-224 | Source |
|--------|-----------|-------------|--------|
| Brute kNN (56x56 pooled) | 92.67% | 78.17% | Doc 63 |
| Full hot map | 11.60% | 46.77% | Doc 62 |
| Sparse hot map (6.25% knots) | 11.35% | 41.65% | Doc 62 |
| Hierarchical fused | — | 45.37% | Doc 62 |
| % of brute baseline (full) | 12.5% | 59.8% | Computed |
| % of brute baseline (sparse) | 12.2% | 53.3% | Computed |

**Key context:** 224x224 images are nearest-neighbor upscaling of 28x28. Zero new information, 64x more block positions, same training set size. MNIST-224 hot map classifies nearly everything as digit "1" (mode collapse). Fashion-224 is genuinely usable but still well below brute baseline.

---

## 4. What Is Novel

### 4.1 K-Invariance (the central empirical finding)

**Claim:** Top-50 candidates by vote contain every relevant neighbor out of 60,000 training images. Accuracy is identical from K=50 through K=1000.

| K | Accuracy (bytepacked cascade) |
|---|------|
| 50 | 96.28% |
| 100 | 96.28% |
| 200 | 96.28% |
| 500 | 96.28% |
| 1000 | 96.28% |

**Why this matters:** This is a 1200x compression ratio with zero recall loss. The inverted-index vote achieves 99.3% effective recall on the relevant set. This means the retrieval problem is solved at K=50. All remaining classification error is a ranking problem, not a retrieval problem. This reframes the entire research agenda and is the mechanism by which the cascade exceeds brute kNN despite examining 1200x fewer candidates.

**Assessment:** Clean, empirical, reproducible. This is the strongest single result in the project. It deserves to be the central section of the paper, not a subsection of Analysis.

### 4.2 `_mm256_sign_epi8` as Exact Ternary Multiply

**Claim:** The AVX2 instruction `_mm256_sign_epi8(a, b)` performs exact ternary multiplication over {-1, 0, +1} at 32 operations per cycle with no multiply instruction.

**Proof (from paper Appendix B):**

```
sign_epi8(a, b)[i] =
    a[i]   if b[i] > 0
   -a[i]   if b[i] < 0
    0      if b[i] = 0
```

Over the ternary alphabet b[i] in {-1, 0, +1}, this computes `a[i] * b[i]` exactly. Accumulation in int8 is safe for up to 127 iterations; MNIST images have at most 28 blocks per row, well within bounds.

**Assessment:** Small trick, but genuinely clever. I have not seen this documented in the ternary-network literature (TWN, TTN use conditional branching or lookup tables for ternary multiply). Worth a paragraph in any systems paper. The proof of correctness is clean.

### 4.3 Cascade Error Autopsy (Mode A/B/C Decomposition)

**Claim:** Every misclassification can be categorized into three mutually exclusive modes diagnosable at inference time:

- **Mode A (1.7% of errors): Retrieval failure.** Correct class absent from top-K. The vote missed relevant training images. Unfixable by better ranking.
- **Mode B (64.5% of errors): Ranking inversion.** Correct class present in top-K but wrong class wins dot product, typically by 3-20 score units.
- **Mode C (33.8% of errors): Vote dilution.** Dot product correctly identifies nearest neighbor, but k=3 majority vote dilutes it with wrong-class neighbors.

**Derived result:** 98.3% of errors occur at ranking, not retrieval. A fixable ceiling of ~98.6% exists if all Mode B and C errors are resolved. The topological ranking system moves from 96.04% toward this ceiling, fixing +111 MNIST errors and +215 Fashion errors.

**Assessment:** Good methodology. Turns a vague "we got 96%" into a precise engineering target. The decomposition is testable and reproducible — each mode is diagnosable from the data structures available at inference time. This is the kind of analysis that reviewers appreciate.

### 4.4 Bytepacked Encoding D (Pareto Improvement)

**Claim:** Fusing three ternary channels + transition flags into one byte (256-value signatures instead of 27-value) gives +0.16pp accuracy AND 3.5x speed simultaneously.

**Why this is notable:** Pareto improvements — better on both axes — are rare. Usually you trade accuracy for speed or vice versa. The encoding captures cross-channel correlations absent in the independent channels, so it's not just a compression trick.

**Numbers:** Bytepacked cascade: 96.28% at 930us. Three-channel cascade: 96.12% at ~3200us. Same architecture, different encoding.

**Assessment:** The encoding design itself isn't profound (it's a byte-packing scheme). The result is what's interesting. Clean, reproducible.

### 4.5 Confidence Map as Free Byproduct

**Claim:** Vote concentration from the retrieval step predicts classification difficulty at zero additional cost. A 264-entry lookup table achieves 99.6% accuracy on 22% of MNIST images.

**Mechanism:** If the top vote count is high and concentrated in one class (unanimity), the image is "easy." If votes are scattered, the image is "hard." This signal is already computed — it's the vote accumulator — so it costs nothing.

**Router Tier 1 consequence:** On Fashion-MNIST, the unanimity threshold (198/200) classifies 9.6% of the test set at 100.0% accuracy — zero false positives across 960 images. Same threshold, set on MNIST, applied directly to Fashion with no re-tuning.

**Assessment:** Elegant. The zero-false-positive property on Fashion-MNIST is particularly strong because it transfers without tuning. The "free" claim is accurate — no additional computation beyond what the cascade already performs.

---

## 5. What Is Useful (Solid Engineering, Not Novel)

### 5.1 The 87% Scattered-Write Bottleneck

Phase-timer profiling of the full cascade:

| Phase | Runtime share |
|-------|--------------|
| Vote accumulation (scattered writes) | **87%** |
| Ternary dot products | ~1% |
| Top-K selection | ~5% |
| Encoding and overhead | ~7% |

The dominant cost is scattered writes to a 240KB vote accumulator — one write per block per inverted-list entry. The bottleneck is cache-line thrashing, not arithmetic. The `_mm256_sign_epi8` dot products are effectively free.

**Generalization:** For any inverted-index system on modern hardware, the scatter-write memory pattern dominates. Optimizations that reduce arithmetic (SIMD, quantization) provide marginal benefit. Optimizations that reduce scattered memory accesses (blocking, sorting, reducing probe count) are productive. Reducing Hamming-1 probes from 8 to 4 gives 96.07% at 1.38x speed — a better trade-off than any arithmetic optimization.

**Assessment:** Useful insight for anyone building retrieval systems. Not novel (memory-bound workloads are well-understood), but the specific quantification and the practical consequence (reduce probes, not arithmetic) is worth documenting.

### 5.2 Fashion-MNIST Generalization at Zero Tuning

Same code, same thresholds (85/170), same block geometry (3x1, 252 positions), same Encoding D -> 82.89% bytepacked cascade, 85.68% with topological ranking. The only operation that changes is IG re-computation on Fashion training data — a closed-form O(N*P) operation, not tuning.

**Why this is stronger than the MNIST number:** It rules out overfitting to MNIST's specific structure. IG weights automatically redistribute — h-grad positions get higher IG on Fashion (clothing has horizontal texture; digits don't). The +8.82pp lift over brute kNN on Fashion is larger than the +1.15pp on MNIST, meaning the architecture provides more discriminative value on a harder task.

**Assessment:** This is the most convincing evidence that the approach has general value. Underplayed in the README (secondary bullet). Should be co-equal with the MNIST headline.

### 5.3 The Negative Results Archive

Eight documented failures with root-cause analysis:

| Experiment | Result | Root Cause |
|------------|--------|------------|
| Ternary PCA | -16 to -23pp | Double quantization destroys orthogonality; PCA eigenvectors in ternary space lose spatial addressing |
| Soft-prior cascade | Negative at all configs | Weaker classifier (hot map) cannot productively guide stronger (cascade); prior is noise |
| Spectral coarse-to-fine | No improvement | Vote already captures the coarse signal; spectral decomposition adds nothing |
| Tiled routing | Hurts accuracy | Confused pairs (3/8, 4/9) cluster in the same tiles; tiling makes separation harder |
| Taylor jet on CIFAR-10 | ~54% (fails) | Second-order surface curvature signal destroyed by ternary quantization at 32x32 |
| Navier-Stokes stream function | ~10% (random) | Global potential dominated by frame boundaries, not object structure |
| Variance-guided sparse knots | Negative | High-variance regions are spatially correlated; clustering creates coverage gaps |
| Delta map ("maps all the way down") | +19/+20 errors | Lookup collapses per-image identity; recursive quantization loses discriminative information |

**Assessment:** Honestly documented negative results are genuinely valuable. Each has a clear root cause that prevents future researchers from repeating the experiment. The PCA failure is particularly instructive (spatial addressing is load-bearing; destroying it is catastrophic).

### 5.4 The topo1-through-topo9 Ablation Series

Nine incremental experiments, each isolating one addition:

| Config | Addition | MNIST | Delta |
|--------|----------|-------|-------|
| topo1 | Gradient divergence (scalar) | 96.28% | +0.00 |
| topo2 | Grid divergence (2x4) | 96.55% | +0.27 |
| topo3 | + Enclosed centroid | 96.78% | +0.23 |
| topo4 | + Horizontal profile | 96.91% | +0.13 |
| topo5 | Kalman adaptive weighting | 97.01% | +0.10 |
| topo6 | Bayesian sequential (S=1) | 97.05% | +0.04 |
| topo7 | Bayesian sequential (S=2) | 97.11% | +0.06 |
| topo8 | Grid tuning (3x3 for Fashion) | 97.18% | +0.07 |
| topo9 | Full system (all features) | 97.27% | +0.09 |

**Assessment:** Textbook incremental methodology. Each contribution is isolated and measured. This is what makes the final 97.27% trustworthy — you can see exactly what each component buys.

### 5.5 Background Value Discrimination

Per-channel background constants: BG_PIXEL=0, BG_GRAD=13, BG_TRANS=13. Skipping background blocks during indexing reduces inverted-index noise.

**The bug that revealed it:** Before background suppression, gradient channels scored 20-60% accuracy. After: 60-74%. The fix is domain-general — any inverted-index system benefits from skipping uninformative tokens.

### 5.6 Haar vs. WHT Correctness Trap

Standard Haar wavelet is not orthogonal under ternary quantization. WHT (Walsh-Hadamard Transform) preserves pixel-space dot-product rankings exactly (Parseval's theorem). Documented in Doc 10 to prevent time-wasting by anyone applying wavelet transforms to ternary data.

### 5.7 The Three-Tier Router

96.50% at 0.67ms average latency. 2x throughput over non-tiered cascade. Zero false positives on the fast-path slice for both MNIST and Fashion-MNIST. Good systems engineering, well-implemented in `src/sstt_router_v1.c`.

---

## 6. What Is Understated

### 6.1 The Zero-Parameter Claim Itself

The paper draft and README lead with accuracy numbers. They should lead with the constraint.

97.27% on MNIST is not remarkable — dozens of models hit that. A simple 2-layer neural network gets 97%+. What is remarkable is achieving 97.27% **with no learned parameters, no gradient descent, and no floating-point arithmetic at inference.** Every parameter is derived from closed-form training statistics (IG weights, BG thresholds) or exhaustive discrete grid search (dot-product weights: 125 combinations of {0, 64, 128, 192, 256}^3).

The headline is the constraint, not the number.

### 6.2 K-Invariance as the Mechanism

K-invariance is the reason the cascade works. It's the answer to "how does examining 50 candidates beat examining 60,000?" — the 50 candidates are the same quality, and the cascade applies a richer ranking function to them. The paper draft covers this in Section 5.1 but it should be more prominent — it's the central empirical finding, not a supporting analysis.

### 6.3 The Fashion-MNIST Lift

+8.82pp over brute kNN on Fashion-MNIST vs. +1.15pp on MNIST. The architecture provides proportionally more value on a harder task. This is the generalization proof. It should be co-equal with the MNIST headline, not subordinate.

### 6.4 The 98.6% Fixable Ceiling

The error autopsy doesn't just classify errors — it derives a testable, quantitative research agenda. The ceiling is precise: if all Mode B and C errors are fixed, accuracy reaches ~98.6%. Exceeding that requires fixing Mode A (retrieval failures, 1.7% of errors), which requires fundamentally different block signatures or larger inverted lists. This turns a vague "improve accuracy" goal into specific engineering targets.

---

## 7. What Is Fluff

### 7.1 The Lincoln Manifold Method (LMM) Framing

Mapping the pipeline stages to "RAW -> NODES -> REFLECT -> SYNTHESIZE" is post-hoc relabeling of standard processing stages. Lines like "The wood cuts itself when you understand the grain" add zero technical content. This appears in:

- `CONTRIBUTING.md`
- `docs/archived/20-lmm-pass-composite-architecture.md`
- `docs/archived/lmm-findings/` (4 files)
- `docs/archived/lmm-sessions/` (16 files)
- README.md (traces)

**Recommendation:** Strip from all active docs. The archived LMM files can stay archived — they contain some genuine technical observations mixed with the framing. But the framing itself adds narrative weight without technical substance.

### 7.2 "Field-Theoretic" Language Where It Doesn't Earn It

The project uses language from differential geometry and fluid dynamics for operations that are finite differences on small integer grids:

- **"Discrete Green's theorem on ternary fields"** = summing a finite-difference divergence operator on a 28x28 grid. Technically accurate (Green's theorem relates boundary flux to interior divergence), but the invocation of the theorem adds no insight — you're just summing d_h/dx + d_v/dy.
- **"Lagrangian structural skeletons"** = flood-fill from high-gradient points, counting connected components. Not a Lagrangian method in any dynamical-systems sense.
- **"Navier-Stokes stream function"** = computing a scalar potential from gradient fields on a 28x28 grid. The Navier-Stokes equations describe viscous fluid flow; this is a discrete Poisson solve with no viscosity, no time evolution, and no inertia.
- **"Kalman adaptive weighting"** = scaling topological feature weights by 1/(MAD + constant). Structurally similar to Kalman gain, but calling it "Kalman" implies a state estimation framework that isn't present.

**Recommendation:** Describe each operation plainly. "Discrete divergence sum in spatial grid cells." "Flood-fill centroid of enclosed regions." "MAD-based adaptive weighting." Let reviewers decide whether the connections to field theory are meaningful. Overclaiming the math will get the paper desk-rejected at any venue with physics-literate reviewers.

### 7.3 The 224x224 Scaling Work

**MNIST-224:** Hot map achieves 11.60%. Brute kNN achieves 92.67%. The SSTT number is 12.5% of the achievable baseline. The representation has mode-collapsed — nearly everything is classified as digit "1" at 56x56 pooled resolution.

**Fashion-224:** Better — 46.77% full hot map, 59.8% of the brute baseline (78.17%). The feature space is genuinely usable for clothing. But the SSTT system still loses ~20pp to brute kNN.

**The fundamental issue:** 224x224 images are nearest-neighbor upscaling of 28x28. Every 8x8 block in the output is a copy of one original pixel. There is zero new information. The model has 64x more block positions but the same training set size — severely underfitted by construction.

Docs 62 and 63 correctly add baselines that make these numbers interpretable. The work is honest exploration. But it does not produce a publishable result, and framing it as "98.5% accuracy retention" (retention of an 11.6% classifier) is misleading.

**Recommendation:** Keep as archived exploration. Do not present as a success or a paper. The negative insight — that the architecture requires genuinely higher-resolution training data to scale, not upsampled copies — is worth one paragraph in a limitations section.

### 7.4 The CIFAR-10 Experiment Count

45+ CIFAR-10 source files exploring:
- Curvature operators (`sstt_cifar10_curvature.c`)
- Lagrangian particle tracing (`sstt_cifar10_lagrangian.c`)
- Mixture of experts (`sstt_cifar10_moe.c`)
- Multiscale hierarchies (`sstt_cifar10_multiscale.c`)
- Binary classifiers (`sstt_cifar10_binary.c`)
- Adaptive variants (`sstt_cifar10_adaptive.c`)
- Hierarchical decomposition (`sstt_cifar10_hierarchical.c`)
- And 38+ more

The best result across all variants: **50.18%** (stereo-vote -> RGB Gauss map cascade, Doc 43).

The honest ceiling assessment in Doc 60 is correct: inter-class Gauss distance analysis shows cat/dog at 214 vs. airplane/frog at 2145. At 32x32 resolution, visually similar classes cannot be distinguished without learned parameters regardless of the geometric features used.

**Recommendation:** Trim to the 3-4 files that tell the CIFAR-10 story: the first attempt (26.51%), the RGB Gauss map breakthrough (48.31%), the final cascade (50.18%), and the inter-class distance analysis that explains the ceiling. Archive the rest. The 45-file count suggests exhaustive search, not progressive insight.

### 7.5 The Contribution Count (64 Numbered Docs)

Many "contributions" are:
- Incremental parameter sweeps that could be a row in a table
- Negative results that could be a paragraph in an appendix
- Architectural proposals that were never validated
- Renumbered duplicates of the same experiment with different framing

**Recommendation:** Consolidate to ~15-20 focused documents:
1. Architecture overview (merge Docs 01, 05, 06, 07, 08, 09, 11)
2. Bytepacked encoding and cascade (merge Docs 14, 22)
3. Error autopsy and oracle routing (merge Docs 23, 25)
4. Topological ranking series (merge Docs 29, 30, 31)
5. Validation protocol (Doc 35, standalone)
6. CIFAR-10 boundary test (merge Docs 37, 38, 42, 43)
7. Three-tier router (merge Docs 48, 49, 50, 51)
8. Negative results compendium (merge Docs 17, 21, 36, 55, 58)
9. Red-team and audit (merge Docs 32, 34, 54, 60)
10. Scaling exploration (merge Docs 52, 59, 62, 63)

Keep the numbered originals archived for provenance. The consolidated set becomes the active documentation.

### 7.6 Specific Framing Issues

| Claim in Docs | Reality |
|---------------|---------|
| "MESI: The 12th Primitive" (Doc 14) | Read-only shared data being cache-friendly is basic CPU architecture, not a design primitive |
| "Cipher" metaphor for hot maps | A naive Bayes classifier; "cipher" is evocative but misleading |
| "SBT/RT crossover signal theory" (Doc 12) | Analog noise resilience simulation; completely disconnected from the classifier (no classifier uses RT encoding) |
| "TST butterfly cascade" (Doc 03) | Multi-resolution ternary transform; early speculation, never validated in the mature pipeline |
| "Eigenvalue spline framework" (Doc 18) | Eigenvalue-ordered block processing; tested but showed no improvement over IG ordering |

---

## 8. Paper Assessment

### 8.1 The Main Paper (sstt-paper-draft.md) — PUBLISH

**Title:** "Classification as Address Generation: Zero-Parameter Ternary Cascade Inference Exceeding Brute k-NN Accuracy at 1200x Compression"

**Verdict:** This is a publishable paper. The claims are validated, the methodology is sound, and the writing is clear.

**Strengths:**

- Abstract is tight, accurate, and makes exactly the right claims
- Method section (Section 3) is complete and reproducible — another researcher could implement this from the paper
- K-invariance (Section 5.1) is the star result and is presented with a clean table
- Error mode decomposition (Section 5.2) is good methodology that reviewers will appreciate
- Cross-dataset generalization (Section 5.5) strengthens the central claim
- Channel ablation (Section 5.4) shows the system is understood, not just measured
- Compute profile (Section 5.3) provides architectural insight beyond this specific system
- Appendix B (`_mm256_sign_epi8` proof) is a nice addition
- The paper correctly does NOT claim CIFAR-10, Lagrangian methods, or 224x224 scaling

**Weaknesses to fix before submission:**

1. **Title is too long.** "Zero-Parameter Ternary Cascade Classification via Address Generation" or similar. Current title is 18 words.

2. **Lead with the constraint, not the accuracy.** The abstract opens with "97.27% accuracy on MNIST." It should open with "zero learned parameters." The accuracy is unremarkable; the constraint is remarkable.

3. **"Field-theoretic" language in Section 3.7.** Rename to "Topological Ranking Features" without the field-theory framing. Describe the divergence as "discrete divergence of the gradient field" and the centroid as "flood-fill centroid." Drop "Green's theorem" — it's technically correct but will irritate reviewers.

4. **Test-set leakage in Section 6.2.** The dot-product weight grid was determined by "grid search over 125 combinations, each a single test-set evaluation." The val/holdout protocol (Doc 35) confirms the weights are identical under val split, but the paper should present them as val-derived from the start. As written, a careful reviewer will flag this.

5. **Missing baselines.** The paper compares only to brute kNN. It needs at least:
   - Random forest on the same ternary block features (to show the cascade matters, not just the features)
   - SVM on raw pixels (to establish the non-neural-network landscape)
   - A simple 1-hidden-layer MLP (to show what "adding learned parameters" buys)
   - Best published zero-parameter or few-parameter methods on MNIST, if any exist

6. **References incomplete.** "[To be completed]" — needs proper citations including modern non-parametric baselines.

7. **The Kalman framing in Section 3.7.** "Structurally identical to Kalman gain computation" — it's `const / (MAD + const)`. That's a sigmoid-like adaptive weight with one parameter. Calling it Kalman gain implies a state estimation framework. Just call it "MAD-based adaptive weighting."

8. **No statistical significance analysis.** 97.27% on 10,000 images has a 95% confidence interval of roughly +/- 0.32pp (binomial). The paper should report this to show that the improvement over brute kNN (96.12%) is statistically significant.

**Recommended venue:** Workshop paper at ICML or NeurIPS (efficient inference track, non-standard architectures), or full paper at a systems conference (SysML, USENIX ATC). The contribution is real but narrow — it's a systems result about what's achievable under extreme constraints, not a new learning theory or a state-of-the-art accuracy claim.

### 8.2 Paper 1: Adaptive Tiered Inference (01-adaptive-router-systems.md) — FOLD INTO MAIN

**Claimed contributions:**
1. Free confidence signal from vote concentration
2. Three-tier hierarchy (Fast/Light/Deep)
3. Hardware-aware cache-line optimization

**Verdict: Not a standalone paper.** The result is legitimate (96.50% at 0.67ms, zero false positives on Tier 1), but this is a deployment optimization of the main system, not an independent contribution. The "free confidence signal" is checking vote unanimity — one comparison. That's an engineering trick, not a research contribution requiring its own paper.

The main paper draft already covers the router in Section 3.8. It belongs there.

**If forced to justify as standalone:** It would need (a) comparison to other adaptive inference systems (early exit in neural networks, cascade classifiers a la Viola-Jones), (b) evaluation on more datasets, and (c) a theoretical analysis of when unanimity-gating is optimal. Currently it has none of these.

### 8.3 Paper 2: Lagrangian Structural Skeletons (02-lagrangian-skeletons-vision.md) — DROP OR REWRITE

**Claimed contributions:**
1. Discrete Lagrangian vorticity (curl on ternary fields)
2. Structural particle tracing (spawning particles at high-gradient points)
3. Integral vs. differential law (integral measures beat differential on quantized fields)

**Verdict: The framing does not match the results. This is the weakest paper.**

**Problem 1: The headline claim is false.** "Broke the 50% ceiling using topological skeletons as gating logic." The 50.18% CIFAR-10 result comes from the stereo-vote -> RGB Gauss map cascade (Doc 43). The Lagrangian operators — curl, vorticity, particle tracing — were tested independently and **failed**:

- Navier-Stokes stream function: ~10% on CIFAR-10 (random chance). Documented in `sstt_navier_stokes.c`.
- Taylor jet (2nd-order surface curvature): ~54% on CIFAR-10 (barely above stereo baseline). Documented in `sstt_taylor_specialist.c`.
- Doc 58 is the autopsy that confirms these failures.

The paper attributes a result to a method that did not produce it.

**Problem 2: "Integral vs. Differential Law" is empirical, not proven.** The observation that integral operators (divergence sums, flood-fill) are robust under ternary quantization while differential operators (Hessian, Laplacian, curl) are brittle is genuinely interesting. But it's an empirical observation on two datasets at one quantization level. To be publishable as a "law," it needs a theoretical bound: at what quantization level does second-order estimation break down? What is the SNR requirement for curl to be informative? Currently the paper presents an observation as a theorem.

**Problem 3: "Pose invariance" is claimed without evidence.** No rotation experiments. No translation experiments. No pose-varied dataset. The claim is aspirational.

**Problem 4: The language oversells the math.** "Lagrangian structural analysis," "fluid-dynamic operators," "rotational energy cores" — this is finite differences on a 28x28 or 32x32 integer grid. A reviewer in computational physics would find this imprecise. A reviewer in ML would find it obscure. Neither would be impressed.

**Recommendation:** If the integral-vs-differential observation is to be published, reframe as "Discrete Field Operators in Ternary Space: An Empirical Study." Drop all Lagrangian/fluid-dynamics framing. Present the observation honestly: integral operators (divergence, centroid by flood-fill) work under heavy quantization; differential operators (curl, Hessian) do not. Provide quantitative measurements of SNR degradation at different quantization levels. This could be a short workshop paper with proper controls.

### 8.4 Paper 3: Hierarchical Scaling to 224x224 (03-hierarchical-scaling-engineering.md) — DROP

**Claimed contributions:**
1. Linear-quadratic scaling law (model size linear in knots, resolution quadratic)
2. Sparse uniform sampling beats variance-guided sampling
3. Multi-scale Bayesian fusion at 224x224

**Claimed results:**
- 0.14ms latency at 224x224
- 1.61MB model size (L3-resident)
- 98.5% accuracy retention
- +0.5pp from hierarchical fusion on Fashion-224

**Verdict: The numbers do not support the claims.**

**Problem 1: "98.5% accuracy retention" is misleading.** Retention relative to what? The full hot map at 224x224 achieves 11.60% on MNIST-224 (Doc 62). The sparse version achieves 11.35%. That's 97.8% retention — of an 11.6% classifier. The brute kNN baseline at 224x224 is 92.67%. The hot map achieves **12.5% of what brute kNN achieves.** Reporting "98.5% retention" without this context is misleading. Reviewers will compute this ratio immediately.

**Problem 2: Fashion-224 is more defensible but still weak.** Full hot map: 46.77%. Brute kNN: 78.17%. The SSTT system achieves 59.8% of the brute baseline. That's an interesting negative result — the architecture doesn't scale to high resolution because nearest-neighbor upscaling adds no information — but it contradicts the paper's success framing.

**Problem 3: Speed is meaningless without accuracy.** "0.14ms per query" for a classifier that's wrong 88% of the time on MNIST-224 is not a useful number.

**Problem 4: The "scaling law" is definitional.** "Model size scales linearly with knots, resolution scales quadratically" is the definition of bilinear interpolation on a grid. This is not a discovered law; it's how grids work.

**Problem 5: The variance-guided negative result is interesting but undersold.** The finding that high-variance regions are spatially correlated and create coverage gaps when used for knot placement — that's a genuine insight about sampling strategies. But it's one paragraph, not a paper.

**Recommendation:** Archive the 224x224 work as exploration. The negative insights (upsampling adds no information, variance-guided sampling creates gaps) deserve one paragraph each in a limitations section of the main paper. Do not publish as standalone.

---

## 9. Validation Integrity

### 9.1 What Was Done Right

- **Val/holdout protocol (Doc 35):** Applied retroactively to check for test-set leakage. MNIST weights are identical under val split — confirmed no overfitting. Fashion weights differ slightly (divergence weight drops from 200 to 50, Kalman scale increases) — honest correction applied.
- **Sequential processing check:** Discovered that +0.03pp from Bayesian sequential on MNIST was test-set noise. Corrected to zero. Fashion sequential (+1.14pp) confirmed as real.
- **Red-team validation (Doc 54):** Tiered router stress-tested on Fashion-MNIST. Discovered Tier 1 needs disabling for CIFAR-10 (confidence "Death Valley"). Honest.
- **Brute baselines (Docs 62/63):** Added after initial 224x224 work to make numbers interpretable. Self-correction.
- **Error autopsy applied to own system:** Mode A/B/C decomposition identifies what the system cannot fix, not just what it gets right.

### 9.2 What Still Needs Fixing

- **Dot-product weight grid search used test set.** Val/holdout confirms weights are identical, but the paper should present them as val-derived from the start.
- **No confidence intervals reported.** 97.27% on 10,000 images needs error bars.
- **No comparison to non-kNN baselines.** Random forest, SVM, and a simple MLP are needed to position the result.
- **Topological feature weights from ablation, not val.** topo1-9 ablation was on test set. Doc 35 val/holdout confirms no leakage, but the methodology should be val-first.

---

## 10. The Honest Ceilings

### MNIST

- **Current best:** 97.27% (validated)
- **Mode A ceiling:** ~98.6% (if all ranking errors fixed, retrieval failures remain)
- **True architectural ceiling:** ~99-99.2% (based on oracle experiments)
- **What it takes to exceed ceiling:** Better block signatures or larger inverted lists to fix Mode A (1.7% of errors)
- **Dominant confusion pairs:** 4/9 (closed-top 4 resembles 9), 3/5 (similar stroke structure), 1/7 (single stroke angle)

### Fashion-MNIST

- **Current best:** 85.68% (validated)
- **Brute kNN baseline:** 76.86%
- **SSTT lift:** +8.82pp
- **Why it's harder:** Overlapping texture and edge geometry between classes (shirt vs. t-shirt, pullover vs. coat)
- **What it takes to improve:** Better texture discrimination at the block level; current 3x1 blocks are too small for clothing patterns

### CIFAR-10

- **Current best:** 50.18% (5x random)
- **Why it can't go higher:** Inter-class Gauss distance analysis — cat/dog distance is 214 (nearly identical), airplane/frog is 2145 (10x more separable). At 32x32, visually similar classes cannot be distinguished without learned representations.
- **Honest assessment:** The architecture has a hard wall on natural images. 55-60% would require features that a zero-parameter system cannot provide.

### 224x224

- **MNIST-224:** 11.60% (12.5% of brute kNN). Mode collapse. Not viable.
- **Fashion-224:** 46.77% (59.8% of brute kNN). Feature space usable but severely underfitted.
- **What it takes to make 224x224 work:** Genuinely higher-resolution training data (not upsampled copies), and full cascade scaling (not just hot maps).

---

## 11. Repository Health

### Code

- **Compiles cleanly:** `make` and `make experiments` build all 60+ targets with `-Werror`
- **CI passes:** GitHub Actions with AVX2 detection and Haswell fallback
- **Self-contained:** Each C file is a standalone executable with no external dependencies beyond libc + AVX2
- **Code duplication:** Heavy. Boilerplate (data loading, ternary quantization, block encoding) is copy-pasted across 110+ files. A shared library would reduce maintenance burden but this is a research notebook, not production code.
- **Memory safety:** `malloc` return values unchecked in many files. Not a correctness issue for research use (OOM will crash either way), but would fail a production audit.
- **Test coverage:** Minimal. One integration test (`tests/test_kdilute_simple.c`). No unit tests. The ablation series (topo1-9) serves as de facto regression testing.

### Documentation

- **Volume:** 500KB+ across 64 docs. Unusually thorough for a research notebook.
- **Honesty:** Negative results documented with root causes. Self-corrections applied (val/holdout, brute baselines). Overclaims identified in self-audit (Doc 60).
- **Organization:** Could be tighter. 64 docs is too many; 15-20 consolidated documents would be more navigable.

### Build System

- **Makefile:** Clean, well-structured. 60+ targets. Data download with SHA-256 verification.
- **CI:** GitHub Actions, Ubuntu latest, AVX2 detection. Compiles core + experiments. Verifies 20+ key binaries.
- **Dependencies:** GCC with AVX2 support, GNU Make, curl. Nothing else.

---

## 12. Recommendations

### For Publication

1. **Submit the main paper** (`sstt-paper-draft.md`) after fixing the weaknesses identified in Section 8.1.
2. **Add baselines:** Random forest, SVM, 1-layer MLP on same features and on raw pixels.
3. **Add confidence intervals** to all accuracy claims.
4. **Present weights as val-derived** throughout, even where test-derived weights happen to be identical.
5. **Cut "field-theoretic" language.** Describe operations plainly. Let reviewers judge.
6. **Target venue:** Workshop at ICML/NeurIPS or full paper at SysML/USENIX ATC.

### For the Repository

1. **Trim docs from 64 to ~15-20 consolidated documents.** Archive originals for provenance.
2. **Strip LMM framing** from all active documentation.
3. **Trim CIFAR-10 files from 45+ to 3-4** that tell the story. Archive the rest.
4. **Extract shared library** for boilerplate (data loading, quantization, block encoding, dot product) to reduce copy-paste across 110+ files.
5. **Add a proper test suite** if the code will be maintained beyond the paper.

### For Intellectual Honesty

1. **Do not publish Papers 2 or 3.** Paper 2 misattributes results. Paper 3 misframes a negative result as a success. Both would damage credibility if submitted.
2. **The integral-vs-differential observation** from Paper 2 can be salvaged as a workshop note with proper framing and controls.
3. **The 224x224 negative insight** (upsampling adds no information) deserves one paragraph in the main paper's limitations section.
4. **CIFAR-10 at 50.18% is a legitimate result** when honestly framed as a boundary test. Include it as a short section in the main paper, not as a headline.

---

## 13. Summary Table

| Category | Items | Assessment |
|----------|-------|------------|
| **Novel** | K-invariance, `sign_epi8` trick, Mode A/B/C decomposition, Encoding D Pareto, free confidence map | Real contributions; paper-worthy |
| **Useful** | 87% scatter-write profile, Fashion generalization, negative results archive, topo1-9 ablation, BG discrimination, WHT trap, three-tier router | Solid engineering; supports main claims |
| **Understated** | Zero-parameter constraint as headline, K-invariance centrality, Fashion-MNIST lift, 98.6% fixable ceiling | Needs repositioning in paper and README |
| **Fluff** | LMM framing, field-theoretic language, 224x224 as success story, 45+ CIFAR-10 variants, 64-doc count, MESI/cipher/SBT-RT metaphors | Cut or archive; dilutes core claim |
| **Paper-worthy** | Main paper draft (with fixes) | One paper, not three |
| **Drop** | Paper 2 (Lagrangian), Paper 3 (Scaling) | Misattribution and misframing respectively |
| **Fold in** | Paper 1 (Router) | Already covered in main paper Section 3.8 |
