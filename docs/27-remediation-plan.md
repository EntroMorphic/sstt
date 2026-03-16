# Contribution 27: Remediation Plan

Compress SSTT into a diamond. Keep everything, but reorganize so the
signal dominates and the noise is preserved without obscuring it.

---

## Principles

1. **Lead with the result, not the journey.** The README should state the
   headline claim in one sentence. The journey belongs in the changelog.

2. **Archive, don't delete.** Fluff and early explorations are preserved
   in `archived/` with a one-line pointer from the main docs. Nothing is
   lost; it's just not in the reader's path.

3. **Name things accurately.** "Hot map" not "cipher." "Lookup table
   classifier" not "decryption." "Hypothesis-driven methodology" not
   "Lincoln Manifold Method."

4. **Let the numbers talk.** The Pareto frontier, the K-invariance, the
   98.6% ceiling, the 87% scattered-writes bottleneck — these findings
   speak for themselves without metaphor.

---

## Phase 1: Archive

Move the following to `archived/docs/`, preserving their original
filenames and numbering. Add a one-line entry in `archived/README.md`
explaining what each contains and why it was archived.

| File | Reason |
|------|--------|
| `docs/03-tst-butterfly-cascade.md` | Early exploration before architecture solidified |
| `docs/04-sdf-cipher.md` | Early exploration; SDF approach not carried forward |
| `docs/12-sbt-rt-crossover-signal-theory.md` | Disconnected from classifier; separate research interest |
| `docs/18-eigenvalue-spline-framework.md` | Speculative scaling framework with no empirical validation |
| `docs/20-lmm-pass-composite-architecture.md` | Genuine architectural analysis buried under LMM branding |

For doc 20: extract the useful content (confidence-gated hybrid concept,
hot-map-as-collapsed-cascade insight, +0.54pp IG prediction miss) into a
new focused doc. Archive the original with LMM framing intact.

---

## Phase 2: Rename and Reframe

### 2a. Strip the "cipher" metaphor

Rename references to "hot map cipher" → "hot map" across all docs.
Rewrite doc 02's title and framing:

- **Before:** "The Hot Map Cipher" / "encrypted into a frequency table" /
  "decryption via table lookup"
- **After:** "The Hot Map: Naive Bayes via Ternary Block Frequencies" /
  "aggregated into a frequency table" / "classification via table lookup"

The implementation description, performance numbers, and cache analysis
are unchanged.

### 2b. Strip the LMM framing

- **CONTRIBUTING.md:** Replace "Lincoln Manifold Method" with a plain
  description of the actual methodology: "State your hypothesis before
  writing code. Record what the data says, not what you hoped. Document
  failures with the same rigor as successes."

- **README.md:** Remove LMM reference in the Contributing section.

- **All docs:** Remove "LMM" references where they appear as framing
  (keep any reference to specific architectural insights that happen to
  appear in doc 20).

### 2c. Demote "MESI: The 12th Primitive"

In doc 14 (fused kernel), remove the "12th primitive" framing. Keep the
sentence about read-only hot maps being safe for concurrent multi-core
use. Move it to a brief "Concurrency" note at the end instead of
elevating it to a numbered primitive.

### 2d. Renumber contributions

After archiving 5 docs (3, 4, 12, 18, 20), the remaining 22 docs should
be renumbered to close the gaps. This is a significant rename operation.

**Alternative (recommended):** Keep original numbering. Add a
`docs/INDEX.md` that lists only the active contributions in logical
order, grouped by theme. The archived docs retain their numbers; the
index simply omits them. The CHANGELOG already references by number, so
renumbering would break those references.

---

## Phase 3: Bring the Understated to the Front

### 3a. Rewrite README.md

**Current opener:** "A pure C, dependency-free image classifier built
entirely from ternary arithmetic and table lookups."

**Proposed opener:**

> SSTT achieves 96.44% on MNIST with zero learned parameters — no
> gradient descent, no backpropagation, no floating point at inference.
> Classification is integer table lookups, AVX2 byte operations, and
> add/subtract. The entire model fits in L2 cache.

The rest of the README should follow this structure:

1. **Headline result** (one sentence, above)
2. **Pareto frontier table** (all operating points from 1.9μs to 1.1ms)
3. **Architecture diagram** (keep existing, it's good)
4. **Key findings** (rewrite to lead with the understated items)
5. **Quick start** (keep existing)
6. **Experiments table** (keep existing)
7. **Documentation** (point to new `docs/INDEX.md`)

### 3b. Write the Pareto frontier as a first-class result

Add to the Results table in README.md:

| Method | MNIST | Fashion | Latency | Use Case |
|--------|-------|---------|---------|----------|
| Bytepacked Bayesian | 91.83% | 76.52% | 1.9 μs | Embedded real-time |
| Pentary hot map | 86.31% | 77.45% | 4.7 μs | Embedded real-time |
| Bytecascade 4-probe | 96.07% | — | 610 μs | Interactive |
| Bytecascade 8-probe | 96.28% | 82.89% | 930 μs | Interactive |
| Oracle v2 (routed) | 96.44% | — | 1.1 ms | Batch |

Note the latency column is per-image, not per-10K-batch. Convert the
existing batch timings.

### 3c. Rewrite Key Findings

Replace the current Key Findings section with one that leads with the
understated items:

1. **Zero learned parameters.** Every weight is derived from closed-form
   mutual information or grid search over {K, k, probes}. No optimizer.

2. **Near-perfect retrieval.** The inverted-index vote puts the correct
   class in the top-50 candidates for 99.3% of images (K-invariant:
   K=50 = K=1000). The problem is ranking, not retrieval.

3. **98.6% fixable ceiling.** 98.3% of errors occur at the dot-product
   ranking step. Fixing all ranking errors would yield ~98.6%. The vote
   architecture is effectively solved.

4. **87% of compute is memory access.** Vote accumulation (scattered
   writes to 240KB) dominates the pipeline. Dot products cost ~1%.
   The bottleneck is the memory access pattern, not arithmetic.

5. **Generalizes without tuning.** 82.89% on Fashion-MNIST using the
   same code, parameters, and architecture. IG weights automatically
   redistribute across channels for different visual domains.

6. **Sub-2μs operating point.** The bytepacked Bayesian hot map
   classifies at 1.9μs/image (91.83% MNIST) with a 16MB L2-resident
   model — competitive with optimized neural net inference on embedded
   hardware.

Keep the existing findings (routing works, pentary vs ternary, Hamming-2
hurts, calloc cold-miss) as a "Detailed Findings" subsection below.

### 3d. Write docs/INDEX.md

A new index file that groups active contributions by theme, replacing
the numbered list in the README:

```
# SSTT Documentation Index

## Core Architecture
- 01: sign_epi8 ternary multiply
- 02: Hot map — naive Bayes via ternary block frequencies
- 05: Background value discrimination
- 06: Ternary gradient observers
- 07: IG-weighted block voting
- 08: Multi-probe Hamming-1 expansion
- 11: Zero-weight cascade (the full pipeline)

## Empirical Analysis
- 10: Haar vs WHT trap
- 13: WHT brute kNN (Pareto ceiling)
- 15: Fashion-MNIST generalization
- 23: Cascade autopsy + multi-channel dot + compute profile
- 24: Benchmark catalogue (SWO analysis)
- 25: Oracle — multi-specialist routing

## Optimization
- 09: WHT as ternary-native transform
- 14: Fused kernel architecture
- 16: Spectral coarse-to-fine (negative/mixed)
- 19: Composite solutions analysis
- 22: Bytepacked cascade

## Negative Results
- 17: Ternary PCA (failed, -16 to -23pp)
- 21: Soft-prior cascade (failed, prior too weak)

## Meta
- 26: Audit — novel, useful, fluff, understated
- 27: Remediation plan (this document)

## Archived (in archived/docs/)
- 03: TST butterfly cascade (early exploration)
- 04: SDF cipher (early exploration)
- 12: SBT/RT crossover signal theory (disconnected)
- 18: Eigenvalue spline framework (speculative)
- 20: LMM architectural analysis (useful content extracted)
```

---

## Phase 4: Extract Value from Archived Docs

### 4a. New doc from doc 20's useful content

Create a new contribution (e.g., doc 28) titled "Confidence-Gated Hybrid
Architecture" that extracts from doc 20:

- The hot-map-as-collapsed-cascade insight (the spectrum from fully
  collapsed to fully expanded)
- The confidence-gating architecture (fast path for easy images, cascade
  fallback for hard ones)
- The prediction that IG pre-baking would give +15pp (actual: +0.54pp)
  and why (hot map collapses per-image identity)
- The implementation plan for fused multi-probe

Strip all LMM framing. The architectural analysis stands on its own.

### 4b. archived/README.md

```
# Archived Documentation

These documents are preserved for historical context. They represent
early explorations, disconnected research interests, or contributions
whose useful content has been extracted into the main docs.

- 03: TST butterfly cascade — multi-resolution ternary shape transform
- 04: SDF cipher — signed distance field topological compression
- 12: SBT/RT crossover — analog noise resilience of ternary encodings
- 18: Eigenvalue spline — speculative scaling framework for large images
- 20: LMM pass — architectural analysis (content extracted to doc 28)
```

---

## Phase 5: Minor Cleanups

### 5a. CHANGELOG.md

Add a `[0.2.0]` section header for the oracle/parallel/benchmark work.
The current changelog ends at `[0.1.0]` but the VERSION file now says
0.2.0.

### 5b. CONTRIBUTING.md

Replace:

> The project follows the **Lincoln Manifold Method**: understand before
> you build.

With:

> The project follows a hypothesis-driven methodology: state your
> hypothesis before writing code, record what the data says (not what you
> hoped), and document negative results with the same rigor as positive
> ones.

### 5c. Contribution 2 title

Rename `02-hot-map-cipher.md` to `02-hot-map-naive-bayes.md` (or keep
the filename and change only the internal title to avoid breaking any
links).

### 5d. Contribution 14 MESI section

Replace the "MESI: The 12th Primitive" section with:

> **Concurrency.** The hot maps are read-only at inference time. Under
> MESI, multiple cores classify concurrently from L2/L3 without bus
> traffic — no locks, no atomics. Model updates propagate automatically
> via cache invalidation.

---

## Execution Order

| Step | Phase | Effort | Risk |
|------|-------|--------|------|
| 1 | Create `archived/docs/` and `archived/README.md` | Low | None |
| 2 | Move 5 docs to archive | Low | None |
| 3 | Write `docs/INDEX.md` | Low | None |
| 4 | Rewrite README.md | Medium | Tone matters — review carefully |
| 5 | Write doc 28 (extracted from doc 20) | Medium | Must not lose useful content |
| 6 | Strip cipher/LMM/MESI framing | Low | Grep + targeted edits |
| 7 | Rewrite CONTRIBUTING.md | Low | None |
| 8 | Update CHANGELOG.md with 0.2.0 section | Low | None |

Steps 1-3 can be done in parallel. Step 4 (README rewrite) is the
highest-stakes change and should be reviewed before committing.

---

## What This Achieves

**Before:** 25 numbered contributions in chronological order, buried
headline results, metaphorical framing that obscures technical content,
inflated contribution count, architectural insights mixed with
disconnected research.

**After:** 22 active contributions grouped by theme, headline results
in the README's first sentence, plain technical language throughout,
5 archived explorations preserved with clear pointers, a Pareto frontier
table as a first-class result, and the understated findings (zero learned
parameters, near-perfect retrieval, 98.6% ceiling, memory-bound
bottleneck) in the reader's immediate path.

The diamond is already in this repo. It just needs the matrix cleared.

---

## Files

This document: `docs/27-remediation-plan.md`
Audit: `docs/26-audit-novel-useful-fluff-understated.md`
