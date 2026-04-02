# Contribution 60: Comprehensive Project Audit — March 2026

A full, independent audit of the SSTT project covering all 59 contributions,
every source file, and every documented experimental result. This supersedes
the earlier audit in doc 26, which covered only contributions 1–25.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Architecture Overview](#2-architecture-overview)
3. [Complete Results Ledger](#3-complete-results-ledger)
4. [Novel Contributions](#4-novel-contributions)
5. [Useful Contributions](#5-useful-contributions)
6. [Fluff](#6-fluff)
7. [Understated Findings](#7-understated-findings)
8. [Paper-Worthy Analysis](#8-paper-worthy-analysis)
9. [The Honest Ceiling](#9-the-honest-ceiling)
10. [Recommendations](#10-recommendations)

---

## 1. Project Identity

SSTT is a zero-learned-parameter image classifier written entirely in C. It
classifies images using a pipeline of ternary quantization, inverted-index
retrieval, and geometric ranking. No gradient descent. No backpropagation.
No floating point at inference. No matrix multiplication.

The name "SSTT" traces back to "Sign-Score Ternary Transform," reflecting the
core computational primitive: `_mm256_sign_epi8`, an AVX2 instruction that
performs exact ternary multiplication in a single cycle.

The project spans approximately three years of systematic experimental
research documented across 59 numbered contributions, 80+ C source files,
and a complete experimental changelog. Its scope covers:

- **Four datasets:** MNIST (60K train / 10K test), Fashion-MNIST (same
  dimensions), CIFAR-10 (50K train / 10K test, 32×32 RGB), and a synthetic
  224×224 upscale of MNIST/Fashion.
- **Three major architectural paradigms:** Bayesian hot maps (O(1) lookup
  tables), vote-then-refine cascades (inverted-index retrieval + kNN ranking),
  and geometric rankers (Gauss maps, Lagrangian operators, topological
  features).
- **One cross-cutting constraint:** zero learned parameters at every stage.

The project is not a paper. It is a research notebook that happens to contain
enough validated, clean results to support one strong paper, and enough
documented negative results to save future researchers significant time.

---

## 2. Architecture Overview

The mature system (as of contribution 51) is a three-tier adaptive router.
The description below reflects the production state; the path to get here
is covered in sections 4 and 5.

### 2.1 Ternary Quantization

The input (any grayscale or RGB image) is quantized to {-1, 0, +1} per
pixel using fixed thresholds (85/170 for 8-bit pixel values). Three
channels are produced:

- **Pixel channel:** the quantized raw intensity.
- **Horizontal gradient channel:** `clamp(t[y][x+1] - t[y][x])` in ternary
  space, capturing left-right edge direction.
- **Vertical gradient channel:** `clamp(t[y+1][x] - t[y][x])` in ternary
  space, capturing top-bottom edge direction.

A fourth channel — the **bytepacked encoding** — fuses all three into a
single byte: 2 bits each for pixel majority trit, h-grad majority trit,
v-grad majority trit, and transition activity. This gives 256 distinct
block signatures rather than 27.

Per-channel background thresholds are separate: BG_PIXEL=0 (all-dark),
BG_GRAD=13 (flat region). Blocks matching background values are skipped.

### 2.2 Block Encoding

The image is decomposed into overlapping 3×1 horizontal blocks at 252
positions. Each block encodes a 3-trit signature (for the 27-value encoding)
or a 1-byte signature (for the bytepacked encoding). The block geometry is
the same across all three channels.

### 2.3 Inverted Index

Training time: for each block position × signature value, accumulate the
list of training image IDs that share that signature at that position.
This is a standard inverted index over block features.

Inference time: for each block in the query image, look up its inverted
list and add IG-weighted votes to a per-training-image accumulator. After
all blocks, the top-K training images by vote count are the candidates.

**Information-gain weighting:** Each block position is assigned a weight
proportional to its discriminative power (mutual information between block
signature and class label). High-IG positions contribute more votes.

**Multi-probe Hamming-1:** For each block, also look up the 6 Hamming-1
trit-flip neighbors at half weight. Handles single-trit quantization noise.

**3-eye stereoscopic retrieval (CIFAR-10):** Three independent quantization
perspectives — fixed 85/170 thresholds, adaptive P33/P67 thresholds, and
a gradient-dominant encoding — each run the inverted index independently.
Their vote counts are summed. Achieves 99.4% recall at top-200 candidates
on CIFAR-10.

### 2.4 Dot-Product Ranking

The top-K candidates (K=50 suffices) are ranked by a weighted multi-channel
ternary dot product:

```
score = 256 × dot(pixel) + 0 × dot(h-grad) + 192 × dot(v-grad)
```

The weight grid (px=256, hg=0, vg=192) was determined empirically: h-grad
adds noise on MNIST; v-grad captures stroke direction and helps. The dot
product is computed via `_mm256_sign_epi8` for efficiency.

### 2.5 Geometric Ranking (CIFAR-10 and topology)

For MNIST/Fashion, topological features augment the dot product ranking:

- **Gradient divergence** (discrete Green's theorem): flux through the
  boundary of each region identifies enclosed shapes.
- **Enclosed-region centroid:** center of mass of the topological skeleton.
- **Horizontal profile:** row-wise structural intensity.
- **Grid divergence (2×4):** spatial distribution of gradient flux.
- **Kalman adaptive weighting:** per-query weight adjustment based on the
  MAD (median absolute deviation) of candidate divergence scores.
- **Bayesian sequential update:** evidence accumulation with decay (S=2),
  processing candidates in quality order.

For CIFAR-10, the ranking step is the **RGB grid Gauss map**: each pixel's
gradient vector is mapped to the unit sphere, and a spatial 4×4 grid
histogram of sphere positions (16 regions × 45 bins × 4 channels = 2880
features) is compared via L1 distance. k=7 majority vote.

### 2.6 Three-Tier Router

The production system routes each query based on vote concentration:

- **Tier 1 (Fast, ~1μs):** If the top vote count exceeds a unanimity
  threshold (198/200 for clean data), return the plurality immediately.
  Covers ~10–15% of images at 99.9% accuracy.
- **Tier 2 (Light, ~1.5μs):** If moderate confidence, consult the Dual
  Hot Map (fused pixel + Gauss frequency tables). Covers ~15–18% at ~90–97%
  accuracy.
- **Tier 3 (Deep, ~1ms):** All remaining images go through the full
  inverted-index → geometric ranking pipeline.

Global latency: 0.67ms MNIST, 0.88ms Fashion. Global accuracy: 96.50%
MNIST, 83.42% Fashion.

---

## 3. Complete Results Ledger

All figures are on the full 10,000-image test set unless noted.

### 3.1 MNIST

| Method / Contribution | Accuracy | Latency | Notes |
|---|---|---|---|
| Dual Hot Map (pixel only) | 73.23% | ~1.5μs | Baseline hot map |
| Bytepacked Bayesian | 91.83% | 1.9μs | Encoding D, O(1) lookup |
| Pentary Hot Map | 86.31% | 4.7μs | 5-level quantization |
| Zero-Weight Cascade (C11) | 96.04% | — | First 96% result |
| WHT brute kNN (C13) | 96.12% | — | Establishes pixel-space ceiling |
| Fused Kernel (C14) | 225K cls/sec | — | 225,000 classifications/sec |
| Bytepacked Cascade (C22) | 96.28% | 930μs | Best single method |
| Oracle v2 (C25) | 96.44% | 1.1ms | Unanimity-gated specialist routing |
| Sequential Field Ranking (C31) | 97.27% | ~1ms | Kalman + grid + Bayes-CfC |
| Val/Holdout Validation (C35) | **97.27%** | ~1ms | Confirmed on held-out split |
| Three-Tier Router (C51) | 96.50% | **0.67ms** | 2x throughput vs full run |

### 3.2 Fashion-MNIST

| Method / Contribution | Accuracy | Notes |
|---|---|---|
| Brute 1-NN (C15) | 76.86% | First generalization test |
| Cascade vote+refine (C15) | 81.00% | No tuning from MNIST |
| Bytepacked Cascade (C22) | 82.89% | Same code, different data |
| Stereoscopic quantization (C39) | 86.12% | +0.44pp over baseline |
| Val/Holdout Validation (C35) | **85.68%** | Confirmed publication figure |
| Three-Tier Router (C51) | 83.42% | At 0.88ms |
| Three-Tier Router T1 slice | **100.00%** | Zero false positives on easy 9.6% |

### 3.3 CIFAR-10

| Method / Contribution | Accuracy | Notes |
|---|---|---|
| Grayscale ternary cascade (C37) | 26.51% | First CIFAR attempt |
| Flattened RGB Bayesian (C37) | 36.58% | Color blocks, position-specific |
| MT4 full stack (C37) | 42.05% | 81-level + topo + Bayesian |
| Stereo + MT4 stack (C41) | 44.48% | 3-eye + all features |
| Grid Gauss map kNN, gray (C42) | 46.06% | Shape geometry, brute kNN |
| Grid Gauss map kNN, RGB k=5 (C43) | 48.31% | Color Gauss |
| **Cascade: stereo → RGB Gauss k=7 (C43)** | **50.18%** | **Best result, 5× random** |
| Taylor Jet (MNIST) | 91.60% | 512-integer compression |
| Taylor Jet (Cat vs Dog) | 54.45% | Binary only; fails on full set |
| Navier-Stokes Stream Function | ~10% | Random; global potential fails |
| Lagrangian Curl + Gauss (slice) | 39.00% | Small data slice only |

### 3.4 224×224 Scaling

| Method / Contribution | Accuracy | Latency | Model Size | Notes |
|---|---|---|---|---|
| Full position-aware map (MNIST-224) | 13.60% | — | 1.7MB | Low absolute accuracy |
| Sparse 6.25% grid (MNIST-224) | 13.40% | — | 0.1MB | 98.5% retention |
| Fashion-224 fine only (C59) | 41.40% | 0.14ms | 1.35MB | — |
| Fashion-224 coarse only (C59) | 40.90% | — | 0.26MB | — |
| Fashion-224 hierarchical (C59) | **41.90%** | **0.14ms** | **1.61MB** | Best scaling result |

---

## 4. Novel Contributions

These are ideas or compositions that would not be obvious to a practitioner
reading the prior art. Ordered by importance.

---

### 4.1 `_mm256_sign_epi8` as Ternary Multiply

**Contribution 1. Source: `src/sstt_mvp.c` and throughout.**

The AVX2 instruction `_mm256_sign_epi8(a, b)` computes `sign(b) × |a|` for
each byte in a 256-bit register. Over the ternary alphabet {-1, 0, +1}:

- If `b[i] > 0`: output `a[i]` (multiply by +1 = identity)
- If `b[i] < 0`: output `-a[i]` (multiply by -1 = negate)
- If `b[i] == 0`: output `0` (multiply by 0 = zero)

This is exact ternary multiplication with no widening, no actual multiply
instruction, and int8 accumulation safe for up to 127 iterations before
overflow. The result is 32 ternary multiply-accumulates per CPU cycle,
processing an entire 32-pixel block in one instruction.

The insight is narrow but has wide reach: every fast classifier in this
repo depends on it. It eliminates the widening-multiply bottleneck that
would otherwise make ternary dot products slower than binary operations.
Prior work on ternary neural networks (TWN, TTN, etc.) implements ternary
multiplication via conditional branches or lookup tables; this approach
exploits a specific x86 semantic that maps exactly to the ternary algebra.

This is publishable as a standalone trick in the context of ternary/integer
inference systems.

---

### 4.2 Vote-Then-Refine Cascade Composition

**Contributions 7, 8, 11, 22, 23. Source: `src/sstt_bytecascade.c`.**

The cascade assembles four techniques that are individually well-known
into a composition with non-obvious emergent properties:

1. **IG-weighted inverted-index voting** (contribution 7): instead of equal
   weight per block, weight each block's votes by its mutual information
   with the class label. Standard in IR; rarely applied to image block
   features.
2. **Multi-probe Hamming-1** (contribution 8): for each block, look up its
   6 trit-flip neighbors at half weight. Standard in LSH literature.
3. **Ternary dot-product ranking** (contribution 11): rank top-K by weighted
   multi-channel ternary dot product. Standard nearest-neighbor refinement.
4. **k=3 majority vote** (contribution 11): standard kNN vote.

What is novel is the composition finding:

**K-invariance:** K=50 through K=1000 produce identical accuracy on the
bytepacked cascade (contribution 22). The top-50 candidates by vote contain
every relevant neighbor out of 60,000 training images. This is a 1200×
compression ratio with zero recall loss. It means the retrieval stage
achieves near-perfect recall and the entire problem reduces to ranking within
a tiny pre-filtered set.

**Additive stacking:** The four components stack additively with no
hyperparameter interaction. Adding each component does not require re-tuning
any prior component. This is empirically verified by the topo1–topo9
ablation series (contributions 29–31).

**The cascade beats brute kNN:** 96.04% (cascade) vs. 95.87% (brute 60K kNN).
This is the headline claim: a pipeline of sparse lookups beats exhaustive
search. It works because the IG weighting selects more discriminative
features than uniform pixel comparison.

---

### 4.3 Cascade Error Autopsy Methodology

**Contribution 23. Source: `src/sstt_diagnose.c`.**

Classifying every misclassification into three mutually exclusive failure
modes, measurable from the existing data structures at inference time:

- **Mode A (1.7% of errors):** Vote miss. The correct class is absent from
  the top-K candidates. Retrieval failed. These cannot be fixed by a better
  ranker.
- **Mode B (64.5% of errors):** Dot override. The correct class is present
  in top-K but the wrong class wins the dot product step. The correct answer
  was retrieved; ranking failed. These are fixable.
- **Mode C (33.8% of errors):** kNN dilution. The dot product step returns
  the correct class first, but the k=3 majority vote dilutes it. Also fixable.

Key quantitative finding: 98.3% of errors occur at the ranking step, not
the retrieval step. The vote architecture has 99.3% recall; the problem is
entirely in distinguishing between retrieved candidates.

This directly implies a fixable ceiling: if all Mode B and C errors were
resolved, accuracy would be approximately 98.6%. Combined with the K-invariance
finding, the autopsy establishes that retrieval is effectively solved and
only ranking remains.

The methodology is architecture-independent and transferable to any
vote-then-refine pipeline. The three-mode taxonomy (retrieval miss /
ranking inversion / vote dilution) would apply to any inverted-index
nearest-neighbor system.

---

### 4.4 Bytepacked Encoding D

**Contribution 22. Source: `src/sstt_bytecascade.c`.**

Packing three ternary channels into one byte:

```
byte = (pixel_majority_trit << 6) | (hgrad_majority_trit << 4) |
       (vgrad_majority_trit << 2) | transition_activity
```

This compresses 3 channels of 27-value blocks into 256-value single-byte
blocks. The result is simultaneously:

- **+0.16pp accuracy** vs. the 27-value cascade
- **3.5× faster** than the 27-value cascade

Speed-accuracy tradeoffs are almost always present. A Pareto improvement
— better on both dimensions simultaneously — is unusual and worth noting.
The explanation: the bytepacked encoding captures cross-channel correlations
(the joint distribution of pixel × h-grad × v-grad at each position) that
the independent-channel approach misses, while the larger vocabulary (256
vs. 27) requires fewer multi-probe lookups to achieve the same noise
resilience.

The K-invariance finding (K=50 to K=1000 identical) emerges most cleanly
in the bytepacked cascade, suggesting it is the most complete representation
of the retrieval problem.

---

### 4.5 Stereoscopic Quantization

**Contribution 39. Source: `src/sstt_cifar10_stereo.c`.**

The discovery that different quantization functions reveal structurally
different information about the same image — and that the difference between
perspectives is itself a signal.

**Fixed 85/170 thresholds** capture scene-level information: a bright image
maps to the top ternary bin regardless of local structure. Classes that
differ by absolute brightness (airplane over blue sky vs. frog on dark
ground) are easily separated.

**Adaptive P33/P67 thresholds** normalize out absolute brightness, capturing
structural information: the local contrast pattern, regardless of global
illumination. Classes that differ by structure rather than brightness
(automobile vs. truck, both similarly bright) benefit.

Empirical evidence from the correlation analysis (doc 39): brightness is
the #1 per-class predictor of success (airplane: +41.3 brightness delta
correct vs. wrong). Per-class tradeoffs are sharp: adaptive quantization
gives +9.8pp on cat and +8.2pp on automobile while taking −17.0pp on
airplane and −12.4pp on ship.

**The insight:** running both pipelines and combining their posteriors via
Bayesian summation captures both perspectives simultaneously. The analogy
to binocular vision is accurate: two projections of the same object, and
the difference between them reveals depth (in this case, the distinction
between scene-level and structure-level features).

Three-eye stereo (fixed + adaptive + gradient-dominant) achieves 99.4%
recall at top-200 on CIFAR-10, a retrieval quality that enabled the
subsequent Gauss map to achieve 50.18%.

---

### 4.6 RGB Grid Gauss Map for CIFAR-10

**Contribution 42. Source: `src/sstt_cifar10_gauss.c`.**

Applying the differential-geometry Gauss map to the image classification
problem: treating the image intensity function z = f(x,y) as a surface,
mapping each pixel's gradient vector (∂f/∂x, ∂f/∂y) to the unit sphere
(the surface normal), and histogramming the resulting sphere distribution.

This is a known operation in the computer vision literature (gradient
orientation histograms, HOG, SIFT all do variations of this). The novel
elements here are:

1. **Intrinsic background suppression:** Flat background has near-zero
   gradient, maps to the sphere's north pole, and always falls in the same
   histogram bin regardless of background color. There is no explicit
   background removal step — it vanishes geometrically.
2. **RGB four-sphere variant:** Running independent spheres on grayscale,
   R, G, and B channels. The color channels carry edge geometry that the
   grayscale sphere misses; RGB Gauss map outperforms grayscale by 2.5pp
   on CIFAR-10.
3. **Grid decomposition (4×4):** 16 spatial regions, each with a 45-bin
   sphere histogram, for 2880 total features. The grid granularity is a
   parameter: too fine loses position tolerance, too coarse loses spatial
   structure. 4×4 (8×8 pixel regions on 32×32 images) is the optimal
   granularity for CIFAR-10.
4. **The two-system finding:** Block-based stereo and grid Gauss map are
   architecturally orthogonal — they see completely different features and
   make different errors. Dog: block system 40%, Gauss system 30% (block
   is better because dog texture is distinctive). Bird: block system 23.6%,
   Gauss system 38.6% (Gauss is better because bird edge geometry is
   distinctive). Combining them achieves 50.18%.

The inter-class Gauss distance matrix (doc 42) provides a quantitative
statement about the limits of 32×32 resolution: cat↔dog distance is 214
in sphere space, vs. airplane↔frog distance of 2145. Cat and dog have
genuinely similar edge geometry at this resolution — no algorithm that
operates on 32×32 images can fully separate them without higher-level
features.

---

### 4.7 Integral > Differential Law in Ternary Space

**Contributions 46, 57, 58. Distributed across sources.**

Empirically demonstrated across six independent experiments that integral
measures are robust under ternary quantization while differential measures
are brittle:

**Integral measures (robust):**
- Gradient divergence (Green's theorem): +43 MNIST errors fixed (C29)
- Enclosed-region centroid: +49 MNIST errors fixed (C29)
- Information gain (mutual information): +15.55pp MNIST (C7)
- RGB color histograms: +10.07pp CIFAR-10 (C37)
- Sphere histogram (Gauss map): +3.83pp CIFAR-10 (C42)

**Differential measures (brittle):**
- Taylor jet (Hessian): 91.6% MNIST (works) / 54.45% CIFAR-10 cat-dog
  binary (fails; collapses near random on full set)
- Navier-Stokes stream function (Poisson equation on curl field): ~10%
  accuracy on CIFAR-10 (random baseline; global potential dominated by
  frame boundaries, not object structure)
- Discrete curl: 14% standalone / 33% with Gauss on small data slice;
  does not improve on full CIFAR-10 test
- Hessian curvature augmentation: mild improvements at best; brittle
  to quantization noise

The theoretical explanation: differential operators require accurate local
derivative estimates, which in turn require continuous or high-resolution
input. Ternary quantization reduces the field to three levels, making
second-order estimates (which require computing the difference of first-order
differences) essentially noise. Integral operators — summing over a region —
are robust to quantization because individual errors average out.

This is a non-obvious finding with clear empirical support. The Taylor jet
result on MNIST (91.6%) vs. CIFAR-10 (failed) demonstrates the failure mode
precisely: MNIST digits have clean, high-contrast edges that survive ternary
quantization well enough for second-order curvature to be meaningful. CIFAR-10
natural images have low-contrast, high-noise edges at 32×32 resolution where
the ternary quantization destroys the curvature signal.

This generalizes: the finding is not SSTT-specific. It applies to any system
that applies analytic operators to heavily quantized feature fields.

---

### 4.8 Confidence Map as Free Byproduct of Retrieval

**Contributions 31, 35. Source: `src/sstt_confidence_map.c`.**

During the retrieval step, the vote distribution across training images is
already computed. The concentration of votes (how many of the top-K images
belong to the same class, and by how much the top class leads the second)
is a free signal about classification difficulty — available at zero
additional cost.

Empirical results:
- A 264-entry confidence map predicts classification difficulty before
  ranking begins.
- **MNIST:** 99.6% accuracy on 22% of images flagged as confident.
- **Fashion-MNIST:** 97.2% accuracy on the confident 50%.
- **Three-Tier Router:** Using vote unanimity threshold 198/200, T1 captures
  10–15% of images at 99.9% accuracy on MNIST, 100.0% accuracy on Fashion.

The zero-false-positive result on Fashion-MNIST T1 (100.0% accuracy on 9.6%
of images) is striking: the unanimity threshold is conservative enough that
it never fires on a wrong answer in this dataset. This makes it suitable for
safety-critical applications where a guaranteed-correct fast path has value.

---

### 4.9 Kalman Adaptive Weighting for Candidate Quality

**Contribution 31. Source: `src/sstt_topo9.c`.**

Using the median absolute deviation (MAD) of candidate divergence scores
as a per-query gain factor for the geometric ranking step. The insight:
when candidates are tightly clustered in feature space (low MAD), they are
genuine nearest neighbors and geometric features should be trusted more;
when candidates are scattered (high MAD), the retrieval was noisy and
geometric features should be trusted less.

This is structurally identical to Kalman filter gain computation (balancing
process uncertainty vs. measurement uncertainty), applied to the confidence
of the retrieval step. It added +5 MNIST errors fixed without adding any
trainable parameters — the gain is computed from the per-query candidate
statistics.

---

## 5. Useful Contributions

These are not novel, but are well-executed, empirically grounded, and save
future researchers real time.

---

### 5.1 Memory Bottleneck Diagnosis

**Contribution 23. Source: `src/sstt_diagnose.c`.**

Phase-timed profiling of the full cascade pipeline. Key finding:

```
87% of total runtime: vote accumulation (scattered writes to 240KB array)
~1%  of total runtime: ternary dot products
~5%  of total runtime: kNN ranking
~7%  of total runtime: encoding and overhead
```

Vote accumulation is slow because it is a scatter-write pattern: each block
lookup accesses a different memory location in a 240KB accumulator array,
thrashing cache lines. The arithmetic (ternary multiply-accumulate) is
nearly free compared to the cache miss cost.

This diagnosis led directly to the pre-allocation fix: `calloc` inside
`select_top_k` was causing a cold-miss penalty of ~70μs per image. Moving
to a pre-allocated histogram eliminated this entirely.

Broader implication: for any inverted-index system on modern hardware, the
memory access pattern (scatter writes) dominates the compute. This is not
SSTT-specific. It applies to document retrieval engines, recommendation
systems, and any sparse vector inner product computation.

---

### 5.2 Background Threshold Discovery

**Contribution 5. Source: `src/sstt_v2.c`.**

The bug: applying the pixel-channel background threshold (BG_PIXEL=0) to
the gradient channels, which have a different background distribution.
A flat image region has a pixel value near 0 (maps to the threshold correctly)
but a gradient value near 13 (the most common gradient value for flat regions,
not 0). Applying BG_PIXEL=0 to gradients caused the system to treat nearly
all gradient blocks as foreground, flooding the inverted index with
uninformative background-region queries.

Effect: gradient channel accuracy went from ~20% (near random) to 60–74%
after setting BG_GRAD=13. This is a real, diagnosable, reproducible bug
with a specific root cause and fix.

The lesson for multi-modal ternary systems is general: each channel has
its own background distribution, and background suppression must be
calibrated per channel. The fix is trivial once diagnosed; the cost of
missing it is large.

---

### 5.3 Haar vs. WHT Trap

**Contribution 10. Source: `src/sstt_v2.c` documentation.**

Standard Haar wavelet implementation has H × Hᵀ ≠ N × I: it is not
orthogonal in the standard sense. WHT (Walsh-Hadamard Transform) satisfies
`<WHT(x), WHT(y)> = N² × <x,y>`, preserving dot-product rankings exactly
up to a scale factor.

Using Haar and expecting dot products to correspond to pixel-space
similarity is a real trap that costs time to diagnose. The WHT property is
used in contribution 13 (WHT brute kNN) to establish the pixel-space
accuracy ceiling: because WHT preserves rankings, the WHT brute kNN result
(96.12%) is equivalent to pixel-space brute kNN.

Documented clearly with the mathematical proof. Saves a week of debugging.

---

### 5.4 Negative Results Archive

The following experiments were tried, failed, and documented with specific
root cause analysis. Each prevents future researchers from repeating the
same dead end:

**Ternary PCA (contribution 17, −16 to −23pp):**
PCA optimizes variance, not class discrimination. Double quantization
(once to ternary, once to quantize the eigenvectors) destroys
orthogonality. Projection destroys the spatial addressing property that
block encoding depends on. Root cause is specific and documented.

**Soft-prior cascade (contribution 21, negative at all configurations):**
A weaker classifier (84% Bayesian hot map) cannot usefully guide a stronger
one (96% cascade) because it introduces more noise than signal. The prior
must be closer to the truth than the system it guides. This is a clean,
general principle.

**Spectral coarse-to-fine (contribution 16, no improvement):**
WHT prototype pruning does not help vs. vote-based filtering. The vote
already captures the spatial evidence that WHT prototypes try to capture,
but with full per-position specificity.

**Tiled routing (contribution 23, hurts):**
Routing test images to specialized sub-cascades via K-means clustering
concentrates confusion rather than resolving it: confused digit pairs
(4/9, 3/5) cluster together. Global cascade always wins over split cascades
because the global IG weights already capture the discriminative information.

**Taylor jet on CIFAR-10 (contribution 57/58, fails):**
Second-order Hessian curvature is effective on MNIST (91.6%) but collapses
near binary-random on CIFAR-10 (54.45% on cat-vs-dog only; inferred failure
on full set). Root cause: ternary quantization at 32×32 resolution destroys
the subtle curvature signal in natural images.

**Navier-Stokes stream function (contribution 58, ~10% = random):**
Solving the Poisson equation on the curl field to find topological skeletons
produces a global potential dominated by image frame boundaries, not object
structure. The long-range correlation assumption of the Poisson equation is
incompatible with localized object features at this scale.

**Variance-guided knot placement (contribution 55, negative):**
Placing sparse sampling knots at high-variance regions (eigenvalue-guided)
was hypothesized to outperform uniform placement. It does not: high-variance
regions are correlated (center of image has highest variance for most
datasets), creating spatial coverage gaps. Uniform placement wins.

**Delta map "maps all the way down" (contribution 36):**
Replacing the cascade's dot-product ranking step with a lookup-table
comparison (a second "hot map" over candidate IDs) added +19/+20 errors
rather than reducing them. The lookup table collapses per-image identity;
the cascade's dot product preserves it.

**Chirality, noise reduction, divergence refinement (contributions 29–31):**
Multiple variations on the topological ranking step were tried and failed:
chirality (handedness of gradient field), dead-zone noise reduction,
bucket quantization of divergence, edge-masked divergence weighting,
divergence histogram. All documented with the precise accuracy delta.

**Sequential vs. non-sequential MNIST (contribution 35):**
Early results showed +0.03pp from sequential candidate processing.
Val/holdout validation confirmed this is within noise. The finding was
retracted. Sequential processing gives +4 MNIST errors fixed, which is
statistically meaningful but not a breakthrough.

---

### 5.5 Progressive Architecture Tree

**Contribution 24 (benchmark catalogue), updated through contribution 51.**

The complete design space is legible through the documented progression:

```
MNIST:
  Hot map (73.23%) → gradient channels (multi-channel ~80%) →
  IG weighting (~88%) → inverted index cascade (96.04%) →
  bytepacked (96.28%) → multi-channel dot (96.44%) →
  topological ranking (97.27%) → tiered router (96.50% at 0.67ms)

Fashion-MNIST:
  Brute 1-NN (76.86%) → cascade (81.00%) → bytepacked (82.89%) →
  stereo (86.12%) → val-confirmed (85.68%)

CIFAR-10:
  Grayscale cascade (26.51%) → RGB (36.58%) → MT4 (42.05%) →
  stereo stack (44.48%) → Gauss map (46.06%) → cascade Gauss (50.18%)
```

Each step represents a documented experiment with an isolated cause.
This progression is the methodological backbone of the project — a textbook
example of incremental controlled experimentation.

---

### 5.6 Fused Kernel Architecture

**Contribution 14. Source: `src/sstt_fused.c`, `src/sstt_fused_c.c`.**

An implementation that processes raw pixels to class label in one pass with
zero intermediate heap allocation. Cache profile:

- <1KB code
- 232-byte working set (per-image scratchpad)
- 1.3MB model (fits in L2/L3)
- 225,000 classifications/second on standard x86 hardware

The MESI Shared state observation (read-only model data is cache-coherent
across cores) is not novel, but the engineering of a classifier that fits
entirely in cache is a concrete implementation achievement. The C reference
version (75 lines) is a clean proof of concept.

---

### 5.7 Fashion-MNIST Generalization Test

**Contribution 15, confirmed at contribution 35.**

The same architecture, same parameters, same code path achieves 85.68% on
Fashion-MNIST with no tuning whatsoever beyond pointing at different data
files. Channel importance reverses: on MNIST, h-grad adds noise (weight=0
in the optimal grid); on Fashion, h-grad matters because clothing textures
have strong horizontal structure that digit strokes do not.

This reversal happens automatically through IG re-weighting: the IG
computation, which runs on the training set, discovers that h-grad positions
are more informative on Fashion textures than MNIST digit strokes. No
human intervention. No hyperparameter re-tuning.

This is the strongest evidence that the architecture captures real structure
rather than overfitting to MNIST digit topology.

---

## 6. Fluff

These elements add narrative weight without technical substance. They risk
undermining the project's credibility with a technical audience and should
be addressed before publication.

---

### 6.1 Lincoln Manifold Method (LMM) Framing

**Documents: `docs/lmm/`, CONTRIBUTING.md, scattered references.**

The LMM maps the pipeline to four stages: RAW → NODES → REFLECT →
SYNTHESIZE. These are renamed versions of standard pipeline stages:

- RAW = raw input data
- NODES = feature extraction / encoding
- REFLECT = candidate retrieval / filtering
- SYNTHESIZE = ranking / voting / output

The renaming does not predict any behavior, constrain any design decision,
or help build a better classifier. Sentences like "The wood cuts itself when
you understand the grain" appear in the methodology documentation and add
zero technical content.

The LMM framing was already identified as fluff in doc 26 with a specific
recommendation to strip it. It persists in CONTRIBUTING.md and in the
lmm/ subdirectory. The underlying architectural thinking in those documents
(confidence-gated hybrid classification, the hot-map-as-collapsed-cascade
insight) is valuable — the LMM wrapper is not.

The risk: a reviewer reading CONTRIBUTING.md will encounter branded
pseudo-methodology before reaching any technical content. This creates an
unfavorable first impression that the strong experimental results then have
to overcome.

**Status:** Not yet cleaned up. The lmm/ directory is a monument to a
frame that didn't survive contact with results.

---

### 6.2 "MESI: The 12th Primitive"

**Contribution 14, `docs/14-fused-kernel-architecture.md`.**

Read-only model data being cache-coherent (MESI Shared state) is a basic
property of every modern multicore CPU. Every read-only model in every ML
framework benefits from MESI Shared state. Naming it the "12th primitive"
implies it is a discovery; it is a well-known hardware property being
correctly applied.

The cache-residency analysis itself (1.3MB model, 232-byte working set,
how the data fits in L2/L3) is legitimate engineering. The "12th primitive"
label should be dropped; the analysis should stand on its own.

---

### 6.3 The "Cipher" Metaphor

**Contribution 2, hot map documentation.**

The hot map is a naive Bayes classifier with discretized ternary block
features. Describing training images as "encrypted" into the hot map and
classification as "decryption" is evocative but technically misleading. The
metaphor suggests reversibility (you cannot "decrypt" the original image
from a hot map) and implies a security-related property that doesn't exist.

The hot map accumulates class-conditional frequency counts for each
(position, block-value) pair. This is straightforward and doesn't benefit
from a cipher metaphor. The implementation and results are the contribution;
the metaphor should be dropped.

---

### 6.4 SBT/RT Crossover Signal Theory

**Contribution 12, archived at `docs/archived/docs/12-sbt-rt-crossover-signal-theory.md`.**

This contribution studies analog noise resilience of Redundant Ternary (RT,
13-level) vs. Standard Balanced Ternary (SBT, 27-level) encodings under
Gaussian noise at various SNR levels. The simulation is competent.

It is completely disconnected from the classifier. Nothing in SSTT operates
in an analog noise regime. No classifier uses RT encoding. The crossover
finding has no application within this project. This was correctly archived;
it belongs to a separate research interest unrelated to image classification.

Its presence in the contribution count (as C12) inflates the number of
claimed contributions while providing no path to the key results.

---

### 6.5 The Papers Folder as Currently Written

**`docs/papers/sstt-paper-draft.md`, `01-adaptive-router-systems.md`,
`02-lagrangian-skeletons-vision.md`, `03-hierarchical-scaling-engineering.md`.**

These documents have a fundamental problem: they describe the project as
it was planned to work, not as it demonstrably works. Several specific
discrepancies:

**Paper 2 (Lagrangian Structural Skeletons)** claims to have broken 50% on
CIFAR-10 using "topological skeletons." The actual 50.18% result (doc 43)
came from 3-eye stereoscopic retrieval + RGB grid Gauss map kNN — not from
Lagrangian vorticity or particle tracing. The Lagrangian curl experiment
(doc 46) achieved 39% on a *small data slice*, never validated on the full
10,000-image test set. The Navier-Stokes experiment (doc 58) collapsed to
random (10%).

**Paper 3 (Hierarchical Scaling)** presents the scaling framework as having
validated "98.5% accuracy retention" and "0.14ms latency at 224×224." Both
figures are accurate, but the absolute accuracy at 224×224 is 41.9% on
Fashion (doc 59). This is barely above the 40.9% coarse-only baseline. A
+0.5pp hierarchical lift on 41% absolute accuracy is not a result worth
leading a paper with.

**The headline paper** presents Lagrangian skeletons, Navier-Stokes, and
particle tracing as core contributions of a production engine. The
experimental record (docs 46–47, 57–58) shows these are failed experiments
that revealed an important general principle (integral > differential) but
do not contribute to any validated result.

**Unified conclusion:** The papers folder reads like grant proposals written
before the experiments were run, then not updated to reflect what actually
worked. A reviewer will notice the mismatch between these claims and the
experimental record immediately.

---

### 6.6 Inflated Contribution Count

Several contributions in the official count are either:
- **Negative results described as contributions** (Taylor jet, Navier-Stokes,
  Lagrangian curl): These are valuable as documented negative results with
  root cause analysis. They should be labeled as such, not listed as
  equivalent contributions to the bytepacked cascade or the topological
  ranker.
- **Early explorations predating the architecture** (TST butterfly cascade
  C3, SDF cipher C4): These were correctly archived. They should not appear
  in the contribution count used in paper abstracts.
- **Speculative frameworks without empirical validation** (Eigenvalue-Spline
  C18): The framework paper was written before the scaling experiments. The
  experiments (C52, C55, C59) validated a stripped-down version of the
  framework; the full "eigenvalue-ordered traversal" has not been validated.

---

## 7. Understated Findings

These deserve significantly more prominence than they currently receive.

---

### 7.1 Zero Learned Parameters at 97% MNIST Accuracy

The project's most significant claim is buried. The README opens with
accuracy numbers; it never makes the following statement clearly:

> This system achieves 97.27% accuracy on MNIST using zero gradient descent,
> zero backpropagation, zero floating-point arithmetic at inference, and
> zero learned weights. Every decision is an integer table lookup, an
> add/subtract operation, or a comparison. The closest relevant system
> (brute-force pixel kNN) achieves 96.12%. SSTT exceeds it.

Zero-parameter systems are rare. Systems that exceed or match their fully
parametric counterparts are rarer still. The fact that SSTT beats brute kNN
with a 1200× compression of the search space is a strong empirical result
that the current README obscures under architecture details.

---

### 7.2 Fashion-MNIST Cross-Generalization With Zero Tuning

85.68% on Fashion-MNIST, same code, same parameters as MNIST. The channel
importance reversal (h-grad matters on Fashion, not on MNIST) happens
automatically through IG re-weighting — no human intervention.

This is the strongest evidence that the architecture learns real structure
rather than digit-specific features. The +8.94pp lift over brute-force kNN
on Fashion is also larger than the MNIST lift, suggesting the architecture
provides more value on harder problems.

Currently treated as a validation footnote. Should be foregrounded as
evidence of genuine generalization.

---

### 7.3 The 1200× Compression Finding (K-Invariance)

K=50 through K=1000 produce identical accuracy. The inverted-index vote
achieves near-perfect recall of the relevant neighborhood from 60,000
training images using a comparison of compressed block signatures. This
is a 1200× compression ratio with zero information loss on the retrieval
task.

The implication: the retrieval problem is solved at K=50. The remaining
problem is entirely in ranking 50 pre-filtered candidates, not in finding
them. This reframes the architecture from "approximate kNN" to "exact kNN
within a pre-filtered set" — a stronger claim.

---

### 7.4 The 98.6% Fixable Ceiling

Mode A errors (retrieval miss, 1.7% of errors) represent the true floor
of the current architecture. Everything else (98.3% of errors) is a ranking
failure, where the correct answer was retrieved but the wrong answer won.

If all Mode B errors (correct class in top-K, wrong class wins dot product)
were resolved, accuracy would reach approximately 98.6%. The current gap
from 97.27% to the 98.6% ceiling is entirely a ranking problem.

This is a crisp research agenda: better ranking within the pre-filtered
top-50. The retrieval is not the bottleneck. The dot-product ranking is.
The divergence/centroid/topological features (+111 MNIST errors fixed in
contributions 29–31) are already moving toward this ceiling; the remaining
~1.3pp requires better discrimination between Mode B confusions.

---

### 7.5 Tier 1 Zero False Positive Rate on Fashion-MNIST

The Three-Tier Router's Fast Tier (T1) achieves **100.0% accuracy** on the
9.6% of Fashion-MNIST images it processes, with zero false positives. The
unanimity threshold of 198/200 is conservative enough that it never fires
on a wrong prediction in this dataset.

This is a remarkable property for a safety-critical context: a zero-latency
path with a guaranteed-correct prediction, available on approximately 10%
of queries, with no learned parameters determining which queries qualify.
The confidence signal is derived purely from vote concentration in the
retrieval step.

This result is currently buried in tables in docs 50 and 51.

---

### 7.6 Per-Class CIFAR-10 Analysis as Evidence of Mechanism

The per-class accuracy tables in docs 42 and 43 are among the most
scientifically informative content in the project, currently buried:

- **Ship: 68.7%** on the cascade Gauss system (vs. 55.9% on the brute Gauss
  baseline). The cascade's texture pre-filtering removes non-ship candidates,
  letting the Gauss map focus on distinguishing ships from other watercraft
  and vehicles rather than from unrelated classes.
- **Cat: 32.9%** (up from 21.6% on the stereo stack). Cat is the hardest
  CIFAR-10 class; the Gauss map's edge geometry captures feline features
  that block texture misses entirely.
- **Frog: 52.0%** (down from 67.7% on brute Gauss). The cascade hurts frog
  because frogs' distinctive green coloring is a strong texture retrieval
  signal that the fixed-threshold stereo quantization captures well; the
  cascade filters by texture first and then relies on Gauss map shape, but
  frog shape is less distinctive than frog color.

These per-class results reveal exactly what the system models. They are the
mechanistic story behind the aggregate accuracy number.

---

### 7.7 The 87% Scattered-Write Bottleneck as Generalizable Finding

The memory bottleneck diagnosis (87% of runtime is scattered writes to the
vote accumulator) is understated as a "profiling note" in contribution 23.
It is a general architectural finding about inverted-index systems on
modern hardware: the bottleneck is not in the arithmetic but in the memory
access pattern.

This finding applies to document search engines, recommendation systems,
and any sparse vector inner product computation. The diagnosis methodology
(phase-timer profiling, calloc elimination, pre-allocated histograms) is
a transferable engineering pattern.

---

## 8. Paper-Worthy Analysis

---

### 8.1 The Strong Case: One Tight MNIST/Fashion Paper

The following result set is clean, validated, reproducible, and publishable
today with no additional experiments required:

**Core claims:**
1. 97.27% MNIST accuracy with zero learned parameters, zero backpropagation,
   zero floating-point at inference.
2. Exceeds brute-force pixel kNN (96.12%) using a 1200× compressed search.
3. 85.68% Fashion-MNIST with zero tuning from MNIST, demonstrating
   cross-dataset generalization.
4. Speed-accuracy Pareto frontier from 1.9μs (91.8% MNIST) to 1ms (97.27%),
   covering embedded real-time through batch processing regimes.
5. Zero false positives on the T1 slice (100.0% Fashion-MNIST at 9.6% coverage).

**Supporting technical contributions:**
- `_mm256_sign_epi8` as ternary multiply: the hardware primitive enabling
  the cascade.
- Vote-then-refine cascade composition: the architecture.
- K-invariance (K=50 = K=1000): the retrieval completeness result.
- Mode A/B/C error autopsy: the ceiling analysis and research agenda.
- 87% scattered-write bottleneck: the memory architecture finding.
- Integral > differential law: demonstrated on MNIST (divergence works,
  PCA fails, Taylor jet works partially).

**Suggested title:**
> *Classification as Address Generation: Zero-Parameter Ternary Cascade
> Inference Matching k-NN Accuracy at 1200× Compression*

**Venue fit:** ICML, NeurIPS, ICLR (efficiency track), or CVPR (efficient
methods). The zero-parameter claim combined with the accuracy result is
strong; the latency numbers are competitive with quantized neural nets on
embedded hardware.

**What it does not claim:** nothing about CIFAR-10, Lagrangian methods,
scaling to 224×224, or topological skeletons. Those are separate research
threads at various stages of completion.

---

### 8.2 The Possible Case: Quantized Field Operators Paper

The integral > differential law (section 4.7) is a publishable finding if
formalized. The empirical basis exists:

- Six pairs of integral/differential operator comparisons across three datasets
- Clear failure mode: differential operators (Hessian, Curl, Poisson) fail
  when the quantized field is too coarse to support second-order estimates
- Clear success pattern: integral operators (divergence, IG, histogram) are
  robust because individual quantization errors average out over the region

What this paper lacks:
- A theoretical bound: what quantization level is required for reliable
  second-order estimation? The Taylor jet MNIST result (91.6%) shows it
  works on high-contrast 28×28 images; the CIFAR-10 result shows it fails
  on noisy 32×32 images. A quantitative threshold would strengthen the claim.
- Systematic comparison across quantization levels: does Hessian work at
  5-level quantization? 10-level? This experiment has not been run.

Without the theoretical bound, this is an empirical observation. With it,
it is a fundamental result about quantized inference.

---

### 8.3 Not Yet Ready: CIFAR-10

50.18% on CIFAR-10 is 5× random and a legitimate result for a
zero-parameter system. It is not competitive with modern methods (a simple
ResNet-20 achieves ~92%). The gap cannot be closed by adding more geometric
features to the current architecture — the cat/dog inter-class Gauss
distance (214 vs. 2145 for airplane/frog) confirms that 32×32 resolution
is insufficient for distinguishing visually similar classes regardless of
the features used.

The CIFAR-10 story is valuable as a boundary test: it establishes what the
architecture can and cannot do on natural images. It is not an accuracy
result worth leading a paper with.

---

### 8.4 Not Yet Ready: 224×224 Scaling

The scaling results (doc 52, 59) have:
- Strong compression: 16× model size reduction with 98.5% accuracy retention
- Strong latency: 0.14ms at 224×224
- Weak absolute accuracy: 41.9% on Fashion-224

The 41.9% number needs to be contextualized. On resized 28-class Fashion-MNIST
at 224×224, what does brute kNN achieve? The paper as written doesn't compare
against a 224×224 brute baseline, making it impossible to evaluate whether
the compression preserved real information or just preserved a low baseline.
If the brute 224×224 baseline is also ~42%, the compression finding is real.
If it is 90%, the compression finding is meaningless.

This experiment needs to be run before the scaling paper can be written.

---

## 9. The Honest Ceiling

### 9.1 MNIST

**Current best:** 97.27% (validated)
**Mode A ceiling:** ~98.6% (if all Mode B/C ranking errors were fixed)
**True architectural ceiling:** ~99%–99.2% based on brute oracle experiments

The gap from 97.27% to 98.6% is a ranking problem. The topological features
(divergence, centroid, grid decomposition, Kalman weighting) moved from
96.04% to 97.27% — a +1.23pp gain from better ranking. The remaining ~1.33pp
to the Mode A ceiling requires better discrimination between the specific
confusion pairs that survive the topological ranking (primarily 4/9 and 3/5
on MNIST).

The topo1–topo9 ablation series (+111 errors fixed) suggests continued
progress is possible with better ranking features, but each successive
improvement is smaller (diminishing returns). The oracle routing result
(96.44%) shows that per-confusion-pair specialist routing gives marginal
gains, suggesting the remaining errors are genuinely hard cases.

### 9.2 Fashion-MNIST

**Current best:** 85.68% (validated)
**Brute kNN baseline:** 76.74% (approximate)
**SSTT lift over brute kNN:** +8.94pp

Fashion-MNIST is harder than MNIST by architecture: classes overlap in both
texture space (shirt vs. T-shirt) and edge geometry space (sneaker vs.
ankle boot). The +215 Fashion errors fixed by sequential field ranking vs.
the +111 MNIST errors fixed suggests the architecture has more room to grow
on Fashion — the geometric features provide more incremental value on a
harder problem.

### 9.3 CIFAR-10

**Current best:** 50.18% (5× random)
**Inter-class Gauss ceiling:** Cat/dog at 214 sphere-space distance vs.
airplane/frog at 2145 implies ~5–8% of CIFAR-10 errors are fundamentally
irreducible at 32×32 without access to higher-level features.

**Honest assessment:** Reaching 55–60% without learned parameters on
CIFAR-10 would require features that the current architecture does not
possess. The block-based texture system and the Gauss map edge geometry
system are the two cleanest signal sources available at 32×32 ternary
resolution. Both have been pushed to their limits. The remaining gap to
competitive accuracy (~92% for ResNet) requires representation learning
that zero-parameter systems cannot perform.

---

## 10. Recommendations

### 10.1 Immediate (before any submission)

1. **Rewrite the papers folder from the experimental record, not from
   the plan.** Paper 2 (Lagrangian) must either be dropped or rewritten
   to accurately describe the contribution: "Discrete operators in ternary
   space: which ones survive quantization, and why." The current framing
   is inaccurate.

2. **Strip all LMM references from main documentation.** CONTRIBUTING.md,
   README.md, and docs/lmm/ should not appear in the narrative path of
   the paper. The underlying architectural content in the lmm/ docs can
   be preserved in archive form.

3. **Add a 224×224 brute kNN baseline to the scaling experiments.** Without
   it, the compression result cannot be evaluated.

4. **Retitle the "Lagrangian Structural Skeletons" contribution to "Discrete
   Field Operators in Ternary Space: An Empirical Study."** This accurately
   describes what was done (experiments with curl, Hessian, Poisson, tracing)
   and the finding (integral > differential law) without overclaiming.

### 10.2 For the primary paper

5. **Lead with the zero-parameter claim, not the accuracy number.** The
   claim is: "97.27% MNIST with zero learned parameters, exceeding brute
   kNN." The accuracy alone is not remarkable (many models hit 97%); the
   absence of learned parameters is.

6. **Make the K-invariance finding central.** The 1200× compression with
   zero recall loss is the mechanism that makes the cascade better than
   brute kNN. It deserves its own section, not a footnote.

7. **Include the speed-accuracy Pareto frontier as a primary result table.**
   The operating-point menu from 1.9μs to 1ms is the practical value
   proposition for embedded systems. This is not mentioned in any of the
   current paper drafts.

8. **Include the Fashion-MNIST cross-generalization result prominently.**
   It is the proof that the architecture generalizes.

### 10.3 For continued research

9. **The next CIFAR-10 step change is not more geometric features; it is
   resolution.** At 32×32, the inter-class Gauss distance between cat and
   dog is 214 vs. 2145 for airplane/frog. This is a resolution problem.
   Testing the Gauss map cascade on 64×64 CIFAR-10 upsamples (or on
   CIFAR-100 which has more distinct classes) would reveal whether the
   architecture scales.

10. **The integral > differential law needs a theoretical bound to become
    a publishable result.** Run the Taylor jet experiment across quantization
    levels: 3-level (current), 5-level, 10-level, 20-level. Find the
    threshold where second-order estimation becomes reliable. This converts
    an empirical observation into a quantitative principle.

---

## Files

Code (primary): `src/sstt_bytecascade.c`, `src/sstt_router_v1.c`,
  `src/sstt_cifar10_cascade_gauss.c`
This document: `docs/60-audit-2026-comprehensive.md`
Prior audit: `docs/26-audit-novel-useful-fluff-understated.md`
