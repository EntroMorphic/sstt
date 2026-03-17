# Changelog

All notable changes to SSTT are documented here. Each entry corresponds to one
numbered contribution in `docs/`. Accuracy figures are on the MNIST test set
(10,000 images) unless otherwise noted.

## [0.2.0] — 2026-03-16

Oracle routing, benchmark catalogue, parallel cascade, and project audit.

**[24] Benchmark catalogue**
SWO analysis of all 13 classifiers. Progressive architecture tree from hot map
(73%) through cascade (96%). Persistent bottlenecks identified: vote
accumulation (87% of compute), 4/9 topological confusion (hard floor).

**[25] Multi-specialist oracle** → 96.44%
Unanimity-gated routing: 92% of images take fast path (98.82% accuracy), 7.9%
escalate to specialist. Head-to-head ternary vs pentary specialist comparison.

**[oracle v3] Pair-targeted IG re-vote** → 96.37%
Per-confusion-pair IG re-weighting. Hypothesis partially confirmed: pair-aware
IG helps 4/9 but the effect is small (+2 images) and pair-unaware cascade
already captures most of the signal.

**[parallel] Ternary x pentary cascade** → 96.25%
Parallel pipelines confirm pentary is a search specialist (helps vote phase)
and ternary is a ranking specialist (helps dot phase).

**[26-27] Audit and remediation**
Honest assessment of what is novel, useful, fluff, and understated. Archived 5
docs, rewrote README to lead with headline results, stripped decorative framing.

**[28] Confidence-gated hybrid architecture**
Extracted from doc 20: the hot-map/cascade spectrum insight, confidence gating
design, and the finding that IG pre-baking gives only +0.54pp in hot map space.

**[29] Topological ranking** -> 97.11%
Three topological features augment the cascade's dot-product ranking step:
gradient divergence (Green's theorem, +43), enclosed-region centroid (+49),
horizontal profile (+9). Euler failed (binarization too aggressive).
Divergence operates on the gradient field directly — no binarization.

**[30] Eigenplane generalization**
Explored the information/invariance tension. Predicted divergence
histogram, curl, and divergence dot as experiments. Histogram and curl
failed. Divergence dot helps Fashion only. The resolution came from
Kalman adaptive weighting, not from the predicted features.

**[31] Sequential field-theoretic ranking** -> **97.31% MNIST, 85.81% Fashion**
Nine experiments (topo through topo9) building a sequential ranking system:
- Kalman-adaptive weighting (MAD of candidate divergence as gain): +5 MNIST
- Grid spatial decomposition (2x4 MNIST, 3x3 Fashion): +11 MNIST, +28 Fashion
- Bayesian-CfC sequential processing (candidates in quality order with decay):
  +4 MNIST, +101 Fashion
- Total: +111 MNIST errors fixed, +215 Fashion errors fixed over dot baseline
- Chirality, noise reduction (dead zone/bucket), divergence refinement
  (edge-masked/weighted) all tested and failed — documented as negative results
- An emerging algorithm pattern: RETRIEVE → FIELD → DECOMPOSE → ASSESS → ACCUMULATE
- Error profiling: top-10 agreement predicts failure at 100-200x discriminative power
- Confidence map: 264-entry lookup table eliminates 47% of correctness uncertainty
  (MNIST), 97.2% accuracy on Fashion's confident 50%
- Vote-phase routing: 99.89% MNIST / 99.76% Fashion on the easy ~20%, zero ranking
  cost — the routing decision is a single integer comparison on existing data
- Gauss map: discrete (hgrad,vgrad,div) joint histogram as structural fingerprint
- Maps all the way down: every decision converges toward quantized lookup tables

---

## [0.1.0] — 2026-03-15

Initial research release. 23 contributions establishing the ternary cascade
architecture from first principles.

---

### Phase 1: Foundations (contributions 1–6)

**[1] Sign-epi8 ternary multiply**
AVX2 `_mm256_sign_epi8` as a native ternary multiply: `sign(a) × |b|` computes
ternary dot products at full SIMD throughput.

**[2] Hot map cipher**
451 KB frequency table for O(1) classification. Each of 252 block positions ×
27 ternary values stores a 10-class count vector. Pure table lookup, no
per-image computation. Baseline: 73.23%.

**[3] TST butterfly cascade**
Multi-resolution ternary shape transform. Coarse-to-fine block decomposition.

**[4] SDF cipher**
Signed distance field topological compression. Encodes connectivity as a
ternary field rather than raw pixels.

**[5] Background value discrimination**
Per-channel background values: BG_PIXEL=0 (all-dark), BG_GRAD=13 (flat
region). Skipping background blocks reduces noise and improves IG computation.

**[6] Ternary gradient observers**
Horizontal and vertical gradient channels: `h[y][x] = clamp(t[y][x+1] -
t[y][x])`. Adds edge information to the pixel channel. Combined 3-channel hot
map.

---

### Phase 2: Architecture (contributions 7–13)

**[7] IG-weighted block voting** (+15.55 pp)
Information gain weights per block position. Multiply hot map votes by
discriminative importance. 73.23% → ~88% in cascade context.

**[8] Multi-probe Hamming-1** (+2.38 pp)
For each block value, also look up 6 Hamming-1 trit-flip neighbors at half
weight. Handles single-trit quantization noise.

**[9] WHT ternary native transform**
Walsh-Hadamard Transform as an orthogonal basis for ternary data. Preserves
dot products: `<WHT(x), WHT(y)> = N² × <x, y>`.

**[10] Haar vs WHT trap**
Bug discovery: standard Haar wavelet is not orthogonal (H×Hᵀ ≠ N×I). WHT
dot products do not equal pixel-space dot products unless the true WHT is used.

**[11] Zero-weight cascade** → 96.04%
Full pipeline: IG-weighted multi-probe vote → top-K → ternary dot → k=3.
Established the cascade architecture and 96% accuracy ceiling.

**[12] SBT/RT crossover signal theory**
Redundant Ternary (RT, 13 levels) vs Standard Balanced Ternary (SBT, 27
levels). RT provides better noise resilience for certain signal types.

**[13] WHT brute k-NN** → 96.12%
Brute-force 60K×10K AVX2 int16 dot products. WHT preserves dot products, so
this equals pixel-space brute k-NN. Establishes the true ceiling for pixel
similarity search.

---

### Phase 3: Optimization (contributions 14–20)

**[14] Fused ternary kernel**
11 atomic primitives in 1.3 MB, L-cache-resident. Raw pixels → class in one
pass without per-image heap allocation. 225,000 classifications/sec.

**[15] Fashion-MNIST generalization**
Architecture generalizes: 76.86% brute 1-NN, 81% cascade vote+refine.
Confirms the approach is not MNIST-specific.

**[16] Spectral coarse-to-fine**
Spectral prototype pruning (keep top-3 classes by WHT proto dot) does not
help. Vote-based filtering with full spatial evidence is better.

**[17] Ternary PCA** (failed, −16 to −23 pp)
Eigenvector-guided feature space. PCA basis vectors in ternary space lose the
discriminative structure that block encoding preserves.

**[18] Eigenvalue-spline framework**
Scaling from 28×28 to larger images: eigenvalue-ordered block traversal
enables early stopping with 60% fewer lookups at near-full accuracy.

**[19] Composite moneyballs**
Three zero-cost composites: IG pre-baking, fused multi-probe, joint encoding.
Pre-baked IG in hot map space gives only +0.54pp (not +15pp) because the hot
map collapses per-image identity.

**[20] Confidence-gated hybrid architecture**
The cascade is a discrete Kalman filter in log-odds space. Confidence-gated
hybrid design: fast hot map for easy images, cascade fallback for the hard
~15%. IG pre-baking gives only +0.54pp (not +15pp) in hot map space because
the hot map collapses per-image identity.

---

### Phase 4: Refinement (contributions 21–23)

**[21] Soft-prior cascade** (negative result)
Bayesian posterior boosts cascade votes. Hurts accuracy at all configurations:
a weaker classifier (84%) cannot guide a stronger one (96%). Information
perihelion requires the guide to be closer to the truth.

**[22] Bytepacked cascade** → 96.28% MNIST, 82.89% Fashion (3.5× faster)
Encoding D packs 3 channels into 1 byte: pixel majority trit (2 bits),
h-grad majority trit (2 bits), v-grad majority trit (2 bits), transition
activity (2 bits). Inverted index on 256 values vs 27. Same cascade machinery,
richer representation. K-parameter is invariant (K=50 = K=1000).

**[23] Cascade autopsy + multi-channel dot + probe optimization**

*Error autopsy (sstt_diagnose):*
- 407/414 MNIST errors (98.3%) occur at the dot product step, not the vote step
- Mode B (64.5%): correct class in top-K, wrong class wins dot by 3–20 units
- Mode A (1.7%): vote phase failure (correct class not found)
- Estimated fixable ceiling: ~98.6%

*Multi-channel dot (sstt_multidot):*
- Pixel-only dot → weighted 3-channel: `score = 256×dot(px) + 192×dot(vg)`
- +0.34pp for +7 μs overhead (+1.1% cost)
- h-grad hurts (adds noise); v-grad helps (captures stroke direction)
- Grid best: (px=256, hg=0, vg=192)

*Probe optimization:*
- Phase timer: 87% of compute is vote accumulation; dot is negligible
- `select_top_k` pre-allocated histogram: calloc cold-miss eliminated
- Probe sweep: 4 probes → 96.07% at 1.38× speed vs 8 probes
- Pairwise (Hamming-2) probes tested: Hamming-2 overshoot the useful neighborhood, accuracy drops −0.35pp. Bit ordering within 8 probes has zero effect.

*Tiled routing (sstt_tiled):*
- K-means clustering routes test images to specialized sub-cascades
- Routing hurts: confused digit pairs (4/9, 3/5) cluster together, concentrating confusion rather than resolving it. Global cascade always wins over split cascades.

---

## Versioning

This project uses [Semantic Versioning](https://semver.org/). Each `v0.x.0`
release corresponds to a meaningful milestone in the research progression.
Breaking changes (incompatible encoding formats, API changes) increment the
minor version.
