# Contribution 26: Audit — What is Novel, Useful, Fluff, and Understated

An honest assessment of the SSTT project's 25 contributions and supporting
materials, classifying each claim and artifact by its actual weight.

---

## Novel

These are genuinely original contributions — ideas or compositions that
would not be obvious to a practitioner reading the prior art.

### 1. `_mm256_sign_epi8` as ternary multiply (Contribution 1)

The AVX2 conditional-negation instruction computes exact multiplication
over {-1, 0, +1}: 32 ternary multiply-accumulates per cycle, no actual
multiply instruction, int8 accumulation safe for up to 127 iterations.
This gives a 32-wide ternary ALU in one instruction.

The insight is narrow but precise. It eliminates the widening-multiply
bottleneck that would otherwise dominate ternary dot products. Every
classifier in this repo depends on it.

### 2. The vote-then-refine cascade (Contributions 7, 8, 11)

IG-weighted multi-probe inverted-index voting, followed by top-K dot
product refinement, followed by kNN majority vote. Four independently
understood techniques composed into a parameter-free pipeline that
*beats brute-force kNN* (96.04% vs 95.87%).

The composition is the contribution. Each technique is well-known in
isolation. What's new is: (a) the specific ordering, (b) the finding
that they stack additively without hyperparameter interaction, and (c)
that K=50 suffices — 1200x fewer dot products than brute force with
no information loss.

### 3. The cascade error autopsy methodology (Contribution 23)

Classifying every error into Mode A (vote miss, 1.7%), Mode B (dot
override, 64.5%), and Mode C (kNN dilution, 33.8%) — with the finding
that 98.3% of errors occur at the dot product step, not the vote step.

This directly enabled the multi-channel dot fix (+0.34pp) and the oracle
routing architecture. The methodology is transferable to any
vote-then-refine pipeline.

### 4. Encoding D / bytepacked cascade (Contribution 22)

Packing 3 channels + transition activity into 1 byte, then running the
same cascade on 256-value blocks instead of 27-value. Result: +0.16pp
accuracy AND 3.5x speed — a rare Pareto improvement.

The K-invariance finding (K=50 through K=1000 produce identical accuracy)
is an empirical signal that the encoding achieves near-perfect candidate
retrieval: the top-50 by vote contain every relevant neighbor out of 60K
training images.

### 5. Unanimity-gated specialist routing (Contribution 25)

Using kNN vote unanimity as a confidence gate: unanimous 3-0 vote → done
(92% of images, 98.8% accuracy), split 2-1 → invoke specialist with
orthogonal features. Clean, principled, and the head-to-head ternary vs
pentary specialist comparison is honest about where each helps and hurts.

---

## Useful

These are not novel ideas, but they are well-executed, save future
researchers significant time, and provide concrete actionable findings.

### 1. The compute profile (Contribution 23)

87% of cascade time is vote accumulation (scattered writes to 240KB).
Dot products are negligible (~1%). `select_top_k`'s calloc cold-miss
penalty was a real ~70μs/image cost, fixed by pre-allocation. h-grad
hurts MNIST dot products; v-grad helps. Weight grid: (px=256, hg=0,
vg=192).

### 2. The negative results (Contributions 17, 21)

PCA fails at -16 to -23pp. Root cause: PCA optimizes variance, not
discrimination; double quantization destroys eigenvector orthogonality;
projection destroys spatial addressing — the property the hot map
depends on.

Soft-prior cascade fails at all configurations. Root cause: a weaker
classifier (84%) cannot guide a stronger one (96%). Clean principle:
the prior must be closer to the truth than the system it guides.

### 3. The SWO benchmark catalogue (Contribution 24)

All 13 classifiers with strengths, weaknesses, and opportunities in one
document. The progressive architecture tree (hot map 73% → series 84% →
bytepacked 92% → cascade 96%) makes the design space legible.

### 4. Background value discrimination (Contribution 5)

Per-channel background constants (BG_PIXEL=0, BG_GRAD=13) are a general
principle for multi-modal ternary systems. The bug — applying pixel
background to gradient channels — took gradients from ~20% (near random)
to 60-74%. A concrete debugging story with a transferable lesson.

### 5. The Haar vs WHT trap (Contribution 10)

Standard Haar wavelet is not orthogonal (H*H^T != N*I). WHT dot products
preserve pixel-space rankings exactly. This is a real trap that wastes
time if you hit it, and the documentation saves that time.

### 6. Fashion-MNIST generalization (Contribution 15)

82.89% on Fashion-MNIST with no architecture or parameter changes
demonstrates the approach is not overfitted to MNIST digit topology.
Channel importance reverses (h-grad matters more on Fashion), confirming
the architecture adapts through IG weighting alone.

### 7. The fused kernel (Contribution 14)

The cache-residency analysis is solid (<1KB code, 232-byte working set,
1.3MB model). The C reference at 75 lines is a clean proof of concept
for "raw pixels → class label in one pass, zero intermediate arrays."
225K classifications/sec.

---

## Fluff

These elements add narrative weight without technical substance. They
should be preserved for historical context but moved out of the main
documentation path.

### 1. The Lincoln Manifold Method (LMM) framing

Mapping the classifier pipeline to "RAW → NODES → REFLECT → SYNTHESIZE"
is a post-hoc relabeling of standard pipeline stages. It does not predict
behavior, constrain the design space, or help build a better classifier.
Sentences like "The wood cuts itself when you understand the grain" add
no technical content.

The LMM framing appears in: Contribution 20 (throughout), CONTRIBUTING.md,
README.md, and scattered references across docs. The actual architectural
thinking in doc 20 — confidence-gated hybrid classification, the
hot-map-as-collapsed-cascade insight, the +15pp IG prediction that turned
out to be only +0.54pp — is valuable. The LMM wrapper is not.

**Recommendation:** Strip LMM references from main documentation. Preserve
doc 20 in an archive for the genuine architectural analysis it contains.
Rewrite CONTRIBUTING.md to describe the actual methodology (hypothesis →
implement → measure → document honestly) without the branded framing.

### 2. "MESI: The 12th Primitive" (Contribution 14)

Read-only shared data being cache-friendly is a basic property of modern
CPUs, not a design primitive. Every read-only model in every ML framework
benefits from MESI Shared state. This doesn't need a name or a number.

**Recommendation:** Remove the "12th primitive" branding. Keep the
cache-residency analysis, which is the actual contribution.

### 3. The "cipher" metaphor (Contribution 2)

The hot map is a naive Bayes classifier with discretized ternary block
features. Calling it a "cipher" where training data is "encrypted" and
classification is "decryption" is evocative but technically misleading.
The metaphor doesn't illuminate the mechanism.

**Recommendation:** Rename to "hot map" consistently. Keep the
implementation and performance analysis.

### 4. SBT/RT crossover signal theory (Contribution 12)

This studies analog noise resilience of ternary encodings under Gaussian
noise at various SNR levels. The simulation is competent, but it is
completely disconnected from the classifier: nothing in SSTT operates
in an analog noise regime, no classifier uses RT encoding, and the
crossover finding has no application within this project.

**Recommendation:** Move to archive. It's a separate research interest.

### 5. Inflated contribution count

Contributions 3 (TST butterfly), 4 (SDF cipher), and 18 (eigenvalue
spline) are mentioned in the changelog and numbered as contributions,
but 3 and 4 are foundational explorations from before the architecture
solidified, and 18 is a speculative scaling framework with no empirical
validation.

**Recommendation:** Contributions 3, 4, and 18 should be archived as
early explorations / future directions, not counted as validated
contributions.

---

## Understated

These findings deserve significantly more prominence than they currently
receive. They are the actual headline results.

### 1. 96%+ on MNIST with zero learned parameters

This is the single most important claim in the project, and the README
never states it directly. Zero gradient descent, zero backpropagation,
zero learned weights. Every parameter is derived from closed-form
statistics (IG from mutual information) or grid search over a tiny
discrete space (K in {50..1000}, k in {1,3,5,7}, probe count in {0..8}).

**The claim:** You can match brute kNN accuracy on MNIST using only
integer table lookups, add/subtract, and compare. No floating point in
the inference path. No multiply instruction. No learned weights.

This should be the first sentence of the README.

### 2. The speed-accuracy Pareto frontier

The project maps out a remarkably complete tradeoff curve that is never
presented as a unified result:

| Method | Accuracy | Latency | Regime |
|--------|----------|---------|--------|
| Bytepacked Bayesian | 91.83% | 1.9 μs | Real-time embedded |
| Pentary hot map | 86.31% | 4.7 μs | Real-time embedded |
| Multidot 4-probe | 96.20% | 610 μs | Interactive |
| Bytecascade 8-probe | 96.28% | 930 μs | Interactive |
| Oracle v2 | 96.44% | 1.1 ms | Batch |

Sub-2μs methods are competitive with inference-optimized neural nets on
embedded hardware. This operating-point menu is never called out.

### 3. 82.89% on Fashion-MNIST with no tuning

The same architecture, same parameters, same code path — just different
data files — achieves 82.89% on Fashion-MNIST. The IG weights
automatically redistribute to handle clothing textures vs digit strokes.
This is quietly impressive for a method designed around MNIST structure,
and it's treated as a footnote.

### 4. 87% of compute is scattered writes

The finding that vote accumulation (scattered writes to a 240KB array)
dominates the cascade — not the dot products, not the kNN, not the
encoding — is a fundamental architectural constraint. It says something
about any inverted-index system on modern hardware: the memory access
pattern, not the arithmetic, is the bottleneck.

This has implications beyond SSTT and deserves its own treatment, not
burial as a supporting detail in contribution 23.

### 5. K-invariance: near-perfect retrieval at 50/60000

K=50 through K=1000 produce identical accuracy on the bytepacked
cascade. The top 50 candidates by vote contain every relevant neighbor
out of 60,000 training images. This is a 1200x compression ratio with
zero information loss.

This means the inverted-index vote achieves near-perfect recall. The
problem is entirely in the ranking step, not the retrieval step.

### 6. The 98.6% fixable ceiling

The autopsy estimates ~98.6% accuracy if all Mode B errors (correct
class in top-K, wrong class wins dot) are resolved. The vote
architecture itself has 99.3% recall. The remaining gap is purely a
ranking problem.

This reframes the entire research agenda: the retrieval is solved. Only
the ranking remains. This should be front and center in any discussion
of next steps.

---

## Files

This document: `docs/26-audit-novel-useful-fluff-understated.md`
