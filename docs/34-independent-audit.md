# Contribution 34: Independent Audit — Novel, Useful, Fluff, Understated

An independent audit of the SSTT project as of contribution 33, performed
against the full codebase, all 33 prior documents, experimental results,
and red-team validation. This document intentionally overlaps with
contribution 26 (the project's self-audit) and notes where the two
assessments agree, disagree, or where this audit identifies findings
the self-audit missed.

---

## Novel

These are genuinely original contributions — ideas, compositions, or
findings that would not be obvious to a practitioner reading the prior
art on kNN, inverted indexing, or ternary quantization.

### 1. `_mm256_sign_epi8` as ternary multiply (Contribution 1)

**Agrees with self-audit.**

The AVX2 conditional-negation instruction computes exact multiplication
over {-1, 0, +1}: 32 ternary multiply-accumulates per cycle, no actual
multiply instruction, int8 accumulation safe for up to 127 iterations.
Every classifier in this repo depends on it.

The insight is narrow but precise. It eliminates the widening-multiply
bottleneck that dominates ternary dot products. This is a hardware-level
primitive discovery that belongs in any ternary systems toolkit.

### 2. The vote-then-refine cascade composition (Contributions 7, 8, 11, 22)

**Agrees with self-audit.**

IG-weighted multi-probe inverted-index voting, followed by top-K dot
product refinement, followed by kNN majority vote. Four independently
understood techniques composed into a parameter-free pipeline that
beats brute-force kNN.

The composition is the contribution. Each technique is known; what's
new is: (a) the specific ordering, (b) the finding that they stack
additively without hyperparameter interaction, and (c) K-invariance —
K=50 through K=1000 produce identical accuracy, meaning the vote
achieves near-perfect recall at 50/60000 (1200x compression, zero
information loss).

### 3. The cascade error autopsy methodology (Contribution 23)

**Agrees with self-audit.**

Classifying every error into Mode A (vote miss, 1.7%), Mode B (dot
override, 64.5%), and Mode C (kNN dilution, 33.8%) — with the finding
that 98.3% of errors occur at the dot product step, not the vote step.

This directly enabled the topological ranking work and the confidence
routing architecture. The methodology is transferable to any
vote-then-refine pipeline.

### 4. Encoding D / bytepacked cascade (Contribution 22)

**Agrees with self-audit.**

Packing 3 channels + transition activity into 1 byte, then running the
same cascade on 256-value blocks instead of 27-value. Result: +0.16pp
accuracy AND 3.5x speed — a rare Pareto improvement. The K-invariance
finding confirms near-perfect candidate retrieval.

### 5. The confidence map as a difficulty oracle

**Self-audit lists this as understated. This audit promotes it to novel.**

The pre-ranking vote agreement feature has 100-200x discriminative
power for predicting classification difficulty:

| Agreement | Error rate | Coverage |
|-----------|-----------|----------|
| 10/10 top-class agreement | 0.3% | — |
| 1-4/10 agreement | 40-62% | — |
| ≥195/200 same class | 0.11% | 18% of MNIST |
| ≥190/200 same class | 0.39% | 22% of MNIST |

The 264-entry confidence map (11 bins × 6 bins × 4 bins = agreement ×
MAD × margin) eliminates 47% of classification uncertainty on MNIST
and 30% on Fashion with a single table lookup.

A system that says "I'll handle 22% of images at 99.6% and defer the
rest" is architecturally more interesting than squeezing another 0.3pp
out of topology features. This is a novel contribution to the
confidence-routing literature: a difficulty oracle that is a free
byproduct of the retrieval step, requiring zero additional computation.

### 6. Green's theorem divergence on ternary gradient fields

**Not identified in the self-audit. New finding.**

Applying a continuous differential-geometric operator (discrete
divergence via Green's theorem) to a discretized ternary gradient
field, and having it work, is genuinely surprising:

```
div(x,y) = [hgrad[y][x] - hgrad[y][x-1]] + [vgrad[y][x] - vgrad[y-1][x]]
neg_divergence = sum of negative div values
```

The insight: negative divergence concentrates at loop interiors even
in ternary space. This is position-invariant (sensitive to loop
existence, not location), transfers to Fashion-MNIST without
modification (same weight of 200 works on both datasets), and fixed
+43 errors in isolation.

The mathematical grounding — that a discrete analog of Green's theorem
produces a useful topological feature on quantized fields — is novel.
It connects differential geometry to ternary classification in a way
that predicts behavior (closed loops → negative divergence → specific
digit/garment classes).

### 7. Unanimity-gated specialist routing (Contribution 25)

**Agrees with self-audit.**

Using kNN vote unanimity as a confidence gate: unanimous 3-0 vote →
done (92% of images, 98.8% accuracy), split 2-1 → invoke specialist
with orthogonal features. The head-to-head ternary vs pentary
specialist comparison is honest about where each helps and hurts.

---

## Useful

These are not novel ideas, but they are well-executed, save future
researchers significant time, and provide concrete actionable findings.

### 1. The compute profile (Contribution 23)

**Agrees with self-audit.**

87% of cascade time is vote accumulation (scattered writes to 240KB).
Dot products are negligible (~1%). `select_top_k`'s calloc cold-miss
penalty was a real ~70μs/image cost, fixed by pre-allocation. This
finding generalizes: in any inverted-index system on modern hardware,
the memory access pattern — not the arithmetic — is the bottleneck.

### 2. The negative results (Contributions 17, 21)

**Agrees with self-audit.**

PCA fails at -16 to -23pp. Root cause: double quantization destroys
eigenvector orthogonality; projection destroys spatial addressing.

Soft-prior cascade fails at all configurations. Root cause: a weaker
classifier (84%) cannot guide a stronger one (96%). Clean principle:
the prior must be closer to the truth than the system it guides.

### 3. The complete topo1-topo9 ablation series

**Not identified in the self-audit. New finding.**

Nine experiments where each adds exactly one thing, with isolated error
counts and negative results documented at every step:

| File | Addition | MNIST | Fashion | Finding |
|------|----------|-------|---------|---------|
| topo.c | Divergence + centroid + profile | 97.11% | 84.24% | Breaks block-encoding ceiling |
| topo2.c | Generalization (div dot, curl) | Hurts | +43 | Only divergence generalizes |
| topo3.c | Hard filter vs soft weight | 96.61% | — | Soft weighting superior |
| topo4.c | Kalman-adaptive | 97.16% | 84.52% | Per-image adaptive gain works |
| topo5.c | Divergence refinement | 97.16% | 84.52% | Raw better than masked/weighted |
| topo6.c | Dead zone + bucketing | 97.11% | 84.22% | Kalman already handles noise |
| topo7.c | Chirality + quadrant div | 97.24% | 84.57% | Spatial decomposition helps |
| topo8.c | Grid resolution sweep | 97.27% | 84.80% | Different grids per dataset |
| topo9.c | Bayesian-CfC sequential | 97.31% | 85.81% | Sequential decay dominant on Fashion |

This is textbook experimental methodology. Anyone building a similar
system can trace exactly which ideas paid off (+97 errors fixed across
the series) and which didn't (topo3 hard filtering, topo5 divergence
refinement, topo6 dead zones). The negative results within the
progression are as valuable as the positive ones.

### 4. The SWO benchmark catalogue (Contribution 24)

**Agrees with self-audit.**

All 13 classifiers profiled with strengths, weaknesses, and
opportunities in one document. The progressive architecture tree
(hot map 73% → series 84% → bytepacked 92% → cascade 96% →
topological 97%) makes the design space legible.

### 5. Fashion-MNIST generalization (Contribution 15, extended through topo9)

**Agrees with self-audit, but deserves more emphasis.**

85.81% (now 86.02% with Gauss map) on Fashion-MNIST with the same
architecture isn't just a generalization proof — it's evidence that
IG-weighted features auto-adapt across domains. Specific findings:

- Channel importance reverses: h-grad matters more on Fashion
- IG weighting auto-redistributes without manual intervention
- Optimal grid resolution differs: 2×4 for MNIST, 3×3 for Fashion
- Sequential processing adds +2.54pp on Fashion vs +0.03pp on MNIST

The +8.44pp over brute kNN on Fashion is the strongest evidence that
the architecture adds value beyond pixel-space nearest-neighbor.

### 6. Background value discrimination (Contribution 5)

**Agrees with self-audit.**

Per-channel background constants (BG_PIXEL=0, BG_GRAD=13) with the
debugging story of how applying pixel background to gradient channels
took gradients from ~20% to 60-74%. Transferable lesson for any
multi-modal ternary system.

### 7. The Haar vs WHT trap (Contribution 10)

**Agrees with self-audit.**

Standard Haar wavelet is not orthogonal (H*H^T != N*I). WHT dot
products preserve pixel-space rankings exactly. Saves real debugging
time for anyone who encounters this.

### 8. The fused kernel (Contribution 14)

**Agrees with self-audit.**

<1KB code, 232-byte working set, 1.3MB model. 225K
classifications/sec. The cache-residency analysis is the contribution;
the "MESI primitive" branding is fluff (see below).

### 9. The red-team validation methodology (Contribution 32)

**Not explicitly called out in self-audit's useful section.**

The practice of identifying 10 specific risks, testing each with a
concrete experiment, and honestly reporting where claims hold and
where they don't (e.g., sequential processing negligible on MNIST)
is directly useful as a template for validating any ML system.

---

## Fluff

These elements add narrative weight without technical substance.

### 1. The Lincoln Manifold Method (LMM) framing

**Agrees with self-audit.**

Mapping the classifier pipeline to "RAW → NODES → REFLECT →
SYNTHESIZE" is post-hoc relabeling. It does not predict behavior,
constrain the design space, or help build a better classifier.
Sentences like "The wood cuts itself when you understand the grain"
add no technical content. The actual architectural thinking embedded
in these sections is valuable; the branded wrapper is not.

Appears in: Contribution 20, CONTRIBUTING.md, README.md, scattered
references across docs.

**Recommendation:** Strip LMM references from main documentation.
Preserve the genuine architectural analysis underneath.

### 2. "MESI: The 12th Primitive" (Contribution 14)

**Agrees with self-audit.**

Read-only shared data being cache-friendly is a basic property of
modern CPUs. Every read-only model in every ML framework benefits
from MESI Shared state. The cache-residency analysis is real; the
branding is not.

**Recommendation:** Remove the primitive branding. Keep the analysis.

### 3. The "cipher" metaphor (Contribution 2)

**Agrees with self-audit.**

The hot map is a naive Bayes classifier with discretized ternary
block features. The encryption/decryption metaphor is evocative but
technically misleading.

**Recommendation:** Use "hot map" consistently. Keep the
implementation and performance analysis.

### 4. SBT/RT crossover signal theory (Contribution 12)

**Agrees with self-audit.**

Studies analog noise resilience of ternary encodings under Gaussian
noise at various SNR levels. Competent simulation, completely
disconnected from the classifier. Nothing in SSTT operates in an
analog noise regime.

**Recommendation:** Archive. It's a separate research interest.

### 5. Inflated contribution count

**Agrees with self-audit.**

Contributions 3 (TST butterfly), 4 (SDF cipher), and 18 (eigenvalue
spline) are numbered as contributions but are either foundational
explorations from before the architecture solidified or speculative
frameworks with no empirical validation.

**Recommendation:** Archive as early explorations / future directions.

### 6. Bayesian-CfC sequential processing on MNIST

**New finding, not in the self-audit's fluff section.**

The red-team (Contribution 32) showed sequential processing adds
+0.03pp on MNIST — within noise. The self-audit classifies this as
"validated, asymmetric." It should be classified as what it is:

- **On MNIST:** Does not help. The +0.03pp is statistically
  indistinguishable from zero. Any claim that sequential processing
  improves MNIST classification should be explicitly retracted.
- **On Fashion:** Real contribution. +2.54pp (+254 additional correct
  images) over 1-NN control. This is the genuine finding.

The asymmetry itself is interesting (what about Fashion's structure
makes sequential processing valuable?), but claiming the technique
"works" on MNIST is overstating the evidence.

**Recommendation:** Explicitly retract the MNIST sequential
processing claim. Report it as: "Sequential processing adds +2.54pp
on Fashion-MNIST; the effect on MNIST is negligible (+0.03pp, within
noise)."

### 7. The "maps all the way down" narrative (Contribution 33, Option 3)

**New finding, not in the self-audit.**

The idea of collapsing ranking into a lookup table is elegant, but
the experimental evidence to date contradicts the aspiration:

- Gauss map delta discriminative power: 0.0047 bits (MNIST), 0.0108
  bits (Fashion)
- Only 2.4% and 1.8% of uncertainty eliminated by pre-vote Gauss
  map prediction
- The P0 #7 experiment explicitly concluded: "Pre-vote prediction is
  weak; vote is always needed"

The "three table lookups from raw pixels to class label" architectural
endgame framing is aspirational, not evidenced. It risks becoming the
next LMM — a beautiful story that doesn't predict behavior.

**Recommendation:** Preserve the delta map as a hypothesis to test,
but do not frame it as an "architectural endgame" until experiments
confirm the delta map achieves competitive accuracy. If the experiment
fails, document it as a negative result, not a deferred success.

---

## Understated

These findings deserve significantly more prominence than they
currently receive.

### 1. Zero learned parameters

**Agrees with self-audit.**

This is the single most important claim in the project. 97.31% on
MNIST and 85.81% on Fashion-MNIST with:

- Zero gradient descent
- Zero backpropagation
- Zero learned weights
- No floating point at inference
- No multiply instruction in the inference path

Every parameter is derived from closed-form statistics (IG from
mutual information) or grid search over a tiny discrete space.

**This should be the first sentence of everything.** The README,
any paper abstract, any talk introduction. The accuracy number alone
is unremarkable (brute kNN gets 97% too). What makes it remarkable
is *how* it's achieved.

### 2. The speed-accuracy Pareto frontier

**Agrees with self-audit.**

The project maps out a complete tradeoff curve that is never
presented as a unified result:

| Method | Accuracy | Latency | Regime |
|--------|----------|---------|--------|
| Bytepacked Bayesian | 91.83% | 1.9 μs | Real-time embedded |
| Pentary hot map | 86.31% | 4.7 μs | Real-time embedded |
| Multidot 4-probe | 96.20% | 610 μs | Interactive |
| Bytecascade 8-probe | 96.28% | 930 μs | Interactive |
| Sequential field ranking | 97.31% | ~1 ms | Batch |

Sub-2μs methods are competitive with inference-optimized neural nets
on embedded hardware. The existence of a smooth, configurable
speed-accuracy curve — from 2μs/92% to 1ms/97% — is a deployment
story that no individual accuracy number tells.

### 3. K-invariance: near-perfect retrieval at 50/60000

**Agrees with self-audit.**

K=50 through K=1000 produce identical accuracy. The top 50 candidates
by vote contain every relevant neighbor out of 60,000 training images.
This is a 1200x compression ratio with zero information loss.

This means the inverted-index vote achieves near-perfect recall. The
problem is entirely in the ranking step, not the retrieval step. This
reframes the entire research agenda.

### 4. The 98.6% fixable ceiling

**Agrees with self-audit.**

The autopsy estimates ~98.6% accuracy if all Mode B errors (correct
class in top-K, wrong class wins dot) are resolved. The vote
architecture has 99.3% recall. The remaining gap is purely a ranking
problem.

This should be front and center in any discussion of next steps: the
retrieval is solved; only the ranking remains.

### 5. 87% scattered writes — a general finding

**Agrees with self-audit.**

The finding that vote accumulation (scattered writes to a 240KB
array) dominates the cascade has implications beyond SSTT. It says
something general about inverted-index systems on modern hardware:
the memory access pattern, not the arithmetic, is the bottleneck.

### 6. +8.44pp over brute kNN on Fashion-MNIST

**Partially identified in self-audit but not given enough weight.**

The MNIST result is close to brute kNN (+0.34pp on holdout). The
Fashion result is not close — it's a commanding +8.44pp. This is
the strongest evidence that the architecture does something
fundamentally better than pixel-space nearest-neighbor.

The Fashion result is the stronger publishable claim.

### 7. The experimental methodology

**Not identified in the self-audit. New finding.**

40 self-contained C files, each testing one hypothesis. Negative
results documented with root-cause analysis (PCA, soft prior, hard
filtering, dead zones, divergence refinement, bucketing). Red-team
validation against 10 specific risks. Honest retraction where claims
don't hold (sequential processing on MNIST).

The discipline of "implement → measure → document honestly, including
failures" is rarer than any specific algorithmic trick in this repo.
Most ML research papers present a final result; this project shows
the full search path, including dead ends. If published, the
methodology section would be as valuable as the results.

### 8. The test-set weight optimization vulnerability

**Not identified in the self-audit's understated section. New finding.**

This is understated as a *risk*, not a strength. The val/test split
(P0 #1) shows a 2.5pp gap on MNIST (95.74% val vs 98.24% holdout).
Weights are currently optimized on the full 10K test set.

For publication, this is the single biggest methodological
vulnerability. The Fashion gap is smaller (0.38pp), suggesting
Fashion weights are more robust, but the MNIST gap indicates that
some of the reported accuracy comes from test-set information leakage
through weight tuning.

**Recommendation:** Before publishing, freeze all weights on a proper
validation split (or use cross-validation) and report the lower
number. The honest number is likely ~96.5% on MNIST and ~85.5% on
Fashion — still impressive for zero learned parameters, but
meaningfully different from 97.31%.

---

## Summary of Disagreements with Self-Audit (Contribution 26)

| Item | Self-audit says | This audit says | Reason |
|------|----------------|-----------------|--------|
| Confidence map | Understated | **Novel** | 100-200x discriminative power as a free byproduct of retrieval is an original contribution |
| Green's theorem divergence | Not separately identified | **Novel** | Differential geometry on ternary fields working at all is surprising and original |
| topo1-9 ablation series | Not identified | **Useful** | The complete progression with negative results is textbook methodology |
| Red-team methodology | Not identified | **Useful** | The template of 10 risks → 10 experiments → honest results is transferable |
| Sequential on MNIST | "Validated, asymmetric" | **Fluff** | +0.03pp is noise; claim should be explicitly retracted |
| "Maps all the way down" | Not assessed (written later) | **Fluff** | Evidence contradicts the aspiration; risk of becoming next LMM |
| Methodology as contribution | Not identified | **Understated** | The research process itself is rarer than any specific algorithm |
| Test-set weight leakage | Acknowledged in red-team | **Understated risk** | 2.5pp gap is the #1 publication vulnerability |
| +8.44pp Fashion over kNN | Mentioned | **Understated** | This is the strongest publishable claim, not the MNIST number |

---

## Actionable Recommendations

### Before publishing

1. **Fix test-set contamination.** Freeze weights on a validation
   split. Report the lower number. This is non-negotiable for
   credibility.

2. **Retract sequential processing claim on MNIST.** State clearly:
   "negligible effect on MNIST (+0.03pp); significant on Fashion
   (+2.54pp)."

3. **Lead with Fashion, not MNIST.** The +8.44pp over brute kNN on
   Fashion is the stronger result. MNIST parity with kNN is
   expected; Fashion dominance is not.

### For the README and paper

4. **First sentence: zero learned parameters.** The accuracy number
   is the evidence; the architectural claim is the contribution.

5. **Present the Pareto frontier as a unified figure.** The
   speed-accuracy curve from 2μs to 1ms is a deployment story.

6. **Promote the confidence map to a first-class result.** "99.6%
   accuracy on 22% of images with zero ranking cost" is a headline.

7. **Name the methodology.** The hypothesis-driven experimental
   process with documented negative results is transferable and
   should be explicitly described.

### For the codebase

8. **Strip LMM, cipher, and MESI primitive branding.** Keep the
   underlying analysis in every case.

9. **Archive contributions 3, 4, 12, 18.** They are not validated
   contributions to this project.

10. **Run the delta map experiment honestly.** If it fails, document
    it as negative result #3, not a deferred success.

---

## Files

This document: `docs/34-independent-audit.md`
Prior self-audit: `docs/26-audit-novel-useful-fluff-understated.md`
Red-team validation: `docs/32-red-team-validation.md`
Next phase-changes: `docs/33-next-phase-changes.md`
