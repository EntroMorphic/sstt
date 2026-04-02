# Contribution 65: Red-Team Review of Paper Drafts

**Date:** 2026-03-26
**Method:** Adversarial review from three simulated reviewer perspectives:
Reviewer 1 (ML theory), Reviewer 2 (systems/efficiency), Reviewer 3 (skeptic).

---

## Paper 1: "Zero-Parameter Ternary Cascade Classification via Address Generation"

### Reviewer 1 (ML Theory): WEAK REJECT

**Summary:** The paper presents a zero-learned-parameter image classifier
achieving 97.27% on MNIST via inverted-index voting and structural ranking.
The K-invariance finding is interesting. The work is competent but I have
concerns about novelty, positioning, and the strength of the claims.

**Major issues:**

1. **The accuracy is not competitive and the paper doesn't own this clearly
enough.** 97.27% on MNIST is below a 2-layer CNN (~99.2%), a tuned random
forest (~97.5% with hyperparameter search), and even a well-tuned kNN with
PCA preprocessing (~97.5%). The paper positions against brute pixel kNN
(96.12%), which is the weakest possible kNN baseline. A reviewer will
immediately ask: "What about kNN with PCA, LDA, or learned distance
metrics?" The paper needs a kNN+PCA baseline to be honest about the
landscape.

**Action required:** Add kNN with PCA (50-100 components) as a baseline.
This is still zero-gradient-descent but uses an eigendecomposition (SVD)
on the training set. If kNN+PCA beats 97.27%, the paper's core claim
("exceeds brute kNN") weakens to "exceeds brute pixel kNN but not
kNN+PCA." Be honest about this.

2. **The "zero learned parameters" claim is debatable.** The paper states
all parameters are "derived from closed-form statistics." But:
- IG weights are computed from labeled training data (this is learning)
- Dot-product weights are selected by grid search on a validation set
  (this is hyperparameter optimization, which is a form of learning)
- The grid for structural feature weights is discrete and small, but
  it's still selected to maximize accuracy

A strict definition of "zero learned parameters" means "no parameters
that depend on labeled training data." IG weights fail this test. A
more defensible claim is "zero gradient-based parameters" or "no
parameters updated by iterative optimization."

**Action required:** Soften the "zero learned parameters" language to
"zero iteratively-optimized parameters" or "no gradient descent."
Acknowledge in Section 6.2 that IG weights are computed from labeled
data, and that the distinction between "computed" and "learned" is a
continuum, not a binary.

3. **K-invariance is presented as a finding but lacks theoretical
explanation.** Why does the inverted index achieve perfect recall at K=50?
The paper shows it empirically but doesn't explain the mechanism. Is it
because the IG-weighted vote concentrates mass on the correct class? Is
it related to the effective dimensionality of the block-signature space?
Without an explanation, K-invariance is a data point, not a contribution.

**Action suggested (not required):** Add a paragraph in Section 5.1
offering a hypothesis for why K-invariance holds. Even a conjecture
("we hypothesize this arises from the high mutual information between
block signatures and class labels, which concentrates votes on a small
number of class-consistent training images") would elevate the finding.

4. **The error mode decomposition percentages are specific to the
bytepacked cascade, not the full system.** The Mode A/B/C breakdown
(1.7%/64.5%/33.8%) is from the 96.28% cascade. The full system at
97.27% has different error modes (the topo9 ranking fixes many Mode B
errors). The paper should report both decompositions — the cascade
breakdown AND the topo9 breakdown — or clarify which system the
percentages refer to.

**Action required:** Add the topo9 error mode decomposition or clarify
that the Mode A/B/C numbers are from the intermediate cascade system.

### Reviewer 2 (Systems/Efficiency): BORDERLINE ACCEPT

**Summary:** A solid systems paper demonstrating what's achievable under
extreme constraints. The compute profile (87% scattered writes) and the
tiered router are good contributions. The writing is clear.

**Major issues:**

1. **No comparison to ANN systems (FAISS, HNSW, ScaNN).** The paper
compares to brute kNN but the relevant systems baseline is approximate
nearest neighbor search. FAISS with an IVF index can search 60K vectors
in microseconds with >99% recall. How does SSTT's retrieval compare in
speed and recall to FAISS at the same recall level?

**Action suggested:** Add a FAISS/HNSW baseline for retrieval speed at
99%+ recall. SSTT's advantage is that it achieves recall through a
semantically richer representation (block signatures > raw pixels), not
through faster search of the same space. Make this distinction explicit.

2. **The timing numbers lack hardware specifics.** "A single x86 core
with AVX2 support" is insufficient. Which CPU? What clock speed? L2
cache size? The "L2-resident" claim depends on cache size. Report the
CPU model so results are reproducible.

**Action required:** Add CPU model to Appendix A (e.g., "Intel Core
i7-12700, 3.6 GHz, 1.25 MB L2 per core").

3. **The Pareto frontier (Table 3) mixes methods that aren't directly
comparable.** The Dual Hot Map (76.53% at 1.5μs) and the full system
(97.27% at 1ms) aren't alternative configurations of the same system —
they're different architectures with different code paths. A Pareto
frontier should compare configurations of one system, not unrelated
systems. The three-tier router is the only entry that's genuinely a
speed-accuracy tradeoff of one system.

**Action suggested:** Either present Table 3 as "a menu of available
classifiers" (not a Pareto frontier) or focus the speed section on the
three-tier router and its internal tradeoffs.

**Minor issues:**

4. The sequential accumulation (Section 3.7) with decay S=2 is described
but its contribution to accuracy is never ablated. How much does
sequential processing add over static sort-then-vote? The val/holdout
validation (referenced in the text) found that sequential processing was
noise on MNIST (+0.03pp, test-set artifact). This should be disclosed.

5. The code snippet in Section 3.6 (lines 105-110) is unusual for a
paper. It's fine for a workshop paper but a full conference paper should
describe the operation mathematically and relegate implementation details
to an appendix or supplementary material.

### Reviewer 3 (Skeptic): REJECT

**Summary:** This paper achieves 97.27% on MNIST, which is a solved
benchmark. The contribution is narrow, the dataset is uncompetitive,
and the claims of novelty are overstated.

**Major issues:**

1. **MNIST is a solved benchmark and should not be the headline.** Any
paper that leads with MNIST accuracy in 2026 will be desk-rejected at
top venues. MNIST is a sanity check, not a contribution. The paper
acknowledges this implicitly (the constraint is the contribution, not
the accuracy) but then spends 90% of its space on MNIST results.

**Action required:** Either (a) add results on a harder dataset
(CIFAR-10, SVHN, or a subset of ImageNet) as a primary result, not
just a one-paragraph "boundary test," or (b) reposition the paper
entirely as a systems/efficiency contribution where MNIST is the
validation benchmark, not the primary claim.

2. **Fashion-MNIST is not enough to establish generality.** Fashion-MNIST
has the same format, dimensions, and class count as MNIST. It's the same
benchmark in different clothes (literally). A reviewer who sees only
MNIST + Fashion-MNIST will conclude the method is specific to 28×28
grayscale 10-class problems. At minimum, add EMNIST (letters), KMNIST
(Japanese characters), or SVHN (street-view house numbers) to show the
approach works on different visual domains.

**Action required:** Add at least one non-MNIST-format dataset, or
add EMNIST/KMNIST to show robustness across visual domains within
the 28×28 format.

3. **The "zero learned parameters" framing is a constraint, not a
contribution.** The paper presents a self-imposed constraint and then
shows it can be satisfied. But the reader needs to know: *who benefits
from this constraint?* What deployment scenario requires zero learned
parameters AND integer-only inference AND L2-cache residency? If the
answer is "embedded systems with no FPU," the paper should lead with
that use case and demonstrate deployment, not benchmark accuracy.

**Action suggested:** Add a paragraph in the Introduction motivating
*why* zero learned parameters matters. Not "because it's an interesting
question" but "because specific deployment scenarios X, Y, Z require
it." Without motivation, the constraint is arbitrary.

4. **The Related Work section omits the most relevant comparisons.**
- No mention of locality-sensitive hashing (LSH) or product quantization
  (PQ) for fast approximate search
- No mention of FLANN or other fast kNN libraries
- No comparison to decision tree ensembles (XGBoost, LightGBM) which
  are also non-neural, fast at inference, and achieve ~98% on MNIST
- No comparison to template matching or prototype-based methods

**Action required:** Expand Related Work with ANN search methods,
gradient-boosted trees, and prototype/template methods.

5. **The CIFAR-10 boundary paragraph undermines the paper.** Including a
50.18% CIFAR-10 result — even framed as a "boundary test" — invites the
reviewer to compare against 96%+ neural methods on the same dataset.
Either develop the CIFAR-10 story with the 52.41% multi-eye result
(which is more interesting) or remove the paragraph entirely. Half a
result is worse than no result.

**Action suggested:** Either expand CIFAR-10 into a full section with
the 7-eye ensemble result (52.41%) and per-class analysis, or cut the
paragraph.

---

## Paper 2 (not yet drafted): Multi-Threshold Ensemble Findings

### Hypothetical Red-Team (all three reviewers combined)

**What the paper would claim:** Multi-threshold quantization ensemble
improves zero-parameter classification proportionally to dataset
intensity variation. +2.23pp on CIFAR-10, +0.34pp on Fashion-MNIST,
-0.10pp on MNIST. The value of retrieval diversity depends on whether
the ranker is scheme-agnostic.

**Attack vectors:**

1. **The headline result (+2.23pp on CIFAR-10) is an improvement from
50.18% to 52.41%.** This is 5.2× random on a dataset where learned
methods achieve 96%+. A reviewer will ask: "Why should I care about
a 2pp improvement on a system that's 44pp below state of the art?"

**Defense:** The paper isn't competing with neural methods. It's
establishing *where zero-parameter classification breaks and why.*
The +2.23pp is interesting because it comes from the RF-inspired
diversity insight, not from any learned representation. But this
defense requires careful framing — the paper must not oversell.

2. **The MNIST negative result (-0.10pp) weakens the story.** If
multi-threshold hurts on the simplest dataset, reviewers will wonder
whether the Fashion and CIFAR-10 improvements are robust or lucky.

**Defense:** The negative result is explained by a precise mechanism
(scheme-specific ranking features reject candidates from other schemes).
This is a structural insight, not noise. The paper should LEAD with
the negative result and use it to motivate the architectural analysis
of when diversity helps vs hurts.

3. **The Fashion-MNIST improvement (+0.34pp) is within noise.** With
95% CI of ±0.69pp, the +0.34pp is not statistically significant.

**Defense:** The ablation shows a monotonic trend (S=4 at +0.22pp,
S=5 at +0.34pp, driven entirely by adaptive schemes). The trend is
real even if the final number overlaps with CI. But this is a weak
defense. The paper should not lean on Fashion as a headline.

4. **The "unifying insight" (diversity helps when the ranker is
scheme-agnostic) is an observation, not a theorem.** The paper tests
two rankers (topo9 with ternary dots, Gauss map with raw gradients)
on three datasets. That's six data points. The "insight" is a
hypothesis consistent with six data points, not a proven principle.

**Defense:** Fair. Call it a hypothesis, not a law. Present it as
"we observe that..." not "we prove that..."

5. **No comparison to other retrieval diversity methods.** The paper
tests multi-threshold quantization but doesn't compare to other ways
of achieving retrieval diversity: data augmentation at index time,
random projections, multiple hash families (LSH). How do we know
multi-threshold is the best way to get diverse candidates?

**Defense:** The paper claims multi-threshold is a zero-parameter
method for improving retrieval on quantized representations, not that
it's the best retrieval diversity method in general. But the missing
comparison is a real gap.

6. **Seven eyes on CIFAR-10 is a lot of memory and compute.** Each
eye has its own inverted index (~22MB). Seven eyes = ~154MB of indices
plus 7× the vote computation. The paper should report memory and
latency costs alongside accuracy gains.

**Defense:** Easy to add. Report memory and timing.

---

## Consolidated Action Items

### Paper 1 (main paper, ready for submission)

**Must fix (will cause rejection if not addressed):**

| # | Issue | Effort | Section |
|---|-------|--------|---------|
| 1 | Add kNN+PCA baseline | Medium | 4.1, Table 1-2 |
| 2 | Soften "zero learned parameters" to "zero iteratively-optimized parameters" | Low | Abstract, 6.2 |
| 3 | Clarify Mode A/B/C system (bytepacked or topo9?) | Low | 5.2 |
| 4 | Add CPU model to hardware description | Low | Appendix A |
| 5 | Disclose sequential processing was noise on MNIST | Low | 3.7 or 5.x |
| 6 | Add at least EMNIST or KMNIST to show format robustness | Medium | 4.1, new table |

**Should fix (improves paper, may not cause rejection alone):**

| # | Issue | Effort | Section |
|---|-------|--------|---------|
| 7 | Add hypothesis for why K-invariance holds | Low | 5.1 |
| 8 | Expand Related Work: ANN search, gradient-boosted trees, templates | Medium | 2 |
| 9 | Motivate zero-parameter constraint with deployment scenarios | Low | 1 |
| 10 | Decide: expand CIFAR-10 to full section with 52.41% or cut | Medium | 6.3 |
| 11 | Relabel Table 3 as "classifier menu" not "Pareto frontier" | Low | 4.3 |
| 12 | Move code snippet from Section 3.6 to appendix | Low | 3.6, App B |

### Paper 2 (ensemble findings, not yet drafted)

**Must address in the draft:**

| # | Issue | How |
|---|-------|-----|
| 1 | Frame 52.41% CIFAR-10 honestly (not competing with neural methods) | Lead with the insight, not the number |
| 2 | Lead with the MNIST negative result, not hide it | Use it to motivate the scheme-agnostic analysis |
| 3 | Call the unifying insight a "hypothesis" not a "law" | Language throughout |
| 4 | Report memory and latency costs for 7-eye CIFAR-10 | Add a cost table |
| 5 | Acknowledge Fashion +0.34pp is within CI | Lean on ablation trend, not point estimate |
| 6 | Discuss alternative diversity methods (LSH, augmentation) | Related Work |

---

## Venue Assessment (updated)

**Paper 1** in current form: workshop paper at ICML/NeurIPS. With the
fixes above (kNN+PCA baseline, EMNIST, deployment motivation): borderline
full paper at a systems venue (SysML, USENIX ATC) or efficiency track.

**Paper 2** as described: workshop paper or short paper at CVPR/ECCV
efficient methods track. The insight (when does retrieval diversity help?)
is more interesting than the accuracy numbers.

**Combined into one paper:** This might be the strongest option. Paper 1's
weakness is "MNIST is solved." Paper 2's weakness is "52.41% CIFAR-10 is
not competitive." But together they tell a complete story: here's a
zero-parameter system, here's what it achieves, here's where it breaks,
and here's what we learned about retrieval diversity along the way. The
multi-threshold finding on CIFAR-10 gives the paper a result on a harder
dataset, and the negative result on MNIST deepens the K-invariance analysis.
