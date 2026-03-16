# Contribution 30: Eigenplane Generalization — An LMM Pass

Three proposed approaches to improving generalization of the topological
ranking features, explored through the Lincoln Manifold Method.

The problem: contribution 29 showed that only the gradient divergence
feature transfers from MNIST to Fashion-MNIST. Centroid and profile are
digit-specific. Can eigenvector decomposition, eigenvalue analysis, or
micro-spline approximation extract more generalizable signal from the
divergence field?

---

## Phase 1: RAW

The divergence feature works on both datasets at the same weight. That's
the strongest empirical signal in the whole project — nothing else has
transferred at the exact same hyperparameter. But we're reducing a 784-
dimensional field to two scalars (neg_sum and neg_centroid_y). That's a
massive information loss. The spatial *shape* of the divergence field
carries structure we're throwing away.

Three options are on the table:

**Option 1: Divergence eigenplanes.** Per-class eigenvectors of the
divergence field covariance. Project candidates onto the query class's
eigenplane for topology-aware similarity. This is PCA on the divergence
field, not on the pixel field. PCA on pixels failed because it destroyed
spatial addressing. But we're not replacing the block features — we're
augmenting the ranking step. The ranking step doesn't use spatial
addressing. It uses dot products. Dot products compose with projections.
So maybe the PCA failure doesn't apply here.

But wait — per-class eigenvectors means we need to know the query's class
to select the right eigenplane. We don't know the class yet; that's what
we're trying to predict. We could use the top-voted class from the vote
step as a proxy. The vote is 99.3% accurate at recall, so the top-voted
class is usually correct. But when it's wrong is exactly when we need the
ranking to fix it. Using the wrong eigenplane in exactly those cases seems
dangerous.

Alternative: compute eigenplanes per confused pair, not per class. The
3↔8 eigenplane captures the divergence patterns that maximally separate
3 from 8. This doesn't require knowing the class — you project both the
query and candidate onto the pair-specific eigenplane and compare. But
then you need N*(N-1)/2 = 45 eigenplanes for 10 classes. That's a lot
of structure for a ranking augmentation.

Or — simpler — just one global eigenplane. The top-2 eigenvectors of
the divergence field covariance across all training images. This captures
the directions of maximum variance in the divergence field without class
conditioning. If topology varies most between digits with different loop
counts, the first eigenvector should separate loopy from non-loopy. But
PCA finds variance, not discrimination. We already know from contribution
17 that variance and discrimination diverge on MNIST.

**Option 2: Micro-splines over the IG surface.** The 252 IG weights form
a spatial map. On MNIST the high-IG blocks cluster in the image center.
On Fashion, the distribution is different. A low-rank spline fit would
smooth out position-specific noise and might transfer better. But the IG
weights are already data-derived — they're the optimal per-position
discriminative weights given the training data. A spline would be a
smoothed approximation of the optimal. Why would the approximation
transfer better than the exact? Maybe because the exact overfits to the
specific spatial arrangement of MNIST digits, while the smoothed version
captures the "concentrate on the informative region" principle without
locking in the exact positions.

Actually, IG splines address the vote phase, not the ranking phase. The
ranking phase is where the topological features operate. These are
orthogonal concerns. IG splines might help Fashion accuracy by preventing
vote overfitting, but they won't help the topological features generalize.

Unless — the spline basis functions themselves become features. If you
fit a low-rank (say, rank-4) spline to the per-image IG-weighted vote
map, the 4 spline coefficients become a compact spatial summary of where
each image's evidence concentrates. This is a spatial fingerprint that
could work as a ranking feature. But this feels like reinventing the
hot map at a coarser resolution.

**Option 3: Structure tensor on the divergence field.** The eigenvalues
of the 2x2 structure tensor at each pixel in the divergence field tell
you about local shape: two large eigenvalues = isotropic blob, one large
+ one small = ridge, both small = flat. This is a standard technique in
computer vision (Harris corner detection uses it). It's local, rotation-
invariant, and computes from the divergence field without class knowledge.

The structure tensor is:

```
    [ sum(dx*dx)  sum(dx*dy) ]
M = [ sum(dx*dy)  sum(dy*dy) ]
```

where dx, dy are spatial derivatives of the divergence field, summed over
a local window. The eigenvalues λ1, λ2 tell you:

- λ1 >> λ2 >> 0: ridge (stroke boundary in divergence space)
- λ1 ≈ λ2 >> 0: blob (loop interior or junction)
- λ1 ≈ λ2 ≈ 0: flat region

A histogram of (λ1, λ2) pairs across the image would be a rotation-
invariant topological fingerprint. How many blobs? How many ridges? How
many corners? This is closer to what Euler was trying to measure but
without the binarization problem.

But this requires floating-point eigenvalue computation (2x2 eigenvalues
have a closed form, but it involves square roots). The SSTT architecture
is entirely integer. We could use the trace (λ1 + λ2 = sum(dx*dx) +
sum(dy*dy)) and determinant (λ1 * λ2 = dx*dx*dy*dy - (dx*dy)^2) as
integer proxies — they capture the same information as the eigenvalues
without computing them explicitly.

Questions:
- Is the PCA failure relevant to divergence eigenplanes at the ranking
  step, or only to block-feature-replacement at the vote step?
- Does the divergence field have enough variance structure for
  eigendecomposition to be meaningful at 28x28?
- Is the structure tensor overkill for a 28x28 image? The local windows
  would be 3x3 or 5x5 — that's 81 or 625 structure tensors for the
  whole image.
- Could we just pass the full 784-element divergence field through the
  existing ternary dot product? That would be the simplest test.
- What's the actual failure mode we're trying to fix? Centroid and
  profile don't transfer. Divergence sum does. Do we need *more*
  divergence signal, or do we need *different* non-divergence signal?

The last question haunts me. Maybe the right move isn't to extract more
from the divergence field. Maybe it's to find a second domain-general
feature as strong as divergence. What made divergence work? It's a
second-order derivative of the pixel field, computed from existing
gradient channels, that captures curvature and concavity. What else has
those properties?

The curl of the gradient field: curl = dvg/dx - dhg/dy. In a continuous
field, the curl of a gradient is zero (irrotational). But in a discrete
ternary field with quantization noise, the curl is non-zero and carries
information about rotational structure. A digit 6 has clockwise curl at
its loop; a digit 9 has counterclockwise curl at its loop. Curl is
rotation-*sensitive* where divergence is rotation-*invariant*. They're
complementary.

Hmm. But curl requires the same kind of spatial derivative computation
as the structure tensor. And it's not clear curl would generalize to
Fashion any better than divergence already does.

I keep coming back to: the simplest high-value experiment is the
divergence dot product. Quantize the full divergence field to ternary,
use the existing AVX2 dot, and add it to the ranking score. Ten lines
of code. Tests whether the spatial structure of the divergence field has
more signal than the two-scalar summary.

---

## Phase 2: NODES

### Node 1: The Divergence Dot Product (Baseline Experiment)

Before any eigenplane work, the simplest test: quantize the 784-element
divergence field to {-1, 0, +1} and compute `tdot(query_div, cand_div)`
using the existing AVX2 infrastructure. This captures full spatial
structure with zero new algorithms. If this doesn't help, no projection
into a lower-dimensional eigenspace will help either — you can't get
more signal from a subspace than from the full space.

**Tension with Node 3:** The dot product is not rotation-invariant. The
structure tensor approach is. If the divergence field's orientation
matters (digit-specific) but its shape statistics don't (domain-general),
then the dot product will be MNIST-specific just like the profile was.

### Node 2: Divergence Eigenplanes (Option 1)

Per-class or global eigenvectors of the divergence field covariance.
Project both query and candidate onto the top-k eigenvectors, compare
projections.

**Key tension:** PCA finds variance, not discrimination. Contribution 17
proved this distinction matters — PCA on pixels failed because the first
eigenvector captured "how much ink" not "what digit." On the divergence
field, the first eigenvector would capture "how much total concavity"
not "where the concavity is." But we already have neg_sum for that.

**Resolution path:** Use the second and third eigenvectors, not the
first. The first captures the mean divergence structure (redundant with
neg_sum). The second and third capture the directions of next-largest
variance — which is where the topological *shape* information lives.

**Dependency on Node 1:** If the full divergence dot product helps, then
the information is there and eigenplanes are a compressed representation.
If it doesn't, eigenplanes won't help.

### Node 3: Structure Tensor (Option 3)

Eigenvalues of the 2x2 structure tensor computed from spatial derivatives
of the divergence field. Captures local shape (blob vs ridge vs corner)
at each pixel. A histogram of eigenvalue categories gives a rotation-
invariant topological fingerprint.

**What makes this different from divergence itself:** Divergence measures
the magnitude of sinks/sources. The structure tensor measures the *shape*
of the divergence field — whether the divergence changes isotropically
(blob), along one axis (ridge), or at a point (corner). Digit 8 has
two divergence blobs (loop interiors). Digit 4 might have a divergence
ridge (the horizontal bar creates a linear concavity). The structure
tensor distinguishes these.

**Tension with simplicity:** Computing the structure tensor requires
spatial derivatives of the divergence field (a third derivative of the
pixel field), windowed summation, and either eigenvalue computation
(floating point) or trace/determinant proxies (integer). This is the
most complex feature proposed — furthest from SSTT's "everything is
integer table lookup" philosophy.

**Tension with scale:** At 28x28, local windows of 3x3 give only a
few hundred non-trivial structure tensors. The statistics may be too
noisy to be useful.

### Node 4: Micro-Splines (Option 2)

Low-rank spline fit to the IG weight surface for vote-phase
generalization, or to the per-image divergence field for a spatial
summary.

**Key insight:** This addresses a different problem than Nodes 1-3.
IG splines affect the vote phase. Divergence features affect the
ranking phase. They're orthogonal.

**But:** Spline coefficients of the divergence field *would* be a
compact, smooth spatial summary — similar to eigenplane projections
but with a spatial locality prior (splines are local, eigenvectors are
global). On a 28x28 image, a 7x7 grid of B-spline control points
would give 49 coefficients per image — a dramatic compression from 784.

**Tension with Node 2:** Spline coefficients and eigenvector projections
are both low-rank representations of the divergence field. Splines
impose spatial smoothness. Eigenvectors maximize variance. Which prior
is better for generalization?

### Node 5: The Real Question — What Made Divergence Generalize?

Divergence works on both datasets at the exact same weight. Why?

Properties of the divergence feature that might explain its
generalization:
1. **Derived from the gradient field, not the pixel field.** The gradient
   field encodes edges. Edges are structurally meaningful across domains
   (digit strokes, clothing seams, texture boundaries).
2. **Second-order derivative.** It captures curvature, not intensity or
   edge direction. Curvature is more stable across visual domains than
   first-order features.
3. **Scalar summary (neg_sum) is translation-invariant.** The total
   negative divergence doesn't depend on where the concavities are,
   only how much there is. Position-independence is a generalization
   prior.
4. **The similarity function uses absolute difference.** `-abs(q - c)`
   rewards similar total concavity without caring about the spatial
   pattern. This is inherently domain-agnostic.

Property 3 and 4 together explain why the scalar summary generalizes
but also why it's limited: it throws away all spatial information.

**The tension:** To extract more from divergence, we need spatial
information (eigenplanes, structure tensor, splines, or dot product).
But spatial information is what makes features domain-specific. The
centroid and profile failed on Fashion precisely because they encode
spatial patterns specific to upright digits.

**The throughline is forming:** The features that generalize are those
that capture *how much* structure exists, not *where* it is. To extract
more generalizable signal, we need representations that are sensitive
to the *amount and type* of structure but invariant to its *position*.

### Node 6: Curl as a Complementary Feature

The curl of the gradient field: `curl = dvg/dx - dhg/dy`.

In a continuous gradient field, curl is zero. In discrete ternary, it's
non-zero and captures rotational structure. This is complementary to
divergence (which captures convergence/divergence structure).

Like divergence, curl is:
- Computed from existing gradient channels
- A second-order derivative
- Scalar-summarizable (total positive/negative curl)
- Position-invariant in its scalar form

Curl might distinguish digits with clockwise loops (6) from
counterclockwise (9), or garments with spiraling folds from straight
seams. Worth testing alongside eigenplane approaches.

### Tensions Summary

1. **Spatial vs invariant:** More divergence signal requires spatial
   structure, but spatial structure is domain-specific.
2. **Variance vs discrimination:** Eigenplanes maximize variance, which
   doesn't align with class discrimination (the PCA lesson).
3. **Complexity vs philosophy:** Structure tensor requires third
   derivatives and floating point, violating SSTT's integer-only design.
4. **Full dot vs summary:** The full divergence dot product is the
   simplest test but may not generalize (same failure mode as profile).
5. **Vote vs ranking:** IG splines address vote generalization, not
   ranking generalization — orthogonal to the topological features.

---

## Phase 3: REFLECT

### Core Insight

The throughline across all three options is the tension between
**information** and **invariance**. The scalar divergence summary
generalizes because it's invariant (position-independent, scale-
independent). The full divergence field has more information but is
position-dependent and may not generalize.

The resolution: **invariant statistics of the spatial field**.

Not the field itself (position-dependent). Not a single scalar
(information-poor). But statistics that capture the *distribution*
of divergence values and their *relational structure* without encoding
their positions.

This resolves the eigenplane question: global eigenvectors are the
wrong tool because they're position-dependent (eigenvector[i] refers
to pixel position i). Per-class eigenvectors are wrong because they
require class knowledge. Structure tensor eigenvalues are the right
idea — they're local and position-invariant — but the implementation
is unnecessarily complex for 28x28.

### Resolved Tensions

**Tension 1 (spatial vs invariant):** Resolved by using position-
invariant statistics of the spatial field. A histogram of divergence
values (how many pixels have div=-3, -2, -1, 0, +1, etc.) captures
the distribution without position. This is simpler than eigenplanes,
simpler than structure tensors, and inherently invariant.

**Tension 2 (variance vs discrimination):** Resolved by not using PCA
at all. The histogram approach doesn't decompose by variance — it
counts occurrences. This is closer to the hot map philosophy (frequency
counting) than to PCA (linear algebra).

**Tension 4 (full dot vs summary):** The divergence dot product should
be tested as the baseline experiment (Node 1). If it helps on MNIST
but not Fashion, that confirms the spatial structure is domain-specific
and supports the histogram/invariant approach. If it helps on both,
the full dot is the simplest solution and no eigenplane work is needed.

**Tension 5 (vote vs ranking):** Confirmed orthogonal. IG splines are
a separate experiment. Drop from this pass.

### Hidden Assumption

We assumed that extracting "more" from the divergence field requires
complex mathematics (eigenvectors, structure tensors, splines). But the
divergence field is a 9-valued integer field ({-4, ..., +4}). A
histogram of these 9 values is a 9-element vector. That's 7 more
dimensions than neg_sum — and it's inherently position-invariant,
integer, and trivial to compute.

The histogram approach is the "what would this look like if it were
easy?" answer.

### What I Now Understand

The three options (eigenplanes, splines, structure tensors) are all
attempts to find a middle ground between "scalar summary" and "full
spatial field." But the natural middle ground for integer data is not
linear algebra — it's **histograms**.

The hot map itself is a histogram. The IG weights are derived from
histograms. The entire SSTT architecture is built on counting and
lookup. The divergence histogram is the architecturally native way to
extract more signal from the divergence field.

The eigenplane/structure-tensor approaches import continuous-math
concepts (covariance, eigenvalues, projections) into an integer-
arithmetic system. The histogram approach stays within the system's
natural computational paradigm.

**But** — the eigenplane and structure tensor ideas are not worthless.
They point to *what kind of information* we want: the shape of the
divergence distribution, the types of local structure. The histogram
captures the distribution directly. The structure tensor captures
local shape types that the histogram doesn't. There may be a version
of the structure tensor that stays integer (trace and determinant as
proxies for eigenvalues, bucketed into categories) and provides
complementary information.

### Remaining Questions

1. Does the divergence dot product help on both datasets? (Baseline
   test before anything else.)
2. Does a divergence histogram outperform the scalar summary?
3. Does curl (computed like divergence but with the cross terms)
   provide complementary signal?
4. Can the trace/determinant of the structure tensor be bucketed into
   integer categories (blob/ridge/corner) without floating point?

---

## Phase 4: SYNTHESIZE

### Architecture: Three Experiments in Priority Order

#### Experiment A: Divergence Dot Product (10 lines of code)

Quantize the full 784-element divergence field to {-1, 0, +1}. Store
for all training images (aligned, 800 bytes padded). Add one AVX2 dot
product per candidate to the ranking score.

```
score = 256*dot(px) + 192*dot(vg) + w_divdot*tdot(q_div, c_div)
```

**Purpose:** Establishes whether the full spatial divergence field has
more signal than the two-scalar summary. If this helps on MNIST but not
Fashion, the spatial structure is domain-specific and we need invariant
representations. If it helps on both, we're done — the existing AVX2
infrastructure handles everything.

**Cost:** 48 MB for precomputed divergence images (60K × 800 bytes).
One dot product per candidate (negligible).

#### Experiment B: Divergence Histogram (20 lines of code)

Compute a 9-element histogram of divergence values per image. The
divergence field has values in {-4, ..., +4}, so the histogram has 9
bins. Similarity is the negative L1 distance between histograms:

```
hist_sim = -sum(|q_hist[v] - c_hist[v]|) for v in {-4..+4}
```

**Purpose:** Tests whether the *distribution* of divergence values
(not their spatial arrangement) provides more signal than the scalar
sum. This is inherently position-invariant and should generalize.

**Cost:** 9 × 2 bytes per image = 1.08 MB for 60K images. 9
subtractions + absolute values per candidate.

#### Experiment C: Curl Feature (15 lines of code)

Compute the discrete curl of the gradient field:

```
curl(x,y) = vgrad[y][x+1] - vgrad[y][x] - (hgrad[y+1][x] - hgrad[y][x])
```

This captures rotational structure (clockwise vs counterclockwise).
Summarize as: neg_curl_sum (total negative curl), pos_curl_sum (total
positive curl), or a 9-element curl histogram.

**Purpose:** Tests whether rotational structure is complementary to
convergence/divergence structure. Divergence captures sinks/sources;
curl captures vortices. If both help, the gradient field has more
extractable topology than we've used.

**Cost:** Same as divergence features. One pass over gradient arrays.

### Key Decisions

1. **Run Experiment A first.** If the full divergence dot product
   generalizes, the problem is solved without eigenplanes or structure
   tensors. If it doesn't generalize, Experiment B (histogram) is the
   next test.

2. **Drop eigenplanes.** The PCA failure lesson applies — not to the
   ranking step specifically, but to the principle that variance
   decomposition doesn't align with discrimination. The histogram
   approach captures distribution information without variance
   assumptions.

3. **Drop structure tensors.** The information they provide (blob vs
   ridge vs corner) is valuable, but the implementation complexity
   (third derivatives, windowed sums, eigenvalue proxies) is not
   justified at 28x28 scale. If SSTT scales to 224x224, revisit.

4. **Drop IG splines.** Orthogonal to the ranking generalization
   problem. Worth a separate experiment but not part of this pass.

5. **Add curl.** Complementary to divergence, same computational
   paradigm, same cost. Test alongside Experiments A and B.

### Implementation Order

```
1. Experiment A: divergence dot product        (10 lines, 1 hour)
2. Experiment B: divergence histogram          (20 lines, 1 hour)
3. Experiment C: curl feature                  (15 lines, 1 hour)
4. Combined grid search (A+B+C with centroid, profile, existing div)
5. Run on Fashion-MNIST to test generalization
```

### Success Criteria

- [ ] Divergence dot product tested on MNIST and Fashion
- [ ] Divergence histogram tested on MNIST and Fashion
- [ ] Curl feature tested on MNIST and Fashion
- [ ] At least one new feature generalizes at the same weight on both
- [ ] Combined best exceeds 97.11% on MNIST
- [ ] Combined best exceeds 84.24% on Fashion
- [ ] The generalizing features are identified and separated from the
      MNIST-specific ones

### The Throughline

The LMM pass revealed that the three options (eigenplanes, splines,
structure tensors) are all attempts to solve the same underlying
tension: **more information vs more invariance**. The resolution is
not to import continuous-math concepts (eigenvectors, splines) into
an integer system, but to use the system's native paradigm — counting
and comparison — on the divergence field's value distribution.

The divergence histogram is the architecturally native answer. Curl is
the complementary signal. Together with the existing divergence scalar
summary, they form a hierarchy of increasingly rich topological
descriptors:

```
Level 0: neg_sum (1 scalar)              — "how much concavity"
Level 1: divergence histogram (9 ints)   — "what types of divergence"
Level 2: divergence dot (784 trits)      — "where is the divergence"
Level 3: curl summary (1-9 values)       — "rotational structure"
```

Each level adds information. The question is which levels generalize.
The experiments are ordered to answer this from simplest to most
complex.

---

## Experimental Outcomes

The three experiments were implemented and tested (topo2.c). Results:

- **Divergence dot product:** Hurts MNIST (-3 to -27). Helps Fashion
  (+43). The spatial divergence field is position-dependent — helpful
  for complex textures (Fashion), harmful for simple digits (MNIST).
- **Divergence histogram:** Hurts both. The 9-bin distribution is too
  coarse to add value over the scalar neg_sum.
- **Curl:** Hurts both. Rotational structure is redundant with
  divergence for these datasets.

The LMM's core prediction — that the tension between information and
invariance would determine which features generalize — was confirmed.
The histogram approach (predicted as "architecturally native") did not
help because the divergence field's value distribution is already well
captured by the scalar summary.

The actual breakthrough came from a direction the LMM didn't predict:
**Kalman-adaptive per-image weighting** (topo4.c) and **sequential
candidate processing** (topo9.c). These are documented in
[contribution 31](31-sequential-field-ranking.md).

---

## Files

This document: `docs/30-eigenplane-generalization-lmm.md`
