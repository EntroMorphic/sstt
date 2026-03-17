# Stereoscopic Quantization — LMM SYNTHESIZE

Clear action plan from the full LMM analysis.

---

## Key Decision

**Stereoscopic multi-perspective quantization is a publishable
architectural contribution.** It should be combined with existing
techniques and validated across datasets before publication.

## Resolved Tensions

**T1 (Simplicity vs accuracy):** Both. Present the stereo principle
as a clean architectural insight AND combine it with the full stack
for the highest accuracy. The principle matters more than the number,
but the combination proves they're complementary.

**T2 (Per-block independence):** Accepted. The 41-42% ceiling on
CIFAR-10 is the honest boundary of per-block-independent classification.
The stereoscopic principle raises the floor within this constraint.
Document this as a principled limitation.

**T3 (CIFAR diminishing returns):** Redirect. Test stereo on
Fashion-MNIST where the base is higher and marginal gains are
meaningful. Stop chasing CIFAR-10 accuracy.

**T4 (Combining stereo + MT4):** Do it. One experiment to measure
the ceiling of the combined approach.

## Action Plan

### Immediate (this session)

1. **Combine stereo + MT4 stack on CIFAR-10.**
   Three-eye voting → MT4 dot ranking → topo features → combined
   Bayesian posterior. Target: 43%+.

2. **Apply stereo to Fashion-MNIST.**
   Three eyes on flattened RGB Fashion (28x84). Measure whether
   the stereoscopic principle lifts the 85.68% ceiling. This validates
   generality.

### For publication

3. **Document the stereoscopic principle** as contribution 39 (done).
   Include the LMM analysis that discovered it.

4. **The paper has three stories:**
   - MNIST/Fashion: 97%/86%, zero parameters, honest validation
   - CIFAR-10: 42%, what generalizes and what doesn't
   - Stereoscopic quantization: a new architectural primitive

5. **Three orthogonal sources of classification power:**
   - Retrieval (inverted-index voting)
   - Ranking (dot + topological features)
   - Perspective (multi-quantization fusion)
   Present these as the three pillars of the architecture.

### What NOT to do

- Don't chase CIFAR-10 past 43-44%. The boundary is known.
- Don't add more than 3 eyes. Diminishing returns proven.
- Don't complicate the stereo architecture with weighting schemes.
  Equal posterior summation is the right default.
- Don't apply stereo to MNIST. The prediction (≤0.1pp) isn't worth
  the experiment time.

## Success Criteria

- [ ] Stereo + MT4 stack > 42.05% on CIFAR-10
- [ ] Stereo > 85.68% on Fashion-MNIST
- [ ] Document all results in contribution 39 update
- [ ] Commit and push

## The Publication-Ready Claim

**"Multi-perspective quantization — applying multiple quantization
functions to the same image and fusing their Bayesian posteriors —
is a zero-parameter architectural primitive that captures
complementary photometric information. On CIFAR-10, three perspectives
(absolute brightness, relative structure, color relationships) achieve
41.18% through Bayesian fusion alone, approaching the 42.05% of a
much deeper single-perspective pipeline. The principle is orthogonal
to existing retrieval and ranking techniques."**

---

## Files

- `docs/39-stereoscopic-quantization.md` — Full contribution document
- `docs/stereo_lmm_raw.md` — RAW phase
- `docs/stereo_lmm_nodes.md` — NODES phase
- `docs/stereo_lmm_reflect.md` — REFLECT phase
- `docs/stereo_lmm_synth.md` — This document
