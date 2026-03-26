# Gauss Map + Full Architecture — LMM SYNTHESIZE

---

## Key Decision

**Restore the RETRIEVE → RANK pipeline with the best retrieval
(3-eye stereo voting) and the best ranking (grid Gauss map).**

The Gauss map broke the pipeline by doing brute kNN. The block system
has excellent retrieval (98% recall). Reconnecting them uses each
system for what it does best.

## Action Plan

### Experiment 1: Single-eye vote → Gauss rank

Block inverted-index voting (single eye) retrieves top-200.
Grid Gauss map L1 distance re-ranks the 200 candidates.
k=1,3,5 sweep.

This tests: does texture-filtered Gauss ranking beat brute Gauss kNN?

### Experiment 2: 3-eye stereo vote → Gauss rank

Upgrade retrieval to 3-eye stereo voting.
Same Gauss map ranking.

This tests: does better retrieval improve Gauss ranking?

### Experiment 3: RGB grid Gauss map

Extend grid Gauss map to per-channel RGB.
4 channels × 16 regions × 45 bins = 2880 features.
Test both brute kNN and cascade (vote → Gauss rank).

### Experiment 4: Finer grid (8×8)

64 regions × 45 bins = 2880 features (grayscale).
More spatial resolution for shape discrimination.

### Experiment 5: Full composition

Best retrieval + best Gauss features + confidence routing:
- High vote agreement: trust block Bayesian (fast path)
- Low agreement: full Gauss map ranking (accurate path)

## Success Criterion

- [ ] Vote → Gauss rank > 48.31% (proves cascade helps Gauss)
- [ ] RGB grid Gauss > 48.31% (proves color Gauss adds signal)
- [ ] Full composition > 50% (milestone: half correct on CIFAR-10)

## The 50% Question

50% on CIFAR-10 with zero learned parameters would be a meaningful
milestone: 5× random, exceeding all known non-learned methods.
The Gauss map at 48.31% is close. The gap is 1.69pp = 169 images.
If cascade retrieval correctly routes even 200 more images to the
right Gauss map neighbor, we're there.

---

## Files

- `docs/gauss_lmm_raw.md`
- `docs/gauss_lmm_nodes.md`
- `docs/gauss_lmm_reflect.md`
- `docs/gauss_lmm_synth.md` — this document
