# Contribution 33: Three Phase-Changes

The feature engineering on the ranking step is exhausted. Nine
topological experiments, Kalman adaptive weighting, sequential
processing, Gauss map, grid decomposition, chirality, noise reduction,
and filtering have been tested. The red-team validation confirmed
parity with brute kNN on MNIST and +8.44pp above it on Fashion.

Three phase-changes remain, each a different kind of leap.

---

## Option 1: Ship the Confidence Router

### What

The most practically valuable artifact is not the 97% classifier — it
is the 264-entry confidence map that knows which images the classifier
will get right. A system that says "I'm 99% confident on 88% of images
and I'll abstain on the rest" is more useful in deployment than one
that says "I'm 97% on everything."

### Architecture

```
Image → Vote → top-200 candidates → count class labels (FREE)
  │
  ├─ ≥190/200 same class → OUTPUT plurality class
  │     22% of images, 99.6% accuracy, ~1 ms
  │     No dot products. No topology. No ranking.
  │
  ├─ 150-189/200 → LIGHT ranking (dot + divergence only)
  │     32% of images, ~97% accuracy
  │
  └─ <150/200 → FULL pipeline or ROUTE TO SPECIALIST
        46% of images, errors concentrate here
        The specialist can be anything: neural net, human, abstention
```

### What needs building

1. **API packaging.** A single function: `sstt_classify(pixels) →
   (class, confidence, tier)` where tier ∈ {fast, light, full}.
2. **Model file format.** The inverted index, IG weights, and
   confidence map serialized to a single binary file. Currently these
   are computed from raw MNIST data at startup.
3. **Tier dispatch.** The three-tier routing logic as a clean state
   machine, not experiment code.
4. **Integration tests.** Verify that the three tiers produce
   consistent results on known inputs.
5. **Latency benchmarks.** Per-tier timing on representative hardware.

### Effort

Weeks of systems engineering. The algorithms exist; the packaging
does not.

### Success criteria

- [ ] Single-header C library with `sstt_classify()` API
- [ ] Pre-built model files for MNIST and Fashion-MNIST
- [ ] Three-tier routing with configurable confidence thresholds
- [ ] Per-tier latency: fast <10μs, light <200μs, full <2ms
- [ ] Documented precision-coverage curve per dataset

---

## Option 2: Formalize and Publish

### What

The composition of retrieval + field theory + adaptive estimation +
sequential inference into a parameter-free classifier is novel. The
+8.44pp over brute kNN on Fashion-MNIST with zero learned parameters
is a real result. The confidence map as a routing mechanism is
practically useful. This is a publishable paper.

### Paper structure

1. **Abstract.** Parameter-free image classification via ternary
   inverted-index voting, Green's theorem divergence features,
   Kalman-adaptive ranking, and sequential Bayesian candidate
   processing. 97% MNIST, 86% Fashion, zero learned parameters.

2. **Introduction.** The gap between neural methods (which learn
   everything) and classical methods (which compute everything).
   SSTT occupies a third position: compute features from
   differential geometry, vote via inverted index, rank via
   adaptive topology, route via confidence map.

3. **Method.**
   - Ternary quantization and block encoding (contributions 1-2)
   - IG-weighted multi-probe inverted-index voting (contributions 7-8)
   - Bytepacked cascade (contribution 22)
   - Gradient divergence via Green's theorem (contribution 29)
   - Kalman-adaptive per-query weighting (topo4)
   - Grid spatial decomposition (topo8)
   - Sequential Bayesian candidate processing (topo9)
   - Quantized confidence map for routing (confidence_map)

4. **Experiments.**
   - MNIST and Fashion-MNIST results
   - Ablation: each component's isolated contribution
   - Val/test split validation
   - 1-NN control (sequential processing adds nothing on MNIST)
   - Brute kNN baseline comparison
   - Precision-coverage curves
   - 8 documented negative results

5. **Analysis.**
   - Error profiling: where and why the system fails
   - The emerging algorithm pattern
   - Maps all the way down: the architectural convergence

6. **Related work.** kNN variants, inverted indices for image
   retrieval, topological data analysis, confidence-based routing.

7. **Discussion.** Limitations (28×28 grayscale only), test-set
   weight optimization caveat, the Fashion >97% coverage caveat.

### What needs doing

1. **Literature review.** Position SSTT relative to published
   non-neural MNIST results, kNN acceleration methods, and
   topological feature literature.
2. **Rigorous methodology.** Proper train/val/test protocol.
   The current val/test split (P0 #1) is a start but needs
   cross-validation or bootstrap confidence intervals.
3. **Writing.** ~10 pages, LaTeX, figures, tables.
4. **Venue selection.** Workshop paper at a vision or ML
   conference, or a journal like Pattern Recognition Letters.

### Effort

A month of focused writing. The experiments are done; the
narrative needs shaping.

### Success criteria

- [ ] Complete manuscript with all sections
- [ ] Reproducibility: reader can `make && ./sstt_bytecascade`
      and verify the numbers
- [ ] Honest limitations section
- [ ] Submitted to a venue

---

## Option 3: Ground-State Delta Map — Maps All The Way Down

### What

The ranking step is the one place in the pipeline that computes rather
than looks up. The ground-state delta map replaces it with a lookup:

1. Compute per-class mean Gauss map histogram (10 ground states)
2. For each test image, compute delta = image_histogram - class_mean
3. Quantize the delta to ternary per bin (below/at/above mean)
4. The quantized delta is a 45-trit signature
5. Build a hot map on these signatures: `delta_map[quantized_delta] → class_confidence`

This converts the ranking step from computation (dot products, Kalman
gain, Bayesian accumulation) into a single table lookup on a ternary
signature in the differential-geometric feature space.

### Why it matters

If this works, the entire classification pipeline becomes three maps:

```
Vote map:       block_signature → candidate_ids
Confidence map: vote_concentration → route_decision
Delta map:      quantized_gauss_delta → class_prediction
```

No dot products anywhere. No per-candidate computation. Three table
lookups from raw pixels to class label. This is the architectural
endgame — the point where the method fully reduces to its own
primitive (quantize → encode → look up).

### What needs building

1. **Per-class Gauss map means and standard deviations** from training
   data. Already computed in `sstt_gauss_delta.c`.

2. **Ternary delta quantization.** For each of 45 bins: if the image's
   count exceeds the class mean by more than 1 stddev → +1. Below by
   more than 1 stddev → -1. Otherwise → 0.

3. **Delta hot map.** For each class prototype, accumulate the
   quantized delta signatures of all training images. The delta
   hot map stores: for each delta-bin position, for each ternary
   delta value, the class frequency distribution.

4. **Delta classification.** For a test image: compute its delta
   relative to each of the 10 class means. Quantize. Look up in the
   delta hot map. Argmax.

5. **Comparison.** Delta map alone vs combined with vote-phase
   routing. Does the delta map help on the uncertain images where
   the vote is divided?

### The experiment

`sstt_gauss_delta.c` already exists with mode 0 (raw delta L1) and
mode 1 (quantized ternary match). What's missing:

- The hot map on quantized deltas (a new lookup table, not a
  per-candidate comparison)
- Testing as a standalone classifier (bypassing the vote phase
  entirely — just delta map from raw image)
- Testing as a specialist for the uncertain tier in the confidence
  router

### Effort

One focused session. The code infrastructure is in place. The
experiment is well-defined.

### Success criteria

- [ ] Delta hot map built from training data
- [ ] Standalone delta map accuracy on MNIST and Fashion
- [ ] Comparison with vote+ranking pipeline
- [ ] If the delta map works on the uncertain tier: combined
      accuracy exceeding the current 97% / 86%
- [ ] Honest red-team of the results

---

## Recommended Order

1. **Option 3 first.** Tests a theoretical claim (can the ranking
   step collapse into a map?) in one focused session. The result —
   positive or negative — informs both Option 1 (what goes in the
   fast path) and Option 2 (the paper's strongest or weakest claim).

2. **Option 2 second.** Formalize what exists. The delta map result
   completes the experimental program. Writing clarifies the
   contribution.

3. **Option 1 last.** Ship after the method is validated and
   published. Shipping is its own discipline and benefits from the
   clarity that writing forces.

---

## Files

This document: `docs/33-next-phase-changes.md`
Ground-state delta code: `src/sstt_gauss_delta.c` (built, unbenchmarked)
Confidence map code: `src/sstt_confidence_map.c` (benchmarked)
