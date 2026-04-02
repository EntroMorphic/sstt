# Synthesis: SSTT Step-Changes

## The cut

The next step-change is not a better ranker. It's better candidates for the ranker we already have.

The structural ranker has unexploited capacity (proved by K-sensitivity). Brute L2 produces candidates the ranker scores 0.87pp higher (proved by retrieval comparison). The inverted index compensates with speed and volume but its candidates are misaligned with what the ranker needs (proved by the recall-vs-accuracy gap).

The intervention: **re-rank inverted-index candidates by geometric proximity before structural ranking.**

## Architecture

```
Query image
    |
Inverted-index retrieval (K=500, existing code, ~1ms)
    |
L2 re-ranking of 500 candidates → select top-200 (~50us)
    |
Structural ranking on 200 geometrically-filtered candidates (existing code)
    |
k=3 majority vote → class prediction
```

The first stage stays fast. The second stage is cheap (500 SAD comparisons, not 60K). The third stage gets better candidates. Total latency: ~1.1ms. Accuracy target: between 97.49% (brute L2 at K=50) and 97.61% (inverted index at K=1000).

## Why this should work

1. The inverted index at K=500 has 97.57% accuracy with the current ranker. Its candidates include the right answer 99.6% of the time. The pool is good — just noisy.
2. L2 re-ranking of 500 candidates filters the noise. The structural ranker gets candidates that are both class-relevant (from the inverted index) AND geometrically proximate (from the L2 filter). Best of both.
3. K=500→200 re-ranking is 250x cheaper than brute L2 at K=200 (500 vs 60K comparisons). The speed advantage of the inverted index is preserved.

## The experiment

One C file: `sstt_hybrid_retrieval.c`

- Load data, build index, compute features (same as topo9_val)
- For each test image:
  1. `vote()` at K=500 (existing code)
  2. Compute SAD between query and each of 500 candidates (AVX2 `_mm256_sad_epu8`)
  3. Select top-200 by minimum SAD
  4. Run full topo9 structural ranking on the filtered 200
- Report accuracy on full 10K test set
- Compare against: inverted index at K=200 (97.27%), inverted index at K=500 (97.57%), brute L2 at K=50 (97.49%)

Also test: K_retrieve=500, K_rerank={50, 100, 200} to find the sweet spot.

## Success criteria

- [ ] Accuracy exceeds 97.50% (above both brute L2 at K=50 and inverted index at K=200)
- [ ] Latency under 1.5ms per query (no more than 50% slower than current system)
- [ ] If successful, this becomes the new default pipeline and the paper's headline number improves

## What this doesn't address

- The 7 Mode A retrieval failures (these need encoding-level investigation, separate effort)
- CIFAR-10 improvement (requires texture features, different problem)
- The encoding resolution bottleneck (Node 9 — deferred until ranking improvements are exhausted)

## If it fails

If the L2 re-ranking doesn't improve accuracy over K=500 inverted-index-only, then the inverted-index candidates at K=500 are already as good as L2 candidates and the gap we measured at K=50 was a small-K artifact. That would mean the current system at K=500 (97.57%) is already near-optimal and the remaining 239 errors need a fundamentally different approach — likely encoding improvement or new structural features. Either outcome is valuable information.
