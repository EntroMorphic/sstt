# Reflections: SSTT Step-Changes

## Core Insight

The inverted index and the structural ranker are solving for different definitions of "similar." The system's accuracy is bounded not by either component alone but by how well they agree on what a good candidate looks like.

## Why three times on the misalignment

Why does brute L2 produce better candidates? Because L2 distance in pixel space correlates with structural similarity — images that are pixel-close tend to have similar divergence, centroid, and profile. The inverted index selects by signature frequency, which correlates with class membership but not with geometric proximity within a class.

Why does signature frequency diverge from geometric proximity? Because IG weighting amplifies positions where signatures predict class labels — the most discriminative positions. Two images of the same digit written with different stroke thickness could have identical class membership but very different signatures at the high-IG positions. The inverted index retrieves both, but the structural ranker finds them dissimilar.

Why does this matter more for the structural ranker than for the dot product? Because the dot product operates on the same ternary representation as the inverted index — they share the same feature space. The structural ranker operates on derived features (divergence, centroid, profile) that are sensitive to geometric structure the ternary encoding can't fully represent. The structural ranker is looking through a different lens than the retriever, and that lens prefers a different kind of candidate.

## Resolved Tensions

**Node 1 vs Node 8 (alignment vs speed):** These aren't actually in tension. The inverted index can stay as the fast first stage. The question is what happens between retrieval and ranking. A lightweight geometric filter — not full brute L2, but a cheap proxy for structural similarity computed on the already-retrieved candidates — could re-rank the inverted-index candidates to select the subset the structural ranker works best on. This preserves speed (filter runs on K candidates, not 60K) and improves alignment.

**Node 2 vs Node 4 (structural hunger vs dot saturation):** The dot product should not drive K selection. It is K-invariant — it doesn't care. The structural ranker's K-sensitivity should determine the candidate pool size. This means the system should retrieve generously (K=500+) and let the structural ranker operate on the full pool. The dot product becomes a component of the combined score, not the gatekeeper.

**Node 9 vs everything (encoding bottleneck):** This is the deepest question and I'm not sure it's the right one to tackle next. The 5-trit ensemble (98.42%) already showed that multiple quantization perspectives help. But the encoding is load-bearing — changing it changes everything downstream. The safer move is to exhaust what the current encoding can do (close the ranking gap) before redesigning the representation. The 266 solvable errors are solvable within the current encoding, because the correct class is present in the candidate pool with the current encoding.

## Challenged Assumptions

**"Retrieval is solved."** I wrote this in the paper. It's not quite right. Retrieval recall is near-perfect (99.6%). But retrieval quality — producing candidates the ranker can use — is not solved. The inverted index retrieves the right answer but surrounds it with candidates that make ranking harder.

**"K-invariance proves the inverted index works."** K-invariance proves the dot product can't use more candidates. It doesn't prove the candidates are optimal. The structural ranker proves they're not.

**"The 1200x compression is a contribution."** It is — but only for the dot-product ranking. For the structural ranking, the compression comes at a cost of 0.87pp at K=50. The contribution is real but narrower than claimed.

## What I Now Understand

The system has two bottlenecks stacked on each other:

1. **Candidate quality bottleneck.** The inverted index produces candidates optimized for class-frequency prediction, not for structural ranking. Brute L2 candidates are better for the ranker. The gap is 0.87pp at K=50 and narrows as K increases (the inverted index compensates with volume). The fix is to improve candidate selection without brute search.

2. **Ranking capacity bottleneck.** The structural ranker has unexploited capacity — it keeps improving with more candidates up to K=1000. But the improvement is diminishing (96.59→97.27→97.57→97.61 at K=50/200/500/1000). The ranker is also approaching its own ceiling, probably limited by the ternary encoding's resolution.

The step-change is in bottleneck 1. Not because the ranker is perfect, but because candidate quality is the one variable we haven't tried to improve. Everything in the project's history has been about building a better ranker on a fixed retrieval stage. Nobody asked whether the retrieval stage is producing the right candidates for the ranker we built.

## Remaining Questions

- What exactly makes an L2-retrieved candidate better for the structural ranker? Is it that L2 candidates are more geometrically similar, or that L2 excludes noisy candidates the inverted index includes?
- Could a simple L2 re-ranking of the top-500 inverted-index candidates, selecting the best 50-200 for structural ranking, close the gap?
- What is the accuracy of brute L2 at K=200 with the structural ranker? That's the true upper bound for what candidate-quality improvement can achieve.
