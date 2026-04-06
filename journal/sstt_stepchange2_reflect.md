# Reflections: SSTT Step-Changes (Post-MTFP)

## Core Insight

The system just proved that fixing the number system representation — not adding features, not tuning weights, not changing the algorithm — produced a measurable accuracy gain. The representation was the bottleneck, not the algorithm. This suggests the next step-change is also representational, not algorithmic.

## Why three times

Why did MTFP gain +0.26pp? Because the multi-probe was operating in the wrong topology. Binary bit-flips on a ternary encoding produced neighbors that were semantically meaningless — some corrupted the encoding, some changed two trits at once, some flipped between unrelated categories. Trit-flips produce neighbors that differ by exactly one trit in exactly one position. The retrieval stage found better near-miss candidates because the perturbation finally matched the data's structure.

Why does the representation matter more than the algorithm? Because the algorithm was already correct — IG weighting, vote accumulation, structural ranking, MAD adaptive weighting. None of that changed. What changed was how the data is addressed. The algorithm couldn't compensate for the representation being wrong. You can't rank your way out of retrieving the wrong candidates.

Why might the next step-change also be representational? Because the remaining 247 errors are processed by the same correct algorithm on the same correct topology. If the algorithm and topology are right, the remaining errors must come from either (a) the representation losing information the ranker needs, or (b) the ranker lacking features to discriminate what the representation preserves. The MTFP fix was (a). The next move could be either.

## Resolved Tensions

**Node 1 vs everything:** The diagnostics question resolves itself with one experiment. Run MTFP with error mode decomposition. This takes ~15 minutes of compute and answers three questions simultaneously: how many Mode A/B/C, what the new ceiling is, and which confusion pairs dominate. Do this first; everything else depends on it.

**Node 4 vs Node 5 (multi-threshold vs single-threshold):** The last session's diversity insight resolves this clearly. The structural ranker needs diverse candidates. Multi-threshold ensemble provides diversity by construction — different quantization perspectives produce different signatures that retrieve different training images. Single-threshold optimization reduces to finding one view. The system's architecture fundamentally benefits from breadth. **Multi-threshold is the right direction. Single-threshold optimization is a local move on the wrong axis.**

**Node 6 (paper):** Defer. The paper update is a writing task, not a research task. The experiments should finish before the paper moves. Don't update the paper until the experimental landscape is stable.

## Challenged Assumptions

**"The grid search found the same weights, so the algorithm is unchanged."** This is true narrowly (same w_c, w_p, w_d, w_g, sc) but potentially misleading. The same weights on different candidate distributions produce different decisions. The MTFP candidates are different from the joint-index candidates — three independent retrieval channels produce a different mix. The weights are the same; the inputs they operate on are not. A re-sweep at K=500 might find different optimal weights.

**"K=200 is the right default."** This was chosen for topo9. MTFP's per-channel indexing may have changed the vote concentration curve. K=200 might now be too few or too many. The K-sensitivity sweep is mandatory, not optional.

**"The 5-trit ensemble's 98.42% is the ceiling for ensemble approaches."** That number was on the old joint-index system with binary bit-flip multi-probe. MTFP fixes the multi-probe and uses per-channel indexing. MTFP + 5-trit could exceed 98.42% because both the base system and the ensemble benefit from the topology fix.

## What I Now Understand

The priority stack is:

1. **Measure first.** Error decomposition on MTFP + K-sensitivity sweep. These are diagnostics, not experiments. They tell us what to do next.

2. **Fashion-MNIST.** One command. High signal — confirms MTFP generalizes.

3. **Multi-threshold MTFP.** If the diagnostics confirm Mode B (ranking) dominates errors, and if K=500 shows the ranker still has capacity, then multi-threshold retrieval feeds that capacity with more diverse candidates. This is the step-change candidate.

4. **Paper.** After the experimental dust settles.

## Remaining Questions

- Does MTFP change the Mode A count? If the trit-flip fix rescued any Mode A images, the ceiling just rose.
- What's the optimal K for MTFP? The per-channel vote distribution is different from the joint one.
- Does MTFP on Fashion-MNIST gain more than +0.26pp? The hypothesis is yes, because Fashion has more intra-class diversity.
