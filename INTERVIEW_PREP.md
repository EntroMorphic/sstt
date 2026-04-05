# SSTT Interview Prep

Quick reference for explaining the system at multiple depths. Lead with the layer that matches the question, go deeper only when asked.

---

## The Name

**SSTT — Signature Search | Ternary Topology**

- **Signature Search**: retrieval via block-signature lookup in an inverted index
- **Ternary Topology**: ranking via structural features on ternary {-1, 0, +1} representations
- The pipe represents the central finding: retrieval and ranking are separable problems with different solutions

---

## 30-Second Answer

SSTT classifies images without ever learning anything. No training loop, no neural network, no weights that get adjusted. It chops every image into tiny strips, looks up which training images have the same strip patterns, and picks the best match. The whole thing runs on integer math — no floating point — and fits in the CPU's fast memory. It gets 97.53% on MNIST, comparable to methods that actually learn.

---

## 2-Minute Answer (when they ask "but how")

Three stages:

**1. Quantize.** Every pixel becomes one of three values: dark (-1), medium (0), bright (+1). We also compute gradients — how brightness changes horizontally and vertically — and quantize those too. Three channels total.

**2. Retrieve.** We chop the image into 252 tiny 3-pixel strips. Each strip becomes a signature. We've pre-built an index: for every possible signature at every position, we know which training images match. At inference, we look up the test image's signatures and tally votes. Training images sharing lots of signatures get high votes. We take the top candidates. This is like a search engine — the index is the retrieval mechanism.

**3. Rank.** We compare the candidates to the query using richer features: overall shape similarity, whether loops appear in the same places, stroke density row by row. Score each candidate, pick the top 3, majority vote.

**The key finding:** Retrieval is essentially perfect. The top 50 candidates out of 60,000 contain every relevant neighbor. All errors happen at ranking.

---

## Key Lines

Use these when you need a clean sentence:

- "It's classification as address lookup, not as optimization."
- "There's no training loop. The system computes statistics from the data in a single pass, then classifies by table lookup."
- "50 candidates out of 60,000 contain every relevant answer. 1200x compression, zero loss."
- "97% of errors happen at ranking, not retrieval. We know exactly where the ceiling is and why."
- "It fits in L2 cache. Sub-millisecond inference. No GPU."
- "Every prediction traces back to specific training images. No black box."
- "The inverted index provides breadth — class-diverse candidates. The structural ranker provides depth — discrimination within that set."

---

## Anticipated Questions

### "Why does this matter if neural networks get 99%?"

Neural networks are black boxes that require iterative training, floating-point hardware, and produce weights you can't inspect. SSTT is fully transparent — every prediction traces to specific training images and their vote contributions. No weight matrix intervenes. For safety-critical systems, embedded hardware without FPUs, or anywhere you need deterministic bit-exact reproducibility, that matters. The question isn't whether this replaces neural networks. It's what it reveals about how much of classification is actually just address generation.

### "What's novel here?"

Three things:

1. **K-invariance.** We proved that 50 candidates out of 60,000 contain every relevant neighbor. Accuracy is identical from K=50 to K=1000. Retrieval is solved — all error is ranking. Nobody had shown this for an inverted-index classifier before.

2. **Error mode decomposition.** We categorized every single error: 1.7% are retrieval failures, the rest are ranking failures. This gives us a precise ceiling (99.93%) and tells us exactly where to focus.

3. **The sign_epi8 trick.** One Intel instruction does 32 ternary multiplications per clock cycle. It's an instruction designed for something else entirely that happens to be a perfect ternary multiply. 32 operations, one cycle, no actual multiplication hardware used.

### "How accurate is it?"

- **MNIST (handwritten digits):** 97.27% — comparable to kNN+PCA (97.55%), which is the strongest non-learning baseline. A simple 1-layer neural network gets 97.63%. We're in that neighborhood without any learning.
- **Fashion-MNIST (clothing):** 85.68% — same architecture, different dataset, only IG weights recomputed.
- **CIFAR-10 (natural images):** 50.18% — 5x random but honestly a failure. Ternary features capture edges but not texture. Cat vs dog is near chance. We report this to scope where the architecture stops.

### "How fast is it?"

Sub-millisecond per image on a single CPU core. No GPU. The model is about 1.3MB — fits in L2 cache. The three-tier router variant classifies 10% of images in under 1 microsecond with 99.9% accuracy by detecting easy cases instantly.

### "What's the practical application?"

- Embedded microcontrollers without floating-point hardware (ARM Cortex-M)
- Safety-critical systems requiring deterministic, reproducible inference
- Edge devices where model size must fit in fast cache
- Applications requiring full interpretability — every prediction is auditable
- Any context where you can't run a training loop or don't trust one

### "Is this published?"

Paper draft is complete. We're targeting a systems venue (SysML or similar). The full codebase, all experiments, and all documentation are open source.

### "Did AI help build this?"

Yes. Claude was instrumental in auditing the work, red-teaming the paper, running experiments, and structuring the repository. The research direction, hypotheses, and experimental methodology are mine. Claude is a collaborator — credited as co-author on commits.

### "What surprised you most?"

Two things:

First, that retrieval is essentially free. We thought finding the right candidates from 60,000 training images would be the hard part. It's not. The inverted index nails it at 50 candidates. The entire problem is ranking those 50.

Second — and we just discovered this today — that the inverted index's "noisy" candidates are actually valuable. We tried filtering them with pixel-distance to keep only the "best" ones. Accuracy went *down*. The structural ranker needs to see diverse examples of the same digit written different ways. What looks like noise to a pixel-distance metric is diversity to a topology-based ranker. Retrieval provides breadth, ranking provides depth.

### "Where does it fail?"

Honestly: anywhere the visual difference between categories is texture rather than shape. Cat vs dog at 32x32 pixels is near chance (33-40%) because ternary features capture edges and contours but not fur patterns or facial features. This is an architectural boundary, not a tuning problem. No amount of ternary engineering fixes it — you need learned texture features.

On MNIST, the remaining errors are almost all 4-vs-9 and 3-vs-5 — digits that share both pixel similarity and topological structure.

### "What's next?"

The K-invariance finding opened a new direction. The structural ranker keeps improving with more candidates — it gains +1pp going from 50 to 1000 candidates. Today we discovered why: it needs diverse correct-class examples to build a structural consensus. The next step is designing retrieval that optimizes for diversity rather than similarity. We're at the foothills of understanding how retrieval breadth and ranking depth interact.

---

## Numbers Quick Reference

| What | Number | Context |
|------|--------|---------|
| MNIST accuracy (MTFP) | 97.53% | Val/holdout confirmed |
| MNIST accuracy (topo9) | 97.27% | Previous best |
| Fashion-MNIST | 85.68% | Same architecture |
| CIFAR-10 | 50.18% | Boundary test (5x random) |
| K-invariance | K=50 to K=1000 identical | Cascade dot-product only |
| K-sensitivity (topo9) | 96.59% to 97.61% | K=50 to K=1000 |
| Retrieval recall | 99.63% | Correct class in top-50 |
| Retrieval failures | 7 / 10,000 | Only errors ranking can't fix |
| Theoretical ceiling | 99.93% | If ranking were perfect |
| Latency | <1ms | Single CPU core, no GPU |
| Model size | ~1.3 MB | Fits in L2 cache |
| Router Tier 1 | 99.9% accuracy | On 10% of queries, <1 microsecond |
| Training images | 60,000 | Standard MNIST |
| Candidate compression | 1200x | 50 candidates from 60,000 |
| sign_epi8 throughput | 32 ops/cycle | One AVX2 instruction |

---

## What NOT to Say

- Don't say "zero parameters" without qualification. Say "no iteratively optimized parameters." The system uses information-gain weights computed from labeled data.
- Don't overclaim CIFAR-10. It's 50.18%. That's an honest boundary, not a win.
- Don't say "we solved retrieval" unqualified. The dot-product cascade is K-invariant; the structural ranker benefits from larger K. Retrieval is solved for the cascade, not for the full system.
- Don't say "comparable to neural networks." Say "comparable to kNN+PCA, the strongest non-learning baseline." A 1-layer MLP beats us slightly.
- Don't say "zero tuning" for Fashion-MNIST. The feature weights were re-selected on a validation split. The architecture is unchanged; the parameters differ.
- Don't compare latency to GPU-accelerated methods. Our advantage is no-GPU, not faster-than-GPU.

---

## The Story Arc (if they want narrative)

Started with a simple question: how far can you get on image classification with no learning at all? Just address lookups.

First attempt: frequency tables. 73% accuracy. Crude but instant — microseconds per image.

Added information-gain weighting, multi-channel gradients, inverted-index retrieval. Hit 96%. Close to brute k-nearest-neighbor but 1200x fewer comparisons.

Then discovered K-invariance: the top 50 candidates contain everything. Retrieval is solved. The whole problem is ranking.

Added topological features — gradient divergence, enclosed-region centroids, stroke profiles. Broke through to 97.27%. Validated on holdout data, red-teamed the claims, documented the negative results.

Today ran experiments that revealed the retrieval stage provides diversity the ranking stage needs. What looked like noise was signal. The two stages serve complementary functions — breadth and depth — and they were never designed to talk to each other. That's the frontier.

---

## Repo

https://github.com/EntroMorphic/sstt

MIT license. All experiments reproducible with `make && make mnist && ./build/sstt_topo9_val`.
