# Stereoscopic Quantization — LMM REFLECT

Underlying structure, unstated assumptions, and actual leverage.

---

## The Unstated Assumption We Broke

The entire SSTT project, from contribution 1 through contribution 38,
assumed that **classification accuracy increases by deepening the
pipeline on a single representation.** More features → better ranking
→ higher accuracy. This is the "deeper is better" assumption.

The stereoscopic result challenges this: **three shallow pipelines
on different representations outperform (or match) one deep pipeline
on a single representation.** Width beats depth when the
representations are complementary.

This is not a new idea in machine learning (ensemble methods, random
forests, mixture of experts all exploit this). But within the SSTT
architecture, it's a paradigm shift. The original pipeline is:

```
image → quantize → encode → vote → dot → topo → Kalman → kNN → class
```

The stereoscopic pipeline is:

```
image → quantize₁ → encode → hot map → P(c|blocks₁)
     → quantize₂ → encode → hot map → P(c|blocks₂)  → sum → argmax → class
     → quantize₃ → encode → hot map → P(c|blocks₃)
```

No voting. No dot products. No topological features. No cascade.
Just three naive Bayes classifiers with different feature extractors,
combined through posterior summation.

## Why This Works

The mathematical structure:

```
P(class | image) ∝ P(class) × P(blocks₁ | class) × P(blocks₂ | class) × P(blocks₃ | class)
```

In log space:

```
log P(class | image) = const + Σᵢ Σₖ log P(block_k^(eye_i) | class)
```

This is a product of experts model. Each eye contributes an independent
factor to the class posterior. The factors multiply (logs add) because
the three quantizations provide conditionally independent evidence:

- Eye 1 (fixed) says: "the absolute brightness pattern is most
  consistent with class X"
- Eye 2 (adaptive) says: "the relative structure pattern is most
  consistent with class Y"
- Eye 3 (per-channel) says: "the color ratio pattern is most
  consistent with class Z"

When all three agree, confidence is high. When they disagree, the
class with the strongest overall evidence wins. The disagreement
itself carries information — it signals ambiguity that no single eye
can detect.

## Where the Real Leverage Is

**The stereoscopic principle has been tested on CIFAR-10 but not on
the datasets where SSTT already excels.** This is the gap.

On Fashion-MNIST:
- Fixed thresholds 85/170 → dark garments (black shirts, dark coats)
  are dominated by -1 trits
- Adaptive thresholds → all garments normalized to equal trit
  distribution, emphasizing shape over shade
- Fashion's confusion pairs (pullover↔coat, shirt↔t-shirt) are
  partially driven by brightness similarity
- Prediction: stereo adds +1-2pp on Fashion

On MNIST:
- Fixed thresholds already optimal (clean black/white)
- Adaptive thresholds would be identical to fixed (already binary)
- Prediction: stereo adds nothing (<=0.1pp)
- This would CONFIRM the principle: stereo helps when photometric
  variation exists, not when it doesn't

## The Deeper Pattern

The SSTT project has discovered three independent sources of
classification power:

1. **Retrieval** (inverted-index voting): finds candidate images
   with similar block patterns. K-invariant, 97-98% recall.

2. **Ranking** (dot products + topological features): orders
   candidates by similarity. Improved by MT4, divergence, Kalman.

3. **Perspective** (stereoscopic quantization): integrates evidence
   from multiple photometric projections. Improved by adding
   complementary eyes.

These three sources are orthogonal:
- Retrieval operates on block-level matches (local)
- Ranking operates on image-level similarity (global)
- Perspective operates on quantization-level diversity (meta)

The current best results use sources 1+2 (MT4 stack: 42.05%) or
source 3 alone (stereo Bayesian: 41.18%). **Nobody has combined
all three.**

## The Actual Next Experiment

**Combine stereoscopic retrieval + MT4 ranking + Bayesian fusion.**

1. Vote across all 3 eyes' inverted indices (stereoscopic retrieval)
2. Rank candidates using MT4 dot + topo features (deep ranking)
3. Score with combined Bayesian posterior from all 3 eyes (perspective)

This is the first time all three sources of classification power
would be engaged simultaneously.

Prediction: 43-44%. The stereo voting would retrieve different (better)
candidates than single-eye voting. The MT4 ranking would order them
better than Bayesian alone. The combined posterior would score them
with more evidence than single-eye Bayesian.

## What the LMM Reveals About the Project

The SSTT project has followed a consistent pattern:
1. Discover a principle on MNIST (ternary ALU, vote-then-refine, etc.)
2. Push it to diminishing returns
3. Discover a new principle
4. Repeat

The stereoscopic principle is the latest discovery. But it was found
on CIFAR-10, not MNIST. The project should loop back:

- Apply stereo to Fashion-MNIST (where it should help)
- Apply stereo to the MNIST/Fashion full pipeline (contribution 35
  validation numbers)
- Combine stereo + MT4 on CIFAR-10
- Then publish everything

The paper now has THREE stories:
1. The MNIST/Fashion story (97%/86%, zero parameters, honest validation)
2. The CIFAR-10 boundary story (26% → 42%, what generalizes and what doesn't)
3. The stereoscopic principle (multi-perspective quantization as a
   general technique)

Three stories in one paper is ambitious but they support each other.
The third story (stereo) REQUIRES the first two for context — you need
to understand where the architecture succeeds (MNIST) and where it
fails (CIFAR-10 brightness confound) to see why multiple perspectives
help.
