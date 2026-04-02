# Contribution 37: CIFAR-10 Boundary Test

Tests whether the SSTT ternary cascade architecture generalizes from
28x28 grayscale (MNIST/Fashion) to 32x32 natural color images (CIFAR-10).

Code: `src/sstt_cifar10.c`

---

## Methodology

**Simplest faithful test:** change the data, keep the method.

- RGB → grayscale (Y = 0.299R + 0.587G + 0.114B)
- 32x32 → 10 blocks per row (3x1, same as MNIST's 9)
- Same ternary thresholds (85/170)
- Same Encoding D, IG weighting, multi-probe voting, dot ranking, kNN
- No tuning for CIFAR — identical pipeline to `sstt_bytecascade.c`

---

## Results

| Method | Accuracy | Context |
|--------|----------|---------|
| Random baseline | 10.00% | — |
| **Bytepacked Bayesian** | **33.76%** | Position-specific class frequencies |
| Multi-probe IG vote-only | 28.88% | Inverted index vote, argmax |
| Full cascade (K=50, k=7) | 26.51% | Vote → dot → kNN |
| Brute kNN (literature) | ~35-40% | Raw pixel L2 distance |

### Key observations

**1. The architecture has signal.** 33.76% is 3.4x random. The ternary
block encoding captures enough structure in natural images to beat
chance by a wide margin.

**2. The cascade hurts.** The Bayesian hot map (33.76%) outperforms
the full cascade (26.51%) by +7.25pp. This is the opposite of MNIST,
where the cascade dominates (+4pp over Bayesian). The ternary dot
product on grayscale natural images destroys discriminative information
that the per-block-per-class frequency table preserves.

**3. K-invariance still holds.** K=50 through K=1000 produce identical
accuracy (26.51%). The retrieval step works — the inverted index still
finds relevant neighbors. The problem is entirely in the dot product
ranking step.

**4. Trit distribution is flatter.** MNIST is highly sparse (background-
dominated). CIFAR-10 trits: -1=29.6%, 0=47.7%, +1=22.6%. Only 18.6%
of positions match the background value (vs ~60%+ on MNIST). The
inverted index is denser: 13M entries vs 5.6M for MNIST.

**5. IG weights are more uniform.** min=6, max=16, avg=9.8 (on a 1-16
scale). MNIST has much higher variance — edge/corner blocks are highly
informative, center blocks less so. CIFAR-10 blocks carry more uniform
information, which makes IG weighting less discriminative.

**6. Confused pairs are semantically meaningful.** airplane↔ship (413
errors), cat↔dog (340), bird↔cat (310) — these are the visually
similar classes, confirming the system extracts real structure.

---

## Why the cascade fails on CIFAR-10

The SSTT cascade works on MNIST because:

1. **Sparse ternary representation** — most pixels are background (0),
   the few non-zero regions carry digit identity
2. **Ternary dot product** — on sparse binary-like images, dot product
   measures overlap of foreground strokes
3. **Block signatures** — 3-pixel blocks on digit strokes are highly
   class-specific

On CIFAR-10, all three properties break:

1. **Dense representation** — natural images have texture everywhere,
   no sparse foreground/background separation
2. **Dot product on texture** — ternary dot measures correlation of
   quantized intensity patterns, which is dominated by lighting and
   background texture, not object identity
3. **Block signatures on texture** — 3-pixel texture blocks are
   position-dependent and class-independent (a patch of blue sky
   looks the same whether it's behind an airplane or a ship)

The Bayesian hot map partially survives because it encodes position:
"this specific texture pattern at this specific block position" carries
more class information than "overall texture similarity." The cascade
discards position information in the dot product step.

---

## What would help

1. **Color channels.** Processing R, G, B independently and voting
   across channels would preserve color information (blue sky vs
   green grass) that grayscale discards.

2. **Adaptive thresholds.** Per-image percentile-based quantization
   (33rd/67th percentile) would normalize for lighting variation.

3. **Larger blocks or multi-scale.** 3-pixel blocks on 32x32 natural
   images are too small to capture meaningful object structure. 5x5 or
   7x7 blocks might capture edges and corners better.

4. **Keep the Bayesian, skip the dot.** The hot map result suggests
   the Bayesian classifier is the right operating point for texture-
   dominated images. The cascade's dot refinement actively hurts.

---

## Verdict

**The ternary cascade does not generalize to CIFAR-10 in its current
form.** The architecture assumes sparse, stroke-like structure that
natural images lack. However:

- The Bayesian hot map shows real signal (3.4x random)
- K-invariance confirms retrieval still works
- The failure is localized to the dot product ranking step
- Semantically meaningful confusion patterns confirm the features
  capture real structure

This is an **honest boundary result**: the method works on structured
grayscale images (MNIST, Fashion) but not on natural color images
(CIFAR-10) without significant adaptation. This should be reported as
a limitation in any publication.

---

## Files

Code: `src/sstt_cifar10.c`
Data: `data-cifar10/` (downloaded separately, not committed)
This document: `docs/37-cifar10-boundary-test.md`
Build: `make sstt_cifar10`
Run: `./sstt_cifar10`
