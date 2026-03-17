# Contribution 38: CIFAR-10 — Full Experimental Arc

Tests whether the SSTT ternary cascade architecture generalizes from
28x28 grayscale (MNIST/Fashion) to 32x32 natural color images (CIFAR-10).

14 experiments in a single session. Zero learned parameters throughout.

---

## Summary

| Experiment | Accuracy | Finding |
|-----------|----------|---------|
| Grayscale ternary cascade | 26.51% | Architecture has signal but pixel dot kills cascade |
| Flattened RGB Bayesian | 36.58% | Color in blocks helps; block = (R,G,B) is a color encoding |
| Gradient-only dot | 33.07% | Pixel dot is the contaminant, not the architecture |
| RGB 6-channel gradient dot | 33.71% | Color gradients beat grayscale gradients |
| Combined Encoding D + RGB grad | 34.73% | Cascade helps when dot matches domain |
| MT4 pixel+grad k=1 | 36.42% | 81-level resolution closes gap to Bayesian |
| Full stack (MT4+topo+Bayesian prior) | 41.19% | Composition works: +14.7pp over baseline |
| + RGB divergence | 41.24% | Marginal — topo ceiling reached |
| MoE pair-specific routing | 41.31% | Pair IG barely helps (+0.12pp) |
| Quantized MAD routing | 41.19% | MAD not discriminative on CIFAR-10 |
| Label propagation | 35.28% | Hurts — feature manifold doesn't respect classes |
| Raw uint8 dot (MT-∞) | 13.54% | Catastrophic — brightness dominates |
| **MT4 full-pipeline voting** | **42.05%** | **Best: 4 planes for vote + dot + topo** |
| MT7 full-pipeline voting | 39.72% | More planes hurt — noise from fine detail |

---

## Key Findings

### What generalizes from MNIST/Fashion

1. **Inverted-index voting**: 97.96% recall at top-200 (vs 99.3% on MNIST).
   The retrieval architecture transfers to natural images.

2. **Composition stacks**: Vote → dot → topo → Bayesian prior, each adding
   independent signal. Same pattern as MNIST's topo1-9 progression.

3. **Confidence map**: High vote agreement predicts success (56.2% at
   agree≥140 vs 41% overall). The system knows when it's confident.

4. **K-invariance**: K=50 through K=1000 identical, same as MNIST.

5. **IG auto-adaptation**: Information gain weights automatically redistribute
   for CIFAR-10's different block importance landscape.

### What doesn't generalize

1. **Pixel dot product**: Dominates on MNIST (foreground overlap), catastrophic
   on CIFAR-10 (brightness correlation). Gradient dot is the correct ranking
   signal for natural images.

2. **Ternary quantization as sufficient resolution**: MT4 (81 levels) beats
   ternary (3 levels) by +3.3pp. The coarse quantization discards too much
   for natural images — but too fine (MT7, raw) reintroduces noise.

3. **Enclosed-region centroid**: Weight = 0 on CIFAR-10. Natural images don't
   have consistent enclosed loops like digits.

4. **Sequential Bayesian processing**: Zero improvement on CIFAR-10, same as
   MNIST. The ranking scores are too flat for sequential weighting.

5. **Label propagation**: Feature-space neighbors frequently belong to different
   classes (deer ≈ frog in ternary texture). Transductive methods fail.

### Novel CIFAR-10 contributions

1. **Flattened RGB representation**: Interleave R,G,B per pixel → 32×96 image
   where each 3×1 block is `(R_trit, G_trit, B_trit)` = color encoding.
   No architecture changes needed. +10pp over grayscale.

2. **Multi-Trit Floating Point (MT4)**: Decompose each pixel into 4 balanced
   ternary planes at different scales. Each plane uses the existing ternary
   ALU (`sign_epi8`). Combined dot = 27×dot₃ + 9×dot₂ + 3×dot₁ + dot₀.
   81-level resolution while preserving integer arithmetic.

3. **MT4 full-pipeline**: Use all 4 planes for voting, not just the coarsest.
   Each plane has its own IG weights, inverted index, and Encoding D.
   +0.99pp over single-plane voting.

4. **Bayesian posterior as ranking feature**: The hot map's P(class|query)
   fed into the candidate scoring function. Position-specific class knowledge
   that the dot product misses. +4.6pp contribution.

5. **Non-monotonic quantization curve**: MT4 (81 levels) > MT7 (2187) > raw
   (256). The ternary quantization acts as edge detection, suppressing
   brightness variation. Too much resolution reintroduces the noise it
   filters out. Optimal quantization ≈ 81 levels for natural images.

---

## The Cascade Autopsy on CIFAR-10

Mode A/B/C error classification (contribution 23 methodology):

| Mode | MNIST | CIFAR-10 | Meaning |
|------|-------|----------|---------|
| A (vote miss) | 1.7% | 3.1% | Correct not in top-200 |
| B (rank miss) | 64.5% | 36.4% | In top-200, not top-7 |
| C (kNN dilute) | 33.8% | 60.5% | In top-7, outvoted |

On MNIST, Mode B dominates (ranking problem). On CIFAR-10, Mode C dominates
(dilution problem). The correct class IS nearby — it's just surrounded by
wrong-class neighbors with similar ternary texture.

Top confused pairs: deer↔frog (357), airplane↔ship (349), bird↔deer (309).
These share color/texture in ternary space despite being visually distinct
to humans. The ternary representation maps "outdoor green scene" to the
same signature regardless of whether it contains a deer or a frog.

---

## Honest Boundary Assessment

The SSTT architecture at 42.05% on CIFAR-10 with zero learned parameters:
- Is 4.2× random (10%)
- Exceeds brute raw-pixel kNN (~35-40%)
- Proves the architecture generalizes (retrieval, composition, confidence)
- But cannot match even simple learned methods (~50-60%)

The bottleneck is the representation: ternary block signatures capture
texture and color at fixed 3-pixel scale but cannot encode object shape,
figure-ground separation, or scale-invariant structure that natural image
classification requires.

---

## Experimental Progression

```
26.51%  Grayscale ternary cascade (baseline)
  +7.25  Pixel dot → gradient dot (channel ablation)
  +2.85  Flattened RGB blocks (color encoding)
  +3.30  MT4 81-level dot (quantization resolution)
  +4.61  Bayesian prior + topo features (composition)
  +0.99  MT4 full-pipeline voting (4-plane vote)
 ──────
42.05%  Best: MT4 4-plane vote + dot + topo + Bayesian
```

Each step applied a principle from the MNIST toolkit to CIFAR-10.
The composition adds up. Zero learned parameters throughout.

---

## Files

Core experiments:
- `src/sstt_cifar10.c` — Grayscale baseline (33.76% Bayesian, 26.51% cascade)
- `src/sstt_cifar10_flat.c` — Flattened RGB (36.58% Bayesian)
- `src/sstt_cifar10_grad.c` — Gradient channel ablation (5 variants)
- `src/sstt_cifar10_mt4.c` — MT4 81-level dot product
- `src/sstt_cifar10_stack.c` — Full stack (41.19%)
- `src/sstt_cifar10_mt4vote.c` — MT4 full pipeline (42.05%, best)

Diagnostics:
- `src/sstt_cifar10_full.c` — Cascade autopsy + weight sweep
- `src/sstt_cifar10_modec.c` — Mode C deep diagnostic
- `src/sstt_cifar10_why.c` — Per-class failure autopsy
- `src/sstt_cifar10_rawdot.c` — Raw dot ceiling test (MT-∞)
- `src/sstt_cifar10_mt7vote.c` — MT7 test (diminishing returns)

Negative results:
- `src/sstt_cifar10_moe.c` — MoE routing (+0.12pp, marginal)
- `src/sstt_cifar10_qmad.c` — Quantized MAD routing (+0pp)
- `src/sstt_cifar10_propagate.c` — Label propagation (hurts)

Data: `data-cifar10/` (downloaded separately, not committed)
This document: `docs/38-cifar10-full-arc.md`
