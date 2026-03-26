# Contribution 28: Confidence-Gated Hybrid Architecture

Extracted from the architectural analysis in the original doc 20. The
useful content — the hot-map/cascade spectrum insight, the confidence
gating design, and the IG prediction miss — stands on its own.

---

## The Hot Map / Cascade Spectrum

The hot map and cascade are not separate architectures. They are two
ends of a spectrum:

- **Hot map:** Fully collapsed. All per-image identity aggregated into
  per-class counts. Fast (1-5 μs), moderate accuracy (73-92%).
- **Cascade:** Fully expanded. Every per-image vote preserved. Slow
  (600-1100 μs), high accuracy (96%+).

Every composite method in this project sits between these extremes.

## The 73% → 96% Gap Decomposition

The cascade's +23pp over the hot map comes from three sources:

| Enhancement | Predicted gain | Actual gain in hot map | Why the gap |
|-------------|---------------|----------------------|-------------|
| IG weighting | +15pp | +0.54pp | Cascade IG multiplies per-image votes; hot map IG multiplies per-class aggregates |
| Multi-probe | +2pp | +0.26pp | Hot map multi-probe adds at class level, losing per-image discrimination |
| Dot refinement | +5pp | N/A (requires per-image state) | Cannot be done in pure table-lookup mode |

The key finding: **IG pre-baking in the hot map gives only +0.54pp,
not +15pp**, because the hot map collapses per-image identity. The IG
weight multiplies aggregate counts that are already class-averaged,
not individual match signals. The cascade's IG operates on per-image
votes where the weight amplifies the signal of a specific match.

This means the gap between hot map and cascade is not primarily about
missing weights or features — it's about the **loss of per-image
identity** when compressing an inverted index into a frequency table.

## Confidence-Gated Hybrid

The practical solution: don't choose between hot map and cascade. Use
both, gated by confidence.

```
Raw pixels
    │
    ▼  Fused kernel (hot map, ~1-5 μs)
    │
    ├─ High confidence (margin > T) → DONE
    │       ~85% of images, near-instant
    │
    └─ Low confidence (margin ≤ T) → cascade fallback
            │
            ▼  Vote → top-K → dot → k=3 → DONE
                ~15% of images, ~600-1100 μs
```

**Expected:** ~95-96% accuracy at ~10-20 μs average latency, because
85% of images never touch the cascade.

### What sstt_hybrid.c showed

The first attempt at this architecture (sstt_hybrid.c, contribution 19
in the benchmark catalogue) achieved 87.80% with 30% cascade fallback.
It fell short because:

- The hot map's Bayesian posterior is too peaked — margin ≈ 1.0 for most
  images, making the confidence signal nearly useless for gating
- The 70% "high confidence" images were classified at only 85.87%
- IG pre-baking and multi-probe gave marginal gains (+0.54pp, +0.26pp)

The fix is not a better hot map — it's a better confidence signal. The
bytepacked cascade's vote distribution is much more peaked and
meaningful. The oracle architecture (contribution 25) proved this: kNN
vote unanimity is a well-calibrated gate (92% of images, 98.8% accuracy
on the fast path).

## Implementation Plan

**Step 1: Pre-baked IG hot map** — zero runtime cost, +0.54pp
- Multiply IG weights into hot map at training time
- Same fused kernel code, different table values

**Step 2: Fused multi-probe** — 7x more lookups, +0.26pp
- Look up 6 Hamming-1 neighbors per block at half weight
- Still pure table operations, still L2-resident

**Step 3: Confidence margin** — add to fused kernel
- Return (class, margin) where margin = top1_score - top2_score
- Gate: high margin → done, low margin → cascade

**Step 4: Cascade fallback** — for low-confidence images only
- Use existing bytepacked cascade pipeline
- Only triggered for the ~15% where hot map is uncertain

### Success Criteria

- Pre-baked IG hot map exceeds 83% on MNIST
- Multi-probe hot map exceeds 87% on MNIST
- Confidence-gated hybrid achieves 95%+ on MNIST
- Average latency under 20 μs

## Files

- `sstt_hybrid.c`: First confidence-gated attempt (87.80%)
- `sstt_bytecascade.c`: Cascade fallback pipeline
- `sstt_fused_c.c`: Fused kernel to be modified
- Original analysis: `archived/docs/20-lmm-pass-composite-architecture.md`
