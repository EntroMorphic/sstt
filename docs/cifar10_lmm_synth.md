# CIFAR-10 Next Steps — SYNTHESIZE

Clear action plan from the LMM analysis.

---

## Key Decision

**Replace fixed ternary thresholds (85/170) with per-image adaptive
thresholds (33rd/67th percentile of pixel distribution).**

### Rationale

The correlation analysis proved that brightness is the #1 predictor of
classification success. The system classifies by photographic properties
(lighting, color palette) rather than content (shape, structure). Fixed
thresholds encode absolute brightness; adaptive thresholds encode relative
structure. This is the single highest-leverage change available within the
zero-parameter constraint.

### What This Changes

| Property | Fixed (85/170) | Adaptive (P33/P67) |
|----------|---------------|-------------------|
| Trit distribution | Varies by image (29/48/23) | Exactly 33/33/33 |
| Bright image | Mostly +1 trits | Same as dark image |
| Dark image | Mostly -1 trits | Same as bright image |
| Low contrast | Mostly 0 trits | Same as high contrast |
| What's encoded | Absolute brightness | Relative structure |

### What This Preserves

- Zero learned parameters (percentile is a fixed transformation)
- Integer arithmetic (compute percentiles once, then quantize)
- Block encoding, Encoding D, inverted index (all unchanged)
- IG weighting (auto-adapts to new distributions)
- Full pipeline: vote → dot → topo → Bayesian

---

## Action Plan

### Experiment 1: Adaptive Ternary (per-image, grayscale → flattened RGB)

```
For each pixel in each image:
  1. Compute sorted pixel values for this image
  2. P33 = pixel_values[n/3], P67 = pixel_values[2*n/3]
  3. pixel < P33 → -1, pixel > P67 → +1, else → 0
```

Apply to flattened RGB (32×96). Run Bayesian + full stack.
Compare to fixed-threshold baseline (36.58% Bayesian, 42.05% stack).

### Experiment 2: Per-Channel Adaptive (R, G, B independently)

Compute P33/P67 for each color channel separately within each image.
This normalizes for color temperature (bluish vs greenish lighting)
in addition to brightness.

### Experiment 3: Adaptive + MT4

If Experiment 1 helps: adaptive thresholds give uniform trit
distributions. Does MT4 still help, or does adaptive quantization
subsume the MT4 benefit?

### Experiment 4: Adaptive + Full Stack + Fine Grid

If Experiments 1-2 help: add topo features, Bayesian prior, and
test finer grid divergence (8×8) on the adaptively-quantized images.

---

## Success Criteria

- [ ] Adaptive ternary Bayesian > 37% (current fixed: 36.58%)
- [ ] Adaptive full stack > 42% (current: 42.05%)
- [ ] Airplane accuracy less dependent on brightness
- [ ] Cat accuracy > 20% (current: 8-16%)
- [ ] Per-class accuracy more uniform across brightness conditions

## Risk

If adaptive thresholds don't help (Bayesian ≤ 36.58%), then the
representation ceiling is NOT caused by brightness variation but by
the ternary block vocabulary itself. In that case, document as a
negative result and close the CIFAR-10 experimental program.

---

## Files

- `docs/cifar10_lmm_raw.md` — Unfiltered thinking
- `docs/cifar10_lmm_nodes.md` — Key points and tensions
- `docs/cifar10_lmm_reflect.md` — Underlying structure and leverage
- `docs/cifar10_lmm_synth.md` — This action plan
