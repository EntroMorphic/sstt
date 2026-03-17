# CIFAR-10 Next Steps — REFLECT

Underlying structure, unstated assumptions, and actual leverage.

---

## The Unstated Assumption

The entire SSTT architecture assumes that **raw pixel values carry
class-discriminative information at the block level.** On MNIST, this is
true: a digit's stroke IS its pixel pattern. On CIFAR-10, this assumption
breaks: a cat's pixel pattern is dominated by LIGHTING and BACKGROUND, not
by the cat itself.

Every technique we've tried — MT4, topo features, Bayesian prior, MoE,
multi-scale, label propagation — operates on features derived from raw
pixels. If the raw pixels don't carry class information at the block level,
no amount of processing can extract it.

But the raw pixels DO carry class information — just entangled with
irrelevant variation. The correlation data proves this: when lighting
conditions are favorable (bright airplane, dark frog, high-contrast truck),
the system works at 50-60%. The information is there; it's buried under
brightness/contrast variation.

## The Real Tension

The fundamental tension is not block size, quantization depth, or
aggregation method. It's:

**Fixed quantization thresholds assume photometric consistency that
natural images lack.**

On MNIST, photometric consistency is built in: black background, white
foreground, consistent contrast. The 85/170 thresholds reliably separate
foreground from background on every image.

On CIFAR-10, photometric properties vary wildly:
- Brightness: 50-220 range across images
- Contrast: 15-90 range
- Color balance: blue-dominant (sky) to green-dominant (outdoor) to
  gray (indoor)

The ternary quantization encodes ABSOLUTE brightness into the
representation. Two images of the same cat — one in sunlight, one in
shadow — produce completely different ternary patterns. The inverted
index cannot match them because their block signatures don't overlap.

## Where the Real Leverage Is

**L1 (Histogram equalization) is the correct intervention.** Here's why:

1. It directly addresses the #1 predictor of failure (brightness variation)
2. It's zero-parameter (fixed transformation, not learned)
3. It preserves the existing architecture completely
4. It's mathematically principled: it maximizes the entropy of the
   pixel distribution, ensuring the full ternary range is used by
   every image regardless of lighting
5. It converts the quantization from "absolute brightness bins" to
   "relative brightness bins" — which is closer to what the eye does

The specific implementation should be **per-channel percentile
quantization** (L3 from NODES). Instead of fixed 85/170:

```
For each image, for each pixel:
  Sort all pixels → find 33rd and 67th percentile
  pixel < P33 → -1
  pixel > P67 → +1
  else → 0
```

This guarantees exactly 1/3 of each trit value per image. Every image
uses the full ternary range. Brightness, contrast, and color
saturation variation are eliminated. The ternary pattern encodes only
RELATIVE structure — which pixels are darker/brighter than their
neighbors within this specific image.

**This is the same insight as the "quantization is a feature, not a
loss" finding from the raw dot experiment — but taken further.** The
ternary quantization helps by suppressing irrelevant variation. Fixed
thresholds suppress some variation (absolute brightness). Adaptive
thresholds suppress ALL brightness/contrast variation. The ternary
pattern becomes purely structural.

## Why L4 (Finer Grid) and L5 (Center Weighting) Are Secondary

Finer grid divergence would help on the margin, but it operates on the
SAME features that already can't distinguish cat from dog. If the ternary
representation is dominated by lighting, a finer spatial decomposition
of that lighting doesn't add shape information.

Center weighting would help the specific bird case but hurt objects that
aren't centered. It's a dataset-specific hack, not an architectural insight.

Both are worth trying AFTER fixing the brightness confound. If adaptive
quantization lifts the ceiling from 42% to 48%+, THEN finer grids and
center weighting become meaningful refinements.

## What Could Go Wrong

1. **Adaptive thresholds might destroy IG weights.** If every image has
   exactly 1/3 of each trit, the block value distributions become more
   uniform. IG weights might flatten. Counter: the SPATIAL PATTERN of
   trits still varies by class even if the global proportions are fixed.

2. **Background becomes harder to detect.** With fixed thresholds,
   uniform background maps to a consistent block value. With adaptive
   thresholds, background gets distributed across all values. The
   background-skipping optimization loses effectiveness. Counter: the
   system already works with only 18.7% background on CIFAR-10.

3. **MT4 decomposition becomes meaningless.** If thresholds are per-image,
   the balanced ternary decomposition can't use global bins. Counter:
   just use the adaptive ternary (1 plane) without MT4. If it beats
   MT4+fixed, the adaptive threshold is the bigger win.

## The Experiment Order

1. **Per-image adaptive ternary quantization (percentile-based)**
   Test: replace fixed 85/170 with per-image 33rd/67th percentile.
   Keep everything else identical. If this lifts Bayesian above 37%,
   proceed.

2. **Per-channel adaptive quantization**
   Test: normalize each R, G, B channel independently within each
   image. This removes both brightness AND color temperature variation.

3. **Adaptive quantization + MT4 full stack**
   If step 1 helps, combine with the full pipeline.

4. **Finer grid divergence on adaptively-quantized images**
   Now that structure is emphasized over lighting, spatial divergence
   might be more discriminative.
