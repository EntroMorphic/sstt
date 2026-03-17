# Stereoscopic Quantization — LMM RAW

Unfiltered thinking on what the stereoscopic result means.

---

The stereoscopic principle works. Three eyes at 41.18% approaches the
MT4 full stack at 42.05% through a completely different mechanism.
The MT4 stack uses quantization depth + topological features + cascade
ranking + Bayesian prior. The stereo approach uses ONLY Bayesian
posterior fusion across three quantization perspectives. Simpler
architecture, comparable result.

This is a genuine architectural discovery. The insight: classification
power doesn't come from deeper processing of a single representation.
It comes from combining DIFFERENT representations of the same data.
Each quantization function is a lossy projection. The loss is different
for each function. Combining them recovers information that any single
projection discards.

The dog result (38.3% from three eyes, beating both 30.4% and 33.8%
individually) proves this isn't just averaging. If it were averaging,
the combination couldn't exceed the individual maximum. The combination
exceeds it because the three eyes make DIFFERENT errors on DIFFERENT
images. When eye 1 gets a dog wrong, eye 2 often gets it right, and
vice versa.

But we're stuck at 41%. Three eyes, five eyes, MT4, topo, MoE,
propagation — everything converges to 40-42%. Is there a fourth
orthogonal perspective we haven't tried?

What perspectives have we covered?
1. Absolute brightness (fixed)
2. Relative structure (adaptive)
3. Color relationships (per-channel)

What's missing?
- SPATIAL perspective: blocks at different positions treated differently.
  The IG weights partially handle this, but they're computed per eye.
  A spatial eye would quantize based on POSITION — center vs border,
  top vs bottom.
- FREQUENCY perspective: high-frequency texture vs low-frequency shape.
  A blurred image would capture low-frequency structure. The difference
  between sharp and blurred = high-frequency detail.
- TEMPORAL/SEQUENTIAL perspective: multiple quantization thresholds
  applied in sequence, each building on the previous. Like HDR: one
  exposure for highlights, one for shadows, combined.

Actually, the blurred idea is interesting. A 2x downsampled image
quantized at the same thresholds captures 6-pixel-scale structure.
We already tested multi-scale and it added only +0.62pp. But we
tested it as a separate pipeline, not as a stereoscopic eye fused
with the original-scale eyes.

What if we add a 2x-blurred eye to the 3-eye stereo? The blurred eye
sees shapes that the sharp eyes see as texture.

But the multi-scale experiment showed that 2x-downsampled Bayesian
(37.04%) is only +0.46pp over original (36.58%). The information at
different scales is largely redundant. Adding it as a 4th eye would
likely behave like eye 4 (median) — redundant, not complementary.

The deeper question: what information is fundamentally missing from
ALL ternary quantizations of this image?

Answer: SPATIAL RELATIONSHIPS. All our eyes encode "what value is at
this position" but not "how do distant positions relate to each other."
A cat has eyes above a nose above a mouth — a specific spatial
arrangement. A dog has a different arrangement. Neither arrangement
is captured by per-block frequencies because each block is processed
independently.

The inverted index compares images block-by-block at matching positions.
It cannot detect that "the pattern at position A is similar to the
pattern at position B" within the same image. It's purely local.

The topological features (divergence, grid divergence) partially
capture spatial relationships — "there's a closed loop in the upper
left" vs "closed loop in the lower right." But they operate on the
grayscale spatial image, not the flattened RGB, and at coarse
granularity (3x3 grid = 9 regions).

What if we had a stereoscopic eye that encodes PAIRWISE relationships
between blocks? Not individual block values, but "block A and block B
have the same/different ternary pattern." This would capture spatial
structure — symmetry, repetition, composition.

But the number of pairwise relationships is N_BLOCKS^2 = 1M. Too
many for a hot map.

Alternatively: encode spatial relationships through TRANSITION
signatures. We already compute horizontal and vertical transitions
between adjacent blocks. What about diagonal transitions? Long-range
transitions (block k and block k+10)? These would capture structure
at different spatial scales within the same eye.

Actually, this is what the Encoding D already does partially — it
packs transition activity into bits 6-7 of each block's byte value.
But it only captures adjacent transitions, and it binarizes them
(active/inactive).

I think the honest conclusion is: we've found the ceiling for
per-block-independent classification of 32x32 natural images. The
stereoscopic principle adds +4.6pp by integrating multiple
photometric perspectives, but all perspectives still process blocks
independently. The next step-change requires spatial reasoning that
the block-independent architecture cannot provide.

For the paper: stereoscopic quantization is a genuine contribution.
It's simple, zero-parameter, and adds +4.6pp through an architectural
principle (not more features). This should be documented alongside
the MNIST results as a general technique.
