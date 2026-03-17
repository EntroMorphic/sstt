# CIFAR-10 Next Steps — RAW

Unfiltered thinking on what the correlation analysis revealed and what
it means for breaking through 42%.

---

The system classifies by photographic properties, not content. Bright
blue images → airplane/ship. Dark green → frog. High contrast → truck.
This works when the class has a distinctive color signature and fails
when it doesn't (cat, bird, deer — all "medium everything").

The ternary quantization at 85/170 is effectively a color histogram.
Three bins per channel: dark/medium/bright. A 3-pixel block of
interleaved RGB captures (R_trit, G_trit, B_trit) = one of 27 color
categories. The inverted index counts which training images share the
same color category at the same position. This IS a color histogram
with spatial addressing.

Why does this work at 42% and not higher? Because color+position is
enough to distinguish classes with unique color palettes (frog=dark
green, ship=blue+gray, truck=dark+high contrast) but not classes that
share palettes (cat≈dog, deer≈frog at different brightness, bird≈
airplane against sky).

The failing images have moderate brightness (102-154), moderate
contrast (20-60), moderate color dominance (<40). They sit in the
center of every feature distribution. The system can't find distinctive
patterns because there ARE no distinctive ternary patterns — the image
is "average" in every quantized dimension.

What would help? Something that captures SHAPE rather than color. The
divergence theorem captures closed loops — a shape property — and it
helps (+2.8pp). But only on the grayscale spatial image, not on the
color-rich flattened representation. We haven't tried divergence on
per-POSITION color patterns.

Wait — the flattened RGB blocks encode color at each position. The
horizontal gradient between adjacent blocks captures color transitions.
But the transition between block (R₁,G₁,B₁) and block (R₂,G₂,B₂)
in the flattened representation is just the spatial gradient of the
color — which is an EDGE in color space. The divergence of this field
would capture closed color contours. We computed this but it only
helped +0.05pp.

Why didn't per-channel divergence help more? Because divergence on
32x32 is a GLOBAL scalar summary. It says "this image has X amount of
closed loops" but not WHERE they are or what SHAPE they form. On MNIST,
global divergence is enough because digits have consistent topology
(0 always has a loop, 1 never does). On CIFAR-10, a cat and a dog
both have similar amounts of divergence — the difference is in the
spatial ARRANGEMENT of edges, not their total quantity.

What about grid divergence? We did 3×3 grids — 9 regional divergence
values. This captures WHERE the divergence concentrates. It helped
(+2.8pp combined with global). But 3×3 might be too coarse for the
cat-vs-dog distinction. What about finer grids — 4×4, 6×6, 8×8?

Actually, the deeper issue: even with spatial divergence, the TERNARY
gradient field on natural images doesn't have the topological structure
that digits have. A digit is a clean stroke with clear loops and
endpoints. A natural image is textured noise with fuzzy boundaries.
The divergence of textured noise is... noise.

The correlation data shows BG fraction is the strongest discriminator
(d=0.162). Images with more background (more blocks matching the most
common value) are HARDER. This makes sense: more background = less
distinctive content = harder to classify. But it also means the AMOUNT
of foreground predicts success, not the STRUCTURE of it.

The system is essentially counting how many non-average blocks an image
has and matching by their color distribution. When an image has lots of
distinctive blocks (frog: dark green everywhere), the counting works.
When it has few distinctive blocks (cat: mostly medium tones), there's
not enough signal.

Could we use DIFFERENT thresholds per class? Adaptive quantization:
for the "frog" hypothesis, use thresholds that emphasize green
gradients. For the "airplane" hypothesis, use thresholds that
emphasize blue/white boundaries. This is hypothesis-specific
quantization — like having different feature extractors per class.

But that's 10 different quantization schemes × full pipeline each.
Computationally expensive and risks overfitting.

What about the center energy finding? Birds that we get RIGHT have
high center energy (0.342 vs 0.247). The bird IS the image when it's
centered and close-up. When it's small and off-center against sky,
it looks like an airplane. Object localization — knowing WHERE in the
image the object is — would help. But our blocks treat every position
equally (just with different IG weights).

The strongest per-class signal: airplane brightness delta (+41.3).
Bright airplanes against blue sky are trivially classified. Dark
airplanes against clouds or at night are impossible. This is a
LIGHTING problem, not a representation problem. The system can't
normalize for lighting because the ternary thresholds are fixed.

What if we normalized brightness per image before quantizing? Stretch
the histogram to use the full 0-255 range, THEN quantize. This would
make "dark airplane" look like "bright airplane" in ternary space.
Histogram equalization before ternary quantization.

That's actually a clean, zero-parameter transformation. No learning
required. Just map each image's intensity range to [0, 255] before
quantization. Every image would use the full ternary range. Dark
images would no longer cluster together.

This could be the breakthrough. The correlation data says brightness
is the #1 per-class predictor of success. Remove brightness as a
variable and force the system to classify on structure alone.
