# CIFAR-10 Next Steps — NODES

Key points, tensions, and constraints extracted from RAW.

---

## Key Points

1. **The system classifies by photographic properties (brightness,
   color palette, contrast), not by object content (shape, pose,
   parts).** This is the root cause of the 42% ceiling.

2. **Brightness is the #1 per-class predictor of success.**
   - Airplane: +41.3 brightness delta (correct vs wrong)
   - Frog: -39.9 brightness delta (dark frogs classified correctly)
   - Cat: only +6.7 (brightness doesn't help — cats have no
     distinctive brightness)

3. **BG fraction is the strongest global discriminator (d=0.162).**
   More background → harder. The system needs distinctive foreground
   blocks to work. "Average" images are impossible.

4. **Center energy matters for small objects.** Bird correct=0.342
   center energy vs wrong=0.247. When the object fills the center,
   it's easier. When it's small or off-center, background dominates.

5. **Contrast predicts success for animals.** Dog (+9.7), cat (+8.0),
   truck (+11.0). High contrast = clearer edges = better ternary
   gradients. Low contrast animals blur into background.

6. **Per-channel divergence (+0.05pp) was negligible.** The topology
   of ternary gradient fields on natural images is noisy — not the
   clean stroke topology of digits.

## Tensions

**T1: Fixed thresholds vs variable lighting.**
The 85/170 thresholds assume consistent brightness. Natural images
have wildly variable lighting. A dark airplane and a bright airplane
produce completely different ternary patterns despite being the same
object. Histogram equalization would resolve this but adds a
preprocessing step.

**T2: Color is both signal and noise.**
Color distinguishes frog (green) from ship (blue) — real signal.
But color also makes deer (green outdoor) look like frog (green) —
noise. The same feature helps and hurts depending on the class pair.

**T3: Block size vs vocabulary size.**
3-pixel blocks × 27 values = manageable inverted index.
Larger blocks (shape-sensitive) → exponentially more values.
PCA to reduce them destroys spatial addressing (contribution 17).
No known way to capture shape in a small discrete vocabulary.

**T4: Spatial features vs position-invariance.**
The hot map USES position (block k at position p). This helps when
objects are consistent (frog = green everywhere). It hurts when
objects vary in position/scale (bird in center vs corner).
Position-invariant features (divergence) help but not enough.

**T5: Zero parameters vs adaptive quantization.**
Histogram equalization is zero-parameter but image-specific.
Does this violate "no learned parameters"? No — it's a fixed
transformation (normalize to [0,255]) applied uniformly. Like
the existing 85/170 thresholds, but adaptive to the image.

## Constraints

- Zero learned parameters (project constraint)
- Integer arithmetic preferred (architectural constraint)
- Must work with inverted index (retrieval constraint)
- Block vocabulary ≤ 256 values (Encoding D constraint)
- ~1 ms latency target (performance constraint)

## Leverage Points

**L1: Histogram equalization before quantization.**
Removes brightness as a confound. Forces classification on structure.
Zero parameters. Directly addresses the #1 predictor of failure.
Clean, simple, testable in one experiment.

**L2: Per-channel histogram equalization.**
Normalize R, G, B independently. This removes both brightness AND
color saturation as confounds. Forces classification on color
RELATIONSHIPS (which channel is brightest) not absolute values.

**L3: Contrast-adaptive thresholds.**
Instead of fixed 85/170, use percentile thresholds per image.
33rd and 67th percentile → equal ternary distribution. Every image
would have exactly 1/3 dark, 1/3 medium, 1/3 bright. Removes
brightness AND contrast variation.

**L4: Finer grid divergence.**
8×8 grid on 32×32 = 4×4 pixel regions. Captures local edge structure
at finer granularity than 3×3. Might distinguish cat face structure
from dog face structure.

**L5: Object-centric blocks.**
Weight center blocks more heavily than border blocks. The correlation
data shows center energy predicts success for bird, deer, frog. An
IG-like weighting that favors center positions would help centered
objects and hurt border-dominated images less.
