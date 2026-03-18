# Contribution 39: Stereoscopic Multi-Perspective Quantization

A new architectural primitive for SSTT: multiple quantization functions
applied to the same image, each revealing different structure. The
combined Bayesian posterior fuses all perspectives. Zero learned
parameters.

Code: `src/sstt_cifar10_stereo.c`, `src/sstt_cifar10_dual.c`,
`src/sstt_cifar10_adaptive.c`

---

## Discovery Path

### The correlation analysis (sstt_cifar10_correlate.c)

Measured 12 features per test image and compared distributions between
correctly and incorrectly classified images. Key findings:

- **Brightness is the #1 per-class predictor of success.** Airplane:
  +41.3 brightness delta (correct vs wrong). Frog: -39.9 delta.
- **BG fraction is the strongest global discriminator** (d=0.162).
- **The system classifies by photographic properties** (brightness,
  color palette, contrast), not by object content (shape, pose).

### Adaptive Quantization Analysis

The key insight is that **adaptive quantization converts the
representation from absolute brightness bins to relative structure
bins.** This is the same principle as "quantization is a feature"
from the raw dot experiment, taken further.

Adaptive P33/P67 quantization:

| Class | Fixed 85/170 | Adaptive P33/P67 | Delta |
|-------|-------------|------------------|-------|
| cat | 8.2% | **18.0%** | **+9.8pp** |
| automobile | 37.6% | **45.8%** | **+8.2pp** |
| airplane | **50.4%** | 33.4% | **-17.0pp** |
| ship | **50.2%** | 37.8% | **-12.4pp** |

Adaptive helps classes where brightness is noise (cat, automobile) but
hurts classes where brightness IS the signal (airplane, ship). This
confirms the tension between scene-level signal and object-level
structure.

### The stereoscopic insight

If fixed thresholds see scenes (blue sky → airplane) and adaptive
thresholds see structure (edge pattern → airplane regardless of sky),
then **running both pipelines and combining their posteriors captures
both perspectives.** Like binocular vision: two projections of the same
object, and the difference between them carries information neither
contains individually.

---

## Architecture

```
Image ──┬── Eye 1: Fixed 85/170 ──── Encoding D ── Hot Map ── log P(c|blocks)
        │
        ├── Eye 2: Adaptive P33/P67 ── Encoding D ── Hot Map ── log P(c|blocks)
        │
        └── Eye 3: Per-channel P33/P67 ── Encoding D ── Hot Map ── log P(c|blocks)
                                                                         │
                                Combined posterior: sum of log-posteriors ◄┘
                                         │
                                    argmax → class
```

Each eye:
1. Applies its quantization function to the raw image
2. Computes ternary gradients (h-grad, v-grad)
3. Computes block signatures, transitions, Encoding D
4. Builds a Bayesian hot map from training data
5. Produces a 10-class log-posterior for each test image

The combined posterior sums the log-posteriors across all eyes.
IG weights per eye automatically learn which blocks are discriminative
from that quantization perspective.

---

## The Five Eyes Tested

| Eye | Quantization | What it sees | Accuracy |
|-----|-------------|-------------|----------|
| 1 | Fixed 85/170 | Absolute brightness: blue sky, dark animals | 36.58% |
| 2 | Adaptive P33/P67 | Relative structure: edges regardless of lighting | 36.95% |
| 3 | Per-channel P33/P67 | Color relationships: R/G/B ratios normalized | 35.98% |
| 4 | Median split (binary) | Foreground/background only: shape silhouette | 34.12% |
| 5 | Wide P20/P80 | Extreme values only: highlights and shadows | 36.05% |

---

## Results

### Progressive combination

| Eyes | Accuracy | Delta vs 1 eye |
|------|----------|---------------|
| 1 (fixed) | 36.58% | baseline |
| 1+2 (+ adaptive) | 40.20% | +3.62pp |
| **1+2+3 (+ per-channel)** | **41.18%** | **+4.60pp** |
| 1+2+3+4 (+ median) | 40.67% | declining |
| 1+2+3+4+5 (+ wide) | 40.37% | declining |

### Exhaustive subset search (all 31 combinations)

Best combination: **Fixed + Adaptive + Per-channel = 41.18%**

Eyes 4 and 5 provide perspectives too similar to the existing three.
Diminishing returns past 3 complementary views.

### Per-class gains (1 eye → 3 eyes)

| Class | 1 eye | 3 eyes | Gain |
|-------|-------|--------|------|
| airplane | 50.4% | 48.1% | -2.3pp (small loss from diluted sky signal) |
| automobile | 37.6% | 43.1% | +5.5pp |
| bird | 16.9% | 19.8% | +2.9pp |
| **cat** | **8.2%** | **17.2%** | **+9.0pp** |
| **deer** | **26.5%** | **36.3%** | **+9.8pp** |
| **dog** | **30.4%** | **38.3%** | **+7.9pp** |
| frog | 58.8% | 62.7% | +3.9pp |
| horse | 31.0% | 36.3% | +5.3pp |
| ship | 50.2% | 49.6% | -0.6pp |
| truck | 55.8% | 60.4% | +4.6pp |

Every animal class improves significantly. The stereo combination
captures structural similarity (from adaptive eyes) while preserving
scene-level signals (from the fixed eye).

---

## Why It Works

### The dog proof

Dog: Fixed=30.4%, Adaptive=33.8%, Combined=37.3% (dual), 38.3% (triple).
The combination beats BOTH individual eyes. This is not averaging —
it's integration of complementary information.

The fixed eye sees "high contrast close-up" → probably dog.
The adaptive eye sees "specific edge pattern regardless of lighting" → probably dog.
The per-channel eye sees "warm color relationships (more red than blue)" → probably dog.

Each eye is uncertain alone. Together, the three weak signals compound
into a stronger signal because they're independent — a dog looks like
a dog from all three perspectives, while a cat-that-looks-like-a-dog
from one perspective doesn't from another.

### Information-theoretic interpretation

Each quantization function projects the image onto a different subspace.
Fixed thresholds project onto the brightness axis. Adaptive thresholds
project onto the local-contrast axis. Per-channel adaptive projects
onto the color-ratio axis. The combined posterior integrates across
projections — similar to how multiple sensors in sensor fusion provide
more reliable estimates than any single sensor.

The IG weights per eye ensure that each eye's contribution is weighted
by its actual discriminative power at each block position. Blocks where
the fixed eye is informative get high IG in eye 1; blocks where
adaptive is informative get high IG in eye 2.

### Why 3 eyes, not 5

Eyes 4 (median split) and 5 (wide adaptive) are too similar to
existing eyes:
- Median is a degenerate case of adaptive (no zero band)
- Wide adaptive is a conservative version of adaptive (wider zero band)

Neither provides a genuinely new perspective on the data. The three
complementary views are:
1. **Absolute** (what brightness values are present)
2. **Relative** (what the local structure looks like)
3. **Chromatic** (what the color relationships are)

These three span the photometric variation space. Additional eyes
within this space add redundancy, not information.

---

## Comparison to Prior CIFAR-10 Results

| Method | Accuracy | Mechanism |
|--------|----------|-----------|
| Grayscale ternary | 26.51% | Single eye, wrong channel |
| Flattened RGB fixed | 36.58% | Single eye, right channel |
| MT4 4-plane full stack | 42.05% | Quantization resolution + topo + Bayesian |
| **3-eye stereoscopic** | **41.18%** | **Multi-perspective fusion, Bayesian only** |

The stereoscopic approach reaches 41.18% using ONLY Bayesian posterior
fusion — no inverted index, no cascade, no dot products, no
topological features. This is a fundamentally simpler architecture
than the MT4 full stack, achieving comparable accuracy through a
different principle (multiple perspectives vs single deep pipeline).

### Combining stereo + MT4 stack

The two approaches are complementary:
- Stereo captures multi-perspective photometric information
- MT4 stack captures quantization-depth + topological + ranking information

Combining them is a natural next step.

---

## The Stereoscopic Principle

**Multi-perspective quantization is a new architectural primitive.**

Not multi-scale (same quantization, different resolutions).
Not multi-channel (same quantization, different data).
Not multi-trit (same thresholds, different precision).

Multi-perspective: **different quantization functions on the same data,
each revealing different structure, combined through posterior fusion.**

This principle:
- Requires zero learned parameters
- Uses the existing hot map machinery
- Auto-adapts through per-eye IG weights
- Provides diminishing returns past 3 complementary perspectives
- Applies to any dataset where photometric variation confounds classification

---

## Files

Experiments:
- `src/sstt_cifar10_correlate.c` — Feature-accuracy correlation analysis
- `src/sstt_cifar10_adaptive.c` — Adaptive quantization (3 variants)
- `src/sstt_cifar10_dual.c` — Dual quantization (fixed + adaptive)
- `src/sstt_cifar10_stereo.c` — Full 5-eye stereoscopic experiment

This document: `docs/39-stereoscopic-quantization.md`
