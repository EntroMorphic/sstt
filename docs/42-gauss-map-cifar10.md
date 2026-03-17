# Contribution 42: Gauss Map on CIFAR-10 — Shape Geometry Classification

Maps each pixel's gradient vector onto a unit sphere. The distribution
of points on the sphere encodes edge geometry — position-invariant,
brightness-invariant, background-suppressing. Flat background produces
zero gradient, maps to the pole, vanishes from the histogram. Only
edges and structure contribute.

**46.06% on CIFAR-10** — new best, beating stereo+MT4 stack (44.48%)
through a completely different mechanism.

Code: `src/sstt_cifar10_gauss.c`

---

## The Insight

The user asked: "Can we convert 2D information into 3D spheres that
deform based on the information they encode, making geometry the basis
for learning instead of math?"

This is exactly the Gauss map from differential geometry:
- The image intensity z = f(x,y) defines a surface
- Each pixel's gradient (∂f/∂x, ∂f/∂y) defines a surface normal
- Mapping the normal to the unit sphere → a point on the sphere
- The distribution of points = the image's edge geometry fingerprint

For CIFAR-10 with RGB: four spheres (grayscale + R + G + B), each
deformed by that channel's edge structure. Histogrammed into 45 bins
per channel (3 directions × 3 directions × 5 magnitude buckets).

---

## Why It Suppresses Background

Flat background:
- Gradient ≈ (0, 0)
- Maps to the north pole of the sphere
- In the histogram: falls into bin (direction=center, magnitude=0)
- This bin is the same for ALL backgrounds regardless of color
- Effectively suppressed — all backgrounds look identical

Object edges:
- Gradient is non-zero, directional
- Maps to specific regions of the sphere
- Different objects deform the sphere differently
- An airplane (angular, elongated) vs a frog (round, textured) vs
  a truck (rectangular, structured) have different sphere distributions

This is intrinsic background invariance — no hard-coded constraint,
no edge detection threshold. The geometry itself separates signal
from noise.

---

## Results

### Individual methods

| Method | Accuracy | Mechanism |
|--------|----------|-----------|
| Global Gauss map kNN (L1, k=1) | 36.07% | 4-channel × 45-bin histograms, no position |
| **Grid Gauss map kNN (4×4, L1, k=1)** | **46.06%** | 16 spatial regions × 45 bins per channel |
| Combined global + grid | 39.39% | Global dilutes grid signal |
| Nearest class mean | 28.69% | Centroid classifier (too coarse) |

### Grid Gauss map vs prior best

| Method | Accuracy | Architecture |
|--------|----------|-------------|
| Grayscale ternary cascade | 26.51% | Block texture, pixel dot |
| Flattened RGB Bayesian | 36.58% | Color blocks, position-specific |
| MT4 full stack | 42.05% | 81-level + topo + Bayesian |
| Stereo + MT4 stack | 44.48% | 3-eye perspective + all features |
| **Grid Gauss map kNN** | **46.06%** | **Shape geometry, brute kNN** |

### Per-class comparison (Gauss map vs stereo stack)

| Class | Stereo stack | Gauss map | Delta |
|-------|-------------|-----------|-------|
| airplane | 52.3% | 54.0% | +1.7pp |
| automobile | 47.6% | **53.4%** | **+5.8pp** |
| **bird** | 23.6% | **38.6%** | **+15.0pp** |
| cat | 21.4% | 21.6% | +0.2pp |
| **deer** | 41.4% | **49.1%** | **+7.7pp** |
| dog | 40.2% | 30.4% | -9.8pp |
| frog | 61.6% | **67.7%** | +6.1pp |
| horse | 39.0% | **44.6%** | +5.6pp |
| ship | 54.3% | **55.9%** | +1.6pp |
| truck | 63.4% | 45.3% | -18.1pp |

8 of 10 classes improve. Bird +15pp is the largest single-class
improvement in the entire CIFAR-10 program — the sphere captures bird
edge geometry that block texture completely misses. Dog and truck drop
because their textures are more discriminative than their edge geometry.

### Binary machine/animal on Gauss map

Gauss map binary: 69.85% (vs block-based binary: 81.04%). The block
system is better for coarse binary split because it uses full color
information. But the Gauss map is better for fine-grained within-class
distinction (46% vs 44%).

### Inter-class Gauss map distances

The L1 distances between class mean sphere histograms reveal the
geometric similarity structure:

| Pair | Distance | Meaning |
|------|----------|---------|
| cat↔dog | **214** | Almost identical edge geometry |
| auto↔truck | **293** | Similar rectangular structure |
| bird↔cat | 631 | Moderate — both organic but different |
| airplane↔ship | 779 | Different — angular vs rectangular |
| airplane↔frog | 2145 | Very different — geometric vs organic |

Cat and dog are 214 apart in sphere space — confirming that these
classes genuinely have similar edge geometry at 32×32 resolution.
No algorithm can separate them without higher-resolution features.

---

## Architecture

```
32×32 image (per channel: gray, R, G, B)
    │
    ├── Per-pixel gradient: (gx, gy)
    │
    ├── Direction: quantize to 3×3 grid on sphere
    │
    ├── Magnitude: 5 buckets (flat → strong edge)
    │
    └── Spatial grid: 4×4 regions → per-region histogram
                │
                └── 16 regions × 45 bins × 4 channels = 2880 features
                         │
                    Brute kNN (L1 distance)
                         │
                    k=1 → class prediction
```

No inverted index. No block encoding. No Bayesian hot map.
Just sphere histograms and nearest-neighbor matching.

---

## Why Grid Matters

Global Gauss map (36.07%) loses all spatial information — the sphere
knows WHAT edges exist but not WHERE. Grid Gauss map (46.06%) retains
spatial structure: "there are strong horizontal edges in the top-left
region and weak diagonal edges in the bottom-right."

This captures object composition:
- Airplane: strong edges in center-top (fuselage), uniform in
  top (sky) and bottom (ground)
- Frog: edges everywhere in center (textured body), uniform at
  borders (background)
- Truck: strong edges in center-bottom (vehicle), uniform top (sky)

The 4×4 grid is a spatial pyramid at one level. The grid size is a
parameter: too fine → position-dependent, too coarse → position-invariant.
4×4 (8×8 pixel regions) is the right granularity for 32×32 CIFAR-10.

---

## The Two Orthogonal Systems

The SSTT project now has two classification architectures for CIFAR-10:

**System A: Block-based stereo + cascade**
- Classifies on: color texture at specific positions
- Strength: color-distinctive classes (frog=green, ship=blue)
- Weakness: texture-similar classes (cat≈dog, deer≈frog)
- Best: 44.48% (stereo + MT4 stack)

**System B: Grid Gauss map kNN**
- Classifies on: edge geometry in spatial regions
- Strength: shape-distinctive classes (bird, deer, automobile)
- Weakness: texture-distinctive classes (truck, dog)
- Best: 46.06%

These are architecturally orthogonal:
- System A uses block signatures → inverted index → Bayesian posterior
- System B uses sphere histograms → brute kNN → L1 distance
- They see different features (color texture vs edge geometry)
- They make different errors (dog: A gets 40%, B gets 30%)

Combining them should capture both perspectives.

---

## Files

Code: `src/sstt_cifar10_gauss.c`
This document: `docs/42-gauss-map-cifar10.md`
Related: `docs/39-stereoscopic-quantization.md` (stereo principle),
`docs/41-stereo-stack-cifar10.md` (stereo + MT4 stack),
`src/sstt_gauss_map.c` (original MNIST Gauss map)
