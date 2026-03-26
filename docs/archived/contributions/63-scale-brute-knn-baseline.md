# Contribution 63: Brute kNN Baseline at 224×224 Scale

Establishes the hard accuracy ceiling for the 224×224 feature representation.
Makes every retention percentage in C52, C54, C59, and C62 interpretable.

**Binary:** `build/sstt_scale_brute [data-dir/]`
**Train:** 60,000 / **Test:** 10,000
**Features:** 56×56 pooled (4×4 average pool of the 56×56 bilinear upscale),
1,008 block positions, 27 ternary values per position — the same feature space
used throughout the 224×224 scaling experiments.

---

## Results

### MNIST-224

| Method | Accuracy |
|---|---|
| Brute 1-NN | 92.50% |
| Brute k=3 NN | **92.67%** |
| Full hot map (C62) | 11.60% |
| Retention (hot / brute k3) | **12.5%** |

### Fashion-224

| Method | Accuracy |
|---|---|
| Brute 1-NN | 77.66% |
| Brute k=3 NN | **78.17%** |
| Full hot map (C62) | 46.77% |
| Hierarchical (C62) | 45.37% |
| Retention (hot / brute k3) | **59.8%** |
| Retention (hierarchical / brute k3) | **58.1%** |

---

## Per-Class Analysis

### MNIST-224: Brute k=3 vs Hot Map

| Class | Digit | Brute k=3 | Hot Map |
|---|---|---|---|
| 0 | 0 | 98.7% | 2.2% |
| 1 | 1 | 99.6% | **100.0%** |
| 2 | 2 | 89.1% | 0.0% |
| 3 | 3 | 95.1% | 0.0% |
| 4 | 4 | 91.8% | 0.0% |
| 5 | 5 | 93.0% | 0.0% |
| 6 | 6 | 97.4% | 0.0% |
| 7 | 7 | 93.9% | 0.3% |
| 8 | 8 | 77.6% | 0.0% |
| 9 | 9 | 89.6% | 0.0% |

The MNIST hot map at 224×224 classifies almost everything as class 1 (digit "1"). This
is mode collapse: at 56×56 pooled resolution, most digit blocks are background (signature
13, skipped) or produce similar foreground signatures. Only the digit "1" has a signature
distribution distinctive enough for the hot map to discriminate — narrow vertical strokes
produce a unique per-position signature that the hot map learns reliably. All other digit
classes are confounded and their posterior collapses to class 1.

### Fashion-224: Brute k=3 vs Hot Map

| Class | Label | Brute k=3 | Hot Map |
|---|---|---|---|
| 0 | T-shirt/top | 76.0% | 27.6% |
| 1 | Trouser | 95.2% | 96.8% |
| 2 | Pullover | 72.4% | 3.1% |
| 3 | Dress | 79.6% | 48.8% |
| 4 | Coat | 72.0% | 68.7% |
| 5 | Sandal | 72.6% | 37.4% |
| 6 | Shirt | **52.3%** | 0.0% |
| 7 | Sneaker | 87.7% | **98.0%** |
| 8 | Bag | 82.1% | 17.2% |
| 9 | Ankle boot | 91.8% | 70.1% |

Fashion fares better because clothing silhouettes have more consistent positional
structure than digit strokes — trousers and sneakers in particular have stable global
shapes that survive pooling. Shirt (class 6) is the hardest class even for brute kNN
(52.3%), consistent with the CIFAR-10 finding that shirts are difficult without
fine texture detail.

---

## What This Changes

### The retention picture is completely different for MNIST vs Fashion

**MNIST-224:** The hot map retains 12.5% of brute kNN accuracy. The C52 compression
story ("98.5% retention") compresses a classifier that is already only 12.5% as good
as achievable. The compression is real; the baseline it is preserving is not meaningful.

**Fashion-224:** The hot map retains 59.8% of brute kNN accuracy. The 224×224 feature
space is genuinely usable for Fashion-MNIST (brute kNN reaches 78.17%, nearly matching
the 28×28 brute kNN of 76.86%). The hot map is a poor ranker at this scale (59.8%
retention), but the information is present in the features.

### The feature spaces are not equivalent across datasets at this scale

| Dataset | Brute kNN 28×28 | Brute kNN 56×56 pooled | Delta |
|---|---|---|---|
| MNIST | 96.12% (WHT, k=3) | 92.67% | −3.45pp |
| Fashion-MNIST | 76.86% (1-NN) | 78.17% | +1.31pp |

MNIST loses 3.45pp by pooling to 56×56 — fine stroke structure (which distinguishes
4 from 9, 3 from 8) is lost in pooling. Fashion-MNIST gains 1.31pp — pooling smooths
out texture noise that confounds brute kNN at 28×28, and global silhouette (which
distinguishes trouser from t-shirt) is preserved.

### The priority for future 224×224 work

The cascade (inverted-index + ranking) at 28×28 achieves 97.27% vs. the hot map's
73.23% — a +24.04pp lift. If the same lift applied at 56×56 pooled features:

- MNIST-224: 11.60% + 24pp → ~35%? Or something closer to the 92.67% brute ceiling?
- Fashion-224: 46.77% + cascade lift → potentially 65–75%?

The cascade at 224×224 is the natural next experiment (C64). The hot map experiments
(C52–C59, C62) establish the floor; the cascade would establish whether the architecture
can approach the 92.67% / 78.17% brute ceilings.

---

## Files

- `src/sstt_scale_brute.c` (new)
- Supersedes: the uncontextualized retention figures in C52, C54, C59, C62
