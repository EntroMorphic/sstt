# SSTT — Zero-Parameter Ternary Cascade Classifier

[![Build](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](CHANGELOG.md)

**No gradient descent. No backpropagation. No floating point at inference.
No learned parameters.**

SSTT classifies images using only integer table lookups, AVX2 byte
operations, and add/subtract arithmetic. The entire model fits in L2 cache.

It achieves **97.27% on MNIST** — exceeding brute k=3 NN (96.12%) by
+1.15pp while examining **1200x fewer candidates**. The same code,
with zero architectural changes, achieves **85.68% on Fashion-MNIST**
(+8.82pp over brute kNN).

---

## Key Findings

**K-invariance.** Top-50 candidates by vote contain every relevant
neighbor out of 60,000 training images. Accuracy is identical from
K=50 through K=1000 — a 1200x compression ratio with zero recall loss.
Retrieval is solved. All remaining error is a ranking problem.

**Error mode decomposition.** 98.3% of classification errors occur at
the ranking step, not retrieval: 64.5% ranking inversions, 33.8% vote
dilution, 1.7% retrieval failures. This establishes a fixable ceiling
of ~98.6%.

**Cross-dataset generalization.** Fashion-MNIST accuracy of 85.68% with
zero tuning — IG weights automatically redistribute across channels.
The +8.82pp lift over brute kNN (larger than the +1.15pp on MNIST)
shows the architecture provides more value on harder tasks.

**Free confidence signal.** Vote concentration predicts difficulty at
zero cost. The adaptive router classifies 10% of MNIST at 99.9% accuracy
in <1 us, and 9.6% of Fashion-MNIST at 100% accuracy (zero false positives).

---

## Results

| Method | MNIST | Fashion | Latency | Notes |
|--------|-------|---------|---------|-------|
| Brute k=3 NN | 96.12% | 76.86% | — | Pixel-space ceiling |
| Bytepacked Bayesian | 91.83% | — | 1.9 us | O(1) lookup baseline |
| Bytepacked Cascade | 96.28% | 82.89% | 930 us | Best single method |
| **Topological Ranking** | **97.27%** | **85.68%** | ~1 ms | **Val/holdout confirmed** |
| Three-Tier Router | 96.50% | 83.42% | 0.67 ms | Production (2x throughput) |
| CIFAR-10 Cascade | — | — | ~3 ms | 50.18% (5x random; boundary test) |

CIFAR-10 at 50.18% is an honest boundary test, not a success claim. At 32x32
resolution, visually similar classes (cat/dog) cannot be distinguished without
learned parameters.

---

## Architecture

```
Raw image (28x28 uint8)
    |
    v  Ternary quantization: pixel < 85 -> -1, 85-169 -> 0, >= 170 -> +1
    |
    +-- Pixel channel      (3x1 blocks, 252 positions)
    +-- H-grad channel     (horizontal finite difference)
    +-- V-grad channel     (vertical finite difference)
    +-- Bytepacked channel (all 3 fused into 1 byte, 256 values)
              |
              v  IG-weighted inverted index + Hamming-1 multi-probe
         Vote accumulation -> top-50 candidates (K-invariant)
              |
              v  Multi-channel ternary dot product (via _mm256_sign_epi8)
         score = 256 * dot(px) + 192 * dot(vg)
              |
              v  Topological augmentation (divergence, centroid, profile)
         Kalman adaptive weighting + Bayesian sequential accumulation
              |
              v  k=3 majority vote
         Class prediction
```

Every parameter is derived from closed-form training statistics (IG weights,
BG thresholds) or exhaustive discrete grid search (125 combinations). No
parameter is updated by a loss function.

---

## Quick Start

**Requirements:** GCC with AVX2 support, GNU Make, curl.

```bash
git clone https://github.com/anjaustin/s2t2.git
cd s2t2

# Download data
make mnist
make fashion

# Build core classifiers
make

# Run validated best on MNIST
./build/sstt_topo9_val

# Run bytepacked cascade on Fashion-MNIST
./build/sstt_bytecascade data-fashion/
```

## Key Source Files

| File | Role | Result |
|------|------|--------|
| `src/sstt_topo9_val.c` | Validated publication version | 97.27% MNIST, 85.68% Fashion |
| `src/sstt_bytecascade.c` | Best single method | 96.28% MNIST |
| `src/sstt_router_v1.c` | Production router | 96.50% at 0.67ms |
| `src/sstt_topo9.c` | Research best (topo series) | 97.27% MNIST |
| `src/sstt_diagnose.c` | Error autopsy tooling | Mode A/B/C decomposition |
| `src/sstt_cifar10_cascade_gauss.c` | CIFAR-10 boundary | 50.18% |

All 87 source files are self-contained experiments. Build any with
`make build/sstt_<name>`. See [`docs/INDEX.md`](docs/INDEX.md) for
which files matter and why.

## Documentation

25 active contributions in [`docs/`](docs/INDEX.md) covering the full
story from first primitive to validated results. 35 archived contributions
preserved in [`docs/archived/`](docs/archived/) for provenance.

One paper draft: [`docs/papers/sstt-paper-draft.md`](docs/papers/sstt-paper-draft.md)

Full audit: [`AUDIT_SSTT.md`](AUDIT_SSTT.md)

## License

MIT — see [LICENSE](LICENSE).
