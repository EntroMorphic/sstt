# SSTT — Ternary Cascade Classifier

[![Build](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](CHANGELOG.md)

A pure C, dependency-free image classifier built entirely from ternary arithmetic and table lookups. No floating-point during inference. No external ML libraries. Achieves **96.28% on MNIST** and **82.89% on Fashion-MNIST** using a cascaded inverted-index architecture with AVX2-accelerated ternary dot products.

## Results

| Method | MNIST | Fashion | Speed | Model size |
|--------|-------|---------|-------|------------|
| Bytepacked Bayesian (hot map) | 91.83% | 76.52% | 1.2 μs | 16 MB |
| Bytepacked cascade (4-probe) | **96.20%** | — | 6.1s / 10K | 22 MB |
| Bytepacked cascade (8-probe) | 96.28% | 82.89% | 9.3s / 10K | 22 MB |
| 3-channel cascade (reference) | 96.12% | — | 32.9s / 10K | — |

All numbers on the standard MNIST and Fashion-MNIST test sets (10,000 images).

## Architecture

```
Raw image (28×28 uint8)
    │
    ▼  AVX2 ternary quantization
{-1, 0, +1} per pixel (85/170 thresholds)
    │
    ├──── Pixel blocks  (3×1 horizontal, 252 positions, 27 values)
    ├──── H-grad blocks (same geometry, derivative channel)
    ├──── V-grad blocks (same geometry, derivative channel)
    └──── Bytepacked   (all 3 channels → 1 byte, 256 values)
              │
              ▼  Information-gain weighted multi-probe voting
         Inverted index lookup
         IG-weighted votes → top-K candidates
              │
              ▼  Multi-channel ternary dot refinement
         score = 256 × dot(px) + 192 × dot(vg)
              │
              ▼  k=3 majority vote
         Class prediction
```

**Core insight:** quantize → encode blocks → vote over training images by block match → refine top-K with dot product. No matrix multiply. No backpropagation. All computation is integer table lookup and AVX2 byte operations.

## Quick Start

**Requirements:** GCC with AVX2 support (Haswell / AMD Zen 2+), GNU Make, curl.

```bash
git clone https://github.com/anjaustin/s2t2.git
cd s2t2

# Download MNIST (11 MB) and Fashion-MNIST (31 MB)
make mnist
make fashion     # optional

# Build the main classifiers
make

# Run bytepacked cascade on MNIST (~9 sec)
./sstt_bytecascade

# Run on Fashion-MNIST
./sstt_bytecascade data-fashion/

# Run the optimized multi-channel dot version
./sstt_multidot

# Run the error autopsy tool
./sstt_diagnose
```

## Building

```bash
# Default build (4 key binaries)
make

# Build all experiments
make experiments

# Build a specific experiment
make sstt_v2

# Clean compiled binaries
make clean

# Remove everything including downloaded data
make cleanall
```

**Compiler flags:** `-O3 -mavx2 -mfma -march=native`. Requires a CPU with AVX2 (Intel Haswell 2013+ or AMD Zen 2019+).

## Experiments

Each experiment is a self-contained `.c` file. Build individually with `make sstt_<name>`.

| Binary | Accuracy | Description |
|--------|----------|-------------|
| `sstt_v2` | 96.12% | 3-channel IG cascade (reference implementation) |
| `sstt_bytecascade` | 96.28% | Bytepacked cascade — best single method |
| `sstt_multidot` | 96.20% | Multi-channel dot refinement + probe sweep |
| `sstt_diagnose` | — | Error autopsy: failure mode classification + ASCII visualization |
| `sstt_bytepacked` | 91.83% | Bytepacked hot map (Bayesian series) |
| `sstt_pentary` | 86.31% | 5-level quantization |
| `sstt_transitions` | 84.69% | Inter-block transition channels |
| `sstt_eigenseries` | 83.86% | IG-ordered block traversal |
| `sstt_series` | 83.86% | Block-interleaved Bayesian |
| `sstt_softcascade` | 96.12% | Soft Bayesian prior on votes |
| `sstt_hybrid` | 87.80% | Confidence-gated hot map + cascade |
| `sstt_tiled` | 94.68% | K-means tile routing (routing hurts) |
| `sstt_geom` | 73.23% | Original 3-channel hot map |

Negative results are equally documented — each failed approach taught something concrete.

## Documentation

23 numbered contributions in [`docs/`](docs/), each capturing one experiment's findings:

| # | Topic |
|---|-------|
| 1–6 | Foundations: ternary arithmetic, hot maps, gradient observers |
| 7–13 | Architecture: IG weighting, multi-probe, WHT, cascade composition |
| 14–16 | Optimization: fused kernel, Fashion-MNIST, spectral pruning |
| 17–20 | Exploration: PCA (failed), eigenvalue splines, LMM architectural analysis |
| 21–23 | Refinement: soft priors, bytepacked encoding, error autopsy & dot optimization |

Start with [`docs/20-lmm-pass-composite-architecture.md`](docs/20-lmm-pass-composite-architecture.md) for the full architectural picture, or [`docs/23-cascade-autopsy-multidot.md`](docs/23-cascade-autopsy-multidot.md) for the latest findings.

## Key Findings

- **98.3% of errors occur at the dot product step**, not the vote step. The cascade finds the right neighborhood; ternary pixel dot cannot resolve topological differences (4↔9, 3↔5) by margins of 3–20 units.
- **Multi-channel dot is essentially free**: 3-channel vs 1-channel refinement costs +7 μs on a 636 μs pipeline (+1.1%), with +0.34pp accuracy gain.
- **87% of compute is in vote accumulation** (scattered writes to 240 KB votes array). The dot product is negligible.
- **Pairwise (Hamming-2) probes hurt**: diagonal (px±1, vg±1) neighbors overshoot the useful neighborhood. Single-bit Hamming-1 flips are already optimal.
- **Pre-allocating the top-K histogram** eliminates the per-image `calloc` cold-miss penalty (~26 μs/img → ~1 μs/img via sequential `memset`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The project follows the Lincoln Manifold Method for documentation: each experiment records its hypothesis, implementation, results, and honest failure analysis.

## License

MIT — see [LICENSE](LICENSE).
