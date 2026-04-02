# SSTT

[![Build](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Zero-parameter ternary cascade classifier.

No gradient descent. No backpropagation. No floating-point arithmetic at inference. The entire model fits in L2 cache and classifies images in under 1 millisecond using only integer table lookups and AVX2 byte operations.

## Results

| Method | MNIST | Fashion-MNIST | Latency | Validated |
|--------|-------|---------------|---------|-----------|
| Brute k=3 NN (baseline) | 96.12% | 76.86% | — | — |
| Bytepacked cascade | 96.28% | 82.89% | 930 us | — |
| Three-tier router | 96.50% | 83.42% | 0.67 ms | — |
| **Full system (topo9)** | **97.27%** | **85.68%** | ~1 ms | val/holdout |
| CIFAR-10 (boundary test) | — | — | ~3 ms | 50.18% (5x random) |

Confidence interval: +/-0.32pp (binomial 95% CI on 10K samples). Full results with validation conditions: [docs/results.md](docs/results.md).

## What is novel

1. **K-invariance.** The top-50 candidates by vote contain every relevant neighbor from 60,000 training images. Accuracy is identical from K=50 to K=1000 — a 1200x compression with zero recall loss. Retrieval is solved; all remaining error is ranking.

2. **`_mm256_sign_epi8` as exact ternary multiply.** One AVX2 instruction performs 32 ternary multiply-accumulates per cycle over {-1, 0, +1}. No conditional branches, no widening.

3. **Error mode decomposition.** Every misclassification categorized: 1.7% retrieval miss, 64.5% ranking inversion, 33.8% vote dilution. 98.3% of errors occur at ranking, not retrieval. Fixable ceiling: ~98.6%.

## Quick start

```bash
make            # Build 4 core classifiers
make mnist      # Download MNIST (~53 MB)

./build/sstt_topo9_val              # 97.27% MNIST (headline result)
./build/sstt_bytecascade            # 96.28% MNIST (fastest single method)
./build/sstt_router_v1              # 96.50% at 0.67ms (production router)
./build/sstt_topo9_val data-fashion/  # 85.68% Fashion-MNIST (zero tuning)
```

See [docs/reproducing.md](docs/reproducing.md) for full reproduction instructions.

## Repository structure

```
src/
  core/         Publication-ready classifiers (9 files)
  analysis/     Diagnostic tools: error decomposition, validation (7 files)
  ablation/     Topo1-topo9 incremental ablation series (12 files)
  cifar10/      CIFAR-10 boundary experiments (37 files)
  archive/      Exploratory and superseded experiments (34 files)

docs/
  architecture.md     Pipeline description
  results.md          Complete results with validation conditions
  reproducing.md      Step-by-step reproduction guide
  negative-results.md Documented failures with root causes
  contributions/      27 numbered research contributions
  archived/           Superseded documentation

paper/
  sstt-paper-draft.md   Paper draft

tests/      Integration tests
scripts/    Python analysis utilities
meta/       Project audit
```

## Build targets

| Target | Builds |
|--------|--------|
| `make` | 4 core classifiers |
| `make analysis` | 7 diagnostic tools |
| `make ablation` | topo1-topo9 ablation series |
| `make cifar10-experiments` | CIFAR-10 boundary experiments |
| `make archive` | Archived experiments |
| `make experiments` | Everything (~90 binaries) |
| `make reproduce` | Core + MNIST download + run headline results |

## Architecture

Ternary quantization (fixed thresholds) produces three channels: pixel intensity, horizontal gradient, vertical gradient. Block encoding maps 3-pixel horizontal strips to ternary signatures. An inverted index with information-gain weighting retrieves the top-50 candidates from 60K training images. Ranking combines multi-channel ternary dot products with topological features (gradient divergence, enclosed centroid, intensity profile) using Kalman-adaptive weighting.

See [docs/architecture.md](docs/architecture.md) for the full pipeline description.

## Paper

[paper/sstt-paper-draft.md](paper/sstt-paper-draft.md)

## License

MIT
