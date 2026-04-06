# SSTT — Signature Search | Ternary Topology

[![Build](https://github.com/EntroMorphic/sstt/actions/workflows/ci.yml/badge.svg)](https://github.com/EntroMorphic/sstt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Zero-parameter ternary cascade classifier.

No gradient descent. No backpropagation. No floating-point arithmetic at inference. The entire model fits in L2 cache and classifies images in under 1 millisecond using only integer table lookups and AVX2 byte operations.

## Results

| Method | MNIST | Fashion-MNIST | Latency | Validated |
|--------|-------|---------------|---------|-----------|
| Brute k=3 NN (baseline) | 96.12% | 76.86% | — | — |
| Bytepacked cascade | 96.28% | 82.89% | 930 us | — |
| Three-tier router | 96.50% | 83.42% | 0.67 ms | — |
| Full system (topo9, K=200) | 97.27% | 85.68% | ~1 ms | val/holdout |
| **MTFP (native ternary, K=200)** | **97.53%** | **86.54%** | ~1 ms | val/holdout |
| MTFP (K=500) | 97.63% | — | ~1.3 ms | hardcoded weights |
| CIFAR-10 (boundary test) | 50.18% | — | ~3 ms | 5x random |

Confidence interval: +/-0.32pp (binomial 95% CI on 10K samples). Full results: [docs/results.md](docs/results.md).

## Key findings

1. **K-invariance in retrieval.** In the dot-product cascade, accuracy is identical from K=50 to K=1000 — the inverted index achieves near-perfect recall at 1200x compression. The structural ranker breaks K-invariance, gaining +0.64pp from K=50 to K=1000 by exploiting diverse candidates the dot product ignores.

2. **Zero retrieval failures.** MTFP's per-channel trit-flip retrieval achieves zero Mode A errors at K=500 — every correct class is retrievable from 60,000 training images. Theoretical ceiling: **100.00%**. All 237 remaining errors are ranking failures (182 ranking inversions, 55 vote dilutions). Top confusion pairs: 3↔5 (27), 4↔9 (25).

3. **Retrieval provides breadth, ranking provides depth.** Brute L2 retrieval at K=50 produces geometrically closer candidates (+0.87pp over inverted index at same K), but L2 re-ranking of inverted-index candidates *hurts* — the SAD filter removes pixel-distant correct-class candidates that the structural ranker needs for diversity. The inverted index's "noise" is the structural ranker's signal.

4. **Native ternary representation (MTFP).** Converting from binary-stored trits (1 trit per int8) to packed ternary (4 trits per byte) with per-channel indexing and trit-flip multi-probe gained +0.26pp (97.27% to 97.53%) by fixing a topology bug: binary bit-flips don't correspond to ternary perturbations. 3.6x memory reduction.

5. **`_mm256_sign_epi8` as exact ternary multiply.** One AVX2 instruction performs 32 ternary multiply-accumulates per cycle over {-1, 0, +1}.

## Quick start

```bash
make            # Build core classifiers
make mnist      # Download MNIST (~53 MB)

./build/sstt_mtfp                     # 97.53% MNIST (headline result, native ternary)
./build/sstt_topo9_val                # 97.27% MNIST (previous best)
./build/sstt_bytecascade              # 96.28% MNIST (fastest single method)
./build/sstt_router_v1                # 96.50% at 0.67ms (production router)
./build/sstt_mtfp_dsp data-fashion/    # 86.54% Fashion-MNIST (LBP + Bayesian)
./build/sstt_kinvariance              # K-invariance sweep on full topo9
./build/sstt_ann_baseline             # Retrieval method comparison
```

See [docs/reproducing.md](docs/reproducing.md) for full reproduction instructions.

## Repository structure

```
src/
  core/         Publication-ready classifiers and experiments (17 files)
  analysis/     Diagnostic tools: error decomposition, validation (7 files)
  ablation/     Topo1-topo9 incremental ablation series (11 files)
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

journal/    LMM exploration passes
tests/      Integration tests
scripts/    Python analysis utilities
meta/       Project audit
```

## Build targets

| Target | Builds |
|--------|--------|
| `make` | Core classifiers + experiments |
| `make analysis` | 7 diagnostic tools |
| `make ablation` | topo1-topo9 ablation series |
| `make cifar10-experiments` | CIFAR-10 boundary experiments |
| `make archive` | Archived experiments |
| `make experiments` | Everything (~95 binaries) |
| `make reproduce` | Core + MNIST download + run headline results |

## Architecture

Ternary quantization (fixed thresholds) produces three channels: pixel intensity, horizontal gradient, vertical gradient. Block encoding maps 3-pixel horizontal strips to ternary signatures. An inverted index with information-gain weighting retrieves candidates from 60K training images. Ranking combines multi-channel ternary dot products with topological features (gradient divergence, enclosed centroid, intensity profile) using Kalman-adaptive weighting. The retrieval stage provides class-diverse breadth; the ranking stage provides structural depth.

See [docs/architecture.md](docs/architecture.md) for the full pipeline description.

## Paper

[paper/sstt-paper-draft.md](paper/sstt-paper-draft.md)

## License

MIT
