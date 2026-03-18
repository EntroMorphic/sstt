# SSTT Documentation Index

Active contributions grouped by theme. Original contribution numbers
are preserved for cross-reference with the CHANGELOG.

Archived contributions are in [`archived/docs/`](../archived/docs/) —
see [`archived/README.md`](../archived/README.md) for context.

---

## Core Architecture

| # | Title | Key Result |
|---|-------|------------|
| 01 | [sign_epi8 ternary multiply](01-sign-epi8-ternary-multiply.md) | 32-wide ternary ALU in one AVX2 instruction |
| 02 | [Hot map: naive Bayes via ternary block frequencies](02-hot-map-naive-bayes.md) | 73.23% at ~1 us, 451 KB L2-resident model |
| 05 | [Background value discrimination](05-background-value-discrimination.md) | Per-channel BG constants; gradients 20% → 60-74% |
| 06 | [Ternary gradient observers](06-ternary-gradient-observers.md) | 3-channel system: pixel + h-grad + v-grad |
| 07 | [IG-weighted block voting](07-ig-weighted-block-voting.md) | +15.55pp in cascade context, closed-form from data |
| 08 | [Multi-probe Hamming-1 expansion](08-multi-probe-hamming1.md) | +2.38pp from 6 trit-flip neighbors at half weight |
| 11 | [Zero-weight cascade](11-zero-weight-cascade.md) | 96.04% — the full pipeline, zero learned parameters |
| 22 | [Bytepacked cascade](22-bytepacked-cascade.md) | 96.28%, 3.5x faster than 3-channel; K-invariant |

## Empirical Analysis

| # | Title | Key Result |
|---|-------|------------|
| 10 | [Haar vs WHT trap](10-haar-vs-wht-trap.md) | Haar is not orthogonal; WHT preserves dot products exactly |
| 13 | [WHT brute kNN Pareto ceiling](13-wht-brute-knn-pareto.md) | 96.12% — the hard ceiling for pixel-space kNN |
| 15 | [Fashion-MNIST generalization](15-fashion-mnist-generalization.md) | 82.89% with no tuning; IG auto-redistributes across channels |
| 23 | [Cascade autopsy + multi-channel dot](23-cascade-autopsy-multidot.md) | 98.3% of errors at dot step; 87% of compute is scattered writes |
| 24 | [Benchmark catalogue (SWO)](24-benchmark-catalogue.md) | All 13 classifiers profiled with strengths/weaknesses/opportunities |
| 25 | [Oracle: multi-specialist routing](25-oracle-multi-specialist.md) | 96.44% via unanimity gating; 92% fast path at 98.8% |

## Optimization

| # | Title | Key Result |
|---|-------|------------|
| 09 | [WHT as ternary-native transform](09-wht-ternary-native-transform.md) | Add/subtract only; lossless isometry of dot product space |
| 14 | [Fused kernel architecture](14-fused-kernel-architecture.md) | Raw pixels → class in one pass; <1KB code, 232B stack, 1.3MB model |
| 16 | [Spectral coarse-to-fine](16-spectral-coarse-to-fine.md) | Vote-based filtering beats spectral filtering for ternary data |
| 19 | [Composite solutions analysis](19-composite-moneyballs.md) | IG pre-bake +0.54pp (not +15pp); hot map collapses per-image identity |
| 28 | [Confidence-gated hybrid architecture](28-confidence-gated-hybrid-architecture.md) | Hot map / cascade as a spectrum; gating design and plan |
| 29 | [Topological ranking](29-topological-ranking.md) | 97.11% — divergence + centroid + profile break the block-encoding ceiling |

## Negative Results

| # | Title | Key Result |
|---|-------|------------|
| 17 | [Ternary PCA](17-ternary-pca-experiment.md) | Failed: -16 to -23pp; PCA destroys spatial addressing |
| 21 | [Soft-prior cascade](21-soft-prior-cascade.md) | Failed: weaker classifier cannot guide stronger one |

## Topological Ranking Series

| # | Title | Key Result |
|---|-------|------------|
| 29 | [Topological ranking](29-topological-ranking.md) | 97.11% — divergence + centroid + profile break the block-encoding ceiling |
| 30 | [Eigenplane generalization](30-eigenplane-generalization.md) | Information vs invariance tension; histogram/curl/dot tested |
| 31 | [Sequential field-theoretic ranking](31-sequential-field-ranking.md) | **97.27% MNIST, 85.81% Fashion** — the emerging algorithm |

## Experiment Files

All experiment source files in `src/`. Each is self-contained.

| File | Experiment | Key finding |
|------|-----------|-------------|
| sstt_topo.c | Base topological features | 97.11%: divergence + centroid + profile |
| sstt_topo2.c | Generalization: div dot, histogram, curl | Only div dot helps Fashion |
| sstt_topo3.c | Filter-then-rank | Hard filtering loses to soft weighting |
| sstt_topo4.c | Kalman-adaptive weighting | 97.16%: per-image MAD-based gain |
| sstt_topo5.c | Divergence refinement | Raw wins over edge-masked / weighted |
| sstt_topo6.c | Dead zone + bucketing | Kalman already handles noise |
| sstt_topo7.c | Chirality + quadrant divergence | 97.24%: spatial decomposition helps |
| sstt_topo8.c | Grid resolution sweep | 97.27%: 2x4 MNIST, 3x3 Fashion |
| sstt_topo9.c | Bayesian-CfC sequential | 97.27% MNIST, 85.81% Fashion |
| sstt_error_profile.c | Feature-space error autopsy | Top-10 agree: 100-200x predictor |
| sstt_gauss_map.c | Discrete Gauss map | 45+64 bin joint histogram |
| sstt_gauss_delta.c | Ground-state delta encoding | +19 MNIST, +20 Fashion; "maps all the way down" not validated |
| sstt_confidence_map.c | Quantized confidence map | 47% MI, 97.2% on Fashion confident 50% |
| sstt_vote_route.c | Vote-phase free routing | 99.6% on easy 22%, zero ranking cost |
| sstt_topo9_val.c | Proper val/holdout validation | Val-derived weights, holdout-only reporting |
| sstt_cifar10_curvature.c | Second-order curvature | Distinguished rounded vs angular; 51% test slice |
| sstt_cifar10_lagrangian.c | Discrete Curl operator | Identifies rotational energy; 26% alone, 39% combo |
| sstt_cifar10_grid_tracer.c | Structural Particle Tracing | 4x4 grid Lagrangian skeleton; 18% with 64 floats |
| sstt_dual_hot_map.c | Dual Hot Map (Pixel+Gauss) | 76.5% MNIST instantly; O(1) table lookup fusion |
| sstt_router_v1.c | Three-Tier Router Prototype | Adaptive compute: 99.9% accuracy on 15% instantly |
| sstt_benchmark_cifar10.c | Lagrangian Benchmark | Head-to-head Gauss vs Tracer vs Combo |

## Next Phase-Changes

| # | Title | Key Result |
|---|-------|------------|
| 33 | [Three phase-changes](33-next-phase-changes.md) | Ship router, publish paper, ground-state delta map |

## Validation

| # | Title | Key Result |
|---|-------|------------|
| 32 | [Red-team validation](32-red-team-validation.md) | Val/test split, 1-NN control, brute kNN baseline, timing, pre-vote MI |
| 35 | [Val/holdout validation](35-val-holdout-validation.md) | Proper split: MNIST 97.27%, Fashion 85.68% (publication-ready) |
| 36 | [Delta map experiment](36-delta-map-experiment.md) | +19/+20 errors; partial negative — "maps all the way down" not validated |
| — | Fashion-MNIST stereoscopic (`sstt_fashion_stereo.c`) | **86.12%** — stereo principle validated on Fashion (+0.44pp) |
| 37 | [CIFAR-10 boundary test](37-cifar10-boundary-test.md) | 33.76% Bayesian (3.4x random); cascade fails — honest boundary |

## CIFAR-10 Boundary Test

| # | Title | Key Result |
|---|-------|------------|
| 37 | [CIFAR-10 boundary test](37-cifar10-boundary-test.md) | 33.76% Bayesian (3.4x random); cascade fails on grayscale |
| 38 | [CIFAR-10 full arc](38-cifar10-full-arc.md) | 42.05% via MT4 4-plane vote + full stack |
| 39 | [Stereoscopic quantization](39-stereoscopic-quantization.md) | Multi-perspective fusion: 3 eyes → 41.18% Bayesian; new primitive |
| 40 | [Hierarchical decomposition](40-hierarchical-decomposition.md) | Machine/animal binary: 81%; oracle ceiling 48.86% |
| 41 | [Stereo + MT4 stack](41-stereo-stack-cifar10.md) | All 3 power sources: 44.48%; correctly identified 3 mechanisms |
| 43 | [Cascade Gauss — 50%](43-cascade-gauss-50pct.md) | **50.18%** stereo vote → RGB Gauss rank; pipeline reconnected |
| 44 | [Architectural Consolidation](44-step-change-architectural-consolidation.md) | Four step-changes: Tiered Router, Curvature, Delta Map, Scaling |
| 45 | [Second-Order Curvature](45-cifar10-second-order-curvature.md) | CIFAR-10 curvature and symmetry; ~51% on test slice |
| 46 | [Lagrangian Vorticity (Curl)](46-discrete-lagrangian-vorticity.md) | Discrete Curl operator for topological junction detection |
| 47 | [Structural Particle Tracing](47-structural-particle-tracing.md) | Lagrangian path-tracing for structural skeletons |
| 48 | [Dual Hot Map — Fast Path](48-dual-hot-map-fast-path.md) | Fusing independent lookup tables for 76.5% MNIST accuracy |
| 49 | [Three-Tier Router Design](49-tiered-router-design.md) | Architectural specification for adaptive compute per-image |
| 50 | [Tiered Router Validation](50-tiered-router-validation.md) | Prototype achieves 99.9% accuracy on 15% of images instantly |
| 51 | [Production Tiered Inference](51-production-ready-tiered-inference.md) | Consolidated 96.5% engine with adaptive compute tiers |
| 52 | [Scaling Framework 224x224](52-scaling-framework-224x224.md) | Sparse training + interpolation for 16x model compression |
| 53 | [Lagrangian Benchmark Autopsy](53-lagrangian-benchmark-autopsy.md) | Diagnosis of the fusion drop and path to topological gating |
| 54 | [Red-Team Validation Log](54-red-team-validation-log.md) | Dataset sensitivity analysis and zero-FP Tier 1 validation |
| 55 | [Variance-Guided Scaling — Negative](55-eigenvalue-scaling-negative-result.md) | Variance clusters knots and leaves geometric coverage gaps |
| 56 | [LMM: Findings Meta-Analysis](lmm/sstt_findings_synth.md) | Architectural reduction and the "integrated ternary engine" plan |
| 57 | [Taylor Jet Space Classification](57-taylor-jet-space.md) | 2nd-order surface primitives (Ridges/Saddles) for local geometry |
| 58 | [Taylor and Navier-Stokes Autopsy](58-taylor-navier-stokes-autopsy.md) | Differential vs Integral measures; failure of fluid-dynamics on trits |
| 59 | [Hierarchical Geometric Scaling](59-hierarchical-geometric-scaling.md) | Multi-scale Bayesian fusion for high-res texture robustness |

