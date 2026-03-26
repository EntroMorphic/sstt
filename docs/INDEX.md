# SSTT Documentation Index

25 active contributions telling the complete story from first primitive
to validated results. Original contribution numbers preserved for
cross-reference with CHANGELOG and git history.

35 archived contributions (superseded, intermediate, or exploratory)
are in [`archived/`](archived/) with full provenance.

---

## Core Architecture

The pipeline from raw pixels to classification, built incrementally.

| # | Title | Key Result |
|---|-------|------------|
| 01 | [sign_epi8 ternary multiply](01-sign-epi8-ternary-multiply.md) | 32-wide ternary ALU in one AVX2 instruction |
| 02 | [Hot map: naive Bayes](02-hot-map-naive-bayes.md) | 73.23% at ~1 us, 451 KB L2-resident model |
| 05 | [Background value discrimination](05-background-value-discrimination.md) | Per-channel BG constants; gradients 20% -> 60-74% |
| 06 | [Ternary gradient observers](06-ternary-gradient-observers.md) | 3-channel system: pixel + h-grad + v-grad |
| 07 | [IG-weighted block voting](07-ig-weighted-block-voting.md) | +15.55pp in cascade context, closed-form from data |
| 08 | [Multi-probe Hamming-1 expansion](08-multi-probe-hamming1.md) | +2.38pp from 6 trit-flip neighbors at half weight |
| 11 | [Zero-weight cascade](11-zero-weight-cascade.md) | 96.04% — full pipeline, zero learned parameters |
| 22 | [Bytepacked cascade](22-bytepacked-cascade.md) | 96.28%, 3.5x faster (Pareto); K-invariance proven |

## Topological Ranking

Breaking the 96% ceiling with discrete field operators.

| # | Title | Key Result |
|---|-------|------------|
| 29 | [Topological ranking](29-topological-ranking.md) | 97.11% — divergence + centroid + profile |
| 31 | [Sequential field ranking](31-sequential-field-ranking.md) | **97.27% MNIST, 85.81% Fashion** — Kalman + Bayesian CfC |

## Empirical Analysis

Understanding why the system works and where it breaks.

| # | Title | Key Result |
|---|-------|------------|
| 10 | [Haar vs WHT trap](10-haar-vs-wht-trap.md) | Haar is not orthogonal; WHT preserves dot products exactly |
| 13 | [WHT brute kNN Pareto ceiling](13-wht-brute-knn-pareto.md) | 96.12% — the hard ceiling for pixel-space kNN |
| 15 | [Fashion-MNIST generalization](15-fashion-mnist-generalization.md) | 82.89% zero-tuning transfer; IG auto-redistributes |
| 23 | [Cascade autopsy + multi-channel dot](23-cascade-autopsy-multidot.md) | Mode A/B/C decomposition; 98.3% of errors at ranking; 87% compute is scattered writes |
| 24 | [Benchmark catalogue (SWO)](24-benchmark-catalogue.md) | All 13 classifiers profiled: strengths, weaknesses, opportunities |
| 25 | [Oracle: multi-specialist routing](25-oracle-multi-specialist.md) | 96.44% via unanimity gating; 92% fast path at 98.8% |

## Negative Results

Documented failures with root-cause analysis.

| # | Title | Root Cause |
|---|-------|------------|
| 17 | [Ternary PCA](17-ternary-pca-experiment.md) | -16 to -23pp; PCA destroys spatial addressing |
| 21 | [Soft-prior cascade](21-soft-prior-cascade.md) | Weaker classifier cannot guide stronger one |
| 58 | [Taylor and Navier-Stokes autopsy](58-taylor-navier-stokes-autopsy.md) | Differential operators fail on quantized fields; integral operators are robust |

## Production Router

Adaptive compute per query using free confidence signal.

| # | Title | Key Result |
|---|-------|------------|
| 51 | [Production tiered inference](51-production-ready-tiered-inference.md) | 96.50% at 0.67ms; 99.9% on 15% of images instantly |

## Validation

Rigorous checking of headline claims.

| # | Title | Key Result |
|---|-------|------------|
| 35 | [Val/holdout validation](35-val-holdout-validation.md) | **97.27% MNIST, 85.68% Fashion** — publication-ready, no leakage |
| 54 | [Red-team validation log](54-red-team-validation-log.md) | Dataset sensitivity; zero-FP Tier 1; Fashion needs tuning |

## CIFAR-10 Boundary Test

Testing where the architecture breaks on natural images.

| # | Title | Key Result |
|---|-------|------------|
| 37 | [CIFAR-10 boundary test](37-cifar10-boundary-test.md) | 33.76% Bayesian (3.4x random); cascade fails on grayscale |
| 43 | [Cascade Gauss — 50%](43-cascade-gauss-50pct.md) | **50.18%** stereo vote -> RGB Gauss rank; 5x random |

## RF-Inspired Ensemble Methods

| # | Title | Key Result |
|---|-------|------------|
| 64 | [RF-inspired ensemble methods](64-rf-inspired-ensemble-methods.md) | Bagging +0.38pp on bytecascade, +0.10pp on topo9 (largely redundant) |

## Audit

| # | Title | Scope |
|---|-------|-------|
| 60 | [Comprehensive audit (2026)](60-audit-2026-comprehensive.md) | All 59 original contributions; novel/useful/fluff/understated |

---

## Key Source Files

Not all 87 source files matter equally. These are the ones that count:

**Core classifiers:**

| File | Role | Result |
|------|------|--------|
| `src/sstt_bytecascade.c` | Best single method | 96.28% MNIST |
| `src/sstt_topo9.c` | Research best | 97.27% MNIST, 85.81% Fashion |
| `src/sstt_topo9_val.c` | Validated version | 97.27% MNIST, 85.68% Fashion |
| `src/sstt_router_v1.c` | Production router | 96.50% at 0.67ms |

**Supporting implementations:**

| File | Role |
|------|------|
| `src/sstt_v2.c` | Foundation (3-channel cascade) |
| `src/sstt_bytepacked.c` | Encoding D origin (91.83% hot map) |
| `src/sstt_dual_hot_map.c` | Fast path (76.5% at ~1.5us) |
| `src/sstt_diagnose.c` | Error autopsy tooling |
| `src/sstt_kdilute.c` | kNN dilution analysis |
| `src/sstt_confidence_map.c` | Confidence oracle |
| `src/sstt_geom.c` | Geometry primitives and tests |

**CIFAR-10 (the story in 4 files):**

| File | Role | Result |
|------|------|--------|
| `src/sstt_cifar10.c` | First attempt | 26.51% |
| `src/sstt_cifar10_stereo.c` | Stereoscopic retrieval | 41.18% |
| `src/sstt_cifar10_gauss.c` | RGB Gauss map | 48.31% |
| `src/sstt_cifar10_cascade_gauss.c` | Final cascade | 50.18% |

**Topo ablation series** (`src/sstt_topo.c` through `src/sstt_topo9.c`): 10 files showing incremental contributions from 96.28% to 97.27%.

All other source files are exploratory experiments. They compile and run but are not part of the validated results.

---

## Paper

One paper draft: [`papers/sstt-paper-draft.md`](papers/sstt-paper-draft.md)

Three satellite drafts (router, Lagrangian, scaling) were archived — see
[`archived/papers/`](archived/papers/) and [`AUDIT_SSTT.md`](../AUDIT_SSTT.md) Section 8
for the rationale.
