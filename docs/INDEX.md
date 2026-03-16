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
| 02 | [Hot map: naive Bayes via ternary block frequencies](02-hot-map-cipher.md) | 73.23% at ~1 μs, 451 KB L2-resident model |
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
| 30 | [Eigenplane generalization (LMM pass)](30-eigenplane-generalization-lmm.md) | Information vs invariance tension; histogram/curl/dot tested |
| 31 | [Sequential field-theoretic ranking](31-sequential-field-ranking.md) | **97.31% MNIST, 85.81% Fashion** — the emerging algorithm |

## Meta

| # | Title |
|---|-------|
| 26 | [Audit: novel, useful, fluff, understated](26-audit-novel-useful-fluff-understated.md) |
| 27 | [Remediation plan](27-remediation-plan.md) |
