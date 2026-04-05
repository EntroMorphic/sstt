# Negative Results

Documented failures with root cause analysis. These approaches don't work and here's why.

## 1. Ternary PCA (-16 to -23pp)

**Hypothesis:** Project images onto principal components, quantize projections to ternary, use standard hot-map voting.

**Result:** Every configuration is 12-23pp below the pixel baseline (71.35%). Adding more PCs makes accuracy worse, not better.

| PCs | Accuracy | vs Baseline |
|-----|----------|-------------|
| 27 | 55.27% | -16.08pp |
| 54 | 52.63% | -18.72pp |
| 126 | 50.21% | -21.14pp |
| 252 | 48.85% | -22.50pp |

**Root cause:** Three compounding failures:
1. PCA optimizes for total variance, not between-class discrimination. The first eigenvector captures "how much ink" not "which digit."
2. Double quantization (eigenvectors to ternary, then projections to ternary) destroys the orthogonality structure that makes PCA work.
3. PCA projections are ordered by eigenvalue rank, not spatial position. The hot map's power comes from position-dependent frequency counting. After PCA, block 0 means "projection onto PC0" — all spatial locality is lost. **The hot map is fundamentally spatial; PCA is spectral. They're incompatible.**

**Source:** [contributions/17-ternary-pca-experiment.md](contributions/17-ternary-pca-experiment.md), `src/archive/sstt_tpca.c`

## 2. Hard-filtered soft-prior cascade (21-50%)

**Hypothesis:** Use the fast Bayesian hot-map classifier to narrow the cascade's search to its top-N predicted classes.

**Result:** Accuracy drops to 21-50% depending on N.

**Root cause:** The Bayesian classifier is wrong 15% of the time. Hard filtering to its top-N classes permanently removes the correct class on those images. The cascade cannot recover — the information is gone. A weaker classifier cannot productively hard-gate a stronger one.

**What works instead:** Soft priors that boost without excluding: `votes[tid] += w * boost[label[tid]]`, where boost >= 1 always. This lets the cascade override wrong priors via raw vote volume.

**Source:** [contributions/21-soft-prior-cascade.md](contributions/21-soft-prior-cascade.md), `src/archive/sstt_softcascade.c`

## 3. Taylor jet space on CIFAR-10 (54.45% binary cat/dog)

**Hypothesis:** Classify pixels by local curvature using the discrete Hessian matrix. Categorize surface topology into 8 primitives (Ridge, Saddle, Peak, etc.) on a spatial grid.

**Result:** Works on MNIST (91.6% with extreme compression — 512 integers). Fails on CIFAR-10 binary cat/dog (54.45%, near random).

**Root cause:** Ternary quantization destroys the subtle curvature signals in low-resolution natural images. Second-order surface features require continuous gradients; three levels cannot preserve the Hessian eigenvalue structure.

**Source:** [contributions/58-taylor-navier-stokes-autopsy.md](contributions/58-taylor-navier-stokes-autopsy.md), `src/archive/sstt_taylor_jet.c`

## 4. Navier-Stokes stream function (~10%)

**Hypothesis:** Treat the ternary gradient field as a velocity field. Solve the Poisson equation to find the stream function. Use the resulting topological skeleton for classification.

**Result:** Accuracy at random baseline (~10%).

**Root cause:** The global potential field is dominated by image frame boundaries rather than object structure. Boundary effects overwhelm discriminative interior signal. The Poisson solve propagates boundary influence everywhere, washing out local topology.

**Source:** [contributions/58-taylor-navier-stokes-autopsy.md](contributions/58-taylor-navier-stokes-autopsy.md), `src/archive/sstt_navier_stokes.c`

## 5. Sequential processing on MNIST (+0.00pp)

**Hypothesis:** Process top-K candidates sequentially with confidence decay — later candidates contribute less evidence. Bayesian and CfC (continuous-time) variants tested.

**Result:** Zero benefit on MNIST when parameters are searched on a proper validation split. The +0.03pp reported in the original topo9 was noise from test-set search.

**Note:** Sequential processing genuinely helps on Fashion-MNIST (+1.14pp on holdout), where the candidate pool is more heterogeneous.

**Source:** [contributions/35-val-holdout-validation.md](contributions/35-val-holdout-validation.md)

## 6. 224x224 scaling (11.6% MNIST, 46.8% Fashion)

**Hypothesis:** Upscale 28x28 images to 224x224 and apply the cascade with a larger block grid.

**Result:** MNIST mode collapses (11.6% vs 92.67% brute baseline at same resolution). Fashion-MNIST underfits (46.77% vs 78.17% baseline).

**Root cause:** Nearest-neighbor upsampling adds zero new information. The 1008-position block space (vs 252 at 28x28) is 4x sparser per training image, so IG weights are computed from noisier statistics. Background blocks dominate the expanded grid. Upsampled images need real high-resolution data, not interpolation artifacts.

**Source:** `src/archive/sstt_scale_224.c`, `src/archive/sstt_scale_hierarchical.c`

## 7. L2 re-ranking of inverted-index candidates (-0.03 to -0.05pp)

**Hypothesis:** Re-rank inverted-index candidates (K=500) by L2 pixel distance (SAD), select the top 50-200 geometrically closest, then apply topo9 structural ranking. Should combine the speed of the inverted index with the candidate quality of brute L2.

**Result:** Accuracy dropped from 97.48% (K=500 inverted-index-only) to 97.43-97.45% (hybrid). L2 filtering made things worse.

**Root cause:** Diagnostic revealed the mechanism. Image 1465 (digit 4): 49 correct-class candidates in the K=500 pool, only 2 survived the SAD filter. The correct-class candidates removed by the filter were pixel-distant but topologically similar — different handwriting styles of the same digit with the same divergence/centroid/profile patterns. The structural ranker needs diverse correct-class examples to build consensus. The SAD filter destroys this diversity by selecting for pixel uniformity.

**Key insight:** The inverted index's "noise" is diversity. Retrieval provides breadth (class-diverse candidates); ranking provides depth (structural discrimination within that set). Filtering for geometric proximity undermines the breadth that makes the ranker work.

**Source:** `src/core/sstt_hybrid_retrieval.c`, `src/core/sstt_hybrid_diagnose.c`

## Patterns

Two recurring themes:

1. **Operators that depend on continuous or high-resolution structure fail in ternary space.** The winning features are integral measures (divergence sums, centroid positions, profile counts) rather than differential measures (Hessian eigenvalues, curl, stream functions). Summing energy over a region is robust to three-level quantization; computing how that energy changes is not.

2. **Filtering for similarity destroys the diversity the ranker needs.** The structural ranker's power comes from seeing the same class written many different ways. Anything that selects for uniformity — L2 filtering, hard class gating, pixel-space pruning — reduces accuracy by removing the diverse exemplars that drive structural consensus.
