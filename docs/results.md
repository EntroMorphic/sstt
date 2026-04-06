# Results

All results on standard test splits. Validation methodology: 5K val / 5K holdout split on the 10K test set. Grid searches run on val only; final accuracy reported on holdout. See [contributions/35-val-holdout-validation.md](contributions/35-val-holdout-validation.md) for full protocol.

Hardware: x86-64 with AVX2 (tested on Zen 3 and Rocket Lake). Compiler: GCC -O3 -mavx2 -mfma -march=native.

## MNIST (10,000-image test set)

| Method | Accuracy | Latency | Validated | Source |
|--------|----------|---------|-----------|--------|
| Brute 1-NN (pixel L2) | 95.87% | — | — | baseline |
| Brute k=3 NN (WHT) | 96.12% | — | — | baseline |
| Naive Bayes hot map (pixel) | 73.23% | ~1.5 us | — | Doc 02 |
| Bytepacked Bayesian hot map | 91.83% | 1.9 us | — | Doc 22 |
| Dual Hot Map | 76.53% | ~1.5 us | — | Doc 48 |
| Zero-weight cascade (K=50, k=3) | 96.04% | — | — | Doc 11 |
| Bytepacked cascade (Encoding D) | 96.28% | 930 us | — | Doc 22 |
| Oracle v2 (unanimity-gated) | 96.44% | 1.1 ms | — | Doc 25 |
| Three-Tier Router | 96.50% | 0.67 ms | — | Doc 51 |
| Full system (topo9, joint index) | 97.27% | ~1 ms | val/holdout | Doc 31, 35 |
| **MTFP (per-channel, trit-flip)** | **97.53%** | ~1 ms | val/holdout | sstt_mtfp.c |
| MTFP (K=500, hardcoded weights) | 97.63% | ~1.3 ms | — | sstt_mtfp_diagnose.c |

Confidence interval: +/-0.32pp (binomial, 95% CI on 10K samples).

### Val/holdout breakdown

**MTFP (current best):**

| Split | Accuracy | Errors |
|-------|----------|--------|
| Val (images 0-4999) | 96.58% | 171 |
| Holdout (images 5000-9999) | 98.48% | 76 |
| Full 10K | 97.53% | 247 |

**topo9 (previous, joint bytepacked index):**

| Split | Accuracy | Errors |
|-------|----------|--------|
| Val (images 0-4999) | 96.30% | 185 |
| Holdout (images 5000-9999) | 98.24% | 88 |
| Full 10K | 97.27% | 273 |

MTFP gains +0.26pp (26 fewer errors) from the trit-flip multi-probe fix and per-channel indexing. Same grid search finds the same best weights — the improvement is from retrieval topology, not feature weighting.

## K-invariance and K-sensitivity

**Bytepacked cascade (dot-product ranking only):** K-invariant.

| K | Accuracy |
|---|----------|
| 50 | 96.28% |
| 100 | 96.28% |
| 200 | 96.28% |
| 500 | 96.28% |
| 1000 | 96.28% |

**MTFP system (structural ranking, per-channel trit-flip):** K-sensitive.

| K | MTFP Accuracy | Mode A | Mode B | Mode C |
|---|--------------|--------|--------|--------|
| 50 | 96.99% | 17 | 206 | 78 |
| 100 | 97.25% | 8 | 195 | 72 |
| 200 | 97.54% | 4 | 186 | 56 |
| 500 | 97.63% | **0** | 182 | 55 |
| 1000 | 97.63% | 0 | 184 | 53 |

Mode A (retrieval miss) drops to zero at K=500: every correct class is retrievable. Theoretical ceiling is 100.00%. Plateau between K=500 and K=1000.

**Previous topo9 system (joint bytepacked index):** Also K-sensitive but with 7 irreducible Mode A errors.

| K | Topo9 Accuracy |
|---|---------------|
| 50 | 96.59% |
| 200 | 97.27% |
| 500 | 97.57% |
| 1000 | 97.61% |

## Retrieval method comparison (K=50, topo9 ranking)

| Method | Accuracy | Time/query | Retrieval Recall |
|--------|----------|------------|-----------------|
| SSTT Inverted Index | 96.62% | 833 us | 99.63% |
| Brute L2 (SAD, AVX2) | 97.49% | 2532 us | 99.73% |
| Random Projection LSH | 93.46% | 101 us | 99.39% |

Brute L2 produces better candidates (+0.87pp) but is 2.75x slower. L2 re-ranking of inverted-index candidates hurts (see negative results) — the inverted index's "noisy" candidates provide diversity the structural ranker needs.

## Fashion-MNIST (10,000-image test set, zero architectural changes)

| Method | Accuracy | Notes | Validated | Source |
|--------|----------|-------|-----------|--------|
| Brute 1-NN (pixel L2) | 76.86% | baseline | — | — |
| Bytepacked cascade | 82.89% | no tuning from MNIST | — | Doc 15 |
| Three-Tier Router | 83.42% | at 0.88 ms | — | Doc 54 |
| Full system (topo9 + Bayesian) | 85.68% | val-derived weights | val/holdout | Doc 35 |
| MTFP + Pipeline | 85.88% | val-derived weights | val/holdout | sstt_mtfp.c |
| **MTFP + LBP + Bayesian** | **86.54%** | val-derived weights | val/holdout | sstt_mtfp_dsp.c |
| Router Tier 1 (9.6% coverage) | 100.00% | zero false positives | — | Doc 54 |

MTFP Fashion val-derived base weights: grid=3x3, w_c=25, w_p=0, w_d=50, w_g=100, sc=20. LBP (w=200) adds +1.00pp static. Bayesian sequential (dS=1, K=50, tw=100) adds another +0.92pp on holdout. Combined: +1.94pp over MTFP static baseline.

### Fashion-MNIST error analysis

**673 holdout errors** (86.54%, MTFP + LBP + Bayesian). Error structure:

| Confusion pair | Errors | Classes |
|---------------|--------|---------|
| T-shirt ↔ Shirt (0↔6) | 133 | Upper-body |
| Pullover ↔ Shirt (2↔6) | 102 | Upper-body |
| Coat ↔ Shirt (4↔6) | 94 | Upper-body |
| Pullover ↔ Coat (2↔4) | 78 | Upper-body |
| Dress ↔ Shirt (3↔6) | 33 | Upper-body |
| Sandal ↔ Sneaker (5↔7) | ~32 | Footwear |

**61.8% of all errors are inter-confusion among four upper-body garment classes** (T-shirt, Pullover, Coat, Shirt). These share identical silhouette topology at 28x28: similar outer edges, similar enclosed regions (armholes, neckline), similar horizontal intensity profile. They differ only in collar detail (3-4 pixels), fabric texture (sub-pixel at this resolution), and sleeve width (2-3 pixels).

### DSP feature ablation (Fashion-MNIST val)

| Feature | Operates on | Val gain vs MTFP baseline | Val gain vs LBP |
|---------|-------------|--------------------------|-----------------|
| LBP (Local Binary Pattern) | raw uint8 pixels | **+1.00pp** | — |
| Haar wavelet energy | raw uint8 pixels | +0.88pp | subsumed by LBP |
| Gradient orientation histogram | ternary gradients | +0.00pp | +0.00pp |
| Collar-zone WHT | raw pixels, rows 2-8 | — | +0.00pp |
| Vertical symmetry | raw pixels | — | +0.00pp |
| GLCM contrast | raw pixels | — | +0.00pp |

LBP is the only DSP feature with signal on top of the MTFP baseline. Collar WHT, symmetry, and GLCM all fail because 28x28 resolution is below the threshold where spectral and texture features can distinguish garment types. Collar shape is 3-4 pixels, fabric texture is sub-pixel, button plackets are centered and symmetric at this resolution.

### Fashion-MNIST resolution boundary

The remaining ~13.5% error rate is a representational limit at 28x28 ternary. The four upper-body garment classes (0, 2, 4, 6) are genuinely indistinguishable in this representation:
- Silhouette topology: identical (two sleeves, torso, rectangular)
- Edge structure: identical (shoulder seams, side seams, hem)
- LBP texture: nearly identical at 28x28 (knit vs woven is sub-pixel)
- Spectral content: identical (same low-frequency silhouette dominates)

This parallels the CIFAR-10 cat/dog boundary: ternary features capture topology but not texture or fine structural detail. No amount of feature engineering at 28x28 resolution can distinguish a plain T-shirt from a plain shirt — they are the same image at this resolution.

## Mechanism tuning (MNIST)

K × k × h-grad channel scale grid search (30 configurations, all 10K, hardcoded weights):

| Config | Accuracy | Notes |
|--------|----------|-------|
| K=200 k=3 α=100% (baseline) | 97.54% | Current default |
| K=500 k=3 α=100% | 97.63% | K increase |
| K=500 k=3 α=75% | 97.68% | Best in grid |
| K=500 k=1 α=100% | 97.03% | k=1 always worse |

**Statistical significance:** The alpha=75% vs alpha=100% difference is 5 images — not significant (CIs overlap). The K=500 vs K=200 difference is 14 images — directionally consistent across all configurations but not significant at p=0.05 on 10K samples. k=3 vs k=1 (~55 images, ~0.6pp) is the only significant finding.

**Conclusions:** k=3 consensus vote is confirmed optimal. K=500 is marginally better than K=200 (consistent direction, not significant per-comparison). H-grad channel scaling is noise. The validated headline remains **97.53%** (K=200, val/holdout protocol).

## CIFAR-10 (10,000-image test set, 32x32 RGB)

| Method | Accuracy | Notes | Source |
|--------|----------|-------|--------|
| Grayscale ternary cascade | 26.51% | first attempt | Doc 37 |
| MT4 full stack | 42.05% | 81-level + topo | Doc 38 |
| Grid Gauss map (RGB, brute) | 48.31% | shape geometry | Doc 42 |
| **Cascade + RGB Gauss** | **50.18%** | 5x random, best result | Doc 43 |

This is an architectural boundary test. Edge-distinctive classes work (ship 63.7%, truck 62.1%); texture-dependent classes fail (cat 33.6%, dog 40.1%). Ternary block signatures cannot capture texture or semantic content at 32x32.

## Error mode decomposition

**MTFP at K=500 (97.63%, 237 errors):** A: 0 (0%), B: 182 (76.8%), C: 55 (23.2%).

**MTFP at K=200 (97.54%, 246 errors):** A: 4 (1.6%), B: 186 (75.6%), C: 56 (22.8%).

**topo9 at K=200 (97.27%, 273 errors):** A: 7 (2.6%), B: 197 (72.2%), C: 69 (25.3%).

**Bytepacked cascade (96.28%, 372 errors):** A: 7 (1.7%), B: 240 (64.5%), C: 125 (33.8%).

MTFP's per-channel trit-flip retrieval eliminates all Mode A failures by K=500. Theoretical ceiling: **100.00%** (every correct class is retrievable). All remaining errors are ranking (Mode B + C). Top confusion pairs at K=200: 3↔5 (27), 4↔9 (25), 2↔7 (15).

## Channel ablation (MNIST)

**Bytepacked cascade (dot-product weights only, no structural features, K=50, k=3):**

| Weights (px, hg, vg) | Accuracy | Delta |
|--------|----------|-------|
| (256, 0, 0) pixel only | 96.04% | baseline |
| (256, 0, 192) +v-grad | 96.38% | +0.34pp |
| (256, 192, 0) +h-grad | 95.87% | -0.17pp |
| (256, 192, 192) all three | 96.21% | +0.17pp |

**Full topo9 system (dot-product + structural features):**

| Config | Accuracy | Delta |
|--------|----------|-------|
| Pixel only | 96.93% | baseline |
| + V-grad | 97.27% | +0.34pp |
| + H-grad | 96.76% | -0.17pp |
| All three | 97.27% | +0.34pp |

On MNIST, v-grad helps, h-grad hurts in both configurations. On Fashion-MNIST, both help. IG re-weighting automatically redistributes importance per dataset.

## Topological feature ablation (MNIST, topo1 through topo9)

| Step | Addition | Accuracy | Delta |
|------|----------|----------|-------|
| Baseline | Bytepacked cascade | 96.28% | — |
| +divergence | Grid divergence (2x4) | 96.55% | +0.27 |
| +centroid | Enclosed centroid | 96.78% | +0.23 |
| +profile | Horizontal profile | 96.91% | +0.13 |
| +kalman | MAD adaptive weighting | 97.01% | +0.10 |
| +sequential | Bayesian (S=2) | 97.11% | +0.06 |
| +tuning | Grid + param tuning | 97.27% | +0.16 |
