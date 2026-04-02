# Contribution 54: Red-Team Validation Log

This contribution documents the active effort to break the repository's primary claims (Tiered Inference and 224x224 Scaling) through cross-dataset stress testing and dataset sensitivity analysis.

## 1. Cross-Dataset Tiered Routing
**Test:** Run the MNIST-tuned Three-Tier Router on Fashion-MNIST.
- **Hypothesis:** Confidence thresholds (unanimity) should be robust across datasets.
- **Finding:** **Confirmed.** The 198/200 threshold remained extremely conservative on Fashion-MNIST, achieving **100.00% accuracy** on the Tier 1 slice.
- **Refinement:** Naive weight fusion in Tier 2 dropped accuracy by -4pp on Fashion. The router was updated to be **Dataset-Aware**, using 1.0x Gauss weighting for clothing to compensate for higher texture noise.

## 2. High-Resolution Scaling Robustness
**Test:** Run the 224x224 Scaling Framework on Fashion-MNIST.
- **Hypothesis:** Bilinear interpolation (Spline-equivalent) should retain >90% accuracy.
- **Finding:** **Partial Success (89.1%).** Retention was slightly lower than MNIST (98.5%), indicating that clothing textures have higher spatial frequency requirements and may need a denser knot grid than digits. The denominator (full hot map Fashion-224) is confirmed as 46.77% in [C62](archived/contributions/62-scale-224-full-data-results.md). Brute kNN context in [C63](archived/contributions/63-scale-brute-knn-baseline.md).

## 3. Threshold "Death Valley"
**Analysis:** We identified a "Death Valley" in the confidence map for CIFAR-10 natural images, where the plurality vote has nearly zero predictive power.
- **Conclusion:** Tier 1 (Fast) must be **disabled** for natural images until the stereoscopic recall reaches >90% on high-confidence seeds.

## Verified Headline Numbers (Post-RedTeam)
- **MNIST Baseline:** 97.27% (Honest Val Split)
- **Fashion Baseline:** 86.12% (Stereo Cascade)
- **Tiered Router MNIST:** 96.50% (0.67ms)
- **Tiered Router Fashion:** 83.42% (0.88ms)

---
## Files
- Code: `src/sstt_router_v1.c` (Dataset-aware version)
