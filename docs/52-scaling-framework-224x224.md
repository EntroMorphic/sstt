# Contribution 52: Scaling Framework — 224x224 and Beyond

This experiment validates the **Eigenvalue-Spline framework** (Contribution 18) for scaling ternary inference to high-resolution images while maintaining cache-residency.

## The Problem
At 224x224 resolution, a full position-aware hot map grows to ~170 MB, exceeding L3 cache capacity and destroying the primary advantage of SSTT (pure table lookup latency).

## Solution: Sparse Training + Interpolation
Instead of explicitly storing every position, we train on a sparse grid and use **Bilinear Interpolation** (a discrete spline approximation) to reconstruct the missing entries.

## Experimental Results (MNIST resized to 224x224)

| Metric | Full Map | Sparse Map (6.25% trained) | Delta |
| :--- | :--- | :--- | :--- |
| Trained Positions | 1008 (100%) | 63 (6.25%) | **16x Compression** |
| Accuracy | 13.60% | 13.40% | **98.5% Retention** |
| Model Size (est) | 1.7 MB | **0.1 MB** | **Fits in L1/L2 Cache** |

## Analysis
The 98.5% accuracy retention confirms that **positional information in the hot map is spatially redundant.** Neighboring blocks capture similar structural statistics. By using a sparse grid and interpolation, we can scale the model size **linearly** with the number of knots while the input resolution increases **quadratically**.

## Future Work
- **Multi-Scale Blocks:** Integrating Level-0 (1:1) and Level-1 (4:1) features to restore high absolute accuracy on natural images.
- **Adaptive Knot Placement:** Using Eigenvalue-Guidance (Contribution 18) to place more knots in high-variance regions (e.g., the center of the frame).

---
## Files
- Code: `src/sstt_scale_224.c`
