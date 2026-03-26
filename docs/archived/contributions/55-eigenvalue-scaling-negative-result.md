# Contribution 55: Variance-Guided Scaling — A Negative Result

This experiment tested the hypothesis from Contribution 18 that **Eigenvalue-weighted (Variance-Guided)** knot placement would improve the efficiency of high-resolution ternary hot maps.

## The Experiment
We scaled MNIST to 224x224 and compressed the positional dimension of the hot map using two strategies:
1. **Uniform Grid:** Knots placed at regular intervals across the entire field.
2. **Variance-Guided:** Knots concentrated in regions with the highest block-signature variance.

## Results (N=512 Knots, 16x Compression)

| Strategy | Accuracy | Delta |
| :--- | :--- | :--- |
| Uniform Grid | 14.00% | Baseline |
| **Variance-Guided** | 9.00% | **-5.00pp** |

## Diagnosis: The "Clustering Trap"
Variance identifies where data *changes*, which in MNIST corresponds to the "fuzzy edges" of digit strokes. Concentrating knots in these high-variance regions creates a **Spatial Coverage Gap**:
- The "informative" regions were over-sampled, leading to redundant local data.
- The "stable" but critical regions (like the solid centers of strokes or consistent stroke paths) were under-sampled, leaving holes in the geometric representation.

## Conclusion
For sparse, centered data like MNIST, **Geometric Uniformity** is more important than **Statistical Variance**. A scaling framework for natural images should likely use a **Hybrid Strategy**: a uniform base grid with high-resolution "foveal" knots in areas of high Information Gain.

---
## Files
- Code: `src/sstt_scale_pro.c`
