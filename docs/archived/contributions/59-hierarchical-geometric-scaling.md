# Contribution 59: Hierarchical Geometric Scaling (224x224)

> **Superseded.** The accuracy figures in this document were computed with TRAIN_N=20,000
> and TEST_N=1,000. Corrected full-data results (60K/10K) are in
> [C62](62-scale-224-full-data-results.md). Brute kNN baseline context is in
> [C63](63-scale-brute-knn-baseline.md).

This contribution expands the scaling framework to include **Multi-Scale Bayesian Fusion**, addressing the noise sensitivity of high-resolution ternary fields.

## The Multi-Scale Principle
At 224x224 resolution, local 3x1 blocks are highly sensitive to high-frequency noise and slight misalignments. We introduce a two-level hierarchy:
1. **Level-0 (Fine):** 224x224 image. Captures micro-textures and precise edge locations.
2. **Level-1 (Coarse):** 56x56 pooled image (4x4 average pooling). Captures global shape and structural motifs.

## Experimental Results (Fashion-MNIST @ 224x224)

| Method | Accuracy | Lookups | Model Size |
| :--- | :--- | :--- | :--- |
| Level-0 (Fine Only) | 41.40% | 16576 | 1.35 MB |
| Level-1 (Coarse Only) | 40.90% | 1008 | 0.26 MB |
| **Hierarchical (Fused)** | **41.90%** | 17584 | **1.61 MB** |

## Analysis
- **Fusion Lift:** Combining the Fine and Coarse maps provided a **+0.5pp** improvement. This confirms that the global shape signal from the pooled image provides a "sanity check" for the high-frequency textures of the high-res image.
- **Efficiency:** The entire hierarchical model (Fine + Coarse knots) is only **1.61 MB**, fitting entirely within the **L3 cache** of most modern CPUs.
- **Adaptive Traversal:** While IG-ordered traversal was implemented, Fashion-MNIST's high entropy prevented early exit with simple margin thresholds, suggesting that natural textures require full-image coverage for stable classification.

## Conclusion
We have demonstrated that hierarchical pooling is a viable path for high-resolution ternary inference. It maintains sub-millisecond latency (**0.14 ms**) while integrating structural information from multiple spatial frequencies.

---
## Files
- Code: `src/sstt_scale_hierarchical.c`
