# Contribution 48: Dual Hot Map — Fast Path Reduction

This experiment validates the "Maps All The Way Down" hypothesis by fusing two independent, zero-compute lookup tables to create a high-accuracy fast-tier classifier.

## Architecture
1. **Pixel Map:** Ternary block frequencies (Contribution 2). Stores class counts for 252 horizontal 3x1 blocks.
2. **Gauss Map:** Geometric gradient frequencies. Stores class counts for 720 bins in a 4x4 spatial grid.
3. **Fusion:** Scores are accumulated from both maps in a single pass.
   - $\text{TotalScore}(C) = \sum \text{PixelMap}(C) + \frac{1}{2} \sum \text{GaussMap}(C)$

## Results

| Dataset | Pixel Map | Gauss Map | Dual Map (Fused) | Latency |
|---------|-----------|-----------|------------------|---------|
| MNIST | 71.35% | 72.26% | **76.53%** | 1.57 μs |
| Fashion | 48.04% | **62.21%** | 58.12% | 1.82 μs |

## Key Findings
- **Independence:** On MNIST, the two maps capture largely orthogonal information (texture vs. geometry), leading to a **+3.5pp** jump when combined.
- **Signal Dominance:** On Fashion-MNIST, the geometric map is significantly more discriminative than the pixel map. Naive fusion hurts performance, suggesting that the "Fast Tier" should adaptively select or weight maps based on the dataset.
- **Zero Ranking Cost:** This result is achieved using only $O(1)$ table lookups and vector additions. No candidates are retrieved, and no dot products are computed.

## Conclusion
We have established a new baseline for "Instant Inference." By expanding the lookup table to include geometric features, we can move the accuracy floor from 73% to 76.5% with negligible latency increase.

---
## Files
- Code: `src/sstt_dual_hot_map.c`
