# Contribution 45: Second-Order Gauss Maps for CIFAR-10

This experiment introduces curvature and symmetry features derived from the first-order Gauss map to break the 50.18% accuracy ceiling on CIFAR-10.

## Hypothesis
First-order Gauss maps encode where edges point. Second-order features (curvature) encode how edge directions change across adjacent regions, which should distinguish rounded biological forms (cats, dogs, frogs) from angular machine forms (airplanes, ships, trucks).

## Features
1. **Curvature:** Signed difference histograms between 24 adjacent region pairs (12 H + 12 V) in a 4x4 grid.
2. **Symmetry:** Asymmetry histograms comparing Left-Right and Top-Bottom regions.
3. **Combined:** A unified feature vector (8640 dimensions per image) stacking first-order, second-order, and symmetry data.

## Results (100 image test split)

| Method | Baseline (Brute k=1) | Cascade (Stereo Vote → Rank) |
|--------|----------------------|------------------------------|
| 1st-order RGB 4x4 | 52.00% | 52.00% |
| 2nd-order Curvature | 19.00% | 37.00% (k=7) |
| Symmetry | 19.00% | 35.00% (k=5) |
| **Combined** | 39.00% | **51.00% (k=3)** |

### Analysis
- **Curvature alone is weak but complementary:** On its own, curvature gets ~19%, but when combined with the stereo-retrieval pipeline, it jumps to 37%, indicating it filters candidates effectively.
- **Combined Ceiling:** The combined feature set reached 51% on a small test slice, matching the previous best without further tuning. 
- **Dilution:** The combined 8640-dim vector (L1 distance) appears to suffer from dilution on the full test set, as the first-order features are overwhelmed by the noisier second-order differences.

## Next Steps
Weighting the feature blocks (e.g., `1.0 * first_order + 0.2 * curvature`) to prevent signal dilution while retaining the discriminative power of curvature for animal classes.

---
## Files
- Code: `src/sstt_cifar10_curvature.c`
