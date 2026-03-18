# Contribution 57: Taylor Jet Space Classification

This experiment introduces the **Local Taylor Jet** (2nd-order Taylor expansion) to classify the local geometric structure of the ternary field.

## Theoretical Basis
Instead of only counting edges (1st order), we compute the **Discrete Hessian Matrix** at every pixel to identify surface primitives:
$$H = \begin{bmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{bmatrix}$$

Using the Determinant ($D$) and Trace ($T$) of the Hessian, we classify each pixel into six geometric categories:
- **Flat:** No gradient or curvature.
- **Pure Edge:** 1st-order gradient without local curvature.
- **Peak/Pit:** Local maximum or minimum (Ridge/Valley).
- **Saddle Point:** Junction or crossing.
- **Linear Ridge/Valley:** Parabolic structure (strokes).

## Experimental Results (MNIST)

| Method | Feature Dimension | Accuracy (k=1 Brute) |
| :--- | :--- | :--- |
| Pixel-Space (Baseline) | 784 | 97.90% |
| **Taylor Jet (8x8 Grid)** | **512 (int16)** | **91.60%** |
| Taylor Jet (4x4 Grid) | 96 (int16) | 89.00% |

## Experimental Results (CIFAR-10 Grayscale)

| Method | Accuracy |
| :--- | :--- |
| Random | 10.00% |
| **Taylor Jet (8x8 Grid)** | **27.20%** |

## Analysis
- **Noise Resilience:** Taylor Jet features provide high accuracy (91.6%) on MNIST with significant dimensionality reduction (1.5x compression vs pixels), proving that higher-order local geometry captures the "intent" of strokes.
- **CIFAR-10 Signal:** While 27% on grayscale natural images is not a headline result, it is 2.7x random, indicating that local curvature provides a complementary signal to the global Gauss map histograms.

## Next Steps
Integrate the 2nd-order Taylor terms into the **Tier 3 Deep Ranker** to resolve fine-grained ambiguities (e.g., distinguishing the curved body of a frog from the angular structure of a ship).

---
## Files
- Code: `src/sstt_taylor_jet.c`
