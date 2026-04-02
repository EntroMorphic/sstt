# Contribution 51: Production-Ready Tiered Inference

This contribution integrates the high-accuracy topological features into the **Three-Tier Router**, establishing a production-ready engine that adapts compute power to query difficulty.

## Benchmark Results (10K Test Set)

| Dataset | Metric | Global | T1 (Fast) | T2 (Light) | T3 (Deep) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MNIST** | Accuracy | **96.50%** | 99.90% | 97.03% | 95.93% |
| **MNIST** | Coverage | 100% | 10.0% | 15.8% | 74.2% |
| **Fashion** | Accuracy | **83.42%** | 100.00% | 90.11% | 79.65% |
| **Fashion** | Coverage | 100% | 9.6% | 17.5% | 73.0% |

### **Global Latency: 0.67ms (MNIST) / 0.88ms (Fashion)**

## Architectural Integration
The Deep Tier (T3) now utilizes the complete feature set discovered during the research phase:
1.  **Divergence (Green's Theorem):** Flux-based edge density.
2.  **Enclosed Centroid:** Topological skeleton center of mass.
3.  **Horizontal Profile:** Row-wise structural intensity.
4.  **Grid Divergence:** 2x4 spatial distribution of gradient flux.
5.  **Kalman Adaptive Weighting:** Per-query weight adjustment based on candidate variance (MAD).
6.  **Bayesian Sequential Update:** Evidence accumulation with decay ($S=2$).

## Red-Team Refinement (Contribution 54)
Validation on Fashion-MNIST identified a "Feature Scaling Mismatch." The production router was updated to be **Dataset-Aware**:
- **MNIST:** Geometric features (Gauss Map) are weighted at 0.5x to prevent over-powering the clean pixel signal.
- **Fashion:** Geometric features are weighted at 1.0x to handle the high texture variation where pixel overlap is unreliable.
- **Thresholds:** The "Unanimity Threshold" of 198/200 was found to be highly stable, producing **zero false positives** (100% accuracy) on the T1 slice for Fashion-MNIST.

## The Pareto Trade-off
By routing 25.8% of images through the Fast/Light paths, we achieve a **2x throughput increase** compared to a full geometric run, with a minimal accuracy cost of **0.77pp** (96.50% vs 97.27%).

## Stability
The prototype in `src/sstt_router_v1.c` is the most stable and feature-complete entry point in the repository, consolidating 50 prior contributions into a single unified inference logic.

---
## Files
- Code: `src/sstt_router_v1.c`
