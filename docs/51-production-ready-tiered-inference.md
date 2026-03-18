# Contribution 51: Production-Ready Tiered Inference

This contribution integrates the high-accuracy topological features into the **Three-Tier Router**, establishing a production-ready engine that adapts compute power to query difficulty.

## Final Benchmark (MNIST 10K)

| Tier | Mechanism | Coverage | Tier Accuracy | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **T1 (Fast)** | Vote Plurality | 10.0% | **99.90%** | ~1 μs |
| **T2 (Light)**| Dual Hot Map | 15.8% | 97.03% | ~1.5 μs |
| **T3 (Deep)** | Full Topo Ranking | 74.2% | 95.93% | ~1.0 ms |

### **Global Accuracy: 96.50%**
### **Global Latency: 0.67 ms / query**

## Architectural Integration
The Deep Tier (T3) now utilizes the complete feature set discovered during the research phase:
1.  **Divergence (Green's Theorem):** Flux-based edge density.
2.  **Enclosed Centroid:** Topological skeleton center of mass.
3.  **Horizontal Profile:** Row-wise structural intensity.
4.  **Grid Divergence:** 2x4 spatial distribution of gradient flux.
5.  **Kalman Adaptive Weighting:** Per-query weight adjustment based on candidate variance (MAD).
6.  **Bayesian Sequential Update:** Evidence accumulation with decay ($S=2$).

## The Pareto Trade-off
By routing 25.8% of images through the Fast/Light paths, we achieve a **2x throughput increase** compared to a full geometric run, with a minimal accuracy cost of **0.77pp** (96.50% vs 97.27%).

## Stability
The prototype in `src/sstt_router_v1.c` is the most stable and feature-complete entry point in the repository, consolidating 50 prior contributions into a single unified inference logic.

---
## Files
- Code: `src/sstt_router_v1.c`
