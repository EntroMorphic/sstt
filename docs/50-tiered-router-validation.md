# Contribution 50: Tiered Router Validation (MNIST)

This experiment validates the performance of the Three-Tier Router prototype, demonstrating the ability to adapt compute per image without significant accuracy loss on the high-confidence slice.

## Tier Specification & Empirical Results

| Tier | Mechanism | Coverage | Tier Accuracy | Latency (est) |
| :--- | :--- | :--- | :--- | :--- |
| **T1 (Fast)** | Instant Plurality | 14.8% | **99.93%** | <1.0 μs |
| **T2 (Light)**| Dual Hot Map | 33.0% | 92.00% | ~1.5 μs |
| **T3 (Deep)** | Block Dot Ranking| 52.3% | 93.00% | ~1.0 ms |

### Overall Prototype Accuracy: 93.69%

## Key Findings
1.  **The Perfect 15%:** We identified ~15% of MNIST that can be classified with near-perfect accuracy (99.93%) using only the information already available in the retrieval vote buffer.
2.  **Tier Transition:** The "Unanimity Threshold" (195/200) successfully isolates the ultra-easy cases.
3.  **Efficiency Gains:** In a production system, 47.8% of queries (T1 + T2) would bypass the expensive dot-product and geometric stages entirely, providing a **2x increase in throughput** for a 2.5pp drop in global accuracy.

## Next Steps
1.  **Integrate Topology into T3:** Restoring the divergence and centroid features to T3 will pull the global accuracy back toward the 97% mark.
2.  **Fashion-MNIST Thresholds:** Tuning the unanimity thresholds for the more complex Fashion dataset.

---
## Files
- Code: `src/sstt_router_v1.c`
