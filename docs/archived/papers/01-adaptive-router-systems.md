# Paper 1: Adaptive Tiered Inference via Vote-Concentration Confidence

**Abstract.** High-speed industrial vision systems require sub-millisecond classification with minimal compute overhead. We present the **Adaptive Three-Tier Router**, a non-neural inference engine that adapts its compute budget per query using a "free" confidence signal extracted from the retrieval phase. By analyzing the unanimity of class votes in an inverted index, the system isolates "canonical" images for instant return (<1μs), reserving complex geometric ranking for ambiguous cases. On MNIST, the router classifies 15% of the test set with **99.9% accuracy instantly**, doubling total system throughput with a negligible 0.7pp drop in global accuracy.

## Key Contributions
1. **The Free Confidence Signal:** Proving that retrieval-phase vote concentration (plurality vs. margin) is a robust proxy for signal-to-noise ratio in ternary manifolds.
2. **Three-Tier Hierarchy:**
    - **Fast Tier (T1):** Plurality-based instant return.
    - **Light Tier (T2):** Dual Hot Map (Pixel + Gauss) O(1) lookup.
    - **Deep Tier (T3):** Full topological ranking.
3. **Hardware-Aware Design:** Optimizing for cache-line utilization and bus-cycle reduction by bypassing the memory-heavy signature retrieval step for 25-50% of queries.

## Results
- **MNIST:** 96.50% Global Acc / 0.67ms latency.
- **Throughput:** 2x lift compared to non-tiered cascade.
- **Safety:** Zero false positives (100% accuracy) on the T1 slice for Fashion-MNIST clothing.
