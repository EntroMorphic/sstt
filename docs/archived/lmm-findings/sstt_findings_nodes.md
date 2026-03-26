# LMM Phase 2: NODES — Key Points & Tensions

## Key Structures
1. **The Three-Tier Cascade:** A compute-adaptive hierarchy (Fast/Light/Deep) that doubles throughput by isolating "easy" signals.
2. **Lagrangian Skeletons:** A shift from counting static events (histograms) to tracing dynamic flow (path-tracing), identifying pose-invariant structures.
3. **The Spline Approximation:** Using spatial interpolation to scale lookup tables linearly against quadratic resolution growth.

## Primary Tensions
- **T1: Precision vs. Invariance.** Eulerian Gauss maps provide high-precision texture matching but fail on pose changes. Lagrangian tracers are pose-invariant but lack the resolution for fine-grained class separation.
- **T2: Variance vs. Coverage.** Statistical variance identifies "interesting" regions, but geometric classification requires uniform coverage to maintain structural integrity.
- **T3: The Memory Wall.** 87% of time is spent on memory writes. Our speed is bounded by the hardware's random-access throughput, not the ternary arithmetic.
- **T4: Dataset Specialization.** MNIST (clean strokes) prefers high-unanimity plurality; Fashion (noisy textures) requires geometric map refinement even on easy cases.

## Constraints
- **Zero Learned Parameters:** We cannot optimize weights via backprop. All gains must come from closed-form MI or geometric first principles.
- **Cache Residency:** To maintain sub-millisecond latency, the hot maps and indices must stay within L3 (or L2 for Fast tiers).
