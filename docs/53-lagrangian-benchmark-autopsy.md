# Contribution 53: Lagrangian Benchmark Autopsy

This contribution documents the head-to-head comparison of the repository's top-tier CIFAR-10 rankers and analyzes the "Fusion Drop" identified during the final benchmark.

## Benchmark Results (CIFAR-10 N=1000)

| Method | Accuracy | Latency | Discovery |
| :--- | :--- | :--- | :--- |
| **RGB Gauss Map** | **48.50%** | ~180 ms | Geometric Baseline |
| **Lagrangian Tracer** | 17.80% | ~2 ms | Topological Skeleton |
| **Unified Combo** | 34.50% | ~185 ms | **Partial Negative** |

## Diagnosis: The Fusion Drop
The naive weighted fusion ($Gauss + Tracer$) resulted in a **-14pp accuracy drop** compared to the Gauss baseline. 

### 1. Feature Scaling Mismatch
The RGB Gauss features operate on high-precision gradient histograms with distances in the $10^3$-$10^4$ range. The Lagrangian Tracer produces a 64-dimensional sparse skeleton with distances in the $10^1$ range. When summed (even with logarithmic normalization), the noisier high-level skeleton muddies the precision of the geometric match.

### 2. Information Sparsity
The Tracer identifies *types* of shapes (loops vs. lines) but lacks the spatial resolution to distinguish between specific class instances (e.g., a "round" bird vs. a "round" frog).

## Path Forward: Sparse Topological Gating
The benchmark suggests that Lagrangian features should not be *fused* with geometric features, but used as **Gating Logic**:
- **Eulerian Ranker:** Best for machine classes (planes, ships) with rigid geometry.
- **Lagrangian Ranker:** Best for animal classes (cats, dogs) with pose-variant skeletons.
- **Proposal:** Use the Tracer to select the specialized ranker rather than adding their scores.

---
## Files
- Code: `src/sstt_benchmark_cifar10.c`
