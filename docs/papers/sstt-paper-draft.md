# Zero-Parameter Image Classification via Tiered Ternary Cascade, Lagrangian Skeletons, and Gauss Map Geometry

**Authors:** [Names]

**Abstract.** We present SSTT, a parameter-free image classifier that
uses no gradient descent, no backpropagation, and no floating point at
inference. The system achieves 97.27% on MNIST and 85.68% on
Fashion-MNIST through a tiered pipeline of integer table lookups, AVX2
byte operations, and information-theoretic feature weighting. We
introduce three key innovations: (1) **Three-Tiered Routing**, which
adapts compute power to query difficulty, achieving 99.9% accuracy on
the easiest 15% of images instantly; (2) **Lagrangian Structural
Analysis**, using discrete Curl and particle tracing to map topological
skeletons (loops and curves) in natural images; and (3) **Gauss Map
Geometry**, which classifies on edge distributions rather than pixel
textures. On CIFAR-10, we demonstrate that these combined techniques
reach 50.18% (5× random) with zero learned parameters. We document the
complete experimental progression: 51 contributions, 40+ CIFAR-10
experiments, and a rigorous methodology for architectural
consolidation. Code is available at [repository URL].

---

## 1. Introduction

Image classification methods fall into two paradigms: neural networks
that learn everything from data through gradient descent, and classical
methods that compute everything from fixed mathematical operations.
Neural networks achieve state-of-the-art accuracy but require heavy
training infrastructure and floating-point arithmetic. Classical
methods require no training but typically achieve lower accuracy.

We present a third position: **compute features from differential
geometry and information theory, vote via inverted index, rank via
adaptive topology, and route via confidence.** The system uses zero
learned parameters — every weight is derived from closed-form mutual
information. Inference uses only integer arithmetic: table lookups,
add/subtract, and compare. No multiply instruction. No floating point.
The entire model fits in fast cache.

The core contribution of this work is the **architectural consolidation**
of 51 research experiments into a unified, tiered inference engine:

1. **Three-Tier Router:** Adapting the compute budget per-query based
   on a free confidence signal. Easy images are returned instantly via
   Dual Hot Map lookups, while hard cases escalation to geometric
   ranking.
2. **Lagrangian Skeletons:** Moving beyond Eulerian histograms into
   dynamic field analysis. We use discrete Lagrangian vorticity (Curl)
   and path tracing to identify the topological curves of organic forms
   and the straight edges of machines.
3. **Gauss Map Classification:** Mapping pixel gradients onto a unit
   sphere to classify on edge geometry, inherently suppressing
   non-informative backgrounds.

We validate these techniques across MNIST, Fashion-MNIST, and CIFAR-10,
providing a menu of operating points on the speed-accuracy Pareto
frontier.

---

## 2. Method

### 2.1 Ternary Quantization and Block Encoding

Each pixel is quantized to {-1, 0, +1} using fixed (85/170) or adaptive
(P33/P67) thresholds. The AVX2 instruction `_mm256_sign_epi8` computes
exact multiplication over this ternary field: 32 multiply-accumulates
per cycle without a multiply instruction. 

Three adjacent pixels form a block signature. Encoding D packs pixel,
h-gradient, and v-gradient majority trits into a single byte [Contribution 22].
An inverted index maps block values to training IDs. IG-weighted votes
retrieve the top-K candidates for ranking [Contribution 7, 8].

### 2.2 Three-Tier Adaptive Routing [Contribution 49-51]

Inference is performed through an adaptive pipeline that minimizes
unnecessary computation:

- **Tier 1 (Fast):** Direct plurality output from the retrieval phase
  vote concentration. Used when unanimity is high (e.g., top-1 has 10x
  votes of top-2). Latency: <1 μs.
- **Tier 2 (Light):** Dual Hot Map lookup. Fuses texture (Pixel) and
  geometric (Gauss) frequency tables to refine predictions without
  ranking. Latency: ~1.5 μs.
- **Tier 3 (Deep):** Full ranker. Retrieves candidates and applies
  topological features (Divergence, Centroid, Profile) and Gauss map
  geometry. Latency: ~1 ms.

### 2.3 Lagrangian Structural Analysis [Contribution 46-47]

To break the ceiling on natural images, we shift from static histograms
to Lagrangian flow analysis:

**Discrete Curl.** We identify rotational energy (vortices) in the ternary
gradient field: $\text{Curl}(x,y) = \Delta V / \Delta x - \Delta H / \Delta y$.
This identifies topological junctions and corners.

**Structural Particle Tracing.** We spawn "particles" at high-gradient
points and follow the flow perpendicular to the gradient. We track
path length (structural continuity), curvature (rate of direction change),
and closure rate (loop formation). This identifies the rounded edges
of animals vs. the straight hulls of ships.

---

## 3. Results

### 3.1 Speed-Accuracy Pareto Frontier

| Method | MNIST | Fashion | Latency | Regime |
|--------|-------|---------|---------|--------|
| **Dual Hot Map** | 76.53% | 62.21% | 1.5 μs | Instant |
| **Tiered Router v1** | **96.50%** | — | **0.67 ms** | Production |
| **Field-Theoretic** | 97.27% | 85.68% | ~1 ms | Research |

The **Tiered Router** achieves a 2x throughput increase over non-tiered
approaches by routing 25.8% of queries through the Fast/Light paths,
with a negligible 0.7pp drop in global accuracy.

### 3.2 CIFAR-10 Breakthroughs

The Gauss map geometry at 48.31% remains the highest standalone accuracy.
When placed inside a cascade (Texture Retrieval → Gauss Rank), the system
reaches **50.18%**. Second-order curvature and Curl features identify
organic forms, with Curl alone providing 2.6x random accuracy (26%).

---

## 4. Conclusion

SSTT demonstrates that zero-parameter classification through integer
lookup tables can match kNN on structured data and significantly exceed
it on natural images (+13pp over brute kNN). The introduction of **tiered
inference** and **Lagrangian skeletons** marks the transition from
isolated research to a production-ready engine that adapts its power
to the complexity of the world.

The architectural patterns identified—Retrieval → Perspective → Ranking
→ Routing—provide a blueprint for efficient, non-neural inference
on modern SIMD hardware.
