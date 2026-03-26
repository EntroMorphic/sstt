# Contribution 49: Three-Tier Router Design

The Three-Tier Router is the architectural culmination of the SSTT project. It adapts the compute budget per query based on a "Free" confidence signal extracted during the initial vote phase.

## Tier Specification

### 1. Fast Tier (The Instant Path)
- **Signal:** High unanimity in retrieval votes (e.g., top-1 candidate has >10x votes of top-2).
- **Mechanism:** Dual Hot Map (Pixel + Gauss) lookup.
- **Accuracy:** ~76.5% base; 99%+ on high-confidence slice.
- **Latency:** ~1.5 μs.
- **Target Coverage:** ~20-30% of images.

### 2. Light Tier (The Structural Path)
- **Signal:** Moderate vote spread or ambiguous Dual Map output.
- **Mechanism:** Bytepacked block dot products + Lagrangian Particle Tracer (64-dim skeleton).
- **Accuracy:** ~92-96% MNIST.
- **Latency:** ~150-200 μs.
- **Target Coverage:** ~40-50% of images.

### 3. Deep Tier (The Geometric Path)
- **Signal:** High entropy in votes; conflicting structural signals.
- **Mechanism:** 3-Eye Stereoscopic Retrieval → Full RGB Gauss Map + Curvature + Curl ranking.
- **Accuracy:** 97.27% MNIST / 86% Fashion / 50%+ CIFAR-10.
- **Latency:** ~1-3 ms.
- **Target Coverage:** ~20-30% of images.

## Routing Logic

```c
Image Q;
Votes V = retrieve_votes(Q); // Fast O(1) lookup
Confidence C = calculate_confidence(V);

if (C > THRESHOLD_HIGH) {
    return fast_tier_output(V); // Instant
} else if (C > THRESHOLD_LOW) {
    return light_tier_rank(Q, V); // <200us
} else {
    return deep_tier_rank(Q, V); // ~2ms
}
```

## Hardware Advantage
The router allows for **asynchronous inference**. Easy images are returned immediately, freeing up the pipeline for harder geometric analysis. The entire Fast and Light tiers fit comfortably in **L2 cache**.

---
## Implementation
- Prototype: `src/sstt_router_v1.c`
- Tracking: [ ] Benchmark Precision-Coverage
