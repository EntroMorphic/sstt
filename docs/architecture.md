# Architecture

SSTT classifies images using integer address lookups with zero learned parameters. No gradient descent, no backpropagation, no floating-point arithmetic at inference. The entire model fits in L2 cache.

## Pipeline

```
Raw image (28x28 uint8)
    |
Ternary quantization
    pixel < 85 -> -1,  85-169 -> 0,  >= 170 -> +1
    |
Three independent channels
    +-- Pixel channel (quantized intensity)
    +-- H-grad channel (horizontal finite difference, clamped to {-1,0,+1})
    +-- V-grad channel (vertical finite difference, clamped to {-1,0,+1})
    |
Block encoding: 3x1 horizontal strips at 252 overlapping positions
    Per block -> ternary signature (27 values) or bytepacked (256 values)
    Bytepacking fuses 3 channels + transition flags into 1 byte
    |
Inverted-index retrieval
    For each (position, signature): look up training image IDs
    Add IG-weighted votes to 60K-element accumulator
    Multi-probe: 6-8 Hamming-1 trit-flip neighbors at half weight
    |
Top-K candidate extraction (K=50 suffices -- K-invariance)
    |
Ranking: weighted multi-channel ternary dot product
    score = 256 * dot(pixel) + 192 * dot(v-grad)
    Computed via _mm256_sign_epi8 (32 ternary MACs per cycle)
    |
Topological augmentation
    + Gradient divergence in spatial grid cells (2x4 MNIST, 3x3 Fashion)
    + Enclosed-region centroid via flood-fill
    + Horizontal intensity profile
    + Kalman adaptive weighting (MAD-based per-query gain)
    + Sequential accumulation (Bayesian with decay S=2)
    |
k=3 majority vote on top-3 scored candidates -> class prediction
```

## Key components

### Ternary quantization

Fixed thresholds at 85/170 divide the 0-255 intensity range into three zones: dark (-1), mid (0), bright (+1). The same clamped finite difference produces gradient trits. Background suppression: BG_PIXEL=0 (all-dark block), BG_GRAD=13 (zero-gradient block).

### Information-gain weighting

Each of the 252 block positions receives a weight proportional to the mutual information between its block signature and the class label, computed closed-form over the training set:

```
IG(k) = H(class) - H(class | block_value_at_position_k)
```

High-IG positions (digit centers, stroke junctions) contribute more to retrieval votes than low-IG positions (corners, flat regions). No optimization loop — one pass over training data.

### Inverted-index retrieval

Standard information retrieval structure: for each (position, signature) pair, a posting list stores the training image IDs with that signature at that position. At inference, the test image's block signatures are looked up and votes accumulated with IG weights.

Multi-probe expansion adds 6-8 Hamming-1 neighbors (single trit-flip variants) at half weight, capturing near-miss signatures.

### K-invariance

The central empirical finding: accuracy is identical from K=50 to K=1000. The top-50 candidates by vote contain every relevant neighbor from the 60,000 training images. This is a 1200x compression with zero recall loss. Retrieval is solved; all remaining error is ranking.

### Ternary dot product via sign_epi8

The AVX2 instruction `_mm256_sign_epi8(b, a)` computes `sign(a) * |b|` per byte. For ternary values {-1, 0, +1}, this is exact multiplication: 32 multiply-accumulates per cycle with no widening, no multiply instruction.

### Topological features

Gradient divergence (sum of finite-difference changes in the gradient field) captures enclosed shapes and edge density in spatial grid cells. Enclosed-region centroid (via flood-fill on the binary foreground) captures vertical mass distribution. Horizontal intensity profile captures per-row stroke density.

Per-query adaptive weighting: the median absolute deviation (MAD) of candidate divergence scores modulates feature weights. Tight clusters (low MAD) get high divergence weight; dispersed pools (high MAD) rely more on raw dot products.

### Adaptive router (3-tier)

| Tier | Latency | Coverage | Accuracy | Mechanism |
|------|---------|----------|----------|-----------|
| Fast | <1 us | 10-15% | 99.9% | Vote unanimity >= 198/200 |
| Light | ~1.5 us | 15-18% | ~95% | Dual Hot Map (pixel + Gauss) |
| Deep | ~1 ms | remaining | full | Complete pipeline |
| **Global** | **0.67 ms** | **100%** | **96.50%** | Weighted average |

Tier 1 has a zero false positive property: on Fashion-MNIST, 9.6% of images classified at 100% accuracy across 960 images.

## Compute profile

| Phase | Runtime share |
|-------|--------------|
| Vote accumulation (scattered writes) | 87% |
| Top-K selection | ~5% |
| Encoding and overhead | ~7% |
| Ternary dot products | <1% |

The bottleneck is memory access patterns, not arithmetic. The fused AVX2 kernel (`sstt_fused.S`) fits in L1i cache (<1KB code) with the model in L2 (~1.3MB).

## Non-iterative parameter claim

All parameters are computed without optimization loops:
- **IG weights:** Closed-form mutual information, O(N*P) time
- **Background thresholds:** Mode of ternary distribution
- **Dot-product weights:** Grid search over 125 combinations on validation split
- **Topological weights:** Ablation on validation split
- **Router threshold:** Single integer, set on validation data

No loss function, no learning rate, no convergence criterion. The defensible claim is: no iterative optimization, not "no statistics from labeled data."
