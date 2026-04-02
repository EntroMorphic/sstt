# Contribution 22: Bytepacked Cascade

## Hypothesis

The full 3-channel cascade (96.12%) uses independent ternary block signatures
(27 values each). The bytepacked hot map (91.83%) uses a single 8-bit signature
capturing cross-channel correlations. If the cascade's inverted-index voting is
run on the richer bytepacked representation, it should exceed 96.12% because:
- Each vote lookup uses a more informative block key
- Training images that match in multiple channels accumulate votes more
  precisely under the joint encoding

## Architecture

Same pipeline as sstt_v2.c Test E, but with one channel (256 values) instead
of three channels (27 values):

```
Test image
  ├─→ Encoding D: pixel majority trit (2 bits)
  │              h-grad majority trit (2 bits)
  │              v-grad majority trit (2 bits)
  │              h-transition active  (1 bit)
  │              v-transition active  (1 bit)
  │              → 1 byte per block position
  │
  ├─→ IG weights from bytepacked class distributions
  │
  ├─→ Multi-probe: 8 Hamming-1 bit-flip neighbors at half weight
  │   (vs 6 trit-flip neighbors for ternary)
  │
  └─→ Top-K → ternary dot product refinement → k=3 vote
```

## Results

### MNIST

| Method | Accuracy | Time |
|--------|----------|------|
| Bytepacked Bayesian (hot map) | 91.83% | 1.2 μs |
| IG vote-only (no probe) | 85.64% | 2.1 sec |
| Multi-probe IG vote-only | 88.04% | — |
| **Bytepacked cascade (K=50, k=3)** | **96.28%** | **9.3 sec** |
| 3-channel cascade (reference) | 96.12% | 32.9 sec |

**+0.16pp over 3-channel cascade, 3.5× faster.**

Top confused pairs: 4↔9 (47), 3↔5 (41), 3↔8 (28), 1↔7 (28), 7↔9 (25).

### Fashion-MNIST

| Method | Accuracy | Time |
|--------|----------|------|
| Bytepacked Bayesian (hot map) | 76.52% | 2.1 μs |
| IG vote-only (no probe) | 79.27% | 3.7 sec |
| Multi-probe IG vote-only | 79.65% | — |
| **Bytepacked cascade (K=50, k=7)** | **82.89%** | **14.9 sec** |

Top confused pairs: 0↔6 shirt/T-shirt (277), 2↔6 pullover/shirt (258),
2↔4 pullover/coat (218). Class 6 (shirt) is the dominant failure mode.

## Key Observations

### MNIST: marginal win, dramatic speedup
The +0.16pp accuracy gain confirms the hypothesis but it's small.
The much larger win is **speed**: 9.3s vs 32.9s. The inverted index
for 1 bytepacked channel (21.4 MB, ~5.6M entries) is dramatically smaller
than 3 ternary channels. Fewer vote accumulations per query → faster.

The K parameter has no effect (K=50 through K=1000 give identical results).
This means the top-50 candidates by vote already contain all meaningful
neighbors — the bytepacked encoding concentrates votes on the right images.

### Fashion-MNIST: 82.89% — a new ceiling for this architecture
The 3-channel cascade result on Fashion-MNIST was unknown. The bytepacked
cascade gives 82.89% with k=7, suggesting the true 3-channel cascade
would be in the 83-85% range on Fashion. The dominant confusion is
shirt/T-shirt/pullover (classes 0, 2, 6) which have overlapping structure
at ternary resolution.

### Vote density is concentrated
K=50 and K=1000 give identical accuracy on both datasets. This means the
bytepacked representation concentrates votes so tightly that the top-50
candidates are already the right neighborhood. The cascade's dot-product
refinement step picks the correct k=3 from that set regardless of K.

## Files

- `sstt_bytecascade.c`: Self-contained implementation
- `Makefile`: `make sstt_bytecascade`
