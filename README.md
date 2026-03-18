# SSTT — Zero-Parameter Ternary Cascade Classifier

[![Build](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/s2t2/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](CHANGELOG.md)

**Zero learned parameters.** No gradient descent, no backpropagation, no
floating point at inference. Classification is integer table lookups, AVX2
byte operations, and add/subtract. The entire model fits in L2 cache.

### **Results Summary**

*   **MNIST: 97.27%** (Static baseline, sequential is noise)
*   **Fashion-MNIST: 85.68%** (+8.50pp over brute kNN)
*   **CIFAR-10: 50.18%** (5× random, texture retrieval → shape ranking)

The system also knows when it will fail: a 264-entry confidence map predicts
difficulty before ranking, achieving **99.6% accuracy on 22% of images** with
zero additional computation.

---

## **Speed-Accuracy Pareto Frontier**

| Method | Accuracy (MNIST) | Latency | Regime |
|--------|------------------|---------|--------|
| Bytepacked Bayesian | 91.83% | 1.9 μs | Embedded Real-time |
| Pentary Hot Map | 86.31% | 4.7 μs | Embedded Real-time |
| Bytecascade (4-probe) | 96.07% | 610 μs | Interactive |
| Bytecascade (8-probe) | 96.28% | 930 μs | Interactive |
| **Tiered Router v1** | **96.50%** | **670 μs** | **Production Hybrid** |
| Field-Theoretic Ranking | 97.27% | ~1 ms | Batch / Research |
| Curvature-Gauss Cascade | 50.18% | ~3 ms | CIFAR-10 Best |

---

## **Project Audit**

### **Novel (The Breakthroughs)**
*   **`_mm256_sign_epi8` as Ternary Multiply:** A clever hardware-level hack using AVX2 conditional negation to perform 32 ternary multiply-accumulates per cycle without a multiply instruction.
*   **Discrete Green’s Theorem on Ternary Fields:** Applying differential-geometric operators (divergence) to quantized gradient fields to identify topological features like closed loops.
*   **The Confidence Map (Difficulty Oracle):** A "free" byproduct of the retrieval step that predicts classification difficulty before ranking.
*   **3-Eye Stereoscopic Retrieval:** Using multi-perspective voting to reach 99.4% recall on CIFAR-10, narrowing 50,000 images down to 200 candidates.

### **Useful (The Engineering Value)**
*   **Compute Profiling:** The discovery that **87% of execution time is memory access (scattered writes)**, not arithmetic. This generalizes to any inverted-index system on modern hardware.
*   **Negative Results Archive:** Honest documentation of failed experiments (PCA, soft-prior cascades, hard filtering) prevents future researchers from wasting time on the same dead ends.
*   **Ablation Series (topo1-9):** A textbook example of incremental methodology where each improvement is isolated and verified.

### **Understated (The Real "Headline" Claims)**
*   **Zero Learned Parameters:** No backpropagation, no gradient descent, and no floating point at inference. This is the project's most significant claim.
*   **Fashion-MNIST Dominance:** The system beats brute-force kNN by **+8.5pp** on Fashion-MNIST, proving the architecture adds value beyond pixel-space retrieval.
*   **K-Invariance:** The finding that the top 50 candidates contain every relevant neighbor. This **1200x compression** means retrieval is effectively solved.

---

## **Architecture**

```
Raw image (28x28 uint8)
    |
    v  AVX2 ternary quantization
{-1, 0, +1} per pixel (85/170 thresholds)
    |
    +---- Pixel blocks  (3x1 horizontal, 252 positions, 27 values)
    +---- H-grad blocks (same geometry, derivative channel)
    +---- V-grad blocks (same geometry, derivative channel)
    +---- Bytepacked   (all 3 channels -> 1 byte, 256 values)
              |
              v  Information-gain weighted multi-probe voting
         Inverted index lookup
         IG-weighted votes -> top-K candidates
              |
              v  Multi-channel ternary dot refinement
         score = 256 * dot(px) + 192 * dot(vg)
              |
              v  k=3 majority vote
         Class prediction
```

**Core insight:** quantize -> encode blocks -> vote over training images by
block match -> refine top-K with dot product. No matrix multiply. No
backpropagation. All computation is integer table lookup and AVX2 byte
operations.

## **Quick Start**

**Requirements:** GCC with AVX2 support, GNU Make, curl.

```bash
git clone https://github.com/anjaustin/s2t2.git
cd s2t2

# Download MNIST and Fashion-MNIST
make mnist
make fashion

# Build the main classifiers
make

# Run best single method on MNIST
./sstt_bytecascade

# Run on Fashion-MNIST
./sstt_bytecascade data-fashion/
```

## **Experiments**

Each experiment is a self-contained `.c` file. Build individually with `make sstt_<name>`.

| Binary | MNIST | Description |
|--------|-------|-------------|
| `sstt_bytecascade` | 96.28% | Bytepacked cascade — best single method |
| `sstt_topo9_val` | **97.27%** | Honest val/holdout validation: zero sequential benefit on MNIST |
| `sstt_oracle_v2` | 96.44% | Routed oracle: primary + pentary specialist |
| `sstt_diagnose` | — | Error autopsy: failure mode + ASCII visualiser |
| `sstt_bytepacked` | 91.83% | Bytepacked hot map (Bayesian series) |

## **License**

MIT — see [LICENSE](LICENSE).
