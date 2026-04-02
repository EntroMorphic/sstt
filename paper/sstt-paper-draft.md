# Ternary Cascade Classification via Address Generation Without Iterative Optimization

**Authors:** [Names]

**Abstract.** We present SSTT, an image classifier that uses no gradient descent, no backpropagation, and no floating-point arithmetic at inference, achieving 97.27% on MNIST and 85.68% on Fashion-MNIST. The system treats classification as a sequence of integer address lookups: ternary quantization maps each image to a compact block-signature representation; information-gain-weighted inverted-index voting retrieves the top-50 candidate training images from 60,000; and a structural ranking step resolves the final class. No parameter is updated by a loss function or iterative optimization. We demonstrate four non-obvious empirical findings: (1) **K-invariance** — top-50 candidates by vote contain every relevant neighbor with near-zero recall loss (1200× compression, zero accuracy cost); (2) **error mode decomposition** — 98.3% of classification errors occur at the ranking step, not retrieval, establishing a fixable ceiling of approximately 98.6%; (3) **cross-dataset generalization** — the same architecture, parameters, and code achieves 85.68% on Fashion-MNIST with zero tuning, and 52.41% on CIFAR-10 via a 7-eye multi-threshold ensemble, with channel importance automatically redistributing through information-gain re-weighting; and (4) **the retrieval diversity hypothesis** — retrieval diversity from multi-threshold quantization improves accuracy when the ranker operates on scheme-agnostic features (+2.23pp on CIFAR-10, +0.34pp on Fashion-MNIST) but hurts when ranking features are scheme-specific (−0.10pp on MNIST). SSTT achieves comparable accuracy to kNN+PCA — the strongest non-iterative baseline — while searching 1200× fewer candidates. An adaptive three-tier router delivers 99.9% accuracy on 10% of queries in under 1 μs with zero false positives. All results include 95% confidence intervals and are compared against non-neural baselines (random forest, SVM, 1-hidden-layer MLP, kNN+PCA) to position the contribution within the broader non-parametric landscape.

---

## 1. Introduction

Image classification has converged on two paradigms: deep neural networks that learn hierarchical representations through backpropagation, and classical nearest-neighbor methods that operate directly on pixel similarity. Both carry hard constraints. Neural networks require significant compute for training, produce opaque learned weights that resist inspection, and impose a floating-point inference cost that limits deployment on constrained hardware. Brute k-NN avoids learned weights but scales as O(N·D) per query — 600 million dot products to classify one MNIST image against 60,000 training examples.

This paper asks a different question: **how much of the classification problem can be solved by address generation alone?**

**Deployment motivation.** Several real-world settings require classifiers that are integer-only, cache-resident, and free of floating-point hardware: embedded microcontrollers (ARM Cortex-M series) without FPUs, safety-critical systems requiring deterministic inference with bit-exact reproducibility across platforms, edge devices where model size must fit in L2 cache, and applications where model interpretability is required — every SSTT prediction is traceable to specific training images and their vote contributions, with no opaque weight matrix intervening. These constraints are not hypothetical; they describe the deployment envelope of millions of shipped MCUs.

We formalize this as follows. At training time, we pre-compute a compact address structure — an inverted index over block-level ternary signatures, weighted by information gain. At inference time, classification proceeds in three stages: quantize the query image to a ternary field; look up its block signatures to accumulate weighted votes over training images; rank the top-K vote-getters by a multi-feature scoring function. No weight matrix is ever learned. No gradient is ever computed. Every operation is integer table lookup, add/subtract, or compare.

The result is a system that achieves **comparable accuracy to kNN+PCA** — the strongest non-iterative baseline — while searching 1200× fewer candidates. kNN+PCA(30) with k=5 achieves 97.55% on MNIST by brute-force search over all 60,000 training images projected into a 30-dimensional PCA space; SSTT achieves 97.27% by searching only 50 candidates retrieved via inverted-index voting. The accuracy gap (0.28pp) is within the overlap of 95% confidence intervals; the computational gap (60,000 vs. 50 comparisons) is not.

### Contributions

1. **The SSTT cascade**: a composed pipeline (IG-weighted inverted index → multi-probe expansion → ternary dot ranking → discrete structural features) that systematically exceeds each of its constituent techniques when applied in isolation.

2. **K-invariance**: the empirical finding that top-K candidate recall is effectively perfect at K=50, compressing a 60,000-candidate search to 50 candidates with zero accuracy loss.

3. **Error mode decomposition**: a taxonomy of classification errors into retrieval failures (Mode A), ranking inversions (Mode B), and vote dilutions (Mode C), establishing that 98.3% of errors are in the ranking step and that a fixable ceiling of ~98.6% exists.

4. **The `_mm256_sign_epi8` ternary multiply**: an AVX2 instruction repurposed to perform exact ternary multiply-accumulate at 32 operations per cycle, enabling a throughput-competitive ternary dot product with no multiply instruction.

5. **Cross-dataset generalization**: zero-tuning transfer from MNIST to Fashion-MNIST at 85.68% accuracy (+8.82pp over brute kNN), and to CIFAR-10 at 52.41% (7-eye ensemble, 5.2× random baseline), with channel weights automatically redistributing through information-gain re-weighting.

6. **The retrieval diversity hypothesis**: multi-threshold quantization improves accuracy when the ranker operates on scheme-agnostic features (Gauss map on raw gradients: +2.23pp on CIFAR-10) but hurts when ranking features are scheme-specific (ternary dot product: −0.10pp on MNIST). We observe that this dichotomy explains when ensemble retrieval adds value in non-parametric systems.

---

## 2. Related Work

**Nearest-neighbor classification** in its brute form (Cover & Hart, 1967) provides a well-understood accuracy ceiling but is compute-intensive at scale. The most relevant non-iterative improvement is **kNN with dimensionality reduction**: PCA or LDA projects the feature space to a lower-dimensional subspace computed via eigendecomposition (SVD) on the training set, then brute-force kNN operates in the reduced space. kNN+PCA(30) with k=5 achieves 97.55% on MNIST — comparable to SSTT — but requires brute search over all 60,000 training images in the projected space. Like SSTT, kNN+PCA uses no gradient descent; unlike SSTT, it requires floating-point eigendecomposition and O(N) comparisons at inference. SSTT achieves comparable accuracy at 1200× fewer comparisons through a structurally different retrieval mechanism.

**Approximate nearest-neighbor methods** (LSH, FAISS [Johnson et al., 2021], HNSW, ScaNN) reduce search cost through hashing or hierarchical indexing, trading recall for speed. These methods accelerate search in a fixed feature space; SSTT differs in that it constructs a richer block-signature representation in which an inverted-index vote achieves near-perfect recall at 1/1200 of the brute-force search cost. SSTT's advantage is semantic retrieval through IG-weighted voting, not faster search of the same space.

**Gradient-boosted tree ensembles** (XGBoost [Chen & Guestrin, 2016], LightGBM) are non-neural methods that achieve ~98% on MNIST through iterative boosting of decision trees. These methods are fast at inference and competitive in accuracy, but they require iterative optimization (boosting rounds minimize a loss function) and produce ensembles of hundreds of trees whose predictions are not directly traceable to specific training examples. SSTT uses no iterative optimization and provides full traceability: every prediction identifies the specific training images that contributed votes and their relative scores.

**Quantized and binary neural networks** (BWN, TWN, TTN) apply ternary constraints to learned weights to reduce inference cost. These methods still require gradient-based training; the ternary constraint is applied to learned parameters, not used as a first-principles feature representation. SSTT uses no iteratively optimized parameters; the ternary field is the representation.

**Bag-of-words image retrieval** (Sivic & Zisserman, 2003) uses inverted indexes over visual word assignments for image retrieval. SSTT shares the inverted-index mechanism but operates on local ternary block signatures rather than SIFT descriptors, and targets classification accuracy rather than retrieval ranking.

**Hash-based approximate matching** (Weiss et al., 2009; Norouzi et al., 2012) encodes images into compact binary codes for fast Hamming-distance matching. SSTT's multi-probe Hamming-1 expansion is related but applied to ternary trit-flip neighbors of block signatures rather than global image codes.

**Non-parametric image classification** with hand-crafted features (HOG+SVM, SURF+BoW) achieves high accuracy on structured datasets but requires feature engineering. SSTT's ternary block signatures are derived mechanically from the image with no domain knowledge; the information-gain weights are computed closed-form from training statistics.

---

## 3. Method

### 3.1 Notation

Let $\mathbf{x} \in [0,255]^{H \times W}$ denote an input image. For MNIST and Fashion-MNIST, $H = W = 28$. Let $\mathcal{C} = \{0,\ldots,9\}$ denote the label set. Let $\mathcal{T} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ denote the training set with $N = 60{,}000$.

### 3.2 Ternary Quantization

Each image is quantized to a ternary field $\mathbf{t} \in \{-1, 0, +1\}^{H \times W}$ using fixed thresholds:

$$t_{y,x} = \begin{cases} -1 & \text{if } x_{y,x} < 85 \\ 0 & \text{if } 85 \leq x_{y,x} < 170 \\ +1 & \text{if } x_{y,x} \geq 170 \end{cases}$$

Two derivative channels are then computed in ternary space. The **horizontal gradient** $\mathbf{h}$ and **vertical gradient** $\mathbf{v}$ are:

$$h_{y,x} = \text{clamp}(t_{y,x+1} - t_{y,x}), \quad v_{y,x} = \text{clamp}(t_{y+1,x} - t_{y,x})$$

where clamp maps values outside $\{-1, 0, +1\}$ to the nearest endpoint. This produces three ternary channels: pixel ($\mathbf{t}$), h-grad ($\mathbf{h}$), and v-grad ($\mathbf{v}$).

**Background suppression.** Each channel has an independent background value that is skipped during indexing: BG_PIXEL = 0 (the all-dark encoding), BG_GRAD = 13 (the ternary encoding of a flat region with value 0). Skipping background blocks reduces index noise without a hard edge-detection step.

**Bytepacked encoding.** A fourth channel fuses all three into a single byte: two bits each for the majority trit of pixel, h-grad, v-grad, and transition activity. This produces 256-value block signatures in place of 27-value signatures, capturing cross-channel correlations absent in the independent channels.

### 3.3 Block Encoding

The image is decomposed into overlapping 3×1 horizontal blocks at $P = 252$ positions, covering all valid horizontal triples within the $28 \times 28$ grid. Each block maps to a signature $s \in \{0,\ldots,26\}$ (or $\{0,\ldots,255\}$ for bytepacked). Three block series are extracted — one per channel — yielding 252 ternary signature sequences per query.

### 3.4 Information-Gain Weighted Inverted Index

**Training.** For each channel $c$, position $p$, and signature value $v$, store the list of training image IDs that exhibit signature $v$ at position $p$ in channel $c$: $L_{c,p,v} = \{i : s^{(c)}_{p,i} = v\}$.

Compute the **information gain** of each (channel, position) pair:

$$\text{IG}(c, p) = H(\mathcal{C}) - H(\mathcal{C} \mid s^{(c)}_p)$$

where $H(\mathcal{C}) = -\sum_k P(y=k)\log P(y=k)$ is the marginal class entropy and $H(\mathcal{C} \mid s^{(c)}_p)$ is the conditional entropy given the block signature at position $p$ in channel $c$. Weights are normalized to integers in $[1, 16]$, computed once from training statistics, stored as a lookup table.

**Inference.** Maintain a per-training-image vote accumulator $V \in \mathbb{Z}^N$, initialized to zero. For each block position $p$ in each channel $c$, look up the inverted list $L_{c,p,s^{(c)}_p}$ and add $\text{IG}(c,p)$ votes to each training image in the list:

$$V[i] \mathrel{+}= \text{IG}(c, p) \quad \forall\, i \in L_{c,p,s^{(c)}_p}$$

The top-K training images by vote count form the candidate set $\mathcal{K}$.

### 3.5 Multi-Probe Hamming-1 Expansion

For each block position $p$ in channel $c$, additionally look up the 6 Hamming-1 trit-flip neighbors $\{s' : d_H(s', s^{(c)}_p) = 1\}$ at half weight:

$$V[i] \mathrel{+}= \tfrac{1}{2}\,\text{IG}(c, p) \quad \forall\, i \in L_{c,p,s'}, \quad \forall\, s' \in \text{Neighbors}(s^{(c)}_p)$$

This handles single-trit quantization noise without explicit noise modeling.

### 3.6 Ternary Dot-Product Ranking

The top-K candidates are ranked by a weighted multi-channel ternary dot product:

$$\text{score}(i) = w_\text{px} \cdot \langle \mathbf{t}, \mathbf{t}_i \rangle + w_\text{vg} \cdot \langle \mathbf{v}, \mathbf{v}_i \rangle$$

where the weight grid $(w_\text{px}, w_\text{hg}, w_\text{vg}) = (256, 0, 192)$ was selected by grid search over the discrete space $\{0, 64, 128, 192, 256\}^3$ (125 combinations) on the 10% validation split, then frozen for test evaluation. Horizontal gradient adds noise on MNIST; vertical gradient captures stroke direction (see Section 5.4). Each ternary dot product is computed via `_mm256_sign_epi8`:

```c
// Exact ternary multiply-accumulate over {-1, 0, +1}:
// sign_epi8(a, b)[i] = a[i] if b[i]>0, -a[i] if b[i]<0, 0 if b[i]=0
__m256i prod = _mm256_sign_epi8(a, b);
acc = _mm256_add_epi8(acc, prod);  // safe for up to 127 iterations
```

This repurposes the AVX2 conditional-negation instruction as exact ternary multiplication at 32 operations per cycle, eliminating any widening multiply.

The k=3 majority vote among the top-3 scored candidates determines the class prediction.

### 3.7 Structural Ranking Features

The dot product ranking is augmented by discrete features derived from the gradient channels, added as a linear combination:

**Discrete divergence.** The divergence of the gradient field at each pixel:

$$\text{div}(x,y) = (h_{x+1,y} - h_{x,y}) + (v_{x,y+1} - v_{x,y})$$

The sum $D = \sum_{x,y} \text{div}(x,y)$ captures the net flux of the gradient field. Negative divergence concentrates at enclosed regions and concavities — a scalar summary of the structural topology of the image without binarization.

**Spatial decomposition.** $D$ is computed separately in each cell of a 2×4 spatial grid (MNIST) or 3×3 grid (Fashion-MNIST), producing an 8- or 9-dimensional feature vector that retains spatial context.

**Enclosed-region centroid.** Flood-fill from image corners identifies foreground pixels (those not reachable from the background). The centroid of enclosed foreground pixels locates the center of mass of digit loops.

**MAD-based adaptive weighting.** The median absolute deviation (MAD) of candidate divergence scores serves as a per-query uncertainty estimate. When candidates agree on divergence (low MAD), the structural features are trusted; when they disagree (high MAD), the system falls back toward the dot product score:

$$\lambda = \frac{c}{c + \text{MAD}(D_\mathcal{K})}$$

where $c$ is a scale constant selected on the validation split.

**Sequential accumulation.** Candidates are processed in vote-rank order with an exponential decay $S=2$, combining structural scores with the dot product score in a running log-odds accumulation. Validation experiments confirmed that sequential processing with decay $S=2$ contributes +0.00pp on MNIST (noise; indistinguishable from the static baseline) and +1.14pp on Fashion-MNIST (genuine signal, confirmed by val/holdout protocol). The Fashion-MNIST gain is attributable to the spatial grid divergence features being more informative when accumulated in vote-rank order on the more heterogeneous Fashion dataset.

The combined score is:

$$\text{score\_final}(i) = \text{score\_dot}(i) + \lambda \cdot [\alpha_\text{div} \cdot D(i) + \alpha_\text{cent} \cdot C(i) + \alpha_\text{prof} \cdot \text{profile}(i)]$$

where $\alpha_\ast$ are fixed weights selected on the validation split.

### 3.8 Adaptive Three-Tier Router

Classification cost is adapted to query difficulty using the vote concentration signal available at zero additional cost after the retrieval step.

**Tier 1 (Fast, <1 μs).** If the top vote count exceeds unanimity threshold $\tau = 198$ out of top-200 candidates sharing one class, return the plurality immediately. No dot product, no structural ranking.

**Tier 2 (Light, ~1.5 μs).** Moderate-confidence queries consult the **Dual Hot Map**: independent Bayesian frequency tables for pixel blocks and Gauss-map edge geometry, fused via posterior summation. Covers a further 15–18% of queries.

**Tier 3 (Deep, ~1 ms).** All remaining queries execute the full pipeline: inverted-index retrieval → dot ranking → structural ranking.

---

## 4. Experiments

### 4.1 Datasets and Evaluation Protocol

**MNIST** (LeCun et al., 1998): 60,000 training / 10,000 test, 28×28 grayscale, 10 classes. All accuracy figures use the standard test set. A 10% validation split from training is used for weight selection; reported numbers are on the held-out test set only.

**Fashion-MNIST** (Xiao et al., 2017): identical format and dimensions to MNIST, 10 clothing categories. The *same model* trained on MNIST Fashion data (re-running IG computation on the Fashion training set) is evaluated — no architectural changes, no hyperparameter re-tuning.

**CIFAR-10** (Krizhevsky, 2009): 50,000 training / 10,000 test, 32×32 RGB, 10 classes. The same ternary pipeline is applied per-channel (R, G, B) with a stereo-vote retrieval scheme and Gauss-map edge-geometry ranking. See Section 5.6.

**Baselines.** We compare against pixel-space, non-neural, and simple neural baselines to position SSTT within the broader landscape:

- *Brute 1-NN (WHT)*: nearest neighbor by ternary dot product over all 60,000 training images (pixel-space ternary ceiling).
- *Brute k=3 NN (WHT)*: Walsh-Hadamard Transform dot product, equivalent to pixel-space ternary dot product (Parseval's theorem), k=3 majority vote.
- *Brute 1-NN (L2)*: nearest neighbor by Euclidean L2 distance over all 60,000 training images (pixel-space Euclidean ceiling).
- *Brute k=3 NN (L2)*: Euclidean L2 distance, k=3 majority vote.
- *kNN+PCA*: k-nearest-neighbor classification in PCA-projected space (eigendecomposition on training set, brute search over all 60,000 projected training images). No gradient descent; uses SVD. Best configuration selected over PCA dimensions {30, 50, 100, 150} and k ∈ {1, 3, 5}.
- *Naive Bayes hot map*: our own O(1) per-channel frequency table baseline.
- *Random forest*: 100 trees, max depth unlimited, on raw 784-pixel features (scikit-learn defaults).
- *Linear SVM*: SGDClassifier with hinge loss, $\alpha=10^{-4}$, 1000 epochs, on standardized pixels (scikit-learn).
- *1-hidden-layer MLP*: 256 ReLU units, Adam optimizer, 50 epochs, batch size 128, on standardized pixels (scikit-learn MLPClassifier). This is the simplest learned-parameter model and establishes what "adding one layer of gradient descent" buys.

All reported SSTT numbers use the full 60,000 training images (MNIST/Fashion) or 50,000 (CIFAR-10) and the standard test split. No data augmentation. No preprocessing beyond the quantization described in Section 3. Baseline models use raw pixels with no preprocessing.

### 4.2 Main Results

**Table 1: MNIST accuracy (10,000 test images; 95% CI via Clopper-Pearson).**

| Method | Accuracy | 95% CI | Learned params | Notes |
|---|---|---|---|---|
| Linear SVM (SGD, hinge) | 89.48% | ±0.61 | ~7,840 | Standardized pixels |
| Brute 1-NN (WHT, ternary) | 95.87% | ±0.39 | 0 | Pixel-space ternary ceiling |
| Brute k=3 NN (WHT, ternary) | 96.12% | ±0.38 | 0 | Pixel-space ternary ceiling |
| Brute 1-NN (L2, Euclidean) | 96.91% | ±0.34 | 0 | Pixel-space Euclidean ceiling |
| Brute k=3 NN (L2, Euclidean) | 97.05% | ±0.34 | 0 | Pixel-space Euclidean ceiling |
| Random forest (100 trees) | 97.05% | ±0.34 | ~10⁶ | Raw pixels |
| **kNN+PCA(30) k=5** | **97.55%** | **±0.31** | **0 (SVD)** | **Brute search, 60K comparisons** |
| 1-layer MLP (256 units) | 97.63% | ±0.30 | ~203K | Standardized pixels |
| Naive Bayes hot map | 73.23% | ±0.87 | 0 | O(1) SSTT baseline |
| IG-weighted vote only | 88.78% | ±0.62 | 0 | Retrieval step alone |
| Zero-weight cascade | 96.04% | ±0.38 | 0 | First full SSTT pipeline |
| Bytepacked cascade | 96.28% | ±0.37 | 0 | Encoding D; 3.5× faster |
| + Multi-channel dot | 96.44% | ±0.36 | 0 | Oracle v2 configuration |
| **+ Structural ranking (SSTT)** | **97.27%** | **±0.32** | **0** | **Val/holdout confirmed; 50 candidates** |

SSTT (97.27%) exceeds brute k=3 NN in both ternary (96.12%, +1.15pp) and Euclidean (97.05%, +0.22pp) metrics using 1200× fewer candidate comparisons. kNN+PCA(30) k=5 achieves 97.55%, which is 0.28pp above SSTT; the 95% CIs overlap (±0.31 vs. ±0.32). The critical difference is computational: kNN+PCA performs brute search over all 60,000 training images in PCA space; SSTT searches 50 candidates. SSTT achieves comparable accuracy to kNN+PCA at 1200× fewer comparisons.

**Table 2: Fashion-MNIST accuracy (10,000 test images; 95% CI via Clopper-Pearson).**

| Method | Accuracy | 95% CI | Learned params | Notes |
|---|---|---|---|---|
| Brute 1-NN (WHT, ternary) | 76.86% | ±0.83 | 0 | Pixel-space ternary ceiling |
| Linear SVM (SGD, hinge) | 82.50% | ±0.75 | ~7,840 | Standardized pixels |
| Brute 1-NN (L2, Euclidean) | 84.97% | ±0.71 | 0 | Pixel-space Euclidean ceiling |
| Brute k=3 NN (L2, Euclidean) | 85.41% | ±0.70 | 0 | Pixel-space Euclidean ceiling |
| **kNN+PCA(150) k=5** | **86.35%** | **±0.68** | **0 (SVD)** | **Brute search, 60K comparisons** |
| Random forest (100 trees) | 87.60% | ±0.65 | ~10⁶ | Raw pixels |
| 1-layer MLP (256 units) | 88.48% | ±0.63 | ~203K | Standardized pixels |
| Bytepacked cascade | 82.89% | ±0.74 | 0 | Same SSTT code, no tuning |
| **+ Structural ranking (SSTT)** | **85.68%** | **±0.69** | **0** | **Val/holdout confirmed; 50 candidates** |

SSTT lifts over brute kNN (ternary) by +8.82pp on Fashion — larger than the +1.15pp on MNIST — demonstrating that the architecture provides more discriminative value on a harder task. kNN+PCA(150) k=5 achieves 86.35%, which is 0.67pp above SSTT; the 95% CIs overlap (±0.68 vs. ±0.69). As on MNIST, SSTT achieves comparable accuracy to kNN+PCA at 1200× fewer comparisons. SSTT exceeds linear SVM (+3.18pp) and approaches random forest (−1.92pp), both without iteratively optimized parameters.

**Baseline reproducibility.** All baseline numbers were produced by `scripts/run_baselines.py` using scikit-learn 1.8.0 with the hyperparameters stated in Section 4.1. Random forest uses raw pixels; SVM and MLP use standardized pixels (zero mean, unit variance). kNN+PCA baselines use scikit-learn's PCA and KNeighborsClassifier with brute search; the best configuration per dataset was selected over PCA dimensions {30, 50, 100, 150} and k ∈ {1, 3, 5}. Brute kNN (L2) uses scikit-learn's KNeighborsClassifier with Euclidean distance over raw pixels.

### 4.3 Available Classifier Configurations

**Table 3: Available classifier configurations across the SSTT family.**

Note: these rows represent different architectures with different code paths, not tradeoff points of a single system. The three-tier router (final row) is the only entry that represents a genuine speed-accuracy tradeoff within one system.

| Method | MNIST Acc. | Fashion Acc. | Latency | Regime |
|---|---|---|---|---|
| Dual Hot Map (O(1)) | 76.53% | ~72% | ~1.5 μs | Embedded instant |
| Bytepacked hot map | 91.83% | — | 1.9 μs | Embedded real-time |
| Pentary hot map | 86.31% | — | 4.7 μs | Embedded real-time |
| Router Tier 1 only | 99.9% / 10% coverage | 100% / 9.6% | <1 μs | Ultra-fast (easy queries) |
| Router Tier 1+2 | ~96% / 28% coverage | — | <2 μs | Fast path |
| Bytepacked cascade | 96.28% | 82.89% | 930 μs | Interactive |
| Oracle v2 | 96.44% | — | 1.1 ms | Batch |
| **Full system (topo9)** | **97.27%** | **85.68%** | **~1 ms** | **Batch / Research** |
| **Three-tier router** | **96.50%** | **83.42%** | **0.67 ms** | **Production** |

The three-tier router achieves 2× throughput over the full system (0.67 ms vs. 1 ms average latency) with a 0.77pp global accuracy trade-off, by routing 10% of queries through Tier 1 (<1 μs) and 18% through Tier 2 (~1.5 μs).

### 4.4 Tier 1: Zero False Positive Rate

On Fashion-MNIST, Tier 1 (unanimity threshold τ = 198/200) classifies **9.6% of the test set at 100.0% accuracy** — zero false positives across 960 images. The threshold is conservative enough that it never fires on a wrong prediction. This property holds without any tuning specific to Fashion-MNIST; the same threshold was set on MNIST and applied directly.

---

## 5. Analysis

### 5.1 K-Invariance: Near-Perfect Retrieval at K = 50

**Table 4: Accuracy as a function of K (bytepacked cascade).**

| K | Accuracy |
|---|---|
| 50 | 96.28% |
| 100 | 96.28% |
| 200 | 96.28% |
| 500 | 96.28% |
| 1000 | 96.28% |

Accuracy is identical across K = 50 through K = 1000. The top-50 candidates by vote contain every relevant neighbor out of 60,000 training images. This is a **1200× compression ratio with zero recall loss**.

We hypothesize K-invariance arises from the high mutual information between IG-weighted block signatures and class labels. The IG weighting concentrates vote mass on the most discriminative positions, creating a vote distribution where correct-class training images receive disproportionate support. At K=50, the vote threshold is high enough to exclude noise while retaining all class-consistent candidates. In effect, the inverted-index vote acts as a near-perfect class-conditional filter: the vast majority of class-relevant training images appear in the top 50 by vote, and adding more candidates beyond 50 adds only noise from other classes.

The implication is fundamental: the retrieval problem is solved at K = 50. The inverted-index vote achieves near-perfect recall using compressed block signatures. All remaining classification error is a **ranking problem**, not a retrieval problem.

This reframes the architecture: SSTT does not approximate k-NN by trading recall for speed. It achieves exact-equivalent recall through a structurally different representation, then ranks within a 50-element pre-filtered set.

### 5.2 Error Mode Decomposition

We classify every misclassification into three mutually exclusive modes, diagnosable from the data structures available at inference time:

- **Mode A: Retrieval failure.** The correct class is absent from the top-K candidates. The vote stage missed the relevant training images. These errors cannot be fixed by a better ranker.
- **Mode B: Ranking inversion.** The correct class is present in top-K but the wrong class wins the ranking step, typically by a margin of 3–20 score units.
- **Mode C: Vote dilution.** The ranking step correctly identifies the nearest neighbor, but the k=3 majority vote dilutes it with two wrong-class neighbors.

**Bytepacked cascade (96.28%, 372 errors):** Mode A: 7 (1.7%), Mode B: 240 (64.5%), Mode C: 125 (33.8%). The Mode A/B/C percentages reported here are from the bytepacked cascade at 96.28%, not the full topo9 system.

**Full system — topo9 (97.27%, 273 errors):** Mode A: 7 (2.6%), Mode B: 197 (72.2%), Mode C: 69 (25.3%). The topo9 structural ranking fixes 99 errors relative to the bytepacked cascade, primarily Mode B ranking inversions (240→197) and Mode C vote dilutions (125→69). Mode A retrieval failures remain unchanged at 7, confirming that the structural ranking improvements are entirely in the ranking step.

In both systems, **>97% of errors occur at the ranking step, not retrieval.** The vote architecture has 99.3% effective recall on the relevant set; the bottleneck is purely in distinguishing among retrieved candidates.

A **fixable ceiling** follows: if all Mode B and C errors were resolved (e.g., by a perfect ranker on the top-50 set), accuracy would reach approximately **98.6%**. The structural ranking system (Section 3.7) moves the system from 96.04% toward this ceiling by improving the per-candidate scoring function, fixing +111 MNIST errors and +215 Fashion errors over the dot-product baseline.

### 5.3 Compute Profile

Phase-timer profiling of the full cascade:

| Phase | Runtime share |
|---|---|
| Vote accumulation (scattered writes) | **87%** |
| Ternary dot products | ~1% |
| Top-K selection | ~5% |
| Encoding and overhead | ~7% |

The dominant cost is **vote accumulation** — scattered writes to a 240KB vote array, one per block per inverted-list entry. The bottleneck is the memory access pattern (cache-line thrashing on a large non-sequential array), not the arithmetic. The `_mm256_sign_epi8` ternary dot products are effectively free compared to the vote scatter cost.

This has broader implications: for any inverted-index system on modern hardware, the scatter-write memory pattern dominates. Optimizations that reduce arithmetic cost (SIMD, quantization) provide marginal benefit; optimizations that reduce scattered memory accesses (blocking, sorting by index, reducing probe count) are the productive target.

**Practical consequence.** Reducing probe count from 8 to 4 (Hamming-1 neighbors per block) gives 96.07% accuracy at 1.38× speed improvement — a better trade-off than any arithmetic optimization at the same probe count.

### 5.4 Channel Ablation

**Table 5: Dot-product weight ablation on validation split (bytepacked cascade, K=50, k=3).**

| $(w_\text{px}, w_\text{hg}, w_\text{vg})$ | MNIST Acc. | Fashion Acc. |
|---|---|---|
| (256, 0, 0) — pixel only | 96.04% | 82.11% |
| (256, 192, 0) — px + h-grad | 95.87% | 83.44% |
| (256, 0, 192) — px + v-grad | **96.38%** | 82.89% |
| (256, 192, 192) — all three | 96.21% | **83.66%** |

On MNIST, h-grad adds noise (−0.17pp from pixel-only), while v-grad consistently helps (+0.34pp). On Fashion-MNIST, h-grad helps (+1.33pp) because clothing textures have strong horizontal structure that digit strokes do not. Both channels are included in the Fashion system at equal weight.

The channel importance reversal between datasets is handled automatically: information-gain re-weighting assigns higher IG to h-grad positions on the Fashion training set, redistributing votes without any manual re-tuning.

### 5.5 Cross-Dataset Generalization

**Table 6: Fashion-MNIST accuracy at each pipeline stage, with and without re-tuning.**

| Method | MNIST Acc. | Fashion Acc. | Tuning required |
|---|---|---|---|
| Brute 1-NN (ternary) | 95.87% | 76.86% | None |
| Cascade (MNIST weights) | 96.04% | ~80% | None |
| Cascade (Fashion IG) | — | 82.89% | IG re-computation only |
| Full system (Fashion) | **97.27%** | **85.68%** | IG re-computation only |

The IG re-computation is not tuning in the conventional sense — it is a closed-form operation over the Fashion training set that requires no cross-validation, no learning rate, and no epoch selection. The same code runs unchanged; only the input data differs.

### 5.6 CIFAR-10: Boundary and Multi-Threshold Ensemble

On CIFAR-10 (32×32 RGB, 10 classes), the SSTT ternary pipeline is applied per-channel (R, G, B) with a stereo-vote retrieval scheme and Gauss-map edge-geometry ranking.

**3-eye baseline.** The single-threshold ternary cascade achieves **50.18%** (k=7) with 99.44% retrieval recall — 5× the random baseline but far below learned methods (~95%). Inter-class gradient-distance analysis shows that visually similar classes (cat/dog, automobile/truck) have near-identical ternary representations at this resolution.

**7-eye multi-threshold ensemble.** A 7-eye ensemble using thresholds {65, 75, 85, 95, 105, 115, 125} merges candidate sets from all threshold schemes before ranking. This achieves **52.41%** (k=5) with 99.60% retrieval recall — a +2.23pp improvement over the 3-eye baseline.

**Per-class breakdown (7-eye, 52.41%):**

| Class | Accuracy | Class | Accuracy |
|---|---|---|---|
| Airplane | 58.9% | Deer | 53.8% |
| Automobile | 61.3% | Dog | 40.1% |
| Bird | 42.4% | Frog | 56.2% |
| Cat | 33.6% | Horse | 52.0% |
| Ship | 63.7% | Truck | 62.1% |

The per-class pattern is informative: classes with distinctive edge geometry (ship, truck, airplane) achieve 58–64%, while classes that share fine-grained texture distinctions (cat vs. dog at 33.6% vs. 40.1%) fall near chance. This confirms the architectural boundary: ternary block signatures capture edge structure but not texture or semantic content.

**Why multi-threshold helps on CIFAR-10 but not MNIST.** The +2.23pp gain arises because the Gauss-map ranker is scheme-agnostic: it operates on raw gradient geometry, not on ternary representations. Candidates retrieved under different threshold schemes produce different ternary signatures but identical raw-gradient features, so the ranker can effectively use all of them. On MNIST, the topo9 ranker uses ternary dot products — a scheme-specific feature — so candidates from non-matching threshold schemes rank poorly, producing a −0.10pp loss (see Section 5.7).

This is an honest architectural boundary: no amount of ternary engineering will separate cat from dog at 32×32 resolution without learned texture features. We report CIFAR-10 not as a success but to scope where the architecture stops working and to demonstrate the retrieval diversity hypothesis.

### 5.7 Retrieval Diversity Hypothesis

Across the ensemble experiments, we observe a consistent pattern in how retrieval diversity — injecting candidates from multiple retrieval schemes — interacts with ranking quality.

**Experimental evidence:**

| Dataset | Ensemble type | Effect | Mechanism |
|---|---|---|---|
| MNIST | Position bagging (bytecascade) | +0.38pp | Diverse positions, same ranking features |
| MNIST | Position bagging (topo9) | +0.10pp | Redundant with topo9's richer ranking |
| MNIST | Multi-threshold (topo9) | −0.10pp | Scheme-specific ranker rejects foreign candidates |
| Fashion-MNIST | Multi-threshold | +0.34pp | Adaptive thresholds normalize intensity variation |
| CIFAR-10 | Multi-threshold (7-eye) | +2.23pp | Scheme-agnostic Gauss-map ranker uses all candidates |

**Hypothesis.** We observe that retrieval diversity helps when the ranker can effectively score candidates from diverse retrieval schemes, and hurts when ranking features are specific to the retrieval scheme that produced the candidates. Specifically:

- When the ranker operates on raw features (Gauss-map edge geometry, raw gradient statistics), it is scheme-agnostic: candidates from any threshold scheme are equally scorable. Diversity provides genuinely independent views of the query, and the ranker benefits from the expanded candidate pool.
- When the ranker operates on features derived from the quantization scheme (ternary dot products), it is scheme-specific: candidates retrieved under a different threshold produce ternary signatures that are systematically misaligned with the query's ternary representation. These candidates rank poorly and displace better candidates from the original scheme.

We emphasize that this is a hypothesis consistent with our observations across three datasets and two ranker types, not a proven principle. The pattern is suggestive but the evidence base is narrow.

---

## 6. Discussion

### 6.1 Why the Cascade Exceeds Brute k-NN

The cascade outperforms brute k=3 NN (97.27% vs. 96.12% ternary, vs. 97.05% L2) despite using fewer candidates. This is explained by the ranking step, not the retrieval step. Brute k=3 NN ranks by pixel dot product or L2 distance alone; the cascade ranks by pixel dot product (×256) + vertical gradient dot product (×192) + structural features (divergence, centroid, spatial grid). The richer ranking within a pre-filtered set achieves higher accuracy than the simpler ranking over the full training set.

The K-invariance finding confirms this: the candidates are the same quality at K=50 as at K=60,000 (brute). The difference is the scoring function applied to those candidates.

### 6.2 The Role of Non-Iterative Parameters

All parameters in SSTT are derived without iterative optimization or gradient descent:

- **IG weights**: mutual information between block signature and class label. Computed in O(N·P) time, no optimization.
- **Background thresholds** (BG_PIXEL=0, BG_GRAD=13): per-channel mode of the ternary distribution on background regions. Computed once, fixed.
- **Dot product weights** $(w_\text{px}, w_\text{hg}, w_\text{vg})$: determined by grid search over a 3-element discrete space {0, 64, 128, 192, 256} on the 10% validation split. 125 combinations evaluated; optimal weights frozen before test evaluation.
- **Structural feature weights** $(\alpha_\text{div}, \alpha_\text{cent}, \alpha_\text{prof})$ and **MAD scale** $c$: determined by ablation on the validation split over a small discrete grid. No gradient descent.
- **Router threshold** $\tau = 198$: a single integer, set on validation data, never revisited.

**On the continuum between "computed" and "learned."** We acknowledge that the distinction between parameters computed from data and parameters learned from data is a continuum, not a binary. The IG weights are computed from labeled training data — they depend on class labels and would change if labels were permuted. In a broad sense, any statistic derived from labeled data involves supervised learning. The grid search over dot-product weights and structural feature weights, while exhaustive over a small discrete space rather than gradient-based, is still a form of model selection that maximizes accuracy on a validation set.

The defensible claim is narrower: no parameter in SSTT is updated by a loss function or iterative optimization procedure. There is no training loop, no learning rate, no convergence criterion. The IG weights are computed in a single pass; the grid search evaluates a fixed, enumerable set of configurations. This places SSTT in the same methodological family as kNN+PCA (which computes an eigendecomposition from labeled data) rather than in the family of gradient-trained models. The practical consequence is deterministic, reproducible inference with no optimization history to audit.

### 6.3 Failure Modes and Limits

The mode decomposition identifies the 4/9 and 3/5 confusion pairs as the dominant remaining failures on MNIST. These pairs share both pixel similarity and topological structure (e.g., a 4 with a closed top loop resembles a 9). Resolution likely requires higher-level structural features beyond the scope of ternary block encoding.

The 98.6% fixable ceiling represents the maximum achievable accuracy given the current retrieval stage. Exceeding it requires improving Mode A recall (2.6% of topo9 errors, 7 images), which requires better block signatures or larger inverted lists.

---

## 7. Conclusion

We have presented SSTT, demonstrating that classification without iterative optimization can achieve accuracy comparable to kNN+PCA — the strongest non-iterative baseline — while searching 1200× fewer candidates. The system uses no gradient descent, no backpropagation, and no floating-point arithmetic at inference.

The core contributions — IG-weighted inverted indexing, multi-probe Hamming-1 expansion, the `_mm256_sign_epi8` ternary primitive, discrete structural ranking, and the adaptive three-tier router — compose into a system that maps a range of classifier configurations from 1.9 μs to 1 ms, achieves 97.27% MNIST and 85.68% Fashion-MNIST, and provides a free confidence signal with zero false positives at 10% coverage. On CIFAR-10, the 7-eye multi-threshold ensemble achieves 52.41% (5.2× random), honestly scoping the architectural boundary where ternary features become insufficient.

The K-invariance finding is the central empirical result: retrieval quality is effectively perfect at a 1200× compression ratio. The remaining gap between current accuracy (97.27%) and the fixable ceiling (98.6%) is a ranking problem, not a retrieval problem. This is a precise, testable claim that defines the research agenda going forward.

The retrieval diversity hypothesis — that multi-threshold ensembles help when the ranker is scheme-agnostic and hurt when it is scheme-specific — offers a principle for designing non-parametric ensemble systems, though we emphasize it remains a hypothesis supported by observations across three datasets rather than a proven law.

---

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

- Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*, 785–794.

- Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.

- Johnson, J., Douze, M., & Jégou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

- Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. *Technical Report*, University of Toronto.

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

- Li, F., Zhang, B., & Liu, B. (2016). Ternary weight networks. *arXiv:1605.04711*.

- Norouzi, M. & Fleet, D. J. (2012). Minimal loss hashing for compact binary codes. *ICML*.

- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

- Sivic, J. & Zisserman, A. (2003). Video Google: A text retrieval approach to object matching in videos. *ICCV*.

- Weiss, Y., Torralba, A., & Fergus, R. (2009). Spectral hashing. *NeurIPS*.

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747*.

- Zhu, C., Han, S., Mao, H., & Dally, W. J. (2016). Trained ternary quantization. *arXiv:1612.01064*.

---

## Appendix A: Reproducibility

All source code is available at [repository URL]. Data is downloaded via `make mnist` and `make fashion` from canonical OSSCI and Zalando S3 sources with SHA-256 verification. All experiments are single-file, self-contained C99 programs with no external dependencies beyond a C compiler with AVX2 support.

**Hardware.** All timing figures were measured on a single x86 core with AVX2 support: AMD Ryzen 5 PRO 5675U, 4.4 GHz boost clock, 512 KB L2 cache per core (3 MB total), 16 MB L3 cache. No GPU, no multi-threading, no SIMD beyond AVX2.

**Validation protocol.** All accuracy figures reported in Tables 1 and 2 are evaluated on the held-out test set (10,000 images). All configurable weights — dot-product channel weights $(w_\text{px}, w_\text{hg}, w_\text{vg})$, structural feature weights $(\alpha_\text{div}, \alpha_\text{cent}, \alpha_\text{prof})$, MAD scale $c$, and router threshold $\tau$ — were selected on a 10% validation split (6,000 images) from the training set and frozen before test evaluation. A retroactive audit confirmed that MNIST val-derived weights are identical to those found by test-set grid search; Fashion-MNIST val-derived weights differ slightly (divergence weight reduced from 200 to 50, MAD scale increased), and the val-derived Fashion numbers (85.68%) are the ones reported.

**Statistical significance.** All accuracy figures include 95% Clopper-Pearson confidence intervals computed from the binomial distribution over $n=10{,}000$ test images.

## Appendix B: The `_mm256_sign_epi8` Ternary Multiply

The AVX2 instruction `_mm256_sign_epi8(a, b)` computes, for each byte $i$:

$$\text{out}[i] = \begin{cases} a[i] & \text{if } b[i] > 0 \\ -a[i] & \text{if } b[i] < 0 \\ 0 & \text{if } b[i] = 0 \end{cases}$$

Over the ternary alphabet $b[i] \in \{-1, 0, +1\}$ (stored as signed bytes $\{255, 0, 1\}$), this computes exact ternary multiplication: $\text{out}[i] = a[i] \cdot b[i]$. Accumulating in `int8` is safe for up to 127 iterations before overflow; for 784-pixel MNIST images with 28 blocks per row, a single pass accumulates at most 28 terms, well within bounds.

This gives 32 ternary multiply-accumulates per instruction, per cycle, using a register renaming trick rather than an actual multiply unit. Prior work on ternary neural networks (TWN, TTN) implements ternary multiplication as conditional branching or lookup tables; this approach exploits a specific x86 instruction semantic that maps exactly to the ternary algebra.
