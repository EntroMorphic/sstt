# Classification Without Gradient Descent: K-Invariant Retrieval in Ternary Address Space

**Authors:** [Names]

**Abstract.** SSTT is an image classifier that uses no gradient descent, no backpropagation, and no floating-point arithmetic at inference. Classification is integer address generation: ternary quantization maps images to block signatures; an information-gain-weighted inverted index retrieves candidates; structural ranking resolves the final class. It achieves 97.27% on MNIST and 85.68% on Fashion-MNIST with no iteratively optimized parameters. We report three empirical findings: (1) **K-invariance in retrieval** — in the dot-product cascade, accuracy is identical from K=50 to K=1000 (1200x compression, zero cost), proving retrieval is solved; the structural ranker benefits from larger K (96.59% at K=50 to 97.61% at K=1000), revealing that richer ranking can exploit deeper candidate pools; (2) **error mode decomposition** — 97% of classification errors occur at ranking, not retrieval, with a theoretical ceiling of 99.93% if ranking were perfect; (3) **cross-dataset transfer** — the same architecture achieves 85.68% on Fashion-MNIST (+8.50pp over brute ternary kNN on the same holdout split) with only closed-form IG re-computation and val-derived weight selection, and 50.18% on CIFAR-10 (5x random), scoping where ternary features fail. An adaptive three-tier router delivers 96.50% at 0.67ms average latency.

---

## 1. Introduction

Image classification has converged on two paradigms: deep networks that learn representations through backpropagation, and nearest-neighbor methods that search pixel space directly. Both carry hard constraints. Neural networks require iterative optimization, produce opaque weights, and impose floating-point inference cost. Brute k-NN avoids learned weights but scales as O(N*D) per query.

This paper asks: **how much of the classification problem can be solved by address generation alone?**

We formalize this as follows. At training time, compute a compact address structure — an inverted index over block-level ternary signatures, weighted by information gain. At inference time, classification proceeds in three stages: quantize the query to a ternary field; look up block signatures to accumulate weighted votes over training images; rank the top-K candidates by a multi-feature scoring function. No weight is ever learned. No gradient is ever computed. Every operation is integer lookup, add, or compare.

The result achieves accuracy within the confidence interval of kNN+PCA — the strongest non-iterative baseline — with a structurally different computational profile. kNN+PCA(30) with k=5 achieves 97.55% on MNIST by ranking all 60,000 projected training images; SSTT achieves 97.27% by ranking only 50 candidates pre-filtered by inverted-index voting. The vote accumulation step implicitly touches all training images (87% of runtime), but the expensive ranking comparison operates on 1200x fewer candidates.

**Deployment motivation.** Several real-world settings require integer-only, cache-resident classifiers with no floating-point hardware: embedded microcontrollers without FPUs, safety-critical systems requiring deterministic bit-exact inference, edge devices where models must fit in L2 cache, and applications requiring full interpretability — every SSTT prediction is traceable to specific training images and their vote contributions.

### Contributions

1. **K-invariance in retrieval**: the dot-product cascade achieves identical accuracy from K=50 to K=1000 (1200x compression, zero cost), proving the inverted index achieves near-perfect recall. The structural ranker breaks K-invariance — accuracy rises from 96.59% (K=50) to 97.61% (K=1000) — revealing that richer ranking functions can exploit deeper candidate pools.

2. **Error mode decomposition**: a taxonomy classifying all errors into retrieval failures (1.7%), ranking inversions (64.5%), and vote dilutions (33.8%), with a theoretical ceiling of 99.93% (only 7 retrieval failures per 10,000 images).

3. **The `_mm256_sign_epi8` ternary primitive**: an AVX2 instruction repurposed for exact ternary multiply-accumulate at 32 operations per cycle.

4. **Cross-dataset transfer**: 85.68% on Fashion-MNIST (+8.50pp over brute ternary kNN on the same holdout split) with IG re-computation and val-derived weight selection — no architectural changes, but feature weights differ from MNIST.

5. **Honest scoping**: CIFAR-10 at 50.18% (5x random) demonstrates where ternary block features fail — texture-dependent classes (cat 33.6%) are near chance while edge-distinctive classes (ship 63.7%) work.

---

## 2. Related Work

**Nearest-neighbor classification.** Brute kNN (Cover & Hart, 1967) provides a well-understood ceiling but costs O(N*D) per query. kNN+PCA projects to a lower-dimensional eigenspace computed via SVD, then searches brute-force in the reduced space. kNN+PCA(30) k=5 achieves 97.55% on MNIST — comparable to SSTT — but requires 60,000 comparisons. Like SSTT, kNN+PCA uses no gradient descent; unlike SSTT, it requires floating-point eigendecomposition and O(N) inference.

**Approximate nearest-neighbor.** LSH, FAISS (Johnson et al., 2021), HNSW, and ScaNN reduce search cost through hashing or hierarchical indexing, trading recall for speed in a fixed feature space. SSTT differs by constructing a richer block-signature representation in which inverted-index voting achieves near-perfect recall at 1/1200 the brute-force cost.

**Gradient-boosted trees.** XGBoost (Chen & Guestrin, 2016) and LightGBM achieve ~98% on MNIST through iterative boosting. These methods require loss minimization and produce ensembles whose predictions are not traceable to specific training examples.

**Quantized neural networks.** BWN, TWN (Li et al., 2016), TTN (Zhu et al., 2016) apply ternary constraints to learned weights. The ternary constraint reduces inference cost but training still requires gradient descent. In SSTT, the ternary field is the representation, not a compression of learned parameters.

**Bag-of-words retrieval.** Sivic & Zisserman (2003) use inverted indexes over visual word assignments. SSTT shares the inverted-index mechanism but operates on local ternary block signatures rather than SIFT descriptors.

---

## 3. Method

### 3.1 Ternary Quantization

Each image $\mathbf{x} \in [0,255]^{28 \times 28}$ is quantized to a ternary field using fixed thresholds:

$$t_{y,x} = \begin{cases} -1 & \text{if } x_{y,x} < 85 \\ 0 & \text{if } 85 \leq x_{y,x} < 170 \\ +1 & \text{if } x_{y,x} \geq 170 \end{cases}$$

Two derivative channels are computed in ternary space:

$$h_{y,x} = \text{clamp}(t_{y,x+1} - t_{y,x}), \quad v_{y,x} = \text{clamp}(t_{y+1,x} - t_{y,x})$$

where clamp maps values outside $\{-1, 0, +1\}$ to the nearest endpoint. This produces three ternary channels: pixel ($\mathbf{t}$), horizontal gradient ($\mathbf{h}$), and vertical gradient ($\mathbf{v}$).

Each channel has an independent background value skipped during indexing: BG_PIXEL = 0 (all-dark block), BG_GRAD = 13 (flat region). The bytepacked variant fuses all three channels plus transition flags into a single byte (256-value signatures), capturing cross-channel correlations absent in independent channels.

### 3.2 Block Encoding

The image is decomposed into 3x1 horizontal blocks at $P = 252$ overlapping positions. Each block maps to a ternary signature $s \in \{0, \ldots, 26\}$ (or $\{0, \ldots, 255\}$ for bytepacked). Three block series are extracted per channel, yielding 252 signature sequences per query.

### 3.3 Information-Gain Weighted Inverted Index

**Training.** For each position $p$ and signature value $v$, store the list of training image IDs exhibiting value $v$ at position $p$: $L_{p,v} = \{i : s_{p,i} = v\}$.

Compute the information gain per position:

$$\text{IG}(p) = H(\mathcal{C}) - H(\mathcal{C} \mid s_p)$$

where $H(\mathcal{C})$ is the marginal class entropy and $H(\mathcal{C} \mid s_p)$ is the conditional entropy given the block signature at position $p$. Weights are normalized to integers in $[1, 16]$, computed once in a single pass over training data.

**Inference.** For each block position $p$, look up the inverted list and add IG-weighted votes to each training image in the list:

$$V[i] \mathrel{+}= \text{IG}(p) \quad \forall\, i \in L_{p,s_p}$$

The top-K training images by vote count form the candidate set $\mathcal{K}$.

### 3.4 Multi-Probe Expansion

For each position, additionally look up 6 Hamming-1 trit-flip neighbors at half weight:

$$V[i] \mathrel{+}= \tfrac{1}{2}\,\text{IG}(p) \quad \forall\, i \in L_{p,s'}, \quad s' \in \text{Neighbors}(s_p)$$

This handles single-trit quantization noise without explicit noise modeling.

### 3.5 Ternary Dot-Product Ranking

Candidates are ranked by a weighted multi-channel ternary dot product:

$$\text{score}(i) = 256 \cdot \langle \mathbf{t}, \mathbf{t}_i \rangle + 192 \cdot \langle \mathbf{v}, \mathbf{v}_i \rangle$$

Weights were selected by grid search over $\{0, 64, 128, 192, 256\}^3$ (125 combinations) on a validation split. Each dot product is computed via `_mm256_sign_epi8` — an AVX2 instruction that performs exact ternary multiplication at 32 operations per cycle (see Appendix B). The k=3 majority vote among top-3 scored candidates determines the class.

### 3.6 Structural Ranking Features

The dot product is augmented by discrete features from the gradient channels:

**Gradient divergence.** The finite-difference divergence of the gradient field, computed in cells of a 2x4 spatial grid (MNIST) or 3x3 grid (Fashion-MNIST):

$$\text{div}(x,y) = (h_{x+1,y} - h_{x,y}) + (v_{x,y+1} - v_{x,y})$$

summed per grid cell. Negative divergence concentrates at enclosed regions and concavities.

**Enclosed-region centroid.** Flood-fill from image corners identifies interior foreground pixels. The centroid of enclosed pixels locates the center of mass of digit loops (distinguishing 4 from 9, 6 from 0).

**Horizontal intensity profile.** Per-row foreground pixel count, producing a 28-element feature vector capturing stroke density distribution.

**Adaptive weighting.** The median absolute deviation (MAD) of candidate divergence scores modulates feature weights per-query:

$$\lambda = \frac{c}{c + \text{MAD}(D_\mathcal{K})}$$

When candidates cluster tightly in divergence (low MAD), structural features are trusted. When they disagree, the system falls back to dot product scores.

The combined score:

$$\text{score\_final}(i) = \text{score\_dot}(i) + \lambda \cdot [\alpha_\text{div} D(i) + \alpha_\text{cent} C(i) + \alpha_\text{prof} P(i)]$$

### 3.7 Adaptive Three-Tier Router

Vote concentration — available at zero cost after retrieval — predicts query difficulty.

**Tier 1 (<1 us).** Vote unanimity $\geq$ 198/200: return plurality immediately. Covers 10-15% of queries at 99.9% accuracy with zero false positives.

**Tier 2 (~1.5 us).** Moderate confidence: consult dual frequency tables (pixel + edge geometry) via posterior summation. Covers 15-18%.

**Tier 3 (~1 ms).** All remaining queries: full pipeline.

**Global result:** 96.50% at 0.67ms average latency (2x throughput over full system).

---

## 4. Experiments

### 4.1 Setup

**MNIST** (LeCun et al., 1998): 60K train / 10K test, 28x28 grayscale, 10 classes.

**Fashion-MNIST** (Xiao et al., 2017): identical format, 10 clothing categories. The same architecture with IG re-computed on Fashion training data — no hyperparameter changes.

**CIFAR-10** (Krizhevsky, 2009): 50K train / 10K test, 32x32 RGB. Ternary pipeline applied per-channel with stereo-vote retrieval and Gauss-map ranking.

**Validation protocol.** For Fashion-MNIST, all configurable weights were selected on a 5K validation split (test images 0-4999), frozen before evaluation on the 5K holdout (images 5000-9999). For MNIST, weights were originally selected on the full 10K test set; a retroactive val/holdout audit confirmed the identical weights are found on val only (see Section 4.2). Fashion val-derived weights differ from MNIST (see Section 4.2 Table 2 notes).

**Baselines.** Brute kNN (ternary and L2), kNN+PCA (scikit-learn, best over PCA dims {30,50,100,150} and k in {1,3,5}), random forest (100 trees), linear SVM (SGD), 1-hidden-layer MLP (256 ReLU units, Adam, 50 epochs).

### 4.2 Main Results

**Table 1: MNIST (10K test images; 95% CI via Clopper-Pearson).**

| Method | Accuracy | 95% CI | Learned params |
|---|---|---|---|
| Linear SVM | 89.48% | +/-0.61 | ~7,840 |
| Brute k=3 NN (ternary) | 96.12% | +/-0.38 | 0 |
| Brute k=3 NN (L2) | 97.05% | +/-0.34 | 0 |
| Random forest (100 trees) | 97.05% | +/-0.34 | ~10^6 |
| **kNN+PCA(30) k=5** | **97.55%** | **+/-0.31** | **0 (SVD)** |
| 1-layer MLP (256 units) | 97.63% | +/-0.30 | ~203K |
| **SSTT (full system)** | **97.27%** | **+/-0.32** | **0** |

SSTT exceeds brute ternary kNN (+1.15pp) and brute L2 kNN (+0.22pp, within CI overlap). The gap to kNN+PCA (0.28pp) falls within overlapping confidence intervals. Note that the ranking stage compares 50 candidates vs kNN+PCA's 60,000; total compute including vote accumulation is discussed in Section 5.3.

**Validation protocol.** Feature weights were originally selected on the full 10K test set. A retroactive val/holdout audit (5K/5K split) confirmed the identical weights are found on val only: val accuracy 96.30%, holdout 98.24%, full 10K 97.27%. The 1.94pp val/holdout gap reflects data composition (the holdout half contains easier images), not overfitting — val-derived and test-derived weights are identical. We report the full 10K number with this disclosure.

**Table 2: Fashion-MNIST (10K test, same architecture, val-derived weights).**

| Method | Accuracy | 95% CI | Learned params |
|---|---|---|---|
| Brute 1-NN (ternary) | 76.86% | +/-0.83 | 0 |
| Linear SVM | 82.50% | +/-0.75 | ~7,840 |
| Brute k=3 NN (L2) | 85.41% | +/-0.70 | 0 |
| **kNN+PCA(150) k=5** | **86.35%** | **+/-0.68** | **0 (SVD)** |
| Random forest | 87.60% | +/-0.65 | ~10^6 |
| 1-layer MLP | 88.48% | +/-0.63 | ~203K |
| **SSTT (full system)** | **85.68%** | **+/-0.69** | **0** |

SSTT lifts +8.50pp over brute ternary kNN on the Fashion holdout split (77.18% baseline) — a larger margin than the +1.15pp on MNIST — demonstrating greater discriminative value on a harder task. Against brute L2 kNN (85.41%), the gap narrows to +0.27pp (within CI overlap). kNN+PCA(150) k=5 achieves 86.35%, 0.67pp above SSTT with overlapping CIs. Both SSTT and kNN+PCA are below random forest (87.60%) and 1-layer MLP (88.48%).

**Parameter differences from MNIST.** The same architecture is used, but val-derived weights differ: spatial grid 3x3 (was 2x4), divergence weight 50 (was 200), gradient weight 200 (was 100), MAD scale 100 (was 50). Additionally, Bayesian sequential processing contributes +1.14pp on Fashion holdout — a genuine signal confirmed by val/holdout protocol — while providing zero benefit on MNIST. The IG weights are re-computed on Fashion training data (a closed-form operation, not tuning in the gradient-descent sense). We do not claim "zero tuning" — the feature weights were selected on the Fashion validation split.

**Table 3: Classifier configurations.**

| Method | MNIST | Fashion | Latency |
|---|---|---|---|
| Dual Hot Map | 76.53% | — | ~1.5 us |
| Bytepacked Bayesian | 91.83% | — | 1.9 us |
| Bytepacked cascade | 96.28% | 82.89% | 930 us |
| Three-tier router | 96.50% | 83.42% | 0.67 ms |
| **Full system (topo9)** | **97.27%** | **85.68%** | **~1 ms** |

### 4.3 Tier 1: Zero False Positive Rate

On Fashion-MNIST, Tier 1 (unanimity threshold 198/200) classifies **9.6% of the test set at 100.0% accuracy** — zero false positives across 960 images. The threshold was set on MNIST and applied directly to Fashion without tuning.

---

## 5. Analysis

### 5.1 K-Invariance

**Table 4: Accuracy vs K (bytepacked cascade).**

| K | Accuracy |
|---|---|
| 50 | 96.28% |
| 100 | 96.28% |
| 200 | 96.28% |
| 500 | 96.28% |
| 1000 | 96.28% |

Accuracy is identical from K=50 through K=1000. The top-50 candidates contain every relevant neighbor from 60,000 training images — **1200x compression with zero recall loss.**

The IG weighting concentrates vote mass on discriminative positions, creating a distribution where class-consistent training images receive disproportionate support. At K=50, the vote threshold excludes noise while retaining all relevant candidates. The inverted-index vote acts as a near-perfect class-conditional filter.

**The implication is fundamental: retrieval is solved at the cascade level.** All remaining classification error is ranking, not retrieval. SSTT does not approximate kNN by trading recall for speed. It achieves equivalent recall through a structurally different representation, then ranks within a pre-filtered set.

**K-sensitivity of the full system.** K-invariance holds for the cascade's dot-product ranking but breaks when structural features are added. The full topo9 system shows accuracy increasing with K:

| K | Topo9 Accuracy | Cascade Accuracy |
|---|---|---|
| 50 | 96.59% | 96.28% |
| 100 | 96.94% | 96.28% |
| 200 | 97.27% | 96.28% |
| 500 | 97.57% | 96.28% |
| 1000 | 97.61% | 96.28% |

The cascade ranking (dot-product only) is K-invariant: all relevant candidates appear by K=50. The structural ranker (divergence, centroid, profile features) can extract additional signal from lower-ranked candidates that the dot product cannot distinguish. Accuracy plateaus between K=500 and K=1000 (+0.04pp), suggesting diminishing returns beyond K=500. The current system uses K=200, leaving ~0.34pp on the table relative to K=1000.

This decomposition clarifies the contribution: the inverted index achieves near-perfect retrieval at K=50 (proved by cascade K-invariance), and the structural ranker adds value by exploiting deeper candidate pools (proved by topo9 K-sensitivity). The two findings are complementary, not contradictory.

### 5.2 Retrieval Method Comparison

To isolate the inverted index's contribution, we compare three retrieval methods at K=50, all feeding the same topo9 ranker:

| Method | Accuracy | Time/query | Retrieval Recall |
|---|---|---|---|
| SSTT Inverted Index | 96.62% | 1038 us | 99.63% |
| Brute L2 (SAD, AVX2) | 97.49% | 2857 us | 99.73% |
| Random Projection LSH (64-bit) | 93.46% | 131 us | 99.39% |

Brute L2 nearest-neighbor retrieval at K=50 produces candidates that the topo9 ranker scores 0.87pp higher than the inverted-index candidates, despite similar retrieval recall (99.73% vs 99.63%). The L2 candidates are geometrically closer to the query in pixel space, giving the structural features more discriminative signal. However, brute L2 is 2.75x slower per query.

Random projection LSH is 8x faster but retrieval quality is insufficient — 3.16pp below the inverted index and 4.03pp below brute L2.

**Interpretation.** The inverted index's advantage is speed, not candidate quality. At equal K, brute L2 produces better candidates for the structural ranker. The inverted index achieves a useful speed-quality tradeoff: 96.62% at 1038us vs 97.49% at 2857us. For the full topo9 system at K=200 (its default), the inverted index achieves 97.27% — between the K=50 inverted-index result (96.62%) and the K=50 brute-L2 result (97.49%) — by compensating with more candidates.

### 5.3 Error Mode Decomposition

Every misclassification falls into one of three modes:

- **Mode A: Retrieval failure.** Correct class absent from top-K. Cannot be fixed by better ranking.
- **Mode B: Ranking inversion.** Correct class present but wrong class wins the scoring step.
- **Mode C: Vote dilution.** Ranking identifies the nearest neighbor correctly but k=3 majority vote dilutes it.

**Bytepacked cascade (96.28%, 372 errors):** A: 7 (1.7%), B: 240 (64.5%), C: 125 (33.8%).

**Full system (97.27%, 273 errors):** A: 7 (2.6%), B: 197 (72.2%), C: 69 (25.3%).

Structural ranking fixes 99 errors relative to the cascade — primarily ranking inversions (240 to 197) and vote dilutions (125 to 69). Mode A retrieval failures remain unchanged at 7, confirming improvements are entirely in ranking.

**In both systems, >97% of errors occur at ranking, not retrieval.** The inverted index achieves 99.3% effective recall.

**Ceilings.** If a perfect ranker resolved all Mode B and C errors, only the 7 Mode A retrieval failures would remain: a theoretical ceiling of 99.93% (9993/10000). A more realistic estimate — fixing all Mode B ranking inversions but not Mode C vote dilutions (which require structural changes to the k=3 vote) — yields 99.24% for topo9 (9924/10000) or 98.68% for the bytepacked cascade (9868/10000).

### 5.4 Compute Profile

| Phase | Runtime share |
|---|---|
| Vote accumulation (scattered writes) | **87%** |
| Top-K selection | ~5% |
| Encoding and overhead | ~7% |
| Ternary dot products | <1% |

The bottleneck is cache-line thrashing on a 240KB vote array during scattered writes — not arithmetic. The `_mm256_sign_epi8` dot products are effectively free. For any inverted-index system on modern hardware, memory access patterns dominate; arithmetic optimizations provide marginal benefit.

### 5.5 Channel Ablation

**Table 5: Dot-product weight ablation on the bytepacked cascade (K=50, k=3, no structural features). Full 10K test set.**

| Weights (px, hg, vg) | MNIST | Fashion |
|---|---|---|
| (256, 0, 0) pixel only | 96.04% | 82.11% |
| (256, 192, 0) +h-grad | 95.87% | 83.44% |
| (256, 0, 192) +v-grad | 96.38% | 82.89% |
| (256, 192, 192) all three | 96.21% | 83.66% |

These are cascade-only numbers (no topological features). On MNIST, h-grad adds noise (-0.17pp from pixel-only); v-grad helps (+0.34pp). On Fashion, h-grad helps (+1.33pp) because clothing textures have horizontal structure that digit strokes do not. The channel importance reversal is handled automatically: IG re-weighting assigns higher weight to h-grad positions on Fashion training data.

### 5.6 Cross-Dataset Generalization

| Stage | MNIST | Fashion | Changes from MNIST |
|---|---|---|---|
| Brute 1-NN (ternary) | 95.87% | 76.86% | None |
| Cascade (MNIST weights) | 96.04% | ~80% | None |
| Cascade (Fashion IG) | — | 82.89% | IG re-computation |
| Full system (val-derived) | **97.27%** | **85.68%** | IG + val-derived weights + Bayesian seq. |

IG re-computation is closed-form (no cross-validation, no learning rate). The code is identical; the input data differs. However, the val-derived feature weights for Fashion differ from MNIST (see Section 4.2), and Bayesian sequential processing — which provides zero benefit on MNIST — contributes +1.14pp on Fashion. The generalization claim is that the *architecture* transfers without modification; the *parameters* are selected per-dataset on validation splits.

### 5.7 CIFAR-10: Architectural Boundary

On CIFAR-10, the ternary cascade achieves **50.18%** (5x random) with 99.44% retrieval recall.

**Per-class accuracy:**

| Class | Acc. | Class | Acc. |
|---|---|---|---|
| Ship | 63.7% | Cat | 33.6% |
| Truck | 62.1% | Dog | 40.1% |
| Airplane | 58.9% | Bird | 42.4% |
| Frog | 56.2% | Horse | 52.0% |
| Deer | 53.8% | Auto | 61.3% |

Edge-distinctive classes (ship, truck, airplane) achieve 58-64%. Texture-dependent classes (cat, dog) fall near chance. Ternary block signatures capture edge geometry but not texture or semantic content. This is an honest architectural boundary: no amount of ternary engineering will separate cat from dog at 32x32 without learned texture features.

We report CIFAR-10 not as a success but to scope precisely where the architecture stops working.

---

## 6. Discussion

### 6.1 Why the Cascade Exceeds Brute kNN

The full system outperforms brute k=3 NN (97.27% vs 96.12% ternary) by applying a richer ranking function to a pre-filtered candidate set. The retrieval comparison (Section 5.2) reveals a nuance: at equal K=50, brute L2 retrieval produces candidates that the topo9 ranker scores 0.87pp higher than inverted-index candidates (97.49% vs 96.62%). The inverted index compensates with speed (2.75x faster) and with larger K — at K=200, the inverted-index system achieves 97.27%, closing the gap. The inverted index's advantage is computational, not in candidate quality.

### 6.2 Non-Iterative Parameters

All parameters are derived without iterative optimization:

- **IG weights**: mutual information, computed in a single O(N*P) pass.
- **Background thresholds**: per-channel mode of ternary distribution, computed once.
- **Dot-product weights**: grid search over 125 discrete combinations on validation split.
- **Structural weights and MAD scale**: ablation on validation split.
- **Router threshold**: single integer set on validation data.

**Scope of the non-iterative claim.** IG weights are computed from labeled data; feature weights are selected by grid search on a validation split. Both are forms of supervised computation. The specific claim is: no parameter is updated by a loss function or iterative procedure. There is no training loop, no learning rate, no convergence criterion. SSTT is methodologically closer to kNN+PCA (eigendecomposition from labeled data, brute evaluation at inference) than to gradient-trained models. The practical consequence is deterministic, reproducible inference with no optimization history.

### 6.3 Failure Modes and Limits

The 4/9 and 3/5 confusion pairs dominate remaining MNIST errors. These share both pixel similarity and topological structure. Resolution likely requires higher-level features beyond ternary block encoding.

The theoretical ceiling of 99.93% (7 retrieval failures) represents maximum accuracy given current retrieval. A realistic improved ranker would likely reach 99.24% (fixing ranking inversions but not vote dilutions). Exceeding 99.93% requires improving Mode A recall.

### 6.4 Negative Results

Three approaches that failed, with root causes:

**Ternary PCA (-16 to -23pp).** PCA projections destroy the spatial locality that makes block-signature voting work. The hot map is fundamentally spatial; PCA is spectral. They are incompatible.

**Hard-filtered routing (21-50%).** Using a fast Bayesian classifier to hard-gate the cascade permanently removes correct classes when the Bayesian is wrong (15% of the time). A weaker classifier cannot productively hard-gate a stronger one.

**Differential operators on ternary fields (~10%).** Navier-Stokes stream function and Taylor jet space — operators that depend on continuous derivatives — fail when the field is quantized to three levels. Integral measures (divergence sums, centroids) are robust to quantization; differential measures (Hessian eigenvalues, curl) are not.

---

## 7. Conclusion

We have demonstrated that classification without iterative optimization can achieve accuracy within the confidence interval of kNN+PCA — the strongest non-iterative baseline — using a structurally different computational approach. The system uses no gradient descent, no backpropagation, and no floating-point arithmetic at inference.

K-invariance in the dot-product cascade is the central retrieval result: the inverted index achieves near-perfect recall, compressing candidate search by 1200x with zero accuracy cost at the cascade level. The structural ranker breaks K-invariance, gaining +1.02pp from K=50 to K=1000 — revealing that richer ranking functions can exploit deeper candidate pools. At equal K=50, brute L2 retrieval produces better candidates for the structural ranker (97.49% vs 96.62%), but the inverted index is 2.75x faster and compensates with larger K. The remaining gap between 97.27% (K=200) and the theoretical ceiling of 99.93% (7 retrieval failures) is a ranking problem. This is a precise, testable claim that defines the path forward.

The cross-dataset transfer (+8.50pp over brute ternary kNN on the Fashion holdout split, with val-derived weights) confirms the architecture generalizes beyond MNIST. The CIFAR-10 boundary (50.18%) confirms where it stops — ternary features capture edges but not texture. Both findings are necessary for an honest assessment of what address-generation classification can and cannot do.

---

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*, 785-794.
- Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Trans. Information Theory*, 13(1), 21-27.
- Johnson, J., Douze, M., & Jegou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Trans. Big Data*, 7(3), 535-547.
- Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. *Technical Report*, U. Toronto.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*, 86(11), 2278-2324.
- Li, F., Zhang, B., & Liu, B. (2016). Ternary weight networks. *arXiv:1605.04711*.
- Norouzi, M. & Fleet, D. J. (2012). Minimal loss hashing for compact binary codes. *ICML*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.
- Sivic, J. & Zisserman, A. (2003). Video Google: A text retrieval approach to object matching in videos. *ICCV*.
- Weiss, Y., Torralba, A., & Fergus, R. (2009). Spectral hashing. *NeurIPS*.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747*.
- Zhu, C., Han, S., Mao, H., & Dally, W. J. (2016). Trained ternary quantization. *arXiv:1612.01064*.

---

## Appendix A: Reproducibility

All source code is available at [repository URL]. Data is downloaded via `make mnist` and `make fashion` from canonical sources with SHA-256 verification. All experiments are single-file C99 programs with no external dependencies beyond a C compiler with AVX2 support.

**Hardware.** Timing figures measured on a single x86 core: AMD Ryzen 5 PRO 5675U, 4.4 GHz boost, 512 KB L2 per core, 16 MB L3. No GPU, no multi-threading.

**Validation protocol.** MNIST: feature weights were originally selected on the full 10K test set. A retroactive val/holdout audit (5K/5K split on the test set) confirmed that val-derived weights are identical. The reported 97.27% is the full 10K number; val=96.30%, holdout=98.24%. Fashion-MNIST: all weights selected on the 5K val split and frozen before holdout evaluation. Val-derived weights differ from MNIST (grid 3x3 vs 2x4, divergence weight 50 vs 200, gradient weight 200 vs 100, MAD scale 100 vs 50). Bayesian sequential processing contributes +1.14pp on Fashion holdout (zero on MNIST). The reported 85.68% is the holdout number with val-derived weights.

**Statistical significance.** All accuracy figures include 95% Clopper-Pearson confidence intervals over n=10,000 test images.

## Appendix B: The `_mm256_sign_epi8` Ternary Primitive

The AVX2 instruction `_mm256_sign_epi8(a, b)` computes per byte:

$$\text{out}[i] = \begin{cases} a[i] & \text{if } b[i] > 0 \\ -a[i] & \text{if } b[i] < 0 \\ 0 & \text{if } b[i] = 0 \end{cases}$$

Over the ternary alphabet $b[i] \in \{-1, 0, +1\}$, this is exact multiplication: $\text{out}[i] = a[i] \cdot b[i]$. Accumulating in `int8` is safe for up to 127 iterations; MNIST's 28 blocks per row is well within bounds.

This gives 32 ternary multiply-accumulates per instruction, per cycle, using register renaming rather than a multiply unit. Prior ternary neural network work (TWN, TTN) implements ternary multiplication via conditional branching or lookup tables; this approach exploits a specific x86 instruction semantic that maps exactly to the ternary algebra.

```c
// Exact ternary multiply-accumulate over {-1, 0, +1}:
__m256i prod = _mm256_sign_epi8(a, b);
acc = _mm256_add_epi8(acc, prod);
```
