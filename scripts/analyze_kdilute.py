#!/usr/bin/env python3
"""
Mathematical analysis of four kNN dilution fixes.

Tests whether Kalman Adaptive, divergence weighting, and/or
confidence gating can reduce Mode C errors (correct class wins
dot ranking but loses kNN vote).
"""

import numpy as np

# Simulated per-image data structure
class ImageData:
    def __init__(self, true_label, rank1_label, combined_scores, labels, gdiv_scores, mad, test_divneg):
        self.true_label = true_label                   # ground truth
        self.rank1_label = rank1_label                 # rank-1 candidate's label
        self.combined_scores = combined_scores         # scores for top-20 candidates
        self.labels = labels                           # labels for top-20 candidates
        self.gdiv_scores = gdiv_scores                 # divergence scores for top-20
        self.mad = mad                                 # Kalman MAD of candidate pool
        self.test_divneg = test_divneg                 # test image's divergence

        # Baseline: k=3 majority vote
        self.baseline_pred = self._baseline_knn()
        self.is_mode_c = (rank1_label == true_label) and (self.baseline_pred != true_label)

    def _baseline_knn(self, k=3):
        """Standard k=3 majority vote"""
        k = min(k, len(self.labels))
        votes = np.bincount(self.labels[:k], minlength=10)
        return np.argmax(votes)

    def method1_adaptive_k(self, T_mad):
        """Kalman-adaptive k: low MAD → k=1, high MAD → k=7"""
        if self.mad < T_mad:
            k = 1
        elif self.mad < 2 * T_mad:
            k = 3
        else:
            k = 7
        k = min(k, len(self.labels))

        votes = np.bincount(self.labels[:k], minlength=10)
        return np.argmax(votes)

    def method2_weighted_vote(self, k):
        """Score-weighted kNN: vote weight = combined_score"""
        k = min(k, len(self.labels))
        if k < 1:
            return self.baseline_pred

        weights = self.combined_scores[:k] - self.combined_scores[k-1] + 1
        weights = np.maximum(weights, 1)  # avoid negatives

        weighted_votes = np.zeros(10, dtype=float)
        for i in range(k):
            weighted_votes[self.labels[i]] += weights[i]
        return np.argmax(weighted_votes)

    def method3_div_filtered(self, k, sigma_factor):
        """Divergence-filtered kNN: only count candidates within σ stddevs of median divergence"""
        k = min(k, len(self.labels))
        if k < 1:
            return self.baseline_pred

        div_vals = self.gdiv_scores[:k]
        med = np.median(div_vals)
        stddev = np.std(div_vals)
        if stddev == 0:
            stddev = 1  # avoid division by zero

        # Filter candidates
        mask = np.abs(div_vals - med) <= sigma_factor * stddev
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            # Fallback: use rank-1
            return self.labels[0]

        filtered_labels = self.labels[filtered_indices]
        votes = np.bincount(filtered_labels, minlength=10)
        return np.argmax(votes)

    def method4_skip_knn(self, T_mad, T_div):
        """Skip kNN if confident: output rank-1 if mad < T_mad AND test_divneg < T_div"""
        if self.mad < T_mad and self.test_divneg < T_div:
            return self.labels[0]  # Trust rank-1
        else:
            return self.baseline_pred


def generate_synthetic_data(n_total=10000):
    """
    Generate synthetic test data simulating MNIST results.
    - 97.31% baseline accuracy (9731 correct, 269 wrong)
    - 33.8% of errors are Mode C (91 Mode C images)
    """
    np.random.seed(42)
    images = []

    # Correct predictions
    n_correct = 9731
    for i in range(n_correct):
        true_label = i % 10
        # All top-3 from true class, so k=3 vote is correct
        labels = np.array([true_label] * 3 + [np.random.randint(0, 10) for _ in range(17)])
        scores = np.random.uniform(100, 500, 20)
        divscores = np.random.uniform(-100, -10, 20)
        mad = np.random.randint(10, 50)
        test_divneg = np.random.randint(-100, 0)

        img = ImageData(true_label, true_label, scores, labels, divscores, mad, test_divneg)
        images.append(img)

    # Mode C errors (91 images)
    for i in range(91):
        true_label = i % 10
        wrong_label = (true_label + 1) % 10
        # Rank-1 from true class, but rank-2 and rank-3 from wrong class
        labels = np.array([true_label] + [wrong_label] * 2 + [np.random.randint(0, 10) for _ in range(17)])
        scores = np.random.uniform(50, 200, 20)
        divscores = np.random.uniform(-100, -10, 20)
        mad = np.random.randint(15, 40)
        test_divneg = np.random.randint(-80, -20)

        img = ImageData(true_label, true_label, scores, labels, divscores, mad, test_divneg)
        images.append(img)

    # Other errors (178 images)
    for i in range(178):
        true_label = i % 10
        wrong_label = (true_label + 2) % 10
        # Rank-1 from wrong class (not Mode C)
        labels = np.array([wrong_label] + [np.random.randint(0, 10) for _ in range(19)])
        scores = np.random.uniform(50, 200, 20)
        divscores = np.random.uniform(-100, -10, 20)
        mad = np.random.randint(20, 60)
        test_divneg = np.random.randint(-100, 0)

        img = ImageData(true_label, wrong_label, scores, labels, divscores, mad, test_divneg)
        images.append(img)

    return images[:n_total]


def main():
    print("=" * 80)
    print("MATHEMATICAL TEST: Four Methods to Prevent kNN Dilution")
    print("=" * 80)
    print("\nGenerating synthetic MNIST-like data...")
    images = generate_synthetic_data(10000)

    # Baseline stats
    baseline_correct = sum(1 for img in images if img.baseline_pred == img.true_label)
    mode_c_count = sum(1 for img in images if img.is_mode_c)
    print(f"\nBaseline (k=3): {baseline_correct}/{len(images)} = {100*baseline_correct/len(images):.2f}%")
    print(f"Mode C errors: {mode_c_count}")

    # Method 1: Kalman-adaptive k
    print(f"\n{'='*80}\nMETHOD 1: Kalman-adaptive k\n{'='*80}")
    for T in [5, 10, 15, 20, 25, 30, 40, 50]:
        correct = sum(1 for img in images if img.method1_adaptive_k(T) == img.true_label)
        mode_c_fixed = sum(1 for img in images if img.is_mode_c and img.method1_adaptive_k(T) == img.true_label)
        delta_pp = (correct - baseline_correct) * 100.0 / len(images)
        print(f"  T_mad={T:3d}: ACC={correct:5d}/{len(images)}, Δ={delta_pp:+6.3f}pp, MC_FIXED={mode_c_fixed:2d}")

    # Method 2: Score-weighted kNN
    print(f"\n{'='*80}\nMETHOD 2: Score-weighted kNN\n{'='*80}")
    for k in [3, 5, 7, 10]:
        correct = sum(1 for img in images if img.method2_weighted_vote(k) == img.true_label)
        mode_c_fixed = sum(1 for img in images if img.is_mode_c and img.method2_weighted_vote(k) == img.true_label)
        delta_pp = (correct - baseline_correct) * 100.0 / len(images)
        print(f"  k={k:2d}: ACC={correct:5d}/{len(images)}, Δ={delta_pp:+6.3f}pp, MC_FIXED={mode_c_fixed:2d}")

    # Method 3: Divergence-filtered kNN
    print(f"\n{'='*80}\nMETHOD 3: Divergence-filtered kNN\n{'='*80}")
    for k in [3, 5, 7]:
        for sigma in [0.5, 1.0, 1.5, 2.0]:
            correct = sum(1 for img in images if img.method3_div_filtered(k, sigma) == img.true_label)
            mode_c_fixed = sum(1 for img in images if img.is_mode_c and img.method3_div_filtered(k, sigma) == img.true_label)
            delta_pp = (correct - baseline_correct) * 100.0 / len(images)
            print(f"  k={k}, σ={sigma:.1f}: ACC={correct:5d}/{len(images)}, Δ={delta_pp:+6.3f}pp, MC_FIXED={mode_c_fixed:2d}")

    # Method 4: Skip kNN if confident
    print(f"\n{'='*80}\nMETHOD 4: Skip kNN if confident\n{'='*80}")
    for T_mad in [10, 20, 30, 40]:
        for T_div in [-50, -30, -20, -10, 0]:
            correct = sum(1 for img in images if img.method4_skip_knn(T_mad, T_div) == img.true_label)
            mode_c_fixed = sum(1 for img in images if img.is_mode_c and img.method4_skip_knn(T_mad, T_div) == img.true_label)
            routed = sum(1 for img in images if img.mad < T_mad and img.test_divneg < T_div)
            delta_pp = (correct - baseline_correct) * 100.0 / len(images)
            print(f"  T_mad={T_mad:2d}, T_div={T_div:3d}: ACC={correct:5d}/{len(images)}, "
                  f"Δ={delta_pp:+6.3f}pp, MC_FIXED={mode_c_fixed:2d}, ROUTED={routed:4d} ({100*routed/len(images):5.1f}%)")

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print("Computational cost ranking (runtime complexity):")
    print("1. Method 4 (Skip kNN): 2 comparisons per image")
    print("2. Method 1 (Kalman-adaptive k): 2-3 comparisons per image")
    print("3. Method 2 (Score-weighted): ~10 multiplications + 10 additions")
    print("4. Method 3 (Divergence-filtered): median+stddev calculation (~50 ops)")
    print("\nBased on synthetic data:")
    print("- Method 1 is fast and simple, can eliminate some Mode C with low T_mad")
    print("- Method 4 is also very fast, directly skips kNN when confident")
    print("- Methods 2-3 have higher compute cost, potential for better accuracy")


if __name__ == "__main__":
    main()
