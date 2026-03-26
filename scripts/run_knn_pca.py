#!/usr/bin/env python3
"""
kNN+PCA baselines on MNIST, Fashion-MNIST, and EMNIST.
Must-run before paper submission: if kNN+PCA beats 97.27%, the
"exceeds brute kNN" claim needs qualification.
"""

import struct
import numpy as np
from pathlib import Path
from scipy.stats import beta as beta_dist
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import time


def load_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)


def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def clopper_pearson(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    mid = k / n
    return mid, (hi - lo) / 2


def run_knn_pca(name, X_train, y_train, X_test, y_test):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    n_test = len(y_test)
    print(f"  Train: {len(y_train)}, Test: {n_test}, Features: {X_train.shape[1]}")

    X_train_f = X_train.astype(np.float32)
    X_test_f = X_test.astype(np.float32)

    # Brute kNN baselines (no PCA)
    for k in [1, 3]:
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=-1)
        knn.fit(X_train_f, y_train)
        y_pred = knn.predict(X_test_f)
        correct = int((y_pred == y_test).sum())
        acc, ci = clopper_pearson(correct, n_test)
        print(f"  Brute k={k} NN (no PCA): {acc*100:.2f}% +/-{ci*100:.2f}  ({time.time()-t0:.1f}s)")

    # kNN+PCA sweep
    results = []
    for n_comp in [30, 50, 75, 100, 150]:
        t0 = time.time()
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_train_f)
        X_te_pca = pca.transform(X_test_f)

        for k in [1, 3, 5]:
            knn = KNeighborsClassifier(n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=-1)
            knn.fit(X_tr_pca, y_train)
            y_pred = knn.predict(X_te_pca)
            correct = int((y_pred == y_test).sum())
            acc, ci = clopper_pearson(correct, n_test)
            elapsed = time.time() - t0
            results.append((n_comp, k, acc, ci))
            print(f"  PCA({n_comp:>3d}) k={k}: {acc*100:.2f}% +/-{ci*100:.2f}  ({elapsed:.1f}s)")

    print(f"\n  Summary for {name}:")
    print(f"  {'PCA':>5s} {'k':>3s} {'Accuracy':>10s} {'95% CI':>10s}")
    print(f"  {'-'*5} {'-'*3} {'-'*10} {'-'*10}")
    for n_comp, k, acc, ci in results:
        print(f"  {n_comp:>5d} {k:>3d} {acc*100:>9.2f}% {f'±{ci*100:.2f}':>10s}")

    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    # MNIST
    d = root / "data"
    X_tr = load_idx_images(d / "train-images-idx3-ubyte")
    y_tr = load_idx_labels(d / "train-labels-idx1-ubyte")
    X_te = load_idx_images(d / "t10k-images-idx3-ubyte")
    y_te = load_idx_labels(d / "t10k-labels-idx1-ubyte")
    mnist_results = run_knn_pca("MNIST", X_tr, y_tr, X_te, y_te)

    # Fashion-MNIST
    d = root / "data-fashion"
    X_tr = load_idx_images(d / "train-images-idx3-ubyte")
    y_tr = load_idx_labels(d / "train-labels-idx1-ubyte")
    X_te = load_idx_images(d / "t10k-images-idx3-ubyte")
    y_te = load_idx_labels(d / "t10k-labels-idx1-ubyte")
    fashion_results = run_knn_pca("Fashion-MNIST", X_tr, y_tr, X_te, y_te)

    # EMNIST (if available)
    emnist_dir = root / "data-emnist"
    if (emnist_dir / "train-images-idx3-ubyte").exists():
        X_tr = load_idx_images(emnist_dir / "train-images-idx3-ubyte")
        y_tr = load_idx_labels(emnist_dir / "train-labels-idx1-ubyte")
        X_te = load_idx_images(emnist_dir / "t10k-images-idx3-ubyte")
        y_te = load_idx_labels(emnist_dir / "t10k-labels-idx1-ubyte")
        emnist_results = run_knn_pca("EMNIST", X_tr, y_tr, X_te, y_te)
    else:
        print(f"\n  EMNIST not found at {emnist_dir}. Skipping.")

    print("\n\n" + "="*60)
    print("  PAPER TABLE VALUES")
    print("="*60)
    for name, results in [("MNIST", mnist_results), ("Fashion-MNIST", fashion_results)]:
        print(f"\n  {name}:")
        for n_comp, k, acc, ci in results:
            print(f"  | PCA({n_comp}) k={k} | {acc*100:.2f}% | ±{ci*100:.2f} |")
