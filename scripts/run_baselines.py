#!/usr/bin/env python3
"""
Run baseline classifiers (Random Forest, Linear SVM, 1-layer MLP) on MNIST
and Fashion-MNIST using raw pixel features. Reports accuracy with 95%
Clopper-Pearson confidence intervals.

Used to populate Tables 1 and 2 of the SSTT paper draft.
"""

import struct
import numpy as np
from pathlib import Path
from scipy.stats import beta as beta_dist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
    """95% Clopper-Pearson CI half-width for k successes in n trials."""
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    mid = k / n
    return mid, (hi - lo) / 2  # accuracy, half-width


def run_baselines(name, data_dir):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    d = Path(data_dir)
    X_train = load_idx_images(d / "train-images-idx3-ubyte")
    y_train = load_idx_labels(d / "train-labels-idx1-ubyte")
    X_test = load_idx_images(d / "t10k-images-idx3-ubyte")
    y_test = load_idx_labels(d / "t10k-labels-idx1-ubyte")

    n_test = len(y_test)
    print(f"  Train: {len(y_train)}, Test: {n_test}")
    print(f"  Features: {X_train.shape[1]} raw pixels")

    # Normalize for SVM and MLP (RF doesn't need it)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train.astype(np.float32))
    X_test_sc = scaler.transform(X_test.astype(np.float32))

    models = [
        ("Random Forest (100 trees)", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)),
        ("Linear SVM (SGD, hinge)", SGDClassifier(
            loss="hinge", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=42)),
        ("1-layer MLP (256 units)", MLPClassifier(
            hidden_layer_sizes=(256,), activation="relu", solver="adam",
            max_iter=50, batch_size=128, random_state=42)),
    ]

    results = []
    for label, clf in models:
        print(f"\n  Training {label}...", end=" ", flush=True)
        t0 = time.time()

        # RF uses raw pixels; SVM and MLP use scaled
        use_scaled = not isinstance(clf, RandomForestClassifier)
        # SGDClassifier also needs scaled input
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc if use_scaled else X_test

        clf.fit(Xtr, y_train)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

        y_pred = clf.predict(Xte)
        correct = int((y_pred == y_test).sum())
        acc, ci = clopper_pearson(correct, n_test)

        n_params = "~10^6"
        if isinstance(clf, SGDClassifier):
            n_params = f"~{clf.coef_.size:,}"
        elif isinstance(clf, MLPClassifier):
            total = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
            n_params = f"~{total:,}"

        print(f"  {label}: {acc*100:.2f}% +/-{ci*100:.2f}  (params: {n_params})")
        results.append((label, acc, ci, n_params))

    print(f"\n  Summary for {name}:")
    print(f"  {'Method':<35} {'Accuracy':>10} {'95% CI':>10} {'Params':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*12}")
    for label, acc, ci, params in results:
        print(f"  {label:<35} {acc*100:>9.2f}% {f'±{ci*100:.2f}':>10} {params:>12}")

    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    mnist_results = run_baselines("MNIST", root / "data")
    fashion_results = run_baselines("Fashion-MNIST", root / "data-fashion")

    print("\n\n" + "=" * 60)
    print("  PAPER TABLE VALUES (copy into Tables 1 and 2)")
    print("=" * 60)

    for name, results in [("MNIST", mnist_results), ("Fashion-MNIST", fashion_results)]:
        print(f"\n  {name}:")
        for label, acc, ci, params in results:
            print(f"  | {label} | {acc*100:.2f}% | ±{ci*100:.2f} | {params} |")
