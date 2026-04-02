# CIFAR-10 Boundary Experiments

These experiments test where the ternary cascade architecture breaks down on natural images.

**Result: 50.18% accuracy (5x random).** This is an honest architectural boundary, not a success claim. Ternary block signatures capture edge geometry but cannot distinguish texture or semantic content at 32x32 resolution. Cat/dog pairs are barely separable without learned features.

## Key files

| File | Accuracy | What it tests |
|------|----------|---------------|
| sstt_cifar10.c | 26.51% | Straight port from MNIST (grayscale, same thresholds) |
| sstt_cifar10_stereo.c | 41.18% | Multi-perspective quantization |
| sstt_cifar10_gauss.c | 48.31% | Gaussian frequency map ranking |
| sstt_cifar10_cascade_gauss.c | 50.18% | Texture retrieval + shape ranking (best result) |
| sstt_cifar10_5trit_quick.c | — | 5-eye ensemble on CIFAR-10 |

The remaining ~30 files explore curvature, Lagrangian tracing, mixture-of-experts, multi-scale, and other approaches. Most are parameter sweeps or dead ends. They are preserved for reproducibility.

## What this proves

Edge-distinctive classes (ship 63.7%, truck 62.1%) outperform texture-dependent classes (cat 33.6%, dog 40.1%). The architecture extracts real structure but hits a hard wall where learned features are required.
