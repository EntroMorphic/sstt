# Reproducing Results

## Requirements

- x86-64 CPU with AVX2 support (Intel Haswell+ or AMD Zen+)
- GCC 9+ with `-mavx2 -mfma` support
- ~300 MB disk for datasets
- ~2 GB RAM

## Quick start

```bash
# Clone and build core classifiers
git clone https://github.com/<owner>/sstt.git
cd sstt
make

# Download MNIST
make mnist

# Run the headline result (97.27% MNIST)
./build/sstt_topo9_val
```

Expected output:
```
=== SSTT Topo9 Val/Holdout ===
...
Val accuracy:     96.30% (4815/5000)
Holdout accuracy: 98.24% (4912/5000)
Full 10K:         97.27% (9727/10000)
```

## Reproducing each key result

### Bytepacked cascade (96.28% MNIST)

```bash
make sstt_bytecascade
./build/sstt_bytecascade
```

### Three-tier router (96.50% MNIST at 0.67ms)

```bash
make sstt_router_v1
./build/sstt_router_v1
```

### Fashion-MNIST (85.68%)

```bash
make fashion
./build/sstt_topo9_val data-fashion/
```

### CIFAR-10 boundary test (50.18%)

```bash
make cifar10
make sstt_cifar10_cascade_gauss
./build/sstt_cifar10_cascade_gauss
```

## Build targets

| Target | What it builds |
|--------|----------------|
| `make` | 4 core classifiers (topo9_val, bytecascade, router_v1, v2) |
| `make analysis` | 7 diagnostic tools |
| `make ablation` | topo1-topo9 ablation series |
| `make cifar10` | CIFAR-10 experiments |
| `make archive` | Archived/exploratory experiments |
| `make experiments` | Everything |

## Data download

| Target | Dataset | Size |
|--------|---------|------|
| `make mnist` | MNIST (60K train + 10K test) | ~53 MB |
| `make fashion` | Fashion-MNIST | ~53 MB |
| `make cifar10` | CIFAR-10 | ~176 MB |

Data integrity is verified via SHA-256 checksums on download.

## Verifying correctness

```bash
# Run the fused kernel correctness test
make sstt_fused_test
make mnist
./build/sstt_fused_test
```

## Notes

- All binaries default to `data/` for MNIST. Pass a different directory as argv[1] for other datasets (e.g., `./build/sstt_topo9_val data-fashion/`).
- Binaries must run from the repository root (data paths are relative).
- Results may vary by +/-1 error on different hardware due to tie-breaking in argmax. Accuracy should be within +/-0.02pp.
