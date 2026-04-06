# Reproducing Results

## Requirements

- x86-64 CPU with AVX2 support (Intel Haswell+ or AMD Zen+)
- GCC 9+ with `-mavx2 -mfma` support
- ~300 MB disk for datasets
- ~2 GB RAM

## Quick start

```bash
# Clone and build core classifiers
git clone https://github.com/EntroMorphic/sstt.git
cd sstt
make

# Download MNIST
make mnist

# Run the headline result (97.53% MNIST, MTFP native ternary)
./build/sstt_mtfp
```

Expected output:
```
=== SSTT MTFP: Native Ternary Classification (MNIST) ===
...
Val:     96.58% (4829/5000, 171 errors)
Holdout: 98.48% (4924/5000, 76 errors)
...
MTFP full 10K: 97.53%
```

## Reproducing each key result

### MTFP native ternary (97.53% MNIST — current best)

```bash
./build/sstt_mtfp
```

### Topo9 val/holdout (97.27% MNIST — previous best)

```bash
./build/sstt_topo9_val
```

### Bytepacked cascade (96.28% MNIST)

```bash
./build/sstt_bytecascade
```

### Three-tier router (96.50% MNIST at 0.67ms)

```bash
./build/sstt_router_v1
```

### Fashion-MNIST (86.54% holdout, MTFP + LBP + Bayesian)

```bash
make fashion
./build/sstt_mtfp_dsp data-fashion/
```

### CIFAR-10 boundary test (50.18%)

```bash
make cifar10
make build/sstt_cifar10_cascade_gauss
./build/sstt_cifar10_cascade_gauss
```

### MTFP K-sweep + error diagnostics

```bash
./build/sstt_mtfp_diagnose            # MNIST Mode A/B/C at K=50..1000
./build/sstt_mtfp_diagnose data-fashion/  # Fashion-MNIST diagnostics
```

### K-invariance sweep (topo9, historical)

```bash
./build/sstt_kinvariance
```

### Retrieval method comparison

```bash
./build/sstt_ann_baseline
```

### Mechanism tuning grid (K × k × channel scale)

```bash
./build/sstt_mtfp_grid
```

## Build targets

| Target | What it builds |
|--------|----------------|
| `make` | Core classifiers (mtfp, topo9_val, bytecascade, router_v1, v2, experiments) |
| `make analysis` | 7 diagnostic tools |
| `make ablation` | topo1-topo9 ablation series |
| `make cifar10-experiments` | CIFAR-10 boundary experiments |
| `make archive` | Archived/exploratory experiments |
| `make experiments` | Everything (~95 binaries) |
| `make reproduce` | Core + MNIST download + run headline results |

## Data download

| Target | Dataset | Size |
|--------|---------|------|
| `make mnist` | MNIST (60K train + 10K test) | ~53 MB |
| `make fashion` | Fashion-MNIST | ~53 MB |
| `make cifar10` | CIFAR-10 | ~176 MB |

Data integrity is verified via SHA-256 checksums on download (MNIST, Fashion).

## Verifying correctness

```bash
# Run the fused kernel correctness test
make build/sstt_fused_test
make mnist
./build/sstt_fused_test
```

## Notes

- All binaries default to `data/` for MNIST. Pass a different directory as argv[1] for other datasets (e.g., `./build/sstt_mtfp data-fashion/`).
- Binaries must run from the repository root (data paths are relative).
- Results may vary by +/-1 error on different hardware due to tie-breaking in argmax. Accuracy should be within +/-0.02pp.
