# Contribution 9: WHT as Ternary-Native Orthogonal Transform

## Concept

The Walsh-Hadamard Transform (WHT) is the natural spectral transform for
ternary data because its butterfly operations are pure add/subtract — no
multiply needed. For ternary inputs in {-1, 0, +1}, the entire transform
stays in small integers.

The WHT matrix H_n is defined recursively:

```
H_1 = [1]

H_2 = [1  1]
      [1 -1]

H_2n = [H_n  H_n ]
       [H_n -H_n ]
```

The butterfly at each stage:
```c
a' = a + b
b' = a - b
```

For ternary inputs, WHT coefficients are bounded: at 32×32 = 1024 elements,
the maximum coefficient magnitude is 1024 (all inputs +1 or -1), fitting
comfortably in int16.

## Key Property: Dot Product Preservation

WHT is orthogonal: `H * H^T = N * I`

This means: `<WHT(x), WHT(y)> = N × <x, y>`

In 2D (32×32): `<WHT2D(X), WHT2D(Y)> = N² × <X, Y>` where N=32.

**Brute WHT k-NN produces exactly the same rankings as pixel-space
brute k-NN.** The transform is a lossless isometry of the dot product
space.

## Implementation

```c
/* sstt_v2.c:773-784 */
static void wht_1d_inplace(int16_t *data, int n) {
    for (int half = n / 2; half >= 1; half /= 2) {
        for (int i = 0; i < n; i += 2 * half) {
            for (int j = 0; j < half; j++) {
                int16_t a = data[i + j];
                int16_t b = data[i + j + half];
                data[i + j]        = a + b;
                data[i + j + half] = a - b;
            }
        }
    }
}

/* sstt_v2.c:788-805 */
static void wht_2d_full(const int8_t *tern_img, int16_t *out) {
    /* Pad 28×28 → 32×32 with zeros */
    /* Row WHT on each of 32 rows */
    /* Column WHT on each of 32 columns */
}
```

The AVX2 spectral dot product uses `_mm256_madd_epi16` for int16×int16 →
int32 pairwise accumulation, processing 16 values per instruction.

## Results

| Method | Accuracy | Speed |
|--------|----------|-------|
| WHT prototype (pixel) | 56.83% | ~640 ns/query |
| WHT prototype (3-chan) | 62.88% | ~2 μs/query |
| WHT brute k=1 | 95.87% | ~50 sec total |
| WHT brute k=3 | 96.12% | ~50 sec total |

WHT brute k=1 matches pixel-space brute k=1 exactly (95.87%), confirming
the orthogonality. WHT brute k=3 at 96.12% is the best accuracy achieved
in the entire project — a Pareto improvement over the v2 cascade (96.04%).

## What Makes It Novel

1. **Ternary-native** — no multiply in the transform itself, only add/sub.
   The entire pipeline from ternary input through WHT to spectral dot
   product can run on add/subtract/compare hardware.

2. **Exact preservation** — not approximate, not "close enough." The dot
   product ranking is mathematically identical. This is stronger than PCA
   or random projection, which lose information.

3. **Spectral prototypes** — averaging WHT coefficients per class gives
   a spectral centroid classifier. Though less accurate than k-NN
   (56.83% vs 95.87%), it runs in 640 ns/query — sub-microsecond.

## Files

- `sstt_v2.c` lines 769-805: WHT 1D and 2D transforms
- `sstt_v2.c` lines 807-867: bulk computation and prototypes
- `sstt_v2.c` lines 869-921: AVX2 spectral dot product
- `sstt_v2.c` lines 922-1031: Test F (prototypes + brute k-NN)
