# Contribution 1: `sign_epi8` as Single-Instruction Ternary Multiply

## Discovery

The AVX2 instruction `_mm256_sign_epi8(a, b)` was designed for conditional
negation: it returns `a` where `b > 0`, `0` where `b == 0`, and `-a` where
`b < 0`. But for operands in `{-1, 0, +1}`, this IS exact multiplication:

    (-1) × (-1) = +1    sign(-1, -1) = -(-1) = +1  ✓
    (-1) ×  (0) =  0    sign(-1,  0) =     0 =  0  ✓
    (-1) × (+1) = -1    sign(-1, +1) =    -1 = -1  ✓
    (+1) × (-1) = -1    sign(+1, -1) = -(+1) = -1  ✓
    (+1) × (+1) = +1    sign(+1, +1) =    +1 = +1  ✓
     (0) × (anything)    sign( 0,  b) =     0 =  0  ✓

One instruction, 32 lanes, 1 cycle throughput. This turns an AVX2 register
into a 32-wide ternary ALU.

## Implementation

```c
/* sstt_geom.c:293-317, sstt_v2.c:264-281 */
static inline int ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < PAD_ITERS; i++) {               /* 25 iterations */
        __m256i va = _mm256_load_si256((__m256i *)(a + i * 32));
        __m256i vb = _mm256_load_si256((__m256i *)(b + i * 32));
        acc = _mm256_add_epi8(acc, _mm256_sign_epi8(va, vb));
    }
    /* int8 accumulation is safe: max |acc| = 25 < 127 */
    /* widen to int32 via: int8 → int16 (cvtepi8_epi16) → int32 (madd_epi16) */
    ...
}
```

## Why It Matters

- **No multiply instruction needed.** The entire ternary dot product pipeline
  is add/subtract/compare — no `_mm256_mullo_epi8` (which doesn't exist) or
  widening to int16 before multiply.

- **Int8 accumulation is safe.** With PADDED=800 and 32 lanes per iteration,
  each lane accumulates at most 25 products of magnitude ≤ 1. The int8
  accumulator (max 127) never overflows.

- **Throughput.** 800 ternary multiply-accumulates in 25 instructions ≈ 25
  cycles ≈ 8 ns at 3 GHz. One MNIST dot product per 8 nanoseconds.

## Files

- `sstt_geom.c` line 304: first use
- `sstt_v2.c` line 273: carried forward
- `sstt_mvp.c`: not used (float-domain simulation)

## Prior Art

`_mm256_sign_epi8` appears in signal processing literature for conditional
negation. We are unaware of any prior use as a ternary multiplication kernel
for machine learning inference.
