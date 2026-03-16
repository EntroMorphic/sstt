# Contribution 2: The Hot Map — Naive Bayes via Ternary Block Frequencies

## Concept

A 60 MB inverted index (60K images × 261 blocks × 4 bytes per ID) is
aggregated into a 451 KB frequency table:

    hot_map[block_position][block_value][class]  →  uint32 count

Classification becomes pure table lookup: for each of 261 blocks in the
query, read the 10-class frequency vector, accumulate. Argmax of the
accumulated vector is the prediction. No per-image tracking. No branching
on image IDs. No distance computation.

## Architecture

```
Training:
  For each image i with label c:
    For each block k in [0, 260]:
      bv = signature[i][k]          // block value in [0, 26]
      hot_map[k][bv][c] += 1        // increment class counter

Inference:
  scores[0..9] = 0
  For each block k in [0, 260]:
    bv = query_sig[k]
    scores[:] += hot_map[k][bv][:]  // 10-class vector add
  return argmax(scores)
```

## Memory Layout

- 261 block positions × 27 block values × 16 classes (10 padded to 16)
- = 261 × 27 × 16 × 4 bytes = 451,008 bytes ≈ 451 KB
- Fits in L2 cache (typically 256 KB – 1 MB per core)
- CLS_PAD=16 ensures each class vector fills exactly one 256-bit register

## AVX2 Implementation

```c
/* sstt_geom.c:782-802 */
static int hot_classify(const uint8_t *qsig) {
    __m256i acc0 = _mm256_setzero_si256();  /* classes 0-7  */
    __m256i acc1 = _mm256_setzero_si256();  /* classes 8-15 */
    for (int k = 0; k < N_BLOCKS; k++) {
        uint8_t bv = qsig[k];
        if (bv == 0) continue;  /* skip background */
        const uint32_t *row = hot_map[k][bv];
        acc0 = _mm256_add_epi32(acc0, _mm256_load_si256((__m256i *)row));
        acc1 = _mm256_add_epi32(acc1, _mm256_load_si256((__m256i *)(row + 8)));
    }
    /* scalar argmax over 10 classes */
    ...
}
```

## Performance

- **Latency:** ~0.5–1.2 μs per query (after L2 warmup)
- **Throughput:** ~1M classifications per second
- **Accuracy:** 71.12% (v1, 261 1D blocks), 71.35% (v2, 252 2D blocks)
- **Size:** 451 KB (v1) or 425 KB (v2, 252 blocks)

## What Makes It Novel

The hot map is a naive Bayes classifier with ternary block features.
The contribution is the specific physical realization:

1. **Fixed-size lookup table** — no dynamic allocation, no hash tables
2. **L2-cache-resident** — the entire "model" lives in fast cache
3. **Zero branching** — every block contributes the same way (one load + add)
4. **Background skip** — the one branch (skip bv==0) avoids the dominant
   uninformative bucket, saving ~60% of loads

## Files

- `sstt_geom.c` lines 744-851: build + classify + test
- `sstt_v2.c` lines 395-482: multi-channel version (3 × 425 KB)
