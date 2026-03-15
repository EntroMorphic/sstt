# Contribution 3: TST Butterfly Cascade

## Concept

The Ternary Shape Transform (TST) is a multi-resolution cascade filter
inspired by the FFT butterfly. It converts O(n*d) brute-force search into
O(d*log_3(d) + K*d) by exploiting spatial coherence across scales.

The key insight: if two images share the same 3-trit block at position k,
they probably share the same 9-trit superblock (3 adjacent blocks). If they
share the superblock, they likely share the 27-trit megablock. This is the
AND-cascade: each level requires ALL sub-blocks to match.

## Architecture

```
L0: 261 blocks of 3 trits     (weight  1)   base resolution
L1:  87 superblocks of 9 trits (weight  3) = 3 adjacent L0 ANDed
L2:  29 megablocks of 27 trits (weight  9) = 3 adjacent L1 ANDed
L3:   9 hyperblocks of 81 trits(weight 27) = 3 adjacent L2 ANDed
L4:   3 regions of 243 trits   (weight 81) = 3 adjacent L3 ANDed

Score = L0×1 + L1×3 + L2×9 + L3×27 + L4×81  ∈ [0, 1269]
```

The weights are powers of 3 (twiddle factors in the ternary butterfly
analogy). Higher-level matches are exponentially more valuable because
they represent wider spatial coherence — matching across 243 consecutive
trits means the images share extended structure, not just local texture.

## Pipeline

```
Stage 1: Vote via inverted index (block match → image IDs)
Stage 2: Multi-resolution butterfly scoring of top-1000 candidates
Stage 3: Ternary dot product refinement on top-K survivors

vote → butterfly_score → dot_product
```

## Implementation

```c
/* sstt_geom.c:877-931 */
static inline int32_t tst_multi_res_score(const uint8_t *qsig,
                                           const uint8_t *csig) {
    /* L0: AVX2 byte-compare of 261 block signatures */
    uint8_t match[SIG_PAD];
    for (int i = 0; i < SIG_PAD; i += 32) {
        __m256i q = _mm256_load_si256((qsig + i));
        __m256i c = _mm256_load_si256((csig + i));
        __m256i m = _mm256_cmpeq_epi8(q, c);
        _mm256_store_si256((match + i), m);
    }

    /* L1-L4: cascade AND of triplets */
    uint8_t l1m[87];
    for (int j = 0; j < 87; j++)
        l1m[j] = match[3*j] & match[3*j+1] & match[3*j+2];
    /* ... cascade continues through L2, L3, L4 ... */

    return l0*1 + l1*3 + l2*9 + l3*27 + l4*81;
}
```

## Why It Matters

1. **Zero-weight cascade** — the butterfly AND-gate is free: byte AND on
   already-computed match vectors. No additional memory, no computation
   beyond what the vote stage already produced.
2. **Geometric scoring** — the 3^k twiddle weights automatically prioritize
   global shape over local texture. Two images matching at L4 (243 trits)
   are overwhelmingly the same digit.
3. **FFT analogy** — just as FFT decomposes signals into frequency bands,
   TST decomposes ternary patterns into spatial coherence bands. L0 is
   "high frequency" (local), L4 is "low frequency" (global shape).

## Performance

- Multi-resolution only (no dot products): ~90% accuracy
- TST K2=50 cascade: 94.67%
- TST K2=200 cascade: 95.47%
- Speedup over brute 1-NN: ~100-300x

## Files

- `sstt_geom.c` lines 853-1038: TST hierarchy, scoring, and cascade test
- Not carried to `sstt_v2.c` — replaced by IG-weighted multi-probe cascade
