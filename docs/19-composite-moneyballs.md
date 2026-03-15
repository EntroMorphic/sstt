# Contribution 19: Unexplored Composite Solutions

## The Accuracy Gap

The SSTT pipeline spans a wide accuracy range:

```
Hot map (3-channel):           73.23%
Full cascade (K=500, k=3):    96.04%
Gap:                           22.81 pp
```

The cascade achieves its accuracy through a series of additive
improvements, each with a distinct computational cost:

| Enhancement | Accuracy gain | Mechanism | Cost |
|-------------|--------------|-----------|------|
| IG weighting | +15.55 pp | Multiply block votes by class-discriminative weight | Per-block multiply (bakeable) |
| Multi-probe Hamming-1 | +2.38 pp | Also look up 6 Hamming-1 block neighbors | 7× more lookups per block |
| Top-K dot + k=3 vote | +4.88 pp | Select K candidates, rank by dot product | Per-image dot products |

The question is: how much of this gap can be closed with table-only
methods that compose with the fused kernel (contribution #14)?

## Composite 1: Pre-Baked IG Hot Map

**Idea:** Multiply the IG weight into the hot map at training time.
Instead of `hot[k][bv][c] = count`, store
`hot[k][bv][c] = count * ig_weight[k]`.

**Why it works:** The cascade's IG-weighted vote currently computes:

```
for each block k:
    score[c] += ig_weight[k] * hot[k][bv][c]
```

Since `ig_weight[k]` is a constant for each block position, it can be
pre-multiplied into the table entries at training time. The inference
path becomes identical to the current unweighted hot map — same code,
same latency, same number of lookups.

**Expected accuracy:** ~88% (73.23% + 15.55 pp from IG weighting).

**Cost:** Zero additional inference cost. The hot map is 1.3 MB in
either case. The multiplication happens once at training time. The
fused kernel (`sstt_fused_c.c`) requires no code changes — the pre-baked
values are simply different table entries.

**Caveat:** The IG weights are uint16 values in [0, 1000]. Multiplying
by counts (uint32, max ~6000 per bucket) may require uint64 accumulation
or scaled-down weights. A practical approach: quantize IG weights to
[0, 255] and use uint32 accumulation (max product: 255 × 6000 = 1.53M,
well within uint32 range).

## Composite 2: Fused Multi-Probe Hot Map

**Idea:** For each query block, look up not just the exact block value
but also its 6 Hamming-1 neighbors at half weight.

```
for each block k:
    bv = query_sig[k]
    acc[:] += hot[k][bv][:]                           // full weight
    for each bv' in Hamming1(bv):                     // 6 neighbors
        acc[:] += hot[k][bv'][:] >> 1                 // half weight
```

**Why it works:** Multi-probe voting (contribution #8) achieves +2.38 pp
by accounting for single-trit quantization noise. A block value that
differs from the stored template by one trit is still a likely match.
The Hamming-1 neighbors for any block value are precomputed (there are
exactly 6: each of the 3 trit positions can take 2 alternative values).

**Expected accuracy:** ~91% (88% from IG pre-bake + 2.38 pp from
multi-probe).

**Cost:** 7× more hot map lookups per block (1 exact + 6 neighbors).
Each lookup is still a pure table read + SIMD add. No per-image
computation, no dot products, no candidate lists. The pipeline remains
fully fused.

**Working set impact:** The hot map is read 7× more frequently but the
table itself is the same 1.3 MB. L2 cache hit rate may decrease
slightly because access patterns become less sequential (jumping between
block values within the same position), but the entries for the 6
Hamming-1 neighbors of any value are within the same 1728-byte
position stride — likely in the same few cache lines.

## Composite 3: Cross-Channel Joint Blocks

**Idea:** Instead of independent hot maps per channel, encode all three
channels at the same position into a single block value:

```
joint_bv = pixel_trit * 9 + hgrad_trit * 3 + vgrad_trit + 13
```

This produces a single block value in [0, 26] that captures the joint
(pixel, h-gradient, v-gradient) state at each trit position. Then group
three consecutive joint values into one block of 27^1 = 27 values (same
as current encoding).

**Why it works:** The current 3-channel hot map treats each channel
independently — it cannot capture correlations like "dark pixel AND
strong horizontal edge." The joint encoding captures these inter-channel
correlations directly.

**Trade-off:** The current system has 3 channels × 252 blocks × 27
values = 20,412 table entries. The joint encoding has 252 blocks × 27
values = 6,804 entries — fewer entries, but each entry encodes a
richer pattern.

Alternatively, use a larger block value space: the joint trit at each
position takes 27 values (3^3), so a 3-position block has 27^3 = 19,683
possible values. This requires a larger hot map:

```
252 blocks × 19,683 values × 16 classes × 4 bytes ≈ 319 MB
```

This is too large for cache. A hybrid approach: use the 27-value block
encoding (one joint trit per position, 3 positions per block) which
stays at the current table size but captures some inter-channel
correlation.

**Expected accuracy:** Unknown — depends on how much inter-channel
correlation exists. On MNIST, the channels are moderately correlated
(a strong pixel value implies a specific gradient direction). The
potential gain is likely 1–3 pp beyond what independent channels achieve.

## Composition and Ordering

All three composites are compatible with each other and with the fused
kernel architecture. They compose naturally:

```
Composite 1 (IG pre-bake):     changes training, not inference
Composite 2 (multi-probe):     changes inner loop, not data structures
Composite 3 (joint encoding):  changes block encoding, not pipeline
```

**Recommended implementation order:**

1. **IG pre-bake first.** Guaranteed +15 pp with zero inference cost.
   Implement as a post-processing step on the existing hot map. Validate
   by comparing to the IG-weighted vote accuracy in sstt_v2.c Test C.

2. **Multi-probe second.** The Hamming-1 neighbor table is already
   computed in sstt_v2.c. Add 6 additional lookups per block in the
   fused kernel inner loop. Validate against sstt_v2.c Test D.

3. **Joint encoding last.** Requires redesigning the block encoding,
   which affects all pipeline stages. Higher implementation risk.
   Validate by comparing joint vs independent channel accuracy on the
   3-channel hot map.

## Expected Combined Accuracy

```
Baseline (3-channel hot map):      73.23%
+ IG pre-bake:                     ~88%      (zero inference cost)
+ Multi-probe:                     ~91%      (7× lookups, still pure table)
+ Joint encoding:                  ~92-94%   (speculative)
```

The remaining ~2-4 pp to reach 96% likely requires per-image
computation (dot products, k-NN voting) that cannot be eliminated
through table-only methods. This represents the fundamental limit of
pure addressing: the hot map cannot distinguish between two training
images that produce identical block signatures but belong to different
classes.

## Files

- `sstt_v2.c` lines 1098-1185: IG weight computation (Test C)
- `sstt_v2.c` lines 1186-1257: Multi-probe voting (Test D)
- `sstt_v2.c` lines 1258-1435: Full cascade (Test E)
- `sstt_fused_c.c`: Fused kernel to be modified for composites
- `sstt_fused.h`: Constants (background values, block layout)
