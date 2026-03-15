# Contribution 12: SBT/RT Crossover Signal Theory

## Concept

The foundational question of the SSTT project: can ternary computing
justify its density loss through noise resilience?

Two encodings pack 3 trits into a single analog voltage level:

```
SBT (Standard Balanced Ternary):
  level = t0×1 + t1×3 + t2×9    → 27 levels in [-13, +13]
  Capacity: log2(27) = 4.75 bits/symbol
  Voltage spacing: 1/13 = 0.077 (per level)

RT (Redundant Ternary):
  level = t0×1 + t1×2 + t2×3    → 13 levels in [-6, +6]
  Capacity: log2(13) = 3.70 bits/symbol
  Voltage spacing: 1/6 = 0.167 (per level)

Trit baseline:
  Single trit {-1, 0, +1}       → 3 levels
  Capacity: log2(3) = 1.58 bits/symbol
  Voltage spacing: 1.0 (per level)
```

RT trades 22% information capacity for 2.17x wider decision margins.
The hypothesis: under analog Gaussian noise, there exists a "crossover
regime" where RT's noise tolerance compensates for its density loss,
yielding higher *effective throughput* (bits × accuracy).

## Tests

**Test A: Per-symbol error rate sweep** (10M trials per SNR)
- Encode 3 random trits → normalize to [-1,+1] → add Gaussian noise →
  round to nearest valid level → compare to original
- Metric: effective throughput = capacity × (1 - error_rate)
- AVX2 batch processing: 8 trials per SIMD pass

**Test B: Classification accuracy** (10K trials, 16 classes)
- 256 groups of 3 trits per "neuron," 16 neurons
- Encode each group, add noise, decode, accumulate dot product
- Metric: does noisy argmax match true argmax?

## Results

RT outperforms SBT in effective throughput at SNR 10-20 dB — the
"realistic analog hardware range." Below 10 dB, trit dominates (simplest
encoding survives extreme noise). Above 20 dB, SBT dominates (noise is
low enough that density matters more than margins).

The crossover point is a fundamental finding: it defines the noise regime
where redundant ternary encoding is the optimal choice, neither too dense
(SBT) nor too sparse (raw trits).

## What Makes It Novel

1. **Falsification-first methodology** — the code is structured as a
   falsification test. If RT never outperforms SBT, the program reports
   "FALSIFIED" and returns exit code 1. The hypothesis earned its status
   by surviving the test.

2. **Effective throughput metric** — combining raw capacity with error
   rate into a single figure of merit avoids the trap of comparing error
   rates alone (which always favor the sparser encoding) or capacity
   alone (which always favors the denser one).

3. **AVX2 noise simulation** — batch processing 8 encode/noise/decode
   trials in parallel via SIMD, enabling 10M trials per SNR point in
   seconds.

## Files

- `sstt_mvp.c` lines 1-389: complete SBT/RT/Trit signal theory test
- `sstt_mvp.c` lines 110-149: AVX2 batch encode/noise/decode
- `sstt_mvp.c` lines 174-250: Test A (per-symbol error sweep)
- `sstt_mvp.c` lines 263-355: Test B (classification argmax preservation)
- `sstt_mvp.c` lines 363-388: crossover analysis and verdict
