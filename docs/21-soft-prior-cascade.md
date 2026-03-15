# Contribution 21: Soft-Prior Cascade — Information Perihelion

## Motivation

The SSTT project has two complementary classifiers:

| Method | Accuracy | Speed | Mechanism |
|--------|----------|-------|-----------|
| Transition Bayesian | 84.69% | 2 μs | Pure table lookup, Bayesian posterior |
| Full 3-chan cascade | 96.12% | ~300 sec | Per-image voting + dot product + k-NN |

The Bayesian reaches **information perihelion** (closest approach to the
truth) in microseconds. The cascade achieves higher accuracy but spends
300 seconds recomputing from scratch. The question: can the Bayesian's
orbit guide the cascade to converge faster?

A failed attempt (sstt_perihelion.c) HARD-FILTERED the cascade to the
Bayesian's top-N classes. This killed accuracy (21-50%) because 15% of
images have the wrong top class, and hard filtering is irrecoverable.

## The Fix: Soft Priors

Instead of filtering, the Bayesian posterior **boosts** cascade votes:

```
votes[tid] += w × boost[label[tid]]
```

Where `boost[c] = 1 + α × softmax(posterior, T)[c] × 256`.

Key properties:
- `boost[c] >= 1` always — no class is excluded
- High-posterior classes get a vote head start
- The cascade retains full ability to override the Bayesian
- When the Bayesian is right (85% of the time), the cascade converges
  faster because the correct class's training images accumulate votes
  more quickly
- When the Bayesian is wrong (15%), the cascade's raw vote volume
  can overcome the boost

## Architecture

```
Test image
  ├─→ 5-channel Bayesian (2μs)
  │     pixel + hgrad + vgrad blocks (interleaved)
  │     + h-transition + v-transition hot maps
  │     → posterior[10]
  │
  ├─→ Posterior → boost[10] via temperature-scaled softmax
  │     boost[c] = 1 + α × softmax(posterior, T)[c] × 256
  │
  ├─→ Pre-expand: img_boost[tid] = boost[label[tid]] for all 60K
  │     (240KB, L2-resident, avoids per-vote label lookup)
  │
  └─→ Boosted 3-chan IG multi-probe cascade:
        for each block k, channel ch:
          exact:    votes[tid] += w × img_boost[tid]
          neighbor: votes[tid] += w_half × img_boost[tid]
        → top-K → dot product → k=3 vote
```

## Parameters

- **α (alpha)**: Boost strength. α=0 is pure cascade (no prior). Higher
  α gives the Bayesian more influence. Sweep: {1, 2, 4, 8, 16}.

- **T (temperature)**: Flattens the peaked Bayesian posterior. The raw
  posterior after hundreds of multiplicative updates is extremely peaked
  (winner ≈ 1.0, rest ≈ 0.0). Temperature scaling:
  `scaled[c] = pow(posterior[c], 1/T) / Z`. High T flattens toward
  uniform; low T preserves peaks. Sweep: {1.0, 3.0, 10.0}.

## Red-Team Analysis

1. **Prior wrong 15% of the time**: The boost is additive, not
   multiplicative with the vote. Even with boost=4097 (α=16) on the
   wrong class, the cascade processes 252 blocks × 3 channels × 7
   probes = 5292 vote events. If the true class matches at most
   positions, its raw vote volume overcomes the wrong class's boost.

2. **Overflow**: Max boost = 1 + 16 × 256 = 4097. Max IG = 16.
   Max per block per channel = 16 × 4097 × 7 = 459K. Over 756
   block-channels = 347M. uint32 max = 4.3B. Safe.

3. **Speed**: One extra integer multiply per vote in the inner loop.
   The img_boost array (240KB) is L2-resident. Expected ~20-30%
   slower than pure cascade.

4. **Top-K histogram**: With boost, max vote can be ~347M. A counting
   sort histogram of that size would be ~1.4GB. The implementation
   caps at 1M buckets and falls back to linear scan for larger ranges.

## Bug Fix

sstt_v2.c line 1297 uses `if (bv == 0) continue;` for ALL channels.
This applies pixel background (BG_PIXEL=0) to gradient channels, which
should use BG_GRAD=13. The soft cascade fixes this with a per-channel
background array: `chan_bg[3] = {0, 13, 13}`.

This means the pure cascade baseline (α=0) in this file may differ
from sstt_v2.c's reported accuracy. Any improvement in the α=0 baseline
over sstt_v2's 96.04% (cascade) or 96.12% (WHT brute) is partially
attributable to the background fix.

## Files

- `sstt_softcascade.c`: Self-contained implementation (~800 lines)
- `Makefile`: `make sstt_softcascade`
