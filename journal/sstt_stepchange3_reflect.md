# Reflections: SSTT Step-Changes (Post-DSP)

## Core Insight

The feature engineering phase is over. Two sessions of DSP experiments confirmed: LBP is the last feature with signal at 28x28, and the remaining error is split between ranking inversions (Mode B: 77%) that need better discrimination and vote dilutions (Mode C: 23%) that need a better voting mechanism. These are two distinct problems requiring different interventions.

## Why three times on the feature exhaustion

Why did LBP work but nothing else? LBP compares each pixel to its immediate 8 neighbors — it operates at the resolution floor (1 pixel). Haar, WHT, GLCM, and symmetry all aggregate over larger windows, and at 28x28 those windows span the entire distinguishing feature (a 3-4 pixel collar). LBP works because it's the only feature whose operating scale matches the remaining signal scale.

Why can't more features be invented? Because the information content of a 28x28 uint8 image is bounded. With 784 pixels at 8 bits each, there are at most 6,272 bits of raw information. The ternary encoding preserves ~1,240 bits (784 trits × 1.58 bits). LBP operates on the raw 6,272 bits. Together, the ternary pipeline + LBP have harvested the structural and textural information available. Any new feature would be a linear combination of already-captured patterns.

Why is this the right place to stop on features? Because the negative results are categorical. Three independent DSP approaches (spectral, symmetry, co-occurrence) all yielded exactly zero. This is not "we haven't found the right feature yet" — this is "there is no right feature at this resolution."

## Resolved Tensions

**Node 1 vs robustness (k=1 vs k=3):** This tension resolves with data. Run both and measure. If k=1 eliminates 55 Mode C errors but creates N new ones, the net is 55-N. If N < 55, k=1 wins. If N > 55, k=3's consensus value is confirmed. Either way we learn something. Additionally, test k=1 at both K=200 and K=500 to check for interaction.

**Node 4 vs simplicity (nonlinear scoring):** This tension resolves against complexity. The system's identity is transparency and simplicity. Every prediction traces to specific training images. Adding nonlinear feature interactions makes the scoring function harder to interpret without guaranteed gains. The right move is to exhaust the simple interventions first (k adjustment, channel weighting, K optimization). If those don't close the gap, the system has reached its ceiling honestly.

**Node 5 vs Node 1 (K and k interaction):** These should be tested together. A 2D grid: K ∈ {200, 500}, k ∈ {1, 3, 5}. Six combinations, all on the MTFP diagnostic infrastructure. This is a 15-minute compute job.

## Challenged Assumptions

**"k=3 is optimal."** This was chosen early and never revisited after the ranking improvements. The topo9 structural features changed the quality of the top-ranked candidates. If the #1 candidate is now more reliable than it was in the cascade era, k=1 may outperform k=3.

**"All three channels should contribute equally to voting."** The per-channel IG weights handle within-channel importance, but inter-channel relative weighting is currently 1:1:1. If h-grad is noisy on MNIST, scaling its votes by 0.5 might reduce Mode B errors by producing cleaner candidate pools.

**"The 5-trit ensemble's gains were from compensating for broken multi-probe."** This is a hypothesis we stated honestly in the negative results doc. The old ensemble used percentile-based adaptive quantization functions that are genuinely different from fixed thresholds. MTFP + adaptive quantization (not just different thresholds) hasn't been tested. This remains an open thread.

## What I Now Understand

Three cheap experiments, testable in one combined run:

1. **K×k grid** — K ∈ {200, 500}, k ∈ {1, 3, 5}. Six combinations. Measures whether the k=3 convention is still optimal and how K and k interact.

2. **Per-channel vote scaling** — multiply h-grad votes by α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} while keeping pixel and v-grad at 1.0. Five values. Measures whether h-grad retrieval helps or hurts.

3. **Both combined** — the best K, k, and channel scaling from (1) and (2).

These are all modifications to the MTFP diagnostic, not new features. They tune the existing mechanism rather than adding to it. Total grid: 6 × 5 = 30 configurations. Cheap.

The remaining open question — MTFP + adaptive quantization functions — is a larger experiment that should wait until the cheap interventions are exhausted.

## Remaining Questions

- Is k=1 better than k=3 on MTFP at K=500? The structural ranker has improved since k=3 was chosen.
- Does scaling h-grad votes improve MNIST retrieval quality?
- What's the optimal (K, k) pair for MTFP?
- Would MTFP + percentile-based adaptive quantization (the old 5-trit functions) provide gains the threshold-only ensemble couldn't?
