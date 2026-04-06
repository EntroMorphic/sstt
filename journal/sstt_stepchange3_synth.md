# Synthesis: SSTT Step-Changes (Post-DSP)

## The cut

Feature engineering is done. The remaining gains are in mechanism tuning: how many candidates (K), how many votes (k), and how channels are balanced.

## Experiment: K × k × channel-weight grid

One experiment file, one run, 30 configurations.

**Grid:**
- K ∈ {200, 500} (candidate pool size)
- k ∈ {1, 3, 5} (majority vote depth)
- h-grad scale α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (h-grad vote multiplier)

For each configuration: run the full MTFP pipeline on all 10K MNIST test images with hardcoded topo9 weights. Report accuracy.

**Expected outcomes:**

- If k=1 at K=500 beats k=3 at K=200: the structural ranker's top candidate is now reliable enough to trust without consensus. This would be a paradigm shift — the system was built around k=3 vote.
- If α=0.5 beats α=1.0 for h-grad: the h-grad channel is adding retrieval noise on MNIST (consistent with the -0.17pp dot-product ablation).
- If the best (K, k, α) combination beats 97.63%: there's free accuracy from mechanism tuning.

**What this does NOT address:**
- Fashion-MNIST (would need a separate run with Fashion weights)
- New features (exhausted at 28x28)
- Adaptive quantization functions (deferred — larger experiment)
- Paper update (deferred until experimental dust settles)

## Success criteria

- [ ] Identify the optimal (K, k, α) triple for MNIST
- [ ] Measure whether Mode C errors are reducible via k adjustment
- [ ] Determine whether h-grad channel scaling improves retrieval
- [ ] If any configuration exceeds 97.63%, that becomes the new default

## If nothing improves

Then 97.63% (K=500, k=3, α=1.0) is the MTFP ceiling on MNIST with the current ranking features. The system has been fully explored within the ternary block-signature architecture. The remaining 237 errors are genuinely hard cases — topologically similar digit pairs that the representation cannot distinguish. This is an honest and complete result.
