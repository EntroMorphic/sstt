# Raw Thoughts: SSTT Step-Changes (Post-DSP)

## Stream of Consciousness

I've been pushing on features for two sessions and I'm starting to feel the diminishing returns. LBP was a genuine win (+1.00pp on Fashion). But collar WHT, symmetry, GLCM — all zero. The system is telling me something and I need to listen.

What it's telling me: the representation is exhausted at 28x28. Every feature I can extract from these pixels has been extracted. The ternary topology captures what it can capture. LBP captures what the trits miss. Beyond that there is nothing left in 784 pixels to distinguish a plain T-shirt from a plain shirt. They are literally the same image at this resolution.

So where is there room?

Let me inventory what actually worked and what didn't, across every experiment in this project:

Things that moved the needle:
- IG-weighted voting (+15pp from hot map to cascade)
- Multi-probe Hamming expansion (+2.38pp)
- Multi-channel dot product (+0.34pp from v-grad)
- Topological features (divergence, centroid, profile): 96.28% → 97.27% (+0.99pp)
- Bytepacked encoding (+0.16pp and 3.5x speed — Pareto)
- MTFP trit-flip fix (+0.26pp)
- K=500 over K=200 (+0.09pp on MNIST)
- LBP on Fashion (+1.00pp)
- Bayesian sequential on Fashion (+0.92pp on top of LBP)

Things that didn't:
- Ternary PCA (-16pp)
- Hard-filtered routing (-75pp)
- Taylor jets, Navier-Stokes (~random)
- 224x224 scaling (mode collapse)
- Hybrid L2 re-ranking (-0.03pp)
- Multi-threshold MTFP ensemble (+0.00pp)
- Collar WHT, symmetry, GLCM (+0.00pp)

The pattern in what works: things that add genuinely new information (gradient channels, LBP texture), things that fix broken representations (trit-flip multi-probe), and things that better exploit existing information (Bayesian sequential, MAD weighting).

The pattern in what doesn't: things that operate below the resolution floor (WHT, GLCM at 28x28), things that reduce diversity (L2 filtering, hard gating), and things that add redundant perspectives (multi-threshold ensemble after trit-flip fix).

Where haven't I looked?

1. The k=3 majority vote. Mode C errors are 23% of remaining MNIST errors (55/237). These are images where the #1 ranked candidate IS the correct class but the k=3 vote dilutes it. What if k=1 is better than k=3 on the MTFP system? The original k=3 choice was never re-validated after all the ranking improvements.

2. The vote accumulation weights. IG weights are computed per-position. But three channels share the same 252 positions. What if the channels should have different position-level importance? The per-channel IG weights are independent, but they're not weighted relative to each other. Currently all three channels contribute equally to the vote accumulator.

3. The structural feature weights on Fashion are different from MNIST (grid=3x3, w_c=25, w_p=0, w_d=50, w_g=100, sc=20 vs grid=2x4, w_c=50, w_p=16, w_d=200, w_g=100, sc=50). The profile weight drops to ZERO on Fashion. Why? What about Fashion makes horizontal profile useless? Understanding this might reveal a feature that would work.

4. There are 60,000 training images. The system uses all of them equally. But some training images are better exemplars than others. Prototype selection — using a curated subset of training images — could reduce noise in the candidate pool without reducing diversity.

5. The combined scoring function is linear. All features are summed with weights. What if the right combination is nonlinear? Like: use divergence ONLY when centroid agrees, but ignore it when centroid disagrees. The current MAD weighting is a step toward this, but it modulates all structural features together, not conditionally.

## Questions Arising

- What's the k=1 accuracy on MTFP? How many Mode C errors would be eliminated?
- What are the per-channel IG weight distributions? Are some channels contributing noise?
- Why does profile weight go to zero on Fashion?
- Could per-channel vote scaling (weight the 3 channels differently) help?
- Is there a nonlinear feature interaction the linear scoring misses?

## First Instincts

- The k=1 vs k=3 test is a one-line change. Do it first.
- Per-channel vote scaling is a simple experiment with a small grid search.
- The nonlinear interaction idea is interesting but speculative.
- Prototype selection is a research project, not a quick experiment.
