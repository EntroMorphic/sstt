# Archived Experiments

Exploratory, superseded, or dead-end experiments preserved for reproducibility. Each file is self-contained and compilable, but none represent current best results.

## Categories

**Superseded by core:**
- `sstt_mvp.c` — earliest prototype
- `sstt_bytepacked.c` — superseded by sstt_bytecascade.c
- `sstt_oracle.c`, `sstt_oracle_v2.c`, `sstt_oracle_v3.c` — routing concepts folded into router_v1
- `sstt_multidot.c` — weight grid search, results absorbed into topo9
- `sstt_dual_hot_map.c` — map fusion, absorbed into router_v1

**Negative results:**
- `sstt_tpca.c` — ternary PCA (-16 to -23pp)
- `sstt_softcascade.c` — soft-prior cascade (counterproductive)
- `sstt_navier_stokes.c` — fluid-flow topology (~10% accuracy)
- `sstt_taylor_jet.c`, `sstt_taylor_specialist.c` — Taylor expansion (failed on CIFAR-10)

**Scaling (inconclusive):**
- `sstt_scale_224.c`, `sstt_scale_hierarchical.c`, `sstt_scale_brute.c`, `sstt_scale_pro.c`
- MNIST-224 mode collapses (11.6% vs 92.67% baseline). Not publishable.

**Feature exploration:**
- `sstt_pentary.c` — 5-level quantization
- `sstt_eigenseries.c` — eigenvalue-ordered block traversal
- `sstt_ensemble.c` — 9 fusion rules compared
- `sstt_series.c` — Bayesian sequential channel fusion
- `sstt_geom.c` — geometry primitives

**Other:**
- Remaining files are one-off experiments or parameter sweeps.
