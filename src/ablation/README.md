# Topological Ranking Ablation Series

Ten-file incremental ablation from 96.28% to 97.27% MNIST accuracy.

Each file adds one feature to the bytepacked cascade baseline and measures its isolated contribution. The series proves that topological features (gradient divergence, enclosed centroid, horizontal profile) provide genuine signal beyond pixel-space dot products.

| File | Addition | MNIST |
|------|----------|-------|
| sstt_topo.c | Divergence + centroid + profile (baseline topo) | 97.11% |
| sstt_topo2.c | Spatial divergence representations | exploratory |
| sstt_topo3.c | Hard topological filtering | 96.61% |
| sstt_topo4.c | Kalman-adaptive MAD weighting | key contribution |
| sstt_topo5.c | Edge-masked divergence | incremental |
| sstt_topo6.c | Dead zone + bucketed divergence | incremental |
| sstt_topo7.c | Chirality + quadrant decomposition | key contribution |
| sstt_topo8.c | Grid resolution sweep | superseded by topo7 |
| sstt_topo9.c | Sequential Bayesian ranking | 97.27% |
| sstt_topo9_val_5trit.c | 5-eye ensemble variant | 98.42% |
| sstt_topo9_val_5trit_grid.c | 5-eye with full grid search | validation |

Key findings: topo4 (adaptive weighting) and topo7 (spatial decomposition) are the load-bearing contributions. topo9 synthesizes all components with sequential Bayesian processing.
