# Nodes of Interest: SSTT Step-Changes (Post-MTFP)

## Node 1: The error decomposition is stale
All Mode A/B/C numbers are from the old topo9 system (273 errors). MTFP has 247 errors. The 26 fixed errors could be from any mode. If the trit-flip fix primarily fixed Mode B (ranking inversions) — which is likely since better multi-probe produces better candidates for ranking — then Mode A might still be 7 and the ceiling is unchanged. But if some Mode A failures were caused by the wrong-topology multi-probe missing valid neighbors, some retrieval failures may have been fixed too.
Why it matters: Can't plan the next move without knowing where the current 247 errors land.

## Node 2: MTFP K-sensitivity is untested
The K-sensitivity data (96.59% at K=50 → 97.61% at K=1000) is from the old joint-index system. Per-channel indexing changes the vote distribution — three independent channels might produce a different K curve. The MTFP system used K=200 by default. If K=500 gains another +0.2-0.3pp like it did for topo9, that's a free improvement.
Why it matters: Possibly the easiest accuracy gain available — just change a constant.

## Node 3: Fashion-MNIST is untested with MTFP
The trit-flip fix corrected a topology error that was present on all datasets. Fashion-MNIST has more intra-class diversity than MNIST (a t-shirt can look very different from another t-shirt). Wrong-topology multi-probe would have been more damaging on Fashion because the "near-miss" neighbors it found were more often garbage. The fix might gain more than +0.26pp on Fashion.
Why it matters: Cross-dataset evidence strengthens the MTFP contribution. And Fashion is where the system most needs improvement (85.68% vs competitors at 87-88%).

## Node 4: MTFP + multi-threshold ensemble is unexplored
The 5-trit ensemble (98.42% on the old branch) used 5 quantization thresholds. MTFP uses per-channel indexing with trit-flip multi-probe. These are orthogonal improvements — different axes of the same system. Nobody has combined them.
Why it matters: If they stack, this could be the largest single accuracy jump available. 97.53% + the ensemble delta could approach 98.5%+.
Tension with Node 1: Can't evaluate ensemble gains without knowing the error distribution first.

## Node 5: The quantization thresholds are arbitrary
85/170 divide 0-255 into equal thirds. MNIST pixels are bimodally distributed (concentrated at 0 and 255). The thresholds were never optimized — they were chosen for simplicity. A threshold sweep (e.g., 70-100 for low, 155-185 for high) might find a better operating point for the single-threshold system.
Why it matters: Every downstream component depends on the quantization. A better threshold could lift everything — retrieval, ranking, features — simultaneously.
Tension with Node 4: Multi-threshold ensemble sidesteps single-threshold optimization by using several. If the ensemble works, single-threshold optimization is moot.

## Node 6: The paper needs updating
The paper draft still says 97.27% throughout. It describes bytepacked joint encoding as the primary method. It doesn't mention MTFP, per-channel indexing, or trit-flip multi-probe. The paper is now two step-changes behind the code.
Why it matters: The paper is the publication vehicle. It needs to reflect the current system.

## Node 7: The structural features haven't been revisited
Divergence, centroid, profile, grid-div, MAD weighting — all designed for the old system. The MTFP per-channel indexing changes what candidates the ranker sees. The current feature weights (w_c=50, w_p=16, w_d=200, w_g=100, sc=50) were grid-searched on the old system and confirmed identical on MTFP. But are these features themselves still optimal? New features designed for the per-channel retrieval distribution might add signal.
Why it matters: The ranker is the bottleneck (97% of errors). Any improvement must come from ranking.
Tension with Node 1: Need error decomposition before knowing which ranking failures to target.

## Tensions

- **Node 1 vs everything:** Can't plan without diagnostics. But diagnostics take time.
- **Node 4 vs Node 5:** Multi-threshold ensemble vs single-threshold optimization. Do we diversify retrieval or sharpen it?
- **Node 6 vs experimental work:** Paper update is necessary but not where the next accuracy gain comes from.
