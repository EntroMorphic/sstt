# Nodes of Interest: SSTT Step-Changes (Post-DSP)

## Node 1: Mode C is 23% of MNIST errors and trivially addressable
55 of 237 MTFP errors at K=500 are vote dilution: the #1 ranked candidate has the correct class but k=3 majority vote overrides it with two wrong-class neighbors. Switching to k=1 would fix all Mode C errors by definition — but might create new errors where the k=3 consensus was rescuing a wrong #1 candidate.
Why it matters: This is the cheapest possible test. One constant change.

## Node 2: The three channels are weighted equally in vote accumulation
Pixel, h-grad, and v-grad each contribute their full IG-weighted votes to the shared accumulator. But on MNIST, h-grad hurts the dot-product ranking (-0.17pp in channel ablation). Does it also hurt retrieval? If h-grad is noisy for MNIST, its votes may be diluting the candidate pool with wrong-class images.
Why it matters: Channel weighting at the vote level is different from channel weighting at the dot-product level. Both could matter independently.

## Node 3: Fashion profile weight is zero
The val grid search on Fashion set w_p=0 (profile weight). This means the horizontal intensity profile — per-row foreground pixel count — adds zero signal on Fashion. On MNIST it's w_p=16 (small but nonzero). Why? Probably because upper-body garments have similar row-by-row density distributions. The profile can distinguish digits (a 1 is narrow, an 8 is wide) but not garment types (all upper-body garments are roughly the same width per row).
Why it matters: Understanding why features fail on one dataset but work on another reveals what each feature actually captures.

## Node 4: The scoring function is strictly linear
combined = 256*dot_px + 192*dot_vg + λ*(w_c*cent + w_p*prof + w_d*div + w_g*gdiv + w_lbp*lbp + ...)
Every feature is independently weighted and summed. No interaction terms. No conditional logic. But visual classification has natural interactions: centroid only matters if the image has enclosed regions (digits with loops). Divergence only matters if the gradient field has structure (not flat backgrounds).
Why it matters: A nonlinear interaction could extract signal the linear sum misses. But it also risks overfitting the val split.
Tension with simplicity: The system's strength is its simplicity. Adding nonlinear interactions moves toward learned-feature territory.

## Node 5: The system hasn't been tested at K=500 with val/holdout protocol
The 97.63% number at K=500 is from the diagnostic with hardcoded weights. The 97.53% headline is from K=200 with val-derived weights. K=500 with val-derived weights hasn't been run. The improvement from K=200→500 might be different with optimized weights.
Why it matters: There may be a free accuracy gain from simply using K=500 as the default and re-running the val grid search.

## Node 6: Sequential processing does nothing on MNIST but +1.28pp on Fashion
The Bayesian sequential variant with LBP provides massive gains on Fashion but zero on MNIST. On MNIST, the static ranking is already near-optimal. On Fashion, the candidate pool is more heterogeneous and sequential decay helps separate signal from noise.
Why it matters: MNIST might benefit from sequential processing if the candidate pool characteristics change (e.g., with different K or channel weighting).

## Node 7: The 5-trit ensemble (98.42%) was never tested with MTFP trit-flip
The old 5-trit ensemble on the exp branch used 5 different quantization FUNCTIONS (not just different thresholds). Our MTFP ensemble used 5 different threshold PAIRS. These are different experiments. The old ensemble's quantization functions include percentile-based adaptive thresholds which are genuinely different from fixed thresholds.
Why it matters: The comparison was flagged as not direct. The old ensemble might still provide gains with MTFP trit-flip if the quantization functions (not just thresholds) provide genuinely different views.

## Tensions

- **Node 1 vs robustness:** k=1 eliminates Mode C but may create new errors. The k=3 vote is a form of consensus that prevents single-candidate flukes.
- **Node 4 vs Node principles:** Nonlinear interactions add power but complexity. The system's identity is simplicity and transparency.
- **Node 5 vs Node 1:** K=500 and k=1 could interact. More candidates with k=1 might be worse (more single-candidate noise) or better (the top candidate at K=500 is more reliable).
