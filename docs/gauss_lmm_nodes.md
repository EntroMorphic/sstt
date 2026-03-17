# Gauss Map + Full Architecture — LMM NODES

---

## Key Points

1. **Every declared ceiling was a failure to COMBINE, not a hard limit.**
   26→42 (combining block features), 42→44 (combining perspectives),
   44→48 (switching to geometry). The pattern: when stacking within
   one system stalls, CHANGE THE SYSTEM and stack the systems.

2. **The two systems use each other's weaknesses as strengths.**
   - Block system: excellent retrieval (98% recall), poor ranking
     (classifies on background texture)
   - Gauss map: excellent ranking (classifies on shape), no retrieval
     (brute-force O(n))

3. **Sequential composition > score fusion.** The fusion experiment
   (contribution 42) failed because it ADDED scores. The right
   architecture is SEQUENTIAL: use each system for what it does best.

4. **11 validated techniques are available** (see RAW inventory).
   Not all have been combined with the Gauss map.

5. **The Gauss map operates on grayscale only.** Per-channel RGB
   Gauss maps, finer grids, and flattened-RGB Gauss maps are untested.

## Tensions

**T1: Retrieval filtering helps or hurts Gauss map?**
Texture-filtered candidates (top-200 from block voting) contain both
correct-class and confusable-class images. If the Gauss map can
distinguish deer from frog shapes, this is BETTER than brute force
(no irrelevant candidates). If the filter excludes correct neighbors,
it's worse (2% miss rate from retrieval).

**T2: Speed vs accuracy in the Gauss map.**
Brute kNN over 50K images is O(n) per query. Using inverted-index
retrieval reduces this to 200 comparisons — 250× faster. But the
200 might not include the best Gauss map neighbors.

**T3: Grayscale vs RGB Gauss maps.**
The current Gauss map uses 4 channels (gray+R+G+B) for global but
only grayscale for the grid version (which is the best). RGB grid
Gauss maps would have 4× more features — potentially more
discriminative but also noisier.

## Leverage Points

**L1: Vote → Gauss Rank cascade.** Block voting retrieves top-200 →
Gauss map L1 ranks them → k=1 argmax. This uses retrieval from the
block system and ranking from the Gauss map. The original SSTT
cascade architecture, with a different ranking function.

**L2: RGB grid Gauss map.** Extend the 48.31% grid Gauss map to
per-channel RGB. 4 channels × 16 regions × 45 bins = 2880 features.
Standalone first, then as ranking in L1.

**L3: Finer grid.** 8×8 grid (4×4 pixel regions) for more spatial
resolution. The 4×4 grid captures "top-left vs bottom-right". 8×8
captures "upper-left quadrant vs lower-left quadrant."

**L4: Stereoscopic Gauss map.** Multiple gradient threshold parameters
(the Gauss map uses 10/20/50/100 for magnitude buckets — try different
settings and fuse posteriors).

**L5: Three-eye stereo voting → Gauss map ranking.** The best
retrieval (stereo 3-eye) feeding the best ranking (Gauss map).
