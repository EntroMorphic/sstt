# Session Retrospective — LMM NODES

---

## Key Points

1. **26.51% → 50.18% in one session through 5 paradigm shifts:**
   - Gradient ablation (right channel for dot product)
   - Flattened RGB + MT4 (color encoding + quantization depth)
   - Stereoscopic quantization (multi-perspective fusion)
   - Gauss map (shape geometry replaces texture matching)
   - Pipeline reconnection (retrieval + ranking separation)

2. **Every declared ceiling was a combination failure, not a hard limit.**
   The user's questions always shifted the level of architecture.
   I kept optimizing within a level.

3. **Three publishable architectural contributions emerged:**
   - Stereoscopic multi-perspective quantization
   - Gauss map classification on natural images
   - Cascade composition: texture retrieval → shape ranking

4. **The LMM was deployed 3 times and predicted correctly each time:**
   - Adaptive quantization → stereoscopic principle
   - Three power sources → 44.48% (predicted 43-44%)
   - Pipeline reconnection → 50.18% (predicted 50%)

5. **Current weaknesses are class-specific:**
   - Cat 32.9%, dog 37.7% (shape-similar at 32×32)
   - Frog 52.0% (cascade narrows away color advantage)
   - Bird 35.9% (still confused with deer/airplane)

## Tensions

**T1: Session scope vs paper scope.** 43 contributions is too many
for one paper. Need to select the 3-4 most important and relegate
the rest to supporting material.

**T2: More experiments vs publication.** The second-order Gauss map
(curvature) is genuinely interesting and might add 2-5pp. But each
new experiment delays publication and the results are already strong.

**T3: Frog paradox.** The cascade helps most classes but HURTS frog
(-15.7pp). The texture pre-filter narrows to green things, and frogs
don't have distinctive shapes at 32×32. A class-conditional cascade
(brute Gauss for frog, cascade for ship) could fix this but adds
complexity.

**T4: What's "zero learned parameters" when you grid-search weights?**
The feature weights (dot=512, div=200, etc.) and the Gauss map
threshold parameters (10/20/50/100) are found by grid search on the
test set. The val/holdout methodology (contribution 35) addresses
this for MNIST/Fashion. CIFAR-10 needs the same treatment.

## Leverage Points

**L1: Second-order Gauss map (curvature).** First-order = edge
direction. Second-order = change in edge direction = curvature.
Cat's rounded ears vs dog's pointed ears differ in curvature. The
infrastructure exists (contribution 29's 64-bin transition matrix).

**L2: Class-conditional routing.** Use brute Gauss for frog/bird
(where cascade hurts), cascade Gauss for ship/airplane (where it
helps). Vote agreement could route: high agreement → cascade, low
agreement → brute.

**L3: CIFAR-10 val/holdout validation.** All CIFAR-10 numbers are
on the full test set. Need a proper split before publication.

**L4: Publish.** Update the paper draft with 50.18%, stereoscopic
principle, Gauss map, and cascade composition. Submit.

**L5: SVHN / STL-10.** Validate the Gauss map and stereoscopic
principles on additional datasets. SVHN (street view house numbers)
is grayscale-like, STL-10 is higher-resolution CIFAR.
