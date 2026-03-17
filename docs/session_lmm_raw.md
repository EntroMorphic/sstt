# Session Retrospective — LMM RAW

Unfiltered thinking on what happened in this session and what it means.

---

This session started with an audit and ended with 50% on CIFAR-10. The
trajectory was not planned. Each step emerged from the previous one's
data. Let me dump what I actually think happened.

The session had phases:

Phase 1: Audit and cleanup (contributions 34-36). Independent audit,
val/holdout validation, delta map. This was necessary housekeeping.
The important output wasn't the numbers — it was the methodology
correction (sequential on MNIST = noise) and the honest validation
(97.27%/85.68% publication-ready).

Phase 2: CIFAR-10 boundary test (contributions 37-38). The first
CIFAR-10 experiment got 26.51%. I immediately declared this a
"boundary" and "honest limitation." The user pushed. Every technique
from MNIST was applied: gradient ablation, flattened RGB, MT4, topo
features, Bayesian prior. By contribution 38 we were at 42%.

Phase 3: The representation wall (numerous experiments around 41-42%).
I tried MoE (+0.12pp), Kalman (+0pp), label propagation (-5.9pp),
quantized MAD (+0pp), multi-scale (+0.62pp). Each gave <1pp. I kept
saying "ceiling" and "diminishing returns." The user kept pushing.

Phase 4: The stereoscopic breakthrough. The user asked about adaptive
quantization. The LMM analysis identified brightness as the #1
confound and the tension: brightness is signal for some classes, noise
for others. The stereoscopic principle emerged: run BOTH quantizations,
fuse posteriors. 41.18% → 44.48% with MT4 stack.

Phase 5: The Gauss map paradigm shift. The user asked: "What about
converting to spheres like we did for MNIST?" I should have thought of
this earlier — the Gauss map was already in the MNIST arsenal
(contributions 29, sstt_gauss_map.c). But I'd categorized it as a
"marginal feature" (+0.05pp on MNIST) and never considered it as the
PRIMARY classification mechanism. The user saw what I missed: the
sphere IS the classifier, not a feature for the classifier.

48.31% on brute kNN. Then the LMM identified: "the Gauss map broke
the pipeline — restore RETRIEVE → RANK with the best of each." The
cascade: stereo vote → Gauss rank → 50.18%.

WHAT I KEEP GETTING WRONG:

I declare ceilings when I run out of ideas within the CURRENT FRAMEWORK.
The user then asks a question that changes the framework. The question
is never "how do we get +0.5pp?" It's always structural:

- "Why do we need ternary at all?" (→ MT4)
- "Can we see it from multiple perspectives?" (→ stereoscopic)
- "Can we map it to spheres?" (→ Gauss map)
- "Are we using both systems for what they're best at?" (→ cascade)

Each question shifts the LEVEL of the architecture, not the details.
I keep optimizing within a level. The user keeps asking what the next
level is.

WHAT THE DATA SAYS ABOUT NEXT STEPS:

The 50.18% result has specific weaknesses:
- Bird: 35.9% (down from 38.6% brute Gauss)
- Cat: 32.9% (still the worst)
- Dog: 37.7% (still confused with cat)
- Frog: 52.0% (down from 67.7% — the cascade hurts frog)

The cascade helps classes where texture and shape are BOTH informative
(ship: 68.7%, airplane: 62.8%). It hurts classes where the texture
pre-filter narrows to confusable candidates (frog drops because all
green things are in the top-200, and frogs don't have distinctive
shapes at 32×32).

What would the NEXT level be? Not more features, not more
perspectives, not finer grids. Something structural.

Possibilities I see:

A. CLASS-CONDITIONAL CASCADE. Don't use the same retrieval for all
classes. Use texture retrieval for classes where it helps (ship,
airplane) and brute Gauss for classes where pre-filtering hurts (frog,
bird). The confidence map / vote agreement could be the router.

B. GAUSS MAP STEREOSCOPIC. Multiple Gauss maps with different gradient
threshold parameters. Like the block stereo: each Gauss eye sees
different edge structure. The combination captures what no single
threshold captures.

C. HIERARCHICAL GAUSS. Binary machine/animal on blocks (81%) → within-
group Gauss map specialists. The Gauss map trained on only machine
images would have different IG-like weighting than one trained on
animal images.

D. TEMPORAL / SEQUENCE. Process the 200 candidates in Gauss-distance
order with CfC sequential dynamics. The sequential processing that
failed on block features might work on Gauss features because the
score distribution is different.

E. GAUSS MAP ON THE FLATTENED RGB (32×96). The current Gauss map
operates on the spatial 32×32 per-channel images. The flattened RGB
has color transitions encoded in the horizontal gradient (R→G, G→B).
The Gauss map of this field would capture color edge geometry that
the per-channel approach misses.

F. SECOND-ORDER GAUSS MAP. Not just the gradient direction/magnitude,
but the CHANGE in gradient between adjacent regions. This captures
curvature, not just edge direction. A cat's rounded ears vs a dog's
pointed ears might differ in curvature even if they have similar
first-order gradients.

Actually, F is the most interesting. The original MNIST Gauss map
included a second-order (64-bin transition matrix between meta-states).
We haven't tried second-order on CIFAR-10. Curvature is a
genuinely different feature from edge direction.

And the user already has this infrastructure — contribution 29's
sstt_gauss_map.c includes the 64-bin second-order histogram.

WHERE THE PROJECT SHOULD GO:

This session produced 43 contributions. That's too many for one
paper. The paper should focus on:

1. The MNIST/Fashion story (97%/86%, zero parameters, honest validation)
2. The stereoscopic principle (new architectural primitive)
3. The Gauss map + cascade on CIFAR-10 (50%, shape beats texture)

Everything else supports these three stories but doesn't need its own
section. The 10 negative results go in an appendix. The LMM analysis
goes in the methodology section.

The CIFAR-10 work could be its own paper if the next round pushes
past 55%. The stereoscopic principle could be its own paper if
validated on more datasets (SVHN, STL-10, etc.).

But the immediate step-change is: stop experimenting, write the paper,
submit it. The results are strong enough now. 50% on CIFAR-10 with
zero learned parameters is a headline result.

Unless the second-order Gauss map curvature feature breaks through
to 55%. Then THAT's the paper.
