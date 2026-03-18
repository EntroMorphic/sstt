# Gauss Map + Full Architecture — LMM REFLECT

---

## The Pattern I Keep Missing

Every time I declare a ceiling, the user asks a question that reframes
the problem. The pattern:

1. "The representation is the ceiling" → user: "What if we change the
   quantization?" → stereoscopic principle (+4.6pp)
2. "42% is the ceiling for block-based" → user: "Can we map to spheres
   instead?" → Gauss map (+4pp)
3. "Fusion doesn't work" → [what I should ask]: "Are we SEQUENCING
   the systems correctly?"

The unstated assumption each time: **the most recent system is the
final system.** But SSTT has always been about COMPOSITION — stacking
techniques that each contribute independently. I keep declaring
ceilings on INDIVIDUAL systems instead of asking how to COMPOSE them.

## The Deep Structure

The SSTT architecture has always been a pipeline:

```
RETRIEVE → RANK → DECIDE
```

On MNIST: block voting → dot+topo → kNN
On CIFAR-10 (block): stereo voting → MT4 dot+topo+Bayesian → k=1
On CIFAR-10 (Gauss): brute kNN over 50K (no retrieval, no cascade)

The Gauss map breaks the pipeline by doing everything in one step
(brute kNN). This is simpler but loses the RETRIEVE→RANK separation
that made the block system powerful.

**The insight: restore the pipeline with the BEST retrieval and the
BEST ranking.**

Best retrieval: 3-eye stereo inverted-index voting (98% recall, 200 candidates)
Best ranking: grid Gauss map L1 distance (48.31% when brute-forced)

If we use stereo voting for RETRIEVAL and Gauss map for RANKING:

```
Image → 3-eye stereo vote → top-200 candidates → Gauss map L1 → k=1
```

This is architecturally identical to the original SSTT cascade. Only
the ranking function changes: from dot product to Gauss map distance.

## Why This Could Exceed Both Systems

The brute Gauss map at 48.31% compares against ALL 50,000 training
images. Many of those are irrelevant (a truck being compared to
every airplane, bird, frog...). The nearest neighbor might be a
false positive — a bird with airplane-like edge geometry.

The stereo-filtered Gauss map compares against only 200 candidates
that are TEXTURE-PLAUSIBLE. If the image is a deer, the 200
candidates are other deer, frogs, birds, cats — things with similar
green/brown texture. The Gauss map then selects by SHAPE among
these texture-plausible candidates. The frogs (round shape) are
rejected. The deer (elongated shape) are selected.

This is a **two-hypothesis system**: texture constrains the search
space, then shape resolves within it. Neither constraint alone is
sufficient, but together they are stronger than either.

The ceiling prediction: if the 200 texture-filtered candidates
include the correct class (98% of the time), and the Gauss map
can correctly select it from 200 options (better than from 50000,
because the confusable candidates are both present), then accuracy
should be ABOVE 48.31%.

Counter-argument: the 200 candidates are texture-similar, meaning
the Gauss map's job is HARDER (deer vs frog among green things) not
easier. The brute kNN might find a deer that's shape-similar from
a DIFFERENT background (brown deer against mountains), which the
texture filter would exclude.

The resolution: test it. The argument goes both ways. Only the data
knows.

## The Full Stack Composition

If L1 (vote → Gauss rank) works:

1. **3-eye stereo vote** → top-200 candidates
2. **RGB grid Gauss map L1** → re-rank candidates
3. **Gauss map stereoscopic** → fuse multiple Gauss map thresholds
4. **Confidence routing** → high-agreement images trust block Bayesian,
   low-agreement images trust Gauss map
5. **Hierarchical** → binary machine/animal on blocks, then Gauss map
   specialist per group

Each of these layers has been tested individually but NOT in this
specific composition. The SSTT principle: stack techniques that each
contribute independently.

## Experiment Order

1. **L1: Vote → Gauss Rank** (highest priority — tests the key hypothesis)
2. **L2: RGB grid Gauss map** (quick — just add channels to existing code)
3. **L5: 3-eye stereo → Gauss rank** (if L1 works, upgrade retrieval)
4. **L3: 8×8 grid** (if L2 works, finer spatial resolution)
