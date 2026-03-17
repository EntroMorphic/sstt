# Contribution 40: Hierarchical Decomposition — Machines vs Animals

Tests whether decomposing CIFAR-10 into coarse-to-fine stages improves
classification. First distinguish machines from animals (binary), then
classify within each group.

Code: `src/sstt_cifar10_binary.c`

---

## Motivation

The per-class autopsy (contribution 38) showed the system classifies
by photographic properties. Man-made objects (geometric edges, sky/road
backgrounds) form a natural cluster. Animals (organic textures, natural
backgrounds) form another. The confusion matrix confirms:
deer↔frog (357 errors) and cat↔dog (248) are within-animal confusions.
airplane↔ship (349) is within-machine. Cross-group confusions
(airplane↔bird: 270) are less frequent.

If the system can reliably separate these groups, each sub-problem
becomes simpler: 4-class machines or 6-class animals instead of
10-class everything.

---

## Results

### Binary split: Machine vs Animal

Using 3-eye stereoscopic Bayesian, collapse 10-class posterior to
2-class (sum machine posteriors vs sum animal posteriors):

```
Binary accuracy: 81.04%

Confusion:     pred_machine  pred_animal
  machine:        3221          779     (recall 80.5%)
  animal:         1117         4883     (recall 81.4%)
```

**81% binary accuracy.** The system reliably distinguishes the two
groups 4 out of 5 times. Balanced recall: neither group is favored.

### Within-group classification

Given the CORRECT group label (oracle routing):

| Group | Classes | Accuracy |
|-------|---------|----------|
| **Machines** | airplane, auto, ship, truck | **59.23%** |
| **Animals** | bird, cat, deer, dog, frog, horse | **41.95%** |

Per-class within machines:
- airplane: 56.5%
- automobile: 63.3%
- ship: 52.9%
- truck: 64.2%

Per-class within animals:
- bird: 33.8%
- cat: 24.4%
- deer: 36.3%
- dog: 39.7%
- frog: 63.6%
- horse: 53.9%

### Hierarchical vs flat

| Method | Accuracy |
|--------|----------|
| Flat 3-eye Bayesian | 41.18% |
| Hierarchical (binary → within-group) | 41.18% |
| **Oracle hierarchical (perfect binary)** | **48.86%** |

The hierarchical approach produces identical accuracy to flat because
the Bayesian argmax already implicitly groups classes. But the oracle
result reveals: **if the binary split were perfect, accuracy would
jump to 48.86%** — a +7.7pp ceiling from eliminating cross-group
errors.

---

## Analysis

### Why 81% binary is not enough

With 19% binary errors:
- 779 machines misclassified as animals → sent to wrong sub-classifier
- 1117 animals misclassified as machines → sent to wrong sub-classifier
- Total: 1896 images guaranteed wrong (18.96%)
- Remaining: 8104 images correctly routed, classified at within-group rates
- Net: ~41% (matching flat, as observed)

### The 81% → 90% question

To gain from hierarchical decomposition, the binary classifier must
exceed ~87%. Below that, binary errors cancel within-group gains.
The break-even calculation:

```
flat_accuracy = 41.18%
binary_error_rate = B
within_group_accuracy = 48.86% (oracle)
hierarchical = (1-B) * 48.86%

Solve for hierarchical > flat:
(1-B) * 48.86 > 41.18
B < 1 - 41.18/48.86
B < 15.7%

Need binary accuracy > 84.3% to beat flat.
```

Current: 81%. Need: 84.3%. Gap: 3.3pp.

### What the binary classifier gets wrong

The 19% binary errors are concentrated in cross-group confusable pairs:
- airplane↔bird: similar silhouette against sky
- horse↔truck: similar aspect ratio, outdoor scenes
- deer↔automobile: lighting-dependent confusion

These are the images where photometric properties overlap between
machines and animals. The same brightness/texture/color features that
confuse the 10-class classifier also confuse the binary one.

### The atomic next step

To improve the binary classifier from 81% to 84%+:

1. **Dedicated binary IG weights.** Current IG weights are computed for
   10-class discrimination. Binary-specific IG would weight blocks by
   their machine-vs-animal information, not their within-class information.

2. **Binary-specific inverted index.** Build the index with 2-class
   labels (machine=0, animal=1). The IG weights, hot map, and Bayesian
   posterior all become binary-optimized.

3. **Binary stereoscopic.** Three eyes with binary IG — each eye
   sees machine-vs-animal from a different quantization perspective.

4. **Binary cascade.** Full pipeline (vote → dot → topo) optimized
   for the binary distinction. Dot products between machines and
   animals should be more discriminative than between similar classes.

5. **Binary confidence routing.** High-confidence binary predictions
   (>90% posterior) → route immediately. Low-confidence → use the
   flat 10-class classifier as fallback.

---

## Implications for Architecture

The hierarchical decomposition reveals a design principle:

**The current architecture wastes discriminative power on easy
distinctions.** The 10-class IG weights, hot map, and inverted index
allocate capacity to distinguish airplane from truck (easy — different
colors, shapes) AND cat from dog (hard — similar everything). A
hierarchical architecture could allocate more capacity to the hard
distinctions.

This maps to the MoE architecture from contribution 38, but with a
principled decomposition: coarse (machine/animal) → fine (which
machine? which animal?) instead of flat (which of 10?).

The oracle ceiling of 48.86% shows this approach has +7.7pp of
headroom if the routing can be made reliable.

---

## Files

Code: `src/sstt_cifar10_binary.c`
This document: `docs/40-hierarchical-decomposition.md`
