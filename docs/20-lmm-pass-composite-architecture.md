# LMM Pass: Composite Architecture for SSTT

Applying the Lincoln Manifold Method to the next phase of SSTT development.

---

## Phase 1: RAW

The SSTT project has reached a plateau. We have:
- 96.12% on MNIST via brute WHT k=3 (expensive)
- 96.04% via full cascade (expensive)
- 73.23% via 3-channel hot map (fast, pure lookup)
- A fused kernel that does raw pixels → class in one pass (proven correct)
- Fashion-MNIST results showing the architecture generalizes (81% vote→refine)
- A failed ternary PCA experiment (55% vs 71% baseline)
- A failed spectral coarse-to-fine (71.95% original, fixed via vote-based filtering)

The gap that haunts me: 73% (hot map) vs 96% (cascade). The 23pp gap
comes from IG weighting (+15pp), multi-probe (+2pp), and dot product
refinement (+5pp). The first two are pure table operations. They could be
fused into the hot map. Nobody has tried this.

My gut says the pre-baked IG hot map is a guaranteed win. It's just
multiplying a weight into the table at training time. The fused kernel
doesn't change. The accuracy jumps 15pp for free.

Multi-probe is trickier in the fused kernel — 7 lookups per block
instead of 1. But still pure addressing. No dot products. No per-image
state. It should compose.

What scares me: the cross-channel joint encoding. Encoding 3 channels
into one block value means 27 possible values per pixel position. A
3-pixel block of joint values has 27^3 = 19,683 possible values. The
hot map would explode. Unless we use 1-pixel joint blocks (27 values)
which is the same as current. Or 2-pixel joint blocks (729 values).
That's manageable — 729 × 252 × 16 × 4 = ~44MB. Too big.

Maybe joint encoding is wrong. Maybe the right move is to stay with
independent channels but pre-bake IG into each channel's hot map.

The eigenvalue-spline idea is elegant but theoretical. For MNIST it's
unnecessary (252 positions fit in L2). For scaling to larger images it
matters. But we should prove the composites work at MNIST scale first.

What would the LMM method itself say about our architecture? Our method
IS the LMM applied to ternary inference:
- RAW = raw pixels
- NODES = ternary quantization + block encoding (extract key features)
- REFLECT = hot map lookup (consult accumulated knowledge)
- SYNTHESIZE = argmax (produce actionable output)

The pipeline mirrors the cognitive method. That's not an accident.

Questions:
- Is pre-baked IG really +15pp? The cascade IG operates on the inverted
  index (per-image votes), not on the hot map (per-class counts). Are
  these equivalent?
- Does multi-probe in hot map space equal multi-probe in vote space?
- What's the minimal composite that crosses 85%?
- Could the LMM's "loop-back" principle suggest an iterative
  classification approach?

---

## Phase 2: NODES

### Node 1: The IG Equivalence Question
IG weighting in the cascade multiplies per-image vote counts by the
block's IG weight. In the hot map, the vote count IS the hot map entry.
So pre-baking IG into the hot map is: `weighted_hot[k][bv][c] = hot[k][bv][c] * ig[k]`.
This scales each block's contribution by its discriminative importance.
The cascade does the same thing per-image. These should be equivalent
when summed across all blocks — the multiplication distributes over
addition.

### Node 2: Multi-Probe in Hot Map vs Vote Space
In the cascade, multi-probe adds votes from Hamming-1 neighbors of each
block value. In the hot map, this means: for each block k with value bv,
also accumulate hot[k][nbv] at half weight for each neighbor nbv.
This is 7 lookups per block per channel instead of 1. For 3 channels
and 252 blocks: 7 × 3 × 252 = 5,292 lookups total (vs 756 without
multi-probe). Each lookup is 2 cache lines = 64 bytes. Total data
touched: 5,292 × 64 = 339KB per classification. Within L2 capacity.

### Node 3: The Cascade Accuracy Decomposition
- Hot map (no IG, no multi-probe): 73.23%
- IG-weighted hot map (predicted): ~85-88%
- IG + multi-probe hot map (predicted): ~88-91%
- Gap to 96%: the 5pp from dot product refinement

### Node 4: LMM Structure in the Pipeline
The classification pipeline maps onto LMM phases:
- RAW → raw pixels (unprocessed input)
- NODES → quantize + block_encode (extract structure)
- REFLECT → hot map lookup (consult knowledge base)
- SYNTHESIZE → argmax (decide)
The "loop-back" in LMM (return to earlier phase when synthesis is
uncertain) maps to: if confidence is low, refine with dot products.
This is exactly what the cascade does.

### Node 5: Confidence-Gated Refinement
What if the fused kernel's argmax included a confidence score (margin
between top-2 classes)? Low-confidence images get dot product refinement.
High-confidence images are done in microseconds. This would give near-96%
accuracy with near-hot-map speed on most images.

### Node 6: The Cross-Channel Encoding Problem
Independent channels miss inter-channel correlations. But joint encoding
explodes the hot map. The right middle ground might be: independent
channel hot maps PLUS a small joint hot map for the most discriminative
positions (top-K by IG). This focuses the combinatorial cost where it
matters most.

Tensions:
- Node 1 vs Node 3: IG pre-baking should help, but the cascade's +15pp
  operates in a different space (per-image votes with inverted index).
  The hot map IG might not give the full +15pp.
- Node 2 vs Node 3: Multi-probe in hot map space might not match
  cascade multi-probe because the hot map aggregates per-class, losing
  per-image identity that the vote stage preserves.
- Node 5 vs simplicity: Confidence-gated refinement adds branching and
  complexity to the fused kernel.

---

## Phase 3: REFLECT

### Core Insight
The cascade's power comes from per-image discrimination — it knows which
specific training images match, not just which classes. The hot map
collapses per-image identity into per-class counts. IG weighting and
multi-probe partially compensate, but they can't recover the per-image
signal.

The 73→96% gap decomposes into:
- 73→88%: recoverable via IG + multi-probe (no per-image state needed)
- 88→91%: partially recoverable (multi-probe is a form of soft matching)
- 91→96%: requires per-image refinement (dot products)

### Resolved Tensions
- **IG equivalence**: Pre-baked IG won't give the full +15pp because
  the cascade's IG multiplies per-image votes (which are already
  weighted by match quality), while the hot map IG multiplies per-class
  aggregate counts. Expected gain: +10-13pp (to ~83-86%).
- **Multi-probe equivalence**: Hot map multi-probe adds neighbor
  contributions at the class level. Cascade multi-probe adds them at
  the image level. Expected gain: +2-4pp (to ~85-90%).
- **Confidence gating**: This is the bridge between hot map speed and
  cascade accuracy. The fused kernel handles the easy 85% of images.
  The hard 15% get the full cascade treatment. Average latency drops
  dramatically while maintaining 96% accuracy.

### Hidden Assumption
We've been treating the hot map and cascade as separate architectures.
They're actually two ends of a spectrum: the hot map is a fully-collapsed
cascade (all per-image identity removed), the cascade is a fully-expanded
hot map (every per-image vote preserved). The composites we're proposing
are points between these extremes.

### What Changed
The real composite isn't "IG + multi-probe + joint encoding." It's
**confidence-gated hybrid classification**: fast hot map for easy images,
cascade refinement for hard ones. The LMM's loop-back principle applied
to inference.

---

## Phase 4: SYNTHESIZE

### Architecture: Confidence-Gated Hybrid Classifier

```
Raw pixels
    │
    ├─→ Fused kernel (3-chan IG multi-probe hot map)
    │       │
    │       ├─→ High confidence (margin > T) → DONE (microseconds)
    │       │       (~85% of images)
    │       │
    │       └─→ Low confidence (margin ≤ T) → Cascade refinement
    │               │
    │               └─→ Vote → top-K → dot product → k=3 → DONE
    │                       (~15% of images)
    │
    └─→ Expected: ~95-96% accuracy, ~10μs average latency
```

### Implementation Plan

**Step 1: Pre-baked IG hot map** (guaranteed win, zero runtime cost)
- Compute IG weights per block per channel (already implemented)
- Multiply into hot map at training time
- Test in fused kernel
- Expected: ~83-86% (up from 73.23%)

**Step 2: Fused multi-probe** (pure addressing, no per-image state)
- For each block, look up 6 Hamming-1 neighbors at half weight
- Build neighbor table at training time
- Accumulate 7 lookups per block per channel
- Expected: ~87-90%

**Step 3: Confidence margin** (add to fused kernel)
- After argmax, compute margin = votes[best] - votes[second_best]
- Return (class, margin) pair
- High-margin images are done; low-margin images flagged for refinement

**Step 4: Cascade fallback** (for low-confidence images only)
- Use existing vote→refine pipeline from sstt_v2
- Only triggered for the ~15% of images where hot map is uncertain
- Gives full 96% accuracy on those images

### Success Criteria
- [ ] Pre-baked IG hot map exceeds 83% on MNIST
- [ ] Multi-probe hot map exceeds 87% on MNIST
- [ ] Confidence-gated hybrid achieves 95%+ on MNIST
- [ ] Average latency under 20μs (vs ~4μs pure hot map, ~300ms cascade)
- [ ] 100% agreement with C reference on high-confidence subset

### Key Decisions
1. Pre-bake IG because it's free and proven.
2. Fuse multi-probe because it stays in the addressing paradigm.
3. Use confidence gating instead of always running the cascade, because
   85% of images don't need refinement.
4. Defer cross-channel joint encoding — the gain is uncertain and the
   hot map size explosion is real. Confidence gating achieves the same
   accuracy target without it.
5. Defer eigenvalue-spline compression — not needed at MNIST scale.
   Revisit when scaling to larger images.

### Connection to LMM
The confidence-gated architecture IS the Lincoln Manifold applied to
inference:
- Phase 1 (RAW): raw pixels enter
- Phase 2 (NODES): quantize, block encode — extract structure
- Phase 3 (REFLECT): hot map lookup — consult accumulated knowledge
- Phase 4 (SYNTHESIZE): argmax with confidence — the clean cut

**Loop-back trigger**: low confidence → return to earlier phase
(cascade refinement). This is the method's own principle applied to
its own output.

The wood cuts itself when you understand the grain. Most images are
easy — the grain is clear. For the hard ones, sharpen the axe (refine
with dot products) before cutting.
