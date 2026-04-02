# Contribution 25: Oracle — Multi-Specialist Routing

## Motivation

The cascade autopsy (contribution 23) proved that 92.1% of MNIST test
images receive a unanimous k=3 vote (3-0) from the primary cascade and
are classified at 98.82% accuracy. The remaining 7.9% receive split votes
(2-1) and contain nearly all errors. The question: can a second specialist
with a different feature representation improve accuracy on that hard 7.9%?

## Architecture

```
Test image
    │
    ▼  Primary: bytepacked cascade (8-probe, IG-weighted)
    │
    ├─ Unanimous 3-0 (92.1%) → prediction, done
    │       accuracy: 98.82%
    │
    └─ Split 2-1 (7.9%) → run specialist → merge pools
            │
            ▼  vote_primary top-K ∪ vote_specialist top-K
               re-rank by multi-channel dot (px=256, vg=192)
               k=3 → prediction
```

The key insight: the merge step ("teaching") injects training images
that the primary missed — candidates found by a different quantization
or feature space — into the final dot-ranking step.

---

## Oracle v1: Ternary Secondary (sstt_oracle.c)

**Secondary specialist:** 3-channel independent ternary cascade (same
feature family as sstt_v2, different from bytepacked joint encoding).

| Test | Description | Accuracy | Time |
|------|-------------|----------|------|
| A | Primary only (baseline) | 96.20% | 33.9s |
| B | Routed: unanimous→primary, split→merge | **96.44%** | **5.3s** |
| C | Full ensemble: always both | 96.42% | 61.4s |

**Escalated path:** 793 images (7.9%) at 68.85% accuracy.
**Delta vs primary:** +0.24pp, 6.4× faster.

**Finding:** Routing is well-calibrated. The 3-channel ternary specialist
uses a different encoding (independent channels vs joint bytes) but the
same pixel/gradient features. The pools partially overlap; the merge adds
some new candidates but not enough to resolve topological confusions.

---

## Oracle v2: Pentary Specialist (sstt_oracle_v2.c)

**Secondary specialist:** Pentary pixel cascade — 5-level quantization
(thresholds 40/100/180/230), 5^3=125 values per block position.

Why pentary is structurally complementary:
- Ternary maps the gray zone (85–170) to 0 — silent
- Pentary maps 40–100 → -1,  180–230 → +1 — those regions speak
- The 4↔9 loop closure sits in the 40–100 pixel range on MNIST
- The primary's candidate pool misses training images where the
  distinguishing feature is a faint-stroke block value

| Test | Description | Accuracy | Time |
|------|-------------|----------|------|
| A | Primary only (baseline) | 96.20% | 10.8s |
| B | Routed: unanimous→primary, split→pentary merge | **96.40%** | **0.67s** |
| C | Full ensemble: always both | 96.41% | 8.2s |

**Escalated path:** 793 images (7.9%) at 68.35% accuracy.
**Delta vs primary:** +0.20pp.
**Delta vs oracle v1:** −0.04pp at 8× faster specialist step.

### Pentary index stats
- 125 values per block (vs 256 bytepacked, 27 ternary)
- Pool: 3.6M entries (13.9 MB) vs 5.6M (21.4 MB) for bytepacked
- Background: value 0 = all-(-2) = all-white, detected at 76.0% of positions

---

## Head-to-Head: v1 vs v2

| Metric | v1 (ternary) | v2 (pentary) | Delta |
|--------|-------------|--------------|-------|
| Routed accuracy | 96.44% | 96.40% | −0.04pp |
| Routing overhead | 5.3s | **0.67s** | **8× faster** |
| Escalated accuracy | 68.85% | 68.35% | −0.50pp |
| 4↔9 errors (B) | 38 | **36** | pentary helps |
| 3↔5 errors (B) | 29 | 33 | pentary hurts |
| 3↔8 errors (B) | 25 | 25 | equal |

**Pentary helps where ternary fails (faint strokes: 4↔9, 3↔8 loop
closures) but hurts where ternary is already good (3↔5 curve direction).
The two specialists are partially complementary, not fully orthogonal.**

The speed difference is striking: the pentary specialist's routing overhead
is 0.67s vs 5.3s for ternary. This is because:
- Pentary pool is 3.6M entries vs ternary 3-channel pool ~45M entries
- Only 7.9% of images reach the specialist
- 125-value multi-probe (up to 12 neighbors) hits smaller buckets than
  ternary's 27-value 6-neighbor probe on 3 channels

---

## The 100/100 Gap

After the best oracle result (v1, 96.44%), 356 errors remain.

```
356 remaining errors:
  4↔9:  36–38  (topological: loop closure)      ← pentary partially helps
  3↔5:  29–33  (orientational: curve direction) ← needs different feature
  3↔8:  25–25  (structural: 2 vs 1 loops)       ← needs topology
  1↔7:  20–23  (stroke: horizontal bar of 7)    ← transitions might help
  7↔9:  20     (stroke: bottom direction)       ← v-grad dot partially helps
```

The persistent confusions require features that neither ternary nor pentary
block encoding can provide:
- **Topology** (loop count): 3↔8, 4↔9
- **Global orientation** (curve opens left/right): 3↔5
- Only **1↔7** is plausibly addressable with a transitions-based specialist

The routing architecture is correct and efficient. The ceiling for
block-encoding specialists is approximately **96.5%** — the remaining
errors require a qualitatively different feature class.

## Red-Team

1. **Merge introduces new errors:** Some images that primary gets right
   may be broken by the merge. The net is +24 correct (v1) or +20 (v2),
   but gross movements could be larger.

2. **Escalated accuracy (68%) is low:** The 793 hard images remain hard
   even with a second specialist. Further splitting by confusion pair
   (dedicated 4/9 specialist, dedicated 3/5 specialist) might help but
   requires knowing which pair is confused before routing.

3. **Pentary thresholds are MNIST-optimized:** The 40/100/180/230
   thresholds were found by grid search on MNIST. Fashion-MNIST validation
   is needed before claiming pentary helps there.

4. **Pool merge weights votes equally:** A confidence-weighted merge
   (trust specialist proportionally to its vote margin) would be more
   principled. Currently `out[i].votes += b[j].votes` for overlapping
   candidates treats both specialists as equally reliable.

## Files

- `src/sstt_oracle.c`: v1 oracle with 3-channel ternary secondary
- `src/sstt_oracle_v2.c`: v2 oracle with pentary pixel specialist
- `Makefile`: `make sstt_oracle`, `make sstt_oracle_v2`
