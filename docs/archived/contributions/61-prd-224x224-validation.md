# PRD: 224×224 Validation — Contribution 61

**Status:** Spec
**Scope:** Close the interpretability gap in the 224×224 scaling experiments
**Blocked by:** Missing brute kNN baseline; undersized training runs

---

## Problem Statement

Every accuracy number in the 224×224 scaling work (C52, C55, C59) is currently uninterpretable because:

1. **No brute kNN baseline exists at this resolution.** 41.9% on Fashion-224 could mean the architecture retains 99% of the achievable accuracy (if brute kNN is also ~42%) or it could mean the architecture has collapsed (if brute kNN is 80%+). There is no way to know without running the baseline.

2. **Both existing experiments are severely undersized.** `sstt_scale_224.c` trains on 30K (50% of data), tests on 500 (5% of test set). `sstt_scale_hierarchical.c` trains on 20K (33% of data), tests on 1K (10% of test set). The published accuracy figures are unreliable.

3. **The "224×224" representation has zero new information.** The resize function is nearest-neighbor: every 8×8 pixel block in the output is a copy of a single original pixel. The 224×224 representation is a blocky upscale of 28×28. The model at 224×224 has 64× more block positions but the same number of training examples — it is severely underfitted by construction. The published numbers measure underfitting, not resolution.

4. **C54's 89.1% retention claim has no documented baseline.** It references a denominator that does not appear in any doc.

---

## Scope of This Contribution

Four experiments, one new source file, two modified source files, two new docs.

---

## Experiment A — Full-Data Rerun (MNIST-224)

**File:** `src/sstt_scale_224.c`
**Change:** `TRAIN_N 30000 → 60000`, `TEST_N 500 → 10000`
**Add:** Per-run wall-clock timing, latency per query

**What to measure:**
- Full hot map accuracy (all 1,008 positions trained)
- Sparse hot map accuracy (6.25% knots + bilinear interpolation)
- Accuracy retention (sparse / full)
- Latency per query (μs) for both

**Acceptance criteria:**
- N = 60,000 train / 10,000 test confirmed in output header
- Both accuracy figures reported to two decimal places
- Retention reported to one decimal place
- Latency reported in μs

**Expected direction:** Accuracy should increase substantially from the undersized 13.60% / 13.40% numbers. The retention ratio (~98.5%) should hold.

---

## Experiment B — Full-Data Rerun (Fashion-224)

**File:** `src/sstt_scale_hierarchical.c`
**Change:** `TRAIN_N 20000 → 60000`, `TEST_N 1000 → 10000`

**What to measure:**
- Level-0 (fine, 224×224 knots) accuracy
- Level-1 (coarse, 56×56 pooled knots) accuracy
- Hierarchical fused accuracy
- Latency per query (μs) for hierarchical system

**Acceptance criteria:**
- N = 60,000 train / 10,000 test confirmed in output header
- All three accuracy figures reported to two decimal places
- Per-class breakdown (10 classes) reported in output
- Latency confirmed

**Expected direction:** Accuracy should improve from C59's 41.90%, since the model now has 3× more training examples per position. The C59 per-class numbers are not reliable enough to keep; these replace them.

---

## Experiment C — Brute kNN Baseline at 224×224

**File:** `src/sstt_scale_brute.c` (new)
**Purpose:** Establish the hard accuracy ceiling at this resolution/representation. Makes every other 224×224 number interpretable.

**Architecture:**
- Upscale images to 56×56 via 4×4 average pooling (same as the Level-1 representation used throughout).
- Extract 1,008 block signatures per image (same `get_block_sigs` function used in `sstt_scale_224.c`).
- At inference: for each test image, compare against all 60,000 training images by counting matching block positions (Hamming similarity on signature vectors).
- Predict by k=1 and k=3 nearest neighbor (most matching signatures).

**Why 56×56 pooled, not 224×224 full:**
The 224×224 representation is nearest-neighbor upscaling of 28×28 — it contains no new information and has 16× more positions. The 56×56 pooled representation is the canonical "224×224 scaled" feature used in all existing experiments. The brute baseline should operate on the same feature space as the hot maps.

**Training-set storage:** 60,000 images × 1,008 signatures = ~60MB. Fits in RAM.

**Runtime estimate:** 60,000 × 1,008 uint8 comparisons per query × 10,000 queries = 604 billion operations. At AVX2 uint8 comparison rates (~50 GB/s memory throughput): ~60 seconds total. Acceptable for a one-time validation run.

**What to measure:**
- Brute 1-NN accuracy (MNIST-224)
- Brute k=3 NN accuracy (MNIST-224)
- Brute 1-NN accuracy (Fashion-224)
- Brute k=3 NN accuracy (Fashion-224)
- Full hot map accuracy (MNIST-224) — for same-run comparison
- Full hot map accuracy (Fashion-224) — for same-run comparison
- Hot map retention vs. brute k=3 NN

**Acceptance criteria:**
- All four brute figures reported
- Both datasets run in same binary (argv selects data dir)
- Output states N_train, N_test, K explicitly
- Memory usage reported (MB for stored signatures)

**The interpretability unlock:** once these four numbers exist, every retention percentage in C52, C54, C59 becomes interpretable. The story is either "the hot map retains X% of a strong baseline" (good) or "the hot map retains X% of a weak baseline" (context-dependent).

---

## Experiment D — Reconcile C54's 89.1% Retention

**No new code required.**

C54 states: "Fashion-224 bilinear retention: 89.1%." This was run during the red-team phase (C54) using the C52 code (bilinear sparse interpolation) on Fashion data. The code in `sstt_scale_224.c` defaults to `data/` (MNIST). The C54 run must have passed `data-fashion/` as argv[1].

After Experiment A reruns with full data:
- Run `build/sstt_scale_224 data-fashion/` with 60K/10K
- This produces Fashion-224 full hot map and sparse hot map figures from the same architecture as C52
- Reports the denominator that C54's 89.1% was computed against
- Reconciles C52 (MNIST bilinear), C54 (Fashion bilinear), and C59 (Fashion hierarchical) into a consistent table

This is a run, not a code change. Output goes directly into the result doc.

---

## Deliverables

### Code changes

| File | Change |
|---|---|
| `src/sstt_scale_224.c` | `TRAIN_N` 30K→60K, `TEST_N` 500→10K, add timing output |
| `src/sstt_scale_hierarchical.c` | `TRAIN_N` 20K→60K, `TEST_N` 1K→10K, add per-class breakdown |
| `src/sstt_scale_brute.c` | New: brute kNN baseline for both datasets |
| `Makefile` | Add target for `sstt_scale_brute` |

### Result docs

| Doc | Content |
|---|---|
| `docs/62-scale-224-full-data-results.md` | Corrected C52 (MNIST, 60K/10K), C59 (Fashion, 60K/10K), C54 reconciliation, all in one table |
| `docs/63-scale-brute-knn-baseline.md` | Brute kNN results for both datasets, retention analysis |

### Updated docs

| Doc | Change |
|---|---|
| `docs/52-scaling-framework-224x224.md` | Note that numbers are superseded by C62; link to C62 |
| `docs/59-hierarchical-geometric-scaling.md` | Note that numbers are superseded by C62; link to C62 |
| `docs/54-red-team-validation-log.md` | Note that 89.1% is reconciled in C62; link to C62 |

---

## Acceptance Criteria (End-to-End)

1. `make build/sstt_scale_brute && build/sstt_scale_brute` produces brute kNN accuracy for MNIST-224.
2. `build/sstt_scale_brute data-fashion/` produces brute kNN accuracy for Fashion-224.
3. `build/sstt_scale_224` with 60K/10K produces corrected C52 numbers.
4. `build/sstt_scale_224 data-fashion/` produces the C54 denominator.
5. `build/sstt_scale_hierarchical` with 60K/10K produces corrected C59 numbers.
6. Docs C62 and C63 exist with all figures filled in.
7. The sentence "The hot map retains X% of brute kNN accuracy" can be stated with a real X for both datasets.

---

## Out of Scope

- Full cascade (inverted index + ranking) at 224×224. This is a larger project requiring block-index construction at 16,128 positions. Valuable but not blocking the interpretability question.
- Hybrid IG-foveal knot placement (suggested in C55). Requires the brute baseline to evaluate against; can be C64 once C63 exists.
- Bilinear vs. nearest-neighbor upscaling comparison. The current system uses nearest-neighbor; bilinear would genuinely add new information but is a separate experiment.
