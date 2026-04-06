# Synthesis: SSTT Step-Changes (Post-MTFP)

## The cut

Three experiments in priority order. Each informs the next. No experiment depends on writing code from scratch — they're measurements on the system we have.

## Experiment 1: MTFP Diagnostics (measure first)

**What:** Run MTFP with K-sweep and error mode decomposition on the same pass.

- K = {50, 100, 200, 500, 1000} with full topo9 ranking
- For each K: count Mode A (correct class absent), Mode B (ranking inversion), Mode C (vote dilution)
- Report the confusion matrix at the best K

**Why:** Everything downstream depends on knowing where the 247 errors are. If MTFP reduced Mode A from 7 to fewer, the ceiling just changed. If Mode B still dominates, multi-threshold ensemble is the right next move. If Mode C grew, k=3 vote may need to become k=5.

**How:** Modify sstt_mtfp.c (or write sstt_mtfp_diagnose.c) to add error mode classification to the K-sweep. For each test image at each K: run vote_mtfp, select_top_k, rank, check if correct class is in the candidate pool (Mode A vs B+C), check if the correct class wins ranking but loses knn_vote (Mode C vs B).

**Success criterion:** We know the exact Mode A/B/C breakdown for the MTFP system and the K-sensitivity curve with per-channel indexing.

## Experiment 2: Fashion-MNIST on MTFP (one command)

**What:** `./build/sstt_mtfp data-fashion/`

**Why:** Confirms the trit-flip fix generalizes. The hypothesis is that Fashion gains more than MNIST because intra-class diversity is higher and wrong-topology multi-probe was more damaging.

**Success criterion:** Accuracy exceeds 85.68% (the old topo9 result on Fashion). If it does, the MTFP contribution is cross-dataset.

## Experiment 3: Multi-Threshold MTFP Ensemble (the step-change candidate)

**What:** Multiple quantization thresholds feeding the MTFP per-channel pipeline. Each threshold set produces a different set of ternary channels, block signatures, and per-channel indices. Candidate pools from all threshold sets are merged before ranking.

**Why:** The structural ranker's power comes from diverse candidates. Multi-threshold retrieval provides diversity by construction — the same digit quantized at different thresholds produces different block signatures that retrieve different training images. This is the same principle that made the 5-trit ensemble reach 98.42% on the old system, but now with the MTFP topology fix and per-channel indexing.

**How:** Write sstt_mtfp_ensemble.c. For each of N threshold sets (e.g., 85/170, 64/192, 96/160, 75/180, 100/155):
1. Quantize training and test images at that threshold
2. Build per-channel indices with trit-flip neighbors
3. At inference: accumulate votes from all threshold sets into one shared vote array
4. Select top-K from the merged votes
5. Rank with the standard structural features

**Depends on:** Experiment 1 results. If Mode B dominates and K-sensitivity shows the ranker has capacity, this is the right move. If Mode A dominates, encoding changes are more productive.

**Success criterion:** Accuracy exceeds 97.53% (single-threshold MTFP). Target: 98%+.

## What this does NOT include

- Paper update (defer until experiments are done)
- New structural features (premature — measure what we have first)
- Single-threshold optimization (superseded by multi-threshold ensemble)
- CIFAR-10 (architectural boundary — no ternary change fixes texture blindness)

## Order of execution

1. Run Fashion-MNIST (30 seconds, one command)
2. Build and run MTFP K-sweep + error diagnostics (~10 min compute)
3. Based on results: build and run multi-threshold MTFP ensemble (~20 min compute)
4. If all succeed: commit, update docs, push
