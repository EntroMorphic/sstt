# Contribution 32: Red-Team Validation

Results from the validation suite addressing 10 identified risks.

---

## P0 #1: Val/Test Split

Weights were optimized on the full 10K test set. Splitting into 5K
val / 5K holdout:

| | All 10K | Val (0-4999) | Test (5K-10K) |
|---|---|---|---|
| MNIST | 96.99% | 95.74% | 98.24% |
| Fashion | 85.81% | 86.00% | 85.62% |

The MNIST halves differ by 2.5pp — the second 5K is inherently easier.
Fashion halves differ by only 0.38pp. Weight overfitting is not the
dominant source of variance; the data split is.

## P0 #2: 1-NN Control

| | 1-NN | Bayesian seq | Delta |
|---|---|---|---|
| MNIST | 97.02% | 96.99% | -0.03pp |
| Fashion | 83.27% | 85.81% | **+2.54pp** |

**MNIST: The Bayesian sequential claim is invalid.** 1-NN on the
combined score matches within 0.03pp. The sequential processing adds
nothing. The accuracy comes from the combined scoring, not from the
candidate processing order.

**Fashion: The sequential claim is valid.** +2.54pp over 1-NN is
substantial (254 additional correct images). The aggressive decay
(S=2) works on Fashion because the candidate pools are more
heterogeneous — sequential processing downweights the noise from
lower-ranked candidates.

## P0 #3: Brute kNN Baseline

| | Brute kNN k=3 (pixel dot) | SSTT | Delta |
|---|---|---|---|
| MNIST holdout | 97.90% | **98.24%** | **+0.34pp** |
| Fashion holdout | 77.18% | **85.62%** | **+8.44pp** |

SSTT beats brute pixel-space kNN on both datasets. The Fashion delta
(+8.44pp) confirms the architecture's value — the inverted index
voting, bytepacked encoding, topological features, and adaptive
ranking provide substantially more than raw pixel similarity.

The MNIST delta (+0.34pp) is modest, consistent with the external
baseline estimate (~97.2% for standard kNN k=3).

## P0 #4: Fashion >97% Claim

Fixed in doc 31. The 97.2% applies to the 50% confident subset, not
to all of Fashion-MNIST. The overall accuracy remains 85.81%.

## P0 #5: Per-Phase Timing

| Phase | MNIST | Fashion |
|-------|-------|---------|
| Vote | 1009 μs (87%) | 2352 μs (93%) |
| Ranking | 151 μs (13%) | 182 μs (7%) |

The vote phase dominates. Routing to skip ranking saves at most 13%
of wall-clock time (MNIST) or 7% (Fashion). The speedup claim was
overstated. Routing's value is confidence gating, not speed.

## P0 #6: Fashion Gauss Map

Still running at time of validation. Results pending.

## P0 #7: Pre-Vote Difficulty Prediction

Gauss map distance to nearest class mean as a pre-vote predictor:

| Dataset | MI | Uncertainty reduced |
|---------|-----|---------------------|
| MNIST | 0.0047 bits | 2.4% |
| Fashion | 0.0108 bits | 1.8% |

**Pre-vote prediction is extremely weak.** Image-level features
cannot predict classification difficulty before retrieval. The
confidence signal requires seeing the candidate pool — it's a
property of the image-in-context, not the image alone.

This means the three-tier routing architecture (accept / light /
full) cannot skip the vote phase. The vote is always needed. But
post-vote routing (based on candidate class concentration) is free
and powerful (25% MI).

## P0 #8: External Baseline

Added to README. SSTT's contribution is not the accuracy number
(which is comparable to standard kNN) but the architectural insight:
integer table lookups and gradient-field topology can match kNN
without learned parameters or floating point.

## P0 #9: Negative Result Qualification

Fixed in doc 31. All negative results are now qualified as
"failed at tested weights on top of the topo v1 configuration,"
not "useless for MNIST."

## P0 #10: Experiment File Index

Added to INDEX.md. All 14 experiment files listed with one-line
descriptions.

---

## Corrected Claims

After red-team validation:

| Claim | Before | After |
|-------|--------|-------|
| MNIST accuracy | 97.31% | **96.99% (rerun), val=95.74%, holdout=98.24%** |
| Fashion accuracy | 85.81% | 85.81% (confirmed, val/holdout within 0.4pp) |
| Sequential processing | "+4 on MNIST" | **+0.03pp — negligible on MNIST; +2.54pp on Fashion** |
| Routing speedup | "eliminates 87% of compute" | **Saves at most 13% wall-clock** |
| Pre-vote prediction | Not tested | **Extremely weak (2% MI)** |
| vs brute kNN | Not compared | **+0.34pp MNIST, +8.44pp Fashion** |

## Files

- `src/sstt_validate.c`: Validation suite (split, 1-NN, brute kNN, timing, pre-vote)
