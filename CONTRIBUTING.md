# Contributing to SSTT

## Methodology: Hypothesis-Driven Experimental Search

This project follows a specific research discipline:

1. **State the hypothesis before writing code.** Each experiment has a
   documented prediction: "adding X should improve Y by approximately Z
   because of mechanism W."

2. **Implement as a self-contained experiment.** One `.c` file, one idea,
   one test. No coupling between experiments.

3. **Measure honestly.** Record what the data says, not what you hoped.
   If the hypothesis is wrong, say so.

4. **Document negative results equally.** A clean negative result that
   rules out an approach (PCA fails at -16pp, soft-prior cascade fails,
   hard filtering loses to soft weighting) is as valuable as a positive
   one. It constrains the search space for everyone.

5. **Red-team your own claims.** After finding a result, actively try to
   break it: val/test splits, 1-NN controls, brute-force baselines,
   timing validation. Report the honest numbers.

This methodology produced 40 experiments, 8+ documented negative results,
and a red-team validation that corrected two overclaimed findings. The
process is transferable to any experimental ML project.

## Adding an Experiment

1. **Name it** — `sstt_<descriptive_name>.c`, self-contained, no shared state
   with other experiments. Place in the appropriate `src/` subdirectory:
   - `src/core/` — publication-ready implementations only
   - `src/analysis/` — diagnostic and validation tools
   - `src/ablation/` — ablation series experiments
   - `src/cifar10/` — CIFAR-10 boundary experiments
   - `src/archive/` — exploratory or superseded work
2. **Add a Makefile target** — follow the existing `$(BUILD)/sstt_name` pattern.
   Use the appropriate section in the Makefile for your target's group.
   Binaries are output to `build/` (gitignored).
3. **Document it** — a numbered doc in `docs/contributions/` with:
   - Hypothesis
   - Architecture / implementation notes
   - Results (accuracy, speed, comparison to reference)
   - What the failure teaches (if negative)
4. **Update CHANGELOG.md** — one paragraph under the current unreleased section
5. **Reference numbers are sequential** — check the highest existing doc number

## Code Style

- C99, single file per experiment, no external dependencies
- AVX2 intrinsics for hot paths; scalar fallback for correctness reference
- Constants at the top as `#define`; no magic numbers in loops
- `static` for all non-`main` functions
- Error handling: print to stderr, `exit(1)` — this is research code, not
  production

## Tests and Scripts

Standalone test programs live in `tests/`. Utility and analysis scripts
live in `scripts/`. Both follow the same single-file convention as `src/`.

## Reporting Issues

Use GitHub Issues. For reproducibility bugs, include:
- CPU model (AVX2 support is required)
- GCC version (`gcc --version`)
- MNIST/Fashion-MNIST source (the Makefile downloads from the canonical URLs)
- Exact command and observed vs expected output

## Pull Requests

- One experiment or fix per PR
- All existing experiments should still build after your change
- If you change an encoding or index format, update the affected experiments
  and note the accuracy delta

## Data

MNIST and Fashion-MNIST data files are not committed. Run `make mnist` and
`make fashion` to download them. The Makefile fetches from the canonical
sources (OSSCI S3 and Zalando S3 respectively).
