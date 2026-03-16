# Contributing to SSTT

## Philosophy

This project documents a research progression. Every experiment — including
failures — is valuable. A clean negative result that rules out an approach
saves future time. If you add an experiment, document it honestly.

The project follows the **Lincoln Manifold Method**: understand before you
build. Before writing code, write the hypothesis. After running, record what
the data actually says, not what you hoped it would say.

## Adding an Experiment

1. **Name it** — `sstt_<descriptive_name>.c`, self-contained, no shared state
   with other experiments
2. **Add a Makefile target** — follow the existing pattern
3. **Add to `.gitignore`** — compiled binary only, not source
4. **Document it** — a numbered doc in `docs/` with:
   - Hypothesis
   - Architecture / implementation notes
   - Results (accuracy, speed, comparison to reference)
   - What the failure teaches (if negative)
5. **Update CHANGELOG.md** — one paragraph under the current unreleased section
6. **Reference numbers are sequential** — check the highest existing doc number

## Code Style

- C99, single file per experiment, no external dependencies
- AVX2 intrinsics for hot paths; scalar fallback for correctness reference
- Constants at the top as `#define`; no magic numbers in loops
- `static` for all non-`main` functions
- Error handling: print to stderr, `exit(1)` — this is research code, not
  production

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
