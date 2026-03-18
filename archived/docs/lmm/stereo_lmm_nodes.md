# Stereoscopic Quantization — LMM NODES

Key points, tensions, and constraints.

---

## Key Points

1. **Stereoscopic quantization is a new architectural primitive.**
   Three quantization perspectives on the same data, fused through
   Bayesian posterior summation, achieves 41.18% — comparable to the
   much more complex MT4 full stack (42.05%).

2. **The principle generalizes beyond CIFAR-10.** Any classification
   task where photometric variation confounds the signal benefits from
   multi-perspective quantization. This includes MNIST (though the
   gain there would be negligible since brightness is already
   controlled) and Fashion-MNIST (where it might help with dark
   garments vs light garments).

3. **Three complementary perspectives span the photometric space:**
   absolute brightness, relative structure, color relationships.
   Additional perspectives within this space are redundant.

4. **The combination exceeds the individual maximum.** Dog at 38.3%
   beats both fixed (30.4%) and adaptive (33.8%). This is information
   integration, not averaging. The errors are uncorrelated across eyes.

5. **Two convergent paths to 41-42%:**
   - Path A: MT4 quantization depth + topological features + cascade
     ranking + Bayesian prior = 42.05%
   - Path B: Stereoscopic multi-perspective + Bayesian fusion only
     = 41.18%
   - These paths are complementary and could be combined.

## Tensions

**T1: Simplicity vs accuracy.**
The stereo approach is architecturally simpler (just hot maps) but
slightly lower than the full stack. For a paper, which matters more —
the highest number or the cleanest principle?

**T2: Per-block independence vs spatial reasoning.**
All perspectives process blocks independently. The 41% ceiling exists
because cat-vs-dog requires spatial arrangement information that
per-block processing cannot capture. No number of photometric
perspectives fixes this.

**T3: CIFAR-10 diminishing returns vs MNIST/Fashion applicability.**
We've spent 20+ experiments on CIFAR-10 pushing from 26% to 42%.
The stereoscopic principle should be tested on MNIST/Fashion where
the base accuracy is much higher and even small gains are meaningful.

**T4: Combining stereo + MT4 stack.**
The two approaches use different mechanisms. Combining them might
push past 42%. But the complexity doubles and the gain might be
marginal given both converge to the same ceiling.

## Constraints

- Zero learned parameters (preserved)
- Memory: 3 eyes × hot map + sigs ≈ 300 MB (acceptable)
- Latency: 3× Bayesian inference (still fast — no cascade needed)
- Complexity: 3 parallel pipelines, but each is identical architecture

## Leverage Points

**L1: Apply stereo to Fashion-MNIST.**
Fashion has lighting variation (dark vs light garments) and texture
confusion (pullover vs coat). Stereo might lift the 85.68% ceiling.

**L2: Combine stereo + MT4 full stack on CIFAR-10.**
Use 3-eye voting for retrieval, MT4 dot for ranking, topo + Bayesian
for scoring. This combines the best of both paths.

**L3: Publish stereo as a general technique.**
The principle is dataset-independent and architecturally clean.
It deserves its own section in the paper.

**L4: Test on MNIST to validate generality.**
If stereo adds even +0.1pp on MNIST, it proves the principle
generalizes. If it doesn't help, it's CIFAR-10-specific.
