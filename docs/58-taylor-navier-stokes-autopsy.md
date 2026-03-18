# Contribution 58: Taylor and Navier-Stokes Autopsy

This contribution documents the implementation and red-teaming of high-order geometric and fluid-dynamic operators in the ternary engine.

## 1. Taylor Jet Space (2nd-Order Hessian)
**Concept:** Classify pixels based on local curvature (Ridges, Saddles, Peaks) using the discrete Hessian matrix.
- **Success (MNIST):** Achieved **91.60% accuracy** with extreme compression (512 integers). Proved that local curvature captures "stroke intent" better than raw blocks.
- **Failure (CIFAR-10):** Binary "Cat vs Dog" accuracy was only **54.45%**. The ternary quantization destroys the subtle curvature signals in low-resolution natural images.

## 2. Navier-Stokes (Stream Function)
**Concept:** Treat the ternary gradient as a velocity field and solve the Poisson equation ($\nabla^2 \psi = -\omega$) to find the topological skeleton.
- **Failure:** Accuracy dropped to **random baseline (10%)**. The global potential field is dominated by image frame boundaries rather than object structure.

## Technical Verdict
While "Higher Math" provides elegant theoretical frameworks, the **Ternary Constraint** acts as a hard filter. Operators that depend on continuous derivatives (Taylor, Navier-Stokes) lose their discriminative power when the field is quantized to three levels before analysis.

### The Winning Pattern
The most robust features in this repository are **Integral Measures** (Divergence, Centroid, Information Gain) rather than **Differential Measures** (Hessian, Curl). Summing energy over a region is more stable in ternary space than calculating how that energy changes.

---
## Files
- Taylor Implementation: `src/sstt_taylor_jet.c`
- Navier-Stokes Implementation: `src/sstt_navier_stokes.c`
- Cat vs Dog Red-Team: `src/sstt_taylor_specialist.c`
