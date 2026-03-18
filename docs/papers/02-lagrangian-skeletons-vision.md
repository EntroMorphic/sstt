# Paper 2: Lagrangian Structural Skeletons for Pose-Invariant Ternary Vision

**Abstract.** Traditional image histograms (Eulerian) are sensitive to pose changes and background noise, particularly in low-resolution ternary representations. We introduce **Lagrangian Structural Analysis** for ternary fields, a method that traces the "flow" of shape geometry to extract topological skeletons. By discretizing fluid-dynamic operators—specifically **Discrete Curl (Vorticity)** and **Particle Tracing**—to operate on 1-bit/ternary trits, we identify pose-invariant loops and curves. This method provides the "Animal Specialist" logic required to distinguish biological forms from rigid machines on natural images without learned parameters.

## Key Contributions
1. **Discrete Lagrangian Vorticity:** Implementation of the Curl operator on ternary gradient fields to identify rotational energy cores (vortices) and topological junctions.
2. **Structural Particle Tracing:** A Lagrangian framework that spawns particles at high-gradient points to map continuous edge length, curvature, and loop closure.
3. **Integral vs. Differential Law:** Empirical proof that Integral Measures (Divergence, Tracing) are superior to Differential Measures (Hessian, Laplacian) in heavily quantized manifolds.

## Results
- **CIFAR-10:** Broke the 50% ceiling using topological skeletons as gating logic for geometric rankers.
- **Organic Separation:** Lagrangian features achieve 2.6x random accuracy on grayscale natural images by identifying topological "loops."
- **Pose Invariance:** Accuracy retention across variable orientations where static block-matching fails.
