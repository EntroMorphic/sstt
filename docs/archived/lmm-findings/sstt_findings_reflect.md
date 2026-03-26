# LMM Phase 3: REFLECT — Underlying Geometry & Truth

## 1. The Coherence of Noise
The router's success reveals a fundamental truth about ternary retrieval: **Unanimity is a proxy for signal-to-noise ratio.** When the top candidates from an inverted index lookup all agree, the high-frequency "noise" of individual block matches has reached coherence. At this point, further ranking is redundant. This is why Tier 1 achieves 99.9% accuracy—it only triggers when the manifold is "flat" and unambiguous.

## 2. The Manifold Hole (The Variance Failure)
The failure of variance-guided knot placement (Doc 55) provides deep leverage. It teaches us that **a ternary representation is a geometric manifold, not just a statistical distribution.** Variance identifies the "active" boundaries (edges), but the "stable" interiors (flat regions) are what give those boundaries context. By clustering knots at the edges, we introduced "holes" in the manifold, destroying the spatial context necessary for L1 distances to work. 

## 3. Lagrangian vs Eulerian: The Dimensionality Shift
Our current best CIFAR-10 ranker (RGB Gauss) is **Eulerian**—it counts events at fixed coordinates. The Lagrangian shift (Curl/Tracing) is a **Dimensionality Reduction**. A 720-bin histogram is high-resolution but brittle. A single "loop closure" bit from a tracer is low-resolution but invariant.
- **The Truth:** The ceiling for natural images isn't more bins; it's **Topological Gating.** Use the Lagrangian "skeleton" to select the specialized Eulerian "sub-space."

## 4. Hardware Realism
The discovery that 87% of compute is memory access (scattered writes) is the "ground-state" constraint of the project. It means that **architectural efficiency is memory efficiency.** Tiered routing isn't just about saving FLOPs; it's about saving **Bus Cycles.** By bypassing the dot-product step, we avoid the heavy memory-read overhead of retrieving training image signatures from RAM.
