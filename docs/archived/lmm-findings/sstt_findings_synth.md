# LMM Phase 4: SYNTHESIZE — The Integrated Ternary Engine

## The Decision
We will finalize the **Three-Tier Router** as the project's primary architecture. We move from a collection of experiments to a **Unified Inference Engine** that optimizes for the speed-accuracy Pareto frontier.

## Action Plan: The Final Arc

### 1. Robust Routing (Immediate)
- Update `sstt_router_v1.c` with the red-team findings: Implement dataset-aware weighting (1.0x for noisy textures, 0.5x for clean strokes) and lock the 198/200 unanimity threshold as the "Safe Fast Path."

### 2. Lagrangian Gating (T3 Optimization)
- Instead of "fusing" Lagrangian and Eulerian features (which causes the -14pp drop), implement **Sparse Topological Gating.**
- **Logic:** If `tracer.closure_rate > threshold` (detected loop), route to "Animal Specialists." Else, route to "Machine Specialists."

### 3. Scaling Path (224x224)
- Adopt the **Uniform Knot Grid** as the standard high-resolution path. Abandon variance-guided placement for global geometric tasks. 
- Target: Validate a cache-resident ImageNet-scale (224x224) hot map with <1ms latency.

## Success Criteria
- ** MNIST Performance:** >96.5% global accuracy at <0.7ms latency.
- ** Fashion Performance:** >85% global accuracy at <1.0ms latency.
- ** CIFAR-10 Milestone:** Break 53% using Lagrangian gating in the Deep Tier.
- ** Architectural Goal:** Maintain zero learned parameters and zero floating point at inference.

## The Clean Cut
The project has revealed that ternary inference is not just about quantization; it is about **Retrieval-Gated Geometry.** We retrieve texture-plausible candidates using a sparse index, then resolve the final class using high-precision geometric fields.
