# Contribution 47: Structural Particle Tracing (Grid-Lagrangian)

This experiment shifts from Eulerian histograms (counting static events) to a true Lagrangian approach by tracing edge structures through the ternary field.

## The Tracer Algorithm
Instead of looking at pixels in isolation, the algorithm spawns "particles" at high-gradient points and follows the flow perpendicular to the gradient.
1. **Structural Length:** How far a continuous edge travels before breaking.
2. **Structural Curvature:** The rate of direction change along a traced path.
3. **Closure Rate:** The probability that a traced path forms a closed loop.
4. **Vortex Intensity:** The cumulative rotational energy (Discrete Curl) in a region.

## Experimental Results (5000 Train / 200 Test slice)

| Feature Set | Dimensionality | Accuracy (k=1) |
|-------------|----------------|----------------|
| Global Tracer | 4 scalars | 14.00% |
| **Grid Tracer (4x4)** | 64 floats | **18.00%** |
| Random Baseline | - | 10.00% |

## Value Proposition
The tracer provides a **topological skeleton** of the image. 
- **Organic shapes (Frogs, Birds):** High curvature, high closure rate.
- **Machine shapes (Ships, Trucks):** Low curvature, high structural length (long straight lines).

This feature is designed to be the "Animal Specialist" in the Tier 3 deep-ranker, resolving ambiguities where texture and gradient histograms are diluted by background noise.

---
## Files
- Code: `src/sstt_cifar10_grid_tracer.c`
