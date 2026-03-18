# Contribution 46: Discrete Lagrangian Vorticity (Curl)

This experiment introduces the **Discrete Curl** operator to the ternary gradient field to identify topological junctions (vortices) and distinguish different shape types.

## Theoretical Basis
In a 2D field $\vec{F}$, the Curl is defined as:
$$\text{Curl}(\vec{F}) = \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}$$

In our discrete ternary grid:
$$\text{Curl}(x,y) = (V_{x+1,y} - V_{x,y}) - (H_{x,y+1} - H_{x,y})$$
where $V$ is the vertical gradient and $H$ is the horizontal gradient.

Values range from -4 to +4.
- **Positive/Negative Curl:** Indicates rotational energy (corners, joints).
- **Zero Curl:** Indicates laminar flow (straight edges) or flat regions.

## Experimental Results (5000 Train / 100 Test slice)

| Method | Baseline (k=1) | Cascade (Texture → Rank) |
|--------|----------------|--------------------------|
| Curl-Only | 14.00% | 26.00% |
| **Gauss + Curl** | 33.00% | **39.00%** |

*Note: These numbers use a simplified ternary-only Gauss map for faster iteration. The 39% result on a small data slice confirms that Curl adds a complementary signal to the geometric histogram.*

## Lagrangian vs Eulerian
The current implementation is still largely Eulerian (counting events in fixed grid boxes). A true Lagrangian step-change would involve:
1. **Particle Advection:** Tracing paths along the gradient field.
2. **Vortex Core Detection:** Identifying the centers of rotational energy.
3. **Loop Closure:** Detecting topological holes by following the curl flow.

## Next Steps
Integrate the full RGB Gauss map with the Curl features to see if the combination breaks the 53% mark on the full CIFAR-10 test set.

---
## Files
- Code: `src/sstt_cifar10_lagrangian.c`
