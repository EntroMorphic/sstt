# Raw Thoughts: SSTT Step-Changes

## Stream of Consciousness

I have been working on this project all day and I am close to it. Too close. I need to step back and be honest about what I actually understand versus what I've been telling myself.

The project achieves 97.27% on MNIST with zero learned parameters. That number has been the headline for a while. But today two experiments broke open something I wasn't expecting. The K-invariance finding — the one we've been calling the central result — doesn't hold for the full system. The cascade is K-invariant. The structural ranker is not. I treated this as a correction to the paper. But I haven't sat with what it actually means.

What it means is that the structural features see something the dot product doesn't. And they see more of it the more candidates you give them. At K=1000 the system hits 97.61%. That's 34 more images correct than at K=200. Where are those 34 images coming from? Candidates ranked 201-1000 by vote — images the inverted index retrieved but deemed low-priority. The structural ranker disagrees with the inverted index about which candidates matter. That tension is real and I haven't understood it.

The retrieval comparison made it worse. Brute L2 at K=50 beats the inverted index at K=50 by 0.87pp when both feed the same ranker. The inverted index is faster. But "faster at producing worse candidates" is not the story I've been telling. I've been saying the inverted index solves retrieval. It doesn't solve it — it approximates it with a speed advantage.

What actually scares me: maybe the entire inverted-index architecture is the wrong abstraction. Maybe the right approach is brute L2 at K=200 with a structural ranker, and the IG-weighted vote was a detour that happened to work well enough to publish. The speed advantage is real (2.75x) but if someone with a GPU can do brute L2 at K=200 in 100us, the inverted index becomes irrelevant.

What I don't understand: why does brute L2 produce better candidates for the structural ranker? Both methods achieve ~99.7% retrieval recall — the correct class appears in both candidate sets almost always. So it's not about whether the right answer is present. It's about what else is present. The L2 candidates are geometrically closer to the query. Does that mean the structural features work better on geometrically similar images? That would make sense — divergence and centroid comparisons are more discriminative when the images are already similar in pixel space.

What I might be wrong about: I assumed IG weighting is the right retrieval criterion because it maximizes class-conditional information. But maximizing information about class membership is not the same as maximizing information useful to the structural ranker. These are different objectives. I've been conflating them.

Half-formed idea: what if the inverted index and the structural ranker are solving the same problem from opposite ends? The inverted index finds images that are statistically similar (same block patterns in the same positions). The structural ranker finds images that are geometrically similar (same divergence, same centroid, same profile). The ideal candidate set would be images that are both. But the inverted index doesn't know about geometry and the ranker doesn't influence retrieval.

## Questions Arising

- Why exactly do L2 candidates score higher under the structural ranker? Is it the candidates themselves or the absence of noisy candidates that the inverted index includes?
- Is there a closed-form way to compute retrieval weights that optimize for structural-ranker performance rather than class prediction?
- What do the 34 extra-correct images at K=1000 vs K=200 look like? Are they hard cases where the correct class had low vote counts?
- Could a simple post-retrieval filter (remove candidates with low L2 similarity) close the gap without brute search?
- What would happen if you used the structural features as the primary retrieval signal instead of block signatures?

## First Instincts

- Try hybrid retrieval (inverted index → L2 re-rank). Safe bet. Probably works. But is it a step-change or a patch?
- The real step-change might be rethinking what retrieval means in this system. Not "find images with the same block patterns" but "find images the ranker can distinguish."
- I keep coming back to the 7 Mode A failures. 7 out of 60,000 is vanishingly small. The retrieval is almost perfect. So the problem really is ranking. Everything should focus there.
