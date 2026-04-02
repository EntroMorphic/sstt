# Nodes of Interest: SSTT Step-Changes

## Node 1: The ranker and the retriever disagree about what matters
The inverted index ranks candidates by IG-weighted signature frequency. The structural ranker ranks by divergence, centroid, and profile similarity. These are different objective functions operating on different representations. The 0.87pp gap at K=50 is the measured cost of this misalignment.
Why it matters: This isn't a tuning problem. It's a design problem. The two stages were built independently.

## Node 2: K-sensitivity reveals unexploited capacity in the structural ranker
Accuracy climbs from 96.59% (K=50) to 97.61% (K=1000). The structural ranker finds signal in candidates the dot product ignores. 34 additional images classified correctly between K=200 and K=1000.
Why it matters: The current system is leaving accuracy on the table because the retrieval stage throttles the ranking stage.

## Node 3: Retrieval recall is near-identical across methods but accuracy is not
SSTT inverted index: 99.63% recall. Brute L2: 99.73% recall. Only 0.1pp difference. But accuracy differs by 0.87pp. The correct class is present in both cases — the difference is in the quality of the surrounding candidates.
Why it matters: Retrieval recall is necessary but not sufficient. The composition of the candidate set — not just whether it contains the right answer — determines ranking accuracy.

## Node 4: The dot product is saturated
K-invariant from K=50 to K=1000 at 96.28%. It has extracted all the signal it can from the ternary representation. More candidates don't help it. It sees the world in one resolution and has reached that resolution's ceiling.
Why it matters: Any improvement in the system must come from the structural features or from a new signal entirely. The dot product is done.

## Node 5: The 7 Mode A failures define the hard ceiling
7 images where the correct class is absent from the top-K entirely. These are the only errors that better ranking cannot fix. Everything else — 266 errors — is ranking failure with the answer sitting in the candidate pool.
Why it matters: The system has 266 solvable errors. That's a large target. 97.27% → potentially 99.93%. The question is how much of that gap is reachable.

## Node 6: CIFAR-10's class separation pattern is informative
Ship (63.7%) vs cat (33.6%). Edge-distinctive classes work; texture-dependent classes fail. The architecture captures topology but not texture.
Why it matters: This tells us what ternary block features can and cannot represent. Any step-change for MNIST must work within the topology-captures-well regime. CIFAR-10 improvement requires a different kind of feature.

## Node 7: The sequential processing asymmetry
Bayesian sequential processing helps Fashion (+1.14pp) but not MNIST (+0.00pp). Fashion's candidate pools are more heterogeneous. Sequential discrimination works when there's diversity to exploit.
Tension with Node 2: K-sensitivity also increases with candidate pool diversity. These may be the same phenomenon.

## Node 8: Speed is the inverted index's actual advantage
2.75x faster than brute L2. The entire argument for the inverted index is computational. If a faster retrieval method produced L2-quality candidates, the inverted index would have no role.
Tension with Node 1: The IG weighting was designed to maximize classification accuracy at the hot-map level (91.83%). Its success there doesn't mean it's optimal for candidate retrieval.

## Node 9: The encoding may be the real bottleneck
3x1 horizontal blocks, 27 ternary values. This is a very coarse representation. Two images that are visually distinct can have identical block signatures. Two images that are visually similar can differ at a single trit. The encoding resolution limits what both retrieval and ranking can see.
Why it matters: Every component downstream — IG weights, inverted index, dot products, structural features — operates on this encoding. If the encoding is lossy in the wrong places, nothing downstream can recover the lost information.

## Tensions

- **Node 1 vs Node 8:** The retriever and ranker are misaligned, but fixing alignment might sacrifice the speed advantage that justifies the inverted index.
- **Node 2 vs Node 4:** The structural ranker wants more candidates but the dot product is already saturated. Which one should shape the architecture?
- **Node 9 vs everything:** If the encoding is the bottleneck, all downstream improvements are bounded by what the encoding can represent. Should effort go into better encoding or better ranking?
