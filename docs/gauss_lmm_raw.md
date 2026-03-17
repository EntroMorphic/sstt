# Gauss Map + Full Architecture — LMM RAW

Unfiltered thinking on what we haven't combined yet.

---

I keep declaring ceilings and the user keeps breaking them. The pattern:

- "41% is the ceiling" → stereoscopic: 41.18%
- "42% is the ceiling" → stereo+MT4: 44.48%
- "44% is the ceiling" → Gauss map: 48.31%

Every time I said ceiling, it was because I'd exhausted the techniques
I was COMBINING. The user then asked "are we using everything we have?"
and the answer was always no.

So: what haven't we combined?

INVENTORY OF WHAT WORKS INDEPENDENTLY:

1. Block-based stereo Bayesian: 41.18% (3 eyes, Encoding D, hot map)
2. MT4 81-level dot product: adds +3pp to ranking
3. Topological features (divergence, grid div): adds +2.8pp to ranking
4. Bayesian prior as ranking feature: adds +4.6pp
5. Grid Gauss map kNN: 48.31% (4×4 grid, grayscale, brute kNN)
6. Per-channel RGB Gauss maps: 36.07% global (4-channel, 180 features)
7. Inverted-index voting: 98% recall at top-200
8. Confidence map / vote agreement: 100-200x discriminative power
9. Flattened RGB color encoding: +10pp over grayscale
10. Adaptive quantization: cat +9.8pp
11. Information gain weighting: auto-adapts per block per dataset

WHAT HASN'T BEEN TRIED:

A. Grid Gauss map on RGB channels (not just grayscale). The 48.31%
   uses grayscale only. Per-channel grid Gauss maps would have
   4 channels × 16 regions × 45 bins = 2880 features.

B. Gauss map as a VOTING feature (not just kNN). Build an inverted
   index on quantized Gauss map bins. Vote across Gauss map bins
   like we vote across block signatures.

C. Gauss map + stereoscopic. Three Gauss map eyes with different
   gradient threshold parameters.

D. Gauss map as a RANKING feature within the block-based cascade.
   After block-based voting retrieves top-200 candidates, re-rank
   using Gauss map L1 distance (not dot product).

E. Finer Gauss map grids. 4×4 gave 48.31%. What about 8×8 (64
   regions × 45 bins = 2880 per channel)?

F. Gauss map on the FLATTENED RGB (32×96). The h-gradient of
   flattened RGB captures R→G, G→B color transitions. The Gauss
   map of this field would encode color edge geometry.

G. MT4 Gauss map. Instead of computing gradients on ternary, compute
   them on the 81-level MT4 representation. More precise gradients →
   more precise sphere positions.

H. Using block-based voting to RETRIEVE candidates, then Gauss map to
   RANK them. The inverted index is fast (98% recall) and the Gauss
   map is discriminative (48%). Use each for what it's best at.

Actually, H is the key. The fusion experiments failed because we tried
to COMBINE SCORES. But the right architecture is SEQUENTIAL:

1. Block-based inverted index → retrieves top-200 candidates (fast, high recall)
2. Gauss map L1 → re-ranks the 200 candidates (accurate, shape-based)
3. k=1 argmax

This uses the block system for RETRIEVAL (where it excels: 98% recall)
and the Gauss map for RANKING (where it excels: 48% accuracy). Neither
system is asked to do what it's bad at.

The previous fusion failed because we tried to ADD Bayesian posteriors
to Gauss map votes — incompatible scales, the Bayesian posterior
dominated. The sequential approach avoids this: the block system
produces a CANDIDATE SET, not a classification. The Gauss map
classifies within that set.

This is exactly the vote-then-refine cascade from MNIST, but with
a Gauss map ranking step instead of a dot product ranking step.

Also: the Gauss map kNN is currently brute-force (O(50000) per test
image). Using the inverted index to narrow to 200 candidates first
would make it 250x faster AND potentially more accurate (the
candidates are already pre-filtered for block similarity).

Wait — would the pre-filtering HELP accuracy? The block system
retrieves candidates that share texture. The Gauss map then selects
among them by shape. If the top-200 candidates all have the right
TEXTURE but wrong SHAPE (deer vs frog), the Gauss map can resolve it.
But if the top-200 excludes the correct class (2% miss rate), the
Gauss map can't fix it.

At 98% recall, 200 out of 10000 images lose the correct class in
retrieval. For the other 9800, the Gauss map gets to classify among
200 candidates that are texture-similar. This should be EASIER than
brute-force over 50000 because the irrelevant candidates (airplanes
when the image is a frog) are already filtered out.

Actually no — if the query is a deer, the top-200 includes deer AND
frogs (same texture). The Gauss map's job is to distinguish deer from
frogs among these 200. With brute-force kNN, the Gauss map compares
against ALL 50000 training images, including many deer that are NOT
texture-similar. The brute-force might actually have better nearest
deer candidates than the texture-filtered 200.

Hmm. Let me think about this differently. The brute Gauss map at
48.31% finds the nearest training image in SHAPE space across all
50000. The cascade version would find the nearest in shape space
among the 200 most TEXTURE-similar. These are different sets.

The question: is the nearest shape match among texture-similar
candidates better or worse than the nearest shape match overall?

For deer vs frog: the texture-similar candidates include both deer
and frogs. The Gauss map picks the closest shape among them. Since
deer and frogs have different shapes (round vs elongated), the Gauss
map should correctly pick deer. This is BETTER than brute force
because the brute-force candidate set includes EVERYTHING — the
nearest Gauss map neighbor might be a bird that happens to have
similar edge geometry but different texture.

So the cascade (texture retrieval → shape ranking) might EXCEED both
systems individually. It constrains the search to texture-plausible
candidates and then selects by shape.

This is the same insight as the original SSTT cascade: retrieval
narrows the field, ranking selects the winner. We just need to use
the RIGHT retrieval and the RIGHT ranking for CIFAR-10.

E and F are also worth testing quickly. But H is the priority.
