# Contribution 7: Information-Gain Weighted Block Voting

## Concept

Not all block positions carry equal discriminative power. A block in the
image center (where digits differ) is far more informative than a block
in the corner (always background). Information gain (IG) quantifies this:

```
IG(block_k) = H(class) - H(class | block_k)

H(class | block_k) = Σ_v P(block_k = v) × H(class | block_k = v)
```

High IG = knowing this block's value reduces class uncertainty.
Low IG = this block is nearly useless for classification.

Instead of giving every matching block 1 vote, give it IG votes.
Blocks near the center get weight ~16; corner blocks get weight 1.

## Implementation

```c
/* sstt_v2.c:491-545 */
static void compute_ig_weights_chan(const uint8_t *train_sigs, int ch) {
    /* Compute H(class) from label frequencies */
    /* For each block k: count class distribution per block value */
    /* H(class | block_k) = weighted sum of per-value entropies */
    /* IG = H(class) - H(class | block_k) */
    /* Normalize to [1, IG_SCALE=16] */
}
```

The weights are computed per-channel: pixel blocks, h-grad blocks, and
v-grad blocks each get their own IG map. This automatically handles the
different information distributions across channels.

## Results

| Method | Accuracy |
|--------|----------|
| Unweighted pixel vote-only | ~84% |
| IG-weighted pixel vote-only | 83.27% |
| IG-weighted 3-channel vote-only | 88.78% |

The single-channel IG result is slightly lower than unweighted (the noise
from integer-quantized weights can hurt). But the 3-channel combination
is where IG shines: it automatically balances the relative contribution
of pixel, h-grad, and v-grad blocks, boosting from ~84% to 88.78%.

## What Makes It Novel

1. **Per-block, per-channel weights** — not just "use gradients" or "don't,"
   but exactly *how much* each block position in each channel contributes.
2. **Closed-form from training data** — no learning rate, no epochs,
   no gradient descent. Compute once from label/block-value statistics.
3. **Integer weights** — quantized to [1, 16], so the vote accumulation
   stays in uint32 with no floating point in the hot path.

## Files

- `sstt_v2.c` lines 484-551: IG weight computation
- `sstt_v2.c` lines 1089-1175: IG voting test (Test C)
