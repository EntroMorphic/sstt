# Contribution 8: Multi-Probe Hamming-1 Expansion

## Concept

Exact block matching is brittle: a single trit flip in a 3-trit block
creates a completely different block value, losing all votes from that
position. Multi-probe relaxes this by also voting on Hamming-distance-1
neighbors — blocks that differ in exactly one of the three trits.

Each block value v has exactly 6 Hamming-1 neighbors (3 trit positions ×
2 alternative values). The neighbor votes are given half the weight of
exact matches, reflecting lower confidence.

## Architecture

```
For each query block value v at position k:
  1. Exact match:  vote IDs at (k, v) with full weight w
  2. For each of 6 Hamming-1 neighbors nv:
     vote IDs at (k, nv) with half weight w/2
```

This is analogous to multi-probe LSH (locality-sensitive hashing), but
in the ternary block code space rather than random projection space.

## Implementation

```c
/* sstt_v2.c:561-582 — Neighbor table construction */
static void build_neighbor_table(void) {
    for (int v = 0; v < N_BVALS; v++) {
        /* Decode v to 3 trits */
        int t0 = (v / 9) - 1, t1 = ((v / 3) % 3) - 1, t2 = (v % 3) - 1;
        int nc = 0;
        for (int pos = 0; pos < 3; pos++) {
            for (int alt = 0; alt < 3; alt++) {
                if (trits[alt] == orig[pos]) continue;
                /* Flip one trit, encode as neighbor */
                nbr_table[v][nc++] = block_encode(modified trits);
            }
        }
        nbr_count[v] = nc;  /* always 6 */
    }
}

/* sstt_v2.c:1193-1220 — Multi-probe voting */
for (int k = 0; k < N_BLOCKS2D; k++) {
    uint8_t bv = qsig[k];
    /* Exact match at full weight */
    for (j = 0; j < sz; j++) votes_probe[ids[j]] += w;
    /* Hamming-1 neighbors at half weight */
    for (int n = 0; n < nbr_count[bv]; n++) {
        uint8_t nv = nbr_table[bv][n];
        for (j = 0; j < nsz; j++) votes_probe[nids[j]] += w_half;
    }
}
```

## Results

| Method | Accuracy |
|--------|----------|
| Exact-only 3-chan IG vote | 88.78% |
| Multi-probe 3-chan IG vote | 91.16% |

Multi-probe adds +2.38 pp over exact-only, recovering votes from
images that differ by a single quantization boundary in one block.

## What Makes It Novel

1. **Deterministic neighbor structure** — unlike LSH with random hash
   functions, the Hamming-1 neighborhood of a ternary block code is
   fixed and small (exactly 6 neighbors). No randomization needed.

2. **Half-weight heuristic** — simple, effective, no tuning. A neighbor
   match means "one of three trits was different" — roughly 2/3 as
   informative as an exact match, so half weight is conservative but safe.

3. **Stacks with IG weights** — the half-weight is applied to the
   IG-scaled weight, so informative blocks get large neighbor votes
   and uninformative blocks get tiny ones. The two techniques compose.

## Files

- `sstt_v2.c` lines 553-582: neighbor table construction
- `sstt_v2.c` lines 1177-1247: multi-probe voting test (Test D)
- `sstt_v2.c` lines 1249-1425: full cascade using multi-probe (Test E)
