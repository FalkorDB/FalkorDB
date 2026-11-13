# M1: ordering quality of the design's mutual-max matching against feral's SHEM

**Verdict: PASS, with conditions.** On all three graphs, the design's matching (§3 step 3) gives fewer arcs and fewer triangles than stock feral. The median over 5 seeds is lower on every graph, and every one of the 15 paired (graph, seed) runs is lower. The pass depends on three details §3 leaves loose. Each of them, when dropped, fails the gate:

- the maxvwgt cap;
- the parallel 2-hop pass;
- at least 3 proposal rounds.

Ordering time is 21–27 % slower in this serial prototype.

## Setup

- **Code.** feral-metis f127189 is copied with feral-ordering-core and feral-amd into `scratchpad/m1match/crates`. Only `coarsen.rs` and `MetisOptions` changed.
  - The stock path is byte-for-byte equivalent: on all 3 graphs the copy's permutation equals the original crate's (`m1bench same`, 1 run each).
  - The crate's 68 unit tests pass.
- **Mutual-max matching.** This is plain Rust that does what the GraphBLAS formulation does, one synchronous round at a time:
  - Each unmatched vertex proposes to its unmatched neighbour with the largest key `(w << 32) | fmix32(u ^ salt)`. The salt is drawn per level, and the scramble is a bijection, so keys are unique within a row.
  - Mutual proposals match.
  - The rounds repeat over the still-unmatched vertices. All proposals in a round read the state from the start of that round, so the result does not depend on visit order.
  - **Cap:** a pair is allowed only if `vwgt[v] + vwgt[u] ≤ 1.5·tvwgt/CoarsenTo` (METIS `coarsen.c:38`), with CoarsenTo set to the `coarsen_floor` of the call.
  - **2-hop pass:** it runs when a level shrinks by less than 15 %, the same trigger as SHEM. Each unmatched vertex takes its heaviest neighbour as an anchor. Vertices that share an anchor pair up in scramble order, subject to the cap. In GraphBLAS this is a build plus a row sort.
  - **Round cap in the main runs: none.** Rounds run until a round matches nothing, which gives a maximal matching. Observed rounds per level: p50 5, p99 7, max 10. Sensitivity to the cap is below.
- **Data.** SNAP roadNet-CA, PA and TX: undirected skeletons with self-loops dropped and isolated nodes ranked first, as `nd.rs` does.
- **Metrics.**
  - Arcs come from the harness's elimination game (Liu's elimination tree, then child absorption).
  - Triangles are Σ C(\|up(x)\|, 2), the same definition as feral's commit e6db51a.
  - Every run also checks that every skeleton edge is an arc, that parent(r) = up(r)[0], and that closure holds by explicit lookup. All 97 ordering runs passed all three.
  - Height is the number of nodes on the longest leaf-to-root path.
- **Machine and timing.**
  - Apple M5, 10 cores, 32 GB. Every run is single threaded. Load average was 2.6–2.9 from other sessions.
  - Time is the wall time of `metis_order_full` alone. Each seed gets the median of 3 repetitions, and the table gives the median across seeds. All 3 repetitions produced the same permutation.
  - Peak RSS was about 1.05 GB on CA.

**Not the stack's California.** My roadNet-CA skeleton has 1,971,281 nodes (6,075 isolated) and 5,533,214 entries. The stack's skeleton has 1,540,096 nodes (294k isolated) and 3,313,386 entries (u1 §1, `nd.rs`). So the 5,838,003 / 53.9M (feral) and 5,527,199 / 48.6M (C METIS) figures do not apply here. For a reference on this graph, I ran C METIS through the existing driver `scratchpad/cmetis-driver/nd` with default `METIS_NodeND` options and 3 runs; I did not re-verify how that binary was built.

## Results (5 seeds; medians, with the range in brackets)

| Graph | Ordering | Arcs | Triangles | Tree height | Order time (s) |
|---|---|---|---|---|---|
| CA | feral SHEM | 12,631,332 [12.59–12.70M] | 219.0M [210.4–229.6] | 941 [859–976] | 7.07 |
| CA | **mutual-max** | **12,523,852** [12.51–12.59M] | **208.0M** [204.1–215.8] | **900** [856–961] | 8.55 |
| CA | C METIS (1 run) | 11,864,871 | 194.9M | 843 | 4.95 |
| PA | feral SHEM | 7,694,981 [7.62–7.72M] | 186.3M [174.4–191.2] | 878 | 3.60 |
| PA | **mutual-max** | **7,597,818** [7.56–7.61M] | **174.3M** [165.3–176.1] | **854** | 4.57 |
| PA | C METIS | 7,219,125 | 163.7M | 845 | 2.61 |
| TX | feral SHEM | 8,456,370 [8.42–8.49M] | 113.0M [111.6–118.8] | 702 | 4.59 |
| TX | **mutual-max** | **8,398,806** [8.38–8.40M] | **111.2M** [107.5–111.7] | **691** | 5.73 |
| TX | C METIS | 7,920,038 | 100.1M | 631 | 3.33 |

**Paired by seed** (mutual-max ≤ SHEM):
- Arcs: 5/5 on every graph, from −0.13 % to −1.72 %.
- Triangles: 5/5 on every graph, from −0.2 % to −10.9 %.
- Height: 4/5 on each graph.

**Gap to C METIS on these graphs:**

| | SHEM arcs | Mutual-max arcs | SHEM triangles | Mutual-max triangles |
|---|---|---|---|---|
| CA | +6.5 % | +5.6 % | +12.4 % | +6.7 % |
| PA | +6.6 % | +5.2 % | +13.8 % | +6.4 % |
| TX | +6.8 % | +6.0 % | +12.9 % | +11.1 % |

## Coarsening: level count and shrink per level

- **Coarsening the largest component to ≤ 100 vertices, the design's cell floor** (seed 1, one call):
  - SHEM takes 16, 15 and 16 levels (CA, PA, TX). Every level shrinks to 0.527–0.549 of its input.
  - Mutual-max takes **30, 27 and 28** levels. The matching's shrink falls from 0.58 at the first level to 0.80–0.85 at deep levels. Every 2–3 levels the 2-hop pass fires and takes that level to about 0.54. The weighted mean is 0.606–0.608 against 0.543.
  - The coarsening itself is faster: 0.35 s against 0.63 s on CA.
- **Top ND bisection** (CoarsenTo = n/30): SHEM takes 6 levels at 0.533–0.549. Mutual-max takes 8 levels at 0.58 → 0.77, and the last of them uses the 2-hop pass.
- **Whole ordering:** mutual-max builds 34 % more levels on CA (59.5k against 44.5k), and 35 % more on PA and TX. The 2-hop pass runs on about 18 % of them; SHEM never triggered it.
- **Where the extra time goes:** node FM runs once per level, so more levels mean more FM. That explains why ordering is slower even though coarsening is faster (derived from the level counts; not profiled).
- **Design consequence:** about twice as many coarsening depths means about twice as many GraphBLAS calls per cell (derived).

## What carries the result (diagnostics, 1 repetition per seed)

| Variant | CA arcs / triangles | Paired against SHEM (arcs, triangles) over CA / PA / TX |
|---|---|---|
| mutual-max without the cap | 12.68M / 226.6M | 2/5 · 4/5 · 1/5: **fails** on CA and TX |
| SHEM with the cap | 12.62M / 214.5M | 3/5 · 4/5 · 2/5: mixed |
| mutual-max with feral's sequential 2-hop | 12.69M / 226.7M | 1/5 on CA: **fails** |
| round cap 1 | 12.99M / 249.8M | 0/5: fails |
| round cap 2 | 12.72M / 224.5M | 0/5 arcs, 2/5 triangles: fails |
| round cap 3 | 12.57M / 209.0M | 4/5 |
| round cap 4 | 12.57M / 208.3M | 5/5 |
| round cap 6 | 12.57M / 213.5M | 4/5 arcs, 3/5 triangles |
| no round cap | 12.52M / 208.0M | 5/5 |

- **The cap is load-bearing.** Without it, mutual-max builds coarse vertices up to 195k in weight on CA, against about 29k with the cap. Heavy-edge proposals pile onto vertices that are already heavy.
- **Rounds.** Caps of 3 and above are within seed noise of no cap. Caps of 1 and 2 fail. Recommendation: **a cap of 8 rounds**, which covers the observed p99 of 7. Round caps were measured on CA only.
- **The cap needs mutual-max.** With the cap, mutual-max beats SHEM-with-cap on arcs in 15/15 paired runs.

## Deliberate breaks

Each check below was broken on purpose to show it can fail:

- **Closure checker.** Dropping one needed arc from rank 6076 gives 1 violation; the intact supergraph gives 0.
- **Arc and triangle counts respond to a worse order.** Feral's order with 1,000 random position swaps gives 13.52M arcs, 426.5M triangles and height 1,680, against 12.67M, 226.8M and 971.
- **The metric sees matching quality.** Mutual-*min* proposals with the cap (3 seeds on CA) give 12.87M arcs and 250.5M triangles: worse than both SHEM and mutual-max.

## Unmeasured

- A GraphBLAS implementation, and its time per call and per level.
- That the order is the same at every thread count. It holds by construction here: proposals are synchronous and keys unique.
- The stack's 1.54M-node California and a 9M-edge graph.
- Query p99 and memory peak with these orders.
- Round caps on PA and TX.

## Changes to make in the design (§3 step 3)

1. Make the maxvwgt cap mandatory, using METIS's formula above. It is no longer an open item for PR 6.
2. Specify the 2-hop pass as anchor grouping. Feral's sequential pass fails the gate.
3. Set the round cap to 8.
4. Budget for about twice as many coarsening depths as SHEM.

Scratch code and raw output are in `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1match/` (`bench/src/main.rs`, `crates/feral-metis/src/coarsen.rs`, `out/*.txt`, `sweep.sh`).
