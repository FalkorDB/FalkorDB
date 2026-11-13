# M1, D5 option (b): ordering quality when deeper ND depths reuse the depth-0 coarsening

**Verdict: FAIL as specified.** Reusing the depth-0 matching at every deeper ND depth always gives a worse order than mutual-max re-coarsening (R):
- **Against R:** more arcs and more triangles in all 15 paired (graph, seed) runs. Arcs are +0.36 % to +1.56 %, triangles +2.6 % to +16.0 %.
- **Against stock feral SHEM (the M1 floor):** worse on arcs and triangles on all 5 CA seeds. On PA and TX together, arcs are at or below SHEM in 6/10 runs and triangles in 3/10.

The mechanics work and are cheap:
- B1 and B2 build identical child matrices on all 653,998 levels compared.
- No cell ever needed fresh coarsening.
- Straddlers are rare. At the first coarse level they are 0.01–0.12 % of coarse vertices for depths 0–4, and they stay under 3.4 % over any depth's needed levels.

**What fails is the reused vertex weights.** The depth-0 matching was capped against the whole component, so it is too coarse for a smaller cell. In 80–100 % of reused cells at every depth (85–100 % on CA), the cell's heaviest coarsest vertex is above the cap R would have applied to that cell. On CA the median excess is 1.3× at depth 1 and 4–7× at depths 2–6.

**A diagnostic that stops reusing at the cap, called capstop, nearly recovers R's quality.** It stops reusing at the first level that would break R's cap and coarsens the rest fresh. Its median arcs are within 0.2 % of R's on PA and TX and +0.6 % on CA, and it is at or below SHEM in 14/15 runs. It coarsens 14.7 % of the vertices R coarsens after depth 0.

In plain Rust, Reuse is the fastest ordering, 32–34 % faster than R. That does not carry over to GraphBLAS; the cost stage measures that.

## Setup

- **Code.** The harness is the M1 copy of feral-metis f127189 with mutual-max matching (`m1-matching-quality.md`). The changes:
  - `MetisOptions` gains `reuse`, `reuse_check`, `reuse_stop_ge`, `reuse_capstop` and `reuse_break`.
  - `node_nd.rs`: the ND driver carries the ND depth and, under `reuse`, the depth-0 maps plus each vertex's depth-0 id. The stock `best_of_bisections` and `multilevel_node_bisection` are untouched; the reuse path calls copies of them.
  - `coarsen.rs`: the level log records the ND depth, and `contract` is public.
- **The stock and mutual-max paths are unchanged**, shown three ways:
  - `m1bench same` (stock copy against the original feral crate) gives identical orders on CA, PA and TX.
  - `m1bench samebase` (this crate against an untouched copy of the pre-reuse prototype) gives identical orders in 12/12 runs: SHEM and mutual-max, seeds 1–2, all 3 graphs.
  - The sweep's SHEM and R runs reproduce the earlier report's arcs, triangles, height, separator size and level count in 30/30 runs.
  - The crate's 68 unit tests pass.
- **What Reuse does, in feral's recursion:**
  - **Depth 0.** Each connected component is ordered exactly as R orders it.
    - For components of 5,000 or more vertices, that is best-of-5. The component is coarsened to max(100, n/30). Five runs each coarsen on to 120 and bisect, and the lightest separator wins.
    - The kept hierarchy A_0..A_L is the top stage plus the winning run's levels. It reaches ≤ 120 vertices for the whole component.
  - **Every deeper cell** gets B1: each side, and each connected component of a side, since that is how feral recurses.
    - For each depth-0 level in turn, the cell's coarse vertices are grouped by their depth-0 coarse vertex. A straddler's members inside the cell become one coarse vertex. Separator members are not in the cell, so they drop out.
    - The cell's previous level is then contracted with `contract`, R's own routine. This is the brief's "re-contract per child with the reused matching".
  - **Stop rule.** Reuse uses `coarsen`'s rule: stop at the first level at or below CoarsenTo. CoarsenTo is max(100, n/30) for the best-of-5 top stage, then 120.
    - The brief's literal rule ("the deepest level still at or above CoarsenTo") keeps one level less. It is run below as the `ge` diagnostic.
    - In best-of-5 cells, all five runs bisect the same reused bottom levels, so they differ only in their initial partitions. No matching runs after depth 0.
  - **Edge cases.**
    - A depth-0 level that merges nothing inside a cell would be skipped. This happened 0 times.
    - If a cell ran out of depth-0 levels above its floor, the rest would be coarsened fresh. This also happened 0 times; see "Where reuse needed fresh coarsening".
- **Depth 0 is the same as R's.** The largest component's depth-0 separator hash equals R's in 15/15 runs (`out/roots.txt`).
  - CA has 3 more components that get bisected (329–586 vertices each), and TX has one (27,502).
  - Each is bisected after the whole subtree of the component before it. Both variants draw from one random stream, so these components get different draws in R and Reuse. This is a harness artifact, not a reuse effect.
- **Checks.** Every Reuse run gets one extra, untimed repetition with the checks on. Its order must equal the timed repetitions' orders, and it did in all 15 runs.
  - **B1 against B2, at every split.** Each side X ∈ {A, B} gets its levels built twice.
    - B1 is the construction above.
    - B2 restricts the parent's level to X's clean coarse vertices (those with all members in X): the masked split. It re-contracts the straddler rows from X's previous level and fills their columns by transpose. Vertex weights are built the same way.
    - The two are compared as sorted (row, column, weight) lists keyed by depth-0 id, plus vertex weights.
    - **653,998 level comparisons, 0 differences.** The 15 `strad` runs repeated all of them with the same result.
  - **B1 levels are valid:** total weight is conserved, there are no self-loops, and weights are symmetric. 0 failures.
  - **Accounting identity at each parent level:** the coarse vertices summed over both sides equal the number of (parent coarse vertex, side touched) pairs. 0 failures.
  - **Mechanism:** the level log, tagged by ND depth, records 0 coarsening levels after depth 0 in every Reuse run. R records 33,351–59,760 per run.
  - **Order checks, carried over:** every skeleton edge is an arc, parent(r) = up(r)[0], and closure holds by explicit lookup. All 120 orders pass: 45 main, 15 strad, 30 roots and 30 diagnostic.
- **Straddler definitions.**
  - A **side straddler** is a parent coarse vertex with fine members in ≥ 2 of {A, B, separator}. These are the design's cells.
  - A **child straddler** uses the connected components of A and B instead, which are the cells feral recurses into.
  - **Needed levels** are those some child with more than 200 vertices coarsens with. Children of 200 vertices or fewer go to AMD and use no levels.
  - **Depth** below is the depth of the cell being split; its children, at depth + 1, receive the levels.
  - **Level** is the depth-0 level index.
- **Data and machine.**
  - Same as the earlier report: SNAP roadNet-CA, PA and TX (CA: 1,971,281 nodes, 5,533,214 entries). Apple M5, 10 cores, 32 GB. Every run is single-threaded.
  - Time is the wall time of `metis_order_full` alone. Each seed gets the median of 3 repetitions, and the tables give the median across seeds.
  - **Load.** Other sessions were running, and the 1-minute load at run starts was 1.2–16.4 (`LOAD` lines in `out/main.txt`).
    - CA seed 1's SHEM and R times (10.5 s and 11.9 s) are inflated by that load: 6.2 and 16.4 at their starts. The cross-seed medians are not affected.
    - The diagnostics ran at a load of 2.0–2.7.
  - **Peak RSS** was 1.93 GB for CA Reuse with checks on (`RSS` lines in `out/main.txt`). R used 1.13 GB and SHEM 1.07 GB. A 12 GB watchdog never fired; the highest it saw across all runs was 1.96 GB (`out/watchdog.log`).

## Results (5 seeds; medians, range in brackets) — `out/main.txt`, `out/tables.txt`

| Graph | Ordering | Arcs | Triangles | Tree height | Order time (s) |
|---|---|---|---|---|---|
| CA | feral SHEM | 12,631,332 [12.59–12.70M] | 219.0M [210.4–229.6] | 941 [859–976] | 7.20 |
| CA | mutual-max, re-coarsen (R) | 12,523,852 [12.51–12.59M] | 208.0M [204.1–215.8] | 900 [856–961] | 8.65 |
| CA | **mutual-max, reuse** | **12,705,568** [12.70–12.73M] | **232.8M** [229.3–237.5] | **955** [921–980] | **5.72** |
| CA | C METIS (1 run, earlier report) | 11,864,871 | 194.9M | 843 | — |
| PA | feral SHEM | 7,694,981 [7.62–7.72M] | 186.3M [174.4–191.2] | 878 [856–904] | 3.62 |
| PA | R | 7,597,818 [7.56–7.61M] | 174.3M [165.3–176.1] | 854 [830–891] | 4.59 |
| PA | **reuse** | **7,669,957** [7.63–7.72M] | **186.8M** [178.9–194.8] | **882** [871–907] | **3.08** |
| PA | C METIS | 7,219,125 | 163.7M | 845 | — |
| TX | feral SHEM | 8,456,370 [8.42–8.49M] | 113.0M [111.6–118.8] | 702 [690–739] | 4.69 |
| TX | R | 8,398,806 [8.38–8.40M] | 111.2M [107.5–111.7] | 691 [635–704] | 5.75 |
| TX | **reuse** | **8,441,100** [8.42–8.49M] | **115.0M** [113.1–121.6] | **710** [664–735] | **3.90** |
| TX | C METIS | 7,920,038 | 100.1M | 631 | — |

**Paired by seed** (reuse against R / against SHEM):

| Graph | Arcs vs R | Triangles vs R | Arcs ≤ SHEM | Triangles ≤ SHEM | Height ≤ R |
|---|---|---|---|---|---|
| CA | +1.04 % to +1.49 % (0/5 ≤) | +7.2 % to +16.0 % (0/5) | 0/5 | 0/5 | 2/5 |
| PA | +0.42 % to +1.56 % (0/5) | +2.6 % to +15.6 % (0/5) | 3/5 | 2/5 | 0/5 |
| TX | +0.36 % to +1.10 % (0/5) | +3.3 % to +8.8 % (0/5) | 3/5 | 1/5 | 1/5 |

**Gap to C METIS** (medians):

| | SHEM arcs | R arcs | Reuse arcs | SHEM triangles | R triangles | Reuse triangles |
|---|---|---|---|---|---|---|
| CA | +6.5 % | +5.6 % | +7.1 % | +12.4 % | +6.7 % | +19.5 % |
| PA | +6.6 % | +5.2 % | +6.2 % | +13.7 % | +6.5 % | +14.1 % |
| TX | +6.8 % | +6.0 % | +6.6 % | +13.0 % | +11.1 % | +15.0 % |

**Plain-Rust time.** Reuse is 32–34 % faster than R and 15–21 % faster than SHEM. It skips all matching after depth 0, and the five best-of-5 runs share their bottom levels. It still re-contracts every level of every cell (B1). This says nothing about GraphBLAS cost.

## Per depth: levels reused, fresh, and the weight problem (CA, sums over 5 seeds)

"Over R's cap": the cell's heaviest coarsest vertex exceeds 1.5·n/120, the bottom-stage cap R would use for that cell. "Weight ratio" is that vertex's weight divided by the cap. PA and TX show the same pattern (`out/tables.txt`).

| Depth | Cells | Levels reused (per cell) | Fresh levels | Over R's cap | Weight ratio p50 / p90 / max | R: levels coarsened |
|---|---|---|---|---|---|---|
| 1 | 20 | 243 (12.2) | 0 | 85 % | 1.33 / 1.69 / 2.50 | 932 |
| 2 | 20 | 405 (20.2) | 0 | 100 % | 4.07 / 4.25 / 5.74 | 1,616 |
| 3 | 40 | 739 (18.5) | 0 | 100 % | 6.44 / 8.45 / 10.15 | 2,961 |
| 4 | 81 | 1,362 (16.8) | 0 | 100 % | 6.95 / 8.59 / 11.27 | 5,159 |
| 5 | 163 | 2,507 (15.4) | 0 | 100 % | 6.77 / 8.49 / 11.99 | 7,793 |
| 6 | 327 | 4,496 (13.7) | 0 | 100 % | 5.63 / 9.00 / 15.37 | 11,598 |
| 7 | 659 | 7,599 (11.5) | 0 | 100 % | 2.55 / 4.65 / 12.58 | 16,690 |
| 8 | 1,320 | 12,793 (9.7) | 0 | 99 % | 1.83 / 2.56 / 6.12 | 21,827 |
| 9 | 2,650 | 20,986 (7.9) | 0 | 99 % | 1.61 / 2.14 / 3.66 | 24,751 |
| 10 | 5,318 | 31,706 (6.0) | 0 | 99 % | 1.88 / 2.50 / 3.75 | 33,220 |
| 11 | 10,471 | 45,011 (4.3) | 0 | 99 % | 1.94 / 2.50 / 3.67 | 47,443 |
| 12 | 18,364 | 55,954 (3.0) | 0 | 95 % | 1.60 / 2.20 / 3.50 | 60,132 |
| 13 | 18,948 | 44,649 (2.4) | 0 | 92 % | 1.60 / 2.00 / 3.00 | 48,264 |
| 14 | 6,724 | 13,975 (2.1) | 0 | 93 % | 2.00 / 2.00 / 2.67 | 14,918 |
| 15–16 | 623 | 1,228 (2.0) | 0 | 93 % | 2.00 / 2.00 / 2.00 | 1,239 |

Depth 1 has 4 cells per seed: the 2 halves of the large component and 2 from the 586-vertex one, which is why "per cell" is low there. R coarsens more levels than Reuse reuses because each of R's five best-of-5 runs coarsens its own bottom levels.

**Why the weights are wrong (derived from the table and the diagnostics).**
- The depth-0 matching's cap is fixed against the whole component.
  - In the top stage it is about 45. Any cell's own top-stage cap is also about 45, so this stage is harmless for cells of 3,600 or more vertices.
  - In the bottom stage it is 1.5·n/120 (24.5k on CA). A depth-d cell's own bottom cap is 2^d times smaller.
- Cells under 3,600 vertices have their own cap below 45, so for them even the top stage is over-weight. That accounts for the 1.6–2× ratios at depths 9–16.
- The earlier report found the cap load-bearing: mutual-max without it fails on CA and TX. Reuse brings that failure back at every depth.

### Where reuse needed fresh coarsening

Nowhere. Every one of the 15 Reuse runs and 15 `ge` runs coarsened 0 levels after depth 0.
- Of 136,703 (PA), 172,249 (TX) and 243,653 (CA) reused levels, 0 merged nothing inside their cell.
- 12–27 levels per graph shrank a cell by under 5 %.
- The depth-0 hierarchy always reached ≤ 120 vertices for the whole component. A cell has at most as many coarse vertices as its parent at every level, so no cell can run out of levels before its own floor.

Matching would have to be recomputed after depth 0 only if a component's depth-0 coarsening stalled above 120 under the < 5 % rule. That never happened on these graphs.

## Straddlers — `out/strad.txt`, `straddlers.json`

Sums over 5 seeds, needed levels only:
- **Share:** side straddlers divided by parent coarse vertices.
- **Entry share:** the parent entries in straddler rows, which the masked split cannot keep.
- **B2 / B1 input:** the entries B2 re-contracts (rows of straddler members) divided by the entries full re-contraction (B1) reads.

CA. PA and TX follow the same pattern, with shares within a factor of about 2.3 of CA's at every depth (`out/tables.txt`):

| Split depth | Share at level 1 | Share, needed levels | Max (level) | Entry share | B2 / B1 input | Child-straddler share |
|---|---|---|---|---|---|---|
| 0 | 0.01 % | 0.03 % | 5.7 % (l23) | 0.06 % | 0.05 % | 0.03 % |
| 1 | 0.02 % | 0.08 % | 9.5 % (l22) | 0.15 % | 0.12 % | 0.08 % |
| 2 | 0.03 % | 0.13 % | 10.2 % (l19) | 0.23 % | 0.19 % | 0.13 % |
| 3 | 0.05 % | 0.17 % | 10.5 % (l19) | 0.31 % | 0.25 % | 0.17 % |
| 4 | 0.07 % | 0.25 % | 10.7 % (l17) | 0.45 % | 0.35 % | 0.25 % |
| 5 | 0.10 % | 0.35 % | 7.9 % (l16) | 0.62 % | 0.47 % | 0.35 % |
| 6 | 0.16 % | 0.51 % | 6.7 % (l15) | 0.89 % | 0.65 % | 0.51 % |
| 7 | 0.23 % | 0.71 % | 5.2 % (l11) | 1.21 % | 0.85 % | 0.71 % |
| 8 | 0.35 % | 0.98 % | 6.3 % (l12) | 1.65 % | 1.11 % | 0.98 % |
| 9 | 0.53 % | 1.27 % | 6.2 % (l10) | 2.09 % | 1.34 % | 1.27 % |
| 10 | 0.77 % | 1.55 % | 7.1 % (l9) | 2.48 % | 1.47 % | 1.55 % |
| 11 | 1.13 % | 1.88 % | 5.0 % (l7) | 2.89 % | 1.53 % | 1.88 % |
| 12 | 1.47 % | 2.12 % | 4.5 % (l5) | 3.18 % | 1.49 % | 2.12 % |
| 13 | 1.74 % | 2.33 % | 4.4 % (l4) | 3.45 % | 1.53 % | 2.33 % |
| 14 | 1.92 % | 2.50 % | 3.9 % (l3) | 3.64 % | 1.58 % | 2.50 % |
| 15 | 2.52 % | 3.23 % | 4.4 % (l2) | 4.58 % | 2.30 % | 3.23 % |

- The share grows with depth, because cells shrink while the coarse vertices do not, and within a depth it grows with level. The highest shares are at the small, deep levels.
- Child straddlers (split by connected component) are within 0.01 points of side straddlers, so splitting by A/B/separator, as the design's cells do, covers almost all of them.
- **`straddlers.json`** has every (graph, seed, split depth, level) row:
  - parent n and nnz, both for needed levels and for all levels;
  - side straddlers, child straddlers and separator-only vertices;
  - straddler-row entries;
  - the children's n and nnz, and the previous-level nnz B1 reads;
  - B2's re-contraction input and output;
  - the check counts.

  It also has 5-seed sums with shares, and field definitions. It is for the cost stage.

## Diagnostics (1 repetition per seed; orders are deterministic, so only timing is single-shot) — `out/diag.txt`

| Variant | Graph | Arcs median | Triangles median | Arcs, triangles ≤ R | Arcs, triangles ≤ SHEM | Vertices coarsened after depth 0, vs R |
|---|---|---|---|---|---|---|
| capstop | CA | 12,601,507 | 215.7M | 1/5, 3/5 | 4/5, 4/5 | 14.7 % |
| capstop | PA | 7,612,616 | 174.9M | 2/5, 2/5 | 5/5, 5/5 | 14.8 % |
| capstop | TX | 8,390,161 | 109.3M | 4/5, 3/5 | 5/5, 5/5 | 14.7 % |
| ge (brief's literal stop rule) | CA | 12,685,268 | 224.7M | 0/5, 0/5 | 1/5, 2/5 | 0 % |
| ge | PA | 7,693,817 | 184.6M | 0/5, 0/5 | 4/5, 3/5 | 0 % |
| ge | TX | 8,474,855 | 117.8M | 0/5, 0/5 | 2/5, 1/5 | 0 % |

### capstop

- **What it does.** A cell reuses depth-0 levels only while its heaviest coarse vertex stays within the cap R would use at that stage (1.5 × cell weight / CoarsenTo). From there it coarsens fresh, as R does. Best-of-5 cells reuse the top stage only, and each of the five runs coarsens fresh.
- **Quality**, medians against R's medians:
  - TX: −0.10 % arcs, −1.7 % triangles.
  - PA: +0.19 % arcs, +0.34 % triangles. PA's median arcs are 2,306 above R's worst seed.
  - CA: +0.62 % arcs, +3.7 % triangles. 4 of 5 CA seeds have more arcs than R.
  - It is at or below SHEM in 14/15 runs on arcs and 14/15 on triangles.
- **Fresh coarsening**, in vertices coarsened against R:
  - 13–16 % at depths 1–6;
  - 7–14 % at depths 7–10;
  - 15–58 % at depths 11–15, where the cells are small.
- **Time** (1 run): 6.7 s on CA, 3.6 s on PA and 4.4 s on TX.
- **Confound.** For best-of-5 cells, capstop restores two things at once: the cap and the fresh matching in each of the five runs. So it does not separate the two causes for those cells. For cells under 5,000 vertices it changes only the cap.
- One incidental data point, a single run: break 3 below reuses at depths 1–2 only and re-coarsens from depth 3. On PA seed 1 it gives +0.15 % arcs and +2.6 % triangles against R, against +1.46 % and +10.7 % for full reuse. Most of the loss comes from depths 3 and below, where the weight ratio is largest.

### ge

ge uses the brief's literal stop rule, which keeps one level fewer. It fails like Reuse (0/15 against R), so the stop-rule detail is not the cause.

## Deliberate breaks (PA, seed 1) — `out/breaks.txt`, `out/same.txt`

| Check | Break | Result |
|---|---|---|
| B1 = B2 | B2 masks straddlers instead of re-contracting them (their mixed parent weights leak in) | 31,239 of 32,349 levels differ (intact: 0) |
| Accounting identity | separator members counted as A in the side masks | identity fails on 22,093 levels (intact: 0) |
| B1 level validity | +1 on one edge weight in every reused level (asymmetric) | 32,266 of 32,266 levels invalid (intact: 0) |
| No matching after depth 0 | cells at depth ≥ 3 drop the reuse context and re-coarsen | 32,903 coarsening levels after depth 0, 7,273 cells treated as new roots (intact: 0, 0) |
| Depth 0 equals R | one extra random draw before the root bisection | depth-0 hash `ddff4a62…` against `a9d47b06…` |
| Repetitions give the same order | seed + 1 on the second repetition | panic "order differs between repetitions", exit 101 |
| Closure | drop arc (2834, 3822) | 1 violation (intact: 0) |
| Coverage | drop the arc of skeleton edge (846420, 850859) | 2 missing entries, both directions (intact: 0) |
| parent(r) = up(r)[0] | parent(2829) set to up(2829)[1] | 1 mismatch (intact: 0) |
| Stock copy = original feral | the copy run with seed 2 | `false` (intact: `true` on CA, PA, TX) |
| New crate = pre-reuse copy | the new crate run with seed + 1 | `false` (intact: 12/12 `true`) |

## What this means for D5 (derived, not measured)

- **(b) as specified is not viable on quality.** It fails the M1 floor (SHEM) on CA and loses to R everywhere. Whatever B2 costs in GraphBLAS, this order would be worse than feral's.
- **The failure is in the vertex weights, not in the straddlers.** Splitting straddlers is exact (B1 = B2) and touches 0.05–2.3 % of the entries a full re-contraction reads. The masked split itself still reads the whole parent level.
- **A cap-aware reuse (capstop) is the candidate to cost.** It keeps quality near R and at or above SHEM in 14/15 runs. In GraphBLAS it would cost:
  - B2 on the reused levels, which at depths 1–6 are the large ones;
  - fresh matching on the remaining 13–16 % of R's coarsening at those depths. That remainder is on graphs of ≤ n/30 per cell, where per-call overhead may matter more than per-entry cost.

  Neither is measured. Whether the five best-of-5 runs could share one fresh bottom stage, which would cut the 13–16 %, is untested.

## Unmeasured

- The GraphBLAS cost of B2 and of capstop's fresh coarsening. That is the cost stage, which reads `straddlers.json`.
- Which of capstop's two changes carries the recovery: the cap, or fresh matching in the five best-of-5 runs.
- Other fixes, such as splitting over-cap coarse vertices back into their finer parts instead of coarsening fresh.
- **The design's single-stage hierarchy.** This harness keeps feral's two-stage best-of-5: a top stage to n/30, then the bottom stage to 120.
  - The GraphBLAS cost bench used one stage to ≤ 100·2^d, with cap 1.5·(n/2^d)/100.
  - So the level indices and weights in `straddlers.json` follow feral's structure, not the bench's.
- A round cap of 8. These runs used no round cap, like the earlier report's main runs.
- The stack's 1.54M-node California, a 9M-edge graph, query p99, and memory in GraphBLAS.
- The diagnostics' timing: 1 run each.

Scratch code and raw output (nothing committed):
- Harness: `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1match/`. The files are `crates/feral-metis/src/node_nd.rs`, `coarsen.rs`, `lib.rs` and `bench/src/main.rs`; the originals are in `…/scratchpad/m1reuse/orig/`.
- Logs, scripts, the untouched base copy, the frozen binaries and `straddlers.json`: `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1reuse/`. The scripts are `sweep_main.sh`, `post.sh` and `parse_reuse.py`; the logs are in `out/`, with `out/tables.txt` as the parsed tables.
