# M1, D5 option (b): what reusing the depth-0 coarsening costs in GraphBLAS

**Verdict: (b) more than halves the GraphBLAS cost, but at 1 thread it still does not fit in feral's time.**

On all three graphs at 1 thread:
- **B2, the cheaper exact form,** takes **0.92–0.94× feral's whole ordering**. B2 is one masked split per level, with re-contraction of only the straddlers. The time counts its GraphBLAS calls plus its host bookkeeping. With cells whose straddler shares are closer to the quality stage's, B2 takes 0.86–0.89×.
- **B1** (re-contract every level) takes 1.17–1.21×.
- **R** (re-coarsen every depth, as today) takes 2.0–2.1×.
- None of these include initial separators, FM, MMD or the tree. Feral's time includes all of them.

At 4 threads: B2 0.44–0.46×, B1 0.53–0.55×, R 0.80–0.81× of feral.

**capstop, the variant that keeps quality near R, is measured in the addendum at the end**, in a second harness: **0.98–1.10× feral at 1 thread, 0.45–0.58× at 4 threads**, 13–19 % above that harness's B2.

**So (b) is not a way out of D5.**
- As specified, it already fails on quality at any thread count (`m1-reuse-quality.md`: worse than SHEM on all CA seeds).
- At 1 thread, even B2's GraphBLAS part leaves only 6–14 % of feral's time for all the host work.

**The mechanics hold.**
- **B1 and B2 build identical child hierarchies** on all 9,174 level comparisons. A comparison covers dims, live sets, maps, vertex weights, nvals, pattern and values, at 1 and 4 threads.
- **No matching ran after depth 0.** A call counter checks this, and it fires when broken.
- **No cell ever needed fresh matching** to get under its floor (see "Where reuse ran out").

**Where B2's time goes (CA, 1 thread, 6.65 s).**
- 2.42 s: the masked split, one masked `mxm` plus a second pass to copy the values.
- 0.95 s: depth 0, which every path pays.
- 0.91 s: the fine split at each depth, which every path pays.
- 0.66 s: host bookkeeping.
- 0.55 s: merging the straddler rows into the level.
- 0.34 s: everything that grows with the straddler count.

B2 therefore hardly depends on the straddler share. Even if the straddler-scaled calls cost nothing, B2 would take at least 0.89× feral on CA (derived). The merge stays as long as a level has any straddler, and at the quality stage's shares nearly every level does.

## Results — `logs/main_<graph>_<labels>.log`, `logs/tables.txt`

Seconds. Each depth is the median of 3 runs, summed over all depths. Feral is the median of 3 runs in the same process. "fine" cells are refined at the fine level only, as the task describes. "all" cells are refined at every level, which brings the straddler shares closer to the quality stage's (see Straddler shares).

| Graph | Cells | Depths | Threads | R | B1 | B2 | feral | B2 / feral | B1 / feral | R / feral |
|---|---|---|---|---|---|---|---|---|---|---|
| CA | fine | 23 | 1 | 14.88 | 8.57 | **6.65** | 7.08 | **0.94** | 1.21 | 2.10 |
| CA | fine | 23 | 4 | 5.69 | 3.89 | **3.16** | 7.08 | **0.45** | 0.55 | 0.80 |
| PA | fine | 22 | 1 | 7.36 | 4.21 | **3.32** | 3.60 | **0.92** | 1.17 | 2.04 |
| PA | fine | 22 | 4 | 2.93 | 1.97 | **1.64** | 3.60 | **0.46** | 0.55 | 0.81 |
| TX | fine | 19 | 1 | 9.55 | 5.39 | **4.21** | 4.53 | **0.93** | 1.19 | 2.11 |
| TX | fine | 19 | 4 | 3.64 | 2.40 | **1.97** | 4.53 | **0.44** | 0.53 | 0.80 |
| CA | all | 20 | 1 | 14.13 | 7.91 | 6.08 | 6.98 | 0.87 | 1.13 | 2.02 |
| CA | all | 20 | 4 | 5.28 | 3.44 | 2.77 | 6.98 | 0.40 | 0.49 | 0.76 |
| PA | all | 16 | 1 | 7.16 | 3.98 | 3.09 | 3.59 | 0.86 | 1.11 | 1.99 |
| PA | all | 16 | 4 | 2.76 | 1.76 | 1.45 | 3.59 | 0.40 | 0.49 | 0.77 |
| TX | all | 19 | 1 | 9.23 | 5.17 | 4.02 | 4.50 | 0.89 | 1.15 | 2.05 |
| TX | all | 19 | 4 | 3.51 | 2.28 | 1.85 | 4.50 | 0.41 | 0.51 | 0.78 |

- **Against R:** B2 is 0.43–0.45× R at 1 thread and 0.52–0.56× at 4 threads. B1 is 0.56–0.58× and 0.64–0.68×.
- **The two cell modes** differ mostly in depth count and cell sizes; the ratios barely move.
- **Spread across runs is small.** For example, CA fine at 1 thread: R totals 14.87–14.96 s per run, B2 6.65–6.67 s.
- **R here differs from `m1-graphblas-cost.md`'s R** (13.17 s on CA, 15 depths). That run used proxy cells. Real cells take 19–23 depths, and their depths 1–11 each cost about as much as depth 0.

## Per depth on CA (fine cells) — `logs/main_CA_fine.log`

Seconds, median of 3. "split" is the shared fine split, included in every column. Depth 0 is R's own coarsening; B1 and B2 add 0.018 s there to copy the maps to the host.

| Depth | Active cells | Live vertices | R 1 thr | B1 1 thr | B2 1 thr | R 4 thr | B1 4 thr | B2 4 thr | split 1 thr |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1,965,206 | 0.929 | 0.947 | 0.947 | 0.378 | 0.387 | 0.387 | — |
| 1 | 2 | 1,964,710 | 1.025 | 0.536 | 0.351 | 0.396 | 0.228 | 0.154 | 0.047 |
| 2 | 4 | 1,963,319 | 1.015 | 0.529 | 0.349 | 0.396 | 0.233 | 0.157 | 0.049 |
| 3 | 8 | 1,961,552 | 1.020 | 0.533 | 0.365 | 0.398 | 0.234 | 0.170 | 0.052 |
| 4 | 16 | 1,959,591 | 1.023 | 0.538 | 0.367 | 0.397 | 0.233 | 0.170 | 0.052 |
| 5 | 32 | 1,956,714 | 1.025 | 0.538 | 0.369 | 0.399 | 0.242 | 0.175 | 0.052 |
| 6 | 64 | 1,953,262 | 1.032 | 0.541 | 0.376 | 0.393 | 0.233 | 0.177 | 0.053 |
| 7 | 128 | 1,947,965 | 1.019 | 0.537 | 0.387 | 0.400 | 0.236 | 0.181 | 0.053 |
| 8 | 255 | 1,941,295 | 1.022 | 0.539 | 0.392 | 0.388 | 0.236 | 0.182 | 0.054 |
| 9 | 510 | 1,932,145 | 0.997 | 0.530 | 0.392 | 0.382 | 0.244 | 0.195 | 0.052 |
| 10 | 1,002 | 1,919,536 | 0.971 | 0.513 | 0.390 | 0.358 | 0.244 | 0.193 | 0.054 |
| 11 | 1,904 | 1,893,449 | 0.939 | 0.483 | 0.379 | 0.335 | 0.228 | 0.189 | 0.053 |
| 12 | 3,267 | 1,821,795 | 0.830 | 0.431 | 0.350 | 0.296 | 0.202 | 0.174 | 0.053 |
| 13 | 4,332 | 1,597,105 | 0.700 | 0.351 | 0.309 | 0.256 | 0.165 | 0.150 | 0.053 |
| 14 | 3,698 | 1,147,474 | 0.485 | 0.262 | 0.245 | 0.169 | 0.126 | 0.121 | 0.049 |
| 15 | 2,323 | 712,336 | 0.307 | 0.206 | 0.184 | 0.111 | 0.097 | 0.095 | 0.039 |
| 16 | 1,357 | 437,983 | 0.196 | 0.158 | 0.137 | 0.075 | 0.079 | 0.072 | 0.028 |
| 17 | 848 | 283,437 | 0.121 | 0.120 | 0.106 | 0.051 | 0.066 | 0.060 | 0.020 |
| 18 | 789 | 202,392 | 0.114 | 0.126 | 0.118 | 0.045 | 0.069 | 0.059 | 0.044 |
| 19 | 487 | 107,235 | 0.074 | 0.087 | 0.087 | 0.039 | 0.055 | 0.052 | 0.040 |
| 20–22 | 310 | 57,448 | 0.032 | 0.057 | 0.052 | 0.025 | 0.051 | 0.046 | 0.013 |
| **all** | | | **14.88** | **8.57** | **6.65** | **5.69** | **3.89** | **3.16** | **0.91** |

- **Depths 1–11** cost R about as much as depth 0 (0.94–1.03 s each). B2 costs 0.34–0.40× R per depth at 1 thread, and 0.39–0.56× at 4 threads.
- **At the deepest depths** (from 18 on at 1 thread, from 17 on at 4 threads), B1 and B2 cost slightly more than R. R compacts each depth to its live vertices and coarsens only a few small levels. The reuse paths keep stable ids, so their host arrays still span the full id space at every level. Those depths are under 0.13 s each.

## Where the time goes (depths ≥ 1; CA, fine cells) — `logs/tables.txt`

Seconds, summed over depths 1–22. The shared split (0.91 s at 1 thread, 0.28 s at 4) and depth 0 are not in these rows.

| B2 call | 1 thr | 4 thr | | B1 call | 1 thr | 4 thr |
|---|---|---|---|---|---|---|
| masked split `S⟨P_l,struct⟩ = M ANY_PAIR Mᵀ` | 1.498 | 0.517 | | `C'·A` (Cᵀ, desc T0) | 2.250 | 0.933 |
| copy values `eWiseMult(FIRST, P_l, S)` | 0.924 | 0.324 | | `·C'ᵀ` | 1.996 | 0.799 |
| host bookkeeping | 0.660 | 0.688 | | OFFDIAG | 0.876 | 0.267 |
| merge `eWiseAdd(Q_clean, R_strad)` | 0.545 | 0.228 | | host bookkeeping | 0.676 | 0.709 |
| boundary `mxv` | 0.405 | 0.141 | | boundary `mxv` | 0.403 | 0.141 |
| straddler rows: host map and build | 0.239 | 0.242 | | build C'ᵀ | 0.309 | 0.166 |
| project (host) + label vector | 0.224 | 0.162 | | project (host) + label vector | 0.187 | 0.197 |
| build M (clean vertices) | 0.200 | 0.117 | | | | |
| resize for split pieces | 0.071 | 0.041 | | | | |
| extract straddler rows | 0.026 | 0.026 | | | | |

- **B1 saves only the matching.** Its contraction calls (5.12 s) cost the same as R's (4.98 s for `CA`, `CAC'` and `offdiag`), plus 0.68 s of host bookkeeping. R's matching (keys, `mxv`, extract, assign, masker, unkey, 2-hop, cap) and its compaction (0.63 s) are what B1 drops.
- **B2 swaps contraction for the masked split.** The split proper costs 1.50 s; GraphBLAS cannot carry the parent's values through a structural mask, so a second pass (0.92 s) copies them, and the merge adds another (0.55 s).
- **At 4 threads the host bookkeeping is B2's largest single item.** It is serial: 0.69 s on CA, 0.34 s on PA, 0.41 s on TX.

## Setup

- **Code.** Two scratch files in a private copy of the cost bench: `graph/src/graph/graphblas/m1_coarsen_bench.rs`, the parent bench plus 23 lines, and `m1_reuse_bench.rs` (new). Nothing was committed.
  - Location: `scratchpad/m1rcA/wt`, an rsync of `scratchpad/m1gb` at origin/main 7df8e5887 without `.git` or `target`. The final sources are also in `scratchpad/m1rcA/snap/*.final.rs`.
  - The 23 lines add three things: a call counter in `coarsen_level`, `hierarchy_keep`, which hands R's levels to the reuse paths without changing any timed call, and the `mod` line.
  - **Why not in `m1gb`.** Another writer changed `m1gb` during this task:
    - it edited `m1_coarsen_bench.rs` at 01:49;
    - it replaced the runner in my first scratch folder, `scratchpad/m1rc`, at 01:59, which aborted a sweep;
    - it later restored `m1gb` and deleted my target dir there.

    All numbers here come from `m1rcA`. The runs in `m1rc` were discarded.
- **Binary.** `scratchpad/m1rcA/graph-test`, built 02:09:48 with `cargo test --release -p graph --features index-falkordb --lib` and CC/CXX set to Homebrew clang (md5 f05d0dc8…). Every log records its mtime and md5, and `--list` shows `reuse::m1_reuse`.
- **Machine.** Apple M5, 10 cores, 32 GB. Other sessions were running: the 1-minute load was 1.6–5.6 at run starts (`LOAD` lines). Peak RSS was 0.84–1.40 GB per process, under a 12 GB watchdog that never fired.
- **Configuration.** The "best" setup of `m1-graphblas-cost.md`: `Cᵀ` built directly (`M1_CT=1`) under `GxB_JIT_RUN`. Every timed call ends with `GrB_wait(MATERIALIZE)`. Runs used 1 and then 4 GraphBLAS threads, 3 runs each, in one process per graph and cell mode.
- **Graphs.** The SNAP skeletons of `m1-graphblas-cost.md`: CA n = 1,965,206 with 5,533,214 entries, PA 1,088,092 / 3,083,796, TX 1,379,917 / 3,843,320.
- **Depth 0.** R's coarsening of the whole graph: stop at ≤ 100 vertices or < 5 % shrink, cap 1.5·n/100. That gives 20, 24 and 23 levels, ending at 2,806, 292 and 511 vertices. It is identical in every run (fingerprint check), and all three paths use it.
- **Cells for depth ≥ 1: real labels, from a small host ND driver** (untimed, `nd_labels`).
  - **Bisection.** Each active cell is bisected on its coarse graph at the deepest reused level where it still has ≥ 100 coarse vertices (the brief's stop rule). The bisection grows a BFS from a pseudo-peripheral vertex to half the weight. It then takes the lighter boundary that leaves both sides non-empty as the vertex separator.
  - **Refinement.**
    - "fine" projects the separator to the fine level and refines it with a simple boundary FM: thinning from both sides, then zero-gain moves toward the lighter side while the halves differ by over 2 %.
    - "all" applies the same moves at every level on the way down.
  - **Cells and leaves.** The cells are the A and B sides, as in the design. Cells of ≤ 120 vertices are leaves (MMD) and are not coarsened.
  - **Retry.** If a split leaves a side empty, the cell is bisected again one level finer. This happened for 1 cell on CA (fine).
  - **Fix found along the way.** A first version picked a boundary that swallowed a whole side. Cells made of a few heavy disconnected chunks then never split, and depths ran to the cap of 40. Every cell now splits.
- **Shared fine split, in every path.** The design's step 6 on the parent depth's level-0 matrix: `A_{d+1}⟨A_d, struct⟩ = M ANY_PAIR Mᵀ`. Its pattern is checked against a host count.
- **R** (as in `m1-graphblas-cost.md`).
  - It first compacts the live vertices with one `GrB_extract` (R only; 0.63 s on CA at 1 thread), because separators and leaves would otherwise ride along as unmatched singletons.
  - It then runs the parent bench's `hierarchy` with cap 1.5·(live/cells)/100 and stop at ≤ 100·cells or < 5 % shrink. This is a global stop, as before.
- **B1 and B2 share one host bookkeeping pass per level**, O(vertices of the previous level), single-threaded. The pass covers:
  - the group of every parent coarse vertex: one child cell, the separator, or mixed;
  - the child's pieces. A clean vertex keeps its parent id. A straddler's first piece keeps it too, and further pieces are appended.
  - the per-cell stop: a cell keeps level l while it still has ≥ 100 coarse vertices there;
  - the vertex weights. B1 sums every member. B2 copies the parent's weight for clean vertices and sums only the straddler pieces.
- **B1** builds `C'ᵀ` from host arrays (iso), then computes `C'·A_{l-1}` (desc T0), `·C'ᵀ` and OFFDIAG. These are the same calls as R's contraction.
- **B2** works per level:
  - `M_l` over clean vertices of active cells, then `S⟨P_l, struct⟩ = M ANY_PAIR Mᵀ` on the parent's level matrix;
  - `S = eWiseMult(FIRST, P_l, S)` for the values;
  - a resize if new pieces were added;
  - the straddler rows `Q_{l-1}(X, :)` by `GrB_extract`, mapped on the host, and built into `R` with PLUS in both directions;
  - `Q_l = eWiseAdd(S, R)`.
- **Uncoarsening work, in all three paths, as the parent bench times it.** Per level there is one projection of a random 2-way label and one boundary `mxv(A_l, labels)`. R projects with `mxv(Cᵀ, ·)`; B1 and B2 project through their host maps.
- **Not timed:** the label pass, the checks, and B1 vs B2 comparisons.

## Straddler shares — `SHARE` lines; quality stage: `scratchpad/m1reuse/straddlers.json`

A straddler is a parent coarse vertex with fine members in ≥ 2 of {A, B, separator}. The share is computed over the levels some child with > 200 vertices uses ("needed"), summed per split depth, the same definition as the quality stage. Shown for CA; PA and TX follow the same pattern (`logs/tables.txt`).

| Split depth | fine: level 1 | all: level 1 | quality: level 1 | fine: needed | all: needed | quality: needed | fine / quality | all / quality |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.039 % | 0.027 % | 0.008 % | 0.111 % | 0.081 % | 0.035 % | 3.2 | 2.3 |
| 1 | 0.108 % | 0.057 % | 0.020 % | 0.296 % | 0.179 % | 0.081 % | 3.7 | 2.2 |
| 2 | 0.139 % | 0.107 % | 0.033 % | 0.383 % | 0.318 % | 0.127 % | 3.0 | 2.5 |
| 4 | 0.223 % | 0.167 % | 0.068 % | 0.625 % | 0.494 % | 0.249 % | 2.5 | 2.0 |
| 6 | 0.409 % | 0.320 % | 0.159 % | 1.041 % | 0.874 % | 0.512 % | 2.0 | 1.7 |
| 8 | 0.708 % | 0.559 % | 0.353 % | 1.650 % | 1.380 % | 0.983 % | 1.7 | 1.4 |
| 10 | 1.214 % | 1.057 % | 0.774 % | 2.266 % | 2.013 % | 1.547 % | 1.5 | 1.3 |
| 12 | 1.667 % | 1.842 % | 1.466 % | 2.450 % | 2.513 % | 2.115 % | 1.2 | 1.2 |
| 14 | 1.791 % | 1.997 % | 1.916 % | 2.662 % | 2.578 % | 2.497 % | 1.1 | 1.0 |
| 15 | 1.799 % | 2.114 % | 2.522 % | 2.778 % | 2.731 % | 3.228 % | 0.9 | 0.8 |

- **"fine"** cells give 2–3.7× the quality stage's share at depths 0–6. The quality stage's FM refines at every level, so its fine separator mostly follows coarse-vertex boundaries. PA is 2.1–3.4× and TX 1.7–6.6× (TX depth 0).
- **"all"** brings CA and PA to at most 2.7× (0.4–2.7× across depths). TX depth 0 stays at 6.9×.
- **The share hardly matters for cost.** The calls that scale with straddlers sum to 0.34 s of B2's 6.65 s on CA at 1 thread (0.27 s in "all"), and to 0.18 s on PA and 0.21 s on TX. At the quality stage's share they would be about a third to a half of that (derived). Even if those calls cost nothing, B2 would still cost ≥ 6.31 s on CA, 0.89× feral (derived). The merge (0.55 s) stays, because at these shares nearly every level still has a straddler. So B2 at the quality stage's share sits between this floor and the measured runs.

## Where reuse ran out — `RANOUT` lines

- **Matching never had to be recomputed after depth 0.** In every run the matching counter did not move during B1 and B2.
- **Some cells reached the depth-0 hierarchy's last level while still above 100 coarse vertices.**
  - On CA that was 1–11 cells per depth at depths 1–11, with up to 2,729 coarse vertices. But at most 92 of those vertices have an edge. The rest are whole small components that no matching can merge. Depth 0's global < 5 % stop ended at 2,806 vertices because of them: 1,708 of those vertices have no edge (`m1-graphblas-cost.md`).
  - The only cells with more than 100 coarse vertices that have edges are 1–2 per graph, at 110–129 vertices, all in the "fine" runs: CA depths 13 and 15, TX depth 11. They are within 30 % of the floor, and the brief's stop rule ("the deepest level with ≥ CoarsenTo") takes them as they are. The "all" runs have none.
- **Many deep cells reuse no level at all,** because they have under 100 coarse vertices already at level 1. That is up to 1,251 per depth on CA (fine) and 2,357 (all). These cells have at most about 200 vertices and need no coarsening.

## Checks and deliberate breaks — `logs/breaks/` (PA, 1 thread, 2 runs, depths 1–4, final binary)

The control run passes. Each break fails its run with exit 101 and the message shown.

| Check | Break | Result |
|---|---|---|
| B1 = B2: dims, live sets, maps, weights, nvals, pattern, values, every level ≥ 1, every run and thread count | `b2mask`: B2 treats a straddler's first piece as clean, so the parent's mixed row leaks in | "B1 and B2 child matrices differ (depth 1 level 1: nvals 2016452 vs 2017216 …)" |
| B1/B2 vertex weights | `bweight`: +1 on one straddler piece in B2 | "B1/B2 vertex weights differ on 1 vertices (depth 1 level 1)" |
| Weight conservation per level (Σ w = vertices of the cells still active) | `bcons`: +1 on one vertex in both B1 and B2 | "vertex weight not conserved (depth 1 level 1)" |
| No diagonal | `b1offdiag`: skip OFFDIAG in B1 | "B1 level has a diagonal (depth 1 level 1)" |
| Symmetry (`M1_VERIFY`) | `basym`: one asymmetric entry, the same in B1 and B2 | "B1 level not symmetric (depth 1 level 1)" |
| B hierarchy identical across runs and thread counts (hash of all tuples) | `bperm`: reverse the order straddler pieces get ids in run 2 | "B hierarchy differs across runs/threads at depth 1" |
| No matching after depth 0 (counter on `coarsen_level`) | `bmatch`: one matching call inside B2 | "matching ran after depth 0 (depth 1)" |
| Per-cell reused levels equal the label pass's | `lstar`: label pass uses a floor of 101 | "per-cell top level differs from the label pass (depth 4 cell 1)" |
| Separator valid: no A–B edge inside a cell | `sepcut`: move one two-sided separator vertex into A | "separator leaves an A-B edge (depth 1, cell 0, 125238-125241)" |
| Fine split pattern equals a host count | `split`: labels shifted by one | "split pattern (depth 1)" |
| Depth-0 hierarchy identical across runs and threads | `salt`: per-call salt | "depth-0 hierarchy differs across runs/threads" |
| R: no diagonal | `nooffdiag` | "A_c has a diagonal" |
| R: mate is an involution (`M1_VERIFY`) | `nomutual` | "mate is not an involution at 0" |

In the timed runs all of these checks were on, except symmetry and the involution check (`M1_VERIFY` only). R's per-depth fingerprints matched across all 6 runs of each process.

## Derived, not measured

- **capstop, the quality-preserving variant.** The quality stage found that capstop coarsens 14.7 % of the vertices R coarsens after depth 0. If its fresh part cost 14.7 % of R's coarsening calls, then on CA:
  - **1 thread:** capstop ≈ B2 + 0.147 × 11.99 s ≈ **8.4 s, 1.19× feral**.
  - **4 threads:** ≈ 3.16 + 0.147 × 4.71 ≈ **3.9 s, 0.54×**.

  Caveats: the 14.7 % is in vertices, on the quality harness's two-stage hierarchy. The fresh levels are small, so per-call overhead may dominate there.

  Now measured (addendum): 7.35 s (1.00×) and 3.30 s (0.45×) on CA. The fresh share there is 7.6 %, not 14.7 %.
- **A one-pass restriction.** B2 would drop to ≈ 5.18 s on CA at 1 thread (0.73× feral) if the masked split kept values and wrote the merged level in one pass, removing the value copy and the merge. GraphBLAS has no such call for a label test: `select` cannot read a vector. It would need a user operator reading the labels through its thunk.
- **4 threads.** B2 leaves 0.54–0.56× of feral's time for the host work. That is about 3.9 s on CA, against about 2.2 s for R in `m1-graphblas-cost.md`.

## Unmeasured

- **capstop itself:** its GraphBLAS cost, and its fresh matching. *Measured since in the addendum, without best-of-5.*
- **The host work of a full GraphBLAS ND:** initial separators, FM per level, MMD, tree, and peeling.
- **The quality stage's two-stage best-of-5 hierarchy.** This bench uses one stage and one global depth-0 stop, so the level indices differ.
- **A round cap of 8.**
- **Peak memory of the GraphBLAS part.** The process peak of 1.40 GB includes the text graph, the host label pass and feral.
- **Other platforms, sizes and metrics:** Linux, 16 threads, the stack's 1.54M-node California, a 9M-edge graph, and query p99.
- **Proxy cells.** These were not run. The every-level cells are the run closest to the quality stage's shares.

## Files (scratch, nothing committed or pushed)

All under `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1rcA/`:
- Sources: `wt/graph/src/graph/graphblas/{m1_coarsen_bench.rs, m1_reuse_bench.rs}`, also in `snap/*.final.rs`. The parent bench's original is `orig/`.
- Binary: `graph-test`.
- Runner and sweep: `run.sh`, `sweep.sh`.
- Parser: `analyze.py`.
- Logs: `logs/main_{CA,PA,TX}_{fine,all}.log`; parsed into `logs/tables.txt`.
- Breaks: `logs/breaks/*.log`, with `summary.txt`.

The build target dirs were deleted.

## Addendum: capstop, measured (second harness, `scratchpad/m1rcB`)

Added about 03:00 by a second running copy of this agent, with the same agent id. It is the "other writer" named in Setup above, and it ran nothing timed while the sweep above was running. The report above leaves capstop unmeasured. This section measures capstop beside its own R, B1 and B2, in one process per graph. The cells, depth count and code differ from the harness above, so compare within this section.

**Verdict: capstop roughly ties feral's whole ordering at 1 thread and fits at 4 threads.**
- **Against feral:** 0.98–1.10× at 1 thread and 0.45–0.58× at 4 threads. As above, the GraphBLAS part alone, with no host separators, FM or MMD.
- **Against B2:** 13–19 % more. At 1 thread that leaves no room for the host work, so capstop alone does not resolve D5 either.
- **Fresh coarsening:** 5.2–10.0 % of the vertices R coarsens after depth 0, against the plain-Rust 14.7 %. The estimate above (8.4 s, 1.19× on CA) assumed 14.7 %.

| Graph | Threads | R | B1 | B2 | capstop | feral | capstop / feral | capstop / B2 | B2 / feral |
|---|---|---|---|---|---|---|---|---|---|
| CA | 1 | 14.24 | 8.10 | 6.28 | **7.35** | 7.32 | **1.00** | 1.17 | 0.86 |
| CA | 4 | 5.22 | 3.41 | 2.79 | **3.30** | 7.32 | **0.45** | 1.18 | 0.38 |
| PA | 1 | 7.42 | 4.24 | 3.29 | **3.91** | 3.56 | **1.10** | 1.19 | 0.92 |
| PA | 4 | 3.33 | 2.10 | 1.72 | **2.05** | 3.56 | **0.58** | 1.19 | 0.48 |
| TX | 1 | 9.26 | 5.33 | 4.15 | **4.69** | 4.79 | **0.98** | 1.13 | 0.87 |
| TX | 4 | 4.03 | 2.61 | 2.11 | **2.39** | 4.79 | **0.50** | 1.13 | 0.44 |

- **Units.** Seconds: the sum over depths 0–14 of each depth's median over 3 runs. Every depth includes the shared fine split, and depth 0 (R's coarsening) is paid by all four paths. Feral is the median of 3 runs in the same process. Logs: `m1rcB/logs/main_{CA,PA,TX}.log`.
- **R, B1 and B2 agree with the main table above.** Against feral here: R 1.93–2.09×, B1 1.11–1.19×, B2 0.86–0.92× at 1 thread.
- **Spread.** Per-run totals are within 2 %, except PA at 4 threads, where R ranges 2.92–3.63 s and capstop 1.84–2.29 s. The 1-minute load at run starts was 2.8–5.6, from other sessions.

### capstop's fresh share (vertices coarsened after depth 0, against R)

| Graph | This bench | Plain Rust (`m1-reuse-quality.md`) |
|---|---|---|
| CA | 7.6 % | 14.7 % |
| PA | 10.0 % | 14.8 % |
| TX | 5.2 % | 14.7 % |

By depth on CA, against the quality stage's ranges:

| Depths | This bench | Quality stage |
|---|---|---|
| 1–6 | 0.2–3.9 % | 13–16 % |
| 7–10 | 5.9–12.4 % | 7–14 % |
| 11–13 | 14–38 % | 15–58 % |

**The gap is at the shallow depths.** A likely cause (derived, not tested): in cells of 5,000 or more vertices, the quality harness's capstop coarsens the bottom stage fresh in each of the five best-of-5 runs. This bench builds one hierarchy per depth, as R here and the earlier cost bench do. With best-of-5, both capstop's fresh part and R would grow; that is unmeasured.

### Per depth on CA (seconds, median of 3, including the shared fine split)

| Depth | R 1 thr | B1 1 thr | B2 1 thr | capstop 1 thr | R 4 thr | B1 4 thr | B2 4 thr | capstop 4 thr | capstop levels reused / fresh | capstop fresh share |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.959 | 0.974 | 0.974 | 0.974 | 0.369 | 0.376 | 0.376 | 0.376 | — | — |
| 1 | 1.019 | 0.552 | 0.384 | 0.380 | 0.386 | 0.233 | 0.169 | 0.171 | 17 / 3 | 0.2 % |
| 2 | 1.041 | 0.543 | 0.378 | 0.387 | 0.395 | 0.238 | 0.169 | 0.178 | 16 / 5 | 0.4 % |
| 3 | 1.035 | 0.546 | 0.386 | 0.407 | 0.394 | 0.237 | 0.169 | 0.180 | 13 / 6 | 0.7 % |
| 4 | 1.048 | 0.544 | 0.399 | 0.399 | 0.396 | 0.235 | 0.183 | 0.196 | 12 / 7 | 1.2 % |
| 5 | 1.036 | 0.544 | 0.403 | 0.431 | 0.404 | 0.244 | 0.188 | 0.209 | 11 / 8 | 2.2 % |
| 6 | 1.035 | 0.556 | 0.404 | 0.442 | 0.399 | 0.237 | 0.193 | 0.224 | 9 / 8 | 3.9 % |
| 7 | 1.062 | 0.570 | 0.404 | 0.471 | 0.399 | 0.238 | 0.191 | 0.233 | 8 / 8 | 5.9 % |
| 8 | 1.069 | 0.574 | 0.419 | 0.470 | 0.387 | 0.239 | 0.200 | 0.229 | 7 / 6 | 6.0 % |
| 9 | 0.997 | 0.556 | 0.422 | 0.476 | 0.374 | 0.240 | 0.207 | 0.228 | 6 / 4 | 6.6 % |
| 10 | 0.976 | 0.529 | 0.420 | 0.554 | 0.351 | 0.233 | 0.191 | 0.252 | 6 / 4 | 12.4 % |
| 11 | 1.000 | 0.537 | 0.417 | 0.591 | 0.326 | 0.220 | 0.180 | 0.247 | 6 / 2 | 14.4 % |
| 12 | 0.873 | 0.478 | 0.355 | 0.624 | 0.293 | 0.187 | 0.153 | 0.265 | 3 / 2 | 26.8 % |
| 13 | 0.680 | 0.359 | 0.305 | 0.537 | 0.213 | 0.154 | 0.137 | 0.217 | 1 / 1 | 38.2 % |
| 14 | 0.410 | 0.236 | 0.209 | 0.211 | 0.133 | 0.100 | 0.088 | 0.092 | 1 / 0 | 0 % |
| **all** | **14.24** | **8.10** | **6.28** | **7.35** | **5.22** | **3.41** | **2.79** | **3.30** | | **7.6 %** |

- **Columns.** "Reused / fresh" counts the levels across all cells of the depth; each cell stops reusing at its own level. A cell's fresh levels are the levels of the one graph that gathers every stopped cell.
- **Where capstop costs more than B2.** From depth 10 on, cells are small and R's cap is small, so most of a cell's coarsening is fresh. At depths 12–13 capstop costs 0.6–0.8× R.

**Where capstop's extra time goes (CA, depths 1–14).**
- At 1 thread capstop costs 1.10 s more than B2:
  - +1.06 s: fresh matching and contraction on the gathered cells. The largest single call is the cap mask, at 0.24 s; the fresh vertices sit near the cap, so the mask is built at every fresh level. `C·A`, `·Cᵀ` and OFFDIAG take 0.36 s.
  - +0.17 s: gathering the stopped cells, one `GrB_extract` per stop level plus one build.
  - −0.13 s: B2 calls saved on the levels that are no longer reused.
- At 4 threads the difference is 0.49 s.

### Straddler shares (`STRAD` lines)

The labels here come from this harness's label pass, described under Setup. Shares are taken over the reused levels; the quality stage's are over its needed levels, 5 seeds.

| Split depth (CA) | Level 1, here | Level 1, quality | Reused levels, here | Needed levels, quality | Ratio |
|---|---|---|---|---|---|
| 0 | 0.027 % | 0.008 % | 0.099 % | 0.035 % | 2.8 |
| 2 | 0.038 % | 0.033 % | 0.134 % | 0.127 % | 1.1 |
| 4 | 0.128 % | 0.068 % | 0.440 % | 0.249 % | 1.8 |
| 6 | 0.285 % | 0.159 % | 0.881 % | 0.512 % | 1.7 |
| 8 | 0.570 % | 0.353 % | 1.617 % | 0.982 % | 1.6 |
| 10 | 1.175 % | 0.774 % | 2.449 % | 1.547 % | 1.6 |
| 12 | 2.492 % | 1.466 % | 3.349 % | 2.115 % | 1.6 |
| 13 | 3.758 % | 1.735 % | 3.758 % | 2.330 % | 1.6 |

- **Closeness to the quality stage.** These cells' shares are 1.1–2.8× the quality stage's on CA, 1.3–2.1× on PA and 1.0–1.8× on TX. That is close enough that the straddler-scaled calls are not misjudged.
- **B2's re-contraction input** is 0.06–2.6 % of what B1 reads.

### Setup (differences from the harness above)

- **Code.** `scratchpad/m1rcB/wt/graph/src/graph/graphblas/{m1_coarsen_bench.rs, m1_reuse_bench.rs}`, an rsync of `m1gb` without `.git` or `target`.
  - Binary: `m1rcB/graph-test`, built 02:37:58 with `cargo test --release -p graph --features index-falkordb --lib`, Homebrew clang, md5 b5e46b20…. Every log records its mtime and md5.
  - Settings: `M1_CT=1`, under `GxB_JIT_RUN`.
  - Parser: `m1rcB/analyze.py`.
  - Peak RSS: 1.96 GB on CA, under the 12 GB watchdog.
- **Cells.** A host label pass, untimed.
  - Each cell is bisected at the deepest reused level where it still has at least 100 coarse vertices. Coarse vertices are ordered by the feral position of their first member, and each side gets half the weight.
  - The labels are projected down with a greedy boundary refinement at every level: edge cut, 3 % balance.
  - The smaller side's boundary at level 0 becomes the vertex separator.
  - There are 15 depths (0–14), with 2^d cells at depth d.
- **R.** The parent bench's `hierarchy`, with cap 1.5·(n/2^d)/100 and stop at ≤ 100·2^d or < 5 % shrink.
  - Removed separator vertices are masked out instead of compacted: they get no coarse vertex.
  - **R is unchanged otherwise.** With proxy cells it reproduces the earlier bench's per-level n, nnz, rounds and 2-hop flag on all 177 PA levels (`m1rcB/logs/unchanged_pa_chunks.log` against `m1gbtools/best_PA.log`).
- **B1 and B2** follow the same scheme as above. One host numbering pass per level sorts the parent's vertices into clean, straddler and separator-dropped. A clean vertex keeps its id, and straddler parts are appended.
  - B1 re-contracts with `C'·A·C'ᵀ` and OFFDIAG.
  - B2 takes the masked split `M ANY_PAIR Mᵀ` under the parent level's structure, then `eWiseMult(FIRST)` for the values. Straddler rows are re-contracted from the child's previous level, filled in by transpose, and merged with `eWiseAdd`.
- **capstop.**
  - **Reuse:** B2 per level, cell by cell. A cell stops reusing at the first level where one of its coarse vertices would exceed R's cap for that depth, or where its parent cell had already stopped.
  - **Fresh:** the stopped cells are gathered into one graph and coarsened together with R's matching, cap and stop rule (≤ 100 per cell).
  - **Projection** runs through the fresh levels, back into each cell's last reused level, and down the reused levels.
  - **Difference from the quality harness:** here a child can reuse only levels its parent reused, because B2 needs the parent's level matrix. The quality harness composed levels from the depth-0 maps. Cells limited this way went fresh: 32 on CA (depths 10–12), 2 on PA and 2 on TX.
- **Not timed:** the label pass and the checks.

### Checks and deliberate breaks (`m1rcB/logs/breaks/summary.txt`)

**Checks in the timed runs.** Full checks ran on the first run of each thread count; fingerprints ran on every run.

| Check | Result (CA / PA / TX) |
|---|---|
| B1 = B2 on every level: n, nvals, weights, pattern and values | 330 / 294 / 310 levels, 0 differences |
| B1 valid: no diagonal, symmetric, Σw = live vertices | 0 failures |
| Host weights equal `C'ᵀ·w` | 0 failures |
| Accounting identity: live coarse vertices = distinct (depth-0 coarse vertex, cell) pairs | 165 / 147 / 155 levels, 0 failures |
| No matching in B1 or B2 after depth 0 (call counter) | counter stayed at 0 |
| capstop: every reused level within the cap, per cell | 232 / 188 / 204 levels, 0 failures |
| capstop: weight conserved | 28 / 28 / 28, 0 failures |
| Separators separate | 14 / 14 / 14, 0 failures |
| All four paths' hierarchies identical across runs and thread counts | 285 comparisons per graph, 0 differences |

**Breaks.** PA, depths 1–4, 2 runs, 1 thread. The control run passes. Each break fails its run with exit 101.

| Check | Break | Result |
|---|---|---|
| B1 = B2 | `nofill`: B2 skips the transpose fill of straddler columns | 81 / 81 levels differ |
| B1 valid | `b1diag`: B1 skips OFFDIAG | 81 invalid |
| Separator handling in the numbering | `sepclean`: separator members are ignored in the clean test | 81 B1/B2 differences, 81 invalid, 29 weight failures |
| Accounting identity | `mergeparts`: all parts of a straddler share one id | 76 of 80 levels fail |
| No matching after depth 0 | `rematch`: B2 re-coarsens at depth ≥ 3 | counter 66 |
| capstop cap | `cap2`: capstop allows twice the cap | 16 levels over the cap |
| capstop weight conserved | `gfdrop`: the first stopped block is left out of the fresh graph | 4 / 4 fail |
| Separators separate | `nosep`: no separator for cell 0 | 4 / 4 fail |
| Fine split pattern | `split`: labels shifted by one | 8 failures |
| Identical across runs and threads | `salt`: per-call salt | 17 / 17 fingerprints differ |
| R: no diagonal | `nooffdiag` | "A_c has a diagonal" |
| R unchanged against the earlier bench | `salt`, proxy cells | 169 of 177 levels differ |

### Unmeasured (capstop)

- capstop with best-of-5 fresh bottom stages, and R with best-of-5.
- The quality of this bench's capstop orders. The quality stage measured capstop's quality in its own harness only.
- The host work of a full GraphBLAS ND.
- Linux, 16 threads, the stack's California, and a 9M-edge graph.
