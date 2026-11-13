# U3: how CCH ordering works today (C and feral-metis), mapped to GraphBLAS/LAGraph

Revisions read: C `origin/master` a61fbb67 (`civil-kitten`); METIS submodule 272d4a91 (same pin as C master, read from the `constraint-fix` checkout); Rust `origin/main` 7df8e588 and `origin/cch/05-inputs`; feral `FalkorDB/feral@f1271891` (the rev in `Cargo.lock:400-408` on main); LAGraph d01064de; GraphBLAS v10.5.0. Paths below are short forms: `cch.c` = `src/algorithms/cch.c`, `cch_index.c` = `src/index/cch_index.c`, feral files are under `crates/feral-metis/src/`.

## 1. C master: the pipeline, step by step

1. **Weight matrix.** `CCHIndex_Build` builds the FP64 weight matrix W for the indexed relationship types (`cch_index.c:397-398`, `330-372`). It is sized `Graph_RequiredMatrixDim`, so unused and deleted ids appear as isolated vertices (`cch_index.c:400`).
2. **Skeleton.** `_build_undirected_skeleton` computes `Sym = ONEB(A, Aᵀ)` with `GrB_eWiseAdd` and drops the diagonal with `GrB_select OFFDIAG` (`cch.c:175-185`). It then copies Sym into METIS's `xadj/adjncy` with a `GxB_rowIterator` (`cch.c:190-231`). Weights and direction are dropped (`cch.c:157-161`).
3. **Order.** `METIS_NodeND` with `METIS_SetDefaultOptions` and nothing overridden (`cch.c:273-283`). `perm[rank]=node`, `iperm[node]=rank` (`cch.c:270-271`). The METIS defaults that apply to NodeND are:
   - Coarsening is SHEM (`options.c:82`).
   - The initial partition is an edge bisection (`IPTYPE_EDGE`). Refinement is one-sided node FM (`SEP1SIDED`) with `niter=10` (`options.c:65-68`).
   - `nseps=1`, `ufactor=200`, which allows sides up to 1.2 times balanced (`options.c:67,69`, `macros.h:35`).
   - Vertices with identical neighbour lists are merged first (`compress=1`). Components are not ordered separately (`ccorder=0`). No pruning (`pfactor=0`) (`options.c:70-72`).
   - The random seed is fixed at 4321 (`options.c:85`, `util.c:21-24`), so the result is deterministic.

   Inside METIS (`ometis.c`):
   - Each recursion level runs `MlevelNodeBisectionMultiple` (`ometis.c:192`). A graph of 5,000 vertices or more coarsens once to `max(100, n/30)` vertices and keeps the best of 5 bisections (`ometis.c:352-387`).
   - At the core (L1), METIS coarsens to `clamp(n/8, 40, 100)` vertices (`ometis.c:399-403`). It then runs `InitSeparator`: GrowBisection with 7 trials (halved when the graph is already small), then `ConstructSeparator` (`ometis.c:307,405-411`, `initpart.c:82-89`).
   - `ConstructSeparator` puts **every boundary vertex** into the separator. It then runs two-sided FM, then one-sided FM (`separator.c:36-56`).
   - Separator vertices take the highest remaining ranks (`ometis.c:199-204`). Each side then recurses, or is ordered by **MMD** (multiple minimum degree) once it has 120 vertices or fewer (`MMDSWITCH`, `defs.h:52`; `ometis.c:213-223`).
4. **Elimination tree.** Union-find with path compression, walking the original neighbours of each rank in increasing rank order (`cch.c:312-375`).
5. **Chordal supergraph.** `up(r)` = the original higher neighbours of r, plus each elimination-tree child's `up` with r removed (Gilbert–Ng–Peyton). It is stored as one array per rank, and each array is sorted (`cch.c:419-497`). `down` is its transpose (`cch.c:503-523`). The skeleton and the parent array are freed after this (`cch.c:545-547`).
6. **Customize.** Every lower triangle is relaxed in one forward sweep (`cch.c:662-696`). So the customization cost is the triangle count of the supergraph.

**What triggers a rebuild in C, and what is incremental.** Only the metric is incremental. The topology is always rebuilt in full:
- Creating or deleting nodes marks every CCH index for a full rebuild (`graph_hub.c:39-41,77-78,261-263`).
- A new edge whose endpoints have no supergraph arc yet sets `pending_rebuild` (`cch_index.c:575-582`, from `graph_hub.c:145-147,172-182`). A pending rebuild also absorbs later additions (`cch_index.c:568-571`).
- A deleted edge keeps its arc and increments `stale`. A full rebuild runs once `stale ≥ max(1024, n/8)` (`cch_index.c:468-477,585-597,493-496`).
- A weight or property change triggers a scoped recustomization when at most n/8 arcs are dirty and n ≤ 2³², and a full customization otherwise (`cch_index.c:498-524`, `cch.c:738-875`).
- All of this flushes at commit, **synchronously under the write lock** (`cmd_query.c:335-338`, `graphcontext.c:1465-1493`). A rollback discards it (`cmd_query.c:314-318`).
- The hierarchy is never serialized. It is rebuilt when an RDB loads (`cch_index.c:161-163`, `decode_graph.c:449`).

## 2. feral-metis as main uses it

Main calls `feral_metis::metis_order` with the default options (`nd.rs:116-118`). Isolated nodes are removed first and take the lowest ranks (`nd.rs:60-67`); California has 294k of them among 1.54M nodes (`nd.rs:35-38`). The default options (`lib.rs:166-181`) are:
- seed 1, `niparts` 7, `coarsen_floor` 120, `nd_to_amd_switch` 200;
- `two_hop_ratio_threshold` 0.85, `max_imbalance` 0.20, `fm_passes` 10;
- `node_refine` true, dense-column quotient off.

**Discrepancy:** the docs claim these mirror the METIS 5.2.0 defaults (`lib.rs:73-99`). The METIS source we ship disagrees on two of them: METIS switches to its leaf ordering at 120 vertices, not 200, and coarsens to between 40 and 100 vertices, not 120 (see §1).

Steps:
1. **Graph.** Copied from the CSC into `i64` arrays `xadj/adjncy/vwgt/adjwgt`, with all weights set to 1 (`graph.rs:26-40,53-99`). This is the second copy of the skeleton; the first is `CchInput`'s CSR (`input.rs:112-118` on 05-inputs).
2. **Components.** A BFS at every recursion level; each component is ordered separately (`node_nd.rs:39,143-148,463-490`). Single-vertex components are placed directly (`node_nd.rs:87-92`).
3. **Coarsening (SHEM).**
   - Vertices are shuffled, then stably sorted by degree. Each unmatched vertex takes its heaviest unmatched neighbour, ties going to the lower id (`coarsen.rs:74-113`).
   - If the graph shrinks by less than 15% (the 0.85 threshold), a linear 2-hop pass pairs vertices that share a neighbour (`coarsen.rs:115-120,174-230`).
   - Contraction sums the weights of parallel edges, drops self-loops and adds up vertex weights (`coarsen.rs:233-294`).
   - Coarsening stops at `coarsen_floor`, or when a level shrinks the graph by less than 5% (`coarsen.rs:139-160`).
4. **Initial bisection** on the coarsest graph:
   - 7 trials alternating greedy graph growing (GGP) and BFS growing, each scored after edge-cut FM (`node_nd.rs:300-320`; `initial_partition.rs:54-193`; `fm_refine.rs:36-192`).
   - The edge cut is turned into a node separator **by König's theorem**: a Kuhn maximum matching on the boundary bipartite graph, then the vertex cover (`separator.rs:34-149`). Kuhn reallocates its visited array for every A-side vertex (`separator.rs:103-105`).
   - That separator is refined with node FM (`node_nd.rs:323-331`).
5. **Uncoarsening.** At each level the labels are projected through `cmap` and refined with node-separator FM (`node_nd.rs:332-347`). Node FM (METIS's `FM_2WayNodeRefine1Sided`, `fm_refine.rs:293-465`) works like this:
   - It uses a lazy binary heap and alternates the target side each pass.
   - It accepts moves that make the separator heavier, and stops a pass after `min(3(|S|+1), 300)` moves without improvement.
   - It rolls back to the lightest separator seen; among equally light ones it keeps the better-balanced (`fm_refine.rs:368,436-457`).
   - Each pass rescans all n vertices and allocates `moved[n]` (`fm_refine.rs:350-362,378`). METIS only touches the boundary.
6. **Best of 5** for subgraphs with n ≥ 5000: coarsen to `max(100, n/30)`, run 5 full bisections, keep the lightest, then refine it back up the levels (`node_nd.rs:156-160,210-275`).
7. **Recursion.** Separator vertices take the top positions in index order (`node_nd.rs:179-185`). Each side is copied out with `build_induced` (`node_nd.rs:188-206,492-521`). A level's graph is dropped before its halves are ordered (FALKORDB.md commit 3).
8. **Leaves.** `feral_amd::amd_order` orders subgraphs of at most 200 vertices, and also any split that leaves one side empty (`node_nd.rs:151-154,174-177,382-403`). AMD runs with aggressive absorption and `dense_alpha` 10 (`feral-amd/src/lib.rs:34-58`).
9. **Determinism and memory.**
   - The random generator is a call-owned SplitMix64 with seed 1, so the order is the same on every run and platform (`rng.rs:14-24`, `nd.rs:14-17`).
   - Every index is `i64` (`feral-ordering-core/src/lib.rs:54`).
   - The renumbered copy costs `8(m+1) + 8·nnz + 8n` bytes (`nd.rs:88-90`), plus `Graph`'s four arrays at each coarsening level.
   - Peak memory on California was measured as -38% after the "recursion by value" change (FALKORDB.md:20-21). There is no absolute figure, so the peak is **unmeasured** here.
   - There is no vertex-merging (compression) step: `compress` does not appear anywhere in feral-metis `src/`.
10. **Supergraph (Rust).** The elimination-tree and child-absorption passes are fused into one loop; a rank's parent is the first entry of its finished row (`order.rs:381-437`). The result is two flat compressed-row arrays, upward and downward, with 8-byte `Rank` entries (`order.rs:46,129-148,441-467`).

## 3. Each step in GraphBLAS/LAGraph terms

LAGraphX is built and linked (`native-deps/src/recipes/lagraph.rs:58`). The Rust FFI bindings exist for:

| Function | Binding |
|---|---|
| `LAGraph_MaximalMatching` | `lagraphx_bindings.rs:334` |
| `LAGraph_Coarsen_Matching` | `:343` |
| `LAGraph_MaximalIndependentSet` | `:297` |
| `LAGraph_MultiSourceBFS` | `:378` |
| `LAGraph_Hdip_Fiedler` | `:448` |
| `LAGr_PartitionQuality` | `:525` |
| `LAGr_MaximumMatching` | `:550` |

`matrix.rs` already wraps `GxB_rowIterator` and zero-copy `GxB_Container` load/unload (`matrix.rs:75-76,107-113`). A host loop can therefore read a `GrB_Matrix`'s adjacency without copying it into a separate CSR.

| Step | Closest GraphBLAS / LAGraph form | Parallel? | Feasibility / cost |
|---|---|---|---|
| Skeleton | `eWiseAdd(ONEB, A, Aᵀ)` + `select OFFDIAG`. C already does this (`cch.c:181-185`); 05-inputs does it as `pattern_union` with T1 (`input.rs:112-113`) | yes | Already native. The new design should keep this result as a `GrB_Matrix` instead of exporting it to CSR. |
| Isolated nodes, components | Degree with `GrB_reduce`; components with `LAGr_ConnectedComponents` (FastSV, `LAGraph.h:2359`) | yes | Easy. Run once per recursion depth (see Recursion row), not once per subgraph. |
| Heavy-edge matching | (a) `LAGraph_MaximalMatching`: Luby-style on the line graph, which needs the n×e incidence matrix E and its transpose (`Coarsen_Matching.c:406-415`). Its "heavy" mode scores an edge as random/degree × weight (`MaximalMatching.c:178-186`), which is not SHEM. (b) Write our own matching: one masked `mxv` per round where each unmatched vertex picks its heaviest neighbour, and pairs that pick each other are matched. The 2-hop fallback is a masked `mxm` A·A over unmatched vertices. | yes | (a) costs about 4 extra entries per edge for E and Eᵀ, and needs SuiteSparse pack/unpack (`Coarsen_Matching.c:13,183-212`). (b) needs no incidence matrix. Coarsening quality of (a) or (b) against SHEM is **unmeasured**. A·A at a hub is expensive; cap it by degree. |
| Contraction | P(v, cmap[v]) = 1 built with `GxB_Matrix_build_Vector` (`GraphBLAS.h:4110`). Coarse graph = offdiag(Pᵀ·A·P) with PLUS_TIMES over int64; vertex weights = Pᵀ·vwgt. This is exactly `Coarsen_Matching.c:483-506`. Projecting labels down a level is one `mxv` with P. | yes | Clean fit. P replaces `cmap`. Memory per level is about the same as feral's arrays. |
| Initial bisection (about 100 vertices) | BFS growing with `LAGr_BreadthFirstSearch` or `MultiSourceBFS`. A spectral alternative is `LAGraph_Hdip_Fiedler` (`Hdip_Fiedler.c:624-638`). | not worth it | GraphBLAS per-call overhead dominates at about 100 vertices. Use a host loop over a row iterator. Fiedler only makes sense as a non-multilevel alternative; its order quality for CCH is **unmeasured**. |
| Edge-cut → node separator | Boundary = A_mask·1_B. Bipartite graph = `GxB_Matrix_extract_Vector(A, bndA, bndB)` (`GraphBLAS.h:5482`). Maximum matching = `LAGr_MaximumMatching` (Azad–Buluç, `MaximumMatching.c:22-30`). König cover = a level-synchronous alternating BFS (`vxm` from A to B, then follow the match from B back to A). | yes | Feasible. In feral's default path this only runs on the coarsest graph (`node_nd.rs:329`), where a host loop is enough. |
| Node-separator FM | Only the initial gains can be computed in bulk: gain = vwgt − A·(vwgt∘1_other), one `mxv`. The move loop has a priority queue, makes moves that worsen the separator, then rolls back (`fm_refine.rs:380-457`). | **no, sequential** | Keep it as a host loop over the level matrix, through a row iterator or Container. Parallel refiners exist for edge cuts (Jet, Gilbert et al., SISC 2024, [arXiv:2304.13194](https://arxiv.org/abs/2304.13194)). None exists for node separators in GraphBLAS or LAGraph. Seed the queue from the boundary only, not from all n vertices as feral does. |
| Recursion / induced subgraphs | A(I,I) with `GxB_Matrix_extract_Vector`. **Alternative, one level at a time:** keep one matrix and drop the edges between parts with a `select` whose user-defined index operator reads a part-label vector (the same context pattern as C's `cch_index.c:269-328`). Matching, contraction and components never cross a part, so one GraphBLAS call handles every part at a given depth. | yes | Design idea, **unmeasured**. Only FM and the initial bisection are per-part host loops, and those run in parallel across parts. |
| Leaves (≤120–200 vertices) | No GraphBLAS minimum-degree ordering. A parallel AMD picks distance-2 independent sets of pivots; it is 7.3–8.3× faster than SuiteSparse AMD on 64 threads (Chang, Buluç, Demmel, [arXiv:2504.17097](https://arxiv.org/abs/2504.17097)). | across leaves | Leaves are small and independent. Use a host AMD/MMD per leaf, in parallel, or keep bisecting to tiny leaves (pure ND). |
| Elimination tree + chordal supergraph | Rank-space skeleton = `extract(S, rank_to_node, rank_to_node)` + `select TRIU(1)`. Then iterate to a fixed point: U ← U ∪ triu(Pᵀ·U, 1), where P(c, min column of U(c,:)) is one `mxv` with a MIN-on-column-index semiring. | yes, per round | It is correct: every added entry joins two higher neighbours of c, which are a clique in the fill graph, and the fixed point satisfies Liu's parent-containment rule. The number of rounds is at most about the elimination-tree height (**unmeasured** for California). Each round costs O(nnz(U)) unless only changed rows are propagated. The current host loop is near-linear (`order.rs:381-437`); a sensible split is a host loop per ND subtree, run in parallel. |
| Supergraph storage | U as an iso-bool, rank-space, upper-triangular `GrB_Matrix`; `down` = Uᵀ stored by column. | — | With 32-bit index hints (`GraphBLAS.h:1557-1559`), memory is at most `RankRows`' 8 B per entry (`order.rs:142-148`). Phase 2 indexes weights by arc slot = position in the row index array (`order.rs:123-125`), so U must not be rebuilt or reordered behind the weight arrays. |

## 4. Quality knobs that matter for CCH

- **What a better order buys.** Customization cost is the triangle count (`cch.c:662-696`) and the metric's memory is the arc count. Measured on California (FALKORDB.md:29-30):

  | Ordering | Arcs | Triangles |
  |---|---|---|
  | C METIS | 5,527,199 | 48.6M |
  | feral, as now on main | 5,838,003 (+5.6%) | 53.9M (+10.9%) |
  | feral, before the last two changes | 5,932,138 | 59.6M |

  The last two changes were keeping the better-balanced of equally light separators (METIS's `mindiff`) and best-of-5 bisection (FALKORDB.md:24-27). So the multi-trial bisection and separator tie-breaks are measurable levers.
- **Remaining feral vs METIS differences, none measured one at a time:**
  - Leaf ordering: AMD at ≤200 vertices vs MMD at ≤120.
  - Coarsening floor: 120 vs between 40 and 100.
  - Separator construction: König cover vs "all boundary vertices + FM in both directions".
  - No vertex merging, where METIS merges identical-neighbour vertices and runs 2 separator trials when that shrinks the graph by more than 1.5× (`ometis.c:97-113`).
  - Components split at every level vs `ccorder=0`.

  Each is a candidate explanation for the 5.6% / 10.9% gap. A GraphBLAS reimplementation should be judged by the same California arcs/triangles numbers against C METIS.
- **Balance.** METIS allows sides up to 1.2× balanced (`ufactor=200`); feral uses `max_imbalance` 0.20, the same bound. Tighter balance should give a shallower elimination tree at the price of bigger separators; that trade-off and its effect on CCH query time are **unmeasured** here.
- **AMD leaves vs pure ND.** CCH papers order road graphs by recursive bisection with flow-based separators. FlowCutter gives "high-quality results and thus fast query times"; Inertial Flow is fast but its queries are slower (Gottesbüren, Hamann, Uhl, Wagner, [arXiv:1906.11811](https://arxiv.org/abs/1906.11811); Dibbelt, Strasser, Wagner, [arXiv:1402.0402](https://arxiv.org/abs/1402.0402)). Whether minimum-degree leaves beat pure ND for our query (elimination-tree height) is **unmeasured**. Measure arcs, triangles and tree height at leaf thresholds 0, 120 and 200.
- **Determinism.** METIS and feral are both deterministic for a given input. A GraphBLAS version should use integer, MIN and MAX reductions only, with a seeded generator; whether results stay identical across thread counts is **unverified**.

## What this means for the redesign

- The parallel-friendly half maps directly onto GraphBLAS, with LAGraph code to copy: skeleton, components, matching, contraction Pᵀ·A·P, label projection, separator extraction and König cover.
- Node FM and leaf minimum degree are sequential. They run as host loops over `GrB_Matrix` row iterators or Containers. Neither needs a new CSR type.
- Leaves and same-depth parts are independent, so those loops run in parallel across parts.
- C has **no incremental ordering**: every change to the edge pattern ends in a full rebuild under the write lock. Any incremental scheme is new work, with no C reference to match.
