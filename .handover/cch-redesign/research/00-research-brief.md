# CCH redesign: research brief (ordering, supergraph, incremental upkeep)

This brief combines the seven reports in this folder. The tags point to them:

| Tag | Report |
|---|---|
| [U1] | `u1-current-pipeline` |
| [U2] | `u2-native-index-mvcc` |
| [U3] | `u3-c-and-feral` |
| [U4] | `u4-graphblas-primitives` |
| [R1] | `r1-nd-algorithms` |
| [R2] | `r2-incremental` |
| [R3] | `r3-symbolic-la` |

A file:line citation is the report's own unless it is marked "spot-checked".

Revisions read:
- Rust `origin/main` 7df8e5887;
- the top of the CCH stack, `origin/cch/08c-build-catch-up` 31ca1b802 ("tip");
- C `origin/master` a61fbb67a;
- LAGraph d01064de;
- GraphBLAS v10.5.0;
- feral f127189.

Nothing was measured for this brief. "Derived" marks reasoning with no outside source.

It runs about 3,700 words by `wc` (tables included), well past the ~1,500 target, because it merges seven reports and the design phase needs the contradictions resolved in one place.

## 1. The Phase 1 contract

Phases 2 and 3 read only the `Topology` API (`order.rs:214-302`) [U1]:
- `node_count`, `ranks`, `arc_count`;
- `node_of`, `rank_of`;
- `up(r)`, `up_range(r)`, `down(r)`, `arc_slot(a, b)`.

A new Phase 1 must keep these invariants:

- **Ranks.** Node ids and ranks map one-to-one over `0..n`, where n = `node_cap` (`order.rs:104-112`).
  - Isolated nodes and unused id slots take the lowest ranks. Spot-checked at `nd.rs:36-37` and `:61-64` @main.
- **Rows.**
  - `up(r)` is strictly ascending, and every entry ranks above r (I2).
  - `down` is the exact transpose of `up` (I7). Only the scoped recustomization reads it (`partial.rs:173-204` @11).
- **Arc slots.**
  - `up(r)[i]` sits at slot `up_range(r).start + i`.
  - Every metric array is indexed by slot.
  - Slots stay stable for the life of a topology (I3; `customize.rs:36-37, 208-214`).
- **Coverage and closure.**
  - Every skeleton edge is an arc (I4).
  - For each y in `up(x)`, the part of `up(x)` above y is a subset of `up(y)` (I5). `customize.rs:288-293` panics when this fails.
- **Implicit elimination tree.** parent(r) = `up(r)[0]`, and every upper neighbour is an ancestor (I6; `query.rs:76-79`).
- **Processing order.**
  - Increasing rank is a valid order for customization (I8).
  - The metric stores the middle vertex of each arc as a rank (I9). The tie rules depend on rank order (`customize.rs:66-75`).

**What this means for incremental work** [U1 §2, U2 §3]:
- Inserting an arc into today's flat arrays shifts every later slot, so every later chunk of #3774's metric changes.
- Re-ranking a vertex invalidates `node_to_rank`, the stored middles and the ancestor chains.

An incremental design therefore needs arc ids that never move (append-only ids). It may re-rank only inside a region whose metric it then recustomizes.

**The same rule in matrix terms** [U4 §5, R2 §5]. Rows U(v) form a valid topology iff all three hold:
- U(v) is a subset of the ranks above v;
- the skeleton's upper neighbours A⁺(v) are a subset of U(v);
- U(v) without its parent p(v) is a subset of U(p(v)).

A supergraph larger than the minimum is still correct, as long as arcs that are not input edges carry +∞ (survey p.5) [R2 §1].

## 2. What "native-index-like, batched, MVCC-compliant" requires

All from [U2].

- **Jobs.**
  - One job runs per (index, generation): one batch per run on the `falkordb-index-build` pool, resubmitted while the index is `Building` (`online_build.rs:4-9, 322-330`).
  - The pool has 1 worker by default (`src/lib.rs:103`).
  - A batch reads `new_build_version()`, a private copy, with no locks held (`online_build.rs:13-14, 406`).
  - Batches grow from 10k to 4M entities (`:94-147`).
- **Publishing.**
  - A batch publishes under the GIL through `commit_indexer_batch`. The outcome is Same, Swap, CatchUp (Refresh or Insert) or Discarded (`versioned.rs:504-656`).
  - A publish keeps the data version (`mvcc_graph.rs:247-248`).
  - For a column, catch-up is capped at 10,000 changes, about 13 ms (`online_build.rs:126-131`).
- **Writes during a build.**
  - A `Building` index holds exactly the tuples below its cursor.
  - Writers apply additions only below the cursor, and apply every removal (`versioned.rs:29-39, 263-278`).
  - Written ids reach the next batch through `since_last_indexer_step` (`:436-443`).
  - A writer's fork is rebased when a batch overtook it (`src/query_session.rs:251-278`).
- **What CCH lacks today.**
  - `Cursor = ()`: one batch builds everything (`cch/mod.rs:454-458, 595-612` @tip).
  - `index_build_read` returns 0 (`versioned.rs:858-861`).
  - To build in batches, CCH needs:
    - an ordered cursor with a `place(id)` function;
    - a `Batch` type that merges into copy-on-write partial state;
    - a real `batch_len` and `limit_catch_up`;
    - a progress count.
  - With a single batch, the written set grows for the whole build, so every write copies it: O(|written|) per write, unmeasured.
- **Rules for anything the index holds.**
  - Incremental work happens only on a private version (`cow.rs:48-59`).
  - Every GraphBLAS object is fully waited before it is published. A reader that finishes pending work would write to an object other readers share (`mvcc_graph.rs:332-347`).
  - The replaced version is dropped outside the GIL.
- **Copy costs.**
  - A plain `Cow<Matrix>` is deep-copied on its first write. Only `VersionedMatrix` (base plus two deltas, `m/dp/dm`) costs O(delta) (`versioned_matrix.rs:151-171`).
  - Building `effective` for a whole relationship type in every batch costs about 400 MB and 76 ms at 9M edges (`.handover/cch-background-fold/discussion.md`). A batch should read only its own rows, through a masked extract or the three-way merge iterator.
- **GIL budget.**
  - A CCH catch-up or rebase has no limit today: a full customization runs under the GIL, about 0.47 s on California (#3772).
  - On a replica, the same customization runs inside `GRAPH.EFFECT` on the main thread (#3363).
  - Keep only O(written) classification under the GIL, and push the rest into a batch.
- **Where the new code plugs in.**
  - Five `graph.rs` sites record writes into `CchChanges` (`maint.rs:97-112, 216-325` @tip).
  - A structural change becomes a rebuild in exactly two places: `update` returning `(None, Rebuild)` and `absorb` returning `None` (`maint.rs:551-582, 621-623`). Incremental maintenance replaces those two branches.
- **No host calls.** The `graph` crate stays host-agnostic: builder code goes through `BuildHost` (`online_build.rs:44-63`).

## 3. Building blocks

Neither library has nested dissection, vertex separators, an elimination tree, symbolic factorization, FM refinement or minimum degree [U4, R1, R3, by grep].

Every LAGraph routine below is in `experimental/` and has open TODOs. `liblagraphx.a` is linked (`native-deps/src/recipes/lagraph.rs:56-59`).

| Building block | In Rust today | Use for ordering | Use for supergraph |
|---|---|---|---|
| Skeleton: `eWiseAdd(ONEB, A, Aᵀ)` then `select OFFDIAG` | exists (`input.rs:112-113` @05; C `cch.c:181-185`) | input | input |
| Matching by masked `mxv`, each vertex taking its heaviest neighbour (Manne–Bisseling) | new | Coarsening with no incidence matrix. Quality against SHEM is unmeasured. | — |
| `LAGraph_MaximalMatching`, `LAGraph_Coarsen_Matching` | bound, unused | Needs an n×e incidence matrix, about 4 extra entries per edge. Its "heavy" mode is random × weight, not SHEM. Carries no vertex weights. | — |
| Contraction `S·A·Sᵀ` (PLUS_TIMES) with vertex weights `S·w`; label projection by `mxv` | new (semirings, `build_Scalar_Vector`) | Clean fit (`Coarsen_Matching.c:483-506`) | — |
| One call per ND depth: drop cross-part edges with a select on a part-label vector, or `C⟨A⟩ = Sᵀ ANY_PAIR S` | new | design idea, unmeasured | — |
| `LAGr_ConnectedComponents` (FastSV) | used | Components. It unloads `G->A` while it runs, so it needs a private copy (`FastSV7.c:42-46`). | — |
| `LAGraph_KCore` | bound | Peeling trees off the 2-core gives low ranks with no fill (derived) | — |
| `LAGr_MaximumMatching` | bound | König step: edge cut to vertex separator | — |
| `LAGr_MaxFlow`, `LAGraph_MinCut` | MaxFlow used; MinCut not bound | Flow refinement. MinCut returns an edge cut, so a vertex separator needs a node-split network. | — |
| `LAGraph_Hdip_Fiedler` | bound | Spectral split, FP32; weak quality (§4a) | — |
| BFS (`LAGr_BreadthFirstSearch`) | bound | Level structures. The parent output is not deterministic (ANY_SECONDI). | — |
| `GxB_Matrix_extract_Vector` | new | induced subgraphs | B = A(π,π), the skeleton in rank space |
| `select` TRIU or a user `IndexUnaryOp` | partial (OFFDIAG only) | — | U0 = triu(B,1); the frontier stop rule |
| `LAGraph_msf` (Borůvka) | not bound | — | Parallel elimination tree from a minimum spanning forest weighted by max rank (R3; checked at n ≤ 50) |
| `GxB_Vector_sort`, reduce-to-vector, `GrB_Matrix_diag`, concat/split | new | sort keys, degrees | assembly, row tiles |
| `GxB_rowIterator` | exists (`matrix.rs:2080-2202`) | host loops (FM, leaves) | host elimination game; query rows |

Costs that apply to every option:
- **PreJIT.** New semirings and user operators need a Linux PreJIT harvest. Without one they run generic kernels, measured at +6% to +87% (`matrix.rs:172-198`) [U4].
- **Row iterators.** An attach costs 1,654 instructions, a re-seek 298 (`degree_bench.rs:16-19`) [U1], so iterators should be reused.
- **Index width.** GraphBLAS uses 32-bit indices when dimensions are at most 2³¹. A test on main asserts this for a default matrix (`csr.rs:277-278`) [R3].
- **Thin wrappers.** Today's Rust wrappers cover unmasked ANY_PAIR products, element-wise operations and a few selects (`matrix.rs:1062-1179`) [U4]. Most of the table's "new" rows are wrapper work.

## 4. Candidate approaches

### (a) Ordering

**The quality bar** is California:

| Ordering | Arcs | Triangles |
|---|---|---|
| C METIS | 5,527,199 | 48.6M |
| feral on main | +5.6% | +10.9% |

(FALKORDB.md:29-30) [U3]. Ordering is the cost to beat: 3.62 s of the 4.63 s `CREATE` (#3321) [U1].

- **O1. Multilevel ND built from GraphBLAS calls (the baseline)** [U3 §3, R1 A].
  - **Steps:**
    - Rank isolated nodes and peeled trees first.
    - Match with masked `mxv` and contract with `S·A·Sᵀ`.
    - Run the initial bisection and node FM as host loops over each level's matrix.
    - Get the separator by König's theorem, with `LAGr_MaximumMatching` or a host loop at about 100 vertices.
    - Make one GraphBLAS call per ND depth, covering every part.
    - Order the leaves with host AMD or MMD, in parallel.
  - **Expected quality:** about METIS.
  - **Evidence.**
    - Reaching METIS depends on details feral lacks: vertex merging, MMD at ≤120 vertices, coarsening to 40–100 vertices, and a boundary separator refined by FM in both directions.
    - Best-of-5 bisection plus METIS's balance tie-break already took feral from 59.6M to 53.9M triangles [U3 §4].
    - Batched (parallel) refinement measured +6.1% separator size against serial FM (mt-Metis) [R1].
  - **Run time:** unmeasured.
- **O2. O1 plus max-flow refinement on a band around each separator** (`LAGr_MaxFlow`/`MinCut` over a node-split network) [R1 B].
  - KaHIP separators are 10% smaller than METIS's.
  - On continental road graphs, flow-based orders (InertialFlowCutter) give 2.9× fewer triangles and 3.2–4.0× faster queries than METIS. On a 0.44M-node graph the gap is about 1.4× (IFC Table 2) [R1].
  - Unmeasured for us.
- **O3. BFS level structure plus max flow** (Inertial Flow without coordinates). Experimental; quality without coordinates is unknown [R1 C].
- **Rejected as the main method:**
  - pure spectral: 37% more factorization operations than multilevel ND (Karypis–Kumar 1998);
  - pure BFS separators: about 2× the spectral ones (Pothen–Simon–Liou 1990) [R1].
- **A guard for non-road graphs.**
  - For any order, the supergraph holds a clique the size of the minimum balanced separator, inside at least n/3 search spaces (CCH Lemma 2).
  - Heuristic treewidth bounds for social and web graphs are 0.5–27% of n [R1].
  - So estimate the top separator first, and refuse to build or fall back when it is large.

### (b) Supergraph from an order

- **G1. Host elimination game over GraphBLAS input.**
  - Today's child-absorption loop (`order.rs:381-437`) reads `triu(A(π,π),1)` through a row iterator and writes with `GrB_Matrix_build`.
  - Cost: O(|G+|) plus the row sorts; 97 ms on California today [U1].
  - Lowest risk.
- **G2. Tagged frontier (R3's formulation A)** [R3 §2].
  - Steps:
    1. Build the elimination tree: host Liu, or the minimum spanning forest route.
    2. Relabel ranks into postorder. The fill does not change (survey p.8).
    3. Tag each U0 entry with the next row in its column.
    4. Run at most tree-height rounds of `mxm(Pᵀ, F)` followed by `select(i < tag)`.
    5. Assemble once at the end.
  - Each arc is produced once, so the work is O(|G+|).
  - Europe's tree height is at most 1,232, so a full build is about 1.2k pairs of calls.
  - Checked only in Python at n ≤ 50. GraphBLAS cost is unmeasured. It needs a different monoid (§6, C1).
- **G3. Child absorption one tree level at a time** [U4 option 2, R3 B].
  - Correct.
  - Must not read and write the same matrix at every level: pending tuples get reassembled on each read, which can cost O(height × |G+|) (`GrB_examples.tex:170-176`).
- **G4. Repeat until nothing changes.**
  - Two forms: `U ← U ∪ triu(Pᵀ·U, 1)` [U3, R2 S2-GrB] and `F ← F ∪ triu(Fᵀ·F, 1)` [U4].
  - Test oracles only.
- **Batches.**
  - In a postorder, every elimination subtree is a contiguous range of ranks. Its rows depend only on its own edges, and sibling subtrees share no arc (survey Lemma 2, p.9) [R3 §3].
  - So subtrees can be independent builder batches, with the separator rows last.
  - Row and column counts (Gilbert–Ng–Peyton) give exact preallocation and batch sizes.
- **Storage.**
  - U as a boolean rank-space matrix that stores no values, plus Uᵀ: about 8 B/arc, against 16 B/arc in `RankRows` (derived) [U4, R3].
  - But slots (I3) have no GraphBLAS equivalent. The options [U1 §4]:
    - (a) store the slot as the value: +4 to 8 B/arc;
    - (b) key the weights by pattern: this conflicts with #3774;
    - (c) unload a Container: needs a private copy.
  - Storage that tolerates changes [R2 §4, R3 §4]:
    - `VersionedMatrix` using only `dp`, since fill only ever adds;
    - row tiles (`GxB_Matrix_split`/`concat`);
    - a rank-keyed copy-on-write row store (`cow_btree`).

### (c) Incremental maintenance

The order stays fixed unless a row says otherwise.

| Change | Rule | Cost and evidence |
|---|---|---|
| Edge or node delete | No topology change. The weight becomes +∞ and the slot stays (S1). | O(1). Stale arcs accumulate. C rebuilds at max(1024, n/8) deletions (`cch_index.c:468-496`); Rust tip has no such check [U2, U3]. No per-arc multiplicity counters: they cost one integer per arc, and CHOLMOD dropped them as too costly [R2]. |
| Edge insert where v is an ancestor of u | Walk up from u, adding v, until a row already has it (S2) | At most one path segment. On grids (Python): median 6 to 22.5 arcs [R3 §3]. |
| Edge insert across sibling subtrees | Davis–Hager update along the merged path (S2), or re-dissect the subtree at LCA(u,v) (S5) | S2 on grids (Python): 4–7% of all arcs per edge [R3]. Cost is proportional to the changed entries (DH99 Thm 4.1). S5 runs ND on the subtree; if the LCA is the root, it is a full rebuild. |
| Node created within capacity | It already holds one of the lowest ranks, so its first edges are leaf inserts | Degree 1: one arc, no fill. Degree d: its neighbours must become a clique, which S2 pushes up [R2 S3, U4]. |
| Node past capacity | Append it at the top, give it an order-maintenance rank label with gaps (needs 64-bit hypersparse), or reserve ranks | Top placement on grids: 46 / 94 / 190 arcs at n = 256 / 1,024 / 4,096. The rest is unmeasured [R2, R3]. |
| Cleanup | Rebuild U from the kept order in subtree batches (S6). Reorder fully on a decay signal, as a new generation (S7). | Europe: order 341 s against supergraph 1.6 s (survey Table 1) [R2]. |

**Decay.** No CCH measurement exists. The one analogue is DH99 §7.3 [R2]:

| Order computed on | Fill at the start | Peak fill |
|---|---|---|
| the starting matrix | 381k | 7.4M |
| the union of every edge that will ever appear | 831k | 1.49M |

Signals to track (derived) [R2 §3]:
- the arc count against its value at build time;
- elimination-tree height;
- deletions since the build.

## 5. Hard problems the design must answer

1. **Batching the ordering itself.**
   - ND works top-down: the top separator needs the whole graph coarsened, yet every batch reads a fresh snapshot.
   - Open questions:
     - What is the cursor? Perhaps node-id ranges while the skeleton is read into a private matrix, then postorder rank ranges for the subtree batches.
     - How does partial ND state survive between batches, copy-on-write?
     - How do writes made between batches enter that state (§2)?
2. **FM refinement.**
   - It is sequential, so it runs as a host loop over each level's matrix.
   - Seed it from the boundary only; feral rescans all n vertices on every pass (`fm_refine.rs:350-378`). Reuse the row iterators.
   - Neither GraphBLAS nor LAGraph has a parallel node-separator refiner [U3].
3. **Arc slots against GraphBLAS storage** (I3, and #3774's chunked metric). Choose one of the three options in §4b, with append-only ids if rows change.
4. **Inserts that cross a separator.** Accept S2's fill, re-dissect (S5), or move one endpoint into the separator (S4). The thresholds between them are unmeasured on road graphs.
5. **Where a new node goes:** past capacity, and high-degree new nodes left at the bottom.
6. **Deletions and decay.** The valve values, and what a "decayed" order means in arcs, triangles and tree height.
7. **The same answer on every replica.**
   - ND can be deterministic if it uses only seeded integer, MIN and MAX operations. BFS parents and ANY reductions are not deterministic, and independence from the thread count is unverified [U3, U4].
   - Incremental state depends on the write history and on when batches ran. A replica rebuilds from the RDB (`cch_index.c:161-163`; tip `cch/mod.rs:375-377`).
   - C already depends on history, through its stale arcs.
   - Decision needed: must replicas return only the same distances, or the same paths too? Tie rules follow rank order (I9).
8. **Memory parity with C.**
   - Today:
     - C and Rust both spend 48 B/arc: 16 for the topology plus 32 for the metric [U1].
     - feral peaks at 447 MiB over the input, against 363 MiB for METIS (#3321).
     - Rust keeps both CSRs alive through customization, where C frees its scratch (`cch.h:81-93`).
   - Risks in a new design:
     - a coarse matrix at every level;
     - `effective` copies, about 400 MB at 9M edges;
     - U must keep 32-bit indices, which rank labels with gaps would break;
     - each GraphBLAS weight matrix repeats the pattern (weights are out of scope, but storage choices must allow them) [R3 §5].
9. **GIL and replica budget** for S2 on the write path, since a cross-separator insert can be large. Bound it, and pass the rest to a builder batch (`Restart` or a TooLarge-style split).
10. **Per-call overhead times depth.** Tree height reaches about 1.2k on Europe and is unmeasured on California. Every new semiring and operator also needs PreJIT coverage.

## 6. Contradictions and how they were resolved

- **C1. Elimination game cost.** [U4] says there is no work-efficient GraphBLAS elimination game; [R3] says formulation A is O(|G+|).
  - Both hold: nothing has been published, and R3's form is derived from Ribizel–Anzt and checked only in Python.
  - Derived here: two frontier entries in the same column can land on the same row in the same round. One has just reached its segment's end (an LCA); the other is still inside its segment. `ANY_SECOND` keeps either one arbitrarily (`GrB_objects_BinaryOp.tex:32`, as U4 cites), so the entry that should continue can be lost. A MAX on the tag keeps it.
  - G2 is a prototype candidate; G1 stays the fallback.
- **C2. Fixpoint cost.** U3 says at most tree-height rounds, R2 says the round count is unproven, U4 says oracle only. They describe two different fixpoints. Both can cost tree height × whole-matrix work, so both are oracles only.
- **C3. Level-by-level absorption.** U4 recommends it; R3 flags it for reading a matrix it is still writing. Keep each level's output separate and assemble once.
- **C4. The unit of local repair.**
  - R2 says: the smallest ND cell, valid only if no lower-ranked vertex outside the cell has an arc into it. R3 says: any elimination subtree in postorder.
  - Resolved in R3's favour, with a proof sketch. An arc (x, y) with x < y makes y an ancestor of x. So a lower-ranked vertex with an arc into subtree T is itself in T.
  - ND cells stop matching elimination subtrees once inserts cross cells. So the repair unit is the elimination subtree of the current supergraph.
  - Caveat: `up` rows outside the subtree stay identical, but the ancestors' `down` rows and the metric above it change.
- **C5. Zero-copy reads.** U3 lists Container unload as zero-copy. U1 and U4 note that it empties the matrix (`matrix.rs:568-573`; `FastSV7.c:42-46`). So a shared matrix is read with the row iterator only, and a Container is used only on a private copy.
- **C6. Europe ordering times.**
  - The figures:
    - R1: METIS 125 s, IFC8 2,448 s (IFC paper);
    - R2: order 341.2 s against supergraph 1.6 s (survey Table 1), and IFC 242 s on 16 cores;
    - U4: about 6 minutes.
  - These come from different papers and different hardware, and are not reconciled.
  - All agree that ordering costs about two orders of magnitude more than the supergraph, as do our own 3.62 s against 97 ms [U1].
- **C7. California rebuild time.** 4.63 s (#3321) against 4.0 s (#3363); these are different runs.
- **C8. Memory per arc.** U1 says Rust matches C at 48 B; U4 and R3 say 16 B can become 8 B. The second figure covers the topology only, before any slot storage.
- **C9. Placing a new node.** U4: a leaf costs about deg² inserts. R2: an order-maintenance label. R3: append at the top.
  - Spot-checked: `nd.rs:61-64` @main already gives every unused id slot one of the lowest ranks, and `maint.rs:22-27` @08-maint treats creating a node as not structural.
  - So placing new nodes at the bottom is already today's behaviour within capacity. The open choice is for nodes past capacity and for high-degree nodes.
- **feral defaults** (inside U3). The docs say feral uses the METIS 5.2.0 defaults; the source differs:

  | Setting | feral | METIS source |
  |---|---|---|
  | switch to leaf ordering at | 200 vertices | 120 vertices |
  | coarsen down to | 120 vertices | 40–100 vertices |
