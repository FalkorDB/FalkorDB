# Design B: the separator tree as the persistent, versioned CCH index

Tags cite the research reports in `../research/` ([U1]–[U4], [R1]–[R3]; the brief is [B]). "Derived" means reasoning in this document with no outside source. "Unmeasured" means nobody has measured it. Code citations are the reports' own unless marked otherwise. Stack facts come from the PR texts of #3773, #3774, #3363 and #3772, and from `partial.rs` and `maint.rs` on `origin/cch/11-partial` and `origin/cch/08c-build-catch-up`.

## 1. Summary

1. **The separator tree is the index.** It is a tree of cells. Each cell owns a fixed, contiguous rank range laid out as `[child 0][child 1][trees hanging off the cell][separator]`. A version of the index is an `Arc` root, and a write copies only the path from the changed cell to the root.
2. **Rank = cell offset + local position.** Ranks stay dense over `0..n` (I1 is kept). A cell's size never changes within a generation, so re-ordering a cell moves no rank outside it.
3. **The supergraph is stored in tiles.** A tile is a rank-space, iso-bool `GrB_Matrix` holding the upward rows of a subtree (about 64k arcs), or the separator rows of one large cell. A tile can also carry an overlay of replacement rows. The downward rows (`Uᵀ`) are one `VersionedMatrix`.
4. **Every arc has a stable id.** A base row's id is the tile's `arc_base` plus the entry's position. An overlay row stores its ids as values. Ids are only ever appended. The metric (#3774's `WeightArr`) is indexed by id and never shifts.
5. **Ordering** is multilevel nested dissection (ND) written with GraphBLAS: masked `mxv` matching, `Pᵀ·A·P` contraction, and host FM over row iterators. Leaves use host MMD. Expected quality is about METIS's (O1 [B §4a]).
6. **The build** runs as resumable batches on the native builder: Read (node-id slabs) → Dissect (work items) → Assemble (tiles, postorder) → Customize (rank ranges) → Absorb (the backlog, in commit order). The build is a pure function of a frozen skeleton S0. Every write made during it goes into the backlog.
7. **Incremental maintenance.**
   - Deletes and reweights go through scoped recustomization only.
   - An edge into an etree ancestor uses an S2 path walk.
   - An edge between sibling subtrees re-dissects the smallest cell that holds the cross (S5), escalating to the parent if the repair above it would cross again. Reaching the root means a new generation.
   - Arc-less vertices swap ranks for free, which places new nodes.
8. **Recomputation.** One engine handles every change. It collects a seed set D0 (changed seeds, new arcs, new or relabelled triangles) after all topology changes of a commit, then runs one rank-ordered scoped pass that is bit-identical to a full customization.
9. **Where work runs.** The writer does the topology change plus the recomputation within a budget. Past the budget, the index goes to `CatchingUp` and the builder replays the same per-commit function, so the result does not depend on where it ran.
10. **What it gives up:**
    - `up(r)` is no longer a `&[Rank]` that owns its storage;
    - slots are no longer contiguous across all rows;
    - metric arrays need append-only growth and compaction;
    - new nodes never join a cell inside a generation;
    - replicas agree on distances, not on paths;
    - rows are read through the GraphBLAS iterator's internal row view.

## 2. Data structures

### Per published version: `CchVersion`, one `Arc`, so (topology, metric) is atomic

| Structure | Type | Versioned how | Bytes |
|---|---|---|---|
| `tree: Arc<Cell>` | `Cell { lo, sep_lo, hi: u32, children: [Option<Arc<Cell>>; 2], stats: CellStats }`. The root is k-ary: `[bottom pool][component cells…][top pool]` | Path copying: a write copies the changed cell and its ancestors, depth × about 48 B | about 48 B per cell, about 2n/L cells for leaf size L ≈ 60–120, so under 1 B/node (derived) |
| `tiles: Arc<[Arc<Tile>]>` | sorted by `lo`, about 300 entries on California (derived: 5.8M arcs / 64k, plus about 190 separator tiles) | copied whole on change (about 10 KB) | negligible |
| `Tile.base` | `GrB_Matrix` iso bool, by row, sparse, rows `hi-lo`, columns 2³¹−1, 32-bit indices | immutable; replaced only by a fold | 4 B/arc + 4 B/row |
| `Tile.arc_base` | `u64` | — | — |
| `Tile.overlay` | `Option<GrB_Matrix<u32>>`, hypersparse by row; whole replacement rows whose values are arc ids | duplicated on first write (at most base/8 entries before a fold) | 8 B per entry of a changed row only |
| `down` (Uᵀ) | `VersionedMatrix<bool>`, rank space, hypersparse, 32-bit | base plus `dp`/`dm`, O(delta) per version (`versioned_matrix.rs:151-171` [U2]) | 4 B/arc + 8 B per non-empty row |
| `rank_to_node`, `node_to_rank` | chunked copy-on-write `u32` arrays (#3774's `WeightArr` mechanism) | a chunk is copied when written | 4 + 4 B/node |
| `free_ranks` | `Arc<RoaringBitmap>` of ranks whose vertex has no arc | copy-on-write | under 1 B/node (derived) |
| metric | four `WeightArr`s indexed by arc id. **Storage format out of scope**; this design only needs append and compaction | #3774 | 32 B/arc today |
| `stats` | generation, arcs and height at build, deletions since build, intruder arcs, dead slots | — | — |

**Per arc.** Topology costs 8 B (4 in the tile, 4 in Uᵀ). Today it is 16 B in Rust and 16 B in C (`up`/`down`, 8 B each; [U1 §1]). Overlay rows add 8 B per entry, only on rows that changed, and a fold bounds them. With the 32 B metric that gives 40 B/arc, against C's 48 B/arc.

**Per node.** About 21 B (4 for the tile `p`, 8 for the Uᵀ `h`+`p`, 8 for the maps, about 1 for the cells), all derived. Today Rust spends 32 B/node. C spends 16 B for `perm`/`iperm` plus 48 B of row pointers plus `arr_t` headers [U1 §1]. 32-bit indices hold only while dimensions stay at or below 2³¹ ([U4 §4]; `csr.rs:277-278` [R3]).

**What a cell does *not* own.** The angle proposes that cells own local skeleton matrices. This design drops that, because a resident copy of the skeleton costs about 8 B per skeleton entry on top of the graph, and C frees its skeleton after the build (`cch.h:76-93` [U1]). A cell owns its rank range (its vertex set is `rank_to_node[lo..hi)`) and, through its tiles, its rows. A cell's local matrix R_C is extracted from the graph only when the cell is re-dissected (§6).

### The I3 slot problem: stable arc ids

Arc ids, read as rows:
- **Base row r:** `slot(up(r)[i]) = tile.arc_base + pstart(r) + i`, where `pstart` is the row's start position inside the tile.
- **Overlay row:** the id is the stored value.

When a tile is created or folded, an allocator hands it a fresh `arc_base` at the end of the id space. Its old range becomes dead slots. So:
- an arc's id never changes while the arc lives, even across topology versions (stronger than I3's "stable for the life of a Topology");
- a relabelled or extended row keeps the ids of its surviving arcs, so their metric values never move;
- dead slots are counted, and above 12.5% of live arcs a compaction batch renumbers the tiles contiguously and permutes the metric off the GIL, as a new `CchVersion`.

At that threshold the worst-case overhead is 4 B/arc on the 32 B metric, which keeps the total below C's 48 B (derived).

How a row is read:
- Rows are read zero-copy through a `GxB_Iterator` attached to the tile. After `seekRow`, the iterator's `Ai[pstart..pend)` *is* the row slice, and `pstart` is the position.
- Iterators should be reused: an attach costs 1,654 instructions and a re-seek 298 (`degree_bench.rs:16-19` [U1]).
- This relies on the iterator struct defined in `GraphBLAS.h` (risk R1, check M1).
- Fallback: keep a host array of row starts per tile (4 B/row), and walk rows with `nextCol`.

### Build-only state: `BuildState` in a `Building` entry, private and copy-on-write

- S0: the skeleton in node space, as a `VersionedMatrix<bool>`.
- U0 = `triu(S0(π,π), 1)`: the skeleton's upward rows in rank space.
- The ND work queue and the coarse level matrices, held through `Arc`.
- The tiles built so far.
- The partial metric.
- The backlog: a `RoaringTreemap` of edge ids, plus their commit sequence numbers.

## 3. Ordering algorithm

| # | Step | Call | GraphBLAS or host, and why |
|---|---|---|---|
| 0 | Read S0, slab by slab | `GrB_extract` of a row range from each type's `m`, `dp` and `dm`; mask with ¬dm; union, then `GxB_Matrix_concat`; at the end `eWiseAdd(ONEB, W0, W0ᵀ)` + `select OFFDIAG` (`input.rs:112-113` @05, `cch.c:181-185` [U3]) | GraphBLAS. Never builds `effective` for a whole type, which costs about 400 MB and 76 ms at 9M edges (discussion.md) |
| 1 | Degrees and the bottom pool | `GrB_reduce` PLUS over PAIR to a vector; `select` degree = 0 | GraphBLAS |
| 2 | Hanging trees | `LAGraph_KCore` (k = 2) on a private copy; peel order (round, id) | GraphBLAS, plus a host sort. The trees fold into their attachment's cell, below its separator, with no fill [R1 §1, U4 §1] |
| 3 | Components | `LAGr_ConnectedComponents` on a private duplicate, since FastSV unloads `G->A` (`FastSV7.c:42-46` [U4]) | GraphBLAS |
| 4 | Matching | masked `mxv` with MAX over a packed key `(w << 32) \| (~j)`, built by a user IndexUnaryOp; a mutual-pick check by host pass over `pick`; ≤ log n rounds (Manne–Bisseling [R1 §3A]); 2-hop masked `mxm` when a level shrinks under 15% | GraphBLAS, plus an O(n) host check. The packed MAX makes ties go to the lowest id, deterministically |
| 5 | Contraction | `P = GxB_Matrix_build_Scalar_Vector(cmap)`; `A_c = offdiag(Pᵀ·A·P)` over PLUS_TIMES_INT64; `w_c = Pᵀ·w` (`Coarsen_Matching.c:483-506` [U3 §3]) | GraphBLAS |
| 6 | Initial bisection | GGGP, 7 trials, at 40–100 vertices | host: per-call overhead dominates at about 100 vertices [U3 §3] |
| 7 | Edge cut to node separator | all boundary vertices, then two-sided and one-sided FM (METIS `separator.c:36-56` [U3 §1]); König via `LAGr_MaximumMatching` only when the boundary has more than 1,000 vertices | host |
| 8 | Projection | `mxv(P, labels)` | GraphBLAS |
| 9 | Refinement | boundary = `mxv(A_l, side)`; node FM seeded from the boundary only, with a lazy heap and reused iterators | host: FM is sequential [U3 §3, R1 §1]; feral's full rescan is the bug to avoid (`fm_refine.rs:350-378` [B §5.2]) |
| 10 | Split into cells | per ND depth: one `select` with an IndexBinaryOp reading the part labels drops every cross-part edge at once. A small cell uses `GxB_Matrix_extract_Vector` | GraphBLAS. One call per depth instead of one per cell [U3 §3, U4 §2]; unmeasured (M3) |
| 11 | Leaves (≤ 120 vertices) | MMD over the leaf matrix through an iterator | host: no GraphBLAS minimum degree exists [U4] |
| 12 | Best of 5 when n ≥ 5,000; balance 1.2 | — | the METIS/feral lever: 59.6M → 53.9M triangles [U3 §4] |

**Determinism.** Only integer PLUS, MIN and MAX reductions and seeded host loops are used. There is no ANY and no floating-point sum. BFS parents are never used ([U4 §1]; check M5).

**Expected quality.** About METIS. The bar on California [B §4a]:

| Ordering | Arcs | Triangles |
|---|---|---|
| C METIS | 5,527,199 | 48.6M |
| feral | 5,838,003 | 53.9M |

The first PR must match feral. The tuned target is METIS +3%, reached by using METIS's own settings where feral differs [U3 §4]:
- MMD at 120 vertices;
- coarsening down to 40–100 vertices;
- a separator built from all boundary vertices;
- vertex compression.

Flow refinement (O2) is the later upside. On Cal/Nev, IFC8 queries take 21.7 µs against METIS's 47.9 µs (IFC Table 2 [R1 §2]). Unmeasured for us.

**Guard.** After the first bisection of the largest component, the build refuses (`Failed`, and the procedure uses Dijkstra) when the separator exceeds max(2,000, 1% of n) vertices. The reason: a separator of size |S| puts an |S|-clique into at least n/3 search spaces (CCH Lemma 2 [R1]). California's treewidth bound is 252, against 11k–920k for social and knowledge graphs [R1 §2]. The threshold is unmeasured.

## 4. Supergraph construction

After Dissect:
- `π` is final;
- U0 = `select(TRIU, 1)` of one `GxB_Matrix_extract_Vector(S0, π, π)`. Extracting once, instead of per tile, avoids an O(n) inverse map per call [U4 §2].

Tiles are planned from the tree: every maximal cell of at most 16,384 vertices is one subtree tile, and every larger cell's separator rows form one tile.

For each tile, in postorder (Assemble):
1. `B_T = GrB_extract(U0, rows [lo, hi), all columns)`. The range is contiguous, so this costs O(rows).
2. Host elimination game, G1 [B §4b]: up(r) = B_T(r) ∪ ⋃ over children (up(c) ∖ {r}).
   - A child in the same tile is a scratch row.
   - A child in a lower tile is one of that tile's boundary rows, read as a slice.
3. `GrB_Matrix_build` → base; `arc_base` is allocated; `GrB_transpose(base)` goes into `down` as a `dp` part. Parts are collected and assembled once per batch, not per level, to avoid R3's pending-tuple trap [R3 §1].
4. `GrB_wait` before the batch publishes. The rule: no published object may carry pending work (`mvcc_graph.rs:332-347` [U2]).

**Cost.** O(|G+|) plus the row sorts, the same work as today's game (97 ms on California [U1 §5]), plus about 300 build/transpose calls (unmeasured). The tagged frontier (G2 [R3 §2]) is a later optimization behind check M3, which needs the MAX-on-tag fix (C1 [B §6]).

**The Topology contract, and what it costs readers:**

| Invariant | Today | Design B | Readers touched |
|---|---|---|---|
| I1 | dense `0..n`, fixed | dense `0..n_rank`; grows at the top; arc-less vertices may swap ranks between versions | query bound (`query.rs:236-240`), seed |
| I2 | `&[Rank]` ascending | `RankRow`: a base slice, or an overlay slice plus ids; ascending | `gallop_to` (`query.rs:110-114`), the relax cursor (`customize.rs:282-294`), `partial.rs` |
| I3 | slot = flat position; arrays sized by `arc_count` | slot = `arc_base + pstart + i` or a stored id; arrays sized by `slot_capacity()`; ids stable across versions | `x_slots_start + i` (`customize.rs:270-276`), `up_range(r).zip(up(r))` (`query.rs:113`), `slot - up_range.start` (`partial.rs:158`) |
| I4, I5, I6, I8, I9 | — | unchanged; built and maintained by construction | — |
| I7 | exact transpose, a slice | an exact transpose in a `VersionedMatrix`; `down(r)` is a merge iterator (`versioned_matrix.rs:63-70`) | `partial.rs:173-204` |
| `Rank` | `NonZeroUsize`, 8 B | `NonZeroU32`, 4 B; the `Option` niche is kept | everywhere; mechanical |

Every changed function needs a line in `proofs/*/COVERAGE.tsv`.

## 5. Batching on the native index builder

| Phase | Cursor | Batch unit | State kept between batches | Writes |
|---|---|---|---|---|
| Read | node id | a slab of about the budget's edge count (10k growing to 4M, `online_build.rs:94-147` [U2]) | S0 parts, behind `Arc` | an id written after its slab was read goes to the backlog |
| Dissect | work-item index | one coarsening level, one initial bisection, one refinement level, or a group of leaves | coarse matrices and part labels, immutable behind `Arc` | backlog only; S0 is frozen |
| Assemble | rank | tiles until about the budget's arcs | tiles, `down` deltas, the id allocator | backlog |
| Customize | rank | the push sweep over apexes [c, c′); at the batch's end every arc with lower end below c′ is final (I8) | partial metric chunks | backlog |
| Absorb | backlog position | commits, in order, until the budget | the full hierarchy | — |

- **The cursor** is `(phase: u8, position: u64)`, which is `Copy + Eq`.
- **`place(id)`** answers "below" only in Read, for an edge whose source is in a slab already read. Elsewhere it answers "not yet". This is fine because CCH writers never apply tuples to the build state; they only record ids (`maint.rs` `edges`).
- **The written set.** Each batch moves `since_last_indexer_step` into the private backlog. The framework's set therefore stays small, which removes the O(|written|)-per-write growth [U2 §4].
- **Progress.** `index_build_read` reports edges read, vertices dissected and arcs built, so `Sizing` can grow the batches.
- **Stopping.** `stopping()` is checked between work items.

**Why the build can ignore writes until Absorb (derived).**
- Every edge whose state changed after its slab was read is in the backlog: the framework records ids written since the last step, and writes before a slab's read are harmless duplicates.
- So the S0 snapshot plus a replay of the backlog equals the current graph.
- The hierarchy built from S0 is valid for S0. Absorb applies §6's rules, the same function a built index uses, to the backlog in commit order.
- Interleaving therefore needs no special code: there is no partial metric for a write to corrupt, and no half-built row for an insert to touch.

**Publishing.**
- Read, Dissect, Assemble and Customize batches publish `Same` or `Swap` (only the entry's `Arc<BuildState>` changes). `CatchUp` is trivial for them: write ids are appended to the backlog in O(written).
- The last Absorb batch publishes the entry as `Built`. Its catch-up covers only the commits since that batch forked, and runs under the GIL when it fits B_gil (§6b). Otherwise the outcome is `Discarded(CatchUpTooLarge)`, and the next batch absorbs those commits off the GIL.
- This converges whenever write throughput is below absorb throughput. #3772's limit of five overtakes, then `Failed`, stays as the backstop.

**What readers see.**

| Situation | Readers get |
|---|---|
| `CREATE` or RDB load | `Pending`, so the Dijkstra fallback, 222 ms median on California (`fallback.rs:28-31` [U2]) |
| A new generation (§6, S7) | the old generation, still `Built` and maintained by writers, while the new one builds in the shadow and is swapped in at adoption. Peak memory is two hierarchies |
| `CatchingUp` | readers on versions after the hand-off fall back; older versions keep their `Arc` |

## 6. Incremental maintenance

**Classification at the writer.**
- Each written edge is classified in O(1) against the held version (#3363's `Cch::joins`).
- An unjoined pair (u, v) with rank u < v walks the etree from u while `parent(j) < v`. This costs the path length; etree height on California is unmeasured [U1 §5].
- The walk gives the ancestor case (it reaches v), or the row j* where the path passes v.

| Change | Rule | Correctness | Cost | Where |
|---|---|---|---|---|
| Weight SET or REMOVE; edge create on a joined pair; edge delete | metric only (S1) | G′ ⊆ G ⊆ H; no topology change [R2 §5] | 0.22 ms p50 per SET, 0.35 ms per delete, California (#3363) | writer |
| Insert, ancestor case | S2 walk: add v to up(j) up to the first row that holds v; parents are unchanged since v > p(j) | the stop rule holds by closure (checked [R3 §2]) | rows on one path segment; grids: median 6–22.5 arcs [R3 §3] | writer |
| Insert, cross, j* a root with up(j*) ⊆ {v} | attach j* under v | up(j*) ∖ {v} = ∅ | 1 arc | writer |
| Insert, cross | S5: C = the smallest cell containing j* and v. Re-order C on R_C ∪ {(j*, v)}, where R_C = the graph induced on C plus `offdiag(L_Cᵀ ANY_PAIR L_C)`, and L_C = U(rows < lo_C, columns C) are intruder rows. Rebuild C's rows; relabel the intruder rows (overlay, same ids); repair above C with S2 from C's top rows. If that repair meets a cross case, retry at parent(C). If C is the root, start a new generation | outside C's range, rows hold the same sets (fill path; checked [R3 §3]); only C's rows, intruder labels and the boundary cliques above C change | ND on \|C\| plus C's subtree arcs; derived 5–10 ms at \|C\| ≈ 1,000; on grids, S2 would add 4–7% of all arcs per crossing [R3 §3] | writer while \|C\| ≤ 4,096 and within budget; otherwise hand-off |
| Insert, both ends in the bottom pool | Davis–Hager S2 path merge [R2 §2] | DH99 Thm 4.1 | proportional to the changed entries | writer, within budget |
| Node create within capacity | nothing: it is arc-less and already ranked low (`nd.rs:61-64` [B C9]) | it has no edge | 0 | — |
| Node past capacity | append its ranks at the top as arc-less ranks; at its first edge, swap ranks with a free bottom rank | two arc-less vertices swap ranks without changing any arc or value (derived: every arc at both ranks seeds +∞ and has no finite lower path) | 2 map entries | writer |
| Node delete | its edges are deleted (S1); the rank stays; once it has no incident edges, it can swap into the free list, so a reused id starts at the bottom | as above | its degree | writer |
| Relationship-type or weight-definition change | a new index, so a new generation | — | a build | builder |

**Decay signals, all thresholds unmeasured** (C's valve is `cch_index.c:468-496` [U3]):

| Signal | Action |
|---|---|
| deletions ≥ max(1,024, n/8) | S6: rebuild U from the kept order, in tile batches, with fresh ids |
| live arcs / arcs at build > 1.10 | S6 first; if still over, re-dissect the worst cells in the background |
| intruder arcs > 5% of arcs, height > 1.25 × at build, or S5 escalation to the root | S7: a new generation in the shadow |
| dead slots > 12.5% | compaction |
| overlay > base/8 in a tile | fold the tile (a new base and id range; the metric values are copied) |

Ordering costs about 100× the supergraph build, which is why the order is the thing to keep: 341 s against 1.6 s on Europe [R2]; 3.62 s against 97 ms for us [U1].

## 6b. Recomputing modified edges

**Dependencies.** An arc (y, z) with y < z reads its seed and the arcs (x, y) and (x, z) for every x in down(y) ∩ down(z) (`partial.rs:173-204`). So a value changes only when:
- its seed changes;
- its set of lower triangles changes; or
- a lower arc changes bitwise.

Middles also depend on the apexes' ranks (I9).

**The recompute set.** The seed set D0 is recorded as node-id pairs and cell ids, which do not depend on ranks. It is resolved to slots after *all* topology changes of the commit, so S5 relabels need no remapping. Closure D adds the top arc (min(hi, w), max(hi, w)) for each w in up(lo) ∖ {hi} of every arc whose weight changed bitwise. That is the rule in `partial.rs:213-228`, correct because every dependent has a higher lower end.

| Change | D0 | Why that is enough |
|---|---|---|
| SET, REMOVE, joined create, delete, node delete | the pair's arc | only its seed changed |
| S2 ancestor walk over rows R | the new arcs (j, v) for j ∈ R, computed from scratch; the top arcs (v, w) for w ∈ up(j) ∖ {v}, j ∈ R | the only new triangles have apex j ∈ R with v as a corner. A triangle with (j, v) as its top arc is covered by computing (j, v) from scratch. Parents are unchanged |
| S5 on cell C = [a, b) | every arc with lower end in C; every arc (y, z) with y, z ≥ b and an apex x ∈ C; the D0 of the S2 boundary repair | Below a, values depend only on apexes below a, so they are bitwise unchanged; intruder arcs keep their ids and values. C's arcs are new. Arcs above b keep their *sets*, but their apexes in C are relabelled, so ties and middles, and floating-point association, can move. Seed them and let the bitwise test stop the propagation |
| Swap of arc-less vertices | ∅ | every value at both ranks is +∞ with no middle |
| A new generation, S6, compaction | everything: the Customize phase, a full sweep (219 ms on California, #3323) | — |
| A build's backlog | the union over commits, after each commit's topology is applied in order | the metric is a pure function of (topology, seeds) |

**Algorithm.** `recustomize_seeded` (#3363/#3773) is generalized in four ways:
1. D0 items can be *fresh* arcs, which have no old value and always count as changed.
2. `down` rows come from the `VersionedMatrix` merge iterator.
3. An arc budget aborts the pass. Nothing is published, because the pass runs on a private `CchVersion`.
4. For S5, C's own subtree is recomputed with the push sweep restricted to [a, b), at about 0.09 µs per arc [U1 §5], before the heap takes the arcs above b.

Each arc is recomputed term for term, as in the full sweep: same seed, apexes in ascending order, a strict `<`. So the result stays bit-identical to `customize`. That is #3773's test oracle, and it extends to random topology edits.

**Cost on California** (cited unless marked):

| Path | Figure |
|---|---|
| Scoped recustomization | about 1.3 µs per arc; 0.22 ms p50 per SET (#3363) |
| Many pairs | 512 pairs 0.41 s, 1,024 pairs 0.51 s, 4,096 pairs 0.95 s (`maint.rs`) |
| Full path | about 0.5 s; 0.47 s when it runs under the GIL at catch-up (#3772) |
| 200×200 grid | 10.7 ms scoped against 464 ms full (#3773) |
| S2 insert (derived) | about 200 arcs in D0 (22 rows × about 10), so about 0.3–0.5 ms |
| S5 at \|C\| ≈ 1,000 (derived) | 3.8k subtree arcs × 0.09 µs plus 1–5k boundary arcs × 1.3 µs, so about 1.5–7 ms plus ND |

**Interleaving.**
- *An edge modified before the batch that covers its region*: read fresh by that batch.
- *An edge modified after it*: backlog, then Absorb.
- *Writes between batches*: backlog. Phases never read the backlog, so they cannot mix two versions.
- *A subtree rebuilt while its ancestors' metric is already computed*: the S5 row above (the apex-relabel seeds).
- *Order within one commit*: all topology changes first, then D0, then one recompute, then publish.

**Where it runs, and its bounds.**
- **Writer:** within B_w = 16,384 recomputed arcs (about 21 ms at 1.3 µs per arc, derived) and an S5 cell of at most 4,096 vertices.
  - Over budget, the writer drops its private copy of the index, sets the entry to `CatchingUp { base, backlog }`, and appends ids for later commits in O(1).
  - The builder replays the same per-commit function and publishes through `commit_indexer_batch`.
  - This also replaces the full customization that runs inside `GRAPH.EFFECT` on a replica (#3363 "Open").
- **Under the GIL at adoption:** B_gil = 8,192 arcs, about 10 ms, the same order as a column's catch-up cap of about 13 ms (`online_build.rs:126-131` [U2]). Over it, `Discarded(CatchUpTooLarge)`.
- The decision between S2, S5, escalation and a new generation never reads a budget. A budget only decides *where* the work runs, so writer, builder and replica produce identical states.

**MVCC.**
- All recomputation runs on a private `CchVersion` (`new_version`: `Arc` bumps; `WeightArr::set` copies only its chunk [U2 §3]).
- Tiles and `down` are waited before publish.
- The pair is published as one `Arc`, with the graph version (writer) or through `commit_indexer_batch` (builder).
- A reader holds one `Arc<CchVersion>`, so it sees either the old (topology, metric) pair or the new one, never a half-recomputed metric.

## 7. MVCC and replication

**What versions share.**
- Each version owns only what it wrote:
  - the cell path;
  - the tiles' overlays;
  - `down` deltas;
  - map chunks;
  - metric chunks (median 0.23 MB plus a 181 KB table per writing version, #3774).
- Everything else is shared through `Arc`.
- An old snapshot keeps working because nothing it reaches is ever written (`cow.rs:48-59` [U2]).
- Replaced versions are dropped outside the GIL (`online_build.rs:348-360`).
- Open: whether index-held matrices must join the pre-fork `wait_all` [U2 Unknown]. Publishing only waited objects makes that a no-op.

**Determinism: decided "same distances", not "same paths".**
- Maintenance is per commit, deterministic and independent of the budget (§6b), so the master and a replica that apply the same commits to the same hierarchy end up bitwise identical, provided ND does not depend on the thread count (M5).
- But a build reads S0 at batch-timing-dependent versions, and an RDB load rebuilds from the definition only (`cch/mod.rs:375-377` [U2]), so generations differ between nodes.
- Distances are exact on every replica. Paths can differ on ties.
- C already depends on history through its stale arcs [B §5.7].
- Making paths identical would mean replicating a build snapshot, which replicas do not have.

**RDB.** It keeps the definition only, as today. A later, optional format version could save `rank_to_node` (4 B/node, about 6 MB on California) plus the cells (about 1 MB), so a load skips Dissect: Read, Assemble and Customize only (derived, about 0.5 s against about 4 s).

## 8. Memory and time budget

All numbers below are derived unless marked. California: n = 1,540,096; 5.84M arcs (feral) or 5.53M (METIS) [U1, U3].

| | California | 9M edges, road-like (×2.7) | 9M edges, the 1500² grid |
|---|---|---|---|
| Topology, steady | 8 × 5.84M + 21 × 1.54M ≈ **79 MB** (today 142.7 MB measured, #3321) | about 215 MB | ≥ 47M arcs (≥ 20.8 arcs/node, R3's 4,096-node grid) → ≥ 0.43 GB |
| Metric (out of scope; 32 B/arc) | 187 MB, +10% chunk overhead (#3774), at most +12.5% dead | about 510 MB | ≥ 1.5 GB |
| Total against C at 48 B/arc | 40 B/arc + 21 B/node, against 48 B/arc + 64 B or more per node | — | — |
| Build peak over input | max(Dissect: S0 at 20 MB + int64 levels ≤ 80 MB; Customize: S0 + tiles + `down` + metric ≈ 290 MB), against 363 MiB for METIS and 447 MiB for feral (#3321) | Read never builds `effective` (400 MB, discussion.md) | — |
| Shadow generation | + one hierarchy, about 270 MB | — | — |
| Per SET | 0.23 MB median of chunks, 181 KB table (#3774) | — | — |
| Per S2 insert | overlay rows + a tile overlay copy (≤ 64 KB) + `down` delta + 10 KB of tile index + about 1 KB of cell path | — | — |
| Per S5 (\|C\| ≈ 1k) | ≤ 0.5 MB of tiles + 128 KB of new metric slots | — | — |
| Time: SET / S2 / S5 small | 0.22 ms cited / 0.3–0.5 ms / 5–10 ms | — | — |
| Time: full build | ordering unmeasured (feral 3.62 s); game about 97 ms; customize 219 ms | — | — |

## 9. Implementation plan

**Prototypes first, each with a pass/fail bar.**

| # | Question | Pass |
|---|---|---|
| M1 | Iterator row view (`Ai[pstart..pend)`, `pstart`) on v10.5.0; query on tiles | roadNet-CA p99 ≤ 526 µs (#3324) and ≤ +5% against slices |
| M2 | GraphBLAS ND quality, time and peak on California | arcs ≤ 5,838,003, triangles ≤ 53.9M, time ≤ 3.62 s, peak ≤ 447 MiB; target METIS +3% |
| M3 | One call per depth against extract per cell; GraphBLAS call overhead over depth | Dissect ≤ feral's time |
| M4 | Cross inserts on California: LCA cell size for local inserts (endpoints ≤ 3 hops apart) and for random ones; S5 against DH fill | median \|C\| ≤ 2,000 for local inserts; writer p99 ≤ 20 ms |
| M5 | Bitwise-identical order at 1, 4 and 16 OpenMP threads | identical |
| M6 | Overlay costs, S2 end to end | ≤ 1 ms p50 |

Each needs a check that the measurement can fail (break it on purpose).

**PRs, in order.**

| # | PR | What it proves |
|---|---|---|
| 1 | GraphBLAS wrappers: masked mxm/mxv over semirings, `extract_Vector`, `build_Scalar_Vector`, TRIU/OFFDIAG/IndexUnaryOp selects, reduce, the iterator row view; plus a PreJIT harvest (CLAUDE.md) | the building blocks run at PreJIT speed |
| 2 | `CchVersion` (cells, tiles, `down`, maps) built from **feral's order** | the metric is bit-identical to today's `Topology` on the testkit and on California |
| 3 | GraphBLAS ND behind a flag | M2's gates |
| 4 | Delete the `CchInput` CSRs, `nd.rs` and the feral dependency | goal 1 |
| 5 | The batched build (five phases, `BuildState`, backlog) | goal 2; column builds are no longer starved (#3771) |
| 6 | The generalized recompute engine (fresh arcs, merge-iterator `down`, budget abort) | goal 3 oracle: bit-identical to `customize` after random topology and weight edits |
| 7 | S2 ancestor inserts, which replace `Update::Rebuild` there | — |
| 8 | S5 with escalation; R_C extraction | — |
| 9 | `CatchingUp` hand-off and replay | — |
| 10 | Placing nodes past capacity; swaps | — |
| 11 | Decay signals, S6, compaction, S7 shadow generation | — |
| 12 | Optional: the order in the RDB | — |

## 10. Risks and open questions

1. **R1.** The iterator row view depends on `GraphBLAS.h`'s iterator struct. The fallback is a host row-start array plus row cursors (M1).
2. GraphBLAS ND quality and time (M2). Nothing published exists [R1, U4]. If M2 fails, ordering may keep feral, fed through iterators.
3. How often cross inserts happen on real workloads, and how far S5 escalates, is unknown (M4). Random global inserts tend toward the root.
4. New nodes stay as intruders from the pool, so quality decays. The S7 thresholds are guesses.
5. Determinism at different thread counts (M5); the "same distances" contract has to be documented.
6. Reading `down` through a three-way merge is slower than a slice in scoped passes. Its fold policy runs on the builder (unmeasured).
7. The hand-off sends readers to the 222 ms fallback where #3363 would spend 0.5 s inside the writer. B_w is a policy choice to make with the team.
8. `WeightArr` must support appends and compaction remaps. This touches the out-of-scope metric storage.
9. A large backlog after a long build may not converge under heavy writes; the backstop is `Failed` after five overtakes.
10. The contract changes (`Rank` as u32, `RankRow`, `slot_capacity`) ripple through `query.rs`, `customize.rs`, `partial.rs`, `maint.rs`, the procedure and the Lean coverage table.
11. The peak during a shadow generation is two hierarchies. Whether that matches C's rebuild peak is unmeasured.
