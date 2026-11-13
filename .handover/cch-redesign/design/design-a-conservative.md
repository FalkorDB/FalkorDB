# Design A (conservative): CCH ordering, supergraph and incremental upkeep on GraphBLAS

Angle: lowest risk. Keep the Phase 1 contract (I2, I4–I9) and the open stack (#3771, #3773, #3774, #3363, #3772), and change one thing at a time. Tags such as [U1] and [R3] point to the reports in `../research/`. "Derived" marks reasoning with no outside source. "Unmeasured" means nobody has measured it. Nothing in this document was measured for it.

## 1. Summary

1. **Ordering (O1).** Multilevel nested dissection over a private GraphBLAS copy of the skeleton.
   - Large cells: matching by masked `mxv`, contraction by `S·A·Sᵀ`.
   - Initial bisection, node FM and leaf minimum degree are host loops. They read the private level matrices through a zero-copy Container unload.
   - Small cells run entirely on the host.
   - METIS's settings, not feral's.
2. **Supergraph (G1).** Today's elimination game, reading `U0 = triu(S(π,π),1)` from GraphBLAS and emitting two GraphBLAS matrices in rank space:
   - `Up`: values are `u32` arc ids;
   - `Down`: its transpose, also carrying arc ids.
3. **Arc ids.** Stored as values and append-only within a lineage. They equal today's flat slots on a fresh build, so the metric layout (#3774) is unchanged.
4. **Batched build.** The native-index builder runs in phases: read skeleton → peel → dissect → supergraph → customize. The order is computed from a stale private snapshot, which is allowed, because only the supergraph has to cover the current graph. Writes during the build are logged as edge ids and absorbed at the end by the incremental rules.
5. **Incremental updates.** The order stays fixed.
   - Deletes become +∞ (S1).
   - Inserts get a rank-ordered closure repair (S2, Davis–Hager), on the writer when small and on the builder when large.
   - New nodes take bottom ranks reserved at build time.
   - Decay triggers a resymbol (S6) or a re-order (S7) as a new generation.
6. **Recompute (goal 3).** #3773's scoped recustomization, extended:
   - new arcs enter at (+∞, none) and go into the front;
   - the heap carries `(lo, hi, id)`;
   - rank-ordered pops make the result bit-identical to a full customization of the new topology.
   - Bounded at 10,000 arcs on any path that holds the GIL. Past that it hands off to a builder "fix-up" batch.

**What it gives up:**
- METIS-class order quality only. That means 2–4× slower queries than flow-based orders on large road graphs [R1 §2, IFC Table 2].
- No local re-dissection (S5) in the first ship.
- New versions fall back to Dijkstra while a hand-off or re-order is pending.
- Replicas guarantee the same distances, not the same paths.
- Little memory gain: topology stays at 16 B/arc in the default, because `Down` keeps its ids. Only per-node memory halves. The 12 B/arc option is a measured choice (§2).

## 2. Data structures

All rank-space matrices are `n_r × n_r`. Here `n_r = node_cap + reserve`, and `reserve = max(1024, node_cap/64)` (§6). Row-major, sparse, waited before publish.

| Structure | Type | Kept | Versioned how | Bytes |
|---|---|---|---|---|
| `Up` | `VersionedMatrix<u32>`, rank space; value = arc id | yes | base `m` shared by `Arc`. Inserts go to `dp` only, since fill only grows; `dm` stays empty [R3 §4, U4 §5]. `dp` is folded into `m` on the builder. | 4 (col) + 4 (id) per arc; 4 per rank |
| `Down` | `VersionedMatrix<u32>` = `Upᵀ` with ids | yes | same | 8 per arc (4 if iso, the option below); 4 per rank |
| `rank_of`, `node_of` | `ChunkedArr<u32>` (#3774's `WeightArr<T>` generalized, plus an append) | yes | chunk copy-on-write | 8 per node |
| Metric (4 arrays) | `WeightArr` indexed by arc id, plus an append | yes | as #3774 | 32 per arc (format out of scope) |
| `LineageStats` | arcs at build, deletions, dead ids, nodes past reserve, generation | yes | copied (small) | O(1) |
| Build state | `Arc<BuildState>` in the `Building` entry (§5) | while building | `Arc` per part | transient |

**Arc identity (the I3 slot problem).** The I3 options are listed in [U1 §4]. This design takes option (a), the slot as the value: the id is the value of `Up(lo, hi)` and of `Down(hi, lo)`.
- A fresh build assigns ids in row-major order, so `id = up_range(r).start + i` as today. Every metric consumer and #3774's chunking keep working unchanged.
- S2 inserts append ids `arc_capacity, arc_capacity+1, …`. The metric arrays grow by appending chunks; the last chunk is copied on write.
- Ids are never reused inside a lineage. A lineage is one topology and the incremental versions descended from it.
- S6 and S7 start a new lineage, with dense ids again.
- `Down` carrying ids removes both `arc_slot` binary searches per apex in `partial.rs`'s merge (`partial.rs:173-204` @11, spot-checked at tip). That is why the default keeps the values. An iso `Down` (12 B/arc) is the fallback if memory decides it (gate M2).

**Contract changes (and every reader they touch).**
- I3 becomes "id = stored value".
  - Removed: `up_range` and "`arc_count` sizes the arrays".
  - Added: `up_ids(r)`, which yields `(hi, id)`.
  - Readers to change:
    - `relax`'s `x_slots_start + i` (`customize.rs:270-276`);
    - `query.rs:113` (`up_range.zip(up)`);
    - `partial.rs:158`. Its heap stores `(lo, hi, id)`, so it no longer maps a slot back to `hi`.
  - Arrays are sized by `arc_capacity`. Dead ids hold (+∞, none) and nothing reads them.
- I1 is unchanged: the reserved ids are ordinary unused id slots, ranked lowest as today (`nd.rs:61-64`).
- I7 is strengthened: `Down` also carries ids.
- I2, I4, I5, I6, I8 and I9 stay exactly as stated in [U1 §2].

**Against C.** C spends 48 B/arc [U1 §1].
- This design: 8 + 8 + 32 = 48 B/arc, or 44 with an iso `Down`.
- Per node: 8 (maps) + 8 (two 32-bit row-pointer arrays) = 16 B, against Rust's 32 B today. C spends 64 B plus a header per row [U1 §1].
- This assumes GraphBLAS keeps 32-bit indices at these sizes (`csr.rs:277-278`, [R3 §5]). Gate M3.
- No per-arc counters. Multiplicities are rejected, as CHOLMOD rejected them [R2 §2].

## 3. Ordering algorithm (O1)

Input: the skeleton `S` (iso bool, node space, symmetric, no diagonal) = `eWiseAdd(ONEB, A, Aᵀ)` then `select OFFDIAG` per indexed type, merged (`input.rs:112-113` @05; C `cch.c:181-185`) [U3 §3]. It is read in node-id slices from `m/dp/dm` (§5), never through a whole-type `effective` copy, which costs about 400 MB at 9M edges (`cch-background-fold/discussion.md`). Every matrix below is a private copy, so a Container unload is legal [C5].

| Step | Operation | GraphBLAS or host, and why |
|---|---|---|
| 0 Degrees, isolated nodes | `d = reduce_rows(PLUS, S)` (`count_in_rows`, `matrix.rs:852`). Rank `d=0` lowest, in id order, as `nd.rs:62-64` does. | GraphBLAS: one call |
| 1 Peel trees | `LAGraph_KCore` (bound) gives the 2-core. Peeled vertices take the next ranks in peel-round order, then id order. They add no fill [U4 §1, R1 §1]. | GraphBLAS; deterministic (exact) |
| 2 Components | `LAGr_ConnectedComponents` on the private 2-core (it unloads `G->A`: `FastSV7.c:42-46`). Each component is a cell with a contiguous rank range. | GraphBLAS |
| 3 Cell ≥ 5,000 vertices: coarsen | Repeat until ≤ 100 vertices or < 5% shrink (METIS `ometis.c:399-403`; feral `coarsen.rs:139-160`). **Match:** key matrix `K = apply(A, idxop: (w≪32) \| tiehash(j))`; `y = mxv(K, unmatched, MAX_FIRST_UINT64)` masked to unmatched rows; mutual pairs via `GxB_Vector_extract_Vector(partner, partner)`. Rounds run to a fixpoint (Manne–Bisseling [R1 §3]). 2-hop fallback below 15% shrink: masked `A·A` over unmatched vertices, degree-capped [U3 §3]. **Contract:** `P = build(cmap)`, `Ac = select(OFFDIAG, P·A·Pᵀ)` with PLUS_TIMES_INT32, `vw_c = P·vw` (`Coarsen_Matching.c:483-506`). | GraphBLAS: whole-matrix parallel. Integer MAX with an index tie-break is deterministic for any thread count (to verify, M1). |
| 4 Initial separator (≤ 100 vertices) | Host: 7 GGP/BFS trials plus edge FM (METIS `initpart.c:82-89`). The separator is the lighter of two: METIS `ConstructSeparator` (boundary plus 2-sided then 1-sided FM, `separator.c:36-56`) and a König cover (host Kuhn at this size). | Host: per-call overhead dominates at ~100 vertices [U3 §3] |
| 5 Uncoarsen and refine | Project with `mxv(Pᵀ, side)`, then node FM (`FM_2WayNodeRefine1Sided` semantics) seeded **from the boundary only**, unlike feral's full rescan (`fm_refine.rs:350-378`). It reads the level's `Ap/Aj/Ax` through a Container unload. Best of 5 for cells ≥ 5,000 (`ometis.c:352-387`). | Host: FM is sequential; no node-separator refiner exists in GraphBLAS or LAGraph [U3, R1] |
| 6 Recurse | Separator vertices take the top ranks of the cell's range (`ometis.c:199-204`). Sides become cells via `GxB_Matrix_extract_Vector(A, side, side)`. | GraphBLAS per large cell |
| 7 Cell < 5,000 | The whole multilevel ND on the host over the unloaded arrays. | Host: avoids per-call overhead × ~2n/120 cells (derived: about 26k cells on California) |
| 8 Leaf ≤ 120 vertices (METIS `MMDSWITCH`, `defs.h:52`) | Minimum degree over a 120×120 bitset, exact elimination, tie → lowest id. | Host: our own code, O(k³/64) per leaf |

Optional and measured, behind flags: vertex compression (METIS `compress=1`, `ometis.c:97-113`), and components at every level (feral) against `ccorder=0` (METIS). These are the remaining candidate causes of feral's +5.6% arcs and +10.9% triangles [U3 §4].

**Expected quality.**
- Same family as METIS and feral, so expect between feral (5,838,003 arcs, 53.9M triangles) and C METIS (5,527,199, 48.6M) on California (FALKORDB.md:29-30) [U3 §4].
- Evidence: best-of-5 and the `mindiff` tie-break alone moved feral from 59.6M to 53.9M triangles [U3 §4]. Serial FM avoids mt-Metis's +6.1% separator loss from batched refinement [R1 §1].
- Flow refinement (O2) is deferred. Its payoff is real only on large road graphs (2.2× query time on Cal/Nev, 3.2× on Europe; IFC Table 2) [R1 §2].

**Determinism.** Seeded integer hashes and MIN/MAX/PLUS over integers only. No `ANY` and no BFS parents [U4 §1]. So the order depends only on the skeleton snapshot (verified by M1).

**Guard for large separators.**
- (a) After the top bisection, if the top separator exceeds 4,096 vertices, the build stops with `Failed("separator too large")`. CCH Lemma 2 puts that clique into ≥ n/3 search spaces [R1 §2]. For reference, California's treewidth upper bound is 252 (Maniu et al.).
- (b) The elimination game (§4) stops when arcs exceed `max_arcs` (default 64·n, configurable), before allocating the metric.

## 4. Supergraph construction (G1)

1. `B = GxB_Matrix_extract_Vector(S, node_of, node_of)`, then `U0 = select(TRIU, B, 1)`. This is rank space, private [U1 §4, R3 §2].
2. Unload `U0` into a Container, which is zero-copy and legal because the matrix is private. Run today's child-absorption loop (`order.rs:381-437`) in increasing rank:
   - row(r) = `U0(r,:)` ∪ ⋃children (row(c) ∖ {r});
   - sort and dedup;
   - parent = first entry.
   - Scratch is `first_child` and `next_sibling` as `u32`.
3. Assign ids in row-major order. Hand `(Ap, Aj, Ax=id)` to `GxB_load` / `GrB_Matrix_build` as `Up.m`. Load versus build, which is copy versus zero-copy, is settled by measurement in PR 3.
4. `Down.m = GrB_transpose(Up.m)`, then `GrB_wait` on both.

**Cost.** O(|U0| + arcs) plus the row sorts. 97 ms on California over the CSR today [U1 §5]. Over a Container the inner loop is the same; the extract, transpose and load add whole-matrix passes. Expect ≤ 200 ms (unmeasured; gate M2).

It meets I2 (sorted rows, every entry above r), I4 (U0 ⊆ Up), I5 and I6 (the child-absorption fixpoint [R3 §1]), I7 (exact transpose) and I3' (ids). G2, the tagged frontier, stays a prototype: it is checked only at n ≤ 50, and C1 means it needs a MAX on the tag [Brief §6].

## 5. Batching on the native-index builder

**Key simplification (derived).** Any order is valid. Only the supergraph must cover the graph that is published (I4). So ordering batches may read slices from different snapshots and ignore writes entirely. Correctness is restored once, at the supergraph phase and the adoption, by the same incremental rules a built index uses (§6).

**Cursor.** `CchCursor { phase, position }`, ordered lexicographically. `place(edge_id)` always answers "above", so writers never apply into build state. They only record ids, as a `Building` entry does today (`maint.rs:297-314`). `index_build_read` returns units done (rows read, vertices ranked, arcs made), so `Sizing` can grow batches (`online_build.rs:94-147`).

| Phase | Batch unit | Partial state (copy-on-write) | Budget per batch |
|---|---|---|---|
| ReadSkeleton | Node-id row range `[a,b)`: `extract(m, a:b)` ∖ `dm` ∪ `dp` per type → slice. Transient memory is O(slice). | `Vec<Arc<Matrix>>` of slices, each immutable. At phase end: `GxB_Matrix_concat`, then `ONEB(A, Aᵀ)`. | rows ≈ 2M edges |
| Peel and components | one batch, whole skeleton | `Arc<Matrix>` skeleton, `ChunkedArr<u32>` ranks so far | O(nnz) |
| Dissect | One coarsening run of a big cell, or its refinement, or a set of cells from the work queue | Queue of `Arc<Cell>` (immutable submatrix plus rank range), level stack `Vec<Arc<Matrix>>`, rank chunks. A batch replaces only what it consumed. | edges touched ≈ 4M |
| Supergraph | One batch (§4) over `S ∪ log pairs` (below) | `Arc<Topology>` | O(arcs) |
| Customize | One full customization when estimated triangles ≤ 100M (California: 48.6–53.9M). Otherwise rank-range pull batches (§6b) | metric `WeightArr`s | triangles |
| FixUp | apply the log since the last pass (§6, §6b) | as Built | R_gil at adoption |

**Writes between batches.**
- The framework hands each batch's catch-up its written ids (`since_last_indexer_step`, `versioned.rs:436-443`).
- CCH's catch-up only appends them as one immutable `Arc<[EdgeId]>` segment to `BuildState.log`. That is O(written) under the GIL, with no limit needed. The set the framework copies is cleared every batch, so the "grows for the whole build" cost goes away [U2 §4].
- The Supergraph phase resolves the log against its build version. A created edge whose pair is new joins `U0`. A deleted edge needs nothing, because a superset is valid (+∞ arc) [R2 §1].
- Writes after the Supergraph fork are handled in the Customize and FixUp phases (§6b).

**Publish outcomes.**
- **Same / Swap:** move the new `Arc<BuildState>`.
- **CatchUp:** append the log segment. In phases up to Supergraph this costs only O(written).
- **Adoption** (the last Customize or FixUp batch): the framework fit runs #3772's `absorb` with the §6 rules, bounded at R_gil.
  - Within the bound, publish `Built`.
  - Past it, return `Absorbed::Restart`. Restart now **resumes at FixUp** with the built hierarchy kept in `BuildState`, instead of rebuilding from scratch.
  - The 5-restart `Failed` limit stays (`cch/mod.rs:394-399`).
- **Discarded:** the published `BuildState` is untouched, so the next batch redoes that unit.

**Readers.** While `Building`, every new version answers with the Dijkstra fallback (222 ms median on California, `fallback.rs:28-31` @09) [U2 §2]. Older versions keep their `Arc<Cch>`.

## 6. Incremental maintenance (built index, order fixed)

One flush per `Commit` segment, on the private version (`pending.rs:1473-1476`, `maint.rs:467-536`). The order of work is always **classify → topology → recompute → publish**.

| Change | Rule | Correctness | Cost | Where |
|---|---|---|---|---|
| Weight SET/REMOVE, insert on a joined pair, edge delete | No topology change. Reseed the pair's arc; delete = +∞ (S1). | G' ⊆ H unchanged [R2 S1, R3 §4] | 0.22 ms p50 SET, 0.35 ms delete (#3363) | writer |
| Node delete | Its edges cascade as deletes. The rank stays; `deletions += deg`. | as S1 | O(deg) plus the cones | writer |
| Edge insert (u,v), not joined, any position | **Rank-ordered closure repair (S2/DH).** Add v to `up(u)`. Pop rows by rank from a heap; for row j with additions D: if `p(j)` is unchanged, push `D ∖ {p}` into `up(p)`; if the parent changed (only when the endpoints were in sibling subtrees), push `up(j) ∖ {p_new}` into `up(p_new)`. Stop when nothing is new. Several inserts share one pass (DH01 Thm 4.2) [R2 §2]. New arcs get appended ids. | Adds only arcs of fill(H + uv) and stops when `U(j)∖{p(j)} ⊆ U(p(j))` holds everywhere. That is the validity rule [U4 §5, R2 §5], so I4–I6 hold. The tree may change, and nothing needs postorder contiguity. | Ancestor case: one path segment, median 6–22.5 arcs on grids. Across the top separator: about 0.9·n on grids, 4–7% of all arcs [R3 §3]. Road graphs: unmeasured (M4). | writer if \|N\| ≤ N_max = 4,096, else builder |
| Node created within `n_r` | Already isolated at a bottom rank. Its edges are inserts: degree 1 adds one arc and no fill; degree d makes its neighbours a clique through S2 [R2 S3]. | as S2 | small for road edits (derived) | writer |
| Node past `n_r` | `GrB_Matrix_resize` on the private `Up`/`Down` adds a block of `max(1024, n/64)` ranks at the **top**. A new node takes the next one, and its edges are S2 inserts (the union of neighbours' root paths [R2 §2 ereach]). `nodes_past_reserve += 1`. | Any rank is valid | O(n) once per block (the row pointers), plus path unions: 46/94/190 arcs on grids [R3 §3] | builder (resize), the rest writer |
| Relationship types | The index's type set is fixed. Edges of indexed types are the rows above; others are ignored (`maint.rs:216-325`). | — | — | — |

**Bounds and hand-off.**
- S2 runs on a private copy and aborts at N_max new arcs. The recompute aborts at R_gil = 10,000 arcs. That is about 13 ms at 1.3 µs per arc (`maint.rs:85-94`), the same as the column catch-up cap (`online_build.rs:126-131`).
- On abort, the entry becomes `Building{FixUp, base: old Arc<Cch>, log}` under a new generation. The builder applies the same rules there without bounds.
- On the master's serialized writer, #3363's behaviour may stay (scoped up to the valve, full past it) until M4 decides (open question 1). Under the GIL (replica `GRAPH.EFFECT`, adoption catch-up, writer rebase), R_gil always applies. That removes the 0.47 s customization under the GIL (#3772) and the 0.5 s one in `GRAPH.EFFECT` (#3363).

**Decay** (derived; thresholds to be set by M6). The C valve is `cch_index.c:468-496`.

| Signal | Action |
|---|---|
| deletions ≥ max(1024, n/8) (C parity) or dead ids ≥ 12.5% | S6: rebuild `Up` from the same order, new lineage, full customization (about 97 + 219 ms on California) |
| arcs ≥ 1.25 × arcs at build, or a single insert with \|N\| > arcs/8, or `nodes_past_reserve` ≥ reserve | S7: full re-order as a new generation |

In the first ship both run through the `Building` path, with fallback while they build. A later PR keeps the old lineage serving and maintained while a shadow generation builds (§9, PR 10).

## 6b. Recomputing modified edges (goal 3)

**What an arc's value depends on.** Arc a = (lo, hi) has value `min(seed(a), min over x ∈ down(lo) ∩ down(hi) of dn(x,lo)+up(x,hi))`, with the middle being the first x to win under strict `<` (`customize.rs:66-75`). So a's value or middle changes only when:
- (i) its seed changes;
- (ii) its apex set changes, meaning an arc (x,lo) or (x,hi) is new;
- (iii) a bottom arc's value changes;
- (iv) the rank labels of its apexes change.

**Recompute set per change type.** `cone(F)` is the closure under "a changed arc (lo,hi) enqueues (min(hi,w), max(hi,w)) for every w ∈ up(lo)". This is #3773's rule (`partial.rs:225-240` @tip). An arc is a bottom arc only of triangles whose apex is its own lower end, so the closure is complete (derived).

| Change | Front F | Why complete |
|---|---|---|
| SET/REMOVE, delete, node delete, insert on a joined pair | the pairs' arcs | (i) |
| S2 insert (any) | the input arcs plus all new arcs N, each preset to (+∞, none) | (ii): a new triangle has a new bottom arc or a new top arc, since an all-old triangle is old (I5). A new top arc is in N. A new bottom arc with a finite value changed from +∞, so it enqueues its tops. One that stays +∞ can neither lower a top nor win a strict-`<` tie. |
| Node past reserve | as S2 | as S2 |
| S6, S7 (new lineage) | every arc: a full customization | new ids |
| Re-dissected subtree T (S5, later) | every arc with lo ∈ T, plus the boundary clique B = {(y,z) : y,z ∈ up(x)∖T for some x ∈ T} | (iv): only arcs with an apex in T can name a T rank. Those are exactly B and T's own arcs. No row outside T contains a T vertex: an arc (y,t) with t ∈ T makes y a descendant of t. So B's pattern is unchanged, and so are its values: a basic-customized value is the best path through lower ranks [CCH §2], and all of T stays below B. Only floating-point summation order can move a value, and the cone absorbs that (derived; checked by M5). |

**Algorithm.** `recustomize_seeded` (`partial.rs` @11) with three changes:
1. The front takes arc ids directly.
2. The heap holds `(lo, hi, id)`.
3. Apexes come from merging `Down(lo)` and `Down(hi)`, which yields the two ids with no binary search.

Arcs pop in increasing `lo`, and every bottom arc has a lower end x < lo, so each arc is final before anything reads it (I8). Each recomputation is from scratch (reseed, then all apexes in increasing x), so the result is **bit-identical to a full customization of the new topology** whatever the preset values were. The preset only decides whether propagation fires.

Cost: Σ over popped arcs of (|down(lo)| + |down(hi)|) plus two seed lookups. Measured at about 1.3 µs per arc on California, seeds included (`maint.rs:85-94`), and 10.7 ms against 464 ms full on a 200×200 grid (#3773). A median SET is about 0.22 ms (#3363), so about 170 arcs (derived). A cross-top-separator insert on grids, at 4–7% of arcs, would be about 290k arcs on California, about 0.4 s, so it goes to the builder (derived from [R3 §3]).

**Interleaving with a batched build.**
- **During ordering phases** there is no metric, so no recompute. Writes only grow the log.
- **Customize as one batch** (the common case): it reads all seeds from its own snapshot V_k. Writes after V_k are absorbed at adoption: classify → S2 on the build's private `Up` → scoped recompute, bounded at R_gil, else FixUp.
- **Customize split into rank-range batches** (pull form, the per-arc kernel above):
  - Invariant: *every arc with lo < cursor is correct for the latest batch snapshot; arcs above the cursor hold only seeds.*
  - Batch k+1 first resolves the log segment since V_k.
  - (1) Topology first: S2 on the private `Up`; new arcs at (+∞, none).
  - (2) Arcs that are new, or reseeded, with lo ≥ cursor are just reseeded. Pull form computes them later from final inputs.
  - (3) Arcs with lo < cursor are the front of a scoped pass **capped at the cursor**. Arcs it would enqueue at or above the cursor are dropped, because pull form computes them later from scratch.
  - (4) Then the cursor advances.
  - So an edge modified *before* the batch that covers its region is simply seeded fresh. One modified *after* is repaired by the capped pass.
- **Subtree rebuilt while its ancestors are computed:** the B rule above, on a private version, published with the topology.
- **Order of operations, always:** topology (new arcs and ids, the `dp` deltas, metric arrays extended) → recompute → `GrB_wait` → publish. A recompute never runs on a topology missing the arcs, because a pair with no arc would read as `NeedsRebuild` (`partial.rs` refusal).

**Where it runs and its bound** (summary): writer up to N_max and R_gil; builder FixUp beyond that; S6 and S7 on the builder only.

**MVCC.**
- All of it happens on a private version: the writer's fork or the build version. `WeightArr::set` copies only its chunk (`customize.rs:182-189`). `Up.dp` and `Down.dp` are duplicated per changing version.
- The version's `Arc<Cch>`, holding the topology and metric together, is swapped atomically at publish. A reader therefore holds one consistent (topology, metric) pair and never a half-recomputed one.
- New ids lie beyond an old reader's `arc_capacity`, so old readers never see them.

## 7. MVCC and replication

- **Each version holds** `IndexEntry<CchIndex>` → `Built(Arc<Cch>{ Arc<TopologyVersion>{Up, Down, rank maps, stats}, Metric })` or `Building(Arc<BuildState>)`.
- **Sharing.** `Up.m` and `Down.m` are shared across a whole lineage. `dp` and the metric chunks are shared until written.
- **The publish rules** [U2 §3] are kept:
  - private-only writes (`cow.rs:48-59`);
  - every GraphBLAS object waited (`mvcc_graph.rs:332-347`);
  - the replaced version dropped outside the GIL (`online_build.rs:348-360`);
  - the data version kept (`mvcc_graph.rs:247-248`).
- **Folding `dp`** happens only on a builder version, and it changes no id, so the metric is untouched. It is triggered by the square-root rule (`versioned_matrix.rs:151-171`) or `|dp| > 64k`.
- **Replicas: same distances only.**
  - The hierarchy depends on write history, on when batches ran, on the bounds and on restarts. Each replica builds its own [Brief §5.7].
  - Shortest distances are unique, so every replica answers them identically.
  - Equal-length paths may differ: tie rules follow rank order (I9).
  - Same paths would need the rank vector and every topology event replicated as effects. That is a new wire format and replica-side apply cost (compare the v3 effects work), for a property C does not have either: C's stale arcs are history too [U3 §1].
  - The order itself is deterministic for a given skeleton (§3).
- **RDB.** Only the definition is saved, as today (`cch/mod.rs:375-377`; C `cch_index.c:161-163`), and a load rebuilds in the background. Persisting the rank vector (4 B/node, about 6 MB on California) to skip the 3.6 s ordering is a later, optional RDB field (open question 4).

## 8. Memory and time budget

California: n = 1.54M skeleton nodes [U1 §1] (the task brief says 1.9M; scale per node). 5.84M arcs with feral, 5.53M with METIS.

| Item | Value | Basis |
|---|---|---|
| Topology, steady state | 16 B × 5.84M + 16 B × 1.54M ≈ **118 MB** (iso `Down`: 95 MB), against 142.7 MB today | derived; today's figure measured (#3321) |
| Metric | 186.8 MB, +10% chunk headers (396–412 MB against 365 MB per held index) | computed [U1]; measured (#3774) |
| Reserve | 24k ranks × 16 B ≈ 0.4 MB | derived |
| Build peak | skeleton about 20 MB, plus ND levels ≤ 2× the 2-core (about 60 MB), plus topology 118 MB, plus weights about 40 MB, plus metric flat-then-chunked 2 × 187 MB ≈ **0.6 GB**, plus the old hierarchy if readers hold it. Against: feral alone peaks at 447 MiB over the input, METIS at 363 MiB (#3321). | derived, unmeasured (M1) |
| Ordering time | target ≤ feral's 3.62 s; C METIS takes 3.07 s | measured baselines (#3321) |
| Supergraph / customize | ≤ 200 ms / 219 ms | 97 ms measured today; customize measured (#3323) |
| Per update | SET 0.22 ms, delete 0.35 ms (#3363). Ancestor insert ≈ (N + cone) × 1.3 µs ≈ 0.25–0.5 ms. Cross insert: builder, ≈ 0.4 s on California. | measured; derived |
| GIL per event | ≤ 13 ms (R_gil), against 0.47 s today (#3772) | derived bound |

**9M edges.**
- A road-like graph with 4.5M pairs, scaled linearly from California's 3.5 arcs per pair: about 15.8M arcs and 4.2M nodes.
  - Topology about 320 MB, metric about 506 MB (derived).
  - Slice reads avoid the 400 MB / 76 ms `effective` transient (measured, `discussion.md`).
- A 1500×1500 grid (8,994,000 directed edges): extrapolating R3's grid arcs per node (9.6 → 14.7 → 20.8 per ×4) gives about 45 arcs per node, about 100M arcs, about 4.8 GB.
  - This is an unmeasured extrapolation, and it is exactly what guard §3(b) stops before the metric is allocated.

## 9. Implementation plan

**Prototype measurements first, before any PR beyond 1:**

| Id | Measure | Pass |
|---|---|---|
| M1 | GraphBLAS ND (O1 prototype) on California and a 9M-edge road graph: arcs, triangles, tree height, time, peak, and the same order across 1/4/16 threads | arcs ≤ 5.84M and triangles ≤ 53.9M (feral); time ≤ 4.5 s (1.25× feral); peak ≤ 447 MiB over the input; order identical across thread counts |
| M2 | `Topology` over `VersionedMatrix<u32>` with feral's order: query p99, full customization, scoped per-arc cost, with `Down` valued and iso | p99 ≤ 526 µs on roadNet-CA (#3324) and not worse than C; customization ≤ 1.2× 219 ms; scoped ≤ 1.3 µs/arc |
| M3 | Integer width of `Up`/`Down` at California size | 32-bit indices |
| M4 | 10k sampled insert pairs on California (local, within 3 hops, and global): \|N\|, \|cone\| | p95 of local inserts ≤ N_max and ≤ R_gil |
| M5 | Randomized property test: S2 plus the extended scoped pass and the B rule, against a full customization and Dijkstra | 0 mismatches over 10k graphs |
| M6 | Write-mix replay (inserts, deletes, node adds): arcs and query p99 over time | calibrates the §6 thresholds |
| M7 | Batched build under concurrent writes: GIL hold per publish, column-build delay | p99 ≤ 13 ms; delay ≤ largest batch (about 0.5 s) |

**PRs** (each small, each closing its own issue):
1. GraphBLAS wrappers: masked `mxv`/`mxm` over general semirings, `extract_Vector`, TRIU/positional select, MIN/MAX reduce, `Matrix<u32>`, private Container unload, plus a PreJIT harvest. *Proves* the kernels; generic kernels cost +6–87% [U4 §4].
2. `ChunkedArr<T>` with append, generalized from `WeightArr`. *Proves* copy-on-write append.
3. Topology over GraphBLAS, still on the feral order (G1 over GraphBLAS input, ids as values, the I3' readers changed). *Proves* I2–I9 with the existing tests, and gates M2 and M3.
4. GraphBLAS ND behind a `CCH_ORDERING` switch (default: feral). *Proves* M1.
5. Flip the default; delete feral and the `CchInput` CSR. *Closes* goal 1.
6. Phased batched build plus the build log, resuming at FixUp. *Proves* M7.
7. S2 closure repair plus the extended scoped recompute, replacing `maint.rs` `(None, Rebuild)` and `absorb → None` [U2 §3]. Adds the node reserve. *Proves* M4 and M5.
8. R_gil bound and the FixUp hand-off in GIL paths.
9. Decay signals, S6 and S7 as new generations.
10. Later: shadow generation (serve while re-ordering), S5 local re-dissection, O2 flow refinement, each behind M6 data.

## 10. Risks and open questions

1. **Writer bound versus fallback.** With R_gil on the master's writer too, a 512-SET write (0.41 s today, scoped) would serve Dijkstra (222 ms p50) until the builder publishes. Which is worse is decided by M4 and M6.
2. **Query latency over GraphBLAS rows.** Iterator attaches cost 1,654 instructions and re-seeks 298 (`degree_bench.rs:16-19`). Merging `m`+`dp` per row may break the 526 µs p99. Mitigation: fold `dp` eagerly on the builder, and reuse iterators.
3. **Order quality.** The O1 matching quality against SHEM is unmeasured [U3 §3]. If M1 fails, keep feral until it passes. PR 3 still removes the CSR from the supergraph and customization paths.
4. **RDB order persistence**, and whether replicas should load it (it would make rebuilds deterministic).
5. **Cross-separator inserts on real road data** (unmeasured). If they dominate, S5 moves forward.
6. **Two hierarchies alive** during S7 or S6 with readers: peak about 2 × 0.3 GB on California.
7. **PreJIT coverage** for every new operator, and determinism across thread counts (unverified [U3, U4]).
8. **The derived claims** (stale ordering snapshots, B-rule completeness, +∞ preset) rest on proof sketches until M5 runs.
