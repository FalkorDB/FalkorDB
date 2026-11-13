# CCH ordering, supergraph and incremental upkeep on GraphBLAS: final design

This document revises `01-synthesis.md` against three adversarial reviews. Every review issue gets a disposition in the appendix. No repo file was modified, and nothing was measured for this design. It runs about 7,800 words by `wc`, tables and appendix included. That is over the 2,500–4,000 target, because the tables are dense and the appendix answers 26 review issues.

**Sources.**
- [B] is the research brief, `../research/00-research-brief.md`.
- [U1]–[U4] and [R1]–[R3] are the source reports.
- [FOLD] is `../../cch-background-fold/discussion.md`.
- Stack code was read at `origin/cch/08c-build-catch-up` ("@08c"). That is the stack tip, and it contains #3773's `partial.rs`, #3774's `WeightArr`, #3363's `maint.rs` and #3772's `absorb`.
- File:line facts the reviews raised were re-checked at @08c for this revision:
  - `versioned.rs:86-93` (`Written`) and `:443-448` (the clear in `indexer_apply`);
  - `versioned.rs:564-583` (`Same`/`Swap`/`CatchUp`) and `:677-700` (`catch_up_index`);
  - `maint.rs:82-95` (`PAIR_CAP` and the cost line), `:121-137` (`set_structural`, `push`) and `:297-315` (`record`);
  - `graphblas/versioned_matrix.rs:206, 241, 889, 1114, 1162, 1367` (the fold policy; `set` exists only on `bool`);
  - `GraphBLAS.h:6977-6990` (v10 checkout): `GxB_Vector_load` with `GxB_IS_READONLY`;
  - `data_structures/cow_btree/mod.rs:1-30` @main: a persistent `(u64 key, u64 doc)` B⁺-tree with path copying, ≤ 16 B per entry and range cursors;
  - `src/lib.rs:103` @main: `INDEX_BUILD_THREAD_COUNT` defaults to 1;
  - `git grep rayon` on `origin/main`'s `Cargo.toml` files has no hits.

**How numbers are marked.**
- **cited:** the source is named.
- **derived:** arithmetic or reasoning in this document.
- **unmeasured:** nobody has measured it.

## 1. Summary

1. **Ordering.** We write our own multilevel nested dissection. At each depth, **every** cell of that depth is processed together by GraphBLAS calls over one block-diagonal matrix:
   - matching by masked `mxv` with unique integer keys;
   - contraction by `C·A·Cᵀ`;
   - label projection;
   - a split by a masked `ANY_PAIR` dot product.

   Initial bisection, boundary-seeded node FM and minimum degree on the leaves are serial host loops on the builder thread. GraphBLAS runs under a `GxB_Context` with its own thread budget.
2. **Supergraph.** Steps, in order:
   - Liu's elimination tree, a postorder relabel, and Gilbert–Ng–Peyton counts. The counts give the exact arc count, which serves as the size guard.
   - Today's elimination game, over one private extract.
   - The output is a rank-space topology with **arc ids as values**.
3. **Topology representation.**
   - Each of `Up` and `Down` is a **base plus a delta**.
   - The base is a GraphBLAS matrix whose arrays are owned by Rust in an `Arc`. GraphBLAS reads it as read-only (`GxB_IS_READONLY`), so the query and customization paths read the very same memory as flat slices.
   - The delta is a `cow_btree` keyed by `(row, column)`, whose value is the arc id. Writes path-copy, and it never folds by itself.
4. **Arc identity.**
   - An arc's id is its stored value.
   - On a fresh build, ids are dense and row-major, so #3774's layout is unchanged.
   - Within a lineage, ids are only ever appended, in `(lo, hi)` order within each pass.
5. **The build log.**
   - Writers record **node pairs at write time**, into a persistent append-only log of per-commit segments.
   - The log lives in the CCH index's own state, not in the framework's `Written` id set. So no batch publish (`Same`, `Swap` or `CatchUp`) can clear it, and no id needs resolving against an older version.
6. **Batched build.** The phases are ReadSkeleton → Order → Supergraph → Seed → Relax → Seal → (Repair).
   - The topology covers one snapshot plus the log.
   - The metric is a push sweep from one seed snapshot plus the log, and it is never patched mid-sweep.
7. **Incremental upkeep.** The order stays fixed.
   - Deletes reseed the arc to +∞.
   - Inserts on pairs with no arc get a closure repair (Davis–Hager).
   - Ranks are lazy: a node gets a rank on its first indexed edge, from a pool of arc-less ranks, or else a new top rank that lives only in the delta.
   - Decay starts a new lineage or a new generation.
8. **Recompute (goal 3).**
   - First the topology changes, then one scoped recustomization runs in rank order (#3773's engine, generalized). It is bit-identical to a full customization of the new topology.
   - On any path that holds the host lock, the work is bounded by a **time budget** checked by the clock. Past the budget, a priority Repair batch takes over.
   - Readers swap one `Arc<Cch>` holding the (topology, metric) pair.

**What it gives up:**
- Order quality targets METIS, with feral as the floor. Flow refinement comes later.
- New versions answer with the Dijkstra fallback (222 ms p50, cited) while a hand-off is pending. A single SET on a top arc can trigger one (§6b).
- Above a write rate of roughly 1,250 SET/s (derived), a Repair never adopts within budget. The backstop is a team decision (D2).
- Replicas promise equal distances, not equal hierarchies or paths.

## 2. Data structures

**Rank space.** `n_r = node_cap + reserve`, where `reserve = max(1,024, node_cap/64)` (derived). Ranks are `NonZeroU32` (4 B, with the `Option` niche) [B §1].

| Structure | Type | Versioning | Bytes |
|---|---|---|---|
| `Up.base` | `RankMatrix`: CSR arrays `Ap: Arc<[u32]>`, `Aj: Arc<[u32]>`, `Ax: Arc<[u32]>` (value = arc id), loaded into a `GrB_Matrix` with `GxB_IS_READONLY` whenever a GraphBLAS op needs it (`GraphBLAS.h:6977-6990`) | Immutable for its lineage. Shared by `Arc`. Replaced only by a builder fold | 8 per arc; 4 per rank |
| `Down.base` | the same, holding `Upᵀ` with the same ids | same | 8 per arc; 4 per rank |
| `Up.delta`, `Down.delta` | `CowBTree`: key `(row ≪ 32) \| col`, doc = id (`cow_btree/mod.rs:1-30` @main) | `clone` is an O(1) root bump. Each insert path-copies O(log n) pages. **No automatic fold** | ≤ 16 per delta entry (cited); bounded by the fold trigger |
| `rank_of` | `ChunkedArr<Option<Rank>>`: #3774's `WeightArr` chunks plus append | chunk copy-on-write (`customize.rs:182-189` @08c) | 4 per node |
| `node_of` | `ChunkedArr<u32>` (u64 once `node_cap` ≥ 2³²) | same | 4 per rank |
| `free_ranks` | `Arc<RoaringBitmap>`: ranks whose holder has no arc | copied when changed. It holds the isolated vertices plus the reserve | small |
| Metric (4 arrays) | indexed by arc id, length `arc_capacity`. **Format out of scope** | #3774's chunks, plus append | 32 per arc |
| `LineageStats`, `touched_since_fold` | counters; a persistent list of (commit seq, ranks whose delta rows changed) | copied / O(1) append | O(1) / O(written) |
| CCH log | `CchLog = Option<Arc<LogNode{seq, pairs: Box<[(u64,u64)]>, prev}>>`, plus a sticky `many: bool` | a persistent list: append is O(1) plus the segment. Consumed off the lock | 16 per logged pair |

**Why base plus delta, and not `VersionedMatrix<u32>`.**
- `VersionedMatrix` decides folds itself:
  - `dup()` latches a fold (`versioned_matrix.rs:1367`, `fold_balance` at `:241`, `WRITE_FOLD_K` at `:206`);
  - the next `flush()` then runs an O(|m|) eWiseAdd (`:1162`) on the writer, under the lock.
- Its `set` exists only for `bool` (`:889, :1114`).
- Writing a GraphBLAS delta while reading it pays a pending-tuple assembly on every read (R3 §1; GrB_examples.tex:170-176).
- The `cow_btree` delta avoids all three:
  - an insert costs O(log n), not O(|delta|), on every lock path;
  - it has no pending tuples;
  - a rank above the base's dimension can live in the delta, so **growing the node count never resizes or copies the base**.
- It is an existing engine structure, so goal 1's "our own structures" holds.

**Why the base exposes raw arrays.**
- Relax opens `up(y)` once per arc (5.84M on California) and steps once per triangle (≈54M), at `customize.rs:497-530` @08c.
- A re-seeked GraphBLAS row iterator costs ≈298 instructions, and every entry is an out-of-line FFI call (`degree_bench.rs:16-19`; `matrix.rs:2210-2229` @main).
- Derived from those: ≈1.7G instructions in seeks alone, so relax would take 0.6–0.9 s against today's 219 ms.
- A read-only Container load lets GraphBLAS operate on the same arrays with no copy. GraphBLAS still does all structural algebra: extracts, transposes, masks and folds. No second CSR exists. M2 measures both read paths. Decision D3.

**Arc identity (the I3 slot problem).** The id is the value at `Up(lo,hi)` and at `Down(hi,lo)`.
- **Fresh build.** `id = Ap[lo] + i`, as today, so the metric layout and chunk locality are unchanged.
- **S2.** A pass sorts its new arcs by `(lo, hi)` and appends ids `arc_capacity, arc_capacity+1, …`.
  - The metric grows by appending chunks.
  - New entries are preset to (+∞, +∞, none, none).
- **No reuse.** Ids are never reused within a lineage. Older readers have shorter arrays and older deltas, so they never see new ids.
- **New lineages.** S6 and S7 start a new lineage with dense ids. Dead ids hold (+∞, none) until then.

**Contract changes, and every reader they touch.**
- **I1′.** `rank_of` is partial and injective. A rank-less node has no indexed edge.
  - Readers: the query's endpoint lookup, which answers "no path" unless source = target; and `partial.rs:115-118`, which must check for the rank's presence as well as the bound.
- **I3′.** `up_ids(r)` yields ascending `(hi, id)` pairs from merging base and delta. Arrays are sized by `arc_capacity`.
  - Readers:
    - relax's `x_slots_start + i` (`customize.rs:497-513`);
    - `query.rs:113`;
    - `partial.rs:151,170` (the heap carries `(lo, hi, id)`);
    - `Cch::joins`.
- **I7** is strengthened: the ids are equal. **I2, I4, I5, I6, I8 and I9** are unchanged [U1 §2].
- Every changed function gets a line in `proofs/*/COVERAGE.tsv`.

**Against C** (48 B/arc, 64 B/node plus row headers [U1 §1]).
- Per arc: 8 + 8 + 32 = **48 B**, or 44 with an iso `Down` (D7).
- Per node: 4 (`rank_of`) + 4 (`node_of`) + 8 (two row-pointer arrays) = **16 B**.
- The delta adds ≤ 16 B per *delta* entry only, never per entity at steady state.
- This holds while the arrays are 32-bit (M3).

## 3. Ordering algorithm

**Input skeleton**, read in row slabs by node id (§5):
- per indexed type, take `pattern(m) ∪ pattern(dp)` of the slab's rows, **ignoring `dm`**;
- take the union across types;
- finally `S = select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ))` (`input.rs:112-113` @05; C `cch.c:181-185` [U3 §3]).

Ignoring `dm` gives a superset of the live pattern, which is valid for any order and for I4 [R2 §1]. It also avoids the `effective` merge: ~400 MB and ~76 ms at 9M edges [FOLD].

| Step | Operation | Where, and why |
|---|---|---|
| 0 Degrees | `d = reduce_rows(PLUS, S)`. Degree-0 vertices and unused ids enter `free_ranks` and take the lowest ranks in id order (`nd.rs:61-64`) | GraphBLAS: one call |
| 1 Hanging trees | Degree-1 peeling: masked `mxv` (PLUS_PAIR) over the frontier; ranks in (round, id) order; a host pass after 64 rounds [U4 §1, R1 §1] | GraphBLAS; host for long chains |
| 2 Components | `LAGr_ConnectedComponents` on a private copy (it unloads `G->A`; `FastSV7.c:42-46` [U4 §1]) | GraphBLAS |
| 3 Coarsen all cells of depth d | **Match** (Manne–Bisseling [R1 §3]):<br>- keys `K = apply(A_d, positional op: (w ≪ 32) \| scramble(j))`, which are unique, so MAX is deterministic;<br>- `y⟨unmatched⟩ = K MAX_FIRST unmatched`;<br>- mutual pairs via `GxB_Vector_extract_Vector`;<br>- a 2-hop pass when the shrink is under 15 % [U3 §2].<br>A mask drops candidate pairs whose combined vertex weight exceeds a cap modelled on METIS's `maxvwgt`, which the synthesis left out. The exact METIS formula is not cited here; PR 6 verifies it against the METIS source.<br>**Contract:** `A_c = select(OFFDIAG, C·A_d·Cᵀ)` with PLUS_TIMES_INT32, and weights `C·w` (`Coarsen_Matching.c:483-506` [U3 §3]).<br>A cell stops at ≤ 100 vertices or < 5 % shrink (`ometis.c:399-403`) and is then carried by identity | GraphBLAS: a few calls per level, for **all** cells at once. Per-call overhead does not scale with the cell count |
| 4 Initial separator per cell | 7 greedy-growing trials plus edge FM (`initpart.c:82-89`); the lighter of METIS's boundary with two-sided and one-sided FM (`separator.c:36-56`) and a König cover | Host, **serial** on the builder thread: the graph crate has no host pool, and adding one would compete with queries |
| 5 Uncoarsen | Project labels with `mxv(Cᵀ, labels)` per level, all cells at once; find the boundary with `mxv(A, side)`; run node FM **seeded from the boundary only** (feral's full rescan, `fm_refine.rs:350-378`, is the bug to avoid [B §5.2]), over a private Container unload [B C5] | Host FM, serial [R1 §1] |
| 6 Split | From the label vector, build the cell-membership matrix `M` (n × cells) with `GxB_Matrix_build_Scalar_Vector`. Separator vertices get no row. Then `A_{d+1}⟨A_d, struct⟩ = M ANY_PAIR Mᵀ`. Values are iso, so ANY is deterministic. Separators take the top ranks of their cell (`ometis.c:199-204`) | GraphBLAS: one masked dot per depth. `select` cannot read a vector (`GraphBLAS.h:6053-6062, 3089-3110`) |
| 7 Leaves ≤ 120 (`MMDSWITCH`, `defs.h:52`) | exact minimum degree on 128-bit bitsets; ties go to the lowest id | Host |
| 8 Tree, counts, relabel | One private extract `B = S(π,π)`. Then Liu's etree (C `cch.c:312-375`), postorder q, and **a host relabel of B's private arrays by q** (no second extract), then row and column counts [R3 §1] | Host over the unloaded `B` |

There is one implementation, not two: the synthesis's host-only multilevel ND for cells under 5,000 vertices is removed. If M1 shows that per-depth fixed cost dominates at the deep levels, cells below a floor run steps 4–5 directly on their own level without coarsening. That is the same routines, with the quality measured.

**Threads.**
- The builder engages a `GxB_Context` with `CCH_BUILD_THREADS` (default 1, D5) [U4 §4], so a build never takes every core from queries.
- `set_nthreads` is global (`matrix.rs:222-233` @main), and the context avoids it.

**PreJIT.** PR 1 harvests every user operator here (the positional key, `scramble`) in each sparsity variant it hits (`select_bitmap`, `phase1`, `phase2` …). A generic kernel costs +6–87 % [U4 §4].

**Expected quality.** Cited from FALKORDB.md:29-30 [U3 §4]:

| Ordering | Arcs | Triangles |
|---|---|---|
| C METIS | 5,527,199 | 48.6M |
| feral | 5,838,003 | 53.9M |

- The design adopts METIS's settings: MMD at 120, coarsening to ≤ 100, boundary FM, best-of-5 at the top depths and the `mindiff` tie-break. Those settings moved feral from 59.6M to 53.9M triangles (cited).
- Whether parallel mutual-max matching matches SHEM is **unmeasured** [U3 §3]. Feral is the gate floor, and C is the target (M1). The synthesis's "between METIS and feral" was not evidence and is withdrawn.

**Guards.**
- **(a) Separator size.** A top separator over 4,096 vertices fails the build early. CCH Lemma 2: that clique lies in ≥ n/3 search spaces [R1 §2].
- **(b) Arc count.** After step 8, the column counts give the exact arc count. Above `max_arcs` (default 8 per skeleton entry; California has 1.76, derived), the build fails before the game runs or the metric is allocated.

## 4. Supergraph construction

1. From step 8's relabelled private `B`, take `U0 = triu(B, 1)` on its arrays, plus the log pairs resolved at V_sg (§5).
2. Run today's child-absorption game (`order.rs:381-437`) in increasing rank: row(r) = U0(r,:) ∪ ⋃children (row(c) ∖ {r}). The column counts preallocate `Aj` exactly.
3. Assign ids row-major. These arrays are `Up.base`. Then `Down.base = GrB_transpose` of a read-only load, unloaded into Rust-owned arrays.

**Cost.** Derived and unmeasured (M2):
- O(|U0| + arcs) plus the row sorts.
- The game takes 97 ms on California today [U1 §5].
- The transpose and loads are expected to add tens of ms.

**Contract.** It meets I2, I4, I5 and I6 (the absorption fixpoint [R3 §1]), I7, I1′ and I3′. The order is a postorder, so at build time every elimination subtree is a contiguous rank range. S5 and G2 (both later) use this.

## 5. Batching on the native index builder

**Principle (derived).**
- Any order is valid. Only the supergraph must cover the published graph (I4), and only the metric must match it.
- So ordering slabs may come from different snapshots.
- The topology covers V_sg plus the log; the metric covers V_c plus the log.

**The log (rewritten after review).**
- **Writers record pairs.** `record()` already receives `(id, u, v)` for creates, deletes (including the node-delete cascade) and weight SETs (`maint.rs:297-315`). For both a `Building` and a `Built` entry, the writer appends `(min(u,v), max(u,v))` to the commit's segment.
- **Nothing is ever dropped.**
  - `set_structural` and the `PAIR_CAP` overflow no longer discard pairs (`maint.rs:121-137` do today).
  - A segment costs O(written), which the write already pays.
  - "Many" is a **decision**, not a loss: a front over the valve runs a full sweep.
- **Where the log lives.**
  - The log is in `CchIndex`'s own per-version state. CCH contributes no ids to the framework's `Written`, so `since_last_indexer_step` stays empty for CCH.
  - So `indexer_apply`'s clear (`versioned.rs:443-448`) and its `Arc::make_mut` copy at publish (`:453-461`) never touch it.
  - Segments carry the commit's data version as `seq` (`mvcc_graph.rs:247-248`).
- **Consumption** happens only off the lock, in the phase that resolves the pairs: Supergraph (pairs ≤ V_sg), Seal or Repair (pairs ≤ the fork). It records `consumed ≤ seq`.
- **Catch-up.**
  - It keeps the batch's list and prepends the published copy's segments with `seq > fork` (walked from the head). That is O(new segments) = O(written).
  - It never refuses, and it resolves nothing against an older version.
  - A `Same` or `Swap` publish carries the batch's copy, which already holds every segment before its fork, including writes in the gap between batches.
- **Writer rebase** appends the writer's own segments onto the newer published log.
- **Correctness.** A pair's state (edges, seeds) is a function of the current graph only. So resolving a pair at any later version is idempotent, and an id deleted or reused in any gap cannot hide its old pair. Both reviewers' counter-examples now log `(a,b)`.
- **Bound.** If the unconsumed log passes `max(1M pairs, arcs/8)`, the segments are dropped and the sticky `many` flag is set. `many` survives every publish and is consumed only by the phase that handles it:
  - **before V_sg:** the Supergraph phase re-reads the full skeleton at its fork;
  - **after V_sg:** the Seal or Repair finds new adjacencies with `triu(S_now(π,π),1)⟨¬struct(Up)⟩`, then runs the full seed pass and a full push sweep.

| Phase | Batch unit | State kept (copy-on-write) | Snapshot |
|---|---|---|---|
| ReadSkeleton | node-id slab, ~2M edges | `Vec<Arc<Matrix>>` slabs | any |
| Order | one depth; huge graphs split depth 0 by level | `Arc` level stack, label and rank vectors | none (private) |
| Supergraph | §4 over S ∪ log pairs at the fork V_sg | `Arc<Topology>` | V_sg; log ≤ V_sg consumed |
| Seed (V_c = V_sg) | the full-seed pass below, slab by slab | flat seed arrays, builder scratch | V_c |
| Relax | apex ranks [a, b), ~50M triangles per batch | flat metric, builder-owned | frozen at V_c |
| Seal | resolve the log against its fork V_s: §6 topology, then §6b recompute (unbounded, off the lock); then chunk the metric one array at a time (`WeightArr::from_vec`) | finished `Cch` | V_s |
| Repair | as Seal, from `base: Arc<Cch>` | `base` + log | its fork |

**The full-seed pass.** It does not build `effective`. Per indexed type and node slab:
- score `m⟨¬struct(dm)⟩` and `dp` with the weight index op (as `weight_ops.rs:372-385`, minus `effective_fwd`), MIN-reduced per pair and direction;
- pairs that have any entry in `dm` (multi-edge pairs that lost an edge) are reseeded pair by pair through `PairWeights`;
- permute to rank space with `W(π,π)`;
- align to `Up`'s pattern: `eWiseAdd(MIN, triu(W), iso-+∞ pattern of Up.base)⟨struct(Up.base)⟩`, read row-major, which is id order on a fresh lineage, straight into the seed arrays.

Its time and peak are unmeasured. The derived peak is one slab plus `W` in rank space (≈40 MB per direction at 3.3M entries) plus the seed arrays. M2 measures it.

**Publish.**
- **Same / Swap:** the batch's state, log included.
- **CatchUp:** the log merge above.
- **Adoption** (Seal or Repair): the new segments are absorbed under the lock by the §6/§6b writer path.
  - If that fits the time budget B_gil (§6b), the entry becomes `Built`.
  - Otherwise it becomes `Building{Repair, base = the batch's hierarchy, log = the unabsorbed segments}`.
- **Scheduling.** Repair batches go ahead of column batches on the builder queue. The default pool has one thread (`src/lib.rs:103`). A column batch already running still delays a Repair by up to its own length (unmeasured, M7).

**Convergence (derived, recalculated with mean costs).**
- A Repair absorbs k pairs in ≈ k × c(k).
  - c(512) ≈ 0.8 ms, c(1,024) ≈ 0.5 ms and c(4,096) ≈ 0.23 ms, from `maint.rs:82-94`. Cones overlap at large k.
  - For small k, c is at least ≈ 0.8 ms; the tail is up to tens of thousands of arcs per pair (cited).
- The backlog ratio per round is ≈ rate × c. So it shrinks only below ≈ 1,250 SET/s, and slowly near that rate:
  - at 1,000 SET/s the ratio is 0.8, so going from 410 pairs to the ≈30 that fit a 25 ms B_gil takes ≈ 12 rounds, ≈ 2 s of Repair;
  - at 100 SET/s it takes 1–2 rounds.
- `MAX_REDOS` (`cch/mod.rs:396`) counts only rounds whose backlog did not shrink. What happens past it is D2.

**Readers while Building.** New versions use the Dijkstra fallback: 222 ms p50 and 409 ms p95 on California (`fallback.rs:28-31` [U2]). Older versions keep their `Arc<Cch>`.

## 6. Incremental maintenance (built index, fixed order)

There is one flush per `Commit` segment, on the private version (`maint.rs:467-536`). The steps are:

**append segment → resolve pairs against the private version → classify → topology (S2, ranks) → recompute → publish.**

Structural decisions are pure functions of (held hierarchy, resolved pairs). The budget only decides where the work runs.

| Change | Rule | Correctness | Cost | Where |
|---|---|---|---|---|
| Weight SET/REMOVE; insert on a pair with an arc; edge delete | Metric only (S1). The seed is the pair's minimum valid weight per direction, or +∞ if none is left | G′ ⊆ H is unchanged [R2 S1, R3 §4] | p50 0.22 ms SET, 0.35 ms delete (#3363); mean ≳ 0.8 ms (derived); tail up to tens of thousands of arcs (`maint.rs:82-94`) | writer within B_gil |
| Node delete | Its edges cascade as logged pairs (`maint.rs:238-258`). Its rank and arcs stay; `deletions += deg` | as S1 | O(deg) + cones | writer |
| Insert (u,v) on a pair with no arc, both ranked | **S2** (Davis–Hager): add v to up(u) in a **host overlay** (rank → sorted additions), pop rows by rank reading merge(base, delta, overlay), push U(j)∖{p} into up(p), stop at fixpoint (DH01 Thm 4.2 [R2 §2]). Then sort the new arcs by (lo,hi), append ids, insert them into both deltas, and preset the metric to +∞ | Adds arcs only, and ends when the validity rule holds [U4 §5, R2 §5]: I2, I4–I6 | ancestor case: 6–22.5 new arcs on grids; across the top separator: 4–7 % of arcs [R3 §3]; roads unmeasured (M4) | writer if ≤ N_max = 4,096 new arcs and within B_gil, else Repair |
| First indexed edge of a rank-less node | Take the lowest rank in `free_ranks`; the holder becomes rank-less (valid: it has no arc, and I1′ allows it). If the pool is empty, take the next **top** rank `n_r' + 1`, which lives only in the deltas. Then S2 | Any rank is valid; an arc-less holder is read by no arc | O(1) + S2 (a top rank costs a union of root paths: 46/94/190 arcs on grids [R3 §3]) | writer |
| Node created, or id reused | Nothing until its first indexed edge. A reused id keeps its old rank and its arcs (at +∞) | — | 0 | — |
| Relationship types | Fixed by the definition (`spanning`, `maint.rs:317-325`) | — | — | — |

This replaces the synthesis's rank *swap*, which stripped the rank from a live isolated node. In Rust, every id below `n_r` is live by the time a larger id appears, because freed ids are reused in ascending order.

**Hand-off.**
- On abort (N_max, B_gil), the private topology and metric are dropped.
- The entry becomes `Building{Repair, base: last consistent Arc<Cch>, log: [this commit's full segment]}`. Later commits append segments.
- The Repair runs the same function, unbounded, and adopts as in §5.
- This replaces today's rebuild branches (`maint.rs:552, 576, 582, 621`).

**Decay.** Thresholds are derived; M6 calibrates them (C's valve: `cch_index.c:468-496` [U3 §1]).

| Signal | Action |
|---|---|
| deletions ≥ max(1,024, n/8), or dead ids ≥ 12.5 % | S6: a new lineage from the same order (game + full seed + push sweep, ≈ 97 + seed + 219 ms) |
| arcs ≥ 1.25 × arcs at build, one commit adding > arcs/8, or top ranks ≥ reserve | S7: re-order as a new generation |
| delta entries > max(64k, √\|base\|) | **fold job** (below) |

**Fold job.** A Built entry needs a builder job that keeps serving: a framework addition (D6).
- The job forks F and builds a new base from base ⊕ delta_F with GraphBLAS: eWiseAdd over read-only loads, then unload.
- At publish it catches up by rebuilding a fresh delta from only the rows in `touched_since_fold` with seq > F. That is O(written since F).
- Ids are unchanged, so the metric is untouched.

## 6b. Recomputing modified edges (goal 3)

**What an arc reads.** An arc a = (lo, hi), lo < hi, holds `min(seed(a), min over x ∈ down(lo) ∩ down(hi) of dn(x,lo) + up(x,hi))`, plus the mirror direction. Its middle is the first x to win a strict `<` in increasing x (`partial.rs:176-219`). So it changes only through:
- (i) its seed;
- (ii) a new apex, from a new arc (x,lo) or (x,hi);
- (iii) a bitwise change in a bottom arc;
- (iv) apex re-ranking.

**Propagation.** When (lo, hi) changes bitwise, enqueue (min(hi,w), max(hi,w)) for every w ∈ up(lo) ∖ {hi} (`partial.rs:241-254`).
- **Completeness.** An arc is a bottom arc only of triangles whose apex is its lower end.
- **Order.** Popping by increasing lower rank finalizes inputs first (I8).

**Lemma A (new arcs are born at +∞).** H ⊆ H′ are valid topologies, and N = H′ ∖ H is preset to (+∞, +∞, none, none).
- A new arc has no old apex: the apex would have forced the arc already, by I5.
- A triangle that is new to an old arc has a new bottom arc.
- A new arc that computes +∞ equals its preset: it neither propagates nor wins a tie. One that computes a finite value has changed bitwise and propagates.

So the front `N ∪ {changed seeds}` reaches every arc that differs, and the result is **bit-identical to a full customization of H′** (derived). Reviewer 1 checked this against `partial.rs` and `customize.rs`.

**Lemma B (S5, later).** S5 re-ranks an elimination subtree T within its rank set R_T. Its front is T's arcs plus the boundary clique K_T, and its input includes T's outward edges. It **deletes** delta and base entries, which the `cow_btree` delta supports (`mod.rs:20-23`). Base entries need a "replaced row" marker. That contract change ships in the S5 PR, which now starts a new topology version, not "dm stays empty".

**Recompute set per change** (fronts are node pairs, resolved to ids after all topology changes of the unit):

| Change | Front, then closed by propagation | Why it is complete |
|---|---|---|
| SET/REMOVE, insert on an existing arc, delete | that pair's arc, with its new seed | (i) |
| Node delete | the arcs of its deg(v) logged pairs. Its fill arcs reach +∞ through the cone | (i) + (iii) |
| S2 insert (new fill) | the inserted pairs' arcs ∪ N | Lemma A |
| Rank taken from the pool, or a new top rank | as S2 | an arc-less holder is read by nothing (I1′) |
| S5 (later) | T's arcs ∪ K_T | Lemma B |
| S6/S7 | everything: full seed + push sweep | new ids |
| Log at Seal/Repair (any gap, any id reuse) | the union of the rows above over all logged pairs, after all topology changes | pair state = f(current graph); §5 |
| `many` | full scan + full sweep | — |

**Algorithm.** `recustomize_seeded` (`partial.rs:98-258`) with these changes:
1. The front takes pairs.
2. The heap key is `(lo, hi, id)`.
3. Apex ids come from merging `Down(lo)` and `Down(hi)` (base slice plus delta cursor).
4. A propagated arc's id comes from a cursor walk of up(hi) alongside up(lo) for w > hi (by I5), and from a seek into up(w) for w < hi. That seek stays in the cost model.
5. Budget accounting is in **work units**: merge steps + pushes + seed lookups weighted by multiplicity, with a monotonic clock read every 256 pops. Aborting publishes nothing.

Seeds come pair by pair through `PairWeights` (≈0.65 µs of the 1.3 µs/arc; derived from "two graph lookups seed each", `maint.rs:85-87`).

Valves, off the lock only, switch to the full path (full seed + push):
- the front is over n/8 arcs (`partial.rs:131`);
- the cone passes the break-even of the *new* full path. The synthesis's 385k was derived from a 0.5 s path that built the weight matrix, and is withdrawn. It is now `(seed pass + 219 ms) / 1.3 µs`, recalibrated from M2.

**Cost on California.** Cited unless marked:

| Case | Cost |
|---|---|
| scoped, per arc (random cones) | ≈ 1.3 µs (`maint.rs:85-87`); for top-clique arcs unmeasured (M7) |
| one SET | 0.22 ms p50 (#3363) ≈ 170 arcs (derived); mean ≳ 0.8 ms ≈ 615 arcs (derived from 0.41 s / 512) |
| k random pairs | 0.41 s at 512; 0.51 s at 1,024; 0.95 s at 4,096 |
| 200×200 grid, one edge | 10.7 ms scoped against 464 ms full (#3773) |
| ancestor insert | ≈ 0.25–0.5 ms (derived from [R3 §3]) |
| insert across the top separator | ≈ 290k arcs ≈ 0.4 s if roads behave like grids (derived) → Repair |
| full push sweep | 219 ms (#3323) |
| today, under the lock | 0.47 s at a build commit (#3772), ≈ 0.5 s in `GRAPH.EFFECT` (#3363). Both are removed |

**Interleaving with a batched build.**

| When the edge changed | Topology | Metric |
|---|---|---|
| before its ReadSkeleton slab | in S | — |
| after its slab, before V_sg (in any batch or gap) | its logged pair joins U0 | seeded at V_c = V_sg |
| after V_c: during Seed or Relax, **or between two batches** (a `Same` publish keeps the segment) | S2 at the Seal, with new arcs at +∞ | **never patched mid-sweep**: the Seal's scoped pass recomputes from scratch |
| during the Seal or a Repair | segments with seq > fork, merged at CatchUp | absorbed at adoption within B_gil, otherwise the next Repair |
| after an inline abort, before the Repair forks | appended to the Repair's log | resolved at the Repair's fork |
| after adoption | §6 | §6 |

**Order.** Topology always goes first (S2/S5, new ids, deltas, metric arrays extended), then the recompute, then publish. The recompute never runs over a topology that lacks an arc: every pair with an edge and no arc is in the log, and S2 runs before the resolve. M5 adds a debug check that `pattern(S_now) ⊆ Up`. The synthesis's claim that `partial.rs` guards coverage is withdrawn: it checks only its front (`partial.rs:111-145`).

**Worked example (derived, mean costs).**
- California's build: ≈ 3.6 s Order, then Supergraph + Seed + Relax ≈ 0.5 s.
- At 100 SET/s, writes after V_c are ≈ 50 pairs. The Seal absorbs them off the lock in ≈ 40 ms; ≈ 4 pairs land during the Seal; adoption costs ≈ 3 ms under the lock.
- At 1,000 SET/s: 500 pairs, a Seal of 0.4 s, then ≈ 12 Repair rounds (§5).

**Where it runs, and the bound.**
- **Lock paths:**
  - the master writer's flush;
  - a replica's `GRAPH.EFFECT`;
  - adoption catch-up;
  - writer rebase.
- **Each lock path** gets:
  - S2 ≤ N_max = 4,096 new arcs;
  - recompute + S2 within B_gil. The default is 25 ms (D1), counted in work units plus the clock, and calibrated on the top clique, not on the average.
- Fixed costs are counted too: `cow_btree` inserts (O(new arcs × log n)) and the chunk copies.
- **A single SET on a top arc** can exceed B_gil and send new versions to the fallback for one Repair. M4 measures how often (the p99 and max cone). If more than 0.1 % of single SETs hand off, D1 is revisited.
- **Off the lock only:** the full-sweep valves, S5, S6, S7 and folds.

**MVCC.**
- Every write goes to a private version:
  - `WeightArr::set` copies its chunk (`customize.rs:182-189`);
  - `cow_btree` inserts path-copy;
  - the base is never written.
- `Cch{Arc<TopologyVersion>, Metric}` is swapped as one `Arc`, so a reader holds a consistent pair.
- An aborted attempt publishes `Building`, which never serves a hierarchy. No reader sees a half-recomputed metric or a topology missing an arc.

## 7. MVCC and replication

- **Versions.** A version holds `Built(Arc<Cch>)` or `Building(phase state, CchLog)`.
  - Bases are shared across a lineage.
  - Deltas, maps and metric chunks are shared until written.
- **Kept rules** [U2 §3]:
  - private writes (`cow.rs:48-59`);
  - every GraphBLAS object waited before publish (`mvcc_graph.rs:332-347`); read-only loads are unloaded or freed before publish;
  - replaced versions dropped outside the lock (`online_build.rs:348-360`).
- **Determinism: same distances only.** Ids and ranks depend on how commits are grouped and on fork points:
  - S2's id order;
  - which node takes which pool rank;
  - whether a Repair sees an insert that was deleted before its fork.

  The values are exact on every node. Equal-cost paths may differ (I9). With non-integer weights, differing hierarchies can differ in the last bit (derived). M5 compares by node pair, not by id. Identical paths would need the order and the topology events on the wire, which C lacks too [B §5.7].
- **RDB.** As today, only the definition is saved, and a load rebuilds (`cch/mod.rs:375-377`; C `cch_index.c:161-163`). Persisting `rank_of` (4 B/node) to skip ordering is D4.

## 8. Memory and time budget

California: 1.54M node capacity [U1 §1]; 5.84M arcs with feral, 5.53M with METIS. The task's 1.9M nodes scale per node.

| Item | Value | Basis |
|---|---|---|
| Topology, steady | 16 × 5.84M + 16 × 1.54M ≈ **118 MB**, against 142.7 MB today | derived; today's figure cited (#3321) |
| Delta at the fold trigger | ≤ 64k entries × 16 B × 2 ≈ 2 MB | derived from `cow_btree` ≤ 16 B |
| Metric | 187 MB (+ chunk headers; 396–412 MB process against 365, #3774) | derived / cited |
| Total against C | ≈ 305 MB against C's ≈ 364 MB | derived |
| Peak by phase, over the input | Order ≈ skeleton 20 + levels 60 ≈ 80 MB; Supergraph ≈ 150 MB; Seed ≈ 118 + 94 + ≤ 80 ≈ 290 MB; Relax/Seal ≈ 118 + 187 + 47 (one array converting) ≈ **352 MB** | derived, unmeasured. C's whole-build peak is unmeasured; M1/M2 compare against `:edge-c` |
| Ordering time | ≤ 3.62 s (feral today), target ≤ 3.07 s (C METIS) | cited (#3321) |
| Supergraph / full seed / customize | ≈ 100–150 ms / unmeasured / ≤ 219 ms | cited / derived |
| Per update | SET p50 0.22 ms, mean ≳ 0.8 ms; delete 0.35 ms; ancestor insert 0.25–0.5 ms | cited / derived |
| Lock hold | ≤ B_gil (25 ms default) + fixed costs, against 0.47 s today | derived / cited |
| Fallback window per hand-off | one Repair plus queueing behind a running column batch | unmeasured (M7) |

**9M edges, road-like:** ≈ 15.8M arcs and 4.2M nodes; topology ≈ 320 MB and metric ≈ 506 MB. Slab reads avoid the 400 MB / 76 ms `effective` transient [FOLD]. All derived.

**1500×1500 grid:** ≈ 100M arcs by extrapolating R3's 9.6 → 14.7 → 20.8 arcs/node [R3 §3]. Guard (b) refuses it before allocation (unmeasured).

## 9. Implementation plan

**Measurements first.** Each one includes a deliberate break, to prove it can fail.

| Id | Measure | Pass |
|---|---|---|
| M1 | ND prototype on California and a 9M-edge road graph, at 1/4/16 GraphBLAS threads | time ≤ 3.62 s (report against 3.07 s); peak ≤ 363 MiB over the input, or the gap documented; arcs ≤ 5.84M and triangles ≤ 53.9M (report the gap to 5.53M/48.6M); tree height ≤ feral's; **query p99 with the new order ≤ `:edge-c`'s**; identical order across thread counts |
| M2 | read paths with feral's order: GxB iterators against read-only Container slices; relax; scoped µs/arc; full-seed pass time and peak; valued against iso `Down` | query p99 ≤ `:edge-c` on roadNet-CA (and ≤ 526 µs, #3324); customize ≤ 219 ms; scoped ≤ 1.3 µs/arc on random pairs |
| M3 | 32-bit arrays at California size | 32-bit |
| M4 | cone and new-arc distributions (p50/p95/p99/max) for single SETs, deletes, and local/random/top-separator inserts | p99 single-SET cone fits B_gil; hand-off rate reported |
| M5 | randomized property test against a full customization (by pair) and Dijkstra. Cases: deletes and id reuse in the gap between batches with a `Same` publish; a delete between an inline abort and the Repair fork; a > 1,024-pair commit with a new adjacency; `many`; more than `reserve` isolated nodes, then connect an early one | 0 mismatches over 10k graphs; `pattern(S_now) ⊆ Up` |
| M6 | write-mix replay | sets the decay thresholds and B_gil |
| M7 | batched build under writes; top-separator SET | lock-hold p99 ≤ B_gil, max ≤ 2 × B_gil; **concurrent read p99 during a build ≤ 1.1 × idle**; fallback window p99 reported |
| M8 | Seal/Repair convergence at 100, 1,000 and 2,000 SET/s | adopts within 2 rounds at 100 and within 15 at 1,000; the non-converging rate reported, with the D2 backstop engaged |

**PRs.** Each is small, closes its own issue, and lists its functions in `COVERAGE.tsv`.
1. GraphBLAS wrappers: masked `mxv`/`mxm`, `extract_Vector`, `build_Scalar_Vector`, reduce, `Matrix<u32>`, read-only Container load/unload, `GxB_Context`, plus the PreJIT harvest of each user operator and variant.
2. `ChunkedArr<T>` with append; metric append with a +∞ preset.
3. Topology as base + `cow_btree` delta with I1′/I3′, still fed by feral, with the readers changed. Bit-identical metric; gates M2 and M3. **If M2 fails, stop and revisit the read path before PR 4.**
4. Writer-side pair log in `CchIndex` state, with no drops and the generalized recompute engine (work-unit budget). Proves M5's log cases.
5. S2 with the host overlay, lazy ranks, the `Building{Repair}` hand-off with priority scheduling, and B_gil on every lock path. Replaces both rebuild branches; proves M4 and M5.
6. GraphBLAS ND behind `CCH_ORDERING` (default feral), with guards. Proves M1.
7. Flip the default; delete feral and the `CchInput` CSR.
8. Phased batched build with the full-seed pass. Proves M7 and M8.
9. Decay, S6, S7 and the fold job (framework support for Built-entry jobs).
10. Later, behind data: S5, a shadow generation, G2, O2.

## 10. Risks and open questions

1. **Fallback windows.** A top-arc SET or about 30 random SETs in one commit hand off (derived). M4 and M7 decide whether 25 ms is right (D1).
2. **Write rates above ≈ 1,250 SET/s** never converge (derived). D2.
3. **Matching quality** against SHEM is unmeasured [U3 §3]. If M1 fails, keep feral behind PR 3's API; the CSR still leaves the supergraph and customization.
4. **Read-only Container semantics.** Lifetime and wait rules for `GxB_IS_READONLY` arrays under concurrent read-only loads need PR 1 tests.
5. **Fold jobs on Built entries** need a framework addition (D6).
6. **Cross-separator inserts on roads** are unmeasured. If frequent, S5 moves earlier.
7. **Two hierarchies alive** during S6/S7: ≈ 2 × 0.3 GB on California (derived).
8. **Proofs.** Lemmas A/B, log completeness and lazy ranks rest on M5 until the Lean coverage lands.
9. **Serial host FM.** The ordering time could exceed 3.62 s with one thread. M1 decides whether D5 needs more threads.

## Decisions the team must make

- **D1. Lock budget B_gil.**
  - Options: (a) one time budget for every lock path; (b) a larger budget for the master writer than for replicas; (c) count arcs, as in the synthesis.
  - **Recommend (a): 25 ms, a config, in work units plus the clock**, calibrated by M4/M7.
- **D2. Above the convergence rate.**
  - Options: (a) after `MAX_REDOS` non-shrinking rounds, force one adoption over budget (bounded by the full path, ≈ 0.4 s, like today's 0.47 s); (b) stay on the fallback and report "repairing" in `db.indexes()`; (c) throttle writers.
  - **Recommend (a) plus the status report in (b).**
- **D3. The read representation for goal 1.**
  - Options: (a) GxB row iterators only; (b) Rust-owned arrays loaded read-only into GraphBLAS for algebra and read as slices.
  - **Recommend (b) unless M2 shows that (a) meets the p99 and 219 ms gates.**
- **D4. Persist `rank_of` in the RDB** to skip the 3.62 s ordering on load.
  - Options: persist it, or rebuild.
  - **Recommend rebuild first; persist only if load time matters.**
- **D5. Builder GraphBLAS threads.**
  - Options: 1, a small default, or all cores.
  - **Recommend 1 (a config)**, raised only if M1 and M7 show query p99 holds.
- **D6. Folds on a Built index.**
  - Options: (a) a framework job type for Built entries; (b) fold only inside Repair/S6.
  - **Recommend (a).** Under (b), a quiet index's delta never shrinks.
- **D7. Valued against iso `Down`.**
  - Options: 48 B/arc with no apex searches, against 44 B/arc with searches.
  - **Recommend valued**, unless M2 shows that memory decides.
- **D8. The replica promise.**
  - Options: equal distances, or equal paths (needs a new effects format).
  - **Recommend equal distances.**

## Appendix: Review issues and dispositions

R1 is review 1, R2 review 2 and R3 review 3.

| # | Severity | Issue | Disposition | Change made |
|---|---|---|---|---|
| R1.1 | fatal | Writes between batches are lost: `indexer_apply` clears the id set, `Same` skips catch-up, and the "before" pair is read at a fork that already contains the gap | accepted | Pairs are logged at write time in CCH's own state (§5); the F/P resolution is deleted; M5 gap cases added |
| R1.2 | major | The `Arc` hash-set log is cloned on every append | accepted | Persistent segment list, deduped off the lock (§2, §5) |
| R1.3 | major | `set_structural`/`PAIR_CAP` drop the pairs S2 needs; `many` is not sticky | accepted | Pairs are never dropped; `many` is sticky and consumed by the phase that handles it (§5) |
| R1.4 | major | The rank swap strips the rank from a live isolated node | accepted | Lazy `Option` ranks, a `free_ranks` pool, top ranks in the delta; I1′ (§2, §6) |
| R1.5 | major | `VersionedMatrix` folds on the writer; `set` exists only for bool | accepted | Base + `cow_btree` delta with no automatic fold; fold job (§2, §6) |
| R1.6 | minor | Bit-identical hierarchies across paths is false for ids | accepted | Claim weakened to distances; S2 ids sorted; M5 compares by pair (§7) |
| R2.1 | fatal | Same gap loss, including inline abort → Repair fork | accepted | As R1.1, with an abort case in M5 |
| R2.2 | major | Sizing used the p50 (0.22 ms), not the mean (≈ 0.8 ms) | accepted | Mean and tail used; convergence ≈ 1,250 SET/s; example redone; M4 distributions (§5, §6b) |
| R2.3 | major | Counting arcs does not bound lock time near the top | accepted | Work units plus the clock, calibrated on the top clique (§6b, M7) |
| R2.4 | major | `PAIR_CAP` overflow loses structural pairs; `partial.rs` does not guard coverage | accepted | No drops; the coverage claim is withdrawn; `S ⊆ Up` debug check (§5, §6b) |
| R2.5 | minor | Determinism depends on fork timing | accepted | As R1.6 |
| R2.6 | minor | S5 deletes entries, contradicting "dm stays empty" | accepted | The `cow_btree` delta supports deletes; S5 ships its contract change (§6b) |
| R2.7 | minor | Copying `dp` is O(\|dp\|) under the lock | accepted | `cow_btree` path copying is O(log n) per insert (§2) |
| R3.1 | major | Full-seed path unspecified; the 385k valve rests on the deleted weight matrix | accepted | Masked per-slab scoring with no `effective` (§5); valve rederived (§6b) |
| R3.2 | major | GraphBLAS rows break relax and query costs | accepted | Read-only Container arrays read as slices; M2 compares, with a stop gate (§2, D3) |
| R3.3 | major | R_gil is crossed by ≈ 16 SETs, giving long fallback windows | accepted | Time budget (D1), priority Repair, the window measured in M7 |
| R3.4 | major | Convergence used p50; M8 would fail | accepted | Recomputed with c(k); M8 thresholds revised; D2 backstop |
| R3.5 | major | M1/M2 gates accepted regressions against C | accepted | Gates tightened against feral/C, plus tree height and p99 with the new order (§9) |
| R3.6 | major | No host pool; global GraphBLAS threads contend with queries | accepted | Serial host loops, `GxB_Context` (D5), a read-p99 check in M7 |
| R3.7 | major | S2 writing `dp` pays pending-tuple assembly | accepted | Host overlay plus `cow_btree` delta; no GraphBLAS writes on the lock (§6) |
| R3.8 | minor | The metric transient was counted twice | accepted | One array at a time; peak given by phase (§8) |
| R3.9 | minor | `select` cannot read a label vector | accepted | Masked `M ANY_PAIR Mᵀ` split; PreJIT per variant (§3) |
| R3.10 | minor | Propagation still needs an id lookup | accepted | Cursor walk for w > hi, a seek for w < hi, both in the cost model (§6b) |
| R3.11 | minor | Same as R1.2 | accepted | As R1.2 |
| R3.12 | minor | A resize past the reserve copies the shared base | accepted | Top ranks live in the delta; no resize (§2, §6) |
| R3.13 | minor | Two ND implementations; `S(π,π)` extracted twice | accepted | One level-synchronous implementation; one extract plus a host relabel (§3) |
