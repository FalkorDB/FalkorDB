# CCH ordering, supergraph and incremental upkeep on GraphBLAS: design, revision 3

This document revises `02-final-design.md` against three adversarial re-reviews (round 2) of its write log, rank allocation and Built-entry jobs. Each of their 23 issues was triaged against the code at @08c. All 23 are confirmed. For seven of them the fix proposed by the reviewer was replaced by a simpler or cheaper one. Appendix B gives every disposition. Round 1's dispositions stay in Appendix A.

No repo file was modified, and nothing was measured for this design.

**What changed from 02, in one list.**
- The log's overflow is a **log node with a seq**, not a side flag (§5).
- Consumed segments are freed. A Built entry keeps no log unless one of its upkeep jobs is in flight (§5, §7).
- Writer rebase and batch catch-up read the log from the entries, not from `Written`. Rebase is defined for every pair of fork and published states (§5).
- Built-entry jobs (fold, S6) get a ticket, a catch-up and a writer rebase. A writer whose fork predates their publish no longer reverts them (§6, D6).
- **Pool ranks have no holder.** Isolated nodes and unused ids get no rank, so taking a pool rank never strips anyone (§2, §3).
- **Rank allocation is its own phase**, run before classification by every resolver: writer flush, Supergraph, Seal, Repair, the `many` paths and S6 (§6).
- New nodes take a pool rank only at degree ≤ 1, and a top rank otherwise (§6).
- The full-seed pass scores `m⟨¬dm ∧ ¬dp⟩ ∪ dp` with `me` merged, seeds both directions, and writes seeds **by arc id** (§5).
- An inline hand-off takes a new generation. Its base is the private hierarchy at the abort (§6).
- B_gil is one budget per lock hold, shared by every index (§6b).

**Revision 3.1** (2026-11-12, agreed with Dvir):
- The skeleton read masks pending deletions per slab, as `m⟨¬dm⟩ ∪ dp` (§3).
- Supergraph drops logged pairs that have no live edge at V_sg, as well as adding the live ones (§4).
- So the supergraph is the exact live pattern at V_sg, and no dead arc survives the build.

The prototype amendments in `../prototypes/00-results.md` ("Change") are not merged here yet.

**Sources.**
- [B] is the research brief, `../research/00-research-brief.md`.
- [U1]–[U4] and [R1]–[R3] are the source reports.
- [FOLD] is `../../cch-background-fold/discussion.md`.
- Stack code was read at `origin/cch/08c-build-catch-up` 31ca1b802 ("@08c"). That is the stack tip, and it contains #3773's `partial.rs`, #3774's `WeightArr`, #3363's `maint.rs` and #3772's `absorb`.
- File:line facts were re-checked at @08c for this revision:
  - `versioned.rs:86-94` (`Written`), `:423-453` (`indexer_apply`: refuses a non-`Building` entry, clears `since_last_indexer_step`, sets `Ready` on `next = None`), `:457-464` (`publish`);
  - `versioned.rs:551-584` (`fit`: `CatchUp` whenever the data version moved; `Same`/`Swap` only when it did not), `:677-706` (`catch_up_index` starts from the batch's copy, `caught = ours.clone()`; `Restart` keeps the generation);
  - `versioned.rs:749-765` (`overtaken` requires `ours.is_building()`), `:768-810` (`rebase_index` starts from the published copy, `rebased = theirs.clone()`, carries only `by_this_version`; `Restart` takes `next_generation()`), `:910-922` (`building_since`: a job is enqueued only for a new generation);
  - `mvcc_graph.rs:155-160` (`is_stale`), `:205-223` (`commit` publishes the fork and enqueues new generations), `:247-248, 293-304` (only a writer bumps the data version);
  - `cch/mod.rs:517-539` (`writer_rebase` returns at once on an empty `written` or a `Building` published copy), `:549-573` (`indexer_catch_up` returns at once on an empty `written`), `:630-634` (`memory_usage` counts only the `Cch`);
  - `maint.rs:8-10` (one flush per `Commit` segment), `:121-138` (`set_structural`, `push` drop pairs), `:238-253` (node-delete cascade logged per edge), `:297-314` (`record`), `:467-536` (`flush` loops over every held index), `:551-584` (rebuild branches);
  - `falkordb_index.rs:1286-1296` (`rebuild_cch_in_background` takes `next_generation()`);
  - `online_build.rs:373-375, 447-457` (`standing`: a job continues only while its generation is `Building`);
  - `tensor.rs:52-108` (`dp` shadows `m`; `dm` marks pure deletions; states H and I), `weight_ops.rs:375-425` (`effective_fwd`, `multi_edges.effective()`);
  - `customize.rs:243-248` (`Metric` holds upward and downward arrays), `partial.rs:111-145` (only the front is checked; the `n/8` valve at `:131`);
  - `graph/src/runtime/ops/commit.rs:85-103` (`pending.commit` per segment, then `upgrade_to_write`).
- Round 1's checks still stand: `graphblas/versioned_matrix.rs:206, 241, 889, 1114, 1162, 1367`; `GraphBLAS.h:6977-6990`; `cow_btree/mod.rs:1-30` @main; `src/lib.rs:103` @main; no `rayon` on `origin/main`.

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
   - rank allocation for the endpoints of logged pairs that have no rank;
   - Liu's elimination tree, a postorder relabel, and Gilbert–Ng–Peyton counts. The counts give the exact arc count, which serves as the size guard;
   - today's elimination game, over one private extract.
   - The output is a rank-space topology with **arc ids as values**.
3. **Topology representation.**
   - Each of `Up` and `Down` is a **base plus a delta**.
   - The base is a GraphBLAS matrix whose arrays are owned by Rust in an `Arc`. GraphBLAS reads it as read-only (`GxB_IS_READONLY`), so the query and customization paths read the very same memory as flat slices.
   - The delta is a `cow_btree` keyed by `(row, column)`, whose value is the arc id. Writes path-copy, and it never folds by itself.
4. **Arc identity.**
   - An arc's id is its stored value.
   - On a fresh build, ids are dense and row-major, so #3774's layout is unchanged.
   - Within a lineage, ids are only ever appended, in `(lo, hi)` order within each pass.
   - Every whole-array write (the full-seed pass, a fold) addresses arcs **by id**, never by position.
5. **The build log.**
   - Writers record **node pairs at write time**, into a persistent append-only log of per-commit nodes, each carrying its commit's data version as `seq`.
   - A node is either a pair segment or an **Overflow** marker that stands for dropped segments. Both follow the same seq rules.
   - The log lives in the CCH index's own state, not in the framework's `Written` id set. Every catch-up and rebase reads it from the entries.
   - A consumer frees what it consumed. A Built entry keeps no log except while one of its upkeep jobs is in flight.
6. **Batched build.** The phases are ReadSkeleton → Order → Supergraph → Seed → Relax → Seal → (Repair).
   - The topology covers one snapshot plus the log.
   - The metric is a push sweep from one seed snapshot plus the log, and it is never patched mid-sweep.
7. **Incremental upkeep.** The order stays fixed.
   - Deletes reseed the arc to +∞.
   - Inserts on pairs with no arc get a closure repair (Davis–Hager).
   - Ranks are lazy and allocated before classification. A node with no rank gets one when it first has an indexed edge: a holderless bottom rank from the pool if its degree is ≤ 1, otherwise a new top rank that lives only in the delta.
   - Decay runs a fold or a new lineage (S6) as a Built-entry job that keeps serving, or a new generation (S7).
8. **Recompute (goal 3).**
   - First the topology changes, then one scoped recustomization runs in rank order (#3773's engine, generalized). It is bit-identical to a full customization of the new topology.
   - Each lock hold has **one time budget** for all its indexes, checked by the clock. Past the budget, a priority Repair batch takes over.
   - Readers swap one `Arc<Cch>` holding the (topology, metric) pair.

**What it gives up:**
- Order quality targets METIS, with feral as the floor. Flow refinement comes later.
- New versions answer with the Dijkstra fallback (222 ms p50, cited) while a hand-off is pending or an S7 rebuild runs. A single SET on a top arc can trigger a hand-off (§6b).
- Above a write rate of roughly 1,250 SET/s (derived), a Repair never adopts within budget. The backstop is a team decision (D2).
- Replicas promise equal distances, not equal hierarchies or paths.

## 2. Data structures

**Rank space.** At build, `n_r = ranked + reserve`, where `ranked` counts the node ids with an indexed edge in the skeleton or the log at V_sg, and `reserve = max(1,024, node_cap/64)` (derived). Top ranks raise it to `n_r'`. Ranks are `NonZeroU32` (4 B, with the `Option` niche) [B §1].

| Structure | Type | Versioning | Bytes |
|---|---|---|---|
| `Up.base` | `RankMatrix`: CSR arrays `Ap: Arc<[u32]>`, `Aj: Arc<[u32]>`, `Ax: Arc<[u32]>` (value = arc id), loaded into a `GrB_Matrix` with `GxB_IS_READONLY` whenever a GraphBLAS op needs it (`GraphBLAS.h:6977-6990`) | Immutable for its lineage. Shared by `Arc`. Replaced only by a fold | 8 per arc; 4 per rank |
| `Down.base` | the same, holding `Upᵀ` with the same ids | same | 8 per arc; 4 per rank |
| `Up.delta`, `Down.delta` | `CowBTree`: key `(row ≪ 32) \| col`, doc = id (`cow_btree/mod.rs:1-30` @main) | `clone` is an O(1) root bump. Each insert path-copies O(log n) pages. **No automatic fold** | ≤ 16 per delta entry (cited); bounded by the fold trigger |
| `rank_of` | `ChunkedArr<Option<Rank>>` over node ids: #3774's `WeightArr` chunks plus append | chunk copy-on-write (`customize.rs:182-189` @08c) | 4 per node |
| `node_of` | `ChunkedArr<Option<NodeId>>` over ranks, `None` for a pool rank (u32 ids; u64 once `node_cap` ≥ 2³²) | same | 4 per rank |
| `free_ranks` | `Arc<RoaringBitmap>`: the **holderless** ranks, exactly `{r : node_of[r] = None}` | copied when changed. Starts as the lowest `reserve` ranks | small |
| Metric (4 arrays) | indexed by arc id, length `arc_capacity`. **Format out of scope** | #3774's chunks, plus append | 32 per arc |
| `LineageStats` | counters: deletions, dead ids, arcs at build, top ranks used, **arcs added by S2 at pool-ranked rows** | copied | O(1) |
| `added_since_fold` | persistent list of `(seq, Arc<[(lo, hi, id)]>)`, the arcs S2 added since the last fold | O(1) append; cleared by a fold | 16 per delta entry, so bounded by the fold trigger |
| CCH log | `CchLog { head: Option<Arc<LogNode>>, pairs: u64 }`, `LogNode { seq, kind: Pairs(Arc<[(u64,u64)]>) \| Overflow, prev }` | persistent list; append and relink are O(1) per node, and no pairs are copied (§5) | 16 per retained pair + ≈40 per node |
| `upkeep` (Built only) | `Option<Upkeep { kind: Fold \| S6, ticket: u64, since: seq }>` | copied | O(1) |

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
  - The pass appends its arcs to `added_since_fold`.
- **No reuse.** Ids are never reused within a lineage. Older readers have shorter arrays and older deltas, so they never see new ids.
- **No positional writes.** After the first S2 of a lineage, row-major position is no longer id order. Every whole-array writer (the full-seed pass, a fold, S6) reads each entry's id from `Ax` or the delta doc and writes `array[id]`.
- **New lineages.** S6 and S7 start a new lineage with dense ids. Dead ids hold (+∞, none) until then.

**Contract changes, and every reader they touch.**
- **I1′.** `rank_of` is partial and injective. A rank-less node has no indexed edge at that version.
  - Readers: the query's endpoint lookup, which answers "no path" unless source = target; and `partial.rs:115-118`, which must check for the rank's presence as well as the bound.
- **P (pool).** `r ∈ free_ranks ⇔ node_of[r] = None`, and then `up(r) = down(r) = ∅` (base and delta merged). Taking a pool rank never strips a holder. Within a lineage a rank never goes back to the pool (deleted nodes keep their rank, §6); only S6 and S7 rebuild the pool.
- **I3′.** `up_ids(r)` yields ascending `(hi, id)` pairs from merging base and delta. Arrays are sized by `arc_capacity`.
  - Readers:
    - relax's `x_slots_start + i` (`customize.rs:497-513`);
    - `query.rs:113`;
    - `partial.rs:151,170` (the heap carries `(lo, hi, id)`);
    - `Cch::joins`;
    - the full-seed pass and the fold (write by id).
- **I7** is strengthened: the ids are equal. **I2, I4, I5, I6, I8 and I9** are unchanged [U1 §2].
- M5's debug checks: `pattern(S_now) ⊆ Up`, P, and "every node with an indexed edge has a rank".
- Every changed function gets a line in `proofs/*/COVERAGE.tsv`.

**Against C** (48 B/arc, 64 B/node plus row headers [U1 §1]).
- Per arc: 8 + 8 + 32 = **48 B**, or 44 with an iso `Down` (D7).
- Per node: 4 (`rank_of`) + 4 (`node_of`) + 8 (two row-pointer arrays) = **16 B**.
- The delta and `added_since_fold` add ≤ 32 B per *delta* entry, bounded by the fold trigger.
- The log holds 16 B per pair only while it is unconsumed (Building) or retained for an in-flight upkeep job (Built), bounded in both cases (§5). It is never a per-entity cost at steady state.
- This holds while the arrays are 32-bit (M3).

## 3. Ordering algorithm

**Input skeleton**, read in row slabs by node id (§5):
- per indexed type, take the live pattern of the slab's rows, `pattern(m)⟨¬struct(dm)⟩ ∪ pattern(dp)`: the same `dm` mask the full-seed pass applies (§5);
- take the union across types;
- finally `S = select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ))` (`input.rs:112-113` @05; C `cch.c:181-185` [U3 §3]).

Masking `dm` per slab gives the live pattern at the slab's version.
- **Why not a superset.** A superset (ignoring `dm`) would also be valid for any order and for I4 [R2 §1]. But it leaves arcs and fill for dead edges for the life of the lineage, uncounted by the S6 trigger. Nothing keeps `dm` small: a fold is forced only when `2·|dm| ≥ |m|` (`delta_dominates_base`, `versioned_matrix.rs:265` @08c).
- **Cost.** The mask still avoids the whole-matrix `effective` merge: ~400 MB extra and ~15 ms of merge at 9M edges [FOLD]. It costs one complement mask per ~2M-edge slab (unmeasured; M1).
- **Values.** A stale `m` *value* is harmless here, because only the pattern is read. The full-seed pass in §5 does read values, and must not ignore `dp`.

Revision 3.1, agreed with Dvir on 2026-11-12. With §4 step 2's drop, B′ is the exact live pattern at V_sg.

| Step | Operation | Where, and why |
|---|---|---|
| 0 Degrees | `d = reduce_rows(PLUS, S)`. Degree-0 vertices and unused ids get **no rank** (I1′). The lowest `reserve` ranks are created holderless and form `free_ranks`; ranked vertices take the ranks above them | GraphBLAS: one call |
| 1 Hanging trees | Degree-1 peeling: masked `mxv` (PLUS_PAIR) over the frontier; ranks in (round, id) order; a host pass after 64 rounds [U4 §1, R1 §1] | GraphBLAS; host for long chains |
| 2 Components | `LAGr_ConnectedComponents` on a private copy (it unloads `G->A`; `FastSV7.c:42-46` [U4 §1]) | GraphBLAS |
| 3 Coarsen all cells of depth d | **Match** (Manne–Bisseling [R1 §3]):<br>- keys `K = apply(A_d, positional op: (w ≪ 32) \| scramble(j))`, which are unique, so MAX is deterministic;<br>- `y⟨unmatched⟩ = K MAX_FIRST unmatched`;<br>- mutual pairs via `GxB_Vector_extract_Vector`;<br>- a 2-hop pass when the shrink is under 15 % [U3 §2].<br>A mask drops candidate pairs whose combined vertex weight exceeds a cap modelled on METIS's `maxvwgt`. The exact METIS formula is not cited here; PR 6 verifies it against the METIS source.<br>**Contract:** `A_c = select(OFFDIAG, C·A_d·Cᵀ)` with PLUS_TIMES_INT32, and weights `C·w` (`Coarsen_Matching.c:483-506` [U3 §3]).<br>A cell stops at ≤ 100 vertices or < 5 % shrink (`ometis.c:399-403`) and is then carried by identity | GraphBLAS: a few calls per level, for **all** cells at once. Per-call overhead does not scale with the cell count |
| 4 Initial separator per cell | 7 greedy-growing trials plus edge FM (`initpart.c:82-89`); the lighter of METIS's boundary with two-sided and one-sided FM (`separator.c:36-56`) and a König cover | Host, **serial** on the builder thread: the graph crate has no host pool, and adding one would compete with queries |
| 5 Uncoarsen | Project labels with `mxv(Cᵀ, labels)` per level, all cells at once; find the boundary with `mxv(A, side)`; run node FM **seeded from the boundary only** (feral's full rescan, `fm_refine.rs:350-378`, is the bug to avoid [B §5.2]), over a private Container unload [B C5] | Host FM, serial [R1 §1] |
| 6 Split | From the label vector, build the cell-membership matrix `M` (n × cells) with `GxB_Matrix_build_Scalar_Vector`. Separator vertices get no row. Then `A_{d+1}⟨A_d, struct⟩ = M ANY_PAIR Mᵀ`. Values are iso, so ANY is deterministic. Separators take the top ranks of their cell (`ometis.c:199-204`) | GraphBLAS: one masked dot per depth. `select` cannot read a vector (`GraphBLAS.h:6053-6062, 3089-3110`) |
| 7 Leaves ≤ 120 (`MMDSWITCH`, `defs.h:52`) | exact minimum degree on 128-bit bitsets; ties go to the lowest id | Host |
| 8 Tree, counts, relabel | One private extract `B = S(π,π)`. Then Liu's etree (C `cch.c:312-375`), postorder q, and **a host relabel of B's private arrays by q** (no second extract), then row and column counts [R3 §1] | Host over the unloaded `B` |

The rank-space dimension leaves the pool at the bottom: π maps the ranked vertices to `reserve+1 ..= n_r`.

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
- Whether parallel mutual-max matching matches SHEM is **unmeasured** [U3 §3]. Feral is the gate floor, and C is the target (M1).

**Guards.**
- **(a) Separator size.** A top separator over 4,096 vertices fails the build early. CCH Lemma 2: that clique lies in ≥ n/3 search spaces [R1 §2].
- **(b) Arc count.** After step 8 (and again in §4 if the log adds pairs), the column counts give the exact arc count. Above `max_arcs` (default 8 per skeleton entry; California has 1.76, derived), the build fails before the game runs or the metric is allocated.

## 4. Supergraph construction

1. **Resolve and rank.** The Supergraph batch forks at V_sg and resolves every log node with `seq ≤ V_sg` against it (an Overflow node: §5 "Bound"). It runs the §6 **rank allocation** over the resolved pairs: every rank-less endpoint with a live indexed edge at V_sg gets a pool rank (degree ≤ 1) or a new top rank `n_r+1, …` (degree ≥ 2). At build time a top rank simply raises the dimension to `n_r'`; nothing lives in a delta yet. π is extended with the new ranks.
2. **Update B.** `B' = (B ∖ D) ∪ A`:
   - A holds `(π(u), π(v))` and `(π(v), π(u))` for every resolved pair with a live indexed edge at V_sg, in any type and either direction.
   - D holds both orientations of every resolved pair with none.

   **Why the drop is valid.** Every pair written after its slab was read is in the log, and is resolved here. So B′ is the exact live pattern at V_sg. A ranked node that loses all its edges keeps its rank and gets no arcs. This step is Revision 3.1, agreed with Dvir on 2026-11-12.

   If B' ≠ B, the etree and the column counts are recomputed over B' (near-linear, unmeasured), and guard (b) is checked again. The postorder relabel is not redone: contiguous subtrees then hold only for ranks Order assigned, and S5 (later) checks contiguity before relying on it.
3. Take `U0 = triu(B', 1)` and run today's child-absorption game (`order.rs:381-437`) in increasing rank: row(r) = U0(r,:) ∪ ⋃children (row(c) ∖ {r}). The column counts preallocate `Aj` exactly.
4. Assign ids row-major. These arrays are `Up.base`. Then `Down.base = GrB_transpose` of a read-only load, unloaded into Rust-owned arrays.
5. `free_ranks` = the pool ranks not allocated in step 1. P holds by construction: a holderless rank has no row in B'.

**Cost.** Derived and unmeasured (M2):
- O(|U0| + arcs) plus the row sorts.
- The game takes 97 ms on California today [U1 §5].
- The transpose and loads are expected to add tens of ms.

**Contract.** It meets I2, I4, I5 and I6 (the absorption fixpoint [R3 §1]), I7, I1′, P and I3′ at V_sg: every node with an indexed edge at V_sg has a rank, and every such edge has an arc.

## 5. Batching on the native index builder

**Principle (derived).**
- Any order is valid. Only the supergraph must cover the published graph (I4), and only the metric must match it.
- So ordering slabs may come from different snapshots.
- The topology covers V_sg plus the log; the metric covers V_c plus the log.

**The log.**
- **Writers record pairs.** `record()` already receives `(id, u, v)` for creates, deletes (including the node-delete cascade, `maint.rs:238-253`) and weight SETs (`maint.rs:297-314`). The writer adds `(min(u,v), max(u,v))` to the commit segment's pair set. Self-loops are dropped, as at `maint.rs:131`.
- **Nothing is dropped silently.** `set_structural` and the `PAIR_CAP` overflow no longer discard pairs (`maint.rs:121-138` do today). Past the bound, the dropped segments are replaced by an Overflow node, which every consumer treats as "rescan" (below).
- **Where the log lives.**
  - The log is in `CchIndex`'s own per-version state. CCH contributes no ids to the framework's `Written`, so `since_last_indexer_step` stays empty for CCH, and `indexer_apply`'s clear (`versioned.rs:446-448`) never touches it.
  - Each node carries the commit's data version as `seq`: a writer's private version is `published.version + 1` (`mvcc_graph.rs:301`), and batch publishes keep the version (`:304`). So a node's seq is its writer's version, and no batch adds a node. Several segments of one query share a seq.
- **Nodes are cheap to relink.** A node's pairs are an `Arc<[(u64,u64)]>`. Re-creating a node over another `prev` (in a CatchUp or a rebase) copies ≈40 B, never the pairs.

**Consumption.** Only the phase that resolves the pairs consumes them, off the lock: Supergraph (nodes ≤ V_sg), Seal or Repair (nodes ≤ its fork). The consumer's own copy sets `head = None` and `pairs = 0`. Nothing is marked; consumed nodes are freed when the last version that holds them drops.

**Publish rules for a Building entry.** The fork of a batch is `batch_version.version`. Every rule is by seq.

| Publish | Fit (`versioned.rs:564-583`) | Log rule |
|---|---|---|
| Same / Swap | no writer committed since the fork | the batch's copy, which holds every node of its fork (including writes in the gap between batches), minus what it consumed |
| CatchUp | a writer committed | the batch's copy, with the published copy's nodes of `seq > fork` relinked on top (walked from the head, stopping at the first `seq ≤ fork`). O(new nodes). `pairs` = the batch's remainder plus theirs |
| Adoption (Seal/Repair, a CatchUp or Same with `next = None`) | as above | then the new nodes are absorbed under the lock by the §6 path within the remaining B_gil. If that fits, the entry becomes `Built` with no log. Otherwise `Absorbed::Restart` keeps the generation and the running job (`versioned.rs:695-702`; `online_build.rs:447-457`): `Building{Repair, base = the batch's hierarchy, log = the relinked nodes}` |

This replaces `CchIndex::indexer_catch_up`'s early return on an empty `written` (`cch/mod.rs:557`). The hook reads both entries' logs; PR 4 changes it.

**Writer rebase, by published state.** A writer that forked before a batch published and escalates after it is `overtaken` (`versioned.rs:749-765`). The framework today replaces the fork's copy with the published one and carries only `by_this_version` (`:786, :805`), which is empty for CCH, and `CchIndex::writer_rebase` returns at once on an empty set (`cch/mod.rs:525`). So the writer's own nodes would be lost. PR 4 changes the hook to receive the fork's entry (`ours`), and CCH rebases by the published state:

| Fork's entry | Published entry | Rebase |
|---|---|---|
| Building | Building (same generation) | Relink the fork's own nodes (`seq = fork.version`, above every published seq) onto the published log. An own Overflow is relinked like a segment |
| Building | Built (the Seal or Repair adopted) | Run the §6 flush over the own nodes, resolved against the fork graph, onto the published hierarchy, within the remaining B_gil. Over budget, or an own Overflow: `Absorbed::Restart`, which takes `next_generation()` (`versioned.rs:792-795`), with `Building{Repair, base = the published hierarchy, log = the own nodes}` |
| Built | Built, upkeep job published (D6) | Fold: re-apply the fold to the fork's copy (§6 "Fold job"). S6: replay the own nodes through §6 onto the new lineage, within B_gil; over budget, `Restart` as above |
| Built | Building, same generation | Unreachable: a Built entry becomes Building only through a writer's commit (a hand-off or S7, both new generations), which makes this fork stale (`mvcc_graph.rs:155-160`) |

A writer always keeps its own nodes in its private version until it publishes, whatever the entry state, so every row above has them. At its publish, a Built entry with no upkeep job drops its log (§7).

**Correctness.** A pair's state (edges, seeds) is a function of the current graph only: `PairWeights` re-reads every indexed type and parallel edge per direction. So resolving a pair at any later version is idempotent, and an id deleted or reused in any gap cannot hide its old pair.

**Bound and Overflow.** If a copy's unconsumed pairs pass `max(1M pairs, arcs/8)`:
- the writer replaces that copy's whole unconsumed list with one `Overflow{seq = its version}` node, and resets `pairs` to 0;
- an Overflow node is relinked, consumed and freed exactly like a segment. A batch forked before it gets it through CatchUp (seq > fork). A phase forked at or after it consumes it, and its own copy then holds no Overflow, so a later CatchUp does not bring it back.
- The phase that consumes an Overflow rescans:
  - **at Supergraph:** it re-reads the full skeleton at its fork and runs the allocation over every node of S with no rank;
  - **at a Seal or Repair:** first `d = reduce_rows(PLUS, S_now)⟨rank-less node ids⟩` finds rank-less nodes with edges, and the §6 allocation ranks them (top ranks extend the dimension to `n_r'`); then π covers `n_r'`, and the new adjacencies are `triu(S_now(π,π),1)⟨¬struct(Up_all)⟩`, where `Up_all` is a read-only load of the base plus a host-built matrix of the delta, both at `n_r'`. S2 runs on them, then the full seed pass and a full push sweep.
- An Overflow is never created on a Built entry (§7 "Retention").

**Phases.**

| Phase | Batch unit | State kept (copy-on-write) | Snapshot |
|---|---|---|---|
| ReadSkeleton | node-id slab, ~2M edges | `Vec<Arc<Matrix>>` slabs | any |
| Order | one depth; huge graphs split depth 0 by level | `Arc` level stack, label and rank vectors | none (private) |
| Supergraph | §4 over S ∪ log nodes at the fork V_sg, with rank allocation | `Arc<Topology>` | V_sg; log ≤ V_sg consumed |
| Seed (V_c = V_sg) | the full-seed pass below, slab by slab | flat seed arrays, builder scratch | V_c |
| Relax | apex ranks [a, b), ~50M triangles per batch | flat metric, builder-owned | frozen at V_c |
| Seal | resolve the log against its fork V_s: allocation, §6 topology, then §6b recompute (unbounded, off the lock); then chunk the metric one array at a time (`WeightArr::from_vec`) | finished `Cch` | V_s; log ≤ V_s consumed |
| Repair | as Seal, from `base: Arc<Cch>` | `base` + log | its fork; log ≤ fork consumed |

**The full-seed pass.** It does not build `effective_fwd`. Per indexed type and node slab:
- **Forward values.** Score the disjoint union `m⟨¬struct(dm) ∧ ¬struct(dp)⟩ ∪ dp` with the weight index op (as `weight_ops.rs:375-387`, minus `effective_fwd`). By the tensor's invariants (`tensor.rs:52-63`), `dp` holds the live value wherever it shadows `m` (states H and I), and `dp ∩ dm = ∅`. So the union is exactly the effective value, with no MIN across layers.
- **Multi-edge pairs** (value `M`) take their minimum from `me` read through its own effective merge, per block (`weight_ops.rs:411-415`). Only multi-edge rows are touched. That covers a pair that loses one of three or more edges, which changes only `me`.
- MIN-reduce across types per ordered pair, giving W in node space, and permute: `W(π,π)` over `n_r`.
- **Both directions.** Upward weights come from `triu(W(π,π))` and downward weights from `triu(W(π,π)ᵀ)`: arc (lo, hi) is `node_of(lo) → node_of(hi)` upward, and the reverse downward (`customize.rs:243-248`).
- **By id.** Align each to the base: `eWiseAdd(MIN, X, iso-+∞ pattern of Up.base)⟨struct(Up.base)⟩` gives values in the base's position order. Then scatter `seeds[Ax[k]] = val[k]`, reading the id from `Ax`. For the delta (bounded by the fold trigger), load its `(lo, hi, id)` entries as a small read-only matrix and do the same. On a fresh lineage the scatter is the identity, at no extra cost.

Its time and peak are unmeasured. The derived peak is one slab plus `W` in rank space (≈40 MB per direction at 3.3M entries) plus the seed arrays. M2 measures it.

**Scheduling.** Repair batches go ahead of column batches on the builder queue. The default pool has one thread (`src/lib.rs:103`). A column batch already running still delays a Repair by up to its own length (unmeasured, M7).

**Convergence (derived, with mean costs).**
- A Repair absorbs k pairs in ≈ k × c(k).
  - c(512) ≈ 0.8 ms, c(1,024) ≈ 0.5 ms and c(4,096) ≈ 0.23 ms, from `maint.rs:85-94`. Cones overlap at large k.
  - For small k, c is at least ≈ 0.8 ms; the tail is up to tens of thousands of arcs per pair (cited).
- The backlog ratio per round is ≈ rate × c. So it shrinks only below ≈ 1,250 SET/s, and slowly near that rate:
  - at 1,000 SET/s the ratio is 0.8, so going from 410 pairs to the ≈30 that fit a 25 ms B_gil takes ≈ 12 rounds, ≈ 2 s of Repair;
  - at 100 SET/s it takes 1–2 rounds.
- `MAX_REDOS` (`cch/mod.rs:396`) counts only rounds whose backlog did not shrink. What happens past it is D2.

**Readers while Building.** New versions use the Dijkstra fallback: 222 ms p50 and 409 ms p95 on California (`fallback.rs:28-31` [U2]). Older versions keep their `Arc<Cch>`.

## 6. Incremental maintenance (built index, fixed order)

There is one flush per `Commit` segment, on the private version (`maint.rs:8-10, 467-536`). The steps are:

**append segment → resolve pairs against the private version → allocate ranks → classify → topology (S2) → recompute → publish.**

Structural decisions are pure functions of (held hierarchy, resolved pairs). The budget only decides where the work runs. Seal, Repair, Supergraph, the Overflow paths, S6 and writer rebase run the same steps.

**Rank allocation** (a phase of its own, before any classification):
1. Resolve every pair of the unit. Let E be the endpoints of pairs that have a live indexed edge in either direction.
2. For each rank-less node x in E, in id order, compute its degree d(x): its distinct neighbours over indexed types at this version. All of x's edges are new, since a rank-less node had no indexed edge before (I1′), so d(x) is exact and comes from the resolved pairs.
3. If d(x) ≤ 1 and `free_ranks` is not empty, take its lowest rank. Otherwise take the next top rank `n_r' + 1`, which lives only in the deltas.
4. Only then classify each pair with its final ranks.

Under P, a pool rank has no holder, so step 3 never strips a node, and the outcome does not depend on the order in which pairs are processed. The degree rule follows C9 [B]. A bottom rank with d neighbours makes those neighbours a clique (≈ d²/2 fill, pushed up), while a top rank costs the union of d root paths (46/94/190 arcs per path on grids [R3 §3]). A degree-1 node at the bottom costs one arc and no fill. A pool-ranked node that later gains edges adds fill against its earlier neighbours; `LineageStats` counts those arcs, and they feed the S7 trigger.

| Change | Rule | Correctness | Cost | Where |
|---|---|---|---|---|
| Weight SET/REMOVE; insert on a pair with an arc; edge delete | Metric only (S1). The seed is the pair's minimum valid weight per direction, or +∞ if none is left | G′ ⊆ H is unchanged [R2 S1, R3 §4] | p50 0.22 ms SET, 0.35 ms delete (#3363); mean ≳ 0.8 ms (derived); tail up to tens of thousands of arcs (`maint.rs:85-94`) | writer within B_gil |
| Node delete | Its edges cascade as logged pairs (`maint.rs:238-253`). Its rank and arcs stay; `deletions += deg`. Every arc at its rank resolves to +∞ by induction on rank (each triangle through it uses another arc at its rank) | as S1 | O(deg) + cones | writer |
| Insert (u,v) on a pair with no arc, both ranked (after allocation) | **S2** (Davis–Hager): add v to up(u) in a **host overlay** (rank → sorted additions), pop rows by rank reading merge(base, delta, overlay), push U(j)∖{p} into up(p), stop at fixpoint (DH01 Thm 4.2 [R2 §2]). Then sort the new arcs by (lo,hi), append ids, insert them into both deltas and `added_since_fold`, and preset the metric to +∞ | Adds arcs only, and ends when the validity rule holds [U4 §5, R2 §5]: I2, I4–I6; P holds because only ranked rows gain arcs | ancestor case: 6–22.5 new arcs on grids; across the top separator: 4–7 % of arcs [R3 §3]; roads unmeasured (M4) | writer if ≤ N_max = 4,096 new arcs and within B_gil, else Repair |
| A rank-less endpoint with a live edge | rank allocation above, then S2 | any rank is valid; a holderless rank is read by no arc (P) | pool rank: one arc at d = 1; top rank: the union of root paths (46/94/190 arcs per path on grids [R3 §3]) | writer |
| Node created, or id reused | Nothing until its first indexed edge. A reused id keeps its old rank and its arcs (at +∞) | — | 0 | — |
| Relationship types | Fixed by the definition (`spanning`, `maint.rs:317-325`) | — | — | — |

**Hand-off.**
- On abort (N_max, B_gil), the private topology and metric of *this segment's attempt* are dropped.
- The entry becomes `Building{Repair, base, log: [this segment]}` under **`next_generation()`**, as `rebuild_cch_in_background` does today (`falkordb_index.rs:1286-1296`), so this version's commit enqueues the Repair job (`versioned.rs:910-922`; `mvcc_graph.rs:213-221`). The new generation also discards any in-flight fold or S6 job as `Gone`.
- `base` is the entry's hierarchy **in this private version at the moment of the abort**. It already holds every earlier segment of the same query that was absorbed inline, so nothing between the fork and the abort is lost. Later segments of the query, and later commits, append to the log.
- The Repair runs the same steps, unbounded, and adopts as in §5. This replaces today's rebuild branches (`maint.rs:551-584`).
- The adoption-failure hand-off (§5) is the other kind: it keeps the generation through `Absorbed::Restart` in catch-up, so the running job continues.

**Decay.** Thresholds are derived; M6 calibrates them (C's valve: `cch_index.c:468-496` [U3 §1]).

| Signal | Action | Readers meanwhile |
|---|---|---|
| deletions ≥ max(1,024, n/8), or dead ids ≥ 12.5 % | **S6** (Built-entry job): a new lineage from the current order (game + full seed + push sweep, ≈ 97 + seed + 219 ms) | the old hierarchy |
| arcs ≥ 1.25 × arcs at build, one commit adding > arcs/8, top ranks ≥ reserve, or pool-row fill > arcs/16 | **S7**: re-order as a new generation through the batched build, with this commit's segment as its log | Dijkstra fallback (≈ one build); a shadow generation is PR 10 |
| delta entries > max(64k, √\|base\|) | **fold** (Built-entry job) | the old hierarchy |

**Built-entry jobs (D6).** A fold or S6 runs while the entry stays Built and keeps serving.
- **Start.** The writer whose flush crosses a trigger sets `upkeep = Some{kind, ticket, since = its version}`; `ticket` comes from the same counter as generations. `indexes_to_build_since` also returns a Built entry whose ticket is new against the replaced version, so that commit enqueues the job (the D6 framework addition).
- **Retention.** While `upkeep` is set, every writer keeps its nodes in the published log after its commit; without it, the log is dropped at publish (§7). Past the §5 bound, the writer clears `upkeep` and the log instead of writing an Overflow: the job is abandoned, and a later commit re-triggers it.
- **Run.** The job forks at F off the lock and builds its result.
- **Publish under the lock.** It is discarded if the generation or the ticket changed. Otherwise:
  - **S6:** replay the retained nodes with `seq > F` through the §6 steps (allocation first) onto the new lineage, resolved against the published graph, within B_gil. If that does not fit, publish nothing; the job runs another round off the lock from a new fork F₂, replaying (F, F₂] there, up to `MAX_REDOS` rounds, and then D2.
  - **fold:** see below.
  - Then clear `upkeep` and the log, and count the publish in the entry's `indexer_commits`, so writers forked before it are `overtaken` (§5 rebase table; `versioned.rs:759` drops its `is_building()` requirement for this case).
- **S6's input order** is the current `rank_of` over `n_r'`, top ranks included. Ranks whose holders have no indexed edge at F go back to the pool (`node_of = None`), which keeps P. Rank-less endpoints of later pairs get ranks through allocation at the replay.

**Fold job.** A fold changes representation, not values: same ranks, same ids, same metric.
- The job forks at F and builds new bases from base ⊕ delta_F with GraphBLAS: eWiseAdd over read-only loads, then unload. Ids are carried as values.
- **Its result is the published Cch with only the four structures (Up.base, Down.base, Up.delta, Down.delta) replaced.** The metric, `rank_of`, `node_of`, `free_ranks`, `arc_capacity` and `LineageStats` come from the published copy at publish. The new deltas hold exactly the arcs in the published `added_since_fold` with `seq > F`, inserted into empty `cow_btree`s on both sides. This is O(arcs added since F × log n), under the lock. `added_since_fold` keeps only those arcs.
- **A writer that overtook a fold** applies the same function to its own copy: its bases become the folded ones, and its deltas become the arcs of its own `added_since_fold` with `seq > F`. No segment is replayed, because no value changes.
- Without retention, a fold needs no log: metric-only writes after F stay in the published metric, which the fold keeps.

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

So the front `N ∪ {changed seeds}` reaches every arc that differs, and the result is **bit-identical to a full customization of H′** (derived). Round-2 reviewer 3 re-checked it against `partial.rs:168-255` and `customize.rs:491-539`. Its premises are that every edge has an arc before the recompute, and that the front holds every changed seed. Rank allocation before classification (§6) and the corrected full-seed pass (§5) are what keep those premises.

**Lemma B (S5, later).** S5 re-ranks an elimination subtree T within its rank set R_T. Its front is T's arcs plus the boundary clique K_T, and its input includes T's outward edges. It **deletes** delta and base entries, which the `cow_btree` delta supports (`mod.rs:20-23`). Base entries need a "replaced row" marker, and `added_since_fold` gains deletions. That contract change ships in the S5 PR, which starts a new topology version.

**Recompute set per change** (fronts are node pairs, resolved to ids after all topology changes of the unit):

| Change | Front, then closed by propagation | Why it is complete |
|---|---|---|
| SET/REMOVE, insert on an existing arc, delete | that pair's arc, with its new seed | (i) |
| Node delete | the arcs of its deg(v) logged pairs. Its fill arcs reach +∞ through the cone | (i) + (iii) |
| S2 insert (new fill) | the inserted pairs' arcs ∪ N | Lemma A |
| Rank allocated (pool or top) | as S2 | a holderless rank is read by no arc (P) |
| S5 (later) | T's arcs ∪ K_T | Lemma B |
| S6/S7 | everything: full seed + push sweep | new ids |
| Log at Seal/Repair (any gap, any id reuse) | the union of the rows above over all logged pairs, after allocation and all topology changes | pair state = f(current graph); §5 |
| Overflow | rescan (§5) + full seed + full sweep | — |

**Algorithm.** `recustomize_seeded` (`partial.rs:98-258`) with these changes:
1. The front takes pairs, and `partial.rs:115-118` checks rank presence.
2. The heap key is `(lo, hi, id)`.
3. Apex ids come from merging `Down(lo)` and `Down(hi)` (base slice plus delta cursor).
4. A propagated arc's id comes from a cursor walk of up(hi) alongside up(lo) for w > hi (by I5), and from a seek into up(w) for w < hi. That seek stays in the cost model.
5. Budget accounting is in **work units**: merge steps + pushes + seed lookups weighted by multiplicity, with a monotonic clock read every 256 pops, against the lock hold's shared deadline. Aborting publishes nothing.

Seeds come pair by pair through `PairWeights` (≈0.65 µs of the 1.3 µs/arc; derived from "two graph lookups seed each", `maint.rs:85-87`).

Valves, off the lock only, switch to the full path (full seed by id + push):
- the front is over n/8 arcs (`partial.rs:131`);
- the cone passes the break-even of the *new* full path, `(seed pass + 219 ms) / 1.3 µs`, recalibrated from M2.

A Seal or Repair whose S2 put arcs in the delta can take this path, for example after one insert across the top separator (≈290k new arcs > n/8 ≈ 190k on California, derived). The by-id scatter in §5 is what makes that path correct for delta arcs.

**Cost on California.** Cited unless marked:

| Case | Cost |
|---|---|
| scoped, per arc (random cones) | ≈ 1.3 µs (`maint.rs:85-87`); for top-clique arcs unmeasured (M7) |
| one SET | 0.22 ms p50 (#3363) ≈ 170 arcs (derived); mean ≳ 0.8 ms ≈ 615 arcs (derived from 0.41 s / 512) |
| k random pairs | 0.41 s at 512; 0.51 s at 1,024; 0.95 s at 4,096 |
| 200×200 grid, one edge | 10.7 ms scoped against 464 ms full (#3773) |
| ancestor insert | ≈ 0.25–0.5 ms (derived from [R3 §3]) |
| new node, degree d, top rank | ≤ d root paths, 46–190 arcs each on grids (cited [R3 §3]); roads unmeasured (M4) |
| insert across the top separator | ≈ 290k arcs ≈ 0.4 s if roads behave like grids (derived) → Repair |
| full push sweep | 219 ms (#3323) |
| fold publish | O(arcs added since F × log n); unmeasured |
| today, under the lock | 0.47 s at a build commit (#3772), ≈ 0.5 s in `GRAPH.EFFECT` (#3363). Both are removed |

**Interleaving with a batched build.**

| When the edge changed | Topology | Metric |
|---|---|---|
| before its ReadSkeleton slab | in S | — |
| after its slab, before V_sg (in any batch or gap) | its logged pair is added to B' (live edge) or dropped from it (no edge) at Supergraph, after allocation ranks any new endpoint | seeded at V_c = V_sg |
| a node created and connected during ReadSkeleton or Order | rank allocation at Supergraph (pool or top rank) | seeded at V_c |
| after V_c: during Seed or Relax, **or between two batches** (a `Same` publish keeps the node) | allocation, then S2 at the Seal, with new arcs at +∞ | **never patched mid-sweep**: the Seal's scoped pass recomputes from scratch |
| during the Seal or a Repair | nodes with seq > fork, relinked at CatchUp | absorbed at adoption within B_gil, otherwise the next Repair |
| a writer's query that a batch publish overtakes | its own nodes are relinked (Building) or replayed (Built), §5 | as the topology |
| after an inline abort, before the Repair forks | appended to the Repair's log | resolved at the Repair's fork |
| an overflow anywhere | an Overflow node with its seq; the next consumer forked at or after it rescans | full seed + sweep |
| after adoption | §6 | §6 |

**Order.** Topology always goes first (allocation, S2/S5, new ids, deltas, metric arrays extended), then the recompute, then publish. The recompute never runs over a topology that lacks an arc: every pair with an edge and no arc is in the log, and allocation and S2 run before the resolve to ids. M5 adds debug checks for `pattern(S_now) ⊆ Up`, P, and ranks on every node with an indexed edge. `partial.rs` checks only its front (`partial.rs:111-145`), so it is no guard.

**Worked example (derived, mean costs).**
- California's build: ≈ 3.6 s Order, then Supergraph + Seed + Relax ≈ 0.5 s.
- At 100 SET/s, writes after V_c are ≈ 50 pairs. The Seal absorbs them off the lock in ≈ 40 ms; ≈ 4 pairs land during the Seal; adoption costs ≈ 3 ms under the lock.
- At 1,000 SET/s: 500 pairs, a Seal of 0.4 s, then ≈ 12 Repair rounds (§5).

**Where it runs, and the bound.**
- **Lock paths:**
  - the master writer's flush;
  - a replica's `GRAPH.EFFECT`;
  - adoption catch-up;
  - writer rebase;
  - a Built-entry job's publish (S6 replay, fold delta rebuild).
- **Each lock hold** gets one deadline, B_gil after the hold starts. Every index the hold updates checks the same deadline: the flush's loop over held indexes (`maint.rs:480-519`), and every index a rebase or catch-up touches. When it passes, the index in progress and every index not yet done hand off (§6), each under its own new generation. The default is 25 ms (D1), counted in work units plus the clock, and calibrated on the top clique, not on the average.
- Also per hold: S2 ≤ N_max = 4,096 new arcs per index.
- Fixed costs are counted too: `cow_btree` inserts (O(new arcs × log n)) and the chunk copies.
- **A single SET on a top arc** can exceed B_gil and send new versions to the fallback for one Repair. M4 measures how often (the p99 and max cone). If more than 0.1 % of single SETs hand off, D1 is revisited.
- **Off the lock only:** the full-sweep valves, Overflow rescans, S5, S6, S7 and the fold's base build.

**MVCC.**
- Every write goes to a private version:
  - `WeightArr::set` copies its chunk (`customize.rs:182-189`);
  - `cow_btree` inserts path-copy;
  - the base is never written.
- `Cch{Arc<TopologyVersion>, Metric}` is swapped as one `Arc`, so a reader holds a consistent pair.
- An aborted attempt publishes `Building`, which never serves a hierarchy. No reader sees a half-recomputed metric or a topology missing an arc.

## 7. MVCC and replication

- **Versions.** A version holds one of:
  - `Built { cch: Arc<Cch>, upkeep, log }`, where `log` is empty unless `upkeep` is set;
  - `Building { phase state, log }`.

  Bases are shared across a lineage. Deltas, maps and metric chunks are shared until written.
- **Retention.** At a writer's publish:
  - a Building entry keeps its log (a later phase consumes it);
  - a Built entry with `upkeep = None` drops its log;
  - a Built entry with `upkeep = Some{since}` keeps its nodes with `seq > since`, until the job publishes, is discarded, or is abandoned past the bound.
- **Accounting.** `CchIndex::memory_usage` (`cch/mod.rs:630-634`) adds the retained log: 16 B per pair plus each node. The bound counts retained pairs, not only unconsumed ones.
- **Kept rules** [U2 §3]:
  - private writes (`cow.rs:48-59`);
  - every GraphBLAS object waited before publish (`mvcc_graph.rs:332-347`); read-only loads are unloaded or freed before publish;
  - replaced versions dropped outside the lock (`online_build.rs:348-360`).
- **Framework changes**, all in PR 4 except D6's (PR 9):
  - `writer_rebase` receives the fork's entry, and `indexer_catch_up` reads the published entry; neither gates on `Written`;
  - `overtaken` also fires for a Built entry whose `indexer_commits` moved (D6);
  - `indexes_to_build_since` also enqueues a Built entry with a new upkeep ticket (D6);
  - a Built-entry job publish is fitted against generation and ticket (D6).
- **Determinism: same distances only.** Ids and ranks depend on how commits are grouped and on fork points:
  - S2's id order;
  - which node takes which pool or top rank;
  - whether a Repair sees an insert that was deleted before its fork.

  The values are exact on every node. Equal-cost paths may differ (I9). With non-integer weights, differing hierarchies can differ in the last bit (derived). M5 compares by node pair, not by id. Identical paths would need the order and the topology events on the wire, which C lacks too [B §5.7].
- **Replicas.** `GRAPH.EFFECT` flushes inline under the lock with the same steps and the same shared B_gil.
- **RDB.** As today, only the definition is saved, and a load rebuilds (`cch/mod.rs:375-377`; C `cch_index.c:161-163`). Persisting `rank_of` (4 B/node) to skip ordering is D4.

## 8. Memory and time budget

California: 1.54M node capacity [U1 §1]; 5.84M arcs with feral, 5.53M with METIS. The task's 1.9M nodes scale per node.

| Item | Value | Basis |
|---|---|---|
| Topology, steady | 16 × 5.84M + 16 × 1.54M ≈ **118 MB**, against 142.7 MB today | derived; today's figure cited (#3321) |
| Delta + `added_since_fold` at the fold trigger | ≤ 64k entries × (16 × 2 + 16) B ≈ 3 MB | derived from `cow_btree` ≤ 16 B |
| Log, Built, no job | 0 | derived (§7) |
| Log, retained during S6 (≈ 0.5 s) | 16 B × pairs written meanwhile: ≈ 8 KB at 1,000 SET/s; capped by the bound, ≤ 16 MB | derived |
| Metric | 187 MB (+ chunk headers; 396–412 MB process against 365, #3774) | derived / cited |
| Total against C | ≈ 305 MB against C's ≈ 364 MB | derived |
| Peak by phase, over the input | Order ≈ skeleton 20 + levels 60 ≈ 80 MB; Supergraph ≈ 150 MB; Seed ≈ 118 + 94 + ≤ 80 ≈ 290 MB; Relax/Seal ≈ 118 + 187 + 47 (one array converting) ≈ **352 MB** | derived, unmeasured. C's whole-build peak is unmeasured; M1/M2 compare against `:edge-c` |
| Ordering time | ≤ 3.62 s (feral today), target ≤ 3.07 s (C METIS) | cited (#3321) |
| Supergraph / full seed / customize | ≈ 100–150 ms / unmeasured / ≤ 219 ms | cited / derived |
| Per update | SET p50 0.22 ms, mean ≳ 0.8 ms; delete 0.35 ms; ancestor insert 0.25–0.5 ms | cited / derived |
| Lock hold | ≤ B_gil (25 ms default) + fixed costs per hold, all indexes together, against 0.47 s today | derived / cited |
| Fallback window per hand-off | one Repair plus queueing behind a running column batch | unmeasured (M7) |
| Fallback window per S7 | ≈ one batched build (≈ 4.1 s on California) | derived |

**9M edges, road-like:** ≈ 15.8M arcs and 4.2M nodes; topology ≈ 320 MB and metric ≈ 506 MB. Slab reads avoid the 400 MB / 76 ms `effective` transient [FOLD]. All derived.

**1500×1500 grid:** ≈ 100M arcs by extrapolating R3's 9.6 → 14.7 → 20.8 arcs/node [R3 §3]. Guard (b) refuses it before allocation (unmeasured).

## 9. Implementation plan

**Measurements first.** Each one includes a deliberate break, to prove it can fail.

| Id | Measure | Pass |
|---|---|---|
| M1 | ND prototype, including the masked skeleton read, on California and a 9M-edge road graph, at 1/4/16 GraphBLAS threads | time ≤ 3.62 s (report against 3.07 s); peak ≤ 363 MiB over the input, or the gap documented; arcs ≤ 5.84M and triangles ≤ 53.9M (report the gap to 5.53M/48.6M); tree height ≤ feral's; **query p99 with the new order ≤ `:edge-c`'s**; identical order across thread counts |
| M2 | read paths with feral's order: GxB iterators against read-only Container slices; relax; scoped µs/arc; full-seed pass time and peak (by-id scatter included); valued against iso `Down` | query p99 ≤ `:edge-c` on roadNet-CA (and ≤ 526 µs, #3324); customize ≤ 219 ms; scoped ≤ 1.3 µs/arc on random pairs |
| M3 | 32-bit arrays at California size | 32-bit |
| M4 | cone and new-arc distributions (p50/p95/p99/max) for single SETs, deletes, local/random/top-separator inserts, and new nodes of degree 1, 4 and 20 under both pool and top ranks | p99 single-SET cone fits B_gil; hand-off rate reported; the degree threshold of §6 confirmed or moved |
| M5 | randomized property test against a full customization (by pair) and Dijkstra. Cases listed below | 0 mismatches over 10k graphs; `pattern(S_now) ⊆ Up`; P; every node with an indexed edge ranked |
| M6 | write-mix replay | sets the decay thresholds and B_gil |
| M7 | batched build under writes; top-separator SET; three indexes on one type, one top-arc SET each; fold and S6 under back-to-back writers | lock-hold p99 ≤ B_gil, max ≤ 2 × B_gil **per hold**; **concurrent read p99 during a build ≤ 1.1 × idle**; fallback window p99 reported; fold and S6 publishes stick (delta shrinks) |
| M8 | Seal/Repair convergence at 100, 1,000 and 2,000 SET/s | adopts within 2 rounds at 100 and within 15 at 1,000; the non-converging rate reported, with the D2 backstop engaged |

**M5 cases.**
- Deletions: a bulk delete still pending in `dm` at ReadSkeleton must leave no arc for its pairs. A delete made after its slab is read and before V_sg must drop its pair from B′. Each case needs a break run with the mask, or the drop, removed.
- Round 1: deletes and id reuse in the gap between batches with a `Same` publish; a delete between an inline abort and the Repair fork; a > 1,024-pair commit with a new adjacency; Overflow; more than `reserve` isolated nodes, then connect an early one.
- Log and rebase:
  - an overflow in the gap before a CatchUp publish;
  - an overflow on a writer's fork, with a `Same` publish before it escalates;
  - a Repair that consumes an Overflow, then a CatchUp with new writes (the Overflow must not come back);
  - a batch publish (Same/Swap) during a writer's query, entry Building;
  - an adoption between a writer's fork and its escalation, with an insert that needs S2, both within and over B_gil;
  - a two-segment query whose second segment aborts (base = H′);
  - an inline abort, then verify that a job runs (a new generation).
- Ranks:
  - a node created and connected during Order, during Relax, and during a Seal;
  - more than `reserve` nodes created with edges during ReadSkeleton or Order;
  - Overflow together with more than `reserve` new nodes;
  - the pair (a, h) with both rank-less, and a two-pair unit in both orders;
  - a node isolated in its slab that gains an edge before V_sg.
- Seeds:
  - tensor states H and I, id reuse into another pair, a multi-edge pair going 3 → 2, one-way edges;
  - a top-separator insert at a Seal and at a Repair that crosses n/8 (the full path over delta arcs).
- Built-entry jobs:
  - a metric-only SET, a lazy rank, and an S2 between a fold's fork and its publish;
  - the same between an S6 fork and its publish;
  - a writer forked before a fold or S6 publish that commits after it.

**PRs.** Each is small, closes its own issue, and lists its functions in `COVERAGE.tsv`.
1. GraphBLAS wrappers: masked `mxv`/`mxm`, `extract_Vector`, `build_Scalar_Vector`, reduce, `Matrix<u32>`, read-only Container load/unload, `GxB_Context`, plus the PreJIT harvest of each user operator and variant.
2. `ChunkedArr<T>` with append; metric append with a +∞ preset.
3. Topology as base + `cow_btree` delta with I1′/P/I3′, still fed by feral, with the readers changed. Bit-identical metric; gates M2 and M3. **If M2 fails, stop and revisit the read path before PR 4.**
4. Writer-side pair log in `CchIndex` state (seq'd nodes, Overflow, relink, consumption, retention), the framework hook changes of §7 for Building entries, the generalized recompute engine with a per-hold deadline, and rank allocation. Proves M5's log, rebase and rank cases.
5. S2 with the host overlay, the `Building{Repair}` hand-off under a new generation with priority scheduling, and B_gil on every lock path. Replaces both rebuild branches; proves M4 and M5.
6. GraphBLAS ND behind `CCH_ORDERING` (default feral), with guards. Proves M1.
7. Flip the default; delete feral and the `CchInput` CSR.
8. Phased batched build with the corrected full-seed pass (by-id scatter). Proves M7 and M8.
9. D6: Built-entry jobs (ticket, enqueue, fit, `overtaken` for Built entries), then decay, S6, S7 and the fold. Proves M5's job cases and M7's "publishes stick".
10. Later, behind data: S5, a shadow generation (removes S7's fallback window), G2, O2.

## 10. Risks and open questions

1. **Fallback windows.** A top-arc SET or about 30 random SETs in one commit hand off (derived). M4 and M7 decide whether 25 ms is right (D1).
2. **Write rates above ≈ 1,250 SET/s** never converge (derived). D2.
3. **Matching quality** against SHEM is unmeasured [U3 §3]. If M1 fails, keep feral behind PR 3's API; the CSR still leaves the supergraph and customization.
4. **Read-only Container semantics.** Lifetime and wait rules for `GxB_IS_READONLY` arrays under concurrent read-only loads need PR 1 tests.
5. **Built-entry jobs** need the D6 framework addition. Under writes past the bound during one S6 (≈ 2M pairs/s on California, derived), S6 is abandoned and retried, and the decay waits.
6. **Cross-separator inserts on roads** are unmeasured. If frequent, S5 moves earlier.
7. **Two hierarchies alive** during S6: ≈ 2 × 0.3 GB on California (derived).
8. **S7's fallback window** is about one build until the shadow generation (PR 10) lands.
9. **Pool versus top ranks.** The degree-1 threshold is derived from C9, not measured. M4 decides it. Many new high-degree nodes consume top ranks and trigger S7 sooner.
10. **Proofs.** Lemmas A/B, log completeness, rank allocation and P rest on M5 until the Lean coverage lands.
11. **Serial host FM.** The ordering time could exceed 3.62 s with one thread. M1 decides whether D5 needs more threads.

## Decisions the team must make

- **D1. Lock budget B_gil.**
  - Options: (a) one time budget per lock hold, shared by every index; (b) a larger budget for the master writer than for replicas; (c) count arcs, as in the synthesis.
  - **Recommend (a): 25 ms per hold, a config, in work units plus the clock**, calibrated by M4/M7.
- **D2. Above the convergence rate.**
  - Options: (a) after `MAX_REDOS` non-shrinking rounds, force one adoption over budget (bounded by the full path, ≈ 0.4 s, like today's 0.47 s); (b) stay on the fallback and report "repairing" in `db.indexes()`; (c) throttle writers.
  - **Recommend (a) plus the status report in (b).** The same rule ends an S6 that keeps missing its publish budget.
- **D3. The read representation for goal 1.**
  - Options: (a) GxB row iterators only; (b) Rust-owned arrays loaded read-only into GraphBLAS for algebra and read as slices.
  - **Recommend (b) unless M2 shows that (a) meets the p99 and 219 ms gates.**
- **D4. Persist `rank_of` in the RDB** to skip the 3.62 s ordering on load.
  - Options: persist it, or rebuild.
  - **Recommend rebuild first; persist only if load time matters.**
- **D5. Builder GraphBLAS threads.**
  - Options: 1, a small default, or all cores.
  - **Recommend 1 (a config)**, raised only if M1 and M7 show query p99 holds.
- **D6. Jobs on a Built index (fold, S6).**
  - Options: (a) a framework job type for Built entries, with a ticket, log retention, a catch-up and a writer rebase (§6); (b) fold only inside Repair/S6, and run S6 as a new generation.
  - **Recommend (a).** Under (b), a quiet index's delta never shrinks, and every S6 opens a fallback window.
- **D7. Valued against iso `Down`.**
  - Options: 48 B/arc with no apex searches, against 44 B/arc with searches.
  - **Recommend valued**, unless M2 shows that memory decides.
- **D8. The replica promise.**
  - Options: equal distances, or equal paths (needs a new effects format).
  - **Recommend equal distances.**

## Appendix A: Review issues and dispositions (round 1)

R1 is review 1, R2 review 2 and R3 review 3 of `01-synthesis.md`. Where a round-2 change supersedes a round-1 fix, Appendix B says so.

| # | Severity | Issue | Disposition | Change made |
|---|---|---|---|---|
| R1.1 | fatal | Writes between batches are lost: `indexer_apply` clears the id set, `Same` skips catch-up, and the "before" pair is read at a fork that already contains the gap | accepted | Pairs are logged at write time in CCH's own state (§5); the F/P resolution is deleted; M5 gap cases added |
| R1.2 | major | The `Arc` hash-set log is cloned on every append | accepted | Persistent segment list, deduped off the lock (§2, §5) |
| R1.3 | major | `set_structural`/`PAIR_CAP` drop the pairs S2 needs; `many` is not sticky | accepted | Pairs are never dropped; overflow handled by the phase that consumes it (§5; superseded in round 2 by the Overflow node) |
| R1.4 | major | The rank swap strips the rank from a live isolated node | accepted | Lazy `Option` ranks, a `free_ranks` pool, top ranks in the delta; I1′ (§2, §6; round 2 makes pool ranks holderless) |
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

## Appendix B: Re-verification (round 2) dispositions

V1 is round-2 review 1 (log lifecycle), V2 review 2 (log semantics and seeds), V3 review 3 (ranks and Lemma A premises). Every issue was re-checked at @08c; the evidence column names what was read. "Confirmed, fix replaced" means the defect is real and the reviewer's proposed fix was not taken as written.

| # | Severity | Issue | Disposition and evidence | Change made |
|---|---|---|---|---|
| V1.1 | fatal | The `many` flag has no seq: CatchUp loses an overflow in the gap, or re-raises it forever | **Confirmed.** `catch_up_index` starts from the batch's copy (`versioned.rs:694`); `rebase_index` from the published one (`:786`); 02 §5 prepends only *segments* with seq > fork | Overflow is a log node with a seq, relinked, consumed and freed like a segment; a consumer forked at or after it clears it for good (§5). M5 cases for the gap, the writer fork, and "does not come back" |
| V1.2 | major | Writer rebase onto an adopted (Built) copy appends to a log nothing consumes; the hooks return early on an empty `written` | **Confirmed.** `overtaken` is true for a Building fork against a Ready copy (`versioned.rs:759-761`); `writer_rebase`/`indexer_catch_up` return on an empty `written` (`cch/mod.rs:525, 557`), and CCH puts no ids there | Rebase defined per (fork, published) state; Building → Built runs the §6 flush on the own nodes within B_gil, else `Restart` under a new generation (§5). The hooks read the entries (§7, PR 4). M5 case |
| V1.3 | major | A Built entry's log lifecycle is contradictory: retained forever, or no catch-up source for Built-entry jobs | **Confirmed.** 02 §5 has writers append for Built entries, and only Supergraph/Seal/Repair consume; the fold catches up from `touched_since_fold`, which misses metric-only SETs | Retention only while an upkeep job is in flight (`upkeep.since`); past the bound the job is abandoned, never an Overflow on Built (§6, §7). S6 replays retained nodes > F; the fold needs none because it keeps the published metric (§6). M5 cases |
| V1.4 | major | The fold's catch-up keeps the batch's metric, maps and `arc_capacity` | **Confirmed.** `caught = ours.clone()` (`versioned.rs:694`); 02 §6 said "the metric is untouched" without saying whose | Confirmed, fix replaced: the fold result is the *published* Cch with only the bases and deltas replaced; the new deltas come from `added_since_fold` (seq > F), which records each S2 arc once on both sides, not from touched rows (§6). M5 case |
| V1.5 | major | Writers forked before a Built-entry job's publish revert it, since `overtaken` needs a Building fork | **Confirmed.** `versioned.rs:759` (`ours.is_building()`), `mvcc_graph.rs:247-248` (a batch keeps the data version, so the writer is not stale), `:205-223` (commit publishes the fork) | Confirmed, fix partly replaced: the job publish counts in `indexer_commits` and `overtaken` covers Built entries (D6). A writer over a fold re-applies the fold to its own copy (no replay); over S6 it replays its own nodes (§5 table). M7 "publishes stick" |
| V1.6 | major | Log pairs whose endpoints were created after Order have no rank at Supergraph, Seal or `many` | **Confirmed.** 02 §4 step 1 and §5 resolve log pairs in rank space; the lazy-rank rule existed only in 02 §6 | Rank allocation is a phase of every resolver, before classification (§6); Supergraph extends π and B' (§4); the Overflow rescan ranks nodes before extracting (§5). M5 cases |
| V1.7 | minor | The inline hand-off must take a new generation or no job starts | **Confirmed.** `building_since` (`versioned.rs:910-922`), `commit` enqueues only those (`mvcc_graph.rs:213-221`); today's rebuild bumps it (`falkordb_index.rs:1286-1296`) | Hand-off takes `next_generation()`, discarding any upkeep job; adoption failure keeps the generation through `Restart` (§6) |
| V1.8 | minor | The abort base is ambiguous for multi-segment queries | **Confirmed.** One flush per `Commit` segment (`maint.rs:8-10`; `commit.rs:85-103`) | Base = the entry's hierarchy in the private version at the abort, which holds earlier segments (§6). M5 case |
| V2.1 | major | The boolean `many` cannot be both sticky and cleared | **Confirmed**, same defect as V1.1 (reviewer rated it major, V1 fatal; kept fatal there) | As V1.1. The reviewer's `many_through: Option<Seq>` is equivalent; a node was chosen because rebase and relink then need no special case |
| V2.2 | major | Consumed segments are never unlinked; a Built index grows 16 B per written pair; relinking `Box` pairs copies them | **Confirmed.** 02 §5 "records consumed ≤ seq"; 02 §2 `pairs: Box<[..]>`; `memory_usage` counts only the Cch (`cch/mod.rs:630-634`) | Consumers drop what they consumed; Built entries retain nothing without a job; pairs are `Arc<[..]>`; retained pairs count in `memory_usage` and the bound (§2, §5, §7) |
| V2.3 | major | The framework rebase replaces the fork's copy and carries only `by_this_version`, dropping the writer's segment (Building → Building) | **Confirmed.** `versioned.rs:785-808`, `cch/mod.rs:525`. A lost write in a common interleaving; the reviewer's major is kept because 02 stated the intent and missed only the hook | The hook receives the fork's entry; Building → Building relinks the own nodes (§5). The alternative of a `Changes = CchLogSegment` carried by the framework is rejected: reading the entry needs no new framework type. M5 case |
| V2.4 | major | Fold, S6 and S7 publishes are overwritten by writers forked before them | **Confirmed**, same as V1.5 | As V1.5. S7 is now a new generation (Building), so it cannot be overwritten that way |
| V2.5 | major | Nodes without a rank in π break Supergraph and both `many` paths | **Confirmed**, same as V1.6 | As V1.6. The reviewer's "leave unranked pairs unconsumed" alternative is rejected: allocating at Supergraph keeps I4 at V_sg and avoids S2 at the Seal |
| V2.6 | major | The full-seed pass scores stale `m` under `dp`, misses `me`-only deletions, and seeds only one direction | **Confirmed.** `tensor.rs:55-60, 86, 91` (dp shadows m, dp ∩ dm = ∅), `weight_ops.rs:380, 415`, `customize.rs:243-248` (upward and downward arrays) | Score `m⟨¬dm ∧ ¬dp⟩ ∪ dp`; `me` through its effective merge; `triu(W)` upward and `triu(Wᵀ)` downward (§5). M5 seed cases |
| V2.7 | minor | Pool ranks (bottom) give O(d²) fill for new hubs | **Confirmed.** A bottom rank's up-row is a clique (`order.rs:381-437`; [B] C9) | Degree rule: pool rank only at d ≤ 1, else a top rank; pool-row fill counted in `LineageStats` and an S7 trigger (§6). M4 cases |
| V2.8 | minor | B_gil is per index, but one flush updates every held index | **Confirmed.** `maint.rs:480-519` loops over every held index in one flush | One deadline per lock hold, shared; indexes not done hand off (§6b, D1). M7 three-index case |
| V3.1 | major | Nothing removes a rank from `free_ranks` when its holder gains arcs; a later pop strips a node with arcs | **Confirmed.** 02 §3 step 0 put isolated nodes on pool ranks, and 02 §6's both-ranked S2 row never touches the pool | Confirmed, fix replaced: pool ranks are **holderless** (isolated nodes and unused ids get no rank), so a pop never strips anyone; invariant P and its M5 check (§2, §3). The reviewer's "r is in the pool iff up(r) and down(r) are empty" follows from P, but taken alone it still lets a held rank with no arcs be popped, which strips its holder |
| V3.2 | major | Classification before rank allocation makes the result depend on pair order; a pop can strip an endpoint of the same pair | **Confirmed.** 02 §6 pipeline "classify → topology (S2, ranks)"; 02 §6 row "the holder becomes rank-less" | Allocation is a phase before classification, in id order (§6). The reviewer's step (b), removing ranked endpoints from the pool, is unnecessary under holderless ranks. M5 cases in both orders |
| V3.3 | major | The build path has no rank rule for nodes created past `n_r` during ReadSkeleton/Order; the step-0 pool uses slab-time degrees | **Confirmed**, same root as V1.6 | §4 steps 1, 2 and 5: allocation at Supergraph, B' and counts recomputed, pool computed after allocation. M5 case with more than `reserve` new nodes |
| V3.4 | major | The full path aligns seeds only to `struct(Up.base)`, so delta arcs stay +∞, and a single top-separator insert reaches that path | **Confirmed.** 02 §5 "read row-major, which is id order on a fresh lineage"; `partial.rs:131` (n/8 valve); 02 §6b's 290k-arc case | Seeds are scattered by id from `Ax` and from the delta's entries; "no positional writes" added to arc identity (§2, §5). M5 case at a Seal and a Repair |
| V3.5 | major | The `many` extract uses a partial π and a base-dimension mask; `struct(Up)` is not one GraphBLAS object | **Confirmed.** 02 §5 Bound; I1′ | Overflow rescan: reduce over rank-less rows, allocation, π over `n_r'`, a mask from a read-only base load plus a delta matrix at `n_r'` (§5). M5 case |
| V3.6 | major | The fold and S6 say nothing about rank maps and top ranks changed after the fork | **Confirmed**, same root as V1.4 | Fold takes every map from the published copy (§6); S6 starts from the current `rank_of` over `n_r'`, returns edge-less holders' ranks to the pool, and replays nodes > F with allocation (§6) |
| V3.7 | minor | Pool-first has no cost model; a bottom rank cliques the neighbours | **Confirmed**, same as V2.7 | As V2.7; M4 measures pool against top ranks at degrees 1, 4, 20 |

**Rejected:** no issue was rejected. These proposed fixes were not taken as written, and the reason is in each row: V1.4 and V1.5 (fold), V2.1 (`many_through`), V2.3 (a framework-carried segment type), V2.5 (leave pairs unconsumed), V3.1 (the "rows empty" pool), V3.2 (step b).

**Severity notes.** V1.1 and V2.1 are one defect. V1.1 rated it fatal because it loses a write, and that rating is kept. V2.3 also loses a write. It stays major because 02 stated the right intent ("writer rebase appends the writer's own segments") and missed only the hook that would carry it.

**Does the write log hold now (derived, on paper).** Every interleaving the three reviews raised now has a rule, and each rule is by seq or by entry state, never by `Written`:
- the gap and Same/Swap (the batch's copy);
- CatchUp (relink > fork);
- writer rebase (by state, own nodes always kept);
- adoption (B_gil, else Restart);
- Overflow (a seq'd node);
- inline hand-off (new generation, base H′);
- Built-entry jobs (ticket, retention, `overtaken`);
- ranks (allocation before classification, holderless pool).

That is not evidence. It holds only once M5's listed cases pass, with a deliberate break in each (for example, drop the Overflow relink) that a test must catch.
