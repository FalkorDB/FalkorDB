import CchTopologyReview.Basic
import CchTopologyReview.Model
import CchTopologyReview.Inv
import CchTopologyReview.Alloc
import CchTopologyReview.S2
import CchTopologyReview.Lineage
import CchTopologyReview.Props
import CchTopologyReview.Fold
import CchTopologyReview.Lifetime
import CchTopologyReview.Chunked
import CchTopologyReview.Fresh
import CchTopologyReview.Delta
import CchTopologyReview.Findings
import CchTopologyReview.Witness
import CchTopologyReview.RecomputeBridge
import CchTopologyReview.FitBridge

/-!
# CCH redesign review: the topology's invariants, rank allocation, MVCC isolation and the fold

**What is reviewed.** `.handover/cch-redesign/design/03-design.md`, revision 3.1, in the worktree on branch
`cch/graphblas-redesign-handoff` (based on `origin/main` d6fcd5aac). The parts reviewed: §2 in full (the data
structure table, "Why base plus delta", "Arc identity", "Contract changes": I1′, P, I3′, I7); §4 steps 1–5 as the
fresh build's contract; §6 (the classification table, rank allocation, node delete, id reuse, Built-entry jobs,
the fold job); §6b "MVCC"; §7 "Versions". Real code modelled: `WeightArr` (`graph/src/index/falkordb/cch/
customize.rs:112-240` at `origin/cch/12a-chunked-weights` b83a3564c, identical at `cch/08c-build-catch-up`
31ca1b802, which 03 cites as `customize.rs:182-189` @08c), and the `cow_btree` the deltas use (`graph/src/index/
falkordb/data_structures/cow_btree/` @main, through `proofs/cow_btree`).

Lake path dependencies: `proofs/cow_btree` (the deltas) and `proofs/cch_recompute_review` (RecomputeBridge.lean),
which requires `proofs/cch_supergraph_review` and, through it, `proofs/cch_redesign_review` (FitBridge.lean). No
cycle: redesign ← supergraph ← recompute ← topology.

The rest has no code yet. Each definition cites its 03 section, and each reading is listed below.

## The model

One version's hierarchy is a value `Cch α` (Model.lean): `Up.base`/`Down.base` as rows of `(col, id)`,
`Up.delta`/`Down.delta` as key-sorted `(row, col, id)` lists, `rank_of`, `node_of`, `free_ranks`,
`arc_capacity`, `added_since_fold`, the metric (abstract values `α`, preset `inf`), `LineageStats` and the copy's
data version. `Inv = RInv ∧ AInv ∧ Held` (Inv.lean) is the invariant. Two storage layers justify treating a version
as a value:
* `Chunked.lean` models the heap of `Arc`s under `WeightArr`, with `Arc::make_mut`.
* `Delta.lean` ties the delta to `proofs/cow_btree`'s tree.

Operations:
* `fresh` (§4): the Supergraph's output.
* `fork`: a writer's private copy at `version + 1`.
* `alloc`/`allocOne` (§6 "Rank allocation").
* `s2` (§6 S2 row, §2 "S2"): S2's append. Its new arcs come from `dhRun`, Davis–Hager's closure for any pop order,
  via `s2New`.
* `ndel` (node delete), `reuse` (node created or id reused), `mset` (a metric write by id).
* `fold` (§6 "Fold job").

Lineages:
* `WLin tF t`: writers only since `tF`, every step on a copy above `tF`'s version.
* `Lin`: writers and fitted folds, in any order.

## 03 section / Rust line ↔ Lean

| 03 / Rust | Lean |
|---|---|
| §2 `Up.base`, `Down.base` (`Ap`, `Aj`, `Ax`, value = id) | `Cch.upB`, `Cch.downB : Rows`, `bRow`, `inB` (Model.lean); `freshRows`, `ap` = `Ap[r]` (Fresh.lean) |
| §2 `Up.delta`, `Down.delta` (`CowBTree`, key `(row ≪ 32) \| col`, doc = id) | `Cch.upD`, `Cch.downD`, `Ent`, `keyLt`, `KSorted`, `sinsK`, `insAll`, `dRow`; `keyE`, `encE`, `key_eq` (Delta.lean) |
| §2 I3′ `up_ids(r)` (merge base and delta) | `mergeRow` (Basic.lean), `upIds`, `downIds` |
| §2 `rank_of`, `node_of` (`ChunkedArr`), `free_ranks` | `Cch.rankOf`, `Cch.nodeOf`, `Cch.free`; storage `Chunked.WA` over `T = Option Nat` |
| §2 Metric, `arc_capacity`, `LineageStats`, `added_since_fold` | `Cch.met`, `Cch.cap`, `Stats`, `Cch.added`, `keepAfter`, `ents` |
| §2 Arc identity "Fresh build", §4 steps 3–5 | `fresh`, `FreshIn`, `FreshOk`, `TransposeSpec` (Fresh.lean) |
| §2 Arc identity "S2", §6 S2 row ("sort … by (lo,hi), append ids, insert into both deltas and `added_since_fold`, preset the metric") | `s2`, `upEnts`, `S2Ok`; `dhUp`, `dhPush`, `dhRun`, `s2New`, `hasArc` (S2.lean) |
| §6 "Rank allocation" steps 1–4 | `allocOne`, `allocTop`, `alloc` (Model.lean) |
| §6 table: node delete; node created or id reused; S1 writes | `ndel`, `reuse`, `mset` |
| §5 "a writer's private version is `published.version + 1`" (`mvcc_graph.rs:301` @08c) | `fork`, `WLin` (Lineage.lean) |
| §6 "Fold job" | `fold`, `AddSpec` (GraphBLAS `eWiseAdd` + unload) (Fold.lean) |
| §6 "Built-entry jobs", S6's input order | `releaseLit`, `releaseFix` (Findings.lean) |
| `WeightArr { chunks: Arc<[Arc<[T]>]>, len }` `customize.rs:112-115` @12a | `Chunked.WA`, `Chunked.Heap`, `Chunked.St` |
| `CHUNK_BITS`, `CHUNK_LEN` `:126-127` | `CHUNK_BITS`, `CHUNK_LEN`, `cix`, `oix`, `cix_eq`, `oix_eq` |
| `Index::index` `:158-163` | `read` |
| `WeightArr::set` `:182-189` (two `Arc::make_mut`s) | `setW` = `mmTab` + `setW2` (`mmChunk`, `writeAt`) |
| `WeightArr::new_version` `:192-194`; `Metric::new_version` `:252-259` (each array) | `newVersion` |
| §2 "plus append" (PR 2's `ChunkedArr`, design) | `appendW` = `mmTab` + `appendW2` (`mmChunk`, `pushAt`) |
| `CowBTree::insert_batch` `cow_btree/mod.rs:241-255` @main | `CowBTree.insertBatch` (imported) in `delta_insertBatch` |
| `CowBTree::range_tuples` `mod.rs:408` (the `(key, doc)` cursor; "No caller today") | `CowBTree.rangeWith TupleExtract` (imported) in `delta_row_read` |
| `CowBTree::range` `mod.rs:314` (docs only) | `CowBTree.range` in `delta_snapshot_isolated` |
| `insert_batch`'s `assert!` `mod.rs:250-253`; `from_sorted`'s `mod.rs:204-207` | `CowBTree.SortedLe` in `down_batch_unsorted`, `fold_concat_unsorted` |

## Theorems

**1. I3′.**
* `upIds_strict`, `downIds_strict` (Props.lean; 03 §2 I3′): `up_ids(r)` and `down_ids(r)` are strictly
  ascending by column, so by `(hi, id)`.
* `mem_upIds`, `mem_downIds`: they hold exactly the base's and the delta's entries of row `r`.
* `mergeRow_strict` (Basic.lean): the merge of two strictly ascending rows with no column in common is strictly
  ascending.
* `delta_disjoint`: no key is in both the base and the delta. Every operation keeps it, because each keeps `Inv`:
  `fresh_inv`, `alloc_inv`, `s2_inv`, `fork_inv`, `ndel_inv`, `reuse_inv`, `mset_inv`, `fold_facts`.
* `delta_row_read` (Delta.lean): the delta row the merge reads is a `CowBTree::range_tuples` (`mod.rs:408`)
  over keys `[r ≪ 32, (r ≪ 32) | u32::MAX]`, decoded, in order. This uses cow_btree's `rangeTuples_spec`.
  **PR 3 note**: 03 §6b's delta cursor needs `range_tuples`; `CowBTree::range` (`mod.rs:314`) yields docs only
  (`rangeDocs_spec`), so it cannot give the column.
* `p_rows`, `p_transpose`: merged rows strictly ascending, above their rank, inside the rank space; `Down` is
  `Upᵀ` with the same ids, rows strictly ascending.
* `toTopo_valid`, `toTopo_downOK` (RecomputeBridge.lean): **cch_recompute_review's P-rows and P-transpose,
  formally**: for `toTopo t := ⟨t.nr, (upIds t ·).map Prod.fst, (downIds t ·).map Prod.fst⟩`,
  `Inv t → (toTopo t).Valid` and `Inv t → (toTopo t).DownOK` (its `Basic.lean:102-112`).
  `lineage_valid_downOK`: both hold at every state of every lineage from a fresh build.
  `lineage_scoped_agrees`: `CchRecompute.scoped_agrees` instantiated there, with **P-chordal still a premise**.

**2. Arc ids.**
* `ids_unique`: unique across base and delta.
* `key_determines_id`: one id per `(lo, hi)`.
* `ids_lt_cap`: below `arc_capacity`, and the metric's length is `arc_capacity`.
* `i7`, `i7_rows`, `id_names_one_pair` (03 §2 I7, strengthened): `Up(lo,hi) = id ⇔ Down(hi,lo) = id`. One id
  names one unordered pair across `Up` and `Down`.
* `upEnts_getElem?`: S2 numbers `N[i]` as `cap + i`.
* `s2_metric`: new arcs are preset to `inf`, and old entries are untouched.
* `fresh_ids`: on a fresh build, `id = Ap[lo] + i`.
* `no_reuse` (Lifetime.lean): an id held at any state names the same arc at every later state of the lineage,
  across writers and folds.
* `ids_only_appended`: `arc_capacity` never decreases. Arcs persist with their ids. A new arc's id is at or
  above the old `arc_capacity`.

**3. I1′ and P.**
* `node_of_inverse`, `rank_of_injective`, `node_of_injective` (Props.lean): `rank_of` and `node_of` are partial,
  injective inverses.
* `pool_iff`: `r ∈ free_ranks ⇔ node_of[r] = None` inside the rank space. `past_space`: above it, a rank has no
  holder and is not free.
* `pool_rows_empty`, `free_rows_empty`: a holderless rank has `up(r) = down(r) = ∅` (merged).
* `allocOne_inv`, `alloc_inv` (Alloc.lean): allocation keeps all of this.
* `allocOne_rule`: degree ≤ 1 with a free rank takes the lowest one; otherwise the top rank, index `nr`.
* `alloc_ranks_all`: after the phase, every endpoint has a rank. This is Lemma A's premise "every edge has an arc
  before the recompute", on the rank side.
* `allocOne_keeps_holder`: "taking a pool rank never strips a holder".
* `rank_never_returns`, `pool_only_shrinks`, `rank_of_stable` (Lifetime.lean): within a lineage, a holder keeps
  its rank, the rank is never free again, and the pool only shrinks. This includes node delete and id reuse,
  which keep the rank.
* `s2New_ok` (S2.lean): the new arcs of any Davis–Hager closure from the version's arcs plus any inserted pairs
  `lo < hi` of held ranks (one pair, every pair of a commit unit, or a whole adjacency for an Overflow rescan)
  satisfy `S2Ok`. Every arc joins two held ranks, lower to higher (`dh_ends`, `dh_lt`). That is P's "only ranked
  rows gain arcs", for every pop order.
* `fresh_inv`, `lineage_inv` (Fresh.lean): a fresh build satisfies `Inv`, so every state of a lineage that starts
  from one does.

**4. MVCC isolation.**
* `setW_spec` (Chunked.lean; `WeightArr::set` `customize.rs:182-189` @12a): on success, every other live version
  reads exactly what it read before, at every slot. The written version reads the new value at the slot and its
  old values elsewhere. `WF` is kept. The cases covered: the table shared or not, and the chunk shared or not;
  this includes a chunk the table holds once while another table holds it too.
* `appendW_spec` (design append): every other version reads what it read before, `WF` is kept, and the
  appended version's length grows by 1. What the appended version itself reads is not stated: that needs a
  shape invariant (table length, full chunks but the last) the model does not carry.
* `newVersion_spec`, `dropV_spec`, `perm_wf`: taking, dropping and reordering versions keep `WF` and every read.
  Chained, these give isolation across any history.
* `setW_isolated`, `appendW_isolated`: the other versions' halves.
* `read_frame`: a version's read depends on the heap only through its own table and its chunks.
* Deltas, imported from proofs/cow_btree:
  * `delta_snapshot_isolated` (`snapshot_reads_committed`, which is cow_btree's `range_spec`): a reader's root
    reads its own entries. In the value model that is immediate; isolation rests on cow_btree's argued fact
    that pages are never written in place (its Snapshot.lean).
  * `delta_path_copy` (`insertOne_shares_siblings`): an insert copies only the routed path.
  * `delta_insertBatch` (`insertBatch_wf`): `insert_batch` of S2's `Up` batch (sorted by key,
    `upEnts_keys`) is the abstract insert and keeps `TreeWF`.
  * `delta_insertBatch_sorted`, with `insAll_sorted_batch` and `s2_batches`: the `Down` batch, sorted by key
    first, gives the same delta as the model's unsorted insert. This is T1's fix, proved.
* Bases: `step_shape`, `wlin_shape` (Lineage.lean). No writer step writes a base. Only a fold replaces one, with
  new arrays (`fold_keeps` lists what it keeps).

**5. Fold equivalence.**
* `fold_rows` (Fold.lean; 03 §6 "Fold job"): after a fold, every rank's `up_ids` and `down_ids` are identical:
  same ranks, same ids, same order.
* `fold_keeps`: the metric, `rank_of`, `node_of`, `free_ranks`, `arc_capacity`, `LineageStats` and the version
  are the published copy's.
* `fold_new_delta`: the new delta is exactly the copy's arcs that are not in the fork's delta, which are the
  arcs of `added_since_fold` with `seq > F`. `added_since_fold` keeps only those.
* `fold_facts`: the fold keeps `Inv`.
* `fold_view`: a reader sees no difference (`ReaderView`).
* `fold_of_published`: a fold of a published state onto itself reads as that state, with empty deltas.
* The same theorem covers "a writer that overtook a fold applies the same function to its own copy": that copy is
  also `WLin`-reachable from `F`.

## Premises (each theorem names the ones it takes)

* **FoldFit** (`WLin tF t` in the fold theorems). Only writers ran between the fold's fork and its publish. Each
  ran on a private copy above `F`.
  * The first half, in the write-log model, is **formal** (FitBridge.lean), for any run (`Trace`) of
    `cch_redesign_review`'s model from the state a job forked at to the state it publishes onto, fitted:
    * `fitted_job_no_builder_publish`: every step in the window that keeps the data version leaves this
      index's entry unchanged, so no batch or job of this index publishes;
    * `fitted_job_only_writer_steps`: every step in the window is a writer step (the version moves by one) or
      leaves the entry unchanged;
    * `fitted_window_writer_no_rebase`: every writer forking in the window starts from the published entry and
      does not rebase (`writerE` is its own segments on the published entry, then retention);
    * `window_gen`, `trace_gen_mono`: the window stays in the fork's generation.
    They are built from that project's `fitted_job_only_writers_between`, `builder_publish_bumps_commits`,
    `commits_monotone`, `step_gen_mono`, `job_window_frozen` and `upkeep_writer_not_overtaken`. (A job publish
    clears `upkeep`; the next writer that sets it draws a fresh ticket.)
  * **Argued, not proved: only the cross-model step.** That the write-log model's writer steps are this
    model's `fork` followed by `Step`s, i.e. `WLin`. The two models have different state types (the write-log
    model abstracts the topology to `res`) and are not composed.
  * The second half is 03 §5 (`mvcc_graph.rs:155-160, 301` @08c): writers are serialized and fork at the latest
    data version.
  * `B3_fold_loses_unforked_write` shows the second half is needed.
* **AddSpec**, **TransposeSpec**: GraphBLAS C contracts, as hypothesis structures (not axioms).
  * AddSpec: `eWiseAdd` with disjoint patterns, then an unload that is not jumbled.
  * TransposeSpec: `GrB_transpose`, then unload.
  * `cch_supergraph_review` proves I7 from its own `TransposeContract`.
* **FreshOk**: Order's and Supergraph's output. `π` is injective. Rows are ascending, above their rank, and
  join held ranks (03 §4 "Contract", step 5 "P holds by construction"). `cch_supergraph_review` proves the game's
  rows ascending and the row-major ids for today's game. Injectivity of `π` and held endpoints are named and not
  discharged here.
* **S2Ok**: the append's input. Every field is discharged by `s2New_ok` for any Davis–Hager closure seeded with
  the version's arcs and any inserted pairs of held endpoints (`alloc_ranks_all` gives the ranks). Not discharged: that the closure ends at a valid topology (I2,
  I4–I6, chordality; DH01 Thm 4.2, [R2 §2]). That is cch_recompute_review's **P-chordal after S2**, still open.
* **std `Arc::make_mut`**: it clones exactly when the strong count is not 1. The count is derived from the holders
  (`cntT`, `cntC`), so this is the only Arc fact assumed.
* **cow_btree pages are never written in place.** This is proofs/cow_btree's Snapshot.lean argument
  (`make_private` always clones; it is proven in proofs/index_layer). It is what lets a delta be a value. The
  delta refinement also takes `TreeWF`, which `Reachable_wf` gives for every tree the API builds, and `Fits`
  (u32 rows, columns and ids).

## Readings chosen (where 03 is ambiguous)

* **R1** (§2, §6 step 3). Ranks are 0-based indices, as `Rank::index`. 03's "next top rank `n_r' + 1`" is index
  `nr`, and the pool is `0 … reserve-1`.
* **R2** (§2 P). "`r ∈ free_ranks ⇔ node_of[r] = None`" is read inside the rank space `r < n_r'`. Above it,
  `node_of` is `None` and the rank is not free (`past_space`).
* **R3** (§2 I3′). "Ascending `(hi, id)` pairs" means strictly ascending by `hi`. Ids are not ascending within
  a row after an S2 (`B2_positional_id_breaks_after_s2`), as §2 "No positional writes" says.
* **R4** (§2 P). `up(r)`, `down(r)` are the merged base + delta rows.
* **R5** (§2 delta). One id per `(row, col)` key is an invariant of the design (`KSorted`), not a cow_btree
  property: cow_btree holds `(key, doc)` pairs and would hold two docs for one key.
* **R6** (§6 S2 row). S2's new arcs are the closure's arcs that are not arcs yet (`s2New`), sorted by `(lo, hi)`.
  The closure is modelled for any pop order and number of pops.
* **R7** (§2 "S2", §6). Every pass appends `(seq, arcs)` to `added_since_fold`, empty or not. `seq` is the
  copy's version.
* **R8** (§2 `LineageStats`). "Arcs added by S2 at pool-ranked rows" counts new arcs with `lo < reserve`.
* **R9** (§6 S2 row; §6 "Fold job"). The `Down` batch is sorted by `(hi, lo)` before `insert_batch`. The fold's
  new deltas are built by sorted insertion. See T1.
* **R10** (§2 table vs §6). §2 says `added_since_fold` is "cleared by a fold". It is read as §6: it "keeps only
  those arcs" (`seq > F`).
* **R11** (§6 table). Node delete changes only `deletions` in the topology; its S1 writes are `mset`. A created or
  reused id changes nothing ("A reused id keeps its old rank and its arcs").
* **R12** (§6 step 2). The allocation takes `E` as `(node, degree)` in id order. The degree comes from the
  resolved pairs and is not modelled.
* **R13** (§2 "plus append"). An append copies the table if it is shared. A full last chunk is followed by a new
  chunk `[x]`. Otherwise the last chunk is copied if shared, then extended (as walkthrough SIMPLIFICATIONS #28).
* **R14** (§6 S6). Releasing a rank also clears `rank_of[x]` and adds the rank to `free_ranks`. See T2.
* **R15** (§6 "Fold job"). "A fold of a published state" is the fold applied to the published copy itself
  (`fold_of_published`).
* **R16** (§6 "Fold job", §2 "a rank above the base's dimension can live in the delta"). Rows are unbounded
  lists here (`bRow` past the end is `[]`), so `AddSpec` hides the dimensions: the base is `n_r` wide, the delta
  `n_r'`, and the fold's new base needs `Ap` of length `n_r' + 1`. See T3.
* **R17** (§6 S2 row, §5 Overflow). The closure is seeded with the version's arcs and any inserted pairs of held
  ranks: one pair, a unit's pairs, or a rescan's whole adjacency (`s2New_ok`).

## Findings (design-level, no code to reproduce)

* **T1 (minor; implementation note for PR 3).** 03 §6's S2 row says to sort the new arcs by `(lo, hi)` and
  insert them into both deltas.
  * Sorted that way, the `Down` entries (key `(hi ≪ 32) | lo`) are not sorted, and `insert_batch`'s `assert!`
    (`cow_btree/mod.rs:250-253` @main) panics (`down_batch_unsorted`: `N = [(0,3), (1,2)]`).
  * Likewise, the fold's "inserted into empty `cow_btree`s" from several passes' arcs is unsorted for
    `from_sorted`'s `assert!` (`mod.rs:204-207`) (`fold_concat_unsorted`).
  * Fix: sort the `Down` batch by `(hi, lo)`, and sort the fold's arcs (or insert one at a time).
    `delta_insertBatch_sorted` proves the sorted `Down` batch gives the intended delta.
  * Applies only if PR 3 uses `insert_batch` or `from_sorted`. 03's costs ("`cow_btree` inserts (O(new arcs ×
    log n))", §6b "Where it runs") read as single `insert`s, which take any order.
* **T2 (major if read literally).** 03 §6 says S6's input order releases ranks: "Ranks whose holders have no
  indexed edge at F go back to the pool (`node_of = None`), which keeps P".
  * Read as a patch of `node_of` alone, it breaks I1′: `rank_of[x]` still names the rank
    (`T2_literal_release_breaks_I1`).
  * The next degree-1 node then takes that free rank, so two nodes share it. When `x` regains an edge, the
    allocation skips it as already ranked, and its pairs are classified at the other node's rank
    (`T2_literal_release_then_alloc_shares_a_rank`). Wrong hierarchy, wrong distances.
  * Fix: "(`node_of[r] = None`, `rank_of[x] = None`, `r` into `free_ranks`)". `T2_release_keeps_rinv` proves
    that this keeps I1′ and P's first half. Building S6's lineage with `fresh` from the released holders does
    the same, since `fresh` derives `rank_of` as the inverse. (`releaseLit` keeps the pool ascending with
    `insN`, so the counterexample does not rest on an unsorted pool.)
* **T3 (minor; design gap).** The fold's `eWiseAdd` (03 §6 "Fold job") is over a read-only load of the base,
  `n_r × n_r` with `Ap` of `n_r + 1` entries, and a matrix of the delta, which holds rows and columns of top
  ranks up to `n_r'` (03 §2: top ranks live only in the deltas). GraphBLAS needs equal dimensions, and the new
  base needs `Ap` of `n_r' + 1` entries. 03 does not say how. Fix: load the base at `n_r'` (extend `Ap` with
  copies of its last offset, which adds empty rows) before the `eWiseAdd`. The model's unbounded rows hide this
  (R16).
* **Wording.** P should say "for `r < n_r'`" (R2). §2's "cleared by a fold" should say "keeps the arcs with
  `seq > F`" (R10).

## Breaks (each premise or rule the proofs use is needed; Findings.lean)

* `B1_merge_needs_disjoint`: without disjointness, `up_ids` repeats a column.
* `B2_positional_id_breaks_after_s2`: after one S2, position `i` of a row is not id `Ap[lo] + i`. Every I3′
  reader 03 lists must read the id: relax's `x_slots_start + i` (`customize.rs:497-513` @08c) and `query.rs:113`.
* `B3_fold_loses_unforked_write`: an S2 at `seq = F` (no fork) is lost by the fold.
* `B4_fold_ge_duplicates`: keeping `seq ≥ F` instead of `> F` duplicates an arc in the new base and the new delta.

## Witness (Witness.lean): the premises are satisfiable together

A fresh build of 4 ranks (3 held, 1 pool) with game rows `[[1,2],[2],[],[]]`:
* `w_fresh_ok`, `w_transpose`, `w_inv0`: FreshOk, TransposeSpec and `fresh_inv` hold.
* `w_alloc`: a fork's allocation puts node 13 on pool rank 3.
* `w_s2ok`, `w_s2_id`: an S2 adds arc `(2,3)` with id 3, and `S2Ok` holds.
* `w_lin`, `w_inv2`: `Lin` reaches it, and `Inv` holds.
* `w_addUp`, `w_addDown`: `AddSpec` holds for `[[(1,0),(2,1)],[(2,2)],[(3,3)],[]]` and its transpose.
* `w_fold_rows`: `fold_rows` instantiates.
* `w_st0_wf`, `w_chunked`: a `WF` `WeightArr` state where two versions share a table; `setW` succeeds, copying
  the table and the chunk; the other version still reads 2, the written one reads 7.

## For the other review projects

* **`cch_recompute_review` P-rows and P-transpose: discharged formally** for the redesign by
  `CchTopo.Bridge.toTopo_valid` and `toTopo_downOK` (`Inv t → (toTopo t).Valid`, `Inv t → (toTopo t).DownOK`), at
  every state of a lineage that starts from a fresh build (`lineage_valid_downOK`), under FreshOk,
  TransposeSpec, AddSpec and FoldFit. `lineage_scoped_agrees` instantiates `CchRecompute.scoped_agrees` there;
  `CchRecompute.lemmaA` takes `H'.Valid` and `H'.DownOK` the same way.
* **`cch_recompute_review` P-chordal after S2**: **not** discharged, and still a premise of
  `lineage_scoped_agrees` and of `lemmaA`. Only the closure's endpoint and direction properties are proved
  (`dh_ends`, `dh_lt`), not that it reaches the fixpoint.
* **`cch_redesign_review`'s gaps** "the fold's bases and deltas (`added_since_fold` with `seq > F`)" and "ranks,
  I1′, P": covered for any sequence of §6 steps and fitted folds from a FreshOk build (`fold_rows`,
  `fold_new_delta`, group 3). FreshOk's `inj` and `arcs` are not discharged, and this model is not composed with
  the write-log model's transitions. FitBridge.lean proves, in that model, that only writers run between a
  fitted job's fork and its publish, each starting from the published entry without a rebase.

## Rows to move (stack code, not on `origin/main`; for when `cch/12a-chunked-weights` lands)

They are in `native_index/COVERAGE.tsv` on that branch as NOT COVERED ("not modelled yet: CCH chunked
copy-on-write weight arrays (12a)"):
* `customize.rs  Index::index  158  PROVEN  CchTopo.Chunked.read, read_frame`
* `customize.rs  WeightArr::set  182  PROVEN  CchTopo.Chunked.setW: setW_spec (isolation, own read, WF)`
* `customize.rs  WeightArr::new_version  192  PROVEN  CchTopo.Chunked.newVersion: newVersion_spec`
* `customize.rs  Metric::new_version  252  PROVEN  newVersion_spec on each of the four arrays`

Not modelled: `WeightArr::from_vec` (`:169`), `memory_usage` (`:199`), `Buffers::into_metric`, and
`Cch::new_version` (`:610`, which also clones the topology `Arc`). No `origin/main` function is newly modelled
here: the `cow_btree` rows are proofs/cow_btree's, and `order.rs` (today's fresh ids, `arc_slot`) is
proofs/cch_supergraph_review's.

## Gaps

* **The closure's validity**: chordality, I2, I4–I6 after S2; Lemma A/B; S5's deletions.
* **S6/S7 topology**: new lineages; `fresh` covers their start.
* **Not modelled**: decay triggers and their thresholds, the full-seed pass, metric values, budgets.
* **The write log and `upkeep`**: in proofs/cch_redesign_review.
* **The CSR arrays** are rows here. `fresh_ids` ties ids to `Ap`. The read-only GraphBLAS load and the M2 read
  paths are not modelled.
* **u32 overflow**: 03 gives no rule for `arc_capacity` or the rank space reaching `2^32` (`Ax: u32`, ranks
  `NonZeroU32`; 03 §2 "while the arrays are 32-bit (M3)"). The delta refinement assumes `Fits`. The abstract
  model uses `Nat`.
* **`WeightArr` heap model**:
  * Versions are a multiset; the written one is moved to the front, which `perm_wf` allows.
  * Garbage objects stay in the heap. `Weak` is not used.
  * `from_vec` is not modelled; a fresh `from_vec` allocates only fresh addresses, so it cannot alias.
* **The deltas' page-level immutability** is cow_btree's argued fact (its Snapshot.lean), not re-proved.

`bash proofs/lean_ci.sh cch_topology_review` passes, with no `sorry`, `admit`, `native_decide` or `axiom`.
-/
