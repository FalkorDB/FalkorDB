import OpsTraverse.Basic
import OpsTraverse.CondTraverse
import OpsTraverse.BidirDedup
import OpsTraverse.VarLen
import OpsTraverse.ShortestPaths
import OpsTraverse.Scans
import OpsTraverse.Drive
import OpsTraverse.IndexScan
import OpsTraverse.VarLenGlue
import OpsTraverse.CTGlue
import OpsTraverse.ASPFold
import OpsTraverse.ASPInv
import OpsTraverse.ASPStep
import OpsTraverse.ASPFinal
import OpsTraverse.ASPMain
import OpsTraverse.ASPFuel
import OpsTraverse.SlotFeed
import OpsTraverse.PathVlt
import OpsTraverse.ExpandIntoPending

/-
# Traversal operators: CondTraverse, ExpandInto, var-length, scans, allShortestPaths, PathBuilder

**Re-targeted to main b29a2885c** (from 5e475e6c9): #3463 adds the pending-deleted endpoint
check to `ExpandIntoOp::expand_row` (`expand_into.rs:158-160`); `expandIntoP` is now the fixed
model and satisfies `LiveExpandInto` (`expandIntoP_live`), old counterexamples renamed
`pre3463_*`; `expand_into.rs` pins after `:158` moved +7. New open item: CondTraverse ignores
`Pending` (`bug_condTraverse_*`).

**Re-targeted to main 5e475e6c9** (from b84473a7f). #2383 (index serves `IN [...]` natively): the node and
edge index scans now hand the evaluated list to the index as ONE `InList` instead of an `Or` of `Equal`s
(`IndexScan.evalIQ_inList`, new `IQ.inListV`); `can_utilize_index` gained an `InList` arm (non-empty and every
member indexable), proved to decide exactly as the `Or` of `Equal`s it replaced (`canUse_inListV_eq_or`,
`canUseE_inListV_eq_or`, so `IN [NULL]` still falls back to a label scan). W2-index-3 (`IN [date, 2]` drops the
date member before the index is asked) is unchanged and still open. `node_by_index_scan.rs` `can_utilize_index` /
`is_indexable` / `next` re-pinned :251/:254/:300, `edge_by_index_scan.rs` `next` :316. #3436 (`rt.g.borrow()` ->
`rt.graph()`) touches `cond_traverse`, `cond_var_len_traverse`, `all_shortest_paths`, `expand_into`, the label/id
scans and `eval_shortest_path` without changing what they compute; no other pin moved.

**Re-targeted to main b6567cb97** (from 737dfb5da): no change to the traversal
operators. #2954 grew the `ListComprehension` arm of `eval.rs`, shifting the
`eval_shortest_path` / `bfs_shortest_path` rows by +27 (COVERAGE.tsv re-pinned).

Earlier: re-targeted to main 737dfb5da (from 289c201fe): no change to the traversal
operators. #2899 (`e63165cb5`) rewrote `RangeIter` in `eval.rs`, shifting the
`NeighborIter` / `shortestPath` rows by +10 (COVERAGE.tsv re-pinned); #2950
(temporal.rs) and #3041 (parser) touch nothing here.

Earlier: re-targeted to main 289c201fe (from fb3d516d0): no change to the traversal
operators; COVERAGE.tsv line numbers in `eval.rs` and `algo_procedures.rs` re-pinned.

A Lean 4 model of FalkorDB-rs's pattern-matching operators, checked against
openCypher's semantics (relationship isomorphism inside one MATCH, direction,
self-loops, `*min..max` including `*0`) and against C FalkorDB on live servers.
201 theorems, no `sorry` / `admit` / `axiom`. The GraphBLAS matrix product is
the only FFI primitive the proofs lean on, and it enters as a *hypothesis*
(`CondTraverse.MxmSpec`: the GrB_mxm ANY_PAIR spec), discharged once by the
reference `mxmRef`. Per-function buckets are in `COVERAGE.tsv`; repros are in
`graph/tests/lean_ops_traverse.rs` (`cargo test -p graph --test lean_ops_traverse`).

## What is modelled

| here | there |
| --- | --- |
| `Basic.pairs`, `edgesBetween`, `step` | relationship-matrix pair iterator, `Tensor::get`, `Graph::get_node_relationships_by_type` (`graph/graph.rs:2079`) |
| `CondTraverse.processPairs`, `expandRow` | `CondTraverseOp::process_pairs` / `expand_row` (`runtime/ops/cond_traverse.rs:944`, `:749`) |
| `CondTraverse.expandInto` | `ExpandIntoOp::expand_row` (`runtime/ops/expand_into.rs:125`) |
| `CondTraverse.MxmSpec`, `chain` | `Matrix::delta_lmxm` + `expand_batch` (`cond_traverse.rs:443-742`) |
| `CondTraverse.twoHop` | two chained traverses with sibling uniqueness (`ops/mod.rs:155`) |
| `BidirDedup.dedupRows`, `dedupBatches` | the `bidir_dedup` block (`cond_traverse.rs:914-935`, reset `:1176`) |
| `VarLen.dfs`, `varLen` | `VarLenIter::begin_start_node` / `advance` (`runtime/ops/cond_var_len_traverse.rs:152`, `:196`) |
| `ASP.relax`, `bfs`, `back`, `asp` | `AllShortestPathsOp::expand_row` (`runtime/ops/all_shortest_paths.rs:85-305`) |
| `ASP.buildNodes` | `PathBuilderOp::next`, `Value::List` branch (`runtime/ops/path_builder.rs:148-172`) |
| `PathVlt.vltMain`, `revWalk` | `PathBuilderOp::next`, `Value::Path` branch (`path_builder.rs:88-139`, direction inferred at `:102-105`) |
| `EIP.expandIntoP`, `Pending` | `ExpandIntoOp::expand_row` with pending deletes (`expand_into.rs:158-235`; `pending.rs:1115-1134`) |
| `Scans.labelScan`, `idSeek`, `labelIdScan` | `node_by_label_scan.rs:77`, `node_by_id_seek.rs:55`, `node_by_label_and_id_scan.rs:59` |
| `ASPFold.Rel`, `ASPInv.Inv`, `bstep`, `ASPStep`, `ASPFinal`, `ASPMain`, `ASPFuel` | the BFS of `all_shortest_paths.rs:141-265` (one `pop_front` iteration = `bstep`), its invariant, and the backtrack `:278-304` |
| `Drive.drive`, `capDrive`, `driveE` | the emitter-driven `next` loops (scans, CondTraverse with `trim_to_cap`, ExpandInto, CondVarLenTraverse, AllShortestPaths) |
| `CTGlue` | `CondTraverseOp::new`/`build_state`/`trim_to_cap`/`out_alias_id`/`null_pad`/`next`, `ExpandIntoOp::new`, scan/builder constructors |
| `VarLenGlue` | `path_value`, `VarLenIter::next` (frame-tree machine), `expand_row` endpoint plan, `new` |
| `SlotFeed.aggRow`, `inputRow`, `declare` | how an Aggregate row feeds a later CondTraverse across `WITH`: `aggregate.rs:1150-1213`, `cond_traverse.rs:648-652,764-777`, binder slot ids `binder.rs:2155-2269` |
| `IndexScan` | node/edge index scans (`evaluate_index_query`, `can_utilize_index`, `next` with the RediSearch contract `IndexSound` as hypothesis), fulltext and vector scans |

## What is proved (plain English)

* Directed single hop, emit mode: a row `(from,to,e)` comes out iff `e` is an
  edge `from→to` not bound to a sibling alias (`emit_directed_iff`).
* Undirected single hop: stored orientation, plus the reverse for non-loops —
  a self-loop matches once (`emit_bidir_iff`); ExpandInto likewise
  (`mem_expandInto_emit`).
* Collapse (anonymous edge) only emits rows emit mode would, one per matrix
  pair when directed (`collapse_sub_emit`, `collapse_directed_iff`,
  `collapse_directed_keys_nodup`).
* Batched F·A path = per-row collapse, duplicate-free; a fused chain returns
  `(row,dst)` iff a walk through the chain's matrices exists — for *any* mxm
  meeting the GraphBLAS spec (`batched_eq_collapse`, `fused_chain_semantics`,
  `chain_nodup`, `mxmRef_spec`).
* Two chained hops in emit mode = the openCypher match set with `r ≠ s`
  (`twoHop_emit_iff`).
* Var-length: the result set is exactly the trails from the start with length
  in `[min,max]` (`varLen_iff` = `varLen_sound` + `varLen_complete`); `*0..0`
  is the start node alone (`zero_zero`).
* allShortestPaths: every predecessor the BFS records is a traversable edge
  (`bfs_ok`), the backtrack yields reversed walks (`back_sound`), so every
  non-cycle result is a walk src→dst (`asp_sound`). For `src ≠ dst`, `min_hops = 1`:
  the BFS keeps TRUE shortest distances, a two-level FIFO queue and closed
  predecessor sets (`Inv`, `inv_bfs`); every returned path is no longer than any
  walk src→dst (`asp_minimal`); every shortest walk of ≤ `max_hops` edges is
  returned (`asp_complete'`), exactly once (`asp_nodup`); the BFS drains within
  `|nodes| + 1` iterations (`fin_drained`).
* The emitter loops conserve rows and apply LIMIT exactly (`drive_flatten`,
  `capDrive_flatten`); the var-length frame machine yields every frame's items
  (`run_complete`); index scans are sound pre-filters under the RediSearch
  contract (`scanRow_spec`); edge index scans honour bound endpoints (`edgeRow_sound`) and, since #3081
  (`ff3d24ba7`), read an undirected pattern both ways round, a self-loop once
  (`orient_mem`, `orient_length`; historical `pre3081_undirected_once`).
* PathBuilder's "other endpoint" rule reconstructs any walk's nodes when edge
  ids are unique (`buildNodes_walk`).
* The bidir dedup keeps unique keys and never loses a key (`dedupRows_keys_nodup`,
  `dedupRows_keys_complete`) — the bugs are in *which* key it uses.
* The id-range label scan returns exactly labelled ∩ range for a sorted
  `get_nodes` (`labelIdScan_correct`).
* Re-target to origin/main 8743953a8 (#2390, d2c42e032 "plan an inline property map once"):
  CondTraverse / ExpandInto no longer evaluate inline attrs per row (the planner lowers
  them to a `Filter` and sets `emit_relationship` for an edge predicate), so the models'
  attr-free `processPairs` / `expandInto` are now literal; the batched F·A path lost its
  inline-attr gate (`ctNew_eligible_iff`, `pre2390_eligible_le`: it only widened); and a
  var-length walk's absorbed edge filter now raises a type error on a non-boolean value,
  exactly as `FilterOp` (`edgeVerdict_eq_filterRow`; was a silent skip, `pre2390_edgeVerdict_differs`).

## Confirmed bugs (Rust repro fails; C and/or openCypher disagree)

1. **Unreferenced named edge collapsed while a sibling traverse reads it for
   uniqueness** — `planner/optimizer/reduce_expand_into.rs:30-55` (its
   `ir_references_variable` ignores ancestor `sibling_edges`).
   `MATCH (a)-[r]->(x)<-[s]-(c) RETURN count(*)` on `a⇉b→c`: Rust 1, C 2 (Cypher 2),
   while `RETURN r.w, s.w` lists 2 rows. Lean `twoHop_collapse_changes_count`.
   Test `bug_unreferenced_edge_collapse_breaks_sibling_uniqueness`. Related to #2896.
   **Still present on 8743953a8** (re-checked live: Rust 1, C 2): #2390 moved the oracle to
   `optimizer/references.rs`, which still classifies `CondTraverse`/`ExpandInto` (and so their
   `sibling_edges`) as reading nothing.
2. **Bidirectional dedup of chained anonymous undirected hops drops rows**
   (`cond_traverse.rs:264-290, 947-968, 1267`): key ignores other columns
   (`UNWIND [1,2] AS x MATCH (a)-[]-()-[]-(b)`: Rust 5 rows, C 10), uses an
   intermediate node as source for ≥3 hops (`(a)-[]-()-[]-()-[]-(b)`: Rust 7
   pairs, C 12), and is reset per batch (K(2,3000): Rust counts 3/4, C 1).
   Lean `drops_outer_rows`, `three_hops_collide`, `batch_split_changes_result`.
   Still present on 8743953a8 (live: `UNWIND [1,2]` 5 vs C 10; 3-hop star 7 vs C 12).
3. **Undirected anonymous collapse is per direction, not per pair**
   (`cond_traverse.rs:849-909`, `expand_into.rs:183-222`): `a→b, b→a, a→b`,
   `MATCH (a)-[]-(b) RETURN a.id, b.id` Rust 4 rows, C 2.
   Lean `bidir_collapse_duplicates_pair`. Still present on 8743953a8 (live: 4 rows vs C 2).
4. **Consecutive MATCH clauses are merged into one pattern**
   (`parser/cypher.rs:1549-1556`, outside the target files but it decides
   uniqueness scope): `MATCH (a)-[r]->(b) MATCH (c)-[q]->(b)` Rust 2 rows, C 4
   (Cypher 4); `OPTIONAL MATCH (y:Nope) MATCH (z:B) RETURN y, z.id` Rust
   `null|null` (Cypher `null|2`; C rejects the query).
5. **allShortestPaths on a directed cycle returns the edge list backwards**
   (`all_shortest_paths.rs:284-288`): `0→1→2→0`, `(a)-[*]->(a)`: Rust nodes
   `[0,2,1,0]` rels `[3,2,1]`, expected `[0,1,2,0]`/`[1,2,3]` (C `[0,2,1,0]`/`[1,2,3]`,
   also wrong). Lean `directed_cycle_backtrack_order`, `directed_cycle_not_a_walk`.
6. **algo.SPpaths / SSpaths default relationship cost 0** (`functions/algo_procedures.rs:2308,
   2397, 2518`; C: 1 per edge): `maxCost: 1` without `costProp` returns
   `[0,1,3]` cost 0 (C `[0,3]` cost 1). Test `bug_sppaths_default_cost_is_zero`.
7. Out of target: `[x IN null | x]` returns `[]` (C `null`).

8. **#3372: a grouping key's pre-projection slot leaks past `WITH`**
   (`aggregate.rs:1187-1196` re-binds `c` at its scope-0 id; the binder numbers
   each scope from 0, `binder.rs:2261`; CondTraverse reads the stale node as a bound
   destination, `cond_traverse.rs:648-652,771-777`). `MATCH (a)-[:E]->(b)-[:E]->(c)
   WITH c, count(*) AS k MATCH (c)-[:E]->(d)`: old `c` and new `d` are both slot 2,
   Rust 0 rows (expected 4). Property `SlotFeed.ScopedRow`; Lean
   `bug3372_slot_reuse_drops_rows`, `bug3372_split_product_zero`,
   `bug3372_aggregate_leaks_preprojection_slot`, `bug3372_shift_probes`.

9. **#3451: incoming var-length cycle in a named path is not reversed**
   (`path_builder.rs:102-110`: the direction is inferred by comparing the VLT path's
   first node with the previous path node; on a cycle `n = m` it says "forward").
   `MATCH p=(a {id:0})<-[:R*3..3]-(a)` on `0→1→2→0`: Rust `[0,1,2,0]`/`[1,2,3]`,
   expected `[0,2,1,0]`/`[3,2,1]` (C `[0,2,1,0]`/`[1,2,3]`). Property
   `PathVlt.PathFollowsPattern`; Lean `bug3451_main_violates`,
   `bug3451_cycle_not_reversed`, `bug3451_expected`, `bug3451_noncyclic_ok`; main is
   right on every non-cyclic walk (`vltMain_follows_of_ne`); recording the direction
   at plan time fixes it (`vltFix_follows`).

10. **#3452: ExpandInto matched an edge whose endpoint is pending-deleted** — FIXED by
   #3463 (b29a2885c): `expand_into.rs:158-160` now yields no row when either endpoint
   is `is_node_deleted`. Property `EIP.LiveExpandInto` is proved for main
   (`EIP.expandIntoP_live`); the old counterexamples are `EIP.pre3463_violates`,
   `pre3463_matches_doomed_edge`, `pre3463_collapse_matches_doomed_edge` (on
   `expandIntoPre`).

11. **#3463 follow-up: CondTraverse still matches an edge of a pending-deleted node**
   (`cond_traverse.rs` never reads `Pending`; `expand_row` `:749`, `process_pairs` `:944`).
   `MATCH (a:A) DETACH DELETE a MERGE (a)-[r:R]->(x:B)` takes the CondTraverse path
   (`x` unbound) and matches the doomed edge. Property `EIP.LiveCondTraverse`; Lean
   `EIP.bug_condTraverse_main_violates`, `bug_condTraverse_matches_doomed_edge`. Not fixed.

Gaps closed 2026-10-22: PathBuilder's `Value::Path` (var-length) branch is modelled
(`PathVlt.lean`, catches #3451); ExpandInto reads pending deletes (`ExpandIntoPending.lean`,
catches #3452; `expandIntoP_empty` keeps the earlier theorems as its no-pending case).

Gap closed 2026-10-20: operator composition across `WITH` (Aggregate → CondTraverse)
is now modelled at slot level (`SlotFeed.lean`), which catches #3372.

## Shared deviations from openCypher (C behaves the same or worse)

* Anonymous edges and named-but-unreferenced edges collapse parallel edges
  (`MATCH (a)-[r]->(b) RETURN count(*)` = 2 with 3 edges) — C semantics.
* No relationship uniqueness for var-length hops against sibling edges
  (`VarLen.no_sibling_uniqueness`), nor for anonymous edges or named paths.
* Undirected allShortestPaths cycles reuse their first edge
  (`undirected_cycle_repeats_edge`; test `shared_with_c_undirected_asp_cycle_repeats_edge`).
* Where Rust is right and C wrong: undirected self-loops under `*` (C twice),
  isomorphism of named edges within one MATCH (C ignores it), allShortestPaths
  undirected self-loop (C duplicates).

## Gaps / assumptions

* Labels and WHERE edge filters are pure filters and are abstracted away;
  relationship types are a pre-filtered edge list. Inline property maps no longer
  reach the fixed-length operators at all (#2390: stripped by the planner); a
  var-length walk still prunes on its edge's own attrs, which `advance` checks per edge
  and the model abstracts like the WHERE filter.
* Iterator `seek` (forward / transposed) is modelled as a filter of the full
  pair list; emission *order* (stack vs recursion, `swap_remove`) is not modelled.
* `get_nodes` (label-matrix intersection) is assumed sorted ascending.
* allShortestPaths minimality/completeness are proved for the non-cycle case only;
  the cycle case (`src = dst`) has the known ordering bug and is not claimed.
* `shortestPath()` (bidirectional BFS in `runtime/eval.rs:1421`) and the
  `algo.*paths` procedures are differential-tested against C only.
* Record caps, OPTIONAL null padding and emitter loops are modelled at row level
  (the emitter's batch packing is proved in proofs/columnar).
-/
