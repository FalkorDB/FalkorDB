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

/-
# Traversal operators: CondTraverse, ExpandInto, var-length, scans, allShortestPaths, PathBuilder

A Lean 4 model of FalkorDB-rs's pattern-matching operators, checked against
openCypher's semantics (relationship isomorphism inside one MATCH, direction,
self-loops, `*min..max` including `*0`) and against C FalkorDB on live servers.
166 theorems, no `sorry` / `admit` / `axiom`. The GraphBLAS matrix product is
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
| `Scans.labelScan`, `idSeek`, `labelIdScan` | `node_by_label_scan.rs:77`, `node_by_id_seek.rs:55`, `node_by_label_and_id_scan.rs:59` |
| `ASPFold.Rel`, `ASPInv.Inv`, `bstep`, `ASPStep`, `ASPFinal`, `ASPMain`, `ASPFuel` | the BFS of `all_shortest_paths.rs:141-265` (one `pop_front` iteration = `bstep`), its invariant, and the backtrack `:278-304` |
| `Drive.drive`, `capDrive`, `driveE` | the emitter-driven `next` loops (scans, CondTraverse with `trim_to_cap`, ExpandInto, CondVarLenTraverse, AllShortestPaths) |
| `CTGlue` | `CondTraverseOp::new`/`build_state`/`trim_to_cap`/`out_alias_id`/`null_pad`/`next`, `ExpandIntoOp::new`, scan/builder constructors |
| `VarLenGlue` | `path_value`, `VarLenIter::next` (frame-tree machine), `expand_row` endpoint plan, `new` |
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
  contract (`scanRow_spec`); edge index scans honour bound endpoints (`edgeRow_sound`).
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
   (`cond_traverse.rs:849-909`, `expand_into.rs:176-215`): `a→b, b→a, a→b`,
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
* `shortestPath()` (bidirectional BFS in `runtime/eval.rs:1400`) and the
  `algo.*paths` procedures are differential-tested against C only.
* Record caps, OPTIONAL null padding and emitter loops are modelled at row level
  (the emitter's batch packing is proved in proofs/columnar).
-/
