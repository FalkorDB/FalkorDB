import FalkorOpsApply.Ctors
import FalkorOpsApply.CreateAttrs
import FalkorOpsApply.CreateBatch
import FalkorOpsApply.Delete
import FalkorOpsApply.DeleteCancel
import FalkorOpsApply.SetRemove
import FalkorOpsApply.CartVhj
import FalkorOpsApply.RowWrites
import FalkorOpsApply.MergeSlot
import FalkorOpsApply.BatchRows
import FalkorOpsApply.VhjSep
import FalkorOpsApply.OpSep
/-
# ops_apply: MERGE, correlated sub-plans, FOREACH and UNION (FalkorDB-rs)

Lean 4 models of `graph/src/runtime/ops/{merge,apply,optional,semi_apply,
or_apply_multiplexer,foreach,union}.rs` (plus the argument-batch idiom of
`runtime/batch.rs`), proved against per-row openCypher reference semantics and
checked against the C engine on live servers.  Build: `lake build` (core Lean
4.36, no Mathlib); no `sorry`/`admit`/`axiom`.  Machine-readable coverage:
`COVERAGE.tsv`.  Repros: `graph/tests/lean_ops_apply.rs`
(`cargo test -p graph --test lean_ops_apply -- --nocapture`; every `bug_*` test
asserts C's answer and FAILS today).

## Lean ↔ Rust

| here | there |
| --- | --- |
| `Correlated.tag`                          | `Batch::clone_active_rows_seq_origin` (`batch.rs:1170`) |
| `Correlated.SubPlan`, `Separable`         | `run_batch` + `set_argument_batch` (`batch.rs:1603`); separability = origin-respecting, row-local |
| `Correlated.applyBatched`                 | `ApplyOp::next_batched` (`apply.rs:152`), `merge_over_input` (`batch.rs:944`) |
| `Correlated.applyPerRow`                  | `ApplyOp::next_per_row` (`apply.rs:238`) |
| `Correlated.optionalBatched`              | `OptionalOp::next_batched` (`optional.rs:125`), Optional arm of `apply.rs:210-229` |
| `Correlated.optionalPerRow`               | `OptionalOp::next_per_row` (`optional.rs:195`) |
| `Correlated.semiBatched`                  | `SemiApplyOp::next` (`semi_apply.rs:66`) |
| `Correlated.orApply`                      | `OrApplyMultiplexerOp::next` (`or_apply_multiplexer.rs:73`) |
| `Correlated.canBatchApply/Optional`       | `can_batch` in `ApplyOp::new` (`apply.rs:119`), `OptionalOp::new` (`optional.rs:104`) |
| `OpSep.Adm`, `DSep`, `expandF`/`optF`/`semiF`/`orF`/`vhjF`/`perRowF` | the operators `can_batch` admits under a batched sub-plan (inventory in `OpSep.lean`'s header); `Adm.dsep` = separability on distinct-origin arguments |
| `OpSep.applyBatched_rel`, `applyNext`, `optionalBatched_perm_ref_adm`, `semiBatched_eq_ref_adm`, `orApply_eq_ref_adm` | `ApplyOp::next_batched`/`next` (`apply.rs:152,343`), `OptionalOp::next_batched`/`next` (`optional.rs:125,292`), `SemiApplyOp::next` (`semi_apply.rs:66`), `OrApplyMultiplexerOp::next` (`or_apply_multiplexer.rs:73`) |
| `Correlated.vhj`                          | the pre-per-origin `ValueHashJoinOp::next` (`value_hash_join.rs:366-446` at 343e077f2; now per-origin, `CartVhj.vhjRowsPO`) |
| `Union.next`, `Union.St`                  | `UnionOp::next` (`union.rs:68`), `current`/`current_child` |
| `ForEach.chunks`, `execList`, `runChunks` | `ForEachOp::execute_list` (`foreach.rs:86`) |
| `ForEach.inner`, `go`, `next`             | `ForEachOp::next` (`foreach.rs:124`) |
| `ForEach.drainP`                          | `ops::drain_pending` (`ops/mod.rs:127`) |
| `Merge.createFallback`, `Cache`           | `MergeOp::do_create_fallback` (`merge.rs:117`), `Runtime::merge_pattern_cache` (`runtime.rs:201`) |
| `MergeEager.matchAll`                     | `MergeOp::match_all` (`merge.rs:260`) |
| `MergeEager.rowMain`, `mainMerge`         | `MergeOp::next` (`merge.rs:329`) |
| `Merge.phaseA`, `rustBatch`, `rustMerge`  | the per-batch-match `MergeOp::next` (history) |
| `Merge.phaseB`                            | `MergeOp::drain_pending` (`merge.rs:302`) |
| `SetRemove.clearForReplace`, `setFrom`    | `clear_node_attrs_for_replace` (`set.rs:125`), `clear_relationship_attrs_for_replace` (`set.rs:138`), `set_inner` (`set.rs:151`) |
| `Merge.refMerge`                          | openCypher MERGE, per row |
| `BatchRows.Bat`, `gather`, `intoCompacted`, `cloneSeq`, `originRow` | `Batch`, `gather` (`batch.rs:898`), `into_compacted`, `clone_active_rows_seq_origin` (`:1170`), `origin_row` (`:1305`) |
| `BatchRows.mergeOverInput`                | `Batch::merge_over_input` (`batch.rs:944`) |
| `BatchRows.setArg`                        | `BatchOp::set_argument_batch` (`batch.rs:1603`) |
| `MergeSlot.Row.ofCells`                   | `BatchRow::to_owned_row` (`batch.rs:1439`, one slot per column) |

## Proven (headline)

* `applyBatched_eq_ref` — batched Apply = per-row Apply for separable sub-plans.
* `applyPerRow_eq_ref`, `optionalPerRow_eq_ref` — per-row modes are the reference for ANY sub-plan.
* `optionalBatched_perm_ref` — batched Optional is a permutation of the reference.
* `semiBatched_eq_ref`, `orApply_eq_ref`, `orApply_single` — (anti-)semi-apply / OR-apply = per-row EXISTS.
* `rowLocal_separable`, `compose_separable` — row-local operators and their pipelines are separable.
* `vhj_not_separable`, `count_not_separable`, `canBatchApply_admits_vhj`, `vhj_batched_16_vs_8` (the ValueHashJoin ones describe the join before it became per-origin, `CartVhj.vhjRowsPO`).
* `OpSep.Adm.dsep`, `applyBatched_rel`/`applyBatched_eq_ref_adm`, `applyNext_perm_ref`, `optionalBatched_perm_ref_adm`, `optionalNext_perm_ref`, `semiBatched_eq_ref_adm`, `orApply_eq_ref_adm` — the six CONDITIONAL rows, discharged: every operator the `can_batch` guards let through is modelled (scans/traverses/Unwind/ProcedureCall/Filter/Project/PathBuilder = `expandF`, fused optional traverse = `optF`, `SemiApply`/`OrApply`, per-origin `ValueHashJoin`, nested per-row `Apply`), separability composes through the tree (`DSep`, origins tracked instead of required distinct), and batched = per-row (exactly, or up to permutation when a fused optional `CondTraverse` is inside). Counterexamples kept: `OpSep.dedupF_not_separable` (#3106), `limitF_not_separable`, `optF_not_exact` (#3070 family); writes: `RowWrites.bug2738_batched_call_hides_writes` (#2738).
* `VhjSep.vhjPO_sepT`, `applyBatched_vhjPO`, `vhjPO_16_vs_8_fixed` — the per-origin ValueHashJoin is separable on the tag-shaped batches Apply builds (`SepT`), and batched Apply over it equals the reference.
* `BatchRows.cloneSeq_spec`, `cloneSeq_tag`, `gather_row`, `intoCompacted_row`, `originRow_spec`, `mergeOverInput_row`, `slots_setArg`, `MergeSlot.ofCells_slot` — the plumbing under the idiom: stamping, gathering, merging over the input and installing the argument batch.
* `union_drain`, `drain_eq`, `next_spec`, `stream_all_ok`, `next_none_stays` — UNION ALL = concatenation; failed branch instantiation ends the stream with its error.
* `chunks_flatten`, `chunks_size`, `inner_ok`, `go_ok`, `next_ok`, `run_ok`, `foreach_correct`, `calls_items` — FOREACH passes every row through unchanged, in order, in batches of 1..B rows, and runs the body on every item of every row exactly once, in order, in chunks ≤ B.
* `phaseA_ref`, `rustBatch_ref`, `rust_eq_ref_noSet`, `merge_noSet_any_batch_size`, `heads_tails_perm` — MERGE without SET equals per-row openCypher for EVERY batching (same graph, same ids, output rows a permutation).
* Counterexamples (by `decide`): `optionalBatched_order_differs`; MERGE: `cross_clause_keyed`, `per_clause_cache_fixes`, `allBound_path_keeps_all`, `path_shortcut_off`, `fixed3067_batch_independent`.

## CONFIRMED bugs (Rust repro + live Rust vs C, ports 18240/18241)

5. **Optional fallback rows reordered** (low severity; order is unspecified by
   openCypher but observable via `collect`): `UNWIND [3,1,2] AS x OPTIONAL MATCH (a:A {w:x})
   RETURN collect(x)` → Rust `[1, 2, 3]`, C `[3, 1, 2]`.  Test `bug_optional_fallback_rows_reordered`.

## Observed, not in target / not a Rust bug

* Creating and deleting a node in one query reports no `Nodes created/deleted`
  statistics (`CREATE (a:D) DELETE a`, also via MERGE); C reports 1/1 (pending/stats code).
* C bugs seen while comparing: `UNWIND [1,2] AS x CALL { MERGE (n:P {v:1}) ON CREATE SET n.v = 2 RETURN n }`
  returns only the x=1 row in C; `UNWIND [1,2] AS x MERGE (a:E {v:1}) ON MATCH SET a.m = x ON CREATE SET a.c = x`
  gives C `m=1, c=2` (Rust `m=2, c=1`, matching openCypher).
* `FOREACH (i IN 1 | …)`: Rust type error, C iterates once (Rust follows openCypher).
* W2-pending-3 (writes to a DETACH-deleted relationship are applied/counted; last live-checked at 2c874022a):
  `... DETACH DELETE a SET r.x = 1` Rust `Properties set: 1`, C none.

## Modelling gaps

* Sub-plans are abstract functions; separability of each real operator is a
  hypothesis, not derived from the operator code (except the ValueHashJoin model).
* MERGE: single node pattern, one property, hash = key (collisions assumed away,
  already known); ON MATCH/ON CREATE only as key updates; relationship patterns
  and the null-property error not modelled; the all-bound shortcut only via its flag.
* Batch boundaries of emitted output (sizes) are modelled for FOREACH only;
  Apply/Optional/Merge are compared on flattened streams.
* Errors: FOREACH/Union error paths modelled but only the error-free theorems proven.


## Wave-5 additions
| Lean | Rust |
|---|---|
| `Ctors` | `MergeOp::new`, the three `OnceCell` resolvers, `compute_merge_pattern_hash`; `new` of SemiApply/OrApplyMultiplexer/ForEach; `UnionOp::store_argument_batch` |
| `CreateAttrs`, `CreateBatch` | `build_attr_template`, `resolve_map_attrs`, `resolve_pattern`, `create_batch`, `CreateOp::new/next` (create.rs) |
| `Delete`, `DeleteCancel` | every fn of delete.rs; the unwinds can fail — `DeleteCancel` is the fallible model (`delNodeE`, `bulkScanE`, `delNodesBulkE`, `delEntityE`, `deleteBatchE`), equal to the infallible one when the graph accepts every cancel (`deleteBatchE_ok`), refusal = operator error (`delNodeE_refusedP/C`) |
| `SetRemove` | every fn of set.rs and remove.rs |
| `CartVhj` | cartesian_product.rs (odometer) and value_hash_join.rs |
Highlights: template positions sort attribute ids strictly, last duplicate key wins
(`rank_strict`, `tplBuild_lookup`); CREATE binds one fresh node per active row and logs it
(`createNode_spec`), relationships take each row's endpoints and refuse deleted ones
(`createRel_spec`); DELETE leaves every bound node/relationship gone (`deleteBatch_vars`,
`delNodesBulk_spec`, `delRelsBulk_spec`); `SET n = {map}` on a node makes exactly the map's keys
present (`setFrom_replace`, any source); the odometer only visits valid combinations
(`orbit_valid`) and the product has `∏ lens` rows (`allTuples_length`).

### Also seen (wave 5)
Also seen: W2-rewrites-3 (hash join keys Int/Float near 2^53, `keyAsI64_spec`).

## Gap closed 2026-10-20 — per-row write visibility through Apply (`RowWrites`)
The Apply model treated a sub-plan as a pure function of its row, so writes made by
one outer row and read (or recycled) by a later one were outside every theorem.
`RowWrites.WSub` adds the graph state, the queued writes and the Commit; batched Apply
(`applyBatchedW`, one Commit per input batch) vs per-row (`applyRefW`, one per row).
Properties stated (each refuted on main, each live-reproduced Rust vs C):
* `WriteSeparable` — #2738, `bug2738_batched_call_hides_writes` (`can_batch` admits
  Create/Delete/Set/Commit: `canBatchApply_admits_write`); fix target
  `applyBatchedW_eq_ref_of_readIndep`.
* `PartitionInvariant` — #2729, `bug2729_batch_split_flips_endpoint_check`, built from
  `CreateBatch.createRel` + `Delete.deleteBatch` streaming per batch.
* `NoStaleAlias` — #3164, `bug3164_stale_ref_deletes_recycled_id` (per-row commits
  release ids that `IdSpace::reserve` reclaims while rows still hold them).

## MERGE reads a value-only slot as its node (`MergeSlot`) — open (#3368)
`Merge.lean` keyed MERGE by the property alone, so the input row never mattered.
`MergeSlot` adds the row (values + bound bits, `to_owned_row`) and the real hash input
(`Ctors.mergeHashInput` with `get_by_id`, merge.rs:193): a value-only leftover at the
variable's id (ops_aggregate `bug3368/3369_count_acc_in_merge_slot`) becomes the hash.
Properties: `KeyFromBoundNodeOnly`, `MergeBindsOwnNode` (fix target: `fixKey_from_bound_node`,
`fixed_binds_own_node`). The cache key is `(op, hash)` (`Merge.Clause.op` / `MergeSlot.MClause.op`), so
two different MERGE operators no longer share an entry
(`MergeSlot.bug3368_filed_repro_fixed_by_3050`, kept as a historical counterexample). The leftover
hash itself still collides inside ONE operator (`MergeSlot.same_op_leftover_shares_entry`, model only).

-/
import FalkorOpsApply.Correlated
import FalkorOpsApply.Union
import FalkorOpsApply.ForEach
import FalkorOpsApply.Merge
import FalkorOpsApply.MergeEager
