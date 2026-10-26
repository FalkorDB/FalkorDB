import FalkorOpsApply.Ctors
import FalkorOpsApply.CreateAttrs
import FalkorOpsApply.CreateBatch
import FalkorOpsApply.Delete
import FalkorOpsApply.DeleteCancel
import FalkorOpsApply.SetRemove
import FalkorOpsApply.CartVhj
import FalkorOpsApply.RowWrites
import FalkorOpsApply.MergeSlot
/-
# ops_apply: MERGE, correlated sub-plans, FOREACH and UNION (FalkorDB-rs)

Lean 4 models of `graph/src/runtime/ops/{merge,apply,optional,semi_apply,
or_apply_multiplexer,foreach,union}.rs` (plus the argument-batch idiom of
`runtime/batch.rs`), proved against per-row openCypher reference semantics and
checked against the C engine on live servers.  Build: `lake build` (core Lean
4.35, no Mathlib); no `sorry`/`admit`/`axiom`.  Machine-readable coverage:
`COVERAGE.tsv`.  Repros: `graph/tests/lean_ops_apply.rs`
(`cargo test -p graph --test lean_ops_apply -- --nocapture`; every `bug_*` test
asserts C's answer and FAILS today).

## Lean ↔ Rust

| here | there |
| --- | --- |
| `Correlated.tag`                          | `Batch::clone_active_rows_seq_origin` (`batch.rs:1163`) |
| `Correlated.SubPlan`, `Separable`         | `run_batch` + `set_argument_batch` (`batch.rs:1590`); separability = origin-respecting, row-local |
| `Correlated.applyBatched`                 | `ApplyOp::next_batched` (`apply.rs:152`), `merge_over_input` (`batch.rs:943`) |
| `Correlated.applyPerRow`                  | `ApplyOp::next_per_row` (`apply.rs:238`) |
| `Correlated.optionalBatched`              | `OptionalOp::next_batched` (`optional.rs:125`), Optional arm of `apply.rs:210-229` |
| `Correlated.optionalPerRow`               | `OptionalOp::next_per_row` (`optional.rs:195`) |
| `Correlated.semiBatched`                  | `SemiApplyOp::next` (`semi_apply.rs:66`) |
| `Correlated.orApply`                      | `OrApplyMultiplexerOp::next` (`or_apply_multiplexer.rs:73`) |
| `Correlated.canBatchApply/Optional`       | `can_batch` in `ApplyOp::new` (`apply.rs:119`), `OptionalOp::new` (`optional.rs:104`) |
| `Correlated.vhj`                          | `ValueHashJoinOp::next` (`value_hash_join.rs:366-446`) |
| `Union.next`, `Union.St`                  | `UnionOp::next` (`union.rs:68`), `current`/`current_child` |
| `ForEach.chunks`, `execList`, `runChunks` | `ForEachOp::execute_list` (`foreach.rs:86`) |
| `ForEach.inner`, `go`, `next`             | `ForEachOp::next` (`foreach.rs:124`) |
| `ForEach.drainP`                          | `ops::drain_pending` (`ops/mod.rs:127`) |
| `Merge.createFallback`, `Cache`           | `MergeOp::do_create_fallback` (`merge.rs:113`), `Runtime::merge_pattern_cache` (`runtime.rs:201`) |
| `Merge.phaseA`, `rustBatch`, `rustMerge`  | `MergeOp::next` (`merge.rs:279`) |
| `Merge.phaseB`                            | `MergeOp::drain_pending` (`merge.rs:252`) |
| `Merge.refMerge`                          | openCypher MERGE, per row |

## Proven (headline)

* `applyBatched_eq_ref` — batched Apply = per-row Apply for separable sub-plans.
* `applyPerRow_eq_ref`, `optionalPerRow_eq_ref` — per-row modes are the reference for ANY sub-plan.
* `optionalBatched_perm_ref` — batched Optional is a permutation of the reference.
* `semiBatched_eq_ref`, `orApply_eq_ref`, `orApply_single` — (anti-)semi-apply / OR-apply = per-row EXISTS.
* `rowLocal_separable`, `compose_separable` — row-local operators and their pipelines are separable.
* `vhj_not_separable`, `count_not_separable`, `canBatchApply_admits_vhj`, `vhj_batched_16_vs_8`.
* `union_drain`, `drain_eq`, `next_spec`, `stream_all_ok`, `next_none_stays` — UNION ALL = concatenation; failed branch instantiation ends the stream with its error.
* `chunks_flatten`, `chunks_size`, `inner_ok`, `go_ok`, `next_ok`, `run_ok`, `foreach_correct`, `calls_items` — FOREACH passes every row through unchanged, in order, in batches of 1..B rows, and runs the body on every item of every row exactly once, in order, in chunks ≤ B.
* `phaseA_ref`, `rustBatch_ref`, `rust_eq_ref_noSet`, `merge_noSet_any_batch_size`, `heads_tails_perm` — MERGE without SET equals per-row openCypher for EVERY batching (same graph, same ids, output rows a permutation).
* Counterexamples (by `decide`): `batch_dependent`, `pre3050_cross_clause_unbound` (fixed: `cross_clause_keyed`, `per_clause_cache_fixes`), `pre3055_allBound_first_only` (fixed: `allBound_path_keeps_all`, `path_shortcut_off`), `optionalBatched_order_differs`.

## CONFIRMED bugs (Rust repro + live Rust vs C, ports 18240/18241)

1. **ValueHashJoin under batched Apply/Optional joins different outer rows.**
   `apply.rs:119-133`, `optional.rs:104-106` omit `IR::ValueHashJoin` from the
   `can_batch` blacklist; the join never compares `origin_row`.
   `UNWIND [1,2] AS x MATCH (a:A),(b:B) WHERE a.v = b.v RETURN count(*)` over 2 A + 2 B (v=1):
   Rust 16, C 8.  OPTIONAL MATCH form duplicates rows (Rust 16 vs C 8; with
   `a.w = x AND b.w = x`: each row twice).  Tests `bug_value_hash_join_under_*`.
   Fix: add `IR::ValueHashJoin` to both blacklists (or make the join key include origin).
2. **`merge_pattern_cache` shared by all MERGE clauses** — FIXED by #3050 (closes
   #3038, main 5e475e6c9): the cache is keyed by `(operator idx, pattern hash)`
   (`merge.rs:122,172`; `runtime.rs:201`). `MERGE (a:M {v:1}) ON CREATE SET a.v = 2 MERGE (b:M {v:1})
   RETURN a.v, b.v` used to give `Variable b not found`. Lean: `Merge.pre3050_cross_clause_unbound`
   (old model = every clause the same `op`), `Merge.cross_clause_keyed` (main).
3. **MERGE result depends on BATCH_SIZE** (`merge.rs:313-343`): the match runs once
   per batch and sees earlier batches' creations only.  `UNWIND range(1,N) AS x WITH x
   WHERE x = 1 OR x = N MERGE (n:L {v: CASE x WHEN 1 THEN 1 ELSE 2 END}) ON CREATE SET n.v = 2`:
   N=2 → 2 nodes, N=1025 → 1 node (C: 2 and 2).  Test `bug_merge_result_depends_on_batch_boundary`.
   Fix: match all rows before any creation (C's eager phases), or re-match per row.
4. **All-bound shortcut drops matches visible through a path** — FIXED by #3055
   (closes #3040, main 5e475e6c9): the shortcut applies only when every relationship is
   bound or anonymous and not exposed by a path variable (`merge.rs:369-388`).
   `MATCH (a:A),(b:B) MERGE p=(a)-[:R]->(b) RETURN count(p)` over two parallel edges was
   Rust 1, C 2. Lean: `Merge.pre3055_allBound_first_only`, `Merge.allBound_path_keeps_all`,
   `Merge.allBoundMain_imp_pre`, `Merge.path_shortcut_off`.
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

## Modelling gaps

* Sub-plans are abstract functions; separability of each real operator is a
  hypothesis, not derived from the operator code (except the ValueHashJoin model).
* MERGE: single node pattern, one property, hash = key (collisions assumed away,
  already known); ON MATCH/ON CREATE only as key updates; relationship patterns
  and the null-property error not modelled; the all-bound shortcut only via its flag.
* Batch boundaries of emitted output (sizes) are modelled for FOREACH only;
  Apply/Optional/Merge are compared on flattened streams.
* Errors: FOREACH/Union error paths modelled but only the error-free theorems proven.

## Re-target to 2c874022a (#2846)
`create_batch` reserves through the graph's batch (`g.node_id_space().reserve(n,
&pending.created_nodes)`, create.rs:158; relationships :280) — the abstract
`reserveN`/`reserveR` hooks are unchanged. DELETE's unwinds now return the ids to
the graph themselves and can fail (`?` at delete.rs:254-260, :408-412, :447-453):
`DeleteCancel.lean`. #2776 re-checked live: **still present** (`CREATE (a {x:1}),(b
{y:2}) SET a = b` Rust `{x: 1, y: 2}`, C `{y: 2}`). W2-pending-3 (writes to a
DETACH-deleted relationship are applied/counted): **still present** (`... DETACH DELETE
a SET r.x = 1` Rust `Properties set: 1`, C none).

## Wave-5 additions
| Lean | Rust |
|---|---|
| `Ctors` | `MergeOp::new`, the three `OnceCell` resolvers, `compute_merge_pattern_hash`; `new` of SemiApply/OrApplyMultiplexer/ForEach; `UnionOp::store_argument_batch` |
| `CreateAttrs`, `CreateBatch` | `build_attr_template`, `resolve_map_attrs`, `resolve_pattern`, `create_batch`, `CreateOp::new/next` (create.rs) |
| `Delete`, `DeleteCancel` | every fn of delete.rs; since #2846 (Rust at 2c874022a) the unwinds can fail — `DeleteCancel` is the fallible model (`delNodeE`, `bulkScanE`, `delNodesBulkE`, `delEntityE`, `deleteBatchE`), equal to the infallible one when the graph accepts every cancel (`deleteBatchE_ok`), refusal = operator error (`delNodeE_refusedP/C`) |
| `SetRemove` | every fn of set.rs and remove.rs |
| `CartVhj` | cartesian_product.rs (odometer) and value_hash_join.rs |
Highlights: template positions sort attribute ids strictly, last duplicate key wins
(`rank_strict`, `tplBuild_lookup`); CREATE binds one fresh node per active row and logs it
(`createNode_spec`), relationships take each row's endpoints and refuse deleted ones
(`createRel_spec`); DELETE leaves every bound node/relationship gone (`deleteBatch_vars`,
`delNodesBulk_spec`, `delRelsBulk_spec`); `SET n = {map}` on a node makes exactly the map's keys
present (`setFromMapNode_replace`); the odometer only visits valid combinations
(`orbit_valid`) and the product has `∏ lens` rows (`allTuples_length`).

### CONFIRMED (wave 5)
* **`SET e = …` keeps attributes written earlier in the same query** (set.rs:202-249 Node←Node/Rel,
  :279-376 every relationship branch: only COMMITTED keys are nulled, pending ones survive; only the
  Node←Map branch calls `clear_node_attributes`). `CREATE (a {x:1}), (b {y:2}) SET a = b RETURN
  properties(a)`: Rust `{x: 1, y: 2}`, C `{y: 2}`. `MATCH ()-[r:R]->() SET r.w = 5 SET r = {y:2}
  RETURN properties(r)`: Rust `{w: 5, y: 2}`, C `{y: 2}`. Lean `noClear_keeps_pending`.
  Tests `bug_set_replace_node_from_node_keeps_pending_attrs`, `bug_set_replace_rel_from_map_keeps_pending_attrs`.
  Fix: clear the target's pending attributes in every `replace` branch.
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

## Gap closed 2026-10-20 — MERGE reads a value-only slot as its node (`MergeSlot`)
`Merge.lean` keyed MERGE by the property alone, so the input row never mattered.
`MergeSlot` adds the row (values + bound bits, `to_owned_row`) and the real hash input
(`Ctors.mergeHashInput` with `get_by_id`, merge.rs:189): a value-only leftover at the
variable's id (ops_aggregate `bug3368/3369_count_acc_in_merge_slot`) becomes the hash, and
equal leftovers in two MERGEs share a cache entry (merge.rs:122-127).
Properties: `KeyFromBoundNodeOnly`, `MergeBindsOwnNode` (fix target: `fixKey_from_bound_node`,
`fixed_binds_own_node`). Refuted, live-reproduced Rust vs C:
* #3368 `pre3050_bug3368_merge_takes_leftover_as_node` (`c` = `[B] b1`), `pre3050_bug3368_fires_iff_leftovers_equal` (filed repro fixed by #3050);
* #3369 `pre3050_bug3369_merge_leaves_int_in_node_var` (`c` = `Int 1`), `pre3050_bug3369_controls` (filed repro fixed by #3050).

## Re-target to 5e475e6c9 (#3050, #3055, #3462, #3436)
`Merge.Clause.op` / `MergeSlot.MClause.op` carry the operator idx and the cache key is
`(op, hash)` (#3050); the pre-#3050 shared cache is the model with every `op` equal
(`pre3050_*`). `Merge.allBoundMain` is the #3055 shortcut condition. `Delete.DeleteSt.new`
takes `inspect` (`snapshot = inspect || snapshot_required`, #3462). #3436 (`self.g.borrow()`
→ `self.graph()`) moves lines in create/delete only; no model change. #3368/#3369 as filed
(two different MERGE operators sharing an entry) no longer share one: `MergeSlot.bug3368_filed_repro_fixed_by_3050`.
The leftover hash itself is unchanged and still collides inside ONE operator
(`MergeSlot.same_op_leftover_shares_entry`, model only).
-/
import FalkorOpsApply.Correlated
import FalkorOpsApply.Union
import FalkorOpsApply.ForEach
import FalkorOpsApply.Merge
