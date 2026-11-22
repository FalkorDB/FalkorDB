import FalkorAggOps.Agg
import FalkorAggOps.Ops
import FalkorAggOps.GroupId
import FalkorAggOps.Bitmap
import FalkorAggOps.Ctors
import FalkorAggOps.SortCmp
import FalkorAggOps.SortKey
import FalkorAggOps.AggPlan
import FalkorAggOps.AggState
import FalkorAggOps.AggNext
import FalkorAggOps.StreamOps
import FalkorAggOps.LoadCsv
import FalkorAggOps.Discharge
import FalkorAggOps.SlotReuse
import FalkorAggOps.LoadCsvRedirect
import FalkorAggOps.IncludePendingExact
import FalkorAggOps.MaxScalar

/-!
# FalkorAggOps — Aggregate, Distinct, Sort, Skip, Limit, Project, Unwind, aggregation functions

Build: `cd proofs/ops_aggregate && lake build` (Lean 4.36.0, core only, no Mathlib).
No `sorry` / `admit` / `axiom`. 258 theorems. Coverage per Rust fn: `COVERAGE.tsv`.
Repros: the `bug_*` / `agrees_*` Rust tests named below live in `graph/tests/lean_ops_aggregate.rs`,
which is kept locally and NOT committed (see README); the bug entries below are also reproduced as
Cypher against a live server. Run locally with
`cargo test -p graph --test lean_ops_aggregate -- --nocapture` (the `bug_*` tests for open findings
fail, the 3 `agrees_*` pass; `bug_list_literal_of_aggregates_collapses` passes).

## Model (here | there, all under graph/src/runtime/)
| Lean | Rust |
|---|---|
| `skipFold`, `rowDriver`, `batchDriver` | kernels' null-skipping loop; `run_agg_expr` ops/aggregate.rs:998 / `fold` :243; keyless fast path `consume_input_vectorized` :523-694 |
| `countBatch`/`countStarBatch`, `sumBatch`, `collectBatch`, `mmBatch`, `avgBatch`+`aboutToOverflow`+`finalizeAvg`, `stdevBatch`+`finalizeStdev`, `pctFold` | functions/aggregation.rs:350, :394, :371, :421/:498, :451/:338/:599, :565/:683, :531 (8621212fc) |
| `maxScalar` | aggregation.rs:119 (scalar max body; NaN comparisons ignored) |
| `limitOp`, `skipOp` | ops/limit.rs:52, ops/skip.rs:50 |
| `ins`/`isort`, `rustStep`, `semStep` | ops/sort.rs:302 `build_full_sort`, :403 `build_top_k`, :504 `next` (cap = k+s) |
| `project_flatten`, `unwindOne`/`chunks` | ops/project.rs:54, ops/unwind.rs:90 + batched emitter |
| `groupInsert`/`groupAll`/`lookupG`, `keylessGroups` | ops/aggregate.rs `GroupMap` :65, per-row :710/:972, keyless pre-insert :533-540/:891-898 |
| `dedupBy`, `distinctCountBatches` | ops/distinct.rs:62; apply.rs:283 dedupers clear |
| `GroupId.insertKey`/`gid`/`dcgIdGo` (history `dcgGo`/`distinctCountGrouped`: key hash) | DISTINCT key (node, group id), the id = insertion index (`next_group_id` aggregate.rs:709/:970, `entry.2` :728/:976, `value_dedupers` :746, eval.rs:854) |
| `evalStack` (`compile`/`run`) vs `evalRef` | eval.rs:883-893 `eval_compound` aggregate arm pushes and `continue`s (289c201fe) |
| `finishEval`/`finishTail`/`finishGroup` (`preFinishGroup`) | `AggregateOp::next` per-group closure aggregate.rs:1162-1228: combined :1163-1174, outputs :1175-1191, accumulators unbound :1192-1201 BEFORE the key aliases/key row :1202-1217 (the earlier order is `preFinishGroup`), outputs re-bound :1218-1226 |
| `Bitmap` | batch.rs:89-167 `NullBitmap` (`BitVec 64` words) |
A batch = its list of ACTIVE rows (`active_indices`, batch.rs:1204); a stream = `List (List α)`.

## Proven (highlights)
* Any batching of an aggregate's input column gives the same accumulator as row-at-a-time
  (`vectorized_eq_rowwise`, `count_hom`…`stdev_hom`, `minmax_hom`); `count` = #non-null,
  `collect` = non-null filter, `count(*)` per-row = batch; avg count law; avg without
  overflow = plain sum; avg of empty/all-null = null; stdev empty = 0.
* scalar `max` = `max_batch` over an ABSTRACT compare that is antisymmetric away from NaN
  (`maxScalar_eq_batch`), now INSTANTIATED for `Value` (`MaxScalar.lean`): `maxV` is the body on
  value_order's `cmpV`, `maxV_spec` states what it returns, and on NaN-free non-null scalars the three
  premises are proved (`anti_scalar`, `nanFlag_scalar`, `nullFlag_scalar`), giving `maxV_eq_batch`. A
  list with a nested NaN breaks them (`compare_list` drops the flag); `maxScalar_eq_batch_nan` checks
  one flagged scalar NaN (both ignore NaN comparisons). The scalar body is dead code
  (`batch_agg` is always preferred); `max_batch` is covered by `minmax_hom`.
* `finalizeStdev_small/_big`, `finalizeStdevp_*` (divisors n-1 / n), `avgStep_switch_iff`
  (`about_to_overflow` drives the incremental-averaging switch), `StreamOps.pcNew_spec`,
  `srcPos_fallback`, `maskOf_lt` (procedure_call.rs:59: an unnamed yield reads its yield position).
* Rows for batch.rs (columnar lanes, `proofs/columnar`) and for the scalar aggregate bodies
  (`proofs/value_math` `AggScalar`) are not repeated in `COVERAGE.tsv`: they are `#` comments there, the
  functions are PROVEN by those projects.
* `LIMIT k` = `take k`, `SKIP s` = `drop s`, `SKIP s LIMIT k` = window, LIMIT 0 = [].
* Top-k heap = first `cap` of the full sort (`topk_eq_take_sort`); full sort is sorted;
  `ORDER BY … SKIP s LIMIT k` window equals the full-sort window.
* Project / Unwind are invariant under batching; unwind(null) = [].
* Grouping: every group holds exactly the rows with its key, in order (`groupAll_spec`);
  keyed-empty ⇒ no groups, keyless ⇒ exactly one.
* DistinctOp correct iff the hash is injective (`dedup_injective`).
* percentileDisc/Cont indices in bounds for p ∈ (0,1]. NullBitmap set/clear/all/from_values exact.

## CONFIRMED bugs (Rust test fails; live C 18230 vs Rust 18231)
1. apply.rs:283 clears runtime-global `value_dedupers` after each CALL{} run, wiping an outer
   aggregate's DISTINCT state per batch. `UNWIND range(1,3000) AS i CALL { WITH i RETURN i % 2 AS v
   LIMIT 1 } RETURN count(DISTINCT v)` C 2, Rust 6. Test `bug_count_distinct_across_call_subquery_overcounts`.

## Gaps / assumptions
Sort order inside `SortCmp` is a parameter `cv`; on main it is `Value::sort_cmp`, whose laws are
proven in proofs/value_order and instantiated here (`Discharge.lean`, which imports
proofs/value_order as a Lake path dependency).
Floats: no IEEE reasoning in the aggregates; the avg/sum/stdev overflow behaviour is checked with
`#eval` against C only. HashMap iteration order and the typed-column machinery (batch.rs
builders, classify_*, concat, gather) are not modelled. The scalar aggregate bodies are dead code
because `batch_agg` is always preferred.

`gkHash_consistent` (`GroupKey::hash`) takes "`Value`'s Hash agrees with the key equality" as `hc`;
`GroupKey::eq` is `sort_cmp == Equal` (aggregate.rs:99) with NaN/f32 hashing canonicalised, and
`Discharge.hc_main` proves `hc` from value_order `Main.main_hash_consistent`
(`gk_hash_consistent_main`; `veqMain_equiv`: the key equality is an equivalence).
`cmpKey_lawful`/`heapCmp_swap`/`heapCmp_refl` take `Lawful cv`; `cv` is
`sort_cmp` and `Discharge.lawful_main` proves it from value_order `Main.scMain_total`
(`cmpKey_lawful_main`, `cmpKey_trans_main`, `heapCmp_main`). The sort.rs `cmp_key`/`cmp`/`cmp`
and aggregate.rs `hash` rows are PROVEN. Also `tiebreak` (sort.rs:237):
`tiebreak_strict`, `tiebreak_swap`.

## `AggregateOp::next` unbinds accumulators before it writes the keys
`finishGroup` is `finishEval`, then unbind (aggregate.rs:1192-1201), then `finishTail` (the key
aliases and the key row, :1202-1217), then the outputs, so a null key sharing an accumulator's slot id stays bound.
The earlier order is `preFinishGroup` (with `PreKeysBound`), kept as a historical counterexample.
`fixed3522_null_key_bound` (`RETURN null AS x, count(1)`),
`keysBound_swap_3522` (`WITH 1 AS a, 2 AS b RETURN b AS x, a AS y, count(*)`), and the general
`finishGroup_keys_bound`/`keysBound_main`: for any keys/accumulators/evaluator, with no key name an
output name and the group evaluating, `KeysBound` holds on main (`unbind_agg_accumulators` :1251).

## Gap closed 2026-10-20: binder slot ids across an aggregating WITH/RETURN (`SlotReuse.lean`)
The aggregate model kept outputs, accumulators and pre-projection aliases in separate abstract
slots; the binder numbers each scope from 0 and `Row` is keyed by the bare id, so they collide.
`projLayout` mints the ids as the binder does (accumulators at `len` of the pre-projection scope,
binder.rs `bind_projection`; aliases 0.. in the new scope), `finishCore` /
`finishVals` / `cell` add what `unbind` keeps (row.rs:160) and what a reader sees (batch.rs:631-640,
:763-778). Properties a fix should prove: `OutputsOwnAcc`, `KeysBound`, `EmitsOnlyScope`
(`AggRowSpec`). Refuted on main, each mirroring the live repro (Rust 9d3869aa8 vs C):
`bug3375_output_overwrites_next_acc` (#3375: sum = count, 3 vs 6), `bug3375_avg_panics`
(`finalize_avg` unreachable), `bug3333_alias_binds_next_var` (#3333: alias of `m` binds
`b`'s id 2), `bug3369_count_acc_in_merge_slot` / `bug3368_count_acc_in_merge_slot` (#3369, #3368:
`count(*)` accumulators left value-only at the ids of later MERGE nodes; the MERGE that then reads
them as bound is merge.rs, not modelled here). Controls that agree with live: `agree3375_one_key`,
`agree3333_controls`.

## Wave-5 additions
| Lean | Rust |
|---|---|
| `Ctors` | `new` of Limit/Skip/Distinct/Project/Unwind/Sort/Commit |
| `SortCmp`, `SortKey` | `OrderedKey`/`HeapEntry` orders, `compare_row_content`, `classify_sort_key` (sort.rs) |
| `AggPlan`, `AggState`, `AggNext` | `subtree_has_aggregate`, `analyze`, `analyze_agg_tree`, `set_agg_expr_zero`, `unbind_agg_accumulators`, `GroupKey` Eq/Hash, `args_for`, `new`, the column extractors and `next` (aggregate.rs) |
| `StreamOps` | `filter.rs`, `include_pending.rs`, `procedure_call.rs` |
| `LoadCsv` | `load_csv.rs`: SSRF filter, size cap, pinned resolver, CSV records, path containment |
Highlights: the heap order is antisymmetric/reflexive and strict on distinct arrival (`heapCmp_swap`,
`heapCmp_seq_strict`, given lawful `compare_value`); a sort key reads back exactly
(`classifySortKey_get`); every accumulator zeroed is unbound again (`unbind_after_zero`), latent
leak for a one-child aggregate (`one_child_not_unbound`); output bindings win (`finishGroup_outputs`);
`next` emits every group once, 1024 per batch (`drain_all`); an accepted LOAD CSV URL only reaches
public addresses (`validateRemote_spec`) and never more than the byte limit (`erRun_le`); `file://`
paths stay inside the import folder (`resolvePath_spec`).
ORDER BY over floats with NaN: the float lane is `sort_cmp_f64` (sort.rs:364; proofs/columnar `compareAt_spec`).

## LOAD CSV redirects and IncludePending (`LoadCsvRedirect.lean`, `IncludePendingExact.lean`)
**`LoadCsvRedirect.lean`.** `fetch` models the agent's request loop. `http_config` (load_csv.rs:223-234) sets
`https_only(true)` and `max_redirects(0)`, and `open_csv_records` (:357-) turns a 3xx reply into the error
"LOAD CSV does not follow redirects" (:388-399), as C does. `httpCfgMain` is that config and `loadRemote` that arm:
`fixed3443_eachHopValidated` (`loadRemote` satisfies `EachHopValidated` for every DNS answer and server),
`fixed3443_redirect_is_error`, `noRedirect_single`. `pinned_spec`/`httpConfig_spec` cover only the
address bound / OnceLock idempotence (COVERAGE reasons say so).
**`IncludePendingExact.lean`.** `get_pending_nodes_with_labels(label_ids, g)` returns the live nodes that gain a
pattern label, and Phase 1 skips them (`extra_nodes`). The file models that code
(`pendingWith g`, `includePendingMain` with `childRows … extra`, `StreamOps.endChild`) and proves
`includePendingMain_exact`; `fixed3453_deleted_node`, `fixed3453_duplicate_row`,
`fixed3453_split_multilabel`.
-/
