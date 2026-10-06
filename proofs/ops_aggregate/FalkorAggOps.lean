import FalkorAggOps.Agg
import FalkorAggOps.Ops
import FalkorAggOps.Bitmap
import FalkorAggOps.Ctors
import FalkorAggOps.SortCmp
import FalkorAggOps.SortKey
import FalkorAggOps.AggPlan
import FalkorAggOps.AggState
import FalkorAggOps.AggNext
import FalkorAggOps.StreamOps
import FalkorAggOps.LoadCsv

/-!
# FalkorAggOps — Aggregate, Distinct, Sort, Skip, Limit, Project, Unwind, aggregation functions

Build: `cd proofs/ops_aggregate && lake build` (Lean 4.34.0, core only, no Mathlib).
No `sorry` / `admit` / `axiom`. 70 theorems. Coverage per Rust fn: `COVERAGE.tsv`.
Repros: `cargo test -p graph --test lean_ops_aggregate -- --nocapture`
(graph/tests/lean_ops_aggregate.rs; the 5 `bug_*` tests fail today, the 3 `agrees_*` pass).

## Model (here | there, all under graph/src/runtime/)
| Lean | Rust |
|---|---|
| `skipFold`, `rowDriver`, `batchDriver` | kernels' null-skipping loop; `run_agg_expr` ops/aggregate.rs:981 / `fold` :229; keyless fast path `consume_input_vectorized` :509-689 |
| `countBatch`/`countStarBatch`, `sumBatch`, `collectBatch`, `mmBatch`, `avgBatch`+`aboutToOverflow`+`finalizeAvg`, `stdevBatch`+`finalizeStdev`, `pctFold` | functions/aggregation.rs:343, :387, :364, :412/:487, :440/:331/:583, :549/:675, :518 |
| `maxScalar` | aggregation.rs:119 (scalar max body) |
| `limitOp`, `skipOp` | ops/limit.rs:52, ops/skip.rs:50 |
| `ins`/`isort`, `rustStep`, `semStep` | ops/sort.rs:274 `build_full_sort`, :387 `build_top_k`, :488 `next` (cap = k+s) |
| `project_flatten`, `unwindOne`/`chunks` | ops/project.rs:54, ops/unwind.rs:90 + batched emitter |
| `groupInsert`/`groupAll`/`lookupG`, `keylessGroups` | ops/aggregate.rs `GroupMap` :59, per-row :871/:912, keyless pre-insert :524/:879 |
| `dedupBy`, `dcgGo`/`distinctCountGrouped`, `distinctCountBatches` | ops/distinct.rs:62; DISTINCT key (node, hash(group key)) aggregate.rs:715/:959; apply.rs:283 dedupers clear |
| `evalStack` vs `evalRef` | eval.rs:877 `return` in `eval_compound` |
| `Bitmap` | batch.rs:89-167 `NullBitmap` (`BitVec 64` words) |
A batch = its list of ACTIVE rows (`active_indices`, batch.rs:1204); a stream = `List (List α)`.

## Proven (highlights)
* Any batching of an aggregate's input column gives the same accumulator as row-at-a-time
  (`vectorized_eq_rowwise`, `count_hom`…`stdev_hom`, `minmax_hom`); `count` = #non-null,
  `collect` = non-null filter, `count(*)` per-row = batch; avg count law; avg without
  overflow = plain sum; avg of empty/all-null = null; stdev empty = 0.
* scalar `max` = `max_batch` under an antisymmetric flag-free compare; fails with NaN.
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
2. aggregate.rs:715/:959: per-group DISTINCT keyed by FxHash of the group key, so colliding groups
   share one seen-set. Keys [0,0] and [1,-1452335207727870361]: C c=1,1; Rust c=1,0.
   (Related to FINDINGS #6 but a different site.) Test `bug_grouped_count_distinct_shares_state_on_group_hash_collision`.
3. aggregation.rs:615/:645 sort with `partial_cmp().unwrap_or(Equal)`, which is not transitive with NaN
   (`partialCmpEq_not_transitive`), so std sort panics and the server crashes. C returns nan.
   Test `bug_percentile_with_nan_panics`.
4. aggregation.rs:539 (and :253): every row overwrites the percentile, so the last one wins
   (`pct_last_wins`). `percentileDisc(x, x/10.0)` over 1..10: C 1, Rust 10.
   Test `bug_percentile_row_dependent_argument_last_wins`.
5. eval.rs:877: an aggregate inside a list literal `return`s out of the stack machine
   (`stackEval_return_drops_frame`). `[min(x),max(x)]` C [1,2], Rust 2; `[count(x),5]` C [2,5], Rust 2;
   `size([count(x)])` C 1, Rust type error. Fix: push the result and `continue`. Test `bug_list_literal_of_aggregates_collapses`.

## Gaps / assumptions
Sort order is abstracted to a total order on Nat, so it is only valid where `compare_value` is total
(known #2891). Floats: no IEEE reasoning; the avg/sum/stdev overflow behaviour is checked with
`#eval` against C only. HashMap iteration order and the typed-column machinery (batch.rs
builders, classify_*, concat, gather) are not modelled. The scalar aggregate bodies are dead code
because `batch_agg` is always preferred.

## Wave-5 additions
| Lean | Rust |
|---|---|
| `Ctors` | `new` of Limit/Skip/Distinct/Project/Unwind/Sort/Commit |
| `SortCmp`, `SortKey` | `OrderedKey`/`HeapEntry` orders, `compare_row_content`, `classify_sort_key` (sort.rs) |
| `AggPlan`, `AggState`, `AggNext` | `subtree_has_aggregate`, `analyze`, `analyze_agg_tree`, `set_agg_expr_zero`, `unbind_agg_accumulators`, `GroupKey` Eq/Hash, `hash_u64`, `args_for`, `new`, the column extractors and `next` (aggregate.rs) |
| `StreamOps` | `filter.rs`, `include_pending.rs`, `procedure_call.rs` |
| `LoadCsv` | `load_csv.rs`: SSRF filter, size cap, pinned resolver, CSV records, path containment |
Highlights: the heap order is antisymmetric/reflexive and strict on distinct arrival (`heapCmp_swap`,
`heapCmp_seq_strict`, given lawful `compare_value`); a sort key reads back exactly
(`classifySortKey_get`); every accumulator zeroed is unbound again (`unbind_after_zero`), latent
leak for a one-child aggregate (`one_child_not_unbound`); output bindings win (`finishGroup_outputs`);
`next` emits every group once, 1024 per batch (`drain_all`); an accepted LOAD CSV URL only reaches
public addresses (`validateRemote_spec`) and never more than the byte limit (`erRun_le`); `file://`
paths stay inside the import folder (`resolvePath_spec`).
Also seen (known #2891): ORDER BY over floats with NaN crashes the server (float lane
`unwrap_or(Less)`, sort.rs:348; `compareAt_spec` in proofs/columnar).
-/
