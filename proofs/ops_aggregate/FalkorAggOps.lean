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
import FalkorAggOps.Discharge
import FalkorAggOps.SlotReuse
import FalkorAggOps.LoadCsvRedirect
import FalkorAggOps.IncludePendingExact

/-!
# FalkorAggOps — Aggregate, Distinct, Sort, Skip, Limit, Project, Unwind, aggregation functions

Build: `cd proofs/ops_aggregate && lake build` (Lean 4.35.0, core only, no Mathlib).
No `sorry` / `admit` / `axiom`. 235 theorems. Coverage per Rust fn: `COVERAGE.tsv`.
Repros: `cargo test -p graph --test lean_ops_aggregate -- --nocapture`
(graph/tests/lean_ops_aggregate.rs; the `bug_*` tests for open findings fail, the 3 `agrees_*` pass;
`bug_list_literal_of_aggregates_collapses` passes since #2951).

## Model (here | there, all under graph/src/runtime/)
| Lean | Rust |
|---|---|
| `skipFold`, `rowDriver`, `batchDriver` | kernels' null-skipping loop; `run_agg_expr` ops/aggregate.rs:990 / `fold` :238; keyless fast path `consume_input_vectorized` :518-689 |
| `countBatch`/`countStarBatch`, `sumBatch`, `collectBatch`, `mmBatch`, `avgBatch`+`aboutToOverflow`+`finalizeAvg`, `stdevBatch`+`finalizeStdev`, `pctFold` | functions/aggregation.rs:350, :394, :371, :421/:498, :451/:338/:599, :565/:683, :531 (8621212fc) |
| `maxScalar` | aggregation.rs:119 (scalar max body; NaN comparisons ignored since #2917) |
| `limitOp`, `skipOp` | ops/limit.rs:52, ops/skip.rs:50 |
| `ins`/`isort`, `rustStep`, `semStep` | ops/sort.rs:302 `build_full_sort`, :403 `build_top_k`, :504 `next` (cap = k+s) |
| `project_flatten`, `unwindOne`/`chunks` | ops/project.rs:54, ops/unwind.rs:90 + batched emitter |
| `groupInsert`/`groupAll`/`lookupG`, `keylessGroups` | ops/aggregate.rs `GroupMap` :59, per-row :880/:921, keyless pre-insert :524/:879 |
| `dedupBy`, `dcgGo`/`distinctCountGrouped`, `distinctCountBatches` | ops/distinct.rs:62; DISTINCT key (node, hash(group key)) aggregate.rs:724/:968; apply.rs:283 dedupers clear |
| `evalStack` (`compile`/`run`) vs `evalRef` | eval.rs:883-893 `eval_compound` aggregate arm pushes and `continue`s (289c201fe); `pre2951_evalStack`: the old `return` at eval.rs:877 |
| `finishEval`/`finishTail`/`finishGroup` (`preFinishGroup`) | `AggregateOp::next` per-group closure aggregate.rs:1141-1219: outputs :1154-1182, accumulators unbound :1184-1192 BEFORE the key aliases/key row :1193-1209 (#3522; `pre3522_*` is the old order), outputs re-bound :1210-1217 |
| `Bitmap` | batch.rs:89-167 `NullBitmap` (`BitVec 64` words) |
A batch = its list of ACTIVE rows (`active_indices`, batch.rs:1204); a stream = `List (List α)`.

## Proven (highlights)
* Any batching of an aggregate's input column gives the same accumulator as row-at-a-time
  (`vectorized_eq_rowwise`, `count_hom`…`stdev_hom`, `minmax_hom`); `count` = #non-null,
  `collect` = non-null filter, `count(*)` per-row = batch; avg count law; avg without
  overflow = plain sum; avg of empty/all-null = null; stdev empty = 0.
* scalar `max` = `max_batch` under a compare antisymmetric away from NaN, NaN included
  (`maxScalar_eq_batch`, `maxScalar_eq_batch_nan`; #2917 made both ignore NaN comparisons;
  history `pre2917_maxScalar_ne_batch_nan`).
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
3. **FIXED by #2912 (b033aecc6).** Was: aggregation.rs:615/:645 sorted with
   `partial_cmp().unwrap_or(Equal)`, not transitive with NaN (`pre2912_partialCmpEq_not_transitive`),
   so std sort panicked and the server crashed (C returns nan). Main sorts by `sort_cmp_f64`
   (:625/:651), a total preorder (`pctCmp_total`). Test `bug_percentile_with_nan_panics`.
4. **FIXED by #3007 (1d3b029e1, closes #2980).** Was: aggregation.rs:539 (and :253) at 289c201fe
   overwrote the percentile on every row, so the last one won (`pre3007_pct_last_wins`,
   `pre3007_pct_repro`). `percentileDisc(x, x/10.0)` over 1..10: C 1, Rust 10. Main stores it only
   while the collected list is empty (aggregation.rs:552-555, :257-260): `pct_first_wins`,
   `pct_stable`, `pct_repro_first_row`. Test `bug_percentile_row_dependent_argument_last_wins`;
   main's flow test `test_row_dependent_percentile_uses_first_row`.
5. **FIXED by #2951 (0912f4500, closes #2945).** Was: eval.rs:877 an aggregate inside a list literal
   `return`ed out of the stack machine (`pre2951_stackEval_return_drops_frame`,
   `pre2951_count_lit_and_size`). `[min(x),max(x)]` C [1,2], Rust 2; `[count(x),5]` C [2,5], Rust 2;
   `size([count(x)])` C 1, Rust type error. Main pushes the result and `continue`s (eval.rs:883-893):
   `evalStack_eq_evalRef`, `evalStack_list_of_aggs`, `main_min_max_list`, `main_count_lit_list`,
   `main_size_singleton_agg`. Test `bug_list_literal_of_aggregates_collapses`.

**Re-target to 8621212fc (2026-10-20): W2-agg-4 fixed by #3007.** `pctStep` now stores the
row's percentile only while no value has been collected, as aggregation.rs:552-555 does at
8621212fc; `pct_first_wins` proves the aggregate's percentile is the first non-null row's (leading
nulls skipped) and `pct_stable` that no later row changes it. The old overwrite model is kept as
`pre3007_pctStep`/`pre3007_pctFold`; the counterexample is renamed `pre3007_pct_last_wins`.
COVERAGE.tsv aggregation.rs lines re-pinned (+3/+6 after :254/:549).

**Re-target to 289c201fe (2026-10-20): W2-agg-5 fixed by #2951.** `evalStack` is now a postfix
stack machine (`compile`, `run`, `step`) whose aggregate leaf pushes its finalized value, as
eval.rs:883-893 does at 289c201fe; `evalStack_eq_evalRef` proves it equals the recursive reference
on every expression, and `evalStack_list_of_aggs` that a list literal of finalized aggregates yields
one element per item. The old `return` model is kept as `pre2951_evalStack`; the counterexample is
renamed `pre2951_stackEval_return_drops_frame`.

## Gaps / assumptions
Sort order inside `SortCmp` is a parameter `cv`; on main it is `Value::sort_cmp`, whose laws are
proven in proofs/value_order and instantiated here (`Discharge.lean`, which imports
proofs/value_order as a Lake path dependency).
Floats: no IEEE reasoning in the aggregates; the avg/sum/stdev overflow behaviour is checked with
`#eval` against C only. HashMap iteration order and the typed-column machinery (batch.rs
builders, classify_*, concat, gather) are not modelled. The scalar aggregate bodies are dead code
because `batch_agg` is always preferred.

**CONDITIONAL rows discharged (re-target to 4a4f2a72f, 2026-10-12).** `gkHash_consistent`
(`GroupKey::hash`) took "`Value`'s Hash agrees with the key equality" as `hc`; #2917 (7fdf06a0e)
made `GroupKey::eq` `sort_cmp == Equal` (aggregate.rs:94) and canonicalised NaN/f32 hashing, and
`Discharge.hc_main` proves `hc` from value_order `Main.main_hash_consistent`
(`gk_hash_consistent_main`; `veqMain_equiv`: the key equality is an equivalence).
`cmpKey_lawful`/`heapCmp_swap`/`heapCmp_refl` took `Lawful cv`; with #2912 (b033aecc6) `cv` is
`sort_cmp` and `Discharge.lawful_main` proves it from value_order `Main.scMain_total`
(`cmpKey_lawful_main`, `cmpKey_trans_main`, `heapCmp_main`). The sort.rs `cmp_key`/`cmp`/`cmp`
and aggregate.rs `hash` rows are PROVEN again. New fn `tiebreak` (sort.rs:237):
`tiebreak_strict`, `tiebreak_swap`.

## Re-target to ca36bfa15: #3150 is fixed (#3522)
`AggregateOp::next` now unbinds the accumulators (aggregate.rs:1184-1192) before it writes the key
aliases and the key row (:1193-1209), so a null key sharing an accumulator's slot id stays bound.
`finishGroup` is that order (`finishEval`, then unbind, then `finishTail`, then the outputs); the old
order is `preFinishGroup`, and the counterexample is renamed `pre3522_null_key_unbound` (with
`PreKeysBound`/`preEmitCell`). New: `fixed3522_null_key_bound` (`RETURN null AS x, count(1)`),
`keysBound_swap_3522` (`WITH 1 AS a, 2 AS b RETURN b AS x, a AS y, count(*)`), and the general
`finishGroup_keys_bound`/`keysBound_main`: for any keys/accumulators/evaluator, with no key name an
output name and the group evaluating, `KeysBound` holds on main. COVERAGE.tsv line pins refreshed
(`unbind_agg_accumulators` :1243, `hash_u64` :1267/:1271).

## Gap closed 2026-10-20: binder slot ids across an aggregating WITH/RETURN (`SlotReuse.lean`)
The aggregate model kept outputs, accumulators and pre-projection aliases in separate abstract
slots; the binder numbers each scope from 0 and `Row` is keyed by the bare id, so they collide.
`projLayout` mints the ids as the binder does (accumulators at `len` of the pre-projection scope,
binder.rs:1072-1078/:2193-2200; aliases 0.. in the new scope, :1149-1167), `finishCore` /
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
Fixed by #2912 (b033aecc6): ORDER BY over floats with NaN crashed the server (float lane
`unwrap_or(Less)`); main's lane is `sort_cmp_f64` (sort.rs:364; proofs/columnar `compareAt_spec`).
## Re-target to 5e475e6c9: #3443 is fixed (#3458)
`http_config` (load_csv.rs:223-234) now sets `https_only(true)` and `max_redirects(0)`, and
`open_csv_records` (:357-) turns a 3xx reply into the error "LOAD CSV does not follow redirects"
(:388-399), as C does. `LoadCsvRedirect.httpCfgMain` is that config and `loadRemote` that arm;
the old config/arm are `httpCfgPre3443`/`loadRemotePre3443`, and the refutations are renamed
`pre3443_redirect_uses_first_host_ip`, `pre3443_redirect_downgrades_to_http`. New:
`fixed3443_eachHopValidated` (main's `loadRemote` satisfies `EachHopValidated` for every DNS
answer and server) and `fixed3443_redirect_is_error`; `noRedirect_single` now holds for either
`https_only`. #3436/#3462 only shift line pins (load_csv.rs `next` 260/458, `new` 328).

## Gaps closed 2026-10-22 (#3443, #3453; b84473a7f)
**#3443 (`LoadCsvRedirect.lean`).** `LoadCsv.lean` treated a remote LOAD CSV as one request to the
validated URL. The ureq 3.4.2 agent (load_csv.rs:369-378) follows redirects: `http_config` (:217-225)
sets only timeouts, so `max_redirects = 10`, `https_only = false`; `PinnedResolver::resolve`
(:193-214) ignores the URI. `fetch` models the request loop. Property `EachHopValidated` (every
connection is https, to an address validated for its own host) is refuted:
`bug3443_redirect_uses_first_host_ip` (cross-host redirect connects to the first host's IP; live: TLS
name mismatch), `bug3443_redirect_downgrades_to_http` (live: `http status: 400`). Fix property:
`fixed_eachHopValidated` (https_only + per-request validate-and-pin, any server, any budget);
`noRedirect_single` (C: no redirects followed). `pinned_spec`/`httpConfig_spec` stand but cover only the
address bound / OnceLock idempotence (COVERAGE reasons say so). `V4`/`IP` now derive `DecidableEq`.
**#3453 (`IncludePendingExact.lean`).** `includePending_spec` proved each phase against whatever lists
it is handed; composed with `get_pending_nodes_with_labels` (pending.rs:1040-1055: `set_labels` only,
no deleted filter), `deleted_node` (:772, keeps `set_labels`) and `stage_node_labels` (:661, reached
from set.rs:403 without a committed-label check), property `IncludePendingExact` (duplicate-free, exactly
the live nodes whose effective labels contain the pattern) is refuted by the three live repros:
`bug3453_yields_deleted_node`, `bug3453_duplicate_row`, `bug3453_misses_split_multilabel`. Fix property:
`includePendingFixed_exact` (Phase 2 filtered by liveness/effective labels, minus Phase 1's nodes).
-/
