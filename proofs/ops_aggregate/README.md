# ops_aggregate: aggregation, DISTINCT, ORDER BY, SKIP and LIMIT

This project proves that the operators at the end of most read queries return what the query asks for. Several user-facing features depend on it:

- aggregation functions (`count`, `sum`, `avg`, `min`, `max`, `collect`, `stDev`, `percentileDisc`, `percentileCont`) and grouping;
- `DISTINCT` and `count(DISTINCT …)`;
- `ORDER BY`, including the top-k heap used for `ORDER BY … LIMIT k`;
- `SKIP` and `LIMIT`;
- `UNWIND` and `RETURN`/`WITH` projection;
- `LOAD CSV`'s URL and file checks.

The engine runs most of these operators a batch at a time, and the aggregate operator has a fast columnar path. The central question is whether the batched code gives the same answer as the simple row-by-row meaning of the query, wherever the batch boundaries fall.

The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorAggOps.lean`](FalkorAggOps.lean) and in the table at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does batching change an aggregate's result?**

No. Every aggregation kernel is a left fold over its input that skips `Null`. `vectorized_eq_rowwise` proves that feeding the column in any number of batches, split anywhere, gives the same accumulator as feeding it one row at a time. The per-function versions are `count_hom`, `sum_hom`, `collect_hom`, `avg_hom`, `stdev_hom` and `minmax_hom`.

Some consequences the project also proves directly:

- `count(x)` is the number of non-null values and `collect(x)` is the non-null values in order (`countBatch_spec`, `collectBatch_spec`);
- `avg` over an empty or all-null input is `null` (`avg_empty`), and below the overflow threshold it is a plain running sum (`avg_no_overflow_is_sum`);
- `stDev` of an empty input is 0 (`stdev_empty`).

**2. Is grouping correct?**

Yes. `groupAll_spec` proves that each group holds exactly the rows with its key, in arrival order, so the aggregate of group `k` is the aggregate of "rows where key = k". `null` is an ordinary key, as in C. A keyed aggregate over no rows has no groups (`keyed_empty`); a keyless one has exactly one (`keyless_one_row`), which is why `MATCH (n:Missing) RETURN count(n)` returns one row with 0.

Grouping uses a `HashMap<GroupKey, _>`, which is only correct if equal keys hash the same. `gk_hash_consistent_main` proves that on main, using `value_order`'s `main_hash_consistent`.

**3. Does `ORDER BY … SKIP s LIMIT k` return the right window?**

Yes. For `ORDER BY … LIMIT k` the engine does not sort everything; it keeps a bounded heap of `k + s` rows. `topk_eq_take_sort` proves that the heap returns exactly the first `cap` rows of the full sort, and `sort_skip_limit` proves the `SKIP s LIMIT k` window equals the same window of the full sort. `heapCmp_seq_strict` proves two rows with different arrival order never tie in the heap, so the result is fully determined.

All of this needs the value comparison to be a valid total order. On main it is `Value::sort_cmp`, and `lawful_main` imports that fact from `value_order` (`scMain_total`).

`LIMIT k` is `take k` and `SKIP s` is `drop s` over the row stream, whatever the batch sizes (`limitOp_spec`, `skipOp_spec`); `LIMIT 0` returns nothing (`limit_zero`).

**4. Does `DISTINCT` keep exactly one row per value?**

Only if the hash it keys on never collides. `dedup_injective` proves `DISTINCT` is correct when the hash is injective. `dedup_collision_drops` shows the other side: with a collision, a row is dropped. The engine keys on a 64-bit FxHash, so this is the known #2780. The project also found two new places where DISTINCT state is shared wrongly (see [Bugs found](#bugs-found)).

**5. Is `LOAD CSV` safe?**

For the parts that are modelled, yes:

- `validateRemote_spec`: an accepted remote URL is `https://` and every address it resolves to is public, so the request cannot reach an internal address (SSRF);
- `erRun_le`: a remote read never delivers more than the byte limit;
- `resolvePath_spec`: an accepted `file://` path stays inside the import folder, and only `file://` and `https://` are accepted.

DNS resolution and path canonicalisation are parameters of the model, not proven.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `vectorized_eq_rowwise` | Any batching of an aggregate's input gives the same accumulator as row-at-a-time | `Agg.lean` |
| `count_hom` … `minmax_hom` | The same, for each concrete kernel | `Agg.lean` |
| `maxScalar_eq_batch` | The scalar `max` body and `max_batch` agree under a comparison that is antisymmetric away from NaN | `Agg.lean` |
| `groupAll_spec` | Each group holds exactly the rows with its key, in order | `Ops.lean` |
| `gk_hash_consistent_main` | On main, equal `GroupKey`s hash the same (the `HashMap` contract) | `Discharge.lean` |
| `topk_eq_take_sort`, `sort_skip_limit` | The top-k heap returns the first `k + s` rows of the full sort, so the `SKIP`/`LIMIT` window is right | `Ops.lean` |
| `lawful_main`, `cmpKey_lawful_main`, `heapCmp_main` | The sort comparators are lawful on main, because `sort_cmp` is | `Discharge.lean` |
| `classifySortKey_get` | The typed sort-key column reads back exactly the evaluated key, with nulls in place | `SortKey.lean` |
| `limitOp_spec`, `skipOp_spec` | `LIMIT` is `take`, `SKIP` is `drop`, for any batching | `Ops.lean` |
| `project_flatten`, `unwind_batches`, `unwind_null` | Project and Unwind do not depend on batching; `UNWIND null` gives no rows | `Ops.lean` |
| `dedup_injective` | `DISTINCT` is correct exactly when its hash is injective | `Ops.lean` |
| `drain_all` | The aggregate's `next` emits every group once, 1024 per batch | `AggNext.lean` |
| `finishGroup_outputs` | Output bindings of a finished group take precedence | `AggNext.lean` |
| `bug3375_output_overwrites_next_acc` … `bug3368_count_acc_in_merge_slot` | With the ids the binder really assigns, `next` breaks `AggRowSpec` in five open issues | `SlotReuse.lean` |
| `unbind_after_zero` | Every accumulator the aggregate zeroes is unbound again afterwards | `AggState.lean` |
| `pctCmp_total` | The percentile sort comparator is a total preorder, so it cannot panic on NaN | `Ops.lean` |
| `pct_first_wins`, `pct_stable` | A per-row percentile argument is taken from the first non-null row and never overwritten, as in C (#3007) | `Agg.lean` |
| `validateRemote_spec`, `erRun_le`, `resolvePath_spec` | LOAD CSV: public addresses only, byte cap, path containment | `LoadCsv.lean` |
| `fixed3443_eachHopValidated` | Main's `loadRemote` (https_only, no redirects, pinned resolver, since #3458) makes only https connections to addresses validated for their own host (the property #3443 broke) | `LoadCsvRedirect.lean` |
| `fixed_eachHopValidated` | The general fix shape: with `https_only` and a resolver that validates and pins per request, every hop of any redirect chain is https and goes to an address validated for its own host | `LoadCsvRedirect.lean` |
| `includePendingFixed_exact` | A Phase 2 that filters by liveness and effective labels, and skips what Phase 1 emitted, makes MERGE's node scan see each live matching node exactly once (the property #3453 breaks) | `IncludePendingExact.lean` |

### How `vectorized_eq_rowwise` is proved

1. **Model the type.** `V` is the slice of `Value` the aggregates see: `null`, `int`, `flt` and `other` for any non-numeric value.
2. **Model the kernels.** Each `*_batch` kernel in `aggregation.rs` is a left fold over the batch slice that skips `Null`. That is `skipFold step`, where `step` is the kernel's per-value update. The concrete kernels (`countBatch`, `sumBatch`, `avgBatch`, `stdevBatch`, `mmBatch`, `collectBatch`) are written line by line from the Rust.
3. **Model the two drivers.** `rowDriver` is the per-row path (`run_agg_expr`): skip a null argument, else call the kernel on a one-element slice. `batchDriver` is the keyless fast path: one kernel call per input batch.
4. **Two small lemmas.** `skipFold_append` says folding over `xs ++ ys` is folding over `xs` then over `ys`. `rowDriver_eq_batch` says the per-row driver equals one fold over the whole list.
5. **Combine.** Folding batch by batch is folding over the concatenation (`batchDriver_eq_join`, by `skipFold_append`), which is the per-row driver (`rowDriver_eq_batch`).

There are no hypotheses: the result holds for every kernel shaped like a null-skipping fold, and every concrete kernel is one.

## Bugs found

Every bug was reproduced against a live server before it was recorded (see [`../FINDINGS.md`](../FINDINGS.md)). The Rust repros are in `graph/tests/lean_ops_aggregate.rs`, kept locally and not committed.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W2-agg-1 | `distinct_fold_clear_overcounts` | `CALL {}` clears the runtime-wide DISTINCT state after each run (`apply.rs:283`), wiping an outer `count(DISTINCT)`'s state every batch. `UNWIND range(1,3000) AS i CALL { WITH i RETURN i % 2 AS v LIMIT 1 } RETURN count(DISTINCT v)` gives 6; C gives 2 | **Open** |
| W2-agg-2 | removed (17ccc5412), `distinct_by_group_id_exact`, `fixed2977_collision_example` | **Fixed by #2977 (c652e505e, closes #2969): the state is keyed by a group id (the insertion index; `aggregate.rs:709`, `:970`).** Was: per-group DISTINCT state keyed by the FxHash of the group key (`aggregate.rs:715`, `:959` at ca36bfa15), so two groups whose keys collide share one seen-set. Keys `[0,0]` and `[1,-1452335207727870361]`: C counts 1 and 1, Rust 1 and 0 | Fixed by #2977 |
| W2-agg-3 | removed (17ccc5412) | `percentileDisc`/`percentileCont` sorted with `partial_cmp().unwrap_or(Equal)`, which is not transitive with NaN; the sort panicked and the server crashed | Fixed by #2912 (`pctCmp_total`) |
| W2-agg-4 | removed (17ccc5412) | A per-row percentile argument was overwritten by every row, so the last row won. `percentileDisc(x, x/10.0)` over 1..10: C 1, Rust 10. Low severity | Fixed by #3007 (`pct_first_wins`, `pct_stable`, `pct_repro_first_row`; main's test `test_row_dependent_percentile_uses_first_row`) |
| W2-agg-5 | removed (17ccc5412) | An aggregate inside a list literal `return`ed out of the expression stack machine (`eval.rs:877`). `[min(x), max(x)]` gave `2` where C gives `[1,2]`; `size([count(x)])` was a type error | Fixed by #2951 (`evalStack_eq_evalRef`, `evalStack_list_of_aggs`, `main_min_max_list`, `main_count_lit_list`, `main_size_singleton_agg`; main's test `test_list_literal_of_aggregates`) |
| #3375 | `bug3375_output_overwrites_next_acc`, `bug3375_avg_panics` (property `OutputsOwnAcc`) | The binder gives `count(x)`'s output the same id as `sum(x)`'s accumulator, and `next` writes the output into `combined` before `sum` reads it. `UNWIND [1,2,3] AS x RETURN 'a', 'b', count(x), sum(x)`: Rust 3, 3; C 3, 6. With `avg` the server panics (`finalize_avg expects a list`) | **Open** (PR #3374) |
| #3150 | removed (17ccc5412) (property `KeysBound`) | `RETURN null AS x, count(1)`: the accumulator had id 0, the id of `x`; unbinding it unbound `x`, the null column was dropped and `reply.rs:655` panicked. With `'a' AS x` the value survived unbound, so the reply still printed it | **Fixed by #3522** (accumulators are unbound before the keys are written): `finishGroup_keys_bound`, `keysBound_main`, `fixed3522_null_key_bound`, `keysBound_swap_3522` |
| #3333 | `bug3333_alias_binds_next_var` (property `EmitsOnlyScope`) | `… (x)-[:R]->(m) WITH m, count(*) AS c1 MATCH (m)-[r]->(b …)`: the pre-projection alias of `m` (id 2) is bound on the emitted row, and `b` gets id 2, so `b` starts bound to `m`. Rust `[]`, C `[3,1,4]` | **Open** |
| #3369 | `bug3369_count_acc_in_merge_slot` (property `EmitsOnlyScope`) | Two `count(*)` accumulators are left, unbound but present, at the ids of the next segments' MERGE nodes (2 and 3). The MERGE that then treats them as bound is in `merge.rs`, which this project does not model | **Open** |
| #3368 | `bug3368_count_acc_in_merge_slot` (property `EmitsOnlyScope`) | Same cause as #3369: the counts are left at id 2, which is `b` in the second segment and `c` in the last. Rust returns `[B] b1` for `c`; C `[C] c1` | **Open** |
| #3443 | removed (17ccc5412) (property `EachHopValidated`) | ureq followed redirects (default `max_redirects = 10`), but `PinnedResolver` ignores the URI and returns the first host's addresses, and `https_only` was off. A cross-host redirect connected to the first host's IP (live: TLS "certificate not valid for name"); a redirect to `http://` was attempted (live: `http status: 400`). C follows no redirect | **Fixed by #3458** (`https_only(true)`, `max_redirects(0)`, a 3xx is an error): `fixed3443_eachHopValidated`, `fixed3443_redirect_is_error` |
| #3453 | `bug3453_yields_deleted_node`, `bug3453_duplicate_row`, `bug3453_misses_split_multilabel` (property `IncludePendingExact`) | MERGE's pending-node scan reads `set_labels` only: a node deleted after `SET n:B` is matched (`MERGE (m:B)` creates nothing), a re-`SET` committed label yields the node twice (ON MATCH runs twice), and `MERGE (m:A:B)` misses a node with `A` committed and `B` staged (a duplicate is created). C is right in all three | **Open** (#3453) |
| — | `dedup_collision_drops` | `DISTINCT`, `count(DISTINCT)` and the MERGE cache keep only a 64-bit FxHash; a collision drops a row | **Present on main** (#2780 closed as not planned) |
| — | removed (17ccc5412) | The scalar `max` body and `max_batch` disagreed on NaN | Fixed by #2917 (`maxScalar_eq_batch_nan`); harmless anyway, the scalar body is never called |
| — | `one_child_not_unbound` | An aggregate call with only an accumulator child would be zeroed but never unbound, leaking its accumulator downstream | Latent: no parsed call has that shape |

## Gap closed 2026-10-20: slot ids shared across the projection

Until this date the aggregate model kept outputs, accumulators and pre-projection key aliases in separate slots, so `finishGroup_outputs` and `unbind_other` could not see that the binder gives them overlapping ids. `SlotReuse.lean` now mints the ids as the binder does and models that `unbind` keeps the value (a value-present-but-unbound column). Five open issues come from this one cause; see the bug table. The MERGE half of #3369 and #3368 (a value-only Int read as a bound node) lives in `merge.rs` and is not modelled here.

### CONDITIONAL rows, and how they were discharged

Until #2912 and #2917, four rows rested on hypotheses about `Value` that were false on main:

- `GroupKey::hash` assumed "`Value`'s hash agrees with key equality" (`hc` in `gkHash_consistent`). `value_order` refuted it on the old code.
- sort.rs `cmp_key` and the two `cmp`s assumed the value comparison is lawful (`Lawful cv`). It was not, before #2912.

They were tagged CONDITIONAL. On main, `Discharge.lean` proves both hypotheses from `value_order` (`hc_main` from `main_hash_consistent`, `lawful_main` from `scMain_total`), so the rows are PROVEN again. This project has no CONDITIONAL rows today.

## Files

| File | What's in it |
|---|---|
| `Agg.lean` | The aggregate value type `V`, the null-skipping fold, both drivers, every `*_batch` kernel, the batch-independence theorems, `max` and NaN, the percentile first-row rule (`pct_first_wins`) and its pre-#3007 history |
| `LoadCsvRedirect.lean` | LOAD CSV's ureq redirect loop and pinned resolver: #3443, `fixed3443_eachHopValidated`, `fixed_eachHopValidated` |
| `IncludePendingExact.lean` | `IncludePendingOp` composed with `Pending`'s label scans: #3453 counterexamples, `includePendingFixed_exact` |
| `Ops.lean` | Limit, Skip, the full sort and top-k heap, Project, Unwind, grouping, DISTINCT and its counterexamples, the stack-machine `return` bug, percentile index bounds and comparator |
| `SortCmp.lean` | `OrderedKey` and `HeapEntry` comparators, `compare_row_content`, `tiebreak`, over an abstract value comparison `cv` |
| `SortKey.lean` | `classify_sort_key`: the typed key column reads back the evaluated key |
| `AggPlan.lean` | Aggregate operator setup: `subtree_has_aggregate`, `analyze`, `analyze_agg_tree`, `args_for`, `new` |
| `AggState.lean` | Accumulator zeroing and unbinding, `GroupKey` equality and hash, `hash_u64` |
| `AggNext.lean` | Column extraction and the aggregate's `next`: groups finished and emitted in batches |
| `SlotReuse.lean` | Binder ids for one projection (`projLayout`), what the emitted row stores and what a reader sees (`finishVals`, `cell`), the properties `OutputsOwnAcc`, `KeysBound`, `EmitsOnlyScope`, and the #3375/#3150/#3333/#3369/#3368 counterexamples |
| `Discharge.lean` | Instantiates `SortCmp` and `AggState` with main's `sort_cmp` and hash from `value_order` |
| `Bitmap.lean` | `NullBitmap` from `batch.rs`: set, clear, `all`, `from_values` |
| `Ctors.lean` | `new` of Limit, Skip, Distinct, Project, Unwind, Sort and Commit |
| `StreamOps.lean` | Filter, IncludePending and ProcedureCall |
| `LoadCsv.lean` | LOAD CSV: SSRF filter, byte cap, pinned resolver, CSV records, path containment |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

This project depends on [`../value_order`](../value_order) (a Lake path dependency), which in turn depends on [`../core`](../core). `Value`, the float model, `sort_cmp` and the hash come from there; they are imported, not copied.

## Coverage

`COVERAGE.tsv` has 157 rows:

| Bucket | Rows |
|---|---|
| PROVEN | 87 |
| MODELLED | 13 |
| NOT COVERED | 57 |

All operator files (`aggregate.rs`, `sort.rs`, `limit.rs`, `skip.rs`, `distinct.rs`, `project.rs`, `unwind.rs`, `load_csv.rs`, `filter.rs`, `include_pending.rs`, `procedure_call.rs`, `commit.rs`) are fully PROVEN. The rest:

- **NOT COVERED, `batch.rs` (47 rows):** the model treats a batch as its list of active rows, so the typed columns, builders, `concat` and `gather` are abstracted away. `compare_at` and `compare_rows_at` are proven in `value_order` and `columnar` instead.
- **MODELLED, `batch.rs` (11 rows):** accessors such as `active_indices`, `value_at` and the row iterators are what "a batch is its active rows" stands for; the `agrees_*` Rust tests exercise them.
- **NOT COVERED, `aggregation.rs` (10 rows):** the scalar aggregate bodies (`count`, `sum`, `min`, `avg`, `percentile`, `stdev`, `collect`) are dead code, because every aggregate also registers a batch kernel and the operator always prefers it. Also the registration table, `modf` (float library split) and `finalize_stdevp` (checked only by a Rust test).
- **MODELLED, `aggregation.rs` (2 rows):** `about_to_overflow` and `finalize_stdev` are modelled and checked against C with `#eval`, without float reasoning.

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Floats.** There is no IEEE reasoning in the aggregates. Overflow handling in `avg`, and the results of `sum` and `stDev` on floats, are checked against C with `#eval` only.
- **Sort keys.** The top-k proofs model rows by their position in the sort's total order (`Nat`). That the real comparator is a total order is proven separately, in `value_order`, and connected through `Discharge.lean`.
- **The columnar batch.** Typed lanes, builders, `concat` and `gather` are not modelled; `proofs/columnar` covers the batch layout.
- **`HashMap` iteration order.** Groups may come out in any order; the theorems hold for every order.
- **LOAD CSV I/O.** DNS, HTTP and path canonicalisation are parameters. The theorems say what the code does with their answers. Since #3443 the ureq agent's redirect loop is modelled (`LoadCsvRedirect.fetch`, with ureq 3.4.2's `max_redirects`/`https_only` defaults); TLS, the response body and `Location` parsing are not.

## Build and check

```bash
cd proofs/ops_aggregate && lake build    # this project (also builds ../value_order and ../core)
bash proofs/lean_ci.sh ops_aggregate     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                  # repo-wide coverage report
```

**When you change `ops/aggregate.rs`, `functions/aggregation.rs`, `ops/sort.rs`, `ops/distinct.rs`, `ops/limit.rs`, `ops/skip.rs`, `ops/project.rs`, `ops/unwind.rs` or `ops/load_csv.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If you change `Value::sort_cmp` or `Hash for Value`, rebuild this project too: `Discharge.lean` depends on `value_order`'s theorems about them. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
