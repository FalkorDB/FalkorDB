# columnar: the batch layer computes the same thing as row-at-a-time evaluation

The runtime moves rows between operators in *batches*: a set of typed columns, an optional selection vector naming the active rows, and per-row origins. Expressions run *columnar*: one kernel over a whole column instead of one `Value` per row. This project proves that each columnar operation gives the same answer as its row-at-a-time definition. It also records the two places where it does not.

User-facing behaviour that depends on it:

- `WHERE`, `SKIP` and `LIMIT`, which build selection vectors over a batch;
- every expression evaluated in batch mode: comparisons, `+ - * / %`, `NOT`, `CASE`, function calls, variable reads;
- operators that fan one input row out into many output rows (Expand, ExpandInto, UNWIND, procedure calls) through `BatchedResultEmitter`;
- `UNION` and `CALL {}` bodies, which concatenate batches with `Batch::concat`;
- operators that drop back to per-row `Row`s (`BatchRow::to_owned_row`).

The Rust is in `graph/src/runtime/batch.rs`, `vectorized.rs`, `vector_expr.rs`, `row.rs` and `ops/batched_result_emitter.rs`. Each file under `Columnar/` starts with its own Lean↔Rust `file:line` table, and [`Columnar.lean`](Columnar.lean) is the project report. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does a selection vector select the right rows?**

`WHERE`, `SKIP` and `LIMIT` don't copy rows. They write the indices of the surviving rows into a `u16` selection vector. The project proves that the active indices are always in range and strictly increasing, and that Filter, Skip and Limit, and any composition of them, keep exactly `filter`, `drop` and `take` of the previously active rows, in order (`filterSel_spec`, `takeSel_spec`, `dropSel_spec`, `skip_then_limit`, `filter_then_filter`).

All of these need the batch to have at most 65536 rows, because of the `as u16` cast. `skip_u16_wraps` shows what happens past that: skipping one row of a 65537-row batch selects row 0 again, so one row is lost and another is duplicated. Batches stay below the limit only because every producer caps them at `BATCH_SIZE` (1024). Nothing checks it. See [Bugs found](#bugs-found).

**2. Do gather and compaction keep the rows they should?**

Yes:

- `gather` reads row `idx[k]` into row `k` (value, bound bit and origin) and never panics on in-range indices (`gather_obs`, `gather_isSome`, `gather_wf`);
- two gathers compose into one (`gather_gather_obs`);
- `into_compacted` gives `active_len` dense rows equal to the active rows in order (`compact_spec`);
- `write_column` is a scatter: active row `k` gets `vals[k]` and inactive rows keep their value (`writeColumn_spec`).

**3. Does a columnar expression return what per-row evaluation returns?**

Yes, for every lane:

- **Comparisons.** `compareColumns_agree`: `compare_columns` equals the per-row comparison at every row. That holds for the int lane, the float lane, int→float promotion, a flipped constant (`5 < n.x` evaluated as `n.x > 5`), the pairwise lanes and the generic `compare_value` loop. NaN makes every ordering false and `<>` true, as in the per-row path.
- **Arithmetic.** `arithmetic_agree`: the int and float lanes equal `Value` arithmetic row by row, including the *first* error. A zero divisor sends `/` and `%` to the generic lane, and `Int ⊕ Int` never takes the float lane.
- **`CASE`.** `caseCol_agree` gives the same result, and `caseArm_then_only_on_hits` proves the laziness that matters in practice: in `CASE WHEN n.d <> 0 THEN 1 / n.d ELSE 0 END`, the `THEN` is only evaluated on rows whose `WHEN` matched, so the query can't divide by zero.
- **The rest.** `NOT` (`negate_agree`), scalar function folding (`foldCall_agree`), variable reads (`evalVariable_agree`) and the Filter mask (`filterPass_agree`).

`eval_sound` ties these together: for every expression tree, whatever column `VectorEval::eval` returns equals the per-row evaluator on each row. The guards send each node to exactly one arm or to `eval_per_row`.

**4. Does the emitter lose, duplicate or reorder rows?**

No. `emitLazy_spec` proves that one `emit_lazy` call takes the next `min(ceiling, remaining)` `(row, item)` pairs and leaves the rest untouched. `emitAll_spec` proves that the emitted batches, concatenated, are exactly the per-row expansion, each one non-empty and at most `BATCH_SIZE` rows. The ceiling stays in `[1, 1024]` and grows back after a record cap lowers it (`grow_bounds`, `grow_reaches`). Every packed parent index is an active row, so `finish_batch`'s gather never panics (`batch_rows_active`).

There was a suspicion that ExpandInto's `set_selection(0..remaining)` trim could select rows that don't exist. It is refuted: `expandTrim_ok` shows that the output is dense, the trim fires only when `remaining < n`, and every index is `< n < 65536`.

**5. Is `Batch::concat` the same as pushing the rows one at a time?**

Only when no input batch is empty. `concat_obs_eq` proves that, if every input batch has an active row, then the value and the bound bit at every (row, slot) equal those of the per-row `BatchBuilder` concat. Row counts and origins also match (`concat_len`, `concat_origins`). With an empty batch in the input it differs (`empty_batch_binds`, bug 1 below).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `eval_sound` | For every expression tree, `VectorEval::eval`'s output column equals per-row evaluation row by row, given the lane theorems | `Dispatch.lean` |
| `compareColumns_agree` | Every comparison lane equals the per-row comparison at every row | `VectorExpr.lean` |
| `arithmetic_agree` | The `+ - * / %` lanes equal `Value` arithmetic, values and first error | `Arith.lean` |
| `caseCol_agree`, `caseArm_then_only_on_hits` | Columnar `CASE` equals per-row `CASE`, and a `THEN` runs only where its `WHEN` matched | `Lanes.lean` |
| `filterSel_spec`, `takeSel_spec`, `dropSel_spec` | `WHERE`, `LIMIT` and `SKIP` keep exactly the right active rows, in order (≤ 65536 rows) | `Batch.lean` |
| `compact_spec`, `gather_gather_obs`, `writeColumn_spec` | Compaction, gather and scatter move exactly the right cells | `Batch.lean` |
| `concat_obs_eq` | `concat` equals the per-row builder at every slot, when no input batch is empty | `Concat.lean` |
| `emitLazy_spec`, `emitAll_spec` | The emitter outputs the `(row, item)` stream with nothing lost, duplicated or reordered, in non-empty batches of ≤ 1024 rows | `Emitter.lean` |
| `expandTrim_ok` | ExpandInto's record-cap trim only selects existing rows | `Emitter.lean` |
| `gather_spec` | Each of the seven `GatherItem` lanes builds row `k` from item `k` | `Gather.lean` |
| `compareAt_spec` | `Column::compare_at`'s float lane is antisymmetric, NaN included (since #2912) | `BatchMisc.lean` |
| `emit_origin_sound` | Every emitted row carries the origin of the parent row it came from (the `CALL {}` record boundary of #2845) | `CallBoundary.lean` |

### How `eval_sound` is proved

1. **Model the tree.** `Node` is an expression node: a `Kind` (the `ExprIR` variants `eval` inspects) and its children.
2. **Model the function.** `eval` is `VectorEval::eval`, with its arms in source order and the same guards: the child count, the `CmpOp::from_expr_ir` check, the function-invocation filter and the `hasLabels` attempt. A node no arm accepts goes to `perRow`, which is `eval_per_row`. Since #3335 `Paren` has no arm of its own, because the binder removes every `Paren`.
3. **State what each arm promises.** `Lawful` has one field per arm: given children that are sound, that arm's output column is sound. These fields are what the lane theorems above prove (`compareColumns_agree`, `arithmetic_agree`, `caseCol_agree`, …).
4. **Induction on the tree.** The proof follows the Rust recursion. Children are smaller, so they are sound by induction, and the arm's `Lawful` field finishes the case.

`Dispatch.lean` takes `Lawful` as an argument. Since 2026-10-21 the walk is also modelled over proofs/expr_semantics' expression tree `E` (`VecEval.lean`), and `CoreBridge.lean` **builds** the `Lawful` instance over a batch: every field is proved from the lane theorems against `ExprEval::eval_node` itself (`AndOrLane.lean`, `CaseLane.lean`, `FuncLane.lean`, `HasLabelsFn.lean`, plus `CoreOps.lean` for the shared comparison/arithmetic forms), so `vec_sound` says: whenever `VectorEval::eval` returns a column, entry `i` is the per-row evaluator on row `i`. The property fast path's runtime contract (bulk fetch = per-row read) is proved in `RuntimeReads.lean`. Two further assumptions:

- `f64` is abstract (`FloatModel`), and its one IEEE fact, `partial_cmp` antisymmetry, is a class field;
- function calls are pure functions of their arguments (the registry's functions are a parameter of the tree; `hasLabels`, which the dispatch special-cases by name, is modelled, `HasLabelsFn.lean`).

## Bugs found

Both bugs were confirmed on the real code through the public API. No Cypher query has triggered either of them yet, so they are **latent**. They are recorded in [`Columnar.lean`](Columnar.lean), not in [`../FINDINGS.md`](../FINDINGS.md).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `empty_batch_binds` | `Batch::concat`'s fallback loop (`batch.rs` ~1008) sets `bound_anywhere` from batches with **no active rows**. A slot bound only in an empty batch comes out bound (`Some(Null)`), where the per-row concat leaves it unbound (`None`). The fix is to skip `active_len() == 0` batches, as `concat_typed_column` already does | **Open**, latent (still present at 289c201fe) |
| 2 | `toOwned_trailing_unbound`, `evalVariable_beyond` | `BatchRow::to_owned_row` drops trailing unbound columns, so `value_at(i)` is `Some(Null)` on the view but `None` on the owned row ("Variable not found"). `eval_variable` reads Null past the column space where the per-row view errors. 12 OPTIONAL/CALL/comprehension queries were tried and all were correct | **Open**, latent |
| — | `pre2973_withBinding`, `withBinding_eq_pre2973` | #2922: LLVM dropped the `cap < BATCH_SIZE` guard of the pack-ceiling `match` on aarch64-apple-darwin, so a cap above a batch came out as the ceiling itself (`miscompiled_differs`). #2973 rewrote it as `cap.clamp(1, BATCH_SIZE)`; the two forms compute the same ceiling for every cap | Fixed by #2973 (a source-form change; the model is the clamp, the guarded arm is history) |
| — | `skip_u16_wraps` | Selection vectors are `u16`. Past 65536 rows the cast wraps, a row is lost and another duplicated. Batches are kept small only by the `BATCH_SIZE` caps, which is exactly the guard the LLVM miscompile in #2922 dropped on macOS arm64 (`miscompiled_differs`) | Hazard, not a bug today |
| — | `finishLen_drops` | A no-alias emitter over a zero-column parent with no origins emits a 0-row batch. Since #2845 a parent that carries origins is gathered (`finishLen_origins_kept`) | Hazard, unreachable: ExpandInto always has columns |
| — | `classifyColumn_lossy` | `classify_column` can lose information | Harmless: it has no caller |
| — | `pre2912_compareAt_nan_not_antisymmetric` | Before #2912 the float lane of `compare_at` was `unwrap_or(Less)`, which isn't antisymmetric with NaN (issue #2891) | Fixed by #2912 (`compareAt_spec`) |

One more difference that isn't counted as a bug: columnar `CASE` or `AND` can report a different row's error message than the per-row path. Both paths still raise an error.

This project has no CONDITIONAL rows.

## Files

| File | What's in it |
|---|---|
| `Column.lean` | `FloatModel`, the engine value (proofs/core `FalkorCore.V F`), `Column` and its typed lanes, `NullBitmap` as a bit list, `classify_*`, `extend_active_slice` |
| `Batch.lean` | `Batch`, its well-formedness `WF`, selection vectors and their producers (Filter, Limit, Skip), gather, `into_compacted`, `set_column`, `write_column`; `skip_u16_wraps` |
| `Row.lean` | `Row` (`insert`, `take`, `merge`, `clone_with`) and `BatchRow::value_at`/`to_owned_row`; bug 2 |
| `Concat.lean` | `Batch::concat` and `concat_typed_column` against the per-row `BatchBuilder`; bug 1 |
| `Emitter.lean` | `BatchedResultEmitter` (`emit_lazy`, the ceiling, `finish_batch` length) and the ExpandInto trim |
| `Gather.lean` | The `GatherItem` trait proven once (`gather_spec`), then its seven impls; `RowIter` |
| `VectorExpr.lean` | `CmpOp`, the comparison kernels, `ExprColumn`, `union_nulls`, `compare_columns` |
| `Arith.lean` | `int_lane`, `float_lane`, `arithmetic` against `Value` `+ - * / %` |
| `Lanes.lean` | `NOT`, `CASE`, function folding, `eval_variable`, the Filter mask |
| `BatchMisc.lean` | The rest of `batch.rs`: `compare_at`, builder push paths and merge plan, constructors and accessors, `merge_over_input`, `rows_only`, `has_origins`, `set_argument_batch`, `BatchOp::next` |
| `CallBoundary.lean` | #2845's columnar side: `rows_only`, `has_origins`, the empty-projection fast path, origins through the emitter (from the retired `pr2845_review`) |
| `Dispatch.lean` | `VectorEval::eval` over an abstract `Kind`/`Node` and its leaves (`eval_values`, `property_values`, `eval_has_labels`, `eval_per_row`); `eval_sound` |
| `VecEval.lean` | `VectorEval::eval` / `eval_values` over expr_semantics' `E`; `eval_sound`, `evalValues_sound` |
| `CoreOps.lean` | the engine's operations as expr_semantics' `Ops`; `all_equals` / `all_not_equals` / comparisons = `compare_values`, `NOT`, `CASE` test |
| `AndOrLane.lean`, `CaseLane.lean`, `FuncLane.lean`, `HasLabelsFn.lean`, `LaneWF.lean` | `eval_and_or`, `eval_case`, `eval_function`, `eval_has_labels` (and the registry's `hasLabels`) against `ExprEval`; every lane returns `len` rows |
| `CoreBridge.lean` | the `Lawful` instance over a batch (`lawful`, `vec_sound`) |
| `RuntimeReads.lean` | the runtime's attribute reads: bulk fetch = per-row read (`matNode_eq`, `matRel_eq`, `propReads`) |
| `EvalBatch.lean` | expr_semantics' `BatchLaws` for `eval_batch`, discharged |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project imports `../core` (the engine value) and `../expr_semantics` (the expression tree and `ExprEval`) through Lake path dependencies.

## Coverage

`COVERAGE.tsv` has 165 rows: 66 in `batch.rs`, 39 in `batched_result_emitter.rs`, 26 in `vector_expr.rs`, 21 in `row.rs`, 7 in `vectorized.rs`, 4 in `runtime.rs` (the `materialize_*` reads) and 2 in `eval.rs` (`eval_batch`).

| Bucket | Rows |
|---|---|
| PROVEN | 147 |
| MODELLED | 16 |
| NOT COVERED | 2 |

- **MODELLED:** small `row.rs` accessors and constructors (`new`, `get`, `len`, `is_empty`, `is_bound_by_id`, the `RowView` trait declarations), `NullBitmap::set`/`clear` (bits as a list, with the word packing proven in `ops_aggregate`), and the emitter's `seed`/`reset`. Each is a definition in the model with no theorem of its own.
- **NOT COVERED:** `Row::has_bindings` and `Row::hash`, which depend on `BitSet` and hashing, both covered in `runtime_ds`.

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Floats.** `f64` arithmetic and ordering are abstract (`FloatModel`). Only `partial_cmp` antisymmetry is assumed.
- **Values.** `compare_value` and arithmetic beyond numerics and null are an abstract `other` function, passed through unchanged by both paths.
- **Batch internals.** `MergePlan`, `push_merged` and `push_planned` are modelled in `BatchMisc.lean` but not proven equal to a per-row definition. Property fetch, operator dispatch and profiling are not modelled.
- **Emitter layout.** The emitter's lane layout (transpose, shared alias) is not modelled; only the `(row, item)` stream and the gather are.
- **Purity.** The `CASE` and function proofs assume pure sub-expressions.
- **Word packing.** `BitSet` and `NullBitmap` word packing are proven in `runtime_ds` and `ops_aggregate`, not here.

## Build and check

```bash
cd proofs/columnar && lake build      # this project
bash proofs/lean_ci.sh columnar       # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh               # repo-wide coverage report
```

**When you change `batch.rs`, `row.rs`, `vectorized.rs`, `vector_expr.rs` or `batched_result_emitter.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its model in the matching `Columnar/*.lean` file, including the `file:line` in that file's header table.
3. Re-run the build.

Two changes need extra care. A new arm in `VectorEval::eval` needs a new `Lawful` field and a lane theorem. Anything that lets a batch grow past 65536 rows breaks every selection theorem.
