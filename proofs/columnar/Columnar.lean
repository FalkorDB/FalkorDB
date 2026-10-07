import Columnar.Column
import Columnar.Batch
import Columnar.Row
import Columnar.Concat
import Columnar.Emitter
import Columnar.VectorExpr
import Columnar.Arith
import Columnar.Lanes
import Columnar.Gather
import Columnar.BatchMisc
import Columnar.Dispatch

/-
# Columnar batch layer: every columnar operation = its row-at-a-time definition

Lean 4 model of `runtime/batch.rs`, `runtime/vectorized.rs`,
`runtime/vector_expr.rs`, `runtime/row.rs` and
`runtime/ops/batched_result_emitter.rs`. 179 theorems, no `sorry` / `admit` /
`axiom`. `f64` is abstract (`FloatModel`); the one IEEE fact used —
`partial_cmp` antisymmetry — is a class field (hypothesis). Per-module
Lean↔Rust tables are in each file's header; per-function buckets in
`COVERAGE.tsv`; Rust repros in `graph/tests/lean_columnar.rs`
(`cargo test -p graph --test lean_columnar`).

| module | there |
| --- | --- |
| `Column` | `Column`, `classify_*`, `NullBitmap`, `extend_active_slice` (`batch.rs:90-414`) |
| `Batch` | `Batch` selection / gather / compaction / `set_column` / `write_column` (`batch.rs:805-1394`), selection producers `filter.rs`, `limit.rs`, `skip.rs`, `runtime.rs:529` |
| `Row` | `Row` (`row.rs`), `BatchRow::value_at` / `to_owned_row` (`batch.rs:1415-1447`) |
| `Concat` | `Batch::concat`, `concat_typed_column` vs `BatchBuilder` (`batch.rs:525-797`, `:979-1151`) |
| `Emitter` | `BatchedResultEmitter` (`batched_result_emitter.rs:487-783`), ExpandInto trim (`expand_into.rs:285-294`) |
| `VectorExpr` | `CmpOp`, comparison kernels, `ExprColumn`, `union_nulls`, `compare_columns` |
| `Arith` | `arithmetic`, `int_lane`, `float_lane` vs `Value` `+ - * / %` |
| `Lanes` | `negate_bools`, `eval_case`, `eval_function` folding, `eval_variable`, filter mask |

## Proved (plain English)
* gather reads row `idx[k]` (value, bound bit, origin); gather∘gather composes;
  gather never panics on physical indices and preserves WF.
* `into_compacted` yields `active_len` dense rows equal, slot by slot, to the
  active rows in order.
* Selection vectors: active indices are `< len` and strictly increasing; Filter,
  Limit, Skip and their compositions produce exactly `filter`/`take`/`drop` of the
  active rows, provided `len ≤ 65536` (the `as u16` cast); past that the cast
  wraps (`skip_u16_wraps`).
* `write_column` = scatter: active row k reads `vals[k]`, inactive rows keep their value.
* `set_column` binds exactly its slot; `classify_stored_column` and
  `classify_exact_column` are lossless; `classify_column` is lossy (latent, no caller).
* `concat` = per-row `BatchBuilder` concat (value and bound bit at every row and
  slot, origins, row count) when no input batch is empty; the typed fast path
  equals the fallback.
* Emitter: `emit_lazy` conserves the `(row, item)` stream (no loss, duplication or
  reordering), fills to `min(ceiling, remaining)`, all batches non-empty and
  `≤ BATCH_SIZE`; the ceiling is in `[1, 1024]` (source semantics) and grows back;
  every packed parent index is an active row, so `finish_batch`'s gather never panics.
* ExpandInto `set_selection(0..remaining)` suspicion REFUTED (`expandTrim_ok`):
  the output is dense with `n ≤ ceiling` rows, the trim fires only when
  `remaining < n`, indices are `< n < 65536`, `produced ≤ cap`.
* `compare_columns` = per-row comparison in every lane (int, float, int→float
  promotion, flipped constant, pairwise, generic); `CmpOp::flip` sound (ints;
  floats via IEEE antisymmetry); NaN makes orderings false and `<>` true.
* `union_nulls` exact except (Values, non-null constant), which no typed lane sees.
* Arithmetic lanes = `Value` arithmetic row by row, including the first error
  (zero divisors fall to the generic lane; `Int ⊕ Int` never takes the float lane).
* NOT, CASE (incl. laziness: THEN only on matched rows; value-form test =
  `compare_value == (Equal, None)`), scalar function folding, `eval_variable`
  (in the column space), and the Filter mask agree with the per-row evaluator.
* `Row::insert_by_id` / `take` / `merge` (overlay of bound slots only);
  `to_owned_row` agrees with `BatchRow::value_at` on every bound slot.

## Findings
Confirmed on the real code (pub API; no Cypher trigger found, so latent):
1. `Batch::concat` (`batch.rs:1008-1015`) takes `bound_anywhere` from batches with
   no active rows: a slot bound only in an empty batch comes out bound
   (`Some(Null)`) where the per-row definition leaves it unbound (`None`).
   Lean `empty_batch_binds`; test `latent_concat_empty_batch_binds_column`
   (concat: bound true / `Some(null)`; per-row: false / `None`). Fix: skip
   `active_len() == 0` batches in the fallback loop, as `concat_typed_column` does.
2. `BatchRow::to_owned_row` (`batch.rs:1432`) drops trailing Unbound columns, so
   `view.value_at(i) = Some(Null)` but `to_owned_row().value_at(i) = None`
   ("Variable not found" in `resolve_var`). Lean `toOwned_trailing_unbound`; test
   `latent_to_owned_row_drops_trailing_unbound_slot`. Same shape: `eval_variable`
   reads Null beyond the column space where the per-row view errors
   (`evalVariable_beyond`). 12 OPTIONAL/CALL/comprehension queries tried: all correct.

Hazards (not bugs today): selection `u16` casts need `len ≤ 65536`, guaranteed
only by `BATCH_SIZE` caps — exactly the guard #2922's miscompile drops (this
build inlines it correctly: `probe_capped_unwind_batch_sizes` sees 1024-row
batches; `agrees_skip_selection_large_cap` passes); no-alias emitter over a
zero-column parent with no origins emits a 0-row batch (`finishLen_drops`;
unreachable, ExpandInto always has columns; since #2845 a parent carrying origins is
gathered, `finishLen_origins_kept`); `classify_column` lossy (no caller); columnar CASE/AND may
report a different row's error message than the per-row path (both error).

## Gaps
f64 arithmetic/ordering abstract; `compare_value` beyond numerics/null abstract
(`other`); MergePlan / `push_merged` / `push_planned` / `merge_over_input`,
property fetch, `hasLabels`, operator dispatch/profiling not modelled; emitter
lane layout (transpose, shared alias) not modelled; CASE/function proofs assume
pure sub-expressions; BitSet/NullBitmap word packing is proofs/runtime_ds /
proofs/ops_aggregate.

## Wave-5 additions
* `Gather`: the `GatherItem` trait proved ONCE (`gather_spec`: for a lawful lane, finishing the
  pushed items sets column `cols b [i]` to `xs.map (val b i)`), then each of the 7 impls shown
  lawful (`giNode_lawful` … `giValue_lawful`); `RowIter`; emitter constructors.
* `BatchMisc`: the rest of batch.rs — column helpers (`compareAt_spec`: NaN compares Less both
  ways, known #2891), builder push paths and merge plan, constructors/accessors,
  `merge_over_input`, `clone_active_rows_seq_origin`, `size_hint`, `set_argument_batch`
  (every Argument leaf gets the batch, emitter-backed nodes reset), `inspect_context`, `BatchOp::next`.

## Re-target to a9377c636 (#2845)
* `BatchMisc.rowsOnly` (`Batch::rows_only`, batch.rs:853): `rowsOnly_spec` (len rows, no
  column, origin 0, WF). `BatchMisc.hasOrigins` (`Batch::has_origins`, batch.rs:1312):
  `hasOrigins_spec`, `hasOrigins_zero_not_unset` (it separates "all origins 0" from
  "never stamped", which `origin_row` cannot). `Emitter.finishLen` takes the new
  `has_origins` arm of `start_batch`. The per-row origin argument is in
  proofs/pr2845_review (`emit_origin_sound`).

## Wave 6 (`Dispatch`)
`VectorEval::eval` dispatch PROVEN: **`eval_sound`** — for every expression tree, given
each arm's lane theorem (stated as `Lawful`), any column `eval` returns equals the per-row
evaluator row by row; guards route each shape to exactly one arm (`eval_cmp`).
`CmpOp::from_expr_ir` (`fromExprIR_sem`, `flip_sem`), `eval_values` (`evalValues_sound`),
`property_values` (`propertyValues_spec`/`_none_iff`, bulk read = per-row read as the
`Store.Lawful` contract), `eval_has_labels` (`hasLabels_sound`, incl. the unregistered-label
shortcut; `hasLabels_typecheck_first` = test11's ordering), `eval_per_row`
(`perRow_spec`, `perRow_error`), `new`.
-/
