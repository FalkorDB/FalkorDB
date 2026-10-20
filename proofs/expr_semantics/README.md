# expr_semantics: what scalar expressions evaluate to, per row and per column

This project models how the engine evaluates scalar expressions. It proves two things:

- the columnar evaluator (`VectorEval`), which Filter, Project and Aggregate actually run, gives the same answer as the per-row evaluator (`ExprEval::eval_node`);
- the per-row evaluator follows openCypher's three-valued logic.

It also pins down, with counterexamples, the places where the per-row evaluator departs from openCypher or from C FalkorDB.

User-facing behaviour that depends on it:

- `WHERE` predicates: comparisons, `AND`, `OR`, `NOT`, `XOR`, `IN`;
- arithmetic in `RETURN`, `WITH` and `SET`, including integer overflow;
- `CASE` expressions;
- `^`, `sign()`, `toInteger()`, `coalesce()`;
- `UNWIND range(…)`;
- `shortestPath`, whose bidirectional BFS lives in `runtime/eval.rs`.

The Rust is mostly in `graph/src/runtime/eval.rs`, `vector_expr.rs`, `vectorized.rs` and `value.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorExprSemantics.lean`](FalkorExprSemantics.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does the columnar path give the same answer as the per-row path?**

Yes, for every lane this project models:

- **comparison kernels:** the `Int` kernel, the `f64` kernel, mixed `Int`/`Float` promotion, the flipped-constant form (`5 < n.x` run as `n.x > 5`) and the generic lane (`vec_cmp_int`, `vec_cmp_float`, `vec_cmp_int_flip`, `vec_cmp_float_flip`, `vec_cmp_generic`);
- **whole-column comparison:** `compare_columns`, over all its arms (`compareColumns_agrees`);
- **arithmetic:** the same values row for row, and an error exactly when, and with exactly the message, the per-row evaluation errors (`arith_agrees`);
- **`AND`/`OR` with row narrowing:** the per-row short-circuit fold on every row (`andOr_agrees`);
- **`CASE`:** each `WHEN`, `THEN` and `ELSE` runs for exactly the rows the per-row path runs it for (`case_agrees`, `case_match_agrees`). So `CASE WHEN n.d <> 0 THEN 1 / n.d ELSE 0 END` can't divide by zero on either path.

Agreement is on success versus error, not on the error *message*. The columnar path reports the first error child by child, while the per-row path reports it row by row. Since #3294 both paths word an `AND`/`OR` type error the same way (`boolMismatch`), but they may report it for different rows.

**2. Is the logic openCypher's three-valued logic?**

Yes:

- `AND`, `OR` and `XOR` are Kleene logic (`and_kleene`, `or_kleene`, `xor_kleene`);
- De Morgan holds (`deMorgan_and`, `deMorgan_or`);
- `x IN list` is Kleene "any" of `x = element` for scalars (`contains_is_kleene_any`), so `1 IN [2, null]` is `null` and not `false`;
- `<>` is exactly `NOT (=)` (`neq_eq_not_eq`), since #2917;
- a non-Boolean, non-null `AND`/`OR`/`XOR` operand fails with C's message naming its type, `Type mismatch: expected Boolean or Null but was Integer` (`and_or_mismatch_names_type`, `xor_mismatch_names_type`), since #3294.

**3. What does integer arithmetic do at the edges?**

`i64` arithmetic wraps, as in a Rust release build. It equals mathematical arithmetic whenever the result is in range (`add_exact`, `div_exact`). At the edges:

- `i64::MIN / -1` wraps to `MIN` (`min_div_neg_one`);
- `i64::MIN % -1` is `0` (`min_rem_neg_one`);
- `i64::MAX + 1` wraps silently (`max_add_one`), as C does;
- unary minus is *checked* but binary minus *wraps*: `-x` errors on `i64::MIN`, while `0 - x` returns `i64::MIN` (`neg_vs_sub_zero`).

**4. Can `shortestPath` panic while it reconstructs the path?**

No. `bfs_sound` proves that whenever the bidirectional BFS meets, the `expect("BFS parent chain broken")` calls never fire. The reconstructed node sequence starts at the source, ends at the destination, and every step follows an edge the search used. That the path is the *shortest* one is argued in the source comment, not proven here.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `compareColumns_agrees` | `compare_columns` equals the per-row comparison, whichever lane it takes | `FalkorExprSemantics.lean` |
| `vec_cmp_generic`, `vec_cmp_int`, `vec_cmp_float` (+ `_flip`) | Each comparison kernel answers what `eval_node` answers | `FalkorExprSemantics.lean` |
| `arith_agrees` | Columnar `+ - * / %` equals `Value` arithmetic, values and errors | `FalkorExprSemantics.lean` |
| `andOr_agrees` | Columnar `AND`/`OR` with row narrowing is the per-row short-circuit fold | `FalkorExprSemantics.lean` |
| `case_agrees`, `case_match_agrees` | Columnar `CASE` evaluates the same arms on the same rows | `FalkorExprSemantics.lean` |
| `neq_eq_not_eq` | `<>` is `NOT (=)` for every pair of values (#2917) | `FalkorExprSemantics.lean` |
| `contains_is_kleene_any`, `and_kleene`, `or_kleene`, `xor_kleene` | `IN`, `AND`, `OR`, `XOR` are three-valued logic | `FalkorExprSemantics.lean` |
| `add_exact`, `neg_vs_sub_zero` | Wrapping arithmetic is exact in range; `-x` and `0 - x` differ only at `i64::MIN` | `FalkorExprSemantics.lean` |
| `sign_float_is_int` | `sign()` of a Float is an Integer in {-1, 0, 1} (#2907) | `FalkorExprSemantics.lean` |
| `rangeIter_correct`, `evalIter_range_is_iter` | For every `i64` start/end and nonzero step, the lazy `UNWIND range()` iterator yields exactly the eager `range()` list and never overflows, in debug and release (#2899) | `FalkorExprSemantics.lean` |
| `bfs_sound` | `shortestPath`'s BFS never hits its parent-chain panic and returns a real walk from source to destination | `BFS.lean` |
| `evalOperand_eq`, `evalBatch_eq_perRow` | The inline leaf path and the bulk property path equal the full per-row evaluator | `Eval.lean`, `EvalMore.lean` |

### How `andOr_agrees` is proved

1. **Model the values.** `V F` is the scalar `Value`: Null, Bool, Int, Float (with abstract `f64`) and String. Anything else is out of scope here.
2. **Model the per-row path.** `scalarAndOr` is the `And`/`Or` arm of `eval_compound`: fold the children left to right, stop at the first child that decides the result, and return the first error.
3. **Model the columnar path.** `vecAndOr` is `VectorEval::eval_and_or`. It evaluates each child over the rows still *live* (not yet decided) and narrows the live set after each child. Rust keeps a `result`/`nulls`/`live` triple, which the model keeps per row as `RowSt`, advanced by `stepRow`.
4. **Swap the loops.** The per-row path loops rows then children; the columnar path loops children then rows. `fold_omap_comm` proves the two orders give the same per-row state whenever each step only looks at its own row. `rowFold_scalar` then shows that one row's state at the end is `scalarAndOr` on that row.
5. **The one hypothesis.** A child's columnar evaluation over the live rows is assumed to equal its per-row evaluation mapped over them. That is the induction hypothesis for the enclosing expression tree, and it is literally what `eval_per_row` does. `case_agrees` follows the same pattern with `WHEN`/`THEN`/`ELSE` phases.

`f64` is abstract (`FloatModel`), so these proofs hold for *any* `partial_cmp` and any float arithmetic: both paths call the same primitive. The one assumption is that Rust's `f64` `<`, `<=`, `==`, `!=`, `>`, `>=` read `partial_cmp` the IEEE way, which `PartialOrd for f64` documents.

## Bugs found

Each bug was reproduced against the real engine (see [`../FINDINGS.md`](../FINDINGS.md), wave 1).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `pre2904_pow_drops_error`, `pre2904_pow_constant_true` | `^` dropped an operand's error: `2 ^ (1/0)` returned `2`, unpowered. In the optimizer's constant evaluator every variable is an error, so `WHERE x ^ true` folded to `true` and the Filter was deleted: every row came back although `x ^ true` was `null` | Fixed by #2904 (`fd2c76a6c`, issue #2902): `pow_propagates_error`, `pow_constant_not_true`, `pow_bool_type_error` (`x ^ true` is now C's type error), `pow_binary_eq_spec`, `pow_agrees_without_errors`. The optimizer side is proven sound in [`../optimizer_rewrites`](../optimizer_rewrites) (`ConstEval.constEval_sound`) |
| 2 | `pre2899_rangeIter_wraps`, `pre2899_rangeIter_panics`, `pre2899_rangeIter_panics_down` | `UNWIND range(i64::MAX - 1, i64::MAX)` yielded `MAX-1, MAX, MIN, MIN+1, …` and never stopped in release, and panicked on overflow in debug. The eager `range()` gives the two elements | Fixed by #2899 (`e63165cb5`, issue #2894): `rangeIter_correct` (all inputs), `rangeIter_max_fixed`, `rangeIter_bounds_examples` |
| 3 | `pre2907_sign_float_is_float` | `sign(-2.5)` returned a Float | Fixed by #2907 (`sign_float_is_int`) |
| 4 | `int_float_eq_not_transitive` | Whenever two distinct integers round to the same `f64`, `=` stops being transitive: `2^53+1 = 2^53.0` and `2^53 = 2^53.0`, yet `2^53+1 <> 2^53` | Harmless: C promotes the same way (Neo4j compares exactly). `ORDER BY` no longer depends on it since #2912 (see `value_order`) |
| 5 | `bug2724_prev_missing` (`FnRegistry.lean`) | `prev()` is a documented, user-callable C function but is not registered in Rust ("Unknown function 'prev'"). Property `CoversC` (every non-internal C built-in resolves); `covers_all_but_prev` shows it is the only gap | **Open** (#2724) |
| — | `toInteger_float_vs_string` | `toInteger(1e30)` saturates to `i64::MAX`, while `toInteger('1e30')` returns `null` | Noted difference, not filed |

**CONDITIONAL rows.** Four rows are covered but not proven outright, because each rests on a hypothesis about other Rust code:

- `NeighborIter::new` and `new_reversed` assume the graph laws: the adjacency matrix is the union of the relation matrices, and `matrix_t` is the transpose.
- `eval_batch` and `try_eval_batch_property` assume `BatchLaws`: the bulk `materialize_node_property_values` equals `get_node_attribute` per id. That is `runtime_core`'s `materialize_eq`, which is itself CONDITIONAL.

## Files

| File | What's in it |
|---|---|
| `FalkorExprSemantics.lean` | The report header, i64 helpers, the scalar `V F` and `compare_value`, the per-row evaluator, every columnar lane and its agreement theorem, three-valued logic, `IN`, `^` (and its pre-#2904 history), `RangeIter` (and its pre-#2899 history), `sign`, `toInteger`, `coalesce`; bugs 1–4 |
| `Eval.lean` | `ExprEval` entry points and fast paths: `eval`, `eval_operand`, variable resolution, constants and parameters |
| `EvalMore.lean` | Batch evaluation (`eval_batch`), `classify_join_keys`, UNWIND iteration, compiled regex, map projection |
| `BFS.lean` | `NeighborIter` and the bidirectional BFS behind `shortestPath`; `bfs_sound` |
| `FnRegistry.lean` | #2724: the registered function names vs. C's user-callable built-ins |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Shared foundations (`wrap`, `wAdd`/`wSub`/`wMul`/`wDiv`/`wRem`, `InI64`, `I64MIN`/`I64MAX`, `wrap_id`) come from [`../core`](../core) through a Lake path dependency.

## Coverage

Line numbers are pinned to main 737dfb5da (#2899 rewrites `RangeIter` and adds `RangeIter::size_hint`, shifting every later `eval.rs` row; earlier, #2904 changed the `Pow` arm of `eval_compound` and `apply_pow`; #2951 only changes `eval_compound`'s aggregate-finalizer arm, modelled in [`../ops_aggregate`](../ops_aggregate)).

`COVERAGE.tsv` has 94 rows: 31 in `eval.rs`, 26 in `vector_expr.rs`, 17 in `functions/math.rs`, 10 in `value.rs`, 7 in `vectorized.rs`, and one each in `conversion.rs`, `eliminate_true_filters.rs` and `functions/mod.rs` (`init_functions`, MODELLED: the registered name set, #2724).

| Bucket | Rows |
|---|---|
| PROVEN | 45 |
| CONDITIONAL | 4 |
| MODELLED | 16 |
| NOT COVERED | 29 |

- **MODELLED:** helpers the proofs use but don't state a theorem about, for example `compare_value` (scalar subset), `compare_floats`, `ExprColumn::get`, `range` (used as the spec of `rangeIter_correct`) and `eval_per_row` (taken as the induction hypothesis). `eval_compound` is MODELLED because only its `And`/`Or`/`Xor`/`Negate`/`Pow`/`In`/comparison arms are modelled; the list, map and comprehension arms are not.
- **NOT COVERED:** the `VectorEval` dispatcher and its column accessors (`eval`, `eval_variable`, `property_values`, `eval_has_labels`, `negate_bools`, `CmpOp::from_expr_ir`), which are proven in `columnar`; the math functions `abs`, `ceil`, `floor`, `round`, `sqrt`, `log`, `exp` and the nondeterministic `rand` and `randomUUID`, several of which are proven in `value_math`; and the list functions `eval_quantifier` and `get_elements`, since lists are out of scope here.

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Value types.** Lists, maps, temporal values, points and vectors are out of scope; they appear as `V.other` or not at all.
- **Floats.** `f64` is abstract. Nothing here depends on IEEE rounding.
- **Null bitmaps.** A typed column's null bitmap is an `Option` per row. Rust keeps a placeholder (0 or 0.0) in null rows and masks it afterwards. The wrapping ops can't panic on a placeholder, and the division gate skips null rows, so the two behave the same.
- **Error text.** Agreement is on success versus error, not on which row's message is reported.
- **BFS minimality.** `bfs_sound` proves that the path is a real walk, not that it is a shortest one.
- **Matrix reads.** Matrix rows in `BFS.lean` are abstract `Nat → List Nat`; the GraphBLAS reads are the FFI boundary.

## Build and check

```bash
cd proofs/expr_semantics && lake build    # this project
bash proofs/lean_ci.sh expr_semantics     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                   # repo-wide coverage report
```

**When you change `eval.rs`, `vector_expr.rs`, `vectorized.rs` or the `Value` operators in `value.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If an agreement theorem stops holding, the columnar and per-row paths now disagree on some input, and the failing goal shows which one. When a PR fixes a bug proven here, turn the counterexample into a correctness theorem and keep the old one under a `pre<PR>_` name (as `pre2904_pow_drops_error` and `pre2899_rangeIter_wraps` were kept).
