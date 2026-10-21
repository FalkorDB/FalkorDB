import Pr3437Review.ListLoop
import Pr3437Review.Chunked
import Pr3437Review.Drive
import Pr3437Review.Classifier
import Pr3437Review.Bridge
import Pr3437Review.ValueOfType
import Pr3437Review.NegFold
import Pr3437Review.ExactProp
import Pr3437Review.Discharge

/-!
# Review of PR #3437 — "perf: list comprehensions evaluate their body over the list in chunks"

PR branch `perf/list-comprehension` @ a739b990f (6efd69cd6 + a739b990f, which sends bodies with
a nested binder per element; on origin/main 9212add39); every cite is to a739b990f unless
marked *main* (or, in `ClassifierPre`, 6efd69cd6). No `sorry` / `admit` / `native_decide` / axiom;
headline theorems use only `propext`, `Quot.sound`, `Classical.choice`. Definitions are
imported from proofs/core (`V`, `F`, i64), proofs/columnar (`Dispatch.eval_sound`),
proofs/value_order (`W.valueOfType`), proofs/functions_str_list (`evalQuantifier`, `K`) and
proofs/expr_semantics (`checkedNeg`).

| module | Rust |
| --- | --- |
| `ListLoop` | main's per-element comprehension / quantifier loops (*main* eval.rs:1010-1025, :1072-1080); `LoopSink`, `ListLoop` (eval.rs:289-311); the per-element half of `list_loop_chunk` (:1494-1546) |
| `Chunked` | `list_loop_columnar` (eval.rs:1550-1652) |
| `Drive` | `list_loop_chunk` (:1456-1547), the chunk loops, `eval_list_comprehension` (:1372-1449), `ExprEval::eval_quantifier` (:1310-1363), `size` (functions/list.rs:61) |
| `Classifier` | `collect_body_vars` (eval.rs:380-402), `ListLoop::vector_plan` (:344-370) |
| `ClassifierPre` | HISTORICAL: the 6efd69cd6 classifier, which accepted nested binders |
| `Bridge` | `VectorEval` over the chunk batch vs columnar's `eval_sound`; `counts_comprehension` (eval.rs:315) and its guard in `VectorEval::eval` (vector_expr.rs:253-269); `loop_var_column` (eval.rs:405), `broadcast_column` (:417) |
| `ValueOfType` | `value_of_type` (value.rs:1315-1350) |
| `NegFold` | the unary level of `parse_expr_inner` (cypher.rs:2547-2580), `ExprIR::Negate` (eval.rs:895-907), `evaluate_param` (:2277-2286) |
| `Discharge` | `ListLoop::new` over expr_semantics' `E`; `RowAgree` for every accepted body; `BatchRow` of eval.rs:1560-1564 |
| `ExactProp` | `materialize_node_property_exact` (runtime.rs:1566-1587), `get_node_int_attributes_by_idx` (graph.rs:2602-2613), `get_int_attrs_by_idx_batch_into` (attribute_store.rs:1186-1216), `eval_property` (vector_expr.rs:337-353) |

## Verdicts

1. **Chunked = per-element** — CONDITIONAL on one premise, `hS` (columnar's per-row reference = expr_semantics' `evalNode`; see below). `Body.Drops` and `RowAgree` are discharged; no binder premise is left (a739b990f).
   `comprehension_eq_main`: for every source (streamed literal/`range`, stored list, null,
   scalar) the PR's comprehension is `Except`-equal to main's — same list, same order, or the
   same error. `quantifier_eq_main`: `any`/`all`/`none`/`single` likewise (both sides feed
   the same counts to `evalQuantifier`, so Cypher's three-valued answers are unchanged).
   Both rest on `prRun_eq`, which is stronger than the PR needs: *any* partition of the list
   into chunks, with *any* per-chunk choice of columnar or per-element path, gives main's
   loop over the concatenation (`chunksOf_flatten`, `streamLoop_flatten` cover the two
   partitions the PR uses). `columnar_sound`: a chunk the columnar path finishes gives what
   main's loop gives, through every lane (`Bools` fast paths `passed_bools`, `count_bools`,
   `tally_bools`; the all-pass `extend_from_slice` `filterPass_full`).
2. **Error behaviour** — CONDITIONAL (`hS` only). The equalities in 1 are equalities
   of `Except`, so the reported error is main's. `ref_error_first`: main's error is the
   first element whose step fails, all earlier ones having succeeded;
   `refStep_collect_error`: within an element, the predicate's error, else (only if it
   passed) the projection's. A chunk whose columnar evaluation errors is re-run per element
   from the sink as it was (`chunkStep_eq`; every `?` in `list_loop_columnar` precedes its
   sink write), so no partial result escapes.
3. **Fallback classifier** — PROVEN (a739b990f). `collect_rejects_iff`: `collect_body_vars`
   returns `false` iff the body contains a nested `ListComprehension` / `Quantifier` /
   `Reduce`, a `NestedPlan`, `Pattern`, `PatternComprehension`, `Distinct`, a writing or an
   aggregating function, at any depth; `vectorPlan_no_bad`: no plan (so no columnar path)
   for such a predicate or projection; `vectorPlan_broadcasts`: every variable the bodies
   read is the loop variable or broadcast with `env.value_at(id)` (`vectorPlan_vals`).
   History: `pre_a739b990f_accepts_nested_binder` (6efd69cd6 accepted `any(z IN [x] …)`).
4. **`size(<comprehension>)`** — CONDITIONAL (`hS` only). `size_eq_main`: counting = `size` of
   main's list for every source (null source → null; empty `range` → 0);
   `size_projection_error`: a non-identity projection's error still surfaces; `eval_sound_pr`
   / `counted_goes_per_row`: the new `!counts_comprehension` guard sends the shape to
   `eval_per_row`, and columnar's `eval_sound` still holds for the PR's `VectorEval::eval`.
5. **`value_of_type(List, List(Any))`** — PROVEN. `vtPR_eq`: the PR's `value_of_type`
   (early arm at every recursion level) equals main's for every value and type;
   `size_arg_list` for `size()`'s union argument.
6. **Negated literal folding** — PROVEN. `fold_eq_eval` / `fold_eq_evalParam`: for every
   tree the PR's unary level evaluates (`ExprEval` and `evaluate_param`) to main's — same
   value with the same float bits, or same error. `fold_never_min`: `-9223372036854775808`
   takes the pre-existing `i64::MIN` branch, so the folded (wrapping) `-i` never sees
   `i64::MIN` and debug builds cannot panic there; `paren_min_not_folded`:
   `-(-9223372036854775808)` is not folded and still raises; `negF_zero` /
   `neg_zero_literal`: `-0.0` is the constant with bits `0x8000000000000000`, as main
   computed; `bare_min_refused`.
7. **`materialize_node_property_exact`** — PROVEN. `exact_eq_main`: equal to main's
   `classify_exact_column(materialize_node_property_values(..))` in every case — on the
   read-only int/null path the same `Ints` lane and null bits (`getIntLoop_spec`), otherwise
   it is that call (`getIntLoop_other`: any other stored tag gives up); `evalProp_eq`.

## Hypotheses: what was discharged, what is left (`Discharge.lean`)

* **`Body.Drops` — DISCHARGED** (`newBody_drops`): `ListLoop::new` over expr_semantics' `E`
  with main's per-element evaluator `evalNode` on `env + {var: v}`.
* **ExprEval frame property — DISCHARGED** (`evalNode_frame`) for expr_semantics' evaluator.
* **`RowAgree` — DISCHARGED for every body `vector_plan` accepts** (`rowAgree`): since
  a739b990f the only bound variable of a columnar body is the loop variable; every other
  read is broadcast with `env.value_at(id)` and the batch columns read back the element /
  outer value (`loopVar_get`, `broadcast_get`). No binder id-uniqueness premise remains.
* **Left: `hS`** — columnar's abstract per-row reference `Dispatch.RowSem.rowEval` (the
  premise of `eval_sound`, through `Dispatch.Lawful`) equals expr_semantics' `evalNode` on the
  batch row (`colSound_accepted` takes it as an argument). Blocked, precisely:
  1. no owner instantiates `Dispatch.Lawful`: columnar proves each lane against its own
     operator-level references over `Columnar.V F` and never assembles them;
  2. expr_semantics' `E` models `AND` / `OR` / `CASE` (and `NOT`, `XOR`, function calls, `IN`,
     `IS NULL`, …) only as the eager value-only `compound`, which cannot short-circuit
     (`compound_cannot_short_circuit`; live: `RETURN false AND 1/0 = 1` → `false` on main),
     and the classifier accepts these forms;
  3. the arms both projects do model (`=`/`<>`/`<`…, `+ - * / %`, `prop`, `var`, `const`,
     `param`) are over different value types, and no owner proves expr_semantics'
     `evalCmp`/`all_equals` equal to columnar's `compareValues`/`rowCmp`, or `O.arith` to
     `valueArith`.
  Needed from the owners: a tree-level per-row evaluator shared by columnar and
  expr_semantics with faithful `AND`/`OR`/`CASE`/`NOT`/call arms (or a translation
  `Columnar.V F ↔ FalkorCore.V F` plus those arms in `E`), and a `Dispatch.Lawful` instance
  built from columnar's lane theorems.

## Proposed parser_grammar re-statements on the PR's parser

`Model.step`, level 9, gains the fold (cypher.rs:2558-2572):
`| .node .neg [lf (.int n)] => ret (lf (.int (-n))) K ts` (after the `i64::MIN` case, which
already yields `-2^63`; `Tok.int` is non-negative, so `-n` is exact), and
`| .node .neg [lf (.float f)] => ret (lf (.float (-f))) K ts` if the leaf alphabet has floats.
Then:
```
theorem neg_binds_tighter_than_pow (a b : Nat) :
    parseAll 400 [.dash, .int a, .pow, .int b] =
      .ok (.node .pow [lf (.int (-(a : Int))), lf (.int b)]) [] := by rfl
theorem sign_sequences :
    parseAll 400 [.plus, .dash, .int 1] = .ok (lf (.int (-1))) [] ∧ … (other conjuncts unchanged)
/-- `foldNeg` rewrites, bottom-up, each `node .neg [lf (.int n)]` to `lf (.int (-n))`
(only a *direct* literal child: `-(1)` keeps its `paren` node, as the parser does). -/
theorem bounded_agree :
    ∀ l ∈ lists 3, agreeB (parseAll 400 l) ((G.all 200 l).map foldNeg) = true ∨ known l = true
```
(`-x.p` in `Theorems.lean:85` is unchanged: its child is a `prop` node, not a literal.)

## Live checks (main 9212add39 vs PR 6efd69cd6 — before a739b990f, which only sends more bodies per element — release dylibs, ports 6540/6541)

~110 queries, all identical (values and error strings): comprehensions / quantifiers over
16-3000-element lists with predicate and projection errors at chosen elements
(predicate-before-projection, across the 1024 chunk boundary), mixed-type, NaN and -0.0
lists, node-property bodies (int/float/null/missing/string), pending `SET`, `SET … = NULL`,
`DELETE`, `CREATE` in the same query, nested comprehensions / quantifiers / `reduce` /
pattern comprehensions, shadowed and outer variables (`WITH … WHERE`, `ORDER BY`, `CALL {}`),
`rand()`, `size(..)` with null/scalar sources; negative literals `-1`, `-1.5`, `-0.0`, `-0`,
`-2^63`, `-(-2^63)`, `-0x10`, `-2 ^ 2`, `LIMIT -1`, `SKIP -1`, index seeks `p.v = -1` /
`-0.0` / `-2.5` / `IN [-1, -2.5]`. No counterexample found.

## Existing proofs affected (fix at re-target)

* **proofs/parser_grammar — stale model; tree-shape theorems false for the PR's parser.**
  `Model.step` at level 9 returns `res` unchanged; the PR folds `Negate(literal)`. So
  `neg_binds_tighter_than_pow` (`-a ^ b` ↦ `Pow[Neg[a], b]`; the PR builds `Pow[Const(-a), b]`),
  the first conjunct of `sign_sequences` (`+-1` ↦ `Neg[1]`; PR: `Const(-1)`) and the
  `parseAll = G.all` agreement (`bounded_agree`) on inputs containing `- <int>` describe
  main's tree, not the PR's. Semantics are unchanged (`fold_eq_eval`); the tree shape is not.
* **proofs/columnar `Dispatch.eval`** lacks the `!counts_comprehension` guard;
  `eval_sound_pr` shows soundness survives (instantiate `Arms` with `armsPR`).
* **proofs/functions_str_list / expr_semantics**: the `ListComprehension` and `Quantifier`
  arms moved into `eval_list_comprehension` / `ExprEval::eval_quantifier`; their models of
  main stay true of the PR by `comprehension_eq_main` / `quantifier_eq_main` (under the
  hypotheses above). Their `eval_quantifier` rows (line 1664) no longer resolve: the free fn
  is now at eval.rs:2109 and a method of the same name was added at :1310.
* **proofs/value_order `W.valueOfType`** lacks the early arm; `vtPR_eq` transfers every theorem.
* Cite shifts only: vector_expr.rs +3 (`eval_values`..`eval_property`), +12 (from
  `property_values` on); eval.rs +1 (lines 86-231), +166/+170 (276-535), +96 (1150-1196),
  +448 (1226 on); graph.rs +15 (from 2610); runtime.rs +3 (from 209; decl/impl rows of
  `get_variables`/`get_return_names` then resolve to the wrong line), +33 (from 1562);
  value.rs +2 (from 1357; same decl/impl effect for `get_type`); attribute_store.rs +39 (from
  1182, `structural_memory_usage` 1414 → 1453); cypher.rs +15 (from 2663).
  `COVERAGE.tsv` lists the 13 new functions and the changed ones (commented out until merge).
-/
