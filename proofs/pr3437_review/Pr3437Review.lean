import Pr3437Review.ListLoop
import Pr3437Review.Chunked
import Pr3437Review.ChunkedSound
import Pr3437Review.Drive
import Pr3437Review.Classifier
import Pr3437Review.Bridge
import Pr3437Review.ValueOfType
import Pr3437Review.NegFold
import Pr3437Review.ExactProp
import Pr3437Review.Discharge
import Pr3437Review.Final

/-!
# Review of PR #3437 — "perf: list comprehensions evaluate their body over the list in chunks"

PR branch `perf/list-comprehension` @ a739b990f (6efd69cd6 + a739b990f, which sends bodies with
a nested binder per element; on origin/main 9212add39); every cite is to a739b990f unless
marked *main* (or, in `ClassifierPre`, 6efd69cd6). No `sorry` / `admit` / `native_decide` / axiom;
headline theorems use only `propext`, `Quot.sound`, `Classical.choice`. Definitions are
imported from proofs/core (`V`, `F`, i64), proofs/columnar (`Dispatch.eval_sound`),
proofs/value_order (`W.valueOfType`), proofs/functions_str_list (`evalQuantifier`, `K`) and
proofs/expr_semantics (`checkedNeg`, `E`/`evalNode` and its frame property).

| module | Rust |
| --- | --- |
| `ListLoop` | main's per-element comprehension / quantifier loops (*main* eval.rs:1010-1025, :1072-1080); `LoopSink`, `ListLoop` (eval.rs:289-311); the per-element half of `list_loop_chunk` (:1494-1546) |
| `Chunked`, `ChunkedSound` | `list_loop_columnar` (eval.rs:1550-1652); the projection over the passed row indices (`passedIdx`) |
| `Drive` | `list_loop_chunk` (:1456-1547), the chunk loops, `eval_list_comprehension` (:1372-1449), `ExprEval::eval_quantifier` (:1310-1363), `size` (functions/list.rs:61) |
| `Classifier` | `collect_body_vars` (eval.rs:380-402), `ListLoop::vector_plan` (:344-370) |
| `ClassifierPre` | HISTORICAL: the 6efd69cd6 classifier, which accepted nested binders |
| `Bridge` | `VectorEval` over the chunk batch vs columnar's `eval_sound`; `counts_comprehension` (eval.rs:315) and its guard in `VectorEval::eval` (vector_expr.rs:253-269); `loop_var_column` (eval.rs:405), `broadcast_column` (:417) |
| `ValueOfType` | `value_of_type` (value.rs:1315-1350) |
| `NegFold` | the unary level of `parse_expr_inner` (cypher.rs:2547-2580), `ExprIR::Negate` (eval.rs:895-907), `evaluate_param` (:2277-2286) |
| `Discharge` | `ListLoop::new` over expr_semantics' `E` (`newBody`, `newBody_drops`); the classifier's view of `E` (`toBT`) |
| `ChunkBatch` | the chunk batch of `list_loop_columnar` (eval.rs:1560-1564) over columnar's `Batch`: well formed, rows `0..len`, the batch row reads the element / outer values |
| `PRArms` | the PR's `VectorEval` (vector_expr.rs:211-283, 336-353 at a739b990f) as columnar arms: the exact `NodeIds` fetch is main's column (`evalPropertyPR_eq`), `counts_comprehension` routes per row; `lawfulPR` |
| `Concrete` | what `list_loop_columnar` calls (`colOf`) and **`colOf_sound`** (`Col.Sound`, the former premise `hS`); `RowAgree` on the real batch (`rowAgree`) |
| `Runtime` | the attribute state behind the exact fetch (`nodeStore_matValues`, `withStore_rtOf`) |
| `Final` | the headline theorems with nothing but `hreg` assumed |
| `ExactProp` | `materialize_node_property_exact` (runtime.rs:1566-1587), `get_node_int_attributes_by_idx` (graph.rs:2602-2613), `get_int_attrs_by_idx_batch_into` (attribute_store.rs:1186-1216), `eval_property` (vector_expr.rs:337-353) |

## Verdicts

1. **Chunked = per-element** — **PROVEN** (2026-10-21; was CONDITIONAL on `hS`).
   `Final.pr_comprehension_eq_main`: for every source (streamed literal/`range`, stored list,
   null, scalar) the PR's comprehension, with the PR's real `VectorEval` over the real chunk
   batch, is `Except`-equal to main's — same list, same order, or the same error;
   `Final.pr_quantifier_eq_main`: `any`/`all`/`none`/`single` likewise;
   `Final.pr_chunkStep_eq` (`list_loop_chunk`), `Final.pr_columnar_sound` (`list_loop_columnar`).
   They rest on `prRun_eq` (any partition into chunks, any per-chunk choice of path) and
   `Concrete.colOf_sound`, so `hS` is gone: the vector evaluator's column is, row for row, main's
   per-element `ExprEval::eval_node` (columnar `CoreBridge.lawful` + `VecEval.eval_sound` for the
   PR's arms, `PRArms.lawfulPR`), and the batch row agrees with the owned row on every variable an
   accepted body reads (`Concrete.rowAgree`). The `_of` forms take an abstract runtime and
   `PropReads`; the headline forms take the runtime built from its attribute state, for which
   `PropReads` is proved (columnar `RuntimeReads.propReads`).
2. **Error behaviour** — **PROVEN**. The equalities in 1 are equalities of `Except`, so the
   reported error is main's. `ref_error_first`, `refStep_collect_error` as before; a chunk whose
   columnar evaluation errors is re-run per element from the sink as it was (`chunkStep_eq`).
3. **Fallback classifier** — PROVEN (a739b990f). `collect_rejects_iff`, `vectorPlan_no_bad`,
   `vectorPlan_broadcasts`, `vectorPlan_vals`; history `pre_a739b990f_accepts_nested_binder`.
4. **`size(<comprehension>)`** — **PROVEN**. `Final.pr_size_eq_main`; `size_projection_error`;
   `PRArms.counted_goes_per_row` (the `!counts_comprehension` guard on the PR's real arms).
5. **`value_of_type(List, List(Any))`** — PROVEN. `vtPR_eq`, `size_arg_list`.
6. **Negated literal folding** — PROVEN. `fold_eq_eval`, `fold_eq_evalParam`, `fold_never_min`,
   `paren_min_not_folded`, `negF_zero`/`neg_zero_literal`, `bare_min_refused`.
7. **`materialize_node_property_exact`** — PROVEN. `exact_eq_main`, `evalProp_eq`; and the PR's
   `eval_property` column is main's (`PRArms.evalPropertyPR_eq`, through `toECx_eq`: the two
   models of `classify_exact_column` agree).

## What is assumed, and how `hS` was discharged

* **`hS` — DISCHARGED** (`Concrete.colOf_sound`, `Final.prCol_sound`). The three gaps the
  previous review listed were closed in the owners: (1) columnar's `Dispatch.Lawful` now has an
  instance — `VecEval` models `VectorEval::eval` over expr_semantics' `E` and `CoreBridge.lawful`
  proves every field from the lane theorems (`evalAndOr_sound`, `evalCase_sound`,
  `evalFunction_sound`, `hasLabelsLane_sound`, `negate_agree`, `arithmetic_agree`,
  `compareColumns_agree`, `evalVariable_agree`, `classifyExact_exact`); (2) expr_semantics' `E` has
  faithful AND/OR/XOR/NOT/CASE/call/IN/list/subscript/projection arms with main's short-circuit and
  error order (`and_false_short_circuits`, `case_hit_lazy`, `list_error_rightmost`, …; the old
  eager catch-all is kept as `pre20261021_compound_cannot_short_circuit`); (3) columnar runs over
  `FalkorCore.V F` and `CoreOps` proves the shared forms equal (`allEq_rowCmp`, `allNeq_rowCmp`,
  `cmpResult_rowCmp`, `caseHit_caseMatch`, `notRow_eq`; arithmetic is `valueArith` on both sides).
* **`Body.Drops`** — DISCHARGED (`newBody_drops`). **Frame property** — expr_semantics
  `evalNode_frame` (every arm). **`RowAgree`** — `Concrete.rowAgree` on `ChunkBatch.chunkBatch`.
* **Bulk property reads** — DISCHARGED for the runtime built from its attribute state (columnar
  `RuntimeReads.matNode_eq` / `matRel_eq`; the PR's exact fetch reads the same state,
  `Runtime.nodeStore_matValues`).
* **Left, stated as a premise**: `hreg` (`Final.Registry`) — every `FuncInvocation` named
  `hasLabels` in the bodies holds the registered `hasLabels` (`HasLabelsFn.hasLabelsFn`,
  functions/entity.rs:62). It is the shape of the trees the parser and binder build
  (`get_functions().get("hasLabels")`, registry names are unique), needed because
  `VectorEval::eval` dispatches on the function's *name*.
* **Modelling choices** (not premises): function calls are pure functions of their arguments
  (`rand()`, `randomUUID()` are compared in distribution only; both paths make the same calls
  for zero-argument functions); `agg_group_key` is not modelled (it only affects aggregate calls
  and `DISTINCT`, both rejected by `collect_body_vars`); attribute-id resolution is the
  memoized lookup's value (proofs/runtime_core `attrIdM_correct`).

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
* **proofs/columnar `VecEval.eval` / `CoreBridge.armsOf`** model main's `VectorEval`; the PR's
  arms are `PRArms.armsPR` (the `!counts_comprehension` guard and the exact `NodeIds` fetch) and
  `PRArms.lawfulPR` shows soundness survives; at merge, fold them into columnar. (The coarser
  `Dispatch` model: `Bridge.eval_sound_pr`.)
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
