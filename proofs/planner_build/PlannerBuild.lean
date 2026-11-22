/-
# Plan construction vs openCypher clause-sequence semantics

A model of `graph/src/planner/mod.rs` plan construction (`Planner::plan`,
`plan_query` stitching, `plan_project`, the clause arms) and of the IR's
reference semantics, with machine-checked agreement against an openCypher
driving-table semantics (each clause maps a bag of records, plus the graph
state, to a bag of records). `lake build` succeeds; no `sorry`/`admit`/`axiom`.
576 theorems. Rust source: origin/main `256ab4d3f` (cites checked against it). `ExprIR::and_of`/`or_of` flattening is
modelled in `AndOf.lean` (`tv_andOf`/`tv_orOf`); there is no `Paren` past the binder (`expr_to_plan` does not
unwrap one, `PatternMode::descend` does not treat it as a connective); the clauses after a procedure CALL are
validated (`Ast.call_validates_rest`); the parser rejects a relationship variable repeated in a named-path CREATE
or a MERGE (`create_named_path_dup_rejected`), as C does. Rust repros: `cargo test -p graph --test lean_planner_build`
(`bug_*` tests assert the C/openCypher answer and FAIL today).

## Lean ↔ Rust

| here | there |
| --- | --- |
| `Op`, `Plan` | `enum IR` mod.rs, `DynTree<IR>` |
| `ev`, `evUnion`, `evRights`, `evAny` | `Runtime::build_batch_op` / `children_to_recurse` runtime/runtime.rs (child-0 input, `pop_or_once`, CP right branches materialised on the argument row) |
| `forRows` | openCypher "for each record" |
| `walkFirst` / `walkLoop` | insertion-point walks mod.rs / 3085-3138 |
| `projDescend`, `applyChain` | mod.rs |
| `descendOne`, `descendClause`, `satChain`, `saturated` | mod.rs (`descend_clause_expr_applies`, `descend_one_clause_expr_chain`), 1133-1138 (`is_saturated_apply`) |
| `insertStep`, `stitchLoop`, `rustStitch` | mod.rs (`push_sibling_tree(Left)` / `push_child_tree`, Apply-wrap of a CartesianProduct) |
| `needsApplyWrapping`, `addArgs` | mod.rs |
| `planProject`, `ProjC` | `plan_project` mod.rs |
| `planMerge` | MERGE arm mod.rs |
| `planProc` | CALL procedure arm mod.rs |
| `planUnion` | UNION arm mod.rs |
| `.apply [.aggregate _ [b]]`, `.apply [b]` | CALL {} arm mod.rs |
| `.forEach e x [b]` | FOREACH arm mod.rs |
| `.optional vs [m]`, `.apply [.optional vs [m]]`, `.apply [m]` | MATCH arm mod.rs |
| `isPlannerScanSubtree` | optimizer/select_scan_node.rs |
| `E.rsRemove`, `E.specRemove`, `E.project`, `E.VerifiedSound` (RemoveVerified.lean) | REMOVE arm mod.rs, `carried_verified` in `plan_project` mod.rs |
| `E.Ex`, `E.D`, `E.QG`, `E.V` (Expr.lean) | `DynTree<ExprIR<Variable>>`, `QueryGraph`, `(id, scope_id)` |
| `E.containsInlineVar`, `hasPatternExpr`, `patternExprScope`, `Mode`, `needsExtraction`, `inlineAttrsToFilter`, `hasLabelsFilter`, `patternExprVariables` | `contains_inline_var` :1108, `has_pattern_expr` :1119, `pattern_expr_scope` :1528, `PatternMode` :953, `needs_extraction` :1552, `inline_attrs_to_filter` :545, `has_labels_filter` :876, `pattern_expr_variables` :1435 |
| `E.tv` (Decomp.lean) | openCypher three-valued WHERE (Kleene AND/OR/NOT; pattern = has a match) |
| `E.collect`, `E.mint` | `collect_patterns_and_rebuild` mod.rs |
| `E.FP`, `E.run`, `E.toPlan`, `orClass`, `andFold` (ToPlan.lean) | Filter/SemiApply/AntiSemiApply/OrApplyMultiplexer runtime; `expr_to_plan` / `or_expr_to_plan` / `and_expr_to_plan` mod.rs |
| `E.planFilter` (FilterPlan.lean) | `plan_filter` mod.rs |
| `E.PSt`, `E.fresh`, `E.EC`, `E.build`, `E.hoist`, `E.extract` (Nested.lean) | Planner state, `Planner::fresh_var` mod.rs, `ExtractedComprehension`, `build_pattern_comprehension_plan` :1589, `hoist_or_nest` :1388, `extract_pattern_comprehensions` :1252 |
| `E.chainOf`, `extractClause`, `extractFilter` (Nested2.lean) | `extract_clause_expr_comprehensions` :1491 / `extract_list_expr_comprehensions` :1474 / `extract_filter_comprehensions` :1547 |
| `E.QN`, `QR`, `Comp`, `MSt`, `MP`, `firstRel`, `chainRel`, `nodeOnly`, `planComp`, `planMatch` (PlanMatch*.lean) | `plan_match` mod.rs, `mark_labels_verified`, `unverified_labels`, `build_pattern_sub_plan` |
| `HopOrder.selfLoopsAfterBinders`, `step`, `flush`, `deferP`; `pre3102_reference` (HopOrder.lean); `E.hopShape`, `E.orderRels` | `self_loops_after_binders` hop_order.rs; `tests::reference` :84; the call mod.rs |
| `E.varsOf`, `scopedVars`, `singleVarConj`, `endpointScanScore`, `walkScansTo`, `WI` (Anchor.lean); `leadingWalk`, `leadOrder`, `Outward`, `WI.forClause` (PlanMatch2.lean) | `collect_expr_scoped_variables` optimizer/mod.rs, `collect_single_variable_conjuncts` :120 / `walk` :123, `endpoint_scan_score` :160, `walk_scans_to_endpoint` :171; the leading walk and outward reorder mod.rs |
| `E.lowerInline`, `stripNode`, `stripRel`, `stripEndpoints`, `MP.stripped` (PlanMatch.lean) | `lower_inline_attrs` mod.rs, `strip_node_attrs` :803, `strip_rel_attrs` :824, `strip_endpoint_attrs` :853 |
| `E.patVars`, `readsPending`, `keepAttrs`, `liftItem`, `liftPattern`, `attrsReadVisited` (Lift.lean) | `lift_cross_entity_attrs` mod.rs, `split_inline_attrs` :581-619, `pattern_attrs_read_visited` :622-634 |
| `branchWalk`, `joinsComponents`, `joinAsCartesianBranch`, `wrapDisjoint` (Stitch.lean) | `join_as_cartesian_branch` mod.rs, `joins_components` :3212-3225, `disjoint_match` in `plan_query` :3028-3047 |
| `subtreeContains`, `stitchBelow`, `ensureInput`, `setIP`, `isRedundantOptional` (Misc.lean) | `subtree_contains` :336, `stitch_below_apply_chain` :1197, `ensure_apply_has_input` :3270, `set_include_pending_on_scans` :1084, `is_redundant_optional_match` :2971 |
| `Fmt.*` | EXPLAIN text mod.rs |
| `Ast.*`, `Ast2.*` | parser/ast.rs: `Alias`, `Variable` impls (`fmt` :173, `eq` :185, `hash` :196, `key` :214, `as_str` :236), `QueryGraph` builders, `filter_visited`, `connected_components`/`dfs`, `validate`/`inner_validate`, `return_column_names`, the `Display` impls |
| `E.renameProj`, `newPSt` (Rename.lean) | `rename_projection_outputs` mod.rs, `Planner::new` mod.rs |
| `E.MP.pathVars`, `E.BindsAllPaths`, `E.compVars`, `E.planMatchArm`, `E.MP.toPlan`, `Plan.argIsLeaf` (PathBind.lean) | paths a `plan_match` tree binds (PathBuilder mod.rs, elision :2621-2657); `QueryGraph::variables` ast.rs; MATCH arm mod.rs; Argument builds no children runtime.rs, emits one row :1285 |

## Proven (plain English)

* `self_loops_after_binders` (HopOrder.lean): `selfLoops_perm` (permutation), `kept_sublist` (non-deferred
  hops keep their order), `deferred_after_binder` (a deferred self-loop comes after its first later binder),
  `single_pass` (one fold, one step per hop), `early_return_sound`, **`eq_reference`** (= the earlier loop
  `pre3102_reference` whenever `self_loop ⇒ from = to`; `wf_needed` shows the premise matters; in the planner
  `orderRels_eq_reference` needs alias ids unique per pattern), `pr_cases`.
* `stitchLoop_eq_nest`, `rustStitch_eq_nestStitch`: the Rust stitching loop builds exactly the nested
  fill `pₙ[pₙ₋₁[…p₁]]`, each plan put at the next plan's own insertion point (tree-surgery locality:
  `modify_get_append`, `insertStep_embed`, `stitchLoop_embed`, walks land on real nodes `*_valid`).
* `query_correct`: if each clause plan implements its clause at its slot, the stitched plan evaluates
  to the openCypher composition `Fₙ ∘ … ∘ F₂ ∘ ⟦c₁⟧`.
* Per clause, at either walk's slot, for any input plan: `unwind_correct`, `create_correct`,
  `set_correct`, `remove_correct`, `delete_correct`, `proc_correct` (CALL … YIELD … WHERE),
  `project_correct` (WITH/RETURN: projection or aggregation, DISTINCT, ORDER BY, SKIP, LIMIT, WITH's
  WHERE, Commit; given DISTINCT∘aggregate = aggregate), `foreach_correct`, `call_correct`,
  `unit_call_correct` (keyless Aggregate yields one row), `optional_correct`,
  `optional_apply_correct`, `match_bound_correct`, `union_correct` (UNION / UNION ALL),
  `merge_loop_correct` (MERGE with or without named path, not last), `merge_first_correct`
  (MERGE without path as last clause), `create_loop_correct` (CREATE with or without named path, not
  last; the path is bound), `create_first_correct` (CREATE without path as last clause).
* `cp_stitch_sound`: prepending a MATCH plan to a CartesianProduct is right when no other component
  reads its variables.
* WHERE decomposition (Decomp/ToPlan/Filter/FilterPlan): `collect_value`, `collect_passes` — rebuilding a
  predicate with pattern placeholders is exact in three-valued logic; `collect_inv`/`good_final`/`fresh_final`
  — the inline ids (keyed by id only) are distinct and fresh when the patterns share one scope;
  `toPlan_correct` — SemiApply/AntiSemiApply/OrApplyMultiplexer/Filter keep exactly the rows where the
  predicate is true, given two-valued atoms; `planFilter_scalar_correct` (3VL) / `planFilter_correct`.
* Comprehensions (Nested/Nested2): `extract_frame` (loop/pattern vars, visited restored), `extract_fresh`
  (every minted variable distinct and new), `extract_clean` / `extract_noPat` (nothing left to extract),
  `hoist_spec`/`hoist_nested`/`hoisted_loop_invariant` (hoisting is sound exactly when no loop variable is
  read), `build_shape`, `chainOf_innermost` (clause Apply chain leaves a single-child slot),
  `extractFilter_clean` (feeds `plan_filter` its precondition).
* `plan_match` (PlanMatch*): operator tables `firstRel_core`/`chainRel_core`, `sortRels_spec`/`sortRels_tiers`
  (fixed < var-length < shortest, stable permutation), the leading walk `leadingWalk_spec`,
  `sortRels_lead_first`, `leadOrder_spec`/`outwardGo_outward`/`planOrder_lead` (lead first, then outward from the bound
  set), `orderRels_perm`, `ex2558_walk_leads`, `planComp_binds`, bound-label filters
  `labelStep_spec`/`labelStep_verifies`, `nodeOnly_*` (synthetic self-loop id `u32::MAX - id`), path elision
  `elide_*`, joining `planMatch_join`/`planMatch_cp`, `buildPatternSubPlan_restores`.
* Inline maps lowered once (PlanMatch*): `lowerInline_once` (one Filter per (alias, map) —
  no more `And(p, p)`), `lowerInline_two_maps`; `firstRel_lowers_from`/`_to`, `chainRel_lowers_from`/`_to`
  (both endpoints of every hop lowered to Filters, bound or not, for every operator kind);
  `firstRel_edge_pred`/`chainRel_edge_pred` (a fixed hop's edge map is a Filter right above an
  operator with `emit_relationship` set); `walk_keeps_edge_attrs`; `planMatch_stripped` (no scan or
  fixed-length operator carries an inline map — the precondition of proofs/optimizer_rewrites
  `refsNew_complete`); live: `MATCH (a:N) WITH a MATCH
  (a {x: 1})-[:R*1..2]->(b)` gives 2, 3 as C. Match: `rebuild_keeps_where`.
* Helpers (Misc/Fmt/Rename): `subtreeContains_iff`, `addArgs_leaves`, `stitchBelow_get`/`_ev`,
  `ensureInput_ev_local`/`_saturates`, `setIP_scansLeaf`, `redundant_optional_identity`, `renameProj_*`,
  `inlineAttrs_passes`, EXPLAIN strings.
* Counterexamples backing the bugs: `merge_path_last_misplaced`, `merge_path_last_skips_merge`,
  `create_path_last_misplaced`, `create_path_last_skips_create`,
  `selfloop_ignores_chain`, `cp_stitch_shape` + `cp_stitch_loses_correlation` (by `decide`),
  `where_scan_indistinguishable`.

## CONFIRMED bugs (Rust release vs C `bin/macos-arm64v8-release/falkordb.so`, live servers; plus Rust tests)
Graph: `CREATE (a:A {v:1})-[:R {w:2}]->(b:B {v:0}), (:B {v:5}), (a)-[:R {w:3}]->(b), (b)-[:S {w:1}]->(:C {v:7})`.
Re-checked live on 2026-11-22 (Rust release build of origin/main, Rust vs C): 1, 5, 8 and #2311/#2743/#2744/#2727/#2725 still reproduce.
1. **MERGE with a named path as the last clause** (of a query, a `CALL {}` body or a FOREACH body)
   fails: the first walk mod.rs (and FOREACH's body walk mod.rs) does not step over
   `PathBuilder`, so the previous clause becomes `PathBuilder(prev, Merge(..))` and `Merge` never runs.
   `MATCH (a:A) MERGE p=(a)-[:T]->(x:X)` → Rust `Variable ? not found` (unnamed variables print `?<id>`), C creates 1 node, 1 edge (reconfirmed 2026-11-22).
   Also `UNWIND [1,2] AS i MERGE p=(x:X {v:i})`, `… CALL { WITH a MERGE p=… }`,
   `FOREACH (i IN [1] | CREATE (y:Y) MERGE p=(y)-[:T]->(:X))`. Adding `RETURN …` works.
   Fix: add `IR::PathBuilder(_)` to the first walk's list (and the FOREACH body walk), or reuse the
   loop walk. Tests `bug_merge_named_path_as_last_clause`, `…_last_in_call_body`, `…_last_in_foreach_body`.
5. **`NOT (complex)` over an inline pattern predicate keeps NULL rows** (NEW): `expr_to_plan` plans
   `NOT c` with `c` containing an inline pattern as `AntiSemiApply(input, plan(c))` (mod.rs); the
   inner plan keeps rows where `c` is *true*, so the anti-join keeps rows where `c` is false **or null**.
   openCypher (and C) keep only `c = false`. Graph `CREATE (:N {v:1}), (:N), (:N {v:5})-[:R]->(:M)`:
   `MATCH (a:N) WHERE NOT (a.v > 1 OR (a)-->()) RETURN count(a)` → Rust 2, C 1;
   on `CREATE (:N)-[:R]->(:M)`: `MATCH (a:N) WHERE NOT (a.v > 0 AND (a)-->()) RETURN count(a)` → Rust 1, C 0.
   Lean `toPlan_not_null_bug` (`toPlan_correct` holds once atoms are two-valued). Fix: push `NOT` inward
   (De Morgan, valid in Kleene logic: `not3_or3`, `not3_and3`) so it only ever wraps a scalar (→ `Filter`) or
   an inline pattern (→ `AntiSemiApply`). Tests `bug_not_over_or_with_pattern_keeps_null_rows`,
   `bug_not_over_and_with_pattern_keeps_null_rows` (fail today).

Also open: `utilize_node_by_id` should skip a scan with no parent instead of `unwrap()` (e.g. a scan at the plan root).

8. **CREATE with a named path as the last clause, after any other clause** (NEW — a regression from
   #2829, `eab2722c4`, which wraps a CREATE that declares a path in `PathBuilder`, mod.rs):
   bug 1's shape — the first walk (mod.rs) does not step over `PathBuilder`, so the previous
   clause lands as `PathBuilder(prev, Create(..))` and `Create` never runs. Live on e8f8a3017 and again on 2026-11-22 (Rust
   release vs C): `MATCH (a:A) CREATE p=(a)-[:R]->(:B2)` → Rust `Variable ? not found`, nothing
   created; C creates 1 node + 1 edge; the pre-#2829 Rust build created them too. Same for
   `UNWIND [1,2] AS x CREATE p=(:N {x:x})` (Rust error, C 2 nodes). With `RETURN p` (not last) it works:
   `create_loop_correct`. Lean `create_path_last_misplaced`, `create_path_last_skips_create`.
   Fix: the bug 1 fix (#3101 adds `PathBuilder` to the first walk) covers this too.

## Gap closed 2026-10-20: named paths over an already-bound node (PathBind.lean)
Open issues #2311, #2727, #2737, #2743, #2744, #2725 were missed: `nodeOnly_bound_*` described the
bound node-only branch (`plan_match`, mod.rs) but no property said a named path gets built, and the MATCH
arm's Apply decision was not modelled. Now: the property `E.BindsAllPaths` (every named path of a
MATCH / OPTIONAL MATCH pattern is bound by the plan) and `Plan.argIsLeaf` (no `Argument` has a child);
root causes `nodeOnly_bound_no_paths` (the bound branch never emits `PathBuilder`) and
`planMatchArm_path_unwrapped` (`all_visited`, mod.rs, counts the new path variable, so the plan is
not Apply-wrapped and stitching puts the previous clauses under its `Argument`, which the runtime never
runs: `argument_fill_ignores_input`). Counterexamples mirroring the live repros (Rust 9d3869aa8 vs C):
`bug2311_optional_path_null`, `bug2737_optional_path_null` (OPTIONAL: path null),
`bug2743_bound_path_drops_rows` (count 0 vs 1), `bug2744_bound_path_unfed_argument`,
`bug2727_alias_path_drops_rows` (0 rows vs 1), `bug2725_call_then_bound_path_loses_var` (one empty row:
`Variable n0 not found`). `unbound_path_built`: the unbound branch satisfies the property.
Not in this project: #2502 (the parser's `parse_pattern`, cypher.rs, called for a WHERE pattern
predicate, swallows a following `MATCH` keyword and merges that clause's pattern into the
predicate — `MATCH (a:A) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` → `'m' not defined`), and
#2556 (the planner's Filter for `t {id:100}` sits right above the hop binding `t` —
`firstRel_lowers_to`/`chainRel_lowers_to`; the optimizer moves it below, see the working variants'
EXPLAIN).

## Gap closed 2026-10-22: `verified_labels` survive REMOVE (#3450, RemoveVerified.lean)
`MSt.verified` was only ever unioned and `remove_correct` keeps REMOVE's graph effect opaque
(`specWrite`), so no property tied a verified label to the graph. Now: `VerifiedSound st gl` (every
verified label is on the node in the graph the next operator reads) and the property
`RemoveKeepsVerifiedSound`; `specRemove_sound` (dropping the removed labels preserves it),
`project_sound` (`plan_project`'s `carried_verified`, mod.rs, preserves it).
`bug3450_remove_keeps_verified_label`: the REMOVE arm (mod.rs) leaves `verified_labels`
alone, so `MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH (a:A)-[:R]->(b)` plans the bound `a` with no
`hasLabels` filter (`bound_filters` for an endpoint already in `visited`, mod.rs / `unverified_labels` :1978): Rust 1 row,
C 0. The binder half (`node_labels`) is in proofs/binder `RemoveLabels.lean`.

## PRs touching planner/mod.rs (model = origin/main `256ab4d3f`)
* #3101 (MERGE/CREATE p=… as last clause) is **not merged**: bugs 1 and 8 reproduce live (2026-11-22).

## Deviations vs openCypher (and C)
* `SET n.v = 10, n.w = n.v` → Rust `n.w = 10` (items in order, openCypher/Neo4j), C uses the old value.
* `Labels added` counts differ (C counts per node) — known (#2655).

## Suspected, not confirmed
* `expr_to_plan`/`or_expr_to_plan` key `inline_map` by id only (mod.rs); ids are per scope.
  `planFilter_correct` needs all patterns of one WHERE to start in one scope and the predicate's ids to be
  below that scope's length (`Pre`); no query breaking it was found.
* `emit_rel` path membership (mod.rs: `v.id == rel.alias.id`) still compares ids without scope; `filter_vars` and `visited` are keyed by `VarKey` (mod.rs).
* `connected_components` visited set is id-only (ast.rs: `visited` holds `node.alias.id`).

## Gaps
Re-checked against the Rust at 256ab4d3f (2026-11-22, every CHANGED row): the clause plans omit the
expression-comprehension Apply chains, SET's multi-`Set` split, LOAD CSV and index DDL (see Clauses.lean);
`clauseMin` lacks LOAD CSV and the fulltext/vector scan leaves; `patternExprVariables` abstracts the order of
variables found only in pattern attribute maps.
Scans/hops/expressions/writes are parameters (`Sem`, `atom`, `sub`); `plan_match`'s operator choice is proved
as a table plus binding/label invariants, not against a graph-matching semantics (C-parity checked by
differential tests); runtime errors in expressions are not modelled (Kleene logic only); a pattern's inline
attribute expressions are abstracted to the variables they read (`QG.attrVars`); the optimizer and batching
are other projects. Clause semantics
thread the graph state clause-at-a-time (eager), matching the plan's `Commit`/eager operators, not
per-row interleaving. Sub-plans (MERGE match branch, FOREACH/CALL bodies) are hypotheses
(`MatchBranch`, `Body`). See COVERAGE.tsv.
-/
import PlannerBuild.Core
import PlannerBuild.Stitch
import PlannerBuild.Clauses
import PlannerBuild.Match
import PlannerBuild.Expr
import PlannerBuild.Anchor
import PlannerBuild.Decomp
import PlannerBuild.AndOf
import PlannerBuild.ToPlan
import PlannerBuild.Filter
import PlannerBuild.FilterPlan
import PlannerBuild.Nested
import PlannerBuild.Nested2
import PlannerBuild.Misc
import PlannerBuild.PlanMatch
import PlannerBuild.Lift
import PlannerBuild.PlanMatch2
import PlannerBuild.RemoveVerified
import PlannerBuild.HopOrder
import PlannerBuild.Fmt
import PlannerBuild.Rename
import PlannerBuild.Ast
import PlannerBuild.Ast2
import PlannerBuild.PathBind
