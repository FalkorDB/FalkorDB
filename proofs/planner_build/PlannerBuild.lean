/-
# Plan construction vs openCypher clause-sequence semantics

A model of `graph/src/planner/mod.rs` plan construction (`Planner::plan`,
`plan_query` stitching, `plan_project`, the clause arms) and of the IR's
reference semantics, with machine-checked agreement against an openCypher
driving-table semantics (each clause maps a bag of records, plus the graph
state, to a bag of records). `lake build` succeeds; no `sorry`/`admit`/`axiom`.
522 theorems. Rust source: origin/main `b84473a7f` (re-targeted across #2390, d2c42e032
"plan an inline property map once", and #3335: `ExprIR::and_of`/`or_of` flattening —
`AndOf.lean`, `tv_andOf`/`tv_orOf` — and no `Paren` past the binder: `expr_to_plan` no
longer unwraps one, `PatternMode::descend` no longer treats it as a connective; #3297: the
clauses after a procedure CALL are validated, `Ast.call_validates_rest`). Rust repros: `cargo test -p graph --test lean_planner_build`
(`bug_*` tests assert the C/openCypher answer and FAIL today).

## Lean ↔ Rust

| here | there |
| --- | --- |
| `Op`, `Plan` | `enum IR` mod.rs:65-328, `DynTree<IR>` |
| `ev`, `evUnion`, `evRights`, `evAny` | `Runtime::build_batch_op` / `children_to_recurse` runtime/runtime.rs:654-1180 (child-0 input, `pop_or_once`, CP right branches materialised on the argument row) |
| `forRows` | openCypher "for each record" |
| `walkFirst` / `walkLoop` | insertion-point walks mod.rs:2743-2753 / 2712-2729 |
| `projDescend`, `applyChain` | mod.rs:2756-2769, 2730-2743 |
| `descendOne`, `descendClause`, `satChain`, `saturated` | mod.rs:974-1017, 836-841 |
| `insertStep`, `stitchLoop`, `rustStitch` | mod.rs:2774-2850 (`push_sibling_tree(Left)` / `push_child_tree`, Apply-wrap of a CartesianProduct) |
| `needsApplyWrapping`, `addArgs` | mod.rs:2874-2907, 748-782 |
| `planProject`, `ProjC` | `plan_project` mod.rs:2415-2586 |
| `planMerge` | MERGE arm mod.rs:3100-3133 |
| `planProc` | CALL procedure arm mod.rs:2941-3032 |
| `planUnion` | UNION arm mod.rs:3312-3329 |
| `.apply [.aggregate _ [b]]`, `.apply [b]` | CALL {} arm mod.rs:3330-3415 |
| `.forEach e x [b]` | FOREACH arm mod.rs:3416-3479 |
| `.optional vs [m]`, `.apply [.optional vs [m]]`, `.apply [m]` | MATCH arm mod.rs:3034-3083 |
| `isPlannerScanSubtree` | optimizer/select_scan_node.rs:239-252 |
| `E.rsRemove`, `E.specRemove`, `E.project`, `E.VerifiedSound` (RemoveVerified.lean) | REMOVE arm mod.rs:3212-3218, `carried_verified` in `plan_project` mod.rs:2479-2492 |
| `E.Ex`, `E.D`, `E.QG`, `E.V` (Expr.lean) | `DynTree<ExprIR<Variable>>`, `QueryGraph`, `(id, scope_id)` |
| `E.containsInlineVar`, `hasPatternExpr`, `patternExprScope`, `Mode`, `needsExtraction`, `inlineAttrsToFilter`, `hasLabelsFilter`, `patternExprVariables` | mod.rs:933, 822, 1223, 660-689, 1247, 543, 574, 1128 |
| `E.tv` (Decomp.lean) | openCypher three-valued WHERE (Kleene AND/OR/NOT; pattern = has a match) |
| `E.collect`, `E.mint` | `collect_patterns_and_rebuild` mod.rs:1692-1784 |
| `E.FP`, `E.run`, `E.toPlan`, `orClass`, `andFold` (ToPlan.lean) | Filter/SemiApply/AntiSemiApply/OrApplyMultiplexer runtime; `expr_to_plan` / `or_expr_to_plan` / `and_expr_to_plan` mod.rs:1493-1670 |
| `E.planFilter` (FilterPlan.lean) | `plan_filter` mod.rs:2597-2635 |
| `E.PSt`, `E.fresh`, `E.EC`, `E.build`, `E.hoist`, `E.extract` (Nested.lean) | Planner state, `fresh_var`, `ExtractedComprehension`, `build_pattern_comprehension_plan`, `hoist_or_nest`, `extract_pattern_comprehensions` |
| `E.chainOf`, `extractClause`, `extractFilter` (Nested2.lean) | `extract_clause_expr_comprehensions` / `extract_list_expr_comprehensions` / `extract_filter_comprehensions` |
| `E.QN`, `QR`, `Comp`, `MSt`, `MP`, `firstRel`, `chainRel`, `nodeOnly`, `planComp`, `planMatch` (PlanMatch*.lean) | `plan_match` mod.rs:1832-2403, `mark_labels_verified`, `unverified_labels`, `build_pattern_sub_plan` |
| `HopOrder.selfLoopsAfterBinders`, `step`, `flush`, `deferP`; `pre3102_reference` (HopOrder.lean); `E.hopShape`, `E.orderRels` | `self_loops_after_binders` hop_order.rs:25-62; `tests::reference` :84; the call mod.rs:1912-1926 |
| `E.lowerInline`, `stripNode`, `stripRel`, `stripEndpoints`, `MP.stripped` (PlanMatch.lean) | `lower_inline_attrs` mod.rs:598, `strip_node_attrs` :623, `strip_rel_attrs` :644, `strip_endpoint_attrs` :673 |
| `subtreeContains`, `stitchBelow`, `ensureInput`, `setIP`, `isRedundantOptional` (Misc.lean) | mod.rs:331, 900, 2806, 787, 2582 |
| `Fmt.*` | EXPLAIN text mod.rs:341-536 |
| `Ast.*`, `Ast2.*` | parser/ast.rs: `Variable` impls, `QueryGraph` builders, `filter_visited`, `connected_components`/`dfs`, `validate`/`inner_validate`, `return_column_names`, the `Display` impls |
| `E.renameProj`, `newPSt` (Rename.lean) | `rename_projection_outputs` mod.rs:2646, `Planner::new` mod.rs:814 |
| `E.MP.pathVars`, `E.BindsAllPaths`, `E.compVars`, `E.planMatchArm`, `E.MP.toPlan`, `Plan.argIsLeaf` (PathBind.lean) | paths a `plan_match` tree binds (PathBuilder mod.rs:1977-1980, elision :2350-2387); `QueryGraph::variables` ast.rs:816-822; MATCH arm mod.rs:3034-3082; Argument builds no children runtime.rs:658-660, emits one row :1285 |

## Proven (plain English)

* `self_loops_after_binders` (#3102, HopOrder.lean): `selfLoops_perm` (permutation), `kept_sublist` (non-deferred
  hops keep their order), `deferred_after_binder` (a deferred self-loop comes after its first later binder),
  `single_pass` (one fold, one step per hop), `early_return_sound`, **`eq_reference`** (= the pre-#3102 loop
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
  last; the path bound since #2829), `create_first_correct` (CREATE without path as last clause).
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
  (fixed < var-length < shortest, stable permutation), `planComp_binds`, bound-label filters
  `labelStep_spec`/`labelStep_verifies`, `nodeOnly_*` (synthetic self-loop id `u32::MAX - id`), path elision
  `elide_*`, joining `planMatch_join`/`planMatch_cp`, `buildPatternSubPlan_restores`.
* #2390 (inline maps lowered once, PlanMatch*): `lowerInline_once` (one Filter per (alias, map) —
  no more `And(p, p)`), `lowerInline_two_maps`; `firstRel_lowers_from`/`_to`, `chainRel_lowers_from`/`_to`
  (both endpoints of every hop lowered to Filters, bound or not, for every operator kind);
  `firstRel_edge_pred`/`chainRel_edge_pred` (a fixed hop's edge map is a Filter right above an
  operator with `emit_relationship` set); `walk_keeps_edge_attrs`; `planMatch_stripped` (no scan or
  fixed-length operator carries an inline map — the precondition of proofs/optimizer_rewrites
  `refsNew_complete`); fixed bug (counterexample removed; in 17ccc5412) (live: `MATCH (a:N) WITH a MATCH
  (a {x: 1})-[:R*1..2]->(b)` gave 2, 3, 3; now 2, 3 as C). Match: `rebuild_keeps_where` (bug 3 fixed).
* Helpers (Misc/Fmt/Rename): `subtreeContains_iff`, `addArgs_leaves`, `stitchBelow_get`/`_ev`,
  `ensureInput_ev_local`/`_saturates`, `setIP_scansLeaf`, `redundant_optional_identity`, `renameProj_*`,
  `inlineAttrs_passes`, EXPLAIN strings.
* Counterexamples backing the bugs: `merge_path_last_misplaced`, `merge_path_last_skips_merge`,
  `create_path_last_misplaced`, `create_path_last_skips_create`,
  `selfloop_ignores_chain`, `cp_stitch_shape` + `cp_stitch_loses_correlation` (by `decide`),
  `where_scan_indistinguishable`.

## CONFIRMED bugs (Rust release vs C `bin/macos-arm64v8-release/falkordb.so`, live servers; plus Rust tests)
Graph: `CREATE (a:A {v:1})-[:R {w:2}]->(b:B {v:0}), (:B {v:5}), (a)-[:R {w:3}]->(b), (b)-[:S {w:1}]->(:C {v:7})`.
Re-checked live on 8743953a8 (Rust release vs C): 1, 2, 4, 5, 6 (crash), 7 still reproduce; 3 is fixed.
1. **MERGE with a named path as the last clause** (of a query, a `CALL {}` body or a FOREACH body)
   fails: the first walk mod.rs:2743-2749 (and FOREACH's mod.rs:3450) does not step over
   `PathBuilder`, so the previous clause becomes `PathBuilder(prev, Merge(..))` and `Merge` never runs.
   `MATCH (a:A) MERGE p=(a)-[:T]->(x:X)` → Rust `Variable _anon_0 not found`; C creates 1 node, 1 edge.
   Also `UNWIND [1,2] AS i MERGE p=(x:X {v:i})`, `… CALL { WITH a MERGE p=… }`,
   `FOREACH (i IN [1] | CREATE (y:Y) MERGE p=(y)-[:T]->(:X))`. Adding `RETURN …` works.
   Fix: add `IR::PathBuilder(_)` to the first walk's list (and the FOREACH body walk), or reuse the
   loop walk. Tests `bug_merge_named_path_as_last_clause`, `…_last_in_call_body`, `…_last_in_foreach_body`.
2. **A `MATCH … WHERE` followed by a multi-component MATCH is cross-joined, not correlated**:
   `needs_apply_wrapping` (mod.rs:2874-2894) treats a Filter-over-scan as a plain CP branch and
   mod.rs:2809-2813 prepends it; the component that mentions the earlier variable was planned as if it
   were bound, but under the CartesianProduct it re-binds it. `MATCH (a) WHERE a.v = 0 MATCH (a)-[:R]->(b), (c:C) RETURN a.v, b.v, c.v`
   → Rust `[1,0,7]`, C `[]`; `MATCH (a) WHERE a.v = 0 MATCH (x)-[:S]->(a), (b:B) …` → Rust 2 rows
   (a.v = 7!), C `[]`; `MATCH (a {v:5}) WHERE true MATCH (x:C), (a)-[:R]->(b) RETURN count(*)` → 1 vs 0.
   Fix: Apply-wrap whenever `n` binds a variable any CP branch uses (or always for a previous clause).
   Tests `bug_prior_match_stitched_as_cartesian_branch`, `bug_prior_match_cartesian_branch_incoming`.
3. **(FIXED by #2390, d2c42e032 — select_scan_node `filters_of` keeps the spine's Filters on
   rebuild; live on 8743953a8: Rust 0 = C 0; Lean `rebuild_keeps_where`; counterexample removed, in 17ccc5412.)** **The WHERE of a `MATCH (a) WHERE φ` is dropped when the next MATCH traverses from `a`**:
   stitching puts `Filter(φ, Scan a)` under the next clause's leaf CondTraverse (mod.rs:2815), the
   exact shape `is_planner_scan_subtree` (select_scan_node.rs:239) prunes and rebuilds as a bare scan.
   `MATCH (a) WHERE a.v = 99 MATCH (a)-[r:R]->(b) RETURN count(r)` → Rust 2, C 0; also into writes:
   `… MATCH (a)-[:R]->(b) DELETE b` deletes a node (C nothing), `SET b.hit = 1` sets 1 property.
   (`WITH a` between them is fine.) Fix: Apply-wrap the previous clause, or mark planner-built scans.
   Tests `bug_where_of_previous_match_dropped`, `…_before_delete`.
4. **FIXED by #3102 (61284ea4f; #2999)** — *was:* **A fixed-length self-loop hop after other hops drops them**: when var-length hops are sorted after
   fixed ones (mod.rs:1877-1909) a self-loop on an unbound node is planned as `ExpandInto(scan, res)`
   (mod.rs:2247-2264); the runtime builds only child 0, so `res` is lost.
   `MATCH (a)-[:R]->(b)-[:S*]->(c)-[:L]->(c) RETURN a.v, b.v, c.v` on `(:A{v:1})-[:R]->(:B{v:2})-[:S]->(c:C{v:3}), (c)-[:L]->(c)`
   → Rust `[null,2,3]`, C `[1,2,3]`. Test `bug_self_loop_hop_drops_earlier_hops`.
   Now: `orderRels` (= `self_loops_after_binders`, HopOrder.lean) plans such a self-loop after the hop
   binding its node, and the no-binder case is `ExpandInto(Scan(res))` — `chainRel_selfloop_unbound`,
   `Match.selfloop_scan_keeps_chain`; history `selfloop_ignores_chain`.

5. **`NOT (complex)` over an inline pattern predicate keeps NULL rows** (NEW): `expr_to_plan` plans
   `NOT c` with `c` containing an inline pattern as `AntiSemiApply(input, plan(c))` (mod.rs:1543-1549); the
   inner plan keeps rows where `c` is *true*, so the anti-join keeps rows where `c` is false **or null**.
   openCypher (and C) keep only `c = false`. Graph `CREATE (:N {v:1}), (:N), (:N {v:5})-[:R]->(:M)`:
   `MATCH (a:N) WHERE NOT (a.v > 1 OR (a)-->()) RETURN count(a)` → Rust 2, C 1;
   on `CREATE (:N)-[:R]->(:M)`: `MATCH (a:N) WHERE NOT (a.v > 0 AND (a)-->()) RETURN count(a)` → Rust 1, C 0.
   Lean `toPlan_not_null_bug` (`toPlan_correct` holds once atoms are two-valued). Fix: push `NOT` inward
   (De Morgan, valid in Kleene logic: `not3_or3`, `not3_and3`) so it only ever wraps a scalar (→ `Filter`) or
   an inline pattern (→ `AntiSemiApply`). Tests `bug_not_over_or_with_pattern_keeps_null_rows`,
   `bug_not_over_and_with_pattern_keeps_null_rows` (fail today).

6. **Nothing after a procedure CALL was validated** — **FIXED by #3297** (b774fdfad, on main b84473a7f):
   `inner_validate`'s `Call` arm (parser/ast.rs:1224-1263) now continues with
   `iter.next().map_or(Ok(()), |first| first.inner_validate(iter))`. Lean `call_validates_rest`,
   `call_transparent`, `call_then_match_rejected`, `call_then_unwind_rejected`,
   `call_then_create_multi_type_rejected` (each now gives C's error). Before the fix:
   `CALL db.labels() YIELD label MATCH (n)` reached a scan at the plan root (`utilize_node_by_id`
   `parent().unwrap()`), `… UNWIND [1] AS x` was accepted, `… CREATE (a)-[:R|S]->(b)` created an edge.
   Still open independently: `utilize_node_by_id` should skip a scan with no parent instead of `unwrap()`.
7. **A query may end with WITH** (NEW, deviation from C and openCypher): the `With`/`Return` arm
   (ast.rs:1268-1271) accepts a final WITH: `MATCH (n) WITH n` / `UNWIND [1] AS x WITH x` → Rust returns
   rows, C `Query cannot conclude with WITH (must be a RETURN clause, …)`. Lean `with_last_accepted`.
8. **CREATE with a named path as the last clause, after any other clause** (NEW — a regression from
   #2829, `eab2722c4`, which wraps a CREATE that declares a path in `PathBuilder`, mod.rs:3134-3153):
   bug 1's shape — the first walk (mod.rs:2743-2749) does not step over `PathBuilder`, so the previous
   clause lands as `PathBuilder(prev, Create(..))` and `Create` never runs. Live on e8f8a3017 (Rust
   release vs C): `MATCH (a:A) CREATE p=(a)-[:R]->(:B)` → Rust `Variable _anon_0 not found`, nothing
   created; C creates 1 node + 1 edge; the pre-#2829 Rust build created them too. Same for
   `UNWIND [1,2] AS x CREATE p=(:N {x:x})` (Rust error, C 2 nodes). With `RETURN p` (not last) it works:
   `create_loop_correct`. Lean `create_path_last_misplaced`, `create_path_last_skips_create`.
   Fix: the bug 1 fix (#3101 adds `PathBuilder` to the first walk) covers this too.

## Gap closed 2026-10-20: named paths over an already-bound node (PathBind.lean)
Open issues #2311, #2727, #2737, #2743, #2744, #2725 were missed: `nodeOnly_bound_*` described the
bound node-only branch (mod.rs:1915-1962) but no property said a named path gets built, and the MATCH
arm's Apply decision was not modelled. Now: the property `E.BindsAllPaths` (every named path of a
MATCH / OPTIONAL MATCH pattern is bound by the plan) and `Plan.argIsLeaf` (no `Argument` has a child);
root causes `nodeOnly_bound_no_paths` (the bound branch never emits `PathBuilder`) and
`planMatchArm_path_unwrapped` (`all_visited`, mod.rs:3067, counts the new path variable, so the plan is
not Apply-wrapped and stitching puts the previous clauses under its `Argument`, which the runtime never
runs: `argument_fill_ignores_input`). Counterexamples mirroring the live repros (Rust 9d3869aa8 vs C):
`bug2311_optional_path_null`, `bug2737_optional_path_null` (OPTIONAL: path null),
`bug2743_bound_path_drops_rows` (count 0 vs 1), `bug2744_bound_path_unfed_argument`,
`bug2727_alias_path_drops_rows` (0 rows vs 1), `bug2725_call_then_bound_path_loses_var` (one empty row:
`Variable n0 not found`). `unbound_path_built`: the unbound branch satisfies the property.
Not in this project: #2502 (the parser's `parse_pattern`, cypher.rs:1562-1572, called for a WHERE pattern
predicate at cypher.rs:2119, swallows a following `MATCH` keyword and merges that clause's pattern into the
predicate — `MATCH (a:A) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` → `'m' not defined`), and
#2556 (the planner's Filter for `t {id:100}` sits right above the hop binding `t` —
`firstRel_lowers_to`/`chainRel_lowers_to`; the optimizer moves it below, see the working variants'
EXPLAIN).

## Gap closed 2026-10-22: `verified_labels` survive REMOVE (#3450, RemoveVerified.lean)
`MSt.verified` was only ever unioned and `remove_correct` keeps REMOVE's graph effect opaque
(`specWrite`), so no property tied a verified label to the graph. Now: `VerifiedSound st gl` (every
verified label is on the node in the graph the next operator reads) and the property
`RemoveKeepsVerifiedSound`; `specRemove_sound` (dropping the removed labels preserves it),
`project_sound` (`plan_project`'s `carried_verified`, mod.rs:2479-2492, preserves it).
`bug3450_remove_keeps_verified_label`: the REMOVE arm (mod.rs:3212-3218) leaves `verified_labels`
alone, so `MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH (a:A)-[:R]->(b)` plans the bound `a` with no
`hasLabels` filter (`boundLabelFilters`, mod.rs:1840-1856 / `unverified_labels` :1793): Rust 1 row,
C 0. The binder half (`node_labels`) is in proofs/binder `RemoveLabels.lean`.

## Open PRs touching planner/mod.rs (model = origin/main)
* #3101 (MERGE p=… as last clause): adds `PathBuilder` to the first walk and steps over it in the FOREACH /
  CALL body walk — fixes bug 1 (`merge_path_last_misplaced` no longer applies: `walkFirst` would pass it).
* #3100 (correlate MATCH … WHERE with the next CP): Apply-wraps when the CP subtree reads a variable the
  previous plan binds (`reads_plan_variables`) — fixes bug 2 (`cp_stitch_sound` is the remaining CP case).
* #3102 merged (61284ea4f), see bug 4 and HopOrder.lean.
* #2981 (inline attrs reading another pattern variable): splits inline maps (`split_inline_attrs`) and checks
  the moved entries where the variable is bound — changes `inline_attrs_to_filter` call sites in `plan_match`
  (binder #2923 case); `inlineAttrs_passes` still describes each produced filter. #2390 now routes those
  call sites through `lower_inline_attrs`, so #2981 needs a rebase; the #2923 node case is still
  present on 8743953a8 (live: `MATCH (a:A), (b:B {v: a.v})` Rust [], C 2 rows).

## Deviations vs openCypher (and C)
* (FIXED by #2829, `eab2722c4`) `CREATE p=(…) RETURN p` → Rust `null` (the Create arm added no
  PathBuilder); now the path (openCypher), live `[(0), [0], (1)]`; C rejects (`'p' not defined`).
  Lean `create_loop_correct`. Interplay with parser_grammar's
  `create_named_path_drops_dup_rel` (W4-parse-2): `CREATE p=(a)-[r:R]->(b), q=(c)-[r:R]->(d) RETURN p, q`
  now returns `q = [(2), [0], (3)]`, a path through edge 0, whose endpoints are (0)→(1) — the dropped
  duplicate `r` is now visible as an inconsistent path (C rejects the query).
* `SET n.v = 10, n.w = n.v` → Rust `n.w = 10` (items in order, openCypher/Neo4j), C uses the old value.
* `Labels added` counts differ (C counts per node) — known (#2655).

## Suspected, not confirmed
* `expr_to_plan`/`or_expr_to_plan` key `inline_map` by id only (mod.rs:1499, 1388); ids are per scope.
  `planFilter_correct` needs all patterns of one WHERE to start in one scope and the predicate's ids to be
  below that scope's length (`Pre`); no query breaking it was found.
* `emit_rel` path membership and `filter_vars` compare ids without scope (mod.rs:1903, 1882).
* `connected_components` visited set is id-only (ast.rs:823-893).

## Gaps
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
import PlannerBuild.Decomp
import PlannerBuild.AndOf
import PlannerBuild.ToPlan
import PlannerBuild.Filter
import PlannerBuild.FilterPlan
import PlannerBuild.Nested
import PlannerBuild.Nested2
import PlannerBuild.Misc
import PlannerBuild.PlanMatch
import PlannerBuild.PlanMatch2
import PlannerBuild.RemoveVerified
import PlannerBuild.HopOrder
import PlannerBuild.Fmt
import PlannerBuild.Rename
import PlannerBuild.Ast
import PlannerBuild.Ast2
import PlannerBuild.PathBind
