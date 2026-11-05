/-
# The binder's scopes: what a variable id refers to, and when ids collide

A model of `graph/src/planner/binder.rs` (and the id-minting side of
`graph/src/planner/mod.rs`) and a machine-checked account of its scoping.
`lake build` succeeds; no `sorry`, `admit` or `axiom`; every theorem depends at
most on `propext`, `Quot.sound`, `Classical.choice`. 185 theorems. Rust source: origin/main `5c7af54f2`
(the line numbers below are read off that tree; binder.rs was re-targeted across #3335: `(x)` binds to `x`'s tree — `bind_paren` — and AND/OR
bind flat via `and_of`/`or_of` — `bind_andOr_paren`, `flatB_assoc`; planner/mod.rs cites
re-targeted across #2390).

## The one invariant that matters

Every id is minted as the **current size of its scope table**: `fresh_var`
(binder.rs:2260), the parent copy in `resolve_name` (binder.rs:2192, calls `fresh_var` at 2200/2211), and the
planner's `fresh_var` (planner/mod.rs:838, called at 1068, 1108, 1114 and 1461), which gets
`scope_vars[s] = sorted_scope_vars(env_stack[s])`. `mint_fresh_iff` proves the
planner's ids `len, len+1, …` avoid every id `H` the bound IR uses iff
`∀ i ∈ H, i < len`. Minting under a new key keeps that (`mint_new_covers`,
`resolve_covers`, `projectName_new_covers`, `foreachRestore_covers`,
`mergeSV_covers`). **Removing** a key breaks it, and the binder removes keys
in two places while the removed ids are still in the IR
(`erase_breaks_covers`, `where_copy_id_reused`, `comprehension_cleanup_breaks`).

## Gap closed 2026-10-20: pattern predicates may declare variables (#1334)
`bind_expr_node`'s `Pattern` arm (binder.rs:2082-2113) binds a WHERE pattern predicate with
`bind_graph(…, false)`, which mints any unknown alias fresh. Property `PatternPredDeclaresNothing`
(an accepted pattern predicate names only visible variables); refuted by
`bug1334_pattern_predicate_declares_var`: `MATCH (n0) WITH * WHERE () -- (n1) RETURN *` binds with
`n1` fresh (id 2) — live Rust returns every `n0`, C rejects it (issue #1334; the C error message is not recorded here).
The slot-reuse family (#3333, #3369, #3375, #3368, #3150) is modelled in proofs/ops_aggregate
`SlotReuse.lean` (`projLayout` = this binder's per-scope minting fed to the aggregate).

## Gap closed 2026-10-22: REMOVE does not clear accumulated labels (#3450)
`LClause` gains `.remove` (`bind_ir` QueryIR::Remove, binder.rs:477-482, leaves `node_labels`
alone); `bind_projection` copies the parent's label set to the `WITH` child (:1135-1146/:1162-1173) and
`update_all_node_labels` (:122-153) stamps it on the next MATCH. Property `StampedLabelsHeld` (a stamped
label is written by that MATCH or still held, point-in-time via `held`); `specStamp_held` proves the fix
satisfies it, `bug3450_remove_label_kept_across_with` refutes it on main: `MATCH (a:A) REMOVE a:A WITH a
MATCH (a)-[:R]->(b)` stamps `(a:A)` — Rust 0 rows, C 1. (RemoveLabels.lean; the planner half is in
proofs/planner_build `RemoveVerified`; the issue's CREATE/MERGE half does not reproduce.)

## Lean ↔ Rust

| here | there |
| --- | --- |
| `Env`, `Env.insert/erase/retain` | `HashMap<Arc<String>, Variable>` scope table (binder.rs:47) |
| `freshVar` | `Binder::fresh_var` binder.rs:2260 |
| `mint_fresh_iff` | `Planner::fresh_var` planner/mod.rs:838-851 |
| `St`, `pushScope` | `env_stack`, `use_parent_scope`, `parent_to_child_scope`, `copy_from_parent`; `push_scope` :251, `commit_scope` :256 |
| `resolve`, `lookupLocals` | `resolve_name` binder.rs:2192-2258 |
| `defineName`, `ensureType` | `define_name_in_scope` :2160, `ensure_type` :2276 |
| `projectName` | `project_name` :2182 |
| `bindProjection`, `projectItems` | `bind_projection` :1064-1306 (non-`*` form) |
| `rustBind` / `specBind`, `Elem`, `split` | `bind_graph` :1333 (nodes 1356, then rels 1373) / C's textual order |
| `fold`, `bindClauses` | `parse_pattern` part separator parser/cypher.rs:1565-1582 (and :2117 for predicates) |
| `mergeSV`, `lenAt` | `merge_scope_vars` :2711, `Binder::bind` :99 |
| `foreachRestore` | FOREACH arm of `bind_ir` :673-709 |
| `compScope`/`compCleaned` | cleanup in `bind_expr_node` :1927 and :2106 |
| `callBodyScope` | `bind_call_body` :824, stack push :845 (Query) vs UNION branch binder :903 |
| `accumulate` / `specScanLabels` | `node_labels` accumulation :1360-1366, `update_all_node_labels` :122 |
| `held`, `rsStamp`, `specStamp` (RemoveLabels.lean) | REMOVE arm :477, WITH label copy :1164-1173, stamp :122-150 |
| `unionCheck` | UNION arm :267-294 |
| `validateImport` | `validate_import_with` :1008 |
| `BSt`, `sortedScopeVars`, `updNode` (Stack.lean) | `env_stack` & co., `default` :74, `scope_vars` :114, `current_env(_mut)` :239/245, `push_scope` :251, `commit_scope` :256, `sorted_scope_vars` :2703, `update_graph_labels` :160 |
| `NE`/`BE`, `bindNode`, `bindSetItems`, `selfRef` (BindExpr.lean) | `bind_expr_node` :1751, `bind_expr(_with_locals)` :1730/1737, `bind_set_items` :1705, `check_unbound_self_referential` :1687 |
| `RC`, `bindIR`, `shortestOk`, `mergeOk`, `yieldLoop`, `bindYield`, `bindCallSubquery`, `importProjections`, `createCheck` (BindIR.lean) | `bind_ir` :262, `bind_call_subquery` :716, `build_import_projections` :984, `bind_graph_create` :1562 |
| `mayBool`, `mayEntity`, `validate` (Types.lean) | `expr_may_return_boolean` :2323, `expr_may_return_entity` :2392, `validate_boolean_operands(_impl)` :2297/2301 |
| `nodeToValue`, `collectConstArgs`, `foldCall`, `rewriteStruct`, `rewriteRegex` (Fold.lean) | `expr_tree_to_value` :2465, `collect_constant_args` :2497, folding :2124, `rewrite_struct_constructor` :2515, `rewrite_compiled_regex` :2580 |
| `exprIrEq`, `strip`/`eqS`, `structEq`, `replaceAgg` (AggEq.lean) | `expr_ir_eq` :2682, `nodes_eq` :2652, `raw_exprs_structurally_equal` :2644, `replace_agg_subtrees` :2614 |

## Proven (one line each)
* `fresh_not_used`, `mint_new_covers`, `reinsert_covers`: minting under a new key is fresh and keeps the table covering.
* `mint_fresh_iff`: the planner's len-based ids are collision-free iff every used id is below the reported length.
* `resolve_ok_iff`: `resolve_name` succeeds exactly on the spec-visible names (locals, clause scope, pre-projection scope while ORDER BY/WHERE is bound).
* `resolve_innermost`: it returns the innermost binding (locals shadow clause scope shadow parent).
* `resolve_covers`: the parent copy keeps the table covering.
* `ensureType_table`: kind conflicts rejected are exactly node/rel/path used as one another.
* `projectItems_keys`: a projection scope begins with exactly its aliases, in order.
* `with_plain`, `where_copy_removed`, `orderBy_old_name_of_projected`: projection scopes agree with the spec in those cases.
* `agree_on_outer_refs`: nodes-first binding and textual binding accept the same patterns when property maps only mention earlier-clause variables.
* `mergeSV_ge`, `mergeSV_covers`: merged scope tables are pointwise at least as long as each input.
* `foreachRestore_covers`: FOREACH restore keeps every inner id reserved.
* `callBody_query_disjoint`, `callBody_union_disjoint`: a CALL body's scopes, plain or UNION branch, are disjoint from the live outer scopes (UNION since #3027, d42d121c6).
* `unionCheck_ok_iff`, `validateImport_ok`: UNION column rule and CALL import rule as stated.
* Counterexample theorems (each backs a confirmed bug below): `erase_breaks_covers`, `where_copy_id_reused`,
  `comprehension_cleanup_breaks`, `orderBy_leaks`, `defineName_ignores_parent`, `node_attr_on_rel_rust_rejects`,
  `node_attr_on_own_rel`, `rel_attr_forward_node`, `q2923_cause`, `fold_merges`,
  `labels_leak`, `agg_orderBy_by_name`, `projectName_present_uncovered` (guarded, no bug).

* Wave 5 (all binder.rs fns PROVEN): `sortedScopeVars_spec`, `push_toSt`, `updNode_union`/`_idem`;
  `bindNode_covers` (pattern-free expressions keep the table covering), `loop_var_shadows`,
  `bind_iterable_outer`; per-arm `bindIR` theorems (`shortestOk_iff`, `mergeOk_*`, `yieldLoop_ok`,
  `callSubquery_covers`, `importProjections_covers`, `createCheck_*`); `mayBool_sound_except`, `mayEntity_eq`;
  `nodeToValue_sound`/`_complete`, `foldCall_sound`, `rewriteStruct_shape`, `rewriteRegex_spec`;
  `nodesEq_refl`/`_symm`, `exprIrEq_exact`, `replaceAgg_id`/`_root`.
* #2845 CALL entry projection (`CallBoundary`, folded in from the retired pr2845_review):
  `Pr2845.bodyNew_entry`, `importProj_spec`, `entry_ids_below`, `body_own_unbound`,
  `body_import_value`, `union_branch_label_fixed` (#3027), `pushed_read_unbound` (W6-opt-1).

## Confirmed bugs (repro: `cargo test -p graph --test lean_binder`; tests assert the C answer and fail while the bug is Open)
1. **Clause fold** (general cause of #2923): `parse_pattern`'s part separator (parser/cypher.rs:1565-1582) folds
   `MATCH p MATCH q` (and `CREATE`, and `OPTIONAL MATCH p MATCH q`) into one pattern; cypher.rs:2117 parses a
   WHERE pattern predicate with the same loop, so it swallows a following MATCH. Combined with `bind_graph`
   binding nodes before relationships (binder.rs:1356 vs 1373): `bug_node_attrs_cannot_see_rel_of_previous_match`
   (#2923; live: `MATCH (a)-[r:R]->(x) MATCH (c:B {v: id(r)*0+r.w})` → Rust "'r' not defined", C 2 rows),
   `bug_consecutive_create_cannot_see_previous_create` (Rust "'a' not defined", C 1),
   `bug_pattern_predicate_swallows_next_match` (count 1 vs C 2), `bug_relationship_uniqueness_across_match_clauses`
   (0 vs 1), `bug_optional_match_swallows_following_match` (row [null] vs spec none; C rejects).
   Also single-MATCH `(a)-[r]->(b {v: r.w-2})` rejected (C accepts).
2. **Inline attrs of one component referencing another** are filtered on that component's own scan under the
   Cartesian product (planner/mod.rs `lower_inline_attrs` :598, applied at :1919, a Filter over the scan since #2390):
   `bug_inline_attr_referencing_other_component` ([] vs C 2 rows) — #2923's node case. Still present on
   8743953a8 (live: `MATCH (a:A), (b:B {v: a.v}) RETURN a.v, b.v` → Rust [], C [[1,1],[2,2]]). #2923's
   relationship repro is the clause fold (item 1), not this scan filter.
3. **Removed-key id reuse** (binder.rs:1262-1273 WHERE/SKIP/LIMIT copies; :1927/:2106 comprehension/pattern locals):
   `bug_with_where_copy_id_reused_by_next_match` ([] vs [[0,1,0]]), `..._by_pattern_comprehension`,
   `bug_pattern_comprehension_locals_reused` (4 rows vs 3), `bug_where_comprehension_locals_reused`
   (runtime error "Invalid node id for 'to' in relationship pattern").
4. ~~CALL { UNION } branches start at scope 0~~ **Fixed by #3027** (d42d121c6): each branch's `Binder` now starts with
   `env_stack: vec![HashMap::new(); outer + 1]` (binder.rs:903), so its scopes are above the outer ones
   (`callBody_union_disjoint`, `union_branch_label_fixed`); the test `bug_call_union_branch_scopes_alias_outer`
   passes. Kept as a number so the other items keep theirs.
5. **ORDER BY copies cross the projection** (snapshot at binder.rs:1248 is after ORDER BY): `bug_order_by_variable_crosses_with`
   (`WITH n.v AS k ORDER BY n.v RETURN n` accepted, `RETURN *` returns `k, n`), `bug_order_by_variable_returned_from_call`
   ("Variable `n` already declared in outer scope"; C returns rows).
6. **Label accumulation leaks** from CREATE and pattern comprehensions into the earlier MATCH scan (binder.rs:1562→1360,
   1881-1938 lack the save/restore of :339-345 and :2096-2101): `bug_create_labels_leak_into_match` (0 vs 3),
   `bug_pattern_comprehension_labels_leak_into_match` (1 row vs 3).
7. **Pattern predicate in WITH … WHERE rebinds a parent name** (define_name_in_scope :2160 ignores the parent copy that
   pattern comprehensions pre-make at :1882): `bug_with_where_pattern_predicate_ignores_outer_variable` (3 rows vs 1).

8. **A subscript is never accepted as a boolean** (NEW, `expr_may_return_boolean` binder.rs:2347-2356 lists
   `GetElement` as non-boolean; also `Negate`, `Length`, `ListComprehension`, `MapProjection`, which can be null).
   Live Rust (release) vs C: `CREATE (:N {flags:[true,false]}), (:N {flags:[false]})` then
   `MATCH (n:N) WHERE n.flags[0] RETURN count(n)` → Rust `Expected boolean predicate`, C `1`;
   `WITH [true] AS l RETURN true AND l[0]` → Rust `Type mismatch: expected Boolean`, C `true`;
   `UNWIND [[true],[false]] AS l WITH l WHERE l[0] RETURN count(*)` → Rust error, C `1`;
   `RETURN true AND -null` → Rust error, C `null`. Lean `getElement_rejected`, `negate_rejected`
   (`mayBool_sound_except`: sound for every other variant). Fix: move `GetElement` (and the null-capable
   unary/list forms) to the "runtime-typed → true" arm.
9. **ORDER BY aggregate matched to the wrong projection** (NEW, `expr_ir_eq` binder.rs:2682 falls back to
   `discriminant` equality, so every two `Constant`s of different kinds — Int vs Float, Null vs Int,
   Bool vs Int, String vs Int — and any two quantifiers compare equal): `CREATE (:N {g:1, v:3}), (:N {g:2, v:4})`,
   `MATCH (n:N) RETURN n.g AS g, max(n.v % 3) AS m ORDER BY max(n.v % 4.0)` → Rust rows `[1,0],[2,1]`
   (sorted by `m`), C `failed to map aggregation expression within ORDER BY clause, please use alias`
   (openCypher order by `max(v % 4.0)` would be `[2,1],[1,0]`); with `% 4` (Int) Rust also rejects.
   Lean `order_by_wrong_aggregate`, `const_kinds_conflated`. Fix: compare `Constant` values with
   type-aware equality (`Value` equality, not discriminant) and compare quantifier kinds.

## Spec vs C disagreements (not Rust-vs-C bugs)
* `RETURN false AND (1+2)` / `WITH 2 AS x RETURN false AND x` / `false AND size([1])` → Rust `false`
  (arithmetic, variables and calls are "may be boolean" — `arith_accepted` — and AND short-circuits at
  runtime), C `Type mismatch` (static check). openCypher: type error.
* `MATCH (a)-[r*1..2]->(c) WITH r WHERE r.w = 1 RETURN count(*)` → Rust `Type mismatch: … but was Path`
  (the projected `r` is no longer in `varlen_rel_var_ids`, `prop_on_path`), C `1` (per-edge reading);
  without the WITH both return `1`. openCypher rejects `r.w` on a list.
* `date({year:2020, month:null})`, `duration({days:null})`, `localtime({hour:null})` (literal or map
  argument): Rust treats a null field as absent (`2020-01-01`, `PT0S`, `00:0:0`), C errors. Not caused by
  `rewrite_struct_constructor` (the map path agrees) — temporal functions' territory.
* Rust follows openCypher and C does not: `MATCH p=(a)-->(b) MATCH (p)` rejected; `reduce(a = 0, a IN …)` rejected;
  `RETURN [(n)-->(m)|m], m` rejected; within one MATCH `(a)-[r]->(), (a)-[s]->()` enforces r≠s.
* Both accept what openCypher rejects: `WITH 1 AS a MATCH (a)` (`ensureType_accepts_any_binding`); `WITH n.v AS k WHERE n.v > 0`.
* Rust over-accepts vs C: `RETURN n.v+1 AS k, count(*) ORDER BY n.v` (`agg_orderBy_by_name`); `(a)-[r {w: b.v}]->(b)` (`rel_attr_forward_node`).

## Open PRs touching binder.rs (model = origin/main)
* #2981: `bind_graph` defines named relationships before nodes' inline maps are bound and splits inline maps
  reading pending pattern variables (`split_inline_attrs`) — removes `node_attr_on_rel_rust_rejects` / bug 2.
* #3016: keeps hidden scope variables' ids reserved and hides ORDER BY copies — fixes `erase_breaks_covers`,
  `where_copy_id_reused`, `orderBy_leaks` (bugs 3, 5).
* #3049: CREATE and pattern-comprehension labels stay out of the MATCH label set — fixes `labels_leak` (bug 6).
* #3062: pattern predicates in WITH … WHERE resolve pre-projection names — fixes `defineName_ignores_parent` (bug 7).
* #3011: generalises `replace_agg_subtrees` to `replace_projected_subtrees` (non-aggregate ORDER BY
  expressions repeating a projection read its alias), guarded by `is_alias_rewritable`, which admits
  `Constant(Bool|Int|Float|String)` — exactly the kinds `expr_ir_eq` conflates with each other
  (`const_kinds_conflated`). With it, bug 9 would extend to plain expressions, e.g.
  `RETURN n.v % 3 AS k ORDER BY n.v % 4.0` (sorted by `k`). Fix bug 9 first (or compare constants by value).

## Gaps
Expressions are abstracted to the names they mention (BindExpr's `NE` keeps the variants the binder treats
specially); `bind_graph` is `defineName` per name (Pattern.lean has the node/relationship order); sub-binders
of `bind_ir` are parameters (`Subs`); the reserved-key formatting is a hypothesis (`KeysFresh`); function
contracts (`pure_fn` = runtime fn, `struct_fn` = map fn, compiled regex = runtime regex) are hypotheses.
The link from an id collision to a wrong answer is established by the Rust tests, not in Lean.
-/
import FalkorBinder.Table
import FalkorBinder.Resolve
import FalkorBinder.Projection
import FalkorBinder.Pattern
import FalkorBinder.Scopes
import FalkorBinder.RemoveLabels
import FalkorBinder.Stack
import FalkorBinder.BindExpr
import FalkorBinder.BindIR
import FalkorBinder.Types
import FalkorBinder.Fold
import FalkorBinder.AggEq
import FalkorBinder.CallBoundary
