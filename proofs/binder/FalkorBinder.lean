/-
# The binder's scopes: what a variable id refers to, and when ids collide

A model of `graph/src/planner/binder.rs` (and the id-minting side of
`graph/src/planner/mod.rs`) and a machine-checked account of its scoping.
`lake build` succeeds; no `sorry`, `admit` or `axiom`; every theorem depends at
most on `propext`, `Quot.sound`, `Classical.choice`. 238 theorems. Rust source: origin/main `256ab4d3f`
(re-targeted across #3335: `(x)` binds to `x`'s tree — `bind_paren` — and AND/OR bind flat via
`and_of`/`or_of`; #3303: unnamed entities live in the scope's own list and `lookup_anon` finds them;
#3270: a subscript may be a predicate; #3641: a UNION CALL body skips the outer label pass).

## The one invariant that matters

Every id is minted as the **current size of its scope table**: `fresh_var`
(binder.rs:2492), the parent copy in `resolve_name` (binder.rs:2434, minting at `len()` in the parent-copy arm), and the
planner's `fresh_var` (planner/mod.rs:1018, called at 1261, 1301, 1307 and 1649), which gets
`scope_vars[s] = sorted_scope_vars(env_stack[s])`. `mint_fresh_iff` proves the
planner's ids `len, len+1, …` avoid every id `H` the bound IR uses iff
`∀ i ∈ H, i < len`. Minting under a new key keeps that (`mint_new_covers`,
`resolve_covers`, `projectName_new_covers`, `foreachRestore_covers`,
`mergeSV_covers`). **Removing** a key breaks it, and the binder removes keys
in two places while the removed ids are still in the IR
(`erase_breaks_covers`, `where_copy_id_reused`, `comprehension_cleanup_breaks`).

## Gap closed 2026-10-20: pattern predicates may declare variables (#1334)
`bind_expr_node`'s `Pattern` arm (binder.rs:2270-2297) binds a WHERE pattern predicate with
`bind_graph(…, false)`, which mints any unknown alias fresh. Property `PatternPredDeclaresNothing`
(an accepted pattern predicate names only visible variables); refuted by
`bug1334_pattern_predicate_declares_var`: `MATCH (n0) WITH * WHERE () -- (n1) RETURN *` binds with
`n1` fresh (id 2) — live Rust returns every `n0`, C rejects it (issue #1334; the C error message is not recorded here).
The slot-reuse family (#3333, #3369, #3375, #3368, #3150) is modelled in proofs/ops_aggregate
`SlotReuse.lean` (`projLayout` = this binder's per-scope minting fed to the aggregate).

## Gap closed 2026-10-22: REMOVE does not clear accumulated labels (#3450)
`LClause` gains `.remove` (`bind_ir` QueryIR::Remove, binder.rs:599-604, leaves `node_labels`
alone); `bind_projection` copies the parent's label set to the `WITH` child (`node_labels.insert(bound_var.key(), labels)`, :1254-1310) and
`update_all_node_labels` (:220-252) stamps it on the next MATCH. Property `StampedLabelsHeld` (a stamped
label is written by that MATCH or still held, point-in-time via `held`); `specStamp_held` proves the fix
satisfies it, `bug3450_remove_label_kept_across_with` refutes it on main: `MATCH (a:A) REMOVE a:A WITH a
MATCH (a)-[:R]->(b)` stamps `(a:A)` — Rust 0 rows, C 1. (RemoveLabels.lean; the planner half is in
proofs/planner_build `RemoveVerified`; the issue's CREATE/MERGE half does not reproduce.)

## Lean ↔ Rust

Rust source: origin/main `256ab4d3f`; every line below is read off that tree.

| here | there |
| --- | --- |
| `Env`, `Env.insert/erase/retain/lookupAnon/pushUnnamed` | `Scope` (named map + unnamed list), binder.rs:46-131 (`get` :53, `insert` :67, `lookup_anon` :84, `push_unnamed` :94, `retain` :126) |
| `freshVar`, `freshAnon` | `Binder::fresh_var` binder.rs:2492-2506 |
| `plannerFresh`, `mint_fresh_iff` | `Planner::fresh_var` planner/mod.rs:1018-1030 |
| `St`, `pushScope` | `env_stack`, `use_parent_scope`, `parent_to_child_scope`, `copy_from_parent`; `push_scope` :349, `commit_scope` :354 |
| `resolve`, `resolveName`, `lookupLocals` | `resolve_name` binder.rs:2434-2490 |
| `defineName`, `defineAlias`, `lookupAlias`, `resolveAlias`, `denotes`, `ensureType` | `define_name_in_scope` :2347, `define_alias_in_scope` :2373, `lookup_alias` :2394, `resolve_alias` :2406, `denotes` :2935, `ensure_type` :2508 |
| `projectName` | `project_name` :2420 |
| `bindProjection`, `projectItems` | `bind_projection` :1208-1442 (non-`*` form) |
| `rustBind` / `specBind`, `defRels`, `Elem`, `split` | `bind_graph` :1480-1715 (named rels :1497, nodes :1522, then rels :1535) / C's textual order |
| `fold`, `bindClauses` | `parse_pattern` part separator parser/cypher.rs (#2923 clause fold) |
| `mergeSV`, `lenAt` | `merge_scope_vars` :2950, `Binder::bind` :197 |
| `foreachRestore` | FOREACH arm of `bind_ir` :810-835 |
| `compScope`/`compCleaned` | cleanup in `bind_expr_node` :2113 and :2291 |
| `callBodyScope`, `relabelBody` | `bind_call_body` :971-1126, stack push :992 (Query) vs UNION branch binder :1050; step 2b of `bind_call_subquery` :856-870 |
| `accumulate` / `accumulateFrom` / `specScanLabels` | `node_labels` accumulation (`bind_graph`), `with_node_labels_rolled_back` :1723, `update_all_node_labels` :220 |
| `held`, `rsStamp`, `specStamp` (RemoveLabels.lean) | REMOVE arm of `bind_ir`, WITH label copy in `bind_projection`, stamp `update_all_node_labels` :220 |
| `unionCheck` | UNION arm of `bind_ir` |
| `validateImport` | `validate_import_with` :1152 |
| `BSt`, `sortedScopeVars`, `updNode` (Stack.lean) | `env_stack` & co., `default` :171, `scope_vars` :212, `current_env(_mut)` :337/343, `push_scope` :349, `commit_scope` :354, `sorted_scope_vars` :2942, `update_graph_labels` :258 |
| `NE`/`BE`, `bindNode`, `bindSetItems`, `selfRef` (BindExpr.lean) | `bind_expr_node` :1935-2345, `bind_expr(_with_locals)` :1914/1921, `bind_set_items` :1889, `check_unbound_self_referential` :1867 |
| `RC`, `bindIR`, `shortestOk`, `mergeOk`, `yieldLoop`, `bindYield`, `bindCallSubquery`, `importProjections`, `createCheck` (BindIR.lean) | `bind_ir` :360-842, `bind_call_subquery` :844-938, `build_import_projections` :1132-1150, `bind_graph_create` :1742-1860 |
| `mayBool`, `mayEntity`, `validate` (Types.lean) | `expr_may_return_boolean` :2555-2620, `expr_may_return_entity` :2623-2688, `validate_boolean_operands(_impl)` :2529/2533 |
| `nodeToValue`, `collectConstArgs`, `foldCall`, `rewriteStruct`, `rewriteRegex` (Fold.lean) | `expr_tree_to_value` :2696, `collect_constant_args` :2728, `rewrite_struct_constructor` :2746, `rewrite_compiled_regex` :2811 |
| `exprIrEq`, `strip`/`eqS`, `structEq`, `replaceAgg` (AggEq.lean) | `expr_ir_eq` :2913, `nodes_eq` :2883, `raw_exprs_structurally_equal` :2875, `replace_agg_subtrees` :2845 |

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
  `comprehension_cleanup_breaks`, `orderBy_leaks`, `defineName_ignores_parent`,
  `rel_attr_forward_node`, `q2923_cause`, `fold_merges`,
  `pre3049_labels_leak` (fixed by #3049: `labels_leak_fixed`), `agg_orderBy_by_name`, `projectName_present_uncovered` (guarded, no bug).

* Wave 5 (all binder.rs fns PROVEN): `sortedScopeVars_spec`, `push_toSt`, `updNode_union`/`_idem`;
  `bindNode_covers` (pattern-free expressions keep the table covering), `loop_var_shadows`,
  `bind_iterable_outer`; per-arm `bindIR` theorems (`shortestOk_iff`, `mergeOk_*`, `yieldLoop_ok`,
  `callSubquery_covers`, `importProjections_covers`, `createCheck_*`); `mayBool_sound_except`, `mayEntity_eq`;
  `nodeToValue_sound`/`_complete`, `foldCall_sound`, `rewriteStruct_shape`, `rewriteRegex_spec`;
  `nodesEq_refl`/`_symm`, `exprIrEq_exact`, `replaceAgg_id`/`_root`.
* #2845 CALL entry projection (`CallBoundary`, folded in from the retired pr2845_review):
  `Pr2845.bodyNew_entry`, `importProj_spec`, `entry_ids_below`, `body_own_unbound`,
  `body_import_value`, `union_branch_label_fixed` (#3027), `pushed_read_unbound` (W6-opt-1).

## Confirmed bugs (live Rust vs C repros; the former `graph/tests/lean_binder` test file no longer exists)
1. **Clause fold** (general cause of #2923): `parse_pattern`'s part separator (parser/cypher.rs `parse_pattern`) folds
   `MATCH p MATCH q` (and `CREATE`, and `OPTIONAL MATCH p MATCH q`) into one pattern; cypher.rs:2117 parses a
   WHERE pattern predicate with the same loop, so it swallows a following MATCH. Combined with `bind_graph`
   binding nodes before relationships (binder.rs:1522 vs 1535; since #2981 named relationships are defined first, `defRels`): `bug_node_attrs_cannot_see_rel_of_previous_match`
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
3. **Removed-key id reuse** (binder.rs:1401-1409 WHERE/SKIP/LIMIT copies removed; :2087-2120 / :2273-2297 comprehension/pattern locals; **not reproduced live on 2026-11-22**: `MATCH (n:A) WITH n.v AS k WHERE n.v = 0 MATCH (x)-[]->(y) RETURN k, id(x), id(y)` gives C's `[0,0,1]` — the Lean collision is real at the binder level, the wrong-answer link is unconfirmed):
   `bug_with_where_copy_id_reused_by_next_match` ([] vs [[0,1,0]]), `..._by_pattern_comprehension`,
   `bug_pattern_comprehension_locals_reused` (4 rows vs 3), `bug_where_comprehension_locals_reused`
   (runtime error "Invalid node id for 'to' in relationship pattern").
4. ~~CALL { UNION } branches start at scope 0~~ **Fixed by #3027** (d42d121c6): each branch's `Binder` now starts with
   `env_stack: vec![HashMap::new(); outer + 1]` (binder.rs:1050), so its scopes are above the outer ones; since #3641 the outer binder also skips its label pass over a UNION body (`union_body_never_relabelled`)
   (`callBody_union_disjoint`, `union_branch_label_fixed`); the test `bug_call_union_branch_scopes_alias_outer`
   passes. Kept as a number so the other items keep theirs.
5. **ORDER BY copies cross the projection** (the `env_keys_before_filter` snapshot, binder.rs:1384-1409, is after ORDER BY; live 2026-11-22: `MATCH (n:N) WITH n.v AS k ORDER BY n.v RETURN *` → Rust columns `k, n`, C `k`): `bug_order_by_variable_crosses_with`
   (`WITH n.v AS k ORDER BY n.v RETURN n` accepted, `RETURN *` returns `k, n`), `bug_order_by_variable_returned_from_call`
   ("Variable `n` already declared in outer scope"; C returns rows).
6. ~~Label accumulation leaks~~ **Fixed by #3049** (375532b01): `with_node_labels_rolled_back` (binder.rs:1723) restores `node_labels` after a CREATE (keeping the labels of nodes the CREATE declares) and a pattern comprehension (`pre3049_labels_leak` → `labels_leak_fixed`, `accumulate_bound_match_only`, `create_new_keeps_labels`). Kept as a number so the other items keep theirs.
7. **Pattern predicate in WITH … WHERE rebinds a parent name** (`define_name_in_scope` :2347 ignores the parent copy that
   pattern comprehensions pre-make at :2068-2085; live 2026-11-22: `CREATE (:A {v:1})-[:R]->(:B), (:C {v:2})`, `MATCH (n) WITH n.v AS k WHERE (n)-->() RETURN k` → Rust `1, 2`, C `1`): `bug_with_where_pattern_predicate_ignores_outer_variable` (3 rows vs 1).

8. **Null-capable unary forms are never accepted as a boolean** (`expr_may_return_boolean` binder.rs:2555 lists
   `Negate`, `Length`, `ListComprehension`, `MapProjection` as non-boolean, though they can be null). The subscript half
   (`GetElement`) was **fixed by #3270** (3643a301c): `getElement_accepted`, `pre3270_getElement_rejected`; live Rust now
   returns `1` / `true` for `MATCH (n:N) WHERE n.flags[0] RETURN count(n)` and `WITH [true] AS l RETURN true AND l[0]`, as C does.
   Still open (live, Rust vs C, 2026-11-22): `RETURN true AND -null` → Rust `Type mismatch: expected Boolean`, C `null`.
   Lean `negate_rejected` (`mayBool_sound_except`: sound for every other variant). Fix: move the null-capable
   unary/list forms to the "runtime-typed → true" arm.
9. **ORDER BY aggregate matched to the wrong projection** (NEW, `expr_ir_eq` binder.rs:2913 falls back to
   `discriminant` equality, so every two `Constant`s of different kinds — Int vs Float, Null vs Int,
   Bool vs Int, String vs Int — and any two quantifiers compare equal): `CREATE (:N {g:1, v:3}), (:N {g:2, v:4})`,
   `MATCH (n:N) RETURN n.g AS g, max(n.v % 3) AS m ORDER BY max(n.v % 4.0)` → Rust rows `[1,0],[2,1]`
   (sorted by `m`; reconfirmed live 2026-11-22), C `failed to map aggregation expression within ORDER BY clause, please use alias`
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

## PRs touching binder.rs (#2981, #3049 merged; #3016, #3062, #3011 not in `256ab4d3f`)
* #2981 (merged, modelled): `bind_graph` defines named relationships before nodes' inline maps are bound (`defRels`)
  — `node_attr_on_rel_rust_accepts`, `node_attr_on_own_rel_accepts`, `q2923_fixed` replace
  `node_attr_on_rel_rust_rejects` / bug 2; a MATCH reusing an earlier MATCH's relationship variable is rejected
  (`match_rel_reused`, `pruneMatchRels_spec`).
* #3016: keeps hidden scope variables' ids reserved and hides ORDER BY copies — fixes `erase_breaks_covers`,
  `where_copy_id_reused`, `orderBy_leaks` (bugs 3, 5).
* #3049 (merged, modelled: `accumulateFrom`): CREATE and pattern-comprehension labels stay out of the MATCH label set — fixes bug 6 (`pre3049_labels_leak` → `labels_leak_fixed`, `accumulate_bound_match_only`).
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
