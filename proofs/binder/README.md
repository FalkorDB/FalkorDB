# binder: which variable a name refers to, and when two variables share an id

This project proves how the binder resolves variable names to numeric ids, and shows with concrete counterexamples where two different variables end up with the same id. The binder runs on every query between the parser and the planner. User-facing behaviour that depends on it:

- which variables a clause can see after `WITH`, `RETURN`, `ORDER BY`, `UNWIND`, `FOREACH` and `CALL {}`;
- whether `MATCH (a)-[r]->(b) MATCH (c {v: r.w})` binds at all;
- which errors you get: "'x' not defined", "already declared", "Type mismatch", "Expected boolean predicate";
- constant folding and the `ORDER BY` rewrite that maps an aggregate back to its alias.

The Rust code is `graph/src/planner/binder.rs`, plus the id-minting side of `graph/src/planner/mod.rs` and the clause loop in `graph/src/parser/cypher.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorBinder.lean`](FalkorBinder.lean). This README explains *what* is proven and *why it matters*.

## The one invariant

Every variable id is minted as the **current size of its scope table**. The binder's `fresh_var`, the parent copy in `resolve_name`, and the planner's own `fresh_var` all hand out `len()`, `len()+1`, …. That is collision-free only if every id the bound plan uses is below `len()`. In Lean, a table that satisfies this is `Env.Covers`.

- Adding a name keeps it (`mint_new_covers`, `resolve_covers`, `projectName_new_covers`, `foreachRestore_covers`, `mergeSV_covers`).
- **Hiding** a name keeps it: since #3062 the binder never removes a name whose id the plan may still use; it moves the entry to the unnamed list (`hide_names`), which still counts toward `len()` (`hide_covers`, `cleanup_covers`, `hide_keeps_where_copy`, `where_copy_id_fresh`, `comprehension_cleanup_keeps_ids`).
- A second name `import_parent_names` enters for a renamed parent variable takes no id: `Scope::len` subtracts the `aliased` count (`Env.withAliases_len`), and `drop_aliases` restores the scope (`Env.dropAliases_withAliases`).

## The questions this project answers

**1. When is id minting collision-free?**

`mint_fresh_iff` proves the planner's ids `len, len+1, …` avoid every id the bound plan uses **if and only if** every used id is below `len`. So `Env.Covers` on the table is not enough by itself: an id that was removed from the table but is still read by a Filter breaks the condition.

- **Yes, since #3062.** After `MATCH (n) WITH n.v AS k WHERE n.v = 0`, the WITH scope is `{k ↦ 0, n ↦ 1}` (`n` is copied for the WHERE), then `n` is hidden, not removed. The next `MATCH (x)-->(y)` gives `x` id 2, not the slot 1 the Filter reads (`hide_keeps_where_copy`, `where_copy_id_fresh`). Comprehension and pattern-predicate locals are hidden the same way (`comprehension_cleanup_keeps_ids`, `cleanup_covers`).

**2. Does `resolve_name` find exactly the names that are in scope?**

- **Yes.** `resolve_ok_iff` proves `resolve_name` succeeds exactly on the names the spec makes visible: comprehension locals, the current clause's scope, and, only while a projection's ORDER BY / WHERE / SKIP / LIMIT is bound, the scope before the projection.
- `resolve_innermost` proves it returns the innermost binding: a comprehension local shadows the clause scope, which shadows the pre-projection scope.

**3. Does a `WITH`/`RETURN` scope contain exactly its aliases?**

- **Without ORDER BY, yes.** `projectItems_keys` proves a projection scope starts with exactly its aliases, in order; `with_plain` and `where_copy_removed` check the common cases.
- **With ORDER BY, no.** In `MATCH (n) WITH n.v AS k ORDER BY n.v RETURN n`, the ORDER BY copy of `n` survives into the next clause, so the query is accepted and `RETURN *` returns `k, n` (`orderBy_leaks`). openCypher and C give the scope `[k]`.

**4. Do consecutive clauses bind in order?**

openCypher binds a pattern left to right, clause by clause.

- `agree_on_outer_refs` proves the binder's order (all nodes, then all relationships) and textual order accept the same patterns when every inline property map only mentions variables from earlier clauses.
- **Clauses are not folded.** `MATCH p MATCH q`, `CREATE … CREATE …` and `OPTIONAL MATCH p MATCH q` stay separate, so `MATCH (a)-[r:R]->(b) MATCH (c {v: r.w})` binds (`q2923_fixed`, `match_rel_reused`). (Issue #2923.)

**5. Are the type checks right?**

- `ensureType_table` proves the kind conflicts rejected are exactly a node, relationship or path used as one of the others.
- `mayBool_sound_except` proves `expr_may_return_boolean` says "maybe" for every expression that can produce a boolean or null, **except** a few listed shapes: `true AND -null` is rejected where C returns `null` (`negate_rejected`). A subscript (`WHERE n.flags[0]`) is accepted (`getElement_accepted`).
- `mayEntity_eq` proves `expr_may_return_entity` matches its spec exactly.

**6. Does constant folding preserve meaning?**

- **Yes.** `nodeToValue_sound` and `nodeToValue_complete` prove that turning a constant expression tree into a value gives exactly what evaluating it would, and succeeds on every pure tree. `foldCall_sound` proves folding a function call at bind time gives the value the runtime call would, given the registration contract that the function's bind-time and run-time versions agree.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `mint_fresh_iff` | Length-based minting is collision-free iff every used id is below the table length | `Table.lean` |
| `mint_new_covers`, `fresh_not_used` | Minting under a new name keeps the table covering, and the new id is unused | `Table.lean` |
| `resolve_ok_iff`, `resolve_innermost` | `resolve_name` succeeds exactly on visible names and returns the innermost binding | `Resolve.lean` |
| `projectItems_keys` | A projection scope begins with exactly its aliases, in order | `Projection.lean` |
| `agree_on_outer_refs` | Nodes-first and textual binding agree when inline maps only mention earlier variables | `Pattern.lean` |
| `mergeSV_covers`, `foreachRestore_covers` | Merged scope tables and the FOREACH restore keep every inner id reserved | `Scopes.lean` |
| `callBody_query_disjoint` | A plain `CALL {}` body's scopes are disjoint from the live outer scopes | `Scopes.lean` |
| `unionCheck_ok_iff`, `validateImport_ok` | The UNION column rule and the CALL import rule hold as stated | `Scopes.lean` |
| `bindNode_covers` | Binding a pattern-free expression keeps the table covering | `BindExpr.lean` |
| `bind_paren`, `bind_andOr_paren` | `(x)` binds exactly like `x`, and nested AND/OR bind flat | `BindExpr.lean` |
| `mayBool_sound_except`, `mayEntity_eq` | The boolean and entity type checks, and their exact exceptions | `Types.lean` |
| `nodeToValue_sound`, `foldCall_sound` | Constant folding gives the same value as evaluation | `Fold.lean` |
| `exprIrEq_exact`, `exprIrEq_exact_any_kind` | `expr_ir_eq` is exact on expressions it can compare by value; constants of different kinds never match | `AggEq.lean` |
| `order_by_float_rejected`, `order_by_float_equal` | `ORDER BY max(n.v % 4.0)` does not match `max(n.v % 3)`; `% 3.0` still does | `AggEq.lean` |
| `importProj_spec`, `entry_ids_below`, `body_import_value` | A `CALL {}` body's entry projection imports exactly the listed variables, below the body's own ids | `CallBoundary.lean` |

### How `mint_fresh_iff` and the covering lemmas fit together

1. **Model the table.** `Env` is a list of `(name, Var)` pairs standing for the Rust `HashMap<Arc<String>, Variable>`. `Env.insert`, `Env.erase` and `Env.hide` (`hide_names`: remove, then push to the unnamed list) are the mutations; the second names of `import_parent_names` are kept beside the table (`RScope`, `Env.withAliases`). `freshVar` is `Binder::fresh_var`: the new id is `e.length`.
2. **State the invariant.** `Env.Covers e` says every id in `e` is below `e.length`. `fresh_not_used` follows directly: `e.length` is not among the ids.
3. **Check every mutation.** Each place the binder inserts under a new key has a `_covers` theorem. Re-inserting an existing key with a smaller id is `reinsert_covers`.
4. **Link to the planner.** `mint_fresh_iff` is pure arithmetic: ids `len + j` miss a set `H` for every `j` exactly when every element of `H` is below `len`. The binder bug W2-binder-3 (fixed by #3062) was the case where `H` (ids the plan still reads) contained an id that a removal dropped from the table, so it was no longer below `len`; `hide` keeps the length and the ids (`Env.hide_length`, `Env.hide_ids`).
5. **Hypotheses.** `bindNode_covers` assumes `KeysFresh`: the reserved internal names (`_quant_`, `_lc_`, `_reduce_`, …) never clash with a name already in the table. This is a property of how the Rust formats those names, stated rather than proven.

## Bugs found

Each bug has a Rust repro (`cargo test -p graph --test lean_binder`, not committed; the tests assert C's answer and fail while the bug is Open) and was compared against C (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W2-binder-5 | `orderBy_leaks` | ORDER BY copies leak past WITH/RETURN: a spurious "already declared", and `RETURN *` returns ORDER BY-only variables | **Open** (PR #3016 not merged) |
| W4-plan-2 | `negate_rejected` | `RETURN true AND -null` → Rust "Type mismatch", C `null`; `Length`, `ListComprehension`, `MapProjection` are rejected the same way | **Open** |
| #1334 | `bug1334_pattern_predicate_declares_var` (property `PatternPredDeclaresNothing`) | A WHERE pattern predicate may mint a new named variable: `MATCH (n0) WITH * WHERE () -- (n1) RETURN *` binds and returns every `n0`; C rejects it (issue #1334; the C error message is not recorded here) | **Open** |
| #3450 | `bug3450_remove_label_kept_across_with` (property `StampedLabelsHeld`, `RemoveLabels.lean`) | `REMOVE a:A` leaves `A` in the accumulated label set, which `WITH a` copies and `update_all_node_labels` stamps on the next MATCH: `MATCH (a:A) REMOVE a:A WITH a MATCH (a)-[:R]->(b)` gives 0 rows, C 1. The fixed stamp `specStamp` satisfies the property (`specStamp_held`). The issue's CREATE/MERGE back-propagation half does not reproduce | **Open** |

**Gap closed 2026-10-22.** `StampedLabelsHeld` (in `RemoveLabels.lean`) states that a label stamped on a MATCH node after `WITH` is one that MATCH wrote or one the node still carries (`held`: REMOVE clears); `bug3450_remove_label_kept_across_with` refutes it. The planner's half (`verified_labels`) is in [`../planner_build`](../planner_build).

**Gap closed 2026-10-20.** `PatternPredDeclaresNothing` (in `BindExpr.lean`) states that an accepted pattern predicate names only variables that are already visible; `bug1334_pattern_predicate_declares_var` refutes it on main. The slot-reuse family (#3333, #3369, #3375, #3368) comes from this binder's per-scope numbering, but it is modelled and refuted in [`../ops_aggregate`](../ops_aggregate) (`SlotReuse.lean`), where the colliding ids are consumed.

Smaller over-acceptances, not filed as bugs: `RETURN n.v+1 AS k, count(*) ORDER BY n.v` is accepted where C rejects (`agg_orderBy_by_name`), and so is `(a)-[r {w: b.v}]->(b)` (`rel_attr_forward_node`). `projectName_present_uncovered` shows a case where the invariant could break but a guard in the Rust prevents it; no bug.

There are no CONDITIONAL rows in this project.

## Files

| File | What's in it |
|---|---|
| `FalkorBinder.lean` | The report header and imports |
| `Table.lean` | The scope table `Env`, `Env.Covers`, `freshVar`, `mint_fresh_iff`, `Env.hide` (`hide_names`), `RScope` (the `aliased` counter: `len`, `insert_alias`, `remove_alias`, `drop_aliases`) |
| `Resolve.lean` | The binder state `St`, `resolve` (`resolve_name`), `visible`, `ensureType`, `defineName`, `projectName` |
| `Projection.lean` | `bindProjection` (WITH/RETURN), `projectItems`, the ORDER BY counterexample, the WHERE-copy fix |
| `Pattern.lean` | Nodes-first `rustBind` vs textual `specBind` |
| `Scopes.lean` | `mergeSV`, `foreachRestore`, comprehension cleanup, `callBodyScope`, label accumulation, `unionCheck`, `validateImport` |
| `RemoveLabels.lean` | #3450: `held` (point-in-time label set, REMOVE clears), `rsStamp`/`specStamp`, property `StampedLabelsHeld` |
| `Stack.lean` | The scope stack `BSt`, `sortedScopeVars`, label updates (`updNode`) |
| `BindExpr.lean` | Expression binding (`NE`, `BE`, `bindNode`), `importParent` (`import_parent_names`) and `cleanup` (`hide_new_names`) around patterns (`pattern_pred_uses_parent`, `pattern_pred_renamed_parent`: W2-binder-7 fixed by #3062), comprehension locals, AND/OR flattening, `bindSetItems`, self-reference checks |
| `BindIR.lean` | Per-clause rules of `bind_ir` (`RC`, `bindIR`): shortestPath, MERGE, YIELD, `CALL {}` subqueries, CREATE |
| `Types.lean` | `mayBool`, `mayEntity`, `validate` |
| `Fold.lean` | Constant folding (`nodeToValue`, `foldCall`), struct-constructor and regex rewrites |
| `AggEq.lean` | `expr_ir_eq` (`exprIrEq`), structural equality, `replaceAgg` |
| `CallBoundary.lean` | The `CALL {}` entry projection (namespace `Pr2845`) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 44 rows: 43 for functions of `binder.rs` and 1 (`Planner::fresh_var`) for `planner/mod.rs`:

| Bucket | Rows |
|---|---|
| PROVEN | 44 |

Every function in `binder.rs` (and `Planner::fresh_var`) is PROVEN. "PROVEN" here means the function's model has theorems stating what it does; for the functions involved in the bugs above, those theorems include the counterexamples.

## Spec vs C disagreements (not Rust-vs-C bugs)

- `RETURN false AND (1+2)`, `WITH 2 AS x RETURN false AND x`, `false AND size([1])`: Rust `false` (arithmetic, variables and calls are "may be boolean", `arith_accepted`, and AND short-circuits at runtime), C `Type mismatch` (static check). openCypher: type error.
- `MATCH (a)-[r*1..2]->(c) WITH r WHERE r.w = 1 RETURN count(*)`: Rust `Type mismatch: … but was Path` (the projected `r` is no longer in `varlen_rel_var_ids`), C `1`; without the WITH both return `1`. openCypher rejects `r.w` on a list.
- `date({year:2020, month:null})`, `duration({days:null})`, `localtime({hour:null})`: Rust treats a null field as absent, C errors. Not caused by `rewrite_struct_constructor` (the map path agrees).
- Rust follows openCypher and C does not: `MATCH p=(a)-->(b) MATCH (p)` rejected; `reduce(a = 0, a IN …)` rejected; `RETURN [(n)-->(m)|m], m` rejected; within one MATCH `(a)-[r]->(), (a)-[s]->()` enforces r≠s.
- Both accept what openCypher rejects: `WITH 1 AS a MATCH (a)` (`ensureType_accepts_any_binding`), `WITH n.v AS k WHERE n.v > 0`.
- #3011 (not in `256ab4d3f`) generalises `replace_agg_subtrees` to `replace_projected_subtrees`, guarded by `is_alias_rewritable`, which admits `Constant(Bool|Int|Float|String)`; `expr_ir_eq` compares constants by value and kind (`const_kinds_distinguished`).

## What the model does not cover

- Expressions are abstracted to the names they mention. `NE` keeps only the expression variants the binder treats specially (comprehensions, quantifiers, reduce, pattern predicates, property access).
- `bind_graph` is modelled as `defineName` per name; `Pattern.lean` has the node/relationship order. The sub-binders that `bind_ir` calls are parameters (`Subs`), each proven in its own file.
- The reserved-name formatting is a hypothesis (`KeysFresh`).
- Function contracts are hypotheses: a function marked pure behaves the same at bind time and at run time; a struct constructor is the map function; a compiled regex behaves like the runtime regex.
- The step from "two variables share an id" to "wrong answer" is shown by the Rust tests, not in Lean.

## Build and check

```bash
(cd proofs/binder && lake build)         # this project
bash proofs/lean_ci.sh binder          # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `binder.rs`, the id minting in `planner/mod.rs`, or the clause loop in `parser/cypher.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. A PR that fixes one of the bugs above should replace its counterexample with the matching correctness theorem and note the fix in prose (no `pre<PR>_` history theorem).
