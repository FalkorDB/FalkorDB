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
- **Removing** a name breaks it, and the binder removes names in two places while their ids are still used by the plan (`erase_breaks_covers`, `where_copy_id_reused`, `comprehension_cleanup_breaks`). The next variable then gets an id that is already taken, and two variables share one row slot.

## The questions this project answers

**1. When is id minting collision-free?**

`mint_fresh_iff` proves the planner's ids `len, len+1, …` avoid every id the bound plan uses **if and only if** every used id is below `len`. So `Env.Covers` on the table is not enough by itself: an id that was removed from the table but is still read by a Filter breaks the condition.

- **On main, no, in three places.** After `MATCH (n) WITH n.v AS k WHERE n.v = 0`, the WITH scope is `{k ↦ 0, n ↦ 1}` (`n` is copied for the WHERE), then `n` is removed. The next `MATCH (x)-->(y)` gives `x` id 1, the same slot the Filter reads (`erase_breaks_covers`, `where_copy_id_reused`). Comprehension and pattern-comprehension locals are cleaned up the same way (`comprehension_cleanup_breaks`).

**2. Does `resolve_name` find exactly the names that are in scope?**

- **Yes.** `resolve_ok_iff` proves `resolve_name` succeeds exactly on the names the spec makes visible: comprehension locals, the current clause's scope, and, only while a projection's ORDER BY / WHERE / SKIP / LIMIT is bound, the scope before the projection.
- `resolve_innermost` proves it returns the innermost binding: a comprehension local shadows the clause scope, which shadows the pre-projection scope.

**3. Does a `WITH`/`RETURN` scope contain exactly its aliases?**

- **Without ORDER BY, yes.** `projectItems_keys` proves a projection scope starts with exactly its aliases, in order; `with_plain` and `where_copy_removed` check the common cases.
- **With ORDER BY, no.** In `MATCH (n) WITH n.v AS k ORDER BY n.v RETURN n`, the ORDER BY copy of `n` survives into the next clause, so the query is accepted and `RETURN *` returns `k, n` (`orderBy_leaks`). openCypher and C give the scope `[k]`.

**4. Do consecutive clauses bind in order?**

openCypher binds a pattern left to right, clause by clause.

- `agree_on_outer_refs` proves the binder's order (all nodes, then all relationships) and textual order accept the same patterns when every inline property map only mentions variables from earlier clauses.
- **But the parser folds clauses.** `MATCH p MATCH q`, `CREATE … CREATE …` and `OPTIONAL MATCH p MATCH q` become one pattern (`fold_merges`). Combined with nodes-first binding, `MATCH (a)-[r:R]->(b) MATCH (c {v: r.w})` fails with "'r' not defined" (`q2923_cause`, issue #2923). Each half alone is harmless: with a `WITH` between the clauses the query binds.

**5. Are the type checks right?**

- `ensureType_table` proves the kind conflicts rejected are exactly a node, relationship or path used as one of the others.
- `mayBool_sound_except` proves `expr_may_return_boolean` says "maybe" for every expression that can produce a boolean or null, **except** a few listed shapes: `WHERE n.flags[0]` is rejected with "Expected boolean predicate" where C returns rows (`getElement_rejected`, `negate_rejected`).
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
| `bind_paren`, `bind_andOr_paren` | `(x)` binds exactly like `x`, and nested AND/OR bind flat (since #3335) | `BindExpr.lean` |
| `mayBool_sound_except`, `mayEntity_eq` | The boolean and entity type checks, and their exact exceptions | `Types.lean` |
| `nodeToValue_sound`, `foldCall_sound` | Constant folding gives the same value as evaluation | `Fold.lean` |
| `exprIrEq_exact` | `expr_ir_eq` is exact on expressions it can compare by value | `AggEq.lean` |
| `importProj_spec`, `entry_ids_below`, `body_import_value` | A `CALL {}` body's entry projection (#2845) imports exactly the listed variables, below the body's own ids | `CallBoundary.lean` |

### How `mint_fresh_iff` and the covering lemmas fit together

1. **Model the table.** `Env` is a list of `(name, Var)` pairs standing for the Rust `HashMap<Arc<String>, Variable>`. `Env.insert`, `Env.erase` and `Env.retain` are the three mutations. `freshVar` is `Binder::fresh_var`: the new id is `e.length`.
2. **State the invariant.** `Env.Covers e` says every id in `e` is below `e.length`. `fresh_not_used` follows directly: `e.length` is not among the ids.
3. **Check every mutation.** Each place the binder inserts under a new key has a `_covers` theorem. Re-inserting an existing key with a smaller id is `reinsert_covers`.
4. **Link to the planner.** `mint_fresh_iff` is pure arithmetic: ids `len + j` miss a set `H` for every `j` exactly when every element of `H` is below `len`. The binder bugs are the cases where `H` (ids the plan still reads) contains an id that `erase` or `retain` dropped from the table, so it is no longer below `len`.
5. **Hypotheses.** `bindNode_covers` assumes `KeysFresh`: the reserved internal names (`_quant_`, `_lc_`, `_reduce_`, …) never clash with a name already in the table. This is a property of how the Rust formats those names, stated rather than proven.

## Bugs found

Each bug has a Rust repro (`cargo test -p graph --test lean_binder`, not committed; the tests assert C's answer and fail today) and was compared against C (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W2-binder-1 | `q2923_cause`, `fold_merges`, `node_attr_on_rel_rust_rejects` | Consecutive MATCH/CREATE clauses are folded into one pattern and bound nodes first: "'a' not defined" for consecutive CREATEs; relationship uniqueness across separate MATCHes (0 rows vs C 1); a MATCH after OPTIONAL MATCH becomes optional | **Open** (#2923; PR #2981, not merged, fixes the in-pattern case `node_attr_on_rel_rust_rejects`) |
| W2-binder-2 | (planner side; see the header) | `MATCH (a:A), (b:B {v: a.v})` filters `b`'s inline map on `b`'s own scan: Rust `[]`, C 2 rows. The node case of #2923 | **Open** (#2923; PR #2981 not merged) |
| W2-binder-3 | `erase_breaks_covers`, `where_copy_id_reused`, `comprehension_cleanup_breaks` | A removed WHERE copy or comprehension local frees an id still in use; the next variable reuses it. Wrong rows, extra rows, or "Invalid node id for 'to'" | **Open** (PR #3016 not merged) |
| W2-binder-4 | `callBody_union_aliases`, `union_branch_label_alias` | UNION branches inside `CALL {}` start at scope 0, shared with the outer query | **Open** (PR #3027 not merged; `union_branch_label_3027` checks its fix) |
| W2-binder-5 | `orderBy_leaks` | ORDER BY copies leak past WITH/RETURN: a spurious "already declared", and `RETURN *` returns ORDER BY-only variables | **Open** (PR #3016 not merged) |
| W2-binder-6 | `labels_leak` | Labels from CREATE and from pattern comprehensions tighten an earlier MATCH scan: 0 or 1 rows where C returns 3 | **Open** (PR #3049 not merged) |
| W2-binder-7 | `defineName_ignores_parent` | A pattern predicate in `WITH … WHERE` rebinds a parent variable as a fresh one: 3 rows where C returns 1 | **Open** (PR #3062 not merged) |
| W4-plan-2 | `getElement_rejected`, `negate_rejected` | `WHERE n.flags[0]` → "Expected boolean predicate" (C: 1); same for `true AND l[0]` and `true AND -null` | **Open** |
| W4-plan-3 | `order_by_wrong_aggregate`, `const_kinds_conflated` | `expr_ir_eq` treats all Int/Float/Null/Bool/String constants as equal, so `ORDER BY max(n.v % 4.0)` sorts by the alias of `max(n.v % 3)`; C rejects the query | **Open**; PR #3011 would widen it to non-aggregate ORDER BY, so fix this first |
| W6-opt-1 | `pushed_read_unbound` | Filter push-down gives an OPTIONAL MATCH inside `CALL {}` the outer query's ids: "Variable n not found" where C returns rows | **Open** |

Smaller over-acceptances, not filed as bugs: `RETURN n.v+1 AS k, count(*) ORDER BY n.v` is accepted where C rejects (`agg_orderBy_by_name`), and so is `(a)-[r {w: b.v}]->(b)` (`rel_attr_forward_node`). `projectName_present_uncovered` shows a case where the invariant could break but a guard in the Rust prevents it; no bug.

There are no CONDITIONAL rows in this project.

## Files

| File | What's in it |
|---|---|
| `FalkorBinder.lean` | The report header and imports |
| `Table.lean` | The scope table `Env`, `Env.Covers`, `freshVar`, `mint_fresh_iff`, `erase_breaks_covers` |
| `Resolve.lean` | The binder state `St`, `resolve` (`resolve_name`), `visible`, `ensureType`, `defineName`, `projectName` |
| `Projection.lean` | `bindProjection` (WITH/RETURN), `projectItems`, the ORDER BY and WHERE-copy counterexamples |
| `Pattern.lean` | Nodes-first `rustBind` vs textual `specBind`, the clause `fold`, `q2923_cause` |
| `Scopes.lean` | `mergeSV`, `foreachRestore`, comprehension cleanup, `callBodyScope`, label accumulation, `unionCheck`, `validateImport` |
| `Stack.lean` | The scope stack `BSt`, `sortedScopeVars`, label updates (`updNode`) |
| `BindExpr.lean` | Expression binding (`NE`, `BE`, `bindNode`), comprehension locals, AND/OR flattening, `bindSetItems`, self-reference checks |
| `BindIR.lean` | Per-clause rules of `bind_ir` (`RC`, `bindIR`): shortestPath, MERGE, YIELD, `CALL {}` subqueries, CREATE |
| `Types.lean` | `mayBool`, `mayEntity`, `validate` |
| `Fold.lean` | Constant folding (`nodeToValue`, `foldCall`), struct-constructor and regex rewrites |
| `AggEq.lean` | `expr_ir_eq` (`exprIrEq`), structural equality, `replaceAgg` |
| `CallBoundary.lean` | The `CALL {}` entry projection added by #2845 (namespace `Pr2845`) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 44 rows, all of `binder.rs`:

| Bucket | Rows |
|---|---|
| PROVEN | 44 |

Every function in `binder.rs` is PROVEN. "PROVEN" here means the function's model has theorems stating what it does; for the functions involved in the bugs above, those theorems include the counterexamples.

## What the model does not cover

- Expressions are abstracted to the names they mention. `NE` keeps only the expression variants the binder treats specially (comprehensions, quantifiers, reduce, pattern predicates, property access).
- `bind_graph` is modelled as `defineName` per name; `Pattern.lean` has the node/relationship order. The sub-binders that `bind_ir` calls are parameters (`Subs`), each proven in its own file.
- The reserved-name formatting is a hypothesis (`KeysFresh`).
- Function contracts are hypotheses: a function marked pure behaves the same at bind time and at run time; a struct constructor is the map function; a compiled regex behaves like the runtime regex.
- The step from "two variables share an id" to "wrong answer" is shown by the Rust tests, not in Lean.

## Build and check

```bash
cd proofs/binder && lake build         # this project
bash proofs/lean_ci.sh binder          # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `binder.rs`, the id minting in `planner/mod.rs`, or the clause loop in `parser/cypher.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. A PR that fixes one of the bugs above should turn its counterexample into a `pre<PR>_` history theorem and add the matching correctness theorem.
