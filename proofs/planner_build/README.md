# planner_build: does the plan the planner builds mean what the query says?

This project proves that the execution plan `Planner::plan` builds for a query computes what openCypher says the query means, clause by clause. It also gives concrete counterexamples where the plan differs. Every query goes through it. User-facing behaviour that depends on it:

- the order clauses run in, and which rows flow from one clause into the next;
- `WHERE` with pattern predicates (`WHERE (a)-->()`, `NOT (… OR (a)-->())`), planned as SemiApply / AntiSemiApply;
- `MERGE`, `CREATE`, `FOREACH`, `CALL {}`, `UNION`, `WITH`/`RETURN` with aggregation, DISTINCT, ORDER BY, SKIP, LIMIT;
- `MATCH` patterns: hop order, self-loops, inline property maps such as `(a {v: 1})`.

The Rust code is `graph/src/planner/mod.rs`, `graph/src/planner/hop_order.rs` and the query-graph and validation code in `graph/src/parser/ast.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`PlannerBuild.lean`](PlannerBuild.lean). This README explains *what* is proven and *why it matters*.

## What "correct" means here

openCypher defines a query as a sequence of clauses. Each clause maps a bag of records (the driving table), plus the graph state, to a new bag of records. `MATCH` extends each record by its matches; `WITH` projects; `UNWIND` expands; and so on.

The model gives the plan tree its own meaning, `ev`, following how the batched runtime evaluates the IR: child 0 is the input, and Cartesian-product right branches are run on the argument row. A plan is correct if `ev` of the plan equals the openCypher composition of its clauses.

## The questions this project answers

**1. Does stitching the clause plans together give the clause sequence?**

The planner plans each clause on its own, then stitches the plans in reverse: the last clause's plan is the root, and each earlier plan is inserted at an *insertion point* found by walking down the plan built so far.

- **Yes.** `rustStitch_eq_nestStitch` proves the Rust stitching loop builds exactly the nested fill `pₙ[pₙ₋₁[…p₁]]`, each plan at the next plan's own insertion point.
- `query_correct` proves that if each clause plan implements its clause at its insertion point, the stitched plan evaluates to the openCypher composition `Fₙ ∘ … ∘ F₂ ∘ ⟦c₁⟧`.

**2. Does each clause's plan implement that clause?**

- **For every clause kind, yes, in the positions the proof covers.** Each theorem holds for *any* input plan placed at the clause's insertion point. The clause theorems are `unwind_correct`, `create_correct`, `set_correct`, `remove_correct`, `delete_correct`, `proc_correct`, `project_correct`, `foreach_correct`, `call_correct`, `unit_call_correct`, `optional_correct`, `optional_apply_correct`, `match_bound_correct`, `union_correct`, `merge_loop_correct`, `merge_first_correct`, `create_loop_correct` and `create_first_correct`.
- **MERGE or CREATE with a named path as the last clause, no.** The first insertion-point walk does not step over `PathBuilder`, so `Merge`/`Create` never runs. `MATCH (a:A) MERGE p=(a)-[:T]->(x:X)` fails with "Variable _anon_0 not found"; C creates the edge (`merge_path_last_misplaced`, `create_path_last_misplaced`). Adding `RETURN …` after it works.

**3. Is a `MATCH … WHERE` correlated with the next `MATCH`?**

- **When the next MATCH is one connected pattern, yes.**
- **When it has several components, no.** The previous clause is prepended to the Cartesian product as a plain branch, so the component that mentions its variable re-binds it instead of reading it. `MATCH (a) WHERE a.v = 0 MATCH (a)-[:R]->(b), (c:C) RETURN a.v, b.v, c.v` → Rust `[1,0,7]`, C `[]` (`cp_stitch_loses_correlation`). `cp_stitch_sound` proves the prepend is right when no other component reads the variables.

**4. Is `WHERE` with pattern predicates planned correctly?**

- **Rebuilding the predicate is exact.** `collect_value` and `collect_passes` prove that replacing each pattern with a placeholder variable keeps the three-valued truth value.
- **With two-valued atoms, the operator tree is exact.** `toPlan_correct` proves SemiApply / AntiSemiApply / OrApplyMultiplexer / Filter keep exactly the rows where the predicate is `true`, given that the non-pattern atoms are never null.
- **With a null atom under `NOT`, no.** `NOT c` is planned as an anti-join against the rows where `c` is *true*, so it keeps rows where `c` is false **or null**. `MATCH (a:N) WHERE NOT (a.v > 1 OR (a)-->()) RETURN count(a)` → Rust 2, C 1 (`toPlan_not_null_bug`).
- Since #3335 AND/OR are built flat through `ExprIR::and_of`/`or_of` (`tv_andOf`, `tv_orOf`).

**5. Are `MATCH` patterns planned with the right hops?**

- `sortRels_spec`: hops are sorted fixed-length, then variable-length, then shortest-path, as a stable permutation. `planComp_binds`: each component binds its variables.
- **Self-loops, fixed by #3102.** Before it, `MATCH (a)-[:R]->(b)-[:S*]->(c)-[:L]->(c)` dropped the first hop (`[null,2,3]`; C `[1,2,3]`). Now `self_loops_after_binders` places a self-loop after the hop that binds its node: `selfLoops_perm`, `deferred_after_binder`, and `eq_reference` (equals the pre-#3102 loop whenever a self-loop really has `from = to`).
- **Inline property maps, since #2390, are lowered once.** `lowerInline_once` proves there is one Filter per (alias, map). `planMatch_stripped` proves no scan or fixed-length operator carries an inline map afterwards; [`../optimizer_rewrites`](../optimizer_rewrites) relies on that.

**6. What does validation let through?**

- **Too much after a procedure CALL — fixed by #3297.** `inner_validate` used to return `Ok(())` at a `CALL` without checking the clauses after it, so `CALL db.labels() YIELD label CREATE (a)-[:R|S]->(b)` created an edge with two types. On main b84473a7f the clauses after a CALL are validated (`call_validates_rest`, `call_then_match_rejected`, `call_then_unwind_rejected`, `call_then_create_multi_type_rejected`); history `pre3297_call_skips_rest`, `pre3297_call_then_match_accepted`.
- **A final `WITH`.** `MATCH (n) WITH n` returns rows; C and openCypher reject it (`with_last_accepted`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `rustStitch_eq_nestStitch` | The Rust stitching loop builds the nested fill of the clause plans | `Stitch.lean` |
| `query_correct` | If every clause plan implements its clause, the whole plan evaluates to the openCypher clause sequence | `Match.lean` |
| `project_correct` | WITH/RETURN (projection or aggregation, DISTINCT, ORDER BY, SKIP, LIMIT, WHERE) is correct at its slot | `Clauses.lean` |
| `merge_loop_correct`, `create_loop_correct` | MERGE and CREATE, with or without a named path, are correct when not last | `Clauses.lean` |
| `union_correct`, `call_correct`, `foreach_correct` | UNION / UNION ALL, `CALL {}` and FOREACH are correct | `Clauses.lean` |
| `cp_stitch_sound` | Prepending a MATCH to a Cartesian product is right when no other component reads its variables | `Match.lean` |
| `toPlan_correct` | The SemiApply/AntiSemiApply/OrApplyMultiplexer/Filter tree keeps exactly the rows where WHERE is true (two-valued atoms) | `ToPlan.lean` |
| `planFilter_correct` | `plan_filter` end to end, for a WHERE without pattern comprehensions | `Filter.lean` |
| `extract_fresh`, `hoist_spec` | Pattern-comprehension extraction mints fresh variables; hoisting is sound exactly when no loop variable is read | `Nested.lean` |
| `selfLoops_perm`, `deferred_after_binder`, `eq_reference` | #3102's hop order keeps every hop and places a self-loop after its binder | `HopOrder.lean` |
| `lowerInline_once`, `planMatch_stripped` | Inline maps become one Filter each, and no operator keeps one | `PlanMatch.lean`, `PlanMatch2.lean` |
| `rebuild_keeps_where` | The previous MATCH's WHERE survives the scan rebuild (since #2390) | `Match.lean` |

### How `query_correct` is proved

1. **Model the plan.** `Plan` is the IR tree with operator payloads replaced by ids. A `Sem` gives each id its meaning: scans, traversals, predicates and writes are *parameters* here, since other projects prove them.
2. **Model evaluation.** `ev` is the reference meaning of a plan tree, following the batched runtime.
3. **Model the stitching.** `walkFirst` and `walkLoop` are the two insertion-point walks of `plan_query`; `insertStep` and `stitchLoop` are the tree surgery. `rustStitch_eq_nestStitch` shows the loop computes the nested fill.
4. **State "implements".** `Implements w ok p F` says: put any input plan at the slot that walk `w` finds in `p`, and `ev` of the result is `F` applied to `ev` of the input. Each clause theorem in `Clauses.lean` proves this for one clause kind.
5. **Compose.** `query_correct` inducts on the clause list: the nested fill of plans that each implement their clause evaluates to the composition of the clause meanings.
6. **Hypotheses.** Sub-plans that come from elsewhere are hypotheses. The MERGE match branch must implement its pattern (`MatchBranch`), and FOREACH and `CALL {}` bodies must implement their clauses. `project_correct` assumes DISTINCT of an aggregate equals the aggregate. These are proven in other projects or are standard; they are stated, not linked.

## Bugs found

Each bug was checked live, Rust release module vs C module, and has a Rust repro (`cargo test -p graph --test lean_planner_build`, not committed; `bug_*` tests assert C's answer and fail today). See [`../FINDINGS.md`](../FINDINGS.md).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `merge_path_last_misplaced`, `merge_path_last_skips_merge` | MERGE with a named path as the last clause (of a query, a `CALL {}` body or a FOREACH body) never runs: "Variable _anon_0 not found" | **Open** (PR #3101 not merged) |
| 2 | `cp_stitch_shape`, `cp_stitch_loses_correlation` | A `MATCH … WHERE` followed by a multi-component MATCH is cross-joined, not correlated: Rust `[1,0,7]`, C `[]` | **Open** (PR #3100 not merged) |
| 3 | `pre2390_rebuild_drops_where` | The previous MATCH's WHERE was dropped when the next MATCH traversed from its variable; `… DELETE b` deleted data C does not | Fixed by #2390 (`rebuild_keeps_where`) |
| 4 | `selfloop_ignores_chain`, `pre3102_chainOp_selfloop_unbound` | A self-loop hop after other hops dropped them: `[null,2,3]` vs C `[1,2,3]` | Fixed by #3102 (#2999; `chainRel_selfloop_unbound`, `selfloop_scan_keeps_chain`) |
| 5 (W4-plan-1) | `toPlan_not_null_bug` | `NOT (complex)` with an inline pattern keeps null rows: Rust 2, C 1 | **Open** |
| 6 (W4-plan-4) | `call_validates_rest`, `call_then_match_rejected`; history `pre3297_call_skips_rest`, `pre3297_call_then_match_accepted` | Nothing after a procedure CALL was validated; `CALL db.labels() YIELD label MATCH (n)` reached a `parent().unwrap()` in `utilize_node_by_id` | **Fixed by #3297** (b774fdfad); the `unwrap()` in `utilize_node_by_id` remains as a hazard |
| 7 (W4-plan-5) | `with_last_accepted` | A query may end with WITH; C and openCypher reject it | **Open** |
| 8 (W8-plan-1) | `create_path_last_misplaced`, `create_path_last_skips_create` | Regression from #2829: CREATE with a named path as the last clause, after another clause, never runs | **Open** (PR #3101 not merged; covers bugs 1 and 8) |
| #2311 | `bug2311_optional_path_null` | `MATCH (n0:P) OPTIONAL MATCH p0=(n0) RETURN n0.id, p0`: Rust `1, null`, C `1, [(0)]`. The bound node-only branch never builds the path (`BindsAllPaths` fails) | **Open** |
| #2737 | `bug2737_optional_path_null` | `MATCH (n1:X) OPTIONAL MATCH p6 = (n1) RETURN p6`: Rust `null`, C `[(0)]`. Same cause as #2311 | **Open** |
| #2743 | `bug2743_bound_path_drops_rows` | `MATCH (n0:V) WITH * MATCH p0=(n0) RETURN count(*)`: Rust 0, C 1. No path, and the plan is not Apply-wrapped, so the earlier clauses land under an `Argument` that never runs them (`argIsLeaf` fails) | **Open** |
| #2744 | `bug2744_bound_path_unfed_argument` | `MATCH (n0:A) WITH n0 MATCH p=(n0) RETURN n0.k8, p`: Rust 0 rows, C 1 row. Plan `Project(ExpandInto(Argument(Project(Scan))))`, as the live EXPLAIN | **Open** |
| #2727 | `bug2727_alias_path_drops_rows` | `MATCH (n0:A) WITH n0 AS alias4 MATCH (alias4), p1=(alias4) RETURN alias4.k0`: Rust 0 rows, C `'x'` | **Open** |
| #2725 | `bug2725_call_then_bound_path_loses_var` | `MATCH (n0) CALL db.labels() YIELD label MATCH (n0), p0=(n0) RETURN …`: Rust `Variable n0 not found`, C `1, A, [(0)]`. Plan `Project(Argument(ProcedureCall(Scan)))`: one empty row | **Open** |
| #3450 | `bug3450_remove_keeps_verified_label` (property `RemoveKeepsVerifiedSound`, `RemoveVerified.lean`) | `REMOVE a:A` leaves `A` in `verified_labels`, `WITH` carries it, and the next MATCH skips the `hasLabels` check on the bound `a`: `MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH (a:A)-[:R]->(b)` returns 1 row `[] [B]`, C 0. `specRemove_sound`, `project_sound` prove the fixed REMOVE and the WITH carry keep `VerifiedSound` | **Open** |

Also: `where_scan_indistinguishable` shows why bug 3 happened (a planner-built `Filter` over a scan looks the same as the shape the optimizer rebuilds). `planFilter_correct` assumes all patterns of one WHERE start in one scope, because the planner keys inline patterns by variable id only; no query breaking that was found.

The six issues #2311 … #2725 share two root causes, each a general theorem in `PathBind.lean`: `nodeOnly_bound_no_paths` (the already-bound node-only branch of `plan_match`, mod.rs:1915-1962, never adds a `PathBuilder`) and `planMatchArm_path_unwrapped` (the MATCH arm's `all_visited`, mod.rs:3067, counts the new path variable, so the plan is not wrapped in `Apply`). The properties a fix must prove are `BindsAllPaths` (every named path of the pattern is bound by the plan; `unbound_path_built` shows the unbound case already does) and `Plan.argIsLeaf` (no `Argument` has a child).

Triaged here but not in this project: #2502 is the parser (`parse_pattern`, cypher.rs:1562-1572, called for a WHERE pattern predicate at cypher.rs:2119, takes a following `MATCH` keyword as a continuation and merges that clause into the predicate: `MATCH (a:A) WHERE (a)-[:Q]->() MATCH (a)-[:R]->(m) RETURN m` → `'m' not defined`; parenthesising the WHERE fixes it). #2556 is the optimizer: the planner puts the `t {id:100}` Filter directly above the hop that binds `t` (`firstRel_lowers_to`, `chainRel_lowers_to`), and the optimized plan has it below that hop.

There are no CONDITIONAL rows in this project.

## Files

| File | What's in it |
|---|---|
| `PlannerBuild.lean` | The report header and imports |
| `Core.lean` | Records, bags, the IR tree `Plan`, its meaning `ev`, and the `Sem` parameters |
| `Stitch.lean` | `plan_query` stitching: the walks, `insertStep`, `stitchLoop`, `rustStitch_eq_nestStitch` |
| `Clauses.lean` | Each clause plan (`planProject`, `planMerge`, `planProc`, `planUnion`, …) against its openCypher meaning |
| `Match.lean` | `query_correct`, the Cartesian-product stitch, and the MATCH-specific bugs |
| `Expr.lean` | Expressions as the planner sees them (`Ex`, `QG`), inline-variable helpers |
| `Decomp.lean` | Three-valued WHERE (`tv`) and `collect_patterns_and_rebuild` (`collect`) |
| `ToPlan.lean` | `expr_to_plan` (`toPlan`), `toPlan_correct`, `toPlan_not_null_bug` |
| `Filter.lean`, `FilterPlan.lean` | `plan_filter` end to end |
| `AndOf.lean` | `ExprIR::and_of`/`or_of` flattening (#3335) |
| `Nested.lean`, `Nested2.lean` | Pattern-comprehension extraction and hoisting |
| `PlanMatch.lean`, `PlanMatch2.lean` | `plan_match`: hop operators, sorting, label filters, path elision, inline-map lowering |
| `HopOrder.lean` | `self_loops_after_binders` (#3102) and the old loop |
| `Misc.lean`, `Rename.lean`, `Fmt.lean` | Plan-tree helpers, `rename_projection_outputs`, EXPLAIN text |
| `Ast.lean`, `Ast2.lean` | `parser/ast.rs`: variables, query graphs, connected components, validation, `Display` |
| `PathBind.lean` | Named paths over an already-bound node: `BindsAllPaths`, `planMatchArm` (the MATCH arm), `Plan.argIsLeaf`, bugs #2311, #2727, #2737, #2743, #2744, #2725 |
| `RemoveVerified.lean` | #3450: `VerifiedSound` (verified labels are on the node), REMOVE / WITH effect on `verified_labels`, `bug3450_remove_keeps_verified_label` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 90 rows: 48 in `planner/mod.rs`, 41 in `parser/ast.rs` and 1 in `planner/hop_order.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 90 |

Every function listed is PROVEN. For the functions involved in the open bugs, the theorems include the counterexamples.

## What the model does not cover

- Scans, hops, expressions and writes are parameters (`Sem`). Their meaning is proven in other projects.
- `plan_match`'s operator choice is proven as a table plus binding and label invariants, not against a graph-matching semantics. Parity with C is checked by differential tests.
- Runtime errors in expressions are not modelled; WHERE uses Kleene logic only.
- A pattern's inline property expressions are abstracted to the variables they read.
- The optimizer and batching are other projects ([`../optimizer_rewrites`](../optimizer_rewrites), [`../optimizer_scan`](../optimizer_scan)).
- Clause meanings thread the graph state one clause at a time (eager), matching the plan's `Commit` and eager operators. Per-row interleaving of reads and writes is not modelled.
- Sub-plans (the MERGE match branch, FOREACH and `CALL {}` bodies) are hypotheses.
- Variable ids are compared without scope in a few places (`emit_rel` path membership, `filter_vars`, `connected_components`); these are suspected, not confirmed, issues.

## Build and check

```bash
cd proofs/planner_build && lake build    # this project
bash proofs/lean_ci.sh planner_build     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                  # repo-wide coverage report
```

**When you change `planner/mod.rs`, `planner/hop_order.rs` or the query-graph and validation code in `parser/ast.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. A change to the insertion-point walks or to a clause arm usually breaks a `*_correct` theorem in `Clauses.lean` or `Stitch.lean` first; that is where to look.
