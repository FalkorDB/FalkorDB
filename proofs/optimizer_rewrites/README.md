# optimizer_rewrites: which optimizer passes keep the query result, and which do not

This project checks each optimizer pass in `graph/src/planner/optimizer/` against a reference meaning of the plan, and proves which rewrites keep the query result unchanged. It also gives concrete counterexamples where a rewrite changes the answer or makes a working query fail. Every query that goes through the optimizer depends on it. Examples:

- `WHERE` conjuncts are pushed down below joins, sorts and traversals;
- stacked filters are merged;
- `WHERE true`-style filters are removed;
- `id(n) = 5` becomes a NodeByIdSeek;
- a Cartesian product with `a.x = b.x` becomes a hash join;
- anonymous traversal chains are fused;
- an edge nobody reads is collapsed.

The passes for scans and indexes (`select_scan_node.rs`, `utilize_index.rs`) are in [`../optimizer_scan`](../optimizer_scan). The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorOptimizer.lean`](FalkorOptimizer.lean) and in the header of each file. This README explains *what* is proven and *why it matters*.

## What "correct" means here

The model gives the plan a reference meaning:

- **Rows** are association lists (`Row`).
- **Expressions** follow `eval.rs`: three-valued logic, errors short-circuit, and `AND` is evaluated exactly as the Rust n-ary loop (`eval_andL`).
- **Filter** keeps `true` and drops `false` and `null`.
- **Operators** are list functions: Cartesian product, extend, take, drop, sort.

A rewrite is correct if it keeps the multiset of result rows (and their order where ORDER BY applies), and does not make a query that succeeded start to fail.

That criterion ignores writes. Since 2026-10-20, `PushPlan.lean` also gives the plan a graph state that `CREATE`/`MERGE` change, and states the stronger `EffectSound`: a rewrite keeps the rows **and** the graph the query leaves behind. `run_readonly` proves a write-free plan leaves the graph unchanged, so between write-free plans `EffectSound` reduces to the rows criterion and the earlier theorems carry over; #748 and #997 are plans where they differ (`bug748_rows_only_blind`).

## The questions this project answers

**1. Can a `WHERE` conjunct be pushed down safely?**

- **Below a Cartesian product, yes, into the branch that binds its variables** (`push_cp_left`, `push_cp_right`).
- **Below Unwind, CondTraverse, Apply and ProcedureCall, yes** (`push_extend`): they keep their input columns.
- **Below Sort, yes:** same rows, still sorted (`push_sort_perm`, `push_sort_sorted`).
- **Below LIMIT or SKIP, no.** It changes which rows the LIMIT keeps (`limit_not_transparent`, `skip_not_transparent`). The exclusion list in `push_filters_down.rs` does not exclude them (`limit_skip_not_excluded`). Rust and C both return the wrong count here, so it is a shared spec bug, not a Rust regression. The only safe case is a conjunct whose verdict is the same on every row (`push_limit_const`).
- **Routing by variable id and scope is sound** (`routesScoped_sound`): it only pushes where the variable is bound.
- **Main routes by bare id, not scope** (#3352): `bug3352_routes_unbound` and `bug3352_slot_collision_drops_match`.
- **A variable-free conjunct enters every child, including the rows that feed a write** (#748, #997): `bug748_merge_skipped`, `bug997_create_skipped`.
- **Re-ordering a traversal chain can land an inline filter below the hop that binds its variable** (#2556, in `select_scan_node`): `bug2556_filter_below_binding`.

**2. Is merging two stacked filters exact?**

- **Only if the inner conjunct is evaluated first.** `merge_child_first_row` proves child-first order is exact.
- **On main, the outer one is evaluated first.** `MATCH (n:N) WHERE n.v <> 1 MATCH (m:N) WHERE 1/(n.v-1) = 0` fails with "Division by zero"; C returns 15 (`merge_order_rust_errors`).

**3. Is removing an always-true filter exact?**

- **Yes** (`constTrue_filter`, `andLoop_drop_constTrue`, `filter_true`, `filter_and_single`). The plan-time constant evaluator is the runtime `eval_node` with no row, so every variable is an error. `constEval_sound` proves that whenever it returns a value, every row gives that value. This holds because, since #2904, no modelled operator swallows a child's error.
- Before #2904 that was **false for `^`** (#2902): the `Pow` arm dropped errors with `flat_map`, so `WHERE x ^ true` folded to `true` and its filter was removed. Now `x ^ true` is not constant (`pow_var_not_constTrue`), and at runtime it is C's type error (`pow_true_type_error`).
- Since #3335 the rebuilt filter goes through `ExprIR::and_of`, which keeps the verdict, errors included (`flat_passes`, `flat_eval`, `scL_splice`).

**4. Is the id seek exact?**

- **For integer bounds in `[0, 2⁶³)`, yes.** `idFilter_correct` proves a node is in the seek range iff it satisfies every comparison, and `step_no_wrap` that the `+1`/`-1` adjustments never overflow.
- **Outside that range, no.** `id(n) > -1` returns 0 rows (C: all of them) because the bound is cast `as u64` (`id_gt_neg_empty`). `id(n) = null` raises "Node ID must be an integer" where C returns 0 rows (`id_eq_null_errors`).

**5. Does the hash join return the same rows as the filter it replaces?**

- **When key equality is Cypher `=`, yes** (`vhj_correct`).
- **Near 2⁵³, no.** The hash key compares integers and floats exactly, while `=` goes through `i as f64`. Two nodes with `x = 9007199254740993` and `x = 9007199254740992.0` give `MATCH (a:L),(b:L) WHERE a.x = b.x RETURN count(*)` → Rust 2, C 4 (`hash_join_drops_row`).

**6. When is a variable "read" by the rest of the plan?**

Several passes collapse an edge, a bound edge or a variable-length path when nobody reads it, using the oracle in `optimizer/references.rs` (added by #2390).

- **Per operator, yes.** `refsNew_complete` proves the oracle reports every read of every IR variant, for plans the planner builds; `refsNew_sound` proves it reports nothing else.
- **For fusion, yes.** Fusion checks the whole plan (`anyRead_plug`, `wholeRead_covers`, `fuse_unread_sound`).
- **For the three `reduce_*` passes, no.** They walk ancestors only, so a reader in a sibling subtree is missed: `MATCH (a)-[r]->(b) CALL { WITH r RETURN r.w AS w } RETURN w` → Rust 1 row, C 3 (`main_misses_call_sibling`). The oracle also ignores `sibling_edges` (`refsNew_misses_sibling_edges`).

**7. Do the remaining passes keep the result?**

- `fuse_same_support`: anonymous-chain fusion keeps the set of rows. It collapses paths to (start, end) pairs exactly as C does (`fuse_changes_multiplicity`).
- `fuse_optional_correct`: Optional-over-traverse fusion is column-exact.
- `reorder_perm`, `reorder_preserves`: `reorder_labels` cannot change results.
- `reduce_count_correct`: `reduce_count` returns the label cardinality of the graph at plan time.
- `absorb_correct`: absorbing edge-only conjuncts into a variable-length traverse keeps the rows, under the per-edge reading of `r.w` that both engines use.
- `applyAll_sound`: if every pass preserves a relation, `optimize` as a whole does.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `push_cp_left`, `push_cp_right`, `push_extend`, `push_sort_perm` | Push-down is exact below CP, per-row operators and Sort, for conjuncts that do not error (`NoErr`) | `Basic.lean` |
| `merge_child_first_row` | Merging stacked filters is exact when the inner conjunct runs first | `Basic.lean` |
| `constEval_sound`, `constTrue_filter`, `andLoop_drop_constTrue` | The plan-time constant verdict is the runtime verdict on every row, so removing a constant-true filter or conjunct is exact | `ConstEval.lean` |
| `flat_passes` | Rebuilding a filter with `and_of` gives the same verdict, errors included | `AndOf.lean` |
| `refsNew_complete`, `refsNew_sound` | The #2390 "is this variable read?" oracle is exact per operator | `References.lean` |
| `fuse_unread_sound` | Fusion's whole-plan check sees every reader | `References2.lean` |
| `idFilter_correct`, `step_no_wrap` | The id seek is exact for integer bounds in `[0, 2⁶³)` | `Passes.lean` |
| `vhj_correct` | The hash join equals Filter(=) over CP when key equality is `=` | `Passes.lean` |
| `fuse_same_support`, `fuse_optional_correct` | Traversal fusion keeps the rows | `Passes.lean` |
| `reorder_preserves`, `reduce_count_correct` | `reorder_labels` and `reduce_count` are exact | `Passes.lean` |
| `routesScoped_sound` | Push-down routing by `(id, scope)` only pushes where the variable is bound | `Passes.lean` |
| `applyAll_sound` | Pass-by-pass correctness gives correctness of `optimize` | `Sequence.lean` |
| `absorb_correct` | Absorbing edge filters into a var-length traverse keeps the rows (per-edge reading) | `Sequence.lean` |

### How `push_cp_left` is proved

1. **Model the rows and the product.** `cp xs ys` is every left row extended with every right row, in order: `xs.flatMap (fun x => ys.map …)`.
2. **Model the filter.** `passB p` is the Filter verdict for conjunct `p`: keep only on `true`.
3. **State the precondition.** `Covers p xs` says every variable `p` reads is bound by the left rows. `eval_congr` proves an expression depends only on the variables it mentions, so evaluating `p` on a combined row gives the same verdict as on its left half.
4. **Induct on the left rows.** For each left row, either `p` holds and every combined row passes, or it fails and none does. That is exactly filtering the left side first.

The theorem needs no hypothesis about other Rust code. The one assumption to watch is about errors. `passB` reads an error as "drop the row", and `filterRows_noErr` connects it to the real, error-propagating Filter only under `NoErr` (the conjunct never errors on those rows). Pushing into one branch evaluates the conjunct even when the other branch is empty, so a conjunct that can raise an error may make a query fail that did not before (see [What the model does not cover](#what-the-model-does-not-cover)).

## Bugs found

Each bug has a Rust repro (`cargo test -p graph --test lean_optimizer_rewrites`, not committed) and was compared live against the C module (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `limit_skip_not_excluded`, `limit_not_transparent`, `skip_not_transparent` | A filter is pushed below LIMIT/SKIP: `… ORDER BY n.v LIMIT 2 MATCH (m:N) WHERE n.v > 1 RETURN count(*)` → 10 (openCypher 5) | **Open**, shared with C |
| 2 (W2-rewrites-1) | `merge_order_rust_errors` | Filter merge evaluates the outer conjunct first: "Division by zero" where C returns 15 | **Open** (PR #3006 not merged) |
| 3 (W2-rewrites-2) | `id_gt_neg_empty`, `id_le_neg_all` | `id(n) > -1` → 0 rows (C: 5); `id(n) <= -1` returns every node on both (#2266) | **Open** |
| 4 (W2-rewrites-2) | `id_eq_null_errors` | `id(n) = null`, `= 1.0`, `> 1.5` → "Node ID must be an integer" (C: 0 rows) | **Open** |
| 5 (W2-rewrites-3) | `hash_join_drops_row` | The hash join drops Int/Float pairs near 2⁵³: 2 rows where C gives 4 | **Open** |
| 6 (W2-rewrites-4) | `id_only_routing_unsound` | A filter inside an uncorrelated `CALL {}` was pushed onto the Argument, swapping values (#2557) | No longer reproduces on main (most likely fixed by #2845); the Lean still describes id-only routing |
| 7 | `pre2390_misses_*`, `main_misses_call_sibling`, `refsNew_misses_sibling_edges` | An edge read outside the traverse's ancestors is collapsed (#2896, W2-traverse-1) | Per-operator part fixed by #2390; sibling part **open** (PR #2918 not merged) |
| 8 (W4-plan-4) | `root_scan_has_no_parent` | `CALL db.labels() YIELD label MATCH (n)` makes a scan the plan root; `utilize_node_by_id` calls `parent().unwrap()` on it | **Repro closed by #3297** (validation now rejects the query, main b84473a7f); the `unwrap()` remains as a hazard |
| W6-opt-1 | `filter_pushed_through_boundary` | Push-down gives an OPTIONAL MATCH inside `CALL {}` the outer ids: "Variable n not found" where C returns rows. `fixed_keeps_filter` checks the trial fix | **Open** |
| #748 | `bug748_merge_skipped` (property `EffectSound`; `bug748_rows_only_blind`) | `WITH 0 AS n0 CALL { MERGE () RETURN 0 AS n1 } MATCH () WHERE false RETURN 0`: the constant `false` is pushed into both Apply branches, so the MERGE never runs. Rust 0 nodes, C 1; with `WITH *` both 1 (`bug748_with_star_control`) | **Open** |
| #997 | `bug997_create_skipped` (property `EffectSound`; `bug997_rows_only_blind`) | `CALL { CREATE () } UNWIND 0 AS x MATCH () WHERE false RETURN 0`: `false` lands on the Apply's `Argument`, the CREATE never runs. Rust 0 nodes, C 1; the CASE-wrapped `false` reads `x` and stays above the Apply, 1 on both (`bug997_case_control`) | **Open** |
| #3352 | `bug3352_slot_collision_drops_match`, `bug3352_routes_unbound` (properties `RowSound`, `RoutesBound`) | `… WITH p, count(r0) AS n OPTIONAL MATCH (r:Rule)-[:SCOPED_TO]->(p) WHERE r.status = 'revoked' …`: Case 2 inherits the bare ids of the whole pre-WITH subtree (through `Aggregate`), id 2 is taken for `r`, the filter lands on the `Argument`. Rust `e = null`, C 123 | **Open** (PR #3367) |
| #2556 | `bug2556_filter_below_binding`, `bug2556_t_seeded`, `bug2556_plan` (property `KeepsFiltersBound`) | `… WITH collect(x) AS fs UNWIND fs AS f MATCH (f)<-[:HAS_CREATOR]-(post:Post), (post)-[:HAS_TAG]->(t:Tag {id:100}), (post)-[:HAS_TAG]->(tag:Tag) RETURN count(post)`: `select_scan_node` picks `t` as the best endpoint, seeds it into `initial_bound` although the outer `Unwind` child is kept and no scan for `t` is built, and places `t.id = 100` after the first hop, before `t` is bound. Rust 0, C 3 (the issue's query: a type error). Not `push_filters_down` (`bug2556_push_not_cause`); `bug2556_fixed` gives the C plan | **Open** |

Shared with C, not Rust regressions: `reduce_count` bakes the count at plan time, so a count after a write in the same query reads 0 (`reduce_count_stale`); fusion returns one row per (start, end) pair (`fuse_changes_multiplicity`).

**Fixed: #2902 (`^` made `eliminate_true_filters` drop a live filter).** `is_constant_true` and `eliminate_true_filters` used to be CONDITIONAL on `hconst`, the assumption that the plan-time constant evaluator agrees with runtime evaluation. That was false for `^` until #2904 (`fd2c76a6c`). `ConstEval.constEval_sound` now proves it for the modelled expression language, which includes `^`, so both rows are PROVEN. The counterexample was removed; in 17ccc5412.

`refsNew_complete` also assumes `PlanInv`: inline property maps have already been turned into Filters. That is proven in [`../planner_build`](../planner_build) (`planMatch_stripped`), but the two projects are not linked, so here it is a stated assumption.

## Files

| File | What's in it |
|---|---|
| `FalkorOptimizer.lean` | The report header and imports |
| `Basic.lean` | Rows, `Val`, `eval` (three-valued, errors, `^` since #2904), Filter, push-down below CP/extend/Sort/Limit, filter merge, `eliminate_true_filters` |
| `ConstEval.lean` | The plan-time constant evaluator and its soundness (`constEval_sound`, #2904) |
| `AndOf.lean` | `ExprIR::and_of` (#3335): splicing an AND into an AND keeps the runtime value |
| `Helpers.lean` | Syntactic helpers: parameters, variable references, count extraction, fusion preconditions, Project/Aggregate outputs |
| `Passes.lean` | Fusion, optional fusion, the id seek, `reduce_count`, `reorder_labels`, hash join, push-down routing and the exclusion list |
| `References.lean` | The "is this variable read?" oracle (`refsNew`), the pre-#2390 oracle and the PR #2918 walk |
| `References2.lean` | The tree walks: ancestors only for the `reduce_*` passes, whole plan for fusion |
| `Sequence.lean` | `optimize` sequencing, the tree rebuilds (`rebuild_with_hash_joins`), edge-filter absorption |
| `CallBoundary.lean` | W6-opt-1: push-down across the `CALL {}` entry projection (namespace `Pr2845`) |
| `PushPlan.lean` | Gap closed 2026-10-20: push-down on a plan tree with graph writes and main's id-only routing; `EffectSound` (rows and writes kept), `RowSound`, `RoutesBound`; #748, #997, #3352. Also `select_scan_node`'s chain rebuild and `KeepsFiltersBound` (#2556) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 80 function rows:

| Bucket | Rows |
|---|---|
| PROVEN | 38 |
| NOT COVERED | 42 |

- **PROVEN:** every production function in the optimizer files this project owns.
- **NOT COVERED:** all test or benchmark code. 16 rows are `#[cfg(test)]` plan-shape unit tests in `reduce_bound_edge.rs` and `reduce_var_len_path.rs`. 26 rows are the GraphBLAS `*_bench.rs` harnesses, listed here only so `bash proofs/coverage.sh` can exclude them.

## What the model does not cover

- Row values are a four-constructor `Val`. Floats, lists, maps and nodes are abstract. Hash-join numbers use an explicit `toF`, exact on `[0, 2⁵⁴]`.
- The Filter operator is vectorized in Rust; the model evaluates row by row. The AND test queries agree with it.
- Laziness is not modelled: a LIMIT above that stops upstream errors early.
- Unbound variables read as NULL in the model. In Rust they are resolved by id, which is how bug 6 read the outer `o1`.
- Tree rewrites (index bookkeeping, `take_out`, `prune`) are modelled as the algebraic identity they perform, not as tree mutations. `rebuild_with_hash_joins` re-processing is bounded by fuel; the Rust relies on each rewrite consuming one conjunct.
- `absorb_correct` assumes the per-edge reading of an edge-only conjunct. That is C's semantics and Rust's after absorption; openCypher would reject `r.w` on a list.
- Pushing into one CP branch evaluates the conjunct even when the other branch is empty, so an erroring conjunct can make a query fail. This is outside the push-down theorems; C shows the same behaviour.

## Build and check

```bash
cd proofs/optimizer_rewrites && lake build   # this project
bash proofs/lean_ci.sh optimizer_rewrites    # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                      # repo-wide coverage report
```

**When you change a pass in `graph/src/planner/optimizer/` (other than `select_scan_node.rs` and `utilize_index.rs`), or `references.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. A fix for one of the open bugs should turn its counterexample into a `pre<PR>_` history theorem and add the matching correctness theorem.
