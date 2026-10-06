/-
# Optimizer rewrites: which passes preserve the query result, and which do not

A Lean 4 model of the optimizer passes in `graph/src/planner/optimizer/`
(origin/main `2363723ac`), except `select_scan_node.rs` and `utilize_index.rs`
(another agent's). Each pass is judged against a reference semantics of the IR:
rows are association lists (`Basic.Row`), expressions follow `eval.rs`
(three-valued, errors short-circuit, `AND` exactly as `eval.rs:674`), `Filter`
keeps `true` and drops `false`/`null` (`filter.rs:48-80`), and operators are
list functions (`cp`, `extend`, `take`, `drop`, `mergeSort`, …). A rewrite is
correct if it preserves the result multiset — and row order where ORDER BY
applies — and does not make a succeeding query fail.

Files: `Basic` (rows, eval, Filter, push-down, merge, eliminate_true_filters),
`References` (the "is this variable read?" oracle behind reduce_expand_into /
reduce_bound_edge / reduce_var_len_path / fuse_anonymous_traverse, origin/main
vs PR #2918), `Passes` (fusion, optional fusion, node-by-id, reduce_count,
reorder_labels, hash join, the push-down routing and exclusion list).
Each file's header maps its definitions to Rust `file:line`.
Files (wave 5): `Helpers` (the syntactic helpers: parameters, variable
references, count extraction, fusion preconditions, Project/Aggregate outputs),
`Sequence` (`optimize` sequencing, the tree rebuilds, edge-filter absorption).
Build: `lake build` — 98 theorems, no `sorry`/`admit`/`axiom`. Every row of
COVERAGE.tsv is PROVEN. Rust source: origin/main `3fec7d7c9`
(an origin/main checkout; optimizer files identical to `2363723ac`).

## Proven (plain English)

* `eval_andL` — the binary AND fold is the Rust n-ary `And` loop.
* `eval_congr` — an expression depends only on the variables it mentions.
* `push_cp_left` / `push_cp_right` — a conjunct covered by one CartesianProduct
  branch (and not bound by the other) can move into that branch.
* `push_extend` — …and below any per-row operator that keeps its input
  columns (Unwind, CondTraverse, Apply, ProcedureCall).
* `push_sort_perm` + `push_sort_sorted` — below Sort: same multiset, still sorted.
* `push_limit_const` — below Limit only when the verdict is constant (Apply-argument case).
* `merge_child_first_row` — merging stacked filters is exact *if the inner
  conjunct comes first* (and is not NULL).
* `filter_true`, `andLoop_drop_true`, `filter_and_single`, `eliminate_whole_filter`
  — eliminate_true_filters is exact given that plan-time constant evaluation agrees
  with runtime evaluation (false for `^`, known #2902).
* `refsPR_complete` — PR #2918's `ir_references_variable` sees every read of every IR variant.
* `readOutside_iff` — PR #2918's `variable_read_outside` is exactly "some
  operator outside the traverse's subtree reads v"; `pr_keeps_flag_when_read`.
* `refsMain_sound`, `ancestors_le_outside` — origin/main errs only toward
  collapsing (never keeps a flag it should drop).
* `fuse_same_support` — anonymous-chain fusion keeps the set of rows (and,
  `fuse_changes_multiplicity`, collapses paths to pairs exactly as C does).
* `fuse_optional_correct` — Optional-over-traverse fusion is column-exact.
* `flip_holds`, `step_correct`, `step_no_wrap`, `idFilter_correct` —
  `evaluate_id_filter` is exact for integer bounds in `[0, 2^63)`; its `+1`/`-1` never wrap.
* `reorder_perm`, `reorder_preserves` — reorder_labels cannot change results.
* `vhj_correct` — ValueHashJoin equals Filter(=) over CP whenever key equality
  coincides with Cypher `=`.
* `reduce_count_correct` — the count is the label cardinality of the plan-time graph.
* `routesScoped_sound` — routing on `(id, scope)` only pushes where the variable is bound.

* Wave 5: `passes_order`, `applyAll_sound` (each pass preserving a relation ⇒
  `optimize` does), `optimize_nested`; `rebuildHJ_id`/`rebuildHJ_sound`
  (`rebuild_with_hash_joins` is a copy plus sound local rewrites);
  `rebuildCP_off`/`_on`, `splitCP_perm`; `findVlt_spec`, `absorb_correct`
  (absorbing edge-only conjuncts into the var-length traverse's per-hop filter
  keeps the rows, under the per-edge reading both engines use);
  `hasParam_iff`, `subst_params`, `evX_subst`; `refsVar_iff`, `refsVarId_iff`,
  `refsVar_le_refsVarId`; `countVar_spec`; `attrsEmpty_iff`, `canFuse_spec`;
  `fusable_spec`, `fusedCT_spec`; `branchOut_spec`, `branchOut_copy_new`.

## CONFIRMED bugs (Rust repro: `cargo test -p graph --test lean_optimizer_rewrites`;
## Cypher compared on live servers, Rust release module vs C `bin/macos-arm64v8-release/falkordb.so`)

1. **Filter pushed below LIMIT / SKIP** — `push_filters_down.rs:165-178` does not
   exclude `Limit`/`Skip` (Lean `limit_skip_not_excluded`, `limit_not_transparent`,
   `skip_not_transparent`). 5 nodes v=1..5:
   `MATCH (n:N) WITH n ORDER BY n.v LIMIT 2 MATCH (m:N) WHERE n.v > 1 RETURN count(*)`
   → Rust 10, C 10, openCypher 5. `… SKIP 1 MATCH (m:N) WHERE n.v = 5` → 0 / 0 / 5.
   Shared with C (spec bug). Fix: add `IR::Limit | IR::Skip` to the exclusion list
   (or allow only conjuncts whose vars are all Apply-inherited — `push_limit_const`).
   Tests `bug_filter_pushed_below_limit`, `bug_filter_pushed_below_skip`.
2. **Filter merge evaluates the pushed (outer) conjunct first** — `push_filters_down.rs:122`
   `for f in [&filter, &child_filter]`. `MATCH (n:N) WHERE n.v <> 1 MATCH (m:N) WHERE 1/(n.v-1) = 0 RETURN count(*)`
   → Rust `Division by zero`, C 15. Lean `merge_order_rust_errors`.
   Fix: `[&child_filter, &filter]` (`merge_child_first_row`). Test `bug_filter_merge_evaluates_outer_conjunct_first`.
3. **NodeByIdSeek with a negative bound** — `runtime.rs:1319` `id as u64`:
   `MATCH (n) WHERE id(n) > -1 RETURN count(n)` → Rust 0, C 5 (also `>= -5`: both 0,
   `<= -1`: both all = #2266). Lean `id_gt_neg_empty`, `id_le_neg_all`. Fix: clamp
   negative Int bounds (Gt/Ge → no constraint; Lt/Le/Eq → empty) before the cast.
   Test `bug_node_by_id_gt_negative_returns_nothing`.
4. **NodeByIdSeek errors on non-integer bounds** — `runtime.rs:1324`: `id(n) = null`,
   `= 1.0`, `= '1'`, `> 1.5`, `= true` → Rust "Node ID must be an integer"; C returns
   0 rows; the Filter being replaced returns 0 rows (null) or, for `1.0`, node 1.
   Lean `id_eq_null_errors`. Fix: only rewrite when the bound is an Int at runtime,
   else fall back to a Filter (Null → empty, whole Float → Int). Test `bug_node_by_id_null_errors`.
5. **Hash join drops Int/Float pairs near 2^53** — `replace_cartesian_with_hash_join.rs`
   introduces `ValueHashJoin`, whose key (`value_hash_join.rs:153 key_as_i64`) is exact
   while `=` compares via `i as f64`. Nodes x=9007199254740993, x=9007199254740992.0:
   `MATCH (a:L),(b:L) WHERE a.x = b.x RETURN count(*)` → Rust 2, C 4; `(a.x=b.x)=true`
   → 4 on both. Lean `hash_join_drops_row` (and `vhj_correct` for when it is safe).
   Root is the Int/Float equality of #2891, but this is the optimizer changing a
   result. Test `bug_hash_join_disagrees_with_filter_on_int_float`.
6. **(known #2557, new path)** Uncorrelated `CALL {}` filter pushed onto `Argument`:
   `MATCH (o1:N) CALL { MATCH (q:N) WHERE q.v = 1 RETURN q } RETURN o1.v, q.v`
   → Rust (1,1),(1,2)…(1,5); C (1,1),(2,1)…(5,1). Scope-blind ids (mod.rs:91-109)
   plus Apply Case 2 inheritance (push_filters_down.rs:214-261). Lean
   `id_only_routing_unsound`. Test `known_2557_call_subquery_filter_reads_outer_variable`.
7. **(known #2896, fixed by PR #2918)** origin/main `ir_references_variable`
   misses Unwind/ForEach lists, Create/Merge pattern properties, ProcedureCall
   args, LoadCsv, scan/traverse inline attrs, Skip/Limit, and every sibling branch:
   `MATCH (a)-[r]->(b) UNWIND [r.w] AS w RETURN w` → Rust [1],[3] of [1],[2],[3];
   `… CREATE (:C {w: r.w})` creates 2 of 3; `FOREACH`, pattern comprehension
   likewise; `()-[r*1..1]->() UNWIND r` loses rows (reduce_var_len_path).
   Lean `main_misses_*`; PR #2918 proven complete (`refsPR_complete`, `readOutside_iff`).

8. **(NEW) `utilize_node_by_id` panics on a root scan** — `utilize_node_by_id.rs:118`
   `optimized_plan.node(idx).parent().unwrap()`: `CALL db.labels() YIELD label MATCH (n)`
   (accepted because `inner_validate` skips everything after a CALL — planner_build bug 6)
   plans `NodeByLabelScan`/`AllNodeScan` as the root → **Rust server crash**
   ("called `Option::unwrap()` on a `None` value"); C: `Query cannot conclude with MATCH`.
   Lean `root_scan_has_no_parent`. Fix: `let Some(parent) = …parent() else { continue }`.

## Shared with C / deviations from openCypher (not Rust regressions)

* `reduce_count` bakes the count at plan time: `CREATE (:Z) RETURN 1 AS c UNION ALL
  MATCH (n:Z) RETURN count(n) AS c` → 0 on Rust and C, `count(*)` sees the write
  (1). Lean `reduce_count_stale`.
* `fuse_anonymous_traverse` returns one row per (a, c) pair, as C does (`fuse_changes_multiplicity`).
* `absorb_edge_filters_into_vlt` treats `WHERE r.w = 1` on a var-len list as per-hop
  (C semantics); unabsorbed Rust errors on `r.w` over a Path, so absorption turns an
  error into rows; `size(r) = 2` / `all(x IN r …)` are also absorbed per-hop and
  error on both engines (openCypher: list predicates). Not modelled beyond this note.
* `MATCH ()-[r]-() RETURN count(r)` counts each edge once on both engines (openCypher: twice).

## Suspected, unconfirmed

* `branch_output_variables` (push_filters_down.rs:65) takes `copies.0` as the
  (Lean `branchOut_copy_new`)
  Project's output, but `copies` is `(old, new)` (binder.rs:2234, project.rs:115) —
  outputs are the new vars. Currently only makes push-down more conservative, but
  with scope-blind ids it can also admit a conjunct whose id matches an old var.
* Filters on zero-variable conjuncts (`rand() < 0.5`) are pushed into every CP/Apply branch.
* Pushing into one CP branch evaluates the conjunct even when the other branch is
  empty, so a query can start to error (Lean `push_cp_left` needs `NoErr`).
  Observed on C (`MATCH (n:P),(m:Missing) WHERE 1/(n.v-n.v)=0` → C error, Rust 0).

## Open PRs touching these files (model = origin/main)

* #2918 — `ir_references_variable` completeness and `variable_read_outside`
  (fuse_anonymous_traverse, reduce_bound_edge, reduce_expand_into,
  reduce_var_len_path): modelled as `refsPR`/`readOutside` (`refsPR_complete`,
  `readOutside_iff`); fixes bug 7.
* #3006 — `push_filters_down.rs:122` merges `[child_filter, filter]`: the
  `merge_child_first_row` order; fixes bug 2.
* #2981 — `push_filters_down` routes with `branch_visible_variables` (below an
  env-resetting operator only its outputs survive; ids of the subtree minus its
  children's) instead of `collect_subtree_variables` for the Apply left side —
  narrows `id_only_routing_unsound`'s collision window (still ids only).
* #3092 — `collect_subtree_variables` stops at Project/Aggregate (their outputs
  only) instead of `get_variables` of the whole subtree — same direction as
  #2981; `mem_subtreeIds` describes origin/main.
* #3100 — exports `ir_references_variable` for the planner's CP-correlation check
  (planner bug 2); no change to the oracle itself.
* #3054 — `reduce_expand_into` removes a collapsed (no longer emitted) edge from
  its siblings' `sibling_edges` uniqueness lists (`forget_sibling_edge`); the
  emit-flag decision (`refsMain`) is unchanged.

## Modelling gaps

* Row values are a 4-constructor `Val`; Float, lists, maps, nodes are abstract.
  Hash-join numbers use an explicit `toF` (exact for `[0, 2^54]`).
* The Filter operator is vectorized (`VectorEval`); the model uses row-wise
  `eval`, which the `AND` test queries agree with.
* Laziness (a Limit above stopping upstream errors) is not modelled.
* Unbound variables read as NULL; the real env apparently resolves by id and read
  the outer `o1` in bug 6.
* The tree rewrites themselves (index bookkeeping, `take_out`, `prune`) are
  modelled as the algebraic identity they perform, not as orx-tree mutations;
  `rebuild_with_hash_joins` re-processing is bounded by fuel (Rust relies on
  each rewrite consuming one conjunct).
* `absorb_correct` assumes the per-edge reading of an edge-only conjunct
  (`Per.perEdge`) — C's semantics and Rust's when absorbed; openCypher would
  reject `r.w` on a list.
-/
import FalkorOptimizer.Basic
import FalkorOptimizer.References
import FalkorOptimizer.Passes
import FalkorOptimizer.Helpers
import FalkorOptimizer.Sequence
