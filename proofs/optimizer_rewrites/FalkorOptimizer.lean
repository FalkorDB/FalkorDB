/-
# Optimizer rewrites: which passes preserve the query result, and which do not

Rust source: origin/main `256ab4d3f`. `Variable::is_unnamed` is PROVEN (`Helpers.varIsUnnamed_iff`); every
production row is PROVEN (the NOT COVERED rows are test/bench code). The `VarKey` refactor
(`var.key() == alias.key()` is the `(id, scope_id)` equality the models already use) needed no change to a
statement. Line pins inside bug histories below keep the revision they were written for; COVERAGE.tsv is authoritative.

`^` propagates operand errors and type-checks its operands: `Basic.eval` has `pow` (`powV`) and
`Err.notFound`; `sub`/`div` bodies are `subV`/`divV`. `ConstEval`: the plan-time constant evaluator
(`evalG` with every variable an error) is sound — `constEval_sound` — which discharges `hconst` for
`eliminate_whole_filter` and `htrue` for `andLoop_drop_true`
(`constTrue_filter`, `andLoop_drop_constTrue`): `is_constant_true` and
`eliminate_true_filters` are PROVEN.

A Lean 4 model of the optimizer passes in `graph/src/planner/optimizer/`
(origin/main), except `select_scan_node.rs`, `utilize_index.rs` and
`utilize_node_by_id.rs`'s scan changes (another agent's). Of `select_scan_node.rs` only
the #2556 chain-rebuild is modelled here (`PushPlan.selectScan`); the pass itself is in
`proofs/optimizer_scan`. Each pass is judged against a reference semantics of the IR:
rows are association lists (`Basic.Row`), expressions follow `eval.rs`
(three-valued, errors short-circuit, `AND` exactly as `eval.rs`), `Filter`
keeps `true` and drops `false`/`null` (`filter.rs`), and operators are
list functions (`cp`, `extend`, `take`, `drop`, `mergeSort`, …). A rewrite is
correct if it preserves the result multiset — and row order where ORDER BY
applies — and does not make a succeeding query fail.

Files: `Basic` (rows, eval, Filter, push-down, merge, eliminate_true_filters),
`References` (the "is this variable read?" oracle `optimizer/references.rs` behind
reduce_expand_into / reduce_bound_edge / reduce_var_len_path / fuse_anonymous_traverse,
also the a9377c636 and PR #2918 oracles), `References2` (the tree walks:
ancestor-only for the three `reduce_*` passes, whole-plan for fusion), `Passes` (fusion, optional fusion, node-by-id, reduce_count,
reorder_labels, hash join, the push-down routing and exclusion list).
Each file's header maps its definitions to Rust `file:line`.
Files (wave 5): `Helpers` (the syntactic helpers: parameters, variable
references, count extraction, fusion preconditions, Project/Aggregate outputs),
`Sequence` (`optimize` sequencing, the tree rebuilds, edge-filter absorption).
`PushPlan` (gap closed 2026-10-20: push_filters_down on a plan tree with write
effects and scope-blind ids; #748, #997, #3352; and select_scan_node's chain
re-ordering, #2556).
`AndOf` (every rebuilt filter goes through `ExprIR::and_of`; splicing an
AND into an AND keeps the runtime value — `scL_splice`, `flat_eval`, `flat_passes`).
Build: `lake build` — 162 theorems, no `sorry`/`admit`/`axiom`. Every production row of
COVERAGE.tsv is PROVEN (the NOT COVERED rows are test/bench code). The four `reduce_*`/fusion passes use `references.rs`; fusion checks the whole plan; filters are rebuilt with `and_of`.

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
* `filter_true`, `andLoop_drop_true`, `filter_and_single`, `eliminate_whole_filter`,
  and `constEval_sound`, `constTrue_filter`, `andLoop_drop_constTrue` —
  eliminate_true_filters is exact: the plan-time constant verdict is the runtime
  verdict on every row.
* `refsNew_complete` — `ir_references_variable` (references.rs) sees every read of every
  IR variant of a plan the planner built (`PlanInv`: inline maps are Filters —
  proofs/planner_build `planMatch_stripped`); `refsNew_sound` — it reports only reads or
  variables the operator binds; `iqLoop_iff` — the index-query stack loop finds an operand at any
  `And`/`Or` depth; `setItemsRefs_iff`, `exprRefs_iff`.
* `anyRead_plug`, `wholeRead_covers`, `fuse_unread_sound` — fusion's whole-plan check
  sees every reader of the intermediate and both edges, inside or outside the hops.
* `refsPR_complete`, `readOutside_iff`, `pr_keeps_flag_when_read` — PR #2918 (its walk is still
  not merged); `ancestors_le_outside` — the old oracle erred only toward collapsing.
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
  `fusable_spec`, `fusedCT_spec`; `branchOut_spec`, `branchOut_copy_new` (historical `pre3281_branchOut_copy_old`).
  W6-opt-1 Case-2 walk (`CallBoundary`, from the retired pr2845_review):
  `Pr2845.filter_pushed_through_boundary` (earlier walk `inheritedPre3281`), fix
  `fixed_keeps_filter`/`fixed_conservative`; the current walk (stops at a two-child `Optional`,
  inherits `branch_visible_variables`): `keeps_filter_main`, `main_sub_fixed`, `main_eq_fixed`;
  `visT_reset`, `visT_semi_first`, `visT_plain`, `visT_pattern_predicate`; on the plan tree
  (`PushPlan`, `Cfg.pr3281`): `bv_sub_gv`, `bvPre3281_sub_bv`, `bv_rebinding_kept`,
  `opt2_walk_stops`, `pr3281_3352_fixed`, `pr3281_748_fixed`, `pr3281_997_fixed`, `pr3281_controls`.

## CONFIRMED bugs (Rust repro: `cargo test -p graph --test lean_optimizer_rewrites`;
## Cypher compared on live servers, Rust release module vs C `bin/macos-arm64v8-release/falkordb.so`)

1. **Filter pushed below LIMIT / SKIP** — `push_filters_down.rs` does not
   exclude `Limit`/`Skip` (Lean `limit_skip_not_excluded`, `limit_not_transparent`,
   `skip_not_transparent`). 5 nodes v=1..5:
   `MATCH (n:N) WITH n ORDER BY n.v LIMIT 2 MATCH (m:N) WHERE n.v > 1 RETURN count(*)`
   → Rust 10, C 10, openCypher 5. `… SKIP 1 MATCH (m:N) WHERE n.v = 5` → 0 / 0 / 5.
   Shared with C (spec bug). Fix: add `IR::Limit | IR::Skip` to the exclusion list
   (or allow only conjuncts whose vars are all Apply-inherited — `push_limit_const`).
   Tests `bug_filter_pushed_below_limit`, `bug_filter_pushed_below_skip`.
3. **NodeByIdSeek with a negative bound** — `runtime.rs` `id as u64`:
   `MATCH (n) WHERE id(n) > -1 RETURN count(n)` → Rust 0, C 5 (also `>= -5`: both 0,
   `<= -1`: both all = #2266). Lean `id_gt_neg_empty`, `id_le_neg_all`. Fix: clamp
   negative Int bounds (Gt/Ge → no constraint; Lt/Le/Eq → empty) before the cast.
   Test `bug_node_by_id_gt_negative_returns_nothing`.
4. **NodeByIdSeek errors on non-integer bounds** — `runtime.rs`: `id(n) = null`,
   `= 1.0`, `= '1'`, `> 1.5`, `= true` → Rust "Node ID must be an integer"; C returns
   0 rows; the Filter being replaced returns 0 rows (null) or, for `1.0`, node 1.
   Lean `id_eq_null_errors`. Fix: only rewrite when the bound is an Int at runtime,
   else fall back to a Filter (Null → empty, whole Float → Int). Test `bug_node_by_id_null_errors`.
5. **Hash join drops Int/Float pairs near 2^53** — `replace_cartesian_with_hash_join.rs`
   introduces `ValueHashJoin`, whose key (`value_hash_join.rs key_as_i64`) is exact
   while `=` compares via `i as f64`. Nodes x=9007199254740993, x=9007199254740992.0:
   `MATCH (a:L),(b:L) WHERE a.x = b.x RETURN count(*)` → Rust 2, C 4; `(a.x=b.x)=true`
   → 4 on both. Lean `hash_join_drops_row` (and `vhj_correct` for when it is safe).
   Root is the Int/Float equality of #2891, but this is the optimizer changing a
   result. Test `bug_hash_join_disagrees_with_filter_on_int_float`.
6. (model only) `id_only_routing_unsound`: push-down routing uses bare ids (mod.rs) and Apply Case 2
   inheritance (push_filters_down.rs); the uncorrelated `CALL {}` repro (known #2557,
   `MATCH (o1:N) CALL { MATCH (q:N) WHERE q.v = 1 RETURN q } RETURN o1.v, q.v`) no longer reproduces live.
7. **(known #2896) Sibling readers.** The three `reduce_*` passes walk
   ancestors only, so a reader in a sibling subtree is missed —
   `MATCH (a)-[r]->(b) CALL { WITH r RETURN r.w AS w } RETURN w` → Rust 1 row, C 3;
   `RETURN [(a)-->(b) | r.w]` → Rust `[1]`, C three rows (Lean `main_misses_call_sibling`);
   and the oracle ignores `sibling_edges` (W2-traverse-1: `MATCH (a)-[r]->(x)<-[s]-(c)
   RETURN count(*)` → Rust 1, C 2; Lean `refsNew_misses_sibling_edges`).

## `PushPlan`: push-down on a plan tree with writes

The push-down theorems above compare result rows only and route by `(id, scope)`.
`PushPlan` gives push_filters_down a plan tree with graph state that writes
change, and the routing exactly as main computes it (bare ids, `get_variables`
walking through `Aggregate`, Case 1/Case 2 inheritance). Property `EffectSound`
(rows **and** the node count left behind are kept; the only write modelled is creating a bare
node, labels, relationships and attributes are not written); `RowSound` is the old criterion;
`RoutesBound` (a pushed conjunct reads only variables available where it lands).

## Gap closed 2026-10-20 (#2556) — `PushPlan`, select_scan_node chain re-ordering

The push-down proofs judged where `push_filters_down` moves a Filter, but no
model followed an inline-attr Filter through `select_scan_node`'s chain
rebuild. `PushPlan` now has `IR::CondTraverse` with its relationship and
`transposed` (`Op.hop`), the rebuild branch of `select_scan_node`
(`selectScan`, select_scan_node.rs) and the property
`KeepsFiltersBound` (a rewrite never moves a Filter below the operator that
binds what it reads).

11. **#2556: an inline property filter lands below the hop that binds it** —
   with the bottom hop's outer child kept (`Unwind`), `initial_bound` still
   seeds the best endpoint (`select_scan_node.rs`); `t {id:100}` wins
   the endpoint choice on cardinality, so `t` is "bound" before any hop and
   its Filter is placed after the first hop (:1202-1230).
   `MATCH (x:Person {id:2}) WITH collect(x) AS fs UNWIND fs AS f MATCH
   (f)<-[:HAS_CREATOR]-(post:Post), (post)-[:HAS_TAG]->(t:Tag {id:100}),
   (post)-[:HAS_TAG]->(tag:Tag) RETURN count(post)` → Rust 0, C 3 (live). Lean
   `bug2556_plan`, `bug2556_t_seeded`, `bug2556_filter_below_binding`
   (`¬ KeepsFiltersBound`, count 0 vs 3); `bug2556_fixed` (no seeding: the C plan);
   `bug2556_push_not_cause` (`push_filters_down` leaves the Filter in place).

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

* `utilize_node_by_id.rs` `optimized_plan.node(idx).parent().unwrap()` panics on a root scan
  (Lean `root_scan_has_no_parent`); no known query reaches it, since validation rejects
  `CALL db.labels() YIELD label MATCH (n)` (planner_build `call_then_match_rejected`). Hazard only.
* Filters on zero-variable conjuncts (`rand() < 0.5`) are pushed into every CP/Apply branch.
* Pushing into one CP branch evaluates the conjunct even when the other branch is
  empty, so a query can start to error (Lean `push_cp_left` needs `NoErr`).
  Observed on C (`MATCH (n:P),(m:Missing) WHERE 1/(n.v-n.v)=0` → C error, Rust 0).

## Open PRs touching these files (model = origin/main)

* #2918 — `variable_read_outside` for the three `reduce_*` passes (its per-operator
  oracle is superseded by references.rs): modelled as `readOutside`
  (`readOutside_iff`); would fix the sibling part of bug 7 (`main_misses_call_sibling`).
* #3092 — `collect_subtree_variables` stops at Project/Aggregate (their outputs
  only) instead of `get_variables` of the whole subtree — same direction as
  the Apply-left routing of `branch_visible_variables`; `mem_subtreeIds` describes origin/main.
* #3100 — exports `ir_references_variable` for the planner's CP-correlation check
  (planner bug 2); no change to the oracle itself (needs a rebase onto references.rs).
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
import FalkorOptimizer.ConstEval
import FalkorOptimizer.References
import FalkorOptimizer.References2
import FalkorOptimizer.Passes
import FalkorOptimizer.Helpers
import FalkorOptimizer.Sequence
import FalkorOptimizer.CallBoundary
import FalkorOptimizer.AndOf
import FalkorOptimizer.PushPlan
