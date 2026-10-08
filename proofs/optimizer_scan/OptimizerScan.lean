/-
# FalkorDB-rs scan selection passes: Lean 4 model, proofs, bug report

Targets: `graph/src/planner/optimizer/select_scan_node.rs`, `utilize_index.rs`,
`utilize_node_by_id.rs`, `reorder_labels.rs`, plus the runtime pieces their rewrites hand
work to (`Runtime::evaluate_id_filter`, `NodeByIndexScanOp`, `Index::build_query_node`).

Build: `lake build` (Lean 4.34, core only). No `sorry`, `admit`, `axiom`, `native_decide`;
`utilize_sound` and `seek_sound` depend only on `propext`, `Quot.sound`, `Classical.choice`.
64 theorems, 34 `decide`d examples (counterexamples and sanity checks).
Coverage per Rust function: `COVERAGE.tsv`. Rust repros: `graph/tests/lean_optimizer_scan.rs`
(`cargo test -p graph --test lean_optimizer_scan`; `bug_*` tests FAIL today by design).
Index repros need RediSearch, so they run live: `venv/bin/python proofs/optimizer_scan/repro.py`
(Rust release module on port 18200, C `bin/macos-arm64v8-release/falkordb.so` on 18201).

Reference semantics: a plan is a membership test on nodes (each scan yields a node at most
once), a hop chain is the conjunction of its hops, and the rewrite must select exactly what
`Filter → NodeByLabelScan` / `Filter → AllNodeScan` selects under Cypher's three-valued
comparison (only `true` keeps a row).

## Lean ↔ Rust

| Lean | Rust |
| --- | --- |
| `IdSeek.getIdFilter`, `Op.flip`, `collectFilters` | `utilize_node_by_id.rs:52-92, 106-119` |
| `IdSeek.step`, `evalIdFilter`, `asU64` | `runtime/runtime.rs:1309-1367` (`id as u64` at 1322) |
| `IdSeek.maxNodeId`, `seek` | `graph/graph.rs:1602` (→ `IdSpace::max_id`), `runtime/ops/node_by_id_seek.rs:63-73` |
| `Index.enc`, `arrEnc` | `index/mod.rs:716-845` (`Document::set`) |
| `Index.build`/`buildAll`/`buildSome`, `bsel`, `idxSel` | `index/mod.rs:1476-1687`, `Index::query` 1690 |
| `Index.canUtilize`, `evalIQ` | `runtime/ops/node_by_index_scan.rs:98-287` |
| `Index.hasT`, `firstP`, `nonIdxT`, `needsPost` | `utilize_index.rs:357-368, 547-595, 802-873` |
| `Index.buildOp`, `trySingle`, `mergeRange` | `utilize_index.rs:315-355, 602-704, 481-544` |
| `Index.pushAnd`/`mergeInto`, `pushOr`, `tryPushdown` | `utilize_index.rs:727-783` |
| `Index.utilize`, `Plan.sel` | `utilize_index.rs:973-1046`, `node_by_index_scan.rs:301-328` |
| `ScanOrder.Hop.swap`, `pick`, `order` | `select_scan_node.rs:370-387, 1000-1060` |
| `ScanOrder.place`, `runScoped` | `select_scan_node.rs:979-986, 1185-1235` |
| `ScanOrder.reorderLabels`, `withPrimary` | `reorder_labels.rs:16-47`, `utilize_index.rs:199-215, 286-309` |

## Proven (plain English)

Id seeks (`IdSeek.lean`):
- `flip_correct`, `flip_flip`: the `id()`-on-the-right flip table is right.
- `step_none`, `step_some`: each `evaluate_id_filter` arm keeps exactly the ids of the running range that satisfy it.
- `step_nonempty`: the running `min ≤ max` invariant holds, so `insert_range(min..=max)` is never inverted.
- `fold_none`, `fold_some`, `go_eq_fold`, `asU64_of_small`: the whole fold is exact for operands in `[0, 2^63)`.
- `seek_sound`: on a non-empty graph with non-negative integer operands, `NodeByIdSeek` = `Filter(id(n) op v …)` over `AllNodeScan`.

Index utilization (`Index.lean`, `IndexSound.lean`, `IndexPipeline.lean`):
- `ordI_swap`, `ord3_swap`, `eq3_swap`, `flip_correct`: comparison flip when the property is on the right.
- `sel_eq`: `Equal` on a faithful literal (non-lossy Int, String) and faithful stored value (Null/Int/String) is exact.
- `sel_range`: every one- or two-sided numeric or string range is exact except the equal-string-bounds-with-an-exclusive-flag trap.
- `sel_buildOp`: `build_op_query` is exact for all five operators.
- `sel_inList`, `buildSome_eqs`, `cu_inList`: `n.k IN [literals]` is exact and runtime-usable when non-empty.
- `sel_and2`: an index `And` is the intersection (a null child nulls both sides).
- `sel_merge`, `rangeOK_merge`, `cu_merge`: `merge_range_queries` selects the conjunction and stays well-formed and usable.
- `trySingle_spec`: `try_single_filter_scan` on covered atoms is exact, on `L`, well-formed, usable unless `IN`.
- `pushAnd_spec`, `pushOr_spec`, `buildSome_any`: the AND / OR branches of `try_filter_pushdown` select the conjunction and disjunction.
- `utilize_sound` (headline): single-label scan, Null/Int/String values, atoms `n.k op lit`, `lit op n.k`,
  `n.k IN [lits]`, unindexable conjuncts, combined by AND (no strict string bound) or OR — rewritten plan = original.

Scan selection (`ScanOrder.lean`):
- `swap_holds`, `swap_swap`: reversing a hop and flipping `transposed` tests the same edge.
- `run_perm`, `run_append`: reordering or re-sequencing tests keeps the selected rows.
- `pick_perm`, `order_sound`, `order_length`: the greedy hop ordering keeps every hop exactly once, oriented soundly.
- `runScoped_wellScoped`, `place_wellScoped`, `wellScoped_filts`: filter re-placement is sound **iff** the seed bound set is honest.
- `hasAll_perm`, `sortBy_perm`, `insertBy_perm`, `reorderLabels_sound`: `reorder_labels` cannot change a scan.
- `withPrimary_sound`, `primary_split`: putting the indexed label first + checking the rest = the full label set.

## CONFIRMED bugs (new in this wave)

1. **`select_scan_node` places an inline-attribute Filter before its variable is bound**
   — `select_scan_node.rs:979-986` seeds `initial_bound` with `best_node` even when the
   outer-context child is kept (no scan for `best_node` is built, line 1198). Lean:
   `ScanOrder` counterexample after `place_wellScoped`. Query
   `MATCH (b:A) WITH b LIMIT 1 MATCH (b)<-[]-(d)-[:R]->(c {v:1})-[:R]->(z) RETURN count(*)`
   → Rust 0, C 1 (also `ORDER BY`/`SKIP`/`DISTINCT` children; `WITH b` alone is fine).
   Found by the oracle fuzz (`fuzz3.py`, 1 Rust-only mismatch in 480 queries). Fix: insert
   `best_node` into `initial_bound` only when `existing_child` is `None`.
   Test `bug_select_scan_node_places_filter_before_its_variable_is_bound`.
   STILL PRESENT at 8743953a8 (re-checked live on a build of main's planner, b3582e34a: Rust 0,
   expected 1; the EXPLAIN shows the `c` Filter right after `(b)<-(d)`).
2. **`NodeByIdSeek` returns a node that never existed, and writes to it persist** —
   `Graph::max_node_id` (`graph.rs:1602`, `IdSpace::max_id` since #2846) is 0 on a graph with no nodes, so
   `evaluate_id_filter` yields `{0}` and `node_by_id_seek.rs:67` only subtracts deleted ids.
   Lean `IdSeek.gEmpty` examples. On a fresh graph `MATCH (n) WHERE id(n) = 0 RETURN n`
   → Rust 1 row, C 0. `MATCH (n) WHERE id(n)=0 SET n.x=1, n:Z` then `CREATE (m:Q)` →
   `MATCH (n) RETURN labels(n), properties(n)` Rust `[Z, Q] {x: 1}`, C `[Q] {}`;
   `DELETE` errors "0 was never allocated here"; `CREATE (n)-[:R]->(:W)` makes a self-loop.
   Fix: return `Ok(None)` when the graph has no id bound (`node_count + deleted == 0`), or
   make the range `[0, bound)` exclusive. Tests `bug_id_seek_phantom_node_*`.
   Re-checked live at 2c874022a (after #2846): still present (Rust 1 row, C 0).
3. **Computed constants ≥ 2^52 return every node** — `is_non_indexable_subexpr`
   (`utilize_index.rs:855-873`) flags only literal lossy ints; `4503599627370495 +
   4503599627370495 + 3` is not flagged, the Filter is dropped, `can_utilize_index` rejects
   the runtime value and the op falls back to a label scan with no Filter. Lean `IndexCex` C4.
   `MATCH (n:L) WHERE n.v = 4503599627370495 + 4503599627370495 + 3` → Rust all 6 nodes,
   Rust without index 0, C 0 (same for `n.v > 4503599627370495 * 2`). Fix: keep the Filter
   whenever the op may fall back (any non-literal value side), or have the fallback re-apply it.
4. **`IN` with a computed left side is pushed as `n.a IN [...]`** — `try_in_filter_scan`
   (`utilize_index.rs:620-645`) only asks that the side *contain* a property, and takes
   the first one. Lean C7/C7'. `abs(n.a) IN [1]` → Rust `[1]` (misses a=-1), without index
   `[1],[2]`, C `[1],[2]`; `toString(n.a) IN ['1']` → Rust `[]`, C `[1]`;
   `n.a + 1 IN [2]` → Rust `[]`, without index `[1]` (C errors). Fix: require
   `ExprIR::Property` of the scanned alias as the whole left side (as the comparison path does at 693).
5. **`x IN [n.a, n.b]` becomes array-contains on `n.a`** — same function, `(false, true)`
   arm with a list literal on the right. Lean C8. `2 IN [n.a, n.b]` → Rust `[]`, C `[3]`;
   `-1 IN [n.a]` → Rust `[]`, without index `[2]`. Fix: array-contains only when the right
   side *is* `Property(scanned alias)`.

## Also reproduced here but already reported by parallel wave-2 agents
- `proofs/index_layer` bugs 1, 3, 4, 5, 6, 8: folded `date()` constant returns every node
  (C3); `IN [.., date()]` drops the item (C9); multi-label AND → nothing / OR loses rows
  (C5, C6); temporals indexed as numbers (C2, FIXED by #3076); `n.v > 'B' AND n.v < 'B'`
  exact-match trap (C11, FIXED by #3072 — both `IndexCex` examples now state agreement);
  Bool/Int conflation (C1, shared with C). `utilize_sound` still assumes no temporal values and no
  strict string bound in an AND (gap: those hypotheses could now be weakened).
- `proofs/optimizer_rewrites` bugs 3, 4: `id(n) > -1` / `>= -5` / `< -1` / `<= -1` via
  `id as u64` (`bug_id_seek_negative_operand` here), non-integer id operands error.
- Array-contains keeps its Filter only at the root (`utilize_index.rs:773-780`): inside AND
  it is dropped, so `1 IN n.arr AND n.k > 0` returns `arr: [true]` (Bool/Int share the numeric
  array field). Rust with index `[4],[5]`, without `[5]`, C `[4],[5]` — index-dependent result
  shared with C (Lean C10).

## Suspected / hazards (not confirmed as wrong results)
- `utilize_node_by_id.rs:106` `parent().unwrap()` panics if a scan is the plan root. W4-plan-4's
  repro (`CALL db.labels() YIELD label MATCH (n) RETURN label`) no longer crashes: on builds at
  1c9994e37+#3076 (88e92ae25) and at main's planner (b3582e34a) the plan is
  `Project → AllNodeScan(n) → ProcedureCall`, so the scan has a parent. The `unwrap` remains (hazard).
- `evaluate_id_filter` stops at the first empty conjunct, so a type error in a later one is
  masked: `id(n) = 100 AND id(n) = 'x'` → 0 rows, swapped → error (`probe_id_seek_error_order`).
- `count(*)` with unused edge variables collapses parallel/multi-type edges (`(d:B)-[e:R|S]->(a)`
  count(*) 4 vs 5 rows) — shared with C; belongs to `reduce_bound_edge` (another wave's target).
- C-only: `MATCH (c)-[e0:R {w:0}]-(b)-[e1:R]-(b) WHERE id(c) < 0` hangs the C module; C
  ignores relationship uniqueness across comma-separated patterns (most `C-WRONG` fuzz hits).

## Modelling gaps
- Floats, NaN, ±0, Point/geo/distance, Maps and parameters are not modelled; strings are
  `Nat` codes and RediSearch lex order is taken as Cypher string order (the TAG encoding is
  not order-preserving — see `proofs/index_layer` bug 7).
- RediSearch node semantics are *definitions* in `build` (numeric range, exact tag, lex
  range, union skips null, intersection nulls on null), not proven against RediSearch.
- Stored ints ≥ 2^53 (f64 rounding on write) are treated exactly.
- `firstP` is leftmost-DFS, `extract_attribute_from_subtree` is BFS: they agree on every
  subtree used here (one property per depth).
- Scan selection is proven in a row-set model: which operator binds a variable is only
  modelled for filter placement (`runScoped`); `CondTraverseOp` expansion, `IncludePending`,
  `Argument`, var-len traversal and the endpoint *scores* (cost only) are not modelled.
- Edge index scans (`EdgeByIndexScan`, `prune_all_node_scan_child`, `add_to_labels_filter`)
  were NOT COVERED at the time (wave 4 below covers them).

## Wave 4 additions (origin/main 3fec7d7c9)
- `EdgeInline`: both `IndexSubject` impls (`nMatch_spec`, `eMatch_spec`), the edge headline
  `edge_utilize_sound` (EdgeByIndexScan + bound-endpoint filter = Filter → CondTraverse),
  the inline-attribute path (now historical `pre2390_applyInline_*`), `distance()`
  scans (`distance_sound`: sound with a complete GEO filter; the Filter is always kept),
  and `NodeByIndexScanOp` (`scanRow_eq_sel`, `evalIQE_*`, `canUtilize_empty`).
- `Cleanup`: BFS walkers (`mem_bfs`, `extractAttr_*`, `hasPropOf_iff`, `nonIdxD_spec`),
  `pre2390_refsVar_iff` (historical), `prune_sound` (dropping the AllNodeScan child keeps the edge multiset),
  `addToLabels_idem`, the fixed-point driver (`untilStable_preserves`, `untilStable_stable`),
  `tryIndexRewrite_sound`, `utilizeIndex_sound`.
- `ScanTree`: `select_scan_node` helpers on a path-addressed rose tree (`nodePath_eq`,
  `resolve_nodePath`, `pruneAt_parent`, `plannerScan_agree`, `makeScanSubtree_planner`,
  `collectLoop_eq`, `inMergeLoop_spec`, `childSubtreeBinds_spec`), and
  `vl_reverse_sound` (reversing a var-length leaf binds the same pairs).
  Remark (not a bug): `score_bound_not_dominant` — a filtered+attributed+labelled endpoint (5)
  outscores a bare bound one (3), contrary to the "bound has highest priority" doc comment.
- Hazard: `prune_all_node_scan_child` prunes the AllNodeScan *subtree*; `prune_sound` assumes
  the scan is a leaf (no live query found where it is not).

## Re-target to 8743953a8 (#2390 d2c42e032: plan an inline property map once)
The planner now lowers inline attributes to `IR::Filter`s and strips them from the pattern; the
optimizer's own inline paths are gone. Model changes (every citation re-mapped):
- `ScanTree`: `score_endpoint` returns `(score, filter_runs_late, cardinality)` and no longer adds
  2 for attributes (`scoreEndpoint_spec`; `score_bound_dominant`: a bound endpoint is never
  outscored now; the old double count is `pre2390_score_bound_not_dominant`);
  `collect_filtered_vars` returns `above` and `all` (the downward spine: `spineVars_eq`,
  `filteredVars_above_sub`); `make_scan_subtree` takes salvaged filters and `filters_of` collects
  them (`makeScanSubtree_planner`, `filtersOf_wrap`, `filtersOf_makeScanSubtree`);
  `select_var_len_scan_node` (`vlNonLeaf_keeps_wrappers`, `vlLeaf_guards`).
- `ScanOrder`: salvaged filters re-attached above the re-ordered chain are well-scoped
  (`salvage_wellScoped`, `boundAfter_hop`).
- `Cleanup`: `governing_filter` (`governingFilter_spec`), `match_scan_with_filter`, and
  `apply_filter_pushdown`'s `over_pending` (`utilizeP_false`, `utilizeP_true`, `overPending_sound`:
  over `IncludePending` the kept Filter makes the index scan exact for any pending set;
  `overPending_needs_filter`: dropping it would not be); `try_index_rewrite` has only the filter
  path (`tryIndexRewrite_sound`, `tryIndexRewrite_pending_sound`). Removed fns
  (`get_inline_attr_index`, `needs_inline_post_filter`, `apply_inline_rewrite`, `inline_attrs`,
  `index_query_references_var`, `references_var`) are kept as `pre2390_*` history;
  `pre2390_applyInline_eq_utilize` shows the inline path was already `utilize` of the lowered filter.
- `IdSeek`: `get_id_filter` compares `(id, scope_id)` and asks `subtree_references_variable`
  (`getIdFilter_spec`, `getIdFilter_other_scope`).

NEW CONFIRMED BUG (pre-existing; present before and after #2390): `select_var_len_scan_node`'s leaf
rewrite prunes the whole child subtree (`select_scan_node.rs:661`), including a `Filter` wrapper that
`planner_scan_alias` looked through. `MATCH (a) WHERE a.v = 1 MATCH (a)-[*1..2]->(b:B {w:2})
RETURN a.v, b.w` on `(:A {v:1})-[:R]->(:B {w:2}), (:A {v:5})-[:R]->(:B {w:2})` returns
`[1,2],[5,2]` (should be `[1,2]`; the fixed-length `-[:R]->` form is right) on builds 88e92ae25
and b3582e34a. `ScanTree.vl_leaf_fires_over_filter`, `vl_leaf_drops_filter`. Fix: refuse the leaf
rewrite when the wrapper holds a `Filter`, or salvage it above the traverse with `filters_of`.
-/
import OptimizerScan.IdSeek
import OptimizerScan.Index
import OptimizerScan.IndexCex
import OptimizerScan.IndexSound
import OptimizerScan.IndexPipeline
import OptimizerScan.ScanOrder
import OptimizerScan.EdgeInline
import OptimizerScan.Cleanup
import OptimizerScan.ScanTree
