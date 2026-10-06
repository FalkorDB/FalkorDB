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
| `IdSeek.getIdFilter`, `Op.flip`, `collectFilters` | `utilize_node_by_id.rs:51-89, 118-131` |
| `IdSeek.step`, `evalIdFilter`, `asU64` | `runtime/runtime.rs:1309-1367` (`id as u64` at 1322) |
| `IdSeek.maxNodeId`, `seek` | `graph/graph.rs:1602` (→ `IdSpace::max_id`), `runtime/ops/node_by_id_seek.rs:63-73` |
| `Index.enc`, `arrEnc` | `index/mod.rs:716-846` (`Document::set`) |
| `Index.build`/`buildAll`/`buildSome`, `bsel`, `idxSel` | `index/mod.rs:1463-1674`, `Index::query` 1677 |
| `Index.canUtilize`, `evalIQ` | `runtime/ops/node_by_index_scan.rs:98-287` |
| `Index.hasT`, `firstP`, `nonIdxT`, `needsPost` | `utilize_index.rs:367-378, 557-605, 845-928` |
| `Index.buildOp`, `trySingle`, `mergeRange` | `utilize_index.rs:325-365, 612-714, 491-554` |
| `Index.pushAnd`/`mergeInto`, `pushOr`, `tryPushdown` | `utilize_index.rs:770-826` |
| `Index.utilize`, `Plan.sel` | `utilize_index.rs:995-1085`, `node_by_index_scan.rs:301-328` |
| `ScanOrder.Hop.swap`, `pick`, `order` | `select_scan_node.rs:294-311, 914-974` |
| `ScanOrder.place`, `runScoped` | `select_scan_node.rs:893-900, 1067-1117` |
| `ScanOrder.reorderLabels`, `withPrimary` | `reorder_labels.rs:16-47`, `utilize_index.rs:206-222, 296-319` |

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
   — `select_scan_node.rs:893-900` seeds `initial_bound` with `best_node` even when the
   outer-context child is kept (no scan for `best_node` is built, line 1080). Lean:
   `ScanOrder` counterexample after `place_wellScoped`. Query
   `MATCH (b:A) WITH b LIMIT 1 MATCH (b)<-[]-(d)-[:R]->(c {v:1})-[:R]->(z) RETURN count(*)`
   → Rust 0, C 1 (also `ORDER BY`/`SKIP`/`DISTINCT` children; `WITH b` alone is fine).
   Found by the oracle fuzz (`fuzz3.py`, 1 Rust-only mismatch in 480 queries). Fix: insert
   `best_node` into `initial_bound` only when `existing_child` is `None`.
   Test `bug_select_scan_node_places_filter_before_its_variable_is_bound`.
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
   (`utilize_index.rs:910-928`) flags only literal lossy ints; `4503599627370495 +
   4503599627370495 + 3` is not flagged, the Filter is dropped, `can_utilize_index` rejects
   the runtime value and the op falls back to a label scan with no Filter. Lean `IndexCex` C4.
   `MATCH (n:L) WHERE n.v = 4503599627370495 + 4503599627370495 + 3` → Rust all 6 nodes,
   Rust without index 0, C 0 (same for `n.v > 4503599627370495 * 2`). Fix: keep the Filter
   whenever the op may fall back (any non-literal value side), or have the fallback re-apply it.
4. **`IN` with a computed left side is pushed as `n.a IN [...]`** — `try_in_filter_scan`
   (`utilize_index.rs:630-655`) only asks that the side *contain* a property, and takes
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
  (C5, C6); temporals indexed as numbers (C2); `n.v > 'B' AND n.v < 'B'` exact-match trap
  (C11); Bool/Int conflation (C1, shared with C).
- `proofs/optimizer_rewrites` bugs 3, 4: `id(n) > -1` / `>= -5` / `< -1` / `<= -1` via
  `id as u64` (`bug_id_seek_negative_operand` here), non-integer id operands error.
- Array-contains keeps its Filter only at the root (`utilize_index.rs:816-823`): inside AND
  it is dropped, so `1 IN n.arr AND n.k > 0` returns `arr: [true]` (Bool/Int share the numeric
  array field). Rust with index `[4],[5]`, without `[5]`, C `[4],[5]` — index-dependent result
  shared with C (Lean C10).

## Suspected / hazards (not confirmed as wrong results)
- `utilize_node_by_id.rs:118` `parent().unwrap()` panics if a scan is the plan root (no
  query found that does this).
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
  and the inline-attribute path are NOT COVERED (see `COVERAGE.tsv`).

## Wave 4 additions (origin/main 3fec7d7c9)
- `EdgeInline`: both `IndexSubject` impls (`nMatch_spec`, `eMatch_spec`), the edge headline
  `edge_utilize_sound` (EdgeByIndexScan + bound-endpoint filter = Filter → CondTraverse),
  the inline-attribute path (`applyInline_eq_utilize`, `applyInline_sound`), `distance()`
  scans (`distance_sound`: sound with a complete GEO filter; the Filter is always kept),
  and `NodeByIndexScanOp` (`scanRow_eq_sel`, `evalIQE_*`, `canUtilize_empty`).
- `Cleanup`: BFS walkers (`mem_bfs`, `extractAttr_*`, `hasPropOf_iff`, `nonIdxD_spec`),
  `refsVar_iff`, `prune_sound` (dropping the AllNodeScan child keeps the edge multiset),
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
