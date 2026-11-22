/-
# FalkorDB-rs scan selection passes: Lean 4 model, proofs, bug report

Targets: `graph/src/planner/optimizer/select_scan_node.rs`, `utilize_index.rs`,
`utilize_node_by_id.rs`, `reorder_labels.rs`, plus the runtime pieces their rewrites hand
work to (`Runtime::evaluate_id_filter`, `NodeByIndexScanOp`, `Index::build_query_node`).

**Re-synced to main 256ab4d3f (2026-11-22)**: new `IndexArray` (the array-contains arm of `try_filter_pushdown`, `utilize_sound_ac`, and the OR branch that keeps no filter: bug 7); `faithV` now admits list-valued properties (the array tree), every existing theorem unchanged. Table pins below are the function starts at 256ab4d3f; COVERAGE.tsv is authoritative for every row. IdSeek/IndexSound/ScanOrder header cites that point *inside* a body (e.g. `select_scan_node.rs`) keep the line of the revision they were written for.

Build: `lake build` (Lean 4.36, core only). No `sorry`, `admit`, `axiom`, `native_decide`;
`utilize_sound` and `seek_sound` depend only on `propext`, `Quot.sound`, `Classical.choice`.
77 theorems, 36 `decide`d examples (counterexamples and sanity checks).
Coverage per Rust function: `COVERAGE.tsv`. Rust repros: `graph/tests/lean_optimizer_scan.rs`
(`cargo test -p graph --test lean_optimizer_scan`; `bug_*` tests FAIL today by design).
Index repros need RediSearch, so they run live: `venv/bin/python proofs/optimizer_scan/repro.py`
(Rust release module on port 18200, C `bin/macos-arm64v8-release/falkordb.so` on 18201).

Reference semantics: a plan is a membership test on nodes (each scan yields a node at most
once), a hop chain is the conjunction of its hops, and the rewrite must select exactly what
`Filter → NodeByLabelScan` / `Filter → AllNodeScan` selects under Cypher's three-valued
comparison (only `true` keeps a row).

Re-targeted to origin/main 5e475e6c9: #2383 keeps an `IN [...]` list whole as `IndexQuery::InList`
(node and edge index scans) instead of an `Or` of `Equal`s; `Index.Q` gained `inList`, `evalIQ`
builds it, `canUtilize` has its arm, and the new query builds the same union and is accepted/refused by `can_utilize_index` exactly as the old `Or`
(the equivalence is proved as `build_inList_eq_or` and `canUtilize_inList_eq_or` in `Index.lean`). `utilize_sound` is unchanged in statement. #3436 only
shifts line pins (`Runtime::graph()`); `evaluate_id_filter` moved to `runtime.rs`.

## Lean ↔ Rust

| Lean | Rust |
| --- | --- |
| `IdSeek.getIdFilter`, `Op.flip`, `collectFilters` | `utilize_node_by_id.rs` |
| `IdSeek.step`, `evalIdFilter`, `asU64` | `runtime/runtime.rs` (`id as u64` at 1463) |
| `IdSeek.evalIdFilterMain` | `runtime/runtime.rs` (`node_count() == 0 → Ok(None)`) |
| `IdSeek.maxNodeId`, `seek` (`pre3394_seek` is the version that needs `nodeCount ≠ 0`, kept as a stepping stone) | `graph/graph.rs` (→ `IdSpace::max_id`), `runtime/ops/node_by_id_seek.rs` (`range -= deleted_nodes()` at :67) |
| `IdSeek.idMax` (`pre2979_seek` is the earlier range loop, kept as history) | `runtime/runtime.rs`: the loop starts at `node_id_bound().saturating_sub(1)`; `seek_eq_pre2979` |
| `Index.enc`, `arrEnc` | `index/mod.rs` (`Document::set`) |
| `Index.build`/`buildAll`/`buildSome`, `bsel`, `idxSel` | `index/mod.rs`, `Index::query` 1690 |
| `Index.canUtilize`, `evalIQ` | `runtime/ops/node_by_index_scan.rs` |
| `Index.hasT`, `firstP`, `nonIdxT`, `needsPost` | `utilize_index.rs` |
| `Index.buildOp`, `trySingle`, `mergeRange` | `utilize_index.rs` |
| `IndexArray.contains_no_fn`, `pushAnd_spec_ac`, `utilize_sound_ac` | `is_array_contains` (`utilize_index.rs`): the array-contains conjunct pushed as `ArrayContains` with its filter kept |
| `Index.pushAnd`/`mergeInto`, `isArrayContains`, `pushOr`, `tryPushdown` | `utilize_index.rs` |
| `Index.utilize`, `Plan.sel` | `utilize_index.rs`, `node_by_index_scan.rs` |
| `ScanOrder.Hop.swap`, `pick`, `order` | `select_scan_node.rs` |
| `ScanOrder.place`, `runScoped`, `initialBound` | `select_scan_node.rs` |
| `ScanOrder.reorderLabels`, `withPrimary` | `reorder_labels.rs`, `utilize_index.rs` |

## Proven (plain English)

Id seeks (`IdSeek.lean`):
- `flip_correct`, `flip_flip`: the `id()`-on-the-right flip table is right.
- `step_none`, `step_some`: each `evaluate_id_filter` arm keeps exactly the ids of the running range that satisfy it.
- `step_nonempty`: the running `min ≤ max` invariant holds, so `insert_range(min..=max)` is never inverted.
- `fold_none`, `fold_some`, `go_eq_fold`, `asU64_of_small`: the whole fold is exact for operands in `[0, 2^63)`.
- `seek_sound`: on any well-formed graph (`G.wf`: `node_count` counts the live ids), including one that never held a node, with non-negative integer operands, `NodeByIdSeek` = `Filter(id(n) op v …)` over `AllNodeScan`. `empty_graph_seek_none`: a never-populated graph seeks nothing. The version that needs `nodeCount ≠ 0` survives as `pre3394_seek_sound` in `IdSeek.lean`, the stepping stone of `seek_sound`.

Index utilization (`Index.lean`, `IndexSound.lean`, `IndexPipeline.lean`, `IndexArray.lean`):
- `contains_no_fn`, `pushAnd_spec_ac`, `utilize_sound_ac`: the index never loses a row of `x IN n.k`, and an array-contains conjunct (alone or in an AND) is rewritten soundly because its filter is kept; `tryPushdown_or_ac_drops_filter` is the OR counterexample (bug 7).
- `ordI_swap`, `ord3_swap`, `eq3_swap`, `flip_correct`: comparison flip when the property is on the right.
- `sel_eq`: `Equal` on a faithful literal (non-lossy Int, String) and faithful stored value (Null/Int/String) is exact.
- `sel_range`: every one- or two-sided numeric or string range is exact except the equal-string-bounds-with-an-exclusive-flag trap.
- `sel_buildOp`: `build_op_query` is exact for all five operators.
- `sel_inList`, `buildSome_eqs`, `cu_inList`: `n.k IN [literals]` is exact and runtime-usable when non-empty.
- `sel_and2`: an index `And` is the intersection (a null child nulls both sides).
- `sel_merge`, `rangeOK_merge`, `cu_merge`: `merge_range_queries` selects the conjunction and stays well-formed and usable.
- `trySingle_spec`: `try_single_filter_scan` on covered atoms is exact, on `L`, well-formed, usable unless `IN`.
- `trySingle_inn_bare`: an `IN` is pushed only when its index side is the bare property (`n.k IN [...]` or `x IN n.k`).
- `pushAnd_spec`, `pushOr_spec`, `buildSome_any`: the AND / OR branches of `try_filter_pushdown` select the conjunction and disjunction.
- `utilize_sound` (headline): single-label scan, Null/Int/String values, atoms `n.k op lit`, `lit op n.k`,
  `n.k IN [lits]`, unindexable conjuncts, combined by AND (no strict string bound) or OR — rewritten plan = original.

Scan selection (`ScanOrder.lean`):
- `swap_holds`, `swap_swap`: reversing a hop and flipping `transposed` tests the same edge.
- `run_perm`, `run_append`: reordering or re-sequencing tests keeps the selected rows.
- `pick_perm`, `order_sound`, `order_length`: the greedy hop ordering keeps every hop exactly once, oriented soundly.
- `runScoped_wellScoped`, `place_wellScoped`, `wellScoped_filts`: filter re-placement is sound **iff** the seed bound set is honest.
- `initialBound_wellScoped`, `initialBound_keeps_row`: `initial_bound` is honest — `{best_node}` only without a kept child, else the child's variables — so re-placement is well-scoped; the dishonest-seed counterexample is kept in `ScanOrder.lean` and finds its row here.
- `hasAll_perm`, `sortBy_perm`, `insertBy_perm`, `reorderLabels_sound`: `reorder_labels` cannot change a scan.
- `withPrimary_sound`, `primary_split`: putting the indexed label first + checking the rest = the full label set.

## CONFIRMED bugs (new in this wave)

3. **Computed constants ≥ 2^52 return every node** — `is_non_indexable_subexpr`
   (`utilize_index.rs`) flags only literal lossy ints; `4503599627370495 +
   4503599627370495 + 3` is not flagged, the Filter is dropped, `can_utilize_index` rejects
   the runtime value and the op falls back to a label scan with no Filter. Lean `IndexCex` C4.
   `MATCH (n:L) WHERE n.v = 4503599627370495 + 4503599627370495 + 3` → Rust all 6 nodes,
   Rust without index 0, C 0 (same for `n.v > 4503599627370495 * 2`). Fix: keep the Filter
   whenever the op may fall back (any non-literal value side), or have the fallback re-apply it.
6. **`select_scan_node` salvages a filter whose variable the pruned child bound (`Unwind`/`WITH`)** (W2-scan-6, #2972, fixed by open PR #3099) —
   `is_planner_scan_subtree` accepts a scan with *any* child; `argument_leaf_of`
   keeps only an `Argument`, so the rest of the subtree (an `Unwind` that the planner stitched
   below the scan) is pruned while the `Filter` above the scan is salvaged and
   re-attached above the chain (`filters_of`). The filter reads `k`, which nothing in the
   rebuilt plan binds. Lean `Salvage.premise_fails`, `salvage_loses_row`, `rebuilt_drops_unwind`.
   `WITH 1 AS k MATCH (b:B) WHERE b.v = k MATCH (b)-[:H]->(c:C) RETURN count(*)` on
   `(:B {v:1})-[:H]->(:C {v:1}), (:B {v:2})-[:H]->(:C {v:5})`: Rust 0, C 1.
   `UNWIND [1,1] AS k MATCH (b:B) WHERE b.v = 1 MATCH (b)-[:H]->(c:C) RETURN count(*)`:
   Rust 1, C 2 (EXPLAIN has no `Unwind`). Fix: treat a scan as planner-added only when its
   children are empty or a single `Argument`; otherwise leave the subtree alone.

## Also reproduced here but already reported by parallel wave-2 agents
- `proofs/index_layer` bugs 1, 3, 4, 5, 6, 8 (the open ones: see its README): folded `date()` constant returns every node
  (C3); `IN [.., date()]` drops the item (C9); multi-label AND → nothing / OR loses rows
  (C5, C6); Bool/Int conflation (C1, shared with C). `utilize_sound` still assumes no temporal
  values and no strict string bound in an AND (gap: those hypotheses could now be weakened).
- `proofs/optimizer_rewrites` bugs 3, 4: `id(n) > -1` / `>= -5` / `< -1` / `<= -1` via
  `id as u64` (`bug_id_seek_negative_operand` here), non-integer id operands error.
7. **The OR branch of `try_filter_pushdown` keeps no filter for an array-contains disjunct** — the same
   family as #3736 (which fixed AND and the single atom): `try_filter_pushdown` OR arm
   (`utilize_index.rs`) pushes `1 IN n.arr` as an `ArrayContains` and returns no leftover
   filter, `needs_post_filter` flags nothing, and the array tree answers `1 IN [true]` (Bool and Int
   share the numeric key). Lean `IndexArray.tryPushdown_or_ac_drops_filter`, `Cex C12` (by `decide`).
   Live at 256ab4d3f on `CREATE INDEX FOR (n:L) ON (n.arr, n.k)` and nodes `{arr:[true], k:0}`,
   `{arr:[1], k:0}`, `{arr:[2], k:9}`: `MATCH (n:L) WHERE 1 IN n.arr OR n.k = 9 RETURN n.arr, n.k`
   Rust with index `[true],0 / [1],0 / [2],9`, Rust without index and openCypher `[1],0 / [2],9`,
   C with index `[true],0 / [1],0 / [2],9` (C also returns `[true]` for `1 IN n.arr` alone, which
   Rust no longer does). EXPLAIN: `Sort → Project → Node By Index Scan` (no Filter). Fix: keep the
   original filter whenever any OR branch `is_array_contains`.

## Suspected / hazards (not confirmed as wrong results)
- `utilize_node_by_id.rs` `parent().unwrap()` panics if a scan is the plan root. W4-plan-4's
  repro (`CALL db.labels() YIELD label MATCH (n) RETURN label`) is rejected by validation, so the scan has a parent; the `unwrap` remains (hazard).
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
  the inline-attribute path (gone from the Rust; only an unchecked historical note remains), `distance()`
  scans (`distance_sound`: sound with a complete GEO filter; the Filter is always kept),
  and `NodeByIndexScanOp` (`scanRow_eq_sel`, `evalIQE_*`, `canUtilize_empty`).
- `Cleanup`: BFS walkers (`mem_bfs`, `extractAttr_*`, `hasPropOf_iff`, `nonIdxD_spec`),
  `prune_sound` (dropping the AllNodeScan child keeps the edge multiset),
  `addToLabels_idem`, the fixed-point driver (`untilStable_preserves`, `untilStable_stable`),
  `tryIndexRewrite_sound`, `utilizeIndex_sound`.
- `ScanTree`: `select_scan_node` helpers on a path-addressed rose tree (`nodePath_eq`,
  `resolve_nodePath`, `pruneAt_parent`, `plannerScan_agree`, `makeScanSubtree_planner`,
  `collectLoop_eq`, `inMergeLoop_spec`, `childSubtreeBinds_spec`), and
  `vl_reverse_sound` (reversing a var-length leaf binds the same pairs).
  Remark (not a bug): `score_bound_not_dominant` — a filtered+attributed+labelled endpoint (5)
  outscores a bare bound one (3), contrary to the "bound has highest priority" doc comment.
- `prune_all_node_scan_child` re-attaches the scan's children beneath the edge scan
  (`utilize_index.rs`); `Cleanup.prune_sound_rows` proves the rows are kept per input row.

## Outer-context child beneath the new scan (#3594)
- `make_scan_subtree` takes a whole subtree beneath the scan (`ScanTree.makeScanSubtree` with
  `Option PT`; `filtersOf_makeScanSubtree_tree`).
- `projected_outputs`/`collect_output_pairs` (new) and `collect_output_aliases`: a copy `(old, new)`
  outputs `new`, an aggregation its names and copies (`collectOutputAliases_spec`,
  `collectOutputAliases_aggregate`, `collectOutputAliases_projected`, `collectOutputPairs_spec`).
- `select_scan_node`'s outer-context child moved beneath the new scan (`transparent_child`, lines
  1005-1057) only adds, to the seed, ids that subtree binds beneath the scan (minus the chain's ids
  when transparent, lines 1198-1224): `ScanOrder.initialBoundOuter`, `initialBoundOuter_none`,
  `outerSeed_transparent_no_chain`, `initialBoundOuter_wellScoped`. The planner-scan path (`has_planner_scan`)
  is untouched, so `Salvage.premise_fails`/`salvage_loses_row` (#2972) still describe it
  (`rebuilt_drops_unwind` now with `argument_leaf_of(..).map(tree!)`).
- The no-swap leaf rewrite salvages Filters only from a discarded planner scan (`if has_planner_scan`,
  `select_scan_node.rs`, `17547791c`); a transparent child keeps its own beneath the new scan:
  `Salvage.leafSteps3594`, `leafSteps3594_planner`, `leafSteps3594_transparent_wellScoped`,
  `transparent_filters_once`.

## Re-target to 8743953a8 (#2390 d2c42e032: plan an inline property map once)
The planner now lowers inline attributes to `IR::Filter`s and strips them from the pattern; the
optimizer's own inline paths are gone. Model changes (every citation re-mapped):
- `ScanTree`: `score_endpoint` returns `(score, filter_runs_late, cardinality)` and no longer adds
  2 for attributes (`scoreEndpoint_spec`; `score_bound_dominant`: a bound endpoint is never
  outscored);
  `collect_filtered_vars` returns `above` and `all` (the downward spine: `spineVars_eq`,
  `filteredVars_above_sub`); `make_scan_subtree` takes salvaged filters and `filters_of` collects
  them (`makeScanSubtree_planner`, `filtersOf_wrap`, `filtersOf_makeScanSubtree`);
  `select_var_len_scan_node` (`vlNonLeaf_keeps_wrappers`, `vlLeaf_guards`).
- `ScanOrder`: salvaged filters re-attached above the re-ordered chain are well-scoped, given that they
  read only variables the chain binds (`salvage_wellScoped`, `boundAfter_hop`).
- `Salvage` (new): models `filters_of`'s output with the variables each filter reads, and takes up that
  premise: `salvaged_wellScoped`, `premise_of_spine` (a sufficient condition: the premise holds when the
  spine's filters read only the scanned node and the `Argument`'s variables); REFUTED for one concrete
  `Filter → NodeByLabelScan → Unwind` tree (`exChild_shape`, `rebuilt_drops_unwind`, `premise_fails`,
  `salvage_loses_row`, whose filter is the equality `b.v = k` over a property store, `exPred_is_eq`; bug 6 above).
- `Cleanup`: `governing_filter` (`governingFilter_spec`), `match_scan_with_filter`, and
  `apply_filter_pushdown`'s `over_pending` (`utilizeP_false`, `utilizeP_true`, `overPending_sound`:
  over `IncludePending` the kept Filter makes the index scan exact for any pending set;
  `overPending_needs_filter`: dropping it would not be); `try_index_rewrite` has only the filter
  path (`tryIndexRewrite_sound`, `tryIndexRewrite_pending_sound`). Removed fns
  (`get_inline_attr_index`, `needs_inline_post_filter`, `apply_inline_rewrite`, `inline_attrs`,
  `index_query_references_var`, `references_var`) are gone (the inline path was `utilize` of the lowered filter).
- `IdSeek`: `get_id_filter` compares `(id, scope_id)` and asks `subtree_references_variable`
  (`getIdFilter_spec`, `getIdFilter_other_scope`).

`select_var_len_scan_node`'s leaf rewrite refuses a `Filter` in the wrapper chain and any
operator under the scan other than a childless `Argument` (`select_scan_node.rs`). Model:
`VLCase.scanOnlyArgBelow`, `vlLeaf` vs `vlLeafPre` (the earlier rewrite, kept as history);
`vlLeaf_keeps_filter`, `vl_never_drops_filter`, `vlLeaf_eq_pre`, `vl_leaf_bug_case_fixed`
(`ScanTree.vl_leaf_fires_over_filter`, `vl_leaf_drops_filter` show what `vlLeafPre` did).

## PR #3593 (var-length leaf rewrite keeps children, allows undirected walks)
`select_var_len_scan_node` (`select_scan_node.rs`) no longer refuses an undirected walk (the
runtime now reverses it: `vlLeaf_bidir`, sound by `vl_reverse_sound_symm`), keeps every child of the
old scan under the new `to` scan (`leafScan_keeps_below`, `vlLeaf_over_input`), and refuses when the
edge attrs read an endpoint (`vlNonLeaf_attrs`) or a kept child may bind `to` (`subtree_may_bind`
:568, `subtreeMayBind_spec`, `childSubtreeBinds_mayBind`, `vl_leaf_never_rebinds_to`). The #3264
guards are `vlGuards3264`/`vlLeaf3264` (history; `vlLeaf3264_guards`, `vlLeaf_eq_3264`).

## PR #3821 (f71981fa3: equality-pinned endpoint tie-break)
`score_endpoint` returns `(score, has_equality, filter_runs_late, cardinality)`;
`collect_filtered_vars` also fills `equality` via the new `collect_equalities` (top-level `AND`
conjuncts `v.p = <variable-free expr>`, either side). `ScanTree.scoreEndpoint_spec` (re-stated with
the new flag), `collectEqualities_sub`/`_other`/`_examples`, `filteredVars_equality_sub_all`
(equality ⊆ all, so `score_equality_filtered`: the flag only breaks ties between filtered
endpoints), `equality_beats_late` (the comparator key `startKeyLt`). Start choice affects cost only;
`ScanOrder`'s soundness theorems hold for any start, so they are unchanged.
-/
import OptimizerScan.IdSeek
import OptimizerScan.Index
import OptimizerScan.IndexCex
import OptimizerScan.IndexSound
import OptimizerScan.IndexPipeline
import OptimizerScan.IndexArray
import OptimizerScan.ScanOrder
import OptimizerScan.EdgeInline
import OptimizerScan.Cleanup
import OptimizerScan.ScanTree
import OptimizerScan.Salvage
