# optimizer_scan: how FalkorDB picks the scan that starts a MATCH

This project proves that the optimizer passes which choose how a pattern is scanned return the same nodes as the plain plan they replace. It also gives concrete counterexamples where they do not. Every `MATCH` goes through these passes:

- **`select_scan_node`** picks the node to start from and the order to walk the hops.
- **`utilize_index`** replaces `Filter → NodeByLabelScan` with an index scan when a range or fulltext index covers the property.
- **`utilize_node_by_id`** turns `WHERE id(n) = 5` (and `<`, `>`, …) into a NodeByIdSeek.
- **`reorder_labels`** puts the cheapest label first.

It also covers the runtime pieces these rewrites hand their work to: `Runtime::evaluate_id_filter`, `NodeByIndexScanOp` and `Index::build_query_node`. The other optimizer passes are in [`../optimizer_rewrites`](../optimizer_rewrites). The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`OptimizerScan.lean`](OptimizerScan.lean). This README explains *what* is proven and *why it matters*.

## What "correct" means here

A plan is a membership test on nodes: each scan yields a node at most once, and a hop chain is the conjunction of its hops. A rewrite is correct if it selects exactly the nodes that `Filter → NodeByLabelScan` (or `Filter → AllNodeScan`) selects under Cypher's three-valued comparison, where only `true` keeps a row.

## The questions this project answers

**1. Does an index scan return the same nodes as the filter it replaces?**

- **On the shapes the proof covers, yes.** `utilize_sound` is the headline. It covers a single-label scan whose property values are Null, Int or String, and filters built from `n.k op literal`, `literal op n.k`, `n.k IN [literals]` and unindexable conjuncts, combined by AND or OR. For those, the rewritten plan selects exactly the nodes the original plan does.
- The building blocks are proven one by one: `sel_eq` (equality), `sel_range` (one- and two-sided ranges), `sel_inList` (`IN`; the list reaches the index whole as `IndexQuery::InList` since #2383, `inList_same_as_pre2383`), `sel_and2` (an index AND is the intersection), `sel_merge` (merging range conditions), and `pushAnd_spec`/`pushOr_spec` (the AND and OR branches of `try_filter_pushdown`).
- **Outside those shapes, no.** Each hypothesis of `utilize_sound` is there because the `IndexCex.lean` examples break the result without it: a computed constant ≥ 2⁵², a computed left side of `IN`, a list of properties on the right of `IN`, two labels, Bool or Date values. See [Bugs found](#bugs-found).

**2. Does the id seek return the same nodes as `WHERE id(n) …`?**

- **With non-negative integer bounds, yes — on any graph since #3394.** `seek_sound` proves `NodeByIdSeek` equals `Filter(id(n) op v …)` over `AllNodeScan` on every well-formed graph (`node_count` counts the live ids), including one that never held a node (main b84473a7f).
- `step_some`, `step_none` and `step_nonempty` prove each arm of `evaluate_id_filter` keeps exactly the ids that satisfy it, and the running range never inverts.
- **On a graph that never held a node, not before #3394.** `MATCH (n) WHERE id(n) = 0 RETURN n` returned a phantom node 0, and writes to it persisted into the next `CREATE` (`pre3394_phantom_node0`). #3394 returns no range when `node_count() == 0` (`empty_graph_seek_none`).

**3. Does reordering the hops keep the rows?**

- **Yes.** `swap_holds` proves reversing a hop and flipping its `transposed` flag tests the same edge. `order_sound` and `pick_perm` prove the greedy hop ordering keeps every hop exactly once, correctly oriented. `run_perm` proves the order of the tests does not change the selected rows.
- `vl_reverse_sound` proves reversing a variable-length leaf binds the same (start, end) pairs.

**4. Is every filter placed after the variables it reads are bound?**

- **If the starting set of bound variables is honest, yes.** `place_wellScoped` and `runScoped_wellScoped` prove filter re-placement is sound **exactly when** the seed set only contains variables that really are bound. `salvage_wellScoped` covers filters re-attached above a re-ordered chain.
- **On main, the seed is not always honest.** `select_scan_node` adds the start node to the seed even when it keeps an outer child and builds no scan for it. So `MATCH (b:A) WITH b LIMIT 1 MATCH (b)<-[]-(d)-[:R]->(c {v:1})-[:R]->(z) RETURN count(*)` returns 0; C returns 1.

**5. Do `reorder_labels` and the primary-label choice keep the scan?**

- **Yes.** `reorderLabels_sound` proves reordering labels cannot change the result. `withPrimary_sound` and `primary_split` prove scanning the indexed label first and checking the rest equals checking the full label set.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `utilize_sound` | On single-label scans with Null/Int/String values and the listed filter shapes, the index rewrite selects exactly what the filter does | `IndexPipeline.lean` |
| `sel_eq`, `sel_range`, `sel_inList`, `sel_and2`, `sel_merge` | The pieces: equality, ranges, `IN`, AND, merged ranges | `IndexSound.lean` |
| `trySingle_spec`, `pushAnd_spec`, `pushOr_spec` | Each branch of the index pushdown selects the right condition | `IndexSound.lean`, `IndexPipeline.lean` |
| `edge_utilize_sound` | EdgeByIndexScan plus the bound-endpoint filter equals Filter → CondTraverse | `EdgeInline.lean` |
| `distance_sound` | A `distance()` index scan is sound, and the Filter is always kept | `EdgeInline.lean` |
| `utilizeIndex_sound`, `tryIndexRewrite_sound` | The whole `utilize_index` driver, including its fixed-point loop, preserves results | `Cleanup.lean` |
| `overPending_sound` | Over `IncludePending` the kept Filter makes the index scan exact for any set of pending writes | `Cleanup.lean` |
| `prune_sound` | Dropping the AllNodeScan child of an edge index scan keeps the edges | `Cleanup.lean` |
| `seek_sound` | NodeByIdSeek equals the id filter on every well-formed graph (empty included, since #3394) with bounds in `[0, 2⁶³)` | `IdSeek.lean` |
| `empty_graph_seek_none` | a graph that never held a node seeks nothing (#3394) | `IdSeek.lean` |
| `order_sound`, `run_perm`, `swap_holds` | Hop ordering and orientation keep the rows | `ScanOrder.lean` |
| `place_wellScoped`, `salvage_wellScoped` | Filters are placed after their variables are bound, given an honest seed | `ScanOrder.lean` |
| `reorderLabels_sound`, `withPrimary_sound` | Label order and the primary-label choice cannot change a scan | `ScanOrder.lean` |
| `score_bound_dominant` | Since #2390 a bound endpoint is never outscored as the start node | `ScanTree.lean` |
| `vl_reverse_sound` | Reversing a var-length leaf binds the same pairs | `ScanTree.lean` |

### How `utilize_sound` is proved

1. **Model the values and the index.** A node has labels and properties. Values are Null, Int, String, plus Bool and Date for the counterexamples. `Faithful n` says every stored value is Null, Int or String. Literals must be `goodV`: a String, or an Int the index can store without losing precision (`lossy` excludes the large ones).
2. **Model RediSearch as definitions.** `build` turns an index query into a node predicate: numeric range, exact tag, lexical range, union skips null, intersection is null on null. These are *definitions* of what RediSearch does, not proven against RediSearch.
3. **Model the rewrite.** `utilize` follows `utilize_index.rs`: `trySingle` for one atom, `pushAnd` and `pushOr` for AND and OR, `mergeRange` for two bounds on one property. `reference` is the original `Filter → NodeByLabelScan`.
4. **Split on the filter shape.** `goodF` lists the allowed shapes; AND additionally excludes a strict string bound. For a node without the label both sides are false. For a node with it, each atom reduces to one of the `sel_*` lemmas, and AND and OR reduce to `pushAnd_spec` and `pushOr_spec`.
5. **Hypotheses.** `goodF` and `Faithful` are real restrictions, and each one is needed: the counterexamples in `IndexCex.lean` break the theorem when it is dropped. The strict-string-bound exclusion was needed before #3072 and could now be weakened.

## Bugs found

Each bug has a Rust repro (`cargo test -p graph --test lean_optimizer_scan`, not committed; `bug_*` tests fail today by design). Index repros need RediSearch, so they run live in [`repro.py`](repro.py) against a Rust module and a C module (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Counterexample | What it means | Status |
|---|---|---|---|
| W2-scan-1 | `place` examples over `exHops` in `ScanOrder.lean` (honest seed passes, seed with `best_node` fails) | A filter runs before its variable is bound: the `MATCH (b:A) WITH b LIMIT 1 …` query above returns 0, C 1 | **Open** |
| W2-scan-2 (#3353) | `seek_sound`, `empty_graph_seek_none`; history `pre3394_phantom_node0`, `pre3394_seek_sound` | On a graph that never held a node, `id(n) = 0` returned a phantom node. `SET` on it leaked into the next `CREATE`, `DELETE` errored | **Fixed by #3394** (9212add39, on main b84473a7f) |
| W2-scan-3 | `IndexCex` C4 | A computed constant ≥ 2⁵² (`n.v = 4503599627370495 + 4503599627370495 + 3`): the Filter is dropped and the op falls back to a label scan, returning every node (C: 0) | **Open** |
| W2-scan-4 | `IndexCex` C7, C7' | `abs(n.a) IN [1]` is pushed as `n.a IN [1]` and misses `a = -1`; `toString(n.a) IN ['1']` returns `[]` (C: `[1]`) | **Open** |
| W2-scan-5 | `IndexCex` C8 | `2 IN [n.a, n.b]` becomes array-contains on `n.a`: `[]` (C: `[3]`) | **Open** |
| W7-scan-1 | `vl_leaf_fires_over_filter`, `vl_leaf_drops_filter` | The var-length leaf rewrite prunes a Filter wrapper: `MATCH (a) WHERE a.v = 1 MATCH (a)-[*1..2]->(b:B {w:2})` returns an extra row | **Open** |
| — | `IdSeek.lean` examples on `g5` | `id(n) > -1` returns 0 rows, and non-integer bounds error. Reported as bugs 3–4 of [`../optimizer_rewrites`](../optimizer_rewrites) | **Open** |

The `IndexCex.lean` examples also reproduce bugs owned by [`../index_layer`](../index_layer): a folded `date()` constant returns every node (C3); `IN [.., date()]` drops the item (C9); multi-label AND/OR (C5, C6); Bool/Int conflation (C1, shared with C); array-contains inside AND (C10, shared with C). C2 (temporals indexed as numbers, #3076) and C11 (`n.v > 'B' AND n.v < 'B'`, #3072) are fixed; their examples now state agreement.

**Hazard, not a confirmed bug:** `utilize_node_by_id` calls `parent().unwrap()` on the scan it rewrites. W4-plan-4's repro (`CALL db.labels() YIELD label MATCH (n)`) is rejected by validation since #3297 (and had stopped crashing before that, because the scan has a parent), but the `unwrap` remains. [`../optimizer_rewrites`](../optimizer_rewrites) models it as `root_scan_has_no_parent`.

There are no CONDITIONAL rows in this project.

## Files

| File | What's in it |
|---|---|
| `OptimizerScan.lean` | The report header and imports |
| `IdSeek.lean` | `get_id_filter`, the flip table, `evaluate_id_filter` (`step`, `fold`, `evalIdFilter`, `evalIdFilterMain`), `seek` vs `reference`, `seek_sound`, `pre3394_*` history, the id-seek examples |
| `Index.lean` | The value and node model, index encoding, query nodes (`build`), `utilize_index`'s helpers and rewrite (`utilize`) |
| `IndexSound.lean` | `sel_*` lemmas for each query node, `trySingle_spec` |
| `IndexPipeline.lean` | The AND/OR pushdown, `goodF`, and the headline `utilize_sound` |
| `IndexCex.lean` | The `decide`d counterexamples C1–C11 (namespace `Cex`) |
| `EdgeInline.lean` | Edge index scans, `distance()` scans, `NodeByIndexScanOp`, the pre-#2390 inline-attribute path as history |
| `Cleanup.lean` | BFS walkers, `prune_all_node_scan_child`, `add_to_labels_filter`, the fixed-point driver, `IncludePending` handling, `utilizeIndex_sound` |
| `ScanOrder.lean` | Hop orientation and ordering, filter placement (`place`, `runScoped`), `reorder_labels`, the primary label |
| `ScanTree.lean` | `select_scan_node` helpers on a path-addressed tree: endpoint scores, `make_scan_subtree`, `filters_of`, `select_var_len_scan_node` |
| `repro.py` | Live repros for the index bugs, Rust vs C |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 72 rows: `select_scan_node.rs`, `utilize_index.rs`, `utilize_node_by_id.rs`, `reorder_labels.rs`, `node_by_index_scan.rs`, one function of `runtime.rs` and one of `index/mod.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 71 |
| MODELLED | 1 |

- **MODELLED:** `Index::build_query_node` in `index/mod.rs`, a support function, not a target file. The RediSearch semantics of its query nodes are definitions here (`build`), used by `sel_eq`, `sel_range` and `sel_and2`.

## What the model does not cover

- Floats, NaN, ±0, Point/geo values, maps and parameters are not modelled. Strings are numeric codes, and RediSearch's lexical order is taken to be Cypher's string order (the TAG encoding is not order-preserving; see [`../index_layer`](../index_layer) bug 7).
- RediSearch's query semantics are definitions, not proven against RediSearch.
- Stored integers ≥ 2⁵³ (rounded to f64 on write) are treated as exact.
- `firstP` searches depth first and the Rust `extract_attribute_from_subtree` breadth first. They agree on every subtree used here (one property per depth).
- Scan selection is proven in a row-set model. Which operator binds a variable is modelled only for filter placement (`runScoped`). `CondTraverseOp` expansion, `IncludePending`, `Argument`, var-length traversal and the endpoint *scores* (cost only) are not modelled.
- `prune_sound` assumes the pruned AllNodeScan is a leaf. The Rust prunes the whole subtree; no live query was found where it is not a leaf.

## Build and check

```bash
cd proofs/optimizer_scan && lake build       # this project
bash proofs/lean_ci.sh optimizer_scan        # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                      # repo-wide coverage report
venv/bin/python proofs/optimizer_scan/repro.py   # live index repros (Rust on 18200, C on 18201)
```

**When you change `select_scan_node.rs`, `utilize_index.rs`, `utilize_node_by_id.rs`, `reorder_labels.rs`, `node_by_index_scan.rs` or `evaluate_id_filter`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. If you widen what `utilize_index` accepts (a new value type, a new filter shape), extend `goodF` or `Faithful` and add the case to `utilize_sound`, or add an `IndexCex` example showing why it stays out.
