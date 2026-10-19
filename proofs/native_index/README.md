# native_index: the native numeric index answers like a label scan plus a filter

This project proves that the native numeric index (#2350, with writes batched by #3307) returns exactly the rows a full label scan plus a `WHERE` filter would return, and that every graph write keeps the index in step with the data. The index is built only with `--features index-falkordb` (off by default in the shipped module); when it is on, it serves range and equality predicates on indexed numeric properties instead of RediSearch:

```cypher
CREATE INDEX FOR (p:Person) ON (p.age)
MATCH (p:Person) WHERE p.age > 30 RETURN p      -- Node By Index Scan, no post-filter
MATCH (p:Person) WHERE p.age = 42 RETURN p
```

`Node By Index Scan` does **not** re-check the predicate on the rows it returns, so the index has to be exactly right on its own. A value whose key sorts on the wrong side of the bound is a silently wrong query result, not a slow one.

The Rust is `graph/src/index/falkordb/{encode,numeric,falkordb_index,pending_writes,graph_writes}.rs`, the native-index parts of `graph/src/graph/graph.rs`, and the native route of `runtime/ops/{node,edge}_by_index_scan.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorNativeIndex.lean`](FalkorNativeIndex.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does the key encoding sort numbers the way Cypher compares them?**

Each indexed value becomes a `u64` key (`encode_f64`): set the sign bit of a non-negative double, flip all bits of a negative one, and map `-0.0` to `0.0`. `encodeF64_cmp` proves that for any two non-NaN doubles, comparing the keys gives exactly `f64::partial_cmp`: the encoding preserves *and* reflects order. `encodeF64_inj` proves two keys are equal only for the same bits or the two zeros. The model works on the real IEEE bit pattern, so it covers both zeros, every NaN sign and payload, and ±∞. NaN and the temporal types are not indexed at all (`nan_unindexed`, `temporal_unindexed`).

**2. Are integers safe, given they are stored as `f64`?**

An `Int` is indexed as `i as f64`, which loses precision above 2⁵³ (`p53_collision`: 2⁵³ and 2⁵³+1 get one key). The planner guards against this with `int_loses_f64_precision` on the query bound. `int_cmp_agrees` proves the guard is sound for bounds with `|b| < 2⁵²`: against such a bound, comparing the `f64` images gives the exact `i64` order for every stored integer. The guard has one hole, `i64::MIN` (see [Bugs found](#bugs-found)).

**3. Does an index scan return exactly what a label scan plus filter returns?**

Yes, for non-Bool data and a guarded numeric bound. `filter_iff_key` proves that for every data value (any variant, any NaN payload, `-0.0`, ±∞, any `i64`) and all five operators (`<`, `<=`, `>`, `>=`, `=`), the `WHERE` predicate holds iff the value is indexed and its key satisfies the same comparison against the bound's key. `range_keys` proves the inclusive key bounds `range` computes (including the `±1` steps for exclusive bounds) are exact. Then:

- `range_agrees` and `point_agrees`: a column built from `(value, id)` pairs answers a `Range` or an `Equal` with exactly the ids whose value passes the filter;
- `nodeScan_native_agrees`: the same, end to end, for `NodeByIndexScanOp` on the native route;
- `queryNumeric_spec`: only numeric `Equal`/`Range` queries on an existing column take the native route.

**4. Does batching writes per unit of work (#3307) change the result?**

No. #3307 stages index changes during a query and applies them once at commit: `last_write_wins` collapses each tuple's history, then one `remove_batch` and one `insert_batch` run per column. `lastWriteWins_spec` proves the collapse keeps exactly each tuple's last staged change, and `applyColumn_eq_sequential` proves the batched apply leaves the same tree as applying every staged change one by one, in order, for any interleaving of adds and removes.

**5. Do `SET`, `DELETE` and label changes keep the index in step?**

Yes. Every staging function in `graph_writes.rs` is proven to stage exactly its ops (`nodeAttrsSet_changes` and the other `*_changes` theorems). `lastOp_ops` then says what one unit of work does to any index entry: the last op for that `(id, attribute)` decides. Per path:

- `SET n.age = 5` removes the old entry and adds the new one (`nodeAttrsSet_hit`, `edgeAttrsSetWith_hit`);
- `DELETE n` removes every entry of `n` and adds none (`nodesDeleted_hit`, `nodesDeleted_noadd`, and the edge versions);
- `REMOVE n:Person` and `SET n:Person` remove or add the label's entries (`labelsRemoved_hit`, `labelsAdded_hit`);
- `CREATE` adds the new node's entries (`newNodeAttrs_hit`).

**6. Can a reader see a write committed after it started?**

No. `snapshot_isolated` proves a reader holding version `v` gets the same answer to every index query after any sequence of later units of work, refused or not. `runUnit_spec` proves a committed unit publishes exactly its private copy of the index with its staged changes applied once, and a refused unit publishes nothing.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `encodeF64_cmp` | For non-NaN doubles, key order = `f64::partial_cmp` (preserves and reflects order) | `Float.lean` |
| `encodeF64_inj` | Two keys are equal only for the same bits or `±0.0` | `Float.lean` |
| `int_cmp_agrees` | For a bound with `\|b\| < 2⁵²`, the `f64` image order is the exact `i64` order | `Value.lean` |
| `filter_iff_key` | The `WHERE` comparison holds iff the stored key satisfies it against the bound's key | `Agree.lean` |
| `range_agrees`, `point_agrees` | A built column answers `Range`/`Equal` with exactly the ids the filter keeps | `Numeric.lean` |
| `nodeScan_native_agrees` | Node index scan on the native route = label scan + filter | `Ddl.lean` |
| `lastWriteWins_spec` | The collapse keeps exactly each tuple's last staged change | `Pending.lean` |
| `applyColumn_eq_sequential` | #3307's batched apply = applying each staged change one by one | `Pending.lean` |
| `nodeAttrsSet_hit`, `nodesDeleted_hit` | An update replaces the old entry; a delete removes every entry | `Writes.lean` |
| `snapshot_isolated`, `runUnit_spec` | Readers keep their version's index; a commit publishes its staged changes once | `Ddl.lean` |

### How `range_agrees` is proved

1. **Model the doubles as bits.** An `f64` is its IEEE-754 encoding, a `Nat < 2⁶⁴` (`Float.lean`). `fpcmp` is IEEE comparison defined from the bits (sign-magnitude order, `±0` equal, NaN unordered). No float axiom is used.
2. **Model the integers.** `i64ToF64` is `i as f64`, built from the round-to-nearest-even `roundNat` in [`../core`](../core). `magBits_strictMono` proves the conversion exact and strictly increasing below 2⁵³.
3. **Model the values and the filter.** `Value` has `Int`, `Float`, `Bool`, the four temporals, `String`, `Null` and `other` for the rest. `compareValue` is `compare_value`, and `cmpPred` is the filter's comparison operators.
4. **Prove each comparison agrees with the keys** (`filter_iff_key`), splitting on the data variant: floats by `encodeF64_cmp`, ints by `int_cmp_agrees`, and every other variant is unindexed and never passes a numeric comparison.
5. **Prove the range bounds are exact** (`range_keys`). `encodeF64_bounds` proves every real key is strictly between 0 and `u64::MAX`, so the `checked_add`/`checked_sub` in the exclusive-bound steps never fire.
6. **Combine with the tree contents.** The tree is its sorted entry list, and `range` returns the entries between the two keys. `Contract.lean` proves each tree operation is exactly that list model, for every tree the index can hold (below).

The hypotheses on `range_agrees`:

- **No `Bool` data** (`NotBool`). A `Bool` shares a key with `0`/`1` (`bool_divergence`). This matches C and the RediSearch path, and is pinned in a Rust test.
- **A guarded bound** (`GoodBound`): a non-NaN float, or an int with `|b| < 2⁵²`. The planner only routes such bounds to the index, except for `i64::MIN`.
- **The tree contents** follow `cow_btree`'s specs. Those are proven under `TreeWF`, and since #3283 every tree the API can build satisfies it (`cow_btree` `Reachable_wf_asserted`), so this is no longer a hypothesis: `Contract.lean` (`contract_insert`, `contract_remove`, `contract_insertBatch`, `contract_removeBatch`, `contract_fromSorted`, `contract_range`) proves each Rust tree operation equals its `Tree.lean` model on every reachable tree.

## Bugs found

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `i64min_hole`, `i64min_window` | The guard lets `i64::MIN` through, but `i64::MIN+1 … i64::MIN+512` round to the same `f64`. In the model, a `>` range scan (which starts at `key + 1`) misses those rows and an `=` scan would include them. Confirmed live for `>`: `WHERE n.v > -9223372036854775808` dropped `i64::MIN+1` and `i64::MIN+100` | **Open.** Shared with C on `>` (C returns the same rows), so not a native-index regression; on `=` the live Rust returned only the correct row while C returned the extra ones. Suggested fix: reject `unsigned_abs() >= 1 << 52` |
| 2 | `bool_divergence` | `true` and `1` share a key, but `true = 1` is false in the filter | Harmless: C and the RediSearch path behave the same; pinned in `encode.rs` tests |

Related: `guard_rejects_2p53` and `p53_collision` show the guard correctly rejects 2⁵³, whose collision with 2⁵³+1 is real.

**Formerly CONDITIONAL rows.** 13 `numeric.rs` and `falkordb_index.rs` rows read through the tree, whose content specs come from [`../cow_btree`](../cow_btree) under its invariant `TreeWF`. `cow_btree` Bug 3 (W7-btree-1) showed `remove_batch` could break `TreeWF`, and since #3307 `remove_batch` runs in production (`apply_writes` → `remove_encoded`). #3283 (799968a8b) fixed it; `cow_btree` proves `remove_batch`, `insert_batch` and `from_sorted` keep or build well-formed trees (`removeBatchTree_wf`, `insertBatch_wf`, `fromSorted_wf`, together `Reachable_wf_asserted`), and `Contract.lean` here connects each operation to its model. These rows are now PROVEN.

## Files

| File | What's in it |
|---|---|
| `Float.lean` | `f64` as bits, `isNaN`, `fpcmp`, `encodeF64`; `encodeF64_cmp`, `encodeF64_inj`, `encodeF64_bounds` |
| `IntF.lean` | `i as f64` (`magBits`, `i64ToF64`), exact and monotone below 2⁵³ |
| `Value.lean` | `Value`, `compareValue`, `cmpPred`, `encodeNumeric`, the guard `intLosesPrecision`; `guard_sound`, `int_cmp_agrees` |
| `Agree.lean` | `filter_iff_key`; counterexamples `bool_divergence`, `i64min_hole`, `p53_collision` |
| `Tree.lean` | The `CowBTree` contract as its contents (`tInsert`, `tRemoveBatch`, `tRange`, …) |
| `Contract.lean` | The contract discharged: on every reachable tree each `CowBTree` operation (the `cow_btree` model) equals its `Tree.lean` model (`contract_*`) |
| `Numeric.lean` | `NumericIndex`, `rangeBounds`, `range_keys`, `range_agrees`, `point_agrees`, per-write contents specs |
| `Index.lean` | `HashMap` as an association list, `IndexColumn`, `ColumnEntry.ready`, `FalkorDbIndex` |
| `Pending.lean` | Staged writes (`IndexWrite`, `IndexChanges`, `PendingIndexWrites`), `lastWriteWins`, `applyColumn_eq_sequential` |
| `Writes.lean` | The staging functions of `graph_writes.rs` (`nodeAttrsSet`, `nodesDeleted`, `labelsAdded`, …), `lastOp_ops`, the per-path `*_hit` theorems |
| `Ddl.lean` | `createIndex`, `populate`, `buildColumn`, the scan routing `scanRoute`, `nodeScan_native_agrees`, `snapshot_isolated`, `runUnit_spec` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Path dependencies (`lakefile.toml`): [`../cow_btree`](../cow_btree) for the entry encoding `enc` and the sorted-list lemmas, and [`../core`](../core) for `roundNat`/`bitlen` (IEEE round-to-nearest-even).

## Coverage

`COVERAGE.tsv` has 83 function rows (plus one comment row):

| Bucket | Rows |
|---|---|
| PROVEN | 83 |

- The 13 rows that read or write the tree (`from_entries`, `add`, `remove`, `add_batch`, `remove_batch`, `add_encoded`, `remove_encoded`, `is_empty`, `point`, `range`, `empty`, and `FalkorDbIndex::apply` / `apply_writes`) were CONDITIONAL on the `cow_btree` invariant `TreeWF`, which `remove_batch` broke (W7-btree-1). #3283 fixed it, `cow_btree` now proves every reachable tree is well-formed, and `Contract.lean` ties each operation to its model, so they are PROVEN.
- Three PROVEN rows (`node_attrs_set`, `new_node_attrs`, `edge_attrs_set_with`) state their per-entry result (`*_hit`) under a caller premise: each node's written attribute ids are unique. `Pending::set_node_attribute` keeps them sorted and unique; `set_node_attributes` only `debug_assert!`s it.

## What the model does not cover

- **The tree itself.** The tree is its sorted content list; the tree code is proven in `cow_btree` (including that every reachable tree is well-formed) and `index_layer` (the leaf bytes).
- **Caller uniqueness.** The `*_hit` theorems assume each node's written attribute ids are unique. Created nodes' `new_labels` and bulk/replica record labels are taken to be the node's labels, which is the Rust's own premise (documented in `graph_writes.rs`).
- **The graph's own stores.** How the graph mutates its attribute store, label matrices and `remove_all` is modelled in other projects; here each per-entry outcome is stated against the graph *before* the write.
- **`HashMap` order.** Iteration order is arbitrary, so theorems about a whole map are stated per key. The `bulk`, `rows` and `product` paths have exact staged-ops theorems (`*_changes`); their per-entry outcome follows from `lastOp_ops` as for the other paths.
- **Schema.** Attribute names are assumed unique (`attrNames.Nodup`), which the single `attrs_name` dictionary guarantees.
- **RediSearch.** When a query is not routed natively, it goes to RediSearch through FFI, which is not modelled here.

## Build and check

```bash
cd proofs/native_index && lake build   # this project (also builds ../cow_btree and ../core)
bash proofs/lean_ci.sh native_index    # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `graph/src/index/falkordb/`, the native-index functions in `graph.rs`, or the native route of `node_by_index_scan.rs` / `edge_by_index_scan.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If you change the encoding or the precision guard, re-check `filter_iff_key`; it is the theorem that justifies having no post-filter. If you change `cow_btree`, rebuild this project too, since it imports `cow_btree`. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
