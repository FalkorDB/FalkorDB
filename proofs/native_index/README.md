# native_index: the native numeric index answers like a label scan plus a filter

This project proves that the native numeric index (#2350, with writes batched by #3307 and a build state with indexer batches since #2351) returns exactly the rows a full label scan plus a `WHERE` filter would return, and models the graph-write staging paths (what each write stages for the index) under the caller premises (`*_hit`: unique attribute ids) and the tree contract documented below; the graph store's own mutation is not modelled. The index is built only with `--features index-falkordb` (off by default in the shipped module); when it is on, it is the **only** Range index (#3547, #2386): RediSearch is not fed Range fields and there is no fallthrough to it. It serves range and equality predicates, `IN` lists and same-attribute `OR`s on indexed numeric properties, answers a read on a column still being built by scanning, and raises an error for any predicate it cannot serve:

```cypher
CREATE INDEX FOR (p:Person) ON (p.age)
MATCH (p:Person) WHERE p.age > 30 RETURN p      -- Node By Index Scan, no post-filter
MATCH (p:Person) WHERE p.age = 42 RETURN p
```

`Node By Index Scan` does **not** re-check the predicate on the rows it returns, so the index has to be exactly right on its own. A value whose key sorts on the wrong side of the bound is a silently wrong query result, not a slow one.

Since #2383 the index also serves `IN [...]`, and an `OR` of equalities and `IN` lists on one attribute, as a union of point lookups (`MATCH (p:Person) WHERE p.age IN [30, 40]`).

The Rust is `graph/src/index/falkordb/{encode,numeric,falkordb_index,pending_writes,graph_writes,versioned,unique,mod}.rs`, the native-index parts of `graph/src/graph/graph.rs` (including `redisearch_fields`, `node_matches`, `relationship_matches`), and the index route of `graph/src/runtime/ops/{node,edge}_by_index_scan.rs`. The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorNativeIndex.lean`](FalkorNativeIndex.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does the key encoding sort numbers the way Cypher compares them?**

Each indexed value becomes a `u64` key (`encode_f64`): set the sign bit of a non-negative double, flip all bits of a negative one, and map `-0.0` to `0.0`. `encodeF64_cmp` proves that for any two non-NaN doubles, comparing the keys gives exactly `f64::partial_cmp`: the encoding preserves *and* reflects order. `encodeF64_inj` proves two keys are equal only for the same bits or the two zeros. The model works on the real IEEE bit pattern, so it covers both zeros, every NaN sign and payload, and ±∞. NaN and the temporal types are not indexed at all (`nan_unindexed`, `temporal_unindexed`).

**2. Are integers safe, given they are stored as `f64`?**

An `Int` is indexed as `i as f64`, which loses precision above 2⁵³ (`p53_collision`: 2⁵³ and 2⁵³+1 get one key). The planner guards against this with `int_loses_f64_precision` on the query bound. `int_cmp_agrees` proves the guard is sound for bounds with `|b| < 2⁵²`: against such a bound, comparing the `f64` images gives the exact `i64` order for every stored integer. The guard has one hole, `i64::MIN` (see [Bugs found](#bugs-found)).

**3. Does an index scan return exactly what a label scan plus filter returns?**

Yes, for non-Bool data and a guarded numeric bound. `filter_iff_key` proves that for every data value (any variant, any NaN payload, `-0.0`, ±∞, any `i64`) and all five operators (`<`, `<=`, `>`, `>=`, `=`), the `WHERE` predicate holds iff the value is indexed and its key satisfies the same comparison against the bound's key. `range_keys` proves the inclusive key bounds `range` computes (including the `±1` steps for exclusive bounds) are exact. Then:

- `range_agrees` and `point_agrees`: a column built from `(value, id)` pairs answers a `Range` or an `Equal` with exactly the ids whose value passes the filter;
- `nodeScan_native_agrees`: the same, end to end, for `NodeByIndexScanOp` on the native route;
- `queryNumeric_spec`: `query_numeric`'s three-way answer (below): rows from a `Ready` column, a `ScanFilter` from a `Building` one, `None` for a shape the column does not serve.

**3b. Does an `IN` list (#2383) return what a label scan plus `IN` filter returns?**

Yes for a non-empty list whose members are all guarded numerics, and it declines rather than answering part. One open gap sits before the column: a mixed list such as `[date(..), 2]` loses its non-scalar members in the scan op before the native route sees them, so the column is asked about a shorter list and answers wrongly (W2-index-3, FINDINGS.md; not modelled here). `n.a IN [v1, ..]` reaches the column as one `InList`; `numeric.rs::in_list` encodes every member and returns them as a `KeySet::Keys` (sorted, deduplicated: `KeySet::distinct`) that `lookup` turns into the union of the point scans (since #2386 the keys are computed apart from the lookup).

- `inList_agrees`: a served list returns exactly the ids whose value equals some member (members guarded, no `Bool` data), via `point_agrees` per key;
- `inList_isSome_iff`: the column answers only a non-empty list whose members all encode. Serving the numeric subset would drop rows held by another kind (a string member), and an empty list must not mean "no rows"; the scan op does not re-check the predicate, so both are declined (`inListKeys_isSome_iff`), and under `index-falkordb` a decline is an error, not a RediSearch query;
- `union_agrees`, `union_nil`, `union_attr_mismatch`, `union_range_declines`: an `OR` is served only when every child is an `Equal` or a non-empty `IN` list on one attribute, and then returns the ids equal to some member. An empty `OR` is not match-all, and an `OR` over two attributes would return wrong rows from one column, so both are declined. A member `a = null` holds for nothing and adds no key (`union_null_only`, #2386);
- `unionNext_spec` (`UnionIter::next`): the lazy chain builds one cursor at a time and each `next` yields the head of what remains, so draining it enumerates the point scans in key order, and the eager model `lookup (.keys ks)` is exact (`unionIter_initial`). The tree snapshot is pinned when the union is built (the model tree is an immutable value);
- `queryNumeric_spec`: the facade routes an `IN` list to its attribute's column and an `OR` by its first member's attribute, which `unionKeys` rechecks;
- `nodeScan_native_inList_agrees`: end to end, a non-empty guarded `IN` list on a built column returns exactly the ids the filter keeps.

Order is deliberately unspecified (the chain is not merged into id order), so every statement is about membership. Uniqueness of ids rests on the encoder rejecting lists (a doc has one key per column) and is not modelled. A `Bool` member shares a key with `0`/`1` as in `bool_divergence`.

**4. Does batching writes per unit of work (#3307) change the result?**

No. #3307 stages index changes during a query and applies them once at commit: `last_write_wins` collapses each tuple's history, then one `remove_batch` and one `insert_batch` run per column. `lastWriteWins_spec` proves the collapse keeps exactly each tuple's last staged change, and `writerRecord_eq_sequential` proves the batched apply leaves the same tree as applying every staged change one by one (filtered by the column's build state; unfiltered for a `Ready` column), in order, for any interleaving of adds and removes.

**4b. Is a column that is still being built ever read, and do writes during the build get lost? (#2351)**

A column is created `Building` (`create_building`) and filled by indexer batches; the batch that reaches the end flips it `Ready`. While `Building`, `query_numeric` answers a shape the column serves with a `ScanFilter`, never with rows (`queryNumeric_building`), and the scan op scans instead (below; before #2386 it fell through to RediSearch); a `Ready` column answers from its tree (`queryNumeric_ready`). What is declined does not depend on the build state (`queryNumeric_isSome_iff`): an unservable shape is never turned into a quiet scan. Two things keep the part of the column below the cursor current while the rest is still unread:

- **Writes.** `writerRecord_eq_sequential` states what a unit of work does to a column at any state: its staged changes, filtered by `stateLog`, applied one by one. `stateLog_removes` / `stateLog_adds` say the filter keeps every removal and an addition only for an entity `place`d below the cursor (a node by its id, an edge by its `(src, dst)` pair, `None` for an edge deleted in the unit). The filter runs *before* the collapse: `building_reuse_removed` shows a removed-then-re-added tuple for an entity at or above the cursor (a reused edge id) is still removed, where collapsing first would have kept it (`collapse_first_would_keep`). Writes never move the cursor (`applyWrites_state`).
- **Batches.** `scanNodes_below_cursor` and `scanEdges_below_cursor` prove the entities a batch read are exactly those that `precede` the cursor it returns (with the matrices iterated in ascending order, a GraphBLAS fact), so "indexed" and "below the cursor" are the same set; a batch never stops inside a multi-edge `(src, dst)` pair (`edgeLoop_spec`). `prepareBatch_building` / `prepareBatch_idle` say a batch adds its tuples and moves the cursor (or flips `Ready`), and changes nothing on a column that is gone or not `Building`.
- **CREATE INDEX** is one batch of unlimited size: `buildInOneBatch_entry` proves the column is then `Ready` and holds exactly `from_entries` of the live pairs, so `createIndex_nodeScan_agrees` carries `nodeScan_native_agrees` over to the new batch path.

**4c. What does a read do under `index-falkordb`, and is there a fallthrough? (#3547, #2386)**

`query_numeric` has three outcomes: `Some(Ok(rows))` (a `Ready` column), `Some(Err(scan_filter))` (a servable shape on a `Building` column) and `None` (anything else), which the scan op turns into `unsupported_by_native_index`. Nothing reads RediSearch for a Range index.

- `scanOp_spec`, `scanOp_none_iff`, `nodeScanOp_error_iff`, `edgeScanOp_error_iff`: the op errors exactly when the native index declines the predicate, never because a column is building;
- `queryNumeric_range_served`, `nodeScanOp_range_served`, `edgeScanOp_range_served`: a `Range` (bounds `Null` or encodable) on an existing column is never declined, so it is rows or a scan;
- `queryNumeric_null_equal`: `= null` holds for nothing and is answered with no rows, not declined as a value no column indexes (#2386);
- `nodeFilteredScan_agrees`, `edgeFilteredScan_agrees` (`node_matches` / `relationship_matches`): **the scan that stands in for a building column keeps exactly the ids the built column returns for the same keys**, because both use one `KeySet` (`lookup_fromEntries`). `nodeScan_building_agrees` / `edgeScan_building_agrees` state it for the whole read: if `query_numeric` answers with a `ScanFilter`, the scan equals what the column `CREATE INDEX` would build answers. The planner drops a predicate it hands to an index scan, so this filter is the only check the scanned rows get.
- `describe` / `unsupportedByNativeIndex` (`Decline.lean`): the error names the entity, label and predicate shape. The three union reasons are decided by the flattened members (`describe_or`) and the "spanning" and "non-member" reasons are always real declines (`describe_or_reason_sound`). One cosmetic inexactness: `a = 1 OR a IN []` is described as "not all-numeric" (`describe_empty_in_misreported`).

**4d. Who feeds RediSearch now? (`redisearch_fields`)**

With the feature, RediSearch gets every field except Range: `redisearchFields_no_range`, `redisearchFields_keeps_nonrange` (Fulltext/Vector fields stay, also on a label that has a Range field), `redisearchFields_nonempty` (a key left with no field is dropped), `redisearchFields_drops_range_only`, `redisearchFields_idem`. `native_owns_range`: an attribute with a Range field is built natively by `populate` (from every field) and never fed to RediSearch. Without the feature `redisearch_fields` is the identity (`redisearchFieldsAll_id`). `index_only_writes` (the `FALKORDB_INDEX_ONLY` measurement gate) is gone and its theorem with it.

**4e. Where does a unique-constraint check look? (`unique.rs`, #3334)**

`equalCandidates_some` / `_none`: a lookup is made exactly when there is a column, it is not `Building` (it has not seen every entity) and its kind indexes the value; an empty answer from a `Ready` column really means no entity holds the value (`equalCandidates_complete`, `rows_mem`). `nativeUniqueQuery_none_iff`: no lookups exactly when no column can answer exactly. `fewest_eq_minRow`, `fewest_mem`, `fewest_minimal`: the round-robin loop returns the first shortest lookup. `readOnce_first`, `readOnce_repeat`, `readOnce_skip_sound`: a lookup set is read once and a repeat reads nothing (the same ids give the same rows, `rows_eq_of_id_eq`).

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
| `inList_agrees`, `union_agrees`, `inList_isSome_iff` | #2383: an `IN` list / same-attribute `OR` is a union of point lookups equal to the filter, and is declined unless served whole | `Numeric.lean` |
| `unionNext_spec` | The lazy cursor chain yields exactly the point scans in key order | `Numeric.lean` |
| `nodeScan_native_inList_agrees` | Node `IN` scan on the native route = label scan + `IN` filter | `Ddl.lean` |
| `nodeScan_native_agrees` | Node index scan on the native route = label scan + filter | `Ddl.lean` |
| `nodeFilteredScan_agrees`, `edgeFilteredScan_agrees`, `nodeScan_building_agrees` | The scan standing in for a building column keeps exactly what the built column returns | `Scan.lean` |
| `queryNumeric_isSome_iff`, `queryNumeric_range_served`, `scanOp_none_iff` | A decline does not depend on the build state; a Range is never declined; the op errors exactly on a decline | `Query.lean`, `Ddl.lean`, `Scan.lean` |
| `redisearchFields_no_range`, `redisearchFields_keeps_nonrange` | RediSearch is fed every field but Range | `Scan.lean` |
| `nativeUniqueQuery_none_iff`, `fewest_eq_minRow`, `readOnce_repeat` | The unique-constraint lookups: declines exactly when no column can answer, reads the smallest, reads once | `Unique.lean` |
| `lastWriteWins_spec` | The collapse keeps exactly each tuple's last staged change | `Pending.lean` |
| `writerRecord_eq_sequential` | #3307's batched apply = applying each staged change one by one (filtered by the column's build state since #2351) | `Pending.lean` |
| `queryNumeric_building` | a `Building` column is answered with a `ScanFilter`, never rows | `Query.lean` |
| `scanNodes_below_cursor`, `scanEdges_below_cursor` | a batch reads exactly the entities that precede its returned cursor | `Build.lean` |
| `buildInOneBatch_entry` | CREATE INDEX leaves a `Ready` column holding `from_entries` of the live pairs | `Ddl.lean` |
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
| `Numeric.lean` | `NumericIndex`, `KeySet`, `rangeBounds`, `range_keys`, `range_agrees`, `point_agrees`, `IndexQuery` (with `inList` and `or`), `keysOf`, `lookup`, `inListKeys`, `unionKeys`, `unionNext`, `lookup_fromEntries`, per-write contents specs |
| `Index.lean` | `HashMap` as an association list, `IndexColumn`, `BuildCursor`, `BuildState`, `IndexEntry`, `FalkorDbIndex`, `rowsAt` |
| `Query.lean` | `ScanFilter`, `queryKey`, `queryNumeric` (three-way answer), `EqualCandidates`, `equalCandidates` |
| `Pending.lean` | Staged writes (`IndexWrite`, `IndexChanges`, `PendingIndexWrites`), `lastWriteWins`, `stateLog`, `writerRecord`, `writerRecord_eq_sequential`, `applyWrites_numeric` |
| `Writes.lean` | The staging functions of `graph_writes.rs` (`nodeAttrsSet`, `nodesDeleted`, `labelsAdded`, …), `lastOp_ops`, the per-path `*_hit` theorems |
| `Build.lean` | Indexer batches: `scan`/`scanNodes`/`scanEdges`/`edgeLoop`, `prepareBatch`, `scanNodes_below_cursor`, `scanEdges_below_cursor` |
| `Ddl.lean` | `createIndex`, `populate`, `buildInOneBatch`, `queryIndexNumericNodes/Edges`, `scanOp`, `nodeScan_native_agrees`, `snapshot_isolated`, `runUnit_spec` |
| `Scan.lean` | `nodeMatches`, `relationshipMatches`, the filtered scans, `nodeScanOp`/`edgeScanOp`, `redisearchFields` |
| `Unique.lean` | `nativeUniqueQuery`, `fewest`, `ReadOnce` |
| `Decline.lean` | `unsupported_by_native_index`'s `describe`, `unionMembers` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Path dependencies (`lakefile.toml`): [`../cow_btree`](../cow_btree) for the entry encoding `enc` and the sorted-list lemmas, and [`../core`](../core) for `roundNat`/`bitlen` (IEEE round-to-nearest-even).

## Coverage

`COVERAGE.tsv` has 190 function rows (plus one header row):

| Bucket | Rows |
|---|---|
| PROVEN | 187 |
| CONDITIONAL | 3 |

- The 13 rows that read or write the tree (`from_entries` (test-only since #2351), `add`, `remove`, `add_batch`, `remove_batch`, `add_encoded`, `remove_encoded`, `is_empty`, `point`, `range`, `empty`, and `FalkorDbIndex::apply` / `apply_writes`) were CONDITIONAL on the `cow_btree` invariant `TreeWF`, which `remove_batch` broke (W7-btree-1). #3283 fixed it, `cow_btree` now proves every reachable tree is well-formed, and `Contract.lean` ties each operation to its model, so they are PROVEN.
- Three CONDITIONAL rows (`node_attrs_set`, `new_node_attrs`, `edge_attrs_set_with`) state their per-entry result (`*_hit`) under an undischarged caller premise: each node's written attribute ids are unique. `Pending::set_node_attribute` keeps them sorted and unique; `set_node_attributes` only `debug_assert!`s it.

## What the model does not cover

- **The tree itself.** The tree is its sorted content list; the tree code is proven in `cow_btree` (including that every reachable tree is well-formed) and `index_layer` (the leaf bytes).
- **Caller uniqueness.** The `*_hit` theorems assume each node's written attribute ids are unique. Created nodes' `new_labels` and bulk/replica record labels are taken to be the node's labels, which is the Rust's own premise (documented in `graph_writes.rs`).
- **The graph's own stores.** How the graph mutates its attribute store, label matrices and `remove_all` is modelled in other projects; here each per-entry outcome is stated against the graph *before* the write.
- **`HashMap` order.** Iteration order is arbitrary, so theorems about a whole map are stated per key. The `bulk`, `rows` and `product` paths have exact staged-ops theorems (`*_changes`); their per-entry outcome follows from `lastOp_ops` as for the other paths.
- **Schema.** Attribute names are assumed unique (`attrNames.Nodup`), which the single `attrs_name` dictionary guarantees.
- **RediSearch.** Under `index-falkordb` nothing reads a Range index from RediSearch. Fulltext and Vector indexes still do (FFI, not modelled here).
- **Scan op details.** The scan ops' label scan (the pattern's labels, `extra_labels`, the edge cache, undirected orientation) belongs to `ops_traverse`; here the scan is the index label's nodes / the type's edges.

## Build and check

```bash
(cd proofs/native_index && lake build)   # this project (also builds ../cow_btree and ../core)
bash proofs/lean_ci.sh native_index    # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `graph/src/index/falkordb/`, the native-index functions in `graph.rs`, or the native route of `node_by_index_scan.rs` / `edge_by_index_scan.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If you change the encoding or the precision guard, re-check `filter_iff_key`; it is the theorem that justifies having no post-filter. If you change `cow_btree`, rebuild this project too, since it imports `cow_btree`. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
