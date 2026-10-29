import FalkorNativeIndex.Float
import FalkorNativeIndex.IntF
import FalkorNativeIndex.Value
import FalkorNativeIndex.Agree
import FalkorNativeIndex.Tree
import FalkorNativeIndex.Contract
import FalkorNativeIndex.Numeric
import FalkorNativeIndex.Index
import FalkorNativeIndex.Pending
import FalkorNativeIndex.Writes
import FalkorNativeIndex.Build
import FalkorNativeIndex.Ddl

/-!
# `native_index`: the FalkorDB native numeric index (#2350 ec2448f32, #3307 18836a355)

Targets origin/main e860645bb (re-targeted from ca36bfa15: #2351 `d1a73f65a` (a column's build state and the
indexer batches that fill it: new `versioned.rs`, `EncodedTuples`, `BuildCursor`, `scan`, `create_building`;
`build_column`/`collect_*_entries`/`build_numeric`/`ColumnEntry` are gone; `from_entries` is `#[cfg(test)]`),
#3538 `aa19b5270` (the managed index is generic over its kind: `VersionedIndex`, `IndexEntry`),
#3550 `a299692a2` (`unlock_fair` between population batches: lock handover only, no model change; it is
`populate_index_batch`, in `search_concurrency`). Earlier: #2383 `IN [...]` as a union of point lookups,
#3436 `g.borrow()` → `graph()`, #3283 fixed the cow_btree invariant this project's tree rows rested on.)
Files: `graph/src/index/falkordb/{encode,numeric,falkordb_index,versioned,
pending_writes,graph_writes}.rs`, the native-index parts of `graph/src/graph/graph.rs`
(`apply_index_writes`, `commit_index_docs`, `query_index_numeric_{nodes,edges}`, `new_version`'s
fork) and the native route of `runtime/ops/{node,edge}_by_index_scan.rs::next`.

## Lean ↔ Rust
| Lean | Rust |
| --- | --- |
| `Float.lean`: `F64` bits, `isNaN`, `pos`, `fpcmp`, `encodeF64` | `f64::to_bits`/`partial_cmp` (IEEE 754 §3.4/§5.11), `encode.rs:50-55` |
| `IntF.lean`: `roundNat`, `magBits`, `i64ToF64` | `i as f64` (proofs/core `roundNat`, imported) |
| `Value.lean`: `Value`, `compareValue`, `cmpPred`, `encodeNumeric`, `intLosesPrecision` | `value.rs:1181-1270`, `eval.rs:446-462`, `:1665-1686`, `encode.rs:29-41`, `index/mod.rs:1371` |
| `Tree.lean`: `tInsert`…`tRange` (CowBTree contents; encoding/sorted lemmas imported from cow_btree) | `cow_btree/mod.rs:190-324` |
| `Contract.lean`: `contract_fromSorted`, `contract_insert`, `contract_remove`, `contract_insertBatch`, `contract_removeBatch`, `contract_range` | `cow_btree/mod.rs:190-324` on every reachable tree |
| `Numeric.lean`: `EncodedTuples.*`, `NumericIndex.*` (`addEncoded` bulk-builds into an empty tree), `rangeBounds`, `IndexQuery`, `encodeAll`, `unionOf`, `inList`, `unionMember`/`unionGo`/`union`, `UState`/`unionNext` | `numeric.rs:92-426` (`EncodedTuples` :92-128, `add_encoded` :226, `UnionIter::next` :79, `in_list` :356, `union` :376, `union_of` :408; `from_entries` :434 is test-only) |
| `Index.lean`: `hmGet`…, `IndexColumn.*` (`pushEncoded`), `BuildCursor`, `BuildState`, `IndexEntry.*`, `FalkorDbIndex.*` (`entry`, `insertColumn`, `createBuilding`, `applyIndexerBatch`), `queryKey`, `queryNumeric` | `falkordb_index.rs:41-143`, `:218-257` (`BuildCursor`), `:273-502`, `:560-596`; `versioned.rs:36-185` |
| `Pending.lean`: `IndexWrite`, `IndexChanges.*`, `PendingIndexWrites.*`, `lastWriteWins`, `stateLog`, `IndexColumn.writerRecord`, `IndexEntry.writerRecord`, `applyStep`, `FalkorDbIndex.applyWrites`/`apply` | `pending_writes.rs:24-179`, `falkordb_index.rs:177-193` (`writer_record`), `:516-549`; `versioned.rs:99-156` |
| `Writes.lean`: `G`, `StageOp`, `stage`, `nodeAttrsSet`, `nodeRowsSet`, `newNodeAttrs`, `bulkNodeAttrs`, `labelsAdded`, `labelsProductAdded`, `labelsRemoved`, `nodesDeleted`, `edgeAttrsSetWith`/`edgeAttrsSet`/`edgeAttrsSetOfType`, `newEdgeAttrs`, `bulkEdgeAttrs`, `edgesDeleted`, `addNodeUnderLabel`, `removeNodeUnderLabel`, `nodeLabelIdsInto`, `edgeType` | `graph_writes.rs:65-396`, `:466-516` |
| `Build.lean`: `pushAttr`, `scanNodes`, `edgeLoop`, `scanEdges`, `scan`, `G.place`, `IndexColumn.indexerRead`/`indexerAdd`, `G.prepareBatch`, `G.indexerPrepare` | `graph_writes.rs:528-632` (`scan` :528, `scan_nodes` :553, `scan_edges` :589); `falkordb_index.rs:197-212`; `versioned.rs:187-218`; `graph.rs:3795-3808` (`place`) |
| `Ddl.lean`: `collectNodeEntries`, `collectEdgeEntries` (specifications), `buildInOneBatch`, `createIndex`, `populate`, `indexOnlyWrites`, `applyIndexWrites`, `commitIndexDocs`, `queryIndexNumericNodes/Edges`, `scanRoute`, `newVersion`, `runUnit` | `graph_writes.rs:49-55`, `:411-458`; `graph.rs:1072`, `:3790-3811`, `:3821-3835`, `:4109-4144`; `node_by_index_scan.rs:300`, `edge_by_index_scan.rs:316` |

## Proven
- **Encoding** (`Float.lean`, bit-level f64: both zeros, every NaN sign/payload, ±inf):
  `encodeF64_cmp` (for non-NaN `a, b`: `partial_cmp a b = Some(enc a).cmp(enc b)` — order preserving
  *and* reflecting), `encodeF64_inj` (equal keys ⇔ same bits or both zeros), `encodeF64_negZero`,
  `encodeF64_inf_extremes`, `encodeF64_bounds` (`0 < key < u64::MAX`, so `range`'s `checked_*`
  never fire on a real key); `nan_unindexed`, `temporal_unindexed`, `negZero_agrees`.
- **`i64` vs `f64`** (`IntF.lean`, `Value.lean`): `magBits_strictMono` (exact below 2^53),
  `magBits_ge`/`magBits_lt_2p52`, `i64ToF64_lt` (finite), `int_cmp_agrees` (**the precision guard is
  sound for `|bound| < 2^52`**: exact `i64` order = order of the f64 images for any data `i64`),
  `i64ToF64_collide`/`p53_collision` (2^53 vs 2^53+1 share a key), `guard_rejects_2p53`, `guard_sound`.
- **Index scan ≡ label scan + filter** (`Agree.lean`, `Numeric.lean`, `Ddl.lean`): `filter_iff_key`
  (every data variant vs a guarded non-NaN numeric bound, all five operators), `range_keys` (the
  inclusive key bounds incl. exclusive `±1` steps are exact), `range_agrees`, `point_agrees`,
  `nodeScan_native_agrees` (a built column answers a `Range` with exactly the ids the Filter keeps),
  `queryNumeric_spec` (routing: a numeric `Equal`/`Range`, an `IN` list, or a same-attribute `OR` on an existing column).
- **`IN` lists and `OR` unions of point lookups (#2383)** (`Numeric.lean`, `Ddl.lean`): `inList_agrees`, `union_agrees` (served result = ids equal to some member), `inList_isSome_iff` (declined unless non-empty and every member encodes), `union_nil`, `union_attr_mismatch`, `union_range_declines` (declined, not answered in part), `unionNext_spec` (the lazy cursor chain yields exactly the point scans in key order), `unionIter_initial`, `nodeScan_native_inList_agrees`. Order is unspecified: statements are membership. Not modelled: id uniqueness across keys (rests on the encoder rejecting lists).
- **#2351 / #3538, build state and indexer batches** (`Index.lean`, `Pending.lean`, `Build.lean`, `Ddl.lean`):
  `queryNumeric_building` / `queryNumeric_ready` (a `Building` column is never read from),
  `writerRecord_eq_sequential` (a column records a unit's staged changes filtered by its state, applied one by
  one; `writerRecord_ready` is #3307's claim), `stateLog_removes` / `stateLog_adds` (every removal stays, an
  addition only for an entity placed below the cursor), `building_reuse_removed` + `collapse_first_would_keep`
  (the filter must run before the collapse: a reused edge id's removal survives), `applyWrites_state`
  (writes never move the cursor), `scanNodes_below_cursor` / `scanEdges_below_cursor` (what a batch read is
  exactly what `precedes` the returned cursor; `edgeLoop_spec`: a multi-edge pair is never split),
  `scan_sorted`, `scan_kind` / `kind_place` / `kind_new` (`precedes`'s `unreachable!` never fires),
  `prepareBatch_building` / `prepareBatch_idle`, `indexerApply_building` / `_idle`,
  `buildInOneBatch_entry` (CREATE INDEX = one unlimited batch: the column is `Ready` and holds
  `from_entries` of the live pairs), `createIndex_builds`, `createIndex_ready`,
  `createIndex_nodeScan_agrees`, `addEncoded_spec` / `addEncoded_sorted` (an empty tree is bulk-built).
- **#3307, apply once per unit of work** (`Pending.lean`): `sortLog_filter` (stable sort keeps each
  tuple's history), `lwwGo_spec`, `lastWriteWins_spec` (keeps exactly each tuple's last change),
  `writerRecord_eq_sequential` (collapsed `remove_batch`+`insert_batch` = applying every staged change
  one by one), `applyWrites_numeric` (per column; dropped columns skipped), `extend_log` (staging order
  kept per column), `apply_empties`, `set_then_delete`.
- **Writes keep the index in step** (`Writes.lean`): `*_changes` (each staging function stages exactly
  its ops), `lastOp_ops` (what a list of ops does to any tuple), and per path: `nodeAttrsSet_hit/miss`
  and `edgeAttrsSetWith_hit` (update: old entry removed, new present), `nodesDeleted_hit`/`_noadd`,
  `edgesDeleted_hit`/`_noadd` (delete removes every entry, adds none), `labelsRemoved_hit`,
  `labelsAdded_hit`, `newNodeAttrs_hit`.
- **DDL / MVCC** (`Ddl.lean`): `buildInOneBatch_entry`, `createIndex_builds`, `collect*_spec`,
  `populate_builds/skips`, `numeric_dropColumn`, `dropIndex_removes`, `createUnpopulated_creates`,
  `node_edge_independent`; `snapshot_isolated`, `runUnit_spec` (a committed version's index is the
  private copy with its staged changes applied once; a refused unit publishes nothing),
  `newVersion_idx`.

## Bugs / divergences (counterexamples, `Agree.lean`)
1. `i64min_hole`, `i64min_window` — the guard `int_loses_f64_precision` admits `i64::MIN`
   (`unsigned_abs = 2^63` is outside the mask) but `i64::MIN+1 … i64::MIN+512` round onto it.
   **Confirmed live** (release `--features index-falkordb`, port 19350): with
   `CREATE INDEX FOR (n:L) ON (n.v)` over `v ∈ {MIN, MIN+1, MIN+100, 5}`,
   `WITH -9223372036854775807-1 AS m MATCH (n:L) WHERE n.v > m` returns only `five` (unindexed:
   `five, min1, min100`). C (`bin/macos-arm64v8-release/falkordb.so`, port 19351) also returns only
   `five`, and for `=` returns `min, min1, min100` (Rust: `min`). Shared with C on `>`; not a
   native-index regression. Fix: reject `|i| ≥ 2^52` including `i64::MIN` (`i.unsigned_abs() >= 1<<52`).
2. `bool_divergence` — Bool and number share a key; pinned in the Rust (`encode.rs:248-294`) as the
   behaviour C and the RediSearch path share (measured there); not a regression.

## Gaps / hypotheses
- **CowBTree contract** (`Tree.lean`): the tree is its content list, each op its content spec. No longer a
  hypothesis: since #3283 (799968a8b, W7-btree-1 fixed) cow_btree proves every reachable tree is
  `TreeWF` (`Reachable_wf_asserted`: `from_sorted`, `insert`, `remove`, `insert_batch`, `remove_batch`)
  and `Contract.lean` proves each op equals its `t*` model there. The 13 tree rows moved from
  CONDITIONAL to PROVEN.
- **Caller uniqueness** (`nodeAttrsSet_hit`, `edgeAttrsSetWith_hit`, `newNodeAttrs_hit`): each node's
  written attribute ids are unique — `Pending::set_node_attribute` keeps them sorted/unique,
  `set_node_attributes` only `debug_assert!`s it (CONDITIONAL rows). Created nodes' `new_labels` /
  bulk / replica record labels are taken as the node's labels (the Rust's own premise, doc
  `graph_writes.rs:95-97`, `:131-132`).
- **Matrix order and size** (`Build.lean`, `Ddl.lean`): `scanNodes_below_cursor` / `scanEdges_below_cursor` assume
  the label matrix is iterated in strictly ascending row order and a tensor by `(src, dst)` (GraphBLAS iteration,
  a C API fact; hypotheses `hasc`), and `buildInOneBatch_entry` that the entity count fits `usize::MAX`
  (`G.Fits`). Not proven: the global invariant "a `Building` column holds exactly the current tuples of the
  entities below its cursor" as an induction over interleaved batches and units of work; its two halves are
  proven separately (`writerRecord_eq_sequential` + `stateLog_*` for writes, `scan*_below_cursor` +
  `prepareBatch_building` for batches). `idx as u16` (attribute ids above 65535) is not modelled, as before.
- Schema: attribute names unique (`attrNames.Nodup`: the single `attrs_name` dictionary). The graph's
  mutation of its own stores (`insert_attrs`, label matrix, `remove_all`) is modelled by other
  projects; here the per-tuple outcome is stated against the graph *before* the write.
- `HashMap` iteration order is arbitrary: theorems are per key. `bulk`/`rows`/`product` paths have
  exact staged-ops theorems (`*_changes`); their per-tuple outcome follows from `lastOp_ops` as for
  the other add paths.
-/
