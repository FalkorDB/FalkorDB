import FalkorNativeIndex.Float
import FalkorNativeIndex.IntF
import FalkorNativeIndex.Value
import FalkorNativeIndex.Agree
import FalkorNativeIndex.Tree
import FalkorNativeIndex.Contract
import FalkorNativeIndex.Numeric
import FalkorNativeIndex.Index
import FalkorNativeIndex.Query
import FalkorNativeIndex.Pending
import FalkorNativeIndex.Writes
import FalkorNativeIndex.Build
import FalkorNativeIndex.Ddl
import FalkorNativeIndex.Scan
import FalkorNativeIndex.Unique
import FalkorNativeIndex.Decline
import FalkorNativeIndex.Batch
import FalkorNativeIndex.Rebase
import FalkorNativeIndex.Cch
import FalkorNativeIndex.Memory
import FalkorNativeIndex.Rows
import FalkorNativeIndex.Bridge
import FalkorNativeIndex.Online
import FalkorNativeIndex.CchOrder
import FalkorNativeIndex.CchTopo
import FalkorNativeIndex.CchMetric
import FalkorNativeIndex.CchPaths
import FalkorNativeIndex.CchSearch
import FalkorNativeIndex.CchRoute
import FalkorNativeIndex.CchSmall
import FalkorNativeIndex.CchSeq

/-!
# `native_index`: the FalkorDB native numeric index (#2350 ec2448f32, #3307 18836a355)

Targets the worktree at origin/main 256ab4d3f (re-targeted from 43936bab9). New since then, all modelled here:
#2453 `6b5a31c04` (list values: a second tree per column, the array tree, holding a list's numeric elements; a
tuple is `(kind, key, doc)`; `ArrayContains` and `And` queries are served), #3736 `ea64405dd` (a Range column
that answers with a `Selection`: `range.rs`), #3737 `a6a53fd72` (a union or conjunction across attributes is one
column per attribute, `combine_columns`, `DistinctIter`; `ScanFilter` is `All`/`Any` over several attributes),
#3585 `7472d6a8a` (CREATE INDEX returns before the build; `online_build.rs`: only `Sizing::batch` is modelled),
#3856 (the background drop removes only an entry with no fields), CCH phases 1-3 (`cch/`: not modelled).
Model changes: `Value.list` carries each element's `encode_numeric` image (all `for_each_stored_key` reads of an
element); `EncodedTuples`/`NumericIndex` have two trees; `IndexWrite` has a `kind` and `last_write_wins` sorts by
`(kind, key, id)`; `queryNumeric`/`keysOf`/`unionKeys` are replaced by `Selection.of`/`queryColumn`
(`describe_or_reason_sound`, `unionKeys_served` are gone: a union across attributes is now served, so
"spanning" is no longer a decline of the single-column union). `Rebase.lean`/`Batch.lean` model the scalar tree's
tuples (their array tuples are not modelled: see Gaps). Earlier targets: 43936bab9 (re-targeted from 5c7af54f2): #3568 `bdcabd96d`, a batch that writers overtook refreshes or inserts, by cost: `Batch.lean`, `Rebase.lean`; #3569 `65654bc23`, `GRAPH.MEMORY USAGE` counts the native index: `Memory.lean`; and before (see git history).
Files: `graph/src/index/falkordb/{encode,numeric,range,falkordb_index,versioned,pending_writes,graph_writes,unique,mod,online_build}.rs`, the native-index parts of `graph/src/graph/graph.rs` and the index route of `runtime/ops/{node,edge}_by_index_scan.rs::next`.

## Lean ↔ Rust
| Lean | Rust |
| --- | --- |
| `Float.lean`: `F64` bits, `isNaN`, `pos`, `fpcmp`, `encodeF64` | `f64::to_bits`/`partial_cmp` (IEEE 754 §3.4/§5.11), `encode.rs:106-111` |
| `IntF.lean`: `roundNat`, `magBits`, `i64ToF64` | `i as f64` (proofs/core `roundNat`, imported) |
| `Value.lean`: `Value` (with `list`: each element's `encode_numeric` image), `compareValue`, `cmpPred`, `encodeNumeric`, `intLosesPrecision` | `value.rs:1181-1270`, `eval.rs:446-462`, `:1665-1686`, `encode.rs:29-41`, `index/mod.rs:1371` |
| `Tree.lean`: `tInsert`…`tRange` (CowBTree contents; encoding/sorted lemmas imported from cow_btree) | `cow_btree/mod.rs:190-324` |
| `Contract.lean`: `contract_fromSorted`, `contract_insert`, `contract_remove`, `contract_insertBatch`, `contract_removeBatch`, `contract_range` | `cow_btree/mod.rs:190-324` on every reachable tree |
| `Numeric.lean`: `KeyKind`, `forEachStoredKey`, `EncodedTuples.*` (`get`, `push`, two kinds), `NumericIndex.*` (`tree`/`arrayTree`; `addTrees`/`removeTrees` = `RangeKind`; `add`/`remove`/`addBatch`/`addEncoded` = `RangeIndex`), `pushStored`, `KeySet` (`distinct`, `ofKeys`, `and`, `contains`, `matchesValue`), `pointKeys`, `rangeBounds`/`rangeKeys`, `lookup`, `listsHolding`, `lookupLists`, `lookupSel`, `IndexQuery`, `Selection` (`equal`, `equalToAny`, `range`, `listsHoldingOf`, `disjunction`, `conjunction`, `and`, `of`, `matches`), `attrsOf`, `atoms`, `onOneAttribute`, `stores`, `UState`/`unionNext` | `encode.rs:81` (`for_each_stored_key`); `numeric.rs:117-199` (`KeySet`: `distinct` :133, `of_keys` :143, `contains` :153, `and` :173), `:224` (`UnionIter::next`), `:244-301` (`EncodedTuples`), `:314-345` (`NumericIndex`, `range_keys` :345), `:427-531` (`RangeKind for NumericIndex`, `stores`); `range.rs:98-399` (`Kind`, `Selection`, `atoms`, `on_one_attribute`, `attributes_of`), `:429-616` (`RangeIndex`); `from_entries` is the specification of a built column |
| `Index.lean`: `hmGet`…, `IndexColumn.*` (`pushEncoded`, `rowsAt`, `lookup`, `storedKeys`), `BuildCursor`, `BuildState`, `IndexEntry.*`, `FalkorDbIndex.*` (`entry`, `insertColumn`, `createBuilding`, `applyIndexerBatch`, `storedKeys`) | `falkordb_index.rs:47-186` (`IndexColumn`), `:526-598` (`BuildCursor`), `:598-1250` (`FalkorDbIndex`); `versioned.rs:45-130` |
| `Query.lean`: `ScanFilter` (`one`, `matches`, `resolve`), `queryColumn`, `combineColumns`, `groupList`, `distinctDocs`, `interDocs`, `EqualCandidates`, `equalCandidates` | `falkordb_index.rs:288-308` (`EqualCandidates`), `:450-515` (`ScanFilter`), `:998-1150` (`query_column`, `combine_columns`), `:1159-1177` (`equal_candidates`); `numeric.rs:78-113` (`DistinctIter`) |
| `Pending.lean`: `IndexWrite` (`kind`), `IndexChanges.*`, `PendingIndexWrites.*`, `lastWriteWins`, `kindOps`/`lastOpK`/`seqOps`/`seqIndex`, `stateLog`, `IndexColumn.writerRecord`, `IndexEntry.writerRecord`, `applyStep`, `FalkorDbIndex.applyWrites`/`apply` | `pending_writes.rs:24-190`, `falkordb_index.rs:342-361` (`writer_record`), `:944-996` (`apply`, `apply_writes`); `versioned.rs:99-156` |
| `Writes.lean`: `G`, `StageOp`, `stage`, `pushL`, `lastOpS`, `nodeAttrsSet`, `nodeRowsSet`, `newNodeAttrs`, `bulkNodeAttrs`, `labelsAdded`, `labelsProductAdded`, `labelsRemoved`, `nodesDeleted`, `edgeAttrsSetWith`/`edgeAttrsSet`/`edgeAttrsSetOfType`, `newEdgeAttrs`, `bulkEdgeAttrs`, `edgesDeleted`, `addNodeUnderLabel`, `removeNodeUnderLabel`, `nodeLabelIdsInto`, `edgeType` | `graph_writes.rs:54-400`, `:469-590` |
| `Build.lean`: `pushAttr`, `scanNodes`, `edgeLoop`, `scanEdges`, `scan`, `G.place`, `IndexColumn.indexerRead`/`indexerAdd`, `G.prepareBatch`, `G.indexerPrepare` | `graph_writes.rs:600-717` (`scan` :600, `scan_nodes` :625, `scan_edges` :661); `falkordb_index.rs:412-428`; `versioned.rs:478-503`, `:760-781`; `graph.rs:3902-3916` (`build_place`, `apply_index_writes`) |
| `Batch.lean`: `SE` (with `since`, `written`), `discardedAgainst`, `V` (`newBuildVersion`, `createEntry`, `setBatchesPublished`, `batchPublishedSince`, `applyBatch`, `withBatchIndex`, `insertedOnto`, `caughtUp`), `CatchUp`, `catchUpBy`, `IndexColumn.batchLen`, `indexerFit` (reads `SE.written`, `batchLen work`, the catch-up limit and `read`), `commitIndexerBatch`, `commitIndexerBatchE` | `versioned.rs:64-80`, `:349-465` (`IndexEntry`), `:465-570` (`apply_indexer_batch`, `indexer_prepare`, `indexer_fit`, `with_batch_index`); `:745` (`take_index`); `falkordb_index.rs:899-918`; `graph.rs:1125` (`new_build_version`); `mvcc_graph.rs:262`, `:293-388` |
| `Rebase.lean`: `Ver`, `tupl`, `entityTuples`, `rawTuples`, `placedBelow`, `IndexColumn.refresh`/`writerRebase`/`indexerCatchUp`, `Written`, `BS`, `EN` (`writerRecord`, `indexerApply`, `publish`, `rebase`, `catchUp`), `overtaken`, `BuildCursor.fromId`, `publishAll`, `columnNames`, `rebaseAll`, `overtakenAny`, `mvccRebase`, `V.rebased` | `falkordb_index.rs:196-285` (`refresh`, `placed_in`), `:362`, `:373`, `:559` (`from_id`), `:877`, `:885`; `versioned.rs:45-100` (`BuildState`, `Written`), `:284`, `:298`, `:374-446`, `:570-760`; `graph_writes.rs:469` (`entity_tuples`); `mvcc_graph.rs:171` (`rebase_onto_indexer_batches`); `graph.rs:3902` (`build_place`) |
| `Memory.lean`: `NumericIndex.memoryUsage`, `IndexColumn.memoryUsage`, `EN.memoryUsage`, `memoryUsage`, `memoryUsage_two_trees`, `IndexColumn.batchLen_spec` (`IndexColumn.batchLen` is in `Batch.lean`) | `numeric.rs:474` (`NumericIndex::memory_usage` = both trees' `CowBTree::heap_bytes`), `falkordb_index.rs:429` (`VersionedIndex::memory_usage`), `:406` (`batch_len`), `:660` (`FalkorDbIndex::memory_usage`); `versioned.rs:345`, `:389` (`IndexEntry::memory_usage`) |
| `Cch.lean`: `Csr` (`row?`, `isWellFormed`, `isSymmetricSimple`), `defNew`, the `unimplemented!()` bodies as always-panicking functions (the CCH algorithms themselves are not modelled) | `cch/mod.rs:58-142` (`Csr`), `:189` (`CchIndexDef::new`), `:286-309` (`Cch::*`), `:357-452` (`VersionedIndex for CchIndex`); `falkordb_index.rs:1228-1255`; `graph.rs:5348-5397` |
| `CchOrder.lean`, `CchTopo.lean`, `CchMetric.lean`: `Pth`, `rowOf`/`buildRows` (the fill-in rows), `CS`/`csStep` (the loop with `first_child`/`next_sibling`), `flatRows` (offsets + neighbours), `invert`, `downRows` (`transpose`), `Topo`, `fromOrder`, `ndOrder` (backend arbitrary), `WP`/`IsMin`, `relaxA`/`relaxM`, `customizeRun`, `unpackRec` | `cch/order.rs:40-470`, `cch/nd.rs:46-125`, `cch/customize.rs:187-310`, `cch/query.rs:206-229` |
| `Ddl.lean`: `collectNodeEntries`, `collectEdgeEntries` (specifications), `buildInOneBatch`, `createIndex`, `buildColumns`, `populate`, `applyIndexWrites`, `commitIndexDocs`, `queryIndexNumericNodes/Edges`, `scanOp`, `newVersion`, `runUnit` | `graph_writes.rs:418-536`; `graph.rs:1125`, `:3902-3950`, `:4246`, `:4264`; `node_by_index_scan.rs:302`, `edge_by_index_scan.rs:318` |
| `Scan.lean`: `resolveGo`/`matches_resolve`, `G.nodeMatches`/`relationshipMatches`, `nodeFilteredScan`/`edgeFilteredScan`, `NodeCond`/`EdgeCond`/`FilterHolds`, `nodeScanOp`/`edgeScanOp`, `redisearchFieldsAll`/`redisearchFields` | `graph.rs:4296` (`node_matches`), `:4308` (`relationship_matches`), `:145`/`:152` (`redisearch_fields`); `falkordb_index.rs:471-515` (`ScanFilter::resolve`/`matches`); the `Some(Err(scan_filter))` arms of `node_by_index_scan.rs:~330`, `edge_by_index_scan.rs:~375` |
| `Unique.lean`: `nativeUniqueQuery`, `roundGo`/`fewestLoop`/`fewest`, `ReadOnce` | `unique.rs:23`, `:51`, `:72` |
| `Decline.lean`: `Pred`, `describe`, `oneKey`, `unionMembers`, `unsupportedByNativeIndex` | `mod.rs:53` (`unsupported_by_native_index`), `:69` (`describe_predicate`), `:161` (`union_members`) |
| `Rows.lean`: `keyset_and_contains`, `atoms_and_no_and`, `selection_and_matches`, `distinctGo`/`distinctRem`, `isReady`, `hasBuildingColumns`, `indexGeneration`, `buildingSince`, `indexesToBuildSince`, `sizingBatch`, `queryRange` | `numeric.rs:97-113`, `:173`; `range.rs:336`, `:363`, `:616`; `falkordb_index.rs:851`, `:1198`; `versioned.rs:796-844`; `online_build.rs:137` |

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
  `queryNumeric_spec` (routing: a numeric `Equal`/`Range`, an `IN` list, or a same-attribute `OR` on an existing column; three-way answer).
- **`IN` lists and `OR` unions of point lookups (#2383)** (`Numeric.lean`, `Ddl.lean`): `inList_agrees`, `union_agrees` (served result = ids equal to some member), `inListKeys_isSome_iff` / `inList_isSome_iff` (declined unless non-empty and every member encodes), `union_nil`, `union_attr_mismatch`, `union_range_declines` (declined, not answered in part), `union_null_only` (`a = null` adds no key, #2386), `keysOf_sorted` (`Keys` is sorted, so `contains` = membership), `unionNext_spec` (the lazy cursor chain yields exactly the point scans in key order), `unionIter_initial`, `nodeScan_native_inList_agrees`. Order is unspecified: statements are membership. Not modelled: id uniqueness across keys (rests on the encoder rejecting lists).
- **#2351 / #3538, build state and indexer batches** (`Index.lean`, `Pending.lean`, `Build.lean`, `Ddl.lean`):
  `queryNumeric_building` / `queryNumeric_ready` (a `Building` column is answered with a `ScanFilter`, never rows),
  `writerRecord_eq_sequential` (a column records a unit's staged changes filtered by its state, applied one by
  one; `writerRecord_ready` is #3307's claim), `stateLog_removes` / `stateLog_adds` (every removal stays, an
  addition only for an entity placed below the cursor), `building_reuse_removed` + `collapse_first_would_keep`
  (the filter must run before the collapse: a reused edge id's removal survives), `applyWrites_state`
  (writes never move the cursor), `scanNodes_below_cursor` / `scanEdges_below_cursor` (what a batch read is
  exactly what `precedes` the returned cursor; `edgeLoop_spec`: a multi-edge pair is never split),
  `scan_sorted`, `scan_kind` / `kind_place` / `kind_new` (`precedes`'s `unreachable!` never fires),
  `prepareBatch_building` / `prepareBatch_idle`, `indexerApply_building` / `_idle`,
  `buildInOneBatch_entry` (the RDB load's build = one unlimited batch: the column is `Ready` and holds
  `from_entries` of the live pairs), `buildColumns_builds`, `buildColumns_ready`,
  `buildColumns_nodeScan_agrees`, `addEncoded_spec` / `addEncoded_sorted` (an empty tree is bulk-built).
  Since the online build (#2355), CREATE INDEX only creates each column `Building` for the background builder:
  `createIndex_entry`, `createIndex_not_ready` (it serves no reads until its build is over).
- **#3566, an indexer batch commits as a version of its own** (`Batch.lean`): the fit of a batch built on a
  build version against what was published meanwhile: `fit_fresh` (nothing committed: `Same`),
  `fit_after_other_batch` (`Swap`), `fit_after_writer` (`CatchUp` since #3567; it was `Discarded Writers`), `fit_after_same_batch`
  (`AnotherBatch`), `fit_after_drop` / `fit_after_recreate` (`Gone`; a re-created index has the next
  generation); the commit: `commit_discarded` (nothing published), `commit_same`, `commit_swap` (only the
  batch's index comes from the batch), `commit_invalid`, `commit_batches` (+1 per publish),
  `commit_keeps_version` / `commit_not_stale` (the data version never moves, so a writer's fork is not made
  stale). The model is on the one index the batch is about; the rest of the graph is opaque.
- **#3567, writers rebase onto indexer batches; a batch catches up on writers** (`Rebase.lean`, `Batch.lean`):
  closes the gap recorded at a477d9aa3 ("`is_stale` compares only the data version"). `refresh_mem`,
  `entityTuples_spec`, `placedBelow_mem`: what `IndexColumn::refresh` does, exactly; `refresh_inv`: a column
  holding exactly the tuples of the entities its version puts in it still does after a refresh over a set
  outside which the two versions agree (`Frame`); `rebase_progress` (a rebased fork has the published copy's
  generation, commit count and cursor: the batches' progress is kept), `rebase_keeps_writes`,
  `rebase_index_spec`, `rebase_inv`, `catchUp_spec`, `catchUp_inv`, `insert_inv` (the Insert path, #3568), `fit_catchUp_how`, `commit_catchUp`, `commit_catchUp_refresh`/`_insert`, `V.rebased_keeps_data`
  (no data version change, so a rebased fork is not stale); `writerRecord_proj`/`indexerApply_proj`/
  `publish_proj` (the pre-#3567 model is the projection of the new one), `writerRecord_records` (every change's
  entity is recorded). The gap #3567 closed (an unrebased writer's commit published the *older* commit count, so the
  next batch was judged `AnotherBatch`; model-only, not reproduced live; see FINDINGS) is covered by
  `rebased_commit_keeps_chain` and `rebased_batch_fits`. `Frame` (outside the
  written set, the two versions agree on what each entity contributes and where it sits) is assumed: the three
  invariant theorems (`refresh_inv`, `rebase_inv`, `catchUp_inv`, `insert_inv`) are conditional on it, the rows rest on the unconditional specs.
- **#3569, `GRAPH.MEMORY USAGE` counts the native index** (`Memory.lean`): `memoryUsage_add` / `memoryUsage_nil`
  (every node and every edge column, CCH not counted), `EN.memoryUsage_ready` / `_building` / `_building_quiet` (a column's
  pages, plus the serialized size of the `since` set while `Building` and non-empty), `batchLen_spec`. The page walk
  `CowBTree::heap_bytes` is the parameter `hb`: it is proved in `cow_btree` (`heapBytes_eq`, under `TreeWF`, which every
  reachable tree satisfies, `Reachable_wf_asserted`).
- **#3568, the fit chooses Refresh or Insert** (`Batch.lean`, `Rebase.lean`): `indexerFit` reads the published copy's
  `since_last_indexer_step.len()` (`SE.since`, `SE.written`) and `batch_len(&work)` (`IndexColumn.batchLen work`) off the
  model, and `fit_catchUp_how` / `fit_after_writer` say `CatchUp` is `catchUpBy` of exactly those two; since the online
  build (#2355), a batch above the floor has a catch-up limit, past which it is discarded (`Discarded.catchUpTooLarge`):
  `fit_after_writer_within`, `fit_after_writer_too_large`; `catchUpBy_refresh_iff`,
  `catchUpBy_saturating`; `placedIn_mem` (node column: the set cut by id at both ends, no hypothesis), `placedIn_rel_mem`
  (relationship column, any ends), `placedIn_edge_spec` / `placedIn_node_spec`, `placedIn_none_eq_placedBelow_*`;
  `indexerInsert_mem`, `insert_inv`, `commit_catchUp_insert`; `publish_since_len` (publishing moves the `since` count).
- **#3319, the CCH contract skeleton** (`Cch.lean`): `Csr::is_well_formed` is exactly the type's shape rules
  (`isWellFormed_iff`), `is_symmetric_simple` exactly a simple undirected adjacency (`isSymmetricSimple_iff`,
  `_any_bs`), `row` never panics on a well-formed `Csr` (`row?_isSome_of_wf`), `CchIndexDef::new` makes equal
  sets equal indexes (`defNew_eq_of_same_members`); every `unimplemented!()` body is an always-panicking
  function (`*_panics`).
- **#3547 / #2386, the native index is the only Range index** (`Query.lean`, `Scan.lean`, `Ddl.lean`,
  `Unique.lean`, `Decline.lean`): `queryNumeric_isSome_iff` (what is declined does not depend on the build
  state), `queryNumeric_ok` / `_error` (rows only from a `Ready` column, a `ScanFilter` only from a `Building`
  one, with the query's keys and attribute), `queryNumeric_range_served` (a Range is never declined),
  `queryNumeric_null_equal`, `queryNumeric_declined`, `scanOp_spec` / `scanOp_none_iff` /
  `nodeScanOp_error_iff` / `edgeScanOp_error_iff` (the scan op errors exactly when the index declines, never
  falls through), `nodeScanOp_range_served` / `edgeScanOp_range_served`; **the scan that stands in for a
  building column equals what the built column returns** (`lookup_fromEntries`, `nodeFilteredScan_agrees`,
  `edgeFilteredScan_agrees`, `nodeScan_building_agrees`, `edgeScan_building_agrees`; `nodeMatches_spec`,
  `relationshipMatches_spec`); `redisearchFields_mem` / `_no_range` / `_keeps_nonrange` / `_nonempty` /
  `_drops_range_only` / `_idem`, `redisearchFieldsAll_id`, `native_owns_range`;
  `equalCandidates_some` / `_none` / `_complete`, `rows_mem`, `nativeUniqueQuery_some` / `_none_iff`,
  `fewest_eq_minRow` / `fewest_mem` / `fewest_minimal`, `readOnce_first` / `_repeat` / `_twice` /
  `_skip_sound`, `rows_eq_of_id_eq`; `unsupported_spec`, `describe_or`, `describe_or_reason_sound`.
  Removed: `indexOnlyWrites_spec` (`index_only_writes` is gone), `scanRoute_spec` (there is no RediSearch
  branch), `unionOf_mem` (`union_of` is `lookup`: `lookup_keys_mem`).
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
- **DDL / MVCC** (`Ddl.lean`): `buildInOneBatch_entry`, `buildColumns_builds`, `createIndex_entry`, `collect*_spec`,
  `populate_builds/skips`, `numeric_dropColumn`, `dropIndex_removes`, `createUnpopulated_creates`,
  `node_edge_independent`; `snapshot_isolated`, `runUnit_spec` (a committed version's index is the
  private copy with its staged changes applied once; a refused unit publishes nothing),
  `newVersion_idx`.

- **#2453 / #3736 / #3737, lists, `Selection`, several attributes** (`Numeric.lean`, `Query.lean`, `Pending.lean`,
  `Scan.lean`, `Rows.lean`): `forEachStoredKey_scalar/_array` (a scalar is one key in the scalar tree, a list one key
  per numeric element in the array tree), `add_array_spec`/`remove_array_spec`/`addEncoded_array_spec`,
  `lookupLists_mem` (a `ListsHolding` lookup returns the lists holding every probed element),
  **`lookupSel_fromEntries`** (for every selection, scalars and lists, the scan that stands in for a building
  column keeps exactly what the built column returns; one value per entity), `pushEqualKeys_spec`,
  `equalToAny_agrees`, `disjunction_agrees` (an `IN`/`OR` on one attribute is exactly the label scan + filter; a
  `NULL` member adds no key), `of_or_nil`, `atoms_and_no_and`, `selection_and_matches`, `keyset_and_contains`,
  `ofKeys_sorted`/`keyset_and_sorted`; `queryColumn_single` / `_building` / `_ready` / `_isSome_iff` /
  `_range_served` / `_null_equal` / `_declined`, `combine_or_ok` / `combine_and_ok` / `combine_building`
  (several attributes: one column each, intersected or unioned, one `Building` column makes the whole query a
  scan), `distinctGo_spec` (`DistinctIter::next`), **`nodeFilteredScan_agrees` / `edgeFilteredScan_agrees`** (the
  resolved `All`/`Any` scan filter equals the built columns' answer: every / some condition), `matches_resolve`;
  `lastWriteWins_spec` (per tree: a tuple is kept iff the last staged change to it, in that tree, is of that side;
  keys and ids `u64`), `writerRecord_eq_sequential` (both trees, `LogWF`), `lastOpS_stored` (a list value stages
  array writes only, so the scalar-tuple staging theorems of `Writes.lean` are unaffected).

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

3. `describe_empty_in_misreported` (cosmetic, `mod.rs:describe_predicate`): `a = 1 OR a IN []` is described as
   "a union ... that is not all-numeric"; the real reason it is declined is the empty `IN` list (likewise a
   non-list `IN` payload). Message text only; not reachable from the planner today.

## Gaps / hypotheses
- **Scan ops** (`Scan.lean`): the filtered scan is modelled over the index label's nodes / the type's edges; the
  Rust scans the pattern's labels (`get_nodes(&node_pattern.labels)`) and the op's `extra_labels` filter follows,
  so the rows agree; that part and the edge cache / undirected orientation are `ops_traverse`'s.
  `nodeFilteredScan_agrees` takes `hlt` (ids fit a `u64`, the model's `W`). `EqualCandidates.id` is the column's
  address in the Rust; the model uses the column's slot (`rows_eq_of_id_eq`): distinct entries are distinct
  objects. `fewest` assumes at least one lookup (the Rust loops forever on none; `native_unique_query` never
  passes none).
- **Array tuples in `Rebase.lean`**: `refresh` and `indexer_insert` push both kinds (`refresh_array_mem`, `indexerInsert_array_mem`); the invariant theorems (`Inv`, `refresh_inv`, `catchUp_inv`, `insert_inv`) are stated for the scalar tree only (their array analogues are not proven).
- **CCH (`cch/`)**: Phases 1-3 are proven over `Nat` weights in rank space. `CchOrder.lean` (`row_iff`: the chordal rows are exactly the fill-in), `CchTopo.lean` (the loop with its linked lists, `invert_spec`, `downRows_spec`, `arcSlot_spec`, `fromOrder_*`, `ndOrder_*` for an arbitrary backend, `topoBuild_correct`), `CchMetric.lean` (`customize_correct`, `unpack_correct`), `CchPaths.lean` (up-down paths: `valley`, `updown_complete`/`updown_sound`), `CchSearch.lean` (chain, `relaxFrom`, the DP invariant `DPInv`/`dp_step`, the climb `loop_inv`, `query_core`, `shared_spec`), `CchRoute.lean` (`arcsUp_spec`, `unpack_fwd/bwd`, `eraseLoops_spec`, `query_correct`, `cch_query_end_to_end`: **`Cch::query` returns a shortest path**), `CchSmall.lean` (`gallopTo_spec`, `validate_spec`, `entries_mem`, `backendCsr_spec`, `arc_via_slots`, `queryWrap_spec`), `CchSeq.lean` (`relax_sequential`, `buildCch_correct`). Scope: pair-indexed arcs linked to the slot arrays by `arc_via_slots`; weights `Nat` (the f64 left-fold structure is not modelled); the ordering backend is an arbitrary function (its permutation is checked by `invert`); node-space and rank-space are related by `invert_spec`, not composed into one statement. Not proven: the slot writes of `seed` (`seedA` is its pair-level result).
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
- **Counters** (`Batch.lean`): the `u64` generation / commit / version / batch counters are modelled as unbounded
  `Nat`; a wrap at 2^64 is assumed unreachable, not proven.
- **Matrix order and size** (`Build.lean`, `Ddl.lean`): `scanNodes_below_cursor` / `scanEdges_below_cursor` assume
  the label matrix is iterated in strictly ascending row order and a tensor by `(src, dst)` (an assumption about
  GraphBLAS iteration order, taken as the hypothesis `hasc` / `G.Ordered`, not proven here), and `buildInOneBatch_entry` that the entity count fits `usize::MAX`
  (`G.Fits`). Not proven: the global invariant "a `Building` column holds exactly the current tuples of the
  entities below its cursor" as an induction over interleaved batches and units of work; its parts are
  proven separately (since #3567 also `Rebase.refresh_inv`/`rebase_inv`/`catchUp_inv`, conditional on `Frame`) (`writerRecord_eq_sequential` + `stateLog_*` for writes, `scan*_below_cursor` +
  `prepareBatch_building` for batches). `idx as u16` (attribute ids above 65535) is not modelled, as before.
- Schema: attribute names unique (`attrNames.Nodup`: the single `attrs_name` dictionary). The graph's
  mutation of its own stores (`insert_attrs`, label matrix, `remove_all`) is modelled by other
  projects; here the per-tuple outcome is stated against the graph *before* the write.
- `HashMap` iteration order is arbitrary: theorems are per key. `bulk`/`rows`/`product` paths have
  exact staged-ops theorems (`*_changes`); their per-tuple outcome follows from `lastOp_ops` as for
  the other add paths.
-/
