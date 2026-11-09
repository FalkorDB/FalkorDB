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

/-!
# `native_index`: the FalkorDB native numeric index (#2350 ec2448f32, #3307 18836a355)

Targets the worktree at origin/main 43936bab9 (re-targeted from 5c7af54f2: #3568 `bdcabd96d`, a batch that writers overtook refreshes or inserts, by cost: `Batch.lean`, `Rebase.lean`; #3569 `65654bc23`, `GRAPH.MEMORY USAGE` counts the native index: `Memory.lean`. Before that, from a477d9aa3: #3567 `b18530683`, writers rebase onto indexer batches and a batch catches up on writers: `Rebase.lean`; #3319 `1ff68d538`, the CCH path-index contract skeleton: `Cch.lean`; #3320 `5c7af54f2`, the graph test kit, test-only. Before that, from b02266c2b: #3566 `29e85ffc3`, an indexer batch commits as a version of its own: `Batch.lean`; earlier, from e860645bb: #3547 `155a246f8` (native-only
writes: no Range field is fed to RediSearch under `index-falkordb`; `redisearch_fields`; `index_only_writes` is
gone), #2386 `41ed78225` (no fallback under the flag: `query_numeric` answers `Option<Result<rows, ScanFilter>>`,
a predicate the native index cannot serve is the error `unsupported_by_native_index`, a column still building is
answered by a scan whose `ScanFilter` is built from the same `KeySet` the column looks up; `= null` and null
bounds are answered with no rows; new `unique.rs` for the unique-constraint lookups `equal_candidates`,
`native_unique_query`, `ReadOnce`, `fewest_candidates`). Earlier: #2351 `d1a73f65a` (a column's build state and
the indexer batches that fill it: `versioned.rs`, `EncodedTuples`, `BuildCursor`, `scan`, `create_building`),
#3538 `aa19b5270` (the managed index is generic over its kind), #3550 `a299692a2` (lock handover only), #2383
`IN [...]` as a union of point lookups, #3436 `g.borrow()` → `graph()`, #3283 fixed the cow_btree invariant.)
Files: `graph/src/index/falkordb/{encode,numeric,falkordb_index,versioned,unique,mod,
pending_writes,graph_writes}.rs`, the native-index parts of `graph/src/graph/graph.rs`
(`redisearch_fields`, `apply_index_writes`, `commit_index_docs`, `query_index_numeric_{nodes,edges}`,
`node_matches`, `relationship_matches`, `new_version`'s fork) and the index route of
`runtime/ops/{node,edge}_by_index_scan.rs::next`.

## Lean ↔ Rust
| Lean | Rust |
| --- | --- |
| `Float.lean`: `F64` bits, `isNaN`, `pos`, `fpcmp`, `encodeF64` | `f64::to_bits`/`partial_cmp` (IEEE 754 §3.4/§5.11), `encode.rs:50-55` |
| `IntF.lean`: `roundNat`, `magBits`, `i64ToF64` | `i as f64` (proofs/core `roundNat`, imported) |
| `Value.lean`: `Value`, `compareValue`, `cmpPred`, `encodeNumeric`, `intLosesPrecision` | `value.rs:1181-1270`, `eval.rs:446-462`, `:1665-1686`, `encode.rs:29-41`, `index/mod.rs:1371` |
| `Tree.lean`: `tInsert`…`tRange` (CowBTree contents; encoding/sorted lemmas imported from cow_btree) | `cow_btree/mod.rs:190-324` |
| `Contract.lean`: `contract_fromSorted`, `contract_insert`, `contract_remove`, `contract_insertBatch`, `contract_removeBatch`, `contract_range` | `cow_btree/mod.rs:190-324` on every reachable tree |
| `Numeric.lean`: `EncodedTuples.*`, `NumericIndex.*` (`addEncoded` bulk-builds into an empty tree), `KeySet` (`distinct`, `contains`, `matchesValue`), `pointKeys`, `rangeBounds`/`rangeKeys`, `lookup`, `point`, `pointKey`, `range`, `IndexQuery`, `encodeAll`, `inListKeys`, `unionMember`/`unionGo`/`unionKeys`, `keysOf`, `query`, `UState`/`unionNext` | `numeric.rs:60-95` (`KeySet`, `distinct` :77, `contains` :85), `:119` (`UnionIter::next`), `:141-170` (`EncodedTuples`), `:266` (`add_encoded`), `:295-330` (`point`, `point_key`, `range`, `query`), `:341` (`keys`), `:367` (`lookup`), `:385`/`:389` (`point_keys`/`range_keys`), `:448` (`in_list`), `:466` (`union`); `from_entries` :503 is test-only |
| `Index.lean`: `hmGet`…, `IndexColumn.*` (`pushEncoded`, `rowsAt`), `BuildCursor`, `BuildState`, `IndexEntry.*`, `FalkorDbIndex.*` (`entry`, `insertColumn`, `createBuilding`, `applyIndexerBatch`) | `falkordb_index.rs:41-153`, `:280-322` (`BuildCursor`), `:338-632`; `versioned.rs:36-185` |
| `Query.lean`: `ScanFilter`, `queryKey`, `queryNumeric` (three-way `Option (Except ScanFilter rows)`), `EqualCandidates`, `equalCandidates` | `falkordb_index.rs:162-182` (`EqualCandidates`), `:258-277` (`ScanFilter`), `:633-695` (`query_numeric`), `:697-715` (`equal_candidates`) |
| `Pending.lean`: `IndexWrite`, `IndexChanges.*`, `PendingIndexWrites.*`, `lastWriteWins`, `stateLog`, `IndexColumn.writerRecord`, `IndexEntry.writerRecord`, `applyStep`, `FalkorDbIndex.applyWrites`/`apply` | `pending_writes.rs:24-179`, `falkordb_index.rs:215-232` (`writer_record`), `:581-632`; `versioned.rs:99-156` |
| `Writes.lean`: `G`, `StageOp`, `stage`, `nodeAttrsSet`, `nodeRowsSet`, `newNodeAttrs`, `bulkNodeAttrs`, `labelsAdded`, `labelsProductAdded`, `labelsRemoved`, `nodesDeleted`, `edgeAttrsSetWith`/`edgeAttrsSet`/`edgeAttrsSetOfType`, `newEdgeAttrs`, `bulkEdgeAttrs`, `edgesDeleted`, `addNodeUnderLabel`, `removeNodeUnderLabel`, `nodeLabelIdsInto`, `edgeType` | `graph_writes.rs:50-380`, `:451-510` |
| `Build.lean`: `pushAttr`, `scanNodes`, `edgeLoop`, `scanEdges`, `scan`, `G.place`, `IndexColumn.indexerRead`/`indexerAdd`, `G.prepareBatch`, `G.indexerPrepare` | `graph_writes.rs:513-620` (`scan` :513, `scan_nodes` :538, `scan_edges` :574); `falkordb_index.rs:235-250`; `versioned.rs:187-230`; `graph.rs:3858-3888` (`place`, in `apply_index_writes`) |
| `Batch.lean`: `SE` (with `since`, `written`), `discardedAgainst`, `V` (`newBuildVersion`, `createEntry`, `setBatchesPublished`, `batchPublishedSince`, `applyBatch`, `withBatchIndex`, `insertedOnto`, `caughtUp`), `CatchUp`, `catchUpBy`, `IndexColumn.batchLen`, `indexerFit` (reads `SE.written` and `batchLen work`), `commitIndexerBatch`, `commitIndexerBatchE` | `versioned.rs:54-65`, `:173-241` (`IndexEntry`), `:248-340` (`apply_indexer_batch`, `indexer_prepare`, `indexer_fit`, `with_batch_index`, `take_index`); `falkordb_index.rs:452`, `:565`, `:575`; `graph.rs:1138` (`new_build_version`); `mvcc_graph.rs:264`, `:293-388` |
| `Rebase.lean`: `Ver`, `tupl`, `entityTuples`, `rawTuples`, `placedBelow`, `IndexColumn.refresh`/`writerRebase`/`indexerCatchUp`, `Written`, `BS`, `EN` (`writerRecord`, `indexerApply`, `publish`, `rebase`, `catchUp`), `overtaken`, `BuildCursor.fromId`, `publishAll`, `columnNames`, `rebaseAll`, `overtakenAny`, `mvccRebase`, `V.rebased` | `falkordb_index.rs:169-244` (`refresh`, `placed_below`), `:326`, `:337`, `:437` (`from_id`), `:716`, `:724`; `versioned.rs:54-81` (`BuildState`, `Written`), `:213`, `:227`, `:280-334`, `:438-565`; `graph_writes.rs:445` (`entity_tuples`); `mvcc_graph.rs:165` (`rebase_onto_indexer_batches`); `graph.rs:3879` (`build_place`) |
| `Memory.lean`: `NumericIndex.memoryUsage`, `IndexColumn.memoryUsage`, `EN.memoryUsage`, `memoryUsage`, `IndexColumn.batchLen_spec` (`IndexColumn.batchLen` is in `Batch.lean`) | `numeric.rs:296` (`NumericIndex::memory_usage` = `CowBTree::heap_bytes`, `cow_btree/mod.rs:419`), `falkordb_index.rs:405` (`VersionedIndex::memory_usage`), `:382` (`batch_len`), `:582` (`FalkorDbIndex::memory_usage`); `versioned.rs:307`, `:351` (`IndexEntry::memory_usage`) |
| `Cch.lean`: `Csr` (`row?`, `isWellFormed`, `isSymmetricSimple`), `defNew`, the `unimplemented!()` bodies as always-panicking functions | `cch/mod.rs:57-107` (`Csr`), `:139` (`CchIndexDef::new`), `:193-271` (`Cch::*`), `:314-413` (`VersionedIndex for CchIndex`); `falkordb_index.rs:967-991`; `graph.rs:5225-5275` |
| `Ddl.lean`: `collectNodeEntries`, `collectEdgeEntries` (specifications), `buildInOneBatch`, `createIndex`, `buildColumns`, `populate`, `applyIndexWrites`, `commitIndexDocs`, `queryIndexNumericNodes/Edges`, `scanOp`, `newVersion`, `runUnit` | `graph_writes.rs:396-450`; `graph.rs:1083`, `:3858`, `:3889`, `:4181`, `:4199`; `node_by_index_scan.rs:302`, `edge_by_index_scan.rs:318` |
| `Scan.lean`: `G.nodeMatches`/`relationshipMatches`, `nodeFilteredScan`/`edgeFilteredScan`, `nodeScanOp`/`edgeScanOp`, `redisearchFieldsAll`/`redisearchFields` | `graph.rs:4230` (`node_matches`), `:4244` (`relationship_matches`), `:144`/`:151` (`redisearch_fields`); the `Some(Err(scan_filter))` arms of `node_by_index_scan.rs:~330`, `edge_by_index_scan.rs:~375` |
| `Unique.lean`: `nativeUniqueQuery`, `roundGo`/`fewestLoop`/`fewest`, `ReadOnce` | `unique.rs:20`, `:48`, `:69` |
| `Decline.lean`: `Pred`, `describe`, `oneKey`, `unionMembers`, `unsupportedByNativeIndex` | `mod.rs:43` (`unsupported_by_native_index`), `:59` (`describe_predicate`), `:145` (`union_members`) |

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
