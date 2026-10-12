import FalkorNativeIndex.Float
import FalkorNativeIndex.IntF
import FalkorNativeIndex.Value
import FalkorNativeIndex.Agree
import FalkorNativeIndex.Tree
import FalkorNativeIndex.Numeric
import FalkorNativeIndex.Index
import FalkorNativeIndex.Pending
import FalkorNativeIndex.Writes
import FalkorNativeIndex.Ddl

/-!
# `native_index`: the FalkorDB native numeric index (#2350 ec2448f32, #3307 18836a355)

Targets origin/main 4a4f2a72f: `graph/src/index/falkordb/{encode,numeric,falkordb_index,
pending_writes,graph_writes}.rs`, the native-index parts of `graph/src/graph/graph.rs`
(`apply_index_writes`, `commit_index_docs`, `query_index_numeric_{nodes,edges}`, `new_version`'s
fork) and the native route of `runtime/ops/{node,edge}_by_index_scan.rs::next`.

## Lean ↔ Rust
| Lean | Rust |
| --- | --- |
| `Float.lean`: `F64` bits, `isNaN`, `pos`, `fpcmp`, `encodeF64` | `f64::to_bits`/`partial_cmp` (IEEE 754 §3.4/§5.11), `encode.rs:50-55` |
| `IntF.lean`: `roundNat`, `magBits`, `i64ToF64` | `i as f64` (value_order `roundNat`, copied) |
| `Value.lean`: `Value`, `compareValue`, `cmpPred`, `encodeNumeric`, `intLosesPrecision` | `value.rs:1181-1270`, `eval.rs:446-462`, `:1665-1686`, `encode.rs:29-41`, `index/mod.rs:1371` |
| `Tree.lean`: `tInsert`…`tRange` (CowBTree contents, copied from cow_btree) | `cow_btree/mod.rs:173-307` |
| `Numeric.lean`: `NumericIndex.*`, `rangeBounds`, `IndexQuery` | `numeric.rs:39-254` |
| `Index.lean`: `hmGet`…, `IndexColumn.*`, `ColumnEntry.ready`, `FalkorDbIndex.*` | `falkordb_index.rs:38-355`, `:409-432` |
| `Pending.lean`: `IndexWrite`, `IndexChanges.*`, `PendingIndexWrites.*`, `lastWriteWins`, `applyColumn`, `applyStep`, `FalkorDbIndex.applyWrites`/`apply` | `pending_writes.rs:24-179`, `falkordb_index.rs:364-400` |
| `Writes.lean`: `G`, `StageOp`, `stage`, `nodeAttrsSet`, `nodeRowsSet`, `newNodeAttrs`, `bulkNodeAttrs`, `labelsAdded`, `labelsProductAdded`, `labelsRemoved`, `nodesDeleted`, `edgeAttrsSetWith`/`edgeAttrsSet`/`edgeAttrsSetOfType`, `newEdgeAttrs`, `bulkEdgeAttrs`, `edgesDeleted`, `addNodeUnderLabel`, `removeNodeUnderLabel`, `nodeLabelIdsInto`, `edgeType` | `graph_writes.rs:62-393`, `:438-508` |
| `Ddl.lean`: `collectNodeEntries`, `collectEdgeEntries`, `buildColumn`, `createIndex`, `populate`, `indexOnlyWrites`, `applyIndexWrites`, `commitIndexDocs`, `queryIndexNumericNodes/Edges`, `scanRoute`, `newVersion`, `runUnit` | `graph_writes.rs:46-52`, `:404-432`, `:468-552`; `graph.rs:1048-1051`, `:3766-3794`, `:4073-4108`; `node_by_index_scan.rs:306-323`, `edge_by_index_scan.rs:329-349` |

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
  `queryNumeric_spec` (routing: only numeric `Equal`/`Range` on an existing column).
- **#3307, apply once per unit of work** (`Pending.lean`): `sortLog_filter` (stable sort keeps each
  tuple's history), `lwwGo_spec`, `lastWriteWins_spec` (keeps exactly each tuple's last change),
  `applyColumn_eq_sequential` (collapsed `remove_batch`+`insert_batch` = applying every staged change
  one by one), `applyWrites_numeric` (per column; dropped columns skipped), `extend_log` (staging order
  kept per column), `apply_empties`, `set_then_delete`.
- **Writes keep the index in step** (`Writes.lean`): `*_changes` (each staging function stages exactly
  its ops), `lastOp_ops` (what a list of ops does to any tuple), and per path: `nodeAttrsSet_hit/miss`
  and `edgeAttrsSetWith_hit` (update: old entry removed, new present), `nodesDeleted_hit`/`_noadd`,
  `edgesDeleted_hit`/`_noadd` (delete removes every entry, adds none), `labelsRemoved_hit`,
  `labelsAdded_hit`, `newNodeAttrs_hit`.
- **DDL / MVCC** (`Ddl.lean`): `buildColumn_numeric`, `createIndex_builds`, `collect*_spec`,
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
- **CowBTree contract** (`Tree.lean`): the tree is its content list, each op its proven content spec
  (cow_btree / index_layer), which holds under cow_btree's `TreeWF`; `remove_batch` breaks `TreeWF`
  (cow_btree Bug 3) and now has a production caller (`apply_writes` → `remove_encoded`). A model search
  of batch-only sequences found no content/range/empty-page failure. Rows that read through the tree
  are CONDITIONAL on this.
- **Caller uniqueness** (`nodeAttrsSet_hit`, `edgeAttrsSetWith_hit`, `newNodeAttrs_hit`): each node's
  written attribute ids are unique — `Pending::set_node_attribute` keeps them sorted/unique,
  `set_node_attributes` only `debug_assert!`s it (CONDITIONAL rows). Created nodes' `new_labels` /
  bulk / replica record labels are taken as the node's labels (the Rust's own premise, doc
  `graph_writes.rs:95-97`, `:131-132`).
- Schema: attribute names unique (`attrNames.Nodup`: the single `attrs_name` dictionary). The graph's
  mutation of its own stores (`insert_attrs`, label matrix, `remove_all`) is modelled by other
  projects; here the per-tuple outcome is stated against the graph *before* the write.
- `HashMap` iteration order is arbitrary: theorems are per key. `bulk`/`rows`/`product` paths have
  exact staged-ops theorems (`*_changes`); their per-tuple outcome follows from `lastOp_ops` as for
  the other add paths.
-/
