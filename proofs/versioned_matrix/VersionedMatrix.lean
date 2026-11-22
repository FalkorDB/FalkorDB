/-
# proofs/versioned_matrix — `VersionedMatrix`, `Delta`, `Tensor`, `Cow`, MVCC

Wave 4 additions (every fn of versioned_matrix.rs / tensor.rs / cow.rs /
mvcc_graph.rs is PROVEN; see COVERAGE.tsv):
- `Policy`     fold policy: no fold on read-only tx / tiny delta, escape hatch,
               monotonicity, write-path decision ⇒ read-path decision, exact form.
- `RowFilter`  bit-exact (`BitVec 64`, `bv_decide`) row filter: never says "no"
               for a recorded row; word index < 512.
- `Mat`/`DeltaBook` every `Delta` method with its bookkeeping: row filter sound in
               every state, counter exact after resync / under exact inserts.
- `VMOps`      wait/latch, wait_all/is_synced, from_matrix, transpose, clear_deltas,
               encode/decode round trip.
- `IterSeek`   `Iter::new` detaching = `from_layers`; `seek` = a fresh iterator.
- `TensorBlocks`/`TensorOps`/`TensorResize` whole-tensor model: me blocks,
               widening, multi_pairs, degrees (no `unreachable!`), fold/dup/wait,
               encode's MSB values, `Iter::new/seek`.

Re-sync to 256ab4d3f (from 43936bab9): no model change; the 24 CHANGED rows are behaviour-preserving.
`should_fold`, `should_fold_read`, `fold_balance`, `delta_dominates_base`, `compound_key(_inverse)` became
`const fn` (#3787); `Tensor::me_block_mut`/`set_all_from_slices`/`remove_all`/`encode`/`decode` were rewritten
with `if let`/`map_or_default`/`unwrap_or_else` and reordered `if` branches (same case split);
`Delta::tombstone_masked` and the `flush` folds (`VersionedMatrix::flush`, `Tensor::flush`) now use the
structural descriptors `S`/`RSC` (#3471), which on layers that only store `true` select the same coordinates as
the valued masks the model reads (see `Delta.lean`); the `Tensor::encode` fast path is already `TensorSave`
(#3438). #3683 made `MvccGraph` hold an `ArcSwap<Graph>` (`fork` returns an owned `Graph`, `commit` takes it by
value and returns the published `Arc<Graph>`), and split `commit_version` into `prepare` + `publish`: the model
(`Mvcc`, `sstep .commit`) is the published version only and is unchanged. Every `file:line` cite was re-pinned
by function name. `VersionedMatrix::count_in_rows` is now PROVEN: `CountRows.vmCountInRowsOn_spec` states the exactness theorem over a
well-formedness-indexed `LayerCountOn P`, and `proofs/graphblas_wrappers` `CountLink.vmCountInRows_linked` supplies
`Matrix::count_in_rows` as the counter (the old `LayerCount`/`vmCountInRows_spec` statements are unchanged).

`TensorSave` models `Encode for Tensor`'s
fast path (tensor.rs:1472-1561): with `dp`/`dm` empty and no multi-edge id it writes the committed `m`
and two empty matrices instead of rebuilding. `fast_base_eq` (same pattern and values as the rebuild,
and the rebuild's `multi` list is empty, from `AllInv`/`BlkInv`), `encPR_eq_encMain` (the earlier
rebuild is `encMain`), `encode_one_state` (check, write and `edge_count` read one state),
`decode_same` (FFI hypothesis `MatrixRT`). The fast path passes the shared committed `m` itself to
`Matrix::encode`, so it depends on graphblas_wrappers `EncodeDup.pr_self_never_unloaded`. Live: 16
fast-path tensors decode as the rebuild (15 byte-identical).


Re-target to main 1d15bf713 (#2905): `Tensor::resize` (tensor.rs:823) never
shrinks; `resizeT` clamps to `max` and keeps every pair state and `me` row:
`resizeT_eq`, `resizeT_edges` (no entry dropped), `resizeT_allInv`, `resizeT_blkInv`,
`resizeT_hasMultiEdge`, `resizeT_dims`; accessors `Tensor::nrows`/`ncols`
(`nrowsT_resizeT`). `Cow` has a debug-only `branched` flag (#2905); the release
model here is unchanged, the debug model is in `runtime_ds` (`CowModel.dbg_*`).
`old_write_leaks_into_new_version` stays as the documented release-build contract.

Re-target to e8f8a3017 (#3022): `CountRows` — `VersionedMatrix::count_in_rows` is exact
(`vmCountInRows_spec`: `|m| + |dp| − |dm|` over the selected rows = the effective entries
there, no underflow, given `dm ⊆ m`), `Tensor::count_pairs_into` (`countPairsInto_spec`).

Re-target to 2c874022a (#2846): `MvccGraph::commit` (mvcc_graph.rs:139) validates the
write version first and rolls back on a refusal (:148-151). `sstep` takes `valid`
(= `Graph::validate`); `snapshot_isolation`/`reader_view_frozen` hold for every
`valid`; new `refused_commit_rolls_back` (committed version unchanged, write flag
cleared, no in-flight version; renamed below).

Re-target to a477d9aa3 (#3657, fork-and-commit): `MvccGraph::write`/`rollback`/the write flag
are gone. `Mvcc` is the published version only; `Ev.rollback` is `Ev.abandon` (drop an
uncommitted version); `single_writer` is replaced by `fork_takes_nothing` and
`refused_commit_rolls_back` by `refused_commit_publishes_nothing`; `snapshot_isolation` and
`reader_view_frozen` are unchanged in statement and hold without the flag invariant. New
`MvccGraph::commit_indexer_batch`/`commit_version` publish a version at the same data
version (see `search_concurrency` `Mvcc.batch_keeps_forks`); the batch changes no logical data
(it only folds deltas and trims attribute stores), so logical content and isolation are
unchanged. The shared `commit_version` tail (`fold_oversized_deltas`, `trim_attr_stores`,
schema bump) is not modelled.
-/
import VersionedMatrix.Delta
import VersionedMatrix.DeltaProofs
import VersionedMatrix.Iter
import VersionedMatrix.Cow
import VersionedMatrix.Tensor
import VersionedMatrix.TensorProofs
import VersionedMatrix.Policy
import VersionedMatrix.RowFilter
import VersionedMatrix.Mat
import VersionedMatrix.DeltaBook
import VersionedMatrix.VMOps
import VersionedMatrix.IterSeek
import VersionedMatrix.TensorBlocks
import VersionedMatrix.TensorOps
import VersionedMatrix.TensorResize
import VersionedMatrix.CowGlue
import VersionedMatrix.CountRows
import VersionedMatrix.TensorSave
