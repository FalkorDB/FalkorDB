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
- **W4-matrix-1, fixed by #3250 (history):** `Tensor::resize` shrink never trimmed `me`
  (0f45a8685 tensor.rs:814-828; counterexample removed; in 17ccc5412); repro then
  `cargo test -p graph --test lean_versioned_matrix tensor_shrink_keeps_orphan_me_rows`
  → after shrink `iter_edges` = [(0,0,3),(5,5,1),(5,5,2)], `edge_count` = 2
  (expected [(0,0,3)], 1); after grow + re-add (5,5,9): `get` = [9] but
  `iter_edges` also yields 1,2 and `edge_count` = 3 (expected 2). Not reachable
  from Cypher: the only shrink caller, `rebuild_derived_matrices`, keeps
  `node_cap` ≥ every node id.

Re-target to main 73077bb4a (#3033, #3086): no model change here. #3033 makes
`Tensor::decode` (tensor.rs:1553) refuse a payload that breaks promotion
completeness (`single_me`/`multi_me`); proved in graphblas_wrappers
(`Decode.decodeCheck_ok_iff`, `decodeCheck_accepts_encoded`). tensor.rs fns after
decode moved +20 lines; the `mvcc_graph.rs` rows were re-pointed to the post-#3290
lines (new :89, commit :186, rollback :268, drop :274; #3290's `fork`/`claim` are
covered by other projects; #3290 left `write`'s semantics and `commit`/`rollback`/`drop`
unchanged, so `sstep` stands).

Re-target to main 1d15bf713 (#3250, #2905): `Tensor::resize` (tensor.rs:823) never
shrinks; `resizeT` clamps to `max` and keeps every pair state and `me` row:
`resizeT_eq`, `resizeT_edges` (no entry dropped), `resizeT_allInv`, `resizeT_blkInv`,
`resizeT_hasMultiEdge`, `resizeT_dims`; new accessors `Tensor::nrows`/`ncols`
(`nrowsT_resizeT`). `Cow` gained a debug-only `branched` flag (#2905); the release
model here is unchanged, the debug model is in `runtime_ds` (`CowModel.dbg_*`).
`old_write_leaks_into_new_version` stays as the documented release-build contract.

Re-target to e8f8a3017 (#3022): `CountRows` — `VersionedMatrix::count_in_rows` is exact
(`vmCountInRows_spec`: `|m| + |dp| − |dm|` over the selected rows = the effective entries
there, no underflow, given `dm ⊆ m`), `Tensor::count_pairs_into` (`countPairsInto_spec`).

Re-target to 2c874022a (#2846): `MvccGraph::commit` (mvcc_graph.rs:139) validates the
write version first and rolls back on a refusal (:148-151). `sstep` takes `valid`
(= `Graph::validate`); `snapshot_isolation`/`reader_view_frozen` hold for every
`valid`; new `refused_commit_rolls_back` (committed version unchanged, write flag
cleared, no in-flight version).
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
