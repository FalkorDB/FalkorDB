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
- **Bug (API-level, confirmed):** `Tensor::resize` shrink never trims `me`
  (tensor.rs:814-828): `shrink_orphans_me`; repro
  `cargo test -p graph --test lean_versioned_matrix tensor_shrink_keeps_orphan_me_rows`
  → after shrink `iter_edges` = [(0,0,3),(5,5,1),(5,5,2)], `edge_count` = 2
  (expected [(0,0,3)], 1); after grow + re-add (5,5,9): `get` = [9] but
  `iter_edges` also yields 1,2 and `edge_count` = 3 (expected 2). Not reachable
  from Cypher: the only shrink caller, `rebuild_derived_matrices`, keeps
  `node_cap` ≥ every node id (`shrink_safe` proves that case correct).

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
