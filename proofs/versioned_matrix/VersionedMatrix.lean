/-
# proofs/versioned_matrix

Models `VersionedMatrix`, `Tensor`, `Cow` and `MvccGraph` (`graph/src/graph/graphblas/versioned_matrix.rs`,
`tensor.rs`, `cow.rs`, `mvcc_graph.rs`): the delta layers and their fold policy, the relationship
tensor, copy-on-write handles and snapshot isolation across forks and commits.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
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
import VersionedMatrix.TensorClean
