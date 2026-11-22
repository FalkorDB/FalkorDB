/-
# GraphBLAS FFI wrappers

Models the Rust side of the GraphBLAS C API in `graph/src/graph/graphblas/{matrix,vector,mod}.rs` and
the decode and iterator code of `tensor.rs`: status-code handling, `Arc<Handle>` ownership, row
iterators, blob and tensor decoding, `delta_lmxm`, `Encode<19>` of a matrix. The C calls are
axiomatised as hypotheses; the wrapper logic is proved.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import GraphblasWrappers.GrbInfo
import GraphblasWrappers.Index
import GraphblasWrappers.Own
import GraphblasWrappers.RowIter
import GraphblasWrappers.Codes
import GraphblasWrappers.Decode
import GraphblasWrappers.Misc
import GraphblasWrappers.Glue
import GraphblasWrappers.Serial
import GraphblasWrappers.Bulk
import GraphblasWrappers.DeltaLmxm
import GraphblasWrappers.CountLink
import GraphblasWrappers.EncodeDup
