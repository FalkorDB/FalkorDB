import FalkorOpsApply.Ctors
import FalkorOpsApply.CreateAttrs
import FalkorOpsApply.CreateBatch
import FalkorOpsApply.Delete
import FalkorOpsApply.DeleteCancel
import FalkorOpsApply.SetRemove
import FalkorOpsApply.CartVhj
import FalkorOpsApply.RowWrites
import FalkorOpsApply.MergeSlot
import FalkorOpsApply.BatchRows
import FalkorOpsApply.VhjSep
import FalkorOpsApply.OpSep
/-
# ops_apply

Models of `graph/src/runtime/ops/{merge,apply,optional,semi_apply,or_apply_multiplexer,
foreach,union,create,delete,set,remove,cartesian_product,value_hash_join}.rs` and the
argument-batch idiom of `runtime/batch.rs`, against per-row openCypher reference semantics.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import FalkorOpsApply.Correlated
import FalkorOpsApply.Union
import FalkorOpsApply.ForEach
import FalkorOpsApply.Merge
import FalkorOpsApply.MergeEager
