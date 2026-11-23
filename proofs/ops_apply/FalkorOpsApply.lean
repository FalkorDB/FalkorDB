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

The Apply/Optional/SemiApply/OR-apply batching theorems are proofs for modelled `OpSep.Adm` trees. The
premise that the sub-plan the `can_batch` guard admits is such a tree is NOT discharged: it is the hypothesis
of the six CONDITIONAL rows. `guard_admits_classified` classifies the admitted IR kinds (no writing operator is admitted since #2968) but
the bidirectional-dedup `CondTraverse` (#3106) has no `Adm` constructor (counterexample), and
the planner's output is not machine-checked to land in `Adm`.
-/
import FalkorOpsApply.Correlated
import FalkorOpsApply.Union
import FalkorOpsApply.ForEach
import FalkorOpsApply.Merge
import FalkorOpsApply.MergeEager
