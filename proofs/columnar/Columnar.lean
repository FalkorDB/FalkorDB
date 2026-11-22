import Columnar.Column
import Columnar.Batch
import Columnar.Row
import Columnar.Concat
import Columnar.Emitter
import Columnar.VectorExpr
import Columnar.Arith
import Columnar.Lanes
import Columnar.Gather
import Columnar.BatchMisc
import Columnar.Dispatch
import Columnar.CallBoundary
import Columnar.VecEval
import Columnar.CoreOps
import Columnar.AndOrLane
import Columnar.CaseLane
import Columnar.FuncLane
import Columnar.HasLabelsFn
import Columnar.LaneWF
import Columnar.CoreBridge
import Columnar.RuntimeReads
import Columnar.EvalBatch

/-
# Columnar batch layer: every columnar operation = its row-at-a-time definition

Lean 4 model of `runtime/batch.rs`, `runtime/vectorized.rs`, `runtime/vector_expr.rs`,
`runtime/row.rs` and `runtime/ops/batched_result_emitter.rs`, with proofs that each columnar
operation (gather, compaction, selection, concat, comparison and arithmetic lanes, `VectorEval`)
equals its per-row definition. `f64` is abstract (`FloatModel`).
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
