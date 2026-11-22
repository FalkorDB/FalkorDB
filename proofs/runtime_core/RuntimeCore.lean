/-
# runtime_core: `graph/src/runtime/runtime.rs`, `runtime/ops/mod.rs`, `runtime/ops/batched_result_emitter.rs`

Models the runtime's graph access, record-cap budget, batched emitter, `run_batch`, entity accessors and result-set cap.
See README.md for the theorems, open bugs (B1-B3) and gaps; COVERAGE.tsv for the per-fn map.
-/
import RuntimeCore.Budget
import RuntimeCore.Emitter
import RuntimeCore.Query
import RuntimeCore.RunBatch
import RuntimeCore.Accessors
import RuntimeCore.Misc
import RuntimeCore.RelAccessors
import RuntimeCore.Plan
import RuntimeCore.Cell
import RuntimeCore.Reuse
import RuntimeCore.QueryGraph
import RuntimeCore.Discharge
