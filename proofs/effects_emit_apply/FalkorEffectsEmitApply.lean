import FalkorEffectsEmitApply.Basic
import FalkorEffectsEmitApply.Emit
import FalkorEffectsEmitApply.Apply
import FalkorEffectsEmitApply.EmitDDL
import FalkorEffectsEmitApply.Liveness
import FalkorEffectsEmitApply.CancelGap

/-!
# effects v3: emit / apply (wave 2)

Models `graph/src/effects/v3/emit.rs`, `graph/src/effects/v3/apply.rs` and the gate in front of the
emitter (`runtime/ops/commit.rs`, `Pending::effects_count` in `runtime/pending.rs`): the emitted
stream, the replica's checked apply arms, and emit-then-apply with cancellation.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
