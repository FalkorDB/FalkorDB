/-
# optimizer_rewrites

The optimizer passes (`graph/src/planner/optimizer/*.rs`): push-down of filters, reduce passes,
hash-join and fusion rewrites, constant evaluation and the reference oracle.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import FalkorOptimizer.Basic
import FalkorOptimizer.ConstEval
import FalkorOptimizer.References
import FalkorOptimizer.References2
import FalkorOptimizer.Passes
import FalkorOptimizer.Helpers
import FalkorOptimizer.Sequence
import FalkorOptimizer.CallBoundary
import FalkorOptimizer.AndOf
import FalkorOptimizer.PushPlan
