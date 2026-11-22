/-
# binder

The binder's scopes: what a variable id refers to and when ids collide. Models
`graph/src/planner/binder.rs` and the id-minting side of `graph/src/planner/mod.rs`.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import FalkorBinder.Table
import FalkorBinder.Resolve
import FalkorBinder.Projection
import FalkorBinder.Pattern
import FalkorBinder.Scopes
import FalkorBinder.RemoveLabels
import FalkorBinder.Stack
import FalkorBinder.BindExpr
import FalkorBinder.BindIR
import FalkorBinder.Types
import FalkorBinder.Fold
import FalkorBinder.AggEq
import FalkorBinder.CallBoundary
