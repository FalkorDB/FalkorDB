import FalkorAggOps.Agg
import FalkorAggOps.Ops
import FalkorAggOps.GroupId
import FalkorAggOps.Bitmap
import FalkorAggOps.Ctors
import FalkorAggOps.SortCmp
import FalkorAggOps.SortKey
import FalkorAggOps.AggPlan
import FalkorAggOps.AggState
import FalkorAggOps.AggNext
import FalkorAggOps.StreamOps
import FalkorAggOps.LoadCsv
import FalkorAggOps.Discharge
import FalkorAggOps.SlotReuse
import FalkorAggOps.LoadCsvRedirect
import FalkorAggOps.IncludePendingExact
import FalkorAggOps.MaxScalar

/-
# ops_aggregate

Aggregate, Distinct, Sort, Skip, Limit, Project, Unwind, LOAD CSV, Filter, IncludePending and
the aggregation kernels (`graph/src/runtime/ops/*.rs`, `functions/aggregation.rs`).
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
