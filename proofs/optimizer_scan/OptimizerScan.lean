/-
# FalkorDB-rs scan selection passes

Models the optimizer passes in `graph/src/planner/optimizer/` (`select_scan_node.rs`,
`utilize_index.rs`, `utilize_node_by_id.rs`, `reorder_labels.rs`) and the runtime pieces their
rewrites hand work to (`evaluate_id_filter`, `NodeByIndexScanOp`, `Index::build_query_node`), and
proves each rewrite selects the nodes the plain `Filter -> scan` plan selects.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import OptimizerScan.IdSeek
import OptimizerScan.Index
import OptimizerScan.IndexCex
import OptimizerScan.IndexSound
import OptimizerScan.IndexPipeline
import OptimizerScan.IndexArray
import OptimizerScan.ScanOrder
import OptimizerScan.EdgeInline
import OptimizerScan.Cleanup
import OptimizerScan.ScanTree
import OptimizerScan.Salvage
import OptimizerScan.BoolOperand
