import OpsTraverse.Basic
import OpsTraverse.CondTraverse
import OpsTraverse.BidirDedup
import OpsTraverse.VarLen
import OpsTraverse.ShortestPaths
import OpsTraverse.Scans
import OpsTraverse.Drive
import OpsTraverse.IndexScan
import OpsTraverse.VarLenGlue
import OpsTraverse.CTGlue
import OpsTraverse.ASPFold
import OpsTraverse.ASPInv
import OpsTraverse.ASPStep
import OpsTraverse.ASPFinal
import OpsTraverse.ASPMain
import OpsTraverse.ASPFuel
import OpsTraverse.SlotFeed
import OpsTraverse.PathVlt
import OpsTraverse.ExpandIntoPending
import OpsTraverse.UnionStream
import OpsTraverse.UnionGlue
import OpsTraverse.AlgoPaths
import OpsTraverse.EvalSP
import OpsTraverse.NeighborIter
import OpsTraverse.RecordFound
import OpsTraverse.RunAlgo
import OpsTraverse.SearchBase
import OpsTraverse.BfsFind
import OpsTraverse.Dijkstra
import OpsTraverse.DijkstraNat
import OpsTraverse.Enum
import OpsTraverse.EnumSpec
import OpsTraverse.BidirBfs
import OpsTraverse.AlgoSearch
import OpsTraverse.BspWire

/-
# Traversal operators: CondTraverse, ExpandInto, var-length, scans, allShortestPaths, PathBuilder

A Lean 4 model of FalkorDB-rs's pattern-matching operators (`graph/src/runtime/ops/`: `cond_traverse.rs`,
`expand_into.rs`, `cond_var_len_traverse.rs`, `all_shortest_paths.rs`, `path_builder.rs`, the scans,
`algo_procedures.rs`), checked against openCypher's semantics and against C. The GraphBLAS matrix
product enters only as a hypothesis (`CondTraverse.MxmSpec`).
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
