import FalkorNativeIndex.Float
import FalkorNativeIndex.IntF
import FalkorNativeIndex.Value
import FalkorNativeIndex.Agree
import FalkorNativeIndex.Tree
import FalkorNativeIndex.Contract
import FalkorNativeIndex.Numeric
import FalkorNativeIndex.Index
import FalkorNativeIndex.Query
import FalkorNativeIndex.Pending
import FalkorNativeIndex.Writes
import FalkorNativeIndex.Build
import FalkorNativeIndex.Ddl
import FalkorNativeIndex.Scan
import FalkorNativeIndex.Unique
import FalkorNativeIndex.Decline
import FalkorNativeIndex.Batch
import FalkorNativeIndex.Rebase
import FalkorNativeIndex.Cch
import FalkorNativeIndex.Memory
import FalkorNativeIndex.Rows
import FalkorNativeIndex.Bridge
import FalkorNativeIndex.Online
import FalkorNativeIndex.CchOrder
import FalkorNativeIndex.CchTopo
import FalkorNativeIndex.CchMetric
import FalkorNativeIndex.CchPaths
import FalkorNativeIndex.CchSearch
import FalkorNativeIndex.CchRoute
import FalkorNativeIndex.CchSmall
import FalkorNativeIndex.CchSeq
import FalkorNativeIndex.Tag
import FalkorNativeIndex.TagSel

/-
# `native_index`: the FalkorDB native numeric index (`graph/src/index/falkordb/**`)

Models the f64/i64 key encoding, the numeric index (scalar and array trees), build state and indexer batches,
write staging, scans, DDL/MVCC, memory usage and the CCH path index. Index scan = label scan + filter.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
