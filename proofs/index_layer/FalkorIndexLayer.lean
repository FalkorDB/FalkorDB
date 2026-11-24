/-
# index_layer

The index layer: `graph/src/index/mod.rs`, `indexer.rs`, `falkordb/` and the RediSearch bridge.
Key and TAG encoding, query building, index-scan exactness against a full scan, index
maintenance, tickets, field metadata and the COW B+-tree page formats.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import FalkorIndexLayer.Model
import FalkorIndexLayer.Proofs
import FalkorIndexLayer.Temporal
import FalkorIndexLayer.Strings
import FalkorIndexLayer.Maintenance
import FalkorIndexLayer.Bugs
import FalkorIndexLayer.Meta
import FalkorIndexLayer.IndexM
import FalkorIndexLayer.SetVector
import FalkorIndexLayer.IndexerM
import FalkorIndexLayer.Iter
import FalkorIndexLayer.IndexGaps
import FalkorIndexLayer.EdgeResync
import FalkorIndexLayer.InList
import FalkorIndexLayer.LeafBytes
import FalkorIndexLayer.LeafMerge
import FalkorIndexLayer.LeafAos
import FalkorIndexLayer.LeafCompact
import FalkorIndexLayer.LeafIndexed
import FalkorIndexLayer.LeafIndexedOps
import FalkorIndexLayer.LeafIndexedMerge
import FalkorIndexLayer.LeafDispatch
import FalkorIndexLayer.LeafBlockCopy
import FalkorIndexLayer.LeafBlockCopy2
import FalkorIndexLayer.LeafOps
import FalkorIndexLayer.LeafTree
import FalkorIndexLayer.LeafAosD
