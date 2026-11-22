/-
# pending_commit

Pending mutations, `Commit` and the attribute store: `graph/src/runtime/pending.rs`,
`graph/src/graph/attribute_store.rs`, deleted-entity snapshots, unique-constraint checks and the
SAVE encoder.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import PendingCommit.Span
import PendingCommit.Arena
import PendingCommit.Labels
import PendingCommit.Props
import PendingCommit.Snapshot
import PendingCommit.Stats
import PendingCommit.BinSearch
import PendingCommit.NameMap
import PendingCommit.Pack
import PendingCommit.Content
import PendingCommit.ContentMerge
import PendingCommit.ContentCompact
import PendingCommit.Store
import PendingCommit.StoreFold
import PendingCommit.StoreApi
import PendingCommit.StoreCodec
import PendingCommit.StoreMem
import PendingCommit.PendingAttrs
import PendingCommit.PendingState
import PendingCommit.PendingRels
import PendingCommit.PendingDeg
import PendingCommit.PendingDocs
import PendingCommit.PendingCommitOp
import PendingCommit.UniqueIndex
import PendingCommit.UniqueQuery
import PendingCommit.PendingCancel
import PendingCommit.CancelRust
import PendingCommit.CommitOpNext
import PendingCommit.StagedBugs
import PendingCommit.Hold
import PendingCommit.SaveRoaring
import PendingCommit.SaveNthLive
import PendingCommit.SaveSlice
import PendingCommit.SaveLayout
import PendingCommit.KeyBorrow
import PendingCommit.AttrInv
import PendingCommit.AccessorBridge
import PendingCommit.AttrCallers
import PendingCommit.AttrAlloc
