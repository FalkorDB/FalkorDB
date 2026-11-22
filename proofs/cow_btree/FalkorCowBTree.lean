import FalkorCowBTree.Model
import FalkorCowBTree.Basic
import FalkorCowBTree.Invariant
import FalkorCowBTree.Leaf
import FalkorCowBTree.Insert
import FalkorCowBTree.Remove
import FalkorCowBTree.Cursor
import FalkorCowBTree.Snapshot
import FalkorCowBTree.Batch
import FalkorCowBTree.BatchWF
import FalkorCowBTree.Lookup
import FalkorCowBTree.Extract
import FalkorCowBTree.BatchAssert
import FalkorCowBTree.InsertBatchWF
import FalkorCowBTree.Assert

/-!
# `cow_btree`: Lean 4 model and proofs

Models `graph/src/index/falkordb/data_structures/cow_btree/{mod,node,cursor}.rs` and the list-level
behaviour of `leaf/mod.rs` (leaf byte encodings are in `proofs/index_layer`), and proves that every
tree the API can build is well-formed and that insert, remove, batch ops, cursors and lookups
match their sorted-set specs.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
