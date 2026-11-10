import FalkorNativeIndex.Rebase
/-!
# `GRAPH.MEMORY USAGE` counts the native index (#3569) and `VersionedIndex::batch_len` (#3568)

| Lean | Rust |
| --- | --- |
| `NumericIndex.memoryUsage` | `NumericIndex::memory_usage` (`numeric.rs:296`) = `tree.heap_bytes()` |
| `IndexColumn.memoryUsage` | `VersionedIndex::memory_usage` for `IndexColumn` (`falkordb_index.rs:405`) |
| `EN.memoryUsage` | `IndexEntry::memory_usage` (`versioned.rs:351`) |
| `memoryUsage` | `FalkorDbIndex::memory_usage` (`falkordb_index.rs:582`) |
| `IndexColumn.batchLen` | `VersionedIndex::batch_len` for `IndexColumn` (`falkordb_index.rs:382`) = `EncodedTuples::len` |

`CowBTree::heap_bytes` (a walk of every page) is proved in `proofs/cow_btree` (`heapBytes_eq`, for a tree
satisfying `TreeWF`; every tree reachable through the public API does: `Reachable_wf_asserted`); it is the
parameter `hb` here, so these theorems say how the sizes are combined, not what a page weighs. The
serialized size of a `RoaringTreemap` (`Written::since_last_indexer_step`) is the parameter `ser`.
-/
namespace NI

/-- `NumericIndex::memory_usage`: `hb` is `CowBTree::heap_bytes` of the tree. -/
def NumericIndex.memoryUsage (hb : NumericIndex → Nat) (ix : NumericIndex) : Nat := hb ix

def IndexColumn.memoryUsage (hb : NumericIndex → Nat) : IndexColumn → Nat
  | .numeric ix => ix.memoryUsage hb

/-- `IndexEntry::memory_usage`: the column's pages, plus, while `Building` with entities written since a batch
    last committed, their serialized size. `by_this_version` is not counted (a published version has it
    empty). -/
def EN.memoryUsage (hb : NumericIndex → Nat) (ser : List Nat → Nat) (e : EN) : Nat :=
  let written := match e.state with
    | .building _ w => if w.since.isEmpty then 0 else ser w.since
    | .ready => 0
  e.index.memoryUsage hb + written

/-- `FalkorDbIndex::memory_usage`: every node column and every edge column. -/
def memoryUsage (hb : NumericIndex → Nat) (ser : List Nat → Nat) (nodeCols edgeCols : List EN) : Nat :=
  ((nodeCols ++ edgeCols).map (EN.memoryUsage hb ser)).sum

theorem EN.memoryUsage_ready (hb : NumericIndex → Nat) (ser : List Nat → Nat) (e : EN)
    (h : e.state = .ready) : e.memoryUsage hb ser = e.index.memoryUsage hb := by
  simp [EN.memoryUsage, h]

theorem EN.memoryUsage_building (hb : NumericIndex → Nat) (ser : List Nat → Nat) (e : EN) (n : BuildCursor)
    (w : Written) (h : e.state = .building n w) :
    e.memoryUsage hb ser = e.index.memoryUsage hb + (if w.since.isEmpty then 0 else ser w.since) := by
  simp [EN.memoryUsage, h]

/-- Nothing written since the last batch: the pages alone. -/
theorem EN.memoryUsage_building_quiet (hb : NumericIndex → Nat) (ser : List Nat → Nat) (e : EN)
    (n : BuildCursor) (w : Written) (h : e.state = .building n w) (hs : w.since = []) :
    e.memoryUsage hb ser = e.index.memoryUsage hb := by
  simp [EN.memoryUsage, h, hs]

theorem memoryUsage_nil (hb : NumericIndex → Nat) (ser : List Nat → Nat) : memoryUsage hb ser [] [] = 0 := rfl

/-- The node columns and the edge columns add. -/
theorem memoryUsage_add (hb : NumericIndex → Nat) (ser : List Nat → Nat) (ns es : List EN) :
    memoryUsage hb ser ns es =
      (ns.map (EN.memoryUsage hb ser)).sum + (es.map (EN.memoryUsage hb ser)).sum := by
  simp [memoryUsage, List.map_append, List.sum_append]

/-- `batch_len`: the number of tuples in the batch (`EncodedTuples::len`; `IndexColumn.batchLen` is defined in `Batch.lean`, where the fit reads it). -/
theorem IndexColumn.batchLen_spec (b : EncodedTuples) : IndexColumn.batchLen b = b.scalar.length := rfl

end NI
