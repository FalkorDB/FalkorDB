import FalkorNativeIndex.Tree
import FalkorCowBTree.InsertBatchWF
import FalkorCowBTree.Extract

/-!
# The `CowBTree` contract of `Tree.lean`, discharged

`Tree.lean` models the tree as its sorted entry list, each operation as its content spec. Those specs
are proven in `proofs/cow_btree` for well-formed trees (`TreeWF`). Since #3283 (799968a8b, W7-btree-1
fixed) every tree the Rust API can build is well-formed (`CowBTree.Reachable_wf_asserted`: `new`,
`from_sorted`, then any sequence of `insert`, `remove`, `insert_batch`, `remove_batch`), so the
hypothesis the numeric-index rows rested on is discharged. Each theorem below takes a reachable tree
on a configuration the const assert admits and shows the Rust operation (the cow_btree model) returns
a reachable tree whose entries are exactly this project's `t*` model of it.
-/

namespace NI

open CowBTree (Cfg Node Reachable constAssert HI SortedLe toList)

/-- `CowBTree::from_sorted` (`cow_btree/mod.rs:190`) is `tFromSorted`. -/
theorem contract_fromSorted (c : Cfg) (pairs : List Nat) (hs : Sorted pairs) (hlt : ∀ e ∈ pairs, e < HI) :
    Reachable c (CowBTree.fromSorted c pairs).2 (CowBTree.fromSorted c pairs).1 ∧
      toList (CowBTree.fromSorted c pairs).2 (CowBTree.fromSorted c pairs).1 = tFromSorted pairs :=
  ⟨.fromSorted pairs hs hlt, (CowBTree.fromSorted_wf c pairs hs hlt).2⟩

/-- `CowBTree::insert` (`cow_btree/mod.rs:223`) is `tInsert`. -/
theorem contract_insert (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) {h : Nat} {t : Node}
    (hr : Reachable c h t) (x : Nat) (hx : x < HI) :
    Reachable c (CowBTree.insert c h t x).2 (CowBTree.insert c h t x).1 ∧
      toList (CowBTree.insert c h t x).2 (CowBTree.insert c h t x).1 = tInsert (toList h t) x :=
  ⟨.insert x hr hx, (CowBTree.insert_spec c h t x (CowBTree.Reachable_wf_asserted c D ha hr) hx).2⟩

/-- `CowBTree::remove` (`cow_btree/mod.rs:260`) is `tRemove`. -/
theorem contract_remove (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) {h : Nat} {t : Node}
    (hr : Reachable c h t) (x : Nat) (hx : x < HI) :
    Reachable c (CowBTree.remove c h t x).2 (CowBTree.remove c h t x).1 ∧
      toList (CowBTree.remove c h t x).2 (CowBTree.remove c h t x).1 = tRemove (toList h t) x :=
  ⟨.remove x hr hx,
    (CowBTree.remove_spec c (CowBTree.constAssert_B4 ha) h t x (CowBTree.Reachable_wf_asserted c D ha hr) hx).2⟩

/-- `CowBTree::insert_batch` (`cow_btree/mod.rs:241`) is `tInsertBatch`. -/
theorem contract_insertBatch (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) {h : Nat} {t : Node}
    (hr : Reachable c h t) (batch : List Nat) (hb : SortedLe batch) (hlt : ∀ e ∈ batch, e < HI) :
    Reachable c (CowBTree.insertBatch c h t batch).2 (CowBTree.insertBatch c h t batch).1 ∧
      toList (CowBTree.insertBatch c h t batch).2 (CowBTree.insertBatch c h t batch).1 =
        tInsertBatch (toList h t) batch := by
  have hw := CowBTree.Reachable_wf_asserted c D ha hr
  obtain ⟨_, hs, hm⟩ := CowBTree.insertBatch_wf c h t batch hw hb hlt
  refine ⟨.insertBatch batch hr hb hlt, sorted_ext hs (sorted_insertBatch batch _ (CowBTree.TreeWF_sorted hw))
    (fun e => by rw [hm e, mem_insertBatch])⟩

/-- `CowBTree::remove_batch` (`cow_btree/mod.rs:286`) is `tRemoveBatch` (and does not panic on a
    sorted batch). -/
theorem contract_removeBatch (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) {h : Nat} {t : Node}
    (hr : Reachable c h t) (batch : List Nat) (hb : SortedLe batch) :
    ∃ t' h', CowBTree.removeBatchTree c h t batch = some (t', h') ∧ Reachable c h' t' ∧
      toList h' t' = tRemoveBatch (toList h t) batch := by
  obtain ⟨t', h', he, _, ht⟩ :=
    CowBTree.removeBatchTree_wf c (CowBTree.constAssert_B4 ha) h t batch (CowBTree.Reachable_wf_asserted c D ha hr) hb
  refine ⟨t', h', he, .removeBatch batch hr hb he, ?_⟩
  rw [ht]
  simp [tRemoveBatch]

/-- `CowBTree::range` (`cow_btree/mod.rs:314`, collected; `n` bounds the collection) is `tRange`. -/
theorem contract_range (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) {h : Nat} {t : Node}
    (hr : Reachable c h t) (lo hi n : Nat) (hn : (toList h t).length ≤ n) :
    CowBTree.rangeWith CowBTree.DocExtract h t lo hi n = tRange (toList h t) lo hi :=
  CowBTree.rangeDocs_spec c h t lo hi n (CowBTree.Reachable_wf_asserted c D ha hr) hn

end NI
