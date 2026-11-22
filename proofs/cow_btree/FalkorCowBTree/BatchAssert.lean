import FalkorCowBTree.Extract
import FalkorCowBTree.BatchWF
import FalkorCowBTree.Assert

/-!
# `remove_batch` on every configuration the const assert admits
-/

namespace CowBTree

/-- **On every configuration the const assert admits**: `remove_batch` keeps
    `TreeWF` and leaves exactly the entries not in the batch (`removeBatchTree_wf`, `4 ≤ BRANCH_MAX`
    discharged by the assert). -/
theorem removeBatchTree_wf_asserted (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) (h : Nat) (t : Node)
    (batch : List E) (hw : TreeWF c h t) (hb : SortedLe batch) :
    ∃ t' h', removeBatchTree c h t batch = some (t', h') ∧ TreeWF c h' t' ∧
      toList h' t' = (toList h t).filter (fun e => decide (e ∉ batch)) :=
  removeBatchTree_wf c (constAssert_B4 ha) h t batch hw hb

end CowBTree
