import FalkorCowBTree.Extract
import FalkorCowBTree.BatchWF
import FalkorCowBTree.Assert

/-!
# `remove_batch` on a height-3 tree, and a follow-up `remove`

`rb12` is a well-formed height-3 tree with `LEAF_MAX = 2`, `BRANCH_MAX = 4`. `remove_batch([5, 6, 7])`
leaves a well-formed tree with the same contents (`rb_remove_batch_fixed`), so a follow-up single-key
`remove` keeps it well-formed (`rb_then_remove_fixed`); `removeBatchTree_wf_asserted` is the general
statement (`BatchWF.lean`: `removeBatchTree_wf`). The Rust test is
`remove_batch_never_leaves_a_single_child_branch` (`tests.rs`):

```rust
let pairs: Vec<(u64, u64)> = (0..96u64).map(|d| (d, 0)).collect();
let mut t = CowBTree::<2, 4>::from_sorted(&pairs);   // root -> 3 subtrees of 32
t.remove_batch(&(33..64u64).map(|d| (d, 0)).collect::<Vec<_>>());
check_invariants(&t, false);
assert!(t.remove(32, 0));
t.insert_batch(&[(0, 1), (95, 1)]);
```
-/

namespace CowBTree

def c24 : Cfg := ⟨2, 4, by decide, by decide⟩

private def lf (x : Nat) : Node := .leaf [x]

/-- A well-formed height-3 tree: root → 3 branches → 2 branches each → 2 one-entry leaves each. -/
def rb12 : Node :=
  .branch [4, 8] [.branch [2] [.branch [1] [lf 0, lf 1], .branch [3] [lf 2, lf 3]],
                  .branch [6] [.branch [5] [lf 4, lf 5], .branch [7] [lf 6, lf 7]],
                  .branch [10] [.branch [9] [lf 8, lf 9], .branch [11] [lf 10, lf 11]]]

theorem rb12_wf : TreeWF c24 3 rb12 := by
  simp [TreeWF, WFb, COK, Sorted, HI, W, c24, rb12, lf]

/-- A single-child-branch shape of `remove_batch([5, 6, 7])`: the middle subtree keeps only `4`, under the branch `.branch [] [lf 4]` (not well-formed; kept as a historical counterexample shape). -/
def rb12b : Node :=
  .branch [4] [.branch [2] [.branch [1] [lf 0, lf 1], .branch [3] [lf 2, lf 3]],
               .branch [8, 10] [.branch [] [lf 4], .branch [9] [lf 8, lf 9], .branch [11] [lf 10, lf 11]]]

/-- One single-key `remove(4)` then leaves an empty non-root leaf as the leftmost page of the root's
    second child... -/
def rb12c : Node :=
  .branch [4] [.branch [2] [.branch [1] [lf 0, lf 1], .branch [3] [lf 2, lf 3]],
               .branch [10] [.branch [8, 9] [.leaf [], lf 8, lf 9], .branch [11] [lf 10, lf 11]]]

/-! ## On main -/

/-- The same `remove_batch([5, 6, 7])` on main: the result is a well-formed tree with the same contents
    (`#eval` shows the lone leaf `4` merged into the next subtree, `.branch [8, 9] [lf 4, lf 8, lf 9]`). -/
theorem rb_remove_batch_fixed : ∃ t' h', removeBatchTree c24 3 rb12 [5, 6, 7] = some (t', h') ∧
    TreeWF c24 h' t' ∧ toList h' t' = [0, 1, 2, 3, 4, 8, 9, 10, 11] := by
  obtain ⟨t', h', he, hw, ht⟩ := removeBatchTree_wf c24 (by decide) 3 rb12 [5, 6, 7] rb12_wf (by decide)
  exact ⟨t', h', he, hw, by rw [ht]; decide⟩

/-- ...so a follow-up single-key `remove` (of any entry) keeps it well-formed: no empty page, no
    `Node::min` panic. -/
theorem rb_then_remove_fixed (x : E) (hx : x < HI) : ∃ t' h', removeBatchTree c24 3 rb12 [5, 6, 7] = some (t', h') ∧
    TreeWF c24 (remove c24 h' t' x).2 (remove c24 h' t' x).1 := by
  obtain ⟨t', h', he, hw, _⟩ := rb_remove_batch_fixed
  exact ⟨t', h', he, (remove_spec c24 (by decide) h' t' x hw hx).1⟩

/-- **On every configuration the const assert admits**: `remove_batch` keeps
    `TreeWF` and leaves exactly the entries not in the batch (`removeBatchTree_wf`, `4 ≤ BRANCH_MAX`
    discharged by the assert). -/
theorem removeBatchTree_wf_asserted (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) (h : Nat) (t : Node)
    (batch : List E) (hw : TreeWF c h t) (hb : SortedLe batch) :
    ∃ t' h', removeBatchTree c h t batch = some (t', h') ∧ TreeWF c h' t' ∧
      toList h' t' = (toList h t).filter (fun e => decide (e ∉ batch)) :=
  removeBatchTree_wf c (constAssert_B4 ha) h t batch hw hb

end CowBTree
