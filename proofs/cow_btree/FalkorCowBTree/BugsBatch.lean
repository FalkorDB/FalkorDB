import FalkorCowBTree.Extract
import FalkorCowBTree.BatchWF
import FalkorCowBTree.Assert

/-!
# Bug 3 (W7-btree-1, introduced by #2278, FIXED by #3283 799968a8b): `remove_batch` left a single-child
# non-root branch, at any `BRANCH_MAX`

HISTORY (the pre-#3283 counterexamples (removed; in 17ccc5412) theorems and the pre-#3283 model were removed; in 17ccc5412).
`Node::remove_batch` (`node.rs:483-538` before #3283) repaired under-full children with `merge_underfull`, one level
at a time. A branch `X` that ends with one child is flagged under-full (`1 < BRANCH_MAX / 2` for
`BRANCH_MAX >= 4`) — but only its parent `P` can repair it, and if `P` itself ends with `X` as its
*only* child, `P`'s `merge_underfull` loop does not run (`children.len() > 1`, `node.rs:153`). `P` is
then merged one level up by `combine`, which concatenates child lists: `X` lands in the combined node
with one child and is never re-examined. The tree is no longer well-formed (`TreeWF` / the Rust
`check_invariants`: "branch must have >= 2 children"). One later single-key `remove` that drains `X`'s
lone leaf then leaves an **empty non-root leaf** (`Branch::rebalance` on `X` is a no-op, `node.rs:211`,
and the parent's `combine` keeps the empty page), and the next `Node::min` over it panics
(`leaf.key(0)` out of bounds, `mod.rs:83`) — e.g. in `insert_batch`'s `pack_branches`.

Since #3307 (18836a355) `remove_batch` has a production caller (`FalkorDbIndex::apply_writes` →
`remove_encoded`), so the single-child branch is reachable; the panic chain still needs a single-key
`remove`, which only tests call. Confirmed on the real Rust (`cow_btree` sources
copied verbatim into a scratch crate, test appended to `tests.rs`):

```rust
let pairs: Vec<(u64, u64)> = (0..96u64).map(|d| (d, 0)).collect();
let mut t = CowBTree::<2, 4>::from_sorted(&pairs);   // root -> 3 subtrees of 32
t.remove_batch(&(33..64u64).map(|d| (d, 0)).collect::<Vec<_>>());
check_invariants(&t, false);   // panics: "branch must have >= 2 children ...: 1"
assert!(t.remove(32, 0));      // leaves an empty non-root leaf (`leaves()` lists a 0-byte page)
t.insert_batch(&[(0, 1), (95, 1)]);   // panics: "range end index 8 out of range for slice of length 0"
```

The fix (#3283, issue #3237) is the first suggestion made here: after each `combine` in `merge_underfull`,
`merge_underfull_below` re-runs the repair over the produced node's children. The Rust test
`remove_batch_never_leaves_a_single_child_branch` (`tests.rs:1640`) is the repro above.

The pre-#3283 counterexamples (removed; in 17ccc5412) theorems replay the mechanism on a 12-entry tree with `LEAF_MAX = 2`, `BRANCH_MAX = 4`;
`rb_remove_batch_fixed`, `rb_remove_batch_fixed_wf`, `rb_then_remove_fixed` replay the same input on
main, and `removeBatchTree_wf_asserted` is the general statement (`BatchWF.lean`: `removeBatchTree_wf`).
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

/-- Before #3283, `remove_batch([5, 6, 7])`: the middle subtree keeps only `4`, under the branch `.branch [] [lf 4]`. -/
def rb12b : Node :=
  .branch [4] [.branch [2] [.branch [1] [lf 0, lf 1], .branch [3] [lf 2, lf 3]],
               .branch [8, 10] [.branch [] [lf 4], .branch [9] [lf 8, lf 9], .branch [11] [lf 10, lf 11]]]

/-- One single-key `remove(4)` then leaves an empty non-root leaf as the leftmost page of the root's
    second child... -/
def rb12c : Node :=
  .branch [4] [.branch [2] [.branch [1] [lf 0, lf 1], .branch [3] [lf 2, lf 3]],
               .branch [10] [.branch [8, 9] [.leaf [], lf 8, lf 9], .branch [11] [lf 10, lf 11]]]

/-! ## On main (#3283) -/

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

/-- **W7-btree-1 fixed, on every configuration the const assert admits**: `remove_batch` keeps
    `TreeWF` and leaves exactly the entries not in the batch (`removeBatchTree_wf`, `4 ≤ BRANCH_MAX`
    discharged by the assert). -/
theorem removeBatchTree_wf_asserted (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) (h : Nat) (t : Node)
    (batch : List E) (hw : TreeWF c h t) (hb : SortedLe batch) :
    ∃ t' h', removeBatchTree c h t batch = some (t', h') ∧ TreeWF c h' t' ∧
      toList h' t' = (toList h t).filter (fun e => decide (e ∉ batch)) :=
  removeBatchTree_wf c (constAssert_B4 ha) h t batch hw hb

end CowBTree
