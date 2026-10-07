import FalkorCowBTree.Extract

/-!
# Bug 3 (NEW with #2278): `remove_batch` leaves a single-child non-root branch, at any `BRANCH_MAX`

`Node::remove_batch` (`node.rs:483-538`) repairs under-full children with `merge_underfull`, one level
at a time. A branch `X` that ends with one child is flagged under-full (`1 < BRANCH_MAX / 2` for
`BRANCH_MAX >= 4`) — but only its parent `P` can repair it, and if `P` itself ends with `X` as its
*only* child, `P`'s `merge_underfull` loop does not run (`children.len() > 1`, `node.rs:153`). `P` is
then merged one level up by `combine`, which concatenates child lists: `X` lands in the combined node
with one child and is never re-examined. The tree is no longer well-formed (`TreeWF` / the Rust
`check_invariants`: "branch must have >= 2 children"). One later single-key `remove` that drains `X`'s
lone leaf then leaves an **empty non-root leaf** (`Branch::rebalance` on `X` is a no-op, `node.rs:211`,
and the parent's `combine` keeps the empty page), and the next `Node::min` over it panics
(`leaf.key(0)` out of bounds, `mod.rs:83`) — e.g. in `insert_batch`'s `pack_branches`.

Latent: `remove_batch` has no production caller yet. Confirmed on the real Rust (`cow_btree` sources
copied verbatim into a scratch crate, test appended to `tests.rs`):

```rust
let pairs: Vec<(u64, u64)> = (0..96u64).map(|d| (d, 0)).collect();
let mut t = CowBTree::<2, 4>::from_sorted(&pairs);   // root -> 3 subtrees of 32
t.remove_batch(&(33..64u64).map(|d| (d, 0)).collect::<Vec<_>>());
check_invariants(&t, false);   // panics: "branch must have >= 2 children ...: 1"
assert!(t.remove(32, 0));      // leaves an empty non-root leaf (`leaves()` lists a 0-byte page)
t.insert_batch(&[(0, 1), (95, 1)]);   // panics: "range end index 8 out of range for slice of length 0"
```

Suggested fix: after `combine` merges two branches in the batch path, re-run `merge_underfull` over the
merged node's children (or never return a one-child branch below the root: hand its lone child to the
parent's merge).

The theorems below replay the mechanism on a 12-entry tree with `LEAF_MAX = 2`, `BRANCH_MAX = 4`.
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

/-- `remove_batch([5, 6, 7])`: the middle subtree keeps only `4`, under the branch `.branch [] [lf 4]`. -/
def rb12b : Node :=
  .branch [4] [.branch [2] [.branch [1] [lf 0, lf 1], .branch [3] [lf 2, lf 3]],
               .branch [8, 10] [.branch [] [lf 4], .branch [9] [lf 8, lf 9], .branch [11] [lf 10, lf 11]]]

theorem rb_remove_batch : removeBatchTree c24 3 rb12 [5, 6, 7] = some (rb12b, 3) := by
  simp (config := {decide := true}) [removeBatchTree, removeBatch, route, subtract, stripEmpty, mergeUnderfull,
    cws, combine, delAt, setAt, collapse, rb12, rb12b, lf, c24]

/-- ...which is not a well-formed tree: a non-root branch with one child. -/
theorem rb_remove_batch_not_wf : ¬ TreeWF c24 3 rb12b := by
  simp [TreeWF, WFb, COK, rb12b, lf]

/-- The contents are still right (`removeBatchTree_spec`): only the shape is broken. -/
theorem rb_remove_batch_content : toList 3 rb12b = [0, 1, 2, 3, 4, 8, 9, 10, 11] := by
  simp [toList, rb12b, lf]

/-- One single-key `remove(4)` then leaves an empty non-root leaf as the leftmost page of the root's
    second child... -/
def rb12c : Node :=
  .branch [4] [.branch [2] [.branch [1] [lf 0, lf 1], .branch [3] [lf 2, lf 3]],
               .branch [10] [.branch [8, 9] [.leaf [], lf 8, lf 9], .branch [11] [lf 10, lf 11]]]

theorem rb_then_remove : remove c24 3 rb12b 4 = (rb12c, 3) := by rfl

/-- ...so `Node::min` of that child — computed by `pack_branches` for every non-first child whenever
    `insert_batch` repacks the root — is the out-of-bounds panic. -/
theorem rb_min_panics : minOpt 2 (.branch [10] [.branch [8, 9] [.leaf [], lf 8, lf 9], .branch [11] [lf 10, lf 11]]) = none := by
  rfl

end CowBTree
