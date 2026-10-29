import FalkorCowBTree.Cursor

/-!
# Counterexamples (both reproduced in `graph/tests/lean_cow_btree.rs`)
-/

namespace CowBTree

/-! ## Bug 1 (historical, FIXED by #2278, 3597b3a82): `insert_batch` dropped `(u64::MAX, u64::MAX)`

Before #2278 `apply_batch` gave the last child the upper bound `TOP = (u64::MAX, u64::MAX)` and kept a
batch entry for a child only while `entry < upper`. `TOP < TOP` is false, so `TOP` was routed to *no*
child, and the sweep simply ended — the entry was lost. The old sweep and its counterexamples were removed (in 17ccc5412);
`route_keeps_all` / `route_routes_TOP` (`Batch.lean`) prove the fix. -/

/-! ## Bug 2 (historical, FIXED by #2897, 7b60f28db): with `BRANCH_MAX = 3` remove broke the tree

Before #2897 the const assert (`mod.rs:152-155`) only demanded `BRANCH_MAX >= 3`. The underflow flag of a
branch is `children < BRANCH_MAX / 2 = 1` (`node.rs:197`, reported at `:473`): a branch merged
down to ONE child is not reported, so its parent never repairs it. Its lone child then cannot
rebalance (`node.rs:212` guard) and drains to an empty non-root leaf. `is_empty()` then lies, and
a later `insert_batch` hits `Node::min` on the empty leaf (out-of-bounds panic, `node.rs:234`) or the
single-child `debug_assert` (`node.rs:71`). The `pre2897_b3_*` counterexamples (removed; in 17ccc5412) replayed it on
`CowBTree::<2, 3>`, which no longer compiles (`Assert.lean`: `constAssert_rejects_B3`, and
`remove_spec_asserted` for every configuration the assert admits). -/

def c23 : Cfg := ⟨2, 3, by decide, by decide⟩

def insAll (c : Cfg) (xs : List E) : Node × Nat := xs.foldl (fun t x => insert c t.2 t.1 x) (.leaf [], 0)
def remAll (c : Cfg) (t : Node × Nat) (xs : List E) : Node × Nat := xs.foldl (fun t x => remove c t.2 t.1 x) t

/-- Rust: `CowBTree::<2, 3>`, insert `(0, 0..5)`. -/
def t5 : Node × Nat := insAll c23 [0, 1, 2, 3, 4]

theorem t5_shape : t5 = (.branch [2] [.branch [1] [.leaf [0], .leaf [1]], .branch [3] [.leaf [2], .leaf [3, 4]]], 2) := by
  rfl

/-- ...which is a well-formed tree (so the break is caused by `remove`, not by a bad input). -/
theorem t5_wf : TreeWF c23 t5.2 t5.1 := by
  rw [t5_shape]
  simp [TreeWF, WFb, COK, Sorted, HI, W, c23]

end CowBTree
