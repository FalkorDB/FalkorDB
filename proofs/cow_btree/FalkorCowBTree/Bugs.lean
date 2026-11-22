import FalkorCowBTree.Cursor

/-!
# The `BRANCH_MAX = 3` configuration

`CowBTree::<2, 3>` no longer compiles (`Assert.lean`: `constAssert_rejects_B3`, and
`remove_spec_asserted` for every configuration the assert admits). `c23` and a well-formed tree on it
are kept for those statements.
-/

namespace CowBTree

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
