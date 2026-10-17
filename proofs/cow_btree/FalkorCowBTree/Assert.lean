import FalkorCowBTree.Lookup
import FalkorCowBTree.Bugs

/-!
# The const assert of `CowBTree::{default,new,from_sorted}` (`mod.rs:151-160`, `:173-182`, `:191-200`)

#2897 (7b60f28db) raised the `BRANCH_MAX` bound from 3 to 4; #3285 (cab3cf192) gated `DOC_BYTES` with
`is_read_width` (`mod.rs:88`). `Cfg` keeps the weaker `3 ≤ B` that `pack_branches` needs, so that the
historical `BRANCH_MAX = 3` counterexample (`Bugs.lean`) can still be stated; `constAssert` is the
assert a configuration must pass to build a tree at all.

* `constAssert_B4`, `asserted_single_child_underfull`: under the assert a branch merged down to one
  child is flagged under-full (`1 < BRANCH_MAX / 2`), the contract `remove` needs.
* `remove_spec_asserted`, `removeRet_spec_asserted`: `remove` keeps `TreeWF` and erases exactly the
  tuple on every configuration the assert admits — the `4 ≤ B` premise of `remove_spec` is discharged.
* `pre2897_b3_excluded`, `pre2897_assert_admitted_c23`, `constAssert_rejects_B3`: the W1 #13 bug 2
  configuration `CowBTree::<2, 3>` passed the old assert and is rejected by the new one.
-/

namespace CowBTree

/-- `is_read_width` (`mod.rs:88-90`): `matches!(width, 1 | 2 | 4 | 8)` (bytes proven in `proofs/index_layer`). -/
def isReadWidth (w : Nat) : Bool := w == 1 || w == 2 || w == 4 || w == 8

/-- The const assert (`mod.rs:152-159`): `LEAF_MAX >= 2 && LEAF_MAX <= 256 && BRANCH_MAX >= 4` and
    `is_read_width(DOC_BYTES)`. -/
def constAssert (L B D : Nat) : Bool := decide (2 ≤ L) && decide (L ≤ 256) && decide (4 ≤ B) && isReadWidth D

/-- HISTORICAL: the assert before #2897 / #3285 (`BRANCH_MAX >= 3`, `DOC_BYTES` in `1..=8`). -/
def pre2897_constAssert (L B D : Nat) : Bool :=
  decide (2 ≤ L) && decide (L ≤ 256) && decide (3 ≤ B) && decide (1 ≤ D) && decide (D ≤ 8)

theorem constAssert_B4 {L B D : Nat} (h : constAssert L B D = true) : 4 ≤ B := by
  simp only [constAssert, Bool.and_eq_true, decide_eq_true_eq] at h; exact h.1.2

/-- Every configuration the assert admits is a `Cfg`. -/
def Cfg.ofAssert (L B D : Nat) (h : constAssert L B D = true) : Cfg :=
  ⟨L, B, by simp only [constAssert, Bool.and_eq_true, decide_eq_true_eq] at h; exact ⟨h.1.1.1, h.1.1.2⟩,
    by have := constAssert_B4 h; omega⟩

/-- **#2897: under the assert, a single-child branch is flagged under-full** (`node.rs:197`), so its
    parent repairs it. -/
theorem asserted_single_child_underfull (c : Cfg) (D : Nat) (h : constAssert c.L c.B D = true) (n : Node) :
    branchUnderfull c [n] = true := by
  have := constAssert_B4 h
  simp only [branchUnderfull, List.length_singleton, decide_eq_true_eq]; omega

/-- **#2897: `CowBTree::remove` (`mod.rs:260`) is correct on every configuration the assert admits.** -/
theorem remove_spec_asserted (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) (h : Nat) (t : Node)
    (x : E) (hw : TreeWF c h t) (hx : x < HI) :
    TreeWF c (remove c h t x).2 (remove c h t x).1 ∧
    toList (remove c h t x).2 (remove c h t x).1 = (toList h t).erase x :=
  remove_spec c (constAssert_B4 ha) h t x hw hx

theorem removeRet_spec_asserted (c : Cfg) (D : Nat) (ha : constAssert c.L c.B D = true) (h : Nat) (t : Node)
    (x : E) (hw : TreeWF c h t) (hx : x < HI) :
    (removeRet c h t x).2 = decide (x ∈ toList h t) ∧
    toList (removeRet c h t x).1.2 (removeRet c h t x).1.1 = (toList h t).erase x :=
  removeRet_spec c (constAssert_B4 ha) h t x hw hx

/-- `BRANCH_MAX = 3` is rejected whatever the other parameters. -/
theorem constAssert_rejects_B3 (L D : Nat) : constAssert L 3 D = false := by
  simp [constAssert]

/-- **W1 #13 bug 2 is excluded by #2897**: the counterexample configuration `CowBTree::<2, 3>` of
    `pre2897_b3_one_remove_not_wf` / `pre2897_b3_drained_is_empty_lies` / `pre2897_b3_drained_min_panics`
    no longer passes the const assert, for any `DOC_BYTES`. -/
theorem pre2897_b3_excluded (D : Nat) : constAssert c23.L c23.B D = false := constAssert_rejects_B3 2 D

/-- ...while the old assert admitted it (with the default `DOC_BYTES = 8`). -/
theorem pre2897_assert_admitted_c23 : pre2897_constAssert c23.L c23.B 8 = true := rfl

/-- The default configuration `CowBTree::<256, 256, 8>` passes. -/
theorem constAssert_default : constAssert 256 256 8 = true := rfl

end CowBTree
