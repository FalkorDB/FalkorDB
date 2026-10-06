import FalkorCowBTree.Cursor

/-!
# Counterexamples (both reproduced in `graph/tests/lean_cow_btree.rs`)
-/

namespace CowBTree

/-! ## Bug 1: `insert_batch` drops `(u64::MAX, u64::MAX)` (`node.rs:198-206`)

`apply_batch` gives the last child the upper bound `TOP = (u64::MAX, u64::MAX)` and keeps a batch
entry for a child only while `entry < upper`. `TOP < TOP` is false, so `TOP` is routed to *no*
child, and the sweep simply ends — the entry is lost. -/

/-- No slice handed to any child ever contains `TOP`, whatever the separators. -/
theorem route_never_routes_TOP (seps : List E) (hs : ∀ s ∈ seps, s ≤ TOP) :
    ∀ (cs : List Node) (batch : List E), ∀ p ∈ route seps cs batch, TOP ∉ p.2 := by
  intro cs
  induction cs generalizing seps with
  | nil => intro batch p hp; simp [route] at hp
  | cons ch cs ih =>
    intro batch p hp
    simp only [route, List.mem_cons] at hp
    rcases hp with rfl | hp
    · intro hT
      have := mem_takeWhile_p hT
      simp only [decide_eq_true_eq] at this
      have hup : seps.head?.getD TOP ≤ TOP := by
        cases seps with
        | nil => simp
        | cons s ss => simpa using hs s List.mem_cons_self
      omega
    · exact ih seps.tail (fun s h => hs s (List.mem_of_mem_tail h)) _ p hp

/-- Concretely: a two-leaf root, batch `[TOP]` — the routed slices are all empty. -/
theorem route_drops_TOP_example :
    route [enc 2 0] [.leaf [enc 0 0, enc 1 0], .leaf [enc 2 0, enc 3 0]] [TOP] =
      [(.leaf [enc 0 0, enc 1 0], []), (.leaf [enc 2 0, enc 3 0], [])] := by
  simp [route, enc, TOP, W]

/-! ## Bug 2: with `BRANCH_MAX = 3` (allowed by the assert, `mod.rs:131`) remove breaks the tree

The underflow flag of a branch is `children < BRANCH_MAX / 2 = 1` (`node.rs:380`): a branch merged
down to ONE child is not reported, so its parent never repairs it. Its lone child then cannot
rebalance (`node.rs:122` guard) and drains to an empty non-root leaf. `is_empty()` then lies, and
a later `insert_batch` hits `Node::min` on the empty leaf (out-of-bounds panic, `node.rs:159`) or the
single-child `debug_assert` (`node.rs:71`). -/

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

/-- One remove already leaves a single-child branch (`TreeWF` violated). -/
theorem b3_one_remove : remAll c23 t5 [0] =
    (.branch [2] [.branch [] [.leaf [1]], .branch [3] [.leaf [2], .leaf [3, 4]]], 2) := by
  rfl

theorem b3_one_remove_not_wf : ¬ TreeWF c23 (remAll c23 t5 [0]).2 (remAll c23 t5 [0]).1 := by
  rw [b3_one_remove]
  simp [TreeWF, WFb, COK]

/-- Removing everything leaves a tree with no entries whose `is_empty()` is `false`. -/
theorem b3_drained_is_empty_lies :
    toList (remAll c23 t5 [0, 1, 2, 3, 4]).2 (remAll c23 t5 [0, 1, 2, 3, 4]).1 = [] ∧
    isEmpty (remAll c23 t5 [0, 1, 2, 3, 4]).1 = false := by
  decide

/-- ...and its right subtree's `Node::min` is the panic case. -/
theorem b3_drained_min_panics :
    remAll c23 t5 [0, 1, 2, 3, 4] =
      (.branch [2] [.branch [] [.leaf []], .branch [] [.leaf []]], 2) ∧
    minOpt 1 (.branch [] [.leaf []]) = none :=
  ⟨rfl, rfl⟩

end CowBTree
