/-
# Plan-depth guard and plan stack (#3304)

| here | there |
| --- | --- |
| `T`, `TL` | `DynTree<IR>`; `cp` = the node is `IR::CartesianProduct` |
| `size`, `height` | `DynTree::len`, `root().height()` (a leaf has height 0) |
| `bound` | `optimizer_depth_bound` mod.rs:798 (recursive spec of the iterative walk) |
| `stackSizeFor` | `stack_size_for` mod.rs:753 (`usize` saturation not modelled: `Nat`) |
| `withPlanStack` | `with_plan_stack` mod.rs:768 |
| `withOptimizerStack` | `with_optimizer_stack` mod.rs:839 |
| `checkDepth` | `Planner::check_depth` mod.rs:988 |
| `checkHeight` | `Planner::check_height` mod.rs:999 |

`stacker::grow` runs `f` on a fresh stack segment of the thread, so the value `f`
returns is the same; the model returns that value together with the segment size
reserved (`none` = ran on the caller's stack).
-/
namespace PlannerBuild.PlanDepth

mutual
inductive T where
  | node (cp : Bool) (cs : TL)
inductive TL where
  | nil
  | cons (t : T) (ts : TL)
end

mutual
def size : T → Nat
  | .node _ cs => 1 + sizeL cs
def sizeL : TL → Nat
  | .nil => 0
  | .cons t ts => size t + sizeL ts
end

def lenL : TL → Nat
  | .nil => 0
  | .cons _ ts => 1 + lenL ts

mutual
def height : T → Nat
  | .node _ .nil => 0
  | .node _ cs@(.cons _ _) => 1 + heightL cs
def heightL : TL → Nat
  | .nil => 0
  | .cons t ts => max (height t) (heightL ts)
end

def LEVELS_PER_CP_BRANCH : Nat := 3

-- `optimizer_depth_bound`: `extra[i] + if leaf {0} else {1 + below[i]}`, where
-- `below` is the worst child and `extra = 3 · #children` on a `CartesianProduct`.
mutual
def bound : T → Nat
  | .node _ .nil => 0
  | .node cp cs@(.cons _ _) =>
      (if cp then LEVELS_PER_CP_BRANCH * lenL cs else 0) + (1 + boundL cs)
def boundL : TL → Nat
  | .nil => 0
  | .cons t ts => max (bound t) (boundL ts)
end

-- No `CartesianProduct` anywhere in the tree.
mutual
def noCP : T → Prop
  | .node cp cs => cp = false ∧ noCPL cs
def noCPL : TL → Prop
  | .nil => True
  | .cons t ts => noCP t ∧ noCPL ts
end

mutual
theorem size_pos : ∀ t : T, 1 ≤ size t
  | .node _ cs => by simp only [size]; omega
theorem lenL_le_sizeL : ∀ cs : TL, lenL cs ≤ sizeL cs
  | .nil => by simp [lenL, sizeL]
  | .cons t ts => by
    have := size_pos t; have := lenL_le_sizeL ts
    simp only [lenL, sizeL]; omega
end

-- The bound never under-reports the input's own height.
mutual
theorem height_le_bound : ∀ t : T, height t ≤ bound t
  | .node _ .nil => by simp [height, bound]
  | .node cp (.cons t ts) => by
    have := heightL_le_boundL (.cons t ts)
    simp only [height, bound]; split <;> omega
theorem heightL_le_boundL : ∀ cs : TL, heightL cs ≤ boundL cs
  | .nil => by simp [heightL, boundL]
  | .cons t ts => by
    have := height_le_bound t; have := heightL_le_boundL ts
    simp only [heightL, boundL]; omega
end

-- A plan without a `CartesianProduct` gets exactly its height
-- (`wide_plan_without_a_cartesian_product_reserves_nothing_extra`).
mutual
theorem bound_eq_height_noCP : ∀ t : T, noCP t → bound t = height t
  | .node _ .nil, _ => by simp [height, bound]
  | .node cp (.cons t ts), h => by
    have hc : cp = false := h.1
    have := boundL_eq_heightL_noCP (.cons t ts) h.2
    subst hc; simp only [height, bound]; simp; omega
theorem boundL_eq_heightL_noCP : ∀ cs : TL, noCPL cs → boundL cs = heightL cs
  | .nil, _ => by simp [heightL, boundL]
  | .cons t ts, h => by
    have := bound_eq_height_noCP t h.1; have := boundL_eq_heightL_noCP ts h.2
    simp only [heightL, boundL]; omega
end

-- Every node adds at most 4 levels to the bound: `bound ≤ 4 (len - 1)`.
mutual
theorem bound_le_size : ∀ t : T, bound t + 4 ≤ 4 * size t
  | .node _ .nil => by simp [bound, size, sizeL]
  | .node cp (.cons t ts) => by
    have := boundL_le_sizeL (.cons t ts) (by simp)
    simp only [bound, size, LEVELS_PER_CP_BRANCH]; split <;> omega
theorem boundL_le_sizeL : ∀ cs : TL, cs ≠ .nil → 3 * lenL cs + boundL cs + 1 ≤ 4 * sizeL cs
  | .nil, h => absurd rfl h
  | .cons t .nil, _ => by
    have := bound_le_size t
    simp only [lenL, boundL, sizeL]; omega
  | .cons t (.cons u us), _ => by
    have h1 := bound_le_size t
    have h2 := boundL_le_sizeL (.cons u us) (by simp)
    have h3 := size_pos t
    have h4 := lenL_le_sizeL (.cons u us)
    simp only [lenL, boundL, sizeL] at *; omega
end

-- A plan of `n` nodes is less than `n` tall.
mutual
theorem height_lt_size : ∀ t : T, height t + 1 ≤ size t
  | .node _ .nil => by simp [height, size, sizeL]
  | .node _ (.cons t ts) => by
    have := heightL_lt_sizeL (.cons t ts)
    simp only [height, size]; simp at this; omega
theorem heightL_lt_sizeL : ∀ cs : TL, heightL cs + 1 ≤ sizeL cs ∨ cs = .nil
  | .nil => .inr rfl
  | .cons t ts => by
    have h1 := height_lt_size t
    have h2 := heightL_lt_sizeL ts
    left; simp only [heightL, sizeL]
    cases h2 with
    | inl h => omega
    | inr h => subst h; simp only [heightL, sizeL]; omega
end

/-! ## Constants (release build; the instrumented build is `S = 32`, `PER = 64 KiB`) -/

def MAX_PLAN_DEPTH : Nat := 10000
def STACK_BASE : Nat := 1024 * 1024

/-- `stack_size_for`. -/
def stackSizeFor (per levels : Nat) : Nat := STACK_BASE + levels * per

theorem stackSizeFor_covers (per levels : Nat) : levels * per ≤ stackSizeFor per levels := by
  simp [stackSizeFor]

theorem stackSizeFor_mono (per : Nat) {a b : Nat} (h : a ≤ b) :
    stackSizeFor per a ≤ stackSizeFor per b := by
  simp only [stackSizeFor]; have := Nat.mul_le_mul_right per h; omega

/-- `with_plan_stack`: (`f`'s value, the stack segment reserved). -/
def withPlanStack {R : Type} (S per : Nat) (t : T) (f : Unit → R) : R × Option Nat :=
  if size t ≤ S then (f (), none)
  else if height t ≤ S then (f (), none)
  else (f (), some (stackSizeFor per (height t)))

/-- `with_optimizer_stack`: as `with_plan_stack`, sized from `bound`. -/
def withOptimizerStack {R : Type} (S per : Nat) (t : T) (f : Unit → R) : R × Option Nat :=
  if size t ≤ S then (f (), none)
  else if bound t ≤ S then (f (), none)
  else (f (), some (stackSizeFor per (bound t)))

/-- `Planner::check_depth`: `true` = `Ok(())`. -/
def checkDepth (S : Nat) (t : T) : Bool := !(decide (size t > S) && decide (bound t > MAX_PLAN_DEPTH))

/-- `Planner::check_height`: `true` = `Ok(())`. -/
def checkHeight (t : T) : Bool := !decide (height t > MAX_PLAN_DEPTH)

/-- Running on a grown stack is transparent: `query`/`get_plan` see `f`'s own result. -/
theorem withPlanStack_result {R : Type} (S per : Nat) (t : T) (f : Unit → R) :
    (withPlanStack S per t f).1 = f () := by
  simp only [withPlanStack]; split <;> (try split) <;> rfl

theorem withOptimizerStack_result {R : Type} (S per : Nat) (t : T) (f : Unit → R) :
    (withOptimizerStack S per t f).1 = f () := by
  simp only [withOptimizerStack]; split <;> (try split) <;> rfl

/-- A plan taller than `S` runs on a segment with `per` bytes for every level. -/
theorem withPlanStack_reserves {R : Type} (S per : Nat) (t : T) (f : Unit → R)
    (h : S < height t) :
    ∃ n, (withPlanStack S per t f).2 = some n ∧ height t * per ≤ n := by
  have : ¬ size t ≤ S := by
    have := height_lt_size t; omega
  refine ⟨stackSizeFor per (height t), ?_, stackSizeFor_covers _ _⟩
  simp [withPlanStack, this, Nat.not_le.mpr h]

/-- The `len()` shortcut of `with_plan_stack` is exact: a plan of at most `S` nodes
is at most `S` tall. -/
theorem withPlanStack_none_iff {R : Type} (S per : Nat) (t : T) (f : Unit → R) :
    (withPlanStack S per t f).2 = none ↔ height t ≤ S := by
  have := height_lt_size t
  simp only [withPlanStack]
  split
  · simp; omega
  · split <;> simp <;> omega

theorem withOptimizerStack_reserves {R : Type} (S per : Nat) (t : T) (f : Unit → R)
    (h1 : S < size t) (h2 : S < bound t) :
    ∃ n, (withOptimizerStack S per t f).2 = some n ∧ bound t * per ≤ n := by
  refine ⟨stackSizeFor per (bound t), ?_, stackSizeFor_covers _ _⟩
  simp [withOptimizerStack, Nat.not_le.mpr h1, Nat.not_le.mpr h2]

/-- `check_depth`'s `len()` shortcut is exact (for `S ≤ 2500`, true for both builds):
it accepts exactly the plans whose optimizer bound is at most `MAX_PLAN_DEPTH`. -/
theorem checkDepth_iff (S : Nat) (hS : S ≤ 2500) (t : T) :
    checkDepth S t = true ↔ bound t ≤ MAX_PLAN_DEPTH := by
  have := bound_le_size t
  simp only [checkDepth, MAX_PLAN_DEPTH]
  by_cases h1 : size t > S <;> by_cases h2 : bound t > 10000 <;> simp [h1, h2] <;> omega

theorem checkHeight_iff (t : T) : checkHeight t = true ↔ height t ≤ MAX_PLAN_DEPTH := by
  simp [checkHeight]

/-- A plan accepted by `check_depth` has at most `MAX_PLAN_DEPTH` levels, so the
optimizer's segment is capped at `stack_size_for(MAX_PLAN_DEPTH)`. -/
theorem checkDepth_caps_stack {R : Type} (S per : Nat) (hS : S ≤ 2500) (t : T) (f : Unit → R)
    (h : checkDepth S t = true) (n : Nat) (hn : (withOptimizerStack S per t f).2 = some n) :
    n ≤ stackSizeFor per MAX_PLAN_DEPTH := by
  have hb := (checkDepth_iff S hS t).1 h
  simp only [withOptimizerStack] at hn
  split at hn
  · cases hn
  · split at hn
    · cases hn
    · cases hn; exact stackSizeFor_mono per hb

/-! ## The Rust unit tests -/

def leaves (cp : Bool) : Nat → TL
  | 0 => .nil
  | n + 1 => .cons (.node cp .nil) (leaves cp n)

theorem lenL_leaves (cp : Bool) : ∀ n, lenL (leaves cp n) = n
  | 0 => rfl
  | n + 1 => by simp only [leaves, lenL, lenL_leaves cp n]; omega

theorem boundL_leaves (cp : Bool) : ∀ n, boundL (leaves cp n) = 0
  | 0 => by simp [leaves, boundL]
  | n + 1 => by simp [leaves, boundL, bound, boundL_leaves cp n]

theorem heightL_leaves (cp : Bool) : ∀ n, heightL (leaves cp n) = 0
  | 0 => by simp [leaves, heightL]
  | n + 1 => by simp [leaves, heightL, height, heightL_leaves cp n]

/-- `optimizer_stack_follows_cartesian_product_branches`: one `CartesianProduct`
over `n ≥ 1` leaves has bound `height + 3n`. -/
theorem wide_cp_bound (n : Nat) :
    bound (.node true (leaves false (n + 1))) = height (.node true (leaves false (n + 1))) + 3 * (n + 1) := by
  have h1 := lenL_leaves false (n + 1)
  have h2 := boundL_leaves false (n + 1)
  have h3 := heightL_leaves false (n + 1)
  simp only [leaves] at h1 h2 h3 ⊢
  simp only [bound, height, LEVELS_PER_CP_BRANCH, ite_true, h1, h2, h3]; omega

/-- `sibling_cartesian_products_are_not_summed`, the nested half: a 4-way
`CartesianProduct` under a 2-way one adds up (`2 + 3·2 + 3·4`). -/
theorem nested_cp_bound :
    bound (.node true (.cons (.node true (leaves false 4)) (.cons (.node false .nil) .nil))) =
      2 + 3 * 2 + 3 * 4 := by decide +kernel

/-- Lean-only observation: `with_optimizer_stack`'s `len() ≤ SHALLOW_PLAN` shortcut
can keep a plan whose bound exceeds `SHALLOW_PLAN` on the caller's stack (a
`CartesianProduct` over 44 leaves: 45 nodes, bound 133 > 128). The bound is at most
`4·(len-1)` (`bound_le_size`), so at most 508 levels are ever run this way. -/
theorem optimizer_shortcut_can_exceed_shallow :
    size (.node true (leaves false 44)) ≤ 128 ∧ 128 < bound (.node true (leaves false 44)) ∧
      (withOptimizerStack 128 (16 * 1024) (.node true (leaves false 44)) (fun _ => ())).2 = none := by
  decide +kernel

end PlannerBuild.PlanDepth
