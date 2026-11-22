/-
# Plan-depth caps and stack sizing (#3305)

`planner/mod.rs`: `stack_size_for`, `with_plan_stack`, `optimizer_depth_bound`,
`with_optimizer_stack`, `Planner::check_depth`, `Planner::check_height`.

| here | there |
| --- | --- |
| `PT` | `DynTree<IR>`, with only "is it a `CartesianProduct`" kept per node |
| `PT.size` | `plan.len()` (node count) |
| `PT.height` | orx-tree `height()` (edges on the longest root-to-leaf path; a leaf is 0) |
| `PT.bound` | `optimizer_depth_bound` (recursive form of its pre-order / reverse fold) |
| `stackSizeFor` | `stack_size_for` (saturating `usize` arithmetic) |
| `planStack`, `optStack` | the stack `with_plan_stack` / `with_optimizer_stack` reserve: `none` runs `f` on the caller's stack |
| `checkDepth`, `checkHeight` | `Planner::check_depth`, `Planner::check_height` |

`instr` is `INSTRUMENTED_FRAMES` (`cfg!(debug_assertions)`). `stacker::grow(n, f)` (a
third-party crate) runs `f` on a fresh stack of `n` bytes and returns its result, so the
result of both wrappers is `f ()` either way (`withPlanStack_eq`, `withOptimizerStack_eq`);
what differs is the stack reserved.
-/
namespace PlannerBuild.PlanStack

inductive PT where
  | node (cp : Bool) (cs : List PT)

mutual
def PT.size : PT → Nat
  | .node _ cs => 1 + PT.sizeL cs
def PT.sizeL : List PT → Nat
  | [] => 0
  | c :: cs => PT.size c + PT.sizeL cs
end

mutual
/-- Edges on the longest root-to-leaf path. -/
def PT.height : PT → Nat
  | .node _ [] => 0
  | .node _ (c :: cs) => 1 + PT.heightL (c :: cs)
def PT.heightL : List PT → Nat
  | [] => 0
  | c :: cs => max (PT.height c) (PT.heightL cs)
end

/-- `LEVELS_PER_CP_BRANCH`. -/
def levelsPerCpBranch : Nat := 3

mutual
/-- `optimizer_depth_bound`: `extra + (if leaf then 0 else 1 + below)`, `extra` being
`LEVELS_PER_CP_BRANCH * n` on a `CartesianProduct` of `n` children, `below` the worst
child. -/
def PT.bound : PT → Nat
  | .node _ [] => 0
  | .node cp (c :: cs) =>
    (if cp then levelsPerCpBranch * (c :: cs).length else 0) + (1 + PT.boundL (c :: cs))
def PT.boundL : List PT → Nat
  | [] => 0
  | c :: cs => max (PT.bound c) (PT.boundL cs)
end

/-- `MAX_PLAN_DEPTH`. -/
def maxPlanDepth : Nat := 10000
/-- `SHALLOW_PLAN`. -/
def shallowPlan (instr : Bool) : Nat := if instr then 32 else 128
/-- `STACK_PER_LEVEL`. -/
def stackPerLevel (instr : Bool) : Nat := if instr then 64 * 1024 else 16 * 1024
/-- `STACK_BASE`. -/
def stackBase : Nat := 1024 * 1024
/-- `usize::MAX` (64-bit). -/
def usizeMax : Nat := 2 ^ 64 - 1

/-- `saturating_add` / `saturating_mul` on `usize`. -/
def satAdd (a b : Nat) : Nat := min (a + b) usizeMax
def satMul (a b : Nat) : Nat := min (a * b) usizeMax

/-- `stack_size_for`. -/
def stackSizeFor (instr : Bool) (levels : Nat) : Nat :=
  satAdd stackBase (satMul levels (stackPerLevel instr))

/-- `with_plan_stack`: the stack it reserves (`none`: `f` runs right here). -/
def planStack (instr : Bool) (t : PT) : Option Nat :=
  if t.size ≤ shallowPlan instr then none
  else if t.height ≤ shallowPlan instr then none
  else some (stackSizeFor instr t.height)

/-- `with_optimizer_stack`: the stack it reserves. -/
def optStack (instr : Bool) (t : PT) : Option Nat :=
  if t.size ≤ shallowPlan instr then none
  else if t.bound ≤ shallowPlan instr then none
  else some (stackSizeFor instr t.bound)

/-- `stacker::grow(n, f)`: runs `f` on a fresh `n`-byte stack and returns its value. -/
def grow {R : Type} (_n : Nat) (f : Unit → R) : R := f ()

def runOn {R : Type} (s : Option Nat) (f : Unit → R) : R :=
  match s with
  | none => f ()
  | some n => grow n f

def withPlanStack {R : Type} (instr : Bool) (t : PT) (f : Unit → R) : R := runOn (planStack instr t) f
def withOptimizerStack {R : Type} (instr : Bool) (t : PT) (f : Unit → R) : R := runOn (optStack instr t) f

/-- `TOO_DEEP` followed by the cap. -/
def tooDeep : String := "Query nesting exceeds the maximum depth of " ++ toString maxPlanDepth

/-- `Planner::check_depth`. -/
def checkDepth (instr : Bool) (t : PT) : Except String Unit :=
  if t.size > shallowPlan instr && t.bound > maxPlanDepth then .error tooDeep else .ok ()

/-- `Planner::check_height`. -/
def checkHeight (t : PT) : Except String Unit :=
  if t.height > maxPlanDepth then .error tooDeep else .ok ()

/-! ## The wrappers are transparent -/

theorem withPlanStack_eq {R : Type} (instr : Bool) (t : PT) (f : Unit → R) :
    withPlanStack instr t f = f () := by
  unfold withPlanStack runOn grow; split <;> rfl

theorem withOptimizerStack_eq {R : Type} (instr : Bool) (t : PT) (f : Unit → R) :
    withOptimizerStack instr t f = f () := by
  unfold withOptimizerStack runOn grow; split <;> rfl

/-! ## `stack_size_for` -/

/-- No saturation up to `MAX_PLAN_DEPTH` levels: the base plus the per-level share. -/
theorem stackSizeFor_exact (instr : Bool) (levels : Nat) (h : levels ≤ maxPlanDepth) :
    stackSizeFor instr levels = stackBase + levels * stackPerLevel instr := by
  unfold stackSizeFor satAdd satMul usizeMax stackBase stackPerLevel
  unfold maxPlanDepth at h
  cases instr <;> simp <;> omega

theorem stackSizeFor_mono (instr : Bool) {a b : Nat} (h : a ≤ b) :
    stackSizeFor instr a ≤ stackSizeFor instr b := by
  unfold stackSizeFor satAdd satMul
  have := Nat.mul_le_mul_right (stackPerLevel instr) h
  omega

/-! ## `optimizer_depth_bound` -/

theorem PT.size_pos : ∀ t : PT, 1 ≤ t.size
  | .node _ cs => by simp [PT.size]

theorem PT.sizeL_ge_length : ∀ l : List PT, l.length ≤ PT.sizeL l
  | [] => by simp [PT.sizeL]
  | c :: cs => by
    simp only [PT.sizeL, List.length_cons]
    have := PT.size_pos c; have := PT.sizeL_ge_length cs; omega

mutual
/-- The bound never falls below the plan's own height. -/
theorem PT.height_le_bound : ∀ t : PT, t.height ≤ t.bound
  | .node _ [] => by simp [PT.height]
  | .node cp (c :: cs) => by
    have := PT.heightL_le_boundL (c :: cs)
    simp only [PT.height, PT.bound]; omega
theorem PT.heightL_le_boundL : ∀ l : List PT, PT.heightL l ≤ PT.boundL l
  | [] => by simp [PT.heightL, PT.boundL]
  | c :: cs => by
    have := PT.height_le_bound c; have := PT.heightL_le_boundL cs
    simp only [PT.heightL, PT.boundL]; omega
end

mutual
def PT.noCP : PT → Prop
  | .node cp cs => cp = false ∧ PT.noCPL cs
def PT.noCPL : List PT → Prop
  | [] => True
  | c :: cs => PT.noCP c ∧ PT.noCPL cs
end

mutual
/-- A plan with no `CartesianProduct` gets no allowance: the bound is its height. -/
theorem PT.bound_noCP : ∀ t : PT, t.noCP → t.bound = t.height
  | .node _ [], _ => by simp [PT.bound, PT.height]
  | .node cp (c :: cs), h => by
    have hl := PT.boundL_noCP (c :: cs) h.2
    have hcp := h.1
    subst hcp
    simp only [PT.bound, PT.height]; simp [hl]
theorem PT.boundL_noCP : ∀ l : List PT, PT.noCPL l → PT.boundL l = PT.heightL l
  | [], _ => by simp [PT.boundL, PT.heightL]
  | c :: cs, h => by
    simp only [PT.boundL, PT.heightL, PT.bound_noCP c h.1, PT.boundL_noCP cs h.2]
end

mutual
/-- Each node adds at most 4 to the bound (1 level, 3 per CP branch). -/
theorem PT.bound_le_size : ∀ t : PT, t.bound + 4 ≤ 4 * t.size
  | .node cp [] => by simp [PT.bound, PT.size, PT.sizeL]
  | .node cp (c :: cs) => by
    have := PT.boundL_le_sizeL (c :: cs)
    simp only [PT.bound, PT.size, levelsPerCpBranch] at *
    simp only [List.length_cons] at *
    split <;> omega
theorem PT.boundL_le_sizeL : ∀ l : List PT, PT.boundL l + 4 * l.length ≤ 4 * PT.sizeL l
  | [] => by simp [PT.boundL, PT.sizeL]
  | c :: cs => by
    have h1 := PT.bound_le_size c; have h2 := PT.boundL_le_sizeL cs
    have h3 := PT.sizeL_ge_length cs
    simp only [PT.boundL, PT.sizeL, List.length_cons]
    omega
end

/-- A `CartesianProduct` of `n ≥ 1` leaves: `1 + 3n` (the unit test's `height + 3 * n`). -/
theorem bound_cp_leaves (n : Nat) (h : 0 < n) :
    (PT.node true (List.replicate n (.node false []))).bound = 1 + 3 * n := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have hz : ∀ k, PT.boundL (List.replicate k (.node false [])) = 0 := by
    intro k; induction k with
    | zero => rfl
    | succ k ih => simp [List.replicate, PT.boundL, PT.bound, ih]
  simp only [List.replicate, PT.bound, levelsPerCpBranch, ite_true, List.length_cons,
    List.length_replicate]
  rw [← List.replicate_succ, hz]; omega

/-! ## `Planner::check_depth` / `check_height` -/

/-- **PROVEN**: the O(1) `len()` shortcut is sound. `check_depth` accepts exactly the plans
whose optimizer bound is within `MAX_PLAN_DEPTH`: a plan short enough to skip the walk
(at most 128 nodes) has a bound of at most 508. -/
theorem checkDepth_ok_iff (instr : Bool) (t : PT) :
    checkDepth instr t = .ok () ↔ t.bound ≤ maxPlanDepth := by
  have hb := PT.bound_le_size t
  unfold checkDepth
  by_cases h1 : t.size > shallowPlan instr <;> by_cases h2 : t.bound > maxPlanDepth <;>
    simp [h1, h2] <;> first | omega | skip
  all_goals (unfold shallowPlan maxPlanDepth at *; cases instr <;> simp at * <;> omega)

theorem checkDepth_error (instr : Bool) (t : PT) (e : String) :
    checkDepth instr t = .error e → e = tooDeep := by
  unfold checkDepth; split <;> simp_all

theorem checkHeight_ok_iff (t : PT) : checkHeight t = .ok () ↔ t.height ≤ maxPlanDepth := by
  unfold checkHeight; by_cases h : t.height > maxPlanDepth <;> simp [h] <;> omega

/-- A plan `check_depth` accepts is itself no taller than the cap. -/
theorem checkDepth_ok_height (instr : Bool) (t : PT) (h : checkDepth instr t = .ok ()) :
    t.height ≤ maxPlanDepth :=
  Nat.le_trans (PT.height_le_bound t) ((checkDepth_ok_iff instr t).1 h)

/-! ## The stack reserved -/

/-- `with_plan_stack` reserves a stack only for a plan taller than `SHALLOW_PLAN`, and for a
plan `check_height` accepts, exactly the base plus the per-level share for every level. -/
theorem planStack_spec (instr : Bool) (t : PT) (n : Nat) (h : planStack instr t = some n)
    (hc : checkHeight t = .ok ()) :
    shallowPlan instr < t.height ∧ n = stackBase + t.height * stackPerLevel instr := by
  unfold planStack at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · next h2 =>
      simp only [Option.some.injEq] at h; subst h
      exact ⟨by omega, stackSizeFor_exact instr _ ((checkHeight_ok_iff t).1 hc)⟩

/-- `with_optimizer_stack` reserves a stack only when the optimizer bound exceeds
`SHALLOW_PLAN`, and for a plan `check_depth` accepts, the base plus the per-level share for
every level the optimizer can build, so at least what the plan's own height needs. -/
theorem optStack_spec (instr : Bool) (t : PT) (n : Nat) (h : optStack instr t = some n)
    (hc : checkDepth instr t = .ok ()) :
    shallowPlan instr < t.bound ∧ n = stackBase + t.bound * stackPerLevel instr ∧
      stackBase + t.height * stackPerLevel instr ≤ n := by
  unfold optStack at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · next h2 =>
      simp only [Option.some.injEq] at h; subst h
      have he := stackSizeFor_exact instr _ ((checkDepth_ok_iff instr t).1 hc)
      have := Nat.mul_le_mul_right (stackPerLevel instr) (PT.height_le_bound t)
      exact ⟨by omega, he, by omega⟩

end PlannerBuild.PlanStack
