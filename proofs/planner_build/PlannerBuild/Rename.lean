import PlannerBuild.ToPlan
import PlannerBuild.Nested
/-
# Odds and ends of planner/mod.rs

| here | there |
| --- | --- |
| `Proj`, `outerOf`, `renameProj` | `rename_projection_outputs` mod.rs:2646-2679 |
| `newPSt` | `Planner::new` mod.rs:814-824 |
| `inlineAttrs_passes` | `inline_attrs_to_filter` mod.rs:543-571 (meaning of the filter) |
-/
namespace PlannerBuild.E

/-! ## `rename_projection_outputs` -/

structure Proj where
  exprs : List (V × Ex)
  copies : List (V × V)

inductive Root
  | project (p : Proj)
  | other

def outerOf (remap : List (V × V)) (v : V) : Option V := (remap.find? (fun p => p.1 == v)).map Prod.snd

def renameProj (remap : List (V × V)) : Root → Option Proj
  | .project p =>
    if p.exprs.all (fun x => (outerOf remap x.1).isSome) && p.copies.all (fun x => (outerOf remap x.2).isSome)
    then some { exprs := p.exprs.map (fun x => ((outerOf remap x.1).getD x.1, x.2)),
                copies := p.copies.map (fun x => (x.1, (outerOf remap x.2).getD x.2)) }
    else none
  | .other => none

/-- `false` exactly when the root is not a `Project` or some output is not remapped. -/
theorem renameProj_none (remap : List (V × V)) (r : Root) :
    renameProj remap r = none ↔
      ∀ p, r = .project p → (∃ x ∈ p.exprs, outerOf remap x.1 = none) ∨ (∃ x ∈ p.copies, outerOf remap x.2 = none) := by
  cases r with
  | other => simp [renameProj]
  | project p =>
    simp only [renameProj, Root.project.injEq, forall_eq']
    split
    · rename_i h
      simp only [Bool.and_eq_true, List.all_eq_true, Option.isSome_iff_ne_none, ne_eq] at h
      simp only [reduceCtorEq, false_iff, not_or, not_exists, not_and]
      exact ⟨fun x hx => h.1 x hx, fun x hx => h.2 x hx⟩
    · rename_i h
      simp only [true_iff]
      apply Classical.byContradiction; intro hc
      simp only [not_or, not_exists, not_and] at hc
      apply h
      simp only [Bool.and_eq_true, List.all_eq_true]
      exact ⟨fun x hx => Option.isSome_iff_ne_none.2 (hc.1 x hx),
        fun x hx => Option.isSome_iff_ne_none.2 (hc.2 x hx)⟩

/-- When it succeeds, only the output variables change, each to its outer name;
expressions and copy sources are untouched. -/
theorem renameProj_some (remap : List (V × V)) (p q : Proj) (h : renameProj remap (.project p) = some q) :
    q.exprs.map Prod.snd = p.exprs.map Prod.snd ∧ q.copies.map Prod.fst = p.copies.map Prod.fst ∧
    q.exprs.map Prod.fst = p.exprs.map (fun x => (outerOf remap x.1).getD x.1) ∧
    (∀ x ∈ p.exprs, (outerOf remap x.1).isSome) ∧ (∀ x ∈ p.copies, (outerOf remap x.2).isSome) := by
  simp only [renameProj] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, List.all_eq_true] at hc
    cases h
    refine ⟨by simp, by simp, by simp, hc.1, hc.2⟩
  · cases h

/-- Renaming the outputs in place = projecting, then renaming the columns (the
remapping `Project` it saves). -/
def projRow {Val : Type} (ev : Ex → (V → Val) → Val) (p : Proj) (ρ : V → Val) : List (V × Val) :=
  p.exprs.map (fun x => (x.1, ev x.2 ρ)) ++ p.copies.map (fun x => (x.2, ρ x.1))

theorem renameProj_row {Val : Type} (ev : Ex → (V → Val) → Val) (remap : List (V × V)) (p q : Proj)
    (h : renameProj remap (.project p) = some q) (ρ : V → Val) :
    projRow ev q ρ = (projRow ev p ρ).map (fun x => ((outerOf remap x.1).getD x.1, x.2)) := by
  simp only [renameProj] at h
  split at h
  · cases h
    simp only [projRow, List.map_append, List.map_map]
    congr 1 <;> apply List.map_congr_left <;> intro x _ <;> rfl
  · cases h

/-! ## `Planner::new` -/

def newPSt (lens : Nat → Nat) : PSt := ⟨lens, [], [], [], 0, []⟩

/-- A new planner has nothing bound, no nested plans, and mints its first
variable of scope `s` at the binder's reported length of `s`. -/
theorem newPSt_spec (lens : Nat → Nat) (s : Nat) :
    (newPSt lens).visited = [] ∧ (newPSt lens).nestedN = 0 ∧ (fresh (newPSt lens) s).1 = ⟨lens s, s⟩ :=
  ⟨rfl, rfl, rfl⟩

/-! ## What an inline-attribute filter means -/

variable {R : Type} (atom : Ex → R → Option Bool) (sub : QG → R → List R)

/-- `(n {k1: v1, k2: v2})` passes a row iff every `n.ki = vi` is true there. -/
theorem inlineAttrs_passes (ι : List (Nat × QG)) (alias : V) (attrs : List (String × Ex)) (f : Ex)
    (h : inlineAttrsToFilter alias attrs = some f) (y : R) :
    passes (tv atom sub ι f y) = attrs.all (fun kv => passes (atom (attrEq alias kv) y)) := by
  unfold inlineAttrsToFilter at h
  have heq : ∀ kv, tv atom sub ι (attrEq alias kv) y = atom (attrEq alias kv) y := by
    intro kv; simp only [attrEq, tv]
  match attrs, h with
  | [a], h => cases h; simp [heq]
  | a :: b :: rest, h =>
    cases h
    simp only [tv, passes_and3, tvL_map, List.all_map, List.map_cons, List.all_cons, heq]
    congr 2

end PlannerBuild.E
