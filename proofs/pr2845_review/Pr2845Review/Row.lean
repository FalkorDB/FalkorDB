import Pr2845Review.Binder
/-
# PR #2845, runtime side: the entry projection is a record boundary

A runtime row is a dense array indexed by `Variable.id` alone
(runtime/row.rs; `scope_id` never reaches the row). Modelled as a function
`Nat → Option V` (`none` = unbound slot).

* `Argument` (runtime/ops/argument.rs) hands the sub-plan the outer row as is.
* `ProjectOp::eval` (runtime/ops/project.rs:54-147) builds a fresh row from
  its projections only (`Row::with_capacity`, `result.insert(target, …)`,
  then `BatchBuilder::push_row`) and drops its input.
-/
namespace Pr2845

abbrev RowF (V : Type) := Nat → Option V

/-- The fresh row an entry projection `(inner, outer)` pairs builds. -/
def projRow {V : Type} (ps : List (Var × Var)) (r : RowF V) : RowF V := fun i =>
  match ps.find? (fun p => p.1.id == i) with
  | some p => r p.2.id
  | none => none

/-- What the first operator of the body sees: origin/main had no projection
when nothing was imported (the `Argument` row itself), the merged code always
has one. -/
def bodyInputOld {V : Type} (ps : List (Var × Var)) (imported : Bool) (r : RowF V) : RowF V :=
  if imported then projRow ps r else r

def bodyInputNew {V : Type} (ps : List (Var × Var)) (r : RowF V) : RowF V := projRow ps r

theorem projRow_unbound {V : Type} (ps : List (Var × Var)) (r : RowF V) (i : Nat)
    (h : i ∉ ps.map (·.1.id)) : projRow ps r i = none := by
  unfold projRow
  have : ps.find? (fun p => p.1.id == i) = none := by
    rw [List.find?_eq_none]
    intro p hp heq
    simp only [beq_iff_eq] at heq
    exact h (heq ▸ List.mem_map.mpr ⟨p, hp, rfl⟩)
  rw [this]

/-- An imported slot carries exactly the outer value (targets are distinct:
`importProj_spec` mints them `0..k-1`). -/
theorem projRow_import {V : Type} (ps : List (Var × Var)) (r : RowF V) (p : Var × Var)
    (hp : p ∈ ps) (hd : (ps.map (·.1.id)).Nodup) : projRow ps r p.1.id = r p.2.id := by
  unfold projRow
  induction ps with
  | nil => simp at hp
  | cons q qs ih =>
    simp only [List.find?_cons]
    by_cases hq : q.1.id = p.1.id
    · have : q = p := by
        rcases List.mem_cons.mp hp with rfl | hmem
        · rfl
        · exfalso
          have hn := (List.nodup_cons.mp hd).1
          apply hn
          show q.1.id ∈ _
          rw [hq]
          exact List.mem_map.mpr ⟨p, hmem, rfl⟩
      subst this; simp
    · have hmem : p ∈ qs := by
        rcases List.mem_cons.mp hp with rfl | hmem
        · exact absurd rfl hq
        · exact hmem
      have hb : (q.1.id == p.1.id) = false := by simpa using hq
      rw [hb]
      exact ih hmem (List.nodup_cons.mp hd).2

/-- **Claim 2 (no slot aliasing), plain bodies.** For any outer row, every
variable the body mints itself (id `≥ k`, `entry_ids_below`) is unbound
when the body starts — so a body scan endpoint never looks already bound
(#2601) and a body predicate never reads an outer value (#2602). -/
theorem body_own_unbound {V : Type} (s : Nat) (imp : List (Name × Var)) (hd : Distinct imp)
    (r : RowF V) (j : Nat) (hj : imp.length ≤ j) :
    bodyInputNew (importProj [] s imp).2 r j = none := by
  apply projRow_unbound
  obtain ⟨-, -, h3, -, -⟩ := importProj_spec s imp [] hd (by simp)
  rw [h3]; simp; omega

/-- … and every imported variable reads its outer value. -/
theorem body_import_value {V : Type} (s : Nat) (imp : List (Name × Var)) (hd : Distinct imp)
    (r : RowF V) (p : Var × Var) (hp : p ∈ (importProj [] s imp).2) :
    bodyInputNew (importProj [] s imp).2 r p.1.id = r p.2.id := by
  apply projRow_import _ _ _ hp
  obtain ⟨-, -, h3, -, -⟩ := importProj_spec s imp [] hd (by simp)
  rw [h3]; simp only [List.length_nil, Nat.zero_add, List.map_id']
  exact List.nodup_range

/-- origin/main, import-less body: the body's first variable (id 0) reads
the outer row's slot 0 — the `x` of `MATCH (x:A:B) CALL { MATCH (n:A:B
{id:1}) … }`. -/
theorem pre2845_old_aliases : ∃ r : RowF Nat, bodyInputOld [] false r 0 = some 7 ∧
    bodyInputNew [] r 0 = none :=
  ⟨fun _ => some 7, rfl, rfl⟩

/-- Runtime choice of a scan endpoint (#2601): an already-bound slot turns a
full scan into an expand from that slot. -/
def scanMode {V : Type} (r : RowF V) (endpoint : Nat) : Bool := (r endpoint).isSome

theorem new_scan_scans {V : Type} (s : Nat) (imp : List (Name × Var)) (hd : Distinct imp)
    (r : RowF V) (j : Nat) (hj : imp.length ≤ j) :
    scanMode (bodyInputNew (importProj [] s imp).2 r) j = false := by
  simp [scanMode, body_own_unbound s imp hd r j hj]

end Pr2845
