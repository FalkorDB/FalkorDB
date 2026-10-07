import PlannerBuild.Filter
/-
# `plan_filter` (mod.rs:2597-2635)

`planFilter` follows the Rust after `extract_filter_comprehensions` has
returned the predicate unchanged (no pattern comprehension, no pattern outside
an AND/OR/NOT/paren chain — `needsExtraction e .semiApply = false`, the early
return at mod.rs:1384): rebuild, then `expr_to_plan` if the inline map is
non-empty, else a `Filter` unless the rebuilt root is the literal `true`, then
one SemiApply/AntiSemiApply per extracted pattern, in extraction order.
-/
namespace PlannerBuild.E

variable {R : Type} (atom : Ex → R → Option Bool) (sub : QG → R → List R)

def foldExt (res : FP) : List (QG × Bool) → FP
  | [] => res
  | (g, anti) :: es => foldExt (if anti then .anti res (.patSub g) else .semi res (.patSub g)) es

def st0 (lens : Nat → Nat) : CSt := ⟨lens, [], []⟩

def planFilter (lens : Nat → Nat) (e : Ex) (inp : FP) : FP × CSt :=
  let x := collect (st0 lens) true e
  let res :=
    if x.2.inl.isEmpty = false then toPlan x.2.inl x.1 inp
    else if isTT x.1 then inp else .filter x.1 inp
  (foldExt res x.2.ext, x.2)

theorem foldExt_run (ι : List (Nat × QG)) (up : R → List R) :
    ∀ (es : List (QG × Bool)) (res : FP) (r : R),
    run atom sub ι up (foldExt res es) r =
      (run atom sub ι up res r).filter (fun y => es.all (sat sub y))
  | [], res, r => by
    simp only [foldExt, List.all_nil]
    exact (List.filter_eq_self.2 (fun _ _ => rfl)).symm
  | (g, anti) :: es, res, r => by
    simp only [foldExt]
    rw [foldExt_run ι up es _ r]
    cases anti <;> simp only [Bool.false_eq_true, ↓reduceIte, run, List.filter_filter, List.all_cons]
      <;> congr 1 <;> funext y <;> simp [sat, Bool.and_comm]

theorem allSat_iff (ext : List (QG × Bool)) (y : R) : AllSat sub ext y ↔ ext.all (sat sub y) = true := by
  simp [AllSat]

/-- Shared hypotheses: the predicate as `extract_filter_comprehensions` leaves
it, patterns in one scope `s0`, mentioned ids below that scope's length. -/
structure Pre (lens : Nat → Nat) (s0 : Nat) (e : Ex) : Prop where
  shape : shapeOK e = true
  noExt : needsExtraction e .semiApply = false
  scope : SameScope s0 e
  below : ∀ i ∈ varIds e, i < lens s0

theorem pre_inv (lens : Nat → Nat) (s0 : Nat) (e : Ex) (h : Pre lens s0 e) :
    Inv s0 (lens s0) (collect (st0 lens) true e).2 :=
  collect_inv s0 (lens s0) _ true e h.scope ⟨by simp [st0], by simp [keys, st0], Nat.le_refl _⟩

theorem filter_and_iff {α : Type} (l : List α) (p q r : α → Bool) (h : ∀ a, (p a && q a) = r a) :
    (l.filter p).filter q = l.filter r := by
  rw [List.filter_filter]; congr 1; funext a; rw [← h a, Bool.and_comm]

/-- Common core: if the middle plan keeps exactly the rows where the rebuilt
predicate is true, the whole `plan_filter` plan is exact. -/
theorem planFilter_of_res (lens : Nat → Nat) (s0 : Nat) (e : Ex) (inp : FP) (up : R → List R) (r : R)
    (h : Pre lens s0 e) (hTT : ∀ c y, isTT c = true → atom c y = some true)
    (res : FP) (hres : run atom sub (collect (st0 lens) true e).2.inl up res r =
      (run atom sub (collect (st0 lens) true e).2.inl up inp r).filter (fun y =>
        passes (tv atom sub (collect (st0 lens) true e).2.inl (collect (st0 lens) true e).1 y))) :
    run atom sub (collect (st0 lens) true e).2.inl up (foldExt res (collect (st0 lens) true e).2.ext) r =
      (run atom sub (collect (st0 lens) true e).2.inl up inp r).filter (fun y => passes (tv atom sub [] e y)) := by
  have hinv := pre_inv lens s0 e h
  have hg := good_final s0 _ _ hinv
  have hf := fresh_final s0 _ _ hinv (varIds e) h.below
  rw [foldExt_run, hres, filter_and_iff]
  intro y
  have key := collect_passes atom sub _ y (hTT tt y rfl) (st0 lens) e h.shape h.noExt hf hg
  have h0 : AllSat sub (st0 lens).ext y := by simp [AllSat, st0]
  rw [Bool.eq_iff_iff]
  simp only [Bool.and_eq_true, passes, beq_iff_eq, ← allSat_iff]
  constructor
  · rintro ⟨a, b⟩; exact (key.2 ⟨a, b⟩).1
  · intro a; have := key.1 ⟨a, h0⟩; exact this

/-- No inline pattern (every pattern was a top-level conjunct, or there were
none): `plan_filter` is exact in three-valued logic. -/
theorem planFilter_scalar_correct (lens : Nat → Nat) (s0 : Nat) (e : Ex) (inp : FP) (up : R → List R)
    (r : R) (h : Pre lens s0 e) (hTT : ∀ c y, isTT c = true → atom c y = some true)
    (hinl : (collect (st0 lens) true e).2.inl = []) :
    run atom sub (planFilter lens e inp).2.inl up (planFilter lens e inp).1 r =
      (run atom sub (planFilter lens e inp).2.inl up inp r).filter (fun y => passes (tv atom sub [] e y)) := by
  have hx := planFilter_of_res atom sub lens s0 e inp up r h hTT
    (if isTT (collect (st0 lens) true e).1 then inp else .filter (collect (st0 lens) true e).1 inp) (by
      split
      · rename_i htt
        refine (List.filter_eq_self.2 (fun y _ => ?_)).symm
        rw [tv_isTT atom sub _ _ y htt, hTT _ y htt]; rfl
      · rfl)
  simp only [planFilter, hinl, List.isEmpty_nil, Bool.true_eq_false, ↓reduceIte]
  rw [hinl] at hx
  exact hx

/-- With two-valued atoms, `plan_filter` is exact whatever the shape. -/
theorem planFilter_correct (lens : Nat → Nat) (s0 : Nat) (e : Ex) (inp : FP) (up : R → List R)
    (r : R) (h : Pre lens s0 e) (hTT : ∀ c y, isTT c = true → atom c y = some true)
    (H2 : ∀ e y, atom e y ≠ none) :
    run atom sub (planFilter lens e inp).2.inl up (planFilter lens e inp).1 r =
      (run atom sub (planFilter lens e inp).2.inl up inp r).filter (fun y => passes (tv atom sub [] e y)) := by
  cases hinl : (collect (st0 lens) true e).2.inl.isEmpty
  · have hinv := pre_inv lens s0 e h
    have hg := good_final s0 _ _ hinv
    have hf := fresh_final s0 _ _ hinv (varIds e) h.below
    have hw := collect_wf _ (st0 lens) true e h.shape h.noExt hf hg
    have hx := planFilter_of_res atom sub lens s0 e inp up r h hTT
      (toPlan (collect (st0 lens) true e).2.inl (collect (st0 lens) true e).1 inp)
      (toPlan_correct atom sub _ H2 hTT up _ inp r hw.2 hw.1)
    simp only [planFilter, hinl, ↓reduceIte]
    exact hx
  · have : (collect (st0 lens) true e).2.inl = [] := List.isEmpty_iff.1 hinl
    exact planFilter_scalar_correct atom sub lens s0 e inp up r h hTT this

end PlannerBuild.E
