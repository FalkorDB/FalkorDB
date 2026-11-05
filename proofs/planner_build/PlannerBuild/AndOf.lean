import PlannerBuild.Decomp
/-
# `ExprIR::and_of` / `or_of` keep the three-valued meaning (ast.rs:342-380, #3335)

`flat` splices an operand of the same connective into its parent. Kleene
AND/OR are associative, so the spliced tree has the same truth value on every
row (`tv_andOf`, `tv_orOf`): the AND of the operands' values. The identity
constant for no operands needs the atom hypothesis that `true` / `false` mean
themselves. The error-propagating runtime loop is treated in
`proofs/optimizer_rewrites` (`AndOf.flat_eval`).
-/
namespace PlannerBuild.E

theorem and3_eq (l : List (Option Bool)) :
    and3 l = if some false ∈ l then some false else if none ∈ l then none else some true := by
  induction l with
  | nil => simp [and3]
  | cons x xs ih =>
    simp only [and3, ih, List.mem_cons]
    rcases x with _ | _ | _ <;> by_cases h1 : some false ∈ xs <;> by_cases h2 : none ∈ xs <;> simp [h1, h2]

theorem or3_eq (l : List (Option Bool)) :
    or3 l = if some true ∈ l then some true else if none ∈ l then none else some false := by
  induction l with
  | nil => simp [or3]
  | cons x xs ih =>
    simp only [or3, ih, List.mem_cons]
    rcases x with _ | _ | _ <;> by_cases h1 : some true ∈ xs <;> by_cases h2 : none ∈ xs <;> simp [h1, h2]

theorem and3_append (l1 l2 : List (Option Bool)) : and3 (l1 ++ l2) = and3 (and3 l1 :: l2) := by
  rw [and3_eq (l1 ++ l2), and3_eq (and3 l1 :: l2), and3_eq l1]
  by_cases a : some false ∈ l1 <;> by_cases b : none ∈ l1 <;>
    by_cases c : some false ∈ l2 <;> by_cases d : none ∈ l2 <;> simp [a, b, c, d]

theorem or3_append (l1 l2 : List (Option Bool)) : or3 (l1 ++ l2) = or3 (or3 l1 :: l2) := by
  rw [or3_eq (l1 ++ l2), or3_eq (or3 l1 :: l2), or3_eq l1]
  by_cases a : some true ∈ l1 <;> by_cases b : none ∈ l1 <;>
    by_cases c : some true ∈ l2 <;> by_cases d : none ∈ l2 <;> simp [a, b, c, d]

theorem and3_cons_congr (x : Option Bool) (l1 l2 : List (Option Bool)) (h : and3 l1 = and3 l2) :
    and3 (x :: l1) = and3 (x :: l2) := by simp only [and3, h]

theorem or3_cons_congr (x : Option Bool) (l1 l2 : List (Option Bool)) (h : or3 l1 = or3 l2) :
    or3 (x :: l1) = or3 (x :: l2) := by simp only [or3, h]

theorem and3_single (x : Option Bool) : and3 [x] = x := by
  rcases x with _ | _ | _ <;> rfl

theorem or3_single (x : Option Bool) : or3 [x] = x := by
  rcases x with _ | _ | _ <;> rfl

variable {R : Type} (atom : Ex → R → Option Bool) (sub : QG → R → List R) (ι : List (Nat × QG))

/-- Splicing an AND operand into an AND does not change the AND's value. -/
theorem and3_flat (xs : List Ex) (y : R) :
    and3 ((xs.flatMap (spliceOp .and)).map (tv atom sub ι · y)) = and3 (xs.map (tv atom sub ι · y)) := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.flatMap_cons, List.map_append, and3_append, List.map_cons]
    have hx : and3 ((spliceOp .and x).map (tv atom sub ι · y)) = tv atom sub ι x y := by
      obtain ⟨d, cs⟩ := x
      by_cases hd : d = .and
      · subst hd; simp only [spliceOp_same, tv, tvL_map]
      · rw [spliceOp_other .and (.node d cs) hd, List.map_cons, List.map_nil, and3_single]
    rw [hx]; exact and3_cons_congr _ _ _ ih

theorem or3_flat (xs : List Ex) (y : R) :
    or3 ((xs.flatMap (spliceOp .or)).map (tv atom sub ι · y)) = or3 (xs.map (tv atom sub ι · y)) := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.flatMap_cons, List.map_append, or3_append, List.map_cons]
    have hx : or3 ((spliceOp .or x).map (tv atom sub ι · y)) = tv atom sub ι x y := by
      obtain ⟨d, cs⟩ := x
      by_cases hd : d = .or
      · subst hd; simp only [spliceOp_same, tv, tvL_map]
      · rw [spliceOp_other .or (.node d cs) hd, List.map_cons, List.map_nil, or3_single]
    rw [hx]; exact or3_cons_congr _ _ _ ih

/-- **`and_of` is the AND of its operands** (three-valued, every row): flattening,
the lone-operand return and the `true` identity all keep the value. -/
theorem tv_andOf (y : R) (hT : atom tt y = some true) (xs : List Ex) :
    tv atom sub ι (andOf xs) y = and3 (xs.map (tv atom sub ι · y)) := by
  rw [← and3_flat atom sub ι xs y, andOf, flat_eq_body]
  unfold flatBody
  generalize xs.flatMap (spliceOp .and) = l
  match l with
  | [] => exact hT
  | [c] => simp only [List.map_cons, List.map_nil, and3_single]
  | a :: b :: rest => simp only [tv, tvL_map]

/-- **`or_of` is the OR of its operands** (three-valued, every row). -/
theorem tv_orOf (y : R) (hF : atom (.node (.const (.bool false)) []) y = some false) (xs : List Ex) :
    tv atom sub ι (orOf xs) y = or3 (xs.map (tv atom sub ι · y)) := by
  rw [← or3_flat atom sub ι xs y, orOf, flat_eq_body]
  unfold flatBody
  generalize xs.flatMap (spliceOp .or) = l
  match l with
  | [] => exact hF
  | [c] => simp only [List.map_cons, List.map_nil, or3_single]
  | a :: b :: rest => simp only [tv, tvL_map]

/-- A row passes `Filter(and_of xs)` iff it passes every operand. -/
theorem passes_andOf (y : R) (hT : atom tt y = some true) (xs : List Ex) :
    passes (tv atom sub ι (andOf xs) y) = xs.all (fun x => passes (tv atom sub ι x y)) := by
  rw [tv_andOf atom sub ι y hT, Bool.eq_iff_iff]
  simp only [passes, beq_iff_eq, List.all_eq_true, and3_true, List.mem_map]
  constructor
  · intro h x hx; exact h _ ⟨x, hx, rfl⟩
  · rintro h _ ⟨x, hx, rfl⟩; exact h x hx

end PlannerBuild.E
