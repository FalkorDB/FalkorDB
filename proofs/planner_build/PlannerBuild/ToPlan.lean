import PlannerBuild.AndOf
/-
# `expr_to_plan` / `or_expr_to_plan` / `and_expr_to_plan` (mod.rs)

The binder removes every `Paren` (binder.rs), so
`expr_to_plan` does not unwrap one: a `Paren` is not a connective here
(`wfI` admits inline variables only under AND/OR/NOT) and is planned like any
other scalar. The scalar conjunction is `ExprIR::and_of` (`scalarFilter`).

`FP` is the fragment of the IR these functions build, `run` its runtime
meaning: every operator filters the rows of child 0 (`input` = the upstream
stream `up`, `Argument` = the argument row); `SemiApply`/`AntiSemiApply` keep a
row iff the sub-plan run on it is non-empty / empty; `OrApplyMultiplexer`
keeps it iff some branch `i` has `has_result ^ anti_flags[i]`
(or_apply_multiplexer.rs).

* `toPlan_correct3`: the plan keeps exactly the rows on which the predicate
  is `true` (`tv ι e`), in three-valued logic. Since #3260 a `NOT` over
  `AND`/`OR`/`NOT` is pushed down to the leaves (`pushNot`, `push_not_down`)
  before planning, so no anti-join inverts a predicate that can be null.
  `toPlan_correct` is the two-valued form the callers use.
-/
namespace PlannerBuild.E

inductive FP
  | input
  | arg
  | filter (e : Ex) (c : FP)
  | patSub (g : QG)
  | semi (x s : FP)
  | anti (x s : FP)
  | orMux (fl : List Bool) (x : FP) (bs : List FP)

variable {R : Type} (atom : Ex → R → Option Bool) (sub : QG → R → List R)

mutual
def run (ι : List (Nat × QG)) (up : R → List R) : FP → R → List R
  | .input, r => up r
  | .arg, r => [r]
  | .filter e c, r => (run ι up c r).filter (fun y => passes (tv atom sub ι e y))
  | .patSub g, r => sub g r
  | .semi x s, r => (run ι up x r).filter (fun y => !(run ι up s y).isEmpty)
  | .anti x s, r => (run ι up x r).filter (fun y => (run ι up s y).isEmpty)
  | .orMux fl x bs, r => (run ι up x r).filter (fun y => anyBr ι up fl bs y)
def anyBr (ι : List (Nat × QG)) (up : R → List R) : List Bool → List FP → R → Bool
  | a :: fl, b :: bs, y => ((!(run ι up b y).isEmpty) != a) || anyBr ι up fl bs y
  | _, _, _ => false
end

def inlVar (ι : List (Nat × QG)) : Ex → Option QG
  | .node (.var v) _ => lk ι v.id
  | _ => none

def notInl (ι : List (Nat × QG)) : Ex → Option QG
  | .node .not (c :: _) => inlVar ι c
  | _ => none

def keysOf (ι : List (Nat × QG)) : List Nat := ι.map Prod.fst

/-- The scalar conjunction of `and_expr_to_plan` (mod.rs): `Filter(and_of(scalars))`. -/
def scalarFilter (xs : List Ex) (inp : FP) : FP :=
  match xs with
  | [] => inp
  | xs => .filter (andOf xs) inp

/- `Planner::push_not_down` (mod.rs): the negation of `c` with the `NOT`
moved onto its operands, when `c` (under any `Paren`) is `AND`/`OR`/`NOT`. -/
mutual
def pushNot : Ex → Option Ex
  | .node .paren (c :: _) => pushNot c
  | .node .not (c :: _) => some c
  | .node .and cs => some (.node .or (negL cs))
  | .node .or cs => some (.node .and (negL cs))
  | _ => none
/-- `negate` in `push_not_down`, over the operands. -/
def negL : List Ex → List Ex
  | [] => []
  | c :: cs => (pushNot c).getD (.node .not [c]) :: negL cs
end

/- Termination measure for `expr_to_plan` (a `NOT` weighs twice its operand). -/
mutual
def msz : Ex → Nat
  | .node .not cs => 2 * mszL cs - 1
  | .node _ cs => mszL cs + 1
def mszL : List Ex → Nat
  | [] => 0
  | c :: cs => msz c + mszL cs + 1
end

theorem msz_not (cs : List Ex) : msz (.node .not cs) = 2 * mszL cs - 1 := by simp [msz]
theorem msz_other (d : D) (cs : List Ex) (h : d ≠ .not) : msz (.node d cs) = mszL cs + 1 := by
  cases d <;> simp_all [msz]

mutual
theorem pushNot_msz : ∀ c n, pushNot c = some n → msz n ≤ 2 * msz c
  | .node d cs, n, h => by
    cases d <;> (try simp [pushNot] at h)
    case and => subst h; have := negL_msz cs; simp only [msz_other .or _ (by decide), msz_other .and _ (by decide)]; omega
    case or => subst h; have := negL_msz cs; simp only [msz_other .or _ (by decide), msz_other .and _ (by decide)]; omega
    case not =>
      match cs, h with
      | c :: rest, h =>
        simp only [pushNot, Option.some.injEq] at h; subst h
        simp only [msz_not, mszL]; omega
    case paren =>
      match cs, h with
      | c :: rest, h =>
        simp only [pushNot] at h
        have := pushNot_msz c n h
        simp only [msz_other .paren _ (by decide), mszL]; omega
theorem negL_msz : ∀ cs, mszL (negL cs) ≤ 2 * mszL cs
  | [] => by simp [negL, mszL]
  | c :: cs => by
    have ih := negL_msz cs
    simp only [negL, mszL]
    cases h : pushNot c with
    | none => simp only [Option.getD_none, msz_not, mszL]; omega
    | some n => have := pushNot_msz c n h; simp only [Option.getD_some]; omega
end

theorem msz_pos_lt (c : Ex) (cs : List Ex) : msz c < mszL (c :: cs) := by simp [mszL]; omega
theorem mszL_lt_tail (c : Ex) (cs : List Ex) : mszL cs < mszL (c :: cs) := by simp only [mszL]; omega

mutual
def toPlan (ι : List (Nat × QG)) : Ex → FP → FP
  | .node d cs, inp =>
      match inlVar ι (.node d cs) with
      | some g => .semi inp (.patSub g)
      | none => match notInl ι (.node d cs) with
        | some g => .anti inp (.patSub g)
        | none =>
          if containsInlineVar (keysOf ι) (.node d cs) = false then .filter (.node d cs) inp
          else match d, cs with
            | .or, cs => .orMux ((orClass ι cs).1.map (fun _ => false) ++ (orClass ι cs).2.map Prod.snd)
                inp ((orClass ι cs).1 ++ (orClass ι cs).2.map Prod.fst)
            | .and, cs => andFold ι cs (scalarFilter
                (cs.filter (fun c => !containsInlineVar (keysOf ι) c && !isTT c)) inp)
            | .not, c :: _ => match _h : pushNot c with
              | some n => toPlan ι n inp
              | none => .anti inp (toPlan ι c .arg)
            | d, cs => .filter (.node d cs) inp
termination_by e _ => msz e
decreasing_by
  all_goals first
    | (simp only [msz_other .or _ (by decide), msz_other .and _ (by decide)]; omega)
    | (have := pushNot_msz _ _ ‹pushNot _ = some _›; simp only [msz_not, mszL]; omega)
    | (simp only [msz_not, mszL]; omega)
/-- `or_expr_to_plan`'s branch classification (mod.rs): scalar
branches, and (plan, anti) branches, each in child order. -/
def orClass (ι : List (Nat × QG)) : List Ex → List FP × List (FP × Bool)
  | [] => ([], [])
  | c :: cs =>
    match inlVar ι c with
    | some g => ((orClass ι cs).1, (.patSub g, false) :: (orClass ι cs).2)
    | none => match notInl ι c with
      | some g => ((orClass ι cs).1, (.patSub g, true) :: (orClass ι cs).2)
      | none =>
        if containsInlineVar (keysOf ι) c = false then (.filter c .arg :: (orClass ι cs).1, (orClass ι cs).2)
        else ((orClass ι cs).1, (toPlan ι c .arg, false) :: (orClass ι cs).2)
termination_by cs => mszL cs
decreasing_by all_goals first | exact mszL_lt_tail _ _ | exact msz_pos_lt _ _
/-- `and_expr_to_plan`'s loop over the non-scalar conjuncts (mod.rs). -/
def andFold (ι : List (Nat × QG)) : List Ex → FP → FP
  | [], p => p
  | c :: cs, p => andFold ι cs (if containsInlineVar (keysOf ι) c then toPlan ι c p else p)
termination_by cs _ => mszL cs
decreasing_by all_goals first | exact mszL_lt_tail _ _ | exact msz_pos_lt _ _
end

/-- The body of `expr_to_plan` (mod.rs), by connective. -/
def toPlanGen (ι : List (Nat × QG)) (d : D) (cs : List Ex) (inp : FP) : FP :=
  match inlVar ι (.node d cs) with
  | some g => .semi inp (.patSub g)
  | none => match notInl ι (.node d cs) with
    | some g => .anti inp (.patSub g)
    | none =>
      if containsInlineVar (keysOf ι) (.node d cs) = false then .filter (.node d cs) inp
      else match d with
        | .or => .orMux ((orClass ι cs).1.map (fun _ => false) ++ (orClass ι cs).2.map Prod.snd)
            inp ((orClass ι cs).1 ++ (orClass ι cs).2.map Prod.fst)
        | .and => andFold ι cs (scalarFilter
            (cs.filter (fun c => !containsInlineVar (keysOf ι) c && !isTT c)) inp)
        | .not => match cs with
          | c :: _ => match _h : pushNot c with
            | some n => toPlan ι n inp
            | none => .anti inp (toPlan ι c .arg)
          | [] => .filter (.node .not []) inp
        | _ => .filter (.node d cs) inp

theorem toPlan_gen (ι : List (Nat × QG)) (d : D) (cs : List Ex) (inp : FP) :
    toPlan ι (.node d cs) inp = toPlanGen ι d cs inp := by
  rw [toPlan.eq_def]
  cases d <;> (try cases cs) <;> first | rfl | (simp only [toPlanGen]; done)

/-! ## Correctness -/

mutual
/-- Inline variables occur only where `expr_to_plan` takes the predicate apart. -/
def wfI (ι : List (Nat × QG)) : Ex → Bool
  | .node d cs =>
    !containsInlineVar (keysOf ι) (.node d cs) || (inlVar ι (.node d cs)).isSome ||
      (notInl ι (.node d cs)).isSome ||
      (match d with
       | .not | .or | .and => wfIL ι cs
       | _ => false)
def wfIL (ι : List (Nat × QG)) : List Ex → Bool
  | [] => true
  | c :: cs => wfI ι c && wfIL ι cs
end

theorem and3_some (l : List (Option Bool)) (h : ∀ x ∈ l, x ≠ none) : and3 l ≠ none := by
  induction l with
  | nil => simp [and3]
  | cons x xs ih =>
    simp only [List.mem_cons, forall_eq_or_imp] at h
    have := ih h.2
    simp only [and3]
    revert this; generalize and3 xs = y
    rcases x with _ | _ | _ <;> rcases y with _ | _ | _ <;> simp_all

theorem or3_some (l : List (Option Bool)) (h : ∀ x ∈ l, x ≠ none) : or3 l ≠ none := by
  induction l with
  | nil => simp [or3]
  | cons x xs ih =>
    simp only [List.mem_cons, forall_eq_or_imp] at h
    have := ih h.2
    simp only [or3]
    revert this; generalize or3 xs = y
    rcases x with _ | _ | _ <;> rcases y with _ | _ | _ <;> simp_all

variable (ι : List (Nat × QG)) (H2 : ∀ e y, atom e y ≠ none)
include H2

mutual
theorem tv_some : ∀ e y, tv atom sub ι e y ≠ none
  | .node d cs, y => by
    have hl := tvL_some cs y
    cases d <;> (try simp only [tv]) <;> (try exact H2 _ _)
    case and => exact and3_some _ hl
    case or => exact or3_some _ hl
    case pat g => simp [patT]
    case not =>
      match cs, hl with
      | [c], _ => have := tv_some c y; simp only [tv]; revert this; cases tv atom sub ι c y <;> simp [not3]
      | [], _ => exact H2 _ _
      | _ :: _ :: _, _ => exact H2 _ _
    case paren =>
      match cs, hl with
      | [c], _ => exact tv_some c y
      | [], _ => exact H2 _ _
      | _ :: _ :: _, _ => exact H2 _ _
    case var v => cases lk ι v.id <;> simp [patT, H2]
theorem tvL_some : ∀ l y, ∀ x ∈ tvL atom sub ι l y, x ≠ none
  | [], _ => by simp [tvL]
  | c :: cs, y => by
    simp only [tvL, List.mem_cons, forall_eq_or_imp]
    exact ⟨tv_some c y, tvL_some cs y⟩
end

omit H2 in
theorem anyBr_append (up : R → List R) (y : R) : ∀ (f1 : List Bool) (b1 : List FP) (f2 : List Bool) (b2 : List FP),
    f1.length = b1.length →
    anyBr atom sub ι up (f1 ++ f2) (b1 ++ b2) y = (anyBr atom sub ι up f1 b1 y || anyBr atom sub ι up f2 b2 y)
  | [], [], _, _, _ => by simp [anyBr]
  | a :: f1, b :: b1, f2, b2, h => by
    simp only [List.cons_append, anyBr, List.length_cons, Nat.add_right_cancel_iff] at h ⊢
    rw [anyBr_append up y f1 b1 f2 b2 h, Bool.or_assoc]

omit H2 in
theorem containsInl_child (d : D) (c : Ex) (cs : List Ex) (hc : c ∈ cs)
    (h : containsInlineVar (keysOf ι) (.node d cs) = false) : containsInlineVar (keysOf ι) c = false := by
  unfold containsInlineVar at *
  rw [anyN_eq] at *
  simp only [nodes, List.any_cons, Bool.or_eq_false_iff, List.any_eq_false] at h ⊢
  intro x hx
  apply h.2
  clear h
  induction cs with
  | nil => simp at hc
  | cons c' cs ih =>
    simp only [nodesL, List.mem_append]
    rcases List.mem_cons.1 hc with rfl | hc
    · exact Or.inl hx
    · exact Or.inr (ih hc)

omit H2 in
theorem wfI_noInl (e : Ex) (h : containsInlineVar (keysOf ι) e = false) : wfI ι e = true := by
  cases e; simp [wfI, h]

theorem passes_not3 (c : Ex) (y : R) :
    passes (not3 (tv atom sub ι c y)) = !passes (tv atom sub ι c y) := by
  have := tv_some atom sub ι H2 c y
  revert this; cases tv atom sub ι c y with
  | none => simp
  | some b => cases b <;> simp [passes, not3]

omit H2 in
theorem passes_and3 (l : List (Option Bool)) : passes (and3 l) = l.all passes := by
  rw [Bool.eq_iff_iff]
  simp only [passes, beq_iff_eq, List.all_eq_true]
  exact and3_true l

omit H2 in
theorem passes_or3 (l : List (Option Bool)) : passes (or3 l) = l.any passes := by
  rw [Bool.eq_iff_iff]
  simp only [passes, beq_iff_eq, List.any_eq_true]
  exact or3_true l

omit H2 in
theorem run_filter_arg (up : R → List R) (e : Ex) (y : R) :
    run atom sub ι up (.filter e .arg) y = if passes (tv atom sub ι e y) then [y] else [] := by
  simp only [run]; cases h : passes (tv atom sub ι e y) <;> simp [List.filter, h]

omit H2 in
theorem scalarFilter_correct (up : R → List R) (hT : ∀ y, atom tt y = some true)
    (xs : List Ex) (inp : FP) (r : R) :
    run atom sub ι up (scalarFilter xs inp) r =
      (run atom sub ι up inp r).filter (fun y => xs.all fun x => passes (tv atom sub ι x y)) := by
  match xs with
  | [] =>
    simp only [scalarFilter, List.all_nil]
    exact (List.filter_eq_self.2 (fun _ _ => rfl)).symm
  | x :: rest =>
    simp only [scalarFilter, run]
    congr 1; funext y
    exact passes_andOf atom sub ι y (hT y) _

omit H2 in
theorem filter_filter' {α : Type} (l : List α) (p q : α → Bool) :
    (l.filter p).filter q = l.filter (fun a => p a && q a) := by
  rw [List.filter_filter]; congr 1; funext a; exact Bool.and_comm _ _

omit H2 in
theorem tv_inlVar (e : Ex) (g : QG) (h : inlVar ι e = some g) (y : R) :
    tv atom sub ι e y = patT sub g y := by
  match e, h with
  | .node (.var v) cs, h => simp only [inlVar] at h; simp only [tv, h]

omit H2 in
theorem tv_notInl (e : Ex) (g : QG) (h : notInl ι e = some g) (hs : shapeOK e = true) (y : R) :
    tv atom sub ι e y = not3 (patT sub g y) := by
  match e, h with
  | .node .not (c :: rest), h =>
    simp only [notInl] at h
    obtain ⟨c', hc, -⟩ := shape_one .not (c :: rest) (Or.inl rfl) hs
    injection hc with h1 h2; subst h2
    simp only [tv, tv_inlVar atom sub ι c g h]

omit H2 in
theorem inlVar_none (d : D) (cs : List Ex) (hd : ∀ v, d ≠ .var v) : inlVar ι (.node d cs) = none := by
  cases d <;> simp_all [inlVar]

omit H2 in
theorem notInl_none (d : D) (cs : List Ex) (hd : d ≠ .not) : notInl ι (.node d cs) = none := by
  cases d <;> simp_all [notInl]

omit H2 in
theorem passes_patT (g : QG) (y : R) : passes (patT sub g y) = !(sub g y).isEmpty := by
  simp [passes, patT]

omit H2 in
theorem tv_isTT (c : Ex) (y : R) (h : isTT c = true) : tv atom sub ι c y = atom c y := by
  match c, h with
  | .node (.const (.bool true)) cs, _ => simp only [tv]

omit H2 in
theorem wfI_conn (d : D) (cs : List Ex) (hd : isConn d = true)
    (h1 : inlVar ι (.node d cs) = none) (h2 : notInl ι (.node d cs) = none)
    (h3 : containsInlineVar (keysOf ι) (.node d cs) = true) (hw : wfI ι (.node d cs) = true) :
    wfIL ι cs = true := by
  simp only [wfI, h1, h2, h3, Option.isSome_none, Bool.not_true, Bool.false_or] at hw
  cases d <;> simp_all [isConn]

omit H2 in
theorem wfI_nonconn (d : D) (cs : List Ex) (hd : isConn d = false)
    (h1 : inlVar ι (.node d cs) = none) (h2 : notInl ι (.node d cs) = none)
    (h3 : containsInlineVar (keysOf ι) (.node d cs) = true) (hw : wfI ι (.node d cs) = true) : False := by
  simp only [wfI, h1, h2, h3, Option.isSome_none, Bool.not_true, Bool.false_or] at hw
  cases d <;> simp_all [isConn]

/-! ## `push_not_down` is the three-valued negation -/

omit H2

theorem not3_not3 (x : Option Bool) : not3 (not3 x) = x := by
  cases x with
  | none => rfl
  | some b => cases b <;> rfl

mutual
theorem pushNot_tv : ∀ (c n : Ex) (y : R), shapeOK c = true → pushNot c = some n →
    tv atom sub ι n y = not3 (tv atom sub ι c y)
  | .node d cs, n, y, hs, h => by
    cases d <;> (try simp [pushNot] at h)
    case and =>
      subst h
      simp only [tv]
      rw [negL_tv cs y (shapeL_of _ _ rfl hs), not3_and3]
    case or =>
      subst h
      simp only [tv]
      rw [negL_tv cs y (shapeL_of _ _ rfl hs), not3_or3]
    case not =>
      obtain ⟨c, rfl, -⟩ := shape_one .not cs (Or.inl rfl) hs
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [tv, not3_not3]
    case paren =>
      obtain ⟨c, rfl, hsc⟩ := shape_one .paren cs (Or.inr rfl) hs
      simp only [pushNot] at h
      simp only [tv]
      exact pushNot_tv c n y hsc h
theorem negL_tv : ∀ (cs : List Ex) (y : R), shapeOKL cs = true →
    tvL atom sub ι (negL cs) y = (tvL atom sub ι cs y).map not3
  | [], _, _ => rfl
  | c :: cs, y, hs => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [negL, tvL, List.map_cons]
    rw [negL_tv cs y hs.2]
    cases h : pushNot c with
    | none => simp [tv]
    | some n => simp only [Option.getD_some, pushNot_tv c n y hs.1 h]
end

mutual
theorem pushNot_shape : ∀ (c n : Ex), shapeOK c = true → pushNot c = some n → shapeOK n = true
  | .node d cs, n, hs, h => by
    cases d <;> (try simp [pushNot] at h)
    case and => subst h; simpa [shapeOK] using negL_shape cs (shapeL_of _ _ rfl hs)
    case or => subst h; simpa [shapeOK] using negL_shape cs (shapeL_of _ _ rfl hs)
    case not =>
      obtain ⟨c, rfl, hsc⟩ := shape_one .not cs (Or.inl rfl) hs
      simp only [pushNot, Option.some.injEq] at h; subst h
      exact hsc
    case paren =>
      obtain ⟨c, rfl, hsc⟩ := shape_one .paren cs (Or.inr rfl) hs
      simp only [pushNot] at h
      exact pushNot_shape c n hsc h
theorem negL_shape : ∀ (cs : List Ex), shapeOKL cs = true → shapeOKL (negL cs) = true
  | [], _ => rfl
  | c :: cs, hs => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [negL, shapeOKL, negL_shape cs hs.2, Bool.and_true]
    cases h : pushNot c with
    | none => simp [shapeOK, shapeOKL, hs.1]
    | some n => exact pushNot_shape c n hs.1 h
end

section
variable (p : D → Bool) (hp : p .not = false ∧ p .or = false ∧ p .and = false)
include hp

mutual
theorem pushNot_any : ∀ (c n : Ex), pushNot c = some n → anyN p n = true → anyN p c = true
  | .node d cs, n, h, ha => by
    cases d <;> (try simp [pushNot] at h)
    case and =>
      subst h
      simp only [anyN, hp.2.1, hp.2.2, Bool.false_or] at ha ⊢
      exact negL_any cs ha
    case or =>
      subst h
      simp only [anyN, hp.2.1, hp.2.2, Bool.false_or] at ha ⊢
      exact negL_any cs ha
    case not =>
      match cs, h with
      | c :: rest, h =>
        simp only [pushNot, Option.some.injEq] at h; subst h
        simp [anyN, anyL, ha]
    case paren =>
      match cs, h with
      | c :: rest, h =>
        simp only [pushNot] at h
        simp [anyN, anyL, pushNot_any c n h ha]
theorem negL_any : ∀ (cs : List Ex), anyL p (negL cs) = true → anyL p cs = true
  | [], ha => ha
  | c :: cs, ha => by
    simp only [negL, anyL, Bool.or_eq_true] at ha ⊢
    rcases ha with ha | ha
    · left
      cases h : pushNot c with
      | none => simpa [h, anyN, anyL, hp.1] using ha
      | some n => rw [h] at ha; exact pushNot_any c n h ha
    · exact Or.inr (negL_any cs ha)
end
end

theorem pushNot_noInl (c n : Ex) (h : pushNot c = some n)
    (hc : containsInlineVar (keysOf ι) c = false) : containsInlineVar (keysOf ι) n = false := by
  cases hn : containsInlineVar (keysOf ι) n with
  | false => rfl
  | true =>
    have := pushNot_any (isInl (keysOf ι)) ⟨rfl, rfl, rfl⟩ c n h hn
    unfold containsInlineVar at hc; rw [hc] at this; exact absurd this (by decide)

mutual
theorem pushNot_wfI : ∀ (c n : Ex), shapeOK c = true → wfI ι c = true → pushNot c = some n →
    wfI ι n = true
  | .node d cs, n, hs, hw, h => by
    cases h3 : containsInlineVar (keysOf ι) (.node d cs) with
    | false => exact wfI_noInl ι n (pushNot_noInl ι _ n h h3)
    | true =>
      cases d <;> (try simp [pushNot] at h)
      case and =>
        subst h
        have hw' := wfI_conn ι .and cs rfl rfl rfl h3 hw
        have := negL_wfI cs (shapeL_of _ _ rfl hs) hw'
        simp [wfI, this]
      case or =>
        subst h
        have hw' := wfI_conn ι .or cs rfl rfl rfl h3 hw
        have := negL_wfI cs (shapeL_of _ _ rfl hs) hw'
        simp [wfI, this]
      case not =>
        obtain ⟨c, rfl, -⟩ := shape_one .not cs (Or.inl rfl) hs
        simp only [pushNot, Option.some.injEq] at h; subst h
        cases h2 : notInl ι (.node .not [c]) with
        | some g =>
          simp only [notInl] at h2
          cases c; simp [wfI, h2]
        | none =>
          have := wfI_conn ι .not [c] rfl rfl h2 h3 hw
          simpa [wfIL] using this
      case paren =>
        exact (wfI_nonconn ι .paren cs rfl rfl rfl h3 hw).elim
theorem negL_wfI : ∀ (cs : List Ex), shapeOKL cs = true → wfIL ι cs = true → wfIL ι (negL cs) = true
  | [], _, _ => rfl
  | c :: cs, hs, hw => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [wfIL, Bool.and_eq_true] at hw
    simp only [negL, wfIL, negL_wfI cs hs.2 hw.2, Bool.and_true]
    cases h : pushNot c with
    | none => simp [wfI, wfIL, hw.1]
    | some n => exact pushNot_wfI c n hs.1 hw.1 h
end

/-- `push_not_down` succeeds on every connective `NOT` operand that `wfI` admits,
so the `AntiSemiApply(input, plan(c))` fallback is never taken on one. -/
theorem pushNot_total (c : Ex) (hs : shapeOK c = true) (hw : wfI ι c = true)
    (h1 : inlVar ι c = none) (h3 : containsInlineVar (keysOf ι) c = true) (hp : pushNot c = none) :
    False := by
  cases c with
  | node d cs =>
    cases d <;> (try simp [pushNot] at hp)
    case not =>
      obtain ⟨c, rfl, -⟩ := shape_one .not cs (Or.inl rfl) hs
      simp [pushNot] at hp
    all_goals first
      | exact wfI_nonconn ι _ cs rfl h1 (notInl_none ι _ _ (by decide)) h3 hw
      | exact wfI_nonconn ι _ cs rfl h1 rfl h3 hw

variable (hTT : ∀ c y, isTT c = true → atom c y = some true) (up : R → List R)
include hTT

theorem and_split (cs : List Ex) (y : R) :
    ((cs.filter (fun c => !containsInlineVar (keysOf ι) c && !isTT c)).all
        (fun x => passes (tv atom sub ι x y)) &&
      cs.all (fun c => !containsInlineVar (keysOf ι) c || passes (tv atom sub ι c y))) =
    cs.all (fun c => passes (tv atom sub ι c y)) := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.filter_cons, List.all_cons]
    have key : ∀ (P F G : Bool), (F && G) = cs.all (fun c => passes (tv atom sub ι c y)) →
        (((P && F) && (true && G)) = (P && cs.all (fun c => passes (tv atom sub ι c y)))) ∧
        ((F && ((false || P) && G)) = (P && cs.all (fun c => passes (tv atom sub ι c y)))) := by
      intro P F G h; rw [← h]; cases P <;> cases F <;> cases G <;> simp
    cases h1 : containsInlineVar (keysOf ι) c <;> cases h2 : isTT c
    · simp only [Bool.not_false, Bool.and_self, ↓reduceIte, List.all_cons, Bool.not_false,
        Bool.true_or]
      have := (key (passes (tv atom sub ι c y)) _ _ ih).1
      simpa using this
    · have hp : passes (tv atom sub ι c y) = true := by
        rw [tv_isTT atom sub ι c y h2, hTT c y h2]; rfl
      simp only [Bool.not_false, Bool.not_true, Bool.and_false, Bool.false_eq_true, ↓reduceIte,
        Bool.true_or, Bool.true_and, hp]
      exact ih
    · simp only [Bool.not_true, Bool.false_and, Bool.false_eq_true, ↓reduceIte, Bool.false_or]
      have := (key (passes (tv atom sub ι c y)) _ _ ih).2
      simpa using this
    · simp only [Bool.not_true, Bool.false_and, Bool.false_eq_true, ↓reduceIte, Bool.false_or]
      have := (key (passes (tv atom sub ι c y)) _ _ ih).2
      simpa using this

mutual
/-- The plan keeps exactly the rows on which the predicate is `true`, in
three-valued logic (null atoms allowed). -/
theorem toPlan_correct3 : ∀ (e : Ex) (inp : FP) (r : R), shapeOK e = true → wfI ι e = true →
    run atom sub ι up (toPlan ι e inp) r =
      (run atom sub ι up inp r).filter (fun y => passes (tv atom sub ι e y))
  | .node d cs, inp, r, hs, hw => by
    rw [toPlan_gen ι d cs inp]
    unfold toPlanGen
    cases h1 : inlVar ι (.node d cs) with
    | some g =>
      simp only [run]
      congr 1; funext y
      rw [tv_inlVar atom sub ι _ g h1, passes_patT]
    | none =>
      cases h2 : notInl ι (.node d cs) with
      | some g =>
        simp only [run]
        congr 1; funext y
        rw [tv_notInl atom sub ι _ g h2 hs]
        simp [patT, passes, not3]
      | none =>
        cases h3 : containsInlineVar (keysOf ι) (.node d cs) with
        | false => simp [run]
        | true =>
          simp only [Bool.true_eq_false, ↓reduceIte]
          cases d
          case or =>
            have hw' := wfI_conn ι .or cs rfl h1 h2 h3 hw
            simp only [run]
            congr 1; funext y
            rw [anyBr_append atom sub ι up y _ _ _ _ (by simp), orClass_correct cs y (shapeL_of _ _ rfl hs) hw']
            simp only [tv, passes_or3, tvL_map, List.any_map]; rfl
          case and =>
            have hw' := wfI_conn ι .and cs rfl h1 h2 h3 hw
            simp only
            rw [andFold_correct _ _ r (shapeL_of _ _ rfl hs) hw',
              scalarFilter_correct atom sub ι up (fun y => hTT tt y rfl), filter_filter']
            congr 1; funext y
            rw [and_split atom sub ι hTT cs y]
            simp only [tv, passes_and3, tvL_map, List.all_map]; rfl
          case not =>
            obtain ⟨c, rfl, hsc⟩ := shape_one .not cs (Or.inl rfl) hs
            have hw' := wfI_conn ι .not [c] rfl h1 h2 h3 hw
            simp only [wfIL, Bool.and_true] at hw'
            dsimp only
            split
            next n hp =>
              rw [toPlan_correct3 n inp r (pushNot_shape c n hsc hp) (pushNot_wfI ι c n hsc hw' hp)]
              congr 1; funext y
              rw [pushNot_tv atom sub ι c n y hsc hp]
              simp only [tv]
            next hp =>
              have h1' : inlVar ι c = none := by simpa [notInl] using h2
              have h3' : containsInlineVar (keysOf ι) c = true := by
                simpa [containsInlineVar, anyN, anyL, isInl] using h3
              exact (pushNot_total ι c hsc hw' h1' h3' hp).elim
          all_goals exact (wfI_nonconn ι _ cs rfl h1 h2 h3 hw).elim
termination_by e _ _ _ _ => msz e
decreasing_by
  all_goals subst_vars
  all_goals first
    | exact mszL_lt_tail _ _
    | exact msz_pos_lt _ _
    | (simp only [msz_other .or _ (by decide), msz_other .and _ (by decide)]; omega)
    | (have := pushNot_msz _ _ ‹pushNot _ = some _›; simp only [msz_not, mszL]; omega)
theorem orClass_correct : ∀ (cs : List Ex) (y : R), shapeOKL cs = true → wfIL ι cs = true →
    (anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y ||
      anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y) =
    cs.any (fun c => passes (tv atom sub ι c y))
  | [], y, _, _ => by simp [orClass, anyBr]
  | c :: cs, y, hs, hw => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [wfIL, Bool.and_eq_true] at hw
    have ih := orClass_correct cs y hs.2 hw.2
    simp only [List.any_cons]
    rw [← ih]
    cases h1 : inlVar ι c with
    | some g =>
      simp only [orClass, h1, List.map_cons, anyBr, run, tv_inlVar atom sub ι c g h1, passes_patT]
      generalize anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y = A
      generalize anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y = B
      cases (sub g y).isEmpty <;> cases A <;> cases B <;> rfl
    | none =>
      cases h2 : notInl ι c with
      | some g =>
        simp only [orClass, h1, h2, List.map_cons, anyBr, run, tv_notInl atom sub ι c g h2 hs.1]
        have : passes (not3 (patT sub g y)) = (sub g y).isEmpty := by
          simp [patT, passes, not3]
        rw [this]
        generalize anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y = A
        generalize anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y = B
        cases (sub g y).isEmpty <;> cases A <;> cases B <;> rfl
      | none =>
        cases h3 : containsInlineVar (keysOf ι) c with
        | false =>
          simp only [orClass, h1, h2, h3, ↓reduceIte, List.map_cons, anyBr, run_filter_arg]
          generalize anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y = A
          generalize anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y = B
          cases passes (tv atom sub ι c y) <;> cases A <;> cases B <;> rfl
        | true =>
          simp only [orClass, h1, h2, h3, Bool.true_eq_false, ↓reduceIte, List.map_cons, anyBr]
          rw [toPlan_correct3 c .arg y hs.1 hw.1]
          simp only [run]
          generalize anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y = A
          generalize anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y = B
          cases hp : passes (tv atom sub ι c y) <;> cases A <;> cases B <;> simp [List.filter, hp]
termination_by cs _ _ _ => mszL cs
decreasing_by
  all_goals subst_vars
  all_goals first
    | exact mszL_lt_tail _ _
    | exact msz_pos_lt _ _
    | (simp only [msz_other .or _ (by decide), msz_other .and _ (by decide)]; omega)
    | (have := pushNot_msz _ _ ‹pushNot _ = some _›; simp only [msz_not, mszL]; omega)
theorem andFold_correct : ∀ (cs : List Ex) (p : FP) (r : R), shapeOKL cs = true → wfIL ι cs = true →
    run atom sub ι up (andFold ι cs p) r =
      (run atom sub ι up p r).filter
        (fun y => cs.all fun c => !containsInlineVar (keysOf ι) c || passes (tv atom sub ι c y))
  | [], p, r, _, _ => by
    simp only [andFold, List.all_nil]
    exact (List.filter_eq_self.2 (fun _ _ => rfl)).symm
  | c :: cs, p, r, hs, hw => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [wfIL, Bool.and_eq_true] at hw
    simp only [andFold]
    rw [andFold_correct cs _ r hs.2 hw.2]
    cases h : containsInlineVar (keysOf ι) c with
    | false => simp [h]
    | true =>
      simp only [↓reduceIte]
      rw [toPlan_correct3 c p r hs.1 hw.1, filter_filter']
      simp [h]
termination_by cs _ _ _ _ => mszL cs
decreasing_by
  all_goals subst_vars
  all_goals first
    | exact mszL_lt_tail _ _
    | exact msz_pos_lt _ _
    | (simp only [msz_other .or _ (by decide), msz_other .and _ (by decide)]; omega)
    | (have := pushNot_msz _ _ ‹pushNot _ = some _›; simp only [msz_not, mszL]; omega)
end

include H2 in
/-- The form `plan_filter`'s proofs use (`toPlan_correct3` needs no `H2`). -/
theorem toPlan_correct : ∀ (e : Ex) (inp : FP) (r : R), shapeOK e = true → wfI ι e = true →
    run atom sub ι up (toPlan ι e inp) r =
      (run atom sub ι up inp r).filter (fun y => passes (tv atom sub ι e y)) :=
  toPlan_correct3 atom sub ι hTT up

end PlannerBuild.E
