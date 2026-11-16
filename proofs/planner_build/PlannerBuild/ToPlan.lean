import PlannerBuild.AndOf
/-
# `expr_to_plan` / `push_not_down` / `or_expr_to_plan` / `and_expr_to_plan` (mod.rs:1485-1685)

Since #3335 the binder removes every `Paren` (binder.rs:1955-1965), so
`expr_to_plan` no longer unwraps one: a `Paren` is not a connective here
(`wfI` admits inline variables only under AND/OR/NOT) and is planned like any
other scalar. The scalar conjunction is `ExprIR::and_of` (`scalarFilter`).

`FP` is the fragment of the IR these functions build, `run` its runtime
meaning: every operator filters the rows of child 0 (`input` = the upstream
stream `up`, `Argument` = the argument row); `SemiApply`/`AntiSemiApply` keep a
row iff the sub-plan run on it is non-empty / empty; `OrApplyMultiplexer`
keeps it iff some branch `i` has `has_result ^ anti_flags[i]`
(or_apply_multiplexer.rs:80-86).

Since #3260, `NOT` over an AND / OR / NOT (under any `Paren`) is rewritten by
`push_not_down` (`pushNot`, De Morgan) and planned again (mod.rs:1531-1540);
`toPlan` is well-founded on `wt`, where a `NOT` weighs twice its operand.

* `toPlan_correct3`: the plan keeps exactly the rows on which the predicate is
  `true` (`tv ι e`), in three-valued logic (no hypothesis on the atoms);
  `toPlan_correct` is its two-valued corollary (the pre-#3260 statement).
* `pre3260_not_null_bug`: the plan before #3260, `AntiSemiApply(input,
  plan(c))` for `NOT c`, kept a row where `c` is null — `NOT (x OR p)` with
  `x` null and no `p` match (W4-plan-1 / #3241, CONFIRMED Rust vs C);
  `toPlan_not_null_fixed`: the current plan drops it.
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

/-- The scalar conjunction of `and_expr_to_plan` (mod.rs:1643-1647): `Filter(and_of(scalars))`. -/
def scalarFilter (xs : List Ex) (inp : FP) : FP :=
  match xs with
  | [] => inp
  | xs => .filter (andOf xs) inp

/-! ## `push_not_down` (mod.rs:1560-1574) -/

mutual
/-- `Planner::push_not_down`: the negation of `node` with the `NOT` moved onto
its operands, when `node` (under any `Paren`) is an `AND`, `OR` or `NOT`. -/
def pushNot : Ex → Option Ex
  | .node .paren (c :: _) => pushNot c
  | .node .not (c :: _) => some c
  | .node .and cs => some (.node .or (negL cs))
  | .node .or cs => some (.node .and (negL cs))
  | _ => none
/-- The `negate` closure mapped over the children (mod.rs:1561-1566). -/
def negL : List Ex → List Ex
  | [] => []
  | c :: cs => (pushNot c).getD (.node .not [c]) :: negL cs
end

/- Termination weight for `expr_to_plan` after #3260: a `NOT` weighs twice its
operands, so pushing it one level down makes the predicate lighter. -/
mutual
def wt : Ex → Nat
  | .node .not cs => 2 * max 1 (wtL cs)
  | .node .and cs => 1 + wtL cs
  | .node .or cs => 1 + wtL cs
  | .node .paren cs => 1 + wtL cs
  | .node _ cs => 1 + wtL cs
def wtL : List Ex → Nat
  | [] => 0
  | c :: cs => wt c + wtL cs
end

theorem wt_pos : ∀ e, 1 ≤ wt e
  | .node d cs => by cases d <;> simp only [wt] <;> omega

theorem wt_child (d : D) (c : Ex) (cs : List Ex) (h : c ∈ cs) : wt c < wt (.node d cs) := by
  have hle : wt c ≤ wtL cs := by
    induction cs with
    | nil => simp at h
    | cons x xs ih =>
      simp only [wtL]
      rcases List.mem_cons.1 h with rfl | h
      · omega
      · have := ih h; omega
  have := wt_pos c
  cases d <;> simp only [wt] <;> omega

theorem wt_node (d : D) (cs : List Ex) (hd : d ≠ .not) : wt (.node d cs) = 1 + wtL cs := by
  cases d <;> simp_all [wt]

mutual
theorem pushNot_wt : ∀ (c n : Ex), pushNot c = some n → wt n < 2 * wt c
  | .node d cs, n, h => by
    cases d
    case paren =>
      match cs, h with
      | x :: rest, h =>
        have := pushNot_wt x n (by simpa [pushNot] using h)
        simp only [wt, wtL]; omega
      | [], h => simp [pushNot] at h
    case not =>
      match cs, h with
      | x :: rest, h =>
        simp only [pushNot, Option.some.injEq] at h; subst h
        have := wt_pos x
        simp only [wt, wtL]; omega
      | [], h => simp [pushNot] at h
    case and =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      have := negL_wt cs
      simp only [wt]; omega
    case or =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      have := negL_wt cs
      simp only [wt]; omega
    all_goals simp [pushNot] at h
theorem negL_wt : ∀ cs, wtL (negL cs) ≤ 2 * wtL cs
  | [] => by simp [negL, wtL]
  | c :: cs => by
    have ih := negL_wt cs
    simp only [negL, wtL]
    cases h : pushNot c with
    | some m => have := pushNot_wt c m h; simp only [Option.getD_some]; omega
    | none =>
      have := wt_pos c
      simp only [Option.getD_none, wt, wtL]; omega
end

theorem wt_not_push (c n : Ex) (rest : List Ex) (h : pushNot c = some n) :
    2 * wt n + 1 < 2 * wt (.node .not (c :: rest)) + 1 := by
  have := pushNot_wt c n h
  simp only [wt, wtL]; omega

theorem wt_not_child (c : Ex) (rest : List Ex) :
    2 * wt c + 1 < 2 * wt (.node .not (c :: rest)) + 1 := by
  have := wt_pos c
  simp only [wt, wtL]; omega

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
            | .not, c :: rest => match h : pushNot c with
              | some n =>
                have := wt_not_push c n rest h
                toPlan ι n inp
              | none =>
                have := wt_not_child c rest
                .anti inp (toPlan ι c .arg)
            | d, cs => .filter (.node d cs) inp
  termination_by e => 2 * wt e + 1
  decreasing_by
    all_goals first
      | assumption
      | (simp only [wt]; omega)
/-- `or_expr_to_plan`'s branch classification (mod.rs:1586-1618): scalar
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
  termination_by cs => 2 * wtL cs + 2
  decreasing_by
    all_goals (have := wt_pos c; simp only [wtL]; omega)
/-- `and_expr_to_plan`'s loop over the non-scalar conjuncts (mod.rs:1675-1679). -/
def andFold (ι : List (Nat × QG)) : List Ex → FP → FP
  | [], p => p
  | c :: cs, p => andFold ι cs (if containsInlineVar (keysOf ι) c then toPlan ι c p else p)
  termination_by cs => 2 * wtL cs + 2
  decreasing_by
    all_goals (have := wt_pos c; simp only [wtL]; omega)
end

/-- The body of `expr_to_plan` (mod.rs:1486-1540), by connective. -/
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
          | c :: _ => match pushNot c with
            | some n => toPlan ι n inp
            | none => .anti inp (toPlan ι c .arg)
          | [] => .filter (.node .not []) inp
        | _ => .filter (.node d cs) inp

theorem toPlan_gen (ι : List (Nat × QG)) (d : D) (cs : List Ex) (inp : FP) :
    toPlan ι (.node d cs) inp = toPlanGen ι d cs inp := by
  rw [toPlan.eq_def]; unfold toPlanGen
  cases d <;> (try cases cs) <;> (try rfl)
  rename_i x xs
  dsimp only
  cases inlVar ι (.node .not (x :: xs)) <;> dsimp only
  cases notInl ι (.node .not (x :: xs)) <;> dsimp only
  split <;> (try rfl)
  split <;> rename_i heq <;> simp [heq]

/-! ## Correctness with two-valued atoms -/

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

/-! ## `push_not_down` is De Morgan in Kleene logic, and keeps the side conditions -/

omit H2

mutual
theorem tv_pushNot : ∀ (c n : Ex) (y : R), shapeOK c = true → pushNot c = some n →
    tv atom sub ι n y = not3 (tv atom sub ι c y)
  | .node d cs, n, y, hs, h => by
    cases d
    case paren =>
      obtain ⟨x, rfl, hx⟩ := shape_one .paren cs (Or.inr rfl) hs
      have h' : pushNot x = some n := by simpa [pushNot] using h
      rw [tv_pushNot x n y hx h']; simp only [tv]
    case not =>
      obtain ⟨x, rfl, hx⟩ := shape_one .not cs (Or.inl rfl) hs
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [tv]; cases tv atom sub ι x y with
      | none => rfl
      | some b => cases b <;> rfl
    case and =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [tv]
      rw [tvL_negL cs y (shapeL_of .and cs rfl hs), not3_and3]
    case or =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [tv]
      rw [tvL_negL cs y (shapeL_of .or cs rfl hs), not3_or3]
    all_goals simp [pushNot] at h
theorem tvL_negL : ∀ (cs : List Ex) (y : R), shapeOKL cs = true →
    tvL atom sub ι (negL cs) y = (tvL atom sub ι cs y).map not3
  | [], _, _ => rfl
  | c :: cs, y, hs => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [negL, tvL, List.map_cons]
    rw [tvL_negL cs y hs.2]
    cases h : pushNot c with
    | some m => simp only [Option.getD_some]; rw [tv_pushNot c m y hs.1 h]
    | none => simp only [Option.getD_none, tv]
end

mutual
theorem shape_pushNot : ∀ (c n : Ex), shapeOK c = true → pushNot c = some n → shapeOK n = true
  | .node d cs, n, hs, h => by
    cases d
    case paren =>
      obtain ⟨x, rfl, hx⟩ := shape_one .paren cs (Or.inr rfl) hs
      exact shape_pushNot x n hx (by simpa [pushNot] using h)
    case not =>
      obtain ⟨x, rfl, hx⟩ := shape_one .not cs (Or.inl rfl) hs
      simp only [pushNot, Option.some.injEq] at h; subst h; exact hx
    case and =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [shapeOK]; exact shapeL_negL cs (shapeL_of .and cs rfl hs)
    case or =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [shapeOK]; exact shapeL_negL cs (shapeL_of .or cs rfl hs)
    all_goals simp [pushNot] at h
theorem shapeL_negL : ∀ cs, shapeOKL cs = true → shapeOKL (negL cs) = true
  | [], _ => rfl
  | c :: cs, hs => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [negL, shapeOKL, Bool.and_eq_true]
    refine ⟨?_, shapeL_negL cs hs.2⟩
    cases h : pushNot c with
    | some m => exact shape_pushNot c m hs.1 h
    | none => simp [shapeOK, shapeOKL, hs.1]
end

mutual
theorem anyN_pushNot (ks : List Nat) : ∀ (c n : Ex), anyN (isInl ks) c = false →
    pushNot c = some n → anyN (isInl ks) n = false
  | .node d cs, n, ha, h => by
    cases d
    case paren =>
      match cs, h with
      | x :: rest, h =>
        simp only [anyN, anyL, Bool.or_eq_false_iff] at ha
        exact anyN_pushNot ks x n ha.2.1 (by simpa [pushNot] using h)
      | [], h => simp [pushNot] at h
    case not =>
      match cs, h with
      | x :: rest, h =>
        simp only [pushNot, Option.some.injEq] at h; subst h
        simp only [anyN, anyL, Bool.or_eq_false_iff] at ha; exact ha.2.1
      | [], h => simp [pushNot] at h
    case and =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [anyN, Bool.or_eq_false_iff] at ha
      simp only [anyN, isInl, Bool.false_or]
      exact anyL_negL ks cs ha.2
    case or =>
      simp only [pushNot, Option.some.injEq] at h; subst h
      simp only [anyN, Bool.or_eq_false_iff] at ha
      simp only [anyN, isInl, Bool.false_or]
      exact anyL_negL ks cs ha.2
    all_goals simp [pushNot] at h
theorem anyL_negL (ks : List Nat) : ∀ cs, anyL (isInl ks) cs = false → anyL (isInl ks) (negL cs) = false
  | [], _ => rfl
  | c :: cs, ha => by
    simp only [anyL, Bool.or_eq_false_iff] at ha
    simp only [negL, anyL, Bool.or_eq_false_iff]
    refine ⟨?_, anyL_negL ks cs ha.2⟩
    cases h : pushNot c with
    | some m => exact anyN_pushNot ks c m ha.1 h
    | none => simp [anyN, anyL, isInl, ha.1]
end

theorem wfI_of_inlVar (e : Ex) (h : (inlVar ι e).isSome = true) : wfI ι e = true := by
  rcases e with ⟨d, cs⟩; simp [wfI, h]

mutual
theorem wfI_pushNot : ∀ (c n : Ex), shapeOK c = true → wfI ι c = true → pushNot c = some n →
    wfI ι n = true
  | .node d cs, n, hs, hw, h => by
    cases hc : containsInlineVar (keysOf ι) (.node d cs) with
    | false => exact wfI_noInl ι n (anyN_pushNot (keysOf ι) _ n hc h)
    | true =>
      cases d
      case paren => simp [wfI, hc, inlVar, notInl] at hw
      case not =>
        obtain ⟨x, rfl, hx⟩ := shape_one .not cs (Or.inl rfl) hs
        simp only [pushNot, Option.some.injEq] at h; subst h
        simp only [wfI, hc, inlVar, notInl, wfIL, Bool.not_true, Option.isSome_none, Bool.false_or,
          Bool.and_true, Bool.or_eq_true] at hw
        rcases hw with hw | hw
        · exact wfI_of_inlVar ι x hw
        · exact hw
      case and =>
        simp only [pushNot, Option.some.injEq] at h; subst h
        simp only [wfI, hc, inlVar, notInl, Bool.not_true, Option.isSome_none, Bool.false_or] at hw
        have := wfIL_negL cs (shapeL_of .and cs rfl hs) hw
        simp [wfI, this]
      case or =>
        simp only [pushNot, Option.some.injEq] at h; subst h
        simp only [wfI, hc, inlVar, notInl, Bool.not_true, Option.isSome_none, Bool.false_or] at hw
        have := wfIL_negL cs (shapeL_of .or cs rfl hs) hw
        simp [wfI, this]
      all_goals simp [pushNot] at h
theorem wfIL_negL : ∀ cs, shapeOKL cs = true → wfIL ι cs = true → wfIL ι (negL cs) = true
  | [], _, _ => rfl
  | c :: cs, hs, hw => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [wfIL, Bool.and_eq_true] at hw
    simp only [negL, wfIL, Bool.and_eq_true]
    refine ⟨?_, wfIL_negL cs hs.2 hw.2⟩
    cases h : pushNot c with
    | some m => exact wfI_pushNot c m hs.1 hw.1 h
    | none => simp [wfI, wfIL, hw.1]
end

/-- Under `wfI`, the `AntiSemiApply` fallback of mod.rs:1545-1550 is never
reached: a `NOT` whose operand `push_not_down` cannot rewrite and is not an
inline pattern has no inline pattern below it (so it was planned as a Filter). -/
theorem noInl_of_pushNot_none (c : Ex) (hp : pushNot c = none) (hi : inlVar ι c = none)
    (hw : wfI ι c = true) : containsInlineVar (keysOf ι) c = false := by
  rcases c with ⟨d, cs⟩
  cases hc : containsInlineVar (keysOf ι) (.node d cs) with
  | false => rfl
  | true =>
    exfalso
    simp only [wfI, hc, hi, Bool.not_true, Option.isSome_none, Bool.false_or] at hw
    cases d <;> (try cases cs) <;> simp_all [pushNot, notInl, inlVar, containsInlineVar, anyN, anyL, isInl]

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
/-- Since #3260, exact in three-valued logic: no hypothesis on the atoms. -/
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
          cases hd : d
          case or =>
            subst hd
            have hw' := wfI_conn ι .or cs rfl h1 h2 h3 hw
            simp only [run]
            congr 1; funext y
            rw [anyBr_append atom sub ι up y _ _ _ _ (by simp), orClass_correct cs y (shapeL_of _ _ rfl hs) hw']
            simp only [tv, passes_or3, tvL_map, List.any_map]; rfl
          case and =>
            subst hd
            have hw' := wfI_conn ι .and cs rfl h1 h2 h3 hw
            simp only
            rw [andFold_correct _ _ r (shapeL_of _ _ rfl hs) hw',
              scalarFilter_correct atom sub ι up (fun y => hTT tt y rfl), filter_filter']
            congr 1; funext y
            rw [and_split atom sub ι hTT cs y]
            simp only [tv, passes_and3, tvL_map, List.all_map]; rfl
          case not =>
            subst hd
            obtain ⟨c, hcs, hsc⟩ := shape_one .not cs (Or.inl rfl) hs
            rw [hcs] at hs hw h1 h2 h3 ⊢
            have hw' := wfI_conn ι .not [c] rfl h1 h2 h3 hw
            simp only [wfIL, Bool.and_true] at hw'
            simp only
            cases hn : pushNot c with
            | some n =>
              have := wt_not_push c n [] hn
              simp only
              rw [toPlan_correct3 n inp r (shape_pushNot c n hsc hn) (wfI_pushNot ι c n hsc hw' hn)]
              congr 1; funext y
              rw [tv_pushNot atom sub ι c n y hsc hn]; simp only [tv]
            | none =>
              exfalso
              have hi : inlVar ι c = none := by simpa [notInl] using h2
              have := noInl_of_pushNot_none ι c hn hi hw'
              simp only [containsInlineVar, anyN, anyL, isInl, Bool.false_or, Bool.or_false] at h3 this
              rw [this] at h3; exact Bool.false_ne_true h3
          all_goals (subst hd; exact (wfI_nonconn ι _ cs rfl h1 h2 h3 hw).elim)
  termination_by e => 2 * wt e + 1
  decreasing_by
    all_goals
      subst_vars
      first
        | assumption
        | omega
        | (simp only [wt, wtL] at *; omega)
theorem orClass_correct : ∀ (cs : List Ex) (y : R), shapeOKL cs = true → wfIL ι cs = true →
    (anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y ||
      anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y) =
    cs.any (fun c => passes (tv atom sub ι c y))
  | [], y, _, _ => by simp [orClass, anyBr]
  | c :: cs, y, hs, hw => by
    have := wt_pos c
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
          have := wt_pos c
          rw [toPlan_correct3 c .arg y hs.1 hw.1]
          simp only [run]
          generalize anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y = A
          generalize anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y = B
          cases hp : passes (tv atom sub ι c y) <;> cases A <;> cases B <;> simp [List.filter, hp]
  termination_by cs => 2 * wtL cs + 2
  decreasing_by all_goals (simp only [wtL]; omega)
theorem andFold_correct : ∀ (cs : List Ex) (p : FP) (r : R), shapeOKL cs = true → wfIL ι cs = true →
    run atom sub ι up (andFold ι cs p) r =
      (run atom sub ι up p r).filter
        (fun y => cs.all fun c => !containsInlineVar (keysOf ι) c || passes (tv atom sub ι c y))
  | [], p, r, _, _ => by
    simp only [andFold, List.all_nil]
    exact (List.filter_eq_self.2 (fun _ _ => rfl)).symm
  | c :: cs, p, r, hs, hw => by
    have := wt_pos c
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
  termination_by cs => 2 * wtL cs + 2
  decreasing_by all_goals (simp only [wtL]; omega)
end

include H2 in
/-- The pre-#3260 statement (two-valued atoms), now a corollary of `toPlan_correct3`. -/
theorem toPlan_correct : ∀ (e : Ex) (inp : FP) (r : R), shapeOK e = true → wfI ι e = true →
    run atom sub ι up (toPlan ι e inp) r =
      (run atom sub ι up inp r).filter (fun y => passes (tv atom sub ι e y)) :=
  toPlan_correct3 atom sub ι hTT up

/-! ## The three-valued counterexample (CONFIRMED bug, fixed by #3260) -/

namespace Cex
def g : QG := ⟨0, [⟨0, 0⟩], [], [], []⟩
def inl : List (Nat × QG) := [(7, g)]
/-- `x OR p`: `x` an atom (say `a.v > 1`), `p` the inline pattern id 7. -/
def c : Ex := .node .or [.node (.other 0) [], Ex.v ⟨7, 0⟩]
/-- `NOT (x OR p)`. -/
def e : Ex := .node .not [c]
/-- `a.v` is null (only the literal `true` is true). -/
def at0 : Ex → Unit → Option Bool := fun e _ => if isTT e then some true else none
def sub0 : QG → Unit → List Unit := fun _ _ => []       -- no `(a)-->()` match
end Cex

omit hTT in
/-- Before #3260, mod.rs planned `NOT c` (any `c` with an inline pattern) as
`AntiSemiApply(input, expr_to_plan(c))`. openCypher: `NOT (null OR false)` is
null, so the row is filtered out; that plan kept it (Rust 2, C 1 on
`MATCH (a:N) WHERE NOT (a.v > 1 OR (a)-->()) RETURN count(a)`). -/
theorem pre3260_not_null_bug :
    tv Cex.at0 Cex.sub0 Cex.inl Cex.e () = none ∧
    run Cex.at0 Cex.sub0 Cex.inl (fun r => [r]) (.anti .input (toPlan Cex.inl Cex.c .arg)) () = [()] := by
  refine ⟨rfl, ?_⟩
  have hT : ∀ c (y : Unit), isTT c = true → Cex.at0 c y = some true := by
    intro c y h; simp [Cex.at0, h]
  have key := fun y => toPlan_correct3 Cex.at0 Cex.sub0 Cex.inl hT (fun r => [r]) Cex.c .arg y rfl rfl
  simp only [run, key]
  rfl

omit hTT in
/-- Since #3260 (`push_not_down`, mod.rs:1531-1540) the same predicate drops the row. -/
theorem toPlan_not_null_fixed :
    tv Cex.at0 Cex.sub0 Cex.inl Cex.e () = none ∧
    run Cex.at0 Cex.sub0 Cex.inl (fun r => [r]) (toPlan Cex.inl Cex.e .input) () = [] := by
  refine ⟨rfl, ?_⟩
  have hT : ∀ c (y : Unit), isTT c = true → Cex.at0 c y = some true := by
    intro c y h; simp [Cex.at0, h]
  rw [toPlan_correct3 Cex.at0 Cex.sub0 Cex.inl hT (fun r => [r]) Cex.e .input () rfl rfl]
  rfl

end PlannerBuild.E
