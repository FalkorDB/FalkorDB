import PlannerBuild.Decomp
/-
# `expr_to_plan` / `or_expr_to_plan` / `and_expr_to_plan` (mod.rs:1371-1548)

`FP` is the fragment of the IR these functions build, `run` its runtime
meaning: every operator filters the rows of child 0 (`input` = the upstream
stream `up`, `Argument` = the argument row); `SemiApply`/`AntiSemiApply` keep a
row iff the sub-plan run on it is non-empty / empty; `OrApplyMultiplexer`
keeps it iff some branch `i` has `has_result ^ anti_flags[i]`
(or_apply_multiplexer.rs:80-86).

* `toPlan_correct`: with two-valued atoms, the plan keeps exactly the rows on
  which the predicate is `true` (`tv ι e`).
* `toPlan_not_null_bug`: without that hypothesis it is false — `NOT (x OR p)`
  with `x` null and no `p` match keeps the row (mod.rs:1421-1427 plans
  `NOT(complex)` as `AntiSemiApply(input, plan(complex))`, which inverts
  "not true", not "false"). CONFIRMED Rust vs C (see PlannerBuild.lean).
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

/-- The scalar conjunction of `and_expr_to_plan` (mod.rs:1531-1538). -/
def scalarFilter (xs : List Ex) (inp : FP) : FP :=
  match xs with
  | [] => inp
  | [x] => .filter x inp
  | xs => .filter (.node .and xs) inp

mutual
def toPlan (ι : List (Nat × QG)) : Ex → FP → FP
  | .node d cs, inp =>
    match d, cs with
    | .paren, c :: _ => toPlan ι c inp
    | d, cs =>
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
            | .not, c :: _ => .anti inp (toPlan ι c .arg)
            | d, cs => .filter (.node d cs) inp
/-- `or_expr_to_plan`'s branch classification (mod.rs:1452-1484): scalar
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
/-- `and_expr_to_plan`'s loop over the non-scalar conjuncts (mod.rs:1541-1545). -/
def andFold (ι : List (Nat × QG)) : List Ex → FP → FP
  | [], p => p
  | c :: cs, p => andFold ι cs (if containsInlineVar (keysOf ι) c then toPlan ι c p else p)
end

/-- The body of `expr_to_plan` after the `Paren` unwrap (mod.rs:1386-1431). -/
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
          | c :: _ => .anti inp (toPlan ι c .arg)
          | [] => .filter (.node .not []) inp
        | _ => .filter (.node d cs) inp

theorem toPlan_gen (ι : List (Nat × QG)) (d : D) (cs : List Ex) (inp : FP) (h : d ≠ .paren) :
    toPlan ι (.node d cs) inp = toPlanGen ι d cs inp := by
  cases d <;> (try cases cs) <;> first | rfl | exact absurd rfl h | (simp only [toPlan, toPlanGen]; done)

/-! ## The three-valued counterexample (CONFIRMED bug) -/

namespace Cex
def g : QG := ⟨0, [⟨0, 0⟩], [], [], []⟩
def ι : List (Nat × QG) := [(7, g)]
/-- `NOT (x OR p)`: `x` an atom (say `a.v > 1`), `p` the inline pattern id 7. -/
def e : Ex := .node .not [.node .or [.node (.other 0) [], Ex.v ⟨7, 0⟩]]
def at0 : Ex → Unit → Option Bool := fun _ _ => none   -- `a.v` is null
def sub0 : QG → Unit → List Unit := fun _ _ => []       -- no `(a)-->()` match
end Cex

/-- openCypher: `NOT (null OR false)` is null, so the row is filtered out; the
plan keeps it. -/
theorem toPlan_not_null_bug :
    tv Cex.at0 Cex.sub0 Cex.ι Cex.e () = none ∧
    run Cex.at0 Cex.sub0 Cex.ι (fun r => [r]) (toPlan Cex.ι Cex.e .input) () = [()] := by
  constructor <;> rfl

/-! ## Correctness with two-valued atoms -/

mutual
/-- Inline variables occur only where `expr_to_plan` takes the predicate apart. -/
def wfI (ι : List (Nat × QG)) : Ex → Bool
  | .node d cs =>
    !containsInlineVar (keysOf ι) (.node d cs) || (inlVar ι (.node d cs)).isSome ||
      (notInl ι (.node d cs)).isSome ||
      (match d with
       | .paren | .not | .or | .and => wfIL ι cs
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
theorem scalarFilter_correct (up : R → List R) (xs : List Ex) (inp : FP) (r : R) :
    run atom sub ι up (scalarFilter xs inp) r =
      (run atom sub ι up inp r).filter (fun y => xs.all fun x => passes (tv atom sub ι x y)) := by
  match xs with
  | [] =>
    simp only [scalarFilter, List.all_nil]
    exact (List.filter_eq_self.2 (fun _ _ => rfl)).symm
  | [x] => simp [scalarFilter, run]
  | x1 :: x2 :: rest =>
    simp only [scalarFilter, run, tv]
    congr 1; funext y
    rw [tvL_map, passes_and3, List.all_map]; rfl

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
theorem toPlan_correct : ∀ (e : Ex) (inp : FP) (r : R), shapeOK e = true → wfI ι e = true →
    run atom sub ι up (toPlan ι e inp) r =
      (run atom sub ι up inp r).filter (fun y => passes (tv atom sub ι e y))
  | .node d cs, inp, r, hs, hw => by
    by_cases hpar : d = .paren
    · subst hpar
      obtain ⟨c, rfl, hsc⟩ := shape_one .paren cs (Or.inr rfl) hs
      have hwc : wfI ι c = true := by
        cases h3 : containsInlineVar (keysOf ι) (.node .paren [c])
        · exact wfI_noInl ι c (containsInl_child ι .paren c [c] (by simp) h3)
        · have := wfI_conn ι .paren [c] rfl rfl rfl h3 hw
          simpa [wfIL] using this
      simp only [toPlan]
      rw [toPlan_correct c inp r hsc hwc]
      simp only [tv]
    rw [toPlan_gen ι d cs inp hpar]
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
            rw [andFold_correct _ _ r (shapeL_of _ _ rfl hs) hw', scalarFilter_correct, filter_filter']
            congr 1; funext y
            rw [and_split atom sub ι H2 hTT cs y]
            simp only [tv, passes_and3, tvL_map, List.all_map]; rfl
          case not =>
            obtain ⟨c, rfl, hsc⟩ := shape_one .not cs (Or.inl rfl) hs
            have hw' := wfI_conn ι .not [c] rfl h1 h2 h3 hw
            simp only [wfIL, Bool.and_true] at hw'
            simp only [run]
            congr 1; funext y
            rw [toPlan_correct c .arg y hsc hw']
            simp only [run, tv, passes_not3 atom sub ι H2]
            cases hp : passes (tv atom sub ι c y) <;> simp [List.filter, hp]
          all_goals first
            | exact absurd rfl hpar
            | exact (wfI_nonconn ι _ cs rfl h1 h2 h3 hw).elim
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
          rw [toPlan_correct c .arg y hs.1 hw.1]
          simp only [run]
          generalize anyBr atom sub ι up ((orClass ι cs).1.map (fun _ => false)) (orClass ι cs).1 y = A
          generalize anyBr atom sub ι up ((orClass ι cs).2.map Prod.snd) ((orClass ι cs).2.map Prod.fst) y = B
          cases hp : passes (tv atom sub ι c y) <;> cases A <;> cases B <;> simp [List.filter, hp]
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
      rw [toPlan_correct c p r hs.1 hw.1, filter_filter']
      simp [h]
end

end PlannerBuild.E
