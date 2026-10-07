import PlannerBuild.Expr
/-
# WHERE decomposition: pattern predicates into SemiApply / AntiSemiApply / OrApplyMultiplexer

Reference semantics: openCypher's three-valued WHERE. `tv ι e r` is the truth
value (`none` = null) of a predicate on row `r`: Kleene `AND`/`OR`/`NOT`,
`Paren` transparent, a bare pattern is `true` iff its sub-plan has a row, and a
synthetic inline variable (id in `ι`, the `inline_map`) stands for its pattern.
Everything else is an atom (`atom`, a parameter). A row passes a `Filter` iff
the value is `true` (filter.rs). `tv [] e` is the meaning of the user's WHERE.

| here | there |
| --- | --- |
| `CSt`, `mint`, `collect`/`collectL` | `collect_patterns_and_rebuild` mod.rs:1692-1784 (inline ids minted at mod.rs:1712/1615) |
| `FP`, `run` | IR filter operators: `Filter` (filter.rs), `SemiApply`/`AntiSemiApply` (semi_apply.rs), `OrApplyMultiplexer` (or_apply_multiplexer.rs: row passes iff some branch has `has_result ^ is_anti`), `Argument` |
| `FP.patSub g` | `build_pattern_sub_plan` mod.rs:1038-1049 (its rows are `sub g`) |
| `toPlan` | `expr_to_plan` mod.rs:1493-1554 |
| `orClass`, `orPlan` | `or_expr_to_plan` mod.rs:1560-1622 |
| `andFold`, `andPlan` | `and_expr_to_plan` mod.rs:1626-1670 |
| `planFilter` | `plan_filter` mod.rs:2597-2635 (after `extract_filter_comprehensions`) |

Results: `collect_value` / `collect_passes` (the rebuild is exact, also in
three-valued logic); `toPlan_correct` (the decomposition is exact when atoms
are two-valued); `planFilter_scalar_correct` (no inline patterns: exact in
three-valued logic); `toPlan_not_null_bug` (counterexample: `NOT (x OR p)`
with `x` null keeps the row — CONFIRMED on Rust vs C); `not3_or3`/`not3_and3`
(De Morgan holds in Kleene logic, the basis of the suggested fix).
-/
namespace PlannerBuild.E

/-! ## Kleene logic -/

def not3 : Option Bool → Option Bool
  | some b => some !b
  | none => none

def and3 : List (Option Bool) → Option Bool
  | [] => some true
  | x :: xs => match x, and3 xs with
    | some false, _ => some false
    | _, some false => some false
    | some true, y => y
    | none, _ => none

def or3 : List (Option Bool) → Option Bool
  | [] => some false
  | x :: xs => match x, or3 xs with
    | some true, _ => some true
    | _, some true => some true
    | some false, y => y
    | none, _ => none

def passes (x : Option Bool) : Bool := x == some true

theorem and3_true (l : List (Option Bool)) : and3 l = some true ↔ ∀ x ∈ l, x = some true := by
  induction l with
  | nil => simp [and3]
  | cons x xs ih =>
    rw [List.forall_mem_cons, ← ih]
    simp only [and3]
    generalize and3 xs = y
    rcases x with _ | _ | _ <;> rcases y with _ | _ | _ <;> simp

theorem or3_true (l : List (Option Bool)) : or3 l = some true ↔ ∃ x ∈ l, x = some true := by
  induction l with
  | nil => simp [or3]
  | cons x xs ih =>
    simp only [List.mem_cons, exists_eq_or_imp]
    rw [← ih]
    simp only [or3]
    generalize or3 xs = y
    rcases x with _ | _ | _ <;> rcases y with _ | _ | _ <;> simp

theorem not3_or3 (l : List (Option Bool)) : not3 (or3 l) = and3 (l.map not3) := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [or3, and3, List.map_cons, ← ih]
    rcases x with _ | _ | _ <;> rcases or3 xs with _ | _ | _ <;> rfl

theorem not3_and3 (l : List (Option Bool)) : not3 (and3 l) = or3 (l.map not3) := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [or3, and3, List.map_cons, ← ih]
    rcases x with _ | _ | _ <;> rcases and3 xs with _ | _ | _ <;> rfl

/-! ## Predicate semantics -/

variable {R : Type} (atom : Ex → R → Option Bool) (sub : QG → R → List R)

def lk (ι : List (Nat × QG)) (i : Nat) : Option QG := ι.lookup i

def patT (g : QG) (r : R) : Option Bool := some !(sub g r).isEmpty

mutual
def tv (ι : List (Nat × QG)) : Ex → R → Option Bool
  | .node d cs, r => match d with
    | .and => and3 (tvL ι cs r)
    | .or => or3 (tvL ι cs r)
    | .not => match cs with
      | [c] => not3 (tv ι c r)
      | _ => atom (.node d cs) r
    | .paren => match cs with
      | [c] => tv ι c r
      | _ => atom (.node d cs) r
    | .pat g => patT sub g r
    | .var v => match lk ι v.id with
      | some g => patT sub g r
      | none => atom (.node d cs) r
    | _ => atom (.node d cs) r
def tvL (ι : List (Nat × QG)) : List Ex → R → List (Option Bool)
  | [], _ => []
  | c :: cs, r => tv ι c r :: tvL ι cs r
end

theorem tvL_map (ι : List (Nat × QG)) (l : List Ex) (r : R) :
    tvL atom sub ι l r = l.map (fun c => tv atom sub ι c r) := by
  induction l with
  | nil => rfl
  | cons c cs ih => simp [tvL, ih]

/-! ## `collect_patterns_and_rebuild` (mod.rs:1692-1784) -/

structure CSt where
  lens : Nat → Nat
  ext : List (QG × Bool)
  inl : List (Nat × QG)

def scopeOf (g : QG) : Nat := (g.vars.head?.map V.scope).getD 0

/-- Inline variable minted at mod.rs:1708-1715 / 1611-1618: id = current length
of the pattern's first variable's scope table, which then grows by one. -/
def mint (st : CSt) (g : QG) : Nat × CSt :=
  let s := scopeOf g
  (st.lens s, { st with lens := fun t => if t = s then st.lens s + 1 else st.lens t,
                        inl := (st.lens s, g) :: st.inl })

/-- `NOT` directly wrapping a pattern (mod.rs:1722-1724). -/
def notPat : List Ex → Option QG
  | .node (.pat g) _ :: _ => some g
  | _ => none

mutual
def collect (st : CSt) (ce : Bool) : Ex → Ex × CSt
  | .node d cs => match d with
    | .pat g =>
      if ce then (tt, { st with ext := st.ext ++ [(g, false)] })
      else ((Ex.v ⟨(mint st g).1, scopeOf g⟩), (mint st g).2)
    | .not => match notPat cs with
      | some g =>
        if ce then (tt, { st with ext := st.ext ++ [(g, true)] })
        else (.node .not [Ex.v ⟨(mint st g).1, scopeOf g⟩], (mint st g).2)
      | none => (.node .not (collectL st false cs).1, (collectL st false cs).2)
    | .and => (.node .and (collectL st ce cs).1, (collectL st ce cs).2)
    | d => (.node d (collectL st false cs).1, (collectL st false cs).2)
def collectL (st : CSt) (ce : Bool) : List Ex → List Ex × CSt
  | [] => ([], st)
  | c :: cs => ((collect st ce c).1 :: (collectL (collect st ce c).2 ce cs).1,
                (collectL (collect st ce c).2 ce cs).2)
end

/-! ### Syntactic side conditions -/

/- Shape facts the parser guarantees: `NOT`/`Paren` have one child, patterns none. -/
mutual
def shapeOK : Ex → Bool
  | .node .not cs => cs.length == 1 && shapeOKL cs
  | .node .paren cs => cs.length == 1 && shapeOKL cs
  | .node (.pat _) cs => cs.isEmpty
  | .node _ cs => shapeOKL cs
def shapeOKL : List Ex → Bool
  | [] => true
  | c :: cs => shapeOK c && shapeOKL cs
end

/- Variables of the predicate (real ones: no inline id is among them). -/
mutual
def varIds : Ex → List Nat
  | .node (.var v) cs => v.id :: varIdsL cs
  | .node _ cs => varIdsL cs
def varIdsL : List Ex → List Nat
  | [] => []
  | c :: cs => varIds c ++ varIdsL cs
end

def Fresh (ι : List (Nat × QG)) (ids : List Nat) : Prop := ∀ i ∈ ids, lk ι i = none

/-- `ι` agrees with every inline entry of `st`. -/
def Good (ι : List (Nat × QG)) (st : CSt) : Prop := ∀ p ∈ st.inl, lk ι p.1 = some p.2

theorem good_of_suffix (ι : List (Nat × QG)) (st st' : CSt) (h : ∃ l, st'.inl = l ++ st.inl)
    (hg : Good ι st') : Good ι st := by
  obtain ⟨l, hl⟩ := h
  intro p hp; exact hg p (by rw [hl]; exact List.mem_append_right _ hp)

mutual
theorem collect_inl : ∀ (st : CSt) ce e, ∃ l, (collect st ce e).2.inl = l ++ st.inl
  | st, ce, .node d cs => by
    have hL := collectL_inl st false cs
    have hL' := collectL_inl st ce cs
    cases d <;> simp only [collect] <;> try exact hL
    case pat g =>
      cases ce
      · exact ⟨[((mint st g).1, g)], by simp [mint]⟩
      · exact ⟨[], by simp⟩
    case not =>
      cases h : notPat cs with
      | none => simpa using hL
      | some g =>
        cases ce
        · exact ⟨[((mint st g).1, g)], by simp [mint]⟩
        · exact ⟨[], by simp⟩
    case and => exact hL'
theorem collectL_inl : ∀ (st : CSt) ce l, ∃ m, (collectL st ce l).2.inl = m ++ st.inl
  | st, _, [] => ⟨[], rfl⟩
  | st, ce, c :: cs => by
    obtain ⟨a, ha⟩ := collect_inl st ce c
    obtain ⟨b, hb⟩ := collectL_inl (collect st ce c).2 ce cs
    exact ⟨b ++ a, by simp [collectL, hb, ha]⟩
end

mutual
theorem collect_noPat : ∀ (st : CSt) ce e, hasPatternExpr e = false → collect st ce e = (e, st)
  | st, ce, .node d cs, h => by
    have hL : hasPatternExprL cs = false := by
      cases d <;> simp_all [hasPatternExpr]
    have ih := fun c => collectL_noPat st c cs hL
    cases d <;> simp only [collect, ih] <;> simp_all [hasPatternExpr]
    case not =>
      cases h' : notPat cs with
      | none => simp [ih]
      | some g =>
        match cs, h' with
        | .node (.pat g') _ :: _, _ => simp [hasPatternExprL, hasPatternExpr] at hL
theorem collectL_noPat : ∀ (st : CSt) ce l, hasPatternExprL l = false → collectL st ce l = (l, st)
  | st, _, [], _ => rfl
  | st, ce, c :: cs, h => by
    simp only [hasPatternExprL, Bool.or_eq_false_iff] at h
    simp [collectL, collect_noPat st ce c h.1, collectL_noPat st ce cs h.2]
end

/-! ### The rebuild is exact -/

def sat (r : R) (p : QG × Bool) : Bool := (!(sub p.1 r).isEmpty) != p.2

def AllSat (ext : List (QG × Bool)) (r : R) : Prop := ∀ p ∈ ext, sat sub r p = true

theorem good_collectL_head (ι : List (Nat × QG)) (st : CSt) (ce : Bool) (c : Ex) (cs : List Ex)
    (h : Good ι (collectL st ce (c :: cs)).2) : Good ι (collect st ce c).2 :=
  good_of_suffix ι _ _ (collectL_inl _ ce cs) h

theorem fresh_varIds (ι : List (Nat × QG)) (d : D) (cs : List Ex)
    (h : Fresh ι (varIds (.node d cs))) : Fresh ι (varIdsL cs) := by
  intro i hi; apply h; cases d <;> simp [varIds, hi]

theorem fresh_cons (ι : List (Nat × QG)) (c : Ex) (cs : List Ex) (h : Fresh ι (varIdsL (c :: cs))) :
    Fresh ι (varIds c) ∧ Fresh ι (varIdsL cs) :=
  ⟨fun i hi => h i (by simp [varIdsL, hi]), fun i hi => h i (by simp [varIdsL, hi])⟩

theorem lk_mint (ι : List (Nat × QG)) (st : CSt) (g : QG) (h : Good ι (mint st g).2) :
    lk ι (mint st g).1 = some g := h ((mint st g).1, g) (by simp [mint])

theorem needs_conn (d : D) (cs : List Ex) (hd : isConn d = true)
    (h : needsExtraction (.node d cs) .semiApply = false) : needsExtractionL cs .semiApply = false := by
  have : Mode.semiApply.descend d = .semiApply := by simp [Mode.descend, hd]
  cases d <;> simp_all [needsExtraction, isConn]

theorem needs_other (d : D) (cs : List Ex) (hd : isConn d = false) (hp : isPat d = false)
    (h : needsExtraction (.node d cs) .semiApply = false) : hasPatternExprL cs = false := by
  have : Mode.semiApply.descend d = .exists_ := by simp [Mode.descend, hd]
  rw [← needsExtractionL_exists]
  cases d <;> simp_all [needsExtraction, isConn, isPat]

theorem shape_one (d : D) (cs : List Ex) (hd : d = .not ∨ d = .paren) (h : shapeOK (.node d cs) = true) :
    ∃ c, cs = [c] ∧ shapeOK c = true := by
  rcases hd with rfl | rfl <;>
  · simp only [shapeOK, Bool.and_eq_true, beq_iff_eq] at h
    match cs, h with
    | [c], ⟨_, h2⟩ => exact ⟨c, rfl, by simpa [shapeOKL] using h2⟩

theorem shapeL_of (d : D) (cs : List Ex) (hp : isPat d = false) (h : shapeOK (.node d cs) = true) :
    shapeOKL cs = true := by
  cases d <;> simp_all [shapeOK, isPat]

variable (ι : List (Nat × QG)) (r : R)

/-- Case of a non-connective, non-pattern node: nothing below it changes. -/
theorem collect_other (st : CSt) (d : D) (cs : List Ex) (hd : isConn d = false) (hp : isPat d = false)
    (hn : needsExtraction (.node d cs) .semiApply = false) (hf : Fresh ι (varIds (.node d cs))) :
    collectL st false cs = (cs, st) ∧ tv atom sub ι (.node d cs) r = tv atom sub [] (.node d cs) r := by
  refine ⟨collectL_noPat st false cs (needs_other d cs hd hp hn), ?_⟩
  cases d <;> simp_all [isConn, isPat, tv]
  case var v =>
    have : lk ι v.id = none := hf v.id (by simp [varIds])
    rw [this]; rfl

mutual
theorem collect_value : ∀ (st : CSt) (e : Ex), shapeOK e = true →
    needsExtraction e .semiApply = false → Fresh ι (varIds e) → Good ι (collect st false e).2 →
    tv atom sub ι (collect st false e).1 r = tv atom sub [] e r ∧ (collect st false e).2.ext = st.ext
  | st, .node d cs, hs, hn, hf, hg => by
    cases d
    case pat g =>
      simp only [collect] at hg ⊢
      simp only [Bool.false_eq_true, ↓reduceIte] at hg ⊢
      refine ⟨?_, rfl⟩
      simp [tv, Ex.v, lk_mint ι st g hg]
    case patComp g => simp [needsExtraction] at hn
    case not =>
      obtain ⟨c, rfl, hsc⟩ := shape_one .not cs (Or.inl rfl) hs
      have hnc := needs_conn .not [c] rfl hn
      simp only [needsExtractionL, Bool.or_false] at hnc
      have hfc := (fresh_cons ι c [] (fresh_varIds ι _ _ hf)).1
      cases hp : notPat [c] with
      | some g =>
        simp only [collect, hp, Bool.false_eq_true, ↓reduceIte] at hg ⊢
        match c, hp with
        | .node (.pat g') _, hp =>
          simp [notPat] at hp; subst hp
          refine ⟨?_, rfl⟩
          simp [tv, Ex.v, lk_mint ι st _ hg, tvL]
      | none =>
        simp only [collect, hp] at hg ⊢
        simp only [collectL] at hg ⊢
        have ih := collect_value st c hsc hnc hfc (good_collectL_head ι st false c [] hg)
        refine ⟨?_, ih.2⟩
        simp [tv, ih.1]
    case paren =>
      obtain ⟨c, rfl, hsc⟩ := shape_one .paren cs (Or.inr rfl) hs
      have hnc := needs_conn .paren [c] rfl hn
      simp only [needsExtractionL, Bool.or_false] at hnc
      have hfc := (fresh_cons ι c [] (fresh_varIds ι _ _ hf)).1
      simp only [collect, collectL] at hg ⊢
      have ih := collect_value st c hsc hnc hfc (good_collectL_head ι st false c [] hg)
      exact ⟨by simp [tv, ih.1], ih.2⟩
    case and =>
      simp only [collect] at hg ⊢
      have ih := collectL_value st cs (shapeL_of _ _ rfl hs) (needs_conn .and cs rfl hn)
        (fresh_varIds ι _ _ hf) hg
      exact ⟨by simp [tv, ih.1], ih.2⟩
    case or =>
      simp only [collect] at hg ⊢
      have ih := collectL_value st cs (shapeL_of _ _ rfl hs) (needs_conn .or cs rfl hn)
        (fresh_varIds ι _ _ hf) hg
      exact ⟨by simp [tv, ih.1], ih.2⟩
    all_goals
      first
      | (have ho := collect_other atom sub ι r st _ cs rfl rfl hn hf
         simp only [collect, ho.1]
         exact ⟨ho.2, by trivial⟩)
theorem collectL_value : ∀ (st : CSt) (l : List Ex), shapeOKL l = true →
    needsExtractionL l .semiApply = false → Fresh ι (varIdsL l) → Good ι (collectL st false l).2 →
    tvL atom sub ι (collectL st false l).1 r = tvL atom sub [] l r ∧ (collectL st false l).2.ext = st.ext
  | st, [], _, _, _, _ => ⟨rfl, rfl⟩
  | st, c :: cs, hs, hn, hf, hg => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [needsExtractionL, Bool.or_eq_false_iff] at hn
    have hfc := fresh_cons ι c cs hf
    simp only [collectL] at hg ⊢
    have ih1 := collect_value st c hs.1 hn.1 hfc.1 (good_of_suffix ι _ _ (collectL_inl _ false cs) hg)
    have ih2 := collectL_value (collect st false c).2 cs hs.2 hn.2 hfc.2 hg
    exact ⟨by simp [tvL, ih1.1, ih2.1], by rw [ih2.2, ih1.2]⟩
end

theorem allSat_append (ext : List (QG × Bool)) (p : QG × Bool) :
    AllSat sub (ext ++ [p]) r ↔ AllSat sub ext r ∧ sat sub r p = true := by
  simp [AllSat, or_imp, forall_and]

theorem tvL_true (ι : List (Nat × QG)) (l : List Ex) :
    (∀ x ∈ tvL atom sub ι l r, x = some true) ↔ ∀ c ∈ l, tv atom sub ι c r = some true := by
  rw [tvL_map]; simp

theorem tv_tt (ι : List (Nat × QG)) : tv atom sub ι tt r = atom tt r := by
  simp only [tt, tv]

theorem collect_true_eq_false (st : CSt) (d : D) (cs : List Ex) (h1 : d ≠ .and) (h2 : d ≠ .not)
    (h3 : isPat d = false) : collect st true (.node d cs) = collect st false (.node d cs) := by
  cases d <;> simp_all [collect, isPat]

mutual
/-- Top-level conjuncts: a row passes the original predicate and the queued
semi-joins iff it passes the rebuilt predicate and the (longer) queue. Exact in
three-valued logic. -/
theorem collect_passes (hTT : atom tt r = some true) : ∀ (st : CSt) (e : Ex), shapeOK e = true →
    needsExtraction e .semiApply = false → Fresh ι (varIds e) → Good ι (collect st true e).2 →
    ((tv atom sub [] e r = some true ∧ AllSat sub st.ext r) ↔
     (tv atom sub ι (collect st true e).1 r = some true ∧ AllSat sub (collect st true e).2.ext r))
  | st, .node d cs, hs, hn, hf, hg => by
    by_cases hand : d = .and
    · subst hand
      simp only [collect] at hg ⊢
      have ih := collectL_passes hTT st cs (shapeL_of _ _ rfl hs) (needs_conn .and cs rfl hn)
        (fresh_varIds ι _ _ hf) hg
      simp only [tv, and3_true, tvL_true] at ih ⊢
      exact ih
    by_cases hnot : d = .not
    · subst hnot
      obtain ⟨c, rfl, hsc⟩ := shape_one .not cs (Or.inl rfl) hs
      cases hp : notPat [c] with
      | some g =>
        simp only [collect, hp, ↓reduceIte] at hg ⊢
        match c, hp with
        | .node (.pat g') _, hp =>
          simp [notPat] at hp; subst hp
          rw [allSat_append]
          simp only [tv, tv_tt, hTT, patT, sat, not3]
          cases (sub g' r).isEmpty <;> simp
      | none =>
        have hv := collect_value atom sub ι r st (.node .not [c]) hs hn hf
          (by simpa [collect, hp] using hg)
        have e1 : collect st true (.node .not [c]) = collect st false (.node .not [c]) := by
          simp [collect, hp]
        rw [e1, hv.1, hv.2]
    by_cases hpat : isPat d = true
    · cases d <;> simp [isPat] at hpat
      case pat g =>
        simp only [collect, ↓reduceIte, tv, patT, allSat_append, sat]
        simp only [tt, tv] at hTT ⊢
        rw [hTT]
        cases (sub g r).isEmpty <;> simp [and_comm]
      case patComp g => simp [needsExtraction] at hn
    · have e1 := collect_true_eq_false st d cs hand hnot (by simpa using hpat)
      rw [e1] at hg ⊢
      have hv := collect_value atom sub ι r st (.node d cs) hs hn hf hg
      rw [hv.1, hv.2]
theorem collectL_passes (hTT : atom tt r = some true) : ∀ (st : CSt) (l : List Ex), shapeOKL l = true →
    needsExtractionL l .semiApply = false → Fresh ι (varIdsL l) → Good ι (collectL st true l).2 →
    (((∀ c ∈ l, tv atom sub [] c r = some true) ∧ AllSat sub st.ext r) ↔
     ((∀ c ∈ (collectL st true l).1, tv atom sub ι c r = some true) ∧
       AllSat sub (collectL st true l).2.ext r))
  | st, [], _, _, _, _ => by simp [collectL]
  | st, c :: cs, hs, hn, hf, hg => by
    simp only [shapeOKL, Bool.and_eq_true] at hs
    simp only [needsExtractionL, Bool.or_eq_false_iff] at hn
    have hfc := fresh_cons ι c cs hf
    simp only [collectL] at hg ⊢
    have ih1 := collect_passes hTT st c hs.1 hn.1 hfc.1 (good_of_suffix ι _ _ (collectL_inl _ true cs) hg)
    have ih2 := collectL_passes hTT (collect st true c).2 cs hs.2 hn.2 hfc.2 hg
    simp only [List.mem_cons, forall_eq_or_imp]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3⟩
      have a := ih1.1 ⟨h1, h3⟩
      have b := ih2.1 ⟨h2, a.2⟩
      exact ⟨⟨a.1, b.1⟩, b.2⟩
    · rintro ⟨⟨h1, h2⟩, h3⟩
      have b := ih2.2 ⟨h2, h3⟩
      have a := ih1.2 ⟨h1, b.2⟩
      exact ⟨⟨a.1, b.1⟩, a.2⟩
end

end PlannerBuild.E
