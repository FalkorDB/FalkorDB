/-
# Expression constructs outside the fragment of `Model.lean`: CASE,
# quantifiers, reduce, list / pattern comprehensions, shortestPath
# (cypher.rs:1568-1924, 2752-2913)

Sub-expressions are the oracle `pe` (the expression parser re-entered).
-/
import FalkorParserGrammar.C.ClausesG
import FalkorParserGrammar.C.ClausesNP

namespace FalkorParserGrammar.C

/-- The `parse_expr` re-entry used inside these constructs. -/
abbrev PE := P ET

/-- `if let Some(func) = find_aggregate_name(&e) { Err(..) }`. -/
def noAgg (e : ET) : P ET := if (aggName e).isSome then fail else pure e

/-! ## CASE (cypher.rs:1568-1598; Cypher.g4:378-388) -/

def caseAlts (pe : PE) : Nat → List ET → P (List ET)
  | 0, _ => fuelP
  | n + 1, acc => do
    let w ← optK .when
    if w then do
      let c ← pe
      tokK .then
      let r ← pe
      caseAlts pe n (acc ++ [c, r])
    else pure acc

/-- `parse_case_expression`, entered on CASE. -/
def caseP (pe : PE) (F : Nat) : P ET := do
  next
  let c ← peek
  let subj := c != .kw .when
  let sub ← (if subj then do let e ← pe; pure [e] else pure [])
  let conds ← caseAlts pe F []
  if conds = [] then fail else do
  let el ← optK .else_
  let e ← (if el then pe else pure (leaf .null))
  tokK .end_
  pure (.node (.case_ subj) (sub ++ [.node .list conds, e]))

inductive GAlts (pe : PE) : PS → List ET → List ET → PS → Prop
  | nil {s acc} : GAlts pe s acc acc s
  | cons {s acc c s1 r s2 out s'} : s.cur = .kw .when → pe s.adv = .ok c s1 → s1.cur = .kw .then →
      pe s1.adv = .ok r s2 → GAlts pe s2 (acc ++ [c, r]) out s' → GAlts pe s acc out s'

/-- `oC_CaseExpression` after CASE. -/
inductive GCase (pe : PE) : PS → ET → PS → Prop
  | mk {s subj sub s1 conds s2 e s3} :
      (subj = true ∧ s.cur ≠ .kw .when ∧ (∃ x, pe s = .ok x s1 ∧ sub = [x]) ∨
        subj = false ∧ s.cur = .kw .when ∧ sub = [] ∧ s1 = s) →
      GAlts pe s1 [] conds s2 → conds ≠ [] →
      (s2.cur = .kw .else_ ∧ pe s2.adv = .ok e s3 ∨ e = leaf .null ∧ s3 = s2) → s3.cur = .kw .end_ →
      GCase pe s (.node (.case_ subj) (sub ++ [.node .list conds, e])) s3.adv

theorem caseAlts_sound (pe : PE) : ∀ n acc (s : PS) out s', caseAlts pe n acc s = .ok out s' →
    GAlts pe s acc out s'
  | 0, _, _, _, _, h => by simp [caseAlts] at h
  | n + 1, acc, s, out, s', h => by
    unfold caseAlts at h
    peel h => w s1 h1
    cases w
    · have := (optK_eq h1).2 rfl; subst s1
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil
    · obtain ⟨hw, rfl⟩ := (optK_eq h1).1 rfl
      simp only [ite_true] at h
      peel h => c s2 h2
      peel h => u s3 h3; obtain ⟨ht, rfl⟩ := tok_ok h3
      peel h => r s4 h4
      exact .cons hw h2 ht h4 (caseAlts_sound pe n _ _ _ _ h)

theorem caseP_sound (pe : PE) (F : Nat) (s : PS) t s' (h : caseP pe F s = .ok t s') :
    GCase pe s.adv t s' := by
  unfold caseP at h
  peel h => u s1 h1; have := next_ok h1; subst this
  peel h => c s2 h2
  obtain ⟨hc0, hs⟩ := peek_ok h2; subst hc0; subst s2
  peel h => sub s3 h3
  peel h => conds s4 h4
  split at h
  · simp at h
  · rename_i hne
    peel h => el s5 h5
    peel h => e s6 h6
    peel h => u s7 h7; obtain ⟨hend, rfl⟩ := tok_ok h7
    obtain ⟨rfl, rfl⟩ := pure_ok h
    have gsub : ((s.adv.cur != .kw .when) = true ∧ s.adv.cur ≠ .kw .when ∧ (∃ x, pe s.adv = .ok x s3 ∧ sub = [x])) ∨
        ((s.adv.cur != .kw .when) = false ∧ s.adv.cur = .kw .when ∧ sub = [] ∧ s3 = s.adv) := by
      split at h3
      · rename_i hs
        peel h3 => x s8 h8
        obtain ⟨rfl, rfl⟩ := pure_ok h3
        exact .inl ⟨hs, by simpa using hs, x, h8, rfl⟩
      · rename_i hs
        obtain ⟨rfl, rfl⟩ := pure_ok h3
        exact .inr ⟨by simpa using hs, by simpa using hs, rfl, rfl⟩
    have gel : s4.cur = .kw .else_ ∧ pe s4.adv = .ok e s6 ∨ e = leaf .null ∧ s6 = s4 := by
      cases el
      · have := (optK_eq h5).2 rfl; subst s5
        simp only [Bool.false_eq_true, ite_false] at h6
        obtain ⟨rfl, rfl⟩ := pure_ok h6; exact .inr ⟨rfl, rfl⟩
      · obtain ⟨he, rfl⟩ := (optK_eq h5).1 rfl
        simp only [ite_true] at h6; exact .inl ⟨he, h6⟩
    exact .mk gsub (caseAlts_sound pe _ _ _ _ _ h4) hne gel hend

/-! ## Quantifiers (cypher.rs:1600-1657; Cypher.g4:400-414) -/

def quantCode : CT → Option Nat
  | .kw .all => some 0 | .kw .any => some 1 | .kw .none => some 2 | .kw .single => some 3
  | _ => none

/-- `parse_quantifier_expr`; `unreachable!()` on any other keyword. -/
def quantP (pe : PE) : P ET := do
  let c ← peek
  match quantCode c with
  | none => oops
  | some q => do
    next
    tok .lparen
    let v ← ident
    tokK .in_
    let e ← pe
    let w ← optK .where_
    if !w then fail else do
    let cond ← pe
    tok .rparen
    pure (.node (.quant q v) [e, cond])

/-- `oC_Quantifier` with its (here mandatory) `oC_Where`. -/
inductive GQuant (pe : PE) : PS → ET → PS → Prop
  | mk {s q v e s1 cond s2} : quantCode s.cur = some q → s.adv.cur = .lparen →
      identOf s.adv.adv.cur = some v → s.adv.adv.adv.cur = .kw .in_ →
      pe s.adv.adv.adv.adv = .ok e s1 → s1.cur = .kw .where_ → pe s1.adv = .ok cond s2 →
      s2.cur = .rparen → GQuant pe s (.node (.quant q v) [e, cond]) s2.adv

theorem quantP_sound (pe : PE) (s : PS) t s' (h : quantP pe s = .ok t s') : GQuant pe s t s' := by
  unfold quantP at h
  peel h => c s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  generalize hq : quantCode s.cur = q at h
  cases q with
  | none => simp at h
  | some q =>
    dsimp only at h
    peel h => u s2 h2; have := next_ok h2; subst this
    peel h => u s3 h3; obtain ⟨hl, rfl⟩ := tok_ok h3
    peel h => v s4 h4; obtain ⟨hv, rfl⟩ := ident_ok h4
    peel h => u s5 h5; obtain ⟨hin, rfl⟩ := tok_ok h5
    peel h => e s6 h6
    peel h => w s7 h7
    cases w
    · simp at h
    · obtain ⟨hw, rfl⟩ := (optK_eq h7).1 rfl
      simp only [Bool.not_true, Bool.false_eq_true, ite_false] at h
      peel h => cond s8 h8
      peel h => u s9 h9; obtain ⟨hr, rfl⟩ := tok_ok h9
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact .mk hq hl hv hin h6 hw h8 hr

/-- The quantifier's `unreachable!()` is only reached on a non-quantifier
keyword; `parse_primary_expr` calls it only on ALL/ANY/NONE/SINGLE. -/
theorem quantP_np (pe : PE) (hpe : NP pe) (s : PS) (hq : (quantCode s.cur).isSome) :
    quantP pe s ≠ .panic := by
  unfold quantP; rw [run_bind]; simp only [peek]
  cases h : quantCode s.cur with
  | none => simp [h] at hq
  | some q => dsimp only; refine NP.run ?_ s; np_go

/-! ## reduce (cypher.rs:1659-1735; Neo4j syntax, not in Cypher.g4) -/

/-- `parse_reduce_expr`, after `reduce(`. -/
def reduceP (pe : PE) : P ET := do
  let a ← tryIdent
  match a with
  | none => fail
  | some acc => do
    let eq ← opt .eq
    if !eq then fail else do
    let init ← pe
    let init ← noAgg init
    let cm ← opt .comma
    if !cm then fail else do
    let it ← ident
    let i ← optK .in_
    if !i then fail else do
    let l ← pe
    let l ← noAgg l
    let p ← opt .pipe
    if !p then fail else do
    let b ← pe
    let b ← noAgg b
    tok .rparen
    pure (.node (.reduce acc it) [init, l, b])

/-- `reduce( acc = init , it IN list | body )`. -/
inductive GReduce (pe : PE) : PS → ET → PS → Prop
  | mk {s acc init s1 it l s2 b s3} : identOf s.cur = some acc → s.adv.cur = .eq →
      pe s.adv.adv = .ok init s1 → (aggName init).isSome = false → s1.cur = .comma →
      identOf s1.adv.cur = some it → s1.adv.adv.cur = .kw .in_ → pe s1.adv.adv.adv = .ok l s2 →
      (aggName l).isSome = false → s2.cur = .pipe → pe s2.adv = .ok b s3 →
      (aggName b).isSome = false → s3.cur = .rparen →
      GReduce pe s (.node (.reduce acc it) [init, l, b]) s3.adv

theorem noAgg_ok {e : ET} {s t s'} (h : noAgg e s = .ok t s') : t = e ∧ s' = s ∧ (aggName e).isSome = false := by
  unfold noAgg at h; split at h
  · simp at h
  · obtain ⟨rfl, rfl⟩ := pure_ok h
    refine ⟨rfl, rfl, ?_⟩
    rename_i hh; cases h2 : (aggName e).isSome <;> simp_all

theorem reduceP_sound (pe : PE) (s : PS) t s' (h : reduceP pe s = .ok t s') : GReduce pe s t s' := by
  unfold reduceP at h
  peel h => a s1 h1
  rcases tryIdent_ok h1 with ⟨acc, rfl, ha, rfl⟩ | ⟨rfl, _, rfl⟩
  · dsimp only at h
    peel h => eq s2 h2
    cases eq
    · simp at h
    · obtain ⟨he, rfl⟩ := (opt_eq h2).1 rfl
      simp only [Bool.not_true, Bool.false_eq_true, ite_false] at h
      peel h => i0 s3 h3
      peel h => i1 s4 h4; obtain ⟨rfl, rfl, ai⟩ := noAgg_ok h4
      peel h => cm s5 h5
      cases cm
      · simp at h
      · obtain ⟨hc, rfl⟩ := (opt_eq h5).1 rfl
        simp only [Bool.not_true, Bool.false_eq_true, ite_false] at h
        peel h => it s6 h6; obtain ⟨hit, rfl⟩ := ident_ok h6
        peel h => i s7 h7
        cases i
        · simp at h
        · obtain ⟨hin, rfl⟩ := (optK_eq h7).1 rfl
          simp only [Bool.not_true, Bool.false_eq_true, ite_false] at h
          peel h => l0 s8 h8
          peel h => l1 s9 h9; obtain ⟨rfl, rfl, al⟩ := noAgg_ok h9
          peel h => p s10 h10
          cases p
          · simp at h
          · obtain ⟨hp, rfl⟩ := (opt_eq h10).1 rfl
            simp only [Bool.not_true, Bool.false_eq_true, ite_false] at h
            peel h => b0 s11 h11
            peel h => b1 s12 h12; obtain ⟨rfl, rfl, ab⟩ := noAgg_ok h12
            peel h => u s13 h13; obtain ⟨hr, rfl⟩ := tok_ok h13
            obtain ⟨rfl, rfl⟩ := pure_ok h
            exact .mk ha he h3 ai hc hit hin h8 al hp h11 ab hr
  · simp at h

/-! ## List comprehension with projection (cypher.rs:2824-2859; Cypher.g4:394-395) -/

/-- `parse_list_comprehension`, after `[ var IN`. -/
def listCompP (pe : PE) (v : Nat) : P ET := do
  let l ← pe
  let w ← optK .where_
  let c ← (if w then do let c ← pe; pure (some c) else pure none)
  let p ← opt .pipe
  let e ← (if p then do let e ← pe; let e ← noAgg e; pure (some e) else pure none)
  tok .rbrack
  pure (.node (.listComp v) [l, c.getD (leaf (.bool true)), e.getD (leaf (.var v))])

inductive GListComp (pe : PE) (v : Nat) : PS → ET → PS → Prop
  | mk {s l s1 c s2 e s3} : pe s = .ok l s1 →
      (s1.cur = .kw .where_ ∧ (∃ x, pe s1.adv = .ok x s2 ∧ c = some x) ∨ c = none ∧ s2 = s1) →
      (s2.cur = .pipe ∧ (∃ x, pe s2.adv = .ok x s3 ∧ (aggName x).isSome = false ∧ e = some x) ∨
        e = none ∧ s3 = s2) → s3.cur = .rbrack →
      GListComp pe v s (.node (.listComp v) [l, c.getD (leaf (.bool true)), e.getD (leaf (.var v))]) s3.adv

theorem listCompP_sound (pe : PE) (v : Nat) (s : PS) t s' (h : listCompP pe v s = .ok t s') :
    GListComp pe v s t s' := by
  unfold listCompP at h
  peel h => l s1 h1
  peel h => w s2 h2
  peel h => c s3 h3
  peel h => p s4 h4
  peel h => e s5 h5
  peel h => u s6 h6; obtain ⟨hr, rfl⟩ := tok_ok h6
  obtain ⟨rfl, rfl⟩ := pure_ok h
  refine GListComp.mk (s1 := s1) (s2 := s3) (s3 := s5) h1 ?_ ?_ hr
  · cases w
    · have := (optK_eq h2).2 rfl; subst s2
      simp only [Bool.false_eq_true, ite_false] at h3
      obtain ⟨rfl, rfl⟩ := pure_ok h3; exact .inr ⟨rfl, rfl⟩
    · obtain ⟨hw, rfl⟩ := (optK_eq h2).1 rfl
      simp only [ite_true] at h3
      peel h3 => x s7 h7
      obtain ⟨rfl, rfl⟩ := pure_ok h3; exact .inl ⟨hw, x, h7, rfl⟩
  · cases p
    · have := (opt_eq h4).2 rfl; subst s4
      simp only [Bool.false_eq_true, ite_false] at h5
      obtain ⟨rfl, rfl⟩ := pure_ok h5; exact .inr ⟨rfl, rfl⟩
    · obtain ⟨hp, rfl⟩ := (opt_eq h4).1 rfl
      simp only [ite_true] at h5
      peel h5 => x0 s7 h7
      peel h5 => x s8 h8; obtain ⟨rfl, rfl, ax⟩ := noAgg_ok h8
      obtain ⟨rfl, rfl⟩ := pure_ok h5; exact .inl ⟨hp, _, h7, ax, rfl⟩

end FalkorParserGrammar.C
