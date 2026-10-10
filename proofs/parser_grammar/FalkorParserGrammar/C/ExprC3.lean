/-
# Soundness and no-panic for pattern comprehensions, `[`-dispatch and
# shortestPath; the full `parse_primary_expr` dispatch (cypher.rs:1926-2117)
-/
import FalkorParserGrammar.C.ExprC2

namespace FalkorParserGrammar.C

section
variable (o : Oracle)

/-- `oC_RelationshipsPattern` chain inside a pattern comprehension. -/
inductive GPcChain (F : Nat) : QNode Nat → QG Nat → PS → QG Nat → PS → Prop
  | nil {left g s} : GPcChain F left g s g s
  | cons {left g s rr s1 out s'} : GRel o Ext.rust left s rr s1 →
      GPcChain F rr.2 ((g.addNode rr.2).2.addRel rr.1).2 s1 out s' → GPcChain F left g s out s'

theorem pcChain_sound (F : Nat) : ∀ n left g (s : PS) out s', pcChain o F n left g s = .ok out s' →
    GPcChain o F left g s out s'
  | 0, _, _, _, _, _, h => by simp [pcChain] at h
  | n + 1, left, g, s, out, s', h => by
    unfold pcChain at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => rr s2 h2
      exact .cons (relP_sound o F _ left _ _ _ h2) (pcChain_sound F n _ _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil

/-- `oC_PatternComprehension` after `[` and the optional `p =`. -/
inductive GPatComp (F : Nat) : PS → ET → PS → Prop
  | mk {s first s1 g s2 cond s3 res s4} : GNode o s first s1 → (s1.cur = .dash ∨ s1.cur = .lt) →
      GPcChain o F first (QG.empty.addNode first).2 s1 g s2 →
      (s2.cur = .kw .where_ ∧ o.pe false s2.adv = .ok cond s3 ∨ cond = leaf (.bool true) ∧ s3 = s2) →
      s3.cur = .pipe → o.pe false s3.adv = .ok res s4 → s4.cur = .rbrack →
      GPatComp F s (.node .patComp [cond, res]) s4.adv

theorem patCompP_sound (F : Nat) (pv : Option Nat) (s : PS) t s' (h : patCompP o F pv s = .ok t s') :
    GPatComp o F s t s' := by
  unfold patCompP at h
  peel h => first s1 h1
  peel h => c s2 h2
  obtain ⟨hc0, hs⟩ := peek_ok h2; subst hc0; subst s2
  split at h
  · simp at h
  · rename_i hd
    peel h => g s3 h3
    peel h => w s4 h4
    peel h => cond s5 h5
    peel h => u s6 h6; obtain ⟨hp, rfl⟩ := tok_ok h6
    peel h => res s7 h7
    peel h => u s8 h8; obtain ⟨hr, rfl⟩ := tok_ok h8
    obtain ⟨rfl, rfl⟩ := pure_ok h
    have gw : s3.cur = .kw .where_ ∧ o.pe false s3.adv = .ok cond s5 ∨ cond = leaf (.bool true) ∧ s5 = s3 := by
      cases w
      · have := (optK_eq h4).2 rfl; subst s4
        simp only [Bool.false_eq_true, ite_false] at h5
        obtain ⟨rfl, rfl⟩ := pure_ok h5; exact .inr ⟨rfl, rfl⟩
      · obtain ⟨hw, rfl⟩ := (optK_eq h4).1 rfl
        simp only [ite_true] at h5; exact .inl ⟨hw, h5⟩
    exact .mk (nodeP_sound o F _ _ _ h1) (by revert hd; cases s1.cur <;> simp)
      (pcChain_sound o F _ _ _ _ _ _ h3) gw hp h7 hr

theorem attempt_ok {α} {x : P α} {s r s'} (h : attempt x s = .ok r s') :
    (∃ a, r = some a ∧ x s = .ok a s') ∨ (r = none ∧ s' = s) := by
  unfold attempt at h; split at h <;> simp_all

/-- What follows `[` (Cypher.g4:394-398 and the list literal 534-535). The
`bool` is `recurse`: a list literal whose elements the caller parses. -/
inductive GBracket (F : Nat) (forb : Bool) : PS → ET × Bool → PS → Prop
  | comp {s v t s'} : identOf s.cur = some v → s.adv.cur = .kw .in_ →
      GListComp (o.pe false) v s.adv.adv t s' → GBracket F forb s (t, false) s'
  | namedPC {s v t s'} : identOf s.cur = some v → s.adv.cur = .eq → s.adv.adv.cur = .lparen →
      GPatComp o F s.adv.adv t s' → forb = false → ¬ HasAgg t → GBracket F forb s (t, false) s'
  | anonPC {s t s'} : s.cur = .lparen → GPatComp o F s t s' → forb = false → ¬ HasAgg t →
      GBracket F forb s (t, false) s'
  | empty {s} : s.cur = .rbrack → GBracket F forb s (leaf .list, false) s.adv
  | elems {s} : s.cur ≠ .rbrack → GBracket F forb s (leaf .list, true) s

theorem rejectAggregate_ok {t : ET} {s u s'} (h : rejectAggregate t s = .ok u s') : ¬ HasAgg t ∧ s' = s := by
  unfold rejectAggregate at h
  rw [← find_aggregate_name_spec]
  split at h
  · simp at h
  · obtain ⟨_, rfl⟩ := pure_ok h; exact ⟨‹_›, rfl⟩

theorem listLitP_sound (F : Nat) (forb : Bool) (s : PS) r s' (h : listLitP o F forb s = .ok r s') :
    GBracket o F forb s r s' := by
  unfold listLitP at h
  peel h => s0 s1 h1
  simp only [run_getS, R.ok.injEq] at h1; obtain ⟨rfl, rfl⟩ := h1
  peel h => a s2 h2
  peel h => isIn s3 h3
  -- branch 1
  have b1 : ∀ v, a = some v → isIn = true → GBracket o F forb s r s' := by
    intro v ha hi; subst ha hi
    rcases tryIdent_ok h2 with ⟨_, he, hv, rfl⟩ | ⟨he, _, _⟩
    · cases he
      obtain ⟨hin, rfl⟩ := (optK_eq h3).1 rfl
      dsimp only at h
      peel h => t s4 h4
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact .comp hv hin (listCompP_sound _ v _ _ _ h4)
    · cases he
  by_cases hb : ∃ v, a = some v ∧ isIn = true
  · obtain ⟨v, ha, hi⟩ := hb; exact b1 v ha hi
  · have hh : listLitRest o F forb s s2 = .ok r s' := by
      cases a with
      | none => exact h
      | some v => cases isIn with
        | false => exact h
        | true => exact absurd ⟨v, rfl, rfl⟩ hb
    clear h; have h := hh; unfold listLitRest at h
    peel h => u s4 h4
    simp only [run_setS, R.ok.injEq] at h4; obtain ⟨-, rfl⟩ := h4
    peel h => a2 s5 h5
    peel h => eq s6 h6
    peel h => c s7 h7
    obtain ⟨hc0, hs⟩ := peek_ok h7; subst hc0; subst s7
    peel h => r2 s8 h8
    -- branch 2 succeeded?
    cases r2 with
    | some t =>
      dsimp only at h
      split at h
      · simp at h
      · rename_i hf
        peel h => u s9 h9
        obtain ⟨hna, rfl⟩ := rejectAggregate_ok h9
        obtain ⟨rfl, rfl⟩ := pure_ok h
        rcases tryIdent_ok h5 with ⟨v, rfl, hv, rfl⟩ | ⟨rfl, _, rfl⟩
        · cases eq
          · simp at h8
          · obtain ⟨he, rfl⟩ := (opt_eq h6).1 rfl
            generalize hcc : s.adv.adv.cur = cc at h8
            cases cc
            case lparen =>
              rcases attempt_ok h8 with ⟨t', ht, h8'⟩ | ⟨ht, _⟩
              · cases ht
                exact .namedPC hv he hcc (patCompP_sound o F _ _ _ _ h8') (by simpa using hf) hna
              · cases ht
            all_goals simp at h8
        · simp at h8
    | none =>
      dsimp only at h
      peel h => u s9 h9
      simp only [run_setS, R.ok.injEq] at h9; obtain ⟨-, rfl⟩ := h9
      peel h => c3 s10 h10
      obtain ⟨hc0, hs⟩ := peek_ok h10; subst hc0; subst s10
      peel h => r3 s11 h11
      cases r3 with
      | some t =>
        dsimp only at h
        split at h
        · simp at h
        · rename_i hf
          peel h => u s12 h12
          obtain ⟨hna, rfl⟩ := rejectAggregate_ok h12
          obtain ⟨rfl, rfl⟩ := pure_ok h
          split at h11
          · rename_i hl
            rcases attempt_ok h11 with ⟨t', ht, h11'⟩ | ⟨ht, _⟩
            · cases ht
              exact .anonPC hl (patCompP_sound o F _ _ _ _ h11') (by simpa using hf) hna
            · cases ht
          · simp at h11
      | none =>
        dsimp only at h
        peel h => u s12 h12
        simp only [run_setS, R.ok.injEq] at h12; obtain ⟨-, rfl⟩ := h12
        peel h => rb s13 h13
        obtain ⟨rfl, rfl⟩ := pure_ok h
        cases rb
        · have := (opt_eq h13).2 rfl; subst s13
          exact .elems (by unfold opt at h13; split at h13 <;> simp_all)
        · obtain ⟨hr, rfl⟩ := (opt_eq h13).1 rfl
          exact .empty hr

end

end FalkorParserGrammar.C
