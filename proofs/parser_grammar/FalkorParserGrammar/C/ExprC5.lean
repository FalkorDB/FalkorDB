/-
# Soundness of `parse_shortest_path_expr` (cypher.rs:1737-1924)
-/
import FalkorParserGrammar.C.ExprC4

namespace FalkorParserGrammar.C

theorem asU32_one : asU32 1 = 1 := by decide

theorem spTypes_sound : ∀ n acc (s : PS) out s', spTypes n acc s = .ok out s' → GSpTypes s acc out s'
  | 0, _, _, _, _, h => by simp [spTypes] at h
  | n + 1, acc, s, out, s', h => by
    unfold spTypes at h
    peel h => t s1 h1; obtain ⟨ht, rfl⟩ := ident_ok h1
    peel h => p s2 h2
    peel h => c s3 h3
    cases p
    · rcases opt_ok h2 with ⟨h0, _⟩ | ⟨_, hnp, e2⟩
      · cases h0
      subst e2
      cases c
      · rcases opt_ok h3 with ⟨h0, _⟩ | ⟨_, hnc, e3⟩
        · cases h0
        subst e3
        simp only [Bool.or_false, Bool.false_eq_true, ite_false] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact .last ht hnp hnc
      · obtain ⟨hcc, rfl⟩ := (opt_eq h3).1 rfl
        simp only [Bool.false_or, ite_true] at h
        exact .more ht (.inr ⟨hnp, hcc, rfl⟩) (spTypes_sound n _ _ _ _ h)
    · obtain ⟨hpp, rfl⟩ := (opt_eq h2).1 rfl
      simp only [Bool.true_or, ite_true] at h
      cases c
      · rcases opt_ok h3 with ⟨h0, _⟩ | ⟨_, hnc, e3⟩
        · cases h0
        subst e3
        exact .more ht (.inl ⟨hpp, .inr ⟨hnc, rfl⟩⟩) (spTypes_sound n _ _ _ _ h)
      · obtain ⟨hcc, rfl⟩ := (opt_eq h3).1 rfl
        exact .more ht (.inl ⟨hpp, .inl ⟨hcc, rfl⟩⟩) (spTypes_sound n _ _ _ _ h)

/-- The range part of shortestPath's detail. -/
theorem spRange_sound (s : PS) mm s' (h : spRangeP s = .ok mm s') : GSpRange s mm s' := by
  unfold spRangeP at h
  peel h => st s1 h1
  cases st
  · rcases opt_ok h1 with ⟨h0, _⟩ | ⟨_, hn, e1⟩
    · cases h0
    subst e1
    simp only [Bool.false_eq_true, ite_false] at h
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .none hn
  · obtain ⟨hs, rfl⟩ := (opt_eq h1).1 rfl
    simp only [ite_true] at h
    peel h => a s2 h2
    peel h => dd s3 h3
    have ga := hopP_sound _ _ _ h2
    cases dd
    · have := (opt_eq h3).2 rfl; subst s3
      simp only [Bool.false_eq_true, ite_false] at h
      cases a with
      | some i => obtain ⟨rfl, rfl⟩ := pure_ok h; cases ga; exact .star hs (.exact ‹_›)
      | none => obtain ⟨rfl, rfl⟩ := pure_ok h; cases ga; exact .star hs .bare
    · obtain ⟨hd, rfl⟩ := (opt_eq h3).1 rfl
      simp only [ite_true] at h
      peel h => b s4 h4
      obtain ⟨rfl, rfl⟩ := pure_ok h
      have key := GRangeBody.dots ga hd (hopP_sound _ _ _ h4)
      cases a <;> simpa [asU32_one] using GSpRange.star hs key

/-- **parse_shortest_path_expr is sound**: an accepted `shortestPath(...)`
is the Neo4j `shortestPath` pattern with bound end nodes, at most one hop
minimum and no relationship filter, and the tree records exactly its
types, bounds and direction (the left node is the source unless the arrow
points left only). -/
theorem shortestP_sound (F : Nat) (s : PS) t s' (h : shortestP F s = .ok t s') : GShortest s t s' := by
  unfold shortestP at h
  peel h => c0 s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  split at h
  · simp at h
  · rename_i hl0
    peel h => u s2 h2; obtain ⟨hl, rfl⟩ := tok_ok h2
    peel h => a s3 h3
    rcases tryIdent_ok h3 with ⟨src, rfl, hsrc, rfl⟩ | ⟨rfl, _, rfl⟩
    · dsimp only at h
      peel h => c1 s4 h4
      obtain ⟨hc0, hs⟩ := peek_ok h4; subst hc0; subst s4
      split at h
      · simp at h
      · peel h => u s5 h5; obtain ⟨hr, rfl⟩ := tok_ok h5
        peel h => inc s6 h6
        peel h => u s7 h7
        peel h => det s8 h8
        peel h => d s9 h9
        peel h => u s10 h10; obtain ⟨hd2, rfl⟩ := tok_ok h10
        peel h => out s11 h11
        peel h => u s12 h12; obtain ⟨hl2, rfl⟩ := tok_ok h12
        peel h => b s13 h13
        rcases tryIdent_ok h13 with ⟨dst, rfl, hdst, rfl⟩ | ⟨rfl, _, rfl⟩
        · dsimp only at h
          peel h => c3 s14 h14
          obtain ⟨hc0, hs⟩ := peek_ok h14; subst hc0; subst s14
          split at h
          · simp at h
          · peel h => u s15 h15; obtain ⟨hr2, rfl⟩ := tok_ok h15
            peel h => u s16 h16; obtain ⟨hr3, rfl⟩ := tok_ok h16
            split at h
            · simp at h
            · rename_i hmn
              split at h
              · simp at h
              · rename_i hf
                obtain ⟨rfl, rfl⟩ := pure_ok h
                -- direction prefix
                have gi : inc = true ∧ s.adv.adv.adv.cur = .lt ∧ s.adv.adv.adv.adv.cur = .dash ∧
                      s7 = s.adv.adv.adv.adv.adv ∨
                    inc = false ∧ s.adv.adv.adv.cur ≠ .lt ∧ s.adv.adv.adv.cur = .dash ∧
                      s7 = s.adv.adv.adv.adv := by
                  cases inc
                  · rcases opt_ok h6 with ⟨h0, _⟩ | ⟨_, hn, e6⟩
                    · cases h0
                    subst e6
                    obtain ⟨hd, rfl⟩ := tok_ok h7; exact .inr ⟨rfl, hn, hd, rfl⟩
                  · obtain ⟨hlt, rfl⟩ := (opt_eq h6).1 rfl
                    obtain ⟨hd, rfl⟩ := tok_ok h7; exact .inl ⟨rfl, hlt, hd, rfl⟩
                -- the detail
                have gd : GSpDet s7 (d.1, d.2.1, d.2.2.1) s9 := by
                  cases det
                  · rcases opt_ok h8 with ⟨h0, _⟩ | ⟨_, hn, e8⟩
                    · cases h0
                    subst e8
                    simp only [Bool.false_eq_true, ite_false] at h9
                    obtain ⟨rfl, rfl⟩ := pure_ok h9; exact .bare hn
                  · obtain ⟨hb, rfl⟩ := (opt_eq h8).1 rfl
                    simp only [ite_true] at h9
                    peel h9 => v s17 h17
                    peel h9 => col s18 h18
                    peel h9 => ts s19 h19
                    peel h9 => mm s20 h20
                    have gr := spRange_sound _ _ _ h20
                    peel h9 => c2 s21 h21
                    obtain ⟨hc0, hs⟩ := peek_ok h21; subst hc0; subst s21
                    generalize hfv : isFilter s20.cur = fv at h9
                    cases fv
                    case true =>
                      rw [if_pos rfl] at h9
                      peel h9 => u s22 h22
                      peel h9 => u s23 h23
                      obtain ⟨rfl, rfl⟩ := pure_ok h9
                      simp at hf
                    rw [if_neg (by decide)] at h9
                    have hnf : isFilter s20.cur = false := hfv
                    have hnp : ∀ p, s20.cur ≠ .param p := by
                      intro p hp; rw [hp] at hnf; simp [isFilter] at hnf
                    have hnl : s20.cur ≠ .lbrace := by
                      intro hp; rw [hp] at hnf; simp [isFilter] at hnf
                    peel h9 => u s23 h23; obtain ⟨hrb, rfl⟩ := tok_ok h23
                    obtain ⟨rfl, rfl⟩ := pure_ok h9
                    have gv : identOf s7.adv.cur ≠ none ∧ s17 = s7.adv.adv ∨
                        identOf s7.adv.cur = none ∧ s17 = s7.adv := by
                      rcases tryIdent_ok h17 with ⟨_, _, hv, rfl⟩ | ⟨_, hv, rfl⟩
                      · exact .inl ⟨by simp [hv], rfl⟩
                      · exact .inr ⟨hv, rfl⟩
                    have gt : ts = [] ∧ s17.cur ≠ .colon ∧ s19 = s17 ∨
                        s17.cur = .colon ∧ GSpTypes s17.adv [] ts s19 := by
                      cases col
                      · rcases opt_ok h18 with ⟨h0, _⟩ | ⟨_, hn, e18⟩
                        · cases h0
                        subst e18
                        simp only [Bool.false_eq_true, ite_false] at h19
                        obtain ⟨rfl, rfl⟩ := pure_ok h19; exact .inl ⟨rfl, hn, rfl⟩
                      · obtain ⟨hcol, rfl⟩ := (opt_eq h18).1 rfl
                        simp only [ite_true] at h19
                        exact .inr ⟨hcol, spTypes_sound _ _ _ _ _ h19⟩
                    exact .det hb gv gt gr hnp hnl hrb
                have go : out = true ∧ s9.adv.cur = .gt ∧ s11 = s9.adv.adv ∨ out = false ∧ s11 = s9.adv := by
                  cases out
                  · exact .inr ⟨rfl, (opt_eq h11).2 rfl⟩
                  · obtain ⟨hg, rfl⟩ := (opt_eq h11).1 rfl; exact .inl ⟨rfl, hg, rfl⟩
                have := GShortest.mk (by simpa using hl0) hsrc hr gi gd hd2 go hl2 hdst hr2 hr3 (by simpa using hmn)
                rcases (show (inc && !out) = true ∨ (inc && !out) = false by cases (inc && !out) <;> simp)
                  with hc | hc <;> simp only [hc, ite_true, ite_false, Bool.false_eq_true] at this ⊢ <;>
                  exact this
        · simp at h
    · simp at h

end FalkorParserGrammar.C
