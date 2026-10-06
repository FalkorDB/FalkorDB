/-
# Clause grammar (`graph/src/Cypher.g4:30-212`), part 1: projections, WHERE,
# ORDER BY / SKIP / LIMIT, UNWIND, and the updating-clause items

Each relation is a Cypher.g4 rule with its semantic action; `oC_Expression`
is the oracle (`o.pe`). The soundness theorems say: whatever the Rust
accepts is derivable, and the IR it builds is the rule's.
-/
import FalkorParserGrammar.C.Clauses
import FalkorParserGrammar.C.PatternG2

namespace FalkorParserGrammar.C

/-- Clause-level places where the Rust accepts more than Cypher.g4. #3060 (`fe619ac5f`,
fixes #3057) removed three: `MATCH MATCH (n)`, `LOAD CSV WITH FROM` (WITH without
HEADERS) and a call's trailing comma are now refused, so their flags are gone and the
relations below describe those constructs exactly as Cypher.g4 does. -/
structure CExt where
  /-- Consecutive DELETE / SET / REMOVE clauses folded into one IR clause
  (cypher.rs:1172-1187, 3224-3226, 3305-3307). -/
  foldUpd : Bool
  /-- A SET / REMOVE target that is not a variable or a property expression
  (`SET (n.x) = 5`, cypher.rs:3243-3270). Since #3060 a `[` no longer stands in for `(`. -/
  anyTarget : Bool
  /-- `CALL proc` without parentheses inside a query (cypher.rs:1041-1047). -/
  callImplicit : Bool
  /-- A single query ending in reading clauses only, no RETURN and no
  update (left to `validate`, ast.rs:1203-1214). -/
  noReturnEnd : Bool

def CExt.g4 : CExt := ⟨false, false, false, false⟩
def CExt.rust : CExt := ⟨true, true, true, true⟩

section
variable (o : Oracle)

/-- `( SP? oC_Where )?` (Cypher.g4:209-210). -/
inductive GWhere : PS → Option ET → PS → Prop
  | none {s} : GWhere s none s
  | some {s e s'} : s.cur = .kw .where_ → o.pe true s.adv = .ok e s' → GWhere s (some e) s'

theorem whereP_sound (s : PS) f s' (h : whereP o s = .ok f s') : GWhere o s f s' := by
  unfold whereP at h
  peel h => c s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  split at h
  · peel h => u s2 h2
    have := next_ok h2; subst this
    peel h => e s3 h3
    split at h
    · simp at h
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .some ‹_› h3
  · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .none

/-- `oC_SortItem` (Cypher.g4:198-199): the flag is "descending". -/
inductive GSortItem : PS → ET × Bool → PS → Prop
  | plain {s e s1} : o.pe false s = .ok e s1 → GSortItem s (e, false) s1
  | asc {s e s1 k} : o.pe false s = .ok e s1 → (k = CK.asc ∨ k = .ascending) → s1.cur = .kw k →
      GSortItem s (e, false) s1.adv
  | desc {s e s1 k} : o.pe false s = .ok e s1 → (k = CK.desc ∨ k = .descending) → s1.cur = .kw k →
      GSortItem s (e, true) s1.adv

/-- `oC_SortItem ( ',' SP? oC_SortItem )*`. -/
inductive GSortItems : PS → List (ET × Bool) → List (ET × Bool) → PS → Prop
  | one {s acc it s1} : GSortItem o s it s1 → GSortItems s acc (acc ++ [it]) s1
  | more {s acc it s1 out s'} : GSortItem o s it s1 → s1.cur = .comma →
      GSortItems s1.adv (acc ++ [it]) out s' → GSortItems s acc out s'

theorem optK_eq {k s b s'} (h : optK k s = .ok b s') :
    (b = true → s.cur = .kw k ∧ s' = s.adv) ∧ (b = false → s' = s) := opt_eq h

theorem orderItems_sound : ∀ (n : Nat) acc (s : PS) out s',
    orderItems o n acc s = .ok out s' → GSortItems o s acc out s'
  | 0, _, _, _, _, h => by simp [orderItems] at h
  | n + 1, acc, s, out, s', h => by
    unfold orderItems at h
    peel h => e s1 h1
    peel h => a1 s2 h2
    peel h => asc s3 h3
    peel h => desc s4 h4
    peel h => more s5 h5
    -- the sort item
    have gi : GSortItem o s (e, desc) s4 := by
      cases a1
      · have := (optK_eq h2).2 rfl; subst this
        simp only [Bool.false_eq_true, ite_false] at h3
        cases asc
        · have := (optK_eq h3).2 rfl; subst this
          simp only [Bool.false_eq_true, ite_false] at h4
          peel h4 => d1 s6 h6
          cases d1
          · have := (optK_eq h6).2 rfl; subst this
            simp only [Bool.false_eq_true, ite_false] at h4
            cases desc
            · have := (optK_eq h4).2 rfl; subst this; exact .plain h1
            · obtain ⟨hk, rfl⟩ := (optK_eq h4).1 rfl; exact .desc h1 (.inr rfl) hk
          · obtain ⟨hk, rfl⟩ := (optK_eq h6).1 rfl
            simp only [ite_true] at h4
            obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .desc h1 (.inl rfl) hk
        · obtain ⟨hk, rfl⟩ := (optK_eq h3).1 rfl
          simp only [ite_true] at h4
          obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .asc h1 (.inr rfl) hk
      · obtain ⟨hk, rfl⟩ := (optK_eq h2).1 rfl
        simp only [ite_true] at h3
        obtain ⟨rfl, rfl⟩ := pure_ok h3
        simp only [ite_true] at h4
        obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .asc h1 (.inl rfl) hk
    cases more
    · have := (opt_eq h5).2 rfl; subst this
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .one gi
    · obtain ⟨hc, rfl⟩ := (opt_eq h5).1 rfl
      simp only [ite_true] at h
      exact .more gi hc (orderItems_sound n _ _ _ _ h)

/-- `( SP oC_Skip )?` / `( SP oC_Limit )?` (Cypher.g4:188-194). -/
inductive GCount (k : CK) : PS → Option ET → PS → Prop
  | none {s} : GCount k s none s
  | some {s e s'} : s.cur = .kw k → o.pe false s.adv = .ok e s' → GCount k s (some e) s'

theorem countP_sound (k : CK) (s : PS) r s' (h : countP o k s = .ok r s') : GCount o k s r s' := by
  unfold countP at h
  peel h => b s1 h1
  cases b
  · have := (optK_eq h1).2 rfl; subst this
    simp only [Bool.false_eq_true, ite_false] at h
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .none
  · obtain ⟨hk, rfl⟩ := (optK_eq h1).1 rfl
    simp only [ite_true] at h
    peel h => e s2 h2
    split at h
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .some hk h2
    · simp at h

/-- `( SP oC_Order )? ( SP oC_Skip )? ( SP oC_Limit )?` (Cypher.g4:167). -/
inductive GOsl : PS → List (ET × Bool) × Option ET × Option ET → PS → Prop
  | mk {s ob s1 sk s2 li s3} :
      (ob = [] ∧ s1 = s ∨ s.cur = .kw .order ∧ s.adv.cur = .kw .by ∧ GSortItems o s.adv.adv [] ob s1) →
      GCount o .skip s1 sk s2 → GCount o .limit s2 li s3 → GOsl s (ob, sk, li) s3

theorem oslP_sound (n : Nat) (s : PS) r s' (h : oslP o n s = .ok r s') : GOsl o s r s' := by
  unfold oslP at h
  peel h => b s1 h1
  peel h => ob s2 h2
  peel h => sk s3 h3
  peel h => li s4 h4
  obtain ⟨rfl, rfl⟩ := pure_ok h
  refine .mk ?_ (countP_sound o _ _ _ _ h3) (countP_sound o _ _ _ _ h4)
  cases b
  · have := (optK_eq h1).2 rfl; subst this
    simp only [Bool.false_eq_true, ite_false] at h2
    obtain ⟨rfl, rfl⟩ := pure_ok h2; exact .inl ⟨rfl, rfl⟩
  · obtain ⟨hk, rfl⟩ := (optK_eq h1).1 rfl
    simp only [ite_true] at h2
    unfold orderbyP at h2
    peel h2 => u s5 h5
    obtain ⟨hb, rfl⟩ := tok_ok h5
    exact .inr ⟨hk, hb, orderItems_sound o _ _ _ _ _ h2⟩

/-- `oC_ProjectionItem` (Cypher.g4:176-179) with its column name: the alias,
the variable's own name, or the source text. -/
inductive GProjItem : PS → Nat × ET → PS → Prop
  | alias {s e s1 v} : o.pe true s = .ok e s1 → s1.cur = .kw .as_ → identOf s1.adv.cur = some v →
      GProjItem s (v, e) s1.adv.adv
  | var {s e s1 v} : o.pe true s = .ok e s1 → e.root = .var v → GProjItem s (v, e) s1
  | text {s e s1} : o.pe true s = .ok e s1 → GProjItem s (o.txt s s1, e) s1

inductive GProjItems : PS → List (Nat × ET) → List (Nat × ET) → PS → Prop
  | one {s acc it s1} : GProjItem o s it s1 → GProjItems s acc (acc ++ [it]) s1
  | more {s acc it s1 out s'} : GProjItem o s it s1 → s1.cur = .comma →
      GProjItems s1.adv (acc ++ [it]) out s' → GProjItems s acc out s'

theorem namedExprs_sound (m : Bool) : ∀ (n : Nat) acc (s : PS) out s',
    namedExprs o m n acc s = .ok out s' → GProjItems o s acc out s'
  | 0, _, _, _, _, h => by simp [namedExprs] at h
  | n + 1, acc, s, out, s', h => by
    unfold namedExprs at h
    peel h => s0 s1 h1
    simp only [run_getS, R.ok.injEq] at h1; obtain ⟨rfl, rfl⟩ := h1
    peel h => e s2 h2
    peel h => c s3 h3
    obtain ⟨hc0, hs⟩ := peek_ok h3; subst hc0; subst s3
    peel h => it s4 h4
    have gi : GProjItem o s it s4 := by
      split at h4
      · peel h4 => u s5 h5
        have := next_ok h5; subst this
        peel h4 => v s6 h6
        obtain ⟨hv, rfl⟩ := ident_ok h6
        obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .alias h2 ‹_› hv
      · generalize hr : e.root = r at h4
        cases r <;> dsimp only at h4
        case var v => obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .var h2 hr
        all_goals
          split at h4
          · simp at h4
          · peel h4 => s7 s8 h8
            simp only [run_getS, R.ok.injEq] at h8; obtain ⟨rfl, rfl⟩ := h8
            obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .text h2
    peel h => c2 s5 h5
    obtain ⟨hc0, hs⟩ := peek_ok h5; subst hc0; subst s5
    split at h
    · peel h => u s6 h6
      have := next_ok h6; subst this
      exact .more gi ‹_› (namedExprs_sound m n _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .one gi

/-- `oC_ProjectionBody` (Cypher.g4:166-174). -/
inductive GProj : PS → Proj → PS → Prop
  | mk {s d s1 all es s2 osl s3} :
      (d = true ∧ s.cur = .kw .distinct ∧ s1 = s.adv ∨ d = false ∧ s1 = s) →
      (all = true ∧ s1.cur = .star ∧ (es = [] ∧ s2 = s1.adv ∨
          s1.adv.cur = .comma ∧ GProjItems o s1.adv.adv [] es s2) ∨
        all = false ∧ GProjItems o s1 [] es s2) →
      GOsl o s2 osl s3 → GProj s ⟨d, all, es, osl.1, osl.2.1, osl.2.2⟩ s3

theorem projP_sound (m : Bool) (n : Nat) (s : PS) p s' (h : projP o m n s = .ok p s') :
    GProj o s p s' := by
  unfold projP at h
  peel h => d s1 h1
  peel h => st s2 h2
  peel h => ae s3 h3
  peel h => osl s4 h4
  obtain ⟨rfl, rfl⟩ := pure_ok h
  have gd : d = true ∧ s.cur = .kw .distinct ∧ s1 = s.adv ∨ d = false ∧ s1 = s := by
    cases d
    · exact .inr ⟨rfl, (optK_eq h1).2 rfl⟩
    · obtain ⟨hk, e⟩ := (optK_eq h1).1 rfl; exact .inl ⟨rfl, hk, e⟩
  refine .mk gd ?_ (oslP_sound o n _ _ _ h4)
  cases st
  · have := (opt_eq h2).2 rfl; subst this
    simp only [Bool.false_eq_true, ite_false] at h3
    peel h3 => es s5 h5
    obtain ⟨rfl, rfl⟩ := pure_ok h3
    exact .inr ⟨rfl, namedExprs_sound o m _ _ _ _ _ h5⟩
  · obtain ⟨hs, rfl⟩ := (opt_eq h2).1 rfl
    simp only [ite_true] at h3
    peel h3 => cm s5 h5
    cases cm
    · have := (opt_eq h5).2 rfl; subst this
      simp only [Bool.false_eq_true, ite_false] at h3
      obtain ⟨rfl, rfl⟩ := pure_ok h3
      exact .inl ⟨rfl, hs, .inl ⟨rfl, rfl⟩⟩
    · obtain ⟨hc, rfl⟩ := (opt_eq h5).1 rfl
      simp only [ite_true] at h3
      peel h3 => es s6 h6
      obtain ⟨rfl, rfl⟩ := pure_ok h3
      exact .inl ⟨rfl, hs, .inr ⟨hc, namedExprs_sound o m _ _ _ _ _ h6⟩⟩

/-- `oC_With` after WITH (Cypher.g4:156-157). -/
theorem withP_sound (w : Bool) (n : Nat) (s : PS) r s' (h : withP o w n s = .ok r s') :
    ∃ p s1 f, GProj o s p s1 ∧ GWhere o s1 f s' ∧ r = .with_ p f w := by
  unfold withP at h
  peel h => p s1 h1
  peel h => f s2 h2
  obtain ⟨rfl, rfl⟩ := pure_ok h
  exact ⟨p, s1, f, projP_sound o true n _ _ _ h1, whereP_sound o _ _ _ h2, rfl⟩

/-- `oC_Return` after RETURN (Cypher.g4:161-162). -/
theorem returnP_sound (w : Bool) (n : Nat) (s : PS) r s' (h : returnP o w n s = .ok r s') :
    ∃ p, GProj o s p s' ∧ r = .return_ p w := by
  unfold returnP at h
  peel h => p s1 h1
  obtain ⟨rfl, rfl⟩ := pure_ok h
  exact ⟨p, projP_sound o false n _ _ _ h1, rfl⟩

/-- `oC_Unwind` after UNWIND (Cypher.g4:87-88). -/
inductive GUnwind : PS → IR → PS → Prop
  | mk {s e s1 v} : o.pe false s = .ok e s1 → s1.cur = .kw .as_ → identOf s1.adv.cur = some v →
      GUnwind s (.unwind e v) s1.adv.adv

theorem unwindP_sound (s : PS) r s' (h : unwindP o s = .ok r s') : GUnwind o s r s' := by
  unfold unwindP at h
  peel h => e s1 h1
  peel h => u s2 h2
  peel h => v s3 h3
  obtain ⟨hk, rfl⟩ := tok_ok h2
  obtain ⟨hv, rfl⟩ := ident_ok h3
  obtain ⟨rfl, rfl⟩ := pure_ok h
  exact .mk h1 hk hv

end

end FalkorParserGrammar.C
