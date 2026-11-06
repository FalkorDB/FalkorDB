/-
# Soundness of the full `parse_primary_expr` (Cypher.g4:361-374, 425-426)
-/
import FalkorParserGrammar.C.Primary
import FalkorParserGrammar.C.ClausesG2

namespace FalkorParserGrammar.C

/-- An argument list after `(` (`oC_FunctionInvocation`'s arguments):
`( oC_Expression ( ',' oC_Expression )* )? ')'`. Since #3060 a `,` is always followed by an
expression; before it the `close` case could follow `more`, i.e. `f(1,)` was derivable here. -/
inductive GItemsE (pe : PE) : PS → List ET → List ET → PS → Prop
  | last {s acc e s1} : pe s = .ok e s1 → s1.cur = .rparen → GItemsE pe s acc (acc ++ [e]) s1.adv
  | more {s acc e s1 out s'} : pe s = .ok e s1 → s1.cur = .comma →
      GItemsE pe s1.adv (acc ++ [e]) out s' → GItemsE pe s acc out s'

inductive GArgsE (pe : PE) : PS → List ET → List ET → PS → Prop
  | close {s acc} : s.cur = .rparen → GArgsE pe s acc acc s.adv
  | items {s acc out s'} : s.cur ≠ .rparen → GItemsE pe s acc out s' → GArgsE pe s acc out s'

theorem argItemsP_sound (pe : PE) : ∀ n acc (s : PS) out s', argItemsP pe n acc s = .ok out s' →
    GItemsE pe s acc out s'
  | 0, _, _, _, _, h => by simp [argItemsP] at h
  | n + 1, acc, s, out, s', h => by
    unfold argItemsP at h
    peel h => e s2 h2
    peel h => c2 s3 h3
    obtain ⟨hc0, hs⟩ := peek_ok h3; subst hc0; subst s3
    split at h
    · peel h => u s4 h4; have := next_ok h4; subst this
      exact .more h2 ‹_› (argItemsP_sound pe n _ _ _ _ h)
    · peel h => u s4 h4; obtain ⟨hr, rfl⟩ := tok_ok h4
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .last h2 hr

theorem argsP_sound (pe : PE) (n : Nat) acc (s : PS) out s' (h : argsP pe n acc s = .ok out s') :
    GArgsE pe s acc out s' := by
  unfold argsP at h
  peel h => c s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  split at h
  · peel h => u s2 h2; have := next_ok h2; subst this
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .close ‹_›
  · exact .items ‹_› (argItemsP_sound pe n _ _ _ _ h)

/-- `oC_FunctionInvocation` after `name(`, with `count(*)` (Cypher.g4:365). -/
inductive GCallTail (pe : PE) (fi : FnI) : PS → ET → PS → Prop
  | countStar {s} : fi.agg = true → fi.count = true → s.cur = .star → s.adv.cur = .rparen →
      GCallTail pe fi s (.node (.func fi.id true) [leaf (.int 1), leaf (.var nmPlaceholder)]) s.adv.adv
  | agg {s dis s1 args s2} : fi.agg = true →
      (dis = true ∧ s.cur = .kw .distinct ∧ s1 = s.adv ∨ dis = false ∧ s1 = s) → s1.cur ≠ .star →
      GArgsE pe s1 [] args s2 → fi.lo ≤ args.length → args.length ≤ fi.hi →
      (∀ a ∈ args, ¬ HasAgg a) →
      GCallTail pe fi s (.node (.func fi.id true)
        ((if dis then [ET.node .distinct args] else args) ++ [leaf (.var nmPlaceholder)])) s2
  | plain {s dis s1 args s2} : fi.agg = false →
      (dis = true ∧ s.cur = .kw .distinct ∧ s1 = s.adv ∨ dis = false ∧ s1 = s) →
      GArgsE pe s1 [] args s2 → fi.lo ≤ args.length → args.length ≤ fi.hi →
      (dis = true → args ≠ []) → GCallTail pe fi s (.node (.func fi.id false) args) s2

theorem callTail_sound (pe : PE) (F : Nat) (fi : FnI) (s : PS) t s' (h : callTail pe F fi s = .ok t s') :
    GCallTail pe fi s t s' := by
  unfold callTail at h
  peel h => dis s1 h1
  have gd : dis = true ∧ s.cur = .kw .distinct ∧ s1 = s.adv ∨ dis = false ∧ s1 = s := by
    cases dis
    · exact .inr ⟨rfl, (optK_eq h1).2 rfl⟩
    · obtain ⟨hk, e⟩ := (optK_eq h1).1 rfl; exact .inl ⟨rfl, hk, e⟩
  split at h
  · rename_i hagg
    peel h => st s2 h2
    cases st
    · rcases opt_ok h2 with ⟨h0, _⟩ | ⟨_, hns, e2⟩
      · cases h0
      subst e2
      simp only [Bool.false_eq_true, ite_false] at h
      peel h => args s3 h3
      split at h
      · simp at h
      · rename_i har
        split at h
        · simp at h
        · rename_i hna
          obtain ⟨rfl, rfl⟩ := pure_ok h
          have har' : fi.lo ≤ args.length ∧ args.length ≤ fi.hi := by simpa using har
          refine .agg hagg gd hns (argsP_sound pe _ _ _ _ _ h3) har'.1 har'.2 ?_
          intro a ha hh
          exact hna (List.any_eq_true.2 ⟨a, ha, by rw [find_aggregate_name_spec]; exact hh⟩)
    · obtain ⟨hs, rfl⟩ := (opt_eq h2).1 rfl
      simp only [ite_true] at h
      split at h
      · simp at h
      · rename_i hc
        split at h
        · simp at h
        · rename_i hd
          peel h => u s3 h3; obtain ⟨hr, rfl⟩ := tok_ok h3
          obtain ⟨rfl, rfl⟩ := pure_ok h
          have : dis = false := by simpa using hd
          subst this
          rcases gd with ⟨h0, _⟩ | ⟨_, rfl⟩
          · cases h0
          exact .countStar hagg (by simpa using hc) hs hr
  · rename_i hagg
    peel h => args s2 h2
    split at h
    · simp at h
    · rename_i har
      split at h
      · simp at h
      · rename_i hde
        obtain ⟨rfl, rfl⟩ := pure_ok h
        have har' : fi.lo ≤ args.length ∧ args.length ≤ fi.hi := by simpa using har
        refine .plain (by simpa using hagg) gd (argsP_sound pe _ _ _ _ _ h2) har'.1 har'.2 ?_
        intro hd hn; subst hn; simp_all

section
variable (o : Oracle) (fr : List Nat → Option FnI) (F : Nat) (allow forb : Bool)

/-- `oC_Atom` (Cypher.g4:361-374) as FalkorDB reads it. The flag is
`recurse` (a parenthesised expression or list literal to be finished by the
caller). -/
inductive GAtom : PS → ET × Bool → PS → Prop
  | case_ {s t s'} : s.cur = .kw .case_ → GCase (o.pe false) s.adv t s' → GAtom s (t, false) s'
  | null {s} : s.cur = .kw .null → GAtom s (leaf .null, false) s.adv
  | true_ {s} : s.cur = .kw .true_ → GAtom s (leaf (.bool true), false) s.adv
  | false_ {s} : s.cur = .kw .false_ → GAtom s (leaf (.bool false), false) s.adv
  | quant {s t s'} : isQuantKw s.cur = true → s.adv.cur = .lparen → GQuant (o.pe allow) s t s' →
      GAtom s (t, false) s'
  | quantVar {s v} : isQuantKw s.cur = true → s.adv.cur ≠ .lparen → identOf s.cur = some v →
      GAtom s (leaf (.var v), false) s.adv
  | var {s v name s1} : isQuantKw s.cur = false → identOf s.cur = some v →
      s.cur ∉ [.kw .case_, .kw .null, .kw .true_, .kw .false_] →
      GDotted s.adv [v] name s1 → s1.cur ≠ .lparen → GAtom s (leaf (.var v), false) s.adv
  | call {s v name s1 fi t s'} : isQuantKw s.cur = false → identOf s.cur = some v →
      s.cur ∉ [.kw .case_, .kw .null, .kw .true_, .kw .false_] →
      GDotted s.adv [v] name s1 → s1.cur = .lparen →
      name ≠ [nmReduce] → name ≠ [nmShortest] → name ≠ [nmAllShortest] → fr name = some fi →
      GCallTail (o.pe allow) fi s1.adv t s' → GAtom s (t, false) s'
  | reduce {s v s1 t s'} : isQuantKw s.cur = false → identOf s.cur = some v →
      s.cur ∉ [.kw .case_, .kw .null, .kw .true_, .kw .false_] →
      GDotted s.adv [v] [nmReduce] s1 → s1.cur = .lparen → GReduce (o.pe allow) s1.adv t s' →
      GAtom s (t, false) s'
  | shortest {s v s1 t s'} : isQuantKw s.cur = false → identOf s.cur = some v →
      s.cur ∉ [.kw .case_, .kw .null, .kw .true_, .kw .false_] →
      GDotted s.adv [v] [nmShortest] s1 → s1.cur = .lparen → GShortest s1.adv t s' →
      GAtom s (t, false) s'
  | param {s p} : s.cur = .param p → GAtom s (leaf (.param p), false) s.adv
  | int {s i} : s.cur = .int i → GAtom s (leaf (.int i), false) s.adv
  | float {s} : s.cur = .float → GAtom s (leaf (.other 0), false) s.adv
  | str {s t} : s.cur = .str t → GAtom s (leaf (.str t), false) s.adv
  | bracket {s r s'} : s.cur = .lbrack → GBracket o F forb s.adv r s' → GAtom s r s'
  | map {s m s'} : s.cur = .lbrace → o.pmap s = .ok m s' → GAtom s (m, false) s'
  | pattern {s g s'} : allow = true → s.cur = .lparen →
      GPat o Ext.rust GExt.rust .match_ (QG.empty, []) s g s' → g.rels ≠ [] →
      GAtom s (leaf .pattern, false) s'
  | paren {s} : s.cur = .lparen → GAtom s (leaf .paren, true) s.adv

theorem identArm_sound (s : PS) r s' (hq : isQuantKw s.cur = false)
    (hn : s.cur ∉ [.kw .case_, .kw .null, .kw .true_, .kw .false_])
    (h : identArm o fr F allow s = .ok r s') : GAtom o fr F allow forb s r s' := by
  unfold identArm at h
  peel h => s0 s1 h1
  simp only [run_getS, R.ok.injEq] at h1; obtain ⟨rfl, rfl⟩ := h1
  peel h => v s2 h2; obtain ⟨hv, rfl⟩ := ident_ok h2
  peel h => name s3 h3
  have gd := dottedP_sound _ _ _ _ _ h3
  peel h => lp s4 h4
  cases lp
  · rcases opt_ok h4 with ⟨h0, _⟩ | ⟨_, hnl, e4⟩
    · cases h0
    subst e4
    simp only [Bool.false_eq_true, ite_false] at h
    peel h => u s5 h5
    simp only [run_setS, R.ok.injEq] at h5; obtain ⟨-, rfl⟩ := h5
    peel h => v' s6 h6; obtain ⟨hv', rfl⟩ := ident_ok h6
    obtain ⟨rfl, rfl⟩ := pure_ok h
    rw [hv] at hv'; cases hv'
    exact .var hq hv hn gd hnl
  · obtain ⟨hl, rfl⟩ := (opt_eq h4).1 rfl
    simp only [ite_true] at h
    split at h
    · rename_i hr; subst hr
      peel h => t s5 h5
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact .reduce hq hv hn gd hl (reduceP_sound _ _ _ _ h5)
    · rename_i hr
      split at h
      · rename_i hs; subst hs
        peel h => t s5 h5
        obtain ⟨rfl, rfl⟩ := pure_ok h
        exact .shortest hq hv hn gd hl (shortestP_sound F _ _ _ h5)
      · rename_i hs
        split at h
        · simp at h
        · rename_i ha
          generalize hf : fr name = fo at h
          cases fo with
          | none => simp at h
          | some fi =>
            dsimp only at h
            peel h => t s5 h5
            obtain ⟨rfl, rfl⟩ := pure_ok h
            exact .call hq hv hn gd hl hr hs ha hf (callTail_sound _ F fi _ _ _ h5)

/-- **parse_primary_expr is sound** over the whole token set: every accepted
primary is an `oC_Atom` alternative (literals, parameter, CASE, quantifier,
function call incl. `count(*)` / DISTINCT, `reduce`, `shortestPath`,
list / pattern comprehension, list literal, map, pattern predicate,
parenthesised expression, variable), and the tree is that alternative's. -/
theorem primaryC_sound (s : PS) r s' (h : primaryC o fr F allow forb s = .ok r s') :
    GAtom o fr F allow forb s r s' := by
  unfold primaryC at h
  peel h => c s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  generalize hcur : s.cur = c at h
  cases c
  case kw k =>
    cases k <;> (try dsimp only at h)
    case case_ =>
      peel h => t s2 h2
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact .case_ hcur (caseP_sound _ F _ _ _ h2)
    case null => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .null hcur
    case true_ => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .true_ hcur
    case false_ => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .false_ hcur
    all_goals
      split at h
      · rename_i hq
        unfold quantArm at h
        peel h => s0 s2 h2
        simp only [run_getS, R.ok.injEq] at h2; obtain ⟨rfl, rfl⟩ := h2
        peel h => u s3 h3; have := next_ok h3; subst this
        peel h => c2 s4 h4
        obtain ⟨hc0, hs⟩ := peek_ok h4; subst hc0; subst s4
        peel h => u s5 h5
        simp only [run_setS, R.ok.injEq] at h5; obtain ⟨-, rfl⟩ := h5
        split at h
        · rename_i hl
          peel h => t s6 h6
          obtain ⟨rfl, rfl⟩ := pure_ok h
          exact .quant (by rw [hcur]; exact hq) hl (quantP_sound _ _ _ _ h6)
        · rename_i hl
          peel h => v s6 h6; obtain ⟨hv, rfl⟩ := ident_ok h6
          obtain ⟨rfl, rfl⟩ := pure_ok h
          exact .quantVar (by rw [hcur]; exact hq) hl hv
      · rename_i hq
        exact identArm_sound o fr F allow forb s r s' (by rw [hcur]; simpa using hq)
          (by rw [hcur]; simp) h
  case ident v =>
    exact identArm_sound o fr F allow forb s r s' (by rw [hcur]; rfl) (by rw [hcur]; simp) h
  case param p => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .param hcur
  case int i => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .int hcur
  case float => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .float hcur
  case str t => peel h => u s2 h2; have := next_ok h2; subst this; obtain ⟨rfl, rfl⟩ := pure_ok h; exact .str hcur
  case lbrack =>
    peel h => u s2 h2; have := next_ok h2; subst this
    exact .bracket hcur (listLitP_sound o F forb _ _ _ h)
  case lbrace =>
    peel h => m s2 h2
    obtain ⟨rfl, rfl⟩ := pure_ok h
    exact .map hcur h2
  case lparen =>
    unfold parenArm at h
    peel h => s0 s2 h2
    simp only [run_getS, R.ok.injEq] at h2; obtain ⟨rfl, rfl⟩ := h2
    peel h => pat s3 h3
    split at h
    · rename_i hp
      obtain ⟨rfl, rfl⟩ := pure_ok h
      cases allow
      · simp only [Bool.false_eq_true, ite_false] at h3
        obtain ⟨rfl, rfl⟩ := pure_ok h3; simp [isPatG] at hp
      · simp only [ite_true] at h3
        rcases attempt_ok h3 with ⟨g, rfl, hg⟩ | ⟨rfl, _⟩
        · exact .pattern rfl hcur (patternP_sound o F _ _ _ _ hg) (by simpa [isPatG] using hp)
        · simp [isPatG] at hp
    · peel h => u s4 h4
      simp only [run_setS, R.ok.injEq] at h4; obtain ⟨-, rfl⟩ := h4
      peel h => u s5 h5; have := next_ok h5; subst this
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact .paren hcur
  all_goals simp at h

end

end FalkorParserGrammar.C
