/-
# `oC_Pattern` (Cypher.g4:214-231) and soundness of `parse_pattern`
-/
import FalkorParserGrammar.C.PatternG

namespace FalkorParserGrammar.C

/-- Extensions specific to the pattern *graph* (as opposed to its syntax). -/
structure GExt where
  /-- A repeated relationship variable inside one pattern is silently
  dropped instead of rejected: MERGE, and CREATE in a named path
  (cypher.rs:1514-1522 checks MATCH only). -/
  dupDrop : Bool

def GExt.g4 : GExt := ⟨false⟩
def GExt.rust : GExt := ⟨true⟩

/-- Adding a relationship: fresh, or (with `dupDrop`) silently skipped. -/
def RelOk (y : GExt) (g : QG Nat) (r : QRel Nat) : Prop := (g.addRel r).1 = true ∨ y.dupDrop = true

/-- `( SP? oC_PatternElementChain )*` of an anonymous path. -/
inductive GChainA (o : Oracle) (x : Ext) (y : GExt) (cl : CK) :
    QNode Nat → QG Nat × List Nat → PS → QG Nat × List Nat → PS → Prop
  | nil {left gs s} : GChainA o x y cl left gs s gs s
  | cons {left gs s rr s1 out s'} : GRel o x left s rr s1 → RelOk y gs.1 rr.1 →
      GChainA o x y cl rr.2 (addPatNode cl ((gs.1.addRel rr.1).2, gs.2) rr.2) s1 out s' →
      GChainA o x y cl left gs s out s'

/-- The same for a named path `p = ...`: the path's variables are collected
and the path is added at the end. -/
inductive GChainN (o : Oracle) (x : Ext) (y : GExt) (cl : CK) (p : Nat) :
    QNode Nat → List Nat → QG Nat × List Nat → PS → QG Nat × List Nat → PS → Prop
  | nil {left vars gs s} : GChainN o x y cl p left vars gs s ((gs.1.addPath ⟨p, vars⟩).2, gs.2) s
  | cons {left vars gs s rr s1 out s'} : GRel o x left s rr s1 → RelOk y gs.1 rr.1 →
      GChainN o x y cl p rr.2 (vars ++ [rr.1.alias, rr.2.alias])
        (addPatNode cl ((gs.1.addRel rr.1).2, gs.2) rr.2) s1 out s' →
      GChainN o x y cl p left vars gs s out s'

theorem relOk_of {y : GExt} {cl : CK} {g : QG Nat} {r : QRel Nat}
    (h : ¬((!(g.addRel r).1 && (cl = .match_ || cl = .create)) = true)) (hc : cl = .match_ ∨ cl = .create ∨ y.dupDrop = true) :
    RelOk y g r := by
  unfold RelOk
  cases e : (g.addRel r).1
  · rcases hc with rfl | rfl | hd
    · simp [e] at h
    · simp [e] at h
    · exact .inr hd
  · exact .inl rfl

theorem chainA_sound (o : Oracle) (F : Nat) (cl : CK) : ∀ (n : Nat) left gs (s : PS) out s',
    chainA o F cl n left gs s = .ok out s' → GChainA o Ext.rust GExt.rust cl left gs s out s'
  | 0, _, _, _, _, _, h => by simp [chainA] at h
  | n + 1, left, gs, s, out, s', h => by
    unfold chainA at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => rr s2 h2
      split at h
      · simp at h
      · rename_i hd
        exact .cons (relP_sound o F cl left _ _ _ h2) (relOk_of hd (.inr (.inr rfl)))
          (chainA_sound o F cl n _ _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil

theorem chainN_sound (o : Oracle) (F : Nat) (cl : CK) (p : Nat) :
    ∀ (n : Nat) left vars gs (s : PS) out s',
    chainN o F cl p n left vars gs s = .ok out s' → GChainN o Ext.rust GExt.rust cl p left vars gs s out s'
  | 0, _, _, _, _, _, _, h => by simp [chainN] at h
  | n + 1, left, vars, gs, s, out, s', h => by
    unfold chainN at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => rr s2 h2
      split at h
      · simp at h
      · rename_i hd
        refine .cons (relP_sound o F cl left _ _ _ h2) ?_ (chainN_sound o F cl p n _ _ _ _ _ _ h)
        exact .inr rfl
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil

/-- The allShortestPaths relationship loop (extension `asp`). -/
inductive GAsp (o : Oracle) (x : Ext) (y : GExt) (cl : CK) (la : Nat) :
    QNode Nat → Bool → List Nat → QG Nat × List Nat → PS →
      Bool × List Nat × (QG Nat × List Nat) → PS → Prop
  | nil {prev found vars gs s} : GAsp o x y cl la prev found vars gs s (found, vars, gs) s
  | varLen {prev found vars gs s rr s1 out s'} : GRel o x prev s rr s1 →
      (rr.1.minH.isSome || rr.1.maxH.isSome) = true → found = false →
      minTooBig rr.1 = false →
      RelOk y gs.1 (aspRel rr.1 la) →
      GAsp o x y cl la rr.2 true (vars ++ [(aspRel rr.1 la).alias, rr.2.alias])
        (addPatNode cl ((gs.1.addRel (aspRel rr.1 la)).2, gs.2) rr.2) s1 out s' →
      GAsp o x y cl la prev found vars gs s out s'
  | fixed {prev found vars gs s rr s1 out s'} : GRel o x prev s rr s1 →
      (rr.1.minH.isSome || rr.1.maxH.isSome) = false → RelOk y gs.1 rr.1 →
      GAsp o x y cl la rr.2 found (vars ++ [rr.1.alias, rr.2.alias])
        (addPatNode cl ((gs.1.addRel rr.1).2, gs.2) rr.2) s1 out s' →
      GAsp o x y cl la prev found vars gs s out s'

theorem aspLoop_sound (o : Oracle) (F : Nat) (cl : CK) (la : Nat) :
    ∀ (n : Nat) prev found vars gs (s : PS) out s',
    aspLoop o F cl la n prev found vars gs s = .ok out s' →
      GAsp o Ext.rust GExt.rust cl la prev found vars gs s out s'
  | 0, _, _, _, _, _, _, _, h => by simp [aspLoop] at h
  | n + 1, prev, found, vars, gs, s, out, s', h => by
    unfold aspLoop at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => rr s2 h2
      have gr := relP_sound o F cl prev _ _ _ h2
      split at h
      · rename_i hv
        split at h
        · simp at h
        · rename_i hf
          split at h
          · simp at h
          · rename_i hm
            split at h
            · simp at h
            · exact .varLen gr hv (by simpa using hf) (by simpa using hm) (.inr rfl)
                (aspLoop_sound o F cl la n _ _ _ _ _ _ _ h)
      · rename_i hv
        split at h
        · simp at h
        · exact .fixed gr (by simpa using hv) (.inr rfl) (aspLoop_sound o F cl la n _ _ _ _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil

/-- One `oC_PatternPart` (Cypher.g4:217-228), with how the Rust loop goes on. -/
inductive GPart (o : Oracle) (x : Ext) (y : GExt) (cl : CK) :
    QG Nat × List Nat → PS → PEnd × (QG Nat × List Nat) → PS → Prop
  | anon {gs s left s1 out s'} : GNode o s left s1 →
      GChainA o x y cl left (addPatNode cl gs left) s1 out s' → GPart o x y cl gs s (.sep, out) s'
  | named {gs s p left s1 out s'} : identOf s.cur = some p → s.adv.cur = .eq →
      GNode o s.adv.adv left s1 →
      GChainN o x y cl p left [left.alias] (addPatNode cl gs left) s1 out s' →
      GPart o x y cl gs s (.sep, out) s'
  | asp {gs s p left s1 res s2 e} : x.asp = true → identOf s.cur = some p → s.adv.cur = .eq →
      identOf s.adv.adv.cur = some nmAllShortest → s.adv.adv.adv.cur = .lparen →
      GNode o s.adv.adv.adv.adv left s1 → (s1.cur = .dash ∨ s1.cur = .lt) →
      GAsp o x y cl left.alias left false [left.alias] (addPatNode cl gs left) s1 res s2 →
      res.1 = true → s2.cur = .rparen →
      (e = PEnd.cont ∧ s2.adv.cur = .comma ∨ e = .stop) →
      GPart o x y cl gs s (e, ((res.2.2.1.addPath ⟨p, res.2.1⟩).2, res.2.2.2))
        (if e = .cont then s2.adv.adv else s2.adv)

theorem partP_sound (o : Oracle) (F : Nat) (cl : CK) (gs : QG Nat × List Nat) (s : PS) r s'
    (h : partP o F cl gs s = .ok r s') : GPart o Ext.rust GExt.rust cl gs s r s' := by
  unfold partP at h
  peel h => a s1 h1
  rcases tryIdent_ok h1 with ⟨p, rfl, hp, rfl⟩ | ⟨rfl, hp, rfl⟩
  · dsimp only at h
    peel h => u s2 h2
    obtain ⟨he, rfl⟩ := tok_ok h2
    peel h => c s3 h3
    obtain ⟨hc0, hs⟩ := peek_ok h3; subst hc0; subst s3
    split at h
    · simp at h
    · split at h
      · rename_i _ hasp
        unfold aspPart at h
        peel h => u s4 h4
        have := next_ok h4; subst this
        peel h => u s5 h5
        obtain ⟨hl, rfl⟩ := tok_ok h5
        peel h => left s6 h6
        peel h => c s7 h7
        obtain ⟨hc0, hs⟩ := peek_ok h7; subst hc0; subst s7
        split at h
        · simp at h
        · rename_i hdash
          peel h => res s8 h8
          split at h
          · simp at h
          · rename_i hres
            peel h => u s9 h9
            obtain ⟨hr, rfl⟩ := tok_ok h9
            peel h => c s10 h10
            obtain ⟨hc0, hs⟩ := peek_ok h10; subst hc0; subst s10
            have ga := aspLoop_sound o F cl left.alias F left false [left.alias] _ _ _ _ h8
            have gn := nodeP_sound o F _ _ _ h6
            have hd : s6.cur = .dash ∨ s6.cur = .lt := by revert hdash; cases s6.cur <;> simp
            have hres' : res.1 = true := by simpa using hres
            split at h
            · rename_i hcm
              peel h => u s11 h11
              have := next_ok h11; subst this
              obtain ⟨rfl, rfl⟩ := pure_ok h
              exact .asp rfl hp he hasp hl gn hd ga hres' hr (.inl ⟨rfl, hcm⟩)
            · obtain ⟨rfl, rfl⟩ := pure_ok h
              exact .asp rfl hp he hasp hl gn hd ga hres' hr (.inr rfl)
      · peel h => left s4 h4
        peel h => out s5 h5
        obtain ⟨rfl, rfl⟩ := pure_ok h
        exact .named hp he (nodeP_sound o F _ _ _ h4) (chainN_sound o F cl p F _ _ _ _ _ _ h5)
  · dsimp only at h
    peel h => left s2 h2
    peel h => out s3 h3
    obtain ⟨rfl, rfl⟩ := pure_ok h
    exact .anon (nodeP_sound o F _ _ _ h2) (chainA_sound o F cl F _ _ _ _ _ h3)

/-- `oC_Pattern` (Cypher.g4:214-215): parts separated by `,`; MERGE takes
one part (`oC_Merge` has an `oC_PatternPart`); with `fold`, the clause's
own keyword also separates parts. -/
inductive GPat (o : Oracle) (x : Ext) (y : GExt) (cl : CK) :
    QG Nat × List Nat → PS → QG Nat → PS → Prop
  | last {gs s e gs1 s1} : GPart o x y cl gs s (e, gs1) s1 → e ≠ .cont → GPat o x y cl gs s gs1.1 s1
  | comma {gs s gs1 s1 out s'} : GPart o x y cl gs s (.sep, gs1) s1 → cl ≠ .merge →
      s1.cur = .comma → GPat o x y cl gs1 s1.adv out s' → GPat o x y cl gs s out s'
  | kw {gs s gs1 s1 out s'} : x.fold = true → GPart o x y cl gs s (.sep, gs1) s1 → cl ≠ .merge →
      s1.cur = .kw cl → GPat o x y cl gs1 s1.adv out s' → GPat o x y cl gs s out s'
  | cont {gs s gs1 s1 out s'} : GPart o x y cl gs s (.cont, gs1) s1 →
      GPat o x y cl gs1 s1 out s' → GPat o x y cl gs s out s'

theorem patternLoop_sound (o : Oracle) (F : Nat) (cl : CK) : ∀ (n : Nat) gs (s : PS) out s',
    patternLoop o F cl n gs s = .ok out s' → GPat o Ext.rust GExt.rust cl gs s out s'
  | 0, _, _, _, _, h => by simp [patternLoop] at h
  | n + 1, gs, s, out, s', h => by
    unfold patternLoop at h
    peel h => r s1 h1
    have gp := partP_sound o F cl gs s r s1 h1
    obtain ⟨e, gs1⟩ := r
    cases e <;> dsimp only at h
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .last gp (by simp)
    · exact .cont gp (patternLoop_sound o F cl n _ _ _ _ h)
    · peel h => more s2 h2
      unfold sepP at h2
      split at h2
      · obtain ⟨rfl, rfl⟩ := pure_ok h2
        simp only [Bool.false_eq_true, ite_false] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact .last gp (by simp)
      · rename_i hm
        peel h2 => c s3 h3
        obtain ⟨hc0, hs⟩ := peek_ok h3; subst hc0; subst s3
        generalize hc : s1.cur = c at h2
        cases c <;> dsimp only at h2
        case comma =>
          peel h2 => u s4 h4
          have := next_ok h4; subst this
          obtain ⟨rfl, rfl⟩ := pure_ok h2
          simp only [ite_true] at h
          exact .comma gp hm hc (patternLoop_sound o F cl n _ _ _ _ h)
        case kw k =>
          split at h2
          · rename_i hk; subst k
            peel h2 => u s4 h4
            have := next_ok h4; subst this
            obtain ⟨rfl, rfl⟩ := pure_ok h2
            simp only [ite_true] at h
            exact .kw rfl gp hm hc (patternLoop_sound o F cl n _ _ _ _ h)
          · obtain ⟨rfl, rfl⟩ := pure_ok h2
            simp only [Bool.false_eq_true, ite_false] at h
            obtain ⟨rfl, rfl⟩ := pure_ok h; exact .last gp (by simp)
        all_goals (obtain ⟨rfl, rfl⟩ := pure_ok h2
                   simp only [Bool.false_eq_true, ite_false] at h
                   obtain ⟨rfl, rfl⟩ := pure_ok h; exact .last gp (by simp))

/-- **parse_pattern is sound**: an accepted MATCH / CREATE / MERGE pattern
is an `oC_Pattern` (one `oC_PatternPart` for MERGE) under the extensions
`[:A:B]`, keyword folding, `allShortestPaths` and silently dropped
duplicate relationship variables, and the query graph is the one the
rules build. -/
theorem patternP_sound (o : Oracle) (F : Nat) (cl : CK) (s : PS) g s'
    (h : patternP o F cl s = .ok g s') : GPat o Ext.rust GExt.rust cl (QG.empty, []) s g s' :=
  patternLoop_sound o F cl F _ _ _ _ h

end FalkorParserGrammar.C
