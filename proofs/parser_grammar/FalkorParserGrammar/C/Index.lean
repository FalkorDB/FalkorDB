/-
# Index DDL and the entry point: `parse_index_ops`, `match_dot_property_separator`,
# `parse` (cypher.rs:535-753, 2623-2640)
-/
import FalkorParserGrammar.C.ClausesG3
import FalkorParserGrammar.C.ClausesNP

namespace FalkorParserGrammar.C

/-- `current_str().eq_ignore_ascii_case("constraint")` (cypher.rs:556, 657). -/
def nmConstraint : Nat := 6000001

/-- `match_dot_property_separator` (cypher.rs:2623-2640): a `.`; anything
else (including a float token `.1`) is an error. -/
def dotSepP : P Unit := tok .dot

/-- The entity of `FOR (...)` after `FOR (` (cypher.rs:584-608 / 683-707):
`(n:L)` or `()-[e:R]-()` / `()-[e:R]->()`. -/
def idxEntity : P (Nat × Nat × Bool) := do
  let rp ← opt .rparen
  if rp then do
    tok .dash
    tok .lbrack
    let nkey ← ident
    tok .colon
    let label ← ident
    tok .rbrack
    tok .dash
    let _ ← opt .gt
    tok .lparen
    tok .rparen
    pure (nkey, label, true)
  else do
    let nkey ← ident
    let c ← peek
    if c ≠ .colon then fail else do
    next
    let label ← ident
    tok .rparen
    pure (nkey, label, false)

/-- `, n.p` repeated (cypher.rs:617-624). -/
def idxMore (nkey : Nat) : Nat → List Nat → P (List Nat)
  | 0, _ => fuelP
  | n + 1, acc => do
    let cm ← opt .comma
    if cm then do
      let key ← ident
      dotSepP
      if nkey ≠ key then fail else do
      let a ← propName
      idxMore nkey n (acc ++ [a])
    else pure acc

/-- `ident (, ident)*` of the old syntax `ON :L(a, b)` (cypher.rs:568-571). -/
def identList : Nat → List Nat → P (List Nat)
  | 0, _ => fuelP
  | n + 1, acc => do
    let cm ← opt .comma
    if cm then do let a ← ident; identList n (acc ++ [a]) else pure acc

/-- The body of CREATE/DROP after the keyword (cypher.rs:552-648 / 653-749);
`create` selects the OPTIONS part. `none` = "not an index command". -/
def idxBody (o : Oracle) (F : Nat) (create : Bool) : P (Option IR) := do
  let ft ← optK .fulltext
  let vec ← (if ft then pure false else optK .vector)
  let idx ← optK .index
  if !idx then do
    let c ← peek
    if identOf c = some nmConstraint then fail else pure none
  else do
    let on ← (if !ft && !vec then optK .on else pure false)
    if on then do
      tok .colon
      let label ← ident
      tok .lparen
      let a ← ident
      let attrs ← identList F [a]
      tok .rparen
      pure (some (if create then .createIndex label attrs .range false none
                  else .dropIndex label attrs .range false))
    else do
      tokK .for_
      tok .lparen
      let ent ← idxEntity
      tokK .on
      tok .lparen
      let key ← ident
      dotSepP
      if ent.1 ≠ key then fail else do
      let a ← propName
      let attrs ← idxMore ent.1 F [a]
      tok .rparen
      let ity : IdxTy := if ft then .fulltext else if vec then .vector else .range
      if create then do
        let op ← (if vec || ft then optK .options else pure false)
        let opts ← (if op then do let m ← o.pmap; pure (some m) else pure none)
        pure (some (.createIndex ent.2.1 attrs ity ent.2.2 opts))
      else pure (some (.dropIndex ent.2.1 attrs ity ent.2.2))

/-- `parse_index_ops` (cypher.rs:551-753). -/
def indexOps (o : Oracle) (F : Nat) : P (Option IR) := do
  let cr ← optK .create
  if cr then idxBody o F true
  else do
    let dr ← optK .drop
    if dr then idxBody o F false else pure none

/-- `Parser::parse` (cypher.rs:535-547): index DDL, or (after restoring the
saved state) a query; then `expect_end_of_input`. -/
def parseP (o : Oracle) (F n : Nat) : P IR := do
  let s0 ← getS
  let r ← indexOps o F
  match r with
  | some ir => do endP n; pure ir
  | none => do
    setS s0
    let q ← queryP o F n
    endP n
    pure q

/-! ## No panics -/

section
variable (o : Oracle) (ho : o.Safe)
include ho

theorem np_idxEntity : NP idxEntity := by unfold idxEntity; np_go
theorem np_idxMore (k : Nat) : ∀ n acc, NP (idxMore k n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_idxMore k n; unfold idxMore dotSepP propName; np_go
theorem np_identList : ∀ n acc, NP (identList n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_identList n; unfold identList; np_go
theorem np_idxBody (F : Nat) (c : Bool) : NP (idxBody o F c) := by
  have := np_idxEntity o ho; have := np_idxMore o ho; have := np_identList o ho
  have := np_pmap o ho
  unfold idxBody dotSepP propName; np_go
theorem np_indexOps (F : Nat) : NP (indexOps o F) := by
  have := np_idxBody o ho F; unfold indexOps; np_go

/-- **`Parser::parse` never panics** (for an expression parser that does
not): index DDL, every clause, and the end-of-input check. -/
theorem parse_no_panic (F n : Nat) : NP (parseP o F n) := by
  have := np_indexOps o ho F; have := queryP_no_panic o ho F n; have := np_endP o ho
  have : ∀ s, NP (setS s) := fun _ => ⟨fun _ => by simp⟩
  unfold parseP; np_go

end

/-! ## Grammar (FalkorDB's index syntax; not part of Cypher.g4) and soundness -/

section
variable (o : Oracle)

inductive GIdxEntity : PS → Nat × Nat × Bool → PS → Prop
  | rel {s k l s'} : s.cur = .rparen → s.adv.cur = .dash → s.adv.adv.cur = .lbrack →
      identOf s.adv.adv.adv.cur = some k → s.adv.adv.adv.adv.cur = .colon →
      identOf s.adv.adv.adv.adv.adv.cur = some l → s.adv.adv.adv.adv.adv.adv.cur = .rbrack →
      s.adv.adv.adv.adv.adv.adv.adv.cur = .dash →
      (let t := s.adv.adv.adv.adv.adv.adv.adv.adv
       (t.cur = .gt ∧ t.adv.cur = .lparen ∧ t.adv.adv.cur = .rparen ∧ s' = t.adv.adv.adv) ∨
       (t.cur = .lparen ∧ t.adv.cur = .rparen ∧ s' = t.adv.adv)) →
      GIdxEntity s (k, l, true) s'
  | node {s k l} : identOf s.cur = some k → s.adv.cur = .colon → identOf s.adv.adv.cur = some l →
      s.adv.adv.adv.cur = .rparen → GIdxEntity s (k, l, false) s.adv.adv.adv.adv

inductive GIdxMore (k : Nat) : PS → List Nat → List Nat → PS → Prop
  | nil {s acc} : GIdxMore k s acc acc s
  | cons {s acc a out s'} : s.cur = .comma → identOf s.adv.cur = some k → s.adv.adv.cur = .dot →
      identOf s.adv.adv.adv.cur = some a → GIdxMore k s.adv.adv.adv.adv (acc ++ [a]) out s' →
      GIdxMore k s acc out s'

inductive GIdentList : PS → List Nat → List Nat → PS → Prop
  | nil {s acc} : GIdentList s acc acc s
  | cons {s acc a out s'} : s.cur = .comma → identOf s.adv.cur = some a →
      GIdentList s.adv.adv (acc ++ [a]) out s' → GIdentList s acc out s'

/-- An index command after CREATE / DROP. -/
inductive GIdx (create : Bool) : PS → IR → PS → Prop
  | old {s l a attrs s1} :
      s.cur = .kw .index → s.adv.cur = .kw .on → s.adv.adv.cur = .colon →
      identOf s.adv.adv.adv.cur = some l → s.adv.adv.adv.adv.cur = .lparen →
      identOf s.adv.adv.adv.adv.adv.cur = some a →
      GIdentList s.adv.adv.adv.adv.adv.adv [a] attrs s1 → s1.cur = .rparen →
      GIdx create s (if create then .createIndex l attrs .range false none
        else .dropIndex l attrs .range false) s1.adv
  | new {s ity s0 ent s1 a attrs s2 opts s'} :
      (ity = IdxTy.fulltext ∧ s.cur = .kw .fulltext ∧ s0 = s.adv ∨
       ity = .vector ∧ s.cur = .kw .vector ∧ s0 = s.adv ∨ ity = .range ∧ s0 = s) →
      s0.cur = .kw .index → s0.adv.cur = .kw .for_ → s0.adv.adv.cur = .lparen →
      GIdxEntity s0.adv.adv.adv ent s1 → s1.cur = .kw .on → s1.adv.cur = .lparen →
      identOf s1.adv.adv.cur = some ent.1 → s1.adv.adv.adv.cur = .dot →
      identOf s1.adv.adv.adv.adv.cur = some a →
      GIdxMore ent.1 s1.adv.adv.adv.adv.adv [a] attrs s2 → s2.cur = .rparen →
      (opts = none ∧ s' = s2.adv ∨
        create = true ∧ ity ≠ .range ∧ s2.adv.cur = .kw .options ∧ ∃ m, opts = some m ∧
          o.pmap s2.adv.adv = .ok m s') →
      GIdx create s (if create then .createIndex ent.2.1 attrs ity ent.2.2 opts
        else .dropIndex ent.2.1 attrs ity ent.2.2) s'

theorem idxEntity_sound (s : PS) e s' (h : idxEntity s = .ok e s') : GIdxEntity s e s' := by
  unfold idxEntity at h
  peel h => rp s1 h1
  cases rp
  · have := (opt_eq h1).2 rfl; subst s1
    simp only [Bool.false_eq_true, ite_false] at h
    peel h => k s2 h2; obtain ⟨hk, rfl⟩ := ident_ok h2
    peel h => c s3 h3; obtain ⟨hc0, hs⟩ := peek_ok h3; subst hc0; subst s3
    split at h
    · simp at h
    · rename_i hcol
      peel h => u s4 h4; have := next_ok h4; subst this
      peel h => l s5 h5; obtain ⟨hl, rfl⟩ := ident_ok h5
      peel h => u s6 h6; obtain ⟨hr, rfl⟩ := tok_ok h6
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact .node hk (by simpa using hcol) hl hr
  · obtain ⟨hr, rfl⟩ := (opt_eq h1).1 rfl
    simp only [ite_true] at h
    peel h => u s2 h2; obtain ⟨h2', rfl⟩ := tok_ok h2
    peel h => u s3 h3; obtain ⟨h3', rfl⟩ := tok_ok h3
    peel h => k s4 h4; obtain ⟨hk, rfl⟩ := ident_ok h4
    peel h => u s5 h5; obtain ⟨h5', rfl⟩ := tok_ok h5
    peel h => l s6 h6; obtain ⟨hl, rfl⟩ := ident_ok h6
    peel h => u s7 h7; obtain ⟨h7', rfl⟩ := tok_ok h7
    peel h => u s8 h8; obtain ⟨h8', rfl⟩ := tok_ok h8
    peel h => g s9 h9
    peel h => u s10 h10
    peel h => u s11 h11
    obtain ⟨rfl, rfl⟩ := pure_ok h
    refine .rel hr h2' h3' hk h5' hl h7' h8' ?_
    cases g
    · have := (opt_eq h9).2 rfl; subst s9
      obtain ⟨a, rfl⟩ := tok_ok h10; obtain ⟨b, rfl⟩ := tok_ok h11
      exact .inr ⟨a, b, rfl⟩
    · obtain ⟨hg, rfl⟩ := (opt_eq h9).1 rfl
      obtain ⟨a, rfl⟩ := tok_ok h10; obtain ⟨b, rfl⟩ := tok_ok h11
      exact .inl ⟨hg, a, b, rfl⟩

theorem idxMore_sound (k : Nat) : ∀ n acc (s : PS) out s', idxMore k n acc s = .ok out s' →
    GIdxMore k s acc out s'
  | 0, _, _, _, _, h => by simp [idxMore] at h
  | n + 1, acc, s, out, s', h => by
    unfold idxMore at h
    peel h => cm s1 h1
    cases cm
    · have := (opt_eq h1).2 rfl; subst s1
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil
    · obtain ⟨hc, rfl⟩ := (opt_eq h1).1 rfl
      simp only [ite_true] at h
      peel h => key s2 h2; obtain ⟨hk, rfl⟩ := ident_ok h2
      peel h => u s3 h3; obtain ⟨hd, rfl⟩ := tok_ok h3
      split at h
      · simp at h
      · rename_i hne
        have : k = key := by simpa using hne
        subst this
        peel h => a s4 h4; obtain ⟨ha, rfl⟩ := ident_ok h4
        exact .cons hc hk hd ha (idxMore_sound k n _ _ _ _ h)

theorem identList_sound : ∀ n acc (s : PS) out s', identList n acc s = .ok out s' →
    GIdentList s acc out s'
  | 0, _, _, _, _, h => by simp [identList] at h
  | n + 1, acc, s, out, s', h => by
    unfold identList at h
    peel h => cm s1 h1
    cases cm
    · have := (opt_eq h1).2 rfl; subst s1
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil
    · obtain ⟨hc, rfl⟩ := (opt_eq h1).1 rfl
      simp only [ite_true] at h
      peel h => a s2 h2; obtain ⟨ha, rfl⟩ := ident_ok h2
      exact .cons hc ha (identList_sound n _ _ _ _ h)

/-- **parse_index_ops is sound**: an accepted index command matches the
documented CREATE/DROP INDEX syntax and builds that command. -/
theorem idxBody_sound (F : Nat) (c : Bool) (s : PS) r s' (h : idxBody o F c s = .ok (some r) s') :
    GIdx o c s r s' := by
  unfold idxBody at h
  peel h => ft s1 h1
  peel h => vec s2 h2
  peel h => idx s3 h3
  cases idx
  · simp only [Bool.not_false, ite_true] at h
    peel h => cc s4 h4
    split at h <;> simp at h
  · obtain ⟨hidx, rfl⟩ := (optK_eq h3).1 rfl
    simp only [Bool.not_true, Bool.false_eq_true, ite_false] at h
    peel h => on s4 h4
    -- which index type, and where INDEX sits
    have gty : ∀ ity, ity = (if ft then IdxTy.fulltext else if vec then .vector else .range) →
        (ity = .fulltext ∧ s.cur = .kw .fulltext ∧ s2 = s.adv ∨
         ity = .vector ∧ s.cur = .kw .vector ∧ s2 = s.adv ∨ ity = .range ∧ s2 = s) := by
      intro ity hi; subst hi
      cases ft
      · have := (optK_eq h1).2 rfl; subst s1
        simp only [Bool.false_eq_true, ite_false] at h2 ⊢
        cases vec
        · have := (optK_eq h2).2 rfl; subst s2; simp
        · obtain ⟨hv, rfl⟩ := (optK_eq h2).1 rfl; simp [hv]
      · obtain ⟨hf, rfl⟩ := (optK_eq h1).1 rfl
        simp only [ite_true] at h2 ⊢
        obtain ⟨rfl, rfl⟩ := pure_ok h2; simp [hf]
    cases on
    · simp only [Bool.false_eq_true, ite_false] at h
      peel h => u s5 h5; obtain ⟨hfor, rfl⟩ := tok_ok h5
      peel h => u s6 h6; obtain ⟨hlp, rfl⟩ := tok_ok h6
      peel h => ent s7 h7
      peel h => u s8 h8; obtain ⟨hon, rfl⟩ := tok_ok h8
      peel h => u s9 h9; obtain ⟨hlp2, rfl⟩ := tok_ok h9
      peel h => key s10 h10; obtain ⟨hkey, rfl⟩ := ident_ok h10
      peel h => u s11 h11; obtain ⟨hdot, rfl⟩ := tok_ok h11
      split at h
      · simp at h
      · rename_i hne
        have hk : ent.1 = key := by simpa using hne
        peel h => a s12 h12; obtain ⟨ha, rfl⟩ := ident_ok h12
        peel h => attrs s13 h13
        peel h => u s14 h14; obtain ⟨hrp, rfl⟩ := tok_ok h14
        have hs4 : s4 = s2.adv := by
          split at h4
          · exact (optK_eq h4).2 rfl
          · exact (pure_ok h4).2.symm
        subst hs4
        have ge := idxEntity_sound _ _ _ h7
        have gm := idxMore_sound _ _ _ _ _ _ h13
        rw [← hk] at hkey
        cases c
        · simp only [Bool.false_eq_true, ite_false] at h
          obtain ⟨hh, rfl⟩ := pure_ok h
          cases hh
          exact .new (gty _ rfl) hidx hfor hlp ge hon hlp2 hkey hdot ha gm hrp (.inl ⟨rfl, rfl⟩)
        · simp only [ite_true] at h
          peel h => op s15 h15
          peel h => opts s16 h16
          obtain ⟨hh, rfl⟩ := pure_ok h
          cases hh
          have gopt : opts = none ∧ s16 = s13.adv ∨ true = true ∧
              (if ft then IdxTy.fulltext else if vec then .vector else .range) ≠ .range ∧
              s13.adv.cur = .kw .options ∧ ∃ m, opts = some m ∧ o.pmap s13.adv.adv = .ok m s16 := by
            cases op
            · have hs15 : s15 = s13.adv := by
                split at h15
                · exact (optK_eq h15).2 rfl
                · exact (pure_ok h15).2.symm
              subst hs15
              simp only [Bool.false_eq_true, ite_false] at h16
              obtain ⟨rfl, rfl⟩ := pure_ok h16; exact .inl ⟨rfl, rfl⟩
            · split at h15
              · rename_i hfv
                obtain ⟨hk2, rfl⟩ := (optK_eq h15).1 rfl
                simp only [ite_true] at h16
                peel h16 => m s17 h17
                obtain ⟨rfl, rfl⟩ := pure_ok h16
                refine .inr ⟨rfl, ?_, hk2, m, rfl, h17⟩
                cases ft <;> cases vec <;> simp_all
              · simp at h15
          exact .new (gty _ rfl) hidx hfor hlp ge hon hlp2 hkey hdot ha gm hrp gopt
    · -- old syntax `INDEX ON :L(a, ...)`
      have hfv : ft = false ∧ vec = false := by
        cases ft <;> cases vec <;> simp_all
      obtain ⟨rfl, rfl⟩ := hfv
      have := (optK_eq h1).2 rfl; subst s1
      have := (optK_eq h2).2 rfl; subst s2
      simp only [Bool.not_false, Bool.and_self, ite_true] at h4
      obtain ⟨hon, rfl⟩ := (optK_eq h4).1 rfl
      simp only [ite_true] at h
      peel h => u s5 h5; obtain ⟨hcol, rfl⟩ := tok_ok h5
      peel h => l s6 h6; obtain ⟨hl, rfl⟩ := ident_ok h6
      peel h => u s7 h7; obtain ⟨hlp, rfl⟩ := tok_ok h7
      peel h => a s8 h8; obtain ⟨ha, rfl⟩ := ident_ok h8
      peel h => attrs s9 h9
      peel h => u s10 h10; obtain ⟨hrp, rfl⟩ := tok_ok h10
      obtain ⟨hh, rfl⟩ := pure_ok h
      cases hh
      exact .old hidx hon hcol hl hlp ha (identList_sound _ _ _ _ _ h9) hrp

/-- `oC_Cypher` (Cypher.g4:30-31) plus FalkorDB's index commands. -/
inductive GCypher : PS → IR → PS → Prop
  | create {s r s1 s'} : s.cur = .kw .create → GIdx o true s.adv r s1 → GSemis s1 s' → GCypher s r s'
  | drop {s r s1 s'} : s.cur = .kw .drop → GIdx o false s.adv r s1 → GSemis s1 s' → GCypher s r s'
  | query {s r s1 s'} : GQuery o CExt.rust s r s1 → GSemis s1 s' → GCypher s r s'

/-- **`Parser::parse` is sound**: anything it accepts is a Cypher.g4 query
(under the listed extensions) or a FalkorDB index command, followed by
`;`s and the end of input, and the IR is the grammar's. -/
theorem parseP_sound (hpp : o.PPRec) (F n : Nat) (s : PS) r s' (h : parseP o F n s = .ok r s') :
    GCypher o s r s' := by
  unfold parseP at h
  peel h => s0 s1 h1
  simp only [run_getS, R.ok.injEq] at h1; obtain ⟨rfl, rfl⟩ := h1
  peel h => ir s2 h2
  cases ir with
  | some ir =>
    dsimp only at h
    peel h => u s3 h3
    obtain ⟨rfl, rfl⟩ := pure_ok h
    have ge := endP_sound _ _ _ _ h3
    unfold indexOps at h2
    peel h2 => cr s4 h4
    cases cr
    · have := (optK_eq h4).2 rfl; subst s4
      simp only [Bool.false_eq_true, ite_false] at h2
      peel h2 => dr s5 h5
      cases dr
      · simp at h2
      · obtain ⟨hd, rfl⟩ := (optK_eq h5).1 rfl
        simp only [ite_true] at h2
        exact .drop hd (idxBody_sound o F false _ _ _ h2) ge
    · obtain ⟨hc, rfl⟩ := (optK_eq h4).1 rfl
      simp only [ite_true] at h2
      exact .create hc (idxBody_sound o F true _ _ _ h2) ge
  | none =>
    dsimp only at h
    peel h => u s3 h3
    simp only [run_setS, R.ok.injEq] at h3; obtain ⟨-, rfl⟩ := h3
    peel h => q s4 h4
    peel h => u s5 h5
    obtain ⟨rfl, rfl⟩ := pure_ok h
    exact .query (queryP_sound o hpp F n _ _ _ h4) (endP_sound _ _ _ _ h5)

end

end FalkorParserGrammar.C
