/-
# Clause grammar, part 3: queries (Cypher.g4:30-78) and soundness of
# `parse_query` / `parse_single_query` / the clause loops / CALL / FOREACH
-/
import FalkorParserGrammar.C.ClausesG2

namespace FalkorParserGrammar.C

section
variable (o : Oracle) (x : CExt)

/-- What may follow a single query (cypher.rs:886-910). -/
def GEnd (s s' : PS) : Prop :=
  (s.cur = .kw .union ∨ s.cur = .semi ∨ s.cur = .rbrace) ∧ s' = s ∨ s.cur = .eof ∧ s' = s.adv

mutual
/-- `oC_RegularQuery` (Cypher.g4:41-47). -/
inductive GQuery : PS → IR → PS → Prop
  | one {s r s'} : GSingle s r s' → GQuery s r s'
  | union {s r1 s1 all s2 r2 s3 out s'} : GSingle s r1 s1 → s1.cur = .kw .union →
      (all = true ∧ s1.adv.cur = .kw .all ∧ s2 = s1.adv.adv ∨ all = false ∧ s2 = s1.adv) →
      GSingle s2 r2 s3 → GUnion s3 all [r1, r2] out s' → GQuery s out s'
/-- `( SP? oC_Union )*`, all of one kind. -/
inductive GUnion : PS → Bool → List IR → IR → PS → Prop
  | done {s all bs} : GUnion s all bs (.union bs all) s
  | more {s all bs s1 r s2 out s'} : s.cur = .kw .union →
      (all = true ∧ s.adv.cur = .kw .all ∧ s1 = s.adv.adv ∨ all = false ∧ s1 = s.adv) →
      GSingle s1 r s2 → GUnion s2 all (bs ++ [r]) out s' → GUnion s all bs out s'
/-- `oC_SingleQuery` (Cypher.g4:53-64): WITH-separated parts, then RETURN, or
an updating tail (or, with `noReturnEnd`, nothing). -/
inductive GSingle : PS → IR → PS → Prop
  | ret {s cl w s1 p s2 s'} : GSegs s [] false (cl, w) s1 → s1.cur = .kw .return_ →
      GProj o s1.adv p s2 →
      (s2.cur = .eof ∨ s2.cur = .kw .union ∨ s2.cur = .semi ∨ s2.cur = .rbrace) →
      GEnd s2 s' → GSingle s (.query (cl ++ [.return_ p w]) false) s'
  | noRet {s cl w s1 s'} : GSegs s [] false (cl, w) s1 → (w = true ∨ x.noReturnEnd = true) →
      GEnd s1 s' → GSingle s (.query cl w) s'
/-- `( oC_ReadingClause* oC_UpdatingClause* oC_With )*` then the last part's
reading and updating clauses. -/
inductive GSegs : PS → List IR → Bool → List IR × Bool → PS → Prop
  | stop {s cl w cl1 s1 out s2} : GReads s cl cl1 s1 → GWrites s1 cl1 false out s2 →
      GSegs s cl w out s2
  | with_ {s cl w cl1 s1 cl2 w2 s2 p s3 f s4 out s'} : GReads s cl cl1 s1 →
      GWrites s1 cl1 false (cl2, w2) s2 → s2.cur = .kw .with_ → GProj o s2.adv p s3 →
      GWhere o s3 f s4 → GSegs s4 (cl2 ++ [.with_ p f w2]) false out s' → GSegs s cl w out s'
/-- `( oC_ReadingClause SP? )*`. -/
inductive GReads : PS → List IR → List IR → PS → Prop
  | nil {s cl} : GReads s cl cl s
  | read {s cl r s1 out s'} : GReading o s r s1 → GReads s1 (cl ++ [r]) out s' → GReads s cl out s'
  | call {s cl cs s1 out s'} : s.cur = .kw .call → GCall s.adv cs s1 →
      GReads s1 (cl ++ cs) out s' → GReads s cl out s'
/-- `( oC_UpdatingClause SP? )*`. -/
inductive GWrites : PS → List IR → Bool → List IR × Bool → PS → Prop
  | nil {s cl w} : GWrites s cl w (cl, w) s
  | cons {s cl w r s1 out s'} : GWriting s r s1 → GWrites s1 (cl ++ [r]) true out s' →
      GWrites s cl w out s'
/-- `oC_UpdatingClause` (Cypher.g4:66-72) plus FOREACH. -/
inductive GWriting : PS → IR → PS → Prop
  | create {s r s'} : s.cur = .kw .create →
      (∃ p, GPat o Ext.rust GExt.rust .create (QG.empty, []) s.adv p s' ∧ r = .create p) →
      GWriting s r s'
  | merge {s r s'} : s.cur = .kw .merge →
      (∃ p s1 a, GPat o Ext.rust GExt.rust .merge (QG.empty, []) s.adv p s1 ∧
        GMergeActions o x s1 [] [] a s' ∧ r = .merge p a.1 a.2) → GWriting s r s'
  | delete {s d s1 r s'} :
      (d = false ∧ s.cur = .kw .delete ∧ s1 = s.adv ∨
        d = true ∧ s.cur = .kw .detach ∧ s.adv.cur = .kw .delete ∧ s1 = s.adv.adv) →
      (∃ es s2, GExprs1 o s1 [] es s2 ∧ GDelMore o x s2 es d r s') → GWriting s r s'
  | set {s r s'} : s.cur = .kw .set →
      (∃ is s1, GSetItems o x s.adv [] is s1 ∧ GSetMore o x s1 is r s') → GWriting s r s'
  | remove {s r s'} : s.cur = .kw .remove →
      (∃ is s1, GRemoveItems o x s.adv [] is s1 ∧ GRemoveMore o x s1 is r s') → GWriting s r s'
  | foreach {s r s'} : s.cur = .kw .foreach → GForeach s.adv r s' → GWriting s r s'
/-- `CALL { query }` or `oC_InQueryCall`, after CALL. -/
inductive GCall : PS → List IR → PS → Prop
  | sub {s body s1} : s.cur = .lbrace → GQuery s.adv body s1 → s1.cur = .rbrace →
      GCall s [.callSub body (bodyHasReturn body)] s1.adv
  | proc {s r s'} : GCallProc o x s r s' → GCall s [r] s'
/-- `FOREACH ( var IN expr | updating+ )`, after FOREACH. -/
inductive GForeach : PS → IR → PS → Prop
  | mk {s v l s1 body s2} : s.cur = .lparen → identOf s.adv.cur = some v →
      s.adv.adv.cur = .kw .in_ → o.pe false s.adv.adv.adv = .ok l s1 → s1.cur = .pipe →
      GBody s1.adv [] body s2 → body ≠ [] → s2.cur = .rparen → GForeach s (.forEach l v body) s2.adv
inductive GBody : PS → List IR → List IR → PS → Prop
  | nil {s b} : GBody s b b s
  | cons {s b r s1 out s'} : GWriting s r s1 → GBody s1 (b ++ [r]) out s' → GBody s b out s'
end

/-- Soundness of the mutually recursive clause layer at budget `n`. -/
def MutSound (F n : Nat) : Prop :=
  (∀ s r s', queryP o F n s = .ok r s' → GQuery o CExt.rust s r s') ∧
  (∀ a bs s r s', unionMore o F n a bs s = .ok r s' → GUnion o CExt.rust s a bs r s') ∧
  (∀ s r s', singleP o F n s = .ok r s' → GSingle o CExt.rust s r s') ∧
  (∀ cl w s r s', segments o F n cl w s = .ok r s' → GSegs o CExt.rust s cl w r s') ∧
  (∀ cl s r s', readings o F n cl s = .ok r s' → GReads o CExt.rust s cl r s') ∧
  (∀ cl w s r s', writings o F n cl w s = .ok r s' → GWrites o CExt.rust s cl w r s') ∧
  (∀ s r s', writingP o F n s = .ok r s' → GWriting o CExt.rust s r s') ∧
  (∀ s r s', callP o F n s = .ok r s' → GCall o CExt.rust s r s') ∧
  (∀ s r s', foreachP o F n s = .ok r s' → GForeach o CExt.rust s r s') ∧
  (∀ b s r s', foreachBody o F n b s = .ok r s' → GBody o CExt.rust s b r s')

theorem unionHead {s : PS} {all : Bool} {s1 : PS} (h : optK .all s = .ok all s1) :
    all = true ∧ s.cur = .kw .all ∧ s1 = s.adv ∨ all = false ∧ s1 = s := by
  cases all
  · exact .inr ⟨rfl, (optK_eq h).2 rfl⟩
  · obtain ⟨hk, e⟩ := (optK_eq h).1 rfl; exact .inl ⟨rfl, hk, e⟩

/-- A writing clause from its keyword (shared by the clause loop and FOREACH). -/
theorem writing_arm (hpp : o.PPRec) (F n : Nat)
    (ife : ∀ s r s', foreachP o F n s = .ok r s' → GForeach o CExt.rust s r s')
    (s : PS) r s' (k : CK)
    (h : (match CT.kw k with
      | .kw .create => (do next; createP o F : P IR)
      | .kw .merge => do next; mergeP o F
      | .kw .detach | .kw .delete => do
        let d ← optK .detach
        tokK .delete
        deleteP o n d
      | .kw .set => do next; setP o F
      | .kw .remove => do next; removeP o F
      | .kw .foreach => do next; foreachP o F n
      | _ => oops) s = .ok r s') (hc : s.cur = .kw k) : GWriting o CExt.rust s r s' := by
  cases k <;> (try dsimp only at h)
  case create =>
    peel h => u s1 h1; have := next_ok h1; subst this
    exact .create hc (createP_sound o F _ _ _ h)
  case merge =>
    peel h => u s1 h1; have := next_ok h1; subst this
    exact .merge hc (mergeP_sound o hpp F _ _ _ h)
  case set =>
    peel h => u s1 h1; have := next_ok h1; subst this
    exact .set hc (setP_sound o hpp F _ _ _ h)
  case remove =>
    peel h => u s1 h1; have := next_ok h1; subst this
    exact .remove hc (removeP_sound o hpp F _ _ _ h)
  case foreach =>
    peel h => u s1 h1; have := next_ok h1; subst this
    exact .foreach hc (ife _ _ _ h)
  case delete =>
    peel h => d s1 h1
    have := (optK_eq h1).2 (by cases d; rfl; obtain ⟨hk, _⟩ := (optK_eq h1).1 rfl; simp [hc] at hk)
    subst s1
    peel h => u s2 h2
    obtain ⟨hd, rfl⟩ := tok_ok h2
    have hd0 : d = false := by cases d; rfl; obtain ⟨hk, _⟩ := (optK_eq h1).1 rfl; simp [hc] at hk
    subst hd0
    exact .delete (.inl ⟨rfl, hd, rfl⟩) (deleteP_sound o n false _ _ _ h)
  case detach =>
    peel h => d s1 h1
    have hd1 : d = true := by
      cases d
      · have := (optK_eq h1).2 rfl; simp [optK, opt, hc] at h1
      · rfl
    subst hd1
    obtain ⟨_, rfl⟩ := (optK_eq h1).1 rfl
    peel h => u s2 h2
    obtain ⟨hd, rfl⟩ := tok_ok h2
    exact .delete (.inr ⟨rfl, hc, hd, rfl⟩) (deleteP_sound o n true _ _ _ h)
  all_goals simp at h

theorem sound_mut (hpp : o.PPRec) (F : Nat) : ∀ n, MutSound o F n
  | 0 => by
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> intros <;>
      simp_all [queryP, unionMore, singleP, segments, readings, writings, writingP, callP,
        foreachP, foreachBody]
  | n + 1 => by
    obtain ⟨iq, iu, isg, iseg, ird, iwr, iwp, ical, ife, ifb⟩ := sound_mut hpp F n
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · -- parse_query
      intro s r s' h
      simp only [queryP] at h
      peel h => first s1 h1
      peel h => u s2 h2
      cases u
      · have := (optK_eq h2).2 rfl; subst s2
        simp only [Bool.false_eq_true, ite_false] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact .one (isg _ _ _ h1)
      · obtain ⟨hu, rfl⟩ := (optK_eq h2).1 rfl
        simp only [ite_true] at h
        peel h => all s3 h3
        peel h => second s4 h4
        exact .union (isg _ _ _ h1) hu (unionHead h3) (isg _ _ _ h4) (iu _ _ _ _ _ h)
    · -- the UNION loop
      intro a bs s r s' h
      simp only [unionMore] at h
      peel h => u s1 h1
      cases u
      · have := (optK_eq h1).2 rfl; subst s1
        simp only [Bool.false_eq_true, ite_false] at h
        obtain ⟨rfl, rfl⟩ := pure_ok h; exact .done
      · obtain ⟨hu, rfl⟩ := (optK_eq h1).1 rfl
        simp only [ite_true] at h
        peel h => a' s2 h2
        split at h
        · simp at h
        · rename_i hne
          have : a' = a := by simpa using hne
          subst this
          peel h => b s3 h3
          exact .more hu (unionHead h2) (isg _ _ _ h3) (iu _ _ _ _ _ h)
    · -- parse_single_query
      intro s r s' h
      simp only [singleP] at h
      peel h => seg s1 h1
      peel h => ret s2 h2
      peel h => res s3 h3
      peel h => c s4 h4
      obtain ⟨hc0, hs⟩ := peek_ok h4; subst hc0; subst s4
      have gend : GEnd s3 s' ∧ r = .query res.1 res.2 := by
        split at h
        · obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨.inl ⟨by simpa using ‹_›, rfl⟩, rfl⟩
        · peel h => u s5 h5
          obtain ⟨he, rfl⟩ := tok_ok h5
          obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨.inr ⟨he, rfl⟩, rfl⟩
      obtain ⟨ge, rfl⟩ := gend
      have gs := iseg _ _ _ _ _ h1
      cases ret
      · have := (optK_eq h2).2 rfl; subst s2
        simp only [Bool.false_eq_true, ite_false] at h3
        obtain ⟨rfl, rfl⟩ := pure_ok h3
        exact .noRet gs (.inr rfl) ge
      · obtain ⟨hr, rfl⟩ := (optK_eq h2).1 rfl
        simp only [ite_true] at h3
        peel h3 => rc s5 h5
        obtain ⟨p, gp, rfl⟩ := returnP_sound o _ n _ _ _ h5
        peel h3 => c s6 h6
        obtain ⟨hc0, hs⟩ := peek_ok h6; subst hc0; subst s6
        split at h3
        · obtain ⟨rfl, rfl⟩ := pure_ok h3
          exact .ret gs hr gp (by simpa using ‹_›) ge
        · simp at h3
    · -- the WITH-separated parts
      intro cl w s r s' h
      simp only [segments] at h
      peel h => cl1 s1 h1
      peel h => wr s2 h2
      peel h => wi s3 h3
      have gr := ird _ _ _ _ h1
      have gw := iwr _ _ _ _ _ h2
      cases wi
      · have := (optK_eq h3).2 rfl; subst s3
        simp only [Bool.false_eq_true, ite_false] at h
        have : r = wr ∧ s' = s2 := by
          split at h
          · peel h => c s4 h4
            obtain ⟨hc0, hs⟩ := peek_ok h4; subst hc0; subst s4
            split at h
            · simp at h
            · obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨rfl, rfl⟩
          · obtain ⟨rfl, rfl⟩ := pure_ok h; exact ⟨rfl, rfl⟩
        obtain ⟨rfl, rfl⟩ := this
        exact .stop gr gw
      · obtain ⟨hw, rfl⟩ := (optK_eq h3).1 rfl
        simp only [ite_true] at h
        peel h => wc s4 h4
        obtain ⟨p, s5, f, gp, gf, rfl⟩ := withP_sound o _ n _ _ _ h4
        obtain ⟨cl2, w2⟩ := wr
        exact .with_ gr gw hw gp gf (iseg _ _ _ _ _ h)
    · -- reading clauses
      intro cl s r s' h
      simp only [readings] at h
      peel h => c s1 h1
      obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
      split at h
      · split at h
        · rename_i _ hcall
          peel h => u s2 h2
          have := next_ok h2; subst this
          peel h => cs s3 h3
          exact .call hcall (ical _ _ _ h3) (ird _ _ _ _ h)
        · peel h => rr s2 h2
          exact .read (readingP_sound o F _ _ _ h2) (ird _ _ _ _ h)
      · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil
    · -- updating clauses
      intro cl w s r s' h
      simp only [writings] at h
      peel h => c s1 h1
      obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
      split at h
      · peel h => rr s2 h2
        exact .cons (iwp _ _ _ h2) (iwr _ _ _ _ _ h)
      · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil
    · -- parse_writing_clause
      intro s r s' h
      simp only [writingP] at h
      peel h => c s1 h1
      obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
      generalize hc : s.cur = c at h
      cases c
      case kw k => exact writing_arm o hpp F n ife s r s' k h hc
      all_goals simp at h
    · -- parse_call_clause
      intro s r s' h
      simp only [callP] at h
      peel h => c s1 h1
      obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
      split at h
      · peel h => u s2 h2
        have := next_ok h2; subst this
        peel h => body s3 h3
        peel h => u s4 h4
        obtain ⟨hr, rfl⟩ := tok_ok h4
        obtain ⟨rfl, rfl⟩ := pure_ok h
        exact .sub ‹_› (iq _ _ _ h3) hr
      · peel h => rr s2 h2
        obtain ⟨rfl, rfl⟩ := pure_ok h
        exact .proc (callProcP_sound o F _ _ _ h2)
    · -- parse_foreach_clause
      intro s r s' h
      simp only [foreachP] at h
      peel h => u s1 h1
      obtain ⟨hl, rfl⟩ := tok_ok h1
      peel h => v s2 h2
      obtain ⟨hv, rfl⟩ := ident_ok h2
      peel h => u s3 h3
      obtain ⟨hin, rfl⟩ := tok_ok h3
      peel h => l s4 h4
      split at h
      · simp at h
      · peel h => u s5 h5
        obtain ⟨hp, rfl⟩ := tok_ok h5
        peel h => body s6 h6
        peel h => u s7 h7
        obtain ⟨hr, rfl⟩ := tok_ok h7
        split at h
        · simp at h
        · obtain ⟨rfl, rfl⟩ := pure_ok h
          exact .mk hl hv hin h4 hp (ifb _ _ _ _ h6) ‹_› hr
    · -- the FOREACH body
      intro b s r s' h
      simp only [foreachBody] at h
      peel h => c s1 h1
      obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
      generalize hc : s.cur = c at h
      cases c
      case kw k =>
        cases k <;> (try dsimp only at h)
        case create =>
          peel h => u s2 h2; have := next_ok h2; subst this
          peel h => rr s3 h3
          exact .cons (.create hc (createP_sound o F _ _ _ h3)) (ifb _ _ _ _ h)
        case merge =>
          peel h => u s2 h2; have := next_ok h2; subst this
          peel h => rr s3 h3
          exact .cons (.merge hc (mergeP_sound o hpp F _ _ _ h3)) (ifb _ _ _ _ h)
        case set =>
          peel h => u s2 h2; have := next_ok h2; subst this
          peel h => rr s3 h3
          exact .cons (.set hc (setP_sound o hpp F _ _ _ h3)) (ifb _ _ _ _ h)
        case remove =>
          peel h => u s2 h2; have := next_ok h2; subst this
          peel h => rr s3 h3
          exact .cons (.remove hc (removeP_sound o hpp F _ _ _ h3)) (ifb _ _ _ _ h)
        case foreach =>
          peel h => u s2 h2; have := next_ok h2; subst this
          peel h => rr s3 h3
          exact .cons (.foreach hc (ife _ _ _ h3)) (ifb _ _ _ _ h)
        case delete =>
          peel h => d s2 h2
          have hd0 : d = false := by
            cases d; rfl; obtain ⟨hk, _⟩ := (optK_eq h2).1 rfl; simp [hc] at hk
          subst hd0
          have := (optK_eq h2).2 rfl; subst s2
          peel h => u s3 h3
          obtain ⟨hd, rfl⟩ := tok_ok h3
          peel h => rr s4 h4
          exact .cons (.delete (.inl ⟨rfl, hd, rfl⟩) (deleteP_sound o n false _ _ _ h4)) (ifb _ _ _ _ h)
        case detach =>
          peel h => d s2 h2
          have hd1 : d = true := by
            cases d
            · simp [optK, opt, hc] at h2
            · rfl
          subst hd1
          obtain ⟨_, rfl⟩ := (optK_eq h2).1 rfl
          peel h => u s3 h3
          obtain ⟨hd, rfl⟩ := tok_ok h3
          peel h => rr s4 h4
          exact .cons (.delete (.inr ⟨rfl, hc, hd, rfl⟩) (deleteP_sound o n true _ _ _ h4)) (ifb _ _ _ _ h)
        all_goals (obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil)
      all_goals (obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil)

/-- **parse_query is sound**: every query the Rust clause parser accepts is
an `oC_RegularQuery` of Cypher.g4 under the listed extensions (`CExt.rust`,
`Ext.rust`, `GExt.rust`), and the `QueryIR` it returns is the one the
rules' semantic actions build. -/
theorem queryP_sound (hpp : o.PPRec) (F n : Nat) (s : PS) r s' (h : queryP o F n s = .ok r s') :
    GQuery o CExt.rust s r s' := (sound_mut o hpp F n).1 s r s' h

/-- `oC_Cypher`'s tail (Cypher.g4:31): `;`s then end of input
(`expect_end_of_input`, cypher.rs:516-524; Cypher.g4 allows one `;`). -/
inductive GSemis : PS → PS → Prop
  | eof {s} : s.cur = .eof → GSemis s s
  | semi {s s'} : s.cur = .semi → GSemis s.adv s' → GSemis s s'

theorem endP_sound : ∀ n (s : PS) u s', endP n s = .ok u s' → GSemis s s'
  | 0, _, _, _, h => by simp [endP] at h
  | n + 1, s, u, s', h => by
    unfold endP at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => u s2 h2
      have := next_ok h2; subst this
      exact .semi ‹_› (endP_sound n _ _ _ h)
    · split at h
      · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .eof ‹_›
      · simp at h

end

end FalkorParserGrammar.C
