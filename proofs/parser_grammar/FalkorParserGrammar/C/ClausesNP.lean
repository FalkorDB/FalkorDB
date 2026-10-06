/-
# No panics at the clause level

The two `unreachable!()` of the clause dispatchers (cypher.rs:963, 1012)
are reached only through `parse_single_query`'s and FOREACH's keyword
guards, so the whole clause layer never panics whenever the expression
parser does not (`parseExpr_no_panic`).
-/
import FalkorParserGrammar.C.Clauses

namespace FalkorParserGrammar.C

theorem np_fuelP {α} : NP (fuelP : P α) := ⟨fun _ => by simp⟩
theorem np_getS : NP getS := ⟨fun _ => by simp⟩
theorem np_anonFresh : NP anonFresh := ⟨fun _ => by simp [anonFresh]⟩

theorem np_peek_bind {β} {f : CT → P β} (h : ∀ c, NP (f c)) : NP (peek >>= f) :=
  np_bind np_peek h

theorem np_match_opt {β} (x : Option Nat → P β) (a : Option Nat) (h : ∀ a, NP (x a)) : NP (x a) := h a

syntax "np_go" : tactic
macro_rules
  | `(tactic| np_go) => `(tactic| repeat (first
      | exact np_pure _ | exact np_fail | exact np_fuelP | exact np_tok _ | exact np_opt _
      | exact np_ident | exact np_next | exact np_peek | exact np_tryIdent | exact np_getS
      | exact np_anonFresh
      | assumption
      | (apply_assumption <;> assumption)
      | (apply np_peek_bind; intro _)
      | (apply np_bind)
      | (intro _)
      | (apply np_ite)
      | split))

section
variable (o : Oracle) (ho : o.Safe)
include ho

theorem np_pe (b : Bool) : NP (o.pe b) := ho.pe b
theorem np_pp : NP o.pp := ho.pp
theorem np_pmap : NP o.pmap := ho.pmap

theorem np_labelsP : ∀ n acc, NP (labelsP n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by unfold labelsP; have := np_labelsP n; np_go

theorem np_propsP : NP (propsP o) := by have := np_pmap o ho; unfold propsP; np_go

theorem np_aliasOr (a : Option Nat) : NP (aliasOr a) := by cases a <;> simp [aliasOr] <;> np_go

theorem np_nodeP (F : Nat) : NP (nodeP o F) := by
  have := np_labelsP o ho; have := np_propsP o ho; have := np_aliasOr o ho
  unfold nodeP; np_go

theorem np_typesP : ∀ n acc, NP (typesP n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by unfold typesP; have := np_typesP n; np_go

theorem np_hopP : NP hopP := by unfold hopP; np_go
theorem np_varLenP : NP varLenP := by have := np_hopP; unfold varLenP; np_go

theorem np_detailP (F : Nat) (cl : CK) : NP (detailP o F cl) := by
  have := np_aliasOr o ho; have := np_typesP; have := np_varLenP; have := np_propsP o ho
  unfold detailP; np_go

theorem np_relP (F : Nat) (cl : CK) (src : QNode Nat) : NP (relP o F cl src) := by
  have := np_detailP o ho F cl; have := np_nodeP o ho F
  unfold relP; np_go

theorem np_chainA (F : Nat) (cl : CK) : ∀ n left gs, NP (chainA o F cl n left gs)
  | 0, _, _ => np_fuelP
  | n + 1, left, gs => by
    unfold chainA; have := np_chainA F cl n; have := np_relP o ho F cl; np_go

theorem np_chainN (F : Nat) (cl : CK) (p : Nat) : ∀ n left vars gs, NP (chainN o F cl p n left vars gs)
  | 0, _, _, _ => np_fuelP
  | n + 1, left, vars, gs => by
    unfold chainN; have := np_chainN F cl p n; have := np_relP o ho F cl; np_go

theorem np_aspLoop (F : Nat) (cl : CK) (la : Nat) :
    ∀ n prev found vars gs, NP (aspLoop o F cl la n prev found vars gs)
  | 0, _, _, _, _ => np_fuelP
  | n + 1, prev, found, vars, gs => by
    unfold aspLoop; have := np_aspLoop F cl la n; have := np_relP o ho F cl; np_go

theorem np_partP (F : Nat) (cl : CK) (gs : QG Nat × List Nat) : NP (partP o F cl gs) := by
  have := np_nodeP o ho F; have := np_chainA o ho F cl; have := np_chainN o ho F cl
  have := np_aspLoop o ho F cl
  unfold partP aspPart; np_go

theorem np_sepP (cl : CK) : NP (sepP cl) := by unfold sepP; np_go

theorem np_patternLoop (F : Nat) (cl : CK) : ∀ n gs, NP (patternLoop o F cl n gs)
  | 0, _ => np_fuelP
  | n + 1, gs => by
    unfold patternLoop; have := np_patternLoop F cl n; have := np_partP o ho F cl
    have := np_sepP; np_go

theorem np_patternP (F : Nat) (cl : CK) : NP (patternP o F cl) := np_patternLoop o ho F cl F _

theorem np_whereP : NP (whereP o) := by have := np_pe o ho; unfold whereP; np_go
theorem np_orderItems : ∀ n acc, NP (orderItems o n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_orderItems n; have := np_pe o ho; unfold orderItems; np_go
theorem np_orderbyP (n : Nat) : NP (orderbyP o n) := by
  have := np_orderItems o ho; unfold orderbyP; np_go
theorem np_countP (k : CK) : NP (countP o k) := by have := np_pe o ho; unfold countP; np_go
theorem np_oslP (n : Nat) : NP (oslP o n) := by
  have := np_orderbyP o ho; have := np_countP o ho; unfold oslP; np_go
theorem np_namedExprs (m : Bool) : ∀ n acc, NP (namedExprs o m n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_namedExprs m n; have := np_pe o ho; unfold namedExprs; np_go
theorem np_projP (m : Bool) (n : Nat) : NP (projP o m n) := by
  have := np_namedExprs o ho m; have := np_oslP o ho; unfold projP; np_go
theorem np_withP (w : Bool) (n : Nat) : NP (withP o w n) := by
  have := np_projP o ho; have := np_whereP o ho; unfold withP; np_go
theorem np_returnP (w : Bool) (n : Nat) : NP (returnP o w n) := by
  have := np_projP o ho; unfold returnP; np_go
theorem np_unwindP : NP (unwindP o) := by have := np_pe o ho; unfold unwindP; np_go
theorem np_matchP (F : Nat) (b : Bool) : NP (matchP o F b) := by
  have := np_patternP o ho; have := np_whereP o ho; unfold matchP; np_go
theorem np_createP (F : Nat) : NP (createP o F) := by have := np_patternP o ho; unfold createP; np_go
theorem np_exprList1 : ∀ n acc, NP (exprList1 o n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_exprList1 n; have := np_pe o ho; unfold exprList1; np_go
theorem np_exprItemsR : ∀ n acc, NP (exprItemsR o n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_exprItemsR n; have := np_pe o ho; unfold exprItemsR; np_go
theorem np_exprListR (n : Nat) (acc : List ET) : NP (exprListR o n acc) := by
  have := np_exprItemsR o ho n acc; unfold exprListR; np_go
theorem np_deleteMore : ∀ n es d, NP (deleteMore o n es d)
  | 0, _, _ => np_fuelP
  | n + 1, es, d => by have := np_deleteMore n; have := np_exprList1 o ho; unfold deleteMore; np_go
theorem np_deleteP (n : Nat) (d : Bool) : NP (deleteP o n d) := by
  have := np_deleteMore o ho; have := np_exprList1 o ho; unfold deleteP; np_go
theorem np_dotChain : ∀ n e, NP (dotChain n e)
  | 0, _ => np_fuelP
  | n + 1, e => by have := np_dotChain n; unfold dotChain; np_go
theorem np_targetP : NP (targetP o) := by
  have := np_pe o ho; have := np_pp o ho; unfold targetP rejectUnparenP; np_go
theorem np_setItems (F : Nat) : ∀ n acc, NP (setItems o F n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by
    have := np_setItems F n; have := np_pe o ho; have := np_targetP o ho; have := np_dotChain
    have := np_labelsP; unfold setItems; np_go
theorem np_setMore (F : Nat) : ∀ n acc, NP (setMore o F n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_setMore F n; have := np_setItems o ho F; unfold setMore; np_go
theorem np_setP (F : Nat) : NP (setP o F) := by
  have := np_setMore o ho F; have := np_setItems o ho F; unfold setP; np_go
theorem np_removeItems (F : Nat) : ∀ n acc, NP (removeItems o F n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by
    have := np_removeItems F n; have := np_targetP o ho; have := np_dotChain
    have := np_labelsP; unfold removeItems; np_go
theorem np_removeMore (F : Nat) : ∀ n acc, NP (removeMore o F n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_removeMore F n; have := np_removeItems o ho F; unfold removeMore; np_go
theorem np_removeP (F : Nat) : NP (removeP o F) := by
  have := np_removeMore o ho F; have := np_removeItems o ho F; unfold removeP; np_go
theorem np_mergeActions (F : Nat) : ∀ n a b, NP (mergeActions o F n a b)
  | 0, _, _ => np_fuelP
  | n + 1, a, b => by
    have := np_mergeActions F n; have := np_setItems o ho F; unfold mergeActions; np_go
theorem np_mergeP (F : Nat) : NP (mergeP o F) := by
  have := np_patternP o ho; have := np_mergeActions o ho F; unfold mergeP; np_go
theorem np_loadCsvP : NP (loadCsvP o) := by have := np_pe o ho; unfold loadCsvP; np_go
theorem np_dottedP : ∀ n acc, NP (dottedP n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_dottedP n; unfold dottedP; np_go
theorem np_yieldItems : ∀ n a b, NP (yieldItems n a b)
  | 0, _, _ => np_fuelP
  | n + 1, a, b => by have := np_yieldItems n; unfold yieldItems; np_go
theorem np_callProcP (F : Nat) : NP (callProcP o F) := by
  have := np_dottedP; have := np_exprListR o ho; have := np_yieldItems; have := np_whereP o ho
  unfold callProcP; np_go

theorem bind_np_of {α β} {x : P α} {f : α → P β} {s : PS} (hx : x s ≠ .panic) (hf : ∀ a, NP (f a)) :
    (x >>= f) s ≠ .panic := bind_np hx (fun a s1 _ => (hf a).run s1)

/-- `parse_reading_clasue`'s `unreachable!()` is not reached when it is
entered on a reading keyword other than CALL. -/
theorem readingP_np (F : Nat) (s : PS) (hg : isReadKw s.cur = true) (hc : s.cur ≠ .kw .call) :
    readingP o F s ≠ .panic := by
  have hm := np_matchP o ho F; have hu := np_unwindP o ho; have hl := np_loadCsvP o ho
  have hcase : s.cur = .kw .optional ∨ s.cur = .kw .match_ ∨ s.cur = .kw .unwind ∨ s.cur = .kw .load := by
    revert hg hc; cases s.cur <;> simp [isReadKw]
    rename_i k; cases k <;> simp
  unfold readingP
  rcases hcase with h | h | h | h
  · have : optK .optional s = .ok true s.adv := by simp [optK, opt, h]
    rw [run_bind, this]; simp only [ite_true]
    exact (np_bind (np_tok _) (fun _ => hm true)).run _
  all_goals
    have : optK .optional s = .ok false s := by simp [optK, opt, h]
    rw [run_bind, this]; simp only [Bool.false_eq_true, ite_false]
    rw [run_bind]; simp only [peek, h]
  · exact (np_bind np_next (fun _ => hm false)).run _
  · exact (np_bind np_next (fun _ => hu)).run _
  · exact (np_bind np_next (fun _ => hl)).run _

theorem np_peek_state {β} {f : CT → P β} (h : ∀ s, f s.cur s ≠ .panic) : NP (peek >>= f) :=
  ⟨fun s => by simp only [run_bind, peek]; exact h s⟩

/-- The whole mutually recursive clause layer. -/
def MutNP (o : Oracle) (F n : Nat) : Prop :=
  NP (queryP o F n) ∧ (∀ a bs, NP (unionMore o F n a bs)) ∧ NP (singleP o F n) ∧
  (∀ cl w, NP (segments o F n cl w)) ∧ (∀ cl, NP (readings o F n cl)) ∧
  (∀ cl w, NP (writings o F n cl w)) ∧
  (∀ s, isWriteKw s.cur = true → writingP o F n s ≠ .panic) ∧ NP (callP o F n) ∧
  NP (foreachP o F n) ∧ (∀ b, NP (foreachBody o F n b))

theorem np_mut (F : Nat) : ∀ n, MutNP o F n
  | 0 => ⟨np_fuelP, fun _ _ => np_fuelP, np_fuelP, fun _ _ => np_fuelP, fun _ => np_fuelP,
      fun _ _ => np_fuelP, fun s _ => by simp [writingP], np_fuelP, np_fuelP, fun _ => np_fuelP⟩
  | n + 1 => by
    obtain ⟨iq, iu, isg, iseg, ird, iwr, iwp, ical, ife, ifb⟩ := np_mut F n
    have := np_withP o ho; have := np_returnP o ho; have := np_createP o ho
    have := np_mergeP o ho; have := np_deleteP o ho; have := np_setP o ho
    have := np_removeP o ho; have := np_callProcP o ho; have := np_pe o ho
    have hwp : ∀ s, isWriteKw s.cur = true → writingP o F (n + 1) s ≠ .panic := by
      intro s hg
      simp only [writingP]; rw [run_bind]; simp only [peek]
      generalize hc : s.cur = c at hg ⊢
      cases c <;> simp only [isWriteKw, Bool.false_eq_true] at hg
      rename_i k
      cases k <;> simp only [isWriteKw, Bool.false_eq_true] at hg <;> dsimp only <;>
        (refine NP.run ?_ s; np_go)
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, hwp, ?_, ?_, ?_⟩
    · unfold queryP; np_go
    · intro a bs; unfold unionMore; np_go
    · unfold singleP; np_go
    · intro cl w; unfold segments; np_go
    · intro cl; unfold readings; apply np_peek_state o ho; intro s
      split
      · split
        · exact (np_bind np_next (fun _ => np_bind ical (fun _ => ird _))).run s
        · exact bind_np_of o ho (readingP_np o ho F s ‹_› ‹_›) (fun _ => ird _)
      · simp
    · intro cl w; unfold writings; apply np_peek_state o ho; intro s
      split
      · exact bind_np_of o ho (iwp s ‹_›) (fun _ => iwr _ _)
      · simp
    · unfold callP; np_go
    · unfold foreachP; np_go
    · intro b; unfold foreachBody; np_go

/-- **The clause layer never panics** (given an expression parser that
does not): `parse_query` and everything under it, including the two
`unreachable!()` dispatchers. -/
theorem queryP_no_panic (F n : Nat) : NP (queryP o F n) := (np_mut o ho F n).1

theorem np_endP : ∀ n, NP (endP n)
  | 0 => np_fuelP
  | n + 1 => by have := np_endP n; unfold endP; np_go

end

end FalkorParserGrammar.C