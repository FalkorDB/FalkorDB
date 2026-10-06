/-
# Clause grammar, part 2: SET / REMOVE / DELETE / MERGE / CREATE / MATCH /
# LOAD CSV / CALL procedure / reading clauses (Cypher.g4:66-154)
-/
import FalkorParserGrammar.C.ClausesG

namespace FalkorParserGrammar.C

section
variable (o : Oracle) (x : CExt)

/-- What `parse_primary_expr` promises about `recurse`: it is set only after
consuming `(` — with a `Paren` tree — or `[` — with a list tree (cypher.rs:2097-2111,
2807-2810). The expression model's `primary` satisfies it (`lf .paren` / `lf .list`). -/
def Oracle.PPRec (o : Oracle) : Prop :=
  ∀ s t s', o.pp s = .ok (t, true) s' →
    (s.cur = .lparen ∧ t.root = .paren) ∨ (s.cur = .lbrack ∧ t.root = .list)

/-- The atom of a SET / REMOVE target: `oC_Atom`, where a parenthesised
expression must be closed by `)`. -/
inductive GTarget : PS → ET → PS → Prop
  | plain {s t s1} : o.pp s = .ok (t, false) s1 → GTarget s t s1
  | paren {s t0 s1 t s2} : o.pp s = .ok (t0, true) s1 → s.cur = .lparen →
      o.pe false s1 = .ok t s2 → s2.cur = .rparen → GTarget s t s2.adv

theorem targetP_sound (hpp : o.PPRec) (s : PS) t s' (h : targetP o s = .ok t s') :
    GTarget o s t s' := by
  unfold targetP at h
  peel h => r s1 h1
  obtain ⟨t0, rec⟩ := r
  cases rec
  · simp only [Bool.false_eq_true, ite_false] at h
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .plain h1
  · simp only [ite_true] at h
    peel h => u0 s1' h0
    unfold rejectUnparenP at h0
    split at h0
    · rename_i hroot
      obtain ⟨-, rfl⟩ := pure_ok h0
      peel h => e s2 h2
      peel h => u s3 h3
      obtain ⟨hr, rfl⟩ := tok_ok h3
      obtain ⟨rfl, rfl⟩ := pure_ok h
      have hlp : s.cur = .lparen := by
        rcases hpp _ _ _ h1 with ⟨h, _⟩ | ⟨_, h⟩
        · exact h
        · rw [h] at hroot; cases hroot
      exact .paren h1 hlp h2 hr
    · simp [fail] at h0

/-- `( SP? oC_PropertyLookup )*` (Cypher.g4:273). -/
inductive GDots : PS → ET → ET → PS → Prop
  | nil {s t} : GDots s t t s
  | cons {s t p t' s'} : s.cur = .dot → identOf s.adv.cur = some p →
      GDots s.adv.adv (.node (.prop p) [t]) t' s' → GDots s t t' s'

theorem dotChain_sound : ∀ (n : Nat) t (s : PS) t' s', dotChain n t s = .ok t' s' → GDots s t t' s'
  | 0, _, _, _, _, h => by simp [dotChain] at h
  | n + 1, t, s, t', s', h => by
    unfold dotChain at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => u s2 h2
      have := next_ok h2; subst this
      peel h => p s3 h3
      obtain ⟨hp, rfl⟩ := ident_ok h3
      exact .cons ‹_› hp (dotChain_sound n _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil

/-- `oC_SetItem` (Cypher.g4:116-121). -/
inductive GSetItem : PS → SetItem → PS → Prop
  | prop {s t s1 t' s2 v s3} : GTarget o s t s1 → s1.cur = .dot → GDots s1 t t' s2 →
      s2.cur = .eq → o.pe false s2.adv = .ok v s3 → GSetItem s (.attr t' v false) s3
  | label {s t s1 v ls s2} : GTarget o s t s1 → t.root = .var v → s1.cur = .colon →
      GLabels s1 [] ls s2 → GSetItem s (.label v ls) s2
  | assign {s t s1 e s2} : GTarget o s t s1 → ((∃ v, t.root = .var v) ∨ x.anyTarget = true) →
      s1.cur = .eq → o.pe false s1.adv = .ok e s2 → GSetItem s (.attr t e true) s2
  | merge {s t s1 e s2} : GTarget o s t s1 → ((∃ v, t.root = .var v) ∨ x.anyTarget = true) →
      s1.cur = .plusEq → o.pe false s1.adv = .ok e s2 → GSetItem s (.attr t e false) s2

inductive GSetItems : PS → List SetItem → List SetItem → PS → Prop
  | one {s acc it s1} : GSetItem o x s it s1 → GSetItems s acc (acc ++ [it]) s1
  | more {s acc it s1 out s'} : GSetItem o x s it s1 → s1.cur = .comma →
      GSetItems s1.adv (acc ++ [it]) out s' → GSetItems s acc out s'

theorem setItems_sound (hpp : o.PPRec) (F : Nat) : ∀ (n : Nat) acc (s : PS) out s',
    setItems o F n acc s = .ok out s' → GSetItems o CExt.rust s acc out s'
  | 0, _, _, _, _, h => by simp [setItems] at h
  | n + 1, acc, s, out, s', h => by
    unfold setItems at h
    peel h => t s1 h1
    have gt := targetP_sound o hpp _ _ _ h1
    peel h => c s2 h2
    obtain ⟨hc0, hs⟩ := peek_ok h2; subst hc0; subst s2
    peel h => it s3 h3
    have gi : GSetItem o CExt.rust s it s3 := by
      split at h3
      · peel h3 => t' s4 h4
        peel h3 => u s5 h5
        obtain ⟨he, rfl⟩ := tok_ok h5
        peel h3 => v s6 h6
        obtain ⟨rfl, rfl⟩ := pure_ok h3
        exact .prop gt ‹_› (dotChain_sound _ _ _ _ _ h4) he h6
      · split at h3
        · rename_i _ hcol
          generalize hr : t.root = r at h3
          cases r <;> dsimp only at h3
          case var v =>
            peel h3 => ls s4 h4
            obtain ⟨rfl, rfl⟩ := pure_ok h3
            exact .label gt hr hcol (labelsP_sound _ _ _ _ _ h4)
          all_goals simp at h3
        · peel h3 => eq s4 h4
          cases eq
          · have := (opt_eq h4).2 rfl; subst s4
            simp only [Bool.false_eq_true, ite_false] at h3
            peel h3 => u s5 h5
            obtain ⟨hp, rfl⟩ := tok_ok h5
            peel h3 => v s6 h6
            obtain ⟨rfl, rfl⟩ := pure_ok h3
            exact .merge gt (.inr rfl) hp h6
          · obtain ⟨he, rfl⟩ := (opt_eq h4).1 rfl
            simp only [ite_true] at h3
            peel h3 => v s6 h6
            obtain ⟨rfl, rfl⟩ := pure_ok h3
            exact .assign gt (.inr rfl) he h6
    peel h => more s4 h4
    cases more
    · have := (opt_eq h4).2 rfl; subst s4
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .one gi
    · obtain ⟨hc, rfl⟩ := (opt_eq h4).1 rfl
      simp only [ite_true] at h
      exact .more gi hc (setItems_sound hpp F n _ _ _ _ h)

/-- `oC_Set` (Cypher.g4:111-112), with folding of `SET .. SET ..`. -/
inductive GSetMore : PS → List SetItem → IR → PS → Prop
  | done {s is} : GSetMore s is (.set is) s
  | more {s is is' s1 out s'} : x.foldUpd = true → s.cur = .kw .set →
      GSetItems o x s.adv is is' s1 → GSetMore s1 is' out s' → GSetMore s is out s'

theorem setMore_sound (hpp : o.PPRec) (F : Nat) : ∀ (n : Nat) acc (s : PS) out s',
    setMore o F n acc s = .ok out s' → GSetMore o CExt.rust s acc out s'
  | 0, _, _, _, _, h => by simp [setMore] at h
  | n + 1, acc, s, out, s', h => by
    unfold setMore at h
    peel h => b s1 h1
    cases b
    · have := (optK_eq h1).2 rfl; subst s1
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .done
    · obtain ⟨hk, rfl⟩ := (optK_eq h1).1 rfl
      simp only [ite_true] at h
      peel h => is s2 h2
      exact .more rfl hk (setItems_sound o hpp F _ _ _ _ _ h2) (setMore_sound hpp F n _ _ _ _ h)

theorem setP_sound (hpp : o.PPRec) (F : Nat) (s : PS) r s' (h : setP o F s = .ok r s') :
    ∃ is s1, GSetItems o CExt.rust s [] is s1 ∧ GSetMore o CExt.rust s1 is r s' := by
  unfold setP at h
  peel h => is s1 h1
  exact ⟨is, s1, setItems_sound o hpp F _ _ _ _ _ h1, setMore_sound o hpp F _ _ _ _ _ h⟩

/-- `oC_RemoveItem` (Cypher.g4:135-138). -/
inductive GRemoveItem : PS → ET → PS → Prop
  | prop {s t s1 t' s2} : GTarget o s t s1 → s1.cur = .dot → GDots s1 t t' s2 → GRemoveItem s t' s2
  | label {s t s1 ls s2} : GTarget o s t s1 → ((∃ v, t.root = .var v) ∨ x.anyTarget = true) →
      s1.cur = .colon → GLabels s1 [] ls s2 →
      GRemoveItem s (.node (.func fnHasLabels false) [t, .node .list (ls.map (fun l => leaf (.str l)))]) s2

inductive GRemoveItems : PS → List ET → List ET → PS → Prop
  | one {s acc it s1} : GRemoveItem o x s it s1 → GRemoveItems s acc (acc ++ [it]) s1
  | more {s acc it s1 out s'} : GRemoveItem o x s it s1 → s1.cur = .comma →
      GRemoveItems s1.adv (acc ++ [it]) out s' → GRemoveItems s acc out s'

theorem removeItems_sound (hpp : o.PPRec) (F : Nat) : ∀ (n : Nat) acc (s : PS) out s',
    removeItems o F n acc s = .ok out s' → GRemoveItems o CExt.rust s acc out s'
  | 0, _, _, _, _, h => by simp [removeItems] at h
  | n + 1, acc, s, out, s', h => by
    unfold removeItems at h
    peel h => t s1 h1
    have gt := targetP_sound o hpp _ _ _ h1
    peel h => c s2 h2
    obtain ⟨hc0, hs⟩ := peek_ok h2; subst hc0; subst s2
    peel h => it s3 h3
    have gi : GRemoveItem o CExt.rust s it s3 := by
      split at h3
      · exact .prop gt ‹_› (dotChain_sound _ _ _ _ _ h3)
      · split at h3
        · peel h3 => ls s4 h4
          obtain ⟨rfl, rfl⟩ := pure_ok h3
          exact .label gt (.inr rfl) ‹_› (labelsP_sound _ _ _ _ _ h4)
        · simp at h3
    peel h => more s4 h4
    cases more
    · have := (opt_eq h4).2 rfl; subst s4
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .one gi
    · obtain ⟨hc, rfl⟩ := (opt_eq h4).1 rfl
      simp only [ite_true] at h
      exact .more gi hc (removeItems_sound hpp F n _ _ _ _ h)

inductive GRemoveMore : PS → List ET → IR → PS → Prop
  | done {s is} : GRemoveMore s is (.remove is) s
  | more {s is is' s1 out s'} : x.foldUpd = true → s.cur = .kw .remove →
      GRemoveItems o x s.adv is is' s1 → GRemoveMore s1 is' out s' → GRemoveMore s is out s'

theorem removeMore_sound (hpp : o.PPRec) (F : Nat) : ∀ (n : Nat) acc (s : PS) out s',
    removeMore o F n acc s = .ok out s' → GRemoveMore o CExt.rust s acc out s'
  | 0, _, _, _, _, h => by simp [removeMore] at h
  | n + 1, acc, s, out, s', h => by
    unfold removeMore at h
    peel h => b s1 h1
    cases b
    · have := (optK_eq h1).2 rfl; subst s1
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .done
    · obtain ⟨hk, rfl⟩ := (optK_eq h1).1 rfl
      simp only [ite_true] at h
      peel h => is s2 h2
      exact .more rfl hk (removeItems_sound o hpp F _ _ _ _ _ h2) (removeMore_sound hpp F n _ _ _ _ h)

theorem removeP_sound (hpp : o.PPRec) (F : Nat) (s : PS) r s' (h : removeP o F s = .ok r s') :
    ∃ is s1, GRemoveItems o CExt.rust s [] is s1 ∧ GRemoveMore o CExt.rust s1 is r s' := by
  unfold removeP at h
  peel h => is s1 h1
  exact ⟨is, s1, removeItems_sound o hpp F _ _ _ _ _ h1, removeMore_sound o hpp F _ _ _ _ _ h⟩

/-- `oC_Expression ( SP? ',' SP? oC_Expression )*`. -/
inductive GExprs1 : PS → List ET → List ET → PS → Prop
  | one {s acc e s1} : o.pe false s = .ok e s1 → GExprs1 s acc (acc ++ [e]) s1
  | more {s acc e s1 out s'} : o.pe false s = .ok e s1 → s1.cur = .comma →
      GExprs1 s1.adv (acc ++ [e]) out s' → GExprs1 s acc out s'

theorem exprList1_sound : ∀ (n : Nat) acc (s : PS) out s',
    exprList1 o n acc s = .ok out s' → GExprs1 o s acc out s'
  | 0, _, _, _, _, h => by simp [exprList1] at h
  | n + 1, acc, s, out, s', h => by
    unfold exprList1 at h
    peel h => e s1 h1
    peel h => c s2 h2
    obtain ⟨hc0, hs⟩ := peek_ok h2; subst hc0; subst s2
    split at h
    · peel h => u s3 h3
      have := next_ok h3; subst this
      exact .more h1 ‹_› (exprList1_sound n _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .one h1

/-- `oC_Delete` (Cypher.g4:123-124) after `DETACH? DELETE`, with folding. -/
inductive GDelMore : PS → List ET → Bool → IR → PS → Prop
  | done {s es d} : GDelMore s es d (.delete es d) s
  | more {s es d d' s1 more s2 out s'} : x.foldUpd = true →
      (d' = false ∧ s.cur = .kw .delete ∧ s1 = s.adv ∨
        d' = true ∧ s.cur = .kw .detach ∧ s.adv.cur = .kw .delete ∧ s1 = s.adv.adv) →
      GExprs1 o s1 [] more s2 → GDelMore s2 (es ++ more) (d || d') out s' → GDelMore s es d out s'

theorem deleteMore_sound : ∀ (n : Nat) es d (s : PS) out s',
    deleteMore o n es d s = .ok out s' → GDelMore o CExt.rust s es d out s'
  | 0, _, _, _, _, _, h => by simp [deleteMore] at h
  | n + 1, es, d, s, out, s', h => by
    unfold deleteMore at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => d' s2 h2
      peel h => u s3 h3
      peel h => more s4 h4
      obtain ⟨hd, rfl⟩ := tok_ok h3
      have gm := exprList1_sound o _ _ _ _ _ h4
      have ih := deleteMore_sound n _ _ _ _ _ h
      cases d'
      · have := (optK_eq h2).2 rfl; subst s2
        exact .more rfl (.inl ⟨rfl, hd, rfl⟩) gm ih
      · obtain ⟨hk, rfl⟩ := (optK_eq h2).1 rfl
        exact .more rfl (.inr ⟨rfl, hk, hd, rfl⟩) gm ih
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .done

theorem deleteP_sound (n : Nat) (d : Bool) (s : PS) r s' (h : deleteP o n d s = .ok r s') :
    ∃ es s1, GExprs1 o s [] es s1 ∧ GDelMore o CExt.rust s1 es d r s' := by
  unfold deleteP at h
  peel h => es s1 h1
  exact ⟨es, s1, exprList1_sound o _ _ _ _ _ h1, deleteMore_sound o _ _ _ _ _ _ h⟩

/-- `( SP oC_MergeAction )*` (Cypher.g4:95-102). -/
inductive GMergeActions : PS → List SetItem → List SetItem → List SetItem × List SetItem → PS → Prop
  | done {s oc om} : GMergeActions s oc om (oc, om) s
  | onMatch {s oc om om' s1 out s'} : s.cur = .kw .on → s.adv.cur = .kw .match_ →
      s.adv.adv.cur = .kw .set → GSetItems o x s.adv.adv.adv om om' s1 →
      GMergeActions s1 oc om' out s' → GMergeActions s oc om out s'
  | onCreate {s oc om oc' s1 out s'} : s.cur = .kw .on → s.adv.cur = .kw .create →
      s.adv.adv.cur = .kw .set → GSetItems o x s.adv.adv.adv oc oc' s1 →
      GMergeActions s1 oc' om out s' → GMergeActions s oc om out s'

theorem mergeActions_sound (hpp : o.PPRec) (F : Nat) : ∀ (n : Nat) oc om (s : PS) out s',
    mergeActions o F n oc om s = .ok out s' → GMergeActions o CExt.rust s oc om out s'
  | 0, _, _, _, _, _, h => by simp [mergeActions] at h
  | n + 1, oc, om, s, out, s', h => by
    unfold mergeActions at h
    peel h => on s1 h1
    cases on
    · have := (optK_eq h1).2 rfl; subst s1
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .done
    · obtain ⟨hon, rfl⟩ := (optK_eq h1).1 rfl
      simp only [ite_true] at h
      peel h => m s2 h2
      cases m
      · have := (optK_eq h2).2 rfl; subst s2
        simp only [Bool.false_eq_true, ite_false] at h
        peel h => c s3 h3
        cases c
        · simp at h
        · obtain ⟨hcr, rfl⟩ := (optK_eq h3).1 rfl
          simp only [ite_true] at h
          peel h => u s4 h4
          obtain ⟨hs, rfl⟩ := tok_ok h4
          peel h => oc' s5 h5
          exact .onCreate hon hcr hs (setItems_sound o hpp F _ _ _ _ _ h5)
            (mergeActions_sound hpp F n _ _ _ _ _ h)
      · obtain ⟨hm, rfl⟩ := (optK_eq h2).1 rfl
        simp only [ite_true] at h
        peel h => u s4 h4
        obtain ⟨hs, rfl⟩ := tok_ok h4
        peel h => om' s5 h5
        exact .onMatch hon hm hs (setItems_sound o hpp F _ _ _ _ _ h5)
          (mergeActions_sound hpp F n _ _ _ _ _ h)

theorem mergeP_sound (hpp : o.PPRec) (F : Nat) (s : PS) r s' (h : mergeP o F s = .ok r s') :
    ∃ p s1 a, GPat o Ext.rust GExt.rust .merge (QG.empty, []) s p s1 ∧
      GMergeActions o CExt.rust s1 [] [] a s' ∧ r = .merge p a.1 a.2 := by
  unfold mergeP at h
  peel h => p s1 h1
  peel h => a s2 h2
  obtain ⟨rfl, rfl⟩ := pure_ok h
  exact ⟨p, s1, a, patternP_sound o F _ _ _ _ h1, mergeActions_sound o hpp F _ _ _ _ _ _ h2, rfl⟩

theorem matchP_sound (F : Nat) (b : Bool) (s : PS) r s' (h : matchP o F b s = .ok r s') :
    ∃ p s1 f, GPat o Ext.rust GExt.rust .match_ (QG.empty, []) s p s1 ∧ GWhere o s1 f s' ∧
      r = .match_ p f b := by
  unfold matchP at h
  peel h => p s1 h1
  peel h => f s2 h2
  obtain ⟨rfl, rfl⟩ := pure_ok h
  exact ⟨p, s1, f, patternP_sound o F _ _ _ _ h1, whereP_sound o _ _ _ h2, rfl⟩

theorem createP_sound (F : Nat) (s : PS) r s' (h : createP o F s = .ok r s') :
    ∃ p, GPat o Ext.rust GExt.rust .create (QG.empty, []) s p s' ∧ r = .create p := by
  unfold createP at h
  peel h => p s1 h1
  obtain ⟨rfl, rfl⟩ := pure_ok h
  exact ⟨p, patternP_sound o F _ _ _ _ h1, rfl⟩

/-- LOAD CSV after LOAD (FalkorDB extension; not in Cypher.g4). -/
inductive GLoadCsv : PS → IR → PS → Prop
  | mk {s hd s1 path s2 v dl s3} : s.cur = .kw .csv →
      (hd = true ∧ s.adv.cur = .kw .with_ ∧ s.adv.adv.cur = .kw .headers ∧ s1 = s.adv.adv.adv ∨
       hd = false ∧ s.adv.cur ≠ .kw .with_ ∧ s1 = s.adv) →
      s1.cur = .kw .from → o.pe false s1.adv = .ok path s2 → s2.cur = .kw .as_ →
      identOf s2.adv.cur = some v →
      (s2.adv.adv.cur = .kw .fieldterminator ∧ o.pe false s2.adv.adv.adv = .ok dl s3 ∨
       dl = leaf (.str strComma) ∧ s3 = s2.adv.adv) →
      GLoadCsv s (.loadCsv path hd dl v) s3

theorem loadCsvP_sound (s : PS) r s' (h : loadCsvP o s = .ok r s') : GLoadCsv o s r s' := by
  unfold loadCsvP at h
  peel h => u s1 h1
  obtain ⟨hcsv, rfl⟩ := tok_ok h1
  peel h => w s2 h2
  peel h => hd s3 h3
  peel h => u s4 h4
  obtain ⟨hfrom, rfl⟩ := tok_ok h4
  peel h => path s5 h5
  peel h => u s6 h6
  obtain ⟨has, rfl⟩ := tok_ok h6
  peel h => v s7 h7
  obtain ⟨hv, rfl⟩ := ident_ok h7
  peel h => ft s8 h8
  peel h => dl s9 h9
  obtain ⟨rfl, rfl⟩ := pure_ok h
  have gft : s5.adv.adv.cur = .kw .fieldterminator ∧ o.pe false s5.adv.adv.adv = .ok dl s9 ∨
      dl = leaf (.str strComma) ∧ s9 = s5.adv.adv := by
    cases ft
    · have := (optK_eq h8).2 rfl; subst s8
      simp only [Bool.false_eq_true, ite_false] at h9
      obtain ⟨rfl, rfl⟩ := pure_ok h9; exact .inr ⟨rfl, rfl⟩
    · obtain ⟨hk, rfl⟩ := (optK_eq h8).1 rfl
      simp only [ite_true] at h9; exact .inl ⟨hk, h9⟩
  have ghd : hd = true ∧ s.adv.cur = .kw .with_ ∧ s.adv.adv.cur = .kw .headers ∧ s3 = s.adv.adv.adv ∨
      hd = false ∧ s.adv.cur ≠ .kw .with_ ∧ s3 = s.adv := by
    cases w
    · rcases opt_ok h2 with ⟨hb, _, _⟩ | ⟨-, hnw, rfl⟩
      · cases hb
      simp only [Bool.false_eq_true, ite_false] at h3
      obtain ⟨rfl, rfl⟩ := pure_ok h3
      exact .inr ⟨rfl, hnw, rfl⟩
    · obtain ⟨hw, rfl⟩ := (optK_eq h2).1 rfl
      simp only [ite_true] at h3
      peel h3 => u sh hh
      obtain ⟨hhd, rfl⟩ := tok_ok hh
      obtain ⟨rfl, rfl⟩ := pure_ok h3
      exact .inl ⟨rfl, hw, hhd, rfl⟩
  exact .mk hcsv ghd hfrom h5 has hv gft

/-- `oC_ReadingClause` other than CALL (Cypher.g4:74-81) and LOAD CSV. -/
inductive GReading : PS → IR → PS → Prop
  | optMatch {s r s'} : s.cur = .kw .optional → s.adv.cur = .kw .match_ →
      (∃ p s1 f, GPat o Ext.rust GExt.rust .match_ (QG.empty, []) s.adv.adv p s1 ∧ GWhere o s1 f s' ∧
        r = .match_ p f true) → GReading s r s'
  | match_ {s r s'} : s.cur = .kw .match_ →
      (∃ p s1 f, GPat o Ext.rust GExt.rust .match_ (QG.empty, []) s.adv p s1 ∧ GWhere o s1 f s' ∧
        r = .match_ p f false) → GReading s r s'
  | unwind {s r s'} : s.cur = .kw .unwind → GUnwind o s.adv r s' → GReading s r s'
  | load {s r s'} : s.cur = .kw .load → GLoadCsv o s.adv r s' → GReading s r s'

theorem readingP_sound (F : Nat) (s : PS) r s' (h : readingP o F s = .ok r s') :
    GReading o s r s' := by
  unfold readingP at h
  peel h => op s1 h1
  cases op
  · have := (optK_eq h1).2 rfl; subst s1
    simp only [Bool.false_eq_true, ite_false] at h
    peel h => c s2 h2
    obtain ⟨hc0, hs⟩ := peek_ok h2; subst hc0; subst s2
    generalize hc : s.cur = c at h
    cases c <;> (try dsimp only at h)
    case kw k =>
      cases k <;> (try dsimp only at h)
      case match_ =>
        peel h => u s3 h3
        have := next_ok h3; subst this
        exact .match_ hc (matchP_sound o F false _ _ _ h)
      case unwind =>
        peel h => u s3 h3
        have := next_ok h3; subst this
        exact .unwind hc (unwindP_sound o _ _ _ h)
      case load =>
        peel h => u s3 h3
        have := next_ok h3; subst this
        exact .load hc (loadCsvP_sound o _ _ _ h)
      all_goals simp at h
    all_goals simp at h
  · obtain ⟨ho, rfl⟩ := (optK_eq h1).1 rfl
    simp only [ite_true] at h
    peel h => u s2 h2
    obtain ⟨hm, rfl⟩ := tok_ok h2
    exact .optMatch ho hm (matchP_sound o F true _ _ _ h)

/-- `oC_ExplicitProcedureInvocation`'s argument list after `(`:
`( oC_Expression ( ',' oC_Expression )* )? ')'` — after a `,` an expression (#3060). -/
inductive GItems : PS → List ET → List ET → PS → Prop
  | last {s acc e s1} : o.pe false s = .ok e s1 → s1.cur = .rparen → GItems s acc (acc ++ [e]) s1.adv
  | more {s acc e s1 out s'} : o.pe false s = .ok e s1 → s1.cur = .comma →
      GItems s1.adv (acc ++ [e]) out s' → GItems s acc out s'

inductive GArgs : PS → List ET → List ET → PS → Prop
  | close {s acc} : s.cur = .rparen → GArgs s acc acc s.adv
  | items {s acc out s'} : s.cur ≠ .rparen → GItems o s acc out s' → GArgs s acc out s'

theorem exprItemsR_sound : ∀ (n : Nat) acc (s : PS) out s',
    exprItemsR o n acc s = .ok out s' → GItems o s acc out s'
  | 0, _, _, _, _, h => by simp [exprItemsR] at h
  | n + 1, acc, s, out, s', h => by
    unfold exprItemsR at h
    peel h => e s2 h2
    peel h => c2 s3 h3
    obtain ⟨hc0, hs⟩ := peek_ok h3; subst hc0; subst s3
    split at h
    · peel h => u s4 h4
      have := next_ok h4; subst this
      exact .more h2 ‹_› (exprItemsR_sound n _ _ _ _ h)
    · peel h => u s4 h4
      obtain ⟨hr, rfl⟩ := tok_ok h4
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .last h2 hr

theorem exprListR_sound (n : Nat) acc (s : PS) out s'
    (h : exprListR o n acc s = .ok out s') : GArgs o s acc out s' := by
  unfold exprListR at h
  peel h => c s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  split at h
  · peel h => u s2 h2
    have := next_ok h2; subst this
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .close ‹_›
  · exact .items ‹_› (exprItemsR_sound o n _ _ _ _ h)

/-- `oC_ProcedureName` (dotted). -/
inductive GDotted : PS → List Nat → List Nat → PS → Prop
  | nil {s acc} : GDotted s acc acc s
  | cons {s acc v out s'} : s.cur = .dot → identOf s.adv.cur = some v →
      GDotted s.adv.adv (acc ++ [v]) out s' → GDotted s acc out s'

theorem dottedP_sound : ∀ (n : Nat) acc (s : PS) out s', dottedP n acc s = .ok out s' → GDotted s acc out s'
  | 0, _, _, _, _, h => by simp [dottedP] at h
  | n + 1, acc, s, out, s', h => by
    unfold dottedP at h
    peel h => c s1 h1
    obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
    split at h
    · peel h => u s2 h2
      have := next_ok h2; subst this
      peel h => v s3 h3
      obtain ⟨hv, rfl⟩ := ident_ok h3
      exact .cons ‹_› hv (dottedP_sound n _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil

/-- `oC_YieldItems` without WHERE (Cypher.g4:150-154). -/
inductive GYield : PS → List Nat → List (Option Nat) → List Nat × List (Option Nat) → PS → Prop
  | one {s outs als v p s1} : identOf s.cur = some v →
      (p = (v, none) ∧ s1 = s.adv ∨ ∃ al, s.adv.cur = .kw .as_ ∧ identOf s.adv.adv.cur = some al ∧
        p = (al, some v) ∧ s1 = s.adv.adv.adv) →
      GYield s outs als (outs ++ [p.1], als ++ [p.2]) s1
  | more {s outs als v p s1 out s'} : identOf s.cur = some v →
      (p = (v, none) ∧ s1 = s.adv ∨ ∃ al, s.adv.cur = .kw .as_ ∧ identOf s.adv.adv.cur = some al ∧
        p = (al, some v) ∧ s1 = s.adv.adv.adv) → s1.cur = .comma →
      GYield s1.adv (outs ++ [p.1]) (als ++ [p.2]) out s' → GYield s outs als out s'

theorem yieldItems_sound : ∀ (n : Nat) outs als (s : PS) out s',
    yieldItems n outs als s = .ok out s' → GYield s outs als out s'
  | 0, _, _, _, _, _, h => by simp [yieldItems] at h
  | n + 1, outs, als, s, out, s', h => by
    unfold yieldItems at h
    peel h => v s1 h1
    obtain ⟨hv, rfl⟩ := ident_ok h1
    peel h => a s2 h2
    peel h => p s3 h3
    have gp : p = (v, none) ∧ s3 = s.adv ∨ ∃ al, s.adv.cur = .kw .as_ ∧ identOf s.adv.adv.cur = some al ∧
        p = (al, some v) ∧ s3 = s.adv.adv.adv := by
      cases a
      · have := (optK_eq h2).2 rfl; subst s2
        simp only [Bool.false_eq_true, ite_false] at h3
        obtain ⟨rfl, rfl⟩ := pure_ok h3; exact .inl ⟨rfl, rfl⟩
      · obtain ⟨ha, rfl⟩ := (optK_eq h2).1 rfl
        simp only [ite_true] at h3
        peel h3 => al s4 h4
        obtain ⟨hal, rfl⟩ := ident_ok h4
        obtain ⟨rfl, rfl⟩ := pure_ok h3; exact .inr ⟨al, ha, hal, rfl, rfl⟩
    peel h => more s4 h4
    cases more
    · have := (opt_eq h4).2 rfl; subst s4
      simp only [Bool.false_eq_true, ite_false] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h; exact .one hv gp
    · obtain ⟨hc, rfl⟩ := (opt_eq h4).1 rfl
      simp only [ite_true] at h
      exact .more hv gp hc (yieldItems_sound n _ _ _ _ _ h)

/-- `oC_InQueryCall` after CALL (Cypher.g4:140-154), with the registry's
default outputs when there is no YIELD. -/
inductive GCallProc : PS → IR → PS → Prop
  | mk {s v name s1 fi args s2 r s'} : identOf s.cur = some v → GDotted s.adv [v] name s1 →
      o.proc name = some fi →
      (s1.cur = .lparen ∧ GArgs o s1.adv [] args s2 ∨ x.callImplicit = true ∧ args = [] ∧ s2 = s1) →
      (∃ ya s3 f, s2.cur = .kw .yield_ ∧ GYield s2.adv [] [] ya s3 ∧ GWhere o s3 f s' ∧
          r = .call name args ya.1 ya.2 f true) ∨
        (s' = s2 ∧ r = if fi.isProc then .call name args fi.outputs (fi.outputs.map (fun _ => none)) none false
          else .call name args [] [] none false) →
      GCallProc s r s'

theorem callProcP_sound (F : Nat) (s : PS) r s' (h : callProcP o F s = .ok r s') :
    GCallProc o CExt.rust s r s' := by
  unfold callProcP at h
  peel h => v s1 h1
  obtain ⟨hv, rfl⟩ := ident_ok h1
  peel h => name s2 h2
  generalize hp : o.proc name = pr at h
  cases pr with
  | none => simp at h
  | some fi =>
    dsimp only at h
    peel h => lp s3 h3
    peel h => args s4 h4
    have ga : s2.cur = .lparen ∧ GArgs o s2.adv [] args s4 ∨
        CExt.rust.callImplicit = true ∧ args = [] ∧ s4 = s2 := by
      cases lp
      · have := (opt_eq h3).2 rfl; subst s3
        simp only [Bool.false_eq_true, ite_false] at h4
        obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .inr ⟨rfl, rfl, rfl⟩
      · obtain ⟨hl, rfl⟩ := (opt_eq h3).1 rfl
        simp only [ite_true] at h4; exact .inl ⟨hl, exprListR_sound o _ _ _ _ _ h4⟩
    split at h
    · simp at h
    · peel h => y s5 h5
      refine .mk hv (dottedP_sound _ _ _ _ _ h2) hp ga ?_
      cases y
      · have := (optK_eq h5).2 rfl; subst s5
        simp only [Bool.false_eq_true, ite_false] at h
        right
        split at h
        · obtain ⟨rfl, rfl⟩ := pure_ok h; simp [*]
        · obtain ⟨rfl, rfl⟩ := pure_ok h; simp [*]
      · obtain ⟨hy, rfl⟩ := (optK_eq h5).1 rfl
        simp only [ite_true] at h
        peel h => ya s6 h6
        peel h => f s7 h7
        obtain ⟨rfl, rfl⟩ := pure_ok h
        exact .inl ⟨ya, s6, f, hy, yieldItems_sound _ _ _ _ _ _ h6, whereP_sound o _ _ _ h7, rfl⟩

end

end FalkorParserGrammar.C
