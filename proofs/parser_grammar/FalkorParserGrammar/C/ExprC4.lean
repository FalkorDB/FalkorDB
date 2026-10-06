/-
# shortestPath soundness; no panics in the expression constructs
-/
import FalkorParserGrammar.C.ExprC3

namespace FalkorParserGrammar.C

/-- shortestPath's type list: pushed in order (cypher.rs:1772-1780). -/
inductive GSpTypes : PS → List Nat → List Nat → PS → Prop
  | last {s acc t} : identOf s.cur = some t → s.adv.cur ≠ .pipe → s.adv.cur ≠ .colon →
      GSpTypes s acc (acc ++ [t]) s.adv
  | more {s acc t s1 out s'} : identOf s.cur = some t →
      (s.adv.cur = .pipe ∧ (s.adv.adv.cur = .colon ∧ s1 = s.adv.adv.adv ∨ s.adv.adv.cur ≠ .colon ∧ s1 = s.adv.adv) ∨
       s.adv.cur ≠ .pipe ∧ s.adv.cur = .colon ∧ s1 = s.adv.adv) →
      GSpTypes s1 (acc ++ [t]) out s' → GSpTypes s acc out s'

/-- shortestPath's `*` range (cypher.rs:1784-1807). -/
inductive GSpRange : PS → Nat × Option Nat → PS → Prop
  | none {s} : s.cur ≠ .star → GSpRange s (1, some 1) s
  | star {s r s'} : s.cur = .star → GRangeBody s.adv r s' → GSpRange s r s'

/-- shortestPath's `[ ... ]` (cypher.rs:1765-1850), with no edge filter. -/
inductive GSpDet : PS → List Nat × Nat × Option Nat → PS → Prop
  | bare {s} : s.cur ≠ .lbrack → GSpDet s ([], 1, some 1) s
  | det {s sa ts sb r sc} : s.cur = .lbrack →
      (identOf s.adv.cur ≠ none ∧ sa = s.adv.adv ∨ identOf s.adv.cur = none ∧ sa = s.adv) →
      (ts = [] ∧ sa.cur ≠ .colon ∧ sb = sa ∨ sa.cur = .colon ∧ GSpTypes sa.adv [] ts sb) →
      GSpRange sb r sc → (∀ p, sc.cur ≠ .param p) → sc.cur ≠ .lbrace → sc.cur = .rbrack →
      GSpDet s (ts, r.1, r.2) sc.adv

/-- `shortestPath( (src) <? - [..]? - >? (dst) )` after `shortestPath(`
(Neo4j syntax; FalkorDB restrictions: bound nodes, min ≤ 1, no filter). -/
inductive GShortest : PS → ET → PS → Prop
  | mk {s src inc s1 d s2 out s3 dst} :
      s.cur = .lparen → identOf s.adv.cur = some src → s.adv.adv.cur = .rparen →
      (inc = true ∧ s.adv.adv.adv.cur = .lt ∧ s.adv.adv.adv.adv.cur = .dash ∧ s1 = s.adv.adv.adv.adv.adv ∨
       inc = false ∧ s.adv.adv.adv.cur ≠ .lt ∧ s.adv.adv.adv.cur = .dash ∧ s1 = s.adv.adv.adv.adv) →
      GSpDet s1 d s2 → s2.cur = .dash →
      (out = true ∧ s2.adv.cur = .gt ∧ s3 = s2.adv.adv ∨ out = false ∧ s3 = s2.adv) →
      s3.cur = .lparen → identOf s3.adv.cur = some dst → s3.adv.adv.cur = .rparen →
      s3.adv.adv.adv.cur = .rparen → d.2.1 ≤ 1 →
      GShortest s (.node (.shortest d.1 d.2.1 d.2.2 (inc || out))
        [leaf (.var (if inc && !out then dst else src)), leaf (.var (if inc && !out then src else dst))])
        s3.adv.adv.adv.adv

/-- No-panic and termination-free facts used below. -/
theorem np_spTypes : ∀ n acc, NP (spTypes n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_spTypes n; unfold spTypes; np_go

theorem np_skipP : ∀ n d, NP (skipP n d)
  | 0, _ => np_fuelP
  | n + 1, d => by have := np_skipP n; unfold skipP; np_go

theorem np_spRangeP : NP spRangeP := by have := np_hopP; unfold spRangeP; np_go

theorem np_shortestP (F : Nat) : NP (shortestP F) := by
  have := np_spTypes; have := np_skipP; have := np_spRangeP
  unfold shortestP; np_go

theorem np_caseAlts (pe : PE) (hpe : NP pe) : ∀ n acc, NP (caseAlts pe n acc)
  | 0, _ => np_fuelP
  | n + 1, acc => by have := np_caseAlts pe hpe n; unfold caseAlts; np_go

theorem np_caseP (pe : PE) (hpe : NP pe) (F : Nat) : NP (caseP pe F) := by
  have := np_caseAlts pe hpe; unfold caseP; np_go

theorem np_noAgg (e : ET) : NP (noAgg e) := by unfold noAgg; np_go

theorem np_reduceP (pe : PE) (hpe : NP pe) : NP (reduceP pe) := by
  have := np_noAgg; unfold reduceP; np_go

theorem np_listCompP (pe : PE) (hpe : NP pe) (v : Nat) : NP (listCompP pe v) := by
  have := np_noAgg; unfold listCompP; np_go

theorem np_attempt {α} {x : P α} (hx : NP x) : NP (attempt x) :=
  ⟨fun s => by unfold attempt; have := hx.run s; split <;> simp_all⟩

theorem np_rejectAggregate (t : ET) : NP (rejectAggregate t) := by unfold rejectAggregate; np_go

section
variable (o : Oracle) (ho : o.Safe)
include ho

theorem np_pcChain (F : Nat) : ∀ n left g, NP (pcChain o F n left g)
  | 0, _, _ => np_fuelP
  | n + 1, left, g => by
    have := np_pcChain F n; have := np_relP o ho F; unfold pcChain; np_go

theorem np_patCompP (F : Nat) (pv : Option Nat) : NP (patCompP o F pv) := by
  have := np_nodeP o ho F; have := np_pcChain o ho F; have := np_pe o ho
  unfold patCompP; np_go

theorem np_listLitP (F : Nat) (forb : Bool) : NP (listLitP o F forb) := by
  have := np_patCompP o ho F; have := np_listCompP (o.pe false) (np_pe o ho false)
  have := np_rejectAggregate; have hs : ∀ s, NP (setS s) := fun _ => ⟨fun _ => by simp⟩
  have ha : ∀ pv, NP (attempt (patCompP o F pv)) := fun pv => np_attempt (np_patCompP o ho F pv)
  unfold listLitP listLitRest; np_go

end

end FalkorParserGrammar.C
