/-
# The pattern grammar (`graph/src/Cypher.g4:214-270`) and soundness of the
# node / relationship pattern parsers

Each `G*` relation is a rule of Cypher.g4 with its semantic action (the
tree it denotes). Implementation extensions are guarded by a flag of `Ext`;
`Ext.g4` switches them all off, `Ext.rust` is what the Rust accepts.
-/
import FalkorParserGrammar.C.Pattern

namespace FalkorParserGrammar.C

/-- Places where the Rust parser accepts more than Cypher.g4. -/
structure Ext where
  /-- `[:A:B]`: a type list separated by `:` alone (cypher.rs:2981-2984;
  Cypher.g4:255 needs `|`). C rejects it. -/
  typeColon : Bool
  /-- `MATCH a MATCH b` / `CREATE a CREATE b`: the next clause's keyword
  continues the same pattern (cypher.rs:1555-1560, W2-binder-1). -/
  fold : Bool
  /-- `p = allShortestPaths(...)` in MATCH (Neo4j syntax, not in Cypher.g4). -/
  asp : Bool

def Ext.g4 : Ext := ⟨false, false, false⟩
def Ext.rust : Ext := ⟨true, true, true⟩

/-- `( oC_Variable SP? )?` with the fresh-name action for an absent one. -/
inductive GOptVar : PS → Nat → PS → Prop
  | named {s v} : identOf s.cur = some v → GOptVar s v s.adv
  | anon {s} : identOf s.cur = none → GOptVar s (anonName s.2) (s.1, s.2 + 1)

/-- `oC_NodeLabels?` (Cypher.g4:257-261), accumulated as a set. -/
inductive GLabels : PS → List Nat → List Nat → PS → Prop
  | nil {s acc} : GLabels s acc acc s
  | cons {s acc l out s'} : s.cur = .colon → identOf s.adv.cur = some l →
      GLabels s.adv.adv (sins l acc) out s' → GLabels s acc out s'

/-- `oC_Properties?` (Cypher.g4:249-252): a map literal or a parameter. -/
inductive GProps (o : Oracle) : PS → ET → PS → Prop
  | none {s} : GProps o s (leaf .map) s
  | param {s p} : s.cur = .param p → GProps o s (leaf (.param p)) s.adv
  | map {s m s'} : s.cur = .lbrace → o.pmap s = .ok m s' → GProps o s m s'

/-- `oC_NodePattern` (Cypher.g4:233-234). -/
inductive GNode (o : Oracle) : PS → QNode Nat → PS → Prop
  | mk {s a s1 ls s2 m s3} : s.cur = .lparen → GOptVar s.adv a s1 → GLabels s1 [] ls s2 →
      GProps o s2 m s3 → s3.cur = .rparen → GNode o s ⟨a, ls, m⟩ s3.adv

theorem labelsP_sound : ∀ (n : Nat) (acc : List Nat) (s : PS) out s',
    labelsP n acc s = .ok out s' → GLabels s acc out s'
  | 0, _, _, _, _, h => by simp [labelsP] at h
  | n + 1, acc, s, out, s', h => by
    unfold labelsP at h
    peel h => c s1 h1
    obtain ⟨rfl, rfl⟩ := peek_ok h1
    split at h
    · rename_i hc
      peel h => u s2 h2
      peel h => l s3 h3
      have e2 := next_ok h2; subst e2
      obtain ⟨hl, rfl⟩ := ident_ok h3
      exact .cons hc hl (labelsP_sound n _ _ _ _ h)
    · obtain ⟨rfl, rfl⟩ := pure_ok h; exact .nil

theorem propsP_sound (o : Oracle) (s : PS) m s' (h : propsP o s = .ok m s') : GProps o s m s' := by
  unfold propsP at h
  peel h => c s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  generalize hc : s.cur = c at h
  cases c <;> dsimp only at h
  case param p =>
    peel h => u s2 h2
    have e2 := next_ok h2; subst e2
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .param hc
  case lbrace => exact .map hc h
  all_goals (obtain ⟨rfl, rfl⟩ := pure_ok h; exact .none)

theorem optVar_sound {a : Option Nat} {s s1 s2 : PS} {al : Nat}
    (h1 : tryIdent s = .ok a s1) (h2 : aliasOr a s1 = .ok al s2) :
    GOptVar s al s2 := by
  rcases tryIdent_ok h1 with ⟨v, rfl, hv, rfl⟩ | ⟨rfl, hv, rfl⟩
  · obtain ⟨rfl, rfl⟩ := pure_ok (show (pure v : P Nat) s.adv = _ from h2); exact .named hv
  · simp only [aliasOr, anonFresh, R.ok.injEq] at h2; obtain ⟨rfl, rfl⟩ := h2; exact .anon hv

/-- **parse_node_pattern is sound**: an accepted node pattern is an
`oC_NodePattern` and the node built is the rule's. -/
theorem nodeP_sound (o : Oracle) (F : Nat) (s : PS) n s' (h : nodeP o F s = .ok n s') :
    GNode o s n s' := by
  unfold nodeP at h
  peel h => u s1 h1
  peel h => a s2 h2
  peel h => al s3 h3
  peel h => ls s4 h4
  peel h => m s5 h5
  peel h => u2 s6 h6
  obtain ⟨rfl, rfl⟩ := pure_ok h
  obtain ⟨hc1, rfl⟩ := tok_ok h1
  obtain ⟨hc6, rfl⟩ := tok_ok h6
  exact .mk hc1 (optVar_sound h2 h3) (labelsP_sound _ _ _ _ _ h4) (propsP_sound o _ _ _ h5) hc6

/-- `oC_RelationshipTypes` (Cypher.g4:254-255): `:` already consumed. After a
name, `|` then an optional `:` continues; with `typeColon`, `:` alone does. -/
inductive GTypes (x : Ext) : PS → List Nat → List Nat → PS → Prop
  | last {s acc t} : identOf s.cur = some t → GTypes x s acc (sins t acc) s.adv
  | pipe {s acc t out s'} : identOf s.cur = some t → s.adv.cur = .pipe →
      GTypes x s.adv.adv (sins t acc) out s' → GTypes x s acc out s'
  | pipeColon {s acc t out s'} : identOf s.cur = some t → s.adv.cur = .pipe →
      s.adv.adv.cur = .colon → GTypes x s.adv.adv.adv (sins t acc) out s' → GTypes x s acc out s'
  | colon {s acc t out s'} : x.typeColon = true → identOf s.cur = some t → s.adv.cur = .colon →
      GTypes x s.adv.adv (sins t acc) out s' → GTypes x s acc out s'

theorem opt_eq {t s b s'} (h : opt t s = .ok b s') : (b = true → s.cur = t ∧ s' = s.adv) ∧
    (b = false → s' = s) := by
  rcases opt_ok h with ⟨rfl, h1, h2⟩ | ⟨rfl, _, h2⟩ <;> simp_all

theorem typesP_sound : ∀ (n : Nat) (acc : List Nat) (s : PS) out s',
    typesP n acc s = .ok out s' → GTypes Ext.rust s acc out s'
  | 0, _, _, _, _, h => by simp [typesP] at h
  | n + 1, acc, s, out, s', h => by
    unfold typesP at h
    peel h => t s1 h1
    peel h => p s2 h2
    peel h => c s3 h3
    obtain ⟨ht, rfl⟩ := ident_ok h1
    have ih := typesP_sound n
    cases p <;> cases c <;> simp only [Bool.or_false, Bool.or_true, Bool.false_or, ite_true] at h
    · obtain ⟨rfl, rfl⟩ := pure_ok h
      have := (opt_eq h2).2 rfl; subst this
      have := (opt_eq h3).2 rfl; subst this
      exact .last ht
    · have := (opt_eq h2).2 rfl; subst this
      obtain ⟨hc, rfl⟩ := (opt_eq h3).1 rfl
      exact .colon rfl ht hc (ih _ _ _ _ h)
    · obtain ⟨hp, rfl⟩ := (opt_eq h2).1 rfl
      have := (opt_eq h3).2 rfl; subst this
      exact .pipe ht hp (ih _ _ _ _ h)
    · obtain ⟨hp, rfl⟩ := (opt_eq h2).1 rfl
      obtain ⟨hc, rfl⟩ := (opt_eq h3).1 rfl
      exact .pipeColon ht hp hc (ih _ _ _ _ h)

/-- `oC_IntegerLiteral?` inside a range. -/
inductive GHop : PS → Option Int → PS → Prop
  | none {s} : GHop s none s
  | some {s i} : s.cur = .int i → GHop s (some i) s.adv

/-- `oC_RangeLiteral` (Cypher.g4:263-264) after `*`, with FalkorDB's
reading of the bounds (`*` = 1.., `*n` = exactly n, `*a..b`). -/
inductive GRangeBody : PS → Nat × Option Nat → PS → Prop
  | dots {s a s1 b s2} : GHop s a s1 → s1.cur = .dotdot → GHop s1.adv b s2 →
      GRangeBody s (asU32 (a.getD 1), b.map asU32) s2
  | exact {s i} : s.cur = .int i → GRangeBody s (asU32 i, some (asU32 i)) s.adv
  | bare {s} : GRangeBody s (1, none) s

inductive GRange : PS → Option (Nat × Option Nat) → PS → Prop
  | none {s} : GRange s none s
  | star {s r s'} : s.cur = .star → GRangeBody s.adv r s' → GRange s (some r) s'

theorem hopP_sound (s : PS) a s' (h : hopP s = .ok a s') : GHop s a s' := by
  unfold hopP at h
  peel h => c s1 h1
  obtain ⟨hc0, hs⟩ := peek_ok h1; subst hc0; subst s1
  generalize hc : s.cur = c at h
  cases c <;> dsimp only at h
  case int i =>
    peel h => u s2 h2
    have := next_ok h2; subst this
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .some hc
  all_goals (obtain ⟨rfl, rfl⟩ := pure_ok h; exact .none)

theorem varLenP_sound (s : PS) r s' (h : varLenP s = .ok r s') : GRange s r s' := by
  unfold varLenP at h
  peel h => st s1 h1
  cases st
  · simp only [Bool.false_eq_true, ite_false] at h
    have := (opt_eq h1).2 rfl; subst this
    obtain ⟨rfl, rfl⟩ := pure_ok h; exact .none
  · simp only [ite_true] at h
    obtain ⟨hs, rfl⟩ := (opt_eq h1).1 rfl
    peel h => a s2 h2
    peel h => dd s3 h3
    have ga := hopP_sound _ _ _ h2
    cases dd
    · simp only [Bool.false_eq_true, ite_false] at h
      have := (opt_eq h3).2 rfl; subst this
      cases a with
      | some i =>
        obtain ⟨rfl, rfl⟩ := pure_ok h
        cases ga; exact .star hs (.exact ‹_›)
      | none =>
        obtain ⟨rfl, rfl⟩ := pure_ok h
        cases ga; exact .star hs .bare
    · simp only [ite_true] at h
      obtain ⟨hd, rfl⟩ := (opt_eq h3).1 rfl
      peel h => b s4 h4
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact .star hs (.dots ga hd (hopP_sound _ _ _ h4))

/-- `oC_RelationshipDetail` (Cypher.g4:246-247) after `[`, with the
`[*1..1]` → single hop normalisation (cypher.rs:3021-3025). -/
inductive GDetail (o : Oracle) (x : Ext) : PS → Nat × List Nat × ET × Option (Nat × Option Nat) → PS → Prop
  | mk {s a s1 ts s2 r s3 m s4} : GOptVar s a s1 →
      (ts = [] ∧ s2 = s1 ∨ s1.cur = .colon ∧ GTypes x s1.adv [] ts s2) →
      GRange s2 r s3 → GProps o s3 m s4 → s4.cur = .rbrack →
      GDetail o x s (a, ts, m, normVL r) s4.adv

/-- The arrows of `oC_RelationshipPattern` (Cypher.g4:239-244): `<`? `-`
detail? `-` `>`?; the relationship reads right-to-left only for `<-...-`. -/
inductive GRel (o : Oracle) (x : Ext) : QNode Nat → PS → QRel Nat × QNode Nat → PS → Prop
  | mk {src s inc s1 d s2 out s3 dst s4} :
      (inc = true ∧ s.cur = .lt ∧ s1 = s.adv.adv ∨ inc = false ∧ s1 = s.adv) →
      (inc = true → s.adv.cur = .dash) → (inc = false → s.cur = .dash) →
      (s1.cur = .lbrack ∧ GDetail o x s1.adv d s2 ∨
        d = (anonName s1.2, [], leaf .map, none) ∧ s2 = (s1.1, s1.2 + 1)) →
      s2.cur = .dash →
      (out = true ∧ s2.adv.cur = .gt ∧ s3 = s2.adv.adv ∨ out = false ∧ s3 = s2.adv) →
      GNode o s3 dst s4 →
      GRel o x src s
        ((if inc == out then QRel.new d.1 d.2.1 d.2.2.1 src dst true (d.2.2.2.map (·.1)) (d.2.2.2.bind (·.2))
          else if inc then QRel.new d.1 d.2.1 d.2.2.1 dst src false (d.2.2.2.map (·.1)) (d.2.2.2.bind (·.2))
          else QRel.new d.1 d.2.1 d.2.2.1 src dst false (d.2.2.2.map (·.1)) (d.2.2.2.bind (·.2))), dst) s4

theorem detailP_sound (o : Oracle) (F : Nat) (cl : CK) (s : PS) d s'
    (h : detailP o F cl s = .ok d s') : GDetail o Ext.rust s d s' := by
  unfold detailP at h
  peel h => a s1 h1
  peel h => al s2 h2
  peel h => col s3 h3
  peel h => ts s4 h4
  peel h => r s5 h5
  split at h
  · simp at h
  · split at h
    · simp at h
    · peel h => m s6 h6
      peel h => u s7 h7
      obtain ⟨rfl, rfl⟩ := pure_ok h
      obtain ⟨hc, rfl⟩ := tok_ok h7
      cases col
      · have := (opt_eq h3).2 rfl; subst this
        simp only [Bool.false_eq_true, ite_false] at h4
        obtain ⟨rfl, rfl⟩ := pure_ok h4
        exact .mk (optVar_sound h1 h2) (.inl ⟨rfl, rfl⟩) (varLenP_sound _ _ _ h5)
          (propsP_sound o _ _ _ h6) hc
      · obtain ⟨hc3, e⟩ := (opt_eq h3).1 rfl; subst e
        simp only [ite_true] at h4
        exact .mk (optVar_sound h1 h2) (.inr ⟨hc3, typesP_sound _ _ _ _ _ h4⟩)
          (varLenP_sound _ _ _ h5) (propsP_sound o _ _ _ h6) hc

/-- **parse_relationship_pattern is sound**: an accepted relationship
pattern is an `oC_PatternElementChain` step (with `[:A:B]` as the only
extension) and the relationship built is the rule's. -/
theorem relP_sound (o : Oracle) (F : Nat) (cl : CK) (src : QNode Nat) (s : PS) rr s'
    (h : relP o F cl src s = .ok rr s') : GRel o Ext.rust src s rr s' := by
  unfold relP at h
  peel h => inc s1 h1
  peel h => u s2 h2
  peel h => det s3 h3
  peel h => d s4 h4
  peel h => u2 s5 h5
  peel h => out s6 h6
  peel h => dst s7 h7
  obtain ⟨hd2, rfl⟩ := tok_ok h2
  obtain ⟨hd5, rfl⟩ := tok_ok h5
  have gi : (inc = true ∧ s.cur = .lt ∧ s1.adv = s.adv.adv ∨ inc = false ∧ s1.adv = s.adv) ∧
      (inc = true → s.adv.cur = .dash) ∧ (inc = false → s.cur = .dash) := by
    cases inc
    · have := (opt_eq h1).2 rfl; subst this; exact ⟨.inr ⟨rfl, rfl⟩, by simp, fun _ => hd2⟩
    · obtain ⟨hl, rfl⟩ := (opt_eq h1).1 rfl; exact ⟨.inl ⟨rfl, hl, rfl⟩, fun _ => hd2, by simp⟩
  have gd : s1.adv.cur = .lbrack ∧ GDetail o Ext.rust s1.adv.adv d s4 ∨
      d = (anonName s1.adv.2, [], leaf .map, none) ∧ s4 = (s1.adv.1, s1.adv.2 + 1) := by
    cases det
    · have := (opt_eq h3).2 rfl; subst this
      simp only [Bool.false_eq_true, ite_false] at h4
      peel h4 => a s8 h8
      simp only [anonFresh, R.ok.injEq] at h8; obtain ⟨rfl, rfl⟩ := h8
      obtain ⟨rfl, rfl⟩ := pure_ok h4; exact .inr ⟨rfl, rfl⟩
    · obtain ⟨hb, rfl⟩ := (opt_eq h3).1 rfl
      simp only [ite_true] at h4
      exact .inl ⟨hb, detailP_sound o F cl _ _ _ h4⟩
  have go : out = true ∧ s4.adv.cur = .gt ∧ s6 = s4.adv.adv ∨ out = false ∧ s6 = s4.adv := by
    cases out
    · exact .inr ⟨rfl, (opt_eq h6).2 rfl⟩
    · obtain ⟨hg, rfl⟩ := (opt_eq h6).1 rfl; exact .inl ⟨rfl, hg, rfl⟩
  have gn := nodeP_sound o F _ _ _ h7
  have key := GRel.mk (o := o) (x := Ext.rust) (src := src) (d := d) gi.1 gi.2.1 gi.2.2 gd hd5 go gn
  revert key h
  cases inc <;> cases out <;> intro h key
  all_goals (simp only [beq_self_eq_true, ite_true, Bool.false_eq_true, ite_false] at h key)
  all_goals first
    | (obtain ⟨rfl, rfl⟩ := pure_ok h; simpa using key)
    | (split at h
       · simp at h
       · obtain ⟨rfl, rfl⟩ := pure_ok h; simpa using key)

end FalkorParserGrammar.C
