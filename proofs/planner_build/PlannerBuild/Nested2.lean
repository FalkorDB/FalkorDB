import PlannerBuild.Nested
/-
# Comprehension sub-plans and the clause / WHERE wrappers

* `build_shape`: `build_pattern_comprehension_plan` (mod.rs:1406-1488) is
  `Aggregate(collect(result, acc))` over the WHERE `Filter` over one `Apply` per
  nested comprehension (in order, the first innermost) over the `PathBuilder`
  over `plan_match(pattern)`; `acc` is minted in the comprehension's scope
  after the nested ones'; `visited` is restored.
* `interleave_length`: an existential pattern's path has `n` nodes and `n-1` rels.
* `chain_*`: `extract_clause_expr_comprehensions` (mod.rs:1309-1340) builds
  `Apply(…Apply(sub_k)…, sub_1)`: the innermost Apply single-child (stitching
  fills it, `descend_clause_expr_applies`), the first comprehension outermost.
* `extractFilter_clean`: whatever `extract_filter_comprehensions`
  (mod.rs:1364-1402) returns satisfies `plan_filter`'s precondition
  `needsExtraction · .semiApply = false` (FilterPlan.lean).
* `extract_noPat`: in Collect/Exists mode no pattern is left at all.
-/
namespace PlannerBuild.E

/-- Nested sub-plans, built left to right with the state threaded. -/
def buildPlans (st : PSt) : List EC → List CP × PSt
  | [] => ([], st)
  | i :: is =>
    let p := build st i
    let q := buildPlans p.2 is
    (p.1 :: q.1, q.2)

theorem buildL_eq (st : PSt) (sub : CP) (l : List EC) :
    buildL st sub l = ((buildPlans st l).1.foldl .apply sub, (buildPlans st l).2) := by
  induction l generalizing st sub with
  | nil => rfl
  | cons i is ih => simp only [buildL, buildPlans, List.foldl_cons]; rw [ih]

theorem build_shape (st : PSt) (var : V) (g : QG) (wh : Option Ex) (res : Ex)
    (path : Option (V × List V)) (nested : List EC) :
    let b := buildPlans { st with visited := st.visited ++ g.vars } nested
    build st (.mk var g wh res path nested) =
      (.agg var ⟨b.2.lens var.scope, var.scope⟩ res
          (withFilter wh (b.1.foldl .apply (withPath path (.matchG g st.visited)))),
       { b.2 with visited := st.visited,
                  lens := fun t => if t = var.scope then b.2.lens var.scope + 1 else b.2.lens t,
                  minted := b.2.minted ++ [⟨b.2.lens var.scope, var.scope⟩] }) := by
  intro b
  simp only [build, buildL_eq, fresh]
  rfl

theorem interleave_length : ∀ (ns rs : List V), rs.length + 1 = ns.length →
    (interleave ns rs).length = ns.length + rs.length
  | [], _, h => by simp at h
  | [n], [], _ => rfl
  | n :: m :: ns, [], h => by simp at h
  | n :: ns, r :: rs, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    simp only [interleave, List.length_cons]
    rw [interleave_length ns rs h]; omega

theorem interleave_nodes : ∀ (ns rs : List V) (i : Nat), i < ns.length → rs.length + 1 = ns.length →
    (interleave ns rs)[2 * i]? = ns[i]?
  | [], _, i, h, _ => by simp at h
  | [n], [], 0, _, _ => rfl
  | [n], [], i + 1, h, _ => by simp at h
  | n :: m :: ns, [], _, _, h => by simp at h
  | n :: ns, r :: rs, 0, _, _ => rfl
  | n :: ns, r :: rs, i + 1, h, hl => by
    simp only [List.length_cons, Nat.add_right_cancel_iff, Nat.add_lt_add_iff_right] at h hl
    have : 2 * (i + 1) = (2 * i) + 1 + 1 := by omega
    simp only [interleave, this, List.getElem?_cons_succ]
    exact interleave_nodes ns rs i h hl

/-! ## `extract_clause_expr_comprehensions` -/

/-- The Apply chain, built innermost first from the reversed list. -/
def chainOf (st : PSt) (chain : Option CP) : List EC → Option CP × PSt
  | [] => (chain, st)
  | c :: cs =>
    let b := build st c
    chainOf b.2 (some (match chain with
      | some inner => .apply inner b.1
      | none => .applyIn b.1)) cs

/-- One expression of the clause (mod.rs:1319-1331): skipped when it has no
pattern, else rewritten in Collect mode in the given scope or the pattern's. -/
def clauseStep (scope : Option Nat) (x : Ex × PSt × List EC) : Ex × PSt × List EC :=
  match patternExprScope x.1 with
  | none => x
  | some ps => extract (scope.getD ps) x.2.1 x.2.2 .collect x.1

def extractClause (scope : Option Nat) (st : PSt) (exprs : List Ex) : List Ex × Option CP × PSt :=
  let step := fun (acc : List Ex × PSt × List EC) (e : Ex) =>
    let r := clauseStep scope (e, acc.2.1, acc.2.2)
    (acc.1 ++ [r.1], r.2.1, r.2.2)
  let x := exprs.foldl step ([], st, [])
  let ch := chainOf x.2.1 none x.2.2.reverse
  (x.1, ch.1, { ch.2 with visited := ch.2.visited ++ x.2.2.map EC.var })

theorem chainOf_some (st : PSt) (ch : Option CP) (l : List EC) (h : l ≠ []) :
    (chainOf st ch l).1.isSome := by
  induction l generalizing st ch with
  | nil => exact absurd rfl h
  | cons c cs ih =>
    simp only [chainOf]
    cases cs with
    | nil => simp [chainOf]
    | cons d ds => exact ih _ _ (by simp)

theorem chainOf_none (st : PSt) : (chainOf st none []).1 = none := rfl

/-- The innermost Apply (built first, for the last comprehension) has a single
child: the slot stitching fills with the preceding clause. -/
def innermost : CP → Option CP
  | .apply x _ => innermost x
  | .applyIn s => some (.applyIn s)
  | _ => none

theorem chainOf_keeps_innermost (st : PSt) (inner : CP) (s : CP) (h : innermost inner = some (.applyIn s))
    (l : List EC) : ∃ p, (chainOf st (some inner) l).1 = some p ∧ innermost p = some (.applyIn s) := by
  induction l generalizing st inner with
  | nil => exact ⟨inner, rfl, h⟩
  | cons d ds ih =>
    simp only [chainOf]
    exact ih _ (.apply inner (build st d).1) (by simpa [innermost] using h)

theorem chainOf_innermost (st : PSt) (l : List EC) (c : EC) :
    ∃ p, (chainOf st none (c :: l)).1 = some p ∧ ∃ s, innermost p = some (.applyIn s) := by
  simp only [chainOf]
  obtain ⟨p, h1, h2⟩ := chainOf_keeps_innermost (build st c).2 (.applyIn (build st c).1) _ rfl l
  exact ⟨p, h1, _, h2⟩

theorem extractClause_visited (scope : Option Nat) (st : PSt) (exprs : List Ex) (c : EC)
    (h : c ∈ (exprs.foldl (fun (acc : List Ex × PSt × List EC) (e : Ex) =>
      let r := clauseStep scope (e, acc.2.1, acc.2.2); (acc.1 ++ [r.1], r.2.1, r.2.2)) ([], st, [])).2.2) :
    c.var ∈ (extractClause scope st exprs).2.2.visited := by
  simp only [extractClause]
  exact List.mem_append_right _ (List.mem_map_of_mem h)

/-! ## In Collect / Exists mode nothing pattern-like survives -/

mutual
theorem extract_noPat (sc : Nat) : ∀ (st : PSt) acc m (e : Ex), m ≠ .semiApply → pcOK e = true →
    hasPatternExpr (extract sc st acc m e).1 = false
  | st, acc, m, .node d cs, hm, hp => by
    have hpl : pcOKL cs = true := by cases d <;> simp_all [pcOK]
    have hd : m.descend d = m := by simp [Mode.descend, hm]
    have hL := extractL_noPat sc st acc d (binds d).1 (binds d).2 (m.descend d) 0 cs (by rw [hd]; exact hm) hpl
    cases d
    case patComp g =>
      match cs, hp with
      | w :: r :: rest, _ => simp only [extract]; exact hoist_noPat _ _ _ _
      | [], hp => simp [pcOK] at hp
      | [_], hp => simp [pcOK] at hp
    case pat g =>
      simp only [extract, hm, ↓reduceIte]
      split
      · exact gt0_noPat _ (hoist_noPat _ _ _ _)
      · exact hoist_noPat _ _ _ _
    all_goals (simp only [extract, hasPatternExpr]; exact hL)
theorem extractL_noPat (sc : Nat) : ∀ (st : PSt) acc d bound first cm i (l : List Ex),
    cm ≠ .semiApply → pcOKL l = true →
    hasPatternExprL (extractL sc st acc d bound first cm i l).1 = false
  | st, acc, d, bound, first, cm, i, [], _, _ => rfl
  | st, acc, d, bound, first, cm, i, c :: cs, hm, hp => by
    simp only [pcOKL, Bool.and_eq_true] at hp
    simp only [extractL, hasPatternExprL, Bool.or_eq_false_iff]
    refine ⟨?_, extractL_noPat sc _ _ d bound first cm (i + 1) cs hm hp.2⟩
    apply extract_noPat sc _ _ _ c _ hp.1
    split <;> simp_all
end

/-- The rewritten clause expressions hold no pattern or comprehension. -/
theorem clauseStep_noPat (scope : Option Nat) (x : Ex × PSt × List EC) (hp : pcOK x.1 = true)
    (hn : PatsNamed (nodes x.1)) : hasPatternExpr (clauseStep scope x).1 = false := by
  unfold clauseStep
  split
  · rename_i h
    rw [patternExprScope_eq _ hn, List.findSome?_eq_none_iff] at h
    rw [hasPatternExpr_eq, List.any_eq_false]
    intro d hd hpat
    have := hn d hd hpat
    rw [h d hd] at this; simp at this
  · exact extract_noPat _ _ _ _ _ (by simp) hp

/-! ## `extract_filter_comprehensions` -/

def extractFilter (st : PSt) (e : Ex) (sc : Nat) : Ex × List CP × PSt :=
  if needsExtraction e .semiApply = false then (e, [], st)
  else
    let x := extract sc st [] .semiApply e
    let ps := buildPlans x.2.1 x.2.2
    (x.1, ps.1, { ps.2 with visited := ps.2.visited ++ x.2.2.map EC.var })

/-- Whatever comes back is a valid input for `plan_filter`'s decomposition. -/
theorem extractFilter_clean (st : PSt) (e : Ex) (sc : Nat) (hp : pcOK e = true) :
    needsExtraction (extractFilter st e sc).1 .semiApply = false := by
  unfold extractFilter
  split
  · assumption
  · exact extract_clean sc st [] .semiApply e hp

/-- `plan_filter` applies the comprehension sub-plans below the predicate. -/
def applyAll (res : CP) (ps : List CP) : CP := ps.foldl .apply res

end PlannerBuild.E
