import PlannerBuild.Expr
/-
# Anchors of a var-length walk (#2558, PR #3593): the optimizer helpers `plan_match` reads

| here | there |
| --- | --- |
| `varsOf`, `scopedVars` | `collect_expr_scoped_variables` optimizer/mod.rs:105-113 (BFS, `(scope_id, id)`) |
| `singleVarConj`, `singleVarConjL` | `collect_single_variable_conjuncts` optimizer/mod.rs:120-154 and its `walk` :125-153 |
| `endpointScanScore` | `endpoint_scan_score` optimizer/mod.rs:160-165 |
| `walkScansTo` | `walk_scans_to_endpoint` optimizer/mod.rs:171-176 |
| `WI` | what `plan_match` derives from its WHERE: `filter_vars`, `anchor_vars`, `has_pattern_expr` (mod.rs:1119-1124) |

A `HashSet` is modelled by a list (membership is all the planner reads).
-/
namespace PlannerBuild.E

/-- The variables of an expression, in walk order (with repeats). -/
def varsOf (e : Ex) : List V :=
  (nodes e).filterMap fun d => match d with | .var v => some v | _ => none

/-- `collect_expr_scoped_variables`: every variable the expression reads, keyed by
`(scope_id, id)` (here `V`, whose equality is on both). -/
def scopedVars (e : Ex) : List V := varsOf e

theorem mem_varsOf (e : Ex) (v : V) : v ∈ varsOf e ↔ D.var v ∈ nodes e := by
  unfold varsOf; rw [List.mem_filterMap]
  constructor
  · rintro ⟨d, hd, h⟩; cases d <;> simp at h; subst h; exact hd
  · intro h; exact ⟨_, h, rfl⟩

/-- **PROVEN**: the set is exactly the variables of the tree, scope included. -/
theorem mem_scopedVars (e : Ex) (v : V) : v ∈ scopedVars e ↔ D.var v ∈ nodes e := mem_varsOf e v

/-- Same id, other scope: two entries (the id-only `collect_expr_variables` merges them). -/
theorem scopedVars_keeps_scope :
    scopedVars (.node .eq [Ex.v ⟨3, 0⟩, Ex.v ⟨3, 1⟩]) = [⟨3, 0⟩, ⟨3, 1⟩] := rfl

/-- The `vars.len() == 1` test of a conjunct (a set of exactly one variable). -/
def single : List V → List V
  | [] => []
  | v :: rest => if rest.all (· == v) then [v] else []

theorem mem_single {l : List V} {v : V} : v ∈ single l ↔ v ∈ l ∧ ∀ w ∈ l, w = v := by
  cases l with
  | nil => simp [single]
  | cons a rest =>
    unfold single
    by_cases h : rest.all (· == a) = true
    · simp only [h, ite_true, List.mem_singleton]
      have h' : ∀ w ∈ rest, w = a := by simpa using h
      constructor
      · rintro rfl; refine ⟨by simp, ?_⟩; intro w hw
        rcases List.mem_cons.1 hw with rfl | hw
        · rfl
        · exact h' w hw
      · rintro ⟨_, hall⟩; exact (hall a (by simp)).symm
    · simp only [h, Bool.false_eq_true, ite_false, List.not_mem_nil, false_iff]
      rintro ⟨_, hall⟩
      apply h
      have ha : a = v := hall a (by simp)
      subst ha
      simpa using fun w hw => hall w (by simp [hw])

/- `collect_single_variable_conjuncts` / `walk`: descend through `AND` and `Paren`; any other
node is a conjunct, kept when it holds no pattern and reads exactly one variable. -/
mutual
def singleVarConj : Ex → List V
  | .node d cs => match d with
    | .and | .paren => singleVarConjL cs
    | _ => if hasPatternExpr (.node d cs) then [] else single (varsOf (.node d cs))
def singleVarConjL : List Ex → List V
  | [] => []
  | c :: cs => singleVarConj c ++ singleVarConjL cs
end

/-- `c` is a top-level conjunct of `e` (reached through `AND` / `Paren` only). -/
inductive Conj : Ex → Ex → Prop
  | self (d : D) (cs : List Ex) (hd : d ≠ .and ∧ d ≠ .paren) : Conj (.node d cs) (.node d cs)
  | and (cs : List Ex) (c x : Ex) : c ∈ cs → Conj c x → Conj (.node .and cs) x
  | paren (cs : List Ex) (c x : Ex) : c ∈ cs → Conj c x → Conj (.node .paren cs) x

mutual
theorem singleVarConj_sound : ∀ (e : Ex) (v : V), v ∈ singleVarConj e →
    ∃ c, Conj e c ∧ hasPatternExpr c = false ∧ v ∈ varsOf c ∧ ∀ w ∈ varsOf c, w = v
  | .node d cs, v, h => by
    cases d with
    | and =>
      obtain ⟨c, hc, x, hx, h1, h2, h3⟩ := singleVarConjL_sound cs v (by simpa [singleVarConj] using h)
      exact ⟨x, .and cs c x hc hx, h1, h2, h3⟩
    | paren =>
      obtain ⟨c, hc, x, hx, h1, h2, h3⟩ := singleVarConjL_sound cs v (by simpa [singleVarConj] using h)
      exact ⟨x, .paren cs c x hc hx, h1, h2, h3⟩
    | _ =>
      simp only [singleVarConj] at h
      split at h
      · simp at h
      · rename_i hp
        obtain ⟨h1, h2⟩ := mem_single.1 h
        exact ⟨_, .self _ cs (by simp), by simpa using hp, h1, h2⟩
theorem singleVarConjL_sound : ∀ (cs : List Ex) (v : V), v ∈ singleVarConjL cs →
    ∃ c ∈ cs, ∃ x, Conj c x ∧ hasPatternExpr x = false ∧ v ∈ varsOf x ∧ ∀ w ∈ varsOf x, w = v
  | [], v, h => by simp [singleVarConjL] at h
  | c :: cs, v, h => by
    simp only [singleVarConjL, List.mem_append] at h
    rcases h with h | h
    · obtain ⟨x, hx, h1, h2, h3⟩ := singleVarConj_sound c v h
      exact ⟨c, by simp, x, hx, h1, h2, h3⟩
    · obtain ⟨c', hc', x, hx, h1, h2, h3⟩ := singleVarConjL_sound cs v h
      exact ⟨c', by simp [hc'], x, hx, h1, h2, h3⟩
end

theorem nodes_conj : ∀ {e x : Ex}, Conj e x → ∀ d ∈ nodes x, d ∈ nodes e
  | _, _, .self _ _ _, d, h => h
  | _, _, .and cs c x hc hx, d, h => by
    have := nodes_conj hx d h
    simp only [nodes, List.mem_cons]
    exact Or.inr (mem_nodesL_of cs c hc d this)
  | _, _, .paren cs c x hc hx, d, h => by
    have := nodes_conj hx d h
    simp only [nodes, List.mem_cons]
    exact Or.inr (mem_nodesL_of cs c hc d this)
where
  mem_nodesL_of : ∀ (cs : List Ex) (c : Ex), c ∈ cs → ∀ d ∈ nodes c, d ∈ nodesL cs
    | [], _, h, _, _ => by simp at h
    | c' :: cs, c, h, d, hd => by
      simp only [nodesL, List.mem_append]
      rcases List.mem_cons.1 h with rfl | h
      · exact Or.inl hd
      · exact Or.inr (mem_nodesL_of cs c h d hd)

/-- **PROVEN** (`collect_single_variable_conjuncts` is sound): every anchor is the only
variable of some pattern-free top-level conjunct of the WHERE (so that conjunct can run as
soon as the anchor is scanned), and in particular a WHERE variable (`anchor_vars ⊆ filter_vars`). -/
theorem singleVarConj_spec (e : Ex) (v : V) (h : v ∈ singleVarConj e) :
    (∃ c, Conj e c ∧ hasPatternExpr c = false ∧ v ∈ varsOf c ∧ ∀ w ∈ varsOf c, w = v) ∧
    v ∈ scopedVars e := by
  obtain ⟨c, hc, h1, h2, h3⟩ := singleVarConj_sound e v h
  refine ⟨⟨c, hc, h1, h2, h3⟩, ?_⟩
  rw [mem_scopedVars]
  exact nodes_conj hc _ ((mem_varsOf c v).1 h2)

/-- `a.x = 1 AND b.y = 2`: both are anchors. -/
theorem singleVarConj_and :
    singleVarConj (.node .and [.node .eq [.node (.prop "x") [Ex.v ⟨0, 0⟩], .node (.const (.int 1)) []],
      .node .eq [.node (.prop "y") [Ex.v ⟨1, 0⟩], .node (.const (.int 2)) []]]) = [⟨0, 0⟩, ⟨1, 0⟩] := by
  decide

/-- `p.id = length(r)` reads two variables: it anchors nothing (the comment at mod.rs:2012-2013). -/
theorem singleVarConj_two_vars :
    singleVarConj (.node .eq [.node (.prop "id") [Ex.v ⟨0, 0⟩], .node .length [Ex.v ⟨1, 0⟩]]) = [] := by
  decide

/-- `a.x = 1 OR a.y = 2` is one conjunct over `a`; an `OR` over two variables anchors nothing. -/
theorem singleVarConj_or :
    singleVarConj (.node .or [.node (.prop "x") [Ex.v ⟨0, 0⟩], .node (.prop "y") [Ex.v ⟨0, 0⟩]]) = [⟨0, 0⟩] ∧
    singleVarConj (.node .or [.node (.prop "x") [Ex.v ⟨0, 0⟩], .node (.prop "y") [Ex.v ⟨1, 0⟩]]) = [] := by
  decide

/-- A conjunct holding a pattern anchors nothing, even over one variable. -/
theorem singleVarConj_pattern (g : QG) :
    singleVarConj (.node .not [.node (.pat g) [], Ex.v ⟨0, 0⟩]) = [] := by
  simp [singleVarConj, hasPatternExpr, hasPatternExprL]

/-! ## Scores -/

/-- `endpoint_scan_score`: a predicate on the node counts 2, a label 1. -/
def endpointScanScore (filtered labelled : Bool) : Nat :=
  (if filtered then 2 else 0) + (if labelled then 1 else 0)

/-- `walk_scans_to_endpoint`: scan `to` only when it strictly outscores `from`. -/
def walkScansTo (fromScore toScore : Nat) : Bool := decide (fromScore < toScore)

/-- **PROVEN**: the score is at most 3; a filtered endpoint outscores any unfiltered one; a
tie keeps the pattern's direction (scan `from`). -/
theorem endpointScanScore_spec (f l f' l' : Bool) :
    endpointScanScore f l ≤ 3 ∧
    (f = true → f' = false → endpointScanScore f' l' < endpointScanScore f l) ∧
    (f = f' → l = true → l' = false → endpointScanScore f' l' < endpointScanScore f l) := by
  cases f <;> cases l <;> cases f' <;> cases l' <;> decide

theorem walkScansTo_spec (a b : Nat) : walkScansTo a b = true ↔ a < b := by simp [walkScansTo]

theorem walkScansTo_tie (a : Nat) : walkScansTo a a = false := by simp [walkScansTo]

/-! ## What `plan_match` reads off its WHERE -/

/-- `filter_vars`, `anchor_vars`, and whether the WHERE holds a pattern. `boundBefore`
(`bound_before_clause`) and `single` (`single_component`) are set by `planMatch` for the clause. -/
structure WI where
  fv : List V := []
  anchors : List V := []
  patternWhere : Bool := false
  boundBefore : Bool := false
  single : Bool := false

/-- mod.rs:2008-2022 and the `has_pattern_expr` test of mod.rs:1119-1124. -/
def WI.ofWhere : Option Ex → WI
  | none => {}
  | some f => { fv := scopedVars f, anchors := singleVarConj f, patternWhere := hasPatternExpr f }

/-- **PROVEN**: every anchor is a WHERE variable. -/
theorem WI.ofWhere_anchors_sub (f : Option Ex) (v : V) (h : v ∈ (WI.ofWhere f).anchors) :
    v ∈ (WI.ofWhere f).fv := by
  cases f with
  | none => simp [WI.ofWhere] at h
  | some f => exact (singleVarConj_spec f v h).2

end PlannerBuild.E
