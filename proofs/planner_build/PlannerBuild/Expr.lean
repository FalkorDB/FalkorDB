/-
# Expressions as the planner sees them (`ExprIR<Variable>` trees)

`Ex` is `DynTree<ExprIR<Variable>>` (parser/ast.rs:155-330): a node payload `D`
and an ordered child list. Variables are `(id, scope_id)` pairs (`V`), compared
exactly as the planner's `HashSet<(u32, u32)>` / `is_loop_var` do. A pattern
payload (`QG`, `QueryGraph`) keeps what the planner reads of it: its
`variables()` (nodes, then relationships, then paths — ast.rs:761-767) and the
variables its inline attribute maps mention (`attrVars`; the attribute trees
themselves are not re-modelled).

| here | there |
| --- | --- |
| `containsInlineVar` | `Planner::contains_inline_var` mod.rs:933-940 |
| `hasPatternExpr` | `Planner::has_pattern_expr` mod.rs:944-949 |
| `patternExprScope` | `Planner::pattern_expr_scope` mod.rs:1345-1352 |
| `Mode`, `Mode.descend` | `PatternMode`, `PatternMode::descend` mod.rs:782-811 |
| `needsExtraction` | `extract_filter_comprehensions::needs_extraction` mod.rs:1369-1382 |
| `inlineAttrsToFilter` | `inline_attrs_to_filter` mod.rs:543-571 |
| `hasLabelsFilter` | `has_labels_filter` mod.rs:696-709 |
| `patternExprVariables` | `Planner::pattern_expr_variables` mod.rs:1250-1282 |
-/
namespace PlannerBuild.E

/-- `Variable` as the planner compares it: `(id, scope_id)`. -/
structure V where
  id : Nat
  scope : Nat
  deriving DecidableEq, Repr

/-- `QueryGraph`: identity, `variables()` in order, variables read by its inline maps. -/
structure QG where
  gid : Nat
  vars : List V
  attrVars : List V := []
  /-- node aliases, in pattern order -/
  nodes : List V := []
  /-- relationship aliases, in pattern order -/
  rels : List V := []
  deriving DecidableEq, Repr

inductive C
  | null | bool (b : Bool) | int (i : Int) | str (s : String)
  deriving DecidableEq, Repr

/-- `ExprIR` payloads (the ones the planner distinguishes; `other` for the rest). -/
inductive D
  | const (c : C)
  | var (v : V)
  | and | or | not | paren | xor | eq | gt | length | list
  | prop (k : String)
  | func (name : String)
  | pat (g : QG)
  | patComp (g : QG)
  | listComp (v : V)
  | quant (v : V)
  | reduce (acc it : V)
  | nested (id : Nat) (res : V)
  | other (tag : Nat)
  deriving DecidableEq, Repr

/-- `DynTree<ExprIR<Variable>>`. -/
inductive Ex
  | node (d : D) (cs : List Ex)
  deriving Repr

instance : Inhabited Ex := ⟨.node (.other 0) []⟩

def Ex.d : Ex → D | .node d _ => d
def Ex.cs : Ex → List Ex | .node _ cs => cs

def tt : Ex := .node (.const (.bool true)) []
def Ex.v (v : V) : Ex := .node (.var v) []

/-- Root is the literal `true` (`matches!(root, Constant(Bool(true)))`). -/
def isTT : Ex → Bool
  | .node (.const (.bool true)) _ => true
  | _ => false

/- Pre-order list of payloads (what a BFS/DFS `walk_with(..).any` inspects). -/
mutual
def nodes : Ex → List D
  | .node d cs => d :: nodesL cs
def nodesL : List Ex → List D
  | [] => []
  | c :: cs => nodes c ++ nodesL cs
end

mutual
def anyN (p : D → Bool) : Ex → Bool
  | .node d cs => p d || anyL p cs
def anyL (p : D → Bool) : List Ex → Bool
  | [] => false
  | c :: cs => anyN p c || anyL p cs
end

mutual
theorem anyN_eq (p : D → Bool) : ∀ e, anyN p e = (nodes e).any p
  | .node d cs => by simp [anyN, nodes, anyL_eq p cs]
theorem anyL_eq (p : D → Bool) : ∀ l, anyL p l = (nodesL l).any p
  | [] => rfl
  | c :: cs => by simp [anyL, nodesL, anyN_eq p c, anyL_eq p cs]
end

/-! ## `contains_inline_var` (mod.rs:933-940): BFS `any` over the subtree, id only -/

def isInl (ids : List Nat) : D → Bool
  | .var v => ids.contains v.id
  | _ => false

def containsInlineVar (ids : List Nat) (e : Ex) : Bool := anyN (isInl ids) e

theorem containsInlineVar_iff (ids : List Nat) (e : Ex) :
    containsInlineVar ids e = true ↔ ∃ v, D.var v ∈ nodes e ∧ v.id ∈ ids := by
  unfold containsInlineVar
  rw [anyN_eq, List.any_eq_true]
  constructor
  · rintro ⟨d, hd, h⟩
    cases d with
    | var v => exact ⟨v, hd, by simpa [isInl] using h⟩
    | _ => simp [isInl] at h
  · rintro ⟨v, hv, h⟩
    exact ⟨_, hv, by simpa [isInl] using h⟩

/-- The check is by id only: a variable of another scope with the same id
counts as an inline-pattern variable. -/
theorem containsInlineVar_ignores_scope (i s s' : Nat) :
    containsInlineVar [i] (Ex.v ⟨i, s'⟩) = true ∧ s ≠ s' ∨ s = s' := by
  by_cases h : s = s'
  · exact Or.inr h
  · exact Or.inl ⟨by simp [containsInlineVar, Ex.v, anyN, isInl, anyL], h⟩

/-! ## `has_pattern_expr` (mod.rs:944-949) -/

def isPat : D → Bool
  | .pat _ | .patComp _ => true
  | _ => false

mutual
def hasPatternExpr : Ex → Bool
  | .node d cs => match d with
    | .pat _ | .patComp _ => true
    | _ => hasPatternExprL cs
def hasPatternExprL : List Ex → Bool
  | [] => false
  | c :: cs => hasPatternExpr c || hasPatternExprL cs
end

mutual
theorem hasPatternExpr_eq : ∀ e, hasPatternExpr e = (nodes e).any isPat
  | .node d cs => by
    cases d <;> simp [hasPatternExpr, nodes, isPat, hasPatternExprL_eq cs]
theorem hasPatternExprL_eq : ∀ l, hasPatternExprL l = (nodesL l).any isPat
  | [] => rfl
  | c :: cs => by simp [hasPatternExprL, nodesL, hasPatternExpr_eq c, hasPatternExprL_eq cs]
end

/-! ## `pattern_expr_scope` (mod.rs:1345-1352) -/

def patScope : D → Option Nat
  | .pat g | .patComp g => g.vars.head?.map V.scope
  | _ => none

mutual
def patternExprScope : Ex → Option Nat
  | .node d cs => match d with
    | .pat g | .patComp g => g.vars.head?.map V.scope
    | _ => patternExprScopeL cs
def patternExprScopeL : List Ex → Option Nat
  | [] => none
  | c :: cs => match patternExprScope c with
    | some s => some s
    | none => patternExprScopeL cs
end

/-- Every pattern payload has at least one variable (the binder names
anonymous pattern elements, so `variables()` is never empty). -/
def PatsNamed (l : List D) : Prop := ∀ d ∈ l, isPat d = true → (patScope d).isSome

theorem findSome_patScope_ne (l : List D) (h : (l.findSome? patScope).isSome = false) :
    l.findSome? patScope = none := by
  cases hh : l.findSome? patScope <;> simp_all

mutual
/-- The scope is the scope of the first variable of the first pattern in
pre-order (when patterns are named). -/
theorem patternExprScope_eq : ∀ e, PatsNamed (nodes e) →
    patternExprScope e = (nodes e).findSome? patScope
  | .node d cs, h => by
    cases d with
    | pat g =>
      have := h (.pat g) (by simp [nodes]) rfl
      simp [patternExprScope, nodes, List.findSome?, patScope]
      cases hg : g.vars.head? <;> simp_all [patScope]
    | patComp g =>
      have := h (.patComp g) (by simp [nodes]) rfl
      simp [patternExprScope, nodes, List.findSome?, patScope]
      cases hg : g.vars.head? <;> simp_all [patScope]
    | _ =>
      simp only [patternExprScope, nodes, List.findSome?, patScope]
      exact patternExprScopeL_eq cs (fun d hd => h d (by simp [nodes, hd]))
theorem patternExprScopeL_eq : ∀ l, PatsNamed (nodesL l) →
    patternExprScopeL l = (nodesL l).findSome? patScope
  | [], _ => rfl
  | c :: cs, h => by
    have h1 : PatsNamed (nodes c) := fun d hd => h d (by simp [nodesL, hd])
    have h2 : PatsNamed (nodesL cs) := fun d hd => h d (by simp [nodesL, hd])
    simp only [patternExprScopeL, nodesL, List.findSome?_append]
    rw [patternExprScope_eq c h1, patternExprScopeL_eq cs h2]
    cases (nodes c).findSome? patScope <;> rfl
end

/-! ## `PatternMode` (mod.rs:782-811) and `needs_extraction` (mod.rs:1369-1382) -/

inductive Mode | collect | semiApply | exists_
  deriving DecidableEq, Repr

/-- The connectives `PatternMode::descend` keeps SemiApply under (mod.rs:795-806).
Since #3335 `Paren` is not among them: the binder removes every `Paren`
(binder.rs:1955-1965), so none reaches the planner. -/
def isConn : D → Bool
  | .and | .or | .not => true
  | _ => false

def Mode.descend (m : Mode) (parent : D) : Mode :=
  if m = .semiApply ∧ isConn parent = false then .exists_ else m

theorem descend_table (m : Mode) (p : D) :
    m.descend p = (if m = .semiApply then (if isConn p then .semiApply else .exists_) else m) := by
  unfold Mode.descend; cases m <;> cases h : isConn p <;> simp [h]

mutual
def needsExtraction : Ex → Mode → Bool
  | .node d cs, m => match d with
    | .patComp _ => true
    | .pat _ => m == .exists_
    | _ => needsExtractionL cs (m.descend d)
def needsExtractionL : List Ex → Mode → Bool
  | [], _ => false
  | c :: cs, m => needsExtraction c m || needsExtractionL cs m
end

mutual
/-- Outside SemiApply mode, extraction is needed exactly when the tree has a
pattern or pattern comprehension. -/
theorem needsExtraction_exists : ∀ e, needsExtraction e .exists_ = hasPatternExpr e
  | .node d cs => by
    have hd : Mode.exists_.descend d = .exists_ := by simp [Mode.descend]
    cases d <;> simp [needsExtraction, hasPatternExpr, hd, needsExtractionL_exists cs]
theorem needsExtractionL_exists : ∀ l, needsExtractionL l .exists_ = hasPatternExprL l
  | [] => rfl
  | c :: cs => by
    simp [needsExtractionL, hasPatternExprL, needsExtraction_exists c, needsExtractionL_exists cs]
end

/-- In SemiApply mode a bare pattern directly under connectives needs nothing. -/
theorem needs_semi_pat (g : QG) : needsExtraction (.node (.pat g) []) .semiApply = false := rfl
theorem needs_semi_xor (g : QG) (b : Ex) :
    needsExtraction (.node .xor [.node (.pat g) [], b]) .semiApply = true := by
  simp [needsExtraction, needsExtractionL, Mode.descend, isConn]

/-! ## `ExprIR::and_of` / `or_of` / `flat` (ast.rs:342-380, #3335)

`flat op operands identity`: an `op` node over `operands`, where an operand whose
root is `op` itself contributes its children in place (orx_tree `take_out` keeps
them at the operand's position, in order). A lone operand that is not an `op`
is returned unchanged (the early return); no operands give the identity
constant; one operand left after splicing is returned on its own. Semantics in
`AndOf.lean`. -/

/-- What one operand contributes: its children when its root is `op`, else itself. -/
def spliceOp (op : D) : Ex → List Ex
  | .node d cs => if d = op then cs else [.node d cs]

/-- `discriminant(e) == discriminant(op)` (ast.rs:361; `op` is `And` or `Or`, payload-free). -/
def isOp (op : D) (e : Ex) : Bool := decide (e.d = op)

/-- The tree built after the early return (ast.rs:365-378). -/
def flatBody (op : D) (identity : Bool) (xs : List Ex) : Ex :=
  match xs.flatMap (spliceOp op) with
  | [] => .node (.const (.bool identity)) []
  | [c] => c
  | cs => .node op cs

/-- `ExprIR::flat` (ast.rs:355-380). -/
def flat (op : D) (identity : Bool) (xs : List Ex) : Ex :=
  match xs with
  | [x] => if isOp op x then flatBody op identity [x] else x
  | xs => flatBody op identity xs

/-- `ExprIR::and_of` (ast.rs:342-344). -/
def andOf (xs : List Ex) : Ex := flat .and true xs
/-- `ExprIR::or_of` (ast.rs:348-350). -/
def orOf (xs : List Ex) : Ex := flat .or false xs

/-- The early return is only a shortcut: a lone non-`op` operand splices to itself. -/
theorem flat_eq_body (op : D) (identity : Bool) (xs : List Ex) :
    flat op identity xs = flatBody op identity xs := by
  unfold flat
  split
  · rename_i x
    split
    · rfl
    · rename_i h
      obtain ⟨d, cs⟩ := x
      have hd : ¬ (d = op) := by intro e; apply h; simp [isOp, Ex.d, e]
      simp [flatBody, spliceOp, hd]
  · rfl

theorem flat_nil (op : D) (identity : Bool) :
    flat op identity [] = .node (.const (.bool identity)) [] := rfl

/-- A lone operand of another kind is returned as it is. -/
theorem flat_single (op : D) (identity : Bool) (x : Ex) (h : x.d ≠ op) : flat op identity [x] = x := by
  obtain ⟨d, cs⟩ := x
  have : ¬ (d = op) := h
  simp [flat, isOp, Ex.d, this]

/-- A lone operand of the same kind with at least two children is returned as it is. -/
theorem flat_single_same (op : D) (identity : Bool) (cs : List Ex) (h : 2 ≤ cs.length) :
    flat op identity [.node op cs] = .node op cs := by
  match cs, h with
  | a :: b :: rest, _ => simp [flat, isOp, Ex.d, flatBody, spliceOp]

/-- With at least two operands after splicing, the result is the `op` node over all of them,
in order. -/
theorem flat_many (op : D) (identity : Bool) (xs : List Ex) (h : 2 ≤ (xs.flatMap (spliceOp op)).length) :
    flat op identity xs = .node op (xs.flatMap (spliceOp op)) := by
  rw [flat_eq_body]; unfold flatBody
  split
  · rename_i heq; rw [heq] at h; simp at h
  · rename_i heq; rw [heq] at h; simp at h
  · rfl

theorem spliceOp_same (op : D) (cs : List Ex) : spliceOp op (.node op cs) = cs := by
  simp [spliceOp]

/-- Operands that are not `op` nodes are never spliced (e.g. AND over `OR`/`XOR`/`NOT`). -/
theorem spliceOp_other (op : D) (x : Ex) (h : x.d ≠ op) : spliceOp op x = [x] := by
  obtain ⟨d, cs⟩ := x
  have : ¬ (d = op) := h
  simp [spliceOp, this]

/-- The Rust unit test `nested_and_is_flattened_in_order` (ast.rs:1489). -/
theorem andOf_nested_in_order (a b c d e : Ex) (ha : a.d ≠ .and) (hb : b.d ≠ .and) (hc : c.d ≠ .and)
    (hd : d.d ≠ .and) (he : e.d ≠ .and) :
    andOf [.node .and [a, b], c, .node .and [d, e]] = .node .and [a, b, c, d, e] := by
  unfold andOf
  rw [flat_many] <;>
    simp only [List.flatMap_cons, List.flatMap_nil, spliceOp_same, spliceOp_other _ _ hc,
      List.append_nil, List.cons_append, List.nil_append, List.length_cons] <;> omega

/-- `different_operators_are_not_merged` (ast.rs:1507): an OR operand of AND stays a child. -/
theorem andOf_keeps_or (a b c : Ex) (hc : c.d ≠ .and) :
    andOf [.node .or [a, b], c] = .node .and [.node .or [a, b], c] := by
  have ho : spliceOp .and (.node .or [a, b]) = [.node .or [a, b]] := spliceOp_other _ _ (by simp [Ex.d])
  unfold andOf
  rw [flat_many] <;>
    simp only [List.flatMap_cons, List.flatMap_nil, ho, spliceOp_other _ _ hc,
      List.append_nil, List.cons_append, List.nil_append, List.length_cons] <;> omega

/-! ## `inline_attrs_to_filter` (mod.rs:543-571) and `has_labels_filter` (mod.rs:696-709) -/

/-- `alias.k = value` -/
def attrEq (alias : V) (kv : String × Ex) : Ex :=
  .node .eq [.node (.prop kv.1) [Ex.v alias], kv.2]

/-- Since #3335: `None` for no attributes, else `ExprIR::and_of` of the equalities. -/
def inlineAttrsToFilter (alias : V) (attrs : List (String × Ex)) : Option Ex :=
  match attrs.map (attrEq alias) with
  | [] => none
  | fs => some (andOf fs)

/-- `and_of` changes nothing here: an equality is never an AND, so this is the
pre-#3335 `f` / `And(fs)` split. -/
theorem inlineAttrsToFilter_eq (alias : V) (attrs : List (String × Ex)) :
    inlineAttrsToFilter alias attrs =
      match attrs.map (attrEq alias) with
      | [] => none
      | [f] => some f
      | fs => some (.node .and fs) := by
  unfold inlineAttrsToFilter
  have hs : ∀ l : List (String × Ex), (l.map (attrEq alias)).flatMap (spliceOp .and) = l.map (attrEq alias) := by
    intro l; induction l with
    | nil => rfl
    | cons a l ih => simp [attrEq, spliceOp] at ih ⊢; exact ih
  match h : attrs.map (attrEq alias) with
  | [] => rfl
  | [f] =>
    have : f.d ≠ .and := by
      have : f ∈ attrs.map (attrEq alias) := by simp [h]
      obtain ⟨a, -, rfl⟩ := List.mem_map.1 this; simp [attrEq, Ex.d]
    simp [andOf, flat_single _ _ _ this]
  | f :: g :: rest =>
    have h2 := hs attrs; rw [h] at h2
    simp only [andOf]; rw [flat_many]
    · rw [h2]
    · rw [h2]; simp

/-- The filter's conjuncts are exactly the per-attribute equalities, in map order. -/
def conjuncts : Ex → List Ex
  | .node .and cs => cs
  | e => [e]

theorem inlineAttrsToFilter_none (alias : V) (attrs : List (String × Ex)) :
    inlineAttrsToFilter alias attrs = none ↔ attrs = [] := by
  rw [inlineAttrsToFilter_eq]
  cases attrs with
  | nil => simp
  | cons a as => cases as <;> simp

theorem inlineAttrsToFilter_conj (alias : V) (attrs : List (String × Ex)) (f : Ex)
    (h : inlineAttrsToFilter alias attrs = some f) : conjuncts f = attrs.map (attrEq alias) := by
  rw [inlineAttrsToFilter_eq] at h
  match attrs, h with
  | [a], h => cases h; rfl
  | a :: b :: rest, h => cases h; rfl

def hasLabelsFilter (v : V) (labels : List String) : Ex :=
  .node (.func "hasLabels") [Ex.v v, .node .list (labels.map fun l => .node (.const (.str l)) [])]

theorem hasLabelsFilter_shape (v : V) (labels : List String) :
    (hasLabelsFilter v labels).cs.length = 2 ∧ (hasLabelsFilter v labels).d = .func "hasLabels" ∧
    nodes (hasLabelsFilter v labels) =
      [.func "hasLabels", .var v, .list] ++ labels.map (fun l => .const (.str l)) := by
  refine ⟨rfl, rfl, ?_⟩
  simp only [hasLabelsFilter, nodes, Ex.v, nodesL, List.nil_append, List.cons_append,
    List.append_nil]
  congr 3
  induction labels with
  | nil => rfl
  | cons l ls ih => simp [nodesL, nodes, ih]

/-! ## `pattern_expr_variables` (mod.rs:1250-1282)

A stack walk: pop, record, push. Popping the children pushed last-first is a
pre-order walk with children right-to-left (`pevL` folds the tail first); a pattern's attribute maps are
pushed below its children, so they come after them. `add` keeps the first
occurrence of each `(id, scope)`. -/

def addV (acc : List V) (v : V) : List V := if acc.contains v then acc else acc ++ [v]

def addAll (acc : List V) (vs : List V) : List V := vs.foldl addV acc

def ownVars : D → List V
  | .var v => [v]
  | .pat g | .patComp g => g.vars
  | _ => []

def attrVarsD : D → List V
  | .pat g | .patComp g => g.attrVars
  | _ => []

mutual
def pev (acc : List V) : Ex → List V
  | .node d cs => addAll (pevL (addAll acc (ownVars d)) cs) (attrVarsD d)
/-- children right-to-left: the last child is popped first -/
def pevL (acc : List V) : List Ex → List V
  | [] => acc
  | c :: cs => pev (pevL acc cs) c
end

def patternExprVariables (e : Ex) : List V := pev [] e

/- Everything mentioned: own variables, pattern variables and attribute variables. -/
mutual
def mentioned : Ex → List V
  | .node d cs => ownVars d ++ mentionedL cs ++ attrVarsD d
def mentionedL : List Ex → List V
  | [] => []
  | c :: cs => mentioned c ++ mentionedL cs
end

theorem addV_mem (acc : List V) (v w : V) : w ∈ addV acc v ↔ w ∈ acc ∨ w = v := by
  unfold addV; split <;> simp_all

theorem addAll_mem (acc vs : List V) (w : V) : w ∈ addAll acc vs ↔ w ∈ acc ∨ w ∈ vs := by
  induction vs generalizing acc with
  | nil => simp [addAll]
  | cons v vs ih =>
    simp only [addAll, List.foldl_cons] at *
    rw [ih, addV_mem]; simp [or_assoc, or_comm (a := w = v)]

theorem addV_nodup (acc : List V) (v : V) (h : acc.Nodup) : (addV acc v).Nodup := by
  unfold addV; split
  · exact h
  · rename_i hc
    rw [List.nodup_append]; refine ⟨h, by simp, ?_⟩
    intro a ha b hb; simp at hb; subst hb; intro e; subst e; simp_all

theorem addAll_nodup (acc vs : List V) (h : acc.Nodup) : (addAll acc vs).Nodup := by
  induction vs generalizing acc with
  | nil => exact h
  | cons v vs ih => exact ih _ (addV_nodup _ _ h)

mutual
theorem pev_mem (acc : List V) : ∀ e w, w ∈ pev acc e ↔ w ∈ acc ∨ w ∈ mentioned e
  | .node d cs, w => by
    simp only [pev, mentioned, List.mem_append]
    rw [addAll_mem, pevL_mem (addAll acc (ownVars d)) cs w, addAll_mem]
    simp only [or_assoc]
theorem pevL_mem (acc : List V) : ∀ l w, w ∈ pevL acc l ↔ w ∈ acc ∨ w ∈ mentionedL l
  | [], w => by simp [pevL, mentionedL]
  | c :: cs, w => by
    simp only [pevL, mentionedL, List.mem_append]
    rw [pev_mem _ c w, pevL_mem acc cs w]
    simp only [or_assoc, or_comm (a := w ∈ mentionedL cs)]
end

mutual
theorem pev_nodup (acc : List V) (h : acc.Nodup) : ∀ e, (pev acc e).Nodup
  | .node d cs => addAll_nodup _ _ (pevL_nodup _ (addAll_nodup _ _ h) cs)
theorem pevL_nodup (acc : List V) (h : acc.Nodup) : ∀ l, (pevL acc l).Nodup
  | [] => h
  | c :: cs => pev_nodup _ (pevL_nodup acc h cs) c
end

/-- `pattern_expr_variables` lists every mentioned variable exactly once. -/
theorem patternExprVariables_spec (e : Ex) :
    (patternExprVariables e).Nodup ∧ ∀ w, w ∈ patternExprVariables e ↔ w ∈ mentioned e := by
  refine ⟨pev_nodup [] List.nodup_nil e, fun w => ?_⟩
  simp [patternExprVariables, pev_mem]

end PlannerBuild.E
