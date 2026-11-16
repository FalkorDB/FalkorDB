import FalkorOptimizer.Basic
/-
# Syntactic helpers of the optimizer passes

| here | there |
| --- | --- |
| `XD`, `X` | `ExprIR<Variable>` payloads / trees (variables, parameters, constants, calls, the rest) |
| `anyX` | a BFS `walk().any(..)` over an expression |
| `hasParam` | `expr_has_parameter` eliminate_true_filters.rs:75-79 |
| `subst`, `evX` | `substitute_params` eliminate_true_filters.rs:83-98; a parameter-aware evaluator |
| `refsVar` | `expr_references_variable` / `subtree_references_variable` references.rs:317-334 (a BFS, by `(id, scope)`) |
| `refsVarId` | `references_var` utilize_node_by_id.rs:92-104 (`Variable ==` is by id only, ast.rs:113) |
| `collectIds` | `collect_expr_variables` optimizer/mod.rs:92-100 |
| `PT`, `subtreeVars`, `subtreeIds` | `collect_subtree_variables` optimizer/mod.rs:110-137 (#3092: stops at Project/Aggregate, their outputs only) |
| `countVar` | `extract_count_variable` reduce_count.rs:193-212 |
| `isAnon`, `attrsEmpty` | `is_anon`, `rel_attrs_empty` fuse_anonymous_traverse.rs:36-51 (`node_attrs_empty` removed by #2390) |
| `CT`, `canFuse` | `can_fuse` fuse_anonymous_traverse.rs:83-192 (`unread` = `intermediate_unreferenced`, References2.wholeRead) |
| `fusable`, `fusedCT` | `fusable_traverse`, `fused_traverse` fuse_optional_traverse.rs:40-103 |
| `branchOut` | `branch_output_variables` push_filters_down.rs:59-66 (#3092: `collect_subtree_variables` of a Project/Aggregate) |
| `OutIR`, `branchOutPre3092`, `subtreeIdsPre3092` | history: both before #3092 (`copies.0`/`projections.0`; ids of `get_variables`) |
-/
namespace Falkor.Opt

inductive XD
  | var (v : Var)
  | param (n : String)
  | const (c : Val)
  | func (name : String)
  | map
  | other (t : Nat)
  deriving DecidableEq

inductive X
  | node (d : XD) (cs : List X)

mutual
def anyX (p : XD → Bool) : X → Bool
  | .node d cs => p d || anyXL p cs
def anyXL (p : XD → Bool) : List X → Bool
  | [] => false
  | c :: cs => anyX p c || anyXL p cs
end

mutual
def nodesX : X → List XD
  | .node d cs => d :: nodesXL cs
def nodesXL : List X → List XD
  | [] => []
  | c :: cs => nodesX c ++ nodesXL cs
end

mutual
theorem anyX_eq (p : XD → Bool) : ∀ e, anyX p e = (nodesX e).any p
  | .node d cs => by simp [anyX, nodesX, anyXL_eq p cs]
theorem anyXL_eq (p : XD → Bool) : ∀ l, anyXL p l = (nodesXL l).any p
  | [] => rfl
  | c :: cs => by simp [anyXL, nodesXL, anyX_eq p c, anyXL_eq p cs]
end

/-! ## Parameters (`eliminate_true_filters`) -/

def isParam : XD → Bool | .param _ => true | _ => false

def hasParam (e : X) : Bool := anyX isParam e

theorem hasParam_iff (e : X) : hasParam e = true ↔ ∃ n, XD.param n ∈ nodesX e := by
  unfold hasParam; rw [anyX_eq, List.any_eq_true]
  constructor
  · rintro ⟨d, hd, h⟩; cases d <;> simp [isParam] at h; exact ⟨_, hd⟩
  · rintro ⟨n, hn⟩; exact ⟨_, hn, rfl⟩

def substD (ps : List (String × Val)) : XD → XD
  | .param n => match ps.lookup n with
    | some v => .const v
    | none => .param n
  | d => d

mutual
def subst (ps : List (String × Val)) : X → X
  | .node d cs => .node (substD ps d) (substL ps cs)
def substL (ps : List (String × Val)) : List X → List X
  | [] => []
  | c :: cs => subst ps c :: substL ps cs
end

mutual
/-- After substitution only parameters missing from the map remain. -/
theorem subst_params (ps : List (String × Val)) : ∀ e n, XD.param n ∈ nodesX (subst ps e) → ps.lookup n = none
  | .node d cs, n, h => by
    simp only [subst, nodesX, List.mem_cons] at h
    rcases h with h | h
    · cases d with
      | param m =>
        simp only [substD] at h
        cases hl : ps.lookup m with
        | none => rw [hl] at h; simp at h; subst h; exact hl
        | some v => rw [hl] at h; simp at h
      | _ => simp [substD] at h
    · exact substL_params ps cs n h
theorem substL_params (ps : List (String × Val)) : ∀ l n, XD.param n ∈ nodesXL (substL ps l) → ps.lookup n = none
  | [], n, h => by simp [substL, nodesXL] at h
  | c :: cs, n, h => by
    simp only [substL, nodesXL, List.mem_append] at h
    rcases h with h | h
    · exact subst_params ps c n h
    · exact substL_params ps cs n h
end

/- Evaluation with parameter values: any combinator `comb` for the other
nodes; a parameter reads its value (unknown → error). -/
variable {V : Type} (lift : Val → V) (comb : XD → List V → Option V)

mutual
def evX (ps : List (String × Val)) : X → Option V
  | .node (.param n) _ => (ps.lookup n).map lift
  | .node (.const c) _ => some (lift c)
  | .node d cs => (evXL ps cs).bind (comb d)
def evXL (ps : List (String × Val)) : List X → Option (List V)
  | [] => some []
  | c :: cs => match evX ps c, evXL ps cs with
    | some v, some vs => some (v :: vs)
    | _, _ => none
end

mutual
/-- Substituting parameters does not change the value (the constant evaluator
then sees exactly what the runtime would). -/
theorem evX_subst (ps : List (String × Val)) : ∀ e, evX lift comb ps (subst ps e) = evX lift comb ps e
  | .node (.param n) cs => by
    simp only [subst, substD]
    cases hl : ps.lookup n with
    | none => simp [evX, hl]
    | some v => simp [evX, hl]
  | .node (.const c) cs => by simp [subst, substD, evX]
  | .node (.var v) cs => by simp [subst, substD, evX, evXL_subst ps cs]
  | .node (.func f) cs => by simp [subst, substD, evX, evXL_subst ps cs]
  | .node .map cs => by simp [subst, substD, evX, evXL_subst ps cs]
  | .node (.other t) cs => by simp [subst, substD, evX, evXL_subst ps cs]
theorem evXL_subst (ps : List (String × Val)) : ∀ l, evXL lift comb ps (substL ps l) = evXL lift comb ps l
  | [] => rfl
  | c :: cs => by simp [substL, evXL, evX_subst ps c, evXL_subst ps cs]
end

/-! ## Variable references -/

def isVarS (v : Var) : XD → Bool | .var w => w == v | _ => false
def isVarId (i : Nat) : XD → Bool | .var w => w.id == i | _ => false

/-- `expr_references_variable`: by `(id, scope)`. -/
def refsVar (e : X) (v : Var) : Bool := anyX (isVarS v) e
/-- `references_var` (utilize_node_by_id): `Variable ==` compares ids only. -/
def refsVarId (e : X) (v : Var) : Bool := anyX (isVarId v.id) e

theorem refsVar_iff (e : X) (v : Var) : refsVar e v = true ↔ XD.var v ∈ nodesX e := by
  unfold refsVar; rw [anyX_eq, List.any_eq_true]
  constructor
  · rintro ⟨d, hd, h⟩; cases d <;> simp [isVarS] at h; subst h; exact hd
  · intro h; exact ⟨_, h, by simp [isVarS]⟩

theorem refsVarId_iff (e : X) (v : Var) : refsVarId e v = true ↔ ∃ w, XD.var w ∈ nodesX e ∧ w.id = v.id := by
  unfold refsVarId; rw [anyX_eq, List.any_eq_true]
  constructor
  · rintro ⟨d, hd, h⟩; cases d <;> simp [isVarId] at h; exact ⟨_, hd, h⟩
  · rintro ⟨w, hw, h⟩; exact ⟨_, hw, by simp [isVarId, h]⟩

/-- The id-only check is conservative: it can only see more references. -/
theorem refsVar_le_refsVarId (e : X) (v : Var) (h : refsVar e v = true) : refsVarId e v = true :=
  (refsVarId_iff e v).2 ⟨v, (refsVar_iff e v).1 h, rfl⟩

/-- …and does: a same-id variable of another scope counts (only blocks a rewrite). -/
theorem refsVarId_other_scope :
    refsVarId (.node (.var ⟨3, 1⟩) []) ⟨3, 0⟩ = true ∧ refsVar (.node (.var ⟨3, 1⟩) []) ⟨3, 0⟩ = false := by
  decide

/-- `collect_expr_variables`: the ids (only) of the variables an expression mentions. -/
def collectIds (e : X) : List Nat :=
  (nodesX e).filterMap fun d => match d with | .var v => some v.id | _ => none

theorem mem_collectIds (e : X) (i : Nat) : i ∈ collectIds e ↔ ∃ v, XD.var v ∈ nodesX e ∧ v.id = i := by
  unfold collectIds; rw [List.mem_filterMap]
  constructor
  · rintro ⟨d, hd, h⟩; cases d <;> simp at h; exact ⟨_, hd, h⟩
  · rintro ⟨v, hv, rfl⟩; exact ⟨_, hv, rfl⟩

/-- History (before #3092): `collect_subtree_variables` was the ids of what `get_variables` reports. -/
def subtreeIdsPre3092 (getVars : List Var) : List Nat := getVars.map Var.id

theorem pre3092_mem_subtreeIds (gv : List Var) (i : Nat) : i ∈ subtreeIdsPre3092 gv ↔ ∃ v ∈ gv, v.id = i := by
  simp [subtreeIdsPre3092]

/-- A plan subtree as `collect_subtree_variables` reads it: `Project` (`exprs`,
`copies` as `(old, new)` pairs), `Aggregate` (`names`, `projections` as `(old, new)`),
and every other operator with the variables `push_node_variables` reports for it. -/
inductive PT
  | project (es : List Var) (cs : List (Var × Var)) (kids : List PT)
  | aggregate (ns : List Var) (ps : List (Var × Var)) (kids : List PT)
  | other (own : List Var) (kids : List PT)

mutual
/-- `collect_subtree_variables` (mod.rs:110-137): a Project/Aggregate contributes its
outputs (the *new* side of each copy/projection) and is not descended; every other
node contributes its own variables and is descended. (Rust uses a stack and a
`HashSet`; only membership matters.) -/
def subtreeVars : PT → List Var
  | .project es cs _ => es ++ cs.map Prod.snd
  | .aggregate ns ps _ => ns ++ ps.map Prod.snd
  | .other own kids => own ++ subtreeVarsL kids
def subtreeVarsL : List PT → List Var
  | [] => []
  | t :: ts => subtreeVars t ++ subtreeVarsL ts
end

def subtreeIds (t : PT) : List Nat := (subtreeVars t).map Var.id

theorem mem_subtreeIds (t : PT) (i : Nat) : i ∈ subtreeIds t ↔ ∃ v ∈ subtreeVars t, v.id = i := by
  simp [subtreeIds]

theorem mem_subtreeVarsL (ts : List PT) (v : Var) : v ∈ subtreeVarsL ts ↔ ∃ t ∈ ts, v ∈ subtreeVars t := by
  induction ts with
  | nil => simp [subtreeVarsL]
  | cons t ts ih => simp [subtreeVarsL, ih]

/-- Below a Project/Aggregate nothing is reported: the children do not matter. -/
theorem subtreeIds_resets (es : List Var) (cs : List (Var × Var)) (k k' : List PT) :
    subtreeIds (.project es cs k) = subtreeIds (.project es cs k') ∧
    subtreeIds (.aggregate es cs k) = subtreeIds (.aggregate es cs k') := ⟨rfl, rfl⟩

/-- Any other node: its own variables and those of every child subtree. -/
theorem mem_subtreeIds_other (own : List Var) (kids : List PT) (i : Nat) :
    i ∈ subtreeIds (.other own kids) ↔ (∃ v ∈ own, v.id = i) ∨ ∃ t ∈ kids, i ∈ subtreeIds t := by
  simp only [mem_subtreeIds, subtreeVars, List.mem_append, mem_subtreeVarsL]
  constructor
  · rintro ⟨v, hv | ⟨t, ht, hv⟩, rfl⟩
    · exact .inl ⟨v, hv, rfl⟩
    · exact .inr ⟨t, ht, v, hv, rfl⟩
  · rintro (⟨v, hv, rfl⟩ | ⟨t, ht, v, hv, rfl⟩)
    · exact ⟨v, .inl hv, rfl⟩
    · exact ⟨v, .inr ⟨t, ht, hv⟩, rfl⟩

/-- #3092 vs before: an `Aggregate` over a child that binds id 2 (the #3352 shape).
`get_variables` walked through it (`names` and the child's variables); now only its
outputs are reported. -/
theorem subtreeIds_stops_at_aggregate :
    subtreeIds (.aggregate [⟨0, 1⟩, ⟨1, 1⟩] [] [.other [⟨2, 0⟩] []]) = [0, 1] ∧
    2 ∈ subtreeIdsPre3092 [⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 0⟩] := by decide

/-! ## `extract_count_variable` -/

def countVar : X → Option Nat
  | .node (.func "count") (.node (.var v) _ :: _) => some v.id
  | _ => none

theorem countVar_spec (e : X) (i : Nat) :
    countVar e = some i ↔ ∃ v cs rest, e = .node (.func "count") (.node (.var v) cs :: rest) ∧ v.id = i := by
  constructor
  · intro h
    unfold countVar at h
    split at h
    · rename_i v cs rest; simp at h; exact ⟨v, cs, rest, rfl, h⟩
    · simp at h
  · rintro ⟨v, cs, rest, rfl, rfl⟩; rfl

/-! ## `fuse_anonymous_traverse` preconditions -/

def isAnon (name : Option String) : Bool := name.any (·.startsWith "_anon")

/-- An inline attribute map is "empty" iff it is a `Map` with no entries. -/
def attrsEmpty : X → Bool
  | .node .map [] => true
  | _ => false

theorem attrsEmpty_iff (a : X) : attrsEmpty a = true ↔ a = .node .map [] := by
  constructor
  · intro h; match a, h with | .node .map [], _ => rfl
  · rintro rfl; rfl

theorem isAnon_none : isAnon none = false := rfl
theorem isAnon_some (n : String) : isAnon (some n) = n.startsWith "_anon" := rfl

/-- What `can_fuse` reads of a CondTraverse. -/
structure CT where
  bind : Bool
  optional : Bool
  transposed : Bool
  edgeName : Option String
  emit : Bool
  siblings : List Nat
  bidirectional : Bool
  varLen : Bool
  edgeAttrs : X
  fromV : Var
  fromName : Option String
  fromLabels : List String
  toV : Var
  /-- The edge alias (`relationship.alias`). -/
  edgeV : Var

/-- `can_fuse(parent, child)`; `unread` = `intermediate_unreferenced` (whole plan since #2390).
#2390 dropped the `node_attrs_empty(intermediate)` test (its map is a Filter now, which the
reference check sees) and added the two edges to the reference check (:181-185). -/
def canFuse (p c : CT) (unread : Var → Bool) : Bool :=
  p.bind && c.bind && !p.optional && !c.optional && !p.transposed && !c.transposed &&
  isAnon p.edgeName && isAnon c.edgeName && !p.emit && !c.emit &&
  p.siblings.isEmpty && c.siblings.isEmpty && !p.bidirectional && !c.bidirectional &&
  !p.varLen && !c.varLen && attrsEmpty p.edgeAttrs && attrsEmpty c.edgeAttrs &&
  p.fromV == c.toV && isAnon p.fromName && p.fromLabels.isEmpty &&
  unread p.fromV && unread p.edgeV && unread c.edgeV

/-- Fusion only happens on two plain, storage-direction, anonymous, unfiltered,
fixed-length hops whose shared middle node is anonymous, unlabelled and — like both
edges — read by no operator of the plan (its inline map, if any, is a Filter that reads
it) — the preconditions of `fuse_same_support` (Passes.lean). -/
theorem canFuse_spec (p c : CT) (u : Var → Bool) (h : canFuse p c u = true) :
    p.optional = false ∧ c.optional = false ∧ p.emit = false ∧ c.emit = false ∧
    isAnon p.edgeName = true ∧ isAnon c.edgeName = true ∧ p.fromV = c.toV ∧
    isAnon p.fromName = true ∧ p.fromLabels = [] ∧
    p.edgeAttrs = .node .map [] ∧ c.edgeAttrs = .node .map [] ∧ p.varLen = false ∧ c.varLen = false ∧
    p.siblings = [] ∧ c.siblings = [] ∧ u p.fromV = true ∧ u p.edgeV = true ∧ u c.edgeV = true := by
  simp only [canFuse, Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, List.isEmpty_iff] at h
  simp only [← attrsEmpty_iff]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    first | exact h.2 | exact h.1.2 | exact h.1.1.2 | simp_all

/-! ## `fuse_optional_traverse` -/

/-- What `fusable_traverse` reads of the Optional's sub-plan. -/
structure OptSub where
  isCT : Bool
  optional : Bool
  bind : Bool
  chain : Nat
  childIsArgument : Bool
  childCount : Nat
  transposed : Bool
  edge : Var
  fromV : Var
  toV : Var

def fusable (s : OptSub) (vars : List Var) : Bool :=
  s.isCT && !s.optional && s.bind && s.chain == 0 && s.childCount == 1 && s.childIsArgument &&
  !(s.fromV == s.toV) &&
  (let out := if s.transposed then s.fromV else s.toV
   vars.contains out && vars.all fun v => v == s.edge || v == out)

/-- Fusable exactly when the Optional's null-padded variables are the traverse's
edge and destination: the hypothesis of `fuse_optional_correct`. -/
theorem fusable_spec (s : OptSub) (vars : List Var) (h : fusable s vars = true) :
    s.isCT = true ∧ s.optional = false ∧ s.chain = 0 ∧ s.childCount = 1 ∧ s.childIsArgument = true ∧
    s.fromV ≠ s.toV ∧
    (if s.transposed then s.fromV else s.toV) ∈ vars ∧
    ∀ v ∈ vars, v = s.edge ∨ v = (if s.transposed then s.fromV else s.toV) := by
  simp only [fusable, Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, beq_eq_false_iff_ne,
    List.contains_iff_mem, List.all_eq_true, Bool.or_eq_true] at h
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp_all

/-- `fused_traverse`: the same traverse, now optional. -/
def fusedCT (c : CT) : CT := { c with optional := true }

theorem fusedCT_spec (c : CT) : (fusedCT c).optional = true ∧ { fusedCT c with optional := c.optional } = c := by
  cases c; exact ⟨rfl, rfl⟩

/-! ## `branch_output_variables` -/

/-- `branch_output_variables` (push_filters_down.rs:59-66, since #3092): a
Project/Aggregate reports `collect_subtree_variables` of itself; anything else `none`. -/
def branchOut : PT → Option (List Nat)
  | t@(.project ..) | t@(.aggregate ..) => some (subtreeIds t)
  | .other .. => none

/-- Above a Project/Aggregate only its outputs are visible — the *new* variable of
each copy/projection (project.rs:115, aggregate.rs:716 insert `new_var`); elsewhere
`none` (the subtree's variables). Ids only — scope is dropped (see `id_only_routing_unsound`). -/
theorem branchOut_spec (t : PT) :
    (∀ es cs k, t = .project es cs k → ∀ i, i ∈ (branchOut t).getD [] ↔ ∃ v ∈ es ++ cs.map Prod.snd, v.id = i) ∧
    (∀ ns ps k, t = .aggregate ns ps k → ∀ i, i ∈ (branchOut t).getD [] ↔ ∃ v ∈ ns ++ ps.map Prod.snd, v.id = i) ∧
    (∀ own k, t = .other own k → branchOut t = none) := by
  refine ⟨?_, ?_, ?_⟩
  · rintro es cs k rfl i; simp [branchOut, mem_subtreeIds, subtreeVars]
  · rintro ns ps k rfl i; simp [branchOut, mem_subtreeIds, subtreeVars]
  · rintro own k rfl; rfl

/-- The output contract lives in one place: `branch_output_variables` of a
Project/Aggregate is `collect_subtree_variables` of it. -/
theorem branchOut_eq_subtreeIds (t : PT) (h : (branchOut t).isSome) : branchOut t = some (subtreeIds t) := by
  cases t <;> simp_all [branchOut]

/-- The copy's *new* variable — what the Project actually outputs — is reported
(fixed by #3092; before it, only the parent-scope id was: `pre3092_branchOut_copy_old`). -/
theorem branchOut_copy_new :
    branchOut (.project [] [(⟨0, 0⟩, ⟨5, 1⟩)] []) = some [5] := rfl

/-! ### History: `branch_output_variables` before #3092 -/

/-- `Project.copies` holds `(old, new)` pairs (binder.rs:2234; project.rs:115
inserts `new_var`); before #3092 `branch_output_variables` read the first component. -/
inductive OutIR
  | project (exprs : List Var) (copies : List (Var × Var))
  | aggregate (names projections : List Var)
  | other

def branchOutPre3092 : OutIR → Option (List Nat)
  | .project es cs => some (es.map Var.id ++ cs.map (·.1.id))
  | .aggregate ns ps => some (ns.map Var.id ++ ps.map Var.id)
  | .other => none

theorem pre3092_branchOut_spec (o : OutIR) :
    (∀ es cs, o = .project es cs → ∀ i, i ∈ (branchOutPre3092 o).getD [] ↔ ∃ v ∈ es ++ cs.map Prod.fst, v.id = i) ∧
    (∀ ns ps, o = .aggregate ns ps → ∀ i, i ∈ (branchOutPre3092 o).getD [] ↔ ∃ v ∈ ns ++ ps, v.id = i) ∧
    (o = .other → branchOutPre3092 o = none) := by
  refine ⟨?_, ?_, ?_⟩
  · rintro es cs rfl i; simp [branchOutPre3092, or_and_right, exists_or]
  · rintro ns ps rfl i; simp [branchOutPre3092, or_and_right, exists_or]
  · rintro rfl; rfl

/-- Before #3092 the copy's *new* variable was not reported, only the parent-scope id. -/
theorem pre3092_branchOut_copy_old :
    branchOutPre3092 (.project [] [(⟨0, 0⟩, ⟨5, 1⟩)]) = some [0] := rfl

/-! ## `utilize_node_by_id`'s parent lookup -/

/-- The parent of a node addressed by its child-index path (`None` at the root). -/
def parentPath : List Nat → Option (List Nat)
  | [] => none
  | p => some p.dropLast

/-- `utilize_node_by_id.rs:118` unwraps the parent of every label/all-node scan;
a scan at the root (the plan of `CALL … YIELD … MATCH (n)`, which validation lets
through — planner_build `call_then_match_accepted`) has none: the server panics. -/
theorem root_scan_has_no_parent : parentPath [] = none := rfl

theorem nonroot_has_parent (i : Nat) (p : List Nat) : (parentPath (i :: p)).isSome := rfl

end Falkor.Opt
