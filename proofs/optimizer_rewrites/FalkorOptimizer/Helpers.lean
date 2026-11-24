import FalkorOptimizer.Basic
/-
# Syntactic helpers of the optimizer passes
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

/-- `collect_subtree_variables`: the ids of what `get_variables` reports. -/
def subtreeIds (getVars : List Var) : List Nat := getVars.map Var.id

theorem mem_subtreeIds (gv : List Var) (i : Nat) : i ∈ subtreeIds gv ↔ ∃ v ∈ gv, v.id = i := by
  simp [subtreeIds]

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

/-- `Variable::is_unnamed` (ast.rs, #3303): an unnamed pattern entity has no name.
Before #3303 the test was the `_anon` prefix of the name (`is_anon`), which a user name such
as `_anon_x` also passed. -/
def isUnnamed (name : Option String) : Bool := name.isNone

/-- An inline attribute map is "empty" iff it is a `Map` with no entries. -/
def attrsEmpty : X → Bool
  | .node .map [] => true
  | _ => false

theorem attrsEmpty_iff (a : X) : attrsEmpty a = true ↔ a = .node .map [] := by
  constructor
  · intro h; match a, h with | .node .map [], _ => rfl
  · rintro rfl; rfl

theorem isUnnamed_none : isUnnamed none = true := rfl
/-- No written name, `_anon…` included, counts as unnamed (#3303). -/
theorem isUnnamed_some (n : String) : isUnnamed (some n) = false := rfl

/-- `Alias` (ast.rs): a name the query wrote, or a numbered unnamed entity. -/
inductive Alias
  | named (n : String)
  | anon (k : Nat)

/-- `Alias::name` (ast.rs). -/
def Alias.name : Alias → Option String
  | .named n => some n
  | .anon _ => none

/-- `Variable::name` then `Variable::is_unnamed` (`self.name().is_none()`): the variable's
`alias : Option<Alias>` is `None` for a planner temporary. -/
def varIsUnnamed (alias : Option Alias) : Bool := (alias.bind Alias.name).isNone

/-- **PROVEN** (`Variable::is_unnamed`, #3303): a variable is unnamed exactly when its alias is
absent (a planner temporary) or numbered (`Anon`); a written name, `_anon_x` included, never is.
`isUnnamed` above is the test applied to the name this yields. -/
theorem varIsUnnamed_iff (a : Option Alias) :
    varIsUnnamed a = true ↔ (a = none ∨ ∃ k, a = some (.anon k)) := by
  cases a with
  | none => simp [varIsUnnamed]
  | some al => cases al <;> simp [varIsUnnamed, Alias.name]

theorem varIsUnnamed_named (n : String) : varIsUnnamed (some (.named n)) = false := rfl

theorem varIsUnnamed_eq (a : Option Alias) : varIsUnnamed a = isUnnamed (a.bind Alias.name) := rfl

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
  isUnnamed p.edgeName && isUnnamed c.edgeName && !p.emit && !c.emit &&
  p.siblings.isEmpty && c.siblings.isEmpty && !p.bidirectional && !c.bidirectional &&
  !p.varLen && !c.varLen && attrsEmpty p.edgeAttrs && attrsEmpty c.edgeAttrs &&
  p.fromV == c.toV && isUnnamed p.fromName && p.fromLabels.isEmpty &&
  unread p.fromV && unread p.edgeV && unread c.edgeV

/-- Fusion only happens on two plain, storage-direction, anonymous, unfiltered,
fixed-length hops whose shared middle node is anonymous, unlabelled and — like both
edges — read by no operator of the plan (its inline map, if any, is a Filter that reads
it) — the preconditions of `fuse_same_support` (Passes.lean). -/
theorem canFuse_spec (p c : CT) (u : Var → Bool) (h : canFuse p c u = true) :
    p.optional = false ∧ c.optional = false ∧ p.emit = false ∧ c.emit = false ∧
    isUnnamed p.edgeName = true ∧ isUnnamed c.edgeName = true ∧ p.fromV = c.toV ∧
    isUnnamed p.fromName = true ∧ p.fromLabels = [] ∧
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

/-- `Project.copies` and `Aggregate.projections` hold `(old, new)` pairs
(binder.rs; project.rs and aggregate.rs write `new_var`). Since
#3281 `branch_output_variables(node, include_copies)` (push_filters_down.rs)
reads the second component (the slot the copy is written to), and only when
`include_copies` (the routing sets, :251); `branch_visible_variables` (:99) passes `false`. -/
inductive OutIR
  | project (exprs : List Var) (copies : List (Var × Var))
  | aggregate (names : List Var) (projections : List (Var × Var))
  | other

def copied (inc : Bool) (cs : List (Var × Var)) : List Nat :=
  if inc then cs.map (·.2.id) else []

def branchOut (inc : Bool) : OutIR → Option (List Nat)
  | .project es cs => some (es.map Var.id ++ copied inc cs)
  | .aggregate ns ps => some (ns.map Var.id ++ copied inc ps)
  | .other => none

/-- Above a Project/Aggregate only its outputs are visible (the named ones, and
the copies' destinations when `inc`); elsewhere `none` (the subtree's variables).
Ids only — scope is dropped (see `id_only_routing_unsound`). -/
theorem branchOut_spec (inc : Bool) (o : OutIR) :
    (∀ es cs, o = .project es cs → ∀ i, i ∈ (branchOut inc o).getD [] ↔
      ∃ v ∈ es ++ (if inc then cs.map Prod.snd else []), v.id = i) ∧
    (∀ ns ps, o = .aggregate ns ps → ∀ i, i ∈ (branchOut inc o).getD [] ↔
      ∃ v ∈ ns ++ (if inc then ps.map Prod.snd else []), v.id = i) ∧
    (o = .other → branchOut inc o = none) := by
  refine ⟨?_, ?_, ?_⟩
  · rintro es cs rfl i
    cases inc <;> simp [branchOut, copied, or_and_right, exists_or]
    exact or_congr_right ⟨fun ⟨a, b, h1, h2⟩ => ⟨b, ⟨a, h1⟩, h2⟩, fun ⟨b, ⟨a, h1⟩, h2⟩ => ⟨a, b, h1, h2⟩⟩
  · rintro ns ps rfl i
    cases inc <;> simp [branchOut, copied, or_and_right, exists_or]
    exact or_congr_right ⟨fun ⟨a, b, h1, h2⟩ => ⟨b, ⟨a, h1⟩, h2⟩, fun ⟨b, ⟨a, h1⟩, h2⟩ => ⟨a, b, h1, h2⟩⟩
  · rintro rfl; rfl

/-- Since #3281 the copy's *new* variable — what the Project actually outputs — is
reported for routing, and no copy is visible to a later clause. -/
theorem branchOut_copy_new :
    branchOut true (.project [] [(⟨0, 0⟩, ⟨5, 1⟩)]) = some [5] ∧
    branchOut false (.project [] [(⟨0, 0⟩, ⟨5, 1⟩)]) = some [] := ⟨rfl, rfl⟩

/-! ## `branch_visible_variables` -/

/-- A plan node as `branch_visible_variables` (push_filters_down.rs) sees it:
`reset` is a `Project`/`Aggregate` with `branch_output_variables(_, false)` =
`outs`; `semi` a `SemiApply`/`AntiSemiApply`/`OrApplyMultiplexer`; `plain` any
other operator. `own` is `collect_subtree_variables` of the lone node. -/
inductive VT where
  | reset (outs : List Nat) (kids : List VT)
  | semi (own : List Nat) (kids : List VT)
  | plain (own : List Nat) (kids : List VT)

mutual
def visT : VT → List Nat
  | .reset outs _ => outs
  | .semi own [] => own
  | .semi own (k :: _) => own ++ visT k
  | .plain own ks => own ++ visL ks
def visL : List VT → List Nat
  | [] => []
  | k :: ks => visT k ++ visL ks
end

/-- A `Project`/`Aggregate` hides everything beneath it. -/
theorem visT_reset (outs : List Nat) (ks : List VT) : visT (.reset outs ks) = outs := by
  simp [visT]

/-- A (anti) semi-join or `OrApplyMultiplexer` exposes only its first child:
the existence sub-plans' variables never count. -/
theorem visT_semi_first (own : List Nat) (k : VT) (ks ks' : List VT) :
    visT (.semi own (k :: ks)) = visT (.semi own (k :: ks')) := by
  simp [visT]

/-- Any other operator exposes its own variables and every child's. -/
theorem visT_plain (own : List Nat) (ks : List VT) :
    visT (.plain own ks) = own ++ visL ks := by
  simp [visT]

/-- `MATCH (x:A:B) WHERE (x)-[:R]->() CALL { OPTIONAL MATCH (n:A:B {id: 3}) … }`
(the PR's test): the left branch is `SemiApply(Scan x, Traverse(_anon…, Argument))`;
only `x` (id 0) is visible, not the existence sub-plan's anonymous ids 1, 2. -/
theorem visT_pattern_predicate :
    visT (.semi [] [.plain [0] [], .plain [1, 0, 2] [.plain [] []]]) = [0] := by
  simp [visT, visL]

/-! ## `utilize_node_by_id`'s parent lookup -/

/-- The parent of a node addressed by its child-index path (`None` at the root). -/
def parentPath : List Nat → Option (List Nat)
  | [] => none
  | p => some p.dropLast

/-- `utilize_node_by_id.rs` unwraps the parent of every label/all-node scan;
a scan at the root (the plan of `CALL … YIELD … MATCH (n)`, which validation lets
through — planner_build `call_then_match_accepted`) has none: the server panics. -/
theorem root_scan_has_no_parent : parentPath [] = none := rfl

theorem nonroot_has_parent (i : Nat) (p : List Nat) : (parentPath (i :: p)).isSome := rfl

end Falkor.Opt
