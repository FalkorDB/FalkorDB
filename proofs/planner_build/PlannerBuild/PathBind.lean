import PlannerBuild.Misc
import PlannerBuild.PlanMatch2
/-
# Named paths over an already-bound node (#2311, #2727, #2737, #2743, #2744, #2725; fixed by #3526)

Before #3526 `MATCH p = (n)` / `OPTIONAL MATCH p = (n)` with `n` bound lost the path or the rows,
for two reasons, both fixed:

1. **`plan_match` builds the path.** The node-only, already-bound branch (mod.rs) emits
   `Argument` (no labels) or a synthetic self-loop `ExpandInto(Argument)` (labels; the binder
   copies the node's labels onto every later occurrence). Since #3526 every node-only branch puts
   the component's paths in a `PathBuilder` over its plan, and the bound branch also pushes
   `n IS NOT NULL` (`is_not_null_filter`) to the bound filters: `nodeOnly_bound_binds_paths`,
   `boundNode_path_built` (`BindsAllPaths`).
2. **The MATCH arm Apply-wraps it.** `all_visited` (mod.rs) runs over the pattern's nodes and
   relationships only, not its path variables, so `MATCH p = (n0)` over a bound `n0` is wrapped in
   `Apply` (`planMatchArm_path_wrapped`), and the previous clauses are no longer stitched under an
   `Argument`, which the runtime never runs (`argument_fill_ignores_input`; property
   `Plan.argIsLeaf`).

The repros of the six issues now give the openCypher rows: `fix2311_optional_path_bound`,
`fix2737_optional_path_bound`, `fix2743_bound_path_counts_rows`, `fix2744_bound_path_fed`,
`fix2727_alias_path_keeps_rows`, `fix2725_call_then_bound_path_keeps_var`.
-/
set_option linter.unusedSimpArgs false

namespace PlannerBuild.E

/-! ## Path variables a match plan binds -/

mutual
/-- Path variables bound by a `plan_match` tree. -/
def MP.pathVars : MP → List V
  | .arg => []
  | .allScan _ cs => MP.pathVarsL cs
  | .labelScan _ cs => MP.pathVarsL cs
  | .filter _ c => c.pathVars
  | .pathB ps c => ps.map Prod.fst ++ c.pathVars
  | .cvlt _ _ pv cs => pv.toList ++ MP.pathVarsL cs
  | .shortest _ cs => MP.pathVarsL cs
  | .expandInto _ _ cs => MP.pathVarsL cs
  | .condTr _ _ cs => MP.pathVarsL cs
  | .cp cs => MP.pathVarsL cs
def MP.pathVarsL : List MP → List V
  | [] => []
  | c :: cs => c.pathVars ++ MP.pathVarsL cs
end

/-- **Every named path is built.** For every component of the pattern and every
named path `p = (…)` it declares, the plan `plan_match` builds binds `p` (a
`PathBuilder` lists it, or the var-length traverse it was elided onto carries it).
This is what openCypher requires of `MATCH`/`OPTIONAL MATCH`: each output row binds
every path variable of the pattern. Holds for every component with a
relationship and, since #3526, for every node-only component, bound or not
(`nodeOnly_path_built`). -/
def BindsAllPaths (st : MSt) (wi : WI) (comps : List Comp) : Prop :=
  ∀ c ∈ comps, ∀ p ∈ c.paths, p.1 ∈ (planMatch st wi comps none).1.pathVars

/-! ## Fix 1: every node-only branch builds the path -/

theorem withPaths_pathVars (ps : List (V × List V)) (m : MP) (p : V × List V) (hp : p ∈ ps) :
    p.1 ∈ (withPaths ps m).pathVars := by
  have hne : ps.isEmpty = false := by cases ps <;> simp_all
  simp only [withPaths, hne, Bool.false_eq_true, ↓reduceIte, MP.pathVars, List.mem_append, List.mem_map]
  exact Or.inl ⟨p, hp, rfl⟩

/-- The bound branch (mod.rs) binds every path of the component (#3526). -/
theorem nodeOnly_bound_binds_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) (p : V × List V) (hp : p ∈ ps) :
    p.1 ∈ (nodeOnly st lw n ps).1.pathVars := by
  unfold nodeOnly; simp only [hb, ↓reduceIte]
  split <;> exact withPaths_pathVars _ _ _ hp

/-- …and filters out a row where the bound node is null (an earlier OPTIONAL MATCH left it
unmatched): `n IS NOT NULL` is one of its bound filters as soon as it has a path. -/
theorem nodeOnly_bound_notNull (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) (hne : ps ≠ []) :
    isNotNullFilter n.alias ∈ (nodeOnly st lw n ps).2.2.1 := by
  rw [nodeOnly_bound_filter _ _ _ _ hb]
  cases ps with
  | nil => exact absurd rfl hne
  | cons _ _ => simp

/-- The unbound branch does (mod.rs). -/
theorem nodeOnly_unbound_binds_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = false) (p : V × List V) (hp : p ∈ ps) :
    p.1 ∈ (nodeOnly st lw n ps).1.pathVars := by
  unfold nodeOnly; simp only [hb, Bool.false_eq_true, ↓reduceIte]
  exact withPaths_pathVars _ _ _ hp

theorem nodeOnly_binds_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (p : V × List V) (hp : p ∈ ps) : p.1 ∈ (nodeOnly st lw n ps).1.pathVars := by
  cases hb : st.bound n.alias
  · exact nodeOnly_unbound_binds_paths _ _ _ _ hb _ hp
  · exact nodeOnly_bound_binds_paths _ _ _ _ hb _ hp

theorem orderRels_nil (st : MSt) (fv : List V) (lead : Option V) : orderRels st fv lead [] = [] := by
  cases lead <;> simp [orderRels, sortRels, leadOrder, pickFirst, HopOrder.selfLoopsAfterBinders]

theorem keepNode_alias (pending : List V) (n : QN) : (keepNode pending n).alias = n.alias := by
  unfold keepNode; split <;> rfl

/-- A single node-only component: `plan_match` = the node-only plan of its node with the
cross-entity entries lifted (`n'`), under the `and_of` of the lifted and bound filters if any
(mod.rs). -/
theorem planMatch_nodeOnly (st : MSt) (wi : WI) (n : QN) (ps : List (V × List V)) :
    planMatch st wi [{ nodes := [n], rels := [], paths := ps }] none =
      let n' := keepNode (pendingOf st [{ nodes := [n], rels := [], paths := ps }]) n
      let L := (liftPattern st [{ nodes := [n], rels := [], paths := ps }]).2
      ((match L ++ (nodeOnly st [] n' ps).2.2.1 with
        | [] => (nodeOnly st [] n' ps).1
        | fs => .filter (andOf fs) (nodeOnly st [] n' ps).1), (nodeOnly st [] n' ps).2.1) := by
  simp only [planMatch, planFrom, liftPattern, rewriteComp, List.map_cons, List.map_nil, planComp,
    boundLabelFilters, orderRels_nil, List.flatMap_nil, List.foldl_nil, List.foldl_cons, List.nil_append]
  generalize liftItems _ _ _ = L
  cases h : L ++ (nodeOnly st [] (keepNode (pendingOf st [{ nodes := [n], rels := [], paths := ps }]) n) ps).2.2.1 <;>
    simp [h]

theorem planMatch_nodeOnly_pathVars (st : MSt) (wi : WI) (n : QN) (ps : List (V × List V)) :
    (planMatch st wi [{ nodes := [n], rels := [], paths := ps }] none).1.pathVars =
      (nodeOnly st [] (keepNode (pendingOf st [{ nodes := [n], rels := [], paths := ps }]) n) ps).1.pathVars := by
  rw [planMatch_nodeOnly]
  dsimp only
  split <;> rfl

/-- A map-free node is not rewritten. -/
theorem keepNode_noattrs (pending : List V) (n : QN) (h : n.attrs = []) : keepNode pending n = n := by
  simp [keepNode, keepAttrs, h]

/-- **Fix 1, for every node-only component, bound or not**: `MATCH p = (n)` /
`OPTIONAL MATCH p = (n)` satisfies `BindsAllPaths` (before #3526 it failed whenever `n` was bound). -/
theorem nodeOnly_path_built (st : MSt) (wi : WI) (n : QN) (ps : List (V × List V)) :
    BindsAllPaths st wi [{ nodes := [n], rels := [], paths := ps }] := by
  intro c hc p hp
  simp only [List.mem_singleton] at hc; subst hc
  rw [planMatch_nodeOnly_pathVars]
  exact nodeOnly_binds_paths _ _ _ _ _ hp

theorem boundNode_path_built (st : MSt) (wi : WI) (n : QN) (p : V) :
    BindsAllPaths st wi [{ nodes := [n], rels := [], paths := [(p, [n.alias])] }] :=
  nodeOnly_path_built _ _ _ _

/-! ## Fix 2: the MATCH arm's Apply decision -/

/-- `pattern.variables()` of a one-component pattern (ast.rs). -/
def compVars (c : Comp) : List V := c.nodes.map (·.alias) ++ c.rels.map (·.alias) ++ c.paths.map (·.1)

/-- The node and relationship aliases of a one-component pattern (`pattern.nodes()` then
`pattern.relationships()`): what the MATCH arm's `all_visited` runs over since #3526. -/
def compEnts (c : Comp) : List V := c.nodes.map (·.alias) ++ c.rels.map (·.alias)

mutual
/-- The `plan_match` result as an IR tree: operator kinds, payloads dropped (spec id 0). -/
def MP.toPlan : MP → Plan
  | .arg => .node .argument []
  | .allScan _ cs => .node (.scan 0) (MP.toPlanL cs)
  | .labelScan _ cs => .node (.scan 0) (MP.toPlanL cs)
  | .filter _ c => .node (.filter 0) [c.toPlan]
  | .pathB _ c => .node (.pathBuilder 0) [c.toPlan]
  | .cvlt _ _ _ cs => .node (.hop .varLen 0) (MP.toPlanL cs)
  | .shortest _ cs => .node (.hop .shortest 0) (MP.toPlanL cs)
  | .expandInto _ _ cs => .node (.hop .expandInto 0) (MP.toPlanL cs)
  | .condTr _ _ cs => .node (.hop .condTraverse 0) (MP.toPlanL cs)
  | .cp cs => .node .cartesian (MP.toPlanL cs)
/-- Children. -/
def MP.toPlanL : List MP → List Plan
  | [] => []
  | c :: cs => c.toPlan :: MP.toPlanL cs
end

/-- `Planner::plan`'s MATCH arm (mod.rs) around the `plan_match` result
`m`; `vars` = `pattern.variables()`, `ents` = the pattern's node and relationship aliases (no
paths, #3526), `ar` = `pattern_attrs_read_visited` (an inline map reads a bound variable, #2981),
`st` = the planner state *before* `plan_match`.
OPTIONAL: `Optional(new vars)`, under `Apply` if any variable is bound; MATCH:
`Apply(m)` if **all** nodes and relationships are bound, or (#2981) if a variable is read
(`any_visited`) and `m` joins several components (`joins_components`); else `m` as is. The
Argument taps of the second case bind only the earlier variables (`Some(bound)`);
`Plan.argument` carries no variable list, so that restriction is not modelled. -/
def planMatchArm (st : MSt) (vars ents : List V) (ar : Bool) (optional : Bool) (m : Plan) : Plan :=
  if optional then
    let ovs := (vars.filter fun v => !st.bound v).map V.id
    if vars.any st.bound then .node .apply [.node (.optional ovs) [addArgs m]]
    else .node (.optional ovs) [addArgs m]
  else if ents.all st.bound || ((vars.any st.bound || ar) && joinsComponents m) then .node .apply [addArgs m]
  else m

/-- **Fix 2** (#3526): a MATCH whose nodes and relationships are all bound is Apply-wrapped,
whatever new path variables it declares (before, a new path variable blocked the wrap). -/
theorem planMatchArm_path_wrapped (st : MSt) (c : Comp) (ar : Bool) (m : Plan)
    (hb : (compEnts c).all st.bound = true) :
    planMatchArm st (compVars c) (compEnts c) ar false m = .node .apply [addArgs m] := by
  simp [planMatchArm, hb]

/-- #2981: a MATCH that reads a bound variable (through a pattern variable or an inline
map) and joins several components is Apply-wrapped, so its components read that variable
from the Argument row instead of being stitched beside the earlier clause as one more
CartesianProduct branch (bug 2, `Match.cp_stitch_loses_correlation`). -/
theorem planMatchArm_joined_wrapped (st : MSt) (vars ents : List V) (ar : Bool) (m : Plan)
    (hr : vars.any st.bound = true ∨ ar = true) (hj : joinsComponents m = true) :
    planMatchArm st vars ents ar false m = .node .apply [addArgs m] := by
  rcases hr with h | h <;> simp [planMatchArm, h, hj]

/-- A disjoint MATCH (nothing bound read) is not wrapped, whatever its plan. -/
theorem planMatchArm_disjoint (st : MSt) (c : Comp) (m : Plan)
    (hne : c.nodes ≠ []) (hd : (compVars c).any st.bound = false) :
    planMatchArm st (compVars c) (compEnts c) false false m = m := by
  have : (compEnts c).all st.bound = false := by
    obtain ⟨v, vs, hv⟩ := List.exists_cons_of_ne_nil hne
    simp only [compVars, hv, List.map_cons, List.cons_append, List.any_cons, Bool.or_eq_false_iff] at hd
    simp [compEnts, hv, hd.1]
  simp [planMatchArm, this, hd]

theorem refill_argument (q y : Plan) :
    refill (.node .argument []) (slotLoop (.node .argument [])) q y = .node .argument [y] := by
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem refill_eiArg (s : Nat) (q y : Plan) :
    refill (.node (.hop .expandInto s) [.node .argument []])
      (slotLoop (.node (.hop .expandInto s) [.node .argument []])) q y =
      .node (.hop .expandInto s) [.node .argument [y]] := by
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- Stitched under an unwrapped `Argument`, the previous clauses are never run: the
clause yields the argument row whatever they are (runtime.rs). -/
theorem argument_fill_ignores_input {Val Gr : Type} (S : Sem Val Gr) (q y : Plan)
    (r : Rec Val) (g : Gr) :
    ev S (refill (.node .argument []) (slotLoop (.node .argument [])) q y) r g = ([r], g) := by
  rw [refill_argument]; simp [ev]

/-- The same for the labelled form `ExpandInto(Argument)`: the label check runs on the
argument row only. -/
theorem expandInto_fill_ignores_input {Val Gr : Type} (S : Sem Val Gr) (s : Nat) (q y : Plan)
    (r : Rec Val) (g : Gr) :
    ev S (refill (.node (.hop .expandInto s) [.node .argument []])
      (slotLoop (.node (.hop .expandInto s) [.node .argument []])) q y) r g =
      S.step (.hop .expandInto s) r g := by
  rw [refill_eiArg]; simp [ev, Op.isBag, forRows_single]

/-! ## The stitched-plan property -/


end PlannerBuild.E

namespace PlannerBuild

mutual
/-- **No `Argument` has a child.** The runtime never builds an Argument's children
(runtime.rs), so a clause stitched under one never runs. A fix keeps every
`Argument` a leaf of an `Apply`/`Optional`/`Merge` sub-plan. -/
def Plan.argIsLeaf : Plan → Bool
  | .node .argument cs => cs.isEmpty
  | .node _ cs => Plan.argIsLeafL cs
/-- Children. -/
def Plan.argIsLeafL : List Plan → Bool
  | [] => true
  | c :: cs => c.argIsLeaf && Plan.argIsLeafL cs
end

end PlannerBuild

namespace PlannerBuild.E

/-! ## The repros

Records map variable ids to values: node `7` (the one node of each repro graph),
path `100` (`[(0)]`), label `55`, a count under id `9`. -/

/-- One-node repro graph. Scans bind `n0` (id 0); the synthetic self-loop
`ExpandInto` checks the label of a bound `n0` and yields nothing when `n0` is
missing; `PathBuilder` binds the path (id 1) to `[(n0)]`; the procedure yields a
label (id 2); `count(*)` puts the row count under id 9. -/
def pathS : Sem Nat Unit where
  scan := fun _ r _ => [r.set 0 7]
  hop := fun _ r _ => if r.lookup 0 = some 7 then [r] else []
  test := fun _ _ _ => true
  proj := fun _ r _ => r
  agg := fun _ _ rows => [[(9, rows.length)]]
  sort := fun _ rows => rows
  dedup := id
  items := fun _ _ _ => []
  path := fun _ r => r.set 1 100
  proc := fun _ r _ => [r.set 2 55]
  create := fun _ r g => (r, g)
  write := fun _ _ g => g
  pad := fun vs r => vs.foldl (fun r v => r.set v 0) r

/-- openCypher `MATCH p = (n0)` / `OPTIONAL MATCH p = (n0)` on a row binding `n0`:
the row, extended by the one-node path. -/
def pathMatch (r : Rec Nat) : List (Rec Nat) := if r.lookup 0 = some 7 then [r.set 1 100] else []

/-- Each output row binds every variable in `xs` (a `RETURN` of them is not null /
not "Variable … not found"). -/
def RowsBind (xs : List Var) (rows : List (Rec Nat)) : Prop := ∀ row ∈ rows, ∀ x ∈ xs, (row.lookup x).isSome

instance (xs : List Var) (rows : List (Rec Nat)) : Decidable (RowsBind xs rows) := by
  unfold RowsBind; infer_instance

def n0 : V := ⟨0, 0⟩
def p0 : V := ⟨1, 0⟩
def pathComp (n : QN) (p : V) : Comp := { nodes := [n], rels := [], paths := [(p, [n.alias])] }
/-- After `MATCH (n0:…)` (and `WITH n0`, `WITH *`, `CALL … YIELD`): `n0` bound. -/
def boundSt : MSt := { visited := [n0], verified := [] }

theorem boundSt_n0 : boundSt.bound n0 = true := rfl

/-- The labelled bound node: `Filter(n IS NOT NULL)` over `PathBuilder(ExpandInto(Argument))`. -/
theorem planMatch_labelled (st : MSt) (n p : V) (l : String) (hb : st.bound n = true) :
    (planMatch st {} [pathComp { alias := n, labels := [l], ptr := 0 } p] none).1.toPlan =
      .node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]] := by
  simp only [pathComp]; rw [planMatch_nodeOnly, keepNode_noattrs _ _ rfl]
  simp [liftPattern, liftOrder, liftItems, liftItem, nodeOnly, withPaths, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan, MP.toPlanL]

/-- The unlabelled bound node: `Filter(n IS NOT NULL)` over `PathBuilder(Argument)`. -/
theorem planMatch_plain (st : MSt) (n p : V) (hb : st.bound n = true) :
    (planMatch st {} [pathComp { alias := n, ptr := 0 } p] none).1.toPlan =
      .node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]] := by
  simp only [pathComp]; rw [planMatch_nodeOnly, keepNode_noattrs _ _ rfl]
  simp [liftPattern, liftOrder, liftItems, liftItem, nodeOnly, withPaths, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan, MP.toPlanL]

theorem addArgs_fpeiArg : addArgs (.node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]]) =
    .node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]] := by
  simp [addArgs]

theorem addArgs_fpArg : addArgs (.node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]]) =
    .node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]] := by
  simp [addArgs]

/-- Shared shape of the OPTIONAL repros: `MATCH (n0:L) OPTIONAL MATCH p0 = (n0) RETURN n0, p0`. -/
def optQuery (l : String) : List Plan :=
  [.node (.scan 0) [],
   planMatchArm boundSt (compVars (pathComp { alias := n0, labels := [l], ptr := 0 } p0))
     (compEnts (pathComp { alias := n0, labels := [l], ptr := 0 } p0))
     (attrsReadVisited boundSt [pathComp { alias := n0, labels := [l], ptr := 0 } p0]) true
     (planMatch boundSt {} [pathComp { alias := n0, labels := [l], ptr := 0 } p0] none).1.toPlan,
   .node (.project 0) []]

/-- `Project(Apply(Scan, Optional(Filter(PathBuilder(ExpandInto(Argument))))))`. -/
theorem optQuery_plan (l : String) :
    rustStitch (optQuery l) =
      .node (.project 0) [.node .apply [.node (.scan 0) [],
        .node (.optional [1]) [.node (.filter 0) [.node (.pathBuilder 0)
          [.node (.hop .expandInto 0) [.node .argument []]]]]]] := by
  simp only [optQuery, planMatch_labelled boundSt n0 p0 l boundSt_n0, addArgs_fpeiArg]
  simp [planMatchArm, compVars, compEnts, pathComp, boundSt, MSt.bound, n0, p0, addArgs_fpeiArg, refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]


theorem optQuery_rows (l : String) :
    (ev pathS (rustStitch (optQuery l)) [] ()).1 = [[(1, 100), (0, 7)]] := by
  rw [optQuery_plan]; decide

/-- **#2311 fixed** (`MATCH (n0:P) OPTIONAL MATCH p0=(n0) RETURN n0.id, p0` → `1, [(0)]`, as C).
`plan_match` satisfies `BindsAllPaths`; the stitched plan
`Project(Apply(Scan, Optional(Filter(PathBuilder(ExpandInto(Argument))))))` returns the row with
`p0` bound, the openCypher answer. -/
theorem fix2311_optional_path_bound :
    BindsAllPaths boundSt {} [pathComp { alias := n0, labels := ["P"], ptr := 0 } p0] ∧
    (rustStitch (optQuery "P")).argIsLeaf = true ∧
    RowsBind [0, 1] (ev pathS (rustStitch (optQuery "P")) [] ()).1 ∧
    (ev pathS (rustStitch (optQuery "P")) [] ()).1 = [[(0, 7)]].flatMap pathMatch := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_, ?_⟩
  · rw [optQuery_plan]; decide
  · rw [optQuery_rows]; decide
  · rw [optQuery_rows]; decide

/-- **#2737 fixed** (`MATCH (n1:X) OPTIONAL MATCH p6 = (n1) RETURN p6` → `[(0)]`, as C). -/
theorem fix2737_optional_path_bound :
    BindsAllPaths boundSt {} [pathComp { alias := n0, labels := ["X"], ptr := 0 } p0] ∧
    RowsBind [1] (ev pathS (rustStitch (optQuery "X")) [] ()).1 := by
  refine ⟨boundNode_path_built _ _ _ _, ?_⟩
  rw [optQuery_rows]; decide

/-- Shared shape of the MATCH repros: `MATCH (n:L) WITH … MATCH p0 = (n) RETURN …`;
`st` is the state after the WITH (`n` bound), `ret` the RETURN plan (`Project` or `Aggregate`). -/
def withQuery (st : MSt) (n : V) (l : String) (ret : Plan) : List Plan :=
  [.node (.scan 0) [], .node (.project 0) [],
   planMatchArm st (compVars (pathComp { alias := n, labels := [l], ptr := 0 } p0))
     (compEnts (pathComp { alias := n, labels := [l], ptr := 0 } p0))
     (attrsReadVisited st [pathComp { alias := n, labels := [l], ptr := 0 } p0]) false
     (planMatch st {} [pathComp { alias := n, labels := [l], ptr := 0 } p0] none).1.toPlan,
   ret]

theorem withQuery_wrapped (st : MSt) (n : V) (l : String) (hb : st.bound n = true) :
    planMatchArm st (compVars (pathComp { alias := n, labels := [l], ptr := 0 } p0))
     (compEnts (pathComp { alias := n, labels := [l], ptr := 0 } p0))
     (attrsReadVisited st [pathComp { alias := n, labels := [l], ptr := 0 } p0]) false
      (planMatch st {} [pathComp { alias := n, labels := [l], ptr := 0 } p0] none).1.toPlan =
      .node .apply [.node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]]] := by
  rw [planMatch_labelled st n p0 l hb, planMatchArm_path_wrapped st _ _ _ (by simp [compEnts, pathComp, hb]),
    addArgs_fpeiArg]

theorem withQuery_project (st : MSt) (n : V) (l : String) (hb : st.bound n = true) (k : Nat) :
    rustStitch (withQuery st n l (.node (.project k) [])) =
      .node (.project k) [.node .apply [.node (.project 0) [.node (.scan 0) []],
        .node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]]]] := by
  simp only [withQuery, withQuery_wrapped st n l hb]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem withQuery_aggregate (st : MSt) (n : V) (l : String) (hb : st.bound n = true) (k : Nat) :
    rustStitch (withQuery st n l (.node (.aggregate k) [])) =
      .node (.aggregate k) [.node .apply [.node (.project 0) [.node (.scan 0) []],
        .node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]]]] := by
  simp only [withQuery, withQuery_wrapped st n l hb]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2744 fixed** (`MATCH (n0:A) WITH n0 MATCH p=(n0) RETURN n0.k8, p` → one row `1, [(0)]`, as
C). The path is built and the clause is Apply-wrapped, so the WITH feeds it:
`Project(Apply(Project(Scan), Filter(PathBuilder(ExpandInto(Argument)))))`, every `Argument` a
leaf; the rows are the openCypher ones. -/
theorem fix2744_bound_path_fed :
    BindsAllPaths boundSt {} [pathComp { alias := n0, labels := ["A"], ptr := 0 } p0] ∧
    (rustStitch (withQuery boundSt n0 "A" (.node (.project 0) []))).argIsLeaf = true ∧
    (ev pathS (rustStitch (withQuery boundSt n0 "A" (.node (.project 0) []))) [] ()).1 =
      (ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_⟩
  · rw [withQuery_project boundSt n0 _ rfl]; decide
  · rw [withQuery_project boundSt n0 _ rfl]; decide

/-- **#2743 fixed** (`MATCH (n0:V) WITH * MATCH p0=(n0) RETURN count(*)` → 1, as C). -/
theorem fix2743_bound_path_counts_rows :
    BindsAllPaths boundSt {} [pathComp { alias := n0, labels := ["V"], ptr := 0 } p0] ∧
    (rustStitch (withQuery boundSt n0 "V" (.node (.aggregate 0) []))).argIsLeaf = true ∧
    (ev pathS (rustStitch (withQuery boundSt n0 "V" (.node (.aggregate 0) []))) [] ()).1 =
      pathS.agg 0 [] ((ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch) ∧
    (ev pathS (rustStitch (withQuery boundSt n0 "V" (.node (.aggregate 0) []))) [] ()).1 = [[(9, 1)]] := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_, ?_⟩
  · rw [withQuery_aggregate boundSt n0 _ rfl]; decide
  · rw [withQuery_aggregate boundSt n0 _ rfl]; decide
  · rw [withQuery_aggregate boundSt n0 _ rfl]; decide

/-- `WITH n0 AS alias4`: the alias is a fresh variable of the next scope. -/
def alias4 : V := ⟨0, 1⟩
def aliasSt : MSt := { visited := [alias4], verified := [] }

/-- **#2727 fixed** (`MATCH (n0:A) WITH n0 AS alias4 MATCH (alias4), p1=(alias4) RETURN alias4.k0`
→ `'x'`, as C). The pattern has one node-only component carrying `p1` (`connected_components`,
ast.rs); the binder copies `:A` onto `alias4`, so it is the labelled shape again. -/
theorem fix2727_alias_path_keeps_rows :
    BindsAllPaths aliasSt {} [pathComp { alias := alias4, labels := ["A"], ptr := 0 } p0] ∧
    (rustStitch (withQuery aliasSt alias4 "A" (.node (.project 0) []))).argIsLeaf = true ∧
    (ev pathS (rustStitch (withQuery aliasSt alias4 "A" (.node (.project 0) []))) [] ()).1 =
      (ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_⟩
  · rw [withQuery_project aliasSt alias4 _ rfl]; decide
  · rw [withQuery_project aliasSt alias4 _ rfl]; decide

/-- `MATCH (n0) CALL db.labels() YIELD label MATCH (n0), p0=(n0) RETURN n0.k11, label, p0`. -/
def callQuery : List Plan :=
  [.node (.scan 0) [], .node (.procCall 0) [],
   planMatchArm boundSt (compVars (pathComp { alias := n0, ptr := 0 } p0))
     (compEnts (pathComp { alias := n0, ptr := 0 } p0))
     (attrsReadVisited boundSt [pathComp { alias := n0, ptr := 0 } p0]) false
     (planMatch boundSt {} [pathComp { alias := n0, ptr := 0 } p0] none).1.toPlan,
   .node (.project 0) []]

/-- `Project(Apply(ProcedureCall(Scan), Filter(PathBuilder(Argument))))`. -/
theorem callQuery_plan :
    rustStitch callQuery =
      .node (.project 0) [.node .apply [.node (.procCall 0) [.node (.scan 0) []],
        .node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]]]] := by
  have hw := planMatchArm_path_wrapped boundSt (pathComp { alias := n0, ptr := 0 } p0)
    (attrsReadVisited boundSt [pathComp { alias := n0, ptr := 0 } p0])
    (.node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]]) rfl
  simp only [callQuery, planMatch_plain boundSt n0 p0 boundSt_n0, hw, addArgs_fpArg]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2725 fixed** (`MATCH (n0) CALL db.labels() YIELD label MATCH (n0), p0=(n0) RETURN …` →
`1, A, [(0)]`, as C; before: `Variable n0 not found`). The unlabelled bound node is
`Filter(PathBuilder(Argument))` under `Apply`, so the CALL feeds it and every row binds `n0`,
`label` and `p0`. -/
theorem fix2725_call_then_bound_path_keeps_var :
    BindsAllPaths boundSt {} [pathComp { alias := n0, ptr := 0 } p0] ∧
    (rustStitch callQuery).argIsLeaf = true ∧
    RowsBind [0, 1, 2] (ev pathS (rustStitch callQuery) [] ()).1 ∧
    (ev pathS (rustStitch callQuery) [] ()).1 =
      (ev pathS (.node (.procCall 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_, ?_⟩
  · rw [callQuery_plan]; decide
  · rw [callQuery_plan]; decide
  · rw [callQuery_plan]; decide

/-- Without the earlier binding (`MATCH p0 = (n0:A)` alone) the plan is `PathBuilder(Scan)`. -/
theorem unbound_path_built (l : String) :
    BindsAllPaths { visited := [], verified := [] } {} [pathComp { alias := n0, labels := [l], ptr := 0 } p0] :=
  nodeOnly_path_built _ _ _ _

end PlannerBuild.E
