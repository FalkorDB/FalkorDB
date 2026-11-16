import PlannerBuild.Misc
import PlannerBuild.PlanMatch2
/-
# Named paths over an already-bound node (fixed by #3526)

Issues #2311, #2727, #2737, #2743, #2744, #2725: one root cause, two places. Both
are fixed by #3526; the pre-#3526 code is kept below as `nodeOnlyPre3526` /
`planMatchArmPre3526` with its counterexamples renamed `pre3526_*`.

1. **`plan_match` did not build the path.** The node-only, already-bound branch
   emitted `Argument` (no labels) or a synthetic self-loop `ExpandInto(Argument)`
   (labels; the binder copies the node's labels onto every later occurrence), and
   only the unbound branch added `PathBuilder(component.paths())`. Since #3526 both
   branches end in `PathBuilder` (mod.rs:2003-2007), and the bound branch adds
   `n IS NOT NULL` to the bound filters (mod.rs:1935-1939): `nodeOnly_binds_paths`,
   `nodeOnly_bound_not_null`, `boundNode_path_built`.
2. **The MATCH arm did not Apply-wrap it.** `all_visited` ran over
   `pattern.variables()`, which includes the path variables (ast.rs:816-822), so a
   new path variable blocked the `Apply` wrap and stitching put the previous clause
   *under* the `Argument` (the runtime never builds an Argument's children,
   runtime.rs:658-660). Since #3526 `all_visited` (mod.rs:3090-3095) runs over nodes
   and relationships only: `planMatchArm_bound_wrapped`.

OPTIONAL MATCH is Apply-wrapped (`any_visited`, mod.rs:3066-3068); it now gets the path too.

| here | there |
| --- | --- |
| `MP.pathVars` | the variables a match plan binds as paths: `PathBuilder` entries, a CondVarLenTraverse's elided path |
| `BindsAllPaths` | the property: every named path of the pattern is bound by the plan |
| `compVars` | `QueryGraph::variables` ast.rs:816-822 (nodes, relationships, paths) |
| `nrVars` | the nodes and relationships `all_visited` reads (mod.rs:3090-3095) |
| `planMatchArm` | `Planner::plan`'s MATCH arm mod.rs:3054-3103 |
| `MP.toPlan` | the `plan_match` result as an IR tree (operator kinds; payloads dropped) |
| `Plan.argIsLeaf` | the property: no `Argument` has a child |

Fixed: `bug2311_fixed`, `bug2737_fixed`, `bug2743_fixed`, `bug2744_fixed`,
`bug2727_fixed`, `bug2725_fixed`. Pre-#3526 counterexamples:
`pre3526_bug2311_optional_path_null`, `pre3526_bug2737_optional_path_null`,
`pre3526_bug2743_bound_path_drops_rows`, `pre3526_bug2744_bound_path_unfed_argument`,
`pre3526_bug2727_alias_path_drops_rows`, `pre3526_bug2725_call_then_bound_path_loses_var`.
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
every path variable of the pattern. Holds for every node-only component, bound or
not (`nodeOnly_binds_paths`, `boundNode_path_built`); before #3526 it failed for a
bound one (`pre3526_nodeOnly_bound_no_paths`). -/
def BindsAllPaths (st : MSt) (fv : List Nat) (comps : List Comp) : Prop :=
  ∀ c ∈ comps, ∀ p ∈ c.paths, p.1 ∈ (planMatch st fv comps none).1.pathVars

/-! ## Root cause 1: the bound node-only branch -/

/-- `nodeOnly` before #3526: the bound branch built no `PathBuilder` and no null check. -/
def nodeOnlyPre3526 (st : MSt) (lw : LW) (n : QN) (paths : List (V × List V)) : MP × MSt × List Ex × LW :=
  if st.bound n.alias then
    let (af, lw) := lowerN lw n
    let bf := af.toList
    if n.labels.isEmpty then (.arg, st, bf, lw)
    else
      let rel : QR := { alias := ⟨u32max - n.alias.id, n.alias.scope⟩, named := false,
                        frm := { alias := n.alias, ptr := 0 }, to := { alias := n.alias, labels := n.labels, ptr := 0 }, ptr := 0 }
      (.expandInto rel false [.arg], markLabels st n.alias n.labels, bf, lw)
  else
    let (af, lw) := lowerN lw n
    let res := withF af (scanOf (stripNode n))
    let st' := markLabels { st with visited := st.visited ++ [n.alias] } n.alias n.labels
    (if paths.isEmpty then res else .pathB paths res, st', [], lw)

/-- Pre-#3526: the bound branch built no path, whatever paths the component had. -/
theorem pre3526_nodeOnly_bound_no_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) : (nodeOnlyPre3526 st lw n ps).1.pathVars = [] := by
  unfold nodeOnlyPre3526; simp only [hb, ↓reduceIte]
  split <;> rfl

theorem pathVars_wrapP (ps : List (V × List V)) (m : MP) :
    (wrapP ps m).pathVars = (if ps.isEmpty then [] else ps.map Prod.fst) ++ m.pathVars := by
  unfold wrapP; split <;> simp [MP.pathVars]

/-- **Every path of a node-only component is built**, bound or not (mod.rs:2003-2007). -/
theorem nodeOnly_binds_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (p : V × List V) (hp : p ∈ ps) : p.1 ∈ (nodeOnly st lw n ps).1.pathVars := by
  have hne : ps.isEmpty = false := by cases ps <;> simp_all
  have hin : p.1 ∈ (if ps.isEmpty then [] else ps.map Prod.fst) ++ ([] : List V) ∨ True := Or.inr trivial
  clear hin
  unfold nodeOnly
  by_cases hb : st.bound n.alias
  · simp only [hb, ↓reduceIte]
    split <;> simp only [pathVars_wrapP, hne, Bool.false_eq_true, ↓reduceIte, List.mem_append,
      List.mem_map] <;> exact Or.inl ⟨p, hp, rfl⟩
  · simp only [hb, Bool.false_eq_true, ↓reduceIte, pathVars_wrapP, hne, List.mem_append, List.mem_map]
    exact Or.inl ⟨p, hp, rfl⟩

theorem nodeOnly_unbound_binds_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (_hb : st.bound n.alias = false) (p : V × List V) (hp : p ∈ ps) :
    p.1 ∈ (nodeOnly st lw n ps).1.pathVars :=
  nodeOnly_binds_paths st lw n ps p hp

/-- A path over a bound node filters out the rows where it is null (an earlier
OPTIONAL MATCH left it unmatched): `n IS NOT NULL` is a bound filter (mod.rs:1935-1939),
and it keeps exactly the non-null rows (`isNotNullFilter_passes`). -/
theorem nodeOnly_bound_not_null (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) (hne : ps ≠ []) :
    isNotNullFilter n.alias ∈ (nodeOnly st lw n ps).2.2.1 := by
  have : ps.isEmpty = false := by cases ps <;> simp_all
  rw [nodeOnly_bound_filter _ _ _ _ hb]; simp [this]

theorem orderRels_nil (st : MSt) (fv : List Nat) : orderRels st fv [] = [] := by
  simp [orderRels, sortRels, HopOrder.selfLoopsAfterBinders]

/-- A single node-only component: `plan_match` = the node-only plan, under the
`and_of` of its bound filters if any (mod.rs:2391-2412). -/
theorem planMatch_nodeOnly (st : MSt) (fv : List Nat) (n : QN) (ps : List (V × List V)) :
    planMatch st fv [{ nodes := [n], rels := [], paths := ps }] none =
      ((match (nodeOnly st [] n ps).2.2.1 with
        | [] => (nodeOnly st [] n ps).1
        | fs => .filter (andOf fs) (nodeOnly st [] n ps).1), (nodeOnly st [] n ps).2.1) := by
  simp only [planMatch, planComp, boundLabelFilters, orderRels_nil, List.flatMap_nil, List.foldl_nil,
    List.foldl_cons, List.nil_append]
  cases h : (nodeOnly st [] n ps).2.2.1 <;> simp [h]

theorem planMatch_nodeOnly_pathVars (st : MSt) (fv : List Nat) (n : QN) (ps : List (V × List V)) :
    (planMatch st fv [{ nodes := [n], rels := [], paths := ps }] none).1.pathVars =
      (nodeOnly st [] n ps).1.pathVars := by
  rw [planMatch_nodeOnly]
  split <;> rfl

/-- **Root cause 1 fixed, for every node (bound or not) and every named path over it**:
`MATCH p = (n)` / `OPTIONAL MATCH p = (n)` satisfies `BindsAllPaths` (pre-#3526 it
failed whenever `n` was bound). -/
theorem boundNode_path_built (st : MSt) (fv : List Nat) (n : QN) (ps : List (V × List V)) :
    BindsAllPaths st fv [{ nodes := [n], rels := [], paths := ps }] := by
  intro c hc p hp
  simp only [List.mem_singleton] at hc; subst hc
  rw [planMatch_nodeOnly_pathVars]
  exact nodeOnly_binds_paths _ _ _ _ p hp

/-! ## Root cause 2: the MATCH arm's Apply decision -/

/-- `pattern.variables()` of a one-component pattern (ast.rs:816-822). -/
def compVars (c : Comp) : List V := c.nodes.map (·.alias) ++ c.rels.map (·.alias) ++ c.paths.map (·.1)

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

/-- The nodes and relationships of a component: what `all_visited` reads since #3526
(mod.rs:3090-3095); path variables are left out. -/
def nrVars (c : Comp) : List V := c.nodes.map (·.alias) ++ c.rels.map (·.alias)

/-- `Planner::plan`'s MATCH arm (mod.rs:3054-3103) around the `plan_match` result
`m`; `vars` = `pattern.variables()`, `nr` = its nodes and relationships, `st` = the
planner state *before* `plan_match`. OPTIONAL: `Optional(new vars)`, under `Apply`
if any of `vars` is bound; MATCH: `Apply(m)` only if **all** of `nr` are bound, else `m`. -/
def planMatchArm (st : MSt) (vars nr : List V) (optional : Bool) (m : Plan) : Plan :=
  if optional then
    let ovs := (vars.filter fun v => !st.bound v).map V.id
    if vars.any st.bound then .node .apply [.node (.optional ovs) [addArgs m]]
    else .node (.optional ovs) [addArgs m]
  else if nr.all st.bound then .node .apply [addArgs m]
  else m

/-- **Root cause 2 fixed**: a MATCH whose nodes and relationships are all bound is
Apply-wrapped, whatever path variables it declares. -/
theorem planMatchArm_bound_wrapped (st : MSt) (c : Comp) (m : Plan)
    (hall : (nrVars c).all st.bound = true) :
    planMatchArm st (compVars c) (nrVars c) false m = .node .apply [addArgs m] := by
  simp [planMatchArm, hall]

/-- The arm before #3526: MATCH tested `all_visited` over `pattern.variables()`. -/
def planMatchArmPre3526 (st : MSt) (vars : List V) (optional : Bool) (m : Plan) : Plan :=
  if optional then
    let ovs := (vars.filter fun v => !st.bound v).map V.id
    if vars.any st.bound then .node .apply [.node (.optional ovs) [addArgs m]]
    else .node (.optional ovs) [addArgs m]
  else if vars.all st.bound then .node .apply [addArgs m]
  else m

theorem planMatchArm_optional (st : MSt) (vars nr : List V) (m : Plan) :
    planMatchArm st vars nr true m = planMatchArmPre3526 st vars true m := rfl

/-- Pre-#3526: a path variable is new, so a MATCH that declares one was never
Apply-wrapped, even when every node of it was bound. -/
theorem pre3526_planMatchArm_path_unwrapped (st : MSt) (c : Comp) (m : Plan) (p : V × List V)
    (hp : p ∈ c.paths) (hnew : st.bound p.1 = false) :
    planMatchArmPre3526 st (compVars c) false m = m := by
  have : (compVars c).all st.bound = false := by
    rw [List.all_eq_false]
    exact ⟨p.1, by simp [compVars]; exact Or.inr (Or.inr ⟨p.2, hp⟩), by simp [hnew]⟩
  simp [planMatchArmPre3526, this]

theorem refill_argument (q y : Plan) :
    refill (.node .argument []) (slotLoop (.node .argument [])) q y = .node .argument [y] := by
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem refill_eiArg (s : Nat) (q y : Plan) :
    refill (.node (.hop .expandInto s) [.node .argument []])
      (slotLoop (.node (.hop .expandInto s) [.node .argument []])) q y =
      .node (.hop .expandInto s) [.node .argument [y]] := by
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- Stitched under an unwrapped `Argument`, the previous clauses are never run: the
clause yields the argument row whatever they are (runtime.rs:658-660, 1285). -/
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
(runtime.rs:658-660), so a clause stitched under one never runs. A fix keeps every
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

/-! ### Before #3526 (the old model; no bound filter, so `plan_match` was the `nodeOnly` plan) -/

/-- The pre-#3526 plan of `MATCH p = (n)` for a bound `n`. -/
def preMatch (st : MSt) (n : QN) (p : V) : Plan := (nodeOnlyPre3526 st [] n [(p, [n.alias])]).1.toPlan

/-- The labelled bound node: `ExpandInto(Argument)`, no `PathBuilder`. -/
theorem pre3526_planMatch_labelled (st : MSt) (n p : V) (l : String) (hb : st.bound n = true) :
    preMatch st { alias := n, labels := [l], ptr := 0 } p =
      .node (.hop .expandInto 0) [.node .argument []] := by
  simp [preMatch, nodeOnlyPre3526, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan, MP.toPlanL]

/-- The unlabelled bound node: a bare `Argument`. -/
theorem pre3526_planMatch_plain (st : MSt) (n p : V) (hb : st.bound n = true) :
    preMatch st { alias := n, ptr := 0 } p = .node .argument [] := by
  simp [preMatch, nodeOnlyPre3526, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan]

theorem addArgs_eiArg : addArgs (.node (.hop .expandInto 0) [.node .argument []]) =
    .node (.hop .expandInto 0) [.node .argument []] := by
  simp [addArgs]

/-- Shared shape of the OPTIONAL repros: `MATCH (n0:L) OPTIONAL MATCH p0 = (n0) RETURN n0, p0`. -/
def optQueryPre (l : String) : List Plan :=
  [.node (.scan 0) [],
   planMatchArmPre3526 boundSt (compVars (pathComp { alias := n0, labels := [l], ptr := 0 } p0)) true
     (preMatch boundSt { alias := n0, labels := [l], ptr := 0 } p0),
   .node (.project 0) []]

/-- The live EXPLAIN shape `Project(Apply(Scan, Optional(ExpandInto(Argument))))`. -/
theorem optQueryPre_plan (l : String) :
    rustStitch (optQueryPre l) =
      .node (.project 0) [.node .apply [.node (.scan 0) [],
        .node (.optional [1]) [.node (.hop .expandInto 0) [.node .argument []]]]] := by
  simp only [optQueryPre, pre3526_planMatch_labelled boundSt n0 p0 l boundSt_n0, addArgs_eiArg]
  simp [planMatchArmPre3526, compVars, pathComp, boundSt, MSt.bound, n0, p0, addArgs_eiArg, refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem optQueryPre_rows (l : String) :
    (ev pathS (rustStitch (optQueryPre l)) [] ()).1 = [[(0, 7)]] := by
  rw [optQueryPre_plan]; decide

/-- **#2311** (`MATCH (n0:P) OPTIONAL MATCH p0=(n0) RETURN n0.id, p0` → Rust `1, null`,
C `1, [(0)]`). `plan_match` violated `BindsAllPaths`; the stitched plan
`Project(Apply(Scan, Optional(ExpandInto(Argument))))` (the live EXPLAIN) returns the
row without `p0`; openCypher returns it with the path. -/
theorem pre3526_bug2311_optional_path_null :
    (nodeOnlyPre3526 boundSt [] { alias := n0, labels := ["P"], ptr := 0 } [(p0, [n0])]).1.pathVars = [] ∧
    (ev pathS (rustStitch (optQueryPre "P")) [] ()).1 = [[(0, 7)]] ∧
    ¬ RowsBind [0, 1] (ev pathS (rustStitch (optQueryPre "P")) [] ()).1 ∧
    [[(0, 7)]].flatMap pathMatch = [[(1, 100), (0, 7)]] := by
  refine ⟨pre3526_nodeOnly_bound_no_paths _ _ _ _ boundSt_n0, optQueryPre_rows _, ?_, by decide⟩
  rw [optQueryPre_rows]; decide

/-- **#2737** (`MATCH (n1:X) OPTIONAL MATCH p6 = (n1) RETURN p6` → Rust `null`, C `[(0)]`):
the same shape as #2311. -/
theorem pre3526_bug2737_optional_path_null :
    (nodeOnlyPre3526 boundSt [] { alias := n0, labels := ["X"], ptr := 0 } [(p0, [n0])]).1.pathVars = [] ∧
    ¬ RowsBind [1] (ev pathS (rustStitch (optQueryPre "X")) [] ()).1 := by
  refine ⟨pre3526_nodeOnly_bound_no_paths _ _ _ _ boundSt_n0, ?_⟩
  rw [optQueryPre_rows]; decide

/-- Shared shape of the MATCH repros: `MATCH (n:L) WITH … MATCH p0 = (n) RETURN …`;
`st` is the state after the WITH (`n` bound), `ret` the RETURN plan (`Project` or `Aggregate`). -/
def withQueryPre (st : MSt) (n : V) (l : String) (ret : Plan) : List Plan :=
  [.node (.scan 0) [], .node (.project 0) [],
   planMatchArmPre3526 st (compVars (pathComp { alias := n, labels := [l], ptr := 0 } p0)) false
     (preMatch st { alias := n, labels := [l], ptr := 0 } p0),
   ret]

theorem withQueryPre_unwrapped (st : MSt) (n : V) (l : String) (hb : st.bound n = true)
    (hp : st.bound p0 = false) :
    planMatchArmPre3526 st (compVars (pathComp { alias := n, labels := [l], ptr := 0 } p0)) false
      (preMatch st { alias := n, labels := [l], ptr := 0 } p0) =
      .node (.hop .expandInto 0) [.node .argument []] := by
  rw [pre3526_planMatch_labelled st n p0 l hb]
  exact pre3526_planMatchArm_path_unwrapped st _ _ (p0, [n]) (List.mem_singleton_self _) hp

/-- The live EXPLAIN shape: the WITH and the first MATCH under the `Argument`. -/
theorem withQueryPre_project (st : MSt) (n : V) (l : String) (hb : st.bound n = true)
    (hp : st.bound p0 = false) (k : Nat) :
    rustStitch (withQueryPre st n l (.node (.project k) [])) =
      .node (.project k) [.node (.hop .expandInto 0) [.node .argument [.node (.project 0) [.node (.scan 0) []]]]] := by
  simp only [withQueryPre, withQueryPre_unwrapped st n l hb hp]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem withQueryPre_aggregate (st : MSt) (n : V) (l : String) (hb : st.bound n = true)
    (hp : st.bound p0 = false) (k : Nat) :
    rustStitch (withQueryPre st n l (.node (.aggregate k) [])) =
      .node (.aggregate k) [.node (.hop .expandInto 0) [.node .argument [.node (.project 0) [.node (.scan 0) []]]]] := by
  simp only [withQueryPre, withQueryPre_unwrapped st n l hb hp]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2744** (`MATCH (n0:A) WITH n0 MATCH p=(n0) RETURN n0.k8, p` → Rust 0 rows,
C one row `1, [(0)]`). Both root causes: no `PathBuilder` (`BindsAllPaths` fails),
and the unwrapped `Argument` gets the previous clauses as a child — the live EXPLAIN
`Project(ExpandInto(Argument(Project(Scan))))` — violating `argIsLeaf`; the row is lost. -/
theorem pre3526_bug2744_bound_path_unfed_argument :
    (nodeOnlyPre3526 boundSt [] { alias := n0, labels := ["A"], ptr := 0 } [(p0, [n0])]).1.pathVars = [] ∧
    (rustStitch (withQueryPre boundSt n0 "A" (.node (.project 0) []))).argIsLeaf = false ∧
    (ev pathS (rustStitch (withQueryPre boundSt n0 "A" (.node (.project 0) []))) [] ()).1 = [] ∧
    (ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch =
      [[(1, 100), (0, 7)]] := by
  refine ⟨pre3526_nodeOnly_bound_no_paths _ _ _ _ boundSt_n0, ?_, ?_, by decide⟩
  · rw [withQueryPre_project boundSt n0 _ rfl rfl]; decide
  · rw [withQueryPre_project boundSt n0 _ rfl rfl]; decide

/-- **#2743** (`MATCH (n0:V) WITH * MATCH p0=(n0) RETURN count(*)` → Rust 0, C 1). -/
theorem pre3526_bug2743_bound_path_drops_rows :
    (nodeOnlyPre3526 boundSt [] { alias := n0, labels := ["V"], ptr := 0 } [(p0, [n0])]).1.pathVars = [] ∧
    (rustStitch (withQueryPre boundSt n0 "V" (.node (.aggregate 0) []))).argIsLeaf = false ∧
    (ev pathS (rustStitch (withQueryPre boundSt n0 "V" (.node (.aggregate 0) []))) [] ()).1 = [[(9, 0)]] ∧
    pathS.agg 0 [] ((ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch) =
      [[(9, 1)]] := by
  refine ⟨pre3526_nodeOnly_bound_no_paths _ _ _ _ boundSt_n0, ?_, ?_, by decide⟩
  · rw [withQueryPre_aggregate boundSt n0 _ rfl rfl]; decide
  · rw [withQueryPre_aggregate boundSt n0 _ rfl rfl]; decide

/-- `WITH n0 AS alias4`: the alias is a fresh variable of the next scope. -/
def alias4 : V := ⟨0, 1⟩
def aliasSt : MSt := { visited := [alias4], verified := [] }

/-- **#2727** (`MATCH (n0:A) WITH n0 AS alias4 MATCH (alias4), p1=(alias4) RETURN alias4.k0`
→ Rust 0 rows, C `'x'`; the full repro errors `Variable alias4 not found`). The pattern
has one node-only component carrying `p1` (`connected_components`, ast.rs:926-928);
the binder copies `:A` onto `alias4`, so the plan is `ExpandInto(Argument)` again. -/
theorem pre3526_bug2727_alias_path_drops_rows :
    (nodeOnlyPre3526 aliasSt [] { alias := alias4, labels := ["A"], ptr := 0 } [(p0, [alias4])]).1.pathVars = [] ∧
    (rustStitch (withQueryPre aliasSt alias4 "A" (.node (.project 0) []))).argIsLeaf = false ∧
    (ev pathS (rustStitch (withQueryPre aliasSt alias4 "A" (.node (.project 0) []))) [] ()).1 = [] := by
  refine ⟨pre3526_nodeOnly_bound_no_paths _ _ _ _ rfl, ?_, ?_⟩
  · rw [withQueryPre_project aliasSt alias4 _ rfl rfl]; decide
  · rw [withQueryPre_project aliasSt alias4 _ rfl rfl]; decide

/-- `MATCH (n0) CALL db.labels() YIELD label MATCH (n0), p0=(n0) RETURN n0.k11, label, p0`. -/
def callQueryPre : List Plan :=
  [.node (.scan 0) [], .node (.procCall 0) [],
   planMatchArmPre3526 boundSt (compVars (pathComp { alias := n0, ptr := 0 } p0)) false
     (preMatch boundSt { alias := n0, ptr := 0 } p0),
   .node (.project 0) []]

theorem callQueryPre_plan :
    rustStitch callQueryPre =
      .node (.project 0) [.node .argument [.node (.procCall 0) [.node (.scan 0) []]]] := by
  have hu := pre3526_planMatchArm_path_unwrapped boundSt (pathComp { alias := n0, ptr := 0 } p0)
    (.node .argument []) (p0, [n0]) (List.mem_singleton_self _) rfl
  simp only [callQueryPre, pre3526_planMatch_plain boundSt n0 p0 boundSt_n0, hu]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2725** (`… CALL algo.BFS(…) YIELD … MATCH (n0), p0 = (n0), (n0) WHERE … RETURN *`
→ Rust `Variable n0 not found`, C no error; minimal with `db.labels()`: Rust
`Variable n0 not found`, C `1, A, [(0)]`). The unlabelled bound node is a bare
`Argument`; the CALL and the first MATCH are stitched under it (live EXPLAIN
`Project(Argument(ProcedureCall(All Node Scan)))`), so the one output row is empty:
`n0`, `label` and `p0` are all unbound. -/
theorem pre3526_bug2725_call_then_bound_path_loses_var :
    (nodeOnlyPre3526 boundSt [] { alias := n0, ptr := 0 } [(p0, [n0])]).1.pathVars = [] ∧
    (rustStitch callQueryPre).argIsLeaf = false ∧
    (ev pathS (rustStitch callQueryPre) [] ()).1 = [[]] ∧
    ¬ RowsBind [0] (ev pathS (rustStitch callQueryPre) [] ()).1 ∧
    (ev pathS (.node (.procCall 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch =
      [[(1, 100), (2, 55), (0, 7)]] := by
  refine ⟨pre3526_nodeOnly_bound_no_paths _ _ _ _ boundSt_n0, ?_, ?_, ?_, by decide⟩
  · rw [callQueryPre_plan]; decide
  · rw [callQueryPre_plan]; decide
  · rw [callQueryPre_plan]; decide


/-! ### Since #3526 -/

/-- The labelled bound node: `Filter(PathBuilder(ExpandInto(Argument)))`, the filter
being `n IS NOT NULL`. -/
theorem planMatch_labelled (st : MSt) (n p : V) (l : String) (hb : st.bound n = true) :
    (planMatch st [] [pathComp { alias := n, labels := [l], ptr := 0 } p] none).1.toPlan =
      .node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]] := by
  simp only [pathComp]; rw [planMatch_nodeOnly]
  simp [nodeOnly, wrapP, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan, MP.toPlanL]

/-- The unlabelled bound node: `Filter(PathBuilder(Argument))`. -/
theorem planMatch_plain (st : MSt) (n p : V) (hb : st.bound n = true) :
    (planMatch st [] [pathComp { alias := n, ptr := 0 } p] none).1.toPlan =
      .node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]] := by
  simp only [pathComp]; rw [planMatch_nodeOnly]
  simp [nodeOnly, wrapP, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan, MP.toPlanL]

def fpe : Plan := .node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]]
def fpa : Plan := .node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]]

theorem addArgs_fpe : addArgs fpe = fpe := by simp [fpe, addArgs]
theorem addArgs_fpa : addArgs fpa = fpa := by simp [fpa, addArgs]

/-- `MATCH (n0:L) OPTIONAL MATCH p0 = (n0) RETURN n0, p0`. -/
def optQuery (l : String) : List Plan :=
  [.node (.scan 0) [],
   planMatchArm boundSt (compVars (pathComp { alias := n0, labels := [l], ptr := 0 } p0))
     (nrVars (pathComp { alias := n0, labels := [l], ptr := 0 } p0)) true
     (planMatch boundSt [] [pathComp { alias := n0, labels := [l], ptr := 0 } p0] none).1.toPlan,
   .node (.project 0) []]

theorem optQuery_plan (l : String) :
    rustStitch (optQuery l) =
      .node (.project 0) [.node .apply [.node (.scan 0) [], .node (.optional [1]) [fpe]]] := by
  simp only [optQuery, planMatch_labelled boundSt n0 p0 l boundSt_n0]
  rw [show (Plan.node (.filter 0) [.node (.pathBuilder 0) [.node (.hop .expandInto 0) [.node .argument []]]]) = fpe from rfl]
  have harm : planMatchArm boundSt (compVars (pathComp { alias := n0, labels := [l], ptr := 0 } p0))
      (nrVars (pathComp { alias := n0, labels := [l], ptr := 0 } p0)) true fpe =
      .node .apply [.node (.optional [1]) [fpe]] := by
    have hany : (compVars (pathComp { alias := n0, labels := [l], ptr := 0 } p0)).any boundSt.bound = true := by
      simp [compVars, pathComp, boundSt_n0]
    simp only [planMatchArm, hany, ↓reduceIte, addArgs_fpe]
    rfl
  rw [harm]
  simp [fpe, refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem optQuery_rows (l : String) :
    (ev pathS (rustStitch (optQuery l)) [] ()).1 = [[(1, 100), (0, 7)]] := by
  rw [optQuery_plan]; decide

/-- **#2311 fixed** (`MATCH (n0:P) OPTIONAL MATCH p0=(n0) RETURN n0.id, p0` → `1, [(0)]`,
as C): the path is built and the row binds it, as openCypher's `pathMatch`. -/
theorem bug2311_fixed :
    BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["P"], ptr := 0 } p0] ∧
    (ev pathS (rustStitch (optQuery "P")) [] ()).1 = [[(0, 7)]].flatMap pathMatch ∧
    RowsBind [0, 1] (ev pathS (rustStitch (optQuery "P")) [] ()).1 := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_⟩ <;> rw [optQuery_rows] <;> decide

/-- **#2737 fixed** (`MATCH (n1:X) OPTIONAL MATCH p6 = (n1) RETURN p6` → `[(0)]`). -/
theorem bug2737_fixed :
    BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["X"], ptr := 0 } p0] ∧
    RowsBind [1] (ev pathS (rustStitch (optQuery "X")) [] ()).1 := by
  refine ⟨boundNode_path_built _ _ _ _, ?_⟩
  rw [optQuery_rows]; decide

/-- `MATCH (n:L) WITH … MATCH p0 = (n) RETURN …`. -/
def withQuery (st : MSt) (n : V) (l : String) (ret : Plan) : List Plan :=
  [.node (.scan 0) [], .node (.project 0) [],
   planMatchArm st (compVars (pathComp { alias := n, labels := [l], ptr := 0 } p0))
     (nrVars (pathComp { alias := n, labels := [l], ptr := 0 } p0)) false
     (planMatch st [] [pathComp { alias := n, labels := [l], ptr := 0 } p0] none).1.toPlan,
   ret]

theorem withQuery_wrapped (st : MSt) (n : V) (l : String) (hb : st.bound n = true) :
    planMatchArm st (compVars (pathComp { alias := n, labels := [l], ptr := 0 } p0))
      (nrVars (pathComp { alias := n, labels := [l], ptr := 0 } p0)) false
      (planMatch st [] [pathComp { alias := n, labels := [l], ptr := 0 } p0] none).1.toPlan =
      .node .apply [fpe] := by
  rw [planMatch_labelled st n p0 l hb, planMatchArm_bound_wrapped st _ _ (by simp [nrVars, pathComp, hb])]
  exact congrArg (fun x => Plan.node .apply [x]) addArgs_fpe

/-- The stitched plan: the WITH and the first MATCH feed the `Apply`. -/
theorem withQuery_plan (st : MSt) (n : V) (l : String) (hb : st.bound n = true) (k : Nat) (op : Op)
    (hop : op = .project k ∨ op = .aggregate k) :
    rustStitch (withQuery st n l (.node op [])) =
      .node op [.node .apply [.node (.project 0) [.node (.scan 0) []], fpe]] := by
  simp only [withQuery, withQuery_wrapped st n l hb]
  rcases hop with rfl | rfl <;> simp [fpe, refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2744 fixed** (`MATCH (n0:A) WITH n0 MATCH p=(n0) RETURN n0.k8, p` → one row
`1, [(0)]`, as C): the path is built, no `Argument` has a child, and the rows are
openCypher's. -/
theorem bug2744_fixed :
    BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["A"], ptr := 0 } p0] ∧
    (rustStitch (withQuery boundSt n0 "A" (.node (.project 0) []))).argIsLeaf = true ∧
    (ev pathS (rustStitch (withQuery boundSt n0 "A" (.node (.project 0) []))) [] ()).1 =
      (ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_⟩ <;>
    rw [withQuery_plan boundSt n0 _ rfl 0 _ (Or.inl rfl)] <;> decide

/-- **#2743 fixed** (`MATCH (n0:V) WITH * MATCH p0=(n0) RETURN count(*)` → 1, as C). -/
theorem bug2743_fixed :
    BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["V"], ptr := 0 } p0] ∧
    (rustStitch (withQuery boundSt n0 "V" (.node (.aggregate 0) []))).argIsLeaf = true ∧
    (ev pathS (rustStitch (withQuery boundSt n0 "V" (.node (.aggregate 0) []))) [] ()).1 =
      pathS.agg 0 [] ((ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch) := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_⟩ <;>
    rw [withQuery_plan boundSt n0 _ rfl 0 _ (Or.inr rfl)] <;> decide

/-- **#2727 fixed** (`MATCH (n0:A) WITH n0 AS alias4 MATCH (alias4), p1=(alias4) RETURN alias4.k0`
→ `'x'`, as C). -/
theorem bug2727_fixed :
    BindsAllPaths aliasSt [] [pathComp { alias := alias4, labels := ["A"], ptr := 0 } p0] ∧
    (rustStitch (withQuery aliasSt alias4 "A" (.node (.project 0) []))).argIsLeaf = true ∧
    (ev pathS (rustStitch (withQuery aliasSt alias4 "A" (.node (.project 0) []))) [] ()).1 =
      [[(1, 100), (0, 7)]] := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_⟩ <;>
    rw [withQuery_plan aliasSt alias4 _ rfl 0 _ (Or.inl rfl)] <;> decide

/-- `MATCH (n0) CALL db.labels() YIELD label MATCH (n0), p0=(n0) RETURN n0.k11, label, p0`. -/
def callQuery : List Plan :=
  [.node (.scan 0) [], .node (.procCall 0) [],
   planMatchArm boundSt (compVars (pathComp { alias := n0, ptr := 0 } p0))
     (nrVars (pathComp { alias := n0, ptr := 0 } p0)) false
     (planMatch boundSt [] [pathComp { alias := n0, ptr := 0 } p0] none).1.toPlan,
   .node (.project 0) []]

theorem callQuery_plan :
    rustStitch callQuery =
      .node (.project 0) [.node .apply [.node (.procCall 0) [.node (.scan 0) []], fpa]] := by
  have hw := planMatchArm_bound_wrapped boundSt (pathComp { alias := n0, ptr := 0 } p0) fpa
    (by simp [nrVars, pathComp, boundSt_n0])
  rw [addArgs_fpa] at hw
  simp only [callQuery, planMatch_plain boundSt n0 p0 boundSt_n0]
  rw [show (Plan.node (.filter 0) [.node (.pathBuilder 0) [.node .argument []]]) = fpa from rfl, hw]
  simp [fpa, refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2725 fixed** (`MATCH (n0) CALL db.labels() YIELD label MATCH (n0), p0=(n0) RETURN …`
→ `1, A, [(0)]`, as C): `n0`, `label` and `p0` are all bound. -/
theorem bug2725_fixed :
    BindsAllPaths boundSt [] [pathComp { alias := n0, ptr := 0 } p0] ∧
    (rustStitch callQuery).argIsLeaf = true ∧
    (ev pathS (rustStitch callQuery) [] ()).1 =
      (ev pathS (.node (.procCall 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch ∧
    RowsBind [0, 1, 2] (ev pathS (rustStitch callQuery) [] ()).1 := by
  refine ⟨boundNode_path_built _ _ _ _, ?_, ?_, ?_⟩ <;> rw [callQuery_plan] <;> decide

/-! ## The unbound branch (unchanged by #3526) -/

theorem unbound_path_built (l : String) :
    BindsAllPaths { visited := [], verified := [] } [] [pathComp { alias := n0, labels := [l], ptr := 0 } p0] :=
  boundNode_path_built _ _ _ _

end PlannerBuild.E
