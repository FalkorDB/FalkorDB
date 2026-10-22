import PlannerBuild.Misc
import PlannerBuild.PlanMatch2
/-
# Named paths over an already-bound node (gap closed 2026-10-20)

Open issues #2311, #2727, #2737, #2743, #2744, #2725: one root cause, two places.

1. **`plan_match` never builds the path.** The node-only, already-bound branch
   (mod.rs:1915-1962) emits `Argument` (no labels) or a synthetic self-loop
   `ExpandInto(Argument)` (labels; the binder copies the node's labels onto every
   later occurrence, so `MATCH (n0:A) … MATCH p=(n0)` takes this branch). Only the
   unbound branch adds `PathBuilder(component.paths())` (mod.rs:1977-1980). So
   `p = (n)` with `n` bound is never bound (`nodeOnly_bound_no_paths`).
2. **The MATCH arm does not Apply-wrap it.** `all_visited` (mod.rs:3067-3069) runs
   over `pattern.variables()`, which includes the path variables (ast.rs:816-822).
   A new path variable is never visited, so the plan of `MATCH p=(n0)` is not
   wrapped in `Apply` (mod.rs:3075-3080) although its only leaf is an `Argument`
   (`planMatchArm_path_unwrapped`). Stitching then puts the previous clause *under*
   that `Argument` (mod.rs:2809-2816), and the runtime never builds an Argument's
   children (`children_to_recurse`, runtime.rs:658-660): it emits one empty row
   (runtime.rs:1285). `ev` already models that (`argument_fill_ignores_input`).

OPTIONAL MATCH is Apply-wrapped (`any_visited`, mod.rs:3044-3064), so its rows
survive, but the path is still missing: `RETURN p` gives null.

| here | there |
| --- | --- |
| `MP.pathVars` | the variables a match plan binds as paths: `PathBuilder` entries, a CondVarLenTraverse's elided path (mod.rs:2350-2387) |
| `BindsAllPaths` | the property: every named path of the pattern is bound by the plan |
| `compVars` | `QueryGraph::variables` ast.rs:816-822 (nodes, relationships, paths) |
| `planMatchArm` | `Planner::plan`'s MATCH arm mod.rs:3034-3082 |
| `MP.toPlan` | the `plan_match` result as an IR tree (operator kinds; payloads dropped) |
| `Plan.argIsLeaf` | the property: no `Argument` has a child |

Bug theorems: `bug2311_optional_path_null`, `bug2737_optional_path_null`,
`bug2743_bound_path_drops_rows`, `bug2744_bound_path_unfed_argument`,
`bug2727_alias_path_drops_rows`, `bug2725_call_then_bound_path_loses_var`.
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
relationship and for an unbound node-only component (`nodeOnly_unbound_binds_paths`);
fails for a bound node-only one (`boundNode_path_not_built`). -/
def BindsAllPaths (st : MSt) (fv : List Nat) (comps : List Comp) : Prop :=
  ∀ c ∈ comps, ∀ p ∈ c.paths, p.1 ∈ (planMatch st fv comps none).1.pathVars

/-! ## Root cause 1: the bound node-only branch -/

/-- The bound branch (mod.rs:1915-1962) builds no path, whatever paths the component has. -/
theorem nodeOnly_bound_no_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = true) : (nodeOnly st lw n ps).1.pathVars = [] := by
  unfold nodeOnly; simp only [hb, ↓reduceIte]
  split <;> rfl

/-- The unbound branch does (mod.rs:1977-1980). -/
theorem nodeOnly_unbound_binds_paths (st : MSt) (lw : LW) (n : QN) (ps : List (V × List V))
    (hb : st.bound n.alias = false) (p : V × List V) (hp : p ∈ ps) :
    p.1 ∈ (nodeOnly st lw n ps).1.pathVars := by
  have hne : ps.isEmpty = false := by cases ps <;> simp_all
  unfold nodeOnly; simp only [hb, Bool.false_eq_true, ↓reduceIte, hne]
  simp only [MP.pathVars, List.mem_append, List.mem_map]
  exact Or.inl ⟨p, hp, rfl⟩

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

/-- **Root cause 1, for every bound node and every named path over it**: `MATCH p = (n)`
/ `OPTIONAL MATCH p = (n)` with `n` already bound violates `BindsAllPaths`. -/
theorem boundNode_path_not_built (st : MSt) (fv : List Nat) (n : QN) (p : V)
    (hb : st.bound n.alias = true) :
    ¬ BindsAllPaths st fv [{ nodes := [n], rels := [], paths := [(p, [n.alias])] }] := by
  intro h
  have := h _ (List.mem_singleton_self _) (p, [n.alias]) (List.mem_singleton_self _)
  rw [planMatch_nodeOnly_pathVars, nodeOnly_bound_no_paths _ _ _ _ hb] at this
  simp at this

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

/-- `Planner::plan`'s MATCH arm (mod.rs:3034-3082) around the `plan_match` result
`m`; `vars` = `pattern.variables()`, `st` = the planner state *before* `plan_match`.
OPTIONAL: `Optional(new vars)`, under `Apply` if any variable is bound; MATCH:
`Apply(m)` only if **all** variables are bound, else `m` as is. -/
def planMatchArm (st : MSt) (vars : List V) (optional : Bool) (m : Plan) : Plan :=
  if optional then
    let ovs := (vars.filter fun v => !st.bound v).map V.id
    if vars.any st.bound then .node .apply [.node (.optional ovs) [addArgs m]]
    else .node (.optional ovs) [addArgs m]
  else if vars.all st.bound then .node .apply [addArgs m]
  else m

/-- A path variable is new, so a MATCH that declares one is never Apply-wrapped,
even when every node of it is bound. -/
theorem planMatchArm_path_unwrapped (st : MSt) (c : Comp) (m : Plan) (p : V × List V)
    (hp : p ∈ c.paths) (hnew : st.bound p.1 = false) :
    planMatchArm st (compVars c) false m = m := by
  have : (compVars c).all st.bound = false := by
    rw [List.all_eq_false]
    exact ⟨p.1, by simp [compVars]; exact Or.inr (Or.inr ⟨p.2, hp⟩), by simp [hnew]⟩
  simp [planMatchArm, this]

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

/-- The labelled bound node: `ExpandInto(Argument)`, no `PathBuilder`. -/
theorem planMatch_labelled (st : MSt) (n p : V) (l : String) (hb : st.bound n = true) :
    (planMatch st [] [pathComp { alias := n, labels := [l] } p] none).1.toPlan =
      .node (.hop .expandInto 0) [.node .argument []] := by
  simp only [pathComp]; rw [planMatch_nodeOnly]
  simp [nodeOnly, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan, MP.toPlanL]

/-- The unlabelled bound node: a bare `Argument`. -/
theorem planMatch_plain (st : MSt) (n p : V) (hb : st.bound n = true) :
    (planMatch st [] [pathComp { alias := n } p] none).1.toPlan = .node .argument [] := by
  simp only [pathComp]; rw [planMatch_nodeOnly]
  simp [nodeOnly, hb, lowerN, lowerInline, inlineAttrsToFilter, MP.toPlan]

theorem addArgs_eiArg : addArgs (.node (.hop .expandInto 0) [.node .argument []]) =
    .node (.hop .expandInto 0) [.node .argument []] := by
  simp [addArgs]

/-- Shared shape of the OPTIONAL repros: `MATCH (n0:L) OPTIONAL MATCH p0 = (n0) RETURN n0, p0`. -/
def optQuery (l : String) : List Plan :=
  [.node (.scan 0) [],
   planMatchArm boundSt (compVars (pathComp { alias := n0, labels := [l] } p0)) true
     (planMatch boundSt [] [pathComp { alias := n0, labels := [l] } p0] none).1.toPlan,
   .node (.project 0) []]

/-- The live EXPLAIN shape `Project(Apply(Scan, Optional(ExpandInto(Argument))))`. -/
theorem optQuery_plan (l : String) :
    rustStitch (optQuery l) =
      .node (.project 0) [.node .apply [.node (.scan 0) [],
        .node (.optional [1]) [.node (.hop .expandInto 0) [.node .argument []]]]] := by
  simp only [optQuery, planMatch_labelled boundSt n0 p0 l boundSt_n0, addArgs_eiArg]
  simp [planMatchArm, compVars, pathComp, boundSt, MSt.bound, n0, p0, addArgs_eiArg, refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem optQuery_rows (l : String) :
    (ev pathS (rustStitch (optQuery l)) [] ()).1 = [[(0, 7)]] := by
  rw [optQuery_plan]; decide

/-- **#2311** (`MATCH (n0:P) OPTIONAL MATCH p0=(n0) RETURN n0.id, p0` → Rust `1, null`,
C `1, [(0)]`). `plan_match` violates `BindsAllPaths`; the stitched plan
`Project(Apply(Scan, Optional(ExpandInto(Argument))))` (the live EXPLAIN) returns the
row without `p0`; openCypher returns it with the path. -/
theorem bug2311_optional_path_null :
    ¬ BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["P"] } p0] ∧
    (ev pathS (rustStitch (optQuery "P")) [] ()).1 = [[(0, 7)]] ∧
    ¬ RowsBind [0, 1] (ev pathS (rustStitch (optQuery "P")) [] ()).1 ∧
    [[(0, 7)]].flatMap pathMatch = [[(1, 100), (0, 7)]] := by
  refine ⟨boundNode_path_not_built _ _ _ _ boundSt_n0, optQuery_rows _, ?_, by decide⟩
  rw [optQuery_rows]; decide

/-- **#2737** (`MATCH (n1:X) OPTIONAL MATCH p6 = (n1) RETURN p6` → Rust `null`, C `[(0)]`):
the same shape as #2311. -/
theorem bug2737_optional_path_null :
    ¬ BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["X"] } p0] ∧
    ¬ RowsBind [1] (ev pathS (rustStitch (optQuery "X")) [] ()).1 := by
  refine ⟨boundNode_path_not_built _ _ _ _ boundSt_n0, ?_⟩
  rw [optQuery_rows]; decide

/-- Shared shape of the MATCH repros: `MATCH (n:L) WITH … MATCH p0 = (n) RETURN …`;
`st` is the state after the WITH (`n` bound), `ret` the RETURN plan (`Project` or `Aggregate`). -/
def withQuery (st : MSt) (n : V) (l : String) (ret : Plan) : List Plan :=
  [.node (.scan 0) [], .node (.project 0) [],
   planMatchArm st (compVars (pathComp { alias := n, labels := [l] } p0)) false
     (planMatch st [] [pathComp { alias := n, labels := [l] } p0] none).1.toPlan,
   ret]

theorem withQuery_unwrapped (st : MSt) (n : V) (l : String) (hb : st.bound n = true)
    (hp : st.bound p0 = false) :
    planMatchArm st (compVars (pathComp { alias := n, labels := [l] } p0)) false
      (planMatch st [] [pathComp { alias := n, labels := [l] } p0] none).1.toPlan =
      .node (.hop .expandInto 0) [.node .argument []] := by
  rw [planMatch_labelled st n p0 l hb]
  exact planMatchArm_path_unwrapped st _ _ (p0, [n]) (List.mem_singleton_self _) hp

/-- The live EXPLAIN shape: the WITH and the first MATCH under the `Argument`. -/
theorem withQuery_project (st : MSt) (n : V) (l : String) (hb : st.bound n = true)
    (hp : st.bound p0 = false) (k : Nat) :
    rustStitch (withQuery st n l (.node (.project k) [])) =
      .node (.project k) [.node (.hop .expandInto 0) [.node .argument [.node (.project 0) [.node (.scan 0) []]]]] := by
  simp only [withQuery, withQuery_unwrapped st n l hb hp]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

theorem withQuery_aggregate (st : MSt) (n : V) (l : String) (hb : st.bound n = true)
    (hp : st.bound p0 = false) (k : Nat) :
    rustStitch (withQuery st n l (.node (.aggregate k) [])) =
      .node (.aggregate k) [.node (.hop .expandInto 0) [.node .argument [.node (.project 0) [.node (.scan 0) []]]]] := by
  simp only [withQuery, withQuery_unwrapped st n l hb hp]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2744** (`MATCH (n0:A) WITH n0 MATCH p=(n0) RETURN n0.k8, p` → Rust 0 rows,
C one row `1, [(0)]`). Both root causes: no `PathBuilder` (`BindsAllPaths` fails),
and the unwrapped `Argument` gets the previous clauses as a child — the live EXPLAIN
`Project(ExpandInto(Argument(Project(Scan))))` — violating `argIsLeaf`; the row is lost. -/
theorem bug2744_bound_path_unfed_argument :
    ¬ BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["A"] } p0] ∧
    (rustStitch (withQuery boundSt n0 "A" (.node (.project 0) []))).argIsLeaf = false ∧
    (ev pathS (rustStitch (withQuery boundSt n0 "A" (.node (.project 0) []))) [] ()).1 = [] ∧
    (ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch =
      [[(1, 100), (0, 7)]] := by
  refine ⟨boundNode_path_not_built _ _ _ _ boundSt_n0, ?_, ?_, by decide⟩
  · rw [withQuery_project boundSt n0 _ rfl rfl]; decide
  · rw [withQuery_project boundSt n0 _ rfl rfl]; decide

/-- **#2743** (`MATCH (n0:V) WITH * MATCH p0=(n0) RETURN count(*)` → Rust 0, C 1). -/
theorem bug2743_bound_path_drops_rows :
    ¬ BindsAllPaths boundSt [] [pathComp { alias := n0, labels := ["V"] } p0] ∧
    (rustStitch (withQuery boundSt n0 "V" (.node (.aggregate 0) []))).argIsLeaf = false ∧
    (ev pathS (rustStitch (withQuery boundSt n0 "V" (.node (.aggregate 0) []))) [] ()).1 = [[(9, 0)]] ∧
    pathS.agg 0 [] ((ev pathS (.node (.project 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch) =
      [[(9, 1)]] := by
  refine ⟨boundNode_path_not_built _ _ _ _ boundSt_n0, ?_, ?_, by decide⟩
  · rw [withQuery_aggregate boundSt n0 _ rfl rfl]; decide
  · rw [withQuery_aggregate boundSt n0 _ rfl rfl]; decide

/-- `WITH n0 AS alias4`: the alias is a fresh variable of the next scope. -/
def alias4 : V := ⟨0, 1⟩
def aliasSt : MSt := { visited := [alias4], verified := [] }

/-- **#2727** (`MATCH (n0:A) WITH n0 AS alias4 MATCH (alias4), p1=(alias4) RETURN alias4.k0`
→ Rust 0 rows, C `'x'`; the full repro errors `Variable alias4 not found`). The pattern
has one node-only component carrying `p1` (`connected_components`, ast.rs:926-928);
the binder copies `:A` onto `alias4`, so the plan is `ExpandInto(Argument)` again. -/
theorem bug2727_alias_path_drops_rows :
    ¬ BindsAllPaths aliasSt [] [pathComp { alias := alias4, labels := ["A"] } p0] ∧
    (rustStitch (withQuery aliasSt alias4 "A" (.node (.project 0) []))).argIsLeaf = false ∧
    (ev pathS (rustStitch (withQuery aliasSt alias4 "A" (.node (.project 0) []))) [] ()).1 = [] := by
  refine ⟨boundNode_path_not_built _ _ _ _ rfl, ?_, ?_⟩
  · rw [withQuery_project aliasSt alias4 _ rfl rfl]; decide
  · rw [withQuery_project aliasSt alias4 _ rfl rfl]; decide

/-- `MATCH (n0) CALL db.labels() YIELD label MATCH (n0), p0=(n0) RETURN n0.k11, label, p0`. -/
def callQuery : List Plan :=
  [.node (.scan 0) [], .node (.procCall 0) [],
   planMatchArm boundSt (compVars (pathComp { alias := n0 } p0)) false
     (planMatch boundSt [] [pathComp { alias := n0 } p0] none).1.toPlan,
   .node (.project 0) []]

theorem callQuery_plan :
    rustStitch callQuery =
      .node (.project 0) [.node .argument [.node (.procCall 0) [.node (.scan 0) []]]] := by
  have hu := planMatchArm_path_unwrapped boundSt (pathComp { alias := n0 } p0)
    (.node .argument []) (p0, [n0]) (List.mem_singleton_self _) rfl
  simp only [callQuery, planMatch_plain boundSt n0 p0 boundSt_n0, hu]
  simp [refill, rustStitch, stitchLoop, slotFirst, slotLoop, slotWith, walkFirst, walkLoop, Plan.get, projDescend, descendClause, descendOne, Plan.depth, insertStep, insertAt, Plan.modify, Plan.push0, loopPass, isApply, clauseMin, applyChain]

/-- **#2725** (`… CALL algo.BFS(…) YIELD … MATCH (n0), p0 = (n0), (n0) WHERE … RETURN *`
→ Rust `Variable n0 not found`, C no error; minimal with `db.labels()`: Rust
`Variable n0 not found`, C `1, A, [(0)]`). The unlabelled bound node is a bare
`Argument`; the CALL and the first MATCH are stitched under it (live EXPLAIN
`Project(Argument(ProcedureCall(All Node Scan)))`), so the one output row is empty:
`n0`, `label` and `p0` are all unbound. -/
theorem bug2725_call_then_bound_path_loses_var :
    ¬ BindsAllPaths boundSt [] [pathComp { alias := n0 } p0] ∧
    (rustStitch callQuery).argIsLeaf = false ∧
    (ev pathS (rustStitch callQuery) [] ()).1 = [[]] ∧
    ¬ RowsBind [0] (ev pathS (rustStitch callQuery) [] ()).1 ∧
    (ev pathS (.node (.procCall 0) [.node (.scan 0) []]) [] ()).1.flatMap pathMatch =
      [[(1, 100), (2, 55), (0, 7)]] := by
  refine ⟨boundNode_path_not_built _ _ _ _ boundSt_n0, ?_, ?_, ?_, by decide⟩
  · rw [callQuery_plan]; decide
  · rw [callQuery_plan]; decide
  · rw [callQuery_plan]; decide

/-! ## What a fix must give: the unbound branch is already right

Without the earlier binding (`MATCH p0 = (n0:A)` alone) the plan is
`PathBuilder(Scan)` and the property holds. -/
theorem unbound_path_built (l : String) :
    BindsAllPaths { visited := [], verified := [] } [] [pathComp { alias := n0, labels := [l] } p0] := by
  intro c hc p hp
  simp only [List.mem_singleton] at hc; subst hc
  simp only [pathComp, List.mem_singleton] at hp; subst hp
  simp only [pathComp]; rw [planMatch_nodeOnly_pathVars]
  exact nodeOnly_unbound_binds_paths _ _ _ _ rfl _ (List.mem_singleton_self _)

end PlannerBuild.E
