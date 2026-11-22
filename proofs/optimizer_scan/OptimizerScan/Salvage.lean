import OptimizerScan.ScanTree
import OptimizerScan.ScanOrder

/-
# Salvaged filters: what `filters_of` hands `select_scan_node` to re-attach

`ScanOrder.salvage_wellScoped` takes the premise "every salvaged filter reads only variables
the new chain binds". This file models where those filters come from, with the variables each
reads as the Rust computes them, and decides the premise.

| here | there |
| --- | --- |
| `ScanTree.IRk.filter vs`, `filtersOf`  | `IR::Filter(expr)`, `filters_of` `select_scan_node.rs`; `vs` = the `(id, scope_id)` of every `ExprIR::Variable` of the filter (`collect`) |
| `ScanTree.isPlannerScan`, `scanInputOf` | `is_planner_scan_subtree` (a scan is accepted whatever its children are), `scan_input_of` (#3099) |
| `inputVars`, `seedBound` | the variables bound below the first hop of the rebuilt plan: `best_node` (`initial_bound`) and the ids the preserved input binds (`collect_output_aliases` of each of its operators; `initial_bound` also adds `collect_subtree_variables`) |
| `salvagedSteps`                        | `salvaged_filters` wrapped above the chain |
| `rebuilt`                              | `make_scan_subtree(&node, in_merge, preserved_argument, ..)` |
| `leafSteps`                            | the no-swap leaf rewrite: `salvaged_filters` inside the new scan, or `filters_above` the traverse when `planner_scan_alias(child) ≠ from` (#2981) |
| `leafSteps3594`                        | the same rewrite since #3594: filters salvaged only `if has_planner_scan`; a transparent child keeps its own |

Since #3099 the pruned planner scan's whole input (`scan_input_of`: an `Argument`, or an earlier
clause's `Unwind`/`WITH` rows) is re-attached beneath the new scan (`ScanTree.scanInputOf_makeScanSubtree`),
so the variables it binds are in the seed. On `Filter(b.v = k) → Scan(b) → Unwind(k)` (#2972:
`WITH 1 AS k MATCH (b:B) WHERE b.v = k MATCH (b)-[:H]->(c:C) RETURN count(*)`, C 1) the premise
now holds (`premise_holds`) and the row generate-and-test keeps is kept (`salvage_keeps_row`).
`premise_of_spine` gives sufficient conditions under which the premise holds (the converse is not proved).

The bound sets here are keyed on the variable id alone, as the Rust keys them: `initial_bound` and `placed_bound`
are `HashSet<u32>`, and a filter is placed when its `HashSet<u32>` of ids is a subset (notes that the id-only sets decide nothing scope-sensitive). So `salvagedSteps` keeps
`v.1` of each `(id, scope_id)` exactly where the Rust drops the scope; `filtersOf` still carries the pairs.
-/
namespace OptimizerScan.Salvage
open OptimizerScan.ScanTree OptimizerScan.ScanOrder

/-- Ids bound by an `Argument` leaf (`Argument(None)` binds nothing the model knows). -/
def argVars : Option IRk → List Nat
  | some (.argument (some vs)) => vs.map (·.1)
  | _ => []

/-- Ids the input preserved beneath the new scan binds: `collect_output_aliases` of each of its
operators (the part of `initial_bound`'s outer ids that names what each operator outputs). -/
def inputIds (t : PT) : List Nat := t.walk.flatMap collectOutputAliases

def inputVars (child : PT) : List Nat := ((scanInputOf child).map inputIds).getD []

/-- The bound set below the first hop of the rebuilt plan: the new scan's node and what the
preserved input binds. -/
def seedBound (best : Nat) (child : PT) : List Nat := best :: inputVars child

/-- **PROVEN**: `seedBound` is `ScanOrder.initialBoundOuter` with no kept child and the planner scan's
input (not transparent) preserved, so `initialBoundOuter_wellScoped` applies to it. -/
theorem seedBound_initialBoundOuter (best : Nat) (child : PT) (chain : List Nat) :
    seedBound best child =
      initialBoundOuter best none ((scanInputOf child).map inputIds) chain false := by
  cases h : scanInputOf child <;> simp [seedBound, inputVars, initialBoundOuter, initialBound, outerSeed, h]

/-- The `Argument` `argument_leaf_of` finds is on the walk. -/
theorem argumentLeafOf_mem_walk : ∀ (t : PT) (a : IRk), argumentLeafOf t = some a → a ∈ t.walk
  | .node ir cs, a, h => by
    unfold argumentLeafOf at h
    split at h
    · cases h; simp [PT.walk]
    · split at h
      · next c =>
        have := argumentLeafOf_mem_walk c a h
        simp [PT.walk, PT.walkL, this]
      · cases h

/-- **PROVEN** (#3099 keeps what the old code kept): on a planner scan, the variables of the
`Argument` leaf `argument_leaf_of` used to preserve are among those the preserved input binds. -/
theorem argVars_sub_inputVars : ∀ (child : PT), isPlannerScan child = true →
    ∀ x ∈ argVars (argumentLeafOf child), x ∈ inputVars child
  | .node ir cs, hp, x, hx => by
    cases ha : scanAlias ir with
    | some al =>
      have hd : scanDepth (.node ir cs) = some 0 := by unfold scanDepth; simp [ha]
      have hnarg : ∀ v, ir ≠ .argument v := by intro v e; subst e; simp [scanAlias] at ha
      match cs, hx with
      | [c], hx =>
        have hl : argumentLeafOf (.node ir [c]) = argumentLeafOf c := by
          cases ir <;> simp_all [argumentLeafOf]
        rw [hl] at hx
        have hin : scanInputOf (.node ir [c]) = some c := by
          simp [scanInputOf, plannerScanIdx, hd, getAt, PT.cs]
        rw [inputVars, hin]
        cases hc : argumentLeafOf c with
        | none => simp [hc, argVars] at hx
        | some a =>
          rw [hc] at hx
          have hw := argumentLeafOf_mem_walk c a hc
          simp only [Option.map_some, Option.getD_some, inputIds, List.mem_flatMap]
          refine ⟨a, hw, ?_⟩
          match a, hx with
          | .argument (some vs), hx => simpa [collectOutputAliases, projectedOutputs, argVars] using hx
      | [], hx =>
        have hl : argumentLeafOf (.node ir []) = none := by cases ir <;> simp_all [argumentLeafOf]
        simp [hl, argVars] at hx
      | c1 :: c2 :: cs', hx =>
        have hl : argumentLeafOf (.node ir (c1 :: c2 :: cs')) = none := by
          cases ir <;> simp_all [argumentLeafOf]
        simp [hl, argVars] at hx
    | none =>
      have hw : isWrapper ir = true := by
        cases hwi : isWrapper ir
        · unfold isPlannerScan scanDepth at hp; simp [ha, hwi] at hp
        · rfl
      match cs, hp with
      | [c], hp =>
        have hpc : isPlannerScan c = true := by
          cases hd : scanDepth c <;> simp_all [isPlannerScan, scanDepth]
        have hl : argumentLeafOf (.node ir [c]) = argumentLeafOf c := by
          cases ir <;> simp_all [isWrapper, argumentLeafOf]
        rw [hl] at hx
        have := argVars_sub_inputVars c hpc x hx
        simpa [inputVars, scanInputOf_wrap ir hw c] using this
      | [], hp => simp [isPlannerScan, scanDepth, ha, hw] at hp
      | _ :: _ :: _, hp => simp [isPlannerScan, scanDepth, ha, hw] at hp

/-- The salvaged filters as `Step`s: variable ids (`collect_expr_variables` keeps ids). -/
def salvagedSteps (child : PT) (pred : List Var → (Nat → Nat) → Bool) : List Step :=
  (filtersOf child).map (fun vs => Step.filt (vs.map (·.1)) (pred vs))

/-- The premise of `salvage_wellScoped` for the filters of `child`. -/
def SalvagePremise (child : PT) (B : List Nat) (steps : List Step) : Prop :=
  ∀ vs ∈ filtersOf child, ∀ v ∈ vs, v.1 ∈ boundAfter B steps

/-- Variables bound by a chain are still bound after more steps. -/
theorem mem_boundAfter_mono : ∀ (steps : List Step) (B : List Nat) (v : Nat), v ∈ B →
    v ∈ boundAfter B steps
  | [], _, _, h => h
  | .hop h :: rest, B, v, hv => mem_boundAfter_mono rest _ v (by simp [hv])
  | .filt _ _ :: rest, B, v, hv => mem_boundAfter_mono rest B v hv

/-- **PROVEN**: with the premise, `salvage_wellScoped` applies to the salvaged filters of `child`
(the statement of `salvage_wellScoped` is unchanged; this instantiates it). -/
theorem salvaged_wellScoped (child : PT) (B : List Nat) (steps : List Step)
    (pred : List Var → (Nat → Nat) → Bool) (hw : wellScoped B steps)
    (hp : SalvagePremise child B steps) :
    wellScoped B (steps ++ salvagedSteps child pred) := by
  have := salvage_wellScoped B steps hw ((filtersOf child).map (fun vs => (vs.map (·.1), pred vs)))
    (by
      intro f hf v hv
      simp only [List.mem_map] at hf
      obtain ⟨vs, hvs, rfl⟩ := hf
      simp only [List.mem_map] at hv
      obtain ⟨w, hw', rfl⟩ := hv
      exact hp vs hvs w hw')
  simpa [salvagedSteps, Function.comp_def] using this

/-- **PROVEN** (a sufficient condition for the premise): if the spine's filters read only the scanned node and
variables the preserved input binds, and those and the scanned node are bound by the new chain, the premise holds.
These are conditions on the planner's scan subtree, not something `select_scan_node` checks; the converse is not proved. -/
theorem premise_of_spine (child : PT) (a : Var) (B : List Nat) (steps : List Step)
    (hb : a.1 ∈ boundAfter B steps)
    (hv : ∀ vs ∈ filtersOf child, ∀ v ∈ vs, v = a ∨ v.1 ∈ inputVars child)
    (hargs : ∀ x ∈ inputVars child, x ∈ boundAfter B steps) :
    SalvagePremise child B steps := by
  intro vs hvs v hv'
  rcases hv vs hvs v hv' with rfl | h
  · exact hb
  · exact hargs _ h

/-- **PROVEN**: from the seed, the premise holds whenever the spine's filters read only the new scan's
node and what the preserved input binds (the input is beneath the new scan since #3099). -/
theorem premise_of_seed (child : PT) (a : Var) (steps : List Step)
    (hv : ∀ vs ∈ filtersOf child, ∀ v ∈ vs, v = a ∨ v.1 ∈ inputVars child) :
    SalvagePremise child (seedBound a.1 child) steps :=
  premise_of_spine child a _ steps (mem_boundAfter_mono _ _ _ (by simp [seedBound]))
    hv (fun x hx => mem_boundAfter_mono _ _ _ (by simp [seedBound, hx]))

/-- Does a plan tree contain an `Unwind` within `n` levels? -/
def hasUnwindF : Nat → PT → Bool
  | 0, _ => false
  | n + 1, .node ir cs => (match ir with | .unwind _ => true | _ => false) || cs.any (hasUnwindF n)

def hasUnwind (t : PT) : Bool := hasUnwindF 8 t

/-! ### #2972 fixed by #3099: `Filter(b.v = k) → Scan(b) → Unwind(k)`

Variable ids: b = 1, k = 2, c = 3 (edge 10). The planner stitches `UNWIND`/`WITH ... AS k` below the
next clause's scan, so the child of the bottom CondTraverse is `Filter → NodeByLabelScan → Unwind`. -/

def b : Var := (1, 0)
def k : Var := (2, 0)
def exChild : PT :=
  .node (.filter [b, k]) [.node (.labelScan b [0]) [.node (.unwind k) []]]

/-- The pass takes it for a planner scan, finds its `Unwind` input, and salvages the filter. -/
theorem exChild_shape :
    isPlannerScan exChild = true ∧ scanInputOf exChild = some (.node (.unwind k) []) ∧
    filtersOf exChild = [[b, k]] ∧ inputVars exChild = [k.1] :=
  ⟨by decide, rfl, by decide, by decide⟩

/-- **PROVEN** (#3099): the rebuilt scan subtree keeps the `Unwind` that binds `k`. -/
theorem rebuilt_keeps_unwind :
    hasUnwind (makeScanSubtree b [0] false (scanInputOf exChild) []) = true := by
  decide

/-- The rebuilt chain `(b)-[10]->(c)` from the seed `[b, k]`. -/
def exSteps : List Step := [.hop ⟨1, 3, 10, 0, false⟩]

/-- **PROVEN** (#3099): the salvaged filter reads `b` and `k`, both bound by the seed. -/
theorem premise_holds : SalvagePremise exChild (seedBound b.1 exChild) exSteps := by
  apply premise_of_seed exChild b exSteps
  intro vs hvs v hv
  simp [filtersOf, exChild] at hvs
  subst hvs
  simp at hv
  rcases hv with rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (by decide)

/-- The one edge `1 -[10]-> 3`. -/
def exE : Store := fun r a z e => r == 0 && a == 1 && z == 3 && e == 10

/-- The property store: `exV n` is node `n`'s `v`. Node 1 (`b`) has `v = 1`; every other node `v = 0`. -/
def exV : Nat → Nat := fun n => if n == 1 then 1 else 0

/-- The row of the match: `b` is node 1, `c` node 3, the edge 10, and `k` (the `UNWIND`/`WITH` variable) the
value 1. A variable's value is `σ id`. -/
def exσ : Nat → Nat := fun x => if x == k.1 then 1 else x

/-- The salvaged filter `b.v = k`: the property `v` of the node `b` is bound to, compared with the value of `k`.
It reads a variable list `vs` only to be the filter of `filtersOf`; its truth is the equality. -/
def exPred : List Var → (Nat → Nat) → Bool := fun _ σ => exV (σ b.1) == σ k.1

/-- `exPred` is a real equality, not a constant: it holds on the row where `k = 1 = b.v`, and fails on the
row where `k = 2`. -/
theorem exPred_is_eq :
    exPred [b, k] exσ = true ∧ exPred [b, k] (fun x => if x == k.1 then 2 else x) = false := by decide

/-- **PROVEN** (#2972 fixed by #3099): the chain the pass builds is well-scoped from the seed, so it keeps
the row generate-and-test (Cypher) accepts (the hop holds and `b.v = k`, both 1). -/
theorem salvage_keeps_row :
    (exSteps ++ salvagedSteps exChild exPred).all (fullHolds exE exσ) = true ∧
    runScoped exE exσ (seedBound b.1 exChild) (exSteps ++ salvagedSteps exChild exPred) = true ∧
    wellScoped (seedBound b.1 exChild) (exSteps ++ salvagedSteps exChild exPred) := by
  have hw : wellScoped (seedBound b.1 exChild) (exSteps ++ salvagedSteps exChild exPred) :=
    salvaged_wellScoped exChild _ exSteps exPred (by simp [wellScoped, exSteps]) premise_holds
  refine ⟨by decide, ?_, hw⟩
  rw [runScoped_wellScoped exE exσ _ _ hw]
  decide

/-! ### The input's own Filters are salvaged as well

`filters_of` walks the single-child spine through the scan into its input, so a `Filter` inside the
input (`UNWIND ... AS i WITH i WHERE i > 1 WITH 0 AS z MATCH (d:L) WHERE d.v = 1 MATCH (d)-[:R]->()`)
is both salvaged and, since #3099, kept beneath the new scan: the rebuilt subtree carries it twice.
In this id-keyed model the copy is harmless (its ids are seeded); #3594's transparent path stopped
salvaging such Filters (`transparent_filters_once`) because a copy above a projection that reused
the id read the wrong variable. Not confirmed live for the planner-scan path. -/

def d : Var := (5, 0)
def i : Var := (6, 0)
def exChild2 : PT :=
  .node (.filter [d]) [.node (.labelScan d [0]) [.node (.filter [i]) [.node (.unwind i) []]]]

/-- **PROVEN**: rebuilt with its salvaged filters inside (the same-node leaf rewrite), the subtree
carries the input's `Filter` on `i` twice. -/
theorem input_filters_twice :
    filtersOf exChild2 = [[d], [i]] ∧
    filtersOf (makeScanSubtree d [0] false (scanInputOf exChild2) (filtersOf exChild2)) =
      [[d], [i], [i]] := by
  decide

/-! ### The no-swap leaf rewrite (`select_scan_node.rs`)

When the bottom traverse keeps its direction (`effectively_leaf`) and its child is a planner scan,
the scan is pruned and `make_scan_subtree(from, ..)` scans the traverse's `from`. Before #2981 the
spine's filters always went inside the new scan subtree (`salvaged_filters`), so a filter on the
pruned scan's own node, when that node is the traverse's *other* endpoint (an earlier clause stitched
in, `MATCH (a {v: 1}) MATCH (a)<-[]-(b {v: 0})`), was evaluated before anything bound it. Now, when
`planner_scan_alias(child) ≠ (from.id, from.scope)`, they go above the traverse (`filters_above`,
:1412-1426, :1454-1477). -/

/-- The chain this branch builds from the seed (the new scan's node and the `Argument`): the
salvaged filters inside the scan subtree (below the hop), or, when the pruned scan was for another
node, above the hop (#2981). -/
def leafSteps (fr : Var) (child : PT) (h : Hop) (pred : List Var → (Nat → Nat) → Bool) :
    List Step :=
  if isPlannerScan child && (plannerScanAlias child).any (· != fr) then
    .hop h :: salvagedSteps child pred
  else salvagedSteps child pred ++ [.hop h]

/-- **PROVEN**: a scan for the same node keeps its filters inside, as before. -/
theorem leafSteps_same (fr : Var) (child : PT) (h : Hop) pred
    (hs : plannerScanAlias child = some fr) :
    leafSteps fr child h pred = salvagedSteps child pred ++ [.hop h] := by
  simp [leafSteps, hs]

/-- **PROVEN**: when the pruned scan was for the hop's other endpoint `a` and its filters read only
`a` and what its preserved input binds, the new placement is well-scoped from the seed. -/
theorem leafSteps_other_wellScoped (fr a : Var) (child : PT) (h : Hop) pred
    (hs : plannerScanAlias child = some a) (hne : a ≠ fr) (ha : a.1 = h.src ∨ a.1 = h.dst)
    (hv : ∀ vs ∈ filtersOf child, ∀ v ∈ vs, v = a ∨ v.1 ∈ inputVars child) :
    wellScoped (seedBound fr.1 child) (leafSteps fr child h pred) := by
  have hp : isPlannerScan child = true := by
    have := (plannerScan_agree child).2.1; rw [this, hs]; rfl
  have e : leafSteps fr child h pred = [.hop h] ++ salvagedSteps child pred := by
    simp [leafSteps, hp, hs, hne]
  rw [e]
  apply salvaged_wellScoped child _ [.hop h] pred (by simp [wellScoped])
  apply premise_of_spine child a _ _ _ hv
  · intro x hx
    simp [boundAfter, seedBound, hx]
  · simp only [boundAfter, List.mem_cons]; omega

namespace Leaf2981

/-- `MATCH (a {v: 1}) MATCH (a)<-[]-(b {v: 0})`: a = 0, b = 1, the edge 2 (stored `b → a`). The
traverse starts from `b`; its child is the first clause, `Filter(a.v = 1) → Scan(a)`. -/
def a : Var := (0, 0)
def bN : Var := (1, 0)
def child : PT := .node (.filter [a]) [.node (.allNodeScan a) []]
def hop : Hop := ⟨1, 0, 2, 0, false⟩

/-- Node 11 -[12]-> node 10; `σ x = x + 10`. Node 10 (`a`) has `v = 1`. -/
def E : Store := fun r x y e => r == 0 && x == 11 && y == 10 && e == 12
def σ : Nat → Nat := fun x => x + 10
def pred : List Var → (Nat → Nat) → Bool := fun _ σ => (if σ a.1 == 10 then 1 else 0) == 1

/-- **PROVEN** (#2981): the filter on `a` sits above the hop that binds `a`, so the placement is
well-scoped and the row generate-and-test keeps is kept. -/
theorem fixed2981_filter_above_hop :
    (leafSteps bN child hop pred).all (fullHolds E σ) = true ∧
    wellScoped (seedBound bN.1 child) (leafSteps bN child hop pred) ∧
    runScoped E σ (seedBound bN.1 child) (leafSteps bN child hop pred) = true := by
  refine ⟨by decide, ?_, by decide⟩
  exact leafSteps_other_wellScoped bN a child hop pred (by decide) (by decide) (by decide)
      (by intro vs hvs v hv; simp [filtersOf, child] at hvs; subst hvs; simp at hv; exact Or.inl hv)

end Leaf2981

/-! ### The no-swap leaf rewrite over a transparent child (#3594, `select_scan_node.rs`)

Since #3594 the bottom traverse's child may be an outer subtree (`transparent_child`: an `Unwind`,
a `Project`, an earlier clause's plan) that binds none of the chain's variables. It is not discarded:
it rides whole beneath the new scan (`make_argument()`, `make_scan_subtree(.., Some(child), ..)`),
keeping its own Filters. Filters are salvaged (inside, or `filters_above`) only when the child is a
discarded planner scan (`if has_planner_scan`); otherwise none are. -/

/-- The chain the no-swap leaf rewrite builds since #3594: the #2981 placement over a planner scan,
and the bare hop otherwise (a transparent child keeps its Filters beneath the new scan). -/
def leafSteps3594 (hasPlanner : Bool) (fr : Var) (child : PT) (h : Hop)
    (pred : List Var → (Nat → Nat) → Bool) : List Step :=
  if hasPlanner then leafSteps fr child h pred else [.hop h]

/-- **PROVEN** (#3594): over a planner scan the rewrite is the #2981 one, so `leafSteps_same`,
`leafSteps_other_wellScoped` and `Leaf2981.fixed2981_filter_above_hop` still describe it. -/
theorem leafSteps3594_planner (fr : Var) (child : PT) (h : Hop) pred :
    leafSteps3594 true fr child h pred = leafSteps fr child h pred := rfl

/-- **PROVEN** (#3594, `17547791c`): over a transparent child nothing is salvaged, so the chain is
the bare hop, well-scoped from any seed. -/
theorem leafSteps3594_transparent_wellScoped (fr : Var) (child : PT) (h : Hop) pred (B : List Nat) :
    leafSteps3594 false fr child h pred = [.hop h] ∧
    wellScoped B (leafSteps3594 false fr child h pred) := by
  simp [leafSteps3594, wellScoped]

/-- **PROVEN** (#3594, `17547791c`): the subtree the transparent rewrite builds carries each of the
child's own Filters exactly once (beneath the new scan, where its variables are bound), not a second
copy outside it. Before `17547791c` the child's spine Filters were also salvaged
(`filtersOf_makeScanSubtree_tree` with `fs := filtersOf t` gives them twice); the copy above a later
projection that reused the variable's id read the wrong value:
`UNWIND [1, 2, 3] AS i WITH i WHERE i > 1 WITH 0 AS z MATCH (d:L)-[:R]->() RETURN count(*)`. -/
theorem transparent_filters_once (alias : Var) (labels : List Nat) (pend : Bool) (t : PT) :
    filtersOf (makeScanSubtree alias labels pend (some t) []) = filtersOf t ∧
    filtersOf (makeScanSubtree alias labels pend (some t) (filtersOf t)) =
      filtersOf t ++ filtersOf t := by
  constructor
  · simpa using filtersOf_makeScanSubtree_tree alias labels pend t []
  · exact filtersOf_makeScanSubtree_tree alias labels pend t (filtersOf t)

end OptimizerScan.Salvage
