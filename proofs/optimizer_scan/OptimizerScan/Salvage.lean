import OptimizerScan.ScanTree
import OptimizerScan.ScanOrder

/-
# Salvaged filters: what `filters_of` hands `select_scan_node` to re-attach (#2390)

`ScanOrder.salvage_wellScoped` takes the premise "every salvaged filter reads only variables
the new chain binds". This file models where those filters come from, with the variables each
reads as the Rust computes them, and decides the premise.

| here | there |
| --- | --- |
| `ScanTree.IRk.filter vs`, `filtersOf`  | `IR::Filter(expr)`, `filters_of` `select_scan_node.rs:250`; `vs` = the `(id, scope_id)` of every `ExprIR::Variable` of the filter (`collect`, `:160`) |
| `ScanTree.isPlannerScan`, `argumentLeafOf` | `is_planner_scan_subtree` `:315` (a scan is accepted whatever its children are), `argument_leaf_of` `:276` |
| `argVars`, `seedBound` | the variables bound below the first hop of the rebuilt plan: `best_node` (`initial_bound`, `:979-986`) and the preserved `Argument` leaf (`preserved_argument`, `:952`, `:1181`) |
| `salvagedSteps`                        | `salvaged_filters` wrapped above the chain (`:1240-1246`, `:1156-1176`) |
| `rebuilt`                              | `make_scan_subtree(&node, in_merge, preserved_argument, ..)` (`:214`) |

Result: the premise is **not** a fact about the Rust. `is_planner_scan_subtree` accepts a scan
with *any* child, but `argument_leaf_of` keeps only an `Argument`; a child such as `Unwind`
is pruned with the scan (`prune`, `:1131`, `:1294`). The filters on the spine then read a
variable (`k`) that nothing in the rebuilt plan binds (`premise_fails`), and `runScoped` loses a
row generate-and-test keeps (`salvage_loses_row`). Query:
`WITH 1 AS k MATCH (b:B) WHERE b.v = k MATCH (b)-[:H]->(c:C) RETURN count(*)` (Rust 0, C 1; also
`UNWIND [1,1] AS k MATCH (b:B) WHERE b.v = 1 MATCH (b)-[:H]->(c:C)` Rust 1, C 2 — the `Unwind`
operator itself is gone from the plan). `premise_of_spine` gives sufficient conditions under which the premise holds (the converse is not proved).

The bound sets here are keyed on the variable id alone, as the Rust keys them: `initial_bound` and `placed_bound`
are `HashSet<u32>`, and a filter is placed when its `HashSet<u32>` of ids is a subset (`:979-986`, `:1185`,
`:1221-1222`; `:424-428` notes that the id-only sets decide nothing scope-sensitive). So `salvagedSteps` keeps
`v.1` of each `(id, scope_id)` exactly where the Rust drops the scope; `filtersOf` still carries the pairs.
-/
namespace OptimizerScan.Salvage
open OptimizerScan.ScanTree OptimizerScan.ScanOrder

/-- Ids bound by the preserved `Argument` leaf (`Argument(None)` binds nothing the model knows). -/
def argVars : Option IRk → List Nat
  | some (.argument (some vs)) => vs.map (·.1)
  | _ => []

/-- The bound set below the first hop of the rebuilt plan: the new scan's node and the
preserved `Argument`'s variables. -/
def seedBound (best : Nat) (child : PT) : List Nat := best :: argVars (argumentLeafOf child)

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
the `Argument`'s variables, and the scanned node is bound by the new chain, the premise holds. These are
conditions on the planner's scan subtree, not something `select_scan_node` checks; the converse is not proved. -/
theorem premise_of_spine (child : PT) (a : Var) (B : List Nat) (steps : List Step)
    (hb : a.1 ∈ boundAfter B steps)
    (hv : ∀ vs ∈ filtersOf child, ∀ v ∈ vs, v = a ∨ v.1 ∈ argVars (argumentLeafOf child))
    (hargs : ∀ x ∈ argVars (argumentLeafOf child), x ∈ boundAfter B steps) :
    SalvagePremise child B steps := by
  intro vs hvs v hv'
  rcases hv vs hvs v hv' with rfl | h
  · exact hb
  · exact hargs _ h

/-- Does a plan tree contain an `Unwind` within `n` levels? -/
def hasUnwindF : Nat → PT → Bool
  | 0, _ => false
  | n + 1, .node ir cs => (match ir with | .unwind _ => true | _ => false) || cs.any (hasUnwindF n)

def hasUnwind (t : PT) : Bool := hasUnwindF 8 t

/-! ### The counterexample: `Filter(b.v = k) → Scan(b) → Unwind(k)`

Variable ids: b = 1, k = 2, c = 3 (edge 10). The planner stitches `UNWIND`/`WITH ... AS k` below the
next clause's scan, so the child of the bottom CondTraverse is `Filter → NodeByLabelScan → Unwind`. -/

def b : Var := (1, 0)
def k : Var := (2, 0)
def exChild : PT :=
  .node (.filter [b, k]) [.node (.labelScan b [0]) [.node (.unwind k) []]]

/-- The pass takes it for a planner scan, finds no `Argument` leaf to keep, and salvages the filter. -/
theorem exChild_shape :
    isPlannerScan exChild = true ∧ argumentLeafOf exChild = none ∧
    filtersOf exChild = [[b, k]] ∧ hasUnwind exChild = true := by decide

/-- The rebuilt scan subtree (`make_scan_subtree` with the preserved Argument leaf, none here) has
no `Unwind`: the operator that bound `k` is pruned with the scan. -/
theorem rebuilt_drops_unwind :
    hasUnwind (makeScanSubtree b [0] false (argumentLeafOf exChild) []) = false := by decide

/-- The rebuilt chain `(b)-[10]->(c)` from the seed `[b]`. -/
def exSteps : List Step := [.hop ⟨1, 3, 10, 0, false⟩]

/-- **PROVEN** (the premise is false for the Rust code): `k` is neither the new node, an
`Argument` variable, nor a hop endpoint. -/
theorem premise_fails : ¬ SalvagePremise exChild (seedBound b.1 exChild) exSteps := by
  intro h
  have := h [b, k] (by decide) k (by decide)
  revert this
  decide

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

/-- **PROVEN** (the wrong result): generate-and-test (Cypher) accepts the row (the hop holds and `b.v = k`,
both 1), but the chain the pass builds reads `k` before anything binds it, so the filter fails and the row is
lost. -/
theorem salvage_loses_row :
    (exSteps ++ salvagedSteps exChild exPred).all (fullHolds exE exσ) = true ∧
    runScoped exE exσ (seedBound b.1 exChild) (exSteps ++ salvagedSteps exChild exPred) = false ∧
    ¬ wellScoped (seedBound b.1 exChild) (exSteps ++ salvagedSteps exChild exPred) := by
  refine ⟨by decide, by decide, ?_⟩
  intro hw
  simp [wellScoped, exSteps, salvagedSteps, filtersOf, exChild, seedBound, b, k, argVars,
    argumentLeafOf] at hw

end OptimizerScan.Salvage
