import OptimizerScan.ScanTree
import OptimizerScan.ScanOrder

/-
# The seed of the hop ordering for a kept child (`select_scan_node`, #3099)

| here | there (`select_scan_node.rs`, PR #3099 head) |
| --- | --- |
| `keptOld` | pre-#3099: a planner scan subtree child (`is_planner_scan_subtree`) or a transparent `Argument` was replaced by a new scan of `best_node`; any other child was kept |
| `keptNew` | `existing_child` (`:970-982`): every non-leaf child is kept unless it is a transparent `Argument` |
| `subtreeVars` | `collect_subtree_variables(child)` (approximated by `collect_output_aliases` over the subtree) |
| `seed` | `initial_bound` (`:997-1004`): `best_node` plus the kept child's variables |
| `boundBelow` | what actually binds below the first hop: the kept child (`subtree = existing_child…`, `:1167`), or the new scan of `best_node` |

A filter is placed as soon as its variables are in `placed_bound = initial_bound` plus the hops so
far (`:1171-1203`, `ScanOrder.place`). So the seed must be *honest*: everything in it bound below.

Before #3099 a plain earlier `MATCH (b)` child was a planner scan subtree and was replaced, so the
seed `[best]` was exactly what the new scan binds (`pre3099_plannerScan_seed_honest`). #3099 keeps
it, but still seeds `best_node`, which that child need not bind (`plannerScan_seed_dishonest`). The
`ScanOrder` bug-1 run then loses its row (`plannerScan_loses_row`):
`CREATE (:A)<-[:S]-(:D)-[:R]->(:L {v:1})-[:R]->(:Z)` then
`MATCH (b) MATCH (b)<-[:S]-(d)-[:R]->(c:L {v:1})-[:R]->(z) RETURN count(*)` — before #3099 1, now 0
(`c` wins the scoring: bound `b` and filtered, labelled `c` both score 3; `c`'s label cardinality is
lower; the seed is `{c, b}`, the first hop is `d→b`, and `c.v = 1` is placed right after it).
-/
namespace OptimizerScan.SelectSeed
open OptimizerScan.ScanTree OptimizerScan.ScanOrder

def subtreeVars (t : PT) : List Nat := t.walk.flatMap collectOutputAliases

/-- Pre-#3099: is the non-leaf child kept? -/
def keptOld (c : PT) (argTransparent : Bool) : Bool := !(isPlannerScan c || argTransparent)
/-- #3099: the child is kept unless it is a transparent `Argument`. -/
def keptNew (_c : PT) (argTransparent : Bool) : Bool := !argTransparent

/-- `initial_bound` for a non-leaf chain. -/
def seed (kept : Bool) (best : Nat) (c : PT) : List Nat :=
  if kept then best :: subtreeVars c else [best]
/-- What binds below the first hop: the kept child, or the new scan of `best_node`. -/
def boundBelow (kept : Bool) (best : Nat) (c : PT) : List Nat :=
  if kept then subtreeVars c else [best]

def Honest (seed bound : List Nat) : Prop := ∀ v ∈ seed, v ∈ bound

instance (s b : List Nat) : Decidable (Honest s b) := by unfold Honest; infer_instance

/-- **PROVEN** (the old property): before #3099 a planner-scan child is replaced, so the seed is
    honest. -/
theorem pre3099_plannerScan_seed_honest (c : PT) (best : Nat) (argT : Bool)
    (h : isPlannerScan c = true) :
    Honest (seed (keptOld c argT) best c) (boundBelow (keptOld c argT) best c) := by
  simp [Honest, seed, boundBelow, keptOld, h]

/-- **PROVEN**: the seed is honest whenever the child is not kept, or the kept child binds
    `best_node`. -/
theorem seed_honest_of_bound (kept : Bool) (best : Nat) (c : PT)
    (h : kept = true → best ∈ subtreeVars c) : Honest (seed kept best c) (boundBelow kept best c) := by
  cases kept
  · simp [Honest, seed, boundBelow]
  · intro v hv
    simp only [seed, boundBelow, if_true, List.mem_cons] at hv ⊢
    rcases hv with rfl | hv
    · exact h rfl
    · exact hv

/-- `MATCH (b)`: the earlier clause's plain `AllNodeScan(b)` (`b` = 0). -/
def scanB : PT := .node (.allNodeScan (0, 0)) []

theorem scanB_plannerScan : isPlannerScan scanB = true ∧ keptOld scanB false = false ∧
    keptNew scanB false = true := by decide

/-- **COUNTEREXAMPLE** (#3099): the old property fails — the kept `MATCH (b)` child does not bind
    `best_node = c` (2), yet it is seeded. -/
theorem plannerScan_seed_dishonest :
    ¬ Honest (seed (keptNew scanB false) 2 scanB) (boundBelow (keptNew scanB false) 2 scanB) := by
  decide

/-- **COUNTEREXAMPLE** (#3099): with that seed `c.v = 1` is placed after the first hop `d→b`, before
    `c` is bound; the scoped run loses the row generate-and-test keeps. Before #3099 the seed was the
    honest `[c]` of a fresh scan of `c`. -/
theorem plannerScan_loses_row :
    seed (keptNew scanB false) 2 scanB = [2, 0] ∧
    boundBelow (keptNew scanB false) 2 scanB = [0] ∧
    (place (seed (keptNew scanB false) 2 scanB) exHops [exFilter]).1.all (fullHolds exE exσ) = true ∧
    runScoped exE exσ (boundBelow (keptNew scanB false) 2 scanB)
      (place (seed (keptNew scanB false) 2 scanB) exHops [exFilter]).1 = false := by
  decide

end OptimizerScan.SelectSeed
