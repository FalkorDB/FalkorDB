import OptimizerScan.ScanOrder

/-
# The seed bound set of `select_scan_node` (#3046, fixes the W2-scan-1 seed bug)

| here | there |
| --- | --- |
| `XK`, `XT` | `IR`, `DynNode<IR>` as `exposed_variables` reads it |
| `XK.own` | `get_variables` of the operator alone (`DynTree::new(node.data().clone())`), `runtime.rs:260` |
| `exposed` | `exposed_variables` `select_scan_node.rs:479-520` |
| `initialBound` | `initial_bound` `select_scan_node.rs:1040-1043` (`existing_child` `:999-1020`) |

`initial_bound` is `{best_node}` when a scan is built for `best_node` (`existing_child = None`), and
`exposed_variables` of the kept outer-context child otherwise. Before #3046 it was
`{best_node} ∪ collect_subtree_variables(child)`; `pre3046InitialBound` keeps that, and
`pre3046_seed_loses_row` is the old counterexample (the `MATCH (b:A) WITH b LIMIT 1 …` query).
Ids only, as in the Rust (`HashSet<u32>`).
-/
namespace OptimizerScan.Seed
open OptimizerScan.ScanOrder

/-- The operator kinds `exposed_variables` distinguishes. `other own` is any other operator,
with `own` = what `get_variables` names for the operator alone. -/
inductive XK
  | project (exprs copies : List Nat)        -- `copies`: the new side of each `(old, new)` pair
  | aggregate (names projections : List Nat) -- `projections`: the new side of each pair
  | argument (payload : Option (List Nat))
  | forEach (var : Nat)
  | passThrough                             -- `SemiApply`, `AntiSemiApply`, `OrApplyMultiplexer`
  | other (own : List Nat)
  deriving DecidableEq, Repr

inductive XT where
  | node (k : XK) (cs : List XT)

/-- `get_variables` on the one-node tree (`Argument` names nothing, `ForEach` its loop variable). -/
def XK.own : XK → List Nat
  | .project exprs _ => exprs
  | .aggregate names _ => names
  | .argument _ => []
  | .forEach v => [v]
  | .passThrough => []
  | .other own => own

/-- The first `match` of `exposed_variables`: the carried-over copies and the `Argument` payload. -/
def XK.carried : XK → List Nat
  | .project _ copies => copies
  | .aggregate _ projs => projs
  | .argument (some vs) => vs
  | _ => []

mutual
/-- `exposed_variables` (`select_scan_node.rs:479`). -/
def exposed : XT → List Nat
  | .node k cs =>
    match k with
    | .project _ _ | .aggregate _ _ => k.own ++ k.carried
    | .forEach _ => if cs.length = 2 then exposedHead cs else []
    | .passThrough => k.own ++ k.carried ++ exposedHead cs
    | _ => k.own ++ k.carried ++ exposedAll cs
/-- Child 0 only (when there is one). -/
def exposedHead : List XT → List Nat
  | [] => []
  | c :: _ => exposed c
/-- Every child. -/
def exposedAll : List XT → List Nat
  | [] => []
  | c :: cs => exposed c ++ exposedAll cs
end

/-- `initial_bound` (`select_scan_node.rs:1040-1043`). -/
def initialBound (best : Nat) : Option XT → List Nat
  | none => [best]
  | some child => exposed child

/-- `collect_subtree_variables` before #3046: every variable `get_variables` names in the subtree
(its BFS stops at the first `Project`; a superset of that is enough for the counterexample). -/
def pre3046InitialBound (best : Nat) (childVars : Option (List Nat)) : List Nat :=
  best :: childVars.getD []

/-! ## `exposed_variables`, arm by arm -/

/-- **PROVEN**: `Project` / `Aggregate` hide everything below them; they expose what they name
plus the new side of their copies (`project_exposes_copied_variables`,
`aggregate_exposes_copied_variables`, `hidden_variables_are_not_exposed`). -/
theorem exposed_project (es cp : List Nat) (cs : List XT) :
    exposed (.node (.project es cp) cs) = es ++ cp := by simp [exposed, XK.own, XK.carried]

theorem exposed_aggregate (ns ps : List Nat) (cs : List XT) :
    exposed (.node (.aggregate ns ps) cs) = ns ++ ps := by simp [exposed, XK.own, XK.carried]

/-- **PROVEN**: an `Argument` exposes its payload, `Argument(None)` nothing
(`argument_exposes_its_payload`). -/
theorem exposed_argument (p : Option (List Nat)) :
    exposed (.node (.argument p) []) = p.getD [] := by
  cases p <;> simp [exposed, exposedAll, XK.own, XK.carried]

/-- **PROVEN**: `ForEach` exposes only its input (child 0 of two); its loop variable and body do
not leak, and the body-only form exposes nothing (`foreach_exposes_only_its_input`). -/
theorem exposed_forEach_input (v : Nat) (input body : XT) :
    exposed (.node (.forEach v) [input, body]) = exposed input := by
  simp [exposed, exposedHead]

theorem exposed_forEach_body (v : Nat) (body : XT) :
    exposed (.node (.forEach v) [body]) = [] := by simp [exposed]

/-- **PROVEN**: `SemiApply`/`AntiSemiApply`/`OrApplyMultiplexer` pass child 0 through; their
predicate branches bind nothing visible. -/
theorem exposed_passThrough (c : XT) (preds : List XT) :
    exposed (.node .passThrough (c :: preds)) = exposed c := by
  simp [exposed, exposedHead, XK.own, XK.carried]

/-- **PROVEN**: every other operator exposes its own variables and its children's. -/
theorem exposed_other (own : List Nat) (cs : List XT) :
    exposed (.node (.other own) cs) = own ++ exposedAll cs := by
  simp [exposed, XK.own, XK.carried]

/-! ## The seed -/

/-- **PROVEN**: when the outer child is kept, `best_node` is in the seed only if the child
exposes it — scoring best no longer binds it. -/
theorem initialBound_kept (best : Nat) (child : XT) :
    best ∈ initialBound best (some child) ↔ best ∈ exposed child := Iff.rfl

/-- **PROVEN**: with no kept child the seed is the new scan's node alone. -/
theorem initialBound_scan (best : Nat) : initialBound best none = [best] := rfl

/-- **PROVEN**: placement from the seed is well scoped whenever what the chain's leaf really binds
(`B`) contains the seed: the scan's node, or what the kept child exposes. -/
theorem seed_place_wellScoped (best : Nat) (child : Option XT) (B : List Nat)
    (hB : ∀ v ∈ initialBound best child, v ∈ B) (hs : List Hop)
    (fs : List (List Nat × ((Nat → Nat) → Bool))) :
    wellScoped B (place (initialBound best child) hs fs).1 :=
  place_wellScoped _ B hs fs hB

/-! ## The W2-scan-1 query, before and after #3046

`MATCH (b:A) WITH b LIMIT 1 MATCH (b)<-[]-(d)-[:R]->(c {v:1})-[:R]->(z) RETURN count(*)`:
the kept child is `Limit → Project(copies b) → NodeByLabelScan(b_old)`, `best_node` = `c` (2),
b = 0, `b_old` = 7 (a hidden id of the old scope). -/

def exLimitChild : XT :=
  .node (.other []) [.node (.project [] [0]) [.node (.other [7]) []]]

/-- **PROVEN**: the kept child exposes `b` only. -/
theorem exLimitChild_exposed : exposed exLimitChild = [0] := by decide

/-- **PROVEN** (#3046): the seed is `[b]`, the filter on `c` lands after `d→c`, the row is found,
agreeing with generate-and-test. -/
theorem seed_keeps_row :
    runScoped exE exσ [0] (place (initialBound 2 (some exLimitChild)) exHops [exFilter]).1 = true ∧
    (place (initialBound 2 (some exLimitChild)) exHops [exFilter]).1.all (fullHolds exE exσ) = true := by
  decide

/-- Before #3046 (historical): the seed `{c} ∪ {b, …}` placed the filter on `c` after the first
hop, before `c` is bound, and lost the row (Rust 0, C 1). -/
theorem pre3046_seed_loses_row :
    runScoped exE exσ [0] (place (pre3046InitialBound 2 (some [0])) exHops [exFilter]).1 = false := by
  decide

/-- `WITH count(*) AS n MATCH …`: the scans under an `Aggregate` bind nothing the later `MATCH`
sees, even where their ids (here 3 = `z`) collide with the new scope's. -/
theorem aggregate_hides_scan_ids :
    exposed (.node (.aggregate [5] []) [.node (.other [3]) [.node (.other [4]) []]]) = [5] ∧
    3 ∉ initialBound 1 (some (.node (.aggregate [5] []) [.node (.other [3]) []])) := by
  decide

end OptimizerScan.Seed
