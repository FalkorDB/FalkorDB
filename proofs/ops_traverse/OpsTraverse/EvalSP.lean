/-
# `ExprEval::eval_shortest_path` (`graph/src/runtime/eval.rs`)

The glue in front of the bidirectional BFS: evaluate the two endpoint expressions, reject or
short-circuit on their values, take the graph, answer the zero-hop case, then hand the search its
level cap. The BFS itself (`bfs_shortest_path`, :1459) is a parameter `bfs`; nothing here assumes
anything about it.

| here | there |
| --- | --- |
| `evalSP` | `eval_shortest_path` (eval.rs) |
| `maxLevel` | `max_hops.map_or(u64::MAX, |m| m as u64)` (:1409) |
-/
namespace OpsTraverse.EvalSP

/-- The runtime values `eval_shortest_path` looks at. -/
inductive SV where
  | node (id : Nat)
  | null
  | other
  | path (nodes : List Nat)

/-- `u64::MAX`. -/
def u64max : Nat := 18446744073709551615

/-- `max_hops.map_or(u64::MAX, |m| m as u64)`. -/
def maxLevel : Option Nat → Nat
  | none => u64max
  | some m => m

/-- `eval_shortest_path`: `src`/`dst` are the results of evaluating the two child expressions,
`rt` the result of `self.rt()?`, `bfs src dst max_level min_hops directed` the search. -/
def evalSP (src dst : Except String SV) (rt : Except String Unit) (minHops : Nat)
    (maxHops : Option Nat) (directed : Bool) (bfs : Nat → Nat → Nat → Nat → Bool → SV) :
    Except String SV := do
  let sv ← src
  let dv ← dst
  let sid ← match sv with
    | .node id => pure (some id)
    | .null => pure none
    | _ => throw "A shortestPath requires bound nodes"
  let did ← match dv with
    | .node id => pure (some id)
    | .null => pure none
    | _ => throw "A shortestPath requires bound nodes"
  match sid, did with
  | some s, some d =>
    rt
    if minHops == 0 && s == d then pure (.path [s])
    else pure (bfs s d (maxLevel maxHops) minHops directed)
  | _, _ => pure .null

/-- Two nodes: take the graph, answer the zero-hop case, else search. -/
def evalSPTail (rt : Except String Unit) (s d minHops : Nat) (maxHops : Option Nat) (directed : Bool)
    (bfs : Nat → Nat → Nat → Nat → Bool → SV) : Except String SV := do
  rt
  if minHops == 0 && s == d then pure (.path [s])
  else pure (bfs s d (maxLevel maxHops) minHops directed)

/-- **PROVEN** (`eval_shortest_path`, everything but the search): both children are evaluated first
(an error in either propagates, the source's first); then a non-node, non-null endpoint is the error
"A shortestPath requires bound nodes" (source checked first); a `Null` endpoint gives `Null` without
touching the graph; with two nodes the graph is taken (`rt` error propagates), `min_hops == 0` on
equal nodes is the one-node path, and every other case is the search with `max_level` from
`max_hops`. -/
theorem evalSP_spec (rt : Except String Unit) (minHops : Nat) (maxHops : Option Nat) (directed : Bool)
    (bfs : Nat → Nat → Nat → Nat → Bool → SV) :
    (∀ dst e, evalSP (.error e) dst rt minHops maxHops directed bfs = .error e) ∧
    (∀ sv e, evalSP (.ok sv) (.error e) rt minHops maxHops directed bfs = .error e) ∧
    (∀ dv, evalSP (.ok .other) (.ok dv) rt minHops maxHops directed bfs
        = .error "A shortestPath requires bound nodes") ∧
    (∀ ps dv, evalSP (.ok (.path ps)) (.ok dv) rt minHops maxHops directed bfs
        = .error "A shortestPath requires bound nodes") ∧
    (∀ s dv, evalSP (.ok (.node s)) (.ok dv) rt minHops maxHops directed bfs
        = match dv with
          | .node d => evalSPTail rt s d minHops maxHops directed bfs
          | .null => .ok .null
          | _ => .error "A shortestPath requires bound nodes") ∧
    (∀ dv, evalSP (.ok .null) (.ok dv) rt minHops maxHops directed bfs
        = match dv with
          | .node _ | .null => .ok .null
          | _ => .error "A shortestPath requires bound nodes") := by
  refine ⟨fun _ _ => rfl, fun _ _ => rfl, fun dv => ?_, fun ps dv => ?_, fun s dv => ?_, fun dv => ?_⟩
  · cases dv <;> rfl
  · cases dv <;> rfl
  · cases dv <;> rfl
  · cases dv <;> rfl

/-- The zero-hop case: `shortestPath((a)-[*0..]->(a))` is the single node. -/
theorem evalSP_zero_hop (s : Nat) (maxHops : Option Nat) (directed : Bool)
    (bfs : Nat → Nat → Nat → Nat → Bool → SV) :
    evalSP (.ok (.node s)) (.ok (.node s)) (.ok ()) 0 maxHops directed bfs = .ok (.path [s]) := by
  simp [evalSP, bind, Except.bind, pure, Except.pure]

/-- Otherwise (two distinct nodes, or `min_hops > 0`) the answer is the search's, given
`maxLevel maxHops` (`u64::MAX` for an unbounded walk). -/
theorem evalSP_search (s d minHops : Nat) (maxHops : Option Nat) (directed : Bool)
    (bfs : Nat → Nat → Nat → Nat → Bool → SV) (h : ¬ (minHops = 0 ∧ s = d)) :
    evalSP (.ok (.node s)) (.ok (.node d)) (.ok ()) minHops maxHops directed bfs =
      .ok (bfs s d (maxLevel maxHops) minHops directed) := by
  have : (minHops == 0 && s == d) = false := by
    by_cases h1 : minHops = 0 <;> by_cases h2 : s = d <;> simp_all
  simp [evalSP, bind, Except.bind, pure, Except.pure, this]

theorem maxLevel_unbounded : maxLevel none = u64max := rfl

end OpsTraverse.EvalSP
