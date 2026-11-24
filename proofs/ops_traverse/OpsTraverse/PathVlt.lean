import OpsTraverse.ShortestPaths
/-
# PathBuilder, `Value::Path` branch: inlining a variable-length segment

`ShortestPaths.lean` models PathBuilder's `Value::List` branch (allShortestPaths).
This file models the other branch: a variable-length relationship variable is
bound by `CondVarLenTraverseOp` to a `Value::Path` `[Node, Rel, Node, …, Node]`
oriented from the relationship's `from` to its `to` (`path_value`), and
PathBuilder splices it into the named path.

| here | there |
| --- | --- |
| `PV` | the `Value::Node` / `Value::Relationship` elements of a `Value::Path` |
| `enc` | the VLT binding `Value::Path([Node(s), Rel(e₁), Node(n₁), …])` (`runtime/ops/cond_var_len_traverse.rs`, `path_value`) |
| `vltNext` | `PathBuilderOp::next`'s `Value::Path` arm (`runtime/ops/path_builder.rs`): `path.is_incoming(var_idx)` picks `rev().skip(1)` or `skip(1)` |

For an incoming pattern `(m)<-[*]-(n)` the parser swaps the endpoints, so the
VLT walks `n → … → m` and binds `[Node n, …, Node m]`; PathBuilder must emit it
reversed. Since #3460 the direction the pattern was written in is recorded at
plan time (`QueryPath::incoming`, read by `QueryPath::is_incoming`), so the
cyclic case `MATCH p = (a {id: 0})<-[:R*3..3]-(a)` on `0→1→2→0` (w 1, 2, 3)
gives `[0,2,1,0]`/`[3,2,1]` (#3451, `fixed3451_cycle_reversed`). Before, the
direction was inferred by comparing the VLT's first node with the previous path
node, which says "forward" on a cycle. Property `PathFollowsPattern`.
-/

namespace OpsTraverse.PathVlt

open OpsTraverse.ASP

inductive PV where
  | node (n : Nat)
  | rel (e : Nat)
deriving DecidableEq, Repr

/-- The VLT binding for a walk from `s` taking `(edge, next node)` steps. -/
def enc (s : Nat) (steps : List (Nat × Nat)) : List PV :=
  .node s :: steps.flatMap fun p => [.rel p.1, .node p.2]

/-- The walk's last node. -/
def endOf (s : Nat) : List (Nat × Nat) → Nat
  | [] => s
  | p :: r => endOf p.2 r

/-- Each step's edge connects the current node and the next one, in either
stored orientation (a VLT step follows `step`, so an incoming pattern's edges
are stored opposite to the path direction). -/
def Linked (g : Graph) : Nat → List (Nat × Nat) → Prop
  | _, [] => True
  | s, (e, n) :: r =>
    (∃ x ∈ g, x.id = e ∧ ((x.src = s ∧ x.dst = n) ∨ (x.src = n ∧ x.dst = s))) ∧ Linked g n r

/-- `elems.iter().rev().find_map(Node)`: the last node already in the path. -/
def lastNode (elems : List PV) : Option Nat :=
  elems.reverse.findSome? fun | .node n => some n | _ => none

/-- `PathBuilderOp::next`, `Value::Path` arm (`path_builder.rs`): `incoming` is
`path.is_incoming(var_idx)`, whether the hop was written `<-`. -/
def vltNext (incoming : Bool) (elems pe : List PV) : List PV :=
  if incoming then elems ++ pe.reverse.tail else elems ++ pe.tail

/-- **Property**: splicing a VLT walk `s → … → t` follows the pattern: written
forward (`prev = s`) it is appended as is; written `<-` (`prev = t`) it is
appended reversed — for every walk, cycles included. `build` gets the pattern
direction; an implementation may ignore it. -/
def PathFollowsPattern (build : Graph → Bool → List PV → List PV → List PV) : Prop :=
  ∀ (g : Graph) (s : Nat) (steps : List (Nat × Nat)) (elems : List PV) (rev : Bool),
    g.WF → Linked g s steps → steps ≠ [] →
    lastNode elems = some (if rev then endOf s steps else s) →
    build g rev elems (enc s steps) =
      elems ++ (if rev then (enc s steps).reverse.tail else (enc s steps).tail)

/-- **PathBuilder satisfies the property** (#3451 fixed by #3460): the plan-time
direction flag picks the order, so a cycle is reversed too. -/
theorem vltNext_follows : PathFollowsPattern (fun _ rev elems pe => vltNext rev elems pe) := by
  intro g s steps elems rev _ _ _ _
  cases rev <;> rfl

/-- A walk of one or more hops: the dropped first element is the node already
in `elems`, the splice adds one relationship and one node per hop. -/
theorem vltNext_length (incoming : Bool) (elems : List PV) (s : Nat) (steps : List (Nat × Nat)) :
    (vltNext incoming elems (enc s steps)).length = elems.length + 2 * steps.length := by
  have h : (enc s steps).length = 2 * steps.length + 1 := by
    simp only [enc, List.length_cons, List.length_flatMap]
    induction steps with
    | nil => rfl
    | cons p r ih => simp only [List.map_cons, List.sum_cons, List.length_cons, List.length_nil] at ih ⊢; omega
  cases incoming <;> simp [vltNext, List.length_tail, List.length_reverse, h]

/-! ## The cyclic incoming case (#3451) -/

/-- The issue's triangle `0→1→2→0`, edge ids = the `w` values 1, 2, 3. -/
def gTri : Graph := [⟨1, 0, 1⟩, ⟨2, 1, 2⟩, ⟨3, 2, 0⟩]

/-- `(a)<-[:R*3..3]-(a)`: the VLT walks `0 →1→ 1 →2→ 2 →3→ 0`. -/
def triWalk : List (Nat × Nat) := [(1, 1), (2, 2), (3, 0)]

/-- **Fixed by #3460**: the incoming cycle comes out reversed, nodes
`[0,2,1,0]`, rels `[3,2,1]`. -/
theorem fixed3451_cycle_reversed :
    vltNext true [.node 0] (enc 0 triWalk) =
      [.node 0, .rel 3, .node 2, .rel 2, .node 1, .rel 1, .node 0] := by decide

/-- The non-cyclic incoming case `(a {id:0})<-[:R*2..2]-(b)`: the VLT walks
`1 → 2 → 0` and PathBuilder emits `[0,2,1]`/`[3,2]`. -/
theorem fixed3451_noncyclic :
    vltNext true [.node 0] (enc 1 [(2, 2), (3, 0)]) =
      [.node 0, .rel 3, .node 2, .rel 2, .node 1] := by decide

/-- The outgoing cycle `(a)-[:R*3..3]->(a)` keeps traversal order. -/
theorem fixed3451_outgoing_cycle :
    vltNext false [.node 0] (enc 0 triWalk) =
      [.node 0, .rel 1, .node 1, .rel 2, .node 2, .rel 3, .node 0] := by decide

end OpsTraverse.PathVlt
