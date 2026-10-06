import AlgoUdf.Paths
/-! # Shared graph view for the algo.SPpaths / algo.SSpaths searches

`graph/src/runtime/functions/algo_procedures.rs` (origin/main 3fec7d7c9):

| here | there |
| --- | --- |
| `G.rels u`  | `Graph::get_node_relationships_by_type(u, relTypes, relDirection)` as `(src, dst, eid)` (AXIOMATISED: any finite list) |
| `G.wt e`    | `edge_numeric_attr(g, e, weightProp, 1.0)` :2104, `none` when the sum is non-finite (the edge is skipped, :2264-2269, :2531) |
| `G.cost e`  | `edge_numeric_attr(g, e, costProp, 0.0)` :2104 |
| `Edge`      | one relaxation candidate: `far_endpoint` :2121 accepts it and the weight is finite |
| `Walk`      | an edge sequence in traversal order, edges kept in their stored direction (`FoundPath.0`) |
| `succs`     | `node_successors` :2153 |

Weights are exact non-negative naturals: f64 rounding, NaN/inf (already
filtered by the Rust `is_finite` checks) and negative weights are out of the
model. Negative weights are a documented precondition shared with C
(`Dijkstra_ShortestPath` and `algo.SPpaths` both assume them non-negative,
see `PathDispatch.neg_weight_bound_loses_path`).
-/
namespace AlgoUdf.Graph
open AlgoUdf.Paths (Dir farEndpoint)

abbrev Rel := Nat × Nat × Nat   -- (src, dst, eid)

/-- Point update of a hash map / hash set modelled as a function. -/
def upd {α : Type} (f : Nat → α) (k : Nat) (a : α) : Nat → α := fun x => if x = k then a else f x

@[simp] theorem upd_same {α : Type} (f : Nat → α) (k : Nat) (a : α) : upd f k a k = a := by simp [upd]
@[simp] theorem upd_ne {α : Type} (f : Nat → α) (k x : Nat) (a : α) (h : x ≠ k) : upd f k a x = f x := by
  simp [upd, h]

structure G where
  rels : Nat → List Rel
  dir : Dir
  wt : Nat → Option Nat
  cost : Nat → Nat

/-- One traversable step from `u` to `v` through stored edge `r` costing `c`. -/
def Edge (g : G) (u v : Nat) (r : Rel) (c : Nat) : Prop :=
  r ∈ g.rels u ∧ farEndpoint u r.1 r.2.1 g.dir = some v ∧ g.wt r.2.2 = some c

/-- A walk from `u` to `w` along `es` (traversal order) of total weight `W`. -/
inductive Walk (g : G) : Nat → List Rel → Nat → Nat → Prop
  | nil (u : Nat) : Walk g u [] u 0
  | cons {u v w : Nat} {r : Rel} {c : Nat} {rest : List Rel} {W : Nat} :
      Edge g u v r c → Walk g v rest w W → Walk g u (r :: rest) w (c + W)

/-- Unweighted reachability step (BFS ignores weights). -/
def Adj (g : G) (u v : Nat) (r : Rel) : Prop :=
  r ∈ g.rels u ∧ farEndpoint u r.1 r.2.1 g.dir = some v

/-- A hop-walk: nodes `u = x₀, …, xₖ = w` joined by `Adj` steps. -/
inductive HWalk (g : G) : Nat → List Rel → Nat → Prop
  | nil (u : Nat) : HWalk g u [] u
  | cons {u v w : Nat} {r : Rel} {rest : List Rel} :
      Adj g u v r → HWalk g v rest w → HWalk g u (r :: rest) w

/-- `node_successors` :2153: `filter_map(far_endpoint)` over the typed relationships. -/
def succs (g : G) (u : Nat) : List (Rel × Nat) :=
  (g.rels u).filterMap (fun r => (farEndpoint u r.1 r.2.1 g.dir).map (fun far => (r, far)))

theorem mem_succs (g : G) (u : Nat) (r : Rel) (v : Nat) :
    (r, v) ∈ succs g u ↔ Adj g u v r := by
  unfold succs Adj
  simp only [List.mem_filterMap, Option.map_eq_some_iff, Prod.mk.injEq]
  constructor
  · rintro ⟨r', hr', far, hf, rfl, rfl⟩; exact ⟨hr', hf⟩
  · rintro ⟨hr, hf⟩; exact ⟨r, hr, v, hf, rfl, rfl⟩

/-- `node_successors` keeps the stored relationship order of `rels`. -/
theorem succs_map_fst_sublist (g : G) (u : Nat) :
    ((succs g u).map (·.1)).Sublist (g.rels u) := by
  unfold succs
  induction g.rels u with
  | nil => simp
  | cons r t ih =>
    simp only [List.filterMap_cons]
    cases h : farEndpoint u r.1 r.2.1 g.dir with
    | none => simp only [Option.map_none]; exact ih.cons r
    | some far => simp only [Option.map_some, List.map_cons]; exact ih.cons_cons r

theorem Walk.hwalk {g : G} {u es w W} (h : Walk g u es w W) : HWalk g u es w := by
  induction h with
  | nil u => exact .nil u
  | cons he _ ih => exact .cons ⟨he.1, he.2.1⟩ ih

theorem Walk.append {g : G} {u es v W es' w W'} (h : Walk g u es v W) (h' : Walk g v es' w W') :
    Walk g u (es ++ es') w (W + W') := by
  induction h with
  | nil u => simpa using h'
  | cons he _ ih => rw [Nat.add_assoc]; exact .cons he (ih h')

theorem Walk.snoc {g : G} {u es v W r w c} (h : Walk g u es v W) (he : Edge g v w r c) :
    Walk g u (es ++ [r]) w (W + c) := by
  have := h.append (Walk.cons he (Walk.nil w)); simpa using this

/-- A hop-walk recording the nodes entered after the start (`nodes` minus
`source` in `enumerate_paths`). -/
inductive NWalk (g : G) : Nat → List Rel → List Nat → Nat → Prop
  | nil (u : Nat) : NWalk g u [] [] u
  | cons {u v w : Nat} {r : Rel} {rest : List Rel} {ns : List Nat} :
      Adj g u v r → NWalk g v rest ns w → NWalk g u (r :: rest) (v :: ns) w

theorem NWalk.snoc {g : G} {u es ns v r w} (h : NWalk g u es ns v) (ha : Adj g v w r) :
    NWalk g u (es ++ [r]) (ns ++ [w]) w := by
  induction h with
  | nil u => exact .cons ha (.nil w)
  | cons he _ ih => exact .cons he (ih ha)

theorem NWalk.hwalk {g : G} {u es ns w} (h : NWalk g u es ns w) : HWalk g u es w := by
  induction h with
  | nil u => exact .nil u
  | cons he _ ih => exact .cons he ih

theorem NWalk.length {g : G} {u es ns w} (h : NWalk g u es ns w) : ns.length = es.length := by
  induction h with
  | nil u => rfl
  | cons _ _ ih => simp [ih]

theorem HWalk.nil_eq {g : G} {u w} (h : HWalk g u [] w) : u = w := by cases h; rfl

theorem HWalk.nwalk {g : G} {u es w} (h : HWalk g u es w) : ∃ ns, NWalk g u es ns w := by
  induction h with
  | nil u => exact ⟨[], .nil u⟩
  | @cons u v w r rest ha _ ih => obtain ⟨ns, h⟩ := ih; exact ⟨v :: ns, .cons ha h⟩

end AlgoUdf.Graph
