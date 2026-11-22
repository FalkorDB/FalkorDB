/-! # `node.getNeighbors()` (`graph/src/udf/js_classes.rs`)
-/
namespace AlgoUdf.Neighbors

structure E where
  id : Nat
  src : Nat
  dst : Nat
  ty : String
  deriving DecidableEq, Repr

inductive Dir | outgoing | incoming | both deriving DecidableEq

/-- `get_node_relationships_by_type` with no type filter (graph.rs): the outgoing half if
enumerated, then the incoming half, which drops the self-loop (`src != id`) when the outgoing half
is also enumerated. -/
def nodeRels (es : List E) (v : Nat) : Dir → List E
  | .outgoing => es.filter (·.src == v)
  | .incoming => es.filter (·.dst == v)
  | .both => es.filter (·.src == v) ++ (es.filter (·.dst == v)).filter (·.src != v)

/-- `collect_edges` (js_classes.rs): the direction is handed to the graph. -/
def collectEdges (es : List E) (v : Nat) (d : Dir) : List E := nodeRels es v d

/-- What C returns: each incident edge once. -/
def spec (es : List E) (v : Nat) : Dir → List E
  | .outgoing => es.filter (·.src == v)
  | .incoming => es.filter (·.dst == v)
  | .both => es.filter (fun e => e.src == v || e.dst == v)

/-- **Fixed by #3084.** `collect_edges` returns exactly the incident edges of the spec, in every
direction, self-loops included. -/
theorem collectEdges_mem (es : List E) (v : Nat) (d : Dir) (e : E) :
    e ∈ collectEdges es v d ↔ e ∈ spec es v d := by
  cases d <;> simp only [collectEdges, nodeRels, spec, List.mem_append, List.mem_filter, beq_iff_eq,
    bne_iff_ne, ne_eq, Bool.or_eq_true]
  by_cases h : e.src = v <;> simp [h]

/-- ... and each of them once (a self-loop is no longer in both halves). -/
theorem collectEdges_nodup (es : List E) (v : Nat) (d : Dir) (h : es.Nodup) :
    (collectEdges es v d).Nodup := by
  cases d with
  | outgoing => exact h.filter _
  | incoming => exact h.filter _
  | both =>
    simp only [collectEdges, nodeRels]
    rw [List.nodup_append]
    refine ⟨h.filter _, (h.filter _).filter _, ?_⟩
    intro a ha b hb hab
    subst hab
    simp only [List.mem_filter, beq_iff_eq, bne_iff_ne, ne_eq] at ha hb
    exact hb.2 ha.2

/-- A self-loop is listed once (#3084). -/
theorem fixed3084_self_loop_once :
    let es := [E.mk 0 0 1 "R", E.mk 2 0 0 "R"]
    (collectEdges es 0 .outgoing).map (·.id) = [0, 2] ∧
    (spec es 0 .outgoing).map (·.id) = [0, 2] ∧
    (collectEdges es 0 .both).map (·.id) = [0, 2] ∧
    (collectEdges es 0 .incoming).map (·.id) = [2] := by decide

/-- JS `types` array elements: `some s` for strings, `none` for anything else. -/
def typeFilters (xs : List (Option String)) : List String := xs.filterMap id

def applyTypes (fs : List String) (es : List E) : List E :=
  if fs.isEmpty then es else es.filter (fun e => fs.contains e.ty)

/-- SUSPICION (C crashes on the same input, so no reference): `types: [123]`
drops the non-string silently and then applies no type filter at all, returning
every edge, while `types: ['Nope']` returns none. -/
theorem non_string_types_disable_filter (es : List E) :
    applyTypes (typeFilters [none]) es = es := rfl

theorem string_types_filter (es : List E) (fs : List String) (h : fs ≠ []) :
    ∀ e ∈ applyTypes fs es, e.ty ∈ fs := by
  intro e he
  unfold applyTypes at he
  have : fs.isEmpty = false := by cases fs <;> simp_all
  simp [this] at he
  exact he.2

/-- Node results are deduplicated (`seen_neighbors`), so the self-loop bug only
shows with `returnType:'edges'`. -/
def neighborNodes (v : Nat) : List E → List Nat → List Nat
  | [], seen => seen.reverse
  | e :: es, seen =>
    let n := if e.src = v then e.dst else e.src
    if seen.contains n then neighborNodes v es seen else neighborNodes v es (n :: seen)

theorem neighborNodes_nodup_example :
    neighborNodes 0 [E.mk 0 0 1 "R", E.mk 2 0 0 "R", E.mk 2 0 0 "R"] [] = [1, 0] := by decide

end AlgoUdf.Neighbors
