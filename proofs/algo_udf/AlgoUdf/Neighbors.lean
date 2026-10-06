/-! # `node.getNeighbors()` (`graph/src/udf/js_classes.rs`)

| here | there |
| --- | --- |
| `nodeRels`       | AXIOMATISED `Graph::get_node_relationships` contract: every outgoing edge of `v`, then every incoming edge of `v` (a self-loop is in both) |
| `collectEdges`   | `collect_edges` :97 |
| `typeFilters`    | `types` parsing in `js_get_neighbors` :386-397 (non-string elements skipped) |
| `applyTypes`     | the `type_filters` retain :452-459 |
| `neighborNodes`  | node result with `seen_neighbors` dedup :476-487 |
-/
namespace AlgoUdf.Neighbors

structure E where
  id : Nat
  src : Nat
  dst : Nat
  ty : String
  deriving DecidableEq, Repr

inductive Dir | outgoing | incoming | both deriving DecidableEq

def nodeRels (es : List E) (v : Nat) : List E :=
  es.filter (·.src == v) ++ es.filter (·.dst == v)

def collectEdges (es : List E) (v : Nat) : Dir → List E
  | .outgoing => (nodeRels es v).filter (·.src == v)
  | .incoming => (nodeRels es v).filter (·.dst == v)
  | .both => nodeRels es v

/-- What C returns: each incident edge once. -/
def spec (es : List E) (v : Nat) : Dir → List E
  | .outgoing => es.filter (·.src == v)
  | .incoming => es.filter (·.dst == v)
  | .both => es.filter (fun e => e.src == v || e.dst == v)

/-- Without self-loops at `v`, outgoing agrees with the spec. -/
theorem outgoing_ok (es : List E) (v : Nat) (h : ∀ e ∈ es, e.src = v → e.dst ≠ v) :
    collectEdges es v .outgoing = spec es v .outgoing := by
  simp only [collectEdges, spec, nodeRels, List.filter_append, List.filter_filter]
  have : List.filter (fun e => (e.src == v) && (e.dst == v)) es = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro e he; have := h e he; simp; intro hs; exact this hs
  have e1 : List.filter (fun e => (e.src == v) && (e.src == v)) es = List.filter (·.src == v) es := by
    simp
  rw [e1]
  have e2 : List.filter (fun e => (e.src == v) && (e.dst == v)) es =
      List.filter (fun e => (e.dst == v) && (e.src == v)) es := by
    congr 1; funext e; exact Bool.and_comm _ _
  rw [this, List.append_nil]

/-- BUG (confirmed, `bug_udf_get_neighbors_self_loop_twice`): e0 a→b, e2 a→a:
outgoing edges of a are [e0, e2, e2]. -/
theorem self_loop_twice :
    let es := [E.mk 0 0 1 "R", E.mk 2 0 0 "R"]
    (collectEdges es 0 .outgoing).map (·.id) = [0, 2, 2] ∧
    (spec es 0 .outgoing).map (·.id) = [0, 2] ∧
    (collectEdges es 0 .both).map (·.id) = [0, 2, 2] := by decide

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
