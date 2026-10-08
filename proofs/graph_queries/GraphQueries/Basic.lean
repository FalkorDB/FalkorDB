/-
# Abstract property-graph state for `graph/src/graph/graph.rs`

GraphBLAS objects are represented by their mathematical content, the
abstraction `proofs/versioned_matrix` justifies (`VM.run_repr`: a
`VersionedMatrix<bool>`'s logical contents behave as a plain set of
coordinates; `VMTensor.edges`: a `Tensor` pair holds a set of edge ids):

| here | there |
| --- | --- |
| `Mat` (+ `nr`,`nc` dims) | `VersionedMatrix<bool>` — `(m ∖ dm) ∪ dp` as a coordinate list |
| `Ten` | `Tensor` — `(src, dst, edge_id)` triples |
| `Mat.get` | `VersionedMatrix::get` (versioned_matrix.rs:1009); `None` out of bounds = GrB_INVALID_INDEX (matrix.rs:1345) |
| `Mat.row`, `Ten.out`, `Ten.inc` | `iter(i, i)`, `Tensor::iter(i, i, false/true)` (original orientation) |
| `G` | `struct Graph` (graph.rs:360) |

Machine integers: ids are `Nat`; every u64 count in `G` stays far below 2^64
(ids are dense, see `Ids.lean`), so wrap never happens; the one narrowing cast
(`usize as u16` on attribute ids) is modelled explicitly in `Attrs.lean`.
-/
namespace GQ

/-- A boolean sparse matrix: dimensions plus its stored coordinates. -/
structure Mat where
  nr : Nat
  nc : Nat
  ents : List (Nat × Nat)
deriving Repr, DecidableEq, Inhabited

namespace Mat
def empty (r c : Nat) : Mat := ⟨r, c, []⟩
/-- `get` (versioned_matrix.rs:1009): `GrB_Matrix_extractElement` fails
(→ `None`) outside the dimensions. -/
def get (m : Mat) (i j : Nat) : Bool :=
  decide (i < m.nr) && decide (j < m.nc) && decide ((i, j) ∈ m.ents)
def nvals (m : Mat) : Nat := m.ents.length
def row (m : Mat) (i : Nat) : List (Nat × Nat) := m.ents.filter (·.1 == i)
def rowsFrom (m : Mat) (lo : Nat) : List (Nat × Nat) := m.ents.filter (lo ≤ ·.1)
/-- `set` when in bounds (GraphBLAS rejects out-of-range indices). -/
def set (m : Mat) (i j : Nat) : Mat :=
  if i < m.nr ∧ j < m.nc ∧ (i, j) ∉ m.ents then { m with ents := m.ents ++ [(i, j)] } else m
def remove (m : Mat) (i j : Nat) : Mat := { m with ents := m.ents.filter (· != (i, j)) }
/-- `remove_mask` (versioned_matrix.rs:989): `eWiseMult` / masked transpose
require equal dimensions; on `GrB_DIMENSION_MISMATCH` the release build
(`debug_assert_eq!` only, matrix.rs:927) leaves the matrix unchanged. -/
def removeMask (m : Mat) (mr mc : Nat) (mask : List (Nat × Nat)) : Mat :=
  if mr = m.nr ∧ mc = m.nc then { m with ents := m.ents.filter (fun p => p ∉ mask) } else m
def resize (m : Mat) (r c : Nat) : Mat :=
  ⟨r, c, m.ents.filter (fun p => p.1 < r ∧ p.2 < c)⟩
end Mat

/-- A relationship tensor's logical content. -/
abbrev Ten := List (Nat × Nat × Nat)

namespace Ten
def out (t : Ten) (n : Nat) : Ten := t.filter (·.1 == n)
def inc (t : Ten) (n : Nat) : Ten := t.filter (·.2.1 == n)
def pair (t : Ten) (s d : Nat) : List Nat := (t.filter (fun e => e.1 == s && e.2.1 == d)).map (·.2.2)
def edgeCount (t : Ten) : Nat := t.length
def rowDegree (t : Ten) (n : Nat) : Nat := (out t n).length
def colDegree (t : Ten) (n : Nat) : Nat := (inc t n).length
def pattern (t : Ten) : List (Nat × Nat) := t.map (fun e => (e.1, e.2.1))
end Ten

/-- Constraint (constraint.rs:48). -/
inductive CType | unique | mandatory deriving DecidableEq, Repr
inductive CStatus | underConstruction | operational | failed deriving DecidableEq, Repr
inductive EType | node | rel deriving DecidableEq, Repr

structure Constraint where
  id : Nat
  ct : CType
  et : EType
  label : String
  props : List String
  status : CStatus
deriving DecidableEq, Repr

/-- `struct Graph` (graph.rs:360). Attribute values are abstract (`V`), the
attribute store is the map `entity → [(attr id, value)]`. Indexers are not
part of the state; their behaviour is a parameter of each index theorem.

Since #2846 the graph keeps no counters of its own: `node_ids` /
`relationship_ids : IdSpace` (graph.rs:371/:358). Their four fields are
spelled out here — `nodeCount`/`delNodes` are `node_ids.live`/`.recycled`,
`nodeEB`/`nodeTaken` its `entry_bound`/`taken` (id_space.rs:164) — and
reassembled by `nodeIds`/`relIds` (Ids.lean). -/
structure G (V : Type) where
  name : String
  nodeCap : Nat
  relCap : Nat
  nodeCount : Nat
  relCount : Nat
  delNodes : List Nat
  delRels : List Nat
  nodeEB : Nat := 0
  nodeTaken : List Nat := []
  relEB : Nat := 0
  relTaken : List Nat := []
  zero : Mat
  adj : Mat
  nodeLabels : Mat
  relType : Mat
  labelMs : List Mat
  relMs : List Ten
  endpoints : Nat → Option (Nat × Nat)
  attrs : List String
  nodeAttrs : Nat → List (Nat × V)
  relAttrs : Nat → List (Nat × V)
  labels : List String
  labelsIndex : String → Option Nat
  types : List String
  constraints : List Constraint
  version : Nat
  schemaVersion : Nat

end GQ
