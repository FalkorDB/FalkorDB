/-
# Remaining wrapper logic: `grown`/`resize` policy, `Vector::Iter`, multi-edge `Iter`

Smaller pieces of Rust-side logic in `matrix.rs` / `vector.rs` / `tensor.rs`.
-/

import GraphblasWrappers.RowIter

namespace GBW

/-! ## `Matrix::grown` (matrix.rs:701) and `resize` (tensor.rs:809)

`grown` asserts `nrows ≥ r0 && ncols ≥ c0` (matrix.rs:707) — it cannot shrink,
because `GrB_Matrix_resize` on smaller dims drops entries. `Tensor::resize`
(tensor.rs:809) branches: shrink takes the flush + straight `resize` path; grow
re-emits each layer. -/

/-- The `grown` precondition (matrix.rs:707-710). -/
def grownOk (r0 c0 nrows ncols : Nat) : Bool := decide (nrows ≥ r0 ∧ ncols ≥ c0)

/-- The assert fires (returns `false`) exactly on a shrink of either dimension —
so `grown` never silently drops entries. -/
theorem grown_rejects_shrink (r0 c0 nrows ncols : Nat) :
    grownOk r0 c0 nrows ncols = false ↔ (nrows < r0 ∨ ncols < c0) := by
  simp only [grownOk, decide_eq_false_iff_not]
  omega

/-- `Tensor::resize`'s shrink test (tensor.rs:809): the `nrows < m.nrows() ||
ncols < m.ncols()` guard picks the shrink path. Its complement is exactly
`grownOk`, so the grow path runs precisely when `grown` would accept. -/
theorem resize_grow_iff_grownOk (r0 c0 nrows ncols : Nat) :
    (decide (nrows < r0 ∨ ncols < c0) = false) ↔ grownOk r0 c0 nrows ncols = true := by
  simp only [grownOk, decide_eq_false_iff_not, decide_eq_true_eq]
  omega

/-! ## `Vector::Iter` (vector.rs): yields every stored index once, ascending

`GxB_Vector_Iterator_seek(0)` then `next` until `GxB_EXHAUSTED` (vector.rs). The
stored indices come out ascending; the driver yields each once. Modelled as the
stored-index list; `depleted` is set at `EXHAUSTED`. -/

/-- Drive the vector iterator: yield each stored index once (vector.rs
`Iterator for Iter`). -/
def vecDrain : List Nat → List Nat
  | [] => []
  | i :: rest => i :: vecDrain rest

theorem vecDrain_eq (idxs : List Nat) : vecDrain idxs = idxs := by
  induction idxs with
  | nil => rfl
  | cons i r ih => simp [vecDrain, ih]

/-- So the vector iterator enumerates the stored indices exactly, once each. -/
theorem vecDrain_count (idxs : List Nat) (x : Nat) :
    (vecDrain idxs).count x = idxs.count x := by rw [vecDrain_eq]

/-! ## `Tensor::iter` multi-edge expansion (tensor.rs:1740) / `EdgeIds`

`Iter::next` yields the inline id of a single-edge pair, or every id of a
`MULTI_EDGE` pair's `me` row (ascending), buffered and drained one at a time
(the `buf`/`buf_pos` machinery). Not covered by `proofs/versioned_matrix` (which
models one pair's `edges`, not the streaming expansion across pairs). Modelled
here: a base sequence of pairs, each tagged inline-id or a list of me-ids. -/

/-- A decoded base pair: either a single inline edge id, or a promoted pair
carrying its `me` row (in ascending order, length ≥ 2 by promotion). -/
inductive PairEdges where
  | single (id : Nat)
  | multi (ids : List Nat)
deriving Repr

/-- Expand one pair to its edge ids — `EdgeIds` (tensor.rs:1653). -/
def expand : PairEdges → List Nat
  | .single id => [id]
  | .multi ids => ids

/-- `Iter::next` streamed over the base pairs: the inline id, or the buffered
`me` row, for each pair in turn (tensor.rs:1740-1798). -/
def iterExpand : List PairEdges → List Nat
  | [] => []
  | p :: rest => expand p ++ iterExpand rest

/-- The iterator yields exactly the concatenation of every pair's edges, in
base order — nothing dropped, nothing duplicated, `me` rows spliced in place. -/
theorem iterExpand_eq_flatMap (ps : List PairEdges) :
    iterExpand ps = ps.flatMap expand := by
  induction ps with
  | nil => rfl
  | cons p r ih => simp [iterExpand, ih, List.flatMap_cons]

/-- Total edge count of an iteration is the sum of per-pair counts. -/
theorem iterExpand_length (ps : List PairEdges) :
    (iterExpand ps).length = (ps.map (fun p => (expand p).length)).sum := by
  induction ps with
  | nil => rfl
  | cons p r ih => simp [iterExpand, ih, List.length_append]

/-- A single-edge graph (`me` empty everywhere) iterates as one id per pair —
the allocation-free common case (tensor.rs:1283 `iter_edges` doc). -/
theorem iterExpand_all_single (ids : List Nat) :
    iterExpand (ids.map .single) = ids := by
  induction ids with
  | nil => rfl
  | cons i r ih => simp [iterExpand, expand, ih]

end GBW
