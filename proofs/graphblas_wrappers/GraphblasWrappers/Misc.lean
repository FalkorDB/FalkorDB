/-
# Remaining wrapper logic: `grown`/`resize` policy, `Vector::Iter`, multi-edge `Iter`

Smaller pieces of Rust-side logic in `matrix.rs` / `vector.rs` / `tensor.rs`.
-/

import GraphblasWrappers.RowIter

namespace GBW

/-! ## `Matrix::grown` (matrix.rs:715) and `resize` (tensor.rs:823)

`grown` asserts `nrows ≥ r0 && ncols ≥ c0` (matrix.rs:721) — it cannot shrink,
because `GrB_Matrix_resize` on smaller dims drops entries. Since #3250
`Tensor::resize` (tensor.rs:823) never shrinks: it clamps each target to
`max(target, current)` (:834-835) and re-emits each layer with `grown`, whose
precondition therefore always holds (`resize_clamp_grownOk`). Before #3250 it
branched: shrink took a flush + straight `resize` path, grow re-emitted each
layer (`pre3250_resize_grow_iff_grownOk`). -/

/-- The `grown` precondition (matrix.rs:721-724). -/
def grownOk (r0 c0 nrows ncols : Nat) : Bool := decide (nrows ≥ r0 ∧ ncols ≥ c0)

/-- The assert fires (returns `false`) exactly on a shrink of either dimension —
so `grown` never silently drops entries. -/
theorem grown_rejects_shrink (r0 c0 nrows ncols : Nat) :
    grownOk r0 c0 nrows ncols = false ↔ (nrows < r0 ∨ ncols < c0) := by
  simp only [grownOk, decide_eq_false_iff_not]
  omega

/-- `Tensor::resize` on main (tensor.rs:834-835): the clamped dims always
satisfy `grown`'s precondition, so the assert never fires and nothing is
dropped. -/
theorem resize_clamp_grownOk (r0 c0 nrows ncols : Nat) :
    grownOk r0 c0 (max nrows r0) (max ncols c0) = true := by
  simp only [grownOk, decide_eq_true_eq]
  omega

/-- Main: the clamp never asks for a smaller dimension. -/
theorem resize_clamp_not_shrink (r0 c0 nrows ncols : Nat) :
    ¬ (max nrows r0 < r0 ∨ max ncols c0 < c0) := by omega

/-- History (before #3250, 0f45a8685 tensor.rs:809): the `nrows < m.nrows() ||
ncols < m.ncols()` guard picked the shrink path. Its complement is exactly
`grownOk`, so the grow path ran precisely when `grown` would accept. -/
theorem pre3250_resize_grow_iff_grownOk (r0 c0 nrows ncols : Nat) :
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

/-! ## `Tensor::iter` multi-edge expansion (tensor.rs:1763) / `EdgeIds`

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

/-- Expand one pair to its edge ids — `EdgeIds` (tensor.rs:1676). -/
def expand : PairEdges → List Nat
  | .single id => [id]
  | .multi ids => ids

/-- `Iter::next` streamed over the base pairs: the inline id, or the buffered
`me` row, for each pair in turn (tensor.rs:1763-1821). -/
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
the allocation-free common case (tensor.rs:1293 `iter_edges` doc). -/
theorem iterExpand_all_single (ids : List Nat) :
    iterExpand (ids.map .single) = ids := by
  induction ids with
  | nil => rfl
  | cons i r ih => simp [iterExpand, expand, ih]

end GBW
