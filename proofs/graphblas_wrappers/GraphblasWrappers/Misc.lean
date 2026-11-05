/-
# Remaining wrapper logic: `grown`/`resize` policy, `Vector::Iter`, multi-edge `Iter`

Smaller pieces of Rust-side logic in `matrix.rs` / `vector.rs` / `tensor.rs`.
-/

import GraphblasWrappers.RowIter

namespace GBW

/-! ## `Matrix::grown` (matrix.rs:739) and `resize` (tensor.rs:823)

`grown` asserts `nrows ≥ r0 && ncols ≥ c0` (matrix.rs:745-748) — it cannot shrink,
because `GrB_Matrix_resize` on smaller dims drops entries. Since #3250
`Tensor::resize` (tensor.rs:823) never shrinks: it clamps each target to
`max(target, current)` (:834-835) and re-emits each layer with `grown`, whose
precondition therefore always holds (`resize_clamp_grownOk`). Before #3250 it
branched: shrink took a flush + straight `resize` path, grow re-emitted each
layer. -/

/-- The `grown` precondition (matrix.rs:745-748). -/
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

/-! ## `Vector::Iter` (vector.rs:601-658): the cursor state machine

`Iter::new` attaches and calls `GxB_Vector_Iterator_seek(0)` (`depleted` iff that returns
`GxB_EXHAUSTED`, :619-625); `next` returns `None` once `depleted`, otherwise reads
`GxB_Vector_Iterator_getIndex` (and the value, for `Iter<u64>`), calls
`GxB_Vector_Iterator_next` and sets `depleted` from `GxB_EXHAUSTED` (:632-655).

**C iterator contract** (GraphBLAS user guide, `GxB_Vector_Iterator_*`; stated here by
the model `cInfo`/`cIndex`, not proved): the iterator visits the stored entries in
ascending index order; `seek(0)` positions on the first entry and reports `EXHAUSTED`
iff there is none; `getIndex` reads the entry under the cursor; `next` advances one
entry and reports `EXHAUSTED` exactly when it moved past the last. The entry list `es`
is the vector's stored `(index, value)` pairs in that order. -/
section VecIter
variable {α : Type}

/-- Rust `Iter` state: the C cursor position and the `depleted` flag. -/
structure VIter where
  pos : Nat
  depleted : Bool

/-- `GxB_*_Iterator_next` / `seek` report `EXHAUSTED` iff no entry is left at `pos`. -/
def cExhausted (es : List α) (pos : Nat) : Bool := decide (es.length ≤ pos)

/-- `Iter::new`: `seek(0)`, `depleted := (info == GxB_EXHAUSTED)` (vector.rs:619-625). -/
def VIter.new (es : List α) : VIter := ⟨0, cExhausted es 0⟩

/-- `Iterator::next` (vector.rs:632 for `bool`, :647 for `u64`): `getIndex`/`get` at the
cursor, then `GxB_Vector_Iterator_next` and `depleted := EXHAUSTED`. -/
def VIter.next (es : List α) (s : VIter) : Option α × VIter :=
  if s.depleted then (none, s)
  else (es[s.pos]?, ⟨s.pos + 1, cExhausted es (s.pos + 1)⟩)

/-- Call `next` until `None`, at most `fuel` times. -/
def VIter.drain (es : List α) : Nat → VIter → List α
  | 0, _ => []
  | f + 1, s =>
    match s.next es with
    | (none, _) => []
    | (some x, s') => x :: VIter.drain es f s'

/-- The state invariant: `depleted` iff the cursor is past the last entry (and it never
runs further than one past it). -/
def VIter.Inv (es : List α) (s : VIter) : Prop :=
  s.depleted = cExhausted es s.pos ∧ s.pos ≤ es.length

theorem VIter.new_inv (es : List α) : (VIter.new es).Inv es := by
  simp [VIter.new, VIter.Inv, cExhausted]

theorem VIter.new_depleted_iff (es : List α) : (VIter.new es).depleted = true ↔ es = [] := by
  simp [VIter.new, cExhausted]

theorem VIter.next_inv (es : List α) (s : VIter) (h : s.Inv es) : (s.next es).2.Inv es := by
  unfold VIter.next
  split
  · exact h
  · rename_i hd
    have hd' : s.depleted = false := by simpa using hd
    have h1 := h.1
    rw [hd'] at h1
    simp [cExhausted] at h1
    exact ⟨rfl, by show s.pos + 1 ≤ es.length; omega⟩

/-- Once depleted, `next` is `None` forever (the iterator is fused). -/
theorem VIter.next_depleted (es : List α) (s : VIter) (h : s.depleted = true) :
    s.next es = (none, s) := by simp [VIter.next, h]

/-- While not depleted, `next` yields the entry under the cursor. -/
theorem VIter.next_some (es : List α) (s : VIter) (h : s.Inv es) (hd : s.depleted = false) :
    ∃ x, es[s.pos]? = some x ∧ (s.next es).1 = some x := by
  have h1 := h.1
  simp [cExhausted, hd] at h1
  have : s.pos < es.length := by omega
  refine ⟨es[s.pos], by simp [this], ?_⟩
  simp [VIter.next, hd, this]

/-- Draining from any invariant state yields exactly the entries from the cursor on, once
each, in order. -/
theorem VIter.drain_eq (es : List α) :
    ∀ (f : Nat) (s : VIter), s.Inv es → es.length - s.pos < f → VIter.drain es f s = es.drop s.pos
  | 0, _, _, h => by omega
  | f + 1, s, hi, hf => by
    by_cases hd : s.depleted = true
    · have h1 := hi.1
      rw [hd] at h1
      simp [cExhausted] at h1
      rw [VIter.drain, VIter.next_depleted es s hd]
      simp [List.drop_eq_nil_of_le h1]
    · have hd' : s.depleted = false := by simpa using hd
      have h1 := hi.1
      rw [hd'] at h1
      simp [cExhausted] at h1
      have hlt : s.pos < es.length := by omega
      have ih := VIter.drain_eq es f ⟨s.pos + 1, cExhausted es (s.pos + 1)⟩
        ⟨rfl, by show s.pos + 1 ≤ es.length; omega⟩ (by show es.length - (s.pos + 1) < f; omega)
      have hn : s.next es = (some es[s.pos], ⟨s.pos + 1, cExhausted es (s.pos + 1)⟩) := by
        unfold VIter.next
        rw [if_neg (by simpa using hd'), List.getElem?_eq_getElem hlt]
      rw [VIter.drain, hn]
      show es[s.pos] :: VIter.drain es f ⟨s.pos + 1, cExhausted es (s.pos + 1)⟩ = _
      rw [ih]
      exact (List.drop_eq_getElem_cons hlt).symm

/-- **`Iter<bool>::next` and `Iter<u64>::next`**: a fresh iterator drains to exactly the
vector's stored entries, each once, in the C iterator's (ascending) order; for `Iter<bool>`
the entries are the indices, for `Iter<u64>` the `(index, value)` pairs. -/
theorem VIter.drain_new (es : List α) : VIter.drain es (es.length + 1) (VIter.new es) = es := by
  have := VIter.drain_eq es (es.length + 1) (VIter.new es) (VIter.new_inv es)
    (by simp [VIter.new])
  simpa [VIter.new] using this

theorem VIter.drain_new_count [DecidableEq α] (es : List α) (x : α) :
    (VIter.drain es (es.length + 1) (VIter.new es)).count x = es.count x := by
  rw [VIter.drain_new]
end VecIter

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

/-! ### `Tensor::iter_edges` (tensor.rs:1293): singles first, then every `me` id

Unlike `Iter::next` above, `iter_edges` does **not** stream pair by pair: it filters the
`MULTI_EDGE` sentinels out of the forward adjacency (`fwd_iter(..).filter(id != MULTI_EDGE)`,
:1304-1305) and chains every `me` id after them (`edge_versioned_all().flat_map(..)`,
:1295-1300; empty when `!has_multi_edge()`, :1301-1303). -/

/-- The inline edge ids of the single-edge pairs, in base order. -/
def singles : List PairEdges → List Nat
  | [] => []
  | .single id :: r => id :: singles r
  | .multi _ :: r => singles r

/-- Every `me` id of the multi-edge pairs, in base order. -/
def multis : List PairEdges → List Nat
  | [] => []
  | .single _ :: r => multis r
  | .multi ids :: r => ids ++ multis r

/-- `Tensor::iter_edges`: filter-then-chain. -/
def iterEdges (ps : List PairEdges) : List Nat := singles ps ++ multis ps

/-- The order differs from `iterExpand` only when a multi-edge pair exists: the same ids
with the same multiplicities, so the two enumerations agree as multisets. -/
theorem iterEdges_count (ps : List PairEdges) (x : Nat) :
    (iterEdges ps).count x = (iterExpand ps).count x := by
  induction ps with
  | nil => rfl
  | cons p r ih =>
    cases p with
    | single id =>
      simp only [iterEdges, singles, multis, iterExpand, expand, List.cons_append] at ih ⊢
      simp [List.count_cons, ih]
    | multi ids =>
      simp only [iterEdges, singles, multis, iterExpand, expand] at ih ⊢
      simp only [List.count_append] at ih ⊢
      omega

theorem iterEdges_length (ps : List PairEdges) :
    (iterEdges ps).length = (iterExpand ps).length := by
  have : ∀ ps : List PairEdges, (iterEdges ps).length = (iterExpand ps).length := by
    intro ps
    induction ps with
    | nil => rfl
    | cons p r ih =>
      cases p <;> simp [iterEdges, singles, multis, iterExpand, expand] at ih ⊢ <;> omega
  exact this ps

/-- With no promoted pair (`has_multi_edge()` false, `me` empty) the order is the base
order, the allocation-free common case. -/
theorem iterEdges_all_single (ids : List Nat) : iterEdges (ids.map .single) = ids := by
  have h : ∀ ids : List Nat, singles (ids.map .single) = ids ∧ multis (ids.map .single) = [] := by
    intro ids
    induction ids with
    | nil => simp [singles, multis]
    | cons i r ih => simp [singles, multis, ih.1, ih.2]
  simp [iterEdges, (h ids).1, (h ids).2]

/-- A promoted pair's ids come after every single edge, so `iter_edges` is *not* the
base-order stream `iterExpand` once a `MULTI_EDGE` pair precedes a single one. -/
example : iterEdges [.multi [1, 2], .single 3] = [3, 1, 2] := by decide
example : iterExpand [.multi [1, 2], .single 3] = [1, 2, 3] := by decide

end GBW
