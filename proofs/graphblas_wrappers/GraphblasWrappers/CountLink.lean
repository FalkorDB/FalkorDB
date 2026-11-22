import GraphblasWrappers.Bulk
import VersionedMatrix.CountRows
/-!
# Linking `Matrix::count_in_rows` to the versioned-matrix exactness theorem

`versioned_matrix`'s `vmCountInRowsOn_spec` is exact for any counter satisfying `LayerCountOn P`.
Here the counter is `Matrix::count_in_rows` itself (`Bulk.countInRows`, `from_sorted_indices` of the row list
then `GrB_mxv` + reduce), and `P` is what that call is specified for: the layer is a well-formed GraphBLAS
matrix (`MatWf`: unique in-bounds coordinates, fewer than `2^64` entries) and the row list is sorted and
inside the matrix's height (`from_sorted_indices`'s `debug_assert!`ed precondition, and `rows` must be `nrows`
long). `countLayer_spec` proves the contract from `countInRows_spec`, so
`VersionedMatrix::count_in_rows` is exact under those conditions on its three layers, with no remaining
hypothesis about `count_in_rows`.
-/
namespace GBWBulk
open VMMat VMCount

/-- A `versioned_matrix` layer as a `MatC`. -/
def toMatC (l : Mat Unit) : MatC := ⟨keys l, l.nrows, l.ncols⟩

/-- What `count_in_rows` is called on: a well-formed matrix and a sorted in-bounds row list. -/
def CountOk (l : Mat Unit) (S : List Nat) : Prop :=
  MatWf (toMatC l) ∧ S.Pairwise (· < ·) ∧ ∀ i ∈ S, i < l.nrows

/-- `Matrix::count_in_rows(rows)` with `rows = from_sorted_indices(nrows, S)`. -/
def countLayer (G : BulkSpec) (l : Mat Unit) (S : List Nat) : Nat :=
  countInRows G (toMatC l) (fromSortedIndices G l.nrows S)

theorem countLayer_spec (G : BulkSpec) (l : Mat Unit) (S : List Nat) (h : CountOk l S) :
    countLayer G l S = (keys l).countP (inRows S) := by
  unfold countLayer
  have := countInRows_fromSorted G (toMatC l) S h.2.1 h.2.2 h.1
  change countInRows G (toMatC l) (fromSortedIndices G l.nrows S) = _ at this
  rw [this, List.countP_eq_length_filter]
  rfl

/-- The `LayerCountOn` instance: `Matrix::count_in_rows` on well-formed inputs. -/
def layerCountG (G : BulkSpec) : LayerCountOn CountOk :=
  ⟨countLayer G, fun l S h => countLayer_spec G l S h⟩

/-- **`VersionedMatrix::count_in_rows`, linked.** With `Matrix::count_in_rows` as the layer counter, the result is
the number of effective entries `(m ∖ dm) ∪ dp` in the selected rows and `layer(dm) ≤ layer(m)` (no `u64`
underflow), for a sorted in-bounds row list and well-formed layers satisfying `dm ⊆ m` with duplicate-free
`m`/`dm` keys (the layer invariants). -/
theorem vmCountInRows_linked (G : BulkSpec) (v : VMOps.BVM) (S : List Nat)
    (hm : (keys v.m).Nodup) (hdm : (keys v.dm.layer).Nodup)
    (hsub : ∀ p ∈ keys v.dm.layer, p ∈ keys v.m)
    (hPm : CountOk (VMMat.wait (VMOps.wait v).m) S) (hPdp : CountOk (VMOps.wait v).dp.layer S)
    (hPdm : CountOk (VMOps.wait v).dm.layer S) :
    vmCountInRowsOn (layerCountG G) v S = (effKeys v).countP (inRows S) ∧
    layerOn (layerCountG G) (VMOps.wait v).dm.layer S ≤
      layerOn (layerCountG G) (VMMat.wait (VMOps.wait v).m) S :=
  vmCountInRowsOn_spec (layerCountG G) v S hm hdm hsub hPm hPdp hPdm

end GBWBulk
