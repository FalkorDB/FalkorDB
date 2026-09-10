/-
# `nvals`

`nvals` returns `|m| + |dp| − |dm|` on `u64`. Two things need proving, and the
Rust doc comment claims both:

* it is the *right* number — the size of the effective set;
* the unsigned subtraction cannot wrap.

Both rest on the two invariants, exactly as the comment says: `dm ⊆ m` bounds the
subtrahend, and `dp ∩ m = ∅` is what stops a pair being counted twice. The
`Nat` subtraction here truncates where the Rust's `u64` would wrap, so the
no-underflow theorem is a real check rather than a restatement.
-/
import VersionedMatrix.Write

namespace FalkorDB
namespace VersionedMatrix

variable {v : VersionedMatrix}

/-- **`nvals` counts the effective set.** -/
theorem nvals_eq_card (h : Inv v) : nvals v = (eff v).card := by
  unfold nvals eff
  rw [Finset.card_union_of_disjoint (eff_disjoint h), Finset.card_sdiff_of_subset h.dm_sub_m]
  -- `dm ⊆ m` is what lets the two truncating subtractions be reassociated.
  have := Finset.card_le_card h.dm_sub_m
  omega

/-- **The `u64` subtraction never wraps.** `dm ⊆ m` gives `|dm| ≤ |m|`, so the
subtrahend is dominated by the first summand alone. -/
theorem nvals_no_underflow (h : Inv v) : v.dm.card ≤ v.m.card + v.dp.card :=
  le_trans (Finset.card_le_card h.dm_sub_m) (Nat.le_add_right _ _)

/-- The `Nat` truncation is therefore vacuous: the subtraction is exact. -/
theorem nvals_add_dm (h : Inv v) : nvals v + v.dm.card = v.m.card + v.dp.card := by
  unfold nvals
  have := nvals_no_underflow h
  omega

/-! ## `nvals` under the writes

The counter moves the way each operation's effect on `eff` says it must. These are
corollaries of `Count`'s headline theorem and `Write`'s `eff_*` lemmas, and they
are what a caller reasoning about `GRAPH.MEMORY` or an entity count actually
uses. -/

theorem nvals_set (h : Inv v) {p : Coord} (hb : InBounds v p) (hp : p ∉ eff v) :
    nvals (set v p) = nvals v + 1 := by
  rw [nvals_eq_card (inv_set h hb), nvals_eq_card h, eff_set p, Finset.card_insert_of_notMem hp]

theorem nvals_set_of_mem (h : Inv v) {p : Coord} (hb : InBounds v p) (hp : p ∈ eff v) :
    nvals (set v p) = nvals v := by
  rw [nvals_eq_card (inv_set h hb), nvals_eq_card h, eff_set p, Finset.insert_eq_self.mpr hp]

theorem nvals_remove (h : Inv v) {p : Coord} (hp : p ∈ eff v) :
    nvals (remove v p) = nvals v - 1 := by
  rw [nvals_eq_card (inv_remove h p), nvals_eq_card h, eff_remove h p, Finset.card_erase_of_mem hp]

theorem nvals_removeMask (h : Inv v) (mask : Finset Coord) :
    nvals (removeMask v mask) = (eff v \ mask).card := by
  rw [nvals_eq_card (inv_removeMask h mask), eff_removeMask mask]

end VersionedMatrix
end FalkorDB
