/-
# A model of `graph/src/graph/graphblas/versioned_matrix.rs`

The abstract state of a `VersionedMatrix<bool>`, its denotation, and the two
invariants the Rust module docs state. Every later file proves that one operation
preserves the invariants and acts on the denotation the way its doc comment says.

## Why `bool` only

`VersionedMatrix<T>` is generic, but every operation — `set`, `remove`, `get`,
`set_all`, `remove_mask`, `resize`, `flush`, `iter`, `transpose` — lives in
`impl VersionedMatrix<bool>`. The `u64` instantiation exists only as the
`Delta<u64>` layers that `Tensor` owns and drives itself, and those are covered by
the `Tensor` development in this same project. So a `bool` versioned matrix is a
pure *pattern*: three `Finset Coord`s, no values.

## What is modelled

* `m`, `dp`, `dm` as explicit finite sets of coordinates, because that is what the
  operations manipulate and where the invariants live.
* The denotation is `eff v = (m ∖ dm) ∪ dp`, the "effective state" of the module
  docs.
* GraphBLAS pending work (`wait`, `wait_base`, `wait_all`, `is_synced`) has no
  denotational content: every operation here behaves as if the layers were
  materialized, which the Rust guarantees by waiting on entry. `memory_usage` and
  `print` are likewise outside the model.
* The fold *policy* (`should_fold`, `should_fold_read`, `delta_dominates_base`,
  and the approximate `Delta::count` counters they read) is deliberately not
  modelled: `flush` takes its decision as a parameter, so every theorem holds for
  every decision. See `VersionedMatrix/Fold.lean`.

## What this buys the `Tensor` proof

`Tensor` models its `mt` and `me` fields — both `VersionedMatrix<bool>` in Rust —
as plain `Finset`s of effective entries, and calls that its one abstraction
boundary. The four theorems `eff_set`, `eff_remove`, `eff_removeMask` and
`nvals_eq_card` are exactly the interface it assumes there, so this development
discharges that boundary rather than leaving it on trust.
-/
import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Lattice.Basic
import Mathlib.Data.Finset.Sort

namespace FalkorDB

/-- The abstract state of a `VersionedMatrix<bool>`: the committed base, the two
delta patterns, and the matrix bounds. -/
structure VersionedMatrix where
  /-- Base committed matrix, shared with readers. -/
  m : Finset (Nat × Nat)
  /-- Delta-plus: pending additions. -/
  dp : Finset (Nat × Nat)
  /-- Delta-minus: pending deletions (a mask over `m`). -/
  dm : Finset (Nat × Nat)
  nrows : Nat
  ncols : Nat

namespace VersionedMatrix

/-- A matrix coordinate. -/
abbrev Coord := Nat × Nat

variable {v : VersionedMatrix}

/-! ## Denotation -/

/-- **Effective state**: `(m UNION dp) MINUS dm`, as the module header draws it.
Written `(m ∖ dm) ∪ dp` — the same set, because `dp ∩ dm = ∅` follows from the
invariants, and this orientation is the one the read path computes. -/
def eff (v : VersionedMatrix) : Finset Coord := (v.m \ v.dm) ∪ v.dp

@[simp] theorem mem_eff {p : Coord} : p ∈ eff v ↔ (p ∈ v.m ∧ p ∉ v.dm) ∨ p ∈ v.dp := by
  simp [eff]

/-- A coordinate the caller may write: inside the bounds. The Rust callers
`resize` first, which is what makes this a precondition rather than a check. -/
def InBounds (v : VersionedMatrix) (p : Coord) : Prop := p.1 < v.nrows ∧ p.2 < v.ncols

/-! ## Invariants

The two clauses `set` / `remove` rely on, from the module docs. They are what
makes branching on the committed base alone sound, and what makes `nvals`'s
`|m| + |dp| − |dm|` arithmetic and `Iter`'s merge correct. -/

structure Inv (v : VersionedMatrix) : Prop where
  /-- `dp ∩ m = ∅`: a pair live in the base cannot also sit in the pending adds,
  so the `m` branch of `set`/`remove` cannot be shadowing a `dp` entry it fails
  to clear. (Valued layers weaken this to allow shadowing; `bool` layers do not.) -/
  dp_disj_m : Disjoint v.dp v.m
  /-- `dm ⊆ m`: a tombstone only ever masks a committed entry, so the `dp` branch
  cannot be adding a pair a tombstone still hides. -/
  dm_sub_m : v.dm ⊆ v.m
  /-- Every stored coordinate is inside the bounds. -/
  in_range : ∀ p ∈ v.m ∪ v.dp, InBounds v p

/-- The derived third disjointness: nothing is both a pending add and a
tombstone. -/
theorem Inv.dp_disj_dm (h : Inv v) : Disjoint v.dp v.dm :=
  Finset.disjoint_left.mpr fun _ hp hq => Finset.disjoint_left.mp h.dp_disj_m hp (h.dm_sub_m hq)

theorem Inv.not_mem_dp_of_mem_m (h : Inv v) {p : Coord} (hp : p ∈ v.m) : p ∉ v.dp :=
  fun hdp => Finset.disjoint_left.mp h.dp_disj_m hdp hp

theorem Inv.not_mem_m_of_mem_dp (h : Inv v) {p : Coord} (hp : p ∈ v.dp) : p ∉ v.m :=
  Finset.disjoint_left.mp h.dp_disj_m hp

/-- With `dp ∩ m = ∅` the union in `eff` is disjoint, which is what the counting
argument in `Count.lean` needs. -/
theorem eff_disjoint (h : Inv v) : Disjoint (v.m \ v.dm) v.dp :=
  Finset.disjoint_left.mpr fun _ hp hq =>
    h.not_mem_m_of_mem_dp hq (Finset.mem_sdiff.mp hp).1

end VersionedMatrix
end FalkorDB
