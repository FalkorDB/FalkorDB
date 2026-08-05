/-
# A model of `graph/src/graph/graphblas/tensor.rs`

This file defines the abstract state of `Tensor`, its *denotation* (the multi-graph
it represents), and the invariants stated in the Rust module docs.  Every later
file proves that one operation of `tensor.rs` preserves the invariants and acts
on the denotation the way its doc comment claims.

## What is modelled

* `Layer α` — a GraphBLAS matrix: a finite support (`dom`) plus a value function.
  `nvals` is `dom.card`, so counting arguments are plain `Finset.card` arguments.
* The three forward layers `m`, `dp`, `dm` are modelled explicitly, because
  `tensor.rs` manipulates them directly and all the delicate invariants live there.
* `mt` (backward adjacency) and `me` (multi-edge id storage) are `VersionedMatrix`
  values in Rust.  Their *own* three-layer algebra belongs to
  `versioned_matrix.rs`; from the tensor's point of view they are sets
  (`mt : Finset Pair` of `(dst, src)`, `me : Finset (key × edge_id)`), which is
  exactly the interface `tensor.rs` uses (`set`/`remove`/`remove_mask`/`iter`/
  `nvals`).  That is the one abstraction boundary of this development.
* GraphBLAS "pending work" (`wait`, `wait_all`, `is_synced`, `pending`) is a
  runtime/materialisation concern with no denotational content: every operation
  here behaves as if `wait_fwd()` had already run, which is what `tensor.rs`
  guarantees by calling it on entry.  `Tensor::memory_usage` is likewise outside
  the model.
-/
import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Lattice.Basic
import Mathlib.Data.Finset.Sort
import Mathlib.Algebra.Order.BigOperators.Group.Finset

namespace FalkorDB

/-- A `(src, dst)` node-id pair, i.e. a matrix coordinate. -/
abbrev Pair := Nat × Nat

/-- Node ids are packed two-to-a-`u64` by `compound_key`, so each side must fit
in a `u32`.  `tensor.rs` asserts this unconditionally. -/
def Bounded (p : Pair) : Prop := p.1 < 2 ^ 32 ∧ p.2 < 2 ^ 32

instance (p : Pair) : Decidable (Bounded p) := by
  unfold Bounded; infer_instance

/-- `GrB_INDEX_MAX`, the largest GraphBLAS index (`2^60 - 1`). -/
def GrBIndexMax : Nat := 2 ^ 60 - 1

/-- `MULTI_EDGE`: the sentinel inline value of a pair with more than one edge. -/
def MULTI : Nat := 2 ^ 64 - 1

/-- A real edge id: bounded by `GrB_INDEX_MAX`, hence never equal to `MULTI`. -/
def ValidId (i : Nat) : Prop := i ≤ GrBIndexMax

theorem ValidId.ne_multi {i : Nat} (h : ValidId i) : i ≠ MULTI := by
  have hlt : (2:Nat) ^ 60 < 2 ^ 64 := Nat.pow_lt_pow_right (by omega) (by omega)
  simp only [ValidId, GrBIndexMax] at h
  simp only [MULTI]
  omega

theorem multi_not_valid : ¬ ValidId MULTI := fun h => h.ne_multi rfl

/-! ## A GraphBLAS matrix as a finite map -/

/-- A sparse matrix: the sparsity pattern `dom` together with a value at every
coordinate (only the values on `dom` are observable). -/
structure Layer (α : Type) where
  /-- The sparsity pattern (`GrB_Matrix_nvals` counts this). -/
  dom : Finset Pair
  /-- The stored value; only meaningful on `dom`. -/
  val : Pair → α

namespace Layer

variable {α : Type}

/-- `GrB_Matrix_extractElement`: `none` outside the pattern. -/
def get (L : Layer α) (p : Pair) : Option α :=
  if p ∈ L.dom then some (L.val p) else none

/-- `GrB_Matrix_nvals`. -/
def nvals (L : Layer α) : Nat := L.dom.card

/-- `GrB_Matrix_setElement`. -/
def set (L : Layer α) (p : Pair) (v : α) : Layer α :=
  { dom := insert p L.dom, val := fun q => if q = p then v else L.val q }

/-- `GrB_Matrix_removeElement`. -/
def remove (L : Layer α) (p : Pair) : Layer α := { L with dom := L.dom.erase p }

/-- `Matrix::remove_all` / `GrB_assign` with a complemented mask. -/
def removeAll (L : Layer α) (s : Finset Pair) : Layer α := { L with dom := L.dom \ s }

/-- `GrB_Matrix_clear`. -/
def clear (L : Layer α) : Layer α := { L with dom := ∅ }

/-- The empty matrix. -/
def empty [Inhabited α] : Layer α := { dom := ∅, val := fun _ => default }

@[simp] theorem mem_dom_set {L : Layer α} {p q : Pair} {v : α} :
    q ∈ (L.set p v).dom ↔ q = p ∨ q ∈ L.dom := by
  simp [set]

@[simp] theorem get_set_self {L : Layer α} {p : Pair} {v : α} : (L.set p v).get p = some v := by
  simp [get, set]

@[simp] theorem get_set_ne {L : Layer α} {p q : Pair} {v : α} (h : q ≠ p) :
    (L.set p v).get q = L.get q := by
  simp [get, set, h]

@[simp] theorem get_remove_self {L : Layer α} {p : Pair} : (L.remove p).get p = none := by
  simp [get, remove]

@[simp] theorem get_remove_ne {L : Layer α} {p q : Pair} (h : q ≠ p) :
    (L.remove p).get q = L.get q := by
  simp [get, remove, Finset.mem_erase, h]

@[simp] theorem get_clear {L : Layer α} {p : Pair} : L.clear.get p = none := by
  simp [get, clear]

@[simp] theorem get_removeAll_mem {L : Layer α} {s : Finset Pair} {p : Pair} (h : p ∈ s) :
    (L.removeAll s).get p = none := by
  simp [get, removeAll, h]

@[simp] theorem get_removeAll_not_mem {L : Layer α} {s : Finset Pair} {p : Pair} (h : p ∉ s) :
    (L.removeAll s).get p = L.get p := by
  simp [get, removeAll, h]

theorem get_eq_none {L : Layer α} {p : Pair} : L.get p = none ↔ p ∉ L.dom := by
  simp [get]

theorem get_isSome {L : Layer α} {p : Pair} : (L.get p).isSome ↔ p ∈ L.dom := by
  simp [get]

theorem get_eq_some {L : Layer α} {p : Pair} {v : α} :
    L.get p = some v ↔ p ∈ L.dom ∧ L.val p = v := by
  simp [get]

theorem get_of_mem {L : Layer α} {p : Pair} (h : p ∈ L.dom) : L.get p = some (L.val p) := by
  simp [get, h]

@[simp] theorem dom_remove {L : Layer α} {p : Pair} : (L.remove p).dom = L.dom.erase p := rfl

@[simp] theorem dom_removeAll {L : Layer α} {s : Finset Pair} :
    (L.removeAll s).dom = L.dom \ s := rfl

@[simp] theorem dom_clear {L : Layer α} : (L.clear).dom = (∅ : Finset Pair) := rfl

@[simp] theorem dom_set {L : Layer α} {p : Pair} {v : α} :
    (L.set p v).dom = insert p L.dom := rfl

end Layer

/-! ## The tensor state -/

/-- The abstract state of `Tensor` (`tensor.rs`).

`m`/`dp`/`dm` are the three forward delta layers holding **inline edge ids**;
`mt` is the backward `(dst, src)` structure; `me` maps `compound_key src dst` to
the edge ids of multi-edge pairs; `multiCount` counts pairs whose effective
inline value is `MULTI_EDGE`. -/
structure Tensor where
  /-- Committed base, `(src, dst) ↦ inline edge id` (or `MULTI`). -/
  m : Layer Nat
  /-- Pending additions / in-place updates. -/
  dp : Layer Nat
  /-- Pending deletions: a pure mask over `m`. -/
  dm : Finset Pair
  /-- Backward adjacency, oriented `(dst, src)`, structure only. -/
  mt : Finset Pair
  /-- Multi-edge ids: `(compound_key src dst, edge_id)`. -/
  me : Finset (Nat × Nat)
  /-- Number of pairs whose effective inline value is `MULTI`. -/
  multiCount : Nat
  /-- Forward-matrix row capacity (`GrB_Matrix_nrows`), grown by `resize`. -/
  nrows : Nat
  /-- Forward-matrix column capacity (`GrB_Matrix_ncols`), grown by `resize`. -/
  ncols : Nat

namespace Tensor

/-- `compound_key src dst = (src << 32) | dst`, as a number.  `Key.lean` proves
this agrees with the bitwise form and is injective on `Bounded` pairs. -/
def key (p : Pair) : Nat := p.1 * 2 ^ 32 + p.2

/-- `Tensor::eff_get`: `dp` wins, else `m` unless masked by `dm`. -/
def effGet (t : Tensor) (p : Pair) : Option Nat :=
  match t.dp.get p with
  | some v => some v
  | none => if p ∈ t.dm then none else t.m.get p

/-- The effective forward sparsity pattern `(m ∖ dm) ∪ dp`. -/
def effDom (t : Tensor) : Finset Pair := (t.m.dom \ t.dm) ∪ t.dp.dom

/-- The edge ids stored under compound key `k` in an `me` set. -/
def meRowOf (s : Finset (Nat × Nat)) (k : Nat) : Finset Nat :=
  (s.filter (fun x => x.1 = k)).image (fun x => x.2)

/-- The `me` row of a compound key: the edge ids stored there. -/
def meRow (t : Tensor) (k : Nat) : Finset Nat := meRowOf t.me k

/-- **The denotation**: the set of edge ids the tensor stores at pair `p`.
A single-edge pair answers from its inline value; a `MULTI` pair from its `me`
row; an absent or deleted pair is empty. -/
def edgesAt (t : Tensor) (p : Pair) : Finset Nat :=
  match t.effGet p with
  | none => ∅
  | some v => if v = MULTI then t.meRow (key p) else {v}

/-- The pairs that currently have at least one edge. -/
def support (t : Tensor) : Finset Pair := t.effDom

/-- A coordinate an operation may write: it must fit the compound key (`u32` per
side, asserted by `compound_key`) and lie inside the matrix capacity (the caller
grows it with `resize` first). -/
def InBounds (t : Tensor) (p : Pair) : Prop :=
  Bounded p ∧ p.1 < t.nrows ∧ p.2 < t.ncols

/-- The pairs whose effective inline value is the `MULTI` sentinel. -/
def multiPairs (t : Tensor) : Finset Pair :=
  t.effDom.filter (fun p => t.effGet p = some MULTI)

/-! ### Basic facts about the effective view -/

@[simp] theorem mem_effDom_iff_isSome {t : Tensor} {p : Pair} :
    p ∈ t.effDom ↔ (t.effGet p).isSome := by
  by_cases hdp : p ∈ t.dp.dom
  · simp [effDom, effGet, Layer.get_of_mem hdp, hdp]
  · have h1 : t.dp.get p = none := Layer.get_eq_none.mpr hdp
    by_cases hdm : p ∈ t.dm
    · simp [effDom, effGet, h1, hdp, hdm]
    · by_cases hm : p ∈ t.m.dom
      · simp [effDom, effGet, h1, hdp, hdm, hm, Layer.get_of_mem hm]
      · simp [effDom, effGet, h1, hdp, hdm, hm, Layer.get_eq_none.mpr hm]

theorem effGet_eq_none_iff {t : Tensor} {p : Pair} : t.effGet p = none ↔ p ∉ t.effDom := by
  rw [mem_effDom_iff_isSome]; cases t.effGet p <;> simp

theorem edgesAt_eq_empty_of_not_mem {t : Tensor} {p : Pair} (h : p ∉ t.effDom) :
    t.edgesAt p = ∅ := by
  simp [edgesAt, effGet_eq_none_iff.mpr h]

theorem effGet_of_dp {t : Tensor} {p : Pair} {v : Nat} (h : t.dp.get p = some v) :
    t.effGet p = some v := by simp [effGet, h]

theorem effGet_of_m {t : Tensor} {p : Pair} (h1 : t.dp.get p = none) (h2 : p ∉ t.dm) :
    t.effGet p = t.m.get p := by simp [effGet, h1, h2]

theorem mem_meRowOf {s : Finset (Nat × Nat)} {k i : Nat} : i ∈ meRowOf s k ↔ (k, i) ∈ s := by
  simp only [meRowOf, Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨⟨k', i'⟩, ⟨hmem, hk⟩, hi⟩
    simp only at hk hi
    subst hk; subst hi; exact hmem
  · intro h; exact ⟨(k, i), ⟨h, rfl⟩, rfl⟩

theorem mem_meRow {t : Tensor} {k i : Nat} : i ∈ t.meRow k ↔ (k, i) ∈ t.me := mem_meRowOf

@[simp] theorem meRowOf_empty {k : Nat} : meRowOf ∅ k = ∅ := by simp [meRowOf]

@[simp] theorem meRowOf_insert_self {s : Finset (Nat × Nat)} {k i : Nat} :
    meRowOf (insert (k, i) s) k = insert i (meRowOf s k) := by
  ext j; simp [mem_meRowOf]

theorem meRowOf_insert_ne {s : Finset (Nat × Nat)} {k k' i : Nat} (h : k' ≠ k) :
    meRowOf (insert (k, i) s) k' = meRowOf s k' := by
  ext j; simp [mem_meRowOf, h]

@[simp] theorem meRowOf_erase_self {s : Finset (Nat × Nat)} {k i : Nat} :
    meRowOf (s.erase (k, i)) k = (meRowOf s k).erase i := by
  ext j; simp [mem_meRowOf]

theorem meRowOf_erase_ne {s : Finset (Nat × Nat)} {k k' i : Nat} (h : k' ≠ k) :
    meRowOf (s.erase (k, i)) k' = meRowOf s k' := by
  ext j; simp [mem_meRowOf, h]

/-! ### Transferring the effective view between related states

Every operation touches only some of the eight fields; these lemmas turn "the
fields `effGet` reads are unchanged" into "`effGet` is unchanged". -/

theorem effGet_congr {t t' : Tensor} (hm : t'.m = t.m) (hdp : t'.dp = t.dp) (hdm : t'.dm = t.dm)
    (q : Pair) : t'.effGet q = t.effGet q := by simp [effGet, hm, hdp, hdm]

theorem effGet_congr_at {t t' : Tensor} {q : Pair} (hm : t'.m = t.m)
    (hdp : t'.dp.get q = t.dp.get q) (hdm : q ∈ t'.dm ↔ q ∈ t.dm) :
    t'.effGet q = t.effGet q := by
  simp only [effGet, hdp, hm]
  cases t.dp.get q with
  | some _ => rfl
  | none => by_cases h : q ∈ t.dm <;> simp [h, hdm]

theorem effDom_congr {t t' : Tensor} (hm : t'.m = t.m) (hdp : t'.dp = t.dp) (hdm : t'.dm = t.dm) :
    t'.effDom = t.effDom := by simp [effDom, hm, hdp, hdm]

theorem edgesAt_congr {t t' : Tensor} (hm : t'.m = t.m) (hdp : t'.dp = t.dp) (hdm : t'.dm = t.dm)
    (hme : t'.me = t.me) (q : Pair) : t'.edgesAt q = t.edgesAt q := by
  simp [edgesAt, effGet_congr hm hdp hdm q, meRow, hme]

/-- The effective domain, computed from the effective lookup: this is how every
`effDom` update below is discharged. -/
theorem effDom_eq_of_effGet {t t' : Tensor} {p : Pair}
    (hself : (t'.effGet p).isSome) (hne : ∀ q, q ≠ p → t'.effGet q = t.effGet q) :
    t'.effDom = insert p t.effDom := by
  ext q
  by_cases hq : q = p
  · subst hq; simp [mem_effDom_iff_isSome, hself]
  · simp [mem_effDom_iff_isSome, hq, hne q hq]

theorem effDom_eq_of_effGet_of_mem {t t' : Tensor} {p : Pair} (hp : p ∈ t.effDom)
    (hself : (t'.effGet p).isSome) (hne : ∀ q, q ≠ p → t'.effGet q = t.effGet q) :
    t'.effDom = t.effDom := by
  rw [effDom_eq_of_effGet hself hne, Finset.insert_eq_self.mpr hp]

theorem edgesAt_eq_of_effGet_eq {t t' : Tensor} {q : Pair} (hme : t'.me = t.me)
    (hget : t'.effGet q = t.effGet q) : t'.edgesAt q = t.edgesAt q := by
  simp [edgesAt, hget, meRow, hme]

@[simp] theorem edgesAt_of_none {t : Tensor} {q : Pair} (hg : t.effGet q = none) :
    t.edgesAt q = ∅ := by simp [edgesAt, hg]

theorem edgesAt_of_multi {t : Tensor} {q : Pair} (hg : t.effGet q = some MULTI) :
    t.edgesAt q = t.meRow (key q) := by simp [edgesAt, hg]

theorem edgesAt_of_single {t : Tensor} {q : Pair} {w : Nat} (hg : t.effGet q = some w)
    (hM : w ≠ MULTI) : t.edgesAt q = {w} := by simp [edgesAt, hg, hM]

/-- Two states agree at `q` as soon as the effective value and the `me` row agree
there. -/
theorem edgesAt_congr_at {t t' : Tensor} {q : Pair} (hget : t'.effGet q = t.effGet q)
    (hrow : t'.meRow (key q) = t.meRow (key q)) : t'.edgesAt q = t.edgesAt q := by
  unfold edgesAt
  rw [hget]
  cases t.effGet q with
  | none => rfl
  | some w => by_cases hM : w = MULTI <;> simp [hM, hrow]

theorem dp_get_eq_none_of_effGet_none {t : Tensor} {p : Pair} (h : t.effGet p = none) :
    t.dp.get p = none := by
  unfold effGet at h
  cases hdp : t.dp.get p with
  | none => rfl
  | some v => rw [hdp] at h; exact absurd h (by simp)

theorem m_get_eq_none_of_effGet_none {t : Tensor} {p : Pair} (h : t.effGet p = none)
    (hdm : p ∉ t.dm) : t.m.get p = none := by
  rw [effGet_of_m (dp_get_eq_none_of_effGet_none h) hdm] at h
  exact h

/-- Transposing a coordinate is injective. -/
theorem swap_eq_iff {q p : Pair} : ((q.2, q.1) = (p.2, p.1)) ↔ q = p := by
  simp [Prod.ext_iff]
  tauto

/-- `mt` mirrors the effective structure after a single-pair insertion. -/
theorem mt_eq_insert {t t' : Tensor} {p : Pair}
    (h : ∀ q : Pair, q ∈ t.effDom ↔ (q.2, q.1) ∈ t.mt)
    (hdom : t'.effDom = insert p t.effDom) (hmt : t'.mt = insert (p.2, p.1) t.mt) :
    ∀ q : Pair, q ∈ t'.effDom ↔ (q.2, q.1) ∈ t'.mt := by
  intro q
  rw [hdom, hmt, Finset.mem_insert, Finset.mem_insert, h q, Prod.ext_iff, Prod.ext_iff]
  tauto

/-! ### `multiPairs` under a single-pair update -/

theorem multiPairs_congr {t t' : Tensor} (hdom : t'.effDom = t.effDom)
    (hget : ∀ q, t'.effGet q = t.effGet q) : t'.multiPairs = t.multiPairs := by
  ext q; simp [multiPairs, hdom, hget q]

theorem multiPairs_eq_insert {t t' : Tensor} {p : Pair} (hdom : t'.effDom = insert p t.effDom)
    (hself : t'.effGet p = some MULTI) (hne : ∀ q, q ≠ p → t'.effGet q = t.effGet q) :
    t'.multiPairs = insert p t.multiPairs := by
  ext q
  simp only [multiPairs, Finset.mem_filter, hdom, Finset.mem_insert]
  by_cases hq : q = p
  · subst hq; simp [hself]
  · simp [hq, hne q hq]

theorem multiPairs_eq_of_not_multi {t t' : Tensor} {p : Pair} (hp : p ∉ t.effDom)
    (hdom : t'.effDom = insert p t.effDom) (hself : t'.effGet p ≠ some MULTI)
    (hne : ∀ q, q ≠ p → t'.effGet q = t.effGet q) : t'.multiPairs = t.multiPairs := by
  ext q
  simp only [multiPairs, Finset.mem_filter, hdom, Finset.mem_insert]
  by_cases hq : q = p
  · subst hq
    simp only [hself, and_false, false_iff]
    exact fun hc => hp hc.1
  · simp [hq, hne q hq]

theorem mem_multiPairs {t : Tensor} {q : Pair} :
    q ∈ t.multiPairs ↔ q ∈ t.effDom ∧ t.effGet q = some MULTI := Finset.mem_filter

/-- `p ∉ multiPairs` when `p` is absent (used by the first-edge branch). -/
theorem not_mem_multiPairs_of_not_mem {t : Tensor} {p : Pair} (hp : p ∉ t.effDom) :
    p ∉ t.multiPairs := fun hc => hp (Finset.mem_filter.mp hc).1

theorem not_mem_multiPairs_of_ne {t : Tensor} {p : Pair} (hp : t.effGet p ≠ some MULTI) :
    p ∉ t.multiPairs := fun hc => hp (Finset.mem_filter.mp hc).2

/-! ## The invariants

These are exactly the "Delta-Layer Invariants" of the Rust module docs. -/

/-- Every invariant except the one about the backward matrix `mt`.

`decode` leaves `mt` empty on purpose — the caller restores it with
`rebuild_backward` — so that one step is separated out. -/
structure InvCore (t : Tensor) : Prop where
  /-- `dm ⊆ m`: the deletion mask only masks committed entries. -/
  dm_sub_m : t.dm ⊆ t.m.dom
  /-- `dp ∩ dm = ∅`: `dm` marks pure deletions only. -/
  dp_disj_dm : Disjoint t.dp.dom t.dm
  /-- *Cancel-to-clean*: `dp` never holds a value equal to the pair's live `m`
  value (in particular `dp = M` never shadows `m = M`). -/
  cancel_clean : ∀ p ∈ t.dp.dom, t.m.get p ≠ some (t.dp.val p)
  /-- A pair has ≥ 2 edges iff its effective inline value is `MULTI`, and then
  *all* of its ids live in `me`. -/
  multi_iff : ∀ p, t.effGet p = some MULTI → 2 ≤ (t.meRow (key p)).card
  /-- Otherwise `me` has no row for the pair. -/
  row_empty : ∀ p, Bounded p → t.effGet p ≠ some MULTI → t.meRow (key p) = ∅
  /-- Every `me` entry belongs to a live pair, and pairs are `u32`-bounded so
  their compound keys cannot collide. -/
  me_keyed : ∀ x ∈ t.me, ∃ p, Bounded p ∧ p ∈ t.effDom ∧ x.1 = key p
  /-- Node ids fit in a `u32` (`compound_key`'s assertion). -/
  bounded : ∀ p ∈ t.effDom, Bounded p
  /-- Every stored coordinate — committed or pending — is inside the matrix
  capacity.  `resize` only grows, so it preserves this. -/
  in_range : ∀ p ∈ t.m.dom ∪ t.dp.dom, p.1 < t.nrows ∧ p.2 < t.ncols
  /-- `multi_count` counts the `MULTI` pairs. -/
  multi_count_eq : t.multiCount = t.multiPairs.card
  /-- Stored edge ids are real GraphBLAS indices, so they never collide with the
  `MULTI` sentinel. -/
  valid_ids : ∀ p, ∀ i ∈ t.edgesAt p, ValidId i

/-- The invariant bundle every `Tensor` operation must preserve. -/
structure Inv (t : Tensor) : Prop extends InvCore t where
  /-- `mt` mirrors the effective forward structure. -/
  mt_eq : ∀ p : Pair, p ∈ t.effDom ↔ (p.2, p.1) ∈ t.mt

/-! ### Layer reshuffles

`flush` and `decode` move entries between the layers without changing the
effective view.  These two lemmas discharge the invariants for any such step. -/

theorem invCore_of_effGet_eq {t t' : Tensor} (h : InvCore t)
    (hget : ∀ q, t'.effGet q = t.effGet q) (hme : t'.me = t.me)
    (hmc : t'.multiCount = t.multiCount) (hsub : t'.dm ⊆ t'.m.dom)
    (hdisj : Disjoint t'.dp.dom t'.dm)
    (hcc : ∀ q ∈ t'.dp.dom, t'.m.get q ≠ some (t'.dp.val q))
    (hrange : ∀ q ∈ t'.m.dom ∪ t'.dp.dom, q.1 < t'.nrows ∧ q.2 < t'.ncols) : InvCore t' := by
  have hdom : t'.effDom = t.effDom := by
    ext q; rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, hget q]
  have hrow : ∀ k, t'.meRow k = t.meRow k := fun k => by rw [meRow, meRow, hme]
  have hedges : ∀ q, t'.edgesAt q = t.edgesAt q :=
    fun q => edgesAt_congr_at (hget q) (hrow _)
  refine { dm_sub_m := hsub, dp_disj_dm := hdisj, cancel_clean := hcc, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, bounded := ?_, in_range := hrange,
           multi_count_eq := ?_, valid_ids := ?_ }
  · intro q hq
    rw [hrow]
    exact h.multi_iff q (by rw [← hget q]; exact hq)
  · intro q hbq hq
    rw [hrow]
    exact h.row_empty q hbq (by rw [← hget q]; exact hq)
  · intro x hx
    obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed x (by rw [hme] at hx; exact hx)
    exact ⟨q, hbq, by rw [hdom]; exact hqdom, hqk⟩
  · rw [hdom]; exact h.bounded
  · rw [hmc, h.multi_count_eq]
    exact congrArg Finset.card (multiPairs_congr hdom hget).symm
  · intro q i hi
    rw [hedges q] at hi
    exact h.valid_ids q i hi

theorem inv_of_effGet_eq {t t' : Tensor} (h : Inv t)
    (hget : ∀ q, t'.effGet q = t.effGet q) (hme : t'.me = t.me) (hmt : t'.mt = t.mt)
    (hmc : t'.multiCount = t.multiCount) (hsub : t'.dm ⊆ t'.m.dom)
    (hdisj : Disjoint t'.dp.dom t'.dm)
    (hcc : ∀ q ∈ t'.dp.dom, t'.m.get q ≠ some (t'.dp.val q))
    (hrange : ∀ q ∈ t'.m.dom ∪ t'.dp.dom, q.1 < t'.nrows ∧ q.2 < t'.ncols) : Inv t' := by
  have hdom : t'.effDom = t.effDom := by
    ext q; rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, hget q]
  exact { invCore_of_effGet_eq h.toInvCore hget hme hmc hsub hdisj hcc hrange with
          mt_eq := by intro q; rw [hdom, hmt]; exact h.mt_eq q }


end Tensor

end FalkorDB
