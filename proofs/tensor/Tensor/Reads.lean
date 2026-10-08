/-
# `new`, `dup`/`clone`, `get`, `has_multi_edge`, `extract`, `structural_iter`, `resize`

The read-only operations and the two trivial state operations.  Each theorem is
the doc comment of the corresponding Rust function, made precise.
-/
import Tensor.Ops

namespace FalkorDB
namespace Tensor

variable {t : Tensor} {p : Pair}

/-! ## A pair is in the effective domain iff it has an edge -/

/-- The support of the denotation is exactly the effective forward pattern.
This is what makes `structural_iter` / `extract` / `mt` meaningful: they all
describe "the pairs that have at least one edge". -/
theorem edgesAt_nonempty_iff (h : Inv t) : (t.edgesAt p).Nonempty ↔ p ∈ t.effDom := by
  constructor
  · rintro ⟨i, hi⟩
    by_contra hp
    rw [edgesAt_eq_empty_of_not_mem hp] at hi
    simp at hi
  · intro hp
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (mem_effDom_iff_isSome.mp hp)
    by_cases hM : v = MULTI
    · subst hM
      have h2 := h.multi_iff p hv
      have : (t.meRow (key p)).Nonempty := Finset.card_pos.mp (by omega)
      simpa [edgesAt, hv] using this
    · simp [edgesAt, hv, hM]

theorem edgesAt_card_le_one_of_not_multi (h : t.effGet p ≠ some MULTI) :
    (t.edgesAt p).card ≤ 1 := by
  unfold edgesAt
  cases hv : t.effGet p with
  | none => simp
  | some v =>
    have : v ≠ MULTI := by rintro rfl; exact h hv
    simp [this]

/-! ## `Tensor::new` -/

/-- A fresh tensor satisfies every invariant. -/
theorem inv_new (nrows ncols : Nat) : Inv (new nrows ncols) where
  dm_sub_m := by simp [new]
  dp_disj_dm := by simp [new]
  cancel_clean := by simp [new]
  multi_iff := by intro p hp; simp [new, effGet, Layer.get] at hp
  row_empty := by intro p _; simp [new, meRow]
  me_keyed := by simp [new]
  in_range := by intro p hp; simp [new] at hp
  mt_eq := by intro p; simp [new, effDom]
  valid_ids := by intro p i hi; simp [new, edgesAt, effGet, Layer.get] at hi

/-- A fresh tensor stores no edges. -/
@[simp] theorem edgesAt_new (nrows ncols : Nat) : (new nrows ncols).edgesAt p = ∅ := by
  simp [new, edgesAt, effGet, Layer.get]

@[simp] theorem edgeCount_new (nrows ncols : Nat) : edgeCount (new nrows ncols) = 0 := by
  simp [new, edgeCount, Layer.nvals]

/-! ## `Tensor::dup` and `Clone` -/

/-- `dup` (a new MVCC version) and `clone` (a handle copy) change no observable
state: same layers, same counter, hence same denotation and invariants. -/
@[simp] theorem edgesAt_dup : (dup t).edgesAt p = t.edgesAt p := rfl

theorem inv_dup (h : Inv t) : Inv (dup t) := h

@[simp] theorem edgeCount_dup : edgeCount (dup t) = edgeCount t := rfl

/-! ## `Tensor::get` -/

/-- **`get` is correct**: the `EdgeIds` iterator yields exactly the pair's edge
ids, each once, in ascending order. -/
theorem getIds_eq_sort : getIds t p = (t.edgesAt p).sort (· ≤ ·) := by
  unfold getIds edgesAt
  cases hv : t.effGet p with
  | none => simp
  | some v =>
    by_cases hM : v = MULTI
    · simp [hM]
    · simp [hM]

/-- `get` yields the pair's edge ids and nothing else. -/
theorem mem_getIds {i : Nat} : i ∈ getIds t p ↔ i ∈ t.edgesAt p := by
  rw [getIds_eq_sort, Finset.mem_sort]

/-- `get` never repeats an id. -/
theorem getIds_nodup : (getIds t p).Nodup := by
  rw [getIds_eq_sort]; exact Finset.sort_nodup _ _

/-- "in ascending edge-id order", as the doc comment claims (strictly ascending,
since an id cannot repeat). -/
theorem getIds_pairwise_lt : (getIds t p).Pairwise (· < ·) := by
  rw [getIds_eq_sort]; exact (Finset.sortedLT_sort _).pairwise

/-- The single-edge case answers straight from the inline value — no `me` lookup. -/
theorem getIds_single {v : Nat} (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    getIds t p = [v] := by simp [getIds, hv, hM]

/-! ## `Tensor::has_multi_edge` -/

/-- `has_multi_edge` (`me` non-empty) really does detect a pair with two or more
edges — this is the invariant "all ids of a multi-edge pair live in `me`, and
`me` is empty otherwise". -/
theorem hasMultiEdge_iff (h : Inv t) :
    hasMultiEdge t = true ↔ ∃ p, 2 ≤ (t.edgesAt p).card := by
  simp only [hasMultiEdge, decide_eq_true_eq]
  constructor
  · rintro ⟨x, hx⟩
    obtain ⟨p, hdom, hk⟩ := h.me_keyed x hx
    refine ⟨p, ?_⟩
    have hrow : (t.meRow (key p)).Nonempty := ⟨x.2, mem_meRow.mpr (by rw [← hk]; exact hx)⟩
    have hmulti : t.effGet p = some MULTI := by
      by_contra hne
      exact absurd (h.row_empty p hne) (Finset.nonempty_iff_ne_empty.mp hrow)
    have := h.multi_iff p hmulti
    simpa [edgesAt, hmulti] using this
  · rintro ⟨p, hp⟩
    have hmulti : t.effGet p = some MULTI := by
      by_contra hne
      exact absurd hp (by have := edgesAt_card_le_one_of_not_multi hne; omega)
    have hp' : 2 ≤ (t.meRow (key p)).card := by simpa [edgesAt, hmulti] using hp
    obtain ⟨i, hi⟩ := Finset.card_pos.mp (show 0 < (t.meRow (key p)).card by omega)
    exact ⟨(key p, i), mem_meRow.mp hi⟩

/-! ## `Tensor::extract` and `Tensor::structural_iter` -/

/-- `extract` materialises exactly the effective forward pattern `(m ∖ dm) ∪ dp`
— the conditional `nvals > 0` guards in the Rust are pure optimisations. -/
theorem extract_eq_effDom : extract t = t.effDom := by
  have hdm : t.dm.card = 0 → t.dm = ∅ := fun h => Finset.card_eq_zero.mp h
  have hdp : t.dp.nvals = 0 → t.dp.dom = ∅ := fun h => Finset.card_eq_zero.mp h
  unfold extract effDom
  split
  · split
    · rfl
    · rename_i h2
      rw [hdp (by omega)]
      simp
  · rename_i h1
    rw [hdm (by omega)]
    split
    · simp
    · rename_i h2
      rw [hdp (by omega)]
      simp

/-- `extract` is the set of pairs that have at least one edge. -/
theorem mem_extract_iff (h : Inv t) : p ∈ extract t ↔ (t.edgesAt p).Nonempty := by
  rw [extract_eq_effDom, edgesAt_nonempty_iff h]

/-- `structural_iter` yields exactly the live pairs of the requested row range,
each once. -/
theorem mem_structuralIter_iff (h : Inv t) {a b : Nat} :
    p ∈ structuralIter t a b ↔ (t.edgesAt p).Nonempty ∧ a ≤ p.1 ∧ p.1 ≤ b := by
  simp only [structuralIter, Finset.mem_filter, edgesAt_nonempty_iff h]

/-! ## `Tensor::resize` -/

/-- `resize` only grows capacity: it keeps every entry, so the denotation is
unchanged and the invariants — including "every entry is in range" — survive. -/
theorem inv_resize (h : Inv t) {nr nc : Nat} (hr : t.nrows ≤ nr) (hc : t.ncols ≤ nc) :
    Inv (resize t nr nc) where
  dm_sub_m := h.dm_sub_m
  dp_disj_dm := h.dp_disj_dm
  cancel_clean := h.cancel_clean
  multi_iff := h.multi_iff
  row_empty := h.row_empty
  me_keyed := h.me_keyed
  in_range := by
    intro q hq
    simp only [resize] at hq ⊢
    obtain ⟨h1, h2⟩ := h.in_range q hq
    omega
  mt_eq := h.mt_eq
  valid_ids := h.valid_ids

@[simp] theorem edgesAt_resize {nr nc : Nat} : (resize t nr nc).edgesAt p = t.edgesAt p := rfl

@[simp] theorem edgeCount_resize {nr nc : Nat} : edgeCount (resize t nr nc) = edgeCount t := rfl

end Tensor
end FalkorDB
