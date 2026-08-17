/-
# `remove_all`

Two paths, both proved here.

**Fast path** (`¬has_multi_edge`): a handful of GraphBLAS ops on the whole mask —
`dm<mask> = mask ∩ m`, `dp &= ¬mask`, `mt.remove_mask(maskᵗ)`.  Every masked pair
becomes empty, every other pair is untouched, and all invariants survive.

**Slow path**: per edge.  A single-edge pair is deleted; a `MULTI` pair drops the
id from `me` and *demotes* when one id is left (cancelling the delta if the
survivor is the committed value).

A by-product of the invariants: the third branch of the `MULTI` case — "all ids
removed at once, the pair is gone" — is **unreachable** (`removeOne_survivor`).
A `MULTI` pair has ≥ 2 ids, so erasing one always leaves one behind.
-/
import Tensor.Add

namespace FalkorDB
namespace Tensor

variable {t : Tensor} {p : Pair} {id : Nat}

/-! ## The fast path -/

section Fast

variable {mask : Finset Pair}

@[simp] theorem removeFast_m : (removeFast t mask).m = t.m := rfl
@[simp] theorem removeFast_me : (removeFast t mask).me = t.me := rfl
@[simp] theorem removeFast_nrows : (removeFast t mask).nrows = t.nrows := rfl
@[simp] theorem removeFast_ncols : (removeFast t mask).ncols = t.ncols := rfl

theorem removeFast_mem_dm {q : Pair} :
    q ∈ (removeFast t mask).dm ↔ (q ∈ t.dm ∧ q ∉ mask) ∨ (q ∈ mask ∧ q ∈ t.m.dom) := by
  simp [removeFast]

/-- A masked pair is gone: `dp` no longer holds it and `dm` masks its committed
entry (if it had one). -/
theorem removeFast_effGet_mem (hq : p ∈ mask) : (removeFast t mask).effGet p = none := by
  have hdp : (removeFast t mask).dp.get p = none := by
    simp [removeFast, Layer.get, hq]
  by_cases hm : p ∈ t.m.dom
  · have hdm : p ∈ (removeFast t mask).dm := removeFast_mem_dm.mpr (Or.inr ⟨hq, hm⟩)
    simp [effGet, hdp, hdm]
  · have : (removeFast t mask).m.get p = none := Layer.get_eq_none.mpr (by simpa using hm)
    by_cases hdm : p ∈ (removeFast t mask).dm
    · simp [effGet, hdp, hdm]
    · rw [effGet_of_m hdp hdm, this]

theorem removeFast_effGet_not_mem (hq : p ∉ mask) :
    (removeFast t mask).effGet p = t.effGet p := by
  refine effGet_congr_at rfl (by simp [removeFast, Layer.get, Layer.removeAll, hq]) ?_
  rw [removeFast_mem_dm]
  constructor
  · rintro (⟨h, _⟩ | ⟨h, _⟩)
    · exact h
    · exact absurd h hq
  · intro h; exact Or.inl ⟨h, hq⟩

theorem removeFast_effDom : (removeFast t mask).effDom = t.effDom \ mask := by
  ext q
  by_cases hq : q ∈ mask
  · simp [mem_effDom_iff_isSome, removeFast_effGet_mem hq, hq]
  · rw [mem_effDom_iff_isSome, removeFast_effGet_not_mem hq, ← mem_effDom_iff_isSome]
    simp [hq]

/-- With no multi-edge pair anywhere, no pair's inline value is the sentinel. -/
theorem not_multi_of_no_me (h : Inv t) (hme : t.me = ∅) (q : Pair) : t.effGet q ≠ some MULTI := by
  intro hc
  have := h.multi_iff q hc
  rw [meRow, hme, meRowOf_empty] at this
  simp at this

/-- **Fast path, masked pairs**: every pair in the mask ends up with no edges. -/
theorem edgesAt_removeFast_mem (hq : p ∈ mask) : (removeFast t mask).edgesAt p = ∅ := by
  simp [edgesAt, removeFast_effGet_mem hq]

/-- **Fast path, other pairs**: untouched. -/
theorem edgesAt_removeFast_not_mem (hq : p ∉ mask) :
    (removeFast t mask).edgesAt p = t.edgesAt p :=
  edgesAt_eq_of_effGet_eq rfl (removeFast_effGet_not_mem hq)

theorem inv_removeFast (h : Inv t) (hme : t.me = ∅) : Inv (removeFast t mask) := by
  have hnm := not_multi_of_no_me h hme
  have hmp : t.multiPairs = ∅ := by
    apply Finset.eq_empty_of_forall_notMem
    intro q hq
    exact hnm q (Finset.mem_filter.mp hq).2
  refine { dm_sub_m := ?_, dp_disj_dm := ?_, cancel_clean := ?_, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, bounded := ?_, in_range := ?_,
           valid_ids := ?_,
           mt_eq := by
             intro q
             rw [removeFast_effDom, Finset.mem_sdiff, h.mt_eq q]
             simp [removeFast, Finset.mem_filter] }
  · intro q hq
    rcases removeFast_mem_dm.mp hq with ⟨hq', _⟩ | ⟨_, hq'⟩
    · exact h.dm_sub_m hq'
    · exact hq'
  · rw [Finset.disjoint_left]
    intro q hq hq'
    have hq1 : q ∈ t.dp.dom ∧ q ∉ mask := by simpa [removeFast] using hq
    rcases removeFast_mem_dm.mp hq' with ⟨hq2, _⟩ | ⟨hq2, _⟩
    · exact Finset.disjoint_left.mp h.dp_disj_dm hq1.1 hq2
    · exact hq1.2 hq2
  · intro q hq
    have hq1 : q ∈ t.dp.dom := by
      have : q ∈ t.dp.dom ∧ q ∉ mask := by simpa [removeFast] using hq
      exact this.1
    exact h.cancel_clean q hq1
  · intro q hq
    by_cases hqm : q ∈ mask
    · rw [removeFast_effGet_mem hqm] at hq; exact absurd hq (by simp)
    · rw [removeFast_effGet_not_mem hqm] at hq
      exact absurd hq (hnm q)
  · intro q _ _
    rw [meRow, removeFast_me, hme, meRowOf_empty]
  · intro x hx
    rw [removeFast_me, hme] at hx
    exact absurd hx (by simp)
  · intro q hq
    rw [removeFast_effDom] at hq
    exact h.bounded q (Finset.mem_sdiff.mp hq).1
  · intro q hq
    rcases Finset.mem_union.mp hq with hq' | hq'
    · exact h.in_range q (Finset.mem_union_left _ (by simpa using hq'))
    · have : q ∈ t.dp.dom ∧ q ∉ mask := by simpa [removeFast] using hq'
      exact h.in_range q (Finset.mem_union_right _ this.1)
  · intro q i hi
    by_cases hqm : q ∈ mask
    · rw [edgesAt_removeFast_mem hqm] at hi; exact absurd hi (by simp)
    · rw [edgesAt_removeFast_not_mem hqm] at hi; exact h.valid_ids q i hi

end Fast

/-! ## The slow path, one edge at a time -/

section Slow

theorem rowAfterErase_eq : rowAfterErase t p id = (t.meRow (key p)).erase id := by
  simp [rowAfterErase, meRow]

/-- **The "all ids removed at once" branch of `remove_all` is dead code**: a
`MULTI` pair holds at least two ids, so erasing one leaves a survivor. -/
theorem removeOne_survivor (h : Inv t) (hv : t.effGet p = some MULTI) :
    (rowAfterErase t p id).Nonempty := by
  have h2 := h.multi_iff p hv
  rw [rowAfterErase_eq]
  refine Finset.card_pos.mp ?_
  have hle : (t.meRow (key p)).card - 1 ≤ ((t.meRow (key p)).erase id).card :=
    Finset.pred_card_le_card_erase
  omega


/-! ### Deleting a pair -/

@[simp] theorem deletePair_m : (deletePair t p).m = t.m := rfl
@[simp] theorem deletePair_me : (deletePair t p).me = t.me := rfl
@[simp] theorem deletePair_nrows : (deletePair t p).nrows = t.nrows := rfl
@[simp] theorem deletePair_ncols : (deletePair t p).ncols = t.ncols := rfl
@[simp] theorem deletePair_mt : (deletePair t p).mt = t.mt.erase (p.2, p.1) := rfl

theorem deletePair_mem_dm {q : Pair} :
    q ∈ (deletePair t p).dm ↔ q ∈ t.dm ∨ (q = p ∧ p ∈ t.m.dom) := by
  unfold deletePair
  by_cases hm : p ∈ t.m.dom
  · simp [hm]; tauto
  · simp [hm]

theorem deletePair_effGet_self : (deletePair t p).effGet p = none := by
  have hdp : (deletePair t p).dp.get p = none := by simp [deletePair]
  by_cases hm : p ∈ t.m.dom
  · have hdm : p ∈ (deletePair t p).dm := deletePair_mem_dm.mpr (Or.inr ⟨rfl, hm⟩)
    simp [effGet, hdp, hdm]
  · have hmn : (deletePair t p).m.get p = none := Layer.get_eq_none.mpr (by simpa using hm)
    by_cases hdm : p ∈ (deletePair t p).dm
    · simp [effGet, hdp, hdm]
    · rw [effGet_of_m hdp hdm, hmn]

theorem deletePair_effGet_ne {q : Pair} (hq : q ≠ p) :
    (deletePair t p).effGet q = t.effGet q := by
  refine effGet_congr_at rfl (by simp [deletePair, hq]) ?_
  rw [deletePair_mem_dm]
  constructor
  · rintro (hd | ⟨hd, _⟩)
    · exact hd
    · exact absurd hd hq
  · exact Or.inl

theorem deletePair_effDom : (deletePair t p).effDom = t.effDom.erase p := by
  ext q
  by_cases hq : q = p
  · subst hq; simp [mem_effDom_iff_isSome, deletePair_effGet_self]
  · rw [Finset.mem_erase, mem_effDom_iff_isSome, deletePair_effGet_ne hq,
      ← mem_effDom_iff_isSome]
    simp [hq]

@[simp] theorem edgesAt_deletePair_self : (deletePair t p).edgesAt p = ∅ := by
  simp [edgesAt, deletePair_effGet_self]

theorem edgesAt_deletePair_ne {q : Pair} (hq : q ≠ p) :
    (deletePair t p).edgesAt q = t.edgesAt q :=
  edgesAt_eq_of_effGet_eq rfl (deletePair_effGet_ne hq)

/-- `mt` mirrors the effective structure after a single-pair deletion. -/
private theorem mt_eq_erase (h : ∀ q : Pair, q ∈ t.effDom ↔ (q.2, q.1) ∈ t.mt) :
    ∀ q : Pair, q ∈ (deletePair t p).effDom ↔ (q.2, q.1) ∈ (deletePair t p).mt := by
  intro q
  rw [deletePair_effDom, deletePair_mt, Finset.mem_erase, Finset.mem_erase, h q, ne_eq, ne_eq,
    swap_eq_iff]

theorem inv_deletePair (h : Inv t) (hrow : t.meRow (key p) = ∅)
    (hnm : t.effGet p ≠ some MULTI) : Inv (deletePair t p) := by
  have hkey : ∀ x ∈ t.me, x.1 ≠ key p := by
    intro x hx hc
    have : x.2 ∈ t.meRow (key p) := mem_meRow.mpr (by rw [← hc]; exact hx)
    rw [hrow] at this
    exact absurd this (by simp)
  refine { dm_sub_m := ?_, dp_disj_dm := ?_, cancel_clean := ?_, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, bounded := ?_, in_range := ?_,
           mt_eq := mt_eq_erase h.mt_eq, valid_ids := ?_ }
  · intro q hq
    rcases deletePair_mem_dm.mp hq with hq' | ⟨rfl, hq'⟩
    · exact h.dm_sub_m hq'
    · exact hq'
  · rw [Finset.disjoint_left]
    intro q hq hq'
    have hq1 : q ∈ t.dp.dom.erase p := by simpa [deletePair] using hq
    rcases deletePair_mem_dm.mp hq' with hq2 | ⟨rfl, _⟩
    · exact Finset.disjoint_left.mp h.dp_disj_dm (Finset.mem_of_mem_erase hq1) hq2
    · exact (Finset.mem_erase.mp hq1).1 rfl
  · intro q hq
    have hq1 : q ∈ t.dp.dom := by
      have : q ∈ t.dp.dom.erase p := by simpa [deletePair] using hq
      exact Finset.mem_of_mem_erase this
    exact h.cancel_clean q hq1
  · intro q hq
    have hqp : q ≠ p := by
      rintro rfl
      rw [deletePair_effGet_self] at hq
      exact absurd hq (by simp)
    rw [deletePair_effGet_ne hqp] at hq
    rw [meRow, deletePair_me]
    exact h.multi_iff q hq
  · intro q hbq hq
    rw [meRow, deletePair_me]
    by_cases hqp : q = p
    · subst hqp; exact hrow
    · rw [deletePair_effGet_ne hqp] at hq
      exact h.row_empty q hbq hq
  · intro x hx
    rw [deletePair_me] at hx
    obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed x hx
    refine ⟨q, hbq, ?_, hqk⟩
    rw [deletePair_effDom, Finset.mem_erase]
    exact ⟨fun hc => hkey x hx (by rw [hqk, hc]), hqdom⟩
  · intro q hq
    rw [deletePair_effDom] at hq
    exact h.bounded q (Finset.mem_of_mem_erase hq)
  · intro q hq
    rcases Finset.mem_union.mp hq with hq' | hq'
    · exact h.in_range q (Finset.mem_union_left _ (by simpa using hq'))
    · have : q ∈ t.dp.dom.erase p := by simpa [deletePair] using hq'
      exact h.in_range q (Finset.mem_union_right _ (Finset.mem_of_mem_erase this))
  · intro q i hi
    by_cases hqp : q = p
    · subst hqp; rw [edgesAt_deletePair_self] at hi; exact absurd hi (by simp)
    · rw [edgesAt_deletePair_ne hqp] at hi; exact h.valid_ids q i hi

/-! ### The three reachable shapes of `removeOne` -/

/-- A `MULTI` pair is never `dm`-masked (`dp ∩ dm = ∅`). -/
theorem not_mem_dm_of_multi (h : Inv t) (hv : t.effGet p = some MULTI) : p ∉ t.dm := by
  cases hdp : t.dp.get p with
  | some w =>
    exact Finset.disjoint_left.mp h.dp_disj_dm (Layer.get_isSome.mp (by rw [hdp]; rfl))
  | none =>
    intro hdm
    rw [effGet, hdp] at hv
    simp [hdm] at hv

/-- Shape 1: still multi — only the `me` entry goes. -/
theorem removeOne_still_multi (hv : t.effGet p = some MULTI)
    (hcard : 2 ≤ (rowAfterErase t p id).card) :
    removeOne t id p = ({ t with me := t.me.erase (key p, id) }, none) := by
  simp [removeOne, hv, hcard]

/-- Shape 2a: demotion that *cancels* — the survivor is the committed value, so
both deltas are dropped and the pair returns to the clean state. -/
theorem removeOne_demote_cancel (hv : t.effGet p = some MULTI)
    (hcard : ¬ 2 ≤ (rowAfterErase t p id).card) {last : Nat}
    (hlast : (rowAfterErase t p id).min = some last) (hm : t.m.get p = some last) :
    removeOne t id p =
      ({ t with me := (t.me.erase (key p, id)).erase (key p, last),
                dp := t.dp.remove p }, none) := by
  simp [removeOne, hv, hcard, hlast, hm]

/-- Shape 2b: demotion that *shadows* — the survivor differs from the committed
value, so `dp` carries it live. -/
theorem removeOne_demote_shadow (hv : t.effGet p = some MULTI)
    (hcard : ¬ 2 ≤ (rowAfterErase t p id).card) {last : Nat}
    (hlast : (rowAfterErase t p id).min = some last) (hm : t.m.get p ≠ some last) :
    removeOne t id p =
      ({ t with me := (t.me.erase (key p, id)).erase (key p, last),
                dp := t.dp.set p last }, none) := by
  simp [removeOne, hv, hcard, hlast, hm]

/-- Shape 3: a single-edge pair whose id matches — the pair is deleted. -/
theorem removeOne_single (hv : t.effGet p = some id) (hM : id ≠ MULTI) :
    removeOne t id p = (deletePair t p, some p) := by
  simp [removeOne, hv, hM]

/-- Shape 4: nothing to remove (unknown id, or absent pair). -/
theorem removeOne_noop_of_not_mem (hid : id ∉ t.edgesAt p)
    (hv : t.effGet p ≠ some MULTI) : removeOne t id p = (t, none) := by
  cases hg : t.effGet p with
  | none => simp [removeOne, hg]
  | some v =>
    have hvM : v ≠ MULTI := by rintro rfl; exact hv hg
    have hne : v ≠ id := by
      rintro rfl
      exact hid (by simp [edgesAt, hg, hvM])
    simp [removeOne, hg, hvM, fun hc : v = id => hne hc]


/-! ### Demotion

Both demotion shapes (cancel / shadow) share everything except how `dp` ends up
holding the survivor, so the bulk of the reasoning is done once. -/

private theorem rowAfterErase_singleton
    (hcard : ¬ 2 ≤ (rowAfterErase t p id).card) {last : Nat}
    (hlast : (rowAfterErase t p id).min = some last) :
    rowAfterErase t p id = {last} := by
  have hmem : last ∈ rowAfterErase t p id := Finset.mem_of_min hlast
  refine Finset.eq_singleton_iff_unique_mem.mpr ⟨hmem, fun x hx => ?_⟩
  by_contra hxl
  have hsub : ({x, last} : Finset Nat) ⊆ rowAfterErase t p id := by
    intro y hy
    rcases Finset.mem_insert.mp hy with rfl | hy
    · exact hx
    · rw [Finset.mem_singleton.mp hy]; exact hmem
  have h2 : 2 ≤ (rowAfterErase t p id).card := by
    have hc := Finset.card_le_card hsub
    rw [Finset.card_insert_of_notMem (by simpa using hxl), Finset.card_singleton] at hc
    omega
  exact hcard h2

/-- The part of demotion that both shapes share. -/
private theorem demote_core (h : Inv t) (hbp : Bounded p) (hv : t.effGet p = some MULTI)
    {last : Nat} (hrow : rowAfterErase t p id = {last}) {t' : Tensor}
    (hm : t'.m = t.m) (hdm : t'.dm = t.dm) (hmt : t'.mt = t.mt)
    (hme : t'.me = (t.me.erase (key p, id)).erase (key p, last))
    (hnr : t'.nrows = t.nrows) (hnc : t'.ncols = t.ncols)
    (hself : t'.effGet p = some last) (hne : ∀ q, q ≠ p → t'.effGet q = t.effGet q)
    (hdpdom : t'.dp.dom ⊆ insert p t.dp.dom) (hdisj : Disjoint t'.dp.dom t'.dm)
    (hcc : ∀ q ∈ t'.dp.dom, t'.m.get q ≠ some (t'.dp.val q)) :
    Inv t' ∧ t'.edgesAt p = (t.edgesAt p).erase id ∧
      ∀ q, q ≠ p → t'.edgesAt q = t.edgesAt q := by
  have hpdom : p ∈ t.effDom := mem_effDom_iff_isSome.mpr (by rw [hv]; rfl)
  have hedges : t.edgesAt p = t.meRow (key p) := edgesAt_of_multi hv
  have hrow_erase : (t.meRow (key p)).erase id = {last} := by rw [← rowAfterErase_eq, hrow]
  have hlast_mem : last ∈ t.edgesAt p := by
    rw [hedges]
    exact Finset.mem_of_mem_erase (s := t.meRow (key p)) (a := id) (by rw [hrow_erase]; simp)
  have hlastM : last ≠ MULTI := (h.valid_ids p last hlast_mem).ne_multi
  have hrow' : t'.meRow (key p) = ∅ := by
    rw [meRow, hme, meRowOf_erase_self, meRowOf_erase_self, ← meRow, hrow_erase]
    simp
  have hrowne : ∀ q : Pair, key q ≠ key p → t'.meRow (key q) = t.meRow (key q) := by
    intro q hkq
    rw [meRow, hme, meRowOf_erase_ne hkq, meRowOf_erase_ne hkq, ← meRow]
  have hdom : t'.effDom = t.effDom :=
    effDom_eq_of_effGet_of_mem hpdom (by rw [hself]; rfl) hne
  have hedges_p : t'.edgesAt p = (t.edgesAt p).erase id := by
    rw [edgesAt_of_single hself hlastM, hedges, hrow_erase]
  have hedges_ne : ∀ q, q ≠ p → t'.edgesAt q = t.edgesAt q := by
    intro q hq
    cases hgq : t.effGet q with
    | none => rw [edgesAt_of_none (by rw [hne q hq, hgq]), edgesAt_of_none hgq]
    | some w =>
      by_cases hMw : w = MULTI
      · subst hMw
        have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by rw [hgq]; rfl))
        rw [edgesAt_of_multi (by rw [hne q hq, hgq]), edgesAt_of_multi hgq,
          hrowne q (key_ne hbq hbp hq)]
      · rw [edgesAt_of_single (by rw [hne q hq, hgq]) hMw, edgesAt_of_single hgq hMw]
  refine ⟨?_, hedges_p, hedges_ne⟩
  refine { dm_sub_m := ?_, dp_disj_dm := hdisj, cancel_clean := hcc, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, bounded := ?_, in_range := ?_,
           valid_ids := ?_,
           mt_eq := by intro q; rw [hdom, hmt]; exact h.mt_eq q }
  · rw [hdm, hm]; exact h.dm_sub_m
  · intro q hq
    have hqp : q ≠ p := by
      rintro rfl
      rw [hself] at hq
      exact hlastM (Option.some_inj.mp hq)
    rw [hne q hqp] at hq
    have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by rw [hq]; rfl))
    rw [hrowne q (key_ne hbq hbp hqp)]
    exact h.multi_iff q hq
  · intro q hbq hq
    by_cases hqp : q = p
    · subst hqp; exact hrow'
    · rw [hne q hqp] at hq
      rw [hrowne q (key_ne hbq hbp hqp)]
      exact h.row_empty q hbq hq
  · intro x hx
    rw [hme] at hx
    have hx' : x ∈ t.me := Finset.mem_of_mem_erase (Finset.mem_of_mem_erase hx)
    obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed x hx'
    exact ⟨q, hbq, by rw [hdom]; exact hqdom, hqk⟩
  · rw [hdom]; exact h.bounded
  · intro q hq
    rw [hnr, hnc]
    rcases Finset.mem_union.mp hq with hq' | hq'
    · rw [hm] at hq'; exact h.in_range q (Finset.mem_union_left _ hq')
    · rcases Finset.mem_insert.mp (hdpdom hq') with rfl | hq''
      · rcases Finset.mem_union.mp hpdom with hq3 | hq3
        · exact h.in_range q (Finset.mem_union_left _ (Finset.mem_sdiff.mp hq3).1)
        · exact h.in_range q (Finset.mem_union_right _ hq3)
      · exact h.in_range q (Finset.mem_union_right _ hq'')
  · intro q i hi
    by_cases hqp : q = p
    · subst hqp
      rw [hedges_p] at hi
      exact h.valid_ids q i (Finset.mem_of_mem_erase hi)
    · rw [hedges_ne q hqp] at hi
      exact h.valid_ids q i hi

/-! ### `removeOne`, all shapes together -/

/-- Shape 1 in full: only the `me` entry goes; the pair stays `MULTI`. -/
private theorem still_multi_spec (h : Inv t) (hbp : Bounded p) (hv : t.effGet p = some MULTI)
    (hcard : 2 ≤ (rowAfterErase t p id).card) :
    Inv (removeOne t id p).1 ∧ (removeOne t id p).1.edgesAt p = (t.edgesAt p).erase id ∧
      ∀ q, q ≠ p → (removeOne t id p).1.edgesAt q = t.edgesAt q := by
  rw [removeOne_still_multi hv hcard]
  have hget : ∀ q, ({ t with me := t.me.erase (key p, id) } : Tensor).effGet q = t.effGet q :=
    fun _ => rfl
  have hrow : ({ t with me := t.me.erase (key p, id) } : Tensor).meRow (key p)
      = (t.meRow (key p)).erase id := by rw [meRow, meRowOf_erase_self, ← meRow]
  have hrowne : ∀ q : Pair, key q ≠ key p →
      ({ t with me := t.me.erase (key p, id) } : Tensor).meRow (key q) = t.meRow (key q) := by
    intro q hkq
    rw [meRow, meRowOf_erase_ne hkq, ← meRow]
  have hedges_ne : ∀ q, q ≠ p →
      ({ t with me := t.me.erase (key p, id) } : Tensor).edgesAt q = t.edgesAt q := by
    intro q hq
    cases hgq : t.effGet q with
    | none => rw [edgesAt_of_none (by rw [hget q, hgq]), edgesAt_of_none hgq]
    | some w =>
      by_cases hMw : w = MULTI
      · subst hMw
        have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by rw [hgq]; rfl))
        rw [edgesAt_of_multi (by rw [hget q, hgq]), edgesAt_of_multi hgq,
          hrowne q (key_ne hbq hbp hq)]
      · rw [edgesAt_of_single (by rw [hget q, hgq]) hMw, edgesAt_of_single hgq hMw]
  have hedges_p : ({ t with me := t.me.erase (key p, id) } : Tensor).edgesAt p
      = (t.edgesAt p).erase id := by
    rw [edgesAt_of_multi (by rw [hget p]; exact hv), hrow, edgesAt_of_multi hv]
  refine ⟨?_, hedges_p, hedges_ne⟩
  refine { dm_sub_m := h.dm_sub_m, dp_disj_dm := h.dp_disj_dm, cancel_clean := h.cancel_clean,
           multi_iff := ?_, row_empty := ?_, me_keyed := ?_, bounded := h.bounded,
           in_range := h.in_range, mt_eq := h.mt_eq, valid_ids := ?_ }
  · intro q hq
    rw [hget q] at hq
    by_cases hqp : q = p
    · subst hqp
      rw [hrow, ← rowAfterErase_eq]
      exact hcard
    · have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by rw [hq]; rfl))
      rw [hrowne q (key_ne hbq hbp hqp)]
      exact h.multi_iff q hq
  · intro q hbq hq
    rw [hget q] at hq
    have hqp : q ≠ p := by rintro rfl; exact hq hv
    rw [hrowne q (key_ne hbq hbp hqp)]
    exact h.row_empty q hbq hq
  · intro x hx
    exact h.me_keyed x (Finset.mem_of_mem_erase hx)
  · intro q i hi
    by_cases hqp : q = p
    · subst hqp
      rw [hedges_p] at hi
      exact h.valid_ids q i (Finset.mem_of_mem_erase hi)
    · rw [hedges_ne q hqp] at hi
      exact h.valid_ids q i hi

/-- **`remove_all` removes exactly the named edge from the named pair, and
preserves every invariant.** -/
theorem removeOne_spec (h : Inv t) (hbp : Bounded p) :
    Inv (removeOne t id p).1 ∧ (removeOne t id p).1.edgesAt p = (t.edgesAt p).erase id ∧
      ∀ q, q ≠ p → (removeOne t id p).1.edgesAt q = t.edgesAt q := by
  cases hg : t.effGet p with
  | none =>
    have h0 : removeOne t id p = (t, none) := by simp [removeOne, hg]
    rw [h0]
    exact ⟨h, by rw [edgesAt_of_none hg]; simp, fun _ _ => rfl⟩
  | some v =>
    by_cases hM : v = MULTI
    · subst hM
      by_cases hcard : 2 ≤ (rowAfterErase t p id).card
      · exact still_multi_spec h hbp hg hcard
      · obtain ⟨x, hx⟩ := removeOne_survivor (id := id) h hg
        obtain ⟨last, hlast⟩ := Finset.min_of_mem hx
        have hlast' : (rowAfterErase t p id).min = some last := hlast
        have hrow := rowAfterErase_singleton hcard hlast'
        have hpdm : p ∉ t.dm := not_mem_dm_of_multi h hg
        by_cases hmv : t.m.get p = some last
        · rw [removeOne_demote_cancel hg hcard hlast' hmv]
          refine demote_core h hbp hg hrow rfl rfl rfl rfl rfl rfl
            ((effGet_of_m (by simp) (by simpa using hpdm)).trans hmv)
            (fun q hq => effGet_congr_at rfl (by simp [hq]) (by simp)) ?_ ?_ ?_
          · exact fun q hq => Finset.mem_insert_of_mem (Finset.mem_of_mem_erase (by simpa using hq))
          · rw [Finset.disjoint_left]
            intro q hq hq'
            exact Finset.disjoint_left.mp h.dp_disj_dm
              (Finset.mem_of_mem_erase (by simpa using hq)) (by simpa using hq')
          · intro q hq
            exact h.cancel_clean q (Finset.mem_of_mem_erase (by simpa using hq))
        · rw [removeOne_demote_shadow hg hcard hlast' hmv]
          refine demote_core h hbp hg hrow rfl rfl rfl rfl rfl rfl
            (effGet_of_dp (by simp))
            (fun q hq => effGet_congr_at rfl (by simp [hq]) (by simp)) ?_ ?_ ?_
          · exact fun q hq => by simpa using hq
          · rw [Finset.disjoint_left]
            intro q hq hq'
            have hq'' : q ∈ insert p t.dp.dom := by simpa using hq
            rcases Finset.mem_insert.mp hq'' with rfl | hq3
            · exact hpdm (by simpa using hq')
            · exact Finset.disjoint_left.mp h.dp_disj_dm hq3 (by simpa using hq')
          · intro q hq
            by_cases hqp : q = p
            · subst hqp
              simpa [Layer.set] using hmv
            · have hq3 : q ∈ t.dp.dom := by
                have h4 : q ∈ insert p t.dp.dom := by simpa using hq
                rcases Finset.mem_insert.mp h4 with rfl | h5
                · exact absurd rfl hqp
                · exact h5
              simpa [Layer.set, hqp] using h.cancel_clean q hq3
    · by_cases hvid : v = id
      · subst hvid
        rw [removeOne_single hg hM]
        have hrow : t.meRow (key p) = ∅ := h.row_empty p hbp (by rw [hg]; simpa using hM)
        refine ⟨inv_deletePair h hrow (by rw [hg]; simpa using hM), ?_, fun q hq =>
          edgesAt_deletePair_ne hq⟩
        rw [edgesAt_deletePair_self, edgesAt_of_single hg hM]
        simp
      · have h0 : removeOne t id p = (t, none) := by
          simp [removeOne, hg, hM, fun hc : v = id => hvid hc]
        rw [h0]
        refine ⟨h, ?_, fun _ _ => rfl⟩
        rw [edgesAt_of_single hg hM,
          Finset.erase_eq_of_notMem (by simpa using fun hc => hvid hc.symm)]


/-- Which pair became empty, and when: `remove_all` reports a pair exactly when
that removal took its last edge. -/
theorem removeOne_emptied (h : Inv t) :
    (removeOne t id p).2 = some p ↔ id ∈ t.edgesAt p ∧ (t.edgesAt p).erase id = ∅ := by
  cases hg : t.effGet p with
  | none => simp [removeOne, hg]
  | some v =>
    by_cases hM : v = MULTI
    · subst hM
      by_cases hcard : 2 ≤ (rowAfterErase t p id).card
      · rw [removeOne_still_multi hg hcard]
        have hne : (t.edgesAt p).erase id ≠ ∅ := by
          rw [edgesAt_of_multi hg, ← rowAfterErase_eq]
          exact fun hc => by simp [hc] at hcard
        simp [hne]
      · obtain ⟨x, hx⟩ := removeOne_survivor (id := id) h hg
        obtain ⟨last, hlast⟩ := Finset.min_of_mem hx
        have hlast' : (rowAfterErase t p id).min = some last := hlast
        have hrow := rowAfterErase_singleton hcard hlast'
        have hne : (t.edgesAt p).erase id ≠ ∅ := by
          rw [edgesAt_of_multi hg, ← rowAfterErase_eq, hrow]
          simp
        by_cases hmv : t.m.get p = some last
        · rw [removeOne_demote_cancel hg hcard hlast' hmv]; simp [hne]
        · rw [removeOne_demote_shadow hg hcard hlast' hmv]; simp [hne]
    · by_cases hvid : v = id
      · subst hvid
        rw [removeOne_single hg hM, edgesAt_of_single hg hM]
        simp
      · have h0 : removeOne t id p = (t, none) := by
          simp [removeOne, hg, hM, fun hc : v = id => hvid hc]
        rw [h0, edgesAt_of_single hg hM]
        simp
        exact fun hc => absurd hc.symm hvid

/-- The pair `remove_all` may report is always the pair it was asked about. -/
theorem removeOne_snd (t : Tensor) (id : Nat) (p : Pair) :
    (removeOne t id p).2 = none ∨ (removeOne t id p).2 = some p := by
  cases hg : t.effGet p with
  | none => simp [removeOne, hg]
  | some v =>
    by_cases hM : v = MULTI
    · subst hM
      by_cases hcard : 2 ≤ (rowAfterErase t p id).card
      · simp [removeOne, hg, hcard]
      · by_cases hnone : (rowAfterErase t p id).min = none
        · simp [removeOne, hg, hcard, hnone]
        · obtain ⟨last, hlast⟩ := Option.ne_none_iff_exists'.mp hnone
          by_cases hmv : t.m.get p = some last <;> simp [removeOne, hg, hcard, hlast, hmv]
    · by_cases hvid : v = id
      · have hM' : ¬ (id = MULTI) := by rw [← hvid]; exact hM
        simp [removeOne, hg, hM', hvid]
      · simp [removeOne, hg, hM, hvid]

/-! ## The slow path over a whole list -/

/-- The ids the request removes from pair `q`. -/
def removedIds (rels : List (Nat × Pair)) (q : Pair) : Finset Nat :=
  ((rels.filter (fun r => r.2 = q)).map Prod.fst).toFinset

@[simp] theorem removedIds_nil {q : Pair} : removedIds [] q = ∅ := rfl

theorem removedIds_cons_self {r : Nat × Pair} {rels : List (Nat × Pair)} :
    removedIds (r :: rels) r.2 = insert r.1 (removedIds rels r.2) := by
  simp [removedIds]

theorem removedIds_cons_ne {r : Nat × Pair} {rels : List (Nat × Pair)} {q : Pair}
    (hq : q ≠ r.2) : removedIds (r :: rels) q = removedIds rels q := by
  simp [removedIds, Ne.symm hq]

@[simp] theorem removeSlow_nil : removeSlow t [] = (t, []) := rfl

theorem removeSlow_cons {r : Nat × Pair} {rels : List (Nat × Pair)} :
    removeSlow t (r :: rels) =
      ((removeSlow (removeOne t r.1 r.2).1 rels).1,
        (removeOne t r.1 r.2).2.toList ++ (removeSlow (removeOne t r.1 r.2).1 rels).2) := rfl

/-- **The slow path of `remove_all`**: invariants preserved, exactly the requested
edges removed, and every reported pair really is empty afterwards. -/
theorem removeSlow_spec {rels : List (Nat × Pair)} (h : Inv t)
    (hb : ∀ r ∈ rels, Bounded r.2) :
    Inv (removeSlow t rels).1 ∧
      (∀ q, (removeSlow t rels).1.edgesAt q = t.edgesAt q \ removedIds rels q) ∧
      ∀ q ∈ (removeSlow t rels).2, (removeSlow t rels).1.edgesAt q = ∅ := by
  induction rels generalizing t with
  | nil => exact ⟨h, fun q => by simp, fun q hq => absurd hq (by simp)⟩
  | cons r rest ih =>
    obtain ⟨h1, hedge, hempt⟩ := removeOne_spec (id := r.1) (p := r.2) h (hb r (List.mem_cons_self ..))
    obtain ⟨h2, hedge2, hempt2⟩ := ih h1 (fun r' hr' => hb r' (List.mem_cons_of_mem _ hr'))
    have hstep : ∀ q, (removeSlow t (r :: rest)).1.edgesAt q
        = t.edgesAt q \ removedIds (r :: rest) q := by
      intro q
      rw [removeSlow_cons, hedge2 q]
      by_cases hqr : q = r.2
      · subst hqr
        rw [hedge, removedIds_cons_self]
        ext i
        simp only [Finset.mem_sdiff, Finset.mem_erase, Finset.mem_insert]
        tauto
      · rw [hempt q hqr, removedIds_cons_ne hqr]
    refine ⟨h2, hstep, ?_⟩
    intro q hq
    rw [removeSlow_cons] at hq
    rcases List.mem_append.mp hq with hq' | hq'
    · rcases removeOne_snd t r.1 r.2 with h0 | h0
      · rw [h0] at hq'; simp at hq'
      · have hqr : q = r.2 := by rw [h0] at hq'; simpa using hq'
        subst hqr
        have hz := (removeOne_emptied (id := r.1) h).mp h0
        rw [removeSlow_cons, hedge2 r.2, hedge, hz.2]
        simp
    · rw [removeSlow_cons]
      exact hempt2 q hq'

/-! ## `remove_all` -/

/-- **`Tensor::remove_all`**: whichever path it takes, it removes exactly the
requested edges, keeps every invariant, and its returned list contains only pairs
that really are empty afterwards.

The precondition is the one the callers satisfy: the request names edges that
exist (they come from the graph).  It is what makes the fast path — which deletes
whole pairs without checking ids — correct. -/
theorem removeAll_spec {rels : List (Nat × Pair)} (h : Inv t)
    (hb : ∀ r ∈ rels, Bounded r.2) (hex : ∀ r ∈ rels, r.1 ∈ t.edgesAt r.2) :
    Inv (removeAll t rels).1 ∧
      (∀ q, (removeAll t rels).1.edgesAt q = t.edgesAt q \ removedIds rels q) ∧
      ∀ q ∈ (removeAll t rels).2, (removeAll t rels).1.edgesAt q = ∅ := by
  unfold removeAll
  split
  · rename_i hnil
    have : rels = [] := List.isEmpty_iff.mp hnil
    subst this
    exact ⟨h, fun q => by simp, fun q hq => absurd hq (by simp)⟩
  · split
    -- fast path
    · rename_i hnil hmulti
      have hme : t.me = ∅ := by
        by_contra hc
        exact hmulti (by simpa [hasMultiEdge] using Finset.nonempty_iff_ne_empty.mpr hc)
      have hmask : ∀ q : Pair, q ∈ (rels.map (fun r => r.2)).toFinset ↔ ∃ r ∈ rels, r.2 = q := by
        intro q; simp [List.mem_toFinset]
      refine ⟨inv_removeFast h hme, ?_, ?_⟩
      · intro q
        by_cases hq : q ∈ (rels.map (fun r => r.2)).toFinset
        · obtain ⟨r, hr, hrq⟩ := (hmask q).mp hq
          have hcard : (t.edgesAt q).card ≤ 1 :=
            edgesAt_card_le_one_of_not_multi (not_multi_of_no_me h hme q)
          have hmem : r.1 ∈ t.edgesAt q := hrq ▸ hex r hr
          have hsingle : t.edgesAt q = {r.1} := by
            refine Finset.eq_singleton_iff_unique_mem.mpr ⟨hmem, fun x hx => ?_⟩
            by_contra hxr
            have hsub : ({x, r.1} : Finset Nat) ⊆ t.edgesAt q := by
              intro y hy
              rcases Finset.mem_insert.mp hy with rfl | hy
              · exact hx
              · rw [Finset.mem_singleton.mp hy]; exact hmem
            have := Finset.card_le_card hsub
            rw [Finset.card_insert_of_notMem (by simpa using hxr), Finset.card_singleton] at this
            omega
          have hrem : r.1 ∈ removedIds rels q := by
            simp only [removedIds, List.mem_toFinset, List.mem_map, List.mem_filter]
            exact ⟨r, ⟨hr, by simp [hrq]⟩, rfl⟩
          rw [edgesAt_removeFast_mem hq, hsingle]
          exact (Finset.sdiff_eq_empty_iff_subset.mpr (Finset.singleton_subset_iff.mpr hrem)).symm
        · have hrem : removedIds rels q = ∅ := by
            apply Finset.eq_empty_of_forall_notMem
            intro i hi
            simp only [removedIds, List.mem_toFinset, List.mem_map, List.mem_filter] at hi
            obtain ⟨r, ⟨hr, hrq⟩, _⟩ := hi
            exact hq ((hmask q).mpr ⟨r, hr, by simpa using hrq⟩)
          rw [edgesAt_removeFast_not_mem hq, hrem]
          simp
      · intro q hq
        exact edgesAt_removeFast_mem (List.mem_toFinset.mpr hq)
    -- slow path
    · exact removeSlow_spec h hb

end Slow

end Tensor
end FalkorDB
