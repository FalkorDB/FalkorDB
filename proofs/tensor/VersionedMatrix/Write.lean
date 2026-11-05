/-
# `set`, `remove`, `remove_mask`, `set_all`, `set_all_new`

Each write preserves the invariants and acts on the denotation as an insert, an
erase, or a set difference. The reads (`get`, `extract`) agree with the denotation.

The substantial results are the last two:

* **`setAllFast_eq_setAllSlow`** — on an empty `dm`, `set_all`'s batched arm and
  its per-entry `set` arm compute the same state. The Rust keeps two paths for
  speed and asserts they agree in a comment; this is that agreement.
* **`setAllNew_eq_setAll`** — `set_all_new`, which drops the per-entry base probe,
  agrees with `set_all` exactly when the caller's freshness guarantee holds. This
  is the `debug_assert!` in `set_all_inner`, discharged: it is the *only* thing
  standing between the unchecked path and a broken `dp ∩ m = ∅`.
-/
import VersionedMatrix.Ops

namespace FalkorDB
namespace VersionedMatrix

variable {v : VersionedMatrix}

/-! ## Reads agree with the denotation -/

/-- **`get` is the denotation**, pointwise: it answers `Some` exactly on the
effective set. Note it probes only the base and one delta, never both. -/
theorem get_isSome_iff_mem_eff (h : Inv v) (p : Coord) :
    (get v p).isSome ↔ p ∈ eff v := by
  unfold get
  by_cases hm : p ∈ v.m
  · by_cases hdm : p ∈ v.dm
    · simp [hm, hdm, h.not_mem_dp_of_mem_m hm]
    · simp [hm, hdm]
  · have hdm : p ∉ v.dm := fun hp => hm (h.dm_sub_m hp)
    by_cases hdp : p ∈ v.dp <;> simp [hm, hdm, hdp]

@[simp] theorem mem_extract (p : Coord) : p ∈ extract v ↔ p ∈ eff v := Iff.rfl

@[simp] theorem mem_iter {lo hi : Nat} (p : Coord) :
    p ∈ iter v lo hi ↔ p ∈ eff v ∧ lo ≤ p.1 ∧ p.1 ≤ hi := by
  simp [iter]

/-- A range covering every row yields the whole effective set — the shape the
`u64::MAX` callers use. -/
theorem iter_eq_eff (h : Inv v) {hi : Nat} (hhi : v.nrows ≤ hi + 1) :
    iter v 0 hi = eff v := by
  ext p
  simp only [mem_iter, Nat.zero_le, true_and, and_iff_left_iff_imp]
  intro hp
  have hin : p.1 < v.nrows := (h.in_range p (by
    rcases mem_eff.mp hp with ⟨hpm, _⟩ | hpdp
    · exact Finset.mem_union_left _ hpm
    · exact Finset.mem_union_right _ hpdp)).1
  omega

/-! ## Construction -/

@[simp] theorem eff_new (nrows ncols : Nat) : eff (new nrows ncols) = ∅ := by
  simp [eff, new]

theorem inv_new (nrows ncols : Nat) : Inv (new nrows ncols) where
  dp_disj_m := by simp [new]
  dm_sub_m := by simp [new]
  in_range := by simp [new]

@[simp] theorem eff_fromMatrix {s : Finset Coord} {nr nc : Nat} :
    eff (fromMatrix s nr nc) = s := by
  simp [eff, fromMatrix]

theorem inv_fromMatrix {s : Finset Coord} {nr nc : Nat}
    (hb : ∀ p ∈ s, p.1 < nr ∧ p.2 < nc) : Inv (fromMatrix s nr nc) where
  dp_disj_m := by simp [fromMatrix]
  dm_sub_m := by simp [fromMatrix]
  in_range := by
    intro p hp
    simp only [fromMatrix, Finset.union_empty] at hp
    exact hb p hp

@[simp] theorem eff_dup : eff (dup v) = eff v := rfl

theorem inv_dup (h : Inv v) : Inv (dup v) := h

/-! ## `set` -/

@[simp] theorem set_m {p : Coord} : (set v p).m = v.m := by
  unfold set; split <;> rfl

@[simp] theorem set_nrows {p : Coord} : (set v p).nrows = v.nrows := by
  unfold set; split <;> rfl

@[simp] theorem set_ncols {p : Coord} : (set v p).ncols = v.ncols := by
  unfold set; split <;> rfl

/-- **`set` adds exactly its pair.** Needs no invariant: whichever branch is
taken, the effective set gains `p` and nothing else. -/
theorem eff_set (p : Coord) : eff (set v p) = insert p (eff v) := by
  unfold set
  split
  · rename_i hm
    ext q
    by_cases hqp : q = p
    · subst hqp; simp [hm]
    · -- Off the written pair, erasing `p` from `dm` is invisible.
      have hdm : q ∈ v.dm.erase p ↔ q ∈ v.dm := by simp [Finset.mem_erase, hqp]
      simp [mem_eff, Finset.mem_insert, hqp, hdm]
  · rename_i hm
    ext q
    by_cases hqp : q = p
    · subst hqp; simp [hm]
    · simp [mem_eff, Finset.mem_insert, hqp]

/-- **`set` preserves the invariants.** -/
theorem inv_set (h : Inv v) {p : Coord} (hb : InBounds v p) : Inv (set v p) := by
  unfold set
  split
  · rename_i hm
    exact { h with
      dp_disj_m := h.dp_disj_m
      dm_sub_m := fun q hq => h.dm_sub_m (Finset.mem_of_mem_erase hq) }
  · rename_i hm
    refine { dp_disj_m := ?_, dm_sub_m := h.dm_sub_m, in_range := ?_ }
    · refine Finset.disjoint_left.mpr fun q hq hqm => ?_
      rcases Finset.mem_insert.mp hq with rfl | hq'
      · exact hm hqm
      · exact Finset.disjoint_left.mp h.dp_disj_m hq' hqm
    · intro q hq
      rcases Finset.mem_union.mp hq with hqm | hqdp
      · exact h.in_range q (Finset.mem_union_left _ hqm)
      · rcases Finset.mem_insert.mp hqdp with rfl | hq'
        · exact hb
        · exact h.in_range q (Finset.mem_union_right _ hq')

/-! ## `remove` -/

@[simp] theorem remove_m {p : Coord} : (remove v p).m = v.m := by
  unfold remove; split <;> rfl

/-- **`remove` deletes exactly its pair.** In the committed branch this is where
`dp ∩ m = ∅` earns its keep: without it, tombstoning `p` would leave a `dp` entry
still making `p` effective. -/
theorem eff_remove (h : Inv v) (p : Coord) : eff (remove v p) = (eff v).erase p := by
  unfold remove
  split
  · rename_i hm
    ext q
    by_cases hqp : q = p
    · subst hqp; simp [hm, h.not_mem_dp_of_mem_m hm]
    · have hdm : q ∈ insert p v.dm ↔ q ∈ v.dm := by simp [Finset.mem_insert, hqp]
      simp [mem_eff, Finset.mem_erase, hqp, hdm]
  · rename_i hm
    ext q
    by_cases hqp : q = p
    · subst hqp; simp [hm]
    · have hdp : q ∈ v.dp.erase p ↔ q ∈ v.dp := by simp [Finset.mem_erase, hqp]
      simp [mem_eff, Finset.mem_erase, hqp, hdp]

/-- **`remove` preserves the invariants.** -/
theorem inv_remove (h : Inv v) (p : Coord) : Inv (remove v p) := by
  unfold remove
  split
  · rename_i hm
    refine { dp_disj_m := h.dp_disj_m, dm_sub_m := ?_, in_range := h.in_range }
    intro q hq
    rcases Finset.mem_insert.mp hq with rfl | hq'
    · exact hm
    · exact h.dm_sub_m hq'
  · refine { dm_sub_m := h.dm_sub_m, dp_disj_m := ?_, in_range := ?_ }
    · exact Finset.disjoint_left.mpr fun q hq =>
        Finset.disjoint_left.mp h.dp_disj_m (Finset.mem_of_mem_erase hq)
    · intro q hq
      rcases Finset.mem_union.mp hq with hqm | hqdp
      · exact h.in_range q (Finset.mem_union_left _ hqm)
      · exact h.in_range q (Finset.mem_union_right _ (Finset.mem_of_mem_erase hqdp))

/-! ## `remove_mask` -/

/-- **`remove_mask` is a bulk `remove`**: exactly the masked pairs leave the
effective set, in two GraphBLAS ops rather than N. -/
theorem eff_removeMask (mask : Finset Coord) :
    eff (removeMask v mask) = eff v \ mask := by
  ext q
  by_cases hmask : q ∈ mask
  · -- Inside the mask: the `dm` assign tombstones it if committed, and the `dp`
    -- difference drops it if pending, so it cannot survive either way.
    simp [removeMask, mem_eff, Finset.mem_sdiff, Finset.mem_union, Finset.mem_inter, hmask]
  · simp [removeMask, mem_eff, Finset.mem_sdiff, Finset.mem_union, Finset.mem_inter, hmask]

/-- **`remove_mask` preserves the invariants.** The `dm` clause is where the
"masked assign without replace" shape matters: entries outside the mask survive,
and `mask ∩ m ⊆ m` covers the ones added. -/
theorem inv_removeMask (h : Inv v) (mask : Finset Coord) : Inv (removeMask v mask) := by
  refine { dp_disj_m := ?_, dm_sub_m := ?_, in_range := ?_ }
  · exact Finset.disjoint_left.mpr fun q hq =>
      Finset.disjoint_left.mp h.dp_disj_m (Finset.mem_sdiff.mp hq).1
  · intro q hq
    rcases Finset.mem_union.mp hq with hq' | hq'
    · exact h.dm_sub_m (Finset.mem_sdiff.mp hq').1
    · exact (Finset.mem_inter.mp hq').2
  · intro q hq
    rcases Finset.mem_union.mp hq with hqm | hqdp
    · exact h.in_range q (Finset.mem_union_left _ hqm)
    · exact h.in_range q (Finset.mem_union_right _ (Finset.mem_sdiff.mp hqdp).1)

/-! ### `remove_mask` versus a fold of `remove`

`Graph::delete_nodes` used to build a diagonal mask and hand it to `remove_mask`;
it now loops `remove` per deleted entity instead, because `remove_mask`'s
`element_wise_multiply` takes `m` as an operand and so costs `O(base)` however
small the delete is. The Rust argues the swap is safe in a comment, from
`dp ∩ m = ∅`. Here it is, as a theorem: both take exactly the masked pairs out of
the effective set. -/

theorem inv_foldl_remove (h : Inv v) (l : List Coord) : Inv (l.foldl remove v) := by
  induction l generalizing v with
  | nil => exact h
  | cons p l ih => exact ih (inv_remove h p)

theorem eff_foldl_remove (h : Inv v) (l : List Coord) :
    eff (l.foldl remove v) = eff v \ l.toFinset := by
  induction l generalizing v with
  | nil => simp
  | cons p l ih =>
    rw [List.foldl_cons, ih (inv_remove h p), eff_remove h p]
    ext q
    simp only [Finset.mem_sdiff, Finset.mem_erase, List.toFinset_cons, Finset.mem_insert]
    tauto

/-- **The substitution `delete_nodes` makes is denotationally exact**: masking the
pairs out in two bulk ops and erasing them one at a time land on the same
effective set. The `bool` invariant `dp ∩ m = ∅` is what makes it true —
`remove_mask`'s two effects (tombstone `mask ∩ m`, drop `mask ∩ dp`) can never
both apply to one pair, which is precisely the choice `remove` makes per entry. -/
theorem eff_removeMask_eq_foldl_remove (h : Inv v) (l : List Coord) :
    eff (removeMask v l.toFinset) = eff (l.foldl remove v) := by
  rw [eff_removeMask l.toFinset, eff_foldl_remove h]

/-! ## `set_all`: the two arms agree

`set_all_inner` checks `dm` emptiness once and then runs either a batched loop or
per-entry `set`. Nothing in the Rust forces those to coincide, so here it is
proved. Both preserve `dm = ∅` and neither touches `m`, which is what makes the
induction go through. -/

theorem setAllFast_dm {l : List Coord} : (setAllFast v l).dm = v.dm := by
  induction l generalizing v with
  | nil => rfl
  | cons p l ih => unfold setAllFast; simp only [List.foldl_cons]; split <;> exact ih

theorem setAllFast_m {l : List Coord} : (setAllFast v l).m = v.m := by
  induction l generalizing v with
  | nil => rfl
  | cons p l ih => unfold setAllFast; simp only [List.foldl_cons]; split <;> exact ih

/-- On an empty `dm`, `set` on a *committed* pair is a no-op: it erases from an
already-empty tombstone set. This is what makes the fast path's "skip it" and the
general path's "un-delete it" the same step. -/
theorem set_of_mem_m_of_dm_empty (hdm : v.dm = ∅) {p : Coord} (hm : p ∈ v.m) :
    set v p = v := by
  unfold set
  simp only [hm, if_true]
  have herase : v.dm.erase p = v.dm := by rw [hdm]; exact Finset.erase_empty p
  rw [herase]

/-- **The fast path is the general path**, whenever `dm` is empty — the condition
the Rust checks before taking it. -/
theorem setAllFast_eq_setAllSlow (hdm : v.dm = ∅) (l : List Coord) :
    setAllFast v l = setAllSlow v l := by
  induction l generalizing v with
  | nil => rfl
  | cons p l ih =>
    show setAllFast (if p ∈ v.m then v else { v with dp := insert p v.dp }) l
        = setAllSlow (set v p) l
    by_cases hm : p ∈ v.m
    · simp only [hm, if_true, set_of_mem_m_of_dm_empty hdm hm]
      exact ih hdm
    · rw [set, if_neg hm]
      simp only [hm, if_false]
      exact ih (by simpa using hdm)

/-- **`set_all_new` agrees with `set_all`** exactly under its documented
precondition. Skipping the base probe is safe *because* no entry is committed; the
`FreshEntries` hypothesis is the `debug_assert!` the Rust carries, and without it
the unchecked path would push a committed pair into `dp` and break
`dp ∩ m = ∅`. -/
theorem setAllNewFast_eq_setAllFast {l : List Coord} (hfresh : FreshEntries v l) :
    setAllNewFast v l = setAllFast v l := by
  induction l generalizing v with
  | nil => rfl
  | cons p l ih =>
    have hp : p ∉ v.m := hfresh p (List.mem_cons_self ..)
    show setAllNewFast { v with dp := insert p v.dp } l
        = setAllFast (if p ∈ v.m then v else { v with dp := insert p v.dp }) l
    simp only [hp, if_false]
    exact ih fun q hq => hfresh q (List.mem_cons_of_mem _ hq)

theorem setAllNew_eq_setAll {l : List Coord} (hfresh : FreshEntries v l) :
    setAllNew v l = setAll v l := by
  unfold setAllNew setAll
  split
  · exact setAllNewFast_eq_setAllFast hfresh
  · rfl

/-! ### What a batch does

With the arms identified, `set_all`'s specification is the one for repeated
`set`: the effective set gains exactly the batch. -/

theorem inv_setAllSlow {l : List Coord} (h : Inv v) (hb : ∀ p ∈ l, InBounds v p) :
    Inv (setAllSlow v l) := by
  induction l generalizing v with
  | nil => exact h
  | cons p l ih =>
    refine ih (inv_set h (hb p (List.mem_cons_self ..))) ?_
    intro q hq
    have := hb q (List.mem_cons_of_mem _ hq)
    unfold InBounds at this ⊢
    simpa using this

theorem eff_setAllSlow {l : List Coord} (h : Inv v) (hb : ∀ p ∈ l, InBounds v p) :
    eff (setAllSlow v l) = eff v ∪ l.toFinset := by
  induction l generalizing v with
  | nil => simp [setAllSlow]
  | cons p l ih =>
    have hbp := hb p (List.mem_cons_self ..)
    have hb' : ∀ q ∈ l, InBounds (set v p) q := by
      intro q hq
      have := hb q (List.mem_cons_of_mem _ hq)
      unfold InBounds at this ⊢
      simpa using this
    show eff (setAllSlow (set v p) l) = _
    rw [ih (inv_set h hbp) hb', eff_set p]
    ext q
    simp [Finset.mem_insert, or_comm, or_assoc]

/-- **`set_all` adds exactly its batch**, whichever arm it takes. -/
theorem eff_setAll {l : List Coord} (h : Inv v) (hb : ∀ p ∈ l, InBounds v p) :
    eff (setAll v l) = eff v ∪ l.toFinset := by
  unfold setAll
  split
  · rename_i hdm
    rw [setAllFast_eq_setAllSlow hdm]
    exact eff_setAllSlow h hb
  · exact eff_setAllSlow h hb

theorem inv_setAll {l : List Coord} (h : Inv v) (hb : ∀ p ∈ l, InBounds v p) :
    Inv (setAll v l) := by
  unfold setAll
  split
  · rename_i hdm
    rw [setAllFast_eq_setAllSlow hdm]
    exact inv_setAllSlow h hb
  · exact inv_setAllSlow h hb

/-- …and so does `set_all_new`, given its freshness guarantee. -/
theorem eff_setAllNew {l : List Coord} (h : Inv v) (hb : ∀ p ∈ l, InBounds v p)
    (hfresh : FreshEntries v l) : eff (setAllNew v l) = eff v ∪ l.toFinset := by
  rw [setAllNew_eq_setAll hfresh]; exact eff_setAll h hb

theorem inv_setAllNew {l : List Coord} (h : Inv v) (hb : ∀ p ∈ l, InBounds v p)
    (hfresh : FreshEntries v l) : Inv (setAllNew v l) := by
  rw [setAllNew_eq_setAll hfresh]; exact inv_setAll h hb

end VersionedMatrix
end FalkorDB
