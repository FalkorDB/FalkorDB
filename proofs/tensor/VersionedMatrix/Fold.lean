/-
# `flush`, `resize`, `transpose`, and the fold entry points

`flush` folds a delta into the committed base once the policy says the rewrite
earns itself:

```rust
let fold_dp = self.dp.take_fold();
let fold_dm = self.dm.take_fold();
match (fold_dp, fold_dm) {
    (true,  true)  => new_m.element_wise_add(Some(&self.dm), Some(&self.m),
                                             Some(&*self.dp), Some(Descriptor::RC)),
    (true,  false) => new_m.element_wise_add(None, Some(&self.m), Some(&*self.dp), None),
    (false, true)  => new_m.select(&self.dm, &self.m),
    (false, false) => unreachable!(),
}
```

Everything here is proved for an **arbitrary** fold decision, never for the policy
that produces one — see `Model.lean` on why, and
`eff_flush_decision_irrelevant` for the statement that makes it worth it.

`resize`'s grow path is the interesting one: unlike the tensor's, the `bool`
version *folds* while it grows, merging `(m ∖ dm) ∪ dp` into a fresh base. That
merge is a two-way stream that assumes `dp ∩ m = ∅` — `eff_resize` is where that
assumption is discharged, and it is exactly why the `u64` tensor, which permits
shadowing, needs a different grow path.
-/
import VersionedMatrix.Count

namespace FalkorDB
namespace VersionedMatrix

variable {v : VersionedMatrix}

/-! ## Folding `dp` into `m` -/

@[simp] theorem foldDp_dm : (foldDp v).dm = v.dm := rfl
@[simp] theorem foldDm_dp : (foldDm v).dp = v.dp := rfl

/-- Folding the pending adds in leaves the effective set alone. Uses
`dp ∩ dm = ∅`: were a pending add also tombstoned, merging it into the base would
hand it to `dm` to delete. -/
theorem eff_foldDp (h : Inv v) : eff (foldDp v) = eff v := by
  have hd : ∀ q, q ∈ v.dp → q ∉ v.dm := fun _ hq => Finset.disjoint_left.mp h.dp_disj_dm hq
  ext q
  simp only [eff, foldDp, Finset.union_empty, Finset.mem_sdiff, Finset.mem_union]
  constructor
  · rintro ⟨hqm | hqdp, hq⟩
    · exact Or.inl ⟨hqm, hq⟩
    · exact Or.inr hqdp
  · rintro (⟨hqm, hq⟩ | hqdp)
    · exact ⟨Or.inl hqm, hq⟩
    · exact ⟨Or.inr hqdp, hd q hqdp⟩

theorem inv_foldDp (h : Inv v) : Inv (foldDp v) where
  dp_disj_m := by simp [foldDp]
  dm_sub_m := fun q hq => Finset.mem_union_left _ (h.dm_sub_m hq)
  in_range := by
    intro q hq
    simp only [foldDp, Finset.union_empty] at hq
    exact h.in_range q hq

/-! ## Folding `dm` out of `m` -/

/-- Applying the tombstones leaves the effective set alone. Uses `dp ∩ m = ∅`
only through `Inv`; the `dm` side is unconditional. -/
theorem eff_foldDm : eff (foldDm v) = eff v := by
  simp [eff, foldDm]

theorem inv_foldDm (h : Inv v) : Inv (foldDm v) where
  dp_disj_m := Finset.disjoint_left.mpr fun q hq hqm =>
    Finset.disjoint_left.mp h.dp_disj_m hq (Finset.mem_sdiff.mp hqm).1
  dm_sub_m := by simp [foldDm]
  in_range := by
    intro q hq
    refine h.in_range q ?_
    rcases Finset.mem_union.mp hq with hq' | hq'
    · exact Finset.mem_union_left _ (Finset.mem_sdiff.mp hq').1
    · exact Finset.mem_union_right _ hq'

/-! ## `flush` -/

/-- **`flush` is invisible to readers**, whichever layers it folds. -/
theorem eff_flush (h : Inv v) (fdp fdm : Bool) : eff (flush v fdp fdm) = eff v := by
  cases fdp <;> cases fdm
  · rfl
  · exact eff_foldDm
  · exact eff_foldDp h
  · exact eff_foldDm.trans (eff_foldDp h)

/-- **`flush` preserves every invariant.** -/
theorem inv_flush (h : Inv v) (fdp fdm : Bool) : Inv (flush v fdp fdm) := by
  cases fdp <;> cases fdm
  · exact h
  · exact inv_foldDm h
  · exact inv_foldDp h
  · exact inv_foldDm (inv_foldDp h)

/-- …and so leaves `nvals` alone, even though all three of its inputs moved. -/
theorem nvals_flush (h : Inv v) (fdp fdm : Bool) : nvals (flush v fdp fdm) = nvals v := by
  rw [nvals_eq_card (inv_flush h fdp fdm), nvals_eq_card h, eff_flush h fdp fdm]

/-- **The fold decision is denotationally invisible.** Any two decisions leave the
same effective set, so the whole policy — `WRITE_FOLD_K`, `READ_FOLD_K`,
`MIN_FOLD_DELTA`, the `delta_dominates_base` escape hatch, and the approximate
counters they are read off — is a throughput concern that cannot affect what the
matrix denotes. Retuning any of it leaves this development standing. -/
theorem eff_flush_decision_irrelevant (h : Inv v) (a b c d : Bool) :
    eff (flush v a b) = eff (flush v c d) := by
  rw [eff_flush h a b, eff_flush h c d]

@[simp] theorem flush_nrows (fdp fdm : Bool) : (flush v fdp fdm).nrows = v.nrows := by
  cases fdp <;> cases fdm <;> rfl

@[simp] theorem flush_ncols (fdp fdm : Bool) : (flush v fdp fdm).ncols = v.ncols := by
  cases fdp <;> cases fdm <;> rfl

/-! ### The entry-point fold is absorbed

`set`, `remove`, `set_all` and `remove_mask` all open with `self.flush()`, so the
state their specs describe is a possibly-just-folded matrix. The specs are stated
over an arbitrary `Inv` matrix and phrased in `eff`, and a fold lands back in that
class with the same `eff` and the same bounds — so they transfer. Rather than
assert that, these check it. -/

theorem inBounds_flush {p : Coord} (fdp fdm : Bool) (hb : InBounds v p) :
    InBounds (flush v fdp fdm) p := by
  unfold InBounds at hb ⊢
  simpa using hb

theorem eff_set_after_flush (h : Inv v) {p : Coord} (fdp fdm : Bool) :
    eff (set (flush v fdp fdm) p) = insert p (eff v) := by
  rw [eff_set p, eff_flush h fdp fdm]

theorem eff_remove_after_flush (h : Inv v) {p : Coord} (fdp fdm : Bool) :
    eff (remove (flush v fdp fdm) p) = (eff v).erase p := by
  rw [eff_remove (inv_flush h fdp fdm) p, eff_flush h fdp fdm]

theorem eff_removeMask_after_flush (h : Inv v) (mask : Finset Coord) (fdp fdm : Bool) :
    eff (removeMask (flush v fdp fdm) mask) = eff v \ mask := by
  rw [eff_removeMask mask, eff_flush h fdp fdm]

theorem eff_setAll_after_flush {l : List Coord} (h : Inv v) (hb : ∀ p ∈ l, InBounds v p)
    (fdp fdm : Bool) : eff (setAll (flush v fdp fdm) l) = eff v ∪ l.toFinset := by
  rw [eff_setAll (inv_flush h fdp fdm) (fun p hp => inBounds_flush fdp fdm (hb p hp)),
    eff_flush h fdp fdm]

/-! ## `resize` -/

/-- **The grow-and-fold rebuild preserves the effective set.** This is the
theorem behind the streamed two-way merge in `VersionedMatrix<bool>::resize`:
because `dp ∩ m = ∅`, emitting `(m ∖ dm)` and `dp` in order yields their union
with nothing counted twice. A `u64` layer, where `dp` may shadow `m`, breaks the
premise — which is why `Tensor::resize` grows each layer separately instead. -/
@[simp] theorem eff_resize (nr nc : Nat) : eff (resize v nr nc) = eff v := by
  simp [resize, eff]

/-- **`resize` preserves the invariants**, and lands with both deltas empty. -/
theorem inv_resize (h : Inv v) {nr nc : Nat} (hr : v.nrows ≤ nr) (hc : v.ncols ≤ nc) :
    Inv (resize v nr nc) where
  dp_disj_m := by simp [resize]
  dm_sub_m := by simp [resize]
  in_range := by
    intro q hq
    simp only [resize, Finset.union_empty] at hq
    have hin := h.in_range q (by
      rcases mem_eff.mp hq with ⟨hqm, _⟩ | hqdp
      · exact Finset.mem_union_left _ hqm
      · exact Finset.mem_union_right _ hqdp)
    exact ⟨lt_of_lt_of_le hin.1 hr, lt_of_lt_of_le hin.2 hc⟩

@[simp] theorem resize_dp (nr nc : Nat) : (resize v nr nc).dp = ∅ := rfl
@[simp] theorem resize_dm (nr nc : Nat) : (resize v nr nc).dm = ∅ := rfl

/-- With both deltas already empty the Rust skips the merge and only `grown`s the
base — the same function on that input, which is what licenses the fast path. -/
theorem resize_of_deltas_empty (hdp : v.dp = ∅) (hdm : v.dm = ∅) (nr nc : Nat) :
    (resize v nr nc).m = v.m := by
  simp [resize, eff, hdp, hdm]

/-! ## `transpose` -/

theorem eff_transpose : eff (transpose v) = (eff v).image Prod.swap := by
  ext q
  simp only [transpose, mem_eff, Finset.mem_image]
  constructor
  · rintro (⟨⟨r, hr, rfl⟩, hq⟩ | ⟨r, hr, rfl⟩)
    · exact ⟨r, Or.inl ⟨hr, fun hc => hq ⟨r, hc, rfl⟩⟩, rfl⟩
    · exact ⟨r, Or.inr hr, rfl⟩
  · rintro ⟨r, (⟨hrm, hr⟩ | hrdp), rfl⟩
    · refine Or.inl ⟨⟨r, hrm, rfl⟩, ?_⟩
      rintro ⟨s, hs, hsr⟩
      exact hr (by rwa [← Prod.swap_injective hsr])
    · exact Or.inr ⟨r, hrdp, rfl⟩

theorem inv_transpose (h : Inv v) : Inv (transpose v) where
  dp_disj_m := by
    refine Finset.disjoint_left.mpr ?_
    rintro q hq hqm
    obtain ⟨r, hr, rfl⟩ := Finset.mem_image.mp hq
    obtain ⟨s, hs, hsr⟩ := Finset.mem_image.mp hqm
    exact Finset.disjoint_left.mp h.dp_disj_m hr (by rwa [← Prod.swap_injective hsr])
  dm_sub_m := by
    intro q hq
    obtain ⟨r, hr, rfl⟩ := Finset.mem_image.mp hq
    exact Finset.mem_image.mpr ⟨r, h.dm_sub_m hr, rfl⟩
  in_range := by
    intro q hq
    simp only [transpose] at hq
    rcases Finset.mem_union.mp hq with hq' | hq' <;>
      obtain ⟨r, hr, rfl⟩ := Finset.mem_image.mp hq'
    · have := h.in_range r (Finset.mem_union_left _ hr)
      unfold InBounds at this ⊢
      exact ⟨this.2, this.1⟩
    · have := h.in_range r (Finset.mem_union_right _ hr)
      unfold InBounds at this ⊢
      exact ⟨this.2, this.1⟩

end VersionedMatrix
end FalkorDB
