import CchTopologyReview.Fresh
import CchRecomputeReview
import FalkorCowBTree

/-!
# P-rows and P-transpose for `cch_recompute_review`, from this project's invariant

`toTopo t` is the redesign's topology as `cch_recompute_review` reads it (`CchRecompute.Topo`,
`CchRecomputeReview/Basic.lean:92-97`): the rank count and, per rank, the columns of the merged rows
`up_ids(r)` and `down_ids(r)`.

* `toTopo_valid`: `Inv t → (toTopo t).Valid` (**P-rows**, `Basic.lean:102-105`).
* `toTopo_downOK`: `Inv t → (toTopo t).DownOK` (**P-transpose**, `Basic.lean:111-112`).
* `lineage_valid_downOK`: both hold at every state of every lineage that starts from a fresh build.
* `lineage_scoped_agrees`: `CchRecompute.scoped_agrees` (`Step.lean:302`), instantiated at such a state with
  P-rows and P-transpose discharged. Its remaining premise **P-chordal** stays open after S2: this project
  models Davis–Hager's closure for any pop order but does not prove it reaches a chordal fixpoint.
  `CchRecompute.lemmaA` (`Main.lean:139`) takes `H'.Valid` and `H'.DownOK` the same way, which
  `lineage_valid_downOK` gives for `H' = toTopo t'`.
-/

namespace CchTopo.Bridge

open CchTopo

variable {α : Type}

/-- The redesign's topology as `cch_recompute_review` reads it. -/
def toTopo (t : Cch α) : CchRecompute.Topo :=
  ⟨t.nr, fun r => (upIds t r).map Prod.fst, fun r => (downIds t r).map Prod.fst⟩

/-- **P-rows.** -/
theorem toTopo_valid {t : Cch α} (h : Inv t) : (toTopo t).Valid := by
  refine ⟨fun x => ?_, fun x y hy => ?_, fun x y hy => ?_⟩
  · show ((upIds t x).map Prod.fst).Pairwise (· < ·)
    rw [List.pairwise_map]; exact upIds_strict h x
  · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hy
    exact ((p_rows h x).2 p hp).1
  · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hy
    exact ((p_rows h x).2 p hp).2

/-- `(c, ·)` is in a row exactly when `c` is among its columns. -/
theorem mem_cols (l : List RE) (c : Nat) : c ∈ l.map Prod.fst ↔ ∃ i, (c, i) ∈ l := by
  simp only [List.mem_map]
  constructor
  · rintro ⟨⟨a, i⟩, hp, rfl⟩; exact ⟨i, hp⟩
  · rintro ⟨i, hi⟩; exact ⟨(c, i), hi, rfl⟩

/-- **P-transpose.** `down(r)` is exactly the ranks `x < nr` with `r ∈ up(x)`, ascending. -/
theorem toTopo_downOK {t : Cch α} (h : Inv t) : (toTopo t).DownOK := by
  intro r
  show (downIds t r).map Prod.fst = (List.range t.nr).filter (fun x => decide (r ∈ (upIds t x).map Prod.fst))
  have hl : CowBTree.Sorted ((downIds t r).map Prod.fst) := by
    unfold CowBTree.Sorted; rw [List.pairwise_map]; exact downIds_strict h r
  have hr : CowBTree.Sorted ((List.range t.nr).filter (fun x => decide (r ∈ (upIds t x).map Prod.fst))) :=
    List.pairwise_lt_range.sublist List.filter_sublist
  refine CowBTree.sorted_ext hl hr (fun x => ?_)
  rw [List.mem_filter, List.mem_range, decide_eq_true_eq, mem_cols, mem_cols]
  constructor
  · rintro ⟨i, hi⟩
    have hu : (r, i) ∈ upIds t x := (i7_rows h x r i).2 hi
    have hx : x < t.nr := by
      have he : upE t ⟨x, r, i⟩ := (mem_upIds t x r i).1 hu
      obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 (h.2.2 _ he).1
      exact h.1.node_lt x y hy
    exact ⟨hx, i, hu⟩
  · rintro ⟨_, i, hi⟩
    exact ⟨i, (i7_rows h x r i).1 hi⟩

/-- **At every state of every lineage that starts from a fresh build**, `cch_recompute_review`'s P-rows and
    P-transpose hold. -/
theorem lineage_valid_downOK (inf : α) (fi : FreshIn) (down : Rows) (hok : FreshOk fi)
    (ht : TransposeSpec (freshRows 0 fi.pat) down) {t : Cch α} (hl : Lin inf (fresh inf fi down) t) :
    (toTopo t).Valid ∧ (toTopo t).DownOK :=
  have h := lineage_inv inf fi down hok ht hl
  ⟨toTopo_valid h, toTopo_downOK h⟩

/-- **`scoped_agrees` at a lineage state**: the scoped recompute (`run`, 03 §6b) agrees with a full
    customization on every arc, with P-rows and P-transpose discharged here. **P-chordal (`hC`) is still a
    premise**: it is open after S2. -/
theorem lineage_scoped_agrees (inf : α) (fi : FreshIn) (down : Rows) (hok : FreshOk fi)
    (ht : TransposeSpec (freshRows 0 fi.pat) down) {t : Cch α} (hl : Lin inf (fresh inf fi down) t)
    (hC : (toTopo t).Chordal)
    {W : Type} {o : CchRecompute.Ops W} {R : W → W → Prop} (hA : CchRecompute.Agree o R)
    {S : CchRecompute.Seeds W} {F M0 : CchRecompute.Metric W} (hF : CchRecompute.IsFull o (toTopo t) S F)
    {fr : List (Nat × Nat)} (hfr_arc : ∀ p ∈ fr, p.2 ∈ (toTopo t).up p.1) (hfr_nodup : fr.Nodup)
    (hCons : ∀ a b, b ∈ (toTopo t).up a → (a, b) ∉ fr →
      CchRecompute.ER R (M0 a b) (CchRecompute.recomp o (toTopo t) S M0 a b))
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : CchRecompute.PickSpec pick) {fuel : Nat}
    (hfuel : CchRecompute.arcCount (toTopo t) < fuel) :
    (CchRecompute.run o (toTopo t) S pick fuel (CchRecompute.init M0 fr)).heap = [] ∧
      ∀ lo hi, hi ∈ (toTopo t).up lo →
        CchRecompute.ER R ((CchRecompute.run o (toTopo t) S pick fuel (CchRecompute.init M0 fr)).M lo hi) (F lo hi) :=
  have hvd := lineage_valid_downOK inf fi down hok ht hl
  CchRecompute.scoped_agrees hvd.1 hC hvd.2 hA hF hfr_arc hfr_nodup hCons hpick hfuel

end CchTopo.Bridge
