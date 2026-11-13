import CchRecomputeReview.Relax
import CchSupergraphReview

/-!
# Bridge: today's `Topology` meets every premise `relax` needs

`cch_supergraph_review` models `Topology::from_order` (`order.rs:195-212` @main) line by line. This module
reads its result as this project's `Topo` (`ofSG`) and proves:

* `bridge`: for every `fromOrder sk r2n = .ok t` on a mirrored skeleton, `ofSG t` satisfies `Valid` (P-rows),
  `Chordal` (P-chordal) and `DownOK` (P-transpose). It uses `fromOrder_spec`, `gameRow_sorted`, `up_lt`,
  `up_supp`, `topo_chordal`, and the fact that rows at or above `n` are empty (`up_past`, `down_past`).
* `rd1_slots` (RD1 discharged): the slots `relax` indexes are `x_slots_start + i` (`customize.rs:270, 275, 285`)
  and `y_slots_start + cursor` (`:279, 294`). Each is in range and holds the arc it means, and `arc_slot`
  returns the same slot. Distinct arcs have distinct slots, and every slot below `arc_count` is one arc's
  (`arc_ids`). So the boxed slices indexed by slot and the model's metric indexed by arc are the same data.
  `cursorStep_get`: the cursor lands on `z`, so `y_slots_start + cursor` is arc `(y, z)`'s slot.
* `relax_fromOrder`, `relax_build`: on the topology `Topology::from_order` or `Topology::build` returns,
  `relax` never panics, and its result is the full customization. `relax_build` discharges `Mirrored` through
  `nd.rs`'s symmetric-simple check (`ndChecks_symm`, `mirrored_of_symmetricSimple`), given `binary_search`'s std
  contract.
-/

namespace CchRecompute

open CchSG (fromOrder Mirrored)

variable {W : Type}

/-- Today's topology (`CchSG.Topo`, `order.rs:129-139`) as this project's `Topo`. -/
def ofSG (t : CchSG.Topo) : Topo := ⟨t.nodeCount, t.up, t.down⟩

section
variable {sk : NI.Cch.Csr} {r2n : List Nat} {t : CchSG.Topo}

theorem fromOrder_down (h : fromOrder sk r2n = .ok t) :
    (t.downOff, t.downNb) = CchSG.transpose t.upOff t.upNb := by
  unfold fromOrder at h
  cases hwf : sk.isWellFormed <;> simp [hwf] at h
  cases hinv : CchSG.invert r2n sk.n <;> simp [hinv] at h
  rw [← h]

theorem rowSlots_past {off : List Nat} {r : Nat} (h : off.length ≤ r + 1) : (CchSG.rowSlots off r).2 = 0 := by
  simp [CchSG.rowSlots, List.getD, List.getElem?_eq_none h]

theorem slice_past (nb off : List Nat) {r : Nat} (h : off.length ≤ r + 1) :
    CchSG.slice nb (CchSG.rowSlots off r) = [] := by
  simp [CchSG.slice, rowSlots_past h]

/-- Rows at or above `n` are empty: `upward.offsets` has `n + 1` entries. -/
theorem up_past (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) {r : Nat} (hr : sk.n ≤ r) : t.up r = [] := by
  obtain ⟨_, _, _, _, hoff, _, _⟩ := CchSG.fromOrder_spec h hm
  unfold CchSG.Topo.up CchSG.Topo.upRange
  exact slice_past _ _ (by rw [hoff, CchSG.offsOf_length]; omega)

/-- The same for `downward`, whose offsets `transpose` builds with `n + 1` entries. -/
theorem down_past (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) {r : Nat} (hr : sk.n ≤ r) :
    t.down r = [] := by
  obtain ⟨hok, _, _, _, hoff, hnb, _⟩ := CchSG.fromOrder_spec h hm
  have hs := CchSG.rankAdj_supp hok
  have ht := CchSG.transpose_spec (R := CchSG.gameRow sk.n (CchSG.rankAdj sk r2n t.n2r)) (n := sk.n)
    (fun l _ => CchSG.gameRow_sorted _ _ _) (fun l _ x hx => (CchSG.up_supp hs ((CchSG.mem_gameRow hs).1 hx)).2)
  have hd := fromOrder_down h
  rw [hoff, hnb, ht] at hd
  have hdo : t.downOff = CchSG.offsOf (CchSG.lowersK (CchSG.gameRow sk.n (CchSG.rankAdj sk r2n t.n2r)) sk.n) sk.n :=
    (Prod.mk.inj hd).1
  unfold CchSG.Topo.down
  exact slice_past _ _ (by rw [hdo, CchSG.offsOf_length]; omega)

/-- **The bridge.** Today's topology satisfies P-rows, P-chordal and P-transpose. -/
theorem bridge (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) :
    (ofSG t).Valid ∧ (ofSG t).Chordal ∧ (ofSG t).DownOK := by
  obtain ⟨hok, hn, hup, hdown, _, _, _⟩ := CchSG.fromOrder_spec h hm
  have hs := CchSG.rankAdj_supp hok
  have hrow : ∀ x y, y ∈ t.up x → x < sk.n ∧ CchSG.up (CchSG.rankAdj sk r2n t.n2r) x y = true := by
    intro x y hy
    by_cases hx : x < sk.n
    · rw [hup x hx] at hy; exact ⟨hx, (CchSG.mem_gameRow hs).1 hy⟩
    · rw [up_past h hm (by omega)] at hy; simp at hy
  have hV : (ofSG t).Valid := by
    refine ⟨fun x => ?_, fun x y hy => CchSG.up_lt (hrow x y hy).2, fun x y hy => ?_⟩
    · by_cases hx : x < sk.n
      · show (t.up x).Pairwise (· < ·)
        rw [hup x hx]; exact CchSG.gameRow_sorted _ _ _
      · show (t.up x).Pairwise (· < ·)
        rw [up_past h hm (by omega)]; exact List.Pairwise.nil
    · show y < t.nodeCount
      rw [hn]; exact (CchSG.up_supp hs (hrow x y hy).2).2
  refine ⟨hV, fun x y z hy hz hyz => ?_, fun r => ?_⟩
  · have hx := (hrow x y hy).1
    have hy' : y < sk.n := by have := hV.bound x y hy; simp only [ofSG] at this; omega
    exact (CchSG.topo_chordal h hm).1 x y z hx hy' hy hz hyz
  · show t.down r = (List.range t.nodeCount).filter (fun x => decide (r ∈ t.up x))
    rw [hn]
    by_cases hr : r < sk.n
    · rw [hdown r hr, CchSG.lowersK]
      apply List.filter_congr
      intro l hl
      rw [hup l (List.mem_range.1 hl)]
      simp
    · rw [down_past h hm (by omega)]
      symm
      apply List.filter_eq_nil_iff.2
      intro x _ hx
      have := (CchSG.up_supp hs (hrow x r (by simpa using hx)).2).2
      omega

/-! ## RD1: the slots `relax` indexes are the arcs' -/

theorem position_get : ∀ {l : List Nat} {z p : Nat}, position l z = some p → l[p]? = some z
  | [], _, _, h => by simp [position] at h
  | a :: t, z, p, h => by
    simp only [position] at h
    split at h
    · rename_i hz; cases h; simp [hz]
    · cases hq : position t z with
      | none => rw [hq] at h; simp at h
      | some q => rw [hq] at h; simp at h; subst h; simpa using position_get hq

/-- The cursor walk (`customize.rs:288-294`) lands on `z`: slot `y_slots_start + cursor` is arc `(y, z)`. -/
theorem cursorStep_get {row : List Nat} {c z c' : Nat} (h : cursorStep row c z = some c') : row[c']? = some z := by
  simp only [cursorStep] at h
  cases hp : position (row.drop c) z with
  | none => rw [hp] at h; simp at h
  | some p =>
    rw [hp] at h; simp at h; subst h
    have := position_get hp
    rwa [List.getElem?_drop] at this

/-- **RD1 discharged** (`arc_ids`). Row-major slots: the slot of row `lo`'s `i`-th arc is `up_range(lo).start + i`
(`x_slots_start + i`, and `y_slots_start + cursor` by `cursorStep_get`). It is below `arc_count`, holds that arc's
upper end, and is what `arc_slot` returns. Slots are injective on arcs and onto `0 .. arc_count`. -/
theorem rd1_slots (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) {bs : List Nat → Nat → Option Nat}
    (hbs : CchSG.BsIdx bs) :
    (∀ lo i hi, (t.up lo)[i]? = some hi →
      (t.upRange lo).1 + i < t.arcCount ∧ t.upNb[(t.upRange lo).1 + i]? = some hi ∧
      CchSG.arcSlot bs t lo hi = some ((t.upRange lo).1 + i)) ∧
    (∀ lo lo' i i', i < (t.up lo).length → i' < (t.up lo').length →
      (t.upRange lo).1 + i = (t.upRange lo').1 + i' → lo = lo' ∧ i = i') ∧
    (∀ k, k < t.arcCount → ∃ lo i, i < (t.up lo).length ∧ k = (t.upRange lo).1 + i) := by
  obtain ⟨_, _, hup, _, hoff, _, _⟩ := CchSG.fromOrder_spec h hm
  obtain ⟨hids, honto, hinj⟩ := CchSG.arc_ids h hm hbs
  have hstart : ∀ lo, (t.upRange lo).1 =
      (CchSG.offsOf (CchSG.gameRow sk.n (CchSG.rankAdj sk r2n t.n2r)) sk.n).getD lo 0 := fun lo => by
    simp [CchSG.Topo.upRange, CchSG.rowSlots, hoff]
  have hlo : ∀ lo, (t.up lo) ≠ [] → lo < sk.n := fun lo hne => by
    apply Classical.byContradiction; intro hc; exact hne (up_past h hm (by omega))
  refine ⟨fun lo i hi hget => ?_, fun lo lo' i i' hi hi' he => ?_, fun k hk => ?_⟩
  · have hl := hlo lo (fun hn => by rw [hn] at hget; simp at hget)
    rw [hup lo hl] at hget
    have hilen : i < (CchSG.gameRow sk.n (CchSG.rankAdj sk r2n t.n2r) lo).length :=
      (List.getElem?_eq_some_iff.1 hget).1
    obtain ⟨e1, e2⟩ := hids lo i hl hilen
    rw [hstart]
    refine ⟨?_, by rw [e1]; exact hget, e2 hi hget⟩
    have : t.upNb[(CchSG.offsOf (CchSG.gameRow sk.n (CchSG.rankAdj sk r2n t.n2r)) sk.n).getD lo 0 + i]? =
        some hi := by rw [e1]; exact hget
    exact (List.getElem?_eq_some_iff.1 this).1
  · have hl := hlo lo (fun hn => by rw [hn] at hi; simp at hi)
    have hl' := hlo lo' (fun hn => by rw [hn] at hi'; simp at hi')
    rw [hup lo hl] at hi
    rw [hup lo' hl'] at hi'
    rw [hstart, hstart] at he
    exact hinj lo lo' i i' hl hl' hi hi' he
  · obtain ⟨lo, i, hl, hi, hk'⟩ := honto k hk
    refine ⟨lo, i, by rw [hup lo hl]; exact hi, by rw [hstart]; exact hk'⟩

end

/-! ## `relax` on today's topology -/

/-- **`relax` on the topology `from_order` returns** never panics, and its result is the full customization:
no premise left but the mirrored skeleton. -/
theorem relax_fromOrder (o : Ops W) {sk : NI.Cch.Csr} {r2n : List Nat} {t : CchSG.Topo}
    (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) (S : Seeds W) :
    ∃ R, relax o (ofSG t) (seedM S) = some R ∧ IsFull o (ofSG t) S R :=
  let ⟨hV, hC, _⟩ := bridge h hm
  relax_some o hV hC S

/-- **`relax` on the topology `Topology::build` returns** (`order.rs:180-182` with `nd.rs:48-58`'s checks): the
skeleton is mirrored because `nd` checked it symmetric and simple. The only premise is `binary_search`'s std
contract. -/
theorem relax_build (o : Ops W) (bs : List Nat → Nat → Bool) (hbs : NI.Cch.BsSpec bs)
    (backend : NI.Cch.Csr → Except CchSG.OErr (List Nat)) {sk : NI.Cch.Csr} {t : CchSG.Topo}
    (h : CchSG.build (CchSG.ndChecks bs backend) sk = .ok t) (S : Seeds W) :
    ∃ R, relax o (ofSG t) (seedM S) = some R ∧ IsFull o (ofSG t) S R := by
  unfold CchSG.build at h
  cases ho : CchSG.ndChecks bs backend sk with
  | error e => simp [ho] at h
  | ok ord =>
    simp only [ho] at h
    exact relax_fromOrder o h (CchSG.mirrored_of_symmetricSimple bs hbs sk (CchSG.ndChecks_symm bs backend sk ord ho)) S

end CchRecompute
