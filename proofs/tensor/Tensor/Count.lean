/-
# `edge_count`

```rust
pub fn edge_count(&self) -> u64 {
    self.wait_fwd();
    let shadow = if self.dp.nvals() == 0 { 0 } else { self.dp.intersection_nvals(&self.m) };
    self.m.nvals() + self.dp.nvals() - self.dm.nvals() - shadow - self.multi_pairs()
        + self.me.nvals()
}
```

`multi_pairs()` derives the count from `me` — `nvals` when nothing is multi-edge,
the hyper vector count when `me` is assembled, an `O(multi)` walk only while its
deltas are live. The model matches: the count is `multiPairs.card`, computed from
`me`, and not a field. Before #2439 the Rust kept a `multi_count` cache and the
model carried a matching field with an `Inv` clause tying the two together; both
are gone, so nothing here has to be maintained across mutations.

Proved here:

* `edgeCount_eq_sum` — the formula really is the number of edges the tensor
  stores, i.e. `∑ over pairs, |edges at that pair|`;
* `edgeCount_no_underflow` — each `u64` subtraction in that expression stays
  non-negative, so evaluating it left-to-right in unsigned arithmetic cannot
  wrap.

The two ingredients are `effDom_card` (the effective `nvals`, from `dm ⊆ m` and
`dp ∩ dm = ∅`) and `me_card_eq_sum` (`me` is partitioned into one row per
`MULTI` pair — which needs `compound_key` injectivity).
-/
import Tensor.Reads

namespace FalkorDB
namespace Tensor

variable {t : Tensor}

/-! ## The effective `nvals` -/

/-- `|(m ∖ dm) ∪ dp| + |dm| + |dp ∩ m| = |m| + |dp|`, i.e. the Rust's
`m.nvals() + dp.nvals() - dm.nvals() - shadow`, in a subtraction-free form. -/
theorem effDom_card_add (h : Inv t) :
    t.effDom.card + t.dm.card + (t.dp.dom ∩ t.m.dom).card = t.m.dom.card + t.dp.dom.card := by
  have hsdiff : (t.m.dom \ t.dm).card = t.m.dom.card - t.dm.card := by
    rw [Finset.card_sdiff, Finset.inter_eq_left.mpr h.dm_sub_m]
  have hdmle : t.dm.card ≤ t.m.dom.card := Finset.card_le_card h.dm_sub_m
  have hinter : (t.m.dom \ t.dm) ∩ t.dp.dom = t.m.dom ∩ t.dp.dom := by
    ext q
    simp only [Finset.mem_inter, Finset.mem_sdiff]
    constructor
    · rintro ⟨⟨h1, _⟩, h2⟩; exact ⟨h1, h2⟩
    · rintro ⟨h1, h2⟩
      exact ⟨⟨h1, Finset.disjoint_left.mp h.dp_disj_dm h2⟩, h2⟩
  have hunion := Finset.card_union_add_card_inter (t.m.dom \ t.dm) t.dp.dom
  rw [hinter, Finset.inter_comm t.m.dom t.dp.dom] at hunion
  simp only [effDom]
  omega

/-! ## `me` is one row per `MULTI` pair -/

/-- The fibre of `me` over a key is its row (the `snd` projection is injective on
a fibre). -/
theorem card_filter_eq_card_meRow {k : Nat} :
    (t.me.filter (fun x => x.1 = k)).card = (meRowOf t.me k).card := by
  refine (Finset.card_image_of_injOn ?_).symm
  intro x hx y hy hxy
  have hx1 := (Finset.mem_filter.mp hx).2
  have hy1 := (Finset.mem_filter.mp hy).2
  exact Prod.ext (by rw [hx1, hy1]) hxy

/-- Every `me` key is the key of a `MULTI` pair, and conversely. -/
theorem me_keys_eq (h : Inv t) : t.me.image Prod.fst = t.multiPairs.image key := by
  ext k
  simp only [Finset.mem_image]
  constructor
  · rintro ⟨x, hx, rfl⟩
    obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed x hx
    refine ⟨q, ?_, hqk.symm⟩
    refine Finset.mem_filter.mpr ⟨hqdom, ?_⟩
    by_contra hne
    have : x.2 ∈ t.meRow (key q) := mem_meRow.mpr (by rw [← hqk]; exact hx)
    rw [h.row_empty q hbq hne] at this
    exact absurd this (by simp)
  · rintro ⟨q, hq, rfl⟩
    have hmulti := (Finset.mem_filter.mp hq).2
    have h2 := h.multi_iff q hmulti
    obtain ⟨i, hi⟩ := Finset.card_pos.mp (show 0 < (t.meRow (key q)).card by omega)
    exact ⟨(key q, i), mem_meRow.mp hi, rfl⟩

/-- **`me.nvals` is the total number of edges of multi-edge pairs.** -/
theorem me_card_eq_sum (h : Inv t) :
    t.me.card = ∑ q ∈ t.multiPairs, (t.meRow (key q)).card := by
  have hfib : ∀ x ∈ t.me, x.1 ∈ t.multiPairs.image key := by
    intro x hx
    rw [← me_keys_eq h]
    exact Finset.mem_image_of_mem _ hx
  have hinj : ∀ x ∈ t.multiPairs, ∀ y ∈ t.multiPairs, key x = key y → x = y := by
    intro x hx y hy hxy
    exact key_inj (h.bounded x (mem_multiPairs.mp hx).1) (h.bounded y (mem_multiPairs.mp hy).1) hxy
  rw [Finset.card_eq_sum_card_fiberwise hfib, Finset.sum_image hinj]
  exact Finset.sum_congr rfl (fun q _ => card_filter_eq_card_meRow)

/-! ## The count -/

/-- The number of edges the tensor denotes. -/
def totalEdges (t : Tensor) : Nat := ∑ q ∈ t.effDom, (t.edgesAt q).card

theorem totalEdges_add_multiCount (h : Inv t) :
    totalEdges t + t.multiCount = t.effDom.card + t.me.card := by
  classical
  have hsplit := Finset.sum_filter_add_sum_filter_not t.effDom
    (fun q => t.effGet q = some MULTI) (fun q => (t.edgesAt q).card)
  have hmulti : ∑ q ∈ t.effDom.filter (fun q => t.effGet q = some MULTI), (t.edgesAt q).card
      = t.me.card := by
    rw [me_card_eq_sum h, multiPairs]
    exact Finset.sum_congr rfl (fun q hq => by
      rw [edgesAt_of_multi (Finset.mem_filter.mp hq).2])
  have hsingle : ∑ q ∈ t.effDom.filter (fun q => ¬ t.effGet q = some MULTI), (t.edgesAt q).card
      = (t.effDom.filter (fun q => ¬ t.effGet q = some MULTI)).card := by
    rw [Finset.card_eq_sum_ones]
    refine Finset.sum_congr rfl (fun q hq => ?_)
    obtain ⟨hqdom, hqne⟩ := Finset.mem_filter.mp hq
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (mem_effDom_iff_isSome.mp hqdom)
    rw [edgesAt_of_single hv (by rintro rfl; exact hqne hv), Finset.card_singleton]
  have hcards := Finset.card_filter_add_card_filter_not
    (s := t.effDom) (p := fun q => t.effGet q = some MULTI)
  rw [totalEdges, ← hsplit, hmulti, hsingle, multiCount, multiPairs]
  omega

/-- The `if dp.nvals == 0` guard is a pure optimisation: `intersection_nvals`
would return 0 anyway. -/
theorem edgeCount_eq (t : Tensor) :
    edgeCount t = t.m.dom.card + t.dp.dom.card - t.dm.card - (t.dp.dom ∩ t.m.dom).card
      - t.multiCount + t.me.card := by
  by_cases hz : t.dp.dom = ∅
  · simp [edgeCount, Layer.nvals, hz]
  · simp [edgeCount, Layer.nvals, hz]

/-- **`edge_count` counts edges.** -/
theorem edgeCount_eq_sum (h : Inv t) : edgeCount t = totalEdges t := by
  have hdom := effDom_card_add h
  have hsum := totalEdges_add_multiCount h
  have hmc : t.multiCount ≤ t.effDom.card :=
    Finset.card_le_card (Finset.filter_subset _ _)
  rw [edgeCount_eq]
  omega

/-- Each intermediate result of the `u64` expression is non-negative, so the
left-to-right unsigned evaluation in Rust cannot wrap. -/
theorem edgeCount_no_underflow (h : Inv t) :
    t.dm.card ≤ t.m.dom.card + t.dp.dom.card ∧
      (t.dp.dom ∩ t.m.dom).card ≤ t.m.dom.card + t.dp.dom.card - t.dm.card ∧
      t.multiCount ≤ t.m.dom.card + t.dp.dom.card - t.dm.card - (t.dp.dom ∩ t.m.dom).card := by
  have hdom := effDom_card_add h
  have hmc : t.multiCount ≤ t.effDom.card :=
    Finset.card_le_card (Finset.filter_subset _ _)
  refine ⟨by omega, by omega, by omega⟩

end Tensor
end FalkorDB
