/-
# The iterators: `iter_edges`, `Iter` (forward and transposed)

Each iterator is characterised twice — which triples it yields, and that it
never repeats one.  For a multiset those two facts pin it down exactly, so
together they say "yields every edge exactly once".

* `mem_iterEdges` / `nodup_iterEdges` — the single streaming pass: inline ids
  straight from the forward matrix, then all of `me` with `(src, dst)` recovered
  by `>> 32` / `& 0xFFFF_FFFF` (this is where `compound_key`'s round-trip is
  needed).
* `mem_iterFwd` / `nodup_iterFwd` — `Tensor::iter(.., false)`.
* `mem_iterBwd` / `nodup_iterBwd` — `Tensor::iter(.., true)`: walks `mt`, so its
  row range selects by *destination*, and it recovers ids via `eff_get`.
* `iterBwd_eff_get_isSome` — the `unwrap_or(0)` fallback in the backward branch
  is dead code: `mt` never holds a pair the forward matrix has lost.
-/
import Tensor.Count

namespace FalkorDB
namespace Tensor

variable {t : Tensor}

/-! ## What one pair contributes -/

theorem iterAt_none {p : Pair} (hg : t.effGet p = none) : t.iterAt p = 0 := by
  simp [iterAt, hg]

theorem iterAt_multi {p : Pair} (hg : t.effGet p = some MULTI) :
    t.iterAt p = (t.meRow (key p)).val.map (fun i => (p.1, p.2, i)) := by
  simp [iterAt, hg]

theorem iterAt_single {p : Pair} {v : Nat} (hg : t.effGet p = some v) (hM : v ≠ MULTI) :
    t.iterAt p = {(p.1, p.2, v)} := by
  simp [iterAt, hg, hM]

theorem mem_iterAt {p : Pair} {x : Nat × Nat × Nat} :
    x ∈ t.iterAt p ↔ x.1 = p.1 ∧ x.2.1 = p.2 ∧ x.2.2 ∈ t.edgesAt p := by
  cases hg : t.effGet p with
  | none => simp [iterAt_none hg, edgesAt_of_none hg]
  | some v =>
    by_cases hM : v = MULTI
    · subst hM
      rw [iterAt_multi hg, edgesAt_of_multi hg]
      simp only [Multiset.mem_map, Finset.mem_val]
      constructor
      · rintro ⟨i, hi, rfl⟩; exact ⟨rfl, rfl, hi⟩
      · rintro ⟨h1, h2, h3⟩
        exact ⟨x.2.2, h3, by rw [← h1, ← h2]⟩
    · rw [iterAt_single hg hM, edgesAt_of_single hg hM]
      simp only [Multiset.mem_singleton, Finset.mem_singleton]
      constructor
      · rintro rfl; exact ⟨rfl, rfl, rfl⟩
      · rintro ⟨h1, h2, h3⟩
        exact Prod.ext h1 (Prod.ext h2 h3)

theorem nodup_iterAt {p : Pair} : (t.iterAt p).Nodup := by
  cases hg : t.effGet p with
  | none => simp [iterAt_none hg]
  | some v =>
    by_cases hM : v = MULTI
    · subst hM
      rw [iterAt_multi hg]
      refine Multiset.Nodup.map_on ?_ (t.meRow (key p)).nodup
      intro x _ y _ hxy
      simpa using hxy
    · simp [iterAt_single hg hM]

/-- Distinct pairs contribute disjoint triples (the first two components are the
pair), so the whole iteration is duplicate-free. -/
theorem iterAt_disjoint {p q : Pair} (hpq : p ≠ q) :
    Disjoint (t.iterAt p) (t.iterAt q) := by
  rw [Multiset.disjoint_left]
  intro x hx hx'
  obtain ⟨h1, h2, _⟩ := mem_iterAt.mp hx
  obtain ⟨h3, h4, _⟩ := mem_iterAt.mp hx'
  exact hpq (Prod.ext (by rw [← h1, h3]) (by rw [← h2, h4]))

/-! ## `Tensor::iter` (forward) -/

/-- **The forward iterator yields exactly the edges of the requested rows.** -/
theorem mem_iterFwd {a b : Nat} {x : Nat × Nat × Nat} :
    x ∈ iterFwd t a b ↔ x.2.2 ∈ t.edgesAt (x.1, x.2.1) ∧ a ≤ x.1 ∧ x.1 ≤ b := by
  simp only [iterFwd, Multiset.mem_bind, Finset.mem_val, Finset.mem_filter]
  constructor
  · rintro ⟨p, ⟨_, hrange⟩, hx⟩
    obtain ⟨h1, h2, h3⟩ := mem_iterAt.mp hx
    refine ⟨?_, ?_, ?_⟩
    · rw [show (x.1, x.2.1) = p from Prod.ext h1 h2]; exact h3
    · rw [h1]; exact hrange.1
    · rw [h1]; exact hrange.2
  · rintro ⟨hmem, h1, h2⟩
    have hdom : (x.1, x.2.1) ∈ t.effDom := by
      by_contra hc
      rw [edgesAt_eq_empty_of_not_mem hc] at hmem
      exact absurd hmem (by simp)
    exact ⟨(x.1, x.2.1), ⟨hdom, ⟨h1, h2⟩⟩, mem_iterAt.mpr ⟨rfl, rfl, hmem⟩⟩

theorem nodup_iterFwd {a b : Nat} : (iterFwd t a b).Nodup := by
  refine Multiset.nodup_bind.mpr ⟨fun p _ => nodup_iterAt, ?_⟩
  exact Multiset.Nodup.pairwise (fun p _ q _ hpq => iterAt_disjoint hpq)
    (Finset.filter (inRows a b) t.effDom).nodup

/-! ## `Tensor::iter` (transposed) -/

/-- `mt` is exactly the transpose of the effective forward pattern. -/
theorem mt_eq_image (h : Inv t) : t.mt = t.effDom.image (fun p => (p.2, p.1)) := by
  ext r
  simp only [Finset.mem_image]
  constructor
  · intro hr
    exact ⟨(r.2, r.1), (h.mt_eq (r.2, r.1)).mpr hr, rfl⟩
  · rintro ⟨p, hp, rfl⟩
    exact (h.mt_eq p).mp hp

/-- **The `unwrap_or(0)` in the backward branch of `Iter::next` is unreachable.** -/
theorem iterBwd_eff_get_isSome (h : Inv t) {r : Pair} (hr : r ∈ t.mt) :
    (t.effGet (r.2, r.1)).isSome :=
  mem_effDom_iff_isSome.mp ((h.mt_eq (r.2, r.1)).mpr hr)

/-- **The transposed iterator yields exactly the edges whose destination is in
the requested row range** (its rows are destinations). -/
theorem mem_iterBwd (h : Inv t) {a b : Nat} {x : Nat × Nat × Nat} :
    x ∈ iterBwd t a b ↔ x.2.2 ∈ t.edgesAt (x.1, x.2.1) ∧ a ≤ x.2.1 ∧ x.2.1 ≤ b := by
  simp only [iterBwd, Multiset.mem_bind, Finset.mem_val, Finset.mem_filter]
  constructor
  · rintro ⟨r, ⟨hr, hrange⟩, hx⟩
    obtain ⟨h1, h2, h3⟩ := mem_iterAt.mp hx
    refine ⟨?_, ?_, ?_⟩
    · rw [show (x.1, x.2.1) = (r.2, r.1) from Prod.ext h1 h2]; exact h3
    · rw [h2]; exact hrange.1
    · rw [h2]; exact hrange.2
  · rintro ⟨hmem, h1, h2⟩
    have hdom : (x.1, x.2.1) ∈ t.effDom := by
      by_contra hc
      rw [edgesAt_eq_empty_of_not_mem hc] at hmem
      exact absurd hmem (by simp)
    refine ⟨(x.2.1, x.1), ⟨(h.mt_eq (x.1, x.2.1)).mp hdom, ⟨h1, h2⟩⟩, ?_⟩
    exact mem_iterAt.mpr ⟨rfl, rfl, hmem⟩

theorem nodup_iterBwd {a b : Nat} : (iterBwd t a b).Nodup := by
  refine Multiset.nodup_bind.mpr ⟨fun r _ => nodup_iterAt, ?_⟩
  refine Multiset.Nodup.pairwise (fun r _ r' _ hrr => ?_) (Finset.filter (inRows a b) t.mt).nodup
  exact iterAt_disjoint (fun hc => hrr (Prod.ext (congrArg (fun z => z.2) hc)
    (congrArg Prod.fst hc)))

/-! ## `Tensor::iter_edges` -/

/-- **`iter_edges` yields exactly every `(src, dst, edge id)` of the tensor.**

The `me` half is where `compound_key` is inverted: the stored row key is split
back into `(src, dst)` with `>> 32` and `& 0xFFFF_FFFF`. -/
theorem mem_iterEdges (h : Inv t) {x : Nat × Nat × Nat} :
    x ∈ iterEdges t ↔ x.2.2 ∈ t.edgesAt (x.1, x.2.1) := by
  simp only [iterEdges, Multiset.mem_add, Multiset.mem_filter, fwdIter, Multiset.mem_map,
    Finset.mem_val, Finset.mem_filter]
  constructor
  · rintro (⟨⟨p, ⟨hp, _⟩, hx⟩, hne⟩ | ⟨y, hy, hxy⟩)
    · obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (mem_effDom_iff_isSome.mp hp)
      have hx1 : x.1 = p.1 := by rw [← hx]
      have hx2 : x.2.1 = p.2 := by rw [← hx]
      have hx3 : x.2.2 = v := by rw [← hx]; simp [hv]
      have hMv : v ≠ MULTI := by
        rw [← hx3]; exact hne
      rw [show (x.1, x.2.1) = p from Prod.ext hx1 hx2, edgesAt_of_single hv hMv, hx3]
      simp
    · obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed y hy
      have hrow : y.2 ∈ t.meRow (key q) := mem_meRow.mpr (by rw [← hqk]; exact hy)
      have hmulti : t.effGet q = some MULTI := by
        by_contra hne
        rw [h.row_empty q hbq hne] at hrow
        exact absurd hrow (by simp)
      have h1 : x.1 = q.1 := by rw [← hxy]; simpa [hqk] using keyHi hbq
      have h2 : x.2.1 = q.2 := by rw [← hxy]; simpa [hqk] using keyLo hbq
      have h3 : x.2.2 = y.2 := by rw [← hxy]
      rw [show (x.1, x.2.1) = q from Prod.ext h1 h2, edgesAt_of_multi hmulti, h3]
      exact hrow
  · intro hmem
    have hdom : (x.1, x.2.1) ∈ t.effDom := by
      by_contra hc
      rw [edgesAt_eq_empty_of_not_mem hc] at hmem
      exact absurd hmem (by simp)
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (mem_effDom_iff_isSome.mp hdom)
    by_cases hM : v = MULTI
    · subst hM
      refine Or.inr ⟨(key (x.1, x.2.1), x.2.2), ?_, ?_⟩
      · exact mem_meRow.mp (by rw [← edgesAt_of_multi hv]; exact hmem)
      · have hb : Bounded (x.1, x.2.1) := h.bounded _ hdom
        rw [show ((key (x.1, x.2.1)) >>> 32, key (x.1, x.2.1) % 2 ^ 32, x.2.2)
          = (x.1, x.2.1, x.2.2) from by rw [keyHi hb, keyLo hb]]
    · refine Or.inl ⟨⟨(x.1, x.2.1), ⟨hdom, ⟨Nat.zero_le _, ?_⟩⟩, ?_⟩, ?_⟩
      · have hb : Bounded (x.1, x.2.1) := h.bounded _ hdom
        have h1' : x.1 < 2 ^ 32 := hb.1
        have h32 : (2:Nat) ^ 32 < 2 ^ 64 := Nat.pow_lt_pow_right (by omega) (by omega)
        omega
      · have hxv : x.2.2 = v := by
          have hs := edgesAt_of_single hv hM
          rw [hs] at hmem
          simpa using hmem
        rw [hv, ← hxv]
        rfl
      · have : x.2.2 = v := by
          have hs := edgesAt_of_single hv hM
          rw [hs] at hmem
          simpa using hmem
        rw [this]
        exact hM


/-- The two halves of `iter_edges`, characterised separately: the streaming pass
only yields non-sentinel inline ids… -/
theorem mem_iterEdges_fwd {x : Nat × Nat × Nat}
    (hx : x ∈ (t.fwdIter 0 (2 ^ 64 - 1)).filter (fun y => y.2.2 ≠ MULTI)) :
    t.effGet (x.1, x.2.1) = some x.2.2 ∧ x.2.2 ≠ MULTI := by
  simp only [Multiset.mem_filter, fwdIter, Multiset.mem_map, Finset.mem_val,
    Finset.mem_filter] at hx
  obtain ⟨⟨p, ⟨hp, _⟩, hxp⟩, hne⟩ := hx
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (mem_effDom_iff_isSome.mp hp)
  have h1 : x.1 = p.1 := by rw [← hxp]
  have h2 : x.2.1 = p.2 := by rw [← hxp]
  have h3 : x.2.2 = v := by rw [← hxp]; simp [hv]
  refine ⟨?_, hne⟩
  rw [show (x.1, x.2.1) = p from Prod.ext h1 h2, hv, h3]

/-- …and the `me` pass only yields ids of sentinel pairs. -/
theorem mem_iterEdges_me (h : Inv t) {x : Nat × Nat × Nat}
    (hx : x ∈ t.me.val.map (fun y => (y.1 >>> 32, y.1 % 2 ^ 32, y.2))) :
    t.effGet (x.1, x.2.1) = some MULTI := by
  simp only [Multiset.mem_map, Finset.mem_val] at hx
  obtain ⟨y, hy, hxy⟩ := hx
  obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed y hy
  have hrow : y.2 ∈ t.meRow (key q) := mem_meRow.mpr (by rw [← hqk]; exact hy)
  have hmulti : t.effGet q = some MULTI := by
    by_contra hne
    rw [h.row_empty q hbq hne] at hrow
    exact absurd hrow (by simp)
  have h1 : x.1 = q.1 := by rw [← hxy]; simpa [hqk] using keyHi hbq
  have h2 : x.2.1 = q.2 := by rw [← hxy]; simpa [hqk] using keyLo hbq
  rw [show (x.1, x.2.1) = q from Prod.ext h1 h2]
  exact hmulti

/-- **`iter_edges` never yields the same edge twice** — the two passes cover
disjoint pairs (sentinel vs non-sentinel), and neither repeats. -/
theorem nodup_iterEdges (h : Inv t) : (iterEdges t).Nodup := by
  refine Multiset.nodup_add.mpr ⟨?_, ?_, ?_⟩
  · refine Multiset.Nodup.filter _ ?_
    refine Multiset.Nodup.map_on ?_ (Finset.filter (inRows 0 (2 ^ 64 - 1)) t.effDom).nodup
    intro p _ q _ hpq
    exact Prod.ext (congrArg (fun z => z.1) hpq) (congrArg (fun z => z.2.1) hpq)
  · refine Multiset.Nodup.map_on ?_ t.me.nodup
    intro y hy z hz hyz
    obtain ⟨q, hbq, _, hqk⟩ := h.me_keyed y (Finset.mem_val.mp hy)
    obtain ⟨r, hbr, _, hrk⟩ := h.me_keyed z (Finset.mem_val.mp hz)
    have h1 : y.1 = z.1 := by
      have e1 : y.1 >>> 32 = z.1 >>> 32 := congrArg Prod.fst hyz
      have e2 : y.1 % 2 ^ 32 = z.1 % 2 ^ 32 := congrArg (fun w => w.2.1) hyz
      rw [hqk, hrk] at e1 e2 ⊢
      rw [keyHi hbq, keyHi hbr] at e1
      rw [keyLo hbq, keyLo hbr] at e2
      exact congrArg key (Prod.ext e1 e2)
    exact Prod.ext h1 (congrArg (fun w => w.2.2) hyz)
  · rw [Multiset.disjoint_left]
    intro x hx hx'
    have h1 := mem_iterEdges_fwd hx
    have h2 := mem_iterEdges_me h hx'
    rw [h2] at h1
    exact h1.2 (Option.some_inj.mp h1.1).symm

end Tensor
end FalkorDB
