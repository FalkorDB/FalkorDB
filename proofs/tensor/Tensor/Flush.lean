/-
# `flush` and `rebuild_backward`

`flush` folds oversized deltas into the committed base:

```rust
if self.dp.nvals() >= 10000 { self.m.element_wise_add_second(&self.dp); self.dp.clear(); }
if self.dm.nvals() >= 10000 { self.m.remove_all(&self.dm); self.dm.clear(); }
```

Proved: each fold leaves `eff_get` — hence the denotation and `edge_count` —
untouched, and re-establishes the delta invariants (afterwards the folded delta
is empty, so `dp ∩ dm = ∅` and cancel-to-clean hold trivially).  The doc comment's
claim that "the two merges are order-independent" is witnessed by the fact that
each one separately preserves `eff_get` pointwise.

`rebuild_backward` recomputes `mt` from the effective forward pattern; it is the
step that *establishes* the `mt` invariant (used after `decode`, which leaves `mt`
empty).
-/
import Tensor.Count

namespace FalkorDB
namespace Tensor

variable {t : Tensor}

/-! ## Folding `dp` into `m` -/

theorem foldDp_effGet (h : Inv t) (q : Pair) : (foldDp t).effGet q = t.effGet q := by
  have hdpc : (foldDp t).dp.get q = none := by simp [foldDp, Layer.clear, Layer.get]
  by_cases hq : q ∈ t.dp.dom
  · have h2 : q ∉ t.dm := Finset.disjoint_left.mp h.dp_disj_dm hq
    rw [effGet_of_m hdpc (by simpa [foldDp] using h2), effGet_of_dp (Layer.get_of_mem hq)]
    show (t.m.mergeSecond t.dp).get q = some (t.dp.val q)
    rw [Layer.get_mergeSecond_of_mem_right hq, Layer.get_of_mem hq]
  · have h1 : t.dp.get q = none := Layer.get_eq_none.mpr hq
    have h3 : (foldDp t).m.get q = t.m.get q := Layer.get_mergeSecond_of_not_mem_right hq
    by_cases hqdm : q ∈ t.dm
    · rw [show (foldDp t).effGet q = none by simp [effGet, foldDp, hqdm],
        show t.effGet q = none by simp [effGet, h1, hqdm]]
    · rw [effGet_of_m hdpc (by simpa [foldDp] using hqdm), effGet_of_m h1 hqdm, h3]

theorem inv_foldDp (h : Inv t) : Inv (foldDp t) := by
  refine inv_of_effGet_eq h (foldDp_effGet h) rfl rfl rfl ?_ ?_ ?_ ?_
  · exact fun q hq => Finset.mem_union_left _ (h.dm_sub_m hq)
  · simp [foldDp, Layer.clear]
  · intro q hq
    simp [foldDp, Layer.clear] at hq
  · intro q hq
    refine h.in_range q ?_
    simp only [foldDp, Layer.dom_mergeSecond, Layer.dom_clear, Finset.union_empty] at hq
    exact hq

/-! ## Folding `dm` out of `m` -/

theorem foldDm_effGet (q : Pair) : (foldDm t).effGet q = t.effGet q := by
  have hdp : (foldDm t).dp.get q = t.dp.get q := rfl
  by_cases hq : q ∈ t.dp.dom
  · rw [effGet_of_dp (by rw [hdp]; exact Layer.get_of_mem hq),
      effGet_of_dp (Layer.get_of_mem hq)]
  · have h1 : t.dp.get q = none := Layer.get_eq_none.mpr hq
    have h1' : (foldDm t).dp.get q = none := by rw [hdp]; exact h1
    by_cases hqdm : q ∈ t.dm
    · rw [effGet_of_m h1' (by simp [foldDm]), show t.effGet q = none by simp [effGet, h1, hqdm]]
      exact Layer.get_removeAll_mem hqdm
    · rw [effGet_of_m h1' (by simp [foldDm]), effGet_of_m h1 hqdm]
      exact Layer.get_removeAll_not_mem hqdm

theorem inv_foldDm (h : Inv t) : Inv (foldDm t) := by
  refine inv_of_effGet_eq h foldDm_effGet rfl rfl rfl ?_ ?_ ?_ ?_
  · simp [foldDm]
  · simp [foldDm]
  · intro q hq
    have hq' : q ∈ t.dp.dom := hq
    have h2 : q ∉ t.dm := Finset.disjoint_left.mp h.dp_disj_dm hq'
    show (t.m.removeAll t.dm).get q ≠ some (t.dp.val q)
    rw [Layer.get_removeAll_not_mem h2]
    exact h.cancel_clean q hq'
  · intro q hq
    refine h.in_range q ?_
    rcases Finset.mem_union.mp hq with hq' | hq'
    · exact Finset.mem_union_left _ (Finset.mem_sdiff.mp hq').1
    · exact Finset.mem_union_right _ hq'

/-! ## `flush` -/

theorem flush_effGet (h : Inv t) (q : Pair) : (flush t).effGet q = t.effGet q := by
  unfold flush
  split
  · split
    · rw [foldDm_effGet, foldDp_effGet h]
    · exact foldDp_effGet h q
  · split
    · exact foldDm_effGet q
    · rfl

/-- **`flush` preserves every invariant.** -/
theorem inv_flush (h : Inv t) : Inv (flush t) := by
  unfold flush
  split
  · split
    · exact inv_foldDm (inv_foldDp h)
    · exact inv_foldDp h
  · split
    · exact inv_foldDm h
    · exact h

/-- **`flush` is invisible to readers**: same edges at every pair. -/
theorem edgesAt_flush (h : Inv t) (q : Pair) : (flush t).edgesAt q = t.edgesAt q := by
  refine edgesAt_congr_at (flush_effGet h q) ?_
  rw [meRow, meRow]
  congr 1
  unfold flush
  split <;> split <;> rfl

/-- …and to `edge_count`, even though the formula's inputs all changed. -/
theorem edgeCount_flush (h : Inv t) : edgeCount (flush t) = edgeCount t := by
  rw [edgeCount_eq_sum (inv_flush h), edgeCount_eq_sum h, totalEdges, totalEdges]
  have hdom : (flush t).effDom = t.effDom := by
    ext q
    rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, flush_effGet h q]
  rw [hdom]
  exact Finset.sum_congr rfl (fun q _ => by rw [edgesAt_flush h q])

/-! ## `rebuild_backward` -/

/-- **`rebuild_backward` establishes the backward-matrix invariant** from the
effective forward pattern — this is why `decode` may leave `mt` empty. -/
theorem inv_rebuildBackward (h : InvCore t) : Inv (rebuildBackward t) := by
  refine { h with mt_eq := ?_ }
  intro q
  show q ∈ t.effDom ↔ (q.2, q.1) ∈ (extract t).image (fun p => (p.2, p.1))
  rw [extract_eq_effDom]
  simp only [Finset.mem_image]
  constructor
  · exact fun hq => ⟨q, hq, rfl⟩
  · rintro ⟨r, hr, hrq⟩
    rwa [swap_eq_iff.mp hrq] at hr

@[simp] theorem edgesAt_rebuildBackward {q : Pair} :
    (rebuildBackward t).edgesAt q = t.edgesAt q := rfl

@[simp] theorem edgeCount_rebuildBackward : edgeCount (rebuildBackward t) = edgeCount t := rfl

end Tensor
end FalkorDB
