/-
# `flush` and `rebuild_backward`

`flush` folds a delta into the committed base once the policy says it earns the
rewrite:

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

Proved: each fold leaves `eff_get` — hence the denotation and `edge_count` —
untouched, and re-establishes the delta invariants (afterwards the folded delta
is empty, so `dp ∩ dm = ∅` and cancel-to-clean hold trivially). The doc comment's
claim that "the two merges are order-independent" is witnessed by the fact that
each one separately preserves `eff_get` pointwise.

Everything here is proved for an **arbitrary** fold decision, never for the
policy that produces one. That is deliberate, and it is the theorem worth having:
the decision comes from a cost heuristic over approximate counters, so what needs
proving is that the heuristic *cannot matter*. `edgesAt_flush_decision_irrelevant`
states it directly, and `edgesAt_setAll_after_flush` /
`removeAll_after_flush_spec` discharge the consequence — that the `self.flush()`
now opening `set_all_from_slices` and `remove_all` cannot perturb their specs.

`rebuild_backward` recomputes `mt` from the effective forward pattern; it is the
step that *establishes* the `mt` invariant (used after `decode`, which leaves `mt`
empty).
-/
import Tensor.Count
-- `Tensor.Remove` (and `Tensor.Add` through it) for the entry-point absorption
-- corollaries at the end of this file.
import Tensor.Remove

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
  refine inv_of_effGet_eq h (foldDp_effGet h) rfl rfl ?_ ?_ ?_ ?_
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
  refine inv_of_effGet_eq h foldDm_effGet rfl rfl ?_ ?_ ?_ ?_
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

theorem flush_effGet (h : Inv t) (fdp fdm : Bool) (q : Pair) :
    (flush t fdp fdm).effGet q = t.effGet q := by
  cases fdp <;> cases fdm
  · rfl
  · exact foldDm_effGet q
  · exact foldDp_effGet h q
  · exact (foldDm_effGet q).trans (foldDp_effGet h q)

/-- **`flush` preserves every invariant**, whichever layers it folds. -/
theorem inv_flush (h : Inv t) (fdp fdm : Bool) : Inv (flush t fdp fdm) := by
  cases fdp <;> cases fdm
  · exact h
  · exact inv_foldDm h
  · exact inv_foldDp h
  · exact inv_foldDm (inv_foldDp h)

/-- **`flush` is invisible to readers**: same edges at every pair. -/
theorem edgesAt_flush (h : Inv t) (fdp fdm : Bool) (q : Pair) :
    (flush t fdp fdm).edgesAt q = t.edgesAt q := by
  refine edgesAt_congr_at (flush_effGet h fdp fdm q) ?_
  rw [meRow, meRow]
  congr 1
  cases fdp <;> cases fdm <;> rfl

/-- …and to `edge_count`, even though the formula's inputs all changed. -/
theorem edgeCount_flush (h : Inv t) (fdp fdm : Bool) :
    edgeCount (flush t fdp fdm) = edgeCount t := by
  rw [edgeCount_eq_sum (inv_flush h fdp fdm), edgeCount_eq_sum h, totalEdges, totalEdges]
  have hdom : (flush t fdp fdm).effDom = t.effDom := by
    ext q
    rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, flush_effGet h fdp fdm q]
  rw [hdom]
  exact Finset.sum_congr rfl (fun q _ => by rw [edgesAt_flush h fdp fdm q])

/-- **The fold decision is denotationally invisible.** Any two decisions leave the
same edges at every pair, so the whole fold policy — `WRITE_FOLD_K`,
`READ_FOLD_K`, `MIN_FOLD_DELTA`, the `delta_dominates_base` escape hatch, and the
approximate counters they are evaluated on — is a pure throughput concern that
cannot affect what the tensor denotes. Re-tuning any of it leaves every theorem in
this development standing. -/
theorem edgesAt_flush_decision_irrelevant (h : Inv t) (a b c d : Bool) (q : Pair) :
    (flush t a b).edgesAt q = (flush t c d).edgesAt q := by
  rw [edgesAt_flush h a b, edgesAt_flush h c d]

/-- `flush` keeps the bounds, so an `InBounds` precondition survives a fold. -/
@[simp] theorem flush_nrows (fdp fdm : Bool) : (flush t fdp fdm).nrows = t.nrows := by
  cases fdp <;> cases fdm <;> rfl

@[simp] theorem flush_ncols (fdp fdm : Bool) : (flush t fdp fdm).ncols = t.ncols := by
  cases fdp <;> cases fdm <;> rfl

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

/-! ## The entry-point fold is absorbed

`set_all_from_slices` and `remove_all` both now open with `self.flush()`, so the
state their specs are proved about is a *possibly just-folded* tensor rather than
the caller's. Those specs are stated over an arbitrary `Inv` tensor and phrased in
`edgesAt`, and a fold lands back in that class with the same `edgesAt` and the
same bounds — so they transfer. Rather than assert that, the two corollaries below
check it. -/

/-- The batch preconditions are `edgesAt`- and bounds-level, so a fold preserves
them. -/
theorem writableBatch_flush {l : List (Pair × Nat)} (fdp fdm : Bool)
    (hb : WritableBatch t l) : WritableBatch (flush t fdp fdm) l := by
  intro e he
  obtain ⟨⟨hr, hc⟩, hid⟩ := hb e he
  exact ⟨⟨by simpa using hr, by simpa using hc⟩, hid⟩

theorem freshBatch_flush {l : List (Pair × Nat)} (h : Inv t) (fdp fdm : Bool)
    (hf : FreshBatch t l) : FreshBatch (flush t fdp fdm) l := by
  refine ⟨hf.1, fun e he q => ?_⟩
  rw [edgesAt_flush h fdp fdm q]
  exact hf.2 e he q

/-- **`set_all_from_slices`' entry `flush()` cannot change its result.** A batch
applied after a fold lands on the same edges as the same batch applied before it. -/
theorem edgesAt_setAll_after_flush {l : List (Pair × Nat)} (h : Inv t)
    (hb : WritableBatch t l) (hf : FreshBatch t l) (fdp fdm : Bool) (q : Pair) :
    (setAll (flush t fdp fdm) l).edgesAt q = t.edgesAt q ∪ batchIds l q := by
  rw [edgesAt_setAll (inv_flush h fdp fdm) (writableBatch_flush fdp fdm hb)
    (freshBatch_flush h fdp fdm hf) q, edgesAt_flush h fdp fdm q]

theorem inv_setAll_after_flush {l : List (Pair × Nat)} (h : Inv t)
    (hb : WritableBatch t l) (hf : FreshBatch t l) (fdp fdm : Bool) :
    Inv (setAll (flush t fdp fdm) l) :=
  inv_setAll (inv_flush h fdp fdm) (writableBatch_flush fdp fdm hb)
    (freshBatch_flush h fdp fdm hf)

/-- **`remove_all`'s entry `flush()` cannot change its result** either: the same
edges are removed, the same pairs reported emptied, and the invariants hold. -/
theorem removeAll_after_flush_spec {rels : List (Nat × Pair)} (h : Inv t)
    (hex : ∀ r ∈ rels, r.1 ∈ t.edgesAt r.2)
    (fdp fdm : Bool) :
    Inv (removeAll (flush t fdp fdm) rels).1 ∧
      (∀ q, (removeAll (flush t fdp fdm) rels).1.edgesAt q
              = t.edgesAt q \ removedIds rels q) ∧
      ∀ q ∈ (removeAll (flush t fdp fdm) rels).2,
        (removeAll (flush t fdp fdm) rels).1.edgesAt q = ∅ := by
  obtain ⟨hinv, hedges, hemptied⟩ :=
    removeAll_spec (inv_flush h fdp fdm)
      (fun r hr => by rw [edgesAt_flush h fdp fdm r.2]; exact hex r hr)
  refine ⟨hinv, fun q => ?_, hemptied⟩
  rw [hedges q, edgesAt_flush h fdp fdm q]

end Tensor
end FalkorDB
