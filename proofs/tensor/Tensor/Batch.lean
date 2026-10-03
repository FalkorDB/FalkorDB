/-
# The `batch` map of `set_all_from_slices`

`set_all_from_slices` does not process edges one at a time: it runs a *read
phase* that decides each edge's placement while reading only the pre-batch
layers, and a *write phase* that applies the queued inline values afterwards.
The one place where that differs from per-edge processing is a pair appearing
twice **within one batch**:

```rust
Entry::Occupied(mut e) => {
    let idx = *e.get();
    if idx != usize::MAX {
        // Second edge of a pair new in this batch: promote the
        // pending inline slot in place.
        self.me.set(key, m_ids[idx], true);
        m_ids[idx] = MULTI_EDGE;
        self.multi_count += 1;
        e.insert(usize::MAX);
    }
    self.me.set(key, id, true);
}
```

The pending inline slot is rewritten to the sentinel *retroactively*, keeping the
`m_masked` value that was recorded when the pair was first seen — whereas
sequential processing would recompute it from the intermediate state.  Those two
`m_masked` values are generally different, and this file proves it does not
matter: the retroactive promotion lands in exactly the same state as promoting
sequentially.  `retro_promote_effGet` is the case the doc comment's "record the
committed value so the write phase cancels re-promotion back to clean" is about.

What is *not* mechanised here is the list-level plumbing of the read phase (the
`FxHashMap` and the parallel `Vec`s); `Add.lean` proves the sequential semantics
of a whole batch (`inv_setAll`, `edgesAt_setAll`), and this file closes the one
semantic gap between that and the batched implementation.
-/
import Tensor.Add

namespace FalkorDB
namespace Tensor

variable {t : Tensor} {p : Pair} {i1 i2 : Nat}

/-- The state the read phase leaves for a pair that was absent before the batch
and gained a second edge: sentinel queued for the inline slot, both ids in `me`,
both ids in `me` (which is what makes the pair count as multi-edge, there being
no counter to bump), and the `m_masked` value **recorded at first sight**. -/
def retroPromote (t : Tensor) (p : Pair) (i1 i2 : Nat) : Tensor :=
  writeInline
    { t with me := insert (key p, i1) (insert (key p, i2) t.me) }
    p MULTI (if p ∈ t.dm then t.m.get p else none)

private theorem seq_def (hv : t.effGet p = none) (hid1 : ValidId i1) :
    addEdge (addEdge t p i1) p i2 =
      writeInline
        { addEdge t p i1 with
          me := insert (key p, i1) (insert (key p, i2) (addEdge t p i1).me) }
        p MULTI (if ((addEdge t p i1).dp.get p).isSome then (addEdge t p i1).m.get p else none) := by
  have h1 : (addEdge t p i1).effGet p = some i1 := by
    rw [addEdge_first_def hv]
    exact writeInline_effGet_self first_mm
  exact addEdge_promote_def h1 hid1.ne_multi

/-- **The retroactive promotion agrees with sequential promotion**, everywhere:
same effective value at every pair, same `me`, same multi-edge pair count
(which follows from the first, the count being derived), same `mt`. -/
theorem retro_promote_agrees (hv : t.effGet p = none) (hid1 : ValidId i1) :
    (∀ q, (addEdge (addEdge t p i1) p i2).effGet q = (retroPromote t p i1 i2).effGet q) ∧
      (addEdge (addEdge t p i1) p i2).me = (retroPromote t p i1 i2).me ∧
      (addEdge (addEdge t p i1) p i2).multiCount = (retroPromote t p i1 i2).multiCount ∧
      (addEdge (addEdge t p i1) p i2).mt = (retroPromote t p i1 i2).mt := by
  have hme1 : (addEdge t p i1).me = t.me := by
    rw [addEdge_first_def hv]; exact writeInline_me
  have hmt1 : (addEdge t p i1).mt = insert (p.2, p.1) t.mt := by
    rw [addEdge_first_def hv]; exact writeInline_mt
  have hm1 : (addEdge t p i1).m = t.m := by rw [addEdge_first_def hv]; exact writeInline_m
  -- both sides put the sentinel at `p` and leave every other pair alone
  have hget : ∀ q, (addEdge (addEdge t p i1) p i2).effGet q
      = (retroPromote t p i1 i2).effGet q := by
    intro q
    by_cases hq : q = p
    · subst hq
      rw [seq_def hv hid1, writeInline_effGet_self mm_of_if, retroPromote,
        writeInline_effGet_self mm_of_if]
    · rw [seq_def hv hid1, writeInline_effGet_ne hq, retroPromote, writeInline_effGet_ne hq]
      show (addEdge t p i1).effGet q = t.effGet q
      rw [addEdge_first_def hv]
      exact writeInline_effGet_ne hq
  refine ⟨hget, ?_, ?_, ?_⟩
  · rw [seq_def hv hid1, writeInline_me, retroPromote, writeInline_me]
    show insert (key p, i1) (insert (key p, i2) (addEdge t p i1).me) = _
    rw [hme1]
  · -- the count is derived from the effective view, so the first conjunct gives
    -- this one: there is no separately-maintained counter left to compare.
    have hdom : (addEdge (addEdge t p i1) p i2).effDom
        = (retroPromote t p i1 i2).effDom := by
      ext q; rw [mem_effDom_iff_isSome, mem_effDom_iff_isSome, hget q]
    rw [multi_count_eq, multi_count_eq, multiPairs_congr hdom hget]
  · rw [seq_def hv hid1, writeInline_mt, retroPromote, writeInline_mt]
    show insert (p.2, p.1) (addEdge t p i1).mt = insert (p.2, p.1) t.mt
    rw [hmt1, Finset.insert_idem]

/-- Hence the two agree on the denotation: the batch's retroactive promotion
stores exactly the two edges, at exactly this pair. -/
theorem retro_promote_edgesAt (hv : t.effGet p = none) (hid1 : ValidId i1) (q : Pair) :
    (retroPromote t p i1 i2).edgesAt q = (addEdge (addEdge t p i1) p i2).edgesAt q := by
  obtain ⟨hget, hme, _, _⟩ := retro_promote_agrees (i2 := i2) hv hid1
  exact edgesAt_congr_at (hget q).symm (by rw [meRow, meRow, hme])

end Tensor
end FalkorDB
