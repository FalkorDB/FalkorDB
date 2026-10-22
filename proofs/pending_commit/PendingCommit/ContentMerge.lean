import PendingCommit.Content
/-
# `Block::merge_span` with contents (attribute_store.rs:636-795)

`mergeSpanC` takes the same path `merge_span` takes (`old.cap != 0` guards the
two fast paths, :646/:671), computes the merged list with the path's loop
(`Span.patch`, `Span.pureRemove`, `Span.mergeGen`) and lays it out in the arena
the way that path does: fast path 1 overwrites in place (same length); fast
path 2 shrinks in place or retires an emptied span; the general path writes in
place when `0 < new_len ≤ cap`, else retires the old span (if any) and appends.
(The per-entry write loops are summarised as "the span's first `new_len`
cells become `L`"; cells past `new_len` inside `cap` are slack nobody reads.)

* `mergeL_eq` — the path choice on `cap` is `Span.mergeSpan`'s choice on
  `span ≠ []` under `CInv`.
* **`mergeSpanC_ok`** — slot `i` then reads `(Span.mergeSpan old pairs).1`
  (so, with `Span.mergeSpan_correct`, the reference semantics), the counts are
  `mergeSpan`'s, every other slot is unchanged, and `CInv` holds.
-/
namespace PendingCommit.Content
open Arena (Slot)

/-- Merged list, counts, path (0 = patch, 1 = pure removal, 2 = general). -/
def mergeL (b : CB) (i : Nat) (pairs : List Entry) : List Entry × Nat × Nat × Nat :=
  let cur := span b i
  if (slotAt b i).cap ≠ 0 ∧ allPresentNonNull cur pairs = true then
    (patch cur pairs, pairs.length, pairs.length, 0)
  else if (slotAt b i).cap ≠ 0 ∧ allNull pairs = true then
    let r := pureRemove cur pairs
    (r.1, r.2, 0, 1)
  else
    let r := mergeGen cur pairs
    (r.1, r.2.1, r.2.2, 2)

def retireTo (b : CB) (i : Nat) : CB :=
  { b with dead := b.dead + (slotAt b i).cap, slack := b.slack - ((slotAt b i).cap - (slotAt b i).len) }

def inPlace (b : CB) (i : Nat) (L : List Entry) : CB :=
  setSlot { b with arena := writeAt b.arena (slotAt b i).off L,
                   slack := b.slack + (slotAt b i).len - L.length } i
    ⟨(slotAt b i).off, u16 L.length, (slotAt b i).cap⟩

def reloc (b : CB) (i : Nat) (L : List Entry) : CB :=
  setSlot { b with arena := b.arena ++ L, dead := b.dead + (slotAt b i).cap,
                   slack := b.slack - ((slotAt b i).cap - (slotAt b i).len) } i
    ⟨b.arena.length, u16 L.length, u16 L.length⟩

/-- `Block::merge_span` (:636): new block and `(nremoved, nset)`. -/
def mergeSpanC (b0 : CB) (i : Nat) (pairs : List Entry) : CB × Nat × Nat :=
  let b := grow b0 i
  let m := mergeL b i pairs
  let L := m.1
  let b' :=
    match m.2.2.2 with
    | 0 => { b with arena := writeAt b.arena (slotAt b i).off L }
    | 1 => if L.length = 0 then setSlot (retireTo b i) i Slot.empty else inPlace b i L
    | _ =>
      if L.length = 0 then
        (if (slotAt b i).cap ≠ 0 then setSlot (retireTo b i) i Slot.empty else setSlot b i Slot.empty)
      else if L.length ≤ (slotAt b i).cap then inPlace b i L
      else reloc b i L
  (b', m.2.1, m.2.2.1)

/-! ## Layout lemmas -/

theorem span_length (b : CB) (h : CInv b) (i : Nat) : (span b i).length = (slotAt b i).len := by
  simp only [span, spanOf, List.length_take, List.length_drop]
  have h2 := len_le_cap b h i
  rcases slot_inb b h i with h1 | h1
  · omega
  · have := cap0_len0 b h i h1; omega

theorem inPlace_ok (b : CB) (h : CInv b) (i : Nat) (hi : i < b.slots.length) (L : List Entry)
    (hpos : 1 ≤ L.length) (hle : L.length ≤ (slotAt b i).cap) (hn : L.length ≤ 65535) :
    CInv (inPlace b i L) ∧ span (inPlace b i L) i = L ∧ ∀ j, j ≠ i → span (inPlace b i L) j = span b j := by
  have hsl := Arena.slot_slack_le h.arena hi
  have ha := h.arena.arena_eq
  have hs := h.arena.slack_eq
  rw [aslot] at hsl; simp only [toBlk] at hsl ha hs
  have hlc := len_le_cap b h i
  have hinb := slot_inb b h i
  have hin : (slotAt b i).off + L.length ≤ b.arena.length := by rcases hinb with h1 | h1 <;> omega
  have hwl := writeAt_length b.arena (slotAt b i).off L hin
  unfold inPlace; rw [u16_id hn]
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · have := Arena.inv_setSlot h.arena hi (x := ⟨(slotAt b i).off, L.length, (slotAt b i).cap⟩)
      (arena := b.arena.length) (dead := b.dead) (slack := b.slack + (slotAt b i).len - L.length)
      (by rw [aslot]; simp only [toBlk]; omega) (by rw [aslot]; simp only [toBlk]; omega)
      (.inr ⟨hpos, hle⟩) (by rcases hinb with h1 | h1 <;> simp only <;> omega) (Nat.le_refl _)
    simp only [toBlk, setSlot, hwl] at this ⊢; exact this
  · exact disj_set b h i hi _ (fun j hj => by
      have := h.disj i j (Ne.symm hj); unfold Apart at this ⊢; simpa using this) _ rfl
  · simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, ite_true]
    exact spanOf_writeAt_self _ _ _ hin
  · intro j hj
    simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, hj, ite_false]
    rw [← slotAt_def]
    apply spanOf_writeAt_frame _ _ _ hin
    have hap := h.disj i j (Ne.symm hj)
    have := len_le_cap b h j
    unfold Apart at hap; omega

theorem reloc_ok (b : CB) (h : CInv b) (i : Nat) (hi : i < b.slots.length) (L : List Entry)
    (hpos : 1 ≤ L.length) (hn : L.length ≤ 65535) :
    CInv (reloc b i L) ∧ span (reloc b i L) i = L ∧ ∀ j, j ≠ i → span (reloc b i L) j = span b j := by
  have hsl := Arena.slot_slack_le h.arena hi
  have ha := h.arena.arena_eq
  have hs := h.arena.slack_eq
  rw [aslot] at hsl; simp only [toBlk] at hsl ha hs
  have hlc := len_le_cap b h i
  unfold reloc; rw [u16_id hn]
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · have := Arena.inv_setSlot h.arena hi (x := ⟨b.arena.length, L.length, L.length⟩)
      (arena := b.arena.length + L.length) (dead := b.dead + (slotAt b i).cap)
      (slack := b.slack - ((slotAt b i).cap - (slotAt b i).len))
      (by rw [aslot]; simp only [toBlk]; omega) (by rw [aslot]; simp only [toBlk]; omega)
      (.inr ⟨hpos, Nat.le_refl _⟩) (.inl (by simp)) (by simp [toBlk])
    simp only [toBlk, setSlot, List.length_append] at this ⊢; exact this
  · exact disj_set b h i hi _ (fun j _ => by
      have := slot_inb b h j; unfold Apart; simp only; omega) _ rfl
  · simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, ite_true]
    exact spanOf_append_self _ _ _
  · intro j hj
    simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, hj, ite_false]
    rw [← slotAt_def]
    apply spanOf_append_frame
    have := slot_inb b h j; have := len_le_cap b h j; omega

theorem emptyCap0_ok (b : CB) (h : CInv b) (i : Nat) (hi : i < b.slots.length) (hc : (slotAt b i).cap = 0) :
    CInv (setSlot b i Slot.empty) ∧ span (setSlot b i Slot.empty) i = [] ∧
      ∀ j, j ≠ i → span (setSlot b i Slot.empty) j = span b j := by
  have ha := h.arena.arena_eq
  have hs := h.arena.slack_eq
  simp only [toBlk] at ha hs
  have h0 := cap0_len0 b h i hc
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · have := Arena.inv_setSlot h.arena hi (x := Slot.empty) (arena := b.arena.length) (dead := b.dead)
      (slack := b.slack) (by rw [aslot]; simp only [toBlk, Slot.empty]; omega)
      (by rw [aslot]; simp only [toBlk, Slot.empty]; omega) (.inl ⟨rfl, rfl⟩) (.inr rfl) (Nat.le_refl _)
    exact this
  · exact disj_set b h i hi Slot.empty (fun _ _ => .inl rfl) _ rfl
  · simp only [span, setSlot_arena, slotAt_set _ _ _ _ hi, ite_true]; simp [spanOf, Slot.empty]
  · intro j hj; simp only [span, setSlot_arena, slotAt_set _ _ _ _ hi, hj, ite_false]

theorem retire_empty_eq (b : CB) (i : Nat) : setSlot (retireTo b i) i Slot.empty = freeSpanC b i ∨
    (slotAt b i).cap = 0 := by
  by_cases hc : (slotAt b i).cap = 0
  · exact .inr hc
  · left; simp [freeSpanC, hc, retireTo]

theorem nonempty_iff_cap (b : CB) (h : CInv b) (i : Nat) : (span b i ≠ []) ↔ (slotAt b i).cap ≠ 0 := by
  have hl := span_length b h i
  have := slot_wf b h i
  constructor
  · intro hne hc; apply hne; apply List.eq_nil_of_length_eq_zero; rw [hl]; exact cap0_len0 b h i hc
  · intro hc hnil; rw [hnil] at hl; simp at hl
    rcases this with ⟨h1, _⟩ | ⟨h1, _⟩ <;> omega

/-- **`mergeL_eq`**: `mergeL` is `Span.mergeSpan` on the current span. -/
theorem mergeL_eq (b : CB) (h : CInv b) (i : Nat) (pairs : List Entry) :
    ((mergeL b i pairs).1, (mergeL b i pairs).2.1, (mergeL b i pairs).2.2.1) = mergeSpan (span b i) pairs := by
  have e := nonempty_iff_cap b h i
  unfold mergeL mergeSpan
  by_cases hc : (slotAt b i).cap = 0
  · have hs : span b i = [] := Classical.byContradiction fun hne => (e.1 hne) hc
    simp [hc, hs]
  · have hs : span b i ≠ [] := e.2 hc
    simp only [ne_eq, hc, not_false_eq_true, true_and, hs]
    by_cases a1 : allPresentNonNull (span b i) pairs = true
    · simp [a1]
    · by_cases a2 : allNull pairs = true
      · simp [a1, a2]
      · simp [a1, a2]

theorem set_self (b : CB) (i : Nat) (hi : i < b.slots.length) : b.slots.set i (slotAt b i) = b.slots := by
  apply List.ext_getElem?; intro k
  by_cases hk : k = i
  · subst hk; simp [slotAt, List.getD_eq_getElem?_getD, hi]
  · simp [List.getElem?_set_ne (Ne.symm hk)]

theorem patch_length (old pairs : List Entry) : (patch old pairs).length = old.length := by simp [patch]

theorem inPlace_same (b : CB) (h : CInv b) (i : Nat) (hi : i < b.slots.length) (L : List Entry)
    (hl : L.length = (span b i).length) (hn : L.length ≤ 65535) :
    inPlace b i L = { b with arena := writeAt b.arena (slotAt b i).off L } := by
  have := span_length b h i
  unfold inPlace setSlot
  have e : (⟨(slotAt b i).off, u16 L.length, (slotAt b i).cap⟩ : Slot) = slotAt b i := by
    rw [u16_id hn, hl, this]
  simp only [e, set_self b i hi]
  congr 1; omega

theorem mergeL_cases (b : CB) (i : Nat) (pairs : List Entry) :
    ((slotAt b i).cap ≠ 0 ∧ allPresentNonNull (span b i) pairs = true ∧
      mergeL b i pairs = (patch (span b i) pairs, pairs.length, pairs.length, 0)) ∨
    ((slotAt b i).cap ≠ 0 ∧ allNull pairs = true ∧
      mergeL b i pairs = ((pureRemove (span b i) pairs).1, (pureRemove (span b i) pairs).2, 0, 1)) ∨
    (mergeL b i pairs = ((mergeGen (span b i) pairs).1, (mergeGen (span b i) pairs).2.1,
      (mergeGen (span b i) pairs).2.2, 2)) := by
  unfold mergeL
  by_cases hA : (slotAt b i).cap ≠ 0 ∧ allPresentNonNull (span b i) pairs = true
  · exact .inl ⟨hA.1, hA.2, by rw [if_pos hA]⟩
  · by_cases hB : (slotAt b i).cap ≠ 0 ∧ allNull pairs = true
    · exact .inr (.inl ⟨hB.1, hB.2, by rw [if_neg hA, if_pos hB]⟩)
    · exact .inr (.inr (by rw [if_neg hA, if_neg hB]))

/-- **`mergeSpanC_ok`**: `merge_span` leaves slot `i` reading the merged list
`Span.mergeSpan` computes, returns its counts, keeps every other slot and
`CInv`. Needs the merged span to fit the `u16` length. -/
theorem mergeSpanC_ok (b0 : CB) (h0 : CInv b0) (i : Nat) (pairs : List Entry)
    (hn : (mergeSpan (span b0 i) pairs).1.length ≤ 65535) :
    let r := mergeSpanC b0 i pairs
    CInv r.1 ∧ span r.1 i = (mergeSpan (span b0 i) pairs).1 ∧
      (r.2.1, r.2.2) = (mergeSpan (span b0 i) pairs).2 ∧ ∀ j, j ≠ i → span r.1 j = span b0 j := by
  have hg := grow_inv b0 h0 i
  have hi := (grow_slotAt b0 i i).2
  have hframe : ∀ j, span (grow b0 i) j = span b0 j := fun j => by simp only [span, (grow_slotAt b0 i j).1]; rfl
  simp only [mergeSpanC]
  rw [← hframe i] at hn ⊢
  have hm := mergeL_eq (grow b0 i) hg i pairs
  have hcases := mergeL_cases (grow b0 i) i pairs
  generalize grow b0 i = b at hg hi hframe hm hn hcases
  rw [← hm] at hn ⊢
  refine (fun (H : CInv _ ∧ _ ∧ ∀ j, j ≠ i → _) => ⟨H.1, H.2.1, rfl, H.2.2⟩) ?_
  have hfr : ∀ (b' : CB), (∀ j, j ≠ i → span b' j = span b j) → ∀ j, j ≠ i → span b' j = span b0 j :=
    fun b' h j hj => (h j hj).trans (hframe j)
  have hsl := span_length b hg i
  have hlc := len_le_cap b hg i
  generalize mergeL b i pairs = m at hn hcases ⊢
  rcases hcases with ⟨hc, _, rfl⟩ | ⟨hc, _, rfl⟩ | rfl
  · simp only at hn ⊢
    have hl : (patch (span b i) pairs).length = (span b i).length := patch_length _ _
    rw [← inPlace_same b hg i hi _ hl hn]
    have hpos : 1 ≤ (patch (span b i) pairs).length := by
      rw [hl, hsl]; rcases slot_wf b hg i with ⟨h1, _⟩ | ⟨h1, _⟩
      · exact absurd h1 hc
      · exact h1
    obtain ⟨a, c, d⟩ := inPlace_ok b hg i hi _ hpos (by omega) hn
    exact ⟨a, c, hfr _ d⟩
  · simp only at hn ⊢
    have hsub := (pureRemove_sublist (span b i) pairs).length_le
    by_cases h0 : (pureRemove (span b i) pairs).1.length = 0
    · rw [if_pos h0]
      have hfree : setSlot (retireTo b i) i Slot.empty = freeSpanC b i := by
        simp [freeSpanC, hc, retireTo]
      rw [hfree]
      obtain ⟨a, c, d, -⟩ := freeSpanC_ok b hg i hi
      exact ⟨a, by rw [c]; exact (List.eq_nil_of_length_eq_zero h0).symm, hfr _ d⟩
    · rw [if_neg h0]
      obtain ⟨a, c, d⟩ := inPlace_ok b hg i hi _ (by omega) (by omega) hn
      exact ⟨a, c, hfr _ d⟩
  · simp only at hn ⊢
    by_cases h0 : (mergeGen (span b i) pairs).1.length = 0
    · rw [if_pos h0]
      have hnil := (List.eq_nil_of_length_eq_zero h0).symm
      by_cases hc : (slotAt b i).cap ≠ 0
      · rw [if_pos hc]
        have hfree : setSlot (retireTo b i) i Slot.empty = freeSpanC b i := by
          simp [freeSpanC, hc, retireTo]
        rw [hfree]
        obtain ⟨a, c, d, -⟩ := freeSpanC_ok b hg i hi
        exact ⟨a, by rw [c]; exact hnil, hfr _ d⟩
      · rw [if_neg hc]
        obtain ⟨a, c, d⟩ := emptyCap0_ok b hg i hi (by omega)
        exact ⟨a, by rw [c]; exact hnil, hfr _ d⟩
    · rw [if_neg h0]
      by_cases hle : (mergeGen (span b i) pairs).1.length ≤ (slotAt b i).cap
      · rw [if_pos hle]
        obtain ⟨a, c, d⟩ := inPlace_ok b hg i hi _ (by omega) hle hn
        exact ⟨a, c, hfr _ d⟩
      · rw [if_neg hle]
        obtain ⟨a, c, d⟩ := reloc_ok b hg i hi _ (by omega) hn
        exact ⟨a, c, hfr _ d⟩

end PendingCommit.Content
