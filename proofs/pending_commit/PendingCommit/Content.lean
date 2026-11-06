import PendingCommit.Span
import PendingCommit.Arena
/-
# `Block` with contents: what each span reads after `set_span` / `free_span`

`Arena.lean` tracks a block's counters with contents abstracted; this file adds
the arena *contents* (entries as `(attr_id, value)`, i.e. after `unpack`; the
byte packing is `Pack.lean`) and proves the reads:

* `span b i` = `arena[off .. off+len]` of slot `i` (`SpanRef::entries`, :845).
* `CInv` = `Arena.Inv` (counters, in-bounds, `1 ≤ len ≤ cap`) + live spans
  pairwise apart in the arena.
* `setSpanC_ok` — `set_span` (:573) of `n ≤ 65535` pairs: slot `i` then reads
  exactly `pairs`, every other slot reads what it read before, `CInv` holds.
* `freeSpanC_ok` — `free_span` (:616): slot `i` reads `[]`, others unchanged.
* `grow_slotAt` — `grow_slots` (:560) only appends empty slots.
* `setSpan_u16_wrap` — the `n as u16` (:597, :608): 65,536 pairs leave a
  slot reading `[]` while the arena grew by 65,536 entries no counter covers,
  i.e. `Arena.Inv` breaks (and `compact`'s `debug_assert_eq!` would fire).
  Reachable only with ≥ 65,536 entries for one entity (`import_attrs_resolved`
  / `decode_with_count` do not deduplicate); `setSpan_cast_exact` shows the cast
  is exact whenever ids are strictly ascending and below `ATTRIBUTE_ID_NONE`.
-/
namespace PendingCommit.Content
open Arena (Slot)

structure CB where
  slots : List Slot
  arena : List Entry
  dead : Nat
  slack : Nat

def toBlk (b : CB) : Arena.Blk := ⟨b.slots, b.arena.length, b.dead, b.slack⟩
def slotAt (b : CB) (i : Nat) : Slot := b.slots.getD i Slot.empty
def spanOf (a : List Entry) (s : Slot) : List Entry := (a.drop s.off).take s.len
def span (b : CB) (i : Nat) : List Entry := spanOf b.arena (slotAt b i)

def Apart (a c : Slot) : Prop := a.cap = 0 ∨ c.cap = 0 ∨ a.off + a.cap ≤ c.off ∨ c.off + c.cap ≤ a.off

structure CInv (b : CB) : Prop where
  arena : Arena.Inv (toBlk b)
  disj : ∀ i j, i ≠ j → Apart (slotAt b i) (slotAt b j)

/-- `n as u16`. -/
def u16 (n : Nat) : Nat := n % 65536

/-- In-place overwrite of `L.len` entries from `off` (the `store_packed_value`
loop with `index < arena.len()`). -/
def writeAt (a : List Entry) (off : Nat) (L : List Entry) : List Entry :=
  a.take off ++ L ++ a.drop (off + L.length)

def grow (b : CB) (i : Nat) : CB :=
  { b with slots := b.slots ++ List.replicate (i + 1 - b.slots.length) Slot.empty }

def setSlot (b : CB) (i : Nat) (s : Slot) : CB := { b with slots := b.slots.set i s }

/-- `Block::free_span` (:616); `release_span_values` is `Pack.releaseSpan`. -/
def freeSpanC (b : CB) (i : Nat) : CB :=
  let s := slotAt b i
  if s.cap = 0 then b
  else setSlot { b with dead := b.dead + s.cap, slack := b.slack - (s.cap - s.len) } i Slot.empty

/-- `Block::set_span` (:573). -/
def setSpanC (b0 : CB) (i : Nat) (pairs : List Entry) : CB :=
  let b := grow b0 i
  if pairs = [] then freeSpanC b i
  else
    let old := slotAt b i
    let n := pairs.length
    if n ≤ old.cap then
      setSlot { b with arena := writeAt b.arena old.off pairs, slack := b.slack + old.len - n } i
        ⟨old.off, u16 n, old.cap⟩
    else
      setSlot { b with arena := b.arena ++ pairs, dead := b.dead + old.cap,
                       slack := b.slack - (old.cap - old.len) } i
        ⟨b.arena.length, u16 n, u16 n⟩

/-! ## List lemmas -/

theorem spanOf_getElem? (a : List Entry) (s : Slot) (k : Nat) :
    (spanOf a s)[k]? = if k < s.len then a[s.off + k]? else none := by
  unfold spanOf
  rw [List.getElem?_take]
  split <;> simp [List.getElem?_drop]

theorem writeAt_length (a : List Entry) (off : Nat) (L : List Entry) (h : off + L.length ≤ a.length) :
    (writeAt a off L).length = a.length := by
  simp [writeAt]; omega

theorem writeAt_getElem? (a : List Entry) (off : Nat) (L : List Entry) (h : off + L.length ≤ a.length)
    (k : Nat) : (writeAt a off L)[k]? = if off ≤ k ∧ k < off + L.length then L[k - off]? else a[k]? := by
  unfold writeAt
  have ht : (a.take off).length = off := by simp; omega
  by_cases h1 : k < off
  · rw [List.getElem?_append_left (by simp; omega), List.getElem?_append_left (by omega),
      List.getElem?_take_of_lt h1, if_neg (by omega)]
  · by_cases h2 : k < off + L.length
    · rw [List.getElem?_append_left (by simp; omega), List.getElem?_append_right (by omega), ht,
        if_pos ⟨by omega, h2⟩]
    · rw [List.getElem?_append_right (by simp; omega), if_neg (by omega), List.getElem?_drop]
      simp only [List.length_append, ht]; congr 1; omega

theorem spanOf_writeAt_frame (a : List Entry) (off : Nat) (L : List Entry) (h : off + L.length ≤ a.length)
    (s : Slot) (hd : s.len = 0 ∨ s.off + s.len ≤ off ∨ off + L.length ≤ s.off) :
    spanOf (writeAt a off L) s = spanOf a s := by
  apply List.ext_getElem?; intro k
  rw [spanOf_getElem?, spanOf_getElem?]
  split
  · rw [writeAt_getElem? a off L h]; rw [if_neg (by omega)]
  · rfl

theorem spanOf_writeAt_self (a : List Entry) (off : Nat) (L : List Entry) (h : off + L.length ≤ a.length) :
    spanOf (writeAt a off L) ⟨off, L.length, c⟩ = L := by
  apply List.ext_getElem?; intro k
  rw [spanOf_getElem?]; simp only
  split
  · rw [writeAt_getElem? a off L h, if_pos (by omega)]; congr 1; omega
  · rw [List.getElem?_eq_none (by omega)]

theorem spanOf_append_frame (a L : List Entry) (s : Slot) (h : s.len = 0 ∨ s.off + s.len ≤ a.length) :
    spanOf (a ++ L) s = spanOf a s := by
  apply List.ext_getElem?; intro k
  rw [spanOf_getElem?, spanOf_getElem?]
  split
  · rw [List.getElem?_append_left (by omega)]
  · rfl

theorem spanOf_append_self (a L : List Entry) (c : Nat) :
    spanOf (a ++ L) ⟨a.length, L.length, c⟩ = L := by
  simp [spanOf]

theorem getD_set (l : List Slot) (i j : Nat) (x : Slot) (hi : i < l.length) :
    (l.set i x).getD j Slot.empty = if j = i then x else l.getD j Slot.empty := by
  simp only [List.getD_eq_getElem?_getD]
  by_cases h : j = i
  · subst h; simp [hi]
  · simp [h, List.getElem?_set_ne (Ne.symm h)]

theorem slotAt_set (b : CB) (i j : Nat) (x : Slot) (hi : i < b.slots.length) :
    slotAt (setSlot b i x) j = if j = i then x else slotAt b j := getD_set _ _ _ _ hi

@[simp] theorem slotAt_with (b : CB) (a : List Entry) (d s j : Nat) :
    slotAt { b with arena := a, dead := d, slack := s } j = slotAt b j := rfl
@[simp] theorem slotAt_with' (b : CB) (d s j : Nat) :
    slotAt { b with dead := d, slack := s } j = slotAt b j := rfl
@[simp] theorem slotAt_with'' (b : CB) (a : List Entry) (s j : Nat) :
    slotAt { b with arena := a, slack := s } j = slotAt b j := rfl

theorem aslot (b : CB) (i : Nat) : Arena.slotAt (toBlk b) i = slotAt b i := rfl

@[simp] theorem setSlot_arena (b : CB) (i : Nat) (x : Slot) : (setSlot b i x).arena = b.arena := rfl
@[simp] theorem setSlot_slots (b : CB) (i : Nat) (x : Slot) : (setSlot b i x).slots = b.slots.set i x := rfl
theorem slotAt_mk_set (sl : List Slot) (a : List Entry) (d s i j : Nat) (x : Slot) :
    slotAt (setSlot (CB.mk sl a d s) i x) j = (sl.set i x).getD j Slot.empty := rfl
theorem slotAt_def (b : CB) (j : Nat) : slotAt b j = b.slots.getD j Slot.empty := rfl

/-- **`grow_slotAt`**: `grow_slots` only appends empty slots and covers `i`. -/
theorem grow_slotAt (b : CB) (i j : Nat) : slotAt (grow b i) j = slotAt b j ∧ i < (grow b i).slots.length := by
  constructor
  · simp only [slotAt, grow, List.getD_eq_getElem?_getD]
    rw [List.getElem?_append]
    split
    · rfl
    · rename_i h
      rw [List.getElem?_eq_none (by omega : b.slots.length ≤ j)]
      simp [List.getElem?_replicate]; split <;> rfl
  · simp [grow]; omega

theorem toBlk_grow (b : CB) (i : Nat) : toBlk (grow b i) = Arena.grow (toBlk b) i := rfl

theorem grow_inv (b : CB) (h : CInv b) (i : Nat) : CInv (grow b i) :=
  ⟨by rw [toBlk_grow]; exact (Arena.inv_grow h.arena i).1,
   fun x y hxy => by rw [(grow_slotAt b i x).1, (grow_slotAt b i y).1]; exact h.disj x y hxy⟩

theorem slot_wf (b : CB) (h : CInv b) (j : Nat) : Arena.SlotWF (slotAt b j) := by
  by_cases hj : j < b.slots.length
  · exact Arena.slot_wf h.arena hj
  · simp [slotAt, List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega : b.slots.length ≤ j),
      Arena.SlotWF, Slot.empty]

theorem slot_inb (b : CB) (h : CInv b) (j : Nat) :
    (slotAt b j).off + (slotAt b j).cap ≤ b.arena.length ∨ (slotAt b j).cap = 0 := by
  by_cases hj : j < b.slots.length
  · exact Arena.slot_inb h.arena hj
  · right; simp [slotAt, List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega : b.slots.length ≤ j),
      Slot.empty]

theorem apart_symm {a c : Slot} (h : Apart a c) : Apart c a := by
  unfold Apart at *; omega

theorem len_le_cap (b : CB) (h : CInv b) (j : Nat) : (slotAt b j).len ≤ (slotAt b j).cap := by
  rcases slot_wf b h j with ⟨_, _⟩ | ⟨_, _⟩ <;> omega

theorem cap0_len0 (b : CB) (h : CInv b) (j : Nat) (hc : (slotAt b j).cap = 0) : (slotAt b j).len = 0 := by
  have := len_le_cap b h j; omega

/-- Disjointness after replacing slot `i` by `x`, given `x` is apart from every other slot. -/
theorem disj_set (b : CB) (h : CInv b) (i : Nat) (hi : i < b.slots.length) (x : Slot)
    (hx : ∀ j, j ≠ i → Apart x (slotAt b j)) (b' : CB) (hs : b'.slots = b.slots.set i x) :
    ∀ p q, p ≠ q → Apart (slotAt b' p) (slotAt b' q) := by
  have e : ∀ j, slotAt b' j = if j = i then x else slotAt b j := by
    intro j; have := slotAt_set b i j x hi; simp only [slotAt, setSlot] at this ⊢; rw [hs]; exact this
  intro p q hpq
  rw [e p, e q]
  by_cases hp : p = i
  · subst hp; rw [if_pos rfl, if_neg (Ne.symm hpq)]; exact hx q (Ne.symm hpq)
  · by_cases hq : q = i
    · subst hq; rw [if_neg hp, if_pos rfl]; exact apart_symm (hx p hp)
    · rw [if_neg hp, if_neg hq]; exact h.disj p q hpq

/-- **`freeSpanC_ok`**: `free_span` empties slot `i`, leaves every other span
as it was, and keeps `CInv`. -/
theorem freeSpanC_ok (b : CB) (h : CInv b) (i : Nat) (hi : i < b.slots.length) :
    CInv (freeSpanC b i) ∧ span (freeSpanC b i) i = [] ∧
      (∀ j, j ≠ i → span (freeSpanC b i) j = span b j) ∧ (freeSpanC b i).slots.length = b.slots.length := by
  unfold freeSpanC
  by_cases hc : (slotAt b i).cap = 0
  · rw [if_pos hc]
    exact ⟨h, by simp [span, spanOf, cap0_len0 b h i hc], fun _ _ => rfl, rfl⟩
  · rw [if_neg hc]
    have hsl := Arena.slot_slack_le h.arena hi
    have ha := h.arena.arena_eq
    have hs := h.arena.slack_eq
    rw [aslot] at hsl
    simp only [toBlk] at hsl ha hs
    have hlc := len_le_cap b h i
    refine ⟨⟨?_, ?_⟩, ?_, ?_, by simp [setSlot]⟩
    · have := Arena.inv_setSlot h.arena hi (x := Slot.empty) (arena := b.arena.length)
        (dead := b.dead + (slotAt b i).cap) (slack := b.slack - ((slotAt b i).cap - (slotAt b i).len))
        (by rw [aslot]; simp only [toBlk, Slot.empty]; omega)
        (by rw [aslot]; simp only [toBlk, Slot.empty]; omega)
        (.inl ⟨rfl, rfl⟩) (.inr rfl) (Nat.le_refl _)
      exact this
    · exact disj_set b h i hi Slot.empty (fun _ _ => .inl rfl) _ rfl
    · simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, ite_true]
      simp [spanOf, Slot.empty]
    · intro j hj
      simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, hj, ite_false]; rfl

theorem u16_id {n : Nat} (h : n ≤ 65535) : u16 n = n := by unfold u16; omega

/-- **`setSpanC_ok`**: after `set_span` of `n ≤ 65535` pairs, slot `i` reads
exactly `pairs` and every other slot reads what it did. -/
theorem setSpanC_ok (b0 : CB) (h0 : CInv b0) (i : Nat) (pairs : List Entry) (hn : pairs.length ≤ 65535) :
    CInv (setSpanC b0 i pairs) ∧ span (setSpanC b0 i pairs) i = pairs ∧
      ∀ j, j ≠ i → span (setSpanC b0 i pairs) j = span b0 j := by
  have hg := grow_inv b0 h0 i
  have hi := (grow_slotAt b0 i i).2
  have hframe : ∀ j, span (grow b0 i) j = span b0 j := fun j => by simp only [span, (grow_slotAt b0 i j).1]; rfl
  unfold setSpanC
  simp only
  generalize grow b0 i = b at hg hi hframe
  by_cases hp : pairs = []
  · subst hp; simp only [ite_true]
    obtain ⟨a1, a2, a3, -⟩ := freeSpanC_ok b hg i hi
    exact ⟨a1, a2, fun j hj => (a3 j hj).trans (hframe j)⟩
  · simp only [hp, ite_false]
    have hpos : 1 ≤ pairs.length := by cases pairs; exact absurd rfl hp; simp
    have hsl := Arena.slot_slack_le hg.arena hi
    have ha := hg.arena.arena_eq
    have hs := hg.arena.slack_eq
    have hlc := len_le_cap b hg i
    have hinb := slot_inb b hg i
    rw [u16_id hn]
    by_cases hle : pairs.length ≤ (slotAt b i).cap
    · simp only [hle, ite_true]
      have hin : (slotAt b i).off + pairs.length ≤ b.arena.length := by rcases hinb with h1 | h1 <;> omega
      have hwl := writeAt_length b.arena (slotAt b i).off pairs hin
      refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
      · have := Arena.inv_setSlot hg.arena hi (x := ⟨(slotAt b i).off, pairs.length, (slotAt b i).cap⟩)
          (arena := b.arena.length) (dead := b.dead)
          (slack := b.slack + (slotAt b i).len - pairs.length)
          (by simp only [toBlk, Arena.slotAt, slotAt] at *; omega)
          (by simp only [toBlk, Arena.slotAt, slotAt] at *; omega)
          (.inr ⟨hpos, hle⟩) (by rcases hinb with h1 | h1 <;> simp_all <;> omega) (Nat.le_refl _)
        simp only [toBlk, setSlot, hwl] at this ⊢; exact this
      · exact disj_set b hg i hi _ (fun j hj => by
          have := hg.disj i j (Ne.symm hj); unfold Apart at this ⊢; simpa using this) _ rfl
      · simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, ite_true]
        exact spanOf_writeAt_self _ _ _ hin
      · intro j hj
        rw [← hframe j]
        simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, hj, ite_false]
        rw [← slotAt_def]
        apply spanOf_writeAt_frame _ _ _ hin
        have hap := hg.disj i j (Ne.symm hj)
        have := len_le_cap b hg j
        unfold Apart at hap; omega
    · simp only [hle, ite_false]
      refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
      · have := Arena.inv_setSlot hg.arena hi (x := ⟨b.arena.length, pairs.length, pairs.length⟩)
          (arena := b.arena.length + pairs.length) (dead := b.dead + (slotAt b i).cap)
          (slack := b.slack - ((slotAt b i).cap - (slotAt b i).len))
          (by simp only [toBlk, Arena.slotAt, slotAt] at *; omega)
          (by simp only [toBlk, Arena.slotAt, slotAt] at *; omega)
          (.inr ⟨hpos, Nat.le_refl _⟩) (.inl (by simp)) (by simp [toBlk])
        simp only [toBlk, setSlot, List.length_append] at this ⊢; exact this
      · exact disj_set b hg i hi _ (fun j _ => by
          have := slot_inb b hg j; unfold Apart; simp only; omega) _ rfl
      · simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, ite_true]
        exact spanOf_append_self _ _ _
      · intro j hj
        rw [← hframe j]
        simp only [span, setSlot_arena, slotAt_mk_set, getD_set _ _ _ _ hi, hj, ite_false]
        rw [← slotAt_def]
        apply spanOf_append_frame
        have := slot_inb b hg j; have := len_le_cap b hg j; omega

/-- **`setSpan_u16_wrap`**: 65,536 pairs into an empty block: the slot reads as
having no attributes (`len = cap = 65536 as u16 = 0`) while the arena holds
65,536 entries that `dead + Σcap` (= 0) does not account for, so
`Arena.Inv` is false afterwards. -/
theorem setSpan_u16_wrap (pairs : List Entry) (hn : pairs.length = 65536) :
    let b := setSpanC ⟨[], [], 0, 0⟩ 0 pairs
    span b 0 = [] ∧ b.arena.length = 65536 ∧ ¬ Arena.Inv (toBlk b) := by
  have hp : pairs ≠ [] := by intro h; subst h; simp at hn
  simp only [setSpanC, hp, ite_false, grow, slotAt]
  simp [Slot.empty, hn, u16]
  refine ⟨by simp [span, slotAt, setSlot, spanOf], ?_⟩
  intro h
  have := h.arena_eq
  simp [toBlk, setSlot, Arena.sumBy, hn] at this

/-- **`setSpan_cast_exact`**: strictly ascending ids that are all `< 65535`
(`ATTRIBUTE_ID_NONE` excluded) number at most 65,535, so `n as u16 = n`. -/
theorem setSpan_cast_exact (pairs : List Entry) (hs : Sorted pairs) (hb : ∀ e ∈ pairs, e.1 < 65535) :
    pairs.length ≤ 65535 ∧ u16 pairs.length = pairs.length := by
  have key : ∀ (l : List Entry) (lo : Nat), lo ≤ 65535 → Sorted l →
      (∀ e ∈ l, lo ≤ e.1 ∧ e.1 < 65535) → l.length + lo ≤ 65535 := by
    intro l
    induction l with
    | nil => intro lo h _ _; simp; omega
    | cons e es ih =>
      intro lo hlo hs h
      have he := h e (by simp)
      have := ih (e.1 + 1) (by omega) (List.Pairwise.of_cons hs) (fun x hx =>
        ⟨Nat.succ_le_of_lt (List.rel_of_pairwise_cons hs hx), (h x (by simp [hx])).2⟩)
      simp; omega
  have := key pairs 0 (by omega) hs (fun e he => ⟨Nat.zero_le _, hb e he⟩)
  exact ⟨by omega, u16_id (by omega)⟩

end PendingCommit.Content
