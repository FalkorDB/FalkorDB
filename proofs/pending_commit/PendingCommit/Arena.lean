/-
# `Block` arena accounting: `dead`, `slack`, spans, `compact`

`graph/src/graph/attribute_store.rs`. A `Block` keeps one `Slot {offset, len,
cap}` per entity into a bump-allocated `arena`, plus two counters:

* `dead`  — arena entries no span references any more (`retire_span`, :535);
* `slack` — reserved-but-unused entries inside live spans (`cap - len`).

`compact` (:798) trusts them: `live = arena.len() - dead - slack` and then
`debug_assert_eq!(new_arena.len(), live)`. The counters are `u32` and are
updated with plain `-`, which panics in debug and wraps in release, so an
accounting slip anywhere is either a crash or a wrong compaction trigger.

Here the entry *contents* are abstracted away (they are `Span.lean`'s job) and
each operation is modelled as its effect on the slot table and counters, with
every Rust subtraction a checked one (`none` = the u32 subtraction would
underflow). The theorem is that from any well-formed block every operation
succeeds (never underflows) and yields a well-formed block, and that `live`
equals the sum of live span lengths, which is exactly the `debug_assert`.
-/
namespace PendingCommit.Arena

structure Slot where
  off : Nat
  len : Nat
  cap : Nat
  deriving DecidableEq, Repr

def Slot.empty : Slot := ⟨0, 0, 0⟩

structure Blk where
  slots : List Slot
  arena : Nat
  dead : Nat
  slack : Nat
  deriving Repr

def sumBy (f : Slot → Nat) (l : List Slot) : Nat := (l.map f).sum

/-- `cap == 0` means "no span"; a live span has `1 <= len <= cap` (the doc
comment on `Slot`, attribute_store.rs:319-321). -/
def SlotWF (s : Slot) : Prop := (s.cap = 0 ∧ s.len = 0) ∨ (1 ≤ s.len ∧ s.len ≤ s.cap)

structure Inv (b : Blk) : Prop where
  arena_eq : b.arena = b.dead + sumBy (·.cap) b.slots
  slack_eq : b.slack = sumBy (fun s => s.cap - s.len) b.slots
  wf : ∀ s ∈ b.slots, SlotWF s
  inb : ∀ s ∈ b.slots, s.off + s.cap ≤ b.arena ∨ s.cap = 0

/-- Checked u32 subtraction. -/
def csub (a b : Nat) : Option Nat := if b ≤ a then some (a - b) else none

/-- `retire_span` (:535): `dead += cap; slack -= cap - len`. -/
def retire (b : Blk) (s : Slot) : Option Blk :=
  (csub b.slack (s.cap - s.len)).map fun sl => { b with dead := b.dead + s.cap, slack := sl }

/-- `resize_span_slack` (:548): `slack = slack + old_len - new_len`. -/
def resize (b : Blk) (oldLen newLen : Nat) : Option Blk :=
  (csub (b.slack + oldLen) newLen).map fun sl => { b with slack := sl }

def setSlot (b : Blk) (i : Nat) (s : Slot) : Blk := { b with slots := b.slots.set i s }

def slotAt (b : Blk) (i : Nat) : Slot := b.slots.getD i Slot.empty

/-- `grow_slots` (:560): extend with empty slots to cover `i`. -/
def grow (b : Blk) (i : Nat) : Blk :=
  { b with slots := b.slots ++ List.replicate (i + 1 - b.slots.length) Slot.empty }

/-- `Block::free_span` (:616). -/
def freeSpan (b : Blk) (i : Nat) : Option Blk :=
  let s := slotAt b i
  if s.cap = 0 then some b else (retire b s).map (setSlot · i Slot.empty)

/-- `Block::set_span` (:573), for `n` = the number of pairs written. -/
def setSpan (b0 : Blk) (i n : Nat) : Option Blk :=
  let b := grow b0 i
  if n = 0 then freeSpan b i
  else
    let old := slotAt b i
    if n ≤ old.cap then
      (resize b old.len n).map (setSlot · i ⟨old.off, n, old.cap⟩)
    else
      (retire b old).map fun b' =>
        setSlot { b' with arena := b'.arena + n } i ⟨b'.arena, n, n⟩

/-- `Block::merge_span` (:636), for the path taken and the merged length `n`.
`path = 0`: fast path 1 (len unchanged); `1`: pure removal; `2`: general. -/
def mergeSpan (b0 : Blk) (i : Nat) (path n : Nat) : Option Blk :=
  let b := grow b0 i
  let old := slotAt b i
  match path with
  | 0 => some b
  | 1 =>
      if old.cap = 0 then some b  -- not reached: fast paths need a span
      else if n = 0 then (retire b old).map (setSlot · i Slot.empty)
      else (resize b old.len n).map fun b' => setSlot b' i { old with len := n }
  | _ =>
      if n = 0 then
        if old.cap ≠ 0 then (retire b old).map (setSlot · i Slot.empty) else some (setSlot b i Slot.empty)
      else if n ≤ old.cap then
        (resize b old.len n).map (setSlot · i ⟨old.off, n, old.cap⟩)
      else
        let r := if old.cap ≠ 0 then retire b old else some b
        r.map fun b' => setSlot { b' with arena := b'.arena + n } i ⟨b'.arena, n, n⟩

/-- Offsets `compact` assigns: live spans packed back to back, in slot order. -/
def packOffsets : Nat → List Slot → List Slot
  | _, [] => []
  | o, s :: ss => if s.cap = 0 then s :: packOffsets o ss else ⟨o, s.len, s.len⟩ :: packOffsets (o + s.len) ss

/-- `Block::compact` (:798). `none` if `live` underflows. -/
def compact (b : Blk) : Option Blk :=
  (csub b.arena b.dead).bind fun a => (csub a b.slack).map fun _ =>
    { slots := packOffsets 0 b.slots, arena := sumBy (·.len) b.slots, dead := 0, slack := 0 }

/-- `live` as `compact` computes it. -/
def live (b : Blk) : Nat := b.arena - b.dead - b.slack

/-! ## Sum lemmas -/

theorem sumBy_cons (f : Slot → Nat) (s : Slot) (l : List Slot) :
    sumBy f (s :: l) = f s + sumBy f l := by simp [sumBy]

theorem sumBy_append (f : Slot → Nat) (a c : List Slot) :
    sumBy f (a ++ c) = sumBy f a + sumBy f c := by simp [sumBy, List.sum_append]

theorem sumBy_set (f : Slot → Nat) : ∀ (l : List Slot) (i : Nat) (x : Slot), i < l.length →
    sumBy f (l.set i x) + f (l.getD i Slot.empty) = sumBy f l + f x
  | [], _, _, h => by simp at h
  | s :: l, 0, x, _ => by simp [sumBy_cons]; omega
  | s :: l, i + 1, x, h => by
      have := sumBy_set f l i x (by simpa using h)
      simp [sumBy_cons] at this ⊢; omega

theorem getD_mem : ∀ {l : List Slot} {i : Nat}, i < l.length → l.getD i Slot.empty ∈ l
  | [], _, h => by simp at h
  | x :: xs, 0, _ => by simp
  | x :: xs, i + 1, h => by
      have := getD_mem (l := xs) (i := i) (by simpa using h)
      simp at this ⊢; exact .inr this

theorem le_sumBy {f : Slot → Nat} {l : List Slot} {s : Slot} (h : s ∈ l) : f s ≤ sumBy f l := by
  induction l with
  | nil => simp at h
  | cons x xs ih =>
    rw [sumBy_cons]; simp at h
    rcases h with rfl | h
    · omega
    · have := ih h; omega

theorem mem_set_cases {l : List Slot} {i : Nat} {x s : Slot} (h : s ∈ l.set i x) : s = x ∨ s ∈ l := by
  induction l generalizing i with
  | nil => simp at h
  | cons y ys ih =>
    cases i with
    | zero => simp at h; rcases h with rfl | h; exact .inl rfl; exact .inr (by simp [h])
    | succ i =>
      simp at h; rcases h with rfl | h
      · exact .inr (by simp)
      · rcases ih h with h | h; exact .inl h; exact .inr (by simp [h])

/-! ## Invariant preservation -/

theorem inv_grow {b : Blk} (h : Inv b) (i : Nat) : Inv (grow b i) ∧ i < (grow b i).slots.length := by
  have hr : ∀ f : Slot → Nat, f Slot.empty = 0 →
      sumBy f (List.replicate (i + 1 - b.slots.length) Slot.empty) = 0 := by
    intro f hf; simp [sumBy, List.map_replicate, hf]
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · simp only [grow]; rw [sumBy_append, hr _ rfl, h.arena_eq]; omega
  · simp only [grow]; rw [sumBy_append, hr _ rfl, h.slack_eq]; omega
  · intro s hs; simp [grow] at hs; rcases hs with hs | ⟨_, rfl⟩
    · exact h.wf s hs
    · exact .inl ⟨rfl, rfl⟩
  · intro s hs; simp [grow] at hs; rcases hs with hs | ⟨_, rfl⟩
    · exact h.inb s hs
    · exact .inr rfl
  · simp [grow]; omega

/-- Replacing slot `i` by `x`, with the counters adjusted by the caller. -/
theorem inv_setSlot {b : Blk} (h : Inv b) {i : Nat} (hi : i < b.slots.length) {x : Slot}
    {arena dead slack : Nat}
    (hA : arena + (slotAt b i).cap = dead + sumBy (·.cap) b.slots + x.cap)
    (hS : slack + ((slotAt b i).cap - (slotAt b i).len) =
      sumBy (fun s => s.cap - s.len) b.slots + (x.cap - x.len))
    (hwf : SlotWF x) (hinb : x.off + x.cap ≤ arena ∨ x.cap = 0) (hgrow : b.arena ≤ arena) :
    Inv { slots := b.slots.set i x, arena := arena, dead := dead, slack := slack } := by
  have e1 := sumBy_set (·.cap) b.slots i x hi
  have e2 := sumBy_set (fun s => s.cap - s.len) b.slots i x hi
  simp only [slotAt] at hA hS
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only; omega
  · simp only; omega
  · intro s hs'; rcases mem_set_cases hs' with rfl | hs'
    · exact hwf
    · exact h.wf s hs'
  · intro s hs'; rcases mem_set_cases hs' with rfl | hs'
    · exact hinb
    · rcases h.inb s hs' with h1 | h1
      · exact .inl (by dsimp only; omega)
      · exact .inr h1

theorem slot_slack_le {b : Blk} (h : Inv b) {i : Nat} (hi : i < b.slots.length) :
    (slotAt b i).cap - (slotAt b i).len ≤ b.slack := by
  rw [h.slack_eq]; exact le_sumBy (f := fun s => s.cap - s.len) (getD_mem hi)

theorem slot_wf {b : Blk} (h : Inv b) {i : Nat} (hi : i < b.slots.length) : SlotWF (slotAt b i) :=
  h.wf _ (getD_mem hi)

theorem slot_inb {b : Blk} (h : Inv b) {i : Nat} (hi : i < b.slots.length) :
    (slotAt b i).off + (slotAt b i).cap ≤ b.arena ∨ (slotAt b i).cap = 0 :=
  h.inb _ (getD_mem hi)

theorem retire_ok {b : Blk} (h : Inv b) {i : Nat} (hi : i < b.slots.length) :
    ∃ b', retire b (slotAt b i) = some b' ∧ b'.slots = b.slots ∧ b'.arena = b.arena ∧
      b'.dead = b.dead + (slotAt b i).cap ∧ b'.slack + ((slotAt b i).cap - (slotAt b i).len) = b.slack := by
  have := slot_slack_le h hi
  refine ⟨{ b with dead := b.dead + (slotAt b i).cap,
                   slack := b.slack - ((slotAt b i).cap - (slotAt b i).len) }, ?_, rfl, rfl, rfl, ?_⟩
  · simp [retire, csub, this]
  · simp only; omega

/-- Close the counter side-conditions of `inv_setSlot`. -/
macro "arith" : tactic => `(tactic| (first | omega | (dsimp only; omega) | (simp only [Slot.empty] at *; omega) | (dsimp only at *; simp only [Slot.empty] at *; omega)))

/-- **`freeSpan_ok`**: `free_span` never underflows and keeps the block well-formed. -/
theorem freeSpan_ok {b : Blk} (h : Inv b) {i : Nat} (hi : i < b.slots.length) :
    ∃ b', freeSpan b i = some b' ∧ Inv b' ∧ b'.slots.length = b.slots.length := by
  have ha := h.arena_eq; have hs0 := h.slack_eq
  unfold freeSpan
  by_cases hc : (slotAt b i).cap = 0
  · simp only [hc, ite_true]; exact ⟨b, rfl, h, rfl⟩
  · simp only [hc, ite_false]
    obtain ⟨b', hr, hsl, har, hd, hs⟩ := retire_ok h hi
    rw [hr]
    refine ⟨_, rfl, ?_, by simp [setSlot, hsl]⟩
    simp only [setSlot, Option.map_some]
    rw [hsl]
    exact inv_setSlot h hi (by arith) (by arith) (.inl ⟨rfl, rfl⟩) (.inr rfl) (by arith)

/-- **`setSpan_ok`**: `set_span` never underflows and keeps the block well-formed. -/
theorem setSpan_ok {b0 : Blk} (h0 : Inv b0) (i n : Nat) :
    ∃ b', setSpan b0 i n = some b' ∧ Inv b' := by
  obtain ⟨h, hi⟩ := inv_grow h0 i
  unfold setSpan
  simp only
  generalize grow b0 i = b at h hi ⊢
  have ha := h.arena_eq; have hs0 := h.slack_eq
  by_cases hn : n = 0
  · simp only [hn, ite_true]; obtain ⟨b', e, hb', _⟩ := freeSpan_ok h hi; exact ⟨b', e, hb'⟩
  · simp only [hn, ite_false]
    have hwf := slot_wf h hi
    have hinb := slot_inb h hi
    have hsl := slot_slack_le h hi
    have hlc : (slotAt b i).len ≤ (slotAt b i).cap := by rcases hwf with ⟨_, h2⟩ | ⟨_, h2⟩ <;> omega
    by_cases hle : n ≤ (slotAt b i).cap
    · simp only [hle, ite_true, resize, csub]
      have : n ≤ b.slack + (slotAt b i).len := by
        rcases hwf with ⟨h1, _⟩ | ⟨_, _⟩ <;> omega
      simp only [this, ite_true, Option.map_some]
      refine ⟨_, rfl, ?_⟩
      exact inv_setSlot h hi (by arith) (by arith) (.inr ⟨by arith, hle⟩) hinb (Nat.le_refl _)
    · simp only [hle, ite_false]
      obtain ⟨b', hr, hsl', har, hd, hs⟩ := retire_ok h hi
      rw [hr]; simp only [Option.map_some]
      refine ⟨_, rfl, ?_⟩
      simp only [setSlot]; rw [hsl']
      exact inv_setSlot h hi (by arith) (by arith)
        (.inr ⟨by arith, Nat.le_refl _⟩) (.inl (by arith)) (by arith)

/-- **`mergeSpan_ok`**: every path of `merge_span` keeps the block well-formed
and never underflows, for any merged length `n` the path can produce
(pure removal only shrinks: `n ≤ old.len`). -/
theorem mergeSpan_ok {b0 : Blk} (h0 : Inv b0) (i path n : Nat)
    (hpure : path = 1 → n ≤ (slotAt (grow b0 i) i).len) :
    ∃ b', mergeSpan b0 i path n = some b' ∧ Inv b' := by
  obtain ⟨h, hi⟩ := inv_grow h0 i
  unfold mergeSpan
  simp only
  generalize hb : grow b0 i = b at h hi hpure ⊢
  have ha := h.arena_eq; have hs0 := h.slack_eq
  have hwf := slot_wf h hi
  have hinb := slot_inb h hi
  have hsl := slot_slack_le h hi
  have hlc : (slotAt b i).len ≤ (slotAt b i).cap := by rcases hwf with ⟨_, h2⟩ | ⟨_, h2⟩ <;> omega
  have setEmpty : ∀ b', b'.slots = b.slots → b'.arena = b.arena → b'.dead = b.dead + (slotAt b i).cap →
      b'.slack + ((slotAt b i).cap - (slotAt b i).len) = b.slack → Inv (setSlot b' i Slot.empty) := by
    intro b' e1 e2 e3 e4
    simp only [setSlot]; rw [e1]
    exact inv_setSlot h hi (by arith) (by arith) (.inl ⟨rfl, rfl⟩) (.inr rfl) (by arith)
  match path with
  | 0 => exact ⟨b, rfl, h⟩
  | 1 =>
    have hp := hpure rfl
    simp only
    by_cases hc : (slotAt b i).cap = 0
    · simp only [hc, ite_true]; exact ⟨b, rfl, h⟩
    · simp only [hc, ite_false]
      by_cases hn : n = 0
      · simp only [hn, ite_true]
        obtain ⟨b', hr, e1, e2, e3, e4⟩ := retire_ok h hi
        rw [hr]; exact ⟨_, rfl, setEmpty b' e1 e2 e3 e4⟩
      · simp only [hn, ite_false, resize, csub]
        have : n ≤ b.slack + (slotAt b i).len := by omega
        simp only [this, ite_true, Option.map_some]
        refine ⟨_, rfl, ?_⟩
        exact inv_setSlot h hi (by arith) (by arith)
          (.inr ⟨by arith, by dsimp only; omega⟩) hinb (Nat.le_refl _)
  | k + 2 =>
    simp only
    by_cases hn : n = 0
    · simp only [hn, ite_true]
      by_cases hc : (slotAt b i).cap = 0
      · simp only [hc, ne_eq, not_true_eq_false, ite_false]
        refine ⟨_, rfl, ?_⟩
        simp only [setSlot]
        have : (slotAt b i).len = 0 := by rcases hwf with ⟨_, h2⟩ | ⟨_, _⟩ <;> omega
        exact inv_setSlot h hi (by arith) (by arith) (.inl ⟨rfl, rfl⟩) (.inr rfl) (Nat.le_refl _)
      · simp only [hc, ne_eq, not_false_eq_true, ite_true]
        obtain ⟨b', hr, e1, e2, e3, e4⟩ := retire_ok h hi
        rw [hr]; exact ⟨_, rfl, setEmpty b' e1 e2 e3 e4⟩
    · simp only [hn, ite_false]
      by_cases hle : n ≤ (slotAt b i).cap
      · simp only [hle, ite_true, resize, csub]
        have : n ≤ b.slack + (slotAt b i).len := by
          rcases hwf with ⟨h1, _⟩ | ⟨_, _⟩ <;> omega
        simp only [this, ite_true, Option.map_some]
        refine ⟨_, rfl, ?_⟩
        exact inv_setSlot h hi (by arith) (by arith) (.inr ⟨by arith, hle⟩) hinb (Nat.le_refl _)
      · simp only [hle, ite_false]
        by_cases hc : (slotAt b i).cap = 0
        · simp only [hc, ne_eq, not_true_eq_false, ite_false, Option.map_some]
          refine ⟨_, rfl, ?_⟩
          simp only [setSlot]
          have : (slotAt b i).len = 0 := by rcases hwf with ⟨_, h2⟩ | ⟨_, _⟩ <;> omega
          exact inv_setSlot h hi (by arith) (by arith)
            (.inr ⟨by arith, Nat.le_refl _⟩) (.inl (by arith)) (by arith)
        · simp only [hc, ne_eq, not_false_eq_true, ite_true]
          obtain ⟨b', hr, e1, e2, e3, e4⟩ := retire_ok h hi
          rw [hr]; simp only [Option.map_some]
          refine ⟨_, rfl, ?_⟩
          simp only [setSlot]; rw [e1]
          exact inv_setSlot h hi (by arith) (by arith)
            (.inr ⟨by arith, Nat.le_refl _⟩) (.inl (by arith)) (by arith)

/-! ## `compact` -/

theorem sumLen_le_sumCap : ∀ (l : List Slot), (∀ s ∈ l, SlotWF s) →
    sumBy (·.len) l + sumBy (fun s => s.cap - s.len) l = sumBy (·.cap) l
  | [], _ => by simp [sumBy]
  | s :: l, h => by
      have := sumLen_le_sumCap l (fun x hx => h x (by simp [hx]))
      have hs := h s (by simp)
      simp only [sumBy_cons]
      rcases hs with ⟨_, _⟩ | ⟨_, _⟩ <;> omega

/-- **`live_eq_sum_len`**: the `live` that `compact` computes is exactly the
number of entries live spans hold, which is the
`debug_assert_eq!(new_arena.len(), live)` at attribute_store.rs:821. -/
theorem live_eq_sum_len {b : Blk} (h : Inv b) : live b = sumBy (·.len) b.slots := by
  unfold live
  have := sumLen_le_sumCap b.slots h.wf
  rw [h.arena_eq, h.slack_eq]; omega

theorem packOffsets_props : ∀ (o : Nat) (l : List Slot), (∀ s ∈ l, SlotWF s) →
    (∀ s ∈ packOffsets o l, SlotWF s ∧ s.cap = s.len ∧ (s.off + s.cap ≤ o + sumBy (·.len) l ∨ s.cap = 0)) ∧
    sumBy (·.cap) (packOffsets o l) = sumBy (·.len) l ∧
    sumBy (fun s => s.cap - s.len) (packOffsets o l) = 0
  | _, [], _ => by simp [packOffsets, sumBy]
  | o, s :: l, h => by
      have hs := h s (by simp)
      have hl : ∀ x ∈ l, SlotWF x := fun x hx => h x (by simp [hx])
      simp only [packOffsets]
      by_cases hc : s.cap = 0
      · have h0 : s.len = 0 := by rcases hs with ⟨_, h2⟩ | ⟨_, _⟩ <;> omega
        obtain ⟨ih1, ih2, ih3⟩ := packOffsets_props o l hl
        simp only [hc, ite_true, sumBy_cons, h0]
        refine ⟨?_, by omega, by omega⟩
        intro x hx; simp at hx; rcases hx with rfl | hx
        · exact ⟨hs, by omega, .inr hc⟩
        · obtain ⟨a, b, c⟩ := ih1 x hx; exact ⟨a, b, c.elim (fun c => .inl (by omega)) .inr⟩
      · obtain ⟨ih1, ih2, ih3⟩ := packOffsets_props (o + s.len) l hl
        simp only [hc, ite_false, sumBy_cons]
        refine ⟨?_, by omega, by first | (simp; omega) | simp⟩
        intro x hx; simp at hx; rcases hx with rfl | hx
        · refine ⟨?_, rfl, .inl (by first | (simp; omega) | simp)⟩
          rcases hs with ⟨h1, _⟩ | ⟨h1, _⟩
          · exact absurd h1 hc
          · exact .inr ⟨h1, Nat.le_refl _⟩
        · obtain ⟨a, b, c⟩ := ih1 x hx; exact ⟨a, b, c.elim (fun c => .inl (by omega)) .inr⟩

/-- **`compact_ok`**: compaction never underflows, leaves no dead or slack
entries, and yields a well-formed block with every span `cap == len`. -/
theorem compact_ok {b : Blk} (h : Inv b) :
    ∃ b', compact b = some b' ∧ Inv b' ∧ b'.dead = 0 ∧ b'.slack = 0 ∧ b'.arena = live b := by
  have := sumLen_le_sumCap b.slots h.wf
  have ha := h.arena_eq
  have hs := h.slack_eq
  obtain ⟨p1, p2, p3⟩ := packOffsets_props 0 b.slots h.wf
  refine ⟨{ slots := packOffsets 0 b.slots, arena := sumBy (·.len) b.slots, dead := 0, slack := 0 }, ?_, ⟨?_, ?_, ?_, ?_⟩, rfl, rfl, ?_⟩
  · simp only [compact, csub]
    rw [if_pos (by omega)]; simp only [Option.bind_some]
    rw [if_pos (by omega)]; rfl
  · simp only; omega
  · simp only; omega
  · intro s hs'; exact (p1 s hs').1
  · intro s hs'; have := (p1 s hs').2.2; simp at this ⊢; omega
  · rw [live_eq_sum_len h]

end PendingCommit.Arena
