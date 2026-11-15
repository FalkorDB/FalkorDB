import PendingCommit.ContentMerge
import PendingCommit.BinSearch
/-
# `Block::compact` / `maybe_compact` with contents, and `SpanRef` reads

* **`compactC_ok`** — `compact` (attribute_store.rs:798) rebuilds the arena
  from the live spans in slot order: every slot reads exactly what it read
  before, `CInv` holds, `dead = slack = 0`, and the arena is `Σ len` long
  (`Arena.compact_ok` for the counters). The heap remap is
  `Pack.compactHeap_unpack`.
* `maybeCompact_ok` — `maybe_compact` (:828) fires iff
  `2·(dead+slack) > arena.len() ∧ arena.len() > 256`, and either way reads and
  `CInv` are preserved.
* `spanGet_eq` — `SpanRef::get` (:856), a std `binary_search_by_key` over the
  span's ids (`BinSearch.bsearch`), is first-match lookup `Span.get` on a
  strictly sorted span; `SpanRef::entries`/`len`/`iter` (:845-872) are the
  span, its length and its entries (`spanRef_len`).
-/
namespace PendingCommit.Content
open Arena (Slot)

def compactAux (a : List Entry) : Nat → List Slot → List Slot × List Entry
  | _, [] => ([], [])
  | o, s :: ss =>
    if s.cap = 0 then
      let r := compactAux a o ss
      (s :: r.1, r.2)
    else
      let r := compactAux a (o + s.len) ss
      (⟨o, s.len, s.len⟩ :: r.1, spanOf a s ++ r.2)

/-- `Block::compact` (:798). -/
def compactC (b : CB) : CB :=
  let r := compactAux b.arena 0 b.slots
  ⟨r.1, r.2, 0, 0⟩

def LiveIn (a : List Entry) (s : Slot) : Prop := (s.cap = 0 ∧ s.len = 0) ∨ (s.cap ≠ 0 ∧ s.off + s.len ≤ a.length)

theorem compactAux_slots (a : List Entry) : ∀ o ss, (compactAux a o ss).1 = Arena.packOffsets o ss
  | _, [] => rfl
  | o, s :: ss => by
    simp only [compactAux, Arena.packOffsets]
    split <;> simp [compactAux_slots a]

theorem spanOf_len (a : List Entry) (s : Slot) (h : s.off + s.len ≤ a.length) : (spanOf a s).length = s.len := by
  simp [spanOf]; omega

theorem compactAux_len (a : List Entry) : ∀ o ss, (∀ s ∈ ss, LiveIn a s) →
    (compactAux a o ss).2.length = Arena.sumBy (·.len) ss
  | _, [], _ => rfl
  | o, s :: ss, h => by
    have hs := h s (by simp)
    have hr : ∀ x ∈ ss, LiveIn a x := fun x hx => h x (by simp [hx])
    unfold LiveIn at hs
    simp only [Arena.sumBy_cons]
    by_cases hc : s.cap = 0
    · simp only [compactAux, hc, ite_true]
      rw [compactAux_len a o ss hr]; omega
    · simp only [compactAux, hc, ite_false, List.length_append]
      rw [spanOf_len a s (by omega), compactAux_len a (o + s.len) ss hr]

theorem compactAux_span (a : List Entry) : ∀ (ss : List Slot) (o : Nat) (pre : List Entry), pre.length = o →
    (∀ s ∈ ss, LiveIn a s) → ∀ j,
    spanOf (pre ++ (compactAux a o ss).2) ((compactAux a o ss).1.getD j Slot.empty) =
      spanOf a (ss.getD j Slot.empty)
  | [], _, _, _, _, j => by simp [compactAux, spanOf, Slot.empty]
  | s :: ss, o, pre, hpre, h, j => by
    have hs := h s (by simp)
    have hrest : ∀ x ∈ ss, LiveIn a x := fun x hx => h x (by simp [hx])
    unfold LiveIn at hs
    by_cases hc : s.cap = 0
    · simp only [compactAux, hc, ite_true]
      have h0 : s.len = 0 := by omega
      cases j with
      | zero => simp [spanOf, h0]
      | succ j => simpa using compactAux_span a ss o pre hpre hrest j
    · simp only [compactAux, hc, ite_false]
      have hin : s.off + s.len ≤ a.length := by omega
      have hl := spanOf_len a s hin
      cases j with
      | zero =>
        simp only [List.getD_cons_zero]
        conv => lhs; unfold spanOf
        simp only
        rw [List.drop_append_of_le_length (by omega), List.drop_eq_nil_of_le (by omega), List.nil_append]
        rw [← hpre] at *
        try simp only [Nat.sub_self, List.drop_zero]
        unfold spanOf
        rw [List.take_append_of_le_length (by simp; omega), List.take_take, Nat.min_self]
      | succ j =>
        simp only [List.getD_cons_succ]
        have := compactAux_span a ss (o + s.len) (pre ++ spanOf a s)
          (by simp [hpre, hl]) hrest j
        simpa [List.append_assoc] using this

def Before (a c : Slot) : Prop := a.cap = 0 ∨ c.cap = 0 ∨ a.off + a.cap ≤ c.off

theorem pack_apart : ∀ (o : Nat) (ss : List Slot),
    (∀ s ∈ Arena.packOffsets o ss, s.cap = 0 ∨ o ≤ s.off) ∧ List.Pairwise Before (Arena.packOffsets o ss)
  | _, [] => by simp [Arena.packOffsets]
  | o, s :: ss => by
    by_cases hc : s.cap = 0
    · obtain ⟨ih1, ih2⟩ := pack_apart o ss
      simp only [Arena.packOffsets, hc, ite_true]
      refine ⟨?_, List.Pairwise.cons (fun c _ => .inl hc) ih2⟩
      intro x hx; simp at hx; rcases hx with rfl | hx
      · exact .inl hc
      · exact ih1 x hx
    · obtain ⟨ih1, ih2⟩ := pack_apart (o + s.len) ss
      simp only [Arena.packOffsets, hc, ite_false]
      refine ⟨?_, List.Pairwise.cons (fun c hc' => ?_) ih2⟩
      · intro x hx; simp at hx; rcases hx with rfl | hx
        · exact .inr (Nat.le_refl _)
        · rcases ih1 x hx with h | h
          · exact .inl h
          · exact .inr (by omega)
      · rcases ih1 c hc' with h | h
        · exact .inr (.inl h)
        · exact .inr (.inr (by simp only; omega))

theorem apart_of_pairwise (l : List Slot) (hp : List.Pairwise Before l) (p q : Nat) (hpq : p ≠ q) :
    Apart (l.getD p Slot.empty) (l.getD q Slot.empty) := by
  have hpw := List.pairwise_iff_getElem.1 hp
  by_cases hpl : p < l.length
  · by_cases hql : q < l.length
    · simp only [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hpl, List.getElem?_eq_getElem hql,
        Option.getD_some]
      rcases Nat.lt_or_gt_of_ne hpq with h | h
      · have := hpw p q hpl hql h; unfold Before at this; unfold Apart; omega
      · have := hpw q p hql hpl h; unfold Before at this; unfold Apart; omega
    · right; left
      simp [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega : l.length ≤ q), Slot.empty]
  · left
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega : l.length ≤ p), Slot.empty]

theorem liveIn_of_inv (b : CB) (h : CInv b) : ∀ s ∈ b.slots, LiveIn b.arena s := by
  intro s hs
  have hw := h.arena.wf s hs
  have hb := h.arena.inb s hs
  simp only [toBlk] at hw hb
  unfold LiveIn; unfold Arena.SlotWF at hw
  by_cases hc : s.cap = 0
  · left; exact ⟨hc, by omega⟩
  · right; exact ⟨hc, by omega⟩

/-- **`compactC_ok`**: compaction preserves every slot's reads and `CInv`,
and leaves no dead or slack entries. -/
theorem compactC_ok (b : CB) (h : CInv b) :
    CInv (compactC b) ∧ (∀ j, span (compactC b) j = span b j) ∧ (compactC b).dead = 0 ∧
      (compactC b).slack = 0 ∧ (compactC b).arena.length = Arena.sumBy (·.len) b.slots := by
  have hl := liveIn_of_inv b h
  have hlen := compactAux_len b.arena 0 b.slots hl
  have hsl := compactAux_slots b.arena 0 b.slots
  obtain ⟨b', hb', hinv, -, -, -⟩ := Arena.compact_ok h.arena
  have ha := h.arena.arena_eq
  have hs := h.arena.slack_eq
  have hsum := Arena.sumLen_le_sumCap b.slots h.arena.wf
  simp only [toBlk] at ha hs
  have hc : Arena.compact (toBlk b) =
      some ⟨Arena.packOffsets 0 b.slots, Arena.sumBy (·.len) b.slots, 0, 0⟩ := by
    have h1 : b.dead ≤ b.arena.length := by omega
    have h2 : b.slack ≤ b.arena.length - b.dead := by omega
    simp [Arena.compact, Arena.csub, toBlk, h1, h2]
  rw [hc] at hb'; cases hb'
  refine ⟨⟨?_, ?_⟩, ?_, rfl, rfl, hlen⟩
  · simp only [toBlk, compactC, hsl, hlen]; exact hinv
  · intro p q hpq
    simp only [slotAt, compactC, hsl]
    exact apart_of_pairwise _ (pack_apart 0 b.slots).2 p q hpq
  · intro j
    have := compactAux_span b.arena b.slots 0 [] rfl hl j
    simpa [span, slotAt, compactC] using this

/-- `Block::maybe_compact` (:828). -/
def maybeCompact (b : CB) : CB :=
  if (b.dead + b.slack) * 2 > b.arena.length ∧ b.arena.length > 256 then compactC b else b

theorem maybeCompact_ok (b : CB) (h : CInv b) :
    CInv (maybeCompact b) ∧ ∀ j, span (maybeCompact b) j = span b j := by
  unfold maybeCompact
  split
  · obtain ⟨a, c, -⟩ := compactC_ok b h; exact ⟨a, c⟩
  · exact ⟨h, fun _ => rfl⟩

/-! ## `SpanRef` -/

/-- `SpanRef::get` (:856): binary search over the span's ids. -/
def spanGet (es : List Entry) (k : Nat) : Option Val :=
  match BinSearch.bsearch (es.map (·.1)) k with
  | .ok p => some (es.getD p (0, .null)).2
  | .err _ => none

theorem sorted_keys (es : List Entry) (h : Sorted es) : BinSearch.SSorted (es.map (·.1)) := by
  unfold BinSearch.SSorted; unfold Sorted at h
  exact List.pairwise_map.2 h

theorem get_at (es : List Entry) (h : Sorted es) : ∀ p, p < es.length →
    get es (es.getD p (0, .null)).1 = some (es.getD p (0, .null)).2 := by
  induction es with
  | nil => intro p hp; simp at hp
  | cons e es ih =>
    intro p hp
    cases p with
    | zero => simp [get]
    | succ p =>
      simp only [List.getD_cons_succ]
      have hp' : p < es.length := by simpa using hp
      have hlt : e.1 < (es.getD p (0, .null)).1 := by
        apply List.rel_of_pairwise_cons h
        simp only [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hp', Option.getD_some]
        exact List.getElem_mem hp'
      simp only [get]; rw [if_neg (by omega)]; exact ih (List.Pairwise.of_cons h) p hp'

theorem get_none_of_not_mem (es : List Entry) (k : Nat) (h : k ∉ es.map (·.1)) : get es k = none := by
  induction es with
  | nil => rfl
  | cons e es ih =>
    simp only [List.map_cons, List.mem_cons, not_or] at h
    simp only [get]; rw [if_neg (fun e' => h.1 e'), ih h.2]

/-- **`spanGet_eq`**: the binary search returns the first-match lookup. -/
theorem spanGet_eq (es : List Entry) (h : Sorted es) (k : Nat) : spanGet es k = get es k := by
  unfold spanGet
  have hs := sorted_keys es h
  cases e : BinSearch.bsearch (es.map (·.1)) k with
  | ok p =>
    have ⟨hp, hk⟩ := (BinSearch.bsearch_ok_iff _ hs k p).1 e
    simp only [List.length_map] at hp
    have : (es.map (·.1)).getD p 0 = (es.getD p (0, .null)).1 := by
      simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hp]
    rw [this] at hk
    rw [← hk, get_at es h p hp]
  | err p =>
    have ⟨_, hlo, hhi⟩ := BinSearch.bsearch_err _ hs k p e
    symm; apply get_none_of_not_mem
    intro hm
    obtain ⟨j, hj, hjk⟩ := List.getElem_of_mem hm
    have hjk' : (es.map (·.1)).getD j 0 = k := by
      simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, hjk]
    rcases Nat.lt_or_ge j p with h1 | h1
    · have := hlo j h1; omega
    · have := hhi j h1 hj; omega

/-- `SpanRef::len` (:851) is the span's length; `entries` (:845) and `iter`
(:867) are the span itself (values via `Pack.unpack`). -/
theorem spanRef_len (b : CB) (h : CInv b) (i : Nat) : (slotAt b i).len = (span b i).length :=
  (span_length b h i).symm

end PendingCommit.Content
