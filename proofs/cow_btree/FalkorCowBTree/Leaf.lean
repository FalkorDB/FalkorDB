import FalkorCowBTree.Invariant

/-!
# Leaf pages: `Leaf::insert`, `Leaf::remove`, and the midpoint split (`leaf/mod.rs`)
-/

namespace CowBTree

theorem take_takeWhile_length {α} (p : α → Bool) : ∀ (l : List α), l.take (l.takeWhile p).length = l.takeWhile p
  | [] => rfl
  | a :: l => by
    rw [List.takeWhile_cons]
    split
    · simp [take_takeWhile_length p l]
    · simp

theorem drop_takeWhile_length {α} (p : α → Bool) : ∀ (l : List α), l.drop (l.takeWhile p).length = l.dropWhile p
  | [] => rfl
  | a :: l => by
    rw [List.takeWhile_cons, List.dropWhile_cons]
    split
    · simp [drop_takeWhile_length p l]
    · simp

theorem mem_takeWhile_p {α} {p : α → Bool} {l : List α} {a : α} (h : a ∈ l.takeWhile p) : p a = true :=
  List.all_eq_true.1 List.all_takeWhile a h

theorem takeWhile_length_le {α} (p : α → Bool) (l : List α) : (l.takeWhile p).length ≤ l.length :=
  (List.takeWhile_sublist p).length_le

/-- Everything in a sorted list's `dropWhile (· < x)` is `>= x`. -/
theorem dropWhile_lt_ge (x : E) : ∀ (l : List E), Sorted l → ∀ e ∈ l.dropWhile (fun e => decide (e < x)), x ≤ e
  | [], _, e, he => by simp at he
  | a :: l, hs, e, he => by
    rw [List.dropWhile_cons] at he
    rw [sorted_cons] at hs
    split at he
    · exact dropWhile_lt_ge x l hs.2 e he
    · rename_i hax
      simp at hax
      rcases List.mem_cons.1 he with he | he
      · omega
      · have := hs.1 e he; omega

/-- What `Leaf::lower_bound_entry`'s partition point tells us on a sorted page. -/
theorem lbe_facts (es : List E) (x : E) (hs : Sorted es) :
    let pos := lowerBoundEntry es x
    (∀ e ∈ es.take pos, e < x) ∧ (∀ e ∈ es.drop pos, x ≤ e) ∧ (es[pos]? = some x ↔ x ∈ es) ∧ pos ≤ es.length := by
  intro pos
  have ht : es.take pos = es.takeWhile (fun e => decide (e < x)) := take_takeWhile_length _ es
  have hd : es.drop pos = es.dropWhile (fun e => decide (e < x)) := drop_takeWhile_length _ es
  have h1 : ∀ e ∈ es.take pos, e < x := by
    rw [ht]; intro e he; simpa using mem_takeWhile_p he
  have h2 : ∀ e ∈ es.drop pos, x ≤ e := by rw [hd]; exact dropWhile_lt_ge x es hs
  refine ⟨h1, h2, ?_, takeWhile_length_le _ es⟩
  constructor
  · intro h; exact List.mem_of_getElem? h
  · intro hx
    have hsplit := List.take_append_drop pos es
    have hx' : x ∈ es.take pos ++ es.drop pos := by rw [hsplit]; exact hx
    rcases List.mem_append.1 hx' with hx' | hx'
    · have := h1 x hx'; omega
    · have hlt : pos < es.length := by
        have : (es.drop pos).length > 0 := List.length_pos_of_mem hx'
        simp at this; omega
      have hdc := List.drop_eq_getElem_cons hlt
      rw [hdc] at hx' h2
      have hsd : Sorted (es.drop pos) := Sorted.sublist (List.drop_sublist _ _) hs
      rw [hdc, sorted_cons] at hsd
      rw [List.getElem?_eq_getElem hlt]
      rcases List.mem_cons.1 hx' with hx' | hx'
      · rw [hx']
      · have := hsd.1 x hx'; have := h2 es[pos] List.mem_cons_self; omega

theorem mem_insAt {α} (l : List α) (i : Nat) (x e : α) : e ∈ insAt l i x ↔ e ∈ l ∨ e = x := by
  unfold insAt
  conv => rhs; rw [← List.take_append_drop i l]
  simp only [List.mem_append, List.mem_cons]
  constructor
  · rintro (h | h | h)
    · exact Or.inl (Or.inl h)
    · exact Or.inr h
    · exact Or.inl (Or.inr h)
  · rintro ((h | h) | h)
    · exact Or.inl h
    · exact Or.inr (Or.inr h)
    · exact Or.inr (Or.inl h)

theorem length_insAt {α} (l : List α) (i : Nat) (x : α) (h : i ≤ l.length) : (insAt l i x).length = l.length + 1 := by
  unfold insAt; simp; omega

/-- Inserting at the lower bound keeps a page sorted (when the entry is absent). -/
theorem sorted_insAt_lbe (es : List E) (x : E) (hs : Sorted es) (hn : es[lowerBoundEntry es x]? ≠ some x) :
    Sorted (insAt es (lowerBoundEntry es x) x) := by
  obtain ⟨h1, h2, h3, _⟩ := lbe_facts es x hs
  have hxn : x ∉ es := fun hx => hn (h3.2 hx)
  unfold insAt
  have hsplit := List.take_append_drop (lowerBoundEntry es x) es
  have hs' := hs; rw [← hsplit] at hs'
  obtain ⟨st, sd, _⟩ := sorted_append.1 hs'
  refine sorted_append.2 ⟨st, sorted_cons.2 ⟨?_, sd⟩, ?_⟩
  · intro b hb
    have := h2 b hb
    have : b ≠ x := fun hbx => hxn (by rw [← hsplit]; exact List.mem_append_right _ (hbx ▸ hb))
    omega
  · intro a ha b hb
    rcases List.mem_cons.1 hb with hb | hb
    · subst hb; exact h1 a ha
    · have := h1 a ha; have := h2 b hb; omega

/-- A sorted list cut at `mid` is bounded by its `mid`-th element. -/
theorem sorted_cut (ps : List E) (hs : Sorted ps) (mid : Nat) (hm : mid < ps.length) :
    (∀ e ∈ ps.take mid, e < ps.getD mid 0) ∧ (∀ e ∈ ps.drop mid, ps.getD mid 0 ≤ e) := by
  rw [getD_eq_getElem _ _ _ hm]
  have hsp := split_at ps mid hm
  have hs' := hs; rw [hsp] at hs'
  obtain ⟨_, sd, hcross⟩ := sorted_append.1 hs'
  refine ⟨fun e he => hcross e he _ List.mem_cons_self, ?_⟩
  intro e he
  rw [List.drop_eq_getElem_cons hm] at he
  rcases List.mem_cons.1 he with he | he
  · omega
  · have := (sorted_cons.1 sd).1 e he; omega

/-- **`Leaf::insert` (`leaf/mod.rs:372`)**: absent ⇒ a sorted page with the entry added, split at the
    midpoint on overflow into two non-empty, `>= LEAF_MAX / 2`-full halves bounded by the separator. -/
theorem leafInsert_spec (c : Cfg) (es : List E) (x : E) (hs : Sorted es) (hl : es.length ≤ c.L) :
    match leafInsert c es x with
    | none => x ∈ es
    | some (.fit ps) => Sorted ps ∧ (∀ e, e ∈ ps ↔ e ∈ es ∨ e = x) ∧ ps.length ≤ c.L ∧ ps.length = es.length + 1
    | some (.split l s r) =>
        Sorted l ∧ Sorted r ∧ (∀ e, e ∈ l ∨ e ∈ r ↔ e ∈ es ∨ e = x) ∧
        (∀ e ∈ l, e < s) ∧ (∀ e ∈ r, s ≤ e) ∧ l ≠ [] ∧ r ≠ [] ∧
        c.L / 2 ≤ l.length ∧ l.length ≤ c.L ∧ c.L / 2 ≤ r.length ∧ r.length ≤ c.L := by
  have hL := c.hL
  obtain ⟨_, _, h3, hle⟩ := lbe_facts es x hs
  by_cases hsome : es[lowerBoundEntry es x]? = some x
  · simp only [leafInsert, hsome, ↓reduceIte]; exact h3.1 hsome
  · have hsrt := sorted_insAt_lbe es x hs hsome
    have hlen := length_insAt es (lowerBoundEntry es x) x hle
    have hmem := mem_insAt es (lowerBoundEntry es x) x
    simp only [leafInsert, hsome, ↓reduceIte]
    generalize insAt es (lowerBoundEntry es x) x = ps at hsrt hlen hmem ⊢
    by_cases hfit : ps.length ≤ c.L
    · simp only [hfit, ↓reduceIte]
      exact ⟨hsrt, hmem, by simp [hfit], hlen⟩
    · simp only [hfit, ↓reduceIte]
      have hmid : ps.length / 2 < ps.length := by omega
      obtain ⟨hc1, hc2⟩ := sorted_cut ps hsrt _ hmid
      have hsp := List.take_append_drop (ps.length / 2) ps
      have hs' := hsrt; rw [← hsp] at hs'
      obtain ⟨st, sd, _⟩ := sorted_append.1 hs'
      refine ⟨st, sd, ?_, hc1, hc2, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · intro e; rw [← hmem e, ← List.mem_append, hsp]
      · intro h0; have := congrArg List.length h0; rw [List.length_take, List.length_nil] at this; omega
      · intro h0; have := congrArg List.length h0; rw [List.length_drop, List.length_nil] at this; omega
      all_goals simp only [List.length_take, List.length_drop]; omega

/-- **`Leaf::remove` (`leaf/mod.rs:432`)**: present ⇒ a sorted page without it, one shorter, with
    the underflow flag `new_count < LEAF_MAX / 2`; absent ⇒ `None`. -/
theorem leafRemove_spec (c : Cfg) (es : List E) (x : E) (hs : Sorted es) :
    match leafRemove c es x with
    | none => x ∉ es
    | some (es', u) => Sorted es' ∧ (∀ e, e ∈ es' ↔ e ∈ es ∧ e ≠ x) ∧ es'.length + 1 = es.length ∧
        u = decide (es'.length < c.L / 2) := by
  obtain ⟨h1, h2, h3, hle⟩ := lbe_facts es x hs
  by_cases hsome : es[lowerBoundEntry es x]? = some x
  · simp only [leafRemove, hsome, ↓reduceIte]
    have hlt : lowerBoundEntry es x < es.length := by
      rcases Nat.lt_or_ge (lowerBoundEntry es x) es.length with h | h
      · exact h
      · rw [List.getElem?_eq_none h] at hsome; simp at hsome
    have hx : es[lowerBoundEntry es x] = x := by
      rw [List.getElem?_eq_getElem hlt] at hsome; simpa using hsome
    generalize lowerBoundEntry es x = pos at hlt hx
    have hsp := split_at es pos hlt
    rw [hx] at hsp
    have hs' := hs; rw [hsp] at hs'
    obtain ⟨st, sd, hcross⟩ := sorted_append.1 hs'
    have sd' := sorted_cons.1 sd
    unfold delAt
    refine ⟨sorted_append.2 ⟨st, sd'.2, fun a ha b hb => by
      have := hcross a ha x List.mem_cons_self; have := sd'.1 b hb; omega⟩, ?_, ?_, ?_⟩
    · intro e
      conv => rhs; rw [hsp]
      simp only [List.mem_append, List.mem_cons]
      constructor
      · rintro (h | h)
        · exact ⟨Or.inl h, by have := hcross e h x List.mem_cons_self; omega⟩
        · exact ⟨Or.inr (Or.inr h), by have := sd'.1 e h; omega⟩
      · rintro ⟨h | h | h, hne⟩
        · exact Or.inl h
        · exact absurd h hne
        · exact Or.inr h
    · simp; omega
    · simp; omega
  · simp only [leafRemove, hsome, ↓reduceIte]
    exact fun hx => hsome (h3.2 hx)

end CowBTree
