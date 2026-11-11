import FalkorCowBTree.Model

/-!
# Basic facts: the entry encoding, sorted lists, and the separator-chain predicate `COK`
-/

namespace CowBTree

/-! ## The encoding is an order isomorphism -/

/-- Lexicographic `(k, d) < (k', d')` on `u64` pairs is `<` on the encoding. -/
theorem enc_lt_iff (k d k' d' : Nat) (hd : d < W) (hd' : d' < W) :
    enc k d < enc k' d' ↔ (k < k' ∨ (k = k' ∧ d < d')) := by
  simp only [enc, W] at *
  constructor
  · intro h
    by_cases hk : k < k'
    · exact Or.inl hk
    · right
      have : k' ≤ k := by omega
      by_cases hkk : k = k'
      · subst hkk; exact ⟨rfl, by omega⟩
      · have : k' + 1 ≤ k := by omega
        have : (k' + 1) * 18446744073709551616 ≤ k * 18446744073709551616 := Nat.mul_le_mul_right _ this
        omega
  · rintro (h | ⟨rfl, h⟩)
    · have : (k + 1) * 18446744073709551616 ≤ k' * 18446744073709551616 := Nat.mul_le_mul_right _ h
      omega
    · omega

theorem keyOf_enc (k d : Nat) (hd : d < W) : keyOf (enc k d) = k := by
  simp only [keyOf, enc, W] at *; omega

theorem docOf_enc (k d : Nat) (hd : d < W) : docOf (enc k d) = d := by
  simp only [docOf, enc, W] at *; omega

/-- Seeking to `(lo, 0)`: an entry is below it iff its key is below `lo`. -/
theorem lt_enc_lo_iff (e lo : Nat) : e < enc lo 0 ↔ keyOf e < lo := by
  simp only [keyOf, enc, W]; omega

/-! ## Sorted (strictly increasing) lists -/

def Sorted (l : List E) : Prop := l.Pairwise (· < ·)

theorem sorted_nil : Sorted [] := List.Pairwise.nil

theorem sorted_cons {a : E} {l : List E} : Sorted (a :: l) ↔ (∀ b ∈ l, a < b) ∧ Sorted l :=
  List.pairwise_cons

theorem sorted_append {l₁ l₂ : List E} :
    Sorted (l₁ ++ l₂) ↔ Sorted l₁ ∧ Sorted l₂ ∧ ∀ a ∈ l₁, ∀ b ∈ l₂, a < b :=
  List.pairwise_append

theorem Sorted.sublist {l₁ l₂ : List E} (h : l₁.Sublist l₂) (hs : Sorted l₂) : Sorted l₁ :=
  List.Pairwise.sublist h hs

/-- Two strictly sorted lists with the same members are equal: the reference "sorted list" model
    is determined by its membership, so all content theorems can be stated as membership facts. -/
theorem sorted_ext : ∀ {l₁ l₂ : List E}, Sorted l₁ → Sorted l₂ → (∀ e, e ∈ l₁ ↔ e ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => absurd ((h b).2 (List.mem_cons_self)) (List.not_mem_nil)
  | a :: _, [], _, _, h => absurd ((h a).1 (List.mem_cons_self)) (List.not_mem_nil)
  | a :: as, b :: bs, h₁, h₂, h => by
    rw [sorted_cons] at h₁ h₂
    have hab : a = b := by
      have ha := (h a).1 List.mem_cons_self
      have hb := (h b).2 List.mem_cons_self
      rcases List.mem_cons.1 ha with ha | ha
      · exact ha
      rcases List.mem_cons.1 hb with hb | hb
      · exact hb.symm
      have := h₂.1 a ha; have := h₁.1 b hb; omega
    subst hab
    congr 1
    apply sorted_ext h₁.2 h₂.2
    intro e
    constructor
    · intro he
      have := (h e).1 (List.mem_cons_of_mem _ he)
      rcases List.mem_cons.1 this with h' | h'
      · subst h'; have := h₁.1 e he; omega
      · exact h'
    · intro he
      have := (h e).2 (List.mem_cons_of_mem _ he)
      rcases List.mem_cons.1 this with h' | h'
      · subst h'; have := h₂.1 e he; omega
      · exact h'

/-! ## The reference model: a sorted list with insert / erase -/

/-- Reference insert into a sorted list (set semantics: idempotent). -/
def sins (x : E) : List E → List E
  | [] => [x]
  | y :: ys => if x < y then x :: y :: ys else if x = y then y :: ys else y :: sins x ys

theorem mem_sins (x : E) : ∀ (l : List E) (e : E), e ∈ sins x l ↔ e ∈ l ∨ e = x
  | [], e => by simp [sins]
  | y :: ys, e => by
    unfold sins
    split
    · simp only [List.mem_cons]
      constructor
      · rintro (h | h | h) <;> simp [h]
      · rintro ((h | h) | h) <;> simp [h]
    · split
      · subst_vars; simp only [List.mem_cons]
        constructor
        · rintro (h | h) <;> simp [h]
        · rintro ((h | h) | h) <;> simp [h]
      · simp only [List.mem_cons, mem_sins x ys e]
        constructor
        · rintro (h | h | h) <;> simp [h]
        · rintro ((h | h) | h) <;> simp [h]

theorem sorted_sins (x : E) : ∀ (l : List E), Sorted l → Sorted (sins x l)
  | [], _ => by simp [sins, Sorted]
  | y :: ys, h => by
    unfold sins
    rw [sorted_cons] at h
    split
    · rw [sorted_cons]; refine ⟨?_, sorted_cons.2 h⟩
      intro b hb; rcases List.mem_cons.1 hb with hb | hb
      · omega
      · have := h.1 b hb; omega
    · split
      · exact sorted_cons.2 h
      · rw [sorted_cons]; refine ⟨?_, sorted_sins x ys h.2⟩
        intro b hb; rcases (mem_sins x ys b).1 hb with hb | hb
        · exact h.1 b hb
        · omega

/-- The reference erase on a sorted list: `List.erase`. Membership spec on a sorted (nodup) list. -/
theorem mem_erase_sorted (x : E) : ∀ (l : List E), Sorted l → ∀ e, e ∈ l.erase x ↔ e ∈ l ∧ e ≠ x
  | [], _, e => by simp
  | y :: ys, h, e => by
    rw [sorted_cons] at h
    by_cases hxy : y = x
    · subst hxy
      simp only [List.erase_cons_head]
      constructor
      · intro he; exact ⟨List.mem_cons_of_mem _ he, by have := h.1 e he; omega⟩
      · rintro ⟨he, hne⟩; rcases List.mem_cons.1 he with he | he
        · exact absurd he hne
        · exact he
    · rw [List.erase_cons_tail (by simpa using hxy)]
      simp only [List.mem_cons, mem_erase_sorted x ys h.2 e]
      constructor
      · rintro (he | ⟨he, hne⟩)
        · subst he; exact ⟨Or.inl rfl, hxy⟩
        · exact ⟨Or.inr he, hne⟩
      · rintro ⟨he | he, hne⟩
        · exact Or.inl he
        · exact Or.inr ⟨he, hne⟩

theorem sorted_erase (x : E) (l : List E) (h : Sorted l) : Sorted (l.erase x) :=
  Sorted.sublist (List.erase_sublist) h

/-! ## Vec edits -/

theorem setAt_eq {α} (l : List α) (i : Nat) (x : α) : setAt l i x = l.take i ++ [x] ++ l.drop (i + 1) := by
  simp [setAt]

theorem split_at {α} (l : List α) (i : Nat) (h : i < l.length) :
    l = l.take i ++ l[i] :: l.drop (i + 1) := by
  conv => lhs; rw [← List.take_append_drop i l]
  rw [List.drop_eq_getElem_cons h]

theorem getD_eq_getElem {α} (l : List α) (i : Nat) (d : α) (h : i < l.length) : l.getD i d = l[i] := by
  simp [List.getD, List.getElem?_eq_getElem h]

/-- `Vec::insert(i + 1, r)` after `v[i] = c'`. -/
theorem insAt_setAt {α} (l : List α) (i : Nat) (c r : α) (h : i < l.length) :
    insAt (setAt l i c) (i + 1) r = l.take i ++ [c, r] ++ l.drop (i + 1) := by
  unfold insAt setAt
  have hl : (l.take i).length = i := by simp; omega
  have e1 : (l.take i).take (i + 1) = l.take i := List.take_of_length_le (by rw [hl]; omega)
  have e2 : (l.take i).drop (i + 1) = [] := List.drop_of_length_le (by rw [hl]; omega)
  rw [List.take_append, List.drop_append, hl, e1, e2]
  simp

end CowBTree
