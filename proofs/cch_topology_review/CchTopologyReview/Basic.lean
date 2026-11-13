/-!
# Rows, the row merge, and strictly ascending lists

A topology row entry is `(col, id)`: for `Up` row `lo` it is `(hi, id)`, for `Down` row `hi` it is
`(lo, id)` (03 §2 "Arc identity": "The id is the value at `Up(lo,hi)` and at `Down(hi,lo)`").

`mergeRow` is the two-pointer merge 03 §2 I3′ reads `up_ids(r)` with ("ascending `(hi, id)` pairs from
merging base and delta") and §6b step 3 reads `Down(lo)`, `Down(hi)` with ("base slice plus delta
cursor"). 03 does not say what the merge does on equal columns; with the delta disjoint from the base
(I3′'s premise) that case never arises, and `B1_merge_needs_disjoint` (Findings.lean) shows what happens without it.
-/

namespace CchTopo

/-- A row entry `(col, id)`. -/
abbrev RE := Nat × Nat

/-- Strictly ascending by column (03 §2 I3′: "ascending `(hi, id)` pairs"; strict since a row holds one
    arc per column). -/
def StrictCol (l : List RE) : Prop := l.Pairwise (fun a b => a.1 < b.1)

/-- The merge of a base row and a delta row, by column. -/
def mergeRow : List RE → List RE → List RE
  | [], ys => ys
  | x :: xs, [] => x :: xs
  | x :: xs, y :: ys =>
    if x.1 < y.1 then x :: mergeRow xs (y :: ys) else y :: mergeRow (x :: xs) ys
termination_by xs ys => xs.length + ys.length

theorem mergeRow_nil_left (ys : List RE) : mergeRow [] ys = ys := by simp [mergeRow]

theorem mergeRow_nil_right (xs : List RE) : mergeRow xs [] = xs := by
  cases xs <;> simp [mergeRow]

/-- The merge holds exactly the entries of both rows. -/
theorem mem_mergeRow : ∀ (xs ys : List RE) (p : RE), p ∈ mergeRow xs ys ↔ p ∈ xs ∨ p ∈ ys
  | [], ys, p => by simp [mergeRow]
  | x :: xs, [], p => by simp [mergeRow]
  | x :: xs, y :: ys, p => by
    unfold mergeRow
    split
    · rw [List.mem_cons, mem_mergeRow xs (y :: ys) p]
      simp only [List.mem_cons, or_assoc]
    · rw [List.mem_cons, mem_mergeRow (x :: xs) ys p]
      simp only [List.mem_cons, or_assoc, or_left_comm]
termination_by xs ys => xs.length + ys.length

theorem strictCol_cons {a : RE} {l : List RE} : StrictCol (a :: l) ↔ (∀ b ∈ l, a.1 < b.1) ∧ StrictCol l := by
  simp [StrictCol, List.pairwise_cons]

/-- **The merge of two strictly ascending rows with no column in common is strictly ascending.** -/
theorem mergeRow_strict : ∀ (xs ys : List RE), StrictCol xs → StrictCol ys →
    (∀ x ∈ xs, ∀ y ∈ ys, x.1 ≠ y.1) → StrictCol (mergeRow xs ys)
  | [], ys, _, hy, _ => by simpa [mergeRow] using hy
  | x :: xs, [], hx, _, _ => by simpa [mergeRow] using hx
  | x :: xs, y :: ys, hx, hy, hd => by
    rw [strictCol_cons] at hx hy
    unfold mergeRow
    split
    · rename_i hlt
      rw [strictCol_cons]
      refine ⟨fun b hb => ?_, mergeRow_strict xs (y :: ys) hx.2 (strictCol_cons.2 hy) ?_⟩
      · rcases (mem_mergeRow xs (y :: ys) b).1 hb with h | h
        · exact hx.1 b h
        · rcases List.mem_cons.1 h with h | h
          · subst h; exact hlt
          · exact Nat.lt_trans hlt (hy.1 b h)
      · intro a ha b hb; exact hd a (List.mem_cons_of_mem _ ha) b hb
    · rename_i hge
      have hne : x.1 ≠ y.1 := hd x (List.mem_cons_self ..) y (List.mem_cons_self ..)
      have hyx : y.1 < x.1 := by omega
      rw [strictCol_cons]
      refine ⟨fun b hb => ?_, mergeRow_strict (x :: xs) ys (strictCol_cons.2 hx) hy.2 ?_⟩
      · rcases (mem_mergeRow (x :: xs) ys b).1 hb with h | h
        · rcases List.mem_cons.1 h with h | h
          · subst h; exact hyx
          · exact Nat.lt_trans hyx (hx.1 b h)
        · exact hy.1 b h
      · intro a ha b hb; exact hd a ha b (List.mem_cons_of_mem _ hb)
termination_by xs ys => xs.length + ys.length

/-- **A strictly ascending row is determined by its entries.** -/
theorem strict_ext : ∀ {l₁ l₂ : List RE}, StrictCol l₁ → StrictCol l₂ → (∀ p, p ∈ l₁ ↔ p ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => by have := (h b).2 (List.mem_cons_self ..); simp at this
  | a :: _, [], _, _, h => by have := (h a).1 (List.mem_cons_self ..); simp at this
  | a :: as, b :: bs, h1, h2, h => by
    rw [strictCol_cons] at h1 h2
    have hab : a = b := by
      rcases List.mem_cons.1 ((h a).1 (List.mem_cons_self ..)) with e | e
      · exact e
      · rcases List.mem_cons.1 ((h b).2 (List.mem_cons_self ..)) with e' | e'
        · exact e'.symm
        · have := h2.1 a e; have := h1.1 b e'; omega
    subst hab
    congr 1
    refine strict_ext h1.2 h2.2 (fun p => ⟨fun hp => ?_, fun hp => ?_⟩)
    · rcases List.mem_cons.1 ((h p).1 (List.mem_cons_of_mem _ hp)) with e | e
      · subst e; have := h1.1 _ hp; omega
      · exact e
    · rcases List.mem_cons.1 ((h p).2 (List.mem_cons_of_mem _ hp)) with e | e
      · subst e; have := h2.1 _ hp; omega
      · exact e

/-- A strictly ascending row holds each column at most once. -/
theorem strictCol_col_unique : ∀ {l : List RE}, StrictCol l → ∀ p ∈ l, ∀ q ∈ l, p.1 = q.1 → p = q
  | [], _, p, hp, _, _, _ => by simp at hp
  | a :: l, h, p, hp, q, hq, he => by
    rw [strictCol_cons] at h
    rcases List.mem_cons.1 hp with e | e <;> rcases List.mem_cons.1 hq with e' | e'
    · rw [e, e']
    · subst e; have := h.1 q e'; omega
    · subst e'; have := h.1 p e; omega
    · exact strictCol_col_unique h.2 p e q e' he

theorem le_sum_of_mem' : ∀ {l : List Nat} {a : Nat}, a ∈ l → a ≤ l.sum
  | [], _, h => by simp at h
  | x :: l, a, h => by
    rcases List.mem_cons.1 h with e | e
    · subst e; simp
    · have := le_sum_of_mem' e; simp; omega

/-- Sums of a function over a list without duplicates bound two distinct members' values. -/
theorem two_le_sum {β} [DecidableEq β] (f : β → Nat) : ∀ {l : List β}, l.Nodup → ∀ {a b : β}, a ∈ l → b ∈ l → a ≠ b →
    f a + f b ≤ (l.map f).sum
  | [], _, _, _, ha, _, _ => by simp at ha
  | x :: l, hn, a, b, ha, hb, hab => by
    rw [List.nodup_cons] at hn
    simp only [List.map_cons, List.sum_cons]
    rcases List.mem_cons.1 ha with e | e <;> rcases List.mem_cons.1 hb with e' | e'
    · exact absurd (e.trans e'.symm) hab
    · subst e
      have : f b ≤ (l.map f).sum := le_sum_of_mem' (List.mem_map_of_mem e')
      omega
    · subst e'
      have : f a ≤ (l.map f).sum := le_sum_of_mem' (List.mem_map_of_mem e)
      omega
    · have := two_le_sum f hn.2 e e' hab; omega

end CchTopo
