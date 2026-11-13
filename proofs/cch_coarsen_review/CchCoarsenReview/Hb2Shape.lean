import CchCoarsenReview.Hb2
import CchCoarsenReview.Overflow
import CchCoarsenReview.Gath
/-!
# `hb2`'s output shape, symmetry and value bounds; the gathered graph's values

`b2_splice` hands its arrays to `pack_csr_t` with `jumbled = false` (m1_coarsen_bench.rs:306
(m1opt.patch @7df8e5887)), so GraphBLAS takes every row as strictly ascending. `hb2_eq` compares
denotations (`lookup` finds the first match), which by itself does not rule out an unsorted row or a
repeated column; `hb2_row_shape` does. It also discharges `B2Pre.parStrict` and `B2Pre.pStrict` one level
down, where this level is the parent's or the previous level.

`hb2_symm` and `hb2_no_diag` give the next level's `B2Pre.pSymm` and the matching's `KSpec` (gap G3's B2
half). `hb2_vals` bounds B2's values (`v += tmp[q].1`, m1_reuse_bench.rs:879, is an `i32`).
-/

namespace CchCoarsen

theorem mem_of_lookup {α : Type} {row : List (Nat × α)} {k : Nat} {v : α} (h : row.lookup k = some v) :
    (k, v) ∈ row := by
  induction row with
  | nil => simp at h
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    rw [List.lookup_cons] at h
    by_cases hk : k = a
    · subst hk; simp at h; subst h; simp
    · have : (k == a) = false := by simp [hk]
      rw [this] at h; exact List.mem_cons_of_mem _ (ih h)

theorem ewAdd_isSome {α : Type} (f : α → α → α) (A B : Mat α) (i j : Nat) :
    (ewAdd f A B i j).isSome = true ↔ ((A i j).isSome = true ∨ (B i j).isSome = true) := by
  unfold ewAdd; cases A i j <;> cases B i j <;> simp

theorem ewMult_isSome {α β γ : Type} (f : α → β → γ) (A : Mat α) (B : Mat β) (i j : Nat) :
    (ewMult f A B i j).isSome = true ↔ ((A i j).isSome = true ∧ (B i j).isSome = true) := by
  unfold ewMult; cases A i j <;> cases B i j <;> simp

section
variable {ncells nprev : Nat} {parA pA : Rows Int} {N : B2Num} {srt : List (Nat × Int) → List (Nat × Int)}

theorem tcolRow_keys (i : Nat) : ∀ e ∈ tcolRow srt pA N i, N.np ≤ e.1 := by
  intro e he
  simp only [tcolRow, List.mem_filterMap, Option.map_eq_some_iff] at he
  obtain ⟨s, hs, v, _, rfl⟩ := he
  obtain ⟨k, _, rfl⟩ := List.mem_range'.1 hs
  simp

theorem tcolRow_strict (i : Nat) : StrictRow (tcolRow srt pA N i) := by
  unfold tcolRow StrictRow
  refine List.Pairwise.filterMap _ ?_ (List.pairwise_lt_range' (s := N.np) (n := N.n - N.np))
  intro a a' hlt b hb b' hb'
  simp only [Option.map_eq_some_iff] at hb hb'
  obtain ⟨_, _, rfl⟩ := hb
  obtain ⟨_, _, rfl⟩ := hb'
  exact hlt

/-- **`b2_splice`'s rows are strictly ascending** (sorted, no duplicates): the clean entries are below
`np` and sorted (a filter of the parent's row), the appended `tcol` entries are parts, at or above `np`
and ascending, and a part row is a merge of a key-sorted list. -/
theorem hb2_row_shape (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt) (i : Nat) :
    StrictRow (b2Rows srt parA pA N i) := by
  unfold b2Rows
  split
  · unfold StrictRow
    rw [List.pairwise_append]
    refine ⟨?_, tcolRow_strict i, ?_⟩
    · split
      · exact List.Pairwise.sublist List.filter_sublist (hp.parStrict i)
      · simp
    · intro a ha b hb
      have hb' := tcolRow_keys i b hb
      have ha' : a.1 < N.np := by
        split at ha
        · exact hp.parColLt i a (List.mem_filter.1 ha).1
        · simp at ha
      omega
  · exact (sortGrp_spec hS _).1

theorem b2GB_isSome (i c : Nat) :
    (b2GB ncells nprev (den parA) (den pA) N i c).isSome = true ↔
      ((zB2 ncells (den parA) N i c).isSome = true ∨ (RGB nprev (den pA) N i c).isSome = true ∨
        (RGB nprev (den pA) N c i).isSome = true) := by
  unfold b2GB
  rw [ewAdd_isSome, ewAdd_isSome]
  unfold xB2
  rw [ewMult_isSome]
  unfold tr
  constructor
  · rintro (⟨_, h⟩ | h | h)
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)
  · rintro (h | h | h)
    · refine Or.inl ⟨?_, h⟩
      unfold zB2 at h
      by_cases hp : (den parA i c).isSome = true
      · exact hp
      · simp [hp] at h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)

/-- **B2's level is symmetric in pattern when the parent's level is** (`R ∪ Rᵀ` is, and `x` keeps a
symmetric relation, one cell for both ends). With `hcon_symm` this closes the pattern half of gap G3:
every level the next depth reads is symmetric in pattern. -/
theorem hb2_symm (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt)
    (hsym : ∀ q q', (den parA q q').isSome = true → (den parA q' q).isSome = true) (i c : Nat)
    (h : (den (b2Rows srt parA pA N) i c).isSome = true) : (den (b2Rows srt parA pA N) c i).isSome = true := by
  rw [hb2_eq hp hS] at h ⊢
  rw [b2GB_isSome] at h ⊢
  rcases h with h | h | h
  · obtain ⟨h1, hi, hc, hs, hcc⟩ := (zB2_isSome hp i c).1 h
    exact Or.inl ((zB2_isSome hp c i).2 ⟨hsym i c h1, hc, hi, by rw [hcc]; exact hs, hcc.symm⟩)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

/-- **No diagonal in B2's level** when the parent's level has none (`R` is `OFFDIAG`). -/
theorem hb2_no_diag (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt)
    (hdiag : ∀ q, (den parA q q).isSome = false) (i : Nat) :
    (den (b2Rows srt parA pA N) i i).isSome = false := by
  rw [hb2_eq hp hS]
  cases h : (b2GB ncells nprev (den parA) (den pA) N i i).isSome
  · rfl
  · exfalso
    rcases (b2GB_isSome i i).1 h with h | h | h
    · have := ((zB2_isSome hp i i).1 h).1; rw [hdiag i] at this; exact Bool.noConfusion this
    · simp [RGB, offdiag] at h
    · simp [RGB, offdiag] at h

/-! ## Values -/

theorem contrTmp_total_le (live : Nat → Prop) [DecidablePred live] (A : Rows Int) (C : Nat → List Nat)
    (ca : Nat → Nat) (r : Nat) (hnn : ∀ i, ∀ e ∈ A i, 0 ≤ e.2) :
    lsum (contrTmp live A C ca r) Prod.snd ≤ lsum (C r) (fun i => lsum (A i) Prod.snd) := by
  simp only [contrTmp, lsum_flatMap, lsum_filterMap]
  refine lsum_le (fun i _ => lsum_le (fun e he => ?_))
  by_cases h : live e.1 ∧ ca e.1 ≠ r
  · simp [h]
  · simp only [h, ite_false]; exact hnn i e he

/-- A part row's values are nonnegative and at most the previous level's total. -/
theorem prow_bounds (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt)
    (hnn : ∀ x, ∀ e ∈ pA x, 0 ≤ e.2) (s : Nat) :
    ∀ e ∈ prow srt pA N s, 0 ≤ e.2 ∧ e.2 ≤ lsum (List.range nprev) (fun x => lsum (pA x) Prod.snd) := by
  have hnnT : ∀ e ∈ srt (contrTmp (kept N) pA N.mem N.cmap s), 0 ≤ e.2 := by
    intro e he
    have := ((hS _).1.mem_iff).1 he
    simp only [contrTmp, List.mem_flatMap, List.mem_filterMap] at this
    obtain ⟨x, _, e', he', hx⟩ := this
    by_cases h : kept N e'.1 ∧ N.cmap e'.1 ≠ s
    · have hx' : (N.cmap e'.1, e'.2) = e := by simpa [h] using hx
      rw [← hx']; exact hnn x e' he'
    · simp [h] at hx
  intro e he
  obtain ⟨h1, h2⟩ := grp_val_le _ hnnT e he
  refine ⟨h1, Int.le_trans h2 ?_⟩
  rw [lsum_perm (hS _).1]
  refine Int.le_trans (contrTmp_total_le _ _ _ _ s hnn) ?_
  have hmem := hp.contr
  rw [← lsum_range_indicator (hmem.cNodup s) (hmem.cLt s)]
  refine lsum_le (fun x _ => ?_)
  by_cases hx : x ∈ N.mem s
  · simp [hx]
  · simp only [hx, ite_false]; exact lsum_nonneg (fun e he => hnn x e he)

/-- **B2's values**: each stored value of `b2_splice`'s row is a value of the parent's row (the clean
part), or a part-row sum, nonnegative and at most the previous level's total (a part row, or its
transposed copy in `tcol`). -/
theorem hb2_vals (hp : B2Pre ncells nprev parA pA N) (hS : SortSpec srt)
    (hnn : ∀ x, ∀ e ∈ pA x, 0 ≤ e.2) (i : Nat) :
    ∀ e ∈ b2Rows srt parA pA N i,
      e ∈ parA i ∨ (0 ≤ e.2 ∧ e.2 ≤ lsum (List.range nprev) (fun x => lsum (pA x) Prod.snd)) := by
  intro e he
  unfold b2Rows at he
  split at he
  · rcases List.mem_append.1 he with h | h
    · left
      split at h
      · exact (List.mem_filter.1 h).1
      · simp at h
    · right
      simp only [tcolRow, List.mem_filterMap, Option.map_eq_some_iff] at h
      obtain ⟨s, _, v, hv, rfl⟩ := h
      exact prow_bounds hp hS hnn s (i, v) (mem_of_lookup hv)
  · exact Or.inr (prow_bounds hp hS hnn i e he)

end

/-- **The gathered graph's values are its blocks' values** (no new sums), and a gathered row is no longer
than the row it comes from: capstop's fresh graph inherits the stopped levels' bounds. -/
theorem gath_vals (A : Rows Int) (ids : List Nat) (off x : Nat) :
    (∀ e ∈ gathRow A ids off x, ∃ e' ∈ A x, e'.2 = e.2) ∧ (gathRow A ids off x).length ≤ (A x).length := by
  refine ⟨fun e he => ?_, List.length_filterMap_le _ _⟩
  simp only [gathRow, List.mem_filterMap, Option.map_eq_some_iff] at he
  obtain ⟨e', he', _, _, rfl⟩ := he
  exact ⟨e', he', rfl⟩

end CchCoarsen
