/-
# `versioned_matrix::Iter`: the three-way sorted merge yields the logical
# matrix, sorted, each entry exactly once

Model of `impl Iterator for Iter<E>` (`versioned_matrix.rs:1478-1521`).

A GraphBLAS row iterator yields a layer's stored entries in ascending
`(row, col)` order; a coordinate is modelled by its position in that order, a
`Nat` key (row-major order on a bounded grid is order-isomorphic to `Nat`).
Each stream element carries a payload `α` (`()` for `BoolExtract`, the `u64`
edge id for `Uint64Extract`), so the theorem also covers `Tensor`'s valued
forward iterator, where `dp` may *shadow* `m` and must win.

A layer's `GxB_Iterator` plus the struct's one-element lookahead
(`m_next` / `dp_next` / `dm_next`) is, observationally, one list: the
buffered element followed by what the GraphBLAS iterator has not yet
produced. So the iterator state is three lists `M`, `D`, `P`.

| here | there |
| --- | --- |
| `dropLt`   | `while let Some(dm) = self.dm_next { if dm < mp { advance } else break }` (:1488-1494) |
| `skipDel`  | the outer `while let Some(m) = &self.m_next` loop (:1486-1501) |
| `next`     | the final `match (&self.m_next, &self.dp_next)` (:1505-1520) |
| `drain`    | `Iterator::collect` over `next` |
| `mergeW`   | the specification: sorted union, `dp` wins on a shared key |

The result, `drain_eq`: the stream is exactly
`mergeW (M.filter (key ∉ D)) P` — sorted, duplicate-free, and containing
`(m ∖ dm) ∪ dp` with `dp`'s value on a shadowed key. Notably this needs only
that each stream is strictly ascending: **not** `dm ⊆ m` (a stray tombstone
is skipped harmlessly by `dropLt`) and **not** `dp ∩ m = ∅` (a shared key is
emitted once, from `dp`).
-/

namespace VMIter

variable {α : Type}

abbrev Entry (α : Type) := Nat × α

/-- Strictly ascending by key: what a GraphBLAS row iterator yields. -/
def Sorted (l : List (Entry α)) : Prop := l.Pairwise (fun a b => a.1 < b.1)
def SortedN (l : List Nat) : Prop := l.Pairwise (· < ·)

/-- Advance the tombstone stream past every key `< k`. -/
def dropLt (k : Nat) : List Nat → List Nat
  | [] => []
  | d :: ds => if d < k then dropLt k ds else d :: ds

/-- Drop every leading `m` entry whose key the tombstone stream covers. -/
def skipDel : List (Entry α) → List Nat → List (Entry α) × List Nat
  | [], ds => ([], ds)
  | m :: ms, ds =>
    match dropLt m.1 ds with
    | d :: ds' => if d = m.1 then skipDel ms ds' else (m :: ms, d :: ds')
    | [] => (m :: ms, [])

structure St (α : Type) where
  M : List (Entry α)
  D : List Nat
  P : List (Entry α)

/-- One call of `Iter::next`. -/
def next (s : St α) : Option (Entry α) × St α :=
  let r := skipDel s.M s.D
  match r.1, s.P with
  | m :: ms, d :: ps =>
    if d.1 ≤ m.1 then (some d, ⟨if d.1 = m.1 then ms else m :: ms, r.2, ps⟩)
    else (some m, ⟨ms, r.2, d :: ps⟩)
  | m :: ms, [] => (some m, ⟨ms, r.2, []⟩)
  | [], d :: ps => (some d, ⟨[], r.2, ps⟩)
  | [], [] => (none, ⟨[], r.2, []⟩)

/-- Collect with a call budget. -/
def drain : Nat → St α → List (Entry α)
  | 0, _ => []
  | n + 1, s =>
    match next s with
    | (none, _) => []
    | (some x, s') => x :: drain n s'

/-- The specification: sorted union by key, `P` (dp) winning a tie. -/
def mergeW : List (Entry α) → List (Entry α) → List (Entry α)
  | [], ps => ps
  | m :: ms, [] => m :: ms
  | m :: ms, p :: ps =>
    if p.1 ≤ m.1 then p :: mergeW (if p.1 = m.1 then ms else m :: ms) ps
    else m :: mergeW ms (p :: ps)
termination_by a b => a.length + b.length
decreasing_by all_goals (simp_wf; try split) <;> simp_all <;> omega

def live (M : List (Entry α)) (D : List Nat) : List (Entry α) :=
  M.filter (fun e => !(decide (e.1 ∈ D)))

/-! ### `dropLt` -/

theorem dropLt_sub (k : Nat) : ∀ (D : List Nat), (dropLt k D).Sublist D
  | [] => by simp [dropLt]
  | d :: ds => by
    unfold dropLt; split
    · exact (dropLt_sub k ds).cons d
    · exact List.Sublist.refl _

theorem dropLt_mem {k : Nat} : ∀ {D : List Nat}, SortedN D → ∀ x, k ≤ x → (x ∈ dropLt k D ↔ x ∈ D)
  | [], _, _, _ => by simp [dropLt]
  | d :: ds, h, x, hx => by
    have hs : SortedN ds := (List.pairwise_cons.1 h).2
    unfold dropLt; split
    · rename_i hlt
      rw [dropLt_mem hs x hx]; simp only [List.mem_cons]
      constructor
      · exact Or.inr
      · rintro (h' | h')
        · omega
        · exact h'
    · exact Iff.rfl

theorem dropLt_head {k : Nat} : ∀ {D : List Nat} {d : Nat} {ds : List Nat}, dropLt k D = d :: ds → k ≤ d
  | [], _, _, h => by simp [dropLt] at h
  | e :: es, d, ds, h => by
    unfold dropLt at h; split at h
    · exact dropLt_head h
    · simp at h; omega

theorem sortedN_sub {l l' : List Nat} (h : l'.Sublist l) (hs : SortedN l) : SortedN l' :=
  List.Pairwise.sublist h hs

theorem sorted_sub {l l' : List (Entry α)} (h : l'.Sublist l) (hs : Sorted l) : Sorted l' :=
  List.Pairwise.sublist h hs

/-! ### `skipDel` -/

theorem skipDel_spec : ∀ (M : List (Entry α)) (D : List Nat), Sorted M → SortedN D →
    let r := skipDel M D
    live r.1 r.2 = live M D ∧ r.1.Sublist M ∧ r.2.Sublist D ∧
      (∀ m ms, r.1 = m :: ms → m.1 ∉ r.2)
  | [], D, _, _ => by simp [skipDel, live]
  | m :: ms, D, hM, hD => by
    have hms : Sorted ms := (List.pairwise_cons.1 hM).2
    have hgt : ∀ e ∈ ms, m.1 < e.1 := (List.pairwise_cons.1 hM).1
    have hsub := dropLt_sub m.1 D
    have hmem : ∀ x, m.1 ≤ x → (x ∈ dropLt m.1 D ↔ x ∈ D) := dropLt_mem hD
    -- the live part of `m :: ms` only depends on tombstones `≥ m.1`
    have liveEq : ∀ D' : List Nat, (∀ x, m.1 ≤ x → (x ∈ D' ↔ x ∈ D)) →
        live (m :: ms) D' = live (m :: ms) D := by
      intro D' hD'; unfold live; apply List.filter_congr; intro e he
      have : m.1 ≤ e.1 := by
        rcases List.mem_cons.1 he with rfl | he
        · exact Nat.le_refl _
        · exact Nat.le_of_lt (hgt e he)
      simp only [hD' e.1 this]
    simp only [skipDel]
    split
    · rename_i d ds' heq
      have hd : m.1 ≤ d := dropLt_head heq
      split
      · rename_i hdm
        have hDs : SortedN ds' := by
          have : SortedN (dropLt m.1 D) := sortedN_sub hsub hD
          rw [heq] at this; exact (List.pairwise_cons.1 this).2
        have ih := skipDel_spec ms ds' hms hDs
        refine ⟨?_, ih.2.1.cons m, ?_, ih.2.2.2⟩
        · rw [ih.1]
          have hl : live (m :: ms) D = live ms D := by
            have hmD : m.1 ∈ D := by rw [← hmem m.1 (Nat.le_refl _), heq, hdm]; simp
            simp [live, hmD]
          rw [hl]; unfold live; apply List.filter_congr; intro e he
          have hlt := hgt e he
          have : e.1 ∈ ds' ↔ e.1 ∈ D := by
            rw [← hmem e.1 (Nat.le_of_lt hlt), heq, List.mem_cons]
            constructor
            · exact Or.inr
            · rintro (h' | h')
              · omega
              · exact h'
          simp only [this]
        · have h2 : ds'.Sublist (dropLt m.1 D) := by rw [heq]; exact List.sublist_cons_self d ds'
          exact ih.2.2.1.trans (h2.trans hsub)
      · rename_i hdm
        refine ⟨?_, List.Sublist.refl _, by rw [← heq]; exact hsub, ?_⟩
        · apply liveEq; intro x hx; rw [← heq]; exact hmem x hx
        · intro m' ms' h'; simp at h'; obtain ⟨rfl, rfl⟩ := h'
          intro hin
          have hDd : SortedN (d :: ds') := by rw [← heq]; exact sortedN_sub hsub hD
          rcases List.mem_cons.1 hin with h1 | h1
          · exact hdm h1.symm
          · have := (List.pairwise_cons.1 hDd).1 _ h1; omega
    · rename_i heq
      refine ⟨?_, List.Sublist.refl _, List.nil_sublist _, ?_⟩
      · apply liveEq; intro x hx; rw [← hmem x hx, heq]
      · intro _ _ _; simp

/-! ### one `next` step against the spec -/

theorem live_cons_of_not_mem {m : Entry α} {ms : List (Entry α)} {D : List Nat} (h : m.1 ∉ D) :
    live (m :: ms) D = m :: live ms D := by
  simp [live, h]

theorem mergeW_cons_le {m p : Entry α} {ms ps : List (Entry α)} (h : p.1 ≤ m.1) :
    mergeW (m :: ms) (p :: ps) = p :: mergeW (if p.1 = m.1 then ms else m :: ms) ps := by
  rw [mergeW]; simp [h]

theorem mergeW_cons_gt {m p : Entry α} {ms ps : List (Entry α)} (h : ¬ p.1 ≤ m.1) :
    mergeW (m :: ms) (p :: ps) = m :: mergeW ms (p :: ps) := by
  rw [mergeW]; simp [h]

def Good (s : St α) : Prop := Sorted s.M ∧ SortedN s.D ∧ Sorted s.P

def spec (s : St α) : List (Entry α) := mergeW (live s.M s.D) s.P
def meas (s : St α) : Nat := s.M.length + s.P.length

theorem next_spec (s : St α) (h : Good s) :
    ((next s).1 = none ∧ spec s = []) ∨
    (∃ x s', next s = (some x, s') ∧ spec s = x :: spec s' ∧ Good s' ∧ meas s' < meas s) := by
  obtain ⟨hM, hD, hP⟩ := h
  have sp := skipDel_spec s.M s.D hM hD
  obtain ⟨hlive, hsubM, hsubD, hhead⟩ := sp
  have hlenM : (skipDel s.M s.D).1.length ≤ s.M.length := hsubM.length_le
  have hMr : Sorted (skipDel s.M s.D).1 := sorted_sub hsubM hM
  have hDr : SortedN (skipDel s.M s.D).2 := sortedN_sub hsubD hD
  have hspec : spec s = mergeW (live (skipDel s.M s.D).1 (skipDel s.M s.D).2) s.P := by
    rw [spec, hlive]
  unfold next meas
  rw [hspec]
  revert hlive hsubM hsubD hhead hlenM hMr hDr
  generalize skipDel s.M s.D = r
  intro _ _ _ hhead hlenM hMr hDr
  obtain ⟨M', D'⟩ := r
  simp only at hhead hlenM hMr hDr ⊢
  obtain ⟨sM, sD, sP⟩ := s
  simp only at hP hlenM ⊢
  cases M' with
  | nil =>
    cases sP with
    | nil => left; simp [live, mergeW]
    | cons d ps =>
      right
      refine ⟨d, ⟨[], D', ps⟩, rfl, ?_, ⟨List.Pairwise.nil, hDr, (List.pairwise_cons.1 hP).2⟩, ?_⟩
      · simp [spec, live, mergeW]
      · simp [meas]; omega
  | cons m ms =>
    have hmD := hhead m ms rfl
    have hms : Sorted ms := (List.pairwise_cons.1 hMr).2
    rw [live_cons_of_not_mem hmD]
    simp only [List.length_cons] at hlenM
    right
    cases sP with
    | nil =>
      refine ⟨m, ⟨ms, D', []⟩, rfl, ?_, ⟨hms, hDr, List.Pairwise.nil⟩, ?_⟩
      · simp only [spec]
        cases live ms D' <;> simp [mergeW]
      · simp [meas]; omega
    | cons d ps =>
      have hps : Sorted ps := (List.pairwise_cons.1 hP).2
      by_cases hle : d.1 ≤ m.1
      · refine ⟨d, ⟨if d.1 = m.1 then ms else m :: ms, D', ps⟩, by simp [hle], ?_, ⟨?_, hDr, hps⟩, ?_⟩
        · rw [mergeW_cons_le hle]; simp only [spec]
          by_cases he : d.1 = m.1 <;> simp [he, live_cons_of_not_mem hmD]
        · by_cases he : d.1 = m.1 <;> simp [he, hms, hMr]
        · simp only [meas]; by_cases he : d.1 = m.1 <;> simp [he] <;> omega
      · refine ⟨m, ⟨ms, D', d :: ps⟩, by simp [hle], ?_, ⟨hms, hDr, hP⟩, ?_⟩
        · rw [mergeW_cons_gt hle]; rfl
        · simp [meas]; omega

/-- **The iterator yields exactly the specification.** Any budget of at least
`|M| + |P| + 1` calls drains it. -/
theorem drain_eq : ∀ (n : Nat) (s : St α), Good s → meas s < n → drain n s = spec s
  | 0, _, _, h => by omega
  | n + 1, s, hg, hn => by
    rcases next_spec s hg with ⟨h1, h2⟩ | ⟨x, s', h1, h2, h3, h4⟩
    · unfold drain
      cases hq : next s with
      | mk o s'' => simp only [hq] at h1; subst h1; simp [h2]
    · unfold drain; rw [h1]; simp only
      rw [h2, drain_eq n s' h3 (by omega)]

/-! ### properties of the specification -/

theorem mem_mergeW : ∀ (a b : List (Entry α)) (x : Entry α),
    x ∈ mergeW a b → x ∈ a ∨ x ∈ b
  | [], b, x, h => by simp [mergeW] at h; exact Or.inr h
  | m :: ms, [], x, h => by simp [mergeW] at h; exact Or.inl (by simpa using h)
  | m :: ms, p :: ps, x, h => by
    rw [mergeW] at h
    split at h
    · rcases List.mem_cons.1 h with rfl | h
      · exact Or.inr (by simp)
      · rcases mem_mergeW _ ps x h with h | h
        · split at h
          · exact Or.inl (List.mem_cons_of_mem _ h)
          · exact Or.inl h
        · exact Or.inr (List.mem_cons_of_mem _ h)
    · rcases List.mem_cons.1 h with rfl | h
      · exact Or.inl (by simp)
      · rcases mem_mergeW ms _ x h with h | h
        · exact Or.inl (List.mem_cons_of_mem _ h)
        · exact Or.inr h

/-- Every key that is in either input is in the output (dp's entry wins a tie). -/
theorem key_mem_mergeW : ∀ (a b : List (Entry α)) (k : Nat),
    ((∃ e ∈ a, e.1 = k) ∨ (∃ e ∈ b, e.1 = k)) → ∃ e ∈ mergeW a b, e.1 = k
  | [], b, k, h => by simpa [mergeW] using h
  | m :: ms, [], k, h => by simpa [mergeW] using h
  | m :: ms, p :: ps, k, h => by
    rw [mergeW]
    split
    · rename_i hle
      by_cases hk : p.1 = k
      · exact ⟨p, by simp, hk⟩
      · obtain ⟨e, he, hek⟩ := key_mem_mergeW (if p.1 = m.1 then ms else m :: ms) ps k (by
          rcases h with ⟨e, he, hek⟩ | ⟨e, he, hek⟩
          · rcases List.mem_cons.1 he with rfl | he
            · left; split
              · rename_i h'; exact absurd (h'.trans hek) hk
              · exact ⟨e, by simp, hek⟩
            · left; split
              · exact ⟨e, he, hek⟩
              · exact ⟨e, List.mem_cons_of_mem _ he, hek⟩
          · rcases List.mem_cons.1 he with rfl | he
            · exact absurd hek hk
            · exact Or.inr ⟨e, he, hek⟩)
        exact ⟨e, List.mem_cons_of_mem _ he, hek⟩
    · rename_i hgt
      by_cases hk : m.1 = k
      · exact ⟨m, by simp, hk⟩
      · obtain ⟨e, he, hek⟩ := key_mem_mergeW ms (p :: ps) k (by
          rcases h with ⟨e, he, hek⟩ | h
          · rcases List.mem_cons.1 he with rfl | he
            · exact absurd hek hk
            · exact Or.inl ⟨e, he, hek⟩
          · exact Or.inr h)
        exact ⟨e, List.mem_cons_of_mem _ he, hek⟩

theorem mergeW_lower : ∀ (a b : List (Entry α)) (y : Nat),
    (∀ e ∈ a, y < e.1) → (∀ e ∈ b, y < e.1) → ∀ e ∈ mergeW a b, y < e.1 := by
  intro a b y ha hb e he
  rcases mem_mergeW a b e he with h | h
  · exact ha e h
  · exact hb e h

/-- The output is strictly ascending by key: **each coordinate exactly once**. -/
theorem mergeW_sorted : ∀ (a b : List (Entry α)), Sorted a → Sorted b → Sorted (mergeW a b)
  | [], b, _, hb => by simpa [mergeW] using hb
  | m :: ms, [], ha, _ => by simpa [mergeW] using ha
  | m :: ms, p :: ps, ha, hb => by
    have hms := (List.pairwise_cons.1 ha).2
    have hps := (List.pairwise_cons.1 hb).2
    have gm := (List.pairwise_cons.1 ha).1
    have gp := (List.pairwise_cons.1 hb).1
    rw [mergeW]
    split
    · rename_i hle
      refine List.pairwise_cons.2 ⟨?_, ?_⟩
      · apply mergeW_lower
        · split
          · rename_i heq; intro e he; have := gm e he; omega
          · intro e he; rcases List.mem_cons.1 he with rfl | he
            · rename_i hne; omega
            · have := gm e he; omega
        · exact gp
      · apply mergeW_sorted
        · split
          · exact hms
          · exact ha
        · exact hps
    · rename_i hgt
      refine List.pairwise_cons.2 ⟨?_, mergeW_sorted ms (p :: ps) hms hb⟩
      apply mergeW_lower
      · exact gm
      · intro e he; rcases List.mem_cons.1 he with rfl | he
        · omega
        · have := gp e he; omega

theorem live_sorted {M : List (Entry α)} (D : List Nat) (h : Sorted M) : Sorted (live M D) :=
  List.Pairwise.sublist (List.filter_sublist) h

/-- **Iterator correctness.** Starting from the three layer streams (strictly
ascending, as GraphBLAS yields them), the iterator emits a strictly
ascending stream — so every coordinate exactly once — containing precisely
the keys of `(m ∖ dm) ∪ dp`, each element coming from `m` or `dp`. -/
theorem iter_correct (M P : List (Entry α)) (D : List Nat)
    (hM : Sorted M) (hD : SortedN D) (hP : Sorted P) :
    let out := drain (M.length + P.length + 1) ⟨M, D, P⟩
    Sorted out ∧
      (∀ x ∈ out, (x ∈ M ∧ x.1 ∉ D) ∨ x ∈ P) ∧
      (∀ k, ((∃ e ∈ M, e.1 = k ∧ k ∉ D) ∨ (∃ e ∈ P, e.1 = k)) → ∃ e ∈ out, e.1 = k) := by
  simp only
  rw [drain_eq _ _ ⟨hM, hD, hP⟩ (by simp [meas]), spec]
  refine ⟨mergeW_sorted _ _ (live_sorted D hM) hP, ?_, ?_⟩
  · intro x hx
    rcases mem_mergeW _ _ x hx with h | h
    · left; simp only [live, List.mem_filter, Bool.not_eq_true', decide_eq_false_iff_not] at h
      exact h
    · exact Or.inr h
  · intro k hk
    apply key_mem_mergeW
    rcases hk with ⟨e, he, hek, hkD⟩ | h
    · left; refine ⟨e, ?_, hek⟩
      simp only [live, List.mem_filter, Bool.not_eq_true', decide_eq_false_iff_not]
      exact ⟨he, hek ▸ hkD⟩
    · exact Or.inr h

/-- Shadowing: when `dp` and `m` share a key, the `dp` entry is emitted and the
`m` entry is not (the in-place value update of `Tensor`'s forward layer). -/
theorem shadow_dp_wins :
    drain 10 (⟨[(5, "m")], [], [(5, "dp")]⟩ : St String) = [(5, "dp")] := by decide

/-- A tombstone with no committed entry under it (`dm ⊄ m`) is harmless to the
iterator — it is skipped by `dropLt`. -/
theorem stray_tombstone_harmless :
    drain 10 (⟨[(3, ()), (7, ())], [5], []⟩ : St Unit) = [(3, ()), (7, ())] := by decide

end VMIter
