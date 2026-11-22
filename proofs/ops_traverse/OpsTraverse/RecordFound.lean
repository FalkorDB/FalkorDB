/-
# `record_found_path` (graph/src/runtime/functions/algo_procedures.rs)

Folds a newly found path into the result set and tightens the weight bound. Three disciplines by
`path_count`: `1` the single best by `cmp_found_path`, `0` every path tied at the minimum weight,
`k` the `k` best kept sorted. The comparator is a parameter `cmp` (`AlgoPaths.cmpFound`), the weight
tests `gtF`/`ltF` are `f64` `>` / `<`.
-/
namespace OpsTraverse.RecordFound

variable {α F : Type}

/-- The sorted insertion of the `k` branch: `partition_point(|kept| cmp(kept, found) == Less)` then
`insert` at that index. -/
def ins (cmp : α → α → Ordering) (found : α) (rs : List α) : List α :=
  rs.takeWhile (fun e => cmp e found == .lt) ++ found :: rs.dropWhile (fun e => cmp e found == .lt)

/-- `record_found_path`. `pcount` is `path_count` (`k as usize` in the last arm). -/
def record (cmp : α → α → Ordering) (w : α → F) (gtF ltF : F → F → Bool) (pcount : Nat)
    (rs : List α) (found : α) (bound : F) : List α × F :=
  match pcount with
  | 1 =>
    if rs.head?.all (fun best => cmp found best == .lt) then ([found], w found) else (rs, bound)
  | 0 =>
    match rs.head? with
    | some best =>
      if gtF (w found) (w best) then (rs, bound)
      else if ltF (w found) (w best) then ([found], w found)
      else (rs ++ [found], bound)
    | none => ([found], w found)
  | k =>
    if rs.length == k && cmp found (rs.getLastD found) != .lt then (rs, bound)
    else
      let rs1 := if rs.length == k then rs.dropLast else rs
      let rs2 := ins cmp found rs1
      (rs2, if rs2.length == k then w (rs2.getLastD found) else bound)

/-- **PROVEN** (`path_count = 1`): keep the one path that compares `Less` than the current best
(or the first one), and move the bound to its weight; otherwise nothing changes. -/
theorem record_one (cmp : α → α → Ordering) (w : α → F) (gtF ltF : F → F → Bool)
    (rs : List α) (found : α) (bound : F) :
    record cmp w gtF ltF 1 rs found bound =
      (if rs.head?.all (fun best => cmp found best == .lt) then ([found], w found) else (rs, bound)) := rfl

theorem record_one_len (cmp : α → α → Ordering) (w : α → F) (gtF ltF : F → F → Bool)
    (rs : List α) (found : α) (bound : F) (h : rs.length ≤ 1) :
    (record cmp w gtF ltF 1 rs found bound).1.length ≤ 1 := by
  simp only [record]
  split
  · simp
  · exact h

/-- **PROVEN** (`path_count = 0`): a heavier path is dropped, a lighter one replaces the set, any
other (equal or unordered weight) is appended; the first path sets the bound. -/
theorem record_zero (cmp : α → α → Ordering) (w : α → F) (gtF ltF : F → F → Bool)
    (rs : List α) (found : α) (bound : F) :
    (rs = [] → record cmp w gtF ltF 0 rs found bound = ([found], w found)) ∧
    (∀ best t, rs = best :: t → gtF (w found) (w best) = true →
        record cmp w gtF ltF 0 rs found bound = (rs, bound)) ∧
    (∀ best t, rs = best :: t → gtF (w found) (w best) = false → ltF (w found) (w best) = true →
        record cmp w gtF ltF 0 rs found bound = ([found], w found)) ∧
    (∀ best t, rs = best :: t → gtF (w found) (w best) = false → ltF (w found) (w best) = false →
        record cmp w gtF ltF 0 rs found bound = (rs ++ [found], bound)) := by
  refine ⟨fun h => ?_, fun b t h g => ?_, fun b t h g l => ?_, fun b t h g l => ?_⟩
  · subst h; simp [record]
  · subst h; simp [record, g]
  · subst h; simp [record, g, l]
  · subst h; simp [record, g, l]

theorem tw_mem {α : Type} (p : α → Bool) : ∀ (l : List α) (x : α), x ∈ l.takeWhile p → p x = true
  | [], x, h => by simp at h
  | a :: l, x, h => by
    simp only [List.takeWhile] at h
    split at h
    · next hp =>
      rcases List.mem_cons.mp h with rfl | hm
      · exact hp
      · exact tw_mem p l x hm
    · simp at h

section Sorted
variable (cmp : α → α → Ordering)

/-- Ascending by `cmp`. -/
def Sorted (rs : List α) : Prop := rs.Pairwise (fun a b => cmp a b ≠ .gt)

variable {cmp}

theorem ins_perm (found : α) (rs : List α) : (ins cmp found rs).Perm (found :: rs) := by
  unfold ins
  have h := List.takeWhile_append_dropWhile (p := fun e => cmp e found == .lt) (l := rs)
  have : (rs.takeWhile (fun e => cmp e found == .lt) ++ found :: rs.dropWhile (fun e => cmp e found == .lt)).Perm
      (found :: (rs.takeWhile (fun e => cmp e found == .lt) ++ rs.dropWhile (fun e => cmp e found == .lt))) :=
    List.perm_middle
  rw [h] at this
  exact this

/-- **PROVEN**: inserting at the partition point keeps a list ascending, for a `cmp` that is
transitive on `≠ Greater` and antisymmetric (`swap`) — what `cmp_found_path` is on ordered weights
(`AlgoPaths.cmpFound_swap`). -/
theorem ins_sorted (htrans : ∀ a b c, cmp a b ≠ .gt → cmp b c ≠ .gt → cmp a c ≠ .gt)
    (hanti : ∀ a b, cmp b a = (cmp a b).swap) (found : α) (rs : List α) (hs : Sorted cmp rs) :
    Sorted cmp (ins cmp found rs) := by
  unfold Sorted ins
  have hsplit := List.takeWhile_append_dropWhile (p := fun e => cmp e found == .lt) (l := rs)
  have hs' := hs
  unfold Sorted at hs'
  rw [← hsplit] at hs'
  rw [List.pairwise_append] at hs'
  obtain ⟨hl, hr, hlr⟩ := hs'
  rw [List.pairwise_append]
  refine ⟨hl, ?_, ?_⟩
  · rw [List.pairwise_cons]
    refine ⟨fun y hy => ?_, hr⟩
    -- found ≤ every kept-after element
    cases hd : rs.dropWhile (fun e => cmp e found == .lt) with
    | nil => rw [hd] at hy; simp at hy
    | cons e0 rest =>
      have hne : ¬ (cmp e0 found == .lt) = true := by
        have h2 : rs.dropWhile (fun e => cmp e found == .lt) ≠ [] := by rw [hd]; simp
        have := List.head_dropWhile_not (fun e => cmp e found == .lt) h2
        simpa [hd] using this
      have h0 : cmp found e0 ≠ .gt := by
        intro hg
        apply hne
        have := hanti found e0
        rw [hg] at this
        simp [this]
      rw [hd] at hy hr
      rcases List.mem_cons.mp hy with rfl | hm
      · exact h0
      · have := (List.pairwise_cons.mp hr).1 y hm
        exact htrans _ _ _ h0 this
  · intro x hx y hy
    rcases List.mem_cons.mp hy with h | hm
    · have := tw_mem (fun e => cmp e found == .lt) rs x hx
      simp only [beq_iff_eq] at this
      rw [h, this]; simp
    · exact hlr x hx y hm

end Sorted

/-- **PROVEN** (`path_count = k > 1`): the result keeps at most `k` paths, ascending by `cmp`, and
every kept path is the new one or an old one; it is rejected exactly when the set is full and it
does not compare `Less` than the worst kept. -/
theorem record_k (cmp : α → α → Ordering) (w : α → F) (gtF ltF : F → F → Bool) (k : Nat) (hk : 2 ≤ k)
    (rs : List α) (found : α) (bound : F)
    (htrans : ∀ a b c, cmp a b ≠ .gt → cmp b c ≠ .gt → cmp a c ≠ .gt)
    (hanti : ∀ a b, cmp b a = (cmp a b).swap)
    (hs : Sorted cmp rs) (hlen : rs.length ≤ k) :
    let r := (record cmp w gtF ltF k rs found bound).1
    r.length ≤ k ∧ Sorted cmp r ∧ (∀ x ∈ r, x = found ∨ x ∈ rs) ∧
    ((rs.length = k ∧ cmp found (rs.getLastD found) ≠ .lt) → r = rs) := by
  have hk0 : k ≠ 0 := by omega
  have hk1 : k ≠ 1 := by omega
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 2 := ⟨k - 2, by omega⟩
  simp only [record]
  by_cases hrej : (rs.length == j + 2 && cmp found (rs.getLastD found) != .lt) = true
  · simp only [hrej, ↓reduceIte]
    simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hrej
    exact ⟨hlen, hs, fun x hx => Or.inr hx, fun _ => by trivial⟩
  · simp only [hrej, Bool.false_eq_true, ↓reduceIte]
    have hsub : Sorted cmp (if (rs.length == j + 2) = true then rs.dropLast else rs) := by
      split
      · exact List.Pairwise.sublist (List.dropLast_sublist _) hs
      · exact hs
    have hlen1 : (if (rs.length == j + 2) = true then rs.dropLast else rs).length ≤ j + 1 := by
      split
      · next h => simp at h; simp [h]
      · next h => simp at h; omega
    have hp := ins_perm (cmp := cmp) found (if (rs.length == j + 2) = true then rs.dropLast else rs)
    refine ⟨?_, ins_sorted htrans hanti found _ hsub, fun x hx => ?_, fun h => ?_⟩
    · have := hp.length_eq
      rw [List.length_cons] at this; omega
    · have := hp.subset hx
      rcases List.mem_cons.mp this with h | h
      · exact Or.inl h
      · right
        split at h
        · exact List.dropLast_subset _ h
        · exact h
    · exfalso; apply hrej
      simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq]
      exact h

end OpsTraverse.RecordFound
