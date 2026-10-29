import FalkorAggOps.Ops
/-
# Per-group DISTINCT state is keyed by a group id (#2977, fixes #2969)

Before c652e505e the dedup table `Runtime::value_dedupers` was keyed `(distinct node,
hash(group key row))` (`hash_u64`, aggregate.rs:1267 at ca36bfa15), so two groups whose key rows
collided in FxHash shared one seen-set (counterexample removed; in 17ccc5412). Now `GroupMap`'s value carries the group's id
(aggregate.rs:59-65): `groups.len()` at insertion (`next_group_id`, :709 vectorized, :970 per-row,
`0` for the keyless pre-insert :532-535, :893-897), and both paths read it back (`entry.2`, :728,
:976) as the key. The `hash_u64` trait and its `Row` impl were removed.

| here | there |
| --- | --- |
| `insertKey`, `gid` | `groups.entry(key).or_insert_with(.. next_group_id)`; `entry.2` |
| `dcgIdGo` | the per-row DISTINCT check `value_dedupers.entry((didx, group_id))` (aggregate.rs:753, eval.rs:854) |
| `dcgRef` | the exact per-group semantics (`dedupBy` on the group's own values) |
-/
namespace FalkorAggOps.GroupId

variable {K : Type} [DecidableEq K]

/-- The group list in insertion order; a key already present keeps its slot. -/
def insertKey (gs : List K) (k : K) : List K := if k ∈ gs then gs else gs ++ [k]

/-- The id of key `k` = its insertion index (`groups.len()` at insertion). -/
def gid : List K → K → Nat
  | [], _ => 0
  | x :: xs, k => if x = k then 0 else 1 + gid xs k

theorem gid_append_mem (a b : List K) (k : K) (h : k ∈ a) : gid (a ++ b) k = gid a k := by
  induction a with
  | nil => simp at h
  | cons x xs ih =>
    simp only [List.cons_append, gid]
    by_cases hx : x = k
    · simp [hx]
    · have : k ∈ xs := by simpa [Ne.symm hx] using h
      simp [hx, ih this]

theorem gid_append_new (a : List K) (k : K) (h : k ∉ a) : gid (a ++ [k]) k = a.length := by
  induction a with
  | nil => simp [gid]
  | cons x xs ih =>
    have hx : x ≠ k := fun e => h (by simp [e])
    have hk : k ∉ xs := fun e => h (by simp [e])
    simp only [List.cons_append, gid, hx, if_false, List.length_cons]
    rw [ih hk]; omega

/-- A new group gets id = the number of groups so far (`next_group_id = groups.len()`). -/
theorem insert_new_id (gs : List K) (k : K) (h : k ∉ gs) :
    gid (insertKey gs k) k = gs.length := by
  simp [insertKey, h, gid_append_new gs k h]

/-- An existing group keeps its id when other groups are added. -/
theorem gid_stable (gs : List K) (k k' : K) (h : k' ∈ gs) :
    gid (insertKey gs k) k' = gid gs k' := by
  unfold insertKey; split
  · rfl
  · exact gid_append_mem gs [k] k' h

theorem mem_insertKey (gs : List K) (k k' : K) : k' ∈ insertKey gs k ↔ k' ∈ gs ∨ k' = k := by
  unfold insertKey; split
  · rename_i h; constructor
    · intro x; exact Or.inl x
    · rintro (x | rfl) <;> assumption
  · simp

/-- **Distinct groups have distinct ids.** -/
theorem gid_inj (gs : List K) (k1 k2 : K) (h1 : k1 ∈ gs) (h2 : k2 ∈ gs)
    (h : gid gs k1 = gid gs k2) : k1 = k2 := by
  induction gs with
  | nil => simp at h1
  | cons x xs ih =>
    simp only [gid] at h
    by_cases e1 : x = k1
    · by_cases e2 : x = k2
      · rw [← e1, ← e2]
      · exfalso; subst e1; simp [e2] at h; omega
    · by_cases e2 : x = k2
      · exfalso; subst e2; simp [e1] at h
      · simp [e1, e2] at h
        exact ih (by simpa [Ne.symm e1] using h1) (by simpa [Ne.symm e2] using h2) h

/-- Per-group `count(DISTINCT v)` with the seen-set keyed by `(group id, v)` as on main:
`gs` the groups so far, `seen` the table, one input row `(group key, value)` at a time. The
result counts the first-seen values of group `k`. -/
def dcgIdGo (k : K) : List K → List (Nat × Nat) → List (K × Nat) → Nat
  | _, _, [] => 0
  | gs, seen, (k', v) :: rs =>
    let gs' := insertKey gs k'
    let i := gid gs' k'
    if (i, v) ∈ seen then dcgIdGo k gs' seen rs
    else (if k' = k then 1 else 0) + dcgIdGo k gs' ((i, v) :: seen) rs

/-- Exact reference: per group `k`, dedup the pairs `(key, v)` themselves. -/
def dcgRef (k : K) : List (K × Nat) → List (K × Nat) → Nat
  | _, [] => 0
  | seenP, (k', v) :: rs =>
    if (k', v) ∈ seenP then dcgRef k seenP rs
    else (if k' = k then 1 else 0) + dcgRef k ((k', v) :: seenP) rs

/-- Seen table = the reference's seen pairs, read through the ids. -/
def Inv (gs : List K) (seen : List (Nat × Nat)) (seenP : List (K × Nat)) : Prop :=
  (∀ p ∈ seenP, p.1 ∈ gs) ∧
  ∀ i v, (i, v) ∈ seen ↔ ∃ k, k ∈ gs ∧ gid gs k = i ∧ (k, v) ∈ seenP

theorem inv_insert (gs : List K) (seen : List (Nat × Nat)) (seenP : List (K × Nat)) (k' : K)
    (h : Inv gs seen seenP) : Inv (insertKey gs k') seen seenP := by
  obtain ⟨h1, h2⟩ := h
  refine ⟨fun p hp => (mem_insertKey _ _ _).2 (Or.inl (h1 p hp)), fun i v => ?_⟩
  rw [h2]
  constructor
  · rintro ⟨k, hk, rfl, hp⟩
    exact ⟨k, (mem_insertKey _ _ _).2 (Or.inl hk), (gid_stable gs k' k hk).symm ▸ rfl, hp⟩
  · rintro ⟨k, hk, rfl, hp⟩
    have hk0 : k ∈ gs := h1 _ hp
    exact ⟨k, hk0, (gid_stable gs k' k hk0).symm ▸ rfl, hp⟩

theorem dcgIdGo_eq_ref (k : K) :
    ∀ (rs : List (K × Nat)) (gs : List K) (seen : List (Nat × Nat)) (seenP : List (K × Nat)),
      Inv gs seen seenP → dcgIdGo k gs seen rs = dcgRef k seenP rs := by
  intro rs
  induction rs with
  | nil => intros; rfl
  | cons r rs ih =>
    obtain ⟨k', v⟩ := r
    intro gs seen seenP hI
    have hI' := inv_insert gs seen seenP k' hI
    have hk' : k' ∈ insertKey gs k' := (mem_insertKey _ _ _).2 (Or.inr rfl)
    simp only [dcgIdGo, dcgRef]
    generalize insertKey gs k' = gs' at hI' hk' ⊢
    have hiff : ((gid gs' k', v) ∈ seen) ↔ ((k', v) ∈ seenP) := by
      rw [hI'.2]
      constructor
      · rintro ⟨k, hk, hg, hp⟩
        have := gid_inj gs' k k' hk hk' hg
        rwa [this] at hp
      · intro hp; exact ⟨k', hk', rfl, hp⟩
    by_cases hs : (k', v) ∈ seenP
    · simp only [hiff.2 hs, hs, ite_true]
      exact ih gs' seen seenP hI'
    · have hn : (gid gs' k', v) ∉ seen := fun e => hs (hiff.1 e)
      simp only [hn, hs, ite_false]
      congr 1
      refine ih gs' _ _ ⟨?_, fun i v' => ?_⟩
      · intro p hp
        simp only [List.mem_cons] at hp
        rcases hp with rfl | hp
        · exact hk'
        · exact hI'.1 p hp
      · simp only [List.mem_cons, hI'.2 i v']
        constructor
        · rintro (h | ⟨k, hk, hg, hp⟩)
          · obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
            exact ⟨k', hk', rfl, Or.inl rfl⟩
          · exact ⟨k, hk, hg, Or.inr hp⟩
        · rintro ⟨k, hk, hg, hp | hp⟩
          · obtain ⟨rfl, rfl⟩ := Prod.mk.inj hp
            exact Or.inl (by rw [← hg])
          · exact Or.inr ⟨k, hk, hg, hp⟩

/-- **Per-group DISTINCT is exact on main** (#2977): with the seen-set keyed by group ids,
the count for group `k` equals the reference that dedups `(key, value)` pairs, for any input
and any number of groups (no hash, so no collision). -/
theorem distinct_by_group_id_exact (rows : List (K × Nat)) (k : K) :
    dcgIdGo k [] [] rows = dcgRef k [] rows :=
  dcgIdGo_eq_ref k rows [] [] [] ⟨by simp, by simp⟩

/-- **Fixed by #2977**: the filed collision shape. Keys 0 and 2 (colliding under `% 2`), one
value each. The hash-keyed table gave group 2 a count of 0;
the id-keyed table gives 1, as the reference does. -/
theorem fixed2977_collision_example :
    dcgIdGo 2 [] [] [(0, 7), (2, 7)] = 1 ∧ dcgRef 2 [] [(0, 7), (2, 7)] = 1 := by decide

end FalkorAggOps.GroupId
