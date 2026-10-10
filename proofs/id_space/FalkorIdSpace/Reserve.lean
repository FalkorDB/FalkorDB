import FalkorIdSpace.Sets
/-! `new`, `restored`, `reclaim_ids`, `reserve`: what they return. -/
namespace IdSpaceModel

/-- `IdSpace::new`: nothing live, nothing free, an empty batch at 0. -/
theorem new_spec : (IdSpace.new).live = 0 ∧ (IdSpace.new).eb = 0 ∧
    (∀ i, (IdSpace.new).recycled i = false) ∧ ∀ i, (IdSpace.new).taken i = false :=
  ⟨rfl, rfl, fun _ => rfl, fun _ => rfl⟩

/-- `IdSpace::restored`: the batch opens at the decoded boundary. -/
theorem restored_spec (N live : Nat) (rec : IdSet) :
    (IdSpace.restored N live rec).eb = (IdSpace.restored N live rec).bound N ∧
    (IdSpace.restored N live rec).live = live ∧ (IdSpace.restored N live rec).recycled = rec ∧
    ∀ i, (IdSpace.restored N live rec).taken i = false :=
  ⟨rfl, rfl, rfl, fun _ => rfl⟩

/-- `new_version` is `restored` over the same live count and free set: the
batch is not carried forward. -/
theorem newVersion_spec (N : Nat) (sp : IdSpace) :
    sp.newVersion N = IdSpace.restored N sp.live sp.recycled := rfl

/-- `max_id` never underflows: when something is live the boundary is ≥ 1. -/
theorem maxId_spec (N : Nat) (sp : IdSpace) :
    sp.maxId N = (if sp.live = 0 then 0 else sp.bound N - 1) ∧
    (sp.live ≠ 0 → 1 ≤ sp.bound N) := by
  refine ⟨rfl, fun h => ?_⟩
  unfold IdSpace.bound; omega

/-- `pool - taken - issued` is "in the pool, taken by nobody, issued to nobody". -/
theorem freeOf_spec (pool taken issued : IdSet) (i : Nat) :
    freeOf pool taken issued i = (pool i && !taken i && !issued i) := by
  simp [freeOf, sDiff]

theorem mem_free {pool taken issued : IdSet} {i : Nat} :
    freeOf pool taken issued i = true ↔ pool i = true ∧ taken i = false ∧ issued i = false := by
  rw [freeOf_spec]; simp [Bool.and_assoc]

/-- `reclaim_ids` appends the `count` lowest free ids and returns how many it
appended: `min(count, |free|)`. -/
theorem reclaimIds_spec (N : Nat) (pool taken issued : IdSet) (count : Nat) (out : List Nat) :
    reclaimIds N pool taken issued count out =
      (out ++ (asc N (freeOf pool taken issued)).take count,
       min count (cnt N (freeOf pool taken issued))) := by
  simp [reclaimIds, asc_length]

/-- Every reclaimed id was in the pool and is neither taken nor issued. -/
theorem reclaimed_free (N : Nat) (pool taken issued : IdSet) (count : Nat) :
    ∀ x ∈ (asc N (freeOf pool taken issued)).take count,
      x < N ∧ pool x = true ∧ taken x = false ∧ issued x = false := by
  intro x hx
  have := mem_asc.mp (List.mem_of_mem_take hx)
  exact ⟨this.1, mem_free.mp this.2⟩

theorem reclaimed_sorted (N : Nat) (pool taken issued : IdSet) (count : Nat) :
    ((asc N (freeOf pool taken issued)).take count).Pairwise (· < ·) :=
  (asc_sorted N _).sublist (List.take_sublist _ _)

/-- The fresh run starts at `entry_bound + |taken ≥ eb| + |issued ≥ eb|`. -/
def IdSpace.start (N : Nat) (sp : IdSpace) (issued : IdSet) : Nat :=
  sp.eb + above N sp.taken sp.eb + above N issued sp.eb

/-- `reserve` succeeds iff the allocation does, and then returns the reclaimed
ids followed by the fresh run `[start, start + (count - reclaimed))`. -/
theorem reserve_spec (N : Nat) (sp : IdSpace) (count : Nat) (issued : IdSet) :
    sp.reserve N true count issued =
      .ok ((asc N (freeOf sp.recycled sp.taken issued)).take count ++
           List.range' (sp.start N issued)
             (count - min count (cnt N (freeOf sp.recycled sp.taken issued)))) := by
  simp [IdSpace.reserve, reclaimIds_spec, IdSpace.start]

theorem reserve_alloc_fail (N : Nat) (sp : IdSpace) (count : Nat) (issued : IdSet) :
    sp.reserve N false count issued = .error s!"failed to reserve {count} ids" := rfl

/-- It always hands out exactly `count` ids. -/
theorem reserve_length (N : Nat) (sp : IdSpace) (count : Nat) (issued : IdSet) (ids : List Nat)
    (h : sp.reserve N true count issued = .ok ids) : ids.length = count := by
  rw [reserve_spec] at h
  cases h
  simp [List.length_take, asc_length]
  omega

/-- **No id is handed out twice, and none the batch has taken or issued.**
Given the precondition the Rust doc states — what the batch holds at or above
the boundary is accounted for by `start`, i.e. lies below it — and that the bin
only holds ids below the boundary or ids the batch has taken (`Inv.binOk`): the
result is duplicate-free and disjoint from `taken` and `issued`. The reclaimed
part is below the boundary and the fresh part at or above it. -/
theorem reserve_fresh (N : Nat) (sp : IdSpace) (count : Nat) (issued : IdSet) (ids : List Nat)
    (h : sp.reserve N true count issued = .ok ids)
    (hdense : ∀ i, (sp.taken i = true ∨ issued i = true) → sp.eb ≤ i → i < sp.start N issued)
    (hbin : ∀ i, i < N → sp.recycled i = true → i < sp.eb ∨ sp.taken i = true) :
    ids.Nodup ∧ (∀ x ∈ ids, sp.taken x = false ∧ issued x = false) ∧
    (∀ x ∈ ids, (sp.recycled x = true ∧ x < sp.eb) ∨ (sp.start N issued ≤ x)) := by
  rw [reserve_spec] at h
  cases h
  have hrec := reclaimed_free N sp.recycled sp.taken issued count
  have hlo : ∀ x ∈ (asc N (freeOf sp.recycled sp.taken issued)).take count, x < sp.eb := by
    intro x hx
    obtain ⟨hxN, hp, ht, _⟩ := hrec x hx
    rcases hbin x hxN hp with h | h
    · exact h
    · rw [ht] at h; cases h
  have hge : sp.eb ≤ sp.start N issued := by unfold IdSpace.start; omega
  refine ⟨?_, ?_, ?_⟩
  · rw [List.nodup_append]
    refine ⟨(reclaimed_sorted N _ _ _ _).imp (fun h => Nat.ne_of_lt h), List.nodup_range', ?_⟩
    intro a ha b hb e
    have := hlo a ha
    have := (List.mem_range'_1.mp hb).1
    omega
  · intro x hx
    rcases List.mem_append.mp hx with hx | hx
    · exact ⟨(hrec x hx).2.2.1, (hrec x hx).2.2.2⟩
    · have hxs := (List.mem_range'_1.mp hx).1
      constructor
      · cases hsx : sp.taken x with
        | false => rfl
        | true => have := hdense x (.inl hsx) (by omega); omega
      · cases hsx : issued x with
        | false => rfl
        | true => have := hdense x (.inr hsx) (by omega); omega
  · intro x hx
    rcases List.mem_append.mp hx with hx | hx
    · exact .inl ⟨(hrec x hx).2.1, hlo x hx⟩
    · exact .inr (List.mem_range'_1.mp hx).1

end IdSpaceModel
