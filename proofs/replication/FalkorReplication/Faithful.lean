/-
# Faithful copies of the engine steps the abstract model abstracts

From proofs/effects_emit_apply (`ApplyHelpers.lean`: `intern`, `verifyId`,
`applyAddSchema`) and proofs/id_space (`IdSpace::reserve` on a fresh batch, and
the `live`/`recycled` half of `IdSpace` that `Graph` reads its
boundary from), restated over lists so this project stays standalone.
`FalkorReplication.lean` proves the abstract `applyRec`/`idsOf`/`createOne`/
`deleteOne`/`GState.bound` equal to these.
-/
namespace FalkorFaithful

/-- `findIdx?` lookup, as `get_label_id` does over the dictionary. -/
def idxOfF (d : List String) (n : String) : Option Nat := d.findIdx? (· == n)

/-- `get_label_id_mut` / `get_type_id_mut` / `add_node_attribute_name`. -/
def internF (d : List String) (n : String) : Nat × List String :=
  match idxOfF d n with
  | some i => (i, d)
  | none => (d.length, d ++ [n])

/-- The attribute dictionary's capacity (`AttrNameMap::insert` refuses a new name past it: 65,535). -/
def attrCap : Nat := 65535

/-- `add_node_attribute_name` with its capacity: a name already there keeps its id; a new one is refused
(`none`, `AttributeLimitReached`) once the dictionary is full. `internF` is this without the cap, which is
what labels and types do. The abstract `applyRec` uses `internCap` for attributes (`internCap_refines`). -/
def internCapF (cap : Nat) (d : List String) (n : String) : Option (Nat × List String) :=
  match idxOfF d n with
  | some i => some (i, d)
  | none => if d.length < cap then some (d.length, d ++ [n]) else none

/-- Below capacity (or for a known name) the capped intern is the uncapped one. -/
theorem internCapF_of_room (cap : Nat) (d : List String) (n : String) (h : d.length < cap ∨ idxOfF d n ≠ none) :
    internCapF cap d n = some (internF d n) := by
  unfold internCapF internF
  cases hi : idxOfF d n with
  | some i => simp
  | none =>
    rcases h with h | h
    · simp [h]
    · exact absurd hi h

/-- A full dictionary refuses a new name. -/
theorem internCapF_full (cap : Nat) (d : List String) (n : String) (hf : ¬ d.length < cap) (hn : idxOfF d n = none) :
    internCapF cap d n = none := by
  simp [internCapF, hn, hf]

/-- `verify_id` (`apply.rs:586`), success only. -/
def verifyIdF (expected assigned : Nat) : Bool := assigned == expected

/-- `apply_add_schema` (`apply.rs:560`): intern, then `verify_id`. -/
def applyAddSchemaF (d : List String) (id : Nat) (n : String) : Option (List String) :=
  let (assigned, d') := internF d n
  if verifyIdF id assigned then some d' else none

/-- The attribute arm of `apply_record` (`apply.rs:175`): `add_node_attribute_name` (capped), read the id
back, `AttributeLimitReached` (`none`) if the name is not registered, then `verify_id`. -/
def applyAddAttributeF (cap : Nat) (d : List String) (id : Nat) (n : String) : Option (List String) :=
  match internCapF cap d n with
  | none => none
  | some (assigned, d') => if verifyIdF id assigned then some d' else none

/-- `IdSpace::reserve` (`id_space.rs:516`) on a batch opened at `bound` that has
taken and been issued nothing: `reclaim_ids` (`:244`) takes the lowest `count`
ids of `recycled - ∅ - ∅`, then fresh ids from `entry_bound + above(∅) +
above(∅) = bound + 0`. `bin` is the bin's ascending listing. -/
def reserveFresh (bound : Nat) (bin : List Nat) (count : Nat) : List Nat :=
  let reclaimed := bin.take count
  reclaimed ++ List.range' (bound + 0) (count - reclaimed.length)

/-- `IdSpace::reserve` (`id_space.rs:516`): `reclaim_ids` also skips the
`quarantined` ids — what earlier batches of this version freed or cancelled (`open_batch`, `:470`). The
fresh ids above the boundary are unchanged. -/
def reserveQ (bound : Nat) (bin quarantined : List Nat) (count : Nat) : List Nat :=
  let reclaimed := (bin.filter (fun i => !quarantined.contains i)).take count
  reclaimed ++ List.range' (bound + 0) (count - reclaimed.length)

/-- On the first batch of a version `quarantined` is empty (`IdSpace::new_version`, `restored`, `new`
all start it empty), and then `reserve` is `reserveFresh`: the replication model's `idsOf`
(`idsOf_refines`) is exactly that case. A later batch only *removes* candidates from the bin, so what it
issues is still a free or fresh id (`reserveQ_mem`), which is all the replica's `okCreate` asks. -/
theorem reserveQ_nil (bound : Nat) (bin : List Nat) (count : Nat) :
    reserveQ bound bin [] count = reserveFresh bound bin count := by
  have : bin.filter (fun i => !([] : List Nat).contains i) = bin := by
    simp
  simp only [reserveQ, reserveFresh, this]

theorem reserveQ_mem (bound : Nat) (bin q : List Nat) (count i : Nat)
    (h : i ∈ reserveQ bound bin q count) : i ∈ bin ∨ bound ≤ i := by
  simp only [reserveQ, List.mem_append, List.mem_range'_1] at h
  rcases h with h | h
  · left
    exact (List.mem_filter.1 (List.mem_of_mem_take h)).1
  · right; omega

/-- The half of the node `IdSpace` (`id_space.rs:164`) that `Graph` reads:
`live` and `recycled` (a set, here a duplicate-free list). -/
structure Ids where
  live : Nat
  bin  : List Nat

/-- `IdSpace::bound` (`:359`) = `Graph::node_id_bound` (`graph.rs:1709`). -/
def Ids.bound (s : Ids) : Nat := s.live + s.bin.length

/-- The state `IdSpace::create` (`:621`) moves to on `{id}` once its refusals
pass: `recycled -= nodes; live += nodes.len()` — what `Graph::create_nodes`
(`graph.rs:1748`) does to the id space. -/
def Ids.create1 (s : Ids) (id : Nat) : Ids := ⟨s.live + 1, s.bin.erase id⟩

/-- Insert `id` into an ascending list at its sorted place. -/
def insAsc (id : Nat) : List Nat → List Nat
  | [] => [id]
  | b :: bs => if id ≤ b then id :: b :: bs else b :: insAsc id bs

theorem insAsc_perm (id : Nat) : ∀ l : List Nat, (insAsc id l).Perm (id :: l)
  | [] => List.Perm.refl _
  | b :: bs => by
    unfold insAsc
    split
    · exact List.Perm.refl _
    · exact ((insAsc_perm id bs).cons b).trans (List.Perm.swap id b bs)

theorem insAsc_mem (id x : Nat) (l : List Nat) : x ∈ insAsc id l ↔ x = id ∨ x ∈ l := by
  rw [(insAsc_perm id l).mem_iff]; simp

theorem insAsc_length (id : Nat) (l : List Nat) : (insAsc id l).length = l.length + 1 := by
  rw [(insAsc_perm id l).length_eq]; simp

theorem insAsc_nodup (id : Nat) (l : List Nat) (h : id ∉ l) : (insAsc id l).Nodup ↔ l.Nodup := by
  rw [(insAsc_perm id l).nodup_iff]; simp [h]

theorem insAsc_sorted (id : Nat) : ∀ l : List Nat, l.Pairwise (· ≤ ·) → (insAsc id l).Pairwise (· ≤ ·)
  | [], _ => by simp [insAsc]
  | b :: bs, h => by
    unfold insAsc
    split
    · rename_i hle
      refine List.Pairwise.cons ?_ h
      intro x hx
      rcases List.mem_cons.mp hx with rfl | hx
      · exact hle
      · exact Nat.le_trans hle (List.rel_of_pairwise_cons h hx)
    · rename_i hlt
      have h' := List.pairwise_cons.mp h
      refine List.Pairwise.cons ?_ (insAsc_sorted id bs h'.2)
      intro x hx
      have := (insAsc_perm id bs).mem_iff.mp hx
      rcases List.mem_cons.mp this with rfl | hx
      · omega
      · exact h'.1 x hx

/-- The state `IdSpace::release` (`:739`) moves to on `requested = freed = {id}`
once its refusals pass (`id` not free): `recycled |= freed; live -= freed.len()`
— `Graph::delete_nodes` (`graph.rs:2310`). The id goes in at its sorted place, so the bin stays an
ascending listing (`release1_sorted`), which `reserveFresh` (`bin.take`) relies on. -/
def Ids.release1 (s : Ids) (id : Nat) : Ids := ⟨s.live - 1, insAsc id s.bin⟩

theorem Ids.release1_sorted (s : Ids) (id : Nat) (h : s.bin.Pairwise (· ≤ ·)) :
    (s.release1 id).bin.Pairwise (· ≤ ·) := by
  simp only [Ids.release1]
  exact insAsc_sorted id s.bin h

/-- **`IdSpace::release`'s `self.live -= freed.len()` cannot underflow on a live id** (the u64
subtraction the model does in `Nat`): in an id space whose bin is a duplicate-free set of ids below the
boundary, a live id (`id < bound`, not in the bin) forces `1 ≤ live`, because `bound = live + |bin|` and
the bin misses `id`, so `|bin| ≤ bound - 1`. -/
theorem Ids.live_pos (s : Ids) (hn : s.bin.Nodup) (hlt : ∀ b ∈ s.bin, b < s.bound)
    (id : Nat) (hid : id < s.bound) (hnf : id ∉ s.bin) : 1 ≤ s.live := by
  by_cases h : 1 ≤ s.live
  · exact h
  · exfalso
    have hl : s.live = 0 := by omega
    have hsub : s.bin ⊆ (List.range s.bound).erase id := by
      intro b hb
      have hbl := hlt b hb
      have hne : b ≠ id := fun e => hnf (e ▸ hb)
      exact (List.mem_erase_of_ne hne).2 (List.mem_range.2 hbl)
    have hle := hn.length_le_of_subset hsub
    have hmem : id ∈ List.range s.bound := List.mem_range.2 hid
    rw [List.length_erase_of_mem hmem, List.length_range] at hle
    have hb : s.bound = s.bin.length := by simp [Ids.bound, hl]
    omega

end FalkorFaithful
