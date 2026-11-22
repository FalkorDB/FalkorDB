import PendingCommit.PendingState
/-!
# Pending's staged attribute maps: disjoint, and each list sorted with unique keys

`Pending` keeps node attributes in two maps, `new_nodes_attrs` (ids this query created) and
`existing_nodes_attrs` (everything else), and the same for relationships (pending.rs nodes,
:979-1066 relationships). Readers (`Runtime::get_node_attribute`, `get_node_attrs`,
`get_relationship_attrs`, proofs/runtime_core `Accessors`/`RelAccessors`) rest on two facts:

* `hdisj` — an entity id is in at most one of the two maps;
* `huniq` — the keys of every staged list are unique (it is sorted by id: `binary_search_by_key`).

The pair `(nw, ex)` below is `(new_*_attrs, existing_*_attrs)` and `created` is the id set whose
membership routes a write to `nw` (`created_nodes` / `created_rel_types`); the staging API is the
model's `stageOne` / `stageAll` / `clearA` (PendingState), with `isNew = (id ∈ created)` as the Rust
code computes it.

`PairInv created nw ex` is the invariant: every list sorted (so unique keys), the maps disjoint,
and no created id has a staged *existing* entry (`fresh`). It holds for the empty pending state
(`pairInv_empty`) and is kept by every operation that touches the maps:

| Rust | here |
| --- | --- |
| `set_node_attribute` / `set_relationship_attribute` | `stageOne_inv` (via `rustUpsert_eq`, `Props.upsert_sorted`) |
| `set_node_attributes` / `set_relationship_attributes` | `stageAll_inv`: **premise** `Sorted attrs`: the doc comment's "sorted by id and unique", checked only by `debug_assert!`. The only callers are `CreateOp`'s template (strictly ascending by construction: ops_apply `CreateAttrs.tplBuild_nodup`, `rank_strict`) and the test-only effects staging; `CREATE (n $map)` is refused by the parser, so the `resolve_map_attrs` path, whose sort would not de-duplicate colliding `ATTRIBUTE_ID_NONE`s, is not reachable (live) |
| `clear_node_attributes`, `clear_relationship_attributes`, `remove_pending_relationships_for_node`'s `remove` | `clearA_inv` |
| `created_nodes`, `created_relationship` | `created_inv`: **premise** `ex id = none` — the id was reserved by the id space, so it is not a live entity and has no staged existing attributes (`Snapshot.createRust_Wf`: a reserved id is not live; staging for an existing entity only happens for a live one, `set.rs` skips deleted ones) |
| `delete_pending_node` | `unwind_inv` (`created.remove(id)`, then the entry of whichever map held it) |

`inv_hdisj` / `inv_huniq` are the two facts in exactly the shape runtime_core's accessor theorems
take them as hypotheses.
-/
namespace PendingCommit.PA
open PendingCommit

/-- The staged-attribute invariant of one `(new, existing)` pair of maps. -/
structure PairInv (created : List Nat) (nw ex : FMap (List Entry)) : Prop where
  sortedN : ∀ id l, fget nw id = some l → Sorted l
  sortedE : ∀ id l, fget ex id = some l → Sorted l
  disj : ∀ id, fget nw id = none ∨ fget ex id = none
  fresh : ∀ id ∈ created, fget ex id = none
  /-- only created ids have a `new` entry -/
  inNew : ∀ id l, fget nw id = some l → id ∈ created

theorem pairInv_empty : PairInv [] [] [] :=
  ⟨fun _ _ h => by simp [fget] at h, fun _ _ h => by simp [fget] at h, fun _ => Or.inl rfl,
   fun _ h => by simp at h, fun _ _ h => by simp [fget] at h⟩

/-- **`hdisj`**: an id is in at most one of the two maps. -/
theorem inv_hdisj {c : List Nat} {nw ex : FMap (List Entry)} (h : PairInv c nw ex) (id : Nat) :
    fget nw id = none ∨ fget ex id = none := h.disj id

/-- **`huniq`**: the keys of every staged list are unique. -/
theorem inv_huniq {c : List Nat} {nw ex : FMap (List Entry)} (h : PairInv c nw ex) (id : Nat)
    (l : List Entry) (hl : fget nw id = some l ∨ fget ex id = some l) : (l.map Prod.fst).Nodup := by
  have hs : Sorted l := hl.elim (h.sortedN id l) (h.sortedE id l)
  have := Content.sorted_keys l hs
  exact List.Pairwise.imp (fun h => Nat.ne_of_lt h) this

theorem sorted_getD (m : FMap (List Entry)) (id : Nat) (hm : ∀ l, fget m id = some l → Sorted l) :
    Sorted ((fget m id).getD []) := by
  cases h : fget m id with
  | none => simp [Sorted]
  | some l => simpa using hm l h

/-- `set_*_attribute`: validate, upsert into the map the id's `created` membership picks. -/
theorem stageOne_inv (valid : Val → Bool) (created : List Nat) (nw ex : FMap (List Entry))
    (h : PairInv created nw ex) (id k : Nat) (v : Val) (nw' ex' : FMap (List Entry))
    (hs : PS.stageOne valid (decide (id ∈ created)) nw ex id k v = some (nw', ex')) :
    PairInv created nw' ex' := by
  unfold PS.stageOne at hs
  split at hs
  · cases hs
  · split at hs
    · rename_i hv hc
      simp only [Option.some.injEq, Prod.mk.injEq] at hs
      obtain ⟨rfl, rfl⟩ := hs
      have hcid : id ∈ created := by simpa using hc
      have hup : Sorted (PA.rustUpsert ((fget nw id).getD []) k v) := by
        rw [PA.rustUpsert_eq _ (sorted_getD nw id (h.sortedN id)) k v]
        exact upsert_sorted _ k v (sorted_getD nw id (h.sortedN id))
      refine ⟨?_, h.sortedE, ?_, h.fresh, ?_⟩
      · intro j l hj
        rw [fget_fset] at hj
        by_cases hji : j = id
        · simp [hji] at hj; subst hj; exact hup
        · simp [hji] at hj; exact h.sortedN j l hj
      · intro j
        rw [fget_fset]
        by_cases hji : j = id
        · right; subst hji; exact h.fresh j hcid
        · simp only [hji, ite_false]; exact h.disj j
      · intro j l hj
        rw [fget_fset] at hj
        by_cases hji : j = id
        · subst hji; exact hcid
        · simp [hji] at hj; exact h.inNew j l hj
    · rename_i hv hc
      simp only [Option.some.injEq, Prod.mk.injEq] at hs
      obtain ⟨rfl, rfl⟩ := hs
      have hcid : id ∉ created := by simpa using hc
      have hup : Sorted (PA.rustUpsert ((fget ex id).getD []) k v) := by
        rw [PA.rustUpsert_eq _ (sorted_getD ex id (h.sortedE id)) k v]
        exact upsert_sorted _ k v (sorted_getD ex id (h.sortedE id))
      refine ⟨h.sortedN, ?_, ?_, ?_, h.inNew⟩
      · intro j l hj
        rw [fget_fset] at hj
        by_cases hji : j = id
        · simp [hji] at hj; subst hj; exact hup
        · simp [hji] at hj; exact h.sortedE j l hj
      · intro j
        rw [fget_fset]
        by_cases hji : j = id
        · left; subst hji
          cases hn : fget nw j with
          | none => rfl
          | some l => exact absurd (h.inNew j l hn) hcid
        · simp only [hji, ite_false]; exact h.disj j
      · intro j hj
        rw [fget_fset]
        have : j ≠ id := fun e => hcid (e ▸ hj)
        simp only [this, ite_false]; exact h.fresh j hj

/-- `set_*_attributes`: validate all, skip empty, replace the entry. **Premise** `Sorted attrs`
(see the header). -/
theorem stageAll_inv (valid : Val → Bool) (created : List Nat) (nw ex : FMap (List Entry))
    (h : PairInv created nw ex) (id : Nat) (attrs : List Entry) (hsorted : Sorted attrs)
    (nw' ex' : FMap (List Entry))
    (hs : PS.stageAll valid (decide (id ∈ created)) nw ex id attrs = some (nw', ex')) :
    PairInv created nw' ex' := by
  unfold PS.stageAll at hs
  split at hs
  · cases hs
  · split at hs
    · simp only [Option.some.injEq, Prod.mk.injEq] at hs
      obtain ⟨rfl, rfl⟩ := hs; exact h
    · split at hs
      · rename_i hv he hc
        simp only [Option.some.injEq, Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl⟩ := hs
        have hcid : id ∈ created := by simpa using hc
        refine ⟨?_, h.sortedE, ?_, h.fresh, ?_⟩
        · intro j l hj
          rw [fget_fset] at hj
          by_cases hji : j = id
          · simp [hji] at hj; subst hj; exact hsorted
          · simp [hji] at hj; exact h.sortedN j l hj
        · intro j
          rw [fget_fset]
          by_cases hji : j = id
          · right; subst hji; exact h.fresh j hcid
          · simp only [hji, ite_false]; exact h.disj j
        · intro j l hj
          rw [fget_fset] at hj
          by_cases hji : j = id
          · subst hji; exact hcid
          · simp [hji] at hj; exact h.inNew j l hj
      · rename_i hv he hc
        simp only [Option.some.injEq, Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl⟩ := hs
        have hcid : id ∉ created := by simpa using hc
        refine ⟨h.sortedN, ?_, ?_, ?_, h.inNew⟩
        · intro j l hj
          rw [fget_fset] at hj
          by_cases hji : j = id
          · simp [hji] at hj; subst hj; exact hsorted
          · simp [hji] at hj; exact h.sortedE j l hj
        · intro j
          rw [fget_fset]
          by_cases hji : j = id
          · left; subst hji
            cases hn : fget nw j with
            | none => rfl
            | some l => exact absurd (h.inNew j l hn) hcid
          · simp only [hji, ite_false]; exact h.disj j
        · intro j hj
          rw [fget_fset]
          have : j ≠ id := fun e => hcid (e ▸ hj)
          simp only [this, ite_false]; exact h.fresh j hj

/-- `clear_node_attributes` / `clear_relationship_attributes`: remove the id from both maps. -/
theorem clearA_inv (created : List Nat) (nw ex : FMap (List Entry)) (h : PairInv created nw ex)
    (id : Nat) : PairInv created (PS.clearA nw ex id).1 (PS.clearA nw ex id).2 := by
  simp only [PS.clearA]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro j l hj; rw [fget_frem] at hj; split at hj
    · cases hj
    · exact h.sortedN j l hj
  · intro j l hj; rw [fget_frem] at hj; split at hj
    · cases hj
    · exact h.sortedE j l hj
  · intro j; simp only [fget_frem]; split
    · exact Or.inl rfl
    · exact h.disj j
  · intro j hj; simp only [fget_frem]; split
    · rfl
    · exact h.fresh j hj
  · intro j l hj; rw [fget_frem] at hj; split at hj
    · cases hj
    · exact h.inNew j l hj

/-- `created_nodes` / `created_relationship`: an id joins `created`. **Premise** `ex id = none`
(see the header: a reserved id is no live entity, so it has no staged existing attributes). -/
theorem created_inv (created : List Nat) (nw ex : FMap (List Entry)) (h : PairInv created nw ex)
    (id : Nat) (hex : fget ex id = none) : PairInv (PS.sins created id) nw ex := by
  refine ⟨h.sortedN, h.sortedE, h.disj, ?_, ?_⟩
  · intro j hj
    rw [PS.mem_sins] at hj
    rcases hj with rfl | hj
    · exact hex
    · exact h.fresh j hj
  · intro j l hj
    rw [PS.mem_sins]; exact Or.inr (h.inNew j l hj)

/-- `delete_pending_node` on a created node: `created.remove(id)`, then the entry of the map that
held it (`new`, else `existing`) is dropped. -/
theorem unwind_inv (created : List Nat) (nw ex : FMap (List Entry)) (h : PairInv created nw ex)
    (id : Nat) :
    PairInv (created.filter (· != id))
      (match fget nw id with | some _ => frem nw id | none => nw)
      (match fget nw id with | some _ => ex | none => frem ex id) := by
  cases hn : fget nw id with
  | some l0 =>
    simp only
    refine ⟨?_, h.sortedE, ?_, ?_, ?_⟩
    · intro j l hj; rw [fget_frem] at hj; split at hj
      · cases hj
      · exact h.sortedN j l hj
    · intro j; simp only [fget_frem]; split
      · exact Or.inl rfl
      · exact h.disj j
    · intro j hj
      simp only [List.mem_filter] at hj
      exact h.fresh j hj.1
    · intro j l hj; rw [fget_frem] at hj; split at hj
      · cases hj
      · simp only [List.mem_filter, bne_iff_ne, ne_eq]
        rename_i hne
        exact ⟨h.inNew j l hj, hne⟩
  | none =>
    simp only
    refine ⟨h.sortedN, ?_, ?_, ?_, ?_⟩
    · intro j l hj; rw [fget_frem] at hj; split at hj
      · cases hj
      · exact h.sortedE j l hj
    · intro j; simp only [fget_frem]; split
      · exact Or.inr rfl
      · exact h.disj j
    · intro j hj
      simp only [List.mem_filter] at hj
      rw [fget_frem]
      have : j ≠ id := by simpa using hj.2
      simp only [this, ite_false]; exact h.fresh j hj.1
    · intro j l hj
      simp only [List.mem_filter, bne_iff_ne, ne_eq]
      refine ⟨h.inNew j l hj, ?_⟩
      rintro rfl; rw [hn] at hj; cases hj

end PendingCommit.PA
