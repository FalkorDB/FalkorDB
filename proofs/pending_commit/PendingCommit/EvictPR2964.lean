import PendingCommit.Snapshot
/-
# MODELS OPEN PR #2964 (not merged; `main` behaves as `Snapshot.createRust`)

PR #2964 ("evict on reissue") fixes #2876 / #2759 differently from the
`createDeferred` proposal in `Snapshot.lean`: `CREATE` keeps the engine's
allocator (recycle bin first) and then drops the deleted-entity snapshot of the
id it just reissued (`forget_deleted_nodes` / `_relationships`, called from
`ops/create.rs` in the PR). Ids stay the same as C's; a variable still bound to
the deleted entity then reads the entity that holds the id, which is also C's
answer.

* `evict_preserves_Good` — eviction keeps the `Good` invariant (no snapshotted
  id is live), so with `Snapshot.sound_under_Good` every snapshot-first read
  stays correct; unlike `deferred_preserves_Good` no freshness side condition
  is needed.
* `evict_reads_live_entity` — on the #2876 trace the new entity reuses id 0
  (as on `main` and in C) and reads itself, not the deleted node.

No other module depends on this file; revisit it only if the PR changes.
-/
namespace PendingCommit.Snapshot

/-- The PR's allocation: allocate as today (bin first), then drop the snapshot
of the id just reissued. -/
def createEvict (s : RS) (d : Nat) : RS × Nat :=
  let (s', id) := createRust s d
  ({ s' with snap := s'.snap.filter (·.1 != id) }, id)

theorem createRust_live (s : RS) (d : Nat) :
    ∃ x, (createRust s d).1.live = ((createRust s d).2, x) :: s.live := by
  unfold createRust; split <;> exact ⟨_, rfl⟩

theorem createRust_snap (s : RS) (d : Nat) : (createRust s d).1.snap = s.snap := by
  unfold createRust; split <;> rfl

/-- **`evict_preserves_Good`**: evicting on reissue keeps every read sound. -/
theorem evict_preserves_Good (s : RS) (h : Good s) (d : Nat) :
    Good (createEvict s d).1 := by
  intro j hj
  obtain ⟨x, hx⟩ := createRust_live s d
  simp only [createEvict, keys, List.mem_map, List.mem_filter] at hj ⊢
  obtain ⟨⟨k, e⟩, ⟨hmem, hne⟩, rfl⟩ := hj
  rw [createRust_snap] at hmem
  simp only [bne_iff_ne, ne_eq] at hne
  rw [hx, find_cons_ne _ _ _ _ hne]
  exact h k (by simp [keys]; exact ⟨e, hmem⟩)

/-- The #2876 trace with eviction: `m` reuses id 0 and reads itself. -/
theorem evict_reads_live_entity :
    let r := (let (a, _) := createRust s0 1
              let a := commit a
              let a := commit (del a 0)
              createEvict a 2)
    r.2 = 0 ∧ readRt r.1 r.2 = some ⟨1, 2⟩ := by
  decide

end PendingCommit.Snapshot
