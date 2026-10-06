/-
# Deleted-entity snapshots vs. id reuse (#2876 and its other shapes)

`Runtime::deleted_nodes` / `deleted_relationships` (runtime.rs:159) are
query-lifetime maps **keyed by id** holding what a deleted entity looked like,
so a later clause can still read it (`DELETE n WITH n RETURN n.v`). Every
entity read consults them *first*:

| accessor | runtime.rs |
| --- | --- |
| `get_node_attribute`                 | :1459 |
| `materialize_node_property_values`   | :1569 (the vectorised `n.v` path, used by `WHERE`) |
| `get_node_labels`, `node_has_label_id` | :1660, :1728 |
| `get_node_attrs` (`properties`, `keys`, `SET x = n`) | :1739 |
| `get_relationship_attribute` / `_attrs` | :1510, :1757 |
| `get_relationship_endpoints`, `get_relationship_type` | :1775, :1788 |

A `Value::Node`/`Value::Relationship` carries only the id, and ids freed by a
committed delete are reissued by the next segment's CREATE (`IdSpace::reserve`,
recycle bin first). Model: an entity is `(gen, data)` — `data` stands for any
field one of the accessors above returns — and a handle is `(id, gen)`.

* `readRt` — the accessors' shape: snapshot map first, then the graph.
* `Good` — no snapshotted id is live.
* `sound_under_Good`: under `Good`, `readRt` returns the right entity for every
  handle the query holds (live or deleted-by-this-query).
* `Good` is preserved by delete, commit and a create that does *not* reissue a
  snapshotted id (`createDeferred`, the fix proposed in #2876) —
  `deferred_preserves_Good`.
* With the engine's allocator (`createRust`, bin first) `Good` breaks and the
  read is wrong: `rust_reads_dead_entity` is the Lean counterexample, the same
  trace as `graph/tests/lean_pending_commit.rs` and the Cypher repros.

The model says nothing specific to nodes, attributes or labels: the bug is the
id-keyed snapshot shadowing a live id, so every accessor above and both entity
kinds have it (confirmed live for node attrs, labels, `WHERE`, `properties`,
MERGE, FOREACH, relationship attrs, type and endpoints).
-/
namespace PendingCommit.Snapshot

structure Ent where
  gen : Nat
  data : Nat
  deriving DecidableEq, Repr

def find : List (Nat × Ent) → Nat → Option Ent
  | [], _ => none
  | p :: ps, id => if p.1 = id then some p.2 else find ps id

def keys (l : List (Nat × Ent)) : List Nat := l.map (·.1)

structure RS where
  live : List (Nat × Ent)
  /-- ids freed by this segment's deletes, not reusable until the segment commits -/
  freed : List Nat
  bin : List Nat
  fresh : Nat
  snap : List (Nat × Ent)
  nextGen : Nat
  deriving Repr

/-- `DELETE` of a live node: snapshot it (delete.rs:259-266), drop it from the graph. -/
def del (s : RS) (id : Nat) : RS :=
  match find s.live id with
  | some e => { s with live := s.live.filter (·.1 != id), freed := id :: s.freed,
                       snap := (id, e) :: s.snap }
  | none => s

/-- A `Commit` boundary: freed ids go to the recycle bin. The snapshot map is
*not* cleared (it is query-lifetime). -/
def commit (s : RS) : RS := { s with bin := s.freed ++ s.bin, freed := [] }

/-- `CREATE` with the engine's allocator: recycle bin first, else fresh. -/
def createRust (s : RS) (d : Nat) : RS × Nat :=
  match s.bin with
  | id :: rest => ({ s with live := (id, ⟨s.nextGen, d⟩) :: s.live, bin := rest,
                             nextGen := s.nextGen + 1 }, id)
  | [] => ({ s with live := (s.fresh, ⟨s.nextGen, d⟩) :: s.live, fresh := s.fresh + 1,
                     nextGen := s.nextGen + 1 }, s.fresh)

/-- The fix in #2876: never reissue an id this query snapshotted; take a fresh one. -/
def createDeferred (s : RS) (d : Nat) : RS × Nat :=
  match s.bin.filter (fun i => !(keys s.snap).contains i) with
  | id :: _ => ({ s with live := (id, ⟨s.nextGen, d⟩) :: s.live, bin := s.bin.erase id,
                          nextGen := s.nextGen + 1 }, id)
  | [] => ({ s with live := (s.fresh, ⟨s.nextGen, d⟩) :: s.live, fresh := s.fresh + 1,
                     nextGen := s.nextGen + 1 }, s.fresh)

/-- Every snapshot-first accessor. -/
def readRt (s : RS) (id : Nat) : Option Ent :=
  match find s.snap id with
  | some e => some e
  | none => find s.live id

def Good (s : RS) : Prop := ∀ id ∈ keys s.snap, find s.live id = none

theorem find_none_iff (l : List (Nat × Ent)) (id : Nat) : find l id = none ↔ id ∉ keys l := by
  induction l with
  | nil => simp [find, keys]
  | cons p ps ih =>
    simp only [find, keys, List.map_cons, List.mem_cons] at ih ⊢
    by_cases hp : p.1 = id
    · simp [hp]
    · simp only [hp, ite_false]; rw [ih]; constructor
      · intro h1 h2; rcases h2 with h2 | h2; exact hp h2.symm; exact h1 h2
      · intro h1 h2; exact h1 (.inr h2)

/-- **`sound_under_Good`**: with no snapshotted id live, a handle to a live
entity reads that entity and a handle to one deleted by this query reads its
snapshot. -/
theorem sound_under_Good (s : RS) (h : Good s) (id : Nat) (e : Ent) :
    (find s.live id = some e → readRt s id = some e) ∧
    (find s.snap id = some e → readRt s id = some e) := by
  constructor
  · intro hl
    have : find s.snap id = none := by
      cases hs : find s.snap id with
      | none => rfl
      | some e' =>
        have hm : id ∈ keys s.snap := Classical.byContradiction fun hn => by
          rw [(find_none_iff _ _).2 hn] at hs; cases hs
        rw [h id hm] at hl; cases hl
    unfold readRt; rw [this]; exact hl
  · intro hs; unfold readRt; rw [hs]

theorem find_filter_ne (l : List (Nat × Ent)) (id j : Nat) (h : j ≠ id) :
    find (l.filter (·.1 != id)) j = find l j := by
  induction l with
  | nil => rfl
  | cons p ps ih =>
    by_cases hp : p.1 = id
    · rw [List.filter_cons_of_neg (by simp [hp])]
      simp only [find]; rw [if_neg (by omega)]; exact ih
    · rw [List.filter_cons_of_pos (by simp [hp])]
      simp only [find]
      by_cases hj : p.1 = j
      · rw [if_pos hj, if_pos hj]
      · rw [if_neg hj, if_neg hj, ih]

theorem find_filter_eq (l : List (Nat × Ent)) (id : Nat) : find (l.filter (·.1 != id)) id = none := by
  rw [find_none_iff]; simp [keys]

theorem del_Good (s : RS) (h : Good s) (id : Nat) : Good (del s id) := by
  unfold del
  cases hf : find s.live id with
  | none => simpa [hf] using h
  | some e =>
    simp only [hf]
    intro j hj
    simp [keys] at hj
    by_cases hji : j = id
    · subst hji; exact find_filter_eq _ _
    · rw [find_filter_ne _ _ _ hji]
      rcases hj with hj | ⟨e', he'⟩
      · exact absurd hj hji
      · exact h j (by simp [keys]; exact ⟨e', he'⟩)

theorem commit_Good (s : RS) (h : Good s) : Good (commit s) := h

/-- The generation-fresh boundary: no live or snapshotted id at or above `fresh`. -/
def FreshOK (s : RS) : Prop := ∀ id ∈ keys s.snap, id < s.fresh

theorem find_cons_ne (l : List (Nat × Ent)) (a : Nat) (x : Ent) (j : Nat) (h : j ≠ a) :
    find ((a, x) :: l) j = find l j := by
  simp only [find]; rw [if_neg (fun h' => h h'.symm)]

/-- **`deferred_preserves_Good`**: allocating around the snapshot keeps every
read sound. -/
theorem deferred_preserves_Good (s : RS) (h : Good s) (hf : FreshOK s) (d : Nat) :
    Good (createDeferred s d).1 := by
  unfold createDeferred
  split
  · rename_i id rest hb
    have hmem : id ∈ s.bin.filter (fun i => !(keys s.snap).contains i) := by rw [hb]; simp
    simp [List.mem_filter] at hmem
    intro j hj
    simp only at hj ⊢
    have hji : j ≠ id := by intro heq; subst heq; exact hmem.2 hj
    rw [find_cons_ne _ _ _ _ hji]; exact h j hj
  · intro j hj
    simp only at hj ⊢
    have := hf j hj
    rw [find_cons_ne _ _ _ _ (by omega)]; exact h j hj

def s0 : RS := ⟨[], [], [], 0, [], 0⟩

/-- `CREATE (n {v:1})` ; `MATCH (n) DELETE n WITH count(*) AS c CREATE (m {v:2}) …`:
the second query's CREATE reuses id 0 after the Commit, and every read of `m`
returns the deleted node. -/
def rustTrace : RS × Nat :=
  let (a, _) := createRust s0 1
  let a := commit a
  let a := commit (del a 0)
  createRust a 2

/-- **`rust_reads_dead_entity`**: `m` got id 0; the graph holds `m` (data 2),
but the read returns the deleted node (data 1). -/
theorem rust_reads_dead_entity :
    rustTrace.2 = 0 ∧ find rustTrace.1.live 0 = some ⟨1, 2⟩ ∧ readRt rustTrace.1 0 = some ⟨0, 1⟩ ∧
      ¬ Good rustTrace.1 := by
  refine ⟨by decide, by decide, by decide, ?_⟩
  intro h
  have := h 0 (by decide)
  revert this; decide

/-- The same trace with the deferred allocator reads `m` correctly. -/
theorem deferred_reads_live_entity :
    let r := (let (a, _) := createRust s0 1
              let a := commit a
              let a := commit (del a 0)
              createDeferred a 2)
    r.2 = 1 ∧ readRt r.1 r.2 = some ⟨1, 2⟩ ∧ readRt r.1 0 = some ⟨0, 1⟩ := by
  decide

end PendingCommit.Snapshot
