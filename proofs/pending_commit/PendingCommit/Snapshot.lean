/-
# Deleted-entity snapshots vs. id reuse (#2876 and its other shapes)

`Runtime::deleted_nodes` / `deleted_relationships` (runtime.rs:193/:195) are
query-lifetime maps **keyed by id** holding what a deleted entity looked like,
so a later clause can still read it (`DELETE n WITH n RETURN n.v`). Every
entity read consults them *first*:

| accessor | runtime.rs |
| --- | --- |
| `get_node_attribute`                 | :1571 |
| `materialize_node_property_values`   | :1668 (the vectorised `n.v` path, used by `WHERE`) |
| `get_node_labels`, `node_has_label_id` | :1787, :1852 |
| `get_node_attrs` (`properties`, `keys`, `SET x = n`) | :1866 |
| `get_relationship_attribute` / `_attrs` | :1622, :1884 |
| `get_relationship_endpoints`, `get_relationship_type` | :1904, :1917 |

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

/-- `DELETE` of a live node: snapshot it (`delete_nodes_bulk`, delete.rs:301-312; `delete_entity`, :485-491), drop it from the graph. -/
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

/-! ## Relationships since #3051: reissued ids bypass the snapshot

`reissued_relationships` (runtime.rs:210) holds the snapshotted ids a later `CREATE`
reissued: `note_relationship_created` (runtime.rs:1714, called per created edge by
`create_batch`, create.rs:293) marks an id that has a snapshot, `record_deleted_relationship`
(runtime.rs:1703, every relationship snapshot in delete.rs) unmarks it, and every
relationship read (`get_relationship_{attribute,attrs,endpoints,type}`,
`materialize_relationship_property_values`) skips the snapshot of a marked id
(`is_reissued_relationship`, runtime.rs:1694). Nodes keep `readRt`. -/

structure RR where
  s : RS
  re : List Nat
  deriving Repr

/-- `DELETE` of a live relationship: `record_deleted_relationship` (snapshot, unmark). -/
def delR (r : RR) (id : Nat) : RR :=
  match find r.s.live id with
  | some _ => { s := del r.s id, re := r.re.filter (· != id) }
  | none => r

def commitR (r : RR) : RR := { r with s := commit r.s }

/-- `CREATE` (engine allocator) + `note_relationship_created`. -/
def createR (r : RR) (d : Nat) : RR × Nat :=
  let (s', id) := createRust r.s d
  ({ s := s', re := if (keys r.s.snap).contains id then id :: r.re else r.re }, id)

/-- Every relationship read: the snapshot unless the id is reissued, then the graph. -/
def readR (r : RR) (id : Nat) : Option Ent :=
  if r.re.contains id then find r.s.live id else readRt r.s id

/-- Every live snapshotted id is marked reissued. -/
def ReOk (r : RR) : Prop := ∀ id, id ∈ keys r.s.snap → (find r.s.live id).isSome → id ∈ r.re

theorem find_some_mem (l : List (Nat × Ent)) (id : Nat) (e : Ent) (h : find l id = some e) :
    id ∈ keys l := Classical.byContradiction fun hn => by
  rw [(find_none_iff _ _).2 hn] at h; cases h

/-- **`readR_live`**: under `ReOk`, a handle to a live relationship reads it (the #2876
shape `rust_reads_dead_entity` is gone for relationships). -/
theorem readR_live (r : RR) (h : ReOk r) (id : Nat) (e : Ent) (hl : find r.s.live id = some e) :
    readR r id = some e := by
  unfold readR
  by_cases hc : r.re.contains id = true
  · simp only [hc, ite_true]; exact hl
  · simp only [hc, Bool.false_eq_true, ite_false]
    have hs : find r.s.snap id = none := by
      cases hs : find r.s.snap id with
      | none => rfl
      | some e' =>
        exfalso; apply hc
        have := h id (find_some_mem _ _ _ hs) (by rw [hl]; rfl)
        simpa using this
    unfold readRt; rw [hs]; exact hl

/-- **`readR_deleted`**: a deleted relationship whose id was not reissued reads its snapshot. -/
theorem readR_deleted (r : RR) (id : Nat) (e : Ent) (hr : r.re.contains id = false)
    (hs : find r.s.snap id = some e) : readR r id = some e := by
  unfold readR readRt; rw [hr]; simp only [Bool.false_eq_true, ite_false, hs]

theorem createRust_shape (s : RS) (d : Nat) :
    (createRust s d).1.snap = s.snap ∧
    ∃ x, (createRust s d).1.live = ((createRust s d).2, x) :: s.live := by
  unfold createRust; split <;> exact ⟨rfl, _, rfl⟩

theorem createR_ReOk (r : RR) (h : ReOk r) (d : Nat) : ReOk (createR r d).1 := by
  obtain ⟨hsn, x, hlv⟩ := createRust_shape r.s d
  intro j hj hl
  simp only [createR] at hj hl ⊢
  rw [hsn] at hj
  rw [hlv] at hl
  by_cases hji : j = (createRust r.s d).2
  · subst hji; simp [hj]
  · rw [find_cons_ne _ _ _ _ hji] at hl
    have := h j hj hl
    split <;> simp [this]

theorem delR_ReOk (r : RR) (h : ReOk r) (id : Nat) : ReOk (delR r id) := by
  unfold delR
  cases hf : find r.s.live id with
  | none => exact h
  | some e =>
    intro j hj hl
    simp only [del, hf, keys, List.map_cons, List.mem_cons] at hj hl ⊢
    by_cases hji : j = id
    · subst hji; rw [find_filter_eq] at hl; cases hl
    · rw [find_filter_ne _ _ _ hji] at hl
      rcases hj with hj | hj
      · exact absurd hj hji
      · simp [h j hj hl, hji]

theorem commitR_ReOk (r : RR) (h : ReOk r) : ReOk (commitR r) := h

/-- **`delR_reads_snapshot`**: right after a delete the id reads the snapshot just taken,
even if it had been reissued before (the re-delete case). -/
theorem delR_reads_snapshot (r : RR) (id : Nat) (e : Ent) (hl : find r.s.live id = some e) :
    readR (delR r id) id = some e := by
  simp [delR, hl, readR, del, readRt, find]

def r0 : RR := ⟨s0, []⟩

/-- `CREATE ()-[:R {v:1}]->()` ; `MATCH ()-[r]->() DELETE r WITH r CREATE ()-[r2:R {v:2}]->()
RETURN r.v, r2.v`: the second CREATE reuses id 0 after the Commit. -/
def relTrace : RR × Nat :=
  let (a, _) := createR r0 1
  let a := commitR a
  let a := commitR (delR a 0)
  createR a 2

/-- **`rel_reissue_reads_new_owner`** (#3051): `r2` got id 0 and reads the new relationship
(before #3051: the deleted one, `rust_reads_dead_entity`). Both handles carry only the id, so
`r` reads it too: `RETURN r.v, r2.v` is `2, 2`, as the C engine returns (checked live on
`falkordb-server:edge-c`; the Rust engine before #3051 returned `1, 1`). -/
theorem rel_reissue_reads_new_owner :
    relTrace.2 = 0 ∧ find relTrace.1.s.live 0 = some ⟨1, 2⟩ ∧ readR relTrace.1 0 = some ⟨1, 2⟩ ∧
      readRt relTrace.1.s 0 = some ⟨0, 1⟩ := by
  decide

end PendingCommit.Snapshot
