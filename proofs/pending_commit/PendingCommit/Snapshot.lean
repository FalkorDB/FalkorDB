/-
# Deleted-entity snapshots vs. id reuse

`Runtime::deleted_nodes` / `deleted_relationships` (runtime.rs) are
query-lifetime maps **keyed by id** holding what a deleted entity looked like,
so a later clause can still read it (`DELETE n WITH n RETURN n.v`). Every
entity read consults them *first*:

A `Value::Node`/`Value::Relationship` carries only the id, and ids freed by a
committed delete are reissued by the next segment's CREATE (`IdSpace::reserve`,
recycle bin first). Model: an entity is `(gen, data)` — `data` stands for any
field one of the accessors above returns — and a handle is `(id, gen)`.

* `readRt` — the accessors' shape: snapshot map first, then the graph.
* `Good` — no snapshotted id is live.
* `sound_under_Good`: under `Good`, `readRt` returns the right entity for every
  handle the query holds (live or deleted-by-this-query).
* `Good` is preserved by delete, commit and a create that does *not* reissue a
  snapshotted id — `deferred_preserves_Good`.

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
  /-- `IdSpace::quarantined` (#3641): ids freed by *earlier batches of this version*. They sit in
  `bin` (the recycle bin) but `reserve` skips them until the version is published, so a later
  segment of the same query never reissues an id the query itself deleted. Not cleared by anything
  in this model: the model is one query, and `MvccGraph` clears it at publication
  (`end_quarantine`, graph_queries `Ids.endIdQuarantine`). -/
  quar : List Nat
  deriving Repr

/-- `DELETE` of a live node: snapshot it (`delete_nodes_bulk`, delete.rs; `delete_entity`, :485-491), drop it from the graph. -/
def del (s : RS) (id : Nat) : RS :=
  match find s.live id with
  | some e => { s with live := s.live.filter (·.1 != id), freed := id :: s.freed,
                       snap := (id, e) :: s.snap }
  | none => s

/-- A `Commit` boundary (`IdSpace::open_batch`, #3641): freed ids go to the recycle bin **and**
into the version's quarantine (`quarantined |= released`). The snapshot map is *not* cleared (it is
query-lifetime). -/
def commit (s : RS) : RS :=
  { s with bin := s.freed ++ s.bin, quar := s.freed ++ s.quar, freed := [] }

/-- The ids `IdSpace::reserve` may reclaim: the recycle bin minus the quarantine
(`reclaim_ids`: `pool - taken - issued - quarantined`; this model has no `taken`/`issued`, and
takes the first such id where Rust takes the smallest: which one is irrelevant to every theorem
below). -/
def avail (s : RS) : List Nat := s.bin.filter (fun i => !s.quar.contains i)

/-- `CREATE` with the engine's allocator since #3641: reclaim from the recycle bin **outside the
quarantine**, else a fresh id above everything ever handed out. -/
def createRust (s : RS) (d : Nat) : RS × Nat :=
  match avail s with
  | id :: _ => ({ s with live := (id, ⟨s.nextGen, d⟩) :: s.live, bin := s.bin.filter (· != id),
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
      simp only [find]; rw [ite_eq_right (by omega)]; exact ih
    · rw [List.filter_cons_of_pos (by simp [hp])]
      simp only [find]
      by_cases hj : p.1 = j
      · rw [ite_eq_left hj, ite_eq_left hj]
      · rw [ite_eq_right hj, ite_eq_right hj, ih]

theorem find_filter_eq (l : List (Nat × Ent)) (id : Nat) : find (l.filter (·.1 != id)) id = none := by
  rw [find_none_iff]; simp [keys]

theorem del_Good (s : RS) (h : Good s) (id : Nat) : Good (del s id) := by
  unfold del
  cases hf : find s.live id with
  | none => simpa [hf] using h
  | some e =>
    simp only
    intro j hj
    simp [keys] at hj
    by_cases hji : j = id
    · subst hji; exact find_filter_eq _ _
    · rw [find_filter_ne _ _ _ hji]
      rcases hj with hj | ⟨e', he'⟩
      · exact absurd hj hji
      · exact h j (by simp [keys]; exact ⟨e', he'⟩)

theorem commit_Good (s : RS) (h : Good s) : Good (commit s) := h

theorem find_cons_ne (l : List (Nat × Ent)) (a : Nat) (x : Ent) (j : Nat) (h : j ≠ a) :
    find ((a, x) :: l) j = find l j := by
  simp only [find]; rw [ite_eq_right (fun h' => h h'.symm)]

theorem find_some_mem (l : List (Nat × Ent)) (id : Nat) (e : Ent) (h : find l id = some e) :
    id ∈ keys l := by
  induction l with
  | nil => simp [find] at h
  | cons p ps ih =>
    simp only [find] at h
    by_cases hp : p.1 = id
    · simp [keys, hp]
    · rw [ite_eq_right hp] at h
      have := ih h
      simp only [keys, List.map_cons, List.mem_cons] at this ⊢
      exact Or.inr this

theorem mem_keys_filter (l : List (Nat × Ent)) (id j : Nat) (h : j ∈ keys (l.filter (·.1 != id))) :
    j ∈ keys l ∧ j ≠ id := by
  simp only [keys, List.mem_map, List.mem_filter, bne_iff_ne, ne_eq] at h ⊢
  obtain ⟨p, ⟨hp, hne⟩, rfl⟩ := h
  exact ⟨⟨p, hp, rfl⟩, hne⟩

/-- The allocator's state invariant, kept by `del`, `commit` and `createRust`:
(a) a live id is in neither the bin nor the freed list, and is below the boundary;
(b) a freed id is not in the bin and is below the boundary; (c) bin and quarantine ids are below
the boundary; (d) every snapshotted id is quarantined or freed by this segment; (e) `Good`. -/
structure Wf (s : RS) : Prop where
  live : ∀ id ∈ keys s.live, id ∉ s.bin ∧ id ∉ s.freed ∧ id < s.fresh
  freed : ∀ id ∈ s.freed, id ∉ s.bin ∧ id < s.fresh
  bin : ∀ id ∈ s.bin, id < s.fresh
  quar : ∀ id ∈ s.quar, id < s.fresh
  snap : ∀ id ∈ keys s.snap, id ∈ s.quar ∨ id ∈ s.freed
  good : Good s

theorem del_Wf (s : RS) (h : Wf s) (id : Nat) : Wf (del s id) := by
  unfold del
  cases hf : find s.live id with
  | none => simpa [hf] using h
  | some e =>
    have hid := find_some_mem _ _ _ hf
    have hl := h.live id hid
    refine ⟨?_, ?_, h.bin, h.quar, ?_, del_Good s h.good id |> fun g => by simpa [del, hf] using g⟩
    · intro j hj
      obtain ⟨hj1, hj2⟩ := mem_keys_filter _ _ _ hj
      have := h.live j hj1
      simp only [List.mem_cons, not_or]
      exact ⟨this.1, ⟨hj2, this.2.1⟩, this.2.2⟩
    · intro j hj
      simp only [List.mem_cons] at hj
      rcases hj with rfl | hj
      · exact ⟨hl.1, hl.2.2⟩
      · exact h.freed j hj
    · intro j hj
      simp only [keys, List.map_cons, List.mem_cons] at hj
      rcases hj with rfl | hj
      · exact Or.inr (by simp)
      · rcases h.snap j (by simpa [keys] using hj) with hq | hq
        · exact Or.inl hq
        · exact Or.inr (by simp [hq])

theorem commit_Wf (s : RS) (h : Wf s) : Wf (commit s) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, h.good⟩
  · intro j hj
    have := h.live j hj
    refine ⟨?_, ?_, this.2.2⟩
    · simp only [commit, List.mem_append, not_or]; exact ⟨this.2.1, this.1⟩
    · simp [commit]
  · intro j hj; simp [commit] at hj
  · intro j hj
    simp only [commit, List.mem_append] at hj
    rcases hj with hj | hj
    · exact (h.freed j hj).2
    · exact h.bin j hj
  · intro j hj
    simp only [commit, List.mem_append] at hj
    rcases hj with hj | hj
    · exact (h.freed j hj).2
    · exact h.quar j hj
  · intro j hj
    rcases h.snap j hj with hq | hq
    · exact Or.inl (by simp [commit, hq])
    · exact Or.inl (by simp [commit, hq])

theorem mem_avail (s : RS) (id : Nat) : id ∈ avail s ↔ id ∈ s.bin ∧ id ∉ s.quar := by
  simp [avail, List.mem_filter]

theorem createRust_Wf (s : RS) (h : Wf s) (d : Nat) : Wf (createRust s d).1 := by
  unfold createRust
  split
  · rename_i id rest hav
    have hm : id ∈ s.bin ∧ id ∉ s.quar := (mem_avail s id).1 (by rw [hav]; simp)
    have hmq : id ∉ s.quar := hm.2
    have hfr : id ∉ s.freed := fun hf => (h.freed id hf).1 hm.1
    refine ⟨?_, ?_, ?_, h.quar, ?_, ?_⟩
    · intro j hj
      simp only [keys, List.map_cons, List.mem_cons] at hj
      rcases hj with rfl | hj
      · exact ⟨by simp, hfr, h.bin _ hm.1⟩
      · have := h.live j (by simpa [keys] using hj)
        exact ⟨fun hb => this.1 (List.mem_filter.1 hb).1, this.2.1, this.2.2⟩
    · intro j hj
      have := h.freed j hj
      refine ⟨fun hb => this.1 (List.mem_filter.1 hb).1, this.2⟩
    · intro j hj; exact h.bin j (List.mem_filter.1 hj).1
    · exact h.snap
    · intro j hj
      simp only
      have hne : j ≠ id := by
        rintro rfl
        rcases h.snap j hj with hq | hq
        · exact hmq hq
        · exact hfr hq
      rw [find_cons_ne _ _ _ _ hne]; exact h.good j hj
  · rename_i hav
    refine ⟨?_, ?_, ?_, ?_, h.snap, ?_⟩
    · intro j hj
      dsimp only at hj ⊢
      simp only [keys, List.map_cons, List.mem_cons] at hj
      rcases hj with rfl | hj
      · refine ⟨fun hb => absurd (h.bin _ hb) (by omega), fun hf => absurd (h.freed _ hf).2 (by omega), by omega⟩
      · have := h.live j (by simpa [keys] using hj)
        exact ⟨this.1, this.2.1, by omega⟩
    · intro j hj; dsimp only at hj ⊢; have := h.freed j hj; exact ⟨this.1, by omega⟩
    · intro j hj; dsimp only at hj ⊢; have := h.bin j hj; omega
    · intro j hj; dsimp only at hj ⊢; have := h.quar j hj; omega
    · intro j hj
      simp only
      have hlt : j < s.fresh := by
        rcases h.snap j hj with hq | hq
        · exact h.quar j hq
        · exact (h.freed j hq).2
      rw [find_cons_ne _ _ _ _ (by omega)]; exact h.good j hj

/-- **`createRust_preserves_Good`** (fixed by #3641): the engine's allocator never reissues an id
this query snapshotted, so every snapshot-first read stays sound (`sound_under_Good`) — over any
sequence of deletes, commits and creates (`Wf` is kept by each). -/
theorem createRust_preserves_Good (s : RS) (h : Wf s) (d : Nat) : Good (createRust s d).1 :=
  (createRust_Wf s h d).good

def s0 : RS := ⟨[], [], [], 0, [], 0, []⟩

theorem wf_s0 : Wf s0 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> intro j hj <;> simp [s0, keys] at hj ⊢

/-- `CREATE (n {v:1})` ; `MATCH (n) DELETE n WITH count(*) AS c CREATE (m {v:2}) …`:
the second query's CREATE no longer takes id 0 after the Commit: `m` gets a fresh id and reads itself
(live: `1|2|[]`; before #3641 `0|1|[A]`, C `0|2|[]`). -/
def trace (create : RS → Nat → RS × Nat) (cm : RS → RS) : RS × Nat :=
  let (a, _) := create s0 1
  let a := cm a
  let a := cm (del a 0)
  create a 2

/-- **Fixed by #3641 (e0d9c2188)**: the #2876 trace now gives `m` a fresh id and every read of
`m` returns `m`, the deleted node is still read from its snapshot. -/
theorem rust_reads_live_entity :
    (trace createRust commit).2 = 1 ∧ readRt (trace createRust commit).1 1 = some ⟨1, 2⟩ ∧
      readRt (trace createRust commit).1 0 = some ⟨0, 1⟩ ∧ Good (trace createRust commit).1 := by
  refine ⟨by decide, by decide, by decide, ?_⟩
  intro j hj
  revert j; decide

end PendingCommit.Snapshot
