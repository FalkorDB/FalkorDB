import FalkorNativeIndex.Batch
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# Writers rebase onto indexer batches; a batch catches up on writers (#3567, b18530683)

Before #3567 an indexer batch that found a writer had committed meanwhile was discarded
(`Discarded::Writers`), and a writer whose fork was made before a batch committed published its own
copy of the building index, without the batch's progress: `MvccGraph::is_stale` compares only the
data version, which a batch does not move (`FINDINGS.md`, the 2026-11-03 gap). #3567 closes both:

* the index keeps, while `Building`, the entities writers changed in it (`Written`: `since_last_indexer_step`,
  `by_this_version`, `versioned.rs:68-81`);
* a writer, as it escalates (`QuerySession::check_fork`, `query_session.rs:246`), is *rebased* onto the batches
  published since its fork (`MvccGraph::rebase_onto_indexer_batches`, `mvcc_graph.rs:166` ->
  `Graph::rebase_onto_indexer_batches`, `versioned.rs:486`): each index a batch moved on becomes the published
  copy, with the fork's own changes re-applied;
* a batch that writers committed after (`IndexerCommit::CatchUp`) is caught up on what they changed
  (`Graph::caught_up_onto`, `versioned.rs:438`), instead of being discarded.

Both use one function, `IndexColumn::refresh` (`falkordb_index.rs:169`).

| Lean | Rust |
| --- | --- |
| `Ver` (`has`, `attr`, `place`), `tupl` | what a `Graph` version says of a column's entities: the label / type check (`node_has_label_id`, `edge_has_type_id`), the attribute read, `build_place` |
| `G.ver` | the model `G` read as a `Ver` of one column |
| `rawTuples` | the removes of `refresh` (`:178-199`): the attribute read from `from`, no label check |
| `entityTuples` | `graph_writes::entity_tuples` (`graph_writes.rs:444`): the label or type is checked per id |
| `placedBelow` | `placed_below` (before #3568; `refresh` now calls `placed_in(.., None, Some(next), ..)`: `placedIn_none_eq_placedBelow_*`) |
| `placedIn`, `IndexColumn.indexerInsert`, `catchUpBy` | `placed_in` (`falkordb_index.rs:222`), `indexer_insert` (`:356`), `catch_up_by` (`versioned.rs:209`) (#3568) |
| `IndexColumn.refresh` | `IndexColumn::refresh` (`:169`), `writer_rebase` (`:326`), `indexer_catch_up` (`:337`) |
| `Written`, `BS`, `EN` | `Written`, `BuildState<BuildCursor>`, `IndexEntry<IndexColumn>` |
| `EN.writerRecord`, `EN.indexerApply`, `EN.publish` | `IndexEntry::writer_record` (`:280`), `indexer_apply` (`:293`), `publish` (`:324`) |
| `overtaken`, `EN.rebase` | `Graph::overtaken` (`:514`), `rebase_index` (`:533`) |
| `EN.catchUp` | `Graph::catch_up_index` (`:454`) |

The old model (`BuildState.building next`, `SE`) is the *projection* of this one that forgets `written`:
`EN.proj`. `writerRecord_proj`, `indexerApply_proj` and `publish_proj` show each new operation projects onto
the old one, so every theorem of `Pending.lean`, `Build.lean` and `Batch.lean` still describes the new code.

Theorems:
* `refresh_mem`: what the refresh does to the column, exactly (the removed and the added tuples);
  `entityTuples_spec`, `rawTuples_spec`, `placedBelow_mem`.
* `refresh_inv` (**conditional on `Frame`**, see below): a column that holds exactly the tuples of the entities
  `below` its state in `from`, refreshed over a set `W` outside which `from` and `to` agree, holds exactly those
  of `to`.
* `rebase_progress`: a rebased fork has the published copy's generation, commit count and build state: the batches'
  progress is kept. `rebase_keeps_writes`: the fork's recorded writes stay in `by_this_version`, so `publish` hands
  them on. `rebase_index_spec`: its column is `refresh` of the published one over those writes.
  `rebase_inv`: with `Frame`, it holds exactly the fork's tuples below the published cursor.
* `catchUp_spec`, `catchUp_inv`: the same for a batch's copy over the writers' `since` set.
* `writerRecord_records`: every id a change names is recorded, applied or not.
* `V.rebased_keeps_data`: a rebase changes no data version (the fork stays current).
* **The fix of the 2026-11-03 gap** (#3567; no history lemma kept, see FINDINGS): `rebased_commit_keeps_chain`,
  `rebased_batch_fits` (the published copy has the batches' count and the next batch fits).

`Frame` is the one thing assumed: outside `W`, the two versions agree on what each entity contributes and on
where it sits. For a writer, `W` = `by_this_version`, which `writerRecord_records` fills with the id of every
change the unit staged (the staged changes of `Writes.lean`, `*_changes`, name every entity whose tuple a write
changes); for a batch, `W` = `since_last_indexer_step`, the union of the writers' sets. That every write the graph
makes is staged is proven per write in `Writes.lean` and not composed here, so the two invariant theorems are
conditional and the Rust rows rest on the unconditional specs.
-/
namespace NI

/-! ## What a version says about a column's entities -/

/-- What a graph version says about the entities of one column: `has` (the label or type check),
    `attr` (the attribute read, `None` if the attribute id does not exist) and `place` (`build_place`). -/
structure Ver where
  has : Nat → Bool
  attr : Nat → Option Value
  place : Nat → Option BuildCursor

/-- The tuples entity `id` holds in the column in a version: none without the label or type. -/
def tupl (c : IndexColumn) (v : Ver) (id : Nat) : List Nat :=
  if v.has id then tupleOf c v.attr id else []

/-- `entity_tuples` (`graph_writes.rs:444-486`): for each id, its tuple if the label or type holds. -/
def entityTuples (c : IndexColumn) (v : Ver) (ids : List Nat) : List Nat := ids.flatMap (tupl c v)

/-- The removes of `refresh` (`falkordb_index.rs:178-199`): the attribute read from `from`, no label
    check. -/
def rawTuples (c : IndexColumn) (v : Ver) (ids : List Nat) : List Nat := ids.flatMap (tupleOf c v.attr)

/-- The model graph `G` read as the `Ver` of column `(e, k)`. -/
def G.ver (g : G) (e : EntityType) (k : IndexKey) : Ver where
  has id := match e with
    | .node => (g.labels.idxOf? k.1).any (fun l => (g.nodeLabels id).contains l)
    | .relationship => (g.types.idxOf? k.1).any (fun t => g.relType id == t)
  attr id := (g.attrId k.2).bind (fun a => match e with | .node => g.nodeAttr id a | .relationship => g.relAttr id a)
  place := g.place e

theorem G.ver_place (g : G) (e : EntityType) (k : IndexKey) : (g.ver e k).place = g.place e := rfl

/-- `placed_below` (`falkordb_index.rs:225-244`): the ids of `ws` that `to` places below `next`: for a node
    cursor one range of the set (`remove_range(next..)`), for an edge cursor one `build_place` lookup each. -/
def placedBelow (to : Ver) (ws : List Nat) (next : BuildCursor) : List Nat :=
  match next with
  | .node m => ws.filter (fun id => decide (id < m))
  | .edge _ _ => ws.filter (fun id => (to.place id).any (fun pl => pl.precedes next))

/-- Whether version `v` puts `id` in the column in state `st`: placed below the cursor while `Building`,
    always when `Ready`. -/
def below (st : BuildState) (v : Ver) (id : Nat) : Bool :=
  match st with
  | .ready => true
  | .building next => (v.place id).any (fun pl => pl.precedes next)

theorem entityTuples_spec (c : IndexColumn) (v : Ver) (ids : List Nat) (t : Nat) :
    t ∈ entityTuples c v ids ↔ ∃ id ∈ ids, t ∈ tupl c v id := by
  simp [entityTuples, List.mem_flatMap]

theorem rawTuples_spec (c : IndexColumn) (v : Ver) (ids : List Nat) (t : Nat) :
    t ∈ rawTuples c v ids ↔ ∃ id ∈ ids, t ∈ tupleOf c v.attr id := by
  simp [rawTuples, List.mem_flatMap]

/-- What `placed_below` keeps: the ids of `ws` that `to` places below the cursor. For a node cursor, the
    range test is the place test whenever `to` places a node at its own id (the node column's `place`). -/
theorem placedBelow_mem (to : Ver) (ws : List Nat) (next : BuildCursor)
    (hk : ∀ m, next = .node m → ∀ id, to.place id = some (.node id)) (id : Nat) :
    id ∈ placedBelow to ws next ↔ id ∈ ws ∧ below (.building next) to id = true := by
  cases next with
  | node m =>
    simp only [placedBelow, below, List.mem_filter, decide_eq_true_eq]
    rw [hk m rfl id]
    simp [BuildCursor.precedes]
  | edge s d => simp [placedBelow, below, List.mem_filter]

/-! ## `refresh` -/

/-- `IndexColumn::refresh` (`falkordb_index.rs:169-222`): the tuples of `ws` as `frm` has them come out, and as
    `to` has them go in, while `st` is `Building` only for an entity `to` places below the cursor. -/
def IndexColumn.refresh (c : IndexColumn) (st : BuildState) (ws : List Nat) (frm to : Ver) : IndexColumn :=
  if ws.isEmpty then c else
  let removes := rawTuples c frm ws
  let adds := match st with
    | .ready => entityTuples c to ws
    | .building next => entityTuples c to (placedBelow to ws next)
  ((c.removeEncoded ⟨removes⟩).1.addEncoded ⟨adds⟩).1

/-- What `refresh` does to the column's content, exactly. -/
theorem refresh_mem (c : IndexColumn) (st : BuildState) (ws : List Nat) (frm to : Ver)
    (hk : ∀ m, st = .building (.node m) → ∀ id, to.place id = some (.node id)) (t : Nat) :
    t ∈ (c.refresh st ws frm to).ix.tree ↔
      (t ∈ c.ix.tree ∧ ∀ id ∈ ws, t ∉ tupleOf c frm.attr id) ∨
      (∃ id ∈ ws, below st to id = true ∧ t ∈ tupl c to id) := by
  cases c with
  | numeric ix =>
    by_cases hW : ws.isEmpty = true
    · have : ws = [] := by simpa using hW
      subst this
      simp [IndexColumn.refresh, IndexColumn.ix]
    · simp only [IndexColumn.refresh, hW, Bool.false_eq_true, if_false, IndexColumn.removeEncoded,
        IndexColumn.addEncoded, IndexColumn.ix, addEncoded_spec, removeEncoded_spec]
      have hr : ∀ t, t ∈ (rawTuples (.numeric ix) frm ws) ↔ ∃ id ∈ ws, t ∈ tupleOf (.numeric ix) frm.attr id :=
        rawTuples_spec _ _ _
      have hrm : t ∈ ix.tree ∧ t ∉ rawTuples (.numeric ix) frm ws ↔
          t ∈ ix.tree ∧ ∀ id ∈ ws, t ∉ tupleOf (.numeric ix) frm.attr id := by
        rw [hr]; simp only [not_exists, not_and]
      have hadd : t ∈ (match st with
          | .ready => entityTuples (.numeric ix) to ws
          | .building next => entityTuples (.numeric ix) to (placedBelow to ws next)) ↔
          ∃ id ∈ ws, below st to id = true ∧ t ∈ tupl (.numeric ix) to id := by
        cases st with
        | ready => simp [entityTuples_spec, below]
        | building next =>
          simp only [entityTuples_spec]
          constructor
          · rintro ⟨id, hid, ht⟩
            have := (placedBelow_mem to ws next (fun m h => hk m (by rw [h])) id).1 hid
            exact ⟨id, this.1, this.2, ht⟩
          · rintro ⟨id, hid, hb, ht⟩
            exact ⟨id, (placedBelow_mem to ws next (fun m h => hk m (by rw [h])) id).2 ⟨hid, hb⟩, ht⟩
      rw [← hrm, ← hadd]

/-- `Inv c st v`: the column holds exactly the tuples of the entities `v` puts in it. -/
def Inv (c : IndexColumn) (st : BuildState) (v : Ver) : Prop :=
  ∀ t, t ∈ c.ix.tree ↔ ∃ id, below st v id = true ∧ t ∈ tupl c v id

/-- Outside `ws`, two versions agree on what each entity contributes and on where it sits. -/
def Frame (c : IndexColumn) (st : BuildState) (ws : List Nat) (frm to : Ver) : Prop :=
  ∀ id, id ∉ ws → below st frm id = below st to id ∧ tupl c frm id = tupl c to id

/-- A tuple of entity `id` is `enc k id`. -/
theorem tupleOf_enc (c : IndexColumn) (attr : Nat → Option Value) (id t : Nat) (h : t ∈ tupleOf c attr id) :
    ∃ k, t = enc k id := by
  unfold tupleOf at h
  cases ha : attr id with
  | none => simp [ha] at h
  | some v =>
    simp only [ha] at h
    cases hk : c.encodeValue v with
    | none => simp [hk] at h
    | some k => simp [hk] at h; exact ⟨k, h⟩

theorem tupl_enc (c : IndexColumn) (v : Ver) (id t : Nat) (h : t ∈ tupl c v id) : ∃ k, t = enc k id := by
  unfold tupl at h
  split at h
  · exact tupleOf_enc c v.attr id t h
  · simp at h

theorem tupl_numeric (ix ix' : NumericIndex) (v : Ver) (id : Nat) :
    tupl (.numeric ix) v id = tupl (.numeric ix') v id := by
  simp [tupl, tupleOf, IndexColumn.encodeValue]

/-- `refresh` keeps the column's kind. -/
theorem refresh_numeric (ix : NumericIndex) (st : BuildState) (ws : List Nat) (frm to : Ver) :
    ∃ ix', (IndexColumn.numeric ix).refresh st ws frm to = .numeric ix' := by
  unfold IndexColumn.refresh
  split
  · exact ⟨ix, rfl⟩
  · exact ⟨_, rfl⟩

/-- **A refresh keeps a column exact.** The column holds the tuples of the entities `frm` puts in it; `ws`
    covers every entity on which `frm` and `to` differ (`Frame`); entity ids fit the tuple's id field (`W`). Then
    the refreshed column holds the tuples of the entities `to` puts in it. -/
theorem refresh_inv (c : IndexColumn) (st : BuildState) (ws : List Nat) (frm to : Ver)
    (hk : ∀ m, st = .building (.node m) → ∀ id, to.place id = some (.node id))
    (hinv : Inv c st frm) (hf : Frame c st ws frm to)
    (hws : ∀ id ∈ ws, id < W) (hb : ∀ id, tupl c frm id ≠ [] → id < W) :
    Inv (c.refresh st ws frm to) st to := by
  intro t
  have hkind : ∀ id, tupl (c.refresh st ws frm to) to id = tupl c to id := by
    intro id
    cases c with
    | numeric ix =>
      obtain ⟨ix', h⟩ := refresh_numeric ix st ws frm to
      rw [h]; exact tupl_numeric _ _ _ _
  simp only [Inv, hkind]
  rw [refresh_mem c st ws frm to hk t]
  have hsub : ∀ id t, t ∈ tupl c frm id → t ∈ tupleOf c frm.attr id := by
    intro id t h; unfold tupl at h; split at h <;> simp_all
  constructor
  · rintro (⟨ht, hn⟩ | ⟨id, hid, hbl, ht⟩)
    · obtain ⟨id, hbl, hid⟩ := (hinv t).1 ht
      by_cases hmem : id ∈ ws
      · exact absurd (hsub id t hid) (hn id hmem)
      · have := hf id hmem
        exact ⟨id, by rw [← this.1]; exact hbl, by rw [← this.2]; exact hid⟩
    · exact ⟨id, hbl, ht⟩
  · rintro ⟨id, hbl, ht⟩
    by_cases hmem : id ∈ ws
    · exact Or.inr ⟨id, hmem, hbl, ht⟩
    · left
      have hfr := hf id hmem
      have hold : t ∈ tupl c frm id := by rw [hfr.2]; exact ht
      refine ⟨(hinv t).2 ⟨id, by rw [hfr.1]; exact hbl, hold⟩, ?_⟩
      intro id' hid' hraw
      obtain ⟨k, hk1⟩ := tupl_enc c frm id t hold
      obtain ⟨k', hk2⟩ := tupleOf_enc c frm.attr id' t hraw
      have hlt : id < W := hb id (by intro h; rw [h] at hold; simp at hold)
      have := (enc_inj k id k' id' hlt (hws id' hid')).1 (hk1.symm.trans hk2)
      exact hmem (this.2 ▸ hid')

/-! ## The entries: `Written`, `BuildState<BuildCursor>`, `IndexEntry` -/

/-- `Written` (`versioned.rs:70-81`): the entities writers changed in a `Building` index. `since` is
    `since_last_indexer_step`, `byThis` is `by_this_version`; sets modelled as lists, only membership matters. -/
structure Written where
  since : List Nat
  byThis : List Nat

/-- `BuildState<BuildCursor>` with its `Written` (`versioned.rs:44-52`). -/
inductive BS where
  | building (next : BuildCursor) (w : Written)
  | ready

/-- `BuildState::building` (`:55-61`): no entity written yet. -/
def BS.mk (next : BuildCursor) : BS := .building next ⟨[], []⟩

/-- The old model's state: forget `written`. -/
def BS.proj : BS → BuildState
  | .building n _ => .building n
  | .ready => .ready

/-- `BuildState::next` (`:63-68`). -/
def BS.next : BS → Option BuildCursor
  | .building n _ => some n
  | .ready => none

theorem BS.next_proj (s : BS) : s.proj = (match s.next with | some n => .building n | none => .ready) := by
  cases s <;> rfl

/-- `IndexEntry<IndexColumn>` (`:96-110`): column, state with its `Written`, generation, commit count. -/
structure EN where
  index : IndexColumn
  state : BS
  generation : Nat
  commits : Nat

/-- The old model's entry. -/
def EN.proj (e : EN) : SE :=
  ⟨⟨e.index, e.state.proj⟩, e.generation, e.commits,
    match e.state with | .building _ w => w.since.eraseDups.length | .ready => 0⟩

def EN.isBuilding (e : EN) : Bool := match e.state with | .building _ _ => true | .ready => false

theorem EN.isBuilding_proj (e : EN) : e.proj.en.isBuilding = e.isBuilding := by
  cases e with | mk i s g c => cases s <;> rfl

/-- `IndexEntry::writer_record` (`versioned.rs:280`) with `IndexColumn::writer_record`
    (`falkordb_index.rs:306-324`): while `Building`, the id of every change goes into `by_this_version`, applied
    or not; the column is recorded as before. -/
def EN.writerRecord (e : EN) (log : List IndexWrite) (place : Nat → Option BuildCursor)
    (sc : EncodedTuples × EncodedTuples) : EN × (EncodedTuples × EncodedTuples) :=
  let st : BS := match e.state with
    | .building n w => .building n ⟨w.since, w.byThis ++ log.map (·.id)⟩
    | .ready => .ready
  let r := e.index.writerRecord e.state.proj log place sc.1 sc.2
  (⟨r.1, st, e.generation, e.commits⟩, (r.2.1, r.2.2))

/-- The new `writer_record` projects onto the old one: same column, same state, same buffers. -/
theorem writerRecord_proj (e : EN) (log : List IndexWrite) (place : Nat → Option BuildCursor)
    (sc : EncodedTuples × EncodedTuples) :
    ((e.writerRecord log place sc).1.proj.en = (e.proj.en.writerRecord log place sc).1) ∧
    (e.writerRecord log place sc).2 = (e.proj.en.writerRecord log place sc).2 ∧
    (e.writerRecord log place sc).1.generation = e.generation ∧ (e.writerRecord log place sc).1.commits = e.commits := by
  cases e with | mk i s g c => cases s <;> simp [EN.writerRecord, EN.proj, SE.mk, IndexEntry.writerRecord, BS.proj]

/-- **Every change's entity is recorded** (while `Building`), whether or not the cursor keeps the change. -/
theorem writerRecord_records (e : EN) (log : List IndexWrite) (place : Nat → Option BuildCursor)
    (sc : EncodedTuples × EncodedTuples) (n : BuildCursor) (w : Written) (h : e.state = .building n w)
    (x : IndexWrite) (hx : x ∈ log) :
    ∃ n' w', (e.writerRecord log place sc).1.state = .building n' w' ∧ x.id ∈ w'.byThis ∧ n' = n ∧ w'.since = w.since := by
  cases e with | mk i s g c =>
    simp only at h; subst h
    exact ⟨n, ⟨w.since, w.byThis ++ log.map (·.id)⟩, by simp [EN.writerRecord],
      by simp [List.mem_append]; exact Or.inr ⟨x, hx, rfl⟩, rfl, rfl⟩

/-- A `Ready` entry records nothing: no state to carry. -/
theorem writerRecord_ready_state (e : EN) (log : List IndexWrite) (place : Nat → Option BuildCursor)
    (sc : EncodedTuples × EncodedTuples) (h : e.state = .ready) :
    (e.writerRecord log place sc).1.state = .ready := by
  cases e with | mk i s g c => simp only at h; subst h; rfl

/-- `IndexEntry::indexer_apply` (`versioned.rs:293-322`): the batch goes in, the commit count rises, the cursor
    moves on (or the index is `Ready`), and `since_last_indexer_step` is emptied: the batch read every entity at
    this version's values, writers' changes included. `by_this_version` stays. -/
def EN.indexerApply (e : EN) (work : EncodedTuples) (next : Option BuildCursor) : EN × EncodedTuples × Bool :=
  match e.state with
  | .ready => (e, work, false)
  | .building _ w =>
    let r := e.index.addEncoded work
    (⟨r.1, (match next with | none => .ready | some n => .building n ⟨[], w.byThis⟩),
      e.generation, e.commits + 1⟩, r.2, true)

theorem indexerApply_proj (e : EN) (work : EncodedTuples) (next : Option BuildCursor) :
    ((e.indexerApply work next).1.proj = (e.proj.indexerApply work next).1) ∧
    (e.indexerApply work next).2 = (e.proj.indexerApply work next).2 := by
  cases e with | mk i s g c =>
    cases s <;> cases next <;> simp [EN.indexerApply, EN.proj, SE.indexerApply, IndexEntry.isBuilding,
      IndexEntry.indexerApply, BS.proj]

/-- After a batch the index's `since` set is empty (while still `Building`), and `by_this_version` is kept. -/
theorem indexerApply_clears_since (e : EN) (work : EncodedTuples) (n' : BuildCursor) (n : BuildCursor) (w : Written)
    (h : e.state = .building n w) :
    ∃ w', (e.indexerApply work (some n')).1.state = .building n' w' ∧ w'.since = [] ∧ w'.byThis = w.byThis := by
  cases e with | mk i s g c =>
    simp only at h; subst h
    exact ⟨⟨[], w.byThis⟩, by simp [EN.indexerApply], rfl, rfl⟩

/-- `IndexEntry::publish` (`:324-334`): while `Building`, what this version changed joins `since`. -/
def EN.publish (e : EN) : EN :=
  match e.state with
  | .building n w => { e with state := .building n ⟨w.since ++ w.byThis, []⟩ }
  | .ready => e

/-- Publishing changes nothing of the entry's old-model content: the column, the cursor (or `Ready`), the
generation and the commit count. (The `since` count of `SE` does move: `publish_since_len`.) -/
theorem publish_proj (e : EN) :
    e.publish.proj.en = e.proj.en ∧ e.publish.proj.generation = e.proj.generation ∧
    e.publish.proj.commits = e.proj.commits := by
  cases e with | mk i s g c => cases s <;> exact ⟨rfl, rfl, rfl⟩

/-- ...and the entities this version changed join the `since` count the batch fit reads. -/
theorem publish_since_len (e : EN) (n : BuildCursor) (w : Written) (h : e.state = .building n w) :
    e.publish.proj.since = (w.since ++ w.byThis).eraseDups.length := by
  cases e with | mk i s g c =>
    simp only at h; subst h; rfl

theorem publish_since (e : EN) (n : BuildCursor) (w : Written) (h : e.state = .building n w) (id : Nat) :
    ∃ w', e.publish.state = .building n w' ∧ w'.byThis = [] ∧
      (id ∈ w'.since ↔ id ∈ w.since ∨ id ∈ w.byThis) := by
  cases e with | mk i s g c =>
    simp only at h; subst h
    exact ⟨⟨w.since ++ w.byThis, []⟩, by simp [EN.publish], rfl, by simp⟩

/-! ## The writer's rebase and the batch's catch-up -/

/-- `VersionedIndex::writer_rebase` for `IndexColumn` (`falkordb_index.rs:326-335`): `refresh` from the published
    version to the fork. The trait declaration is `versioned.rs:213-225`. -/
def IndexColumn.writerRebase (c : IndexColumn) (st : BuildState) (ws : List Nat) (published fork : Ver) : IndexColumn :=
  c.refresh st ws published fork

/-- `VersionedIndex::indexer_catch_up` for `IndexColumn` (`falkordb_index.rs:337-348`): `refresh` from the
    batch's version to the published one. The trait declaration is `versioned.rs:227-237`. -/
def IndexColumn.indexerCatchUp (c : IndexColumn) (st : BuildState) (ws : List Nat) (batchVer published : Ver) :
    IndexColumn :=
  c.refresh st ws batchVer published

theorem writerRebase_spec (c : IndexColumn) (st : BuildState) (ws : List Nat) (published fork : Ver) :
    c.writerRebase st ws published fork = c.refresh st ws published fork := rfl

theorem indexerCatchUp_spec (c : IndexColumn) (st : BuildState) (ws : List Nat) (batchVer published : Ver) :
    c.indexerCatchUp st ws batchVer published = c.refresh st ws batchVer published := rfl

/-- `Graph::overtaken` (`versioned.rs:514-531`): an index still `Building` here that a batch moved on in the
    published copy: the same index (generation), more batches committed on it. -/
def overtaken (ours theirs : EN) : Bool :=
  ours.isBuilding && theirs.generation == ours.generation && theirs.commits != ours.commits

/-- `Graph::rebase_index` (`versioned.rs:533-565`), the slot after it, with `published` and `fork` the two
    versions: the published copy, with the fork's recorded writes re-applied (`refresh`) and still recorded. -/
def EN.rebase (ours theirs : EN) (published fork : Ver) : EN :=
  if overtaken ours theirs then
    match ours.state with
    | .ready => ours
    | .building _ w =>
      let written := w.byThis
      let idx := theirs.index.writerRebase theirs.state.proj written published fork
      let st : BS := match theirs.state with
        | .building n tw => .building n ⟨tw.since, written⟩
        | .ready => .ready
      ⟨idx, st, theirs.generation, theirs.commits⟩
  else ours

/-- Not overtaken: the fork keeps its own copy. -/
theorem rebase_not_overtaken (ours theirs : EN) (published fork : Ver) (h : overtaken ours theirs = false) :
    ours.rebase theirs published fork = ours := by
  simp [EN.rebase, h]

/-- **The batches' progress is kept.** A rebased fork has the published copy's generation, commit count
    and build state (cursor, or `Ready`). -/
theorem rebase_progress (ours theirs : EN) (published fork : Ver) (h : overtaken ours theirs = true) :
    (ours.rebase theirs published fork).generation = theirs.generation ∧
    (ours.rebase theirs published fork).commits = theirs.commits ∧
    (ours.rebase theirs published fork).state.proj = theirs.state.proj := by
  cases ours with | mk i s g c =>
    cases s with
    | ready => simp [overtaken, EN.isBuilding] at h
    | building n w => cases hs : theirs.state <;> simp [EN.rebase, h, hs, BS.proj]

/-- **The fork's writes stay recorded**: `by_this_version` is carried over, so the publish that follows hands
    them on to `since`. -/
theorem rebase_keeps_writes (ours theirs : EN) (published fork : Ver) (n : BuildCursor) (w : Written)
    (hs : ours.state = .building n w) (h : overtaken ours theirs = true) (n' : BuildCursor) (tw : Written)
    (ht : theirs.state = .building n' tw) :
    (ours.rebase theirs published fork).state = .building n' ⟨tw.since, w.byThis⟩ := by
  simp [EN.rebase, h, hs, ht]

/-- The rebased column is the published one refreshed over the fork's writes. -/
theorem rebase_index_spec (ours theirs : EN) (published fork : Ver) (n : BuildCursor) (w : Written)
    (hs : ours.state = .building n w) (h : overtaken ours theirs = true) :
    (ours.rebase theirs published fork).index = theirs.index.refresh theirs.state.proj w.byThis published fork := by
  simp [EN.rebase, h, hs, IndexColumn.writerRebase]

/-- **With `Frame`, the rebased column is exact** for the fork: it holds the tuples of the entities the fork
    places in it, at the published cursor. -/
theorem rebase_inv (ours theirs : EN) (published fork : Ver) (n : BuildCursor) (w : Written)
    (hs : ours.state = .building n w) (h : overtaken ours theirs = true)
    (hk : ∀ m, theirs.state.proj = .building (.node m) → ∀ id, fork.place id = some (.node id))
    (hinv : Inv theirs.index theirs.state.proj published)
    (hf : Frame theirs.index theirs.state.proj w.byThis published fork)
    (hws : ∀ id ∈ w.byThis, id < W) (hb : ∀ id, tupl theirs.index published id ≠ [] → id < W) :
    Inv (ours.rebase theirs published fork).index theirs.state.proj fork := by
  rw [rebase_index_spec ours theirs published fork n w hs h]
  exact refresh_inv _ _ _ _ _ hk hinv hf hws hb

/-- `Graph::catch_up_index` (`versioned.rs:454-484`): the batch's copy, its index refreshed over the entities
    writers changed (`since` of the published copy, none if it is `Ready`) from the batch's version to the
    published one. -/
def EN.catchUp (ours theirs : EN) (batchVer published : Ver) : EN :=
  let written := match theirs.state with | .building _ tw => tw.since | .ready => []
  { ours with index := ours.index.indexerCatchUp ours.state.proj written batchVer published }

theorem catchUp_spec (ours theirs : EN) (batchVer published : Ver) :
    (ours.catchUp theirs batchVer published).state = ours.state ∧
    (ours.catchUp theirs batchVer published).generation = ours.generation ∧
    (ours.catchUp theirs batchVer published).commits = ours.commits ∧
    (ours.catchUp theirs batchVer published).index = ours.index.refresh ours.state.proj
      (match theirs.state with | .building _ tw => tw.since | .ready => []) batchVer published := ⟨rfl, rfl, rfl, rfl⟩

/-- **With `Frame`, the caught-up column is exact** for the published version. -/
theorem catchUp_inv (ours theirs : EN) (batchVer published : Ver)
    (hk : ∀ m, ours.state.proj = .building (.node m) → ∀ id, published.place id = some (.node id))
    (hinv : Inv ours.index ours.state.proj batchVer)
    (hf : Frame ours.index ours.state.proj (match theirs.state with | .building _ tw => tw.since | .ready => [])
      batchVer published)
    (hws : ∀ id ∈ (match theirs.state with | .building _ tw => tw.since | .ready => []), id < W)
    (hb : ∀ id, tupl ours.index batchVer id ≠ [] → id < W) :
    Inv (ours.catchUp theirs batchVer published).index ours.state.proj published :=
  refresh_inv _ _ _ _ _ hk hinv hf hws hb

/-! ## Catching up by insertion (#3568)

A batch that writers overtook either refreshes the entities they changed (above) or, when that costs
more, inserts itself into the *published* copy of the index (`CatchUp::Insert`, `versioned.rs:146-183`,
`Graph::insert_index`, `VersionedIndex::indexer_insert`, `falkordb_index.rs:356-374`). `placed_below` became
`placed_in`, with an optional lower end. -/

/-- `placed_in` (`falkordb_index.rs:222-254`): the ids of `ws` whose place in `to` is at or after `frm` and
    before `next` (a `None` end leaves that side open). A node column cuts the set by id at the ends
    (`remove_range`, whatever `to` says of the place); a relationship column keeps the edges `to` still has
    whose place is not before `frm` and is before `next`, one `build_place` lookup each. -/
def placedIn (e : EntityType) (to : Ver) (ws : List Nat) (frm next : Option BuildCursor) : List Nat :=
  match e with
  | .node =>
    ws.filter (fun id =>
      (match frm with | some (.node f) => decide (f ≤ id) | _ => true) &&
      (match next with | some (.node n) => decide (id < n) | _ => true))
  | .relationship =>
    ws.filter (fun id => (to.place id).any (fun pl =>
      (match frm with | none => true | some f => !pl.precedes f) &&
      (match next with | none => true | some n => pl.precedes n)))

/-- What `placed_in` keeps on a node column: the ids of `ws` cut by id at both ends (`remove_range`), whatever
    `to` says of the place (so it takes no hypothesis about `to`, and a cursor of another kind is no end). -/
theorem placedIn_mem (to : Ver) (ws : List Nat) (frm next : Option BuildCursor) (id : Nat) :
    id ∈ placedIn .node to ws frm next ↔
      id ∈ ws ∧ (match frm with | some (.node f) => f ≤ id | _ => True) ∧
        (match next with | some (.node n) => id < n | _ => True) := by
  simp only [placedIn, List.mem_filter, Bool.and_eq_true]
  rcases frm with _ | _ | _ <;> rcases next with _ | _ | _ <;> simp

/-- What `placed_in` keeps on a relationship column, for any ends: the ids of `ws` whose edge `to` still has
    (`build_place` is `Some`) and places at or after `frm` and before `next` (a `None` end is open). -/
theorem placedIn_rel_mem (to : Ver) (ws : List Nat) (frm next : Option BuildCursor) (id : Nat) :
    id ∈ placedIn .relationship to ws frm next ↔
      id ∈ ws ∧ ∃ pl, to.place id = some pl ∧
        (match frm with | none => True | some f => ¬ pl.precedes f = true) ∧
        (match next with | none => True | some n => pl.precedes n = true) := by
  simp only [placedIn, List.mem_filter]
  constructor
  · rintro ⟨h, h2⟩
    refine ⟨h, ?_⟩
    cases hp : to.place id with
    | none => simp [hp] at h2
    | some pl =>
      refine ⟨pl, rfl, ?_, ?_⟩
      · cases frm <;> simp_all
      · cases next <;> simp_all
  · rintro ⟨h, pl, hp, h1, h2⟩
    refine ⟨h, ?_⟩
    simp only [hp, Option.any_some]
    cases frm <;> cases next <;> simp_all

/-- With no lower end, `placed_in` is the old `placed_below` (the call `refresh` now makes,
    `placed_in(written, entity, None, Some(next), to)`, `falkordb_index.rs:213`). -/
theorem placedIn_none_eq_placedBelow_node (to : Ver) (ws : List Nat) (m : Nat) :
    placedIn .node to ws none (some (.node m)) = placedBelow to ws (.node m) := by
  simp [placedIn, placedBelow]

theorem placedIn_none_eq_placedBelow_edge (to : Ver) (ws : List Nat) (s d : Nat) :
    placedIn .relationship to ws none (some (.edge s d)) = placedBelow to ws (.edge s d) := by
  simp [placedIn, placedBelow]

/-- `VersionedIndex::indexer_insert` for `IndexColumn` (`falkordb_index.rs:356-374`): the batch's tuples of the
    entities writers did not change, plus the written entities placed in `[frm, next)` with their values in
    `published`, added to `c` (the published copy's column, whose cursor is at `frm`). -/
def IndexColumn.indexerInsert (c : IndexColumn) (e : EntityType) (published : Ver) (batch : EncodedTuples)
    (written : List Nat) (frm : BuildCursor) (next : Option BuildCursor) : IndexColumn :=
  let kept := batch.scalar.filter (fun t => !written.contains (docOf t))
  let inRange := placedIn e published written (some frm) next
  (c.addEncoded ⟨kept ++ entityTuples c published inRange⟩).1

/-- **What `indexer_insert` does to the column, exactly**: the published column keeps its tuples; the batch
    contributes those of entities no writer changed; each written entity placed in the batch's range
    contributes its tuple as `published` has it (an entity below `frm` is already right in `c`, one at or past
    `next` is a later batch's). -/
theorem indexerInsert_mem (c : IndexColumn) (e : EntityType) (published : Ver) (batch : EncodedTuples)
    (written : List Nat) (frm : BuildCursor) (next : Option BuildCursor) (t : Nat) :
    t ∈ (c.indexerInsert e published batch written frm next).ix.tree ↔
      t ∈ c.ix.tree ∨ (t ∈ batch.scalar ∧ docOf t ∉ written) ∨
        ∃ id ∈ placedIn e published written (some frm) next, t ∈ tupl c published id := by
  have hc : ∀ x : Nat, ((!written.contains x) = true ↔ x ∉ written) := by intro x; simp
  cases c with
  | numeric ix =>
    simp only [IndexColumn.indexerInsert, IndexColumn.addEncoded, IndexColumn.ix, addEncoded_spec,
      List.mem_append, List.mem_filter, entityTuples_spec, hc]

/-- Whether version `v` places `id` at or after the cursor `frm` (the lower end of `placed_in`). -/
def notBefore (frm : BuildCursor) (v : Ver) (id : Nat) : Bool := (v.place id).any (fun pl => !pl.precedes frm)

/-- The state a batch leaves: `Ready` if it ran out of entities (`next = None`), else `Building next`. -/
def stOf (next : Option BuildCursor) : BuildState :=
  match next with | none => .ready | some n => .building n

/-- `placed_in` on a relationship column is the place test `[frm, next)`: `notBefore frm` and `below`. -/
theorem placedIn_edge_spec (to : Ver) (ws : List Nat) (frm : BuildCursor) (next : Option BuildCursor) (id : Nat) :
    id ∈ placedIn .relationship to ws (some frm) next ↔
      id ∈ ws ∧ notBefore frm to id = true ∧ below (stOf next) to id = true := by
  simp only [placedIn, List.mem_filter, notBefore, stOf]
  cases next with
  | none => simp [below]
  | some n => simp only [below]; generalize to.place id = o; cases o <;> simp

/-- `placed_in` on a node column (cursors are node ids, a node's place is its own id) is the same. -/
theorem placedIn_node_spec (to : Ver) (ws : List Nat) (f : Nat) (next : Option Nat)
    (hk : ∀ id, to.place id = some (.node id)) (id : Nat) :
    id ∈ placedIn .node to ws (some (.node f)) (next.map BuildCursor.node) ↔
      id ∈ ws ∧ notBefore (.node f) to id = true ∧ below (stOf (next.map BuildCursor.node)) to id = true := by
  simp only [placedIn, List.mem_filter, notBefore, stOf, below, hk]
  cases next <;> simp [BuildCursor.precedes]

theorem indexerInsert_numeric (ix : NumericIndex) (e : EntityType) (published : Ver) (batch : EncodedTuples)
    (written : List Nat) (frm : BuildCursor) (next : Option BuildCursor) :
    ∃ ix', (IndexColumn.numeric ix).indexerInsert e published batch written frm next = .numeric ix' := ⟨_, rfl⟩

/-- **An insert keeps a column exact** (#3568, the `CatchUp::Insert` path, as `catchUp_inv` is for `Refresh`).
    `c` is the published copy's column, exact below the cursor `frm` the batch starts from; the batch holds
    exactly the tuples, as `batchVer` has them, of the entities `batchVer` places in `[frm, next)`; `written`
    covers every entity on which `batchVer` and `published` differ there (`Frame`); entity ids fit the tuple's id
    field and have a place. Then the column `indexer_insert` leaves holds exactly the tuples of the entities
    `published` puts in the column in the state the batch leaves (`Building next`, or `Ready`): the same
    conclusion `catchUp_inv` draws for the refresh path. -/
theorem insert_inv (c : IndexColumn) (e : EntityType) (batchVer published : Ver) (batch : EncodedTuples)
    (written : List Nat) (frm : BuildCursor) (next : Option BuildCursor)
    (hplaced : ∀ id, id ∈ placedIn e published written (some frm) next ↔
      id ∈ written ∧ notBefore frm published id = true ∧ below (stOf next) published id = true)
    (hinv : Inv c (.building frm) published)
    (hmono : ∀ id, below (.building frm) published id = true → below (stOf next) published id = true)
    (hbatch : ∀ t, t ∈ batch.scalar ↔ ∃ id, notBefore frm batchVer id = true ∧
      below (stOf next) batchVer id = true ∧ t ∈ tupl c batchVer id)
    (hf : ∀ id, id ∉ written → notBefore frm batchVer id = notBefore frm published id ∧
      below (stOf next) batchVer id = below (stOf next) published id ∧ tupl c batchVer id = tupl c published id)
    (hWv : ∀ id, tupl c batchVer id ≠ [] → id < W) (hWp : ∀ id, tupl c published id ≠ [] → id < W)
    (hplace : ∀ id, tupl c published id ≠ [] → (published.place id).isSome = true) :
    Inv (c.indexerInsert e published batch written frm next) (stOf next) published := by
  intro t
  have hkind : ∀ id, tupl (c.indexerInsert e published batch written frm next) published id =
      tupl c published id := by
    intro id
    cases c with
    | numeric ix =>
      obtain ⟨ix', h⟩ := indexerInsert_numeric ix e published batch written frm next
      rw [h]; exact tupl_numeric _ _ _ _
  simp only [Inv, hkind]
  rw [indexerInsert_mem]
  constructor
  · rintro (h | ⟨hb, hw⟩ | ⟨id, hid, ht⟩)
    · obtain ⟨id, hbl, hit⟩ := (hinv t).1 h
      exact ⟨id, hmono id hbl, hit⟩
    · obtain ⟨id, hnb, hbl, hit⟩ := (hbatch t).1 hb
      obtain ⟨k, hk⟩ := tupl_enc c batchVer id t hit
      have hlt := hWv id (by intro h; rw [h] at hit; simp at hit)
      have hd : docOf t = id := by rw [hk, docOf_enc _ _ hlt]
      have hnw : id ∉ written := hd ▸ hw
      have fr := hf id hnw
      exact ⟨id, by rw [← fr.2.1]; exact hbl, by rw [← fr.2.2]; exact hit⟩
    · exact ⟨id, ((hplaced id).1 hid).2.2, ht⟩
  · rintro ⟨id, hbl, ht⟩
    have hne : tupl c published id ≠ [] := by intro h; rw [h] at ht; simp at ht
    by_cases hbf : below (.building frm) published id = true
    · exact Or.inl ((hinv t).2 ⟨id, hbf, ht⟩)
    · have hnb : notBefore frm published id = true := by
        obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (hplace id hne)
        simp only [below, hpl, Option.any_some, Bool.not_eq_true] at hbf
        simp [notBefore, hpl, hbf]
      by_cases hw : id ∈ written
      · exact Or.inr (Or.inr ⟨id, (hplaced id).2 ⟨hw, hnb, hbl⟩, ht⟩)
      · refine Or.inr (Or.inl ?_)
        have fr := hf id hw
        have hlt := hWp id hne
        obtain ⟨k, hk⟩ := tupl_enc c published id t ht
        have hd : docOf t = id := by rw [hk, docOf_enc _ _ hlt]
        refine ⟨(hbatch t).2 ⟨id, by rw [fr.1]; exact hnb, by rw [fr.2.1]; exact hbl, by rw [fr.2.2]; exact ht⟩, ?_⟩
        rw [hd]; exact hw

/-- **Both catch-ups publish the same shape of entry.** When the batch fits (same generation, so
    `discardedAgainst` found nothing) and the published index is still `Building`, inserting leaves exactly
    the entry that refreshing does, with the column the only difference: `commit_catchUp` therefore covers
    `CatchUp::Insert` as well, with `f` the column `indexer_insert` builds. -/
theorem insertedOnto_eq_caughtUpOnto {R : Type} (ours pub : V R) (g : SE → SE → IndexColumn)
    (o p : SE) (ho : ours.entry = some o) (hp : pub.entry = some p)
    (hg : o.generation = p.generation) (hb : p.en.isBuilding = true) :
    V.insertedOnto ours pub g = V.caughtUpOnto ours pub g := by
  simp only [V.insertedOnto, V.caughtUpOnto, V.newBuildVersion, V.newVersion, ho, hp, hb, if_true]
  cases o with | mk oen og oc => cases oen; simp_all

/-! `CatchUp`, `REFRESH_COST` and `catchUpBy` (`versioned.rs:149-191`) are in `Batch.lean`, where `indexerFit`
chooses the way; `V.insertedOnto` (`Graph::insert_index`) is there too. -/

theorem catchUpBy_refresh_iff (written batch : Nat) :
    catchUpBy written batch = .refresh ↔ written * REFRESH_COST < batch := by
  unfold catchUpBy; split <;> simp_all

/-- The first writer-free case: nothing written since, a non-empty batch refreshes (nothing to refresh, so
    cheapest); an empty batch inserts itself (nothing to insert). -/
theorem catchUpBy_no_writes (batch : Nat) (h : 0 < batch) : catchUpBy 0 batch = .refresh := by
  simp [catchUpBy, REFRESH_COST, h]

theorem catchUpBy_empty_batch (written : Nat) : catchUpBy written 0 = .insert := by
  simp [catchUpBy]

/-- Many writes make the batch insert itself. -/
theorem catchUpBy_many_writes (written batch : Nat) (h : batch ≤ written * REFRESH_COST) :
    catchUpBy written batch = .insert := by
  simp [catchUpBy]; omega

/-- `written.saturating_mul(10)` on `u64` against a `usize` batch length (< 2^64): the saturated product
    compares like the exact one. -/
theorem catchUpBy_saturating (written batch : Nat) (hb : batch < W) :
    (min (written * REFRESH_COST) (W - 1) < batch) ↔ written * REFRESH_COST < batch := by
  have : (W : Nat) = 18446744073709551616 := rfl
  omega

/-! ## Entries as the fit sees them: the gap before #3567 and its closure -/

/-- Whether `ours` was overtaken, as `SE`s: `overtaken` on the projection. -/
def overtakenSE (ours theirs : SE) : Bool :=
  ours.en.isBuilding && theirs.generation == ours.generation && theirs.commits != ours.commits

theorem overtaken_proj (ours theirs : EN) : overtakenSE ours.proj theirs.proj = overtaken ours theirs := by
  cases ours with | mk i s g c => cases s <;> simp [overtakenSE, overtaken, EN.proj, EN.isBuilding, IndexEntry.isBuilding, BS.proj]

/-- The entry slot after `rebase_index` as `V` sees it (`f` = the refreshed column). -/
def rebasedSE (ours theirs : SE) (f : IndexColumn) : SE :=
  if overtakenSE ours theirs then ⟨⟨f, theirs.en.state⟩, theirs.generation, theirs.commits, theirs.since⟩ else ours

/-- `rebase_onto_indexer_batches` on a `V` (`versioned.rs:486-496`): only the batch's index slot changes; the
    data version and the rest of the graph are the fork's. -/
def V.rebased {R : Type} (fork pub : V R) (f : SE → SE → IndexColumn) : V R :=
  match fork.entry, pub.entry with
  | some o, some p => { fork with entry := some (rebasedSE o p (f o p)) }
  | _, _ => fork

/-- **A rebase changes no data**: the fork keeps its data version, so it stays current (`is_stale`), and
    everything else in it. -/
theorem V.rebased_keeps_data {R : Type} (fork pub : V R) (f : SE → SE → IndexColumn) :
    (fork.rebased pub f).version = fork.version ∧ (fork.rebased pub f).rest = fork.rest ∧
    (fork.rebased pub f).indexesCreated = fork.indexesCreated ∧ (fork.rebased pub f).batches = fork.batches := by
  unfold V.rebased; cases fork.entry <;> cases pub.entry <;> simp

theorem V.rebased_not_stale {R : Type} (fork pub : V R) (f : SE → SE → IndexColumn) (q : V R) :
    isStale q (fork.rebased pub f) = isStale q fork := by
  simp [isStale, (V.rebased_keeps_data fork pub f).1]

/-- **With the rebase, the writer publishes the batches' chain**: the rebased copy has the published copy's
    generation and commit count, so the chain continues. -/
theorem rebased_commit_keeps_chain (ours theirs : SE) (f : IndexColumn) (h : overtakenSE ours theirs = true) :
    (rebasedSE ours theirs f).generation = theirs.generation ∧ (rebasedSE ours theirs f).commits = theirs.commits ∧
    (rebasedSE ours theirs f).en.state = theirs.en.state := by
  simp [rebasedSE, h]

/-- **...and the next batch fits**: the writer commits `after := fork.rebased pub f` (derived here from the
    rebase itself, not assumed). A batch prepared on the published copy has that copy's generation and
    `p.commits + 1`; judged against the writer's rebased commit, which has `p.commits`, it is never discarded as
    gone or as another batch's: it is a `CatchUp`, the writer having moved the data version, by whichever way
    `catch_up_by` picks, unless that catch-up would do more than the batch's limit (`limit`, `none` at the floor),
    which discards it for its size alone (`CatchUpTooLarge`). The fit reads only the generation, the commit count
    and the `since` count, so the refreshed column `f` is irrelevant. -/
theorem rebased_batch_fits {R : Type} (work : EncodedTuples) (fork pub prepared : V R) (f : SE → SE → IndexColumn) (o p : SE)
    (ho : fork.entry = some o) (hp : pub.entry = some p) (hov : overtakenSE o p = true)
    (en : IndexEntry) (s : Nat) (hprep : prepared.entry = some ⟨en, p.generation, p.commits + 1, s⟩)
    (hv : (fork.rebased pub f).version ≠ prepared.version) (limit : Option Nat) (read : Nat) :
    (∃ how, indexerFit work prepared (fork.rebased pub f) limit read = .catchUp how) ∨
      indexerFit work prepared (fork.rebased pub f) limit read = .discarded .catchUpTooLarge := by
  have hafter : (fork.rebased pub f).entry = some (rebasedSE o p (f o p)) := by
    simp [V.rebased, ho, hp]
  have hk := rebased_commit_keeps_chain o p (f o p) hov
  have hd : discardedAgainst ⟨en, p.generation, p.commits + 1, s⟩ (some (rebasedSE o p (f o p))) = none := by
    simp [discardedAgainst, hk.1, hk.2.1]
  simp only [indexerFit, hprep, hafter, hd]
  rw [ite_eq_left hv]
  split
  · exact Or.inr rfl
  · exact Or.inl ⟨_, rfl⟩

/-! ## The smaller functions around them -/

/-- `BuildCursor::from_id` (`falkordb_index.rs:437-448`) and `Graph::build_place` (`graph.rs:3876`, which passes
    its `edge_endpoints`): a node by its id, an edge by its endpoints, `None` for an edge the version does not
    have. -/
def BuildCursor.fromId (e : EntityType) (id : Nat) (endpoints : Nat → Option (Nat × Nat)) : Option BuildCursor :=
  match e with
  | .node => some (.node id)
  | .relationship => (endpoints id).map (fun p => .edge p.1 p.2)

/-- It is the place the graph gives an entity (`G.place`, the `apply_index_writes` closure before #3567). -/
theorem fromId_spec (g : G) (e : EntityType) (id : Nat) : BuildCursor.fromId e id g.endpoints = g.place e id := by
  cases e <;> rfl

theorem fromId_kind (e : EntityType) (id : Nat) (endpoints : Nat → Option (Nat × Nat)) (c : BuildCursor)
    (h : BuildCursor.fromId e id endpoints = some c) : BuildCursor.OfKind e c := by
  cases e with
  | node => simp [BuildCursor.fromId] at h; subst h; trivial
  | relationship =>
    simp only [BuildCursor.fromId, Option.map_eq_some_iff] at h
    obtain ⟨_, _, rfl⟩ := h; trivial

/-- `BuildState::building` (`versioned.rs:55-61`): no entity written yet. -/
theorem BS.mk_spec (next : BuildCursor) : (BS.mk next).next = some next ∧ (BS.mk next).proj = .building next := ⟨rfl, rfl⟩

/-- `FalkorDbIndex::publish` (`falkordb_index.rs:716-722`): `IndexEntry::publish` on every column entry. -/
def publishAll (l : List (IndexKey × EN)) : List (IndexKey × EN) := l.map (fun p => (p.1, p.2.publish))

/-- Publishing changes no entry's projected content (column, cursor or `Ready`, generation, commits). -/
theorem publishAll_proj (l : List (IndexKey × EN)) :
    (publishAll l).map (fun p => (p.1, p.2.proj.en.index, p.2.proj.en.state, p.2.proj.generation, p.2.proj.commits)) =
    l.map (fun p => (p.1, p.2.proj.en.index, p.2.proj.en.state, p.2.proj.generation, p.2.proj.commits)) := by
  induction l with
  | nil => rfl
  | cons p ps ih =>
    simp only [publishAll, List.map_cons] at ih ⊢
    obtain ⟨h1, h2, h3⟩ := publish_proj p.2
    rw [ih, h1, h2, h3]

/-- `FalkorDbIndex::column_names` (`falkordb_index.rs:724-736`): every column, by kind and key. -/
def FalkorDbIndex.columnNames (x : FalkorDbIndex) : List (EntityType × IndexKey) :=
  [EntityType.node, EntityType.relationship].flatMap (fun e => (x.columns e).map (fun p => (e, p.1)))

theorem columnNames_mem (x : FalkorDbIndex) (e : EntityType) (k : IndexKey) :
    (e, k) ∈ x.columnNames ↔ k ∈ (x.columns e).map (·.1) := by
  cases e <;> simp [FalkorDbIndex.columnNames, List.mem_flatMap, List.mem_map]

/-- `Graph::rebase_onto_indexer_batches` (`versioned.rs:486-496`): `rebase_index` on every column of the fork;
    `pub k` is the published version's entry under `k`. A column with no published counterpart, or one no batch
    moved on, keeps the fork's entry. -/
def rebaseAll (fork : List (IndexKey × EN)) (pub : IndexKey → Option EN) (published forkV : Ver) :
    List (IndexKey × EN) :=
  fork.map (fun p => (p.1, match pub p.1 with
    | some t => p.2.rebase t published forkV
    | none => p.2))

theorem rebaseAll_keys (fork : List (IndexKey × EN)) (pub : IndexKey → Option EN) (published forkV : Ver) :
    (rebaseAll fork pub published forkV).map (·.1) = fork.map (·.1) := by
  simp [rebaseAll, List.map_map, Function.comp_def]

/-- Each column's slot after the rebase: its own if no published copy, else `rebase_index`'s. -/
theorem rebaseAll_entry (fork : List (IndexKey × EN)) (pub : IndexKey → Option EN) (published forkV : Ver)
    (k : IndexKey) (e : EN) (h : (k, e) ∈ fork) :
    (k, match pub k with | some t => e.rebase t published forkV | none => e) ∈ rebaseAll fork pub published forkV := by
  unfold rebaseAll
  exact List.mem_map.2 ⟨(k, e), h, rfl⟩

/-- `Graph::overtaken_by_indexer_batches` (`versioned.rs:498-512`): a batch was published since this version
    was made (`batch_published_since`) and one of its columns was overtaken. The batch count gate is the cheap
    test every escalating write pays. -/
def overtakenAny (batchPublished : Bool) (cols : List Bool) : Bool := batchPublished && cols.any id

theorem overtakenAny_iff (b : Bool) (cols : List Bool) :
    overtakenAny b cols = true ↔ b = true ∧ ∃ c ∈ cols, c = true := by
  simp [overtakenAny, List.any_eq_true]

/-- No batch published since: nothing to do, however the columns look. -/
theorem overtakenAny_no_batch (cols : List Bool) : overtakenAny false cols = false := rfl

/-- `MvccGraph::rebase_onto_indexer_batches` (`mvcc_graph.rs:166-188`), the decision: nothing overtaken is `Ok`
    with the fork untouched; a borrowed fork is a debug panic and a release `Err(Stale)`; otherwise the fork is
    rebased (`V.rebased`). -/
inductive RebaseOutcome where
  | untouched | rebased | stale | panic
deriving DecidableEq, Repr

def mvccRebase (anyOvertaken borrowed debug : Bool) : RebaseOutcome :=
  if !anyOvertaken then .untouched
  else if borrowed then (if debug then .panic else .stale)
  else .rebased

theorem mvccRebase_noop (borrowed debug : Bool) : mvccRebase false borrowed debug = .untouched := rfl

/-- An escalating writer's fork is not borrowed: the rebase always succeeds, changing the fork or not. -/
theorem mvccRebase_unborrowed (any debug : Bool) :
    mvccRebase any false debug = if any then .rebased else .untouched := by
  cases any <;> simp [mvccRebase]

theorem mvccRebase_borrowed_release : mvccRebase true true false = .stale := rfl
theorem mvccRebase_borrowed_debug : mvccRebase true true true = .panic := rfl

end NI
