import FalkorReplication.Guard
import FalkorReplication.Memo
import FalkorReplication.Faithful
/-
# Effects v3: emit and apply agree, so master and replica cannot silently diverge

A model of FalkorDB's `GRAPH.EFFECT` replication and a machine-checked proof of
two things:

* **`emit_apply`** — the payload `build` produces from a `Pending`, replayed by
  `apply` on the state `commit` ran against, *succeeds* and lands on *exactly*
  the state `commit` produced.
* **`replica_never_serves_a_different_graph`** — therefore no stream of writes
  reaches a state where the replica is serving a graph that differs from its
  master's.

The first is the load-bearing one, and it is a theorem rather than a definition
because `commit` and `emit` are two separate functions here, over one shared
`Pending`, exactly as they are in the engine. The interesting content is that
`emit` does not ship the mutations in the order `commit` applied them: it
partitions the created nodes by shape and the deleted ones by label set, so the
replica rebuilds the same graph from a different ordering. `createBatch_perm`
and `deleteBatch_perm` are what say that is sound, and `specsOf_group` is what
says a record decodes back to exactly the group it came from — that row *k*
belongs to the k-th id in the `IdList` and nothing is dropped, duplicated or
transposed.

Since `ed5b31553` ("feat(effects)!: emit, apply, and effects as the only
replication mechanism") there is no query replay, which is what makes any of
this provable: the replica never re-evaluates anything, so there is no `rand()`,
no clock and no plan choice for the two sides to disagree about.

## What is modelled, and where it lives in the tree

| here | there |
| --- | --- |
| `intern`                 | `Graph::get_label_id_mut`, `get_attribute_id_mut` (`graph/src/graph/graph.rs`) — append-only, id = index |
| `GState.bound`           | `Graph::node_id_bound` (`graph.rs:1585`) = `IdSpace::bound` = `live + recycled.len()` (`id_space.rs:343`, #2846) — `absIds`, `createOne_refines` |
| `GState.free`            | the recycle bin (a `RoaringTreemap`, hence a set — modelled as a predicate) |
| `createOne` / `deleteOne`| `Graph::create_nodes` / `delete_nodes`, one node at a time |
| `idsOf`                  | `IdSpace::reserve` (`graph/src/graph/id_space.rs:483`) — recycled first, then fresh |
| `Pending`                | `runtime::pending::Pending` |
| `commit`                 | `Pending::commit` (`runtime/pending.rs:1099`), driven by `CommitOp::next` (`runtime/ops/commit.rs:69`) |
| `emit`                   | `EffectsFormat::build` → `for_each_record` (`graph/src/effects/v3/emit.rs`) |
| `groupBy NodeSpec.shape` | `digest_created_nodes`' `slots`/`index`/`last` memo |
| `groupBy Prod.snd`       | `digest_deleted_nodes`, grouped by label set |
| `gatherRows`             | `gather_rows` — row-major, one row per id, in `IdList` order |
| `schemaRecs`             | `emit_schema_additions`, against the `SchemaBaseline` |
| `Record`                 | `effects::v3::Record` (`graph/src/effects/v3/records.rs`) |
| `applyRec` / `applyRecs` | `apply_effects` (`graph/src/effects/v3/apply.rs`) |
| `okCreate`               | `verify_schema`, `verify_attribute`, the refusals of `IdSpace::create` (`id_space.rs:582`) |
| `okDelete`               | the refusals of `IdSpace::release` (`:703`): `refuse_not_live` (`:679`) = `refuse_recycled`, `refuse_undeletable` |
| `createOne` / `deleteOne` id-space half | `IdSpace::create` / `IdSpace::release` on one id: `createOne_refines`, `deleteOne_refines` (`Faithful.Ids`) |
| `addLabel`'s id check    | `apply_add_schema` / `verify_id` — the replica re-derives the id and compares |
| `Batch.verify`           | `IdSpace::verify` (`:737`) via `Graph::validate` — `taken ∩ [entry, ∞) == [entry_bound, entry_bound + created)`; since #2846 the batch lives on the graph, opened by `Graph::new_version` at `bound` (= `{ entry := σ.bound, created := [] }`) |
| `PendingWF.disjoint`     | why `digest_cancelled` exists: a node created and deleted in one segment is unwound out of `Pending` and travels as its own create/delete pair |
| `Replica.resync`         | `divergence_guard::on_failure` → `REPLICAOF NO ONE` + `REPLICAOF <master>` |
| `Replica.halted`         | the same guard under `LOADING` → `std::process::exit(1)` |
| `Guard.*` (`Guard.lean`) | every fn of `src/divergence_guard.rs` line by line: `onFailure_client_noop`, `onFailure_loading_exits`, `onFailure_replicated_resyncs`, `forceFullResync_exit_iff`, `masterAddress_some`, `clip_spec`; `guardReplica_spec`, `stepNode_failure_matches_guard` tie them to `stepNode` |
| refinement (wave 4)      | `intern_refines`, `applyRec_addLabel_refines`/`_addAttribute_refines` (= `apply_add_schema`), `idsOf_refines` (= `IdSpace::reserve` on a fresh batch), `memo_refines_groupBy`/`emit_refines` (= the `slots`/`index`/`last` loops, `Memo.lean`) |

## Three mechanisms the proof turns on

1. **Values travel as literals.** A `Record` carries `Val`, never an expression,
   and `applyRec` has no evaluator. Determinism is not a theorem about the
   replica, it is the shape of `applyRec`'s signature: state and record in,
   state out, no oracle.
2. **Entity ids are transmitted and then *validated*.** The replica is told which
   ids to create, so it never allocates; `Batch.verify` is what stops it
   accepting an id space its master does not have.
3. **Schema ids are *re-derived* and then compared.** `applyRec` interns the name
   locally and refuses the buffer unless its own answer matches the master's —
   the one place the two dictionaries are checked against each other.

## A note on the boundary

`createOne` raises `bound` once per id that was **not** in the recycle bin, not
once per id at or above the old boundary. Those agree for one contiguous batch
and diverge the moment records are grouped by shape, since a later record can
carry an id below the boundary an earlier one pushed up. The engine gets this
right by construction — its boundary is `live + recycled.len()`, so a
recycled id moves an entry from the bin to the count and leaves the boundary
alone. A first draft of this model did not, and `createBatch_bound` is where it
failed to compile.

## What is proved

* `specsOf_group`, `chunk_flatMap`, `lookup_row` — a record decodes back to its
  group, rows and ids aligned.
* `createBatch_perm`, `deleteBatch_perm` — regrouping is sound.
* `emit_apply` — emit and apply agree, per payload.
* `stream_simulation` — the same over any sequence of writes.
* `step_preserves` / `no_divergence` /
  `replica_never_serves_a_different_graph` — over the guarded state machine,
  including the failure branch: every reachable state has the replica either
  serving exactly the master's graph, resyncing, or halted.
* `no_spurious_resync` — the guard does not fire on a healthy pair, so the
  checks are not bought by resyncing at the first difficulty.
* The `#guard`s at the bottom are executable and non-vacuous: a replica that
  missed a payload, or that numbered its schema differently, is **refused**.

No `sorry`: the `#print axioms` lines at the end pin every headline theorem to
Lean's three standard axioms.

## What is still assumed

1. **The bytes.** This model pivots on `Record`, exactly as the engine does
   (`emit`: `Pending → Record → bytes`; `apply`: `bytes → Record → Graph`).
   `EffectEncode`/`EffectDecode` and the `IdList` segment encoding — `Range`,
   `Repeat`, the ascending roaring bitmap — are **not** modelled. That codec is
   where a wrong width or a reordering encoder would land, and proving
   `decode (encode r) = r` is the obvious next piece.
2. **Coverage.** `Record` here covers schema registration, node create and node
   delete. Edges, `UPDATE_NODE`/`UPDATE_EDGE`, `SET_LABELS`/`REMOVE_LABELS`,
   the index and constraint DDL, and `digest_cancelled`'s create/delete pair are
   not modelled; `PendingWF.disjoint` assumes the last of those rather than
   deriving it.
   #3022 (`18fc277b9`) added liveness checks to the apply arms: on the arms
   modelled here they change nothing — `DELETE_NODE`'s `require_live` is the
   same `refuse_not_live` that `release` (= `okDelete`) already ran, and its
   `NodeHasRelationships` refusal is vacuous in an edge-free graph — so every
   theorem stands as is. The checked edge/update/label arms are proved in
   `proofs/effects_emit_apply` (`Liveness.lean`, `apply_wf`).
3. **Transport.** `stepsNode` folds the payloads in order; Redis replication
   supplies that ordering, and `GRAPH.EFFECT` supplies the atomicity by rolling
   the graph back on any error.
4. **The master's own batch check.** Since #2846 the master verifies its id
   space too (`Pending::end_segment` → `Graph::roll_id_batches`, and
   `MvccGraph::commit` → `Graph::validate`); `commit` here is total. That the
   master's check never refuses a well-formed write is not proven here (the
   replica-side analogue is the `Batch.verify` half of `emit_apply`; the
   id-space half is `proofs/id_space` `roll_inv`/`lifecycle_fresh`).
   `PendingWF.inLimit` (new with #2911, `fe619ac5f`): every id the master committed is below
   `ID_LIMIT`, because the master's `IdSpace::create` refuses the rest (`proofs/id_space`
   `create_out_of_range`); the replica's `okCreate` now checks the same bound, so a
   `GRAPH.EFFECT` naming 2^61 is refused instead of sizing a matrix to it (#2892).
5. **Every modifying write produces a payload.** `commit_and_replicate` logs a
   warning and returns when `wq.modified` is set but `effects_buffer` is `None`;
   the `build_effects` gate is `REPLICATION_CONSUMERS || AOF`, evaluated when the
   query starts. A replica that attaches mid-query is outside this model and is
   covered by its full sync.
6. **That the Lean `emit`/`applyRec` are faithful to the Rust.** This is a
   correspondence to review, clause by clause, against the table above — not
   something the proof can establish.

## Checking it

    cd proofs/replication && lake build

-/

namespace Falkor

abbrev Name := String

/-! ## Values -/

inductive Val where
  | null
  | num (n : Nat)
deriving Repr, DecidableEq, BEq

/-! ## Bool-valued nodup -/

def nodupB [BEq α] : List α → Bool
  | []      => true
  | a :: as => !as.contains a && nodupB as

theorem nodupB_iff {α} [BEq α] [LawfulBEq α] : ∀ {l : List α}, nodupB l = true ↔ l.Nodup := by
  intro l
  induction l with
  | nil => simp [nodupB]
  | cons a as ih => simp [nodupB, List.nodup_cons, ih]

/-! ## Schema dictionaries

    `Graph::get_label_id_mut` / `get_attribute_id_mut`: append-only, an entry's
    id is its index. -/

def idxOf : List Name → Name → Nat → Option Nat
  | [],      _, _ => none
  | m :: ms, n, i => if m = n then some i else idxOf ms n (i + 1)

def intern (d : List Name) (n : Name) : Nat × List Name :=
  match idxOf d n 0 with
  | some i => (i, d)
  | none   => (d.length, d ++ [n])

theorem idxOf_lt : ∀ (d : List Name) (n : Name) (k i : Nat),
    idxOf d n k = some i → i < k + d.length := by
  intro d
  induction d with
  | nil => intro n k i h; simp [idxOf] at h
  | cons m ms ih =>
    intro n k i h
    simp only [idxOf] at h
    by_cases hm : m = n
    · simp only [hm, ↓reduceIte, Option.some.injEq] at h
      simp only [List.length_cons]; omega
    · simp only [hm, ↓reduceIte] at h
      have := ih n (k + 1) i h
      simp only [List.length_cons]; omega

theorem idxOf_none : ∀ (d : List Name) (n : Name) (k : Nat), n ∉ d → idxOf d n k = none := by
  intro d
  induction d with
  | nil => intro n k _; rfl
  | cons m ms ih =>
    intro n k hn
    simp only [List.mem_cons, not_or] at hn
    have hne : ¬ (m = n) := fun he => hn.1 he.symm
    simp only [idxOf, hne, ↓reduceIte]
    exact ih n (k + 1) hn.2

theorem intern_fresh {d : List Name} {n : Name} (h : n ∉ d) :
    intern d n = (d.length, d ++ [n]) := by
  simp only [intern, idxOf_none d n 0 h]

theorem intern_id_lt (d : List Name) (n : Name) : (intern d n).1 < (intern d n).2.length := by
  unfold intern
  split
  · rename_i i h; show i < d.length; have := idxOf_lt d n 0 i h; omega
  · rename_i _; show d.length < (d ++ [n]).length; simp

/-! ## Chunking a row-major value block

    `Record::CreateNode.rows` is `count × attrs_per_row` values, flat, with no
    per-row id: row *k* belongs to the k-th id in the record's `IdList`. The
    count comes from the record header, which is the `fuel` here. -/

def chunk : Nat → Nat → List Val → List (List Val)
  | 0,     _, _ => []
  | k + 1, n, l => l.take n :: chunk k n (l.drop n)

theorem chunk_flatMap {α} (f : α → List Val) (m : Nat) :
    ∀ (g : List α), (∀ s ∈ g, (f s).length = m) →
      chunk g.length m (g.flatMap f) = g.map f := by
  intro g
  induction g with
  | nil => intro _; rfl
  | cons s rest ih =>
    intro hlen
    have hs : (f s).length = m := hlen s (List.mem_cons_self ..)
    simp only [List.length_cons, chunk, List.flatMap_cons, List.map_cons]
    rw [← hs, List.take_left, List.drop_left, hs]
    exact congrArg _ (ih (fun x hx => hlen x (List.mem_cons_of_mem _ hx)))

/-! ## Attribute lookup -/

def lookupAttr : List (Nat × Val) → Nat → Val
  | [],           _ => .null
  | (k, v) :: as, a => if k = a then v else lookupAttr as a

theorem lookup_row (as : List (Nat × Val)) (h : (as.map Prod.fst).Nodup) :
    (as.map Prod.fst).map (lookupAttr as) = as.map Prod.snd := by
  induction as with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    simp only [List.map_cons, List.nodup_cons] at h
    simp only [List.map_cons]
    have head : lookupAttr ((k, v) :: rest) k = v := by simp [lookupAttr]
    have tail : (rest.map Prod.fst).map (lookupAttr ((k, v) :: rest))
              = (rest.map Prod.fst).map (lookupAttr rest) := by
      refine List.map_congr_left ?_
      intro a ha
      have hne : ¬ (k = a) := fun he => h.1 (by rw [he]; exact ha)
      simp [lookupAttr, hne]
    rw [head, tail, ih h.2]

/-! ## Graph state

    `free` is the recycle bin as a membership predicate rather than a list: the
    bin is a `RoaringTreemap`, so it is a set, and `IdSpace::reserve` takes its
    *lowest* ids. Modelling it as a predicate makes the bin canonical, which is
    what lets a regrouped batch of deletes be proved to land in the same place.

    `bound` is `Graph::node_id_bound` = `IdSpace::bound` = `live + recycled.len()`. Note
    what that makes a create: an id taken from the bin moves one entry from the
    bin to the count and leaves the boundary alone; a fresh id raises it by one.
    The boundary does **not** jump to `max(ids) + 1` — a create that skipped ids
    leaves a hole, which is exactly what `IdSpace::verify` is there to catch. -/

structure GState where
  labels     : List Name
  attrs      : List Name
  bound      : Nat
  free       : Nat → Bool
  nodeLabels : Nat → List Nat
  props      : Nat → Nat → Val

def GState.liveB (σ : GState) (i : Nat) : Bool := decide (i < σ.bound) && !σ.free i

/-- The recycle bin in ascending order — what `reclaim_ids` walks. -/
def GState.bin (σ : GState) : List Nat := (List.range σ.bound).filter σ.free

structure WF (σ : GState) : Prop where
  binOk     : ∀ i, σ.free i = true → i < σ.bound
  labNodup  : σ.labels.Nodup
  attrNodup : σ.attrs.Nodup

/-! ## One node in, one node out -/

structure NodeSpec where
  id     : Nat
  labels : List Nat
  attrs  : List (Nat × Val)
deriving Repr, DecidableEq

/-- `Graph::create_nodes`, for one node. -/
def createOne (σ : GState) (id : Nat) (lbls : List Nat) (row : List (Nat × Val)) : GState :=
  { σ with
      bound      := if σ.free id then σ.bound else σ.bound + 1
      free       := fun i => if i = id then false else σ.free i
      nodeLabels := fun i => if i = id then lbls else σ.nodeLabels i
      props      := fun i a => if i = id then lookupAttr row a else σ.props i a }

/-- `Graph::delete_nodes`, for one node: the id goes back to the bin and the
    boundary does not move (`live` down one, the bin up one: `IdSpace::release`). -/
def deleteOne (σ : GState) (id : Nat) : GState :=
  { σ with
      free       := fun i => if i = id then true else σ.free i
      nodeLabels := fun i => if i = id then [] else σ.nodeLabels i
      props      := fun i a => if i = id then Val.null else σ.props i a }

def createBatch (σ : GState) : List NodeSpec → GState
  | []      => σ
  | s :: rest => createBatch (createOne σ s.id s.labels s.attrs) rest

def deleteBatch (σ : GState) : List Nat → GState
  | []      => σ
  | i :: rest => deleteBatch (deleteOne σ i) rest

theorem createBatch_append (σ : GState) (l₁ l₂ : List NodeSpec) :
    createBatch σ (l₁ ++ l₂) = createBatch (createBatch σ l₁) l₂ := by
  induction l₁ generalizing σ with
  | nil => rfl
  | cons s rest ih => simp only [List.cons_append, createBatch, ih]

theorem deleteBatch_append (σ : GState) (l₁ l₂ : List Nat) :
    deleteBatch σ (l₁ ++ l₂) = deleteBatch (deleteBatch σ l₁) l₂ := by
  induction l₁ generalizing σ with
  | nil => rfl
  | cons i rest ih => simp only [List.cons_append, deleteBatch, ih]

theorem GState.ext' {σ τ : GState} (h1 : σ.labels = τ.labels) (h2 : σ.attrs = τ.attrs)
    (h3 : σ.bound = τ.bound) (h4 : ∀ i, σ.free i = τ.free i)
    (h5 : ∀ i, σ.nodeLabels i = τ.nodeLabels i) (h6 : ∀ i a, σ.props i a = τ.props i a) :
    σ = τ := by
  cases σ; cases τ
  simp only [GState.mk.injEq]
  exact ⟨h1, h2, h3, funext h4, funext h5, funext (fun i => funext (h6 i))⟩

/-! ### What a batch of creates does, in closed form -/

theorem createBatch_dicts (σ : GState) (specs : List NodeSpec) :
    (createBatch σ specs).labels = σ.labels ∧ (createBatch σ specs).attrs = σ.attrs := by
  induction specs generalizing σ with
  | nil => exact ⟨rfl, rfl⟩
  | cons s rest ih => exact ih (createOne σ s.id s.labels s.attrs)

theorem createBatch_free (σ : GState) (specs : List NodeSpec) (i : Nat) :
    (createBatch σ specs).free i
      = (if specs.any (fun s => s.id == i) then false else σ.free i) := by
  induction specs generalizing σ with
  | nil => simp [createBatch]
  | cons s rest ih =>
    simp only [createBatch, ih, List.any_cons, Bool.or_eq_true, beq_iff_eq]
    by_cases hr : rest.any (fun r => r.id == i) = true
    · simp [hr]
    · by_cases hs : s.id = i
      · simp [hs, hr, createOne]
      · have hne : ¬ (i = s.id) := fun h => hs h.symm
        simp [hs, hr, hne, createOne]

theorem createBatch_bound (σ : GState) : ∀ (specs : List NodeSpec),
    (specs.map NodeSpec.id).Nodup →
    (createBatch σ specs).bound = σ.bound + (specs.filter (fun s => !σ.free s.id)).length := by
  intro specs
  induction specs generalizing σ with
  | nil => intro _; simp [createBatch]
  | cons s rest ih =>
    intro hnd
    simp only [List.map_cons, List.nodup_cons] at hnd
    have hne : ∀ r ∈ rest, r.id ≠ s.id := by
      intro r hr he; exact hnd.1 (he ▸ List.mem_map_of_mem hr)
    have hfilter : rest.filter (fun r => !(createOne σ s.id s.labels s.attrs).free r.id)
                 = rest.filter (fun r => !σ.free r.id) := by
      refine List.filter_congr ?_
      intro r hr
      simp [createOne, hne r hr]
    rw [createBatch, ih _ hnd.2, hfilter, List.filter_cons]
    simp only [createOne]
    by_cases hf : σ.free s.id = true <;> simp [hf] <;> omega

/-- Nothing in the batch touches an id the batch does not name. -/
theorem createBatch_miss (σ : GState) : ∀ (specs : List NodeSpec) (i : Nat),
    (∀ s ∈ specs, s.id ≠ i) →
    (createBatch σ specs).nodeLabels i = σ.nodeLabels i
    ∧ ∀ a, (createBatch σ specs).props i a = σ.props i a := by
  intro specs
  induction specs generalizing σ with
  | nil => intro i _; exact ⟨rfl, fun _ => rfl⟩
  | cons t rest ih =>
    intro i h
    have ht : ¬ (i = t.id) := fun he => (h t (List.mem_cons_self ..)) he.symm
    have hrest := ih (createOne σ t.id t.labels t.attrs) i
                     (fun r hr => h r (List.mem_cons_of_mem _ hr))
    rw [createBatch]
    exact ⟨by rw [hrest.1]; simp [createOne, ht],
           fun a => by rw [hrest.2]; simp [createOne, ht]⟩

/-- Every node the batch names ends up with exactly its own labels and values —
    this is `gather_rows`' contract, that row *k* belongs to id *k*. -/
theorem createBatch_hit (σ : GState) : ∀ (specs : List NodeSpec),
    (specs.map NodeSpec.id).Nodup → ∀ s ∈ specs,
    (createBatch σ specs).nodeLabels s.id = s.labels
    ∧ ∀ a, (createBatch σ specs).props s.id a = lookupAttr s.attrs a := by
  intro specs
  induction specs generalizing σ with
  | nil => intro _ s hs; exact absurd hs List.not_mem_nil
  | cons t rest ih =>
    intro hnd s hs
    simp only [List.map_cons, List.nodup_cons] at hnd
    have hne : ∀ r ∈ rest, r.id ≠ t.id := by
      intro r hr he; exact hnd.1 (he ▸ List.mem_map_of_mem hr)
    rcases List.mem_cons.mp hs with he | hr
    · subst he
      have hmiss := createBatch_miss (createOne σ s.id s.labels s.attrs) rest s.id hne
      exact ⟨by rw [createBatch, hmiss.1]; simp [createOne],
             fun a => by rw [createBatch, (hmiss.2 a)]; simp [createOne]⟩
    · exact ih _ hnd.2 s hr

/-! ### A batch of deletes, in closed form -/

theorem deleteBatch_simple (σ : GState) : ∀ (ids : List Nat),
    (deleteBatch σ ids).labels = σ.labels
    ∧ (deleteBatch σ ids).attrs = σ.attrs
    ∧ (deleteBatch σ ids).bound = σ.bound
    ∧ (∀ i, (deleteBatch σ ids).free i = (if i ∈ ids then true else σ.free i))
    ∧ (∀ i, (deleteBatch σ ids).nodeLabels i = (if i ∈ ids then [] else σ.nodeLabels i))
    ∧ (∀ i a, (deleteBatch σ ids).props i a = (if i ∈ ids then Val.null else σ.props i a)) := by
  intro ids
  induction ids generalizing σ with
  | nil => exact ⟨rfl, rfl, rfl, fun _ => rfl, fun _ => rfl, fun _ _ => rfl⟩
  | cons j rest ih =>
    obtain ⟨d1, d2, d3, d4, d5, d6⟩ := ih (deleteOne σ j)
    refine ⟨d1, d2, d3, ?_, ?_, ?_⟩
    · intro i
      show (deleteBatch (deleteOne σ j) rest).free i = _
      rw [d4]; by_cases hj : i = j <;> simp [deleteOne, hj, List.mem_cons]
    · intro i
      show (deleteBatch (deleteOne σ j) rest).nodeLabels i = _
      rw [d5]; by_cases hj : i = j <;> simp [deleteOne, hj, List.mem_cons]
    · intro i a
      show (deleteBatch (deleteOne σ j) rest).props i a = _
      rw [d6]; by_cases hj : i = j <;> simp [deleteOne, hj, List.mem_cons]

/-! ### Regrouping is sound

    `digest_created_nodes` partitions the created nodes by shape and
    `digest_deleted_nodes` partitions the deleted ones by label set, so the
    replica is handed them in a different order from the one the master applied
    them in. These say that does not matter. -/

theorem createBatch_perm (σ : GState) {X Y : List NodeSpec} (hp : X.Perm Y)
    (hX : (X.map NodeSpec.id).Nodup) : createBatch σ X = createBatch σ Y := by
  have hY : (Y.map NodeSpec.id).Nodup := (hp.map NodeSpec.id).nodup hX
  have hany : ∀ (f : NodeSpec → Bool), X.any f = Y.any f := by
    intro f
    by_cases h : X.any f = true
    · obtain ⟨x, hx, hfx⟩ := List.any_eq_true.mp h
      rw [h, List.any_eq_true.mpr ⟨x, hp.mem_iff.mp hx, hfx⟩]
    · simp only [Bool.not_eq_true] at h
      by_cases h2 : Y.any f = true
      · obtain ⟨y, hy, hfy⟩ := List.any_eq_true.mp h2
        exact absurd (List.any_eq_true.mpr ⟨y, hp.mem_iff.mpr hy, hfy⟩) (by simp [h])
      · simp only [Bool.not_eq_true] at h2; rw [h, h2]
  refine GState.ext' ?_ ?_ ?_ ?_ ?_ ?_
  · rw [(createBatch_dicts σ X).1, (createBatch_dicts σ Y).1]
  · rw [(createBatch_dicts σ X).2, (createBatch_dicts σ Y).2]
  · rw [createBatch_bound σ X hX, createBatch_bound σ Y hY, (hp.filter _).length_eq]
  · intro i; rw [createBatch_free, createBatch_free, hany]
  · intro i
    by_cases h : ∃ s, s ∈ X ∧ s.id = i
    · obtain ⟨s, hs, he⟩ := h
      subst he
      rw [(createBatch_hit σ X hX s hs).1, (createBatch_hit σ Y hY s (hp.mem_iff.mp hs)).1]
    · have h' : ∀ s ∈ X, s.id ≠ i := fun s hs he => h ⟨s, hs, he⟩
      rw [(createBatch_miss σ X i h').1,
          (createBatch_miss σ Y i (fun s hs => h' s (hp.mem_iff.mpr hs))).1]
  · intro i a
    by_cases h : ∃ s, s ∈ X ∧ s.id = i
    · obtain ⟨s, hs, he⟩ := h
      subst he
      rw [(createBatch_hit σ X hX s hs).2 a, (createBatch_hit σ Y hY s (hp.mem_iff.mp hs)).2 a]
    · have h' : ∀ s ∈ X, s.id ≠ i := fun s hs he => h ⟨s, hs, he⟩
      rw [(createBatch_miss σ X i h').2 a,
          (createBatch_miss σ Y i (fun s hs => h' s (hp.mem_iff.mpr hs))).2 a]

theorem deleteBatch_perm (σ : GState) {X Y : List Nat} (hp : X.Perm Y) :
    deleteBatch σ X = deleteBatch σ Y := by
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ := deleteBatch_simple σ X
  obtain ⟨b1, b2, b3, b4, b5, b6⟩ := deleteBatch_simple σ Y
  refine GState.ext' (by rw [a1, b1]) (by rw [a2, b2]) (by rw [a3, b3]) ?_ ?_ ?_
  · intro i; rw [a4, b4]
    by_cases h : i ∈ X
    · simp [h, hp.mem_iff.mp h]
    · have h2 : i ∉ Y := fun hy => h (hp.mem_iff.mpr hy)
      simp [h, h2]
  · intro i; rw [a5, b5]
    by_cases h : i ∈ X
    · simp [h, hp.mem_iff.mp h]
    · have h2 : i ∉ Y := fun hy => h (hp.mem_iff.mpr hy)
      simp [h, h2]
  · intro i a; rw [a6, b6]
    by_cases h : i ∈ X
    · simp [h, hp.mem_iff.mp h]
    · have h2 : i ∉ Y := fun hy => h (hp.mem_iff.mpr hy)
      simp [h, h2]

/-! ## Pending, and what the master does with it

    `Pending` is the mutation accumulator a write query fills. The master calls
    `commit` on it and then `build` (here `emit`) on the *same* value. The point
    of separating them is that their agreement becomes a theorem rather than a
    definition. -/

structure Pending where
  /-- `created_nodes` with `set_labels` and `new_nodes_attrs` folded in. -/
  created   : List NodeSpec
  /-- `deleted_nodes` paired with `deleted_node_labels`. -/
  deleted   : List (Nat × List Nat)
  /-- Registered since the `SchemaBaseline`, in id order. -/
  newLabels : List Name
  newAttrs  : List Name

def Pending.ids (p : Pending) : List Nat := p.created.map NodeSpec.id
def Pending.dels (p : Pending) : List Nat := p.deleted.map Prod.fst

/-- `Pending::commit`: register the new schema entries, create, then delete. -/
def commit (p : Pending) (σ : GState) : GState :=
  deleteBatch
    (createBatch { σ with labels := σ.labels ++ p.newLabels,
                          attrs := σ.attrs ++ p.newAttrs } p.created)
    p.dels

/-! ## Grouping

    `digest_created_nodes` keeps a slot per shape in first-appearance order and
    appends each node to its slot — the `slots`/`index`/`last` memo is an
    optimisation over exactly this. -/

def addTo {K} [DecidableEq K] {A} (k : K) (x : A) : List (K × List A) → List (K × List A)
  | []             => [(k, [x])]
  | (k', g) :: rest => if k' = k then (k', g ++ [x]) :: rest else (k', g) :: addTo k x rest

def groupAux {K} [DecidableEq K] {A} (key : A → K) :
    List A → List (K × List A) → List (K × List A)
  | [],      acc => acc
  | x :: rest, acc => groupAux key rest (addTo (key x) x acc)

def groupBy {K} [DecidableEq K] {A} (key : A → K) (l : List A) : List (K × List A) :=
  groupAux key l []

theorem addTo_cons {K} [DecidableEq K] {A} (k k' : K) (x : A) (g : List A)
    (rest : List (K × List A)) :
    addTo k x ((k', g) :: rest)
      = if k' = k then (k', g ++ [x]) :: rest else (k', g) :: addTo k x rest := rfl

theorem addTo_cons_hit {K} [DecidableEq K] {A} (k : K) (x : A) (g : List A)
    (rest : List (K × List A)) : addTo k x ((k, g) :: rest) = (k, g ++ [x]) :: rest := by
  rw [addTo_cons]; simp

theorem addTo_cons_miss {K} [DecidableEq K] {A} (k k' : K) (x : A) (g : List A)
    (rest : List (K × List A)) (h : k' ≠ k) :
    addTo k x ((k', g) :: rest) = (k', g) :: addTo k x rest := by
  rw [addTo_cons]; simp [h]

theorem addTo_flat {K} [DecidableEq K] {A} (k : K) (x : A) :
    ∀ (acc : List (K × List A)),
      ((addTo k x acc).flatMap Prod.snd).Perm ((acc.flatMap Prod.snd) ++ [x]) := by
  intro acc
  induction acc with
  | nil => exact List.Perm.refl _
  | cons hd rest ih =>
    obtain ⟨k', g⟩ := hd
    by_cases hk : k' = k
    · simp only [addTo, hk, ↓reduceIte, List.flatMap_cons, List.append_assoc]
      exact List.Perm.append_left g (List.perm_append_comm)
    · simp only [addTo, hk, ↓reduceIte, List.flatMap_cons, List.append_assoc]
      exact List.Perm.append_left g ih

theorem groupAux_flat {K} [DecidableEq K] {A} (key : A → K) :
    ∀ (l : List A) (acc : List (K × List A)),
      ((groupAux key l acc).flatMap Prod.snd).Perm ((acc.flatMap Prod.snd) ++ l) := by
  intro l
  induction l with
  | nil => intro acc; simp only [groupAux, List.append_nil]; exact List.Perm.refl _
  | cons x rest ih =>
    intro acc
    refine ((ih (addTo (key x) x acc)).trans ?_)
    have h1 : ((addTo (key x) x acc).flatMap Prod.snd ++ rest).Perm
              ((acc.flatMap Prod.snd ++ [x]) ++ rest) :=
      List.Perm.append_right rest (addTo_flat (key x) x acc)
    refine h1.trans ?_
    simp only [List.append_assoc, List.cons_append, List.nil_append]
    exact List.Perm.refl _

theorem groupBy_flat {K} [DecidableEq K] {A} (key : A → K) (l : List A) :
    ((groupBy key l).flatMap Prod.snd).Perm l := by
  show ((groupAux key l []).flatMap Prod.snd).Perm l
  have h := groupAux_flat key l []
  simp only [List.flatMap_nil, List.nil_append] at h
  exact h

/-- Everything in a slot really does carry that slot's key. -/
theorem addTo_keyed {K} [DecidableEq K] {A} (key : A → K) (x : A) :
    ∀ (acc : List (K × List A)),
      (∀ q ∈ acc, ∀ y ∈ q.2, key y = q.1) →
      ∀ q ∈ addTo (key x) x acc, ∀ y ∈ q.2, key y = q.1 := by
  intro acc
  induction acc with
  | nil => intro _ q hq y hy; simp only [addTo, List.mem_singleton] at hq; subst hq
           simp only [List.mem_singleton] at hy; subst hy; rfl
  | cons hd rest ih =>
    obtain ⟨k', g⟩ := hd
    intro hinv q hq y hy
    by_cases hk : k' = key x
    · subst hk
      rw [addTo_cons_hit] at hq
      rcases List.mem_cons.mp hq with he | hr
      · subst he
        rcases List.mem_append.mp hy with h1 | h2
        · exact hinv _ (List.mem_cons_self ..) y h1
        · simp only [List.mem_singleton] at h2; subst h2; rfl
      · exact hinv q (List.mem_cons_of_mem _ hr) y hy
    · rw [addTo_cons_miss _ _ _ _ _ hk] at hq
      rcases List.mem_cons.mp hq with he | hr
      · subst he; exact hinv (k', g) (List.mem_cons_self ..) y hy
      · exact ih (fun z hz => hinv z (List.mem_cons_of_mem _ hz)) q hr y hy

theorem groupAux_keyed {K} [DecidableEq K] {A} (key : A → K) :
    ∀ (l : List A) (acc : List (K × List A)),
      (∀ q ∈ acc, ∀ y ∈ q.2, key y = q.1) →
      ∀ q ∈ groupAux key l acc, ∀ y ∈ q.2, key y = q.1 := by
  intro l
  induction l with
  | nil => intro acc h; exact h
  | cons x rest ih =>
    intro acc h
    exact ih (addTo (key x) x acc) (addTo_keyed key x acc h)

theorem groupBy_keyed {K} [DecidableEq K] {A} (key : A → K) (l : List A) :
    ∀ q ∈ groupBy key l, ∀ y ∈ q.2, key y = q.1 :=
  groupAux_keyed key l [] (by intro q hq; exact absurd hq List.not_mem_nil)
/-! ## Records and the wire-shaped digest -/

abbrev Shape := List Nat × List Nat

/-- `digest_created_nodes`' shape key: the node's labels and its attribute ids.
    The engine sorts and dedups the labels first; that only ever *splits* a
    group, never mis-associates a row, so the model takes them as they come. -/
def NodeSpec.shape (s : NodeSpec) : Shape := (s.labels, s.attrs.map Prod.fst)

inductive Record where
  | addLabel     (id : Nat) (name : Name)
  | addAttribute (id : Nat) (name : Name)
  | createNode   (ids : List Nat) (labels : List Nat) (attrIds : List Nat) (rows : List Val)
  | deleteNode   (ids : List Nat) (labels : List Nat)
deriving Repr, DecidableEq

/-- `gather_rows`: row-major, one row per id, in the order the `IdList` has them. -/
def gatherRows (attrIds : List Nat) (g : List NodeSpec) : List Val :=
  g.flatMap (fun s => attrIds.map (lookupAttr s.attrs))

def createRec (q : Shape × List NodeSpec) : Record :=
  Record.createNode (q.2.map NodeSpec.id) q.1.1 q.1.2 (gatherRows q.1.2 q.2)

def deleteRec (q : List Nat × List (Nat × List Nat)) : Record :=
  Record.deleteNode (q.2.map Prod.fst) q.1

def schemaRecs (mk : Nat → Name → Record) : Nat → List Name → List Record
  | _, []      => []
  | b, n :: ns => mk b n :: schemaRecs mk (b + 1) ns

/-- `EffectsFormat::build`. Schema first — the ids on every later record are only
    meaningful once the replica has agreed on the numbering — then creates, then
    deletes. -/
def emit (p : Pending) (σ : GState) : List Record :=
  schemaRecs Record.addLabel σ.labels.length p.newLabels
  ++ schemaRecs Record.addAttribute σ.attrs.length p.newAttrs
  ++ (groupBy NodeSpec.shape p.created).map createRec
  ++ (groupBy Prod.snd p.deleted).map deleteRec

/-! ## The replica -/

structure Batch where
  entry   : Nat
  created : List Nat

/-- Rebuild the nodes a `CREATE_NODE` record describes. Row *k* goes to the k-th
    id in the `IdList`; nothing else associates a value with an entity. -/
def specsOf (ids labels attrIds : List Nat) (rows : List Val) : List NodeSpec :=
  (ids.zip (chunk ids.length attrIds.length rows)).map
    (fun pr => ⟨pr.1, labels, attrIds.zip pr.2⟩)

/-- `ID_LIMIT` (`id_space.rs:100`) = `GrB_INDEX_MAX` = `2^60 - 1`: `IdSpace::create` refuses
    any id at or above it (#2911, `fe619ac5f`; `id_space.rs:586`). -/
def idLimit : Nat := 2 ^ 60 - 1

/-- `verify_schema` + `verify_attribute` + `IdSpace::create`'s refusals.
    Note the two id checks are guarded by the recycle bin: an id that is free
    right now is neither live nor a double claim, which is what tells a genuine
    re-create from a collision. The last conjunct is the `ID_LIMIT` refusal
    (#2911), which the Rust checks first; a `Bool` conjunction's order is immaterial. -/
def okCreate (σ : GState) (b : Batch) (ids labels attrIds : List Nat) (rows : List Val) : Bool :=
  labels.all (fun l => decide (l < σ.labels.length))
  && attrIds.all (fun a => decide (a < σ.attrs.length))
  && nodupB attrIds
  && nodupB ids
  && decide (rows.length = ids.length * attrIds.length)
  && ids.all (fun i => σ.free i || decide (b.entry ≤ i))
  && ids.all (fun i => σ.free i || !b.created.contains i)
  && ids.all (fun i => decide (i < idLimit))

/-- `IdSpace::release`'s refusals: `refuse_not_live` (= `refuse_recycled` +
`refuse_undeletable`, #3022). -/
def okDelete (σ : GState) (b : Batch) (ids : List Nat) : Bool :=
  nodupB ids
  && ids.all (fun i => !σ.free i)
  && ids.all (fun i => decide (i < b.entry) || b.created.contains i)

def applyRec : Record → GState × Batch → Option (GState × Batch)
  | .addLabel id nm, (σ, b) =>
      if (intern σ.labels nm).1 = id then
        some ({ σ with labels := (intern σ.labels nm).2 }, b) else none
  | .addAttribute id nm, (σ, b) =>
      if (intern σ.attrs nm).1 = id then
        some ({ σ with attrs := (intern σ.attrs nm).2 }, b) else none
  | .createNode ids labels attrIds rows, (σ, b) =>
      if okCreate σ b ids labels attrIds rows then
        some (createBatch σ (specsOf ids labels attrIds rows),
              { b with created := b.created ++ ids.filter (fun i => decide (b.entry ≤ i)) })
      else none
  | .deleteNode ids _, (σ, b) =>
      if okDelete σ b ids then some (deleteBatch σ ids, b) else none

def applyRecs : List Record → GState × Batch → Option (GState × Batch)
  | [],      st => some st
  | r :: rs, st => match applyRec r st with
                   | some st' => applyRecs rs st'
                   | none     => none

theorem applyRecs_append (r1 r2 : List Record) (st : GState × Batch) :
    applyRecs (r1 ++ r2) st = (applyRecs r1 st).bind (applyRecs r2) := by
  induction r1 generalizing st with
  | nil => simp [applyRecs]
  | cons r rs ih =>
    simp only [List.cons_append, applyRecs]
    cases applyRec r st with
    | none => simp
    | some st' => simp [ih]

/-- `IdSpace::verify`: what the batch created at or above the entry boundary
    must be exactly `[entry_bound, entry_bound + created.len())`. Stated as the
    three order-independent facts that say so, since the engine's `created` is a
    set. -/
def Batch.verify (b : Batch) (σ : GState) : Bool :=
  b.created.all (fun i => decide (b.entry ≤ i) && decide (i < σ.bound))
  && decide (b.created.length = σ.bound - b.entry)
  && nodupB b.created

def applyPayload (recs : List Record) (σ : GState) : Option GState :=
  match applyRecs recs (σ, { entry := σ.bound, created := [] }) with
  | some (σ', b) => if b.verify σ' then some σ' else none
  | none         => none
/-! ## Small lemmas for the digest -/

theorem flatMap_length_const {α} (f : α → List Val) (m : Nat) :
    ∀ (g : List α), (∀ s ∈ g, (f s).length = m) → (g.flatMap f).length = g.length * m := by
  intro g
  induction g with
  | nil => intro _; simp
  | cons s rest ih =>
    intro h
    rw [List.flatMap_cons, List.length_append, h s (List.mem_cons_self ..),
        ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]
    simp [Nat.succ_mul, Nat.add_comm]

theorem addTo_nonempty {K} [DecidableEq K] {A} (k : K) (x : A) :
    ∀ acc : List (K × List A), (∀ q ∈ acc, q.2 ≠ []) → ∀ q ∈ addTo k x acc, q.2 ≠ [] := by
  intro acc
  induction acc with
  | nil => intro _ q hq; simp only [addTo, List.mem_singleton] at hq; subst hq; simp
  | cons hd rest ih =>
    obtain ⟨k', g⟩ := hd
    intro h q hq
    by_cases hk : k' = k
    · subst hk
      rw [addTo_cons_hit] at hq
      rcases List.mem_cons.mp hq with he | hr
      · subst he; simp
      · exact h q (List.mem_cons_of_mem _ hr)
    · rw [addTo_cons_miss _ _ _ _ _ hk] at hq
      rcases List.mem_cons.mp hq with he | hr
      · subst he; exact h (k', g) (List.mem_cons_self ..)
      · exact ih (fun z hz => h z (List.mem_cons_of_mem _ hz)) q hr

theorem groupAux_nonempty {K} [DecidableEq K] {A} (key : A → K) :
    ∀ (l : List A) (acc : List (K × List A)),
      (∀ q ∈ acc, q.2 ≠ []) → ∀ q ∈ groupAux key l acc, q.2 ≠ [] := by
  intro l
  induction l with
  | nil => intro acc h; exact h
  | cons x rest ih => intro acc h; exact ih _ (addTo_nonempty (key x) x acc h)

theorem groupBy_nonempty {K} [DecidableEq K] {A} (key : A → K) (l : List A) :
    ∀ q ∈ groupBy key l, q.2 ≠ [] :=
  groupAux_nonempty key l [] (by intro q hq; exact absurd hq List.not_mem_nil)

/-- The record a group becomes decodes back to exactly that group: row *k* lands
    on the k-th id, and nothing is dropped, duplicated or transposed. -/
theorem specsOf_group (sh : Shape) (g : List NodeSpec)
    (hkey : ∀ s ∈ g, s.shape = sh)
    (hnd  : ∀ s ∈ g, (s.attrs.map Prod.fst).Nodup) :
    specsOf (g.map NodeSpec.id) sh.1 sh.2 (gatherRows sh.2 g) = g := by
  have hf : ∀ s ∈ g, (sh.2.map (lookupAttr s.attrs)).length = sh.2.length := by
    intro s _; simp
  have hchunk : chunk (g.map NodeSpec.id).length sh.2.length (gatherRows sh.2 g)
              = g.map (fun s => sh.2.map (lookupAttr s.attrs)) := by
    rw [List.length_map]
    exact chunk_flatMap _ _ g hf
  rw [specsOf, hchunk, List.zip_map', List.map_map]
  refine Eq.trans (List.map_congr_left ?_) (List.map_id g)
  intro s hs
  have hsh := hkey s hs
  simp only [NodeSpec.shape] at hsh
  simp only [Function.comp_def, id_eq]
  rw [← hsh]
  simp only [lookup_row s.attrs (hnd s hs), List.zip_map']
  simp
theorem any_id_false (g : List NodeSpec) (i : Nat) (h : i ∉ g.map NodeSpec.id) :
    g.any (fun s => s.id == i) = false := by
  by_cases hc : g.any (fun s => s.id == i) = true
  · obtain ⟨s, hs, he⟩ := List.any_eq_true.mp hc
    exact absurd (by simpa [beq_iff_eq.mp he] using List.mem_map_of_mem (f := NodeSpec.id) hs) h
  · simpa using hc

/-- Applying the `CREATE_NODE` records a batch of shape groups becomes. -/
theorem applyCreateGroups :
  ∀ (groups : List (Shape × List NodeSpec)) (σ : GState) (b : Batch),
    (∀ q ∈ groups, ∀ s ∈ q.2, s.shape = q.1) →
    (∀ q ∈ groups, ∀ s ∈ q.2, (s.attrs.map Prod.fst).Nodup) →
    (∀ q ∈ groups, ∀ l ∈ q.1.1, l < σ.labels.length) →
    (∀ q ∈ groups, ∀ a ∈ q.1.2, a < σ.attrs.length) →
    (∀ q ∈ groups, (q.1.2).Nodup) →
    ((groups.flatMap Prod.snd).map NodeSpec.id).Nodup →
    (∀ s ∈ groups.flatMap Prod.snd, σ.free s.id = true ∨ b.entry ≤ s.id) →
    (∀ s ∈ groups.flatMap Prod.snd, σ.free s.id = true ∨ s.id ∉ b.created) →
    (∀ s ∈ groups.flatMap Prod.snd, s.id < idLimit) →
    applyRecs (groups.map createRec) (σ, b)
      = some (createBatch σ (groups.flatMap Prod.snd),
              { entry := b.entry,
                created := b.created ++ ((groups.flatMap Prod.snd).map NodeSpec.id).filter
                             (fun i => decide (b.entry ≤ i)) }) := by
  intro groups
  induction groups with
  | nil => intro σ b _ _ _ _ _ _ _ _ _; simp [applyRecs, createBatch]
  | cons q rest ih =>
    intro σ b hkey hattrnd hL hA hAN hnd hfree hclaim hlim
    have hfl : (q :: rest).flatMap Prod.snd = q.2 ++ rest.flatMap Prod.snd := List.flatMap_cons
    rw [hfl] at hnd hfree hclaim hlim
    rw [List.map_append] at hnd
    obtain ⟨hndq, hndr, hdisj⟩ := List.nodup_append.mp hnd
    -- the record this group becomes applies
    have hok : okCreate σ b (q.2.map NodeSpec.id) q.1.1 q.1.2 (gatherRows q.1.2 q.2) = true := by
      simp only [okCreate, Bool.and_eq_true]
      refine ⟨⟨⟨⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩
      · exact List.all_eq_true.mpr fun l hl => by
          simpa using hL q (List.mem_cons_self ..) l hl
      · exact List.all_eq_true.mpr fun a ha => by
          simpa using hA q (List.mem_cons_self ..) a ha
      · exact nodupB_iff.mpr (hAN q (List.mem_cons_self ..))
      · exact nodupB_iff.mpr hndq
      · have : ∀ s ∈ q.2, (q.1.2.map (lookupAttr s.attrs)).length = q.1.2.length := by
          intro _ _; simp
        rw [gatherRows, flatMap_length_const _ q.1.2.length q.2 this, List.length_map]
        simp
      · refine List.all_eq_true.mpr fun i hi => ?_
        obtain ⟨s, hs, he⟩ := List.mem_map.mp hi
        subst he
        rcases hfree s (List.mem_append.mpr (Or.inl hs)) with h | h
        · simp [h]
        · simp [h]
      · refine List.all_eq_true.mpr fun i hi => ?_
        obtain ⟨s, hs, he⟩ := List.mem_map.mp hi
        subst he
        rcases hclaim s (List.mem_append.mpr (Or.inl hs)) with h | h
        · simp [h]
        · simp [h]
      · refine List.all_eq_true.mpr fun i hi => ?_
        obtain ⟨s, hs, he⟩ := List.mem_map.mp hi
        subst he
        simpa using hlim s (List.mem_append.mpr (Or.inl hs))
    have hspecs : specsOf (q.2.map NodeSpec.id) q.1.1 q.1.2 (gatherRows q.1.2 q.2) = q.2 :=
      specsOf_group q.1 q.2 (hkey q (List.mem_cons_self ..)) (hattrnd q (List.mem_cons_self ..))
    -- the state and batch after it
    have hstep : applyRec (createRec q) (σ, b)
        = some (createBatch σ q.2,
                { entry := b.entry,
                  created := b.created ++ (q.2.map NodeSpec.id).filter
                               (fun i => decide (b.entry ≤ i)) }) := by
      simp only [createRec, applyRec, hok, ite_true, hspecs]
    have hdicts := createBatch_dicts σ q.2
    have hfree' : ∀ s ∈ rest.flatMap Prod.snd,
        (createBatch σ q.2).free s.id = σ.free s.id := by
      intro s hs
      have hni : s.id ∉ q.2.map NodeSpec.id := by
        intro hc
        exact hdisj s.id hc s.id (List.mem_map_of_mem hs) rfl
      rw [createBatch_free, any_id_false _ _ hni]
      simp
    simp only [List.map_cons, applyRecs, hstep]
    rw [ih (createBatch σ q.2)
          { entry := b.entry,
            created := b.created ++ (q.2.map NodeSpec.id).filter
                         (fun i => decide (b.entry ≤ i)) }
          (fun x hx => hkey x (List.mem_cons_of_mem _ hx))
          (fun x hx => hattrnd x (List.mem_cons_of_mem _ hx))
          (fun x hx l hl => hdicts.1 ▸ hL x (List.mem_cons_of_mem _ hx) l hl)
          (fun x hx a ha => hdicts.2 ▸ hA x (List.mem_cons_of_mem _ hx) a ha)
          (fun x hx => hAN x (List.mem_cons_of_mem _ hx))
          hndr
          (fun s hs => by
            rw [hfree' s hs]
            exact (hfree s (List.mem_append.mpr (Or.inr hs))).imp id id)
          (fun s hs => by
            rw [hfree' s hs]
            rcases hfree s (List.mem_append.mpr (Or.inr hs)) with h | _
            · exact Or.inl h
            · rcases hclaim s (List.mem_append.mpr (Or.inr hs)) with h2 | h2
              · exact Or.inl h2
              · refine Or.inr fun hc => ?_
                rcases List.mem_append.mp hc with hc1 | hc2
                · exact h2 hc1
                · exact hdisj s.id (List.mem_filter.mp hc2).1 s.id
                    (List.mem_map_of_mem hs) rfl)
          (fun s hs => hlim s (List.mem_append.mpr (Or.inr hs)))]
    rw [hfl, createBatch_append]
    simp only [List.map_append, List.filter_append, List.append_assoc]
/-- Applying the `DELETE_NODE` records a batch of label-set groups becomes. -/
theorem applyDeleteGroups :
  ∀ (groups : List (List Nat × List (Nat × List Nat))) (σ : GState) (b : Batch),
    (∀ pr ∈ groups.flatMap Prod.snd, σ.free pr.1 = false) →
    (∀ pr ∈ groups.flatMap Prod.snd, pr.1 < b.entry ∨ pr.1 ∈ b.created) →
    ((groups.flatMap Prod.snd).map Prod.fst).Nodup →
    applyRecs (groups.map deleteRec) (σ, b)
      = some (deleteBatch σ ((groups.flatMap Prod.snd).map Prod.fst), b) := by
  intro groups
  induction groups with
  | nil => intro σ b _ _ _; simp [applyRecs, deleteBatch]
  | cons q rest ih =>
    intro σ b hfree hown hnd
    have hfl : (q :: rest).flatMap Prod.snd = q.2 ++ rest.flatMap Prod.snd := List.flatMap_cons
    rw [hfl] at hfree hown hnd
    rw [List.map_append] at hnd
    obtain ⟨hndq, hndr, hdisj⟩ := List.nodup_append.mp hnd
    have hok : okDelete σ b (q.2.map Prod.fst) = true := by
      simp only [okDelete, Bool.and_eq_true]
      refine ⟨⟨nodupB_iff.mpr hndq, ?_⟩, ?_⟩
      · refine List.all_eq_true.mpr fun i hi => ?_
        obtain ⟨pr, hpr, he⟩ := List.mem_map.mp hi
        subst he
        simp [hfree pr (List.mem_append.mpr (Or.inl hpr))]
      · refine List.all_eq_true.mpr fun i hi => ?_
        obtain ⟨pr, hpr, he⟩ := List.mem_map.mp hi
        subst he
        rcases hown pr (List.mem_append.mpr (Or.inl hpr)) with h | h <;> simp [h]
    have hstep : applyRec (deleteRec q) (σ, b) = some (deleteBatch σ (q.2.map Prod.fst), b) := by
      simp only [deleteRec, applyRec, hok, ite_true]
    have hfree' : ∀ pr ∈ rest.flatMap Prod.snd,
        (deleteBatch σ (q.2.map Prod.fst)).free pr.1 = false := by
      intro pr hpr
      have hni : pr.1 ∉ q.2.map Prod.fst := by
        intro hc; exact hdisj pr.1 hc pr.1 (List.mem_map_of_mem hpr) rfl
      rw [(deleteBatch_simple σ (q.2.map Prod.fst)).2.2.2.1]
      simp only [hni, ite_false]
      exact hfree pr (List.mem_append.mpr (Or.inr hpr))
    simp only [List.map_cons, applyRecs, hstep]
    rw [ih (deleteBatch σ (q.2.map Prod.fst)) b hfree'
          (fun pr hpr => hown pr (List.mem_append.mpr (Or.inr hpr))) hndr,
        hfl, List.map_append, deleteBatch_append]

/-! ## Schema records -/

theorem applySchemaLabels :
  ∀ (ns : List Name) (d : List Name) (σ : GState) (b : Batch),
    σ.labels = d → (d ++ ns).Nodup →
    applyRecs (schemaRecs Record.addLabel d.length ns) (σ, b)
      = some ({ σ with labels := d ++ ns }, b) := by
  intro ns
  induction ns with
  | nil => intro d σ b hd _; subst hd; simp [schemaRecs, applyRecs]
  | cons n rest ih =>
    intro d σ b hd hnd
    have hnmem : n ∉ d := by
      intro hc
      exact (List.nodup_append.mp hnd).2.2 n hc n (List.mem_cons_self ..) rfl
    have hint : intern σ.labels n = (d.length, d ++ [n]) := by
      rw [hd]; exact intern_fresh hnmem
    have hstep : applyRec (Record.addLabel d.length n) (σ, b)
        = some ({ σ with labels := d ++ [n] }, b) := by
      simp only [applyRec, hint, ite_true]
    have hnd' : ((d ++ [n]) ++ rest).Nodup := by
      rw [List.append_assoc]; exact hnd
    simp only [schemaRecs, applyRecs, hstep]
    rw [show d.length + 1 = (d ++ [n]).length by simp,
        ih (d ++ [n]) { σ with labels := d ++ [n] } b rfl hnd']
    simp [List.append_assoc]

theorem applySchemaAttrs :
  ∀ (ns : List Name) (d : List Name) (σ : GState) (b : Batch),
    σ.attrs = d → (d ++ ns).Nodup →
    applyRecs (schemaRecs Record.addAttribute d.length ns) (σ, b)
      = some ({ σ with attrs := d ++ ns }, b) := by
  intro ns
  induction ns with
  | nil => intro d σ b hd _; subst hd; simp [schemaRecs, applyRecs]
  | cons n rest ih =>
    intro d σ b hd hnd
    have hnmem : n ∉ d := by
      intro hc
      exact (List.nodup_append.mp hnd).2.2 n hc n (List.mem_cons_self ..) rfl
    have hint : intern σ.attrs n = (d.length, d ++ [n]) := by
      rw [hd]; exact intern_fresh hnmem
    have hstep : applyRec (Record.addAttribute d.length n) (σ, b)
        = some ({ σ with attrs := d ++ [n] }, b) := by
      simp only [applyRec, hint, ite_true]
    have hnd' : ((d ++ [n]) ++ rest).Nodup := by
      rw [List.append_assoc]; exact hnd
    simp only [schemaRecs, applyRecs, hstep]
    rw [show d.length + 1 = (d ++ [n]).length by simp,
        ih (d ++ [n]) { σ with attrs := d ++ [n] } b rfl hnd']
    simp [List.append_assoc]
/-! ## The allocator

    `IdSpace::reserve`: recycled ids first, lowest first, then fresh ones from
    the boundary up. -/

def reusedOf (σ : GState) (n : Nat) : List Nat := σ.bin.take n
def freshCount (σ : GState) (n : Nat) : Nat := n - (reusedOf σ n).length
def idsOf (σ : GState) (n : Nat) : List Nat :=
  reusedOf σ n ++ List.range' σ.bound (freshCount σ n)

theorem bin_free {σ : GState} : ∀ i ∈ σ.bin, σ.free i = true := by
  intro i hi; exact (List.mem_filter.mp hi).2

theorem bin_lt {σ : GState} : ∀ i ∈ σ.bin, i < σ.bound := by
  intro i hi; exact List.mem_range.mp (List.mem_filter.mp hi).1

theorem bin_nodup {σ : GState} : σ.bin.Nodup :=
  List.filter_sublist.nodup List.nodup_range

theorem reused_free {σ : GState} (n : Nat) : ∀ i ∈ reusedOf σ n, σ.free i = true :=
  fun i hi => bin_free i ((List.take_sublist n σ.bin).mem hi)

theorem reused_lt {σ : GState} (n : Nat) : ∀ i ∈ reusedOf σ n, i < σ.bound :=
  fun i hi => bin_lt i ((List.take_sublist n σ.bin).mem hi)

theorem ids_length (σ : GState) (n : Nat) : (idsOf σ n).length = n := by
  simp only [idsOf, freshCount, reusedOf, List.length_append, List.length_range',
             List.length_take]
  omega

theorem ids_nodup {σ : GState} (n : Nat) : (idsOf σ n).Nodup := by
  rw [idsOf, List.nodup_append]
  refine ⟨(List.take_sublist n σ.bin).nodup bin_nodup, List.nodup_range' 1, ?_⟩
  intro x hx y hy
  exact Nat.ne_of_lt (Nat.lt_of_lt_of_le (reused_lt n x hx) (List.mem_range'_1.mp hy).1)

/-- Every id the allocator hands out is either recycled or past the boundary —
    which is exactly what `IdSpace::create` will not object to. -/
theorem ids_free_or_fresh {σ : GState} (n : Nat) :
    ∀ i ∈ idsOf σ n, σ.free i = true ∨ σ.bound ≤ i := by
  intro i hi
  rcases List.mem_append.mp hi with h | h
  · exact Or.inl (reused_free n i h)
  · exact Or.inr (List.mem_range'_1.mp h).1

theorem ids_filter_fresh {σ : GState} (hw : WF σ) (n : Nat) :
    (idsOf σ n).filter (fun i => !σ.free i) = List.range' σ.bound (freshCount σ n) := by
  rw [idsOf, List.filter_append]
  have h1 : (reusedOf σ n).filter (fun i => !σ.free i) = [] := by
    rw [List.filter_eq_nil_iff]; intro x hx; simp [reused_free n x hx]
  have h2 : (List.range' σ.bound (freshCount σ n)).filter (fun i => !σ.free i)
          = List.range' σ.bound (freshCount σ n) := by
    rw [List.filter_eq_self]
    intro x hx
    have hge := (List.mem_range'_1.mp hx).1
    have : σ.free x = false := by
      by_cases hf : σ.free x = true
      · exact absurd (hw.binOk x hf) (Nat.not_lt.mpr hge)
      · simpa using hf
    simp [this]
  rw [h1, h2, List.nil_append]

theorem ids_filter_ge {σ : GState} (n : Nat) :
    (idsOf σ n).filter (fun i => decide (σ.bound ≤ i)) = List.range' σ.bound (freshCount σ n) := by
  rw [idsOf, List.filter_append]
  have h1 : (reusedOf σ n).filter (fun i => decide (σ.bound ≤ i)) = [] := by
    rw [List.filter_eq_nil_iff]
    intro x hx; simp only [decide_eq_true_eq]
    exact Nat.not_le.mpr (reused_lt n x hx)
  have h2 : (List.range' σ.bound (freshCount σ n)).filter (fun i => decide (σ.bound ≤ i))
          = List.range' σ.bound (freshCount σ n) := by
    rw [List.filter_eq_self]
    intro x hx; simp [(List.mem_range'_1.mp hx).1]
  rw [h1, h2, List.nil_append]
theorem map_filter_comm {α β} (f : α → β) (q : β → Bool) :
    ∀ (l : List α), (l.filter (fun x => q (f x))).map f = (l.map f).filter q := by
  intro l
  induction l with
  | nil => rfl
  | cons x rest ih =>
    rw [List.filter_cons, List.map_cons, List.filter_cons]
    by_cases h : q (f x) = true <;> simp [h, ih]

/-! ## What the master promises about a `Pending`

    These are the facts the write path establishes before `commit` and `build`
    are called on the same value. `disjoint` is `digest_cancelled`'s reason for
    existing: a node created and deleted inside one segment is unwound out of
    `Pending` entirely, and travels as its own create/delete pair instead. -/

structure PendingWF (p : Pending) (σ : GState) : Prop where
  allocated   : p.ids = idsOf σ p.created.length
  attrsNodup  : ∀ s ∈ p.created, (s.attrs.map Prod.fst).Nodup
  labelsOk    : ∀ s ∈ p.created, ∀ l ∈ s.labels, l < σ.labels.length + p.newLabels.length
  attrsOk     : ∀ s ∈ p.created, ∀ a ∈ s.attrs.map Prod.fst,
                  a < σ.attrs.length + p.newAttrs.length
  newLabelsOk : (σ.labels ++ p.newLabels).Nodup
  newAttrsOk  : (σ.attrs ++ p.newAttrs).Nodup
  delsNodup   : p.dels.Nodup
  delsLive    : ∀ i ∈ p.dels, σ.liveB i = true
  disjoint    : ∀ i ∈ p.ids, i ∉ p.dels
  /-- The master's own `IdSpace::create` (`Graph::create_nodes`) refused any id at or above
      `ID_LIMIT` (#2911), so every id a committed `Pending` carries is below it. -/
  inLimit     : ∀ i ∈ p.ids, i < idLimit

/-- **emit and apply agree.** The payload `build` produces from a `Pending`,
    replayed by `apply` on the state `commit` ran against, succeeds and lands on
    exactly the state `commit` produced. -/
theorem emit_apply (p : Pending) (σ : GState) (hw : WF σ) (hp : PendingWF p σ) :
    applyPayload (emit p σ) σ = some (commit p σ) := by
  have hidnd : p.ids.Nodup := hp.allocated ▸ ids_nodup p.created.length
  have hcnd : (p.created.map NodeSpec.id).Nodup := hidnd
  -- the shape groups
  have hperm : ((groupBy NodeSpec.shape p.created).flatMap Prod.snd).Perm p.created :=
    groupBy_flat _ _
  have hmem : ∀ s ∈ (groupBy NodeSpec.shape p.created).flatMap Prod.snd, s ∈ p.created :=
    fun s hs => hperm.mem_iff.mp hs
  have hmemq : ∀ q ∈ groupBy NodeSpec.shape p.created, ∀ s ∈ q.2, s ∈ p.created := by
    intro q hq s hs
    exact hmem s (List.mem_flatMap.mpr ⟨q, hq, hs⟩)
  have hrep : ∀ q ∈ groupBy NodeSpec.shape p.created, ∃ s ∈ q.2, s.shape = q.1 := by
    intro q hq
    have hne := groupBy_nonempty NodeSpec.shape p.created q hq
    cases hq2 : q.2 with
    | nil => exact absurd hq2 hne
    | cons s tail =>
      have hs : s ∈ q.2 := by rw [hq2]; exact List.mem_cons_self ..
      exact ⟨s, List.mem_cons_self .., groupBy_keyed _ _ q hq s hs⟩
  -- the delete groups
  have hdperm : ((groupBy Prod.snd p.deleted).flatMap Prod.snd).Perm p.deleted :=
    groupBy_flat _ _
  -- the two schema prefixes
  simp only [emit, applyPayload, applyRecs_append]
  rw [applySchemaLabels p.newLabels σ.labels σ { entry := σ.bound, created := [] } rfl
        hp.newLabelsOk]
  simp only [Option.bind_some]
  rw [applySchemaAttrs p.newAttrs σ.attrs { σ with labels := σ.labels ++ p.newLabels }
        { entry := σ.bound, created := [] } rfl hp.newAttrsOk]
  simp only [Option.bind_some]
  have hfnd : (((groupBy NodeSpec.shape p.created).flatMap Prod.snd).map NodeSpec.id).Nodup :=
    (hperm.map NodeSpec.id).symm.nodup hcnd
  rw [applyCreateGroups (groupBy NodeSpec.shape p.created)
        { labels := σ.labels ++ p.newLabels, attrs := σ.attrs ++ p.newAttrs, bound := σ.bound,
          free := σ.free, nodeLabels := σ.nodeLabels, props := σ.props }
        { entry := σ.bound, created := [] }
        (groupBy_keyed _ _)
        (fun q hq s hs => hp.attrsNodup s (hmemq q hq s hs))
        (by intro q hq l hl
            obtain ⟨s, hs, hsh⟩ := hrep q hq
            rw [← hsh] at hl
            simpa using hp.labelsOk s (hmemq q hq s hs) l hl)
        (by intro q hq a ha
            obtain ⟨s, hs, hsh⟩ := hrep q hq
            rw [← hsh] at ha
            simpa using hp.attrsOk s (hmemq q hq s hs) a ha)
        (by intro q hq
            obtain ⟨s, hs, hsh⟩ := hrep q hq
            rw [← hsh]
            exact hp.attrsNodup s (hmemq q hq s hs))
        hfnd
        (by intro s hs
            have : s.id ∈ p.ids := List.mem_map_of_mem (hmem s hs)
            rw [hp.allocated] at this
            exact ids_free_or_fresh _ s.id this)
        (by intro s _; exact Or.inr List.not_mem_nil)
        (by intro s hs; exact hp.inLimit s.id (List.mem_map_of_mem (hmem s hs)))]
  simp only [Option.bind_some]
  rw [applyDeleteGroups (groupBy Prod.snd p.deleted) _ _
        (by intro pr hpr
            have hd : pr.1 ∈ p.dels := List.mem_map_of_mem (hdperm.mem_iff.mp hpr)
            have hlive := hp.delsLive pr.1 hd
            simp only [GState.liveB, Bool.and_eq_true, Bool.not_eq_true',
                       decide_eq_true_eq] at hlive
            rw [createBatch_free]
            by_cases hany : ((groupBy NodeSpec.shape p.created).flatMap Prod.snd).any
                              (fun s => s.id == pr.1) = true
            · simp [hany]
            · simp only [Bool.not_eq_true] at hany; simp [hany, hlive.2])
        (by intro pr hpr
            have hd : pr.1 ∈ p.dels := List.mem_map_of_mem (hdperm.mem_iff.mp hpr)
            have hlive := hp.delsLive pr.1 hd
            simp only [GState.liveB, Bool.and_eq_true, decide_eq_true_eq] at hlive
            exact Or.inl hlive.1)
        ((hdperm.map Prod.fst).symm.nodup hp.delsNodup)]
  -- the states now agree; what is left is `IdSpace::verify`
  rw [createBatch_perm _ hperm hfnd, deleteBatch_perm _ ((hdperm.map Prod.fst))]
  have hcr : ((([] : List Nat) ++ (((groupBy NodeSpec.shape p.created).flatMap Prod.snd).map
                NodeSpec.id).filter (fun i => decide (σ.bound ≤ i)))).Perm
             (List.range' σ.bound (freshCount σ p.created.length)) := by
    simp only [List.nil_append]
    have h2 := (hperm.map NodeSpec.id).filter (fun i => decide (σ.bound ≤ i))
    rw [show p.created.map NodeSpec.id = p.ids from rfl, hp.allocated, ids_filter_ge] at h2
    exact h2
  have hbound : (commit p σ).bound = σ.bound + freshCount σ p.created.length := by
    show (deleteBatch (createBatch _ p.created) p.dels).bound = _
    rw [(deleteBatch_simple _ _).2.2.1, createBatch_bound _ p.created hcnd]
    have hl : (p.created.filter (fun s => !σ.free s.id)).length
            = ((p.created.map NodeSpec.id).filter (fun i => !σ.free i)).length := by
      rw [← map_filter_comm NodeSpec.id (fun i => !σ.free i) p.created, List.length_map]
    have hlen : (p.created.filter (fun s => !σ.free s.id)).length
              = freshCount σ p.created.length := by
      rw [hl, show p.created.map NodeSpec.id = p.ids from rfl, hp.allocated,
          ids_filter_fresh hw, List.length_range']
    exact congrArg (σ.bound + ·) hlen
  have hver : Batch.verify
      { entry := σ.bound,
        created := ([] : List Nat) ++ (((groupBy NodeSpec.shape p.created).flatMap Prod.snd).map
                     NodeSpec.id).filter (fun i => decide (σ.bound ≤ i)) }
      (commit p σ) = true := by
    simp only [Batch.verify, Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨⟨?_, ?_⟩, ?_⟩
    · refine List.all_eq_true.mpr fun i hi => ?_
      have hm := List.mem_range'_1.mp (hcr.mem_iff.mp hi)
      simp only [Bool.and_eq_true, decide_eq_true_eq, hbound]
      exact ⟨hm.1, hm.2⟩
    · rw [hcr.length_eq, List.length_range', hbound]; omega
    · exact nodupB_iff.mpr (hcr.symm.nodup (List.nodup_range' 1))
  show (if Batch.verify
          { entry := σ.bound,
            created := ([] : List Nat) ++ (((groupBy NodeSpec.shape p.created).flatMap Prod.snd).map
                         NodeSpec.id).filter (fun i => decide (σ.bound ≤ i)) }
          (commit p σ) = true
        then some (commit p σ) else none) = some (commit p σ)
  rw [hver]
  simp
/-! ## What a commit does, and that it keeps the graph well formed -/

theorem commit_labels (p : Pending) (σ : GState) :
    (commit p σ).labels = σ.labels ++ p.newLabels := by
  show (deleteBatch (createBatch { σ with labels := σ.labels ++ p.newLabels,
                                          attrs := σ.attrs ++ p.newAttrs } p.created) p.dels).labels
         = _
  rw [(deleteBatch_simple _ _).1, (createBatch_dicts _ p.created).1]

theorem commit_attrs (p : Pending) (σ : GState) :
    (commit p σ).attrs = σ.attrs ++ p.newAttrs := by
  show (deleteBatch (createBatch { σ with labels := σ.labels ++ p.newLabels,
                                          attrs := σ.attrs ++ p.newAttrs } p.created) p.dels).attrs
         = _
  rw [(deleteBatch_simple _ _).2.1, (createBatch_dicts _ p.created).2]

theorem commit_bound (p : Pending) (σ : GState) (hcnd : (p.created.map NodeSpec.id).Nodup) :
    (commit p σ).bound = σ.bound + (p.created.filter (fun s => !σ.free s.id)).length := by
  show (deleteBatch (createBatch { σ with labels := σ.labels ++ p.newLabels,
                                          attrs := σ.attrs ++ p.newAttrs } p.created) p.dels).bound
         = _
  rw [(deleteBatch_simple _ _).2.2.1, createBatch_bound _ p.created hcnd]

theorem commit_free (p : Pending) (σ : GState) (i : Nat) :
    (commit p σ).free i
      = (if i ∈ p.dels then true
         else if p.created.any (fun s => s.id == i) then false else σ.free i) := by
  show (deleteBatch (createBatch { σ with labels := σ.labels ++ p.newLabels,
                                          attrs := σ.attrs ++ p.newAttrs } p.created) p.dels).free i
         = _
  rw [(deleteBatch_simple _ _).2.2.2.1]
  by_cases hd : i ∈ p.dels
  · simp [hd]
  · simp [hd, createBatch_free]

theorem wf_commit (p : Pending) (σ : GState) (hw : WF σ) (hp : PendingWF p σ) :
    WF (commit p σ) := by
  have hidnd : p.ids.Nodup := hp.allocated ▸ ids_nodup p.created.length
  have hcnd : (p.created.map NodeSpec.id).Nodup := hidnd
  have hbound : σ.bound ≤ (commit p σ).bound := by
    rw [commit_bound p σ hcnd]; exact Nat.le_add_right _ _
  refine ⟨?_, ?_, ?_⟩
  · intro i hi
    rw [commit_free] at hi
    by_cases hd : i ∈ p.dels
    · have hlive := hp.delsLive i hd
      simp only [GState.liveB, Bool.and_eq_true, decide_eq_true_eq] at hlive
      exact Nat.lt_of_lt_of_le hlive.1 hbound
    · simp only [hd, ite_false] at hi
      by_cases hany : p.created.any (fun s => s.id == i) = true
      · simp [hany] at hi
      · simp only [Bool.not_eq_true] at hany
        rw [hany] at hi
        have hi' : σ.free i = true := by simpa using hi
        exact Nat.lt_of_lt_of_le (hw.binOk i hi') hbound
  · rw [commit_labels]; exact hp.newLabelsOk
  · rw [commit_attrs]; exact hp.newAttrsOk

/-! ## A stream of writes -/

def masterRun : List Pending → GState → GState × List (List Record)
  | [],      σ => (σ, [])
  | p :: ps, σ =>
      let rs := masterRun ps (commit p σ)
      (rs.1, emit p σ :: rs.2)

def replicaRun : List (List Record) → GState → Option GState
  | [],      σ => some σ
  | b :: bs, σ => match applyPayload b σ with
                  | some σ' => replicaRun bs σ'
                  | none    => none

def PendingsOk : List Pending → GState → Prop
  | [],      _ => True
  | p :: ps, σ => PendingWF p σ ∧ PendingsOk ps (commit p σ)

/-- **No divergence over a stream.** A replica that starts equal to the master
    and receives every payload, in order, ends equal to the master. -/
theorem stream_simulation : ∀ (ps : List Pending) (σ : GState), WF σ → PendingsOk ps σ →
    replicaRun (masterRun ps σ).2 σ = some (masterRun ps σ).1 := by
  intro ps
  induction ps with
  | nil => intro σ _ _; simp [masterRun, replicaRun]
  | cons p rest ih =>
    intro σ hw hok
    obtain ⟨h1, h2⟩ := hok
    simp only [masterRun, replicaRun, emit_apply p σ hw h1]
    exact ih (commit p σ) (wf_commit p σ hw h1) h2

/-! ## The replica as a guarded state machine -/

inductive Replica where
  | inSync (σ : GState)
  | resync
  | halted

structure Node where
  masterState : GState
  replica     : Replica

def stepNode (p : Pending) (nd : Node) : Node :=
  match nd.replica with
  | .inSync σr =>
      match applyPayload (emit p nd.masterState) σr with
      | some σr' => { masterState := commit p nd.masterState, replica := .inSync σr' }
      | none     => { masterState := commit p nd.masterState, replica := .resync }
  | .resync => { masterState := commit p nd.masterState, replica := .resync }
  | .halted => { masterState := commit p nd.masterState, replica := .halted }

def stepsNode : List Pending → Node → Node
  | [],      nd => nd
  | p :: ps, nd => stepsNode ps (stepNode p nd)

def NoDivergence (nd : Node) : Prop :=
  ∀ σr, nd.replica = .inSync σr → σr = nd.masterState

def Inv (nd : Node) : Prop := WF nd.masterState ∧ NoDivergence nd

theorem step_preserves (p : Pending) (nd : Node) (hp : PendingWF p nd.masterState)
    (hi : Inv nd) : Inv (stepNode p nd) := by
  obtain ⟨hwf, hnd⟩ := hi
  refine ⟨?_, ?_⟩
  · cases hr : nd.replica with
    | inSync σr => simp only [stepNode, hr]
                   cases applyPayload (emit p nd.masterState) σr <;>
                     exact wf_commit p nd.masterState hwf hp
    | resync    => simp only [stepNode, hr]; exact wf_commit p nd.masterState hwf hp
    | halted    => simp only [stepNode, hr]; exact wf_commit p nd.masterState hwf hp
  · intro σ' h'
    cases hr : nd.replica with
    | inSync σr =>
      have heq : σr = nd.masterState := hnd σr hr
      rw [heq] at hr
      simp only [stepNode, hr, emit_apply p nd.masterState hwf hp] at h' ⊢
      injection h' with e
      exact e.symm
    | resync => simp only [stepNode, hr] at h'; exact Replica.noConfusion h'
    | halted => simp only [stepNode, hr] at h'; exact Replica.noConfusion h'

/-- No spurious resync: a healthy pair never takes the guard's failure branch. -/
theorem no_spurious_resync (p : Pending) (nd : Node) (hp : PendingWF p nd.masterState)
    (hi : Inv nd) (σr : GState) (hr : nd.replica = .inSync σr) :
    (stepNode p nd).replica = .inSync (commit p nd.masterState) := by
  obtain ⟨hwf, hnd⟩ := hi
  have heq : σr = nd.masterState := hnd σr hr
  rw [heq] at hr
  simp only [stepNode, hr, emit_apply p nd.masterState hwf hp]

def PendingsOkN : List Pending → Node → Prop
  | [],      _  => True
  | p :: ps, nd => PendingWF p nd.masterState ∧ PendingsOkN ps (stepNode p nd)

theorem no_divergence : ∀ (ps : List Pending) (nd : Node), Inv nd → PendingsOkN ps nd →
    Inv (stepsNode ps nd) := by
  intro ps
  induction ps with
  | nil => intro nd hi _; exact hi
  | cons p rest ih =>
    intro nd hi hok
    obtain ⟨h1, h2⟩ := hok
    exact ih (stepNode p nd) (step_preserves p nd h1 hi) h2

/-- The headline: every state the replica ever serves is the master's state. -/
theorem replica_never_serves_a_different_graph
    (ps : List Pending) (σ : GState) (h : WF σ) (hok : PendingsOkN ps ⟨σ, .inSync σ⟩) :
    ∀ σr, (stepsNode ps ⟨σ, .inSync σ⟩).replica = .inSync σr →
          σr = (stepsNode ps ⟨σ, .inSync σ⟩).masterState :=
  (no_divergence ps ⟨σ, .inSync σ⟩ ⟨h, by intro σ' h'; injection h' with e; exact e.symm⟩ hok).2
/-! ## The checks have teeth

    All executable: `#guard` evaluates at elaboration time and the file does not
    compile if one is false. -/

section Sanity

def σ0 : GState :=
  { labels := [], attrs := [], bound := 0, free := fun _ => false,
    nodeLabels := fun _ => [], props := fun _ _ => Val.null }

/- `CREATE (:L {p:1}), (:L {p:2})` — one shape, so one record. -/
def p1 : Pending :=
  { created := [⟨0, [0], [(0, Val.num 1)]⟩, ⟨1, [0], [(0, Val.num 2)]⟩],
    deleted := [], newLabels := ["L"], newAttrs := ["p"] }

#guard emit p1 σ0 == [Record.addLabel 0 "L", Record.addAttribute 0 "p",
                      Record.createNode [0, 1] [0] [0] [Val.num 1, Val.num 2]]
#guard ((applyPayload (emit p1 σ0) σ0).map GState.bound) == some 2
#guard ((applyPayload (emit p1 σ0) σ0).map (fun s => s.props 0 0)) == some (Val.num 1)
#guard ((applyPayload (emit p1 σ0) σ0).map (fun s => s.props 1 0)) == some (Val.num 2)
#guard ((applyPayload (emit p1 σ0) σ0).map (fun s => s.nodeLabels 1)) == some [0]

def σ1 : GState := commit p1 σ0

/- `CREATE (:A {x:1}), (:B {x:2}), (:A {x:3})` — two shapes, so the middle node
   travels in its own record and ids 0 and 2 share the first. The values still
   land on the right nodes, and the boundary still lands on 3: a create raises
   it once per id that was not in the recycle bin, not once per id at or above
   the old boundary. -/
def p2 : Pending :=
  { created := [⟨0, [0], [(0, Val.num 1)]⟩, ⟨1, [1], [(0, Val.num 2)]⟩,
                ⟨2, [0], [(0, Val.num 3)]⟩],
    deleted := [], newLabels := ["A", "B"], newAttrs := ["x"] }

#guard emit p2 σ0 == [Record.addLabel 0 "A", Record.addLabel 1 "B", Record.addAttribute 0 "x",
                      Record.createNode [0, 2] [0] [0] [Val.num 1, Val.num 3],
                      Record.createNode [1] [1] [0] [Val.num 2]]
#guard ((applyPayload (emit p2 σ0) σ0).map
          (fun s => (s.props 0 0, s.props 1 0, s.props 2 0)))
       == some (Val.num 1, Val.num 2, Val.num 3)
#guard ((applyPayload (emit p2 σ0) σ0).map GState.bound) == some 3

/- A replica whose label dictionary is numbered differently refuses the buffer
   rather than writing through the id (`apply_add_schema` / `verify_id`). -/
#guard (applyPayload (emit p1 σ0) { σ0 with labels := ["X"] }).isNone

/- The next write allocates id 2. A replica that **missed** `p1` but happens to
   share the schema is handed it: the ids do not start at its own boundary, so
   `IdSpace::verify` refuses the whole buffer instead of accepting an id space
   its master does not have. -/
def p3 : Pending :=
  { created := [⟨2, [0], [(0, Val.num 9)]⟩], deleted := [], newLabels := [], newAttrs := [] }

#guard emit p3 σ1 == [Record.createNode [2] [0] [0] [Val.num 9]]
#guard (applyPayload (emit p3 σ1) σ1).isSome
#guard (applyPayload (emit p3 σ1) { σ0 with labels := ["L"], attrs := ["p"] }).isNone

/- Delete recycles, and the next create reuses the freed id — on both sides,
   without the boundary moving. -/
def p4 : Pending :=
  { created := [], deleted := [(0, [0])], newLabels := [], newAttrs := [] }
def σ2 : GState := commit p4 σ1

#guard emit p4 σ1 == [Record.deleteNode [0] [0]]
#guard ((applyPayload (emit p4 σ1) σ1).map GState.bound) == some 2
#guard ((applyPayload (emit p4 σ1) σ1).map (fun s => s.free 0)) == some true

def p5 : Pending :=
  { created := [⟨0, [0], [(0, Val.num 7)]⟩], deleted := [], newLabels := [], newAttrs := [] }

#guard emit p5 σ2 == [Record.createNode [0] [0] [0] [Val.num 7]]
#guard ((applyPayload (emit p5 σ2) σ2).map GState.bound) == some 2
#guard ((applyPayload (emit p5 σ2) σ2).map (fun s => s.props 0 0)) == some (Val.num 7)

/- Deleting an id that is already free is refused, not applied twice. -/
#guard (applyPayload [Record.deleteNode [0] [0]] σ2).isNone

end Sanity

/-! ## The abstract failure branch is `divergence_guard::on_failure`

`stepNode` sends a replica whose apply failed to `.resync`, and the abstract
model has a `.halted` state it never enters. `Guard.onFailure` (a line-by-line
model of `src/divergence_guard.rs`) decides which: `guardReplica` reads off the
state its effect list leaves the replica in. -/

/-- What `on_failure`'s effects do to the replica: nothing (client command),
`exit(1)` (halted), or a scheduled full resync. -/
def guardReplica (f : Guard.Flags) : Option Replica :=
  if !Guard.isReplayed f then none else if f.loading then some .halted else some .resync

/-- `guardReplica` is exactly what `onFailure`'s effect list encodes. -/
theorem guardReplica_spec (c) (f : Guard.Flags) (g m1 m2 l) :
    (guardReplica f = none ↔ Guard.onFailure c f g m1 m2 l = []) ∧
    (guardReplica f = some .halted ↔ (Guard.onFailure c f g m1 m2 l).getLast? = some .exit) ∧
    (guardReplica f = some .resync ↔ (Guard.onFailure c f g m1 m2 l).getLast? = some (.timer g)) := by
  cases f with
  | mk r ld =>
    cases r <;> cases ld <;>
      simp [guardReplica, Guard.onFailure, Guard.isReplayed, Guard.getLast?_cons_append_single]

/-- The replication stream is applied under `REPLICATED` and not `LOADING`, so
the abstract failure branch of `stepNode` (`→ .resync`) is exactly what the
real guard does there. -/
theorem stepNode_failure_matches_guard (p : Pending) (nd : Node) (σr : GState)
    (hr : nd.replica = .inSync σr) (hf : applyPayload (emit p nd.masterState) σr = none) :
    some (stepNode p nd).replica = guardReplica ⟨true, false⟩ := by
  simp [stepNode, hr, hf, guardReplica, Guard.isReplayed]

/-- A failed replay of the node's own AOF (`LOADING`) halts rather than resyncs. -/
theorem guard_loading_halts (r : Bool) : guardReplica ⟨r, true⟩ = some .halted := by
  cases r <;> rfl

/-! ## Refinement: the abstract steps are the engine's

Each abstract step of this model is proven equal to a faithful model of the
Rust function it stands for (`FalkorMemo`, copied from
proofs/effects_emit_apply; `FalkorFaithful`, restating proofs/effects_emit_apply
and proofs/id_space). So `emit_apply`, `stream_simulation` and
`replica_never_serves_a_different_graph` are statements about those functions. -/

theorem idxOf_findIdx : ∀ (d : List Name) (n : Name) (k : Nat),
    idxOf d n k = (d.findIdx? (· == n)).map (· + k) := by
  intro d; induction d with
  | nil => intro n k; rfl
  | cons m ms ih =>
    intro n k
    simp only [idxOf, List.findIdx?_cons]
    by_cases h : m = n
    · simp [h]
    · simp only [h, ite_false, beq_iff_eq, ih n (k + 1), Option.map_map]
      congr 1; funext x; simp [Function.comp]; omega

/-- `intern` is `get_label_id_mut` / `add_node_attribute_name`. -/
theorem intern_refines (d : List Name) (n : Name) : intern d n = FalkorFaithful.internF d n := by
  unfold intern FalkorFaithful.internF FalkorFaithful.idxOfF
  rw [idxOf_findIdx]
  cases d.findIdx? (· == n) <;> simp

/-- **`applyRec`'s `ADD_SCHEMA`/`ADD_ATTRIBUTE` arms are `apply_add_schema`.** -/
theorem applyRec_addLabel_refines (id : Nat) (nm : Name) (σ : GState) (b : Batch) :
    applyRec (.addLabel id nm) (σ, b) =
      (FalkorFaithful.applyAddSchemaF σ.labels id nm).map fun d => ({ σ with labels := d }, b) := by
  simp only [applyRec, FalkorFaithful.applyAddSchemaF, ← intern_refines, FalkorFaithful.verifyIdF]
  by_cases h : (intern σ.labels nm).1 = id <;> simp [h]

theorem applyRec_addAttribute_refines (id : Nat) (nm : Name) (σ : GState) (b : Batch) :
    applyRec (.addAttribute id nm) (σ, b) =
      (FalkorFaithful.applyAddSchemaF σ.attrs id nm).map fun d => ({ σ with attrs := d }, b) := by
  simp only [applyRec, FalkorFaithful.applyAddSchemaF, ← intern_refines, FalkorFaithful.verifyIdF]
  by_cases h : (intern σ.attrs nm).1 = id <;> simp [h]

/-- **`idsOf` is `IdSpace::reserve` on a fresh batch** (recycled first, lowest
first, then fresh from the boundary). -/
theorem idsOf_refines (σ : GState) (n : Nat) : idsOf σ n = FalkorFaithful.reserveFresh σ.bound σ.bin n := by
  simp [idsOf, FalkorFaithful.reserveFresh, reusedOf, freshCount]

theorem addTo_bridge {K} [DecidableEq K] {A} (k : K) (x : A) :
    ∀ acc, addTo k x acc = FalkorMemo.addTo k x acc := by
  intro acc; induction acc with
  | nil => rfl
  | cons hd tl ih => obtain ⟨k', g⟩ := hd; simp only [addTo, FalkorMemo.addTo, ih]

theorem groupBy_bridge {K} [DecidableEq K] {A} (key : A → K) (l : List A) :
    groupBy key l = FalkorMemo.groupBy key l := by
  unfold groupBy FalkorMemo.groupBy
  suffices ∀ acc, groupAux key l acc = FalkorMemo.groupAux key l acc from this []
  induction l with
  | nil => intro acc; rfl
  | cons x rest ih => intro acc; simp only [groupAux, FalkorMemo.groupAux, addTo_bridge, ih]

/-- **`emit`'s `groupBy` is the engine's `slots`/`index`/`last` loop**
(`digest_created_nodes`, `digest_deleted_nodes`). -/
theorem memo_refines_groupBy {K} [DecidableEq K] {A} (key : A → K) (l : List A) :
    FalkorMemo.memoGroup key l = groupBy key l := by
  rw [groupBy_bridge]; exact FalkorMemo.memoGroup_eq_groupBy key l

/-- `emit` written with the engine's loops is `emit`. -/
theorem emit_refines (p : Pending) (σ : GState) :
    emit p σ = schemaRecs Record.addLabel σ.labels.length p.newLabels
      ++ schemaRecs Record.addAttribute σ.attrs.length p.newAttrs
      ++ (FalkorMemo.memoGroup NodeSpec.shape p.created).map createRec
      ++ (FalkorMemo.memoGroup Prod.snd p.deleted).map deleteRec := by
  simp only [emit, memo_refines_groupBy]

/-! ## The id space the engine keeps (#2846)

Since #2846 `Graph` holds no counter or bin of its own: `node_id_bound` is
`IdSpace::bound` = `live + recycled.len()`, `create_nodes` is `IdSpace::create`
and `delete_nodes` is `IdSpace::release`. `GState.bound`/`free` are that space
abstracted (`absIds`), and `createOne`/`deleteOne` move it exactly as those two
transitions do. -/

/-- The abstraction: the boundary and the free set of an engine id space. -/
def absIds (s : FalkorFaithful.Ids) : Nat × (Nat → Bool) := (s.bound, fun i => s.bin.contains i)

/-- **`GState.bound` is `node_id_bound`**, and `createOne` moves it as
`IdSpace::create` does: a recycled id leaves the bin and the count grows, so the
boundary is unchanged; a fresh one grows it by one. -/
theorem createOne_refines (σ : GState) (s : FalkorFaithful.Ids) (hn : s.bin.Nodup)
    (hb : σ.bound = s.bound) (hf : ∀ i, σ.free i = s.bin.contains i) (id : Nat) (lbls row) :
    (createOne σ id lbls row).bound = (s.create1 id).bound ∧
    ∀ i, (createOne σ id lbls row).free i = (s.create1 id).bin.contains i := by
  refine ⟨?_, fun i => ?_⟩
  · simp only [createOne, FalkorFaithful.Ids.create1, FalkorFaithful.Ids.bound] at hb ⊢
    rw [hf id]
    by_cases hm : id ∈ s.bin
    · have := List.length_erase_of_mem hm
      have hp : 0 < s.bin.length := List.length_pos_of_mem hm
      simp [hm, this]; omega
    · simp [hm, List.erase_of_not_mem hm]; omega
  · simp only [createOne, FalkorFaithful.Ids.create1]
    by_cases e : i = id
    · subst e; simp [List.Nodup.mem_erase_iff hn]
    · simp [e, hf i, List.Nodup.mem_erase_iff hn]

/-- …and `deleteOne` as `IdSpace::release` does on a live id: the count drops
and the bin grows, so the boundary is unchanged. -/
theorem deleteOne_refines (σ : GState) (s : FalkorFaithful.Ids)
    (hb : σ.bound = s.bound) (hf : ∀ i, σ.free i = s.bin.contains i) (id : Nat)
    (hlive : 1 ≤ s.live) (hnf : id ∉ s.bin) :
    (deleteOne σ id).bound = (s.release1 id).bound ∧
    ∀ i, (deleteOne σ id).free i = (s.release1 id).bin.contains i ∧ (s.release1 id).bin.Nodup = s.bin.Nodup := by
  refine ⟨?_, fun i => ⟨?_, ?_⟩⟩
  · simp only [deleteOne, FalkorFaithful.Ids.release1, FalkorFaithful.Ids.bound] at hb ⊢
    simp; omega
  · simp only [deleteOne, FalkorFaithful.Ids.release1]
    by_cases e : i = id
    · subst e; simp
    · simp [e, hf i]
  · simp [FalkorFaithful.Ids.release1, hnf]

end Falkor

-- Nothing here rests on `sorryAx`: only Lean's three standard axioms.
#print axioms Falkor.specsOf_group
#print axioms Falkor.createBatch_perm
#print axioms Falkor.emit_apply
#print axioms Falkor.stream_simulation
#print axioms Falkor.no_divergence
#print axioms Falkor.replica_never_serves_a_different_graph
