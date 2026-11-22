/-!
# Relationship lookups behind the reply encoders (#3447)

`RespCompact.encC` / `RespVerbose.encVb` take an edge's type id / type name, source and
destination as given fields. On `origin/main` those fields are *looked up*
while the reply is written, and some of the lookups panic when the id is not where the
code expects it (#3447). This file models the lookups over the three places a relationship id can
live during a query, and the two ways a reply's id can end up in none of the expected ones.

`none` is the panic (the server dies). C has no `GRAPH.RECORD`; its compact reply of a
deleted edge resolves the type by the edge's own schema (issue #3447: C returns the edge).
-/
namespace RedisLayer.RelLookup

/-- A relationship as the runtime can describe it: id, type *name*, endpoints. -/
structure Rel where
  id : Nat
  ty : String
  src : Nat
  dst : Nat
  deriving DecidableEq, Repr

structure St where
  graph : List Rel        -- committed in `runtime.g`
  types : List String     -- `relationship_types` registry (type id = position)
  deleted : List Rel      -- `runtime.deleted_relationships`
  pending : List Rel      -- `created_rels_by_type`

def look (l : List Rel) (id : Nat) : Option Rel := l.find? (·.id = id)

/-- `Graph::get_type_id`. -/
def typeId (types : List String) (name : String) : Option Nat := types.idxOf? name

/-- `Graph::get_relationship_type_id`: the type id of a *committed* edge. -/
def graphTyId (s : St) (id : Nat) : Option Nat := do
  let r ← look s.graph id
  typeId s.types r.ty

/-- `Runtime::get_relationship_endpoints`. -/
def endpoints (s : St) (id : Nat) : Option (Nat × Nat) :=
  match look s.deleted id with
  | some r => some (r.src, r.dst)
  | none => match look s.pending id with
    | some r => some (r.src, r.dst)
    | none => (look s.graph id).map fun r => (r.src, r.dst)

/-- `reply_compact_value` on `Value::Relationship`: `(type id, src, dst)` or panic. -/
def compactRel (s : St) (id : Nat) : Option (Nat × Nat × Nat) := do
  let (a, b) ← endpoints s id
  let t ← match look s.deleted id with
    | some r => typeId s.types r.ty            -- `.unwrap()`
    | none => graphTyId s id                   -- `.expect(..)`
  pure (t, a, b)

/-- `reply_verbose_value` on `Value::Relationship` on main (#3462): the type name comes from
`Runtime::get_relationship_type` (`runtime.rs`): deleted → pending → graph. -/
def verboseRel (s : St) (id : Nat) : Option (String × Nat × Nat) := do
  let (a, b) ← endpoints s id
  let t ← match look s.deleted id with
    | some r => some r.ty
    | none => match look s.pending id with
      | some r => some r.ty
      | none => (look s.graph id).map (·.ty)
  pure (t, a, b)

/-- The id is somewhere the runtime can describe it. -/
def Known (s : St) (id : Nat) : Prop :=
  (look s.deleted id).isSome ∨ (look s.pending id).isSome ∨ (look s.graph id).isSome

/-- Property `RegisteredTypes`: every relationship the runtime can still describe — live,
snapshotted or pending — has its type name in the registry. -/
def RegisteredTypes (s : St) : Prop :=
  ∀ r, (r ∈ s.graph ∨ r ∈ s.deleted ∨ r ∈ s.pending) → r.ty ∈ s.types

/-- Property `RecordedLive`: every id captured in a recorded row is still describable when
the rows are rendered. -/
def RecordedLive (captured : List Nat) (s : St) : Prop := ∀ id ∈ captured, Known s id

/-- Property `ReplyTotal`: replies render every relationship the runtime knows. -/
def ReplyTotal (s : St) : Prop :=
  ∀ id, Known s id → (compactRel s id).isSome ∧ (verboseRel s id).isSome

theorem typeId_some {types : List String} {n : String} (h : n ∈ types) :
    (typeId types n).isSome := by
  unfold typeId
  cases h' : types.idxOf? n
  · exact absurd h (List.idxOf?_eq_none_iff.mp h')
  · rfl

theorem look_mem {l : List Rel} {id : Nat} {r : Rel} (h : look l id = some r) : r ∈ l :=
  List.mem_of_find?_eq_some h

/-- What a fix must establish: once nothing is pending (every reply of `GRAPH.QUERY` is
written after commit) and every type is registered, both encoders are total. -/
theorem replyTotal_of_registered (s : St) (hp : s.pending = []) (hr : RegisteredTypes s) :
    ReplyTotal s := by
  intro id hk
  have hpn : look s.pending id = none := by simp [hp, look]
  unfold compactRel verboseRel endpoints
  rcases hd : look s.deleted id with _ | r
  · simp only [hpn]
    rcases hg : look s.graph id with _ | g
    · simp [Known, hd, hpn, hg] at hk
    · have ht := typeId_some (hr g (Or.inl (look_mem hg)))
      obtain ⟨t, ht⟩ := Option.isSome_iff_exists.1 ht
      simp [graphTyId, hg, ht]
  · have ht := typeId_some (hr r (Or.inr (Or.inl (look_mem hd))))
    obtain ⟨t, ht⟩ := Option.isSome_iff_exists.1 ht
    simp [ht]

/-! ## #3447: a cancelled edge of a new type is never registered -/

/-- `CREATE` of `rels` (each flagged `true` when a `DETACH DELETE` of an endpoint cascades it
away) on a graph with registry `pre`, then commit: the cancelled edges become snapshots,
the survivors are committed, and only the survivors' types are registered. -/
def commitCascade (pre : List String) (rels : List (Rel × Bool)) : St :=
  let live := (rels.filter (!·.2)).map (·.1)
  { graph := live
    types := live.foldl (fun ts r => if r.ty ∈ ts then ts else ts ++ [r.ty]) pre
    deleted := (rels.filter (·.2)).map (·.1)
    pending := [] }

/-- `CREATE (a)-[r:NEW_TYPE]->(b) DETACH DELETE a RETURN r` on an empty graph. -/
def s3447 : St := commitCascade [] [(⟨0, "NEW_TYPE", 0, 1⟩, true)]

/-- **#3447**: the snapshot of `r` names `NEW_TYPE`, which nothing registered, so the compact
reply's `get_type_id(..).unwrap()` (`reply.rs`) panics, while the verbose reply renders
it (`NEW_TYPE`, 0 → 1). `RegisteredTypes` and `ReplyTotal` both fail after commit. -/
theorem bug3447_compact_cancelled_new_type_panics :
    Known s3447 0 ∧ compactRel s3447 0 = none ∧
      verboseRel s3447 0 = some ("NEW_TYPE", 0, 1) ∧
      ¬ RegisteredTypes s3447 ∧ ¬ ReplyTotal s3447 := by
  refine ⟨by simp [Known, s3447, commitCascade, look], by decide, by decide, ?_, ?_⟩
  · intro h
    have := h ⟨0, "NEW_TYPE", 0, 1⟩ (Or.inr (Or.inl (by simp [s3447, commitCascade])))
    simp [s3447, commitCascade] at this
  · intro h
    have := (h 0 (by simp [Known, s3447, commitCascade, look])).1
    simp [show compactRel s3447 0 = none by decide] at this

/-- With an edge of the same type that survives, the type is registered and the reply is
fine: the bug needs every edge of the new type to be cancelled. -/
theorem s3447_survivor_ok :
    compactRel (commitCascade [] [(⟨0, "NEW_TYPE", 0, 1⟩, true), (⟨1, "NEW_TYPE", 2, 3⟩, false)]) 0
      = some (0, 0, 1) := by decide

/-! ## #3449 (fixed by #3462): `GRAPH.RECORD` rendered ids the runtime no longer (or not yet) described -/

/-- `MATCH ()-[r]->() DELETE r` under `GRAPH.RECORD`: `r` leaves the graph at commit and is
snapshotted only when `snap` (`DeleteOp::new`, `delete.rs`). -/
def recordDelete (snap : Bool) (g : List Rel) (types : List String) (r : Rel) : St :=
  { graph := g.filter (· ≠ r), types, deleted := if snap then [r] else [], pending := [] }

/-- `snapshot_required` for a `Delete` whose only ancestor is `Commit` (top-level `DELETE`). -/
def snapshotRequiredTopLevel : Bool := false

def snapshot (inspect snapReq : Bool) : Bool := inspect || snapReq

def r0 : Rel := ⟨0, "R", 0, 1⟩
/-- The same recorded `DELETE r` on main. -/
def s3449del : St := recordDelete (snapshot true snapshotRequiredTopLevel) [r0] ["R"] r0

/-- `CREATE (a)-[r:R]->(b) SET a.x = 1/0` under `GRAPH.RECORD`: the error precedes `Commit`;
`r` is still pending (and, the type being new, unregistered) when the rows are rendered. -/
def s3449abort : St := { graph := [], types := [], deleted := [], pending := [r0] }

/-- **#3449 fixed (1)**: with `inspect || snapshot_required` the recorded `DELETE r` is
snapshotted, so every captured id is still describable (`RecordedLive`) and the verbose
reply renders it from the snapshot. -/
theorem fixed3449_record_snapshot :
    RecordedLive [r0.id] s3449del ∧ verboseRel s3449del 0 = some ("R", 0, 1) := by
  refine ⟨fun id hid => ?_, by decide⟩
  simp [r0] at hid; subst hid
  simp [Known, s3449del, snapshot, recordDelete, snapshotRequiredTopLevel, look, r0]

/-- `fixed3449_record_snapshot` for any state of the flags: a recorded query (`inspect`)
always snapshots, whatever `snapshot_required` says. -/
theorem snapshot_inspect (b : Bool) : snapshot true b = true := rfl

/-- **#3449 fixed (2)**: the aborted `CREATE`'s pending `r` renders with its pending type
(`Runtime::get_relationship_type`), `R`, 0 → 1, whatever the registry. -/
theorem fixed3449_record_abort : verboseRel s3449abort 0 = some ("R", 0, 1) := by decide

/-- With the fix the verbose encoder is total on every known id (deleted, pending or
committed), whatever the registry. -/
theorem verboseRelFixed_total (s : St) (id : Nat) (h : Known s id) :
    (verboseRel s id).isSome := by
  unfold verboseRel endpoints
  rcases hd : look s.deleted id with _ | r
  · rcases hp : look s.pending id with _ | p
    · rcases hg : look s.graph id with _ | g
      · simp [Known, hd, hp, hg] at h
      · simp
    · simp
  · simp

end RedisLayer.RelLookup
