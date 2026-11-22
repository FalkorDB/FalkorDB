import RedisLayer.TelemetryLoop
import RedisLayer.GraphCore
/-!
# Telemetry streams kept out of snapshots (#3844, `src/telemetry.rs`, `src/redis_type.rs`)

The queue is a mutex-guarded list plus an atomic flag, both changed only under the lock by
`push`/`take`; each operation is modelled sequentially. The Redis key API (`OpenKey`,
`KeyType`, `DeleteKey`, `CloseKey`) is recorded as the operations performed.

Main property: `freed_stream_stripped` — a graph freed outside a flush, on a thread holding
the GIL, has its stream deleted at the next persistence start although it is no longer in the
registry the strip walks; `stripAll_covers` — every registered graph's stream is stripped too.
-/
namespace RedisLayer.Telemetry

/-! ## `strip_stream` -/

inductive SOp | openRead | openWrite | delete | close
  deriving DecidableEq, Repr

/-- The key-API calls on `telemetry{g}`: a read-only open first (a null handle for a missing
key), a write open and `DeleteKey` only for a stream. -/
def stripOps : KeyType → List SOp
  | .empty => [.openRead]
  | .other => [.openRead, .close]
  | .stream => [.openRead, .close, .openWrite, .delete, .close]

/-- The return value: whether a key was deleted. -/
def stripStream : KeyType → Bool
  | .stream => true
  | _ => false

/-- Only a stream is deleted: a key of another type that shares the name is left alone. -/
theorem strip_deletes_iff (kt : KeyType) : SOp.delete ∈ stripOps kt ↔ kt = .stream := by
  cases kt <;> simp [stripOps]

/-- A key is opened for writing (which aborts a `WATCH`) only when it is deleted. -/
theorem strip_write_open_iff (kt : KeyType) : SOp.openWrite ∈ stripOps kt ↔ kt = .stream := by
  cases kt <;> simp [stripOps]

theorem stripStream_iff (kt : KeyType) : stripStream kt = true ↔ SOp.delete ∈ stripOps kt := by
  cases kt <;> simp [stripStream, stripOps]

/-! ## `FreedGraphStreams`

Each queued name is a `(db, graph name)` pair (#2782): the stream lives in the graph's own
database. -/

/-- A queued stream: the database and the graph name it is named after. -/
abbrev DbName := Nat × List Char

structure FQ where
  names : List DbName
  pending : Bool

/-- `FreedGraphStreams::new`. -/
def FQ.new : FQ := ⟨[], false⟩
/-- `push`: extend the list, then raise the flag. -/
def FQ.push (q : FQ) (ns : List DbName) : FQ := ⟨q.names ++ ns, true⟩
/-- `is_pending`. -/
def FQ.isPending (q : FQ) : Bool := q.pending
/-- `take`: `swap(false)`; empty if it was not raised, else the whole list. -/
def FQ.take (q : FQ) : List DbName × FQ :=
  if !q.pending then ([], q) else (q.names, ⟨[], false⟩)
/-- `swap_dbs` (`SWAPDB a b`): a name queued for `a` is now in `b`, and the other way round. -/
def FQ.swapDbs (q : FQ) (a b : Nat) : FQ :=
  ⟨q.names.map fun (d, n) => (GraphCore.swapDb a b d, n), q.pending⟩

/-- Every queued name is followed by a raised flag. -/
def FQ.Inv (q : FQ) : Prop := q.names ≠ [] → q.pending = true

theorem FQ.new_inv : FQ.new.Inv := fun h => absurd rfl h
theorem FQ.push_inv (q : FQ) (ns : List DbName) : (q.push ns).Inv := fun _ => rfl
theorem FQ.take_inv (q : FQ) (h : q.Inv) : q.take.2.Inv := by
  unfold FQ.take; cases hp : q.pending
  · simpa [hp] using h
  · intro hn; simp at hn
/-- Under the invariant `take` returns every queued name and leaves the queue empty. -/
theorem FQ.take_all (q : FQ) (h : q.Inv) : q.take.1 = q.names ∧ q.take.2 = FQ.new := by
  unfold FQ.take FQ.new; cases hp : q.pending
  · have : q.names = [] := by
      by_cases hn : q.names = []
      · exact hn
      · have := h hn; simp [hp] at this
    cases q; simp_all
  · simp
theorem FQ.isPending_push (q : FQ) (ns : List DbName) : (q.push ns).isPending = true := rfl

/-- `swap_dbs` keeps the invariant, and every queued name follows its database. -/
theorem FQ.swapDbs_inv (q : FQ) (a b : Nat) (h : q.Inv) : (q.swapDbs a b).Inv := by
  intro hn; apply h; intro e; apply hn; simp [FQ.swapDbs, e]
theorem FQ.swapDbs_mem (q : FQ) (a b d : Nat) (n : List Char) (h : (d, n) ∈ q.names) :
    (GraphCore.swapDb a b d, n) ∈ (q.swapDbs a b).names := by
  simp only [FQ.swapDbs, List.mem_map]; exact ⟨(d, n), h, rfl⟩

/-! ## `set_flush_in_progress`, `queue_streams_of_freed_graph`, `delete_streams_of_freed_graphs` -/

def setFlush (_old b : Bool) : Bool := b
theorem setFlush_eq (o b : Bool) : setFlush o b = b := rfl

/-- Nothing for no names, off the GIL (the lazy-free thread) or during a flush. -/
def queueFreed (q : FQ) (ns : List DbName) (holdsGil flushing : Bool) : FQ :=
  if ns.isEmpty || !holdsGil || flushing then q else q.push ns

theorem queue_flushing (q : FQ) (ns : List DbName) (g : Bool) : queueFreed q ns g true = q := by
  simp [queueFreed]
theorem queue_no_gil (q : FQ) (ns : List DbName) (f : Bool) : queueFreed q ns false f = q := by
  simp [queueFreed]
theorem queue_inv (q : FQ) (ns : List DbName) (g f : Bool) (h : q.Inv) : (queueFreed q ns g f).Inv := by
  unfold queueFreed; split
  · exact h
  · exact FQ.push_inv _ _
/-- Outside a flush, on the GIL, every name is queued (and the old ones kept). -/
theorem queue_pushes (q : FQ) (ns : List DbName) (n : DbName) (hn : n ∈ ns ∨ n ∈ q.names) :
    n ∈ (queueFreed q ns true false).names := by
  unfold queueFreed; split
  · rename_i he
    have : ns = [] := by simpa using he
    subst this; simpa using hn
  · simp [FQ.push]; exact hn.symm

/-- `swap_queued_stream_dbs` (called from `on_swapdb`). -/
def swapQueued (q : FQ) (a b : Nat) : FQ := q.swapDbs a b

/-- `delete_streams_of_freed_graphs`: the `(db, name)`s whose stream `strip_stream_in_db` is
called on. -/
def deleteFreed (q : FQ) : List DbName × FQ := q.take

theorem deleteFreed_all (q : FQ) (h : q.Inv) :
    (deleteFreed q).1 = q.names ∧ (deleteFreed q).2 = FQ.new := FQ.take_all q h

/-! ## Database switches (`RestoreSelectedDb`, `strip_stream_in_db`, `delete_stream_in_db`)

`RestoreSelectedDb::select ctx db`: `None` when `RedisModule_SelectDb` refuses (`ok = false`),
else a guard that selects the original database back when dropped. The selected database
after a `*_in_db` helper is therefore always the one before it. -/

/-- `RestoreSelectedDb::select`: the guard (holding the database to restore), if selected. -/
def restoreSelect (orig : Nat) (ok : Bool) : Option Nat := if ok then some orig else none

/-- `strip_stream_in_db` / `delete_stream_in_db`: run `f` in `db` iff it could be selected;
returns the result and the database selected afterwards. -/
def inDb {α} (orig : Nat) (ok : Bool) (f : α) : Option α × Nat :=
  match restoreSelect orig ok with
  | some o => (some f, o)
  | none => (none, orig)

theorem inDb_restores {α} (orig : Nat) (ok : Bool) (f : α) : (inDb orig ok f).2 = orig := by
  cases ok <;> rfl
theorem inDb_runs {α} (orig : Nat) (f : α) : (inDb orig true f).1 = some f := rfl
theorem inDb_refused {α} (orig : Nat) (f : α) : (inDb orig false f).1 = none := rfl

/-- `delete_stream_if_stream`: `DEL` of the stream name only when `TYPE` answers `stream`
(simple or bulk string). Returns the deleted key, if any. -/
def deleteIfStream (typeReply : Option String) (g : List Char) : Option (List Char) :=
  if typeReply = some "stream" then some (streamName g) else none

theorem deleteIfStream_iff (t : Option String) (g : List Char) :
    (deleteIfStream t g).isSome ↔ t = some "stream" := by
  unfold deleteIfStream; split <;> simp_all
/-- What is deleted is the graph's stream name, whatever the caller passed (key or name). -/
theorem deleteIfStream_key (t : Option String) (g : List Char) :
    deleteIfStream t (upToNul g) = deleteIfStream t g := by
  simp [deleteIfStream, streamName_key_name]

/-! ## `strip_telemetry_streams` (`redis_type.rs`) -/

/-- Queued freed graphs first, then every registered graph, each in its own database;
returns the `(db, name)`s stripped. -/
def stripAll (q : FQ) (reg : GraphCore.Reg) : List DbName × FQ :=
  ((deleteFreed q).1 ++ reg.map (fun e => (e.db, e.name)), (deleteFreed q).2)

theorem stripAll_covers (q : FQ) (reg : GraphCore.Reg) (h : q.Inv) (n : DbName)
    (hn : n ∈ q.names ∨ ∃ e ∈ reg, (e.db, e.name) = n) : n ∈ (stripAll q reg).1 := by
  simp only [stripAll, (deleteFreed_all q h).1, List.mem_append, List.mem_map]
  rcases hn with hn | ⟨e, he, rfl⟩
  · exact Or.inl hn
  · exact Or.inr ⟨e, he, rfl⟩

theorem stripAll_empties (q : FQ) (reg : GraphCore.Reg) (h : q.Inv) : (stripAll q reg).2 = FQ.new :=
  (deleteFreed_all q h).2

/-! ## `graph_free` (`graph_core.rs`) -/

/-- The stream the freed graph leaves to delete: its `(db, name)`, unless another graph of
that database (still registered after the removal) shares the name cut at its first NUL —
the stream is then the survivor's. -/
def freedKeys (r : GraphCore.Reg) (g : Nat) : List DbName :=
  match r.get g with
  | some e => if GraphCore.nameIsShared (GraphCore.free r g) e.db e.name then [] else [(e.db, e.name)]
  | none => []

def graphFree (r : GraphCore.Reg) (g : Nat) (q : FQ) (holdsGil flushing : Bool) : GraphCore.Reg × FQ :=
  (GraphCore.free r g, queueFreed q (freedKeys r g) holdsGil flushing)

theorem graphFree_registry (r : GraphCore.Reg) (g : Nat) (q : FQ) (h f : Bool) :
    (graphFree r g q h f).1 = GraphCore.free r g := rfl

/-- The freed graph's `(db, name)` is queued, unless its stream is shared with a survivor. -/
theorem graphFree_queues (r : GraphCore.Reg) (g : Nat) (q : FQ) (e : GraphCore.Ent)
    (hk : r.get g = some e) (hs : GraphCore.nameIsShared (GraphCore.free r g) e.db e.name = false) :
    (e.db, e.name) ∈ (graphFree r g q true false).2.names := by
  apply queue_pushes
  left; simp [freedKeys, hk, hs]

/-- A shared stream is left: it belongs to the graph of the same database and name that
stays registered. -/
theorem graphFree_shared_kept (r : GraphCore.Reg) (g : Nat) (q : FQ) (e : GraphCore.Ent) (h f : Bool)
    (hk : r.get g = some e) (hs : GraphCore.nameIsShared (GraphCore.free r g) e.db e.name = true) :
    (graphFree r g q h f).2 = q := by
  simp [graphFree, freedKeys, hk, hs, queueFreed]

/-- **A freed graph's stream reaches no snapshot.** A graph registered at `(db, k)`, freed on
the GIL outside a flush, has its stream stripped at the next persistence start: queued,
whatever the registry then holds; or, when a survivor of the same database shares the stream
name, through that survivor's own registry entry, as long as it is still registered there
(a survivor that leaves the registry or its key queues or deletes the stream itself). The
stream is identified by its database and its name cut at the first NUL (`streamName`). -/
theorem freed_stream_stripped (r reg' : GraphCore.Reg) (g : Nat) (q : FQ) (e : GraphCore.Ent) (hq : q.Inv)
    (hk : r.get g = some e)
    (hsurv : ∀ e' ∈ GraphCore.free r g, e'.db = e.db → GraphCore.upToNulC e'.name = GraphCore.upToNulC e.name →
      e' ∈ reg') :
    ∃ n, (e.db, n) ∈ (stripAll (graphFree r g q true false).2 reg').1 ∧
      GraphCore.upToNulC n = GraphCore.upToNulC e.name := by
  cases hs : GraphCore.nameIsShared (GraphCore.free r g) e.db e.name with
  | false =>
    exact ⟨e.name, stripAll_covers _ reg' (queue_inv _ _ _ _ hq) _ (Or.inl (graphFree_queues r g q e hk hs)), rfl⟩
  | true =>
    unfold GraphCore.nameIsShared at hs
    obtain ⟨e', he', hm⟩ := List.any_eq_true.1 hs
    simp only [Bool.and_eq_true, beq_iff_eq] at hm
    refine ⟨e'.name, stripAll_covers _ reg' (queue_inv _ _ _ _ hq) _ (Or.inr ⟨e', hsurv e' he' hm.1 hm.2, by rw [hm.1]⟩), hm.2⟩

end RedisLayer.Telemetry
