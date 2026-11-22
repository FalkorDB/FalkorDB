import RedisLayer.TelemetryLoop
import RedisLayer.GraphCore
/-!
# Telemetry streams kept out of snapshots (#3844, `src/telemetry.rs`, `src/redis_type.rs`)

| here | there |
| --- | --- |
| `stripOps`, `stripStream` | `strip_stream` `telemetry.rs` |
| `FQ.*`             | `FreedGraphStreams::new/push/is_pending/take` |
| `setFlush`         | `set_flush_in_progress` |
| `queueFreed`       | `queue_streams_of_freed_graph` |
| `deleteFreed`      | `delete_streams_of_freed_graphs` |
| `stripAll`         | `strip_telemetry_streams`, `redis_type.rs` |
| `graphFree`        | `graph_free`, `graph_core.rs` (registry removal = `GraphCore.free`, then the queue) |

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

/-! ## `FreedGraphStreams` -/

structure FQ where
  names : List (List Char)
  pending : Bool

/-- `FreedGraphStreams::new`. -/
def FQ.new : FQ := ⟨[], false⟩
/-- `push`: extend the list, then raise the flag. -/
def FQ.push (q : FQ) (ns : List (List Char)) : FQ := ⟨q.names ++ ns, true⟩
/-- `is_pending`. -/
def FQ.isPending (q : FQ) : Bool := q.pending
/-- `take`: `swap(false)`; empty if it was not raised, else the whole list. -/
def FQ.take (q : FQ) : List (List Char) × FQ :=
  if !q.pending then ([], q) else (q.names, ⟨[], false⟩)

/-- Every queued name is followed by a raised flag. -/
def FQ.Inv (q : FQ) : Prop := q.names ≠ [] → q.pending = true

theorem FQ.new_inv : FQ.new.Inv := fun h => absurd rfl h
theorem FQ.push_inv (q : FQ) (ns : List (List Char)) : (q.push ns).Inv := fun _ => rfl
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
theorem FQ.isPending_push (q : FQ) (ns : List (List Char)) : (q.push ns).isPending = true := rfl

/-! ## `set_flush_in_progress`, `queue_streams_of_freed_graph`, `delete_streams_of_freed_graphs` -/

def setFlush (_old b : Bool) : Bool := b
theorem setFlush_eq (o b : Bool) : setFlush o b = b := rfl

/-- Nothing for no names, off the GIL (the lazy-free thread) or during a flush. -/
def queueFreed (q : FQ) (ns : List (List Char)) (holdsGil flushing : Bool) : FQ :=
  if ns.isEmpty || !holdsGil || flushing then q else q.push ns

theorem queue_flushing (q : FQ) (ns : List (List Char)) (g : Bool) : queueFreed q ns g true = q := by
  simp [queueFreed]
theorem queue_no_gil (q : FQ) (ns : List (List Char)) (f : Bool) : queueFreed q ns false f = q := by
  simp [queueFreed]
theorem queue_inv (q : FQ) (ns : List (List Char)) (g f : Bool) (h : q.Inv) : (queueFreed q ns g f).Inv := by
  unfold queueFreed; split
  · exact h
  · exact FQ.push_inv _ _
/-- Outside a flush, on the GIL, every name is queued (and the old ones kept). -/
theorem queue_pushes (q : FQ) (ns : List (List Char)) (n : List Char) (hn : n ∈ ns ∨ n ∈ q.names) :
    n ∈ (queueFreed q ns true false).names := by
  unfold queueFreed; split
  · rename_i he
    have : ns = [] := by simpa using he
    subst this; simpa using hn
  · simp [FQ.push]; exact hn.symm

/-- `delete_streams_of_freed_graphs`: the names whose stream `strip_stream` is called on. -/
def deleteFreed (q : FQ) : List (List Char) × FQ := q.take

theorem deleteFreed_all (q : FQ) (h : q.Inv) :
    (deleteFreed q).1 = q.names ∧ (deleteFreed q).2 = FQ.new := FQ.take_all q h

/-! ## `strip_telemetry_streams` (`redis_type.rs`) -/

/-- Queued freed graphs first, then every registered key; returns the names stripped. -/
def stripAll (q : FQ) (reg : GraphCore.Reg) : List (List Char) × FQ :=
  ((deleteFreed q).1 ++ reg.map (·.1), (deleteFreed q).2)

theorem stripAll_covers (q : FQ) (reg : GraphCore.Reg) (h : q.Inv) (n : List Char)
    (hn : n ∈ q.names ∨ ∃ g, (n, g) ∈ reg) : n ∈ (stripAll q reg).1 := by
  simp only [stripAll, (deleteFreed_all q h).1, List.mem_append, List.mem_map]
  rcases hn with hn | ⟨g, hg⟩
  · exact Or.inl hn
  · exact Or.inr ⟨(n, g), hg, rfl⟩

theorem stripAll_empties (q : FQ) (reg : GraphCore.Reg) (h : q.Inv) : (stripAll q reg).2 = FQ.new :=
  (deleteFreed_all q h).2

/-! ## `graph_free` (`graph_core.rs`) -/

/-- The keys the freed graph was registered under. -/
def freedKeys (r : GraphCore.Reg) (g : Nat) : List (List Char) := (r.filter (·.2 = g)).map (·.1)

def graphFree (r : GraphCore.Reg) (g : Nat) (q : FQ) (holdsGil flushing : Bool) : GraphCore.Reg × FQ :=
  (GraphCore.free r g, queueFreed q (freedKeys r g) holdsGil flushing)

theorem graphFree_registry (r : GraphCore.Reg) (g : Nat) (q : FQ) (h f : Bool) :
    (graphFree r g q h f).1 = GraphCore.free r g := rfl

theorem graphFree_queues (r : GraphCore.Reg) (g : Nat) (q : FQ) (k : List Char) (hk : (k, g) ∈ r) :
    k ∈ (graphFree r g q true false).2.names := by
  apply queue_pushes
  left; simp only [freedKeys, List.mem_map, List.mem_filter]
  exact ⟨(k, g), ⟨hk, by simp⟩, rfl⟩

/-- **A freed graph's stream reaches no snapshot.** A graph registered under `k`, freed on
the GIL outside a flush, has `k` stripped at the next persistence start — whatever the
registry then holds. -/
theorem freed_stream_stripped (r reg' : GraphCore.Reg) (g : Nat) (q : FQ) (k : List Char) (hq : q.Inv)
    (hk : (k, g) ∈ r) :
    k ∈ (stripAll (graphFree r g q true false).2 reg').1 :=
  stripAll_covers _ reg' (queue_inv _ _ _ _ hq) k (Or.inl (graphFree_queues r g q k hk))

end RedisLayer.Telemetry
