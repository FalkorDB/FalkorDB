import RedisLayer.TelemetryReg
/-!
# Telemetry: flusher control flow and the remaining helpers (`src/telemetry.rs`)

| here | there |
| --- | --- |
| `nextIds`          | `next_id` (`fetch_add(1)`) |
| `writable`, `streamOps` | `stream_entries` |
| `deleteStreamKey`  | `delete_stream` |
| `runningRow`, `waitingRow` | `running_queries_reply`, `waiting_queries_reply` |
| `distinctAlive`    | `hold_batch_graphs` |
| `releaseSpawns`    | `release_batch_graphs` |
| `keyHolds`         | `key_holds_graph`, `redis_type.rs` (moved there by #3819) |
| `Flusher`, `start`, `stopped` | `start_flusher_thread`, `shutdown_flusher_thread` |
| `iter`, `iterSteps` | one iteration of `flusher_loop` (#3844: no role/pause check; freed streams deleted first) |
| `Pre3844.*`        | historical: the replica/pause branches and `set_is_replica`, removed by #3844 |
-/
namespace RedisLayer.Telemetry

/-- `next_id`: successive calls return `n, n+1, …` — so ids are unique. -/
def nextIds (start k : Nat) : List Nat := (List.range k).map (start + ·)

theorem nextIds_nodup (start k : Nat) : (nextIds start k).Nodup := by
  unfold nextIds List.Nodup
  rw [List.pairwise_map]
  exact (List.pairwise_lt_range).imp (fun h => by omega)

/-- Redis key types (`REDISMODULE_KEYTYPE_*`). -/
inductive KeyType | empty | stream | other

/-- A key of some other type is left alone. -/
def writable : KeyType → Bool
  | .empty | .stream => true
  | .other => false

inductive Op | add (i : Nat) | trim (maxLen : Int) | close
  deriving DecidableEq, Repr

/-- The key-API operations `stream_entries` performs on an opened key, for `n` entries. -/
def streamOps (kt : KeyType) (n : Nat) (maxLen : Int) : List Op :=
  if writable kt then (List.range n).map Op.add ++ [.trim maxLen, .close] else [.close]

/-- Every entry is appended, in order, and the stream is trimmed — unconditionally, so
`MAX_INFO_QUERIES 0` keeps nothing, as in C. -/
theorem streamOps_writable (kt : KeyType) (n : Nat) (m : Int) (h : writable kt) :
    streamOps kt n m = (List.range n).map Op.add ++ [.trim m, .close] := by
  simp [streamOps, h]

theorem streamOps_trim_zero (n : Nat) : Op.trim 0 ∈ streamOps .empty n 0 := by
  simp [streamOps, writable]

theorem streamOps_foreign (n : Nat) (m : Int) : streamOps .other n m = [.close] := rfl

/-- `delete_stream`: `DEL` of the stream name, whatever the caller passed (key or name). -/
def deleteStreamKey (g : List Char) : List Char := streamName g
theorem deleteStream_key_or_name (g : List Char) :
    deleteStreamKey (upToNul g) = deleteStreamKey g := streamName_key_name g

/-- `GRAPH.INFO` rows. -/
inductive RV (F : Type) | bulk (s : String) | int (i : Int) | float (x : F)

def runningRow {F : Type} (recv : Int) (g q : String) (dur : F) (repl : Bool) : List (RV F) :=
  [.bulk "Received at", .int recv, .bulk "Graph name", .bulk g, .bulk "Query", .bulk q,
   .bulk "Execution duration", .float dur, .bulk "Replicated command", .int (if repl then 1 else 0)]

def waitingRow {F : Type} (recv : Int) (g q : String) (dur : F) : List (RV F) :=
  [.bulk "Received at", .int recv, .bulk "Graph name", .bulk g, .bulk "Query", .bulk q,
   .bulk "Wait duration", .float dur]

theorem runningRow_len {F : Type} (r : Int) (g q : String) (d : F) (b : Bool) :
    (runningRow r g q d b).length = 10 := rfl
theorem waitingRow_len {F : Type} (r : Int) (g q : String) (d : F) :
    (waitingRow r g q d).length = 8 := rfl

/-- One row per registered query (`snapshot_*` then `map`). -/
def runningReply {F : Type} (dur : Q → F) (r : Reg) : List (List (RV F)) :=
  (snapshotRunning r).map fun q => runningRow 0 (String.ofList q.graph) (String.ofList q.query) (dur q) false
theorem runningReply_len {F : Type} (dur : Q → F) (r : Reg) :
    (runningReply dur r).length = (snapshotRunning r).length := by simp [runningReply]

/-- `hold_batch_graphs`: distinct graph addresses in first-seen order, kept if still alive
(one upgrade per address). -/
def distinctAlive (alive : Nat → Bool) : List Nat → List Nat → List Nat
  | _, [] => []
  | seen, g :: gs =>
    if g ∈ seen then distinctAlive alive seen gs
    else if alive g then g :: distinctAlive alive (g :: seen) gs
    else distinctAlive alive (g :: seen) gs

theorem distinctAlive_nodup (alive : Nat → Bool) (gs seen : List Nat) :
    (distinctAlive alive seen gs).Nodup ∧ ∀ g ∈ distinctAlive alive seen gs, g ∉ seen ∧ alive g := by
  induction gs generalizing seen with
  | nil => simp [distinctAlive]
  | cons g gs ih =>
    unfold distinctAlive
    split
    · exact ih seen
    · rename_i hg
      split
      · rename_i ha
        obtain ⟨h1, h2⟩ := ih (g :: seen)
        refine ⟨List.nodup_cons.2 ⟨fun hm => (h2 g hm).1 (by simp), h1⟩, ?_⟩
        intro x hx; simp at hx; rcases hx with rfl | hx
        · exact ⟨hg, ha⟩
        · have := h2 x hx; simp at this; exact ⟨this.1.2, this.2⟩
      · obtain ⟨h1, h2⟩ := ih (g :: seen)
        exact ⟨h1, fun x hx => by have := h2 x hx; simp at this; exact ⟨this.1.2, this.2⟩⟩

/-- Every alive graph of the batch is held. -/
theorem distinctAlive_complete (alive : Nat → Bool) (gs seen : List Nat) (g : Nat)
    (hg : g ∈ gs) (hs : g ∉ seen) (ha : alive g) : g ∈ distinctAlive alive seen gs := by
  induction gs generalizing seen with
  | nil => simp at hg
  | cons x xs ih =>
    unfold distinctAlive
    by_cases hx : x = g
    · subst hx; simp [hs, ha]
    · have hg' : g ∈ xs := by simp at hg; rcases hg with h | h; exact absurd h.symm hx; exact h
      split
      · exact ih seen hg' hs
      · split
        · exact List.mem_cons_of_mem _ (ih _ hg' (by simp [hs, Ne.symm hx]))
        · exact ih _ hg' (by simp [hs, Ne.symm hx])

/-- `release_batch_graphs`: the drop is moved off-thread exactly for the last holder. -/
def releaseSpawns (strong : List Nat) : List Bool := strong.map (· = 1)
theorem releaseSpawns_len (s : List Nat) : (releaseSpawns s).length = s.length := by
  simp [releaseSpawns]

/-- `key_holds_graph`: the key holds a graph value whose allocation is this one. -/
def keyHolds (keyValue : Option Nat) (g : Nat) : Bool := keyValue == some g
theorem keyHolds_iff (v : Option Nat) (g : Nat) : keyHolds v g = true ↔ v = some g := by
  simp [keyHolds]

/-! ## Start / stop -/

structure Flusher where
  sender : Bool
  names : Bool
  thread : Bool

/-- `start_flusher_thread`: a second call is a no-op. -/
def start (f : Flusher) : Flusher :=
  if f.sender then f else ⟨true, true, true⟩

theorem start_idem (f : Flusher) : start (start f) = start f := by
  unfold start; split <;> simp_all

inductive Wait | sent | dropped | timeout

/-- `shutdown_flusher_thread`: only a timeout means "still running". -/
def stopped (exit : Option Wait) : Bool :=
  match exit with
  | none => true
  | some .timeout => false
  | some _ => true

/-- Field-name strings are freed only after a confirmed stop. -/
def shutdown (f : Flusher) (exit : Option Wait) : Flusher :=
  if stopped exit then ⟨false, false, false⟩ else { f with sender := false, thread := false }

theorem shutdown_keeps_names_if_parked (f : Flusher) :
    (shutdown f (some .timeout)).names = f.names := rfl
theorem shutdown_frees_on_stop (f : Flusher) (w : Wait) (h : stopped (some w)) :
    (shutdown f (some w)).names = false := by simp [shutdown, h]
theorem shutdown_closes_sender (f : Flusher) (e : Option Wait) : (shutdown f e).sender = false := by
  unfold shutdown; split <;> rfl

/-! ## One flusher iteration (`flusher_loop`, since #3844) -/

inductive Act
  | exitNoWrite          -- disconnected: leave without taking the GIL
  | skip                 -- nothing alive to write
  | write                -- delete freed streams, resolve, group, `stream_entries`
  deriving DecidableEq, Repr

/-- After the batch is collected. Since #3844 there is no role or pause check: the stream
is node-local and written through the key API, which propagates nothing. -/
def iter (disconnected anyAlive : Bool) : Act :=
  if disconnected then .exitNoWrite
  else if !anyAlive then .skip
  else .write

/-- Shutdown cannot deadlock on the GIL: a disconnected flusher never writes. -/
theorem iter_disconnected (a : Bool) : iter true a = .exitNoWrite := rfl
/-- Writing happens only connected, with a live graph — whatever the role. -/
theorem iter_write_iff (d a : Bool) :
    iter d a = .write ↔ d = false ∧ a = true := by
  cases d <;> cases a <;> decide

/-- What one iteration does under the GIL. -/
inductive Step | gil | deleteFreed | resolve | write
  deriving DecidableEq, Repr

/-- `timeout`: `recv_timeout` timed out; then only the freed-stream queue is
served, and not once the channel is closed. Otherwise the batch arm.
`pending` is `FREED_GRAPH_STREAMS.is_pending()`. -/
def iterSteps (timeout disconnected anyAlive pending : Bool) : List Step :=
  if timeout then (if pending && !disconnected then [.gil, .deleteFreed] else [])
  else match iter disconnected anyAlive with
    | .exitNoWrite => []
    | .skip => if pending then [.gil, .deleteFreed] else []
    | .write => [.gil, .deleteFreed, .resolve, .write]

/-- A disconnected flusher never takes the GIL (shutdown holds it while it waits). -/
theorem iterSteps_disconnected (t a p : Bool) : Step.gil ∉ iterSteps t true a p := by
  cases t <;> cases a <;> cases p <;> decide

/-- Queued freed streams are deleted before any write of the same GIL section, so a graph
recreated under a freed graph's key never has its new entries deleted. -/
theorem iterSteps_delete_before_write (t d a p : Bool) (h : Step.write ∈ iterSteps t d a p) :
    iterSteps t d a p = [.gil, .deleteFreed, .resolve, .write] := by
  cases t <;> cases d <;> cases a <;> cases p <;> simp_all [iterSteps, iter]

/-- A connected flusher that takes the GIL with streams pending deletes them. -/
theorem iterSteps_serves_pending (t a : Bool) :
    Step.deleteFreed ∈ iterSteps t false a true := by
  cases t <;> cases a <;> decide

namespace Pre3844
/-- Historical (before #3844): the iteration also discarded the batch on a replica and held
it through a replica pause. -/
inductive Act
  | exitNoWrite | skip | discard | hold | write
  deriving DecidableEq, Repr

def iter (disconnected anyAlive isSlave paused : Bool) : Act :=
  if disconnected then .exitNoWrite
  else if !anyAlive then .skip
  else if isSlave then .discard
  else if paused then .hold
  else .write

theorem iter_replica_never_writes (d a p : Bool) : iter d a true p ≠ .write := by
  unfold iter; split <;> (try split) <;> simp_all

/-- `set_is_replica` (removed by #3844). -/
def setReplica (_old : Bool) (b : Bool) : Bool := b
theorem setReplica_eq (o b : Bool) : setReplica o b = b := rfl
end Pre3844

end RedisLayer.Telemetry
