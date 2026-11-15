/-!
# `src/query_session.rs` — the per-query lock/GIL protocol

Single-thread view of one session; the RwLock and the Redis GIL enter as *events* (the
lock implementations themselves are parking_lot / Redis — AXIOMATISED rows for
`lock_gil`/`release_gil`, which only call `RedisModule_ThreadSafeContext{Lock,Unlock}`).

| here | there |
| --- | --- |
| `Th`, `gilAcquire`, `gilDrop` | `Gil::acquire` `:472`, `Gil::drop` `:495`, `hold_gil` `:450`, `GIL_CTX` |
| `S`, `begin`, `beginWriter`   | `QuerySession::begin` `:158`, `begin_with` `:165`, `begin_writer` `:180` |
| `release`                     | `release_locks` `:268`, `Drop` `:430` |
| `withGraph`, `withGraphMut`   | `with_graph` `:285`, `with_graph_mut` `:301` |
| `escalate`, `upgrade`         | `escalate` `:345`, `upgrade_to_write` `:319` (and the trait shim `:420`) |
| `reauthorize`                 | `reauthorize_write` `:379` |
| `Version`, `SS`, `writeVersion`, `isStale`, `versionIsCurrent`, `checkFork`, `upgradeS`, `upgradeSt` | #3657: `enum Version` `:108`, `write_version` `:211`, `is_stale` `:227`, `version_is_current` `:235`, `is_writer` `:239`, `check_fork` `:246`, `facts` `:198` |
| `CommitError`, `WriteAbort.ofCommit` | `From<CommitError> for WriteAbort` `:74` |
-/
namespace RedisLayer.QuerySession

inductive Ev | readLock | readUnlock | writeLock | writeUnlock | gilLock | gilUnlock
  deriving DecidableEq, Repr

/-- Thread-local state: main thread?, `GIL_CTX` set?, and the event log. -/
structure Th where
  main : Bool
  ctx : Bool
  log : List Ev

/-- `Gil::acquire`: a no-op guard on the main thread or when this thread already holds it. -/
def gilAcquire (t : Th) : Bool × Th :=
  if t.main || t.ctx then (false, t) else (true, { t with ctx := true, log := t.log ++ [.gilLock] })

/-- `Gil::drop`. -/
def gilDrop (needRelease : Bool) (t : Th) : Th :=
  if !needRelease then t
  else if t.ctx then { t with ctx := false, log := t.log ++ [.gilUnlock] } else t

/-- Nested guards: the inner one neither locks nor unlocks. -/
theorem gil_nested (t : Th) (h : t.ctx = true) : gilAcquire t = (false, t) := by
  simp [gilAcquire, h]

theorem gil_stack (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    let (o, t1) := gilAcquire t
    let (i, t2) := gilAcquire t1
    let t3 := gilDrop i t2
    t3.ctx = true ∧ (gilDrop o t3).ctx = false ∧
      (gilDrop o t3).log = t.log ++ [.gilLock, .gilUnlock] := by
  simp [gilAcquire, gilDrop, hm, hc]

/-- The main thread never touches the lock (it already holds it). -/
theorem gil_main (t : Th) (hm : t.main = true) : gilAcquire t = (false, t) := by
  simp [gilAcquire, hm]

/-- Out-of-order drops release the GIL under a still-live inner guard: correctness relies
on guards being dropped in stack order (Rust scoping guarantees this for `hold_gil()`
temporaries; the session's writer guard must not outlive an enclosing `hold_gil`). -/
theorem gil_out_of_order (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    let (o, t1) := gilAcquire t
    let (_i, t2) := gilAcquire t1
    (gilDrop o t2).ctx = false := by
  simp [gilAcquire, gilDrop, hm, hc]

inductive Mode | reader | writer (needRelease : Bool)
  deriving DecidableEq, Repr

structure S where
  mode : Option Mode
  th : Th

def begin (t : Th) : S := ⟨some .reader, { t with log := t.log ++ [.readLock] }⟩

def beginWriter (t : Th) : S :=
  let (nr, t) := gilAcquire t
  ⟨some (.writer nr), { t with log := t.log ++ [.writeLock] }⟩

/-- Dropping the mode: `Mode::Writer { guard, _gil }` drops `guard` (declared first) and
then `_gil`, so the write lock goes before the GIL. -/
def dropMode (m : Option Mode) (t : Th) : Th :=
  match m with
  | none => t
  | some .reader => { t with log := t.log ++ [.readUnlock] }
  | some (.writer nr) => gilDrop nr { t with log := t.log ++ [.writeUnlock] }

def release (s : S) : S := ⟨none, dropMode s.mode s.th⟩

/-- `with_graph`: `expect("session holds no lock")`. -/
def withGraph (s : S) : Option Unit := s.mode.map fun _ => ()
/-- `with_graph_mut`: only a writer. -/
def withGraphMut (s : S) : Bool :=
  match s.mode with
  | some (.writer _) => true
  | _ => false

/-- `escalate`: a writer stays; otherwise drop the read lock *first*, then GIL, then write. -/
def escalate (s : S) : Bool × S :=
  match s.mode with
  | some (.writer _) => (false, s)
  | m =>
    let t := dropMode m s.th
    let (nr, t) := gilAcquire t
    (true, ⟨some (.writer nr), { t with log := t.log ++ [.writeLock] }⟩)

/-- `WriteAbort` (query_session.rs:52). `stale` comes from two places: `check_fork`'s refusal and
`commit`'s stale refusal (`From<CommitError>`, `ofCommit`); `invalid` (#2846) is `commit`'s only. `WriteSlotBusy` is gone with the write slot (#3657). -/
inductive WriteAbort | replicaTrafficPaused | notAMaster | graphUnregistered | stale
  | invalid (msg : String)
  deriving DecidableEq, Repr

/-- `Display` (thiserror): `Invalid(e)` is `"{0}"`, the refusal verbatim. -/
def writeAbortMsg : WriteAbort → String
  | .replicaTrafficPaused => "Write query aborted: replica traffic is currently paused"
  | .notAMaster => "Write query aborted: this instance is not a master"
  | .graphUnregistered => "graph was deleted or replaced while the query was running, aborting"
  | .stale => "Write query aborted: the graph was written while this query was matching"
  | .invalid m => m

theorem writeAbortMsg_invalid (m : String) : writeAbortMsg (.invalid m) = m := rfl

structure Facts where
  replicates : Bool
  originatedHere : Bool

/-- `reauthorize_write`: no thread-local GIL context (main thread) → nothing to check. -/
def reauthorize (f : Facts) (ctx : Option (Bool × Bool)) : Option WriteAbort :=
  match ctx with
  | none => none
  | some (paused, readonly) =>
    if f.replicates && paused then some .replicaTrafficPaused
    else if f.originatedHere && readonly then some .notAMaster
    else none

def upgrade (s : S) (registered : Bool) (ctx : Option (Bool × Bool)) (f : Facts) :
    Option WriteAbort × S :=
  let (esc, s) := escalate s
  if esc then
    if !registered then (some .graphUnregistered, s) else (reauthorize f ctx, s)
  else (none, s)

theorem escalate_mode (s : S) : ∃ nr, (escalate s).2.mode = some (.writer nr) := by
  unfold escalate
  split
  · rename_i nr h; exact ⟨nr, h⟩
  · exact ⟨_, rfl⟩

/-- After `escalate` the session is a writer and may mutate. -/
theorem escalate_writer (s : S) : withGraphMut (escalate s).2 = true := by
  obtain ⟨nr, h⟩ := escalate_mode s
  simp [withGraphMut, h]

theorem escalate_of_writer (s : S) (nr : Bool) (h : s.mode = some (.writer nr)) : escalate s = (false, s) := by
  simp [escalate, h]

theorem upgrade_state (s : S) (r : Bool) (c : Option (Bool × Bool)) (f : Facts) :
    (upgrade s r c f).2 = (escalate s).2 := by
  unfold upgrade
  obtain ⟨e, s'⟩ := escalate s
  simp only
  split <;> (try split) <;> rfl

/-- Escalating a reader releases the read lock before waiting for the GIL (no
read-lock-held-while-blocking-on-GIL, which could deadlock against a GIL holder waiting
for the write lock). -/
theorem escalate_order (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    (escalate (begin t)).2.th.log = t.log ++ [.readLock, .readUnlock, .gilLock, .writeLock] := by
  simp [escalate, begin, dropMode, gilAcquire, hm, hc]

/-- A writer always holds the GIL (on a worker thread). -/
theorem writer_holds_gil (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    (beginWriter t).th.ctx = true ∧ (escalate (begin t)).2.th.ctx = true := by
  simp [beginWriter, escalate, begin, dropMode, gilAcquire, hm, hc]

/-- Releasing a writer unlocks the write lock before the GIL. -/
theorem release_order (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    (release (beginWriter t)).th.log = t.log ++ [.gilLock, .writeLock, .writeUnlock, .gilUnlock] := by
  simp [release, beginWriter, dropMode, gilAcquire, gilDrop, hm, hc]

/-- A second escalation is free and re-checks nothing. -/
theorem upgrade_twice (s : S) (r : Bool) (c : Option (Bool × Bool)) (f : Facts) :
    (upgrade (upgrade s r c f).2 r c f).1 = none := by
  rw [upgrade_state]
  obtain ⟨nr, h⟩ := escalate_mode s
  simp [upgrade, escalate_of_writer _ nr h]

/-- A deleted/replaced graph aborts before any role check. -/
theorem upgrade_unregistered (t : Th) (c : Option (Bool × Bool)) (f : Facts) :
    (upgrade (begin t) false c f).1 = some .graphUnregistered := by
  simp [upgrade, escalate, begin]

/-- Precedence: replica pause, then not-a-master. -/
theorem reauth_paused (o ro : Bool) : reauthorize ⟨true, o⟩ (some (true, ro)) = some .replicaTrafficPaused := by
  simp [reauthorize]
theorem reauth_replica (ro : Bool) (h : ro = true) :
    reauthorize ⟨false, true⟩ (some (true, ro)) = some .notAMaster := by
  simp [reauthorize, h]
/-- Replayed writes (`originated_here = false`, e.g. `GRAPH.EFFECT` on a replica) pass. -/
theorem reauth_replay : reauthorize ⟨false, false⟩ (some (true, true)) = none := rfl
theorem reauth_main (f : Facts) : reauthorize f none = none := rfl

/-- `Drop` releases whatever is held. -/
theorem release_none (s : S) : (release s).mode = none := rfl
theorem withGraph_released (s : S) : withGraph (release s) = none := rfl

/-- `graph_arc`: the session's graph handle, unchanged. -/
def graphArc (g : Nat) : Nat := g
theorem graphArc_id (g : Nat) : graphArc g = g := rfl

/-! ## The session's write version (#3657, `34a708cfa`)

`MvccGraph` enters as its committed `ver`sion only; `fork()` is the version `ver + 1`
(`new_version`) and `is_stale` is `ver + 1 != fork.version`. There is no slot to claim or
release: an uncommitted version is dropped with the runtime. -/

/-- `enum Version` (`:108`). `unchecked` holds the fork's `version`. -/
inductive Version | none | unchecked (fv : Nat) | current | stale
  deriving DecidableEq, Repr

structure SS where
  writer : Bool          -- `is_writer` (`:239`): `Mode::Writer`
  version : Version
  ver : Nat              -- published `Graph::version`
  deriving DecidableEq, Repr

/-- `write_version` (`:211-224`): always a fork (`MvccGraph::fork`, version `ver + 1`); in writer
mode it is `Current` from the start, in reader mode `Unchecked`. Infallible. -/
def writeVersion (s : SS) : SS :=
  { s with version := if s.writer then .current else .unchecked (s.ver + 1) }

def isStale (s : SS) : Bool := s.version == .stale
/-- `version_is_current` (`:235`). -/
def versionIsCurrent (s : SS) : Bool := s.version == .current

/-- `check_fork` (`:246`): only an `Unchecked` fork is looked at; `MvccGraph::is_stale` fork
(`mvcc_graph.rs:144`) is `ver + 1 ≠ fork.version`. Stale ⇒ `Version::Stale` and `Err(Stale)`.
Since #3567 an unchecked fork is also rebased onto the indexer batches published since, as it
becomes current (inside the `Unchecked` arm: `rebase_onto_indexer_batches`, `query_session.rs:261`; the
`Current`/`Stale`/`None` arms return at once): index progress only, no data change
(native_index `Rebase.lean`: `V.rebased_keeps_data`). The rebase fails (`Err(Stale)`, release) only for a
fork that is still borrowed, which an escalating writer's is not (a debug build panics): that is the one
assumption this model makes, not proven here. -/
def checkFork (s : SS) : Except Unit SS :=
  match s.version with
  | .unchecked fv =>
    if s.ver + 1 ≠ fv then .error () else .ok { s with version := .current }
  | _ => .ok s

/-- The state `check_fork` leaves on `Stale`: the fork dropped, `Version::Stale`. -/
def checkForkStale (s : SS) : SS := { s with version := .stale }

/-- `upgrade_to_write` since #3657: escalate; then registry and role checks; then
`check_fork`, under the write lock. `inr` is `WriteAbort::Stale`. -/
def upgradeS (s : SS) (registered : Bool) (auth : Option WriteAbort) : Option WriteAbort ⊕ Unit :=
  if s.writer then .inl none
  else if !registered then .inl (some .graphUnregistered)
  else match auth with
    | some a => .inl (some a)
    | none => match checkFork { s with writer := true } with
      | .ok _ => .inl none
      | .error _ => .inr ()

/-- `upgradeS` together with the state the session is left in: on `Stale` the version is
`Version::Stale` (`check_fork` sets it before returning `Err(Stale)`), which the retry loop reads
through `is_stale`; on success it is what `check_fork` left. -/
def upgradeSt (s : SS) (registered : Bool) (auth : Option WriteAbort) :
    (Option WriteAbort ⊕ Unit) × SS :=
  if s.writer then (.inl none, s)
  else if !registered then (.inl (some .graphUnregistered), s)
  else match auth with
    | some a => (.inl (some a), s)
    | none => match checkFork { s with writer := true } with
      | .ok t => (.inl none, t)
      | .error _ => (.inr (), checkForkStale { s with writer := true })

theorem upgradeSt_fst (s : SS) (registered : Bool) (auth : Option WriteAbort) :
    (upgradeSt s registered auth).1 = upgradeS s registered auth := by
  unfold upgradeSt upgradeS
  split
  · rfl
  · split
    · rfl
    · cases auth with
      | some a => rfl
      | none => cases checkFork { s with writer := true } <;> rfl

/-- **A stale refusal leaves the retry-visible state**: when `upgrade_to_write` reports `Stale`, the
session is `Version::Stale`, so `is_stale` is true and the retry loop reruns the write. -/
theorem upgradeSt_stale_visible (s : SS) (registered : Bool) (auth : Option WriteAbort) (e : Unit)
    (h : (upgradeSt s registered auth).1 = .inr e) :
    isStale (upgradeSt s registered auth).2 = true := by
  by_cases hw : s.writer = true
  · simp [upgradeSt, hw] at h
  · by_cases hr : registered = false
    · simp [upgradeSt, hw, hr] at h
    · cases auth with
      | some a => simp [upgradeSt, hw, hr] at h
      | none =>
        cases hc : checkFork { s with writer := true } with
        | ok t => simp [upgradeSt, hw, hr, hc] at h
        | error u => simp [upgradeSt, hw, hr, hc, isStale, checkForkStale]

/-- A write version takes nothing, in either mode: an inline writer is never refused because a
queued write is matching (#3093, #3290), and no version needs releasing (#3657). A
writer's is `Current` at once. -/
theorem writeVersion_takes_nothing (s : SS) :
    (writeVersion s).ver = s.ver ∧ (writeVersion s).writer = s.writer ∧
    (s.writer = true → versionIsCurrent (writeVersion s) = true) ∧
    (s.writer = false → (writeVersion s).version = .unchecked (s.ver + 1)) := by
  cases h : s.writer <;> simp [writeVersion, versionIsCurrent, h]

/-- **A check succeeds only on a fork of the current version**: then the version is
`Current` and extends the published one. If anything committed since the fork, it is
`Stale`. -/
theorem checkFork_spec (s : SS) (fv : Nat) (hf : s.version = .unchecked fv) :
    (fv = s.ver + 1 → checkFork s = .ok { s with version := .current }) ∧
    (fv ≠ s.ver + 1 → checkFork s = .error ()) := by
  constructor
  · intro e; subst e; simp [checkFork, hf]
  · intro e; simp [checkFork, hf]; intro h; exact absurd h.symm e

/-- A session that is not `Unchecked` is unchanged by the check (a writer from `begin_writer`
is `Current` already; `attempt_settle` forks under the write lock itself). -/
theorem checkFork_noop (s : SS) (h : ∀ fv, s.version ≠ .unchecked fv) : checkFork s = .ok s := by
  unfold checkFork; split
  · rename_i fv hfv; exact absurd hfv (h fv)
  · rfl

/-- `Stale` is sticky: no step but a new `write_version` leaves it, so `is_stale` still
reports it to the retry loop. -/
theorem stale_sticky (s : SS) (h : s.version = .stale) : isStale (checkForkStale s) = true ∧ checkFork s = .ok s := by
  simp [checkForkStale, isStale, checkFork, h]

/-- **`Current` is never refused as stale**: a writer session is `Current` from its fork, and
`check_fork` keeps it so, so `commit` (which refuses only a stale fork) cannot refuse it —
`commit_and_replicate`'s `unreachable!`. -/
theorem current_kept (s : SS) (h : versionIsCurrent s = true) : checkFork s = .ok s := by
  have : s.version = .current := by simpa [versionIsCurrent] using h
  simp [checkFork, this]

/-- `upgrade_to_write` reports `Stale` exactly from the check. -/
theorem upgrade_stale (s : SS) (fv : Nat) (hw : s.writer = false) (hf : s.version = .unchecked fv)
    (hv : fv ≠ s.ver + 1) :
    upgradeS s true none = .inr () := by
  have := (checkFork_spec { s with writer := true } fv hf).2 hv
  simp [upgradeS, hw, this]

/-- `From<CommitError> for WriteAbort` (`:74`). -/
inductive CommitError | stale | invalid (msg : String)
  deriving DecidableEq, Repr

def WriteAbort.ofCommit : CommitError → WriteAbort
  | .stale => .stale
  | .invalid m => .invalid m

theorem ofCommit_stale : WriteAbort.ofCommit .stale = .stale := rfl
theorem ofCommit_invalid (m : String) : WriteAbort.ofCommit (.invalid m) = .invalid m := rfl
/-- The two refusals map to different aborts, which the retry loops tell apart. -/
theorem ofCommit_ne (m : String) : WriteAbort.ofCommit .stale ≠ WriteAbort.ofCommit (.invalid m) := by
  simp [WriteAbort.ofCommit]

end RedisLayer.QuerySession
