/-!
# `src/query_session.rs` — the per-query lock/GIL protocol

Single-thread view of one session; the RwLock and the Redis GIL enter as *events* (the
lock implementations themselves are parking_lot / Redis — AXIOMATISED rows for
`lock_gil`/`release_gil`, which only call `RedisModule_ThreadSafeContext{Lock,Unlock}`).

| here | there |
| --- | --- |
| `Th`, `gilAcquire`, `gilDrop` | `Gil::acquire` `:506`, `Gil::drop` `:529`, `hold_gil` `:484`, `GIL_CTX` |
| `S`, `begin`, `beginWriter`   | `QuerySession::begin` `:159`, `begin_with` `:166`, `begin_writer` `:182` |
| `release`                     | `release_locks` `:303`, `Drop` `:464` |
| `withGraph`, `withGraphMut`   | `with_graph` `:320`, `with_graph_mut` `:336` |
| `escalate`, `upgrade`         | `escalate` `:379`, `upgrade_to_write` `:354` (and the trait shim `:454`) |
| `reauthorize`                 | `reauthorize_write` `:413` |
| `Slot`, `SS`, `writeVersion`, `releaseSlot`, `committed`, `holdsSlot`, `isStale`, `isWriter`, `claimFork`, `upgradeS` | #3290: `enum Slot` `:110`, `write_version` `:216`, `release_slot` `:234`, `committed` `:245`, `holds_slot` `:251`, `is_stale` `:257`, `is_writer` `:261`, `claim_fork` `:267`, `facts` `:201` |
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

/-- `WriteAbort` (query_session.rs:50). `writeSlotBusy` and `invalid` (#2846) are
produced by callers of `MvccGraph::write`/`commit`, not by `upgrade_to_write`. -/
inductive WriteAbort | replicaTrafficPaused | notAMaster | graphUnregistered | writeSlotBusy
  | invalid (msg : String)
  deriving DecidableEq, Repr

/-- `Display` (thiserror): `Invalid(e)` is `"{0}"`, the refusal verbatim. -/
def writeAbortMsg : WriteAbort → String
  | .replicaTrafficPaused => "Write query aborted: replica traffic is currently paused"
  | .notAMaster => "Write query aborted: this instance is not a master"
  | .graphUnregistered => "graph was deleted or replaced while the query was running, aborting"
  | .writeSlotBusy => "Write query aborted: another write is in progress"
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

/-! ## The session's write slot (#3290, `66d4d9d40`)

`MvccGraph` enters as its two fields that matter here: `slotTaken` (the `AtomicBool`) and
the committed `ver`sion; `fork()` is the version `ver + 1` (`new_version`). -/

inductive Slot | none | forked | claimed | stale
  deriving DecidableEq, Repr

structure SS where
  writer : Bool          -- `is_writer` (`:261`): `Mode::Writer`
  slot : Slot
  fork : Option Nat      -- the fork's `version`
  slotTaken : Bool       -- `MvccGraph.write`
  ver : Nat              -- committed `Graph::version`
  deriving DecidableEq, Repr

inductive Abort | busy | stale
  deriving DecidableEq, Repr

/-- `write_version` (`:216-231`): a writer claims the slot now (`MvccGraph::write`), a reader
takes an unclaimed fork (`MvccGraph::fork`). -/
def writeVersion (s : SS) : Except Abort SS :=
  if s.writer then
    if s.slotTaken then .error .busy else .ok { s with slotTaken := true, slot := .claimed }
  else .ok { s with fork := some (s.ver + 1), slot := .forked }

/-- `release_slot` (`:234-242`): rollback if claimed, drop the fork, and reset unless stale. -/
def releaseSlot (s : SS) : SS :=
  let s1 := if s.slot = .claimed then { s with slotTaken := false } else s
  let s2 := { s1 with fork := none }
  if s.slot ≠ .stale then { s2 with slot := .none } else s2

/-- `committed` (`:245`): `MvccGraph::commit` released the slot. -/
def committed (s : SS) : SS := { s with slot := .none }
def holdsSlot (s : SS) : Bool := s.slot == .claimed
def isStale (s : SS) : Bool := s.slot == .stale

/-- `claim_fork` (`:267-292`) → `MvccGraph::claim` (`mvcc_graph.rs:151-167`). -/
def claimFork (s : SS) : Except Abort SS :=
  if s.slot ≠ .forked then .ok s else
  match s.fork with
  | none => .ok s                                   -- `expect` (a fork for `Forked`): unreachable
  | some fv =>
    if s.slotTaken then .error .busy                -- `Claim::Busy`: fork put back
    else if s.ver + 1 ≠ fv then .error .stale     -- state left: `claimForkStale`
    else .ok { s with slotTaken := true, slot := .claimed, fork := none }

/-- The state `claim_fork` leaves on `Stale`: the CAS was undone (`store(false)`), the
fork dropped, `Slot::Stale`. -/
def claimForkStale (s : SS) : SS := { s with slot := .stale, fork := none }

/-- `upgrade_to_write` since #3290: escalate; then registry and role checks; then
`claim_fork`, under the write lock. -/
def upgradeS (s : SS) (registered : Bool) (auth : Option WriteAbort) : Option WriteAbort ⊕ Abort :=
  if s.writer then .inl none
  else if !registered then .inl (some .graphUnregistered)
  else match auth with
    | some a => .inl (some a)
    | none => match claimFork { s with writer := true } with
      | .ok _ => .inl none
      | .error a => .inr a

/-- A reader's write version takes no slot: an inline writer is never refused because a
queued write is matching (#3093, fixed by #3290). -/
theorem reader_takes_no_slot (s : SS) (h : s.writer = false) :
    ∃ s', writeVersion s = .ok s' ∧ s'.slotTaken = s.slotTaken ∧ s'.slot = .forked := by
  simp [writeVersion, h]

/-- A writer claims at once, or is refused without taking anything. -/
theorem writer_claims (s : SS) (h : s.writer = true) :
    (s.slotTaken = false → ∃ s', writeVersion s = .ok s' ∧ s'.slotTaken ∧ holdsSlot s') ∧
    (s.slotTaken = true → writeVersion s = .error .busy) := by
  constructor
  · intro h2; simp [writeVersion, h, h2, holdsSlot]
  · intro h2; simp [writeVersion, h, h2]

/-- **A claim succeeds only on a fork of the current version**: then the slot is held and the
private version extends the committed one. If anything committed since the fork, the claim is
`Stale`. -/
theorem claimFork_spec (s : SS) (fv : Nat) (hf : s.slot = .forked) (hfk : s.fork = some fv)
    (ht : s.slotTaken = false) :
    (fv = s.ver + 1 → claimFork s = .ok { s with slotTaken := true, slot := .claimed, fork := none }) ∧
    (fv ≠ s.ver + 1 → claimFork s = .error .stale) := by
  constructor
  · intro e; subst e; simp [claimFork, hf, hfk, ht]
  · intro e; simp [claimFork, hf, hfk, ht]; intro h; exact absurd h.symm e

/-- A non-forked session's claim is a no-op (a writer from `begin_writer` already holds it;
`attempt_settle` never forks). -/
theorem claimFork_noop (s : SS) (h : s.slot ≠ .forked) : claimFork s = .ok s := by
  simp [claimFork, h]

/-- `release_slot` frees the MVCC slot exactly when this session held it, and is
idempotent. -/
theorem releaseSlot_spec (s : SS) :
    (releaseSlot s).slotTaken = (if s.slot = .claimed then false else s.slotTaken) ∧
    (releaseSlot s).fork = none ∧ releaseSlot (releaseSlot s) = releaseSlot s := by
  unfold releaseSlot
  cases h : s.slot <;> simp [h]

/-- Stale survives `release_slot`, so `is_stale` still reports it to the retry loop. -/
theorem stale_sticky (s : SS) (h : s.slot = .stale) : isStale (releaseSlot s) = true := by
  simp [releaseSlot, isStale, h]

/-- `upgrade_to_write` reports `Stale` exactly from the claim. -/
theorem upgrade_stale (s : SS) (fv : Nat) (hw : s.writer = false) (hf : s.slot = .forked)
    (hfk : s.fork = some fv) (ht : s.slotTaken = false) (hv : fv ≠ s.ver + 1) :
    upgradeS s true none = .inr .stale := by
  have := (claimFork_spec { s with writer := true } fv hf hfk ht).2 hv
  simp [upgradeS, hw, this]

end RedisLayer.QuerySession
