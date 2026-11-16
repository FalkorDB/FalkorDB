/-!
# `src/query_session.rs` — the per-query lock/GIL protocol

Single-thread view of one session; the RwLock, the Redis GIL and the per-graph
`WriterGate` enter as *events* (the lock implementations themselves are parking_lot /
Redis — AXIOMATISED rows for `lock_gil`/`release_gil`, which only call
`RedisModule_ThreadSafeContext{Lock,Unlock}`). Revision: PR #3816 (af228a34f).

| here | there |
| --- | --- |
| `Th`, `holdsGil`, `gilAcquire`, `gilDrop` | `holds_gil` `:681`, `Gil::acquire` `:701`, `Gil::drop` `:724`, `hold_gil` `:674`, `GIL_CTX` |
| `S`, `begin`, `beginWriter`   | `QuerySession::begin` `:178`, `begin_writer` `:205` (main thread only since #3816: `debug_assert!(holds_gil())`) |
| `beginWith`, `readPastGate`, `mayWaitAtGate` | `begin_with` `:185`, `read_past_gate` `:524`, `may_wait_at_gate` `:548`; `WriterGate::is_closed`/`close`/`wait_until_open` `:569-577` are the `gateWait`/`gateClose`/`gateOpen` events |
| `release`                     | `release_locks` `:317`, `Drop` `:652` |
| `withGraph`, `withGraphMut`   | `with_graph` `:334`, `with_graph_mut` `:350` |
| `Att`, `pollLoop`, `acquireWrite` | `acquire_write` `:460` (#3816): per attempt GIL, check, `try_write_arc_for(slice)`; the gate closes after `GATE_AFTER_SLICES` = 2 misses; one blocking try when the interval is ≤ 0 or the thread already holds the GIL |
| `authorize`, `World`          | `authorize_write` `:496`: registry, then `reauthorize_write` |
| `escalate`, `upgrade`         | `escalate` `:399`, `upgrade_to_write` `:377` (and the trait shim `:639`) |
| `reauthorize`                 | `reauthorize_write` `:598` |
| `Version`, `SS`, `writeVersion`, `isStale`, `versionIsCurrent`, `checkFork`, `upgradeS`, `upgradeSt` | #3657: `enum Version` `:128`, `write_version` `:239`, `is_stale` `:255`, `version_is_current` `:263`, `is_writer` `:267`, `check_fork` `:279`, `facts` `:225` |
| `CommitError`, `WriteAbort.ofCommit` | `From<CommitError> for WriteAbort` `:95` |

**#3816 (af228a34f).** Off the main thread `escalate` waits for the write lock in
`WRITE_LOCK_POLL_INTERVAL` slices, giving the GIL back between them, and runs the
registry/role check (`authorize_write`) under each slice's GIL hold, *before* the lock is
taken. A refusal leaves the session a **reader** again (read lock re-taken), so a second
`upgrade_to_write` re-runs every check (`upgrade_refused_rechecks`); before, the refused
session stayed a writer and a second call reported success unchecked
(`Pre3816.pre3816_upgrade_twice_skips_checks`, the hazard `attempt_settle`'s old comment
warned about). `escalate_writer` / `upgrade_twice` therefore now speak of a *successful*
escalation; their old unconditional forms are kept as `Pre3816.escalate_writer` /
`Pre3816.upgrade_twice` on the old model. `upgradeS`/`upgradeSt` already left a refused
session's `writer` flag false, as the new code does.
-/
namespace RedisLayer.QuerySession

inductive Ev | readLock | readUnlock | writeLock | writeUnlock | gilLock | gilUnlock
  | gateClose | gateOpen | gateWait
  deriving DecidableEq, Repr

/-- Thread-local state: main thread?, `GIL_CTX` set?, and the event log. -/
structure Th where
  main : Bool
  ctx : Bool
  log : List Ev

/-- `holds_gil`: the main thread, or a thread whose own context is in `GIL_CTX`. -/
def holdsGil (t : Th) : Bool := t.main || t.ctx

/-- `Gil::acquire`: a no-op guard when this thread already holds it (`holds_gil`). -/
def gilAcquire (t : Th) : Bool × Th :=
  if holdsGil t then (false, t) else (true, { t with ctx := true, log := t.log ++ [.gilLock] })

/-- `Gil::drop`. -/
def gilDrop (needRelease : Bool) (t : Th) : Th :=
  if !needRelease then t
  else if t.ctx then { t with ctx := false, log := t.log ++ [.gilUnlock] } else t

/-- Nested guards: the inner one neither locks nor unlocks. -/
theorem gil_nested (t : Th) (h : t.ctx = true) : gilAcquire t = (false, t) := by
  simp [gilAcquire, holdsGil, h]

theorem gil_stack (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    let (o, t1) := gilAcquire t
    let (i, t2) := gilAcquire t1
    let t3 := gilDrop i t2
    t3.ctx = true ∧ (gilDrop o t3).ctx = false ∧
      (gilDrop o t3).log = t.log ++ [.gilLock, .gilUnlock] := by
  simp [gilAcquire, holdsGil, gilDrop, hm, hc]

/-- The main thread never touches the lock (it already holds it). -/
theorem gil_main (t : Th) (hm : t.main = true) : gilAcquire t = (false, t) := by
  simp [gilAcquire, holdsGil, hm]

/-- Out-of-order drops release the GIL under a still-live inner guard: correctness relies
on guards being dropped in stack order (Rust scoping guarantees this for `hold_gil()`
temporaries; the session's writer guard must not outlive an enclosing `hold_gil`). -/
theorem gil_out_of_order (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    let (o, t1) := gilAcquire t
    let (_i, t2) := gilAcquire t1
    (gilDrop o t2).ctx = false := by
  simp [gilAcquire, holdsGil, gilDrop, hm, hc]

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

/-- `WriteAbort` (query_session.rs:72). `stale` comes from two places: `check_fork`'s refusal and
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

/-- `authorize_write` (`:496`): the registry first, then the role/pause re-check. -/
def authorize (registered : Bool) (ctx : Option (Bool × Bool)) (f : Facts) : Option WriteAbort :=
  if !registered then some .graphUnregistered else reauthorize f ctx

/-! ## Waiting for the write lock (#3816, `acquire_write` `:460`) -/

/-- One attempt of `acquire_write`: the verdict of `check_before_try` under that attempt's GIL
hold, and whether the write lock came within the slice. The world decides both. -/
structure Att where
  check : Option WriteAbort
  got : Bool

/-- `GATE_AFTER_SLICES` (`:430`). -/
def gateAfter : Nat := 2

/-- Dropping the `closed` gate guard (if any). -/
def openGate (closed : Bool) (t : Th) : Th :=
  if closed then { t with log := t.log ++ [.gateOpen] } else t

/-- The sliced loop: GIL, check (an error ends the wait: `gil` drops, then `closed`),
`try_write_arc_for` (got: return with both locks, `closed` drops after), else give the GIL
back, count the miss, close the gate on the `gateAfter`-th, sleep, retry. `none`: still
waiting after the given attempts. -/
def pollLoop : Nat → Bool → Th → List Att → Option (Except WriteAbort Bool × Th)
  | _, _, _, [] => none
  | missed, closed, t, a :: rest =>
    let (nr, t1) := gilAcquire t
    match a.check with
    | some e => some (.error e, openGate closed (gilDrop nr t1))
    | none =>
      if a.got then some (.ok nr, openGate closed { t1 with log := t1.log ++ [.writeLock] })
      else
        let t2 := gilDrop nr t1
        if !closed && gateAfter ≤ missed + 1 then
          pollLoop (missed + 1) true { t2 with log := t2.log ++ [.gateClose] } rest
        else pollLoop (missed + 1) closed t2 rest

/-- `acquire_write`: with the interval ≤ 0, or on a thread that already holds the GIL, one
blocking try (GIL, check, `write_arc`); otherwise the sliced loop. `.ok nr`: both locks
held, `nr` the GIL guard's `need_release`. -/
def acquireWrite (interval : Int) (t : Th) (atts : List Att) : Option (Except WriteAbort Bool × Th) :=
  if interval ≤ 0 || holdsGil t then
    match atts with
    | [] => none
    | a :: _ =>
      let (nr, t1) := gilAcquire t
      match a.check with
      | some e => some (.error e, gilDrop nr t1)
      | none => some (.ok nr, { t1 with log := t1.log ++ [.writeLock] })
  else pollLoop 0 false t atts

/-- `escalate` (`:399`): a writer stays (`Ok(false)`); otherwise drop the read lock *first*,
then `acquire_write`. On success a writer; on a refusal the read lock is re-taken directly
(not through the gate) and the session is a reader again. -/
def escalate (s : S) (interval : Int) (atts : List Att) : Option (Except WriteAbort Bool × S) :=
  match s.mode with
  | some (.writer _) => some (.ok false, s)
  | m =>
    match acquireWrite interval (dropMode m s.th) atts with
    | none => none
    | some (.ok nr, t) => some (.ok true, ⟨some (.writer nr), t⟩)
    | some (.error e, t) => some (.error e, ⟨some .reader, { t with log := t.log ++ [.readLock] }⟩)

/-- What the world shows one attempt: registered?, the GIL context's (paused, readonly)
flags (`none` on the main thread), and whether the lock came in the slice. -/
structure World where
  registered : Bool
  ctx : Option (Bool × Bool)
  got : Bool

def World.att (f : Facts) (w : World) : Att := ⟨authorize w.registered w.ctx f, w.got⟩

/-- `upgrade_to_write` up to `check_fork` (which `upgradeS` adds): escalate with
`authorize_write` as the per-attempt check. -/
def upgrade (s : S) (interval : Int) (ws : List World) (f : Facts) : Option (Option WriteAbort × S) :=
  match escalate s interval (ws.map (World.att f)) with
  | none => none
  | some (.ok _, s') => some (none, s')
  | some (.error e, s') => some (some e, s')

theorem escalate_of_writer (s : S) (nr : Bool) (i : Int) (a : List Att)
    (h : s.mode = some (.writer nr)) : escalate s i a = some (.ok false, s) := by
  simp [escalate, h]

/-- After a successful `escalate` the session is a writer. -/
theorem escalate_mode (s s' : S) (i : Int) (a : List Att) (b : Bool)
    (h : escalate s i a = some (.ok b, s')) : ∃ nr, s'.mode = some (.writer nr) := by
  unfold escalate at h
  split at h
  · rename_i nr hm; simp at h; obtain ⟨-, rfl⟩ := h; exact ⟨nr, hm⟩
  · split at h
    · simp at h
    · rename_i nr _ _; simp at h; obtain ⟨-, rfl⟩ := h; exact ⟨nr, rfl⟩
    · simp at h

/-- …so it may mutate (`with_graph_mut` is `Some`). -/
theorem escalate_writer (s s' : S) (i : Int) (a : List Att) (b : Bool)
    (h : escalate s i a = some (.ok b, s')) : withGraphMut s' = true := by
  obtain ⟨nr, h⟩ := escalate_mode s s' i a b h
  simp [withGraphMut, h]

/-- **A refused escalation leaves a reader** holding the read lock again (#3816), so
`with_graph` still works and `with_graph_mut` refuses. -/
theorem escalate_refused_reader (s s' : S) (i : Int) (a : List Att) (e : WriteAbort)
    (h : escalate s i a = some (.error e, s')) :
    s'.mode = some .reader ∧ withGraphMut s' = false ∧ s'.th.log.getLast? = some .readLock := by
  unfold escalate at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · simp at h
    · simp at h; obtain ⟨-, rfl⟩ := h; simp [withGraphMut]

theorem upgrade_state (s : S) (i : Int) (ws : List World) (f : Facts) :
    (upgrade s i ws f).map (·.2) = (escalate s i (ws.map (World.att f))).map (·.2) := by
  unfold upgrade
  split <;> rename_i h <;> simp [h]

/-- Escalating a reader releases the read lock before waiting for the GIL (no
read-lock-held-while-blocking-on-GIL, which could deadlock against a GIL holder waiting
for the write lock). Here the lock comes in the first slice. -/
theorem escalate_order (t : Th) (hm : t.main = false) (hc : t.ctx = false) (i : Int) (rest : List Att) :
    (escalate (begin t) i (⟨none, true⟩ :: rest)).map (·.2.th.log) =
      some (t.log ++ [.readLock, .readUnlock, .gilLock, .writeLock]) := by
  by_cases hi : i ≤ 0 <;>
    simp [escalate, begin, dropMode, acquireWrite, pollLoop, gilAcquire, holdsGil, openGate, hm, hc, hi]

/-- With misses: every slice takes and gives back the GIL holding no graph lock, the gate
closes after the second miss and opens once the lock is taken. -/
theorem escalate_order_slices (t : Th) (hm : t.main = false) (hc : t.ctx = false) (i : Int) (hi : 0 < i) :
    (escalate (begin t) i [⟨none, false⟩, ⟨none, false⟩, ⟨none, false⟩, ⟨none, true⟩]).map (·.2.th.log) =
      some (t.log ++ [.readLock, .readUnlock, .gilLock, .gilUnlock, .gilLock, .gilUnlock, .gateClose,
        .gilLock, .gilUnlock, .gilLock, .writeLock, .gateOpen]) := by
  have : ¬ i ≤ 0 := by omega
  simp [escalate, begin, dropMode, acquireWrite, pollLoop, gilAcquire, gilDrop, holdsGil, openGate,
    gateAfter, hm, hc, this]

/-- One miss does not close the gate (C's #3685: kept out of the common short wait). -/
theorem one_miss_gate_stays_open (t : Th) (hm : t.main = false) (hc : t.ctx = false) (i : Int) (hi : 0 < i) :
    (acquireWrite i t [⟨none, false⟩, ⟨none, true⟩]).map (·.2.log) =
      some (t.log ++ [.gilLock, .gilUnlock, .gilLock, .writeLock]) := by
  have : ¬ i ≤ 0 := by omega
  simp [acquireWrite, pollLoop, gilAcquire, gilDrop, holdsGil, openGate, gateAfter, hm, hc, this]

/-- A thread that already holds the GIL (main thread, or inside `hold_gil`) waits in one go,
whatever the interval; so does any thread when the interval is 0. -/
theorem acquire_blocking (t : Th) (i : Int) (a : Att) (rest : List Att) (h : i ≤ 0 ∨ holdsGil t = true) :
    acquireWrite i t (a :: rest) = acquireWrite 0 t [a] := by
  rcases h with h | h <;> simp [acquireWrite, h]

def gateBal (t : Th) : Int := (t.log.count .gateClose : Int) - t.log.count .gateOpen

/-- **The sliced wait**, on a thread that does not hold the GIL: on success it holds the GIL
(`need_release`) and took the write lock under the same GIL hold as its last check (the log
ends `gilLock, writeLock`, then at most the gate opening); on a refusal it holds the GIL no
more; either way the gate it closed is open again. -/
theorem pollLoop_spec : ∀ (atts : List Att) (missed : Nat) (closed : Bool) (t : Th) r t',
    t.main = false → t.ctx = false → pollLoop missed closed t atts = some (r, t') →
    gateBal t' = gateBal t - (if closed then 1 else 0) ∧
    (∀ nr, r = .ok nr → nr = true ∧ t'.ctx = true ∧
       ∃ pre, t'.log = pre ++ [.gilLock, .writeLock] ∨ t'.log = pre ++ [.gilLock, .writeLock, .gateOpen]) ∧
    (∀ e, r = .error e → t'.ctx = false)
  | [], _, _, _, _, _, _, _, h => by simp [pollLoop] at h
  | a :: rest, missed, closed, t, r, t', hm, hc, h => by
    have hg : gilAcquire t = (true, { t with ctx := true, log := t.log ++ [.gilLock] }) := by
      simp [gilAcquire, holdsGil, hm, hc]
    simp only [pollLoop, hg] at h
    cases hca : a.check with
    | some e =>
      simp only [hca] at h; simp at h; obtain ⟨rfl, rfl⟩ := h
      refine ⟨?_, by simp, ?_⟩
      · cases closed <;> simp [openGate, gilDrop, gateBal, List.count_append] <;> omega
      · intro e' _; cases closed <;> simp [openGate, gilDrop]
    | none =>
      simp only [hca] at h
      cases hgot : a.got with
      | true =>
        simp [hgot] at h; obtain ⟨rfl, rfl⟩ := h
        refine ⟨?_, ?_, by simp⟩
        · cases closed <;> simp [openGate, gateBal, List.count_append] <;> omega
        · intro nr hnr; simp at hnr; subst hnr
          refine ⟨rfl, by cases closed <;> simp [openGate], ?_⟩
          cases closed
          · exact ⟨t.log, Or.inl (by simp [openGate])⟩
          · exact ⟨t.log, Or.inr (by simp [openGate])⟩
      | false =>
        simp only [hgot] at h
        have hd : gilDrop true { t with ctx := true, log := t.log ++ [.gilLock] } =
            { t with ctx := false, log := t.log ++ [.gilLock, .gilUnlock] } := by simp [gilDrop]
        simp only [Bool.false_eq_true, ite_false, hd] at h
        split at h
        · have := pollLoop_spec rest (missed + 1) true { t with ctx := false, log := t.log ++ [.gilLock, .gilUnlock] ++ [.gateClose] } r t' hm rfl h
          refine ⟨?_, this.2⟩
          rw [this.1]; rename_i hcl
          have : closed = false := by simp at hcl; exact hcl.1
          subst this; simp [gateBal, List.count_append]; omega
        · have := pollLoop_spec rest (missed + 1) closed { t with ctx := false, log := t.log ++ [.gilLock, .gilUnlock] } r t' hm rfl h
          refine ⟨?_, this.2⟩
          rw [this.1]; simp [gateBal, List.count_append]

/-- `acquire_write` from a thread without the GIL, sliced: the gate is open on return. -/
theorem acquire_gate_open (t t' : Th) (i : Int) (hi : 0 < i) (atts : List Att) r
    (hm : t.main = false) (hc : t.ctx = false) (h : acquireWrite i t atts = some (r, t')) :
    gateBal t' = gateBal t := by
  have : ¬ i ≤ 0 := by omega
  simp [acquireWrite, holdsGil, hm, hc, this] at h
  simpa using (pollLoop_spec atts 0 false t r t' hm hc h).1

/-- A writer always holds the GIL (on a worker thread): `begin_writer`, and an escalation
that got the lock in its first slice. -/
theorem writer_holds_gil (t : Th) (hm : t.main = false) (hc : t.ctx = false) (i : Int) (rest : List Att) :
    (beginWriter t).th.ctx = true ∧
      (escalate (begin t) i (⟨none, true⟩ :: rest)).map (·.2.th.ctx) = some true := by
  by_cases hi : i ≤ 0 <;>
    simp [beginWriter, escalate, begin, dropMode, acquireWrite, pollLoop, gilAcquire, holdsGil, openGate, hm, hc, hi]

/-- Releasing a writer unlocks the write lock before the GIL. -/
theorem release_order (t : Th) (hm : t.main = false) (hc : t.ctx = false) :
    (release (beginWriter t)).th.log = t.log ++ [.gilLock, .writeLock, .writeUnlock, .gilUnlock] := by
  simp [release, beginWriter, dropMode, gilAcquire, holdsGil, gilDrop, hm, hc]

/-- A second escalation after a successful one is free and re-checks nothing. -/
theorem upgrade_twice (s s' : S) (i i' : Int) (ws ws' : List World) (f : Facts)
    (h : upgrade s i ws f = some (none, s')) : upgrade s' i' ws' f = some (none, s') := by
  have he : ∃ b, escalate s i (ws.map (World.att f)) = some (.ok b, s') := by
    unfold upgrade at h; split at h
    · simp at h
    · rename_i b _ hs; simp at h; subst h; exact ⟨b, hs⟩
    · simp at h
  obtain ⟨b, he⟩ := he
  obtain ⟨nr, hm⟩ := escalate_mode s s' i _ b he
  simp [upgrade, escalate_of_writer s' nr _ _ hm]

/-- A deleted/replaced graph aborts before any role check. -/
theorem upgrade_unregistered (t : Th) (i : Int) (c : Option (Bool × Bool)) (g : Bool) (ws : List World) (f : Facts) :
    (upgrade (begin t) i (⟨false, c, g⟩ :: ws) f).map (·.1) = some (some .graphUnregistered) := by
  by_cases hi : i ≤ 0
  · by_cases hh : holdsGil t = true <;>
      simp_all [upgrade, escalate, begin, dropMode, acquireWrite, World.att, authorize, holdsGil,
        gilAcquire]
  · by_cases hh : holdsGil t = true <;>
      simp_all [upgrade, escalate, begin, dropMode, acquireWrite, pollLoop, World.att, authorize, holdsGil,
        gilAcquire, ite_eq_right hi]

/-- **A refused upgrade re-checks** (#3816): the session is a reader again, so calling
`upgrade_to_write` again runs every check again and reports the same refusal while it
stands — it no longer reports success (cf. `Pre3816.pre3816_upgrade_twice_skips_checks`). -/
theorem upgrade_refused_rechecks (s s' : S) (i : Int) (c : Option (Bool × Bool)) (g : Bool)
    (ws : List World) (f : Facts) (h : upgrade s i (⟨false, c, g⟩ :: ws) f = some (some .graphUnregistered, s')) :
    (upgrade s' i (⟨false, c, g⟩ :: ws) f).map (·.1) = some (some .graphUnregistered) := by
  have he : escalate s i ((⟨false, c, g⟩ :: ws).map (World.att f)) = some (.error .graphUnregistered, s') := by
    unfold upgrade at h; split at h
    · simp at h
    · simp at h
    · rename_i e _ hs; simp at h; obtain ⟨rfl, rfl⟩ := h; exact hs
  have hr := (escalate_refused_reader s s' i _ _ he).1
  by_cases hi : i ≤ 0
  · by_cases hh : holdsGil (dropMode s'.mode s'.th) = true <;>
      simp_all [upgrade, escalate, dropMode, acquireWrite, World.att, authorize, holdsGil,
        gilAcquire]
  · by_cases hh : holdsGil (dropMode s'.mode s'.th) = true <;>
      simp_all [upgrade, escalate, dropMode, acquireWrite, pollLoop, World.att, authorize, holdsGil,
        gilAcquire, ite_eq_right hi]

/-! ## Readers and the writer gate (#3816, `read_past_gate` `:524`) -/

/-- `may_wait_at_gate` (`:548`): a pool worker that does not hold the GIL. -/
def mayWaitAtGate (pool : Bool) (t : Th) : Bool := pool && !holdsGil t

/-- `read_past_gate`: take the read lock; if the gate is closed (the list: what each try
sees) and this thread may wait, let the lock go and wait at the gate holding nothing. `none`:
still waiting. -/
def readPastGate (mayWait : Bool) : List Bool → Th → Option Th
  | [], _ => none
  | closed :: rest, t =>
    let t1 := { t with log := t.log ++ [.readLock] }
    if !closed || !mayWait then some t1
    else readPastGate mayWait rest { t1 with log := t1.log ++ [.readUnlock, .gateWait] }

/-- `begin_with` (`:185`): `begin` once the reader is past the gate. -/
def beginWith (mayWait : Bool) (gate : List Bool) (t : Th) : Option S :=
  (readPastGate mayWait gate t).map fun t => ⟨some .reader, t⟩

/-- An open gate, or a thread that may not wait, costs nothing: `begin`. -/
theorem beginWith_open (m : Bool) (rest : List Bool) (t : Th) : beginWith m (false :: rest) t = some (begin t) := by
  simp [beginWith, readPastGate, begin]
theorem beginWith_nowait (g : Bool) (rest : List Bool) (t : Th) : beginWith false (g :: rest) t = some (begin t) := by
  simp [beginWith, readPastGate, begin]

/-- A reader that waits at the gate does not hold the GIL, which the gate's writer needs
for every slice. -/
theorem gate_waiter_holds_nothing (pool : Bool) (t : Th) (h : mayWaitAtGate pool t = true) :
    t.main = false ∧ t.ctx = false := by
  simp [mayWaitAtGate, holdsGil] at h; exact ⟨h.2.1, h.2.2⟩

/-- Past the gate, the reader holds the read lock (the last event), and the wait changed
neither its GIL state nor its thread. -/
theorem readPastGate_ends_reading : ∀ (m : Bool) (g : List Bool) (t t' : Th),
    readPastGate m g t = some t' → t'.log.getLast? = some .readLock ∧ t'.ctx = t.ctx ∧ t'.main = t.main
  | _, [], _, _, h => by simp [readPastGate] at h
  | m, c :: rest, t, t', h => by
    simp only [readPastGate] at h
    split at h
    · simp at h; subst h; simp
    · have := readPastGate_ends_reading m rest _ t' h; simpa using this

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

/-! ## Historical: before #3816 (dc05fee89)

`escalate` took the GIL and then the write lock in one go and recorded writer mode
*before* `upgrade_to_write` ran the registry/role checks, so a refused session stayed a
writer and a second `upgrade_to_write` took the already-a-writer early return. -/
namespace Pre3816

def escalate (s : S) : Bool × S :=
  match s.mode with
  | some (.writer _) => (false, s)
  | m =>
    let t := dropMode m s.th
    let (nr, t) := gilAcquire t
    (true, ⟨some (.writer nr), { t with log := t.log ++ [.writeLock] }⟩)

def upgrade (s : S) (registered : Bool) (ctx : Option (Bool × Bool)) (f : Facts) :
    Option WriteAbort × S :=
  let (esc, s) := escalate s
  if esc then
    if !registered then (some .graphUnregistered, s) else (reauthorize f ctx, s)
  else (none, s)

theorem escalate_of_writer (s : S) (nr : Bool) (h : s.mode = some (.writer nr)) : escalate s = (false, s) := by
  simp [escalate, h]

theorem escalate_mode (s : S) : ∃ nr, (escalate s).2.mode = some (.writer nr) := by
  unfold escalate
  split
  · rename_i nr h; exact ⟨nr, h⟩
  · exact ⟨_, rfl⟩

/-- Pre-#3816 `escalate_writer`: every escalation, refused or not, left a writer. -/
theorem escalate_writer (s : S) : withGraphMut (escalate s).2 = true := by
  obtain ⟨nr, h⟩ := escalate_mode s
  simp [withGraphMut, h]

theorem upgrade_state (s : S) (r : Bool) (c : Option (Bool × Bool)) (f : Facts) :
    (upgrade s r c f).2 = (escalate s).2 := by
  unfold upgrade
  obtain ⟨e, s'⟩ := escalate s
  simp only
  split <;> (try split) <;> rfl

/-- Pre-#3816 `upgrade_twice`: a second upgrade reported success unconditionally. -/
theorem upgrade_twice (s : S) (r : Bool) (c : Option (Bool × Bool)) (f : Facts) :
    (upgrade (upgrade s r c f).2 r c f).1 = none := by
  rw [upgrade_state]
  obtain ⟨nr, h⟩ := escalate_mode s
  simp [upgrade, escalate_of_writer _ nr h]

/-- **The hazard #3816 removes**: refused (graph gone), then asked again, the old code
reported success without checking. Fixed: `upgrade_refused_rechecks`. -/
theorem pre3816_upgrade_twice_skips_checks (t : Th) (c : Option (Bool × Bool)) (f : Facts) :
    (upgrade (begin t) false c f).1 = some .graphUnregistered ∧
      (upgrade (upgrade (begin t) false c f).2 false c f).1 = none :=
  ⟨by simp [upgrade, escalate, begin], upgrade_twice _ _ _ _⟩

end Pre3816

end RedisLayer.QuerySession
