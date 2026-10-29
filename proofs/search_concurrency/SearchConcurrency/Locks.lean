/-
# Lock order, the write-loop hand-off, and the bounded-queue deadlock

| here | there |
| --- | --- |
| `Lock.gil < .graph < .indexer` | `graph/src/locks.rs:9` ordering rule; `query_session.rs:30` `Mode` |
| `escalateProg`                 | `QuerySession::escalate` (`query_session.rs:379-394`): release R, GIL, W |
| `inlineWriterProg`             | `QuerySession::begin_writer` (`:182`): GIL, W |
| `populateProg`                 | `populate_index_batch` (`graph.rs:562`): indexer lock only |
| `commitIndexProg`              | `Graph::commit_index` under writer mode (`graph.rs:3799`): GIL, W, indexer |
| `forkProg`                     | `pre_fork_prepare` (`redis_type.rs:350`): GIL (main), registry mutex |
| `rsGlobal`, `rsSpec`, `writerOpenScanProg` | gap closed 2026-10-20 (#3226): RediSearch's global `RWLOCK` and per-spec `rwlock`, held by an index scan from `handleIterCommon` (`deps/RediSearch src/redisearch_api.c:904-905`) to `ResultsIteratorFree` (:1064); `CommitOp::next` (`runtime/ops/commit.rs:93`) commits with the scan open |
| `Drain`                        | `process_write_queued_query` (`src/graph_core.rs:1615`) `write_loop` flag protocol |
| `P3014`, `succ3014`            | since #3014 (b4634fe4b): `admit` (`graph_core.rs:1007`, `QueueSlot::try_acquire`), unbounded pool queue (`threadpool.rs:67`) and write channel (`graph_core.rs:557`), slot released at execution / dequeue |
| `Pool` (historical)            | before #3014: `bounded_blocking(1024)` pool queue + blocking `spawn` from `query_mut` on the Redis main thread |
-/

namespace SC.Locks

/-! ## Deadlock freedom of the lock order -/

/-- Locks, ranked. Every thread acquires in strictly increasing rank. -/
def rank : Nat → Nat := id

/-- A wait-for edge `t → u`: `t` waits for a lock `u` holds. -/
structure WaitState where
  holds : Nat → List Nat         -- thread → locks held (ranks)
  wants : Nat → Option Nat       -- thread → lock requested

/-- Ordered: whatever a thread waits for ranks above everything it holds. -/
def Ordered (s : WaitState) : Prop :=
  ∀ t l, s.wants t = some l → ∀ h ∈ s.holds t, h < l

def waitsOn (s : WaitState) (t u : Nat) : Prop :=
  ∃ l, s.wants t = some l ∧ l ∈ s.holds u

/-- `Chain R a [b, c, …]` : `R a b ∧ R b c ∧ …`. -/
inductive Chain (R : Nat → Nat → Prop) : Nat → List Nat → Prop where
  | nil (a : Nat) : Chain R a []
  | cons {a b : Nat} {l : List Nat} : R a b → Chain R b l → Chain R a (b :: l)

/-- Along a wait-for path the *wanted* lock rank strictly increases. -/
theorem path_increasing (s : WaitState) (ho : Ordered s) :
    ∀ (p : List Nat) (t : Nat) (lt : Nat), s.wants t = some lt →
      Chain (waitsOn s) t p → ∀ u ∈ p.getLast?, ∀ lu, s.wants u = some lu → p ≠ [] → lt < lu := by
  intro p
  induction p with
  | nil => intro _ _ _ _ _ _ _ _ h; exact absurd rfl h
  | cons x xs ih =>
    intro t lt ht hc u hu lu hlu _
    cases hc with
    | cons hw hc' =>
      obtain ⟨l, hl, hin⟩ := hw
      rw [ht] at hl; cases hl
      cases xs with
      | nil =>
        simp at hu; subst hu
        exact ho _ _ hlu _ hin
      | cons y ys =>
        -- x waits on y too
        have hc2 := hc'
        cases hc2 with
        | cons hxy _ =>
          obtain ⟨lx, hlx, _⟩ := hxy
          have h1 : lt < lx := ho _ _ hlx _ hin
          have h2 := ih x lx hlx hc' u (by simpa using hu) lu hlu (by simp)
          omega

/-- **No wait-for cycle** among threads that all acquire in lock order. -/
theorem no_cycle (s : WaitState) (ho : Ordered s) (t : Nat) (p : List Nat) (lt : Nat)
    (ht : s.wants t = some lt) (hc : Chain (waitsOn s) t (p ++ [t])) : False := by
  have := path_increasing s ho (p ++ [t]) t lt ht hc t (by simp) lt ht (by simp)
  omega

/-! The programs, as lock sequences with releases. A program is *ordered* if at every
acquire all currently held locks rank lower. -/

inductive Op where
  | acq (l : Nat)
  | rel (l : Nat)
deriving DecidableEq, Repr

def gil : Nat := 0
def graphR : Nat := 1   -- the per-graph RwLock (read or write mode, same rank)
def indexer : Nat := 2
def registry : Nat := 3

def okProg : List Nat → List Op → Bool
  | _, [] => true
  | held, .acq l :: ops => held.all (· < l) && okProg (l :: held) ops
  | held, .rel l :: ops => okProg (held.erase l) ops

/-- `QuerySession::begin` then `escalate` then commit (with index documents). -/
def writerProg : List Op :=
  [.acq graphR, .rel graphR, .acq gil, .acq graphR, .acq indexer, .rel indexer,
   .acq registry, .rel registry, .rel graphR, .rel gil]
def inlineWriterProg : List Op := [.acq gil, .acq graphR, .acq indexer, .rel indexer, .rel graphR]
def populateProg : List Op := [.acq indexer, .rel indexer]
def dropIndexBgProg : List Op := [.acq indexer, .rel indexer]
def forkProg : List Op := [.acq gil, .acq registry, .rel registry]
def telemetryProg : List Op := [.acq gil, .acq registry, .rel registry, .rel gil]
/-- The #726 inversion that `escalate` exists to avoid: GIL while still reading. -/
def badEscalateProg : List Op := [.acq graphR, .acq gil]

theorem all_programs_ordered :
    okProg [] writerProg ∧ okProg [] inlineWriterProg ∧ okProg [] populateProg ∧
    okProg [] dropIndexBgProg ∧ okProg [] forkProg ∧ okProg [] telemetryProg := by decide

theorem bad_escalate_rejected : okProg [] badEscalateProg = false := by decide

/-! ## Gap closed 2026-10-20 (#3226): the RediSearch locks an open index scan holds

The ranks above stop at the registry; RediSearch's own locks were abstracted away, so
`all_programs_ordered` could not see that an index scan keeps two of them **from scan
open to scan drop**, not only for the duration of a call:

* `rsGlobal`: RediSearch's global `RWLOCK` (`deps/RediSearch@ea1a6f4 src/rwlock.c:54-83`,
  thread-reentrant; a read→write "upgrade" only trips an `RS_ASSERT` compiled out of
  release builds, so it silently proceeds);
* `rsSpec`: the per-spec `sp->rwlock` (`src/spec.c:2263-2310`). Re-entering it as a
  writer while this thread holds it as a reader hits
  `RS_LOG_ASSERT_ALWAYS(e->write, "attempted to upgrade a spec read lock to a write lock")`
  (`spec.c:2304`) and aborts the server.

Both are taken by `handleIterCommon` (`src/redisearch_api.c:904-905`, every
`RediSearch_GetResultsIterator`, i.e. `Index::query` `mod.rs:1691` and the vector and
full-text queries) and released only by `RediSearch_ResultsIteratorFree`
(`redisearch_api.c:1064`), i.e. when the Rust `IndexResultsIter` drops with the scan op.
`RediSearch_IndexAddDocument` (`redisearch_api.c:643-645`, reached from
`Indexer::commit` `indexer.rs:730` → `Index::add_document`) takes both as a **writer**,
and so do `RediSearch_CreateField` / `CreateIndex` (`redisearch_api.c:164`, …) on the
RDB-load (`RESTORE`) path, which runs on the main thread under the GIL.

They rank above the indexer lock: populate and commit take the indexer lock first and
then call into RediSearch (`graph.rs:3828` `commit_index_locked`). -/

def rsGlobal : Nat := 4
def rsSpec : Nat := 5

/-- **Property (#3226).** A thread never re-acquires a lock it already holds. For the
RediSearch spec lock this is exactly the `spec.c:2304` assertion (read held, write
wanted); with the rank order it is what `okProg` already demands, stated on its own so
a fix that keeps the scan open but drops it before commit can prove it separately. -/
def noReacquire : List Nat → List Op → Bool
  | _, [] => true
  | held, .acq l :: ops => !held.contains l && noReacquire (l :: held) ops
  | held, .rel l :: ops => noReacquire (held.erase l) ops

/-- **Property (#3226).** Every locking path is ordered (`okProg`, the premise of
`no_cycle`) once the RediSearch locks are in the model. -/
abbrev LockOrdered (p : List Op) : Prop := okProg [] p = true

/-- A read query's index scan, opened and drained under the per-graph read lock. -/
def readerScanProg : List Op :=
  [.acq graphR, .acq rsGlobal, .acq rsSpec, .rel rsSpec, .rel rsGlobal, .rel graphR]
/-- `populate_index_batch` (`graph.rs:562`): indexer lock, then `add_document`. -/
def populateSpecProg : List Op :=
  [.acq indexer, .acq rsGlobal, .acq rsSpec, .rel rsSpec, .rel rsGlobal, .rel indexer]
/-- A writer whose index scan has been **dropped** before its commit publishes
documents (e.g. `MATCH (n:W) WHERE n.v > 0 SET …` with no `LIMIT`: the scan is
exhausted and freed before `CommitOp` escalates). -/
def writerClosedScanProg : List Op :=
  [.acq graphR, .acq rsGlobal, .acq rsSpec, .rel rsSpec, .rel rsGlobal,
   .rel graphR, .acq gil, .acq graphR, .acq indexer, .acq rsGlobal, .acq rsSpec,
   .rel rsSpec, .rel rsGlobal, .rel indexer, .rel graphR, .rel gil]
/-- `RESTORE` / RDB load on the main thread: GIL, then RediSearch as a writer. -/
def restoreProg : List Op := [.acq gil, .acq rsGlobal, .acq rsSpec, .rel rsSpec, .rel rsGlobal, .rel gil]

/-- With the RediSearch locks added, every path whose scan is closed before its
commit is still ordered and never re-acquires. -/
theorem spec_programs_ordered :
    LockOrdered readerScanProg ∧ LockOrdered populateSpecProg ∧
    LockOrdered writerClosedScanProg ∧ LockOrdered restoreProg ∧
    noReacquire [] readerScanProg ∧ noReacquire [] populateSpecProg ∧
    noReacquire [] writerClosedScanProg ∧ noReacquire [] restoreProg := by decide

/-- Main at fb3d516d0, live repro of #3226:
`MATCH (n:W) WHERE n.v > 0 WITH n LIMIT 10 SET n.v = n.v + 100000 RETURN count(n)`.
The index scan is opened in reader mode (`graphR`, then `handleIterCommon` takes
`rsGlobal` and `rsSpec` as a reader). `LIMIT` stops pulling but the scan op, and so its
results iterator, lives until the plan drops. `CommitOp::next` (`commit.rs:93`) then
escalates (`query_session.rs:386-391`: drop R, take the GIL, take W),
`commit_deferred_indexes` takes the indexer lock (`pending.rs:1690` → `graph.rs:3815`),
and `Indexer::commit` → `RediSearch_IndexAddDocument` takes `rsGlobal` and `rsSpec` as
a writer while the same thread still holds them as a reader. -/
def writerOpenScanProg : List Op :=
  [.acq graphR, .acq rsGlobal, .acq rsSpec,            -- index scan opened, kept open
   .rel graphR, .acq gil, .acq graphR,                  -- escalate (commit.rs:93)
   .acq indexer, .acq rsGlobal, .acq rsSpec,            -- Indexer::commit → IndexAddDocument
   .rel rsSpec, .rel rsGlobal, .rel indexer, .rel graphR, .rel gil,
   .rel rsSpec, .rel rsGlobal]                          -- scan dropped with the plan

/-- **BUG #3226 (abort).** Main re-acquires the RediSearch spec lock it holds as a
reader: the `spec.c:2304` assertion. Live (port 6640, `main.dylib` at 9d3869aa8): the
server aborts with `attempted to upgrade a spec read lock to a write lock`, stack
`IndexSpec_LockWrite ← RediSearch_IndexAddDocument ← Indexer::commit ←
Graph::commit_index ← Pending::commit_deferred_indexes ← CommitOp::next`; C returns 10. -/
theorem bug3226_open_scan_upgrade : noReacquire [] writerOpenScanProg = false := by decide

/-- **BUG #3226 (order).** The same path also takes the GIL while holding the RediSearch
locks, the inversion behind the two-client stall. -/
theorem bug3226_open_scan_unordered : ¬ LockOrdered writerOpenScanProg := by decide

/-- **BUG #3226 (stall), as a wait-for cycle.** Thread 1 is the writer of
`writerOpenScanProg` stopped at its escalation: it holds `rsGlobal` (read, from the
open scan) and wants the GIL. Thread 0 is the main thread running `RESTORE`
(`restoreProg`): it holds the GIL and wants `rsGlobal` as a writer. The issue's
two-client run: PING gets no reply. `no_cycle` does not apply because the state is not
`Ordered`. -/
def stallState : WaitState :=
  { holds := fun t => if t = 0 then [gil] else if t = 1 then [rsGlobal] else []
    wants := fun t => if t = 0 then some rsGlobal else if t = 1 then some gil else none }

theorem bug3226_restore_stall_cycle :
    Chain (waitsOn stallState) 0 [1, 0] ∧ ¬ Ordered stallState := by
  refine ⟨.cons ⟨rsGlobal, rfl, by decide⟩ (.cons ⟨gil, rfl, by decide⟩ (.nil 0)), ?_⟩
  intro h
  have := h 1 gil rfl rsGlobal (by decide)
  exact absurd this (by decide)

/-! ## Write-loop hand-off (`write_loop` flag), exhaustively checked

Two producers (each: `send`, then `CAS(false→true)`, drain if won) and the drain
loop of `process_write_queued_query`:
`try_recv` → on empty: `store(false)`; `is_empty`? return : `CAS` ? continue : return.
Property: when every thread has returned, the queue is empty (no message is left
behind with nobody draining — the lost-wakeup bug the re-check exists for). -/

inductive PC where
  | send | cas | loop | empty | stored | casAgain | done
deriving DecidableEq, Repr

structure D where
  q : Nat
  flag : Bool
  pcs : List PC
deriving DecidableEq, Repr

def stepT (s : D) (i : Nat) : Option D :=
  match s.pcs[i]? with
  | some .send => some { s with q := s.q + 1, pcs := s.pcs.set i .cas }
  | some .cas => if s.flag then some { s with pcs := s.pcs.set i .done }
                 else some { s with flag := true, pcs := s.pcs.set i .loop }
  | some .loop => if s.q > 0 then some { s with q := s.q - 1 }       -- try_recv Ok; run it
                  else some { s with pcs := s.pcs.set i .empty }       -- try_recv Err
  | some .empty => some { s with flag := false, pcs := s.pcs.set i .stored }  -- store(false)
  | some .stored => if s.q = 0 then some { s with pcs := s.pcs.set i .done }
                    else some { s with pcs := s.pcs.set i .casAgain }
  | some .casAgain => if s.flag then some { s with pcs := s.pcs.set i .done }
                      else some { s with flag := true, pcs := s.pcs.set i .loop }
  | _ => none

def succs (s : D) : List D := (List.range s.pcs.length).filterMap (stepT s)

/-- Bounded exhaustive exploration: every state reachable in `n` steps. -/
def explore : Nat → List D → List D
  | 0, fr => fr
  | n + 1, fr => fr ++ explore n (fr.flatMap succs).eraseDups

def terminal (s : D) : Bool := s.pcs.all (· == .done)

def drainInit (k : Nat) : D := { q := 0, flag := false, pcs := List.replicate k .send }

/-- With 2 producers (every interleaving is at most 20 steps long; `explore_saturated`), and every
terminal state has an empty queue. -/
theorem drain_no_lost_message :
    (explore 20 [drainInit 2]).all (fun s => !terminal s || s.q == 0) = true := by decide +kernel

/-- 20 steps is enough: nothing new is reachable beyond depth 20, so the exploration is exhaustive. -/
theorem explore_saturated :
    ((explore 20 [drainInit 2]).flatMap succs).all (fun s => (explore 20 [drainInit 2]).contains s) = true := by
  decide +kernel

/-- The re-check is load-bearing: drop it (return right after `store(false)`) and
a message is stranded. -/
def stepBad (s : D) (i : Nat) : Option D :=
  match s.pcs[i]? with
  | some .stored => some { s with pcs := s.pcs.set i .done }
  | _ => stepT s i

def exploreBad : Nat → List D → List D
  | 0, fr => fr
  | n + 1, fr => fr ++ exploreBad n (fr.flatMap fun s => (List.range s.pcs.length).filterMap (stepBad s)).eraseDups

theorem recheck_needed :
    (exploreBad 20 [drainInit 2]).any (fun s => terminal s && s.q != 0) = true := by decide +kernel

/-! ## Historical: the bounded-queue deadlock (W3-conc-1, CONFIRMED live; fixed by #3014 b4634fe4b)

Resources that are not locks also block: the pool's job queue (capacity `cap`,
`threadpool.rs:59`) and a graph's write channel (capacity 1024, `graph_core.rs:548`).
The Redis main thread holds the GIL for every command callback and `spawn`s into
the pool with a **blocking** send (`threadpool.rs:96`). A pool worker that has
started a write escalates and waits for the GIL. So:

  main (holds GIL) ⟶ waits for a pool-queue slot
  pool-queue slots ⟶ freed only when a worker finishes a job
  every worker ⟶ waits for the GIL (escalating writer), or on a full write channel
                  whose only consumer is one of those GIL-waiting workers

`Pool` below is that system with `w` workers and queue capacity `cap`. -/

inductive Job where
  | write   -- will need the GIL
deriving DecidableEq, Repr

structure Pool where
  queue : Nat          -- jobs queued
  running : Nat        -- workers executing a job (all of them will need the GIL)
  mainBlocked : Bool   -- main thread stuck in spawn, holding the GIL
  pendingClients : Nat -- commands not yet dispatched
deriving DecidableEq, Repr

variable (w cap : Nat)

/-- Events: main dispatches a command (GIL held); a worker picks a job; a running
worker finishes — which needs the GIL, free only when main is not in a callback. -/
def poolSucc (s : Pool) : List Pool :=
  (if s.pendingClients > 0 ∧ ¬ s.mainBlocked then
    if s.queue < cap then [{ s with queue := s.queue + 1, pendingClients := s.pendingClients - 1 }]
    else [{ s with mainBlocked := true }] else []) ++
  (if s.queue > 0 ∧ s.running < w then
    [{ s with queue := s.queue - 1, running := s.running + 1,
              mainBlocked := false }]          -- a freed slot unblocks main
   else []) ++
  (if s.running > 0 ∧ ¬ s.mainBlocked then [{ s with running := s.running - 1 }] else [])

def stuck (s : Pool) : Bool := (poolSucc w cap s).isEmpty && (s.pendingClients > 0 || s.queue > 0 || s.running > 0)

/-- The adversarial schedule: the clients arrive faster than workers can take the
GIL. With 1 worker and a 1-slot queue, 3 simultaneous writes reach a state with no
enabled event: main holds the GIL waiting for queue space, the worker waits for the
GIL. (Live: 11 workers, cap 1024, 1400 writes on one graph or 3000 over 16 graphs.) -/
def deadState : Pool := { queue := 1, running := 1, mainBlocked := true, pendingClients := 1 }

/-- The fix proposed before #3014: `spawn` from the main thread must not block (reject
with "Max pending queries exceeded", or an unbounded queue as C's thpool has). #3014
does both; its actual protocol is `succ3014` below. -/
def poolSuccFixed (s : Pool) : List Pool :=
  (if s.pendingClients > 0 then
    if s.queue < cap then [{ s with queue := s.queue + 1, pendingClients := s.pendingClients - 1 }]
    else [{ s with pendingClients := s.pendingClients - 1 }]   -- rejected with an error
   else []) ++
  (if s.queue > 0 ∧ s.running < w then [{ s with queue := s.queue - 1, running := s.running + 1 }] else []) ++
  (if s.running > 0 then [{ s with running := s.running - 1 }] else [])

theorem fixed_never_stuck (s : Pool) (hw : 0 < w) :
    (s.pendingClients > 0 ∨ s.queue > 0 ∨ s.running > 0) → poolSuccFixed w cap s ≠ [] := by
  intro h
  unfold poolSuccFixed
  rcases h with h | h | h
  · simp [h]; split <;> simp
  · by_cases hr : s.running > 0
    · simp [hr]
    · have : s.running < w := by omega
      simp [h, this]
  · simp [h]

/-! ## The #3014 protocol: admission-bounded, nothing blocks the main thread

`admit` (`graph_core.rs:1007`) takes a `QueueSlot` on the main thread or replies "Max
pending queries exceeded" — it never waits; `spawn` sends into an unbounded queue. A read
gives its slot back when it starts executing (`admission.take()`), a write carries it into
the graph's (unbounded) write channel and gives it back when the drainer dequeues it. -/

structure P3014 where
  pending : Nat    -- commands not yet dispatched
  queue : Nat      -- pool jobs waiting (each holds a slot)
  running : Nat    -- workers executing, no slot held
  matching : Nat   -- workers matching a write, slot held
  writeQ : Nat     -- writes in write channels (each holds a slot)
  count : Nat      -- `QUEUED_QUERIES`
  deriving DecidableEq, Repr

def succ3014 (max : Nat) (s : P3014) : List P3014 :=
  (if s.pending > 0 then
    (if s.count < max then [{ s with pending := s.pending - 1, queue := s.queue + 1, count := s.count + 1 }]
     else [{ s with pending := s.pending - 1 }])                 -- rejected, never waits
   else []) ++
  (if s.queue > 0 ∧ s.running + s.matching < w then
    [{ s with queue := s.queue - 1, running := s.running + 1, count := s.count - 1 },   -- a read starts
     { s with queue := s.queue - 1, matching := s.matching + 1 }]                      -- a write starts matching
   else []) ++
  (if s.matching > 0 then [{ s with matching := s.matching - 1, running := s.running + 1, writeQ := s.writeQ + 1 }]
   else []) ++                                                   -- unbounded send: never waits
  (if s.writeQ > 0 then [{ s with writeQ := s.writeQ - 1, count := s.count - 1 }] else []) ++  -- dequeued
  (if s.running > 0 then [{ s with running := s.running - 1 }] else [])   -- GIL is never held by a blocked main

def Inv3014 (max : Nat) (s : P3014) : Prop :=
  s.count = s.queue + s.matching + s.writeQ ∧ s.count ≤ max

/-- **The count is exactly the queued queries and never exceeds `MAX_QUEUED_QUERIES`**,
in every reachable state. -/
theorem inv3014_step (max : Nat) (s t : P3014) (h : Inv3014 max s) (ht : t ∈ succ3014 w max s) :
    Inv3014 max t := by
  obtain ⟨h1, h2⟩ := h
  by_cases c1 : s.pending > 0 <;> by_cases c2 : s.count < max <;>
    by_cases c3 : (s.queue > 0 ∧ s.running + s.matching < w) <;> by_cases c4 : s.matching > 0 <;>
    by_cases c5 : s.writeQ > 0 <;> by_cases c6 : s.running > 0 <;>
    simp only [succ3014, c1, c2, c3, c4, c5, c6, if_true, if_false, List.mem_append, List.mem_cons,
      List.mem_nil_iff, or_false, false_or, and_self, ite_true, ite_false, true_and] at ht <;>
    (repeat' (rcases ht with ht | ht)) <;>
    (subst_vars; constructor <;> simp <;> omega)

/-- **No state with work left is stuck** (W3-conc-1 is gone): whatever is pending, queued,
matching or running, some step is enabled — the main thread can always dispatch or
reject, and every worker can make progress. -/
theorem never_stuck_3014 (max : Nat) (s : P3014) (hw : 0 < w)
    (h : s.pending > 0 ∨ s.queue > 0 ∨ s.running > 0 ∨ s.matching > 0 ∨ s.writeQ > 0) :
    succ3014 w max s ≠ [] := by
  unfold succ3014
  intro hn
  simp only [List.append_eq_nil_iff] at hn
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := hn
  rcases h with h | h | h | h | h
  · simp [h] at h1; split at h1 <;> simp at h1
  · by_cases hr : s.running + s.matching < w
    · simp [h, hr] at h2
    · rcases Nat.eq_zero_or_pos s.running with h0 | h0
      · have : s.matching > 0 := by omega
        simp [this] at h3
      · simp [h0] at h5
  · simp [h] at h5
  · simp [h] at h3
  · simp [h] at h4

/-- **The main thread never blocks**: with a command pending, dispatching it (admitted or
rejected) is always enabled (the pre-#3014 model blocked on a full queue). -/
theorem main_never_blocks (max : Nat) (s : P3014) (h : s.pending > 0) :
    ∃ t ∈ succ3014 w max s, t.pending = s.pending - 1 := by
  by_cases hc : s.count < max
  · exact ⟨{ s with pending := s.pending - 1, queue := s.queue + 1, count := s.count + 1 },
      by simp [succ3014, h, hc], rfl⟩
  · exact ⟨{ s with pending := s.pending - 1 }, by simp [succ3014, h, hc], rfl⟩

/-- The pre-#3014 deadlock trace, replayed under #3014 (1 worker, `MAX_QUEUED_QUERIES = 1`,
3 writes): the third command is rejected instead of blocking, and everything drains. -/
theorem flood_drains :
    let s0 : P3014 := ⟨3, 0, 0, 0, 0, 0⟩
    let s1 : P3014 := ⟨2, 1, 0, 0, 0, 1⟩      -- admitted
    let s2 : P3014 := ⟨2, 0, 0, 1, 0, 1⟩      -- a write starts matching
    let s3 : P3014 := ⟨1, 0, 0, 1, 0, 1⟩      -- next command rejected ("Max pending queries exceeded")
    let s4 : P3014 := ⟨1, 0, 1, 0, 1, 1⟩      -- queued to the write channel
    let s5 : P3014 := ⟨1, 0, 1, 0, 0, 0⟩      -- dequeued
    s1 ∈ succ3014 1 1 s0 ∧ s2 ∈ succ3014 1 1 s1 ∧ s3 ∈ succ3014 1 1 s2 ∧ s4 ∈ succ3014 1 1 s3 ∧
    s5 ∈ succ3014 1 1 s4 := by decide

end SC.Locks
