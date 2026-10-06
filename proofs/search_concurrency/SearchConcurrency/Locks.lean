/-
# Lock order, the write-loop hand-off, and the bounded-queue deadlock

| here | there |
| --- | --- |
| `Lock.gil < .graph < .indexer` | `graph/src/locks.rs:9` ordering rule; `query_session.rs:30` `Mode` |
| `escalateProg`                 | `QuerySession::escalate` (`query_session.rs:242-257`): release R, GIL, W |
| `inlineWriterProg`             | `QuerySession::begin_writer` (`:144`): GIL, W |
| `populateProg`                 | `populate_index_batch` (`graph.rs:532`): indexer lock only |
| `commitIndexProg`              | `Graph::commit_index` under writer mode (`graph.rs:3532`): GIL, W, indexer |
| `forkProg`                     | `pre_fork_prepare` (`redis_type.rs:350`): GIL (main), registry mutex |
| `Drain`                        | `process_write_queued_query` (`src/graph_core.rs:1525-1554`) `write_loop` flag protocol |
| `Pool`                         | `threadpool.rs:58` `bounded_blocking(1024)` MPMC + `spawn` (`:89`) called from `query_mut` on the Redis main thread (`graph_core.rs:1052`) |
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

/-! ## The bounded-queue deadlock (CONFIRMED on a live server)

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

theorem pool_deadlock_reachable :
    let s0 : Pool := { queue := 0, running := 0, mainBlocked := false, pendingClients := 3 }
    let s1 := { s0 with queue := 1, pendingClients := 2 }
    let s2 := { s1 with queue := 0, running := 1 }
    let s3 := { s2 with queue := 1, pendingClients := 1 }
    s1 ∈ poolSucc 1 1 s0 ∧ s2 ∈ poolSucc 1 1 s1 ∧ s3 ∈ poolSucc 1 1 s2 ∧
    deadState ∈ poolSucc 1 1 s3 ∧ stuck 1 1 deadState = true := by decide

/-- Fix: `spawn` from the main thread must not block (try-send and reply
"Max pending queries exceeded", or an unbounded queue as C's thpool has). Then
main never holds the GIL while waiting, and no state is stuck. -/
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

end SC.Locks
