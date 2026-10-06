/-
# `graph/src/threadpool.rs` (origin/main 3fec7d7c9): pool lifecycle

| Lean | Rust |
| --- | --- |
| `PoolS`, `PoolS.new` | `ThreadPool::new` `threadpool.rs:56` (MPMC queue, `size` workers, each holding a receiver clone) |
| `PoolS.step` | a worker's `while let Ok(job) = rx.recv()` loop body (`:63-78`): `Some(j)` runs (panics contained), `None` exits |
| `PoolS.spawn` | `ThreadPool::spawn` `:89` (send fails only once every receiver is gone, i.e. no worker is alive) |
| `PoolS.pendingCount` | `ThreadPool::pending_count` `:105` |
| `PoolS.shutdown`, `run` | `ThreadPool::shutdown` `:115` (one `None` per worker, then `join`) |
| `Global` | `GLOBAL_THREAD_POOL: OnceCell`, `spawn` `:128`, `pending_count` `:141`, `init_thread_pool` `:152`, `shutdown` `:161` |

Queue capacity (the 1024-slot blocking send) is modelled separately in
`Locks.lean` (`pool_deadlock_reachable`); here the queue is unbounded, which is
what matters for draining and shutdown. Workers are interchangeable: every
`recv` pops the queue head, so the order jobs start in is the queue order
whichever worker takes them.
-/
namespace SC.Pool

structure PoolS where
  queue : List (Option Nat)
  alive : Nat
  executed : List Nat
  size : Nat
  deriving DecidableEq, Repr

def PoolS.new (size : Nat) : PoolS := ⟨[], size, [], size⟩

/-- One worker receive. A panicking job is caught (`catch_unwind`), so the
worker survives either way. -/
def PoolS.step (s : PoolS) : Option PoolS :=
  if s.alive = 0 then none else
  match s.queue with
  | [] => none
  | some j :: r => some { s with queue := r, executed := s.executed ++ [j] }
  | none :: r => some { s with queue := r, alive := s.alive - 1 }

def run : Nat → PoolS → PoolS
  | 0, s => s
  | n + 1, s => match s.step with
    | some s' => run n s'
    | none => s

/-- `spawn`: enqueue, or drop-and-log when the channel is disconnected. -/
def PoolS.spawn (s : PoolS) (j : Nat) : PoolS :=
  if s.alive = 0 then s else { s with queue := s.queue ++ [some j] }

def PoolS.pendingCount (s : PoolS) : Nat := s.queue.length

def PoolS.shutdownSend (s : PoolS) : PoolS := { s with queue := s.queue ++ List.replicate s.size none }

theorem new_spec (n : Nat) : (PoolS.new n).alive = n ∧ (PoolS.new n).pendingCount = 0 := ⟨rfl, rfl⟩

theorem pendingCount_spawn (s : PoolS) (j : Nat) (h : s.alive ≠ 0) :
    (s.spawn j).pendingCount = s.pendingCount + 1 := by
  simp [PoolS.spawn, PoolS.pendingCount, h]

/-- A job spawned after every worker exited is dropped, never run, and does
not panic (the `eprintln!` path, `threadpool.rs:96-103`). -/
theorem spawn_after_exit (s : PoolS) (j : Nat) (h : s.alive = 0) : s.spawn j = s := by
  simp [PoolS.spawn, h]

theorem run_jobs (n a : Nat) : ∀ (q : List Nat) (rest : List (Option Nat)) (e : List Nat) (sz : Nat),
    run (q.length + n) ⟨q.map some ++ rest, a + 1, e, sz⟩ = run n ⟨rest, a + 1, e ++ q, sz⟩
  | [], rest, e, sz => by simp
  | j :: q, rest, e, sz => by
    rw [show (j :: q).length + n = (q.length + n) + 1 by simp; omega]
    simp only [run, PoolS.step, List.map_cons, List.cons_append, Nat.add_one_ne_zero, ite_false]
    rw [run_jobs n a q rest (e ++ [j]) sz]
    simp

theorem run_sentinels : ∀ (n : Nat) (e : List Nat) (sz : Nat),
    run n ⟨List.replicate n none, n, e, sz⟩ = ⟨[], 0, e, sz⟩
  | 0, e, sz => rfl
  | n + 1, e, sz => by
    simp only [run, PoolS.step, List.replicate_succ, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel]
    exact run_sentinels n e sz

/-- **PROVEN** (`shutdown` is clean): with every worker alive and `q` queued,
sending one `None` per worker and letting the workers run executes every
queued job exactly once, in queue order, then every worker exits (so each
`join` returns) and the queue is empty. -/
theorem shutdown_drains (q : List Nat) (n : Nat) (e : List Nat) (hn : 0 < n) :
    run (q.length + n) (PoolS.shutdownSend ⟨q.map some, n, e, n⟩) = ⟨[], 0, e ++ q, n⟩ := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  simp only [PoolS.shutdownSend]
  rw [run_jobs (m + 1) m q, run_sentinels]

/-! ## The global `OnceCell` -/

inductive Res (α : Type) | ok (a : α) | panic

def Global := Option PoolS

/-- `init_thread_pool`: `Ok` the first time, `Err(())` afterwards (pool kept). -/
def init (g : Global) (n : Nat) : Except Unit Global :=
  match g with
  | none => .ok (some (PoolS.new n))
  | some _ => .error ()

def gSpawn (g : Global) (j : Nat) : Res Global :=
  match g with | none => .panic | some p => .ok (some (p.spawn j))
def gPending (g : Global) : Res Nat :=
  match g with | none => .panic | some p => .ok p.pendingCount
/-- `shutdown`: a no-op when uninitialised. -/
def gShutdown (g : Global) : Global := g.map PoolS.shutdownSend

theorem global_spec (n j : Nat) (p : PoolS) :
    init none n = .ok (some (PoolS.new n)) ∧ init (some p) n = .error () ∧
    (match gSpawn none j with | .panic => True | .ok _ => False) ∧
    (match gPending none with | .panic => True | .ok _ => False) ∧
    gShutdown none = none ∧ gPending (some p) = .ok p.pendingCount := by
  refine ⟨rfl, rfl, trivial, trivial, rfl, rfl⟩

end SC.Pool
