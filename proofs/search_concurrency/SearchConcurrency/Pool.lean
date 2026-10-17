/-
# `graph/src/threadpool.rs` (origin/main b080a44c1): pool lifecycle and `QueueSlot`

| Lean | Rust |
| --- | --- |
| `PoolS`, `PoolS.new` | `ThreadPool::new` `threadpool.rs:59` (since #3014 an **unbounded** MPMC queue, `unbounded_blocking()`; `size` workers, each holding a receiver clone) |
| `PoolS.step` | a worker's `while let Ok(job) = rx.recv()` loop body (`:70-84`): `Some(j)` runs (panics contained), `None` exits |
| `PoolS.spawn` | `ThreadPool::spawn` `:97` (the send never waits — unbounded — and fails only once every receiver is gone) |
| `PoolS.shutdown`, `run` | `ThreadPool::shutdown` `:119` (one `None` per worker, then `join`) |
| `Global` | `GLOBAL_THREAD_POOL: OnceCell`, `spawn` `:132`, `init_thread_pool` `:148`, `shutdown` `:157` |
| `acquireLoop` | `QueueSlot::try_acquire_in` `:189` (the CAS loop), `try_acquire` `:185` (on `QUEUED_QUERIES`) |
| `Loc`, `tr`, `Sys` | a `QueueSlot`'s life: taken by `admit` (`src/graph_core.rs:1007`), carried in the pool job, then (writes) in `WriteMessage.slot`, released by `Drop` (`:205`) |

`pending_count` (both) was removed by #3014: the queue is bounded by admission, not by
the channel. Workers are interchangeable: every `recv` pops the queue head, so the order
jobs start in is the queue order whichever worker takes them.
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

/-- `spawn`: enqueue (never waits: the channel is unbounded), or drop-and-log when the
channel is disconnected. -/
def PoolS.spawn (s : PoolS) (j : Nat) : PoolS :=
  if s.alive = 0 then s else { s with queue := s.queue ++ [some j] }

def PoolS.shutdownSend (s : PoolS) : PoolS := { s with queue := s.queue ++ List.replicate s.size none }

theorem new_spec (n : Nat) : (PoolS.new n).alive = n ∧ (PoolS.new n).queue = [] := ⟨rfl, rfl⟩

/-- **`spawn` never blocks** (#3014): with any worker alive, whatever is queued, the job
is appended — there is no capacity to wait for. -/
theorem spawn_never_blocks (s : PoolS) (j : Nat) (h : s.alive ≠ 0) :
    (s.spawn j).queue = s.queue ++ [some j] := by
  simp [PoolS.spawn, h]

/-- A job spawned after every worker exited is dropped, never run, and does
not panic (the `eprintln!` path, `threadpool.rs:104-111`). -/
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
/-- `shutdown`: a no-op when uninitialised. -/
def gShutdown (g : Global) : Global := g.map PoolS.shutdownSend

theorem global_spec (n j : Nat) (p : PoolS) :
    init none n = .ok (some (PoolS.new n)) ∧ init (some p) n = .error () ∧
    (match gSpawn none j with | .panic => True | .ok _ => False) ∧
    gShutdown none = none ∧ (match gSpawn (some p) j with | .ok g => g = some (p.spawn j) | .panic => False) := by
  refine ⟨rfl, rfl, trivial, rfl, rfl⟩

/-! ## `QueueSlot::try_acquire_in` (`:189-201`)

`let mut n = counter.load(); while n < max { match counter.compare_exchange_weak(n, n + 1)
{ Ok(_) => return Some(..), Err(seen) => n = seen } } None`.

Other threads move the counter between this thread's atomic accesses, so the loop is run
against the sequence `obs` of values the counter holds at each `compare_exchange_weak`
(a spurious weak failure behaves like an observation of a different value: it only
delays). `n` is the last value seen. -/

inductive Acq | ok (newCount : Nat) | full | pending
  deriving DecidableEq, Repr

def acquireLoop (max : Nat) : Nat → List Nat → Acq
  | n, obs =>
    if n < max then
      match obs with
      | [] => .pending                                   -- still spinning: lock-free, not wait-free
      | c :: cs => if c = n then .ok (n + 1) else acquireLoop max c cs
    else .full

/-- **Admission never overshoots**: a successful acquire moves the counter from some
value below `max` to that value + 1, so the count it leaves is at most `max`. -/
theorem acquire_bounded (max n : Nat) (obs : List Nat) (v : Nat) (h : acquireLoop max n obs = .ok v) :
    v ≤ max ∧ ∃ c, v = c + 1 ∧ c < max ∧ (c = n ∨ c ∈ obs) := by
  induction obs generalizing n with
  | nil => unfold acquireLoop at h; split at h <;> cases h
  | cons c cs ih =>
    unfold acquireLoop at h
    split at h
    · rename_i hn
      simp only at h
      split at h
      · cases h; exact ⟨by omega, n, rfl, hn, Or.inl rfl⟩
      · obtain ⟨h1, d, h2, h3, h4⟩ := ih c h
        refine ⟨h1, d, h2, h3, ?_⟩
        rcases h4 with rfl | h4
        · exact Or.inr (List.mem_cons_self ..)
        · exact Or.inr (List.mem_cons_of_mem _ h4)
    · cases h

/-- A refusal (`None` → "Max pending queries exceeded") wrote nothing, and comes without
waiting once the counter is seen at or above `max`. -/
theorem acquire_full (max n : Nat) (obs : List Nat) (h : max ≤ n) : acquireLoop max n obs = .full := by
  unfold acquireLoop; simp; omega

/-! ## Exactly-once release (`Drop for QueueSlot`, `:205`)

The only release is `Drop`, and a `QueueSlot` is moved, never copied, so each path that
ends a queued query's life drops it exactly once. Its life, with every path #3014 lists: -/

inductive Loc | main | job | running | writeMsg | writeQ | released
  deriving DecidableEq, Repr

inductive SEv
  | spawnOk        -- `spawn` sent the job (`graph_core.rs:1077`)
  | spawnFail      -- send failed: the job (and its slot) dropped (`threadpool.rs:104`)
  | poolDropped    -- pool torn down with the job still queued
  | start          -- a worker starts the job (`let mut slot = Some(slot)`, `:1079`)
  | readStarts     -- `execute_query`/`execute_profile` run a read: `admission.take()` (`:631`, `:697`)
  | failsOrPanics  -- an error before the write is queued: `slot.filter(..)` drops it (`:1126`); or unwinding
  | isWrite        -- a write: the slot moves into `WriteMessage.slot` (`:1160`)
  | sendOk         -- `tg.sender.send(msg)` (unbounded; `:1163`)
  | sendFail       -- the send failed: the message and its slot are dropped
  | graphFreed     -- the graph freed with writes still queued: channel and messages dropped
  | dequeued       -- the drainer took it: `drop(slot)` (`:1662`)
  deriving DecidableEq, Repr

/-- One move of the slot: where it goes and how many releases (`fetch_sub`) that is. -/
def tr : Loc → SEv → Option (Loc × Nat)
  | .main, .spawnOk => some (.job, 0)
  | .main, .spawnFail => some (.released, 1)
  | .job, .poolDropped => some (.released, 1)
  | .job, .start => some (.running, 0)
  | .running, .readStarts => some (.released, 1)
  | .running, .failsOrPanics => some (.released, 1)
  | .running, .isWrite => some (.writeMsg, 0)
  | .writeMsg, .sendOk => some (.writeQ, 0)
  | .writeMsg, .sendFail => some (.released, 1)
  | .writeQ, .graphFreed => some (.released, 1)
  | .writeQ, .dequeued => some (.released, 1)
  | _, _ => none

/-- Follow a path; `none` if some event is impossible there. Returns the final location and
the releases made. -/
def follow : Loc → List SEv → Option (Loc × Nat)
  | l, [] => some (l, 0)
  | l, e :: es => match tr l e with
    | none => none
    | some (l', r) => (follow l' es).map fun (l'', r') => (l'', r + r')

theorem released_stuck (e : SEv) : tr .released e = none := by cases e <;> rfl

theorem tr_released (l : Loc) (e : SEv) (l' : Loc) (r : Nat) (h : tr l e = some (l', r)) :
    (r = 1 ↔ l' = .released) ∧ r ≤ 1 ∧ l ≠ .released := by
  cases l <;> cases e <;> simp [tr] at h <;> obtain ⟨rfl, rfl⟩ := h <;> simp

/-- **Every admitted slot is released exactly once, on every path**: along any possible
path of a queued query, the releases made are 1 if it ended released and 0 if it is
still queued or running — never 2. -/
theorem release_exactly_once : ∀ (l : Loc) (es : List SEv) (l' : Loc) (r : Nat), l ≠ .released →
    follow l es = some (l', r) → (r = if l' = .released then 1 else 0)
  | l, [], l', r, hl, h => by
    simp [follow] at h; obtain ⟨rfl, rfl⟩ := h; simp [hl]
  | l, e :: es, l', r, hl, h => by
    simp only [follow] at h
    split at h
    · cases h
    · rename_i l1 r1 ht
      obtain ⟨h1, h2, _⟩ := tr_released l e l1 r1 ht
      by_cases hr : l1 = .released
      · subst hr
        cases es with
        | nil => simp [follow] at h; obtain ⟨rfl, rfl⟩ := h; simp; exact h1.2 rfl
        | cons e' es' => simp [follow, released_stuck] at h
      · have hr1 : r1 = 0 := by have : r1 ≠ 1 := fun e => hr (h1.1 e); omega
        cases hf : follow l1 es with
        | none => rw [hf] at h; cases h
        | some p =>
          obtain ⟨l2, r2⟩ := p
          rw [hf] at h; simp at h; obtain ⟨rfl, rfl⟩ := h
          rw [release_exactly_once l1 es l2 r2 hr hf, hr1]; simp

/-- Every path #3014 names ends in exactly one release. -/
theorem paths_release_once :
    follow .main [.spawnFail] = some (.released, 1) ∧
    follow .main [.spawnOk, .poolDropped] = some (.released, 1) ∧
    follow .main [.spawnOk, .start, .readStarts] = some (.released, 1) ∧
    follow .main [.spawnOk, .start, .failsOrPanics] = some (.released, 1) ∧
    follow .main [.spawnOk, .start, .isWrite, .sendFail] = some (.released, 1) ∧
    follow .main [.spawnOk, .start, .isWrite, .sendOk, .graphFreed] = some (.released, 1) ∧
    follow .main [.spawnOk, .start, .isWrite, .sendOk, .dequeued] = some (.released, 1) := by
  decide

/-! ## The counter over many slots

`QUEUED_QUERIES` = the number of live (unreleased) slots, so `Drop`'s
`debug_assert!(prev > 0)` holds and the count never underflows. -/

structure Sys where
  count : Nat
  slots : List Loc
  deriving DecidableEq, Repr

def isLive : Loc → Bool
  | .released => false
  | _ => true

def live (s : Sys) : Nat := (s.slots.filter isLive).length

/-- Admission (`try_acquire`, atomic here — `acquire_bounded` covers the loop). -/
def admitS (max : Nat) (s : Sys) : Option Sys :=
  if s.count < max then some ⟨s.count + 1, .main :: s.slots⟩ else none

/-- Slot `i` takes event `e`; a release is `fetch_sub(1)`. -/
def moveS (s : Sys) (i : Nat) (e : SEv) : Option Sys :=
  match s.slots[i]? with
  | none => none
  | some l => match tr l e with
    | none => none
    | some (l', r) => some ⟨s.count - r, s.slots.set i l'⟩

def ind (b : Bool) : Nat := if b then 1 else 0

theorem live_set (l : List Loc) (i : Nat) (a b : Loc) (h : l[i]? = some a) :
    ((l.set i b).filter isLive).length + ind (isLive a) = (l.filter isLive).length + ind (isLive b) := by
  induction l generalizing i with
  | nil => simp at h
  | cons x xs ih =>
    cases i with
    | zero =>
      simp at h; subst h
      simp only [List.set_cons_zero, List.filter_cons]
      cases hx : isLive x <;> cases hb : isLive b <;> simp [ind] <;> omega
    | succ j =>
      simp only [List.set_cons_succ, List.filter_cons]
      have := ih j (by simpa using h)
      split <;> (try simp only [List.length_cons]) <;> omega

theorem isLive_iff (l : Loc) : isLive l = false ↔ l = .released := by cases l <;> simp [isLive]

/-- **The count is the number of live slots**, whatever the interleaving of admissions
and slot moves; so every release finds `count ≥ 1` (`debug_assert!(prev > 0)`), and
admission keeps `count ≤ max`. -/
theorem count_eq_live (max : Nat) (s s' : Sys) (h : s.count = live s) :
    (admitS max s = some s' → s'.count = live s' ∧ s'.count ≤ max) ∧
    (∀ i e, moveS s i e = some s' → s'.count = live s' ∧
      ∀ l l' r, s.slots[i]? = some l → tr l e = some (l', r) → r = 1 → 1 ≤ s.count) := by
  refine ⟨fun ha => ?_, fun i e hm => ?_⟩
  · unfold admitS at ha
    split at ha
    · cases ha
      refine ⟨?_, ?_⟩
      · simp [live, List.filter_cons, isLive] at h ⊢; omega
      · show s.count + 1 ≤ max; omega
    · cases ha
  · unfold moveS at hm
    split at hm
    · cases hm
    · rename_i l hl
      split at hm
      · cases hm
      · rename_i l' r ht
        cases hm
        obtain ⟨h1, h2, h3⟩ := tr_released l e l' r ht
        have hls := live_set s.slots i l l' hl
        have hla : isLive l = true := by
          cases hh : isLive l
          · exact absurd ((isLive_iff l).1 hh) h3
          · rfl
        rw [hla] at hls
        simp only [live] at h
        refine ⟨?_, ?_⟩
        · show s.count - r = ((s.slots.set i l').filter isLive).length
          cases hb : isLive l'
          · have : r = 1 := h1.2 ((isLive_iff l').1 hb)
            rw [hb] at hls; simp [ind] at hls; omega
          · have : r ≠ 1 := fun e' => by
              have := (isLive_iff l').2 (h1.1 e'); rw [hb] at this; cases this
            have : r = 0 := by omega
            rw [hb] at hls; simp [ind] at hls; omega
        · intro a b c ha hb hc
          rw [hl] at ha; cases ha
          rw [ht] at hb; cases hb
          have hf := (isLive_iff l').2 (h1.1 hc)
          rw [hf] at hls; simp only [ind, ite_true, ite_false, Bool.false_eq_true] at hls
          omega

end SC.Pool
