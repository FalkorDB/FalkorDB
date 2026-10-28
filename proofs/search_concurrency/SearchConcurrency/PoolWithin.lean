import SearchConcurrency.Pool

/-
# `graph/src/threadpool.rs` (main ca36bfa15, #3502): named workers, `shutdown_within`

| Lean | Rust |
| --- | --- |
| `workerName`, `spawnNames` | `ThreadPool::with_names` `threadpool.rs:79-123` (`builder.name(format!("{name}-{i}"))`, `i in 0..size`); `new` `:67` is `with_names(size, None)`, `named` `:72` is `with_names(size, Some(name))` |
| `Fin`, `Fin.done` | a worker's `JoinHandle`: `is_finished()` (std; AXIOMATISED as a monotone predicate of time, see below) |
| `waitLoop` | `shutdown_within`'s `while Instant::now() < deadline && !workers.iter().all(JoinHandle::is_finished) { thread::sleep(1ms) }` `:167-169` |
| `drain` | the `for worker in workers.drain(..)` loop `:171-178`: a finished handle is joined, an unfinished one is dropped (detached) and clears `all_exited` |
| `within` | `ThreadPool::shutdown_within` `:159-181` (sentinels sent first, `:163-165`) |
| `holdEnd` | `ThreadPool::shutdown` `:146-153`: holds the `workers` lock across every `join` |
| `PoolS.shutdownSend`, `PoolS.spawn` (Pool.lean) | the sentinel loop shared by `shutdown` and `shutdown_within`; `spawn` `:124` |

OS and std primitives are hypotheses, not axioms: `JoinHandle::is_finished` is
monotone in time (`Fin.done_mono`, by construction: a worker finishes at one instant or
not at all); `thread::sleep(1ms)` advances the clock by at least 1 ms (`h1`) and at most
`s` ms (`hs`, scheduler overshoot). `Builder::spawn` failing makes `new` panic through
`.expect` (as `thread::spawn` did before #3502); the local `sender` is then dropped, every
already-started worker's `recv` fails and it exits, so nothing leaks.
Abstraction: the drain loop's `is_finished` checks all read the exit instant of the wait
loop. A real check runs later, and `is_finished` is monotone, so a worker model-detached may
really be joined (more finished): `within_true_all_finished` and `within_in_time` still
hold; `within_false_unfinished` is exact for a worker not finished at any later time.

Findings on the new code (both confirmed live, `repro/threadpool_within.rs`):
* **BUG** `shutdown_within` after a timed-out `shutdown_within` returns `true` while the
  straggler still runs (`second_call_true_despite_straggler`): the first call drained the
  handle list, so the second sees an empty list and `all_exited` stays `true`.
* **BUG** a `shutdown_within(t)` called while another thread is inside `shutdown` blocks on
  the `workers` lock for as long as `shutdown`'s `join` (`:149-150`): with a job that never
  returns, never (`within_blocks_behind_shutdown`), so the time limit does not hold.
* Benign, same as `shutdown`: a job spawned after the sentinels is accepted (send `Ok`, no
  log) and never runs (`late_job_lost`).
-/
namespace SC.PoolWithin
open SC.Pool

/-! ## Named workers -/

def workerName (name : Option String) (i : Nat) : Option String :=
  name.map fun n => n ++ "-" ++ toString i

def spawnNames (name : Option String) (size : Nat) : List (Option String) :=
  (List.range size).map (workerName name)

/-- One name per worker: `size` of them, worker `i` named `{name}-{i}` (numbered from 0). -/
theorem spawnNames_length (name : Option String) (size : Nat) : (spawnNames name size).length = size := by
  simp [spawnNames]

theorem spawnNames_get (n : String) (size i : Nat) (h : i < size) :
    (spawnNames (some n) size)[i]? = some (some (n ++ "-" ++ toString i)) := by
  simp [spawnNames, workerName, h]

/-- `new` leaves every worker unnamed. -/
theorem spawnNames_none (size : Nat) : ∀ x ∈ spawnNames none size, x = none := by
  intro x hx
  simp only [spawnNames, List.mem_map] at hx
  obtain ⟨a, _, h⟩ := hx
  rw [← h]; rfl

-- The Rust test `named_workers_carry_the_name`.
#guard spawnNames (some "named-pool") 2 = [some "named-pool-0", some "named-pool-1"]

/-! ## `shutdown_within` -/

/-- A `JoinHandle`: the instant `is_finished()` turns true; `none` is a job that has not
returned at any instant considered. -/
abbrev Fin := Option Nat

def Fin.done (f : Fin) (t : Nat) : Bool :=
  match f with
  | some x => decide (x ≤ t)
  | none => false

theorem Fin.done_mono (f : Fin) {t u : Nat} (h : t ≤ u) (hd : f.done t = true) : f.done u = true := by
  cases f <;> simp_all [Fin.done] <;> omega

def allDone (ws : List Fin) (t : Nat) : Bool := ws.all (·.done t)

/-- The poll loop. `fuel` only makes it structural; `next` is the clock after one
`thread::sleep(1ms)`. -/
def waitLoop (ws : List Fin) (deadline : Nat) (next : Nat → Nat) : Nat → Nat → Nat
  | 0, now => now
  | n + 1, now =>
    if now < deadline ∧ allDone ws now = false then waitLoop ws deadline next n (next now) else now

/-- **The loop ends when its test fails**: given one step of fuel per millisecond of
remaining time, the exit instant is at or past the deadline, or every worker is finished. -/
theorem waitLoop_exit (ws : List Fin) (deadline : Nat) (next : Nat → Nat)
    (h1 : ∀ t, t + 1 ≤ next t) :
    ∀ (fuel now : Nat), deadline - now ≤ fuel →
      deadline ≤ waitLoop ws deadline next fuel now ∨ allDone ws (waitLoop ws deadline next fuel now) = true := by
  intro fuel
  induction fuel with
  | zero => intro now h; simp [waitLoop]; omega
  | succ n ih =>
    intro now h
    simp only [waitLoop]
    split
    · rename_i hc
      have := h1 now
      exact ih (next now) (by omega)
    · rename_i hc
      by_cases hl : now < deadline
      · right; simpa [hl] using hc
      · left; omega

/-- **Bounded overshoot**: if each sleep takes at most `s` ms (`s ≥ 1`), the loop exits
strictly before `deadline + s` unless it started later than that. -/
theorem waitLoop_bound (ws : List Fin) (deadline s : Nat) (next : Nat → Nat)
    (hs : ∀ t, next t ≤ t + s) (hs1 : 1 ≤ s) :
    ∀ (fuel now : Nat), waitLoop ws deadline next fuel now ≤ max now (deadline + s - 1) := by
  intro fuel
  induction fuel with
  | zero => intro now; simp only [waitLoop]; exact Nat.le_max_left _ _
  | succ n ih =>
    intro now
    simp only [waitLoop]
    split
    · rename_i hc
      have h2 := ih (next now)
      have h3 := hs now
      have : next now ≤ deadline + s - 1 := by omega
      rw [Nat.max_eq_right this] at h2
      exact Nat.le_trans h2 (Nat.le_max_right _ _)
    · exact Nat.le_max_left _ _

/-- The drain loop at instant `t`: (`all_exited`, handles joined, handles detached). -/
def drain (ws : List Fin) (t : Nat) : Bool × List Fin × List Fin :=
  ((ws.filter fun w => !w.done t).isEmpty, ws.filter (·.done t), ws.filter fun w => !w.done t)

/-- Result of one `shutdown_within`: (`all_exited`, exit instant, `workers` afterwards,
detached). The sentinels are sent before the wait (`PoolS.shutdownSend`). -/
def within (ws : List Fin) (start timeout fuel : Nat) (next : Nat → Nat) :
    Bool × Nat × List Fin × List Fin :=
  let tx := waitLoop ws (start + timeout) next fuel start
  ((drain ws tx).1, tx, [], (drain ws tx).2.2)

/-- **The time limit holds** (on an uncontended lock): the call returns before
`start + timeout + s`, for any pool and any job behaviour. -/
theorem within_in_time (ws : List Fin) (start timeout s fuel : Nat) (next : Nat → Nat)
    (hs : ∀ t, next t ≤ t + s) (hs1 : 1 ≤ s) :
    (within ws start timeout fuel next).2.1 < start + timeout + s := by
  have := waitLoop_bound ws (start + timeout) s next hs hs1 fuel start
  simp only [within]
  omega

/-- **Soundness of `true`** is the strong direction here: every worker's job returned by
the deadline ⇒ `true` (nothing is detached). -/
theorem within_true_all_finished (ws : List Fin) (start timeout fuel : Nat) (next : Nat → Nat)
    (h1 : ∀ t, t + 1 ≤ next t) (hf : timeout ≤ fuel)
    (hall : ∀ w ∈ ws, ∃ f, w = some f ∧ f ≤ start + timeout) :
    (within ws start timeout fuel next).1 = true := by
  have hx := waitLoop_exit ws (start + timeout) next h1 fuel start (by omega)
  simp only [within, drain, List.isEmpty_iff, List.filter_eq_nil_iff]
  intro w hw
  obtain ⟨f, rfl, hf'⟩ := hall w hw
  rcases hx with hx | hx
  · simp [Fin.done]; omega
  · have := List.all_eq_true.mp hx _ hw
    simp [this]

/-- A `false` answer is exact: some worker was still unfinished at the exit instant. -/
theorem within_false_unfinished (ws : List Fin) (start timeout fuel : Nat) (next : Nat → Nat)
    (h : (within ws start timeout fuel next).1 = false) :
    ∃ w ∈ ws, w.done (within ws start timeout fuel next).2.1 = false := by
  simp only [within, drain] at h ⊢
  refine Classical.byContradiction fun hno => ?_
  have hall : ∀ w ∈ ws, ¬ (w.done (waitLoop ws (start + timeout) next fuel start) = false) := by
    intro w hw hd; exact hno ⟨w, hw, hd⟩
  have : (ws.filter fun w => !w.done (waitLoop ws (start + timeout) next fuel start)).isEmpty = true := by
    rw [List.isEmpty_iff, List.filter_eq_nil_iff]
    intro w hw
    have := hall w hw
    simpa using this
  rw [this] at h
  exact Bool.noConfusion h

theorem filter_split {α : Type} (l : List α) (p : α → Bool) :
    (l.filter p).length + (l.filter fun x => !p x).length = l.length := by
  induction l with
  | nil => rfl
  | cons x r ih => by_cases hp : p x = true <;> simp [hp] <;> omega

/-- **No worker handle is lost**: every handle is joined or detached, none both, and
`workers` is empty afterwards. -/
theorem within_partition (ws : List Fin) (start timeout fuel : Nat) (next : Nat → Nat) :
    (drain ws (within ws start timeout fuel next).2.1).2.1.length +
      (within ws start timeout fuel next).2.2.2.length = ws.length ∧
    (within ws start timeout fuel next).2.2.1 = [] := by
  refine ⟨?_, rfl⟩
  simp only [within, drain]
  exact filter_split ws _

/-- A job that has not returned by the exit instant (e.g. blocked for good) always makes
`shutdown_within` answer `false`, and its handle is detached. -/
theorem blocked_reported (ws : List Fin) (start timeout fuel : Nat) (next : Nat → Nat)
    (h : none ∈ ws) : (within ws start timeout fuel next).1 = false := by
  simp only [within, drain]
  cases hb : (ws.filter fun w => !w.done (waitLoop ws (start + timeout) next fuel start)).isEmpty
  · rfl
  · rw [List.isEmpty_iff, List.filter_eq_nil_iff] at hb
    have := hb none h
    simp [Fin.done] at this

/-- **BUG (second call)**: the first call emptied `workers`, so a second `shutdown_within`
joins nothing and answers `true` although the straggler is still running (live repro
`repro/threadpool_within.rs`, case A: `first=false second=true`). The wait loop is also
skipped (`all` over an empty list), so the answer is immediate. -/
theorem within_empty_true (start timeout fuel : Nat) (next : Nat → Nat) :
    (within [] start timeout fuel next).1 = true := by
  simp [within, drain]

theorem second_call_true_despite_straggler (ws : List Fin) (start timeout fuel : Nat) (next : Nat → Nat)
    (start2 timeout2 fuel2 : Nat) (h : none ∈ ws) :
    (within ws start timeout fuel next).1 = false ∧
    (within (within ws start timeout fuel next).2.2.1 start2 timeout2 fuel2 next).1 = true ∧
    (within ws start timeout fuel next).2.2.2 ≠ [] := by
  refine ⟨blocked_reported ws start timeout fuel next h, ?_, ?_⟩
  · have : (within ws start timeout fuel next).2.2.1 = [] := rfl
    rw [this]; exact within_empty_true _ _ _ _
  · intro hd
    simp only [within, drain] at hd
    rw [List.filter_eq_nil_iff] at hd
    have := hd none h
    simp [Fin.done] at this

/-! ### Contention with `shutdown`: the `workers` lock -/

/-- `Mutex` held while `shutdown` joins: the instant it is released is the time of the
last job to return. -/
def holdStep (acc : Option Nat) (w : Fin) : Option Nat :=
  match acc, w with
  | some a, some f => some (max a f)
  | _, _ => none

/-- `shutdown` holds the `workers` lock across every `join`: until the last handle's job
has returned. `none`: never. -/
def holdEnd (ws : List Fin) : Option Nat := ws.foldl holdStep (some 0)

private theorem fold_none (ws : List Fin) : ws.foldl holdStep none = none := by
  induction ws with
  | nil => rfl
  | cons w r ih => simpa [holdStep] using ih

/-- **BUG (lock contention)**: a `shutdown_within(t)` that arrives while a blocking
`shutdown` is inside its `join`s queues on the `workers` lock, and gets it only when
`holdEnd` says so: never, if a job never returns, whatever `t` is (live repro, case B:
no answer after 3 s for a 50 ms limit). Even for returning jobs it gets the lock no
earlier than the slowest job (`holdEnd_ge`), so the limit is not a limit. -/
theorem within_blocks_behind_shutdown (ws : List Fin) (h : none ∈ ws) : holdEnd ws = none := by
  have : ∀ acc, ws.foldl holdStep acc = none := by
    induction ws with
    | nil => simp at h
    | cons w r ih =>
      intro acc
      cases w with
      | none =>
        have : holdStep acc none = none := by cases acc <;> rfl
        simpa [this] using fold_none r
      | some f =>
        simp only [List.mem_cons, reduceCtorEq, false_or] at h
        simp only [List.foldl_cons]
        rcases hacc : acc with _ | a
        · simpa [holdStep] using fold_none r
        · exact ih h _
  exact this _

private theorem hold_ge : ∀ (ws : List Fin) (acc : Option Nat) (e : Nat),
    ws.foldl holdStep acc = some e → (∀ a, acc = some a → a ≤ e) ∧ ∀ f, some f ∈ ws → f ≤ e := by
  intro ws
  induction ws with
  | nil => intro acc e h; exact ⟨fun a ha => by simp_all, fun f hf => by simp at hf⟩
  | cons w r ih =>
    intro acc e h
    simp only [List.foldl_cons] at h
    obtain ⟨h1, h2⟩ := ih _ _ h
    cases acc with
    | none =>
      have : holdStep none w = none := by cases w <;> rfl
      rw [this, fold_none] at h; cases h
    | some a =>
      cases w with
      | none => simp [holdStep, fold_none] at h
      | some f =>
        have := h1 (max a f) (by simp [holdStep])
        refine ⟨fun a' ha' => by cases ha'; omega, ?_⟩
        intro g hg
        simp only [List.mem_cons, Option.some.injEq] at hg
        rcases hg with rfl | hg
        · omega
        · exact h2 g hg

theorem holdEnd_ge (ws : List Fin) (e : Nat) (h : holdEnd ws = some e) :
    ∀ f, some f ∈ ws → f ≤ e := (hold_ge ws (some 0) e h).2

/-! ### The sentinel protocol (queue model of `Pool.lean`) -/

theorem run_none_prefix : ∀ (m k : Nat) (e : List Nat) (sz : Nat),
    run m ⟨List.replicate (m + k) none, m + k, e, sz⟩ = ⟨List.replicate k none, k, e, sz⟩
  | 0, k, e, sz => by simp [run]
  | m + 1, k, e, sz => by
    have : m + 1 + k = (m + k) + 1 := by omega
    rw [this]
    simp only [run, PoolS.step, List.replicate_succ, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel]
    exact run_none_prefix m k e sz

/-- **No worker leak**: `m` workers are free and `k` are stuck in a job that has not
returned (the ones `shutdown_within` detaches). The `m + k` sentinels still let the free
ones run every queued job and exit, and leave exactly `k` sentinels in the queue: one for
each straggler, which takes it and exits as soon as its job returns (`run k`). So a
detached worker is never stranded, and no queued job is lost. -/
theorem detached_workers_exit (q : List Nat) (m k a : Nat) (e : List Nat) (hn : m + k = a + 1) :
    run (q.length + m) ⟨q.map some ++ List.replicate (m + k) none, m + k, e, m + k⟩ =
      ⟨List.replicate k none, k, e ++ q, m + k⟩ ∧
    run k ⟨List.replicate k none, k, e ++ q, m + k⟩ = ⟨[], 0, e ++ q, m + k⟩ := by
  refine ⟨?_, run_sentinels k (e ++ q) (m + k)⟩
  have h1 := run_none_prefix m k (e ++ q) (m + k)
  have h2 := run_jobs m a q (List.replicate (m + k) none) e (m + k)
  rw [hn] at h1 h2 ⊢
  rw [h2]
  rw [← hn] at h1 ⊢
  exact h1

theorem run_sentinels_tail (t : List (Option Nat)) : ∀ (n : Nat) (e : List Nat) (sz : Nat),
    run n ⟨List.replicate n none ++ t, n, e, sz⟩ = ⟨t, 0, e, sz⟩
  | 0, e, sz => by simp [run]
  | n + 1, e, sz => by
    simp only [run, PoolS.step, List.replicate_succ, List.cons_append, Nat.add_one_ne_zero, ite_false,
      Nat.add_sub_cancel]
    exact run_sentinels_tail t n e sz

/-- **A job spawned after the sentinels is lost, silently** (benign; `shutdown` is the
same): `spawn` still sees live receivers, so the send is `Ok` and nothing is logged, but
the job sits behind the `None`s, never runs, and is freed with the channel. -/
theorem late_job_lost (q : List Nat) (n : Nat) (e : List Nat) (j : Nat) (hn : 0 < n) :
    run (q.length + n) ((PoolS.shutdownSend ⟨q.map some, n, e, n⟩).spawn j) =
      ⟨[some j], 0, e ++ q, n⟩ := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  simp only [PoolS.spawn, PoolS.shutdownSend, Nat.add_one_ne_zero, ite_false, List.append_assoc]
  rw [run_jobs (m + 1) m q _ e (m + 1), run_sentinels_tail]

end SC.PoolWithin
