/-!
# `src/allocator.rs` — per-thread allocation accounting

`ThreadCountingAllocator` (`:31-75`) forwards every request to `RedisAlloc` (FFI: here an
input — whether it returned a non-null pointer, and the pointer itself is passed through
unchanged) and, when the calling thread's `TRACKING_ENABLED` is set, adds `layout.size()`
to that thread's `THREAD_ALLOCATED` / `THREAD_DEALLOCATED`. The helpers (`:80-117`) flip
the flag, read and reset the counters, and compute the net with `saturating_sub`.

| here | there |
| --- | --- |
| `TLS`, `TLS.init`       | the three `thread_local!` cells (`:33-41`; `TRACKING_ENABLED` starts `true`) |
| `Ev.alloc ok n`         | `GlobalAlloc::alloc` (`:44`): counted iff the pointer is non-null and tracking is on |
| `Ev.dealloc n`          | `GlobalAlloc::dealloc` (`:61`): counted iff tracking is on |
| `Ev.enable/disable/reset` | `enable_tracking` (`:80`), `disable_tracking` (`:88`), `reset_counter` (`:114`) |
| `usage`, `net`          | `current_thread_usage` (`:97`), `net_thread_usage` (`:105`) |
| `World`, `World.step`   | one `TLS` per thread: every event touches only its own thread's cells |

`usize` arithmetic: `c.get() + layout.size()` wraps in the release profile (no
`overflow-checks`), so counters are taken mod `2^64`; `no_wrap` shows wrapping needs more
than `2^64` bytes of traffic between resets.
-/
namespace RedisLayer.Allocator

def U : Nat := 2 ^ 64

structure TLS where
  allocated : Nat
  deallocated : Nat
  enabled : Bool
  deriving DecidableEq, Repr

/-- The initial thread-local state: counters 0, tracking **on** (`Cell::new(true)`). -/
def TLS.init : TLS := ⟨0, 0, true⟩

inductive Ev where
  | alloc (nonNull : Bool) (size : Nat)
  | dealloc (size : Nat)
  | enable
  | disable
  | reset
  deriving Repr

/-- One event on the calling thread's cells. -/
def step (t : TLS) : Ev → TLS
  | .alloc ok n => if ok && t.enabled then { t with allocated := (t.allocated + n) % U } else t
  | .dealloc n => if t.enabled then { t with deallocated := (t.deallocated + n) % U } else t
  | .enable => { t with enabled := true }
  | .disable => { t with enabled := false }
  | .reset => { t with allocated := 0, deallocated := 0 }

def run (t : TLS) (es : List Ev) : TLS := es.foldl step t

/-- `GlobalAlloc::alloc` returns `RedisAlloc`'s pointer unchanged (accounting never
alters or replaces it). `p = none` is the null pointer. -/
def allocRet (p : Option Nat) : Option Nat := p

theorem alloc_passthrough (p : Option Nat) : allocRet p = p := rfl

/-- `current_thread_usage` (`:97`). -/
def usage (t : TLS) : Nat × Nat := (t.allocated, t.deallocated)
/-- `net_thread_usage` (`:105`): `allocated.saturating_sub(deallocated)` (Nat `-` is saturating). -/
def net (t : TLS) : Nat := t.allocated - t.deallocated

theorem net_saturates (t : TLS) : net t ≤ t.allocated ∧ (t.allocated ≤ t.deallocated → net t = 0) := by
  simp only [net]; omega

theorem reset_zero (t : TLS) : usage (step t .reset) = (0, 0) ∧ net (step t .reset) = 0 ∧
    (step t .reset).enabled = t.enabled := by
  simp [usage, net, step]

theorem enable_sets (t : TLS) : (step t .enable).enabled = true ∧ usage (step t .enable) = usage t := by
  simp [step, usage]
theorem disable_clears (t : TLS) : (step t .disable).enabled = false ∧ usage (step t .disable) = usage t := by
  simp [step, usage]

/-- A null allocation is never counted (`if !ptr.is_null()`). -/
theorem null_alloc_uncounted (t : TLS) (n : Nat) : step t (.alloc false n) = t := by
  simp [step]

/-- While tracking is off, allocations and frees leave the counters (and flag) alone —
which is why `disable_tracking` before logging keeps the log's own allocations out. -/
theorem disabled_frozen (t : TLS) (h : t.enabled = false) (e : Ev)
    (he : (∃ ok n, e = .alloc ok n) ∨ ∃ n, e = .dealloc n) : step t e = t := by
  rcases he with ⟨ok, n, rfl⟩ | ⟨n, rfl⟩ <;> simp [step, h]

/-- Memory traffic only: no flag change and no reset. -/
def Traffic : Ev → Prop
  | .alloc _ _ => True
  | .dealloc _ => True
  | _ => False

def allocSum : List Ev → Nat
  | [] => 0
  | .alloc true n :: es => n + allocSum es
  | _ :: es => allocSum es

def freeSum : List Ev → Nat
  | [] => 0
  | .dealloc n :: es => n + freeSum es
  | _ :: es => freeSum es

def Bounded (t : TLS) : Prop := t.allocated < U ∧ t.deallocated < U

theorem init_bounded : Bounded TLS.init := by simp [Bounded, TLS.init, U]

theorem step_bounded (t : TLS) (e : Ev) (h : Bounded t) : Bounded (step t e) := by
  obtain ⟨h1, h2⟩ := h
  have hU : 0 < U := by simp [U]
  cases e <;> simp only [step] <;> (try split) <;>
    simp_all [Bounded] <;> exact Nat.mod_lt _ hU

/-- **Traffic with tracking on** adds exactly the successful allocation sizes and the
freed sizes (mod `2^64`), and leaves tracking on. -/
theorem run_traffic (es : List Ev) : ∀ (t : TLS), t.enabled = true → Bounded t → (∀ e ∈ es, Traffic e) →
    run t es = ⟨(t.allocated + allocSum es) % U, (t.deallocated + freeSum es) % U, true⟩ := by
  induction es with
  | nil =>
    intro t h hb _
    obtain ⟨a, d, en⟩ := t
    obtain ⟨h1, h2⟩ := hb
    simp at h h1 h2; subst h
    simp [run, allocSum, freeSum, Nat.mod_eq_of_lt h1, Nat.mod_eq_of_lt h2]
  | cons e es ih =>
    intro t h hb ht
    have hb' := step_bounded t e hb
    have hte : Traffic e := ht e (by simp)
    have hrest : ∀ e ∈ es, Traffic e := fun e he => ht e (by simp [he])
    show run (step t e) es = _
    obtain ⟨a, d, en⟩ := t
    simp at h; subst h
    cases e with
    | alloc ok n =>
      cases ok
      · rw [ih _ (by simp [step]) hb' hrest]; simp [step, allocSum, freeSum]
      · rw [ih _ (by simp [step]) hb' hrest]
        simp only [step, allocSum, freeSum, Bool.and_self, ite_true, TLS.mk.injEq, and_true]
        rw [Nat.mod_add_mod, Nat.add_assoc]
    | dealloc n =>
      rw [ih _ (by simp [step]) hb' hrest]
      simp only [step, allocSum, freeSum, ite_true, TLS.mk.injEq, and_true, true_and]
      rw [Nat.mod_add_mod, Nat.add_assoc]
    | enable => simp [Traffic] at hte
    | disable => simp [Traffic] at hte
    | reset => simp [Traffic] at hte

/-- **Per-query accounting** (`graph_core.rs:1007-1062`: `reset_counter(); enable_tracking();`
… query … `current_thread_usage()`): from any prior thread state, the counters report
exactly the bytes this query's traffic allocated and freed on this thread (below `2^64`
total — wrapping needs 16 EiB of traffic), and `net_thread_usage` is their saturating
difference. -/
theorem query_accounting (t : TLS) (hb : Bounded t) (es : List Ev) (ht : ∀ e ∈ es, Traffic e)
    (ha : allocSum es < U) (hf : freeSum es < U) :
    usage (run t (.reset :: .enable :: es)) = (allocSum es, freeSum es) ∧
    net (run t (.reset :: .enable :: es)) = allocSum es - freeSum es := by
  have e0 : run t (.reset :: .enable :: es) = run ⟨0, 0, true⟩ es := by simp [run, step]
  rw [e0, run_traffic es _ rfl (by simp [Bounded, U]) ht]
  simp [usage, net, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hf]

/-! ## Thread locality -/

/-- Every thread's cells. -/
def World := Nat → TLS

/-- An event on thread `tid` (allocations and frees run on the calling thread). -/
def World.step (w : World) (tid : Nat) (e : Ev) : World :=
  fun t => if t = tid then RedisLayer.Allocator.step (w t) e else w t

/-- Nothing another thread does is visible in this thread's counters: a query's
`net_thread_usage` (the `QUERY_MEM_CAPACITY` check) sees only its own thread's traffic. -/
theorem other_thread_invisible (w : World) (tid me : Nat) (e : Ev) (h : tid ≠ me) :
    (w.step tid e) me = w me := by
  simp [World.step, Ne.symm h]

theorem own_thread_steps (w : World) (tid : Nat) (e : Ev) : (w.step tid e) tid = step (w tid) e := by
  simp [World.step]

/-- A thread that never ran a tracked query still counts (the flag starts `true`): its
counters grow from the first allocation. -/
theorem fresh_thread_counts (n : Nat) (h : n < U) : usage (step TLS.init (.alloc true n)) = (n, 0) := by
  simp [step, TLS.init, usage, Nat.mod_eq_of_lt h]

end RedisLayer.Allocator
