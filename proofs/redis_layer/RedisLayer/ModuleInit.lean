/-!
# Module load: configuration table and event handlers (`src/module_init.rs`, `src/lib.rs`)

Load-time configuration has one source: `graph_init`
(`module_init.rs`) collects the arguments (`to_string_lossy`) and calls
`config::apply_module_args`, refusing the load (`Status::Err`) on any error; `lib.rs` sets `module_args_as_configuration: false`, so redismodule-rs no
longer reads them. That function is modelled in `LoadArgs.lean`; `rust_c_same_accept` proves it
accepts exactly what C's `Config_Init` accepts, for arguments with no embedded NUL, no leading
whitespace in a value and no Rust-only name, given `FsOk` (`File::create` agrees with
`access(W_OK)`); `error_applies_nothing` is a separate property (a refused load applies nothing). `graph_init`'s own
work here is "args → `apply_module_args` → `Err` ⇒ `Status::Err`" (`graphInitArgs`).

`table` (the `redis_module!` `configurations:` list, `lib.rs`) still registers the
`graph.*` redis.conf configs; its defaults are checked below.
-/
namespace RedisLayer.ModuleInit

/-- C's configuration names (`config.c`). -/
def documented : List String :=
  ["TIMEOUT", "TIMEOUT_DEFAULT", "TIMEOUT_MAX", "CACHE_SIZE", "ASYNC_DELETE", "THREAD_COUNT",
   "RESULTSET_SIZE", "OMP_THREAD_COUNT", "VKEY_MAX_ENTITY_COUNT", "MAX_QUEUED_QUERIES",
   "QUERY_MEM_CAPACITY", "DELTA_MAX_PENDING_CHANGES", "NODE_CREATION_BUFFER", "CMD_INFO",
   "MAX_INFO_QUERIES", "EFFECTS_THRESHOLD", "DELAY_INDEXING", "IMPORT_FOLDER", "TEMP_FOLDER",
   "JS_HEAP_SIZE", "JS_STACK_SIZE", "WRITE_LOCK_POLL_INTERVAL"]

inductive Kind | i64 (dflt lo hi : Int) | str (dflt : String) | bool (dflt : Bool)
  deriving DecidableEq, Repr

/-- `redis_module!` `configurations:` (`lib.rs`): the `graph.*` redis.conf configs.
`VKEY_MAX_ENTITY_COUNT` is `0..i64::MAX` since #3419 (`8621212fc`; was `1..1073741824`). -/
def table : List (String × Kind) :=
  [("CACHE_SIZE", .i64 25 0 1000), ("THREAD_COUNT", .i64 0 0 1024),
   ("INDEX_WORKER_THREADS", .i64 0 0 1024), ("NODE_CREATION_BUFFER", .i64 16384 0 1073741824),
   ("VKEY_MAX_ENTITY_COUNT", .i64 100000 0 9223372036854775807), ("JS_HEAP_SIZE", .i64 268435456 0 4294967296),
   ("JS_STACK_SIZE", .i64 1048576 0 4294967296),
   ("IMPORT_FOLDER", .str "/var/lib/FalkorDB/import/"), ("TEMP_FOLDER", .str "/tmp"),
   ("CMD_INFO", .bool true), ("DELAY_INDEXING", .bool false),
   ("INDEX_BUILD_THREAD_COUNT", .i64 1 1 1024)]   -- #3585, `IMMUTABLE`

/-- Table defaults lie inside their declared bounds. -/
def defaultOk : Kind → Bool
  | .i64 d lo hi => decide (lo ≤ d ∧ d ≤ hi)
  | _ => true

theorem table_defaults_in_range : table.all (fun e => defaultOk e.2) = true := by decide

/-! ## `graph_init` -/

/-- `graph_init`'s argument step (`module_init.rs`): any `apply_module_args`
error is logged and refuses the load; otherwise initialisation continues. -/
def graphInitArgs {ε α : Type} (applyArgs : Except ε α) : Bool :=
  match applyArgs with | .ok _ => true | .error _ => false

theorem graphInitArgs_refuses {ε α : Type} (e : ε) : graphInitArgs (.error e : Except ε α) = false := rfl
theorem graphInitArgs_ok {ε α : Type} (a : α) : graphInitArgs (.ok a : Except ε α) = true := rfl

/-! ## Event handlers -/

/-- `on_role_change`: sticky consumer latch, promotion work. Since #3844 it no
longer sets a replica flag (telemetry does not ask about the role). -/
def onRoleChange (subevent : Nat) : Bool × Bool :=
  (true, subevent == 0)

theorem onRoleChange_promote : onRoleChange 0 = (true, true) := rfl
theorem onRoleChange_demote : onRoleChange 1 = (true, false) := rfl

/-- `on_replica_change`: always latches. -/
def onReplicaChange (_subevent : Nat) : Bool := true

theorem onReplicaChange_latches (s : Nat) : onReplicaChange s = true := rfl

/-- `on_loading`: virtual-key cleanup after `ENDED` (3) or `FAILED` (4). -/
def onLoading (subevent : Nat) : Bool := subevent == 3 || subevent == 4
theorem onLoading_only_end (s : Nat) : onLoading s = true ↔ s = 3 ∨ s = 4 := by
  simp [onLoading]

/-- `on_flush` (, #3844): the new `FLUSH_IN_PROGRESS`, raised by
`REDISMODULE_SUBEVENT_FLUSHDB_START` (0) and lowered by `..._END` (1)
(see `Telemetry.queue_flushing`). -/
def onFlush (subevent : Nat) : Bool := subevent == 0
theorem onFlush_start : onFlush 0 = true := rfl
theorem onFlush_end : onFlush 1 = false := rfl

/-- `on_shutdown`: the teardown order. With the native index (`nativeIndex`) the index
builder is stopped before the pool, and GraphBLAS and RediSearch are torn down only if every
builder worker exited (`builderExited`, what `online_build::stop` returns): one left behind may
still read or free a graph version. -/
def onShutdown (nativeIndex builderExited : Bool) : List String :=
  ["telemetry::shutdown_flusher_thread"] ++
  (if nativeIndex then ["online_build::stop"] else []) ++
  ["threadpool::shutdown"] ++
  (if !nativeIndex || builderExited then ["matrix::shutdown", "RediSearch_CleanupModule"] else [])
theorem onShutdown_flusher_first (n b : Bool) :
    (onShutdown n b).head? = some "telemetry::shutdown_flusher_thread" := by
  cases n <;> cases b <;> rfl
/-- GraphBLAS is torn down only once no builder worker is left: never under one still running. -/
theorem onShutdown_no_teardown_under_a_left_worker :
    "matrix::shutdown" ∉ onShutdown true false := by decide
/-- Without the native index, the order is as it was: no builder to stop or wait for. -/
theorem onShutdown_without_native_index (b : Bool) :
    onShutdown false b = ["telemetry::shutdown_flusher_thread", "threadpool::shutdown",
      "matrix::shutdown", "RediSearch_CleanupModule"] := by
  cases b <;> rfl

/-- `on_fork_child`: abort iff this is the main-thread (BGSAVE) fork and some
graph is not fully synced. -/
def forkChildAborts (mainThread : Bool) (synced : List Bool) : Bool :=
  mainThread && synced.any (! ·)
theorem fork_gc_never_aborts (s : List Bool) : forkChildAborts false s = false := rfl
theorem fork_all_synced (s : List Bool) (h : ∀ b ∈ s, b = true) : forkChildAborts true s = false := by
  simp only [forkChildAborts, Bool.true_and]
  rw [List.any_eq_false]; intro b hb; simp [h b hb]

/-- `enforce_pending_constraints_after_promotion`: per registered graph, the
constraints still under construction; graphs with none are skipped. -/
def pendingPerGraph {G C} (underConstruction : C → Bool) (graphs : List (G × List C)) : List (G × List C) :=
  (graphs.map fun (g, cs) => (g, cs.filter underConstruction)).filter (fun p => !p.2.isEmpty)

theorem pending_only_uc {G C} (u : C → Bool) (gs : List (G × List C)) :
    ∀ p ∈ pendingPerGraph u gs, p.2 ≠ [] ∧ ∀ c ∈ p.2, u c := by
  intro p hp
  unfold pendingPerGraph at hp
  rw [List.mem_filter] at hp
  obtain ⟨hm, hne⟩ := hp
  rw [List.mem_map] at hm
  obtain ⟨⟨g, cs⟩, _, rfl⟩ := hm
  refine ⟨fun e => ?_, fun c hc => (List.mem_filter.1 hc).2⟩
  simp only at e
  obtain ⟨x, hx, hu⟩ := (by simpa using hne : ∃ x, x ∈ cs ∧ u x = true)
  have : x ∈ cs.filter u := List.mem_filter.2 ⟨hx, hu⟩
  rw [e] at this; simp at this

/-! ## `on_keyspace_event`: two-phase RENAME -/

structure KS where
  oldName : Option String
  registry : List (String × Nat)
  deleted : List String     -- telemetry streams `DEL`ed

def renameReg (reg : List (String × Nat)) (old new : String) : List (String × Nat) :=
  match reg.find? (·.1 = old) with
  | none => reg
  | some (_, g) => (new, g) :: (reg.filter (fun p => p.1 ≠ old && p.1 ≠ new))

def onKeyspace (s : KS) (event key : String) : KS :=
  if event = "rename_from" then { s with oldName := some key }
  else if event = "rename_to" then
    match s.oldName with
    | some old => ⟨none, renameReg s.registry old key, s.deleted ++ [old]⟩
    | none => s
  else s

/-- `rename_to` without a preceding `rename_from` changes nothing. -/
theorem rename_to_alone (s : KS) (k : String) (h : s.oldName = none) : onKeyspace s "rename_to" k = s := by
  simp [onKeyspace, h]

/-- A full RENAME re-keys the graph and deletes the old key's stream. -/
theorem rename_pair (s : KS) (a b : String) (g : Nat) (h : s.registry.find? (·.1 = a) = some (a, g)) :
    let s' := onKeyspace (onKeyspace s "rename_from" a) "rename_to" b
    (s'.registry.find? (·.1 = b) = some (b, g)) ∧ s'.deleted = s.deleted ++ [a] ∧ s'.oldName = none := by
  simp [onKeyspace, renameReg, h]

/-- Other events are ignored. -/
theorem other_event (s : KS) (e k : String) (h1 : e ≠ "rename_from") (h2 : e ≠ "rename_to") :
    onKeyspace s e k = s := by simp [onKeyspace, h1, h2]

/-- `info_func`: nothing unless for a crash report. -/
def infoSections (forCrash : Bool) (running : List String) : List String :=
  if forCrash then "executing commands" :: running else []
theorem info_normal (r : List String) : infoSections false r = [] := rfl

/-! ## `src/config.rs` helpers -/

/-- `u64::next_power_of_two`, searched upward from `p` (fuel-bounded). -/
def npow : Nat → Nat → Nat → Nat
  | 0, p, _ => p
  | k+1, p, n => if n ≤ p then p else npow k (2 * p) n

/-- `normalize_node_creation_buffer`: `max(val, 128).next_power_of_two()`. -/
def normalizeNCB (v : Int) : Nat := npow 64 1 (max v 128).toNat

theorem npow_ge (k p n : Nat) (hp : 1 ≤ p) (hk : n ≤ p * 2^k) : n ≤ npow k p n := by
  induction k generalizing p with
  | zero => simp [npow] at *; omega
  | succ k ih =>
    unfold npow; split
    · assumption
    · apply ih (2 * p) (by omega)
      have e : p * 2^(k+1) = 2 * p * 2^k := by
        rw [Nat.pow_succ, Nat.mul_comm (2^k) 2, ← Nat.mul_assoc, Nat.mul_comm p 2]
      rw [← e]; exact hk

theorem normalizeNCB_ge (v : Int) (h : v ≤ 2^63) : 128 ≤ normalizeNCB v := by
  have := npow_ge 64 1 ((max v 128).toNat) (by decide) (by omega)
  unfold normalizeNCB; omega

theorem normalizeNCB_examples :
    normalizeNCB 0 = 128 ∧ normalizeNCB 129 = 256 ∧ normalizeNCB 16384 = 16384 := by decide

/-- `get_thread_count`: the configured value, or the machine's parallelism (4 if unknown). -/
def threadCount (cfg : Int) (par : Option Nat) : Int := if cfg > 0 then cfg else (par.getD 4 : Int)
theorem threadCount_pos (c : Int) (p : Option Nat) (hp : ∀ n, p = some n → 0 < n) : 0 < threadCount c p := by
  unfold threadCount; split; · omega
  cases p with
  | none => simp
  | some n => have := hp n rfl; simp; omega

end RedisLayer.ModuleInit
