/-!
# Module load: configuration table and event handlers (`src/module_init.rs`, `src/lib.rs`)

Since #3328 (`ded0b7c3f`) load-time configuration has one source: `graph_init`
(`module_init.rs:139`) collects the arguments (`to_string_lossy`) and calls
`config::apply_module_args`, refusing the load (`Status::Err`) on any error
(`:163-168`); `lib.rs` sets `module_args_as_configuration: false`, so redismodule-rs no
longer reads them. That function is modelled and proven equal to C's `Config_Init` in
`LoadArgs.lean` (`rust_c_same_accept`, `error_applies_nothing`). `graph_init`'s own
work here is "args → `apply_module_args` → `Err` ⇒ `Status::Err`" (`graphInitArgs`).

`table` (the `redis_module!` `configurations:` list, `lib.rs:97-116`) still registers the
`graph.*` redis.conf configs; its defaults are checked below.

**Historical (before #3328, 4a4f2a72f) — W4-redis-3 / #3268, fixed by #3328 (`ded0b7c3f`).**
The table was also fed the module arguments (`module_args_as_configuration: true`,
redismodule-rs `get_*_default_config_value`: the value after the first argument
*byte-equal* to the name; a bool is `value == "yes"`), and `graph_init` had its own
upper-casing loop for the atomics (`pre3328_argLoop`). Confirmed live then:
`CMD_INFO YES` → Rust 0, C 1 (`pre3328_bool_yes_case`); `CMD_INFO garbage` loaded with 0,
C refuses (`pre3328_bool_garbage_off`); `cache_size 10` ignored (`pre3328_table_name_case`);
`ASYNC_DELETE yes` ignored (`pre3328_load_missing`). All four now load as in C
(`LoadArgs.fixed3328_counterexamples`, re-checked live on b080a44c1).
-/
namespace RedisLayer.ModuleInit

/-- C's configuration names (`config.c`). -/
def documented : List String :=
  ["TIMEOUT", "TIMEOUT_DEFAULT", "TIMEOUT_MAX", "CACHE_SIZE", "ASYNC_DELETE", "THREAD_COUNT",
   "RESULTSET_SIZE", "OMP_THREAD_COUNT", "VKEY_MAX_ENTITY_COUNT", "MAX_QUEUED_QUERIES",
   "QUERY_MEM_CAPACITY", "DELTA_MAX_PENDING_CHANGES", "NODE_CREATION_BUFFER", "CMD_INFO",
   "MAX_INFO_QUERIES", "EFFECTS_THRESHOLD", "DELAY_INDEXING", "IMPORT_FOLDER", "TEMP_FOLDER",
   "JS_HEAP_SIZE", "JS_STACK_SIZE"]

inductive Kind | i64 (dflt lo hi : Int) | str (dflt : String) | bool (dflt : Bool)
  deriving DecidableEq, Repr

/-- `redis_module!` `configurations:` (`lib.rs:97-116`): the `graph.*` redis.conf configs. -/
def table : List (String × Kind) :=
  [("CACHE_SIZE", .i64 25 0 1000), ("THREAD_COUNT", .i64 0 0 1024),
   ("INDEX_WORKER_THREADS", .i64 0 0 1024), ("NODE_CREATION_BUFFER", .i64 16384 0 1073741824),
   ("VKEY_MAX_ENTITY_COUNT", .i64 100000 1 1073741824), ("JS_HEAP_SIZE", .i64 268435456 0 4294967296),
   ("JS_STACK_SIZE", .i64 1048576 0 4294967296),
   ("IMPORT_FOLDER", .str "/var/lib/FalkorDB/import/"), ("TEMP_FOLDER", .str "/tmp"),
   ("CMD_INFO", .bool true), ("DELAY_INDEXING", .bool false)]

/-- Historical: names `graph_init`'s own loop understood (old `:160-221`). -/
def pre3328_initNames : List String :=
  ["TIMEOUT", "TIMEOUT_DEFAULT", "TIMEOUT_MAX", "RESULTSET_SIZE", "QUERY_MEM_CAPACITY",
   "DELTA_MAX_PENDING_CHANGES", "EFFECTS_COMPRESSION", "EFFECTS_THRESHOLD", "OMP_THREAD_COUNT",
   "MAX_INFO_QUERIES", "MAX_QUEUED_QUERIES"]

def pre3328_loadNames : List String := table.map (·.1) ++ pre3328_initNames

/-- **Registration table vs documented list**: every documented name is settable at load
except `ASYNC_DELETE`; Rust adds `INDEX_WORKER_THREADS` and `EFFECTS_COMPRESSION`. -/
theorem pre3328_load_missing : documented.filter (· ∉ pre3328_loadNames) = ["ASYNC_DELETE"] := by decide
theorem pre3328_load_extra : pre3328_loadNames.filter (· ∉ documented) = ["INDEX_WORKER_THREADS", "EFFECTS_COMPRESSION"] := by
  decide
/-- The two sources do not overlap, so no name is parsed twice with different rules. -/
theorem pre3328_table_init_disjoint : (table.map (·.1)).filter (· ∈ pre3328_initNames) = [] := by decide

/-- Table defaults lie inside their declared bounds. -/
def defaultOk : Kind → Bool
  | .i64 d lo hi => decide (lo ≤ d ∧ d ≤ hi)
  | _ => true

theorem table_defaults_in_range : table.all (fun e => defaultOk e.2) = true := by decide

/-! ## Historical: redismodule-rs `find_config_value` / `get_*_default_config_value` (before #3328) -/

/-- `args.iter().skip_while(|a| a != name).nth(1)`: the argument after the first one equal
to `name` (anywhere — also in a value position). -/
def pre3328_findValue (args : List String) (name : String) : Option String :=
  match args.dropWhile (· ≠ name) with
  | _ :: v :: _ => some v
  | _ => none

def pre3328_boolDefault (args : List String) (name : String) (dflt : Bool) : Bool :=
  match pre3328_findValue args name with
  | some v => v == "yes"
  | none => dflt

/-- C (`_Config_ParseYesNo`): case-insensitive yes/no, anything else an error. -/
def cBool (v : String) : Option Bool :=
  let l := v.toList.map Char.toLower
  if l == "yes".toList then some true else if l == "no".toList then some false else none

theorem pre3328_bool_yes_case : pre3328_boolDefault ["CMD_INFO", "YES"] "CMD_INFO" true = false ∧
    cBool "YES" = some true := by decide
theorem pre3328_bool_garbage_off : pre3328_boolDefault ["CMD_INFO", "garbage"] "CMD_INFO" true = false ∧
    cBool "garbage" = none := by decide
/-- Names in the table are matched byte-for-byte: a lower-case name is not seen. -/
theorem pre3328_table_name_case : pre3328_findValue ["cache_size", "10"] "CACHE_SIZE" = none := by decide
/-- A name occurring as another option's *value* is taken as a key. -/
theorem pre3328_value_position : pre3328_findValue ["IMPORT_FOLDER", "CACHE_SIZE", "7"] "CACHE_SIZE" = some "7" := by
  decide

/-! ## Historical: `graph_init`'s argument loop (old `:154-223`, removed by #3328) -/

def MAX_INFO_QUERIES_CAP : Int := 1000

inductive Tgt | i64 (n : String) | maxInfo | maxQueued | none
  deriving DecidableEq

def target (upper : String) : Tgt :=
  if upper ∈ ["TIMEOUT", "TIMEOUT_DEFAULT", "TIMEOUT_MAX", "RESULTSET_SIZE", "QUERY_MEM_CAPACITY",
      "DELTA_MAX_PENDING_CHANGES", "EFFECTS_COMPRESSION", "EFFECTS_THRESHOLD", "OMP_THREAD_COUNT"]
  then .i64 upper
  else if upper = "MAX_INFO_QUERIES" then .maxInfo
  else if upper = "MAX_QUEUED_QUERIES" then .maxQueued
  else .none

/-- The loop, with `parse` = `str::parse::<i64>` (abstract). Returns the stores made, or
`none` for `Status::Err`. Unknown names advance by one. -/
def pre3328_argLoop (parse : String → Option Int) : List String → Option (List (String × Int))
  | [] => some []
  | a :: rest =>
    match target a.toUpper with
    | .none => pre3328_argLoop parse rest
    | t =>
      match rest with
      | v :: rest' =>
        match parse v, t with
        | some x, .i64 n => (pre3328_argLoop parse rest').map ((n, x) :: ·)
        | some x, .maxInfo => if 0 ≤ x then (pre3328_argLoop parse rest').map (("MAX_INFO_QUERIES", min x MAX_INFO_QUERIES_CAP) :: ·) else none
        | some x, .maxQueued => if 0 ≤ x then (pre3328_argLoop parse rest').map (("MAX_QUEUED_QUERIES", x) :: ·) else none
        | _, _ => none
      | [] => none

theorem pre3328_target_i64 (u n : String) (h : target u = .i64 n) : n ≠ "MAX_INFO_QUERIES" := by
  unfold target at h
  split at h
  · rename_i hm; cases h; intro e; subst e; revert hm; decide
  · split at h <;> (try split at h) <;> cases h

/-- `MAX_INFO_QUERIES` is clamped, never rejected for being large. -/
theorem pre3328_maxInfo_clamped (parse : String → Option Int) (args : List String) (st : List (String × Int))
    (h : pre3328_argLoop parse args = some st) : ∀ p ∈ st, p.1 = "MAX_INFO_QUERIES" → p.2 ≤ MAX_INFO_QUERIES_CAP := by
  fun_induction pre3328_argLoop parse args generalizing st <;> simp_all
  all_goals first
    | assumption
    | (rename_i ih
       intro a b hab hn
       obtain ⟨l, hl, rfl⟩ := h
       simp at hab
       rcases hab with ⟨rfl, rfl⟩ | hab
       · first
          | exact Int.min_le_right _ _
          | (exact absurd hn (pre3328_target_i64 _ _ (by assumption)))
          | (simp at hn; done)
       · exact ih _ hl _ _ hab hn)

/-- Historical: the cross-checks after parsing (old `:369-383`); now inside `apply_module_args`. -/
def pre3328_timeoutsBad (timeout dflt max : Int) : Bool :=
  (timeout > 0 && (dflt > 0 || max > 0)) || (dflt > 0 && max > 0 && dflt > max)

theorem pre3328_timeouts_ok_inv (t d m : Int) (h : pre3328_timeoutsBad t d m = false) (hd : d > 0) (hm : m > 0) :
    d ≤ m := by
  simp [pre3328_timeoutsBad] at h; omega

/-! ## `graph_init` since #3328 -/

/-- `graph_init`'s argument step (`module_init.rs:159-168`): any `apply_module_args`
error is logged and refuses the load; otherwise initialisation continues. -/
def graphInitArgs {ε α : Type} (applyArgs : Except ε α) : Bool :=
  match applyArgs with | .ok _ => true | .error _ => false

theorem graphInitArgs_refuses {ε α : Type} (e : ε) : graphInitArgs (.error e : Except ε α) = false := rfl
theorem graphInitArgs_ok {ε α : Type} (a : α) : graphInitArgs (.ok a : Except ε α) = true := rfl

/-! ## Event handlers -/

/-- `on_role_change` (`:466`): replica flag, sticky consumer latch, promotion work. -/
def onRoleChange (subevent : Nat) : Bool × Bool × Bool :=
  (subevent == 1, true, subevent == 0)

theorem onRoleChange_promote : onRoleChange 0 = (false, true, true) := rfl
theorem onRoleChange_demote : onRoleChange 1 = (true, true, false) := rfl

/-- `on_replica_change` (`:564`): always latches. -/
def onReplicaChange (_subevent : Nat) : Bool := true

theorem onReplicaChange_latches (s : Nat) : onReplicaChange s = true := rfl

/-- `on_loading` (`:576`): virtual-key cleanup after `ENDED` (3) or `FAILED` (4). -/
def onLoading (subevent : Nat) : Bool := subevent == 3 || subevent == 4
theorem onLoading_only_end (s : Nat) : onLoading s = true ↔ s = 3 ∨ s = 4 := by
  simp [onLoading]

/-- `on_flush` (`:438`): nothing. -/
def onFlush {σ} (s : σ) : σ := s
theorem onFlush_id {σ} (s : σ) : onFlush s = s := rfl

/-- `on_shutdown` (`:451`): the teardown order. -/
def onShutdown : List String := ["telemetry::shutdown_flusher_thread", "threadpool::shutdown",
  "matrix::shutdown", "RediSearch_CleanupModule"]
theorem onShutdown_flusher_first : onShutdown.head? = some "telemetry::shutdown_flusher_thread" := rfl

/-- `on_fork_child` (`:118`): abort iff this is the main-thread (BGSAVE) fork and some
graph is not fully synced. -/
def forkChildAborts (mainThread : Bool) (synced : List Bool) : Bool :=
  mainThread && synced.any (! ·)
theorem fork_gc_never_aborts (s : List Bool) : forkChildAborts false s = false := rfl
theorem fork_all_synced (s : List Bool) (h : ∀ b ∈ s, b = true) : forkChildAborts true s = false := by
  simp only [forkChildAborts, Bool.true_and]
  rw [List.any_eq_false]; intro b hb; simp [h b hb]

/-- `enforce_pending_constraints_after_promotion` (`:496`): per registered graph, the
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

/-! ## `on_keyspace_event` (`:597`): two-phase RENAME -/

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
