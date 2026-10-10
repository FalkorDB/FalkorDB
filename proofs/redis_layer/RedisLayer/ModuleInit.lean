/-!
# Module load: configuration table and event handlers (`src/module_init.rs`, `src/lib.rs`)

Load-time configuration comes from two places:
* the `redis_module!` `configurations:` table (`src/lib.rs:97-116`) with
  `module_args_as_configuration: true`, i.e. redismodule-rs `get_*_default_config_value`
  (`configuration.rs:262-392`, git `91a39b5`): the value is the argument right after the
  first argument *byte-equal* to the name; a bool is `value == "yes"`;
* `graph_init`'s own loop (`module_init.rs:141-223`) for the atomics, upper-casing names.

`documented` is C's list (`src/configuration/config.c` on `master`).

Findings (confirmed live on `loadmodule … <args>`, Rust vs C):
* `CMD_INFO YES` → Rust `CMD_INFO 0`, C `1` (`bool_yes_case`);
  `CMD_INFO garbage` → Rust loads with `0`, C refuses to load (`bool_garbage_off`).
* `cache_size 10` → Rust ignores it (`25`), C applies it (`10`) (`table_name_case`).
* `ASYNC_DELETE yes` → Rust ignores it at load (`load_missing`); since #3021 its default is C's `1`.
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

/-- `redis_module!` `configurations:` (`lib.rs:97-116`). -/
def table : List (String × Kind) :=
  [("CACHE_SIZE", .i64 25 0 1000), ("THREAD_COUNT", .i64 0 0 1024),
   ("INDEX_WORKER_THREADS", .i64 0 0 1024), ("NODE_CREATION_BUFFER", .i64 16384 0 1073741824),
   ("VKEY_MAX_ENTITY_COUNT", .i64 100000 1 1073741824), ("JS_HEAP_SIZE", .i64 268435456 0 4294967296),
   ("JS_STACK_SIZE", .i64 1048576 0 4294967296),
   ("IMPORT_FOLDER", .str "/var/lib/FalkorDB/import/"), ("TEMP_FOLDER", .str "/tmp"),
   ("CMD_INFO", .bool true), ("DELAY_INDEXING", .bool false)]

/-- Names `graph_init`'s loop understands (`:160-221`). -/
def initNames : List String :=
  ["TIMEOUT", "TIMEOUT_DEFAULT", "TIMEOUT_MAX", "RESULTSET_SIZE", "QUERY_MEM_CAPACITY",
   "DELTA_MAX_PENDING_CHANGES", "EFFECTS_COMPRESSION", "EFFECTS_THRESHOLD", "OMP_THREAD_COUNT",
   "MAX_INFO_QUERIES", "MAX_QUEUED_QUERIES"]

def loadNames : List String := table.map (·.1) ++ initNames

/-- **Registration table vs documented list**: every documented name is settable at load
except `ASYNC_DELETE`; Rust adds `INDEX_WORKER_THREADS` and `EFFECTS_COMPRESSION`. -/
theorem load_missing : documented.filter (· ∉ loadNames) = ["ASYNC_DELETE"] := by decide
theorem load_extra : loadNames.filter (· ∉ documented) = ["INDEX_WORKER_THREADS", "EFFECTS_COMPRESSION"] := by
  decide
/-- The two sources do not overlap, so no name is parsed twice with different rules. -/
theorem table_init_disjoint : (table.map (·.1)).filter (· ∈ initNames) = [] := by decide

/-- Table defaults lie inside their declared bounds. -/
def defaultOk : Kind → Bool
  | .i64 d lo hi => decide (lo ≤ d ∧ d ≤ hi)
  | _ => true

theorem table_defaults_in_range : table.all (fun e => defaultOk e.2) = true := by decide

/-! ## redismodule-rs `find_config_value` / `get_*_default_config_value` -/

/-- `args.iter().skip_while(|a| a != name).nth(1)`: the argument after the first one equal
to `name` (anywhere — also in a value position). -/
def findValue (args : List String) (name : String) : Option String :=
  match args.dropWhile (· ≠ name) with
  | _ :: v :: _ => some v
  | _ => none

def boolDefault (args : List String) (name : String) (dflt : Bool) : Bool :=
  match findValue args name with
  | some v => v == "yes"
  | none => dflt

/-- C (`_Config_ParseYesNo`): case-insensitive yes/no, anything else an error. -/
def cBool (v : String) : Option Bool :=
  let l := v.toList.map Char.toLower
  if l == "yes".toList then some true else if l == "no".toList then some false else none

theorem bool_yes_case : boolDefault ["CMD_INFO", "YES"] "CMD_INFO" true = false ∧
    cBool "YES" = some true := by decide
theorem bool_garbage_off : boolDefault ["CMD_INFO", "garbage"] "CMD_INFO" true = false ∧
    cBool "garbage" = none := by decide
/-- Names in the table are matched byte-for-byte: a lower-case name is not seen. -/
theorem table_name_case : findValue ["cache_size", "10"] "CACHE_SIZE" = none := by decide
/-- A name occurring as another option's *value* is taken as a key. -/
theorem value_position : findValue ["IMPORT_FOLDER", "CACHE_SIZE", "7"] "CACHE_SIZE" = some "7" := by
  decide

/-! ## `graph_init`'s argument loop (`:154-223`) -/

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
def argLoop (parse : String → Option Int) : List String → Option (List (String × Int))
  | [] => some []
  | a :: rest =>
    match target a.toUpper with
    | .none => argLoop parse rest
    | t =>
      match rest with
      | v :: rest' =>
        match parse v, t with
        | some x, .i64 n => (argLoop parse rest').map ((n, x) :: ·)
        | some x, .maxInfo => if 0 ≤ x then (argLoop parse rest').map (("MAX_INFO_QUERIES", min x MAX_INFO_QUERIES_CAP) :: ·) else none
        | some x, .maxQueued => if 0 ≤ x then (argLoop parse rest').map (("MAX_QUEUED_QUERIES", x) :: ·) else none
        | _, _ => none
      | [] => none

theorem target_i64 (u n : String) (h : target u = .i64 n) : n ≠ "MAX_INFO_QUERIES" := by
  unfold target at h
  split at h
  · rename_i hm; cases h; intro e; subst e; revert hm; decide
  · split at h <;> (try split at h) <;> cases h

/-- `MAX_INFO_QUERIES` is clamped, never rejected for being large. -/
theorem maxInfo_clamped (parse : String → Option Int) (args : List String) (st : List (String × Int))
    (h : argLoop parse args = some st) : ∀ p ∈ st, p.1 = "MAX_INFO_QUERIES" → p.2 ≤ MAX_INFO_QUERIES_CAP := by
  fun_induction argLoop parse args generalizing st <;> simp_all
  all_goals first
    | assumption
    | (rename_i ih
       intro a b hab hn
       obtain ⟨l, hl, rfl⟩ := h
       simp at hab
       rcases hab with ⟨rfl, rfl⟩ | hab
       · first
          | exact Int.min_le_right _ _
          | (exact absurd hn (target_i64 _ _ (by assumption)))
          | (simp at hn; done)
       · exact ih _ hl _ _ hab hn)

/-- The cross-checks after parsing (`:369-383`): load fails iff these hold. -/
def timeoutsBad (timeout dflt max : Int) : Bool :=
  (timeout > 0 && (dflt > 0 || max > 0)) || (dflt > 0 && max > 0 && dflt > max)

theorem timeouts_ok_inv (t d m : Int) (h : timeoutsBad t d m = false) (hd : d > 0) (hm : m > 0) :
    d ≤ m := by
  simp [timeoutsBad] at h; omega

/-! ## Event handlers -/

/-- `on_role_change` (`:539`): replica flag, sticky consumer latch, promotion work. -/
def onRoleChange (subevent : Nat) : Bool × Bool × Bool :=
  (subevent == 1, true, subevent == 0)

theorem onRoleChange_promote : onRoleChange 0 = (false, true, true) := rfl
theorem onRoleChange_demote : onRoleChange 1 = (true, true, false) := rfl

/-- `on_replica_change` (`:637`): always latches. -/
def onReplicaChange (_subevent : Nat) : Bool := true

theorem onReplicaChange_latches (s : Nat) : onReplicaChange s = true := rfl

/-- `on_loading` (`:649`): virtual-key cleanup after `ENDED` (3) or `FAILED` (4). -/
def onLoading (subevent : Nat) : Bool := subevent == 3 || subevent == 4
theorem onLoading_only_end (s : Nat) : onLoading s = true ↔ s = 3 ∨ s = 4 := by
  simp [onLoading]

/-- `on_flush` (`:511`): nothing. -/
def onFlush {σ} (s : σ) : σ := s
theorem onFlush_id {σ} (s : σ) : onFlush s = s := rfl

/-- `on_shutdown` (`:524`): the teardown order. -/
def onShutdown : List String := ["telemetry::shutdown_flusher_thread", "threadpool::shutdown",
  "matrix::shutdown", "RediSearch_CleanupModule"]
theorem onShutdown_flusher_first : onShutdown.head? = some "telemetry::shutdown_flusher_thread" := rfl

/-- `on_fork_child` (`:120`): abort iff this is the main-thread (BGSAVE) fork and some
graph is not fully synced. -/
def forkChildAborts (mainThread : Bool) (synced : List Bool) : Bool :=
  mainThread && synced.any (! ·)
theorem fork_gc_never_aborts (s : List Bool) : forkChildAborts false s = false := rfl
theorem fork_all_synced (s : List Bool) (h : ∀ b ∈ s, b = true) : forkChildAborts true s = false := by
  simp only [forkChildAborts, Bool.true_and]
  rw [List.any_eq_false]; intro b hb; simp [h b hb]

/-- `enforce_pending_constraints_after_promotion` (`:569`): per registered graph, the
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

/-! ## `on_keyspace_event` (`:670`): two-phase RENAME -/

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
