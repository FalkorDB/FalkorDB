import RedisLayer.Config
/-!
# Module load arguments (`src/config.rs`, #3328 `ded0b7c3f`)

`graph_init` (`src/module_init.rs:139`) hands every `--loadmodule … NAME VALUE …`
argument (`to_string_lossy`) to `apply_module_args` (`src/config.rs:296-354`); `lib.rs`
sets `module_args_as_configuration: false`, so redismodule-rs no longer reads them.

| here | there (`origin/main` b080a44c1) |
| --- | --- |
| `N`, `N.str`, `parseName`        | the string `match`es on the upper-cased name (`config.rs:236-247`, `config_cmd.rs:118-198`) |
| `validateTempFolder`             | `validate_temp_folder` `config.rs:205` |
| `validateLoadOnly`               | `validate_load_only` `config.rs:223` |
| `validateConfigSet`              | `validate_config_set` `config_cmd.rs:114` — each arm *is* `Config.validate` (#3021 model) |
| `checkTimeouts`                  | `check_timeout_constraints` `config_cmd.rs:220` |
| `rstep`, `loop`, `applyModuleArgs` | `apply_module_args` `config.rs:296` (loop body `:316-347`, final check `:349-355`, apply `:357-359`) |
| `LoadArg`, `applyLoadArg`        | `enum LoadArg` `config.rs:185`, `apply_load_arg` `:252` |
| `cParseInteger`, `cYesNo`, `cSet`, `cLookup`, `cInit` | C `_Config_ParseInteger` (`strtoll`), `_Config_ParseYesNo`, `Config_Option_set`, `Config_Contains_field`, `Config_Init` (`src/configuration/config.c` on `master`) |

`FS` is the file system as both engines see it (libc `access`/`stat`, Rust
`Path::exists`/`is_dir`/`File::create`): FFI, a hypothesis structure. `ConfigValue::Uint`
only arises for `MAX_QUEUED_QUERIES`, whose value is positive, so `as_i64` is exact and
values are `Int`s here.

Proven:
* `rust_c_same_accept` — for every argument vector whose values have no leading
  whitespace and that names no Rust-only config, `apply_module_args` succeeds exactly
  when C's `Config_Init` does (case-insensitive names, even arity, known names, C's value
  rules, the TIMEOUT/TIMEOUT_DEFAULT/TIMEOUT_MAX interplay), given only that
  `File::create` in a directory succeeds iff `access(W_OK)` does.
* `error_applies_nothing`, `ok_applies_each_pair` — every pair is validated before any is
  applied: an error applies nothing; success applies one `apply_load_arg` per pair, in order.
* `fixed3328_*` — the W4-redis-3 / #3268 counterexamples now load as in C;
  `ModuleInit.pre3328_*` keep the old behaviour.

Divergences (both confirmed live, ports 19470, Rust b080a44c1 vs C `master`):
* `leading_space_divergence` — C's `strtoll` skips leading whitespace, Rust's
  `str::parse::<i64>` does not: `--loadmodule … CACHE_SIZE " 10"` loads on C (`CACHE_SIZE 10`)
  and is refused by Rust; same for `GRAPH.CONFIG SET RESULTSET_SIZE " 5"` (C OK, Rust
  "Failed to set config value"). Trailing whitespace is refused by both.
* `rust_only_names` — `INDEX_WORKER_THREADS`, `BOLT_PORT`, `EFFECTS_COMPRESSION` load on
  Rust; C refuses the load ("unknown configuration field"). Intended Rust extensions.
-/
namespace RedisLayer.LoadArgs
open RedisLayer

/-! ## Names -/

inductive N where
  | threadCount | timeout | timeoutDefault | timeoutMax | ompThreadCount | vkeyMax | cacheSize
  | resultsetSize | maxQueued | queryMemCapacity | deltaMax | nodeCreationBuffer | asyncDelete
  | cmdInfo | maxInfo | effectsThreshold | delayIndexing | importFolder | tempFolder
  | jsHeap | jsStack
  -- Rust only
  | indexWorkerThreads | boltPort | effectsCompression
  deriving DecidableEq, Repr

def N.str : N → List Nat
  | .threadCount => b "THREAD_COUNT" | .timeout => b "TIMEOUT"
  | .timeoutDefault => b "TIMEOUT_DEFAULT" | .timeoutMax => b "TIMEOUT_MAX"
  | .ompThreadCount => b "OMP_THREAD_COUNT" | .vkeyMax => b "VKEY_MAX_ENTITY_COUNT"
  | .cacheSize => b "CACHE_SIZE" | .resultsetSize => b "RESULTSET_SIZE"
  | .maxQueued => b "MAX_QUEUED_QUERIES" | .queryMemCapacity => b "QUERY_MEM_CAPACITY"
  | .deltaMax => b "DELTA_MAX_PENDING_CHANGES" | .nodeCreationBuffer => b "NODE_CREATION_BUFFER"
  | .asyncDelete => b "ASYNC_DELETE" | .cmdInfo => b "CMD_INFO" | .maxInfo => b "MAX_INFO_QUERIES"
  | .effectsThreshold => b "EFFECTS_THRESHOLD" | .delayIndexing => b "DELAY_INDEXING"
  | .importFolder => b "IMPORT_FOLDER" | .tempFolder => b "TEMP_FOLDER"
  | .jsHeap => b "JS_HEAP_SIZE" | .jsStack => b "JS_STACK_SIZE"
  | .indexWorkerThreads => b "INDEX_WORKER_THREADS" | .boltPort => b "BOLT_PORT"
  | .effectsCompression => b "EFFECTS_COMPRESSION"

/-- C's `Config_Contains_field` chain, in its order. -/
def cList : List N :=
  [.threadCount, .timeout, .timeoutDefault, .timeoutMax, .ompThreadCount, .vkeyMax, .cacheSize,
   .resultsetSize, .maxQueued, .queryMemCapacity, .deltaMax, .nodeCreationBuffer, .asyncDelete,
   .cmdInfo, .maxInfo, .effectsThreshold, .delayIndexing, .importFolder, .tempFolder, .jsHeap, .jsStack]

def rustOnly : List N := [.indexWorkerThreads, .boltPort, .effectsCompression]

/-- A Rust string `match` over these (distinct) literals. -/
def parseName (s : List Nat) : Option N := (cList ++ rustOnly).find? (fun n => s == n.str)

/-! ## The Rust side -/

structure FS where
  exists_ : List Nat → Bool
  isDir : List Nat → Bool
  /-- C: `access(path, W_OK) == 0` -/
  writable : List Nat → Bool
  /-- Rust: `File::create(path/.falkordb_temp_test)` succeeds -/
  createOk : List Nat → Bool

inductive LErr where
  | odd | unknown | failed | readOnly | jsTooSmall | deprecated | defaultAboveMax
  | maxBelowDefault | tempNoExist | tempNotDir | tempNoWrite | timeoutMix
  deriving DecidableEq, Repr

def ofCfg : CfgErr → LErr
  | .failed => .failed | .readOnly => .readOnly | .deprecated => .deprecated
  | .defaultAboveMax => .defaultAboveMax | .maxBelowDefault => .maxBelowDefault
  | .jsTooSmall => .jsTooSmall

def mapE {α : Type} : Except CfgErr α → Except LErr α
  | .ok a => .ok a
  | .error e => .error (ofCfg e)

inductive LoadArg where
  | cacheSize (v : Int) | threadCount (v : Int) | ompThreadCount (v : Int)
  | nodeCreationBuffer (v : Int) | indexWorkerThreads (v : Int)
  | importFolder (v : List Nat) | tempFolder (v : List Nat) | boltPort
  | runtime (name : N) (value : List Nat) (parsed : Int)
  deriving DecidableEq, Repr

/-- `validate_temp_folder` (`config.rs:205-218`). -/
def validateTempFolder (fs : FS) (p : List Nat) : Except LErr Unit :=
  if !fs.exists_ p then .error .tempNoExist
  else if !fs.isDir p then .error .tempNotDir
  else if !fs.createOk p then .error .tempNoWrite
  else .ok ()

def req (c : Option LoadArg) : Option (Except LErr LoadArg) :=
  some (match c with | some a => .ok a | none => .error .failed)

/-- `validate_load_only` (`config.rs:223-249`). -/
def validateLoadOnly (fs : FS) (n : N) (v : List Nat) : Option (Except LErr LoadArg) :=
  let int := rustParseI64 v
  let positive := int.filter (fun x => decide (0 < x))
  let nonNeg := int.filter (fun x => decide (0 ≤ x))
  match n with
  | .cacheSize => req (positive.map .cacheSize)
  | .threadCount => req (positive.map .threadCount)
  | .ompThreadCount => req (positive.map .ompThreadCount)
  | .nodeCreationBuffer => req (nonNeg.map .nodeCreationBuffer)
  | .indexWorkerThreads => req (nonNeg.map .indexWorkerThreads)
  | .importFolder => req (some (.importFolder v))
  | .tempFolder => some (match validateTempFolder fs v with
      | .ok () => .ok (.tempFolder v)
      | .error e => .error e)
  | .boltPort => req (some .boltPort)
  | _ => none

/-- `validate_config_set` (`config_cmd.rs:114-202`); the arms share code in Rust, so each
is the #3021 model `Config.validate` of one member of the arm. `none` = no literal matched. -/
def validateConfigSet (n : Option N) (v : List Nat) : Except LErr Int :=
  match n with
  | some .timeout | some .timeoutDefault | some .timeoutMax | some .queryMemCapacity
  | some .deltaMax | some .effectsCompression | some .effectsThreshold => mapE (validate .timeout v)
  | some .asyncDelete | some .cmdInfo | some .delayIndexing => mapE (validate .cmdInfo v)
  | some .resultsetSize => mapE (validate .resultsetSize v)
  | some .maxQueued => mapE (validate .maxQueued v)
  | some .vkeyMax => mapE (validate .vkeyMax v)
  | some .maxInfo => mapE (validate .maxInfo v)
  | some .jsHeap | some .jsStack => mapE (validate .jsHeap v)
  | some .threadCount | some .indexWorkerThreads | some .ompThreadCount | some .cacheSize
  | some .nodeCreationBuffer | some .boltPort | some .importFolder | some .tempFolder => .error .readOnly
  | none => .error .unknown

/-- `check_timeout_constraints` (`config_cmd.rs:220-253`): fold the batch into the two
values, then the two checks. -/
def checkTimeouts (td tm : Int) (vs : List (N × Int)) : Except LErr (Int × Int) :=
  let td' := vs.foldl (fun a p => if p.1 = .timeoutDefault then p.2 else a) td
  let tm' := vs.foldl (fun a p => if p.1 = .timeoutMax then p.2 else a) tm
  let setting := vs.any (·.1 == .timeout)
  if setting ∧ (td' > 0 ∨ tm' > 0) then .error .deprecated
  else if td' > 0 ∧ tm' > 0 ∧ td' > tm' then
    .error (if vs.any (·.1 == .timeoutDefault) then .defaultAboveMax else .maxBelowDefault)
  else .ok (td', tm')

/-- One iteration of `apply_module_args`' pair loop (`config.rs:316-347`). -/
def rstep (fs : FS) (td tm : Int) (n : N) (v : List Nat) : Except LErr (LoadArg × Int × Int) :=
  match validateLoadOnly fs n v with
  | some (.error e) => .error e
  | some (.ok a) => .ok (a, td, tm)
  | none =>
    match validateConfigSet (some n) v with
    | .error e => .error e
    | .ok x =>
      match checkTimeouts td tm [(n, x)] with
      | .error e => .error e
      | .ok (td', tm') => .ok (.runtime n v x, td', tm')

/-- `args.as_chunks::<2>().0`. -/
def pairs {α : Type} : List α → List (α × α)
  | a :: c :: r => (a, c) :: pairs r
  | _ => []

/-- The pair loop: validated args, and the `old_/new_timeout_specified` flags. -/
def loop (fs : FS) : Int → Int → Bool → Bool → List (List Nat × List Nat) →
    Except LErr (List LoadArg × Bool × Bool)
  | _, _, old, new, [] => .ok ([], old, new)
  | td, tm, old, new, (f, v) :: rest =>
    match parseName (upA f) with
    | none => .error .unknown        -- `validate_config_set`'s `_ => UNKNOWN_FIELD`
    | some n =>
      match rstep fs td tm n v with
      | .error e => .error e
      | .ok (a, td', tm') =>
        -- the flags are set in the run-time branch (`:340-344`); TIMEOUT* never take the
        -- load-only one (`timeouts_runtime`)
        let old' := old || n == .timeout
        let new' := new || n == .timeoutDefault || n == .timeoutMax
        match loop fs td' tm' old' new' rest with
        | .error e => .error e
        | .ok (l, o, w) => .ok (a :: l, o, w)

/-- `apply_module_args`: the list returned is the sequence of `apply_load_arg` calls
made (`:357-359`), i.e. what gets applied. -/
def applyModuleArgs (fs : FS) (td0 tm0 : Int) (args : List (List Nat)) : Except LErr (List LoadArg) :=
  if args.length % 2 = 1 then .error .odd
  else match loop fs td0 tm0 false false (pairs args) with
    | .error e => .error e
    | .ok (vs, old, new) => if old && new then .error .timeoutMix else .ok vs

/-- What `apply_load_arg` (`config.rs:252-283`) writes: the config it sets and the value
(`Runtime` goes through `apply_config_set`, i.e. `Config.apply1` for the shared names). -/
def applyLoadArg : LoadArg → List Nat × Int
  | .cacheSize v => (b "CACHE_SIZE", v) | .threadCount v => (b "THREAD_COUNT", v)
  | .ompThreadCount v => (b "OMP_THREAD_COUNT", v)
  | .nodeCreationBuffer v => (b "NODE_CREATION_BUFFER", v)
  | .indexWorkerThreads v => (b "INDEX_WORKER_THREADS", v)
  | .importFolder _ => (b "IMPORT_FOLDER", 0) | .tempFolder _ => (b "TEMP_FOLDER", 0)
  | .boltPort => ([], 0)
  | .runtime n _ x => (n.str, x)

/-- What a load applies: nothing on error. -/
def applied (r : Except LErr (List LoadArg)) : List LoadArg :=
  match r with | .ok l => l | .error _ => []

/-! ## C's `Config_Init` -/

/-- C `isspace` in the C locale. -/
def isSpace (c : Nat) : Bool := c == 32 || (9 ≤ c && c ≤ 13)

/-- `_Config_ParseInteger`: `strtoll(s, &end, 10)` with `errno == 0`, `end != s`,
`*end == '\0'` — leading whitespace skipped, then exactly `parse::<i64>`'s grammar
(sign, ≥ 1 digit, in range). -/
def cParseInteger (v : List Nat) : Option Int := rustParseI64 (v.dropWhile isSpace)

/-- `_Config_ParseYesNo`: `strcasecmp` against yes / no. -/
def cYesNo (v : List Nat) : Option Bool :=
  if eqIC v (b "yes") then some true else if eqIC v (b "no") then some false else none

/-- `Config_Option_set` for a known field: the new `(timeout_default, timeout_max)`, or
`none` for a refusal. -/
def cSet (fs : FS) (td tm : Int) (n : N) (v : List Nat) : Option (Int × Int) :=
  let int := cParseInteger v
  let pos := int.filter (fun x => decide (0 < x))
  let nn := int.filter (fun x => decide (0 ≤ x))
  let keep {α : Type} (o : Option α) : Option (Int × Int) := o.map fun _ => (td, tm)
  match n with
  | .maxQueued | .cacheSize | .ompThreadCount | .threadCount => keep pos
  | .timeout => match nn with
    | none => none
    | some _ => if td ≠ 0 ∨ tm ≠ 0 then none else some (td, tm)   -- `_Config_check_if_new_timeout_used`
  | .timeoutDefault => match nn with
    | none => none
    | some x => if x ≠ td then (if tm ≠ 0 ∧ x > tm then none else some (x, tm)) else some (td, tm)
  | .timeoutMax => match nn with
    | none => none
    | some x => if x ≠ tm then (if x ≠ 0 ∧ td > x then none else some (td, x)) else some (td, tm)
  | .resultsetSize => keep int
  | .vkeyMax | .queryMemCapacity | .deltaMax | .nodeCreationBuffer | .maxInfo | .effectsThreshold => keep nn
  | .asyncDelete | .cmdInfo | .delayIndexing => keep (cYesNo v)
  | .importFolder => some (td, tm)
  | .tempFolder => if fs.exists_ v && fs.isDir v && fs.writable v then some (td, tm) else none
  | .jsHeap | .jsStack => keep (pos.filter (fun x => decide (1048576 ≤ x)))
  | .indexWorkerThreads | .boltPort | .effectsCompression => none   -- not C fields

/-- `Config_Contains_field`: the first C name equal up to `strcasecmp`. -/
def cLookup (f : List Nat) : Option N := cList.find? (fun n => eqIC f n.str)

def cLoop (fs : FS) : Int → Int → Bool → Bool → List (List Nat × List Nat) → Bool
  | _, _, old, new, [] => !(old && new)
  | td, tm, old, new, (f, v) :: rest =>
    match cLookup f with
    | none => false
    | some n =>
      let new' := new || n == .timeoutDefault || n == .timeoutMax
      let old' := old || n == .timeout
      match cSet fs td tm n v with
      | none => false
      | some (td', tm') => cLoop fs td' tm' old' new' rest

/-- `Config_Init` from `_Config_SetToDefaults` (both timeouts 0). -/
def cInit (fs : FS) (args : List (List Nat)) : Bool :=
  if args.length % 2 = 1 then false else cLoop fs 0 0 false false (pairs args)

/-! ## Names: `strcasecmp` = `to_ascii_uppercase` then `==` -/

def upperOk (c : Nat) : Bool := (65 ≤ c && c ≤ 90) || c == 95

theorem char_fold (c n : Nat) (h : upperOk n = true) : (lowerA c == lowerA n) = (upperA c == n) := by
  simp only [upperOk, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  unfold lowerA upperA
  apply Bool.eq_iff_iff.2
  simp only [beq_iff_eq]
  split <;> split <;> split <;> omega

theorem fold_eq (f s : List Nat) (hs : s.all upperOk = true) : eqIC f s = (upA f == s) := by
  unfold eqIC upA
  induction f generalizing s with
  | nil => cases s <;> simp
  | cons c cs ih =>
    cases s with
    | nil => simp
    | cons n ns =>
      simp only [List.all_cons, Bool.and_eq_true] at hs
      simp only [List.map_cons, List.cons_beq_cons, ih ns hs.2]
      rw [char_fold c n hs.1]

theorem names_upper : (cList ++ rustOnly).all (fun n => n.str.all upperOk) = true := by decide

theorem find_congr {α : Type} (l : List α) (p q : α → Bool) (h : ∀ a ∈ l, p a = q a) :
    l.find? p = l.find? q := by
  induction l with
  | nil => rfl
  | cons a as ih =>
    simp only [List.find?_cons, h a (List.mem_cons_self ..)]
    rw [ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]

/-- C's case-insensitive lookup is Rust's upper-case-then-match, for every C name. -/
theorem cLookup_eq (f : List Nat) : cLookup f = cList.find? (fun n => upA f == n.str) := by
  unfold cLookup
  apply find_congr
  intro n hn
  have := List.all_eq_true.1 names_upper n (List.mem_append_left _ hn)
  exact fold_eq f n.str this

theorem cLookup_parse (f : List Nat) (h : ∀ n ∈ rustOnly, parseName (upA f) ≠ some n) :
    cLookup f = parseName (upA f) := by
  rw [cLookup_eq]
  unfold parseName
  rw [List.find?_append]
  cases hc : cList.find? (fun n => upA f == n.str) with
  | some n => rfl
  | none =>
    simp only [Option.none_or]
    cases hr : rustOnly.find? (fun n => upA f == n.str) with
    | none => rfl
    | some n =>
      exfalso
      have hm := List.mem_of_find?_eq_some hr
      apply h n hm
      unfold parseName; rw [List.find?_append, hc, hr]; rfl

theorem timeouts_runtime (fs : FS) (v : List Nat) :
    validateLoadOnly fs .timeout v = none ∧ validateLoadOnly fs .timeoutDefault v = none ∧
    validateLoadOnly fs .timeoutMax v = none := ⟨rfl, rfl, rfl⟩

/-! ## Per-pair agreement -/

/-- The timeout state both engines keep: non-negative, and a set default ≤ a set max. -/
def TOk (td tm : Int) : Prop := 0 ≤ td ∧ 0 ≤ tm ∧ (td > 0 → tm > 0 → td ≤ tm)

def NoLeadSpace (v : List Nat) : Prop := ∀ c, v.head? = some c → isSpace c = false

theorem drop_noLead (v : List Nat) (h : NoLeadSpace v) : v.dropWhile isSpace = v := by
  cases v with
  | nil => rfl
  | cons c cs => simp [List.dropWhile, h c rfl]

def FsOk (fs : FS) : Prop := ∀ p, fs.exists_ p = true → fs.isDir p = true → fs.createOk p = fs.writable p

set_option maxHeartbeats 8000000 in
theorem step_agree (fs : FS) (hfs : FsOk fs) (td tm : Int) (ht : TOk td tm) (n : N) (hn : n ∉ rustOnly)
    (v : List Nat) (hv : NoLeadSpace v) :
    (match rstep fs td tm n v with | .ok (_, a, c) => some (a, c) | .error _ => none) = cSet fs td tm n v ∧
    (∀ a td' tm', rstep fs td tm n v = .ok (a, td', tm') → TOk td' tm') := by
  obtain ⟨h0, h1, h2⟩ := ht
  have h2' : td = 0 ∨ tm = 0 ∨ td ≤ tm := by
    by_cases a : 0 < td
    · by_cases c : 0 < tm
      · exact Or.inr (Or.inr (h2 a c))
      · omega
    · omega
  clear h2
  have hinv : ¬(0 < td ∧ 0 < tm ∧ tm < td) := by omega
  have hp : cParseInteger v = rustParseI64 v := by rw [cParseInteger, drop_noLead v hv]
  cases n <;> simp [rustOnly] at hn
  all_goals simp only [rstep, validateLoadOnly, validateConfigSet, cSet, hp, req, mapE, validate, jsMinSize,
    checkTimeouts, cYesNo, validateTempFolder, ofCfg]
  case timeout =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases a0 : x < 0 <;> by_cases a1 : 0 < td <;> by_cases a2 : 0 < tm <;> by_cases a3 : tm < td <;> by_cases a4 : td = 0 <;> by_cases a5 : tm = 0 <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case timeoutDefault =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a2 : x = td
      · subst a2
        by_cases b0 : 0 ≤ x <;> by_cases a0 : x < 0 <;> by_cases a1 : 0 < x <;> by_cases a3 : tm = 0 <;>
          by_cases a4 : 0 < tm <;> by_cases a5 : tm < x <;> simp [Option.filter, TOk, *] <;> first | omega | (exfalso; omega)
      · by_cases b0 : 0 ≤ x <;> by_cases a0 : x < 0 <;> by_cases a1 : 0 < x <;> by_cases a3 : tm = 0 <;>
          by_cases a4 : 0 < tm <;> by_cases a5 : tm < x <;> simp [Option.filter, TOk, *] <;> first | omega | (exfalso; omega)
  case timeoutMax =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a2 : x = tm
      · subst a2
        by_cases b0 : 0 ≤ x <;> by_cases a0 : x < 0 <;> by_cases a1 : 0 < x <;> by_cases a3 : td = 0 <;>
          by_cases a4 : 0 < td <;> by_cases a5 : x < td <;> simp [Option.filter, TOk, *] <;> first | omega | (exfalso; omega)
      · by_cases b0 : 0 ≤ x <;> by_cases a0 : x < 0 <;> by_cases a1 : 0 < x <;> by_cases a3 : td = 0 <;>
          by_cases a4 : 0 < td <;> by_cases a5 : x < td <;> simp [Option.filter, TOk, *] <;> first | omega | (exfalso; omega)
  case threadCount =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases b1 : x < 0 <;> by_cases a0 : 0 < x <;> by_cases a1 : x ≤ 0 <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case ompThreadCount =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases b1 : x < 0 <;> by_cases a0 : 0 < x <;> by_cases a1 : x ≤ 0 <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case cacheSize =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases b1 : x < 0 <;> by_cases a0 : 0 < x <;> by_cases a1 : x ≤ 0 <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case maxQueued =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases b1 : x < 0 <;> by_cases a0 : 0 < x <;> by_cases a1 : x ≤ 0 <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case vkeyMax =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a0 : x < 0 <;> by_cases a1 : 0 ≤ x <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case queryMemCapacity =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a0 : x < 0 <;> by_cases a1 : 0 ≤ x <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case deltaMax =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a0 : x < 0 <;> by_cases a1 : 0 ≤ x <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case nodeCreationBuffer =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a0 : x < 0 <;> by_cases a1 : 0 ≤ x <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case maxInfo =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a0 : x < 0 <;> by_cases a1 : 0 ≤ x <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case effectsThreshold =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases a0 : x < 0 <;> by_cases a1 : 0 ≤ x <;> by_cases a2 : 0 < td <;> by_cases a3 : 0 < tm <;> by_cases a4 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case resultsetSize =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases a0 : x < 0 <;> by_cases a1 : 0 < td <;> by_cases a2 : 0 < tm <;> by_cases a3 : tm < td <;>
        simp [Option.filter, TOk, *] at * <;> first | omega | (exfalso; omega)
  case jsHeap =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases b1 : x < 0 <;> by_cases a0 : 0 < x <;> by_cases a1 : 1048576 ≤ x <;> by_cases a3 : 0 < td <;> by_cases a4 : 0 < tm <;> by_cases a5 : tm < td <;>
        simp [Option.filter, TOk, *] <;> first | omega | (exfalso; omega)
  case jsStack =>
    cases rustParseI64 v with
    | none => simp
    | some x =>
      by_cases b0 : 0 ≤ x <;> by_cases b1 : x < 0 <;> by_cases a0 : 0 < x <;> by_cases a1 : 1048576 ≤ x <;> by_cases a3 : 0 < td <;> by_cases a4 : 0 < tm <;> by_cases a5 : tm < td <;>
        simp [Option.filter, TOk, *] <;> first | omega | (exfalso; omega)
  case asyncDelete | cmdInfo | delayIndexing =>
    cases y : eqIC v (b "yes") <;> cases z : eqIC v (b "no") <;>
      simp only [y, z, Bool.false_eq_true, ite_false, ite_true] <;>
      by_cases a3 : 0 < td <;> by_cases a4 : 0 < tm <;> by_cases a5 : tm < td <;>
      simp [TOk, *] at * <;> first | omega | (exfalso; omega)
  case importFolder =>
    by_cases a3 : 0 < td <;> by_cases a4 : 0 < tm <;> by_cases a5 : tm < td <;>
      simp [TOk, *] at * <;> first | omega | (exfalso; omega)
  case tempFolder =>
    cases e1 : fs.exists_ v <;> cases e2 : fs.isDir v <;> simp [TOk] <;> rw [hfs v e1 e2] <;>
      cases fs.writable v <;> simp <;> omega

/-! ## `check_timeout_constraints` is the #3021 cross check

`validate_timeout_cross_constraints` (`config_cmd.rs:206`) now loads the two stored values
and calls `check_timeout_constraints`; `Config.cross` models it. On the names they share
the two models agree. -/

def toN : Name → N
  | .timeout => .timeout | .timeoutDefault => .timeoutDefault | .timeoutMax => .timeoutMax
  | .resultsetSize => .resultsetSize | .maxQueued => .maxQueued | .vkeyMax => .vkeyMax
  | .maxInfo => .maxInfo | .jsHeap => .jsHeap | .cmdInfo => .cmdInfo | .cacheSize => .cacheSize

theorem fold_td (vs : List (Name × Int)) (c : Cfg) :
    (vs.foldl apply1 c).timeoutDefault =
      (vs.map fun p => (toN p.1, p.2)).foldl (fun a p => if p.1 = .timeoutDefault then p.2 else a) c.timeoutDefault := by
  induction vs generalizing c with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨a, x⟩ := p
    simp only [List.foldl_cons, List.map_cons]
    rw [ih]; congr 1
    cases a <;> simp [apply1, toN]

theorem fold_tm (vs : List (Name × Int)) (c : Cfg) :
    (vs.foldl apply1 c).timeoutMax =
      (vs.map fun p => (toN p.1, p.2)).foldl (fun a p => if p.1 = .timeoutMax then p.2 else a) c.timeoutMax := by
  induction vs generalizing c with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨a, x⟩ := p
    simp only [List.foldl_cons, List.map_cons]
    rw [ih]; congr 1
    cases a <;> simp [apply1, toN]

theorem any_toN (vs : List (Name × Int)) (n : Name) :
    ((vs.map fun p => (toN p.1, p.2)).any (·.1 == toN n)) = vs.any (·.1 == n) := by
  induction vs with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.map_cons, List.any_cons, ih]
    congr 1
    cases p.1 <;> cases n <;> rfl

/-- `Config.cross` (the GRAPH.CONFIG SET batch check) is `check_timeout_constraints` from
the stored values: the load path and SET share it. -/
theorem cross_eq (c : Cfg) (vs : List (Name × Int)) :
    mapE (cross c vs) = (checkTimeouts c.timeoutDefault c.timeoutMax
      (vs.map fun p => (toN p.1, p.2))).map (fun _ => ()) := by
  have h1 : ((vs.map fun p => (toN p.1, p.2)).any (·.1 == N.timeout)) = vs.any (·.1 == Name.timeout) :=
    any_toN vs .timeout
  have h2 : ((vs.map fun p => (toN p.1, p.2)).any (·.1 == N.timeoutDefault)) =
      vs.any (·.1 == Name.timeoutDefault) := any_toN vs .timeoutDefault
  unfold cross checkTimeouts
  simp only []
  rw [← fold_td, ← fold_tm, h1, h2]
  by_cases a : (vs.any (·.1 == .timeout) = true ∧
      ((vs.foldl apply1 c).timeoutDefault > 0 ∨ (vs.foldl apply1 c).timeoutMax > 0))
  · simp only [a, if_true]; rfl
  · simp only [a, if_false]
    by_cases b' : ((vs.foldl apply1 c).timeoutDefault > 0 ∧ (vs.foldl apply1 c).timeoutMax > 0 ∧
        (vs.foldl apply1 c).timeoutDefault > (vs.foldl apply1 c).timeoutMax)
    · simp only [b', if_true]
      cases vs.any (·.1 == Name.timeoutDefault) <;> rfl
    · simp only [b', if_false]; rfl

/-! ## The whole load -/

theorem loop_agree (fs : FS) (hfs : FsOk fs) :
    ∀ (ps : List (List Nat × List Nat)) (td tm : Int) (old new : Bool), TOk td tm →
      (∀ p ∈ ps, NoLeadSpace p.2) → (∀ p ∈ ps, ∀ n ∈ rustOnly, parseName (upA p.1) ≠ some n) →
      (match loop fs td tm old new ps with | .ok (_, o, w) => !(o && w) | .error _ => false) =
        cLoop fs td tm old new ps
  | [], _, _, _, _, _, _, _ => rfl
  | (f, v) :: rest, td, tm, old, new, ht, hv, hr => by
    have hr0 := hr (f, v) (List.mem_cons_self ..)
    have hv0 := hv (f, v) (List.mem_cons_self ..)
    simp only [loop, cLoop]
    rw [cLookup_parse f hr0]
    cases hn : parseName (upA f) with
    | none => rfl
    | some n =>
      have hno : n ∉ rustOnly := fun m => hr0 n m hn
      obtain ⟨hs, hok⟩ := step_agree fs hfs td tm ht n hno v hv0
      simp only
      cases hst : rstep fs td tm n v with
      | error e => rw [hst] at hs; simp only at hs; rw [← hs]
      | ok r =>
        obtain ⟨a, td', tm'⟩ := r
        rw [hst] at hs; simp only at hs; rw [← hs]
        simp only
        have ih := loop_agree fs hfs rest td' tm' (old || n == .timeout)
          (new || n == .timeoutDefault || n == .timeoutMax) (hok a td' tm' hst)
          (fun p hp => hv p (List.mem_cons_of_mem _ hp)) (fun p hp => hr p (List.mem_cons_of_mem _ hp))
        rw [← ih]
        cases loop fs td' tm' (old || n == .timeout) (new || n == .timeoutDefault || n == .timeoutMax) rest with
        | error e => rfl
        | ok r => obtain ⟨l, o, w⟩ := r; rfl

/-- **`apply_module_args` accepts exactly what C's `Config_Init` accepts**, for every
argument vector whose values carry no leading whitespace and that names no Rust-only
config, given only the libc/`std::fs` agreement `FsOk`. -/
theorem rust_c_same_accept (fs : FS) (hfs : FsOk fs) (args : List (List Nat))
    (hv : ∀ p ∈ pairs args, NoLeadSpace p.2)
    (hr : ∀ p ∈ pairs args, ∀ n ∈ rustOnly, parseName (upA p.1) ≠ some n) :
    (match applyModuleArgs fs 0 0 args with | .ok _ => true | .error _ => false) = cInit fs args := by
  unfold applyModuleArgs cInit
  by_cases hp : args.length % 2 = 1
  · simp only [hp, if_true]
  · simp only [hp, if_false]
    rw [← loop_agree fs hfs (pairs args) 0 0 false false ⟨by omega, by omega, by omega⟩ hv hr]
    cases loop fs 0 0 false false (pairs args) with
    | error e => rfl
    | ok r =>
      obtain ⟨l, o, w⟩ := r
      cases o <;> cases w <;> rfl

/-- A refused load applies nothing (`apply_load_arg` runs only after every pair and the
TIMEOUT check passed, `config.rs:357`). -/
theorem error_applies_nothing (fs : FS) (td tm : Int) (args : List (List Nat)) (e : LErr)
    (h : applyModuleArgs fs td tm args = .error e) : applied (applyModuleArgs fs td tm args) = [] := by
  rw [h]; rfl

theorem pairs_length {α : Type} : ∀ (l : List α), (pairs l).length = l.length / 2
  | [] => by simp [pairs]
  | [_] => by simp [pairs]
  | a :: c :: r => by
    simp only [pairs, List.length_cons, pairs_length r]; omega

theorem loop_length (fs : FS) : ∀ (ps : List (List Nat × List Nat)) (td tm : Int) (old new : Bool)
    (l : List LoadArg) (o w : Bool), loop fs td tm old new ps = .ok (l, o, w) → l.length = ps.length
  | [], _, _, _, _, l, o, w, h => by simp [loop] at h; rw [h.1]; rfl
  | (f, v) :: rest, td, tm, old, new, l, o, w, h => by
    simp only [loop] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · rename_i l' o' w' hl
          simp only [Except.ok.injEq, Prod.mk.injEq] at h
          rw [← h.1, List.length_cons, loop_length fs rest _ _ _ _ l' o' w' hl, List.length_cons]

/-- A successful load applies one `apply_load_arg` per pair. -/
theorem ok_applies_each_pair (fs : FS) (td tm : Int) (args : List (List Nat)) (vs : List LoadArg)
    (h : applyModuleArgs fs td tm args = .ok vs) : vs.length = args.length / 2 := by
  unfold applyModuleArgs at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i l o w hl
      split at h
      · cases h
      · cases h; rw [loop_length fs _ _ _ _ _ _ _ _ hl, pairs_length]

/-! ## The #3268 / W4-redis-3 counterexamples, and what still differs -/

/-- A file system where every path is a writable directory (none of these touch it). -/
def fs0 : FS := ⟨fun _ => true, fun _ => true, fun _ => true, fun _ => true⟩

/-- Fixed by #3328: `CMD_INFO YES` sets 1 (was 0), `CMD_INFO garbage` refuses the load (was
loaded with 0), a lower-case `cache_size 10` applies (was ignored), `ASYNC_DELETE yes` is
read (was ignored) — each now as C. -/
theorem fixed3328_counterexamples :
    applyModuleArgs fs0 0 0 [b "CMD_INFO", b "YES"] = .ok [.runtime .cmdInfo (b "YES") 1] ∧
    cInit fs0 [b "CMD_INFO", b "YES"] = true ∧
    applyModuleArgs fs0 0 0 [b "CMD_INFO", b "garbage"] = .error .failed ∧
    cInit fs0 [b "CMD_INFO", b "garbage"] = false ∧
    applyModuleArgs fs0 0 0 [b "cache_size", b "10"] = .ok [.cacheSize 10] ∧
    cInit fs0 [b "cache_size", b "10"] = true ∧
    applyModuleArgs fs0 0 0 [b "ASYNC_DELETE", b "yes"] = .ok [.runtime .asyncDelete (b "yes") 1] := by
  decide

/-- Odd arity, unknown names, TIMEOUT mixed with the new knobs, and a default above the
maximum are refused by both (confirmed live). -/
theorem both_refuse :
    applyModuleArgs fs0 0 0 [b "CACHE_SIZE"] = .error .odd ∧ cInit fs0 [b "CACHE_SIZE"] = false ∧
    applyModuleArgs fs0 0 0 [b "NO_SUCH", b "1"] = .error .unknown ∧ cInit fs0 [b "NO_SUCH", b "1"] = false ∧
    applyModuleArgs fs0 0 0 [b "TIMEOUT", b "5", b "TIMEOUT_DEFAULT", b "5"] = .error .timeoutMix ∧
    cInit fs0 [b "TIMEOUT", b "5", b "TIMEOUT_DEFAULT", b "5"] = false ∧
    applyModuleArgs fs0 0 0 [b "TIMEOUT_DEFAULT", b "10", b "TIMEOUT_MAX", b "5"] = .error .maxBelowDefault ∧
    cInit fs0 [b "TIMEOUT_DEFAULT", b "10", b "TIMEOUT_MAX", b "5"] = false ∧
    applyModuleArgs fs0 0 0 [b "CACHE_SIZE", b "-0"] = .error .failed ∧ cInit fs0 [b "CACHE_SIZE", b "-0"] = false := by
  decide

/-- **Divergence (confirmed live)**: C's `strtoll` skips leading whitespace. -/
theorem leading_space_divergence :
    applyModuleArgs fs0 0 0 [b "CACHE_SIZE", b " 10"] = .error .failed ∧
    cInit fs0 [b "CACHE_SIZE", b " 10"] = true ∧
    applyModuleArgs fs0 0 0 [b "TIMEOUT", b "\t5"] = .error .failed ∧
    cInit fs0 [b "TIMEOUT", b "\t5"] = true := by
  decide

/-- **Rust-only names (confirmed live)**: accepted by Rust, an unknown field to C. -/
theorem rust_only_names :
    applyModuleArgs fs0 0 0 [b "INDEX_WORKER_THREADS", b "2"] = .ok [.indexWorkerThreads 2] ∧
    cInit fs0 [b "INDEX_WORKER_THREADS", b "2"] = false ∧
    applyModuleArgs fs0 0 0 [b "BOLT_PORT", b "7687"] = .ok [.boltPort] ∧
    cInit fs0 [b "BOLT_PORT", b "7687"] = false ∧
    applyModuleArgs fs0 0 0 [b "EFFECTS_COMPRESSION", b "1"] = .ok [.runtime .effectsCompression (b "1") 1] ∧
    cInit fs0 [b "EFFECTS_COMPRESSION", b "1"] = false := by
  decide

end RedisLayer.LoadArgs
