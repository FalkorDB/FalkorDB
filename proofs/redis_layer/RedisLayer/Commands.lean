import RedisLayer.QueryArgsSpec
/-!
# Small command handlers (`src/commands/*.rs`, `origin/main`)

Argument handling and reply shapes; key access, locks and replication calls are events.

| here | there |
| --- | --- |
| `debugCmd`        | `debug.rs` `graph_debug` `:5`, `debug_aux` `:23` |
| `deleteCmd`       | `delete.rs` `graph_delete` `:37` |
| `effectCmd`       | `effect.rs` `graph_effect` `:25` |
| `explainLine`, `explainCmd`, `explainCmdA` | `explain.rs` `explain` `:36`, `graph_explain` `:60` (flags `:65`) |
| `infoSections`, `formatAvg` | `info.rs` `graph_info` `:22`, `format_avg` `:88` |
| `cReplyName`, `scanLoop` | `list.rs` `c_reply_name` `:47`, `graph_list` `:60` |
| `memArgs`, `memTotal`, `cSchemaName` | `memory.rs` `graph_memory` `:181` (#3053), `memory_report` `:69`, `c_schema_name` `:65` |
| `slowlogCmd`      | `slowlog.rs` `graph_slowlog` `:16` |
| `udfCmd`, `udfLoadArgs`, `udfListArgs` | `udf.rs` `:47-229` |
| `exited`, `exitCode` | `copy.rs` `graph_copy` `:27` (`waitpid` status decoding) |
| `restoreCmd`      | `restore.rs` `graph_restore` `:12` |
| `bulkReply`       | `bulk_insert.rs` `graph_bulk_insert` success reply |
-/
namespace RedisLayer.Commands

def up (s : String) : String := String.ofList (s.toList.map Char.toUpper)

/-! ## GRAPH.DEBUG -/

inductive Res | int (i : Int) | ok | err (s : String) | wrongArity | noReply
  deriving DecidableEq, Repr

/-- Result and whether the command is replicated (`debug_aux` replicates verbatim on every
path, the unknown-action error included). -/
def debugCmd (args : List String) : Res × Bool :=
  if args.length < 3 then (.wrongArity, false) else
  match args with
  | _ :: sub :: act :: _ =>
    if up sub = "AUX" then
      if up act = "START" then (.int 1, true)
      else if up act = "END" then (.int 0, true)
      else (.err ("Unknown AUX action: " ++ act), true)
    else (.err ("Unknown DEBUG subcommand: " ++ sub), false)
  | _ => (.wrongArity, false)

theorem debug_start : debugCmd ["GRAPH.DEBUG", "aux", "start"] = (.int 1, true) := by decide
theorem debug_bad_action_replicated : (debugCmd ["GRAPH.DEBUG", "AUX", "x"]).2 = true := by decide

/-! ## GRAPH.DELETE -/

inductive Ev | delStream (name : List Char) | delKey | replicate
  deriving DecidableEq, Repr

/-- `isGraph`: the key holds a graph; `name` is `c_graph_name(key)`. -/
def deleteCmd (argc : Nat) (isGraph : Bool) (name : List Char) : Option (List Ev) :=
  if argc ≠ 2 then none
  else if isGraph then some [.delStream name, .delKey, .replicate] else none

theorem delete_stream_first (n : List Char) : deleteCmd 2 true n = some [.delStream n, .delKey, .replicate] := rfl
theorem delete_nongraph (n : List Char) : deleteCmd 2 false n = none := rfl

/-! ## GRAPH.EFFECT -/

inductive EffOut | okNoop | busy | okApplied | failed
  deriving DecidableEq, Repr

/-- `graph_effect`: an empty payload is a no-op; otherwise take the write slot, apply,
commit and replicate, or roll back and report through the divergence guard. Since
#2846 the commit is folded into the result (effect.rs:85): a version `Graph::validate`
refuses (`valid = false`) takes the same failure arm as a rejected buffer. -/
def effectCmd (empty slotFree applyOk : Bool) (valid : Bool := true) : EffOut × List String :=
  if empty then (.okNoop, [])
  else if !slotFree then (.busy, [])
  else if applyOk then
    if valid then (.okApplied, ["commit", "replicate"])
    else (.failed, ["commit_refused", "rollback", "divergence_guard"])
  else (.failed, ["rollback", "divergence_guard"])

theorem effect_replicates_iff (e s a v : Bool) :
    "replicate" ∈ (effectCmd e s a v).2 ↔ e = false ∧ s = true ∧ a = true ∧ v = true := by
  cases e <;> cases s <;> cases a <;> cases v <;> decide

/-- A refused commit forces the same resync as a rejected buffer (the replica must not
keep serving a graph that silently stopped following). -/
theorem effect_refused_resyncs :
    effectCmd false true true false = (.failed, ["commit_refused", "rollback", "divergence_guard"]) := rfl

/-! ## GRAPH.EXPLAIN -/

/-- One plan line: four spaces per depth. -/
def explainLine (depth : Nat) (op : String) : String := String.ofList (List.replicate (depth * 4) ' ') ++ op

theorem explainLine_indent (d : Nat) (op : String) :
    (explainLine d op).toList.take (d * 4) = List.replicate (d * 4) ' ' := by
  simp [explainLine]

/-- `graph_explain`: missing graph → empty-key error; inline → reply now; else spawn. -/
inductive ExOut | emptyKey | inline | spawned
  deriving DecidableEq
def explainCmd (isGraph inline : Bool) : ExOut :=
  if !isGraph then .emptyKey else if inline then .inline else .spawned

theorem explain_missing (i : Bool) : explainCmd false i = .emptyKey := rfl
theorem explain_inline : explainCmd true true = .inline := rfl

/-! ## GRAPH.INFO -/

def lower (s : String) : String := String.ofList (s.toList.map Char.toLower)

/-- Which sections are requested: none named → all; unknown names ignored. -/
def infoSections (args : List String) : Bool × Bool × Bool :=
  let all := args.isEmpty
  let has (n : String) := args.any fun a => lower a == n
  (all || has "runningqueries", all || has "waitingqueries", all || has "objectpool")

theorem info_all : infoSections [] = (true, true, true) := rfl
theorem info_unknown_only : infoSections ["foo"] = (false, false, false) := by decide
theorem info_case : infoSections ["RunningQueries"] = (true, false, false) := by decide

/-- `format_avg`: an integral average prints as an integer. -/
def formatAvg {F : Type} (fract : F → Bool) (asInt : F → Int) (show_ : F → String) (x : F) : String :=
  if fract x then toString (asInt x) else show_ x

theorem formatAvg_int {F : Type} (fr : F → Bool) (ai : F → Int) (sh : F → String) (x : F)
    (h : fr x = true) : formatAvg fr ai sh x = toString (ai x) := by simp [formatAvg, h]

/-! ## GRAPH.LIST -/

/-- `c_reply_name`: graph names are cut at the first NUL (C's `graph_name`). -/
def cReplyName : List Nat → List Nat
  | [] => []
  | x :: xs => if x = 0 then [] else x :: cReplyName xs

theorem cReplyName_nonul (b : List Nat) : 0 ∉ cReplyName b := by
  induction b with
  | nil => simp [cReplyName]
  | cons x xs ih => unfold cReplyName; split <;> simp_all; omega

/-- The `SCAN … TYPE graphdata` loop over a finite cursor chain: names of every page. -/
def scanLoop (pages : List (List (List Nat))) : List (List Nat) :=
  (pages.map fun p => p.map cReplyName).flatten

theorem scanLoop_nonul (pages : List (List (List Nat))) : ∀ n ∈ scanLoop pages, 0 ∉ n := by
  intro n hn
  simp only [scanLoop, List.mem_flatten, List.mem_map] at hn
  obtain ⟨_, ⟨p, _, rfl⟩, hn⟩ := hn
  rw [List.mem_map] at hn
  obtain ⟨b, _, rfl⟩ := hn
  exact cReplyName_nonul b

/-! ## GRAPH.MEMORY -/

def MB : Nat := 2^20

/-- `graph_memory` argument handling (`memory.rs:181-221`): arity 2 or 4, `USAGE`, optional
`SAMPLES n` with any non-negative `n` (#3053, `fe619ac5f`: `parse::<u64>`, `try_into` →
`unwrap_or(usize::MAX)`, the identity on 64-bit), default 100. `parse` is `str::parse::<u64>`.
The clamp to `1..=10000` happens later, in `memory_usage_report`. -/
def memArgs (parse : String → Option Nat) (args : List String) : Except String Nat :=
  match args with
  | [sub, _key] => if up sub = "USAGE" then .ok 100 else .error "ERR unknown subcommand"
  | [sub, _key, kw, cnt] =>
    if up sub ≠ "USAGE" then .error "ERR unknown subcommand"
    else if up kw ≠ "SAMPLES" then .error "ERR expected SAMPLES keyword"
    else match parse cnt with
      | some n => .ok n
      | none => .error "ERR SAMPLES must be a non-negative integer"
  | _ => .error "wrong arity"

theorem memArgs_default (p : String → Option Nat) (k : String) : memArgs p ["usage", k] = .ok 100 := by
  simp [memArgs]; decide

/-- **#3053**: `SAMPLES n` is exactly what `parse::<u64>` yields — `SAMPLES 0` included (C
accepts it too; historical: at 2c874022a `memArgs_pos` proved every accepted count `≥ 1`). -/
theorem memArgs_samples (p : String → Option Nat) (k cnt : String) (n : Nat) (h : p cnt = some n) :
    memArgs p ["USAGE", k, "SAMPLES", cnt] = .ok n := by
  have h1 : up "USAGE" = "USAGE" := by decide
  have h2 : up "SAMPLES" = "SAMPLES" := by decide
  simp [memArgs, h, h1, h2]

theorem memArgs_samples_err (p : String → Option Nat) (k cnt : String) (h : p cnt = none) :
    memArgs p ["USAGE", k, "SAMPLES", cnt] = .error "ERR SAMPLES must be a non-negative integer" := by
  have h1 : up "USAGE" = "USAGE" := by decide
  have h2 : up "SAMPLES" = "SAMPLES" := by decide
  simp [memArgs, h, h1, h2]

/-- `memory_report`: each component floored to MB separately; the total is their sum. -/
structure Mem where
  labels : Nat
  rels : Nat
  nodeBlock : Nat
  unlabeled : Nat
  edgeBlock : Nat
  indices : Nat
  byLabel : List Nat
  byType : List Nat

def memFields (m : Mem) : List Nat :=
  [m.labels / MB, m.rels / MB, m.nodeBlock / MB, m.unlabeled / MB, m.edgeBlock / MB, m.indices / MB]
    ++ m.byLabel.map (· / MB) ++ m.byType.map (· / MB)

def memTotal (m : Mem) : Nat :=
  m.indices / MB + m.nodeBlock / MB + m.unlabeled / MB + m.edgeBlock / MB + m.labels / MB
    + (m.byLabel.map (· / MB)).sum + (m.byType.map (· / MB)).sum + m.rels / MB

/-- The reported total is exactly the sum of the reported parts. -/
theorem memTotal_sum (m : Mem) : memTotal m = (memFields m).sum := by
  simp [memTotal, memFields]; omega

/-- `c_schema_name`: schema names cut at NUL, like `c_reply_name`. -/
def cSchemaName (b : List Nat) : List Nat := cReplyName b

/-! ## GRAPH.SLOWLOG -/

inductive SlOut | wrongArity | emptyKey | reset | unknownSub | reply
  deriving DecidableEq

def slowlogCmd (args : List String) (isGraph : Bool) : SlOut :=
  if args.length < 1 ∨ args.length > 2 then .wrongArity
  else if !isGraph then .emptyKey
  else match args with
    | [_, sub] => if up sub = "RESET" then .reset else .unknownSub
    | _ => .reply

theorem slowlog_reset_ci : slowlogCmd ["g", "reset"] true = .reset := by decide

/-! ## GRAPH.UDF -/

inductive Udf | load (replace : Bool) (lib code : String) | delete (lib : String) | flush
  | list (filter : Option String) (withCode : Bool)
  deriving DecidableEq, Repr

def udfLoadArgs : List String → Except String Udf
  | [] => .error "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command"
  | [first] => if up first = "REPLACE" then .error "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command"
               else .error "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command"
  | first :: a :: rest =>
    if up first = "REPLACE" then
      match rest with
      | [code] => .ok (.load true a code)
      | [] => .error "ERR wrong number of arguments for 'GRAPH.UDF LOAD' command"
      | x :: _ => .error ("Unknown option given: '" ++ x ++ "'")
    else match rest with
      | [] => .ok (.load false first a)
      | x :: _ => .error ("Unknown option given: '" ++ x ++ "'")

def udfListArgs : List String → Option String → Bool → Except String Udf
  | [], f, w => .ok (.list f w)
  | s :: rest, f, w =>
    if up s = "WITHCODE" then udfListArgs rest f true
    else match f with
      | some _ => .error ("Unknown option given: '" ++ s ++ "'")
      | none => udfListArgs rest (some s) w

def udfCmd (args : List String) : Except String Udf :=
  match args with
  | [] | [_] => .error "wrong arity"
  | _ :: sub :: rest =>
    if up sub = "LOAD" then udfLoadArgs rest
    else if up sub = "DELETE" then
      (match rest with
       | [lib] => .ok (.delete lib)
       | _ => .error "ERR wrong number of arguments for 'GRAPH.UDF DELETE' command")
    else if up sub = "FLUSH" then
      (match rest with
       | [] => .ok .flush
       | _ => .error "ERR wrong number of arguments for 'GRAPH.UDF FLUSH' command")
    else if up sub = "LIST" then udfListArgs rest none false
    else .error ("Unknown UDF subcommand: " ++ sub)

theorem udf_load_plain : udfCmd ["GRAPH.UDF", "load", "lib", "code"] = .ok (.load false "lib" "code") := by decide
theorem udf_load_replace : udfCmd ["GRAPH.UDF", "LOAD", "replace", "lib", "code"] = .ok (.load true "lib" "code") := by decide
theorem udf_list_two_filters : udfCmd ["GRAPH.UDF", "LIST", "a", "b"] = .error "Unknown option given: 'b'" := by decide
theorem udf_list_withcode_anywhere : udfCmd ["GRAPH.UDF", "LIST", "WITHCODE", "a"] = .ok (.list (some "a") true) := by decide

theorem udf_delete_one : udfCmd ["GRAPH.UDF", "delete", "lib"] = .ok (.delete "lib") := by decide
theorem udf_delete_extra : (udfCmd ["GRAPH.UDF", "DELETE", "a", "b"]).toOption = none := by decide
theorem udf_flush_noargs : udfCmd ["GRAPH.UDF", "flush"] = .ok .flush := by decide
theorem udf_unknown : (udfCmd ["GRAPH.UDF", "x"]).toOption = none := by decide

/-! ## GRAPH.COPY: `waitpid` status (`copy.rs`) -/

/-- `i32::trailing_zeros`, as a bounded loop (32 for 0). -/
def tz : Nat → Nat → Nat
  | 0, _ => 0
  | k+1, s => if s % 2 = 1 then 0 else 1 + tz k (s / 2)

theorem tz_ge (j : Nat) : ∀ k s, j ≤ k → (j ≤ tz k s ↔ s % 2^j = 0) := by
  induction j with
  | zero => intros; simp [Nat.mod_one]
  | succ j ih =>
    intro k s hk
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    simp only [tz]
    by_cases hs : s % 2 = 1
    · simp only [hs, if_true]
      constructor
      · intro h; omega
      · intro h
        have : s % 2 = 0 := by
          have := Nat.mod_mod_of_dvd s (show 2 ∣ 2^(j+1) from ⟨2^j, by rw [Nat.pow_succ]; omega⟩)
          rw [← this, h]
        omega
    · have he : s % 2 = 0 := by omega
      simp only [hs, if_false]
      rw [show j + 1 ≤ 1 + tz k' (s / 2) ↔ j ≤ tz k' (s / 2) by omega, ih k' (s / 2) (by omega)]
      have hs2 : s = 2 * (s / 2) := by omega
      conv => rhs; rw [hs2, Nat.pow_succ, Nat.mul_comm (2^j) 2, Nat.mul_mod_mul_left]
      omega

/-- `status.trailing_zeros() >= 7` is `WIFEXITED` (`(status & 0x7f) == 0`). -/
theorem exited_iff (s : Nat) : 7 ≤ tz 32 s ↔ s % 128 = 0 := tz_ge 7 32 s (by decide)

def exitCode (s : Nat) : Nat := (s / 256) % 256

/-! ## GRAPH.RESTORE -/

inductive RsOut | wrongArity | notUtf8 | exists_ | decodeErr | ok
  deriving DecidableEq

def restoreCmd (argc : Nat) (utf8 keyTaken decodes : Bool) : RsOut :=
  if argc ≠ 3 then .wrongArity else if !utf8 then .notUtf8
  else if keyTaken then .exists_ else if !decodes then .decodeErr else .ok

/-- A restore never overwrites a key. -/
theorem restore_no_overwrite (u d : Bool) : restoreCmd 3 u true d ≠ .ok := by
  cases u <;> cases d <;> decide

/-! ## GRAPH.BULK reply text -/

def bulkReply (n e : Nat) : String := toString n ++ " nodes created, " ++ toString e ++ " relations created"
/-- C's text (`bulk_insert.c`): `"%llu nodes created, %llu edges created"`. -/
def cBulkReply (n e : Nat) : String := toString n ++ " nodes created, " ++ toString e ++ " edges created"

/-- **Divergence (confirmed live)**: the success reply names relationships differently. -/
theorem bulk_reply_differs (n e : Nat) : bulkReply n e ≠ cBulkReply n e := by
  intro h
  have := congrArg String.length h
  simp [bulkReply, cBulkReply] at this
  exact absurd this (by decide)

/-! ## GRAPH.PROFILE / GRAPH.EXPLAIN flags (`QueryArgs.profileFront`, `explainFront`)

Since `557f18868` (#3010) both go through `parse_query_flags` (`profile.rs:22`,
`explain.rs:65`). Historical: PROFILE used its own loop that silently dropped a garbage
timeout (`profile_timeout_garbage`, now `profile_timeout_garbage_rejected`), and EXPLAIN
validated no flags at all. -/

theorem profile_timeout_parsed :
    profileFront 0 (pre ++ [b "timeout", b "10"]) = .ok (some 10) := by decide
theorem profile_timeout_garbage_rejected :
    profileFront 0 (pre ++ [b "TIMEOUT", b "abc"]) = .error .timeoutParse := by decide
theorem profile_other_args_ignored :
    profileFront 0 (pre ++ [b "--compact"]) = .ok none := by decide

/-- `graph_explain` (`explain.rs:60-102`): the flags are checked first, then the key. -/
def explainCmdA (tmax : Int) (args : List Bytes) (isGraph inline : Bool) : Except QErr ExOut :=
  (explainFront tmax args).map fun _ => explainCmd isGraph inline

/-- A bad flag is reported before the graph is even looked up (as C). -/
theorem explain_flags_first (tmax : Int) (args : List Bytes) (e : QErr) (g i : Bool)
    (h : parseQueryFlags tmax args = .error e) : explainCmdA tmax args g i = .error e := by
  simp [explainCmdA, explainFront, h, Except.map]

theorem explain_flags_ok (tmax : Int) (args : List Bytes) (r : QFlags) (g i : Bool)
    (h : parseQueryFlags tmax args = .ok r) : explainCmdA tmax args g i = .ok (explainCmd g i) := by
  simp [explainCmdA, explainFront, h, Except.map]

end RedisLayer.Commands
