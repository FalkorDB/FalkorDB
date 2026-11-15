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
| `cReplyName`, `replyBytes`, `scanPage`, `scanLoop` | `list.rs` `c_reply_name` `:48`, `reply_bytes` `:59`, `graph_list` `:67` |
| `memArgs`, `memTotal`, `cSchemaName` | `memory.rs` `graph_memory` `:181` (#3053), `memory_report` `:69`, `c_schema_name` `:65` |
| `slowlogCmd`      | `slowlog.rs` `graph_slowlog` `:19` |
| `udfCmd`, `udfLoadArgs`, `udfListArgs` | `udf.rs` `:47-229` |
| `exited`, `exitCode` | `copy.rs` `graph_copy` `:29` (`waitpid` status decoding) |
| `restoreCmd`      | `restore.rs` `graph_restore` `:14` |
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

inductive EffOut | okNoop | okApplied | failed
  deriving DecidableEq, Repr

/-- `graph_effect`: an empty payload is a no-op; otherwise fork (nothing to take since #3657),
apply, commit and replicate, or drop the version and report through the divergence guard. Since
#2846 the commit is folded into the result (effect.rs:85): a version `Graph::validate`
refuses (`valid = false`) takes the same failure arm as a rejected buffer. -/
def effectCmd (empty applyOk : Bool) (valid : Bool := true) : EffOut × List String :=
  if empty then (.okNoop, [])
  else if applyOk then
    if valid then (.okApplied, ["commit", "replicate"])
    else (.failed, ["commit_refused", "divergence_guard"])
  else (.failed, ["divergence_guard"])

theorem effect_replicates_iff (e a v : Bool) :
    "replicate" ∈ (effectCmd e a v).2 ↔ e = false ∧ a = true ∧ v = true := by
  cases e <;> cases a <;> cases v <;> decide

/-- A refused commit forces the same resync as a rejected buffer (the replica must not
keep serving a graph that silently stopped following). -/
theorem effect_refused_resyncs :
    effectCmd false true false = (.failed, ["commit_refused", "divergence_guard"]) := rfl

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

/-- One element of a `SCAN` reply as `call_ext` hands it over: a string (any bytes, not
necessarily UTF-8), another reply type, or an error. -/
inductive ScanElem | str (b : List Nat) | other | err
  deriving DecidableEq, Repr

/-- `reply_bytes`: a string element's bytes, unconverted; anything else is `None`, which
`graph_list` turns into `ERR Failed to list graphs` instead of a panic. -/
def replyBytes : ScanElem → Option (List Nat)
  | .str b => some b
  | _ => none

theorem replyBytes_str (b : List Nat) : replyBytes (.str b) = some b := rfl
theorem replyBytes_nonstr (e : ScanElem) (h : ∀ b, e ≠ .str b) : replyBytes e = none := by
  cases e with
  | str b => exact absurd rfl (h b)
  | other => rfl
  | err => rfl

/-- One `SCAN` page as `graph_list` reads it: each element through `reply_bytes`, its
name through `c_reply_name`; one element that is not a string makes the whole command
`ERR Failed to list graphs` (`none`). -/
def scanPage : List ScanElem → Option (List (List Nat))
  | [] => some []
  | e :: es => (replyBytes e).bind fun b => (scanPage es).map (cReplyName b :: ·)

/-- `graph_list`'s `SCAN … TYPE graphdata` loop over a finite cursor chain: the names of
every page, in order, or `none` as soon as a page has an element that is not a string.

Taken as given rather than modelled: the arity check (list.rs:71); the shape of each reply,
an array whose first two elements are the cursor and an array of keys; a reply that is not an
array, or lacks those two, is also `ERR Failed to list graphs` (:86-92); a
`SCAN` error reply, passed through as the server's own error (:86); the cursor, read through
`reply_bytes` as well and failing the same way if it is not a string (:96); and termination,
by handing the model the cursor chain as a finite list of pages. -/
def scanLoop : List (List ScanElem) → Option (List (List Nat))
  | [] => some []
  | p :: ps => (scanPage p).bind fun ns => (scanLoop ps).map (ns ++ ·)

theorem scanPage_nonul (p : List ScanElem) : ∀ ns, scanPage p = some ns → ∀ n ∈ ns, 0 ∉ n := by
  induction p with
  | nil => intro ns h; simp [scanPage] at h; subst h; simp
  | cons e es ih =>
    intro ns h n hn
    cases e with
    | str b =>
      simp [scanPage, replyBytes] at h
      cases hs : scanPage es with
      | none => simp [hs] at h
      | some ms =>
        simp [hs] at h
        subst h
        simp only [List.mem_cons] at hn
        rcases hn with rfl | hn
        · exact cReplyName_nonul b
        · exact ih ms hs n hn
    | other => simp [scanPage, replyBytes] at h
    | err => simp [scanPage, replyBytes] at h

/-- No name `GRAPH.LIST` replies holds a NUL. -/
theorem scanLoop_nonul (pages : List (List ScanElem)) :
    ∀ ns, scanLoop pages = some ns → ∀ n ∈ ns, 0 ∉ n := by
  induction pages with
  | nil => intro ns h; simp [scanLoop] at h; subst h; simp
  | cons p ps ih =>
    intro ns h n hn
    cases hp : scanPage p with
    | none => simp [scanLoop, hp] at h
    | some ms =>
      cases hs : scanLoop ps with
      | none => simp [scanLoop, hp, hs] at h
      | some rs =>
        simp [scanLoop, hp, hs] at h
        subst h
        rcases List.mem_append.mp hn with hn | hn
        · exact scanPage_nonul p ms hp n hn
        · exact ih rs hs n hn

theorem scanPage_strs (bs : List (List Nat)) :
    scanPage (bs.map ScanElem.str) = some (bs.map cReplyName) := by
  induction bs with
  | nil => rfl
  | cons b bs ih => simp [scanPage, replyBytes, ih]

/-- Pages of strings, any bytes: the reply is every key cut at its first NUL, page by
page — the key's bytes, never a lossy or failed conversion of them. -/
theorem scanLoop_strs (pages : List (List (List Nat))) :
    scanLoop (pages.map (List.map ScanElem.str)) = some (pages.map (List.map cReplyName)).flatten := by
  induction pages with
  | nil => rfl
  | cons p ps ih => simp [scanLoop, scanPage_strs, ih]

theorem scanPage_nonstr (e : ScanElem) (he : ∀ b, e ≠ .str b) :
    ∀ p : List ScanElem, e ∈ p → scanPage p = none := by
  intro p
  induction p with
  | nil => intro h; simp at h
  | cons x xs ih =>
    intro h
    simp only [List.mem_cons] at h
    rcases h with rfl | h
    · cases e with
      | str b => exact absurd rfl (he b)
      | other => simp [scanPage, replyBytes]
      | err => simp [scanPage, replyBytes]
    · cases x with
      | str b => simp [scanPage, replyBytes, ih h]
      | other => simp [scanPage, replyBytes]
      | err => simp [scanPage, replyBytes]

/-- A page that fails fails the whole listing (`ERR Failed to list graphs`), never a
partial reply. -/
theorem scanLoop_page_none (p : List ScanElem) (hp : scanPage p = none) :
    ∀ pages : List (List ScanElem), p ∈ pages → scanLoop pages = none := by
  intro pages
  induction pages with
  | nil => intro h; simp at h
  | cons q qs ih =>
    intro h
    simp only [List.mem_cons] at h
    rcases h with rfl | h
    · simp [scanLoop, hp]
    · cases hq : scanPage q with
      | none => simp [scanLoop, hq]
      | some ns => simp [scanLoop, hq, ih h]

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

def udfLoadParse : List String → Except String Udf
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

/-- udf.rs:117-122 (#3263, e860645bb): after the option parse, an empty library name and then an
empty script are refused (before it, `LOAD "" code` registered functions named `.f` that no
query can call: W5-udf-4). -/
def udfLoadCheck : Udf → Except String Udf
  | .load r lib code =>
    if lib = "" then .error "empty lib name"
    else if code = "" then .error "empty script"
    else .ok (.load r lib code)
  | u => .ok u

/-- `udf_load`'s argument handling (udf.rs:68-122). -/
def udfLoadArgs (args : List String) : Except String Udf :=
  (udfLoadParse args).bind udfLoadCheck

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

/-- **Fixed by #3263**: an accepted LOAD names a non-empty library and has a non-empty script. -/
theorem udfLoad_nonempty (args : List String) (r : Bool) (l c : String)
    (h : udfLoadArgs args = .ok (.load r l c)) : l ≠ "" ∧ c ≠ "" := by
  unfold udfLoadArgs at h
  cases hp : udfLoadParse args with
  | error e => simp [hp, Except.bind] at h
  | ok u =>
    simp only [hp, Except.bind] at h
    cases u with
    | load r' l' c' =>
      simp only [udfLoadCheck] at h
      by_cases h1 : l' = ""
      · simp [h1] at h
      · by_cases h2 : c' = ""
        · simp [h1, h2] at h
        · simp only [h1, h2, if_false, Except.ok.injEq, Udf.load.injEq] at h
          obtain ⟨_, rfl, rfl⟩ := h
          exact ⟨h1, h2⟩
    | delete _ => simp [udfLoadCheck] at h
    | flush => simp [udfLoadCheck] at h
    | list _ _ => simp [udfLoadCheck] at h

theorem fixed3263_empty_lib_name :
    udfCmd ["GRAPH.UDF", "LOAD", "", "code"] = .error "empty lib name" ∧
    udfCmd ["GRAPH.UDF", "LOAD", "REPLACE", "", "code"] = .error "empty lib name" := by decide

theorem fixed3263_empty_script :
    udfCmd ["GRAPH.UDF", "LOAD", "lib", ""] = .error "empty script" ∧
    udfCmd ["GRAPH.UDF", "LOAD", "REPLACE", "lib", ""] = .error "empty script" := by decide

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

def bulkReply (n e : Nat) : String := toString n ++ " nodes created, " ++ toString e ++ " edges created"
/-- C's text (`bulk_insert.c`): `"%llu nodes created, %llu edges created"`. -/
def cBulkReply (n e : Nat) : String := toString n ++ " nodes created, " ++ toString e ++ " edges created"

/-- **Fixed by #3301 (dd333fe32)**: the success reply names relationships as C does
(`bulk_insert.rs:889,1044`; both bulk paths). This is `rfl` between two copies of the text, so it
records the wording only; it would not notice a change to the `format!` in `bulk_insert.rs`. -/
theorem bulk_reply_c (n e : Nat) : bulkReply n e = cBulkReply n e := rfl

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
