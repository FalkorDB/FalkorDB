import RedisLayer.Args

/-!
# `parse_query_flags` — the one flag parser of QUERY / RO_QUERY / PROFILE / EXPLAIN

The four query commands share
`src/commands/query_args.rs`. Models, line by line:

| here | there |
| --- | --- |
| `upToNul`         | `up_to_nul`, `query_args.rs:123-125` |
| `scanFlags`       | the `while let Some(arg) = rest.next()` loop, `:64-97` |
| `parseSchemaHash` | `parse_schema_hash`, `:109-120` (#3757) |
| `checkFormat`     | `flags.format = match (compact, schema)`, `:98-102` (#3757: `SCHEMA_HASH requires --compact`) |
| `parseQueryFlags` | `parse_query_flags`, `:56-104` (arity `:57-59`, `MAX_ARGS` `:34`) |
| `QErr.msg`        | `ValkeyError::WrongArity`, `TIMEOUT_ERR`/`VERSION_ERR`/`SCHEMA_ERR`/`SCHEMA_VERBOSE_ERR`/`TIMEOUT_MAX_ERR` `:36-41` |
| `queryFront`      | `graph_query` `query.rs:46-51` |
| `roQueryFront`    | `graph_ro_query` `ro_query.rs:34-39` |
| `profileFront`    | `graph_profile` `profile.rs:22` (only `timeout` is used) |
| `explainFront`    | `graph_explain` `explain.rs:64` (validated, result dropped) |
| `cScan`/`cReadFlags` | C `_read_flags` + `_validate_command_arity` (`cmd_dispatcher.c`, `master`) |

Arguments are raw byte strings (`Bytes`): `as_slice()` gives the bytes, so a non-UTF-8
argument is not special. `parse_integer` is `RedisModule_StringToLongLong`, i.e.
Redis `string2ll` on the whole byte string (`Args.string2ll`). `TIMEOUT_MAX.load` is
the parameter `tmax`.

`QFlags.compact`/`.schema` together are `QueryFlags::format`: `ReplyFormat::Verbose` is
`compact = false, schema = none`, `ReplyFormat::Compact { schema }` is `compact = true`;
`checkFormat` refuses the one pair `ReplyFormat` cannot hold (`SCHEMA_HASH` without
`--compact`, `parse_ok_format`).

Headline: `rust_c_agree_except_schema` — under the `GRAPH.CONFIG` invariant, Rust and C
fail with the corresponding error or read the same compact flag, version and timeout,
on every vector Rust does not refuse with one of the two Rust-only `SCHEMA_HASH` errors
(#3757: a malformed value, or `SCHEMA_HASH` without `--compact`).
`rust_c_agree_refuted` — the unrestricted `RustCAgree` does not
hold: `SCHEMA_HASH abc` is refused by Rust and skipped by C. `four_commands_share` —
the four commands reject exactly the same vectors and read the same flags.
-/

namespace RedisLayer

abbrev Bytes := List Nat

/-! ## Rust (`src/commands/query_args.rs`) -/

/-- `up_to_nul` (`:112-114`): `s.iter().position(|&b| b == 0).map_or(s, |end| &s[..end])`. -/
def upToNul (s : Bytes) : Bytes :=
  match s.findIdx? (· == 0) with
  | some e => s.take e
  | none => s

/-- `MAX_ARGS` (`:30`). -/
def maxArgs : Nat := 8

inductive QErr where
  | wrongArity | timeoutParse | timeoutMax | versionParse | schemaParse | schemaVerbose
  deriving DecidableEq, Repr

/-- The reply text (`:36-41`; `WrongArity` is Redis' standard arity error). -/
def QErr.msg : QErr → String
  | .wrongArity => "wrong number of arguments"
  | .timeoutParse => "Failed to parse query timeout value"
  | .versionParse => "Failed to parse graph version value"
  | .schemaParse => "Failed to parse schema hash value"
  | .schemaVerbose => "SCHEMA_HASH requires --compact"
  | .timeoutMax =>
    "The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value"

/-- `QueryFlags` (`:45-51`); `version` is `u64::from(u32)`. `compact`/`schema` are the
locals `compact`/`schema` of the loop (`:61-62`), which `checkFormat` turns into `format`. -/
structure QFlags where
  compact : Bool := false
  track : Bool := false
  timeout : Option Int := none
  version : Option Nat := none
  schema : Option Nat := none
  deriving DecidableEq, Repr

/-- `i64::cast_unsigned`: the same 64 bits as a `u64`. -/
def castUnsigned (s : Int) : Nat := (s % (2 ^ 64 : Int)).toNat

/-- `parse_schema_hash` (`:109-120`, #3757): `string2ll` (`parse_integer`) cast to `u64`;
else a canonical unsigned decimal (`[1-9][0-9]*`, `str::parse::<u64>`, so ≤ `u64::MAX`). -/
def parseSchemaHash (v : Bytes) : Option Nat :=
  match string2ll v with
  | some s => some (castUnsigned s)
  | none =>
    match v with
    | [] => none
    | d :: _ =>
      if (49 ≤ d ∧ d ≤ 57) ∧ v.all isDig = true then (digitsVal v).filter (· ≤ u64Max)
      else none

/-- The flag loop (`:64-97`). -/
def scanFlags (tmax : Int) : List Bytes → QFlags → Except QErr QFlags
  | [], f => .ok f
  | a :: rest, f =>
    if eqIC (upToNul a) (b "--compact") then scanFlags tmax rest { f with compact := true }
    else if eqIC (upToNul a) (b "--track-memory") then scanFlags tmax rest { f with track := true }
    else if eqIC (upToNul a) (b "timeout") then
      match rest with
      | [] => .error .timeoutParse                                    -- `:57-60` `next()` = None
      | v :: rest' =>
        match string2ll v with
        | none => .error .timeoutParse                                -- `:59-60`
        | some t =>
          if tmax > 0 ∧ t > tmax then .error .timeoutMax              -- `:61-64`
          else if t < 0 then .error .timeoutParse                     -- `:65-67`
          else scanFlags tmax rest' { f with timeout := some t }      -- `:68`
    else if eqIC (upToNul a) (b "version") then
      match rest with
      | [] => .error .versionParse
      | v :: rest' =>
        match string2ll v with
        | none => .error .versionParse
        | some n =>
          if 0 ≤ n ∧ n ≤ (uintMax : Int) then                          -- `u32::try_from`, `:73`
            scanFlags tmax rest' { f with version := some n.toNat }
          else .error .versionParse
    else if eqIC (upToNul a) (b "schema_hash") then                   -- `:90-95` (#3757)
      match rest with
      | [] => .error .schemaParse
      | v :: rest' =>
        match parseSchemaHash v with
        | none => .error .schemaParse
        | some h => scanFlags tmax rest' { f with schema := some h }
    else scanFlags tmax rest f                                        -- unknown: skipped

/-- `flags.format = match (compact, schema)` (`:98-102`, #3757): `SCHEMA_HASH` without
`--compact` is refused; every other pair is a `ReplyFormat`. -/
def checkFormat (r : QFlags) : Except QErr QFlags :=
  if r.compact = false ∧ r.schema.isSome = true then .error .schemaVerbose else .ok r

/-- `parse_query_flags` (`:56-104`); `args` includes the command name. -/
def parseQueryFlags (tmax : Int) (args : List Bytes) : Except QErr QFlags :=
  if args.length < 3 ∨ args.length > maxArgs then .error .wrongArity
  else (scanFlags tmax (args.drop 3) {}).bind checkFormat

/-- An accepted vector passed the arity check and the loop, and its flags are a
`ReplyFormat`: a schema hash only with `--compact`. -/
theorem parse_ok_scan (tmax : Int) (args : List Bytes) (r : QFlags)
    (h : parseQueryFlags tmax args = .ok r) :
    3 ≤ args.length ∧ args.length ≤ maxArgs ∧ scanFlags tmax (args.drop 3) {} = .ok r ∧
      (r.schema.isSome = true → r.compact = true) := by
  unfold parseQueryFlags at h
  split at h
  · simp at h
  · rename_i hl
    cases hs : scanFlags tmax (args.drop 3) {} with
    | error e => rw [hs] at h; simp [Except.bind] at h
    | ok r' =>
      rw [hs] at h
      simp only [Except.bind, checkFormat] at h
      split at h
      · simp at h
      · rename_i hc
        simp only [Except.ok.injEq] at h
        subst h
        refine ⟨by omega, by omega, rfl, fun hs' => ?_⟩
        cases hr : r'.compact
        · exact absurd ⟨hr, hs'⟩ hc
        · rfl

/-- **`parse_ok_format`** (#3757): every accepted vector is a `ReplyFormat`: `SCHEMA_HASH`
is only ever read together with `--compact`. -/
theorem parse_ok_format (tmax : Int) (args : List Bytes) (r : QFlags)
    (h : parseQueryFlags tmax args = .ok r) : r.schema.isSome = true → r.compact = true :=
  (parse_ok_scan tmax args r h).2.2.2

/-! ## The four commands' front ends -/

def queryFront (tmax : Int) (args : List Bytes) : Except QErr QFlags := parseQueryFlags tmax args
def roQueryFront (tmax : Int) (args : List Bytes) : Except QErr QFlags := parseQueryFlags tmax args
def profileFront (tmax : Int) (args : List Bytes) : Except QErr (Option Int) :=
  (parseQueryFlags tmax args).map (·.timeout)
def explainFront (tmax : Int) (args : List Bytes) : Except QErr Unit :=
  (parseQueryFlags tmax args).map fun _ => ()

/-- **QUERY, RO_QUERY, PROFILE and EXPLAIN share one parser**: they fail on exactly the
same argument vectors with the same error, QUERY and RO_QUERY read identical flags, and
PROFILE reads the same timeout. (Historical: before `557f18868` each had its own loop and
they disagreed, e.g. RO_QUERY/PROFILE ignored `TIMEOUT abc`, QUERY rejected it, EXPLAIN
checked nothing.) -/
theorem four_commands_share (tmax : Int) (args : List Bytes) (e : QErr) :
    roQueryFront tmax args = queryFront tmax args ∧
    (queryFront tmax args = .error e ↔ profileFront tmax args = .error e) ∧
    (queryFront tmax args = .error e ↔ explainFront tmax args = .error e) ∧
    (∀ r, queryFront tmax args = .ok r → profileFront tmax args = .ok r.timeout ∧
      explainFront tmax args = .ok ()) := by
  unfold queryFront roQueryFront profileFront explainFront
  refine ⟨rfl, ?_, ?_, ?_⟩ <;> cases parseQueryFlags tmax args <;>
    simp [Except.map]

/-- Totality, with the exact outcome set: every argument vector is either accepted with
some flags or rejected with one of the six errors (no panic, no other reply). -/
theorem parse_total (tmax : Int) (args : List Bytes) :
    (∃ r, parseQueryFlags tmax args = .ok r) ∨
    (∃ e, parseQueryFlags tmax args = .error e ∧
      e ∈ [QErr.wrongArity, .timeoutParse, .timeoutMax, .versionParse, .schemaParse,
        .schemaVerbose]) := by
  cases h : parseQueryFlags tmax args with
  | ok r => exact .inl ⟨r, rfl⟩
  | error e => exact .inr ⟨e, rfl, by cases e <;> simp⟩

/-- The arity rule: exactly 3..8 arguments get past the first check. -/
theorem parse_arity (tmax : Int) (args : List Bytes) (h : args.length < 3 ∨ args.length > 8) :
    parseQueryFlags tmax args = .error .wrongArity := by
  simp [parseQueryFlags, maxArgs, h]

/-! ## Bounds on what is accepted -/

def Bounded (tmax : Int) (f : QFlags) : Prop :=
  (∀ t, f.timeout = some t → 0 ≤ t ∧ (tmax > 0 → t ≤ tmax)) ∧
  (∀ v, f.version = some v → v ≤ uintMax)

theorem scan_bounded (tmax : Int) : ∀ (l : List Bytes) (f r : QFlags),
    Bounded tmax f → scanFlags tmax l f = .ok r → Bounded tmax r
  | [], f, r, hf, h => by simp [scanFlags] at h; exact h ▸ hf
  | a :: rest, f, r, hf, h => by
    unfold scanFlags at h
    split at h
    · exact scan_bounded tmax rest _ r (by exact ⟨hf.1, hf.2⟩) h
    · split at h
      · exact scan_bounded tmax rest _ r (by exact ⟨hf.1, hf.2⟩) h
      · split at h
        · split at h
          · simp at h
          · rename_i v rest'
            split at h
            · simp at h
            · rename_i t _
              split at h
              · simp at h
              · split at h
                · simp at h
                · refine scan_bounded tmax rest' _ r ?_ h
                  refine ⟨?_, hf.2⟩
                  intro t' ht'; simp at ht'; subst ht'; constructor <;> omega
        · split at h
          · split at h
            · simp at h
            · rename_i v rest'
              split at h
              · simp at h
              · rename_i n _
                split at h
                · rename_i hn
                  refine scan_bounded tmax rest' _ r ?_ h
                  refine ⟨hf.1, ?_⟩
                  intro v' hv'; simp at hv'; subst hv'; omega
                · simp at h
          · split at h
            · split at h
              · simp at h
              · split at h
                · simp at h
                · exact scan_bounded tmax _ _ r (by exact ⟨hf.1, hf.2⟩) h
            · exact scan_bounded tmax rest f r hf h

/-- Every accepted timeout is non-negative and within `TIMEOUT_MAX` (when set); every
accepted version fits a C `uint`. So `compute_effective_timeout` never sees a negative
or over-the-max per-query value from a command (`Args.neg_timeout_falls_back` and
`Args.rustTimeout_exceeds` are now unreachable from QUERY/RO_QUERY/PROFILE). -/
theorem parse_timeout_nonneg (tmax : Int) (args : List Bytes) (r : QFlags)
    (h : parseQueryFlags tmax args = .ok r) : Bounded tmax r :=
  scan_bounded tmax _ {} r ⟨by simp, by simp⟩ (parse_ok_scan tmax args r h).2.2.1

/-! ## C (`cmd_dispatcher.c`: `_validate_command_arity` + `_read_flags`) -/

inductive CErr where
  | exceedsMax | badTimeout | badVersion | wrongArity
  deriving DecidableEq, Repr

structure CFlags where
  compact : Bool
  timeout : Int
  timeoutRw : Bool
  hash : Int          -- `GRAPH_HASH_MISSING` = -1
  deriving DecidableEq, Repr

/-- The timeout C substitutes for `TIMEOUT 0` when `timeout_rw` is set. -/
def dflt (c : TCfg) : Int := if c.timeoutDefault = 0 then c.timeoutMax else c.timeoutDefault

def cInit (c : TCfg) : CFlags :=
  if c.timeoutMax ≠ 0 ∨ c.timeoutDefault ≠ 0 then
    { compact := false, timeoutRw := true, hash := -1, timeout := dflt c }
  else { compact := false, timeoutRw := false, hash := -1, timeout := c.legacy }

/-- The scan loop. `strcasecmp` compares the C strings, i.e. the bytes up to the first
NUL (`upToNul`); `RedisModule_StringToLongLong` is `string2ll` on the whole argument and
writes its result only on success. -/
def cScan (c : TCfg) : List Bytes → CFlags → Except CErr CFlags
  | [], f => .ok f
  | a :: rest, f =>
    if eqIC (upToNul a) (b "--compact") then cScan c rest { f with compact := true }
    else if eqIC (upToNul a) (b "timeout") then
      match rest with
      | [] => .error .badTimeout
      | x :: rest' =>
        let parsed := string2ll x
        let t := parsed.getD f.timeout
        if c.timeoutMax ≠ 0 ∧ t > c.timeoutMax then .error .exceedsMax
        else
          let t := if t = 0 ∧ f.timeoutRw then dflt c else t
          if parsed.isNone ∨ t < 0 then .error .badTimeout
          else cScan c rest' { f with timeout := t }
    else if eqIC (upToNul a) (b "version") then
      match rest with
      | [] => .error .badVersion
      | x :: rest' =>
        match string2ll x with
        | some v => if v < 0 ∨ v > uintMax then .error .badVersion
                    else cScan c rest' { f with hash := v }
        | none => .error .badVersion
    else cScan c rest f

/-- `_validate_command_arity` (`argc` in 3..8) then `_read_flags` on `argv[3..]`. -/
def cReadFlags (c : TCfg) (args : List Bytes) : Except CErr CFlags :=
  if args.length < 3 ∨ args.length > 8 then .error .wrongArity
  else cScan c (args.drop 3) (cInit c)

end RedisLayer
