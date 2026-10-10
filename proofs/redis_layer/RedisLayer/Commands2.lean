import RedisLayer.Commands
import RedisLayer.ModuleInit
import RedisLayer.Telemetry
/-!
# Constraint, record, copy, bulk and config-get handlers (`src/commands/*.rs`, `origin/main`)

| here | there |
| --- | --- |
| `validIdent`              | `constraint.rs` `is_valid_identifier` `:19` |
| `consArgs`, `string2ll`   | `graph_constraint` argument parsing `:415-511`; `parse_integer` = `RedisModule_StringToLongLong` = Redis `string2ll` (util.c) |
| `registerSchema`          | `register_constraint_schema` `:63` |
| `findStatus`              | `find_status` `:82` |
| `settle`, `attempt`       | `settle_constraint` `:171`, `attempt_settle` `:283` |
| `recordShape`             | `record.rs` `record_mut` `:44`, `graph_record` `:173` |
| `copyOut`                 | `copy.rs` `graph_copy` `:27` (`pipe`/`fork`/`_exit`/`waitpid` are libc, AXIOMATISED) |
| `bulkFlow`, `heldReplicate`, `maybeYield`, `publishOps`, `edgeRecords` | `bulk_insert.rs` `:218-1045` |
| `getNames`, `getOne`, `asI64`, `asU64` | `config_cmd.rs` `config_get_one` `:57`, `ConfigValue::as_i64/as_u64` `:294-306` |
-/
namespace RedisLayer.Commands

/-! ## GRAPH.CONSTRAINT -/

def isAlpha (c : Char) : Bool := ('a' ≤ c && c ≤ 'z') || ('A' ≤ c && c ≤ 'Z')
def isDigit (c : Char) : Bool := '0' ≤ c && c ≤ '9'

/-- `is_valid_identifier`: `[A-Za-z_][A-Za-z0-9_]*`. -/
def validIdent : List Char → Bool
  | [] => false
  | c :: cs => (isAlpha c || c == '_') && cs.all fun d => isAlpha d || isDigit d || d == '_'

theorem validIdent_nonempty (s : List Char) (h : validIdent s) : s ≠ [] := by
  intro e; subst e; simp [validIdent] at h
theorem validIdent_no_digit_start (c : Char) (cs : List Char) (h : isDigit c) (h2 : ¬ isAlpha c) (h3 : c ≠ '_') :
    validIdent (c :: cs) = false := by
  simp [validIdent, h2, h3]

inductive CT | unique | mandatory
  deriving DecidableEq, Repr
inductive ET | node | rel
  deriving DecidableEq, Repr

structure ConsReq where
  create : Bool
  ct : CT
  et : ET
  label : String
  props : List String
  deriving DecidableEq, Repr

/-- Redis `string2ll` (`util.c`), behind `RedisString::parse_integer` (`RedisModule_StringToLongLong`)
since #3053 (`fe619ac5f`): canonical decimal only — `"0"`, or an optional `-` then a digit
`1-9` then digits, within `i64`. No `+`, no leading zero, no `-0`, no blanks. This is the
documented C behaviour, stated as the spec of that FFI call. -/
def digitsVal (ds : List Char) : Option Nat :=
  ds.foldl (fun acc c => acc.bind fun n => if isDigit c then some (10 * n + (c.toNat - '0'.toNat)) else none) (some 0)

def string2ll (s : String) : Option Int :=
  match s.toList with
  | ['0'] => some 0
  | '-' :: d :: ds =>
    if '1' ≤ d && d ≤ '9' then (digitsVal (d :: ds)).bind fun n =>
      if n ≤ 2 ^ 63 then some (-(n : Int)) else none
    else none
  | d :: ds =>
    if '1' ≤ d && d ≤ '9' then (digitsVal (d :: ds)).bind fun n =>
      if n < 2 ^ 63 then some (n : Int) else none
    else none
  | [] => none

#guard string2ll "1" = some 1
#guard string2ll "255" = some 255
#guard string2ll "0" = some 0
#guard string2ll "-3" = some (-3)
#guard string2ll "+1" = none
#guard string2ll "01" = none
#guard string2ll "-0" = none
#guard string2ll " 1" = none
#guard string2ll "9223372036854775807" = some 9223372036854775807
#guard string2ll "9223372036854775808" = none

/-- The parser, from the token after the command name; `parse` is `parse_integer`
(`string2ll`, above) since #3053 — it was `str::parse::<i64>`, which takes `+1` and `01`. -/
def consArgs (parse : String → Option Int) : List String → Except String ConsReq
  | op :: _key :: ctS :: etS :: lbl :: kw :: cnt :: rest =>
    let op := up op
    if op ≠ "CREATE" ∧ op ≠ "DROP" then .error "Invalid constraint operation" else
    let ct := if up ctS = "UNIQUE" then some CT.unique else if up ctS = "MANDATORY" then some .mandatory else none
    match ct with
    | none => .error "Invalid constraint type"
    | some ct =>
    let et := if up etS = "NODE" then some ET.node
      else if up etS = "RELATIONSHIP" then some .rel else none
    match et with
    | none => .error "Invalid constraint entity type"
    | some et =>
    if !validIdent lbl.toList then .error ("Label name " ++ lbl ++ " is invalid") else
    if up kw ≠ "PROPERTIES" then .error "Expected PROPERTIES keyword" else
    match parse cnt with
    | none => .error "Number of properties must be an integer between 1 and 255"
    | some n =>
      if n < 1 ∨ n > 255 then .error "Number of properties must be an integer between 1 and 255"
      else if rest.length < n.toNat then .error "wrong arity"
      else if rest.length > n.toNat then .error "Unexpected extra arguments"
      else if rest.any (fun p => !validIdent p.toList) then .error "Property name is invalid"
      else if !rest.Nodup then .error "Properties cannot contain duplicates"
      else .ok ⟨op = "CREATE", ct, et, lbl, rest⟩
  | _ => .error "wrong arity"

/-- Every accepted request has 1..255 distinct, valid property names. -/
theorem consArgs_ok (parse : String → Option Int) (args : List String) (r : ConsReq)
    (h : consArgs parse args = .ok r) :
    1 ≤ r.props.length ∧ r.props.length ≤ 255 ∧ r.props.Nodup ∧ validIdent r.label.toList := by
  unfold consArgs at h
  split at h
  · rename_i op k ctS etS lbl kw cnt rest
    simp only at h
    split at h; · simp at h
    split at h; · simp at h
    rename_i ct _
    split at h; · simp at h
    rename_i et _
    split at h; · simp at h
    rename_i hv
    split at h; · simp at h
    split at h; · simp at h
    rename_i n _
    split at h; · simp at h
    rename_i hn
    split at h; · simp at h
    split at h; · simp at h
    split at h; · simp at h
    split at h; · simp at h
    rename_i hnd
    cases h
    simp at hv hnd ⊢
    refine ⟨by omega, by omega, hnd, hv⟩
  · simp at h

/-- **#3053**: `LABEL`/`EDGE` are no longer entity types (as in C). Historical: at
2c874022a `cons_label_alias` proved `LABEL` accepted as `NODE`. -/
theorem cons_label_refused :
    consArgs string2ll ["CREATE", "g", "UNIQUE", "LABEL", "L", "PROPERTIES", "1", "p"]
      = .error "Invalid constraint entity type" := by decide
theorem cons_edge_refused :
    consArgs string2ll ["CREATE", "g", "UNIQUE", "EDGE", "R", "PROPERTIES", "1", "p"]
      = .error "Invalid constraint entity type" := by decide

/-- **#3053**: the count is `string2ll`-canonical: `+1` and `01` are refused (C too). -/
theorem cons_count_noncanonical :
    consArgs string2ll ["CREATE", "g", "UNIQUE", "NODE", "L", "PROPERTIES", "+1", "p"]
      = .error "Number of properties must be an integer between 1 and 255" ∧
    consArgs string2ll ["CREATE", "g", "UNIQUE", "NODE", "L", "PROPERTIES", "01", "p"]
      = .error "Number of properties must be an integer between 1 and 255" := by decide

theorem cons_node_ok :
    consArgs string2ll ["create", "g", "unique", "node", "L", "properties", "1", "p"]
      = .ok ⟨true, .unique, .node, "L", ["p"]⟩ := by decide

/-- `register_constraint_schema`: interns the label/type and every property name, so a
constraint's properties are always in the attribute table (`Serial.cons_unknown_prop`
cannot arise from this path). -/
def registerSchema (attrs : List String) (props : List String) : List String :=
  attrs ++ props.filter (· ∉ attrs)

theorem registerSchema_contains (attrs props : List String) : ∀ p ∈ props, p ∈ registerSchema attrs props := by
  intro p hp
  by_cases h : p ∈ attrs
  · exact List.mem_append_left _ h
  · exact List.mem_append_right _ (List.mem_filter.2 ⟨hp, by simpa using h⟩)

/-- `find_status`: the status of the first matching constraint. -/
def findStatus {C S} (matches_ : C → Bool) (status : C → S) (cs : List C) : Option S :=
  (cs.find? matches_).map status

theorem findStatus_none {C S} (m : C → Bool) (st : C → S) (cs : List C) (h : ∀ c ∈ cs, m c = false) :
    findStatus m st cs = none := by
  unfold findStatus
  rw [List.find?_eq_none.2 (fun c hc => by simp [h c hc])]; rfl

/-! ### Settling (`settle_constraint` / `attempt_settle`) -/

/-- `WriteAbort` as the settle loop sees it; `invalid` is `WriteAbort::Invalid`
(query_session.rs:71, #2846): `MvccGraph::commit` refused the version. -/
inductive Abort | paused | busy | unregistered | notMaster | invalid
  deriving DecidableEq, Repr

/-- `attempt_settle`'s outcome given: nothing pending, the upgrade result, the write slot,
and whether `commit` accepted the version (`Graph::validate`, constraint.rs:337-340). -/
def attempt (nothingPending : Bool) (upgrade : Option Abort) (slotFree : Bool) (commitOk : Bool := true) :
    Option Abort :=
  if nothingPending then none
  else match upgrade with
    | some a => some a
    | none => if !slotFree then some .busy else if commitOk then none else some .invalid

theorem attempt_nothing (u : Option Abort) (s c : Bool) : attempt true u s c = none := rfl
theorem attempt_busy (c : Bool) : attempt false none false c = some .busy := rfl
theorem attempt_upgrade_fail (a : Abort) (s c : Bool) : attempt false (some a) s c = some a := rfl
/-- A refused version is reported as `Invalid`, not as a busy slot (constraint.rs:402-409): the
retry loop must not spend its whole budget on a permanent engine fault. -/
theorem attempt_invalid : attempt false none true false = some .invalid := rfl

/-- The retry loop on a list of attempt outcomes, with `budget` retries left before the
deadline: returns the number of attempts and whether it settled. -/
def settle : List (Option Abort) → Nat → Nat × Bool
  | [], _ => (0, false)
  | o :: os, budget =>
    match o with
    | none => (1, true)
    | some .paused | some .busy =>
      if budget = 0 then (1, false) else let r := settle os (budget - 1); (r.1 + 1, r.2)
    | some _ => (1, false)   -- unregistered, notMaster, invalid (constraint.rs:254-265): break

/-- The loop never makes more than `budget + 1` attempts. -/
theorem settle_bounded (os : List (Option Abort)) (b : Nat) : (settle os b).1 ≤ b + 1 := by
  induction os generalizing b with
  | nil => simp [settle]
  | cons o os ih =>
    simp only [settle]
    split
    · simp
    · split
      · simp
      · have := ih (b - 1); simp; omega
    · split
      · simp
      · have := ih (b - 1); simp; omega
    · simp

/-- A deleted graph or a demotion stops at once. -/
theorem settle_permanent (os : List (Option Abort)) (b : Nat) :
    settle (some .unregistered :: os) b = (1, false) ∧ settle (some .notMaster :: os) b = (1, false) ∧
    settle (some .invalid :: os) b = (1, false) :=
  ⟨rfl, rfl, rfl⟩

/-! ## GRAPH.RECORD reply shape -/

/-- `record_mut`'s reply: `[records, plan]`, one record `[op_index, ok, payload]` per
recorded step, one plan row `[index, parent|nil, op, vars]` per operator (BFS order). -/
def recordShape (records ops : Nat) : List Nat := [records, ops]

theorem recordShape_two (r o : Nat) : (recordShape r o).length = 2 := rfl

/-- `graph_record` creates the graph when the key is empty (like `GRAPH.QUERY`). -/
inductive RecOut | created | existing
  deriving DecidableEq
def recordKey (isGraph : Bool) : RecOut := if isGraph then .existing else .created

theorem recordKey_creates : recordKey false = .created := rfl

/-! ## GRAPH.COPY flow -/

inductive CopyOut | wrongArity | notUtf8 | emptySrc | destExists | pipeErr | forkErr
  | waitErr | decodeErr | childFailed | ok
  deriving DecidableEq, Repr

def copyOut (argc : Nat) (utf8 srcGraph destTaken pipeOk forkOk waitOk decodeOk : Bool) (status : Nat) : CopyOut :=
  if argc ≠ 3 then .wrongArity else if !utf8 then .notUtf8
  else if !srcGraph then .emptySrc else if destTaken then .destExists
  else if !pipeOk then .pipeErr else if !forkOk then .forkErr
  else if !waitOk then .waitErr else if !decodeOk then .decodeErr
  else if !(7 ≤ tz 32 status) || exitCode status ≠ 0 then .childFailed else .ok

/-- Success needs a clean `exit(0)` of the child: `WIFEXITED` and status 0. -/
theorem copy_ok_iff (st : Nat) :
    copyOut 3 true true false true true true true st = .ok ↔ (st % 128 = 0 ∧ exitCode st = 0) := by
  rw [← exited_iff]
  unfold copyOut
  by_cases h1 : 7 ≤ tz 32 st <;> by_cases h2 : exitCode st = 0 <;> simp [h1, h2]

/-! ## GRAPH.BULK helpers -/

/-- `maybe_yield`: a no-op with no context. -/
def maybeYield (ctxNull : Bool) : List String := if ctxNull then [] else ["yield"]

inductive BOp | node (t : Nat) | edge (t : Nat) | yield | flush
  deriving DecidableEq, Repr

/-- `bulk_insert_sync` = `bulk_insert_sync_yield` with every yield removed: same token
processing, same order. -/
def processTokens (yieldEach : Bool) (nodeToks relToks : List Nat) : List BOp :=
  (nodeToks.map fun t => BOp.node t :: (if yieldEach then [BOp.yield] else [])).flatten
  ++ (relToks.map fun t => BOp.edge t :: (if yieldEach then [BOp.yield] else [])).flatten
  ++ [BOp.flush]

def noYield (o : BOp) : Bool := o != .yield

theorem sync_is_yield_without_yields (n r : List Nat) :
    processTokens false n r = (processTokens true n r).filter noYield := by
  have h : ∀ (f : Nat → BOp) (l : List Nat), (∀ t, f t ≠ .yield) →
      (l.map fun t => f t :: ([] : List BOp)).flatten
        = ((l.map fun t => f t :: [BOp.yield]).flatten).filter noYield := by
    intro f l hf; induction l with
    | nil => rfl
    | cons x xs ih =>
      simp only [List.map_cons, List.flatten_cons, List.filter_append, ← ih]
      have h1 : noYield (f x) = true := by unfold noYield; simp [hf x]
      have h2 : noYield BOp.yield = false := rfl
      simp [List.filter, h1, h2]
  unfold processTokens
  simp only [Bool.false_eq_true, if_false, if_true, List.filter_append]
  rw [← h BOp.node n (fun t => by simp), ← h BOp.edge r (fun t => by simp)]
  rfl

/-- `HeldArgs::replicate`: argv[0] is the command, the rest its arguments. -/
def heldReplicate (argv : List String) : Option (String × List String) :=
  match argv with
  | [] => none
  | c :: rest => some (c, rest)

theorem heldReplicate_verbatim (c : String) (rest : List String) :
    heldReplicate (c :: rest) = some (c, rest) := rfl

/-- `BulkIndexDocs::publish`: commit each non-empty index batch. -/
def publishOps (nodesEmpty edgesEmpty : Bool) : List String :=
  (if nodesEmpty then [] else ["commit_index"]) ++ (if edgesEmpty then [] else ["commit_edge_index"])

/-- `process_edge_token`: records = 16-byte endpoints + properties; more records than
reserved ids is an error, `NULL` properties are dropped. -/
def edgeRecords (reserved : Nat) (records : List (List (Option Nat))) : Except String (List (List Nat)) :=
  if records.length > reserved then .error "bulk data contains more edge records than advertised count"
  else .ok (records.map fun props => props.filterMap id)

theorem edgeRecords_bound (r : Nat) (rs : List (List (Option Nat))) (out : List (List Nat))
    (h : edgeRecords r rs = .ok out) : out.length ≤ r := by
  unfold edgeRecords at h; split at h; · simp at h
  · cases h; simp; omega

/-- `discard_created_graph`: stream then key. -/
def discardOps (name : List Char) : List String := ["DEL " ++ String.ofList (Telemetry.streamName name), "delete key"]

theorem discardOps_stream_first (n : List Char) :
    (discardOps n).head? = some ("DEL " ++ String.ofList (Telemetry.streamName n)) := rfl
theorem maybeYield_null : maybeYield true = [] := rfl
theorem publishOps_empty : publishOps true true = [] := rfl

/-- `graph_bulk_insert` outcome (inline path, bulk_insert.rs:844-848; deferred path :964-975): since #2846 the version is
validated *before* the index documents are published, then committed; a refusal of
either joins the token error, so a failed `BEGIN` load removes the graph it created
and no RediSearch document outlives the discarded fork. -/
def bulkFlow (begin ok : Bool) (valid : Bool := true) : List String :=
  if ok && valid then ["validate", "publish", "commit", "replicate", "reply"]
  else (if ok then ["validate"] else []) ++
    (["rollback"] ++ (if begin then ["discard"] else []) ++ ["error"])

theorem bulk_begin_failure_discards : "discard" ∈ bulkFlow true false := by decide
theorem bulk_append_failure_keeps : "discard" ∉ bulkFlow false false := by decide
/-- A version `validate` refuses is discarded like a failed load: nothing is
published to the index, nothing replicated, and a `BEGIN` removes its key. -/
theorem bulk_invalid_discards : "discard" ∈ bulkFlow true true false ∧ "publish" ∉ bulkFlow true true false ∧
    "replicate" ∉ bulkFlow true true false := by decide
/-- Documents are published only after validation and before the commit. -/
theorem bulk_publish_after_validate :
    bulkFlow true true true = ["validate", "publish", "commit", "replicate", "reply"] := rfl

/-! ## GRAPH.CONFIG GET (`config_get_one`) -/

def getNames : List String :=
  ["TIMEOUT", "TIMEOUT_DEFAULT", "TIMEOUT_MAX", "CACHE_SIZE", "ASYNC_DELETE", "OMP_THREAD_COUNT",
   "THREAD_COUNT", "INDEX_WORKER_THREADS", "RESULTSET_SIZE", "VKEY_MAX_ENTITY_COUNT",
   "MAX_QUEUED_QUERIES", "QUERY_MEM_CAPACITY", "DELTA_MAX_PENDING_CHANGES", "NODE_CREATION_BUFFER",
   "CMD_INFO", "MAX_INFO_QUERIES", "EFFECTS_COMPRESSION", "EFFECTS_THRESHOLD", "BOLT_PORT",
   "DELAY_INDEXING", "IMPORT_FOLDER", "TEMP_FOLDER", "JS_HEAP_SIZE", "JS_STACK_SIZE"]

/-- Every documented name can be read back… -/
theorem get_covers_documented : ModuleInit.documented.all (· ∈ getNames) = true := by decide
/-- …plus three Rust-only ones. -/
theorem get_extra : getNames.filter (· ∉ ModuleInit.documented)
    = ["INDEX_WORKER_THREADS", "EFFECTS_COMPRESSION", "BOLT_PORT"] := by decide

/-- `config_get_one`: `[name, value]`, or the unknown-field error. -/
def getOne (val : String → Option Int) (name : String) : Except String (String × Int) :=
  if name ∈ getNames then match val name with
    | some v => .ok (name, v)
    | none => .error "unreadable"
  else .error "Unknown configuration field"

/-- Since #3021 the unknown-name reply is C's fixed text, without the name (was
`"Unknown configuration field 'X'"`). -/
theorem getOne_unknown_text (val : String → Option Int) :
    getOne val "NOPE" = .error "Unknown configuration field" := by
  simp [getOne, getNames]

/-- `OMP_THREAD_COUNT` falls back to the thread count when unset. -/
def ompGet (omp threads : Int) : Int := if omp > 0 then omp else threads
theorem ompGet_pos (o t : Int) (ht : t > 0) : ompGet o t > 0 := by unfold ompGet; split <;> omega

def asI64 : Int ⊕ Nat → Int
  | .inl v => v
  | .inr v => if v < 2^63 then v else (v : Int) - 2^64
def asU64 : Int ⊕ Nat → Nat
  | .inl v => (v % 2^64).toNat
  | .inr v => v

theorem asI64_small (v : Nat) (h : v < 2^63) : asI64 (.inr v) = v := by simp [asI64, h]
theorem asU64_nonneg (v : Int) (h0 : 0 ≤ v) (h : v < 2^64) : asU64 (.inl v) = v.toNat := by
  have e : v % 2^64 = v := Int.emod_eq_of_lt h0 h
  show (v % 2^64).toNat = v.toNat
  rw [e]
/-- A negative value read as unsigned wraps (`VKEY_MAX_ENTITY_COUNT -5` — known). -/
theorem asU64_neg : asU64 (.inl (-5)) = 2^64 - 5 := by decide

end RedisLayer.Commands
