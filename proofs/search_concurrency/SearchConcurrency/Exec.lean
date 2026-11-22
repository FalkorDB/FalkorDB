import SearchConcurrency.Locks

/-
# `src/graph_core.rs`: constructors, read/detect phase, timeouts, replies

Companion of `Glue` (names, registry). Pure models of the parts of `graph_core.rs` that
`Mvcc`/`Locks` leave out; each theorem states what the Rust returns.

| Lean | Rust (origin/main 256ab4d3f) |
| --- | --- |
| `Tg.new`, `Tg.fromMvcc`        | `ThreadedGraph::new` `graph_core.rs`, `from_mvcc` |
| `WStats.modified`              | `WriteQueryOk::new` (the `modified` predicate) |
| `isWritePlan`, `execQuery`     | `execute_query` (`is_write`, `admission.take()`) |
| `execProfile`                  | `execute_profile` |
| `profileSyncProg`              | `profile_sync` |
| `effTimeout`                   | `compute_effective_timeout` |
| `invalidVersionReply`, `versionMismatch` | `reply_invalid_graph_version`, the check in `query_mut` |
| `replyProfile`                 | `reply_profile` |
-/
namespace SC.Exec

/-! ## `ThreadedGraph::new`, `from_mvcc` -/

/-- A `ThreadedGraph`: the MVCC graph it wraps (identity), the write queue, the `write_loop` flag,
the slow log, and whether its `WriterGate` is closed. -/
structure Tg where
  graph : Nat
  queue : List Nat
  writeLoop : Bool
  slowLog : List Nat
  gateClosed : Bool
deriving DecidableEq, Repr

/-- `from_mvcc`: wraps the given `MvccGraph` as it is, with an empty queue, `write_loop`
off, an empty slow log and an open gate (`Arc::default()` of `WriterGate(RwLock<()>)`). -/
def Tg.fromMvcc (g : Nat) : Tg :=
  { graph := g, queue := [], writeLoop := false, slowLog := [], gateClosed := false }

/-- `new`: the same with a freshly made `MvccGraph` (`fresh`). -/
def Tg.new (fresh : Nat) : Tg := Tg.fromMvcc fresh

theorem Tg.fromMvcc_spec (g : Nat) :
    (Tg.fromMvcc g).graph = g ∧ (Tg.fromMvcc g).queue = [] ∧ (Tg.fromMvcc g).writeLoop = false ∧
    (Tg.fromMvcc g).slowLog = [] ∧ (Tg.fromMvcc g).gateClosed = false := ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem Tg.new_spec (fresh : Nat) :
    (Tg.new fresh).graph = fresh ∧ (Tg.new fresh).queue = [] ∧ (Tg.new fresh).writeLoop = false ∧
    (Tg.new fresh).gateClosed = false := ⟨rfl, rfl, rfl, rfl⟩

/-- Neither constructor starts the write drainer: nothing is queued and `write_loop` is off, so the
first write enqueues and starts it (the `write_loop` protocol of `Locks.drain_no_lost_message`). -/
theorem Tg.fromMvcc_idle (g : Nat) : (Tg.fromMvcc g).writeLoop = false ∧ (Tg.fromMvcc g).queue = [] :=
  ⟨rfl, rfl⟩

/-! ## `WriteQueryOk::new`: did the write change anything -/

/-- The ten counters of `QueryStatistics` the predicate reads, then `runtime.effects_count`. -/
structure WStats where
  nodesCreated : Nat
  nodesDeleted : Nat
  relsCreated : Nat
  relsDeleted : Nat
  propsSet : Nat
  propsRemoved : Nat
  labelsAdded : Nat
  labelsRemoved : Nat
  indexesCreated : Nat
  indexesDropped : Nat
  effectsCount : Nat

/-- `modified`. -/
def WStats.modified (s : WStats) : Bool :=
  decide (s.nodesCreated > 0 ∨ s.nodesDeleted > 0 ∨ s.relsCreated > 0 ∨ s.relsDeleted > 0 ∨
    s.propsSet > 0 ∨ s.propsRemoved > 0 ∨ s.labelsAdded > 0 ∨ s.labelsRemoved > 0 ∨
    s.indexesCreated > 0 ∨ s.indexesDropped > 0 ∨ s.effectsCount > 0)

/-- The write is `modified` iff any counter moved or an effect was recorded. -/
theorem WStats.modified_iff (s : WStats) :
    s.modified = true ↔ (s.nodesCreated > 0 ∨ s.nodesDeleted > 0 ∨ s.relsCreated > 0 ∨
      s.relsDeleted > 0 ∨ s.propsSet > 0 ∨ s.propsRemoved > 0 ∨ s.labelsAdded > 0 ∨
      s.labelsRemoved > 0 ∨ s.indexesCreated > 0 ∨ s.indexesDropped > 0 ∨ s.effectsCount > 0) := by
  simp [WStats.modified]

/-- A write that moved nothing is not `modified`. -/
theorem WStats.untouched_not_modified :
    (WStats.mk 0 0 0 0 0 0 0 0 0 0 0).modified = false := by decide

/-- Index DDL moves no counter but replicates an effect (`effects_count`), so it is `modified`:
the reason the predicate reads `effects_count` too (the comment at). -/
theorem WStats.effect_alone_modified (n : Nat) (h : 0 < n) :
    (WStats.mk 0 0 0 0 0 0 0 0 0 0 n).modified = true := by
  simp [WStats.modified, h]

/-! ## The read/detect phase of `execute_query` / `execute_profile` -/

/-- The plan nodes the write test looks at: `IR::Commit`, `IR::CreateIndex`, `IR::DropIndex`, or
any other. -/
inductive Node | commit | createIndex | dropIndex | other
deriving DecidableEq, Repr

def Node.isWriteNode : Node → Bool
  | .commit | .createIndex | .dropIndex => true
  | .other => false

/-- `plan.iter().any(|n| matches!(n, Commit | CreateIndex | DropIndex))`. -/
def isWritePlan (plan : List Node) : Bool := plan.any Node.isWriteNode

theorem isWritePlan_commit (plan : List Node) (h : Node.commit ∈ plan) : isWritePlan plan = true := by
  simp only [isWritePlan, List.any_eq_true]; exact ⟨_, h, rfl⟩

theorem isWritePlan_read (plan : List Node) (h : ∀ n ∈ plan, n = .other) : isWritePlan plan = false := by
  simp only [isWritePlan, Bool.eq_false_iff, ne_eq, List.any_eq_true, not_exists, not_and]
  intro n hn
  rw [h n hn]; simp [Node.isWriteNode]

/-- What `execute_query` does: refuses, hands a write back to the caller, or runs a read. -/
inductive QOut
  /-- `Err("graph.RO_QUERY is to be executed only on read-only queries")` -/
  | refusedRo
  /-- `Ok(ReadQueryResult { is_write: true, .. })`: nothing executed, nothing replied -/
  | write
  /-- the read ran against the published snapshot and replied -/
  | read
  /-- `compute_effective_timeout` refused (`?`) -/
  | timeoutErr
deriving DecidableEq, Repr

/-- `execute_query` for a plan: `write` is the command's flag (`GRAPH.QUERY` vs
`RO_QUERY`), `admission` whether the query still holds its `QueueSlot`, `tmo` whether the timeout
computation succeeds. Returns the outcome and the admission slot afterwards. -/
def execQuery (plan : List Node) (write admission tmo : Bool) : QOut × Bool :=
  if isWritePlan plan then
    if write then (.write, admission) else (.refusedRo, admission)
  else
    -- `admission.take()`: a read executes here, so it is no longer queued
    if tmo then (.read, false) else (.timeoutErr, false)

/-- **A refused write plan on `RO_QUERY` takes nothing**: no execution, and the admission slot is
kept (the caller drops it on the error path). -/
theorem execQuery_ro_refused (plan : List Node) (adm tmo : Bool) (h : isWritePlan plan = true) :
    execQuery plan false adm tmo = (.refusedRo, adm) := by simp [execQuery, h]

/-- **A write keeps its slot**: the write drainer releases it when it dequeues the write (so
queued writes count against `MAX_QUEUED_QUERIES`), and `execute_query` replies nothing. -/
theorem execQuery_write_keeps_slot (plan : List Node) (adm tmo : Bool) (h : isWritePlan plan = true) :
    execQuery plan true adm tmo = (.write, adm) := by simp [execQuery, h]

/-- **A read releases its slot before it runs** (whatever the timeout verdict), and runs. -/
theorem execQuery_read (plan : List Node) (adm : Bool) (h : isWritePlan plan = false) :
    execQuery plan true adm true = (.read, false) ∧ execQuery plan false adm true = (.read, false) ∧
    (execQuery plan true adm false).2 = false := by simp [execQuery, h]

/-- A read never produces `write`/`refusedRo`: the `write` flag is irrelevant to it. -/
theorem execQuery_read_flag_free (plan : List Node) (adm tmo : Bool) (h : isWritePlan plan = false) :
    execQuery plan true adm tmo = execQuery plan false adm tmo := by simp [execQuery, h]

/-- `ProfileDetect`. -/
inductive PDetect
  | readReplied
  | write
  | timeoutErr
deriving DecidableEq, Repr

/-- `execute_profile`: a write is not executed here (it goes through the write queue and
`execute_query_write` replies it in profile mode); a read releases its slot, profiles and replies. -/
def execProfile (plan : List Node) (admission tmo : Bool) : PDetect × Bool :=
  if isWritePlan plan then (.write, admission)
  else if tmo then (.readReplied, false) else (.timeoutErr, false)

theorem execProfile_write (plan : List Node) (adm tmo : Bool) (h : isWritePlan plan = true) :
    execProfile plan adm tmo = (.write, adm) := by simp [execProfile, h]

theorem execProfile_read (plan : List Node) (adm : Bool) (h : isWritePlan plan = false) :
    execProfile plan adm true = (.readReplied, false) := by simp [execProfile, h]

/-- `execute_query` and `execute_profile` agree on what a write is, and on the slot. -/
theorem execProfile_agrees_execQuery (plan : List Node) (adm tmo : Bool) :
    (execProfile plan adm tmo).2 = (execQuery plan true adm tmo).2 := by
  by_cases h : isWritePlan plan = true <;> simp [execProfile, execQuery, h] <;> split <;> rfl

open SC.Locks in
/-- `profile_sync` as a lock program: the detect phase takes the read lock in
`QuerySession::begin` and drops it with the session (the block at); a write then runs
under `begin_writer` (GIL, write lock) and commits like an inline `GRAPH.QUERY` write. -/
def profileSyncProg : PDetect → List Op
  | .write => [.acq graphR, .rel graphR] ++ inlineWriterProg
  | _ => [.acq graphR, .rel graphR]

open SC.Locks in
/-- Both paths of `profile_sync` follow the lock order and never re-acquire a held lock; the
write path is exactly the inline writer of `all_programs_ordered`, so `Mvcc.inline_write_ok`
applies to a profiled write. -/
theorem profileSync_ordered :
    (∀ d, okProg [] (profileSyncProg d) = true) ∧ profileSyncProg .write = [.acq graphR, .rel graphR] ++ inlineWriterProg := by
  refine ⟨fun d => ?_, rfl⟩
  cases d <;> decide

/-! ## `compute_effective_timeout` -/

/-- `TIMEOUT_MAX`, `TIMEOUT_DEFAULT`, the deprecated `TIMEOUT`. -/
structure TCfg where
  tmax : Int
  tdef : Int
  legacy : Int

/-- `compute_effective_timeout`, `Err` as `none`. -/
def effTimeout (c : TCfg) (pq : Option Int) (isWrite : Bool) : Option (Option Nat) :=
  let rw := decide (c.tmax > 0 ∨ c.tdef > 0)
  let perQuery : Option (Option (Option Nat)) :=
    match pq with
    | none => none
    | some p =>
      if c.tmax > 0 ∧ p > c.tmax then some none
      else if p > 0 ∧ (isWrite = false ∨ rw = true) then some (some (some p.toNat))
      else none
  match perQuery with
  | some r => r
  | none =>
    if c.tdef > 0 then some (some c.tdef.toNat)
    else if c.tmax > 0 then some (some c.tmax.toNat)
    else if c.legacy > 0 ∧ isWrite = false then some (some c.legacy.toNat)
    else some none

/-- A per-query timeout above `TIMEOUT_MAX` is refused, for reads and writes. -/
theorem effTimeout_exceeds (c : TCfg) (p : Int) (w : Bool) (h1 : c.tmax > 0) (h2 : p > c.tmax) :
    effTimeout c (some p) w = none := by
  simp [effTimeout, h1, h2]

/-- A positive in-range per-query timeout applies to a read (always) and to a write once a
new-style config is set (C's `timeout_rw`). -/
theorem effTimeout_pq (c : TCfg) (p : Int) (hp : p > 0) (hle : c.tmax > 0 → p ≤ c.tmax) :
    effTimeout c (some p) false = some (some p.toNat) ∧
    (effTimeout c (some p) true = some (some p.toNat) ↔ c.tmax > 0 ∨ c.tdef > 0) := by
  have hn : ¬ (c.tmax > 0 ∧ p > c.tmax) := fun ⟨a, b⟩ => by have := hle a; omega
  refine ⟨by simp [effTimeout, hn, hp], ?_⟩
  by_cases hr : c.tmax > 0 ∨ c.tdef > 0
  · simp [effTimeout, hn, hp, hr]
  · simp only [not_or] at hr
    have h1 : ¬ c.tmax > 0 := hr.1
    have h2 : ¬ c.tdef > 0 := hr.2
    -- a write under the deprecated `TIMEOUT` alone has no per-query timeout: it falls through
    simp [effTimeout, h1, h2]

/-- Under the deprecated `TIMEOUT` alone only reads time out: a write is unlimited. -/
theorem effTimeout_legacy_write (c : TCfg) (h1 : ¬ c.tmax > 0) (h2 : ¬ c.tdef > 0) :
    effTimeout c none true = some none ∧
    (c.legacy > 0 → effTimeout c none false = some (some c.legacy.toNat)) := by
  refine ⟨by simp [effTimeout, h1, h2], fun h => ?_⟩
  simp [effTimeout, h1, h2, h]

/-- The fallback order with no per-query timeout: `TIMEOUT_DEFAULT`, then `TIMEOUT_MAX`. -/
theorem effTimeout_fallback (c : TCfg) (w : Bool) :
    (c.tdef > 0 → effTimeout c none w = some (some c.tdef.toNat)) ∧
    (¬ c.tdef > 0 → c.tmax > 0 → effTimeout c none w = some (some c.tmax.toNat)) := by
  refine ⟨fun h => by simp [effTimeout, h], fun h h' => by simp [effTimeout, h, h']⟩

/-- Nothing configured, nothing asked: unlimited. -/
theorem effTimeout_unlimited (w : Bool) :
    effTimeout ⟨0, 0, 0⟩ none w = some none := by
  simp [effTimeout]

/-! ## The replies -/

/-- One RESP element the module emits. -/
inductive R
  | arr (n : Nat)
  | err (s : String)
  | int (v : Int)
  | str (s : String)
deriving DecidableEq, Repr

/-- `reply_invalid_graph_version`: `[error, current_version]`. -/
def invalidVersionReply (v : Int) : List R := [.arr 2, .err "ERR invalid graph version", .int v]

/-- The array header counts exactly the two elements that follow. -/
theorem invalidVersionReply_shape (v : Int) :
    invalidVersionReply v = [.arr 2, .err "ERR invalid graph version", .int v] ∧
    (invalidVersionReply v).length = 1 + 2 := ⟨rfl, rfl⟩

/-- `query_mut`'s client-supplied schema version check: the reply is sent exactly on
a mismatch. -/
def versionMismatch (provided : Option Nat) (cur : Nat) : Bool :=
  match provided with
  | some p => decide (p ≠ cur)
  | none => false

theorem versionMismatch_iff (p cur : Nat) : versionMismatch (some p) cur = true ↔ p ≠ cur := by
  simp [versionMismatch]

theorem versionMismatch_none (cur : Nat) : versionMismatch none cur = false := rfl

/-- A profiled operator: its plan-tree index, depth and name. -/
structure POp where
  idx : Nat
  depth : Nat
  name : String

/-- `profile_data.get(&idx).copied().unwrap_or((0, ZERO))`: records produced and time. -/
def pdata (d : Nat → Option (Nat × Nat)) (i : Nat) : Nat × Nat := (d i).getD (0, 0)

/-- One line of `reply_profile`, `fmt` standing for `{time_ms:.6}`. -/
def profileLine (fmt : Nat → String) (d : Nat → Option (Nat × Nat)) (o : POp) : String :=
  String.join (List.replicate o.depth "    ") ++ o.name ++ " | Records produced: " ++
    toString (pdata d o.idx).1 ++ ", Execution time: " ++ fmt (pdata d o.idx).2 ++ " ms"

/-- `reply_profile`: the array length, then one string per operator in DFS order. -/
def replyProfile (fmt : Nat → String) (d : Nat → Option (Nat × Nat)) (ops : List POp) :
    Nat × List String :=
  (ops.length, ops.map (profileLine fmt d))

/-- **Every operator is listed, once** (#2390), and the array header is exactly their number. -/
theorem replyProfile_every_op (fmt : Nat → String) (d : Nat → Option (Nat × Nat)) (ops : List POp) :
    (replyProfile fmt d ops).1 = (replyProfile fmt d ops).2.length ∧
    (replyProfile fmt d ops).2 = ops.map (profileLine fmt d) := by
  simp [replyProfile]

/-- An operator the profile has no entry for (it never produced a record) still gets its line,
with 0 records and zero time. -/
theorem pdata_missing (d : Nat → Option (Nat × Nat)) (i : Nat) (h : d i = none) :
    pdata d i = (0, 0) := by simp [pdata, h]

theorem pdata_present (d : Nat → Option (Nat × Nat)) (i r t : Nat) (h : d i = some (r, t)) :
    pdata d i = (r, t) := by simp [pdata, h]

end SC.Exec
