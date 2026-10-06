import RedisLayer.Resp
/-!
# `src/graph_core.rs` — names, registry, write commit, dispatch decisions

| here | there (`origin/main`) |
| --- | --- |
| `upToNulB`, `cGraphKey`, `cGraphName` | `up_to_nul` `:87`, `c_graph_name` `:104`, `c_graph_key` `:119` |
| `Reg.*`                     | `register_graph` `:140`, `graph_is_registered` `:188`, `rename_graph` `:202`, `graph_free` `:1684` |
| `modified`                  | `WriteQueryOk::new` `:251` |
| `sanitise`                  | `ffi::sanitise_error` `:313` |
| `invalidVersionBytes`       | `reply_invalid_graph_version` `:933` |
| `profileDepth`, `profileOps`| `reply_profile` `:823` |
| `isWrite`, `profileDetect`  | `execute_query` `:576` / `execute_profile` `:659` write detection |
| `writeSteps`                | `execute_query_write` `:715`, `finish_write` `:1436`, `abandon_write` `:1466`, `commit_and_replicate` `:1481` |
| `dispatch`                  | `query_mut` `:945` |
| `syncOutcome`               | `query_sync` `:1154`, `profile_sync` `:1369`, `profile_mut` `:1275` |
| `loopExit`                  | `process_write_queued_query` `:1525` |
| `EFFECT_COMMAND`            | `CtxSink::replicate` `:1671` |
| `TG`                        | `ThreadedGraph::new` `:543`, `from_mvcc` `:559` |
-/
namespace RedisLayer.GraphCore

/-! ## Names -/

abbrev Bytes := List Nat

/-- Bytes before the first NUL. -/
def upToNulB : Bytes → Bytes
  | [] => []
  | b :: bs => if b = 0 then [] else b :: upToNulB bs

/-- `String::from_utf8_lossy` facts: NUL is ASCII and ends any partial sequence, so
lossy decoding splits at it. (Rust `core`, not FFI; a hypothesis structure.) -/
structure Lossy where
  dec : Bytes → List Char
  split : ∀ a b, 0 ∉ a → dec (a ++ 0 :: b) = dec a ++ '\x00' :: dec b
  nonul : ∀ a, 0 ∉ a → '\x00' ∉ dec a

def upToNulC : List Char → List Char
  | [] => []
  | c :: cs => if c = '\x00' then [] else c :: upToNulC cs

/-- `c_graph_key`: the key bytes before the first NUL. -/
def cGraphKey (key : Bytes) : Bytes := upToNulB key
/-- `c_graph_name`: `up_to_nul(to_string_lossy(key))`. -/
def cGraphName (L : Lossy) (key : Bytes) : List Char := upToNulC (L.dec key)

theorem upToNulB_nonul (k : Bytes) : 0 ∉ upToNulB k := by
  induction k with
  | nil => simp [upToNulB]
  | cons b bs ih => unfold upToNulB; split <;> simp_all; omega

theorem upToNulC_nofree (s : List Char) (h : '\x00' ∉ s) : upToNulC s = s := by
  induction s with
  | nil => rfl
  | cons c cs ih =>
    have hc : c ≠ '\x00' := fun e => h (by simp [e])
    simp [upToNulC, hc, ih (fun m => h (by simp [m]))]

theorem upToNulC_split (a b : List Char) (h : '\x00' ∉ a) : upToNulC (a ++ '\x00' :: b) = a := by
  induction a with
  | nil => simp [upToNulC]
  | cons c cs ih =>
    have hc : c ≠ '\x00' := fun e => h (by simp [e])
    simp [upToNulC, hc, ih (fun m => h (by simp [m]))]

theorem upToNulB_split (k : Bytes) :
    (0 ∈ k → ∃ tail, k = upToNulB k ++ 0 :: tail) ∧ (0 ∉ k → upToNulB k = k) := by
  induction k with
  | nil => simp [upToNulB]
  | cons b bs ih =>
    by_cases hb : b = 0
    · subst hb; simp [upToNulB]
    · simp only [upToNulB, hb, if_false]
      refine ⟨fun h => ?_, fun h => ?_⟩
      · have : 0 ∈ bs := by simp at h; rcases h with h | h; exact absurd h.symm hb; exact h
        obtain ⟨t, ht⟩ := ih.1 this
        exact ⟨t, by simp; exact ht⟩
      · have : 0 ∉ bs := fun m => h (by simp [m])
        rw [ih.2 this]

/-- **The name C would see is the lossy decoding of the key C would build**: the two
helpers agree. -/
theorem cGraphName_key (L : Lossy) (key : Bytes) : cGraphName L key = L.dec (cGraphKey key) := by
  unfold cGraphName cGraphKey
  by_cases h0 : 0 ∈ key
  · obtain ⟨t, ht⟩ := (upToNulB_split key).1 h0
    conv => lhs; rw [ht]
    rw [L.split _ _ (upToNulB_nonul key)]
    exact upToNulC_split _ _ (L.nonul _ (upToNulB_nonul key))
  · rw [(upToNulB_split key).2 h0, upToNulC_nofree _ (L.nonul _ h0)]

/-! ## Registry (`GRAPH_REGISTRY`: key → graph allocation) -/

abbrev Reg := List (List Char × Nat)

def Reg.insert (r : Reg) (k : List Char) (g : Nat) : Reg := (k, g) :: r.filter (·.1 ≠ k)
def Reg.get (r : Reg) (k : List Char) : Option Nat := (r.find? (·.1 = k)).map (·.2)

/-- `register_graph`; returns the displaced graph (dropped on another thread). -/
def register (r : Reg) (k : List Char) (g : Nat) : Reg × Option Nat := (r.insert k g, r.get k)
/-- `graph_is_registered`: by allocation, under any key. -/
def isRegistered (r : Reg) (g : Nat) : Bool := r.any (·.2 = g)
/-- `rename_graph`: `remove(old).and_then(|arc| insert(new, arc))`. -/
def rename (r : Reg) (old new : List Char) : Reg × Option Nat :=
  match r.get old with
  | none => (r, none)
  | some g => let r' := r.filter (·.1 ≠ old); (Reg.insert r' new g, Reg.get r' new)
/-- `graph_free`: remove every key mapping to this allocation. -/
def free (r : Reg) (g : Nat) : Reg := r.filter (·.2 ≠ g)

theorem register_registered (r : Reg) (k : List Char) (g : Nat) : isRegistered (register r k g).1 g := by
  simp [register, Reg.insert, isRegistered]

theorem register_get (r : Reg) (k : List Char) (g : Nat) : (register r k g).1.get k = some g := by
  simp [register, Reg.insert, Reg.get]

theorem free_unregisters (r : Reg) (g : Nat) : isRegistered (free r g) g = false := by
  simp [free, isRegistered]

theorem free_keeps_others (r : Reg) (g h : Nat) (hne : h ≠ g) :
    isRegistered (free r g) h = isRegistered r h := by
  simp only [free, isRegistered, List.any_filter]
  congr 1; funext p; by_cases hp : p.2 = h <;> simp [hp, hne]

theorem rename_absent (r : Reg) (o n : List Char) (h : r.get o = none) : rename r o n = (r, none) := by
  simp [rename, h]

theorem get_filter_self (l : Reg) (o : List Char) : Reg.get (l.filter (·.1 ≠ o)) o = none := by
  unfold Reg.get
  rw [List.find?_eq_none.2]; · rfl
  intro x hx; simp at hx ⊢; exact hx.2

theorem rename_moves (r : Reg) (o n : List Char) (g : Nat) (h : r.get o = some g) :
    (rename r o n).1.get n = some g ∧ ((o ≠ n) → (rename r o n).1.get o = none) := by
  unfold rename; rw [h]; simp only
  refine ⟨by simp [Reg.insert, Reg.get], fun hne => ?_⟩
  unfold Reg.insert
  have e : Reg.get ((n, g) :: List.filter (fun x => decide (x.1 ≠ n)) (List.filter (fun x => decide (x.1 ≠ o)) r)) o
      = Reg.get (List.filter (fun x => decide (x.1 ≠ o)) (List.filter (fun x => decide (x.1 ≠ n)) r)) o := by
    simp only [Reg.get, List.find?_cons]
    have : (decide ((n, g).1 = o)) = false := by simp [Ne.symm hne]
    simp only [this, List.filter_filter]
    congr 2; congr 1; funext p; simp [Bool.and_comm]
  rw [e, get_filter_self]

/-! ## Write bookkeeping -/

structure Stats where
  nodesCreated : Nat
  nodesDeleted : Nat
  relsCreated : Nat
  relsDeleted : Nat
  propsSet : Nat
  propsRemoved : Nat
  labelsAdded : Nat
  labelsRemoved : Nat
  idxCreated : Nat
  idxDropped : Nat

/-- `WriteQueryOk::new`'s `modified`. -/
def modified (s : Stats) (effects : Nat) : Bool :=
  s.nodesCreated > 0 || s.nodesDeleted > 0 || s.relsCreated > 0 || s.relsDeleted > 0
  || s.propsSet > 0 || s.propsRemoved > 0 || s.labelsAdded > 0 || s.labelsRemoved > 0
  || s.idxCreated > 0 || s.idxDropped > 0 || effects > 0

theorem modified_zero : modified ⟨0,0,0,0,0,0,0,0,0,0⟩ 0 = false := rfl

inductive Step | commit | signal | replicate | warnNoEffects | rollback | resync | release | reply
  deriving DecidableEq, Repr

/-- `commit_and_replicate`. -/
def commitSteps (mod : Bool) (hasBuf : Bool) : List Step :=
  [.commit, .signal] ++ (if !mod then [] else if hasBuf then [.replicate] else [.warnNoEffects])

/-- `commit_and_replicate` (graph_core.rs:1481) since #2846: `MvccGraph::commit` now
validates, and a refusal here is `unreachable!()` (:1497-1499) — the panic step. The
argument it rests on: every write query ends its last segment through
`Pending::end_segment`, which verifies the batch and rolls it, so the batch reaching
here is fresh (proofs/graph_queries `rollIdBatches_spec`: `entry_bound = bound`,
`taken = []`); that such a batch verifies is the IdSpace contract (proofs/id_space). -/
def commitStepsV (valid mod hasBuf : Bool) : Option (List Step) :=
  if valid then some (commitSteps mod hasBuf) else none   -- `none` = `unreachable!` panic

theorem commitStepsV_valid (mod buf : Bool) : commitStepsV true mod buf = some (commitSteps mod buf) := rfl
theorem commitStepsV_refused (mod buf : Bool) : commitStepsV false mod buf = none := rfl

/-- `execute_query_write` from `runtime.query()` on: abandon on error; else finish (commit,
or rollback if the session is not a writer), release locks, then reply. -/
def writeSteps (ok writer mod hasBuf : Bool) : List Step :=
  if !ok then [.resync, .rollback]
  else (if writer then commitSteps mod hasBuf else [.rollback]) ++ [.release, .reply]

/-- The client is answered only after the commit is published and the locks are gone. -/
theorem reply_after_commit (mod buf : Bool) :
    writeSteps true true mod buf = commitSteps mod buf ++ [.release, .reply] := by
  simp [writeSteps]

/-- Every modified commit either replicates or logs that it could not. -/
theorem modified_replicates_or_warns (buf : Bool) :
    .replicate ∈ commitSteps true buf ∨ .warnNoEffects ∈ commitSteps true buf := by
  cases buf <;> simp [commitSteps]

/-- Unmodified writes replicate nothing. -/
theorem unmodified_silent (buf : Bool) : commitSteps false buf = [.commit, .signal] := by
  simp [commitSteps]

/-- A failed write never commits and never replies here (the caller sends the error). -/
theorem failed_no_commit (w m b : Bool) : .commit ∉ writeSteps false w m b ∧ .reply ∉ writeSteps false w m b := by
  simp [writeSteps]

/-- `CtxSink::replicate`: `GRAPH.EFFECT key payload`. -/
def EFFECT_COMMAND := "GRAPH.EFFECT"
def replicateArgv (key payload : Bytes) : String × List Bytes := (EFFECT_COMMAND, [key, payload])
theorem replicateArgv_shape (k p : Bytes) : replicateArgv k p = ("GRAPH.EFFECT", [k, p]) := rfl

/-! ## Errors and small replies -/

/-- `sanitise_error`: NUL → space. -/
def sanitise (b : Bytes) : Bytes := b.map fun x => if x = 0 then 32 else x

theorem sanitise_nonul (b : Bytes) : 0 ∉ sanitise b := by
  simp [sanitise]; intro x _; split <;> omega
theorem sanitise_len (b : Bytes) : (sanitise b).length = b.length := by simp [sanitise]
theorem sanitise_id (b : Bytes) (h : 0 ∉ b) : sanitise b = b := by
  simp only [sanitise]
  conv => rhs; rw [← List.map_id b]
  apply List.map_congr_left; intro x hx; simp; intro e; subst e; exact absurd hx h

/-- `reply_invalid_graph_version`: `[error, current_version]`. -/
def invalidVersionBytes (v : Int) : String :=
  "*2\r\n" ++ "-ERR invalid graph version\r\n" ++ ":" ++ toString v ++ "\r\n"

theorem invalidVersion_shape : invalidVersionBytes 7 = "*2\r\n-ERR invalid graph version\r\n:7\r\n" := by decide

/-! ## `reply_profile` depth (`:823-857`) -/

/-- Depth printed for an op: tree depth minus the `Commit` ancestors. `anc` lists, for each
ancestor, whether it is a `Commit`. -/
def profileDepth (anc : List Bool) : Nat := anc.length - (anc.filter id).length

theorem profileDepth_eq (anc : List Bool) : profileDepth anc = (anc.filter (! ·)).length := by
  have : ∀ l : List Bool, (l.filter id).length + (l.filter (! ·)).length = l.length := by
    intro l; induction l with
    | nil => rfl
    | cons a as ih => cases a <;> simp_all <;> omega
  have := this anc
  unfold profileDepth; omega

/-- …and the `usize` subtraction cannot underflow. -/
theorem profileDepth_no_underflow (anc : List Bool) : (anc.filter id).length ≤ anc.length :=
  List.length_filter_le _ _

/-- Ops reported: all but `Commit`. -/
def profileOps {α} (isCommit : α → Bool) (ops : List α) : List α := ops.filter (! isCommit ·)

/-! ## Write detection (`execute_query` / `execute_profile`) -/

inductive Op | commit | createIndex | dropIndex | other
  deriving DecidableEq

def isWrite (plan : List Op) : Bool := plan.any fun o => o = .commit || o = .createIndex || o = .dropIndex

inductive Detect | readReplied | write
  deriving DecidableEq

/-- `execute_profile`: a write plan is handed to the write queue without running. -/
def profileDetect (plan : List Op) : Detect := if isWrite plan then .write else .readReplied

theorem profileDetect_write (plan : List Op) : profileDetect plan = .write ↔ isWrite plan = true := by
  unfold profileDetect; split <;> simp_all

/-- `execute_query`: RO on a write plan errors; QUERY on a write plan defers. -/
inductive QOut | roError | deferWrite | readRun
  deriving DecidableEq

def executeQuery (plan : List Op) (writeAllowed : Bool) : QOut :=
  if isWrite plan then (if !writeAllowed then .roError else .deferWrite) else .readRun

theorem ro_write_rejected (plan : List Op) (h : isWrite plan) : executeQuery plan false = .roError := by
  simp [executeQuery, h]

/-! ## `query_mut` dispatch (`:945-1150`) -/

inductive Disp | invalidVersion (cur : Nat) | inline | maxPending | spawn
  deriving DecidableEq

def dispatch (versionArg : Option Nat) (schemaVersion : Nat) (inline : Bool) (pending maxQ : Nat) : Disp :=
  match versionArg with
  | some v => if v ≠ schemaVersion then .invalidVersion schemaVersion
      else if inline then .inline else if pending ≥ maxQ then .maxPending else .spawn
  | none => if inline then .inline else if pending ≥ maxQ then .maxPending else .spawn

/-- A stale `version` never reaches the planner. -/
theorem stale_version (v s : Nat) (i : Bool) (p m : Nat) (h : v ≠ s) :
    dispatch (some v) s i p m = .invalidVersion s := by simp [dispatch, h]
/-- The queue cap does not apply to inline (MULTI/Lua/replay) execution. -/
theorem inline_uncapped (s p m : Nat) : dispatch none s true p m = .inline := rfl
theorem capped (s p m : Nat) (h : m ≤ p) : dispatch none s false p m = .maxPending := by
  simp [dispatch]; omega

/-! ## Sync paths (`query_sync`, `profile_sync`, `profile_mut`) -/

inductive Out | errNoTelemetry | readTelemetry | writeTelemetry | writeErr | profileRead
  deriving DecidableEq

/-- `query_sync`: read errors return without telemetry; reads log a read entry; writes run
under `begin_writer` and log a write entry with zero wait, or fail through the divergence
guard. -/
def syncOutcome (readOk isWrite writeOk : Bool) : Out :=
  if !readOk then .errNoTelemetry
  else if !isWrite then .readTelemetry
  else if writeOk then .writeTelemetry else .writeErr

theorem sync_write_logs : syncOutcome true true true = .writeTelemetry := rfl
theorem sync_err (w o : Bool) : syncOutcome false w o = .errNoTelemetry := rfl

/-- `profile_sync`/`profile_mut`: same detection, write plans run with `profile = true`. -/
def profileOutcome (detectOk : Bool) (d : Detect) (writeOk : Bool) : Out :=
  if !detectOk then .errNoTelemetry
  else match d with
    | .readReplied => .profileRead
    | .write => if writeOk then .writeTelemetry else .writeErr

theorem profile_write_runs_as_write (w : Bool) :
    profileOutcome true .write w = (if w then .writeTelemetry else .writeErr) := rfl
theorem profile_read_replied (w : Bool) : profileOutcome true .readReplied w = .profileRead := rfl

/-! ## `process_write_queued_query` exit protocol (`:1525-1569`) -/

/-- After `try_recv` finds nothing: release the flag, then return only if the queue is
still empty or another thread won the flag back; otherwise loop again as owner. -/
inductive Exit | ret | loop
  deriving DecidableEq

def loopExit (emptyAfterRelease : Bool) (casWon : Bool) : Exit :=
  if emptyAfterRelease then .ret else if casWon then .loop else .ret

/-- **No stranded message**: a consumer leaves only having seen the queue empty after
releasing the flag, or having lost the flag to a thread that will drain. -/
theorem loopExit_safe (e c : Bool) (h : loopExit e c = .ret) : e = true ∨ c = false := by
  cases e <;> cases c <;> simp_all [loopExit]

/-! ## `ThreadedGraph` -/

structure TG where
  capacity : Nat
  writeLoop : Bool

def TG.new : TG := ⟨1024, false⟩
def TG.fromMvcc : TG := ⟨1024, false⟩
theorem TG.new_idle : TG.new.writeLoop = false ∧ TG.fromMvcc = TG.new := ⟨rfl, rfl⟩

end RedisLayer.GraphCore
