import RedisLayer.Resp
/-!
# `src/graph_core.rs` — names, registry, write commit, dispatch decisions
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
    · simp only [upToNulB, hb, ite_false]
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

/-! ## Registry (`GRAPH_REGISTRY`: graph identity → where the graph is, #2782)

Keyed by `graph_id` (the address of the graph's `Arc` allocation, `g` here), like C's
`graphs_in_keyspace`: each entry (`RegisteredGraph`) is the logical database the graph's key
is in and the key's lossy name. A `HashMap`, so no id occurs twice (`Reg.Uniq`). -/

structure Ent where
  g : Nat
  db : Nat
  name : List Char
  deriving DecidableEq, Repr

abbrev Reg := List Ent

/-- `GRAPH_REGISTRY.get(&id)`. -/
def Reg.get (r : Reg) (g : Nat) : Option Ent := r.find? (·.g = g)
/-- The `HashMap` invariant: one entry per id. -/
def Reg.Uniq (r : Reg) : Prop := (r.map (·.g)).Nodup

/-- `register_graph_at`: `entry(graph_id).and_modify(db, name).or_insert(…)`. A graph that
is registered already only moves; nothing is displaced. -/
def registerAt (r : Reg) (db : Nat) (k : List Char) (g : Nat) : Reg :=
  if r.any (·.g = g) then r.map (fun e => if e.g = g then { e with db := db, name := k } else e)
  else ⟨g, db, k⟩ :: r
/-- `register_graph(ctx, name, arc)`: `register_graph_at(selected_db(ctx), …)`. -/
def register (r : Reg) (sel : Nat) (k : List Char) (g : Nat) : Reg := registerAt r sel k g
/-- `graph_is_registered`: `contains_key(&graph_id(arc))`. -/
def isRegistered (r : Reg) (g : Nat) : Bool := r.any (·.g = g)
/-- `graph_free`'s registry part: `remove(&graph_id(&boxed))`. -/
def free (r : Reg) (g : Nat) : Reg := r.filter (·.g ≠ g)
/-- `registered_location`. -/
def registeredLocation (r : Reg) (g : Nat) : Option (Nat × List Char) :=
  (r.get g).map fun e => (e.db, e.name)
/-- `name_is_shared`: some registered graph of `db` has this name, cut at the first NUL. -/
def nameIsShared (r : Reg) (db : Nat) (n : List Char) : Bool :=
  r.any fun e => e.db == db && upToNulC e.name == upToNulC n
/-- `relocate_graph`: move the entry of `g` to key `k` of `db`; where it was. -/
def relocate (r : Reg) (g db : Nat) (k : List Char) : Reg × Option (Nat × List Char) :=
  match r.get g with
  | none => (r, none)
  | some e => (r.map (fun e' => if e'.g = g then { e' with db := db, name := k } else e'),
      some (e.db, e.name))
/-- One database number under `SWAPDB a b`. -/
def swapDb (a b d : Nat) : Nat := if d = a then b else if d = b then a else d
/-- `swap_graph_dbs`. -/
def swapDbs (r : Reg) (a b : Nat) : Reg := r.map fun e => { e with db := swapDb a b e.db }

theorem isRegistered_iff_get (r : Reg) (g : Nat) : isRegistered r g = true ↔ (r.get g).isSome := by
  simp [isRegistered, Reg.get, List.find?_isSome]

theorem get_map_sameG (r : Reg) (f : Ent → Ent) (hf : ∀ e, (f e).g = e.g) (g : Nat) :
    Reg.get (r.map f) g = (r.get g).map f := by
  unfold Reg.get
  rw [List.find?_map]; congr 2; funext e; simp [hf]

theorem get_some (r : Reg) (g : Nat) (e : Ent) (h : r.get g = some e) : e ∈ r ∧ e.g = g := by
  unfold Reg.get at h
  exact ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩

theorem get_none_of (r : Reg) (g : Nat) (h : r.any (·.g = g) = false) : r.get g = none := by
  unfold Reg.get; rw [List.find?_eq_none]; intro x hx; simpa using List.any_eq_false.1 h x hx

theorem register_registered (r : Reg) (sel : Nat) (k : List Char) (g : Nat) :
    isRegistered (register r sel k g) g := by
  unfold register registerAt isRegistered; split
  · rename_i h; simp only [List.any_map]; simp at h ⊢
    obtain ⟨e, he, rfl⟩ := h; exact ⟨e, he, by simp⟩
  · simp

/-- The graph is found by its identity at the key and database it was registered at. -/
theorem register_get (r : Reg) (sel : Nat) (k : List Char) (g : Nat) :
    (register r sel k g).get g = some ⟨g, sel, k⟩ := by
  unfold register registerAt; split
  · rename_i h
    rw [get_map_sameG _ _ (fun e => by split <;> rfl)]
    have : (r.get g).isSome := (isRegistered_iff_get r g).1 h
    obtain ⟨e, he⟩ := Option.isSome_iff_exists.1 this
    rw [he]; have := (get_some r g e he).2; subst this; simp
  · simp [Reg.get]

/-- **No displacement (#2782 fixed)**: registering one graph leaves every other graph's entry
— its database and name included — as it was, whatever names the two have. -/
theorem register_keeps_others (r : Reg) (sel : Nat) (k : List Char) (g h : Nat) (hne : h ≠ g) :
    (register r sel k g).get h = r.get h := by
  unfold register registerAt; split
  · rw [get_map_sameG _ _ (fun e => by split <;> rfl)]
    cases he : r.get h with
    | none => rfl
    | some e => have := (get_some r h e he).2; simp [this, hne]
  · simp [Reg.get, Ne.symm hne]

theorem register_uniq (r : Reg) (sel : Nat) (k : List Char) (g : Nat) (hu : r.Uniq) :
    (register r sel k g).Uniq := by
  unfold register registerAt Reg.Uniq at *; split
  · rw [List.map_map]
    have : ((·.g) ∘ fun e : Ent => if e.g = g then { e with db := sel, name := k } else e) = (·.g) := by
      funext e; by_cases h : e.g = g <;> simp [h]
    rw [this]; exact hu
  · rename_i h; simp only [List.map_cons, List.nodup_cons]; refine ⟨?_, hu⟩
    simp at h ⊢; exact h

theorem isRegistered_register_mono (r : Reg) (sel : Nat) (k : List Char) (g h : Nat)
    (hr : isRegistered r h) : isRegistered (register r sel k g) h := by
  by_cases e : h = g
  · subst e; exact register_registered r sel k h
  · rw [isRegistered_iff_get] at hr ⊢; rw [register_keeps_others r sel k g h e]; exact hr

theorem free_unregisters (r : Reg) (g : Nat) : isRegistered (free r g) g = false := by
  simp [free, isRegistered]

theorem free_keeps_others (r : Reg) (g h : Nat) (hne : h ≠ g) :
    isRegistered (free r g) h = isRegistered r h := by
  simp only [free, isRegistered, List.any_filter]
  congr 1; funext p; by_cases hp : p.g = h <;> simp [hp, hne]

/-- `relocate_graph` of an unregistered graph changes nothing and answers `None`. -/
theorem relocate_absent (r : Reg) (g db : Nat) (k : List Char) (h : r.get g = none) :
    relocate r g db k = (r, none) := by
  simp [relocate, h]

/-- `relocate_graph` moves the entry and answers where it was. -/
theorem relocate_moves (r : Reg) (g db : Nat) (k : List Char) (e : Ent) (h : r.get g = some e) :
    (relocate r g db k).1.get g = some ⟨g, db, k⟩ ∧ (relocate r g db k).2 = some (e.db, e.name) := by
  unfold relocate; rw [h]
  refine ⟨?_, rfl⟩
  show Reg.get (List.map _ r) g = _
  rw [get_map_sameG _ _ (fun e => by split <;> rfl), h]
  have := (get_some r g e h).2; subst this; simp

theorem relocate_keeps_others (r : Reg) (g db : Nat) (k : List Char) (h : Nat) (hne : h ≠ g) :
    (relocate r g db k).1.get h = r.get h := by
  unfold relocate; split
  · rfl
  · simp only
    rw [get_map_sameG _ _ (fun e => by split <;> rfl)]
    cases he : r.get h with
    | none => rfl
    | some e => have := (get_some r h e he).2; simp [this, hne]

theorem swapDb_swap (a b : Nat) : swapDb a b a = b ∧ swapDb a b b = a := by
  unfold swapDb; by_cases h : b = a <;> simp [h]

theorem swapDb_invol (a b d : Nat) : swapDb a b (swapDb a b d) = d := by
  unfold swapDb; by_cases h1 : d = a <;> by_cases h2 : d = b <;> by_cases h3 : b = a <;> simp_all

/-- `SWAPDB a b`: every graph follows its database, nothing else changes. -/
theorem swapDbs_get (r : Reg) (a b g : Nat) :
    (swapDbs r a b).get g = (r.get g).map fun e => { e with db := swapDb a b e.db } := by
  unfold swapDbs; exact get_map_sameG r (fun e => { e with db := swapDb a b e.db }) (fun _ => rfl) g

/-! ## `key_of_write` (#3850/#3851): the key a write is published under

`ks k` is the graph allocation the keyspace holds at key `k` of the database the write's
context has selected (`key_holds_graph`, `redis_type.rs`, is `ks k = some g`: TelemetryLoop
`keyHolds_iff`). `main` is `is_main_thread()`. The registry lookup is
`GRAPH_REGISTRY.get(&graph_id(graph))` (#2782: by identity); `'\uFFFD' ∈ admitted` is
`admitted.contains(char::REPLACEMENT_CHARACTER)` (a lossily decoded, non-UTF-8 key). `none` is
`Err(WriteAbort::GraphUnregistered)`. -/
def keyOfWrite (main : Bool) (ks : List Char → Option Nat) (reg : Reg) (g : Nat)
    (admitted : List Char) : Option (List Char) :=
  if main || ks admitted == some g then some admitted
  else match reg.get g with
    | some e =>
      if ks e.name == some g then some e.name
      else if '\uFFFD' ∈ admitted then some admitted else none
    | none => none

/-- An inline (main-thread) write is published under the key the client named. -/
theorem keyOfWrite_main (ks : List Char → Option Nat) (reg : Reg) (g : Nat) (a : List Char) :
    keyOfWrite true ks reg g a = some a := by simp [keyOfWrite]

/-- The admitted key still holds the graph: that key. -/
theorem keyOfWrite_admitted (m : Bool) (ks : List Char → Option Nat) (reg : Reg) (g : Nat)
    (a : List Char) (h : ks a = some g) : keyOfWrite m ks reg g a = some a := by
  simp [keyOfWrite, h]

/-- **Renamed**: the admitted key no longer holds the graph, the registry's key does — the
write (and its `GRAPH.EFFECT`) goes to the key the graph lives at now. -/
theorem keyOfWrite_renamed (ks : List Char → Option Nat) (reg : Reg) (g : Nat) (a : List Char)
    (e : Ent) (ha : ks a ≠ some g) (hr : reg.get g = some e) (hn : ks e.name = some g) :
    keyOfWrite false ks reg g a = some e.name := by
  simp [keyOfWrite, ha, hr, hn]

/-- **Sound for UTF-8 keys**: a queued write is only ever published under a key that holds
its graph. -/
theorem keyOfWrite_holds (ks : List Char → Option Nat) (reg : Reg) (g : Nat) (a k : List Char)
    (hu : '\uFFFD' ∉ a) (h : keyOfWrite false ks reg g a = some k) : ks k = some g := by
  unfold keyOfWrite at h
  by_cases ha : ks a = some g
  · simp [ha] at h; exact h ▸ ha
  · simp only [Bool.false_or, ha, beq_iff_eq, ite_false] at h
    split at h
    · rename_i e _
      by_cases hn : ks e.name = some g
      · simp [hn] at h; exact h ▸ hn
      · simp [hn, hu] at h
    · simp at h

/-- **Refused**: no key holds the graph (deleted, flushed — even before the lazy-free thread
reaches `graph_free` — or moved to another database): `GraphUnregistered`, whatever the
registry still says. -/
theorem keyOfWrite_unheld (ks : List Char → Option Nat) (reg : Reg) (g : Nat) (a : List Char)
    (hu : '\uFFFD' ∉ a) (h : ∀ k, ks k ≠ some g) : keyOfWrite false ks reg g a = none := by
  cases e : keyOfWrite false ks reg g a with
  | none => rfl
  | some k => exact absurd (keyOfWrite_holds ks reg g a k hu e) (h k)

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

inductive Step | commit | signal | replicate | warnNoEffects | resync | release | reply
  deriving DecidableEq, Repr

/-- `commit_and_replicate`. -/
def commitSteps (mod : Bool) (hasBuf : Bool) : List Step :=
  [.commit, .signal] ++ (if !mod then [] else if hasBuf then [.replicate] else [.warnNoEffects])

/-- `commit_and_replicate` (graph_core.rs) since #2846: `MvccGraph::commit` validates
(and since #3657 refuses a stale fork too: `unreachable!` in `current_not_stale`, see
SearchConcurrency `Mvcc`), and a refusal here is `unreachable!()` — the panic step. The
argument it rests on: every write query ends its last segment through
`Pending::end_segment`, which verifies the batch and rolls it, so the batch reaching
here is fresh (proofs/graph_queries `rollIdBatches_spec`: `entry_bound = bound`,
`taken = []`); that such a batch verifies is the IdSpace contract (proofs/id_space). -/
def commitStepsV (valid mod hasBuf : Bool) : Option (List Step) :=
  if valid then some (commitSteps mod hasBuf) else none   -- `none` = `unreachable!` panic

theorem commitStepsV_valid (mod buf : Bool) : commitStepsV true mod buf = some (commitSteps mod buf) := rfl
theorem commitStepsV_refused (mod buf : Bool) : commitStepsV false mod buf = none := rfl

/-- `abandon_write` since #3657: resync the index only if the session's version is
current (`version_is_current`: a version still matching, stale or refused at escalation
published no document), and nothing else: the version is dropped with the runtime, so there
is no slot to release and nothing to roll back. -/
def abandonSteps (cur : Bool) : List Step := if cur then [.resync] else []

/-- `try_query_write` from `runtime.query()` on: abandon on error; else `finish_write`
(`commit_and_replicate` if the session escalated; otherwise nothing, the version drops with
the runtime), release locks, then reply. Since #3850/#3851 an escalated `finish_write` first
asks `key_of_write` (`keyed` = `keyOfWrite … ≠ none`); refused, nothing is published and the
caller abandons the write (`abandon_write`) and returns `WriteAttempt::Failed`. A session still
in reader mode skips the check (`with_graph_mut` … `.unwrap_or(Ok(()))`). `record_mut` takes
the same two paths. -/
def writeSteps (ok writer keyed cur mod hasBuf : Bool) : List Step :=
  if !ok then abandonSteps cur
  else if writer && !keyed then abandonSteps cur
  else (if writer then commitSteps mod hasBuf else []) ++ [.release, .reply]

/-- The client is answered only after the commit is published and the locks are gone. -/
theorem reply_after_commit (h mod buf : Bool) :
    writeSteps true true true h mod buf = commitSteps mod buf ++ [.release, .reply] := by
  simp [writeSteps]

/-- The same, for every input: an escalated write that replies has committed first. -/
theorem reply_only_after_commit (ok k h mod buf : Bool)
    (hr : .reply ∈ writeSteps ok true k h mod buf) :
    writeSteps ok true k h mod buf = commitSteps mod buf ++ [.release, .reply] := by
  cases ok <;> cases k <;> cases h <;> simp_all [writeSteps, abandonSteps]

/-- A failed write never commits and never replies here (the caller sends the error, or
re-runs a stale one); it resyncs the index only for a current version. -/
theorem failed_no_commit (w k h m b : Bool) :
    .commit ∉ writeSteps false w k h m b ∧ .reply ∉ writeSteps false w k h m b ∧
    (.resync ∈ writeSteps false w k h m b ↔ h = true) := by
  cases h <;> simp [writeSteps, abandonSteps]

/-- **#3850/#3851**: an escalated write whose graph no key holds is undone like a failed
one: no commit, no `GRAPH.EFFECT`, no reply here (the caller sends `GraphUnregistered`). -/
theorem unkeyed_refused (h m b : Bool) :
    writeSteps true true false h m b = abandonSteps h ∧
    .commit ∉ writeSteps true true false h m b ∧ .replicate ∉ writeSteps true true false h m b ∧
    .reply ∉ writeSteps true true false h m b := by
  cases h <;> simp [writeSteps, abandonSteps]

/-- A write that never escalated (`LIMIT 0` short-circuits above `Commit`) publishes and
replicates nothing: the session answers and the version is dropped. -/
theorem unescalated_silent (k h m b : Bool) :
    writeSteps true false k h m b = [.release, .reply] := by simp [writeSteps]

/-! ### `execute_query_write`'s retry (#3290)

`try_query_write` returns `WriteAttempt::Stale` only from the error arm after
`abandon_write` when `session.is_stale()` — nothing mutated, published or replied — and the
loop re-runs on a fresh session (`QuerySession::begin_with`, or `begin_writer_checked` once the
bounded retry below escalates, #3559). -/

inductive Attempt | stale | failed (e : String) | ok
  deriving DecidableEq, Repr

/-- `impl From<String> for WriteAttempt`: every other error is `Failed`. -/
def Attempt.ofString (e : String) : Attempt := .failed e
theorem ofString_not_stale (e : String) : Attempt.ofString e ≠ .stale := by simp [Attempt.ofString]

/-- The loop over the outcomes of successive runs (`none`: still running). -/
def executeWrite : List Attempt → Option Attempt
  | [] => none
  | .stale :: rest => executeWrite rest
  | a :: _ => some a

/-- **A stale run is invisible**: the result is the first run that was not stale. -/
theorem executeWrite_first (k : Nat) (a : Attempt) (rest : List Attempt) (h : a ≠ .stale) :
    executeWrite (List.replicate k .stale ++ a :: rest) = some a := by
  induction k with
  | zero => cases a <;> simp_all [executeWrite]
  | succ k ih => simpa [List.replicate_succ, executeWrite] using ih

/-- A stale run took no write step a client could see: it is not `current`, so it resyncs
nothing, and it committed and replied nothing. -/
theorem stale_run_silent (w k m b : Bool) :
    writeSteps false w k false m b = [] := by simp [writeSteps, abandonSteps]

/-! ### The bounded retry (#3559)

`execute_query_write` counts stale runs; after `MAX_STALE_RETRIES` of them, or
at once after a stale run that took `SLOW_STALE_RUN` (10 ms) or longer, the next run
begins as a writer (`QuerySession::begin_writer_checked`) instead of a reader
(`begin_with`). A run is a function of the mode it begins in; a refused
`begin_writer_checked` is that run's `failed` outcome (`map_err(…)?`). Times are nanoseconds. -/

def maxStaleRetries : Nat := 3
def slowStaleRun : Nat := 10000000

/-- `stale_runs >= MAX_STALE_RETRIES || wasted >= SLOW_STALE_RUN`, after the increment. -/
def escalateNext (staleRuns wasted : Nat) : Bool :=
  decide (maxStaleRetries ≤ staleRuns) || decide (slowStaleRun ≤ wasted)

/-- One run's report: its outcome and how long it took. -/
structure Run where
  a : Attempt
  wasted : Nat

/-- The loop: `s` stale runs so far, `w` whether this run begins as a writer; `runs` the
successive runs, each given the mode it begins in. Returns the result (`none`: still
running) and the mode of every run taken. -/
def loopW (s : Nat) (w : Bool) : List (Bool → Run) → Option Attempt × List Bool
  | [] => (none, [])
  | f :: rest =>
    match (f w).a with
    | .stale =>
      let p := loopW (s + 1) (escalateNext (s + 1) (f w).wasted) rest
      (p.1, w :: p.2)
    | a => (some a, [w])

theorem escalate_after_max (n t : Nat) (h : maxStaleRetries ≤ n) : escalateNext n t = true := by
  simp [escalateNext, h]

theorem reader_before_max (n t : Nat) (h : n < maxStaleRetries) (ht : t < slowStaleRun) :
    escalateNext n t = false := by
  simp [escalateNext]; omega

/-- **Still the first run that was not stale** (as `executeWrite_first`), each run taken in the
mode the loop chose for it. -/
theorem loopW_first (s : Nat) (w : Bool) (runs : List (Bool → Run)) :
    (loopW s w runs).1 =
      executeWrite (List.zipWith (fun f m => (f m).a) runs (loopW s w runs).2) := by
  induction runs generalizing s w with
  | nil => simp [loopW, executeWrite]
  | cons f rest ih =>
    cases ha : (f w).a <;> simp [loopW, ha, executeWrite, ih]

/-- A stale run is never the result. -/
theorem loopW_not_stale (s : Nat) (w : Bool) (runs : List (Bool → Run)) :
    (loopW s w runs).1 ≠ some .stale := by
  induction runs generalizing s w with
  | nil => simp [loopW]
  | cons f rest ih =>
    cases ha : (f w).a <;> simp [loopW, ha, ih]

/-- **Bounded**: at most `MAX_STALE_RETRIES` runs begin as readers, whatever the outcomes. -/
theorem loopW_readers (s : Nat) (w : Bool) (runs : List (Bool → Run)) :
    ((loopW s w runs).2.filter (· = false)).length ≤
      (if w then 0 else 1) + (maxStaleRetries - 1 - s) := by
  induction runs generalizing s w with
  | nil => simp [loopW]
  | cons f rest ih =>
    cases ha : (f w).a
    case stale =>
      have h1 := ih (s + 1) (escalateNext (s + 1) (f w).wasted)
      simp only [loopW, ha]
      cases he : escalateNext (s + 1) (f w).wasted
      · have hlt : s + 1 < maxStaleRetries := by
          refine Nat.lt_of_not_le fun hc => ?_
          rw [escalate_after_max _ _ hc] at he; cases he
        rw [he] at h1
        simp only [maxStaleRetries] at hlt h1 ⊢
        cases w <;> simp at h1 ⊢ <;> omega
      · rw [he] at h1
        simp only [maxStaleRetries] at h1 ⊢
        cases w <;> simp at h1 ⊢ <;> omega
    all_goals cases w <;> simp [loopW, ha]

/-- The number of runs the loop takes when writer runs are never stale. -/
theorem loopW_length (s : Nat) (w : Bool) (runs : List (Bool → Run))
    (hw : ∀ f ∈ runs, (f true).a ≠ .stale) :
    (loopW s w runs).2.length ≤ (if w then 1 else max 2 (maxStaleRetries + 1 - s)) := by
  induction runs generalizing s w with
  | nil => cases w <;> simp [loopW]
  | cons f rest ih =>
    have hrest : ∀ g ∈ rest, (g true).a ≠ .stale := fun g hg => hw g (by simp [hg])
    cases ha : (f w).a
    case stale =>
      cases w
      · have h1 := ih (s + 1) (escalateNext (s + 1) (f false).wasted) hrest
        simp only [loopW, ha, List.length_cons]
        cases he : escalateNext (s + 1) (f false).wasted
        · have hlt : s + 1 < maxStaleRetries := by
            refine Nat.lt_of_not_le fun hc => ?_
            rw [escalate_after_max _ _ hc] at he; cases he
          rw [he] at h1
          simp only [maxStaleRetries] at hlt h1 ⊢
          simp at h1 ⊢; omega
        · rw [he] at h1; simp at h1 ⊢; omega
      · exact absurd ha (hw f (by simp))
    all_goals cases w <;> simp [loopW, ha] <;> omega

/-- Writer runs are never stale (`QuerySession.writer_never_stale`), so the loop takes at most
`MAX_STALE_RETRIES + 1` runs from a reader start, at most `MAX_STALE_RETRIES` of them readers. -/
theorem loopW_bounded (runs : List (Bool → Run)) (hw : ∀ f ∈ runs, (f true).a ≠ .stale) :
    (loopW 0 false runs).2.length ≤ maxStaleRetries + 1 ∧
    ((loopW 0 false runs).2.filter (· = false)).length ≤ maxStaleRetries := by
  refine ⟨?_, ?_⟩
  · have := loopW_length 0 false runs hw; simp [maxStaleRetries] at this ⊢; omega
  · have := loopW_readers 0 false runs; simpa [maxStaleRetries] using this

/-- The loop has its result once it has as many runs as `loopW_length` bounds. -/
theorem loopW_has_result (s : Nat) (w : Bool) (runs : List (Bool → Run))
    (hw : ∀ f ∈ runs, (f true).a ≠ .stale)
    (hl : (if w then 1 else max 2 (maxStaleRetries + 1 - s)) ≤ runs.length) :
    (loopW s w runs).1.isSome := by
  induction runs generalizing s w with
  | nil => simp at hl; split at hl <;> omega
  | cons f rest ih =>
    have hrest : ∀ g ∈ rest, (g true).a ≠ .stale := fun g hg => hw g (by simp [hg])
    cases ha : (f w).a
    case stale =>
      cases w
      · simp only [loopW, ha]
        apply ih _ _ hrest
        simp only [List.length_cons, Bool.false_eq_true, ite_false] at hl
        cases he : escalateNext (s + 1) (f false).wasted
        · have hlt : s + 1 < maxStaleRetries := by
            refine Nat.lt_of_not_le fun hc => ?_
            rw [escalate_after_max _ _ hc] at he; cases he
          simp only [maxStaleRetries] at hlt hl ⊢
          simp; omega
        · simp only [maxStaleRetries] at hl ⊢; simp; omega
      · exact absurd ha (hw f (by simp))
    all_goals simp [loopW, ha]

/-- **Decided**: given `MAX_STALE_RETRIES + 1` runs, the loop has its result. -/
theorem loopW_decided (runs : List (Bool → Run)) (hw : ∀ f ∈ runs, (f true).a ≠ .stale)
    (hl : maxStaleRetries + 1 ≤ runs.length) : (loopW 0 false runs).1.isSome :=
  loopW_has_result 0 false runs hw (by simp [maxStaleRetries] at hl ⊢; omega)

/-! ### One timeout for every run (#3559)

`try_query_write` resolves `WriteTimeout` on the first run only
(`compute_effective_timeout`, here its answer `cfg`) and keeps it for the retries; each run then
gets what is left since `started` (`spent` ms), or fails with `Query timed out` before forking. -/

inductive WTimeout | unresolved | unlimited | total (ms : Nat)
  deriving DecidableEq, Repr

/-- `if matches!(timeout, Unresolved) { *timeout = cfg.map_or(Unlimited, Total) }`. -/
def resolveT (t : WTimeout) (cfg : Option Nat) : WTimeout :=
  match t with
  | .unresolved => match cfg with | some n => .total n | none => .unlimited
  | t => t

/-- The run's `timeout_ms`: `.inl ()` is the `Query timed out` failure, `.inr` the budget. -/
def runBudget (t : WTimeout) (spent : Nat) : Unit ⊕ Option Nat :=
  match t with
  | .total total => if total ≤ spent then .inl () else .inr (some (total - spent))
  | _ => .inr none

/-- Once resolved, a config change during a retry cannot move the timeout. -/
theorem resolveT_kept (t : WTimeout) (cfg cfg' : Option Nat) :
    resolveT (resolveT t cfg) cfg' = resolveT t cfg := by
  cases t <;> try rfl
  cases cfg <;> rfl

/-- A run never gets more than the whole budget less what earlier runs spent, and is refused
exactly when the budget is gone. -/
theorem runBudget_le (total spent : Nat) :
    (runBudget (.total total) spent = .inl () ↔ total ≤ spent) ∧
    ∀ b, runBudget (.total total) spent = .inr (some b) → b + spent = total := by
  unfold runBudget
  constructor
  · by_cases h : total ≤ spent <;> simp [h]
  · intro b hb
    by_cases h : total ≤ spent <;> simp [h] at hb; omega

/-- With no timeout configured, no retry is ever refused for time. -/
theorem runBudget_unlimited (spent : Nat) : runBudget .unlimited spent = .inr none := rfl

/-- The refusal `try_query_write` returns (`WriteAttempt::Failed(abort.to_string())`) is not
`Stale`: `execute_query_write` surfaces it instead of re-running. -/
def unregisteredAttempt : Attempt :=
  .failed "graph was deleted or replaced while the query was running, aborting"
theorem unregistered_not_stale : unregisteredAttempt ≠ .stale := by simp [unregisteredAttempt]

/-- Every modified commit either replicates or logs that it could not. -/
theorem modified_replicates_or_warns (buf : Bool) :
    .replicate ∈ commitSteps true buf ∨ .warnNoEffects ∈ commitSteps true buf := by
  cases buf <;> simp [commitSteps]

/-- Unmodified writes replicate nothing. -/
theorem unmodified_silent (buf : Bool) : commitSteps false buf = [.commit, .signal] := by
  simp [commitSteps]

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

/-! ## `reply_profile`

Since #2390 (`d2c42e032`) every plan node is listed, `Commit` included, at its plain
tree depth — the same rows `GRAPH.EXPLAIN` prints. -/

/-- Depth printed for an op: `node.depth()`, the number of ancestors. `anc` lists, for
each ancestor, whether it is a `Commit` (no longer consulted). -/
def profileDepth (anc : List Bool) : Nat := anc.length

/-- The printed depth is the tree depth: a `Commit` ancestor counts like any other. -/
theorem profileDepth_eq (anc : List Bool) : profileDepth anc = anc.length := rfl

theorem profileDepth_commit_counts (anc : List Bool) :
    profileDepth (true :: anc) = profileDepth anc + 1 := by
  simp [profileDepth]

/-- Ops reported: `plan.root().indices::<Dfs>()`, every node. -/
def profileOps {α} (_isCommit : α → Bool) (ops : List α) : List α := ops

/-- PROFILE reports exactly the DFS node list (Commit included); the reply array
length `ops.len()` equals the plan size. -/
theorem profileOps_all {α} (isCommit : α → Bool) (ops : List α) :
    profileOps isCommit ops = ops ∧ (profileOps isCommit ops).length = ops.length :=
  ⟨rfl, rfl⟩

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

/-! ## `admit` and `query_mut` dispatch (#3014) -/

/-- `admit`: `QueueSlot::try_acquire(MAX_QUEUED_QUERIES)` (the count after admission), or
the reply "Max pending queries exceeded" (`none`). Never waits. -/
def admitQ (count maxQ : Nat) : Option Nat := if count < maxQ then some (count + 1) else none

theorem admit_bounded (c m n : Nat) (h : admitQ c m = some n) : n ≤ m := by
  unfold admitQ at h; split at h <;> simp at h; omega

inductive Disp | invalidVersion (cur : Nat) | inline | maxPending | spawn
  deriving DecidableEq

def dispatch (versionArg : Option Nat) (schemaVersion : Nat) (inline : Bool) (count maxQ : Nat) : Disp :=
  match versionArg with
  | some v => if v ≠ schemaVersion then .invalidVersion schemaVersion
      else if inline then .inline else if (admitQ count maxQ).isNone then .maxPending else .spawn
  | none => if inline then .inline else if (admitQ count maxQ).isNone then .maxPending else .spawn

/-- A stale `version` never reaches the planner. -/
theorem stale_version (v s : Nat) (i : Bool) (p m : Nat) (h : v ≠ s) :
    dispatch (some v) s i p m = .invalidVersion s := by simp [dispatch, h]
/-- The queue cap does not apply to inline (MULTI/Lua/replay) execution. -/
theorem inline_uncapped (s p m : Nat) : dispatch none s true p m = .inline := rfl
/-- With `MAX_QUEUED_QUERIES` queries queued (pool + every write queue), a new one is
refused at once. -/
theorem capped (s p m : Nat) (h : m ≤ p) : dispatch none s false p m = .maxPending := by
  simp [dispatch, admitQ]; omega

/-- Which admission slot a job still holds after `execute_query`/`execute_profile`
(`slot.filter(..)`): only a write keeps it, to carry into the write queue. -/
def admissionKept (ok isWrite : Bool) : Bool := ok && isWrite
theorem read_releases_admission (ok : Bool) : admissionKept ok false = false := by simp [admissionKept]

/-! ## Sync paths (`query_sync`, `profile_sync`, `profile_mut`) -/

inductive Out | errNoTelemetry | readTelemetry | writeTelemetry | writeErr | profileRead
  | readUnlogged | writeUnlogged
  deriving DecidableEq

/-- `dispatch::logs_telemetry` (`dispatch.rs`, #3844): a query is this node's to log
unless it came from the master (`REPLICATED`) or from AOF replay (`LOADING`). -/
def logsTelemetry (replicated loading : Bool) : Bool := !(replicated || loading)

theorem logsTelemetry_iff (r l : Bool) : logsTelemetry r l = true ↔ r = false ∧ l = false := by
  cases r <;> cases l <;> decide

/-- `query_sync`: read errors return without telemetry; reads log a read entry; writes run
under `begin_writer` and log a write entry with zero wait, or fail through the divergence
guard. Since #3844 an entry is enqueued only if `log_query` (`logsTelemetry` of the
context); the reply is the same either way. -/
def syncOutcome (readOk isWrite writeOk logQuery : Bool) : Out :=
  if !readOk then .errNoTelemetry
  else if !isWrite then (if logQuery then .readTelemetry else .readUnlogged)
  else if writeOk then (if logQuery then .writeTelemetry else .writeUnlogged) else .writeErr

theorem sync_write_logs : syncOutcome true true true true = .writeTelemetry := rfl
theorem sync_read_logs : syncOutcome true false true true = .readTelemetry := rfl
theorem sync_err (w o l : Bool) : syncOutcome false w o l = .errNoTelemetry := rfl

/-- #3844: a replicated or replayed query is never logged on this node. -/
theorem sync_unserved_silent (rd w o r l : Bool) (h : r = true ∨ l = true) :
    syncOutcome rd w o (logsTelemetry r l) ≠ .readTelemetry ∧
    syncOutcome rd w o (logsTelemetry r l) ≠ .writeTelemetry := by
  cases r <;> cases l <;> simp at h <;> cases rd <;> cases w <;> cases o <;> decide

/-- …and the decision does not change what is executed: same outcome up to logging. -/
theorem sync_log_only (rd w o : Bool) :
    (syncOutcome rd w o true = .readTelemetry ↔ syncOutcome rd w o false = .readUnlogged) ∧
    (syncOutcome rd w o true = .writeTelemetry ↔ syncOutcome rd w o false = .writeUnlogged) := by
  cases rd <;> cases w <;> cases o <;> decide

/-- `profile_sync`/`profile_mut`: same detection, write plans run with `profile = true`. -/
def profileOutcome (detectOk : Bool) (d : Detect) (writeOk : Bool) : Out :=
  if !detectOk then .errNoTelemetry
  else match d with
    | .readReplied => .profileRead
    | .write => if writeOk then .writeTelemetry else .writeErr

theorem profile_write_runs_as_write (w : Bool) :
    profileOutcome true .write w = (if w then .writeTelemetry else .writeErr) := rfl
theorem profile_read_replied (w : Bool) : profileOutcome true .readReplied w = .profileRead := rfl

/-! ## `process_write_queued_query` exit protocol -/

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

/-- `capacity = none`: the write channel is unbounded since #3014 (bounded by admission). -/
structure TG where
  capacity : Option Nat
  writeLoop : Bool

def TG.new : TG := ⟨none, false⟩
def TG.fromMvcc : TG := ⟨none, false⟩
theorem TG.new_idle : TG.new.writeLoop = false ∧ TG.fromMvcc = TG.new ∧ TG.new.capacity = none := ⟨rfl, rfl, rfl⟩

end RedisLayer.GraphCore
