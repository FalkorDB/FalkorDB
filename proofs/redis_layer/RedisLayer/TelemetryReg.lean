import RedisLayer.Telemetry
/-!
# Telemetry: registry, flusher batching, re-keying and grouping (`src/telemetry.rs`)

`REGISTRY` is 8 mutex-guarded shards; each operation locks one shard, so a sequential model
per operation is exact (snapshots lock shard by shard and are not atomic across shards —
modelled as the concatenation, which is what a quiescent registry returns). Query ids come
from `NEXT_QUERY_ID.fetch_add(1)` and are therefore unique (`Fresh`).
-/
namespace RedisLayer.Telemetry

def REGISTRY_SHARDS := 8

/-- `shard_for`: `id & (REGISTRY_SHARDS - 1)`. -/
def shardFor (id : Nat) : Nat := id % REGISTRY_SHARDS

theorem shardFor_lt (id : Nat) : shardFor id < REGISTRY_SHARDS := Nat.mod_lt _ (by decide)

structure Q where
  id : Nat
  graph : List Char
  query : List Char
  deriving DecidableEq, Repr

structure Shard where
  running : List Q
  waiting : List Q
  deriving Repr

abbrev Reg := Nat → Shard

def Reg.empty : Reg := fun _ => ⟨[], []⟩

def Reg.upd (r : Reg) (k : Nat) (f : Shard → Shard) : Reg := fun j => if j = k then f (r j) else r j

/-- `Vec::swap_remove` of the first element with this id (order is not observable). -/
def removeId (id : Nat) : List Q → List Q
  | [] => []
  | q :: qs => if q.id = id then qs else q :: removeId id qs

def registerRunning (r : Reg) (q : Q) : Reg :=
  r.upd (shardFor q.id) fun s => { s with running := s.running ++ [{ q with query := truncateArc q.query }] }
def registerWaiting (r : Reg) (q : Q) : Reg :=
  r.upd (shardFor q.id) fun s => { s with waiting := s.waiting ++ [{ q with query := truncateArc q.query }] }
def unregisterRunning (r : Reg) (id : Nat) : Reg :=
  r.upd (shardFor id) fun s => { s with running := removeId id s.running }
def unregisterWaiting (r : Reg) (id : Nat) : Reg :=
  r.upd (shardFor id) fun s => { s with waiting := removeId id s.waiting }

/-- `transition_waiting_to_running`: same id, so same shard, one lock. -/
def transition (r : Reg) (id : Nat) : Option Nat × Reg :=
  match (r (shardFor id)).waiting.find? (·.id = id) with
  | none => (none, r)
  | some q => (some id, r.upd (shardFor id) fun s =>
      ⟨s.running ++ [q], removeId id s.waiting⟩)

def shards : List Nat := List.range REGISTRY_SHARDS
def snapshotRunning (r : Reg) : List Q := (shards.map fun k => (r k).running).flatten
def snapshotWaiting (r : Reg) : List Q := (shards.map fun k => (r k).waiting).flatten

/-- `try_snapshot_running`: shards whose lock is held are skipped (crash report must not
block). -/
def trySnapshotRunning (r : Reg) (locked : Nat → Bool) : List Q :=
  (shards.map fun k => if locked k then [] else (r k).running).flatten

theorem trySnapshot_unlocked (r : Reg) : trySnapshotRunning r (fun _ => false) = snapshotRunning r := by
  simp [trySnapshotRunning, snapshotRunning]

theorem trySnapshot_sub (r : Reg) (l : Nat → Bool) : ∀ q ∈ trySnapshotRunning r l, q ∈ snapshotRunning r := by
  intro q hq
  simp only [trySnapshotRunning, snapshotRunning, List.mem_flatten, List.mem_map] at hq ⊢
  obtain ⟨xs, ⟨k, hk, rfl⟩, hq⟩ := hq
  refine ⟨(r k).running, ⟨k, hk, rfl⟩, ?_⟩
  split at hq; · simp at hq
  · exact hq

/-- Every stored entry sits in the shard of its id. -/
def Placed (r : Reg) : Prop :=
  ∀ k, (∀ q ∈ (r k).running, shardFor q.id = k) ∧ (∀ q ∈ (r k).waiting, shardFor q.id = k)

theorem removeId_sub (id : Nat) (l : List Q) : ∀ q ∈ removeId id l, q ∈ l := by
  induction l with
  | nil => simp [removeId]
  | cons x xs ih =>
    intro q hq; simp only [removeId] at hq; split at hq
    · exact List.mem_cons_of_mem _ hq
    · simp at hq; rcases hq with rfl | h; simp; exact List.mem_cons_of_mem _ (ih q h)

theorem upd_placed (r : Reg) (k : Nat) (f : Shard → Shard) (hp : Placed r)
    (hf : (∀ q ∈ (f (r k)).running, shardFor q.id = k) ∧ (∀ q ∈ (f (r k)).waiting, shardFor q.id = k)) :
    Placed (r.upd k f) := by
  intro j; unfold Reg.upd; split
  · subst j; exact hf
  · exact hp j

theorem registerRunning_placed (r : Reg) (q : Q) (hp : Placed r) : Placed (registerRunning r q) := by
  apply upd_placed _ _ _ hp
  refine ⟨fun x hx => ?_, fun x hx => (hp _).2 x hx⟩
  simp at hx; rcases hx with h | rfl
  · exact (hp _).1 x h
  · rfl

theorem transition_placed (r : Reg) (id : Nat) (hp : Placed r) : Placed (transition r id).2 := by
  unfold transition
  split
  · exact hp
  · rename_i q hq
    apply upd_placed _ _ _ hp
    have hqid := List.find?_some hq
    simp at hqid
    refine ⟨fun x hx => ?_, fun x hx => (hp _).2 x (removeId_sub _ _ x hx)⟩
    simp at hx; rcases hx with h | rfl
    · exact (hp _).1 x h
    · rw [hqid]

theorem registerWaiting_placed (r : Reg) (q : Q) (hp : Placed r) : Placed (registerWaiting r q) := by
  apply upd_placed _ _ _ hp
  refine ⟨fun x hx => (hp _).1 x hx, fun x hx => ?_⟩
  simp at hx; rcases hx with h | rfl
  · exact (hp _).2 x h
  · rfl

/-- `transition` keeps the id and moves the entry from waiting to running in its shard. -/
theorem transition_moves (r : Reg) (id : Nat) (q : Q)
    (hq : (r (shardFor id)).waiting.find? (·.id = id) = some q) :
    (transition r id).1 = some id ∧ q ∈ ((transition r id).2 (shardFor id)).running := by
  simp [transition, hq, Reg.upd]

/-- An id that was never registered as waiting does not transition. -/
theorem transition_absent (r : Reg) (id : Nat)
    (h : (r (shardFor id)).waiting.find? (·.id = id) = none) : transition r id = (none, r) := by
  simp [transition, h]

/-- Unregistering an id that is not there is a no-op. -/
theorem removeId_absent (id : Nat) (l : List Q) (h : ∀ q ∈ l, q.id ≠ id) : removeId id l = l := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    have hx := h x (by simp)
    simp [removeId, hx, ih (fun q hq => h q (by simp [hq]))]

/-! ## `WaitingEntry` (`:477-502`) -/

/-- The guard: `some id` while armed. -/
structure WE where
  slot : Option Nat

/-- `promote`: take the id and transition it (`self.0.take().and_then(...)`). -/
def WE.promote (w : WE) (r : Reg) : Option Nat × WE × Reg :=
  match w.slot with
  | none => (none, ⟨none⟩, r)
  | some id => let t := transition r id; (t.1, ⟨none⟩, t.2)

/-- `Drop`: unregister if still armed. -/
def WE.drop (w : WE) (r : Reg) : Reg :=
  match w.slot with
  | none => r
  | some id => unregisterWaiting r id

/-- Once promoted, dropping the guard touches nothing. -/
theorem WE.promote_then_drop (w : WE) (r : Reg) :
    (w.promote r).2.1.drop (w.promote r).2.2 = (w.promote r).2.2 := by
  unfold WE.promote; split <;> rfl

/-- A guard dropped without promotion unregisters its waiting entry. -/
theorem WE.drop_armed (id : Nat) (r : Reg) : (WE.mk (some id)).drop r = unregisterWaiting r id := rfl

/-- Promoting twice transitions once. -/
theorem WE.promote_twice (w : WE) (r : Reg) :
    ((w.promote r).2.1.promote (w.promote r).2.2).1 = none := by
  cases h : w.slot <;> simp [WE.promote, h]

/-! ## Producer gate (`enqueue_entry`, `:892-937`) -/

def QUEUE_MAX := 4 * 256

/-- Whether an entry is sent: `CMD_INFO` on, not a replica, queue below the cap, flusher
running. -/
def enqueueOk (cmdInfo isReplica : Bool) (queued : Nat) (senderUp : Bool) : Bool :=
  cmdInfo && !isReplica && decide (queued < QUEUE_MAX) && senderUp

theorem enqueue_off (r q : Bool) (n : Nat) : enqueueOk false r n q = false := by simp [enqueueOk]
theorem enqueue_replica (c q : Bool) (n : Nat) : enqueueOk c true n q = false := by simp [enqueueOk]
theorem enqueue_full (c r q : Bool) (n : Nat) (h : QUEUE_MAX ≤ n) : enqueueOk c r n q = false := by
  simp [enqueueOk]; intros; omega

/-! ## Flusher batching -/

def FLUSH_BATCH_MAX := 256
def DEFERRED_XADD_MAX := 4 * FLUSH_BATCH_MAX

/-- `drain_queued`: move queued entries into `batch` until it holds `FLUSH_BATCH_MAX`;
`true` iff the channel reported `Disconnected` (empty and closed). -/
def drain {α} (batch queue : List α) (closed : Bool) : List α × List α × Bool :=
  let k := FLUSH_BATCH_MAX - batch.length
  if k ≤ queue.length then
    (batch ++ queue.take k, queue.drop k, false)
  else (batch ++ queue, [], closed)

theorem drain_bound {α} (b q : List α) (c : Bool) (h : b.length ≤ FLUSH_BATCH_MAX) :
    (drain b q c).1.length ≤ FLUSH_BATCH_MAX := by
  simp only [drain]; split <;> simp <;> omega

/-- Nothing is lost or reordered: batch' ++ queue' = batch ++ queue. -/
theorem drain_conserves {α} (b q : List α) (c : Bool) :
    (drain b q c).1 ++ (drain b q c).2.1 = b ++ q := by
  simp only [drain]; split <;> simp

/-- `deferred.drain(..len - DEFERRED_XADD_MAX)`: keep the newest. -/
def capDeferred {α} (d : List α) : List α :=
  if d.length > DEFERRED_XADD_MAX then d.drop (d.length - DEFERRED_XADD_MAX) else d

theorem capDeferred_len {α} (d : List α) : (capDeferred d).length ≤ DEFERRED_XADD_MAX := by
  unfold capDeferred; split
  · simp; omega
  · omega

theorem capDeferred_suffix {α} (d : List α) : ∃ p, p ++ capDeferred d = d := by
  unfold capDeferred; split
  · exact ⟨d.take (d.length - DEFERRED_XADD_MAX), List.take_append_drop _ _⟩
  · exact ⟨[], rfl⟩

/-! ## Re-keying (`resolve_current_names`, `:682-742`) -/

/-- A graph is identified by its allocation (`data_ptr`). -/
structure PE where
  name : List Char
  graph : Nat
  deriving DecidableEq, Repr

structure BG where
  graph : Nat
  captured : List Char
  current : Option (List Char)

/-- The registry's view: key → graph. -/
abbrev Registry := List (List Char × Nat)

def lookup (reg : Registry) (k : List Char) : Option Nat := (reg.find? (·.1 = k)).map (·.2)

/-- First pass: the captured key still names this graph. -/
def pass1 (reg : Registry) (b : BG) : BG :=
  if lookup reg b.captured = some b.graph then { b with current := some b.captured } else b

/-- Second pass, one registry entry at a time: the first unresolved batch graph with this
allocation takes its name. (The early `break` once all are resolved does not change the
result.) -/
def pass2Step (gs : List BG) (e : List Char × Nat) : List BG :=
  match gs.findIdx? (fun b => b.current.isNone && b.graph == e.2) with
  | none => gs
  | some i => gs.modify i fun b => { b with current := some e.1 }

def resolveGraphs (reg : Registry) (gs : List BG) : List BG :=
  reg.foldl pass2Step (gs.map (pass1 reg))

def nameOf (gs : List BG) (g : Nat) : Option (List Char) :=
  (gs.find? (·.graph = g)).bind (·.current)

/-- `deferred.retain_mut`: keep an entry iff its graph resolved, re-keyed to that name. -/
def retain (gs : List BG) : List PE → List PE
  | [] => []
  | pe :: ps => match nameOf gs pe.graph with
    | some n => ⟨n, pe.graph⟩ :: retain gs ps
    | none => retain gs ps

def resolve (reg : Registry) (gs : List BG) (d : List PE) : List PE :=
  retain (resolveGraphs reg gs) d

/-- Every surviving entry is keyed by a name its graph's batch slot resolved to, and no
entry changes graph. -/
theorem retain_sound (gs : List BG) (d : List PE) :
    ∀ pe ∈ retain gs d, nameOf gs pe.graph = some pe.name ∧ ∃ pe0 ∈ d, pe0.graph = pe.graph := by
  induction d with
  | nil => simp [retain]
  | cons p ps ih =>
    intro pe hpe
    simp only [retain] at hpe
    split at hpe
    · rename_i n hn
      simp at hpe; rcases hpe with rfl | h
      · exact ⟨hn, p, by simp⟩
      · obtain ⟨a, b, c, d⟩ := ih pe h; exact ⟨a, b, by simp [c], d⟩
    · obtain ⟨a, b, c, d⟩ := ih pe hpe; exact ⟨a, b, by simp [c], d⟩

/-- Entries of an unresolved graph are dropped — they have no stream to go to. -/
theorem retain_drops (gs : List BG) (d : List PE) (g : Nat) (h : nameOf gs g = none) :
    ∀ pe ∈ retain gs d, pe.graph ≠ g := by
  intro pe hpe e
  have := (retain_sound gs d pe hpe).1
  rw [e, h] at this; cases this

/-- Pass 1 resolves an unmoved graph to the key the query addressed. -/
theorem pass1_unmoved (reg : Registry) (b : BG) (h : lookup reg b.captured = some b.graph) :
    (pass1 reg b).current = some b.captured := by
  simp [pass1, h]

/-- Pass 1 never resolves to a key that names a different graph (`FLUSHALL` + rebind). -/
theorem pass1_rebound (reg : Registry) (b : BG) (g' : Nat) (h : lookup reg b.captured = some g')
    (hne : g' ≠ b.graph) : (pass1 reg b).current = b.current := by
  simp [pass1, h, hne]

/-- Pass 2 only ever assigns a registered name of the very same graph. -/
theorem pass2Step_sound (gs : List BG) (e : List Char × Nat) (i : Nat) (b : BG)
    (hb : (pass2Step gs e)[i]? = some b) (hn : b.current ≠ gs[i]?.bind (·.current)) :
    b.current = some e.1 ∧ b.graph = e.2 := by
  unfold pass2Step at hb
  split at hb
  · rename_i h; simp_all
  · rename_i j hj
    rw [List.findIdx?_eq_some_iff_getElem] at hj
    obtain ⟨hjl, hjp, _⟩ := hj
    simp at hjp
    by_cases hij : i = j
    · subst hij
      simp [List.getElem?_modify] at hb
      obtain ⟨b0, hb0, rfl⟩ := hb
      simp [List.getElem?_eq_getElem hjl] at hb0
      subst hb0
      exact ⟨rfl, hjp.2⟩
    · simp [List.getElem?_modify] at hb
      obtain ⟨a, ha, hab⟩ := hb
      rw [if_neg (Ne.symm hij)] at hab; subst hab
      simp [ha] at hn

/-! ## Grouping by stream (`sort_by` + `chunk_by`) -/
section Group

variable {α : Type} (key : α → Nat) (le : Nat → Nat → Bool)
  (trans : ∀ a b c, le a b → le b c → le a c) (total : ∀ a b, le a b || le b a)
  (refl : ∀ a, le a a)
include trans total refl

/-- Rust's `sort_by` is stable; `List.mergeSort` is too. Each graph's entries therefore
reach `stream_entries` in arrival order — the order consumers read. -/
theorem group_keeps_order (l : List α) (k : Nat) :
    (l.mergeSort (fun a b => le (key a) (key b))).filter (fun a => key a = k)
      = l.filter (fun a => key a = k) := by
  have hpair : (l.filter (fun a => key a = k)).Pairwise (fun a b => le (key a) (key b)) := by
    apply List.Pairwise.imp_of_mem (R := fun a b => key a = k ∧ key b = k)
    · intro a b _ _ ⟨ha, hb⟩; rw [ha, hb]; exact refl k
    · rw [List.pairwise_iff_forall_sublist]
      intro a b hab
      have hs := hab.subset
      have ha := hs (by simp : a ∈ [a, b]); have hb := hs (by simp : b ∈ [a, b])
      simp at ha hb; exact ⟨ha.2, hb.2⟩
  have hsub := List.sublist_mergeSort (le := fun a b => le (key a) (key b))
    (fun a b c => trans _ _ _) (fun a b => total _ _) hpair List.filter_sublist
  have h1 : List.Sublist (l.filter (fun a => key a = k))
      ((l.mergeSort (fun a b => le (key a) (key b))).filter (fun a => key a = k)) := by
    have := hsub.filter (fun a => key a = k)
    rwa [List.filter_filter, show (fun a => decide (key a = k) && decide (key a = k)) = (fun a => decide (key a = k)) by
      funext a; simp] at this
  have hlen : ((l.mergeSort (fun a b => le (key a) (key b))).filter (fun a => key a = k)).length
      = (l.filter (fun a => key a = k)).length :=
    ((List.mergeSort_perm l _).filter _).length_eq
  exact (h1.eq_of_length hlen.symm).symm

end Group

/-! ## Stream entry fields (`StreamTemplate::add`) -/

/-- The ten values, in `FIELD_NAMES` order; `fmt6` is `{:.6}`, `total` the half-ULP-padded
sum computed there. -/
def fieldValues {F : Type} (fmt6 : F → String) (received : Int) (query params : String)
    (total wait exec report : F) (cache write timeout : Bool) : List String :=
  let flag := fun b : Bool => if b then "1" else "0"
  [toString received, query, params, fmt6 total, fmt6 wait, fmt6 exec, fmt6 report,
   flag cache, flag write, flag timeout]

def FIELD_NAMES : List String :=
  ["Received at", "Query", "Query parameters", "Total duration", "Wait duration",
   "Execution duration", "Report duration", "Utilized cache", "Write", "Timeout"]

/-- The `argv` passed to `RM_StreamAdd` (`[name0, value0, …]`, `FIELD_COUNT` pairs). -/
def argv (vals : List String) : List String := (FIELD_NAMES.zip vals).flatMap fun (n, v) => [n, v]

theorem argv_len {F : Type} (fmt6 : F → String) (r : Int) (q p : String) (t w e rp : F) (a b c : Bool) :
    (argv (fieldValues fmt6 r q p t w e rp a b c)).length = 20 := by
  simp [argv, fieldValues, FIELD_NAMES]

/-- `report_failures`: one line per batch, singular/plural, and only if something failed. -/
def failMsg (failed : Nat) (g : String) : Option String :=
  if failed > 0 then
    some ("telemetry: " ++ toString failed ++ " entr" ++ (if failed = 1 then "y" else "ies")
      ++ " for graph '" ++ g ++ "' could not be appended to its stream")
  else none

theorem failMsg_none (g : String) : failMsg 0 g = none := rfl

end RedisLayer.Telemetry
