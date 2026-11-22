import SearchConcurrency.Mvcc
import SearchConcurrency.Locks

/-
# `src/query_session.rs`, `src/dispatch.rs`, `mvcc_graph.rs` constructors: session state, GIL guard

`Mvcc` and `Locks` take a session's steps as events; this file models the session's own
bookkeeping that they leave out: the lock mode and write-version state a session carries,
`with_graph`/`with_graph_mut`/`release_locks`/`Drop`, the GIL guard (`Gil::acquire`/`Drop`),
`must_run_inline`, `reauthorize_write`, `check_fork`, and `MvccGraph`'s constructors and `Drop`.

| Lean | Rust (origin/main 256ab4d3f) |
| --- | --- |
| `Sess`, `Mode`, `Ver`          | `QuerySession` `query_session.rs` (`mode: RefCell<Option<Mode>>`, `version: RefCell<Version>`), `enum Mode`, `enum Version` |
| `Sess.beginWith`, `Sess.begin` | `begin_with` (reader mode), `begin` (`WriteFacts::CLIENT`) |
| `Sess.withGraph`, `withGraphMut`, `graphArc` | `with_graph`, `with_graph_mut`, `graph_arc` |
| `Sess.releaseLocks`, `Sess.drop` | `release_locks`, `Drop for QuerySession` |
| `checkFork`                    | `check_fork` (verdict from `MvccGraph::is_stale` `mvcc_graph.rs`) |
| `G`, `acquire`, `dropGuard`    | `Gil::acquire`, `Drop for Gil`, `holds_gil` |
| `mustRunInline`                | `dispatch.rs::must_run_inline` |
| `reauthorize`                  | `reauthorize_write` |
| `MG`, `MG.fromGraph`, `MG.new`, `MG.drop` | `MvccGraph::from_graph` `mvcc_graph.rs`, `new`, `Drop` |
| `commitNext_*`                 | `CommitOp::next` `runtime/ops/commit.rs`: the lock path of its escalation (`self.runtime.upgrade_to_write()`) and index publish (`commit_deferred_indexes`) |
-/
namespace SC.Session

/-! ## The session -/

inductive Mode | reader | writer
deriving DecidableEq, Repr

/-- `enum Version`: where the session's write version stands. -/
inductive Ver | none | unchecked | current | stale
deriving DecidableEq, Repr

structure Sess where
  graph : Nat
  mode : Option Mode
  ver : Ver
  replicates : Bool
  originatedHere : Bool
deriving DecidableEq, Repr

/-- `begin_with`: `read_past_gate` takes the read lock; no write version yet. -/
def Sess.beginWith (g : Nat) (replicates originated : Bool) : Sess :=
  { graph := g, mode := some .reader, ver := .none, replicates, originatedHere := originated }

/-- `begin`: `WriteFacts::CLIENT`, both facts true. -/
def Sess.begin (g : Nat) : Sess := Sess.beginWith g true true

theorem Sess.begin_spec (g : Nat) :
    (Sess.begin g).mode = some .reader ∧ (Sess.begin g).ver = .none ∧ (Sess.begin g).graph = g ∧
    Sess.begin g = Sess.beginWith g true true := ⟨rfl, rfl, rfl, rfl⟩

theorem Sess.beginWith_spec (g : Nat) (a b : Bool) :
    (Sess.beginWith g a b).mode = some .reader ∧ (Sess.beginWith g a b).ver = .none ∧
    (Sess.beginWith g a b).replicates = a ∧ (Sess.beginWith g a b).originatedHere = b :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- `with_graph`: the graph the session locks, in either mode; a session that holds no
lock panics (`expect("session holds no lock")`), `none` here. -/
def Sess.withGraph (s : Sess) : Option Nat := s.mode.map (fun _ => s.graph)

/-- `with_graph_mut`: `Some` only in writer mode, which is the proof that the write
lock is held and mutation is allowed. -/
def Sess.withGraphMut (s : Sess) : Option Nat :=
  match s.mode with
  | some .writer => some s.graph
  | _ => none

/-- `graph_arc`: the same graph handle, whatever the mode. -/
def Sess.graphArc (s : Sess) : Nat := s.graph

/-- `release_locks` and `Drop` both do `*self.mode.borrow_mut() = None`. -/
def Sess.releaseLocks (s : Sess) : Sess := { s with mode := none }
def Sess.drop (s : Sess) : Sess := { s with mode := none }

theorem Sess.withGraph_spec (s : Sess) :
    (s.withGraph = some s.graph ↔ s.mode ≠ none) ∧ (s.mode = none → s.withGraph = none) := by
  cases h : s.mode <;> simp [Sess.withGraph, h]

/-- **Mutation needs the write lock**: `with_graph_mut` answers only in writer mode. -/
theorem Sess.withGraphMut_iff (s : Sess) : s.withGraphMut = some s.graph ↔ s.mode = some .writer := by
  unfold Sess.withGraphMut
  cases h : s.mode with
  | none => simp
  | some m => cases m <;> simp

/-- A reader session never lends the graph mutably, and neither does a released one. -/
theorem Sess.withGraphMut_reader (s : Sess) (h : s.mode = some .reader ∨ s.mode = none) :
    s.withGraphMut = none := by
  rcases h with h | h <;> simp [Sess.withGraphMut, h]

/-- Releasing is idempotent and holds nothing afterwards (so `with_graph` would panic: call it
last), and `Drop` is the same step. -/
theorem Sess.releaseLocks_spec (s : Sess) :
    s.releaseLocks.mode = none ∧ s.releaseLocks.releaseLocks = s.releaseLocks ∧
    s.releaseLocks.withGraph = none ∧ s.releaseLocks.withGraphMut = none ∧ s.drop = s.releaseLocks ∧
    s.releaseLocks.graph = s.graph ∧ s.releaseLocks.ver = s.ver := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The graph handle survives the release: a caller reaches the graph after dropping the
session (the slow log's own brief read lock). -/
theorem Sess.graphArc_after_release (s : Sess) : s.releaseLocks.graphArc = s.graphArc := rfl

/-! ## The release order of `Mode::Writer` -/

open SC.Locks in
/-- The locks still held after a program runs. -/
def heldAfter (held : List Nat) (p : List Op) : List Nat :=
  p.foldl (fun h o => match o with | .acq l => l :: h | .rel l => h.erase l) held

open SC.Locks in
/-- **`Drop` releases in reverse order and leaves nothing held.** `Mode::Writer { guard, _gil }`
drops its fields in declaration order: the write lock first, then the GIL (the reverse of how
`acquire_write` took them), and a reader drops its read lock. The program is ordered, and
afterwards the thread holds nothing, in either mode. `release_locks` is the same step done early
(`Sess.releaseLocks_spec`: `Drop` is then a no-op, so nothing is released twice). -/
theorem sessionDrop_releases_all :
    okProg [] [.acq gil, .acq graphR, .rel graphR, .rel gil] = true ∧
    heldAfter [] [.acq gil, .acq graphR, .rel graphR, .rel gil] = [] ∧
    okProg [] [.acq graphR, .rel graphR] = true ∧ heldAfter [] [.acq graphR, .rel graphR] = [] := by
  decide

/-! ## `check_fork` -/

/-- `MvccGraph::is_stale` (`mvcc_graph.rs`): `published.version + 1 != fork.version`. -/
def isStale (published fork : Nat) : Bool := decide (published + 1 ≠ fork)

/-- `check_fork`: the new state and whether it answered `Ok`. Only an `Unchecked` fork is
checked; a stale one is gone (`Err(Stale)`), a current one is rebased onto the indexer batches
published since and becomes `Current`. `rebase_onto_indexer_batches` takes the fork by `&mut`
and returns `()` since #3683: there is no third outcome (the borrowed-fork `Err(Stale)` of the
`AtomicRefCell` days is gone). -/
def checkFork (v : Ver) (stale : Bool) : Ver × Bool :=
  match v with
  | .unchecked => if stale then (.stale, false) else (.current, true)
  | v => (v, true)

theorem checkFork_unchecked_stale : checkFork .unchecked true = (.stale, false) := rfl
theorem checkFork_unchecked_current : checkFork .unchecked false = (.current, true) := rfl

/-- Every state but `Unchecked` is left alone and answers `Ok`: a writer-mode fork stays
`Current`, a `Stale` one stays stale (it is dropped), `None` has nothing to check. -/
theorem checkFork_other (v : Ver) (st : Bool) (h : v ≠ .unchecked) : checkFork v st = (v, true) := by
  cases v <;> simp_all [checkFork]

/-- **Verdict = `is_stale`**: the check refuses exactly the forks `MvccGraph::is_stale` calls
stale, and `Mvcc`'s `checkFork` event (`fork version < published version`, `checkFork_stale`)
is the same predicate for a fork made at a published version. -/
theorem checkFork_verdict (published fork : Nat) :
    (checkFork .unchecked (isStale published fork)).2 = false ↔ published + 1 ≠ fork := by
  by_cases h : published + 1 = fork <;> simp [checkFork, isStale, h]

/-- A fork that was not overtaken is current afterwards, and so is never stale again: its
session holds the write lock until it commits. -/
theorem checkFork_fresh (published fork : Nat) (h : published + 1 = fork) :
    checkFork .unchecked (isStale published fork) = (.current, true) := by
  simp [checkFork, isStale, h]

/-- A refusal makes the session stale (so `is_stale()` reports it and the write re-runs), and no
other outcome does. -/
theorem checkFork_stale_iff (v : Ver) (st : Bool) :
    (checkFork v st).1 = .stale ↔ (v = .stale ∨ (v = .unchecked ∧ st = true)) := by
  cases v <;> cases st <;> simp [checkFork]

/-! ## The GIL guard -/

/-- One thread's view: whether it is the main thread, whether `GIL_CTX` holds a context (it took
the GIL itself), and whether the module GIL is locked by it. -/
structure G where
  isMain : Bool
  ctx : Bool
  locked : Bool
deriving DecidableEq, Repr

/-- `holds_gil`: the main thread implicitly, any other through `GIL_CTX`. -/
def G.holds (g : G) : Bool := g.isMain || g.ctx

/-- `Gil::acquire`: no-op when the thread already holds it (the GIL is not recursive),
else lock it and record the context. The `Bool` is `need_release`. -/
def acquire (g : G) : G × Bool :=
  if g.holds then (g, false) else ({ g with ctx := true, locked := true }, true)

/-- `Drop for Gil`: only the guard that took the GIL releases it, taking the context out
of the thread-local so nothing can release it twice. -/
def dropGuard (g : G) (needRelease : Bool) : G :=
  if needRelease then { g with ctx := false, locked := false } else g

/-- The main thread already holds the GIL: acquiring it is the no-op it must be. -/
theorem acquire_main (g : G) (h : g.isMain = true) : acquire g = (g, false) := by
  simp [acquire, G.holds, h]

/-- **Never locked twice**: a nested acquire (the GIL is already held, by a guard or by the main
thread) changes nothing and releases nothing. -/
theorem acquire_nested (g : G) (h : g.holds = true) : acquire g = (g, false) := by
  simp [acquire, h]

/-- An acquire that does take the GIL leaves the thread holding it, locked once. -/
theorem acquire_takes (g : G) (h : g.holds = false) :
    (acquire g).1.holds = true ∧ (acquire g).1.locked = true ∧ (acquire g).2 = true := by
  have hm : g.isMain = false := by simp [G.holds] at h; exact h.1
  have hc : g.ctx = false := by simp [G.holds] at h; exact h.2
  simp [acquire, G.holds, hm, hc]

/-- **Acquire then drop is the identity** on a thread that did not hold the GIL: the context
leaves the thread-local and the GIL is released. -/
theorem acquire_drop (g : G) (h : g.holds = false) (hl : g.locked = false) :
    dropGuard (acquire g).1 (acquire g).2 = g := by
  have hm : g.isMain = false := by simp [G.holds] at h; exact h.1
  have hc : g.ctx = false := by simp [G.holds] at h; exact h.2
  obtain ⟨m, c, l⟩ := g
  simp_all [acquire, dropGuard, G.holds]

/-- A nested guard's drop releases nothing, so the outer guard still holds. -/
theorem nested_drop_keeps (g : G) (h : g.holds = true) :
    dropGuard (acquire g).1 (acquire g).2 = g := by
  simp [acquire, h, dropGuard]

/-! ## `must_run_inline` -/

/-- The context flags `must_run_inline` tests. -/
structure Flags where
  multi : Bool
  replicated : Bool
  lua : Bool
  denyBlocking : Bool
  loading : Bool
deriving DecidableEq, Repr

/-- `ctx.get_flags().intersects(MULTI | REPLICATED | LUA | DENY_BLOCKING | LOADING)`. -/
def mustRunInline (f : Flags) : Bool :=
  f.multi || f.replicated || f.lua || f.denyBlocking || f.loading

/-- Inline exactly when one of the five flags is set; a plain client command blocks and goes to
the pool. -/
theorem mustRunInline_iff (f : Flags) :
    mustRunInline f = true ↔ (f.multi = true ∨ f.replicated = true ∨ f.lua = true ∨
      f.denyBlocking = true ∨ f.loading = true) := by
  simp [mustRunInline, or_assoc]

theorem mustRunInline_plain : mustRunInline ⟨false, false, false, false, false⟩ = false := rfl

/-- AOF/RDB replay (`LOADING`) and a replica applying its master's command (`REPLICATED`) always run
inline, so neither ever reaches `reauthorize_write` off the main thread (its `debug_assert`). -/
theorem mustRunInline_loading (f : Flags) (h : f.loading = true) : mustRunInline f = true := by
  simp [mustRunInline, h]

theorem mustRunInline_replicated (f : Flags) (h : f.replicated = true) : mustRunInline f = true := by
  simp [mustRunInline, h]

/-! ## `reauthorize_write` -/

inductive Abort | replicaTrafficPaused | notAMaster
deriving DecidableEq, Repr

/-- The context flags the re-check reads: `READONLY`, and `avoid_replication_traffic()`. -/
structure RFlags where
  readonly : Bool
  avoidReplication : Bool

/-- `reauthorize_write`. `ctx = none` is `GIL_CTX.get() == None`: the main thread, where
Redis gated the command at dispatch. -/
def reauthorize (ctx : Option RFlags) (replicates originated : Bool) : Option Abort :=
  match ctx with
  | none => none
  | some f =>
    if replicates && f.avoidReplication then some .replicaTrafficPaused
    else if originated && f.readonly then some .notAMaster
    else none

theorem reauthorize_main (r o : Bool) : reauthorize none r o = none := rfl

/-- A pause window open at commit time refuses a write that replicates, before the role check. -/
theorem reauthorize_paused (f : RFlags) (o : Bool) (h : f.avoidReplication = true) :
    reauthorize (some f) true o = some .replicaTrafficPaused := by
  simp [reauthorize, h]

/-- A read-only replica refuses a write this instance decided. -/
theorem reauthorize_replica (f : RFlags) (h1 : f.readonly = true) (h2 : f.avoidReplication = false) :
    reauthorize (some f) true true = some .notAMaster := by
  simp [reauthorize, h1, h2]

/-- **A replayed write is never refused for the role**: only our own decisions require that we are
still a writable master (rejecting a replayed one would diverge us from our master). -/
theorem reauthorize_replay_not_master (f : RFlags) (r : Bool) :
    reauthorize (some f) r false ≠ some .notAMaster := by
  cases r <;> cases h : f.avoidReplication <;> simp [reauthorize, h]

/-- Neither window open: the write is authorized. -/
theorem reauthorize_ok (f : RFlags) (r o : Bool) (h1 : (r && f.avoidReplication) = false)
    (h2 : (o && f.readonly) = false) : reauthorize (some f) r o = none := by
  simp [reauthorize, h1, h2]

/-! ## `MvccGraph::new`, `from_graph`, `Drop` -/

/-- An `MvccGraph`: the published graph (identity), its data version, and whether its indexers
were cancelled. -/
structure MG where
  pub : Nat
  ver : Nat
  cancelled : Bool
deriving DecidableEq, Repr

/-- `from_graph` (`mvcc_graph.rs`): publishes the given graph as it is. -/
def MG.fromGraph (g ver : Nat) : MG := { pub := g, ver, cancelled := false }

/-- `new`: `from_graph(Graph::new(n, e, cache_size, 0, name))`, data version 0. -/
def MG.new (g : Nat) : MG := MG.fromGraph g 0

/-- `read` right after construction returns the graph given (`from_graph`'s whole contract). -/
theorem MG.fromGraph_read (g v : Nat) :
    (MG.fromGraph g v).pub = g ∧ (MG.fromGraph g v).ver = v ∧ (MG.fromGraph g v).cancelled = false :=
  ⟨rfl, rfl, rfl⟩

/-- A new graph is `Mvcc.init`: data version 0, and every interleaving from it is covered by
the `Mvcc` theorems (`Reach.init`). -/
theorem MG.new_is_init (g : Nat) {valid : List SC.Mvcc.W → Bool} :
    (MG.new g).ver = SC.Mvcc.init.ver ∧ (MG.new g).cancelled = false ∧
    SC.Mvcc.Reach valid SC.Mvcc.init := ⟨rfl, rfl, SC.Mvcc.Reach.init⟩

/-- `Drop for MvccGraph`: `self.graph.load().cancel_indexing()`: the indexers of the
published version are cancelled (`index_layer`: `Ixr.cancel`), nothing else changes. -/
def MG.drop (m : MG) : MG := { m with cancelled := true }

theorem MG.drop_spec (m : MG) :
    m.drop.cancelled = true ∧ m.drop.pub = m.pub ∧ m.drop.ver = m.ver ∧ m.drop.drop = m.drop :=
  ⟨rfl, rfl, rfl, rfl⟩

/-! ## `CommitOp::next`: the lock path (#3226) -/

open SC.Locks in
/-- `CommitOp::next` escalates (`upgrade_to_write`: drop the read lock, take the GIL and the
write lock) and then publishes index documents (`commit_deferred_indexes`: indexer lock,
then RediSearch as a writer). With the plan's index scan already dropped, this path is
ordered and never re-acquires a lock it holds. -/
theorem commitNext_closed_scan :
    LockOrdered writerClosedScanProg ∧ noReacquire [] writerClosedScanProg := by decide

open SC.Locks in
/-- With the scan still open (a `LIMIT` stopped pulling but the scan op, and so its RediSearch
results iterator, lives until the plan drops) the same path re-takes the spec lock it holds as
a reader, and takes the GIL while holding RediSearch's locks: **open bug #3226** (live abort and
stall, issue still open at 256ab4d3f). -/
theorem commitNext_open_scan :
    noReacquire [] writerOpenScanProg = false ∧ ¬ LockOrdered writerOpenScanProg :=
  ⟨bug3226_open_scan_upgrade, bug3226_open_scan_unordered⟩

open SC.Locks in
/-- The escalation segment of `CommitOp::next` is `escalate`'s: the read lock goes first
(the #726 order), then the GIL, then the write lock. -/
theorem commitNext_escalates_in_order :
    okProg [] [.acq graphR, .rel graphR, .acq gil, .acq graphR, .acq indexer, .rel indexer,
      .rel graphR, .rel gil] = true := by decide

end SC.Session
