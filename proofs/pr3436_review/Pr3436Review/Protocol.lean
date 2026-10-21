/-
# Claim 2: holding a read runtime's borrow for its whole life is sound

A transition system over the graph versions of one graph key. Every event is one atomic
step of the Rust code; any enabled event may fire (arbitrary interleaving of threads).
The protocol files are byte-identical on the PR branch (d7d166e82) and origin/main
(9212add39) — `mvcc_graph.rs`, `query_session.rs`, `graph_core.rs`, `pending.rs` — so the
cites hold for both.

| here | there |
| --- | --- |
| `C`, `S.next`          | a version = one `Arc<AtomicRefCell<Graph>>` allocation; `next` = a fresh one |
| `S.pub`, `S.published` | `MvccGraph.graph` (`mvcc_graph.rs:70`) and every version it has ever held |
| `S.lock`               | the per-graph `parking_lot::RwLock<ThreadedGraph>` (`QuerySession.mode`, `query_session.rs:31-41`) |
| `S.guards`             | read runtimes' `read_guard` (`runtime.rs:147`, taken `:460-462`) — the PR's new state |
| `S.shared`             | live per-access `AtomicRef`s: write runtimes' `graph()`/`g.borrow()`, and `MvccGraph`'s own borrows of the published version (`fork :140`, `claim :162`, `commit :190-203`) |
| `S.excl`               | live `AtomicRefMut`s (every `borrow_mut()` on a graph cell, table below) |
| `S.priv`               | a thread's private version (`QuerySession.fork` / the claimed `write()` fork) |
| `S.slot`               | `MvccGraph.write: AtomicBool` (`:72`) |
| `S.done`               | a write runtime still alive after its version was published (reply serialization, `graph_core.rs:858-872`) |

| event | Rust |
| --- | --- |
| `rlock t`     | `QuerySession::begin`/`begin_with` → `RwLock::read_arc` (`query_session.rs:159-177`) |
| `runlock t`   | dropping a reader session (`Drop` `:464`) or `escalate`'s `*mode = None` (`:387`). Enabled only once `t`'s read runtime is gone: the runtime is a local of `execute_query` / `execute_profile` (`graph_core.rs:635-671`, `:700-716`) that borrows `&session` from its caller (`:1094-1114`, `:1254-1255`, `:1383-1389`, `:1466-1467`), and `record.rs` drops it explicitly (`:142`) before the session (`:61`) |
| `wlock t`     | `begin_writer` (`:182-201`) / `escalate` (`:379-394`) → `RwLock::write_arc`; `graph.write()` in `effect.rs:62`, `bulk_insert.rs:847`, `constraint.rs:569` |
| `wunlock t`   | `release_locks` (`:303`) / session drop |
| `readNew t`   | `Runtime::new(g = tg.graph.read(), write = false)` — the only three: `graph_core.rs:632→635`, `:698→700`, `record.rs:84→86`, each under its reader session; `g.borrow()` at `runtime.rs:461` |
| `readDrop`    | `Runtime` drop: `read_guard` released (`DropOrder`) |
| `fork t`      | `write_version` in reader mode → `MvccGraph::fork` (`query_session.rs:225`, `mvcc_graph.rs:139-141`) |
| `claimW t`    | `MvccGraph::write` (`:119-129`) under the write lock: `write_version` writer mode (`query_session.rs:218-222`), `effect.rs:62`, `bulk_insert.rs:847`, `constraint.rs:335`, `:569` |
| `claimFork t` | `claim_fork` → `MvccGraph::claim` (`query_session.rs:267-298`, `mvcc_graph.rs:151-167`), writer mode only (`:373`) |
| `dropPriv t`  | `release_slot` (`query_session.rs:234-242`) / `rollback` (`mvcc_graph.rs:268`) / a stale claim |
| `mutAcq t c`  | `borrow_mut()` on `t`'s own private version, see the audit table |
| `mutAcqPub t` | `Pending::resync_published_indexes` (`pending.rs:1684`: `committed.borrow_mut()`), reached only from `abandon_write` (`graph_core.rs:1558-1561`) when `session.holds_slot()`, i.e. in writer mode |
| `publish t c` | `MvccGraph::commit` (`mvcc_graph.rs:186-266`): its `borrow_mut`s `:232,:235,:252` are on `new_graph` *before* `self.graph = new_graph` (`:263`); called under the write lock (`graph_core.rs:1587`, `bulk_insert.rs:887,1015`, `constraint.rs:340,598`, `effect.rs:≈80`) |

**`borrow_mut()` audit** (every call on a graph cell in `graph/src` + `src`, PR branch;
tests/`test_aux.rs`/`emit.rs`/`apply.rs` test modules excluded):

| site | target | model |
| --- | --- | --- |
| `runtime.rs:1435,1454` (`set_pending_*_attr`), `ops/create.rs:176,212,299,331,363,378,387`, `ops/set.rs:107`, `ops/remove.rs:139`, `ops/delete.rs:257,411,452`, `ops/merge.rs:171` | `runtime.g` of a runtime running a SET/CREATE/REMOVE/DELETE/MERGE op | write runtime ⇒ private fork: `mutAcq`. A read runtime never runs these ops (`Reuse.read_plan_no_update`) |
| `pending.rs:480,1235-1352` (`Pending::commit`s), `:1799` (`end_segment`), `runtime.rs:555` (`commit_deferred_indexes`) | `runtime.g` via `CommitOp` (`ops/commit.rs:85-148`) | write runtime (`CommitOp` refuses `!write`, `commit.rs:50`): `mutAcq` |
| `index_ddl.rs:72,104` | `runtime.g` | write runtime (`:38,:95` refuse `!write`): `mutAcq` |
| `mvcc_graph.rs:232,235,252` | `new_graph` | still private: `mutAcq`, then `publish` |
| `pending.rs:1684` | `committed` = **the published version** | `mutAcqPub`, under the write lock |
| `bulk_insert.rs:854,886,962,1012`, `constraint.rs:338,572`, `effect.rs:70` | `g_arc` from `graph.write()` / `write_version()` before `commit` | private: `mutAcq` |

So the PR's comment "a published version is never mutably borrowed again" is **false**
as stated (`published_borrowed_mut`): `resync_published_indexes` mutably borrows the
published version. What makes the PR sound is the weaker fact proven here: that borrow
happens only under the per-graph write lock, and a read runtime lives entirely inside its
session's read lock, so the two never overlap (`reader_guard_sound`).
-/
namespace Pr3436.Proto

abbrev Tid := Nat
abbrev C := Nat

inductive Lock where
  | free
  | readers (ts : List Tid)
  | writer (t : Tid)
  deriving DecidableEq, Repr

def Lock.isReader : Lock → Tid → Bool
  | .readers ts, t => ts.contains t
  | _, _ => false

structure S where
  next : C
  pub : C
  published : List C
  lock : Lock
  guards : List (Tid × C)
  shared : List (Tid × C)
  excl : List (Tid × C)
  priv : List (Tid × C)
  slot : Option Tid
  done : List (Tid × C)
  deriving DecidableEq, Repr

inductive Out where
  | ok (s : S)
  | panic
  | blocked
  deriving DecidableEq, Repr

inductive Ev where
  | rlock (t : Tid)
  | runlock (t : Tid)
  | wlock (t : Tid)
  | wunlock (t : Tid)
  | readNew (t : Tid)
  | readDrop (t : Tid) (c : C)
  | fork (t : Tid)
  | claimW (t : Tid)
  | claimFork (t : Tid)
  | dropPriv (t : Tid)
  | mutAcq (t : Tid) (c : C)
  | mutAcqPub (t : Tid)
  | mutRel (t : Tid) (c : C)
  | sAcq (t : Tid) (c : C)
  | sRel (t : Tid) (c : C)
  | publish (t : Tid) (c : C)
  | doneDrop (t : Tid) (c : C)
  deriving Repr

/-- `t` owns an entry. -/
def hasT (l : List (Tid × C)) (t : Tid) : Bool := l.any (·.1 == t)
/-- Some entry is on cell `c`. -/
def onC (l : List (Tid × C)) (c : C) : Bool := l.any (·.2 == c)

/-- One atomic step. A failed `borrow`/`borrow_mut` is `panic` (atomic_refcell
`lib.rs:132,158`); `blocked` = not enabled. -/
def step (s : S) : Ev → Out
  | .rlock t => match s.lock with
    | .free => .ok { s with lock := .readers [t] }
    | .readers ts => .ok { s with lock := .readers (t :: ts) }
    | .writer _ => .blocked
  | .runlock t => match s.lock with
    | .readers ts =>
      if ts.contains t ∧ hasT s.guards t = false then .ok { s with lock := .readers (ts.erase t) }
      else .blocked
    | _ => .blocked
  | .wlock t => match s.lock with
    | .free => .ok { s with lock := .writer t }
    | .readers [] => .ok { s with lock := .writer t }
    | _ => .blocked
  | .wunlock t =>
    if s.lock = .writer t ∧ hasT s.excl t = false then .ok { s with lock := .free } else .blocked
  | .readNew t =>
    if s.lock.isReader t then
      if onC s.excl s.pub then .panic else .ok { s with guards := (t, s.pub) :: s.guards }
    else .blocked
  | .readDrop t c => if (t, c) ∈ s.guards then .ok { s with guards := s.guards.erase (t, c) } else .blocked
  | .fork t =>
    if (s.lock.isReader t ∨ s.lock = .writer t) ∧ hasT s.priv t = false then
      if onC s.excl s.pub then .panic
      else .ok { s with priv := (t, s.next) :: s.priv, next := s.next + 1 }
    else .blocked
  | .claimW t =>
    if s.lock = .writer t ∧ s.slot = none ∧ hasT s.priv t = false then
      if onC s.excl s.pub then .panic
      else .ok { s with slot := some t, priv := (t, s.next) :: s.priv, next := s.next + 1 }
    else .blocked
  | .claimFork t =>
    if s.lock = .writer t ∧ s.slot = none ∧ hasT s.priv t = true then .ok { s with slot := some t }
    else .blocked
  | .dropPriv t =>
    if hasT s.excl t = false then
      .ok { s with priv := s.priv.filter (·.1 != t), slot := if s.slot = some t then none else s.slot }
    else .blocked
  | .mutAcq t c =>
    if (t, c) ∈ s.priv then
      if onC s.excl c ∨ onC s.shared c ∨ onC s.guards c then .panic
      else .ok { s with excl := (t, c) :: s.excl }
    else .blocked
  | .mutAcqPub t =>
    if s.lock = .writer t ∧ s.slot = some t then
      if onC s.excl s.pub ∨ onC s.shared s.pub ∨ onC s.guards s.pub then .panic
      else .ok { s with excl := (t, s.pub) :: s.excl }
    else .blocked
  | .mutRel t c => if (t, c) ∈ s.excl then .ok { s with excl := s.excl.erase (t, c) } else .blocked
  | .sAcq t c =>
    if (t, c) ∈ s.priv ∨ (t, c) ∈ s.done ∨ c = s.pub then
      if onC s.excl c then .panic else .ok { s with shared := (t, c) :: s.shared }
    else .blocked
  | .sRel t c => if (t, c) ∈ s.shared then .ok { s with shared := s.shared.erase (t, c) } else .blocked
  | .publish t c =>
    if s.lock = .writer t ∧ s.slot = some t ∧ (t, c) ∈ s.priv ∧ hasT s.excl t = false ∧
        onC s.excl c = false then
      .ok { s with pub := c, published := c :: s.published, priv := s.priv.filter (·.2 != c),
                   slot := none, done := (t, c) :: s.done }
    else .blocked
  | .doneDrop t c => if (t, c) ∈ s.done then .ok { s with done := s.done.erase (t, c) } else .blocked

/-- One graph key: version `0` published, no locks, no borrows. -/
def init : S :=
  { next := 1, pub := 0, published := [0], lock := .free, guards := [], shared := [],
    excl := [], priv := [], slot := none, done := [] }

inductive Reach : S → Prop where
  | init : Reach init
  | step {s s' : S} (e : Ev) : Reach s → step s e = .ok s' → Reach s'

/-- The invariant. -/
structure Inv (s : S) : Prop where
  pub_mem : s.pub ∈ s.published
  pub_lt : ∀ c ∈ s.published, c < s.next
  priv_fresh : ∀ p ∈ s.priv, p.2 < s.next ∧ p.2 ∉ s.published
  /-- every held read guard is on a published version, owned by a reader of the lock -/
  guard_ok : ∀ g ∈ s.guards, g.2 ∈ s.published ∧ s.lock.isReader g.1 = true
  /-- a live `borrow_mut` is on its owner's private version, or is resync's borrow of the
  published version under the owner's write lock -/
  excl_ok : ∀ e ∈ s.excl, e ∈ s.priv ∨ (e.2 = s.pub ∧ s.lock = .writer e.1)

theorem inv_init : Inv init := by
  refine ⟨by simp [init], ?_, ?_, ?_, ?_⟩ <;> simp [init]

theorem hasT_false {l : List (Tid × C)} {t : Tid} (h : hasT l t = false) :
    ∀ p ∈ l, p.1 ≠ t := by
  intro p hp he
  have : hasT l t = true := by
    unfold hasT; rw [List.any_eq_true]; exact ⟨p, hp, by simp [he]⟩
  simp_all

theorem onC_false {l : List (Tid × C)} {c : C} (h : onC l c = false) :
    ∀ p ∈ l, p.2 ≠ c := by
  intro p hp he
  have : onC l c = true := by
    unfold onC; rw [List.any_eq_true]; exact ⟨p, hp, by simp [he]⟩
  simp_all

theorem onC_true {l : List (Tid × C)} {c : C} (h : onC l c = true) : ∃ p ∈ l, p.2 = c := by
  unfold onC at h; rw [List.any_eq_true] at h
  obtain ⟨p, hp, he⟩ := h; exact ⟨p, hp, by simpa using he⟩

theorem not_reader_writer {t u : Tid} : (Lock.writer t).isReader u = false := rfl
theorem not_reader_free {u : Tid} : Lock.free.isReader u = false := rfl

theorem inv_step {s s' : S} (e : Ev) (h : Inv s) (hs : step s e = .ok s') : Inv s' := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  cases e with
  | rlock t =>
    simp only [step] at hs
    split at hs
    · rename_i hl
      simp at hs; subst hs
      refine ⟨h1, h2, h3, ?_, ?_⟩
      · intro g hg; have := (h4 g hg).2; rw [hl] at this; simp [Lock.isReader] at this
      · intro e he; rcases h5 e he with h | ⟨_, h⟩
        · exact Or.inl h
        · rw [hl] at h; cases h
    · rename_i ts hl
      simp at hs; subst hs
      refine ⟨h1, h2, h3, ?_, ?_⟩
      · intro g hg; have := h4 g hg; rw [hl] at this
        refine ⟨this.1, ?_⟩; simp [Lock.isReader] at this ⊢; exact Or.inr this.2
      · intro e he; rcases h5 e he with h | ⟨_, h⟩
        · exact Or.inl h
        · rw [hl] at h; cases h
    · cases hs
  | runlock t =>
    simp only [step] at hs
    split at hs
    · rename_i ts hl
      split at hs
      · rename_i hc
        simp at hs; subst hs
        refine ⟨h1, h2, h3, ?_, ?_⟩
        · intro g hg
          have hg4 := h4 g hg; rw [hl] at hg4
          have hne : g.1 ≠ t := hasT_false hc.2 g hg
          refine ⟨hg4.1, ?_⟩
          simp [Lock.isReader] at hg4 ⊢
          exact (List.mem_erase_of_ne hne).2 hg4.2
        · intro e he; rcases h5 e he with h | ⟨_, h⟩
          · exact Or.inl h
          · rw [hl] at h; cases h
      · cases hs
    · cases hs
  | wlock t =>
    simp only [step] at hs
    split at hs
    · rename_i hl
      simp at hs; subst hs
      refine ⟨h1, h2, h3, ?_, ?_⟩
      · intro g hg; have := (h4 g hg).2; rw [hl] at this; simp [Lock.isReader] at this
      · intro e he; rcases h5 e he with h | ⟨_, h⟩
        · exact Or.inl h
        · rw [hl] at h; cases h
    · rename_i hl
      simp at hs; subst hs
      refine ⟨h1, h2, h3, ?_, ?_⟩
      · intro g hg; have := (h4 g hg).2; rw [hl] at this; simp [Lock.isReader] at this
      · intro e he; rcases h5 e he with h | ⟨_, h⟩
        · exact Or.inl h
        · rw [hl] at h; cases h
    · cases hs
  | wunlock t =>
    simp only [step] at hs
    split at hs
    · rename_i hc
      simp at hs; subst hs
      refine ⟨h1, h2, h3, ?_, ?_⟩
      · intro g hg; have := (h4 g hg).2; rw [hc.1] at this; simp [Lock.isReader] at this
      · intro e he; rcases h5 e he with h | ⟨_, h⟩
        · exact Or.inl h
        · rw [hc.1] at h; cases h
          exact absurd rfl (hasT_false hc.2 e he)
    · cases hs
  | readNew t =>
    simp only [step] at hs
    split at hs
    · rename_i hr
      split at hs
      · cases hs
      · simp at hs; subst hs
        refine ⟨h1, h2, h3, ?_, h5⟩
        intro g hg; simp at hg
        rcases hg with rfl | hg
        · exact ⟨h1, hr⟩
        · exact h4 g hg
    · cases hs
  | readDrop t c =>
    simp only [step] at hs
    split at hs
    · simp at hs; subst hs
      exact ⟨h1, h2, h3, fun g hg => h4 g (List.mem_of_mem_erase hg), h5⟩
    · cases hs
  | fork t =>
    simp only [step] at hs
    split at hs
    · split at hs
      · cases hs
      · simp at hs; subst hs
        refine ⟨h1, fun c hc => Nat.lt_succ_of_lt (h2 c hc), ?_, h4, ?_⟩
        · intro p hp; simp at hp
          rcases hp with rfl | hp
          · exact ⟨Nat.lt_succ_self _, fun hm => Nat.lt_irrefl _ (h2 _ hm)⟩
          · exact ⟨Nat.lt_succ_of_lt (h3 p hp).1, (h3 p hp).2⟩
        · intro e he; rcases h5 e he with h | h
          · exact Or.inl (List.mem_cons_of_mem _ h)
          · exact Or.inr h
    · cases hs
  | claimW t =>
    simp only [step] at hs
    split at hs
    · split at hs
      · cases hs
      · simp at hs; subst hs
        refine ⟨h1, fun c hc => Nat.lt_succ_of_lt (h2 c hc), ?_, h4, ?_⟩
        · intro p hp; simp at hp
          rcases hp with rfl | hp
          · exact ⟨Nat.lt_succ_self _, fun hm => Nat.lt_irrefl _ (h2 _ hm)⟩
          · exact ⟨Nat.lt_succ_of_lt (h3 p hp).1, (h3 p hp).2⟩
        · intro e he; rcases h5 e he with h | h
          · exact Or.inl (List.mem_cons_of_mem _ h)
          · exact Or.inr h
    · cases hs
  | claimFork t =>
    simp only [step] at hs
    split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5⟩
    · cases hs
  | dropPriv t =>
    simp only [step] at hs
    split at hs
    · rename_i hc
      simp at hs; subst hs
      refine ⟨h1, h2, fun p hp => h3 p (List.mem_filter.1 hp).1, h4, ?_⟩
      intro e he; rcases h5 e he with h | h
      · refine Or.inl (List.mem_filter.2 ⟨h, ?_⟩)
        have := hasT_false hc e he; simpa using this
      · exact Or.inr h
    · cases hs
  | mutAcq t c =>
    simp only [step] at hs
    split at hs
    · rename_i hp
      split at hs
      · cases hs
      · simp at hs; subst hs
        refine ⟨h1, h2, h3, h4, ?_⟩
        intro e he; simp at he
        rcases he with rfl | he
        · exact Or.inl hp
        · exact h5 e he
    · cases hs
  | mutAcqPub t =>
    simp only [step] at hs
    split at hs
    · rename_i hc
      split at hs
      · cases hs
      · simp at hs; subst hs
        refine ⟨h1, h2, h3, h4, ?_⟩
        intro e he; simp at he
        rcases he with rfl | he
        · exact Or.inr ⟨rfl, hc.1⟩
        · exact h5 e he
    · cases hs
  | mutRel t c =>
    simp only [step] at hs
    split at hs
    · simp at hs; subst hs
      exact ⟨h1, h2, h3, h4, fun e he => h5 e (List.mem_of_mem_erase he)⟩
    · cases hs
  | sAcq t c =>
    simp only [step] at hs
    split at hs
    · split at hs
      · cases hs
      · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5⟩
    · cases hs
  | sRel t c =>
    simp only [step] at hs
    split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5⟩
    · cases hs
  | publish t c =>
    simp only [step] at hs
    split at hs
    · rename_i hc
      obtain ⟨hl, _, hp, hx, hxc⟩ := hc
      simp at hs; subst hs
      refine ⟨List.mem_cons_self, ?_, ?_, ?_, ?_⟩
      · intro c' hc'; simp at hc'
        rcases hc' with rfl | hc'
        · exact (h3 _ hp).1
        · exact h2 c' hc'
      · intro p hp'
        have hpm := (List.mem_filter.1 hp').1
        have hne : p.2 ≠ c := by have := (List.mem_filter.1 hp').2; simpa using this
        refine ⟨(h3 p hpm).1, ?_⟩
        simp; exact ⟨hne, (h3 p hpm).2⟩
      · intro g hg; have := (h4 g hg).2; rw [hl] at this; simp [Lock.isReader] at this
      · intro e he; rcases h5 e he with h | ⟨_, h⟩
        · refine Or.inl (List.mem_filter.2 ⟨h, ?_⟩)
          have := onC_false hxc e he; simpa using this
        · rw [hl] at h; cases h; exact absurd rfl (hasT_false hx e he)
    · cases hs
  | doneDrop t c =>
    simp only [step] at hs
    split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5⟩
    · cases hs

theorem reach_inv {s : S} (h : Reach s) : Inv s := by
  induction h with
  | init => exact inv_init
  | step e _ hs ih => exact inv_step e ih hs

/-! ### The safety theorems -/

/-- No live `borrow_mut` is ever on a version a read runtime holds. -/
theorem guard_never_excl {s : S} (hr : Reach s) : ∀ g ∈ s.guards, onC s.excl g.2 = false := by
  have h := reach_inv hr
  intro g hg
  cases hx : onC s.excl g.2 with
  | false => rfl
  | true =>
    obtain ⟨e, he, hec⟩ := onC_true hx
    have ⟨gpub, gread⟩ := h.guard_ok g hg
    rcases h.excl_ok e he with hp | ⟨_, hl⟩
    · exact absurd (hec ▸ gpub) (h.priv_fresh e hp).2
    · rw [hl] at gread; simp [Lock.isReader] at gread

/-- A read runtime's one `g.borrow()` in `Runtime::new` never panics. -/
theorem readNew_never_panics {s : S} (hr : Reach s) (t : Tid) : step s (.readNew t) ≠ .panic := by
  have h := reach_inv hr
  simp only [step]
  split
  · rename_i hrd
    split
    · rename_i hx
      obtain ⟨e, he, hec⟩ := onC_true hx
      rcases h.excl_ok e he with hp | ⟨_, hl⟩
      · exact absurd (hec ▸ h.pub_mem) (h.priv_fresh e hp).2
      · rw [hl] at hrd; simp [Lock.isReader] at hrd
    · simp
  · simp

/-- A `borrow_mut` on a private version never meets a read guard. -/
theorem priv_unguarded {s : S} (hr : Reach s) {t : Tid} {c : C} (hp : (t, c) ∈ s.priv) :
    onC s.guards c = false := by
  have h := reach_inv hr
  cases hx : onC s.guards c with
  | false => rfl
  | true =>
    obtain ⟨g, hg, hgc⟩ := onC_true hx
    exact absurd (hgc ▸ (h.guard_ok g hg).1) (h.priv_fresh _ hp).2

/-- Resync's `borrow_mut` of the published version happens with no read guard alive. -/
theorem writer_no_guards {s : S} (hr : Reach s) {t : Tid} (hl : s.lock = .writer t) :
    s.guards = [] := by
  have h := reach_inv hr
  cases hg : s.guards with
  | nil => rfl
  | cons g gs =>
    have := (h.guard_ok g (by rw [hg]; exact List.mem_cons_self)).2
    rw [hl] at this; simp [Lock.isReader] at this

/-- **Claim 2 (headline).** In every reachable state: each read guard sits on a published
version with no live `borrow_mut`; the guards never decide whether a `borrow_mut` panics
(its target carries no guard — the panic condition is main's, `excl ∨ shared`); and the
guard's own borrow never panics. -/
theorem reader_guard_sound {s : S} (hr : Reach s) :
    (∀ g ∈ s.guards, g.2 ∈ s.published ∧ onC s.excl g.2 = false) ∧
    (∀ t c, (t, c) ∈ s.priv → onC s.guards c = false) ∧
    (∀ t, s.lock = .writer t → onC s.guards s.pub = false) ∧
    (∀ t, step s (.readNew t) ≠ .panic) := by
  refine ⟨fun g hg => ⟨((reach_inv hr).guard_ok g hg).1, guard_never_excl hr g hg⟩,
    fun t c hp => priv_unguarded hr hp, fun t hl => ?_, readNew_never_panics hr⟩
  rw [writer_no_guards hr hl]; rfl

/-- The PR's guards never decide a `borrow_mut`'s outcome: on every reachable state a
`borrow_mut` of a private version panics exactly when main's would — when another
`borrow_mut` or a per-access `AtomicRef` is live on it (main's readers hold none between
accesses, and none on a private version at all). -/
theorem mutAcq_panic_iff {s : S} (hr : Reach s) {t : Tid} {c : C} (hp : (t, c) ∈ s.priv) :
    step s (.mutAcq t c) = .panic ↔ (onC s.excl c = true ∨ onC s.shared c = true) := by
  have hg := priv_unguarded hr hp
  by_cases h : onC s.excl c = true ∨ onC s.shared c = true
  · rcases h with h | h <;> simp [step, hp, hg, h]
  · simp only [not_or, Bool.not_eq_true] at h
    simp [step, hp, hg, h.1, h.2]

/-- Same for resync's `borrow_mut` of the published version: no read guard can be the
reason it panics. -/
theorem mutAcqPub_panic_iff {s : S} (hr : Reach s) {t : Tid}
    (hc : s.lock = .writer t ∧ s.slot = some t) :
    step s (.mutAcqPub t) = .panic ↔ (onC s.excl s.pub = true ∨ onC s.shared s.pub = true) := by
  have hg : onC s.guards s.pub = false := by rw [writer_no_guards hr hc.1]; rfl
  by_cases h : onC s.excl s.pub = true ∨ onC s.shared s.pub = true
  · rcases h with h | h <;> simp [step, hc.1, hc.2, hg, h]
  · simp only [not_or, Bool.not_eq_true] at h
    simp [step, hc.1, hc.2, hg, h.1, h.2]

/-! ### Traces -/

def run : S → List Ev → Out
  | s, [] => .ok s
  | s, e :: es => match step s e with
    | .ok s' => run s' es
    | o => o

theorem run_reach : ∀ {s : S} (es : List Ev) {s' : S}, Reach s → run s es = .ok s' → Reach s'
  | _, [], _, hr, h => by simp [run] at h; subst h; exact hr
  | s, e :: es, s', hr, h => by
    simp only [run] at h
    cases h₁ : step s e with
    | ok s₁ => rw [h₁] at h; exact run_reach es (Reach.step e hr h₁) h
    | panic => rw [h₁] at h; cases h
    | blocked => rw [h₁] at h; cases h

/-- **Refutes the PR's literal premise.** A published version *is* mutably borrowed again:
writer 1 publishes version 1, writer 2 claims the slot, its query fails after a `Commit`
published index documents, and `abandon_write` → `resync_published_indexes` calls
`committed.borrow_mut()` on version 1 (`pending.rs:1684`). Harmless to readers only
because of the write lock (`reader_guard_sound`). -/
def resyncTrace : List Ev :=
  [.wlock 1, .claimW 1, .publish 1 1, .wunlock 1, .wlock 2, .claimW 2, .mutAcqPub 2]

theorem published_borrowed_mut :
    ∃ s, Reach s ∧ (2, 1) ∈ s.excl ∧ 1 ∈ s.published ∧ s.pub = 1 := by
  refine ⟨_, run_reach resyncTrace Reach.init (s' :=
    { next := 3, pub := 1, published := [1, 0], lock := .writer 2, guards := [], shared := [],
      excl := [(2, 1)], priv := [(2, 2)], slot := some 2, done := [(1, 1)] }) (by decide),
    by decide, by decide, rfl⟩

/-- A reader cannot be inside that window: with writer 2 holding the lock, `readNew` is
not enabled (`rlock` blocks). -/
theorem resync_excludes_readers :
    run init (resyncTrace ++ [.rlock 3]) = .blocked := by decide

/-- **Pre-existing (main and PR alike; write runtimes are unchanged by #3436).** Writer 1
publishes, releases its locks, and serializes its reply through per-access borrows of its
own — now published — version (`graph_core.rs:857-872`, `release_locks` *before*
`reply_verbose`). Writer 2 then takes the lock and the slot, fails after a `Commit`, and
resyncs: its `borrow_mut` of that same version meets writer 1's live `AtomicRef` and
panics ("already immutably borrowed"); the converse order panics writer 1's reply
("already mutably borrowed"). No read guard is involved. -/
def replyRaceTrace : List Ev :=
  [.wlock 1, .claimW 1, .publish 1 1, .wunlock 1, .sAcq 1 1, .wlock 2, .claimW 2]

theorem writer_reply_race :
    (match run init replyRaceTrace with
      | .ok s => step s (.mutAcqPub 2) | o => o) = .panic := by decide

theorem writer_reply_race_rev :
    run init [.wlock 1, .claimW 1, .publish 1 1, .wunlock 1, .wlock 2, .claimW 2, .mutAcqPub 2,
      .sAcq 1 1] = .panic := by decide

end Pr3436.Proto
