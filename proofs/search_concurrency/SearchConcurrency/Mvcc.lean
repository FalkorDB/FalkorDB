/-
# MVCC publish/read, fork-and-commit, GIL, GRAPH.DELETE and BGSAVE fork

A transition system over one graph. Each event is one atomic step of the Rust code;
the interleaving is arbitrary (`Step` is a relation, any enabled event may fire).
Since #3657 (34a708cfa) there is no write slot: a writer forks a private version, and
`MvccGraph::commit` publishes it only if it still extends the published one.

| here | there |
| --- | --- |
| `S.pub`, `S.ver`           | `MvccGraph.graph` (the published `Arc`) and its data `Graph::version` (`graph/src/graph/mvcc_graph.rs:74` `MvccGraph`; `new_version` sets `version + 1`; `new_build_version` keeps it, `graph.rs:1139`) |
| `S.forks`                  | `Version::Unchecked` forks (`QuerySession.version`, `src/query_session.rs:108` `enum Version`): owner, the published version forked, its base |
| `S.cur`                    | the one `Version::Current` version: it is built under the write lock (the owner holds the GIL), so nothing can commit before its own commit |
| `Ev.read`                  | `MvccGraph::read` (`:118`) — clone of the published Arc |
| `Ev.forkW`                 | `MvccGraph::fork` (`mvcc_graph.rs:130`) via `QuerySession::write_version` in reader mode (`query_session.rs:211-224`): takes nothing; refused if the thread has a `Current` version (a session holds one version) |
| `Ev.forkCur`               | the same in writer mode (`Version::Current`), refused if the thread already has an unchecked fork or a `Current` version (`debug_assert!(Version::None)`, "one write version per session"), and `graph.write()` + `fork()` in `effect.rs`/`constraint.rs`/`bulk_insert.rs` |
| `Ev.checkFork`             | `QuerySession::check_fork` (`:247`) → `MvccGraph::is_stale` (`mvcc_graph.rs:144`): `published.version + 1 != fork.version` ⇒ `Version::Stale` + `Err(Stale)`, else `Current`; called from `upgrade_to_write` (`:332`, the call at `:348`) in writer mode. Since #3567 (b18530683) it also rebases the fork onto the indexer batches published since (`MvccGraph::rebase_onto_indexer_batches`, `mvcc_graph.rs:165`): index progress only, data version and data untouched, so the verdict and the model are unchanged (native_index `Rebase.lean`: `V.rebased_keeps_data`, `V.rebased_not_stale`, `rebase_progress`). Its one `Err(Stale)` (a release build, a fork still borrowed) cannot happen to an escalating writer, whose fork its owner has stopped reading; a debug build panics on it |
| `Ev.abandon`               | dropping the session/runtime: an uncommitted version (`Unchecked`, `Current` or `Stale`) is just dropped |
| `Ev.mutate`                | any mutation on the private version (runtime `Commit` op) — requires writer mode |
| `Ev.commit`                | `commit_and_replicate` → `MvccGraph::commit` → `commit_version(.., true)` (`src/graph_core.rs:1553`, `mvcc_graph.rs:205`, `:261`) up to the swap: loads the published pointer (`S.pid` into `S.ld`), `is_stale` ⇒ `Err(Stale)`; `Graph::validate` (`valid`) refused ⇒ `Err(Invalid)`. A refusal publishes nothing and the version is dropped |
| `Ev.swap`                  | the publish, `self.graph.compare_and_swap(&published, new)` (#3545, `mvcc_graph.rs:327-331`): installs the version only if the published pointer is still the one `Ev.commit` loaded (`S.pid = p0`), else `Err(Stale)` ("another version was published while this commit ran") and nothing is published. With the host's global lock nothing publishes between the two (`cas_current_commit_publishes`); a publisher without it, e.g. an unlocked `Ev.batch`, makes the swap fail (`cas_refuses_overtaken_commit`) |
| `Ev.batch`                 | `MvccGraph::commit_indexer_batch` (`mvcc_graph.rs:235`) → `commit_version(.., false)` (`:261`): publishes a version at the *same* data version (`new_build_version`), changes no data, bumps `batches_published` (`S.bver`) and installs a new published pointer (`S.pid`). Its own swap is atomic here and enabled at any time (it may run without the global lock, the adversarial case for a writer's swap); its CAS refusal is native_index `Batch.commit_cas_stale` |
| `S.gil`, `Ev.escalate`     | `QuerySession::escalate` (`query_session.rs:358`): drop read lock, `Gil::acquire`, `write_arc` |
| `S.registered`             | `GRAPH_REGISTRY` membership (`graph_core.rs:140`, `graph_is_registered` `:188`) |
| `Ev.delete`                | `GRAPH.DELETE` (`src/commands/delete.rs:37`) → `graph_free` (`graph_core.rs:1777`), main thread, GIL held |
| `Ev.fork`                  | BGSAVE `fork()` on the main thread with the GIL (`redis_type.rs:350` `pre_fork_prepare`, `module_init.rs:118` `on_fork_child`) |

**#3657 (34a708cfa).** `MvccGraph::write`/`claim`/`rollback` and the `AtomicBool` are gone.
The host serializes writers (`commit` takes `&mut self`, handed out under the per-graph
write lock), so the module needs no lock; what it must still catch is a fork that another
commit overtook, which `is_stale` finds before anything unversioned is published
(`check_fork`) and `commit` finds again as `CommitError::Stale`. A version made under the
write lock is `Current`: no commit can come between its fork and its own commit
(`current_not_stale`). #3290's retry (a queued write forks as a reader, finds its fork
stale, re-runs on top of the inline write) is `stale_then_rerun`.

Transactions are lists of abstract writes; a version is the list of all writes it
contains, in order. The private version is `base ++ pending`.
-/

namespace SC.Mvcc

abbrev W := Nat
abbrev Tid := Nat

structure S where
  /-- published version -/
  pub : List W
  /-- its data `Graph::version` -/
  ver : Nat
  /-- number of indexer batches published (`FalkorDbIndex::batches_published`) -/
  bver : Nat
  /-- the identity of the published `Arc` (`MvccGraph.graph`): every publish, a writer's or a
  batch's, installs a new one, which is what `compare_and_swap` compares (#3545) -/
  pid : Nat
  /-- a commit in flight (#3545): its owner, and the published pointer it loaded (`self.graph.load_full()`)
  and checked against; the `compare_and_swap` publishes over that one only -/
  ld : Option (Tid × Nat)
  /-- the `Current` version: owner, the data version it was forked at, its base, its pending
  writes. Built under the write lock, so its owner holds the GIL. -/
  cur : Option (Tid × Nat × List W × List W)
  /-- `Unchecked` forks: owner, the data version forked, its base -/
  forks : List (Tid × Nat × List W)
  /-- sessions whose fork was found stale (they re-run) -/
  stale : List Tid
  /-- who holds the GIL (the main thread is `0`) -/
  gil : Option Tid
  registered : Bool
  /-- every version ever published by a writer, newest first (history, for the theorems) -/
  hist : List (List W)
  /-- every snapshot a reader or a fork child has taken -/
  seen : List (List W)
  /-- every write whose commit returned OK to a client, in order -/
  acked : List W
deriving Repr, DecidableEq

inductive Ev where
  | read (t : Tid)
  | forkW (t : Tid)
  | forkCur (t : Tid)
  | checkFork (t : Tid)
  | abandon (t : Tid)
  | escalate (t : Tid)
  | release (t : Tid)
  | mutate (t : Tid) (w : W)
  | commit (t : Tid)
  | swap (t : Tid)
  | batch
  | delete
  | fork
deriving Repr

def dropT (fs : List (Tid × Nat × List W)) (t : Tid) : List (Tid × Nat × List W) := fs.filter (·.1 ≠ t)

/-- One atomic step. `none` = not enabled. `valid` is `Graph::validate` on the private
version. -/
def step (valid : List W → Bool) (s : S) : Ev → Option S
  | .read _ => some { s with seen := s.pub :: s.seen }
  | .forkW t =>                                        -- takes nothing; reader mode has no `Current` version
    if s.cur.any (·.1 = t) then none
    else some { s with forks := (t, s.ver, s.pub) :: dropT s.forks t }
  | .forkCur t =>                                      -- writer mode / under `graph.write()`
    -- a session holds one version (`write_version`'s `debug_assert!(Version::None)`): no unchecked
    -- fork of this thread and no `Current` version yet
    if s.gil ≠ some t ∨ s.cur.isSome ∨ (s.forks.find? (·.1 = t)).isSome then none
    else some { s with cur := some (t, s.ver, s.pub, []) }
  | .checkFork t =>
    if s.gil ≠ some t then none else
    match s.forks.find? (·.1 = t) with
    | none => none
    | some (_, fv, b) =>
      if fv = s.ver then some { s with cur := some (t, fv, b, []), forks := dropT s.forks t }
      else some { s with forks := dropT s.forks t, stale := t :: s.stale }   -- `Err(Stale)`
  | .abandon t => some { s with forks := dropT s.forks t, cur := s.cur.filter (·.1 ≠ t),
                                 ld := s.ld.filter (·.1 ≠ t) }
  | .escalate t => match s.gil with
    | none => some { s with gil := some t }
    | some _ => none                                   -- blocks
  | .release t => if s.gil = some t then some { s with gil := none, cur := none, ld := none } else none
  | .mutate t w => match s.cur with
    | some (o, fv, b, p) =>
      if o = t ∧ s.gil = some t ∧ s.registered ∧ s.ld = none then
        some { s with cur := some (o, fv, b, p ++ [w]) } else none
    | none => none
  | .commit t => match s.cur with                      -- `MvccGraph::commit` up to the swap: load, check, validate
    | some (o, fv, b, p) =>
      if o = t ∧ s.gil = some t ∧ s.registered ∧ s.ld = none then
        if fv ≠ s.ver then some { s with cur := none }          -- `CommitError::Stale` (not extending the published)
        else if valid (b ++ p) then some { s with ld := some (t, s.pid) }   -- checks passed against this pointer
        else some { s with cur := none }                        -- `CommitError::Invalid`
      else none
    | none => none
  | .swap t => match s.ld, s.cur with                  -- `compare_and_swap(&published, new)` (`mvcc_graph.rs:327`)
    | some (o', p0), some (o, fv, b, p) =>
      if o = t ∧ o' = t ∧ s.gil = some t ∧ s.registered then
        if s.pid = p0 then
          some { s with pub := b ++ p, ver := s.ver + 1, pid := s.pid + 1, cur := none, ld := none,
                        hist := (b ++ p) :: s.hist, acked := s.acked ++ p }
        else some { s with cur := none, ld := none }            -- `CommitError::Stale` (another version slipped in)
      else none
    | _, _ => none
  | .batch => some { s with bver := s.bver + 1, pid := s.pid + 1 }       -- same data version, same data
  | .delete => if s.gil = none ∨ s.gil = some 0 then some { s with registered := false } else none
  | .fork => if s.gil = none ∨ s.gil = some 0 then some { s with seen := s.pub :: s.seen } else none

def init : S :=
  { pub := [], ver := 0, bver := 0, pid := 0, ld := none, cur := none, forks := [], stale := [], gil := none,
    registered := true, hist := [[]], seen := [], acked := [] }

inductive Reach (valid : List W → Bool) : S → Prop where
  | init : Reach valid init
  | step {s s' : S} (e : Ev) : Reach valid s → step valid s e = some s' → Reach valid s'

/-- The invariant. -/
structure Inv (s : S) : Prop where
  pub_hist : s.hist.head? = some s.pub
  cur_ok : ∀ o fv b p, s.cur = some (o, fv, b, p) → fv = s.ver ∧ b = s.pub ∧ s.gil = some o
  seen_hist : ∀ v ∈ s.seen, v ∈ s.hist
  acked_pub : s.acked = s.pub
  hist_prefix : ∀ v ∈ s.hist, v <+: s.pub
  ver_hist : s.hist.length = s.ver + 1
  fork_ok : ∀ f ∈ s.forks, f.2.1 ≤ s.ver ∧ (f.2.1 = s.ver → f.2.2 = s.pub)
  /-- a session holds one version: a thread with an unchecked fork has no `Current` version -/
  fork_cur : ∀ f ∈ s.forks, ∀ o fv b p, s.cur = some (o, fv, b, p) → f.1 ≠ o

theorem inv_init : Inv init := by
  refine ⟨rfl, ?_, ?_, rfl, ?_, rfl, ?_, ?_⟩
  · intro o fv b p h; cases h
  · intro v h; cases h
  · intro v h; simp [init] at h; subst h; exact List.nil_prefix
  · intro f h; cases h
  · intro f h; cases h

theorem mem_dropT {fs : List (Tid × Nat × List W)} {t : Tid} {f} (h : f ∈ dropT fs t) : f ∈ fs :=
  (List.mem_filter.1 h).1

theorem seen_cons (s : S) (h1 : s.hist.head? = some s.pub) (h3 : ∀ v ∈ s.seen, v ∈ s.hist) :
    ∀ v ∈ s.pub :: s.seen, v ∈ s.hist := by
  intro v hv; simp at hv
  rcases hv with rfl | hv
  · cases hh : s.hist with
    | nil => simp [hh] at h1
    | cons x xs => simp [hh] at h1; subst h1; simp
  · exact h3 v hv

theorem inv_step (valid : List W → Bool) {s s' : S} (e : Ev) (h : Inv s) (hs : step valid s e = some s') :
    Inv s' := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  cases e with
  | read t =>
    simp [step] at hs; subst hs
    exact ⟨h1, h2, seen_cons s h1 h3, h4, h5, h6, h7, h8⟩
  | forkW t =>
    simp only [step] at hs
    split at hs
    · cases hs
    · rename_i hg
      simp at hs; subst hs
      refine ⟨h1, h2, h3, h4, h5, h6, ?_, ?_⟩
      · intro f hf; simp at hf
        rcases hf with rfl | hf
        · exact ⟨Nat.le_refl _, fun _ => rfl⟩
        · exact h7 f (mem_dropT hf)
      · intro f hf o fv' b' p' hc
        simp at hf
        rcases hf with rfl | hf
        · intro ho; apply hg
          have hc' : s.cur = some (o, fv', b', p') := hc
          simp [hc']; exact ho.symm
        · exact h8 f (mem_dropT hf) o fv' b' p' hc
  | forkCur t =>
    simp only [step] at hs
    split at hs
    · cases hs
    · rename_i hg
      simp at hs; subst hs
      rw [not_or, not_or] at hg
      obtain ⟨hg1, hg2, hg3⟩ := hg
      refine ⟨h1, ?_, h3, h4, h5, h6, h7, ?_⟩
      · intro o fv b p hh; simp at hh
        obtain ⟨rfl, rfl, rfl, rfl⟩ := hh
        exact ⟨rfl, rfl, by simpa using hg1⟩
      · intro f hf o fv' b' p' hc; simp at hc
        obtain ⟨rfl, -, -, -⟩ := hc
        intro hft; apply hg3
        exact List.find?_isSome.mpr ⟨f, hf, by simp [hft]⟩
  | checkFork t =>
    simp only [step] at hs
    split at hs
    · cases hs
    · rename_i hg
      split at hs
      · cases hs
      · rename_i t' fv b hf
        have hfm := List.mem_of_find?_eq_some hf
        have ht : t' = t := by simpa using List.find?_some hf
        subst ht
        split at hs
        · rename_i hv
          simp at hs; subst hs
          refine ⟨h1, ?_, h3, h4, h5, h6, fun f hf' => h7 f (mem_dropT hf'), ?_⟩
          · intro o fv' b' p hh; simp at hh
            obtain ⟨rfl, rfl, rfl, rfl⟩ := hh
            exact ⟨hv, (h7 _ hfm).2 hv, by simpa using hg⟩
          · intro f hf' o fv' b' p hh; simp at hh
            obtain ⟨rfl, -, -, -⟩ := hh
            have := (List.mem_filter.1 hf').2
            simpa using this
        · simp at hs; subst hs
          exact ⟨h1, h2, h3, h4, h5, h6, fun f hf' => h7 f (mem_dropT hf'),
            fun f hf' => h8 f (mem_dropT hf')⟩
  | abandon t =>
    simp [step] at hs; subst hs
    have hcur : ∀ o fv b p, (s.cur.filter (·.1 ≠ t)) = some (o, fv, b, p) →
        s.cur = some (o, fv, b, p) := by
      intro o fv b p hh
      cases hc : s.cur with
      | none => simp [hc] at hh
      | some c => simp [hc, Option.filter] at hh; rw [hh.2]
    refine ⟨h1, ?_, h3, h4, h5, h6, fun f hf' => h7 f (mem_dropT hf'), ?_⟩
    · intro o fv b p hh
      exact h2 o fv b p (hcur o fv b p hh)
    · intro f hf' o fv b p hh
      exact h8 f (mem_dropT hf') o fv b p (hcur o fv b p hh)
  | escalate t =>
    simp only [step] at hs; split at hs
    · rename_i hg
      simp at hs; subst hs
      refine ⟨h1, ?_, h3, h4, h5, h6, h7, h8⟩
      intro o fv b p hh
      have := h2 o fv b p hh
      simp [hg] at this
    · cases hs
  | release t =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs
      refine ⟨h1, ?_, h3, h4, h5, h6, h7, ?_⟩
      · intro o fv b p hh; cases hh
      · intro f hf o fv b p hh; cases hh
    · cases hs
  | mutate t w =>
    simp only [step] at hs; split at hs
    · rename_i o fv b p hsl
      split at hs
      · simp at hs; subst hs
        refine ⟨h1, ?_, h3, h4, h5, h6, h7, ?_⟩
        · intro o' fv' b' p' hh; simp at hh
          obtain ⟨rfl, rfl, rfl, -⟩ := hh
          exact h2 _ _ _ _ hsl
        · intro f hf o' fv' b' p' hh; simp at hh
          obtain ⟨rfl, rfl, rfl, -⟩ := hh
          exact h8 f hf _ _ _ _ hsl
      · cases hs
    · cases hs
  | commit t =>
    simp only [step] at hs; split at hs
    · rename_i o fv b p hsl
      split at hs
      · split at hs
        · simp at hs; subst hs
          refine ⟨h1, ?_, h3, h4, h5, h6, h7, ?_⟩
          · intro _ _ _ _ hh; cases hh
          · intro _ _ _ _ _ _ hh; cases hh
        · split at hs
          · simp at hs; subst hs
            exact ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
          · simp at hs; subst hs
            refine ⟨h1, ?_, h3, h4, h5, h6, h7, ?_⟩
            · intro _ _ _ _ hh; cases hh
            · intro _ _ _ _ _ _ hh; cases hh
      · cases hs
    · cases hs
  | swap t =>
    simp only [step] at hs; split at hs
    · rename_i o' p0 o fv b p hld hsl
      have hc := h2 o fv b p hsl
      split at hs
      · split at hs
        · simp at hs; subst hs
          obtain ⟨hv, hb, -⟩ := hc
          subst hb
          refine ⟨rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
          · intro _ _ _ _ hh; cases hh
          · intro v hv'; exact List.mem_cons_of_mem _ (h3 v hv')
          · simp [h4]
          · intro v hv'; simp at hv'
            rcases hv' with rfl | hv'
            · exact List.prefix_refl _
            · exact (h5 v hv').trans (List.prefix_append _ _)
          · simp [h6]
          · intro f hf
            have := (h7 f hf).1
            refine ⟨Nat.le_succ_of_le this, fun e => absurd e ?_⟩
            show f.2.1 ≠ s.ver + 1; omega
          · intro _ _ _ _ _ _ hh; cases hh
        · simp at hs; subst hs
          refine ⟨h1, ?_, h3, h4, h5, h6, h7, ?_⟩
          · intro _ _ _ _ hh; cases hh
          · intro _ _ _ _ _ _ hh; cases hh
      · cases hs
    · cases hs
  | batch =>
    simp [step] at hs; subst hs
    exact ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
  | delete =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
    · cases hs
  | fork =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs
      exact ⟨h1, h2, seen_cons s h1 h3, h4, h5, h6, h7, h8⟩
    · cases hs

theorem reach_inv {valid : List W → Bool} {s : S} (h : Reach valid s) : Inv s := by
  induction h with
  | init => exact inv_init
  | step e _ hs ih => exact inv_step valid e ih hs

/-- **Snapshot isolation / no partial commit.** Every snapshot a reader (or a BGSAVE
child) has ever taken is a version that was published whole by some commit. -/
theorem snapshots_are_committed {valid : List W → Bool} {s : S} (h : Reach valid s) : ∀ v ∈ s.seen, v ∈ s.hist :=
  (reach_inv h).seen_hist

/-- **No lost writes, nothing committed twice.** The published version is exactly the
concatenation of every acknowledged write, each once, in acknowledgement order. -/
theorem no_lost_writes {valid : List W → Bool} {s : S} (h : Reach valid s) : s.acked = s.pub := (reach_inv h).acked_pub

/-- **History is linear.** Every published version is a prefix of the current one:
a later commit never drops or reorders an earlier one. -/
theorem history_linear {valid : List W → Bool} {s : S} (h : Reach valid s) : ∀ v ∈ s.hist, v <+: s.pub :=
  (reach_inv h).hist_prefix

/-- A `Current` version is always built on the published one, at its data version, by the
GIL holder: with no write slot, the write lock alone serializes writers (no write-write
overlap, no lost update). -/
theorem private_on_committed {valid : List W → Bool} {s : S} (h : Reach valid s) (o : Tid) (fv : Nat) (b p : List W)
    (hsl : s.cur = some (o, fv, b, p)) : b = s.pub := ((reach_inv h).cur_ok o fv b p hsl).2.1

/-- **A `Current` version is never stale** (`commit`'s `CommitError::Stale` is unreachable
for a session under the write lock, which is what `commit_and_replicate`'s `unreachable!`
relies on): its fork is at the published data version, and `commit` takes the `valid`
branch whenever `Graph::validate` accepts. -/
theorem current_not_stale {valid : List W → Bool} {s : S} (h : Reach valid s) (o : Tid) (fv : Nat) (b p : List W)
    (hsl : s.cur = some (o, fv, b, p)) : fv = s.ver ∧ s.gil = some o :=
  let h' := (reach_inv h).cur_ok o fv b p hsl; ⟨h'.1, h'.2.2⟩

/-- **A current commit publishes** (#3545): the owner of a `Current` version that `validate` accepts
loads the published pointer and checks against it (`commit`); with nothing published in between, the
`compare_and_swap` succeeds and the version is published, acknowledged and made the new history head
(`swap`). -/
theorem commit_publishes {valid : List W → Bool} {s : S} (h : Reach valid s) (t : Tid) (fv : Nat) (b p : List W)
    (hsl : s.cur = some (t, fv, b, p)) (hg : s.gil = some t) (hr : s.registered = true)
    (hld : s.ld = none) (hv : valid (b ++ p) = true) :
    ∃ s1, step valid s (.commit t) = some s1 ∧
      step valid s1 (.swap t) =
        some { s with pub := b ++ p, ver := s.ver + 1, pid := s.pid + 1, cur := none, ld := none,
                      hist := (b ++ p) :: s.hist, acked := s.acked ++ p } := by
  have := (current_not_stale h t fv b p hsl).1
  refine ⟨{ s with ld := some (t, s.pid) }, ?_, ?_⟩
  · simp [step, hsl, hg, hr, hld, hv, this]
  · simp [step, hsl, hg, hr]

/-- **A refused commit publishes nothing** (`MvccGraph::commit` returns `Err`; test
`a_refused_commit_publishes_nothing`): the published version, its data version and the
acknowledgements are as they were, the version is dropped (nothing to roll back), and the
next writer, holding the write lock, can fork again. -/
theorem refused_commit_publishes_nothing {valid : List W → Bool} {s : S} (h : Reach valid s) (t : Tid)
    (fv : Nat) (b p : List W) (hsl : s.cur = some (t, fv, b, p)) (hg : s.gil = some t) (hr : s.registered = true)
    (hld : s.ld = none) (hv : valid (b ++ p) = false) (hfv : fv = s.ver) :
    ∃ s', step valid s (.commit t) = some s' ∧ s'.cur = none ∧ s'.pub = s.pub ∧ s'.ver = s.ver ∧
      s'.acked = s.acked ∧ s'.hist = s.hist ∧ (step valid s' (.forkCur t)).isSome := by
  have hnf : ∀ x y, (t, x, y) ∉ s.forks := fun x y hm =>
    (reach_inv h).fork_cur _ hm t fv b p hsl rfl
  refine ⟨{ s with cur := none }, ?_, rfl, rfl, rfl, rfl, rfl, by simp [step, hg, hnf]⟩
  simp [step, hsl, hg, hr, hld, hv, hfv]

/-- Even a stale refusal (unreachable for `Current`, see `current_not_stale`) publishes nothing. -/
theorem stale_commit_publishes_nothing (valid : List W → Bool) (s : S) (t : Tid) (fv : Nat) (b p : List W)
    (hsl : s.cur = some (t, fv, b, p)) (hg : s.gil = some t) (hr : s.registered = true)
    (hld : s.ld = none) (hfv : fv ≠ s.ver) :
    step valid s (.commit t) = some { s with cur := none } := by
  simp [step, hsl, hg, hr, hld, hfv]

/-- **The compare-and-swap refuses a commit that another publisher overtook** (#3545,
`CommitError::Stale`: "another version was published while this commit ran"): the commit loaded
pointer `p0`, the published pointer is no longer it (`pid ≠ p0`), and the swap publishes nothing. The
published version, its data version, the history and the acknowledgements are untouched, and the
refused version is dropped. -/
theorem swap_overtaken_publishes_nothing (valid : List W → Bool) (s : S) (t : Tid) (p0 : Nat) (fv : Nat)
    (b p : List W) (hld : s.ld = some (t, p0)) (hsl : s.cur = some (t, fv, b, p)) (hg : s.gil = some t)
    (hr : s.registered = true) (hne : s.pid ≠ p0) :
    step valid s (.swap t) = some { s with cur := none, ld := none } := by
  simp [step, hld, hsl, hg, hr, hne]

/-- …and while the pointer is still the one loaded, it publishes (the other half of the CAS). -/
theorem swap_unchanged_publishes (valid : List W → Bool) (s : S) (t : Tid) (fv : Nat)
    (b p : List W) (hld : s.ld = some (t, s.pid)) (hsl : s.cur = some (t, fv, b, p)) (hg : s.gil = some t)
    (hr : s.registered = true) :
    step valid s (.swap t) =
      some { s with pub := b ++ p, ver := s.ver + 1, pid := s.pid + 1, cur := none, ld := none,
                    hist := (b ++ p) :: s.hist, acked := s.acked ++ p } := by
  simp [step, hld, hsl, hg, hr]

/-- The writer is not writing while its commit is in flight: no `mutate` (so what was validated is what is
published), and the first `commit` of a second one is refused. -/
theorem in_flight_blocks_mutate (valid : List W → Bool) (s : S) (t : Tid) (p0 : Nat) (w : W)
    (hld : s.ld = some (t, p0)) : step valid s (.mutate t w) = none ∧ step valid s (.commit t) = none := by
  constructor <;> simp [step, hld] <;> split <;> simp_all

/-- **At most one writer commits per version** (#3290/#3657). Each published version is one
commit: `ver` counts the versions in the history, and every commit takes version `ver` to
`ver + 1` building on `pub` (`private_on_committed`). After it, every fork made at an
earlier version is behind, so its check can only be `Stale` (`checkFork_stale`). -/
theorem one_commit_per_version {valid : List W → Bool} {s s' : S} (h : Reach valid s) (t : Tid)
    (hs : step valid s (.swap t) = some s') :
    s.hist.length = s.ver + 1 ∧ (s'.ver = s.ver + 1 ∨ s'.ver = s.ver) ∧
    ∀ f ∈ s'.forks, f.2.1 < s'.ver ∨ s'.ver = s.ver := by
  have hi := reach_inv h
  refine ⟨hi.ver_hist, ?_, ?_⟩
  · simp only [step] at hs; split at hs
    · split at hs
      · split at hs
        · simp at hs; subst hs; simp
        · simp at hs; subst hs; simp
      · cases hs
    · cases hs
  · intro f hf
    simp only [step] at hs; split at hs
    · split at hs
      · split at hs
        · simp at hs; subst hs; left; have := (hi.fork_ok f hf).1; show f.2.1 < s.ver + 1; omega
        · simp at hs; subst hs; right; rfl
      · cases hs
    · cases hs

/-- **A stale check means a commit happened since the fork**, and it takes nothing: no
`Current` version is made, and nothing is published or acknowledged. -/
theorem checkFork_stale {valid : List W → Bool} {s s' : S} (h : Reach valid s) (t : Tid)
    (hs : step valid s (.checkFork t) = some s') (hst : t ∈ s'.stale) (hn : t ∉ s.stale) :
    (∃ fv b, (t, fv, b) ∈ s.forks ∧ fv < s.ver) ∧ s'.cur = s.cur ∧ s'.pub = s.pub ∧ s'.acked = s.acked := by
  have hi := reach_inv h
  simp only [step] at hs
  split at hs
  · cases hs
  · split at hs
    · cases hs
    · rename_i t' fv b hf
      have hfm := List.mem_of_find?_eq_some hf
      have ht : t' = t := by simpa using List.find?_some hf
      subst ht
      split at hs
      · simp at hs; subst hs; exact absurd hst hn
      · rename_i hv
        simp at hs; subst hs
        exact ⟨⟨fv, b, hfm, by have := (hi.fork_ok _ hfm).1; simp at this; omega⟩, rfl, rfl, rfl⟩

/-- `is_stale` is exactly `published.version + 1 != fork.version`: a fork of version `fv`
is current iff nothing was published since (`fv = ver`). A successful check yields a
`Current` version on exactly the published graph — every acknowledged write, the inline
one included (#3290: the re-run sees it). -/
theorem checkFork_on_committed {valid : List W → Bool} {s s' : S} (h : Reach valid s) (t : Tid)
    (hs : step valid s (.checkFork t) = some s') (o : Tid) (fv : Nat) (b p : List W)
    (hsl : s'.cur = some (o, fv, b, p)) :
    b = s.pub ∧ b = s.acked ∧ fv = s.ver := by
  have hs' := Reach.step _ h hs
  have h2 := (reach_inv hs').cur_ok o fv b p hsl
  have hp : s'.pub = s.pub ∧ s'.ver = s.ver := by
    simp only [step] at hs; split at hs
    · cases hs
    · split at hs
      · cases hs
      · split at hs <;> (simp at hs; subst hs; exact ⟨rfl, rfl⟩)
  rw [hp.1] at h2; rw [hp.2] at h2
  exact ⟨h2.2.1, h2.2.1.trans (no_lost_writes h).symm, h2.1⟩

/-- **An indexer batch does not make a writer's fork stale** (`commit_indexer_batch` keeps
the data version, `new_build_version`): it changes neither the published writes, nor the
data version, nor anyone's fork, nor the acknowledgements; only `bver` moves. -/
theorem batch_keeps_forks {valid : List W → Bool} (s : S) :
    ∃ s', step valid s .batch = some s' ∧ s'.pub = s.pub ∧ s'.ver = s.ver ∧ s'.forks = s.forks ∧
      s'.cur = s.cur ∧ s'.acked = s.acked ∧ s'.bver = s.bver + 1 ∧ s'.pid = s.pid + 1 :=
  ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- …so a stale/current verdict on a fork is unchanged by a batch. -/
theorem batch_then_check {valid : List W → Bool} (s s' : S) (hb : step valid s .batch = some s') (t : Tid) :
    step valid s' (.checkFork t) = (step valid s (.checkFork t)).map (fun r => { r with bver := r.bver + 1, pid := r.pid + 1 }) := by
  simp [step] at hb; subst hb
  by_cases hg : s.gil = some t
  · cases hf : s.forks.find? (·.1 = t) with
    | none => simp [step, hg, hf]
    | some f =>
      obtain ⟨a, fv, b⟩ := f
      by_cases hv : fv = s.ver <;> simp [step, hg, hf, hv]
  · simp [step, hg]

/-- **GRAPH.DELETE during running queries.** Once the key is gone no write reaches
the graph: `mutate` and `commit` are disabled. (Readers are unaffected: their
snapshots are `Arc`s, freed only when the last one drops.) -/
theorem delete_blocks_writes (valid : List W → Bool) (s : S) (h : s.registered = false) (t : Tid) (w : W) :
    step valid s (.mutate t w) = none ∧ step valid s (.commit t) = none ∧ step valid s (.swap t) = none := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [step] <;> split <;> simp_all

/-- **Fork safety.** A fork needs the GIL; a commit needs the GIL; so a fork child
never runs between the two halves of a commit — its snapshot is `pub`, which by
`snapshots_are_committed` is a whole version. -/
theorem fork_excludes_commit (valid : List W → Bool) (s : S) (t : Tid) (ht : t ≠ 0) (hg : s.gil = some t) :
    step valid s .fork = none ∧ step valid s .delete = none := by
  simp [step, hg, ht]

/-- **A thread with an unchecked fork cannot mutate or commit, in every reachable state.** A
session holds one version (`write_version`'s `debug_assert!`): `forkCur` is refused while the
thread has an unchecked fork or a `Current` version, and `forkW` while it has a `Current`
one (invariant `fork_cur`), so only `checkFork` (which consumes the fork) can give it a
`Current` version, and `mutate`/`commit` need one. -/
theorem unchecked_fork_cannot_write {valid : List W → Bool} {s : S} (h : Reach valid s) (t : Tid)
    (hf : ∃ f ∈ s.forks, f.1 = t) (w : W) :
    step valid s (.mutate t w) = none ∧ step valid s (.commit t) = none ∧ step valid s (.swap t) = none := by
  obtain ⟨f, hfm, hft⟩ := hf
  have hi := (reach_inv h).fork_cur f hfm
  cases hc : s.cur with
  | none => simp [step, hc]
  | some c =>
    obtain ⟨o, fv, b, p⟩ := c
    have hne := hi o fv b p hc
    have hot : o ≠ t := fun e => hne (hft.trans e.symm)
    simp only [step, hc, hot]
    refine ⟨by simp [hot], by simp [hot], ?_⟩
    split
    · rename_i o'' p0 o3 fv3 b3 p3 hld hcc
      simp at hcc; obtain ⟨rfl, -, -, -⟩ := hcc
      simp [hot]
    · rfl

/-- The single trace the Rust test checks: a reader-mode fork (`forkW`) from the initial state
holds no `Current` version, so `commit` is refused (an instance of `unchecked_fork_cannot_write`). -/
theorem reader_cannot_commit : ∀ t, ((step (fun _ => true) init (.forkW 1)).bind (step (fun _ => true) · (.commit t))).isNone := by
  intro t; simp [step, init, dropT, Option.bind]

/-! ## The MULTI write while a queued write matches (#3093, fixed by #3290)

Queued write `1` forks as a reader and matches; an inline writer `0` (`MULTI`/`EXEC`,
`query_sync` → `begin_writer`, main thread, GIL and write lock) runs in that window. -/

def run (s : S) : List Ev → Option S
  | [] => some s
  | e :: es => (step (fun _ => true) s e).bind (run · es)

/-- The inline writer commits `7` while `1` is still matching. -/
def inlineTrace : List Ev := [.forkW 1, .escalate 0, .forkCur 0, .mutate 0 7, .commit 0, .swap 0, .release 0]

/-- **The inline write succeeds** while the queued write is matching (C's behaviour), and
is published. -/
theorem inline_write_ok :
    ((run init inlineTrace).map (·.pub)) = some [7] := by decide

/-- …the queued write then escalates, finds its fork stale, takes nothing, and re-runs:
a fresh fork, checked, builds on the inline write. Final state: `7` then `8`, each
committed once, both acknowledged. -/
def rerunTrace : List Ev :=
  inlineTrace ++ [.escalate 1, .checkFork 1, .abandon 1, .release 1,   -- `Stale`: `WriteAttempt::Stale`
                  .forkW 1, .escalate 1, .checkFork 1, .mutate 1 8, .commit 1, .swap 1, .release 1]

theorem stale_then_rerun :
    ((run init (inlineTrace ++ [.escalate 1, .checkFork 1])).map (fun s => (s.stale, s.cur.isNone))) =
      some ([1], true) ∧
    ((run init rerunTrace).map (fun s => (s.pub, s.acked, s.ver))) = some ([7, 8], [7, 8], 2) := by
  decide

/-- An indexer batch between the queued write's fork and its check does not make it stale:
the check still succeeds, and the commit goes through. -/
theorem batch_does_not_stale :
    ((run init [.forkW 1, .batch, .escalate 1, .checkFork 1, .mutate 1 9, .commit 1, .swap 1]).map
      (fun s => (s.pub, s.stale, s.bver))) = some ([9], [], 1) := by decide

/-- **The compare-and-swap in a trace** (#3545): a writer loads and checks (`commit`), another publisher
(here an indexer batch, which installs a new `Arc`) slips in without the host's lock, and the swap is
refused: nothing is published, nothing acknowledged, and the writer's version is dropped. -/
theorem cas_refuses_overtaken_commit :
    ((run init [.escalate 0, .forkCur 0, .mutate 0 7, .commit 0, .batch, .swap 0]).map
      (fun s => (s.pub, s.ver, s.acked, s.cur.isNone, s.ld.isNone))) = some ([], 0, [], true, true) := by decide

/-- …and with the host's lock held to the end (no publisher in between) the same commit publishes. -/
theorem cas_current_commit_publishes :
    ((run init [.escalate 0, .forkCur 0, .mutate 0 7, .commit 0, .swap 0]).map
      (fun s => (s.pub, s.ver, s.acked))) = some ([7], 1, [7]) := by decide

end SC.Mvcc
