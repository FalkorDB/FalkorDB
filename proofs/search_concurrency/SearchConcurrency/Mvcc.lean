/-
# MVCC publish/read, write slot, GIL, GRAPH.DELETE and BGSAVE fork

A transition system over one graph. Each event is one atomic step of the Rust code;
the interleaving is arbitrary (`Step` is a relation, any enabled event may fire).

| here | there |
| --- | --- |
| `S.pub`, `S.ver`           | `MvccGraph.graph` (the committed `Arc`) and its `Graph::version` (`graph/src/graph/mvcc_graph.rs:70`; `new_version` sets `version + 1`, `graph.rs:1049`) |
| `S.slot`                   | `MvccGraph.write: AtomicBool` (`:72`) together with the private version it hands out |
| `S.forks`                  | unclaimed private versions (`QuerySession.fork`, `Slot::Forked`, `src/query_session.rs:95-118`) |
| `Ev.read`                  | `MvccGraph::read` (`:112`) — clone of the committed Arc |
| `Ev.claim`                 | `MvccGraph::write` (`:119`) — CAS false→true + `fork()`; since #3290 only called with the write lock held (`write_version` in writer mode `query_session.rs:216`, `attempt_settle`, `effect.rs`, bulk/constraint under `graph.write()`) |
| `Ev.forkW`                 | `MvccGraph::fork` (`:139`) via `QuerySession::write_version` in reader mode (`query_session.rs:216-231`) — no slot |
| `Ev.claimFork`             | `QuerySession::claim_fork` (`:267`) → `MvccGraph::claim` (`mvcc_graph.rs:151`): CAS, then `version + 1 != fork.version` ⇒ `store(false)` + `Stale`; called from `upgrade_to_write` (`:354`) in writer mode |
| `Ev.dropFork`              | `release_slot` (`query_session.rs:234`) on an unclaimed fork |
| `Ev.mutate`                | any mutation on the private version (runtime `Commit` op) — requires writer mode, `query_session.rs:354` |
| `Ev.commit`                | `commit_and_replicate` → `MvccGraph::commit` (`src/graph_core.rs:1571`, `mvcc_graph.rs:186`): `Graph::validate` first (`valid`); refused ⇒ `rollback` + `Err`, else Arc swap + `store(false)`, in writer mode |
| `Ev.rollback`              | `MvccGraph::rollback` (`:268`) via `release_slot` (`abandon_write`/`finish_write`, `graph_core.rs:1520,1554`) |
| `S.gil`, `Ev.escalate`     | `QuerySession::escalate` (`query_session.rs:379`): drop read lock, `Gil::acquire`, `write_arc` |
| `S.registered`             | `GRAPH_REGISTRY` membership (`graph_core.rs:140`, `graph_is_registered` `:188`) |
| `Ev.delete`                | `GRAPH.DELETE` (`src/commands/delete.rs:37`) → `graph_free` (`graph_core.rs:1777`), main thread, GIL held |
| `Ev.fork`                  | BGSAVE `fork()` on the main thread with the GIL (`redis_type.rs:350` `pre_fork_prepare`, `module_init.rs:118` `on_fork_child`) |

**#3290 (66d4d9d40, fixes #3093 / BUG 5).** A write that starts as a reader builds into
an unclaimed fork and claims the slot only when it escalates; an inline writer that
committed in between makes the claim `Stale` and the write re-runs from the start
(`execute_query_write`'s retry loop, `graph_core.rs:737`). The pre-#3290 protocol (slot
claimed as a reader) is kept as `stepPre3290` for the historical counterexample.

Transactions are lists of abstract writes; a version is the list of all writes it
contains, in order. The private version is `base ++ pending`.
-/

namespace SC.Mvcc

abbrev W := Nat
abbrev Tid := Nat

structure S where
  /-- committed version -/
  pub : List W
  /-- its `Graph::version` -/
  ver : Nat
  /-- write slot: owner, the base it was claimed on, its pending writes -/
  slot : Option (Tid × List W × List W)
  /-- unclaimed forks: owner, the version forked, its base -/
  forks : List (Tid × Nat × List W)
  /-- sessions whose fork was found stale (they re-run) -/
  stale : List Tid
  /-- who holds the GIL (the main thread is `0`) -/
  gil : Option Tid
  registered : Bool
  /-- every version ever published, newest first (history, for the theorems) -/
  hist : List (List W)
  /-- every snapshot a reader or a fork child has taken -/
  seen : List (List W)
  /-- every write whose commit returned OK to a client, in order -/
  acked : List W
deriving Repr

inductive Ev where
  | read (t : Tid)
  | claim (t : Tid)
  | forkW (t : Tid)
  | claimFork (t : Tid)
  | dropFork (t : Tid)
  | escalate (t : Tid)
  | release (t : Tid)
  | mutate (t : Tid) (w : W)
  | commit (t : Tid)
  | rollback (t : Tid)
  | delete
  | fork
deriving Repr

def dropT (fs : List (Tid × Nat × List W)) (t : Tid) : List (Tid × Nat × List W) := fs.filter (·.1 ≠ t)

/-- One atomic step. `none` = not enabled. `valid` is `Graph::validate` on the private
version. -/
def step (valid : List W → Bool) (s : S) : Ev → Option S
  | .read _ => some { s with seen := s.pub :: s.seen }
  | .claim t =>                                        -- only under the write lock (#3290)
    if s.gil ≠ some t then none else
    match s.slot with
    | none => some { s with slot := some (t, s.pub, []) }
    | some _ => none                                   -- CAS fails: `WriteSlotBusy`
  | .forkW t => some { s with forks := (t, s.ver, s.pub) :: dropT s.forks t }
  | .claimFork t =>
    if s.gil ≠ some t then none else
    match s.forks.find? (·.1 = t) with
    | none => none
    | some (_, fv, b) =>
      match s.slot with
      | some _ => some s                               -- `Claim::Busy`: fork kept, nothing taken
      | none =>
        if fv = s.ver then some { s with slot := some (t, b, []), forks := dropT s.forks t }
        else some { s with forks := dropT s.forks t, stale := t :: s.stale }   -- `Claim::Stale`, slot left free
  | .dropFork t => some { s with forks := dropT s.forks t }
  | .escalate t => match s.gil with
    | none => some { s with gil := some t }
    | some _ => none                                   -- blocks
  | .release t => if s.gil = some t then some { s with gil := none } else none
  | .mutate t w => match s.slot with
    | some (o, b, p) =>
      if o = t ∧ s.gil = some t ∧ s.registered then some { s with slot := some (o, b, p ++ [w]) } else none
    | none => none
  | .commit t => match s.slot with
    | some (o, b, p) =>
      if o = t ∧ s.gil = some t ∧ s.registered then
        if valid (b ++ p) then
          some { s with pub := b ++ p, ver := s.ver + 1, slot := none, hist := (b ++ p) :: s.hist,
                        acked := s.acked ++ p }
        else some { s with slot := none }            -- refused: `self.rollback()`, nothing published
      else none
    | none => none
  | .rollback t => match s.slot with
    | some (o, _, _) => if o = t then some { s with slot := none } else none
    | none => none
  | .delete => if s.gil = none ∨ s.gil = some 0 then some { s with registered := false } else none
  | .fork => if s.gil = none ∨ s.gil = some 0 then some { s with seen := s.pub :: s.seen } else none

def init : S :=
  { pub := [], ver := 0, slot := none, forks := [], stale := [], gil := none, registered := true,
    hist := [[]], seen := [], acked := [] }

inductive Reach (valid : List W → Bool) : S → Prop where
  | init : Reach valid init
  | step {s s' : S} (e : Ev) : Reach valid s → step valid s e = some s' → Reach valid s'

/-- The invariant. -/
structure Inv (s : S) : Prop where
  pub_hist : s.hist.head? = some s.pub
  slot_base : ∀ o b p, s.slot = some (o, b, p) → b = s.pub
  seen_hist : ∀ v ∈ s.seen, v ∈ s.hist
  acked_pub : s.acked = s.pub
  hist_prefix : ∀ v ∈ s.hist, v <+: s.pub
  ver_hist : s.hist.length = s.ver + 1
  fork_ok : ∀ f ∈ s.forks, f.2.1 ≤ s.ver ∧ (f.2.1 = s.ver → f.2.2 = s.pub)

theorem inv_init : Inv init := by
  refine ⟨rfl, ?_, ?_, rfl, ?_, rfl, ?_⟩
  · intro o b p h; cases h
  · intro v h; cases h
  · intro v h; simp [init] at h; subst h; exact List.nil_prefix
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
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  cases e with
  | read t =>
    simp [step] at hs; subst hs
    exact ⟨h1, h2, seen_cons s h1 h3, h4, h5, h6, h7⟩
  | claim t =>
    simp only [step] at hs
    split at hs
    · cases hs
    · split at hs
      · simp at hs; subst hs
        refine ⟨h1, ?_, h3, h4, h5, h6, h7⟩
        intro o b p hh; simp at hh; exact hh.2.1.symm
      · cases hs
  | forkW t =>
    simp [step] at hs; subst hs
    refine ⟨h1, h2, h3, h4, h5, h6, ?_⟩
    intro f hf; simp at hf
    rcases hf with rfl | hf
    · exact ⟨Nat.le_refl _, fun _ => rfl⟩
    · exact h7 f (mem_dropT hf)
  | claimFork t =>
    simp only [step] at hs
    split at hs
    · cases hs
    · split at hs
      · cases hs
      · rename_i t' fv b hf
        have hfm := List.mem_of_find?_eq_some hf
        split at hs
        · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5, h6, h7⟩
        · split at hs
          · rename_i hv
            simp at hs; subst hs
            refine ⟨h1, ?_, h3, h4, h5, h6, fun f hf' => h7 f (mem_dropT hf')⟩
            intro o b' p hh; simp at hh; rw [← hh.2.1]; exact (h7 _ hfm).2 hv
          · simp at hs; subst hs
            exact ⟨h1, h2, h3, h4, h5, h6, fun f hf' => h7 f (mem_dropT hf')⟩
  | dropFork t =>
    simp [step] at hs; subst hs
    exact ⟨h1, h2, h3, h4, h5, h6, fun f hf' => h7 f (mem_dropT hf')⟩
  | escalate t =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5, h6, h7⟩
    · cases hs
  | release t =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5, h6, h7⟩
    · cases hs
  | mutate t w =>
    simp only [step] at hs; split at hs
    · rename_i o b p hsl
      split at hs
      · simp at hs; subst hs
        refine ⟨h1, ?_, h3, h4, h5, h6, h7⟩
        intro o' b' p' hh; simp at hh; rw [← hh.2.1]; exact h2 o b p hsl
      · cases hs
    · cases hs
  | commit t =>
    simp only [step] at hs; split at hs
    · rename_i o b p hsl
      split at hs
      · split at hs
        · simp at hs; subst hs
          have hb := h2 o b p hsl; subst hb
          refine ⟨rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
          · intro _ _ _ hh; cases hh
          · intro v hv; exact List.mem_cons_of_mem _ (h3 v hv)
          · simp [h4]
          · intro v hv; simp at hv
            rcases hv with rfl | hv
            · exact List.prefix_refl _
            · exact (h5 v hv).trans (List.prefix_append _ _)
          · simp [h6]
          · intro f hf
            have := (h7 f hf).1
            refine ⟨Nat.le_succ_of_le this, fun e => absurd e ?_⟩
            show f.2.1 ≠ s.ver + 1; omega
        · simp at hs; subst hs
          refine ⟨h1, ?_, h3, h4, h5, h6, h7⟩
          intro _ _ _ hh; cases hh
      · cases hs
    · cases hs
  | rollback t =>
    simp only [step] at hs; split at hs
    · split at hs
      · simp at hs; subst hs
        refine ⟨h1, ?_, h3, h4, h5, h6, h7⟩
        intro _ _ _ hh; cases hh
      · cases hs
    · cases hs
  | delete =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5, h6, h7⟩
    · cases hs
  | fork =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs
      exact ⟨h1, h2, seen_cons s h1 h3, h4, h5, h6, h7⟩
    · cases hs

theorem reach_inv {valid : List W → Bool} {s : S} (h : Reach valid s) : Inv s := by
  induction h with
  | init => exact inv_init
  | step e _ hs ih => exact inv_step valid e ih hs

/-- **Snapshot isolation / no partial commit.** Every snapshot a reader (or a BGSAVE
child) has ever taken is a version that was published whole by some commit. -/
theorem snapshots_are_committed {valid : List W → Bool} {s : S} (h : Reach valid s) : ∀ v ∈ s.seen, v ∈ s.hist :=
  (reach_inv h).seen_hist

/-- **No lost writes, nothing committed twice.** The committed version is exactly the
concatenation of every acknowledged write, each once, in acknowledgement order. -/
theorem no_lost_writes {valid : List W → Bool} {s : S} (h : Reach valid s) : s.acked = s.pub := (reach_inv h).acked_pub

/-- **History is linear.** Every published version is a prefix of the current one:
a later commit never drops or reorders an earlier one. -/
theorem history_linear {valid : List W → Bool} {s : S} (h : Reach valid s) : ∀ v ∈ s.hist, v <+: s.pub :=
  (reach_inv h).hist_prefix

/-- A private version is always built on the committed one: the single write slot
serializes writers (no write-write overlap, no lost update) — also when it was claimed
late, from a fork (#3290). -/
theorem private_on_committed {valid : List W → Bool} {s : S} (h : Reach valid s) (o : Tid) (b p : List W)
    (hsl : s.slot = some (o, b, p)) : b = s.pub := (reach_inv h).slot_base o b p hsl

/-- **At most one writer commits per version** (#3290). Each published version is one
commit: `ver` counts the versions in the history, and every commit takes version `ver` to
`ver + 1` building on `pub` (`private_on_committed`). After it, every fork made at an
earlier version is behind, so its claim can only be `Stale` (`claimFork_stale`). -/
theorem one_commit_per_version {valid : List W → Bool} {s s' : S} (h : Reach valid s) (t : Tid)
    (hs : step valid s (.commit t) = some s') :
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

/-- **A stale claim means a commit happened since the fork**, and it takes nothing: the
slot stays free and nothing is published or acknowledged. -/
theorem claimFork_stale {valid : List W → Bool} {s s' : S} (h : Reach valid s) (t : Tid)
    (hs : step valid s (.claimFork t) = some s') (hst : t ∈ s'.stale) (hn : t ∉ s.stale) :
    (∃ fv b, (t, fv, b) ∈ s.forks ∧ fv < s.ver) ∧ s'.slot = none ∧ s'.pub = s.pub ∧ s'.acked = s.acked := by
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
      · rename_i hsl
        split at hs
        · simp at hs; subst hs; exact absurd hst hn
        · rename_i hv
          simp at hs; subst hs
          exact ⟨⟨fv, b, hfm, by have := (hi.fork_ok _ hfm).1; simp at this; omega⟩, hsl, rfl, rfl⟩

/-- **The re-run sees the inline write** (#3290): a successful claim of a fork built at
the current version holds the slot on exactly the committed graph — every acknowledged
write, the inline one included. -/
theorem claimFork_on_committed {valid : List W → Bool} {s s' : S} (h : Reach valid s) (t : Tid)
    (hs : step valid s (.claimFork t) = some s') (o : Tid) (b p : List W) (hsl : s'.slot = some (o, b, p)) :
    b = s.pub ∧ b = s.acked := by
  have hs' := Reach.step _ h hs
  have := private_on_committed hs' o b p hsl
  have hp : s'.pub = s.pub := by
    simp only [step] at hs; split at hs
    · cases hs
    · split at hs
      · cases hs
      · split at hs
        · simp at hs; subst hs; rfl
        · split at hs <;> (simp at hs; subst hs; rfl)
  rw [hp] at this
  exact ⟨this, this.trans (no_lost_writes h).symm⟩

/-- **GRAPH.DELETE during running queries.** Once the key is gone no write reaches
the graph: `mutate` and `commit` are disabled. (Readers are unaffected: their
snapshots are `Arc`s, freed only when the last one drops.) -/
theorem delete_blocks_writes (valid : List W → Bool) (s : S) (h : s.registered = false) (t : Tid) (w : W) :
    step valid s (.mutate t w) = none ∧ step valid s (.commit t) = none := by
  constructor <;> simp [step] <;> split <;> simp_all

/-- **Fork safety.** A fork needs the GIL; a commit needs the GIL; so a fork child
never runs between the two halves of a commit — its snapshot is `pub`, which by
`snapshots_are_committed` is a whole version. -/
theorem fork_excludes_commit (valid : List W → Bool) (s : S) (t : Tid) (ht : t ≠ 0) (hg : s.gil = some t) :
    step valid s .fork = none ∧ step valid s .delete = none := by
  simp [step, hg, ht]

/-- **A refused version releases the write slot** (#2846, `MvccGraph::commit`, test
`a_refused_commit_releases_the_write_slot`): nothing is published or acknowledged, and
the next writer (holding the write lock) can claim the slot. -/
theorem refused_commit_releases_slot (valid : List W → Bool) (s : S) (t : Tid) (b p : List W)
    (hsl : s.slot = some (t, b, p)) (hg : s.gil = some t) (hr : s.registered = true)
    (hv : valid (b ++ p) = false) :
    ∃ s', step valid s (.commit t) = some s' ∧ s'.slot = none ∧ s'.pub = s.pub ∧ s'.acked = s.acked ∧
      s'.hist = s.hist ∧ (step valid s' (.claim t)).isSome := by
  refine ⟨{ s with slot := none }, ?_, rfl, rfl, rfl, rfl, by simp [step, hg]⟩
  simp [step, hsl, hg, hr, hv]

/-- A version `validate` accepts is published exactly as before #2846. -/
theorem valid_commit_publishes (valid : List W → Bool) (s : S) (t : Tid) (b p : List W)
    (hsl : s.slot = some (t, b, p)) (hg : s.gil = some t) (hr : s.registered = true)
    (hv : valid (b ++ p) = true) :
    step valid s (.commit t) =
      some { s with pub := b ++ p, ver := s.ver + 1, slot := none, hist := (b ++ p) :: s.hist,
                    acked := s.acked ++ p } := by
  simp [step, hsl, hg, hr, hv]

/-! ## The MULTI write while a queued write matches (#3093, fixed by #3290)

Queued write `1` forks as a reader and matches; an inline writer `0` (`MULTI`/`EXEC`,
`query_sync` → `begin_writer`, main thread, GIL and write lock) runs in that window. -/

def run (s : S) : List Ev → Option S
  | [] => some s
  | e :: es => (step (fun _ => true) s e).bind (run · es)

/-- The inline writer commits `7` while `1` is still matching. -/
def inlineTrace : List Ev := [.forkW 1, .escalate 0, .claim 0, .mutate 0 7, .commit 0, .release 0]

/-- **Fixed (#3290): the inline write succeeds** while the queued write is matching (C's
behaviour), and is published. -/
theorem inline_write_ok :
    ((run init inlineTrace).map (·.pub)) = some [7] := by decide

/-- …the queued write then escalates, finds its fork stale, takes nothing, and re-runs:
a fresh fork, claimed, builds on the inline write. Final state: `7` then `8`, each
committed once, both acknowledged. -/
def rerunTrace : List Ev :=
  inlineTrace ++ [.escalate 1, .claimFork 1, .release 1,          -- `Stale`: `WriteAttempt::Stale`
                  .forkW 1, .escalate 1, .claimFork 1, .mutate 1 8, .commit 1, .release 1]

theorem stale_then_rerun :
    ((run init (inlineTrace ++ [.escalate 1, .claimFork 1])).map (fun s => (s.stale, s.slot))) =
      some ([1], none) ∧
    ((run init rerunTrace).map (fun s => (s.pub, s.acked, s.ver))) = some ([7, 8], [7, 8], 2) := by
  decide

/-! Historical: the pre-#3290 protocol (4a4f2a72f). `execute_query_write` claimed the
slot as a **reader** (old `graph_core.rs:748`, `MvccGraph::write` with no lock held), ran
the whole match phase, and took the GIL only at the first mutation. An inline writer in
that window took the per-graph write lock fine and then failed the CAS: "another write is
in progress, retry the query" (BUG 5, confirmed live 3/3; C creates the node). -/

def stepPre3290 (s : S) : Ev → Option S
  | .claim t => match s.slot with
    | none => some { s with slot := some (t, s.pub, []) }
    | some _ => none
  | e => step (fun _ => true) s e

def runPre3290 (s : S) : List Ev → Option S
  | [] => some s
  | e :: es => (stepPre3290 s e).bind (runPre3290 · es)

theorem pre3290_multi_write_refused :
    (runPre3290 init [.claim 1, .escalate 0]).isSome ∧
    ((runPre3290 init [.claim 1, .escalate 0]).bind (stepPre3290 · (.claim 0))).isNone := by
  decide

/-- In the C protocol (no slot; claim = GIL), the same interleaving lets the inline
writer through. -/
def stepC (s : S) : Ev → Option S
  | .claim _ => some s
  | e => step (fun _ => true) s e

theorem multi_write_ok_in_c :
    ((run init [.escalate 0]).bind (stepC · (.claim 0))).isSome := by decide

/-- Since #3290 the reader-mode claim of the old trace is no longer possible at all: the
slot is only claimed with the write lock (`Ev.claim` needs the GIL). -/
theorem reader_cannot_claim : (run init [.claim 1]).isNone := by decide

end SC.Mvcc
