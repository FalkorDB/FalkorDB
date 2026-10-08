/-
# MVCC publish/read, write slot, GIL, GRAPH.DELETE and BGSAVE fork

A transition system over one graph. Each event is one atomic step of the Rust code;
the interleaving is arbitrary (`Step` is a relation, any enabled event may fire).

| here | there |
| --- | --- |
| `S.pub`                    | `MvccGraph.graph` (the committed `Arc`, `graph/src/graph/mvcc_graph.rs:70`) |
| `S.slot`                   | `MvccGraph.write: AtomicBool` (`:72`) together with the private version it hands out |
| `Ev.read`                  | `MvccGraph::read` (`:103`) — clone of the committed Arc |
| `Ev.claim`                 | `MvccGraph::write` (`:108`) — CAS false→true, `new_version()` of the committed graph |
| `Ev.mutate`                | any mutation on the private version (runtime `Commit` op) — requires writer mode, `query_session.rs:221` |
| `Ev.commit`                | `commit_and_replicate` → `MvccGraph::commit` (`src/graph_core.rs:1497`, `mvcc_graph.rs:139`): since #2846 `Graph::validate` first (`valid`, :148); refused ⇒ `rollback` + `Err` (:148-151), else Arc swap + `store(false)` (:216-217), in writer mode |
| `Ev.rollback`              | `MvccGraph::rollback` (`:221`) via `abandon_write`/`finish_write` (`graph_core.rs:1423,1453`) |
| `S.gil`, `Ev.escalate`     | `QuerySession::escalate` (`query_session.rs:242`): drop read lock, `Gil::acquire`, `write_arc` |
| `S.registered`             | `GRAPH_REGISTRY` membership (`graph_core.rs:136`, `graph_is_registered` `:188`) |
| `Ev.delete`                | `GRAPH.DELETE` (`src/commands/delete.rs:37`) → `graph_free` (`graph_core.rs:1671`), main thread, GIL held |
| `Ev.fork`                  | BGSAVE `fork()` on the main thread with the GIL (`redis_type.rs:350` `pre_fork_prepare`, `module_init.rs:120` `on_fork_child`) |

Transactions are lists of abstract writes; a version is the list of all writes it
contains, in order. The private version is `base ++ pending`.
-/

namespace SC.Mvcc

abbrev W := Nat
abbrev Tid := Nat

structure S where
  /-- committed version -/
  pub : List W
  /-- write slot: owner, the base it was claimed on, its pending writes -/
  slot : Option (Tid × List W × List W)
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
  | escalate (t : Tid)
  | release (t : Tid)
  | mutate (t : Tid) (w : W)
  | commit (t : Tid)
  | rollback (t : Tid)
  | delete
  | fork
deriving Repr

/-- One atomic step. `none` = not enabled. `valid` is `Graph::validate`
(graph.rs:1507) on the private version. -/
def step (valid : List W → Bool) (s : S) : Ev → Option S
  | .read _ => some { s with seen := s.pub :: s.seen }
  | .claim t => match s.slot with
    | none => some { s with slot := some (t, s.pub, []) }
    | some _ => none                                   -- CAS fails
  | .escalate t => match s.gil with
    | none => some { s with gil := some t }
    | some _ => none                                   -- blocks
  | .release t => if s.gil = some t then some { s with gil := none } else none
  | .mutate t w => match s.slot with
    | some (o, b, p) =>
      -- writer mode is required, and escalation re-checks the registry (query_session.rs:227)
      if o = t ∧ s.gil = some t ∧ s.registered then some { s with slot := some (o, b, p ++ [w]) } else none
    | none => none
  | .commit t => match s.slot with
    | some (o, b, p) =>
      if o = t ∧ s.gil = some t ∧ s.registered then
        if valid (b ++ p) then
          some { s with pub := b ++ p, slot := none, hist := (b ++ p) :: s.hist, acked := s.acked ++ p }
        else some { s with slot := none }            -- refused: `self.rollback()`, nothing published
      else none
    | none => none
  | .rollback t => match s.slot with
    | some (o, _, _) => if o = t then some { s with slot := none } else none
    | none => none
  | .delete => if s.gil = none ∨ s.gil = some 0 then some { s with registered := false } else none
  | .fork => if s.gil = none ∨ s.gil = some 0 then some { s with seen := s.pub :: s.seen } else none

def init : S := { pub := [], slot := none, gil := none, registered := true, hist := [[]], seen := [], acked := [] }

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

theorem inv_init : Inv init := by
  refine ⟨rfl, ?_, ?_, rfl, ?_⟩
  · intro o b p h; cases h
  · intro v h; cases h
  · intro v h; simp [init] at h; subst h; exact List.nil_prefix

theorem inv_step (valid : List W → Bool) {s s' : S} (e : Ev) (h : Inv s) (hs : step valid s e = some s') :
    Inv s' := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  cases e with
  | read t =>
    simp [step] at hs; subst hs
    refine ⟨h1, h2, ?_, h4, h5⟩
    intro v hv; simp at hv
    rcases hv with rfl | hv
    · cases hh : s.hist with
      | nil => simp [hh] at h1
      | cons x xs => simp [hh] at h1; subst h1; simp
    · exact h3 v hv
  | claim t =>
    simp only [step] at hs
    split at hs
    · simp at hs; subst hs
      refine ⟨h1, ?_, h3, h4, h5⟩
      intro o b p hh; simp at hh; exact hh.2.1.symm
    · cases hs
  | escalate t =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5⟩
    · cases hs
  | release t =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5⟩
    · cases hs
  | mutate t w =>
    simp only [step] at hs; split at hs
    · rename_i o b p hsl
      split at hs
      · simp at hs; subst hs
        refine ⟨h1, ?_, h3, h4, h5⟩
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
          refine ⟨rfl, ?_, ?_, ?_, ?_⟩
          · intro _ _ _ hh; cases hh
          · intro v hv; exact List.mem_cons_of_mem _ (h3 v hv)
          · simp [h4]
          · intro v hv; simp at hv
            rcases hv with rfl | hv
            · exact List.prefix_refl _
            · exact (h5 v hv).trans (List.prefix_append _ _)
        · simp at hs; subst hs
          refine ⟨h1, ?_, h3, h4, h5⟩
          intro _ _ _ hh; cases hh
      · cases hs
    · cases hs
  | rollback t =>
    simp only [step] at hs; split at hs
    · split at hs
      · simp at hs; subst hs
        refine ⟨h1, ?_, h3, h4, h5⟩
        intro _ _ _ hh; cases hh
      · cases hs
    · cases hs
  | delete =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs; exact ⟨h1, h2, h3, h4, h5⟩
    · cases hs
  | fork =>
    simp only [step] at hs; split at hs
    · simp at hs; subst hs
      refine ⟨h1, h2, ?_, h4, h5⟩
      intro v hv; simp at hv
      rcases hv with rfl | hv
      · cases hh : s.hist with
        | nil => simp [hh] at h1
        | cons x xs => simp [hh] at h1; subst h1; simp
      · exact h3 v hv
    · cases hs

theorem reach_inv {valid : List W → Bool} {s : S} (h : Reach valid s) : Inv s := by
  induction h with
  | init => exact inv_init
  | step e _ hs ih => exact inv_step valid e ih hs

/-- **Snapshot isolation / no partial commit.** Every snapshot a reader (or a BGSAVE
child) has ever taken is a version that was published whole by some commit. -/
theorem snapshots_are_committed {valid : List W → Bool} {s : S} (h : Reach valid s) : ∀ v ∈ s.seen, v ∈ s.hist :=
  (reach_inv h).seen_hist

/-- **No lost writes.** The committed version is exactly the concatenation of every
acknowledged write, in acknowledgement order. -/
theorem no_lost_writes {valid : List W → Bool} {s : S} (h : Reach valid s) : s.acked = s.pub := (reach_inv h).acked_pub

/-- **History is linear.** Every published version is a prefix of the current one:
a later commit never drops or reorders an earlier one. -/
theorem history_linear {valid : List W → Bool} {s : S} (h : Reach valid s) : ∀ v ∈ s.hist, v <+: s.pub :=
  (reach_inv h).hist_prefix

/-- A private version is always built on the committed one: the single write slot
serializes writers (no write-write overlap, no lost update). -/
theorem private_on_committed {valid : List W → Bool} {s : S} (h : Reach valid s) (o : Tid) (b p : List W)
    (hsl : s.slot = some (o, b, p)) : b = s.pub := (reach_inv h).slot_base o b p hsl

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

/-- **A refused version releases the write slot** (#2846, `MvccGraph::commit`
:148-151, test `a_refused_commit_releases_the_write_slot`): nothing is
published or acknowledged, and the next writer can claim the slot. Without the
`rollback` the single `AtomicBool` would refuse every later write. -/
theorem refused_commit_releases_slot (valid : List W → Bool) (s : S) (t : Tid) (b p : List W)
    (hsl : s.slot = some (t, b, p)) (hg : s.gil = some t) (hr : s.registered = true)
    (hv : valid (b ++ p) = false) (t' : Tid) :
    ∃ s', step valid s (.commit t) = some s' ∧ s'.slot = none ∧ s'.pub = s.pub ∧ s'.acked = s.acked ∧
      s'.hist = s.hist ∧ (step valid s' (.claim t')).isSome := by
  refine ⟨{ s with slot := none }, ?_, rfl, rfl, rfl, rfl, by simp [step]⟩
  simp [step, hsl, hg, hr, hv]

/-- A version `validate` accepts is published exactly as before #2846. -/
theorem valid_commit_publishes (valid : List W → Bool) (s : S) (t : Tid) (b p : List W)
    (hsl : s.slot = some (t, b, p)) (hg : s.gil = some t) (hr : s.registered = true)
    (hv : valid (b ++ p) = true) :
    step valid s (.commit t) =
      some { s with pub := b ++ p, slot := none, hist := (b ++ p) :: s.hist, acked := s.acked ++ p } := by
  simp [step, hsl, hg, hr, hv]

/-! ## The write-slot window: the MULTI failure

`execute_query_write` claims the slot as a **reader** (graph_core.rs:748), runs the
whole match phase, and only takes the GIL at the first mutation. An inline writer
(`MULTI`/`EXEC`, `query_sync` → `begin_writer`, main thread, GIL implicit) that runs
in that window takes the per-graph write lock fine and then fails the CAS. C has no
slot: its writer takes GIL then write lock, and the inline write simply goes first. -/

def multiTrace : List Ev := [.claim 1, .escalate 0, .claim 0]

def run (s : S) : List Ev → Option S
  | [] => some s
  | e :: es => (step (fun _ => true) s e).bind (run · es)

theorem multi_write_refused :
    (run init [.claim 1, .escalate 0]).isSome ∧
    ((run init [.claim 1, .escalate 0]).bind (step (fun _ => true) · (.claim 0))).isNone := by
  decide

/-- In the C protocol (no slot; claim = GIL), the same interleaving lets the inline
writer through: `claimC` is just `escalate`. -/
def stepC (s : S) : Ev → Option S
  | .claim _ => some s
  | e => step (fun _ => true) s e

theorem multi_write_ok_in_c :
    ((run init [.escalate 0]).bind (stepC · (.claim 0))).isSome := by decide

end SC.Mvcc
