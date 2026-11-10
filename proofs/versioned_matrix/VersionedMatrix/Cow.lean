/-
# Copy-on-write versions: published snapshots are never written

Model of `graph/src/graph/cow.rs` and `graph/src/graph/mvcc_graph.rs` (fork-and-commit since
#3657, 34a708cfa: no write slot, no `rollback`).

A GraphBLAS handle is a location in a heap `Nat → α` (its contents: for a
matrix, the coordinate set of `Delta.lean`). `Matrix::clone` shares the
handle (`Arc<Handle>`, `matrix.rs:405`), `Matrix::dup` allocates a fresh
one (`matrix.rs:1209`). A `Cow` is a handle plus the `dup` flag.

| here | there |
| --- | --- |
| `Cow`            | `struct Cow { inner, dup }` (cow.rs:60; the debug-only `branched` flag of #2905 is not modelled here, see `runtime_ds` `CowModel.dbg_*`) |
| `Cow.newVersion` | `Cow::new_version` (cow.rs:97): shallow clone, `dup = true` |
| `derefMut`       | `DerefMut for Cow` (cow.rs:134): dup on first write |
| `replace`        | `Cow::replace` (cow.rs:112): fresh contents, `dup = false` |
| `mutate`         | any `&mut Matrix` write through a `Cow` (`Delta::layer_mut`, `insert`, `erase`, `resize`, ...) |
| `Version`        | a graph version: every `Cow` it owns (`Graph::new_version` dups every matrix, graph.rs:1084) |
| `Mvcc`           | `MvccGraph` (mvcc_graph.rs:74): the published version |
| `Mvcc.begin`     | `MvccGraph::fork` (:130): `new_version` of the published graph, takes nothing |
| `Mvcc.commit`    | `MvccGraph::commit` (:205) → `commit_version` (:261): `Graph::validate` (`valid`) — refused ⇒ `Err`, the version is dropped, published unchanged; else publish. (The staleness refusal is in `search_concurrency` `Mvcc`; isolation does not need it.) |
| `Mvcc.abandon`   | dropping an uncommitted version (no `rollback` since #3657) |

`Graph::new_version` goes through `VersionedMatrix::dup`/`Tensor::dup`, which
call `Cow::new_version` / `Delta::new_version` for every layer — so a write
version is exactly `Version.newVersion` of the committed one.

The fold at commit (`fold_oversized_deltas`) and every other write the
committing writer makes still go through `mutate`/`replace` on the *write*
version, so they are covered.
-/

namespace VMCow

variable {α : Type}

structure Cow where
  h : Nat
  dup : Bool
deriving DecidableEq, Repr

def Cow.newVersion (c : Cow) : Cow := ⟨c.h, true⟩

/-- The heap of GraphBLAS objects, with an allocation pointer. -/
structure Heap (α : Type) where
  mem : Nat → α
  fresh : Nat

def Heap.write (H : Heap α) (h : Nat) (x : α) : Heap α :=
  { H with mem := fun k => if k = h then x else H.mem k }

/-- `deref_mut`: duplicate a shared handle into a fresh one first. -/
def derefMut (H : Heap α) (c : Cow) : Heap α × Cow :=
  if c.dup then (⟨fun k => if k = H.fresh then H.mem c.h else H.mem k, H.fresh + 1⟩, ⟨H.fresh, false⟩)
  else (H, c)

/-- A write through the `Cow`: `deref_mut`, then mutate the handle in place. -/
def mutate (H : Heap α) (c : Cow) (f : α → α) : Heap α × Cow :=
  let (H', c') := derefMut H c
  (H'.write c'.h (f (H'.mem c'.h)), c')

/-- `Cow::replace`: freshly built contents in a fresh handle. -/
def replace (H : Heap α) (_c : Cow) (x : α) : Heap α × Cow :=
  (⟨fun k => if k = H.fresh then x else H.mem k, H.fresh + 1⟩, ⟨H.fresh, false⟩)

/-- A version: its `Cow`s, indexed by matrix slot. -/
abbrev Version := Nat → Cow

def Version.newVersion (V : Version) : Version := fun i => (V i).newVersion

/-- What a reader of version `V` sees. -/
def view (H : Heap α) (V : Version) : Nat → α := fun i => H.mem (V i).h

/-- A writer's step on its own version. -/
inductive WOp (α : Type) where
  | mutate (i : Nat) (f : α → α)
  | replace (i : Nat) (x : α)

def wstep (H : Heap α) (W : Version) : WOp α → Heap α × Version
  | .mutate i f => let r := mutate H (W i) f; (r.1, fun j => if j = i then r.2 else W j)
  | .replace i x => let r := replace H (W i) x; (r.1, fun j => if j = i then r.2 else W j)

/-- `Pub h`: handle `h` is referenced by some published snapshot. The writer
may write in place only to handles it owns (`dup = false`) and that no
snapshot references. -/
structure Iso (H : Heap α) (Pub : Nat → Prop) (W : Version) : Prop where
  pub_lt : ∀ h, Pub h → h < H.fresh
  w_lt : ∀ i, (W i).h < H.fresh
  owned_private : ∀ i, (W i).dup = false → ¬ Pub (W i).h

theorem wstep_iso {H : Heap α} {Pub : Nat → Prop} {W : Version} (hi : Iso H Pub W) (op : WOp α) :
    let r := wstep H W op
    Iso r.1 Pub r.2 ∧ ∀ h, Pub h → r.1.mem h = H.mem h := by
  obtain ⟨p1, p2, p3⟩ := hi
  cases op with
  | mutate i f =>
    simp only [wstep, mutate, derefMut]
    by_cases hd : (W i).dup = true
    · simp only [hd, ite_true, Heap.write]
      refine ⟨⟨fun h hp => by have := p1 h hp; simp; omega, fun j => ?_, fun j hj => ?_⟩, fun h hp => ?_⟩
      · by_cases hj : j = i <;> simp [hj]
        have := p2 j; omega
      · by_cases hji : j = i
        · simp [hji]; intro hp; have := p1 _ hp; omega
        · simp [hji] at hj ⊢; exact p3 j hj
      · have := p1 h hp
        have h1 : h ≠ H.fresh := by omega
        simp [h1]
    · have hd' : (W i).dup = false := by simpa using hd
      simp only [hd', Bool.false_eq_true, ite_false, Heap.write]
      refine ⟨⟨p1, fun j => ?_, fun j hj => ?_⟩, fun h hp => ?_⟩
      · by_cases hj : j = i <;> simp [hj, p2]
      · by_cases hji : j = i
        · simp [hji]; exact p3 i hd'
        · simp [hji] at hj ⊢; exact p3 j hj
      · have : h ≠ (W i).h := fun e => p3 i hd' (e ▸ hp)
        simp [this]
  | replace i x =>
    simp only [wstep, replace]
    refine ⟨⟨fun h hp => by have := p1 h hp; simp; omega, fun j => ?_, fun j hj => ?_⟩, fun h hp => ?_⟩
    · by_cases hj : j = i <;> simp [hj]
      have := p2 j; omega
    · by_cases hji : j = i
      · simp [hji]; intro hp; have := p1 _ hp; omega
      · simp [hji] at hj ⊢; exact p3 j hj
    · have := p1 h hp
      have h1 : h ≠ H.fresh := by omega
      simp [h1]

/-- A fresh write version (`MvccGraph::fork` → `Graph::new_version`) starts
with every layer shared (`dup = true`), so `Iso` holds for any `Pub` that
already contains the committed version's handles. -/
theorem newVersion_iso {H : Heap α} {Pub : Nat → Prop} (C : Version)
    (hp : ∀ h, Pub h → h < H.fresh) (hc : ∀ i, (C i).h < H.fresh) :
    Iso H Pub C.newVersion :=
  ⟨hp, fun i => hc i, fun i hi => by simp [Version.newVersion, Cow.newVersion] at hi⟩

/-! ### MVCC -/

structure Mvcc where
  committed : Version

/-- A whole history: writers fork, write, commit or abandon; readers take
snapshots (`MvccGraph::read` clones the `Arc` of the committed version). -/
inductive Ev (α : Type) where
  | begin                   -- `fork()`
  | op (o : WOp α)          -- a mutation by the current writer
  | commit                  -- `commit(new_graph)`
  | abandon                 -- the uncommitted version is dropped

structure Sys (α : Type) where
  H : Heap α
  g : Mvcc
  W : Option Version        -- the in-flight write version, if any

/-- `valid` is `Graph::validate` (graph.rs:1629) on the write version.

One in-flight write version only (`W : Option Version`): `.begin` while a version is
already in flight replaces it, as if it had been abandoned first, without an `.abandon`
event. The engine can hold several forks at once (a writer may fork while still
matching as a reader, and the bulk-insert phase-1 fork): that is not modelled here. Each
such fork is created by `new_version` of the committed version and is written only by its
own writer, so the one-directional-COW argument below applies to each; but the proof
covers histories with one in-flight version at a time, and the stale check
(`is_stale`) that orders concurrent forks' commits is in `search_concurrency` `Mvcc`. -/
def sstep (valid : Version → Bool) (S : Sys α) : Ev α → Sys α
  | .begin => { S with W := some S.g.committed.newVersion }
  | .op o => match S.W with
    | none => S
    | some W => let r := wstep S.H W o; { S with H := r.1, W := some r.2 }
  | .commit => match S.W with
    | none => S
    | some W => if valid W then { S with g := ⟨W⟩, W := none }
      else { S with W := none }   -- refused: `Err`, the version is dropped
  | .abandon => { S with W := none }

/-- Handles published so far: every version that was ever committed. -/
def Published (S : Sys α) (Pub : Nat → Prop) : Prop :=
  (∀ i, Pub (S.g.committed i).h) ∧ (∀ h, Pub h → h < S.H.fresh) ∧
  (∀ i, (S.g.committed i).h < S.H.fresh) ∧
  (∀ W, S.W = some W → Iso S.H Pub W)

/-- **Snapshot isolation.** Along any history, the contents of every handle a
published snapshot references never change, so a reader holding any
committed version sees the same matrices forever — through later
mutations, commits, new writers and abandoned versions. -/
theorem snapshot_isolation (valid : Version → Bool) : ∀ (evs : List (Ev α)) (S : Sys α) (Pub : Nat → Prop),
    Published S Pub →
    ∃ Pub' : Nat → Prop, (∀ h, Pub h → Pub' h) ∧
      ∀ h, Pub h → (evs.foldl (sstep valid) S).H.mem h = S.H.mem h
  | [], S, Pub, _ => ⟨Pub, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
  | e :: es, S, Pub, hS => by
    obtain ⟨c1, c2, c3, c4⟩ := hS
    -- the published set after `e`: commit publishes the write version's handles
    let Pub1 : Nat → Prop := match e, S.W with
      | .commit, some W => fun h => Pub h ∨ ∃ i, (W i).h = h
      | _, _ => Pub
    have step : Published (sstep valid S e) Pub1 ∧ (∀ h, Pub h → Pub1 h) ∧
        ∀ h, Pub h → (sstep valid S e).H.mem h = S.H.mem h := by
      cases e with
      | begin =>
        simp only [sstep, Pub1]
        refine ⟨⟨c1, c2, c3, ?_⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
        intro W hW; simp at hW; subst hW; exact newVersion_iso _ c2 c3
      | op o =>
        simp only [sstep, Pub1]
        cases hW : S.W with
        | none => simp only; exact ⟨⟨c1, c2, c3, c4⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
        | some W =>
          simp only
          have hi := c4 W hW
          have := wstep_iso hi o
          refine ⟨⟨c1, this.1.pub_lt, ?_, ?_⟩, fun _ h => h, this.2⟩
          · intro i
            have hfr : S.H.fresh ≤ (wstep S.H W o).1.fresh := by
              cases o <;> simp [wstep, mutate, derefMut, replace, Heap.write] <;> split <;> simp
            show (S.g.committed i).h < (wstep S.H W o).1.fresh
            have := c3 i; omega
          · intro W' hW'; simp at hW'; subst hW'; exact this.1
      | commit =>
        simp only [sstep, Pub1]
        cases hW : S.W with
        | none => simp only; exact ⟨⟨c1, c2, c3, c4⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
        | some W =>
          simp only
          have hi := c4 W hW
          split
          · refine ⟨⟨fun i => Or.inr ⟨i, rfl⟩, ?_, hi.w_lt, by simp⟩,
              fun _ h => Or.inl h, fun _ _ => by first | rfl | trivial⟩
            intro h hp; rcases hp with hp | ⟨i, rfl⟩
            · exact c2 h hp
            · exact hi.w_lt i
          · refine ⟨⟨fun i => Or.inl (c1 i), ?_, c3, by simp⟩,
              fun _ h => Or.inl h, fun _ _ => by first | rfl | trivial⟩
            intro h hp; rcases hp with hp | ⟨i, rfl⟩
            · exact c2 h hp
            · exact hi.w_lt i
      | abandon =>
        simp only [sstep, Pub1]
        exact ⟨⟨c1, c2, c3, by simp⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
    obtain ⟨hP1, hsub, hmem⟩ := step
    obtain ⟨Pub2, hsub2, hmem2⟩ := snapshot_isolation valid es (sstep valid S e) Pub1 hP1
    refine ⟨Pub2, fun h hp => hsub2 h (hsub h hp), fun h hp => ?_⟩
    simp only [List.foldl]
    rw [hmem2 h (hsub h hp), hmem h hp]

/-- Hence: a reader's view of any version committed so far is frozen. -/
theorem reader_view_frozen (valid : Version → Bool) (evs : List (Ev α)) (S : Sys α) (Pub : Nat → Prop)
    (hS : Published S Pub) :
    view (evs.foldl (sstep valid) S).H S.g.committed = view S.H S.g.committed := by
  obtain ⟨_, _, hmem⟩ := snapshot_isolation valid evs S Pub hS
  funext i; exact hmem _ (hS.1 i)

/-- A fork takes nothing: `begin` leaves the published version as it is (no flag, so no
writer can be refused because another is building, #3093). -/
theorem fork_takes_nothing (valid : Version → Bool) (S : Sys α) :
    (sstep valid S .begin).g = S.g ∧ (sstep valid S .begin).H = S.H := by
  simp [sstep]

/-- **A refused version is never published** (`MvccGraph::commit` returns `Err`): the
published version is unchanged and the version is dropped (nothing to roll back), so the
next `begin` forks as usual. -/
theorem refused_commit_publishes_nothing (valid : Version → Bool) (S : Sys α) (W : Version)
    (hW : S.W = some W) (hv : valid W = false) :
    (sstep valid S .commit).g.committed = S.g.committed ∧ (sstep valid S .commit).W = none ∧
    (sstep valid S .commit).H = S.H := by
  simp [sstep, hW, hv]

/-! ### Where isolation stops: sharing an *owned* handle -/

/-- `Clone` (`VersionedMatrix`/`Tensor`/`Cow` derive a shallow clone that
copies the `dup` flag) of an owned layer shares a writable handle: a write
through one is visible through the other. This is why version creation must
use `dup`/`new_version`, never `clone`. Same mechanism as the confirmed
`probe_iter_outlives_in_place_mutation`: `versioned_matrix::Iter` holds its
layer's handle (an `Arc<GrB_Matrix>`) with no borrow of the owner, so an
in-place write to an owned layer lands under a live iterator. -/
theorem clone_of_owned_is_not_isolated :
    let H : Heap Nat := ⟨fun _ => 0, 1⟩
    let c : Cow := ⟨0, false⟩
    let c2 := c                      -- `clone()`
    let r := mutate H c (fun _ => 7)
    r.1.mem c2.h = 7 := by
  decide

/-- Copy-on-write is **one-directional**: `new_version` marks only the *new*
`Cow` as `dup = true`; the old one keeps `dup = false` if it owned its handle.
A later in-place write through the *old* version therefore lands in the
handle the new version still shares. `snapshot_isolation` is sound because
its history (`sstep`) never writes a version after it has been
`new_version`ed: only the in-flight `W` is written, and `W` is only ever
`new_version`ed after it is committed (by the next `begin`). In the engine,
that is `MvccGraph::fork`/`commit`: the committed graph is only ever read
(`read()` hands out `Arc` clones; read queries never fork), and
every mutation path — `Runtime` for write queries, GRAPH.BULK, index DDL — goes
through the version returned by `fork()`, except the indexer batch, which is applied to
a build version of a published one (`Graph::new_build_version`, graph.rs:1139, via
`Graph::indexer_prepare`) and published by `commit_indexer_batch`. A build version is
made the same way (`new_version`-style dup of every layer), so it is another instance of
the model's write version `W`; `commit_indexer_batch` publishes it through the same
`commit_version`. Reproduced at the API level
by `cow_is_one_directional_old_version_write_leaks` in
`graph/tests/lean_versioned_matrix.rs`. Status on main (#2905, #2903): the
contract is now documented on `Cow`, and debug builds assert it in `deref_mut`
(the `branched` flag, shared by every clone; `runtime_ds`
`CowModel.dbg_write_through_branched_source_panics`). Release behaviour is
unchanged, so this release-semantics counterexample still stands as a
documented latent contract. -/
theorem old_write_leaks_into_new_version :
    let H : Heap Nat := ⟨fun _ => 0, 1⟩
    let old : Cow := ⟨0, false⟩          -- the owned layer of a version
    let new := old.newVersion            -- the next version's layer
    let r := mutate H old (fun _ => 7)   -- a write through the OLD version
    r.1.mem new.h = 7 := by
  decide

end VMCow
