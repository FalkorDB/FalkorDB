/-
# Copy-on-write versions: published snapshots are never written

Model of `graph/src/graph/cow.rs` and `graph/src/graph/mvcc_graph.rs`.

A GraphBLAS handle is a location in a heap `Nat → α` (its contents: for a
matrix, the coordinate set of `Delta.lean`). `Matrix::clone` shares the
handle (`Arc<GrB_Matrix>`, `matrix.rs:374`), `Matrix::dup` allocates a fresh
one (`matrix.rs:1130`). A `Cow` is a handle plus the `dup` flag.

| here | there |
| --- | --- |
| `Cow`            | `struct Cow { inner, dup }` (cow.rs:43) |
| `Cow.newVersion` | `Cow::new_version` (cow.rs:55): shallow clone, `dup = true` |
| `derefMut`       | `DerefMut for Cow` (cow.rs:84): dup on first write |
| `replace`        | `Cow::replace` (cow.rs:66): fresh contents, `dup = false` |
| `mutate`         | any `&mut Matrix` write through a `Cow` (`Delta::layer_mut`, `insert`, `erase`, `resize`, ...) |
| `Version`        | a graph version: every `Cow` it owns (`Graph::new_version` dups every matrix, graph.rs:960) |
| `Mvcc`           | `MvccGraph` (mvcc_graph.rs:68): committed version + `write` flag |
| `Mvcc.write`     | `MvccGraph::write` (:108): CAS, then `new_version` |
| `Mvcc.commit`    | `MvccGraph::commit` (:139): since #2846 `Graph::validate` first (`valid`, :148) — refused ⇒ `rollback` (:149); else publish, clear the flag |
| `Mvcc.rollback`  | `MvccGraph::rollback` (:221) |

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

/-- A fresh write version (`MvccGraph::write` → `Graph::new_version`) starts
with every layer shared (`dup = true`), so `Iso` holds for any `Pub` that
already contains the committed version's handles. -/
theorem newVersion_iso {H : Heap α} {Pub : Nat → Prop} (C : Version)
    (hp : ∀ h, Pub h → h < H.fresh) (hc : ∀ i, (C i).h < H.fresh) :
    Iso H Pub C.newVersion :=
  ⟨hp, fun i => hc i, fun i hi => by simp [Version.newVersion, Cow.newVersion] at hi⟩

/-! ### MVCC -/

structure Mvcc where
  committed : Version
  writing : Bool

/-- A whole history: writers begin, write, commit or roll back; readers take
snapshots (`MvccGraph::read` clones the `Arc` of the committed version). -/
inductive Ev (α : Type) where
  | begin                   -- `write()`
  | op (o : WOp α)          -- a mutation by the current writer
  | commit                  -- `commit(new_graph)`
  | rollback

structure Sys (α : Type) where
  H : Heap α
  g : Mvcc
  W : Option Version        -- the in-flight write version, if any

/-- `valid` is `Graph::validate` (graph.rs:1479) on the write version. -/
def sstep (valid : Version → Bool) (S : Sys α) : Ev α → Sys α
  | .begin => if S.g.writing then S else { S with g := { S.g with writing := true }, W := some S.g.committed.newVersion }
  | .op o => match S.W with
    | none => S
    | some W => let r := wstep S.H W o; { S with H := r.1, W := some r.2 }
  | .commit => match S.W with
    | none => S
    | some W => if valid W then { S with g := ⟨W, false⟩, W := none }
      else { S with g := { S.g with writing := false }, W := none }   -- refused: `self.rollback()`
  | .rollback => { S with g := { S.g with writing := false }, W := none }

/-- Handles published so far: every version that was ever committed. -/
def Published (S : Sys α) (Pub : Nat → Prop) : Prop :=
  (∀ i, Pub (S.g.committed i).h) ∧ (∀ h, Pub h → h < S.H.fresh) ∧
  (∀ i, (S.g.committed i).h < S.H.fresh) ∧
  (∀ W, S.W = some W → Iso S.H Pub W) ∧ (S.W.isSome → S.g.writing = true)

/-- **Snapshot isolation.** Along any history, the contents of every handle a
published snapshot references never change, so a reader holding any
committed version sees the same matrices forever — through later
mutations, commits, new writers and rollbacks. -/
theorem snapshot_isolation (valid : Version → Bool) : ∀ (evs : List (Ev α)) (S : Sys α) (Pub : Nat → Prop),
    Published S Pub →
    ∃ Pub' : Nat → Prop, (∀ h, Pub h → Pub' h) ∧
      ∀ h, Pub h → (evs.foldl (sstep valid) S).H.mem h = S.H.mem h
  | [], S, Pub, _ => ⟨Pub, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
  | e :: es, S, Pub, hS => by
    obtain ⟨c1, c2, c3, c4, c5⟩ := hS
    -- the published set after `e`: commit publishes the write version's handles
    let Pub1 : Nat → Prop := match e, S.W with
      | .commit, some W => fun h => Pub h ∨ ∃ i, (W i).h = h
      | _, _ => Pub
    have step : Published (sstep valid S e) Pub1 ∧ (∀ h, Pub h → Pub1 h) ∧
        ∀ h, Pub h → (sstep valid S e).H.mem h = S.H.mem h := by
      cases e with
      | begin =>
        simp only [sstep, Pub1]
        split
        · exact ⟨⟨c1, c2, c3, c4, c5⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
        · refine ⟨⟨c1, c2, c3, ?_, fun _ => rfl⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
          intro W hW; simp at hW; subst hW; exact newVersion_iso _ c2 c3
      | op o =>
        simp only [sstep, Pub1]
        cases hW : S.W with
        | none => simp only; exact ⟨⟨c1, c2, c3, c4, c5⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
        | some W =>
          simp only
          have hi := c4 W hW
          have := wstep_iso hi o
          refine ⟨⟨c1, this.1.pub_lt, ?_, ?_, fun _ => c5 (by simp [hW])⟩, fun _ h => h, this.2⟩
          · intro i
            have hfr : S.H.fresh ≤ (wstep S.H W o).1.fresh := by
              cases o <;> simp [wstep, mutate, derefMut, replace, Heap.write] <;> split <;> simp
            show (S.g.committed i).h < (wstep S.H W o).1.fresh
            have := c3 i; omega
          · intro W' hW'; simp at hW'; subst hW'; exact this.1
      | commit =>
        simp only [sstep, Pub1]
        cases hW : S.W with
        | none => simp only; exact ⟨⟨c1, c2, c3, c4, c5⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
        | some W =>
          simp only
          have hi := c4 W hW
          split
          · refine ⟨⟨fun i => Or.inr ⟨i, rfl⟩, ?_, hi.w_lt, by simp, by simp⟩,
              fun _ h => Or.inl h, fun _ _ => by first | rfl | trivial⟩
            intro h hp; rcases hp with hp | ⟨i, rfl⟩
            · exact c2 h hp
            · exact hi.w_lt i
          · refine ⟨⟨fun i => Or.inl (c1 i), ?_, c3, by simp, by simp⟩,
              fun _ h => Or.inl h, fun _ _ => by first | rfl | trivial⟩
            intro h hp; rcases hp with hp | ⟨i, rfl⟩
            · exact c2 h hp
            · exact hi.w_lt i
      | rollback =>
        simp only [sstep, Pub1]
        exact ⟨⟨c1, c2, c3, by simp, by simp⟩, fun _ h => h, fun _ _ => by first | rfl | trivial⟩
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

/-- At most one writer: `begin` while writing is a no-op (the CAS fails). -/
theorem single_writer (valid : Version → Bool) (S : Sys α) (h : S.g.writing = true) :
    sstep valid S .begin = S := by
  simp [sstep, h]

/-- **A refused version is never published and frees the write flag**
(`MvccGraph::commit` :148-151): the committed version is unchanged and the
next `begin` succeeds. -/
theorem refused_commit_rolls_back (valid : Version → Bool) (S : Sys α) (W : Version)
    (hW : S.W = some W) (hv : valid W = false) :
    (sstep valid S .commit).g.committed = S.g.committed ∧
    (sstep valid S .commit).g.writing = false ∧ (sstep valid S .commit).W = none := by
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
that is `MvccGraph::write`/`commit`: the committed graph is only ever read
(`read()` hands out `Arc` clones; read queries run with `write = false`), and
every mutation path — `Runtime` for write queries, GRAPH.BULK, index DDL —
goes through the version returned by `write()`. Reproduced at the API level
by `cow_is_one_directional_old_version_write_leaks` in
`graph/tests/lean_versioned_matrix.rs`. -/
theorem old_write_leaks_into_new_version :
    let H : Heap Nat := ⟨fun _ => 0, 1⟩
    let old : Cow := ⟨0, false⟩          -- the owned layer of a version
    let new := old.newVersion            -- the next version's layer
    let r := mutate H old (fun _ => 7)   -- a write through the OLD version
    r.1.mem new.h = 7 := by
  decide

end VMCow
