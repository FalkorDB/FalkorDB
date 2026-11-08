import VersionedMatrix.Cow
/-
# `Cow::new`, `Cow::deref`, `MvccGraph::new`/`from_graph`, `Shared::drop`

| here | there |
| --- | --- |
| `cowWrap`  | `Cow::new` (`pub fn new`, cow.rs:84): stores the handle it is given, no allocation, `dup = false`. The handle may be shared (a `Matrix` clone aliases the `Arc<GrB_Matrix>`), so both cases are modelled: `cowWrap_private_iff` (the wrapper is private iff the handle is unpublished), `cowWrap_shared_leaks` (a write through a wrapper over a published handle lands on it), `cowWrap_private_write_isolated`, `cowWrap_newVersion_isolated` (a `new_version` of any wrapper writes through a dup) |
| `alloc`, `cowNew` | `Matrix::new` / `decode` / `transpose` (each allocates a GraphBLAS handle) followed by `Cow::new`: `cowNew = cowWrap ∘ alloc` (`cowNew_eq`). Every in-tree caller of `Cow::new` is of this shape (below), so `cowNew_spec` (fresh and unpublished, no hypothesis about the caller beyond published handles being allocated) covers them |
| callers of `Cow::new` | in `versioned_matrix.rs`: `Delta::new` (fed `Matrix::new` or `decode`), `Delta::transposed` (a `transpose` result), `VersionedMatrix::new` (`Matrix::new`), `from_matrix` (its only caller, `Tensor::rebuild_backward`'s `extract().transpose()` in `tensor.rs`, passes a fresh matrix), `VersionedMatrix::transpose`, `decode`; in `tensor.rs`: `Tensor::new` (`Matrix::new`), `decode`: none passes a clone or a matrix a published version holds |
| `deref`    | `Cow::deref` (cow.rs:128) |
| `mvccNew`  | `MvccGraph::new` (mvcc_graph.rs:97) / `from_graph` (:111): the published version |
| `dropG`    | `impl Drop for Shared` (:444): cancels the committed graph's indexing, once the graph and every build that reached it have let go |
-/
namespace VMCow

variable {α : Type}

def cowNew (H : Heap α) (x : α) : Heap α × Cow :=
  (⟨fun k => if k = H.fresh then x else H.mem k, H.fresh + 1⟩, ⟨H.fresh, false⟩)

def deref (H : Heap α) (c : Cow) : α := H.mem c.h

/-- `Cow::new` over a handle that already exists: no allocation, no flag, the heap is untouched. -/
def cowWrap (h : Nat) : Cow := ⟨h, false⟩

/-- A GraphBLAS allocation (`Matrix::new`, `decode`, `transpose`, `dup`): a fresh handle holding `x`. -/
def alloc (H : Heap α) (x : α) : Heap α × Nat :=
  (⟨fun k => if k = H.fresh then x else H.mem k, H.fresh + 1⟩, H.fresh)

/-- The in-tree shape of every `Cow::new` call: allocate, then wrap. -/
theorem cowNew_eq (H : Heap α) (x : α) :
    cowNew H x = ((alloc H x).1, cowWrap (alloc H x).2) := rfl

/-- Reading a wrapped handle reads it. -/
theorem cowWrap_deref (H : Heap α) (h : Nat) : deref H (cowWrap h) = H.mem h := rfl

/-- **Private iff unpublished.** The wrapper (`dup = false`, so it writes in place) is isolated from the
published snapshots exactly when its handle is not one of theirs. -/
theorem cowWrap_private_iff (Pub : Nat → Prop) (h : Nat) :
    ((cowWrap h).dup = false → ¬ Pub (cowWrap h).h) ↔ ¬ Pub h := by
  simp [cowWrap]

/-- **A wrapper over a shared (published) handle writes into it**: a write that changes the contents
changes what the published snapshot reads. `Cow::new` does not check; only the caller can. -/
theorem cowWrap_shared_leaks (H : Heap α) (Pub : Nat → Prop) (h : Nat) (hpub : Pub h) (f : α → α)
    (hf : f (H.mem h) ≠ H.mem h) :
    Pub h ∧ (mutate H (cowWrap h) f).1.mem h ≠ H.mem h := by
  refine ⟨hpub, ?_⟩
  simpa [mutate, derefMut, cowWrap, Heap.write] using hf

/-- A wrapper over an unpublished handle leaves every published handle's contents alone. -/
theorem cowWrap_private_write_isolated (H : Heap α) (Pub : Nat → Prop) (h : Nat) (hpriv : ¬ Pub h)
    (f : α → α) : ∀ k, Pub k → (mutate H (cowWrap h) f).1.mem k = H.mem k := by
  intro k hk
  have hne : k ≠ h := fun e => hpriv (e ▸ hk)
  simp [mutate, derefMut, cowWrap, Heap.write, hne]

/-- Whatever the handle's sharing, after `new_version` (`dup = true`) a write goes through a copy: every
existing handle keeps its contents. This is how a `Cow` over a shared handle is made safe to write. -/
theorem cowWrap_newVersion_isolated (H : Heap α) (h : Nat) (f : α → α) :
    ∀ k, k < H.fresh → (mutate H (cowWrap h).newVersion f).1.mem k = H.mem k := by
  intro k hk
  have hne : k ≠ H.fresh := Nat.ne_of_lt hk
  simp [mutate, derefMut, cowWrap, Cow.newVersion, Heap.write, hne]

/-- A `Cow` built over a freshly allocated handle (every in-tree caller) reads its contents, owns its
handle, and that handle is not published, so writing through it is isolated (`Iso.owned_private`).
The only premise is that published handles were allocated before (`hp`), which `Iso`/`Published` carry. -/
theorem cowNew_spec (H : Heap α) (x : α) (Pub : Nat → Prop) (hp : ∀ h, Pub h → h < H.fresh) :
    deref (cowNew H x).1 (cowNew H x).2 = x ∧ (cowNew H x).2.dup = false ∧ ¬ Pub (cowNew H x).2.h ∧
    ∀ k, k < H.fresh → (cowNew H x).1.mem k = H.mem k := by
  refine ⟨by simp [deref, cowNew], rfl, fun h => Nat.lt_irrefl _ (hp _ h), fun k hk => ?_⟩
  simp [cowNew, Nat.ne_of_lt hk]

theorem deref_view (H : Heap α) (V : Version) (i : Nat) : deref H (V i) = view H V i := rfl

/-- `MvccGraph::new` / `from_graph`: the given graph published, nothing in flight. -/
def mvccNew (H : Heap α) (V : Version) : Sys α := ⟨H, ⟨V⟩, none⟩

theorem mvccNew_ready (valid : Version → Bool) (H : Heap α) (V : Version) :
    (sstep valid (mvccNew H V) .begin).W = some V.newVersion ∧
    (sstep valid (mvccNew H V) .begin).g.committed = V := ⟨rfl, rfl⟩

/-- `Drop`: `cancel_indexing` on the committed graph; the model records that
the committed version's indexers were cancelled and nothing else changes. -/
def dropG (S : Sys α) : Sys α × Version := (S, S.g.committed)
theorem dropG_spec (S : Sys α) : (dropG S).1 = S ∧ (dropG S).2 = S.g.committed := ⟨rfl, rfl⟩

end VMCow
