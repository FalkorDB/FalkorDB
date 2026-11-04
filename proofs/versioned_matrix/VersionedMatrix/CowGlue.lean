import VersionedMatrix.Cow
/-
# `Cow::new`, `Cow::deref`, `MvccGraph::new`/`from_graph`/`drop`

| here | there |
| --- | --- |
| `cowNew`   | `Cow::new` (cow.rs:84): the model *allocates* the handle. The Rust only stores the handle it is given (a clone aliases it), so that it is fresh and unpublished is the caller's obligation, stated here as `hp` (every published handle is below `H.fresh`); `cowNew_spec` is conditional on it, discharged at no caller |
| `deref`    | `Cow::deref` (cow.rs:128) |
| `mvccNew`  | `MvccGraph::new` (mvcc_graph.rs:97) / `from_graph` (:111): the published version |
| `dropG`    | `impl Drop for MvccGraph` (:302): cancels the committed graph's indexing |
-/
namespace VMCow

variable {α : Type}

def cowNew (H : Heap α) (x : α) : Heap α × Cow :=
  (⟨fun k => if k = H.fresh then x else H.mem k, H.fresh + 1⟩, ⟨H.fresh, false⟩)

def deref (H : Heap α) (c : Cow) : α := H.mem c.h

/-- A new `Cow` reads its contents, owns its handle, and that handle is not
published, so writing through it is isolated (`Iso.owned_private`). -/
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
