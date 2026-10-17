/-
# `Cow` (`graph/src/graph/cow.rs`): copy-on-write isolation

`T: Clone` for a GraphBLAS `Matrix` is a *shallow* handle clone (matrix.rs:374
shares the `Arc<GrB_Matrix>`), and `Dup::dup` is the deep copy. So the model is
a heap of handles: `Cow = { h : handle, dup : Bool }`, `Clone` copies `h`,
`dup()` allocates a fresh handle holding a copy of the contents.

| here | there |
| --- | --- |
| `Cow`          | `struct Cow<T> { inner: T, dup: bool }` (cow.rs:43) |
| `new`          | `Cow::new` (cow.rs:50) — `dup: false` |
| `newVersion`   | `Cow::new_version` (cow.rs:55) — shallow clone, `dup: true` |
| `clone`        | `#[derive(Clone)]` (cow.rs:42) — shallow clone, `dup` copied |
| `read`         | `Deref` (cow.rs:75) |
| `derefMut`     | `DerefMut` (cow.rs:83) — `if dup { inner = inner.dup(); dup = false }` |
| `write`        | `derefMut` followed by an in-place mutation of the handle |
| `replace`      | `Cow::replace` (cow.rs:66) — a freshly built inner, `dup: false` |
-/
namespace FalkorRuntimeDS.CowModel

variable {V : Type}

structure Heap (V : Type) where
  mem : Nat → V
  next : Nat

structure Cow where
  h : Nat
  dup : Bool

def upd (f : Nat → V) (k : Nat) (v : V) : Nat → V := fun x => if x = k then v else f x

def alloc (H : Heap V) (v : V) : Heap V × Nat := ({ mem := upd H.mem H.next v, next := H.next + 1 }, H.next)

def newVersion (c : Cow) : Cow := { h := c.h, dup := true }

def read (H : Heap V) (c : Cow) : V := H.mem c.h

/-- cow.rs:83 -/
def derefMut (H : Heap V) (c : Cow) : Heap V × Cow :=
  if c.dup then let r := alloc H (H.mem c.h); (r.1, { h := r.2, dup := false }) else (H, c)

def write (H : Heap V) (c : Cow) (f : V → V) : Heap V × Cow :=
  let r := derefMut H c
  ({ r.1 with mem := upd r.1.mem r.2.h (f (r.1.mem r.2.h)) }, r.2)

/-- cow.rs:66 — the new inner is a fresh allocation. -/
def replace (H : Heap V) (_c : Cow) (v : V) : Heap V × Cow :=
  let r := alloc H v; (r.1, { h := r.2, dup := false })

inductive Op (V : Type) where
  | write (f : V → V)
  | replace (v : V)

def step (H : Heap V) (c : Cow) : Op V → Heap V × Cow
  | .write f => write H c f
  | .replace v => replace H c v

def run (H : Heap V) (c : Cow) : List (Op V) → Heap V × Cow
  | [] => (H, c)
  | o :: os => let r := step H c o; run r.1 r.2 os

/-- `c` does not own handle `x`: either it points elsewhere, or it will copy first. -/
def NotOwner (c : Cow) (x : Nat) : Prop := c.h = x → c.dup = true

theorem step_frame (H : Heap V) (c : Cow) (o : Op V) (x : Nat) (hx : x < H.next)
    (hn : NotOwner c x) :
    (step H c o).1.mem x = H.mem x ∧ x < (step H c o).1.next ∧ NotOwner (step H c o).2 x := by
  cases o with
  | write f =>
    unfold step write derefMut
    by_cases hd : c.dup = true
    · simp only [hd, ite_true, alloc, upd, NotOwner]
      refine ⟨?_, by omega, ?_⟩
      · have : x ≠ H.next := by omega
        simp [this]
      · intro h; omega
    · simp only [hd, Bool.false_eq_true, ite_false, upd, NotOwner]
      have : x ≠ c.h := fun e => hd (hn e.symm)
      exact ⟨by simp [this], hx, fun e => absurd e.symm this⟩
  | replace v =>
    unfold step replace
    simp only [alloc, upd, NotOwner]
    refine ⟨?_, by omega, ?_⟩
    · have : x ≠ H.next := by omega
      simp [this]
    · intro h; omega

/-- **Isolation.** After `v2 = v1.new_version()`, no sequence of mutations through
`v2` (or through any further `new_version`/`clone` of an untouched `v2`) changes
what `v1` reads. -/
theorem newVersion_isolated (H : Heap V) (v1 : Cow) (hv : v1.h < H.next) (ops : List (Op V)) :
    read (run H (newVersion v1) ops).1 v1 = read H v1 := by
  suffices ∀ (H : Heap V) (c : Cow), v1.h < H.next → NotOwner c v1.h →
      (run H c ops).1.mem v1.h = H.mem v1.h from this H _ hv (fun _ => rfl)
  induction ops with
  | nil => intro H c _ _; rfl
  | cons o os ih =>
    intro H c hx hn
    obtain ⟨e1, e2, e3⟩ := step_frame H c o v1.h hx hn
    simp only [run]
    rw [ih _ _ e2 e3, e1]

/-- A new version initially reads the same contents (no copy was made). -/
theorem newVersion_read (H : Heap V) (v1 : Cow) : read H (newVersion v1) = read H v1 := rfl

/-- Two sibling versions of the same committed value are isolated from each other:
the first write through either one moves it to a fresh handle. -/
theorem siblings_isolated (H : Heap V) (v1 : Cow) (hv : v1.h < H.next) (f : V → V) :
    let r := write H (newVersion v1) f
    read r.1 (newVersion v1) = read H v1 := by
  simp only [write, derefMut, newVersion, read, alloc, upd]
  have : v1.h ≠ H.next := by omega
  simp [upd, this]

/-! ## The isolation is one-directional

`new_version` protects the *committed* value from the new version, not the
other way round: `v1` keeps `dup = false`, so mutating `v1` after taking
`v2 = v1.new_version()` writes the shared handle and `v2` sees it. The same
holds for the derived `Clone` of any `dup = false` `Cow`. MVCC relies on the
committed version never being mutated once a new version exists.
(Rust: `graph/tests/lean_runtime_ds.rs::cow_isolation`.) -/
example :
    let H : Heap Nat := ⟨fun _ => 1, 1⟩
    let v1 : Cow := ⟨0, false⟩
    let v2 := newVersion v1
    read (write H v1 (fun _ => 99)).1 v2 = 99 := by decide

end FalkorRuntimeDS.CowModel
