/-
# `Cow` (`graph/src/graph/cow.rs`): copy-on-write isolation

`T: Clone` for a GraphBLAS `Matrix` is a *shallow* handle clone (matrix.rs:374
shares the `Arc<GrB_Matrix>`), and `Dup::dup` is the deep copy. So the model is
a heap of handles: `Cow = { h : handle, dup : Bool }`, `Clone` copies `h`,
`dup()` allocates a fresh handle holding a copy of the contents.

| here | there (main 1d15bf713) |
| --- | --- |
| `Cow`          | `struct Cow<T> { inner: T, dup: bool, branched }` (cow.rs:60); `branched` exists in debug builds only |
| `new`          | `Cow::new` (cow.rs:84) — `dup: false` |
| `newVersion`   | `Cow::new_version` (cow.rs:97) — shallow clone, `dup: true` |
| `clone`        | `impl Clone for Cow` (cow.rs:73) — shallow clone, `dup` copied (and, in debug, the `branched` `Arc` shared) |
| `read`         | `Deref` (cow.rs:128) |
| `derefMut`     | `DerefMut` (cow.rs:134) — `if dup { inner = inner.dup(); dup = false }` |
| `write`        | `derefMut` followed by an in-place mutation of the handle |
| `replace`      | `Cow::replace` (cow.rs:112) — a freshly built inner, `dup: false` |
| `DCow`/`dNewVersion`/`dDerefMut`/`dWrite`/`dReplace` | the same functions in a debug build, with the #2905 `branched: Arc<AtomicBool>` flag (cow.rs:69, 101, 117-120, 138-148) |

The release model (`Cow`, `derefMut`, ...) is unchanged by #2905: every
`#[cfg(debug_assertions)]` line is erased in release. `dbg_agrees_with_release`
proves the debug model is the release one plus a panic.
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

/-- cow.rs:134 -/
def derefMut (H : Heap V) (c : Cow) : Heap V × Cow :=
  if c.dup then let r := alloc H (H.mem c.h); (r.1, { h := r.2, dup := false }) else (H, c)

def write (H : Heap V) (c : Cow) (f : V → V) : Heap V × Cow :=
  let r := derefMut H c
  ({ r.1 with mem := upd r.1.mem r.2.h (f (r.1.mem r.2.h)) }, r.2)

/-- cow.rs:112 — the new inner is a fresh allocation. -/
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

/-- cow.rs:73 — release: copy the (shallow) inner handle and the `dup` flag. -/
def clone (c : Cow) : Cow := { h := c.h, dup := c.dup }

theorem clone_eq (c : Cow) : clone c = c := rfl
theorem clone_read (H : Heap V) (c : Cow) : read H (clone c) = read H c := rfl

/-! ## The isolation is one-directional

`new_version` protects the *committed* value from the new version, not the
other way round: `v1` keeps `dup = false`, so mutating `v1` after taking
`v2 = v1.new_version()` writes the shared handle and `v2` sees it. The same
holds for the derived `Clone` of any `dup = false` `Cow`. MVCC relies on the
committed version never being mutated once a new version exists.
(Rust: `graph/tests/lean_runtime_ds.rs::cow_isolation`.) Since #2905 the
contract is documented on `Cow` and debug builds refuse such a write (below);
release builds behave as before, so this stays a documented latent contract. -/
theorem release_old_write_leaks :
    let H : Heap Nat := ⟨fun _ => 1, 1⟩
    let v1 : Cow := ⟨0, false⟩
    let v2 := newVersion v1
    read (write H v1 (fun _ => 99)).1 v2 = 99 := by decide

/-- The same through a `clone` taken before the branch. -/
theorem release_clone_write_leaks :
    let H : Heap Nat := ⟨fun _ => 1, 1⟩
    let v1 : Cow := ⟨0, false⟩
    let alias := clone v1
    let v2 := newVersion v1
    read (write H alias (fun _ => 99)).1 v2 = 99 := by decide

/-! ## Debug builds: the `branched` flag (#2905)

Each `Cow` carries `branched: Arc<AtomicBool>`, a cell `b` in a second heap
`F`. `clone` shares the cell; `new_version` sets the source's cell and gives
the new version a fresh `false` cell; `replace` and the copy in `deref_mut`
give a fresh `false` cell (the inner is fresh, so nobody else shares it);
`deref_mut` then asserts the cell is `false` (`none` = the `debug_assert!`
panic). -/

structure DCow where
  h : Nat
  dup : Bool
  b : Nat

structure DState (V : Type) where
  H : Heap V
  F : Heap Bool

/-- The release view of a debug `Cow`: drop the flag. -/
def DCow.rel (c : DCow) : Cow := ⟨c.h, c.dup⟩

/-- cow.rs:84 — a fresh `false` flag. -/
def dNew (S : DState V) (h : Nat) : DState V × DCow :=
  let r := alloc S.F false; ({ S with F := r.1 }, ⟨h, false, r.2⟩)

/-- cow.rs:73 — the `Arc` is cloned, so the cell is shared. -/
def dClone (c : DCow) : DCow := c

/-- cow.rs:97-106 — `self.branched.store(true)`, new version gets a fresh cell. -/
def dNewVersion (S : DState V) (c : DCow) : DState V × DCow :=
  let F1 : Heap Bool := { S.F with mem := upd S.F.mem c.b true }
  let r := alloc F1 false
  ({ S with F := r.1 }, ⟨c.h, true, r.2⟩)

/-- cow.rs:134-149 — copy if `dup` (fresh inner, fresh flag), then
`debug_assert!(!branched)`. -/
def dDerefMut (S : DState V) (c : DCow) : Option (DState V × DCow) :=
  let p : DState V × DCow :=
    if c.dup then
      let rH := alloc S.H (S.H.mem c.h)
      let rF := alloc S.F false
      (⟨rH.1, rF.1⟩, ⟨rH.2, false, rF.2⟩)
    else (S, c)
  if p.1.F.mem p.2.b then none else some p

def dWrite (S : DState V) (c : DCow) (f : V → V) : Option (DState V × DCow) :=
  (dDerefMut S c).map fun p =>
    ({ p.1 with H := { p.1.H with mem := upd p.1.H.mem p.2.h (f (p.1.H.mem p.2.h)) } }, p.2)

/-- cow.rs:112-121 — fresh inner, fresh flag. -/
def dReplace (S : DState V) (_c : DCow) (v : V) : DState V × DCow :=
  let rH := alloc S.H v
  let rF := alloc S.F false
  (⟨rH.1, rF.1⟩, ⟨rH.2, false, rF.2⟩)

theorem alloc_read_new (H : Heap V) (v : V) : (alloc H v).1.mem (alloc H v).2 = v := by
  simp [alloc, upd]

theorem alloc_read_old (H : Heap V) (v : V) (k : Nat) (hk : k < H.next) :
    (alloc H v).1.mem k = H.mem k := by
  have : k ≠ H.next := by omega
  simp [alloc, upd, this]

/-- **Debug builds catch the leak.** Writing through an owned version
(`dup = false`) after `new_version` panics, so the write never reaches the
new version. `c.b < S.F.next`: its flag cell was allocated. -/
theorem dbg_write_through_branched_source_panics (S : DState V) (c : DCow)
    (hd : c.dup = false) (hb : c.b < S.F.next) (f : V → V) :
    dWrite (dNewVersion S c).1 c f = none := by
  simp only [dWrite, dDerefMut, hd, dNewVersion, Bool.false_eq_true, ite_false]
  have : c.b < ({ S.F with mem := upd S.F.mem c.b true } : Heap Bool).next := hb
  rw [alloc_read_old _ _ _ this]
  simp [upd]

/-- The same for a shallow clone taken before the branch (main's test
`write_through_clone_of_branched_source_is_caught`): the clone shares the cell. -/
theorem dbg_write_through_clone_panics (S : DState V) (c : DCow)
    (hd : c.dup = false) (hb : c.b < S.F.next) (f : V → V) :
    let alias := dClone c
    dWrite (dNewVersion S c).1 alias f = none :=
  dbg_write_through_branched_source_panics S c hd hb f

/-- Writing the new version itself never trips the assert: it copies first
and gets a fresh flag. -/
theorem dbg_new_version_write_ok (S : DState V) (c : DCow) (f : V → V) :
    (dWrite (dNewVersion S c).1 (dNewVersion S c).2 f).isSome := by
  simp only [dWrite, dDerefMut, dNewVersion, ite_true]
  rw [alloc_read_new]; simp

/-- After `replace` the source owns a fresh, unshared inner again and may be
written (main's test `new_version_write_is_isolated`). -/
theorem dbg_replace_then_write_ok (S : DState V) (c : DCow) (v : V) (f : V → V) :
    (dWrite (dReplace S c v).1 (dReplace S c v).2 f).isSome := by
  simp only [dWrite, dDerefMut, dReplace, Bool.false_eq_true, ite_false]
  rw [alloc_read_new]; simp

/-- **Debug = release + a panic.** Whenever the debug write does not panic, its
heap and `Cow` are exactly the release `write`'s. -/
theorem dbg_agrees_with_release (S S' : DState V) (c c' : DCow) (f : V → V)
    (h : dWrite S c f = some (S', c')) :
    (S'.H, c'.rel) = write S.H c.rel f := by
  unfold dWrite dDerefMut at h
  cases hd : c.dup <;> simp only [hd, Bool.false_eq_true, ite_true, ite_false] at h <;>
    split at h <;> simp at h <;> obtain ⟨rfl, rfl⟩ := h <;>
    simp [write, derefMut, DCow.rel, hd]

theorem dbg_replace_agrees_with_release (S : DState V) (c : DCow) (v : V) :
    ((dReplace S c v).1.H, (dReplace S c v).2.rel) = replace S.H c.rel v := rfl

theorem dbg_newVersion_agrees_with_release (S : DState V) (c : DCow) :
    (dNewVersion S c).1.H = S.H ∧ (dNewVersion S c).2.rel = newVersion c.rel := ⟨rfl, rfl⟩

end FalkorRuntimeDS.CowModel
