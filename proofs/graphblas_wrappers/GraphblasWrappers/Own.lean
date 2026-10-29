/-
# Ownership: the `GrB_Matrix` drop protocol as a resource-state model

`Matrix<T>` and `Iter<E>` share one GraphBLAS handle. This file models the drop
protocol as a small-step machine, before and after #3090 (e45be4965, #3089).

## Main (since #3090): `Arc<Handle>`

```rust
struct Handle(GrB_Matrix);                       // matrix.rs:365
impl Drop for Handle {                           // matrix.rs:368
    fn drop(&mut self) { GrB_Matrix_free(&raw mut self.0); }
}
pub struct Matrix<T> { m: Arc<Handle>, .. }      // no Drop impl
pub struct Iter<E>   { m: Arc<Handle>, inner: GxB_Iterator, .. }
impl Drop for Iter<E> {                          // matrix.rs:1689
    fn drop(&mut self) { if !self.inner.is_null() { GxB_Iterator_free(..) } }
}   // then the field `m: Arc<Handle>` drops
```

Dropping a `Matrix` clone (or an `Iter`) is now one `Arc` drop: a single
atomic `fetch_sub` on the strong count, and the thread whose `fetch_sub`
returned `1` runs `Handle::drop`. The decision to free and the decrement are
the *same* atomic step, so there is no window for a second owner to act in
between. Model: `AEv.release` (and `AEv.clone`, the `fetch_add` of
`Matrix::clone` / `Iter::detached`), `arcStep`, `liveRun`. The `Arc` contract
(one atomic decrement; the drop runs on the thread that saw 1) is std's, taken
as the model's step, not proved.

* `arc_frees_once` — for *any* number of owners and *any* interleaving of
  clones and drops by live owners, the matrix is freed at most once, and it is
  freed exactly when the count reaches zero: never a double free, never a leak.
* `arc_all_drop_frees_once` — `n ≥ 1` owners each dropping once, in whatever
  order the scheduler picks, free it exactly once.
* `arc_every_interleaving_frees_once` — the six two-owner interleavings, the
  one that leaked before #3090 included, all free exactly once now.
* `iter_frees_iterator_before_matrix` — `Iter::drop` frees its
  `GxB_Iterator` before its `Arc<Handle>` field can free the matrix.

## Before #3090: `Arc<GrB_Matrix>` + `Arc::get_mut`

```rust
if let Some(m) = Arc::get_mut(&mut self.m) {   //  <- frees iff last owner
    GrB_Matrix_free(m);
}
// then the Arc itself is dropped, decrementing the strong count
```

`Arc::get_mut` returns `Some` iff the strong count is 1. So a drop did two
things: **observe** the count (and free iff it is 1), then **decrement**. Two
separate atomic operations, no lock across them. The leak is the one (test `bug_matrix_concurrent_drop_leaks`: 19 of 200,000 pairs leaked);
main's regression test is `graph/tests/matrix_concurrent_drop.rs`
`concurrently_dropped_clones_free_the_matrix_once`.
-/

namespace GBW

/-! ## Pre-#3090 protocol (history) -/

/-- A single atomic event in the life of a two-owner `GrB_Matrix`. `obs o` is
owner `o`'s `Arc::get_mut` read; `dec o` is owner `o`'s `Arc` drop. -/
inductive Ev where
  | obs (o : Bool)
  | dec (o : Bool)
deriving DecidableEq, Repr

/-- Machine state: strong count still live, and how many times
`GrB_Matrix_free` has run. -/
structure St where
  count : Nat
  frees : Nat
deriving DecidableEq, Repr

/-- Pre-#3090 step. An `obs` frees iff it observes count `1` (`Arc::get_mut` =
`Some`); it does not change the count. A `dec` decrements. -/
def step (s : St) : Ev → St
  | .obs _ => { s with frees := if s.count = 1 then s.frees + 1 else s.frees }
  | .dec _ => { s with count := s.count - 1 }

def run (init : St) (t : List Ev) : St := t.foldl step init

/-- Two owners, count starts at 2, nothing freed yet. -/
def start : St := { count := 2, frees := 0 }

/-- A trace is a *valid interleaving of two drops* when each owner observes
exactly once, decrements exactly once, and observes before it decrements. The
six such traces, enumerated. -/
def validTraces : List (List Ev) :=
  [ [.obs true,  .dec true,  .obs false, .dec false]   -- fully serial: A then B
  , [.obs false, .dec false, .obs true,  .dec true]    -- fully serial: B then A
  , [.obs true,  .obs false, .dec true,  .dec false]   -- interleaved
  , [.obs true,  .obs false, .dec false, .dec true]    -- interleaved
  , [.obs false, .obs true,  .dec false, .dec true]    -- interleaved
  , [.obs false, .obs true,  .dec true,  .dec false] ] -- interleaved

/-! ## Main protocol (#3090): `Arc<Handle>`, free in `Handle::drop` -/

/-- An atomic event on the shared `Arc<Handle>`: `clone` is `Matrix::clone` /
`Iter::detached` (`Arc::clone`, a `fetch_add`); `release` is the `Arc` drop of a
`Matrix` or an `Iter` (a `fetch_sub`, running `Handle::drop` iff it returned 1). -/
inductive AEv where
  | clone
  | release
deriving DecidableEq, Repr

/-- Main's step: the free decision is made by the same atomic step that
decrements, from the count it observed. -/
def arcStep (s : St) : AEv → St
  | .clone   => { s with count := s.count + 1 }
  | .release => { count := s.count - 1,
                  frees := if s.count = 1 then s.frees + 1 else s.frees }

/-- Run a trace in which every event is performed by a live owner: an event at
count `0` is impossible in Rust (nobody holds an `Arc` to clone or drop), and
returns `none` here. -/
def liveRun (s : St) : List AEv → Option St
  | [] => some s
  | e :: t => if s.count = 0 then none else liveRun (arcStep s e) t

/-- The ownership invariant: either owners remain and nothing was freed, or no
owner remains and the matrix was freed exactly once. -/
def Inv (s : St) : Prop := (0 < s.count ∧ s.frees = 0) ∨ (s.count = 0 ∧ s.frees = 1)

theorem arcStep_inv (s : St) (e : AEv) (h : Inv s) (hc : s.count ≠ 0) :
    Inv (arcStep s e) := by
  unfold Inv at *
  cases e <;> simp only [arcStep] <;> (try split) <;> omega

theorem liveRun_inv : ∀ (t : List AEv) (s s' : St),
    Inv s → liveRun s t = some s' → Inv s'
  | [], s, s', h, hr => by
      simp only [liveRun, Option.some.injEq] at hr; exact hr ▸ h
  | e :: t, s, s', h, hr => by
      simp only [liveRun] at hr
      split at hr
      · cases hr
      · exact liveRun_inv t _ s' (arcStep_inv s e h ‹_›) hr

/-- **Main: freed exactly once, never leaked, for any `n` and any schedule.**
Start from `n+1` owners (count `n+1`, nothing freed). After any trace of clones
and drops by live owners — any interleaving of any number of threads — the
matrix has been freed at most once, and it has been freed iff no owner remains.
So no double free, and no leak once every owner has dropped. -/
theorem arc_frees_once (n : Nat) (t : List AEv) (s' : St)
    (hr : liveRun { count := n + 1, frees := 0 } t = some s') :
    s'.frees ≤ 1 ∧ (s'.frees = 1 ↔ s'.count = 0) := by
  have := liveRun_inv t _ s' (by unfold Inv; simp) hr
  unfold Inv at this; omega

theorem liveRun_releases (k f : Nat) :
    liveRun { count := k + 1, frees := f } (List.replicate (k + 1) .release) =
      some { count := 0, frees := f + 1 } := by
  induction k generalizing f with
  | zero => simp [liveRun, List.replicate, arcStep]
  | succ k ih =>
      rw [List.replicate_succ]
      simp only [liveRun, arcStep, Nat.add_one_ne_zero, ↓reduceIte, Nat.add_sub_cancel]
      exact ih f

/-- **Main: `n ≥ 1` owners all dropping free exactly once.** Each owner's drop
is one atomic `release`, so every schedule of the `n` concurrent drops is this
same trace; it reaches count 0 with exactly one free. -/
theorem arc_all_drop_frees_once (n : Nat) (hn : 0 < n) :
    liveRun { count := n, frees := 0 } (List.replicate n .release) =
      some { count := 0, frees := 1 } := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  exact liveRun_releases k 0

/-- Main's reading of a pre-#3090 two-owner trace: the `Arc::get_mut`
observation is gone (nothing is observed outside the atomic decrement), and
each `dec` is one atomic `release`. -/
def arcOfEv : Ev → St → St
  | .obs _, s => s
  | .dec _, s => arcStep s .release

def runArc (init : St) (t : List Ev) : St := t.foldl (fun s e => arcOfEv e s) init

/-- **Bug 4 fixed (#3090).** Every one of the six two-owner interleavings —
including the leaking one — frees exactly once on main. -/
theorem arc_every_interleaving_frees_once :
    ∀ t ∈ validTraces, (runArc start t).frees = 1 := by
  decide

/-- A single owner (the common non-shared case, and every `Iter` over an
otherwise unshared matrix) frees exactly once. -/
theorem single_owner_frees_once :
    liveRun { count := 1, frees := 0 } [.release] = some { count := 0, frees := 1 } := by
  decide

/-! ### `Iter::drop` (matrix.rs:1689): iterator before matrix

`Iter::drop`'s body frees the `GxB_Iterator` (when attached); the `m:
Arc<Handle>` field drops after the body (Rust drops fields after `Drop::drop`),
so the matrix, if this was its last owner, is freed second. -/

/-- A freed resource, in the order the frees ran. -/
inductive Res where
  | iterator
  | matrix
deriving DecidableEq, Repr

/-- `Iter::drop` on an iterator holding one of `s.count` references: free the
iterator if attached, then release the `Arc<Handle>`. Returns the new state and
the frees in order. -/
def iterDrop (attached : Bool) (s : St) : St × List Res :=
  let body := if attached then [Res.iterator] else []
  (arcStep s .release, body ++ (if s.count = 1 then [Res.matrix] else []))

/-- **The iterator is freed before the matrix.** An attached `Iter` frees its
`GxB_Iterator` first, and the matrix only after (and only if it was the last
owner), so the iterator never outlives the matrix it is attached to. -/
theorem iter_frees_iterator_before_matrix (s : St) :
    (iterDrop true s).2 = (if s.count = 1 then [.iterator, .matrix] else [.iterator]) := by
  unfold iterDrop; by_cases h : s.count = 1 <;> simp [h]

/-- A detached `Iter` (null `inner`) frees no iterator, only its share of the
matrix. -/
theorem detached_iter_frees_only_matrix (s : St) :
    (iterDrop false s).2 = (if s.count = 1 then [.matrix] else []) := by
  unfold iterDrop; by_cases h : s.count = 1 <;> simp [h]

/-- `Iter::drop` releases exactly one reference, with `arcStep`'s free rule:
an `Iter` is just another owner in `arc_frees_once`. -/
theorem iterDrop_state (a : Bool) (s : St) : (iterDrop a s).1 = arcStep s .release := rfl

/-! ### `dup` (matrix.rs:1199-1254) — deep copy, fresh ownership

`dup` calls `GrB_Matrix_dup` into a *new* `Arc<Handle>` with a fresh `Mutex`
and a fresh `has_pending`. So a dup shares no handle with its source: dropping
either is a single-owner drop, freed once. -/

/-- After `dup`, source and copy own disjoint `Arc<Handle>` cells (two
independent counts, each starting at 1). Dropping both frees two distinct
matrices, once each. -/
theorem dup_independent_ownership :
    ((liveRun { count := 1, frees := 0 } [.release]).map (·.frees)).getD 0 +
    ((liveRun { count := 1, frees := 0 } [.release]).map (·.frees)).getD 0 = 2 := by
  decide

end GBW
