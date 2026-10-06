/-
# Ownership: `Arc<GrB_Matrix>` drop / dup as a resource-state model

`Matrix<T>` (matrix.rs:361) owns its GraphBLAS handle behind `Arc<GrB_Matrix>`.
The wrappers rely on two facts, and this file models the drop protocol as a
small-step machine and proves both — and exhibits the *race* the protocol does
not defend against.

`Matrix::drop` (matrix.rs:388-400):

```rust
if let Some(m) = Arc::get_mut(&mut self.m) {   //  <- frees iff last owner
    GrB_Matrix_free(m);
}
// then the Arc itself is dropped, decrementing the strong count
```

`Arc::get_mut` returns `Some` iff the strong count is 1. So a `Matrix::drop`
does two things in order: **observe** the count (and free the GraphBLAS matrix
iff it is 1), then **decrement** (the `Arc`'s own drop). The observe and the
decrement are two separate atomic operations, and nothing holds a lock across
them (the `Mutex` in `Matrix` guards `wait`/`dup`, not `drop`). That gap is the
bug reproduced in `graph/tests/lean_graphblas_wrappers.rs`.

`Iter::drop` (matrix.rs:1588-1602) has the identical `Arc::get_mut` shape.
-/

namespace GBW

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

/-- Step. An `obs` frees iff it observes count `1` (`Arc::get_mut` = `Some`); it
does not change the count. A `dec` decrements. -/
def step (s : St) : Ev → St
  | .obs _ => { s with frees := if s.count = 1 then s.frees + 1 else s.frees }
  | .dec _ => { s with count := s.count - 1 }

def run (init : St) (t : List Ev) : St := t.foldl step init

/-- Two owners, count starts at 2, nothing freed yet. -/
def start : St := { count := 2, frees := 0 }

/-- A trace is a *valid interleaving of two drops* when each owner observes
exactly once, decrements exactly once, and observes before it decrements. The
six such traces (4!/(2·2·… ) with the two ordering constraints) enumerated. -/
def validTraces : List (List Ev) :=
  [ [.obs true,  .dec true,  .obs false, .dec false]   -- fully serial: A then B
  , [.obs false, .dec false, .obs true,  .dec true]    -- fully serial: B then A
  , [.obs true,  .obs false, .dec true,  .dec false]   -- interleaved
  , [.obs true,  .obs false, .dec false, .dec true]    -- interleaved
  , [.obs false, .obs true,  .dec false, .dec true]    -- interleaved
  , [.obs false, .obs true,  .dec true,  .dec false] ] -- interleaved

/-- **No double free.** Every valid interleaving of two concurrent drops frees
the GraphBLAS matrix at most once — the `Arc::get_mut` guard is enough to rule
out a double free even under the race. -/
theorem never_double_free : ∀ t ∈ validTraces, (run start t).frees ≤ 1 := by
  decide

/-- **The leak.** Some valid interleaving frees the matrix *zero* times: both
owners run `Arc::get_mut` while the count is still 2, both see `None`, and the
GraphBLAS matrix is never freed. This is the confirmed bug
(`bug_matrix_concurrent_drop_leaks`). -/
theorem race_leaks : ∃ t ∈ validTraces, (run start t).frees = 0 := by
  refine ⟨[.obs true, .obs false, .dec true, .dec false], ?_, ?_⟩
  · decide
  · decide

/-- **Sequential correctness.** When a drop is atomic — observe immediately
followed by that owner's decrement, i.e. the two *serial* traces — the matrix is
freed exactly once. So the leak is purely a scheduling artefact; a lock across
observe+decrement (or `Arc::into_inner`) would remove it. -/
theorem sequential_frees_once :
    (run start [.obs true, .dec true, .obs false, .dec false]).frees = 1 ∧
    (run start [.obs false, .dec false, .obs true, .dec true]).frees = 1 := by
  decide

/-- A single owner (the common non-shared case, and every `Iter::detached`)
always frees exactly once. -/
theorem single_owner_frees_once :
    (run { count := 1, frees := 0 } [.obs true, .dec true]).frees = 1 := by decide

/-! ### `dup` (matrix.rs:1129-1181) — deep copy, fresh ownership

`dup` calls `GrB_Matrix_dup` into a *new* `Arc` with a fresh `Mutex` and a
fresh `has_pending`. So a dup shares no handle with its source: dropping either
is a single-owner drop, freed once. Modelled as: the two matrices have distinct
`Arc` cells, so their drops never interact. -/

/-- After `dup`, source and copy own disjoint cells (distinct `Arc`s). Dropping
both frees two distinct matrices — no interaction, no shared count. -/
theorem dup_independent_ownership :
    (run { count := 1, frees := 0 } [.obs true, .dec true]).frees +
    (run { count := 1, frees := 0 } [.obs false, .dec false]).frees = 2 := by decide

end GBW
