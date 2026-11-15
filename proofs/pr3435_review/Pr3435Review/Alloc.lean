import Pr3435Review.PushPullFuel
/-! # The whole of `bfs_levels_push_pull`: allocation, source check, cleanup

PR branch a76e66420, `graph/src/runtime/functions/algo_procedures.rs`:

| here | there |
| --- | --- |
| `Live`      | which of `v`, `q` is allocated and not yet freed when the fn returns |
| `fail`      | the `fail` closure :1089-1097 — `GrB_Vector_free` on each non-null handle |
| `bfsFull`   | `bfs_levels_push_pull` :1066-1163: `v` alloc :1100-1102, `q` alloc :1103-1105, checked source `setElement`s :1110-1114, the loop (`run`), `GrB_Vector_free(q)` + `Ok(v)` :1161-1162 |
| `bfsFullPre`| the same fn at 28ab35368 :1088-1099: `?` on each alloc (leaks `v` if `q` fails), `setElement` codes ignored |

FFI premises: `newV`, `newQ` are `GrB_Vector_new`'s outcomes (arbitrary: out of
memory may happen), with `GrB_Vector_new` refusing a size above
`GrB_INDEX_MAX + 1 = 2^60` (`hV`); `GrB_Vector_setElement` on index `src` of a
size-`n` vector fails (`GrB_INVALID_INDEX`) iff `src ≥ n`. On failure SuiteSparse
leaves the output handle `NULL`, which is what `fail`'s null checks rely on.
-/
namespace Pr3435.Bfs

structure Live where
  v : Bool
  q : Bool
deriving DecidableEq, Repr

/-- `fail(v, q, e)` (:1089-1097): frees every non-null handle, returns `Err(e)`. -/
def fail (_l : Live) (e : String) : Except String LVec × Live := (.error e, ⟨false, false⟩)

/-- `bfs_levels_push_pull` at a76e66420. -/
def bfsFull {H : Type} (newV newQ : Bool) (g : GB) (hr : Heur H) (h0 : H) (A AT : Pat)
    (n src : Nat) (ml : Int) : Except String LVec × Live :=
  if newV = false then fail ⟨false, false⟩ "BFS level alloc failed" else
  if newQ = false then fail ⟨true, false⟩ "BFS frontier alloc failed" else
  if ¬ src < n then fail ⟨true, true⟩ "BFS source failed" else
  match run g hr A AT ml (n + 1) (init n src h0) with
  | .ok w => (.ok w, ⟨true, false⟩)
  | .error e => fail ⟨true, true⟩ e

/-- **Claim 2 for the whole function.** No `src < n` and no size premise: an
`Ok w` is exactly the BFS levels. (The size bound `n ≤ 2^60` comes from the
successful `GrB_Vector_new`.) -/
theorem bfsFull_correct {H : Type} {newV newQ : Bool} {n : Nat}
    (hV : newV = true → n ≤ GRB_DIM_MAX) {g : GB} (hg : GBSpec n g) (hr : Heur H) (h0 : H)
    {A AT : Pat} (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) (src : Nat) (ml : Int) {w : LVec}
    (h : (bfsFull newV newQ g hr h0 A AT n src ml).1 = .ok w) : Levels A src ml w := by
  unfold bfsFull at h
  by_cases h1 : newV = false
  · rw [if_pos h1] at h; cases h
  rw [if_neg h1] at h
  by_cases h2 : newQ = false
  · rw [if_pos h2] at h; cases h
  rw [if_neg h2] at h
  by_cases hs : src < n
  · rw [if_neg (by simpa using hs)] at h
    have hn := hV (by simpa using h1)
    generalize hr' : run g hr A AT ml (n + 1) (init n src h0) = r at h
    cases r with
    | ok w' =>
      cases h
      exact (pushPull_correct hg hr h0 hT hb hs hn ml).1 _ hr'
    | error _ => cases h
  · rw [if_pos hs] at h; cases h

/-- **Out-of-range source now errors** (a76e66420 :1110-1114), like LAGraph's
`GrB_INVALID_INDEX`. -/
theorem bfsFull_src_out_of_range {H : Type} (newV newQ : Bool) (g : GB) (hr : Heur H) (h0 : H)
    (A AT : Pat) {n src : Nat} (hs : n ≤ src) (ml : Int) :
    ∃ e, (bfsFull newV newQ g hr h0 A AT n src ml).1 = .error e := by
  unfold bfsFull
  by_cases h1 : newV = false
  · rw [if_pos h1]; exact ⟨_, rfl⟩
  rw [if_neg h1]
  by_cases h2 : newQ = false
  · rw [if_pos h2]; exact ⟨_, rfl⟩
  rw [if_neg h2, if_pos (by omega)]; exact ⟨_, rfl⟩

/-- With both allocations succeeding, the source check's error is the one returned. -/
theorem bfsFull_src_error {H : Type} (g : GB) (hr : Heur H) (h0 : H) (A AT : Pat)
    {n src : Nat} (hs : n ≤ src) (ml : Int) :
    bfsFull true true g hr h0 A AT n src ml = (.error "BFS source failed", ⟨false, false⟩) := by
  unfold bfsFull; simp [fail]; omega

/-- **No leak** (a76e66420): on every `Err` path nothing stays allocated; on `Ok`
only the returned `v` (which `algo_bfs` frees, :1335-1337). -/
theorem bfsFull_no_leak {H : Type} (newV newQ : Bool) (g : GB) (hr : Heur H) (h0 : H)
    (A AT : Pat) (n src : Nat) (ml : Int) :
    (∀ e, (bfsFull newV newQ g hr h0 A AT n src ml).1 = .error e →
        (bfsFull newV newQ g hr h0 A AT n src ml).2 = ⟨false, false⟩) ∧
    (∀ w, (bfsFull newV newQ g hr h0 A AT n src ml).1 = .ok w →
        (bfsFull newV newQ g hr h0 A AT n src ml).2 = ⟨true, false⟩) := by
  unfold bfsFull
  by_cases h1 : newV = false
  · simp [h1, fail]
  by_cases h2 : newQ = false
  · simp [h1, h2, fail]
  by_cases hs : src < n
  · simp only [h1, h2, hs, if_false, not_true_eq_false, Bool.false_eq_true]
    cases run g hr A AT ml (n + 1) (init n src h0) <;> simp [fail]
  · simp [h1, h2, hs, fail]

/-! ## History: 28ab35368 -/

/-- `bfs_levels_push_pull` at 28ab35368 :1088-1099: each alloc returns through
`?` (no cleanup), the source `setElement` codes are ignored. -/
def bfsFullPre {H : Type} (newV newQ : Bool) (g : GB) (hr : Heur H) (h0 : H) (A AT : Pat)
    (n src : Nat) (ml : Int) : Except String LVec × Live :=
  if newV = false then (.error "BFS level alloc failed", ⟨false, false⟩) else
  if newQ = false then (.error "BFS frontier alloc failed", ⟨true, false⟩) else
  match bfsLevelsPushPull g hr h0 A AT n src ml with
  | .ok w => (.ok w, ⟨true, false⟩)
  | .error e => (.error e, ⟨false, false⟩)

/-- **Historical (pre-a76e664):** if the frontier allocation failed, the level
vector leaked. Fixed by a76e66420 (`bfsFull_no_leak`). -/
theorem pre_a76e664_frontier_alloc_leaks {H : Type} (g : GB) (hr : Heur H) (h0 : H) (A AT : Pat)
    (n src : Nat) (ml : Int) :
    bfsFullPre true false g hr h0 A AT n src ml = (.error "BFS frontier alloc failed", ⟨true, false⟩) := by
  simp [bfsFullPre]

end Pr3435.Bfs
