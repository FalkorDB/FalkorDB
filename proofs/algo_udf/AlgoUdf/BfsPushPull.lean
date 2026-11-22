/-! # `bfs_levels_push_pull` (algo_procedures.rs)

The direction-optimizing BFS loop. GraphBLAS is read as its spec, like
`versioned_matrix`'s `Delta.lean`: a matrix is its structure `Nat → Nat → Bool`
(the semiring is `ANY_PAIR`, so values are never read), a vector its pattern.

| here | there |
| --- | --- |
| `pushStep` | `GrB_vxm(q, v, NULL, ANY_PAIR, q, a, GrB_DESC_RSC)`: `q' = (qᵀ a)⟨¬s(v)⟩`, replace |
| `pullStep` | `GrB_mxv(q, v, NULL, ANY_PAIR, at, q, GrB_DESC_RSC)`: `q' = (at q)⟨¬s(v)⟩`, replace |
| `assign`   | `GrB_Vector_assign_INT64(v, q, NULL, k, GrB_ALL, n, GrB_DESC_S)` |
| `loop`     | the `loop { … }` with `k`, the `max_level` break and the `nq == 0` break |
| `dir`      | the push/pull switch (`alpha`/`beta` on f64 counts): an arbitrary oracle |

Every theorem quantifies over all direction choices, so the f64 heuristic
(`avg_degree`, `edges_unexplored`) needs no float model. `fuel` bounds the
iterations (each non-final step assigns at least one fresh node, so `n + 1`
suffices); GraphBLAS error returns are not modelled.
-/
namespace AlgoUdf.BfsPushPull

/-- The `GrB_vxm` push step. -/
def pushStep (a : Nat → Nat → Bool) (n : Nat) (lv : Nat → Option Nat) (q : Nat → Bool) (j : Nat) : Bool :=
  (lv j).isNone && (List.range n).any (fun i => q i && a i j)

/-- The `GrB_mxv` pull step over the transpose `at`. -/
def pullStep (aT : Nat → Nat → Bool) (n : Nat) (lv : Nat → Option Nat) (q : Nat → Bool) (j : Nat) : Bool :=
  (lv j).isNone && (List.range n).any (fun i => aT j i && q i)

/-- `GrB_Vector_assign_INT64(v, q, …, k, …, GrB_DESC_S)`. -/
def assign (lv : Nat → Option Nat) (q : Nat → Bool) (k : Nat) (j : Nat) : Option Nat :=
  if q j then some k else lv j

/-- `GrB_Vector_nvals(q)` over the index range. -/
def nvals (n : Nat) (q : Nat → Bool) : Nat := ((List.range n).filter q).length

/-- The level loop from `k`, with the push/pull oracle `dir` (`true` = push). -/
def loop (a aT : Nat → Nat → Bool) (n : Nat) (maxLevel : Int) (dir : Nat → Bool) :
    Nat → Nat → (Nat → Option Nat) → (Nat → Bool) → (Nat → Option Nat)
  | 0, _, lv, _ => lv
  | f + 1, k, lv, q =>
    if 0 ≤ maxLevel ∧ maxLevel < k then lv else
    let q' := if dir k then pushStep a n lv q else pullStep aT n lv q
    if nvals n q' = 0 then lv else loop a aT n maxLevel dir f (k + 1) (assign lv q' k) q'

/-- `bfs_levels_push_pull`: the source at level 0 in the frontier, then the loop from `k = 1`. -/
def levels (a aT : Nat → Nat → Bool) (n : Nat) (src : Nat) (maxLevel : Int) (dir : Nat → Bool)
    (fuel : Nat) : Nat → Option Nat :=
  loop a aT n maxLevel dir fuel 1 (fun j => if j = src then some 0 else none) (fun j => j = src)

/-- With `at` the transpose of `a`, a push step and a pull step compute the same frontier. -/
theorem push_eq_pull (a aT : Nat → Nat → Bool) (hT : ∀ i j, aT j i = a i j) (n : Nat)
    (lv : Nat → Option Nat) (q : Nat → Bool) : pushStep a n lv q = pullStep aT n lv q := by
  funext j; simp only [pushStep, pullStep, hT, Bool.and_comm (a _ j)]

/-- **The levels do not depend on the push/pull choices** (so not on the f64
heuristic), when `at` is `a`'s transpose. -/
theorem loop_dir_indep (a aT : Nat → Nat → Bool) (hT : ∀ i j, aT j i = a i j) (n : Nat) (maxLevel : Int)
    (d1 d2 : Nat → Bool) : ∀ fuel k lv q,
      loop a aT n maxLevel d1 fuel k lv q = loop a aT n maxLevel d2 fuel k lv q := by
  intro fuel
  induction fuel with
  | zero => intro k lv q; rfl
  | succ f ih =>
    intro k lv q
    have hs : ∀ d : Nat → Bool, (if d k then pushStep a n lv q else pullStep aT n lv q) = pushStep a n lv q := by
      intro d; rw [push_eq_pull a aT hT]; split <;> rfl
    simp only [loop, hs, ih]

theorem levels_dir_indep (a aT : Nat → Nat → Bool) (hT : ∀ i j, aT j i = a i j) (n src : Nat) (maxLevel : Int)
    (d1 d2 : Nat → Bool) (fuel : Nat) :
    levels a aT n src maxLevel d1 fuel = levels a aT n src maxLevel d2 fuel :=
  loop_dir_indep a aT hT n maxLevel d1 d2 fuel _ _ _

/-- A level once set is never overwritten (the `¬s(v)` mask). -/
theorem loop_keeps (a aT : Nat → Nat → Bool) (n : Nat) (maxLevel : Int) (dir : Nat → Bool) :
    ∀ fuel k lv q (j l : Nat), lv j = some l → loop a aT n maxLevel dir fuel k lv q j = some l := by
  intro fuel
  induction fuel with
  | zero => intro k lv q j l h; exact h
  | succ f ih =>
    intro k lv q j l h
    have hq : (if dir k then pushStep a n lv q else pullStep aT n lv q) j = false := by
      split <;> simp [pushStep, pullStep, h]
    dsimp only [loop]
    generalize (if dir k then pushStep a n lv q else pullStep aT n lv q) = q' at hq ⊢
    split
    · exact h
    · split
      · exact h
      · exact ih _ _ _ j l (by simp [assign, hq, h])

/-- The source keeps level 0. -/
theorem levels_src (a aT : Nat → Nat → Bool) (n src : Nat) (maxLevel : Int) (dir : Nat → Bool) (fuel : Nat) :
    levels a aT n src maxLevel dir fuel src = some 0 :=
  loop_keeps a aT n maxLevel dir fuel 1 _ _ src 0 (by simp)

/-- **`max_level` is honoured**: with `max_level ≥ 0` no node gets a level above it. -/
theorem loop_le_max (a aT : Nat → Nat → Bool) (n : Nat) (maxLevel : Int) (hm : 0 ≤ maxLevel)
    (dir : Nat → Bool) : ∀ fuel k lv q,
      (∀ (j l : Nat), lv j = some l → (l : Int) ≤ maxLevel) →
      ∀ (j l : Nat), loop a aT n maxLevel dir fuel k lv q j = some l → (l : Int) ≤ maxLevel := by
  intro fuel
  induction fuel with
  | zero => intro k lv q h; exact h
  | succ f ih =>
    intro k lv q h
    dsimp only [loop]
    generalize (if dir k then pushStep a n lv q else pullStep aT n lv q) = q'
    split
    · exact h
    · rename_i hk
      split
      · exact h
      · apply ih
        intro j l hj
        simp only [assign] at hj
        split at hj
        · injection hj with hj; subst hj
          have : ¬ (maxLevel < (k : Int)) := fun h' => hk ⟨hm, h'⟩
          omega
        · exact h j l hj

theorem levels_le_max (a aT : Nat → Nat → Bool) (n src : Nat) (maxLevel : Int) (hm : 0 ≤ maxLevel)
    (dir : Nat → Bool) (fuel : Nat) :
    ∀ (j l : Nat), levels a aT n src maxLevel dir fuel j = some l → (l : Int) ≤ maxLevel := by
  apply loop_le_max a aT n maxLevel hm dir
  intro j l h
  split at h
  · injection h with h; subst h; exact hm
  · cases h

end AlgoUdf.BfsPushPull
