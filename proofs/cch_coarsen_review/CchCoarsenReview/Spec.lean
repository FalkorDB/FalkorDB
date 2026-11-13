import CchCoarsenReview.Sum
/-!
# The GraphBLAS calls the switches replaced, with the meaning the spec gives them

As in `graphblas_wrappers/DeltaLmxm.lean` and `versioned_matrix/Delta.lean`, a GraphBLAS object is
the function from a coordinate to its stored value (`none`: no entry). The calls are *definitions*
here, each with the C API spec it follows (GraphBLAS C API 2.1, SuiteSparse:GraphBLAS 10 user guide,
`GraphBLAS.h`). Nothing is an axiom. Values are modelled in `Int`; `Overflow.lean` shows the bench's
values stay inside `i32`/`u32`, where `Int` and the machine types agree.

| call (Rust, m1opt.patch) | here |
| --- | --- |
| `GrB_mxm(C, NULL, NULL, sr, A, B, desc)`, monoid `PLUS` | `mxm mul k A B`: `C(i,j) = Σ_{t ∈ ind A(i,:) ∩ ind B(:,j)} mul A(i,t) B(t,j)`, no entry when the intersection is empty |
| `GrB_PLUS_TIMES_SEMIRING_INT32`, `GxB_PLUS_FIRST_INT32`, `GxB_PLUS_SECOND_INT32` | `mxm (·*·)`, `mxm (fun a _ => a)`, `mxm (fun _ b => b)` |
| `GrB_DESC_T0` / `GrB_DESC_T1` (transpose an input) | `tr` |
| `GrB_Matrix_select_INT64(.., GrB_OFFDIAG, .., 0)` | `offdiag`: keep `j - i ≠ 0` |
| `GrB_Matrix_select_INT32(.., GrB_VALUELE_INT32, .., y)` | `selLE y` |
| `GrB_Matrix_eWiseMult_BinaryOp(.., f, A, B)` | `ewMult f`: the intersection of the patterns |
| `GrB_Matrix_eWiseAdd_BinaryOp(.., f, A, B)` | `ewAdd f`: the union; `f` only where both are stored |
| `GrB_Matrix_apply_IndexOp_*(C, M, NULL, op, A, y, GrB_DESC_S)` (fresh `C`) | `applyS M op A`: `op` on `A`'s entries under `M`'s structure |
| `GxB_Matrix_build_Scalar(C, I, J, s, n)` (no duplicates) | `buildS I J`: `C(I k, J k) = s` |
| `GrB_Matrix_build_INT32(C, I, J, X, n, NULL)` (no duplicates) | `buildV` |
-/

namespace CchCoarsen

/-- A GraphBLAS matrix: coordinate ↦ stored value. -/
abbrev Mat (α : Type) := Nat → Nat → Option α

/-- Both operands store the inner index `t` of `C(i,j)`. -/
def both {α β : Type} (A : Mat α) (B : Mat β) (i j t : Nat) : Bool := (A i t).isSome && (B t j).isSome

/-- `GrB_mxm` over a semiring with monoid `PLUS` (on `Int`) and multiply `mul`, inner dimension `k`
(C API 2.1 §4.3.1). -/
def mxm (mul : Int → Int → Int) (k : Nat) (A B : Mat Int) : Mat Int := fun i j =>
  if (List.range k).any (both A B i j) then
    some (lsum ((List.range k).filter (both A B i j)) (fun t => mul ((A i t).getD 0) ((B t j).getD 0)))
  else none

/-- `GrB_PLUS_TIMES_SEMIRING_INT32`. -/
abbrev mxmPT := mxm (· * ·)

/-- A transposed operand (`GrB_DESC_T0`, `GrB_DESC_T1`). -/
def tr {α : Type} (A : Mat α) : Mat α := fun i j => A j i

/-- `GrB_select` with `GrB_OFFDIAG` and thunk 0: keeps the entries with `j - i ≠ 0`. -/
def offdiag {α : Type} (A : Mat α) : Mat α := fun i j => if i = j then none else A i j

/-- `GrB_select` with `GrB_VALUELE_INT32` and thunk `y`. -/
def selLE (y : Int) (A : Mat Int) : Mat Int := fun i j =>
  match A i j with
  | some v => if v ≤ y then some v else none
  | none => none

/-- `GrB_eWiseMult` with operator `f`: stored where both are. -/
def ewMult {α β γ : Type} (f : α → β → γ) (A : Mat α) (B : Mat β) : Mat γ := fun i j =>
  match A i j, B i j with
  | some a, some b => some (f a b)
  | _, _ => none

/-- `GrB_eWiseAdd` with operator `f`: stored where either is; `f` where both are. -/
def ewAdd {α : Type} (f : α → α → α) (A B : Mat α) : Mat α := fun i j =>
  match A i j, B i j with
  | some a, some b => some (f a b)
  | some a, none => some a
  | none, some b => some b
  | none, none => none

/-- `GrB_apply` of an index-unary operator into a fresh matrix under a structural mask `M`
(`GrB_DESC_S`; no accumulator): `op` on `A`'s entries where `M` has an entry. -/
def applyS {α β γ : Type} (M : Mat γ) (op : Nat → Nat → α → β) (A : Mat α) : Mat β := fun i j =>
  if (M i j).isSome then (A i j).map (op i j) else none

/-- The same without a mask. -/
def apply {α β : Type} (op : Nat → Nat → α → β) (A : Mat α) : Mat β := fun i j => (A i j).map (op i j)

/-- `GxB_Matrix_build_Scalar(C, I, J, s, n)` into an empty matrix, for tuples without duplicates
(the user guide: the iso value `s` at each `(I k, J k)`). -/
def buildS (s : Int) (t : List (Nat × Nat)) : Mat Int := fun i j => if (i, j) ∈ t then some s else none

/-! ## The facts about `mxm` the proofs use -/

theorem mxm_isSome {mul : Int → Int → Int} {k : Nat} {A B : Mat Int} {i j : Nat} :
    (mxm mul k A B i j).isSome ↔ ∃ t, t < k ∧ (A i t).isSome ∧ (B t j).isSome := by
  unfold mxm
  by_cases h : (List.range k).any (both A B i j) = true
  · simp only [h, ite_true, Option.isSome_some, true_iff]
    rw [List.any_eq_true] at h
    obtain ⟨t, ht, hb⟩ := h
    simp only [both, Bool.and_eq_true] at hb
    exact ⟨t, List.mem_range.1 ht, hb⟩
  · simp only [h, Bool.false_eq_true, ite_false, Option.isSome_none, false_iff]
    rintro ⟨t, ht, h1, h2⟩
    exact h (List.any_eq_true.2 ⟨t, List.mem_range.2 ht, by simp [both, h1, h2]⟩)

theorem mxmPT_isSome {k : Nat} {A B : Mat Int} {i j : Nat} :
    (mxmPT k A B i j).isSome ↔ ∃ t, t < k ∧ (A i t).isSome ∧ (B t j).isSome := mxm_isSome

/-- Over `PLUS_TIMES` the value (0 when absent) is the full sum over the inner dimension: the
indices outside the intersection contribute `0`. -/
theorem mxmPT_getD (k : Nat) (A B : Mat Int) (i j : Nat) :
    (mxmPT k A B i j).getD 0 = lsum (List.range k) (fun t => (A i t).getD 0 * (B t j).getD 0) := by
  have hz : ∀ t, both A B i j t = false → (A i t).getD 0 * (B t j).getD 0 = 0 := by
    intro t ht
    simp only [both, Bool.and_eq_false_iff] at ht
    rcases ht with h | h
    · cases hA : A i t
      · simp
      · rw [hA] at h; simp at h
    · cases hB : B t j
      · simp
      · rw [hB] at h; simp at h
  unfold mxmPT mxm
  by_cases h : (List.range k).any (both A B i j) = true
  · simp only [h, ite_true, Option.getD_some]
    rw [lsum_filter]
    refine lsum_congr (fun t _ => ?_)
    by_cases hb : both A B i j t = true
    · simp [hb]
    · simp only [Bool.not_eq_true] at hb; simp [hb, hz t hb]
  · simp only [h, Bool.false_eq_true, ite_false, Option.getD_none]
    refine (lsum_zero (fun t ht => hz t ?_)).symm
    cases hb : both A B i j t
    · rfl
    · exact absurd (List.any_eq_true.2 ⟨t, ht, hb⟩) h

end CchCoarsen
