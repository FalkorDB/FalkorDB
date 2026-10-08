import GraphblasWrappers.Glue
/-
# Bulk row counting (#3022): `Vector::<bool>::from_sorted_indices`, `Vector::<u64>::{ptr, sum}`,
# `Matrix::count_in_rows`

`Graph::first_node_with_relationships` asks "do any of these rows hold an entry"
in one product. The C calls are axiomatised as the hypothesis `BulkSpec` (their
GraphBLAS-spec meaning); the Rust around them is proved.

| here | there |
| --- | --- |
| `fromSortedIndices` | `Vector::<bool>::from_sorted_indices` (vector.rs:149-171): `Self::new(nrows)`, early return on empty, else one `GxB_Vector_build_Scalar` of an iso `true` scalar |
| `VecU.ptr`          | `Vector::<u64>::ptr` (vector.rs:488) |
| `VecU.sum`          | `Vector::<u64>::sum` (vector.rs:494): `let mut sum = 0; GrB_Vector_reduce_UINT64(&sum, NULL, PLUS_MONOID, v, NULL)` |
| `countInRows`       | `Matrix::count_in_rows` (matrix.rs:774-792): `w = Vector::<u64>::new(ncols)`, `GrB_mxv(w, NULL, NULL, PLUS_PAIR_UINT64, A, rows, DESC_T0)`, `w.sum()` |

`BulkSpec` (GraphBLAS user guide): `GxB_Vector_build_Scalar` into an empty vector
stores exactly the given (distinct, in-range) indices; `GrB_mxv` with `DESC_T0`
computes `w = Aᵀ·x`, and over `PLUS_PAIR` (`PAIR(a,b) = 1`) entry `j` of `w` is the
number of `i` with `A(i,j)` and `x(i)` both present (absent when zero);
`GrB_Vector_reduce_UINT64` with the `PLUS` monoid is the sum of the stored values
(0 for an empty vector). Values are counts of stored entries, far below `2^64`.

Headline: `countInRows_spec` — `count_in_rows(x) = Σ_{i ∈ rows} |A(i, :)|`.
-/
namespace GBWBulk

/-- A matrix by its stored coordinates and column count. -/
structure MatC where
  ents : List (Nat × Nat)
  nr : Nat
  nc : Nat

/-- A vector: its size and the indices it stores (bool, iso `true`). -/
structure VecB where
  n : Nat
  pat : List Nat

/-- A `u64` vector as its dense view (an absent entry reads as 0) over `0..n`. -/
structure VecU where
  handle : Nat
  n : Nat
  val : Nat → Nat

/-- **Hypothesis** — the GraphBLAS C calls, by their spec. -/
structure BulkSpec where
  /-- `GrB_Vector_new(&v, GrB_BOOL, n)`. -/
  vNew : Nat → VecB
  vNew_spec : ∀ n, vNew n = ⟨n, []⟩
  /-- `GxB_Vector_build_Scalar(v, I, true, len)` on an empty vector. -/
  build : VecB → List Nat → VecB
  build_spec : ∀ n ids, ids.Pairwise (· < ·) → (∀ i ∈ ids, i < n) → build ⟨n, []⟩ ids = ⟨n, ids⟩
  /-- `GrB_mxv(w, NULL, NULL, GxB_PLUS_PAIR_UINT64, A, x, GrB_DESC_T0)` into a fresh `w` of size `ncols`. -/
  mxvT : MatC → VecB → VecU
  mxvT_spec : ∀ A x j, (mxvT A x).val j = (A.ents.filter (fun p => decide (p.2 = j) && decide (p.1 ∈ x.pat))).length
  mxvT_n : ∀ A x, (mxvT A x).n = A.nc
  /-- `GrB_Vector_reduce_UINT64(&s, NULL, GrB_PLUS_MONOID_UINT64, w, NULL)`. -/
  reduce : VecU → Nat
  reduce_spec : ∀ w, reduce w = ((List.range w.n).map w.val).sum

/-- `Vector::<bool>::from_sorted_indices` (vector.rs:149). -/
def fromSortedIndices (G : BulkSpec) (nrows : Nat) (indices : List Nat) : VecB :=
  let v := G.vNew nrows
  if indices.isEmpty then v else G.build v indices

/-- **`from_sorted_indices` is the indicator of `indices`**, on both branches,
under its documented (`debug_assert!`ed) precondition. -/
theorem fromSortedIndices_spec (G : BulkSpec) (nrows : Nat) (indices : List Nat)
    (hs : indices.Pairwise (· < ·)) (hb : ∀ i ∈ indices, i < nrows) :
    fromSortedIndices G nrows indices = ⟨nrows, indices⟩ := by
  unfold fromSortedIndices
  simp only [G.vNew_spec]
  split
  · rename_i h; rw [List.isEmpty_iff.mp h]
  · exact G.build_spec nrows indices hs hb

/-- `Vector::<u64>::ptr` (vector.rs:488): the handle. -/
def VecU.ptr (w : VecU) : Nat := w.handle
theorem VecU.ptr_eq (w : VecU) : w.ptr = w.handle := rfl

/-- `Vector::<u64>::sum` (vector.rs:494). -/
def VecU.sum (G : BulkSpec) (w : VecU) : Nat := G.reduce w

theorem VecU.sum_spec (G : BulkSpec) (w : VecU) : w.sum G = ((List.range w.n).map w.val).sum :=
  G.reduce_spec w

/-- `Matrix::count_in_rows` (matrix.rs:774). -/
def countInRows (G : BulkSpec) (A : MatC) (rows : VecB) : Nat := (G.mxvT A rows).sum G

theorem sum_map_add' (l : List Nat) (f g : Nat → Nat) :
    (l.map (fun j => f j + g j)).sum = (l.map f).sum + (l.map g).sum := by
  induction l with
  | nil => rfl
  | cons a t ih => simp [List.sum_cons, ih]; omega

theorem sum_indicator (n c : Nat) (b : Bool) :
    ((List.range n).map (fun j => if (decide (c = j) && b) = true then 1 else 0)).sum =
      if c < n ∧ b = true then 1 else 0 := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [List.range_succ, List.map_append, List.sum_append, ih]
    cases b
    · simp
    · by_cases hc : c = k
      · subst hc; simp
      · have : (c < k + 1 ↔ c < k) := by omega
        simp [hc, this]

/-- Double counting: summing, over every column, the entries in that column whose
row is selected counts every selected entry once. -/
theorem double_count (S : Nat → Bool) (nc : Nat) :
    ∀ (L : List (Nat × Nat)), (∀ p ∈ L, p.2 < nc) →
      ((List.range nc).map (fun j => (L.filter (fun p => decide (p.2 = j) && S p.1)).length)).sum =
        (L.filter (fun p => S p.1)).length
  | [], _ => by
    simp only [List.filter_nil, List.length_nil]
    rename_i h0; clear h0
    induction nc with
    | zero => rfl
    | succ k ih => rw [List.range_succ, List.map_append, List.sum_append, ih]; rfl
  | p :: L, h => by
    have ih := double_count S nc L (fun q hq => h q (List.mem_cons_of_mem _ hq))
    have e : ∀ j, ((p :: L).filter (fun p => decide (p.2 = j) && S p.1)).length =
        (L.filter (fun p => decide (p.2 = j) && S p.1)).length +
        (if (decide (p.2 = j) && S p.1) = true then 1 else 0) := by
      intro j; rw [List.filter_cons]; split <;> simp_all
    simp only [e]
    rw [sum_map_add', ih, sum_indicator, List.filter_cons]
    have hp := h p List.mem_cons_self
    cases hs : S p.1 <;> simp [hp]

/-- **`count_in_rows` counts the entries in the selected rows**:
`Σ_j |{i ∈ rows : A(i,j)}| = |{(i,j) ∈ A : i ∈ rows}|`, since every stored entry
lies in exactly one column `< ncols`. -/
theorem countInRows_spec (G : BulkSpec) (A : MatC) (rows : VecB) (hc : ∀ p ∈ A.ents, p.2 < A.nc) :
    countInRows G A rows = (A.ents.filter (fun p => decide (p.1 ∈ rows.pat))).length := by
  unfold countInRows
  rw [VecU.sum_spec, G.mxvT_n, show (G.mxvT A rows).val = _ from funext (G.mxvT_spec A rows)]
  exact double_count (fun i => decide (i ∈ rows.pat)) A.nc A.ents hc

/-- Over `from_sorted_indices(nrows, ids)`, the count is over `ids`. -/
theorem countInRows_fromSorted (G : BulkSpec) (A : MatC) (ids : List Nat)
    (hs : ids.Pairwise (· < ·)) (hb : ∀ i ∈ ids, i < A.nr) (hc : ∀ p ∈ A.ents, p.2 < A.nc) :
    countInRows G A (fromSortedIndices G A.nr ids) = (A.ents.filter (fun p => decide (p.1 ∈ ids))).length := by
  rw [fromSortedIndices_spec G A.nr ids hs hb]; exact countInRows_spec G A _ hc

/-- Positive iff some selected row holds an entry — the question
`first_node_with_relationships` asks. -/
theorem countInRows_pos (G : BulkSpec) (A : MatC) (rows : VecB) (hc : ∀ p ∈ A.ents, p.2 < A.nc) :
    0 < countInRows G A rows ↔ ∃ p ∈ A.ents, p.1 ∈ rows.pat := by
  rw [countInRows_spec G A rows hc, List.length_pos_iff]
  constructor
  · intro h
    obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ h
    rw [List.mem_filter, decide_eq_true_eq] at hp; exact ⟨p, hp⟩
  · rintro ⟨p, hp, hr⟩ hnil
    have : p ∈ A.ents.filter (fun p => decide (p.1 ∈ rows.pat)) := List.mem_filter.mpr ⟨hp, by simpa using hr⟩
    rw [hnil] at this; cases this

end GBWBulk
