/-
# Index and dimension conversions

Everything on the Rust side of `matrix.rs` / `vector.rs` / `tensor.rs` that
touches `GrB_Index` is a `u64` (`pub type GrB_Index = u64;`, mod.rs), so the
`u64 ↔ GrB_Index` conversion is the identity and cannot truncate. The casts that
*can* lose information are:

* `rows.len() as u64` in `Matrix::build` / `assign_product_true` /
  `Matrix::<u64>::build` (matrix.rs:1294, 1400, 1423) — `usize → u64`;
* `n_bytes as usize` / `blob_size as usize` in `vector.rs` (`Decode`, blob io) —
  `u64 → usize`;
* the dimension bounds GraphBLAS itself enforces (`GB_NMAX = 2^60`).

This file models each and states when it is lossless, and pins the two
dimension constants (`GrB_INDEX_MAX`, `ME_DIM`, `ME_NARROW_NCOLS`) against the
C `GB_NMAX`.
-/

namespace GBW

/-- The bound GraphBLAS enforces in `GB_Matrix_new` / `GB_new`:
`nrows > GB_NMAX || ncols > GB_NMAX ⇒ GrB_INVALID_VALUE`.
`GB_NMAX = 1 << 60` (`matrix/include/GB_index.h:19`). -/
def GB_NMAX : Nat := 1 <<< 60

/-- `matrix/include/GB_index.h:20`, `GB_NMAX32 = 1 << 31`. GraphBLAS keeps
32-bit index arrays while a dimension is `≤ GB_NMAX32`. -/
def GB_NMAX32 : Nat := 1 <<< 31

/-- `tensor.rs:144` `GrB_INDEX_MAX = (1 << 60) - 1`. -/
def GrB_INDEX_MAX : Nat := (1 <<< 60) - 1

/-- `tensor.rs:153` `ME_DIM = GrB_INDEX_MAX + 1`. -/
def ME_DIM : Nat := GrB_INDEX_MAX + 1

/-- `tensor.rs:169` `ME_NARROW_NCOLS = 1 << 31`. -/
def ME_NARROW_NCOLS : Nat := 1 <<< 31

/-- AXIOM-OK: `GB_Matrix_new` (`Source/matrix/GB_Matrix_new.c:28-31`) returns
`GrB_INVALID_VALUE` iff a dimension exceeds `GB_NMAX`, else `SUCCESS`. Modelled
as a pure predicate on the requested dims. -/
def matrixNewOk (nrows ncols : Nat) : Bool :=
  decide (nrows ≤ GB_NMAX ∧ ncols ≤ GB_NMAX)

/-- `ME_DIM` is exactly the largest dimension GraphBLAS accepts. So an `me`
block declared with `ME_DIM` rows always constructs — the comment at
tensor.rs:153 ("one more than the largest key `compound_key` can produce") is
consistent with the C bound, not one past it. -/
theorem me_dim_is_max : ME_DIM = GB_NMAX := by
  unfold ME_DIM GrB_INDEX_MAX GB_NMAX
  decide

theorem me_dim_constructs : matrixNewOk ME_DIM ME_NARROW_NCOLS = true := by
  simp only [matrixNewOk, ME_DIM, GrB_INDEX_MAX, ME_NARROW_NCOLS, GB_NMAX]
  decide

theorem me_dim_wide_constructs : matrixNewOk ME_DIM ME_DIM = true := by
  simp only [matrixNewOk, ME_DIM, GrB_INDEX_MAX, GB_NMAX]
  decide

/-! ### `record_created` / bug #2892 boundary (already filed; modelled here)

`IdSpace::record_created` (id_space.rs:325) refuses only `u64::MAX`, so an id in
`(GrB_INDEX_MAX, u64::MAX)` reaches `mark_nodes_live`, which resizes a matrix to
`id + 1`. If `id + 1 > GB_NMAX` the resize returns `GrB_INVALID_VALUE` and the
`assert_eq!` at matrix.rs:1318/1064 panics. This is the arithmetic of that. -/

/-- A create of node `id` resizes matrices to `id + 1`. It survives iff that
dimension is within the GraphBLAS bound. -/
def resizeForIdOk (id : Nat) : Bool := matrixNewOk (id + 1) 1

/-- The safe id ceiling: `id + 1 ≤ GB_NMAX`, i.e. `id ≤ GB_NMAX - 1 =
GrB_INDEX_MAX`. Any larger id crashes `GrB_Matrix_new`/`resize`. -/
theorem create_safe_iff (id : Nat) : resizeForIdOk id = true ↔ id ≤ GrB_INDEX_MAX := by
  have hnmax : GB_NMAX = 1152921504606846976 := by decide
  have hidx : GrB_INDEX_MAX = 1152921504606846975 := by decide
  simp only [resizeForIdOk, matrixNewOk, hnmax, hidx, decide_eq_true_eq]
  omega

/-- The concrete #2892 counterexample: `2^61` crashes. -/
theorem create_2pow61_crashes : resizeForIdOk (1 <<< 61) = false := by
  simp only [resizeForIdOk, matrixNewOk, GB_NMAX]
  decide

/-- And `record_created`'s only guard (`id ≠ u64::MAX`) lets it through: the
id `2^61` is far below `u64::MAX`, so the wrapper never sees the code that would
have refused it. (`u64::MAX = 2^64 - 1`.) -/
theorem guard_misses_2pow61 : (1 <<< 61 : Nat) ≠ 2^64 - 1 := by decide

/-! ### `usize → u64` on `.len()` casts

`rows.len() as u64` (matrix.rs:1294 etc.). On a 64-bit target `usize` is 64-bit,
so this is lossless for any real slice; the model records the fact so a future
32-bit port would flag it. Here `usize` is `Nat` bounded by `2^64`. -/

/-- A slice length cast to `u64` is lossless exactly when it fits 64 bits — true
for any allocatable slice (`isize::MAX < 2^64`). -/
theorem len_cast_lossless (n : Nat) (h : n < 2 ^ 64) : (n % 2 ^ 64) = n := Nat.mod_eq_of_lt h

/-! ### `u64 → usize` blob-size casts (`vector.rs`)

`n_bytes as usize`, `blob_size as usize`. On 64-bit this is the identity. The
`Decode` path *validates* `n_bytes as usize == arr_data.len()` (vector.rs:332)
before the cast drives `copy_nonoverlapping`, so even under truncation the copy
length matches the source buffer. This is the check that makes it safe. -/

/-- The decode guard: reject unless the declared length equals the buffer
length. Modelled as the exact comparison the Rust performs. -/
def blobLenOk (nBytes bufLen : Nat) : Bool := decide (nBytes = bufLen)

/-- When the guard passes, the length that drives the copy is the buffer's own
length, so the read is in bounds regardless of what `n_bytes` claimed. -/
theorem blob_copy_in_bounds (nBytes bufLen : Nat) (h : blobLenOk nBytes bufLen = true) :
    nBytes = bufLen := by
  simpa [blobLenOk] using h

end GBW
