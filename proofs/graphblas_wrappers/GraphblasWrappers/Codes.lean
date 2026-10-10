/-
# Error-code handling per deserialize wrapper — one confirmed DoS, one hardened path

Every GraphBLAS call returns a `GrB_Info` (see `GrbInfo.lean`). The deserialize
wrappers in `vector.rs` / `matrix.rs` / `tensor.rs` decode an **entirely
attacker-controlled** buffer (`GRAPH.RESTORE`, `GRAPH.COPY`, replica full sync).
For those, an `assert_eq!` on the returned code is a remote crash: the module's
panic hook calls `std::process::exit(1)` (`src/module_init.rs:158`).

This file assigns each deserialize wrapper a `Disposition` (from `GrbInfo`) at
its cited line, axiomatises which codes the underlying GraphBLAS call can return
on a crafted blob (from the C source), and proves which wrappers are DoS-safe and
which are not.

The confirmed bug is `Vector::<bool>::decode_blob` (and the identically-shaped
`Vector::<u64>::decode`): both `assert_eq!` on `GxB_Vector_deserialize`, which
returns `GrB_INVALID_OBJECT` on a corrupt blob (`GB_deserialize.c:51,83`).
`Tensor::decode` (tensor.rs:1599) calls `decode_blob` for every multi-edge pair,
so a crafted `GRAPH.RESTORE` kills the server — reproduced in
`graph/tests/lean_graphblas_wrappers.rs`
(`bug_tensor_decode_panics_on_corrupt_multi_edge_blob`, live-server repro in a local repro script).
-/

import GraphblasWrappers.GrbInfo

namespace GBW

/-- C-API spec (as data, no Lean axiom): `GxB_Vector_deserialize` → `GB_deserialize` returns
`GrB_INVALID_OBJECT` on a blob whose header/section sizes are inconsistent
(`Source/serialize/GB_deserialize.c:51,83`), `GrB_DOMAIN_MISMATCH` on a type
mismatch (`:136,148,156`), `GrB_NULL_POINTER`/`GrB_OUT_OF_MEMORY` otherwise, and
`GrB_SUCCESS` only on a well-formed blob. So a corrupt blob is a *reachable*
non-success code. -/
def vector_deserialize_reachable_codes : List Info :=
  [.success, .invalidObject, .nullPointer, .outOfMemory, .invalidValue]
theorem vector_deserialize_corrupt_returns_invalid_object :
    Info.invalidObject ∈ vector_deserialize_reachable_codes := by decide

/-- Wrapper dispositions at their cited `file:line`. -/
structure Wrapper where
  name : String
  line : Nat
  disp : Disposition

/-- `Vector::<bool>::decode_blob` (vector.rs:211): `assert_eq!(info, SUCCESS,
"GxB_Vector_deserialize failed")` at vector.rs:222. -/
def decodeBlob : Wrapper := ⟨"Vector::<bool>::decode_blob", 211, .assertPanic⟩

/-- `Vector::<u64>::decode` (vector.rs:253): same `assert_eq!` shape. -/
def decodeU64 : Wrapper := ⟨"Vector::<u64>::decode", 253, .assertPanic⟩

/-- `Vector::<bool>::decode` (vector.rs:345): the hardened path — validates
lengths and NUL-termination, then `if info != SUCCESS { …free…; return Err }`
for every GraphBLAS call. -/
def decodeBoolHardened : Wrapper := ⟨"Vector::<bool>::decode", 345, .propagate⟩

/-- **Confirmed bug.** `decode_blob` (and `Vector::<u64>::decode`) `assertPanic`
on a code that a corrupt blob reaches, so a crafted payload kills the process. -/
theorem decodeBlob_is_dos :
    decodeBlob.disp = .assertPanic ∧
    Disposition.run decodeBlob.disp Info.invalidObject = none := by
  refine ⟨rfl, ?_⟩
  decide

theorem decodeU64_is_dos :
    Disposition.run decodeU64.disp Info.invalidObject = none := by decide

/-- The general statement: an `assertPanic` deserialize wrapper is a DoS because
`vector_deserialize_reachable_codes` contains a non-success code. -/
theorem assertPanic_deserialize_dos :
    ∃ i ∈ vector_deserialize_reachable_codes,
      Disposition.run .assertPanic i = none := by
  apply assertPanic_dies_if_reachable
  exact ⟨Info.invalidObject, vector_deserialize_corrupt_returns_invalid_object, rfl⟩

/-- **The hardened path is safe.** `Vector::<bool>::decode` propagates every
code, so no blob — however malformed — can crash through it. Contrast with
`decodeBlob`: same input, opposite outcome, which is the fix (make the two
`assert_eq!`s into the `if info != SUCCESS return Err` the hardened path already
uses). -/
theorem decodeBool_safe (i : Info) :
    Disposition.run decodeBoolHardened.disp i = some i.ok := by
  cases i <;> rfl

theorem decodeBool_never_dies (i : Info) :
    Disposition.run decodeBoolHardened.disp i ≠ none := by
  cases i <;> simp [decodeBoolHardened, Disposition.run]

/-- Side by side: on the very same non-success code the two wrappers diverge —
`decode` returns an error, `decode_blob` dies. -/
theorem hardened_vs_blob_on_invalid_object :
    Disposition.run decodeBoolHardened.disp Info.invalidObject = some false ∧
    Disposition.run decodeBlob.disp Info.invalidObject = none := by
  refine ⟨?_, ?_⟩ <;> decide

/-! ### `Matrix::decode` internal asserts (matrix.rs:445, 477)

`GxB_Container_new` and `GrB_Matrix_new` are called with *fixed* arguments
(a fresh container, dims `0×0`), so their only non-success code is
`GrB_OUT_OF_MEMORY`, which is not attacker-reachable through the payload. The
attacker-controlled decode of the five component vectors goes through
`Vector::<bool>::decode` — the hardened, propagating path — so `Matrix::decode`
does not add a code-assertion DoS beyond the container leak (see `Own`/report).
Its `GrB_Matrix_new(_, GrB_BOOL, 0, 0)` always satisfies the dimension bound. -/

/-- The internal `GrB_Matrix_new(_, _, 0, 0)` in `Matrix::decode` always
succeeds on the dimension check (`0 ≤ GB_NMAX`), so its `assert_eq!` is not
payload-reachable. -/
theorem matrix_decode_internal_new_ok :
    (Nat.ble 0 (1 <<< 60)) = true := by decide

end GBW
