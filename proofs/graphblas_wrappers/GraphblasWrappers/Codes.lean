/-
# Error-code handling per deserialize wrapper — the #2996 DoS and its fix

Every GraphBLAS call returns a `GrB_Info` (see `GrbInfo.lean`). The deserialize
wrappers in `vector.rs` / `matrix.rs` / `tensor.rs` decode an **entirely
attacker-controlled** buffer (`GRAPH.RESTORE`, `GRAPH.COPY`, replica full sync).
For those, an `assert_eq!` on the returned code is a remote crash: the module's
panic hook calls `std::process::exit(1)` (`src/module_init.rs:158`).

This file assigns each deserialize wrapper a `Disposition` (from `GrbInfo`) at
its cited line, axiomatises which codes the underlying GraphBLAS call can return
on a crafted blob (from the C source), and proves which wrappers are DoS-safe.

**Fixed by #2996 (main 1d15bf713).** `Vector::<bool>::decode_blob`
(vector.rs:211) and `Vector::<u64>::decode` (vector.rs:256) used to
`assert_eq!` on `GxB_Vector_deserialize`, which returns `GrB_INVALID_OBJECT` on
a corrupt blob (`GB_deserialize.c:51,83`); `Tensor::decode` (tensor.rs:1622)
calls `decode_blob` for every multi-edge pair, so a crafted `GRAPH.RESTORE`
killed the server (repro then: `graph/tests/lean_graphblas_wrappers.rs`
`bug_tensor_decode_panics_on_corrupt_multi_edge_blob`). Both now
`if info != SUCCESS { return Err(..) }` (vector.rs:226-228, 271-273) and pass the
expected type (`GrB_BOOL` / `GrB_UINT64`) so a well-formed blob of another type
is refused too (`GrB_DOMAIN_MISMATCH`). `decodeBlob_never_dies` /
`decodeU64_never_dies` prove no code kills the process; the old wrappers are
kept as pre-#2996 counterexamples (removed; in 17ccc5412) history.
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

/-- `Vector::<bool>::decode_blob` (vector.rs:211) on main: `GxB_Vector_deserialize`
with `GrB_BOOL`, any non-success code returned as `Err` (vector.rs:226-228). -/
def decodeBlob : Wrapper := ⟨"Vector::<bool>::decode_blob", 211, .propagate⟩

/-- `Vector::<u64>::decode` (vector.rs:256) on main: same, with `GrB_UINT64`
(vector.rs:271-273). -/
def decodeU64 : Wrapper := ⟨"Vector::<u64>::decode", 256, .propagate⟩

/-- `Vector::<bool>::decode` (vector.rs:350): the hardened path — validates
lengths and NUL-termination, then `if info != SUCCESS { …free…; return Err }`
for every GraphBLAS call. -/
def decodeBoolHardened : Wrapper := ⟨"Vector::<bool>::decode", 350, .propagate⟩

/-- **Main: the fix.** On every code — including `GrB_INVALID_OBJECT` from a
corrupt blob and `GrB_DOMAIN_MISMATCH` from a blob of the wrong type — both
wrappers return (an `Err` on non-success); the process never dies. -/
theorem decodeBlob_never_dies (i : Info) : Disposition.run decodeBlob.disp i = some i.ok := by
  cases i <;> rfl

theorem decodeU64_never_dies (i : Info) : Disposition.run decodeU64.disp i = some i.ok := by
  cases i <;> rfl

/-- In particular every code `GxB_Vector_deserialize` can return on a crafted
blob is safe now. -/
theorem decodeBlob_safe_on_reachable :
    ∀ i ∈ vector_deserialize_reachable_codes,
      Disposition.safeOn decodeBlob.disp i ∧ Disposition.safeOn decodeU64.disp i :=
  fun _ _ => ⟨Or.inl rfl, Or.inl rfl⟩

/-- A corrupt blob is refused with an error, not accepted. -/
theorem decodeBlob_rejects_invalid_object :
    Disposition.run decodeBlob.disp Info.invalidObject = some false := rfl

/-- The general statement: an `assertPanic` deserialize wrapper is a DoS because
`vector_deserialize_reachable_codes` contains a non-success code. -/
theorem assertPanic_deserialize_dos :
    ∃ i ∈ vector_deserialize_reachable_codes,
      Disposition.run .assertPanic i = none := by
  apply assertPanic_dies_if_reachable
  exact ⟨Info.invalidObject, vector_deserialize_corrupt_returns_invalid_object, rfl⟩

/-- **The hardened path is safe.** `Vector::<bool>::decode` propagates every
code, so no blob — however malformed — can crash through it. Before #2996,
`decode_blob` gave the opposite outcome on the same input; #2996 made the two
`assert_eq!`s into the `if info != SUCCESS return Err` this path already used. -/
theorem decodeBool_safe (i : Info) :
    Disposition.run decodeBoolHardened.disp i = some i.ok := by
  cases i <;> rfl

theorem decodeBool_never_dies (i : Info) :
    Disposition.run decodeBoolHardened.disp i ≠ none := by
  cases i <;> simp [decodeBoolHardened, Disposition.run]

/-- Main: they now agree on every code. -/
theorem hardened_eq_blob (i : Info) :
    Disposition.run decodeBlob.disp i = Disposition.run decodeBoolHardened.disp i := rfl

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
