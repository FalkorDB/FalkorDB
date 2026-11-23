/-
# Error-code handling per deserialize wrapper — the #2996 DoS and its fix

Every GraphBLAS call returns a `GrB_Info` (see `GrbInfo.lean`). The deserialize
wrappers in `vector.rs` / `matrix.rs` / `tensor.rs` decode an **entirely
attacker-controlled** buffer (`GRAPH.RESTORE`, `GRAPH.COPY`, replica full sync).
For those, an `assert_eq!` on the returned code is a remote crash: the module's
panic hook calls `std::process::exit(1)` (`src/module_init.rs`).

This file assigns each deserialize wrapper a `Disposition` (from `GrbInfo`) at
its cited line, axiomatises which codes the underlying GraphBLAS call can return
on a crafted blob (from the C source), and proves which wrappers are DoS-safe.

**Fixed by #2996 (main 1d15bf713).** `Vector::<bool>::decode_blob`
(vector.rs) and `Vector::<u64>::decode` (vector.rs) used to
`assert_eq!` on `GxB_Vector_deserialize`, which returns `GrB_INVALID_OBJECT` on
a corrupt blob (`GB_deserialize.c:51,83`); `Tensor::decode` (tensor.rs)
calls `decode_blob` for every multi-edge pair, so a crafted `GRAPH.RESTORE`
killed the server (repro then: `graph/tests/lean_graphblas_wrappers.rs`
`bug_tensor_decode_panics_on_corrupt_multi_edge_blob`). Both now
`if info != SUCCESS { return Err(..) }` (vector.rs) and pass the
expected type (`GrB_BOOL` / `GrB_UINT64`) so a well-formed blob of another type
is refused too (`GrB_DOMAIN_MISMATCH`). `decodeBlob_never_dies` /
`decodeU64_never_dies` prove no code kills the process.
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
  [.success, .invalidObject, .domainMismatch, .nullPointer, .outOfMemory, .invalidValue]
theorem vector_deserialize_corrupt_returns_invalid_object :
    Info.invalidObject ∈ vector_deserialize_reachable_codes := by decide

/-- Wrapper dispositions at their cited `file:line`. -/
structure Wrapper where
  name : String
  line : Nat
  disp : Disposition

/-- `Vector::<bool>::decode_blob` (vector.rs) on main: `GxB_Vector_deserialize`
with `GrB_BOOL`, any non-success code returned as `Err` (vector.rs). -/
def decodeBlob : Wrapper := ⟨"Vector::<bool>::decode_blob", 211, .propagate⟩

/-- `Vector::<u64>::decode` (vector.rs) on main: same, with `GrB_UINT64`
(vector.rs). Since #3643 both go through `deserialize_untrusted`, whose added refusals
(`check_blob_header`, `nvals` beyond the capacity) are `Err`s too. -/
def decodeU64 : Wrapper := ⟨"Vector::<u64>::decode", 484, .propagate⟩

/-- `Vector::<bool>::decode` (vector.rs): the hardened path — `Component::read` validates
lengths and NUL-termination, `Component::load` (#3643: `handling` forced to 0, a checked
`n_entries * type_size <= n_bytes`) then `if info != SUCCESS { …free…; return Err }`
for every GraphBLAS call. -/
def decodeBoolHardened : Wrapper := ⟨"Vector::<bool>::decode", 721, .propagate⟩

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

/-! ### `Matrix::decode` internal asserts (matrix.rs)

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

/-- #3643: `GxB_load_Matrix_from_Container` in `Matrix::decode` (matrix.rs) used to be
`debug_assert`ed, so a container the arrays could not back was ignored in release; it is now
`if info != SUCCESS { GrB_Matrix_free; return Err }`. -/
def matrixDecodeLoad : Wrapper := ⟨"Matrix::decode GxB_load_Matrix_from_Container", 787, .propagate⟩

theorem matrixDecodeLoad_never_dies (i : Info) :
    Disposition.run matrixDecodeLoad.disp i = some i.ok := by
  cases i <;> rfl

/-- A refused container is an error, not a matrix built on it. -/
theorem matrixDecodeLoad_rejects (i : Info) (h : i.ok = false) :
    Disposition.run matrixDecodeLoad.disp i = some false := by
  cases i <;> simp_all [matrixDecodeLoad, Disposition.run]

/-! ### `ContainerHeader::parse` (matrix.rs, #3643)

Only six scalar fields are taken from the payload's container bytes (no pointer, arena or
`*_future` word: the `PtrBlind` reliance of the old `copy_nonoverlapping` is gone), and they
must name a sparse or hypersparse, row- or column-major matrix within `max_dim`. -/

/-- `GxB_HYPERSPARSE = 1`, `GxB_SPARSE = 2`, `GxB_BITMAP = 4`, `GxB_FULL = 8`;
`GrB_ROWMAJOR = 0`, `GrB_COLMAJOR = 1`. -/
def headerOk (format orientation nrows ncols maxDim : Nat) : Bool :=
  (format == 1 || format == 2) && (orientation == 0 || orientation == 1) &&
    decide (nrows ≤ maxDim) && decide (ncols ≤ maxDim)

/-- Bitmap and full containers (which would allocate `nrows * ncols`) are refused. -/
theorem headerOk_refuses_dense (o r c d : Nat) :
    headerOk 4 o r c d = false ∧ headerOk 8 o r c d = false := by
  simp [headerOk]

/-- A sparse or hypersparse, row- or column-major header within the limit parses. -/
theorem headerOk_accepts (f o r c d : Nat) (hf : f = 1 ∨ f = 2) (ho : o = 0 ∨ o = 1)
    (hr : r ≤ d) (hc : c ≤ d) : headerOk f o r c d = true := by
  rcases hf with rfl | rfl <;> rcases ho with rfl | rfl <;> simp [headerOk, hr, hc]

/-! ### Who frees what in `Matrix::decode` (#3643 `ContainerGuard`)

`GxB_Container_new` allocates the container and five empty component vectors
(`GxB_Container_new_arena.c`: `p, h, b, i, x`). `decode` reads the five payload components
(pure Rust), checks the structure, then `load`s each into a fresh vector and stores it over its
slot. `ContainerGuard` frees the container and whatever its slots hold on every exit. Nothing
frees the vector a slot held before the store. -/

inductive Blk where
  | cont
  | init (k : Nat)
  | loaded (k : Nat)
deriving DecidableEq, Repr

/-- Where `decode` stops: a read or `check_structure` error, the `k`-th `load` failing
(after `k` stores), `GxB_load_Matrix_from_Container` refusing, or success. -/
inductive DecodeExit where
  | readFail
  | loadFail (k : Nat)
  | containerRefused
  | ok
deriving DecidableEq, Repr

def DecodeExit.stored : DecodeExit → Nat
  | .readFail => 0
  | .loadFail k => min k 5
  | .containerRefused => 5
  | .ok => 5

def allocated (e : DecodeExit) : List Blk :=
  .cont :: ((List.range 5).map .init ++ (List.range e.stored).map .loaded)

/-- The guard's `GxB_Container_free`: the container and its five slots. -/
def freedByGuard (e : DecodeExit) : List Blk :=
  .cont :: (List.range 5).map fun j => if j < e.stored then .loaded j else .init j

def leaked (e : DecodeExit) : List Blk := (allocated e).filter fun b => b ∉ freedByGuard e

/-- #3643 fixes the early error paths: a payload refused before any `load` leaks nothing
(before, the container and every vector decoded so far leaked). -/
theorem decode_readFail_no_leak : leaked .readFail = [] := by decide

/-- Still open: each store overwrites the empty vector `GxB_Container_new` put in the slot,
so a successful decode leaks those five (and a refused container the same five). -/
theorem decode_ok_leaks_initial_components :
    leaked .ok = [.init 0, .init 1, .init 2, .init 3, .init 4] ∧
    leaked .containerRefused = leaked .ok := by decide

/-- A failing `k`-th `load` leaks the initial vectors of the `k` slots already overwritten. -/
theorem decode_loadFail_leaks :
    leaked (.loadFail 0) = [] ∧ leaked (.loadFail 2) = [.init 0, .init 1] ∧
    leaked (.loadFail 4) = [.init 0, .init 1, .init 2, .init 3] := by decide

end GBW
