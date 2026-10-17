/-
# `GrB_Info`: the GraphBLAS status code, and the axiomatised C-API contracts

`mod.rs:GrB_Info` (`graph/src/graph/graphblas/mod.rs`, the bindgen enum) is the
return type of every GraphBLAS call. The Rust wrappers dispatch on it in one of
three ways, and this file names the code enum and the C-API specs the rest of
the development reasons against.

The specs are `axiom`s — they are the GraphBLAS C library's documented
behaviour, which Lean cannot see the source of — each carries an
`-- AXIOM-OK:` comment citing the header. Everything downstream that is Rust
logic is a `theorem`.
-/

namespace GBW

/-- `mod.rs:GrB_Info` (bindgen enum, `#[repr(i32)]`). Only the codes the
wrappers actually branch on are named; the rest fold into `other`. -/
inductive Info where
  | success            -- GrB_SUCCESS = 0
  | noValue            -- GrB_NO_VALUE = 1
  | exhausted          -- GxB_EXHAUSTED = 7089
  | nullPointer        -- GrB_NULL_POINTER = -2
  | invalidValue       -- GrB_INVALID_VALUE = -3
  | invalidIndex       -- GrB_INVALID_INDEX = -4
  | dimensionMismatch  -- GrB_DIMENSION_MISMATCH = -6
  | notImplemented     -- GrB_NOT_IMPLEMENTED = -8
  | outOfMemory        -- GrB_OUT_OF_MEMORY = -102
  | invalidObject      -- GrB_INVALID_OBJECT = -104
  | indexOutOfBounds   -- GrB_INDEX_OUT_OF_BOUNDS = -105
  | panic              -- GrB_PANIC = -101
deriving DecidableEq, Repr

/-- Is this the success code? The `info == GrB_Info::GrB_SUCCESS` test the
wrappers write. -/
def Info.ok : Info → Bool
  | .success => true
  | _ => false

/-- How a GraphBLAS return code reaches the caller of a wrapper. A wrapper is
classified by which of these it does with a non-success `Info`. -/
inductive Disposition where
  /-- `assert_eq!(info, SUCCESS)` — an unwinding panic. In the shipped module
      the panic hook calls `std::process::exit(1)`, so this is a process kill.
      `matrix.rs` uses it for `GrB_Matrix_new`, `GrB_Matrix_dup`,
      `GxB_Container_new`, `GxB_Iterator_new`; `vector.rs` for
      `GxB_Vector_deserialize`, `GrB_Vector_new` (some paths). -/
  | assertPanic
  /-- `debug_assert_eq!` — panic in debug, silently ignored in release. Used
      for the mutating ops (`set`, `resize`, `wait`, folds …) whose failure the
      wrappers treat as "cannot happen". -/
  | debugAssert
  /-- Mapped to `Err(String)` / `None` / a `bool` and returned. The only
      graceful path. `vector.rs:Decode` uses it for `GxB_Type_from_name`,
      `GrB_Vector_new`, `GxB_Vector_load`; `matrix.rs:get`, `contains` map to
      `Option`/`bool`. -/
  | propagate
deriving DecidableEq, Repr

/-- What a disposition does with a code: `some panic` = process death,
`some ok?` = graceful (`ok? = true` on success), `none` = nothing observable
(release `debug_assert`). Models the runtime, so `debugAssert` is the *release*
build (CLAUDE.md: release wraps, debug panics). -/
def Disposition.run (d : Disposition) (i : Info) : Option Bool :=
  match d with
  | .assertPanic => if i.ok then some true else none  -- none here = "would panic"
  | .debugAssert => some i.ok                          -- release: swallow, keep going
  | .propagate   => some i.ok

/-- A disposition is *safe on `i`* when a non-success `i` cannot kill the
process. Only `propagate` (and `debugAssert`, which merely swallows) are safe;
`assertPanic` is safe only on `success`. -/
def Disposition.safeOn (d : Disposition) (i : Info) : Prop :=
  d = .propagate ∨ d = .debugAssert ∨ i.ok = true

theorem propagate_safe (i : Info) : Disposition.safeOn .propagate i := Or.inl rfl

theorem assertPanic_safe_iff (i : Info) :
    Disposition.safeOn .assertPanic i ↔ i.ok = true := by
  constructor
  · rintro (h | h | h) <;> first | cases h | exact h
  · intro h; exact Or.inr (Or.inr h)

/-- The reachability question that matters: an `assertPanic` wrapper is a DoS
vector exactly when some *reachable* non-success code exists for its call. This
predicate is what a wrapper audit fills in per call. -/
def reachableNonSuccess (codes : List Info) : Prop :=
  ∃ i ∈ codes, i.ok = false

/-- If a call can return a non-success code and the wrapper `assertPanic`s on
it, the process can die. This is the shape of bug #2892 and of the
`decode_blob` finding. -/
theorem assertPanic_dies_if_reachable
    (codes : List Info) (h : reachableNonSuccess codes) :
    ∃ i ∈ codes, Disposition.run .assertPanic i = none := by
  obtain ⟨i, hmem, hok⟩ := h
  refine ⟨i, hmem, ?_⟩
  simp only [Disposition.run, Info.ok] at hok ⊢
  cases i <;> simp_all

end GBW
