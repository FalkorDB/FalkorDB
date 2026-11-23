import GraphblasWrappers.Codes
import GraphblasWrappers.Glue
/-
# `Decode<19> for Matrix<T>` (matrix.rs, #3080): `OwnedContainer`, `ContainerHeader`

| here | there |
| --- | --- |
| `fieldSpans`, `field_in_bounds` | `ContainerHeader::field` (offsets of `GxB_Container_struct`, 608 bytes) |
| `flag`, `formatOk`             | `ContainerHeader::parse` (the `flag` closure) / the format check in `decode` |
| `adopt`                        | `decode` writing the nine scalar fields into the fresh container |
| `Heap`, `run`                  | `OwnedContainer::{new, replace, drop}` and the body of `decode` |
| `decodeCalls`                  | the four GraphBLAS calls `decode` makes itself, each now `propagate` |

Before #3080 `decode` `copy_nonoverlapping`-ed the whole payload header over the
container `GxB_Container_new` had just filled, losing its five empty component
vectors (a leak on every RDB load, RESTORE and full sync), and `?` on a
component decode returned without freeing the container. Now:

* `decode_no_leak_no_double_free`: whichever call fails (any of the nine), every
  free hits a live block once, and the heap afterwards holds exactly the
  returned matrix (`Ok`) or nothing (`Err`).
* `adopt_ptr_blind`: the container handed to `GxB_load_Matrix_from_Container`
  does not depend on the payload's pointer fields (they are never read), so the
  `PtrBlind` hypothesis is no longer needed for decode.
* `decodeCalls_never_die`: `GxB_Container_new`, `GrB_Matrix_new`,
  `GxB_load_Matrix_from_Container` and `GrB_Matrix_wait` all return their
  failure as `Err` (the last two were `debug_assert`ed: a matrix that failed to
  load was returned in release).
-/
namespace GBWMatDecode
open GBW GBWGlue

/-! ## `ContainerHeader::parse` -/

/-- `size_of::<GxB_Container_struct>()` (bindgen layout check, mod.rs). -/
def containerSize : Nat := 608

/-- `(offset_of!, N)` of each field `parse` reads: `nrows, ncols, nrows_nonempty,
ncols_nonempty, nvals` (8 bytes), `format, orientation` (4), `iso, jumbled` (1). -/
def fieldSpans : List (Nat × Nat) :=
  [(0, 8), (8, 8), (16, 8), (24, 8), (32, 8), (128, 4), (132, 4), (448, 1), (449, 1)]

/-- Every `bytes[offset..offset + N]` is in range, since `decode` refused a buffer
shorter than `CONTAINER_STRUCT_SIZE`: `field`'s slice and `try_into` never panic. -/
theorem field_in_bounds (len : Nat) (hlen : containerSize ≤ len) :
    ∀ s ∈ fieldSpans, s.1 + s.2 ≤ len := by
  intro s hs
  have : s.1 + s.2 ≤ containerSize := by
    simp only [fieldSpans, List.mem_cons, List.mem_nil_iff, or_false] at hs
    rcases hs with h | h | h | h | h | h | h | h | h <;> subst h <;> decide
  omega

/-- The `flag` closure: a byte is a `bool` only if it is 0 or 1. -/
def flag (b : Nat) : Except String Bool :=
  if b = 0 then .ok false else if b = 1 then .ok true else .error "expected 0 or 1"

theorem flag_ok_iff (b : Nat) : (∃ v, flag b = .ok v) ↔ b ≤ 1 := by
  unfold flag
  constructor
  · rintro ⟨v, h⟩
    by_cases h0 : b = 0
    · omega
    · by_cases h1 : b = 1
      · omega
      · simp [h0, h1] at h
  · intro h
    by_cases h0 : b = 0
    · exact ⟨false, by simp [h0]⟩
    · exact ⟨true, by simp [show b = 1 by omega]⟩

/-- `GxB_HYPERSPARSE, GxB_SPARSE, GxB_BITMAP, GxB_FULL` (mod.rs: 1, 2, 4, 8). -/
def formatOk (f : Int) : Bool := [1, 2, 4, 8].contains f

theorem formatOk_iff (f : Int) : formatOk f = true ↔ f = 1 ∨ f = 2 ∨ f = 4 ∨ f = 8 := by
  simp [formatOk]

/-! ## The header handed to `GxB_load_Matrix_from_Container` -/

/-- `decode` writes the payload field into the fresh container where `copy` is set
(the nine scalars), and keeps the fresh container's field everywhere else
(its component pointers, `Y`, the `*_future` arrays). -/
def adopt : List HField → List HField → List Bool → List HField
  | f :: fs, p :: ps, c :: cs => (if c then p else f) :: adopt fs ps cs
  | fs, _, _ => fs

/-- Only scalar fields are copied: the mask is set on non-pointer fields. -/
def ScalarMask : List HField → List Bool → Prop
  | p :: ps, c :: cs => (c = true → p.ptr = false) ∧ ScalarMask ps cs
  | _, _ => True

/-- **The loaded header ignores the payload's pointer fields.** -/
theorem adopt_ptr_blind (fresh payload : List HField) (mask : List Bool)
    (h : ScalarMask payload mask) :
    adopt fresh (scrub payload) mask = adopt fresh payload mask := by
  induction fresh generalizing payload mask with
  | nil => cases payload <;> cases mask <;> rfl
  | cons f fs ih =>
    cases payload with
    | nil => cases mask <;> rfl
    | cons p ps =>
      cases mask with
      | nil => rfl
      | cons c cs =>
        obtain ⟨hc, hr⟩ := h
        simp only [scrub, List.map_cons, adopt] at ih ⊢
        rw [ih ps cs hr]
        cases c
        · rfl
        · simp [hc rfl]

/-- A field the mask does not copy is the fresh container's. -/
theorem adopt_keeps_uncopied (f p : HField) (fs ps : List HField) (cs : List Bool) :
    adopt (f :: fs) (p :: ps) (false :: cs) = f :: adopt fs ps cs := rfl

/-! ## Ownership: `OwnedContainer` and the decode body -/

/-- Live heap blocks (container structs, component vectors, matrices) by id. -/
structure Heap where
  next : Nat
  live : List Nat
deriving DecidableEq, Repr

def alloc (h : Heap) : Nat × Heap := (h.next, ⟨h.next + 1, h.next :: h.live⟩)

/-- Freeing a block that is not live (double free / wild free) is `none`. -/
def free (h : Heap) (x : Nat) : Option Heap :=
  if x ∈ h.live then some ⟨h.next, h.live.erase x⟩ else none

/-- `GxB_Container_new`: the struct and five empty component vectors. -/
def containerNew (h : Heap) : (Nat × List Nat) × Heap :=
  let (c, h) := alloc h
  let (x, h) := alloc h
  let (hh, h) := alloc h
  let (p, h) := alloc h
  let (i, h) := alloc h
  let (b, h) := alloc h
  ((c, [x, hh, p, i, b]), h)

/-- `OwnedContainer::drop` → `GxB_Container_free`: every component, then the struct. -/
def containerFree (h : Heap) (c : Nat) (slots : List Nat) : Option Heap := do
  let h ← slots.foldlM free h
  free h c

/-- `OwnedContainer::replace`: free the slot's vector, install `v`. -/
def replace (h : Heap) (slots : List Nat) (k : Nat) (v : Nat) : Option (Heap × List Nat) := do
  let h ← free h (slots.getD k 0)
  pure (h, slots.set k v)

/-- Which call fails: `ok[0]` `GxB_Container_new`, `ok[1..5]` the decodes of
`x, h, p, i, b`, `ok[6]` `GrB_Matrix_new`, `ok[7]` the load, `ok[8]` the wait.
ASSUMPTION: a failed `Vector::<Untyped>::decode` is atomic, i.e. it leaves nothing
allocated, so it allocates nothing here. `decodeBool_safe` proves only that it returns `Err`
on every code, not its own cleanup (`GxB_Type_from_name`, allocation, `GxB_Vector_load`), so
the theorems below, and the `Matrix::decode` COVERAGE row, are CONDITIONAL on that contract. Returns the
heap, the container's slots, and whether every decode succeeded. -/
def decodeVecs (ok : List Bool) : Nat → Heap → List Nat → Option (Heap × List Nat × Bool)
  | 0, h, slots => some (h, slots, true)
  | n + 1, h, slots =>
    let k := 5 - (n + 1)
    if ok.getD (1 + k) true then
      let (v, h) := alloc h
      match replace h slots k v with
      | none => none
      | some (h, slots) => decodeVecs ok n h slots
    else some (h, slots, false)

/-- The body of `Matrix::decode` after the header checks. `some (h, some m)` is
`Ok` with matrix `m`, `some (h, none)` an `Err`, `none` a double free. -/
def run (ok : List Bool) (h : Heap) : Option (Heap × Option Nat) :=
  if !ok.getD 0 true then some (h, none) else
  let ((c, slots), h) := containerNew h
  match decodeVecs ok 5 h slots with
  | none => none
  | some (h, slots, false) =>
    -- `?` returned: `container` dropped, freeing the vectors installed so far
    (containerFree h c slots).map (·, none)
  | some (h, slots, true) =>
    if !ok.getD 6 true then (containerFree h c slots).map (·, none) else
    let (m, h) := alloc h
    if !ok.getD 7 true || !ok.getD 8 true then
      -- `GrB_Matrix_free(m)`, then `container` dropped
      (free h m >>= fun h => containerFree h c slots).map (·, none)
    else
      -- the load moved the contents into `m`; the emptied vectors stay in the container
      (containerFree h c slots).map (·, some m)

/-- What is live after `decode` returns: the matrix it returned, nothing else. -/
def Clean (h0 : Heap) (r : Option (Heap × Option Nat)) : Prop :=
  match r with
  | none => False
  | some (h, none) => h.live = h0.live
  | some (h, some m) => h.live = [m] ++ h0.live

instance (h0 : Heap) (r : Option (Heap × Option Nat)) : Decidable (Clean h0 r) := by
  unfold Clean; split <;> infer_instance

/-- **#3080: no leak, no double free.** From an empty heap, whichever of the nine
calls fails (and whatever fails after it), `decode` frees every block it
allocated exactly once, except the matrix it returns. -/
theorem decode_no_leak_no_double_free (b0 b1 b2 b3 b4 b5 b6 b7 b8 : Bool) :
    Clean ⟨0, []⟩ (run [b0, b1, b2, b3, b4, b5, b6, b7, b8] ⟨0, []⟩) := by
  revert b0 b1 b2 b3 b4 b5 b6 b7 b8
  decide

/-- The success path returns a matrix; any failure returns `Err`. -/
theorem decode_ok_iff_all_ok (b0 b1 b2 b3 b4 b5 b6 b7 b8 : Bool) :
    ((run [b0, b1, b2, b3, b4, b5, b6, b7, b8] ⟨0, []⟩).bind (·.2)).isSome
      = (b0 && b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8) := by
  revert b0 b1 b2 b3 b4 b5 b6 b7 b8
  decide

/-! ## Status codes of the calls `decode` makes itself -/

/-- `GxB_Container_new` (`OwnedContainer::new`), `GrB_Matrix_new`,
`GxB_load_Matrix_from_Container`, `GrB_Matrix_wait`: each `if info != SUCCESS
{ return Err }`. -/
def decodeCalls : List Wrapper :=
  [⟨"OwnedContainer::new / GxB_Container_new", 452, .propagate⟩,
   ⟨"Matrix::decode / GrB_Matrix_new", 529, .propagate⟩,
   ⟨"Matrix::decode / GxB_load_Matrix_from_Container", 529, .propagate⟩,
   ⟨"Matrix::decode / GrB_Matrix_wait", 529, .propagate⟩]

theorem decodeCalls_never_die (i : Info) : ∀ w ∈ decodeCalls, Disposition.run w.disp i = some i.ok := by
  intro w hw
  simp only [decodeCalls, List.mem_cons, List.mem_nil_iff, or_false] at hw
  rcases hw with h | h | h | h <;> subst h <;> cases i <;> rfl

/-- A payload GraphBLAS refuses to load is an error, not a matrix. -/
theorem decode_load_refused_is_err :
    ∀ w ∈ decodeCalls, Disposition.run w.disp .invalidObject = some false := by
  intro w hw; rw [decodeCalls_never_die .invalidObject w hw]; rfl

end GBWMatDecode
