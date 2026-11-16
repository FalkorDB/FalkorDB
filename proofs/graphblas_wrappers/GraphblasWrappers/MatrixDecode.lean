/-
# `Matrix::decode` after #3080: no leak, no panic, header parsed field by field

| here | there |
| --- | --- |
| `flag`, `parse`, `Hdr` | `ContainerHeader::parse` matrix.rs:498 (`field` :502, `flag` closure :508-512) |
| `knownFormat` | format check matrix.rs:547-558 (`GxB_HYPERSPARSE=1, SPARSE=2, BITMAP=4, FULL=8`) |
| `accepts` | `Decode<19> for Matrix` matrix.rs:529-558 (length check, parse, format) |
| `Res`, `St`, `free`, `containerFree` | `OwnedContainer` (`new` :452, `replace` :463, `Drop` :476 = `GxB_Container_free`) |
| `run` | matrix.rs:559-616 (scalar copy, 5 × `replace(decode?)`, `GrB_Matrix_new`, load, wait) |
| `runOld` | the pre-#3080 body (the whole-struct `copy_nonoverlapping`, pointers nulled, `?` without a free) |

Resources are tracked as objects: the container `cont`, the five empty
vectors `GxB_Container_new` puts in its slots (`slot k`), each decoded
component's vector object `vec k` and its array `arr k`, and the matrix `mat`.
C contracts used (as model steps, not axioms): `GxB_Container_free` frees the
container, the vector in each slot and the array that vector still holds;
`GxB_load_Matrix_from_Container` on success moves the slot arrays into the
matrix and leaves the (now empty) vector objects in the container, on failure
leaves them in the container; `GrB_Matrix_free` frees the matrix and what it
owns. A failed `Vector::<Untyped>::decode` frees what it allocated before
returning `Err` (vector.rs:431, :442, :470), so it leaves nothing live.
-/

import GraphblasWrappers.GrbInfo

namespace GBWMatDecode

/-! ## Header -/

/-- The `flag` closure (matrix.rs:508-512): a byte that is not 0 or 1 is refused. -/
def flag (b : Nat) : Except String Bool :=
  if b = 0 then .ok false else if b = 1 then .ok true else .error "not 0 or 1"

/-- `field` (matrix.rs:502): `bytes[offset..offset + N].try_into().expect(..)`.
Every field lies inside the struct (`offset_of!(C, f) + size_of f ≤ size_of C`, a
Rust layout fact) and `decode` checked `bytes.len() ≥ CONTAINER_STRUCT_SIZE`
(matrix.rs:533), so the slice is in bounds and has exactly `N` bytes: neither the
index nor the `expect` can panic. -/
theorem field_in_bounds (off n size len : Nat) (hf : off + n ≤ size) (hl : size ≤ len) :
    off + n ≤ len ∧ (off + n) - off = n := ⟨Nat.le_trans hf hl, by omega⟩

/-- The raw bytes of the scalar fields, at their `offset_of!` positions. -/
structure Raw where
  nrows : Nat
  ncols : Nat
  nrowsNonempty : Int
  ncolsNonempty : Int
  nvals : Nat
  format : Int
  orientation : Int
  iso : Nat
  jumbled : Nat

structure Hdr where
  nrows : Nat
  ncols : Nat
  nrowsNonempty : Int
  ncolsNonempty : Int
  nvals : Nat
  format : Int
  orientation : Int
  iso : Bool
  jumbled : Bool

/-- `ContainerHeader::parse` (matrix.rs:498). -/
def parse (r : Raw) : Except String Hdr := do
  let iso ← flag r.iso
  let jumbled ← flag r.jumbled
  return ⟨r.nrows, r.ncols, r.nrowsNonempty, r.ncolsNonempty, r.nvals, r.format,
    r.orientation, iso, jumbled⟩

def Except.isOk {ε α : Type} : Except ε α → Bool
  | .ok _ => true
  | .error _ => false

theorem flag_ok_iff (b : Nat) (v : Bool) :
    flag b = .ok v ↔ (b = 0 ∧ v = false) ∨ (b = 1 ∧ v = true) := by
  unfold flag
  by_cases h0 : b = 0
  · subst h0; cases v <;> simp
  · by_cases h1 : b = 1
    · subst h1; cases v <;> simp
    · simp [h0, h1]

/-- The header parses iff both flag bytes are a valid Rust `bool` (0 or 1); every
scalar field is then copied unchanged and the flags are the bytes' meaning. -/
theorem parse_ok_iff (r : Raw) :
    Except.isOk (parse r) = true ↔ r.iso ≤ 1 ∧ r.jumbled ≤ 1 := by
  unfold parse flag
  by_cases a0 : r.iso = 0 <;> by_cases a1 : r.iso = 1 <;>
    by_cases b0 : r.jumbled = 0 <;> by_cases b1 : r.jumbled = 1 <;>
    simp [a0, a1, b0, b1, Except.isOk, bind, Except.bind, pure, Except.pure] <;> omega

/-- A flag byte that is not 0 or 1 is refused (the pre-#3080 whole-struct read
built an invalid Rust `bool` from it). -/
theorem parse_refuses_non_bool_flag (r : Raw) (h : 1 < r.iso ∨ 1 < r.jumbled) :
    Except.isOk (parse r) = false := by
  cases hp : Except.isOk (parse r)
  · rfl
  · have := (parse_ok_iff r).1 hp; omega

theorem parse_fields (r : Raw) (h : Hdr) (hp : parse r = .ok h) :
    h.nrows = r.nrows ∧ h.ncols = r.ncols ∧ h.nvals = r.nvals ∧ h.format = r.format ∧
    h.orientation = r.orientation ∧ h.nrowsNonempty = r.nrowsNonempty ∧
    h.ncolsNonempty = r.ncolsNonempty ∧
    (h.iso = true ↔ r.iso = 1) ∧ (h.jumbled = true ↔ r.jumbled = 1) := by
  unfold parse flag at hp
  by_cases a0 : r.iso = 0 <;> by_cases a1 : r.iso = 1 <;>
    by_cases b0 : r.jumbled = 0 <;> by_cases b1 : r.jumbled = 1 <;>
    simp [a0, a1, b0, b1, bind, Except.bind, pure, Except.pure] at hp <;>
    (try subst hp) <;> simp <;> omega

/-- matrix.rs:547-558: the four GraphBLAS sparsity formats. -/
def knownFormat (f : Int) : Bool := f == 1 || f == 2 || f == 4 || f == 8

theorem knownFormat_iff (f : Int) : knownFormat f = true ↔ f = 1 ∨ f = 2 ∨ f = 4 ∨ f = 8 := by
  simp [knownFormat, or_assoc]

/-- matrix.rs:531-558: the payload is decoded only past the length check, the
parse and the format check; every refusal is an `Err` before any allocation. -/
def accepts (lenOk : Bool) (r : Raw) : Bool :=
  lenOk && Except.isOk (parse r) && knownFormat r.format

/-- The round trip is not refused by the new checks: `encode` writes the
header of a container GraphBLAS unloaded, whose flags are Rust `bool`s (0 or 1)
and whose format is one of the four. -/
theorem accepts_encoded (r : Raw) (hi : r.iso ≤ 1) (hj : r.jumbled ≤ 1)
    (hf : r.format = 1 ∨ r.format = 2 ∨ r.format = 4 ∨ r.format = 8) :
    accepts true r = true := by
  have h1 := (parse_ok_iff r).2 ⟨hi, hj⟩
  have h2 := (knownFormat_iff r.format).2 hf
  simp [accepts, h1, h2]

/-! ## Ownership -/

inductive Res | cont | slot (k : Nat) | vec (k : Nat) | arr (k : Nat) | mat
deriving DecidableEq, Repr

structure St where
  live : List Res := []
  /-- vector objects in the container's `x, h, p, i, b` slots -/
  cslots : List Res := []
  /-- arrays those vectors hold -/
  carrs : List Res := []
  /-- the matrix handle and what it owns -/
  mown : List Res := []
  /-- a free of something not live: a double free -/
  dbl : Bool := false
deriving DecidableEq, Repr

def free (s : St) (r : Res) : St :=
  if r ∈ s.live then { s with live := s.live.erase r } else { s with dbl := true }

def freeAll (s : St) (rs : List Res) : St := rs.foldl free s

/-- `OwnedContainer::drop` (matrix.rs:476) = `GxB_Container_free`. -/
def containerFree (s : St) : St :=
  let s := freeAll s (Res.cont :: s.cslots ++ s.carrs)
  { s with cslots := [], carrs := [] }

/-- `GrB_Matrix_free` on an error path (matrix.rs:597, :608). -/
def matrixFree (s : St) : St := { freeAll s s.mown with mown := [] }

/-- The outcome of each fallible call, as `info == GrB_SUCCESS`. -/
structure Calls where
  contNew : Bool
  /-- the five component decodes `x, h, p, i, b` -/
  c0 : Bool
  c1 : Bool
  c2 : Bool
  c3 : Bool
  c4 : Bool
  matNew : Bool
  load : Bool
  wait : Bool

/-- `OwnedContainer::replace` (matrix.rs:463): free the slot's vector, put the decoded one in. -/
def replace (s : St) (k : Nat) : St :=
  let s := free s (.slot k)
  { s with cslots := (s.cslots.erase (.slot k)) ++ [.vec k], carrs := s.carrs ++ [.arr k] }

/-- Decode component `k` (`Vector::<Untyped>::decode(r)?`) then `replace`.
`.error s` = the `?` returned `Err` with the state `s` at that point. -/
def stepComp (ok : Bool) (k : Nat) (s : St) : Except St St :=
  if ok then .ok (replace { s with live := s.live ++ [.vec k, .arr k] } k) else .error s

/-- matrix.rs:578-582: the five components `x, h, p, i, b` in order. -/
def comps (c : Calls) (s0 : St) : Except St St := do
  let s ← stepComp c.c0 0 s0
  let s ← stepComp c.c1 1 s
  let s ← stepComp c.c2 2 s
  let s ← stepComp c.c3 3 s
  stepComp c.c4 4 s

/-- `OwnedContainer::new` (matrix.rs:452) on success. -/
def fresh : St := { live := [.cont, .slot 0, .slot 1, .slot 2, .slot 3, .slot 4],
                    cslots := [.slot 0, .slot 1, .slot 2, .slot 3, .slot 4] }

/-- matrix.rs:559-616 after the checks. Result: `(Ok?, final state)`;
every `return Err` runs `OwnedContainer::drop`. -/
def run (c : Calls) : Bool × St :=
  if !c.contNew then (false, {}) else
  match comps c fresh with
  | .error s => (false, containerFree s)
  | .ok s5 =>
    if !c.matNew then (false, containerFree s5) else
    let s := { s5 with live := s5.live ++ [.mat], mown := [.mat] }
    if !c.load then (false, containerFree (matrixFree s)) else
    let s := { s with mown := s.mown ++ s.carrs, carrs := [] }
    if !c.wait then (false, containerFree (matrixFree s)) else
    (true, containerFree s)

/-- **#3080: no leak, no double free.** On every path — every call failing or
succeeding — nothing is freed twice; an `Err` leaves nothing allocated, and
`Ok` leaves exactly the returned matrix and the five component arrays it now
owns. -/
theorem run_no_leak (c : Calls) :
    (run c).2.dbl = false ∧
    ((run c).1 = false → (run c).2.live = []) ∧
    ((run c).1 = true →
      (∀ r ∈ (run c).2.live, r ∈ (run c).2.mown) ∧ ∀ r ∈ (run c).2.mown, r ∈ (run c).2.live) ∧
    ((run c).1 = true → (run c).2.mown = [.mat, .arr 0, .arr 1, .arr 2, .arr 3, .arr 4]) := by
  obtain ⟨a, c0, c1, c2, c3, c4, b, d, e⟩ := c
  cases a <;> cases b <;> cases d <;> cases e <;>
    cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> decide

/-- `Ok` iff every call succeeded: a refusal from GraphBLAS (load or the final
wait, both payload-reachable) is an `Err`, not a `debug_assert`. -/
theorem run_ok_iff (c : Calls) :
    (run c).1 = true ↔
      c.contNew ∧ c.c0 ∧ c.c1 ∧ c.c2 ∧ c.c3 ∧ c.c4 ∧ c.matNew ∧ c.load ∧ c.wait := by
  obtain ⟨a, c0, c1, c2, c3, c4, b, d, e⟩ := c
  cases a <;> cases b <;> cases d <;> cases e <;>
    cases c0 <;> cases c1 <;> cases c2 <;> cases c3 <;> cases c4 <;> decide

/-- No call in the new body is an `assert_eq!`: `GxB_Container_new`,
`GrB_Matrix_new`, the load and the wait all propagate their code (the frees in
`replace`/`drop` are `debug_assert`s on calls that only fail on a null handle). -/
def newDispositions : List GBW.Disposition := [.propagate, .propagate, .propagate, .propagate]

theorem decode_never_dies (i : GBW.Info) :
    ∀ d ∈ newDispositions, GBW.Disposition.run d i = some i.ok := by
  intro d hd; simp [newDispositions] at hd; subst hd; cases i <;> rfl

/-! ## Before #3080 (history): bug 3 -/

/-- The pre-#3080 body: `copy_nonoverlapping` of the whole payload header
over the fresh container, then its pointers nulled — the five `slot k`
vectors are no longer referenced by anything; a `?` returns without freeing the
container; on success `GxB_Container_free` frees what the slots then hold. -/
def stepCompOld (ok : Bool) (k : Nat) (s : St) : Except St St :=
  if ok then .ok { s with live := s.live ++ [.vec k, .arr k], cslots := s.cslots ++ [.vec k],
                          carrs := s.carrs ++ [.arr k] }
  else .error s

def compsOld (c : Calls) (s0 : St) : Except St St := do
  let s ← stepCompOld c.c0 0 s0
  let s ← stepCompOld c.c1 1 s
  let s ← stepCompOld c.c2 2 s
  let s ← stepCompOld c.c3 3 s
  stepCompOld c.c4 4 s

def runOld (c : Calls) : Bool × St :=
  let s0 : St := { live := [.cont, .slot 0, .slot 1, .slot 2, .slot 3, .slot 4] }
  match compsOld c s0 with
  | .error s => (false, s)
  | .ok s5 =>
    let s := { s5 with live := s5.live ++ [Res.mat], mown := [Res.mat] }
    let s := { s with mown := s.mown ++ s.carrs, carrs := [] }
    (true, containerFree s)

/-- Pre-#3080, a successful decode left the five empty vectors
`GxB_Container_new` allocated live and unowned (Rust test
`bug_matrix_decode_leaks_container_components`, 5.00 blocks/call). -/
theorem pre3080_success_leaks_five_slots :
    let c : Calls := ⟨true, true, true, true, true, true, true, true, true⟩
    (runOld c).1 = true ∧ ∀ k ∈ [0, 1, 2, 3, 4], Res.slot k ∈ (runOld c).2.live ∧
      Res.slot k ∉ (runOld c).2.mown := by
  decide

/-- Pre-#3080, a component that fails to decode left the container live. -/
theorem pre3080_error_leaks_container :
    Res.cont ∈ (runOld ⟨true, false, true, true, true, true, true, true, true⟩).2.live := by decide

/-- The same inputs on the new body leak nothing. -/
theorem post3080_same_inputs_no_leak :
    (∀ r ∈ (run ⟨true, true, true, true, true, true, true, true, true⟩).2.live,
      r ∈ [Res.mat, .arr 0, .arr 1, .arr 2, .arr 3, .arr 4]) ∧
    (run ⟨true, false, true, true, true, true, true, true, true⟩).2.live = [] := by decide

end GBWMatDecode
