/-
# `Matrix::decode` (matrix.rs:708): who frees the container's vectors (#3643)

`GxB_Container_new` (GraphBLAS v10.5.0 `GxB_Container_new_arena.c`) allocates the
container struct **and** five empty component vectors `p, h, b, i, x`
(`GB_container_component_new` → `GB_new` with a fresh header). `GxB_Container_free`
frees the struct and whatever vectors its five fields (and `Y`) hold at that moment.
`GxB_load_Matrix_from_Container` moves the *arrays* of the vectors into the matrix and
leaves their headers in the container (`GB_vector_unload` / `GB_vector_reset`), on
success and on failure alike. So the blocks to track are vector headers.

## pre-#3643 (base matrix.rs:448)

```rust
let container = GxB_Container_new(..);              // container + 5 originals
copy_nonoverlapping(payload, container, SIZE);       // fields := payload's
(*container).x = null_mut(); .. ;                   // originals dropped on the floor
(*container).x = ManuallyDrop::new(Vector::<bool>::decode(r)?).ptr();  // `?`: no free
...
GxB_Container_free(&mut container);                  // only on success
```

## #3643 (matrix.rs:708)

```rust
let header = ContainerHeader::parse(&container_bytes)?;   // before any allocation
let guard = ContainerGuard(GxB_Container_new(..));        // Drop (:453) = GxB_Container_free
let x = Component::read(r)?; .. ; check_structure(..)?;  // no GraphBLAS allocation
(*container).x = ManuallyDrop::new(x.load()?).ptr();      // overwrites the original x
...
GrB_Matrix_new; GxB_load_Matrix_from_Container .. else { GrB_Matrix_free; Err }
```

The guard makes every early `return Err` free the container and what it holds
(`decode3643_no_leak_before_load`, against `pre3643_leaks_on_error_path`), but the
assignments `(*container).x = …` still overwrite the five vectors `GxB_Container_new`
put there without freeing them, so a successful decode still leaks five vector headers
(`bug_matrix_decode_still_leaks_container_components`, as `pre3643_leaks_container_components`
did), and a failed `load` leaks the originals it already overwrote
(`decode3643_load_error_leaks`). `fixed_no_leak`: freeing each field before the
assignment (`GrB_Vector_free(&mut (*container).x)`) would leak nothing on every exit.
No exit frees a block twice (`decode3643_no_double_free`).

The GraphBLAS calls are modelled by what they allocate and free (cited above, C source,
not a Lean axiom). Rust-owned buffers (`Component`'s `Vec`s) are freed by Rust and not
tracked.
-/

namespace GBW
namespace MatrixDecode

/-- A heap block one decode can allocate. Component index `k`: 0..4 = `x, h, p, i, b`
(the order decode fills them; GraphBLAS allocates them in another order, which does
not matter here). -/
inductive Blk where
  | container
  | orig (k : Nat)
  | loaded (k : Nat)
  | matrix
deriving DecidableEq, Repr

/-- Live blocks, the five container fields, and whether a free hit a dead block. -/
structure Heap where
  live : List Blk
  slots : List Blk
  doubleFree : Bool
deriving DecidableEq, Repr

def Heap.empty : Heap := ⟨[], [], false⟩

def Heap.free (h : Heap) (b : Blk) : Heap :=
  { h with live := h.live.filter (· != b), doubleFree := h.doubleFree || !(h.live.contains b) }

/-- One GraphBLAS-level step of a decode. -/
inductive Op where
  /-- `GxB_Container_new`: the struct and five fresh components in its fields. -/
  | containerNew
  /-- `(*container).k = load(..)?.ptr()` after a successful `load`: a new vector,
  stored in field `k` over whatever was there. -/
  | load (k : Nat)
  /-- The fix: `GrB_Vector_free(&mut (*container).k)` before the assignment. -/
  | freeSlot (k : Nat)
  | matrixNew
  | matrixFree
  /-- `GxB_Container_free` (the guard's `Drop`, matrix.rs:453): the struct and the
  vectors its fields hold. -/
  | containerFree

def five : List Nat := [0, 1, 2, 3, 4]

def step (h : Heap) : Op → Heap
  | .containerNew =>
    { h with live := h.live ++ .container :: five.map .orig, slots := five.map .orig }
  | .load k => { h with live := h.live ++ [.loaded k], slots := h.slots.set k (.loaded k) }
  | .freeSlot k => match h.slots[k]? with
    | some b => h.free b
    | none => h
  | .matrixNew => { h with live := h.live ++ [.matrix] }
  | .matrixFree => h.free .matrix
  | .containerFree => (h.slots.foldl Heap.free (h.free .container))

def run (ops : List Op) : Heap := ops.foldl step Heap.empty

/-- **`ContainerGuard::drop` (matrix.rs:453).** It frees the container and every vector
its fields hold, and nothing else. -/
theorem containerFree_frees_held (h : Heap) (b : Blk) :
    b ∈ (step h .containerFree).live ↔ b ∈ h.live ∧ b ≠ .container ∧ b ∉ h.slots := by
  simp only [step]
  suffices H : ∀ (l : List Blk) (g : Heap), b ∈ (l.foldl Heap.free g).live ↔ b ∈ g.live ∧ b ∉ l by
    rw [H]; simp [Heap.free, and_assoc]
  intro l
  induction l with
  | nil => intro g; simp
  | cons a t ih =>
    intro g
    simp only [List.foldl_cons, ih, Heap.free, List.mem_filter, bne_iff_ne, ne_eq,
      List.mem_cons, not_or]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3⟩; exact ⟨h1, h2, h3⟩
    · rintro ⟨h1, h2, h3⟩; exact ⟨⟨h1, h2⟩, h3⟩

/-! ## The exits of `Matrix::decode` -/

/-- Where a decode stops. -/
inductive Exit where
  /-- `read_buffer`, the size check, `ContainerHeader::parse` (#3643) — before any
  GraphBLAS allocation. -/
  | header
  /-- #3643: `Component::read` ×5 or `check_structure`. pre-#3643: — (no such step). -/
  | components
  /-- The `k`-th component load (`k < 5`) fails after `k` succeeded (pre-#3643:
  `Vector::<bool>::decode(r)?`). -/
  | load (k : Nat)
  /-- #3643: `GxB_load_Matrix_from_Container`, the row-major conversion or the
  element-size check fails (`GrB_Matrix_free` then `Err`). -/
  | loadMatrix
  | ok
deriving DecidableEq, Repr

def loads (k : Nat) : List Op := (List.range k).map Op.load

/-- #3643 (matrix.rs:708-835). Every exit after `GxB_Container_new` drops the guard. -/
def ops3643 : Exit → List Op
  | .header => []
  | .components => [.containerNew, .containerFree]
  | .load k => .containerNew :: loads k ++ [.containerFree]
  | .loadMatrix => .containerNew :: loads 5 ++ [.matrixNew, .matrixFree, .containerFree]
  | .ok => .containerNew :: loads 5 ++ [.matrixNew, .containerFree]

/-- pre-#3643 (base matrix.rs:448-523). `components` and `loadMatrix` cannot happen
there (no structure check; the load's code was only `debug_assert`ed): they behave as
`load 0` and `ok`. -/
def opsPre : Exit → List Op
  | .header => []
  | .components => [.containerNew]
  | .load k => .containerNew :: loads k
  | .loadMatrix => .containerNew :: loads 5 ++ [.matrixNew, .containerFree]
  | .ok => .containerNew :: loads 5 ++ [.matrixNew, .containerFree]

/-- The fix: free each field before it is overwritten. -/
def fixedLoads (k : Nat) : List Op := (List.range k).flatMap fun j => [Op.freeSlot j, Op.load j]

def opsFixed : Exit → List Op
  | .header => []
  | .components => [.containerNew, .containerFree]
  | .load k => .containerNew :: fixedLoads k ++ [.containerFree]
  | .loadMatrix => .containerNew :: fixedLoads 5 ++ [.matrixNew, .matrixFree, .containerFree]
  | .ok => .containerNew :: fixedLoads 5 ++ [.matrixNew, .containerFree]

/-- Blocks still allocated and owned by no one: the returned matrix is the caller's. -/
def leaked (ops : Exit → List Op) (e : Exit) : List Blk :=
  let h := run (ops e)
  if e = .ok then h.live.filter (· != .matrix) else h.live

/-! ## pre-#3643: the confirmed leak (history) -/

/-- pre-#3643: a successful decode leaked the five components `GxB_Container_new`
made (test then: `bug_matrix_decode_leaks_container_components`, 5.00 blocks/call). -/
theorem pre3643_leaks_container_components :
    leaked opsPre .ok = [.orig 0, .orig 1, .orig 2, .orig 3, .orig 4] := by decide

/-- pre-#3643: a failing component decode leaked the container, its five components
and the components already decoded (`bug_matrix_decode_leaks_on_error_path`: 8.00/call
at the third component). -/
theorem pre3643_leaks_on_error_path :
    (leaked opsPre (.load 0)).length = 6 ∧ (leaked opsPre (.load 2)).length = 8 ∧
    (leaked opsPre (.load 4)).length = 10 := by decide

/-! ## #3643 -/

/-- **Fixed by #3643's guard**: a payload refused before any load — bad header,
dimensions, format, orientation, component framing, or structure — leaks nothing. -/
theorem decode3643_no_leak_before_load :
    leaked ops3643 .header = [] ∧ leaked ops3643 .components = [] := by decide

/-- **Still a leak (#3643):** a successful decode leaks the five vectors
`GxB_Container_new` put in the container, as before: the assignments
`(*container).x = …` (matrix.rs:763-767) overwrite them unfreed. -/
theorem bug_matrix_decode_still_leaks_container_components :
    leaked ops3643 .ok = [.orig 0, .orig 1, .orig 2, .orig 3, .orig 4] ∧
    leaked ops3643 .loadMatrix = [.orig 0, .orig 1, .orig 2, .orig 3, .orig 4] := by decide

/-- A failing `load` (#3643) leaks exactly the originals already overwritten; the
guard frees the container, the untouched originals and the loaded vectors. -/
theorem decode3643_load_error_leaks :
    leaked ops3643 (.load 0) = [] ∧ leaked ops3643 (.load 1) = [.orig 0] ∧
    leaked ops3643 (.load 2) = [.orig 0, .orig 1] ∧
    leaked ops3643 (.load 3) = [.orig 0, .orig 1, .orig 2] ∧
    leaked ops3643 (.load 4) = [.orig 0, .orig 1, .orig 2, .orig 3] := by decide

/-- The exits a decode can take (`load k` for `k < 5`). -/
def exits : List Exit := [.header, .components, .load 0, .load 1, .load 2, .load 3, .load 4,
  .loadMatrix, .ok]

/-- No exit of #3643 frees a block twice (the guard runs once; the matrix freed on
the `loadMatrix` path is not in the container). -/
theorem decode3643_no_double_free : ∀ e ∈ exits, (run (ops3643 e)).doubleFree = false := by
  decide

/-- Freeing each field before overwriting it would close the leak on every exit,
without a double free. -/
theorem fixed_no_leak :
    ∀ e ∈ exits, leaked opsFixed e = [] ∧ (run (opsFixed e)).doubleFree = false := by
  decide

/-- pre-#3643 vs #3643, block counts per exit: the guard strictly reduces the leak on
every error path and leaves the success path unchanged. -/
theorem decode3643_vs_pre :
    ∀ e ∈ exits, (leaked ops3643 e).length ≤ (leaked opsPre e).length := by decide

/-! ## `ContainerHeader::parse` (matrix.rs:478) and `index_width` (:523) -/

/-- `GxB_HYPERSPARSE`, `GxB_SPARSE` (GraphBLAS.h), `GrB_ROWMAJOR`, `GrB_COLMAJOR`. -/
def HYPERSPARSE : Int := 1
def SPARSE : Int := 2
def ROWMAJOR : Int := 0
def COLMAJOR : Int := 1

/-- `MAX_DIM` (matrix.rs:460) = `GrB_INDEX_MAX + 1`. -/
def MAX_DIM : Nat := 2 ^ 60

/-- `DecodeLimits` (serialization.rs:204) and `max_dim` (:239), `u64` saturating. -/
structure Limits where
  node_bound : Nat
  edge_bound : Nat

def satU64 (n : Nat) : Nat := min n (2 ^ 64 - 1)

def Limits.maxDim (l : Limits) : Nat :=
  satU64 (satU64 (max l.node_bound l.edge_bound * 2) + 2 ^ 31)

/-- The dimension limit `parse` applies: `MAX_DIM`, or `max_dim()` capped by it. -/
def dimLimit : Option Limits → Nat
  | none => MAX_DIM
  | some l => min l.maxDim MAX_DIM

/-- The checks of `ContainerHeader::parse` on the fields it read (the field offsets
are `offset_of!` on the bindgen struct; the byte reads are not modelled). -/
def parseCheck (lim : Option Limits) (nrows ncols : Nat) (format orientation : Int) :
    Except String Unit :=
  if format ≠ SPARSE ∧ format ≠ HYPERSPARSE then .error "matrix format is not sparse"
  else if orientation ≠ ROWMAJOR ∧ orientation ≠ COLMAJOR then
    .error "matrix orientation is unknown"
  else if nrows > dimLimit lim ∨ ncols > dimLimit lim then
    .error "matrix dimensions exceed the limit"
  else .ok ()

theorem parseCheck_ok_iff (lim : Option Limits) (nrows ncols : Nat) (format orientation : Int) :
    parseCheck lim nrows ncols format orientation = .ok () ↔
      (format = SPARSE ∨ format = HYPERSPARSE) ∧
      (orientation = ROWMAJOR ∨ orientation = COLMAJOR) ∧
      nrows ≤ dimLimit lim ∧ ncols ≤ dimLimit lim := by
  unfold parseCheck
  split
  · simp only [reduceCtorEq, false_iff]; omega
  · split
    · simp only [reduceCtorEq, false_iff]; omega
    · split
      · simp only [reduceCtorEq, false_iff]; omega
      · simp only [true_iff]; omega

/-- No limits installed: any sparse/hypersparse header of real dimensions passes. -/
theorem parseCheck_none_accepts (nrows ncols : Nat) (hr : nrows ≤ MAX_DIM) (hc : ncols ≤ MAX_DIM) :
    parseCheck none nrows ncols SPARSE ROWMAJOR = .ok () := by
  rw [parseCheck_ok_iff]
  refine ⟨Or.inl rfl, Or.inl rfl, ?_, ?_⟩ <;> simp only [dimLimit] <;> omega

/-- A matrix saved at a capacity up to twice the largest id bound plus the slop
(`DIM_SLOP`, a creation chunk's worth) passes the limit. -/
theorem parseCheck_accepts_capacity (l : Limits) (nrows ncols : Nat)
    (hr : nrows ≤ max l.node_bound l.edge_bound * 2 + 2 ^ 31)
    (hc : ncols ≤ max l.node_bound l.edge_bound * 2 + 2 ^ 31)
    (hb : max l.node_bound l.edge_bound * 2 + 2 ^ 31 ≤ MAX_DIM) :
    parseCheck (some l) nrows ncols SPARSE ROWMAJOR = .ok () := by
  rw [parseCheck_ok_iff]
  refine ⟨Or.inl rfl, Or.inl rfl, ?_, ?_⟩ <;>
    simp only [dimLimit, Limits.maxDim, satU64, MAX_DIM] at * <;> omega

/-- `index_width` (matrix.rs:523): bytes per entry of an index array, by type name. -/
def indexWidth (ty : String) : Except String Nat :=
  if ty = "GrB_UINT32" ∨ ty = "GrB_INT32" then .ok 4
  else if ty = "GrB_UINT64" ∨ ty = "GrB_INT64" then .ok 8
  else .error s!"matrix index array has type {ty}"

theorem indexWidth_ok_iff (ty : String) (w : Nat) :
    indexWidth ty = .ok w ↔
      ((ty = "GrB_UINT32" ∨ ty = "GrB_INT32") ∧ w = 4) ∨
      ((ty = "GrB_UINT64" ∨ ty = "GrB_INT64") ∧ w = 8) := by
  unfold indexWidth
  by_cases h4 : ty = "GrB_UINT32" ∨ ty = "GrB_INT32"
  · have h8 : ¬ (ty = "GrB_UINT64" ∨ ty = "GrB_INT64") := by
      rcases h4 with h | h <;> subst h <;> decide
    simp only [h4, ite_true, Except.ok.injEq, true_and, h8, false_and, or_false]
    exact ⟨fun h => h.symm, fun h => h.symm⟩
  · by_cases h8 : ty = "GrB_UINT64" ∨ ty = "GrB_INT64"
    · simp only [h4, h8, ite_true, ite_false, Except.ok.injEq, true_and, false_and, false_or]
      exact ⟨fun h => h.symm, fun h => h.symm⟩
    · simp only [h4, h8, ite_false, false_and, or_self, iff_false]
      intro h; cases h

/-- Every index type `GxB_load_Matrix_from_Container` accepts (`GB_type_ok`:
`INT32`, `UINT32`, `INT64`, `UINT64`) has a width, so the structure check never
refuses a type the load would take. -/
theorem indexWidth_covers_load_types :
    ∀ ty ∈ ["GrB_INT32", "GrB_UINT32", "GrB_INT64", "GrB_UINT64"],
      ∃ w, indexWidth ty = .ok w := by
  intro ty hty
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hty
  rcases hty with h | h | h | h <;> subst h
  · exact ⟨4, by unfold indexWidth; simp⟩
  · exact ⟨4, by unfold indexWidth; simp⟩
  · exact ⟨8, by unfold indexWidth; simp⟩
  · exact ⟨8, by unfold indexWidth; simp⟩

end MatrixDecode
end GBW
