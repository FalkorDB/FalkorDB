/-
# Glue in `matrix.rs` / `vector.rs`: operator and type selection, descriptor
# table, accessors, `encode` round trip, `Vector::Iter::new`

| here | there |
| --- | --- |
| `addOp`/`applyOp` | `EWiseAdd::add_op` (matrix.rs:267, impls :271 bool, :279 u64) |
| `newMatrix`       | `MatrixType::new_matrix` (:290, impls :297, :306) |
| `Desc`/`descOf`   | `enum Descriptor` (:224) / `From<Descriptor> for GrB_Descriptor` (:315) |
| `encodeM`         | `Encode<19> for Matrix` (:510) |
| `MatH.inner`/`isSynced` | `inner` (:721) / `is_synced` (:803) |
| `sparsityName`    | `sparsity_status` (:1073) |
| `posB`/`posU`     | `IterExtract::pos` (:1532, impls :1549 bool, :1569 u64) |
| `vecFrom`/`ptr`   | `From<GrB_Vector>` (vector.rs:78, :87) / `ptr` (:140) |
| `vIterNew`        | `vector::Iter::new` (vector.rs:543) |

The GraphBLAS constants are named by their C meaning (`GrB_DESC_RC` =
replace + complemented mask, …); that naming is the C API spec, here a
definition (`descFlags`).
-/
namespace GBWGlue

/-! ## `add_op` -/

inductive ElemTy | bool | u64 deriving DecidableEq
inductive BinOp | anyBool | secondU64 deriving DecidableEq

def addOp : ElemTy → BinOp
  | .bool => .anyBool
  | .u64 => .secondU64

/-- `GxB_ANY_BOOL(a,b)` may return either; `GrB_SECOND_UINT64(a,b) = b`. -/
def applyOp (op : BinOp) (a b : Nat) : Nat → Prop
  | r => match op with
    | .anyBool => r = a ∨ r = b
    | .secondU64 => r = b

/-- On a `u64` layer the delta (second operand) wins: the shadowing rule `flush` needs. -/
theorem addOp_u64_second (a b : Nat) : applyOp (addOp .u64) a b b ∧ ∀ r, applyOp (addOp .u64) a b r → r = b :=
  ⟨rfl, fun _ h => h⟩
/-- On a `bool` layer any of the two `true`s is fine: the result is `true`. -/
theorem addOp_bool_true (r : Nat) (h : applyOp (addOp .bool) 1 1 r) : r = 1 := by
  rcases h with h | h <;> exact h

/-! ## `new_matrix` -/

structure Mat where
  ty : ElemTy
  ents : List (Nat × Nat)
  nrows : Nat
  ncols : Nat

def newMatrix (ty : ElemTy) (r c : Nat) : Mat := ⟨ty, [], r, c⟩
theorem newMatrix_spec (ty : ElemTy) (r c : Nat) :
    (newMatrix ty r c).ents = [] ∧ (newMatrix ty r c).nrows = r ∧ (newMatrix ty r c).ncols = c ∧
    (newMatrix ty r c).ty = ty := ⟨rfl, rfl, rfl, rfl⟩

/-! ## `Descriptor` → `GrB_Descriptor` -/

inductive Desc
  | T0 | T1 | T0T1 | C | CT0 | CT1 | CT0T1 | S | ST0 | ST1 | ST0T1 | SC | SCT0 | SCT1 | SCT0T1
  | R | RT0 | RT1 | RT0T1 | RC | RCT0 | RCT1 | RCT0T1 | RS | RST0 | RST1 | RST0T1 | RSC | RSCT0
  | RSCT1 | RSCT0T1
  deriving DecidableEq, Repr

/-- The predefined `GrB_DESC_*` object: (replace, structural, complement, T0, T1). -/
structure Flags where
  r : Bool
  s : Bool
  c : Bool
  t0 : Bool
  t1 : Bool
  deriving DecidableEq, Repr

/-- `From<Descriptor>` (:315): each variant to the `GrB_DESC_` constant of the same name. -/
def descOf : Desc → Flags
  | .T0 => ⟨false, false, false, true, false⟩ | .T1 => ⟨false, false, false, false, true⟩
  | .T0T1 => ⟨false, false, false, true, true⟩ | .C => ⟨false, false, true, false, false⟩
  | .CT0 => ⟨false, false, true, true, false⟩ | .CT1 => ⟨false, false, true, false, true⟩
  | .CT0T1 => ⟨false, false, true, true, true⟩ | .S => ⟨false, true, false, false, false⟩
  | .ST0 => ⟨false, true, false, true, false⟩ | .ST1 => ⟨false, true, false, false, true⟩
  | .ST0T1 => ⟨false, true, false, true, true⟩ | .SC => ⟨false, true, true, false, false⟩
  | .SCT0 => ⟨false, true, true, true, false⟩ | .SCT1 => ⟨false, true, true, false, true⟩
  | .SCT0T1 => ⟨false, true, true, true, true⟩ | .R => ⟨true, false, false, false, false⟩
  | .RT0 => ⟨true, false, false, true, false⟩ | .RT1 => ⟨true, false, false, false, true⟩
  | .RT0T1 => ⟨true, false, false, true, true⟩ | .RC => ⟨true, false, true, false, false⟩
  | .RCT0 => ⟨true, false, true, true, false⟩ | .RCT1 => ⟨true, false, true, false, true⟩
  | .RCT0T1 => ⟨true, false, true, true, true⟩ | .RS => ⟨true, true, false, false, false⟩
  | .RST0 => ⟨true, true, false, true, false⟩ | .RST1 => ⟨true, true, false, false, true⟩
  | .RST0T1 => ⟨true, true, false, true, true⟩ | .RSC => ⟨true, true, true, false, false⟩
  | .RSCT0 => ⟨true, true, true, true, false⟩ | .RSCT1 => ⟨true, true, true, false, true⟩
  | .RSCT0T1 => ⟨true, true, true, true, true⟩

def allDesc : List Desc :=
  [.T0, .T1, .T0T1, .C, .CT0, .CT1, .CT0T1, .S, .ST0, .ST1, .ST0T1, .SC, .SCT0, .SCT1, .SCT0T1,
   .R, .RT0, .RT1, .RT0T1, .RC, .RCT0, .RCT1, .RCT0T1, .RS, .RST0, .RST1, .RST0T1, .RSC, .RSCT0,
   .RSCT1, .RSCT0T1]

theorem allDesc_complete (d : Desc) : d ∈ allDesc := by cases d <;> decide

/-- The table is a bijection onto the 31 non-default flag sets. -/
theorem descOf_injective (a b : Desc) (h : descOf a = descOf b) : a = b := by
  cases a <;> cases b <;> first | rfl | (simp [descOf] at h)

theorem descOf_nondefault (d : Desc) : descOf d ≠ ⟨false, false, false, false, false⟩ := by
  cases d <;> decide

/-- The one the folds use: `RC` = replace + complemented mask (`<!dm, replace>`). -/
theorem descOf_RC : descOf .RC = ⟨true, false, true, false, false⟩ := rfl

/-! ## Handle accessors, `is_synced`, `sparsity_status` -/

structure MatH where
  handle : Nat
  hasPending : Bool

def MatH.inner (m : MatH) : Nat := m.handle
def MatH.isSynced (m : MatH) : Bool := !m.hasPending
theorem inner_eq (m : MatH) : m.inner = m.handle := rfl
theorem isSynced_iff (m : MatH) : m.isSynced = true ↔ m.hasPending = false := by
  simp [MatH.isSynced]

/-- `GxB_HYPERSPARSE = 1`, `GxB_SPARSE = 2`, `GxB_BITMAP = 4`, `GxB_FULL = 8` (GraphBLAS.h). -/
def sparsityName (code : Nat) : String :=
  if code = 1 then "hypersparse" else if code = 2 then "sparse"
  else if code = 4 then "bitmap" else if code = 8 then "full" else "unknown"

theorem sparsityName_known : sparsityName 1 = "hypersparse" ∧ sparsityName 2 = "sparse" ∧
    sparsityName 4 = "bitmap" ∧ sparsityName 8 = "full" := ⟨rfl, rfl, rfl, rfl⟩
theorem sparsityName_other (c : Nat) (h : c ≠ 1 ∧ c ≠ 2 ∧ c ≠ 4 ∧ c ≠ 8) : sparsityName c = "unknown" := by
  simp [sparsityName, h.1, h.2.1, h.2.2.1, h.2.2.2]

/-! ## `IterExtract::pos` -/

def posB (it : Nat × Nat) : Nat × Nat := it
def posU (it : Nat × Nat × Nat) : Nat × Nat := (it.1, it.2.1)
theorem posB_eq (r c : Nat) : posB (r, c) = (r, c) := rfl
/-- The value never takes part in merge order. -/
theorem posU_eq (r c v : Nat) : posU (r, c, v) = (r, c) := rfl
theorem posU_agrees (r c v : Nat) : posU (r, c, v) = posB (r, c) := rfl

/-! ## `Matrix::encode` (:510)

`encode` unloads the matrix into a fresh `GxB_Container`, writes the container
struct bytes then the five component vectors `x, h, p, i, b`, and loads the
container back. The C spec relied on (`GxB_unload`/`GxB_load` are inverse, and
leave/restore the matrix contents) is the hypothesis `UnloadLoad`. -/

structure Container (α : Type) where
  hdr : List Nat
  x : α
  h : α
  p : α
  i : α
  b : α

inductive Tok (α : Type) | bytes (l : List Nat) | vec (v : α)

def encodeM {M α : Type} (unload : M → Container α) (load : Container α → M) (m : M) :
    List (Tok α) × M :=
  let c := unload m
  ([.bytes c.hdr, .vec c.x, .vec c.h, .vec c.p, .vec c.i, .vec c.b], load c)

def UnloadLoad {M α : Type} (unload : M → Container α) (load : Container α → M) : Prop :=
  ∀ m, load (unload m) = m

theorem encodeM_spec {M α : Type} (unload : M → Container α) (load : Container α → M)
    (h : UnloadLoad unload load) (m : M) :
    (encodeM unload load m).2 = m ∧ (encodeM unload load m).1.length = 6 := ⟨h m, rfl⟩

/-- A decoder that reads the six tokens back in order recovers the container. -/
def decodeC {α : Type} : List (Tok α) → Option (Container α)
  | [.bytes hd, .vec x, .vec h, .vec p, .vec i, .vec b] => some ⟨hd, x, h, p, i, b⟩
  | _ => none

theorem decodeC_encodeM {M α : Type} (unload : M → Container α) (load : Container α → M)
    (h : UnloadLoad unload load) (m : M) :
    (decodeC (encodeM unload load m).1).map load = some m := by
  simp [encodeM, decodeC, h m]

/-! ## `Vector` handle and `Iter::new` -/

structure VecH where
  v : Nat

def vecFrom (h : Nat) : VecH := ⟨h⟩
def VecH.ptr (x : VecH) : Nat := x.v
theorem ptr_vecFrom (h : Nat) : (vecFrom h).ptr = h := rfl

/-- `Iter::new` (vector.rs:543): attach, `GxB_Vector_Iterator_seek(0)`;
`depleted` iff the seek reported `GxB_EXHAUSTED`, which per the GraphBLAS
spec happens iff the vector has no entry at position ≥ 0, i.e. is empty. -/
def vIterNew (entries : List Nat) (seekExhausted : Bool) : List Nat × Bool := (entries, seekExhausted)

theorem vIterNew_depleted (entries : List Nat) (seekExhausted : Bool)
    (spec : seekExhausted = true ↔ entries = []) :
    (vIterNew entries seekExhausted).2 = true ↔ entries = [] := spec

end GBWGlue
