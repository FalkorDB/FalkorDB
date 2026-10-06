/-
# Values, typed column lanes, classification, gather

| here | there |
| --- | --- |
| `FloatModel` | `f64`: `as f64` (`ofInt`), `partial_cmp` (`pcmp`), `+ - * / %` — abstract; the one IEEE fact used is a field (hypothesis, not an axiom) |
| `V` | `Value` (`runtime/value.rs`) restricted to the shapes the lanes distinguish (`other t` = String/List/Map/…) |
| `Column` | `Column` (`runtime/batch.rs:321-334`) |
| `Column.get?` | `Column::get` (`batch.rs:340-352`); `none` = index-out-of-bounds panic |
| `Column.len` | `Column::len` (`batch.rs:357-366`) |
| `gatherL`, `Column.gather` | `Column::gather` (`batch.rs:377-389`); `none` = panic |
| `Bits` | `NullBitmap` (`batch.rs:90-167`) as its list of bits (word packing is proved in proofs/ops_aggregate) |
| `intsOf`, `floatsOf`, `classifyNumeric` | `classify_numeric` (`batch.rs:191-231`), both `all()` passes |
| `classifyStored` | `classify_stored_column` (`batch.rs:241-268`) |
| `classifyColumn` / `classifyExact` | `classify_column` (`:279`) / `classify_exact_column` (`:295`) |
| `extendActive` | `extend_active_slice` (`batch.rs:308-317`) |
-/

namespace Columnar

/-- The `f64` operations the lanes use, kept abstract. `pcmp_swap` is the IEEE
fact that `partial_cmp` is antisymmetric (`b.partial_cmp(&a) == a.partial_cmp(&b).map(Ordering::reverse)`),
true for every pair of doubles, NaN included (`None` both ways). -/
class FloatModel (F : Type) where
  ofInt : Int → F
  pcmp : F → F → Option Ordering
  add : F → F → F
  sub : F → F → F
  mul : F → F → F
  div : F → F → F
  rem : F → F → F
  pcmp_swap : ∀ a b, pcmp b a = (pcmp a b).map Ordering.swap

/-- `Value`, restricted. Ints are `Int`s holding an `i64` (arithmetic wraps
explicitly, see `VectorExpr.wrap`). -/
inductive V (F : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (f : F)
  | node (n : Nat)
  | rel (n : Nat)
  | other (tag : Nat)

variable {F : Type}

def V.isNull : V F → Bool
  | .null => true
  | _ => false

inductive Column (F : Type) where
  | nodeIds (v : List Nat)
  | relIds (v : List Nat)
  | ints (v : List Int)
  | floats (v : List F)
  | values (v : List (V F))
  | unbound

/-! ## Gather on one lane -/

/-- `indices.map(|i| v[i]).collect()`: `none` when some index panics. -/
def gatherL {α : Type} : List α → List Nat → Option (List α)
  | _, [] => some []
  | v, i :: is =>
    match v[i]? with
    | none => none
    | some x => (gatherL v is).map (x :: ·)

theorem gatherL_length {α : Type} {v w : List α} :
    ∀ {idx : List Nat}, gatherL v idx = some w → w.length = idx.length
  | [], h => by simp [gatherL] at h; subst h; rfl
  | i :: is, h => by
    simp only [gatherL] at h
    split at h
    · simp at h
    · obtain ⟨w', hw', rfl⟩ := Option.map_eq_some_iff.mp h
      simp [gatherL_length hw']

/-- Gather reads position `k` from `v[idx[k]]`. -/
theorem gatherL_getElem? {α : Type} {v w : List α} :
    ∀ {idx : List Nat}, gatherL v idx = some w → ∀ k : Nat, w[k]? = (idx[k]?).bind (fun i : Nat => v[i]?)
  | [], h, k => by simp [gatherL] at h; subst h; simp
  | i :: is, h, k => by
    simp only [gatherL] at h
    split at h
    · simp at h
    · rename_i x hx
      obtain ⟨w', hw', rfl⟩ := Option.map_eq_some_iff.mp h
      cases k with
      | zero => simp [hx]
      | succ k => simp [gatherL_getElem? hw' k]

/-- Gather never panics iff every index is in range. -/
theorem gatherL_isSome_iff {α : Type} {v : List α} :
    ∀ {idx : List Nat}, (gatherL v idx).isSome ↔ ∀ i ∈ idx, i < v.length
  | [] => by simp [gatherL]
  | i :: is => by
    simp only [gatherL]
    split
    · rename_i h
      simp only [Option.isSome_none, Bool.false_eq_true, List.mem_cons, forall_eq_or_imp,
        false_iff, not_and]
      intro hi
      simp [List.getElem?_eq_getElem hi] at h
    · rename_i x hx
      have hi : i < v.length := by
        rcases Nat.lt_or_ge i v.length with h | h
        · exact h
        · simp [List.getElem?_eq_none h] at hx
      simp [hi, gatherL_isSome_iff (idx := is)]

/-- Gathering by `0..len` is the identity. -/
theorem gatherL_range {α : Type} (v : List α) : gatherL v (List.range v.length) = some v := by
  have hs : (gatherL v (List.range v.length)).isSome :=
    gatherL_isSome_iff.mpr (by intro i hi; simpa using hi)
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp hs
  rw [hw]
  congr 1
  apply List.ext_getElem?
  intro k
  rw [gatherL_getElem? hw k]
  rcases Nat.lt_or_ge k v.length with h | h
  · simp [List.getElem?_range h]
  · have : (List.range v.length)[k]? = none := List.getElem?_eq_none (by simpa using h)
    simp [List.getElem?_eq_none h, this]

/-- gather ∘ gather = gather by the composed index map. -/
theorem gatherL_gatherL {α : Type} {v w u : List α} {i j : List Nat}
    (h1 : gatherL v i = some w) (h2 : gatherL w j = some u) (k : Nat) :
    u[k]? = (j[k]?).bind (fun t => (i[t]?).bind (fun s => v[s]?)) := by
  rw [gatherL_getElem? h2 k]
  cases j[k]? with
  | none => rfl
  | some t => simp [gatherL_getElem? h1 t]

theorem gatherL_map {α β : Type} (f : α → β) {v : List α} :
    ∀ {idx : List Nat}, gatherL (v.map f) idx = (gatherL v idx).map (List.map f)
  | [] => by simp [gatherL]
  | i :: is => by
    simp only [gatherL, List.getElem?_map]
    cases v[i]? with
    | none => rfl
    | some x =>
      simp only [Option.map_some]
      rw [gatherL_map (idx := is)]
      simp [Option.map_map, Function.comp_def]

namespace Column

/-- `Column::get`: `none` models the index panic; `Unbound` reads `Null`. -/
def get? : Column F → Nat → Option (V F)
  | .nodeIds v, r => (v[r]?).map V.node
  | .relIds v, r => (v[r]?).map V.rel
  | .ints v, r => (v[r]?).map V.int
  | .floats v, r => (v[r]?).map V.float
  | .values v, r => v[r]?
  | .unbound, _ => some .null

def len : Column F → Nat
  | .nodeIds v => v.length
  | .relIds v => v.length
  | .ints v => v.length
  | .floats v => v.length
  | .values v => v.length
  | .unbound => 0

def isUnbound : Column F → Bool
  | .unbound => true
  | _ => false

/-- The column as a `Value` list (`None` for `Unbound`, which has no rows). -/
def toValues : Column F → Option (List (V F))
  | .nodeIds v => some (v.map V.node)
  | .relIds v => some (v.map V.rel)
  | .ints v => some (v.map V.int)
  | .floats v => some (v.map V.float)
  | .values v => some v
  | .unbound => none

/-- `Column::gather`. -/
def gather : Column F → List Nat → Option (Column F)
  | .nodeIds v, idx => (gatherL v idx).map .nodeIds
  | .relIds v, idx => (gatherL v idx).map .relIds
  | .ints v, idx => (gatherL v idx).map .ints
  | .floats v, idx => (gatherL v idx).map .floats
  | .values v, idx => (gatherL v idx).map .values
  | .unbound, _ => some .unbound

theorem get?_eq_toValues {c : Column F} {vs : List (V F)} (h : c.toValues = some vs) (r : Nat) :
    c.get? r = vs[r]? := by
  cases c <;> simp [toValues] at h <;> subst h <;> simp [get?]

theorem len_eq_toValues {c : Column F} {vs : List (V F)} (h : c.toValues = some vs) :
    c.len = vs.length := by
  cases c <;> simp [toValues] at h <;> subst h <;> simp [len]

theorem toValues_isSome {c : Column F} (h : c.isUnbound = false) : ∃ vs, c.toValues = some vs := by
  cases c <;> simp_all [toValues, isUnbound]

/-- Every in-range read of a bound column succeeds (no panic). -/
theorem get?_isSome {c : Column F} {r : Nat} (h : c.isUnbound = false) (hr : r < c.len) :
    (c.get? r).isSome := by
  cases c <;> simp_all [get?, len, isUnbound]

theorem gather_isUnbound {c c' : Column F} {idx : List Nat} (h : c.gather idx = some c') :
    c'.isUnbound = c.isUnbound := by
  cases c <;> simp [gather] at h <;>
    first
    | (obtain ⟨w, _, rfl⟩ := h; rfl)
    | (subst h; rfl)

/-- Gather is the lane gather of the value view. -/
theorem gather_toValues {c c' : Column F} {idx : List Nat} (h : c.gather idx = some c')
    {vs : List (V F)} (hv : c.toValues = some vs) :
    c'.toValues = gatherL vs idx := by
  cases c with
  | unbound => simp [toValues] at hv
  | values v =>
    simp [toValues] at hv; subst hv; simp [gather] at h
    obtain ⟨w, hw, rfl⟩ := h; simp [toValues, hw]
  | nodeIds v =>
    simp [toValues] at hv; subst hv; simp [gather] at h
    obtain ⟨w, hw, rfl⟩ := h; simp [toValues, hw, gatherL_map]
  | relIds v =>
    simp [toValues] at hv; subst hv; simp [gather] at h
    obtain ⟨w, hw, rfl⟩ := h; simp [toValues, hw, gatherL_map]
  | ints v =>
    simp [toValues] at hv; subst hv; simp [gather] at h
    obtain ⟨w, hw, rfl⟩ := h; simp [toValues, hw, gatherL_map]
  | floats v =>
    simp [toValues] at hv; subst hv; simp [gather] at h
    obtain ⟨w, hw, rfl⟩ := h; simp [toValues, hw, gatherL_map]

/-- **gather reads row `idx[k]`**: `(c.gather idx).get(k) = c.get(idx[k])`. -/
theorem gather_get? {c c' : Column F} {idx : List Nat} (h : c.gather idx = some c') (k : Nat)
    (hk : k < idx.length) : c'.get? k = c.get? idx[k] := by
  cases hc : c.isUnbound
  · obtain ⟨vs, hv⟩ := toValues_isSome hc
    have hc'u : c'.isUnbound = false := (gather_isUnbound h).trans hc
    obtain ⟨ws, hw⟩ := toValues_isSome hc'u
    have hg : gatherL vs idx = some ws := (gather_toValues h hv).symm.trans hw
    rw [get?_eq_toValues hw, get?_eq_toValues hv, gatherL_getElem? hg k,
      List.getElem?_eq_getElem hk]
    rfl
  · cases c <;> simp [isUnbound] at hc
    simp [gather] at h; subst h; rfl

theorem gather_len {c c' : Column F} {idx : List Nat} (h : c.gather idx = some c')
    (hu : c.isUnbound = false) : c'.len = idx.length := by
  cases c <;> simp only [gather, isUnbound] at h hu <;>
    first
    | (obtain ⟨w, hw, rfl⟩ := Option.map_eq_some_iff.mp h; simp [len, gatherL_length hw])
    | simp at hu

/-- Gather never panics when every index is below the column length. -/
theorem gather_isSome {c : Column F} {idx : List Nat} (h : ∀ i ∈ idx, i < c.len) :
    (c.gather idx).isSome := by
  cases c <;> simp only [gather, len] at h ⊢ <;>
    first
    | (have := gatherL_isSome_iff.mpr h
       simpa [Option.isSome_map] using this)
    | rfl

end Column

/-! ## Null bitmaps and classification -/

/-- `NullBitmap` as its bit list (bit `i` set = row `i` null). -/
abbrev Bits := List Bool

def Bits.none (n : Nat) : Bits := List.replicate n false
def Bits.all (n : Nat) : Bits := List.replicate n true
/-- `NullBitmap::from_values`. -/
def Bits.ofValues (vs : List (V F)) : Bits := vs.map V.isNull
/-- `NullBitmap::is_null` (debug-asserted in range; out of range reads 0). -/
def Bits.isNull (b : Bits) (i : Nat) : Bool := b[i]?.getD false

/-- How a non-int numeric column may fall back to an f64 lane (`FloatLane`, `batch.rs:171`). -/
inductive FloatLane where
  | none | pure | promote
  deriving DecidableEq

/-- First `all()` pass of `classify_numeric` (`batch.rs:196-209`). -/
def intsOf (allowNull : Bool) : List (V F) → Option (List Int)
  | [] => some []
  | .int i :: vs => (intsOf allowNull vs).map (i :: ·)
  | .null :: vs => if allowNull then (intsOf allowNull vs).map (0 :: ·) else none
  | _ :: _ => none

/-- Second `all()` pass (`batch.rs:210-229`). -/
def floatsOf [FloatModel F] (allowNull : Bool) (lane : FloatLane) : List (V F) → Option (List F)
  | [] => some []
  | .int i :: vs =>
    if lane = .promote then (floatsOf allowNull lane vs).map (FloatModel.ofInt i :: ·) else none
  | .float f :: vs => (floatsOf allowNull lane vs).map (f :: ·)
  | .null :: vs =>
    if allowNull then (floatsOf allowNull lane vs).map (FloatModel.ofInt 0 :: ·) else none
  | _ :: _ => none

/-- `classify_numeric` (`batch.rs:191-231`). `0.0` is modelled as `ofInt 0`. -/
def classifyNumeric [FloatModel F] (vs : List (V F)) (allowNull : Bool) (lane : FloatLane) :
    Column F :=
  match intsOf allowNull vs with
  | some is => .ints is
  | none =>
    if lane ≠ .none then
      match floatsOf allowNull lane vs with
      | some fs => .floats fs
      | none => .values vs
    else .values vs

def nodesOf : List (V F) → Option (List Nat)
  | [] => some []
  | .node n :: vs => (nodesOf vs).map (n :: ·)
  | _ :: _ => none

def relsOf : List (V F) → Option (List Nat)
  | [] => some []
  | .rel n :: vs => (relsOf vs).map (n :: ·)
  | _ :: _ => none

/-- `classify_stored_column` (`batch.rs:241-268`). -/
def classifyStored [FloatModel F] (vs : List (V F)) : Column F :=
  if vs = [] then .values vs
  else match nodesOf vs with
    | some ns => .nodeIds ns
    | none => match relsOf vs with
      | some rs => .relIds rs
      | none => classifyNumeric vs false .pure

/-- `classify_column` (`batch.rs:279`). -/
def classifyColumn [FloatModel F] (vs : List (V F)) : Column F × Bits :=
  (classifyNumeric vs true .promote, Bits.ofValues vs)

/-- `classify_exact_column` (`batch.rs:295`). -/
def classifyExact [FloatModel F] (vs : List (V F)) : Column F × Bits :=
  (classifyNumeric vs true .pure, Bits.ofValues vs)

/-- Reading a typed lane back through its null bitmap (`ExprColumn::get`). -/
def readN (c : Column F) (nulls : Bits) (i : Nat) : Option (V F) :=
  if nulls.isNull i then some .null else c.get? i

section lemmas
set_option linter.unusedSectionVars false
variable [FloatModel F]

/-- Without nulls allowed, an int lane reads back exactly. -/
theorem intsOf_false_exact :
    ∀ {vs : List (V F)} {is : List Int}, intsOf false vs = some is → is.map V.int = vs
  | [], is, h => by simp [intsOf] at h; subst h; rfl
  | v :: vs, is, h => by
    cases v <;> simp [intsOf] at h
    obtain ⟨is', h', rfl⟩ := h
    simp [intsOf_false_exact h']

theorem floatsOf_false_pure_exact :
    ∀ {vs : List (V F)} {fs : List F}, floatsOf false .pure vs = some fs → fs.map V.float = vs
  | [], fs, h => by simp [floatsOf] at h; subst h; rfl
  | v :: vs, fs, h => by
    cases v <;> simp [floatsOf] at h
    obtain ⟨fs', h', rfl⟩ := h
    simp [floatsOf_false_pure_exact h']

theorem nodesOf_exact : ∀ {vs : List (V F)} {ns : List Nat}, nodesOf vs = some ns → ns.map V.node = vs
  | [], ns, h => by simp [nodesOf] at h; subst h; rfl
  | v :: vs, ns, h => by
    cases v <;> simp [nodesOf] at h
    obtain ⟨ns', h', rfl⟩ := h
    simp [nodesOf_exact h']

theorem relsOf_exact : ∀ {vs : List (V F)} {ns : List Nat}, relsOf vs = some ns → ns.map V.rel = vs
  | [], ns, h => by simp [relsOf] at h; subst h; rfl
  | v :: vs, ns, h => by
    cases v <;> simp [relsOf] at h
    obtain ⟨ns', h', rfl⟩ := h
    simp [relsOf_exact h']

/-- **`classify_stored_column` is lossless**: the classified column reads back
exactly the input values, at every row. -/
theorem classifyStored_toValues (vs : List (V F)) : (classifyStored vs).toValues = some vs := by
  unfold classifyStored
  split
  · rfl
  · split
    · rename_i ns h; simp [Column.toValues, nodesOf_exact h]
    · split
      · rename_i rs _ h; simp [Column.toValues, relsOf_exact h]
      · unfold classifyNumeric
        split
        · rename_i is h; simp [Column.toValues, intsOf_false_exact h]
        · simp only [ne_eq, reduceCtorEq, not_false_eq_true, ↓reduceIte]
          split
          · rename_i fs h; simp [Column.toValues, floatsOf_false_pure_exact h]
          · rfl

theorem classifyStored_get? (vs : List (V F)) (r : Nat) : (classifyStored vs).get? r = vs[r]? :=
  Column.get?_eq_toValues (classifyStored_toValues vs) r

theorem classifyStored_isUnbound (vs : List (V F)) : (classifyStored vs).isUnbound = false := by
  cases h : classifyStored vs <;> simp [Column.isUnbound]
  have := classifyStored_toValues vs
  rw [h] at this; simp [Column.toValues] at this

theorem classifyStored_len (vs : List (V F)) : (classifyStored vs).len = vs.length :=
  Column.len_eq_toValues (classifyStored_toValues vs)

/-- With a null bitmap, the int lane with `0` placeholders reads back exactly. -/
theorem intsOf_true_exact :
    ∀ {vs : List (V F)} {is : List Int}, intsOf true vs = some is →
      ∀ k, readN (.ints is) (Bits.ofValues vs) k = vs[k]?
  | [], is, h, k => by simp [intsOf] at h; subst h; simp [readN, Bits.isNull, Bits.ofValues, Column.get?]
  | v :: vs, is, h, k => by
    cases v <;> simp [intsOf] at h
    all_goals
      obtain ⟨is', h', rfl⟩ := h
      have ih := intsOf_true_exact h'
      cases k with
      | zero => simp [readN, Bits.isNull, Bits.ofValues, V.isNull, Column.get?]
      | succ k =>
        have := ih k
        simp only [readN, Bits.isNull, Bits.ofValues, Column.get?] at this ⊢
        simpa using this

theorem floatsOf_true_pure_exact :
    ∀ {vs : List (V F)} {fs : List F}, floatsOf true .pure vs = some fs →
      ∀ k, readN (.floats fs) (Bits.ofValues vs) k = vs[k]?
  | [], fs, h, k => by
    simp [floatsOf] at h; subst h; simp [readN, Bits.isNull, Bits.ofValues, Column.get?]
  | v :: vs, fs, h, k => by
    cases v <;> simp [floatsOf] at h
    all_goals
      obtain ⟨fs', h', rfl⟩ := h
      have ih := floatsOf_true_pure_exact h'
      cases k with
      | zero => simp [readN, Bits.isNull, Bits.ofValues, V.isNull, Column.get?]
      | succ k =>
        have := ih k
        simp only [readN, Bits.isNull, Bits.ofValues, Column.get?] at this ⊢
        simpa using this

theorem readN_values (vs : List (V F)) (k : Nat) :
    readN (.values vs) (Bits.ofValues vs) k = vs[k]? := by
  unfold readN Bits.isNull Bits.ofValues
  cases h : vs[k]? with
  | none => simp [h, Column.get?]
  | some v => cases v <;> simp [h, V.isNull, Column.get?]

/-- **`classify_exact_column` is lossless** through its null bitmap — so the
filter kernels read exactly the values the scalar evaluator would. -/
theorem classifyExact_exact (vs : List (V F)) (k : Nat) :
    readN (classifyExact vs).1 (classifyExact vs).2 k = vs[k]? := by
  simp only [classifyExact, classifyNumeric]
  split
  · rename_i is h; exact intsOf_true_exact h k
  · simp only [ne_eq, reduceCtorEq, not_false_eq_true, ↓reduceIte]
    split
    · rename_i fs h; exact floatsOf_true_pure_exact h k
    · exact readN_values vs k

/-- **`classify_column` is lossy** (`FloatLane::Promote`): a mixed int/float
column reads the int back as a float. Latent — `classify_column` and its only
wrappers `Runtime::materialize_{node,relationship}_property` (`runtime.rs:1543`,
`:1605`) have no caller. -/
theorem classifyColumn_lossy (f : F) :
    readN (classifyColumn [V.int 1, V.float f]).1 (classifyColumn [V.int 1, V.float f]).2 0
      = some (V.float (FloatModel.ofInt 1)) := by
  simp [classifyColumn, classifyNumeric, intsOf, floatsOf, readN, Bits.isNull, Bits.ofValues,
    V.isNull, Column.get?]

theorem classifyColumn_ne (f : F) :
    readN (classifyColumn [V.int 1, V.float f]).1 (classifyColumn [V.int 1, V.float f]).2 0
      ≠ [V.int 1, V.float f][0]? := by
  rw [classifyColumn_lossy]; simp

end lemmas

/-! ## `extend_active_slice` -/

/-- `extend_active_slice` (`batch.rs:308-317`) as the slice it appends. -/
def extendActive {α : Type} (src : List α) : Option (List Nat) → Option (List α)
  | none => some src
  | some sel => gatherL src sel

/-- Both arms append exactly the active rows: the dense arm is the gather by
`0..len`. -/
theorem extendActive_eq_gather {α : Type} (src : List α) (sel : Option (List Nat)) :
    extendActive src sel = gatherL src (sel.getD (List.range src.length)) := by
  cases sel with
  | none => simp [extendActive, gatherL_range]
  | some s => rfl

end Columnar
