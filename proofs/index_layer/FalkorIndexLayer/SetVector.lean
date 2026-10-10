import FalkorIndexLayer.IndexM
/-
# `Document::set`, vector arm (`graph/src/index/mod.rs:730-744`, #3087 @ 49f698d22)

```rust
if field.ty == IndexType::Vector {
    if let Value::VecF32(vec) = value
        && field.vector_options.as_ref()
            .is_some_and(|o| o.dimension != 0 && o.dimension == vec.len() as u64)
    { RediSearch_DocumentAddFieldVector(doc, name, vec.as_ptr(), vec.len() * size_of::<f32>()); }
    return;
}
```

`setVector f vec`: `vec = some len` iff the value is a `VecF32` of `len` floats; the
result is the blob size handed to `RediSearch_DocumentAddFieldVector`, `none` = no call.
The Range arm is `setRange`/`docOf` (`Model.lean`); a vector field never reaches it.

RediSearch rejects the whole document when a vector blob is not `dim * 4` bytes or
the field has no vector params (`RsKind.vector none`, which `registerOne` creates
for `dimension = 0` or no options). The theorems below show that, since #3087, `set`
never hands RediSearch such a blob, which is what justifies the `ht` hypothesis of
`Iter.query_eq_scan_filter` (the label's table holds `docOf` of every entity) in the
presence of vector fields. The end-to-end acceptance theorem over the RediSearch
contract is `SC.Search.doc_always_accepted` (proofs/search_concurrency).

HISTORICAL (fe619ac5f, before #3087): `setVectorPre3087` added every `VecF32`
unchecked; `pre3087_*` below are the counterexamples (W3-conc-3 / #3075, and
W5-idx-2's no-options shape), fixed by #3087 (49f698d22).
-/
namespace IndexLayer.Meta

/-- The vector arm of `Document::set` (mod.rs:730-744), literally. -/
def setVector (f : Field) (vec : Option Nat) : Option Nat :=
  if f.ty = .vector then
    match vec with
    | some len => if f.vopts.any (fun o => o.dimension != 0 && o.dimension == len)
                  then some (len * 4) else none
    | none => none
  else none

/-- HISTORICAL: the arm at fe619ac5f (no dimension check). -/
def setVectorPre3087 (f : Field) (vec : Option Nat) : Option Nat :=
  if f.ty = .vector then vec.map (· * 4) else none

/-- **Key property.** For any field options and any value, a vector add has the
field's own non-zero dimension: never a disagreeing or absent one. -/
theorem setVector_sound {f : Field} {vec : Option Nat} {nb : Nat}
    (h : setVector f vec = some nb) :
    f.ty = .vector ∧ ∃ o, f.vopts = some o ∧ o.dimension ≠ 0 ∧ vec = some o.dimension ∧
      nb = o.dimension * 4 := by
  unfold setVector at h
  split at h
  · rename_i hty
    refine ⟨hty, ?_⟩
    cases vec with
    | none => simp at h
    | some len =>
      simp only at h
      split at h
      · rename_i hany
        cases ho : f.vopts with
        | none => simp [ho] at hany
        | some o =>
          simp [ho] at hany
          obtain ⟨h0, rfl⟩ := hany
          cases h
          exact ⟨o, rfl, h0, rfl, rfl⟩
      · cases h
  · cases h

/-- A vector of the index's dimension is still added (the guard drops nothing good). -/
theorem setVector_good {f : Field} {o : VecOpts} (hty : f.ty = .vector)
    (ho : f.vopts = some o) (h0 : o.dimension ≠ 0) :
    setVector f (some o.dimension) = some (o.dimension * 4) := by
  simp [setVector, hty, ho, h0]

/-- Wrong dimension: nothing is added. -/
theorem setVector_wrong_dim {f : Field} {o : VecOpts} {len : Nat}
    (ho : f.vopts = some o) (hne : len ≠ o.dimension) : setVector f (some len) = none := by
  unfold setVector
  split
  · simp only [ho, Option.any_some]
    have : (o.dimension != 0 && o.dimension == len) = false := by
      simp; intro _; exact fun h => hne h.symm
    simp [this]
  · rfl

/-- Dimension 0 or no vector options: nothing is ever added. -/
theorem setVector_noparams {f : Field} (h : f.vopts = none ∨ ∃ o, f.vopts = some o ∧ o.dimension = 0)
    (vec : Option Nat) : setVector f vec = none := by
  cases hs : setVector f vec with
  | none => rfl
  | some nb =>
    obtain ⟨_, o, ho, h0, _⟩ := setVector_sound hs
    rcases h with h | ⟨o', ho', h0'⟩
    · rw [h] at ho; cases ho
    · rw [ho'] at ho; cases ho; exact absurd h0' h0

/-- Link to `register_fields`: whenever `set` adds a vector, the field was registered
*with* vector params (`RsKind.vector (some m)`) of the same `vopts.dimension`. -/
theorem setVector_has_params {f : Field} {vec : Option Nat} {nb : Nat} {fo : Option TextOpts}
    {tieredOk : Bool} {r : List RsReg}
    (h : setVector f vec = some nb) (hr : Idx.registerOne fo tieredOk f = .ok r) :
    ∃ m o, r = [⟨f.name, .vector (some m)⟩] ∧ f.vopts = some o ∧ nb = o.dimension * 4 := by
  obtain ⟨hty, o, ho, h0, _, hnb⟩ := setVector_sound h
  simp only [Idx.registerOne, hty, ho] at hr
  have hpos : o.dimension > 0 := Nat.pos_of_ne_zero h0
  simp only [hpos, ite_true] at hr
  split at hr
  · rename_i m _
    split at hr
    · cases hr; exact ⟨m, o, rfl, ho, hnb⟩
    · cases hr
  · cases hr

/-- Conversely, a field registered *without* params never gets a vector add. -/
theorem setVector_none_of_noparams {f : Field} {fo : Option TextOpts} {tieredOk : Bool}
    (hr : Idx.registerOne fo tieredOk f = .ok [⟨f.name, .vector none⟩]) (vec : Option Nat) :
    setVector f vec = none := by
  cases hs : setVector f vec with
  | none => rfl
  | some nb =>
    obtain ⟨m, _, hr', _⟩ := setVector_has_params hs hr
    cases hr'

/-- #3087 changed only which `VecF32` values are added, never adding a new call. -/
theorem setVector_sub_pre {f : Field} {vec : Option Nat} {nb : Nat}
    (h : setVector f vec = some nb) : setVectorPre3087 f vec = some nb := by
  obtain ⟨hty, o, _, _, rfl, rfl⟩ := setVector_sound h
  simp [setVectorPre3087, hty]

/-! ### HISTORICAL counterexamples (fe619ac5f), fixed by #3087 (49f698d22) -/

def vf (vopts : Option VecOpts) : Field :=
  { name := "vector:v", ty := .vector, options := none, vopts, numArr := none, strArr := none }

/-- `{dimension:2}`, `vecf32([1,2,3])`: a 12-byte blob for an 8-byte field. -/
theorem pre3087_dim_mismatch :
    setVectorPre3087 (vf (some ⟨2, none⟩)) (some 3) = some 12 ∧
    setVector (vf (some ⟨2, none⟩)) (some 3) = none := by
  decide

/-- `{dimension:0}` and no options (W5-idx-2's half-created index): a blob on a field
with no vector params. -/
theorem pre3087_noparams :
    setVectorPre3087 (vf (some ⟨0, none⟩)) (some 3) = some 12 ∧
    setVectorPre3087 (vf none) (some 3) = some 12 ∧
    setVector (vf (some ⟨0, none⟩)) (some 3) = none ∧
    setVector (vf none) (some 3) = none := by
  decide

end IndexLayer.Meta
