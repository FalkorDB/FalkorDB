import FalkorAggOps.Agg
import FalkorValueOrder.Model
/-!
# The scalar `max` body on `Value` (aggregation.rs)

`Agg.maxScalar_eq_batch` relates the scalar `max` body to `max_batch` for any abstract comparison
`c` with three properties (`anti`, `nanSymm`, `noflag`). Those are properties of
`Value::compare_value`, so until instantiated they were an undischarged hypothesis about other
Rust code (the row was CONDITIONAL). Here `c` is the real model `cmpV` (proofs/value_order
`Model.lean`, `compare_value` value.rs):

* `maxV` is the scalar body, line by line, on `cmpV`; `maxV_spec` states what it returns.
* On NaN-free, non-null scalars (the `scalarOk` fragment: Bool, i64-exact Int, finite Float, String,
  Node) the three hypotheses are *proved* for `cmpV` (`anti_scalar`, `nanFlag_scalar`,
  `nullFlag_scalar`), and `maxV_eq_batch` is `maxScalar_eq_batch` instantiated: scalar `max`
  and `max_batch` agree there.
* Outside that fragment they fail (a list with a nested NaN: `[NaN]` against `[5]` is
  `(Less, None)` both ways, `value_order Main` / `compare_list`), so the two bodies can differ. The
  scalar body is unreachable: every aggregate registers `batch_agg`, which aggregate.rs/:1061
  always prefer, so `max_batch` (covered by `Agg.minmax_hom`) is what runs.
-/
namespace FalkorAggOps.MaxScalar
open ValueOrder FalkorCore

/-- `compare_value` as the `Cmp` of `Agg`: ordering, null flag, NaN flag. -/
def cmpOfV : Cmp ValueOrder.V where
  ord a b := (cmpV a b).1
  nullFlag a b := decide ((cmpV a b).2 = .comparedNull)
  nanFlag a b := decide ((cmpV a b).2 = .nan)

/-- The scalar `max` body (aggregation.rs) on `Value`: `args = [a, b]` with `a` the new
value and `b` the accumulator (`None` = `Null` accumulator takes `a` in `max_batch`). -/
def maxV (a : ValueOrder.V) (b : Option ValueOrder.V) : Option ValueOrder.V := maxScalar cmpOfV a b

/-- What the body returns: `a` when `b.compare_value(a)` is `Less` and not a NaN comparison, or a
null comparison; else `b`. -/
theorem maxV_spec (a b : ValueOrder.V) :
    maxV a (some b) =
      if ((cmpV b a).1 = .lt ∧ (cmpV b a).2 ≠ .nan) ∨ (cmpV b a).2 = .comparedNull then some a else some b := by
  simp only [maxV, maxScalar, cmpOfV]
  by_cases h1 : (cmpV b a).1 = .lt <;> by_cases h2 : (cmpV b a).2 = .nan <;>
    by_cases h3 : (cmpV b a).2 = .comparedNull <;> simp [h1, h2, h3]

/-- Non-null scalars without NaN. -/
def scalarOk (v : ValueOrder.V) : Prop := sane v ∧ v ≠ .null

theorem flag_scalar (a b : ValueOrder.V) (ha : scalarOk a) (hb : scalarOk b) :
    (cmpV a b).2 = .none ∨ (cmpV a b).2 = .disjoint := by
  obtain ⟨ha, ha'⟩ := ha
  obtain ⟨hb, hb'⟩ := hb
  cases a with
  | null => exact absurd rfl ha'
  | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at ha
  | flt f => cases f with
    | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at ha
    | num n => cases b with
      | null => exact absurd rfl hb'
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [cmpV, cmpF]
      | int j => simp [cmpV, cmpF]
      | _ => simp [cmpV, isNull]
  | int i => cases b with
      | null => exact absurd rfl hb'
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [cmpV, cmpF]
      | int j => simp [cmpV]
      | _ => simp [cmpV, isNull]
  | bool x => cases b with
      | null => exact absurd rfl hb'
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [cmpV, isNull]
      | _ => simp [cmpV, isNull]
  | str x => cases b with
      | null => exact absurd rfl hb'
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [cmpV, isNull]
      | _ => simp [cmpV, isNull]
  | node x => cases b with
      | null => exact absurd rfl hb'
      | list _ | map _ _ | vecf32 _ | point _ _ => simp [sane] at hb
      | flt g => cases g with
        | nan _ _ | negZero | frac _ _ | negInf | posInf => simp [sane] at hb
        | num m => simp [cmpV, isNull]
      | _ => simp [cmpV, isNull]

/-- The `Cmp` restricted to the scalar fragment. -/
abbrev SV := {v : ValueOrder.V // scalarOk v}

def cmpSV : Cmp SV where
  ord a b := (cmpV a.1 b.1).1
  nullFlag a b := decide ((cmpV a.1 b.1).2 = .comparedNull)
  nanFlag a b := decide ((cmpV a.1 b.1).2 = .nan)

theorem nanFlag_scalar (a b : SV) : cmpSV.nanFlag a b = false := by
  rcases flag_scalar a.1 b.1 a.2 b.2 with h | h <;> simp [cmpSV, h]

theorem nullFlag_scalar (a b : SV) : cmpSV.nullFlag a b = false := by
  rcases flag_scalar a.1 b.1 a.2 b.2 with h | h <;> simp [cmpSV, h]

theorem anti_scalar (a b : SV) :
    cmpSV.nanFlag a b = false → (cmpSV.ord b a = .lt ↔ cmpSV.ord a b = .gt) := by
  intro _
  have h := sane_swap a.1 b.1 a.2.1 b.2.1
  simp only [cmpSV, sortCmp] at h ⊢
  rw [h]
  cases (cmpV a.1 b.1).1 <;> simp [Ordering.swap]

/-- **`maxScalar_eq_batch`, instantiated for `Value`'s `compare_value`** on NaN-free non-null
scalars: the scalar `max` body and `max_batch`'s step coincide. -/
theorem maxV_eq_batch (a : SV) (b : Option SV) :
    maxScalar cmpSV a b = mmStep cmpSV .gt b a :=
  maxScalar_eq_batch cmpSV anti_scalar
    (fun a b => by rw [nanFlag_scalar a b, nanFlag_scalar b a]) nullFlag_scalar a b

/-- A scalar NaN carries the NaN flag in both directions (value_order `nan_breaks_antisymmetry`:
`NaN` is `Less` than `1.0` and `1.0` is `Less` than `NaN`), so `anti` is only required where the
flag is clear; both bodies ignore a flagged comparison, and `Agg.maxScalar_eq_batch_nan` shows
they agree on it. The fragment excludes only what drops the flag: a list with a nested NaN. -/
theorem nan_flag_set :
    (cmpV (.flt F.qnan) (.flt (.num 1))).2 = .nan ∧ (cmpV (.flt (.num 1)) (.flt F.qnan)).2 = .nan := by
  decide

end FalkorAggOps.MaxScalar
