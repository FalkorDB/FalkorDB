import FalkorTemporal.Pure
/-!
# Spatial and vector functions (functions/spatial.rs:38-229), `Point::new` / `Point::distance`
(value.rs:133-176)

f64/f32 are abstract (`Flt`), with only the IEEE facts the theorems use stated as fields.
`simsimd` (vec_distance.rs, FFI) is two partial functions whose only assumed law is the
one the wrapper's comment relies on: `Some` whenever the lengths agree.
-/
namespace FalkorTemporal.Spatial2

structure Flt (F F32 : Type) where
  zero : F
  one : F
  two : F
  radius : F                  -- `EARTH_RADIUS = 6_378_140.0`
  sub : F → F → F
  mul : F → F → F
  div : F → F → F
  mulAdd : F → F → F → F      -- `a.mul_add(b, c)`
  sin : F → F
  cos : F → F
  sqrt : F → F
  atan2 : F → F → F
  sq : F → F                  -- `powi(2)`
  clamp01 : F → F             -- `.clamp(0.0, 1.0)` (since #2983)
  toRadF64 : F32 → F          -- `f64::from(x.to_radians())`
  ofI64 : Int → F32           -- `i as f32`
  ofF64 : F → F32             -- `f as f32`
  /-- IEEE facts (exact in binary64): x − x = 0 for finite x, 0/2 = 0, sin 0 = 0, 0² = 0,
  fma(x, 0, 0) = 0 for finite x, √0 = 0, 1 − 0 = 1, √1 = 1, atan2(0, 1) = 0, 2·0 = 0, R·0 = 0. -/
  finite : F → Bool
  sub_self : ∀ x, finite x = true → sub x x = zero
  div_zero_two : div zero two = zero
  sin_zero : sin zero = zero
  sq_zero : sq zero = zero
  mulAdd_zero : ∀ x, finite x = true → mulAdd x zero zero = zero
  sqrt_zero : sqrt zero = zero
  one_sub_zero : sub one zero = one
  sqrt_one : sqrt one = one
  atan2_zero_one : atan2 zero one = zero
  two_mul_zero : mul two zero = zero
  r_mul_zero : mul radius zero = zero
  clamp01_zero : clamp01 zero = zero
  cos_finite : ∀ x, finite x = true → finite (cos x) = true
  mul_finite : ∀ x y, finite x = true → finite y = true → finite (mul x y) = true

variable {F F32 : Type} (L : Flt F F32)

structure Point (F32 : Type) where
  latitude : F32
  longitude : F32

/-- `Point::new` (value.rs:133). -/
def Point.new (latitude longitude : F32) : Point F32 := { latitude, longitude }

theorem Point.new_fields (a b : F32) : (Point.new a b).latitude = a ∧ (Point.new a b).longitude = b :=
  ⟨rfl, rfl⟩

/-- The haversine term `a = cos φ₁ cos φ₂ · sin²(Δλ/2) + sin²(Δφ/2)` of `Point::distance`
(value.rs:144-160), before the clamp. -/
def havA (p q : Point F32) : F :=
  let lat1 := L.toRadF64 p.latitude
  let lon1 := L.toRadF64 p.longitude
  let lat2 := L.toRadF64 q.latitude
  let lon2 := L.toRadF64 q.longitude
  let dlat := L.sub lat2 lat1
  let dlon := L.sub lon2 lon1
  L.mulAdd (L.mul (L.cos lat1) (L.cos lat2)) (L.sq (L.sin (L.div dlon L.two)))
    (L.sq (L.sin (L.div dlat L.two)))

/-- `c = 2·atan2(√a, √(1 − a))`, times `EARTH_RADIUS` (value.rs:163-167). -/
def havFinish (a : F) : F :=
  L.mul L.radius (L.mul L.two (L.atan2 (L.sqrt a) (L.sqrt (L.sub L.one a))))

/-- `Point::distance` (value.rs:144), haversine, operation by operation; since #2983
(86dcd5a11) `a` is clamped to `[0, 1]`. -/
def distance (p q : Point F32) : F := havFinish L (L.clamp01 (havA L p q))

/-- Pre-#2983 `Point::distance`: no clamp (history, W2-temporal-4). -/
def pre2983_distance (p q : Point F32) : F := havFinish L (havA L p q)

/-- The distance from a (finite) point to itself is exactly 0. -/
theorem distance_self (p : Point F32) (h1 : L.finite (L.toRadF64 p.latitude) = true)
    (h2 : L.finite (L.toRadF64 p.longitude) = true) : distance L p p = L.zero := by
  have hc := L.mul_finite _ _ (L.cos_finite _ h1) (L.cos_finite _ h1)
  simp only [distance, havA, havFinish, L.sub_self _ h1, L.sub_self _ h2, L.div_zero_two,
    L.sin_zero, L.sq_zero, L.mulAdd_zero _ hc, L.clamp01_zero, L.sqrt_zero, L.one_sub_zero,
    L.sqrt_one, L.atan2_zero_one, L.two_mul_zero, L.r_mul_zero]

section Antipodal

/-! ### #2983: antipodal points (W2-temporal-4)

The IEEE facts the NaN argument needs, stated over the same abstract `F`. `le` is IEEE `≤`
(false whenever an operand is NaN). Each is a property of binary64 and of `f64::clamp`:
`clamp_unit`/`clamp_id`/`clamp_nan` are `f64::clamp(0.0, 1.0)`'s documented contract;
`one_sub_unit`: `1 − a ≥ 0` for `a ∈ [0, 1]` (rounding is monotone and `1 − 1 = 0`);
`one_sub_above`: `1 − a < 0` for finite `a > 1`; `sqrt_*`: `√x` is NaN iff `x < 0` or NaN;
`atan2_*`: `atan2` of non-NaN operands is in `[−π, π]`, of a NaN operand NaN;
`mul_nan_right`: NaN propagates through `*`. -/
structure NanLaws (L : Flt F F32) where
  nan : F → Bool
  le : F → F → Prop
  nan_of_finite : ∀ x, L.finite x = true → nan x = false
  clamp_unit : ∀ x, nan x = false → le L.zero (L.clamp01 x) ∧ le (L.clamp01 x) L.one
  clamp_id : ∀ x, le L.zero x → le x L.one → L.clamp01 x = x
  one_sub_unit : ∀ a, le L.zero a → le a L.one → le L.zero (L.sub L.one a)
  one_sub_above : ∀ a, nan a = false → ¬ le a L.one →
    ¬ le L.zero (L.sub L.one a) ∧ nan (L.sub L.one a) = false
  sqrt_nonneg : ∀ x, le L.zero x → nan (L.sqrt x) = false
  sqrt_neg : ∀ x, nan x = false → ¬ le L.zero x → nan (L.sqrt x) = true
  atan2_finite : ∀ y x, nan y = false → nan x = false → L.finite (L.atan2 y x) = true
  atan2_nan_right : ∀ y x, nan x = true → nan (L.atan2 y x) = true
  two_finite : L.finite L.two = true
  radius_finite : L.finite L.radius = true
  mul_nan_right : ∀ x y, nan y = true → nan (L.mul x y) = true

variable {L} (N : NanLaws L)

/-- **W2-temporal-4 (history, fixed by #2983 86dcd5a11).** Before the clamp, whenever the
rounded `a` landed above 1 (antipodal points: `a = 1 + 2⁻⁵²`), `distance` was NaN
(C: 20037518). Repro `bug_distance_antipodal_nan` (35 428 of 6 485 401 antipodal pairs
on a 0.1° grid); the mechanism is `#eval`'d in `Spatial.lean`. -/
theorem pre2983_distance_nan (p q : Point F32) (hn : N.nan (havA L p q) = false)
    (habove : ¬ N.le (havA L p q) L.one) : N.nan (pre2983_distance L p q) = true := by
  unfold pre2983_distance havFinish
  obtain ⟨hneg, hnn⟩ := N.one_sub_above _ hn habove
  exact N.mul_nan_right _ _ (N.mul_nan_right _ _ (N.atan2_nan_right _ _ (N.sqrt_neg _ hnn hneg)))

/-- **W2-temporal-4 fixed.** On main `distance` is finite (so never NaN) for every pair of
points whose haversine term is not NaN (i.e. finite coordinates), antipodal ones included. -/
theorem distance_finite (p q : Point F32) (hn : N.nan (havA L p q) = false) :
    L.finite (distance L p q) = true := by
  unfold distance havFinish
  obtain ⟨h0, h1⟩ := N.clamp_unit _ hn
  have hs1 := N.sqrt_nonneg _ h0
  have hs2 := N.sqrt_nonneg _ (N.one_sub_unit _ h0 h1)
  exact L.mul_finite _ _ N.radius_finite
    (L.mul_finite _ _ N.two_finite (N.atan2_finite _ _ hs1 hs2))

theorem distance_not_nan (p q : Point F32) (hn : N.nan (havA L p q) = false) :
    N.nan (distance L p q) = false := N.nan_of_finite _ (distance_finite N p q hn)

/-- The clamp only changes results that were wrong: whenever the rounded `a` is already in
`[0, 1]`, main and pre-#2983 agree. -/
theorem distance_eq_pre2983 (p q : Point F32) (h0 : N.le L.zero (havA L p q))
    (h1 : N.le (havA L p q) L.one) : distance L p q = pre2983_distance L p q := by
  unfold distance pre2983_distance; rw [N.clamp_id _ h0 h1]

end Antipodal

/-! ## spatial.rs -/

inductive SV (F F32 : Type) where
  | null | int (i : Int) | float (f : F) | str (s : String) | map (kvs : List (String × SV F F32))
  | list (vs : List (SV F F32)) | vec (v : List F32) | point (p : Point F32) | other (name : String)

def SV.name {F F32} : SV F F32 → String
  | .null => "Null" | .int _ => "Integer" | .float _ => "Float" | .str _ => "String"
  | .map _ => "Map" | .list _ => "List" | .vec _ => "VecF32" | .point _ => "Point"
  | .other n => n

inductive Res (α : Type) where
  | ok (a : α) | err (s : String) | unreachable

/-- One coordinate of `point_struct_pure` (spatial.rs:40-60). -/
def coord (field : String) : SV F F32 → Except String F32
  | .float f => .ok (L.ofF64 f)
  | .int i => .ok (L.ofI64 i)
  | .null => .error s!"point() requires '{field}' field"
  | other => .error s!"Type mismatch: '{field}' must be a number, got {other.name}"

/-- `point_struct_pure` (spatial.rs:38); `validate` is `Point::validate` (PROVEN elsewhere). -/
def pointStructPure (validate : Point F32 → Except String Unit) (args : List (SV F F32)) :
    Res (SV F F32) :=
  match coord L "latitude" (args.getD 0 .null) with
  | .error e => .err e
  | .ok lat => match coord L "longitude" (args.getD 1 .null) with
    | .error e => .err e
    | .ok lon => match validate (Point.new lat lon) with
      | .error e => .err e
      | .ok () => .ok (.point (Point.new lat lon))

/-- The map form `point` (spatial.rs:95): a missing key is the "requires" error, a present
non-number (including `null`) is the type-mismatch error. -/
def coordMap (m : List (String × SV F F32)) (field : String) : Except String F32 :=
  match m.lookup field with
  | none => .error s!"point() requires '{field}' field"
  | some (.float f) => .ok (L.ofF64 f)
  | some (.int i) => .ok (L.ofI64 i)
  | some other => .error s!"Type mismatch: '{field}' must be a number, got {other.name}"

def point (validate : Point F32 → Except String Unit) : List (SV F F32) → Res (SV F F32)
  | .map m :: _ =>
    match coordMap L m "latitude" with
    | .error e => .err e
    | .ok lat => match coordMap L m "longitude" with
      | .error e => .err e
      | .ok lon => match validate (Point.new lat lon) with
        | .error e => .err e
        | .ok () => .ok (.point (Point.new lat lon))
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- Map form = slot form whenever no coordinate key is bound to `null`. -/
theorem point_eq_struct (validate : Point F32 → Except String Unit) (m : List (String × SV F F32))
    (h1 : m.lookup "latitude" ≠ some .null) (h2 : m.lookup "longitude" ≠ some .null) :
    point L validate [.map m] = pointStructPure L validate
      [(m.lookup "latitude").getD .null, (m.lookup "longitude").getD .null] := by
  have e : ∀ f, m.lookup f ≠ some .null → coordMap L m f = coord L f ((m.lookup f).getD .null) := by
    intro f hf; unfold coordMap; split <;> rename_i heq <;> simp only [heq, Option.getD, coord] <;> simp_all
  simp only [point, pointStructPure, List.getD, List.getElem?_cons_zero, List.getElem?_cons_succ,
    Option.getD_some, e _ h1, e _ h2]

/-- …but `point({latitude: null, ..})` differs: the map form says "must be a number, got
Null", the slot form (used for constant map literals) "requires 'latitude' field". -/
theorem point_null_lat_msgs (validate : Point F32 → Except String Unit) (lon : SV F F32) :
    point L validate [.map [("latitude", .null), ("longitude", lon)]] =
      .err "Type mismatch: 'latitude' must be a number, got Null" ∧
    pointStructPure L validate [.null, lon] = .err "point() requires 'latitude' field" := by
  constructor <;> rfl

def isNum : SV F F32 → Bool
  | .int _ | .float _ => true
  | _ => false

/-- `get_numeric() as f32` on a number. -/
def numF32 : SV F F32 → F32
  | .int i => L.ofI64 i
  | .float f => L.ofF64 f
  | _ => L.ofI64 0

/-- `vecf32` (spatial.rs:74). -/
def vecf32 : List (SV F F32) → Res (SV F F32)
  | .list vs :: _ =>
    if vs.all isNum then .ok (.vec (vs.map (numF32 L)))
    else .err "vecf32 expects an array of numbers"
  | .null :: _ => .ok .null
  | _ => .unreachable

theorem vecf32_ok (vs : List (SV F F32)) (v : List F32) (h : vecf32 L [.list vs] = .ok (.vec v)) :
    v.length = vs.length ∧ vs.all isNum = true := by
  simp only [vecf32] at h; split at h
  · cases h; simp_all
  · cases h

/-- `distance` (spatial.rs:141): `Null` (here `none`) if either side is null, else
`Point::distance`. -/
def distanceV (args : List (SV F F32)) : Res (Option F) :=
  match args with
  | [.point p, .point q] => .ok (some (distance L p q))
  | [.null, _] => .ok none
  | [_, .null] => .ok none
  | _ => .unreachable

theorem distanceV_spec (a b : SV F F32) (ha : a = .null ∨ ∃ p, a = .point p)
    (hb : b = .null ∨ ∃ p, b = .point p) :
    distanceV L [a, b] = (match a, b with
      | .point p, .point q => .ok (some (distance L p q))
      | _, _ => .ok none) := by
  rcases ha with rfl | ⟨p, rfl⟩ <;> rcases hb with rfl | ⟨q, rfl⟩ <;> rfl

/-- simsimd (FFI) as partial functions; only law: defined on equal lengths. -/
structure Simd (F F32 : Type) where
  euclidean : List F32 → List F32 → Option F
  cosine : List F32 → List F32 → Option F
  euclidean_some : ∀ a b, a.length = b.length → (euclidean a b).isSome
  cosine_some : ∀ a b, a.length = b.length → (cosine a b).isSome

/-- `vec.euclideanDistance` / `vec.cosineDistance` (spatial.rs:157, :187); `dflt` is the
`unwrap_or` value (0.0 resp. 1.0). -/
def vecDist (k : List F32 → List F32 → Option F) (dflt : F) (args : List (SV F F32)) : Res (Option F) :=
  match args with
  | [.vec a, .vec b] =>
    if a.length ≠ b.length then
      .err s!"Vector dimension mismatch, expected {a.length} but got {b.length}"
    else .ok (some ((k a b).getD dflt))
  | [.null, _] => .ok none
  | [_, .null] => .ok none
  | _ => .unreachable

def vecEuclideanDistance (S : Simd F F32) := vecDist (S.euclidean) L.zero
def vecCosineDistance (S : Simd F F32) := vecDist (S.cosine) L.one

/-- The `unwrap_or` default is dead: on equal lengths the kernel's value is returned. -/
theorem vecDist_kernel (S : Simd F F32) (a b : List F32) (h : a.length = b.length) :
    (∃ x, S.euclidean a b = some x ∧ vecEuclideanDistance L S [.vec a, .vec b] = .ok (some x)) ∧
    (∃ x, S.cosine a b = some x ∧ vecCosineDistance L S [.vec a, .vec b] = .ok (some x)) := by
  have he := S.euclidean_some a b h; have hc := S.cosine_some a b h
  constructor
  · cases e : S.euclidean a b with
    | none => simp [e] at he
    | some x => exact ⟨x, rfl, by simp [vecEuclideanDistance, vecDist, h, e]⟩
  · cases e : S.cosine a b with
    | none => simp [e] at hc
    | some x => exact ⟨x, rfl, by simp [vecCosineDistance, vecDist, h, e]⟩

theorem vecDist_mismatch (k : List F32 → List F32 → Option F) (d : F) (a b : List F32)
    (h : a.length ≠ b.length) :
    vecDist k d [.vec a, .vec b] = .err s!"Vector dimension mismatch, expected {a.length} but got {b.length}" := by
  simp [vecDist, h]

/-- `register` (spatial.rs:67): five `cypher_fn!`s and one struct form. -/
def registered : List (String × Nat) :=
  [("vecf32", 1), ("point", 1), ("distance", 2), ("vec.euclideanDistance", 2), ("vec.cosineDistance", 2)]
def structSlots : List (String × List String) := [("point", ["latitude", "longitude"])]

theorem register_wf : (registered.map (·.1)).Nodup ∧
    (∀ p ∈ structSlots, p.1 ∈ registered.map (·.1) ∧ p.2.length = 2) := by decide

end FalkorTemporal.Spatial2
