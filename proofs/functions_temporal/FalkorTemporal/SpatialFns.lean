import FalkorTemporal.Pure
/-!
# Spatial and vector functions (functions/spatial.rs:38-193), `Point::new` / `Point::distance`
(value.rs:142-176)

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
  /-- Magnitude bounds, the only way a product or sum is known not to overflow (finite times
  finite is NOT finite in general, so no such general law is assumed). `bounded1 x` is
  `|x| ≤ 1`; `finite32` is "an `f32` that is neither NaN nor infinite". Each field below is a
  binary64 fact: `|sin|, |cos| ≤ 1`; products and squares of numbers in `[-1, 1]` stay there;
  `fma` of three numbers in `[-1, 1]` is at most 2; `f32::to_radians` of a finite `f32` is
  below 6.0e36 so the f64 conversion and the difference of two of them (`dlat`, `dlon`) are
  finite; halving a finite number is finite. -/
  bounded1 : F → Bool
  finite32 : F32 → Bool
  bounded_finite : ∀ x, bounded1 x = true → finite x = true
  cos_bounded : ∀ x, finite x = true → bounded1 (cos x) = true
  sin_bounded : ∀ x, finite x = true → bounded1 (sin x) = true
  mul_bounded : ∀ x y, bounded1 x = true → bounded1 y = true → bounded1 (mul x y) = true
  sq_bounded : ∀ x, bounded1 x = true → bounded1 (sq x) = true
  mulAdd_bounded : ∀ a b c, bounded1 a = true → bounded1 b = true → bounded1 c = true →
    finite (mulAdd a b c) = true
  toRad_finite : ∀ x, finite32 x = true → finite (toRadF64 x) = true
  toRad_sub_finite : ∀ x y, finite32 x = true → finite32 y = true →
    finite (sub (toRadF64 x) (toRadF64 y)) = true
  div_two_finite : ∀ x, finite x = true → finite (div x two) = true

variable {F F32 : Type} (L : Flt F F32)

structure Point (F32 : Type) where
  latitude : F32
  longitude : F32

/-- `Point::new` (value.rs:142). -/
def Point.new (latitude longitude : F32) : Point F32 := { latitude, longitude }

theorem Point.new_fields (a b : F32) : (Point.new a b).latitude = a ∧ (Point.new a b).longitude = b :=
  ⟨rfl, rfl⟩

/-- The haversine term `a = cos φ₁ cos φ₂ · sin²(Δλ/2) + sin²(Δφ/2)` of `Point::distance`
(value.rs:167-171), before the clamp. -/
def havA (p q : Point F32) : F :=
  let lat1 := L.toRadF64 p.latitude
  let lon1 := L.toRadF64 p.longitude
  let lat2 := L.toRadF64 q.latitude
  let lon2 := L.toRadF64 q.longitude
  let dlat := L.sub lat2 lat1
  let dlon := L.sub lon2 lon1
  L.mulAdd (L.mul (L.cos lat1) (L.cos lat2)) (L.sq (L.sin (L.div dlon L.two)))
    (L.sq (L.sin (L.div dlat L.two)))

/-- `c = 2·atan2(√a, √(1 − a))`, times `EARTH_RADIUS` (value.rs:172-175). -/
def havFinish (a : F) : F :=
  L.mul L.radius (L.mul L.two (L.atan2 (L.sqrt a) (L.sqrt (L.sub L.one a))))

/-- `Point::distance` (value.rs:153), haversine, operation by operation; since #2983
(86dcd5a11) `a` is clamped to `[0, 1]`. -/
def distance (p q : Point F32) : F := havFinish L (L.clamp01 (havA L p q))

/-- The distance from a (finite) point to itself is exactly 0. -/
theorem distance_self (p : Point F32) (h1 : L.finite (L.toRadF64 p.latitude) = true)
    (h2 : L.finite (L.toRadF64 p.longitude) = true) : distance L p p = L.zero := by
  have hc := L.bounded_finite _ (L.mul_bounded _ _ (L.cos_bounded _ h1) (L.cos_bounded _ h1))
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
`atan2_nan_right`: `atan2` of a NaN operand is NaN; `scaled_atan2_finite`: `atan2` of non-NaN
operands is in `[−π, π]`, so `R · 2 · atan2` is finite (about 4.0e7 at most);
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
  /-- `atan2` of non-NaN operands lies in `[−π, π]`, so `EARTH_RADIUS · (2 · atan2 ..)` is at
  most about 4.0e7, far below `f64::MAX`: the product cannot overflow. -/
  scaled_atan2_finite : ∀ y x, nan y = false → nan x = false →
    L.finite (L.mul L.radius (L.mul L.two (L.atan2 y x))) = true
  atan2_nan_right : ∀ y x, nan x = true → nan (L.atan2 y x) = true
  mul_nan_right : ∀ x y, nan y = true → nan (L.mul x y) = true

variable {L} (N : NanLaws L)

/-- **W2-temporal-4 fixed.** On main `distance` is finite (so never NaN) for every pair of
points whose haversine term is not NaN, antipodal ones included. (That the term is non-NaN for
finite coordinates is not derived here: it is the hypothesis `hn`.) -/
theorem distance_finite (p q : Point F32) (hn : N.nan (havA L p q) = false) :
    L.finite (distance L p q) = true := by
  unfold distance havFinish
  obtain ⟨h0, h1⟩ := N.clamp_unit _ hn
  have hs1 := N.sqrt_nonneg _ h0
  have hs2 := N.sqrt_nonneg _ (N.one_sub_unit _ h0 h1)
  exact N.scaled_atan2_finite _ _ hs1 hs2

theorem distance_not_nan (p q : Point F32) (hn : N.nan (havA L p q) = false) :
    N.nan (distance L p q) = false := N.nan_of_finite _ (distance_finite N p q hn)

/-- The haversine term of two points with finite (`f32`) coordinates is finite, hence not NaN:
`cos`/`sin` are bounded by 1, so the fused `a = cos φ₁ cos φ₂ · sin²(Δλ/2) + sin²(Δφ/2)` is at
most 2 and cannot overflow. Discharges the `hn` hypothesis of `distance_finite`. -/
theorem havA_finite (p q : Point F32)
    (h1 : L.finite32 p.latitude = true) (h2 : L.finite32 p.longitude = true)
    (h3 : L.finite32 q.latitude = true) (h4 : L.finite32 q.longitude = true) :
    L.finite (havA L p q) = true := by
  unfold havA
  have hc1 := L.cos_bounded _ (L.toRad_finite _ h1)
  have hc2 := L.cos_bounded _ (L.toRad_finite _ h3)
  have hs1 := L.sq_bounded _ (L.sin_bounded _ (L.div_two_finite _ (L.toRad_sub_finite _ _ h4 h2)))
  have hs2 := L.sq_bounded _ (L.sin_bounded _ (L.div_two_finite _ (L.toRad_sub_finite _ _ h3 h1)))
  exact L.mulAdd_bounded _ _ _ (L.mul_bounded _ _ hc1 hc2) hs1 hs2

/-- **#2983, unconditional on the haversine term:** for any two points with finite coordinates
(every point `point()` builds: NaN is refused and the range checked) `distance` is finite, so
never NaN, antipodal pairs included. Only the binary64 laws of `NanLaws` / `Flt` remain. -/
theorem distance_finite_coords (N : NanLaws L) (p q : Point F32)
    (h1 : L.finite32 p.latitude = true) (h2 : L.finite32 p.longitude = true)
    (h3 : L.finite32 q.latitude = true) (h4 : L.finite32 q.longitude = true) :
    L.finite (distance L p q) = true :=
  distance_finite N p q (N.nan_of_finite _ (havA_finite (L := L) p q h1 h2 h3 h4))

end Antipodal

/-! ## spatial.rs -/

inductive SV (F F32 : Type) where
  | null | int (i : Int) | float (f : F) | str (s : String) | map (kvs : List (String × SV F F32))
  | list (vs : List (SV F F32)) | vec (v : List F32) | point (p : Point F32) | other (name : String)

/-- `Value::name` (value.rs:1395) on the variants spatial.rs inspects; `Vectorf32` since
#2954 (b6567cb97), `VecF32` before. -/
def SV.name {F F32} : SV F F32 → String
  | .null => "Null" | .int _ => "Integer" | .float _ => "Float" | .str _ => "String"
  | .map _ => "Map" | .list _ => "List" | .vec _ => "Vectorf32" | .point _ => "Point"
  | .other n => n

inductive Res (α : Type) where
  | ok (a : α) | err (s : String) | unreachable

/-- The f64 facts `point_from_map_entries` (spatial.rs:38) needs: `numeric` of an `Int` is
`i as f64`, a NaN test, and the two closed range tests `-90.0..=90.0`, `-180.0..=180.0`
(false on NaN, as in Rust). They are separate from `Flt` so that structure stays as it was. -/
structure PtRng (F : Type) where
  ofInt : Int → F
  isNan : F → Bool
  inLat : F → Bool
  inLon : F → Bool
  lat_not_nan : ∀ x, isNan x = true → inLat x = false
  lon_not_nan : ∀ x, isNan x = true → inLon x = false

variable {R : PtRng F}

/-- `numeric` (spatial.rs:74): Float and Int are numbers, anything else (`null`
included) is not. -/
def numeric (R : PtRng F) : SV F F32 → Option F
  | .float f => some f
  | .int i => some (R.ofInt i)
  | _ => none

/-- `point_from_map_entries` (spatial.rs:38-72, since #3301): C's checks and messages.
`keyCount` is the map's length; `lat`/`lon` are `map.get_str(..)` (`none` = key absent,
`some .null` = key set to `null`). The range tests run on the `f64`s; `as f32` happens only
in the result. -/
def pointFromMapEntries (L : Flt F F32) (R : PtRng F) (keyCount : Nat)
    (lat lon : Option (SV F F32)) : Res (SV F F32) :=
  if keyCount ≠ 2 then .err "A point map should have 2 elements, latitude and longitude"
  else match lat with
  | none => .err "Did not find 'latitude' value in point map"
  | some lat => match lon with
  | none => .err "Did not find 'longitude' value in point map"
  | some lon => match numeric R lat, numeric R lon with
  | some la, some lo =>
    if R.isNan la then .err "latitude must be a finite number, got NaN"
    else if R.isNan lo then .err "longitude must be a finite number, got NaN"
    else if R.inLat la && R.inLon lo then .ok (.point (Point.new (L.ofF64 la) (L.ofF64 lo)))
    else if R.inLat la then .err "longitude should be within the -180 to 180 range"
    else .err "latitude should be within the -90 to 90 range"
  | _, _ => .err "'latitude' and 'longitude' values in point map were not both valid numerics"

/-- The `point` fn (spatial.rs:110): a map goes to `pointFromMapEntries`, `null` to `null`. -/
def point (L : Flt F F32) (R : PtRng F) : List (SV F F32) → Res (SV F F32)
  | .map m :: _ => pointFromMapEntries L R m.length (m.lookup "latitude") (m.lookup "longitude")
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- Any map without exactly two keys is refused, whatever they are. -/
theorem point_key_count (m : List (String × SV F F32)) (h : m.length ≠ 2) :
    point L R [.map m] = .err "A point map should have 2 elements, latitude and longitude" := by
  simp [point, pointFromMapEntries, h]

/-- A missing key is "Did not find", a key set to `null` falls to the numerics message. -/
theorem point_missing_lat (lon : SV F F32) :
    pointFromMapEntries L R 2 none (some lon) = .err "Did not find 'latitude' value in point map" := rfl

theorem point_null_not_numeric (lon : SV F F32) :
    pointFromMapEntries L R 2 (some .null) (some lon) =
      .err "'latitude' and 'longitude' values in point map were not both valid numerics" := by
  simp [pointFromMapEntries, numeric]

/-- A NaN coordinate is refused (latitude first), as the real Rust does; C lets it through. -/
theorem point_nan_refused (lat lon : SV F F32) (la lo : F) (ha : numeric R lat = some la)
    (hlo : numeric R lon = some lo) (hn : R.isNan la = true) :
    pointFromMapEntries L R 2 (some lat) (some lon) = .err "latitude must be a finite number, got NaN" := by
  simp [pointFromMapEntries, ha, hlo, hn]

theorem point_nan_lon_refused (lat lon : SV F F32) (la lo : F) (ha : numeric R lat = some la)
    (hlo : numeric R lon = some lo) (hn : R.isNan la = false) (hn' : R.isNan lo = true) :
    pointFromMapEntries L R 2 (some lat) (some lon) = .err "longitude must be a finite number, got NaN" := by
  simp [pointFromMapEntries, ha, hlo, hn, hn']

/-- Success exactly when both are numbers, not NaN, and in range **as f64** — the check
precedes the narrowing, so a value just past a bound is never rounded onto it. The point holds
the narrowed values. -/
theorem point_ok_iff (lat lon : SV F F32) (p : SV F F32) :
    pointFromMapEntries L R 2 (some lat) (some lon) = .ok p ↔
      ∃ la lo, numeric R lat = some la ∧ numeric R lon = some lo ∧ R.inLat la = true ∧
        R.inLon lo = true ∧ p = .point (Point.new (L.ofF64 la) (L.ofF64 lo)) := by
  unfold pointFromMapEntries
  rw [if_neg (by decide)]
  rcases ha : numeric R lat with _ | la
  · simp [ha]
  rcases hb : numeric R lon with _ | lo
  · simp [ha, hb]
  by_cases h1 : R.isNan la = true
  · simp [ha, hb, h1, R.lat_not_nan _ h1]
  · by_cases h2 : R.isNan lo = true
    · simp [ha, hb, h1, h2, R.lon_not_nan _ h2]
    · by_cases h3 : R.inLat la = true <;> by_cases h4 : R.inLon lo = true <;> simp [ha, hb, h1, h2, h3, h4]
      all_goals (try (intro h; cases h))
      all_goals (try exact eq_comm)

/-- Neither coordinate is NaN: a broken bound yields the message for the side that breaks it
(longitude when latitude is in range). -/
theorem point_range_msgs (lat lon : SV F F32) (la lo : F) (ha : numeric R lat = some la)
    (hlo : numeric R lon = some lo) (h1 : R.isNan la = false) (h2 : R.isNan lo = false) :
    (R.inLat la = true → R.inLon lo = false →
      pointFromMapEntries L R 2 (some lat) (some lon) = .err "longitude should be within the -180 to 180 range") ∧
    (R.inLat la = false →
      pointFromMapEntries L R 2 (some lat) (some lon) = .err "latitude should be within the -90 to 90 range") := by
  constructor
  · intro a b; simp [pointFromMapEntries, ha, hlo, h1, h2, a, b]
  · intro a; simp [pointFromMapEntries, ha, hlo, h1, h2, a]

def isNum : SV F F32 → Bool
  | .int _ | .float _ => true
  | _ => false

/-- `get_numeric() as f32` on a number. -/
def numF32 : SV F F32 → F32
  | .int i => L.ofI64 i
  | .float f => L.ofF64 f
  | _ => L.ofI64 0

/-- `vecf32` (spatial.rs:89). -/
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

/-- `distance` (spatial.rs:130): `Null` (here `none`) if either side is null, else
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

/-- `vec.euclideanDistance` / `vec.cosineDistance` (spatial.rs:146, :176); `dflt` is the
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

/-- `register` (spatial.rs:82): five `cypher_fn!`s. -/
def registered : List (String × Nat) :=
  [("vecf32", 1), ("point", 1), ("distance", 2), ("vec.euclideanDistance", 2), ("vec.cosineDistance", 2)]
/-- `register` (spatial.rs:82): since #3301 `point` has no struct form any more
(`set_struct_fn("point", ..)` is gone). -/
theorem register_wf : (registered.map (·.1)).Nodup := by decide

end FalkorTemporal.Spatial2
