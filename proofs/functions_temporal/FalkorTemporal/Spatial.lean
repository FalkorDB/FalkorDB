/-!
# `Point::distance` (`graph/src/runtime/value.rs:144-176`)

`a = cos φ₁ cos φ₂ · sin²(Δλ/2) + sin²(Δφ/2)` is ≤ 1 over the reals, but in f64 it can
round to `1 + 2⁻⁵²` for antipodal points; before #2983 (86dcd5a11) `(1.0 - a).sqrt()` was
then NaN and `atan2(√a, NaN)` NaN (W2-temporal-4; 35 428 of 6 485 401 antipodal pairs on
a 0.1° grid, Rust test `lean_functions_temporal::bug_distance_antipodal_nan`). Main clamps
`a` to `[0, 1]`. Lean's `Float` is IEEE binary64 but opaque to the kernel, so the `#eval`s
below only show the mechanism; the proofs (`pre2983_distance_nan`, `distance_finite`) are in
`SpatialFns.lean` over an abstract float. Rust's `mul_add` (fused) is a plain `*`/`+` here.
-/
namespace FalkorTemporal.Spatial

def EARTH_RADIUS : Float := 6378140.0

/-- Pre-#2983 `Point::distance` with the coordinates already converted to radians as f64
(the Rust code rounds degrees→radians in f32 first; not modelled). History. -/
def pre2983_haversine (lat1 lon1 lat2 lon2 : Float) : Float :=
  let dlat := lat2 - lat1
  let dlon := lon2 - lon1
  let s1 := Float.sin (dlon / 2.0)
  let s2 := Float.sin (dlat / 2.0)
  let a := Float.cos lat1 * Float.cos lat2 * (s1 * s1) + s2 * s2
  let c := 2.0 * Float.atan2 (Float.sqrt a) (Float.sqrt (1.0 - a))
  EARTH_RADIUS * c

/-- Main (#2983): `a.clamp(0.0, 1.0)` (a NaN `a` stays NaN, as with `f64::clamp`). -/
def haversine (lat1 lon1 lat2 lon2 : Float) : Float :=
  let dlat := lat2 - lat1
  let dlon := lon2 - lon1
  let s1 := Float.sin (dlon / 2.0)
  let s2 := Float.sin (dlat / 2.0)
  let a0 := Float.cos lat1 * Float.cos lat2 * (s1 * s1) + s2 * s2
  let a := if a0 > 1.0 then 1.0 else if a0 < 0.0 then 0.0 else a0
  EARTH_RADIUS * (2.0 * Float.atan2 (Float.sqrt a) (Float.sqrt (1.0 - a)))

-- the mechanism: one ulp above 1 is enough
#eval Float.sqrt (1.0 - 1.0000000000000002)              -- NaN
#eval Float.atan2 1.0 (Float.sqrt (1.0 - 1.0000000000000002))  -- NaN
-- after the clamp: a = 1, c = 2·atan2(1, 0) = π, distance = π·R ≈ 20037518
#eval 6378140.0 * (2.0 * Float.atan2 (Float.sqrt 1.0) (Float.sqrt (1.0 - 1.0)))

end FalkorTemporal.Spatial
