/-!
# `Point::distance` (`graph/src/runtime/value.rs:121-141`)

`a = cos φ₁ cos φ₂ · sin²(Δλ/2) + sin²(Δφ/2)` is ≤ 1 over the reals, but in f64 it can
round to `1 + 2⁻⁵²` for antipodal points; `(1.0 - a).sqrt()` is then NaN and
`atan2(√a, NaN)` is NaN. Lean's `Float` is IEEE binary64 but opaque to the kernel, so this
is a MODELLED item: the `#eval`s below show the mechanism, and the Rust test
`lean_functions_temporal::bug_distance_antipodal_nan` shows it on the real code
(35 428 of 6 485 401 antipodal pairs on a 0.1° grid are NaN).
-/
namespace FalkorTemporal.Spatial

def EARTH_RADIUS : Float := 6378140.0

/-- `Point::distance` with the coordinates already converted to radians as f64
(the Rust code rounds degrees→radians in f32 first; not modelled). -/
def haversine (lat1 lon1 lat2 lon2 : Float) : Float :=
  let dlat := lat2 - lat1
  let dlon := lon2 - lon1
  let s1 := Float.sin (dlon / 2.0)
  let s2 := Float.sin (dlat / 2.0)
  let a := Float.cos lat1 * Float.cos lat2 * (s1 * s1) + s2 * s2
  let c := 2.0 * Float.atan2 (Float.sqrt a) (Float.sqrt (1.0 - a))
  EARTH_RADIUS * c

/-- The guarded form a fix would use: clamp `a` into `[0, 1]`. -/
def haversineClamped (lat1 lon1 lat2 lon2 : Float) : Float :=
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

end FalkorTemporal.Spatial
