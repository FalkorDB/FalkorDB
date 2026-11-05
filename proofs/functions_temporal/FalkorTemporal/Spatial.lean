/-!
# `Point::distance` (`graph/src/runtime/value.rs:153-176`)

`a = cos φ₁ cos φ₂ · sin²(Δλ/2) + sin²(Δφ/2)` is ≤ 1 over the reals, but in f64 it can
round to `1 + 2⁻⁵²` for antipodal points; before #2983 (86dcd5a11) `(1.0 - a).sqrt()` was
then NaN and `atan2(√a, NaN)` NaN (W2-temporal-4; 35 428 of 6 485 401 antipodal pairs on
a 0.1° grid, Rust test `lean_functions_temporal::bug_distance_antipodal_nan`). Main clamps
`a` to `[0, 1]`. Lean's `Float` is IEEE binary64 but opaque to the kernel, so the `#eval`s
below only show the mechanism; the proof (`distance_finite`) is in
`SpatialFns.lean` over an abstract float. The full operation-by-operation model of `Point::distance` (degrees → radians, `mul_add`,
the clamp) is `Spatial2.distance` in `SpatialFns.lean`; this file only checks the rounding mechanism.
-/
namespace FalkorTemporal.Spatial

/-- Rust's `EARTH_RADIUS` (value.rs:157). -/
def EARTH_RADIUS : Float := 6378140.0

-- The mechanism, checked at build time (`#guard` fails the build on a mismatch): one ulp above 1
-- is enough for `(1.0 - a).sqrt()` and the `atan2` over it to be NaN (pre-#2983) ...
#guard (Float.sqrt (1.0 - 1.0000000000000002)).isNaN
#guard (Float.atan2 1.0 (Float.sqrt (1.0 - 1.0000000000000002))).isNaN
-- ... and after main's clamp (`a.clamp(0.0, 1.0)`) a = 1, c = 2·atan2(1, 0) = π, distance = π·R.
#guard Float.abs (EARTH_RADIUS * (2.0 * Float.atan2 (Float.sqrt 1.0) (Float.sqrt (1.0 - 1.0)))
  - EARTH_RADIUS * 3.141592653589793) < 1.0e-6
-- `f64::clamp` keeps a NaN `a` NaN, so the clamp alone does not remove NaN inputs (`havA_finite` does).
#guard (if (0.0 / 0.0 : Float) > 1.0 then 1.0 else if (0.0 / 0.0 : Float) < 0.0 then 0.0 else (0.0 / 0.0 : Float)).isNaN

end FalkorTemporal.Spatial
