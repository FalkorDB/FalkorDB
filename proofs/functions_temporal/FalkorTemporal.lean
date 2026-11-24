import FalkorTemporal.Calendar
import FalkorTemporal.EraAll
import FalkorTemporal.CalendarLaws
import FalkorTemporal.Duration
import FalkorTemporal.Constructors
import FalkorTemporal.VectorExpr
import FalkorTemporal.Spatial
import FalkorTemporal.Fns
import FalkorTemporal.Parse
import FalkorTemporal.Pure
import FalkorTemporal.PadDefs
import FalkorTemporal.PadAux
import FalkorTemporal.Components
import FalkorTemporal.DurationComponent
import FalkorTemporal.SpatialFns
import FalkorTemporal.VecDistance
/-
# functions_temporal

Temporal, spatial, conversion and columnar-expression functions: `functions/temporal.rs`,
the temporal half of `value.rs`, `spatial.rs`, `vec_distance.rs`, `vector_expr.rs`,
`conversion.rs`. The 400-year Gregorian era is checked by the kernel (`decide +kernel`, `Era*.lean`).
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
