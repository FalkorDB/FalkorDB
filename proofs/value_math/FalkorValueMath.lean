import FalkorValueMath.AggScalar
import FalkorValueMath.Registry
import FalkorValueMath.Int64
import FalkorValueMath.Value
import FalkorValueMath.CArith
import FalkorValueMath.TypeCheck
import FalkorValueMath.MathFns
import FalkorValueMath.Conversion
import FalkorValueMath.SelfCompare
import FalkorValueMath.Misc
import FalkorValueMath.Trig
import FalkorValueMath.Procedures
import FalkorValueMath.ConvReg
import FalkorValueMath.FloatToString
import FalkorValueMath.IndexDdl
import FalkorValueMath.Nesting
/-!
# Value arithmetic, math/conversion functions and function type checking (FalkorDB-rs)

Lean 4 (core only) models of `graph/src/runtime/value.rs` (arithmetic, formatting, memory accounting,
attribute access, self-comparison, v19 codec), `functions/math.rs`, `functions/conversion.rs` and
`functions/mod.rs` (arity, argument type checking, registry), plus the procedures and index DDL.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
