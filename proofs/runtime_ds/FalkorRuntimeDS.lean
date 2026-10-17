/-
# FalkorDB runtime data structures, modelled and verified

See `REPORT.md` for the Lean ↔ Rust mapping and findings.

## Wave-5 additions
* `OrderMapExtra` / `OrderSetExtra`: constructors, accessors, `Hash`, `Index` (panic iff key/index
  absent), `IntoIterator` of `ordermap.rs` / `orderset.rs` — all rows PROVEN.
* `RowExtra`: the `RowView` trait contract (`rowView_lawful`), `with_capacity`, `from_raw`, `len`,
  `is_empty`, `has_bindings` (iff some slot bound), `to_owned_row`; `Cow::new`; the `OnceLock`
  string-pool global (`globalPool_spec`).
-/
import FalkorRuntimeDS.BitSet
import FalkorRuntimeDS.OrderSet
import FalkorRuntimeDS.OrderMap
import FalkorRuntimeDS.StringPool
import FalkorRuntimeDS.Row
import FalkorRuntimeDS.NarrowInt
import FalkorRuntimeDS.IdentifierLimits
import FalkorRuntimeDS.Cow
import FalkorRuntimeDS.OrderSetExtra
import FalkorRuntimeDS.OrderMapExtra
import FalkorRuntimeDS.RowExtra
