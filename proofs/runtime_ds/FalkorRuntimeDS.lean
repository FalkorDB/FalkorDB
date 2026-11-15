/-
# FalkorDB runtime data structures, modelled and verified

See `REPORT.md` for the Lean ↔ Rust mapping and findings.

## Wave-5 additions
* `OrderMapExtra` / `OrderSetExtra`: constructors, accessors, `Hash`, `Index` (panic iff key/index
  absent), `IntoIterator` of `ordermap.rs` / `orderset.rs` — all rows PROVEN.
* `RowExtra`: the `RowView` trait contract (`rowView_lawful`), `with_capacity`, `from_raw`, `len`,
  `is_empty`, `has_bindings` (iff some slot bound), `to_owned_row`; `Cow::new`; the `OnceLock`
  string-pool global (`globalPool_spec`).

## Re-target to main 1d15bf713 (#2905)
* `OrderSet::from_vec` now drops later duplicates (first wins): `fromVecGo`/`fromVec`,
  `fromVec_nodup`, `containsS_removeS_fromVec`.
* `Cow` gained a debug-only `branched` flag shared by clones: debug model `DCow`/`dWrite`,
  `dbg_write_through_branched_source_panics`, `dbg_write_through_clone_panics`,
  `dbg_agrees_with_release`; new `Cow::clone` row. Release model unchanged
  (`release_old_write_leaks` stays a documented latent contract).
* `OrderMap::from_unique_keys` stays debug-checked only (documented); its counterexamples stand.
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
