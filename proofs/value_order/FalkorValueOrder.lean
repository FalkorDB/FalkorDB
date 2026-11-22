import FalkorValueOrder.Glue
import FalkorValueOrder.Json
import FalkorValueOrder.Model
import FalkorValueOrder.SortCmp
import FalkorValueOrder.Main
/-
# Value order: what `compare_value` promises, and where it breaks

A model of FalkorDB-rs's value comparison (`Value::compare_value`, `sort_cmp`), equality, hashing
and the comparators built on them (ORDER BY, DISTINCT, grouping, `=`/`<>`) in
`graph/src/runtime/value.rs`, plus the value.rs glue and JSON rendering.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
