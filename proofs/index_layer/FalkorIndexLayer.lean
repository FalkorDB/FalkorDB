/-
# FalkorDB-rs index layer: Lean 4 model, proofs, and bug report

Scope: `graph/src/index/mod.rs`, `graph/src/index/indexer.rs`,
`graph/src/index/falkordb/` (the `cow_btree` there is already proven in
`proofs/cow_btree` and is not yet wired into the index; its spec is taken as
given), and `graph/src/index/redisearch/` (FFI bindings only).

There is no RediSearch *query string* builder for range indexes. Cypher
predicates go to RediSearch as LLAPI query-node trees (`build_query_node`),
so no characters need escaping in a query language. The only string
transformation is the TAG key encoding `tag_encode_lower`, used on both the
write and query sides. Full-text queries pass the user's string to
`RediSearch_IterateQuery` unchanged, on purpose. So injection through special
characters comes down to three questions. Can two strings share a key
(`tagEnc_injective`: no)? Can a key contain the separator, NUL, whitespace or
backslash (`tagEnc_printable`, `tagEnc_no_backslash`: no)? Does the encoding
preserve the lexicographic order that range queries need
(`tagEnc_not_monotone`: **no**, bug 7)?

Build: `lake build`. The project has no `sorry`, `admit`, `axiom` or
`native_decide`. The headline theorems depend only on `propext`,
`Quot.sound` and `Classical.choice`. It has 64 theorems.

## Lean ↔ Rust

| Lean (`Model.lean`) | Rust |
| --- | --- |
| `tagEnc`, `escaped`, `hexDigit` | `mod.rs:565` `tag_encode_lower` |
| `hexEncode`, `leBytes`, `nodeKey` | `mod.rs:581` `hex_encode_into`, `Document::new` `mod.rs:657` |
| `hexNibble`, `hexDecode`, `fromLe`, `decodeId` | `mod.rs:591` `hex_nibble`, `mod.rs:601` `decode_id` |
| `edgeKey` | `Document::new_edge` `mod.rs:684`, `decode_triple` `mod.rs:612`, `delete_edge_document` `mod.rs:1932` |
| `intLosesPrecision` | `mod.rs:1371` `int_loses_f64_precision` |
| `setRange`, `docOf` | `Document::set` `mod.rs:716` (Range field arm), field names `mod.rs:121` |
| `setVector` (`SetVector.lean`) | `Document::set` vector arm `mod.rs:730-744` (#3087 dimension guard) |
| `valueToNumeric` | `mod.rs:1357` |
| `buildNumRange` | `mod.rs:1376` `build_numeric_range_node` |
| `buildStrRange` | `mod.rs:1415` `build_string_range_node` |
| `buildEq`, `buildQ` | `mod.rs:1463` `build_query_node` |
| `indexHit` | `Index::query` `mod.rs:1677` (null node ⇒ empty iterator) |
| `expandIn`, `indexable`, `canUtilize`, `scanEmits` | `runtime/ops/node_by_index_scan.rs:216,252,249,309` |
| `holds`, `cyEqScalar`, `cyLt`, `cyLe` | the Cypher predicate replaced by `utilize_index.rs:324 build_op_query` |
| `Table.add/del`, `commit` | `add_document` `mod.rs:1893` (ADD_REPLACE), `delete_document` `mod.rs:1915`, `Indexer::commit` `indexer.rs:714` |
| `Slots.inc/dec/bump/countFor` | `mod.rs:2097/2115/1049/2137`, `indexer.rs:608-701` |
| `rsMatch`, `rsStore`, `inNum`, `inLex` | **axiomatised RediSearch spec** (below) |

Axiomatised boundary. `rsMatch`/`rsStore` state the documented LLAPI
semantics. `CreateNumericNode(min,max,incl)` is an IEEE range test.
`CreateTagTokenNode` is an exact token match on a case-sensitive TAG with
separator `\x01`. `CreateTagLexRangeNode` is a byte-lexicographic range over
tag keys. Union and intersection are or and and. `CreateEmptyNode` matches
nothing. An empty TAG value is not indexed (no INDEXEMPTY). These are written
as definitions, not Lean `axiom`s. Every live run agreed with them. f64 is
abstracted as `ENum` (an order embedding of the finite values into `Int`, plus
±inf and NaN).

## Proven (plain English)
Keys and encoding (`Proofs.lean`):
- `nib_digit`, `hex_roundtrip_byte`: hex nibble encode and decode are inverses.
- `tagDec_tagEnc`, `tagEnc_injective`: the TAG encoding is injective on byte strings.
- `tagEnc_printable`: every encoded byte is > 0x20, so there is no separator, NUL or whitespace.
- `tagEnc_no_backslash`: no `\` in the encoding. `tagEnc_nil_iff`: only the empty string encodes to empty.
- `hexDecode_hexEncode`, `fromLe_leBytesN`, `decodeId_nodeKey`, `nodeKey_injective`, `nodeKey_length`: node keys round-trip and are 16 chars.
- `decode_edgeKey`: the 48-char edge key round-trips `(src,dst,eid)`.
- `intLosesPrecision_sound`: a literal that passes the precision guard has |i| < 2^52, or is i64::MIN, and is exact in f64.

Query exactness (index hits = full scan plus the Cypher filter):
- `ENum.le_le_iff_eq`, `inNum_point`, `rsMatch_union`, `rsMatch_inter`: RS node algebra.
- `equal_exact`: `n.k = literal` is exact for stored Int, Float or non-empty String.
- `inList_exact`: `n.k IN [scalar literals]` is exact under the same conditions.
- `numRange_exact`: numeric `<,<=,>,>=` and two-sided ranges are exact for finite or NaN stored numbers.
- `strRange_exact` (`Strings.lean`): string ranges are exact when no string has a byte ≤ 0x20, `\` or `_`, stored strings are non-empty, and equal bounds are inclusive. `lexLt_irrefl`, `lexLt_asymm`, `lexLt_total`, `tagEnc_id` support it.

Maintenance and tickets (`Maintenance.lean`):
- `foldl_add_lookup`, `foldl_del_lookup`, `commit_lookup`: exactly what `Indexer::commit` leaves for each id.
- `commit_preserves_inv`: index-maintenance consistency. If removed ids have lost the label, ids that lost the label are removed, and ids with the label that were not rebuilt are unchanged, then after commit the RS table matches the committed graph exactly.
- `commit_both_removes`: an id in both the add and remove sets ends up with no document (a hazard, because removes run last).
- `inc_dec_cur`, `inc_dec_stale`, `ok_inc`, `ok_dec`, `ok_bump`, `bump_total`, `bump_fresh`, `dec_stale_keeps_current`, `inc_stale_keeps_current`: the ticket counters stay non-negative. Acquire and release cancel. Recreating an index starts it operational and conserves pending work. Stale workers never touch the new generation.

## Confirmed bugs

Each is a Lean counterexample in `Bugs.lean`, reproduced live with
`venv/bin/python proofs/index_layer/repro.py`. That script runs each query
without and with the index, on Rust (port 18300) and C (port 18301). The C
module is `bin/macos-arm64v8-release`. Output is shown as full scan → index
scan.

1. `bug_folded_temporal_constant` (`utilize_index.rs:631-647` plus `node_by_index_scan.rs:252`).
   - Query: `MATCH (n:L) WHERE n.v = date('2020-01-01')`, and also `(n:L {v: date(..)})`, `n.v > date(..)`, and edges.
   - Rust returns **every** node of the label: `['date']` → `['date','int']`. C: `['date']`.
   - Cause: the constant is folded to `Constant(Date)`, which `is_non_indexable_subexpr` treats as indexable, so the filter is dropped. The runtime then refuses the index and falls back to a label scan.
   - Fix: keep the filter for any non-scalar `Constant`, or treat an index-refused fallback as "filter required".
2. `bug_point_equality` (`mod.rs:1672`; `node_by_index_scan.rs:257` says Point is indexable).
   - Query: `n.v = point({latitude:1.0, longitude:2.0})`.
   - Rust: `['p']` → `[]`. C: `['p']`.
   - Fix: add a Point `Equal` arm (a geo node with radius 0 plus the kept filter), or mark Point not indexable for `Equal`.
3. `bug_in_list_drops_item` (`node_by_index_scan.rs:222`).
   - Query: `n.v IN [date('2020-01-01'), 2]`.
   - Rust: `['date','int']` → `['int']`. C: `['date','int']`.
   - Fix: if any item is dropped, fall back to a label scan (the filter is already kept).
4. `bug_multilabel_and` / `bug_multilabel_or` (`utilize_index.rs:503-531`, `mod.rs:1559,1576`).
   - Setup: `(n:A:B)`, with A indexing x and B indexing y.
   - `WHERE n.x = 1 AND n.y = 2` → `[]`. `WHERE n.y = 2 OR n.x = 1` loses rows. `{x:1, y:2}` → `[]`.
   - C returns the rows.
   - Fix: merge only conjuncts and disjuncts whose label matches the first one. Leave the rest in the post-filter, and bail on OR.
5. `bug_temporal_in_numeric_range` (`mod.rs:797`).
   - Query: `n.v > 0` matches a stored date: `['int']` → `['date','int']`. C: `['int']`.
   - Fix: index temporals under a separate sub-field (C uses per-type field names).
6. `bug_exclusive_equal_bounds` (`mod.rs:1430`).
   - Query: `n.v > 'a' AND n.v < 'a'`.
   - Rust: `[]` → `['a']`. C: `[]`.
   - Fix: use the exact-token shortcut only when both include flags are set; otherwise return an empty node.
7. `bug_string_range_order` / `_gt`, `tagEnc_not_monotone` (`mod.rs:565` used by `mod.rs:1444`).
   - The TAG encoding does not preserve order, so string ranges over strings with a byte ≤ 0x20, `\` or `_` are wrong in both directions.
   - `n.v < 'JohnA'` with `'John Smith'` stored: `['js']` → `[]`. `n.v > 'a!'` with `'a b'` stored: `[]` → `['ab']`.
   - C is also wrong, differently (it does not encode). Batch runs show C's string range index is much worse.
   - Fix: use an order-preserving escape. For example, map bytes b ≤ 0x20 to `0x21 0x21+b`, shift `!` itself, and so on. Or keep the filter for string ranges.
8. `bug_bool_eq_int` (`mod.rs:760,1361`).
   - Query: `n.v = 1` also returns `v = true`, and `n.v = true` returns `v = 1`: `['int']` → `['bool','int']`.
   - C has the same bug.
9. `bug_open_bound_excludes_inf` (`mod.rs:1389,1396`).
   - An absent bound becomes ±inf with an exclusive flag, so `n.v > 0`, `n.v >= 1.0/0.0` and `n.v <= -1.0/0.0` miss stored ±inf: `['-inf','inf']` → `[]`.
   - C has the same bug.
   - Fix: set the include flag to true for an absent bound.
10. `bug_empty_string_not_indexed`: `n.v = ''` → `[]`, and `n.v < 'a'` misses `''`. C has the same bug. Fix: an INDEXEMPTY equivalent, or a sentinel encoding for the empty string.
11. Edge index scan with an undirected pattern (`runtime/ops/edge_by_index_scan.rs`; outside the model, found by the differential run).
    - Query: `MATCH ()-[r:R]-() WHERE r.v >= 1` returns each edge once instead of twice: `['e','e']` → `['e']`.
    - C has the same bug.
12. Geo radius near the poles (RediSearch geohash is limited to |lat| ≤ 85.05).
    - Points at latitude 89.99: the full scan finds 21, the index finds 0.
    - C has the same bug. This is outside the axiomatised boundary; the fix would be to keep the distance filter and fall back beyond |lat| > 85.05.

C-only issue, not a Rust bug: C crashes (SIGSEGV in `RediSearch_ResultsIteratorNext`) on
`MATCH (n:L) WHERE distance(n.v, point({latitude:1.0, longitude:2.0})) <= 0 RETURN n.i`
when a range index exists. Rust returns the right result.

## Differential coverage run

`proofs/index_layer/diff.py` ran 83 predicate shapes over 38 stored values:
numbers, ±0, ±inf, NaN, 2^53+1, booleans, strings (empty, whitespace, `\`, `_`,
unicode, emoji), lists, temporals and points. `diff2.py` ran 22 maintenance
scenarios. Every Rust mismatch between the index and the full scan falls into
one of bugs 1-12. Precision guard, NaN and -0.0 equality, and unicode or emoji
equality were all exact. All maintenance scenarios were consistent: SET and
REMOVE of a property, `SET n = {}`, `+=`, type changes between scalar and
list, DELETE, id reuse, REMOVE then SET of a label in the same query, MERGE,
rollback on error, and edge SET, DELETE and DETACH DELETE.

## Suspected, unconfirmed
- `Indexer::commit` applies removes after adds. An id in both sets loses its document (`commit_both_removes`). No Cypher sequence tried produces both (pending de-duplicates REMOVE then SET of a label), but nothing enforces it.
- Error suppression: `1 IN n.v` over a non-list `n.v` raises a type error in a full scan but returns `[]` or rows through the index. C behaves the same.

## Gaps
- Stored integers of 2^53 or more (f64 rounding on the write side) are not modelled.
- Vector (HNSW) and full-text semantics are left to RediSearch.
- Refcounts, `ArcSwap` and locks are not modelled.
- `create_rs_index` and `register_fields` options are not modelled.
- The graph-side bookkeeping (`graph.rs` `index_add_docs` / `remove_docs`) is not modelled. It is assumed to meet the side conditions of `commit_preserves_inv`, which the live scenarios checked.
- Geo is in the model only as `RsV.geo`; there are no distance theorems.

## Wave 4 additions (origin/main 3fec7d7c9)

New modules, all `sorry`-free:
- `Meta`, `IndexM`: `Field` and the `Index` metadata (field map as an association list,
  `field_order` invariant `Inv` preserved by insert/remove/add/retain), RS spec lifecycle
  (`create_rs_index`, `register_fields`, `recreate_index`) with FFI results as parameters.
- `IndexerM`: `Indexer` (create/drop/remove/lookups/vector metadata/index_info/progress/
  cancel/graph slot/recreate). `createIndex_validates`, `createIndex_ok`, `dropIndex_spec`
  (a well-formed entry never keeps an empty field list), `indexInfo_spec` (sorted permutation).
- `Iter`: result iterators (`drain_next`), RS refcounting (`RC.not_freed`: no use-after-free
  while an iterator holds a clone; `into_raw_needed`), `Document` single free, and the bridge
  `query_eq_scan_filter` / `queryE_eq_scan_filter` (node and edge index scan = label/type scan +
  filter, given the RediSearch LLAPI contract `RunSpec`), `fulltextQ_exact`, `vectorQ_sound`
  (KNN: at most k stored vector docs), `commitEdge_lookup`.
- `Leaf*`: the COW B+-tree page formats byte for byte (AoS, compact, compact-indexed):
  build/read round trips, `splice_insert` (both arms) / `splice_remove` / `merge` /
  `block_copy_merge` all equal the list operation, `from_pairs` round trip for every format the
  size heuristic picks, `merge_batch` fast paths = slow path = `merge_sorted`, plus
  `partition_point`, gallop, `pack_branches`/`build_root`/`from_sorted`/`insert_batch`.

NEW CONFIRMED BUG 13 (`index/mod.rs:1398,1423,1471,1483,1541,1602`, `queryField` /
`IndexerM.bug_range_after_fulltext`): `build_query_node` targets
`self.fields.get(key).and_then(|f| f.first())` — the attribute's *first* field whatever its
type. After `CREATE FULLTEXT INDEX FOR (n:L) ON (n.s)` then `CREATE INDEX FOR (n:L) ON (n.s)`,
every range query on `n.s` is built against the fulltext field `s` instead of `range:s`:
`MATCH (n:L) WHERE n.s = 'foo'` Rust `[]`, C `[2]`; `n.s > 'c'` Rust `[]`, C `[1],[2]`;
`(n:L {s:'bar'})` Rust `[]`, C `[3]`; `n.s IN ['foo','bar']` Rust `[]`, C `[2],[3]`
(live, ports 18850 Rust / 18851 C). Fix: pick the field with `ty == IndexType::Range`.
Suspected, unconfirmed: `create_rs_index` returns `Err` on a NUL in a stopword/language/label
after `RediSearch_CreateIndexOptions` without `RediSearch_FreeIndexOptions` (leak only).

## Re-target to 49f698d22 (#3087)
`Document::set` now adds a `VecF32` only if
`vector_options.is_some_and(|o| o.dimension != 0 && o.dimension == vec.len())`
(`mod.rs:730-744`); every later line of `mod.rs` moved by +12 (all citations updated).
`SetVector.lean`: `setVector_sound` (key property: for any options and value, a vector add
carries the field's own non-zero dimension), `setVector_good`, `setVector_wrong_dim`,
`setVector_noparams`, `setVector_has_params` / `setVector_none_of_noparams` (link to
`registerOne`: an add only ever targets a field registered *with* params), `setVector_sub_pre`.
This discharges, for vector fields, the `ht` premise of `query_eq_scan_filter` (the RS table
holds `docOf` of every entity: a vector can no longer make RediSearch reject the document).
HISTORICAL (fe619ac5f): `pre3087_dim_mismatch`, `pre3087_noparams` — W3-conc-3 / #3075 and the
W5-idx-2 no-options shape; fixed by #3087.
-/
import FalkorIndexLayer.Model
import FalkorIndexLayer.Proofs
import FalkorIndexLayer.Strings
import FalkorIndexLayer.Maintenance
import FalkorIndexLayer.Bugs
import FalkorIndexLayer.Meta
import FalkorIndexLayer.IndexM
import FalkorIndexLayer.SetVector
import FalkorIndexLayer.IndexerM
import FalkorIndexLayer.Iter
import FalkorIndexLayer.LeafBytes
import FalkorIndexLayer.LeafMerge
import FalkorIndexLayer.LeafAos
import FalkorIndexLayer.LeafCompact
import FalkorIndexLayer.LeafIndexed
import FalkorIndexLayer.LeafIndexedOps
import FalkorIndexLayer.LeafIndexedMerge
import FalkorIndexLayer.LeafDispatch
import FalkorIndexLayer.LeafBlockCopy
import FalkorIndexLayer.LeafBlockCopy2
import FalkorIndexLayer.LeafOps
import FalkorIndexLayer.LeafTree
