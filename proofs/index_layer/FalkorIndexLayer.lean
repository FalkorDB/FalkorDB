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
| `tagEnc`, `escaped`, `hexDigit` | `mod.rs:566` `tag_encode_lower` |
| `hexEncode`, `leBytes`, `nodeKey` | `mod.rs:582` `hex_encode_into`, `Document::new` `mod.rs:658` |
| `hexNibble`, `hexDecode`, `fromLe`, `decodeId` | `mod.rs:592` `hex_nibble`, `mod.rs:602` `decode_id` |
| `edgeKey` | `Document::new_edge` `mod.rs:685`, `decode_triple` `mod.rs:613`, `delete_edge_document` `mod.rs:1957` |
| `intLosesPrecision` | `mod.rs:1382` `int_loses_f64_precision` |
| `setRange`, `docOf` | `Document::set` `mod.rs:721` (Range field arm), field names `mod.rs:122` |
| `setVector` (`SetVector.lean`) | `Document::set` vector arm `mod.rs:736-748` (dimension guard) |
| `valueToNumeric` | `mod.rs:1368` |
| `buildNumRange` | `mod.rs:1387` `build_numeric_range_node` |
| `buildStrRange` | `mod.rs:1430` `build_string_range_node` |
| `buildEq`, `buildQ` | `mod.rs:1488` `build_query_node` |
| `indexHit` | `Index::query` `mod.rs:1702` (null node ⇒ empty iterator) |
| `expandIn`, `indexable`, `canUtilize`, `scanEmits` | `runtime/ops/node_by_index_scan.rs:216,256,253,302` |
| `holds`, `cyEqScalar`, `cyLt`, `cyLe` | the Cypher predicate replaced by `utilize_index.rs:315 build_op_query` |
| `Table.add/del`, `commit` | `add_document` `mod.rs:1918` (ADD_REPLACE), `delete_document` `mod.rs:1940`, `Indexer::commit` `indexer.rs:749` |
| `Slots.inc/dec/bump/countFor` | `mod.rs:2122/2140/1053/2162`, `indexer.rs:661-737` |
| `TextOpts.phoneticEnabled`, `PHONETIC_OFF` (`Meta.lean`) | `TextIndexOptions::phonetic_enabled` `text_index_options.rs:48`, `PHONETIC_OFF` `:41`; used by `register_fields` `mod.rs:1260` |
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
- `numRange_exact`: numeric `<,<=,>,>=` and two-sided ranges are exact for every stored number, ±inf included (`numRange_exact_fin`: the finite/NaN form).
- `strRange_exact` (`Strings.lean`): string ranges are exact when no string has a byte ≤ 0x20, `\` or `_` and stored strings are non-empty (equal bounds need no premise). `lexLt_irrefl`, `lexLt_asymm`, `lexLt_total`, `tagEnc_id` support it.

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

1. `bug_folded_temporal_constant` (`utilize_index.rs:895-912` (`is_non_indexable_subexpr`) plus `node_by_index_scan.rs:253`).
   - Query: `MATCH (n:L) WHERE n.v = date('2020-01-01')`, and also `(n:L {v: date(..)})`, `n.v > date(..)`, and edges.
   - Rust returns **every** node of the label: `['date']` → `['date','int']`. C: `['date']`.
   - Cause: the constant is folded to `Constant(Date)`, which `is_non_indexable_subexpr` treats as indexable, so the filter is dropped. The runtime then refuses the index and falls back to a label scan.
   - Fix: keep the filter for any non-scalar `Constant`, or treat an index-refused fallback as "filter required".
2. `bug_point_equality` (`mod.rs:1697`; `node_by_index_scan.rs:256` says Point is indexable).
   - Query: `n.v = point({latitude:1.0, longitude:2.0})`.
   - Rust: `['p']` → `[]`. C: `['p']`.
   - Fix: add a Point `Equal` arm (a geo node with radius 0 plus the kept filter), or mark Point not indexable for `Equal`.
3. `bug_in_list_drops_item` (`node_by_index_scan.rs:226-231`, the `keep` filter of the `InList` arm).
   - Query: `n.v IN [date('2020-01-01'), 2]`.
   - Rust: `['date','int']` → `['int']`. C: `['date','int']`.
   - Fix: if any item is dropped, fall back to a label scan (the filter is already kept).
4. `bug_multilabel_and` / `bug_multilabel_or` (`utilize_index.rs:503-535`, `mod.rs:1587,1603`).
   - Setup: `(n:A:B)`, with A indexing x and B indexing y.
   - `WHERE n.x = 1 AND n.y = 2` → `[]`. `WHERE n.y = 2 OR n.x = 1` loses rows. `{x:1, y:2}` → `[]`.
   - C returns the rows.
   - Fix: merge only conjuncts and disjuncts whose label matches the first one. Leave the rest in the post-filter, and bail on OR.
7. `bug_string_range_order` / `_gt`, `tagEnc_not_monotone` (`mod.rs:565` used by `mod.rs:1457`).
   - The TAG encoding does not preserve order, so string ranges over strings with a byte ≤ 0x20, `\` or `_` are wrong in both directions.
   - `n.v < 'JohnA'` with `'John Smith'` stored: `['js']` → `[]`. `n.v > 'a!'` with `'a b'` stored: `[]` → `['ab']`.
   - C is also wrong, differently (it does not encode). Batch runs show C's string range index is much worse.
   - Fix: use an order-preserving escape. For example, map bytes b ≤ 0x20 to `0x21 0x21+b`, shift `!` itself, and so on. Or keep the filter for string ranges.
8. `bug_bool_eq_int` (`mod.rs:760,1360`).
   - Query: `n.v = 1` also returns `v = true`, and `n.v = true` returns `v = 1`: `['int']` → `['bool','int']`.
   - C has the same bug.
10. `bug_empty_string_not_indexed`: `n.v = ''` → `[]`, and `n.v < 'a'` misses `''`. C has the same bug. Fix: an INDEXEMPTY equivalent, or a sentinel encoding for the empty string.
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
- `IndexerM`: `Indexer` (create/drop/remove_if_empty (`removeIfEmpty_spec`, `removeIfEmpty_keeps_filled`, `dropIndex_removeIfEmpty_erases`)/lookups/vector metadata/index_info/progress/
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

NEW CONFIRMED BUG 13 (`index/mod.rs:1412,1438,1496,1508,1566,1627`, `queryField` /
`IndexerM.bug_range_after_fulltext`): `build_query_node` targets
`self.fields.get(key).and_then(|f| f.first())` — the attribute's *first* field whatever its
type. After `CREATE FULLTEXT INDEX FOR (n:L) ON (n.s)` then `CREATE INDEX FOR (n:L) ON (n.s)`,
every range query on `n.s` is built against the fulltext field `s` instead of `range:s`:
`MATCH (n:L) WHERE n.s = 'foo'` Rust `[]`, C `[2]`; `n.s > 'c'` Rust `[]`, C `[1],[2]`;
`(n:L {s:'bar'})` Rust `[]`, C `[3]`; `n.s IN ['foo','bar']` Rust `[]`, C `[2],[3]`
(live, ports 18850 Rust / 18851 C). Fix: pick the field with `ty == IndexType::Range`.
`create_rs_index` runs every `CString::new` (stopwords, language, label) before the options block is
allocated and frees it on the unsupported-language return. The label NUL check precedes the
unsupported-language check, so `createRsIndex_lang_err` is false (`createRsIndex_lang_err_fails`; label
``L\0`` with `language: 'x'` reports the NUL error instead of `Language is not supported: x`);
`createRsIndex_lang_err_cleanLabel` proves the message for NUL-free labels.

## `SetVector`
`Document::set` adds a `VecF32` only if
`vector_options.is_some_and(|o| o.dimension != 0 && o.dimension == vec.len())`
(`mod.rs:730-744`). `SetVector.lean`: `setVector_sound` (key property: for any options and value, a vector add
carries the field's own non-zero dimension), `setVector_good`, `setVector_wrong_dim`,
`setVector_noparams`, `setVector_has_params` / `setVector_none_of_noparams` (link to
`registerOne`: an add only ever targets a field registered *with* params), `setVector_sub_pre`.
This discharges, for vector fields, the `ht` premise of `query_eq_scan_filter` (the RS table
holds `docOf` of every entity: a vector cannot make RediSearch reject the document).

## `LeafAosD` (`cow_btree`: `AosLeaf` generic over `DOC_BYTES`)
`LeafAosD.lean` models `doc_le_bytes`
(`docLeBytes_spec`: the assert passes iff the doc fits; `docLeBytes_lossless`), the D-wide page
(`aos_roundtripD`, `aosMergeD_spec`, `docLayoutD_spec`, `aosBuildD_length`, `fromPairsD_roundtrip`
for every `DOC_BYTES ∈ {1, 2, 4, 8}`) and proves it is the width-8 model at 8 (`aosD8_*`, `fromPairsD8`).
`is_read_width` (`cow_btree/mod.rs:88`, `LeafBytes.isReadWidth`) gates `DOC_BYTES` in the const asserts:
`readWidth_admitted` (`read_width` loads exactly `w` bytes on
every admitted width), `isReadWidth_upto16`, `docBytesAssert_iff`, `admitted_doc_read`,
`admitted_fromPairs_roundtrip`, `docBytesAssert_rejects_3` (the old and new asserts differ on exactly 3/5/6/7), and
`asserted_fromSorted_contents` (`from_sorted` under the full const assert `constAssert`).
`proofs/cow_btree` also proves that `from_sorted`, `insert_batch` and `remove_batch` build
well-formed trees (`fromSorted_wf`, `insertBatch_wf`, `removeBatchTree_wf`), which this project's
tree-level contents theorems (`Leaf.fromSorted_contents`, `Leaf.insertBatch_contents`) complement.

## `create_index` bumps the generation
`indexer.rs` adds `bump_id` at the end of `create_index` when the
label already had an entry (`indexer.rs:357-359`, after every fallible step).
`IndexerM.build` ends with `regen` (`Env.regenTo` is the next `NEXT_INDEX_ID`):
`createIndex_regen` (the stored entry has
id = generation = `regenTo`, current pending 0, and total pending is conserved: every earlier
ticket is stale, so the old job's release stays balanced), `createIndex_old_ticket_stale`
(`is_ticket_current` holds for exactly the new generation), `createIndex_new_entry` (a new label is not
bumped). The protocol-level consequence (an OPERATIONAL index never misses a field) is
`SC.Tickets.main_operational_complete` in `proofs/search_concurrency`.

## Phonetic option
`TextIndexOptions::phonetic_enabled` (`text_index_options.rs:48`) is what `register_fields` asks
(`mod.rs:1245`): C's `"no"` (`PHONETIC_OFF`, written on every fulltext field C did not enable) is off,
like absent and empty. Modelled as `Meta.TextOpts.phoneticEnabled`; `registerOne_text` is stated on it.
`phoneticEnabled_off_spellings` (the Rust unit test), `phoneticEnabled_code`, `phoneticEnabled_iff_ne_no`,
`fixed3514_phonetic_no_is_off`. The RDB side is in `proofs/redis_layer` (`SerialSchema.decField`).

## Open: stopwords and edge-index rollback (#3166, #2781; proofs cite fb3d516d0, repros on main 9d3869aa8)
Two open bugs the proofs missed, each a stated property and a counterexample
matching the live repro (ports 6640 Rust / 6641 C):
- #3166 (`IndexGaps.lean`). `Idx.rsStop` records the stopwords the RediSearch spec was
  created with (`mod.rs:1119-1133`); `register_fields` never changes them. Property
  `StopwordsAgree` / `CreateIndexKeepsStopwords`: what `db.indexes()` reports is what
  RediSearch applies. `bug3166_stopwords_not_applied`, `bug3166_violates`: a range index
  first, then a fulltext index with `stopwords: ['router']` only registers the field on
  the existing spec (`indexer.rs:300-313`) and records `['router']` (`:341-343`); Rust
  matches 'router' (1 row), C does not (0). `createIndex_fresh_stopwords_agree`: the
  new-label path agrees.
- #2781 (`EdgeResync.lean`). The edge arm of `resync_published_indexes`
  (`pending.rs:1872-1880`) with `commit_edge_index_locked`'s re-resolved key
  (`graph.rs:3988-4063`), documents keyed by `(src, dst, id)`. Property
  `RollbackRestoresEdgeIndex`. `bug2781_orphan_edge_doc`, `bug2781_rollback_not_restored`:
  the orphan `(c, d, 0)` survives beside `(a, b, 0)`, as in the live index scan.
  `resyncEdgesFixed_restores` proves the fix (delete the private key when endpoints changed).

## Vector queries and `InList`
`Graph::vector_query_nodes`/`_edges` gate vector queries on `is_operational` (`graph.rs:4408,4476`):
`gated3581_requires_operational`, `gated3581_i3382_refused`, `vectorQueryGuarded_eq_isOperational`
(`Ixr.vectorQuery` itself has no gate: `Property VectorQueryRequiresOperational` / `VectorQueryComplete`;
historical `pre3581_vector_query_ignores_operational`, `pre3581_partial_answer`).
The `InList` arm (`InList.lean`): `expandIn_eq_filter`, `canUtilize_or`,
`canUtilizeInList_eq_or`, `scan_inlist_faithful` prove the `InList` shape (members filtered by
`keep`, utilized iff non-empty and every member indexable) the same as the `Or` of `Equal`s that
`expandIn`/`canUtilize` model. The native index serves `IN [...]` as an InList union of point
lookups (`proofs/native_index`); under the optional `index-falkordb` feature (off in the shipped
`.so`) a Range index is answered by the native index instead, so this project's range theorems
describe the default build. `resync_published_indexes` and `commit_edge_index_locked` are PROVEN by
their membership theorems in `EdgeResync.lean` (the #2781 rollback property stays refuted).

-/
import FalkorIndexLayer.Model
import FalkorIndexLayer.Proofs
import FalkorIndexLayer.Temporal
import FalkorIndexLayer.Strings
import FalkorIndexLayer.Maintenance
import FalkorIndexLayer.Bugs
import FalkorIndexLayer.Meta
import FalkorIndexLayer.IndexM
import FalkorIndexLayer.SetVector
import FalkorIndexLayer.IndexerM
import FalkorIndexLayer.Iter
import FalkorIndexLayer.IndexGaps
import FalkorIndexLayer.EdgeResync
import FalkorIndexLayer.InList
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
import FalkorIndexLayer.LeafAosD
