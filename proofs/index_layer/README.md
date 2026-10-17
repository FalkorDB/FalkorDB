# index_layer: when an index scan returns the same rows as a full scan

This project proves when a range-index query through RediSearch returns exactly the rows a label scan plus the `WHERE` filter would return, and shows with counterexamples where it does not. It covers `graph/src/index/mod.rs` and `indexer.rs`: the TAG key encoding, the RediSearch query trees the planner builds, index maintenance at commit, and index creation and deletion. It also proves the byte layouts of the copy-on-write B⁺-tree leaf pages under `graph/src/index/falkordb/data_structures/cow_btree/leaf/`.

User-facing behaviour that depends on it:

- `CREATE INDEX FOR (n:L) ON (n.v)` followed by any query the optimizer turns into a `Node By Index Scan` or `Edge By Index Scan`: `n.v = 1`, `n.v IN [1, 2]`, `n.v > 0`, `n.v >= 'a' AND n.v < 'b'`, `(n:L {v: 1})`;
- keeping the index in step with `CREATE`, `SET`, `REMOVE` and `DELETE`;
- `CREATE` / `DROP INDEX`, `db.indexes()`, full-text and vector index queries.

When the optimizer uses an index, it removes the matching filter. So if the index returns too many rows or too few, the query result is wrong, silently.

The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorIndexLayer.lean`](FalkorIndexLayer.lean), with each re-target's history. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Is an index scan the same as a label scan plus the filter?**

For the query shapes below, yes. The bridge is `query_eq_scan_filter` (and `queryE_eq_scan_filter` for edges): if the index holds the right document for every entity with the label, and the predicate is one the exactness theorems cover, an id is returned iff it has the label and passes the predicate. The exactness theorems are:

- `equal_exact`: `n.v = literal`, for a stored Int, Float or non-empty String;
- `inList_exact`: `n.v IN [scalar literals]`, under the same conditions;
- `numRange_exact`: numeric `<`, `<=`, `>`, `>=` and two-sided ranges, for every stored number including NaN and, since #3081, ±∞;
- `strRange_exact`: string ranges, **only** when no string involved contains a byte ≤ 0x20, a `\` or a `_`, and stored strings are non-empty;
- `temporal_numRange_exact`, `temporal_equal_exact`: since #3076, dates and times are not indexed, so a numeric or scalar query never returns them.

Outside these conditions the counterexamples in [Bugs found](#bugs-found) show exactly how it goes wrong.

**2. Can a stored string collide with another, or break the TAG syntax?**

No. `tag_encode_lower` is the only string transformation between Cypher and RediSearch. `tagEnc_injective` proves two different strings never get the same key. `tagEnc_printable` and `tagEnc_no_backslash` prove a key never contains the separator, NUL, whitespace or `\`. So special characters cannot inject anything. But the encoding does **not** preserve order (`tagEnc_not_monotone`), which is why string ranges are exact only on strings it leaves alone.

**3. Does commit keep the index in step with the graph?**

`commit_lookup` proves what `Indexer::commit` leaves for each id. `commit_preserves_inv` proves that if the graph side hands commit consistent add and remove sets, then after commit the index table matches the committed graph exactly. `commit_both_removes` shows a hazard: an id in both sets ends up with no document, because removes run last. No Cypher sequence that was tried produces both.

**4. Are the population tickets and refcounts safe?**

`ok_inc`, `ok_dec`, `ok_bump` prove the ticket counters stay non-negative. `bump_total` proves recreating an index conserves pending work, and `dec_stale_keeps_current` that a stale worker never touches the new generation. `RC.not_freed` proves no use-after-free of a RediSearch handle while an iterator holds a clone.

**5. Do the B⁺-tree leaf pages decode to what was written?**

Yes, for all three page formats (AoS, compact, compact-indexed): build and read round trip (`aos_roundtrip`, `cBuild_roundtrip`, `ciBuild_roundtrip`, `fromPairs_roundtrip`), every in-place splice and merge equals the list operation (`cInsert_spec`, `ciInsert_spec`, `ciMerge_spec`, `bcmL_eq_merge`), and `Leaf::merge_batch`'s fast paths equal its slow path (`mergeBatch_spec`). This is what lets [`../cow_btree`](../cow_btree) treat a leaf as a sorted list.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `query_eq_scan_filter`, `queryE_eq_scan_filter` | Node and edge index scan = label/type scan + filter, given an exact predicate and a consistent index | `Iter.lean` |
| `equal_exact`, `inList_exact` | `=` and `IN` are exact for stored Int, Float and non-empty String | `Proofs.lean` |
| `numRange_exact` | Numeric ranges are exact for every stored number, ±∞ included since #3081 | `Proofs.lean` |
| `strRange_exact` | String ranges are exact on strings without bytes ≤ 0x20, `\`, `_` | `Strings.lean` |
| `tagEnc_injective`, `tagEnc_printable` | The TAG encoding never collides and never emits a separator or whitespace | `Proofs.lean` |
| `commit_preserves_inv` | After commit the index matches the committed graph | `Maintenance.lean` |
| `RC.not_freed` | A RediSearch handle is not freed while an iterator holds it | `Iter.lean` |
| `setVector_sound` | A vector is added to a document only with the field's own non-zero dimension (#3087) | `SetVector.lean` |
| `fromPairs_roundtrip`, `mergeBatch_spec` | Every leaf format round-trips; every leaf fast path equals the slow path | `LeafDispatch.lean`, `LeafOps.lean` |
| `readWidth_admitted` | `read_width` reads exactly `w` bytes for every width the const assert admits (#3285) | `LeafBytes.lean` |

### How `query_eq_scan_filter` is proved

1. **Model the values.** A double is `ENum`: finite values embedded in order into `Int`, plus `±inf` and `NaN` with IEEE comparison. Only `=`, `<` and `<=` are ever used, never arithmetic, so the order embedding loses nothing. `-0.0` and `0.0` are the same value, as both Cypher and RediSearch compare with `==`. Strings are byte lists.
2. **Model the Rust.** `buildQ` is `build_query_node`: the RediSearch query tree for a Cypher predicate. `docOf` is the document `Document::set` writes. `indexHit` is "the index returns this entity", and `holds` is the Cypher predicate itself.
3. **Specify RediSearch.** `rsMatch` states the documented LLAPI semantics: numeric range test, exact TAG token, byte-lexicographic TAG range, union and intersection, the empty node, and "an empty TAG value is not indexed". These are ordinary definitions, not Lean axioms. Every live run agreed with them.
4. **Prove each predicate shape exact.** `equal_exact`, `numRange_exact`, `strRange_exact` and the others each prove `indexHit = holds` for one shape, by cases on the stored value and the bounds.
5. **Combine.** `RunSpec` says RediSearch returns the keys of exactly the stored documents that match. With the maintenance invariant (the table holds `docOf` of every entity with the label), the per-entity exactness lifts to the whole query.

The hypotheses: the RediSearch specification (`RunSpec`, the 82 AXIOMATISED rows); the maintenance invariant, which `commit_preserves_inv` proves given consistent add and remove sets from the graph; and each shape's exactness conditions. Stored integers of 2⁵³ or more are not modelled (`i as f64` is taken as exact; the query-side guard is proven by `intLosesPrecision_sound`).

## Bugs found

Each was reproduced live with `proofs/index_layer/repro.py`, which runs each query without and with the index on a Rust and a C server. Shown as full scan → index scan. Status is at main 0f45a8685: the header's last re-target (cab3cf192) lists bugs 1-4, 7, 8, 10, 12 and 13 as still present, and no later commit touches that code.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 (W2-index-1) | `bug_folded_temporal_constant` | `n.v = date('2020-01-01')` returns **every** node of the label: the constant is folded and treated as indexable, the filter is dropped, then the index refuses it and falls back to a label scan. C correct | **Open** |
| 2 (W2-index-2) | `bug_point_equality` | `n.v = point({latitude:1.0, longitude:2.0})` returns `[]`. C correct | **Open** |
| 3 (W2-index-3) | `bug_in_list_drops_item` | `n.v IN [date('2020-01-01'), 2]` drops the date row. C correct | **Open** |
| 4 (W2-index-4) | `bug_multilabel_and`, `bug_multilabel_or` | On `(n:A:B)` with A indexing `x` and B indexing `y`, `n.x = 1 AND n.y = 2` returns `[]`, and the `OR` loses rows. C correct | **Open** |
| 5 (W2-index-5) | `pre3076_temporal_in_numeric_range` | `n.v > 0` matched a stored date | Fixed by #3076 (issue #2959): `fixed3076_temporal_in_numeric_range`, `temporal_not_indexed` |
| 6 (W2-index-6) | `pre3072_exclusive_equal_bounds` | `n.v > 'a' AND n.v < 'a'` returned `'a'` | Fixed by #3072 (issue #2961): `fixed3072_exclusive_equal_bounds` |
| 7 | `tagEnc_not_monotone`, `bug_string_range_order`, `bug_string_range_order_gt` | The TAG encoding does not preserve order, so string ranges over strings with a byte ≤ 0x20, `\` or `_` are wrong both ways: `n.v < 'JohnA'` misses `'John Smith'` | **Open**. C is also wrong, differently |
| 8 | `bug_bool_eq_int` | `n.v = 1` also returns `v = true`, and the reverse | **Open**. C has the same bug |
| 9 | `pre3081_open_bound_excludes_inf` | `n.v > 0` missed a stored `+∞` | Fixed by #3081: `fixed3081_open_bound_includes_inf` |
| 10 | `bug_empty_string_not_indexed` | `n.v = ''` returns `[]`, and `n.v < 'a'` misses `''` | **Open**. C has the same bug |
| 11 | (proven in `ops_traverse`) | An undirected edge index scan returned each edge once instead of twice | Fixed by #3081 |
| 12 | (outside the model) | Geo radius near the poles: RediSearch geohash is limited to \|lat\| ≤ 85.05, so points at 89.99 are missed | **Open**. C has the same bug |
| 13 (W4-index-1) | `bug_range_after_fulltext` | After a full-text index and then a range index on the same `n.s`, every range query on `n.s` targets the full-text field and returns `[]`. C correct | **Open**. Fix: pick the field whose type is `IndexType::Range` |
| 14 (W7-btree-2) | `pre3285_docBytes3_reads_garbage`, `pre3285_docBytes3_last_oob` | `CowBTree` with `DOC_BYTES = 3/5/6/7` passed the const assert but read garbage docs or panicked | Fixed by #3285: `readWidth_admitted`, `admitted_fromPairs_roundtrip` |
| W3-conc-3 | `pre3087_dim_mismatch`, `pre3087_noparams` | A vector of the wrong dimension (or a vector field with no options) made RediSearch reject the whole document, so the node dropped out of the label's other indexes | Fixed by #3087 (issue #3075): `setVector_sound` |

C-only, not a Rust bug: C crashes on `distance(n.v, point(…)) <= 0` with a range index; Rust returns the right result.

## Files

| File | What's in it |
|---|---|
| `Model.lean` | The executable model of `mod.rs`/`indexer.rs`: `ENum`, `tagEnc`, node and edge keys, `Document::set` (`docOf`), the query builders (`buildQ`), the RediSearch spec (`rsMatch`), `commit`, `Slots` |
| `Proofs.lean` | Key round trips, `tagEnc_injective`, `intLosesPrecision_sound`, `equal_exact`, `inList_exact`, `numRange_exact` |
| `Strings.lean` | `strRange_exact` and the byte-order lemmas (`lexLt_total`, `tagEnc_id`) |
| `Temporal.lean` | Since #3076: `temporal_not_indexed`, `temporal_numRange_exact`, `temporal_equal_exact` |
| `Maintenance.lean` | `Indexer::commit` (`commit_lookup`, `commit_preserves_inv`, `commit_both_removes`) and the ticket counters |
| `Bugs.lean` | Counterexamples 1-10 and the `pre…`/`fixed…` pairs for the fixed ones |
| `SetVector.lean` | The vector arm of `Document::set` (#3087): `setVector_sound` |
| `Meta.lean`, `IndexM.lean` | `Field` and `Index` metadata, the RediSearch spec lifecycle |
| `IndexerM.lean` | `Indexer`: create, drop, lookups, `index_info`; `createIndex_ok`, `dropIndex_spec`, bug 13 |
| `Iter.lean` | Result iterators, RediSearch refcounts (`RC.not_freed`), and the bridges `query_eq_scan_filter`, `queryE_eq_scan_filter`, `fulltextQ_exact`, `vectorQ_sound` |
| `LeafBytes.lean` | Byte helpers of the leaf pages, `read_width` (`readWidth_admitted`) |
| `LeafAos.lean`, `LeafAosD.lean` | The AoS page at `DOC_BYTES = 8` and at any width (#2278), bug 14 |
| `LeafCompact.lean` | `CompactLeaf`: build round trip, splices, merge |
| `LeafIndexed.lean`, `LeafIndexedOps.lean`, `LeafIndexedMerge.lean` | `CompactIndexedLeaf`: build round trip, splices, merge |
| `LeafMerge.lean` | `merge_sorted`, `merge_walk`, `partition_point`, gallop |
| `LeafBlockCopy.lean`, `LeafBlockCopy2.lean` | `block_copy_merge`, list level and byte level |
| `LeafOps.lean` | `Leaf::merge_batch`: every fast path equals the slow path |
| `LeafDispatch.lean` | The `Leaf` enum, `from_pairs` format choice, `fromPairs_roundtrip` |
| `LeafTree.lean` | Tree glue: `pack_branches`, `build_root`, `make_private`, `from_sorted`, `insert_batch` contents |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

No path dependencies.

## Coverage

`COVERAGE.tsv` has 279 rows: all of `index/mod.rs` (94) and `indexer.rs` (43), the `cow_btree` leaf and tree functions this project proves byte for byte (59), and the RediSearch bindings (82, plus one commented row for the Redis module ABI types).

| Bucket | Rows |
|---|---|
| PROVEN | 195 |
| AXIOMATISED | 83 |
| MODELLED | 1 |

- **AXIOMATISED:** the 82 functions of `index/redisearch/mod.rs` are the RediSearch LLAPI FFI bindings; their specification is `rsMatch`/`rsStore` in `Model.lean`. The 83rd row is the commented-out `redis.rs` line for the Redis module ABI types.
- **MODELLED:** `is_ticket_current`, modelled as `ticket.gen = Slots.gen` with no theorem of its own.

## What the model does not cover

- **Large stored integers.** Integers of 2⁵³ or more are rounded when stored as `f64`; the model takes `i as f64` as exact. The query-side guard is proven ([`../native_index`](../native_index) has a bit-level model with the `i64::MIN` hole).
- **RediSearch internals.** Vector (HNSW) search, full-text scoring and geo distance are left to RediSearch. Geo appears only as a value kind; there are no distance theorems (bug 12 is outside the model).
- **Concurrency.** Refcounts are modelled as counters; `ArcSwap` and the locks are not modelled.
- **Index options.** `create_rs_index` and `register_fields` options are FFI results passed as parameters.
- **Graph-side bookkeeping.** `index_add_docs` / `remove_docs` in `graph.rs` are not modelled. They are assumed to meet the side conditions of `commit_preserves_inv`, which the 22 live maintenance scenarios in `diff2.py` checked.
- **The native index.** The `CowBTree` tree logic is in [`../cow_btree`](../cow_btree), and the native numeric index that stores into it is in [`../native_index`](../native_index). This project covers the RediSearch path and the leaf byte layouts.

## Build and check

```bash
cd proofs/index_layer && lake build    # this project
bash proofs/lean_ci.sh index_layer     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `index/mod.rs`, `index/indexer.rs`, the index-scan operators, `utilize_index.rs`, or a `cow_btree/leaf/` file:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build, and re-run `proofs/index_layer/repro.py` against live servers for a query-building change.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. When you fix one of the open bugs above, keep its counterexample as a `pre<PR>_` theorem and add the `fixed<PR>_` or exactness theorem next to it, as was done for #3072, #3076 and #3081.
