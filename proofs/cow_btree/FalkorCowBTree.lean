import FalkorCowBTree.Model
import FalkorCowBTree.Basic
import FalkorCowBTree.Invariant
import FalkorCowBTree.Leaf
import FalkorCowBTree.Insert
import FalkorCowBTree.Remove
import FalkorCowBTree.Cursor
import FalkorCowBTree.Snapshot
import FalkorCowBTree.Bugs
import FalkorCowBTree.Batch
import FalkorCowBTree.Lookup
import FalkorCowBTree.Extract
import FalkorCowBTree.BugsBatch
import FalkorCowBTree.Assert

/-!
# `cow_btree`: Lean 4 model and proofs (re-targeted to origin/main cab3cf192: #2897 7b60f28db, #3285 cab3cf192)

Targets `graph/src/index/falkordb/data_structures/cow_btree/{mod,node,cursor}.rs` and the list-level
behaviour of `leaf/mod.rs`. The byte encodings of the leaf pages (`leaf/aos.rs`, `compact.rs`,
`compact_indexed.rs`, `read_u64`, `read_width`, `doc_le_bytes`) are proven in `proofs/index_layer`
(`FalkorIndexLayer/Leaf*.lean`); here a leaf is the sorted list of entries it decodes to.

## Lean ↔ Rust
| Lean | Rust |
| --- | --- |
| `insert`, `insertOne`, `insertFlag`, `insertRet` | `mod.rs:223`, `node.rs:306-364` |
| `remove`, `removeOne`, `collapse`, `removeRet` | `mod.rs:260-279`, `node.rs:456-475` |
| `rebalance`, `cws`, `combine` | `node.rs:204`, `:113` (`combine_with_sibling`), `:369` (+ `aos_combine` `:432`) |
| `removeBatchTree`, `removeBatch`, `subtract`, `stripEmpty`, `mergeUnderfull` | `mod.rs:286`, `node.rs:483-538`, `:165`, `:148` |
| `isUnderfull`, `branchUnderfull`, `isEmptyNode` | `node.rs:546`, `:197`, `:555` |
| `route` (`pre2278_route`), `applyBatch`, `insertBatch` | `node.rs:269-289` (old `:194-206`), `:244`, `mod.rs:241` |
| `packBranches`, `buildRoot`, `fromSorted` | `node.rs:56`, `:90`, `mod.rs:190` |
| `Cursor.new`/`seek`, `descendLeft`, `advanceLeaf`, `wholeOf` | `cursor.rs:82-116`, `:140`, `:160`, `:129` |
| `Cursor.next`, `Cursor.nextWith`, `Extract`, `DocExtract`, `TupleExtract` | `cursor.rs:181-207`, `:16-53` |
| `range`, `rangeWith` | `mod.rs:314` (`range`), `:324` (`point`), `:408` (`range_tuples`) |
| `firstDocGo`, `firstDoc`, `containsKey`, `minOpt` | `mod.rs:357-404`, `:343`, `node.rs:230` |
| `heapWalk`, `heapBytes`, `Mem` | `mod.rs:419-441` |
| `constAssert`, `isReadWidth`, `Cfg.ofAssert` (`Assert.lean`) | the const asserts `mod.rs:151-160`, `:173-182`, `:191-200`; `is_read_width` `mod.rs:88` |

## Proven
- `insert_spec`, `remove_spec` (`BRANCH_MAX >= 4`, discharged by the const assert since #2897:
  `remove_spec_asserted`): the invariant `TreeWF` is kept and the entries are
  sorted-set insert / `List.erase`; `insertRet_spec`, `removeRet_spec`: the new `bool` returns are
  "was absent" / "was present"; `removeOne_flag`: `remove_one`'s flag is `Node::is_underfull`.
- `range_spec`, `point_spec`, `range_empty`; `nextWith_eq`, `rangeWith_spec`, `rangeDocs_spec`,
  `rangeTuples_spec`, `rangeTuples_enc`: the generic cursor (lazy key read) yields `make(key, doc)` of
  exactly the entries with key in `[lo, hi]`, in `(key, doc)` order.
- `firstDoc_spec`, `firstDoc_eq_point`, `containsKey_spec`: point lookup = head of the filter of the
  sorted multiset (the nearest-right-subtree fallback is right, `minOpt_head`).
- `route_keeps_all`, `route_routes_TOP`, `route_TOP_example`: the batch sweep loses nothing (Bug 1 fixed).
- `subtract_eq`, `stripEmpty_content`, `cws_content`, `mergeUnderfull_content` (with a termination
  proof, `cws_measure`), `route_content`, `removeBatch_content`, `removeBatch_shape`,
  `removeBatchTree_spec`, `removeBatchTree_unsorted`: `remove_batch` on a sorted batch leaves exactly
  the entries not in the batch (`List.filter`), sorted; unsorted input is the `assert!` panic.
- `rebalance_eq_cws`: the #2278 refactor of `Branch::rebalance` is the same function.
- `heapBytes_eq`, `heapSum_split`, `leafBytes_stride`, `heapBytes_aos`: `heap_bytes` counts every page
  once; with AoS leaves it is `entries * (8 + DOC_BYTES)` plus the branch vectors.
- `snapshot_isolated`, `snapshot_reads_committed`, `insertOne_shares_siblings` (`Snapshot.lean`).

## Bugs
1. FIXED by #2278 (3597b3a82): `insert_batch` dropped `(u64::MAX, u64::MAX)` —
   `pre2278_route_never_routes_TOP`, `pre2278_route_drops_TOP_example` (W1 #13, #2893 / PR #2897).
2. FIXED by #2897 (7b60f28db): `BRANCH_MAX = 3` passed the const assert but broke `remove` (and
   `remove_batch`): historical `pre2897_b3_one_remove_not_wf`, `pre2897_b3_drained_is_empty_lies`,
   `pre2897_b3_drained_min_panics` (W1 #13). The assert now requires `BRANCH_MAX >= 4`:
   `pre2897_b3_excluded`, `constAssert_rejects_B3`, `asserted_single_child_underfull`, and
   `remove_spec_asserted` / `removeRet_spec_asserted` (no `4 ≤ B` premise left) in `Assert.lean`.
3. STILL PRESENT (#2278): `remove_batch` leaves a single-child non-root branch at
   any `BRANCH_MAX >= 4`; one later `remove` then leaves an empty non-root leaf and `insert_batch`
   panics in `Node::min`. `rb_remove_batch`, `rb_remove_batch_not_wf`, `rb_then_remove`,
   `rb_min_panics` (`BugsBatch.lean`, with the Rust repro). Since #3307 (18836a355) `remove_batch` has
   a production caller (`FalkorDbIndex::apply_writes` → `NumericIndex::remove_encoded`), so the
   invariant break is reachable; the panic still needs a single-key `remove`, which only tests call
   (`IndexColumn::remove`). A bounded search of the model over batch-only sequences (`remove_batch` /
   `insert_batch`, `LEAF_MAX/BRANCH_MAX` = 2/4 and 3/5, 500 seeds) found no content, range or
   empty-page failure. `proofs/native_index` rests on these content specs (its CONDITIONAL rows).

## Gaps
- `TreeWF` preservation by `remove_batch` is false (Bug 3); only contents are proven. `insert_batch`
  contents are proven in `proofs/index_layer` (`Leaf.insertBatch_contents`), not here.
- `heap_bytes`: `Vec` capacities are parameters (any value); the `Arc` headers (16 B per page) and the
  `Branch` structs themselves are not counted by the Rust ("approximate", per its doc).
- `DOC_BYTES` only affects the AoS byte layout; the narrowing round-trip is proven in `proofs/index_layer`.
- Heights are ghost parameters; f64 is not involved.
-/
