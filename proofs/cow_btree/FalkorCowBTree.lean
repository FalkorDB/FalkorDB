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
import FalkorCowBTree.BatchWF
import FalkorCowBTree.Lookup
import FalkorCowBTree.Extract
import FalkorCowBTree.BugsBatch
import FalkorCowBTree.InsertBatchWF
import FalkorCowBTree.Assert

/-!
# `cow_btree`: Lean 4 model and proofs (re-targeted to origin/main 1d15bf713: #3283 799968a8b; earlier #2897 7b60f28db, #3285 cab3cf192)

Targets `graph/src/index/falkordb/data_structures/cow_btree/{mod,node,cursor}.rs` and the list-level
behaviour of `leaf/mod.rs`. The byte encodings of the leaf pages (`leaf/aos.rs`, `compact.rs`,
`compact_indexed.rs`, `read_u64`, `read_width`, `doc_le_bytes`) are proven in `proofs/index_layer`
(`FalkorIndexLayer/Leaf*.lean`); here a leaf is the sorted list of entries it decodes to.

## Lean ↔ Rust
| Lean | Rust |
| --- | --- |
| `insert`, `insertOne`, `insertFlag`, `insertRet` | `mod.rs:223`, `node.rs:321-379` |
| `remove`, `removeOne`, `collapse`, `removeRet` | `mod.rs:260-279`, `node.rs:471-490` |
| `rebalance`, `cws`, `combine` | `node.rs:219`, `:113` (`combine_with_sibling`), `:384` (+ `aos_combine` `:447`) |
| `removeBatchTree`, `removeBatch`, `subtract`, `stripEmpty` (`BatchWF.lean`, `Batch.lean`) | `mod.rs:286`, `node.rs:498-555`, `:180` |
| `mergeUnderfull` = `muLoop`/`muStep`/`leftOf`, `mergeUnderfullBelow` = `mubF`/`mubBody` (`BatchWF.lean`) | `node.rs:155-191` (`merge_underfull`), `:563-573` (`merge_underfull_below`) |
| `pre3283_mergeUnderfull`, `pre3283_removeBatch`, `pre3283_removeBatchTree` (`Batch.lean`, history) | the same functions before #3283 |
| `isUnderfull`, `branchUnderfull`, `isEmptyNode` | `node.rs:576`, `:212`, `:585` |
| `route` (`pre2278_route`), `applyBatch`, `insertBatch` | `node.rs:284-304` (old `:194-206`), `:259`, `mod.rs:241` |
| `packBranches`, `buildRoot`, `fromSorted`, `mergeWalk`, `mergeBatch`, `chunks` | `node.rs:56`, `:90`, `mod.rs:190`, `leaf/mod.rs:50`/`:97`, `:471` |
| `Cursor.new`/`seek`, `descendLeft`, `advanceLeaf`, `wholeOf` | `cursor.rs:82-116`, `:140`, `:160`, `:129` |
| `Cursor.next`, `Cursor.nextWith`, `Extract`, `DocExtract`, `TupleExtract` | `cursor.rs:181-207`, `:16-53` |
| `range`, `rangeWith` | `mod.rs:314` (`range`), `:324` (`point`), `:408` (`range_tuples`) |
| `firstDocGo`, `firstDoc`, `containsKey`, `minOpt` | `mod.rs:357-404`, `:343`, `node.rs:245` |
| `heapWalk`, `heapBytes`, `Mem` | `mod.rs:419-441` |
| `constAssert`, `isReadWidth`, `Cfg.ofAssert` (`Assert.lean`) | the const asserts `mod.rs:151-160`, `:173-182`, `:191-200`; `is_read_width` `mod.rs:88` |

## Proven
- **Every tree the Rust API can build is well-formed** (`InsertBatchWF.lean`): `Reachable_wf`,
  `Reachable_wf_asserted` — `new`/`default`, `from_sorted`, then any sequence of `insert`, `remove`,
  `insert_batch`, `remove_batch` keeps `TreeWF` (`BRANCH_MAX >= 4`, the const assert). So the content
  specs below hold for every tree a caller holds; nothing rests on an unproven invariant any more.
- `insert_spec`, `remove_spec` (`BRANCH_MAX >= 4`, discharged by the const assert since #2897:
  `remove_spec_asserted`): the invariant `TreeWF` is kept and the entries are
  sorted-set insert / `List.erase`; `insertRet_spec`, `removeRet_spec`: the `bool` returns are
  "was absent" / "was present"; `removeOne_flag`: `remove_one`'s flag is `Node::is_underfull`.
- `removeBatchTree_wf` (#3283, `BatchWF.lean`): `remove_batch` keeps `TreeWF` and leaves exactly
  `List.filter (∉ batch)`; `removeBatchTree_spec`, `removeBatchTree_unsorted` (the `assert!` panic).
  Behind it: `Good` (well-formed, or an under-full lone-child chain), `combine_good`, `mubF_spec`,
  `muLoop_good` (with a termination proof by a lexicographic measure, `muStep_dec`), `stripEmpty_good`,
  `removeBatch_good`, `collapse_good`.
- `insertBatch_wf`, `fromSorted_wf` (`InsertBatchWF.lean`): `insert_batch` keeps `TreeWF` and its
  entries become the sorted union; `from_sorted` builds a well-formed tree holding exactly its input.
  Behind them: `mergeWalk_spec`, `chunks_spec'`, `MinChain`, `reseparate`, `packBranches_chain`,
  `buildRoot_wf`, `applyBatch_chain`, `route_chain`.
- `range_spec`, `point_spec`, `range_empty`; `nextWith_eq`, `rangeWith_spec`, `rangeDocs_spec`,
  `rangeTuples_spec`, `rangeTuples_enc`: the generic cursor (lazy key read) yields `make(key, doc)` of
  exactly the entries with key in `[lo, hi]`, in `(key, doc)` order.
- `firstDoc_spec`, `firstDoc_eq_point`, `containsKey_spec`: point lookup = head of the filter of the
  sorted multiset (the nearest-right-subtree fallback is right, `minOpt_head`).
- `route_keeps_all`, `route_routes_TOP`, `route_TOP_example`: the batch sweep loses nothing (Bug 1 fixed).
- `subtract_eq`, `stripEmpty_content`, `cws_content`, `route_content`, `rebalance_eq_cws`, `muStep_id`
  (the #3283 loop body is `combine_with_sibling` plus the repair).
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
3. FIXED by #3283 (799968a8b, issue #3237, W7-btree-1): `remove_batch` left a single-child non-root
   branch at any `BRANCH_MAX >= 4`; one later `remove` then left an empty non-root leaf and
   `insert_batch` panicked in `Node::min`. Historical `pre3283_rb_remove_batch`,
   `pre3283_rb_remove_batch_not_wf`, `pre3283_rb_then_remove`, `pre3283_rb_min_panics` (`BugsBatch.lean`,
   with the Rust repro, now the test `remove_batch_never_leaves_a_single_child_branch`). On main:
   `removeBatchTree_wf`, `removeBatchTree_wf_asserted`, `rb_remove_batch_fixed`, `rb_then_remove_fixed`.

## Gaps
- `heap_bytes`: `Vec` capacities are parameters (any value); the `Arc` headers (16 B per page) and the
  `Branch` structs themselves are not counted by the Rust ("approximate", per its doc).
- `Arc`: pages are values. `make_private` always clones; `merge_underfull_below`'s `Arc::make_mut` only
  mutates in place a node `combine` just built and clones otherwise, so either way no committed page
  is written (argued in `Snapshot.lean`; `make_private` proven in `proofs/index_layer`).
- `DOC_BYTES` only affects the AoS byte layout; the narrowing round-trip is proven in `proofs/index_layer`.
- Heights are ghost parameters (a drained root gets height 0, `leafAt0`); f64 is not involved.
-/
