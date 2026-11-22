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
# `cow_btree`: Lean 4 model and proofs

Targets `graph/src/index/falkordb/data_structures/cow_btree/{mod,node,cursor}.rs` and the list-level
behaviour of `leaf/mod.rs`. The byte encodings of the leaf pages (`leaf/aos.rs`, `compact.rs`,
`compact_indexed.rs`, `read_u64`, `read_width`, `doc_le_bytes`) are proven in `proofs/index_layer`
(`FalkorIndexLayer/Leaf*.lean`); here a leaf is the sorted list of entries it decodes to.

## Lean ↔ Rust
| Lean | Rust |
| --- | --- |
| `insert`, `insertOne`, `insertFlag`, `insertRet` | `mod.rs`, `node.rs` |
| `remove`, `removeOne`, `collapse`, `removeRet` | `mod.rs`, `node.rs` |
| `rebalance`, `cws`, `combine` | `node.rs` (`combine_with_sibling`) (+ `aos_combine`) |
| `removeBatchTree`, `removeBatch`, `subtract`, `stripEmpty` (`BatchWF.lean`, `Batch.lean`) | `mod.rs`, `node.rs` |
| `mergeUnderfull` = `muLoop`/`muStep`/`leftOf`, `mergeUnderfullBelow` = `mubF`/`mubBody` (`BatchWF.lean`) | `node.rs` (`merge_underfull`) (`merge_underfull_below`) |
| `isUnderfull`, `branchUnderfull`, `isEmptyNode` | `node.rs` |
| `route`, `applyBatch`, `insertBatch` | `node.rs`, `mod.rs` |
| `packBranches`, `buildRoot`, `fromSorted`, `mergeWalk`, `mergeBatch`, `chunks` | `node.rs`, `mod.rs`, `leaf/mod.rs`/ |
| `Cursor.new`/`seek`, `descendLeft`, `advanceLeaf`, `wholeOf` | `cursor.rs` |
| `Cursor.next`, `Cursor.nextWith`, `Extract`, `DocExtract`, `TupleExtract` | `cursor.rs` |
| `range`, `rangeWith` | `mod.rs` (`range`) (`point`) (`range_tuples`) |
| `firstDocGo`, `firstDoc`, `containsKey`, `minOpt` | `mod.rs`, `node.rs` |
| `heapWalk`, `heapBytes`, `Mem` | `mod.rs` |
| `constAssert`, `isReadWidth`, `Cfg.ofAssert` (`Assert.lean`) | the const asserts `mod.rs`; `is_read_width` `mod.rs` |

## Proven
- **Every tree the Rust API can build is well-formed** (`InsertBatchWF.lean`): `Reachable_wf`,
  `Reachable_wf_asserted` — `new`/`default`, `from_sorted`, then any sequence of `insert`, `remove`,
  `insert_batch`, `remove_batch` keeps `TreeWF` (`BRANCH_MAX >= 4`, the const assert). So the content
  specs below hold for every tree a caller holds; nothing rests on an unproven invariant any more.
- `insert_spec`, `remove_spec` (`BRANCH_MAX >= 4`, discharged by the const assert:
  `remove_spec_asserted`): the invariant `TreeWF` is kept and the entries are
  sorted-set insert / `List.erase`; `insertRet_spec`, `removeRet_spec`: the `bool` returns are
  "was absent" / "was present"; `removeOne_flag`: `remove_one`'s flag is `Node::is_underfull`.
- `removeBatchTree_wf` (`BatchWF.lean`): `remove_batch` keeps `TreeWF` and leaves exactly
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
- `route_keeps_all`, `route_routes_TOP`, `route_TOP_example`: the batch sweep loses nothing.
- `subtract_eq`, `stripEmpty_content`, `cws_content`, `route_content`, `rebalance_eq_cws`, `muStep_id`
  (the loop body is `combine_with_sibling` plus the repair).
- `heapBytes_eq`, `heapSum_split`, `leafBytes_stride`, `heapBytes_aos`: `heap_bytes` counts every page
  once; with AoS leaves it is `entries * (8 + DOC_BYTES)` plus the branch vectors.
- `snapshot_isolated`, `snapshot_reads_committed`, `insertOne_shares_siblings` (`Snapshot.lean`).

## Bugs

No open bugs. Fixed ones are in this file's git history.

## Gaps
- `heap_bytes`: `Vec` capacities are parameters (any value); the `Arc` headers (16 B per page) and the
  `Branch` structs themselves are not counted by the Rust ("approximate", per its doc).
- `Arc`: pages are values. `make_private` always clones; `merge_underfull_below`'s `Arc::make_mut` only
  mutates in place a node `combine` just built and clones otherwise, so either way no committed page
  is written (argued in `Snapshot.lean`; `make_private` proven in `proofs/index_layer`).
- `DOC_BYTES` only affects the AoS byte layout; the narrowing round-trip is proven in `proofs/index_layer`.
- Heights are ghost parameters (a drained root gets height 0, `leafAt0`); f64 is not involved.
-/
