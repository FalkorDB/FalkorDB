# cow_btree: the copy-on-write B⁺-tree behind the native numeric index

This project proves that `CowBTree` (`graph/src/index/falkordb/data_structures/cow_btree/`) is a correct sorted set of `(key, doc)` pairs: every insert, remove, batch edit and range scan returns exactly what a plain sorted list would. It also proves that a snapshot taken before a write never sees that write.

`CowBTree` is the storage of the native numeric index (#2350, built with `--features index-falkordb`). Every `(value, node id)` entry of a native index column lives in one. So when you run

```cypher
CREATE INDEX FOR (p:Person) ON (p.age)
MATCH (p:Person) WHERE p.age >= 30 AND p.age <= 40 RETURN p
```

the rows the scan returns are a `CowBTree::range` call, and every `SET p.age = …` or `DELETE p` ends in `insert_batch` / `remove_batch` on the tree (`FalkorDbIndex::apply_writes`). How the index uses the tree is proven in [`../native_index`](../native_index).

The per-function Lean↔Rust mapping is in `COVERAGE.tsv`. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does a single `insert` / `remove` keep the tree valid and change exactly one entry?**

Yes. `insert_spec` and `remove_spec` prove that a well-formed tree (`TreeWF`, below) stays well-formed, and that its entries become exactly the sorted-set insert or `List.erase` of the old entries. The returned `bool` is "was absent" / "was present" (`insertRet_spec`, `removeRet_spec`).

`remove` needs `BRANCH_MAX >= 4`:

The const assert requires `BRANCH_MAX >= 4`, and `remove_spec_asserted` proves `remove` correct on every configuration the assert admits, with no extra premise.

**2. Does a range scan return exactly the entries in range, in order?**

Yes. `range_spec` proves that `range(lo, hi)` yields exactly the entries whose key is in the inclusive range `[lo, hi]`, in `(key, doc)` order, including every duplicate of a key; `lo > hi` yields nothing. `point_spec` is the `range(k, k)` case. The generic cursor, which reads the key lazily, yields the same thing (`nextWith_eq`, `rangeDocs_spec`, `rangeTuples_spec`). `firstDoc_eq_point` shows `first_doc(k)` is the head of the point lookup.

**3. Do the batch paths keep the right entries and a valid tree?**

Yes, on main.

- `route_keeps_all` proves the batch sweep hands every entry to some child, including `(u64::MAX, u64::MAX)`. `insertBatch_wf` proves `insert_batch` keeps `TreeWF` and leaves exactly the old entries plus the batch, strictly sorted.
- `remove_batch` on a sorted batch leaves exactly the entries not in the batch, still sorted. An unsorted batch is the Rust `assert!` panic (`removeBatchTree_unsorted`).
- `removeBatchTree_wf` proves `remove_batch` keeps `TreeWF`.
- `from_sorted` builds a well-formed tree holding exactly its input (`fromSorted_wf`).

**4. Is every tree a caller can hold well-formed?**

Yes. `Reachable_wf_asserted` proves that an empty tree, a `from_sorted` tree, and anything reached from them by `insert`, `remove`, `insert_batch` and `remove_batch` satisfy `TreeWF`, on every configuration the const assert admits. So the content specs above hold for every tree the native index ever holds, with no remaining hypothesis.

**5. Can a reader see a write that happened after its snapshot?**

In the immutable model, no: `snapshot_isolated` proves that after you clone the model tree, any sequence of inserts, removes and batch inserts on the writer leaves every range scan of the clone unchanged. That is a property of the model, not a proof about the Rust heap. Rust snapshot isolation relies on `make_private` always cloning a shared page before mutating it, and on leaf pages never being written after they are built; that argument is not proven against the heap (see "What the model does not cover").

**6. Does `heap_bytes` (used by `GRAPH.MEMORY`) count correctly?**

It counts every page once (`heapBytes_eq`). With AoS leaves it is `entries × (8 + DOC_BYTES)` plus the branch vectors (`heapBytes_aos`), so narrowing `DOC_BYTES` from 8 to 4 saves exactly 4 bytes per entry.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `insert_spec` | `insert` keeps `TreeWF` (growing a level on a root split) and adds exactly the new entry | `Insert.lean` |
| `remove_spec` | For `BRANCH_MAX >= 4`, `remove` keeps `TreeWF` (losing root levels as needed) and removes exactly the entry | `Remove.lean` |
| `remove_spec_asserted` | The same, with `4 ≤ BRANCH_MAX` discharged by the const assert | `Assert.lean` |
| `range_spec` | `range(lo, hi)` = the entries with key in `[lo, hi]`, in order | `Cursor.lean` |
| `firstDoc_eq_point` | `first_doc(k)` is the first doc of `point(k)` | `Lookup.lean` |
| `removeBatchTree_wf` | For `BRANCH_MAX >= 4`, `remove_batch` keeps `TreeWF` and leaves exactly `List.filter (∉ batch)` | `BatchWF.lean` |
| `insertBatch_wf` | `insert_batch` keeps `TreeWF`; its entries are the old ones plus the batch, strictly sorted | `InsertBatchWF.lean` |
| `fromSorted_wf` | `from_sorted` builds a well-formed tree holding exactly its input | `InsertBatchWF.lean` |
| `Reachable_wf_asserted` | Every tree reachable through the API is well-formed | `InsertBatchWF.lean` |
| `route_keeps_all` | The `insert_batch` sweep routes every batch entry to a child  | `Batch.lean` |
| `rebalance_eq_cws` | The refactored `Branch::rebalance` computes the same thing as the original | `Batch.lean` |
| `snapshot_isolated` | A clone's range scans are unchanged by any later writes | `Snapshot.lean` |
| `heapBytes_aos` | `heap_bytes` with AoS leaves = `entries × (8 + DOC_BYTES)` + branch vectors | `Extract.lean` |
| `enc_lt_iff` | Encoding `(key, doc)` as one number keeps the lexicographic order | `Basic.lean` |

### How `insert_spec` and `remove_spec` are proved

1. **Model the type.** `Node` is a leaf (a list of entries) or a branch (separators and children), as in `node.rs`. An entry `(key, doc)` is one number `key × 2⁶⁴ + doc` (`enc`); `enc_lt_iff` proves this is the same order as comparing the pair, so every `<` on tuples in Rust is `<` on numbers here.
2. **Model the leaf pages as lists.** Rust has three byte layouts for a leaf page (AoS, compact, compact-indexed). Here a leaf is the sorted list of entries it decodes to. That the byte layouts decode correctly is proven separately in [`../index_layer`](../index_layer).
3. **State the invariant.** `TreeWF` (via `WFb`) says: every leaf is sorted and holds at most `LEAF_MAX` entries and is non-empty unless it is the root; every branch has `2..=BRANCH_MAX` children; and child `i` only holds entries between separators `i-1` and `i`. The bounds are explicit (`lo`, `hi`) so the proof survives the stale separators a remove leaves behind.
4. **Prove the recursive step with a post-condition.** For insert, `insertOne_spec` proves by induction on the tree height that `insert_one` either returns a node with the same bounds, or splits into two valid halves around the promoted separator (`InsPost`); either way the entries are the old ones plus `x`. For remove, the key is the **underflow flag**: `remove_one` may hand back a *weak* node (an empty leaf, or a branch with one child) only when it raises the flag, and the parent repairs it (`rebalance`). A branch with one child is flagged only if `1 < BRANCH_MAX / 2`, which is why the proof needs `BRANCH_MAX >= 4`.
5. **Heights are explicit.** Rust recurses until it meets a leaf; the model passes the uniform height `h` so Lean sees structural recursion. On a tree of uniform depth the two agree, and `TreeWF` guarantees uniform depth.

The hypotheses: `TreeWF` on the input, and entries below `2¹²⁸` (every `(u64, u64)` pair is). `TreeWF` is not an assumption about callers: `Reachable_wf_asserted` proves every tree the API can build satisfies it.

### How `removeBatchTree_wf` is proved

1. **Model the loop.** `merge_underfull` is `muLoop`: each step is `combine_with_sibling` (`muStep_id`) followed by `merge_underfull_below` on the produced node(s). `merge_underfull_below` re-runs the loop one level down, so the model builds it by recursion on the height (`mubF`), together with the fact that it never grows a node and keeps its under-flow flag unless it shrank it (`Mono`). That fact is what proves the Rust loop terminates: a merge, or a repair that shrinks, drops the node count; a borrow whose repairs don't shrink leaves two nodes that are not under-full.
2. **State what a batch can leave.** `Good`: a node that is well-formed, or an under-full branch with exactly one child that is itself `Good`.
3. **Combine `Good` siblings.** `combine_good`: two `Good` siblings (neither need be well-formed) combine into nodes with at least two `Good` children. `mubF_spec`: the repair turns those into `Good` nodes.
4. **The loop.** `muLoop_good`: it ends with every child `Good` and, if two or more remain, none under-full. A `Good` node that is not under-full is well-formed (`Good_wf`, needs `BRANCH_MAX >= 4`). So every branch with two or more children has well-formed children.
5. **The root.** `collapse_good`: the root collapse walks down the lone-child chain to a node with two or more children, or a leaf, which is well-formed.

## Bugs found

No open bugs. Fixed ones are in this file's git history.

## Files

| File | What's in it |
|---|---|
| `Model.lean` | The executable model: `Node`, `Cfg`, the entry encoding `enc`, leaf ops (`leafInsert`, `leafRemove`, `mergeWalk`), `insertOne`, `removeOne`, `rebalance`, `combine`, `toList`, `minOpt` |
| `Basic.lean` | Encoding order (`enc_lt_iff`), sorted-list lemmas (`Sorted`, `sins`) |
| `Invariant.lean` | The invariant: `WFb`, `TreeWF`, the separator chain `COK` |
| `Leaf.lean` | Leaf insert/remove and the midpoint split (`leafInsert_spec`, `leafRemove_spec`) |
| `Insert.lean` | `insertOne_spec`, `insert_spec` |
| `Remove.lean` | The underflow contract (`Weak`), `rebalance`, `combine`, `remove_spec` |
| `Cursor.lean` | The range cursor: `seek_rest`, `next_spec`, `range_spec` |
| `Snapshot.lean` | Writer operations (`Op`, `applyOps`), `snapshot_isolated`, `insertOne_shares_siblings` |
| `Batch.lean` | The batch paths: routing, `subtract_eq`, `cws`, `stripEmpty`, the shape lemmas |
| `BatchWF.lean` | `remove_batch`: `muLoop`, `mubF`, `mergeUnderfull`, `removeBatch`, `removeBatchTree`; `Good`, `combine_good`, `muLoop_good`, `removeBatchTree_wf` |
| `InsertBatchWF.lean` | `mergeWalk_spec`, `MinChain`, `packBranches_chain`, `buildRoot_wf`, `applyBatch_chain`, `insertBatch_wf`, `fromSorted_wf`, `Reachable`, `Reachable_wf` |
| `Lookup.lean` | Read-side additions: `firstDoc`, `containsKey`, insert/remove return flags |
| `Extract.lean` | The generic cursor (`Extract`, `DocExtract`, `TupleExtract`) and `heap_bytes` |
| `BatchAssert.lean` | `removeBatchTree_wf_asserted`: `remove_batch` on every configuration the const assert admits |
| `Assert.lean` | The const assert (`constAssert`, `isReadWidth`) and `remove_spec_asserted` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

No path dependencies. [`../native_index`](../native_index) imports this project's `Model.lean` and `Basic.lean`.

## Coverage

`COVERAGE.tsv` has 98 function rows (plus one comment row):

| Bucket | Rows |
|---|---|
| PROVEN | 49 |
| MODELLED | 2 |
| NOT COVERED | 47 |

- **NOT COVERED:** the leaf byte encodings (`leaf/aos.rs`, `compact.rs`, `compact_indexed.rs`, `read_u64`, `read_width`, `doc_le_bytes`, the byte accessors in `leaf/mod.rs`). This project abstracts a page to its entry list; every one of these is proven in [`../index_layer`](../index_layer). Also two `#[cfg(test)]` helpers (`count`, `walk (leaves)`) and `make_private`, whose clone-before-mutate is argued in `Snapshot.lean` and proven in `index_layer` (`Leaf.makePrivate_cow`).
- **MODELLED:** `partition_point` (abstracted as the length of the true prefix of a monotone predicate) and `Leaf::to_pairs` (the abstraction function itself). Both are proven in `index_layer`.

`bash proofs/coverage.sh` counts each function once, in its best bucket across projects, so these rows count as PROVEN repo-wide.

## What the model does not cover

- **Leaf byte layouts.** A leaf is its sorted entry list. The three byte formats and the fast paths that edit them in place are proven in `index_layer`, not here.
- **Termination of `merge_underfull`** is proven (the model is a total function with a lexicographic measure), but the proof needs the repair to never grow a node, so `merge_underfull_below` is built together with that fact (`mubF`) rather than as a plain recursive function.
- **`insert_batch` byte paths.** The leaf merge is modelled at list level (`mergeWalk`, `mergeBatch`); the byte-level fast paths are proven in `index_layer` (`Leaf.mergeBatch_spec`).
- **`Arc` and memory safety.** Pages are modelled as immutable values. The argument that this is sound (`make_private` always clones; `merge_underfull_below`'s `Arc::make_mut` only writes in place to a node `combine` just built, and clones if it is shared; leaf `Arc<[u8]>` buffers are never written after construction) is written in the `Snapshot.lean` header and the Rust comment, not proven against the heap.
- **`heap_bytes`.** `Vec` capacities are parameters (any value). The `Arc` headers (16 bytes per page) and the `Branch` structs themselves are not counted by the Rust, whose doc calls it approximate.
- **Heights** are ghost parameters (a fully drained root becomes a leaf of height 0, `leafAt0`). No floats are involved: keys are already `u64` here.

## Build and check

```bash
(cd proofs/cow_btree && lake build)     # this project
bash proofs/lean_ci.sh cow_btree      # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh               # repo-wide coverage report
```

**When you change `cow_btree/mod.rs`, `node.rs`, `cursor.rs` or `leaf/mod.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model in `Model.lean` (or `BatchWF.lean`, `Lookup.lean`, `Extract.lean`) to match.
3. Re-run the build, and also build `../native_index`, which imports this project.

If you change a leaf byte layout, the proofs to update are in `../index_layer`. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
