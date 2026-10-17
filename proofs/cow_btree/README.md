# cow_btree: the copy-on-write B⁺-tree behind the native numeric index

This project proves that `CowBTree` (`graph/src/index/falkordb/data_structures/cow_btree/`) is a correct sorted set of `(key, doc)` pairs: every insert, remove, batch edit and range scan returns exactly what a plain sorted list would. It also proves that a snapshot taken before a write never sees that write.

`CowBTree` is the storage of the native numeric index (#2350, built with `--features index-falkordb`). Every `(value, node id)` entry of a native index column lives in one. So when you run

```cypher
CREATE INDEX FOR (p:Person) ON (p.age)
MATCH (p:Person) WHERE p.age >= 30 AND p.age <= 40 RETURN p
```

the rows the scan returns are a `CowBTree::range` call, and every `SET p.age = …` or `DELETE p` ends in `insert_batch` / `remove_batch` on the tree (since #3307, `FalkorDbIndex::apply_writes`). How the index uses the tree is proven in [`../native_index`](../native_index).

The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorCowBTree.lean`](FalkorCowBTree.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does a single `insert` / `remove` keep the tree valid and change exactly one entry?**

Yes. `insert_spec` and `remove_spec` prove that a well-formed tree (`TreeWF`, below) stays well-formed, and that its entries become exactly the sorted-set insert or `List.erase` of the old entries. The returned `bool` is "was absent" / "was present" (`insertRet_spec`, `removeRet_spec`).

`remove` needs `BRANCH_MAX >= 4`:

- **Before #2897, no.** The const assert accepted `BRANCH_MAX = 3`, and at 3 a single remove broke the tree (see [Bugs found](#bugs-found)).
- **On main, yes.** The assert now requires `BRANCH_MAX >= 4`, and `remove_spec_asserted` proves `remove` correct on every configuration the assert admits, with no extra premise.

**2. Does a range scan return exactly the entries in range, in order?**

Yes. `range_spec` proves that `range(lo, hi)` yields exactly the entries whose key is in the inclusive range `[lo, hi]`, in `(key, doc)` order, including every duplicate of a key; `lo > hi` yields nothing. `point_spec` is the `range(k, k)` case. The generic cursor added by #2278, which reads the key lazily, yields the same thing (`nextWith_eq`, `rangeDocs_spec`, `rangeTuples_spec`). `firstDoc_eq_point` shows `first_doc(k)` is the head of the point lookup.

**3. Do the batch paths lose or keep entries they shouldn't?**

- `insert_batch` used to drop the entry `(u64::MAX, u64::MAX)`. **Fixed by #2278**: `route_keeps_all` proves the batch sweep hands every entry to some child.
- `remove_batch` on a sorted batch leaves exactly the entries not in the batch, still sorted (`removeBatchTree_spec`). An unsorted batch is the Rust `assert!` panic (`removeBatchTree_unsorted`).
- But `remove_batch` does **not** keep the tree well-formed: it can leave a non-root branch with one child (`rb_remove_batch_not_wf`). The contents are still right; the shape is not. This one is still open (Bug 3 below).

**4. Can a reader see a write that happened after its snapshot?**

No. `snapshot_isolated` proves that after you clone the tree, any sequence of inserts, removes and batch inserts on the writer leaves every range scan of the clone unchanged. In Rust this rests on `make_private` always cloning a shared page before mutating it, and leaf pages never being written after they are built (see "What the model does not cover").

**5. Does `heap_bytes` (used by `GRAPH.MEMORY`) count correctly?**

It counts every page once (`heapBytes_eq`). With AoS leaves it is `entries × (8 + DOC_BYTES)` plus the branch vectors (`heapBytes_aos`), so narrowing `DOC_BYTES` from 8 to 4 saves exactly 4 bytes per entry.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `insert_spec` | `insert` keeps `TreeWF` (growing a level on a root split) and adds exactly the new entry | `Insert.lean` |
| `remove_spec` | For `BRANCH_MAX >= 4`, `remove` keeps `TreeWF` (losing root levels as needed) and removes exactly the entry | `Remove.lean` |
| `remove_spec_asserted` | The same, with `4 ≤ BRANCH_MAX` discharged by the const assert (#2897) | `Assert.lean` |
| `range_spec` | `range(lo, hi)` = the entries with key in `[lo, hi]`, in order | `Cursor.lean` |
| `firstDoc_eq_point` | `first_doc(k)` is the first doc of `point(k)` | `Lookup.lean` |
| `removeBatchTree_spec` | `remove_batch` on a sorted batch leaves exactly `List.filter (∉ batch)`, sorted | `Batch.lean` |
| `route_keeps_all` | The `insert_batch` sweep routes every batch entry to a child (the #2278 fix) | `Batch.lean` |
| `rebalance_eq_cws` | The #2278 refactor of `Branch::rebalance` computes the same thing as before | `Batch.lean` |
| `snapshot_isolated` | A clone's range scans are unchanged by any later writes | `Snapshot.lean` |
| `heapBytes_aos` | `heap_bytes` with AoS leaves = `entries × (8 + DOC_BYTES)` + branch vectors | `Extract.lean` |
| `enc_lt_iff` | Encoding `(key, doc)` as one number keeps the lexicographic order | `Basic.lean` |

### How `insert_spec` and `remove_spec` are proved

1. **Model the type.** `Node` is a leaf (a list of entries) or a branch (separators and children), as in `node.rs`. An entry `(key, doc)` is one number `key × 2⁶⁴ + doc` (`enc`); `enc_lt_iff` proves this is the same order as comparing the pair, so every `<` on tuples in Rust is `<` on numbers here.
2. **Model the leaf pages as lists.** Rust has three byte layouts for a leaf page (AoS, compact, compact-indexed). Here a leaf is the sorted list of entries it decodes to. That the byte layouts decode correctly is proven separately in [`../index_layer`](../index_layer).
3. **State the invariant.** `TreeWF` (via `WFb`) says: every leaf is sorted and holds at most `LEAF_MAX` entries and is non-empty unless it is the root; every branch has `2..=BRANCH_MAX` children; and child `i` only holds entries between separators `i-1` and `i`. The bounds are explicit (`lo`, `hi`) so the proof survives the stale separators a remove leaves behind.
4. **Prove the recursive step with a post-condition.** For insert, `insertOne_spec` proves by induction on the tree height that `insert_one` either returns a node with the same bounds, or splits into two valid halves around the promoted separator (`InsPost`); either way the entries are the old ones plus `x`. For remove, the key is the **underflow flag**: `remove_one` may hand back a *weak* node (an empty leaf, or a branch with one child) only when it raises the flag, and the parent repairs it (`rebalance`). A branch with one child is flagged only if `1 < BRANCH_MAX / 2`, which is why the proof needs `BRANCH_MAX >= 4`.
5. **Heights are explicit.** Rust recurses until it meets a leaf; the model passes the uniform height `h` so Lean sees structural recursion. On a tree of uniform depth the two agree, and `TreeWF` guarantees uniform depth.

The hypotheses: `TreeWF` on the input (every tree the Rust builds starts as an empty or `from_sorted` tree, and insert and remove keep it), and entries below `2¹²⁸` (every `(u64, u64)` pair is). The one way to break `TreeWF` on main is `remove_batch` (Bug 3).

## Bugs found

The Rust repros were run against the `cow_btree` sources (see [`../FINDINGS.md`](../FINDINGS.md), wave 1 #13 and wave 7).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `pre2278_route_never_routes_TOP`, `pre2278_route_drops_TOP_example` | `insert_batch` routed `(u64::MAX, u64::MAX)` to no child, so the entry was silently lost | Fixed by #2278 (`route_keeps_all`, `route_routes_TOP`) |
| 2 | `pre2897_b3_one_remove_not_wf`, `pre2897_b3_drained_is_empty_lies`, `pre2897_b3_drained_min_panics` | `BRANCH_MAX = 3` passed the const assert, but one `remove` broke the tree; after draining it, `is_empty` was wrong and `Node::min` panicked | Fixed by #2897 (issue #2893): `constAssert_rejects_B3`, `pre2897_b3_excluded` |
| 3 (W7-btree-1) | `rb_remove_batch`, `rb_remove_batch_not_wf`, `rb_then_remove`, `rb_min_panics` | `remove_batch` can leave a non-root branch with one child at any `BRANCH_MAX >= 4`. One later single-key `remove` then leaves an empty non-root leaf, and the next `insert_batch` panics in `Node::min` | **Open**, fix in PR #3283. Since #3307 `remove_batch` runs in production (`apply_writes` → `remove_encoded`), so the bad shape is reachable; the panic still needs a single-key `remove`, which only tests call |
| W7-btree-2 | `pre3285_docBytes3_reads_garbage`, `pre3285_docBytes3_last_oob` in `../index_layer` (`LeafAosD.lean`) | `DOC_BYTES ∈ {3,5,6,7}` passed the const assert but `read_width` reads only 1/2/4/8, so `CowBTree::<256,256,3>` panicked on its second insert | Fixed by #3285 (issue #3238): the assert now requires `is_read_width(DOC_BYTES)` (`isReadWidth`, `constAssert`) |

Bug 3 in more detail: the batch path repairs under-full children one level at a time. If a branch ends with one child and its parent also ends with only that branch, the parent's merge loop does not run, and the lone-child branch is pushed into the merged node one level up, where nothing looks at it again. `BugsBatch.lean` replays this on a 12-entry tree with `LEAF_MAX = 2`, `BRANCH_MAX = 4`, and its header has the Rust repro. A bounded search of the model over batch-only sequences (the only ones production runs) found no wrong contents, range result or empty page.

Bug 3 is why [`../native_index`](../native_index) has 13 **CONDITIONAL** rows: their contents specs rest on `TreeWF`, which `remove_batch` does not preserve.

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
| `Bugs.lean` | Bug 1 (`pre2278_route_*`) and Bug 2 (`pre2897_b3_*`) counterexamples |
| `Batch.lean` | The batch paths after #2278: routing fix, `subtract_eq`, `mergeUnderfull`, `removeBatchTree_spec` |
| `Lookup.lean` | #2278 read-side additions: `firstDoc`, `containsKey`, insert/remove return flags |
| `Extract.lean` | The generic cursor (`Extract`, `DocExtract`, `TupleExtract`) and `heap_bytes` |
| `BugsBatch.lean` | Bug 3: the `remove_batch` single-child branch, with the Rust repro in the header |
| `Assert.lean` | The const assert (`constAssert`, `isReadWidth`) and `remove_spec_asserted` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

No path dependencies. [`../native_index`](../native_index) imports this project's `Model.lean` and `Basic.lean`.

## Coverage

`COVERAGE.tsv` has 97 function rows (plus one comment row):

| Bucket | Rows |
|---|---|
| PROVEN | 41 |
| MODELLED | 9 |
| NOT COVERED | 47 |

- **NOT COVERED:** the leaf byte encodings (`leaf/aos.rs`, `compact.rs`, `compact_indexed.rs`, `read_u64`, `read_width`, `doc_le_bytes`, the byte accessors in `leaf/mod.rs`). This project abstracts a page to its entry list; every one of these is proven in [`../index_layer`](../index_layer). Also two `#[cfg(test)]` helpers (`count`, `walk (leaves)`) and `make_private`, whose clone-before-mutate is argued in `Snapshot.lean` and proven in `index_layer` (`Leaf.makePrivate_cow`).
- **MODELLED:** `from_sorted`, `insert_batch`, `pack_branches`, `build_root`, `merge_sorted`, `merge_walk`, `partition_point`, `Leaf::to_pairs`, `Leaf::merge_batch`. They are modelled here and their contents are proven in `index_layer` (for example `Leaf.insertBatch_contents`).

`bash proofs/coverage.sh` counts each function once, in its best bucket across projects, so these rows count as PROVEN repo-wide.

## What the model does not cover

- **Leaf byte layouts.** A leaf is its sorted entry list. The three byte formats and the fast paths that edit them in place are proven in `index_layer`, not here.
- **`TreeWF` after `remove_batch`.** False on main (Bug 3); only the contents are proven.
- **`insert_batch` contents.** Proven in `index_layer` (`Leaf.insertBatch_contents`); here only the routing step is proven.
- **`Arc` and memory safety.** Pages are modelled as immutable values. The argument that this is sound (`make_private` always clones; leaf `Arc<[u8]>` buffers are never written after construction) is written in the `Snapshot.lean` header, not proven against the heap.
- **`heap_bytes`.** `Vec` capacities are parameters (any value). The `Arc` headers (16 bytes per page) and the `Branch` structs themselves are not counted by the Rust, whose doc calls it approximate.
- **Heights** are ghost parameters. No floats are involved: keys are already `u64` here.

## Build and check

```bash
cd proofs/cow_btree && lake build     # this project
bash proofs/lean_ci.sh cow_btree      # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh               # repo-wide coverage report
```

**When you change `cow_btree/mod.rs`, `node.rs`, `cursor.rs` or `leaf/mod.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model in `Model.lean` (or `Batch.lean`, `Lookup.lean`, `Extract.lean`) to match.
3. Re-run the build, and also build `../native_index`, which imports this project.

If you change a leaf byte layout, the proofs to update are in `../index_layer`. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. When PR #3283 lands, prove `TreeWF` is kept by `remove_batch` and move the 13 `native_index` CONDITIONAL rows to PROVEN.
