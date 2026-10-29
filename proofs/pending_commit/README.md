# pending_commit: how a write query stages, reads back and commits its changes

This project proves how a write query's changes are held in `runtime::pending::Pending`, read back by later clauses of the same query, and applied at `Commit`. It also covers the attribute store the properties land in (`graph/src/graph/attribute_store.rs`), and the deleted-entity snapshots a query reads after a `DELETE`. Every Cypher write depends on it:

- `CREATE`, `SET`, `REMOVE`, `DELETE` and `MERGE`;
- reading your own writes in the same query (`SET n.v = 2 RETURN n.v`);
- reading a deleted entity later in the query (`DELETE n WITH n RETURN n.v`);
- the write statistics (`Nodes created`, `Properties set`, …);
- `UNIQUE` and `MANDATORY` constraints, checked at commit;
- RDB save and load of properties, and `GRAPH.MEMORY` for the attribute store.

The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`PendingCommit.lean`](PendingCommit.lean), together with the history of each re-target. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does a query read back exactly what it wrote, and does commit store exactly that?**

Yes, for properties and labels.

- Staged properties stay strictly sorted by attribute id and hold the last write per attribute (`stage_sorted_lastWrite`); `rustUpsert_eq` shows the Rust binary-search upsert is that model.
- `read_your_writes_existing` and `read_your_writes_new` prove that what the query reads before `Commit` is exactly what the store holds after it, for existing and newly created entities.
- For labels, `overlay_eq_ref` and `commit_eq_ref` prove that for any sequence of `SET n:L` / `REMOVE n:L`, the query reads, and commit stores, the verdict of the last clause that mentions each label. `removeOp_no_dup_push` shows `REMOVE` never stages a label twice, so `Labels removed` counts each once.

**2. Is the attribute store's span merge correct on all three code paths?**

An entity's properties are a *span*: `(attr_id, value)` pairs, strictly ascending by id, never storing `Null` (a null in an update means "remove"). `Block::merge_span` has three paths: patch in place, pure removal, and a general two-pass merge. `mergeSpan_correct` proves all three leave a sorted, null-free span that reads back as the reference, and report the reference `(nremoved, nset)` counts that feed `Properties set` / `Properties removed`. `newLen_eq` proves the general path's first pass computes the length the second pass writes, which is the Rust `debug_assert!`.

**3. Can the arena's counters underflow, or `compact` lose data?**

No. A `Block` tracks `dead` and `slack` entries in `u32` counters updated with plain `-`, which panics in debug and wraps in release. `setSpan_ok`, `freeSpan_ok`, `mergeSpan_ok` and `compact_ok` prove that from a well-formed block every operation succeeds without underflow and leaves a well-formed block. `live_eq_sum_len` proves `compact`'s `live` count is the live entry count, which is its `debug_assert_eq!`. With contents, `compactC_ok` proves every entity reads exactly what it read before compaction.

**4. Do properties survive an RDB save and load?**

Yes. `encode_decode_roundtrip` proves decoding what was encoded into an empty store restores exactly the encoded entities' spans. `pack_unpack` proves the 12-byte packed value round-trips, with floats as IEEE bits so NaN payloads and `-0.0` survive.

**5. Are UNIQUE constraints enforced correctly since #3334?**

Since #3334, a `UNIQUE` check looks each affected member's key up in the constraint's index instead of scanning the whole label. `uniqueViolated_spec` proves the check reports a violation iff two members share a full key, or a member's key is held by a live entity outside the affected set. `uniqueViolated_agrees` proves this equals the old label-scan verdict. Both need the index to be complete (see CONDITIONAL below).

**6. Does a deleted entity keep reading correctly for the rest of the query?**

**Not when its id is reused.** `sound_under_Good` proves snapshot reads are correct as long as no snapshotted id belongs to a live entity. The engine's allocator reissues freed ids first, which breaks that: `rust_reads_dead_entity` shows a new node reading the deleted node's data. This is #2876, still open (see [Bugs found](#bugs-found)). Two fixes are proven to restore it: deferring id reuse to the end of the query (`deferred_preserves_Good`), and open PR #2964's evict-on-reissue (`evict_preserves_Good`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `mergeSpan_correct` | All three `merge_span` paths give the reference span and the reference counts | `Span.lean` |
| `read_your_writes_existing`, `read_your_writes_new` | What a query reads before Commit is what the store holds after | `Props.lean` |
| `overlay_eq_ref`, `commit_eq_ref` | Labels read and committed follow the last `SET`/`REMOVE` per label | `Labels.lean` |
| `setSpan_ok`, `compact_ok`, `live_eq_sum_len` | Arena counters never underflow; `compact` keeps its `debug_assert` | `Arena.lean` |
| `compactC_ok`, `mergeSpanC_ok` | With contents: every entity reads the same after compaction; a merge changes only its entity | `ContentCompact.lean`, `ContentMerge.lean` |
| `insertAttrs_spec`, `importResolved_spec` | The store's batch writes give each entity exactly its own write | `StoreApi.lean` |
| `encode_decode_roundtrip` | RDB encode then decode restores every span | `StoreCodec.lean` |
| `structural_partition` | Memory accounting: structural plus per-entity bytes equal the whole allocation, no underflow | `StoreMem.lean` |
| `commit_spec` | A successful `Pending::commit` grows each counter (nodes and relationships created, labels removed, nodes deleted, relationships deleted including the implicit ones) by exactly the staged changes, and the constraints hold afterwards | `PendingCommitOp.lean` |
| `uniqueViolated_spec`, `uniqueViolated_agrees` | The #3334 index-lookup UNIQUE check is right, and agrees with the old label scan | `UniqueIndex.lean` |
| `sound_under_Good`, `deferred_preserves_Good` | Snapshot reads are right while no snapshotted id is live; deferring id reuse keeps that | `Snapshot.lean` |
| `c_eq_rust_plus_cancelled` | C's created/deleted counts = Rust's + the entities created and deleted in the same segment | `Stats.lean` |

### How `mergeSpan_correct` is proved

1. **Model the span.** A span is a `List Entry`, `Entry = (attr_id, Val)`. `Val` only distinguishes `null`, an inline scalar and a heap value, because that is all the store looks at (`Span.lean`). Byte packing is proven separately in `Pack.lean`.
2. **Write one reference.** `refGet` says what reading attribute `k` after the merge must return (the update's value if it has one, a removal if it is null, else the old value). `refRemoved` and `refSet` are the counts `insert_attrs` documents.
3. **Model each path with the Rust loop structure.** `patch` is fast path 1, `pureRemove` with `skipBelow` is fast path 2, and `mergeGen` is the general two-pass merge. `mergeSpan` picks the path as `Block::merge_span` does.
4. **Prove each path against the reference**, by induction on the two sorted lists: `patch_get`/`patch_counts`, `pureRemove_ref`/`pureRemove_count`, and `mergeGen_get`/`mergeGen_sorted`/`mergeGen_removed`. The headline is a case split on the chosen path.
5. **Hypotheses.** The old span is sorted and null-free, and the update is sorted. The store keeps the first two everywhere (that is what this theorem and `Content.lean` re-establish); `Pending` hands updates sorted (`stage_sorted_lastWrite`). One exception was `import_attrs_resolved`, which could pass duplicate ids before #3063 (`pre3063_resolved_duplicate`).

## Bugs found

The Rust repros are in `graph/tests/lean_pending_commit.rs` (run locally, not committed; each asserts C's behaviour, so it fails today). See [`../FINDINGS.md`](../FINDINGS.md) for the full repro queries. "Live" means checked against a Rust release module and the C module, last re-checked at 2c874022a.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| #2876 / W2-pending-1 | `rust_reads_dead_entity` | A deleted entity's snapshot is keyed by id, and a `CREATE` later in the query reuses the id, so the new entity reads the deleted one's labels and properties. `MERGE` then creates a duplicate node (persistent damage). Same for relationships. Confirmed live: `MATCH (n) DELETE n WITH count(*) AS c CREATE (m {v:2}) WITH m MATCH (x) RETURN x.v` over `(:A {v:1})` gives Rust `1`, C `2` | **Open** (#2876). Fix proposed in open PR #2964 (`evict_preserves_Good`, `evict_reads_live_entity`) |
| W2-pending-2 | `cancel_example`, `c_eq_rust_plus_cancelled` | Entities created and deleted in the same segment vanish from the statistics: `CREATE (a:L {v:1, w:2}) DELETE a` reports only `Labels added`; C reports `Nodes created: 1, Properties set: 2, Nodes deleted: 1` | **Open** |
| W2-pending-3 | (no Lean theorem; found while modelling) | Writes to a deleted relationship are applied and counted: `REMOVE r.v` after `DELETE r`, and `SET r.v` after a `DETACH DELETE` cascade | **Open** |
| #2769 | `deg_panics_on_pending_deleted` | `CREATE (a)-[r:R]->(b) DELETE r SET b.d = indegree(b)` asks the committed graph for the endpoints of an edge that was never committed, and panics, killing the server. C returns `0` | **Open** (#2769) |
| #2776 | `pre3524_replace_keeps_staged`, `fixed3524_replace_clears_staged`, `replace_reads_src` | `SET x = m` / `SET r = {…}` kept attributes staged earlier in the query instead of replacing them | Fixed by #3524 (af45ca776) |
| setSpan u16 | `setSpan_u16_wrap` | `set_span` stores `n as u16`, so 65,536+ attributes on one entity wrap and the span reads empty. Only a GRAPH.BULK header with that many columns or a malformed RDB can reach it; `setSpan_cast_exact` shows sorted unique ids always fit | Bulk side fixed by #3063 (`normResolved_fits`); the malformed-RDB side remains, not reproduced |
| #3063 | `pre3063_resolved_duplicate` | A GRAPH.BULK header naming a property twice stored two entries for one id | Fixed by #3063 (`resolved_duplicate_fixed`: the first column wins, as in C) |
| #3448 | `bug3448_remove_of_staged_label` (property `RemLCommitted`), `bug3448_cancelled_node_keeps_remL` (property `RemLLive`) | `remove_node_labels` stages the removal of a label the node only had staged (a created node, an earlier `SET`), and `delete_pending_node` keeps the node's `remove_labels`. `CREATE (n:Foo) REMOVE n:Foo DELETE n` then ships `RemoveLabels` before the node's `CreateNode` and the replica forces a full resync (effects_emit_apply). The inflated `Labels removed` stat is the same in C | **Open** (#3448) |
| #3453 | `bug3453_pendingWithLabels_yields_deleted` (property `PendingWithLabelsLive`) | `deleted_node` keeps `set_labels` and `get_pending_nodes_with_labels` does not filter deleted ids, so `SET n:B DELETE n MERGE (m:B)` matches the deleted node. The composed MERGE scan property is in ops_aggregate | **Open** (#3453) |
| #3447 | `bug3447_cascade_drops_type_key` (property `CancelledTypeRegistered`) | A pending edge of a new type cascaded by `DETACH DELETE` empties its `created_rels_by_type` group, which is dropped, so commit never registers the type; the replica then sees an id gap (effects_emit_apply) | **Open** (#3447) |
| #3339 | `pre3339_isValid_any_depth` | Property validation accepted lists nested to any depth | Fixed by #3339 (`validate_depth`: accepted values have depth ≤ 4096) |

Also confirmed live: property counters are net at commit in Rust and per clause in C (`CREATE (n {v:1}) SET n.v = 2`: Rust set 1, C set 2 / removed 1). That is a policy question, not a bug, and is recorded in `FINDINGS.md`. W4-fn-1 (labels staged before a `DELETE` are missing from the deleted snapshot) is also in that area; it was confirmed by another project and its repro tests are in the same test file.

**CONDITIONAL rows.** `check_node_constraint` and `check_edge_constraint` rest on the index contract the Rust documents (hypotheses `hIdx`/`hScan` in `UniqueIndex.lean`): the supporting index returns every live label member not written by this commit whose key values can be matched exactly, and the label matrix (or tensor) lists every live member. The RediSearch side is FFI, so this is not proven here.

## Files

| File | What's in it |
|---|---|
| `Span.lean` | Spans and the three `merge_span` paths: `mergeSpan_correct`, `newLen_eq` |
| `Arena.lean` | `Block` counters (`dead`, `slack`) and slot effects: `setSpan_ok`, `freeSpan_ok`, `compact_ok`, `live_eq_sum_len` |
| `Content.lean`, `ContentMerge.lean`, `ContentCompact.lean` | `Block` with contents: `setSpanC_ok`, `mergeSpanC_ok`, `compactC_ok`; `setSpan_u16_wrap` |
| `Pack.lean` | 12-byte packed values and the block heap: `pack_unpack`, `compactHeap_unpack` |
| `BinSearch.lean` | std `binary_search_by`: `bsearch_ok_iff` |
| `NameMap.lean` | `AttrNameMap` and its `u16` cap: `getOrCreate_spec` |
| `Store.lean`, `StoreFold.lean`, `StoreApi.lean` | `DataBlock` radix directory and every `AttributeStore` method: `runOps_spec`, `insertAttrs_spec`, `importResolved_spec` |
| `StoreCodec.lean`, `StoreMem.lean` | RDB codec, `trim`, memory accounting: `encode_decode_roundtrip`, `structural_partition` |
| `Labels.lean` | Staged labels: `overlay_eq_ref`, `commit_eq_ref`, `removeOp_no_dup_push` |
| `Props.lean` | Staged properties: `stage_sorted_lastWrite`, `read_your_writes_*`, `replace_reads_src` |
| `Snapshot.lean` | Deleted-entity snapshots and id reuse: `sound_under_Good`, `rust_reads_dead_entity`, `deferred_preserves_Good` |
| `EvictPR2964.lean` | **Models open PR #2964, not main**: `evict_preserves_Good` |
| `Stats.lean` | Write statistics: `c_eq_rust_plus_cancelled`, `cancel_example` |
| `PendingAttrs.lean`, `PendingState.lean`, `PendingRels.lean`, `PendingDeg.lean`, `PendingDocs.lean` | All of `pending.rs`: validation, staging, reads, created relationships, degrees (`deg_panics_on_pending_deleted`), `IndexDocs`, `clear`, `effects_count` |
| `PendingCommitOp.lean` | `Pending::commit` and constraint enforcement: `commit_spec`, `check_spec`, `enforce_spec` |
| `UniqueIndex.lean` | #3334's index-lookup UNIQUE check: `uniqueViolated_spec`, `uniqueViolated_agrees`, `checkUnique_spec` |
| `PendingCancel.lean`, `CancelRust.lean` | Unwinding a pending node and handing ids back to the graph: `deletePendingNode_spec_rust`, `cancelN_spec` |
| `CommitOpNext.lean` | `CommitOp::next` with `end_segment`: `commitNext_ok` |
| `StagedBugs.lean` | Staged label/type state that outlives its entity: #3447, #3448, #3453 counterexamples, `deletePendingNode_keeps_remL` |
| `Hold.lean` | #3436/#3439-era borrow holding: a caller-held `graph_mut()` / `graph()` across `Pending::commit`, `end_segment`, `IndexDocs::commit` equals main's sequential borrows (`commit_same`, `single_mut_same`, `single_rd_same`), on `runtime_core`'s `RuntimeCore.Cell` borrow-word model |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Path dependencies (`lakefile.toml`): [`../runtime_core`](../runtime_core), for the atomic_refcell borrow-word model `RuntimeCore.Cell` that `Hold.lean` runs on; and [`../graph_queries`](../graph_queries), for its model of the graph's id spaces (`GQ.IdS`, `GQ.rCancel`), which `CancelRust.lean` uses to prove the cancel contract instead of assuming it.

## Coverage

`COVERAGE.tsv` has 180 rows: all of `attribute_store.rs` (66) and `pending.rs` (67), plus 21 `graph.rs` functions and 26 operator functions (`ops/{commit,create,set,delete,remove}.rs`).

| Bucket | Rows |
|---|---|
| PROVEN | 136 |
| MODELLED | 12 |
| NOT COVERED | 27 |
| AXIOMATISED | 3 |
| CONDITIONAL | 2 |

- **All of `attribute_store.rs` is PROVEN**, and 65 of 67 `pending.rs` functions; the other two are the CONDITIONAL constraint checks.
- **AXIOMATISED:** `set_node_labels_product`, `set_nodes_labels_bulk`, `remove_nodes_labels`. They call GraphBLAS on the label matrices, which is assumed to behave as set insert and delete, so commit is `(C ∪ set) \ remove`.
- **MODELLED:** operator and graph functions whose effect is modelled here but proven elsewhere or only as part of a larger step: `create_batch`, `set_batch`, `set_inner`, the delete paths, `remove`, `create_nodes` and the id bounds (proven in `graph_queries`), `delete_nodes`, `import_node_attrs_resolved`.
- **NOT COVERED:** operator plumbing (`new`, `next`, pattern and attribute resolution), the relationship tensor (`create_relationships_bulk`, `delete_relationships`, `delete_implicit_edges`, proven in `versioned_matrix`), name interning, and the RediSearch commit boundary.

`bash proofs/coverage.sh` counts each function once, in its best bucket across projects.

## What the model does not cover

- **Values.** The store sees `(attr_id, Val)` after `unpack`; `Pack.lean` proves the byte layer separately.
- **`Arc` sharing.** `Arc::make_mut` is modelled as value semantics (copy-on-write). `trim`'s `Arc::get_mut` ownership is a predicate (`owned`), and `shrink_to_fit` capacities are parameters.
- **Write loops.** `merge_span`'s per-entry write loops are summarised by the slice they leave; cells past `new_len` inside `cap` are unread slack.
- **The graph and RediSearch.** Operations `Pending::commit` calls on the graph and RediSearch are uninterpreted. `get_relationship_endpoints` is a partial function, where `none` is the panic.
- **Collections.** `FxHashMap` and `RoaringTreemap` are duplicate-free association lists and lists.
- **Other projects.** Relationship tensors, id spaces and effects emission are proven in `versioned_matrix`, `id_space` and the effects projects.
- **Constraint index.** The UNIQUE check's index completeness is a hypothesis (CONDITIONAL above). Composite keys are taken as injective.

## Build and check

```bash
cd proofs/pending_commit && lake build   # this project (also builds ../graph_queries, ../runtime_core and its ../redis_layer)
bash proofs/lean_ci.sh pending_commit    # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                  # repo-wide coverage report
```

**When you change `runtime/pending.rs`, `graph/attribute_store.rs`, the write operators (`ops/create.rs`, `set.rs`, `delete.rs`, `remove.rs`, `commit.rs`) or the deleted-entity accessors in `runtime.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR. A fix for #2876, #2769 or W2-pending-2 should turn the matching counterexample into a `pre<PR>_` history theorem and add the proof of the fixed behaviour (for #2876, either `deferred_preserves_Good` or `evict_preserves_Good` already covers the two proposed designs).
