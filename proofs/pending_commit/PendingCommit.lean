/-
# Pending mutations, Commit, and the attribute store

A Lean model of how a FalkorDB-rs write query stages mutations in
`runtime::pending::Pending`, reads them back, and applies them at `Commit`; of
the attribute store they land in; and of the deleted-entity snapshots later
clauses read through. 407 theorems, no `sorry`/`admit`/`axiom`. Coverage per
Rust function: `COVERAGE.tsv` — every function of `attribute_store.rs` (66) is PROVEN;
of `pending.rs` (72 non-test) 70 are PROVEN and `check_node_constraint` /
`check_edge_constraint` are CONDITIONAL on the native index lookup's completeness (#3334, #3547). Repros: `graph/tests/lean_pending_commit.rs`
(asserting C's behaviour / the agreed semantics, so each FAILS today).
Line numbers are current main (`5c7af54f2`), as are the COVERAGE.tsv rows, except where a section
names an older commit (it keeps that commit's numbers).

## Id-space quarantine, `Hold` and attribute invariants

* `IdSpace::reserve` skips a per-version quarantine (`quarantined`: ids freed or cancelled by earlier
  batches of the query). `Snapshot.RS` has `quar`; `createRust` is the allocator, `createRust_preserves_Good` /
  `createRust_Wf` (invariant `Wf` kept by `del`, `commit`, `createRust`), `rust_reads_live_entity`
  (live `1|2|[]`). The earlier allocator and counterexample are `pre3641_createRust`,
  `pre3641_rust_reads_dead_entity`, kept as historical counterexamples.
  `remove_pending_relationships_for_node` calls `g.get_type_id_mut` per cancelled edge: `PS.RegSpec`,
  `rmStepG`/`removeRelsG`/`deletePendingNode` take `regT`; `CancelRust.regT_spec` discharges it
  (`removeRelsG_spec`/`deletePendingNode_spec` also say every cascaded edge's type is registered).
* `Hold` is modelled against `RuntimeCore.QueryGraph`
  (`holdMut_panics_iff_graphMut`, `published_read_no_borrow`, `hold_after_publish_panics`).
* `AttrInv` (the two facts runtime_core's accessors assume: `inv_hdisj`, `inv_huniq`, kept by
  the whole staging API), `KeyBorrow` (`schema::Key`'s `Hash`/`Borrow`), `CommitOpNext.commitNew`.
  Operator rows (`ops/create|set|delete|remove.rs`) and the graph.rs rows are PROVEN by reference to
  proofs/ops_apply and proofs/graph_queries. `check_node_constraint`/`check_edge_constraint` stay
  CONDITIONAL: the native `hIdx` needs native_index's global "a column holds exactly the tuples below
  its cursor" (open there), and `hScan` is graph_queries' `LabCons` reachability closure (one-step
  preservation only).
* Still open (re-run live at 256ab4d3f): W2-pending-2 (`CREATE (a) DELETE a` reports no counters).

## SAVE walks attributes by seek

`AttributeStore::encode_with_range` (attribute_store.rs:1427) now finds its first live id with
`nth_live_id` (:1485: rank/select on the deleted-id `RoaringTreemap`) and `take(count)`s, instead of
`window … (max count 1)`; `Encode for RoaringTreemap::encode_with_range` (serialization.rs:180) seeks
with `select`/`advance_to` instead of `skip`. The earlier model `Content.encode_spec` stays as a
statement; every `file:line` cite in these four files was re-read against main.

| Module | Rust | Theorems |
| --- | --- | --- |
| `SaveRoaring` | roaring 0.11.5 `rank`/`select`/`advance_to` as definitions on `RSet` (the owners' abstraction) | `RSet.rank_spec`, `select_spec`, `iterFrom_spec` |
| `SaveNthLive` | `nth_live_id` :1485 | `nthLive_spec`, `nthLive_drop`, `nthLive_empty_deleted`, `nthLive_past_end`, `bsearch_spec`, `mid_bounds`, `rank_le`, `liveThrough_no_overflow` |
| `SaveSlice` | `encode_with_range` :1427 | `prEncode_spec`, `prEncode_eq_main` (= the earlier `Content.encode` for `count > 0`), `count_zero_differs` / `count_zero_witness`, `pr_encode_decode_roundtrip`, `walk_spec` |
| `SaveLayout` | serialization.rs:180, `Graph::encode_payload` graph.rs:5172, `encode_graph` encoder/mod.rs:68 | `encRoarPR_eq` (every `count`/`offset`), `encoder_counts_pos` (`count = 0` unreachable), `chunk_concat`, `encodeGraphPR_eq`, `load_encode_pr`, `multi_load_pr` |

`SaveLayout` needs graph_persist's layout, so pending_commit now requires `../graph_persist`
(graph_persist requires nothing: no cycle; pending_commit already sat above it for `SaveSlice`).
Trust: the roaring crate implements the `RSet` abstraction, the same level as the owners'
`contains`/`iter` (checked by the randomized scratch test `lean3438_roaring_contract`: 3000 sets over
four 2^32 high keys, all pass). Live: 300 random stores, every `offset`/`count`: PR = old for
`count > 0`; Redis with `VKEY_MAX_ENTITY_COUNT 3`: `SAVE` + `DEBUG RELOAD` = before. The tensor and
matrix halves of this change are in versioned_matrix `TensorSave` and graphblas_wrappers `EncodeDup`.

## Unique enforcement through `query_for` + `lookup` (#3547 / #2386)

`unique_violated` (pending.rs:217) takes
`query_for: Fn(Vec<Value>) -> Option<Q>` and `lookup: Fn(Q) -> L`. `use_index` starts `true`, and a
member for which `query_for` is `None` sends the whole set to the label scan (the lookups gathered so far
are dropped, none runs). The RediSearch pair is `redisearch_unique_query` (:174, new) + `get_indexed_*`;
under `index-falkordb` it is `native_unique_query` + `ReadOnce::candidates` (index/falkordb/unique.rs,
not modelled here: an abstract hypothesis). `check_node_constraint` is now :1522, `check_edge_constraint`
:1614, `commit` :1280 (`Pending::commit` in replication), `is_value_valid_for_index_lookup` :160 (compiled
only without `index-falkordb`); the new `native_unique_tests` module is test code.

| Lean | Rust |
| --- | --- |
| `UQ.gather` (`qf : List V → Option Q`) | member loop of `unique_violated` (:231-262): `use_index = true`, `query_for(values)`, `lookups.push(query)`, decline clears `lookups` |
| `UQ.uniqueViolated` | `unique_violated` (:217): `lookups.into_iter().any(\|q\| lookup(q).any(collides))` / `scan().any(collides)` |
| `UQ.checkUnique/checkNode2/checkEdge2` | `check_node_constraint` (:1522), `check_edge_constraint` (:1614) |
| `UQ.redisearchQuery`, `evalQ` (UniqueQuery.lean) | `redisearch_unique_query` (:174), `get_indexed_nodes/_edges` |

`gather_none`, `gather_some` (`use_index` ends true iff no
gathered key is declined), `uniqueViolated_spec`, `uniqueViolated_agrees`, `checkUnique_spec`. Also:
`uniqueViolated_declined` (a declined member: verdict independent of `lookup`), `uniqueViolated_indexed`
(none declined: scan not consulted), `redisearchQuery_none_iff` / `_single` / `_and`, `redisearch_hIdx`
(from Equal-exactness of the index alone), `checkUnique_redisearch_spec`, `redisearch_unindexed_scans`.
The check rows stay CONDITIONAL: the native `(query_for, lookup)` contract (`hIdx`) is undischarged.

## `node_has_changes`

`node_has_changes` (pending.rs:760, inner `has` :764): `PC.nodeHasChanges_spec` (true iff the node is a key of `set_labels`,
`remove_labels`, `new_nodes_attrs`, `existing_nodes_attrs`), `hasKey_eq` (the `is_empty` short-cut
is a no-op), `nodeHasChanges_false_labels` / `_false_attrs` (false ⇒ both overlays are the
identity), `nodeHasChanges_overapprox` (sufficient, not exact). Runtime side: runtime_core
`Accessors.not_differs_labels/_attrs`.

## Index-backed UNIQUE enforcement and bulk attribute import

* UNIQUE constraints are enforced by an index lookup
  per member (`unique_violated`, pending.rs:178; `is_value_valid_for_index_lookup` :153)
  instead of a label scan per member. `UniqueIndex.lean`: `uniqueViolated_spec` (violated
  iff two members share a full key or a member's key is carried by a live candidate outside
  the affected set), `uniqueViolated_agrees` (equals the label-scan verdict `PC.check`
  under the constraint's pre-commit invariant), `checkUnique_spec`, `checkMandatory_eq_old`,
  `exactLookup_spec`. The comparison with a specification needs the index contract the Rust
  doc states (hypotheses `hIdx`/`hScan`: the index lags only this `Commit`, the scan lists
  every live member), hence CONDITIONAL rows for the two check functions.
* `import_attrs_resolved` sorts stably, `dedup_by_key`s and
  drops a trailing `ATTRIBUTE_ID_NONE`. `StoreApi`: `normResolved_sorted`,
  `normResolved_fits` (≤ 65,535 entries, so `set_span`'s `n as u16` is exact for any bulk
  header), `normResolved_get` (the *first* column of a repeated id wins, as C),
  `importResolved_spec` (no length precondition), `importResolved_reads`,
  `resolved_duplicate_fixed`.
* Property validation bounds list nesting at `MAX_VALUE_NESTING` = 4096 —
  `PA.validate_depth` (accepted ⇒ old-valid ∧ depth ≤ 4096), `PA.validate_nest`; the save/load consequence is value_math `stored_roundtrip`, which holds when `VecsFit` does (every vector fits the u32 dimension, < 2^32 elements: property validation accepts vectors of any length).

## The graph's `IdSpace` owns the ids (pins cited at main 5c7af54f2)

| Lean | Rust |
| --- | --- |
| `PS.P` (no `nodeEntry`/`relEntry`) | `struct Pending` (pending.rs:307): the boundary fields and the id-space builders are gone |
| `PS.rmStep` (+ `taken` removal), `removeRelsG` | `remove_pending_relationships_for_node(id, g)` (:902): `taken_relationship_ids.remove` + `g.cancel_relationship_id(rel_id)?` (:944-947) |
| `PS.deletePendingNode` | `delete_pending_node(id, g)` (:851): unwinds and hands the id back via `g.cancel_node_id(id)?` (:888) |
| `PD.endSegment`, `clear` | `end_segment` (:1885), which replaced `clear` + `open_id_boundaries` |
| `CommitOpNext.*` | `CommitOp::next` (ops/commit.rs:69) with `end_segment` (:139) |

New theorems: `removeRelsG_spec`, `foldl_rmStep_taken`, `deletePendingNode_spec`,
`deletePendingNode_refused`, `endSegment_spec`, `commitNext_ok`,
`commitNext_commit_err`, `commitNext_endSegment_err`, `pops_in_order`;
`removeRels_spec` now also states `taken` loses the cascaded ids. The graph's
cancel is the hypothesis `PS.CancelSpec freeN freeR` (= `IdSpace::cancel` inserts into the
free set of its own id space and shrinks neither; proofs/id_space, and `IdSpaceContract.cancel_ok` in proofs/graph_queries).
**Discharged (2026-10-11):** `CancelRust.lean` models `cancel_node_id` /
`cancel_relationship_id` over the two id spaces (`IdSpace::cancel`, id_space.rs:529)
and proves `cancelN_spec`/`cancelR_spec`; `deletePendingNode_spec_rust`,
`removeRelsG_spec_rust` are the instantiated theorems.

## Modelled here ↔ there

| Lean | Rust |
| --- | --- |
| `Span.mergeGen`, `newLen`         | `Block::merge_span` general path, 2nd / 1st pass (attribute_store.rs:699-792) |
| `Span.patch`                      | `merge_span` fast path 1, in-place replace (:648-667) |
| `Span.pureRemove`, `skipBelow`    | `merge_span` fast path 2, pure removal (:671-696) |
| `Span.mergeSpan`                  | `Block::merge_span` path selection (:636) |
| `Span.refGet/refRemoved/refSet`   | reference semantics; counts as documented on `insert_attrs` (:1306) |
| `Arena.Blk`, `Arena.Inv`          | `Block {slots, arena, dead, slack}` (:335), `Slot` doc invariant (:319) |
| `Arena.retire/resize/grow`        | `retire_span` (:535), `resize_span_slack` (:548), `grow_slots` (:560) |
| `Arena.setSpan/freeSpan/mergeSpan`| `Block::set_span` (:573), `free_span` (:616), `merge_span` slot effects |
| `Arena.compact`, `live`           | `Block::compact` (:798) and its `debug_assert_eq!` (:821) |
| `Labels.stage/removeLabels/nodeHas/overlay` | `stage_node_labels` (pending.rs:691), `remove_node_labels` (:741), `node_has_label` (:778), `update_node_labels` (:806) |
| `Labels.stagedNodeLabels/updateNodeLabels/stagedLabelIds` | `staged_node_labels` (pending.rs:824), `update_node_labels` over it, `Runtime::staged_label_ids` (ops/delete.rs:196) |
| `Labels.removeOp`                 | `Runtime::remove` label branch (ops/remove.rs:128-145) |
| `Labels.commitLabels`             | `set_nodes_labels_bulk` + `remove_nodes_labels` in `Pending::commit` (pending.rs:1405-1421) |
| `Props.upsert`, `stageAll`        | `set_node_attribute` / `set_relationship_attribute` (pending.rs:595, :1005) |
| `Props.readQ`, `normQ`            | `get_node_attribute_no_delete_check` (runtime.rs:1591) + evaluator's null |
| `Props.commitNew`                 | `import_node_attrs` → `import_attrs` (graph.rs:1898, attribute_store.rs:1351) |
| existing-entity commit            | `set_nodes_attributes` → `insert_attrs` → `merge_span` (graph.rs:1861/:1871) |
| `Props.replaceFrom`               | `clear_node_attrs_for_replace` / `clear_relationship_attrs_for_replace` + the source's entries, every `SET e = …` branch (ops/set.rs:125-148, and the node/relationship-target arms of `set_inner` :151) |
| `Snapshot.RS`, `readRt`           | `Runtime::deleted_nodes/_relationships` and every snapshot-first accessor (runtime.rs:1571-1999: `get_node_attribute` to `get_node_outdegree_by_type`) |
| `Snapshot.del/commit/createRust`  | `delete_nodes_bulk`/`delete_entity` (ops/delete.rs), `CommitOp` (ops/commit.rs), `IdSpace::reserve` bin-first |
| `Stats.step`, `Stats.P`           | `created_nodes`, `delete_pending_node` (pending.rs:851), commit counters (:1344-1415) |

Hypotheses (the AXIOMATISED rows): GraphBLAS `set_all`/`remove` on the label
matrices behave as set insert/delete, so commit is `(C ∪ set) \ remove`
(`commitLabels`); a reclaimed id has no stored span (`delete_nodes` calls
`remove_all`), which `read_your_writes_new` assumes.

## Wave 4 additions (whole-file coverage of the store and of `Pending`)

| Lean module | Rust |
| --- | --- |
| `BinSearch` | std `binary_search_by` (rustc 1.98.1 `core/src/slice/mod.rs:2976`), used by every sorted lookup |
| `NameMap` | `AttrNameMap` (attribute_store.rs:131-252), the `MAX_ATTRIBUTES` cap and `as u16` mint |
| `Pack` | `Tag`/`PackedAttr`/`heap_index`, `pack_value`, `store_packed_value`, `unpack`, `release_*`, `compact`'s heap remap, `SpanRef::heap_bytes` (:256-532, :806-816, :883) |
| `Content`, `ContentMerge`, `ContentCompact` | `Block` with contents: `grow_slots`, `set_span`, `free_span`, `merge_span`, `compact`, `maybe_compact`, `SpanRef` (:560-872) |
| `Store`, `StoreFold`, `StoreApi`, `StoreCodec`, `StoreMem` | `DataBlock` radix directory and every `AttributeStore` method, RDB encode/decode, `trim`, memory accounting (:896-1497) |
| `PendingAttrs`, `PendingState`, `PendingRels`, `PendingDeg`, `PendingDocs`, `PendingCommitOp` | all of `pending.rs`: validation, staging, reads, created/cancelled relationships, degrees, label scans, `IndexDocs`, `clear`, `effects_count`, constraints, `commit` |

Headline theorems: `bsearch_ok_iff`/`bsearch_err`; `getOrCreate_spec`;
`pack_unpack`, `pack_hinv`, `release_hinv`, `compactHeap_unpack`;
`setSpanC_ok`, `mergeSpanC_ok`, `compactC_ok` (each slot reads exactly the
intended span, every other slot is unchanged, `CInv` holds); `dsetSpan_ok`,
`dmergeSpan_ok`, `dremove_ok`, `dtrim_ok`; `runOps_spec` →
`insertAttrs_spec`/`importAttrs_spec`/`importResolved_spec`/`removeAll_spec`;
`encode_decode_roundtrip`; `structural_partition` (structural + per-entity
bytes = whole allocation, no underflow); `rustUpsert_eq` (the Rust upsert is
`Props.upsert`); `setAttr_spec`, `getA_spec`, `updA_spec`; `removeRels_spec`;
`createdRel_spec`; `deletedDeg_some`; `absorb_spec`, `resync_spec`,
`clear_spec`; `scan_ok`, `check_spec`, `enforce_spec`, `commit_spec`;
`createRust_preserves_Good`, `rust_reads_live_entity`.

Wave-4 findings:
* `setSpan_u16_wrap` — `set_span` stores `n as u16`; with ≥ 65,536 entries for
  one entity the slot reads empty while the arena keeps the entries no counter
  covers (`Arena.Inv` false, `compact`'s `debug_assert_eq!` would fire).
  `import_attrs_resolved` (GRAPH.BULK) deduplicates and drops `ATTRIBUTE_ID_NONE`, so it
  cannot pass that many (`normResolved_fits`); only `decode_with_count` (malformed
  RDB), which does not deduplicate ids, can. `setSpan_cast_exact` shows strictly ascending
  ids below `ATTRIBUTE_ID_NONE` always fit. Not reproduced.

## Proven (earlier waves)

* `mergeSpan_correct` — all three `merge_span` paths produce a strictly sorted,
  null-free span that reads back as the reference and report the reference
  `(nremoved, nset)`; `newLen_eq` — the first pass's length is what the second
  emits (the `debug_assert` at :765); `cntL_eq_refRemoved` — fast path 2's count
  over the span equals `insert_attrs`' count over the pairs.
* `setSpan_ok`, `freeSpan_ok`, `mergeSpan_ok`, `compact_ok` — from a
  well-formed block (`arena = dead + Σcap`, `slack = Σ(cap-len)`, `1 ≤ len ≤ cap`
  for live spans, spans in bounds) every operation succeeds without a u32
  underflow and yields a well-formed block; `live_eq_sum_len` — `compact`'s
  `live` is exactly the live entry count.
* `overlay_eq_ref`, `nodeHas_eq_ref`, `commit_eq_ref`, `run_disjoint` — for any
  sequence of `SET n:…`/`REMOVE n:…` the query reads, and commit stores, the
  last clause's verdict per label; add/remove sets stay disjoint;
  `removeOp_no_dup_push` — REMOVE never stages a label twice, so
  `labels_removed` counts each once; `removeLabels_closed`, `removeLabelsC_count` —
  a removal that only cancels a staged add bumps
  `staged_labels_removed` instead, and every removed label is still counted once.
* `stage_sorted_lastWrite` — staged properties stay strictly sorted and hold
  the last write per attribute; `read_your_writes_existing`,
  `read_your_writes_new` — what the query read before Commit is exactly what
  the store holds after, for existing and newly created entities.
* `sound_under_Good`, `del_Good`, `createRust_preserves_Good`,
  `rust_reads_live_entity` — snapshot reads are correct as long as no
  snapshotted id is live, and deferring id reuse to query end
  maintains that across delete/commit/create.
* `c_eq_rust_plus_cancelled` — C's `Nodes created/deleted` = Rust's +
  |cancelled| on every well-formed segment.

## Counterexamples (Lean) → confirmed bugs (Rust, and vs C on live servers)

2. **Cancelled entities vanish from statistics** (`cancel_example`).
   `CREATE (a:L {v:1, w:2}) DELETE a`: Rust reports nothing but `Labels added`;
   C `Nodes created: 1, Properties set: 2, Nodes deleted: 1`.
   `CREATE (a)-[:R]->(b) DELETE a`: Rust `Nodes created: 1`; C 2/1 nodes, 1/1 edges.
   Inconsistent within Rust too: a pending edge deleted directly is counted
   created + deleted. Fix: add `cancelled_nodes.len()` /
   `cancelled_relationships.len()` (and their staged props/labels) to both sides
   (pending.rs:1344-1415). Tests `bug_cancelled_*`.
3. **Writes to a deleted relationship are applied and counted.** `REMOVE r.v`
   after `DELETE r` (ops/remove.rs:144-146 has no deletion check for relationships;
   nodes have one at :119-121) → Rust `Properties removed: 1`, C none. `SET r.v = 2`
   after `DETACH DELETE a` cascades `r` → Rust set 1/removed 1, C none: the
   cascade is recorded only in `Runtime::deleted_relationships`, which set.rs:274-278
   does not consult. Test `bug_writes_to_deleted_relationship_are_counted`.

Also confirmed vs C (stats only, no Rust test): property counters are net at
commit in Rust and per clause in C — `CREATE (n {v:1}) SET n.v = 2` Rust set 1,
C set 2/removed 1; `MATCH (n) SET n.v = 2 SET n.v = 1` Rust set 1/removed 1, C
none; `UNWIND range(1,3) AS i SET n.v = i` Rust 1/1, C 3/3; `REMOVE n:A SET n:A`
Rust none, C added 1/removed 1.

Re-found but already known: `Labels added` counts schema labels (#2655), plain DELETE detaches (#2573).

## Suspicions (not confirmed)

* `decode_with_count` does not deduplicate attribute ids before `set_span`, which
  assumes a strictly sorted null-free span; reachable only via a malformed RDB.
  (`import_attrs_resolved` deduplicates; its GRAPH.BULK callers drop
  `Null` values before calling it, src/commands/bulk_insert.rs:227, call :460.)

## Modelling assumptions

* Values in the content-level store are `(attr_id, Val)` after `unpack`;
  `Pack` proves the byte layer separately (floats as IEEE bits).
* `Arc::make_mut` gives value semantics (COW); `trim`'s `Arc::get_mut`
  ownership is an `owned` predicate; `shrink_to_fit` capacities are
  parameters of `structural_partition`.
* `merge_span`'s per-entry write loops are summarised by their final slice
  (`writeAt`); cells beyond `new_len` inside `cap` are unread slack.
* Graph/RediSearch operations called by `Pending::commit`, `IndexDocs::commit`
  and `resync_published_indexes` are uninterpreted (`GOps`, `ci`/`ce`);
  `get_relationship_endpoints` is a partial function (`none` = panic).
* `FxHashMap`/`RoaringTreemap` are duplicate-free association lists / lists.
* Relationship tensors, id spaces and effects emission are other projects
  (proofs/versioned_matrix, proofs/id_space, proofs/replication).
## Staged state that outlives what it describes (`StagedBugs.lean`)

The query-time views were proven (`overlay_eq_ref`, `pendingWithLabels_spec`,
`deletePendingNode_spec`); that the staged maps only name entities existing at
commit/replay is stated in `StagedBugs.lean`:
* `RemLCommitted` (a staged removal is of a committed label): `fixed3448_run_remL_committed`
  (via `Labels.removeOp_committed`); `RemLLive` (no created/cancelled node in `remove_labels`):
  `deletePendingNode_clears_remL`, `fixed3448_cancelled_node_clears_remL`.
* `PendingWithLabelsLive`: `pendingWithLabels_live`, `fixed3453_*`. Composed property
  `IncludePendingExact` in ops_aggregate.
* `CancelledTypeRegistered`: `Pending::commit` registers every cancelled relationship's type
  (`PC.stCancelTypes`, `stCancelTypes_registers`); `bug3447_cascade_drops_type_key` is the model of the
  earlier behaviour (the cascade drops the emptied group). #3447 is still open in the compact reply
  (redis_layer `RelLookup`).

## UNDER CONSTRUCTION constraints bind writes
`enforce_constraints` (pending.rs:1500) skips only FAILED constraints (as C's
`Schema_EnforceConstraints`), not every non-OPERATIONAL one. `PC.Constraint.operational : Bool`
became `status : CStatus`; `enforce_spec` now reads "passes iff every non-FAILED constraint's check
passes"; added `enforce_operational`, `enforce_underConstruction`, `enforce_skips_failed`.
-/
import PendingCommit.Span
import PendingCommit.Arena
import PendingCommit.Labels
import PendingCommit.Props
import PendingCommit.Snapshot
import PendingCommit.Stats
import PendingCommit.BinSearch
import PendingCommit.NameMap
import PendingCommit.Pack
import PendingCommit.Content
import PendingCommit.ContentMerge
import PendingCommit.ContentCompact
import PendingCommit.Store
import PendingCommit.StoreFold
import PendingCommit.StoreApi
import PendingCommit.StoreCodec
import PendingCommit.StoreMem
import PendingCommit.PendingAttrs
import PendingCommit.PendingState
import PendingCommit.PendingRels
import PendingCommit.PendingDeg
import PendingCommit.PendingDocs
import PendingCommit.PendingCommitOp
import PendingCommit.UniqueIndex
import PendingCommit.UniqueQuery
import PendingCommit.PendingCancel
import PendingCommit.CancelRust
import PendingCommit.CommitOpNext
import PendingCommit.StagedBugs
import PendingCommit.Hold
import PendingCommit.SaveRoaring
import PendingCommit.SaveNthLive
import PendingCommit.SaveSlice
import PendingCommit.SaveLayout
import PendingCommit.KeyBorrow
import PendingCommit.AttrInv
import PendingCommit.AccessorBridge
import PendingCommit.AttrCallers
import PendingCommit.AttrAlloc
