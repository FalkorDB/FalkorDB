/-
# Pending mutations, Commit, and the attribute store

A Lean model of how a FalkorDB-rs write query stages mutations in
`runtime::pending::Pending`, reads them back, and applies them at `Commit`; of
the attribute store they land in; and of the deleted-entity snapshots later
clauses read through. 322 theorems, no `sorry`/`admit`/`axiom`. Coverage per
Rust function: `COVERAGE.tsv` — every function of `attribute_store.rs` (66)
and `pending.rs` (63 non-test) is PROVEN. Repros: `graph/tests/lean_pending_commit.rs`
(asserting C's behaviour / the agreed semantics, so each FAILS today).
Line numbers are origin/main (`2c874022a`).

## Re-target to 2c874022a (#2846: the graph's `IdSpace` owns the ids)

| Lean | Rust |
| --- | --- |
| `PS.P` (no `nodeEntry`/`relEntry`) | `struct Pending` (pending.rs:135): the boundary fields and the id-space builders are gone |
| `PS.rmStep` (+ `taken` removal), `removeRelsG` | `remove_pending_relationships_for_node(id, g)` (:710): `taken_relationship_ids.remove` + `g.cancel_relationship_id(rel_id)?` (:752-755) |
| `PS.deletePendingNode` | `delete_pending_node(id, g)` (:659): unwinds and hands the id back via `g.cancel_node_id(id)?` (:696) |
| `PD.endSegment`, `clear` | `end_segment` (:1613), which replaced `clear` + `open_id_boundaries` |
| `CommitOpNext.*` | `CommitOp::next` (ops/commit.rs:69) with `end_segment` (:135-143) |

New theorems: `removeRelsG_spec`, `foldl_rmStep_taken`, `deletePendingNode_spec`,
`deletePendingNode_refused`, `endSegment_spec`, `commitNext_ok`,
`commitNext_commit_err`, `commitNext_endSegment_err`, `pops_in_order`;
`removeRels_spec` now also states `taken` loses the cascaded ids. The graph's
cancel is the hypothesis `PS.CancelSpec` (= `IdSpace::cancel` inserts into the
free set; proofs/id_space, and `IdSpaceContract.cancel_ok` in proofs/graph_queries).

Bugs re-checked live at 2c874022a (Rust release module vs C `bin/macos-arm64v8-release`):
* #2876 / W2-pending-1 — **still present**: `MATCH (n) DELETE n WITH count(*) AS c
  CREATE (m {v:2}) WITH m MATCH (x) RETURN id(x), x.v, labels(x)` over `(:A {v:1})`:
  Rust `0|1|[A]`, C `0|2|[]`; the MERGE shape still creates 2 nodes (C 1).
  `rust_reads_dead_entity` stands: `reserve` still reclaims the free set first.
* W2-pending-2 — **still present**: `CREATE (a) DELETE a` Rust no counters, C
  `Nodes created: 1, Nodes deleted: 1` (`c_eq_rust_plus_cancelled`).
* #2769 — **still present**: `CREATE (a)-[r:R]->(b) DELETE r SET b.d = indegree(b), a.o =
  outdegree(a) RETURN b.d, a.o` panics `relationship 0 not found` at graph.rs:3266
  (server killed); C `0|0` (`deg_panics_on_pending_deleted`).
* W4-fn-1 (labels staged before DELETE) — **still present**: `MATCH (n:A) SET n:C DELETE n
  RETURN labels(n), n:C` Rust `[A]|false` (the snapshot copies the committed labels,
  ops/delete.rs:284, :464); note C answers `[]|false` here.

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
| `Labels.stage/removeLabels/nodeHas/overlay` | `stage_node_labels` (pending.rs:529), `remove_node_labels` (:565), `node_has_label` (:590), `update_node_labels` (:618) |
| `Labels.removeOp`                 | `Runtime::remove` label branch (ops/remove.rs:128-139) |
| `Labels.commitLabels`             | `set_nodes_labels_bulk` + `remove_nodes_labels` in `Pending::commit` (pending.rs:1126-1146) |
| `Props.upsert`, `stageAll`        | `set_node_attribute` / `set_relationship_attribute` (pending.rs:440, :791) |
| `Props.readQ`, `normQ`            | `get_node_attribute_no_delete_check` (runtime.rs:1479) + evaluator's null |
| `Props.commitNew`                 | `import_node_attrs` → `import_attrs` (graph.rs:1765, attribute_store.rs:1351) |
| existing-entity commit            | `set_nodes_attributes` → `insert_attrs` → `merge_span` (graph.rs:1733) |
| `Props.replaceFromMap/FromNode`   | `SET n = map` / `SET n = m` (ops/set.rs:185-224) |
| `Snapshot.RS`, `readRt`           | `Runtime::deleted_nodes/_relationships` and every snapshot-first accessor (runtime.rs:1459-1789) |
| `Snapshot.del/commit/createRust`  | `delete_nodes_bulk`/`delete_entity` (ops/delete.rs), `CommitOp` (ops/commit.rs), `IdSpace::reserve` bin-first |
| `Snapshot.createDeferred`         | the fix proposed in #2876 |
| `Stats.step`, `Stats.P`           | `created_nodes`, `delete_pending_node` (pending.rs:621), commit counters (:1105, :1190) |

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
| `EvictPR2964` | **models open PR #2964, not `main`**: evict-on-reissue for #2876 |

Headline theorems: `bsearch_ok_iff`/`bsearch_err`; `getOrCreate_spec`;
`pack_unpack`, `pack_hinv`, `release_hinv`, `compactHeap_unpack`;
`setSpanC_ok`, `mergeSpanC_ok`, `compactC_ok` (each slot reads exactly the
intended span, every other slot is unchanged, `CInv` holds); `dsetSpan_ok`,
`dmergeSpan_ok`, `dremove_ok`, `dtrim_ok`; `runOps_spec` →
`insertAttrs_spec`/`importAttrs_spec`/`importResolved_spec`/`removeAll_spec`;
`encode_decode_roundtrip`; `structural_partition` (structural + per-entity
bytes = whole allocation, no underflow); `rustUpsert_eq` (the Rust upsert is
`Props.upsert`); `setAttr_spec`, `getA_spec`, `updA_spec`; `removeRels_spec`;
`createdRel_spec`; `deletedDeg_none`; `absorb_spec`, `resync_spec`,
`clear_spec`; `scan_ok`, `check_spec`, `enforce_spec`, `commit_spec`;
`evict_preserves_Good`, `evict_reads_live_entity` (PR #2964).

Wave-4 findings:
* `deg_panics_on_pending_deleted` — `pending_deleted_*degree` asks the
  committed graph for the endpoints of every pending-deleted edge, including
  one created in the same batch: `CREATE (a)-[r:R]->(b) DELETE r SET b.d =
  indegree(b)` kills the Rust server (`relationship 0 not found`,
  graph.rs:3266); C `0`. Already #2769 (re-found; reproduced live on origin/main
  code).
* `setSpan_u16_wrap` — `set_span` stores `n as u16`; with ≥ 65,536 entries for
  one entity the slot reads empty while the arena keeps the entries no counter
  covers (`Arena.Inv` false, `compact`'s `debug_assert_eq!` would fire). Only
  `import_attrs_resolved` (GRAPH.BULK, header with ≥ 65,536 columns) and
  `decode_with_count` (malformed RDB) can pass that many, as neither
  deduplicates ids; `setSpan_cast_exact` shows strictly ascending ids below
  `ATTRIBUTE_ID_NONE` always fit. Not reproduced (needs a 65,536-column bulk
  payload).
* `resolved_duplicate` — `import_attrs_resolved` keeps duplicate ids
  (`GRAPH.BULK` header repeating a column name): two entries for one id are
  stored, reads return the second (binary search), `attr_count` says 2. C
  stores duplicates too (`AttributeSet_AddNoClone` only asserts in debug); not
  reproduced live.
* Labels staged before a `DELETE` are missing from the deleted snapshot
  (ops/delete.rs:284, :464) — confirmed by another agent; the repro tests
  `bug_deleted_snapshot_*` / `bug_detach_deleted_snapshot_*` are in the test
  file.

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
  `labels_removed` counts each once.
* `stage_sorted_lastWrite` — staged properties stay strictly sorted and hold
  the last write per attribute; `read_your_writes_existing`,
  `read_your_writes_new` — what the query read before Commit is exactly what
  the store holds after, for existing and newly created entities.
* `sound_under_Good`, `del_Good`, `deferred_preserves_Good`,
  `deferred_reads_live_entity` — snapshot reads are correct as long as no
  snapshotted id is live, and deferring id reuse to query end (the #2876
  proposal) maintains that across delete/commit/create.
* `c_eq_rust_plus_cancelled` — C's `Nodes created/deleted` = Rust's +
  |cancelled| on every well-formed segment.

## Counterexamples (Lean) → confirmed bugs (Rust, and vs C on live servers)

1. **#2876 has more shapes than the issue lists** (`rust_reads_dead_entity`).
   The id-keyed snapshot shadows *any* live entity that inherits the id, through
   every accessor, for nodes *and* relationships, and even when nothing keeps the
   deleted entity reachable (`WITH count(*)`). Rust vs C:
   - `MATCH (n) DELETE n WITH count(*) AS c CREATE (m {v:2}) WITH m MATCH (x) RETURN id(x), x.v, labels(x)`
     over `(:A {v:1})`: Rust `0|1|[A]`, C `0|2|[]`.
   - `... CREATE (m:B {v:2}) WITH m MATCH (x) WHERE x.v = 2 RETURN count(x)`: Rust 0, C 1.
   - `... CREATE (m:L {v:2}) WITH m MERGE (z:L {v:2})`: Rust creates a **duplicate**
     (2 nodes created, 2 `:L {v:2}` stored), C 1 — persistent damage.
   - `... CREATE (m:B) RETURN labels(m), m:B, m:A`: Rust `[A]|false|true`, C `[B]|true|false`.
   - relationships: `MATCH ()-[r]->() DELETE r WITH count(*) AS c MATCH (p:P),(q:Q) CREATE (p)-[m:S {v:2}]->(q) RETURN m.v, type(m), labels(startNode(m)), labels(endNode(m))`:
     Rust `1|R|[X]|[Y]`, C `2|S|[P]|[Q]`.
   - also `SET m.v = 3 RETURN m.v` → 1 (stored value is 3), `properties(m)`,
     FOREACH-created nodes, and DETACH-cascaded edges.
   Tests `bug_reused_id_*`, `bug_reused_relationship_id_reads_deleted_edge`.
   Consequence for the fix: "refuse the query when a *reachable* snapshotted id
   is reclaimed" is not enough (the unreachable case); deferring reuse, or
   dropping snapshot entries no binding references, is.
2. **Cancelled entities vanish from statistics** (`cancel_example`).
   `CREATE (a:L {v:1, w:2}) DELETE a`: Rust reports nothing but `Labels added`;
   C `Nodes created: 1, Properties set: 2, Nodes deleted: 1`.
   `CREATE (a)-[:R]->(b) DELETE a`: Rust `Nodes created: 1`; C 2/1 nodes, 1/1 edges.
   Inconsistent within Rust too: a pending edge deleted directly is counted
   created + deleted. Fix: add `cancelled_nodes.len()` /
   `cancelled_relationships.len()` (and their staged props/labels) to both sides
   (pending.rs:1105/1147/1226). Tests `bug_cancelled_*`.
3. **Writes to a deleted relationship are applied and counted.** `REMOVE r.v`
   after `DELETE r` (ops/remove.rs:141 has no deletion check for relationships;
   nodes have one at :121) → Rust `Properties removed: 1`, C none. `SET r.v = 2`
   after `DETACH DELETE a` cascades `r` → Rust set 1/removed 1, C none: the
   cascade is recorded only in `Runtime::deleted_relationships`, which set.rs:230
   does not consult. Test `bug_writes_to_deleted_relationship_are_counted`.

Also confirmed vs C (stats only, no Rust test): property counters are net at
commit in Rust and per clause in C — `CREATE (n {v:1}) SET n.v = 2` Rust set 1,
C set 2/removed 1; `MATCH (n) SET n.v = 2 SET n.v = 1` Rust set 1/removed 1, C
none; `UNWIND range(1,3) AS i SET n.v = i` Rust 1/1, C 3/3; `REMOVE n:A SET n:A`
Rust none, C added 1/removed 1.

Re-found but already known: `SET x = m` / `SET r = {…}` keep staged attributes
(#2776, `replace_keeps_staged`), `Labels added` counts schema labels (#2655),
plain DELETE detaches (#2573), #2876 shapes A/B.

## Suspicions (not confirmed)

* `effects_count` (pending.rs:1699) ignores `cancelled_nodes`, so a query that
  only cancels (`CREATE (a) DELETE a`) ships no effect although its id went to
  the master's bin; the replica's boundary is then one lower. By hand this
  self-heals on the next create; not run against a replica.
* `import_attrs_resolved` / `decode_with_count` do not filter nulls resp.
  deduplicate attribute ids before `set_span`, which assumes a strictly sorted
  null-free span; reachable only via GRAPH.BULK / a malformed RDB.
* Deleted-node snapshots take labels from the committed matrix only
  (delete.rs:263, :305), ignoring labels staged in the same query; C returns
  `[]` for any deleted node's labels, so this is a divergence either way.

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
-/
import PendingCommit.Span
import PendingCommit.Arena
import PendingCommit.Labels
import PendingCommit.Props
import PendingCommit.Snapshot
import PendingCommit.Stats
import PendingCommit.EvictPR2964
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
import PendingCommit.PendingCancel
import PendingCommit.CommitOpNext
