# U2: what "CCH acts as a native index (batched, MVCC-compliant)" means here

**Revisions.** `main` = origin/main 7df8e5887. `tip` = origin/cch/08c-build-catch-up 31ca1b802, the top of the open stack (#3360 → #3361 → #3362 → #3764 → #3622 → #3771 → #3773 → #3774 → #3363 → #3772). `proc` = origin/cch/09-procedure (#3364). C = civil-kitten origin/master a61fbb67a. Paths are under `graph/src/index/falkordb/` unless they start with `graph/` or `src/`.

## 1. How the native index builds in the background (main)

**Job lifecycle.** A writer's commit that brings in a `Building` index enqueues one job per (index, generation) (mvcc_graph.rs:211-221). A job runs one batch on the `falkordb-index-build` pool, then resubmits itself while that generation is still `Building` (online_build.rs:4-9, 322-330). The pool has `INDEX_BUILD_THREAD_COUNT` workers, default 1 (src/lib.rs:103). A job holds its graph weakly (online_build.rs:292-296).

**What a batch reads.** A batch reads `mvcc.read().new_build_version()`: a private copy of the published version, at the same data version (online_build.rs:406; tip graph/src/graph/graph.rs:1144-1148). It holds no lock while it reads (online_build.rs:13-14). It reads up to `budget` entities from the cursor and applies them to its own copy (versioned.rs:760-778). Batch sizes double from 10k to a 4M cap, sized by how many entities have been read so far (online_build.rs:94-147).

**Publishing.** A batch publishes under the GIL (src/graph_core.rs:1606-1614) through `commit_indexer_batch` (mvcc_graph.rs:262-288). `indexer_fit` (versioned.rs:504-546) picks one outcome:
- **Same:** nothing has committed since the fork, so the batch's version is published as it is.
- **Swap:** only other indexes' batches committed. The batch's entry is cloned into a new build version of the published version (versioned.rs:551-561).
- **CatchUp(Refresh | Insert):** writers committed. Refresh re-reads the written entities in the batch's copy; Insert puts the batch into the published copy. `REFRESH_COST = 10` decides which is cheaper (versioned.rs:183-218, 570-656).
- **Discarded(Gone | AnotherBatch | CatchUpTooLarge):** nothing is published, and the next batch starts from the published cursor (versioned.rs:220-234). TooLarge halves the next batch (online_build.rs:388-392). The catch-up limit is 10,000 changes, about 13 ms of lock time (online_build.rs:126-131).

A batch's commit keeps the data version, so it never makes a writer's fork stale (mvcc_graph.rs:247-248, 301-304). The replaced version is dropped after the lock is released. On an 8M-entity column that brought the GIL hold from 1.6 ms down to 0.01 ms (online_build.rs:348-360).

**Writes during a build.** A `Building` index holds exactly the current tuples of the entities below its cursor. Writers apply additions only below the cursor, and apply every removal (versioned.rs:29-39, 263-278). Writers record the ids they touch in `Written.by_this_version`. Publishing moves those ids into `since_last_indexer_step`, an `Arc<RoaringTreemap>` (versioned.rs:81-92, 436-443). The next batch's apply clears that set (versioned.rs:424-427), and the fit's catch-up reads it (versioned.rs:517-534).

**Writer forks are rebased.** A writer that forked as a reader checks for staleness when it escalates. If a batch moved a `Building` index on in the meantime, the writer takes the published entry and re-applies its own written ids (src/query_session.rs:251-278; mvcc_graph.rs:171-181; versioned.rs:664-742). The overtake is detected with the `batches_published` counter plus each entry's `generation` and `indexer_commits` (falkordb_index.rs:599-605, 894-912; versioned.rs:445-459, 692-708).

**What is versioned, and how it is copy-on-write.**
- `Graph::new_version` clones everything:
  - Plain matrices are `Cow` and are deep-copied on the first write (graph/src/graph/cow.rs:8-32).
  - `VersionedMatrix` and the tensors keep a base plus two deltas, `m/dp/dm` (graph/src/graph/graphblas/versioned_matrix.rs:1-22).
  - `falkordb_index` is cloned whole (tip graph.rs:1134-1135).
- A column is a copy-on-write B+-tree. A clone bumps one root `Arc`, and a write copies only the pages on its path (data_structures/cow_btree/mod.rs:9-13).
- A CCH entry is an `Arc<Cch>` (falkordb_index.rs:606-621).
- The rule that keeps this safe: a version that has been branched from is never written again (cow.rs:48-59). A published version is only ever shared (mvcc_graph.rs:63-66).

## 2. What the open CCH stack already does

- **#3771, background build.** `CREATE` checks the weights under the lock (about 5 ms on California) and holds the entry `Building` (tip graph.rs:5429-5439; cch/input.rs:150-165). CCH becomes a second kind of index (`IndexId::Cch`, `Prepared::Cch`). **One batch builds everything**: input, ordering, customization. It has `Cursor = ()`, ignores its budget and returns `next = None` (tip cch/mod.rs:454-458, 595-612, 674-689). It checks for the builder's stop between steps (tip online_build.rs:487-492).
- **#3773, scoped recustomization** (cch/partial.rs). The result is bit-identical to a full customization: 10.7 ms against 464 ms on a 200x200 grid (PR text). It returns `Scoped`, `NeedsFull` past n/8 changed arcs, or `NeedsRebuild` (tip cch/mod.rs:351-363).
- **#3774, chunked copy-on-write metric.** Weights are held in chunks of 2,048 behind two levels of `Arc` (tip cch/customize.rs:98-127). On California a held index costs +396–412 MB, against +365 MB with flat arrays (PR text).
- **#3363, maintenance through a change set** (cch/maint.rs:1-67):
  - Writes record into `IndexDocs.cch`.
  - A built index classifies each new or reweighted edge with `Cch::joins` (tip customize.rs:645-663).
  - A deletion is never structural. Past 1,024 changed pairs, only "many" is kept (maint.rs:82-95).
  - The changes are flushed once per `Commit` segment, on the private version (runtime/pending.rs:1473-1476; maint.rs:467-536). A flush is one of: a scoped recustomization (0.22 ms p50 per SET on California, PR text); a full customization (about 0.5 s); or, for a structural change, `rebuild_cch_in_background`, which sets the entry to `Pending`, `Building`, under a new generation (tip falkordb_index.rs:1286-1296).
- **#3772, catch-up after a build.** The trait gains `Absorbed { Yes, Restart }` (tip versioned.rs:245-252). A CCH fit always catches up by Refresh and is never limited (tip versioned.rs:165, 545). Both the catch-up and the writer rebase run `maint::absorb` (tip cch/mod.rs:517-573, 646-668; maint.rs:598-630). On `Restart`, a build keeps its generation, and a writer's rebase gives the index a new one (tip versioned.rs:701, 795). The fifth build overtaken this way marks the index `Failed` (cch/mod.rs:394-399).

**What is still rebuilt from scratch.** A rebuild runs input, ordering, elimination game and customization: 4.0 s on California (#3363 text). The triggers:
1. `CREATE`.
2. Any structural write to a built index: an edge between two nodes the hierarchy does not join, or an endpoint at or past its n (maint.rs:22-27, 551-553).
3. A flush error other than an invalid weight (maint.rs:502-510).
4. A structural write that lands during a build, or that overtakes a writer's fork (`Restart`).
5. An RDB load, since only the definition is saved (cch/mod.rs:375-377).

**What is partial.** Weight changes and deletions: a scoped or full customization over the same `Topology`.

**What is missing.** C rebuilds once deletions reach n/8 (at least 1,024), to clear out stale fill-in (C src/index/cch_index.c:468-496). A grep of tip's cch/ finds no such check.

**While a rebuild runs**, every new version holds `Pending`, and the procedure answers with a plain Dijkstra search (proc cch/procedure.rs:224-233). That search takes a 222 ms median on California (proc cch/fallback.rs:28-31). Readers of older versions keep the old `Arc`.

## 3. Where an incremental ordering or supergraph would plug in

- **Where writes are recorded** (tip graph.rs):

  | Write method | Line | Records with |
  |---|---|---|
  | `create_relationships_bulk` | 2785 | `edges_created` |
  | `set_relationships_attributes` | 2926 | `edge_attrs_set` |
  | `set_relationships_attributes_of_type` | 2961 | `edge_attrs_set` |
  | `delete_relationships` | 3070 | `edges_deleted` |
  | `delete_implicit_edges` (a node delete's cascade) | 3257 | `edges_deleted` |

  Each records `(edge id, src, dst)` per `TypeId`, for the indexes that span that type (maint.rs:216-325). Creating a node records nothing (maint.rs:24-27). Each index gets a `CchChanges { structural, overflow, pairs, edges }` (maint.rs:97-112). A `Building` entry keeps only the edge ids (maint.rs:297-314).
- **The one place a structural change becomes a rebuild.** `update` returning `(None, Rebuild)` (maint.rs:551-553, 574-582) and `absorb` returning `None` (maint.rs:621-623). An incremental supergraph update would replace exactly these branches.
- **Bounded batches.** The framework already builds in several batches; CCH opts out with `Cursor = ()`. To build in slices, CCH needs:
  - an ordered cursor and a `place(id)` function, so `writer_record` knows what is already in (versioned.rs:263-278);
  - a `Batch` type and an `indexer_add` that merges into a copy-on-write partial state;
  - a real `batch_len` and `limit_catch_up`, so `CatchUpTooLarge` can halve a batch;
  - a progress count from `index_build_read`, which returns 0 for CCH today (tip versioned.rs:858-861), so `Sizing` can grow the batches.
- **How a hierarchy is held per version.** `IndexEntry<CchIndex>` → `Hierarchy::Built(Arc<Cch>)` → `Arc<Topology>` plus a chunked copy-on-write `Metric` (tip cch/mod.rs:318-324, 365-392; customize.rs:242-259, 605-615). The `Topology` is six immutable boxed slices (order.rs:127-138).
- **Arc ids must stay put.** An arc id is a position in the flat upward CSR, and the metric is indexed by it (order.rs:114-128). Inserting one fill-in arc shifts every later id, so every later metric chunk would change. An incremental supergraph needs ids that do not move (for example, ids that are only ever appended), or #3774's chunked copy-on-write stops saving anything.
- **What a reader on an old snapshot needs.** It keeps the old hierarchy at no cost as long as:
  - (a) all incremental work happens on a private version (a writer's fork or the builder's version) and never writes into anything a published version shares (cow.rs:48-59; `WeightArr::set` copies its chunk through `Arc::make_mut`, customize.rs:182-189);
  - (b) every GraphBLAS object the index holds is fully computed, with no pending work, before it is published. A reader's GraphBLAS call on an object with pending work finishes that work, which writes to an object other readers share (mvcc_graph.rs:332-347; main graph.rs:2805-2834). `prepare` waits only on the graph's own matrices today;
  - (c) the replaced version is dropped outside the GIL (online_build.rs:348-360).

  After a structural write, a new version cannot keep answering from the old hierarchy: it would miss the new edge. That is why the Dijkstra fallback exists.

## 4. Constraints a design must respect

- **Memory per version.**
  - A column pays for the pages a write copies.
  - A CCH index pays for the chunk table (181 KB) plus the chunks it touches (median 0.23 MB) for each version that writes (customize.rs:117-125).
  - A plain `Cow` GraphBLAS matrix is deep-copied whole on its first write (cow.rs:10-14). Only `VersionedMatrix` costs O(|delta|) per write, with a square-root rule for when to fold the deltas in (versioned_matrix.rs:151-171). A GraphBLAS supergraph that changes on writes must therefore live as `m/dp/dm`, not as a plain `Cow<Matrix>`.
  - `GRAPH.MEMORY USAGE` counts only the published version; a build in progress is not counted (main src/commands/memory.rs:73-83; tip cch/mod.rs:626-634).
- **The written set keeps growing.** Every writer's commit to a `Building` index copies `since_last_indexer_step` (`Arc::make_mut`, versioned.rs:436-443; online_build.rs:162-165). With one batch per CCH build, that set grows for the whole build, so each write costs O(|written|). Unmeasured for CCH.
- **GIL hold time.**
  - A column batch publishes in about 0.01 ms, and its catch-up is capped at about 13 ms.
  - A CCH catch-up or rebase has no limit: a full customization runs under the GIL, about 0.47 s on California (#3772 text). #3772 also notes that the catch-up walks the edges written before the build read the graph, and that this is still to be measured.
  - A design should keep only O(written) classification under the GIL and push anything larger into a batch (`Restart`, or a `TooLarge`-style split).
- **What runs on the builder versus the writer.**
  - The builder: anything on its own build version, with no locks and no Redis calls (the `BuildHost` trait, online_build.rs:44-63). It checks `stopping()` between long steps.
  - The writer: classification and scoped recustomization; "no hierarchy is ever built in the write path" (#3363). But past the 1,024-pair cap, a full customization (about 0.5 s) still runs on the writer's version, and on a replica it runs inside `GRAPH.EFFECT` on Redis's main thread (#3363, "Open").
  - With one builder thread by default, a multi-second CCH build delays every column build behind it (#3771, "Open").
- **Folding the deltas versus copying them.**
  - Reading the matrices goes through `effective` = `(m ∖ dm) ∪ dp`. When both deltas are empty it borrows `m`; otherwise it builds a private merged copy (versioned_matrix.rs:88-149).
  - The team draft measured this on a 9M-edge grid (.handover/cch-background-fold/discussion.md):
    - The time is a wash: 76.5 ms with the copy against 75.4 ms folding on the builder's fork.
    - The merge needs about 400 MB above the 710.5 MB graph either way.
    - A fold raises the peak by up to 148 MB while a reader holds the previous version.
    - A fold survives only when the batch publishes as `Same`.
    - The draft rules out folding on the published version (readers reach it with no lock) and folding on the writer (about 14 ms per write at 9M edges).
  - For a batched design: if every batch builds `effective` for a whole relationship type again, that ~400 MB peak and ~76 ms come back with every batch. A batch working on one slice should read only its rows instead: a masked extract over `m/dp/dm`, or the three-way merge iterator, which never builds the merged matrix (versioned_matrix.rs:63-70). Unmeasured.
- **The data version.** An index publish must keep the data version, or every writer in flight goes stale (mvcc_graph.rs:247-248, 301-304).

**Unknown:**
- The peak memory of a CCH rebuild on California (the old hierarchy held by readers, the new one, `CchInput`, and `effective` together) has not been measured.
- It is not known whether GraphBLAS matrices held by the index would need to join the pre-fork `wait_all` (main graph.rs:2836-2848). Today the RDB saves only the definition.
