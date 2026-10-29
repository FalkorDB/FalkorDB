# search_concurrency: snapshot isolation, deadlock freedom and the RediSearch document path

This project proves the concurrency guarantees FalkorDB-rs gives its clients, and shows with counterexamples where they used to break. It models the Redis main thread, the worker pool, the per-graph write slot, the GIL, `GRAPH.DELETE` and the `BGSAVE` fork as a transition system, and proves properties that hold over **every** interleaving. It also covers the RediSearch side of indexing: what `Document::set` sends, index option parsing, index population tickets, and vector KNN queries.

User-facing behaviour that depends on it:

- **Snapshot isolation.** A read query sees one committed version of the graph, never half of a write. Concurrent writes are serialized and none is lost.
- **No hang.** The server must stay responsive however many queries are queued, and lock ordering must never deadlock.
- **Inline writes.** A write inside `MULTI`/`EXEC` (or a Lua script) runs on the main thread; it must succeed even while a queued write is in progress.
- **`GRAPH.DELETE` and `BGSAVE`** while queries run.
- **Indexes.** `CREATE INDEX` must populate the index (also a second index on the same label), an entity must never drop out of an index because of a bad vector, and `db.idx.vector.queryNodes` must not crash on a huge `k`.

Each module's header has its Lean↔Rust `file:line` table; the root [`SearchConcurrency.lean`](SearchConcurrency.lean) lists the modules, the theorems and the bugs. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Can a reader see a partial commit, or a write get lost?**

No. `Mvcc.lean` models each atomic step of the Rust (read, claim the write slot, fork, escalate to the GIL, mutate, commit, rollback, delete, BGSAVE fork) and lets any enabled step fire in any order. Over every reachable state:

- `snapshots_are_committed`: every snapshot a reader or fork child ever took is a version some commit published whole;
- `no_lost_writes`: the committed state is exactly the acknowledged writes, in acknowledgement order;
- `history_linear`: every published version is a prefix of the current one;
- `private_on_committed`: a writer's private version is always based on the current committed one, so there is no lost update;
- `one_commit_per_version`: at most one commit lands on each version.

**2. Can an inline write fail because a queued write is in progress?**

- **Before #3290, yes.** A queued write claimed the write slot while still matching, as a reader. A `MULTI; GRAPH.QUERY g "CREATE (:M)"; EXEC` arriving then was refused with "another write is in progress" (counterexample removed; in 17ccc5412); C creates the node (`multi_write_ok_in_c`).
- **On main, no.** The queued write builds on an unclaimed fork and claims the slot only after it holds the GIL. If an inline write committed in between, the claim is stale and the queued write re-runs (`claimFork_stale`, `stale_then_rerun`); the inline write succeeds (`inline_write_ok`). `reader_cannot_claim` proves a reader can no longer take the slot.

**3. Can the server deadlock?**

- **Lock order.** `no_cycle` proves threads that acquire locks in rank order cannot form a wait-for cycle. `all_programs_ordered` checks that every locking path (the writer including escalation, the inline writer, index population, background index drop, the fork preparation, telemetry flush) obeys GIL < graph < indexer < registry. `bad_escalate_rejected` shows the checker catches a wrong order.
- **RediSearch's locks (gap closed 2026-10-20, #3226).** The ranks now include RediSearch's global `RWLOCK` and per-spec lock, which an index scan holds from opening to dropping. A write that commits with its scan still open re-takes the spec lock as a writer (`bug3226_open_scan_upgrade`, a live abort) and takes the GIL while holding RediSearch's locks (`bug3226_open_scan_unordered`, `bug3226_restore_stall_cycle`). Read scans, population, `RESTORE` and writers whose scan closes before the commit are ordered (`spec_programs_ordered`).
- **The write queue hand-off.** `drain_no_lost_message` checks exhaustively, for 2 producers, that the `write_loop` flag protocol never strands a queued write; `recheck_needed` shows it would without its `is_empty` re-check.
- **Queue overflow.** Before #3014, more than 1024 queued queries hung the server forever (counterexample removed; in 17ccc5412). On main, `never_stuck_3014` and `main_never_blocks` prove the main thread never waits and some step is always possible, and `flood_drains` that a flood drains.

**4. Is a new index always populated?**

- **Before #3086, not a second one on the same label.** After `CREATE INDEX … (n.a)` and then `(n.b)` on a label, the new job saw two pending tickets and bailed, the old job finished with its `{a}` field list, and the index reported OPERATIONAL with `b` in no document (Bug 2 below; counterexample removed; in 17ccc5412).
- **On main, yes.** `create_index` now bumps the generation whenever the label already has an index, so the old job fails `is_ticket_current` at its next batch and the new job indexes everything. `main_operational_complete` proves, over every interleaving of any number of `CREATE INDEX`es on a label with the batches of every populate job, that an OPERATIONAL index has every field in every batch. The proof is an inductive invariant (`MInv`, `reach_inv`): at most one job owns the current generation, it holds the current field list, and the batches before its cursor are complete. `main_lost_order_complete` replays the interleaving that used to lose `b`.
- **One remaining window.** The theorem excludes the moment between `Indexer::create_index` storing the entry and `populate_index` taking its ticket. In that window the entry has no pending ticket, so it looks OPERATIONAL while the new field is in no document (`store_window_operational_incomplete`). The window is a few instructions long and has not been seen on a live server.

The ticket counters themselves are proven sound (`inc_inv`, `dec_inv`, `bump_inv`, `release_exact`, `operational_iff`, `stale_release_isolated`).

**5. Can a bad vector remove an entity from its other indexes?**

- **Before #3087, yes.** A vector of the wrong dimension made RediSearch reject the whole document, so the node vanished from every index on its label (counterexample removed; in 17ccc5412).
- **On main, no.** `docSet_vector_sound` proves a vector is added only with the field's own non-zero dimension, for any value and any options; `doc_always_accepted` and `other_fields_never_dropped` prove the document is accepted and the other fields stay.

**6. Can a KNN query with a huge `k` crash the server?**

- **Before #3088, yes.** `db.idx.vector.queryNodes('L', 'v', 1000000000000000, …)` allocated `k` entries up front and crashed (counterexample removed; in 17ccc5412).
- **On main, no.** `k` is clamped to the entity count: `alloc_bounded` proves every buffer is at most the live entity count, and `candidates_topk` that the result is still the top `k` (the query from #3085 returns its one row, `huge_k_one_row`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `snapshots_are_committed` | Readers and the BGSAVE child only see whole committed versions | `Mvcc.lean` |
| `no_lost_writes`, `history_linear` | Committed state = acknowledged writes, in order; history only grows | `Mvcc.lean` |
| `one_commit_per_version`, `inline_write_ok` | Writes are serialized; since #3290 an inline write is never refused by a queued one | `Mvcc.lean` |
| `delete_blocks_writes`, `fork_excludes_commit` | After `GRAPH.DELETE` no write can proceed; a fork never runs during a commit | `Mvcc.lean` |
| `refused_commit_releases_slot` | A commit refused by `Graph::validate` publishes nothing and frees the slot | `Mvcc.lean` |
| `no_cycle`, `all_programs_ordered` | The lock order rules out deadlock, and every locking path follows it | `Locks.lean` |
| `drain_no_lost_message` | The `write_loop` hand-off never strands a queued write (2 producers, exhaustive) | `Locks.lean` |
| `main_operational_complete` | Since #3086 an OPERATIONAL index has every field populated, over every interleaving of index creations and populate batches | `PoolWithin.lean` | `threadpool.rs` after #3502: named workers, `shutdown_within` (`within_in_time`, `detached_workers_exit`) and its two open bugs |
| `Tickets.lean` |
| `never_stuck_3014`, `main_never_blocks` | Since #3014 the main thread never blocks on a full queue | `Locks.lean` |
| `release_exactly_once`, `count_eq_live` | Every admitted query releases its queue slot exactly once; the counter equals live queries | `Pool.lean` |
| `shutdown_drains` | Pool shutdown runs every queued job once, in order, then all workers exit | `Pool.lean` |
| `docSet_vector_sound`, `other_fields_never_dropped` | A bad vector can no longer drop the entity from other indexes | `Search.lean` |
| `alloc_bounded`, `candidates_topk` | KNN never allocates more than the entity count and still returns the top `k` | `Knn.lean` |

### How `snapshots_are_committed` is proved

1. **Model the state.** `S` holds the committed version `pub` and its version number, the write slot (owner, the base it was claimed on, its pending writes), the unclaimed forks, who holds the GIL, whether the graph is registered, and ghost history: every published version (`hist`), every snapshot taken (`seen`) and every acknowledged write (`acked`). A version is the list of writes it contains.
2. **Model the Rust steps.** `step` has one case per event, each one atomic step of the Rust: `read` clones the committed `Arc`; `claim` is the compare-and-swap on the write flag; `commit` runs `Graph::validate` (a parameter, so every theorem holds for any validation) and then swaps the `Arc`; `delete` and `fork` need the GIL on the main thread.
3. **Any interleaving.** `Reach` is every state reachable from `init` by any sequence of enabled steps.
4. **One inductive invariant.** `Inv` says: `hist`'s head is `pub`, the slot's base is `pub`, everything `seen` is in `hist`, `acked = pub`, every historical version is a prefix of `pub`, and forks are no newer than `pub`. `inv_step` proves each event keeps it; `reach_inv` lifts that to every reachable state. The headline theorems are fields of `Inv`.

The model abstracts what a write does to a version (writes are opaque), the per-graph `RwLock` and the matrices (proven elsewhere), and assumes sequentially consistent memory for the atomics.

## Bugs found

All reproduced on live servers, Rust against C (scripts in `repro/`).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 (W3-conc-1) | removed (17ccc5412) | Over 1024 queued queries: the main thread, holding the GIL, blocks on the full pool queue while workers wait for the GIL. Permanent hang, `PING` times out; C completes | Fixed by #3014 (issue #3013): `never_stuck_3014`, `acquire_bounded`, `release_exactly_once` |
| 2 | removed (17ccc5412) | A second index on a label was never populated: `CREATE INDEX … (n.a)` then `(n.b)` on 300k nodes, and `n.b < 300000` through the index returned 0 rows (C: 300000). `repro/poprace.py` (writes racing population) was attributed to the same mechanism; it has not been re-run on main | Fixed by #3086 (issue #3069): `main_operational_complete`, `main_lost_order_complete` |
| 3 (W3-conc-3) | removed (17ccc5412) | A wrong-dimension vector removed the node from every index on its label | Fixed by #3087 (issue #3075): `docSet_vector_sound`, `vector_dim_mismatch_keeps_range_entry` |
| 4 (W3-conc-4) | removed (17ccc5412) | A huge KNN `k` crashed the server | Fixed by #3088 (issue #3085): `alloc_bounded`, `huge_k_one_row` |
| 5 | removed (17ccc5412) | An inline (`MULTI`/`EXEC`) write failed with "another write is in progress" while a queued write was matching. C creates it | Fixed by #3290 (issue #3093): `inline_write_ok`, `one_commit_per_version` |
| #3226 | `bug3226_open_scan_upgrade`, `bug3226_open_scan_unordered`, `bug3226_restore_stall_cycle` (properties `noReacquire`, `LockOrdered`) | A write that commits while its own index scan is still open (`MATCH (n:W) WHERE n.v > 0 WITH n LIMIT 10 SET n.v = n.v + 100000`) takes RediSearch's spec lock as a writer while it holds it as a reader. The server aborts with `attempted to upgrade a spec read lock to a write lock`; C returns 10. The same path takes the GIL while it holds RediSearch's global lock, which makes a wait-for cycle with a `RESTORE` on the main thread. Paths that close the scan before the commit stay ordered (`spec_programs_ordered`). The BGSAVE-child half of the issue (a `pending_slots` mutex inherited across the fork) is not modelled | **Open** |

Index option validation also differed from C: missing `dimension`/`similarityFunction` were accepted and `'Euclidean'` refused. That was fixed by #3094 (`rust_refuses_missing`, `rust_eq_c`). Two DDL-only differences remain and are pinned by `#guard`s: `OPTIONS {dimension:0, similarityFunction:'bogus'}` creates an index (C refuses), and fulltext `OPTIONS {weight:1.0, foo:true}` is refused (C creates it; `rust_keys_ok_imp_c`).

A 40-second stress run (`repro/stress.py`: 3 writers, 6 readers, `GRAPH.DELETE`, `BGSAVE` every 0.3 s) found no partial commit, non-monotonic read, gap, duplicate or lost write, on Rust or C.

## Files

| File | What's in it |
|---|---|
| `Mvcc.lean` | The transition system: `S`, `Ev`, `step`, `Reach`, the invariant `Inv`; snapshot isolation, no lost writes, #3290's fork-and-claim, delete and fork exclusion |
| `Locks.lean` | Lock-order deadlock freedom (`no_cycle`), the `write_loop` hand-off (`drain_no_lost_message`), the pre-#3014 deadlock and the #3014 model (`never_stuck_3014`) |
| `Pool.lean` | `threadpool.rs` and #3014's `QueueSlot`: `shutdown_drains`, `acquire_bounded`, `release_exactly_once`, `count_eq_live` |
| `Tickets.lean` | Index population tickets (`Slots`): counter soundness, `main_operational_complete` (#3086) |
| `Search.lean` | `Document::set` with RediSearch as a hypothesis structure, vector and fulltext option parsing: `docSet_vector_sound`, `doc_always_accepted`, `rust_eq_c` |
| `Knn.lean` | `vector_query_nodes`/`vector_query_edges` with the #3088 `k` clamp: `alloc_bounded`, `candidates_topk` |
| `Once.lean` | Set-once globals in `thread_id.rs` and `storage/registry.rs` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

No path dependencies.

## Coverage

`COVERAGE.tsv` has 119 rows, from `src/graph_core.rs`, `src/query_session.rs`, `graph/src/graph/mvcc_graph.rs`, `threadpool.rs`, the index files and a few others:

| Bucket | Rows |
|---|---|
| PROVEN | 69 |
| AXIOMATISED | 20 |
| MODELLED | 17 |
| NOT COVERED | 13 |

- **AXIOMATISED:** thin wrappers over the Redis module API (blocking and unblocking clients, thread-safe contexts, the GIL lock and unlock, replication, string helpers), and `Index::add_document`, whose RediSearch behaviour is the `RediSearch` hypothesis structure in `Search.lean` (a whole document is rejected or accepted).
- **MODELLED:** functions that appear as a step of the transition system without a theorem of their own: `MvccGraph::new`/`from_graph` (`init`), `register_graph`, `execute_query`/`execute_profile`/`profile_sync` (a read on a snapshot), the `QuerySession` constructors, lock release and drop order, `must_run_inline`, and the `Field` constructors, and `CommitOp::next` (its lock path only, #3226).
- **NOT COVERED:** string and reply-formatting helpers, config arithmetic (`compute_effective_timeout`), `rename_graph` (RENAME is not modelled), `reauthorize_write` (Redis pause and role flags), borrow helpers, and `MvccGraph::drop`. `reply_profile` is proven in `redis_layer`.

`bash proofs/coverage.sh` counts each function once, in its best bucket across projects.

## What the model does not cover

- **RediSearch semantics.** Tokenizing, stemming, stopwords, scoring and HNSW recall are a hypothesis structure, not modelled.
- **The graph inside a version.** `Mvcc` treats a version as a list of opaque writes. The per-graph `RwLock` and the versioned matrices are covered in other projects.
- **Memory ordering.** Atomics are assumed sequentially consistent; the Rust uses Acquire/Release.
- **Index population.** `Tickets` models `CREATE INDEX` on one label and the populate jobs. It does not model the write path, `DROP INDEX` with fields remaining (recreate and repopulate), or `populate_indexes_sync`.
- **Exhaustive checks are bounded.** The `write_loop` hand-off is checked exhaustively for 2 producers only.
- **The fork child.** Beyond "it sees the committed version", what the child does (GraphBLAS `wait_all`, OpenMP) is not modelled.
- **Write detection.** How `execute_query` decides a query writes is not modelled.

## Build and check

```bash
cd proofs/search_concurrency && lake build   # this project
bash proofs/lean_ci.sh search_concurrency    # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                      # repo-wide coverage report
```

**When you change `src/graph_core.rs`, `src/query_session.rs`, `mvcc_graph.rs`, `threadpool.rs`, `locks.rs`, the population code in `graph.rs`/`indexer.rs`, or `Document::set` and the KNN queries:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match: a new locking path goes into `Locks.lean` and must pass `all_programs_ordered`; a new atomic step or a change to the slot protocol goes into `Mvcc.step`.
3. Re-run the build, and for a concurrency change re-run `repro/stress.py` against a live server.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
