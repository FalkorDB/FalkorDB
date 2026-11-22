# search_concurrency: snapshot isolation, deadlock freedom and the RediSearch document path

This project proves the concurrency guarantees FalkorDB-rs gives its clients, and shows with counterexamples where they break. It models the Redis main thread, the worker pool, the per-graph write lock and fork-and-commit versions (no write slot), the GIL, `GRAPH.DELETE` and the `BGSAVE` fork as a transition system, and proves properties that hold over **every** interleaving. It also covers the RediSearch side of indexing: what `Document::set` sends, index option parsing, index population tickets, and vector KNN queries.

User-facing behaviour that depends on it:

- **Snapshot isolation.** A read query sees one committed version of the graph, never half of a write. Concurrent writes are serialized and none is lost.
- **No hang.** The server must stay responsive however many queries are queued, and lock ordering must never deadlock.
- **Inline writes.** A write inside `MULTI`/`EXEC` (or a Lua script) runs on the main thread; it must succeed even while a queued write is in progress.
- **`GRAPH.DELETE` and `BGSAVE`** while queries run.
- **Indexes.** `CREATE INDEX` must populate the index (also a second index on the same label), an entity must never drop out of an index because of a bad vector, and `db.idx.vector.queryNodes` must not crash on a huge `k`.

Each module's header has its Lean↔Rust `file:line` table; the root [`SearchConcurrency.lean`](SearchConcurrency.lean) lists the modules, the theorems and the bugs. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Can a reader see a partial commit, or a write get lost?**

No. `Mvcc.lean` models each atomic step of the Rust (read, fork a private version as a reader or a writer, check the fork, escalate to the GIL, mutate, commit, abandon, delete, BGSAVE fork) and lets any enabled step fire in any order. Over every reachable state:

- `snapshots_are_committed`: every snapshot a reader or fork child ever took is a version some commit published whole;
- `no_lost_writes`: the committed state is exactly the acknowledged writes, in acknowledgement order;
- `history_linear`: every published version is a prefix of the current one;
- `private_on_committed`: a writer's private version is always based on the current committed one, so there is no lost update;
- `one_commit_per_version`: at most one commit lands on each version.

**2. Can an inline write fail because a queued write is in progress?**

- **No.** The queued write builds on a fork that takes nothing (there is no write slot at all) and checks it only after it holds the GIL. If an inline write committed in between, the check finds it stale and the queued write re-runs (`checkFork_stale`, `stale_then_rerun`); the inline write succeeds (`inline_write_ok`). In every reachable state a thread with an unchecked fork cannot mutate or commit (`unchecked_fork_cannot_write`; `reader_cannot_commit` is the one-trace instance): the model refuses `forkCur` for a thread that already holds a version, as a real session (one version at a time) does. An indexer batch in between does not make a fork stale (`batch_does_not_stale`).

**3. Can the server deadlock?**

- **Lock order.** `no_cycle` proves threads that acquire locks in rank order cannot form a wait-for cycle. `all_programs_ordered` checks that every locking path (the writer including escalation, the inline writer, index population, background index drop, the fork preparation, telemetry flush) obeys GIL < graph < indexer < registry. `bad_escalate_rejected` shows the checker catches a wrong order.
- **RediSearch's locks (gap closed 2026-10-20, #3226).** The ranks now include RediSearch's global `RWLOCK` and per-spec lock, which an index scan holds from opening to dropping. A write that commits with its scan still open re-takes the spec lock as a writer (`bug3226_open_scan_upgrade`, a live abort) and takes the GIL while holding RediSearch's locks (`bug3226_open_scan_unordered`, `bug3226_restore_stall_cycle`). Read scans, population, `RESTORE` and writers whose scan closes before the commit are ordered (`spec_programs_ordered`).
- **The write queue hand-off.** `drain_no_lost_message` checks exhaustively, for 2 producers, that the `write_loop` flag protocol never strands a queued write; `recheck_needed` shows it would without its `is_empty` re-check.
- **Queue overflow.** More than 1024 queued queries cannot hang the server: `never_stuck_3014` and `main_never_blocks` prove the main thread never waits and some step is always possible, and `flood_drains` that a flood drains.

**4. Is a new index always populated?**

- **Yes.** `create_index` bumps the generation whenever the label already has an index, so the old job fails `is_ticket_current` at its next batch and the new job indexes everything. `main_operational_complete` proves, over every interleaving of any number of `CREATE INDEX`es on a label with the batches of every populate job, that an OPERATIONAL index has every field in every batch. The proof is an inductive invariant (`MInv`, `reach_inv`): at most one job owns the current generation, it holds the current field list, and the batches before its cursor are complete. `main_lost_order_complete` replays the second-index interleaving.
- **One remaining window.** The theorem excludes the moment between `Indexer::create_index` storing the entry and `populate_index` taking its ticket. In that window the entry has no pending ticket, so it looks OPERATIONAL while the new field is in no document (`store_window_operational_incomplete`). The window is a few instructions long and has not been seen on a live server.

The ticket counters themselves are proven sound (`inc_inv`, `dec_inv`, `bump_inv`, `release_exact`, `operational_iff`, `stale_release_isolated`).

**5. Can a bad vector remove an entity from its other indexes?**

- **No.** `docSet_vector_sound` proves a vector is added only with the field's own non-zero dimension, for any value and any options; `doc_always_accepted` and `other_fields_never_dropped` prove the document is accepted and the other fields stay.

**6. Can a KNN query with a huge `k` crash the server?**

- **No.** `k` is clamped to the entity count: `alloc_bounded` proves every buffer is at most the live entity count, and `candidates_topk` that the result is still the top `k` (a query with a huge `k` returns its one row, `huge_k_one_row`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `snapshots_are_committed` | Readers and the BGSAVE child only see whole committed versions | `Mvcc.lean` |
| `no_lost_writes`, `history_linear` | Committed state = acknowledged writes, in order; history only grows | `Mvcc.lean` |
| `one_commit_per_version`, `inline_write_ok` | Writes are serialized; an inline write is never refused by a queued one | `Mvcc.lean` |
| `delete_blocks_writes`, `fork_excludes_commit` | After `GRAPH.DELETE` no write can proceed; a fork never runs during a commit | `Mvcc.lean` |
| `swap_overtaken_publishes_nothing`, `swap_unchanged_publishes` | A commit publishes by compare-and-swap against the pointer it loaded; another publisher slipping in refuses it (`Stale`) and nothing is published; with nothing in between it publishes | `Mvcc.lean` |
| `refused_commit_publishes_nothing`, `current_not_stale` | A commit refused by `Graph::validate` publishes nothing and the version is dropped; a version made under the write lock is never refused as stale | `Mvcc.lean` |
| `no_cycle`, `all_programs_ordered` | The lock order rules out deadlock, and every locking path follows it | `Locks.lean` |
| `drain_no_lost_message` | The `write_loop` hand-off never strands a queued write (2 producers, exhaustive) | `Locks.lean` |
| `unchecked_fork_cannot_write` | In every reachable state a thread with an unchecked fork can neither mutate nor commit | `Mvcc.lean` |
| `main_operational_complete` | An OPERATIONAL index has every field populated, over every interleaving of index creations and populate batches, except the store-to-ticket window (the theorem assumes `owed = false`; see "One remaining window", `store_window_operational_incomplete`) | `Guarded.lean` | Eraser-style locksets: which locks every access to shared state holds (`consistent`, `h1_fork_vs_populate_race`) |
| `Tickets.lean` |
| `within_in_time`, `second_call_reports_straggler`, `left_behind_workers_exit` | `threadpool.rs`: named workers; `shutdown_within` returns in time, keeps a stuck worker and reports it on every later call until it exits, and leaks none; one bug stays open (`within_blocks_behind_shutdown`) | `PoolWithin.lean` |
| `never_stuck_3014`, `main_never_blocks` | The main thread never blocks on a full queue | `Locks.lean` |
| `release_exactly_once`, `count_eq_live` | Every admitted query releases its queue slot exactly once; the counter equals live queries | `Pool.lean` |
| `shutdown_drains` | Pool shutdown runs every queued job once, in order, then all workers exit | `Pool.lean` |
| `docSet_vector_sound`, `other_fields_never_dropped` | A bad vector cannot drop the entity from other indexes | `Search.lean` |
| `alloc_bounded`, `candidates_topk` | KNN never allocates more than the entity count and still returns the top `k` | `Knn.lean` |

### How `snapshots_are_committed` is proved

1. **Model the state.** `S` holds the committed version `pub` and its version number, the `Current` version (owner, the data version and base it was forked at, its pending writes), the unchecked forks, who holds the GIL, whether the graph is registered, and ghost history: every published version (`hist`), every snapshot taken (`seen`) and every acknowledged write (`acked`). A version is the list of writes it contains.
2. **Model the Rust steps.** `step` has one case per event, each one atomic step of the Rust: `read` clones the committed `Arc`; `forkW`/`forkCur` take a private version and `checkFork` finds it stale or makes it `Current`; `commit` runs `Graph::validate` (a parameter, so every theorem holds for any validation) and then swaps the `Arc`; `delete` and `fork` need the GIL on the main thread.
3. **Any interleaving.** `Reach` is every state reachable from `init` by any sequence of enabled steps.
4. **One inductive invariant.** `Inv` says: `hist`'s head is `pub`, a `Current` version's base is `pub`, everything `seen` is in `hist`, `acked = pub`, every historical version is a prefix of `pub`, forks are no newer than `pub`, and a thread with an unchecked fork has no `Current` version. `inv_step` proves each event keeps it; `reach_inv` lifts that to every reachable state. The headline theorems are fields of `Inv`.

The model abstracts what a write does to a version (writes are opaque), the per-graph `RwLock` and the matrices (proven elsewhere), and assumes sequentially consistent memory for the atomics.

## Bugs found

All reproduced on live servers, Rust against C (scripts in `repro/`).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| #3226 | `bug3226_open_scan_upgrade`, `bug3226_open_scan_unordered`, `bug3226_restore_stall_cycle` (properties `noReacquire`, `LockOrdered`) | A write that commits while its own index scan is still open (`MATCH (n:W) WHERE n.v > 0 WITH n LIMIT 10 SET n.v = n.v + 100000`) takes RediSearch's spec lock as a writer while it holds it as a reader. The server aborts with `attempted to upgrade a spec read lock to a write lock`; C returns 10. The same path takes the GIL while it holds RediSearch's global lock, which makes a wait-for cycle with a `RESTORE` on the main thread. Paths that close the scan before the commit stay ordered (`spec_programs_ordered`). The BGSAVE-child half of the issue (a `pending_slots` mutex inherited across the fork) is not modelled | **Open** |

| `shutdown_within` | `within_blocks_behind_shutdown` (`repro/threadpool_within.rs`, case B) | A `shutdown_within(t)` called while another thread is inside `shutdown` blocks on the `workers` lock for as long as `shutdown`'s `join`, forever if a job never returns, so the time limit does not hold | **Open** |
| H1 | `Guarded.h1_fork_vs_populate_race` | ForkGC takes only the GIL around `RM_Fork` while background population writes the index under RediSearch's locks but not the GIL, so the child can snapshot a half-written block (`h1_fix_populate_gil`, `h1_fix_gc_spec_lock`: either fix restores a common lock) | **Open** |

Index option validation matches C (`rust_refuses_missing`, `rust_eq_c`, `rust_eq_c_all`) except for one DDL-only difference, pinned by a `#guard`: fulltext `OPTIONS {weight:1.0, foo:true}` is refused (C creates it; `rust_keys_ok_imp_c`).

A 40-second stress run (`repro/stress.py`: 3 writers, 6 readers, `GRAPH.DELETE`, `BGSAVE` every 0.3 s) found no partial commit, non-monotonic read, gap, duplicate or lost write, on Rust or C.

## Files

| File | What's in it |
|---|---|
| `Mvcc.lean` | The transition system: `S`, `Ev`, `step`, `Reach`, the invariant `Inv`; snapshot isolation, no lost writes, fork-and-commit (`unchecked_fork_cannot_write`), delete and fork exclusion |
| `Locks.lean` | Lock-order deadlock freedom (`no_cycle`), the `write_loop` hand-off (`drain_no_lost_message`), the admission protocol (`never_stuck_3014`) |
| `Pool.lean` | `threadpool.rs` and `QueueSlot`: `shutdown_drains`, `acquire_bounded`, `release_exactly_once`, `count_eq_live` |
| `Guarded.lean` | Eraser-style locksets: which locks every access to shared state holds (`consistent`, `h1_fork_vs_populate_race`) |
| `Tickets.lean` | Index population tickets (`Slots`): counter soundness, `main_operational_complete` |
| `Search.lean` | `Document::set` with RediSearch as a hypothesis structure, vector and fulltext option parsing: `docSet_vector_sound`, `doc_always_accepted`, `rust_eq_c` |
| `Knn.lean` | `vector_query_nodes`/`vector_query_edges` with the `k` clamp: `alloc_bounded`, `candidates_topk` |
| `Once.lean` | Set-once globals in `thread_id.rs` and `storage/registry.rs`, and the `POOL_THREAD` cell |
| `Glue.lean` | `graph_core.rs` C name rules (`upToNul_idem`, `cGraphName_eq_lossy_key`) and the graph registry (`registerGraph_*`, `renameGraph_moves`) |
| `Exec.lean` | `ThreadedGraph` constructors, `WriteQueryOk::modified`, `execute_query`/`execute_profile` write detection and admission hand-off (`execQuery_read`, `execQuery_write_keeps_slot`), `compute_effective_timeout`, the two replies |
| `Session.lean` | `QuerySession`'s own state and drop order, `check_fork` (`checkFork_verdict`), the GIL guard (`acquire_nested`, `acquire_drop`), `must_run_inline`, `reauthorize_write`, `MvccGraph::new`/`from_graph`/`Drop`, `CommitOp::next`'s lock path |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

No path dependencies.

## Coverage

`COVERAGE.tsv` has 126 rows, from `src/graph_core.rs`, `src/query_session.rs`, `graph/src/graph/mvcc_graph.rs`, `threadpool.rs`, the index files and a few others:

| Bucket | Rows |
|---|---|
| PROVEN | 106 |
| CONDITIONAL | 0 |
| AXIOMATISED | 20 |
| MODELLED | 0 |
| NOT COVERED | 0 |

- **CONDITIONAL (none left):** `Session.checkFork` has two outcomes only.
- **AXIOMATISED:** thin wrappers over the Redis module API (blocking and unblocking clients, thread-safe contexts, the GIL lock and unlock, replication, string helpers), and `Index::add_document`, whose RediSearch behaviour is the `RediSearch` hypothesis structure in `Search.lean` (a whole document is rejected or accepted).
- **Formerly MODELLED / NOT COVERED (31 rows, now PROVEN in `Glue`, `Exec`, `Session`, `Search`):** the C name helpers, `register_graph`/`rename_graph`, the `ThreadedGraph` constructors, `execute_query`/`execute_profile`/`profile_sync`, `compute_effective_timeout`, the two replies, the `QuerySession` constructors, accessors, release and drop order, `Gil::acquire`/`drop`, `must_run_inline`, `reauthorize_write`, `MvccGraph::new`/`from_graph`/`Drop`, the `Field` constructors and `CommitOp::next` (its lock path, with the #3226 property refuted).

`bash proofs/coverage.sh` counts each function once, in its best bucket across projects.

## What the model does not cover

- **RediSearch semantics.** Tokenizing, stemming, stopwords, scoring and HNSW recall are a hypothesis structure, not modelled.
- **The graph inside a version.** `Mvcc` treats a version as a list of opaque writes. The per-graph `RwLock` and the versioned matrices are covered in other projects.
- **Memory ordering.** Atomics are assumed sequentially consistent; the Rust uses Acquire/Release.
- **Index population.** `Tickets` models `CREATE INDEX` on one label and the populate jobs. It does not model the write path, `DROP INDEX` with fields remaining (recreate and repopulate), or `populate_indexes_sync`.
- **Exhaustive checks are bounded.** The `write_loop` hand-off is checked exhaustively for 2 producers only.
- **The fork child.** Beyond "it sees the committed version", what the child does (GraphBLAS `wait_all`, OpenMP) is not modelled.
- **Plan classification.** `Exec.isWritePlan` models the test on plan node kinds; which nodes the planner emits for a query is the planner's (other projects).

## Build and check

```bash
(cd proofs/search_concurrency && lake build)   # this project
bash proofs/lean_ci.sh search_concurrency    # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                      # repo-wide coverage report
```

**When you change `src/graph_core.rs`, `src/query_session.rs`, `mvcc_graph.rs`, `threadpool.rs`, `locks.rs`, the population code in `graph.rs`/`indexer.rs`, or `Document::set` and the KNN queries:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match: a new locking path goes into `Locks.lean` and must pass `all_programs_ordered`; a new atomic step or a change to the fork/commit protocol goes into `Mvcc.step`.
3. Re-run the build, and for a concurrency change re-run `repro/stress.py` against a live server.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
