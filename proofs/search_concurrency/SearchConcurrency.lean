/-
# Search layer and concurrency: models, proofs, and open bugs (#3226, `shutdown_within` lock contention)

Project `proofs/search_concurrency`. Modules:

* `Search`   — `Document::set`, vector/fulltext option parsing, KNN `k`; RediSearch
               as a hypothesis structure (`RediSearch`), no `axiom`.
* `Knn`      — `vector_query_nodes` / `vector_query_edges` @ 89d68334a (#3088 k clamp).
* `Tickets`  — index-population tickets (`PendingSlots`) and the batch protocol.
* `Mvcc`     — MVCC publish/read + fork-and-commit (no write slot since #3657) + GIL + GRAPH.DELETE + BGSAVE fork,
               as a transition system with an inductive invariant.
* `Locks`    — lock-order deadlock freedom, the `write_loop` hand-off (exhaustive),
               and the bounded-queue deadlock.
* `Glue`     — `graph_core.rs` C name rules (`up_to_nul`, `c_graph_name`/`key`) and the graph
               registry (`register_graph`, `rename_graph`).
* `Exec`     — `ThreadedGraph::new`/`from_mvcc`, `WriteQueryOk::modified`, the read/detect phase of
               `execute_query`/`execute_profile` (write test, admission slot), `profile_sync`'s
               lock program, `compute_effective_timeout`, the two replies.
* `Session`  — `QuerySession`'s own state (`begin`, `with_graph[_mut]`, `release_locks`, `Drop`),
               `check_fork`, the GIL guard, `must_run_inline`, `reauthorize_write`,
               `MvccGraph::new`/`from_graph`/`Drop`, and `CommitOp::next`'s lock path.

Each module's header has its Lean ↔ Rust `file:line` table. Coverage rows are in
`COVERAGE.tsv`. Live-server repros are in `repro/` (ports 18470 Rust, 18471 C
`bin/macos-arm64v8-release/falkordb.so`).


## Wave 4 additions
* `Pool` — `threadpool.rs`: `shutdown_drains` (one sentinel per worker runs every queued job
  once, in order, then all workers exit), `spawn_after_exit`, `global_spec` (OnceCell init).
* `PoolWithin` — `threadpool.rs` after #3502 (ca36bfa15) and #3585 (7472d6a8a), Lean ↔ Rust at 256ab4d3f:

  | Lean | Rust |
  | --- | --- |
  | `workerName`, `spawnNames` | `ThreadPool::with_names` `threadpool.rs:83`, `new` `:70`, `named` `:76` |
  | `waitLoop`, `drain`, `within` | `ThreadPool::shutdown_within` `:167` (wait loop `:174-178`, partition and put back the running handles `:179-185`) |
  | `holdEnd` | `ThreadPool::shutdown` `:151` (holds the `workers` lock across `join`) |

  `within_in_time`, `within_true_all_finished`, `within_false_unfinished`, `within_partition`,
  `blocked_reported`, `left_behind_workers_exit`, `late_job_lost`; one confirmed bug, still **open**
  (`repro/threadpool_within.rs`): `within_blocks_behind_shutdown` (a `shutdown_within` behind a blocking
  `shutdown` waits on the `workers` lock past its limit, forever if a job never returns; live case B
  at 256ab4d3f). `second_call_reports_straggler`, `second_call_joins_when_returned`: a second
  `shutdown_within` keeps and reports a straggler (`pre3585_second_call_true_despite_straggler` is the
  earlier behaviour, kept as a historical counterexample).
* `Once` — `thread_id.rs` and `storage/registry.rs` set-once globals.

## Proven (no sorry / admit / axiom)

Concurrency (`Mvcc`, over every interleaving of read, forkW, forkCur, checkFork, abandon,
escalate, release, mutate, commit, indexer batch, delete, fork):
* `snapshots_are_committed` — a reader / fork child only ever sees a version some
  commit published whole (snapshot isolation; never a partial commit).
* `no_lost_writes` — committed state = all acknowledged writes, in ack order.
* `history_linear` — every published version is a prefix of the current one.
* `private_on_committed`, `current_not_stale` — a `Current` version is always based on the
  published one at its data version, by the GIL holder (write lock ⇒ writes serialized, no
  lost update; `commit`'s `Stale` refusal is unreachable under the lock).
* `delete_blocks_writes` — after GRAPH.DELETE no mutate/commit is enabled.
* `fork_excludes_commit` — fork and delete need the GIL a committing writer holds.
* Forks and commits: `one_commit_per_version`, `checkFork_stale`, `checkFork_on_committed`,
  `inline_write_ok`, `stale_then_rerun`, `unchecked_fork_cannot_write` (every reachable state; `reader_cannot_commit` is its one-trace instance); refused commits
  (`refused_commit_publishes_nothing`, `stale_commit_publishes_nothing`, `commit_publishes`);
  since #3545 the publish is a compare-and-swap (`Ev.swap`): `swap_overtaken_publishes_nothing` (a version
  published between the load and the swap refuses the commit, nothing published), `swap_unchanged_publishes`,
  `cas_refuses_overtaken_commit`, `cas_current_commit_publishes`;
  indexer batches keep the data version (`batch_keeps_forks`, `batch_then_check`,
  `batch_does_not_stale`, #3566).
`Locks`:
* `no_cycle` (+ `path_increasing`) — threads that acquire in rank order cannot form
  a wait-for cycle; `all_programs_ordered` — writer (incl. escalate), inline writer,
  populate, drop_index_bg, fork prepare, telemetry flush all obey GIL < graph <
  indexer < registry; `bad_escalate_rejected` — the #726 order is caught.
* `drain_no_lost_message`, `explore_saturated` — exhaustive (2 producers) check of
  the `write_loop` protocol: no message is stranded; `recheck_needed` — without the
  `is_empty` re-check one is.
* Admission: `inv3014_step` (count = queued, ≤ MAX_QUEUED_QUERIES), `never_stuck_3014`,
  `main_never_blocks`, `flood_drains`.
`Pool` (`QueueSlot`): `acquire_bounded`, `acquire_full`, `release_exactly_once`,
`paths_release_once`, `count_eq_live`, `spawn_never_blocks`.
`Tickets` (at 73077bb4a): `inc_inv`, `dec_inv`, `bump_inv`, `release_exact`,
`operational_iff`, `stale_release_isolated`; on main `create_index` bumps the generation
whenever the label already has an entry (indexer.rs:359-361): `main_operational_complete`
(key property: over every interleaving of any number of `CREATE INDEX`es on a label and the
batches of every populate job, an OPERATIONAL index has every field in every batch, outside
the store→`populate_index` window), via the inductive invariant `MInv` (`reach_inv`);
`main_lost_order_complete` (the interleaving that would lose `b` completes);
`store_window_operational_incomplete` (that window is real, see Gaps);
`fixed_trace_complete` (the shipped `create_index`).
`Search` (temporals are not indexed — `docSet_temporal_not_indexed`): `docSet_vector_sound` (key property: any
value, any field options — a vector add always has the field's own non-zero dimension),
`docSet_good_vector`, `docSet_bad_vector_skipped`, `docSet_noparams_no_vector`,
`doc_always_accepted`, `other_fields_never_dropped`, `docSet_nonvector_eq_pre`,
`vector_dim_mismatch_keeps_range_entry`, `dim0_keeps_range_entry`, `noopts_keeps_range_entry`,
`good_vector_keeps_range_entry`; `fulltext_only_strings`,
`range_never_vector_or_text`, `parseDimension_int/neg/missing`, `parseSim_missing/lower/ok_lower/unknown`,
`rust_refuses_missing`, `rust_eq_c`, `rust_eq_c_all`, `fulltextUnknown_none_iff`, `rust_keys_ok_imp_c`, `metricOf_ok_iff`, `metricOf_toBool`,
`phonetic_false_no_flag`, `phonetic_true_flag`,
`capped_capacity_ok`, `evalK_accepts_huge`; plus `#guard`s for the option checks.

`Knn` (89d68334a): `vector_query_nodes` / `vector_query_edges` (`graph.rs:4354`,
`:4415`) with `k` clamped to `count.max(1)`: `alloc_bounded` (every buffer and the RediSearch
`k` ≤ live entity count, for any user `k`), `no_abort_any_k`, `candidates_topk` (clamped
query = `ranked.take k`, length `min k |index|`), `rows_length_topk`, `huge_k_one_row`
(a huge `k` returns its 1 row, as C), `empty_graph`.

## CONFIRMED bugs (all reproduced on live servers, Rust vs C)

The open ones are #3226 and the `shutdown_within` lock contention (below). Option validation
(re-checked live at fe619ac5f, Rust release vs C `falkordb.so`, ports 19010/19011) agrees with C
(`rust_refuses_missing`, `rust_eq_c`, `rust_eq_c_all`, `parseSim_unknown`) except one divergence
(live, `#guard`): fulltext `OPTIONS {weight:1.0, foo:true}` Rust refuses (`unknown option 'foo'`), C creates
it — C refuses only a non-empty map with no known key (`cFulltextKeysOk`; `rust_keys_ok_imp_c`).
(C itself crashed on `CREATE FULLTEXT INDEX … OPTIONS {phonetic:true}`, on `db.indexes()` after `{language:'klingon'}`,
and on the 1400-write flood — C bugs, noted only.)

## Stress results (no violation found)

`repro/stress.py` (3 writers, 6 readers, GRAPH.DELETE with concurrent reads/writes,
BGSAVE every 0.3 s, 40 s): 0 violations of partial-commit, monotonic reads,
gap/dup, lost writes; 108 BGSAVEs ok, the RDB reloads consistent; deleted graphs
never resurrect content; writes racing a delete abort with "graph was deleted or
replaced". Same on C. (A first run showed "duplicates": redis-py's
retry-on-timeout re-sent writes — a harness artefact, see the MEMORY note.)

## Commit validation and `create_index` generations
`MvccGraph::commit` (mvcc_graph.rs:139) runs `Graph::validate` first and, on a
refusal, performs the rollback itself (:148-151). `Mvcc.step` takes `valid` as a
parameter; every invariant theorem holds for any `valid`; `refused_commit_publishes_nothing`
(nothing published or acked, the next writer can fork — the property the Rust test
`a_refused_commit_publishes_nothing` checks) and `commit_publishes`. `commit_and_replicate` treats a refusal as
`unreachable!` (graph_core.rs:1487).

`create_index` bumps the generation whenever the label already has an entry
(indexer.rs:359-361). `Tickets` keeps the earlier `create_index` as `pre3086_createRange` and models
`store`/`acquire`/`step`/`Reach`, the invariant `MInv` (`inv_create`, `inv_populate`, `inv_batch`,
`reach_inv`) and `main_operational_complete`.

## Gaps

Index population (73077bb4a): between `Indexer::create_index` storing the entry and
`populate_index` acquiring its ticket (graph.rs:3681-3688, no lock held across them) the
entry has pending 0, so `is_label_indexed` (indexer.rs:454) reports the new field usable
while no document has it (`store_window_operational_incomplete`). Unconfirmed live; the
window is a few instructions. `Tickets` does not model write-path tickets, `drop_index`
with fields remaining (recreate + populate), or `populate_indexes_sync`.

RediSearch query semantics (tokenizer, stemming, stopword matching, scoring, HNSW
recall) are hypotheses, not modelled; `Mvcc` abstracts the per-graph RwLock and
versioned matrices (covered elsewhere); the drain check is exhaustive for 2
producers only; memory ordering (Acquire/Release) is assumed SC; fork child
behaviour beyond "sees `pub`" (GraphBLAS `wait_all`, OpenMP) is not modelled.

## Open bug 2026-10-20 (#3226; proofs cite fb3d516d0, repro on main 9d3869aa8): model gap closed, bug still open
`Locks` now ranks RediSearch's global `RWLOCK` (`rsGlobal`) and per-spec `rwlock`
(`rsSpec`) above the indexer lock. Both are taken by `handleIterCommon`
(`deps/RediSearch src/redisearch_api.c:904-905`) and released only by
`RediSearch_ResultsIteratorFree` (:1064), so an index scan holds them until it is
dropped. Properties: `LockOrdered` (`okProg`, the premise of `no_cycle`) and
`noReacquire` (the `spec.c:2304` assertion). `writerOpenScanProg` is the live repro
`MATCH (n:W) WHERE n.v > 0 WITH n LIMIT 10 SET n.v = n.v + 100000 RETURN count(n)`:
the scan stays open below `LIMIT`, `CommitOp::next` escalates (`commit.rs:93`) and
`Indexer::commit` → `RediSearch_IndexAddDocument` (:643-645) takes both locks as a
writer. `bug3226_open_scan_upgrade` (live abort: "attempted to upgrade a spec read lock
to a write lock", port 6640; C returns 10 on 6641), `bug3226_open_scan_unordered`, and
`bug3226_restore_stall_cycle` (writer holds `rsGlobal` and wants the GIL; a `RESTORE`
holds the GIL and wants `rsGlobal`). `spec_programs_ordered`: read scans, population,
`RESTORE` and writers whose scan is closed before commit are ordered. Not modelled: the
issue's BGSAVE half (a `pending_slots` mutex held by another thread at fork time).
-/
import SearchConcurrency.Search
import SearchConcurrency.Tickets
import SearchConcurrency.Mvcc
import SearchConcurrency.Locks
import SearchConcurrency.Pool
import SearchConcurrency.PoolWithin
import SearchConcurrency.Once
import SearchConcurrency.Guarded
import SearchConcurrency.Knn
import SearchConcurrency.Glue
import SearchConcurrency.Exec
import SearchConcurrency.Session

#print axioms SC.Mvcc.snapshots_are_committed
#print axioms SC.Locks.no_cycle
#print axioms SC.Locks.drain_no_lost_message
#print axioms SC.Search.doc_always_accepted
#print axioms SC.Search.other_fields_never_dropped
#print axioms SC.Search.vector_dim_mismatch_keeps_range_entry
#print axioms SC.Tickets.main_operational_complete
#print axioms SC.Knn.alloc_bounded
#print axioms SC.Knn.rows_length_topk
#print axioms SC.Knn.not_operational_refused
#print axioms SC.Mvcc.one_commit_per_version
#print axioms SC.Mvcc.checkFork_on_committed
#print axioms SC.Pool.release_exactly_once
#print axioms SC.Pool.count_eq_live
#print axioms SC.PoolWithin.within_in_time
#print axioms SC.PoolWithin.within_true_all_finished
#print axioms SC.PoolWithin.left_behind_workers_exit
#print axioms SC.PoolWithin.pre3585_second_call_true_despite_straggler
#print axioms SC.PoolWithin.second_call_reports_straggler
#print axioms SC.PoolWithin.within_blocks_behind_shutdown
#print axioms SC.Locks.never_stuck_3014
#print axioms SC.Locks.bug3226_open_scan_upgrade
#print axioms SC.Locks.bug3226_restore_stall_cycle
#print axioms SC.Locks.spec_programs_ordered
#print axioms SC.Guarded.bug3483_naRef_race
#print axioms SC.Guarded.rs29_naRef_free
#print axioms SC.Guarded.h1_fork_vs_populate_race
#print axioms SC.Guarded.consistent_common_lock
#print axioms SC.Glue.cGraphName_eq_lossy_key
#print axioms SC.Glue.renameGraph_moves
#print axioms SC.Exec.execQuery_write_keeps_slot
#print axioms SC.Exec.effTimeout_pq
#print axioms SC.Session.checkFork_verdict
#print axioms SC.Session.acquire_drop
#print axioms SC.Session.commitNext_open_scan
