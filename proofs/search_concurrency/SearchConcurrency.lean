/-
# Search layer and concurrency

Models the RediSearch-facing search layer (`Document::set`, index options, KNN `k`), index-population
tickets, MVCC publish/fork/commit with the GIL, lock order and the `write_loop` hand-off, the
thread pool and admission, and the query-session glue (`graph_core.rs`, `query_session.rs`,
`threadpool.rs`, `mvcc_graph.rs`, `graph/src/index/`). Live repros are in `repro/`.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
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
#print axioms SC.PoolWithin.second_call_reports_straggler
#print axioms SC.PoolWithin.within_blocks_behind_shutdown
#print axioms SC.Locks.never_stuck_3014
#print axioms SC.Locks.bug3226_open_scan_upgrade
#print axioms SC.Locks.bug3226_restore_stall_cycle
#print axioms SC.Locks.spec_programs_ordered
#print axioms SC.Guarded.h1_fork_vs_populate_race
#print axioms SC.Guarded.consistent_common_lock
#print axioms SC.Glue.cGraphName_eq_lossy_key
#print axioms SC.Glue.renameGraph_moves
#print axioms SC.Exec.execQuery_write_keeps_slot
#print axioms SC.Exec.effTimeout_pq
#print axioms SC.Session.checkFork_verdict
#print axioms SC.Session.acquire_drop
#print axioms SC.Session.commitNext_open_scan
