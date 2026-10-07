/-
# Search layer and concurrency: models, proofs, and five confirmed bugs

Project `proofs/search_concurrency`. Modules:

* `Search`   — `Document::set`, vector/fulltext option parsing, KNN `k`; RediSearch
               as a hypothesis structure (`RediSearch`), no `axiom`.
* `Knn`      — `vector_query_nodes` / `vector_query_edges` @ 89d68334a (#3088 k clamp).
* `Tickets`  — index-population tickets (`PendingSlots`) and the batch protocol.
* `Mvcc`     — MVCC publish/read + write slot + GIL + GRAPH.DELETE + BGSAVE fork,
               as a transition system with an inductive invariant.
* `Locks`    — lock-order deadlock freedom, the `write_loop` hand-off (exhaustive),
               and the bounded-queue deadlock.

Each module's header has its Lean ↔ Rust `file:line` table. Coverage rows are in
`COVERAGE.tsv`. Live-server repros are in `repro/` (ports 18470 Rust, 18471 C
`bin/macos-arm64v8-release/falkordb.so`).


## Wave 4 additions
* `Pool` — `threadpool.rs`: `shutdown_drains` (one sentinel per worker runs every queued job
  once, in order, then all workers exit), `spawn_after_exit`, `global_spec` (OnceCell init).
* `Once` — `thread_id.rs` and `storage/registry.rs` set-once globals.

## Proven (no sorry / admit / axiom)

Concurrency (`Mvcc`, over every interleaving of read, claim, escalate, release,
mutate, commit, rollback, delete, fork):
* `snapshots_are_committed` — a reader / fork child only ever sees a version some
  commit published whole (snapshot isolation; never a partial commit).
* `no_lost_writes` — committed state = all acknowledged writes, in ack order.
* `history_linear` — every published version is a prefix of the current one.
* `private_on_committed` — a private version is always based on the committed one
  (single write slot ⇒ writes serialized, no lost update).
* `delete_blocks_writes` — after GRAPH.DELETE no mutate/commit is enabled.
* `fork_excludes_commit` — fork and delete need the GIL a committing writer holds.
* `multi_write_refused` / `multi_write_ok_in_c` — the slot-window divergence below.
`Locks`:
* `no_cycle` (+ `path_increasing`) — threads that acquire in rank order cannot form
  a wait-for cycle; `all_programs_ordered` — writer (incl. escalate), inline writer,
  populate, drop_index_bg, fork prepare, telemetry flush all obey GIL < graph <
  indexer < registry; `bad_escalate_rejected` — the #726 order is caught.
* `drain_no_lost_message`, `explore_saturated` — exhaustive (2 producers) check of
  the `write_loop` protocol: no message is stranded; `recheck_needed` — without the
  `is_empty` re-check one is.
* `pool_deadlock_reachable` / `fixed_never_stuck` — the bounded-queue deadlock and
  its fix.
`Tickets`: `inc_inv`, `dec_inv`, `bump_inv`, `release_exact`, `operational_iff`,
`stale_release_isolated`, `second_index_never_populated`, `lucky_order_complete`,
`fixed_trace_complete`.
`Search` (re-targeted to 8743953a8: #3087 vector guard; #3076 `e20300436` temporals no
longer indexed — `docSet_temporal_not_indexed`, `docSet_eq_pre3076_off_temporal`, historical
`pre3076_temporal_in_numeric`): `docSet_vector_sound` (key property: any
value, any field options — a vector add always has the field's own non-zero dimension),
`docSet_good_vector`, `docSet_bad_vector_skipped`, `docSet_noparams_no_vector`,
`doc_always_accepted`, `other_fields_never_dropped`, `docSet_nonvector_eq_pre`,
`vector_dim_mismatch_keeps_range_entry`, `dim0_keeps_range_entry`, `noopts_keeps_range_entry`,
`good_vector_keeps_range_entry`; historical (fe619ac5f) `pre3087_dim_mismatch_dropped_range_entry`,
`pre3087_noparams_dropped_range_entry`; `fulltext_only_strings`,
`range_never_vector_or_text`, `parseDimension_int/neg/missing`, `parseSim_missing/lower`,
`rust_refuses_missing`, `rust_eq_c` (#3094), `fulltextUnknown_none_iff`, `rust_keys_ok_imp_c`, `metricOf_ok_iff`, `metricOf_toBool`,
`phonetic_false_no_flag`, `phonetic_true_flag`, historical `pre3088_huge_k_kills_server`,
`capped_capacity_ok`, `evalK_accepts_huge`; plus `#guard`s for the option checks.

`Knn` (89d68334a, #3088): `vector_query_nodes` / `vector_query_edges` (`graph.rs:3860`,
`:3920`) with `k` clamped to `count.max(1)`: `alloc_bounded` (every buffer and the RediSearch
`k` ≤ live entity count, for any user `k`), `no_abort_any_k`, `candidates_topk` (clamped
query = `ranked.take k`, length `min k |index|`), `rows_length_topk`, `huge_k_one_row`
(the #3085 repro now returns its 1 row, as C), `empty_graph`.

## CONFIRMED bugs (all reproduced on live servers, Rust vs C)

1. **Server deadlock under >1024 queued queries** (`graph/src/threadpool.rs:59,96`,
   `src/graph_core.rs:1002`). `spawn` does a *blocking* send into a 1024-slot pool
   queue from the Redis main thread, which holds the GIL; running workers escalate
   and wait for the GIL. Permanent hang, 0% CPU, PING times out.
   `repro/flood.py 18470 1400` (one graph) or `repro/flood2.py 18470 3000 16 "UNWIND
   range(1,3000) AS x CREATE (:N)" 45` (16 graphs, <1024 each): Rust hangs forever
   (thread samples: main in `crossfire…send` from `query_mut`, all 11 workers in
   `upgrade_to_write` waiting for the GIL); C completes 3000/3000 in 13 s.
   Model: `Locks.pool_deadlock_reachable`. Fix: never block the main thread in
   `spawn` (try_send → "Max pending queries exceeded", or unbounded queue like C).
   A second cycle with one graph: workers block on the full 1024 write channel
   (`graph_core.rs:547`, `tg.sender.send` `:1082`) while holding a pool worker.
2. **Second index on a label is never populated** (`graph/src/graph/graph.rs:547`
   `ticket_pending_changes > 1` bail + `index/indexer.rs:319-331` no `bump_id` for
   non-vector fields). `CREATE INDEX … (n.a)` then `(n.b)` on 300k nodes: Rust
   OPERATIONAL, `MATCH (n:L) WHERE n.b < 300000` via index = 0; C = 300000. Same
   mechanism loses index entries when writes race population (`repro/poprace.py`:
   8/8 rounds mismatched on Rust, 0/5 on C). Model: `second_index_never_populated`.
   Fix: bump the generation on every field addition (`fixed_trace_complete`).
3. **FIXED by #3087 (49f698d22)** — HISTORICAL (fe619ac5f): a vector of the wrong
   dimension (or any vector on a `{dimension:0}` / no-options vector field) removed the
   node from all of its label's indexes (old `mod.rs:723-733` added the blob unchecked;
   C `src/index/index.c:470-476` skips). `CREATE INDEX (n.name)`, vector index dim 2,
   `CREATE (:L {name:'a', v:vecf32([1,2,3])})`, `MATCH (n:L) WHERE n.name='a'`: Rust
   was `[]`, C `a`. Historical model: `pre3087_dim_mismatch_dropped_range_entry`,
   `pre3087_noparams_dropped_range_entry`. Now `mod.rs:730-744` adds the vector only if
   `vector_options.is_some_and(|o| o.dimension != 0 && o.dimension == vec.len())`:
   `docSet_vector_sound`, `doc_always_accepted`, `other_fields_never_dropped`,
   `vector_dim_mismatch_keeps_range_entry`, `dim0_keeps_range_entry`, `noopts_keeps_range_entry`.
4. **FIXED by #3088 (89d68334a)** — HISTORICAL (W3-conc-4 / #3085, 49f698d22): huge `k`
   killed the server (old `graph.rs:3883,3938` `Vec::with_capacity(k)`). `CALL
   db.idx.vector.queryNodes('L','v', 1000000000000000, vecf32([1,2]))`: Rust crashed
   (SIGSEGV in the OOM path); C returns the 1 row. Historical model:
   `pre3088_huge_k_kills_server`. Now `graph.rs:3880,3939` clamp `k` to
   `node_count().max(1)` / `relationship_count().max(1)` and `:3896,3955` size by results:
   `Knn.alloc_bounded`, `no_abort_any_k`, `candidates_topk`, `huge_k_one_row`.
5. **Inline (MULTI/EXEC) write fails with "another write is in progress"**
   (`src/graph_core.rs:748` slot claimed as a reader, before escalation). A queued
   write in its match phase + `MULTI; GRAPH.QUERY g "CREATE (:M)"; EXEC`: Rust
   replies `another write is in progress, retry the query` and `M` is not created
   (3/3); C creates it (3/3). `repro/multi_race.py`. Model: `multi_write_refused`.
   Fix: claim the slot after escalation (or block on it under the write lock).

Option validation (re-checked live at fe619ac5f, Rust release vs C `falkordb.so`, ports 19010/19011).
Historical (2c874022a, #3091): Rust accepted a vector index without `dimension` or
`similarityFunction`, rejected `'Euclidean'`, and accepted unknown fulltext keys; C the reverse.
**Fixed by #3094**: `rust_refuses_missing`, `rust_eq_c` (agrees with C for `dimension > 0`), live:
all four cases now answer as C. W5-idx-2 (vector index with no OPTIONS half-created on error) is
also fixed: `index_ddl.rs:61-65` refuses before anything is registered — live, `db.indexes()` lists
only the range index and `n.v = 1` still finds the node on both servers.
Still divergent (live, #guards): (a) `OPTIONS {dimension:0, similarityFunction:'bogus'}` Rust
creates the index (metric only checked for `dimension > 0`, mod.rs:1270-1272), C refuses;
(since #3087 a vector on that index no longer drops the node: `dim0_keeps_range_entry`);
(b) fulltext `OPTIONS {weight:1.0, foo:true}` Rust refuses (`unknown option 'foo'`), C creates —
C refuses only a non-empty map with no known key (`cFulltextKeysOk`; `rust_keys_ok_imp_c`).
(C itself crashed on `CREATE FULLTEXT INDEX … OPTIONS {phonetic:true}`, on `db.indexes()` after `{language:'klingon'}`,
and on the 1400-write flood — C bugs, noted only.)

## Stress results (no violation found)

`repro/stress.py` (3 writers, 6 readers, GRAPH.DELETE with concurrent reads/writes,
BGSAVE every 0.3 s, 40 s): 0 violations of partial-commit, monotonic reads,
gap/dup, lost writes; 108 BGSAVEs ok, the RDB reloads consistent; deleted graphs
never resurrect content; writes racing a delete abort with "graph was deleted or
replaced". Same on C. (A first run showed "duplicates": redis-py's
retry-on-timeout re-sent writes — a harness artefact, see the MEMORY note.)

## Re-target to 2c874022a (#2846)
`MvccGraph::commit` (mvcc_graph.rs:139) now runs `Graph::validate` first and, on a
refusal, performs the rollback itself (:148-151). `Mvcc.step` takes `valid` as a
parameter; every invariant theorem holds for any `valid`; new
`refused_commit_releases_slot` (nothing published or acked, the next writer can
claim — the property the Rust test `a_refused_commit_releases_the_write_slot`
checks) and `valid_commit_publishes`. `commit_and_replicate` treats a refusal as
`unreachable!` (graph_core.rs:1484).

## Gaps

RediSearch query semantics (tokenizer, stemming, stopword matching, scoring, HNSW
recall) are hypotheses, not modelled; `Mvcc` abstracts the per-graph RwLock and
versioned matrices (covered elsewhere); the drain check is exhaustive for 2
producers only; memory ordering (Acquire/Release) is assumed SC; fork child
behaviour beyond "sees `pub`" (GraphBLAS `wait_all`, OpenMP) is not modelled.
-/
import SearchConcurrency.Search
import SearchConcurrency.Tickets
import SearchConcurrency.Mvcc
import SearchConcurrency.Locks
import SearchConcurrency.Pool
import SearchConcurrency.Once
import SearchConcurrency.Knn

#print axioms SC.Mvcc.snapshots_are_committed
#print axioms SC.Locks.no_cycle
#print axioms SC.Locks.drain_no_lost_message
#print axioms SC.Search.doc_always_accepted
#print axioms SC.Search.other_fields_never_dropped
#print axioms SC.Search.vector_dim_mismatch_keeps_range_entry
#print axioms SC.Tickets.second_index_never_populated
#print axioms SC.Knn.alloc_bounded
#print axioms SC.Knn.rows_length_topk
