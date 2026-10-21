/-
# pr3436_review: PR #3436 "perf: a read query borrows its graph once, not on every access"

PR branch `perf/read-graph-once` @ d7d166e82 (one commit on origin/main 9212add39), fixes
#3431. Build: `lake build` (Lean 4.35, no Mathlib, no `sorry`/`admit`/`axiom`). Imports
`runtime_core` (owner of `runtime.rs`) and `redis_layer` (owner of `graph_core.rs` /
`query_session.rs`) through Lake path dependencies. COVERAGE rows are commented out until
the PR merges.

The PR adds `Runtime.read_guard: Option<AtomicRef<'static, Graph>>` (`runtime.rs:147`),
taken once in `Runtime::new` for `!write` (`:460-462`, lifetime extended with
`transmute`), and `Runtime::graph() -> GraphRef` (`:530-535`; `GraphRef` + `Deref`
`:111-127`); 21 other files replace `X.g.borrow()` with `X.graph()` verbatim.

| module | claim | Rust |
| --- | --- | --- |
| `Cell`      | 1, 4 | atomic_refcell-0.1.14 borrow word; `Runtime::new` guard, `GraphRef`, `Deref`, `graph()` |
| `Protocol`  | 2    | `MvccGraph` (`mvcc_graph.rs:112-270`), `QuerySession` (`query_session.rs:159-394`), read-runtime construction (`graph_core.rs:632,698`, `record.rs:84`), every `borrow_mut()` on a graph cell |
| `DropOrder` | 3    | `struct Runtime` field order, Rust drop glue |
| `Reuse`     | 2 (premise), 5 | `RedisLayer.Ro` (parser `write` flag + planner `Commit`), `RuntimeCore.Accessors` |

## Verdicts

1. **Semantic equivalence — PROVEN.** `Cell.graph_deref`: whatever `graph()` returns derefs
   to `g`'s cell, as `g.borrow()` does (`borrowMain_deref`); `graph_same_result`: any
   `f : Graph → α` gives the same answer; `read_program_same`: a read runtime's whole
   read-only program returns the same answers on main and the PR whenever the cell is not
   mutably borrowed — which `Protocol.guard_never_excl` proves for every version a reader
   holds. `read_program_pr`: on the PR the reads never panic and never touch the borrow
   word (the perf claim's mechanism: no atomic RMW per access).
2. **Soundness of holding the read borrow — PROVEN; the PR's stated premise is REFUTED.**
   The PR says "a published version is never mutably borrowed again (writers mutate a
   private fork)". False: `Pending::resync_published_indexes` (`pending.rs:1684`) calls
   `committed.borrow_mut()` on the *published* version, reached from `abandon_write`
   (`graph_core.rs:1558-1561`) whenever a write that already ran a `Commit` fails
   (`Protocol.published_borrowed_mut`; live: the crash backtrace below panics inside it).
   The PR is nevertheless sound, for a different reason: that borrow happens only in
   writer mode (per-graph write lock held), and a read runtime lives entirely inside its
   session's read lock. `Protocol.reader_guard_sound`: in every reachable state each read
   guard sits on a published version with no live `borrow_mut`; a `borrow_mut` of a
   private version never meets a guard (`priv_unguarded`); under the write lock there are
   no guards at all (`writer_no_guards`, `resync_excludes_readers`); the guard's own
   borrow never panics (`readNew_never_panics`); and the guards never change whether a
   `borrow_mut` panics (`mutAcq_panic_iff`, `mutAcqPub_panic_iff`: the condition is
   main's, `excl ∨ shared`). The premise "a read runtime runs no `borrow_mut` site" is
   discharged by `Reuse.read_plan_no_update` on redis_layer's parser/planner model. Live
   (PR build): 20 write-shaped queries (`CALL {}` writes, `UNION`, `FOREACH`, `MERGE`,
   index DDL, …) are all refused by `GRAPH.RO_QUERY`; 2 inline `MULTI` failing writes
   (10 072 resyncs of the published version) against 6 reader threads (26 412 read
   queries) for 60 s: no panic, index consistent.
3. **Drop order / lifetime — PROVEN** (on the modelled drop glue). `DropOrder.drop_sound`:
   with `read_guard` declared before `g`, dropping a `Runtime` never writes into freed
   memory even when `g` holds the last strong reference; `swapped_unsound`: the opposite
   order would (`decide`); `swapped_ok_if_shared`: with another strong reference (the
   `MvccGraph` keeps the published `Arc` while the reader holds the read lock) either order
   is safe, so an unwind out of `Runtime::new`'s struct literal cannot dangle either;
   `drop_restores_borrow`; `move_irrelevant` (the guard points into the `Arc` allocation,
   not into `Runtime`). Checked by grep, not Lean: no `impl Drop for Runtime`, no `&mut
   Runtime`, no `let mut runtime`/`mut rt`, so the `pub g` is never reassigned.
4. **Write runtimes unchanged — PROVEN.** `Cell.graph_write_is_borrow` (definitionally
   main's `g.borrow()`), `write_program_same` (any interleaving of reads and the
   runtime's own `borrow_mut`s: same results, panics, final heap).
5. **Existing theorems — unaffected.** runtime_core's accessor models are functions of the
   world `W`, not of how the graph is borrowed (`Reuse.accessor_unchanged`,
   `materialize_unchanged`); `Plan.rtNew_spec` still holds (it does not model the new
   field). Cite shifts (proofs cite b6567cb97): all 48 runtime_core rows on `runtime.rs`
   move — +23 for `get_variables`…`new` (lines 209-406), +50 for `check_timeout` …
   `evaluate_id_filter`, +56 after it (6 of which come from 9212add39, not the PR) — and
   search_concurrency's `map_to_index_options` (1881 → 1937). The other 21 files keep
   every cited line. New fns needing rows at fold time: `GraphRef::deref` (:121),
   `Runtime::graph` (:530); `Runtime::new` gains the guard (fold `Cell.new` into
   runtime_core's `Plan.rtNew`).

## Pre-existing bug found (main and PR alike; not caused by #3436) — CONFIRMED live

`Protocol.writer_reply_race` / `writer_reply_race_rev`: a queued write (drainer thread)
commits, calls `session.release_locks()` (`graph_core.rs:858`) and then serializes its
reply through per-access borrows of its own — now published — version
(`reply.rs:428-470`). An **inline** write (inside `MULTI`: `query_sync` →
`begin_writer`, `graph_core.rs:1278`) that fails after a `Commit` then runs
`resync_published_indexes`, whose `committed.borrow_mut()` meets the live `AtomicRef`:
"already immutably borrowed" on the main thread, and the server dies. Repro:
`CREATE INDEX FOR (n:L) ON (n.v)`; 2 clients loop
`CREATE (x:Q {p0:0,…,p299:299}) WITH x UNWIND range(1,3000) AS i RETURN x`; 1 client loops
`MULTI; GRAPH.QUERY g "CREATE (n:L {v:$i}) WITH n RETURN 1/0"; EXEC`. The PR build and a
local main build (b6567cb97) both crash within ~1 s (backtrace:
`Pending::resync_published_indexes` ← `Runtime::resync_published_indexes` ←
`RedisModule_OnLoad::__do_command`); C (`bin/macos-arm64v8-release/falkordb.so`): 1 659
failed writes in 60 s, no crash, `:L` count 0. Suggested fix: serialize the reply before
`release_locks`, or make resync write the index without `borrow_mut` on the published
cell (the indexer locks it already takes are what guard the index).

## Assumptions (modelled, not proven)

* atomic_refcell-0.1.14 semantics, transcribed from its source (`Cell.B`, `borrow`,
  `borrowMut`, `release`) — a Rust crate, not FFI.
* `parking_lot::RwLock`: readers XOR one writer (`Protocol.step` `rlock`/`wlock`).
* Rust drop glue: struct fields in declaration order (`DropOrder.dropFields`).
* The protocol model over-approximates the interleavings (any enabled event fires); its
  enabling conditions transcribe the Rust (cites in `Protocol`); the `borrow_mut` audit
  table there is a grep over `graph/src` + `src` at d7d166e82.
* redis_layer's `RedisLayer.Ro` is itself a model of the parser/planner write
  classification, proven there.
-/
import Pr3436Review.Cell
import Pr3436Review.Protocol
import Pr3436Review.DropOrder
import Pr3436Review.Reuse
