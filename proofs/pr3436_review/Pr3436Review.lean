/-
# pr3436_review: PR #3436 "perf: a read query borrows its graph once, not on every access"

PR branch `perf/read-graph-once` @ 653f467a9 (d7d166e82 + 896ba73ef + 653f467a9 on
origin/main 9212add39), fixes #3431. Re-cited from d7d166e82 on 2026-10-21 after review
(cubic, Copilot, CodeRabbit) made 653f467a9: the guard now lives in `HeldRead { guard,
_owner }` with its own `Arc` of the cell, `g` is private behind a read-only
`graph_cell()`, and nine split-line `.g`/`.borrow()` sites now go through `graph()`. Build: `lake build` (Lean 4.35, no Mathlib, no `sorry`/`admit`/`axiom`). Imports
`runtime_core` (owner of `runtime.rs`) and `redis_layer` (owner of `graph_core.rs` /
`query_session.rs`) through Lake path dependencies. COVERAGE rows are commented out until
the PR merges.

The PR adds `Runtime.read_guard: Option<HeldRead>` (`runtime.rs:157`; `HeldRead`
`:115-118`), taken once in `Runtime::new` for `!write` (`:471-486`: clone `g` into
`_owner`, borrow through it, lifetime extended with `transmute`), `Runtime::graph() ->
GraphRef` (`:554-559`; `GraphRef` + `Deref` `:121-137`) and `Runtime::graph_cell()`
(`:565-567`); `g` is private (`:160`). Outside `runtime.rs`, 30 sites swap
`X.g.borrow()` → `X.graph()` and 24 swap `X.g` → `X.graph_cell()` (list in `Cell`).

| module | claim | Rust |
| --- | --- | --- |
| `Cell`      | 1, 4 | atomic_refcell-0.1.14 borrow word; `Runtime::new` guard, `GraphRef`, `Deref`, `graph()`, `graph_cell()`, every rewritten site |
| `Protocol`  | 2    | `MvccGraph` (`mvcc_graph.rs:112-270`), `QuerySession` (`query_session.rs:159-394`), read-runtime construction (`graph_core.rs:632,698`, `record.rs:84`), every `borrow_mut()` on a graph cell |
| `DropOrder` | 3    | `HeldRead` / `Runtime` drop glue, strong counts; every use of `g` in `runtime.rs` |
| `Reuse`     | 2 (premise), 5 | `RedisLayer.Ro` (parser `write` flag + planner `Commit`), `RuntimeCore.Accessors` |

## Verdicts

1. **Semantic equivalence — PROVEN.** `Cell.graph_deref`: whatever `graph()` returns derefs
   to `g`'s cell, as `g.borrow()` does (`borrowMain_deref`); `graph_same_result`: any
   `f : Graph → α` gives the same answer; `read_program_same`: a read runtime's whole
   read-only program returns the same answers on main and the PR whenever the cell is not
   mutably borrowed — which `Protocol.guard_never_excl` proves for every version a reader
   holds. `read_program_pr`: on the PR the reads never panic and never touch the borrow
   word (the perf claim's mechanism: no atomic RMW per access). `sites_same`: the same
   for any list of rewritten sites — the d7d166e82 sites and the nine 653f467a9 added
   (`procedures.rs:49,65,81`, `node_by_index_scan.rs:325`,
   `node_by_label_and_id_scan.rs:78`, `remove.rs:130`, `set.rs:287,322,352`).
   `graph_cell()` sites: `graph_cell_is_g` (it is `&self.g`), `graph_cell_borrowMut_same`,
   `graph_cell_borrow_same` (same cell, same outcome as main's `X.g`), `graph_cell_same`
   (`graph()` and `graph_cell()` name the same cell).
2. **Soundness of holding the read borrow — PROVEN; the PR's stated premise is REFUTED.**
   d7d166e82 said "a published version is never mutably borrowed again (writers mutate a
   private fork)" (896ba73ef has since rewritten the comment to state the lock invariant
   proven here, `runtime.rs:463-469`). False: `Pending::resync_published_indexes` (`pending.rs:1684`) calls
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
3. **Lifetime — PROVEN (653f467a9: no longer rests on `Runtime`'s field order).**
   `DropOrder.held_survives`: `HeldRead`'s drop (`guard`, then `_owner`) is defined
   whatever the cell's other holders do — all of them may be dropped, in any order,
   before or after it — because `_owner` is a strong reference only `HeldRead` releases.
   `held_sound_even_if_g_replaced`: the reviewers' scenario (`g`'s `Arc` dropped while
   the guard lives) is now harmless. `drop_sound`: a `Runtime` drop is defined in *either*
   field order (`runtimeFields` and `swappedFields`), interleaved with any drops of the
   other holders. What stays load-bearing is the order inside `HeldRead`
   (`held_swapped_unsound`: `_owner` before `guard` would free the cell under the guard).
   `drop_restores_borrow`, `move_irrelevant`. **`g` is never reassigned** — now a
   modelled fact: `g` is private and `runtime.rs` has no child module, so only
   `runtime.rs` names it; its seven mentions there are the initialisation (`:490`) and six
   `&self` reads (`:557,566,588,593,1467,1486`), with no `&mut self` method in the file;
   `GUse` enumerates them and `g_never_reassigned` / `g_uses_keep_ok` show they leave `g`
   and `read_guard` (and `Rt.ok`) as `new` set them. That enumeration is exhaustive by a
   grep of `runtime.rs` (not by Lean), and each use's action on the record is the
   identity by construction, so the theorem's content is the enumeration plus privacy.
   History (d7d166e82, bare guard, `pub g`): `pre653f467_drop_sound`,
   `pre653f467_swapped_unsound`, `pre653f467_swapped_ok_if_shared`, and
   `pre653f467_g_replaced_unsound` (replacing the `pub g` would free the cell under the
   guard — the reviewers' finding, reproduced in the model).
4. **Write runtimes unchanged — PROVEN.** `Cell.graph_write_is_borrow` (definitionally
   main's `g.borrow()`), `write_program_same` (any interleaving of reads and the
   runtime's own `borrow_mut`s: same results, panics, final heap).
5. **Existing theorems — unaffected.** runtime_core's accessor models are functions of the
   world `W`, not of how the graph is borrowed (`Reuse.accessor_unchanged`,
   `materialize_unchanged`); `Plan.rtNew_spec` still holds (it does not model the new
   field). Cite shifts at 653f467a9 (proofs cite b6567cb97; 9212add39's own +6 after
   `evaluate_id_filter` included): all 48 runtime_core rows on `runtime.rs` (+33 for
   `get_variables`…`new`, +58 for `check_timeout`, +82 for `write_escalation`…
   `evaluate_id_filter`, +88 after); search_concurrency `map_to_index_options`
   1881 → 1969; value_math's 18 `functions/procedures.rs` rows from `db_types` on (−1 to
   −3: 64→63, 81→79, then −3) and `index_ddl.rs` `drop_index` 88→91, `emit_effect`
   116→120; ops_apply and pending_commit `delete.rs` `delete_relationships_bulk`
   297→300, `delete_entity` 393→396. No other rewritten file moves a cited line. New fns
   needing rows at fold time: `GraphRef::deref` (:131), `Runtime::graph` (:554),
   `Runtime::graph_cell` (:565); `Runtime::new` gains the guard (fold `Cell.new` into
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
* Rust drop glue: struct fields in declaration order (`DropOrder.dropFields`); `Arc`
  drop = strong count − 1, free at 0.
* The protocol model over-approximates the interleavings (any enabled event fires); its
  enabling conditions transcribe the Rust (cites in `Protocol`); the `borrow_mut` audit
  table there is a grep over `graph/src` + `src` at 653f467a9 (re-run: same sites, now
  reached through `graph_cell()`).
* redis_layer's `RedisLayer.Ro` is itself a model of the parser/planner write
  classification, proven there.
-/
import Pr3436Review.Cell
import Pr3436Review.Protocol
import Pr3436Review.DropOrder
import Pr3436Review.Reuse
