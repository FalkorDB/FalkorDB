/-
# pr3436_review: PR #3436 "perf: a read query borrows its graph once, not on every access"

PR branch `perf/read-graph-once` @ e30861ea3 (d7d166e82, 896ba73ef, 653f467a9, merge
of origin/main b97da5a1c = 30f8d7cff, e30861ea3; main b84473a7f differs from b97da5a1c only
in Cargo files), fixes #3431. History of re-cites: d7d166e82 → 653f467a9 (2026-10-21,
review by cubic/Copilot/CodeRabbit: the guard lives in `HeldRead { guard, _owner }` with
its own `Arc`, `g` private, nine split-line sites through `graph()`) → e30861ea3
(2026-10-22, Dvir's review: `graph_cell()` removed; `graph_mut()` (asserts `write`) for
every exclusive borrow; `graph_arc()` (an owned clone) for the two places that keep the
graph; the commit/effects functions take `&mut Graph` / `&Graph`). Build: `lake build`
(Lean 4.35, no Mathlib, no `sorry`/`admit`/`axiom`). Imports `runtime_core` (owner of
`runtime.rs`) and `redis_layer` (owner of `graph_core.rs` / `query_session.rs`) through
Lake path dependencies. COVERAGE rows are commented out until the PR merges.

The PR adds `Runtime.read_guard: Option<HeldRead>` (`runtime.rs:157`; `HeldRead`
`:115-118`), taken once in `Runtime::new` for `!write` (`:472-487`: clone `g` into
`_owner`, borrow through it, lifetime extended with `transmute`), `Runtime::graph() ->
GraphRef` (`:555-560`; `GraphRef` + `Deref` `:121-137`), `Runtime::graph_mut() ->
AtomicRefMut` (`:572-575`) and `Runtime::graph_arc()` (`:583-585`); `g` is private
(`:161`). Every other file only swaps how it reaches the graph (list in `Cell`).

| module | claim | Rust |
| --- | --- | --- |
| `Cell`      | 1, 4 | atomic_refcell-0.1.14 borrow word; `Runtime::new` guard, `GraphRef`, `Deref`, `graph()`, `graph_mut()`, every rewritten site |
| `Hold`      | 1 (e30861ea3) | the commit/effects functions now run under one caller-held borrow |
| `Protocol`  | 2    | `MvccGraph` (`mvcc_graph.rs:112-270`), `QuerySession` (`query_session.rs:159-394`), read-runtime construction (`graph_core.rs:632,698`, `record.rs:84`), every `borrow_mut()` on a graph cell |
| `DropOrder` | 3    | `HeldRead` / `Runtime` drop glue, strong counts; every use of `g` in `runtime.rs`; `graph_arc()` |
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
   **`graph_mut()` (e30861ea3)**: `graph_mut_write_same` (on a write runtime it is main's
   `g.borrow_mut()`: same cell, panic, borrow word), `graph_mut_read_asserts` (on a read
   runtime the `assert!` fires before the cell is touched — unreachable, since a read plan
   has no such site), `graph_mut_same_cell` (it borrows the cell `graph()` reads).
   **Caller-held borrows (e30861ea3, `Hold`)**: `Pending::commit`, `end_segment`,
   `IndexDocs::commit`/`commit_deferred_indexes` now run under one `&mut graph_mut()`, the
   `&Graph` readers (`set_schema_baseline`, `enforce_constraints`, resync's `private`, the
   effects builders) under one `&graph()`. `Hold.commit_same`, `single_mut_same`,
   `single_rd_same`: for every external borrow state and every pending state, the result
   (panic / error / committed graph) is main's — because main's `Pending::commit` always
   reaches its unconditional `apply_index_writes` `borrow_mut` before any shared borrow
   (`commitSteps_head_mut`), and the external state is constant for the call:
   `Protocol.priv_borrows_owner_only` proves every live borrow of a private version is its
   owner's (the indexer's `Arc` only ever points at a published version; the population
   thread's borrows are modelled as borrows of any published version). **No
   refutation** — nothing inside `Pending::commit` borrows the cell through another
   handle. `Hold.empty_body_differs` records the only shape that would differ (a body
   with no borrow on main), which none of the converted functions has. Live (e30861ea3
   build, 90 s): indexed CREATE ×2, SET/REMOVE, DETACH DELETE, failing writes, `MULTI`
   failing writes, range-index CREATE/DROP loops (population threads), 4 reader threads —
   2 772 + 1 358 + 1 388 + 1 365 + 7 011 + 1 438 writes, 23 422 read queries, no panic.
   History (653f467a9): `Cell.pre_e30861e_graph_cell_*`.
2. **Soundness of holding the read borrow — PROVEN; the PR's stated premise is REFUTED.**
   d7d166e82 said "a published version is never mutably borrowed again (writers mutate a
   private fork)" (896ba73ef has since rewritten the comment to state the lock invariant
   proven here, `runtime.rs:464-470`). False: `Pending::resync_published_indexes` (`pending.rs:1670`) calls
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
   `drop_restores_borrow`, `move_irrelevant`. **`g` is never reassigned** — a modelled
   fact: `g` is private and `runtime.rs` has no child module, so only `runtime.rs` names
   it; since e30861ea3 its four mentions there are the initialisation (`:491`) and three
   `&self` reads — `graph` `:558`, `graph_mut` `:574`, `graph_arc` `:584` — with no
   `&mut self` method in the file; `GUse` enumerates them and `g_never_reassigned` /
   `g_uses_keep_ok` show they leave `g` and `read_guard` (and `Rt.ok`) as `new` set them
   (653f467a9's six-use enumeration kept as `pre_e30861e_GUse`). **`graph_arc()`**:
   `graph_arc_keeps_g` — it returns a clone of `g`'s `Arc` (same cell, strong count + 1)
   and leaves the runtime record untouched; dropping the clone is one more "other
   holder" for `held_survives`. That enumeration is exhaustive by a
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
   field). No project models the cell parameter e30861ea3 removed from `pending.rs` /
   effects (grep `AtomicRefCell|borrow_mut` in pending_commit, effects_*, replication,
   index_layer: none), so their theorems carry over; only cites move. Cite shifts at
   e30861ea3 against the proofs' main b84473a7f (rows that resolve there): runtime_core
   all 48 `runtime.rs` rows (+34 for `get_variables`…`new`, +59, +100, then +102 from
   `write_escalation` on); search_concurrency `map_to_index_options` 1887 → 1989;
   value_math 18 `procedures.rs` rows (−1 to −3) and `index_ddl.rs` `drop_index` 88→87,
   `emit_effect` 116→114; pending_commit 53 `pending.rs` rows (−1/−2 up to `commit`
   1228→1226, −13/−14 after: `enforce_constraints` 1385→1372, `resync_published_indexes`
   1635→1621, `commit_deferred_indexes` 1690→1676, `end_segment` 1731→1717,
   `effects_count` 1804→1790); index_layer `resync_published_indexes` 1635→1621;
   effects_codec 35 rows in `effects/mod.rs` (−1) and `effects/v3/format.rs` (−2);
   effects_emit_apply 16 and replication 24 rows in `effects/v3/emit.rs` (−1/−2),
   `format.rs` (−2), `pending.rs`. New fns needing rows at fold time:
   `GraphRef::deref` (:131), `Runtime::graph` (:555), `Runtime::graph_mut` (:572),
   `Runtime::graph_arc` (:583); `Runtime::new` gains the guard (fold `Cell.new` into
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
  table there is a grep over `graph/src` + `src` at e30861ea3 (re-run: same sites; the
  runtime ones now through `graph_mut()` or a caller-held `graph_mut()`).
* redis_layer's `RedisLayer.Ro` is itself a model of the parser/planner write
  classification, proven there.
-/
import Pr3436Review.Cell
import Pr3436Review.Protocol
import Pr3436Review.DropOrder
import Pr3436Review.Reuse
import Pr3436Review.Hold
