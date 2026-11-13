import CchRedesignReview.Log
import CchRedesignReview.Model
import CchRedesignReview.Inv
import CchRedesignReview.Writer
import CchRedesignReview.Rebase
import CchRedesignReview.WriterStep
import CchRedesignReview.Builder
import CchRedesignReview.Step
import CchRedesignReview.Main
import CchRedesignReview.Findings

/-!
# CCH redesign review: the write log and its rules, proved for every interleaving

**What is reviewed.** `.handover/cch-redesign/design/03-design.md`, revision 3.1. Read in the worktree on branch
`cch/graphblas-redesign-handoff`, which is based on `origin/main` 8cb999d6d. The parts reviewed are the CCH
write log and the rules around it: §5 in full; §6 (flush, hand-off, decay, Built-entry jobs, fold job); §6b
(interleaving, where it runs and the bound, MVCC); §7 (versions, retention, framework changes); Appendices A and B.
The framework code 03 cites was read at `origin/cch/08c-build-catch-up` 31ca1b802 ("@08c"):
`versioned.rs:423-480` (`indexer_apply`, `publish`, `discarded_against`), `:551-584` (`fit`), `:677-706`
(`catch_up_index`), `:714-810` (`rebase_onto_indexer_batches`, `overtaken_by_indexer_batches`, `overtaken`,
`rebase_index`), `mvcc_graph.rs:155-160, 205-223, 262-310`, `online_build.rs:447-457`, `cch/mod.rs:497-634`,
`maint.rs:467-536`, `commit.rs:85-103`.

There is no Rust code for the design yet. This project models 03's rules, not functions, and adds no
`COVERAGE.tsv` rows. `STALE_WHEN_LANDED.md` lists the existing rows the design will make stale.

## The model

One CCH index as every published version holds it (`Pub`: data version, `batches_published`, entry), plus every
earlier published version (`Sys.hist`; any of them may be a fork that a writer or job still holds), every committed
write `(seq, pair)`, and the shared counter that generations, tickets and lineages come from. `Step` (Model.lean) is:

* `writer`: forks at a published version with the current data version (not stale, `mvcc_graph.rs:155-160`).
  It flushes or logs the segments before escalation on its fork, rebases if the gate and `overtaken` say so,
  runs the rest of its segments, applies retention and publishes at `ver + 1`;
* `batch`: reads on any published version, runs its phase (consumption for Supergraph, Seal and Repair) and
  publishes by the fit: discarded, Same, Swap, CatchUp with relink, adoption or `Restart`;
* `job`: a fold or S6 forked at any published version, fitted against generation and ticket;
* `other`: another index's batch (`Swap` for it).

Each step is atomic, and each actor's work is a function of its fork. Holding the lock from escalation to commit
serializes writers. So "fork from any earlier version, publish onto the current one" covers every interleaving
(03 §6b's table and all the rest). Budget, trigger and phase outcomes are free choices of the step.

The kind's data (topology, metric, ranks) is abstracted to `res p`: the version up to which the structure
reflects pair `p`. Consumers set it to their fork; flushes, adoption absorb and S6 replay set it to the version
they resolve against. A fresh build starts at its birth version, and slab reads and Seed are credited nothing more,
which is conservative. 03 §5 "Correctness" says a pair's state is a function of the current graph only; that is
what makes this abstraction sound.

## 03 section ↔ Lean

| 03 | Lean |
|---|---|
| §2 `CchLog`, `LogNode{seq, Pairs \| Overflow}` | `Node`, `NKind`, `Entry.log`, `Entry.pairs`, `sumPairs`, `covered` (Log.lean) |
| §2 `upkeep`; §7 "Versions" | `Upkeep`, `UKind`, `St`, `Entry` (Model.lean) |
| §5 "The log" (writers record pairs; `seq` = the writer's version; segments share it) | `appendE`, `seg` (Building arm), `writerSys` (writes at `ver + 1`) |
| §5 "Bound and Overflow" | `checkB`, `checkIf`, `covered` (Overflow arm), `consume` (rescan) |
| §5 "Consumption"; "Phases" | `consume`, `consumes`, `batchCopy`, `Phase`, `nextPhase` |
| §5 publish rules: Same/Swap, CatchUp relink | `fit` (`NI.discardedAgainst`), `relinked`, `batchResult` |
| §5 adoption, `Absorbed::Restart` | `adopts`, `batchResult` (adopting branch) |
| §5 "Writer rebase, by published state" | `overtaken` (D6; `NI.overtakenSE` before it), gate `NI.overtakenAny`, `rebase`, `relinkE`, `flushOwn`, `restart`, `refoldE` |
| §6 flush (resolve, allocate, classify, S2, recompute) | `flushB` (pairs resolved at `v`, own node kept in the private version) |
| §6 "Hand-off" | `handoffE` (new generation, base = the private hierarchy) |
| §6 "Decay": S7; fold/S6 triggers | `s7E`; `Out.setUp` |
| §6 "Built-entry jobs (D6)", "Fold job" | `jobResult`, `s6Res`, `jobSys` |
| §6 "Retention"; §7 "Retention" | `flushB` (clear past the bound), `retain` |
| §7 "Framework changes" | `overtaken` (D6), the job fit in `jobResult`, `batches` + gate |
| §5 "Readers while Building"; §6b "MVCC" | `serve` |
| §6b "Interleaving with a batched build" | `Step`, `Reach` (every interleaving) |

## Theorems (all for every reachable state, every bound `B`; Main.lean)

* `reach_inv` (Step.lean): `Inv` holds in every reachable state. It is an inductive invariant: `inv_init`,
  `inv_writer`, `inv_batch`, `inv_job`, `inv_other`.
* **1. `no_lost_write`** (§5, §6, §7; Appendix B's closing claim). In every published version `S` at or after `s`,
  a write `(s, p)` is resolved in `S`'s structure at a version `≥ s` or still in `S`'s log, Building and Built.
  `nothing_from_the_future`: `res ≤ ver` and `seq ≤ ver`, so a resolution comes from a fork, flush or replay at or
  after the write.
* **2. `building_serves_fallback`, `served_is_current`** (§5 Readers, §6b MVCC). Building serves the fallback.
  What is served has every committed write resolved.
* **3. `built_no_upkeep_no_log`, `built_upkeep_log_exact`** (§6/§7 Retention). Without `upkeep` the log is empty.
  With `Some{since}` it is exactly the writers' nodes after `since`: every write after `since` is in a node of
  its seq, every node is a committed write after `since`, and there is no Overflow.
* **4. `stale_job_gone`, `job_publish_fitted`, `job_publish_sound`** (§6 "Publish under the lock", §7, D6). A job
  whose generation or ticket does not match publishes nothing and leaves the system unchanged. One that publishes
  matched, resolves every write, and clears `upkeep` and the log.
* **5. `building_bound`, `pairs_counter_exact`, `overflow_covers`, `consume_overflow_rescans`,
  `consumed_does_not_come_back`** (§5 Bound). While Building, the log is under `B` or holds an Overflow. The
  counter is exact. An Overflow at `o` stands for every write up to `o`, and its consumer rescans; no pair is lost
  (via 1). After a consuming batch publishes, every node left is newer than its fork, so a consumed Overflow does
  not come back.
* **Rebase table** (§5). `rebase_row1`, `rebase_row2`, `rebase_row3` are the rows as written. `priv_rebase`
  (Rebase.lean): every reachable row keeps the writer's own pairs. `row4_unreachable`: a `Built` version is
  followed in its generation only by `Built` ones.
* **Other claims.** `gen_const_within_version`. `commits_monotone`: no batch or job publish is reverted within a
  generation, which is the D6 claim of Appendix B V1.5/V2.4. `relink_exact`: CatchUp's walk "stopping at the first
  `seq ≤ fork`" is exactly the published nodes after the fork. `fit_discard_is_ni`, `overtaken_pre_d6`,
  `overtaken_d6`: these tie the fit and `overtaken` to native_index's proven models of `discarded_against` and
  `overtaken`.

## Readings chosen (where 03 is ambiguous)

* **R1** (§2, §6b). The structure is abstracted to `res`, and the skeleton is credited only the build's birth
  version (above).
* **R2** (§6 Retention). "Past the §5 bound, the writer clears upkeep and the log" means the retained nodes; the
  writer's own nodes stay in its private version (§5: "A writer always keeps its own nodes in its private version
  until it publishes, whatever the entry state"). Read literally, it is finding F1.
* **R3** (§5 row 3, §6 Fold job). "Re-apply the fold to the fork's copy" gives the fork's hierarchy with `upkeep`
  and the retained log cleared and the published count, as the fold's own publish does. Keeping the fork's
  `upkeep` instead re-enqueues a fold that already published (wasted work, no lost write; not separately checked).
* **R4** (§5 adoption). An Overflow among the relinked nodes cannot be absorbed under the lock (§6b: rescans are
  off the lock only), so it forces `Restart`. `B5` shows the alternative loses writes.
* **R5** (§5 Bound). The writer checks the bound after every change it makes to a Building log: append, hand-off,
  S7, relink and Restart. A check only on append is finding F4.
* **R6** (§5, `commit.rs:85-103` @08c). Segments may flush on the fork before escalation, and any segment's flush
  may hand off (N_max or B_gil). The rebase sees the fork's copy after those segments. This makes F2 reachable.
* **R7** (§7, D6). A fold or S6 publish moves `batches_published`, as `prepare(_, _, false)` does
  (`mvcc_graph.rs:310` @08c). The alternative is F5.
* **R8** (§5 publish rules). For this index, `Swap` is `Same` (the batch's copy). An adopting batch on Same or
  Swap has nothing to absorb and becomes Built.
* **R9** (§5 CatchUp). In "`pairs` = the batch's remainder plus theirs", "theirs" means the relinked nodes' pairs.
  The other reading is F3.
* **R10** (§6 Decay). A writer may set `upkeep` while another is set (overwrite, a superset of behaviours). S7 is a
  new generation and lineage whose build reflects every write up to its version.
* **R11**. A writer `Restart` takes the published copy's count (`rebased = theirs.clone()`, `versioned.rs:786`
  @08c) and a new generation. A hand-off is a new entry (count 0) on the base's lineage.
* **R12**. The bound `B` (`max(1M, arcs/8)`) is a constant.

## Findings (design-level, no code to reproduce; traces in Findings.lean)

* **F1 (major under the literal reading of §6 Retention).** The S6 row of the rebase table loses a write
  (`F1_literal_clear_loses_write`). The interleaving: writer W3 forks before an S6 publish, and its pre-escalation
  flush passes the bound together with the retained nodes. Reading "clears upkeep and the log" literally clears
  W3's own node too, so row 3 (S6) "replay[s] the own nodes" over nothing. Published version 3 has pair 9 neither
  resolved (`res = 2`) nor logged. This contradicts §5's "A writer always keeps its own nodes". Fix: in §6, "clears
  `upkeep` and the retained nodes (those with `seq <` its version)". Under that reading (R2), `no_lost_write` holds
  (`F1_reading_R2_keeps_it`).
* **F2 (minor).** A segment that hands off before the writer escalates reverts a published S6 (or fold)
  (`F2_preescalation_handoff_reverts_s6`). The hand-off gives the private copy a new generation, so `overtaken`,
  which needs the same generation, is false. The copy, whose base is the pre-S6 hierarchy, then replaces the
  published S6 lineage (2 → 0). No write is lost (`no_lost_write`), but the S6's work is, and the decay must
  trigger again. That contradicts 03's "A writer whose fork predates their publish no longer reverts them (§6,
  D6)". After escalation the same abort keeps the S6 lineage (`F2_after_escalation_keeps_s6`). Fix: rebase before
  the first segment's flush (escalate first), or decide `overtaken` on the fork's original entry.
* **F3 (minor, wording).** Read with "theirs" as the published counter, CatchUp's "`pairs` = the batch's
  remainder plus theirs" counts the nodes before the fork twice: 3 for a 2-pair log (`F3_counter_reading`). Spurious
  Overflows follow, and with them full rescans. Say "plus the relinked nodes' pairs".
* **F4 (minor, wording).** If the bound is checked only when a segment is appended, a hand-off of a large segment
  leaves a Building log over the bound with no Overflow (`F4_bound_only_on_append`). Say that the writer checks
  the bound whenever it creates or changes a Building log.
* **F5 (minor, missing framework change).** §7's D6 list leaves out that a fold or S6 publish must move
  `batches_published`. `overtaken_by_indexer_batches` gates on `batch_published_since` (`versioned.rs:729-747`
  @08c). Without the bump, D6's `overtaken` is never asked, and the writer reverts the job
  (`B4_job_publish_must_bump_batches`).

## Break checks (the invariant sees the bugs Appendices A and B fixed)

* `B1_no_relink_loses_write`: the rebase as at @08c, with the published copy and only `by_this_version` (V2.3).
* `B2_pre_d6_reverts_job`: `overtaken` without D6 (V1.5/V2.4). The count falls 7 → 6 within the generation.
* `B3_catchup_without_relink_loses_write`: CatchUp treated as Same (A R1.1, B V1.1).
* `B5_adoption_must_not_absorb_overflow`: adoption letting an Overflow pass as absorbed (R4).

Each loses a write or reverts a publish on a reachable state, and passes under the design's rules.

## Gaps

* **Abstracted away.** Topology, metric, ranks and rank allocation (I1′, P, I3′), Lemmas A and B, S2, the
  full-seed pass, the fold's bases and deltas (`added_since_fold` with `seq > F`) and S6's input order are all
  reduced to `res` and `lineage`. Their correctness is M5's, or another proof's.
* **Safety only.** Budgets, decay triggers and phase ends are free choices. Liveness is not modelled: convergence,
  `MAX_REDOS`, D2, and whether a job is ever enqueued (`indexes_to_build_since`, `standing`, tickets in the
  builder's job key). Jobs may fork from any held version at any time, which is a superset.
* **One index**, with other indexes' batches as `other`. The deadline that one lock hold shares across indexes
  (§6b) is not modelled. Replicas' `GRAPH.EFFECT` is a writer here; RDB load, `DROP INDEX` and recreate are not
  modelled beyond S7.
* `B` is a constant; `u64` counters are unbounded `Nat`; pairs are abstract ids. Self-loops, the node-delete
  cascade and per-type/direction weights live inside "pair state = f(graph)" (03 §5 "Correctness"), which is
  assumed.

## Reuse

From `proofs/native_index` (Lake path dependency): `NI.discardedAgainst`, `NI.discardedAgainst_none_iff`,
`NI.overtakenSE`, `NI.overtakenAny`, `NI.overtakenAny_iff`, `NI.IndexerCommit`, `NI.SE`. The CCH entry's
`(generation, commits, Building?)` projection is `Entry.toSE`.

`bash proofs/lean_ci.sh cch_redesign_review` passes, with no `sorry`, `admit`, `native_decide` or `axiom`.
-/
