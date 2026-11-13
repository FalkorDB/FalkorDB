# COVERAGE rows the CCH redesign makes stale

Design: `.handover/cch-redesign/design/03-design.md`, revision 3.1. PR numbers are 03 §9's.
Rows: `proofs/*/COVERAGE.tsv` on `origin/main` 8cb999d6d, buckets PROVEN or CONDITIONAL only.
Framework code: read at `origin/cch/08c-build-catch-up` 31ca1b802 ("@08c").

Column "@08c" says whether the CCH stack already makes the row stale before any of 03 lands.
Where it does, the design changes the code again.

## Rows the design changes or removes

| # | Row (`project/COVERAGE.tsv:line`) | Function | Proved today | What 03 does to it | PR | @08c |
|---|---|---|---|---|---|---|
| 1 | `native_index:139` (`versioned.rs`) | `VersionedIndex::writer_rebase` (trait) | Signature; column impl `Rebase.writerRebase_spec` | The hook receives the fork's entry (`ours`) and stops gating on `Written` (§5 "Writer rebase", §7 "Framework changes"). The signature changes, so the column impl's row is re-cited. | 4 | no |
| 2 | `native_index:140` (`versioned.rs`) | `VersionedIndex::indexer_catch_up` (trait) | Signature; column impl `indexerCatchUp_spec` | It reads the published entry; no `Written` gate (§5 "This replaces `CchIndex::indexer_catch_up`'s early return", §7). | 4 | no |
| 3 | `native_index:132`, `:133` (`falkordb_index.rs`) | column `writer_rebase`, `indexer_catch_up` | `writerRebase_spec`, `indexerCatchUp_spec` | Behaviour unchanged, signatures follow rows 1 and 2. Re-prove against the new signatures. | 4 | no |
| 4 | `native_index:147` (`versioned.rs`) | `Graph::rebase_index` | `rebase_progress`, `rebase_keeps_writes`, `rebase_index_spec`, `rebase_inv` | Passes the fork's entry to the hook. Under D6 it also runs for a `Built` fork (rebase table row 3). Today's theorems carry `by_this_version`, which CCH leaves empty (Appendix B V2.3). | 4, 9 | no |
| 5 | `native_index:146` (`versioned.rs`) | `Graph::overtaken` | `overtaken_proj`: same index, still `Building`, more batches | D6 drops `ours.is_building()` (`versioned.rs:759` @08c). This review: `overtaken_pre_d6`, `overtaken_d6`, `B2_pre_d6_reverts_job`. | 9 | no |
| 6 | `native_index:143` (`versioned.rs`) | `Graph::catch_up_index` | `catchUp_spec`, `catchUp_inv` (given `Frame`) | The CCH hook reads `theirs`' log and relinks `seq > fork`. Adoption absorbs or restarts; Restart keeps the generation as today. | 4 | no |
| 7 | `native_index:123` (`versioned.rs`) | `Graph::indexer_fit` | `fit_fresh`, `fit_after_writer`, … | A Built-entry job publish is "fitted against generation and ticket" (§7, D6): a new fit case or function. | 9 | no |
| 8 | `native_index:96` (`versioned.rs`) | `IndexEntry::indexer_apply` | `indexerApply_idle`, `indexerApply_building`, `indexerApply_clears_since` | D6 says "count the publish in the entry's `indexer_commits`" for a `Built` entry (§6). `indexer_apply` refuses a non-`Building` entry, and it is the only writer of the private `indexer_commits` (`versioned.rs:423-453` @08c). So this function changes, or a new method is added. | 9 | no |
| 9 | `native_index:129`, `versioned_matrix:128`, `search_concurrency:29` (`mvcc_graph.rs`) | `MvccGraph::commit_indexer_batch` | `commit_*` (native_index), `batch_keeps_forks` (search_concurrency) | Gains the job case if fold/S6 publishes go through it. `batch_keeps_forks` stays true as long as the job keeps the data version. | 9 | no |
| 10 | `versioned_matrix:127`, `search_concurrency:28` (`mvcc_graph.rs`) | `MvccGraph::commit` | `snapshot_isolation`, `refused_commit_publishes_nothing` | The enqueue loop also starts a job for a `Built` entry with a new ticket (§7, D6). The claims, which are about publishing, are unaffected; the cited lines move. | 9 | no |
| 11 | `native_index:167` (`cch/mod.rs`) | CCH `writer_record` | `cchVersionedUnreachable_panics` | Writers append seq'd log nodes and apply the bound/Overflow. CCH puts no ids in `Written` (§5 "The log"). | 4 | **yes**: adds ids to `by_this_version` (`cch/mod.rs:497` @08c) |
| 12 | `native_index:168` (`cch/mod.rs`) | CCH `writer_rebase` | `…_panics` (unreachable!) | The rebase-by-state table, rows 1–4 (§5). This review: `rebase_row1..3`, `row4_unreachable`, `priv_rebase`. | 4 | **yes**: `absorb`, early return on empty `written` (`cch/mod.rs:517-539` @08c) |
| 13 | `native_index:169` (`cch/mod.rs`) | CCH `indexer_catch_up` | `…_panics` | Relinks `seq > fork`, then adoption absorbs or `Restart`s (§5). This review: `batch_core`, `relink_exact`. | 4 | **yes**: early return on empty `written` (`:549-573` @08c) |
| 14 | `native_index:170`, `:171` (`cch/mod.rs`) | CCH `indexer_read`, `indexer_add` | `…_panics` | The phased batched build: ReadSkeleton → Order → Supergraph → Seed → Relax → Seal (→ Repair) (§5 "Phases"). | 8 | **yes**: one batch is the whole build (`:604-628` @08c) |
| 15 | `native_index:185` (`cch/mod.rs`) | CCH `memory_usage` | `…_panics` | Adds the retained log: 16 B per pair, plus each node (§7 "Accounting"). | 4 | **yes**: counts the `Cch` (`:630-634` @08c) |
| 16 | `native_index:163` (`cch/mod.rs`) | `Cch::recustomize_scoped` | `cchRecustomizeScoped_panics` (unimplemented!) | Gone on the stack, replaced by `partial.rs` `recustomize_seeded` (#3773). §6b changes that again: the front holds pairs, `:115-118` checks that a rank is present, the heap key is `(lo, hi, id)`, apex ids come from merging base and delta, and the budget counts work units. | 3, 4 | **yes**: function removed (`partial.rs:98` @08c) |
| 17 | `native_index:159`, `:160`, `:161` (`cch/mod.rs`) | `Csr::row`, `Csr::is_well_formed`, `Csr::is_symmetric_simple` | `row?_spec`, `isWellFormed_iff`, `isSymmetricSimple_iff` | Deleted with feral and the `CchInput` CSR (§9 PR 7). The topology becomes `Up`/`Down` base, Rust-owned CSR arrays loaded read-only into GraphBLAS, plus a `cow_btree` delta (§2, PR 3). If M1 fails, risk 3 keeps feral and this CSR behind PR 3's API. | 7 | no (still `cch/mod.rs:87` @08c) |

## Already stale on the @08c stack, not changed by 03

| Row | Function | Proved today | Why stale |
|---|---|---|---|
| `native_index:156`, `:157`, `:158` (`falkordb_index.rs`) | `FalkorDbIndex::cch_index`, `cch_defs`, `drop_cch` | `…_panics` (unimplemented!) | Implemented on the stack (`falkordb_index.rs:1232, 1242, 1314` @08c). 03 does not change them. |

## Checked and unchanged (the examples to confirm or refute)

| Row(s) | Verdict |
|---|---|
| `native_index:183`, `:184` CCH `indexer_insert`, `batch_len` | **Unchanged.** Still `unreachable!` at @08c (`cch/mod.rs:578-593`). 03 keeps CCH's catch-up as `Refresh`, so they stay unreachable; only the cited lines move. |
| `native_index:165`, `:166` CCH `entry`, `entry_mut` | **Unchanged.** |
| `native_index:144` `Graph::rebase_onto_indexer_batches`, `:149` `MvccGraph::rebase_onto_indexer_batches` | **Code unchanged.** |
| `native_index:145` `Graph::overtaken_by_indexer_batches` | **Code unchanged, but a new premise.** The gate is `batch_published_since` (`versioned.rs:729-747` @08c). A fold or S6 publish must move `batches_published`, as `MvccGraph::prepare(_, _, false)` does (`mvcc_graph.rs:310` @08c). Otherwise D6's `overtaken` is never asked, and a writer reverts the job (this review: `B4_job_publish_must_bump_batches`). 03 §7 does not list this. |
| `versioned_matrix:129`, `:130` `MvccGraph::is_stale`, `prepare` | **Unchanged, relied on.** `is_stale` is what makes rebase row 4 unreachable (`row4_unreachable`). `prepare` moves `batches_published` (above). |
| `cow_btree` rows (`CowBTree::new`, `insert`, `remove`, `range`, `contains_key`, `heap_bytes`, `Leaf::insert`, `Leaf::remove`; all PROVEN) | **Refuted: no change.** The delta is an existing `CowBTree` keyed by `(row ≪ 32) \| col`, with the arc id as doc (§2). S5's deletes use the existing `remove` (§6b, `cow_btree/mod.rs:20-23`). These rows back the design's "path-copy, O(log n) per insert" and "≤ 16 B per entry"; they keep holding. |
| `versioned_matrix` `tensor.rs` rows (`fwd_m`/`fwd_dp`/`fwd_dm` accessors, `fwd_iter` with `shadow_dp_wins`, `set_all_from_slices`/`remove_all` with `addBatch_correct`/`removeBatch_correct`, `flush`) | **Refuted: no change; relied on.** The full-seed pass scores `m⟨¬dm ∧ ¬dp⟩ ∪ dp`. It relies on `dp` shadowing `m` and on `dp ∩ dm = ∅` (`tensor.rs:52-108`, §5 "The full-seed pass"). 03 does not modify `tensor.rs`. |
| `versioned_matrix` `versioned_matrix.rs` rows | **Refuted: no change.** §2 rejects `VersionedMatrix<u32>` for the topology. |
| `graphblas_wrappers:1227`, `:1232` `CsrExport::row_count`, `copy_out_csr` | **NOT COVERED, so outside this table.** Deleted with the `CchInput` CSR (PR 7). So are the stack's own rows for the CSR export and `CchInput` (`cch/05-inputs` c42256475). |
| `online_build.rs` rows (`standing`, `Job::step`, …) | **NOT COVERED, but changed.** `standing` lets a job continue only while its generation is `Building` (`online_build.rs:447-457` @08c). A Built-entry job (D6) needs another rule, keyed by ticket. |
