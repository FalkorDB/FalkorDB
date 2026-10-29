# versioned_matrix: the delta matrices, multi-edge tensors and MVCC snapshots under every graph

This project proves that the storage layer under the graph means what it claims. It covers four things:

- **`VersionedMatrix`:** a base matrix plus two deltas (`dp` for added entries, `dm` for deleted ones).
- **`Tensor`:** relationship storage, with multi-edges kept in a separate `me` store.
- **The iterators** that read both.
- **`Cow` and `MvccGraph`:** the copy-on-write versions that let readers and a writer work at the same time.

Every label matrix, the adjacency matrix and every relationship-type tensor is one of these, so every query depends on them:

- `MATCH` returning exactly the stored nodes and relationships, each once;
- parallel relationships (two `:KNOWS` between the same pair) kept apart and counted right;
- `DELETE` taking effect even before the deltas are folded into the base matrix;
- a read query seeing one frozen version while a write query runs.

The full Lean↔Rust mapping is in the header of [`VersionedMatrix.lean`](VersionedMatrix.lean) and in the `file:line` table at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Is the three-layer matrix the matrix it pretends to be?**

A `VersionedMatrix` stores `m`, `dp` and `dm`. What it *means* is the logical matrix `(m ∖ dm) ∪ dp`. `run_repr` proves that after **any** sequence of operations, the logical matrix equals a plain reference set of coordinates and every layer invariant still holds. The operations covered are set, remove, bulk set, bulk product, masked remove, grow, shrink, dup, fold, flush and wait, with any fold decisions. `get_iff_eff` proves `get` answers from that logical matrix, and `nvals_eq` proves the unsigned `|m| + |dp| − |dm|` neither underflows nor miscounts.

**2. Does iterating yield every entry once, in order?**

`drain_eq` proves the three-way merge iterator yields exactly the logical matrix, sorted and without duplicates, with `dp`'s value winning on a shared key. It needs less than the layer invariant: a stray tombstone or a shared key is handled correctly. `newIt_abs` and `seek_abs` prove that construction and `seek` give the same stream. That includes seeking after a layer was skipped, the #2430 regression (`seek_after_a_skipped_layer_still_reads_the_delta`).

**3. Are multi-edges stored correctly?**

A pair with one edge stores the id inline. A pair with more is marked `MULTI_EDGE` and its ids live in a row of `me`. `addBatch_correct` and `removeBatch_correct` prove that adding or removing any batch keeps every invariant and changes the pair's edges by exactly the batch. That includes promotion to multi-edge, demotion back to one inline id, and cancelling back to the committed state. `colDegree_eq` proves `col_degree` never reaches its `unreachable!` and counts every incoming edge.

**4. Does a reader see a frozen snapshot while a writer works?**

`snapshot_isolation` proves that along any history (begin, writes, commit, rollback, new writers) no handle a published version references ever changes. So `reader_view_frozen` holds: a reader's view of any committed version stays fixed. `single_writer` proves a second `begin` while writing is a no-op. Since #2846, `refused_commit_rolls_back` proves a version that fails `Graph::validate` is never published and frees the write flag.

The project also proves where isolation stops. `clone_of_owned_is_not_isolated` shows a shallow `clone` of an owned layer shares a writable handle. `old_write_leaks_into_new_version` shows copy-on-write only protects one direction: a write through the *old* version lands in the new one. Isolation holds because the engine only ever writes the in-flight version. Since #2905 that rule is documented on `Cow` and debug builds assert it (proved in `runtime_ds`: `CowModel.dbg_write_through_branched_source_panics`); release builds behave as before.

**5. Is the fold policy only about performance?**

Yes. Every layer theorem quantifies over every fold decision, so `should_fold` can never change a query result. What the policy itself guarantees:

- `write_implies_read`: whatever the write path folds, the read path folds too;
- `fresh_version_decision`: a version that has added nothing cannot newly decide to fold.

**6. Can the row filter skip a row that has data?**

No. `mayHold_sound` (bit-exact over `BitVec 64`) proves `may_hold` never answers "no" for a recorded row in range. `slot_word_lt` proves the word index is always below 512, so `bits[w]` cannot panic. `mayHoldRows_sound` proves the filter stays sound through every `Delta` method.

**7. Is `count_in_rows` exact?**

`vmCountInRows_spec` proves `VersionedMatrix::count_in_rows` returns the number of logical entries in the selected rows, and that its `u64` subtraction cannot underflow (#3022). This is **CONDITIONAL** on `LayerCount`, the spec of `Matrix::count_in_rows` per layer. `countPairsInto_spec` covers `Tensor::count_pairs_into`.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `run_repr` | After any operation sequence with any fold decisions, the logical matrix is the reference set and the invariants hold | `DeltaProofs.lean` |
| `step_repr` | The one-step case behind it, for every operation | `DeltaProofs.lean` |
| `fold_eff`, `flush_eff` | Folding and flushing never change the logical matrix | `DeltaProofs.lean` |
| `get_iff_eff`, `nvals_eq` | `get` and `nvals` read the logical matrix; no underflow | `DeltaProofs.lean` |
| `drain_eq` | The merge iterator yields the logical matrix, sorted, each entry once | `Iter.lean` |
| `newIt_abs`, `seek_abs` | `Iter::new` and `seek` produce exactly that stream | `IterSeek.lean` |
| `addBatch_correct`, `removeBatch_correct` | Adding or removing edges on a pair keeps the multi-edge invariants and changes exactly the batch | `TensorProofs.lean` |
| `colDegree_eq` | `col_degree` never hits `unreachable!` and counts every incoming edge | `TensorOps.lean` |
| `resizeT_edges`, `resizeT_allInv` | `Tensor::resize` never shrinks (#3250), so it never drops an entry and keeps every invariant | `TensorResize.lean` |
| `snapshot_isolation`, `reader_view_frozen` | A published version's matrices never change | `Cow.lean` |
| `refused_commit_rolls_back` | A version that fails validation is never published | `Cow.lean` |
| `mayHold_sound`, `slot_word_lt` | The row filter never wrongly answers "no" and never indexes out of bounds | `RowFilter.lean` |
| `write_implies_read` | The read path folds whenever the write path would | `Policy.lean` |
| `decode_encode` | `VersionedMatrix` encode/decode round-trips | `VMOps.lean` |

### How `run_repr` is proved

1. **Model the type.** `VM` holds the three layers `m`, `dp`, `dm` as duplicate-free lists of coordinates, plus the dimensions and the latched fold decision. Bool layers only store `true`, so a layer's pattern is its contents. Each GraphBLAS primitive (`setElement`, `removeElement`, `eWiseAdd`, the masked transpose, `resize`, …) is a definition with its spec meaning; the table in `Delta.lean` gives the correspondence. No `axiom` is used.
2. **Define the meaning.** `eff v p` is `(p ∈ m ∧ p ∉ dm) ∨ p ∈ dp`. The reference `Ref` is a plain set of coordinates with dimensions, and `Ref.step` says what each operation should do to it.
3. **State the invariant.** `Inv` is the set of layer facts the Rust doc comments rely on:
   - `dp` and `m` are disjoint;
   - `dm ⊆ m`;
   - no layer has duplicates;
   - every entry is in bounds.
4. **One step.** `step_repr` is a case split over the operation type `Op`. Each case shows the invariant is kept and `eff` moves exactly as the reference moves.
5. **Any sequence.** `run_repr` is induction on the list of operations.
6. **Hypotheses.** `Pre` is the caller contract of each operation:
   - `set` is in bounds;
   - the bulk inserts are in bounds;
   - the bulk inserts' "these entries are new" flag is only passed for entries not already in `m`.

   These are documented preconditions of the Rust methods. This project does not prove that every caller meets them.

## Bugs found

See [`../FINDINGS.md`](../FINDINGS.md) for the repro of each.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W4-matrix-1 | removed (17ccc5412) | `Tensor::resize`'s shrink never trimmed `me`. Orphan multi-edge rows survived: `iter_edges` yielded phantom ids and `edge_count` was too high (repro `tensor_shrink_keeps_orphan_me_rows`) | **Fixed by #3250 (#3246):** `resize` never shrinks; `resizeT_edges` and `resizeT_allInv` prove no entry is dropped and every invariant holds, with no hypothesis |
| #2898 | — (mechanism shown by `clone_of_owned_is_not_isolated`) | `versioned_matrix::Iter` holds its layer's handle with no borrow of the owner, so an in-place write can land under a live iterator (Rust probe `probe_iter_outlives_in_place_mutation`) | **Open, latent:** issue #2898, needs a design choice; no PR |
| #2903 | `old_write_leaks_into_new_version` | Copy-on-write is one-directional: a write through an older version leaks into the newer one. Safe today because only the in-flight version is ever written (repro `cow_is_one_directional_old_version_write_leaks`) | **Latent, documented (release):** PR #2905 (merged, main 1d15bf713) documents the contract and debug builds assert it; release behaviour, and so this counterexample, is unchanged |

**CONDITIONAL row:** `VersionedMatrix::count_in_rows` rests on `LayerCount`, which says `Matrix::count_in_rows` counts a layer's entries in the selected rows. `proofs/graphblas_wrappers` proves that as `countInRows_spec`, but the two projects are not linked, so the hypothesis is not discharged here.

## Files

| File | What's in it |
|---|---|
| `Delta.lean` | `VM`, the logical matrix `eff`, `Inv`, every layer mutation, the operation language `Op` and the reference `Ref` |
| `DeltaProofs.lean` | `step_repr`, `run_repr`, `fold_eff`, `flush_eff`, `get_iff_eff`, `nvals_eq` |
| `Mat.lean` | A GraphBLAS matrix handle as the bookkeeping layer sees it (entries, dimensions, pending work) |
| `DeltaBook.lean` | `Delta<T>` with its counters, latched fold and row filter; `mayHoldRows_sound`, `fresh_version_decision` |
| `RowFilter.lean` | Bit-exact `RowFilter`: `mayHold_sound`, `slot_word_lt` |
| `Policy.lean` | The fold policy (`should_fold`, `should_fold_read`, `delta_dominates_base`) |
| `VMOps.lean` | `wait`, `transpose`, `clear_deltas`, `from_matrix`, encode/decode and the remaining methods |
| `Iter.lean` | The three-way merge iterator: `drain_eq` |
| `IterSeek.lean` | `Iter::new`, `from_layers(_detaching)`, `seek` |
| `CountRows.lean` | `count_in_rows` and `count_pairs_into` (#3022) |
| `Tensor.lean` | One `(src, dst)` pair of a `Tensor`: inline / `MULTI_EDGE` encoding, add and remove batches, the pair invariant |
| `TensorProofs.lean` | `addBatch_correct`, `removeBatch_correct`, `compoundKey_inv` |
| `TensorBlocks.lean` | The whole tensor: `me` blocks, widening, multi-pair counting |
| `TensorOps.lean` | Degrees, extract, flush/fold/dup/wait, iterator plumbing |
| `TensorResize.lean` | `resize` (grow-only since #3250; W4-matrix-1 history), `nrows`/`ncols`, encode's inline values, `EdgeIds::size_hint`, `Iter::new`/`seek` |
| `Cow.lean` | Copy-on-write handles and the MVCC history: `snapshot_isolation`, `refused_commit_rolls_back`, the isolation limits |
| `CowGlue.lean` | `Cow::new`, `Cow::deref`, `MvccGraph::new`/`from_graph`/`drop` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The lakefile has no path dependencies. Other projects rely on its results by citing them: `graph_queries` treats a matrix as its coordinate set because of `run_repr`.

## Coverage

`COVERAGE.tsv` has 132 rows: 69 in `versioned_matrix.rs`, 51 in `tensor.rs`, 7 in `mvcc_graph.rs` and 5 in `cow.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 131 |
| CONDITIONAL | 1 |

- **CONDITIONAL:** `VersionedMatrix::count_in_rows`, on `LayerCount` (above).

## What the model does not cover

- **GraphBLAS calls are definitions with their spec meaning,** not checked against the C library. Pending tuples, `wait` and `resync` never change which coordinates a layer stores, so the set-level model leaves them out. `DeltaBook.lean` and `VMOps.lean` model the counters and pending flags separately.
- **The `Tensor` proofs work one pair at a time.** That pairs are independent rests on `compound_key` being injective (`compoundKey_inv`) and on each batch decision touching only its own pair.
- **Concurrency is modelled as a sequence of events with one writer.** Atomics are plain values (every access is `Relaxed` under the single-writer rule). Thread interleavings and `Arc` reference counts are not modelled here; `proofs/endpoint_index` and `proofs/graphblas_wrappers` model those for their own types.
- **`Iter`'s missing lifetime (#2898) is documented, not modelled:** the iterator model consumes values, not live handles.
- **The `Pre` caller contracts** of `set` and the bulk inserts are assumed, not proven for each caller.

## Build and check

```bash
cd proofs/versioned_matrix && lake build    # this project
bash proofs/lean_ci.sh versioned_matrix     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                     # repo-wide coverage report
```

**When you change `versioned_matrix.rs`, `tensor.rs`, `cow.rs` or `mvcc_graph.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
