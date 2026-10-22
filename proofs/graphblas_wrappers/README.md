# graphblas_wrappers: the Rust glue around the GraphBLAS C library

This project proves what the Rust wrappers in `graph/src/graph/graphblas/` (`matrix.rs`, `vector.rs`, `mod.rs`, `serialization.rs`, and the decode and iterator parts of `tensor.rs`) do around each GraphBLAS C call. It also shows, with concrete counterexamples, where they let a bad input or an unlucky schedule through. The C library itself is not verified: each C call is stated by its documented contract, and the Rust logic around it is proved. User-facing behaviour that depends on this layer:

- `GRAPH.RESTORE`, `GRAPH.COPY`, RDB load and replica full sync, which decode attacker-controlled bytes;
- the server staying up when GraphBLAS returns an error code;
- every matrix scan visiting each stored entry once;
- GraphBLAS memory being freed once, not twice and not never;
- the bulk row count behind `first_node_with_relationships` (#3022).

The full Lean↔Rust mapping is in the header of [`GraphblasWrappers.lean`](GraphblasWrappers.lean) and at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Can a crafted `GRAPH.RESTORE` payload crash the server?**

**Not through a corrupt id blob any more: #2996 fixed it.** Every GraphBLAS call returns a status code. A wrapper either `assert_eq!`s it (a panic, which in the module is a process exit), `debug_assert_eq!`s it (ignored in release), or returns it as an `Err`.

- Before #2996, `Vector::<bool>::decode_blob` and `Vector::<u64>::decode` `assert_eq!`ed on `GxB_Vector_deserialize`, which returns `GrB_INVALID_OBJECT` on a corrupt blob. `pre2996_decodeBlob_is_dos`, `pre2996_decodeU64_is_dos` and `assertPanic_deserialize_dos` prove that the error was reachable and killed the process.
- On main both return the code as an `Err`, and pass the expected type so a blob of the wrong type is refused too. `decodeBlob_never_dies`, `decodeU64_never_dies` and `decodeBlob_safe_on_reachable` prove no code can crash through them; `hardened_eq_blob` says `decode_blob` now behaves exactly like the already-hardened `Vector::<bool>::decode` (`decodeBool_safe`, `decodeBool_never_dies`).
- #2996 also made `Tensor::decode` refuse delta layers whose size differs from the base: `checkLayers_merge_in_bounds` proves an accepted payload only merges entries inside the base (`pre2996_accepts_out_of_bounds_delta` is the old behaviour).

**2. Can a decoded tensor break its own invariant?**

**Not any more: #3033 fixed it.** Promotion completeness says a pair has an `me` row exactly when its forward value is the `MULTI_EDGE` marker. Before #3033, `Tensor::decode` did not check it.

- `edgeCount_no_panic` and `edgeCount_value` prove `edge_count` cannot underflow while the invariant `Wf` holds.
- History: `pre3033_orphan_row_accepted`, `pre3033_orphan_row_breaks_wf` and `pre3033_orphan_row_panics` show the state a crafted payload produced, in which `edge_count` panicked. `pre3033_more_me_rows_than_pattern_panics` generalises it to any decoded tensor with more `me` rows than forward pairs. `Tensor::encode` calls `edge_count`, so a graph restored from such a payload could no longer be saved.
- Now `decode` counts the markers with `Matrix::<u64>::count_value_eq` (a `GrB_select` with `GrB_VALUEEQ_UINT64`, stated as the hypothesis `SelectValueEqSpec`). It refuses a tensor entry that is not an in-bounds marker, and refuses a marker count that differs from the number of `me` pairs with ids.
- `decodeCheck_ok_iff` proves decode accepts exactly when every tensor entry is an in-bounds marker and the rows it builds are promotion complete. `decodeCheck_refuses_incomplete` is the direction that matters for safety: every payload that would break the invariant is refused.
- `decodeCheck_ok_edgeCount_no_panic` proves `edge_count` cannot panic on an accepted tensor.
- `decodeCheck_accepts_encoded` proves no valid payload is refused: decode accepts every tensor `encode` writes.
- main's three new tests are `orphan_row_refused`, `single_pair_ids_refused` and `marker_without_ids_refused`.

**3. Is a GraphBLAS matrix freed exactly once?**

**Yes, since #3090.** `Matrix` and `Iter` now share an `Arc<Handle>`, and `Handle::drop` frees the matrix. An `Arc` drop is one atomic decrement, and the thread that takes the count to zero runs the free, so deciding to free and decrementing happen in the same step. `Own.lean` models this:

- `arc_frees_once`: for any number of owners and any interleaving of clones and drops, the matrix is freed at most once, and it is freed exactly when the last owner is gone. So it is never freed twice and never leaked;
- `arc_all_drop_frees_once`: `n ≥ 1` owners dropping in any order free it exactly once;
- `arc_every_interleaving_frees_once`: the six two-owner interleavings, the one that leaked before included, all free exactly once;
- `iter_frees_iterator_before_matrix`: `Iter::drop` frees its `GxB_Iterator` before its `Arc<Handle>` can free the matrix (`detached_iter_frees_only_matrix` covers a detached iterator);
- `single_owner_frees_once`, `dup_independent_ownership`: a single owner frees once, and `dup` gives a copy that shares nothing.

History: before #3090, `Matrix::drop` checked `Arc::get_mut` (the count is 1) and then dropped the `Arc`, as two separate steps. `pre3090_race_leaks` shows an interleaving where both owners see a count of 2 and neither frees. `pre3090_never_double_free` and `pre3090_sequential_frees_once` show it was only a leak, and only under concurrency.

**4. Does a row-range scan visit each stored entry once?**

Yes. `drain_seek_eq_spec` proves `Iter::seek` followed by `next` until the end yields exactly the stored entries with `min_row ≤ row ≤ max_row`, in row-major order. `drain_no_duplication` adds that each entry appears once, and `drain_empty_matrix` covers an empty matrix.

**5. Are the index and dimension casts safe?**

- `me_dim_is_max` and `me_dim_constructs` prove the `me` block dimension is exactly the largest GraphBLAS accepts.
- `create_safe_iff` proves a node id resizes safely iff it is `≤ GrB_INDEX_MAX`.
- `create_2pow61_crashes` is the #2892 input (fixed by #2911 in `IdSpace::create`, see `proofs/id_space`).
- `len_cast_lossless` and `blob_copy_in_bounds` cover the `usize`↔`u64` casts and the decode length check.

**6. Does `count_in_rows` count what it says?**

`countInRows_spec` proves `Matrix::count_in_rows` returns the number of stored entries in the selected rows (#3022). `countInRows_pos` proves it is positive iff some selected row holds an entry, which is the question `first_node_with_relationships` asks.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `decodeBlob_never_dies`, `decodeU64_never_dies` | Since #2996 no status code from a corrupt blob kills the server | `Codes.lean` |
| `pre2996_decodeBlob_is_dos`, `pre2996_decodeU64_is_dos` | History: before #2996 a corrupt blob reached an `assert_eq!` and killed the server | `Codes.lean` |
| `checkLayers_merge_in_bounds` | `Tensor::decode` only merges delta entries inside the base (#2996) | `Decode.lean` |
| `encodeM_no_heap_addresses` | The encoded container header carries no heap address (#3097) | `Glue.lean` |
| `edgeCount_no_panic`, `edgeCount_value` | `edge_count` is total and correct while the tensor invariant holds | `Decode.lean` |
| `decodeCheck_ok_iff`, `decodeCheck_refuses_incomplete`, `decodeCheck_accepts_encoded` | `Tensor::decode` refuses exactly the payloads that break promotion completeness, and accepts every encoded tensor (#3033) | `Decode.lean` |
| `pre3033_orphan_row_panics`, `pre3033_more_me_rows_than_pattern_panics` | History: before #3033, a decoded tensor with an orphan `me` row made `edge_count` panic | `Decode.lean` |
| `arc_frees_once`, `arc_every_interleaving_frees_once` | Since #3090 the matrix is freed exactly once, under any number of owners and any schedule of drops | `Own.lean` |
| `pre3090_race_leaks` | History: before #3090, two concurrent drops could leak the matrix | `Own.lean` |
| `drain_seek_eq_spec`, `drain_no_duplication` | A row-range scan yields exactly the in-range entries, in order, once | `RowIter.lean` |
| `create_safe_iff`, `me_dim_is_max` | Which ids and dimensions GraphBLAS accepts | `Index.lean` |
| `countInRows_spec`, `countInRows_pos` | `count_in_rows` counts the stored entries in the selected rows | `Bulk.lean` |
| `fromSortedIndices_spec` | `from_sorted_indices` builds exactly the indicator vector of its input | `Bulk.lean` |
| `grown_rejects_shrink`, `resize_clamp_grownOk`, `iterExpand_eq_flatMap` | `grown` refuses a shrink, and `Tensor::resize`'s clamped dims (#3250) always pass that check; the tensor iterator expands each pair to its edge ids | `Misc.lean` |

### How `countInRows_spec` is proved

1. **Model the types.** `MatC` is a matrix as its dimensions plus its stored coordinates. `VecB` is a bool vector as its size plus its pattern. `VecU` is a `u64` vector as its size plus a value per index.
2. **State the C calls as a hypothesis.** `BulkSpec` gives each of the four calls its GraphBLAS user-guide meaning:
   - `GrB_Vector_new` makes an empty vector;
   - `GxB_Vector_build_Scalar` stores exactly the given sorted, in-range indices;
   - `GrB_mxv` with `PLUS_PAIR` and `DESC_T0` puts, at column `j`, the number of selected rows with an entry in column `j`;
   - `GrB_Vector_reduce_UINT64` with `PLUS` is the sum.

   It is a structure passed as an argument, not a Lean `axiom`.
3. **Model the function.** `countInRows` follows `Matrix::count_in_rows`: a fresh `u64` vector of `ncols`, one `mxv`, one `sum`.
4. **Prove the count.** Summing the per-column counts over all columns counts each stored entry in a selected row exactly once (`double_count`), as long as every stored column is `< ncols`.
5. **Hypotheses.** `BulkSpec` is the C library's documented behaviour, which Lean cannot check. The column bound `hc` is a GraphBLAS matrix invariant.

The bug theorems are smaller. `Disposition.run` models what each style of status check does in a release build. Each wrapper is assigned its style at its cited Rust line. `vector_deserialize_reachable_codes` lists the codes the C deserializer can return, taken from the GraphBLAS source. The theorems then follow by `decide`.

## Bugs found

The repros are in `graph/tests/lean_graphblas_wrappers.rs` (see [`../FINDINGS.md`](../FINDINGS.md); bug 1 is W3-gb-1). Status is as of main fb3d516d0.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `pre2996_decodeBlob_is_dos`, `pre2996_decodeU64_is_dos` | A corrupt multi-edge blob in `GRAPH.RESTORE`, an RDB or a full sync hit `assert_eq!` on `GxB_Vector_deserialize` and the server exited (confirmed live with a truncated COPY payload) | **Fixed by #2996** (`decodeBlob_never_dies`) |
| 2 | `pre3033_orphan_row_panics`, `pre3033_orphan_row_breaks_wf`, `pre3033_orphan_row_accepted` | `Tensor::decode` accepted an `me` row with no forward pair. `iter_edges` then yielded a phantom edge, `edge_count` panicked, and the restored graph could not be saved | **Fixed by #3033** (`decodeCheck_ok_iff`, `decodeCheck_accepts_encoded`) |
| 3 | — (Rust test only) | `Matrix::decode` leaks 5 heap blocks per matrix, and 8 on an error path: `copy_nonoverlapping` overwrites vectors `GxB_Container_new` allocated. This happens on every RDB load, RESTORE and full sync | **Open** |
| 4 | `pre3090_race_leaks` | Two `Matrix` clones dropped concurrently could both see a count of 2, so the GraphBLAS matrix was never freed (19 of 200,000 pairs leaked in the test) | **Fixed by #3090** (`arc_frees_once`, `arc_every_interleaving_frees_once`; main's test `concurrently_dropped_clones_free_the_matrix_once`) |
| 5 | — (Rust test only) | `vector::Iter` has no lifetime, so safe code can drop the vector and keep iterating freed memory | **Open, latent:** the only in-scope use is `Tensor::decode`; same class as #2898 |

Also noted, not confirmed:

- `Matrix` encode wrote the raw `GxB_Container`, live heap addresses included, into DUMP and COPY payloads. **Fixed by #3097**: `encodeM_no_heap_addresses` (history `pre3097_encode_leaks_heap_address`). `Matrix::decode` still copies the payload's `*_future` pointer arrays into the live container unchecked; it relies on GraphBLAS never reading them (`PtrBlind`).
- Every mutating call is only `debug_assert`ed, so a GraphBLAS failure there is silently ignored in release.

There are no CONDITIONAL rows.

## Files

| File | What's in it |
|---|---|
| `GrbInfo.lean` | The `GrB_Info` codes and the three dispositions (`assertPanic`, `debugAssert`, `propagate`) |
| `Index.lean` | `GrB_Index` casts, `GB_NMAX`, `ME_DIM`; the #2892 boundary |
| `Own.lean` | The drop protocol as a small-step machine: main's `Arc<Handle>` for any number of owners, and the pre-#3090 `Arc::get_mut` protocol over every two-owner interleaving |
| `RowIter.lean` | `matrix::Iter` `seek`/`next` over a row range |
| `Codes.lean` | Each deserialize wrapper's disposition; bug 1 and its #2996 fix |
| `Decode.lean` | `edge_count`'s checked arithmetic; the #2996 layer-size check; `count_value_eq` and the #3033 promotion-completeness check (bug 2 and its fix) |
| `Misc.lean` | `grown`/`resize` policy, `vector::Iter`, the multi-edge `Iter` expansion |
| `Glue.lean` | Operator/type selection, the descriptor table, accessors, encode round trip, the #3097 header scrub |
| `Serial.lean` | `EncodeState::from_u64` and the `Encode`/`Decode` trait contract |
| `Bulk.lean` | `from_sorted_indices`, `Vector::<u64>::sum`/`ptr`, `count_in_rows` (#3022) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The lakefile has no path dependencies.

## Coverage

`COVERAGE.tsv` has 1212 rows. Most are bindgen `extern` declarations in `mod.rs` and the LAGraph bindings.

| Bucket | Rows |
|---|---|
| PROVEN | 66 |
| AXIOMATISED | 1146 |

**AXIOMATISED** means the function is the C library, or a wrapper whose only content is one C call:

- 978 are GraphBLAS `extern` declarations (the spec is `GraphBLAS.h`);
- 117 are LAGraph and LAGraphX `extern` declarations, called from the algorithm procedures, outside this project;
- 47 are thin wrappers over one `GrB`/`GxB` call whose status is `debug_assert`ed;
- 4 are free, finalize and iterator-construction calls.

None of these has Rust logic left to prove.

## What the model does not cover

- GraphBLAS semantics (eWise, mxm, resize, the serialized byte format) are taken from the C documentation, not modelled or checked.
- The row-iterator model treats "skip empty rows that return `GrB_NO_VALUE`" as "advance to the next stored entry", which the GraphBLAS header documents.
- The `Own` model of main covers any number of owners; it takes `Arc`'s contract (one atomic decrement, the drop runs on the thread that took the count to zero) from std rather than proving it. The pre-#3090 history covers two owners exhaustively. `has_pending` and lock concurrency beyond `drop` are not modelled.
- Bugs 3 and 5 have Rust repros but no Lean theorem; the leak and the dangling iterator are outside what the model expresses.

## Build and check

```bash
cd proofs/graphblas_wrappers && lake build    # this project
bash proofs/lean_ci.sh graphblas_wrappers     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                       # repo-wide coverage report
```

**When you change `matrix.rs`, `vector.rs`, `serialization.rs`, or `tensor.rs`'s decode and iterator code:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match. For a new GraphBLAS call, record its disposition (assert, debug-assert or propagate) in `Codes.lean` or `GrbInfo.lean`.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
