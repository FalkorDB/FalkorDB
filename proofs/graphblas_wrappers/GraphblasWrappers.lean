/-
# GraphBLAS FFI wrappers (`graph/src/graph/graphblas/{matrix,vector,mod}.rs`, `tensor.rs` decode/iter)

**Re-targeted to main e860645bb** (from ca36bfa15): #3492 (a3a99bdbc) rewrites `Matrix::delta_lmxm`
(matrix.rs:1520) and adds `GrB_GT_UINT64` / `GrB_VALUEEQ_BOOL` / `GrB_Matrix_select_BOOL` imports (+3 lines before :1520,
+55 after, so every `matrix.rs` pin moved). It is now modelled (`DeltaLmxm.lean`) and PROVEN, no longer a
bare FFI row: `deltaLmxm_spec` (output stored iff `F*M` reaches it with no `F*DM` path, or `F*M` has strictly
more paths than `F*DM`, or `F*DP` reaches it; no assumption on the layers), `deltaLmxm_correct` (= `F*(M-DM+DP)`
when `dm ⊆ m`), `deltaLmxm_eff` / `deltaLmxm_tensor` (`dm ⊆ m` discharged from `VM.Inv` / `VMTensor.Inv`, so the
project now depends on versioned_matrix). Historical: `pre3492_drops_live_path` (the old pattern mask lost an output that
one deleted and one live base edge both reach), fixed by #3492 (a3a99bdbc); `needs_dm_sub_m` shows the counting
needs the invariant.

Lean 4 model of the Rust-side logic around the GraphBLAS C API. The C calls are
axiomatised (their contracts cited from `GraphBLAS.h` / the GraphBLAS source);
the wrapper logic is proved. Bucket per Rust fn: `COVERAGE.tsv`.
Re-targeted to main fb3d516d0 (#3090 e45be4965, #3026; earlier 73077bb4a: #3033, #3086;
1d15bf713: #2996, #3097, #3250). #3090 replaces `Matrix::drop`'s `Arc::get_mut`
free with `Arc<Handle>` + `Handle::drop` (matrix.rs:371; bug 4 fixed; every later
matrix.rs line +24..+31, see `COVERAGE.tsv`). #3026 touches only `bulk_insert.rs` (not here).
#3033 adds `Matrix::<u64>::count_value_eq` (then matrix.rs:1232; every later matrix.rs
line +24, earlier ones +1 for the import) and makes `Tensor::decode` enforce
promotion completeness (bug 2 fixed). #3086 touches only `indexer.rs` (not here).

## Lean ↔ Rust

| here | there |
| --- | --- |
| `GrbInfo.Info`, `Disposition` | `mod.rs:GrB_Info`; `assert_eq!` / `debug_assert_eq!` / `if info != SUCCESS {Err}` |
| `Index.matrixNewOk`, `GB_NMAX`, `ME_DIM` | `GB_Matrix_new.c:28` bound; `tensor.rs:144-169` constants |
| `Own.arcStep`/`liveRun`/`Inv` | `Handle::drop` matrix.rs:371 behind `Arc<Handle>` (`Matrix::clone` :403), `Iter::drop` :1744 then its `Arc<Handle>` field (#3090) |
| `Own.step`/`run`/`validTraces` | pre-#3090 `Matrix::drop`/`Iter::drop` (`Arc::get_mut` then Arc drop), history `pre3090_*` |
| `RowIter.drainL`/`drainSeek` | `Iter::seek` matrix.rs:1823, `Iter::next` :1856 |
| `Codes.decodeBlob`/`decodeU64`/`decodeBoolHardened` | `vector.rs:211` `decode_blob`, `:256` `Vector<u64>::decode`, `:350` `Vector<bool>::decode` (`pre2996_*`: before #2996) |
| `Decode.edgeCount`, `Wf`, `checkLayers` | `Tensor::edge_count` tensor.rs:1231; `Tensor::decode` :1553; layer-size check :1562-1573 (#2996) |
| `Decode.countValueEq`, `SelectValueEqSpec` | `Matrix::<u64>::count_value_eq` matrix.rs:1263 (`GrB_Matrix_select_UINT64`, `GrB_VALUEEQ_UINT64`, `nvals`) |
| `Decode.tensorLoop`/`decodeCheck`, `PromotionComplete`, `encFwd`/`encTs` | `Tensor::decode` marker count :1617, per-entry check :1632-1636, final count check :1665-1670 (#3033) |
| `Misc.grownOk`, `iterExpand`, `vecDrain` | `Matrix::grown` matrix.rs:735, `Tensor::resize` tensor.rs:823, `tensor::Iter::next` :1783, `vector::Iter::next` |
| `Glue.encodeM`/`scrub` | `Encode<19> for Matrix` matrix.rs:524, header pointers nulled :541-553 (#3097) |
| `DeltaLmxm.deltaLmxm`, `cnt`, `mxmCount`, `deltaLmxm_spec`/`_correct`/`_eff` | `Matrix::delta_lmxm` matrix.rs:1520-1655 (#3492: `F*dm` counted, `F*m` counted at the masked positions, `GT` then select, `restored` added back; the dp accumulation unmasked) |
| `Bulk.fromSortedIndices`, `VecU.sum`/`ptr`, `countInRows` | `Vector::<bool>::from_sorted_indices` vector.rs:149, `Vector::<u64>::sum` :494 / `ptr` :488, `Matrix::count_in_rows` matrix.rs:808 (#3022) |

## Theorems (all proved, no sorry)
#3492 (`DeltaLmxm.lean`): **`deltaLmxm_correct`**, `deltaLmxm_spec`, `cnt_split`, `deltaLmxm_eff`, `deltaLmxm_tensor`,
`delta_lmxm_keeps_live_path`, `delta_lmxm_drops_when_all_paths_deleted`, `needs_dm_sub_m` (history `pre3492_drops_live_path`);
`me_dim_is_max`, `me_dim_constructs`, `create_safe_iff`, `create_2pow61_crashes`,
`len_cast_lossless`, `blob_copy_in_bounds`; #3090: **`arc_frees_once`** (any n owners,
any interleaving of clones/drops: freed at most once, and iff the count reaches 0),
`arc_all_drop_frees_once`, `arc_every_interleaving_frees_once`,
`iter_frees_iterator_before_matrix`, `detached_iter_frees_only_matrix`, `iterDrop_state`,
`single_owner_frees_once`, `dup_independent_ownership` (history `pre3090_race_leaks`,
`pre3090_never_double_free`, `pre3090_sequential_frees_once`);
`drainL_eq_filter`, `drain_seek_eq_spec` (row iterator = in-range entries,
row-major, once), `drain_no_duplication`, `drain_empty_matrix`;
`decodeBlob_never_dies`, `decodeU64_never_dies`, `decodeBlob_safe_on_reachable`,
`decodeBlob_rejects_invalid_object`, `hardened_eq_blob` (#2996 fix; history
`pre2996_decodeBlob_is_dos`, `pre2996_decodeU64_is_dos`,
`pre2996_hardened_vs_blob_on_invalid_object`), `assertPanic_deserialize_dos`,
`decodeBool_safe`, `decodeBool_never_dies`; `checkLayers_ok_iff`,
`checkLayers_merge_in_bounds` (history `pre2996_accepts_out_of_bounds_delta`);
`encodeM_no_heap_addresses`, `scrub_scalars`, `decodeC_encodeM` (#3097; history
`pre3097_encode_leaks_heap_address`);
`edgeCount_no_panic`, `edgeCount_value`; #3033: `countValueEq_spec`,
`countValueEq_markers`, `mget_eq_some_iff`, `tensorLoop_ok_iff`,
**`decodeCheck_ok_iff`** (decode accepts iff every tensor entry is an in-bounds
marker pair and the `me` rows are promotion complete),
`decodeCheck_refuses_incomplete`, `decodeCheck_ok_complete`,
`decodeCheck_ok_edgeCount_no_panic`, **`decodeCheck_accepts_encoded`**, main's
tests `orphan_row_refused`, `single_pair_ids_refused`,
`marker_without_ids_refused` (history `pre3033_orphan_row_accepted`,
`pre3033_orphan_row_breaks_wf`, `pre3033_orphan_row_panics`,
`pre3033_more_me_rows_than_pattern_panics`);
`grown_rejects_shrink`, `resize_clamp_grownOk`, `resize_clamp_not_shrink` (#3250;
history `pre3250_resize_grow_iff_grownOk`), `vecDrain_eq`, `iterExpand_eq_flatMap`,
`iterExpand_length`, `iterExpand_all_single`; plus GrbInfo helper lemmas.
#3022 (`Bulk.lean`): `fromSortedIndices_spec`, `VecU.sum_spec`, **`countInRows_spec`**
(`GrB_mxv` over `PLUS_PAIR` with `DESC_T0`, then a `PLUS` reduce, counts exactly the stored
entries in the selected rows — the double-counting argument over columns), `countInRows_pos`,
`countInRows_fromSorted`. The four C calls are the hypothesis `BulkSpec`, not axioms.

## Confirmed bugs (repros: `graph/tests/lean_graphblas_wrappers.rs`)
1. **Crafted `GRAPH.RESTORE` kills the server — FIXED by #2996 on main
   1d15bf713** (both wrappers now return `Err`; `decodeBlob_never_dies`; the
   text below is the pre-#2996 state, `pre2996_*`). `Vector::<bool>::decode_blob`
   (then vector.rs:222) `assert_eq!`s `GxB_Vector_deserialize`, which returns
   `GrB_INVALID_OBJECT` on a corrupt blob; reached from `Tensor::decode`
   (tensor.rs:1622) per multi-edge pair. Tests `bug_decode_blob_panics_on_corrupt_blob`,
   `bug_tensor_decode_panics_on_corrupt_multi_edge_blob`. Live: genuine COPY
   payload from AOF, one id blob truncated to 8 bytes → release server exits,
   log "FalkorDB panic: … vector.rs:222:13 … GxB_Vector_deserialize failed:
   GrB_INVALID_OBJECT". Same shape: `Vector<u64>::decode` (vector.rs:256). Fix:
   return `Err` like `Vector<bool>::decode` does. (C not comparable: different payload format.)
2. **`Tensor::decode` accepted an `me` row with no forward pair — FIXED by
   #3033 on main 73077bb4a** (`decodeCheck_ok_iff`; text below is the pre-#3033
   state, `pre3033_*`); `edge_count`
   then panics ("multi exceeds the effective pattern … |m|=0 multi=1 |me|=1"),
   and `Tensor::encode` calls `edge_count`, so the restored graph can't be saved.
   `iter_edges` yields a phantom edge (7,9,3). Test
   `bug_tensor_decode_accepts_orphan_me_row_then_edge_count_panics`. Fix: in
   decode, require the pair's forward value to be a multi-edge marker.
3. **`Matrix::decode` leaks 5 heap blocks per matrix** (matrix.rs:467):
   `copy_nonoverlapping` overwrites the p/h/b/i/x vectors `GxB_Container_new`
   allocated. Every RDB load / RESTORE / full sync. 8 blocks per call on the `?`
   error paths (container + decoded components). Tests
   `bug_matrix_decode_leaks_container_components` (5.00/call),
   `bug_matrix_decode_leaks_on_error_path` (8.00/call). Fix: free the fresh
   components first (or copy only scalar fields); free on error.
4. **Concurrent drop of two `Matrix` clones leaked the GrB_Matrix — FIXED by
   #3090 on main e45be4965** (`arc_frees_once`, `arc_every_interleaving_frees_once`;
   text below is the pre-#3090 state, `pre3090_*`). `Matrix::drop` (then
   matrix.rs:390 at the time, same in `Iter::drop`): `Arc::get_mut` then Arc decrement was
   not atomic. `pre3090_race_leaks` witness; `pre3090_never_double_free` shows it
   was only a leak. Test `bug_matrix_concurrent_drop_leaks`: 19 of 200,000 pairs
   leaked. The fix is the suggested one: the free moved into `Handle::drop`
   inside the Arc; main's test `concurrently_dropped_clones_free_the_matrix_once`.
5. **`vector::Iter` has no lifetime** (`fn iter(&self) -> Iter<bool>`): safe
   code can drop the vector and keep iterating freed arrays. Ignored test
   `bug_vector_iter_outlives_vector` (after free it yielded 0,1,2,… instead of
   0,3,6,…). Latent (only in-scope use is Tensor::decode). Same class as #2898.

## Suspicions / notes
- (FIXED by #3097 on main: the header pointers are now written as null,
  `encodeM_no_heap_addresses`.) Matrix `Encode` wrote the raw 608-byte
  `GxB_Container` including live heap addresses (seen in a GRAPH.COPY payload: p/h/b/i/x = 0x104b92000 …): ASLR
  disclosure via DUMP/COPY payloads. `Decode` also copies `header_arena` from the
  payload; probe `probe_matrix_decode_forged_header_arena` shows no effect with
  arenas off (GraphBLAS 10.5 default).
- `debug_assert_eq!` on every mutating op (set/resize/wait/eWise…) means a
  GraphBLAS failure is silently ignored in release (e.g. out-of-range `set`).
- `read_signed()? as i32` for `handling` truncates (harmless: GraphBLAS validates).
- Also seen: #2892 (`assert_eq!` on `GrB_Matrix_new` for ids > 2^60), boundary in `Index`.

## Gaps
GraphBLAS semantics (eWise, mxm, resize, serialize bytes) are axioms/unmodelled;
the row-iterator model folds empty-row `NO_VALUE` skipping into "next stored
entry" per the header spec; `has_pending`/lock concurrency beyond drop is not
modelled; `Own` covers any n owners on main (`Arc`'s atomic-decrement contract
taken from std), the pre-#3090 history two owners exhaustively.
-/
import GraphblasWrappers.GrbInfo
import GraphblasWrappers.Index
import GraphblasWrappers.Own
import GraphblasWrappers.RowIter
import GraphblasWrappers.Codes
import GraphblasWrappers.Decode
import GraphblasWrappers.Misc
import GraphblasWrappers.Glue
import GraphblasWrappers.Serial
import GraphblasWrappers.Bulk
import GraphblasWrappers.DeltaLmxm
