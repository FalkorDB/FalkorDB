/-
# GraphBLAS FFI wrappers (`graph/src/graph/graphblas/{matrix,vector,mod}.rs`, `tensor.rs` decode/iter)

`EncodeDup` models `Encode<19> for Matrix<T>` (matrix.rs) unloading a private `dup()` instead of `self`.
`decode_pr` (decode-equivalent under `UnloadLoad` + `PtrBlind` + `DupSpec`; NOT byte-equal:
`pr3438_bytes_can_differ`, live 4096 vs 2400 byte `x` array for a 300-entry matrix), `encodePR_bytes_eq`,
`pr_self_never_unloaded`, `pr_no_leak_no_double_free`, `pr_writes_self_content`, `pr_panic_after_dup`,
`pr_no_heap_addresses`. `GBWGlue.encodeM_spec` (`.2 = m`, self restored) is the earlier model that
unloads `self`. FFI: `UnloadLoad`, `PtrBlind`, `DupSpec`, and `step`'s per-call C contract.

**Re-targeted to main e860645bb** (from ca36bfa15): #3492 (a3a99bdbc) rewrites `Matrix::delta_lmxm`
(then matrix.rs) and adds `GrB_GT_UINT64` / `GrB_VALUEEQ_BOOL` / `GrB_Matrix_select_BOOL` imports (+3 lines before :1520,
+55 after, so every `matrix.rs` pin moved). It is now modelled (`DeltaLmxm.lean`) and PROVEN, no longer a
bare FFI row: `deltaLmxm_spec` (output stored iff `F*M` reaches it with no `F*DM` path, or `F*M` has strictly
more paths than `F*DM`, or `F*DP` reaches it; no assumption on the layers), `deltaLmxm_correct` (= `F*(M-DM+DP)`
when `dm ⊆ m`), `deltaLmxm_eff` / `deltaLmxm_tensor` (`dm ⊆ m` discharged from `VM.Inv` / `VMTensor.Inv`, so the
project now depends on versioned_matrix). `needs_dm_sub_m` shows the counting
needs the invariant.

**Re-sync to 256ab4d3f (from 43936bab9).** The GraphBLAS v10.5.3 bump regenerated `mod.rs`'s bindgen block
(`GraphBLAS.rs`'s extern decls), so 978 `mod.rs` rows moved by +9 and every `matrix.rs` pin moved (+3 to +55);
all rows were re-cited by function name (`fnchanged.py`, then a script rewrote the MOVED lines; the FFI rows
stay AXIOMATISED, the C calls are the contract). New or changed since: `Matrix::remove_all`/`select` use
`GrB_DESC_RSCT0` (the mask is structural, #3471), `element_wise_*`/`set_pattern` write their operand/mask
selection with `Matrix::inner` paths (#3787): thin wrappers, nothing modelled changes. `Matrix::encode`'s
private-`dup()` unload (#3438) is `EncodeDup`; the `EncodeState` numbers (`CchIndices = 10`, `Final = 11`,
#3793) are `Serial.fromU64_cch`. The `#[cfg(test)] impl Matrix<f64>` (#3320) is test-only. The version
history of the header tables is preserved below; the live pins are the `matrix.rs`/`tensor.rs` lines in the
tables, re-set by function name at 256ab4d3f.

Lean 4 model of the Rust-side logic around the GraphBLAS C API. The C calls are
axiomatised (their contracts cited from `GraphBLAS.h` / the GraphBLAS source);
the wrapper logic is proved. Bucket per Rust fn: `COVERAGE.tsv`.
Earlier re-targets: #3090 replaced `Matrix::drop`'s `Arc::get_mut` free with `Arc<Handle>` +
`Handle::drop` (matrix.rs); #3033 added `Matrix::<u64>::count_value_eq` and made `Tensor::decode`
enforce promotion completeness; #3026 touches only `bulk_insert.rs` and #3086 only `indexer.rs`
(not here).

## Lean ↔ Rust

| here | there |
| --- | --- |
| `GrbInfo.Info`, `Disposition` | `mod.rs:GrB_Info`; `assert_eq!` / `debug_assert_eq!` / `if info != SUCCESS {Err}` |
| `Index.matrixNewOk`, `GB_NMAX`, `ME_DIM` | `GB_Matrix_new.c:28` bound; `tensor.rs` constants |
| `Own.arcStep`/`liveRun`/`Inv` | `Handle::drop` matrix.rs behind `Arc<Handle>` (`Matrix::clone` :407), `Iter::drop` :1886 then its `Arc<Handle>` field |
| `Own.step`/`run`/`validTraces` | the earlier `Matrix::drop`/`Iter::drop` (`Arc::get_mut` then Arc drop), kept as the historical protocol |
| `RowIter.drainL`/`drainSeek` | `Iter::seek` matrix.rs, `Iter::next` :1998 |
| `Codes.decodeBlob`/`decodeU64`/`decodeBoolHardened` | `vector.rs` `decode_blob` `Vector<u64>::decode` `Vector<bool>::decode` |
| `Decode.edgeCount`, `Wf`, `checkLayers` | `Tensor::edge_count` tensor.rs; `Tensor::decode` :1566; layer-size check :1577-1587 |
| `Decode.countValueEq`, `SelectValueEqSpec` | `Matrix::<u64>::count_value_eq` matrix.rs (`GrB_Matrix_select_UINT64`, `GrB_VALUEEQ_UINT64`, `nvals`) |
| `Decode.tensorLoop`/`decodeCheck`, `PromotionComplete`, `encFwd`/`encTs` | `Tensor::decode` marker count :1630, per-entry check :1608-1616, final count check :1676-1680 |
| `Misc.grownOk`, `iterExpand`, `iterEdges`, `VIter` | `Matrix::grown` matrix.rs, `Tensor::resize` tensor.rs, `tensor::Iter::next` :1794, `Tensor::iter_edges` :1292, `vector::Iter::new`/`next` vector.rs (cursor state machine; the C iterator contract is the model `cExhausted`) |
| `Glue.encodeM`/`scrub` | `Encode<19> for Matrix` matrix.rs, header pointers nulled :554-565 |
| `DeltaLmxm.deltaLmxm`, `cnt`, `mxmCount`, `deltaLmxm_spec`/`_correct`/`_eff` | `Matrix::delta_lmxm` matrix.rs (`F*dm` counted, `F*m` counted at the masked positions, `GT` then select, `restored` added back; the dp accumulation unmasked) |
| `Bulk.fromSortedIndices`, `VecU.sum`/`ptr`, `countInRows` | `Vector::<bool>::from_sorted_indices` vector.rs, `Vector::<u64>::sum` :498 / `ptr` :492, `Matrix::count_in_rows` matrix.rs |

## Theorems (all proved, no sorry)
`DeltaLmxm.lean`: **`deltaLmxm_correct`**, `deltaLmxm_spec`, `cnt_split`, `deltaLmxm_eff`, `deltaLmxm_tensor`,
`delta_lmxm_keeps_live_path`, `delta_lmxm_drops_when_all_paths_deleted`, `needs_dm_sub_m`;
`me_dim_is_max`, `me_dim_constructs`, `create_safe_iff`, `create_2pow61_crashes`,
`len_cast_lossless`, `blob_copy_in_bounds`; **`arc_frees_once`** (any n owners,
any interleaving of clones/drops: freed at most once, and iff the count reaches 0),
`arc_all_drop_frees_once`, `arc_every_interleaving_frees_once`,
`iter_frees_iterator_before_matrix`, `detached_iter_frees_only_matrix`, `iterDrop_state`,
`single_owner_frees_once`, `dup_independent_ownership`, `dup_copy_drop_leaves_source`;
`drainL_eq_filter`, `drain_seek_eq_spec` (row iterator = in-range entries,
row-major, once), `drain_no_duplication`, `drain_empty_matrix`;
`decodeBlob_never_dies`, `decodeU64_never_dies`, `decodeBlob_safe_on_reachable`,
`decodeBlob_rejects_invalid_object`, `hardened_eq_blob`, `assertPanic_deserialize_dos`,
`decodeBool_safe`, `decodeBool_never_dies`; `checkLayers_ok_iff`,
`checkLayers_merge_in_bounds`, `checkLayers_merge_in_bounds_dm` (the size check puts both delta layers inside the base);
`encodeM_no_heap_addresses`, `scrub_scalars`, `decodeC_encodeM`;
`edgeCount_no_panic`, `edgeCount_value` (checked `u64` additions, `Wf` bounds them); `countValueEq_spec`,
`countValueEq_markers`, `mget_eq_some_iff`, `tensorLoop_ok_iff`,
**`decodeCheck_ok_iff`** (decode accepts iff every tensor entry is an in-bounds
marker pair and the `me` rows are promotion complete),
`decodeCheck_refuses_incomplete`, `decodeCheck_ok_complete`,
`decodeCheck_ok_edgeCount_no_panic`, **`decodeCheck_accepts_encoded`**, main's
tests `orphan_row_refused`, `single_pair_ids_refused`,
`marker_without_ids_refused`;
`grown_rejects_shrink`, `resize_clamp_grownOk`, `resize_clamp_not_shrink`, `VIter.new_inv`, `VIter.next_inv`, `VIter.next_depleted`, **`VIter.drain_new`**
(a fresh vector iterator drains to exactly the stored entries, once each, ascending), `iterExpand_eq_flatMap`,
`iterExpand_length`, `iterExpand_all_single`, `iterEdges_count`, `iterEdges_length`, `iterEdges_all_single`; plus GrbInfo helper lemmas.
`Bulk.lean`: `fromSortedIndices_spec`, `VecU.sum_spec`, **`countInRows_spec`**
(`GrB_mxv` over `PLUS_PAIR` with `DESC_T0`, then a `PLUS` reduce, counts exactly the stored
entries in the selected rows — the double-counting argument over columns), `countInRows_pos`,
`countInRows_fromSorted`. The four C calls are the hypothesis `BulkSpec`, not axioms.

## Confirmed bugs (the repro tests named below are not in this tree)
3. **`Matrix::decode` leaks 5 heap blocks per matrix** (matrix.rs):
   `copy_nonoverlapping` overwrites the p/h/b/i/x vectors `GxB_Container_new`
   allocated. Every RDB load / RESTORE / full sync. 8 blocks per call on the `?`
   error paths (container + decoded components). Tests
   `bug_matrix_decode_leaks_container_components` (5.00/call),
   `bug_matrix_decode_leaks_on_error_path` (8.00/call). Fix: free the fresh
   components first (or copy only scalar fields); free on error.
5. **`vector::Iter` has no lifetime** (`fn iter(&self) -> Iter<bool>`): safe
   code can drop the vector and keep iterating freed arrays. Ignored test
   `bug_vector_iter_outlives_vector` (after free it yielded 0,1,2,… instead of
   0,3,6,…). Latent (only in-scope use is Tensor::decode). Same class as #2898.

## Suspicions / notes
- `Matrix::decode` copies the payload's `*_future` pointer arrays into the live container unchecked; it relies on
  GraphBLAS never reading them (`PtrBlind`).
- `debug_assert_eq!` on every mutating op (set/resize/wait/eWise…) means a
  GraphBLAS failure is silently ignored in release (e.g. out-of-range `set`).
- `read_signed()? as i32` for `handling` truncates (harmless: GraphBLAS validates).

## Gaps
GraphBLAS semantics (eWise, mxm, resize, serialize bytes) are axioms/unmodelled;
the row-iterator model folds empty-row `NO_VALUE` skipping into "next stored
entry" per the header spec; `has_pending`/lock concurrency beyond drop is not
modelled; `Own` covers any n owners on main (`Arc`'s atomic-decrement contract
taken from std), the earlier two-owner protocol exhaustively.
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
import GraphblasWrappers.CountLink
import GraphblasWrappers.EncodeDup
