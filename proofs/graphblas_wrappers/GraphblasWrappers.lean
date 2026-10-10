/-
# GraphBLAS FFI wrappers (`graph/src/graph/graphblas/{matrix,vector,mod}.rs`, `tensor.rs` decode/iter)

Lean 4 model of the Rust-side logic around the GraphBLAS C API. The C calls are
axiomatised (their contracts cited from `GraphBLAS.h` / the GraphBLAS source);
the wrapper logic is proved. Bucket per Rust fn: `COVERAGE.tsv`.

## Lean ↔ Rust

| here | there |
| --- | --- |
| `GrbInfo.Info`, `Disposition` | `mod.rs:GrB_Info`; `assert_eq!` / `debug_assert_eq!` / `if info != SUCCESS {Err}` |
| `Index.matrixNewOk`, `GB_NMAX`, `ME_DIM` | `GB_Matrix_new.c:28` bound; `tensor.rs:144-169` constants |
| `Own.step`/`run`/`validTraces` | `Matrix::drop` matrix.rs:388, `Iter::drop` :1618 (`Arc::get_mut` then Arc drop) |
| `RowIter.drainL`/`drainSeek` | `Iter::seek` matrix.rs:1699, `Iter::next` :1732 |
| `Codes.decodeBlob`/`decodeBoolHardened` | `vector.rs:211` `decode_blob`, `:253` `Vector<u64>::decode`, `:345` `Vector<bool>::decode` |
| `Decode.edgeCount`, `Wf` | `Tensor::edge_count` tensor.rs:1221; invariant unchecked by `Tensor::decode` :1543 |
| `Misc.grownOk`, `iterExpand`, `vecDrain` | `Matrix::grown` :701, `Tensor::resize` :809, `tensor::Iter::next` :1740, `vector::Iter::next` |
| `Bulk.fromSortedIndices`, `VecU.sum`/`ptr`, `countInRows` | `Vector::<bool>::from_sorted_indices` vector.rs:149, `Vector::<u64>::sum` :494 / `ptr` :488, `Matrix::count_in_rows` matrix.rs:774 (#3022) |

## Theorems (all proved, no sorry)
`me_dim_is_max`, `me_dim_constructs`, `create_safe_iff`, `create_2pow61_crashes`,
`len_cast_lossless`, `blob_copy_in_bounds`; `never_double_free`, `race_leaks`,
`sequential_frees_once`, `single_owner_frees_once`, `dup_independent_ownership`;
`drainL_eq_filter`, `drain_seek_eq_spec` (row iterator = in-range entries,
row-major, once), `drain_no_duplication`, `drain_empty_matrix`;
`decodeBlob_is_dos`, `decodeU64_is_dos`, `assertPanic_deserialize_dos`,
`decodeBool_safe`, `decodeBool_never_dies`, `hardened_vs_blob_on_invalid_object`;
`edgeCount_no_panic`, `edgeCount_value`, `orphan_row_breaks_wf`,
`orphan_row_panics`, `more_me_rows_than_pattern_panics`; `grown_rejects_shrink`,
`resize_grow_iff_grownOk`, `vecDrain_eq`, `iterExpand_eq_flatMap`,
`iterExpand_length`, `iterExpand_all_single`; plus GrbInfo helper lemmas.
#3022 (`Bulk.lean`): `fromSortedIndices_spec`, `VecU.sum_spec`, **`countInRows_spec`**
(`GrB_mxv` over `PLUS_PAIR` with `DESC_T0`, then a `PLUS` reduce, counts exactly the stored
entries in the selected rows — the double-counting argument over columns), `countInRows_pos`,
`countInRows_fromSorted`. The four C calls are the hypothesis `BulkSpec`, not axioms.

## Confirmed bugs (repros: `graph/tests/lean_graphblas_wrappers.rs`)
1. **Crafted `GRAPH.RESTORE` kills the server.** `Vector::<bool>::decode_blob`
   (vector.rs:222) `assert_eq!`s `GxB_Vector_deserialize`, which returns
   `GrB_INVALID_OBJECT` on a corrupt blob; reached from `Tensor::decode`
   (tensor.rs:1599) per multi-edge pair. Tests `bug_decode_blob_panics_on_corrupt_blob`,
   `bug_tensor_decode_panics_on_corrupt_multi_edge_blob`. Live: genuine COPY
   payload from AOF, one id blob truncated to 8 bytes → release server exits,
   log "FalkorDB panic: … vector.rs:222:13 … GxB_Vector_deserialize failed:
   GrB_INVALID_OBJECT". Same shape: `Vector<u64>::decode` (vector.rs:253). Fix:
   return `Err` like `Vector<bool>::decode` does. (C not comparable: different payload format.)
2. **`Tensor::decode` accepts an `me` row with no forward pair**; `edge_count`
   then panics ("multi exceeds the effective pattern … |m|=0 multi=1 |me|=1"),
   and `Tensor::encode` calls `edge_count`, so the restored graph can't be saved.
   `iter_edges` yields a phantom edge (7,9,3). Test
   `bug_tensor_decode_accepts_orphan_me_row_then_edge_count_panics`. Fix: in
   decode, require the pair's forward value to be a multi-edge marker.
3. **`Matrix::decode` leaks 5 heap blocks per matrix** (matrix.rs:453):
   `copy_nonoverlapping` overwrites the p/h/b/i/x vectors `GxB_Container_new`
   allocated. Every RDB load / RESTORE / full sync. 8 blocks per call on the `?`
   error paths (container + decoded components). Tests
   `bug_matrix_decode_leaks_container_components` (5.00/call),
   `bug_matrix_decode_leaks_on_error_path` (8.00/call). Fix: free the fresh
   components first (or copy only scalar fields); free on error.
4. **Concurrent drop of two `Matrix` clones leaks the GrB_Matrix**
   (matrix.rs:390, same in `Iter::drop` :1622): `Arc::get_mut` then Arc
   decrement is not atomic. `race_leaks` witness; `never_double_free` shows it's
   only a leak. Test `bug_matrix_concurrent_drop_leaks`: 19 of 200,000 pairs
   leaked. Fix: put the `GrB_Matrix_free` in the Drop of an inner owned type
   inside the Arc (or `Arc::into_inner`).
5. **`vector::Iter` has no lifetime** (`fn iter(&self) -> Iter<bool>`): safe
   code can drop the vector and keep iterating freed arrays. Ignored test
   `bug_vector_iter_outlives_vector` (after free it yielded 0,1,2,… instead of
   0,3,6,…). Latent (only in-scope use is Tensor::decode). Same class as #2898.

## Suspicions / notes
- Matrix `Encode` writes the raw 608-byte `GxB_Container` including live heap
  addresses (seen in a GRAPH.COPY payload: p/h/b/i/x = 0x104b92000 …): ASLR
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
modelled; `Own` covers two owners exhaustively (6 interleavings), not n.
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
