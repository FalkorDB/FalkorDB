/-
# PR #3438 review — "perf: SAVE no longer re-walks attributes per key or rebuilds edge matrices"

Fixes #3433 / #3434. Branch `perf/rdb-save-fast`, one commit 2fabe1a8a on main
9212add39 (the proofs cite b6567cb97; the four files below are identical at both).
Every `file:line` here is the PR branch. Imports, not copies: pending_commit
(`Content.encode`, main's `encode_with_range`), graph_persist (`encRange`,
`encRoar`, `encPayload`, `load_encode`, `multi_load`, `chunkType`), versioned_matrix
(`TT`, `PairSt`, `effV`, `AllInv`, `BlkInv`, `hasMultiEdge`, `fwdIter`, `waitFwd`,
`encVal`), graphblas_wrappers (`GBWGlue.encodeM`, `UnloadLoad`, `PtrBlind`).

| Lean | Rust (PR 2fabe1a8a) |
| --- | --- |
| `RSet`, `RSet.rank`/`select`/`iterFrom` (`Roaring.lean`) | roaring 0.11.5 `RoaringTreemap::{iter, contains, rank, select}`, `Iter::advance_to`, as definitions on the owners' abstraction (the sorted deleted-id list of graph_persist, membership = pending_commit's `deleted`) |
| `nthLive`, `bsearch` (`NthLive.lean`) | `nth_live_id` attribute_store.rs:1483-1504 |
| `prEncode`, `walk` (`Slice.lean`) | `AttributeStore::encode_with_range` attribute_store.rs:1425-1474 |
| `encRoarPR` (`Layout.lean`) | `Encode for RoaringTreemap::encode_with_range` serialization.rs:180-197 |
| `encPayloadPR`, `encodeGraphPR` (`Layout.lean`) | `Graph::encode_payload` graph.rs:4942, `encode_graph` encoder/mod.rs:66 |
| `encPR`, `fastBase`, `fastCond` / `encMain`, `mainBase` (`TensorFast.lean`) | `Encode for Tensor` tensor.rs:1474-1570 / main 9212add39 :1474-1549 |
| `encodePR`, `Heap`, `step`, `prOps` / `mainOps` (`MatrixDup.lean`) | `Encode<19> for Matrix<T>` matrix.rs:521-575 / main :519-571 |

## Verdict per claim

1. **Slice encoding = main's — PROVEN.** `prEncode_eq_main`:
   for every deleted set, `max_id`, `offset` and `count > 0` the PR writes
   exactly main's entity sequence (ids, order, per-entity writes).
   `count = 0` differs (`count_zero_differs`, `count_zero_witness`: main writes
   one entity, the PR none) and is unreachable: `encoder_counts_pos` (every
   entity entry of `build_payloads`/`build_multi_key_payloads` has `count > 0`;
   `encode_payload` is the only caller). `nthLive_spec`: `nth_live_id` returns
   the `n`-th live id of `0..=max_id`, or `None` past the end; `nthLive_empty_deleted`,
   `nthLive_past_end`; `bsearch_spec` (loop invariant, terminates on `hi - lo`),
   `mid_bounds` (no `u64` overflow), `rank_le`/`liveThrough_no_overflow` (no `u128`
   underflow/overflow), so ids up to `u64::MAX` are fine.
2. **Slices concatenate to the full walk — PROVEN.**
   `chunk_concat`/`chunk_concat_full` (greedy `vkey_max` slices concatenate to one
   walk); `encodeGraphPR_eq`: on every layout the encoder builds, the PR's
   stream *is* main's, hence `load_encode_pr` and `multi_load_pr`
   (graph_persist's round trips, premises unchanged).
3. **Deleted-ids seek = main's skip — PROVEN.**
   `encRoarPR_eq` for every `count`/`offset` (incl. past the end).
4. **Tensor fast path — PROVEN (FFI hypothesis `MatrixRT`).** `fast_base_eq`:
   with `dp`/`dm` empty and no `me` id, the committed base has exactly the
   rebuilt matrix's pattern and values, and main's `multi` list is empty
   (promotion completeness from versioned_matrix's invariants);
   `encPR_eq_encMain`; `encode_one_state` (the check, the write and
   `edge_count` read one state); `decode_same`.
5. **`Matrix::encode` no longer mutates `self` — PROVEN (FFI: `UnloadLoad`,
   `PtrBlind`, `DupSpec`, and `step`'s per-call C contract).** (a) `decode_pr`:
   decode-equivalent, **not** byte-equal in general (`pr3438_bytes_can_differ`;
   live: a 300-entry matrix's `x` array is 4096 bytes on main, allocation slack
   included, and 2400 on the PR); `encodePR_bytes_eq` gives equality exactly when the
   copy unloads to the same container. (b) `pr_self_never_unloaded`;
   (#3434; counterexample removed; in 17ccc5412; live: `nvals = 0` mid-encode on
   main). (c) `pr_no_leak_no_double_free`, `pr_panic_after_dup`,
   `pr_writes_self_content`, `pr_no_heap_addresses` (#3097 kept).
   The tensor fast path *depends* on (b): it is the first caller to pass the
   shared committed `m` itself to `Matrix::encode`.
6. **Existing theorems.** pending_commit `Content.encode_spec` models main's
   `window … (max count 1)`: false for the PR at `count = 0` only; re-target its
   model to `take count` (`prEncode_spec`); `encode_decode_roundtrip` survives
   (`pr_encode_decode_roundtrip`). graph_persist `encRange` likewise; `slice_rt`,
   `kind_rt`, `load_encode`, `multi_load` keep their premises (`count > 0`
   already required). graphblas_wrappers `encodeM_spec` (`.2 = m`, self
   restored) becomes `pr_self_never_unloaded`; `decodeC_encodeM` holds for
   `dup m`. versioned_matrix `encVal_roundtrip` still covers the rebuild branch.
   No owner theorem breaks, and the rows for attribute_store.rs:1425 and
   serialization.rs:180 stay PROVEN: `rank`/`select`/`advance_to` are defined on
   the same abstraction the owners already use for `contains`/`iter`.

## Hypotheses

* No hypothesis about the roaring crate. **The roaring crate is trusted to
  implement the `RSet` abstraction** (ascending `iter`, `contains` = membership,
  `rank x = |{d ∈ s | d ≤ x}|`, `select n` = the `n`-th smallest, `advance_to a`
  positions the sorted iterator at the first element `≥ a`), at the same level
  as the owners' `contains`/`iter` (graph_persist `G.delNodes`/`encRoar`,
  pending_commit `deleted`). Checked against the real crate by the randomized
  scratch test `lean3438_roaring_contract` (3000 sets over four 2^32 high keys
  up to `u64::MAX`; all pass).
* `MatrixRT`, `UnloadLoad`, `PtrBlind`, `DupSpec`, `step`: GraphBLAS C API (FFI).
* versioned_matrix `AllInv`/`BlkInv`: the owner's tensor invariants, maintained by
  its mutator theorems (as for `row_degree`).

## Live / Rust checks (scratch copy of the PR, not committed)

`lean3438_encode_with_range_equals_main` (300 random stores, every
`offset`/`count`: equal for `count > 0`; 7946 `count = 0` cases where main writes
one entity), `lean3438_nth_live_id_u64_range`, `lean3438_roaring_contract`,
`lean3438_deleted_ids_seek_equals_skip`, `lean3438_tensor_fast_path_vs_main_rebuild`
(16 fast-path tensors decode as main's; 15 byte-identical),
`lean3438_matrix_encode_dup_vs_main`: all pass. Redis (PR and main modules,
`VKEY_MAX_ENTITY_COUNT 3`, 29 virtual keys, deleted nodes/edges): graph after
`SAVE` + `DEBUG RELOAD` = before, and PR = main.

## Cite shifts at re-target (main → PR)

attribute_store.rs `decode` 1466→1507, `decode_with_count` 1476→1517 (pending_commit);
new `nth_live_id` 1483. serialization.rs `decode` 195→201, `decode_with_count`
216→222 (graph_persist). matrix.rs: every fn after `encode` +3, 584→587 … 1801→1804
(55 graphblas_wrappers rows). tensor.rs: every fn after `encode` +15:
`decode` 1553→1568, `EdgeIds::next` 1699→1714, `size_hint` 1706→1721, `Iter::new`
1734→1749, `seek` 1766→1781, `Iter::next` 1783→1798 (versioned_matrix 6 rows;
graphblas_wrappers 3 rows).
-/
import Pr3438Review.Roaring
import Pr3438Review.NthLive
import Pr3438Review.Slice
import Pr3438Review.Layout
import Pr3438Review.TensorFast
import Pr3438Review.MatrixDup
