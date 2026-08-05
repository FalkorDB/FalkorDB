# Lean 4 proofs for `Tensor` (`graph/src/graph/graphblas/tensor.rs`)

A machine-checked model of FalkorDB's relationship tensor, with a correctness
theorem for every operation of `tensor.rs`.

* No `sorry`, no `admit`, no custom `axiom`. Every top-level theorem depends only
  on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`) —
  verify with `#print axioms`.
* ~3 800 lines, ~270 theorems; a clean rebuild of all 12 files takes ~10 s once the
  mathlib cache is in place.

## Build

```bash
cd proofs/tensor
lake exe cache get      # first time only: download mathlib oleans
lake build              # green = every proof checks
```

Toolchain: `leanprover/lean4:v4.32.0` + mathlib `v4.32.0` (pinned in
`lean-toolchain` / `lakefile.toml`). `elan` fetches the toolchain automatically.

## What is proved

`Tensor` denotes a multigraph: `edgesAt t p` is the finite set of edge ids stored
at pair `p`. `Inv t` is the "Delta-Layer Invariants" section of the Rust module
docs, stated formally (11 clauses). Every operation gets two kinds of theorem:
it **preserves `Inv`**, and it **acts on `edgesAt` the way its doc comment says**.

| `tensor.rs` | theorems | file |
| --- | --- | --- |
| `compound_key` | `keyBits_eq_key` (bitwise `(src<<32)\|dst` = arithmetic model), `key_lt` (no `u64` truncation), `key_inj`, `keyHi`/`keyLo` (round-trip) | `Key.lean` |
| `new` | `inv_new`, `edgesAt_new`, `edgeCount_new` | `Reads.lean` |
| `eff_get` | `mem_effDom_iff_isSome`, `effGet_of_dp`, `effGet_of_m` | `Model.lean` |
| `get` / `EdgeIds` | `getIds_eq_sort`, `mem_getIds`, `getIds_nodup`, `getIds_pairwise_lt` (ascending), `getIds_single` (allocation-free path) | `Reads.lean` |
| `set_all_from_slices` | `inv_addEdge`, `edgesAt_addEdge_self`, `edgesAt_addEdge_ne` for all three branches (already-multi / promote / first edge) and all three write-phase shapes (`dp` set / un-mask / cancel-to-clean); `inv_setAll`, `edgesAt_setAll` for a whole batch | `Add.lean` |
| …its `batch` map | `retro_promote_agrees` — retroactive promotion of a repeated pair lands in the same state as sequential promotion, *including* the "cancel re-promotion back to clean" case | `Batch.lean` |
| `remove_all` (fast path) | `inv_removeFast`, `edgesAt_removeFast_mem/_not_mem` | `Remove.lean` |
| `remove_all` (slow path) | `removeOne_spec` (invariants + `edgesAt' p = (edgesAt p).erase id` + other pairs untouched), `removeOne_emptied` (a pair is reported iff that removal took its last edge), `removeSlow_spec`, `removeAll_spec` | `Remove.lean` |
| `resize` | `inv_resize` (growth keeps every entry in range), `edgesAt_resize` | `Reads.lean` |
| `flush` | `flush_effGet`, `inv_flush`, `edgesAt_flush`, `edgeCount_flush` | `Flush.lean` |
| `extract` | `extract_eq_effDom`, `mem_extract_iff` | `Reads.lean` |
| `rebuild_backward` | `inv_rebuildBackward` — *establishes* the `mt` invariant from `InvCore` | `Flush.lean` |
| `dup` / `Clone` | `inv_dup`, `edgesAt_dup` | `Reads.lean` |
| `structural_iter` | `mem_structuralIter_iff` | `Reads.lean` |
| `edge_count` | `edgeCount_eq_sum` (= `∑ pairs, #edges`), `edgeCount_no_underflow` (the `u64` subtraction chain never wraps) | `Count.lean` |
| `iter_edges` | `mem_iterEdges` + `nodup_iterEdges` (every edge exactly once; the `me` half inverts `compound_key`) | `Iter.lean` |
| `iter(.., false)` | `mem_iterFwd`, `nodup_iterFwd` | `Iter.lean` |
| `iter(.., true)` | `mem_iterBwd` (range selects by *destination*), `nodup_iterBwd`, `iterBwd_eff_get_isSome` | `Iter.lean` |
| `has_multi_edge` | `hasMultiEdge_iff` (`me` non-empty ⟺ some pair has ≥ 2 edges) | `Reads.lean` |
| `encode` / `decode` | `edgesAt_decode_encode`, `invCore_decode_encode`, `inv_decode_encode`, `edgeCount_roundTrip`, plus `msb_or` / `msb_and_eq_zero` / `msb_and_ne_zero` for the MSB tag | `Codec.lean` |
| per-pair state diagram | 9 arrows of the documented `A`–`J` diagram, on the raw layers: `trans_A_add`, `trans_B_add`, `trans_D_add`, `trans_G_add_cancel`, `trans_G_add_other`, `trans_I_add_cancel`, `trans_D_del`, `trans_E_del`, `trans_F_del_cancel` | `States.lean` |

Two facts the proofs turned up that the code does not state:

* **`remove_all` has a dead branch.** In the `MULTI` case, "all ids removed at
  once; the pair is gone" is unreachable: a `MULTI` pair holds ≥ 2 ids, so
  erasing one always leaves a survivor (`removeOne_survivor`).
* **`Iter`'s backward `unwrap_or(0)` is dead.** `mt` never holds a pair the
  forward matrix has lost, so `eff_get` there is always `some`
  (`iterBwd_eff_get_isSome`).

## Preconditions the theorems assume

These are requirements the *callers* satisfy; they are stated explicitly rather
than assumed silently.

* `InBounds t p` — each node id fits a `u32` (what `compound_key` asserts) and
  the coordinate is inside the matrix (the caller `resize`s first).
* `ValidId id` — an edge id is a GraphBLAS index (`≤ GrB_INDEX_MAX`), hence never
  the `MULTI_EDGE` sentinel.
* `FreshBatch` — edge ids are freshly allocated, so a batch never inserts an id
  that is already stored, nor the same id twice.
* `remove_all`'s `hex` — the request names edges that exist. This is what makes
  the fast path correct: it deletes whole pairs without checking ids.

## Modelling boundary

* `mt` and `me` are `VersionedMatrix` values in Rust; here they are the sets of
  their effective entries, which is the whole interface `tensor.rs` uses
  (`set` / `remove` / `remove_mask` / `iter` / `nvals`). Their own three-layer
  algebra belongs to a proof of `versioned_matrix.rs`. The three *forward* layers
  (`m`, `dp`, `dm`) are modelled explicitly, since `tensor.rs` manipulates them
  directly and that is where the delicate invariants live.
* GraphBLAS pending work (`wait`, `wait_all`, `is_synced`, `pending`) has no
  denotational content: every operation here behaves as if `wait_fwd()` had run,
  which is what `tensor.rs` guarantees by calling it on entry. `memory_usage` is
  likewise outside the model.
* Indices are `Nat`, not `u64`/`u32`. Where width matters the bound is proved
  rather than assumed: `key_lt` (the compound key fits a `u64`),
  `edgeCount_no_underflow` (the unsigned subtraction chain), `msb_*` (the
  serialisation tag).
* The read phase of `set_all_from_slices` is modelled per pair, not as the
  `FxHashMap` + parallel-`Vec` loop. `Add.lean` proves the sequential semantics
  of a whole batch and `Batch.lean` proves the one place the batched
  implementation differs (retroactive promotion of a repeated pair) agrees with
  it; the list-level plumbing between the two is not mechanised.

## Layout

```
Tensor/Model.lean   state, denotation, invariants, transfer lemmas
Tensor/Key.lean     compound_key
Tensor/Ops.lean     every operation, transcribed from the Rust
Tensor/Reads.lean   new, dup, get, has_multi_edge, extract, structural_iter, resize
Tensor/Add.lean     set_all_from_slices
Tensor/Batch.lean   the batch map's retroactive promotion
Tensor/Remove.lean  remove_all (both paths)
Tensor/Count.lean   edge_count
Tensor/Flush.lean   flush, rebuild_backward
Tensor/Iter.lean    iter_edges, Iter (forward and transposed)
Tensor/Codec.lean   encode / decode
Tensor/States.lean  the documented per-pair state diagram
```

`Ops.lean` is the place to look when checking the model against the Rust: each
definition follows its function statement by statement, and its header table maps
each GraphBLAS call to its model (`GrB_Matrix_setElement` → `Layer.set`,
`dm<mask> = mask ∩ m` → `(dm \ mask) ∪ (mask ∩ m.dom)`, …).

## Keeping this honest

The proofs are about `Ops.lean`, so they are only as good as its correspondence
to `tensor.rs`. If you change the Rust, change `Ops.lean` and re-run `lake build`
— a semantics change will break a proof, which is the point.
