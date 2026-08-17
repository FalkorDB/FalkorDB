# Lean 4 proofs for `Tensor` (`graph/src/graph/graphblas/tensor.rs`)

A machine-checked model of FalkorDB's relationship tensor, with a correctness
theorem for every operation of `tensor.rs`.

* No `sorry`, no `admit`, no custom `axiom`. Every top-level theorem depends only
  on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`) —
  verify with `#print axioms`.
* ~4 100 lines, ~300 theorems; a clean rebuild takes ~20 s once the
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
| `flush` | `flush_effGet`, `inv_flush`, `edgesAt_flush`, `edgeCount_flush` — for an *arbitrary* fold decision, plus `edgesAt_flush_decision_irrelevant` | `Flush.lean` |
| `fold_latched`, `fold_oversized`, the `dup`→`flush` path | no separate model: they differ only in when they fire and which policy latches the decision, then run the same fold, so the `flush` theorems cover them | `Flush.lean` |
| the entry `self.flush()` of `set_all_from_slices` / `remove_all` | `edgesAt_setAll_after_flush`, `inv_setAll_after_flush`, `removeAll_after_flush_spec` (+ `writableBatch_flush`, `freshBatch_flush`) — a fold before the batch cannot change its result | `Flush.lean` |
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
| the fold *policy* (`should_fold`) | `foldPoint_optimal` — the square-root rule minimises the cost model it is derived from; `foldCost_foldPoint` (the minimum value); `sq_ge_iff_ge_sqrt` (the `u64` predicate `d² ≥ k·t` is the real test `d ≥ √(k·t)`) | `Cost.lean` |
| `iter_edges`, as output | `iterEdges_output_optimal` — every edge exactly once and nothing else, i.e. the least output any correct enumerator can produce | `Cost.lean` |

Two facts the proofs turned up that the code did not state — and, since they were
found, no longer contains:

* **`remove_all` had a dead branch.** In the `MULTI` case, "all ids removed at
  once; the pair is gone" is unreachable: a `MULTI` pair holds ≥ 2 ids, so
  erasing one always leaves a survivor (`removeOne_survivor`). The Rust now
  carries an `unreachable!()` there instead, which also flattened the demote path
  and dropped an `emptied.push` that could never fire.
* **`Iter`'s backward `unwrap_or(0)` was dead.** `mt` never holds a pair the
  forward matrix has lost, so `eff_get` there is always `some`
  (`iterBwd_eff_get_isSome`). That fallback was worse than merely dead: 0 is a
  *valid* edge id, so had the invariant broken, the iterator would have quietly
  emitted a fabricated edge instead of failing. It now says so.

The models keep both branches, because a Lean definition must be total; they are
simply never taken. That asymmetry is the point — the proof is what licensed
removing them from the code.

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
  (`set` / `remove` / `remove_mask` / `iter` / `nvals`). The three *forward* layers
  (`m`, `dp`, `dm`) are modelled explicitly, since `tensor.rs` manipulates them
  directly and that is where the delicate invariants live.

  **This boundary is now discharged**, not left on trust: the `VersionedMatrix`
  library in this same project (`VERSIONED_MATRIX.md`, `lake build
  VersionedMatrix`) proves `VersionedMatrix<bool>` against that exact interface —
  `eff_set`, `eff_remove`, `eff_removeMask`, `nvals_eq_card`. The two developments
  together cover the path from Cypher-visible behaviour down to the GraphBLAS
  calls with no hand-waved layer in between.
* **`multiCount` is a field here and a derivation there.** The model carries the
  number of `MULTI` pairs as a field, with `multi_count_eq` in `Inv` requiring it
  to agree with `multiPairs.card`. `tensor.rs` deleted the corresponding field in
  #2439 and derives the quantity from `me` at each `edge_count()`. The clause
  therefore holds by construction in the artifact, so the divergence can only make
  the code safer than the model — but the `edgeCount` theorems are stated about a
  field the artifact does not have. Tracked as item 3b-bis in
  `docs/papers/OPEN_WORK.md`.
* GraphBLAS pending work (`wait`, `wait_all`, `wait_base`, `is_synced`,
  `pending`) has no denotational content: every operation here behaves as if the
  layers were materialized, which is what `tensor.rs` guarantees by waiting on
  entry. `memory_usage` is likewise outside the model.

* **The fold policy is a parameter, not a formula.** When a delta folds is decided
  by `should_fold` / `should_fold_read` / `delta_dominates_base` in
  `versioned_matrix.rs` — a cost model whose constants (`WRITE_FOLD_K`,
  `READ_FOLD_K`, `MIN_FOLD_DELTA`) are *measured*, and which is evaluated on
  deliberately approximate counters (`Delta::count` overcounts a shadowing
  `insert`, and `erase` saturates rather than probe a shared layer). Encoding any
  of that here would freeze one tuning decision into the proofs and make them
  stale on the next measurement.

  So `flush` takes the decision as two `Bool`s and every theorem is proved for all
  four combinations. That is strictly stronger than modelling the policy: it says
  no choice of constants, and no drift in the counters that feed them, can change
  what the tensor denotes — `edgesAt_flush_decision_irrelevant` states exactly
  that.

  `Cost.lean` then proves the one thing about the policy that *is* a theorem:
  granting the cost model, `sqrt(2Ft/w)` is where that model is minimised, and the
  `u64` predicate the code evaluates is that test without the square root. Note
  what this does and does not settle. It does not say the implementation is fast:
  `F` and `w` are measured, and if the measurements are wrong the theorem still
  holds and the policy is still miscalibrated. It says the *form* of the rule
  follows from the model, so re-tuning the constants against new measurements
  cannot invalidate it. Everything else about performance —
  `fold_cost_bench.rs`, the benchmark tables — is measurement, and nothing in this
  development speaks to running time.
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
