# Lean 4 proofs for `VersionedMatrix<bool>` (`graph/src/graph/graphblas/versioned_matrix.rs`)

A machine-checked model of the copy-on-write MVCC matrix that `Tensor` — and every
label, adjacency and relationship-type matrix in `Graph` — is built from. Lives in
the same lake project as the `Tensor` proofs (`lean_lib VersionedMatrix`); build
with `lake build VersionedMatrix`, or `lake build` for both.

* No `sorry`, no `admit`, no custom `axiom`, no `native_decide`. Every top-level
  theorem rests only on Lean's three standard axioms — verify with
  `#print axioms`.
* ~940 lines, ~73 theorems; builds in ~4 s once the mathlib cache is in place.

## Why this exists

The `Tensor` development models its `mt` and `me` fields — both
`VersionedMatrix<bool>` in Rust — as plain `Finset`s of effective entries, and
calls that its one abstraction boundary. This development discharges that
boundary: `eff_set`, `eff_remove`, `eff_removeMask` and `nvals_eq_card` are
exactly the interface the tensor assumes there, so the two together leave no
hand-waved layer between the Cypher-visible behaviour and the GraphBLAS calls.

## Scope: `bool` only

`VersionedMatrix<T>` is generic, but every operation lives in
`impl VersionedMatrix<bool>`. The `u64` instantiation exists only as the
`Delta<u64>` layers `Tensor` owns and drives itself, which the `Tensor` proofs
cover. So a `bool` versioned matrix is a pure *pattern*: three `Finset`s, no
values, and the stricter invariant `dp ∩ m = ∅` (no shadowing).

## What is proved

`eff v = (m ∖ dm) ∪ dp` is the denotation. `Inv` is the module docs' invariant
section: `dp ∩ m = ∅`, `dm ⊆ m`, and every stored coordinate in bounds. Every
operation both **preserves `Inv`** and **acts on `eff`** as its doc comment says.

| `versioned_matrix.rs` | theorems | file |
| --- | --- | --- |
| `new`, `from_matrix`, `dup`/`Clone` | `inv_new`, `eff_new`, `inv_fromMatrix`, `eff_fromMatrix`, `inv_dup`, `eff_dup` | `Write.lean` |
| `get` | `get_isSome_iff_mem_eff` — answers `Some` exactly on `eff`, while probing only the base and *one* delta | `Write.lean` |
| `extract` | `mem_extract` | `Write.lean` |
| `set` | `eff_set` (needs no invariant), `inv_set` | `Write.lean` |
| `remove` | `eff_remove`, `inv_remove` | `Write.lean` |
| `remove_mask` | `eff_removeMask`, `inv_removeMask` | `Write.lean` |
| `remove_mask` vs. a `remove` loop | `eff_removeMask_eq_foldl_remove` — the substitution `Graph::delete_nodes` makes | `Write.lean` |
| `set_all` / `set_all_new` | `setAllFast_eq_setAllSlow`, `setAllNew_eq_setAll`, `eff_setAll`, `inv_setAll`, `eff_setAllNew`, `inv_setAllNew` | `Write.lean` |
| `nvals` | `nvals_eq_card`, `nvals_no_underflow`, `nvals_add_dm`, plus `nvals_set`/`_remove`/`_removeMask` | `Count.lean` |
| `flush` | `eff_flush`, `inv_flush`, `nvals_flush` for an *arbitrary* fold decision, plus `eff_flush_decision_irrelevant` | `Fold.lean` |
| `fold_latched`, `fold_oversized`, `dup`→`flush` | no separate model: they differ only in when they fire and which policy latches the decision, then run the same fold | `Fold.lean` |
| the entry `self.flush()` of every mutator | `eff_set_after_flush`, `eff_remove_after_flush`, `eff_removeMask_after_flush`, `eff_setAll_after_flush` | `Fold.lean` |
| `resize` (growth) | `eff_resize`, `inv_resize`, `resize_of_deltas_empty` | `Fold.lean` |
| `transpose` | `eff_transpose`, `inv_transpose` | `Fold.lean` |

### The three results worth reading

**`setAllFast_eq_setAllSlow`.** `set_all_inner` checks `dm` emptiness once and then
runs either a batched loop or per-entry `set`. Nothing in the Rust forces those to
agree — the fast path skips a committed pair outright where `set` would erase it
from `dm` — and this proves they do, because erasing from an empty `dm` is a
no-op. The Rust asserts this in a comment; here it is checked.

**`setAllNew_eq_setAll`.** `set_all_new` drops the per-entry base probe on the
caller's word that no entry is committed. That word is the `debug_assert!` in
`set_all_inner`, and it is the *only* thing between the unchecked path and a
broken `dp ∩ m = ∅` — so it is stated as an explicit `FreshEntries` hypothesis and
the agreement proved under it.

**`eff_removeMask_eq_foldl_remove`.** `Graph::delete_nodes` used to build a
diagonal mask for `remove_mask`; it now loops `remove` per deleted entity, because
`remove_mask`'s `element_wise_multiply` takes `m` as an operand and so costs
`O(base)` however small the delete. The Rust argues the swap is safe in a comment;
this is the theorem.

## The fold policy is a parameter, not a formula

When a delta folds is decided by `should_fold` / `should_fold_read` /
`delta_dominates_base` — a cost model whose constants are *measured*, read off
counters that are deliberately approximate (`Delta::count` overcounts a shadowing
`insert`; `erase` saturates rather than probe a shared layer). Encoding that here
would freeze one tuning decision into the proofs.

So `flush` takes its decision as two `Bool`s and every theorem holds for all four
combinations. `eff_flush_decision_irrelevant` is the payoff: no choice of
constants, and no drift in the counters feeding them, can change what the matrix
denotes. The policy's own claim is about throughput and memory — a measurement
question (`fold_cost_bench.rs`), not a theorem.

## Preconditions the theorems assume

* `InBounds v p` — the coordinate is inside the matrix. The Rust callers `resize`
  first, which is what makes this a precondition rather than a check.
* `FreshEntries v l` — for `set_all_new` only: no entry is live in the committed
  base. A reclaimed id's stale base entry always carries a `dm` tombstone, which
  makes `dm` non-empty and routes to the checked path, so this is only ever
  assumed on the fast arm.
* `resize` is growth-only. Shrinking is a separate branch in the Rust that drops
  entries; the callers only grow.

## Modelling boundary

* GraphBLAS pending work (`wait`, `wait_base`, `wait_all`, `is_synced`) has no
  denotational content: every operation behaves as if the layers were
  materialized, which the Rust guarantees by waiting on entry. `memory_usage`,
  `print`, and the `Encode`/`Decode` blob format are outside the model.
* `Iter`'s three-way sorted merge is modelled by its *result* (`eff` restricted to
  a row range), not as the lookahead algorithm. The same boundary the `Tensor`
  proofs draw for their iterators, and the one remaining place where the code is
  ahead of the proof.
* Indices are `Nat`. Where width matters the bound is proved instead of assumed —
  `nvals_no_underflow` is the `u64` subtraction chain.

## Keeping this honest

The proofs are about `Ops.lean`, so they are only as good as its correspondence to
`versioned_matrix.rs`. Change the Rust, change `Ops.lean`, re-run `lake build` — a
semantics change will break a proof, which is the point.
