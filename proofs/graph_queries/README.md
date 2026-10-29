# graph_queries: the graph's own bookkeeping (ids, labels, types, deletes, constraints)

This project proves what `Graph` in `graph/src/graph/graph.rs` and `constraint.rs` promises about its own state, and shows with concrete counterexamples where it used to break or still does. The graph keeps many structures that must agree with each other: the id spaces, the adjacency matrix, the per-type relationship tensors, the relationship-type matrix, the node×label matrix and the per-label diagonals. Several user-facing behaviours depend on that agreement:

- `DELETE` and `DETACH DELETE` (the entity disappears from every structure, its id is recycled);
- `CREATE` reusing a deleted id (nothing of the old entity shows through);
- `type(r)`, `labels(n)`, degree and neighbour reads;
- `SET n:L` / `REMOVE n:L`;
- `GRAPH.CONSTRAINT CREATE` / `DROP`;
- commit-time validation of a write (`Graph::validate`) and `GRAPH.EFFECT` replay.

The full Lean↔Rust mapping (each module next to the `graph.rs` functions it covers) is in the header of [`GraphQueries.lean`](GraphQueries.lean), and every file starts with its own `file:line` table. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does deleting a relationship erase everything about it?**

`deleteRels_clears` proves that it does, as long as the relationship-type matrix is as wide as the type table: every deleted edge loses its type entry, its endpoints and its tensor entry, and its id goes into the recycle bin.

- **Before #3277, not always.** `get_type_id_mut` registered a new type without resizing `relationship_type_matrix`. The removal then hit a GraphBLAS dimension mismatch, which the release build ignores (only a `debug_assert`), so the deleted edge kept its type. When the id was reused by an edge of a new type, `type(r)` still answered the old type.
- **On main, yes.** `getTypeIdMut_width` proves registration keeps the width right, and `type_cleared_after_registration` proves the delete now clears the type.

**2. Do the id spaces and the graph agree at commit?**

Since #2846 the graph keeps no counters of its own; two `IdSpace`s own the ids. `validate_ok_iff` proves `Graph::validate` accepts exactly when both id batches verify **and** every relationship the batch created still ends at two live nodes (#3022, `verifyCreatedRels_spec`). `markNodesLive_spec` proves `create_nodes` gives every new id a matrix row, and `rollIdBatches_spec` proves rolling to a new version re-anchors both batches with an empty ledger.

These theorems take the id-space behaviour as an argument (`IdSpaceContract`). It is not left as an assumption: `rustIdOps_contract` proves the contract for a line-by-line model of the five `id_space.rs` mutators, and `IdRustUse.lean` applies every such theorem to it (`markNodesLive_spec_rust`, `deleteRels_clears_rust`, …).

**3. Can a matrix be grown past what GraphBLAS accepts?**

No. `growCap_le_max` proves `grow_cap` never exceeds `GrB_INDEX_MAX`, `growCap_ge_needed` that it reaches every size `IdSpace::create` lets through, and `growCap_chunked` that it grows in whole chunks (or stops at the clamp). Before #2911, an effect id near 2^61 asserted in `GrB_Matrix_new`, and `u64::MAX - 1` wrapped the step and spun forever (the old step is kept as `oldStepWrapped`).

**4. Does `first_node_with_relationships` find the right node?**

`firstNodeWithRels_spec` proves it returns exactly the lowest node of the set with a relationship in either direction. Every early `None` is sound. This is **CONDITIONAL**: it assumes the adjacency matrix is exactly the union of the relationship tensors (`AdjExact`). The mutators are meant to keep that true, but this project does not prove they do.

**5. Do labels stay consistent?**

`setLabelsProduct_spec`, `setLabelsBulk_spec`, `removeLabels_spec` and `deleteNodes_spec` prove the node×label matrix and the per-label diagonals stay in agreement (`LabCons`) through every label write and node delete. A deleted node keeps no label and no attributes.

**5b. Does a load keep the matrices it decodes?**

Yes, since #3250. `rebuild_derived_matrices` now takes `node_cap = max(node_cap, saved)` before resizing: `rebuild_never_shrinks` proves no node matrix ends smaller than decoded, and `rebuild_keeps_entries` that every in-bounds entry survives. Before, a decoded matrix larger than `restore`'s `node_cap` was cut down and lost every entry past it.

**6. Is each relationship reported once?**

`byType_both_once` proves `get_node_relationships_by_type(.., Both)` lists every incident edge exactly once, self-loops included. `get_node_relationships` does not: `nodeRels_selfloop_twice` proves it lists a self-loop twice (open, see below).

**7. Is a constraint identified by its property *set*?**

No. `cmatches_iff` proves constraint matching is plain equality of the property *list*, in order. So `UNIQUE (a, b)` and `UNIQUE (b, a)` are two constraints to Rust (`cmatches_order_sensitive`); C answers "Constraint already exists".

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `deleteRels_clears` | A deleted relationship loses its type entry, endpoints and tensor entry, and its id is recycled | `Delete.lean` |
| `type_cleared_after_registration` | Registering a type (constraint creation, a replica's schema record) no longer stops a later delete from clearing the type (#3277) | `Delete.lean` |
| `getTypeIdMut_width`, `registerRelType_width` | Registering a type resizes the type matrix to the new width | `Schema.lean` |
| `validate_ok_iff`, `verifyCreatedRels_spec` | Commit validation accepts exactly two verified batches whose created relationships end at live nodes; an error names a real offender | `Ids.lean` |
| `markNodesLive_spec`, `rollIdBatches_spec`, `newVersion_spec` | Creating nodes, rolling batches and opening a version move the ids exactly as intended | `Ids.lean` |
| `rustIdOps_contract` | The Rust `IdSpace` mutators satisfy the contract the theorems above assume | `IdRust.lean` |
| `growCap_le_max`, `growCap_ge_needed`, `growCap_chunked` | `grow_cap` always terminates, never passes `GrB_INDEX_MAX`, and reaches every legal size (#2911) | `Schema.lean` |
| `firstNodeWithRels_spec` | `first_node_with_relationships` returns the lowest node with a relationship (CONDITIONAL on `AdjExact`) | `FirstRel.lean` |
| `deleteImplicit_no_dangling`, `deleteImplicit_freed` | `DETACH DELETE` leaves no (non-explicit) edge touching a deleted node, and every edge it removes ends up in the recycle bin | `Delete.lean` |
| `createRelsBulk_endpoints`, `createRelsBulk_ragged`, `createRelsBulk_dup` | Bulk relationship creation records the right endpoints and refuses malformed batches (#2846) | `Delete.lean` |
| `setLabelsProduct_spec`, `deleteNodes_spec` | Label writes and node deletes keep the label matrices consistent | `Labels.lean` |
| `getOrCreate_spec` | The attribute dictionary never hands out an id that wraps when cast to `u16` | `Attrs.lean` |
| `byType_both_once` | Typed neighbour reads list each incident edge once | `Queries.lean` |
| `two_phase_eq`, `upsert_twice` | Constraint validation's two phases equal a one-pass update; a constraint announced twice is not duplicated | `Constraints.lean` |

### How `deleteRels_clears` is proved

1. **Model the type.** `G` (`Basic.lean`) is `struct Graph` with every GraphBLAS object replaced by its mathematical content. A matrix `Mat` is its dimensions plus a list of stored coordinates. A tensor `Ten` is a list of `(src, dst, edge_id)` triples. `proofs/versioned_matrix` proves this abstraction is sound (its `run_repr`).
2. **Model the failure mode.** `Mat.removeMask` removes the masked coordinates only when the mask and the matrix have the same dimensions; otherwise it changes nothing. That is what the release build does, because the `GrB_DIMENSION_MISMATCH` is only `debug_assert`ed.
3. **Model the function.** `deleteRels` follows `delete_relationships`. Phase 1 (`resolveRels`) finds each edge's type and endpoints without mutating. Phase 2 releases the ids through the id space `O`, filters the tensors, and removes the mask from the type matrix and the adjacency matrix.
4. **Case structure.** There are three cases: an empty input, a refused release (`deleteRels_refused`: nothing changes), and success. In the success case each of the four facts follows from the definitions.
5. **Hypotheses.** There are two:
   - `hC : IdSpaceContract O` is discharged by `rustIdOps_contract`, so `deleteRels_clears_rust` holds with no assumption.
   - `hw` says the type matrix is as wide as the type table. This is exactly what was false before #3277, and `deleteRels_stale` shows that without it the removal is a no-op.

## Bugs found

Each bug was reproduced against a live server and compared with C before it was recorded (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W4-graph-1 | removed (17ccc5412) | A deleted edge kept its type after a constraint or replica registered a new type; a reused id then showed the old type in `type(r)` (Rust `A`, C `C`) | Fixed by #3277 (closes #3244; `type_cleared_after_registration`) |
| W4-graph-2 | `nodeRels_selfloop_twice` | `get_node_relationships` lists a self-loop twice; UDF `node.getNeighbors` counts it twice (Rust 2, C 1) | **Open** (last re-checked live at 2c874022a; FINDINGS notes it may be the same as #3083) |
| W4-graph-3 | `cmatches_order_sensitive` | `UNIQUE NODE L (a,b)` then `(b,a)` creates both; C says "Constraint already exists" | **Open** |
| #3246 | removed (17ccc5412) | A load resized every node matrix to `restore`'s `node_cap` (from the highest id), below the saving server's grown size, dropping entries past it; for tensors this was where `me` kept ids of cut-off pairs | Fixed by #3250 (`2eacf0ece`): `rebuild_never_shrinks`, `rebuild_keeps_entries` |
| #2892 | — (old step kept as `oldStepWrapped`) | `GRAPH.EFFECT` ids near 2^61 asserted in `GrB_Matrix_new`; `u64::MAX - 1` wrapped `grow_cap` and hung | Fixed by #2911 (`growCap_le_max`) |

Also recorded, not a bug in this project's sense: `CREATE INDEX` on a new label reports `Labels added: 1` in Rust but not in C, because the label is interned before the indexer can refuse.

**CONDITIONAL row:** `first_node_with_relationships` rests on `AdjExact` (the adjacency matrix is exactly the union of the relationship tensors). The graph's mutators are designed to keep it, but no theorem here proves they do.

## Files

| File | What's in it |
|---|---|
| `Basic.lean` | The abstract graph `G`, `Mat`, `Ten`, `Constraint` |
| `Schema.lean` | `grow_cap`/`resize`, label and type tables, `get_*_id_mut`, `register_relationship_type`, label/type checks, counts; the pre-#3277 model |
| `Attrs.lean` | Attribute-name dictionary (`get_or_create`, the `u16` cap), name↔id, store reads |
| `IdContract.lean` | `IdSpace` readers restated verbatim and the hypothesis `IdSpaceContract` |
| `IdRust.lean` | `rustIdOps` (the Rust `IdSpace` mutators) and `rustIdOps_contract` |
| `IdRustUse.lean` | Every contract-taking theorem instantiated at `rustIdOps` (`*_rust`) |
| `Ids.lean` | Id bounds, batches, `validate`, `create_nodes`, `new`, `restore`, `new_version`, derived-matrix rebuild |
| `Queries.lean` | Incident edges, degrees, `get_nodes`, label lookups, matrix builders |
| `Delete.lean` | `delete_relationships`, `delete_implicit_edges`, `create_relationships_bulk`; bug W4-graph-1 |
| `Labels.lean` | Label writes and `delete_nodes` |
| `Index.lean` | Index glue, memory report, payload dispatch, plan cache |
| `Constraints.lean` | `constraint.rs` and constraint management; bug W4-graph-3 |
| `Accessors.lean` | Projections, newtypes, attribute getters |
| `FirstRel.lean` | `first_node_with_relationships` (#3022) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The lakefile has no path dependencies. `IdRust.lean` copies `proofs/id_space`'s model into list form rather than importing it.

## Coverage

`COVERAGE.tsv` has 198 rows (194 in `graph.rs`, 4 in `constraint.rs`):

| Bucket | Rows |
|---|---|
| PROVEN | 185 |
| CONDITIONAL | 1 |
| AXIOMATISED | 12 |

- **CONDITIONAL:** `first_node_with_relationships`, on `AdjExact` (above).
- **AXIOMATISED:** functions that only hand off to code outside this project:
  - GraphBLAS fold and wait (`flush_for_bulk`, `fold_oversized_deltas`, `wait_all`; `proofs/versioned_matrix` proves folds keep the logical content);
  - RediSearch delegations (`is_indexed`, `get_indexed_nodes`, `is_edge_indexed`, `get_indexed_edges`, `fulltext_query_nodes`, `fulltext_query_edges`, `cancel_indexing`);
  - mutex plumbing (`index_locks`, `set_indexer_graph`).

## What the model does not cover

- Matrix dimensions are modelled only where they matter (`Mat.get`, `Mat.removeMask`); elsewhere a matrix is its coordinate set.
- Attribute *store* internals are abstract (they belong to `proofs/pending_commit` and `proofs/columnar`); attribute values are an abstract type `V`.
- The indexer is a parameter: index theorems hold for any indexer behaviour, and the native-index plumbing belongs to `proofs/index_layer`.
- `first_node_with_relationships` takes the GraphBLAS `mxv`/iterator spec as the hypothesis `BulkOps` (`proofs/graphblas_wrappers` proves `count_in_rows` against it).
- Concurrency is not modelled.

## Build and check

```bash
cd proofs/graph_queries && lake build    # this project
bash proofs/lean_ci.sh graph_queries     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                  # repo-wide coverage report
```

**When you change `graph.rs`, `constraint.rs` or `id_space.rs`:**

1. Find the function in `COVERAGE.tsv` (for `id_space.rs`, update `IdRust.lean` as well as `proofs/id_space`).
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
