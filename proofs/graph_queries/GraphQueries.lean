/-
# graph_queries: `graph/src/graph/graph.rs` and `constraint.rs` against an abstract property graph

Reference model: `G` (Basic.lean) — counters, recycle bins, label/type/attribute
tables as lists, GraphBLAS matrices and tensors as their logical coordinate
sets (the abstraction proofs/versioned_matrix proves sound for
`VersionedMatrix<bool>`/`Tensor`). Every non-FFI fn of both files has a
theorem (COVERAGE.tsv); zero sorry/admit/axiom. Rust repros:
`graph/tests/lean_graph_queries.rs`; live servers Rust (release module) vs C.
Line numbers: origin/main `e8f8a3017` (re-targeted from `8743953a8`: #3022 adds
`verify_created_relationships` to `validate` and `first_node_with_relationships`; earlier
`fe619ac5f`: #2911 bounds `grow_cap`).

**Re-targeted to #2846 (IdSpace owns the ids).** `Graph` keeps two `IdSpace`s and
no counters; `G`'s `nodeCount`/`delNodes` (+ new `nodeEB`/`nodeTaken`) are the
node space's four fields, likewise for relationships. Every id move goes through
`O : IdSpaceOps`; its behaviour is the *hypothesis* `IdSpaceContract O`
(IdContract.lean — proofs/id_space owns the proof; the contract is the literal
effect of `create`/`release`/`cancel`/`open_batch` on `live`/`recycled` plus the
two refusals graph.rs relies on). Removed fns (`return_*_id`,
`open_*_id_space`, `mark_nodes_live`) are gone from the model and COVERAGE.

| module | covers |
| --- | --- |
| Schema      | `grow_cap`, `resize*`, label/type tables (`intern_label`, `get_label_id(_mut)`, `get_type_id(_mut)`, `get_*_matrix(_mut)`), `node_has_label*`, `edge_has_type`, counts |
| Attrs       | attribute dictionary (`get_or_create`, u16 cap), name↔id, store reads, index-update tracking |
| IdContract  | `IdSpace` readers (verbatim) and the `IdSpaceContract` hypothesis |
| Ids         | bounds, `max_*_id`, `node_id_space`, `cancel_*_id`, `open/roll/verify_id_batches`, `validate`, `grow_for_nodes`, `create_nodes`, `new`, `restore`, `new_version`, `rebuild_derived_matrices`, `trim_attr_stores` |
| Queries     | incident edges, degrees, `get_nodes`, label lookups, relationship/adjacency matrix builders |
| Delete      | `delete_relationships`, `delete_implicit_edges`, `create_relationships_bulk` |
| Labels      | `set_node_labels_product`, `set_nodes_labels_bulk`, `remove_nodes_labels`, `delete_nodes` |
| Index       | index glue, `memory_usage_report`, `encode_payload`, `get_plan` cache, `is_synced` |
| Constraints | constraint.rs + constraint management |
| Accessors   | projections, newtypes, attribute getters |
| FirstRel    | `first_node_with_relationships` (#3022): `firstNodeWithRels_spec` |

## Headline theorems
`growCap_ge_needed/_le_max/_reaches/_chunked`, `growStep_small` (#2911: `grow_cap` is total and
never exceeds `GrB_INDEX_MAX`), `resize_post`, `internLabel_spec`,
`getLabelIdMut_spec`, `getRelMatMut_spec`, `markNodesLive_spec` (now via
`create_nodes` + contract), `createNodes_refused`, `rollIdBatches_spec`
(a rolled batch is anchored at the boundary with an empty ledger),
`validate_ok_iff` (#3022: two verified batches **and** every relationship the batch
created still ends at two live nodes), `verifyCreatedRels_spec` (the empty-bin fast paths are
sound; an error names an offender from `taken`), **`firstNodeWithRels_spec`** (#3022: the
adjacency `Aᵀ·x` + backward-matrix products + re-seeked iterators return exactly the lowest node
of the set with a relationship in either direction, given the GraphBLAS `mxv`/iterator spec as
the hypothesis `BulkOps` and an exact adjacency matrix), `newVersion_spec` (each version opens a fresh batch),
`restore_ids`, `returnNode_bound`/`isNodeDeleted_return` (`cancel_*_id`),
`deleteRels_refused`, `deleteNodes_ids`, `deleteImplicit_freed`,
`createRelsBulk_ragged`/`_dup` (#2846's shape refusals; `createRelsBulk_endpoints`
no longer needs a Nodup hypothesis),
`restore_LInv`, `fillEndpoints_spec`, `clampTiers_mono`, `getOrCreate_spec`
(cap ⇒ `as u16` never aliases), `track_eq_ofType`, `rowsOfLabels_mem`,
`byType_both_once`, `inDeg_eq`…, `getNodesAll_mem`, `relMatrix_mem`,
`deleteRels_clears`, `deleteRels_adj`, `collectImplicit_mem`,
`deleteImplicit_no_dangling`, `createRelsBulk_endpoints`,
`setLabelsProduct_spec`/`setLabelsBulk_spec`/`removeLabels_spec`/`deleteNodes_spec`
(node×label matrix and label diagonals stay consistent), `labelPass_total`,
`two_phase_eq`, `upsert_twice`, `cmatches_iff`.

## CONFIRMED bugs (Rust vs C, live) — all three re-checked live at 2c874022a: still present
(the code each one cites is unchanged at fe619ac5f; #2911 touched only `grow_cap`)
1. **Deleted edge keeps its type after a type is registered without resize**
   (graph.rs:1229 `get_type_id_mut` has no `self.resize()`; :2853/:2834
   mask `relationship_cap × |types|` vs narrower type matrix → eWiseMult
   `GrB_DIMENSION_MISMATCH`, only `debug_assert`ed). `CREATE (:N{i:1})-[:A]->(:N{i:2})`;
   `GRAPH.CONSTRAINT CREATE g MANDATORY RELATIONSHIP B PROPERTIES 1 x`;
   `MATCH ()-[r:A]->() DELETE r`; `MATCH (a{i:1}),(b{i:2}) CREATE (a)-[:C]->(b)`;
   `MATCH ()-[r]->() RETURN type(r)` → Rust `A`, C `C`. Same via replica
   schema apply (effects/v3/apply.rs:573). Lean: `stale_type_after_registration`.
   Test `deleted_edge_type_survives_type_registration` (debug: panics
   GrB_DIMENSION_MISMATCH). Fix: call `self.resize()` in `get_type_id_mut`.
   (2c874022a: Rust `A`, C `C`.)
2. **Self-loop reported twice by `get_node_relationships`** (graph.rs:2214),
   surfacing in UDF `node.getNeighbors` (udf/js_classes.rs:104): self-loop
   count Rust 2 / C 1 for outgoing, incoming, both (2c874022a: with
   `{returnType:'edges'}` Rust 2, C 1; node results are deduplicated). Lean:
   `nodeRels_selfloop_twice` vs `byType_both_once`. Test `self_loop_reported_once`.
3. **Constraint identity is property-order sensitive** (constraint.rs:79):
   `UNIQUE NODE L (a,b)` then `(b,a)` — Rust creates both, C "Constraint
   already exists" (2c874022a: unchanged). Lean `cmatches_order_sensitive`; test
   `constraint_property_order_is_irrelevant`.

## Unconfirmed / notes
* `CREATE INDEX FOR (n:L) ON (n.a,n.b)` on a new label: Rust also reports
  `Labels added: 1`, C does not (label interned before the indexer can refuse,
  `createIndex_node_spec`).
* C itself refuses `GRAPH.CONSTRAINT DROP … (b,a)` after reporting (a,b) exists.
* UDF `node.getNeighbors()` with no argument: Rust "Error converting from js
  'undefined' into type 'object'", C answers (doc says the config is optional;
  udf/js_classes.rs, outside this project — seen live at 2c874022a).

## Historical: #2892 (fixed by #2911, `fe619ac5f`)
At 2c874022a `grow_cap` stepped with unchecked `u64` arithmetic, and `IdSpace::create` refused only
`u64::MAX`: `GRAPH.EFFECT` ids near 2^61 asserted in `GrB_Matrix_new`, `u64::MAX - 1` wrapped the step
(`oldStepWrapped`, Schema.lean) and spun forever. Now `growCap` terminates by construction, stays
`≤ GrB_INDEX_MAX` (`growCap_le_max`), and the contract clause `create_refuses_limit` (proven in
proofs/id_space, `create_out_of_range`) is what `markNodesLive_spec` uses to show every created id fits.

## IdSpace dependency
Theorems taking `(hC : IdSpaceContract O)` hold for any id space meeting it:
`returnNode_bound`, `returnRel_bound`, `isNodeDeleted_return`, `isRelDeleted_return`,
`rollIdBatches_spec`, `markNodesLive_spec`, `createNodes_refused`,
`deleteRels_clears`, `deleteRels_refused`, `deleteImplicit_freed`,
`deleteNodes_refused`, `deleteNodes_ids`. `verify` is uninterpreted (`O.verify`).

## Gaps
GraphBLAS fold/wait, mutex/indexer plumbing and RediSearch query delegations
are AXIOMATISED; matrix dimensions are modelled only where they matter
(`Mat.get`, `removeMask`); attribute *store* internals are abstract
(proofs/pending_commit, proofs/columnar); concurrency is not modelled.
-/
import GraphQueries.Basic
import GraphQueries.Schema
import GraphQueries.Attrs
import GraphQueries.IdContract
import GraphQueries.Ids
import GraphQueries.Queries
import GraphQueries.Delete
import GraphQueries.Labels
import GraphQueries.Index
import GraphQueries.Constraints
import GraphQueries.Accessors
import GraphQueries.FirstRel
