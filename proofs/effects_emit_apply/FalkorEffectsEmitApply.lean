import FalkorEffectsEmitApply.Basic
import FalkorEffectsEmitApply.Emit
import FalkorEffectsEmitApply.Apply
import FalkorEffectsEmitApply.EmitDDL
import FalkorEffectsEmitApply.Liveness

/-!
# effects v3: emit / apply (wave 2) — report

Target: `graph/src/effects/v3/emit.rs`, `graph/src/effects/v3/apply.rs`, and the
gate in front of the emitter (`runtime/ops/commit.rs:108`,
`Pending::effects_count` at `runtime/pending.rs:1688`). Coverage per Rust fn:
`COVERAGE.tsv` (36 rows, all PROVEN). Repros:
`graph/tests/lean_effects_emit_apply.rs`
(`cargo test -p graph --test lean_effects_emit_apply -- --nocapture --test-threads=1`).
Builds with `lake build`; no `sorry`, `admit` or `axiom`.

## Here | there

| Lean                                   | Rust                                                        |
|----------------------------------------|-------------------------------------------------------------|
| `Basic.memoStep`, `memoGroup`          | the `slots`/`index`/`last` loop, `emit.rs:525-563`, `814-846` |
| `Basic.groupBy`                        | first-appearance grouping (spec; also `proofs/replication`) |
| `Basic.sortDedup`                      | `labels.sort_unstable(); labels.dedup()`                    |
| `Emit.mergeRow`, `gatherRows`          | `gather_rows`, `emit.rs:923-951`                            |
| `Emit.schemaAdditions`                 | `emit_schema_additions`, `emit.rs:61`                       |
| `Emit.digestCancelled`                 | `digest_cancelled`, `emit.rs:397`                           |
| `Emit.forEachRecord`                   | `for_each_record`, `emit.rs:325` (opcode granularity)       |
| `Emit.effectsCount`                    | `Pending::effects_count`, `pending.rs:1688`                 |
| `Emit.commitShips`                     | `if estimated > 0 { buf.build(..) }`, `commit.rs:108`       |
| `Emit.applyAddName`                    | `apply_add_schema` + `verify_id`, `apply.rs:533,516`        |
| `Apply.createOne`                      | `Record::CreateNode` arm, `apply.rs:207`                    |
| `Apply.updateNode`                     | `Record::UpdateNode` arm, `apply.rs:308`                    |
| `Apply.setLabels`                      | `Record::SetLabels` arm, `apply.rs:348`                     |
| `Apply.deleteNode`, `okDelete`         | `Record::DeleteNode` arm, `apply.rs:379`                    |
| `Apply.RG.WF`                          | "nothing hangs off a dead id; edge endpoints live"          |
| `Live.applyRec`, `Live.applyAll`       | the checked arms of `apply_record` since #3022 (`apply.rs:157`) |
| `Live.needLive`, `ApplyHelpers.requireLive` | `require_live`, `apply.rs:648` (#3022)                  |
| `Live.firstWithRels`                   | `Graph::first_node_with_relationships`, `graph.rs:2245` (spec) |
| `Live.danglingRel`                     | `verify_created_relationships`, `graph.rs:1524`            |

## Proven (53 theorems)

* `memoGroup_eq_groupBy` — the memoised grouping loop of `digest_created_nodes`
  / `digest_deleted_nodes` produces exactly first-appearance `groupBy` (same
  slots, same order, same members in order). Closes the gap the replication
  proof left (it assumed `groupBy`).
* `sortDedup_set_eq` — two label vectors with the same members canonicalise to
  one shape key (`label_order_does_not_split_a_shape`, for all inputs).
* `groupBy_flat`, `groupBy_keyed` — grouping drops/duplicates nothing and every
  member carries its slot's key.
* `mergeRow_eq_lookup` — `gather_rows`' linear merge equals per-cell lookup
  when staged pairs and shape are strictly ascending; `gatherRows_length` — the
  emitter's rows always pass `check_attr_shape`. An `example` shows an unsorted
  staged vector makes the merge drop a cell (the invariant is load-bearing).
* `gate_zero_stream` — when `effects_count = 0` the stream the emitter would
  write is exactly the schema additions plus the cancelled pairs;
  `commitShips_eq_iff` — the gate is sound iff that residue is empty.
* `wf_createOne`, `wf_deleteNode` (live, no incident edge),
  `wf_updateNode_live`, `wf_setLabels_live` — with a liveness check each node
  arm preserves well-formedness (the fix, proved).
* Counterexamples (all by `decide`, all reproduced in Rust):
  `gate_drops_cancelled`, `gate_drops_schema_only`, `next_payload_refused`,
  `cancelled_edge_attr_dropped`. Historical, fixed by #3022 (`18fc277b9`):
  `pre3022_update_dead_not_wf`, `pre3022_update_dead_breaks_wf`,
  `pre3022_setlabels_dead_not_wf`, `pre3022_delete_endpoint_not_wf`.

## #3022 (`Liveness.lean`): every record that acts on an entity checks it is live

* **`apply_wf`** — from a well-formed replica, a buffer whose records all apply and
  whose end-of-buffer `validate` passes leaves it well formed: no dead id carries a
  label, property or relationship, and every relationship ends at two live nodes.
  Via `applyRec_binv` (each checked arm keeps the batch invariant `BInv`, which lets
  a relationship *this batch created* end at a node *this batch released*) and
  `wf_of_validate` (`verify_created_relationships` closes exactly that gap).
* `needLive_ok`, `needDead_ok`, `firstWithRels_none`/`_some`; `requireLive_ok`,
  `requireLive_notLive` (`ApplyHelpers`): `require_live` refuses only as `NotLive` of
  the right kind; `fromNodeOp_spec` (`DanglingRelationship`), `validateG_ok` (now
  `verify_id_batches` then `verify_created_relationships`).
* The new Rust tests, replayed by `decide`: `edge_to_dead_refused` (#2924),
  `edge_to_created_applies`, `cancelled_pair` (nets out; without its `DELETE_EDGE`
  `validate` names relationship 1), `updates_on_dead_refused` /
  `updates_on_live_apply` (W2-effects-2), `delete_with_edges_refused` /
  `detach_delete_applies` (W2-effects-3).

## Wave 4 additions (`Digests`, `ApplyHelpers`, `EmitDDL`)

* `digestCreatedEdges_perm`/`_keyed`, `mkEdgeRec_shape` — `digest_created_edges` emits exactly the created
  edges of registered types, columns aligned, rows = ids × attrs, ids ascending, every id carrying its
  record's attribute vector. `digestDeletedEdges_perm` — `digest_deleted_edges` loses/invents nothing and
  files each edge under its own type. `digestUpdates_perm`/`_shape` — every surviving updated entity exactly
  once, a deleted one never. `digestLabels_perm`/`_nonempty`. `digestCancelled_pairs` — the CREATE_EDGE and
  DELETE_EDGE halves name the same edges (the registered cancelled ones), bracketed by the node pair.
* `apply.rs`: `verifyId_ok`, `resolved_ok`, `verifySchema_ok`, `verifyAttribute_ok`,
  `applyAddSchema_eq_applyAddName` (the real get-or-create + verify is the replication model's
  `applyAddName`), `checkedTypeId_ok`, `checkedLabelIds_ok`, `checkAttrShape_ok` (accepts exactly in-range,
  strictly ascending attr ids and `ids × width` rows), `attrMap_row`, `indexOptions_spec`,
  `singleIndexLabel_ok`, `idSpaceErrorMap_injective` (on divergences), `idSpaceErrorMap_internal`
  (#2846: `Inconsistent`/`AlreadyTaken` render as `Graph("{kind} {e}")`, never as a divergence),
  `fromNodeOp_spec` (`From<NodeOpError>`, was `node_op`), `validateG_ok` (`Graph::validate` = node then
  relationship `verify`), `applyEffects_ok` (success ⇒ every record applied in order, `g.validate()`
  passed, then index commit), `applyEffects_validate_refused`.
* `emit.rs`: `replayNames_suffix` (a replica at the baseline accepts every schema addition and ends with the
  master's dictionary), `suffixRecs_ids`, `encodeAll_ok`/`_failed`, `indexFieldFlags_spec`,
  `buildIndexRec_verifies`/`buildConstraintRec_verifies` (emitted DDL always passes the replica's
  `verify_schema`/`verify_attribute`/`single_index_label` when dictionaries agree), `schemaId_spec`,
  `wireIndexOptions_spec`.

## Refinement interface (for proofs/replication)

| replication model | here |
| --- | --- |
| `intern` / `addLabel`'s id check | `FalkorEA.intern`, `applyAddSchema`, `applyAddSchema_eq_applyAddName : (applyAddSchema d i n k ≈ ok) = applyAddName d i n` |
| `schemaRecs` | `Emit.schemaAdditions`, `suffixRecs_ids`, `replayNames_suffix` |
| `groupBy NodeSpec.shape` / `groupBy Prod.snd` | `Basic.memoGroup_eq_groupBy : memoGroup key l = groupBy key l` (same `addTo`/`groupAux`/`groupBy` defs as replication) |
| `gatherRows` | `Emit.gatherRows`, `mergeRow_eq_lookup`, `gatherRows_length` |
| `okCreate` / `okDelete` | `checkAttrShape_ok`, `checkedLabelIds_ok`; id-space side in proofs/id_space (`create_spec`, `release_spec`, `refuseUndeletable_ok`, `refuseRecycled_ok`, `verify_ok_iff`) |
| `applyRecs` | `applyEffects`, `applyEffects_ok` |

## CONFIRMED bugs

**Status at e8f8a3017:** 2 and 3 are **fixed by #3022 (`18fc277b9`)**, as is #2924
(`CREATE_EDGE` to non-live endpoints) — see `Liveness.lean`. 1 and 4 are still
present (their code is unchanged by #3022). The text below is the record as found.

Re-checked against main `2c874022a` (after #2846 IdSpace refactor and #2916), live
release build on Redis 8.6.2, 2026-10-04: **all four still present** (1: replica
refuses the next payload "label id 0 out of range" / "attribute id 0 out of
range"; 2: `UPDATE_NODE [0] {v:7}` on an empty graph accepted, next `CREATE`
returns `n.v = 7`; 3: `DELETE_NODE` of an edge endpoint accepted, the next
`CREATE (:Y)` reuses id 0 and has the edge; 4: `CREATE_INDEX` + a refused
`DELETE_NODE 99` in one buffer leaves the index). #2924 (CREATE_EDGE between
non-live ids) also still accepted. A duplicate id inside one `CREATE_EDGE` is now
refused explicitly (`create_relationships_bulk`, #2846); a duplicate id inside
one `CREATE_NODE` is still silently de-duplicated (count 2, ids `[0,0]` → OK, one
node).

1. **A write that only registers schema or only cancels entities replicates
   nothing** — `runtime/ops/commit.rs:108` gates `buf.build` on
   `effects_count() > 0` (`pending.rs:1688`), which counts neither
   `cancelled_nodes`/`cancelled_relationships` nor names registered since the
   `SchemaBaseline`, so `digest_cancelled` and `emit_schema_additions` never run.
   The master keeps the new label/attr and the moved id space; the next payload's
   `ADD_SCHEMA` is refused ("assigns label 'B' id 1, but this replica would
   assign 0") → forced full resync; during AOF load the server **exits**.
   Repros: `fully_cancelled_write_ships_nothing_and_next_payload_is_refused`,
   `schema_only_write_ships_nothing_and_next_payload_is_refused`,
   `differential_random_with_fully_cancelled_writes` (134/200 seeds diverge; the
   same generator without such writes: 0/200). Live Rust 8.6.2 pair:
   `CREATE (a:A {p:1}) DELETE a` then `CREATE (:B {q:2})` → replica log
   "Replica diverged ... Scheduling a forced full resync"; AOF restart →
   "shutting down". `OPTIONAL MATCH (n:Nope) SET n:L6`, `MERGE (a:M1 {p5:1})
   DELETE a` also force a resync. C: replica and AOF reload agree, no resync.
   Variant, silent: `CREATE (a)-[:T1 {p4:1}]->(b) DELETE a, b` then
   `CREATE (:K4)-[:T2]->()` — replica's attribute dictionary is missing `p4`
   with no error (`cancelled_edge_attribute_is_a_silent_dictionary_divergence`;
   C agrees). Fix: gate on "the emitter produced a record" (or add
   `cancelled_*` and the schema delta to the count).
2. **UPDATE_NODE / UPDATE_EDGE / SET_LABELS / REMOVE_LABELS never check
   liveness** (`apply.rs:308,303,322,333`; `checked_label_ids` bounds-checks
   label ids only). A `GRAPH.EFFECT` naming a recycled id is accepted, and the
   next node that recycles that id is born with the property / label
   (`hostile_update_node_on_dead_id`: fresh node gets `{x: 666}`;
   `hostile_set_labels_on_dead_id`: fresh node is `['A','Fresh']`;
   `hostile_update_edge_on_dead_or_mistyped`: 6th new edge gets `{x: 888}`;
   `hostile_update_node_on_unallocated_id` accepted). Live Rust server: same
   (`update_dead` → next CREATE id 5 `{x: 666}`). Fix: route these ids through
   `IdSpace` like CREATE/DELETE (`wf_updateNode_live` proves it suffices).
   REMOVE_LABELS / SET_LABELS on 2^40 trip a GraphBLAS `debug_assert`
   (`matrix.rs:1139,1339`) in debug and are silently accepted in release —
   same family as #2892 (not re-reported).
3. **DELETE_NODE of a node with edges is accepted** (`apply.rs:379`,
   `delete_nodes` does not cascade or refuse): a dangling edge remains and the
   next created node inherits it (`hostile_delete_node_with_edges`; live server
   same). Sibling of #2924 (CREATE_EDGE to non-live endpoints), different
   record.
4. **A refused buffer is not rolled back for CREATE_INDEX**
   (`apply_effects`, `apply.rs:54`; `Graph::create_index`, `graph.rs:3332`): the
   `Indexer` is shared by `Arc` across versions (`Graph::new_version` clones the
   handle), so `MvccGraph::rollback` cannot undo it. Live: payload
   `[ADD_ATTRIBUTE 1 zz, CREATE_INDEX A(zz), CREATE_NODE 0]` → `ERR ... the
   buffer was not applied`, yet `db.indexes()` lists `A [zz]` while
   `db.propertyKeys()` lacks `zz`. (Constraints are per-version: rolled back
   correctly.) In-process test `refused_buffer_leaves_its_index_behind` is
   `#[ignore]` (needs RediSearch). C has no v3 apply (#2698) to compare.

## Suspected, unconfirmed

* The documented `rel_bound` gap (`emit.rs:450-456`): a cancelled edge of a
  never-registered type moves the master's relationship bound only. Tolerated in
  the differential; not seen to escalate in 200 random runs.
* C master (`bin/macos-arm64v8-release/falkordb.so`, Redis 8.6.2) aborts on
  `OPTIONAL MATCH (n:Nope) SET n:L6` (`blocked.c:346 'server.also_propagate.numops
  == 0'`), with or without a replica. C-side; not investigated further.

## Gaps / assumptions

* `forEachRecord` is at opcode granularity: it decides emptiness and schema,
  not the exact grouping of edges/updates/labels (FxHashMap order unmodelled).
  Row-level emit→apply agreement for nodes is the replication project's theorem;
  here it is only exercised by the differential Rust tests.
* The apply models are abstract replicas (`Apply.RG`, `Live.G`): no types or
  indexes, properties merged by prepend (only emptiness matters for WF), error
  payload strings not modelled. `Live` takes `require_live`'s meaning from
  `proofs/id_space` (`refuseNotLive_live`: accepts exactly the live ids) and
  `first_node_with_relationships`' from `proofs/graph_queries`. IdSpace `verify` is not re-modelled
  (see `proofs/id_space`).
* Index/constraint DDL is checked only by a live Rust-vs-C differential (18
  steps, all agree).
-/
