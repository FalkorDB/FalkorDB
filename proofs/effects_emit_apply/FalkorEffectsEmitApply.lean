import FalkorEffectsEmitApply.Basic
import FalkorEffectsEmitApply.Emit
import FalkorEffectsEmitApply.Apply
import FalkorEffectsEmitApply.EmitDDL
import FalkorEffectsEmitApply.Liveness
import FalkorEffectsEmitApply.CancelGap

/-!
# effects v3: emit / apply (wave 2) — report

Target: `graph/src/effects/v3/emit.rs`, `graph/src/effects/v3/apply.rs`, and the
gate in front of the emitter (`runtime/ops/commit.rs:108`,
`Pending::effects_count` at `runtime/pending.rs:2013`). Coverage per Rust fn:
`COVERAGE.tsv` (36 rows, all PROVEN). Repros:
`graph/tests/lean_effects_emit_apply.rs`
(`cargo test -p graph --test lean_effects_emit_apply -- --nocapture --test-threads=1`).
Builds with `lake build`; no `sorry`, `admit` or `axiom`.

## Here | there

| Lean                                   | Rust                                                        |
|----------------------------------------|-------------------------------------------------------------|
| `Basic.memoStep`, `memoGroup`          | the `slots`/`index`/`last` loop, `emit.rs:520-558`, `808-840` |
| `Basic.groupBy`                        | first-appearance grouping (spec; also `proofs/replication`) |
| `Basic.sortDedup`                      | `labels.sort_unstable(); labels.dedup()`                    |
| `Emit.mergeRow`, `gatherRows`          | `gather_rows`, `emit.rs:917-945`                            |
| `Emit.schemaAdditions`                 | `emit_schema_additions`, `emit.rs:60`                       |
| `Emit.digestCancelled`                 | `digest_cancelled`, `emit.rs:395`                           |
| `Emit.forEachRecord`                   | `for_each_record`, `emit.rs:323` (opcode granularity)       |
| `Emit.effectsCount`                    | `Pending::effects_count`, `pending.rs:2013` (+2 per cancelled entity, plus `staged_labels_removed`) |
| `Emit.commitShips`                     | `if estimated > 0 { buf.build(..) }`, `commit.rs:108`       |
| `Emit.applyAddName`                    | `apply_add_schema` + `verify_id`, `apply.rs:560,586`        |
| `Apply.createOne`                      | `Record::CreateNode` arm, `apply.rs:214`                    |
| `Apply.updateNode`                     | `Record::UpdateNode` arm, `apply.rs:346`                    |
| `Apply.setLabels`                      | `Record::SetLabels` arm, `apply.rs:380`                     |
| `Apply.deleteNode`, `okDelete`         | `Record::DeleteNode` arm, `apply.rs:406`                    |
| `Apply.RG.WF`                          | "nothing hangs off a dead id; edge endpoints live"          |
| `Live.applyRec`, `Live.applyAll`       | the checked arms of `apply_record` (`apply.rs:164`) |
| `Live.needLive`, `ApplyHelpers.requireLive` | `require_live`, `apply.rs:685`                         |
| `Live.firstWithRels`                   | `Graph::first_node_with_relationships`, `graph.rs:2434` (spec) |
| `Live.danglingRel`                     | `verify_created_relationships`, `graph.rs:1648`            |

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
  arm preserves well-formedness.
* Counterexamples (all by `decide`, all reproduced in Rust):
  `gate_drops_schema_only`, `next_payload_refused`. `gate_drops_cancelled` and
  `cancelled_edge_attr_dropped` are kept as `pre3461_*` over `effectsCountPre3461`
  (the gate before it counted cancelled entities, a historical counterexample); the current
  gate is `gate_ships_of_cancelled`, `gate_ships_cancelled`, `cancelled_edge_attr_ships`.

## `Liveness.lean`: every record that acts on an entity checks it is live

* **`apply_wf`** — from a well-formed replica, a buffer whose records all apply and
  whose end-of-buffer `validate` passes leaves it well formed: no dead id carries a
  label, property or relationship, and every relationship ends at two live nodes.
  Via `applyRec_binv` (each checked arm keeps the batch invariant `BInv`, which lets
  a relationship *this batch created* end at a node *this batch released*) and
  `wf_of_validate` (`verify_created_relationships` closes exactly that gap).
* `needLive_ok`, `needDead_ok`, `firstWithRels_none`/`_some`; `requireLive_ok`,
  `requireLive_notLive` (`ApplyHelpers`): `require_live` refuses only as `NotLive` of
  the right kind; `fromNodeOp_spec` (`DanglingRelationship`), `validateG_ok` (
  `verify_id_batches` then `verify_created_relationships`).
* The Rust tests, replayed by `decide`: `edge_to_dead_refused`,
  `edge_to_created_applies`, `cancelled_pair` (nets out; without its `DELETE_EDGE`
  `validate` names relationship 1), `updates_on_dead_refused` /
  `updates_on_live_apply`, `delete_with_edges_refused` /
  `detach_delete_applies`.

## `CREATE_NODE` refuses a repeated id

* The `CREATE_NODE` arm refuses an id list with a repeat (`nodes.len() != ids.count()`, `apply.rs:226`).
  `applyRec .createNode` checks `ids.Nodup` first. `createNode_repeat_refused`, `createNode_ok_nodup`,
  `a_repeated_node_id_in_one_create_is_refused` (an abstract refusal check on the Rust test's input `[2, 3, 2]`: the `.graph` constructor only, the message text is not modelled). The commit gate (W2-effects-1, `commit.rs:107-108`) is as above.

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
  (`Inconsistent`/`AlreadyTaken` render as `Graph("{kind} {e}")`, never as a divergence),
  `fromNodeOp_spec` (`From<NodeOpError>`), `validateG_ok` (`Graph::validate` = node then
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

Still present on main (2026-10-04 live release build, Redis 8.6.2; the code is unchanged since):
1 and 4 below.

1. **A write that only registers schema replicates nothing** — `runtime/ops/commit.rs:108`
   gates `buf.build` on `effects_count() > 0` (`pending.rs:2013`), which does not count
   names registered since the `SchemaBaseline`, so `emit_schema_additions` never runs.
   The master keeps the new label/attr and the moved id space; the next payload's
   `ADD_SCHEMA` is refused ("assigns label 'B' id 1, but this replica would
   assign 0") -> forced full resync; during AOF load the server **exits**.
   Repros: `schema_only_write_ships_nothing_and_next_payload_is_refused`,
   `differential_random_with_fully_cancelled_writes` (134/200 seeds diverge; the
   same generator without such writes: 0/200). `OPTIONAL MATCH (n:Nope) SET n:L6`
   also forces a resync. C: replica and AOF reload agree, no resync.
   Fix: gate on "the emitter produced a record" (or add the schema delta to the count).
4. **A refused buffer is not rolled back for CREATE_INDEX**
   (`apply_effects`, `apply.rs:54`; `Graph::create_index`, `graph.rs:3670`): the
   `Indexer` is shared by `Arc` across versions (`Graph::new_version` clones the
   handle), so `MvccGraph::rollback` cannot undo it. Live: payload
   `[ADD_ATTRIBUTE 1 zz, CREATE_INDEX A(zz), CREATE_NODE 0]` -> `ERR ... the
   buffer was not applied`, yet `db.indexes()` lists `A [zz]` while
   `db.propertyKeys()` lacks `zz`. (Constraints are per-version: rolled back
   correctly.) In-process test `refused_buffer_leaves_its_index_behind` is
   `#[ignore]` (needs RediSearch). C has no v3 apply (#2698) to compare.

## Suspected, unconfirmed

* C master (`bin/macos-arm64v8-release/falkordb.so`, Redis 8.6.2) aborts on
  `OPTIONAL MATCH (n:Nope) SET n:L6` (`blocked.c:346 'server.also_propagate.numops
  == 0'`), with or without a replica. C-side; not investigated further.

## Gaps closed 2026-10-22: emit then apply, with cancellation (`CancelGap.lean`)

The emitter and the replica were proven separately; no theorem related what the
master emits for a segment with cancelled entities to what a replica at the
master's pre-segment state accepts. Two properties:

| Lean | Rust (b84473a7f) |
| --- | --- |
| `CancelGap.verifyRange` | `IdSpace::verify`'s hole check (id_space.rs:737-790) |
| `CancelGap.typesAfterCommit` | `Pending::commit` (pending.rs:1367-1413, #3461): `created_rels_by_type` keys via `create_relationships_bulk`, then every `cancelled_relationships` type via `get_type_id_mut` (`typesAfterCommitPre3461`, the pre-#3461 behaviour kept as a historical counterexample: the keys only) |
| `CancelGap.toLive` | opcode records → `apply_record` arms (`RemoveLabels` `require_live`, apply.rs:387-388) |

* **`RelIdsCovered`** (every relationship id allocated in the segment reaches a `CREATE_EDGE`):
  `digest_cancelled` skips a cancelled edge whose type has no id (emit.rs:451), so commit must
  register every cancelled type. `relIdsCovered_after_commit` (for every `Pending`),
  `fixed3447_cancelled_edge_id_gap`; `relIdsCovered_of_registered` (holds for every `Pending`
  once all created and cancelled types are registered). `pre3461_bug3447_cancelled_edge_id_gap`
  is the historical counterexample, where the payload has `CreateEdge [2]` and nothing for 1.
* **`ReplayApplies`**: `digest_labels(&p.remove_labels, None, …)` (emit.rs:370) precedes
  `digest_cancelled` (:374), so a cancelled node must not keep its `remove_labels`.
  `fixed3448_replay_applies` (pending_commit `fixed3448_cancelled_node_clears_remL`); the gate
  counts `staged_labels_removed` (`gate_keeps_staged_removal`). `bug3448_removeLabels_before_create_refused`
  (`CREATE (n:Foo) REMOVE n:Foo DELETE n` with the stale removal -> `NotLive node 0`) is the
  historical counterexample; control `agree3448_without_stale_remL`.

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

## Re-sync to 256ab4d3f (from 43936bab9)
* **#3446 (`543357f1e`)**: the `CREATE_NODE` arm of `apply_record` checks the attribute shape on the id *count*
  (`check_attr_shape(g, ids.len(), ..)`, the model's `checkAttrShape` already took a count), materialises the
  ids only when there are attributes, and otherwise writes the labels `NODE_LABEL_CHUNK = 2^16` ids per call
  (a record with neither never expands its ids). New in `Apply.lean`: `chunkLoop`, `chunkLoop_eq_write`
  (any chunk size `n > 0`: the chunked loop is the one-shot label write), `chunkLoop_labels`, so `createOne`
  / `wf_createOne` / `applyRec .createNode` stand unchanged. The label write now follows the shape check; a
  refusal is rolled back with the rest of the buffer either way.
* #3787 (clippy): `index_field_flags` takes `IndexType` by value, `index_options` likewise, `map_or_default`
  in `digest_created_edges`, `ok_or_else` in `resolve_type`, `ToString::to_string` paths; no behaviour change.
* #3683: `commit.rs` calls `runtime.upgrade_to_write()` directly; the effects gate is unchanged (`commit.rs:108`).
* Cites re-pinned by function name (`apply.rs` arms +28, `emit.rs`, `pending.rs:2013`).

## Re-pinned to 5e475e6c9 (#3436)
No model change. `emit.rs` (`for_each_record`, `build_index_buffer`, `digest_*`) takes
`&Graph` instead of `&AtomicRefCell<Graph>` and no longer borrows inside; the emitted records
are the same. Pins shifted by -1 or -2 in `emit.rs`, `pending.rs` -14 after `enforce_constraints`;
`CommitOp::next`'s effects gate is still `commit.rs:108`.
-/
