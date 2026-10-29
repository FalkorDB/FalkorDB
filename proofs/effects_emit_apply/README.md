# effects_emit_apply: which effects a write ships, and what a replica does with them

When a write query commits on a master, the engine turns its pending changes into an *effects payload*: a list of records such as `CREATE_NODE`, `UPDATE_EDGE`, `ADD_SCHEMA`, `DELETE_NODE`. The replica (or an AOF reload, or a client `GRAPH.EFFECT`) applies that list to its own copy of the graph. This project proves properties of both ends:

- **emit** (`graph/src/effects/v3/emit.rs`, plus the gate in `runtime/ops/commit.rs` that decides whether to emit at all): which records a commit produces, grouped how, with which rows;
- **apply** (`graph/src/effects/v3/apply.rs`): which records the replica accepts, and whether accepting them can leave its graph in a broken state.

If emit gets this wrong, the replica silently differs from the master, or refuses the next payload and needs a full resync. If apply gets it wrong, a hostile or out-of-sync payload can corrupt the replica: a recycled node id inherits a stranger's properties or an edge.

The byte encoding of the records is a separate project, [`../effects_codec`](../effects_codec). The end-to-end "master and replica agree" theorem is in [`../replication`](../replication), which imports this project.

The full Lean↔Rust mapping is in the header of [`FalkorEffectsEmitApply.lean`](FalkorEffectsEmitApply.lean) and in the table at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does every write that changes the master ship a payload?**

**No, and this is still open (W2-effects-1).** `commit.rs` calls the emitter only when `Pending::effects_count()` is positive. That count does not include entities created and deleted in the same query (*cancelled* entities), nor labels, types and attribute names registered by the query. But the emitter would ship something for both:

- `gate_zero_stream`: when the count is 0, the records the emitter would write are exactly the schema additions plus the cancelled create/delete pairs;
- `commitShips_eq_iff`: the gate is correct exactly when that residue is empty.

The counterexamples are concrete queries:

- `CREATE (a:A {p:1}) DELETE a` registers label `A` and attribute `p` on the master and ships nothing (`gate_drops_cancelled`);
- `OPTIONAL MATCH (n:Nope) SET n:L6` registers `L6` and ships nothing (`gate_drops_schema_only`);
- the next write's `ADD_SCHEMA 1 "K6"` is then refused by the replica, which would assign id 0 (`next_payload_refused`). That forces a full resync, and during AOF load the server exits;
- `CREATE (a)-[:T1 {p4:1}]->(b) DELETE a, b` is worse: no error at all, but the replica's attribute dictionary is missing `p4` (`cancelled_edge_attr_dropped`).

**2. Can a payload that the replica accepts leave its graph broken?**

- **Before #3022, yes.** `UPDATE_NODE`, `UPDATE_EDGE`, `SET_LABELS` and `REMOVE_LABELS` did not check that their ids were live, and `DELETE_NODE` accepted a node that still had edges. A `GRAPH.EFFECT` naming a deleted id was accepted, and the next `CREATE` that recycled the id returned a node that already had `{x: 666}`, or an extra label, or someone else's edge (counterexample removed; in 17ccc5412).
- **On main, no.** `apply_wf`: start from a well-formed replica, apply any list of records that the replica accepts, and pass the end-of-buffer `validate`. The result is well formed:
  - a dead node has no labels and no properties;
  - a dead relationship has no properties;
  - relationship ids are distinct;
  - every relationship ends at two live nodes.

**3. Is the emitter's grouping the one the replication proof assumes?**

Yes. The emitter groups created and deleted nodes by shape (label set and attribute ids) with a hand-written loop that keeps a slot table, an index, and a "same as last time" shortcut. `memoGroup_eq_groupBy` proves the loop produces exactly plain first-appearance grouping: the same groups, in the same order, with the same members in the same order. `replication` states its theorem over the plain grouping and imports this result, so it holds of the real loop. `sortDedup_set_eq` proves label order never splits a shape: `(:A:B)` and `(:B:A)` land in one record.

**4. Does every record the emitter builds pass the replica's checks?**

Yes, given that the replica's dictionaries agree with the master's baseline:

- `replayNames_suffix`: the replica accepts every `ADD_SCHEMA`/`ADD_ATTRIBUTE` the master emits and ends with the master's dictionary;
- `gatherRows_length` and `mkEdgeRec_shape`: the rows always have `ids × attrs` cells, so they pass `check_attr_shape`;
- `buildIndexRec_verifies` and `buildConstraintRec_verifies`: emitted index and constraint DDL always passes `verify_schema`, `verify_attribute` and `single_index_label`.

**5. Is a refused payload fully rolled back?**

Not for indexes, and this is still open (W2-effects-4). `apply_effects` applies records into a new graph version and drops it if any record or `validate` fails (`applyEffects_ok`, `applyEffects_validate_refused`). But the `Indexer` is shared by `Arc` across versions, so a `CREATE_INDEX` earlier in a refused buffer stays. This was found live; the Lean model of the replica has no indexes, so there is no theorem for it.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `apply_wf` | Any accepted list of records, followed by a passing `validate`, keeps the replica well formed | `Liveness.lean` |
| `applyRec_binv`, `wf_of_validate` | The two steps of `apply_wf`: each checked record keeps a weaker mid-batch invariant, and `validate` closes the gap | `Liveness.lean` |
| `memoGroup_eq_groupBy` | The emitter's memoised grouping loop is exactly first-appearance `groupBy` | `Basic.lean` |
| `gate_zero_stream`, `commitShips_eq_iff` | When `effects_count` is 0, the emitter would still write the schema additions and cancelled pairs; the gate is correct only when those are empty | `Emit.lean` |
| `mergeRow_eq_lookup`, `gatherRows_length` | `gather_rows`' linear merge gives the same cells as a lookup per cell, and its rows have the shape `check_attr_shape` wants | `Emit.lean` |
| `digestCreatedEdges_perm`, `digestUpdates_perm`, `digestLabels_perm`, `digestCancelled_pairs` | Each digest loses nothing and invents nothing; a deleted entity never appears as updated | `Digests.lean` |
| `replayNames_suffix` | A replica at the master's baseline accepts every schema addition and ends with the master's dictionary | `EmitDDL.lean` |
| `buildIndexRec_verifies`, `buildConstraintRec_verifies` | Emitted index/constraint DDL always passes the replica's checks | `EmitDDL.lean` |
| `applyAddSchema_eq_applyAddName` | The real get-or-create-and-verify of a name is the replication model's `applyAddName` | `ApplyHelpers.lean` |
| `applyEffects_ok` | `apply_effects` succeeds only if every record applied in order and `validate` passed | `ApplyHelpers.lean` |
| `relIdsCovered_of_registered` | When every created and cancelled relationship's type is registered, every relationship id the master allocated reaches the replica in a `CREATE_EDGE` (the property #3447 breaks) | `CancelGap.lean` |

### How `apply_wf` is proved

1. **Model the replica.** `Live.G` is an abstract graph: which node ids are live, labels and properties per node, the live relationships as `(id, src, dst)`, relationship properties, and the two per-batch lists `IdSpace` keeps (`released` node ids and `taken` relationship ids). Types, indexes and error strings are left out.
2. **Model the records.** `Live.Rec` has the eight entity-facing records. `applyRec` is the matching arm of `apply_record` on main: `needLive` is `require_live` (it refuses an id that is not live), `needDead` is `IdSpace::create`'s refusal of a live id, and `DELETE_NODE` refuses a node with relationships (`firstWithRels`, the spec of `first_node_with_relationships`). `danglingRel` is `verify_created_relationships`, which `validate` runs at the end of the buffer.
3. **Use a weaker invariant mid-batch.** `WF` cannot hold after every record. Records of one kind are grouped together, so a buffer can delete a node and only later create, then delete, an edge onto it. The Rust test `cancelled_pair` replays exactly that: create node 2, delete node 2, create edge 1 from 0 to 2, delete edge 1. So in the middle of a batch, a relationship may end at a node that was released *in this batch*. `BInv` is `WF` with exactly that exemption. `binv_of_wf` starts it, `applyRec_binv` shows each record keeps it, and `applyAll_binv` folds that over the list.
4. **Close the gap at the end.** `wf_of_validate`: if `BInv` holds and `danglingRel` finds no relationship of this batch hanging off a dead node, then `WF` holds.
5. **Hypotheses.** `apply_wf` takes `require_live`'s meaning from `id_space` (`refuseNotLive_live`: it accepts exactly the live ids) and `first_node_with_relationships`' from `graph_queries`. Both are proven there against the Rust, so these are not open assumptions, but the projects are not linked: a change to either function needs both proofs re-checked.

## Bugs found

All four were reproduced in Rust and against a live server (see [`../FINDINGS.md`](../FINDINGS.md), wave 2). The Rust repro tests are in `graph/tests/lean_effects_emit_apply.rs`, which is not committed because some of its tests fail on purpose.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W2-effects-1 | `gate_drops_cancelled`, `gate_drops_schema_only`, `next_payload_refused`, `cancelled_edge_attr_dropped` | A write that only registers a name or only cancels entities ships nothing. The next payload is refused (full resync; AOF load exits), or the replica silently misses an attribute name. 134 of 200 random seeds diverge | **Open**: the gate at `commit.rs:107-108` is unchanged on 1d15bf713; no issue or PR yet |
| W2-effects-2 | removed (17ccc5412) | `UPDATE_NODE`/`UPDATE_EDGE`/`SET_LABELS`/`REMOVE_LABELS` on a dead id were accepted; the next node to recycle the id inherited the property or label | Fixed by #3022 (`wf_updateNode_live`, `wf_setLabels_live`, `apply_wf`) |
| W2-effects-3 | removed (17ccc5412) | `DELETE_NODE` of a node with edges was accepted; the edge dangled and the next node with that id inherited it | Fixed by #3022 (`wf_deleteNode`, `apply_wf`) |
| #3247 | removed (17ccc5412) | A `CREATE_NODE` naming the same id twice (`[2, 3, 2]`) applied: the id set created one node where the record named two, and the repeated id's row had no node of its own | Fixed by #3298 (`2bb51088b`): `createNode_repeat_refused`, `createNode_ok_nodup`, `a_repeated_node_id_in_one_create_is_refused` |
| #3447 | `bug3447_cancelled_edge_id_gap` (property `RelIdsCovered`) | An edge of a new type cascaded away by `DETACH DELETE` never registers its type, so `digest_cancelled` drops its id. With a surviving edge above it in the same segment the replica's relationship id space has a hole: `IdSpace::verify` refuses and forces a full resync. C stays in sync | **Open** (#3447) |
| #3448 | `bug3448_removeLabels_before_create_refused` (property `ReplayApplies`) | `CREATE (n:Foo) REMOVE n:Foo DELETE n` leaves `n` in `remove_labels`; the stream emits `RemoveLabels [0]` before the cancelled pair's `CreateNode [0]`, and the replica refuses node 0 as not live (full resync). C stays in sync | **Open** (#3448) |
| W2-effects-4 | none (the model has no indexes) | A refused buffer's `CREATE_INDEX` is not rolled back, because the `Indexer` is shared by `Arc` across versions | **Open** (PR #3235) |

#2924 (`CREATE_EDGE` between non-live ids) was fixed by the same PR #3022, and `apply_wf` covers it: `edge_to_dead_refused` replays the Rust test.

One smaller observation, not filed:

- An `example` in `Emit.lean`: `gather_rows`' merge drops a cell if the staged attributes are not sorted. The emitter always sorts them, so this only shows the sort is load-bearing.

## Files

| File | What's in it |
|---|---|
| `FalkorEffectsEmitApply.lean` | Root: the report header (Lean↔Rust table, theorem list, bug write-ups, gaps) |
| `Basic.lean` | The values the emitter moves around, canonical label sets (`sortDedup`), first-appearance `groupBy`, the memo loop and `memoGroup_eq_groupBy` |
| `Emit.lean` | `gather_rows` (`mergeRow`, `gatherRows`), the record stream `forEachRecord`, `effectsCount`, the commit gate `commitShips`; the W2-effects-1 counterexamples |
| `Digests.lean` | `digest_created_edges`, `digest_deleted_edges`, `digest_updates`, `digest_labels`, `digest_cancelled`, column by column |
| `EmitDDL.lean` | Schema additions (`replayNames_suffix`), `encode_all`, index/constraint DDL buffers and their verification |
| `Apply.lean` | The pre-#3022 apply arms over a simple replica (`Apply.RG`), its well-formedness, the W2-effects-2/3 counterexamples |
| `ApplyHelpers.lean` | `apply.rs`'s checks function by function: `verify_id`, `require_live`, `check_attr_shape`, error mapping, `apply_effects` |
| `Liveness.lean` | The apply side on main (#3022): `Live.G`, `applyRec`, `BInv`, `apply_wf`, the Rust tests replayed by `decide` |
| `CancelGap.lean` | Emit then apply with cancellation: `RelIdsCovered` (#3447) and `ReplayApplies` (#3448), refuted on main; `relIdsCovered_of_registered` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no Lake dependencies. [`../replication`](../replication) requires it as a path dependency to import the memo loop.

## Coverage

`COVERAGE.tsv` has 36 rows: 18 in `apply.rs`, 16 in `emit.rs`, plus `Pending::effects_count` (`pending.rs`) and the commit gate (`commit.rs`).

| Bucket | Rows |
|---|---|
| PROVEN | 36 |

Every row is PROVEN. For `effects_count` and the gate, "PROVEN" means the model is exact and the theorems say what it does, including that it is wrong for cancelled and schema-only writes. Likewise `for_each_record`, `digest_cancelled` and `digest_labels` are PROVEN as models, and `CancelGap.lean` proves that what they emit is wrong for #3447 and #3448.

## What the model does not cover

- **Record order within an opcode.** `forEachRecord` decides which kinds of record are emitted and the schema additions, not the exact grouping of edges, updates and labels: Rust's `FxHashMap` iteration order is unspecified. The digest theorems are stated order-insensitively (permutations and per-record properties). Row-level emit→apply agreement for nodes is proven in `replication`; for edges it is exercised only by differential Rust tests.
- **The replica is abstract.** `Apply.RG` and `Live.G` have no types and no indexes. Properties are merged by prepending, which is enough for well-formedness (only "empty or not" matters there) but not for property values. Error message strings are not modelled.
- **Indexes and rollback.** W2-effects-4 is outside the model.
- **Index and constraint DDL apply** is checked only by a live Rust-vs-C differential (18 steps, all agree), not by a theorem.
- **`IdSpace` and `first_node_with_relationships`** are specs here, proven in `id_space` and `graph_queries`.
- The `rel_bound` gap documented at `emit.rs:448-454` is no longer a gap: it is #3447, proven in `CancelGap.lean` (`bug3447_cancelled_edge_id_gap`). The replica's `IdSpace::verify` is modelled there only as its hole check (`verifyRange`).

## Build and check

```bash
cd proofs/effects_emit_apply && lake build    # this project
bash proofs/lean_ci.sh effects_emit_apply     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                       # repo-wide coverage report
```

`replication` depends on this project, so rebuild it too after a change here: `cd proofs/replication && lake build`.

**When you change `effects/v3/emit.rs`, `effects/v3/apply.rs`, `Pending::effects_count` or the emit gate in `runtime/ops/commit.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A fix for W2-effects-1 should turn `gate_drops_cancelled` and its siblings into `pre<PR>_` history and prove that the new gate ships whenever `forEachRecord` is non-empty (the right-hand side of `commitShips_eq_iff`). If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
