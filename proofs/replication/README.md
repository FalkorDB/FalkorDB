# replication: a replica never serves a graph that differs from its master's

FalkorDB replicates writes as *effects*, not as queries. The master runs the query, then sends a list of records describing what changed (`ADD_SCHEMA`, `CREATE_NODE`, `DELETE_NODE`, …). The replica applies those records; it never re-runs Cypher. This project proves that the scheme works:

- **`emit_apply`**: the payload the master builds from a write, applied by the replica to the state the master started from, succeeds and lands on *exactly* the state the master committed;
- **`replica_never_serves_a_different_graph`**: over any stream of writes, including the failure path, the replica is always either serving exactly the master's graph, resyncing, or halted. It is never quietly serving a different graph.

Every write query on a deployment with replicas or AOF depends on this. Since `ed5b31553` ("effects as the only replication mechanism") there is no query replay, and that is what makes the property provable: the replica never evaluates `rand()`, reads a clock, or picks a plan, so there is nothing for the two sides to disagree about.

This project is the top of a stack:

- [`../effects_emit_apply`](../effects_emit_apply) proves the emitter's and applier's individual steps (this project imports its grouping loop);
- [`../effects_codec`](../effects_codec) proves the records survive the trip through bytes;
- [`../id_space`](../id_space) and [`../pending_commit`](../pending_commit) prove the id allocator and the master's commit.

The full Lean↔Rust mapping is in the header of [`FalkorReplication.lean`](FalkorReplication.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does the replica end up with the same graph as the master?**

Yes, for each write (`emit_apply`) and for any sequence of writes (`stream_simulation`), as long as each write satisfies `PendingWF`: the facts the master's write path guarantees before it commits (listed under question 5).

The interesting part is that the emitter does **not** ship changes in the order the master applied them. It groups created nodes by *shape* (label set plus attribute ids) and deleted nodes by label set, one record per group. For

```cypher
CREATE (:A {x:1}), (:B {x:2}), (:A {x:3})
```

the master creates ids 0, 1, 2 in order, but the payload carries one `CREATE_NODE` for ids `[0, 2]` and a second one for `[1]`. The replica rebuilds the graph from a different order. Three theorems say that is sound:

- `createBatch_perm` and `deleteBatch_perm`: creating or deleting a set of nodes in any order gives the same graph;
- `specsOf_group`: a record decodes back to exactly the group it came from, so row *k* lands on the k-th id and nothing is dropped, duplicated or transposed.

The `#guard` checks at the end of `FalkorReplication.lean` run this exact example and check that values 1, 2, 3 land on nodes 0, 1, 2.

**2. What if the replica has missed a payload, or numbers its labels differently?**

It refuses the buffer instead of applying it to the wrong graph. Two checks do this:

- **Entity ids are sent, then validated.** The replica is told which ids to create, so it never allocates. `Batch.verify` (`IdSpace::verify`) checks that the batch's fresh ids start exactly at the replica's own id boundary. A replica that missed an earlier write has a different boundary and refuses.
- **Schema ids are re-derived, then compared.** For `ADD_SCHEMA 1 "K6"` the replica interns `K6` itself and refuses unless it also gets id 1 (`apply_add_schema` / `verify_id`).

Both are checked by `#guard`: a replica whose label dictionary is `["X"]` refuses, and a replica that missed `p1` refuses `p3`.

**3. Do those checks fire on a healthy pair?**

No: `no_spurious_resync`. If the replica is in sync and the write is well formed, the replica stays in sync. So correctness is not bought by resyncing at the first difficulty.

**4. What happens when a check does fire?**

`src/divergence_guard.rs` decides, and `Guard.lean` models it function by function:

- a client `GRAPH.EFFECT` that fails just returns an error (`onFailure_client_noop`);
- a replica applying its master's stream schedules a full resync with `REPLICAOF NO ONE` then `REPLICAOF <master>` (`onFailure_replicated_resyncs`);
- while the server is loading (replaying its AOF) it exits, since there is no master to resync from (`onFailure_loading_exits`).

`stepNode_failure_matches_guard` and `guard_loading_halts` tie these to the state machine the headline theorem is about: the abstract `resync` and `halted` states are exactly what the guard does.

**5. What must the master guarantee for this to hold?**

`PendingWF` lists it:

- the ids were allocated by `IdSpace::reserve`: recycled ids first, then fresh ones (`allocated`);
- attribute ids within a node are distinct, and every label and attribute id is registered (`attrsNodup`, `labelsOk`, `attrsOk`);
- the new names are new (`newLabelsOk`, `newAttrsOk`);
- deleted ids are distinct (`delsNodup`);
- no id is both created and deleted in the same write (`disjoint`);
- every id is below `ID_LIMIT` (`inLimit`, from #2911).

`disjoint` is an assumption, not a derived fact. In the engine, a node created and deleted in one query is unwound out of `Pending` and shipped as a separate create/delete pair by `digest_cancelled`, which this model does not include. See [What the model does not cover](#what-the-model-does-not-cover): that is where W2-effects-1 lives.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `emit_apply` | The payload the master builds, applied to the state it started from, succeeds and gives exactly the master's new state | `FalkorReplication.lean` |
| `stream_simulation` | The same over any sequence of writes | `FalkorReplication.lean` |
| `replica_never_serves_a_different_graph` | Over the guarded state machine, whenever the replica is serving, it serves the master's graph | `FalkorReplication.lean` |
| `step_preserves`, `no_divergence` | The invariant behind it: in sync and equal, or resyncing, or halted | `FalkorReplication.lean` |
| `no_spurious_resync` | A healthy pair never takes the failure branch | `FalkorReplication.lean` |
| `createBatch_perm`, `deleteBatch_perm` | Regrouping created/deleted nodes does not change the result | `FalkorReplication.lean` |
| `specsOf_group` | A record decodes back to exactly its group: row k on the k-th id | `FalkorReplication.lean` |
| `createBatch_bound` | Creating a batch raises the id boundary once per id that was not recycled | `FalkorReplication.lean` |
| `memo_refines_groupBy`, `emit_refines` | The real `slots`/`index`/`last` grouping loop is the `groupBy` this proof uses | `FalkorReplication.lean` |
| `intern_refines`, `applyRec_addLabel_refines`, `idsOf_refines`, `createOne_refines`, `deleteOne_refines` | The abstract steps equal faithful copies of the engine's (`intern`, `apply_add_schema`, `IdSpace::reserve`, `IdSpace::create`/`release`) | `FalkorReplication.lean`, `Faithful.lean` |
| `onFailure_client_noop`, `onFailure_replicated_resyncs`, `onFailure_loading_exits`, `forceFullResync_exit_iff`, `clip_spec` | `divergence_guard.rs`, line by line | `Guard.lean` |

### How `emit_apply` is proved

1. **Model the graph.** `GState` holds the label and attribute dictionaries (append-only lists; an id is the index), the id boundary `bound`, the recycle bin `free`, and labels and property values per node. `Val` is just `null` or a number: values travel as literals and are never evaluated, so their contents do not matter.
2. **Model the write.** `Pending` holds the created nodes (`NodeSpec`: id, labels, attributes), the deleted nodes, and the newly registered names. `commit` applies them in the master's order. `emit` builds the records the way `for_each_record` does: schema additions, then one `CREATE_NODE` per shape group (`groupBy NodeSpec.shape`, rows from `gatherRows`), then one `DELETE_NODE` per label-set group.
3. **Model the replica.** `applyRec` applies one record with the replica's checks (`okCreate`, `okDelete`, the name re-derivation), and `applyPayload` applies them all and then runs `Batch.verify`.
4. **Split the payload.** `applySchemaLabels` and `applySchemaAttrs` handle the schema records, `applyCreateGroups` the create groups, `applyDeleteGroups` the delete groups. Each says the checks pass and what state results.
5. **Undo the regrouping.** `groupBy_flat` says the groups together are a permutation of the original list. `createBatch_perm` and `deleteBatch_perm` then say the replica's order gives the master's result. `specsOf_group` says each record's rows reach the right ids.
6. **Get the boundary right.** `createBatch_bound` was where a first draft failed. Raising `bound` once per id *at or above the old boundary* agrees with the engine for one contiguous batch, but not once records are grouped: a later record can carry an id below a boundary an earlier one pushed up. The engine's boundary is `live + recycled.len()`, so reusing a recycled id leaves it alone, and the model now does the same.

## Bugs found

This project found no bugs in the Rust. It did find a bug in its own first draft (the boundary, step 6 above).

One open bug sits exactly in its assumptions: **W2-effects-1**, a write that only registers a name or only cancels entities ships no payload (see [`../effects_emit_apply`](../effects_emit_apply)). The model here has no cancelled pairs and assumes every modifying write produces a payload, so `emit_apply` neither proves nor refutes it.

## Files

| File | What's in it |
|---|---|
| `FalkorReplication.lean` | The report header, the whole model (`GState`, `Pending`, `commit`, `emit`, `Record`, `applyRec`, `Batch.verify`), `emit_apply`, the stream and state-machine theorems, the `#guard` sanity checks, and the refinement theorems to `Faithful.lean` and `effects_emit_apply` |
| `Faithful.lean` | Faithful copies of the engine steps the model abstracts: `intern`, `verify_id`, `apply_add_schema`, `IdSpace::reserve` on a fresh batch, and the `live`/`recycled` half of `IdSpace` |
| `Guard.lean` | `src/divergence_guard.rs`: `is_replayed`, `on_failure`, `force_full_resync`, `master_address`, `log_payload`, `clip` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |
| `todo.tsv` | Empty |

The lakefile requires [`../effects_emit_apply`](../effects_emit_apply) as a path dependency, so the grouping loop `FalkorEA.memoGroup` is imported rather than copied.

## Coverage

`COVERAGE.tsv` has 60 rows: `emit.rs` (16), `apply.rs` (17), `format.rs` (8), `id_space.rs` (6), `divergence_guard.rs` (6), `graph.rs` (4), `commit.rs` (2), `pending.rs` (1).

| Bucket | Rows |
|---|---|
| PROVEN | 57 |
| MODELLED | 2 |
| NOT COVERED | 1 |

- **MODELLED:** `Pending::commit` and `CommitOp::next` are the abstract `commit` here. Their faithful models are PROVEN in `pending_commit`; the emit gate in `CommitOp::next` is covered by `effects_emit_apply` (`commitShips_eq_iff`).
- **NOT COVERED:** `CommitOp::new`, an operator constructor, is outside this abstract model (PROVEN in `pending_commit`).

`bash proofs/coverage.sh` counts each function once, in its best bucket across projects, so these three are PROVEN repo-wide.

## What the model does not cover

- **Only nodes and schema.** `Record` here has schema registration, node create and node delete. Edges, `UPDATE_NODE`/`UPDATE_EDGE`, `SET_LABELS`/`REMOVE_LABELS`, index and constraint DDL, and `digest_cancelled`'s create/delete pairs are not modelled. The liveness checks #3022 added to those arms are proven separately in `effects_emit_apply` (`apply_wf`); on the arms modelled here they change nothing.
- **Cancelled entities and schema-only writes.** `PendingWF.disjoint` assumes them away, and assumption 5 in the header assumes every modifying write produces a payload. W2-effects-1 shows that assumption is false today for writes whose only effect is a cancelled pair or a new name.
- **Bytes.** This model works on `Record`, not bytes. The byte codec is proven in `effects_codec` (`readRecord_encRecord_wf`); the two are not linked by a theorem. (The header's "what is still assumed" item 1 predates that project.)
- **Transport.** Redis replication delivers payloads in order; `GRAPH.EFFECT` is atomic because the graph is rolled back on any error. Both are taken as given.
- **The master's own id check.** Since #2846 the master also verifies its id space at commit. That it never refuses a well-formed write is not proven here (the id-space half is in `id_space`).
- **A replica that attaches mid-query.** Whether to build effects is decided when the query starts; such a replica is covered by its full sync, not by this model.
- **Faithfulness of `emit` and `applyRec` to the Rust.** The refinement theorems cover the steps listed above; the rest is a correspondence to review against the header table.

## Build and check

```bash
cd proofs/replication && lake build    # this project (builds ../effects_emit_apply too)
bash proofs/lean_ci.sh replication     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `effects/v3/emit.rs`, `effects/v3/apply.rs`, `effects/v3/format.rs`, `id_space.rs`'s reserve/create/release/verify, or `src/divergence_guard.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A change to how the emitter groups records, or to what the replica checks, will usually break `emit_apply` first. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
