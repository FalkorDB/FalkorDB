# ops_apply: sub-queries, OPTIONAL MATCH, MERGE, FOREACH, UNION and the write operators

This project proves when the engine's batched operators that run a sub-plan per input row give the same answer as running it row by row, and finds the cases where they do not. Several user-facing features depend on it:

- `OPTIONAL MATCH`, `CALL { … }` sub-queries and pattern predicates (`WHERE (a)-->(b)`, `NOT`, `OR`), which are planned as Apply, Optional, SemiApply and OR-apply;
- `MERGE`;
- `FOREACH`;
- `UNION ALL` (and `UNION`, which is `UNION ALL` plus DISTINCT);
- `CREATE`, `DELETE`, `SET` and `REMOVE`;
- cartesian products and value hash joins (`MATCH (a:A), (b:B) WHERE a.v = b.v`).

openCypher defines these operators one input row at a time. For speed, the engine runs a sub-plan once over a whole batch of input rows, tags each output row with the input row it came from (its *origin*), and matches the outputs back to the inputs. This project proves when that shortcut is safe.

The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorOpsApply.lean`](FalkorOpsApply.lean) and in the table at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. When does running a sub-plan once per batch give the same rows as running it once per row?**

When the sub-plan is *separable*: running it on a tagged batch gives the same output as running it on each row alone and tagging the results with that row's origin. Scans, traversals, filters, projections and `UNWIND` are separable (`rowLocal_separable`), and so is any pipeline of separable operators (`compose_separable`). Aggregates, `ORDER BY`, `LIMIT`, `SKIP`, `DISTINCT`, cartesian products and value hash joins are not: they mix rows from different inputs.

Under that condition:

- batched Apply returns exactly the per-row rows, in the same order (`applyBatched_eq_ref`);
- batched Optional returns the same rows, but not in the same order (`optionalBatched_perm_ref`, see question 2);
- SemiApply and anti-SemiApply keep exactly the rows for which the pattern exists, or does not exist (`semiBatched_eq_ref`);
- the OR-apply multiplexer is the per-row disjunction (`orApply_eq_ref`).

The per-row fallback modes are correct for *any* sub-plan, separable or not (`applyPerRow_eq_ref`, `optionalPerRow_eq_ref`).

The planner decides whether to batch with a blacklist of non-separable operators. That blacklist misses `ValueHashJoin`, so the join is made per-origin (`CartVhj.vhjRowsPO`) instead of being excluded by the blacklist; `canBatchApply_admits_vhj` shows the blacklist admits it, which is why the join must be per-origin (`vhjPO_sepT`).

**2. Does OPTIONAL MATCH keep the input order?**

No, when batched. `optionalBatched_perm_ref` proves the rows are right, but all the `null` rows of a batch come after all the matched rows. `optionalBatched_order_differs` shows it: `UNWIND [3,1,2] AS x OPTIONAL MATCH (a:A {w:x}) RETURN collect(x)` gives `[1, 2, 3]` where C gives `[3, 1, 2]`. openCypher does not specify row order here, but `collect` makes the difference visible.

**3. Is MERGE correct?**

Without `SET` clauses, yes, for every batch size, on the non-all-bound path (`allBound = false`; the all-bound shortcut that keeps only the first match is outside `mainMerge_noSet_eq_ref`, and the condition that gates it is checked in `MergeEager.lean`: `allBound_path_keeps_all`). Since #3067 `MergeOp::next` matches every input row against the graph as it is before any create, then walks the rows in order; `mainMerge_noSet_eq_ref` proves it ends with the same graph and the same next id as the per-row openCypher meaning and emits exactly the reference's rows in the reference's order. `mainMerge_batch_independent` shows the result does not depend on the batch boundaries. This works because MERGE caches what earlier rows created, and the proof shows the cache exactly simulates "later rows see earlier rows' creations".

With `ON CREATE SET` / `ON MATCH SET`, or with more than one MERGE clause, the theorems above do not apply (see the open MERGE row #3368 below).

**4. Does FOREACH run its body exactly once per item?**

Yes. `run_ok` proves that FOREACH passes every input row through unchanged and in order, emits non-empty batches of at most `BATCH_SIZE` rows, and runs the body on every list item of every row exactly once, in order, in chunks of at most `BATCH_SIZE` (`calls_items`). A `null` list runs nothing. `foreach_correct` is the same statement from a fresh operator.

**5. Is UNION ALL a concatenation?**

Yes. `union_drain` proves `UnionOp` emits every branch's rows in branch order. If a branch fails to start, its error is emitted and the stream ends (`next_spec`, `next_none_stays`).

**6. Do CREATE, DELETE, SET and REMOVE write what they should?**

Yes:

- CREATE binds one fresh node per active row and logs it (`createNode_spec`); relationships take each row's endpoints and refuse deleted ones (`createRel_spec`, `createRel_gone`).
- DELETE leaves every bound node and relationship gone (`deleteBatch_vars`, `delNodesBulk_spec`, `delRelsBulk_spec`). The id-space unwinds can fail; `deleteBatchE_ok` proves the fallible model equals the infallible one when they succeed, and `delNodeE_refusedP`/`delNodeE_refusedC` prove a refusal becomes the operator's error.
- `SET e = src` leaves exactly the source's keys, for every target and source shape (`setFrom_replace`, `setAttrItem_replace_shapes`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `applyBatched_eq_ref` | Batched Apply = per-row Apply, for separable sub-plans | `Correlated.lean` |
| `optionalBatched_perm_ref` | Batched Optional returns the per-row rows, possibly reordered | `Correlated.lean` |
| `semiBatched_eq_ref`, `orApply_eq_ref` | Pattern predicates (`EXISTS`, `NOT`, `OR`) are the per-row answer | `Correlated.lean` |
| `applyPerRow_eq_ref`, `optionalPerRow_eq_ref` | The per-row fallbacks are right for every sub-plan | `Correlated.lean` |
| `rowLocal_separable`, `compose_separable` | Row-local operators, and pipelines of them, are separable | `Correlated.lean` |
| `mainMerge_noSet_eq_ref`, `mainMerge_batch_independent`, `fixed3067_batch_independent` | MERGE (all rows matched before any create) without SET matches openCypher in the same row order, and no result depends on the batching | `MergeEager.lean` |
| `run_ok`, `foreach_correct`, `calls_items` | FOREACH is a pass-through that runs its body once per item, in order | `ForEach.lean` |
| `union_drain` | UNION ALL is the concatenation of its branches | `Union.lean` |
| `deleteBatch_vars`, `deleteBatchE_ok` | Every deleted variable's entity is gone; the fallible version agrees on success | `Delete.lean`, `DeleteCancel.lean` |
| `createNode_spec`, `createRel_spec` | CREATE binds one fresh node per row; relationships use each row's endpoints | `CreateBatch.lean` |
| `setFrom_replace`, `setAttrItem_replace_shapes` | `SET e = src` leaves exactly the source's keys (all six shapes) | `SetRemove.lean` |
| `allTuples_length`, `orbit_valid` | A cartesian product has ∏ lengths rows and visits only valid combinations | `CartVhj.lean` |
| `applyBatchedW_eq_ref_of_readIndep` | Batched Apply over a writing sub-plan equals per-row Apply when the sub-plan's reads ignore its own writes (the condition a fix for #2738 must meet) | `RowWrites.lean` |
| `KeyFromBoundNodeOnly`, `MergeBindsOwnNode`, `bug3368_filed_repro_fixed_by_3050` | MERGE over a real row (values + bound bits): the pattern hash reads a value-only slot as the variable's value, the cache hit re-inserts another clause's ids; the fixed key, and #3368 | `MergeSlot.lean` |

### How `applyBatched_eq_ref` is proved

1. **Model the sub-plan.** A `SubPlan` is a function from a list of `(origin, row)` pairs to a list of `(origin, output)` pairs. This is `run_batch` plus `set_argument_batch` plus draining the iterator.
2. **Model the batched operator.** `tag` gives the compacted active rows sequential origins `0..n` (`clone_active_rows_seq_origin`). `applyBatched` runs the sub-plan once over the tagged batch and merges each output with the input row its origin points to (`merge_over_input`).
3. **Model the reference.** `applyRef` runs the sub-plan on each row alone (`single`) and merges the results with that row, in input order. That is the openCypher meaning, and what C does with its per-record argument.
4. **State the condition.** `Separable f` says `f` on a tagged batch equals the concatenation of `f` on each row alone, re-tagged.
5. **Prove by rewriting.** Rewrite `f (tag xs 0)` with `Separable`, then `lookup_tagged` shows that looking up each origin in the tagged input finds exactly the row that produced it.

The hypothesis `Separable` is the whole question. It is proven for row-local operators and their pipelines, and refuted for `count(*)` (`count_not_separable`); the `ValueHashJoin` is per-origin, `CartVhj.vhjRowsPO`, and separable on the batches Apply builds: `vhjPO_sepT`). For the real child operators of each plan it is not derived from their code; that is why these rows are CONDITIONAL on the `Adm` hypothesis (see [Coverage](#coverage)).

## Bugs found

Every bug was reproduced against a live server and compared with C before it was recorded (see [`../FINDINGS.md`](../FINDINGS.md)). The Rust repros are in `graph/tests/lean_ops_apply.rs`, kept locally and not committed; every `bug_*` test asserts C's answer and fails today.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W2-apply-5 | `optionalBatched_order_differs` | Batched Optional moves the `null` rows after the matched rows. Low severity: order is unspecified, but `collect` shows it | **Open** |
| #2738 | `bug2738_batched_call_hides_writes`, `canBatchApply_admits_write`; property `WriteSeparable` | A CALL subquery that writes is batched (`can_batch`, `apply.rs:119-133`, does not blacklist Create/Delete/Set/Commit), so its reads for every row of the batch see the graph from before the batch and one Commit flushes all rows' writes. `UNWIND [1,2,3] AS x CALL { MATCH (a:N) CREATE (:N) RETURN 1 AS one } RETURN count(*)` over one `:N`: Rust 3 (4 `:N` after), C 7 (8 after) | **Open** (#2738) |
| #2729 | `bug2729_batch_split_flips_endpoint_check`; property `PartitionInvariant` | `CreateOp` and `DeleteOp` stream one batch at a time against the shared pending state, so `CREATE (n0)-[:R]->(n0) DETACH DELETE n0` fails with "endpoint was not found" as soon as rows holding the same `n0` reach CREATE in two batches (`create.rs:259-266` sees the earlier batch's pending delete). The issue's query inside `CALL {}` gets a different plan and a different batch split; the same rows at top level arrive in one batch. Also without CALL: `UNWIND range(1,1100) AS x MATCH (n:A {id:1}) CREATE (n)-[:R]->(n) DETACH DELETE n` fails on Rust, works with 1000 rows; C works for both | **Open** (#2729) |
| #3164 | `bug3164_stale_ref_deletes_recycled_id`; property `NoStaleAlias` | Per-row Apply commits each row's sub-plan; a committed DETACH DELETE releases the node id and the next CREATE reclaims it (`id_space.rs:483-498`), while later outer rows still hold `Value::Node(id)`, so their DELETE removes the new node. `UNWIND [1,2] AS x MATCH (n:A) CALL { WITH n DETACH DELETE n RETURN 1 AS k UNION CREATE (:B) RETURN 2 AS k } RETURN x, id(n), k` on `(:A),(:C)`: Rust deletes 2 nodes and keeps one `:B`; C deletes 1 and keeps two | **Open** (#3164) |
| #3368 | `same_op_leftover_shares_entry`; properties `KeyFromBoundNodeOnly`, `MergeBindsOwnNode` | `compute_merge_pattern_hash` reads the MERGE variable's slot with `get_by_id` (`merge.rs:189`), bound or not, so a value-only leftover (a `count(*)` accumulator at the variable's id: before #3579, which fixed it: ops_aggregate `fixed3579_3368_merge_slot_free`) replaces the label and properties in the hash. Inside ONE operator, two rows with equal leftovers hash equal and the second binds the first's node (`same_op_leftover_shares_entry`, model only) | **Open** (model only; the same-operator form is unverified live) |

### CONDITIONAL rows

Six rows rest on one undischarged hypothesis about other Rust code: that the sub-plan the `can_batch` guard admits is an `OpSep.Adm` tree (the `cb = true -> Adm b f` premise of `applyNext_perm_ref`, `applyOpNext_perm_ref`, `optionalNext_perm_ref`; `Adm.dsep` gives separability for every tree built with `Adm`). The theorems are proofs for modelled `Adm` trees. `guard_admits_classified` only gives each IR kind the guard admits a verdict (`exact`, `permOnly`, a bug number, `mergeOnly`, `ddl`): it does not construct an `Adm` for them, and two verdicts have no `Adm` constructor at all, the admitted writing operators (#2738) and the anonymous-bidirectional `CondTraverse` dedup (#3106), kept as counterexample theorems. The planner-output to `Adm` correspondence is not machine-checked.

- `ApplyOp::next_batched` and `OptionalOp::next_batched`: the join that was the known counterexample (a ValueHashJoin) is per-origin. For every other sub-plan the planner lets through, the premise is believed true but not proven.
- `ApplyOp::next` and `OptionalOp::next`: the `can_batch` dispatch (`applyOpNext` models both modes of `ApplyOp`, plain and the Optional arm with `optional_vars`). The per-row branch is proven for every sub-plan; the batched branch carries the `Adm` premise.
- `SemiApplyOp::next` and `OrApplyMultiplexerOp::next`: the planner only feeds them pattern sub-plans, which should be `Adm`, but this is not proven.

## Files

| File | What's in it |
|---|---|
| `Correlated.lean` | The tagged-batch idiom, `Separable`, batched and per-row Apply, Optional, SemiApply and OR-apply, the ValueHashJoin counterexample and the `can_batch` guards |
| `Merge.lean` | MERGE on a single-node pattern: the per-row reference, the pattern cache keyed by `(operator idx, hash)`, `drain_pending` and the cache invariants |
| `ForEach.lean` | FOREACH: chunking, the pending queue, the body log, `run_ok` |
| `Union.lean` | UNION ALL as a lazily instantiated concatenation |
| `CreateAttrs.lean` | CREATE's attribute template, map attributes and label resolution |
| `CreateBatch.lean` | `create_batch`, `CreateOp::new` and `next` |
| `Delete.lean` | Every function of `delete.rs`, with id-space cancels assumed to succeed |
| `DeleteCancel.lean` | The fallible DELETE proven equal to `Delete.lean` on success |
| `SetRemove.lean` | Every function of `set.rs` and `remove.rs` over a committed + pending attribute store |
| `CartVhj.lean` | CartesianProduct (the odometer) and ValueHashJoin (hash table, integer promotion, matching) |
| `MergeSlot.lean` | MERGE over a real row (values + bound bits): the pattern hash reads a value-only slot as the variable's value, the cache hit re-inserts another clause's ids; `KeyFromBoundNodeOnly`, `MergeBindsOwnNode`, the fixed key, and #3368 |
| `RowWrites.lean` | Writing sub-plans under Apply (graph state, queued writes, Commit): batched vs per-row write visibility, the batch-split dependence of streaming CREATE/DELETE, and stale node references after id recycling (#2738, #2729, #3164) |
| `Ctors.lean` | Constructors: `MergeOp::new`, its `OnceCell` resolvers and pattern hash, SemiApply, OR-apply, FOREACH, `UnionOp::store_argument_batch` |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no dependencies beyond Lean core. The id-space `reserve` it calls is a parameter here and is proven in `proofs/id_space`.

## Coverage

`COVERAGE.tsv` has 77 rows:

| Bucket | Rows |
|---|---|
| PROVEN | 71 |
| CONDITIONAL | 6 |

- **CONDITIONAL:** `ApplyOp::next_batched`, `OptionalOp::next_batched`, `SemiApplyOp::next`, `OrApplyMultiplexerOp::next`, all resting on the `Adm` hypothesis above, and the dispatchers `ApplyOp::next` and `OptionalOp::next` that call `next_batched`.
- **Formerly MODELLED / AXIOMATISED, now PROVEN (`BatchRows`, `MergeSlot`):** `clone_active_rows_seq_origin` (`cloneSeq_spec`, `cloneSeq_tag` = `Correlated.tag`), `merge_over_input` (`mergeOverInput_row`: bound slots equal `Row::merge`'s; only a slot unbound on both sides differs), `origin_row`, `gather`, `BatchRow::to_owned_row` (`ofCells_slot`: one slot per column, trailing `Unbound` kept in scope), `BatchOp::set_argument_batch` (`slots_setArg`: every `Argument` leaf holds the batch, compacted below a join's right child; `per_origin = len > 1`).

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Write visibility** (`RowWrites.lean`) is modelled with an abstract graph state and commit; the concrete counterexamples use small graph models (a node count; `(id, serial)` pairs), not the full graph store.
- **Sub-plans are abstract functions.** Separability of each real operator is a hypothesis, except for the per-origin ValueHashJoin model (`vhjPO_sepT`).
- **MERGE is a single node pattern with one property.** The pattern hash is the key, so hash collisions (#2780) are assumed away. `ON MATCH`/`ON CREATE` are modelled only as updates of that key; relationship patterns and the null-property error are not modelled; the all-bound shortcut is a flag.
- **`MergeSlot`** models one node pattern, one row per MERGE, no SET; FxHash is taken as injective on the token stream (map key names dropped from the comparison); the aggregation that leaves the value-only slot is restated as the input row, not imported from ops_aggregate.
- **Output batch sizes** are modelled for FOREACH only. Apply, Optional and MERGE are compared on the flattened row stream.
- **Errors.** FOREACH and UNION have their error paths modelled, but only the error-free theorems are proven.
- **Attributes.** `SET`/`REMOVE` assume the committed key list of an entity is complete.
- **Seen, not Rust bugs.** `FOREACH (i IN 1 | …)` is a type error in Rust (C iterates once; Rust follows openCypher). C returns only the x=1 row of `UNWIND [1,2] AS x CALL { MERGE (n:P {v:1}) ON CREATE SET n.v = 2 RETURN n }`, and gives `m=1, c=2` for `UNWIND [1,2] AS x MERGE (a:E {v:1}) ON MATCH SET a.m = x ON CREATE SET a.c = x` (Rust `m=2, c=1`, as openCypher).
- **Also seen.** Creating and deleting a node in one query reports no `Nodes created/deleted` statistics (`CREATE (a:D) DELETE a`, also via MERGE); C reports 1/1 (pending/stats code, see `pending_commit`). Writes to a DETACH-deleted relationship are applied and counted (W2-pending-3). Hash-join keys Int/Float near 2^53: `keyAsI64_spec` (W2-rewrites-3).

## Build and check

```bash
(cd proofs/ops_apply && lake build)    # this project
bash proofs/lean_ci.sh ops_apply     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh              # repo-wide coverage report
```

**When you change `ops/apply.rs`, `optional.rs`, `semi_apply.rs`, `or_apply_multiplexer.rs`, `merge.rs`, `foreach.rs`, `union.rs`, `create.rs`, `delete.rs`, `set.rs`, `remove.rs`, `cartesian_product.rs` or `value_hash_join.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If you add an operator that can appear inside an Apply or Optional sub-plan, decide whether it is separable and, if not, add it to both `can_batch` blacklists. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
