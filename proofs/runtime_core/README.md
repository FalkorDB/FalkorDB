# runtime_core: LIMIT/SKIP budgets, the result emitter, plan building and entity reads

This project models the core of the query runtime: `graph/src/runtime/runtime.rs`, the helpers in `runtime/ops/mod.rs`, and `runtime/ops/batched_result_emitter.rs`. It answers four practical questions:

- when the runtime stops a traversal early because a `LIMIT` is coming, can rows be lost?
- does the emitter, which fans one input row out into many output rows, ever lose or reorder rows?
- does the iterative plan builder build the same operator tree as the recursive one it replaced?
- do the entity accessors (`n.k`, `properties(n)`, `labels(n)`, `n:L`, degrees) agree with one another, given this query's pending writes and deletes?

User-facing behaviour that depends on it:

- every query with `LIMIT` and/or `SKIP` above a `MATCH` traversal;
- traversals and `UNWIND`, which run through the emitter;
- the `resultset_size` config;
- property and label reads after `SET`, `REMOVE` and `DELETE` in the same query;
- `indegree`/`outdegree`;
- query timeouts and the memory cap;
- index `OPTIONS` validation (vector `dimension`/`similarityFunction`, fulltext keys).

The Lean↔Rust map is [`COVERAGE.tsv`](COVERAGE.tsv), and each file under `RuntimeCore/` starts with its own table. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Can `LIMIT` make a traversal drop rows that should be returned?**

No (since #3071). The runtime computes two row budgets with one walk, `row_budget`, which goes up from an operator to the nearest `LIMIT` and adds every `SKIP` on the way (saturating). `hard_record_cap` (`row_budget(false)`) stops at the first traversal, aggregate or any other barrier, and is the *hard* stop of `CondTraverse`, `ExpandInto` and `Sort`'s top-k. `record_cap` (`row_budget(true)`) walks through `CondTraverse`/`ExpandInto` and is only a batch-size hint (the emitter's pack ceiling, scan sizing). Before #3071 the hint was also the hard stop, so:

```cypher
// a1 -> b1 (dead end), a2 -> b2 -> c
MATCH (a:A)-[:R]->(b)-[:R]->(c) RETURN c.id LIMIT 1
// before #3071: []    C and now: [30]
```

The lower traversal stopped after one row (`b1`), and the upper one found nothing from `b1`. Now its hard cap is none (`hardRecordCap_traverse`, `hardRecordCap_twoHopChain`, `hardRecordCap_cycleChain`), and stacked `SKIP`s all count (`hardRecordCap_stackedSkips`).

The reference is `need`, the smallest sound budget: a `LIMIT n` bounds it, a `SKIP s` adds `s`, a 1:1 `Project` passes it through, and anything else (including a traversal) is a barrier. `need_sound` proves that cutting the stream to `need` rows never changes the result; `hardRecordCap_eq_need` proves `hard_record_cap` is exactly `need` (when it is below `usize::MAX`), and `hardRecordCap_sound` proves the hard stop sound for every ancestor chain. `recordCap_eq_hard_of_need` shows the hint equals the hard cap wherever `need` is defined.

**2. Does the emitter lose, duplicate or reorder rows?**

No. `drive_flatten` and `fresh_drive` prove that the batches an emitter-driven operator yields, concatenated, are exactly the per-row expansions of every input row, in order, **for any initial pack hint**. For the emitter the budget is only a batch-size hint, which is why the soft `record_cap` is harmless there. The pack ceiling stays in `[1, 1024]` (`ceiling_bounds`), doubles only after a non-empty batch (`emit_ceiling`, `emit_none_ceiling`), `lift_pack_ceiling` resets it to 1024 (`lift_spec`), the emitter makes progress whenever work is left (`loop_progress`), and `reset` drops all queued work (`reset_remaining`).

**3. Does `resultset_size` return the right rows, and do writes still run?**

Yes. With `result_set_size = n ≥ 0` the client gets exactly the first `n` rows the plan produces, and with a negative size it gets all of them (`query_rows`). A write query keeps pulling every batch even after the cap is reached, so all pending writes are committed (`query_write_pulls_all`).

**4. Do `n.k`, `properties(n).k`, `labels(n)` and `n:L` agree with each other?**

A read sees three layers, in order: the snapshot of an entity this query deleted, then this query's pending writes, then the committed graph. Under that order:

- `properties(n).k` equals `n.k` (`attrs_agree`);
- the bulk column read used by batched operators equals the per-row read (`materialize_eq`);
- `n:L` is exactly `L IN labels(n)`, with no assumption on the staged add/remove sets (`hasLabelId_iff`);
- degrees never underflow and count exactly the live committed edges plus the edges this query created (`degree_exact`).

Several of these rest on store contracts the project can't see; see the CONDITIONAL rows below.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `need_sound` | The reference budget (LIMIT bounds, SKIP adds, Project passes, anything else is a barrier) never changes results | `Budget.lean` |
| `hardRecordCap_sound`, `hardRecordCap_eq_need` | `hard_record_cap`, the hard stop of CondTraverse/ExpandInto/Sort, never changes results: it is `need` | `Budget.lean` |
| `recordCap_hardCap_sound_oneSkip`, `recordCap_eq_hard_of_need` | The soft `record_cap` equals the hard cap wherever `need` is defined (e.g. Projects and one SKIP below the LIMIT) | `Budget.lean` |
| `hardRecordCap_cycleChain`, `hardRecordCap_twoHopChain`, `hardRecordCap_stackedSkips` | The former counterexamples (through a traversal, stacked SKIPs) now get no cap or the full `limit + Σ skip` | `Budget.lean` |
| `drive_flatten`, `fresh_drive` | The emitter outputs exactly the per-row expansions in order, for any `record_cap` | `Emitter.lean` |
| `runBatch_eq_rec` | The iterative post-order `run_batch` builds the same tree, with the same first error, as the recursive build, for every plan | `RunBatch.lean` |
| `query_rows`, `query_write_pulls_all` | `resultset_size` keeps exactly the first `n` rows; a write query always drains the root operator (every batch plus the terminating `next()`), so a root `CommitOp` always runs | `Query.lean` |
| `graph_deref`, `graph_same_result`, `graph_mut_write_same` | #3436's `Runtime::graph()`, `graph_mut()`, `GraphRef`, `Deref` on an atomic_refcell borrow-word model: the rewrite changes no result and no borrow-word effect | `Cell.lean` |
| `read_plan_no_update`, `accessor_unchanged`, `materialize_unchanged` | A read plan has no updating operator (from `redis_layer`'s `RedisLayer.Ro`), so the accessor theorems transfer through `graph()` | `Reuse.lean` |
| `attrs_agree`, `materialize_eq` | `properties(n).k`, `n.k` and the bulk column read agree | `Accessors.lean` |
| `hasLabelId_iff`, `hasLabel_iff` | `n:L` is exactly `L ∈ labels(n)`, by id or by name | `Accessors.lean` |
| `degree_exact` | Degrees never underflow and count live edges exactly | `Accessors.lean` |
| `checkTimeout_spec`, `checkMem_spec` | The timeout fires iff now ≥ deadline; for usage < 2⁶³, the memory cap fires iff 0 < cap < usage | `Misc.lean` |
| `vector_requires_dimension`, `vector_requires_similarity`, `fulltextUnknown_none_iff` | Index OPTIONS validation as in C (#3094) | `Misc.lean` |

### How `need_sound` is proved (and why `hard_record_cap` meets it)

1. **Model the plan above an operator.** `Anc` is one ancestor, nearest parent first: `limit n`, `skip n`, `project g` (1:1), `condTraverse f` or `expandInto f` (one row to any list of rows, possibly empty), or `other h` (any barrier). `run` applies a list of ancestors to a row stream.
2. **Model the Rust.** `rowBudgetGo`/`rowBudget` is `Runtime::row_budget` walk for walk (saturating `SKIP` sum, `through_traverse` flag); `recordCap` and `hardRecordCap` are `record_cap` and `hard_record_cap`. `hardCap` is the `produced >= cap` stop in `CondTraverseOp` and `ExpandIntoOp`, and SortOp's top-k heap.
3. **Induct on the ancestor list.** `need_sound` follows `need`'s definition. A `limit` at the head is `take n` of `take n`; a `skip s` turns a budget of `c` above into `c + s` below; a `project` commutes with `take`. A traversal or barrier makes `need` return `none`, which means no cap.
4. **Compare.** `hardRecordCap_eq` shows `row_budget(false)` is `need` (saturated); `hardCap_sound_of_need_le` and the saturation bound (a stream never has more than `usize::MAX` rows) give `hardRecordCap_sound`.

`Limit` and `Skip` are stored as `Nat`. That loses nothing: a negative or non-integer value makes `build_batch_op` fail the whole query before anything runs.

## Bugs found

Each bug was compared against C. See [`../FINDINGS.md`](../FINDINGS.md), wave 3. (The LIMIT/SKIP budget bugs B1–B3 were fixed by #3071, which also deleted their `repro_limit.py`; git history keeps them.)

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| — | `Accessors.reuse_diverges` | With a node id present in both pending attribute maps, `n.a` falls through `new_nodes_attrs` to `existing_nodes_attrs` while `properties(n)` reads only `new_nodes_attrs`. Only reachable via deleted-id reuse (#2876), where `deleted_nodes` shadows both anyway | Suspected |
| — | `checkMem_wrap` | `usage_fn() as i64` wraps at 2⁶³, so the memory cap never fires past 8 EiB | Latent |

**CONDITIONAL rows (4).** `get_node_attribute`, `get_node_attribute_no_delete_check`, `get_node_attrs` and `get_relationship_attrs` are proven under two facts about the pending attribute maps: `hdisj` (a node is in at most one pending attribute map) and `huniq` (each staged list has distinct keys). `proofs/pending_commit` proves both for every reachable `Pending` state (`PendingCommit/AttrInv.lean` `PA.inv_hdisj`, `PA.inv_huniq`; `AccessorBridge.lean` instantiates these accessors' theorems with them). The rows stay CONDITIONAL because that proof assumes SET/REMOVE only stage attributes on a live entity, the deleted-entity guard `proofs/ops_apply` proves, and the two are not linked. The store contracts `hc`, `hbg` and `hName` are discharged in `Discharge.lean`.

## Files

| File | What's in it |
|---|---|
| `Budget.lean` | `row_budget`, `record_cap` (soft hint), `hard_record_cap` (hard stop), the reference `need` and the soundness proofs |
| `Emitter.lean` | `RowIter`, `refill_from_cursor`, `drain_pending_entry`, `emit_lazy`, `lift_pack_ceiling`, `seed`, `reset`, and the operator pull loop; `drive_flatten` |
| `Query.lean` | `Runtime::query`: the result-set cap and the write drain |
| `RunBatch.lean` | `run_batch`'s iterative post-order build against the recursive build; `children_to_recurse` |
| `Cell.lean` | The earlier `Arc<AtomicRefCell<Graph>>` borrow-word model (removed by #3683; kept because `pending_commit`'s `Hold` imports it). The current model is `QueryGraph.lean` |
| `Reuse.lean` | Accessor theorems reused through `graph()` for read plans (needs `redis_layer`) |
| `Accessors.lean` | Node attribute reads (deleted > pending > committed), the attr-id memo, labels and label staging, degrees |
| `RelAccessors.lean` | Relationship attribute reads, typed materialisation, `label_id`, endpoints and type, pending attribute writes |
| `Plan.lean` | `get_variables`, `get_return_names`, `inspect_batch`, `Runtime::new`, `default_batch`, `run_nested_plan`, `build_batch_op`'s child popping and checks |
| `Misc.lean` | `check_timeout`, `check_mem_capacity`, `drain_pending`, `edge_already_used`, index OPTIONS validation |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Its `lakefile.toml` requires [`../redis_layer`](../redis_layer) (for `Reuse.lean` only). `evaluate_id_filter` (`runtime.rs`) is proven in [`../optimizer_scan`](../optimizer_scan), and the emitter's typed gather lanes are proven in [`../columnar`](../columnar).

## Coverage

`COVERAGE.tsv` has 64 rows: 52 in `runtime.rs`, 11 in `batched_result_emitter.rs` and 1 in `ops/mod.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 60 |
| CONDITIONAL | 4 |

- **CONDITIONAL:** the four attribute accessors listed above (`get_node_attribute`, `get_node_attribute_no_delete_check`, `get_node_attrs`, `get_relationship_attrs`).
- **Not rows of this ledger:** the `GatherItem` typed-lane methods, which are proven once, generically, in `columnar` (`gather_spec`).

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Batch layout.** Batches are lists of active rows. `GatherItem` lanes, `gather` and selection vectors are abstracted here (and proven in `columnar`).
- **Emitter errors.** The emitter's per-row closure is pure: its `Err` short-circuit isn't modelled.
- **Operator construction.** `build_batch_op`'s per-variant constructor arguments are an abstract `mk`, so `runBatch_eq_rec` holds for every operator but says nothing about any one of them.
- **Profiling and indexes.** Profiling and inspection output, and index resync, aren't modelled.
- **Time.** `Instant` arithmetic is a plain `Nat`.
- **Graph stores.** The graph stores underneath the accessors are hypotheses (the CONDITIONAL rows), not models.

## Build and check

```bash
(cd proofs/runtime_core && lake build)    # this project
bash proofs/lean_ci.sh runtime_core     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                 # repo-wide coverage report
```

**When you change `runtime.rs`, `ops/mod.rs`, `batched_result_emitter.rs`, or the `record_cap` consumers (`cond_traverse.rs`, `expand_into.rs`, `sort.rs`):**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A change to which IR variants `row_budget` walks through (with or without `through_traverse`) changes `Anc` and the budget theorems. Fixing #2876 should discharge `hdisj` and move the attribute rows from CONDITIONAL to PROVEN.
