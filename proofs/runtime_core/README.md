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

Yes, today. The runtime computes a row budget, `record_cap`, by walking up from an operator to the nearest `LIMIT` and adding any `SKIP`. `CondTraverse` and `ExpandInto` treat that budget as a *hard* stop. But the walk passes through other traversals, and a traversal can turn one row into zero rows. Here is an example:

```cypher
// a1 -> b1 (dead end), a2 -> b2 -> c
MATCH (a:A)-[:R]->(b)-[:R]->(c) RETURN c.id LIMIT 1
// Rust: []    C: [30]
```

The lower traversal stops after one row (`b1`). The upper traversal finds nothing from `b1`, so the answer is lost. `hard_cap_unsound_condTraverse` and `hard_cap_unsound_expandInto` prove this on the model with exactly these inputs. `hard_cap_unsound_through_traverse` proves the general form: for every `LIMIT`, some expansion makes the cap drop rows.

The project also proves what a correct budget looks like. `need` is the smallest sound budget: a `LIMIT n` bounds it, a `SKIP s` adds `s`, a 1:1 `Project` passes it through, and anything else (including a traversal) is a barrier. `need_sound` proves that cutting the stream to `need` rows never changes the result. `recordCap_hardCap_sound_oneSkip` proves that today's `record_cap` *is* sound when only `Project`s and at most one `SKIP` sit between the operator and the `LIMIT`.

**2. Does the emitter lose, duplicate or reorder rows?**

No. `drive_flatten` and `fresh_drive` prove that the batches an emitter-driven operator yields, concatenated, are exactly the per-row expansions of every input row, in order, **for any initial `record_cap`**. For the emitter, the budget is only a batch-size hint, which is why the same `record_cap` is harmless there and harmful as a hard stop in traversals. The pack ceiling stays in `[1, 1024]` (`ceiling_bounds`), the emitter makes progress whenever work is left (`loop_progress`), and `reset` drops all queued work (`reset_remaining`).

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
| `recordCap_hardCap_sound_oneSkip` | Today's `record_cap` is a sound hard cap when only Projects and at most one SKIP are below the LIMIT | `Budget.lean` |
| `hard_cap_unsound_expandInto`, `hard_cap_unsound_condTraverse`, `hard_cap_unsound_stacked_skip` | Counterexamples: through a traversal, or with stacked SKIPs, the hard cap loses the answer | `Budget.lean` |
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

### How `need_sound` is proved (and why `record_cap` fails it)

1. **Model the plan above an operator.** `Anc` is one ancestor, nearest parent first: `limit n`, `skip n`, `project g` (1:1), `condTraverse f` or `expandInto f` (one row to any list of rows, possibly empty), or `other h` (any barrier). `run` applies a list of ancestors to a row stream.
2. **Model the Rust.** `effectiveLimit`, `effectiveSkip` and `recordCap` are `Runtime::effective_limit`, `effective_skip` and `record_cap`, walk for walk. `hardCap` is the `produced >= cap` stop in `CondTraverseOp` and `ExpandIntoOp`, and SortOp's top-k heap.
3. **Induct on the ancestor list.** `need_sound` follows `need`'s definition. A `limit` at the head is `take n` of `take n`; a `skip s` turns a budget of `c` above into `c + s` below; a `project` commutes with `take`. A traversal or barrier makes `need` return `none`, which means no cap.
4. **Compare.** `hardCap_sound_of_need_le` shows that any cap at least `need` is also sound, and `recordCap_sound_oneSkip` shows that `record_cap ≥ need` under the one-skip condition. The counterexamples show where that condition fails: `record_cap` walks *through* `condTraverse`/`expandInto` where `need` stops, and `effective_skip` keeps only the *first* `SKIP`.

`Limit` and `Skip` are stored as `Nat`. That loses nothing: a negative or non-integer value makes `build_batch_op` fail the whole query before anything runs.

## Bugs found

Each bug was compared against C. The live repro for B1/B2 is `repro_limit.py PORT` (the Rust test file `graph/tests/lean_runtime_core.rs` no longer exists on main). See [`../FINDINGS.md`](../FINDINGS.md), wave 3.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| B1 (W3-runtime-1) | `hard_cap_unsound_expandInto`, `hard_cap_unsound_condTraverse`, `hard_cap_unsound_through_traverse` | `record_cap` walks through `CondTraverse`/`ExpandInto` (`runtime.rs:699-702`), which then use it as a hard stop. `MATCH (a:A)-[:R]->(b)-[:R]->(a) RETURN a.id LIMIT 1` returns `[]` (C: `[2]`) | **Open** (walk unchanged at 0f45a8685) |
| B2 (W3-runtime-2) | `hard_cap_unsound_stacked_skip` | `effective_skip` returns only the first `SKIP`. `MATCH (a:A)-[:R]->(b) WITH b SKIP 0 RETURN b.id SKIP 1 LIMIT 1` returns `[]` (C: `[20]`) | **Open**. The proposed fix for B1 and B2 is to give hard-cap consumers `need` and keep `record_cap` for the emitter |
| B3 | *(same shape as B1; no separate theorem)* | SortOp's top-k uses the same walk: `MATCH (a:A) WITH a ORDER BY a.id MATCH (a)-[:R]->(b) RETURN b.id LIMIT 1` returns `[]`; expected `[20]` | **Open**, shared with C. The stacked-SKIP Sort variant is #2659 |
| — | `Accessors.reuse_diverges` | With a node id present in both pending attribute maps, `n.a` falls through `new_nodes_attrs` to `existing_nodes_attrs` while `properties(n)` reads only `new_nodes_attrs`. Only reachable via deleted-id reuse (#2876), where `deleted_nodes` shadows both anyway | Suspected |
| — | `checkMem_wrap` | `usage_fn() as i64` wraps at 2⁶³, so the memory cap never fires past 8 EiB | Latent |

**CONDITIONAL rows (4).** `get_node_attribute`, `get_node_attribute_no_delete_check`, `get_node_attrs` and `get_relationship_attrs` are proven under two facts about the pending attribute maps: `hdisj` (a node is in at most one pending attribute map) and `huniq` (each staged list has distinct keys). `proofs/pending_commit` proves both for every reachable `Pending` state (`PendingCommit/AttrInv.lean` `PA.inv_hdisj`, `PA.inv_huniq`; `AccessorBridge.lean` instantiates these accessors' theorems with them). The rows stay CONDITIONAL because that proof assumes SET/REMOVE only stage attributes on a live entity, the deleted-entity guard `proofs/ops_apply` proves, and the two are not linked. The store contracts `hc`, `hbg` and `hName` are discharged in `Discharge.lean`.

## Files

| File | What's in it |
|---|---|
| `Budget.lean` | `effective_limit`, `effective_skip`, `record_cap`, the hard cap, the reference `need`; bugs B1–B3 |
| `Emitter.lean` | `RowIter`, `refill_from_cursor`, `drain_pending_entry`, `emit_lazy`, `seed`, `reset`, and the operator pull loop; `drive_flatten` |
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

`COVERAGE.tsv` has 94 rows: 53 in `runtime.rs`, 39 in `batched_result_emitter.rs` and 2 in `ops/mod.rs`.

| Bucket | Rows |
|---|---|
| PROVEN | 53 |
| CONDITIONAL | 9 |
| MODELLED | 7 |
| NOT COVERED | 25 |

- **CONDITIONAL:** the nine attribute accessors listed above.
- **MODELLED:** the `RowIter` constructors (`one`, `spread`, `many`), `start_batch`, `finish_batch` and the two emitter constructors. The model works on `(row, item)` pairs, so the column gather in these functions is abstracted away.
- **NOT COVERED:** the 24 `GatherItem` typed-lane methods (`new_lanes`/`push_into`/`finish` for each lane) and the trivial `batch` accessor. These lanes are proven once, generically, in `columnar` (`gather_spec`).

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

A change to which IR variants `effective_limit` or `effective_skip` walk through changes `Anc` and the budget theorems. Fixing B1/B2 should turn the `hard_cap_unsound_*` counterexamples into `pre<PR>_` history and prove the new budget against `need`. Fixing #2876 should discharge `hdisj` and move the attribute rows from CONDITIONAL to PROVEN.
