/-
# runtime_core: Lean 4 model of `graph/src/runtime/runtime.rs`, `runtime/ops/mod.rs` and
# `runtime/ops/batched_result_emitter.rs`

Build: `lake build` (core Lean 4.34, no Mathlib). No `sorry`/`admit`/`axiom`: the
contracts of the stores underneath (attribute/label name tables, attribute store batch
read, `lookup_sorted`) are explicit theorem hypotheses. Coverage: `COVERAGE.tsv`.
Rust repros: `graph/tests/lean_runtime_core.rs`
(`cargo test -p graph --test lean_runtime_core`); `bug_*` tests assert the correct
answer and FAIL today, `agrees_*` pass.

| here | there |
| --- | --- |
| `Budget.effectiveLimit/effectiveSkip/recordCap` | `Runtime::effective_limit` (`runtime.rs:575`), `effective_skip` (`:610`), `record_cap` (`:642`) |
| `Budget.hardCap`, `Budget.run` | the hard `record_cap` stop of `CondTraverseOp` (`ops/cond_traverse.rs:1030,1179`), `ExpandIntoOp` (`ops/expand_into.rs:246`), SortOp top-k (`ops/sort.rs:491`) |
| `Emitter.*`   | `BatchedResultEmitter` (`ops/batched_result_emitter.rs:455-785`) |
| `Query.query` | `Runtime::query` result-set cap + write drain (`runtime.rs:514-561`) |
| `RunBatch.runBatch`, `childrenToRecurse` | `Runtime::run_batch` (`runtime.rs:710`), `children_to_recurse` (`:652`) |
| `Accessors.*` | entity accessors `runtime.rs:1371-1872` (attrs, memo, labels, degrees, endpoints/type) |
| `Misc.*`      | `check_timeout` (`:468`), `check_mem_capacity` (`:478`), `map_to_index_options` (`:1881`; #3094 required vector keys, fulltext unknown-key refusal), `drain_pending` / `edge_already_used` (`ops/mod.rs:127,155`) |
| (reused)      | `evaluate_id_filter` (`:1310`) is proven in `proofs/optimizer_scan` (`OptimizerScan.IdSeek.fold_some`, `go_eq_fold`, `seek_sound`) |

## Proven (plain English)
- Budget: the reference budget `need` (Skip adds, Project passes, anything else is a
  barrier) is sound; any larger cap is sound; `record_cap ≥ need` — hence a sound hard
  cap — whenever only Projects and at most one Skip sit below the Limit.
- Budget (bugs): `record_cap` is unsound as a hard cap through `ExpandInto`,
  through `CondTraverse` (for every limit), and with stacked Skips (concrete witnesses).
- Emitter: `RowIter.next` spec; `refill_from_cursor` and `drain_pending_entry` specs; the
  `emit_lazy` loop invariant (packed ++ remaining = remaining before); each batch ≤
  ceiling ≤ BATCH_SIZE; progress (never `None` while work is left); ceiling stays in
  [1, 1024] and doubles; `reset` drops all queued work; headline `drive_flatten` /
  `fresh_drive`: an emitter-driven operator yields exactly the per-row expansions of every
  input row in order, for any `record_cap` (the soft cap cannot change results).
- Query: `result_set_size = n ≥ 0` returns exactly the first `n` rows, negative returns
  all; `keep` never underflows; write queries always pull every batch.
- RunBatch: the iterative post-order build equals the recursive build (same tree, same
  first error) for every tree and every `build_batch_op`; the final pop never fails.
- Accessors: bulk `materialize_node_property_values` = per-row `get_node_attribute` on both
  paths; `properties(n).k` agrees with `n.k` (deleted snapshot > pending > committed; Null
  ≡ absent) under the store contracts; attr-id memo returns the table's answer, stays
  valid and ≤ 32 entries, and survives table growth; `n:L` (`node_has_label_id`) ≡ `L ∈
  labels(n)` with no disjointness assumption; `node_has_label` by name ≡ name ∈
  labels(n); label staging keeps add/remove disjoint and last-clause-wins; degrees never
  underflow and count live committed + created edges.
- Misc: timeout fires iff now ≥ deadline; mem cap fires iff 0 < cap < usage (usage <
  2^63); `drain_pending` moves exactly min(|q|, 1024 - |b|) rows in order;
  `edge_already_used` ≡ ∃ other sibling bound to this edge; vector index options refuse
  every negative integer and, since #3094 (fe619ac5f, fixes #3091), require `dimension` and
  `similarityFunction` and store the latter lower-cased (`vector_requires_dimension`,
  `vector_requires_similarity`, `vector_similarity_lowered`); fulltext refuses the first key
  outside `FULLTEXT_OPTIONS` (`fulltextUnknown_none_iff`); phonetic yields only "dm:en" or "".

## CONFIRMED bugs (Rust repro in `graph/tests/lean_runtime_core.rs`, C = `bin/macos-arm64v8-release/falkordb.so`)
B1. `record_cap` walks through `CondTraverse`/`ExpandInto` (`runtime.rs:593-596`), but
    `CondTraverseOp`/`ExpandIntoOp` treat it as a HARD stop (`cond_traverse.rs:1088`,
    `expand_into.rs:246`). A traverse feeding another row-reducing traverse stops after
    LIMIT rows and the answer is lost.
    `MATCH (a:A)-[:R]->(b)-[:R]->(a) RETURN a.id LIMIT 1` (a1→b1, a2⇄b2): Rust [] , C [2].
    `MATCH (a:A)-[:R]->(b)-[:R]->(c) RETURN c.id LIMIT 1` (a1→b1 dead end, a2→b2→c): Rust [], C [30].
    `... RETURN c.id SKIP 1 LIMIT 1` over 3 chains: Rust [], C [200].
    Tests `bug_limit_through_expand_into_drops_rows`, `bug_limit_through_two_hops_drops_rows`,
    `bug_skip_limit_through_two_hops_drops_rows`. Lean `hard_cap_unsound_expandInto`,
    `hard_cap_unsound_condTraverse`, `hard_cap_unsound_through_traverse`.
B2. `effective_skip` returns only the FIRST Skip (`runtime.rs:617-625`), so stacked Skips
    under one Limit under-budget the hard cap:
    `MATCH (a:A)-[:R]->(b) WITH b SKIP 0 RETURN b.id SKIP 1 LIMIT 1`: Rust [], C [20].
    Test `bug_stacked_skip_under_limit_drops_rows`; Lean `hard_cap_unsound_stacked_skip`.
    Fix for B1+B2: give hard-cap consumers `need` (`Budget.need_sound`): Skip adds, Project
    passes, CondTraverse/ExpandInto are barriers; keep the current hint for the soft
    emitter consumers (proven harmless there: `Emitter.drive_flatten`).
B3. (shared with C, openCypher violation) SortOp's top-k uses the same walk through
    CondTraverse: `MATCH (a:A) WITH a ORDER BY a.id MATCH (a)-[:R]->(b) RETURN b.id LIMIT 1`
    → Rust [], C [] ; expected [20]. Test `bug_sort_topk_through_traverse_drops_rows`.
    The stacked-Skip Sort variant is issue #2659 (also in C).
Also seen (known): `Labels added` counts new schema labels (`runtime.rs:555`) — #2655;
deleted-id reuse breaks the one-map assumption of `attrs_agree` — #2876
(`Accessors.reuse_diverges`).

## Suspected, unconfirmed
- `Accessors.reuse_diverges`: with a node id present in both pending attribute maps,
  `n.a` falls through `new_nodes_attrs` to `existing_nodes_attrs` while
  `properties(n)` reads only `new_nodes_attrs` (`pending.rs:493-500` vs `:509-512`).
  Only reachable via id reuse (#2876), where `deleted_nodes` shadows both anyway.
- `check_mem_capacity`: `usage_fn() as i64` wraps at 2^63 (`Misc.checkMem_wrap`); latent.

## Gaps
Columnar representation (`GatherItem` lanes, `gather`, selection vectors) is abstracted
to row lists; the emitter closure's `Err` path; `build_batch_op` per-variant constructor
arguments (abstract `mk`); `get_variables`/`get_return_names`; profiling/inspect;
index resync; relationship accessors are the node proofs' shape but not separately
instantiated; `Instant` arithmetic.

## Wave-5 additions
* `RelAccessors`: relationship reads, pending attribute writes (`setPendingAttr_spec`), typed
  materialisation, `label_id`, endpoints/type lookups (deleted > pending > committed).
* `Plan`: `get_variables` (BFS walk stopping at the first Project), `get_return_names`,
  `inspect_batch`, `Runtime::new`, delegating accessors, `default_batch`, `run_nested_plan`,
  `build_batch_op` child popping and SKIP/LIMIT/VHJ checks; `children_to_recurse` spec in RunBatch.
-/
import RuntimeCore.Budget
import RuntimeCore.Emitter
import RuntimeCore.Query
import RuntimeCore.RunBatch
import RuntimeCore.Accessors
import RuntimeCore.Misc
import RuntimeCore.RelAccessors
import RuntimeCore.Plan
