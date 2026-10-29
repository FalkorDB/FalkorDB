# ops_traverse: how MATCH walks the graph

This project proves that the operators which match patterns return the rows openCypher defines, and finds where they do not. Several user-facing features depend on it:

- single-hop patterns, `(a)-[r]->(b)` and `(a)-[r]-(b)` (CondTraverse, ExpandInto);
- variable-length patterns, `(a)-[*1..3]->(b)`, including `*0`;
- `allShortestPaths(…)` and named paths (`p = …`, PathBuilder);
- node scans: by label, by id, by label and id range, and index, fulltext and vector scans on nodes and edges;
- relationship uniqueness: inside one MATCH, two relationship variables never bind the same edge.

The operators work on GraphBLAS matrices, merge rows in batches, and take shortcuts when an edge variable is never read. Each shortcut is a place where the result can drift from the pattern's meaning.

The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`OpsTraverse.lean`](OpsTraverse.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does a single hop return exactly the matching edges?**

Yes, when the edge is emitted (the edge variable is read):

- **directed:** a row `(from, to, e)` comes out iff `e` is an edge `from→to`, it fits the row's bound endpoints, and no sibling relationship of the same MATCH already uses it (`emit_directed_iff`);
- **undirected:** every edge matches in its stored direction and, unless it is a self-loop, in reverse too, so a self-loop matches once (`emit_bidir_iff`); ExpandInto, used when both endpoints are bound, does the same (`mem_expandInto_emit`);
- **two hops:** `(a)-[r]->(x)<-[s]-(c)` returns exactly the openCypher match set with `r ≠ s` (`twoHop_emit_iff`).

When the edge is not read, the operator *collapses* parallel edges into one row per node pair. That only ever returns rows emit mode would (`collapse_sub_emit`), and one per pair for a directed hop (`collapse_directed_iff`, `collapse_directed_keys_nodup`). This is C's behaviour, not openCypher's: `MATCH (a)-[]->(b) RETURN count(*)` counts pairs, not edges.

**2. Is the batched matrix-multiply path the same as row-by-row?**

Yes, for any matrix multiply that meets the GraphBLAS spec. For anonymous hops the engine multiplies a frontier matrix `F` by the relationship matrix instead of walking row by row. `batched_eq_collapse` proves the result equals the per-row collapsed expansion, with no duplicate rows. `fused_chain_semantics` proves a chain of such hops returns `(row, dst)` iff a walk through the chain exists.

The spec is the hypothesis `MxmSpec`: `(i, j)` is in `F·A` iff some `k` has `(i, k)` in `F` and `(k, j)` in `A`, with no duplicates. It is the only FFI fact the traversal proofs use, and `mxmRef_spec` shows a reference multiply satisfies it.

**3. Does a variable-length pattern return exactly the trails it should?**

Yes. `varLen_iff` proves `(a)-[*min..max]->(b)` returns exactly the trails from the start (walks that never reuse an edge) whose length is between `min` and `max`. `*0..0` returns just the start node (`zero_zero`). An undirected self-loop is used once, not twice as in C (`undirected_self_loop_once`).

Variable-length hops do not check sibling relationships for uniqueness (`no_sibling_uniqueness`); C does the same.

**4. Does `allShortestPaths` return all the shortest paths, and only those?**

Yes, between two different nodes:

- every returned path is a walk from source to destination (`asp_sound`);
- no returned path is longer than any other walk between them (`asp_minimal`);
- every shortest walk of at most `max_hops` edges is returned (`asp_complete'`);
- no path is returned twice (`asp_nodup`);
- the BFS always finishes, within node-count + 1 iterations (`fin_drained`).

When source and destination are the same node (a cycle) it is wrong; see bug 5.

**5. Do scans and LIMIT lose or duplicate rows?**

No. The `next` loops that pack expanded rows into output batches emit every row once, in order (`drive_flatten`), and a LIMIT pushed into them keeps exactly the first rows (`capDrive_flatten`). The label-and-id-range scan returns exactly the labelled nodes in the range (`labelIdScan_correct`). An index scan is a sound pre-filter under the RediSearch contract (`scanRow_spec`), and an undirected edge-index scan reads each edge both ways, a self-loop once (`orient_length`; fixed by #3081).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `emit_directed_iff`, `emit_bidir_iff` | A single hop returns exactly the matching edges, directed or undirected; a self-loop matches once | `CondTraverse.lean` |
| `twoHop_emit_iff` | Two chained hops return the openCypher match set with distinct relationships | `CondTraverse.lean` |
| `batched_eq_collapse`, `fused_chain_semantics` | The matrix-multiply path equals per-row expansion, for any multiply meeting the GraphBLAS spec | `CondTraverse.lean` |
| `varLen_iff` | Variable-length returns exactly the trails with length in `[min, max]` | `VarLen.lean` |
| `asp_sound`, `asp_minimal`, `asp_complete'`, `asp_nodup` | allShortestPaths (source ≠ destination) returns every shortest path, once, and nothing else | `ShortestPaths.lean`, `ASPMain.lean`, `ASPFuel.lean` |
| `fin_drained` | The allShortestPaths BFS terminates within node-count + 1 steps | `ASPFuel.lean` |
| `buildNodes_walk` | PathBuilder's "other endpoint" rule rebuilds a walk's nodes, given unique edge ids | `ShortestPaths.lean` |
| `revWalk_reconstructs`, `vltFix_follows` | PathBuilder's reversed var-length branch rebuilds any walk backwards; with the pattern direction as input it always follows the pattern | `PathVlt.lean` |
| `expandIntoFix_live` | ExpandInto that also rejects pending-deleted endpoints only matches live edges | `ExpandIntoPending.lean` |
| `drive_flatten`, `capDrive_flatten` | Output batching never loses, duplicates or reorders rows; LIMIT is exact | `Drive.lean` |
| `run_complete` | The variable-length frame machine yields every frame's items | `VarLenGlue.lean` |
| `labelIdScan_correct` | The label + id-range scan returns exactly labelled ∩ range | `Scans.lean` |
| `canUse_inListV_eq_or`, `canUseE_inListV_eq_or` | #2383: an `IN` list reaches the index on the same terms as the `Or` of `Equal`s it replaced (non-empty, every member indexable), so `IN [NULL]` still falls back | `IndexScan.lean` |
| `scanRow_spec`, `edgeRow_sound`, `orient_length` | Index scans are sound pre-filters, honour bound endpoints, and read undirected edges both ways | `IndexScan.lean` |
| `dedupRows_keys_nodup`, `dedupRows_keys_complete` | The bidirectional dedup keeps unique keys and never loses a key (the bugs are in *which* key) | `BidirDedup.lean` |
| `edgeVerdict_eq_filterRow` | A variable-length edge filter treats a non-boolean value exactly as `WHERE` does (since #2390) | `VarLen.lean` |

### How `asp_minimal` is proved

1. **Model the graph.** A `Graph` is a list of edges, each with an id, a source and a destination. `Graph.WF` says edge ids are unique, which the relationship tensor guarantees. `step` is `get_node_relationships_by_type`: the edges leaving a node, in both directions for an undirected pattern.
2. **Model the BFS.** `bstep` is one `pop_front` iteration of the loop in `all_shortest_paths.rs`. It keeps a distance per node, a FIFO queue and, for each node, its predecessor edges. `back` is the backtrack from the destination that turns predecessors into paths.
3. **State the invariant.** `Inv` says: every recorded distance is the true shortest distance from the source (`IsDist`); the queue is duplicate-free and holds at most two consecutive distance levels; every recorded predecessor is a real edge from a node one level closer to the source; and a node is only recorded as a predecessor after it has left the queue, so predecessor sets are closed.
4. **Show every step keeps it.** `inv_bstep` proves one iteration keeps `Inv`, and `inv_bfs` lifts that to the whole run.
5. **Read off minimality.** A returned path follows predecessor edges, each one level closer to the source, so its length is the destination's recorded distance. By the invariant that distance is the true shortest distance, so no walk is shorter.

The hypotheses are `src ≠ dst` (the cycle case has a known bug and is not claimed), `1 ≤ max_hops`, and `Graph.WF`. The last holds for every real graph.

## Bugs found

Every bug was reproduced against a live server and compared with C before it was recorded (see [`../FINDINGS.md`](../FINDINGS.md)). The Rust repros are in `graph/tests/lean_ops_traverse.rs`, kept locally and not committed.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| W2-traverse-1 | `twoHop_collapse_changes_count` | An unreferenced named edge is collapsed even though a sibling traverse still reads it for uniqueness (`reduce_expand_into.rs`; the oracle is now `optimizer/references.rs`). `MATCH (a)-[r]->(x)<-[s]-(c) RETURN count(*)` on `a⇉b→c`: Rust 1, C 2. Related to #2896 | **Open** |
| W2-traverse-2 | `drops_outer_rows`, `three_hops_collide`, `batch_split_changes_result` | The dedup for chained anonymous undirected hops (`cond_traverse.rs`) ignores other columns, keys on an intermediate node at 3+ hops, and resets per batch. `UNWIND [1,2] AS x MATCH (a)-[]-()-[]-(b)`: Rust 5 rows, C 10; a 3-hop star: Rust 7, C 12 | **Open** |
| W2-traverse-3 | `bidir_collapse_duplicates_pair`, `expandInto_bidir_collapse_two_rows` | An undirected anonymous hop collapses per direction, not per pair. With `a→b`, `b→a`, `a→b`, `MATCH (a)-[]-(b) RETURN a.id, b.id`: Rust 4 rows, C 2 | **Open** |
| W2-binder-1 | none here (see `proofs/binder`) | Consecutive MATCH clauses are merged into one pattern by the parser, so relationship uniqueness spans both. `MATCH (a)-[r]->(b) MATCH (c)-[q]->(b)`: Rust 2 rows, C 4 | **Open** |
| W2-traverse-4 | `directed_cycle_backtrack_order`, `directed_cycle_not_a_walk` | allShortestPaths on a directed cycle returns the edge list backwards (`all_shortest_paths.rs:284-288`): on `0→1→2→0`, `(a)-[*]->(a)` gives edges `[3,2,1]` instead of `[1,2,3]`. C is also wrong | **Open** (shared with C) |
| W2-traverse-5 | none (Cypher repro only) | `algo.SPpaths`/`algo.SSpaths` charged cost 0 per relationship without `costProp`; C charges 1 | Fixed by #3113 (the single-path shortcut still charges 0, as C does) |
| #3372 | `bug3372_slot_reuse_drops_rows`, `bug3372_split_product_zero`, `bug3372_aggregate_leaks_preprojection_slot`, `bug3372_shift_probes` | After `WITH c, count(*) AS k`, Aggregate re-binds `c` at its pre-projection slot (`aggregate.rs:1187-1196`); the binder numbers each scope from 0, so the next MATCH's fresh `d` gets the same slot and CondTraverse treats it as already bound to `c` (`cond_traverse.rs:648-652`, `:771-777`). The issue's `sum(in2 * out2)` is 0 (expected 4); the property a fix must establish is `ScopedRow` (`scoped_inputRow` shows it suffices) | **Open** |
| #3451 | `bug3451_main_violates`, `bug3451_cycle_not_reversed` | PathBuilder infers a var-length segment's direction by comparing its first node with the previous path node (`path_builder.rs:102-110`); on a cycle they are equal, so an incoming `(a)<-[:R*3..3]-(a)` on `0→1→2→0` comes out `[0,1,2,0]`/`[1,2,3]` instead of `[0,2,1,0]`/`[3,2,1]` (C: `[0,2,1,0]`/`[1,2,3]`). Property `PathFollowsPattern`; main is right on non-cyclic walks (`vltMain_follows_of_ne`), the plan-time direction flag fixes it (`vltFix_follows`) | **Open** |
| #3452 | `bug3452_main_violates`, `bug3452_matches_doomed_edge` | ExpandInto skips pending-deleted edges but never checks pending-deleted endpoints (`expand_into.rs:197-228`), and `DETACH DELETE` defers a node's edges to commit: `MATCH (a:A),(b:B) DETACH DELETE a MERGE (a)-[r:R]->(b) RETURN count(r)` gives 1 with nothing created (C creates a dangling edge). Property `LiveExpandInto`; fix `expandIntoFix_live` | **Open** |
| W2-strlist-6 | none (out of target; see `proofs/functions_str_list`) | `[x IN null \| x]` returned `[]`; C returns `null` | Fixed by #2954 (`b6567cb97`, closes #2928) |

### Shared with C, not filed as Rust bugs

- Anonymous edges, and named edges that are never read, collapse parallel edges.
- No relationship uniqueness between variable-length hops and sibling edges (`no_sibling_uniqueness`), nor for anonymous edges or named paths.
- An undirected allShortestPaths cycle reuses its first edge (`undirected_cycle_repeats_edge`).
- Where Rust is right and C is wrong: undirected self-loops under `*` (C reports them twice), uniqueness of named edges within one MATCH (C ignores it), and the allShortestPaths undirected self-loop (C duplicates it).

## Files

| File | What's in it |
|---|---|
| `Basic.lean` | The graph model: edges, the pair-level matrix pattern `pairs`, `edgesBetween`, `step` |
| `CondTraverse.lean` | Single hop (emit and collapse, directed and undirected), ExpandInto, two hops, `MxmSpec` and the batched multiply path; the collapse counterexamples |
| `UnionStream.lean` | `UnionIter` (#3492): the k-way merge of per-type pair streams, `drain_spec`, `union_stream`, the layer-merge order (`layerStream_spec`) |
| `UnionGlue.lean` | Multi-type matrix selection, `delta_lmxm_into` (`deltaInto_eff`), the iterator builders, `expand_union_row` (`union_emit_*_iff`, `union_collapse_directed_keys_nodup`) |
| `CTGlue.lean` | CondTraverse/ExpandInto constructors, `build_state`, `trim_to_cap`, null padding, `next`; scan and PathBuilder constructors |
| `BidirDedup.lean` | The bidirectional dedup block and its three counterexamples |
| `VarLen.lean` | The variable-length DFS, `varLen_iff`, the absorbed edge filter |
| `VarLenGlue.lean` | `path_value`, the `VarLenIter` frame machine, the endpoint plan, `new` |
| `ShortestPaths.lean` | allShortestPaths BFS and backtrack, soundness, the cycle counterexamples, PathBuilder |
| `ASPFold.lean`, `ASPInv.lean`, `ASPStep.lean`, `ASPFinal.lean`, `ASPMain.lean`, `ASPFuel.lean` | The allShortestPaths proof: relaxation, the invariant `Inv`, one BFS step, the backtrack, minimality/completeness/no duplicates, termination |
| `Scans.lean` | Label scan, id seek, label + id-range scan |
| `IndexScan.lean` | Node and edge index scans under the `IndexSound` contract, fulltext and vector scans, undirected edge orientation |
| `PathVlt.lean` | PathBuilder's `Value::Path` branch (splicing a var-length segment, reversed for `<-`); property `PathFollowsPattern` and the #3451 counterexample |
| `ExpandIntoPending.lean` | ExpandInto with pending deletes; property `LiveExpandInto` and the #3452 counterexample |
| `SlotFeed.lean` | How an Aggregate output row feeds a later CondTraverse across `WITH`: binder slot ids, the Aggregate row, the endpoint reads; property `ScopedRow` and the #3372 counterexamples |
| `Drive.lean` | The emitter-driven `next` loops and LIMIT propagation |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no dependencies beyond Lean core.

## Coverage

Line numbers are pinned to main 5e475e6c9. No traversal operator changed since fb3d516d0; #2383 changed the index scans (a list now reaches the index as one `InList`, `canUse_inListV_eq_or`), #3436 only replaced `g.borrow()` with `graph()`, and #2899 / #2954 shifted the `eval.rs` rows.

`COVERAGE.tsv` has 77 rows:

| Bucket | Rows |
|---|---|
| PROVEN | 62 |
| NOT COVERED | 23 |

Every operator file under `runtime/ops/` in scope is fully PROVEN (no FFI rows are left since #3492: `delta_lmxm_into` and the iterator builders are modelled). The rest:

- **`cond_traverse.rs` since #3492:** `delta_lmxm_into` is proved (`UnionGlue.deltaInto_eff`: `F * (A1 + .. + Ak)` on the pattern, via `GBWDelta.deltaLmxm_eff`); the iterator builders and the `UnionIter` merge (`UnionStream.drain_spec`) are modelled and proved. The batched path's one-hop theorems still take `MxmSpec` for the multiply.
- **NOT COVERED, `algo_procedures.rs` (18):** the `algo.SPpaths`/`algo.SSpaths` procedures (Dijkstra over `f64` weights, config parsing, path enumeration, registration). They are only differential-tested against C.
- **NOT COVERED, `eval.rs` (5):** `shortestPath()` is a bidirectional BFS in the expression evaluator. It is differential-tested against C; its stopping rule is argued in comments, not formalised.

`bash proofs/coverage.sh` counts each function once, in its best bucket across all projects.

## What the model does not cover

- **Labels and WHERE edge filters** are pure filters and are abstracted away; relationship types are a pre-filtered edge list. A variable-length walk still prunes on its edge's own attributes; the model treats that like the WHERE filter.
- **ExpandInto's `synthetic_label` branch** (`expand_into.rs:155-166`) reads committed labels only and is not modelled: no valid Cypher reaches it with pending label changes (#3452). Pending *deletes* are modelled (`ExpandIntoPending.lean`).
- **Emission order.** Iterator `seek` is modelled as a filter of the full pair list, and the order rows come out in (stack vs recursion, `swap_remove`) is not modelled. The theorems are about which rows come out and how many.
- **`get_nodes`** (label-matrix intersection) is assumed to return ids in ascending order.
- **allShortestPaths cycles.** Minimality and completeness are proven for source ≠ destination only.
- **`shortestPath()` and the `algo.*` path procedures** are tested against C, not proven.
- **RediSearch.** Index scans are proven under `IndexSound` (the index returns exactly the label's nodes that satisfy the query), which is an assumption about RediSearch.
- **Batch packing.** Record caps, OPTIONAL null padding and the emitter loops are modelled at row level; how rows are packed into columnar batches is proven in `proofs/columnar`.

## Build and check

```bash
cd proofs/ops_traverse && lake build    # this project
bash proofs/lean_ci.sh ops_traverse     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                 # repo-wide coverage report
```

**When you change `cond_traverse.rs`, `expand_into.rs`, `cond_var_len_traverse.rs`, `all_shortest_paths.rs`, `path_builder.rs`, the `node_by_*`/`edge_by_*` scans, or the optimizer rules that decide when an edge collapses:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
