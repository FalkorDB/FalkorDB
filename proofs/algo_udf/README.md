# algo_udf: the `algo.*` procedures and JavaScript UDFs

This project covers two parts of the engine that users call directly:

- **the graph algorithms** in `graph/src/runtime/functions/algo_procedures.rs`: `algo.BFS`, WCC, CDLP, `pageRank`, betweenness, HarmonicCentrality, MSF, `maxFlow`, and the path searches `algo.SPpaths` / `algo.SSpaths`;
- **the JavaScript UDF layer** in `graph/src/udf/`: converting values between FalkorDB and QuickJS, the `graph`/`node` objects a UDF sees (`graph.traverse`, `node.getNeighbors`, …), library loading and the function registry.

For the path searches it proves the answers are right: the shortest path really is shortest, and the enumeration finds every qualifying path. For the UDF layer it proves values survive the trip into JavaScript and back, and that deep or cyclic values give an error instead of a crash. Along the way it found a dozen places where Rust answers differently from C.

The full Lean↔Rust mapping is in the header of [`AlgoUdf.lean`](AlgoUdf.lean) and in the table at the top of each file. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does `algo.SPpaths` return a shortest path?**

Yes, for non-negative weights. With `pathCount: 1` and no cost tie-break, `run_path_algo` takes a Dijkstra fast path (`plan_dijkstra_iff` says exactly when). `dijkstra_correct` proves that when the search finishes:

- following the parent chain never panics;
- if it reports a path, the path is a real walk from source to target, its weight is no larger than any other walk's, and its reported cost is the sum of its edges' costs;
- it reports no path only when no walk exists.

`dijkstra_terminates` proves it finishes.

**2. Does the general path enumeration find every qualifying path, and only those?**

Yes. The general case runs a breadth-first pre-pass for a weight bound, then a depth-first enumeration:

- `bfs_sound`, `bfs_complete`: the pre-pass finds a simple path of at most `maxLen` hops exactly when one exists;
- `dfs_sound`: every path the enumeration keeps is a simple path from the source (to the target, for SPpaths) with 1 to `maxLen` hops, within `maxCost`, with the correct weight and cost sums;
- `dfs_complete`: it terminates and records every qualifying path whose weight is within the final bound;
- `dfs_k1` and `dfs_k0`: with `pathCount: 1` the answer is the minimum over *all* qualifying paths; with `pathCount: 0` it is exactly the set of minimum-weight paths;
- `bfs_seed_keeps_answer_k1`, `bfs_seed_keeps_answer_k0`, `bfs_none_sound`: the BFS seed and the "no path" short-cut lose nothing.

For `pathCount ≥ 2`, `dfs_complete` assumes the bound only tightens as the search goes on (`BoundMono`); this is proven for 0 and 1.

**3. Do values survive a round trip through a UDF?**

Yes, for every value C round-trips, since #3074. Take an identity UDF `function id(x) { return x; }`; then `RETURN mylib.id(v)` returns `v`:

- integers, nodes, relationships, points, datetimes: `int_roundtrip`, `node_rt`, `rel_rt`, `point_rt`, `datetime_rt`;
- floats: `float_rt_iff`, and `-0.0` stays a float since #3074 (`neg_zero_rt`);
- vectors, including ones holding ±inf since #3074: `vecf32_rt`, `vecf32_inf_rt`;
- maps with a key named `constructor`: `constructor_key_roundtrips`. Before #3074, `{constructor: {name: 'Date'}}` was taken for a JS `Date` and failed.

A few changes are shared with C and are not bugs: an integral float comes back as an integer (`integral_float_becomes_int`), a `__proto__` key is dropped (`proto_key_dropped`), and array-index keys come first (`index_keys_first`).

**4. Can a deep or cyclic value crash the server through a UDF?**

Not since #3336 (security advisory GHSA-frmf-ffvc-mxx5). Before it, `value_to_js`/`js_to_value` recursed with no limit, so a UDF returning a self-containing array (`a = [a]`) overflowed the stack (`pre3336_self_array_overflows`). Now both directions count depth against `MAX_UDF_VALUE_NESTING = 1024`:

- `fromJs_frames`, `toJs_frames`: at most cap + 1 levels are ever used, for any input, cyclic ones included;
- `fromJs_past_cap`, `toJs_past_cap`: past the cap the result is an error, never an overflow;
- `fromJs_eq_pre3336`, `toJs_eq_pre3336`: below the cap the result is exactly what the old code returned.

The JavaScript side is modelled over a QuickJS heap, so a value that contains itself is a legal input to these theorems.

**5. Do the algorithm results name the right nodes?**

Mostly. The algorithms run on a compact adjacency matrix (index 0..n) and map results back to node ids (`compactEdges_sound`, `harmonic_correct_iff`). Two places get the mapping wrong; see [Bugs found](#bugs-found):

- WCC and CDLP with `nodeLabels` report a compact index as the component id (`wcc_compact_component_id_not_member`);
- HarmonicCentrality sizes its vector by node count instead of the id bound (`unfiltered_drops_high_ids`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `dijkstra_correct` | The Dijkstra fast path returns a minimum-weight path with the right cost, or none only when no path exists, and never panics | `Dijkstra.lean` |
| `dijkstra_terminates` | Dijkstra always finishes | `Dijkstra.lean` |
| `bfs_sound`, `bfs_complete` | The BFS pre-pass finds a path within `maxLen` hops exactly when one exists | `BfsBound.lean` |
| `dfs_sound`, `dfs_complete`, `dfs_k1`, `dfs_k0` | The DFS enumeration keeps only qualifying paths, finds all of them, and picks the right ones for `pathCount` 0 and 1 | `DfsSound.lean`, `DfsComplete.lean`, `DfsResult.lean` |
| `plan_dijkstra_iff`, `bfs_seed_keeps_answer_k1` | `run_path_algo`'s fast path and seeded bound lose nothing | `PathDispatch.lean`, `PathSeed.lean` |
| `lifecycle_frees_once` | The LAGraph adjacency handle is freed exactly once on every path, success or failure | `AlgoGlue.lean` |
| `fromJs_frames`, `toJs_frames`, `fromJs_past_cap`, `toJs_past_cap` | UDF value conversion is depth-bounded in both directions; past the cap it errors | `MarshalCap.lean` |
| `float_rt_iff`, `neg_zero_rt`, `vecf32_inf_rt`, `constructor_key_roundtrips` | Values round-trip through JavaScript | `MarshalThm.lean`, `Marshal.lean` |
| `traverse_nodes_nodup`, `traverse_both_edges_nodup` | `graph.traverse` never lists a node twice, nor an edge twice in `both` mode | `UdfTraverse.lean` |
| `deserialize_atomic`, `serialize_roundtrip` | The UDF repository saves and loads as a unit | `UdfRepoOps.lean` |
| `validate_runtime_same_globals` | Since #3079 the validation context and the runtime context define the same globals, `graph` included | `UdfGlobals.lean` |

### How `dijkstra_correct` is proved

1. **Model the graph.** `Graph.G` gives, for each node, its relationships as `(src, dst, edge id)` and, for each edge, a weight and a cost. The relationship list is any finite list (an AXIOMATISED accessor). Weights and costs are exact non-negative naturals; an edge whose weight is not finite is skipped, as in Rust. `Walk g src es tgt W` says `es` is a sequence of edges from `src` to `tgt` with total weight `W`.
2. **Model the function.** `St` holds Rust's `labels` map (best known weight and parent per node), the `settled` set, the heap, and the `reached` flag. `Step` is one loop turn: pop a superseded entry and skip it, pop the target and stop, or settle a node and relax its edges. `finish` is the result assembly; it returns `none` where Rust's `labels[&cur]` would panic.
3. **The heap pops a minimum.** The model lets a step pop *any* minimum-weight entry. `itemCmp_max_min_weight` proves that `BinaryHeap::pop` under `DijkstraItem::cmp` returns such an entry, so every theorem covers the real pop order.
4. **Invariant.** `Inv` is the textbook Dijkstra invariant, plus a proof-only "settle time" per node: every label is its parent's distance plus one edge (so it is the weight of a real walk), each parent was settled before its child, and a settled node's distance is no larger than any walk to it (its `opt` field). `inv_init` starts it, and `inv_settle` and its siblings prove each step keeps it.
5. **Read the answer off.** At a terminal state, either the target is settled (so walking back the parents gives a minimum walk, `walkBack_ok`), or the heap is empty and no walk reaches the target.
6. **Hypothesis `src ≠ tgt`.** `run_path_algo` never calls Dijkstra with equal endpoints (`dijkstra_never_src_eq_dst`), so this hypothesis costs nothing.

## Bugs found

Every bug was confirmed against live Rust and C servers (see [`../FINDINGS.md`](../FINDINGS.md)). The Rust repro tests (`graph/tests/lean_algo_udf.rs` and a standalone wave-5 test) are not committed; each `bug_*` test asserts C's answer and fails today.

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `wcc_compact_component_id_not_member` | WCC and CDLP with `nodeLabels` return a compact index as the component id, not a node id | **Open** |
| 2 | `unfiltered_drops_high_ids` | HarmonicCentrality sizes its source vector by node count, not the id bound, so nodes with high ids (after deletes) are dropped | **Open** |
| 3 | `pagerank_unknown_label_compact_empty` | `pageRank` with an unknown label errors | **Open** |
| 4 | none (cargo test `bug_betweenness_unknown_label_is_empty_not_error`) | betweenness and HarmonicCentrality with an unknown `nodeLabels` entry return no rows; C errors "unknown label" | **Open** |
| 5 | `pre3113_cost_default_diverges` | SPpaths/SSpaths charged a missing cost as 0; C charges 1 per edge, so `maxCost` bounds hops | Fixed by #3113 (`cost_agree`, `cost_default_agrees`); the Dijkstra fast path still reports 0, as C's does (`dijkstra_cost_no_prop`) |
| 6 | `forged_node_marker`, `forged_edge_marker` | A UDF result object carrying a forged `__falkor_type` marker is trusted, which panics the reply or creates a dangling edge (W3-algo-1) | **Open** |
| 7 | `pre3074_neg_zero_not_preserved` | `-0.0` came back from a UDF as the integer 0 | Fixed by #3074 (`neg_zero_rt`) |
| 8 | `pre3074_constructor_key_breaks_roundtrip`, `pre3074_constructor_regexp_becomes_string` | A map with a `constructor` key was taken for a `Date` or a `RegExp` | Fixed by #3074 |
| 9 | `pre3074_vecf32_inf_not_roundtrip` | A vector holding ±inf could not pass through a UDF | Fixed by #3074 (`vecf32_inf_rt`) |
| 10 | `pre3079_validate_lacks_graph` | The validation context at `GRAPH.UDF LOAD` had no `graph` global | Fixed by #3079 (`graph_global_in_both`) |
| 11 | `pre3336_self_array_overflows`, `pre3336_nestArr_overflows` | Unbounded recursion in UDF value conversion overflowed the stack | Fixed by #3336, GHSA-frmf-ffvc-mxx5 (`fromJs_frames`, `toJs_frames`) |
| 12 | `collision_accepted`, `dotted_collision`, `delete_breaks_other`, `case_folded` | Library and function names can collide after qualification or case folding, and deleting one library can break another | **Open** |
| 13 | `self_loop_twice` | `node.getNeighbors()` lists a self-loop's neighbour twice (W4-graph-2) | **Open** (#3083; PR #3084) |
| 14 | `traverse_self_loop_twice` | `graph.traverse([a], {returnType:'edges'})` returns a self-loop twice: Rust `[1,0,1]`, C `[1,0]` (W5-udf-1) | **Open** |
| 15 | `traverse_drops_self_neighbour` | `graph.traverse([a])` leaves `a` out when it is its own neighbour: Rust `[1]`, C `[0,1]` (W5-udf-2) | **Open** |
| 16 | `getNeighbors_no_arg_errors` | `node.getNeighbors()` with no argument throws "Error converting from js 'undefined'"; C returns the neighbours (W5-udf-3) | **Open** |
| 17 | `empty_lib_name_mismatch` | `GRAPH.UDF LOAD ""` is accepted, but its functions can never be called (registered as `.f`, stored as `f`); C refuses the empty name (W5-udf-4) | **Open** |

Also found by this project's work, but outside the model: SHUTDOWN after a main-thread UDF aborts, because the QuickJS runtime is freed before its contexts (W3-algo-2, **open**).

Model findings, not confirmed live:

- `cap_asymmetry_vecf32`: the two conversion directions count a vector or path leaf differently toward the nesting cap, so 1024 lists around a vecf32 pass into JavaScript but an identity UDF cannot return them. An error, not a crash.
- `keepMin_nan_not_comm`: a NaN edge weight breaks the MSF tie-break's laws, so which of two parallel edges wins could depend on GraphBLAS thread order.
- `non_string_types_disable_filter`: `getNeighbors({types: [123]})` silently turns the type filter off (C crashes on this input).
- `seed0_samples_deleted`: betweenness may sample deleted node slots on the unfiltered path.

Shared with C, not reported: `maxIterations`, `samplingSize` and `maxLen` wrap after the positivity check (`maxIterations_wraps_to_zero`, `samplingSize_wraps`, `maxLen_wraps`), and SPpaths with negative weights can miss a path (outside the model; C does the same).

## Files

| File | What's in it |
|---|---|
| `AlgoUdf.lean` | Root: the report header (Lean↔Rust table, assumptions, theorems, bugs, gaps) |
| `Value.lean` | A cut-down `Value`: only the variants the `algo.*` argument parsers inspect |
| `Config.lean` | Procedure config parsing: `nodeLabels`, `relTypes`, `maxIterations`, `samplingSize`, `maxLen`, `pathCount` |
| `Compact.lean` | Compact adjacency id↔index maps; WCC, CDLP, pageRank, Harmonic result assembly; betweenness sampling |
| `Paths.lean` | `far_endpoint`, `cmp_found_path`, `record_found_path`, cost sums, the Dijkstra dispatch guard |
| `MsfFlow.lean` | MSF's min-score monoid and component keys; maxFlow alignment |
| `Graph.lean` | The graph view for the path searches: `Edge`, weighted `Walk`, hop walks, `node_successors` |
| `DijkstraModel.lean`, `DijkstraInv.lean`, `Dijkstra.lean` | `dijkstra_single_path`: model, invariant preservation, `dijkstra_correct` |
| `BfsModel.lean`, `BfsBound.lean` | `bfs_find_bound` and its soundness/completeness |
| `Dfs.lean`, `DfsSound.lean`, `DfsComplete.lean`, `DfsExplore.lean`, `DfsResult.lean` | `enumerate_paths` with the bound: model, soundness, completeness, termination, `pathCount` results |
| `PathDispatch.lean`, `PathSeed.lean` | `parse_sp_config`/`parse_ss_config`, `run_path_algo`'s fast path and seeding |
| `AlgoGlue.lean` | Messages, MSF scoring callbacks, node collection, LAGraph handle ownership |
| `AlgoBfsReg.lean` | `algo.BFS` result assembly and procedure registration |
| `Marshal.lean`, `MarshalThm.lean` | `value_to_js`/`js_to_value` over an abstract f64 and QuickJS heap; the round-trip theorems |
| `Nesting.lean`, `MarshalCap.lean` | #3336: generic depth-bounded recursion, instantiated for both conversion directions |
| `Repo.lean`, `UdfRepoOps.lean` | The UDF repository: load, delete, registry, serialise/deserialise |
| `Neighbors.lean` | `collect_edges`, `getNeighbors` filters |
| `UdfTraverse.lean` | `graph.traverse` |
| `UdfClasses.lean` | The JS `graph`/`node`/`edge`/`path` objects and native entry points |
| `UdfContext.lean` | Error messages, timeouts, script validation, context rebuild, the call bridge |
| `UdfGlobals.lean` | The JS `falkor.register` bodies, globals, log strings |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

All `.lean` files except the root are under `AlgoUdf/`. The project has no Lake dependencies.

## Coverage

`COVERAGE.tsv` has 112 rows: `algo_procedures.rs` (61) and `graph/src/udf/` (51).

| Bucket | Rows |
|---|---|
| PROVEN | 108 |
| AXIOMATISED | 4 |

The four AXIOMATISED rows are FFI: `extract_vector_f64`, `extract_vector_i64` and `build_bool_pattern` wrap GraphBLAS calls, and `with_stack` is `stacker`'s stack switching, which is transparent in the model (one conversion level = one `with_stack` call).

## What the model does not cover

- **Floating point.** f64 is abstract: only `==`, `<`, `floor`, `abs < 2^53` and `is_finite` are modelled. Path weights and costs are exact non-negative naturals. Rounding, NaN and inf (which Rust filters with `is_finite`) and negative weights are outside the model; negative weights are a documented precondition in C too.
- **`pathCount ≥ 2`** completeness assumes `BoundMono`.
- **LAGraph** is trusted: WCC/CDLP output is a dense vector of indices below n, the MSF component vector is dense, and pageRank rejects n = 0. These are hypotheses, not Lean axioms.
- **QuickJS** behaviour is assumed: `__proto__` never becomes an own key, array-index keys list first, `JS_IsDate`/`JS_IsRegExp` test the internal class, `new Date(ms)` is NaN beyond ±8.64e15 ms.
- **`Graph::get_node_relationships`** is assumed to yield out-edges then in-edges, with a self-loop in both.
- **Timeouts and memory limits** inside the searches only add an error exit and are not modelled.
- **QuickJS runtime lifetime** (W3-algo-2) is not modelled.

## Build and check

```bash
cd proofs/algo_udf && lake build    # this project
bash proofs/lean_ci.sh algo_udf     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh             # repo-wide coverage report
```

**When you change `algo_procedures.rs` or anything under `graph/src/udf/`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

A fix for one of the open bugs should turn its counterexample into `pre<PR>_` history and add the theorem that now holds, as #3074 did. If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
