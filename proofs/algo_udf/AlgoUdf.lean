/-
# algo.* procedures and the JS UDF layer: models, proofs, and the bugs they found

Scope: `graph/src/runtime/functions/algo_procedures.rs` (BFS, WCC, CDLP, pageRank,
betweenness, HarmonicCentrality, MSF, maxFlow, SPpaths/SSpaths) and
`graph/src/udf/` (all files). Rust source of truth: origin/main 256ab4d3f.
Per-function buckets: `COVERAGE.tsv` (every function PROVEN except three
GraphBLAS FFI wrappers, AXIOMATISED). Repros: wave 1-4 in
`graph/tests/lean_algo_udf.rs`; wave 5 in the standalone package
a standalone Rust test (not committed) (built against origin/main with
`cargo test --test w5 -- --test-threads=1` on an origin/main checkout).
Every `bug_*` test asserts C's answer and fails today.

## Modules

| here | there |
| --- | --- |
| `AlgoUdf.Config`     | `opt_string`, `validate_config_map`, `extract_node_labels`/`extract_rel_types`, `parse_config`, samplingSize / maxIterations / maxLen / pathCount parsing |
| `AlgoUdf.Compact`    | `build_compact_adj_*` id ↔ index maps, WCC/CDLP/pageRank/Harmonic result assembly, deleted-id skipping, betweenness sampling |
| `AlgoUdf.Paths`      | `far_endpoint`, `cmp_found_path`, `record_found_path`, cost accumulation, `build_path_batch`, Dijkstra dispatch guard |
| `AlgoUdf.MsfFlow`    | `msf_keep_min_score` monoid laws, MSF component keys, maxFlow flow/edge alignment and super-node freshness |
| `AlgoUdf.Marshal`/`MarshalThm` | `value_to_js` / `js_to_value` over an abstract f64 and QuickJS value/heap model (cyclic values included) |
| `AlgoUdf.Nesting`/`MarshalCap` | the depth cap `MAX_UDF_VALUE_NESTING`, `value_to_js_at` / `js_to_value_at` with an explicit stack |
| `AlgoUdf.Repo`       | `UdfRepo::load`/`delete` + the case-sensitive function registry |
| `AlgoUdf.Neighbors`  | `collect_edges`, `js_get_neighbors` filters |
| `AlgoUdf.Graph`      | shared graph view for the path searches (`Edge`, weighted `Walk`, hop `HWalk`/`NWalk`, `node_successors`) |
| `AlgoUdf.DijkstraModel`/`DijkstraInv`/`Dijkstra` | `dijkstra_single_path`, `DijkstraItem::{cmp,partial_cmp}` |
| `AlgoUdf.BfsModel`/`BfsBound` | `bfs_find_bound` |
| `AlgoUdf.Dfs`/`DfsSound`/`DfsComplete`/`DfsExplore`/`DfsResult` | `enumerate_paths`, `record_found_path` (with the bound) |
| `AlgoUdf.PathDispatch`/`PathSeed` | `parse_sp_config`, `parse_ss_config`, `to_numeric_value`, `algo_sp_paths`, `algo_ss_paths`, `run_path_algo` |
| `AlgoUdf.AlgoGlue`   | `new_msg`, `msg_to_string`, `msf_score` + its GraphBLAS callbacks, `node_value`, `collect_node_ids`, `active_node_set`, LAGraph graph create/delete ownership, `cmp_found_path`, `parse_config` |
| `AlgoUdf.AlgoBfsReg` | `algo_bfs` result assembly, `register` and every `register_*` |
| `AlgoUdf.UdfTraverse`| `js_traverse_impl` (`graph.traverse`) |
| `AlgoUdf.UdfClasses` | `CURRENT_GRAPH` slot, `create_js_node/edge/path`, `getNodeById`, `iterateNodes/Edges`, the native entry points |
| `AlgoUdf.UdfNodeView` | which view (`Runtime` vs committed `Graph`) `create_js_node` reads labels/attributes from; `set_node_labels`, `set_node_attrs` |
| `AlgoUdf.UdfContext` | `caught_error_message`, timeouts, `validate_script`, `ensure_context_current`, `rebuild_context`, `call_udf_bridge` |
| `AlgoUdf.UdfGlobals` | the JS `falkor.register` bodies, `collect_*`, `js_value_to_log_string` |
| `AlgoUdf.UdfRepoOps` | `UdfRepo::{new,default,version,bump_version,flush,list,get_all_libraries,serialize,deserialize}`, `init_udf_repo`, `get_udf_repo` |

## Assumptions (AXIOMATISED as hypotheses/definitions, no Lean `axiom`)
* LAGraph: WCC/CDLP output is a dense vector of representative indices `< n`
  (`Compact.LagWcc`); `LAGraph_msf` component vector is dense
  (`MsfFlow.msf_dense_no_fallback` hypothesis); PageRank rejects n = 0.
* `Graph::get_node_relationships_by_type` is modelled (`Neighbors.nodeRels`), not assumed: outgoing half, then the incoming half minus self-loops (a self-loop once).
* QuickJS: `obj.set("__proto__", v)` never creates an own key; `obj.keys()`
  lists array-index keys first; `obj.get("constructor")` falls back to the
  prototype's constructor; `JS_IsDate` / `JS_IsRegExp`
  test the internal class (`Marshal.JsKind`); `Object.prototype.toString` gives "[object Object]";
  `new Date(ms)` is NaN beyond ±8.64e15 ms.
* f64 is abstracted (`Marshal.F64`, `MsfFlow.Score`): only `==`, `<`, `floor`,
  `abs < 2^53`, `is_finite` are modelled. Path weights are modelled as `Int`.

## Wave 5 main theorems (all proven, zero sorry; weights/costs exact naturals)
* `Dijkstra.dijkstra_correct` — at termination the parent walk never panics; the
  reported path is a walk source→target of the reported weight, no walk is
  lighter, the cost is its edge-cost sum; `None` only when no walk exists.
  `Dijkstra.dijkstra_terminates`; `itemCmp_max_min_weight` (heap pops a min weight).
* `BfsBound.bfs_sound` / `bfs_complete` / `bfs_src_eq_tgt` — the pre-pass finds a
  simple path of ≤ maxLen hops iff one exists; never panics.
* `Dfs.dfs_sound`, `dfs_no_panic` — every kept enumeration result is a simple path
  from the source (to the target for SPpaths) of 1..maxLen hops, within maxCost,
  with exact weight/cost sums; `explore` + `dfs_complete` — the loop terminates
  and records every qualifying path of weight ≤ the final bound;
  `dfs_k1` (pathCount 1: the `cmp_found_path` minimum over *all* qualifying
  paths) and `dfs_k0` (pathCount 0: exactly the minimum-weight paths).
* `PathDispatch.plan_dijkstra_iff`, `bfs_none_sound`, `PathSeed.bfs_seed_keeps_answer_k1/_k0`
  — `run_path_algo`'s fast path, empty short-cut and seeded bound lose nothing.
* `AlgoGlue.lifecycle_frees_once` — the LAGraph adjacency handle is freed exactly
  once on every create/teardown path (borrowed or owned, success or failure).
* `AlgoBfsReg.edgesBranch_aligned`, `branches_agree`, `register_empty`.
* `UdfTraverse.traverse_nodes_nodup`, `traverse_both_edges_nodup`;
  `UdfContext.ensureCurrent_version`, `rebuild_keys`, `bridge_clears`, `effTimeout_bounds`;
  `UdfGlobals.validate_names_nodup`, `runtime_keys_eq_qualified`;
  `UdfRepoOps.deserialize_atomic`, `serialize_roundtrip`, `list_shows_raw`.

## Wave 5 CONFIRMED bugs (live C 18900 / Rust 18901 and `rtest/tests/w5.rs`)
6. `graph.traverse([a])` omits `a` when it is its own neighbour (start pre-marked
   visited, js_classes.rs:744). C `[0,1]`, Rust `[1]` (Rust's `getNeighbors` gives
   `[0,1]`). `traverse_drops_self_neighbour`; `bug_udf_traverse_omits_self_loop_neighbour`.
8. `GRAPH.UDF LOAD "" …` is accepted (src/commands/udf.rs:117 checks only the max
   length) and its functions are uncallable ("UDF function '.f' not found in JS
   context": repository.rs:105 qualifies `.f`, js_globals.rs:125 stores `f`).
   C: "empty lib name". `UdfGlobals.empty_lib_name_mismatch`; `bug_udf_empty_library_name_accepted`.

## Wave 5 shared with C (live-checked)
* SPpaths with negative weights/costs: the BFS-seeded bound is not a bound on
  prefixes, so `(s)-[w:5]->(a)-[w:-5]->(t)` with `maxLen:5` (and a cost-5,-5 path
  plus a cost-0 detour with `maxCost:1`) return no row — C returns the same.
  Negative values are outside the Nat model (`PathDispatch` header).
* C's `graph.traverse` ignores `distance` and treats `'both'` as outgoing;
  Rust reads `maxDepth`. Not reported as bugs (C is the odd one out).

## Gaps (wave 5)
* `dfs_complete` for pathCount ≥ 2 takes bound monotonicity as a hypothesis
  (`BoundMono`; proven for 0/1). f64 rounding, NaN/inf (already filtered by the
  Rust `is_finite` checks) and negative weights are not modelled.
* QuickJS, LAGraph, GraphBLAS and graph accessors stay hypotheses / oracles.

## Main theorems of earlier waves (all proven, zero sorry)
* `Config.strList_ok`, `validate_ok`, `maxLen_small` — config parsing accepts exactly the documented shapes.
* `Compact.compactEdges_sound`, `unfiltered_component_ids_are_members`,
  `spec_component_is_member`, `wcc_rows_eq_spec`, `unfiltered_rows_exact`/`_nodup`, `harmonic_complete`.
* `Paths.far_*`, `recordK_length`, `record1_min`, `record0_ties`, `nextNode_far`,
  `dijkstra_never_src_eq_dst`.
* `MsfFlow.keepMin_comm`, `identity_neutral`, `flowAlign_aligned`, `superFresh`.
* `Marshal.int_roundtrip`, `float_rt_iff`, `node_rt`, `rel_rt`, `path_rt`, `point_rt`,
  `datetime_rt`, `vecf32_finite_rt`, `unesc_esc`, `keep_esc`, `map_never_marked`.
* `Repo.delete_preserves` (under the missing `KeysUnique` check).
* `Neighbors.collectEdges_mem`, `collectEdges_nodup` (every direction, self-loops once), `string_types_filter`.

## Model findings (not confirmed live)
* `cap_asymmetry_vecf32` — `value_to_js` counts only
  List/Map toward the cap, `js_to_value` also counts the array carrying a VecF32 (and the object carrying
  a Path), so `MAX` lists around a vecf32 (or path) convert into JS but an identity UDF cannot return them.

## Shared with C (live-checked, same answer on both) — not divergences
* maxIterations / samplingSize / maxLen are truncated by `as i32`/`as u32`
  after the positivity check (`Config.maxIterations_wraps_to_zero`, `samplingSize_wraps`, `maxLen_wraps`).
* SPpaths Dijkstra fast path ignores the cost/hops tie-break (`Paths.useDijkstra` doc).
* `__proto__` map key dropped (`Marshal.proto_key_dropped`); array-index keys reordered
  (`index_keys_first`); integral floats come back as Int (`integral_float_becomes_int`);
  all UDF libraries share one JS global scope (lib A's `var shared` is overwritten by lib B's).

## Suspicions (not confirmed)
* NaN edge weight breaks `msf_keep_min_score`'s monoid laws (`MsfFlow.keepMin_nan_not_comm`):
  winner of a multi-edge pair can depend on GraphBLAS thread order.
* getNeighbors `types: [123]` silently disables the type filter
  (`Neighbors.non_string_types_disable_filter`); C crashes on that input (C bug).
* betweenness samples deleted slots on the unfiltered path (`Compact.seed0_samples_deleted`).
* Datetime `ts * 1000` is unchecked i64 arithmetic (type_convert.rs:215, 224): the release profile sets no
  `overflow-checks`, so it wraps (`Marshal.wrapI64`, `toJs_datetime`); a debug build would panic.

## Gaps (earlier waves)
Closed in wave 5: Dijkstra, the DFS enumeration, the BFS bound, BFS parent→edge
reconstruction, js_context/js_globals glue and graph.traverse/iterate*.

## Open bugs the model missed (live-checked on main 9d3869aa8 vs C)
* #3380 `Betweenness`: seeded draw loop and the C reference modelled.
  `bug3380_default_size_16` (default 16, C 32; property `DefaultSizeMatchesC`),
  `bug3380_seedless_deterministic` (no clock seed; `SeedlessRunVaries`),
  `bug3380_seeded_fewer_sources` (20 requested on 40 nodes, seed 7 → 15; `SourcesExact`).
-/
import AlgoUdf.Value
import AlgoUdf.Config
import AlgoUdf.Compact
import AlgoUdf.Paths
import AlgoUdf.MsfFlow
import AlgoUdf.Marshal
import AlgoUdf.MarshalThm
import AlgoUdf.MarshalCap
import AlgoUdf.Repo
import AlgoUdf.Neighbors
import AlgoUdf.Graph
import AlgoUdf.DijkstraModel
import AlgoUdf.DijkstraInv
import AlgoUdf.Dijkstra
import AlgoUdf.BfsModel
import AlgoUdf.BfsBound
import AlgoUdf.Dfs
import AlgoUdf.DfsSound
import AlgoUdf.DfsComplete
import AlgoUdf.DfsExplore
import AlgoUdf.DfsResult
import AlgoUdf.PathDispatch
import AlgoUdf.PathSeed
import AlgoUdf.AlgoGlue
import AlgoUdf.AlgoBfsReg
import AlgoUdf.UdfTraverse
import AlgoUdf.UdfClasses
import AlgoUdf.UdfNodeView
import AlgoUdf.UdfContext
import AlgoUdf.UdfGlobals
import AlgoUdf.UdfRepoOps
import AlgoUdf.Betweenness
import AlgoUdf.CostProp
import AlgoUdf.LagContract
