/-
# algo.* procedures and the JS UDF layer: models, proofs, and the bugs they found

Scope: `graph/src/runtime/functions/algo_procedures.rs` (BFS, WCC, CDLP, pageRank,
betweenness, HarmonicCentrality, MSF, maxFlow, SPpaths/SSpaths) and
`graph/src/udf/` (all files). Rust source of truth: origin/main 3fec7d7c9; `type_convert.rs`
and `js_globals.rs` re-targeted to 8743953a8 (#3074, #3079 — see "Fixed on main").
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
| `AlgoUdf.Marshal`/`MarshalThm` | `value_to_js` / `js_to_value` over an abstract f64 and QuickJS value model |
| `AlgoUdf.Repo`       | `UdfRepo::load`/`delete` + the lowercase-keyed function registry |
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
| `AlgoUdf.UdfContext` | `caught_error_message`, timeouts, `validate_script`, `ensure_context_current`, `rebuild_context`, `call_udf_bridge` |
| `AlgoUdf.UdfGlobals` | the JS `falkor.register` bodies, `collect_*`, `js_value_to_log_string` |
| `AlgoUdf.UdfRepoOps` | `UdfRepo::{new,default,version,bump_version,flush,list,get_all_libraries,serialize,deserialize}`, `init_udf_repo`, `get_udf_repo` |

## Assumptions (AXIOMATISED as hypotheses/definitions, no Lean `axiom`)
* LAGraph: WCC/CDLP output is a dense vector of representative indices `< n`
  (`Compact.LagWcc`); `LAGraph_msf` component vector is dense
  (`MsfFlow.msf_dense_no_fallback` hypothesis); PageRank rejects n = 0.
* `Graph::get_node_relationships` yields out-edges then in-edges (self-loop in both).
* QuickJS: `obj.set("__proto__", v)` never creates an own key; `obj.keys()`
  lists array-index keys first; `obj.get("constructor")` falls back to the
  prototype's constructor (historical `pre3074_*` model only); `JS_IsDate` / `JS_IsRegExp`
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
5. `graph.traverse([a], {returnType:'edges'})` returns a self-loop twice
   (js_classes.rs:770 → `collect_edges` :97-127). C `[1,0]`, Rust `[1,0,1]`.
   `UdfTraverse.traverse_self_loop_twice`; `bug_udf_traverse_self_loop_edge_twice`.
6. `graph.traverse([a])` omits `a` when it is its own neighbour (start pre-marked
   visited, js_classes.rs:744). C `[0,1]`, Rust `[1]` (Rust's `getNeighbors` gives
   `[0,1]`). `traverse_drops_self_neighbour`; `bug_udf_traverse_omits_self_loop_neighbour`.
7. `node.getNeighbors()` without an argument throws "Error converting from js
   'undefined' into type 'object'" (js_classes.rs:195 forwards `undefined` to an
   `Opt<Object>`, :352). C returns the neighbours.
   `UdfClasses.getNeighbors_no_arg_errors`; `bug_udf_get_neighbors_without_config_throws`.
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
  `spec_component_is_member`, `unfiltered_rows_exact`/`_nodup`, `harmonic_correct_iff`.
* `Paths.far_*`, `recordK_length`, `record1_min`, `record0_ties`, `nextNode_far`,
  `dijkstra_never_src_eq_dst`.
* `MsfFlow.keepMin_comm`, `identity_neutral`, `flowAlign_aligned`, `superFresh`.
* `Marshal.int_roundtrip`, `float_rt_iff`, `node_rt`, `rel_rt`, `point_rt`,
  `datetime_rt`, `vecf32_finite_rt`, `unesc_esc`, `keep_esc`, `map_never_marked`.
* `Repo.delete_preserves` (under the missing `KeysUnique` check).
* `Neighbors.outgoing_ok` (no self-loop), `string_types_filter`.

## CONFIRMED bugs (Rust vs C; Lean counterexample → cargo repro)
Known before this wave (counterexamples added):
* WCC/CDLP nodeLabels report compact index as component id (:891-895, :1298-1302) —
  `Compact.wcc_compact_component_id_not_member`.
* HarmonicCentrality source vector sized `node_count` not `node_id_bound` (:2777) —
  `Compact.unfiltered_drops_high_ids`.
* pageRank with an unknown label errors (:727-778) — `Compact.pagerank_unknown_label_compact_empty`.
* Forged `__falkor_type` markers accepted (type_convert.rs:258-268) —
  `Marshal.forged_node_marker`, `forged_edge_marker`.
* -0.0 → Int 0 — **FIXED by #3074** (see "Fixed on main").
* Qualified-name collision / case folding (repository.rs:94-137, functions/mod.rs:1121) —
  `Repo.collision_accepted`, `dotted_collision`, `delete_breaks_other`, `case_folded`.
* getNeighbors self-loop twice (js_classes.rs:97-127) — `Neighbors.self_loop_twice`.
New in this wave (checked on live C 18620 / Rust 18621, then cargo):
1. SPpaths/SSpaths charge a missing cost as 0 (algo_procedures.rs:2308, 2397, 2518);
   C charges 1 per edge, so pathCost = hops without costProp and `maxCost` bounds hops.
   `Paths.cost_default_diverges`; `bug_sppaths_missing_cost_defaults_to_zero`
   (C 2, Rust 0), `bug_sppaths_max_cost_ignores_hops` (C no rows, Rust 1 row).
2. **FIXED by #3074** — UDF map with key `constructor: {name:'Date'}` fails ("Date getTime error");
   `{name:'RegExp'}` becomes the string "[object Object]" (type_convert.rs:314-350).
   `Marshal.constructor_key_breaks_roundtrip`, `constructor_regexp_becomes_string`;
   `bug_udf_constructor_key_mistaken_for_date` (C returns the map).
3. **FIXED by #3074** — vecf32 holding ±inf cannot pass through a UDF (type_convert.rs:239-241).
   `Marshal.vecf32_inf_not_roundtrip`; `bug_udf_vecf32_inf_rejected` (C [inf]).
4. betweenness / HarmonicCentrality with an unknown nodeLabels entry return no rows;
   C errors ("unknown label Nope"). `bug_betweenness_unknown_label_is_empty_not_error`.

## Fixed on main (re-target a9377c636 → 8743953a8)
* #3074 (`7a81c83b0`, issue #3073) — `js_to_value` keeps -0.0 a Float (type_convert.rs:203),
  picks Date/RegExp by internal class (`JS_IsDate`/`JS_IsRegExp`, :313-345) instead of a
  `constructor` property, and lets non-finite vecf32 elements through (:236-244).
  Now: `Marshal.float_rt_iff` (−0.0 excluded), `neg_zero_rt`, `num_eq_pre3074_off_negZero`,
  `plain_obj_is_map`, `constructor_key_roundtrips`, `constructor_regexp_roundtrips`,
  `vecf32_rt`, `vecf32_inf_rt`. Historical: `pre3074_neg_zero_not_preserved`,
  `pre3074_constructor_key_breaks_roundtrip`, `pre3074_constructor_regexp_becomes_string`,
  `pre3074_vecf32_inf_not_roundtrip`.
* #3079 (`006428fc5`, issue #3077) — the validation context now defines the `graph` global
  (`setup_graph_global`, js_globals.rs:163-197, called from both setups).
  Now: `UdfGlobals.validate_runtime_same_globals`, `graph_global_in_both`,
  `validate_runtime_same_graph`. Historical: `pre3079_validate_lacks_graph`.

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
* Datetime `ts * 1000` is unchecked i64 arithmetic (type_convert.rs:148, 157) — modelled as an error.

## Gaps (earlier waves)
Closed in wave 5: Dijkstra, the DFS enumeration, the BFS bound, BFS parent→edge
reconstruction, js_context/js_globals glue and graph.traverse/iterate*.
-/
import AlgoUdf.Value
import AlgoUdf.Config
import AlgoUdf.Compact
import AlgoUdf.Paths
import AlgoUdf.MsfFlow
import AlgoUdf.Marshal
import AlgoUdf.MarshalThm
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
import AlgoUdf.UdfContext
import AlgoUdf.UdfGlobals
import AlgoUdf.UdfRepoOps
