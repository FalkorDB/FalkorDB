/-
# algo.* procedures and the JS UDF layer: models, proofs, and the bugs they found

Scope: `graph/src/runtime/functions/algo_procedures.rs` (BFS, WCC, CDLP, pageRank,
betweenness, HarmonicCentrality, MSF, maxFlow, SPpaths/SSpaths) and
`graph/src/udf/` (all files). Rust source of truth: origin/main 1c0e6c8de (re-targeted from e860645bb: #3058, #3586, #3066, #3464 — see "Fixed on main (re-target e860645bb → 1c0e6c8de)"); earlier: 3fec7d7c9, re-checked at 5e475e6c9 (#3436 only: `rt.g.borrow()` -> `rt.graph()` in every `algo_procedures.rs` / `procedures.rs` registration, and `call_udf_bridge` takes `rt.graph_arc()` (the same `Arc` clone `rt.g.clone()` was) for `set_current_graph`; no function moved, nothing modelled changed); `js_globals.rs`
re-targeted to 8743953a8 (#3079), `type_convert.rs` to cab3cf192 (#3074, #3336 — see "Fixed on main").
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
| `AlgoUdf.Nesting`/`MarshalCap` | #3336: the depth cap `MAX_UDF_VALUE_NESTING`, `value_to_js_at` / `js_to_value_at` with an explicit stack |
| `AlgoUdf.Repo`       | `UdfRepo::load`/`delete` + the case-sensitive function registry (#3066) |
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
| `AlgoUdf.UdfNodeView` | #3454 (fixed by #3464): which view (`Runtime` vs committed `Graph`) `create_js_node` reads labels/attributes from; `set_node_labels`, `set_node_attrs` |
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
* pageRank with an unknown label errors — **FIXED by #3058** (`Compact.pre3058_pagerank_unknown_label_compact_empty`; now `pagerank_unknown_label_empty`).
* Forged `__falkor_type` markers accepted (type_convert.rs:351-358) —
  `Marshal.forged_node_marker`, `forged_edge_marker`.
* -0.0 → Int 0 — **FIXED by #3074** (see "Fixed on main").
* Qualified-name collision / case folding (repository.rs:94-137, functions/mod.rs:1121) — **FIXED by #3066**
  (`Repo.pre3066_collision_accepted`, `pre3066_delete_breaks_other`, `pre3066_case_folded`; `dotted_collision` stays as the reason the check compares qualified names).
* getNeighbors self-loop twice (js_classes.rs:97-127) — `Neighbors.self_loop_twice`.
New in this wave (checked on live C 18620 / Rust 18621, then cargo):
1. **FIXED by #3113 (`8a66c01e5`)** — SPpaths/SSpaths charged a missing cost as 0
   (algo_procedures.rs bound and enumeration); C charges 1 per edge, so pathCost = hops
   without costProp and `maxCost` bounds hops. Now default 1.0 (:2399, :2521):
   `Paths.cost_agree`, `cost_no_prop_hops`, `cost_default_agrees`; historical
   `pre3113_cost_default_diverges`. The Dijkstra fast path (:2308) still reports 0 and C's
   fast path does too (live, `Paths.dijkstra_cost_no_prop`).
2. **FIXED by #3074** — UDF map with key `constructor: {name:'Date'}` fails ("Date getTime error");
   `{name:'RegExp'}` becomes the string "[object Object]" (type_convert.rs:314-350).
   `Marshal.constructor_key_breaks_roundtrip`, `constructor_regexp_becomes_string`;
   `bug_udf_constructor_key_mistaken_for_date` (C returns the map).
3. **FIXED by #3074** — vecf32 holding ±inf cannot pass through a UDF (type_convert.rs:239-241).
   `Marshal.vecf32_inf_not_roundtrip`; `bug_udf_vecf32_inf_rejected` (C [inf]).
4. betweenness / HarmonicCentrality with an unknown nodeLabels entry return no rows;
   C errors ("unknown label Nope"). `bug_betweenness_unknown_label_is_empty_not_error` — **FIXED by #3058** (`Compact.checkKnown_*`).

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

## Fixed on main (re-target 8743953a8 → cab3cf192)
* #3336 (`33a956360`, GHSA-frmf-ffvc-mxx5) — `value_to_js` / `js_to_value` recurse through
  `value_to_js_at` (:110) / `js_to_value_at` (:262) with a depth bounded by `MAX_UDF_VALUE_NESTING = 1024`
  (:67; error `udf_nesting_error` :87; `with_stack` :80 is AXIOMATISED stack switching). The JS side
  is modelled over a QuickJS heap, so a self-containing array is an input. Generic `Step` theorems
  (`Nesting.lean`) instantiated for both directions (`MarshalCap.lean`):
  `fromJs_frames`, `toJs_frames` (MAX + 1 levels always suffice, any input, any heap);
  `fromJs_past_cap`, `toJs_past_cap` (past the cap: an `Err`, never an overflow); `fromJs_eq_pre3336`,
  `toJs_eq_pre3336`, `pre3336_finished_eq` (within the cap: exactly the old result on any stack the old
  code finished on); `self_array_error`, `fromJs_nestArr_ok`/`_past`, `advisory_array_error`,
  `nestList_rt`, `toJs_nestList_past` (the Rust tests); `fromJs_needs_all_frames` (bound tight).
  Historical: `pre3336_self_array_overflows` (the old code overflows every stack on `a = [a]`),
  `pre3336_nestArr_overflows` (n nested arrays needed n + 1 levels).
  `fromJs` now also models the `"path"` arm (:359-397), which the earlier model sent to the map branch.
* Model finding (not a crash, not confirmed live): `cap_asymmetry_vecf32` — `value_to_js` counts only
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
* Datetime `ts * 1000` is unchecked i64 arithmetic (type_convert.rs:215, 224) — modelled as an error.

## Gaps (earlier waves)
Closed in wave 5: Dijkstra, the DFS enumeration, the BFS bound, BFS parent→edge
reconstruction, js_context/js_globals glue and graph.traverse/iterate*.

## Gap closed 2026-10-20 (bugs the model missed; all **Open**, live-checked on main 9d3869aa8 vs C)
* #3380 `Betweenness`: seeded draw loop and the C reference modelled.
  `bug3380_default_size_16` (default 16, C 32; property `DefaultSizeMatchesC`),
  `bug3380_seedless_deterministic` (no clock seed; `SeedlessRunVaries`),
  `bug3380_seeded_fewer_sources` (20 requested on 40 nodes, seed 7 → 15; `SourcesExact`).
* #3121 `CostProp`: `costProp` configured vs. absent kept apart (`edge_numeric_attr` :2104).
  `bug3121_fastpath_missing_cost_zero`: the Dijkstra fast path (:2308) charges a configured
  but missing cost 0 (C 1); property `MissingCostIsOne` (the enumeration path satisfies it).
* #3349 `LagContract`: the LAGraph contract now has the `LAGRAPH_INVALID_GRAPH` (-1000)
  outcome (`LAGraph_CheckGraph` rejects by-column storage; GraphBLAS stores n-by-1, hence
  1×1, by column). `bug3349_one_node_projection_fails`; property `ProjectionTotal`.

## Gap closed 2026-10-22 (#3454) — **FIXED by #3464** (see below)
`UdfNodeView` modelled `create_js_node` reading the committed graph while `labels(n)` reads the
runtime; the counterexamples are now `pre3454_deleted_node_empty`, `pre3454_edge_source_empty`.

## Fixed on main (re-target e860645bb → 1c0e6c8de)
* #3058 (fixes #3056) — `check_known_labels_and_types` (algo_procedures.rs:469) makes betweenness (:973),
  labelPropagation (:1268) and HarmonicCentrality (:2759) error on an unknown `nodeLabels` /
  `relationshipTypes` entry with C's messages; pageRank returns an empty result (:729-735, :772-774) for an
  unknown label / type or an empty label. `Compact.checkKnown`, `checkKnown_ok_iff`,
  `checkKnown_unknown_label_errors`, `checkKnown_unknown_type_errors`, `checkKnown_label_first`,
  `known_messages`; `pagerankPath`, `pagerank_unknown_label_empty`, `pagerank_unknown_type_empty`,
  `pagerank_compact_nonempty` (LAGraph is never handed a 0x0 matrix), `pagerank_known_label_full`.
  Historical: `prePagerankPath`, `pre3058_pagerank_unknown_label_compact_empty`.
  Still open (unchanged): `Compact.unfiltered_drops_high_ids` (HarmonicCentrality, #3044; fix PR #3045 not merged).
* #3586 (fixes #3444) — algo.MSF builds the original-id → compact-id table only when `max_id < 4 n`, else
  binary-searches `sorted_ids` (algo_procedures.rs:1442-1455, :1741). `MsfFlow.buildTbl_get`, `compactOf_eq`
  (both branches answer `bsearch`), `denseTable_len` (table ≤ 4 slots per active node), `sparse_builds_no_table`;
  historical `pre3586_table_unbounded`. `bsearch` = `slice::binary_search` on a sorted duplicate-free list: its
  std spec is assumed.
* #3066 (fixes #3065) — registry keys are the case-sensitive qualified names (functions/mod.rs:1066, :1131-1145,
  js_context.rs:261, :292) and `UdfRepo::load` (repository.rs:118-132) rejects a qualified name another
  library owns, compared with `eq_ignore_ascii_case`. `Repo.load` (with `replace`), `collides`,
  `collision_rejected`, `collision_rejected_replace`, `dotted_collision_rejected`, `replace_self_ok`,
  `same_lib_cases_coexist`, `case_sensitive`, `CrossFoldUnique`, `load_preserves_crossFold`,
  `keysUnique_of_crossFold`, `delete_preserves_after3066`. Historical: `preLoad`, `pre3066_collision_accepted`,
  `pre3066_delete_breaks_other`, `pre3066_case_folded`. `UdfContext` is parameterised by the key function
  (`low`); the real code now instantiates it with the identity.
* #3464 (fixes #3454) — `create_js_node(ctx, id, graph, runtime)` (js_classes.rs:134) reads through the runtime
  when `Runtime::node_differs_from_committed` (runtime.rs:1772) holds (deleted this query, or
  `Pending::node_has_changes`, pending.rs:755), else the committed graph; edges and paths pass the runtime on;
  new helpers `set_node_labels` :203, `set_node_attrs` :220. `UdfNodeView.rs_runtime_first`, `rs_edge_ok`,
  `RT.not_differs` (the fast path agrees with the runtime view), `jsNode_eq_helpers`,
  `fixed3454_deleted_node`, `fixed3454_edge_source`. Historical: `preRsJsNode`, `pre3454_deleted_node_empty`,
  `pre3454_edge_source_empty`. Callers without a runtime (getNeighbors, getNodeById, iterateNodes, traverse) still
  pass `None`.
* #3280 — `dbms.functions` / `fulltext.drop` (procedures.rs, graph.rs): not in this project.
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
