/-
# proofs/pr3435_review — PR #3435 "perf: algo.BFS reads the stored matrix in place and pulls on wide levels"

Fixes issue #3430. Rust read from the PR branch `perf/bfs-pushpull` @ 28ab35368
(one commit on origin/main 9212add39). The PR adds `Tensor::clean_fwd` /
`Tensor::clean_bwd` (tensor.rs) and `bfs_levels_push_pull`, and changes `algo_bfs`
(algo_procedures.rs) to lend the stored base instead of extracting a copy, to
pass a `many_expected` estimate to LAGraph, and to run its own push/pull BFS when
a clean transpose is at hand and edges are not wanted.

Imports: versioned_matrix (`VMTensorOps.TT`, `VMTensor.Inv`/`effV`, `extract`,
`VM.VM`/`eff`, the mutator invariant theorems) and algo_udf
(`AlgoBfsReg.levelBranch`). COVERAGE rows stay commented out until the PR merges.

## Modules

| here | there (PR branch 28ab35368) |
| --- | --- |
| `Clean`       | `Tensor::clean_fwd` tensor.rs:989, `Tensor::clean_bwd` :997 |
| `Dist`        | shortest-path distance, walks, `count` (= `GrB_Vector_nvals`) |
| `PushPull`    | `bfs_levels_push_pull` algo_procedures.rs:1066-1148; GraphBLAS spec `GBSpec` |
| `PushPullFuel`| the loop's fuel, exits, and `k: i32` (:1111, :1113, :1144) |
| `AlgoBfs`     | `algo_bfs` path selection :1180-1267 and its level-only `nodes` :1303-1318; LAGraph spec `LagSpec` |
| `Model`       | the FFI hypotheses are satisfiable (`gbModel_spec`, `lagModel_spec`) + `decide` runs |

## Claim 1 — `clean_fwd` / `clean_bwd` (PROVEN; transpose under the owner's `mt` abstraction)

* `Clean.cleanFwd_isSome_iff`: `Some` iff `dp` and `dm` hold no entries.
* `Clean.cleanFwd_eq_extract`: then the lent base's pattern **is** `extract()`'s
  `(m ∖ dm) ∪ dp` (equal lists), so BFS on it sees `build_adjacency_matrix`'s edges
  for that type (`Algo.aPat_eq_buildAdj`).
* `Clean.cleanBwd_isSome_iff`, `Clean.cleanBwd_eff`: `Some(mt.m)` iff `mt`'s deltas
  are empty, and then `mt.m`'s pattern is `mt`'s logical contents.
* `Clean.clean_transpose`, `clean_transpose_reach`, `cleanBwd_transpose_extract`:
  `(j,i) ∈ clean_bwd ↔ (i,j) ∈ clean_fwd`. Rests on (a) `Inv.mt_eff` on every pair —
  **discharged** for every modelled tensor history by `Clean.Reach.allInv` (new;
  `addBatch`/`removeBatch`/`removeFast` under their caller preconditions
  `AddOk`/`RemOk`, `flush`, `fold_latched`, `fold_oversized`, `resize`, `dup`,
  `wait`, decode + `rebuild_backward`; from versioned_matrix's per-op theorems);
  (b) `MtRepr`: `mt`'s layers represent the per-pair `mt` bit — the abstraction
  versioned_matrix's tensor model itself makes (`mt` modelled by its logical set,
  `VM.run_repr`), named here because this PR is the first reader of `mt`'s *base*.

## Claim 2 — `bfs_levels_push_pull` (PROVEN given `GBSpec`; `n < 2^31`)

* `Bfs.pull_eq_push`: with `AT = Aᵀ`, a pull step and a push step give the same frontier.
* `Bfs.pushPull_correct`: for any bounded graph, `src < n`, any `max_level`
  (`< 0` = unbounded), and **any heuristic** `Heur H` (every push/pull sequence),
  an `Ok w` has `w j = d` exactly for the nodes at shortest-path distance
  `d ≤ max_level` (all reachable when unbounded; source at 0) — `Bfs.Levels`;
  otherwise the result is one of the two GraphBLAS-failure `Err`s, never fuel.
  `pushPull_heuristic_irrelevant` (two heuristics, one answer),
  `pushPull_correct_rust` (the Rust heuristic over any float model),
  `pushPull_ok_of_total`.
* Exits: `Bfs.exit_conditions` (`k > max_level` with `max_level ≥ 0`; empty next
  level `nq == 0`); `run_fuel_stable`/`run_fuel_any`: the unbounded `loop` = `run`
  with any fuel `≥ n`; termination because each level adds `nq ≥ 1` unvisited
  nodes (`step_visited`).
* `k: i32`: `k ≤ n` (`k_le_n`); `overflow_needs_huge_graph`: `k += 1` (:1144) can
  only reach `i32::MAX` with a node at shortest-path distance `2^31 - 1`, so
  `n ≥ 2^31` nodes on one shortest path. Then a debug build panics and a release
  build wraps `k` to `i32::MIN`: `i64::from(k) > max_level` never fires again and
  the INT32 levels go negative, so a bounded BFS with `max_level ≥ 2^31 - 1` would
  run on past `max_level` (LAGraph uses an `int64_t` `k` and INT64 levels when
  `n > INT32_MAX`). Not reachable in practice (a 2^31-node chain, 2^31 sequential
  GraphBLAS steps); not confirmable live. Theorems take `n < 2^31`.

## Claim 3 — the path choice does not change `nodes` (CONDITIONAL)

* `Algo.levelVec_correct`: whichever path runs (push/pull on the lent bases, LAGraph
  on the lent base or on the extracted copy) and for **every** `many_expected`,
  an `Ok` level vector is exactly `Levels` of `build_adjacency_matrix`.
* `Algo.nodes_path_independent`, `Algo.nodes_same_as_main`: the level-only `nodes`
  list (`levelBranch`: every node with a level, minus the source, minus deleted
  nodes) is the same for any two `many_expected` values and equals origin/main's
  (LAGraph on the copy, `many_expected = false`). `Algo.levels_unique`.
* Premises: `GBSpec`, `LagSpec` (FFI); `TensorOk` (`Reach` + `MtRepr`, claim 1);
  `Bounded` (an `n × n` GraphBLAS matrix stores in-range entries); `n < 2^31`; and
  **`src < n`** — the source id is below the tensor's `nrows` (= `node_cap`), a
  graph invariant of graph.rs `resize` that is not discharged here. It matters:
  for `src ≥ n` LAGraph returns `GrB_INVALID_INDEX` (an `Err`) while push/pull
  ignores `GrB_Vector_setElement`'s failure and returns empty levels
  (`Algo.pushPull_src_out_of_range`). Live: a source created in the same query
  (`UNWIND range(1,200) CREATE (x) … CALL algo.BFS(x, -1, 'R')`) is in range.
* `many_expected` only reaches LAGraph's `LG_SET_FORMAT_HINT` and the `usePP` gate.

## Claim 4 — existing theorems

No existing theorem is invalidated. algo_udf `AlgoBfsReg.algoBfs` takes LAGraph's
output as the input `lag`; the PR's result assembly (:1269-1338) is unchanged,
so `edgesBranch_aligned`, `branches_agree`, `levelBranch_no_src`, `algoBfs_*`
hold with `lag.level` = either path's vector. graph_queries
`Queries.adjMatrix_single` (build_adjacency_matrix) and every versioned_matrix
tensor theorem are untouched (the PR only adds two `&self` readers). Cites shift:
tensor.rs every fn after :982 by +19 (versioned_matrix 33 rows, graphblas_wrappers
6 rows); algo_procedures.rs +1 for fns after :76, `register_bfs`/`algo_bfs` +92
(:1058→:1150, :1063→:1155), every fn after `algo_bfs` +138 (algo_udf, ops_traverse
rows; algo_udf `AlgoBfsReg` header :1063-1205/:1144 too).

## Live check (PR dylib, port 6521)

`live_bfs.py` (session scratchpad `proof-3435/`): 60 nodes, ~180 random R/S edges,
sources every 7th node, `max_depth ∈ {-1,0,1,2,3}`, type R / S / none,
`yield nodes` and `yield nodes, edges`, on a fresh graph, after `DEBUG RELOAD`
(clean tensors: lent bases, push/pull), after deletes (pending deltas: extract +
LAGraph), and on a dense R block after reload — 0 mismatches against a Python BFS.

## Assumptions

* `Bfs.GBSpec` (GrB_vxm / GrB_mxv with `GxB_ANY_PAIR_BOOL` + `GrB_DESC_RSC`,
  GrB_Vector_assign_INT32 with `GrB_DESC_S`, GrB_Vector_nvals), `Algo.LagSpec`
  (LAGr_BreadthFirstSearch_Extended level output) — FFI, hypothesis structures,
  shown satisfiable (`Model.gbModel_spec`, `Model.lagModel_spec`). No `axiom`.
* Sparsity-control options are storage hints (not modelled).

Headline theorems depend only on `propext`, `Quot.sound`, `Classical.choice`.
-/
import Pr3435Review.Clean
import Pr3435Review.Dist
import Pr3435Review.PushPull
import Pr3435Review.PushPullFuel
import Pr3435Review.AlgoBfs
import Pr3435Review.Model
