/-
# proofs/pr3435_review — PR #3435 "perf: algo.BFS reads the stored matrix in place and pulls on wide levels"

Fixes issue #3430. Rust read from the PR branch `perf/bfs-pushpull` @ a76e66420
(28ab35368, then the review fix a76e66420, on origin/main 9212add39). The PR adds
`Tensor::clean_fwd` / `Tensor::clean_bwd` (tensor.rs) and `bfs_levels_push_pull`,
and changes `algo_bfs` (algo_procedures.rs) to lend the stored base instead of
extracting a copy, to pass a `many_expected` estimate to LAGraph, and to run its
own push/pull BFS when a clean transpose is at hand and edges are not wanted.
a76e66420 acted on this review: `k: i64` + INT64 levels, checked source
`setElement`, both vectors freed on any allocation failure.

Imports: versioned_matrix (`VMTensorOps.TT`, `VMTensor.Inv`/`effV`, `extract`,
`VM.VM`/`eff`, the mutator invariant theorems) and algo_udf
(`AlgoBfsReg.levelBranch`). COVERAGE rows stay commented out until the PR merges.

## Modules

| here | there (PR branch a76e66420) |
| --- | --- |
| `Clean`       | `Tensor::clean_fwd` tensor.rs:989, `Tensor::clean_bwd` :997 |
| `Dist`        | shortest-path distance, walks, `count` (= `GrB_Vector_nvals`) |
| `PushPull`    | the loop of `bfs_levels_push_pull` algo_procedures.rs:1116-1160; GraphBLAS spec `GBSpec` |
| `PushPullFuel`| the loop's fuel, exits, and `k: i64` (:1126, :1128, :1159) |
| `Alloc`       | the whole fn :1066-1163: allocation, checked source, `fail` cleanup; history `bfsFullPre` |
| `AlgoBfs`     | `algo_bfs` path selection :1195-1282 and its level-only `nodes` :1318-1333; LAGraph spec `LagSpec` |
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

## Claim 2 — `bfs_levels_push_pull` (PROVEN given `GBSpec`)

* `Bfs.pull_eq_push`: with `AT = Aᵀ`, a pull step and a push step give the same frontier.
* `Bfs.bfsFull_correct` (the whole fn, no `src < n` and no size premise) /
  `Bfs.pushPull_correct` (the loop): for any bounded graph, any `max_level`
  (`< 0` = unbounded), and **any heuristic** `Heur H` (every push/pull sequence),
  an `Ok w` has `w j = d` exactly for the nodes at shortest-path distance
  `d ≤ max_level` (all reachable when unbounded; source at 0) — `Bfs.Levels`;
  otherwise an `Err`, never fuel. `pushPull_heuristic_irrelevant`,
  `pushPull_correct_rust` (the Rust heuristic over any float model), `pushPull_ok_of_total`.
* Exits: `Bfs.exit_conditions` (`k > max_level` with `max_level ≥ 0`; empty next
  level `nq == 0`); `run_fuel_stable`/`run_fuel_any`: the unbounded `loop` = `run`
  with any fuel `≥ n`; termination because each level adds `nq ≥ 1` unvisited
  nodes (`step_visited`).
* `k: i64` (a76e66420): `Bfs.k_increment_no_overflow` — at every `k += 1`,
  `k + 1 ≤ n ≤ 2^60 < 2^63`, so it never wraps or panics; `n ≤ 2^60` is
  `GrB_Vector_new`'s size limit, so `bfsFull_correct` needs no size premise.
  History: `pre_a76e664_overflow_needs_huge_graph` (28ab35368's `k: i32` could wrap
  only on a `2^31 - 1`-edge shortest path, then ignore `max_level`).
* Source: `Bfs.bfsFull_src_out_of_range` / `bfsFull_src_error` — `src ≥ n` returns
  `Err("BFS source failed: …")` (checked `setElement`, :1110-1114).
* Cleanup: `Bfs.bfsFull_no_leak` — on every `Err` path neither vector stays
  allocated, on `Ok` only the returned `v`. History:
  `pre_a76e664_frontier_alloc_leaks` (28ab35368 leaked `v` when `q`'s alloc failed).

## Claim 3 — the path choice does not change `nodes` (PROVEN, modulo `MtRepr`)

* `Algo.levelVec_correct`: whichever path runs (push/pull on the lent bases, LAGraph
  on the lent base or on the extracted copy) and for **every** `many_expected`,
  an `Ok` level vector is exactly `Levels` of `build_adjacency_matrix`.
* `Algo.nodes_path_independent`, `Algo.nodes_same_as_main`: the level-only `nodes`
  list (`levelBranch`: every node with a level, minus the source, minus deleted
  nodes) is the same for any two `many_expected` values and equals origin/main's
  (LAGraph on the copy, `many_expected = false`). `Algo.levels_unique`.
* `src < n` is **no longer a premise** (a76e66420): `Algo.src_out_of_range_all_err`
  — for `src ≥ n` every path (and origin/main) returns `Err`, so the `Ok`-only
  statements hold for every source. Only the message differs
  (`"BFS source failed: GrB_INVALID_INDEX"` vs `"LAGr_BreadthFirstSearch_Extended failed: …"`).
  History: `pre_a76e664_pushPull_src_out_of_range` (28ab35368 returned an empty
  traversal there).
* Premises left: `GBSpec`, `LagSpec` (FFI; `LagSpec.src_range` is LAGraph's
  `src < n` assert); `TensorOk` (`Reach` + `MtRepr`, claim 1); `Bounded` (an
  `n × n` GraphBLAS matrix stores in-range entries); `hV` (`GrB_Vector_new`
  refuses `n > 2^60`).
* `many_expected` only reaches LAGraph's `LG_SET_FORMAT_HINT` and the `usePP` gate.

## Claim 4 — existing theorems

No existing theorem is invalidated. algo_udf `AlgoBfsReg.algoBfs` takes LAGraph's
output as the input `lag`; the PR's result assembly (:1284-1353) is unchanged,
so `edgesBranch_aligned`, `branches_agree`, `levelBranch_no_src`, `algoBfs_*`
hold with `lag.level` = either path's vector. graph_queries
`Queries.adjMatrix_single` (build_adjacency_matrix) and every versioned_matrix
tensor theorem are untouched (the PR only adds two `&self` readers). Cites shift:
tensor.rs every fn after :982 by +19 (versioned_matrix 33 rows, graphblas_wrappers
6 rows); algo_procedures.rs (at a76e66420) +1 for fns after :76, `register_bfs`/`algo_bfs`
+107 (:1058→:1165, :1063→:1170), every fn after `algo_bfs` +153 (algo_udf,
ops_traverse rows; algo_udf `AlgoBfsReg` header :1063-1205/:1144 too).

## Live check (28ab35368 dylib, port 6521; not re-run for a76e66420)

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
import Pr3435Review.Alloc
import Pr3435Review.AlgoBfs
import Pr3435Review.Model
