import AlgoUdf.Compact
/-! # LAGraph's graph-validity outcome on the compact path — gap closed 2026-10-20

`Compact.LagWcc` (and the `LAGr_PageRank` / `LAGr_Betweenness` assumptions) take LAGraph
to succeed on every square matrix. It does not: `LAGraph_CheckGraph` rejects a matrix
held by column with `LAGRAPH_INVALID_GRAPH` (-1000), and GraphBLAS holds every
`n`-by-1 matrix by column — including the 1-by-1 one the compact builders make for a
one-node projection. This file adds that outcome to the axiomatised contract.

The compact matrix and the cached transpose are pinned by row
(`pin_row_major`), so every non-empty projection passes the check (`projection_total`).

| here | there |
| --- | --- |
| `newOrient`   | GraphBLAS `GxB_Matrix_new_arena` (`deps/GraphBLAS` ea79e35, `Source/matrix/GxB_Matrix_new_arena.c:89-101`): `ncols == 1` → by column; `nrows == 1` → by row; else the global format (by row) |
| `checkGraph`  | `LAGraph_CheckGraph` (`deps/LAGraph` d01064d, `src/utility/LAGraph_CheckGraph.c:46-56`): non-square or not `GrB_ROWMAJOR` → `LAGRAPH_INVALID_GRAPH` |
| `Algo`, `checks` | `LAGr_ConnectedComponents` → `LG_CC_FastSV6/7` (`CheckGraph` at :263/:279), `LAGr_PageRank` (:76), `LAGr_Betweenness` (:131) check; `LAGraph_cdlp` does not |
| `pinRowMajor` | `pin_row_major` `:584`: `GrB_Matrix_set_INT32(m, GrB_ROWMAJOR, GrB_STORAGE_ORIENTATION_HINT)` (held by row afterwards, whatever the shape) |
| `newCompactBoolMatrix`, `compactBuild` | `new_compact_bool_matrix` `:596` (`GrB_Matrix_new(&m, GrB_BOOL, n, n)` `:600`, then `pin_row_major`), called by `build_compact_adj_from_tensors` `:624` (`:680`) / `..._symmetric_...` `:689` (`:749`), `n` = projection size |
| `cachedAT`, `cacheTranspose`, `usesAT` | `LAGraph_Cached_AT` (transpose made by `GrB_Matrix_new(ncols, nrows)`); `cache_transpose` `:608` pins `G->AT` by row; called by `algo_pagerank` `:838`, `algo_betweenness` `:1069` (and `algo_maxflow` `:3326`, not on the compact path); `algo_wcc` caches no `AT` |
| `projectedRun` | `algo_wcc` `:896` (check `:959`), `algo_pagerank` `:777` (check `:856`), `algo_betweenness` `:997` (check `:1111`): `if info != 0 { return Err(format!("… failed: {info}")) }` |
| `compactBuildPre3572`, `projectedRunPre3572` | the same without the pinning (the pre-#3572 behaviour, kept as a historical counterexample): `GrB_Matrix_new` unpinned (1×1 by column) and `LAGraph_Cached_AT` unpinned |

AXIOMATISED (as definitions, no Lean `axiom`): the two FFI functions above. When the
check passes, the algorithm returns one entry per compact index (the existing `LagWcc`
contract), so the procedure yields one row per projected node.
-/
namespace AlgoUdf.LagContract

inductive Orient | byRow | byCol deriving DecidableEq

/-- GraphBLAS storage orientation of a fresh `GrB_Matrix_new(nrows, ncols)`
(global default by row). -/
def newOrient (_nrows ncols : Nat) : Orient :=
  if ncols = 1 then .byCol else .byRow

def LAGRAPH_INVALID_GRAPH : Int := -1000

/-- `LAGraph_CheckGraph` on an adjacency matrix: 0 or `LAGRAPH_INVALID_GRAPH`. -/
def checkGraph (nrows ncols : Nat) (o : Orient) : Int :=
  if nrows ≠ ncols then LAGRAPH_INVALID_GRAPH
  else if o = .byCol then LAGRAPH_INVALID_GRAPH else 0

/-- `LAGraph_CheckGraph` on `G->A` and, when cached, `G->AT` (both must be held by row). -/
def checkGraphAT (a : Nat × Nat × Orient) (at? : Option (Nat × Nat × Orient)) : Int :=
  let ia := checkGraph a.1 a.2.1 a.2.2
  if ia ≠ 0 then ia else
  match at? with
  | none => 0
  | some t => checkGraph t.1 t.2.1 t.2.2

inductive Algo | wcc | pagerank | betweenness | cdlp deriving DecidableEq

/-- Whether the algorithm validates the graph first. -/
def checks : Algo → Bool
  | .cdlp => false
  | _ => true

def Algo.name : Algo → String
  | .wcc => "LAGr_ConnectedComponents"
  | .pagerank => "LAGr_PageRank"
  | .betweenness => "LAGr_Betweenness"
  | .cdlp => "LAGraph_cdlp"

/-- `pin_row_major`: after the orientation hint the matrix is held by row. -/
def pinRowMajor (_ : Orient) : Orient := .byRow

/-- `new_compact_bool_matrix n`: `GrB_Matrix_new(n, n)` then `pin_row_major`. -/
def newCompactBoolMatrix (n : Nat) : Nat × Nat × Orient := (n, n, pinRowMajor (newOrient n n))

/-- The compact matrix for a projection of `n` nodes (both builders). -/
def compactBuild (n : Nat) : Nat × Nat × Orient := newCompactBoolMatrix n

/-- `LAGraph_Cached_AT` on an `r × c` matrix: a fresh `c × r` `GrB_Matrix_new`. -/
def cachedAT (a : Nat × Nat × Orient) : Nat × Nat × Orient :=
  (a.2.1, a.1, newOrient a.2.1 a.1)

/-- `cache_transpose`: `LAGraph_Cached_AT`, then `pin_row_major(G->AT)`. -/
def cacheTranspose (a : Nat × Nat × Orient) : Nat × Nat × Orient :=
  let t := cachedAT a
  (t.1, t.2.1, pinRowMajor t.2.2)

/-- Whether the caller caches `G->AT` before the run (pageRank, betweenness). -/
def usesAT : Algo → Bool
  | .pagerank | .betweenness => true
  | _ => false

/-- The LAGraph call's return code on the compact matrix. -/
def lagInfo (a : Algo) (n : Nat) : Int :=
  let m := compactBuild n
  if checks a then checkGraphAT m (if usesAT a then some (cacheTranspose m) else none) else 0

/-- The procedure on a projection `ids` (non-empty; the empty case is handled earlier):
`Err` on a non-zero code, else one row per projected node. -/
def projectedRun (a : Algo) (ids : List Nat) : Except String Nat :=
  let info := lagInfo a ids.length
  if info ≠ 0 then .error s!"{a.name} failed: {info}" else .ok ids.length

/-- **Every non-empty projection is answered** (C parity: C returns `a, 0` for WCC and
`a, 0.333` for pageRank on a one-node `Node` projection): the procedure returns one
row per projected node. -/
def ProjectionTotal (run : List Nat → Except String Nat) : Prop :=
  ∀ ids : List Nat, ids ≠ [] → run ids = .ok ids.length

theorem compactBuild_byRow (n : Nat) : compactBuild n = (n, n, .byRow) := rfl

theorem lagInfo_zero (a : Algo) (n : Nat) : lagInfo a n = 0 := by
  cases a <;> simp [lagInfo, compactBuild, newCompactBoolMatrix, pinRowMajor, checks, usesAT,
    checkGraphAT, checkGraph, cacheTranspose, cachedAT]

/-- The contract is honest about square matrices of size ≥ 2: they pass. -/
theorem lagInfo_two_le (a : Algo) (n : Nat) (_h : 2 ≤ n) : lagInfo a n = 0 := lagInfo_zero a n

theorem projectedRun_two_le (a : Algo) (ids : List Nat) (_h : 2 ≤ ids.length) :
    projectedRun a ids = .ok ids.length := by
  simp [projectedRun, lagInfo_zero]

/-- **#3349 fixed by #3572**: every algorithm answers every non-empty projection,
including a one-node one. -/
theorem projection_total (a : Algo) : ProjectionTotal (projectedRun a) := by
  intro ids _; simp [projectedRun, lagInfo_zero]

/-- CDLP does not run `LAGraph_CheckGraph`, so it satisfies the property (live: Rust
`algo.labelPropagation({nodeLabels:['Node']})` returns `a, 0`). -/
theorem cdlp_total : ProjectionTotal (projectedRun .cdlp) := projection_total .cdlp

/-- The one-node projection is answered with one row (#3572; was `-1000`). -/
theorem one_node_projection_ok :
    projectedRun .wcc [0] = .ok 1 ∧ projectedRun .pagerank [0] = .ok 1 ∧
    projectedRun .betweenness [0] = .ok 1 :=
  ⟨rfl, rfl, rfl⟩

/-! ### History: no `pin_row_major` (the pre-#3572 behaviour) -/

/-- `GrB_Matrix_new(&compact, GrB_BOOL, n, n)`, unpinned. -/
def compactBuildPre3572 (n : Nat) : Nat × Nat × Orient := (n, n, newOrient n n)

def lagInfoPre3572 (a : Algo) (n : Nat) : Int :=
  let m := compactBuildPre3572 n
  if checks a then checkGraphAT m (if usesAT a then some (cachedAT m) else none) else 0

def projectedRunPre3572 (a : Algo) (ids : List Nat) : Except String Nat :=
  let info := lagInfoPre3572 a ids.length
  if info ≠ 0 then .error s!"{a.name} failed: {info}" else .ok ids.length

theorem pre3572_two_le (a : Algo) (ids : List Nat) (h : 2 ≤ ids.length) :
    projectedRunPre3572 a ids = .ok ids.length := by
  have h1 : ids.length ≠ 1 := by omega
  cases a <;> simp [projectedRunPre3572, lagInfoPre3572, compactBuildPre3572, checks, usesAT,
    checkGraphAT, checkGraph, cachedAT, newOrient, h1]

/-- **BUG #3349 (fixed by #3572, 5029c3c32)**: before the fix a one-node projection made a
1×1 matrix, which GraphBLAS holds by column, so `LAGraph_CheckGraph` returned -1000 and
the Rust caller returned the error. Live: `algo.WCC({nodeLabels:['Node']})` →
"LAGr_ConnectedComponents failed: -1000", `algo.pageRank('Node', null)` →
"LAGr_PageRank failed: -1000" (and `algo.betweenness({nodeLabels:['Node']})` →
"LAGr_Betweenness failed: -1000"); C answers all three. -/
theorem pre3572_one_node_projection_fails :
    projectedRunPre3572 .wcc [0] = .error "LAGr_ConnectedComponents failed: -1000" ∧
    projectedRunPre3572 .pagerank [0] = .error "LAGr_PageRank failed: -1000" ∧
    projectedRunPre3572 .betweenness [0] = .error "LAGr_Betweenness failed: -1000" ∧
    ¬ ProjectionTotal (projectedRunPre3572 .wcc) ∧
    ¬ ProjectionTotal (projectedRunPre3572 .pagerank) := by
  have hw : projectedRunPre3572 .wcc [0] = .error "LAGr_ConnectedComponents failed: -1000" := rfl
  have hp : projectedRunPre3572 .pagerank [0] = .error "LAGr_PageRank failed: -1000" := rfl
  refine ⟨hw, hp, rfl, fun h => ?_, fun h => ?_⟩
  · have := h [0] (by simp); rw [hw] at this; cases this
  · have := h [0] (by simp); rw [hp] at this; cases this

/-- Pinning only `G->A` would not have been enough: the cached 1×1 transpose is held by
column too, so pageRank and betweenness would still fail (why `cache_transpose` exists). -/
theorem at_unpinned_still_fails :
    checkGraphAT (compactBuild 1) (some (cachedAT (compactBuild 1))) = LAGRAPH_INVALID_GRAPH :=
  rfl

end AlgoUdf.LagContract
