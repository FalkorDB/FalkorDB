import AlgoUdf.Compact
/-! # LAGraph's graph-validity outcome on the compact path (#3349) — gap closed 2026-10-20

`Compact.LagWcc` (and the `LAGr_PageRank` / `LAGr_Betweenness` assumptions) took LAGraph
to succeed on every square matrix. It does not: `LAGraph_CheckGraph` rejects a matrix
held by column with `LAGRAPH_INVALID_GRAPH` (-1000), and GraphBLAS holds every
`n`-by-1 matrix by column — including the 1-by-1 one the compact builders make for a
one-node projection. This file adds that outcome to the axiomatised contract.

| here | there |
| --- | --- |
| `newOrient`   | GraphBLAS `GxB_Matrix_new_arena` (`deps/GraphBLAS` ea79e35, `Source/matrix/GxB_Matrix_new_arena.c:89-101`): `ncols == 1` → by column; `nrows == 1` → by row; else the global format (by row) |
| `checkGraph`  | `LAGraph_CheckGraph` (`deps/LAGraph` d01064d, `src/utility/LAGraph_CheckGraph.c:46-56`): non-square or not `GrB_ROWMAJOR` → `LAGRAPH_INVALID_GRAPH` |
| `Algo`, `checks` | `LAGr_ConnectedComponents` → `LG_CC_FastSV6/7` (`CheckGraph` at :263/:279), `LAGr_PageRank` (:76), `LAGr_Betweenness` (:131) check; `LAGraph_cdlp` does not |
| `compactBuild` | `build_compact_adj_from_tensors` `:627` / `..._symmetric_...` `:697`: `GrB_Matrix_new(&compact, GrB_BOOL, n, n)` with `n` = projection size |
| `projectedRun` | `algo_wcc` `:888-908`, `algo_pagerank` `:766-796`, `algo_betweenness` `:1001-1053`: `if info != 0 { return Err(format!("… failed: {info}")) }` — no special case for `n = 1` |

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

/-- The compact matrix for a projection of `n` nodes: `n × n`, orientation from
`GrB_Matrix_new`. -/
def compactBuild (n : Nat) : Nat × Nat × Orient := (n, n, newOrient n n)

/-- The LAGraph call's return code on the compact matrix. -/
def lagInfo (a : Algo) (n : Nat) : Int :=
  let (r, c, o) := compactBuild n
  if checks a then checkGraph r c o else 0

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

/-- The contract is honest about square matrices of size ≥ 2: they pass. -/
theorem lagInfo_two_le (a : Algo) (n : Nat) (h : 2 ≤ n) : lagInfo a n = 0 := by
  have h1 : n ≠ 1 := by omega
  cases a <;> simp [lagInfo, compactBuild, checks, checkGraph, newOrient, h1]

theorem projectedRun_two_le (a : Algo) (ids : List Nat) (h : 2 ≤ ids.length) :
    projectedRun a ids = .ok ids.length := by
  simp [projectedRun, lagInfo_two_le a _ h]

/-- CDLP does not run `LAGraph_CheckGraph`, so it satisfies the property (live: Rust
`algo.labelPropagation({nodeLabels:['Node']})` returns `a, 0`). -/
theorem cdlp_total : ProjectionTotal (projectedRun .cdlp) := by
  intro ids _; simp [projectedRun, lagInfo, checks]

/-- **BUG #3349**: a one-node projection makes a 1×1 matrix, which GraphBLAS holds by
column, so `LAGraph_CheckGraph` returns -1000 and the Rust caller returns the error.
Live: `algo.WCC({nodeLabels:['Node']})` → "LAGr_ConnectedComponents failed: -1000",
`algo.pageRank('Node', null)` → "LAGr_PageRank failed: -1000" (and
`algo.betweenness({nodeLabels:['Node']})` → "LAGr_Betweenness failed: -1000"); C answers
all three. -/
theorem bug3349_one_node_projection_fails :
    projectedRun .wcc [0] = .error "LAGr_ConnectedComponents failed: -1000" ∧
    projectedRun .pagerank [0] = .error "LAGr_PageRank failed: -1000" ∧
    projectedRun .betweenness [0] = .error "LAGr_Betweenness failed: -1000" ∧
    ¬ ProjectionTotal (projectedRun .wcc) ∧ ¬ ProjectionTotal (projectedRun .pagerank) := by
  have hw : projectedRun .wcc [0] = .error "LAGr_ConnectedComponents failed: -1000" := rfl
  have hp : projectedRun .pagerank [0] = .error "LAGr_PageRank failed: -1000" := rfl
  refine ⟨hw, hp, rfl, fun h => ?_, fun h => ?_⟩
  · have := h [0] (by simp); rw [hw] at this; cases this
  · have := h [0] (by simp); rw [hp] at this; cases this

end AlgoUdf.LagContract
