import Pr3435Review.Clean
import Pr3435Review.PushPullFuel
import AlgoUdf.AlgoBfsReg
/-! # Claim 3: `algo_bfs`'s level-only `nodes` list does not depend on the path

PR branch 28ab35368, `graph/src/runtime/functions/algo_procedures.rs`:

| here | there |
| --- | --- |
| `clean`/`cleanB` | :1186-1195 — `clean_fwd` for a single type; `clean_bwd` only when `clean` is `Some` |
| `aPat`           | :1197-1202 — the lent base, else `build_adjacency_matrix` (graph.rs:4724; one type = `Tensor::extract`, graph_queries `Queries.adjMatrix_single`) |
| `manyExpected`   | :1209-1212 — `avg_degree^max_depth ≥ n/64` or unbounded; an **arbitrary** `Bool` here |
| `usePP`          | :1233 — `clean_bwd.filter(many_expected && !want_edges)` |
| `levelVec`       | :1233-1267 — `bfs_levels_push_pull(a, at, ..)` or `LAGr_BreadthFirstSearch_Extended(.., many_expected, ..)` |
| `entries`        | `extract_vector_i64(level)` :1306 (AXIOMATISED in algo_udf: ascending `(index, value)`) |
| `nodesOf`        | the level branch :1303-1318 = algo_udf `AlgoBfsReg.levelBranch` |
| `mainLevelVec`   | origin/main 9212add39 :1089-1130 — always LAGraph on `build_adjacency_matrix`, `many_expected = false` |

## FFI spec assumed (`LagSpec`)

`LAGr_BreadthFirstSearch_Extended(&level, NULL, G, src, max_level, -1, many_expected, msg)`
(LAGraph `experimental/algorithm/LAGr_BreadthFirstSearch_Extended.c` →
`LG_BreadthFirstSearch_SSGrB_template.c`): on success `level(j) = d` exactly for
the nodes at shortest-path distance `d ≤ max_level` (all reachable nodes when
`max_level < 0`) over `G->A`'s pattern (the `LAGraph_any_one_bool` semiring
reads no values, so a UINT64 or BOOL `A` with one pattern give one answer).
`many_expected` only selects `LG_SET_FORMAT_HINT(v, LG_BITMAP + LG_FULL)` (the
template, "if (many_expected)"), so the spec holds for both values. The C code
refuses `src ≥ n` (`GrB_INVALID_INDEX`).
-/
namespace Pr3435.Algo
open Pr3435.Bfs Pr3435.Clean VMTensorOps

/-- A stored pattern as an adjacency predicate. -/
def patOf (l : List (Nat × Nat)) : Pat := fun i j => decide ((i, j) ∈ l)

/-- LAGraph's level-only BFS, as a function of `A`'s pattern. -/
structure LagSpec (n : Nat) (lag : Pat → Nat → Int → Bool → Option LVec) : Prop where
  level : ∀ A src ml me w, Bounded A n → src < n → lag A src ml me = some w → Levels A src ml w

/-- `Levels` pins the vector down: two vectors with the BFS levels are equal. -/
theorem levels_unique {A : Pat} {src : Nat} {ml : Int} {w₁ w₂ : LVec}
    (h₁ : Levels A src ml w₁) (h₂ : Levels A src ml w₂) : w₁ = w₂ := by
  funext j
  cases hw : w₁ j with
  | none =>
    cases hw' : w₂ j with
    | none => rfl
    | some l => have := (h₁ j l).2 ((h₂ j l).1 hw'); rw [hw] at this; exact absurd this (by simp)
  | some l => exact ((h₂ j l).2 ((h₁ j l).1 hw)).symm

/-- `extract_vector_i64(level)`: ascending `(index, value)` entries. -/
def entries (n : Nat) (w : LVec) : List (Nat × Nat) :=
  (List.range n).filterMap (fun j => (w j).map (fun l => (j, l.toNat)))

/-- The level branch (:1303-1318), owner's `levelBranch`. -/
def nodesOf (n src : Nat) (hasDel : Bool) (del : Nat → Bool) (w : LVec) : List Nat :=
  AlgoUdf.AlgoBfsReg.levelBranch src hasDel del (entries n w)

/-- The relationship-type input: one type that exists, with its tensor (forward
`TT` and backward `VM`) and `build_adjacency_matrix`'s pattern; or anything else
(no type, several, unknown), where only `build_adjacency_matrix` is used. -/
inductive RelSel
  | single (t : TT) (mt : VM.VM)
  | other (adj : Pat)

/-- :1186-1189. -/
def clean : RelSel → Option (List (Nat × Nat))
  | .single t _ => cleanFwd t
  | .other _ => none

/-- :1190-1195 (`[t] if clean.is_some()`). -/
def cleanB : RelSel → Option (List (Nat × Nat))
  | .single t mt => if (cleanFwd t).isSome then cleanBwd mt else none
  | .other _ => none

/-- `build_adjacency_matrix(&rel_types)` (graph.rs:4724-4746). -/
def buildAdj : RelSel → Pat
  | .single t _ => patOf (extract t)
  | .other adj => adj

/-- :1197-1202: the lent base when clean, else the materialized copy. -/
def aPat (r : RelSel) : Pat :=
  match clean r with
  | some a => patOf a
  | none => buildAdj r

/-- :1233 — the push/pull path is taken iff a clean transpose exists, the
traversal is expected large, and edges are not wanted. -/
def usePP (r : RelSel) (manyExpected wantEdges : Bool) : Bool :=
  (cleanB r).isSome && manyExpected && !wantEdges

/-- :1226-1267: the level vector, either path. `pp` is `bfs_levels_push_pull`
(given `A`, `AT`), `lag` LAGraph's BFS. -/
def levelVec (pp : Pat → Pat → Except String LVec) (lag : Pat → Nat → Int → Bool → Option LVec)
    (r : RelSel) (src : Nat) (ml : Int) (manyExpected : Bool) : Except String LVec :=
  match cleanB r with
  | some b =>
    -- `many_expected && !want_edges`, in the level-only branch (`want_edges = false`)
    if manyExpected && !(false : Bool) then pp (aPat r) (patOf b)
    else (match lag (aPat r) src ml manyExpected with
          | some w => .ok w | none => .error "LAGr_BreadthFirstSearch_Extended failed")
  | none =>
    match lag (aPat r) src ml manyExpected with
    | some w => .ok w
    | none => .error "LAGr_BreadthFirstSearch_Extended failed"

/-- origin/main: always LAGraph on `build_adjacency_matrix`, `many_expected = false`. -/
def mainLevelVec (lag : Pat → Nat → Int → Bool → Option LVec) (r : RelSel) (src : Nat) (ml : Int) :
    Except String LVec :=
  match lag (buildAdj r) src ml false with
  | some w => .ok w
  | none => .error "LAGr_BreadthFirstSearch_Extended failed"

/-- The lent base has `build_adjacency_matrix`'s pattern (claim 1). -/
theorem aPat_eq_buildAdj (r : RelSel) : aPat r = buildAdj r := by
  cases r with
  | other adj => rfl
  | single t mt =>
    unfold aPat
    cases h : clean (.single t mt) with
    | none => rfl
    | some a =>
      simp only [clean] at h
      simp only [buildAdj, cleanFwd_eq_extract h]

/-- What `algo_bfs` needs from the stored tensor for the push/pull path: `mt`
represents the tensor (versioned_matrix's layer abstraction) and the tensor is
reachable through its own mutators (`Reach`, so `AllInv` holds). -/
def TensorOk : RelSel → Prop
  | .single t mt => Reach t ∧ MtRepr t mt
  | .other _ => True

/-- The lent backward base is the transpose of the forward one. -/
theorem cleanB_transpose {r : RelSel} (hok : TensorOk r) {b : List (Nat × Nat)}
    (hb : cleanB r = some b) : ∀ i j, patOf b j i = aPat r i j := by
  cases r with
  | other adj => simp [cleanB] at hb
  | single t mt =>
    simp only [cleanB] at hb
    split at hb
    · rename_i hc
      obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 hc
      intro i j
      have := clean_transpose_reach hok.1 ha hb hok.2 i j
      simp only [patOf, aPat, clean, ha]
      exact decide_eq_decide.2 this
    · exact absurd hb (by simp)

/-- **Claim 3 (level vector).** Under the FFI specs, the tensor premise, a
bounded adjacency over `n` nodes, `src < n` and `n < 2^31`: whatever
`many_expected` is and whichever path runs (push/pull on the lent bases, or
LAGraph on the lent base or the extracted copy), a successful level vector is
exactly the BFS levels of `build_adjacency_matrix` — the same vector origin/main
computes. The heuristic inside push/pull is arbitrary (`pushPull_correct`). -/
theorem levelVec_correct {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) (h0 : H)
    {lag : Pat → Nat → Int → Bool → Option LVec} (hl : LagSpec n lag)
    (r : RelSel) (hok : TensorOk r) (hb : Bounded (buildAdj r) n) {src : Nat} (hs : src < n)
    (hn : n < 2147483648) (ml : Int) (me : Bool) {w : LVec}
    (hw : levelVec (fun A AT => bfsLevelsPushPull g hr h0 A AT n src ml) lag r src ml me = .ok w) :
    Levels (buildAdj r) src ml w := by
  have hA := aPat_eq_buildAdj r
  unfold levelVec at hw
  split at hw
  · rename_i b hcb
    split at hw
    · have hT := cleanB_transpose hok hcb
      rw [hA] at hw hT
      exact (pushPull_correct hg hr h0 hT hb hs hn ml).1 w hw
    · split at hw
      · rename_i w' hw'
        cases hw
        rw [hA] at hw'
        exact hl.level _ _ _ _ _ hb hs hw'
      · cases hw
  · split at hw
    · rename_i w' hw'
      cases hw
      rw [hA] at hw'
      exact hl.level _ _ _ _ _ hb hs hw'
    · cases hw

/-- **Claim 3.** The level-only `nodes` list (every node with a level, minus the
source, minus deleted nodes) is the same for every `many_expected`, every path,
and equals origin/main's. -/
theorem nodes_path_independent {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) (h0 : H)
    {lag : Pat → Nat → Int → Bool → Option LVec} (hl : LagSpec n lag)
    (r : RelSel) (hok : TensorOk r) (hb : Bounded (buildAdj r) n) {src : Nat} (hs : src < n)
    (hn : n < 2147483648) (ml : Int) (me₁ me₂ : Bool) (hasDel : Bool) (del : Nat → Bool)
    {w₁ w₂ : LVec}
    (h₁ : levelVec (fun A AT => bfsLevelsPushPull g hr h0 A AT n src ml) lag r src ml me₁ = .ok w₁)
    (h₂ : levelVec (fun A AT => bfsLevelsPushPull g hr h0 A AT n src ml) lag r src ml me₂ = .ok w₂) :
    nodesOf n src hasDel del w₁ = nodesOf n src hasDel del w₂ := by
  rw [levels_unique (levelVec_correct hg hr h0 hl r hok hb hs hn ml me₁ h₁)
    (levelVec_correct hg hr h0 hl r hok hb hs hn ml me₂ h₂)]

/-- …and equals origin/main's answer (LAGraph on the copy, `many_expected = false`). -/
theorem nodes_same_as_main {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) (h0 : H)
    {lag : Pat → Nat → Int → Bool → Option LVec} (hl : LagSpec n lag)
    (r : RelSel) (hok : TensorOk r) (hb : Bounded (buildAdj r) n) {src : Nat} (hs : src < n)
    (hn : n < 2147483648) (ml : Int) (me : Bool) (hasDel : Bool) (del : Nat → Bool)
    {w w₀ : LVec}
    (h : levelVec (fun A AT => bfsLevelsPushPull g hr h0 A AT n src ml) lag r src ml me = .ok w)
    (h₀ : mainLevelVec lag r src ml = .ok w₀) :
    nodesOf n src hasDel del w = nodesOf n src hasDel del w₀ := by
  have L₀ : Levels (buildAdj r) src ml w₀ := by
    unfold mainLevelVec at h₀
    split at h₀
    · rename_i w' hw'; cases h₀; exact hl.level _ _ _ _ _ hb hs hw'
    · cases h₀
  rw [levels_unique (levelVec_correct hg hr h0 hl r hok hb hs hn ml me h) L₀]

/-- `usePP` is what `levelVec` dispatches on when edges are not wanted. -/
theorem usePP_iff (r : RelSel) (me : Bool) :
    usePP r me false = true ↔ (cleanB r).isSome = true ∧ me = true := by
  unfold usePP; cases (cleanB r).isSome <;> cases me <;> simp

/-! ## `src ≥ n`: the one input where the two paths would differ

LAGraph refuses `src ≥ n` (`GrB_INVALID_INDEX`, so `algo_bfs` returns `Err`),
while `bfs_levels_push_pull` ignores `GrB_Vector_setElement`'s failure and starts
from empty vectors: its first step yields an empty frontier and it returns `Ok`
with no levels (so `algo_bfs` returns the empty batch). Both paths and main use
the same `n` (the tensor's `nrows`, = `node_cap`), and a live node's id is below
`node_cap` (graph.rs `resize`: every node matrix and tensor is resized to
`node_cap` whenever an id is issued past it), so this input is not reachable;
the premise `src < n` above names it. -/
theorem pushPull_src_out_of_range {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H)
    (h0 : H) (A AT : Pat) {src : Nat} (hs : n ≤ src) (ml : Int) (hml : ¬ (0 ≤ ml ∧ ml < 1))
    (htot : ∀ q v, (g.vxm q v A).isSome ∧ (g.mxv q v AT).isSome) :
    ∃ w, bfsLevelsPushPull g hr h0 A AT n src ml = .ok w ∧ ∀ j, w j = none := by
  unfold bfsLevelsPushPull
  simp only [run]
  rw [if_neg (by simpa [init] using hml)]
  generalize hp : hr.choose (init n src h0).h (init n src h0).nq (init n src h0).lastNq
    (init n src h0).push = push
  have hv : ∀ j, (init n src h0).v j = none := by intro j; simp [init]; omega
  have hq : ∀ j, (init n src h0).q j = false := by intro j; simp [init]; omega
  generalize hr' : (if push then g.vxm (init n src h0).q (init n src h0).v A
      else g.mxv (init n src h0).q (init n src h0).v AT) = res
  cases res with
  | none =>
    cases push
    · have := (htot (init n src h0).q (init n src h0).v).2; simp_all
    · have := (htot (init n src h0).q (init n src h0).v).1; simp_all
  | some q' =>
    have hz : ∀ j, q' j = false := by
      intro j
      cases hj : q' j
      · rfl
      · cases push
        · simp only [Bool.false_eq_true, ite_false] at hr'
          obtain ⟨_, _, i, _, _, hqi⟩ := (hg.mxv _ _ _ _ hr' j).1 hj
          rw [hq i] at hqi; exact absurd hqi (by simp)
        · simp only [ite_true] at hr'
          obtain ⟨_, _, i, _, hqi, _⟩ := (hg.vxm _ _ _ _ hr' j).1 hj
          rw [hq i] at hqi; exact absurd hqi (by simp)
    have h0' : g.nvals q' = 0 := by
      rw [hg.nvals]; exact (count_eq_zero_iff n q').2 (fun j _ => hz j)
    simp only [h0', ite_true]
    exact ⟨_, rfl, hv⟩

end Pr3435.Algo
