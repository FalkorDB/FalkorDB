/-! # Node id ↔ matrix index mapping in the LAGraph-backed procedures

| here | there (`graph/src/runtime/functions/algo_procedures.rs`) |
| --- | --- |
| `idx`, `ids`            | `build_compact_adj_from_tensors` :552 / `..._symmetric_...` :618: `sorted_ids` (compact → id) and `id_to_compact` |
| `compactEdges`          | the tensor loops in the same two functions (edge kept iff both ends are active) |
| `LagWcc`                | AXIOMATISED spec of `LAGr_ConnectedComponents` / `LAGraph_cdlp` output |
| `wccRows`               | algo.WCC result assembly :884-897 (algo.labelPropagation :1293-1305 is identical) |
| `unfilteredRows`        | the `compact_to_id = None` branch of every procedure + `is_node_deleted` skip |
| `harmonicRows`          | algo.HarmonicCentrality :2777-2826 |
| `pagerankPath`          | algo.pageRank `use_unfiltered` :727-729 |
| `seed0Sources`          | algo.betweenness source sampling with `samplingSeed = 0` :974-994 |

LAGraph is not modelled; its output contract is a hypothesis (`LagWcc`), citing
LAGraph's `LAGr_ConnectedComponents` doc: "component(i) = r where r is the
representative (smallest index) of i's component, over indices 0..n-1".
-/
namespace AlgoUdf.Compact

/-- `id_to_compact`: position of `x` in the sorted id list. -/
def idx (ids : List Nat) (x : Nat) : Option Nat := ids.idxOf? x

/-- Compact edge list: an edge survives iff both endpoints are active, and is
renumbered through `idx`. -/
def compactEdges (ids : List Nat) (es : List (Nat × Nat)) : List (Nat × Nat) :=
  es.filterMap fun (s, d) => match idx ids s, idx ids d with
    | some a, some b => some (a, b)
    | _, _ => none

theorem compactEdges_sound (ids : List Nat) (es : List (Nat × Nat)) (a b : Nat) :
    (a, b) ∈ compactEdges ids es →
      ∃ s d, (s, d) ∈ es ∧ idx ids s = some a ∧ idx ids d = some b := by
  simp only [compactEdges, List.mem_filterMap]
  rintro ⟨⟨s, d⟩, hm, h⟩
  refine ⟨s, d, hm, ?_⟩
  revert h
  cases hs : idx ids s <;> cases hd : idx ids d <;> simp

/-- The output contract assumed of LAGraph's WCC/CDLP on an `n`-vertex graph:
a dense vector whose i-th entry is a vertex index `< n` (the representative). -/
structure LagWcc (n : Nat) (comp : Nat → Nat) : Prop where
  lt : ∀ i, i < n → comp i < n

/-- Rust's rows on the compact path: node mapped through `ids`, component id
pushed as-is (`component_ids.push(comp_id)`, :895). -/
def wccRows (ids : List Nat) (comp : Nat → Nat) : List (Nat × Nat) :=
  (List.range ids.length).map fun i => (ids.getD i 0, comp i)

/-- What C (and Rust's own unfiltered path) report: the component id is a node id. -/
def wccSpec (ids : List Nat) (comp : Nat → Nat) : List (Nat × Nat) :=
  (List.range ids.length).map fun i => (ids.getD i 0, ids.getD (comp i) 0)

/-- On the unfiltered path the mapping is the identity, so the two agree. -/
theorem unfiltered_component_ids_are_members (n : Nat) (comp : Nat → Nat)
    (h : LagWcc n comp) : wccRows (List.range n) comp = wccSpec (List.range n) comp := by
  simp only [wccRows, wccSpec, List.length_range]
  apply List.map_congr_left
  intro i hi
  have hi' := List.mem_range.mp hi
  simp [List.getD_eq_getElem?_getD, List.getElem?_range, hi', h.lt i hi']

/-- The spec's component id is always a reported node. -/
theorem spec_component_is_member (ids : List Nat) (comp : Nat → Nat)
    (h : LagWcc ids.length comp) :
    ∀ p ∈ wccSpec ids comp, p.2 ∈ ids := by
  intro p hp
  simp only [wccSpec, List.mem_map, List.mem_range] at hp
  obtain ⟨i, hi, rfl⟩ := hp
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (h.lt i hi)]

/-- BUG (confirmed, `bug_wcc_label_filter_component_ids_are_compact_indices`):
graph of `LABELLED` with nodeLabels:['A'] → active ids [0,2,3], edge 2–3; LAGraph
returns representatives [0,1,1]; Rust reports component `1` for nodes 2 and 3,
and 1 is not an :A node. -/
theorem wcc_compact_component_id_not_member :
    let ids := [0, 2, 3]
    let comp := fun i => if i = 0 then 0 else 1
    LagWcc ids.length comp ∧
    wccRows ids comp = [(0, 0), (2, 1), (3, 1)] ∧
    wccSpec ids comp = [(0, 0), (2, 2), (3, 2)] ∧
    (1 : Nat) ∉ ids := by
  refine ⟨⟨fun i hi => ?_⟩, by decide, by decide, by decide⟩
  simp at hi ⊢; split <;> omega

/-! ## Deleted ids on the unfiltered path -/

/-- Rows kept by the unfiltered path: every matrix index `< dim` that LAGraph
reports, minus deleted ids. -/
def unfilteredRows (dim : Nat) (alive : Nat → Bool) : List Nat :=
  (List.range dim).filter alive

/-- With `dim = node_count + deleted_count` (= `node_id_bound`) every live node is
reported exactly once, in id order. -/
theorem unfiltered_rows_exact (bound : Nat) (alive : Nat → Bool) (x : Nat) :
    x ∈ unfilteredRows bound alive ↔ x < bound ∧ alive x = true := by
  simp [unfilteredRows]

theorem unfiltered_rows_nodup (bound : Nat) (alive : Nat → Bool) :
    (unfilteredRows bound alive).Nodup :=
  (List.nodup_range).filter _

/-- HarmonicCentrality's unfiltered source vector has length `node_count`, not
`node_id_bound` (:2777-2779). A live id `≥ node_count` is dropped. -/
def harmonicRows (nodeCount : Nat) (alive : Nat → Bool) : List Nat :=
  unfilteredRows nodeCount alive

theorem harmonic_correct_iff (nodeCount bound : Nat) (alive : Nat → Bool)
    (hb : ∀ x, alive x = true → x < bound) :
    (∀ x, x ∈ harmonicRows nodeCount alive ↔ x ∈ unfilteredRows bound alive) ↔
      (∀ x, alive x = true → x < nodeCount) := by
  simp only [harmonicRows, unfiltered_rows_exact]
  constructor
  · intro h x hx; exact ((h x).mpr ⟨hb x hx, hx⟩).1
  · intro h x; exact ⟨fun ⟨_, a⟩ => ⟨hb x a, a⟩, fun ⟨_, a⟩ => ⟨h x a, a⟩⟩

/-- BUG (confirmed, `bug_harmonic_centrality_with_deleted_node`): ids 0..4, node 0
deleted: node_count = 4, bound = 5, node 4 is never a source nor reported. -/
theorem unfiltered_drops_high_ids :
    let alive := fun x => decide (1 ≤ x ∧ x ≤ 4)
    harmonicRows 4 alive = [1, 2, 3] ∧ unfilteredRows 5 alive = [1, 2, 3, 4] := by
  decide

/-! ## pageRank fast-path choice -/

/-- `use_unfiltered`: no label, or the label covers all nodes. `none` = compact
path with that many active nodes. -/
def pagerankPath (label : Option Nat) (nodeCount : Nat) : Option Nat :=
  match label with
  | none => none
  | some cnt => if cnt = nodeCount then none else some cnt

/-- BUG (confirmed, `bug_pagerank_unknown_label_errors`): an unknown label has
`label_node_count = 0 ≠ node_count`, so a 0×0 compact matrix reaches
`LAGr_PageRank`, which rejects it (LAGraph doc: n must be > 0). -/
theorem pagerank_unknown_label_compact_empty (nc : Nat) (h : 0 < nc) :
    pagerankPath (some 0) nc = some 0 := by
  simp [pagerankPath]; omega

/-! ## betweenness sources -/

/-- `samplingSeed = 0`: sources are `i % n_nodes` for i < samplingSize, deduplicated. -/
def seed0Sources (s n : Nat) : List Nat :=
  if n = 0 then [] else if s ≥ n then List.range n else (List.range s).map (· % n)

theorem seed0_sources_lt (s n : Nat) : ∀ x ∈ seed0Sources s n, x < n := by
  intro x hx
  unfold seed0Sources at hx
  split at hx
  · simp at hx
  · split at hx
    · exact List.mem_range.mp hx
    · simp at hx; obtain ⟨a, _, rfl⟩ := hx; exact Nat.mod_lt _ (by omega)

/-- On the unfiltered path the index space includes deleted slots, so a sampled
source can be a deleted id (it then contributes nothing). With ids 0..4, node 0
deleted and `samplingSize: 2`, the sources are {0, 1}: half the budget is wasted.
C samples the same way (suspicion, not a divergence). -/
theorem seed0_samples_deleted : seed0Sources 2 5 = [0, 1] := by decide

end AlgoUdf.Compact
