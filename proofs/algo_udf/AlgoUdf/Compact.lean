/-! # Node id ↔ matrix index mapping in the LAGraph-backed procedures

| here | there (`graph/src/runtime/functions/algo_procedures.rs`) |
| --- | --- |
| `idx`, `ids`            | `build_compact_adj_from_tensors` :570 / `..._symmetric_...` :636: `sorted_ids` (compact → id) and `id_to_compact` |
| `compactEdges`          | the tensor loops in the same two functions (edge kept iff both ends are active) |
| `LagWcc`                | AXIOMATISED spec of `LAGr_ConnectedComponents` / `LAGraph_cdlp` output |
| `wccRows`               | algo.WCC result assembly :911-924 (algo.labelPropagation :1327-1339 is identical) |
| `unfilteredRows`        | the `compact_to_id = None` branch of every procedure + `is_node_deleted` skip |
| `harmonicRows`          | algo.HarmonicCentrality :2829-2878 |
| `pagerankPath`          | algo.pageRank early exits :729-735, `use_unfiltered` :750-752, empty active set :772-774 (#3058) |
| `checkKnown`            | `check_known_labels_and_types` :469 (betweenness :973, labelPropagation :1268, HarmonicCentrality :2759) |
| `seed0Sources`          | algo.betweenness source sampling with `samplingSeed = 0` :1001-1021 |

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
pushed as-is (`component_ids.push(comp_id)`, :922). -/
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
`node_id_bound` (:2832-2834). A live id `≥ node_count` is dropped. -/
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

/-! ## Unknown labels / relationship types (#3058, fixes #3056) -/

/-- `check_known_labels_and_types` (:469): the first label the graph does not have, else
the first relationship type it does not have, becomes the procedure's own error. -/
def checkKnown (hasLabel hasType : String → Bool) (labels types : List String)
    (unkL unkT : String → String) : Except String Unit :=
  match labels.find? (fun l => !hasLabel l) with
  | some l => .error (unkL l)
  | none =>
    match types.find? (fun t => !hasType t) with
    | some t => .error (unkT t)
    | none => .ok ()

/-- Accepts exactly when every named label and relationship type exists. -/
theorem checkKnown_ok_iff (hasLabel hasType : String → Bool) (labels types : List String)
    (unkL unkT : String → String) :
    checkKnown hasLabel hasType labels types unkL unkT = .ok () ↔
      (∀ l ∈ labels, hasLabel l = true) ∧ (∀ t ∈ types, hasType t = true) := by
  unfold checkKnown
  cases hl : labels.find? (fun l => !hasLabel l) with
  | some l =>
    have := List.find?_some hl
    have hm := List.mem_of_find?_eq_some hl
    simp only [Except.error.injEq, reduceCtorEq, false_iff, not_and]
    intro h
    have := h _ hm
    simp_all
  | none =>
    have hl' := List.find?_eq_none.mp hl
    cases ht : types.find? (fun t => !hasType t) with
    | some t =>
      have := List.find?_some ht
      have hm := List.mem_of_find?_eq_some ht
      simp only [reduceCtorEq, false_iff, not_and]
      intro _ h
      have := h _ hm
      simp_all
    | none =>
      have ht' := List.find?_eq_none.mp ht
      simp only [true_iff]
      refine ⟨fun l hm => ?_, fun t hm => ?_⟩
      · have := hl' l hm; simpa using this
      · have := ht' t hm; simpa using this

/-- **#3056 fixed**: an unknown label is an error, never an empty result. -/
theorem checkKnown_unknown_label_errors (hasLabel hasType : String → Bool) (labels types : List String)
    (unkL unkT : String → String) (l : String) (hm : l ∈ labels) (hn : hasLabel l = false) :
    ∃ m, checkKnown hasLabel hasType labels types unkL unkT = .error m := by
  cases h : checkKnown hasLabel hasType labels types unkL unkT with
  | error m => exact ⟨m, rfl⟩
  | ok u =>
    cases u
    have := (checkKnown_ok_iff hasLabel hasType labels types unkL unkT).mp h
    have := this.1 l hm
    simp_all

theorem checkKnown_unknown_type_errors (hasLabel hasType : String → Bool) (labels types : List String)
    (unkL unkT : String → String) (t : String) (hm : t ∈ types) (hn : hasType t = false) :
    ∃ m, checkKnown hasLabel hasType labels types unkL unkT = .error m := by
  cases h : checkKnown hasLabel hasType labels types unkL unkT with
  | error m => exact ⟨m, rfl⟩
  | ok u =>
    cases u
    have := (checkKnown_ok_iff hasLabel hasType labels types unkL unkT).mp h
    have := this.2 t hm
    simp_all

/-- A label error wins over a type error (labels are checked first, :476-481). -/
theorem checkKnown_label_first :
    checkKnown (fun l => l == "A") (fun t => t == "R") ["B"] ["S"] (fun l => "label " ++ l)
      (fun t => "type " ++ t) = .error "label B" := rfl

/-- The per-procedure messages (C's): betweenness :973-978, labelPropagation :1268-1273,
HarmonicCentrality :2759-2765. -/
theorem known_messages :
    checkKnown (fun _ => false) (fun _ => true) ["Nope"] []
      (fun l => s!"betweenness configuration, unknown label {l}") (fun t => t) =
        .error "betweenness configuration, unknown label Nope" ∧
    checkKnown (fun _ => true) (fun _ => false) [] ["R"] (fun l => l)
      (fun t => s!"labelPropagation configuration, unknown relationship-type {t}") =
        .error "labelPropagation configuration, unknown relationship-type R" ∧
    checkKnown (fun _ => false) (fun _ => true) ["Nope"] []
      (fun l => s!"harmonic centrality configuration contains non-existent label:{l}") (fun t => t) =
        .error "harmonic centrality configuration contains non-existent label:Nope" :=
  ⟨rfl, rfl, rfl⟩

/-! ## pageRank fast-path choice -/

/-- What algo.pageRank does after the early exits (:729-780). -/
inductive PRPath
  | empty                 -- `empty_procedure_batch()`
  | unfiltered
  | compact (n : Nat)     -- `build_compact_adj_from_tensors` over `n` active nodes
  deriving DecidableEq, Repr

/-- algo.pageRank's path choice on main. `label`: `none` no label argument, `some none` a
label the graph does not have, `some (some c)` a label with `c` live nodes; `relOk`: the
relationship type is absent or known. Returns the 0-node / unknown label / unknown type
early exits (:729-735), the `use_unfiltered` test (:750-752), and the empty-active-set
exit (:772-774). -/
def pagerankPath (nodeCount : Nat) (label : Option (Option Nat)) (relOk : Bool) : PRPath :=
  if nodeCount = 0 || label == some none || !relOk then .empty
  else match label with
    | none => .unfiltered
    | some none => .empty
    | some (some c) => if c = nodeCount then .unfiltered else if c = 0 then .empty else .compact c

/-- **#3056 fixed**: an unknown label or relationship type selects nothing: an empty
result, no LAGraph call. -/
theorem pagerank_unknown_label_empty (nc : Nat) (relOk : Bool) :
    pagerankPath nc (some none) relOk = .empty := by
  unfold pagerankPath; split <;> simp_all

theorem pagerank_unknown_type_empty (nc : Nat) (label : Option (Option Nat)) :
    pagerankPath nc label false = .empty := by
  unfold pagerankPath; simp

/-- LAGraph is never handed an empty matrix: the compact path always has `n > 0`. -/
theorem pagerank_compact_nonempty (nc : Nat) (label : Option (Option Nat)) (relOk : Bool) (n : Nat)
    (h : pagerankPath nc label relOk = .compact n) : 0 < n := by
  unfold pagerankPath at h
  split at h
  · cases h
  · split at h <;> try cases h
    rename_i c
    split at h
    · cases h
    · split at h
      · cases h
      · cases h; omega

/-- A known label covering everything is the unfiltered path, as before. -/
theorem pagerank_known_label_full (nc : Nat) (h : 0 < nc) :
    pagerankPath nc (some (some nc)) true = .unfiltered := by
  unfold pagerankPath; simp; omega

/-- The case #3056 reported: node_count 3, label unknown. -/
theorem pagerank_unknown_label_example : pagerankPath 3 (some none) true = .empty := by decide

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
