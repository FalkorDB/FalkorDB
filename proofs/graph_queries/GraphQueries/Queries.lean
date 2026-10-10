import GraphQueries.Ids
/-
# Read-side queries

| here | there (graph.rs) |
| --- | --- |
| `nodeRels`        | `get_node_relationships` :2214 |
| `nodeRelsByType`  | `get_node_relationships_by_type` :2296 |
| `inDeg`/`outDeg`/`..ByType` | `get_node_indegree` :2338 / `_by_type` :2350 / `get_node_outdegree` :2364 / `_by_type` :2376 |
| `getNodesAll`/`getNodesOne`/`getNodesMany` | the three arms of `get_nodes` :2389 |
| `nodeLabelIds`/`nodeLabelNames` | `get_node_label_ids` :2454 / `get_node_labels` :2463 |
| `srcDestRels`     | `get_src_dest_relationships` :3072 |
| `relMatUnrestricted` | `build_relationship_matrix_unrestricted` :3096 (`set_pattern` = pattern union) |
| `resolveLabelIds` | `resolve_label_ids` :3130 |
| `relMatrix`       | `build_relationship_matrix` :3140 (`rmxm`/`lmxm` by a label diagonal = row / column restriction) |
| `getRelationships`| `get_relationships` :3208 |
| `relTypeId`       | `get_relationship_type_id` :3224 (`expect` = `none`) |
| `relEndpoints`    | `endpoints_for_edge` :3240 / `get_relationship_endpoints` :3258 |
| `relTypeIter`     | `relationship_type_matrix_iter` :3272 |
| `allEdges`        | `get_all_edges` :3926 |
| `adjMatrix`/`symAdj` | `build_adjacency_matrix` :4490 / `build_symmetric_adjacency_matrix` :4518 |
-/
namespace GQ
variable {V : Type}

/-! ## Incident edges -/

def nodeRels (g : G V) (n : Nat) : Ten := g.relMs.flatMap (fun t => Ten.out t n ++ Ten.inc t n)

inductive Dir | outgoing | incoming | both deriving DecidableEq

/-- `EdgeDirection::from_str` (graph.rs:158). -/
def dirOfStr : String → Option Dir
  | "outgoing" => some .outgoing | "incoming" => some .incoming | "both" => some .both | _ => none

theorem dirOfStr_spec : dirOfStr "outgoing" = some .outgoing ∧ dirOfStr "incoming" = some .incoming ∧
    dirOfStr "both" = some .both ∧ dirOfStr "Both" = none := by decide

def selMats (g : G V) (types : List String) : List Ten :=
  if types = [] then g.relMs else types.filterMap (getRelMat g)

/-- One tensor's arm of `get_node_relationships_by_type`: with both halves,
the incoming half drops `src == n` (the self-loops the outgoing half already
produced). -/
def oneByType (n : Nat) (d : Dir) (t : Ten) : Ten :=
  match d with
  | .outgoing => Ten.out t n
  | .incoming => Ten.inc t n
  | .both => Ten.out t n ++ (Ten.inc t n).filter (·.1 != n)

def nodeRelsByType (g : G V) (n : Nat) (types : List String) (d : Dir) : Ten :=
  (selMats g types).flatMap (oneByType n d)

/-- `get_node_relationships` reports a self-loop **twice** (once from the
outgoing scan, once from the incoming scan). -/
theorem nodeRels_selfloop_twice (t : Ten) (n e : Nat) (hnd : t.Nodup) (hm : (n, n, e) ∈ t) :
    (Ten.out t n ++ Ten.inc t n).count (n, n, e) = 2 := by
  simp only [Ten.out, Ten.inc, List.count_append]
  rw [List.count_filter (by simp), List.count_filter (by simp), hnd.count_of_mem hm]

/-- `get_node_relationships_by_type(.., Both)` reports every incident edge
exactly once, self-loops included. -/
theorem byType_both_once (t : Ten) (n : Nat) (x : Nat × Nat × Nat) (hnd : t.Nodup) (hm : x ∈ t)
    (hi : x.1 = n ∨ x.2.1 = n) : (oneByType n .both t).count x = 1 := by
  obtain ⟨s, d, e⟩ := x
  simp only at hi
  by_cases hs : s = n
  · subst hs
    simp only [oneByType, Ten.out, Ten.inc, List.count_append]
    rw [List.count_filter (by simp), hnd.count_of_mem hm, List.count_eq_zero.2 (by simp)]
  · have hd := hi.resolve_left hs; subst hd
    simp only [oneByType, Ten.out, Ten.inc, List.count_append]
    rw [List.count_eq_zero.2 (by simp [hs]), List.count_filter (by simp [hs]),
      List.count_filter (by simp), hnd.count_of_mem hm]

/-- Outgoing-only / incoming-only by type list = the plain out / in scans. -/
theorem byType_out (g : G V) (n : Nat) (types : List String) :
    nodeRelsByType g n types .outgoing = (selMats g types).flatMap (Ten.out · n) := by
  rfl
theorem byType_in (g : G V) (n : Nat) (types : List String) :
    nodeRelsByType g n types .incoming = (selMats g types).flatMap (Ten.inc · n) := by
  rfl

def inDeg (g : G V) (n : Nat) : Nat := (g.relMs.map (Ten.colDegree · n)).sum
def outDeg (g : G V) (n : Nat) : Nat := (g.relMs.map (Ten.rowDegree · n)).sum
def inDegByType (g : G V) (n : Nat) (types : List String) : Nat :=
  ((types.filterMap (getRelMat g)).map (Ten.colDegree · n)).sum
def outDegByType (g : G V) (n : Nat) (types : List String) : Nat :=
  ((types.filterMap (getRelMat g)).map (Ten.rowDegree · n)).sum

theorem sum_len {α} (f : α → List β) : ∀ (l : List α), (l.map (fun t => (f t).length)).sum = (l.flatMap f).length
  | [] => rfl
  | a :: l => by simp only [List.map_cons, List.sum_cons, List.flatMap_cons, List.length_append, sum_len f l]

theorem selMats_nil (g : G V) : selMats g [] = g.relMs := if_pos rfl
theorem selMats_ne (g : G V) (types : List String) (h : types ≠ []) :
    selMats g types = types.filterMap (getRelMat g) := if_neg h

/-- Degrees count exactly the edges the incident-edge scans enumerate. -/
theorem inDeg_eq (g : G V) (n : Nat) : inDeg g n = (nodeRelsByType g n [] .incoming).length := by
  rw [byType_in, selMats_nil]; exact sum_len (Ten.inc · n) g.relMs
theorem outDeg_eq (g : G V) (n : Nat) : outDeg g n = (nodeRelsByType g n [] .outgoing).length := by
  rw [byType_out, selMats_nil]; exact sum_len (Ten.out · n) g.relMs
theorem inDegByType_eq (g : G V) (n : Nat) (types : List String) (h : types ≠ []) :
    inDegByType g n types = (nodeRelsByType g n types .incoming).length := by
  rw [byType_in, selMats_ne g types h]; exact sum_len (Ten.inc · n) _
theorem outDegByType_eq (g : G V) (n : Nat) (types : List String) (h : types ≠ []) :
    outDegByType g n types = (nodeRelsByType g n types .outgoing).length := by
  rw [byType_out, selMats_ne g types h]; exact sum_len (Ten.out · n) _

/-! ## Node scans -/

def getNodesAll (g : G V) (lo : Nat) : List Nat :=
  if g.nodeCount = 0 then [] else
  ((List.range (maxNodeId g + 1)).filter (lo ≤ ·)).filter (fun n => n ∉ g.delNodes)

/-- Unfiltered `MATCH (n)`: the ids `[lo, max_node_id]` not in the bin. -/
theorem getNodesAll_mem (g : G V) (lo n : Nat) (h : g.nodeCount ≠ 0) :
    n ∈ getNodesAll g lo ↔ lo ≤ n ∧ n < nodeBound g ∧ n ∉ g.delNodes := by
  have := maxNodeId_lt g h
  simp only [getNodesAll, h, if_false, List.mem_filter, List.mem_range, decide_eq_true_eq]
  rw [this]
  constructor
  · rintro ⟨⟨a, b⟩, c⟩; exact ⟨b, a, c⟩
  · rintro ⟨a, b, c⟩; exact ⟨⟨b, a⟩, c⟩

theorem getNodesAll_empty (g : G V) (lo : Nat) (h : g.nodeCount = 0) : getNodesAll g lo = [] := by
  simp [getNodesAll, h]

def getNodesOne (g : G V) (lab : String) (lo : Nat) : List Nat :=
  match getLabelMat g lab with
  | some m => (m.rowsFrom lo).map (·.1)
  | none => []

/-- Several labels: the first matrix `eWiseMult`'d with the rest (pattern
intersection); a missing label gives the 0×0 zero matrix. -/
def getNodesMany (g : G V) (labs : List String) (lo : Nat) : List Nat :=
  match labs.mapM (getLabelMat g) with
  | some (m :: ms) => ((m.ents.filter (fun p => ms.all (fun m' => decide (p ∈ m'.ents)))).filter (lo ≤ ·.1)).map (·.1)
  | _ => []

theorem getNodesMany_mem (g : G V) (labs : List String) (lo n : Nat) (m : Mat) (ms : List Mat)
    (h : labs.mapM (getLabelMat g) = some (m :: ms)) :
    n ∈ getNodesMany g labs lo ↔ ∃ c, (n, c) ∈ m.ents ∧ (∀ m' ∈ ms, (n, c) ∈ m'.ents) ∧ lo ≤ n := by
  simp only [getNodesMany, h, List.mem_map, List.mem_filter, List.all_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨a, c⟩, ⟨⟨h1, h2⟩, h3⟩, rfl⟩; exact ⟨c, h1, h2, h3⟩
  · rintro ⟨c, h1, h2, h3⟩; exact ⟨(n, c), ⟨⟨h1, h2⟩, h3⟩, rfl⟩

theorem getNodesMany_missing (g : G V) (labs : List String) (lo : Nat)
    (h : labs.mapM (getLabelMat g) = none) : getNodesMany g labs lo = [] := by
  simp [getNodesMany, h]

def nodeLabelIds (g : G V) (n : Nat) : List Nat := (g.nodeLabels.row n).map (·.2)
def nodeLabelNames (g : G V) (n : Nat) : List String := (nodeLabelIds g n).map (g.labels[·]?.getD "")

theorem nodeLabelIds_mem (g : G V) (n l : Nat) : l ∈ nodeLabelIds g n ↔ (n, l) ∈ g.nodeLabels.ents := by
  simp [nodeLabelIds, Mat.row]

theorem nodeLabelNames_mem (g : G V) (n l : Nat) (s : String) (h : g.labels[l]? = some s)
    (hm : (n, l) ∈ g.nodeLabels.ents) : s ∈ nodeLabelNames g n := by
  simp only [nodeLabelNames, List.mem_map]
  exact ⟨l, (nodeLabelIds_mem g n l).2 hm, by simp [h]⟩

/-! ## Relationship matrices -/

def srcDestRels (g : G V) (s d : Nat) (types : List String) : List Nat :=
  ((if types = [] then g.types else types).filterMap (getRelMat g)).flatMap (Ten.pair · s d)

theorem srcDestRels_mem (g : G V) (s d e : Nat) (types : List String) :
    e ∈ srcDestRels g s d types ↔
    ∃ t ∈ (if types = [] then g.types else types), ∃ tn, getRelMat g t = some tn ∧ (s, d, e) ∈ tn := by
  simp only [srcDestRels, List.mem_flatMap, List.mem_filterMap, Ten.pair, List.mem_map,
    List.mem_filter, Bool.and_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨tn, ⟨t, ht, hg⟩, ⟨⟨s', d', e'⟩, ⟨hm, rfl, rfl⟩, rfl⟩⟩; exact ⟨t, ht, tn, hg, hm⟩
  · rintro ⟨t, ht, tn, hg, hm⟩; exact ⟨tn, ⟨t, ht, hg⟩, ⟨(s, d, e), ⟨hm, rfl, rfl⟩, rfl⟩⟩

def relBase (g : G V) (types : List String) : List (Nat × Nat) :=
  match types.filterMap (getRelMat g) with
  | [] => g.adj.ents
  | m :: rest => Ten.pattern m ++ rest.flatMap Ten.pattern

def relMatUnrestricted (g : G V) (types : List String) : Option (List (Nat × Nat)) :=
  if types ≠ [] ∧ types.filterMap (getRelMat g) = [] then none else some (relBase g types)

theorem relBase_mem (g : G V) (types : List String) (h : types.filterMap (getRelMat g) ≠ []) (s d : Nat) :
    (s, d) ∈ relBase g types ↔ ∃ tn ∈ types.filterMap (getRelMat g), ∃ e, (s, d, e) ∈ tn := by
  unfold relBase
  split
  · rename_i hms; exact absurd hms h
  · rename_i m rest hms
    rw [hms]
    simp only [Ten.pattern, List.mem_append, List.mem_map, List.mem_flatMap, List.mem_cons]
    constructor
    · rintro (⟨⟨a, b, e⟩, hm, he⟩ | ⟨tn, htn, ⟨a, b, e⟩, hm, he⟩) <;> cases he
      · exact ⟨m, Or.inl rfl, e, hm⟩
      · exact ⟨tn, Or.inr htn, e, hm⟩
    · rintro ⟨tn, (rfl | htn), e, hm⟩
      · exact Or.inl ⟨(s, d, e), hm, rfl⟩
      · exact Or.inr ⟨tn, htn, (s, d, e), hm, rfl⟩

theorem relMatUnrestricted_mem (g : G V) (types : List String) (h : types ≠ [])
    (r : List (Nat × Nat)) (hr : relMatUnrestricted g types = some r) (s d : Nat) :
    (s, d) ∈ r ↔ ∃ tn ∈ types.filterMap (getRelMat g), ∃ e, (s, d, e) ∈ tn := by
  simp only [relMatUnrestricted] at hr
  split at hr
  · cases hr
  · rename_i hne
    cases hr
    exact relBase_mem g types (fun hm => hne ⟨h, hm⟩) s d

theorem relMatUnrestricted_none (g : G V) (types : List String) :
    relMatUnrestricted g types = none ↔ types ≠ [] ∧ types.filterMap (getRelMat g) = [] := by
  simp only [relMatUnrestricted]
  split
  · simp_all
  · simp_all

def resolveLabelIds (g : G V) (labs : List String) : Option (List Nat) := labs.mapM (getLabelId g)

theorem resolveLabelIds_none (g : G V) (labs : List String) (s : String) (hs : s ∈ labs)
    (h : getLabelId g s = none) : resolveLabelIds g labs = none := by
  unfold resolveLabelIds
  induction labs with
  | nil => cases hs
  | cons a l ih =>
    rcases List.mem_cons.1 hs with rfl | hl
    · simp [h]
    · simp only [List.mapM_cons]; rw [ih hl]; cases getLabelId g a <;> rfl

def diagAll (ms : List Mat) (n : Nat) : Bool := ms.all (fun m => decide ((n, n) ∈ m.ents))

/-- `build_relationship_matrix`: union of the types' patterns (adjacency if
none given), rows restricted to nodes carrying every source label, columns to
nodes carrying every destination label; any unknown name → zero matrix. -/
def relMatrix (g : G V) (types srcL dstL : List String) : List (Nat × Nat) :=
  match srcL.mapM (getLabelMat g), dstL.mapM (getLabelMat g) with
  | some sm, some dm =>
    if types ≠ [] ∧ types.filterMap (getRelMat g) = [] then [] else
    (relBase g types).filter (fun p => (sm = [] || diagAll sm p.1) && (dm = [] || diagAll dm p.2))
  | _, _ => []

theorem relMatrix_mem (g : G V) (types srcL dstL : List String) (sm dm : List Mat)
    (hs : srcL.mapM (getLabelMat g) = some sm) (hd : dstL.mapM (getLabelMat g) = some dm)
    (ht : types ≠ []) (a b : Nat) :
    (a, b) ∈ relMatrix g types srcL dstL ↔
    (∃ tn ∈ types.filterMap (getRelMat g), ∃ e, (a, b, e) ∈ tn) ∧
    (∀ m ∈ sm, (a, a) ∈ m.ents) ∧ (∀ m ∈ dm, (b, b) ∈ m.ents) := by
  simp only [relMatrix, hs, hd]
  by_cases hh : types.filterMap (getRelMat g) = []
  · simp [ht, hh]
  · have hne : ¬(types ≠ [] ∧ types.filterMap (getRelMat g) = []) := fun h => hh h.2
    rw [if_neg hne]
    simp only [List.mem_filter, Bool.and_eq_true, Bool.or_eq_true, diagAll, List.all_eq_true,
      decide_eq_true_eq]
    rw [relBase_mem g types hh]
    apply and_congr Iff.rfl
    apply and_congr
    · constructor
      · rintro (h | h)
        · have : sm = [] := by simpa using h
          subst this; simp
        · exact h
      · intro h; exact Or.inr h
    · constructor
      · rintro (h | h)
        · have : dm = [] := by simpa using h
          subst this; simp
        · exact h
      · intro h; exact Or.inr h

def getRelationships (g : G V) (types srcL dstL : List String) (fromId toId : Option Nat) :
    List (Nat × Nat) :=
  (relMatrix g types srcL dstL).filter (fun p =>
    (match fromId with | some f => p.1 == f | none => true) &&
    (match toId with | some t => p.2 == t | none => true))

theorem getRelationships_all (g : G V) (types srcL dstL : List String) :
    getRelationships g types srcL dstL none none = relMatrix g types srcL dstL := by
  simp [getRelationships]

def relTypeId (g : G V) (e : Nat) : Option Nat := ((g.relType.row e).map (·.2)).head?

/-- With one type per edge (`relationship_type_matrix` row has a single
entry), `get_relationship_type_id` returns it. -/
theorem relTypeId_unique (g : G V) (e t : Nat) (h : g.relType.row e = [(e, t)]) :
    relTypeId g e = some t := by simp [relTypeId, h]

def relEndpoints (g : G V) (e : Nat) : Option (Nat × Nat) := g.endpoints e
theorem relEndpoints_eq (g : G V) (e : Nat) : relEndpoints g e = g.endpoints e := rfl

def relTypeIter (g : G V) (lo hi : Nat) : List (Nat × Nat) :=
  g.relType.ents.filter (fun p => lo ≤ p.1 && p.1 ≤ hi)
theorem relTypeIter_mem (g : G V) (lo hi e t : Nat) :
    (e, t) ∈ relTypeIter g lo hi ↔ (e, t) ∈ g.relType.ents ∧ lo ≤ e ∧ e ≤ hi := by
  simp [relTypeIter]

def allEdges (g : G V) (s : String) : Ten := (getRelMat g s).getD []
theorem allEdges_unknown (g : G V) (s : String) (h : s ∉ g.types) : allEdges g s = [] := by
  simp [allEdges, getRelMat, h]

def adjMatrix (g : G V) (types : List String) : List (Nat × Nat) :=
  match types with
  | [] => g.adj.ents
  | [t] => match getTypeId g t with
    | some i => Ten.pattern (g.relMs[i]?.getD [])
    | none => []
  | ts => ts.flatMap (fun t => match getTypeId g t with
    | some i => Ten.pattern (g.relMs[i]?.getD [])
    | none => [])

/-- The single-type fast path agrees with the general union. -/
theorem adjMatrix_single (g : G V) (t : String) :
    adjMatrix g [t] = [t].flatMap (fun t => match getTypeId g t with
      | some i => Ten.pattern (g.relMs[i]?.getD []) | none => []) := by
  simp [adjMatrix]

def symAdj (g : G V) (types : List String) : List (Nat × Nat) :=
  adjMatrix g types ++ (adjMatrix g types).map (fun p => (p.2, p.1))

theorem symAdj_symm (g : G V) (types : List String) (a b : Nat) :
    (a, b) ∈ symAdj g types ↔ (b, a) ∈ symAdj g types := by
  simp only [symAdj, List.mem_append, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro (h | ⟨⟨x, y⟩, h, rfl, rfl⟩)
    · exact Or.inr ⟨(a, b), h, rfl, rfl⟩
    · exact Or.inl h
  · rintro (h | ⟨⟨x, y⟩, h, rfl, rfl⟩)
    · exact Or.inr ⟨(b, a), h, rfl, rfl⟩
    · exact Or.inl h

end GQ
