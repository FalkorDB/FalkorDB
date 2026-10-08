import GraphQueries.Delete
/-
# Label writes and node deletion

`node_labels_matrix` (node × label) and the per-label diagonals
`labels_matices[l]` must agree: `LabCons`. `set_all`/`set_product`/`remove`
are set insert/delete on logical contents (proofs/versioned_matrix
`VM.setAll`, `VM.setProduct`, `VM.remove`, all proven against `eff`).

| here | there (graph.rs) |
| --- | --- |
| `setLabelsProduct` | `set_node_labels_product` :1987 |
| `setLabelsBulk`    | `set_nodes_labels_bulk` :2061 |
| `removeLabels`     | `remove_nodes_labels` :2102 |
| `deleteNodes`      | `delete_nodes` :2131 |
-/
namespace GQ
variable {V : Type}

def lm (g : G V) (l : Nat) : Mat := (g.labelMs[l]?).getD (Mat.empty 0 0)

/-- The two label representations agree. -/
def LabCons (g : G V) : Prop := ∀ n l, (n, n) ∈ (lm g l).ents ↔ (n, l) ∈ g.nodeLabels.ents

def addDiag (m : Mat) (ids : List Nat) : Mat := { m with ents := m.ents ++ ids.map (fun i => (i, i)) }

theorem lm_mapIdx (g : G V) (F : Nat → Mat → Mat) (l : Nat) :
    ((g.labelMs.mapIdx F)[l]?).getD (Mat.empty 0 0) =
      match g.labelMs[l]? with | some m => F l m | none => Mat.empty 0 0 := by
  rw [List.getElem?_mapIdx]; cases g.labelMs[l]? <;> rfl

def setLabelsProduct (g : G V) (ids lids : List Nat) : G V :=
  { g with labelMs := g.labelMs.mapIdx (fun l m => if l ∈ lids then addDiag m ids else m)
           nodeLabels := { g.nodeLabels with ents := g.nodeLabels.ents ++ ids.flatMap (fun i => lids.map (i, ·)) } }

theorem lm_none (g : G V) (hc : LabCons g) (n l : Nat) (hm : g.labelMs[l]? = none) :
    (n, l) ∉ g.nodeLabels.ents := by
  intro h; have := (hc n l).2 h; simp [lm, hm, Mat.empty] at this

theorem lt_of_some (g : G V) (l : Nat) (hl : l < g.labelMs.length) : g.labelMs[l]? ≠ none := by
  simp [List.getElem?_eq_getElem hl]

theorem setLabelsProduct_spec (g : G V) (ids lids : List Nat) (hc : LabCons g)
    (hl : ∀ l ∈ lids, l < g.labelMs.length) :
    LabCons (setLabelsProduct g ids lids) ∧
    ∀ i ∈ ids, ∀ l ∈ lids, (i, l) ∈ (setLabelsProduct g ids lids).nodeLabels.ents := by
  refine ⟨fun n l => ?_, fun i hi l hl' => by simp [setLabelsProduct, hi, hl']⟩
  simp only [lm, setLabelsProduct]
  rw [lm_mapIdx]
  cases hm : g.labelMs[l]? with
  | none =>
    have hnl := lm_none g hc n l hm
    have : l ∉ lids := fun h => lt_of_some g l (hl l h) hm
    simp [Mat.empty, hnl, this]
  | some m =>
    have h0 : (n, n) ∈ m.ents ↔ (n, l) ∈ g.nodeLabels.ents := by simpa [lm, hm] using hc n l
    simp only
    split
    · rename_i hli
      simp [addDiag, h0, hli]
    · rename_i hli
      simp [h0, hli]

theorem diag_mem (pairs : List (Nat × Nat)) (n l : Nat) :
    (n, n) ∈ ((pairs.filter (·.2 == l)).map (·.1)).map (fun i => (i, i)) ↔ (n, l) ∈ pairs := by
  simp only [List.map_map, List.mem_map, List.mem_filter, beq_iff_eq, Function.comp, Prod.mk.injEq]
  constructor
  · rintro ⟨⟨a, b⟩, ⟨h, rfl⟩, rfl, -⟩; exact h
  · intro h; exact ⟨(n, l), ⟨h, rfl⟩, rfl, rfl⟩

def setLabelsBulk (g : G V) (pairs : List (Nat × Nat)) : G V :=
  { g with labelMs := g.labelMs.mapIdx (fun l m => addDiag m ((pairs.filter (·.2 == l)).map (·.1)))
           nodeLabels := { g.nodeLabels with ents := g.nodeLabels.ents ++ pairs } }

theorem setLabelsBulk_spec (g : G V) (pairs : List (Nat × Nat)) (hc : LabCons g)
    (hl : ∀ p ∈ pairs, p.2 < g.labelMs.length) : LabCons (setLabelsBulk g pairs) := by
  intro n l
  simp only [lm, setLabelsBulk]
  rw [lm_mapIdx]
  cases hm : g.labelMs[l]? with
  | none =>
    have hnl := lm_none g hc n l hm
    have : (n, l) ∉ pairs := fun h => lt_of_some g l (hl _ h) hm
    simp [Mat.empty, hnl, this]
  | some m =>
    have h0 : (n, n) ∈ m.ents ↔ (n, l) ∈ g.nodeLabels.ents := by simpa [lm, hm] using hc n l
    simp only [addDiag, List.mem_append, h0, diag_mem]

def removeLabels (g : G V) (pairs : List (Nat × Nat)) : G V :=
  { g with labelMs := g.labelMs.mapIdx (fun l m => { m with ents := m.ents.filter (fun p => !(p.1 = p.2 ∧ (p.1, l) ∈ pairs)) })
           nodeLabels := { g.nodeLabels with ents := g.nodeLabels.ents.filter (· ∉ pairs) } }

theorem removeLabels_spec (g : G V) (pairs : List (Nat × Nat)) (hc : LabCons g) :
    LabCons (removeLabels g pairs) ∧ ∀ p ∈ pairs, p ∉ (removeLabels g pairs).nodeLabels.ents := by
  refine ⟨fun n l => ?_, fun p hp => by simp [removeLabels, hp]⟩
  have h0 := hc n l
  simp only [LabCons, lm, removeLabels] at h0 ⊢
  rw [lm_mapIdx]
  cases hm : g.labelMs[l]? with
  | none =>
    simp only [hm, Option.getD_none, Mat.empty, List.not_mem_nil, false_iff] at h0 ⊢
    simp only [List.mem_filter, not_and]; intro h; exact absurd h h0
  | some m =>
    simp only [hm, Option.getD_some] at h0
    simp only [List.mem_filter, Bool.not_eq_true', decide_eq_false_iff_not, decide_eq_true_eq, h0]
    constructor
    · rintro ⟨h1, h2⟩; exact ⟨h1, fun h => h2 ⟨trivial, h⟩⟩
    · rintro ⟨h1, h2⟩; exact ⟨h1, fun h => h2 h.2⟩

/-- `delete_nodes` (:2131): `node_ids.release(dn, dn)` (:2139-2141) judges
and frees in one step, then the `(node, label)` pairs read per deleted row,
per-pair removal from both label structures, and the attribute rows
dropped. `.error "node"` = refused, nothing moved. -/
def deleteNodes (O : IdSpaceOps) (g : G V) (dn : List Nat) : Except String (G V × List (Nat × Nat)) :=
  match O.release (nodeIds g) dn dn with
  | none => .error "node"
  | some sp =>
  let pairs := dn.flatMap (fun n => g.nodeLabels.row n)
  let g1 := removeLabels (setNodeIds g sp) pairs
  .ok ({ g1 with nodeAttrs := fun n => if n ∈ dn then [] else g.nodeAttrs n }, pairs)

theorem deleteNodes_spec (O : IdSpaceOps) (g : G V) (dn : List Nat) (r : G V) (pairs : List (Nat × Nat))
    (h : deleteNodes O g dn = .ok (r, pairs)) (hc : LabCons g) :
    LabCons r ∧ (∀ n l, (n, l) ∈ pairs ↔ n ∈ dn ∧ (n, l) ∈ g.nodeLabels.ents) ∧
    (∀ n ∈ dn, ∀ l, (n, l) ∉ r.nodeLabels.ents ∧ (n, n) ∉ (lm r l).ents ∧ r.nodeAttrs n = []) ∧
    (∀ n ∉ dn, ∀ l, (n, l) ∈ r.nodeLabels.ents ↔ (n, l) ∈ g.nodeLabels.ents) := by
  unfold deleteNodes at h
  split at h
  · cases h
  · rename_i sp _
    have hc : LabCons (setNodeIds g sp) := hc
    cases h
    have hp : ∀ n l, (n, l) ∈ dn.flatMap (fun n => g.nodeLabels.row n) ↔ n ∈ dn ∧ (n, l) ∈ g.nodeLabels.ents := by
      intro n l
      simp only [List.mem_flatMap, Mat.row, List.mem_filter, beq_iff_eq]
      constructor
      · rintro ⟨n', hn', hm, rfl⟩; exact ⟨hn', hm⟩
      · rintro ⟨hn, hm⟩; exact ⟨n, hn, hm, rfl⟩
    obtain ⟨hc', hrm⟩ := removeLabels_spec g (dn.flatMap (fun n => g.nodeLabels.row n)) hc
    refine ⟨fun n l => hc' n l, hp, fun n hn l => ⟨?_, ?_, by simp [hn]⟩, fun n hn l => ?_⟩
    · intro hm
      have h2 := (List.mem_filter.1 hm).1
      exact hrm _ ((hp n l).2 ⟨hn, h2⟩) hm
    · intro hm
      have := (hc' n l).1 hm
      exact hrm _ ((hp n l).2 ⟨hn, (List.mem_filter.1 this).1⟩) this
    · simp only [removeLabels, List.mem_filter, decide_eq_true_eq, hp, not_and]
      constructor
      · exact fun h => h.1
      · exact fun h => ⟨h, fun h' => absurd h' hn⟩

theorem deleteNodes_refused (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (dn : List Nat)
    (n : Nat) (hn : n ∈ dn) (hd : n ∈ g.delNodes) :
    deleteNodes O g dn = .error "node" := by
  simp [deleteNodes, hC.release_refuses_recycled (nodeIds g) dn dn n hn hd]

/-- The id half: the deleted ids are free afterwards and the live count
drops by exactly the batch (no double count — `dn` is a set). -/
theorem deleteNodes_ids (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (dn : List Nat) (r : G V)
    (pairs : List (Nat × Nat)) (h : deleteNodes O g dn = .ok (r, pairs)) :
    r.nodeCount = g.nodeCount - dn.length ∧ ∀ n ∈ dn, n ∈ r.delNodes := by
  unfold deleteNodes at h
  split at h
  · cases h
  · rename_i sp hsp
    cases h
    obtain ⟨h1, h2⟩ := hC.release_ok _ _ _ _ hsp
    simp only [nodeIds] at h1 h2
    refine ⟨by simp [removeLabels, setNodeIds, h2], fun n hn => ?_⟩
    simp only [removeLabels, setNodeIds, h1, union, List.mem_append, List.mem_filter]
    by_cases hh : n ∈ g.delNodes
    · exact Or.inl hh
    · exact Or.inr ⟨hn, decide_eq_true hh⟩

end GQ
