import GraphQueries.IdContract
/-
# Id spaces, construction and versioning (graph.rs after #2846)

`Graph` keeps two `IdSpace`s (`node_ids`/`relationship_ids`, graph.rs:371/358)
and no counter of its own. `nodeIds g`/`relIds g` reassemble one from `G`'s
fields; `setNodeIds`/`setRelIds` write one back. Every mutation goes through
`O : IdSpaceOps`, whose behaviour is the hypothesis `IdSpaceContract O`
(IdContract.lean; proofs/id_space owns its proof). Errors carry the entity
kind (`NodeOpError::node`/`relationship`, graph.rs:324/317) as `Except String`.

| here | there (graph.rs) |
| --- | --- |
| `nodeBound`/`relBound` | `node_id_bound` :1585 / `relationship_id_bound` :1591 |
| `maxNodeId`/`maxRelId` | :1661 / :1666 (`IdSpace::max_id`) |
| `nodeIds`/`relIds` | `node_id_space` :1571 / `relationship_id_space` :1577 |
| `cancelNodeId`/`cancelRelId` | `cancel_node_id` :1427 / `cancel_relationship_id` :1440 |
| `openIdBatches` | `open_id_batches` :1470 |
| `rollIdBatches` | `roll_id_batches` :1489 |
| `verifyIdBatches` / `validateGraph` | `verify_id_batches` :1557 / `validate` :1507 (#3022: then `verify_created_relationships`) |
| `verifyCreatedRels`, `nodeLive` | `verify_created_relationships` :1524 (#3022) |
| `growForNodes` | `grow_for_nodes` :1643 |
| `createNodes` | `create_nodes` :1624 |
| `newG` | `Graph::new` :804 |
| `restoreCaps`, `clampTiers`, `collectIdx` | `Graph::restore` :847 (id spaces :913-914, caps :915-921, tier clamp :899-902, `node_labels_index` :936) |
| `newVersion` | `new_version` :986 |
| `rebuildRelType` | `rebuild_derived_matrices` :956 (type-matrix half; `rebuild_backward` is a GraphBLAS transpose) |
-/
namespace GQ
variable {V : Type}

def nodeIds (g : G V) : IdS := ⟨g.nodeCount, g.delNodes, g.nodeEB, g.nodeTaken⟩
def relIds (g : G V) : IdS := ⟨g.relCount, g.delRels, g.relEB, g.relTaken⟩
def setNodeIds (g : G V) (s : IdS) : G V :=
  { g with nodeCount := s.live, delNodes := s.recycled, nodeEB := s.entryBound, nodeTaken := s.taken }
def setRelIds (g : G V) (s : IdS) : G V :=
  { g with relCount := s.live, delRels := s.recycled, relEB := s.entryBound, relTaken := s.taken }

/-- `node_id_bound` (:1585) = `node_ids.bound()`. -/
def nodeBound (g : G V) : Nat := (nodeIds g).bound
def relBound (g : G V) : Nat := (relIds g).bound
/-- `max_node_id` (:1661) = `node_ids.max_id()`. -/
def maxNodeId (g : G V) : Nat := (nodeIds g).maxId
def maxRelId (g : G V) : Nat := (relIds g).maxId

theorem nodeBound_eq (g : G V) : nodeBound g = g.nodeCount + g.delNodes.length := rfl
theorem relBound_eq (g : G V) : relBound g = g.relCount + g.delRels.length := rfl

theorem maxNodeId_lt (g : G V) (h : g.nodeCount ≠ 0) : maxNodeId g + 1 = nodeBound g := by
  simp [maxNodeId, IdS.maxId, nodeIds, h, nodeBound, IdS.bound]; omega
theorem maxRelId_lt (g : G V) (h : g.relCount ≠ 0) : maxRelId g + 1 = relBound g := by
  simp [maxRelId, IdS.maxId, relIds, h, relBound, IdS.bound]; omega
/-- `node_id_space` (:1571) is a read-only view: the space it lends is
exactly the graph's fields. -/
theorem openNodeSpace_eq (g : G V) : (nodeIds g).bound = g.nodeCount + g.delNodes.length ∧
    setNodeIds g (nodeIds g) = g := ⟨rfl, rfl⟩
theorem openRelSpace_eq (g : G V) : (relIds g).bound = g.relCount + g.delRels.length ∧
    setRelIds g (relIds g) = g := ⟨rfl, rfl⟩

/-- `cancel_node_id` (:1427): `node_ids.cancel(id).map_err(NodeOpError::node)`. -/
def cancelNodeId (O : IdSpaceOps) (g : G V) (id : Nat) : Except String (G V) :=
  match O.cancel (nodeIds g) id with
  | some s => .ok (setNodeIds g s)
  | none => .error "node"
/-- `cancel_relationship_id` (:1440). -/
def cancelRelId (O : IdSpaceOps) (g : G V) (id : Nat) : Except String (G V) :=
  match O.cancel (relIds g) id with
  | some s => .ok (setRelIds g s)
  | none => .error "relationship"

def isNodeDeleted (g : G V) (id : Nat) : Bool := (nodeIds g).isFree id
def isRelDeleted (g : G V) (id : Nat) : Bool := (relIds g).isFree id

/-- Handing back a reserved (fresh, never-counted) id advances the boundary
by exactly one, which is what keeps `[0, bound)` dense; the live count does
not move. -/
theorem returnNode_bound (O : IdSpaceOps) (hC : IdSpaceContract O) (g r : G V) (id : Nat)
    (h : cancelNodeId O g id = .ok r) (hn : id ∉ g.delNodes) :
    nodeBound r = nodeBound g + 1 ∧ r.nodeCount = g.nodeCount := by
  unfold cancelNodeId at h
  split at h
  · rename_i s hs
    cases h
    obtain ⟨h1, h2⟩ := hC.cancel_ok _ _ _ hs
    simp only [nodeIds] at h1 h2
    simp [nodeBound, IdS.bound, nodeIds, setNodeIds, h1, h2, ins, hn]; omega
  · cases h
theorem returnRel_bound (O : IdSpaceOps) (hC : IdSpaceContract O) (g r : G V) (id : Nat)
    (h : cancelRelId O g id = .ok r) (hn : id ∉ g.delRels) :
    relBound r = relBound g + 1 ∧ r.relCount = g.relCount := by
  unfold cancelRelId at h
  split at h
  · rename_i s hs
    cases h
    obtain ⟨h1, h2⟩ := hC.cancel_ok _ _ _ hs
    simp only [relIds] at h1 h2
    simp [relBound, IdS.bound, relIds, setRelIds, h1, h2, ins, hn]; omega
  · cases h
/-- A cancelled id is free afterwards. -/
theorem isNodeDeleted_return (O : IdSpaceOps) (hC : IdSpaceContract O) (g r : G V) (id : Nat)
    (h : cancelNodeId O g id = .ok r) : isNodeDeleted r id = true := by
  unfold cancelNodeId at h
  split at h
  · rename_i s hs
    cases h
    obtain ⟨h1, -⟩ := hC.cancel_ok _ _ _ hs
    simp only [isNodeDeleted, IdS.isFree, nodeIds, setNodeIds, h1, ins]
    by_cases hh : id ∈ g.delNodes <;> simp [hh]
  · cases h
/-- A refusal names the node space. -/
theorem cancelNodeId_err (O : IdSpaceOps) (g : G V) (id : Nat) (h : O.cancel (nodeIds g) id = none) :
    cancelNodeId O g id = .error "node" := by simp [cancelNodeId, h]
theorem cancelRelId_err (O : IdSpaceOps) (g : G V) (id : Nat) (h : O.cancel (relIds g) id = none) :
    cancelRelId O g id = .error "relationship" := by simp [cancelRelId, h]

/-! ## Batches -/

/-- `verify_id_batches` (:1557): node first, then relationship; the error
names which. -/
def verifyIdBatches (O : IdSpaceOps) (g : G V) : Except String Unit :=
  if !O.verify (nodeIds g) then .error "node"
  else if !O.verify (relIds g) then .error "relationship" else .ok ()
/-- The `live` closure of `verify_created_relationships` (:1531-1536):
`n < bound && !(check_free && free.contains(n))`, `check_free = !free.is_empty()`. -/
def liveB (g : G V) (n : Nat) : Bool :=
  decide (n < (nodeIds g).bound) && !(!(nodeIds g).recycled.isEmpty && decide (n ∈ (nodeIds g).recycled))

/-- The loop body (:1540-1547): not skipped as freed (`skip_freed && is_free(id)`,
`skip_freed = !rels.recycled().is_empty()`), and `endpoints_for_edge(id)`
(:3240) is not `Some` with both ends live. -/
def offends (g : G V) (id : Nat) : Bool :=
  !(!(relIds g).recycled.isEmpty && (relIds g).isFree id) &&
  !((g.endpoints id).any (fun p => liveB g p.1 && liveB g p.2))

/-- `verify_created_relationships` (:1524, #3022): the first id of the batch's
`taken` (roaring: ascending) that `offends` is `DanglingRelationship { id }`. -/
def verifyCreatedRels (g : G V) : Except Nat Unit :=
  match (relIds g).taken.find? (offends g) with
  | some id => .error id
  | none => .ok ()

/-- A node is live between batches: below the boundary and not free (the space is
dense once `verify_id_batches` has passed; `proofs/id_space` `verify_ok_iff`). -/
def nodeLive (g : G V) (n : Nat) : Prop := n < (nodeIds g).bound ∧ n ∉ (nodeIds g).recycled

/-- The empty-bin fast path changes nothing: `live` is `nodeLive`. -/
theorem liveB_iff (g : G V) (n : Nat) : liveB g n = true ↔ nodeLive g n := by
  unfold liveB nodeLive
  cases h : (nodeIds g).recycled with
  | nil => simp
  | cons a t => simp

/-- …and `skip_freed && is_free(id)` is `is_free(id)`. -/
theorem skip_iff (g : G V) (id : Nat) :
    (!(relIds g).recycled.isEmpty && (relIds g).isFree id) = true ↔ id ∈ (relIds g).recycled := by
  unfold IdS.isFree
  cases h : (relIds g).recycled with
  | nil => simp
  | cons a t => simp

theorem offends_iff (g : G V) (id : Nat) :
    offends g id = true ↔
      id ∉ (relIds g).recycled ∧ ¬ ∃ s d, g.endpoints id = some (s, d) ∧ nodeLive g s ∧ nodeLive g d := by
  unfold offends
  rw [Bool.and_eq_true, Bool.not_eq_true', Bool.not_eq_true']
  have e1 : (!(relIds g).recycled.isEmpty && (relIds g).isFree id) = false ↔ id ∉ (relIds g).recycled := by
    rw [← skip_iff]; simp
  have e2 : ((g.endpoints id).any (fun p => liveB g p.1 && liveB g p.2)) = false ↔
      ¬ ∃ s d, g.endpoints id = some (s, d) ∧ nodeLive g s ∧ nodeLive g d := by
    cases he : g.endpoints id with
    | none => simp
    | some p =>
      obtain ⟨s, d⟩ := p
      simp only [Option.any_some, Bool.and_eq_false_iff, Option.some.injEq, Prod.mk.injEq]
      constructor
      · rintro h ⟨s', d', ⟨rfl, rfl⟩, h1, h2⟩
        rw [← liveB_iff] at h1 h2
        rcases h with h | h <;> simp_all
      · intro h
        cases h1 : liveB g s
        · exact .inl rfl
        · cases h2 : liveB g d
          · exact .inr rfl
          · exact absurd ⟨s, d, ⟨rfl, rfl⟩, (liveB_iff g s).mp h1, (liveB_iff g d).mp h2⟩ h
  rw [e1, e2]

/-- **`verify_created_relationships` accepts exactly** when every relationship the
batch took and did not free again ends at two live nodes; otherwise it names an
offender from `taken`. -/
theorem verifyCreatedRels_spec (g : G V) :
    (verifyCreatedRels g = .ok () ↔
      ∀ id ∈ (relIds g).taken, id ∉ (relIds g).recycled →
        ∃ s d, g.endpoints id = some (s, d) ∧ nodeLive g s ∧ nodeLive g d) ∧
    (∀ id, verifyCreatedRels g = .error id → id ∈ (relIds g).taken ∧ id ∉ (relIds g).recycled ∧
        ¬ ∃ s d, g.endpoints id = some (s, d) ∧ nodeLive g s ∧ nodeLive g d) := by
  unfold verifyCreatedRels
  constructor
  · cases hf : (relIds g).taken.find? (offends g) with
    | some id =>
      simp only [reduceCtorEq, false_iff, Classical.not_forall]
      have hm := List.mem_of_find?_eq_some hf
      have := (offends_iff g id).mp (List.find?_some hf)
      exact ⟨id, hm, this.1, this.2⟩
    | none =>
      simp only [true_iff]
      intro id hm hr
      have := List.find?_eq_none.mp hf id hm
      apply Classical.byContradiction; intro hn
      exact this ((offends_iff g id).mpr ⟨hr, hn⟩)
  · intro id h
    cases hf : (relIds g).taken.find? (offends g) with
    | none => rw [hf] at h; cases h
    | some id' =>
      rw [hf] at h; cases h
      exact ⟨List.mem_of_find?_eq_some hf, (offends_iff g _).mp (List.find?_some hf)⟩

/-- `validate` (:1507): `verify_id_batches()?`, then (#3022)
`verify_created_relationships()` (`NodeOpError::DanglingRelationship`). -/
def validateGraph (O : IdSpaceOps) (g : G V) : Except String Unit :=
  match verifyIdBatches O g with
  | .error e => .error e
  | .ok () => match verifyCreatedRels g with
    | .error id => .error s!"dangling relationship {id}"
    | .ok () => .ok ()

/-- `open_id_batches` (:1470). -/
def openIdBatches (O : IdSpaceOps) (g : G V) : Except String (G V) :=
  match O.openBatch (nodeIds g) with
  | none => .error "node"
  | some n =>
    let g1 := setNodeIds g n
    match O.openBatch (relIds g1) with
    | none => .error "relationship"
    | some r => .ok (setRelIds g1 r)

/-- `roll_id_batches` (:1489): verify, then open. -/
def rollIdBatches (O : IdSpaceOps) (g : G V) : Except String (G V) :=
  match verifyIdBatches O g with
  | .error e => .error e
  | .ok () => openIdBatches O g

theorem verifyIdBatches_ok_iff (O : IdSpaceOps) (g : G V) :
    verifyIdBatches O g = .ok () ↔ O.verify (nodeIds g) = true ∧ O.verify (relIds g) = true := by
  unfold verifyIdBatches
  cases h1 : O.verify (nodeIds g) <;> cases h2 : O.verify (relIds g) <;> simp

/-- **`validate` accepts exactly** two verified id batches whose created
relationships all still end at live nodes (#3022). -/
theorem validate_ok_iff (O : IdSpaceOps) (g : G V) :
    validateGraph O g = .ok () ↔ O.verify (nodeIds g) = true ∧ O.verify (relIds g) = true ∧
      ∀ id ∈ (relIds g).taken, id ∉ (relIds g).recycled →
        ∃ s d, g.endpoints id = some (s, d) ∧ nodeLive g s ∧ nodeLive g d := by
  rw [← (verifyCreatedRels_spec g).1, ← and_assoc, ← verifyIdBatches_ok_iff]
  unfold validateGraph
  cases verifyIdBatches O g with
  | error e => simp
  | ok u => cases u; cases verifyCreatedRels g <;> simp

/-- Rolling never opens over an unverified batch, and when it succeeds the
two id spaces keep their live counts and free sets, re-anchored at the
current boundary with an empty ledger. -/
theorem rollIdBatches_spec (O : IdSpaceOps) (hC : IdSpaceContract O) (g r : G V)
    (h : rollIdBatches O g = .ok r) :
    verifyIdBatches O g = .ok () ∧
    nodeIds r = ⟨g.nodeCount, g.delNodes, nodeBound g, []⟩ ∧
    relIds r = ⟨g.relCount, g.delRels, relBound g, []⟩ := by
  unfold rollIdBatches at h
  split at h
  · cases h
  · rename_i hv
    refine ⟨hv, ?_⟩
    unfold openIdBatches at h
    simp only at h
    split at h
    · cases h
    · rename_i n hn
      split at h
      · cases h
      · rename_i rr hr
        cases h
        obtain ⟨a1, a2, a3, a4⟩ := hC.open_ok _ _ hn
        obtain ⟨b1, b2, b3, b4⟩ := hC.open_ok _ _ hr
        simp only [relIds, setNodeIds] at b1 b2 b3 b4
        simp only [nodeIds] at a1 a2 a3 a4
        simp [nodeIds, relIds, setRelIds, setNodeIds, a1, a2, a3, a4, b1, b2, b3, b4,
          nodeBound, relBound, IdS.bound]

theorem rollIdBatches_refused (O : IdSpaceOps) (g : G V) (e : String)
    (h : verifyIdBatches O g = .error e) : rollIdBatches O g = .error e := by
  simp [rollIdBatches, h]

/-! ## Creating nodes -/

/-- `grow_for_nodes` (:1643), up to the final `resize`. -/
def mnl2 (chunk : Nat) (hc : 0 < chunk) (g : G V) (nodes : List Nat) : G V :=
  match nodes.max? with
  | some m =>
    if m + 1 > g.nodeCap then
      resizeNodeMs ({ g with nodeCap := growCap chunk g.nodeCap (m + 1) hc } : G V)
    else g
  | none => g

/-- `grow_for_nodes` (:1643). -/
def growForNodes (chunk : Nat) (hc : 0 < chunk) (g : G V) (nodes : List Nat) : G V :=
  resize chunk hc (mnl2 chunk hc g nodes)

theorem resize_nodeCap_ge (chunk : Nat) (hc : 0 < chunk) (g : G V) :
    g.nodeCap ≤ (resize chunk hc g).nodeCap ∧ (resize chunk hc g).nodeCount = g.nodeCount ∧
    (resize chunk hc g).delNodes = g.delNodes := by
  have := growCap_ge_cap chunk g.nodeCap g.nodeCount hc
  simp only [resize]; (repeat' split) <;> simp [resizeNodeMs, resizeRelMs] <;> omega

theorem mnl2_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (nodes : List Nat)
    (hlim : ∀ n ∈ nodes, n < grbMax) :
    (∀ n ∈ nodes, n < (mnl2 chunk hc g nodes).nodeCap) ∧
    (mnl2 chunk hc g nodes).nodeCount = g.nodeCount ∧
    (mnl2 chunk hc g nodes).delNodes = g.delNodes := by
  unfold mnl2
  cases hm : nodes.max? with
  | none =>
    have := List.max?_eq_none_iff.1 hm; subst this; simp
  | some m =>
    have hle := (List.max?_eq_some_iff.1 hm).2
    simp only
    split
    · have hm' := hlim m (List.max?_mem hm)
      have hg := growCap_ge_needed chunk g.nodeCap (m + 1) hc (by omega)
      simp only [resizeNodeMs]
      refine ⟨fun n hn => ?_, ?_, ?_⟩
      · have := hle n hn; omega
      all_goals simp
    · rename_i hgt
      refine ⟨fun n hn => ?_, ?_, ?_⟩
      · have := hle n hn; omega
      all_goals simp

/-- `grow_for_nodes` only sizes: every id fits, the id-space halves are untouched. -/
theorem growForNodes_spec (chunk : Nat) (hc : 0 < chunk) (g : G V) (nodes : List Nat)
    (hlim : ∀ n ∈ nodes, n < grbMax) :
    let r := growForNodes chunk hc g nodes
    (∀ n ∈ nodes, n < r.nodeCap) ∧ r.nodeCount = g.nodeCount ∧ r.delNodes = g.delNodes := by
  obtain ⟨a1, a2, a3⟩ := mnl2_spec chunk hc g nodes hlim
  obtain ⟨b1, b2, b3⟩ := resize_nodeCap_ge chunk hc (mnl2 chunk hc g nodes)
  simp only [growForNodes]
  exact ⟨fun n hn => Nat.lt_of_lt_of_le (a1 n hn) b1, by rw [b2, a2], by rw [b3, a3]⟩

/-- `create_nodes` (:1624): `node_ids.create(nodes)` then `grow_for_nodes`. -/
def createNodes (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (nodes : List Nat) :
    Except String (G V) :=
  match O.create (nodeIds g) nodes with
  | none => .error "node"
  | some s => .ok (growForNodes chunk hc (setNodeIds g s) nodes)

/-- After `create_nodes`, every new node id fits the matrices, the count
moved by the batch size, and the bin lost exactly the batch (the former
`markNodesLive_spec`, now through the id space). -/
theorem markNodesLive_spec (O : IdSpaceOps) (hC : IdSpaceContract O) (chunk : Nat) (hc : 0 < chunk)
    (g r : G V) (nodes : List Nat) (h : createNodes O chunk hc g nodes = .ok r) :
    (∀ n ∈ nodes, n < r.nodeCap) ∧ r.nodeCount = g.nodeCount + nodes.length ∧
    (∀ x, x ∈ r.delNodes ↔ x ∈ g.delNodes ∧ x ∉ nodes) := by
  unfold createNodes at h
  split at h
  · cases h
  · rename_i s hs
    cases h
    obtain ⟨a1, a2⟩ := hC.create_ok _ _ _ hs
    have hlim : ∀ n ∈ nodes, n < grbMax := fun n hn => by
      apply Nat.lt_of_not_le; intro hge
      rw [hC.create_refuses_limit _ _ n hn hge] at hs; cases hs
    obtain ⟨b1, b2, b3⟩ := growForNodes_spec chunk hc (setNodeIds g s) nodes hlim
    simp only [nodeIds] at a1 a2
    refine ⟨b1, by rw [b2]; simp [setNodeIds, a1], fun x => ?_⟩
    rw [b3]; simp [setNodeIds, a2, diff]

/-- An id live before the batch is refused, and the graph is not touched. -/
theorem createNodes_refused (O : IdSpaceOps) (hC : IdSpaceContract O) (chunk : Nat) (hc : 0 < chunk)
    (g : G V) (nodes : List Nat) (n : Nat) (hn : n ∈ nodes) (hl : n ∉ g.delNodes) (hb : n < g.nodeEB) :
    createNodes O chunk hc g nodes = .error "node" := by
  simp [createNodes, hC.create_refuses_live (nodeIds g) nodes n hn hl hb]
theorem createNodes_ok (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (nodes : List Nat)
    (s : IdS) (hs : O.create (nodeIds g) nodes = some s) :
    createNodes O chunk hc g nodes = .ok (growForNodes chunk hc (setNodeIds g s) nodes) := by
  simp [createNodes, hs]

/-! ## Construction -/

def newG (n e version : Nat) (name : String) : G V :=
  { name, nodeCap := n, relCap := e, nodeCount := 0, relCount := 0, delNodes := [], delRels := [],
    zero := Mat.empty 0 0, adj := Mat.empty n n, nodeLabels := Mat.empty 0 0,
    relType := Mat.empty 0 0, labelMs := [], relMs := [], endpoints := fun _ => none, attrs := [],
    nodeAttrs := fun _ => [], relAttrs := fun _ => [], labels := [], labelsIndex := fun _ => none,
    types := [], constraints := [], version, schemaVersion := 0 }

/-- `Graph::new` (:804): both id spaces are `IdSpace::new()` (id_space.rs:266). -/
theorem newG_spec (n e v : Nat) (name : String) :
    let g : G V := newG n e v name
    LInv g ∧ TInv g ∧ nodeBound g = 0 ∧ relBound g = 0 ∧ maxNodeId g = 0 ∧
    nodeIds g = IdS.new ∧ relIds g = IdS.new := by
  simp [newG, LInv, TInv, nodeBound, relBound, maxNodeId, idx, nodeIds, relIds, IdS.new,
    IdS.bound, IdS.maxId]

/-- `restore` caps (:913-921): `IdSpace::restored(count, deleted).bound()
.next_multiple_of(chunk).max(64)`. -/
def restoreCap (chunk count ndel : Nat) : Nat := max (nextMul chunk (count + ndel)) 64

/-- `restore`'s id spaces (:913-914) and caps: the restored space opens a
batch exactly at the decoded boundary, and the cap covers it. -/
theorem restore_ids (chunk count : Nat) (del : List Nat) :
    let s := IdS.restored count del
    s.live = count ∧ s.recycled = del ∧ s.entryBound = s.bound ∧ s.taken = [] ∧
    s.bound ≤ restoreCap chunk count del.length := by
  have := nextMul_ge chunk (count + del.length)
  simp [IdS.restored, IdS.bound, restoreCap]; omega

theorem restoreCap_covers (chunk count ndel : Nat) :
    count + ndel ≤ restoreCap chunk count ndel ∧ 64 ≤ restoreCap chunk count ndel := by
  have := nextMul_ge chunk (count + ndel)
  unfold restoreCap; omega

/-- Tier clamp (:898-902): `acc = len; for slot in starts.rev() { acc = min(acc, slot); slot = acc }`. -/
def clampTiers (len s0 s1 s2 : Nat) : Nat × Nat × Nat :=
  let a2 := min len s2
  let a1 := min a2 s1
  let a0 := min a1 s0
  (a0, a1, a2)

theorem clampTiers_mono (len s0 s1 s2 : Nat) :
    let r := clampTiers len s0 s1 s2
    r.1 ≤ r.2.1 ∧ r.2.1 ≤ r.2.2 ∧ r.2.2 ≤ len := by
  simp only [clampTiers]; omega

/-- `node_labels.iter().enumerate().map(..).collect()` into a map: later
entries overwrite earlier ones. -/
def collectIdx : List String → Nat → (String → Option Nat) → (String → Option Nat)
  | [], _, m => m
  | a :: l, i, m => collectIdx l (i + 1) (fun t => if t = a then some i else m t)

theorem collectIdx_spec : ∀ (l : List String) (i : Nat) (m : String → Option Nat) (s : String),
    l.Nodup → collectIdx l i m s = if s ∈ l then (idx l s).map (· + i) else m s
  | [], _, _, _, _ => by simp [collectIdx]
  | a :: l, i, m, s, hn => by
    have ⟨ha, hl⟩ := List.nodup_cons.1 hn
    rw [collectIdx, collectIdx_spec l (i + 1) _ s hl]
    by_cases hs : s ∈ l
    · have hsa : s ≠ a := fun h => ha (h ▸ hs)
      simp only [hs, ite_true, List.mem_cons, or_true, idx, Ne.symm hsa, ite_false,
        Option.map_map]
      congr 1; funext x; simp [Function.comp]; omega
    · by_cases hsa : s = a
      · subst hsa; simp [idx]; intro h; exact absurd h hs
      · simp [hs, hsa]

/-- `restore` re-establishes the label-table invariant from a duplicate-free
decoded label list. -/
theorem restore_LInv (labels : List String) (h : labels.Nodup) (s : String) :
    collectIdx labels 0 (fun _ => none) s = idx labels s := by
  rw [collectIdx_spec labels 0 _ s h]
  split
  · simp
  · rename_i hs; exact ((idx_none labels s).2 hs).symm

/-- `restore`'s endpoint index: every decoded edge is resolvable, provided
edge ids are unique (they are: one tensor entry per id). -/
def fillEndpoints (es : List (Nat × Nat × Nat)) (ep : Nat → Option (Nat × Nat)) : Nat → Option (Nat × Nat) :=
  es.foldl (fun f (s, d, e) => fun x => if x = e then some (s, d) else f x) ep

theorem fillEndpoints_spec : ∀ (es : List (Nat × Nat × Nat)) (ep : Nat → Option (Nat × Nat)) (e : Nat),
    (es.map (·.2.2)).Nodup →
    fillEndpoints es ep e = match es.find? (·.2.2 == e) with
      | some t => some (t.1, t.2.1) | none => ep e
  | [], _, _, _ => rfl
  | (s, d, e') :: es, ep, e, hn => by
    simp only [List.map_cons, List.nodup_cons] at hn
    show fillEndpoints es (fun x => if x = e' then some (s, d) else ep x) e = _
    rw [fillEndpoints_spec es _ e hn.2]
    simp only [List.find?_cons]
    by_cases he : e' = e
    · subst he
      have : es.find? (·.2.2 == e') = none := by
        rw [List.find?_eq_none]; intro x hx; simp only [beq_iff_eq]
        intro hxe; exact hn.1 (hxe ▸ List.mem_map_of_mem hx)
      simp [this]
    · have : (e' == e) = false := by simp [he]
      simp only [this]
      cases es.find? (·.2.2 == e) <;> simp [Ne.symm he]

/-- `new_version` (:986): same logical graph, `version + 1`
(`dup`/`Arc` clones share content — proofs/versioned_matrix `Cow`), and each
id space cloned with a fresh batch (`IdSpace::new_version`). -/
def newVersion (g : G V) : G V :=
  setRelIds (setNodeIds { g with version := g.version + 1 } (nodeIds g).newVersion) (relIds g).newVersion

/-- The fork has the same live ids and free sets, a batch opened at the
current boundary, and an empty ledger — the previous batch's `taken` never
reaches a published version. -/
theorem newVersion_spec (g : G V) :
    (newVersion g).version = g.version + 1 ∧
    nodeIds (newVersion g) = ⟨g.nodeCount, g.delNodes, nodeBound g, []⟩ ∧
    relIds (newVersion g) = ⟨g.relCount, g.delRels, relBound g, []⟩ ∧
    ({ newVersion g with version := g.version, nodeEB := g.nodeEB, nodeTaken := g.nodeTaken, relEB := g.relEB, relTaken := g.relTaken } : G V) = g := by
  simp [newVersion, setNodeIds, setRelIds, nodeIds, relIds, IdS.newVersion, IdS.restored,
    nodeBound, relBound, IdS.bound]

/-- Type-matrix half of `rebuild_derived_matrices` (:956): resize to
`relationship_cap × |types|`, then `(edge, type_idx)` for every tensor edge. -/
def rebuildRelType (g : G V) : G V :=
  let m := g.relType.resize g.relCap g.types.length
  let es := (List.range g.relMs.length).flatMap (fun t => ((g.relMs[t]?.getD []).map (·.2.2)).map (·, t))
  { g with relType := { m with ents := m.ents ++ es.filter (fun p => p.1 < g.relCap ∧ p.2 < g.types.length ∧ p ∉ m.ents) } }

theorem rebuildRelType_complete (g : G V) (h : TInv g) (t e s d : Nat) (ht : t < g.types.length)
    (he : e < g.relCap) (hm : (s, d, e) ∈ g.relMs[t]?.getD []) :
    (rebuildRelType g).relType.get e t = true := by
  have hmem : (e, t) ∈ (List.range g.relMs.length).flatMap
      (fun t => ((g.relMs[t]?.getD []).map (·.2.2)).map (·, t)) := by
    simp only [List.mem_flatMap, List.mem_range, List.mem_map]
    exact ⟨t, by unfold TInv at h; omega, e, ⟨(s, d, e), hm, rfl⟩, rfl⟩
  simp only [rebuildRelType, Mat.get, Mat.resize]
  by_cases hin : (e, t) ∈ g.relType.ents
  · simp [hin, he, ht]
  · simp only [List.mem_append, List.mem_filter]
    simp [hin, he, ht]
    exact ⟨t, by unfold TInv at h; omega, s, d, e, hm, rfl, rfl⟩

/-- `trim_attr_stores` (:980): arena slop only — the logical stores are unchanged. -/
def trimAttrStores (g : G V) : G V := g
theorem trimAttrStores_id (g : G V) : trimAttrStores g = g := rfl

end GQ
