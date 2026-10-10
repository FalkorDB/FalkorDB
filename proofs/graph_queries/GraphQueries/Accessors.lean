import GraphQueries.Constraints
/-
# Accessors and newtype glue (graph.rs)

Each Rust accessor is a one-line projection or a call into a function
modelled elsewhere; the theorem states exactly what it returns.

| here | there (graph.rs) |
| --- | --- |
| `labelIdToUsize`… | `From` impls :169,:175,:181,:187,:193,:199; `NodeOpError::from` :342; `Plan::new` :206 |
| `nameOf`… `constraintsOf` | `name` :1035, `node_count` :1040, `relationship_count` :1057, `property_key_count` :1086, `node_cap` :1091, `labels_count` :1096, `get_labels` :1101, `get_label_by_id` :1106, `get_types` :1114, `get_type` :1119, `get_attrs` :1131, `node_attribute_count` :1392, `deleted_*` :2725-2740, `label_matrices` :2745, `adjacency_matrix` :2750, `relationship_tensors` :2755, `relationship_matrices_iter` :2288, `relationship_attrs` :3308, `constraints(_mut)` :4099/:3975 |
| `getNodeAttribute`… | `get_node_attribute` :2472, `_by_idx` :2484, `get_node_attributes_by_idx` :2495, relationship twins :3281/:3166/:3187, `get_node_attrs` :3409, `get_node_all_attrs` :3417, `_by_id` :3424, `get_node_attr_count` :3433, relationship twins :3440-3464, `get_*_attribute_id` :1384/:1371/:1384, `get_*_attribute_names` :4765/:4643, `build_global_attrs` :4805, `estimate_entity_attr_size` :4699 |
-/
namespace GQ
variable {V : Type}

/-! ## Newtypes: `LabelId`/`TypeId` → usize, `NodeId`/`RelationshipId` ↔ u64 are identity wrappers. -/
structure NodeId where v : Nat deriving DecidableEq
def nodeIdOf (x : Nat) : NodeId := ⟨x⟩
def nodeIdTo (n : NodeId) : Nat := n.v
theorem nodeId_roundtrip (x : Nat) : nodeIdTo (nodeIdOf x) = x := rfl
theorem nodeId_roundtrip' (n : NodeId) : nodeIdOf (nodeIdTo n) = n := rfl
/-- `RelationshipId`, `LabelId`, `TypeId` are the same shape. -/
abbrev RelId := NodeId

/-- `NodeOpError` (graph.rs:276): `IdSpace { kind, source }` since #2846;
`DanglingRelationship { id }` (graph.rs:318) since #3022, raised by
`verify_created_relationships` (`Ids.verifyCreatedRels`). -/
inductive NodeOpError | graph (s : String) | idSpace (kind : String) (code : Nat)
  | danglingRelationship (id : Nat) deriving DecidableEq
def nodeOpErrorOf (s : String) : NodeOpError := .graph s
theorem nodeOpErrorOf_spec (s : String) : nodeOpErrorOf s = .graph s := rfl
/-- `NodeOpError::node` (:324) / `::relationship` (:333): the id-space refusal
tagged with its entity kind — the `Except String` tag used in Ids.lean. -/
def nodeOpErrorNode (code : Nat) : NodeOpError := .idSpace "node" code
def nodeOpErrorRel (code : Nat) : NodeOpError := .idSpace "relationship" code
theorem nodeOpErrorKind_spec (c : Nat) :
    nodeOpErrorNode c = .idSpace "node" c ∧ nodeOpErrorRel c = .idSpace "relationship" c ∧
    nodeOpErrorNode c ≠ nodeOpErrorRel c := by
  simp [nodeOpErrorNode, nodeOpErrorRel]

structure Plan where
  cached : Bool
  paramsOffset : Nat
def planNew (cached : Bool) (off : Nat) : Plan := ⟨cached, off⟩
theorem planNew_spec (c : Bool) (o : Nat) : (planNew c o).cached = c ∧ (planNew c o).paramsOffset = o :=
  ⟨rfl, rfl⟩

/-! ## Graph field accessors -/

def propertyKeyCount (g : G V) : Nat := g.attrs.length
def nodeAttributeCount (g : G V) : Nat := g.attrs.length
def labelsCount (g : G V) : Nat := g.labels.length
def labelById (g : G V) (i : Nat) : Option String := g.labels[i]?   -- `None` = index panic
def typeById (g : G V) (i : Nat) : Option String := g.types[i]?

theorem propertyKeyCount_eq (g : G V) : propertyKeyCount g = nodeAttributeCount g := rfl

theorem labelById_getLabelId (g : G V) (h : LInv g) (s : String) (i : Nat)
    (hi : getLabelId g s = some i) : labelById g i = some s := by
  unfold labelById; rw [getLabelId, h.1] at hi; exact idx_get _ _ _ hi

theorem typeById_getTypeId (g : G V) (s : String) (i : Nat) (hi : getTypeId g s = some i) :
    typeById g i = some s := idx_get _ _ _ hi

theorem labelsCount_eq (g : G V) (h : LInv g) : labelsCount g = g.labelMs.length := h.2.symm

theorem deletedCounts (g : G V) : nodeBound g - g.nodeCount = g.delNodes.length ∧
    relBound g - g.relCount = g.delRels.length := by
  simp [nodeBound, relBound, nodeIds, relIds, IdS.bound]

/-! ## Attribute accessors (one shared dictionary) -/

def getNodeAttribute (g : G V) (n : Nat) (s : String) : Option V := attrByName g.attrs g.nodeAttrs n s
def getRelAttribute (g : G V) (e : Nat) (s : String) : Option V := attrByName g.attrs g.relAttrs e s
def getNodeAttributeByIdx (g : G V) (n i : Nat) : Option V := byIdx g.nodeAttrs n i
def getRelAttributeByIdx (g : G V) (e i : Nat) : Option V := byIdx g.relAttrs e i
def getNodeAttrs (g : G V) (n : Nat) : List String := attrNames g.attrs g.nodeAttrs n
def getRelAttrs (g : G V) (e : Nat) : List String := attrNames g.attrs g.relAttrs e
def getNodeAllAttrs (g : G V) (n : Nat) : List (String × V) := attrPairs g.attrs g.nodeAttrs n
def getRelAllAttrs (g : G V) (e : Nat) : List (String × V) := attrPairs g.attrs g.relAttrs e
def getNodeAllAttrsById (g : G V) (n : Nat) : List (Nat × V) := g.nodeAttrs n
def getRelAllAttrsById (g : G V) (e : Nat) : List (Nat × V) := g.relAttrs e
def getNodeAttrCount (g : G V) (n : Nat) : Nat := attrCount g.nodeAttrs n
def getRelAttrCount (g : G V) (e : Nat) : Nat := attrCount g.relAttrs e
def getNodeAttributesByIdx (g : G V) (ids : List Nat) (i : Nat) (d : V) (out : List V) : List V :=
  batchByIdx g.nodeAttrs ids i d out
def getRelAttributesByIdx (g : G V) (ids : List Nat) (i : Nat) (d : V) (out : List V) : List V :=
  batchByIdx g.relAttrs ids i d out
/-- `get_node_attribute_id`, `get_relationship_attribute_id`, `get_global_attribute_id`:
three names for one dictionary lookup. -/
def getGlobalAttributeId (g : G V) (s : String) : Option Nat := getAttrId g.attrs s
/-- `get_node_attribute_names` / `get_relationship_attribute_names` / `build_global_attrs`. -/
def attributeNames (g : G V) : List String := g.attrs
def estimateSize (mem : Nat → Nat) (n : Nat) : Nat := mem n

/-- Name lookup = id lookup then indexed read (no `u16` aliasing under the cap). -/
theorem getNodeAttribute_spec (g : G V) (n : Nat) (s : String) (h : g.attrs.length ≤ MAXA) :
    getNodeAttribute g n s = (getGlobalAttributeId g s).bind (getNodeAttributeByIdx g n) :=
  attrByName_spec _ _ _ _ h
theorem getRelAttribute_spec (g : G V) (e : Nat) (s : String) (h : g.attrs.length ≤ MAXA) :
    getRelAttribute g e s = (getGlobalAttributeId g s).bind (getRelAttributeByIdx g e) :=
  attrByName_spec _ _ _ _ h

/-- One dictionary: node and relationship stores resolve a name to the same id. -/
theorem node_rel_same_id (g : G V) (s : String) :
    getGlobalAttributeId g s = getAttrId g.attrs s ∧ attributeNames g = g.attrs := ⟨rfl, rfl⟩

theorem getNodeAttrs_eq (g : G V) (n : Nat) : (getNodeAllAttrs g n).map (·.1) = getNodeAttrs g n :=
  attrPairs_names _ _ _
theorem getRelAttrs_eq (g : G V) (e : Nat) : (getRelAllAttrs g e).map (·.1) = getRelAttrs g e :=
  attrPairs_names _ _ _
theorem getNodeAttrCount_eq (g : G V) (n : Nat) : getNodeAttrCount g n = (getNodeAllAttrsById g n).length := rfl
theorem getRelAttrCount_eq (g : G V) (e : Nat) : getRelAttrCount g e = (getRelAllAttrsById g e).length := rfl
theorem getNodeAttributesByIdx_spec (g : G V) (ids : List Nat) (i : Nat) (d : V) (out : List V)
    (k : Nat) (hk : k < ids.length) :
    (getNodeAttributesByIdx g ids i d out)[out.length + k]? = some ((getNodeAttributeByIdx g ids[k] i).getD d) :=
  batchByIdx_get _ _ _ _ _ _ hk
theorem getRelAttributesByIdx_spec (g : G V) (ids : List Nat) (i : Nat) (d : V) (out : List V)
    (k : Nat) (hk : k < ids.length) :
    (getRelAttributesByIdx g ids i d out)[out.length + k]? = some ((getRelAttributeByIdx g ids[k] i).getD d) :=
  batchByIdx_get _ _ _ _ _ _ hk
theorem estimateSize_eq (mem : Nat → Nat) (n : Nat) : estimateSize mem n = mem n := rfl

theorem addRaw_spec (cs : List Constraint) (c : Constraint) :
    addRaw cs c = cs ++ [c] ∧ (addRaw cs c).length = cs.length + 1 := ⟨rfl, by simp [addRaw]⟩
theorem isRelDeleted_return (O : IdSpaceOps) (hC : IdSpaceContract O) (g r : G V) (id : Nat)
    (h : cancelRelId O g id = .ok r) : isRelDeleted r id = true := by
  unfold cancelRelId at h
  split at h
  · rename_i s hs
    cases h
    obtain ⟨h1, -⟩ := hC.cancel_ok _ _ _ hs
    simp only [isRelDeleted, IdS.isFree, relIds, setRelIds, h1, ins]
    by_cases hh : id ∈ g.delRels <;> simp [hh]
  · cases h
theorem typeEdgeCount_eq (g : G V) (i : Nat) (t : Ten) (h : g.relMs[i]? = some t) :
    typeEdgeCount g i = some t.length := by simp [typeEdgeCount, h, Ten.edgeCount]
theorem labelNodeCountByIdx_eq (g : G V) (i : Nat) (m : Mat) (h : g.labelMs[i]? = some m) :
    labelNodeCountByIdx g i = some m.ents.length := by simp [labelNodeCountByIdx, h, Mat.nvals]

end GQ
