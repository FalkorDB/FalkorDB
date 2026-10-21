import Pr3437Review.PRArms

/-!
# The runtime's attribute reads, and `PropReads` as a theorem

A copy, over the engine value, of the runtime code the property fast path and the per-row
property read run (proofs/runtime_core `Accessors` models the same code over a three-value
`Val`, with the batch read left as a hypothesis `hbg`):

| here | there |
| --- | --- |
| `St` | the state a read sees: `Runtime.deleted_{nodes,relationships}` (as association lists: `get` = `lookup`, `is_empty` = `[]`), `Pending.{new,existing}_*_attrs` (only their emptiness is inspected here; the lookup itself is `pendingN`/`pendingR`), the committed `AttributeStore`s (`packedN`/`packedR`: `data.get(key)?.get(idx)`, attribute_store.rs:1171), and the attribute-id resolution `node_attr_id` / `rel_attr_id` (runtime.rs:1432-1457, memoized `attrs_name.get_index_of(attr) as u16`, graph.rs:1904-1927; the memo returns what the lookup returns, proofs/runtime_core `attrIdM`) |
| `pendGetN` / `pendGetR` | `Pending::get_node_attribute` / `get_relationship_attribute` (pending.rs:614-630, :963-979): `None` unless `has_*_attrs()` |
| `getNodeAttribute` | `Runtime::get_node_attribute` + `_no_delete_check` (runtime.rs:1459-1490) |
| `getRelAttribute` | `Runtime::get_relationship_attribute` (runtime.rs:1511-1535), its committed fallback by name = `attr_by_name` (graph.rs:3490: `get_index_of(attr)? as u16`, then `get_attr_by_idx`) |
| `matNode` / `matRel` | `materialize_{node,relationship}_property_values` (runtime.rs:1556-1598, :1616-1658), the batch read `get_attrs_by_idx_batch_into` (attribute_store.rs:1182-1193) |
| `nodeStore` | the same state as `PRArms.NodeStore` (what `materialize_node_property_exact` reads) |

**`matNode_eq`**, **`matRel_eq`**: the bulk read is, id for id, the per-row read (or
`Null`) — on the read-only fast path and on the deleted / pending path alike. So
`propReads` proves `CoreBridge.PropReads` for the runtime built from the state (`rtOf`), and
**`nodeStore_matValues`** shows the PR's exact fetch reads that same state.
-/

namespace Pr3437.Runtime
open FalkorCore (V)
open Columnar (FloatModel)
open Columnar.CoreOps (RT)
open Pr3437.PRArms (NodeStore withStore)

variable {F : Type}

/-- The state the attribute reads see. -/
structure St (F : Type) where
  attrId : String → Option Nat
  delNodes : List (Nat × List (String × V F))
  delRels : List (Nat × List (String × V F))
  pendNodesEmpty : Bool        -- `new_nodes_attrs.is_empty() && existing_nodes_attrs.is_empty()`
  pendRelsEmpty : Bool
  pendingN : Nat → Nat → Option (V F)   -- the map lookups of `Pending::get_node_attribute`
  pendingR : Nat → Nat → Option (V F)
  packedN : Nat → Nat → Option ExactProp.Packed   -- `node_attrs.data.get(key)?.get(idx)`
  packedR : Nat → Nat → Option (V F)              -- `relationship_attrs` (unpacked)
  unpackOther : ExactProp.Packed → V F

variable (s : St F)

def byIdxN (id i : Nat) : Option (V F) := (s.packedN id i).map (ExactProp.unpack s.unpackOther)
def byIdxR (id i : Nat) : Option (V F) := s.packedR id i

/-- `has_node_attrs()` / `has_relationship_attrs()` (pending.rs:603-611). -/
def hasNodeAttrs : Bool := !s.pendNodesEmpty
def hasRelAttrs : Bool := !s.pendRelsEmpty

def pendGetN (id i : Nat) : Option (V F) := if !hasNodeAttrs s then none else s.pendingN id i
def pendGetR (id i : Nat) : Option (V F) := if !hasRelAttrs s then none else s.pendingR id i

/-- `get_node_attribute_no_delete_check` (runtime.rs:1479). -/
def noDeleteN (id : Nat) (a : String) : Option (V F) :=
  match s.attrId a with
  | none => none
  | some i => match pendGetN s id i with
    | some v => some v
    | none => byIdxN s id i

/-- `Runtime::get_node_attribute` (runtime.rs:1459). -/
def getNodeAttribute (id : Nat) (a : String) : Option (V F) :=
  match (if s.delNodes.isEmpty then none else s.delNodes.lookup id) with
  | some attrs => attrs.lookup a
  | none => noDeleteN s id a

/-- `Runtime::get_relationship_attribute` (runtime.rs:1511). -/
def getRelAttribute (id : Nat) (a : String) : Option (V F) :=
  match (if s.delRels.isEmpty then none else s.delRels.lookup id) with
  | some attrs => attrs.lookup a
  | none =>
    match (s.attrId a).bind (fun i => pendGetR s id i) with
    | some v => some v
    | none => (s.attrId a).bind (byIdxR s id)   -- `g.get_relationship_attribute(id, attr)`

/-- `materialize_node_property_values` (runtime.rs:1556). -/
def matNode (ids : List Nat) (a : String) : List (V F) :=
  if s.delNodes.isEmpty && !hasNodeAttrs s then
    match s.attrId a with
    | some i => ids.map fun id => (byIdxN s id i).getD .null
    | none => ids.map fun _ => .null
  else ids.map fun id =>
    match s.delNodes.lookup id with
    | none => ((s.attrId a).bind fun i => match pendGetN s id i with
        | some v => some v
        | none => byIdxN s id i).getD .null
    | some attrs => (attrs.lookup a).getD .null

/-- `materialize_relationship_property_values` (runtime.rs:1616). -/
def matRel (ids : List Nat) (a : String) : List (V F) :=
  if s.delRels.isEmpty && !hasRelAttrs s then
    match s.attrId a with
    | some i => ids.map fun id => (byIdxR s id i).getD .null
    | none => ids.map fun _ => .null
  else ids.map fun id =>
    match s.delRels.lookup id with
    | none => ((s.attrId a).bind fun i => match pendGetR s id i with
        | some v => some v
        | none => byIdxR s id i).getD .null
    | some attrs => (attrs.lookup a).getD .null

theorem lookup_nil_isEmpty {β : Type} (l : List (Nat × β)) (h : l.isEmpty = true) (id : Nat) :
    l.lookup id = none := by
  cases l with
  | nil => rfl
  | cons _ _ => simp at h

/-- **The bulk node read is the per-row read**, id for id. -/
theorem matNode_eq (ids : List Nat) (a : String) :
    matNode s ids a = ids.map fun id => (getNodeAttribute s id a).getD .null := by
  unfold matNode
  split
  · rename_i hfast
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hfast
    have hp : ∀ id i, pendGetN s id i = none := by intro id i; simp [pendGetN, hfast.2]
    split
    · rename_i i hi
      apply List.map_congr_left; intro id _
      simp [getNodeAttribute, hfast.1, noDeleteN, hi, hp]
    · rename_i hi
      apply List.map_congr_left; intro id _
      simp [getNodeAttribute, hfast.1, noDeleteN, hi]
  · apply List.map_congr_left; intro id _
    simp only [getNodeAttribute]
    cases he : s.delNodes.isEmpty
    · simp only [Bool.false_eq_true, ite_false]
      cases s.delNodes.lookup id with
      | some attrs => rfl
      | none => simp only [noDeleteN]; cases s.attrId a <;> rfl
    · simp only [ite_true, lookup_nil_isEmpty s.delNodes he id]
      simp only [noDeleteN]; cases s.attrId a <;> rfl

/-- **The bulk relationship read is the per-row read** (the per-row committed fallback is by
name, which resolves through the same attribute id). -/
theorem matRel_eq (ids : List Nat) (a : String) :
    matRel s ids a = ids.map fun id => (getRelAttribute s id a).getD .null := by
  unfold matRel
  split
  · rename_i hfast
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at hfast
    have hp : ∀ id i, pendGetR s id i = none := by intro id i; simp [pendGetR, hfast.2]
    split
    · rename_i i hi
      apply List.map_congr_left; intro id _
      simp [getRelAttribute, hfast.1, hi, hp]
    · rename_i hi
      apply List.map_congr_left; intro id _
      simp [getRelAttribute, hfast.1, hi]
  · apply List.map_congr_left; intro id _
    simp only [getRelAttribute]
    cases he : s.delRels.isEmpty
    · simp only [Bool.false_eq_true, ite_false]
      cases s.delRels.lookup id with
      | some attrs => rfl
      | none => cases hai : s.attrId a <;> simp [hai] <;> split <;> simp_all
    · simp only [ite_true, lookup_nil_isEmpty s.delRels he id]
      cases hai : s.attrId a <;> simp [hai] <;> split <;> simp_all

/-! ## The runtime built from the state -/

/-- The runtime record whose attribute reads (per row and bulk) are the state's. -/
def rtOf (rt : RT F) : RT F :=
  { rt with
    nodeAttr := getNodeAttribute s
    relAttr := getRelAttribute s
    matNode := matNode s
    matRel := matRel s }

/-- What `materialize_node_property_exact` reads (runtime.rs:1566-1587): the same store. -/
def nodeStore : NodeStore F where
  entry a k := match s.attrId a with
    | some i => s.packedN k i
    | none => none
  unpackOther := s.unpackOther
  attrIdx := s.attrId
  deletedEmpty := s.delNodes.isEmpty
  noPendingAttrs := !hasNodeAttrs s
  slowAt a id :=
    match s.delNodes.lookup id with
    | none => ((s.attrId a).bind fun i => match pendGetN s id i with
        | some v => some v
        | none => byIdxN s id i).getD .null
    | some attrs => (attrs.lookup a).getD .null

/-- The exact fetch's `materialize_node_property_values` is the state's. -/
theorem nodeStore_matValues (ids : List Nat) (a : String) :
    (nodeStore s).matValues ids a = matNode s ids a := by
  simp only [NodeStore.matValues, ExactProp.matValues, nodeStore, matNode]
  by_cases hc : (s.delNodes.isEmpty && !hasNodeAttrs s) = true
  · simp only [hc, ite_true]
    cases hi : s.attrId a with
    | some i => simp [ExactProp.batchValues, hi, byIdxN]
    | none =>
      apply List.ext_getElem
      · simp
      · intro i h1 h2; simp
  · simp only [hc, Bool.false_eq_true, ite_false]

/-- `withStore` over the state's node store is the state's runtime. -/
theorem withStore_rtOf (rt : RT F) : withStore (rtOf s rt) (nodeStore s) = rtOf s rt := by
  simp only [withStore, rtOf]
  congr 1
  funext ids a
  exact nodeStore_matValues s ids a

/-- **`PropReads` holds**: no runtime hypothesis remains for the property fast path. -/
theorem propReads (rt : RT F) : Columnar.CoreBridge.PropReads (rtOf s rt) where
  matNode ids a := matNode_eq s ids a
  matRel ids a := matRel_eq s ids a

end Pr3437.Runtime
