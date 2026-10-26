import Columnar.CoreBridge

/-!
# The runtime's attribute reads: the bulk fetch is the per-row read (`PropReads`)

A copy, over the engine value, of the runtime code the property fast path (`property_values`,
`eval_batch`) and the per-row property read (`ExprEval`'s `Property` arm) run. proofs/runtime_core
`Accessors`/`RelAccessors` model the same code over a three-value `Val` and leave the graph's
batch read and by-name read as hypotheses (`hbg`, `hName`); here those graph reads are their
definitions, so the agreement needs no premise.

| here | there |
| --- | --- |
| `St` | the state a read sees: `Runtime.deleted_{nodes,relationships}` (association lists: `get` = `lookup`, `is_empty` = `[]`), whether `Pending.{new,existing}_*_attrs` are empty (`pendingN`/`pendingR` are the map lookups of `Pending::get_*_attribute`), the committed stores' `get_attr_by_idx` (`byIdxN`/`byIdxR`, attribute_store.rs:1171), and attribute-id resolution `node_attr_id`/`rel_attr_id` (runtime.rs:1432-1457: memoized `attrs_name.get_index_of(attr) as u16`, graph.rs:1904-1927; the memo answers the lookup, proofs/runtime_core `attrIdM_correct`) |
| `pendGetN` / `pendGetR` | `Pending::get_node_attribute` / `get_relationship_attribute` (pending.rs:614-630, :963-979): `None` unless `has_*_attrs()` (:603-611) |
| `getNodeAttribute` | `Runtime::get_node_attribute` + `_no_delete_check` (runtime.rs:1459-1490) |
| `getRelAttribute` | `Runtime::get_relationship_attribute` (runtime.rs:1511-1535); its committed fallback `g.get_relationship_attribute(id, attr)` is `attr_by_name` (graph.rs:3490: `get_index_of(attr)? as u16`, then `get_attr_by_idx`) |
| `matNode` / `matRel` | `materialize_{node,relationship}_property_values` (runtime.rs:1668-1710, :1728-1770); the hot path's `get_{node,relationship}_attributes_by_idx` (graph.rs:2587, :3430) is `get_attrs_by_idx_batch_into` (attribute_store.rs:1182-1193), the same lookup per key with `Null` as default |

**`matNode_eq`**, **`matRel_eq`**: the bulk read is, id for id, the per-row read (or `Null`),
on the read-only fast path and on the deleted / pending path alike; **`propReads`**: the
runtime record built from the state (`rtOf`) satisfies `CoreBridge.PropReads`.
-/

namespace Columnar.RuntimeReads
open FalkorCore (V)
open Columnar (FloatModel Column Bits classifyColumn)
open Columnar.CoreOps (RT)

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
  /-- `node_attrs.get_attr_by_idx(key, idx)` = `data.get(key)?.get(idx)` (attribute_store.rs:1171) -/
  byIdxN : Nat → Nat → Option (V F)
  byIdxR : Nat → Nat → Option (V F)

variable (s : St F)

def byIdxN (id i : Nat) : Option (V F) := s.byIdxN id i
def byIdxR (id i : Nat) : Option (V F) := s.byIdxR id i

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

/-- `materialize_node_property_values` (runtime.rs:1668). -/
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

/-- `materialize_relationship_property_values` (runtime.rs:1728). -/
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

/-- **`PropReads` holds** for the runtime built from the state. -/
theorem propReads (rt : RT F) : Columnar.CoreBridge.PropReads (rtOf s rt) where
  matNode ids a := matNode_eq s ids a
  matRel ids a := matRel_eq s ids a

/-- `materialize_node_property` (runtime.rs:1654): `classify_column` of the values. -/
def matNodeCol [FloatModel F] (ids : List Nat) (a : String) : Column F × Bits := classifyColumn (matNode s ids a)
def matRelCol [FloatModel F] (ids : List Nat) (a : String) : Column F × Bits := classifyColumn (matRel s ids a)

theorem matNodeCol_eq [FloatModel F] (ids : List Nat) (a : String) :
    matNodeCol s ids a = classifyColumn (ids.map fun id => (getNodeAttribute s id a).getD .null) := by
  simp [matNodeCol, matNode_eq]
theorem matRelCol_eq [FloatModel F] (ids : List Nat) (a : String) :
    matRelCol s ids a = classifyColumn (ids.map fun id => (getRelAttribute s id a).getD .null) := by
  simp [matRelCol, matRel_eq]

end Columnar.RuntimeReads
