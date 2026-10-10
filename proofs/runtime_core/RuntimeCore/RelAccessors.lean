/-
# Relationship accessors, pending attribute writes, typed materialisation, `label_id`

| here | there |
| --- | --- |
| `relNoDelete`              | `Runtime::get_relationship_attribute_no_delete_check`, `runtime/runtime.rs:1493-1508` |
| `getRelAttribute`          | `Runtime::get_relationship_attribute`, `runtime.rs:1510-1535` (committed fallback BY NAME: `Graph::get_relationship_attribute`, `graph/graph.rs:3163`) |
| `materializeRel`           | `Runtime::materialize_relationship_property_values`, `runtime.rs:1616-1658` |
| `materializeNodeProp`, `materializeRelProp` | `materialize_{node,relationship}_property`, `runtime.rs:1542-1551`, `:1604-1613` (`classify_column` ∘ values) |
| `getRelAttrs`              | `Runtime::get_relationship_attrs` (`runtime.rs:1757-1775`) + `Pending::update_relationship_attrs` (`pending.rs:852-873`) |
| `relEndpoints`, `relType`  | `Runtime::get_relationship_endpoints` (`:1777-1788`), `get_relationship_type` (`:1790-1802`) |
| `labelIdOf`                | `Runtime::label_id` (`:1711-1716`) |
| `setPendingAttr`           | `Runtime::set_pending_node_attr` (`:1371-1388`) / `set_pending_relationship_attr` (`:1390-1405`) |

The relationship stores have exactly the node stores' shape (`pending.rs:480,863-902` vs
`:529,541-581`: same `has_*_attrs`, `new`-then-`existing` `lookup_sorted`, same one-map
`update_*_attrs`), so a relationship read is `Accessors.W` instantiated with the
relationship maps (`deleted_relationships`, `{new,existing}_relationships_attrs`,
`Graph::get_relationship_attribute_by_idx`, `rel_attr_id`/`rel_attr_name`). The one code
difference — `get_relationship_attribute`'s committed fallback reads by NAME, not by the
resolved index — is modelled explicitly (`byName`) and discharged by the store contract
`hName : byName id n = (attrId n).bind (committed id)`.
-/
import RuntimeCore.Accessors

namespace RuntimeCore.RelAccessors
open RuntimeCore.Accessors

/-- `get_relationship_attribute_no_delete_check` (`runtime.rs:1493-1508`): resolve the id
(memo = table, `Accessors.attrIdM_correct`), pending first, else committed by index. -/
def relNoDelete (w : W) (id : Nat) (name : String) : Option Val :=
  match w.attrId name with
  | none => none
  | some a =>
    match pendingGet w id a with
    | some v => some v
    | none => w.committed id a

/-- **PROVEN**: the relationship no-delete-check read is the node one over the
relationship maps. -/
theorem relNoDelete_eq (w : W) (id : Nat) (name : String) :
    relNoDelete w id name = noDelete w id name := rfl

/-- **PROVEN** (spec): exactly pending (if the name is registered), else committed. -/
theorem relNoDelete_spec (w : W) (id : Nat) (name : String) :
    relNoDelete w id name =
      (w.attrId name).bind (fun a => (pendingGet w id a).or (w.committed id a)) := by
  simp only [relNoDelete]
  cases w.attrId name with
  | none => rfl
  | some a => cases h : pendingGet w id a <;> simp [h, Option.or]

/-- `get_relationship_attribute` (`runtime.rs:1510-1535`). `byName` is
`Graph::get_relationship_attribute(id, attr)`. -/
def getRelAttribute (w : W) (byName : Nat → String → Option Val) (id : Nat) (name : String) :
    Option Val :=
  match (if !w.deletedEmpty then w.deleted id else none) with
  | some dr => dr.attrs.lookup name
  | none =>
    match (w.attrId name).bind (pendingGet w id) with
    | some v => some v
    | none => byName id name

/-- **PROVEN**: under the by-name/by-index store contract, `get_relationship_attribute`
is exactly the node accessor's three-way answer (deleted snapshot > pending > committed)
over the relationship maps. -/
theorem getRelAttribute_eq (w : W) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id)) (id : Nat) (name : String) :
    getRelAttribute w byName id name = getNodeAttribute w id name := by
  simp only [getRelAttribute, getNodeAttribute, noDelete]
  generalize (if (!w.deletedEmpty) = true then w.deleted id else none) = d
  cases d with
  | some dr => rfl
  | none =>
    simp only
    rw [hName]
    cases w.attrId name with
    | none => rfl
    | some a => simp only [Option.bind_some]; cases pendingGet w id a <;> rfl

/-- `materialize_relationship_property_values` (`runtime.rs:1616-1658`): the same code as
the node one over the relationship maps and batched
`get_relationship_attributes_by_idx`. -/
def materializeRel (w : W) (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) :
    List Val :=
  let idx := w.attrId name
  if w.deletedEmpty && !w.hasAttrs then
    match idx with
    | some i => bg ids i
    | none => ids.map (fun _ => .null)
  else
    ids.map (fun id =>
      match w.deleted id with
      | none =>
        (idx.bind (fun i =>
          match pendingGet w id i with
          | some v => some v
          | none => w.committed id i)).getD .null
      | some dr => (dr.attrs.lookup name).getD .null)

/-- **PROVEN**: the bulk relationship column read equals the per-row
`get_relationship_attribute`, on the hot and on the overlay path. -/
theorem materializeRel_eq (w : W) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id))
    (bg : List Nat → Nat → List Val)
    (hbg : ∀ ids i, bg ids i = ids.map (fun id => (w.committed id i).getD .null))
    (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materializeRel w bg ids name =
      ids.map (fun id => (getRelAttribute w byName id name).getD .null) := by
  have : materializeRel w bg ids name = materialize w bg ids name := rfl
  rw [this, materialize_eq w bg hbg hd]
  apply List.map_congr_left
  intro id _
  rw [getRelAttribute_eq w byName hName]

/-- `materialize_node_property` (`runtime.rs:1542-1551`): `classify_column` of the values
(`classify` abstract; its spec is `proofs/columnar` `classifyColumn_*`). -/
def materializeNodeProp {C : Type} (classify : List Val → C) (w : W)
    (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) : C :=
  classify (materialize w bg ids name)

/-- `materialize_relationship_property` (`runtime.rs:1604-1613`). -/
def materializeRelProp {C : Type} (classify : List Val → C) (w : W)
    (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) : C :=
  classify (materializeRel w bg ids name)

/-- **PROVEN**: the typed node column is `classify_column` of the per-row
`get_node_attribute` values (Null for absent). -/
theorem materializeNodeProp_eq {C : Type} (classify : List Val → C) (w : W)
    (bg : List Nat → Nat → List Val)
    (hbg : ∀ ids i, bg ids i = ids.map (fun id => (w.committed id i).getD .null))
    (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materializeNodeProp classify w bg ids name =
      classify (ids.map (fun id => (getNodeAttribute w id name).getD .null)) := by
  simp only [materializeNodeProp, materialize_eq w bg hbg hd]

/-- **PROVEN**: likewise for relationships. -/
theorem materializeRelProp_eq {C : Type} (classify : List Val → C) (w : W)
    (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id))
    (bg : List Nat → Nat → List Val)
    (hbg : ∀ ids i, bg ids i = ids.map (fun id => (w.committed id i).getD .null))
    (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materializeRelProp classify w bg ids name =
      classify (ids.map (fun id => (getRelAttribute w byName id name).getD .null)) := by
  simp only [materializeRelProp, materializeRel_eq w byName hName bg hbg hd]

/-- `get_relationship_attrs` (`runtime.rs:1757-1775`): deleted snapshot, else the
committed listing overlaid by `update_relationship_attrs` (one pending map). -/
def getRelAttrs (w : W) (id : Nat) : OM :=
  match w.deleted id with
  | some dr => dr.attrs
  | none => updateNodeAttrs w id (w.committedAll id)

/-- **PROVEN**: `properties(r).k` agrees with `r.k` under the same contracts as the node
theorem `Accessors.attrs_agree`, with the by-name committed read. -/
theorem relAttrs_agree (w : W) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id))
    (hn : w.nameOk) (hd : w.deletedOk) (ha : w.attrsOk)
    (hc : ∀ id name, OM.lookup name (w.committedAll id) = (w.attrId name).bind (w.committed id))
    (hdisj : ∀ id, w.newAttrs id = none ∨ w.exAttrs id = none)
    (huniq : ∀ id l, (w.newAttrs id = some l ∨ w.exAttrs id = some l) → (l.map Prod.fst).Nodup)
    (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getRelAttrs w id)) = Val.norm (getRelAttribute w byName id name) := by
  rw [getRelAttribute_eq w byName hName]
  exact attrs_agree w hn hd ha hc hdisj huniq id name

/-! ### Endpoints and type -/

structure DelRel where
  src : Nat
  dst : Nat
  ty : String

/-- `get_relationship_endpoints` (`runtime.rs:1777-1788`). -/
def relEndpoints (del : Nat → Option DelRel) (pend : Nat → Option (Nat × Nat))
    (com : Nat → Nat × Nat) (id : Nat) : Nat × Nat :=
  match del id with
  | some dr => (dr.src, dr.dst)
  | none =>
    match pend id with
    | some e => e
    | none => com id

/-- `get_relationship_type` (`runtime.rs:1790-1802`); `com` = `g.get_type(g.get_relationship_type_id(id))`. -/
def relType (del : Nat → Option DelRel) (pend : Nat → Option String)
    (com : Nat → Option String) (id : Nat) : Option String :=
  match del id with
  | some dr => some dr.ty
  | none =>
    match pend id with
    | some t => some t
    | none => com id

/-- **PROVEN**: endpoints come from the delete snapshot, else this query's created
relationship, else the committed graph — `relLookup` order. -/
theorem relEndpoints_spec (del : Nat → Option DelRel) (pend : Nat → Option (Nat × Nat))
    (com : Nat → Nat × Nat) (id : Nat) :
    relEndpoints del pend com id =
      relLookup (fun i => (del i).map (fun d => (d.src, d.dst))) pend com id := by
  simp only [relEndpoints, relLookup]
  cases del id <;> cases pend id <;> simp

/-- **PROVEN**: same order for the type; a deleted or pending relationship always has one. -/
theorem relType_spec (del : Nat → Option DelRel) (pend : Nat → Option String)
    (com : Nat → Option String) (id : Nat) :
    relType del pend com id =
      ((del id).map (·.ty)).or ((pend id).or (com id)) ∧
    ((del id).isSome || (pend id).isSome → (relType del pend com id).isSome) := by
  simp only [relType]
  cases del id <;> cases pend id <;> simp

/-! ### `label_id` -/

/-- `Runtime::label_id` (`runtime.rs:1711-1716`): the graph's label-name table. -/
def labelIdOf (w : W) (name : String) : Option Nat := w.labelId name

/-- **PROVEN**: `label_id` returns the table's id, and `node_has_label` is exactly
"resolve with `label_id`, then `node_has_label_id`". -/
theorem labelIdOf_spec (w : W) (id : Nat) (name : String) :
    labelIdOf w name = w.labelId name ∧
    hasLabel w id name = ((labelIdOf w name).map (hasLabelId w id)).getD false := by
  refine ⟨rfl, ?_⟩
  simp only [hasLabel, labelIdOf]
  cases w.labelId name <;> rfl

/-! ### `set_pending_{node,relationship}_attr` -/

/-- `identifier_limits.rs:10`. -/
def MAX_IDENTIFIER_LEN : Nat := 512

/-- `set_pending_node_attr` (`runtime.rs:1371-1388`) / `set_pending_relationship_attr`
(`:1390-1405`), generic over the name table `Tb` (node or relationship attribute names),
its `get_or_create_*_attr_id` (`goc`), and the pending setter `setP`
(`Pending::set_{node,relationship}_attribute`). `maxLen` is `MAX_IDENTIFIER_LEN`
(`validate_identifier_len`, `identifier_limits.rs:26-34`: `name.len()` = UTF-8 bytes). -/
def setPendingAttr {Tb P : Type} (maxLen : Nat) (nameOf : Nat → String)
    (goc : Tb → String → Nat × Tb) (setP : P → Nat → Nat → Val → Except String P)
    (t : Tb) (memo : List (Nat × Nat)) (p : P) (id ptr : Nat) (v : Val) :
    Except String (P × Tb × List (Nat × Nat)) :=
  if (nameOf ptr).utf8ByteSize > maxLen then .error "Property name too long"
  else
    let r : Nat × Tb × List (Nat × Nat) :=
      match memoLookup nameOf memo ptr with
      | some a => (a, t, memo)
      | none =>
        let (a, t') := goc t (nameOf ptr)
        (a, t', memoInsert memo ptr a)
    match setP p id r.1 v with
    | .ok p' => .ok (p', r.2.1, r.2.2)
    | .error e => .error e

/-- Contract of `get_or_create_*_attr_id` (`graph.rs:1777-1782`, `attrs_name.get_or_create`):
the returned id is the name's id afterwards, and no existing name changes id. -/
def GocOk {Tb : Type} (lk : Tb → String → Option Nat) (goc : Tb → String → Nat × Tb) : Prop :=
  ∀ t n, lk (goc t n).2 n = some (goc t n).1 ∧ ∀ n' i, lk t n' = some i → lk (goc t n).2 n' = some i

/-- **PROVEN**: an over-long name is refused before anything is touched; otherwise the
pending write is made with the name's (possibly new) id in the table afterwards, the
table only grows, and the memo stays valid (and within its cap) against the new table. -/
theorem setPendingAttr_spec {Tb P : Type} (maxLen : Nat) (nameOf : Nat → String)
    (lk : Tb → String → Option Nat) (goc : Tb → String → Nat × Tb) (hg : GocOk lk goc)
    (setP : P → Nat → Nat → Val → Except String P)
    (t : Tb) (memo : List (Nat × Nat)) (p : P) (id ptr : Nat) (v : Val)
    (hm : MemoOk nameOf (lk t) memo) :
    ((nameOf ptr).utf8ByteSize > maxLen →
      setPendingAttr maxLen nameOf goc setP t memo p id ptr v = .error "Property name too long") ∧
    ((nameOf ptr).utf8ByteSize ≤ maxLen →
      ∃ a t' memo', lk t' (nameOf ptr) = some a ∧ MemoOk nameOf (lk t') memo' ∧
        (∀ n i, lk t n = some i → lk t' n = some i) ∧
        setPendingAttr maxLen nameOf goc setP t memo p id ptr v =
          (match setP p id a v with
           | .ok p' => .ok (p', t', memo')
           | .error e => .error e)) := by
  refine ⟨fun h => by simp [setPendingAttr, h], fun h => ?_⟩
  have hn : ¬ (nameOf ptr).utf8ByteSize > maxLen := by omega
  simp only [setPendingAttr, hn, ite_false]
  cases hl : memoLookup nameOf memo ptr with
  | some a =>
    exact ⟨a, t, memo, memoLookup_ok hm hl, hm, fun _ _ h => h, rfl⟩
  | none =>
    obtain ⟨h1, h2⟩ := hg t (nameOf ptr)
    have hm' : MemoOk nameOf (lk (goc t (nameOf ptr)).2) memo := memo_ok_mono hm h2
    -- the memo after insert, validated through `attrIdM_correct`
    have hc := attrIdM_correct nameOf (lk (goc t (nameOf ptr)).2) memo ptr hm'
    have hl' : memoLookup nameOf memo ptr = none := hl
    simp only [attrIdM, hl', h1] at hc
    exact ⟨(goc t (nameOf ptr)).1, (goc t (nameOf ptr)).2,
      memoInsert memo ptr (goc t (nameOf ptr)).1, h1, hc.2, h2, rfl⟩

end RuntimeCore.RelAccessors
