/-
# Relationship accessors, pending attribute writes, typed materialisation, `label_id`

| here | there |
| --- | --- |
| `relNoDelete`              | `Runtime::get_relationship_attribute_no_delete_check`, `runtime/runtime.rs:1605-1620` |
| `getRelAttribute`          | `Runtime::get_relationship_attribute`, `runtime.rs:1641-1667` (committed fallback BY NAME: `Graph::get_relationship_attribute`, `graph/graph.rs:3480`) |
| `materializeRel`           | `Runtime::materialize_relationship_property_values`, `runtime.rs:1802-1846` |
| `materializeNodeProp`, `materializeRelProp` | `materialize_{node,relationship}_property`, `runtime.rs:1654-1660`, `:1790-1796` (`classify_column` ∘ values) |
| `getRelAttrs`              | `Runtime::get_relationship_attrs` (`runtime.rs:1963-1982`) + `Pending::update_relationship_attrs` (`pending.rs:1044-1066`) |
| `relEndpoints`, `relType`  | `Runtime::get_relationship_endpoints` (`:1984-1997`), `get_relationship_type` (`:1999-2013`) |
| `relMask`, `maskDel`       | `is_reissued_relationship` (`:1694-1699`): every relationship read above skips the snapshot of a reissued id (#3051) |
| `RDel`, `isRelDeleted`, `recordDel`, `noteCreated` | `is_relationship_deleted` (`:1679-1689`), `record_deleted_relationship` (`:1703-1710`), `note_relationship_created` (`:1714-1721`) |
| `labelIdOf`                | `Runtime::label_id` (`:1838-1843`) |
| `setPendingAttr`           | `Runtime::set_pending_node_attr` (`:1483-1498`) / `set_pending_relationship_attr` (`:1502-1517`) |

The relationship stores have exactly the node stores' shape (`Pending::{new,existing}_relationships_attrs` / `update_relationship_attrs`
(`pending.rs:1044-1066`) vs `{new,existing}_nodes_attrs` / `update_node_attrs` (`:662-684`): same `has_*_attrs`, `new`-then-`existing` `lookup_sorted`, same one-map
`update_*_attrs`), so a relationship read is `Accessors.W` instantiated with the
relationship maps (`deleted_relationships`, `{new,existing}_relationships_attrs`,
`Graph::get_relationship_attribute_by_idx`, `rel_attr_id`/`rel_attr_name`). The one code
difference — `get_relationship_attribute`'s committed fallback reads by NAME, not by the
resolved index — is modelled explicitly (`byName`) and discharged by the store contract
`hName : byName id n = (attrId n).bind (committed id)`.

Since #3051 every relationship read consults the snapshot map only for an id that is not in
`reissued_relationships` (`re`): the read is the node accessor over `relMask w re` (the map
with reissued ids masked out), and exactly the old read for an id that was not reissued
(`getRelAttribute_unreissued`).
-/
import RuntimeCore.Accessors

namespace RuntimeCore.RelAccessors
open RuntimeCore.Accessors

/-- `get_relationship_attribute_no_delete_check` (`runtime.rs:1605-1620`): resolve the id
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

/-- The snapshot map as the relationship reads see it (#3051): no entry for a reissued id. -/
def maskDel {X : Type} (re : Nat → Bool) (del : Nat → Option X) (id : Nat) : Option X :=
  if re id then none else del id

/-- `w` with the reissued ids' snapshots masked out (`deletedEmpty` is the whole map's). -/
def relMask (w : W) (re : Nat → Bool) : W := { w with deleted := maskDel re w.deleted }

theorem relMask_deletedOk (w : W) (re : Nat → Bool) (hd : w.deletedOk) : (relMask w re).deletedOk := by
  intro he id; simp [relMask, maskDel, hd he id]

/-- `get_relationship_attribute` (`runtime.rs:1641-1667`). `byName` is
`Graph::get_relationship_attribute(id, attr)`; `re` is `is_reissued_relationship`. -/
def getRelAttribute (w : W) (re : Nat → Bool) (byName : Nat → String → Option Val) (id : Nat) (name : String) :
    Option Val :=
  match (if !w.deletedEmpty then maskDel re w.deleted id else none) with
  | some dr => dr.attrs.lookup name
  | none =>
    match (w.attrId name).bind (pendingGet w id) with
    | some v => some v
    | none => byName id name

/-- **PROVEN**: under the by-name/by-index store contract, `get_relationship_attribute`
is exactly the node accessor's three-way answer (deleted snapshot > pending > committed)
over the relationship maps. -/
theorem getRelAttribute_eq (w : W) (re : Nat → Bool) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id)) (id : Nat) (name : String) :
    getRelAttribute w re byName id name = getNodeAttribute (relMask w re) id name := by
  have hp : pendingGet (relMask w re) = pendingGet w := rfl
  have hE : (relMask w re).deletedEmpty = w.deletedEmpty := rfl
  have hD : (relMask w re).deleted = maskDel re w.deleted := rfl
  have hA : (relMask w re).attrId = w.attrId := rfl
  have hC : (relMask w re).committed = w.committed := rfl
  simp only [getRelAttribute, getNodeAttribute, noDelete, hp, hE, hD, hA, hC]
  generalize (if (!w.deletedEmpty) = true then maskDel re w.deleted id else none) = d
  cases d with
  | some dr => rfl
  | none =>
    simp only
    rw [hName]
    cases w.attrId name with
    | none => rfl
    | some a => simp only [Option.bind_some]; cases pendingGet w id a <;> rfl

/-- **PROVEN** (#3051): for an id that was not reissued the read is the old one, the node
accessor's three-way answer over the unmasked maps (deleted snapshot > pending > committed). -/
theorem getRelAttribute_unreissued (w : W) (re : Nat → Bool) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id)) (id : Nat) (name : String)
    (hr : re id = false) :
    getRelAttribute w re byName id name = getNodeAttribute w id name := by
  rw [getRelAttribute_eq w re byName hName]
  simp only [getNodeAttribute, relMask, maskDel, hr, Bool.false_eq_true, ite_false]
  rfl

/-- **PROVEN** (#3051): a reissued id reads the new relationship (pending, else committed),
never the previous owner's snapshot. -/
theorem getRelAttribute_reissued (w : W) (re : Nat → Bool) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id)) (id : Nat) (name : String)
    (hr : re id = true) :
    getRelAttribute w re byName id name = relNoDelete w id name := by
  rw [getRelAttribute_eq w re byName hName]
  simp only [getNodeAttribute, relMask, maskDel, hr, ite_true, ite_self, relNoDelete_eq, noDelete]
  rfl

/-- `materialize_relationship_property_values` (`runtime.rs:1802-1846`): the same code as
the node one over the relationship maps (snapshot masked for reissued ids) and batched
`get_relationship_attributes_by_idx`. -/
def materializeRel (w : W) (re : Nat → Bool) (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) :
    List Val :=
  let idx := w.attrId name
  if w.deletedEmpty && !w.hasAttrs then
    match idx with
    | some i => bg ids i
    | none => ids.map (fun _ => .null)
  else
    ids.map (fun id =>
      match maskDel re w.deleted id with
      | none =>
        (idx.bind (fun i =>
          match pendingGet w id i with
          | some v => some v
          | none => w.committed id i)).getD .null
      | some dr => (dr.attrs.lookup name).getD .null)

/-- **PROVEN**: the bulk relationship column read equals the per-row
`get_relationship_attribute`, on the hot and on the overlay path. -/
theorem materializeRel_eq (w : W) (re : Nat → Bool) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id))
    (bg : List Nat → Nat → List Val)
    (hbg : ∀ ids i, bg ids i = ids.map (fun id => (w.committed id i).getD .null))
    (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materializeRel w re bg ids name =
      ids.map (fun id => (getRelAttribute w re byName id name).getD .null) := by
  have : materializeRel w re bg ids name = materialize (relMask w re) bg ids name := rfl
  rw [this, materialize_eq (relMask w re) bg hbg (relMask_deletedOk w re hd)]
  apply List.map_congr_left
  intro id _
  rw [getRelAttribute_eq w re byName hName]

/-- `materialize_node_property` (`runtime.rs:1654-1660`): `classify_column` of the values
(`classify` abstract; its spec is `proofs/columnar` `classifyColumn_*`). -/
def materializeNodeProp {C : Type} (classify : List Val → C) (w : W)
    (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) : C :=
  classify (materialize w bg ids name)

/-- `materialize_relationship_property` (`runtime.rs:1790-1796`). -/
def materializeRelProp {C : Type} (classify : List Val → C) (w : W) (re : Nat → Bool)
    (bg : List Nat → Nat → List Val) (ids : List Nat) (name : String) : C :=
  classify (materializeRel w re bg ids name)

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
theorem materializeRelProp_eq {C : Type} (classify : List Val → C) (w : W) (re : Nat → Bool)
    (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id))
    (bg : List Nat → Nat → List Val)
    (hbg : ∀ ids i, bg ids i = ids.map (fun id => (w.committed id i).getD .null))
    (hd : w.deletedOk) (ids : List Nat) (name : String) :
    materializeRelProp classify w re bg ids name =
      classify (ids.map (fun id => (getRelAttribute w re byName id name).getD .null)) := by
  simp only [materializeRelProp, materializeRel_eq w re byName hName bg hbg hd]

/-- `get_relationship_attrs` (`runtime.rs:1963-1982`): deleted snapshot (unless the id was
reissued, #3051), else the committed listing overlaid by `update_relationship_attrs` (one
pending map). -/
def getRelAttrs (w : W) (re : Nat → Bool) (id : Nat) : OM :=
  match maskDel re w.deleted id with
  | some dr => dr.attrs
  | none => updateNodeAttrs w id (w.committedAll id)

/-- **PROVEN**: `properties(r).k` agrees with `r.k` under the same contracts as the node
theorem `Accessors.attrs_agree`, with the by-name committed read. -/
theorem relAttrs_agree (w : W) (re : Nat → Bool) (byName : Nat → String → Option Val)
    (hName : ∀ id n, byName id n = (w.attrId n).bind (w.committed id))
    (hn : w.nameOk) (hd : w.deletedOk) (ha : w.attrsOk)
    (hc : ∀ id name, OM.lookup name (w.committedAll id) = (w.attrId name).bind (w.committed id))
    (hdisj : ∀ id, w.newAttrs id = none ∨ w.exAttrs id = none)
    (huniq : ∀ id l, (w.newAttrs id = some l ∨ w.exAttrs id = some l) → (l.map Prod.fst).Nodup)
    (id : Nat) (name : String) :
    Val.norm (OM.lookup name (getRelAttrs w re id)) = Val.norm (getRelAttribute w re byName id name) := by
  rw [getRelAttribute_eq w re byName hName]
  exact attrs_agree (relMask w re) hn (relMask_deletedOk w re hd) ha hc hdisj huniq id name

/-! ### Endpoints and type -/

structure DelRel where
  src : Nat
  dst : Nat
  ty : String

/-- `get_relationship_endpoints` (`runtime.rs:1984-1997`); `re` = `is_reissued_relationship`. -/
def relEndpoints (del : Nat → Option DelRel) (re : Nat → Bool) (pend : Nat → Option (Nat × Nat))
    (com : Nat → Nat × Nat) (id : Nat) : Nat × Nat :=
  match maskDel re del id with
  | some dr => (dr.src, dr.dst)
  | none =>
    match pend id with
    | some e => e
    | none => com id

/-- `get_relationship_type` (`runtime.rs:1999-2013`); `com` = `g.get_type(g.get_relationship_type_id(id))`. -/
def relType (del : Nat → Option DelRel) (re : Nat → Bool) (pend : Nat → Option String)
    (com : Nat → Option String) (id : Nat) : Option String :=
  match maskDel re del id with
  | some dr => some dr.ty
  | none =>
    match pend id with
    | some t => some t
    | none => com id

/-- **PROVEN**: endpoints come from the delete snapshot (of a non-reissued id), else this
query's created relationship, else the committed graph — `relLookup` order. -/
theorem relEndpoints_spec (del : Nat → Option DelRel) (re : Nat → Bool) (pend : Nat → Option (Nat × Nat))
    (com : Nat → Nat × Nat) (id : Nat) :
    relEndpoints del re pend com id =
      relLookup (fun i => (maskDel re del i).map (fun d => (d.src, d.dst))) pend com id := by
  simp only [relEndpoints, relLookup]
  cases maskDel re del id <;> cases pend id <;> simp

/-- **PROVEN**: same order for the type; a deleted (not reissued) or pending relationship
always has one. -/
theorem relType_spec (del : Nat → Option DelRel) (re : Nat → Bool) (pend : Nat → Option String)
    (com : Nat → Option String) (id : Nat) :
    relType del re pend com id =
      ((maskDel re del id).map (·.ty)).or ((pend id).or (com id)) ∧
    ((maskDel re del id).isSome || (pend id).isSome → (relType del re pend com id).isSome) := by
  simp only [relType]
  cases maskDel re del id <;> cases pend id <;> simp

/-- **PROVEN** (#3051): a reissued id's endpoints and type are the new relationship's
(pending, else committed); a non-reissued id keeps the old snapshot-first answer. -/
theorem relEndpoints_relType_reissued (del : Nat → Option DelRel) (re : Nat → Bool)
    (pe : Nat → Option (Nat × Nat)) (ce : Nat → Nat × Nat) (pt : Nat → Option String)
    (ct : Nat → Option String) (id : Nat) :
    (re id = true → relEndpoints del re pe ce id = relLookup (fun _ => none) pe ce id ∧
      relType del re pt ct id = (pt id).or (ct id)) ∧
    (re id = false → relEndpoints del re pe ce id =
        relLookup (fun i => (del i).map (fun d => (d.src, d.dst))) pe ce id ∧
      relType del re pt ct id = ((del id).map (·.ty)).or ((pt id).or (ct id))) := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · simp only [relEndpoints, relType, relLookup, maskDel, h, ite_true]
    cases pe id <;> cases pt id <;> simp
  · simp only [relEndpoints, relType, relLookup, maskDel, h, Bool.false_eq_true, ite_false]
    cases del id <;> cases pe id <;> cases pt id <;> simp

/-! ### `is_relationship_deleted`, `record_deleted_relationship`, `note_relationship_created` (#3051)

The id-indexed records the write-skip check reads: `Pending::is_relationship_deleted` (`pd`),
`Graph::is_relationship_deleted` (`gd`), `Pending::is_relationship_created` (`pc`),
`reissued_relationships` (`re`) and `deleted_relationships` (`snap`). -/

structure RDel (X : Type) where
  pd : Nat → Bool
  gd : Nat → Bool
  pc : Nat → Bool
  re : Nat → Bool
  snap : Nat → Option X

/-- `is_relationship_deleted` (`runtime.rs:1679-1689`). -/
def isRelDeleted {X : Type} (st : RDel X) (id : Nat) : Bool :=
  st.pd id || (st.gd id && !st.pc id) || (!st.pc id && !st.re id && (st.snap id).isSome)

/-- `record_deleted_relationship` (`runtime.rs:1703-1710`): unmark, then snapshot. -/
def recordDel {X : Type} (st : RDel X) (id : Nat) (x : X) : RDel X :=
  { st with re := fun j => j != id && st.re j, snap := fun j => if j = id then some x else st.snap j }

/-- `note_relationship_created` (`runtime.rs:1714-1721`): mark the id if it has a snapshot. -/
def noteCreated {X : Type} (st : RDel X) (id : Nat) : RDel X :=
  { st with re := fun j => (j == id && (st.snap id).isSome) || st.re j }

/-- **PROVEN**: the write-skip check is the old one (pending-deleted, or graph-deleted and not
re-created) plus the snapshot map, which is the only record of an edge cascaded by `DETACH
DELETE` before commit (#2967), and which does not speak for a re-created or reissued id. -/
theorem isRelDeleted_spec {X : Type} (st : RDel X) (id : Nat) :
    (st.pd id = true → isRelDeleted st id = true) ∧
    (st.gd id = true → st.pc id = false → isRelDeleted st id = true) ∧
    (st.pc id = false → st.re id = false → (st.snap id).isSome → isRelDeleted st id = true) ∧
    (st.pd id = false → st.pc id = true → isRelDeleted st id = false) ∧
    (st.pd id = false → st.gd id = false → st.re id = true → isRelDeleted st id = false) := by
  refine ⟨fun h => ?_, fun h1 h2 => ?_, fun h1 h2 h3 => ?_, fun h1 h2 => ?_, fun h1 h2 h3 => ?_⟩ <;>
    simp_all [isRelDeleted]

/-- **PROVEN**: after recording a snapshot the id is no longer reissued, its snapshot is the
recorded one, and (unless still pending-created) a write to it is skipped; other ids keep
their marks and snapshots. -/
theorem recordDel_spec {X : Type} (st : RDel X) (id : Nat) (x : X) :
    (recordDel st id x).re id = false ∧ (recordDel st id x).snap id = some x ∧
    (st.pc id = false → isRelDeleted (recordDel st id x) id = true) ∧
    (∀ j, j ≠ id → (recordDel st id x).re j = st.re j ∧ (recordDel st id x).snap j = st.snap j) := by
  refine ⟨by simp [recordDel], by simp [recordDel], fun h => by simp [isRelDeleted, recordDel, h],
    fun j hj => by simp [recordDel, hj]⟩

/-- **PROVEN**: creating an id that has a snapshot marks it reissued, so the snapshot no longer
masks the new relationship (reads: `maskDel` is `none`; writes: only the pending/graph records
count); an id without a snapshot, and every other id, is unchanged. -/
theorem noteCreated_spec {X : Type} (st : RDel X) (id : Nat) :
    ((st.snap id).isSome → (noteCreated st id).re id = true ∧
      maskDel (noteCreated st id).re (noteCreated st id).snap id = none ∧
      isRelDeleted (noteCreated st id) id = (st.pd id || (st.gd id && !st.pc id))) ∧
    ((st.snap id).isSome = false → noteCreated st id = st) ∧
    (∀ j, j ≠ id → (noteCreated st id).re j = st.re j) := by
  refine ⟨fun h => ?_, fun h => ?_, fun j hj => by simp [noteCreated, hj]⟩
  · refine ⟨by simp [noteCreated, h], by simp [maskDel, noteCreated, h], ?_⟩
    simp [isRelDeleted, noteCreated, h]
  · cases st with
    | mk pd gd pc re snap =>
      simp only at h
      simp only [noteCreated, RDel.mk.injEq, true_and, and_true]
      funext j; simp [h]

/-! ### `label_id` -/

/-- `Runtime::label_id` (`runtime.rs:1838-1843`): the graph's label-name table. -/
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

/-- `set_pending_node_attr` (`runtime.rs:1483-1498`) / `set_pending_relationship_attr`
(`:1502-1517`), generic over the name table `Tb` (node or relationship attribute names),
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

/-- Contract of `get_or_create_*_attr_id` (`graph.rs:2048`, `get_or_create_rel_attr_id`; node variant `:1979`, `attrs_name.get_or_create`):
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
