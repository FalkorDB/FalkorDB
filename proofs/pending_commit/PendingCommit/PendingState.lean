import PendingCommit.PendingAttrs
/-
# `Pending` bookkeeping (pending.rs:116-1096)

The struct with every field as a finite set (`RoaringTreemap`: a duplicate-free
list) or finite map (`FxHashMap`: `PA.FMap`), and each accessor/mutator as a
function on it. Relationship type names (`Arc<String>`) are `Nat`s.

* attribute staging, generic over node/relationship (`stageOne`, `stageAll`,
  `clearA`, `getA`, `updA`): `setAttr_spec` (the staged list becomes
  `Props.upsert old k v`, validation error leaves `Pending` unchanged),
  `setAttrs_spec`, `clearA_spec`, `getA_spec` (new-entity map first, the
  `has_*_attrs` short-cut changes nothing), `updA_spec` (overlay of the staged
  list on the committed map, last write per name, `Null` removes).
* sets: `created_nodes`, `deleted_*`, `is_*`, `has_*` — membership lemmas.
* created relationships: `RelInv` (`created_rel_types` is exactly the index of
  `created_rels_by_type`, rel ids unique) is kept by `created_relationship`
  (fresh id) and `remove_pending_relationships_for_node`, which removes
  exactly the incident edges (`removeRels_spec`);
  `get_created_relationship_endpoints` answers from the unique entry.
* degrees: `pendingDeg_spec`; `pending_deleted_*degree` returns `none`
  (= `get_relationship_endpoints` panics, graph.rs:3266) as soon as a
  deleted id has no committed endpoints, which is the state after
  `CREATE ()-[r]->() DELETE r` (`deg_panics_on_pending_deleted`, #2769, seen live).
-/
namespace PendingCommit.PS
open PA

def sins (s : List Nat) (x : Nat) : List Nat := if x ∈ s then s else x :: s

theorem mem_sins (s : List Nat) (x y : Nat) : y ∈ sins s x ↔ y = x ∨ y ∈ s := by
  unfold sins; split
  · rename_i h; constructor
    · exact .inr
    · rintro (rfl | h'); exact h; exact h'
  · simp

theorem sins_nodup (s : List Nat) (h : s.Nodup) (x : Nat) : (sins s x).Nodup := by
  unfold sins; split
  · exact h
  · rename_i hx; exact List.nodup_cons.2 ⟨hx, h⟩

/-- `struct Pending` (pending.rs:134). Since #2846 it keeps no id boundary
(`node_entry`/`rel_entry` are gone): the graph holds the batch.
`created` (`created_nodes`, :145) is what the allocator must not reissue —
reserved, not yet recorded by the id space; `taken`
(`taken_relationship_ids`, :145) is its relationship mirror and loses an id
at cancellation (`remove_pending_relationships_for_node`, :712). -/
structure P where
  created : List Nat
  relsByType : FMap (List (Nat × Nat × Nat))
  relTypes : FMap Nat
  taken : List Nat
  deletedN : List Nat
  deletedR : List Nat
  cancelledN : List Nat
  cancelledR : List (Nat × Nat × Nat × Nat)
  newN : FMap (List Entry)
  existN : FMap (List Entry)
  newR : FMap (List Entry)
  existR : FMap (List Entry)
  setL : FMap (List Nat)
  remL : FMap (List Nat)

/-! ## Sets -/

/-- `created_nodes` (:406). -/
def createdNodes (p : P) (ids : List Nat) : P := { p with created := ids.foldl sins p.created }
/-- `deleted_node` (:643), `deleted_relationship` (:876), `deleted_relationships_bulk` (:883). -/
def deletedNode (p : P) (id : Nat) : P := { p with deletedN := sins p.deletedN id }
def deletedRel (p : P) (id : Nat) : P := { p with deletedR := sins p.deletedR id }
def deletedRelsBulk (p : P) (ids : List Nat) : P := { p with deletedR := ids.foldl sins p.deletedR }
def isNodeCreated (p : P) (id : Nat) : Bool := decide (id ∈ p.created)              -- :901
def isNodeDeleted (p : P) (id : Nat) : Bool := decide (id ∈ p.deletedN)             -- :986
def isRelDeleted (p : P) (id : Nat) : Bool := decide (id ∈ p.deletedR)              -- :1000
def deletedNodes (p : P) : List Nat := p.deletedN                                   -- :995
def hasDeletedNodes (p : P) : Bool := !p.deletedN.isEmpty                            -- :971
def hasDeletedRels (p : P) : Bool := !p.deletedR.isEmpty                             -- :976
def hasCreatedRels (p : P) : Bool := !p.relTypes.isEmpty                             -- :981
def isRelCreated (p : P) (id : Nat) : Bool := (fget p.relTypes id).isSome            -- :949
def getRelType (p : P) (id : Nat) : Option Nat := fget p.relTypes id                -- :893

theorem foldl_sins (ids s : List Nat) (y : Nat) : y ∈ ids.foldl sins s ↔ y ∈ s ∨ y ∈ ids := by
  induction ids generalizing s with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons, ih, mem_sins]; simp only [List.mem_cons]
    constructor
    · rintro ((h | h) | h)
      · exact .inr (.inl h)
      · exact .inl h
      · exact .inr (.inr h)
    · rintro (h | h | h)
      · exact .inl (.inr h)
      · exact .inl (.inl h)
      · exact .inr h

theorem sets_spec (p : P) (ids : List Nat) (id y : Nat) :
    (y ∈ (createdNodes p ids).created ↔ y ∈ p.created ∨ y ∈ ids) ∧
    (y ∈ (deletedNode p id).deletedN ↔ y = id ∨ y ∈ p.deletedN) ∧
    (y ∈ (deletedRel p id).deletedR ↔ y = id ∨ y ∈ p.deletedR) ∧
    (y ∈ (deletedRelsBulk p ids).deletedR ↔ y ∈ p.deletedR ∨ y ∈ ids) ∧
    (isNodeCreated p y = true ↔ y ∈ p.created) ∧ (isNodeDeleted p y = true ↔ y ∈ p.deletedN) ∧
    (isRelDeleted p y = true ↔ y ∈ p.deletedR) ∧ deletedNodes p = p.deletedN ∧
    (hasDeletedNodes p = true ↔ p.deletedN ≠ []) ∧ (hasDeletedRels p = true ↔ p.deletedR ≠ []) ∧
    (hasCreatedRels p = true ↔ p.relTypes ≠ []) ∧
    (isRelCreated p y = true ↔ (getRelType p y).isSome) := by
  refine ⟨foldl_sins _ _ _, mem_sins _ _ _, mem_sins _ _ _, foldl_sins _ _ _, by simp [isNodeCreated],
    by simp [isNodeDeleted], by simp [isRelDeleted], rfl, ?_, ?_, ?_, by simp [isRelCreated, getRelType]⟩
  · simp [hasDeletedNodes, List.isEmpty_iff]
  · simp [hasDeletedRels, List.isEmpty_iff]
  · simp [hasCreatedRels, List.isEmpty_iff]

/-! ## Attribute staging, generic over the (new, existing) pair -/

/-- `set_*_attribute` (:444, :813): validate, pick the map, upsert. `none` = `Err`. -/
def stageOne (valid : Val → Bool) (isNew : Bool) (nw ex : FMap (List Entry)) (id k : Nat) (v : Val) :
    Option (FMap (List Entry) × FMap (List Entry)) :=
  if !valid v then none
  else if isNew then some (fset nw id (PA.rustUpsert ((fget nw id).getD []) k v), ex)
  else some (nw, fset ex id (PA.rustUpsert ((fget ex id).getD []) k v))

/-- `set_*_attributes` (:417, :787): validate all, skip empty, replace the list. -/
def stageAll (valid : Val → Bool) (isNew : Bool) (nw ex : FMap (List Entry)) (id : Nat) (attrs : List Entry) :
    Option (FMap (List Entry) × FMap (List Entry)) :=
  if !(attrs.all (fun e => valid e.2)) then none
  else if attrs = [] then some (nw, ex)
  else if isNew then some (fset nw id attrs, ex) else some (nw, fset ex id attrs)

/-- `clear_node_attributes` (:464). -/
def clearA (nw ex : FMap (List Entry)) (id : Nat) : FMap (List Entry) × FMap (List Entry) := (frem nw id, frem ex id)

/-- `has_node_attrs` / `has_relationship_attrs` (:474, :480). -/
def hasA (nw ex : FMap (List Entry)) : Bool := !(nw.isEmpty && ex.isEmpty)

theorem hasA_spec (nw ex : FMap (List Entry)) : hasA nw ex = true ↔ nw ≠ [] ∨ ex ≠ [] := by
  cases nw <;> cases ex <;> simp [hasA]

/-- `get_*_attribute` (:485, :834). -/
def getA (nw ex : FMap (List Entry)) (id k : Nat) : Option Val :=
  if nw.isEmpty && ex.isEmpty then none
  else match (fget nw id).bind (fun a => PA.lookupSorted a k) with
    | some v => some v
    | none => (fget ex id).bind (fun a => PA.lookupSorted a k)

def SortedMap (m : FMap (List Entry)) : Prop := ∀ id a, fget m id = some a → Sorted a

/-- **`setAttr_spec`**. -/
theorem setAttr_spec (valid : Val → Bool) (isNew : Bool) (nw ex : FMap (List Entry)) (hn : SortedMap nw)
    (he : SortedMap ex) (id k : Nat) (v : Val) :
    (valid v = false → stageOne valid isNew nw ex id k v = none) ∧
    (valid v = true → ∃ nw' ex', stageOne valid isNew nw ex id k v = some (nw', ex') ∧
      SortedMap nw' ∧ SortedMap ex' ∧
      (if isNew then fget nw' id = some (upsert ((fget nw id).getD []) k v) ∧ ex' = ex
       else fget ex' id = some (upsert ((fget ex id).getD []) k v) ∧ nw' = nw) ∧
      ∀ j, j ≠ id → fget nw' j = fget nw j ∧ fget ex' j = fget ex j) := by
  have sd : ∀ (m : FMap (List Entry)), SortedMap m → Sorted ((fget m id).getD []) := by
    intro m hm; cases e : fget m id with
    | none => simp [Sorted]
    | some a => exact hm id a e
  have hset : ∀ (m : FMap (List Entry)), SortedMap m →
      SortedMap (fset m id (upsert ((fget m id).getD []) k v)) := by
    intro m hm j a e
    simp only [fget_fset] at e
    split at e
    · cases e; exact upsert_sorted _ _ _ (sd m hm)
    · exact hm j a e
  refine ⟨fun h => by simp [stageOne, h], fun h => ?_⟩
  cases isNew
  · refine ⟨nw, fset ex id (upsert ((fget ex id).getD []) k v), ?_, hn, hset ex he, by simp,
      fun j hj => ⟨rfl, by simp [hj]⟩⟩
    simp [stageOne, h, PA.rustUpsert_eq _ (sd ex he)]
  · refine ⟨fset nw id (upsert ((fget nw id).getD []) k v), ex, ?_, hset nw hn, he, by simp,
      fun j hj => ⟨by simp [hj], rfl⟩⟩
    simp [stageOne, h, PA.rustUpsert_eq _ (sd nw hn)]

/-- **`setAttrs_spec`**: any invalid value is an error with nothing staged; an
empty list stages nothing; otherwise the entity's list is replaced. -/
theorem setAttrs_spec (valid : Val → Bool) (isNew : Bool) (nw ex : FMap (List Entry)) (id : Nat) (attrs : List Entry) :
    stageAll valid isNew nw ex id attrs =
      if attrs.all (fun e => valid e.2) then
        (if attrs = [] then some (nw, ex) else if isNew then some (fset nw id attrs, ex) else some (nw, fset ex id attrs))
      else none := by
  unfold stageAll; by_cases h : attrs.all (fun e => valid e.2) <;> simp [h]

theorem clearA_spec (nw ex : FMap (List Entry)) (id j : Nat) :
    fget (clearA nw ex id).1 j = (if j = id then none else fget nw j) ∧
    fget (clearA nw ex id).2 j = (if j = id then none else fget ex j) := by simp [clearA]

/-- **`getA_spec`**: the staged value from the new-entity map, else from the
existing-entity map (`Some(Null)` for a staged removal). -/
theorem getA_spec (nw ex : FMap (List Entry)) (hn : SortedMap nw) (he : SortedMap ex) (id k : Nat) :
    getA nw ex id k = match (fget nw id).bind (fun a => get a k) with
      | some v => some v
      | none => (fget ex id).bind (fun a => get a k) := by
  have l1 : (fget nw id).bind (fun a => PA.lookupSorted a k) = (fget nw id).bind (fun a => get a k) := by
    cases e : fget nw id with
    | none => rfl
    | some a => exact PA.lookupSorted_eq a (hn id a e) k
  have l2 : (fget ex id).bind (fun a => PA.lookupSorted a k) = (fget ex id).bind (fun a => get a k) := by
    cases e : fget ex id with
    | none => rfl
    | some a => exact PA.lookupSorted_eq a (he id a e) k
  unfold getA; rw [l1, l2]
  split
  · rename_i h
    simp only [Bool.and_eq_true, List.isEmpty_iff] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [fget]
  · rfl

/-! ## `update_*_attrs`: overlay onto an `OrderMap<name, Value>` -/

abbrev OM := List (String × Val)
def oget : OM → String → Option Val
  | [], _ => none
  | (a, b) :: m, s => if a = s then some b else oget m s
def oins (m : OM) (s : String) (v : Val) : OM :=
  if (oget m s).isSome then m.map (fun p => if p.1 = s then (s, v) else p) else m ++ [(s, v)]
def orem (m : OM) (s : String) : OM := m.filter (·.1 != s)

theorem oget_map (m : OM) (s t : String) (v : Val) :
    oget (m.map (fun p => if p.1 = s then (s, v) else p)) t =
      if t = s then (if (oget m s).isSome then some v else none) else oget m t := by
  induction m with
  | nil => simp [oget]
  | cons p m ih =>
    obtain ⟨a, b⟩ := p
    by_cases ha : a = s
    · subst ha; by_cases ht : a = t
      · subst ht; simp [oget]
      · simp [oget, ht, ih, Ne.symm ht]
    · simp only [List.map_cons, ha, ite_false, oget, ih]
      by_cases ht : a = t
      · subst ht; simp [ha]
      · simp [ht]

theorem oget_append (m : OM) (s t : String) (v : Val) :
    oget (m ++ [(s, v)]) t = match oget m t with | some x => some x | none => if s = t then some v else none := by
  induction m with
  | nil => simp [oget]
  | cons p m ih => obtain ⟨a, b⟩ := p; simp only [List.cons_append, oget]; split <;> simp_all

theorem oget_oins (m : OM) (s t : String) (v : Val) : oget (oins m s v) t = if t = s then some v else oget m t := by
  unfold oins
  split
  · rename_i h; rw [oget_map]; simp [h]
  · rename_i h
    rw [oget_append]
    by_cases ht : t = s
    · subst ht; simp at h; simp [h]
    · cases oget m t <;> simp [Ne.symm ht, ht]

theorem oget_orem (m : OM) (s t : String) : oget (orem m s) t = if t = s then none else oget m t := by
  induction m with
  | nil => simp [orem, oget]
  | cons p m ih =>
    obtain ⟨a, b⟩ := p
    simp only [orem] at ih ⊢
    by_cases ha : a = s
    · subst ha; simp only [List.filter_cons, bne_self_eq_false, Bool.false_eq_true, ite_false, ih, oget]
      by_cases ht : t = a
      · simp [ht]
      · simp [ht, Ne.symm ht]
    · simp only [List.filter_cons, bne_iff_ne, ne_eq, ha, not_false_eq_true, ite_true, oget, ih]
      by_cases ht : a = t
      · subst ht; simp [ha, Ne.symm ha]
      · simp [ht]

/-- One staged entry applied (`update_node_attrs` loop body, :511-519). -/
def updStep (nm : Nat → Option String) (o : OM) (e : Entry) : OM :=
  match nm e.1 with
  | none => o
  | some s => if e.2.isNull then orem o s else oins o s e.2

/-- `update_node_attrs` / `update_relationship_attrs` (:503, :852). -/
def updA (nm : Nat → Option String) (nw ex : FMap (List Entry)) (id : Nat) (o : OM) : OM :=
  match fget nw id with
  | some a => a.foldl (updStep nm) o
  | none => match fget ex id with
    | some a => a.foldl (updStep nm) o
    | none => o

/-- Reference: the last staged write naming `s` wins (`Null` = absent). -/
def lastFor (nm : Nat → Option String) (s : String) (r : Option Val) (es : List Entry) : Option Val :=
  es.foldl (fun r e => if nm e.1 = some s then (if e.2.isNull then none else some e.2) else r) r

theorem fold_upd (nm : Nat → Option String) (s : String) : ∀ (es : List Entry) (o : OM),
    oget (es.foldl (updStep nm) o) s = lastFor nm s (oget o s) es
  | [], _ => rfl
  | e :: es, o => by
    simp only [List.foldl_cons, lastFor] at *
    rw [fold_upd nm s es]
    unfold lastFor; congr 1
    unfold updStep
    cases h : nm e.1 with
    | none => simp
    | some t =>
      by_cases hn : e.2.isNull
      · simp only [hn, ite_true, oget_orem]
        by_cases hs : s = t
        · subst hs; simp
        · simp [hs, Ne.symm hs]
      · simp only [hn, Bool.false_eq_true, ite_false, oget_oins]
        by_cases hs : s = t
        · subst hs; simp
        · simp [hs, Ne.symm hs]

/-- **`updA_spec`**: the overlay reads, per name, the last staged write for it. -/
theorem updA_spec (nm : Nat → Option String) (nw ex : FMap (List Entry)) (id : Nat) (o : OM) (s : String) :
    oget (updA nm nw ex id o) s =
      lastFor nm s (oget o s) ((match fget nw id with | some a => some a | none => fget ex id).getD []) := by
  unfold updA
  cases fget nw id with
  | some a => exact fold_upd nm s a o
  | none => cases fget ex id with
    | some a => exact fold_upd nm s a o
    | none => rfl

end PendingCommit.PS
