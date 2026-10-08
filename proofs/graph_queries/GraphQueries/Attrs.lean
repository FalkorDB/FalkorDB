import GraphQueries.Schema
/-
# Attribute-name dictionary and attribute access

| here | there |
| --- | --- |
| `nm : List String` | `AttrNameMap` (attribute_store.rs:139); `index` kept equal to `idx vec` by its only writers `insert`/`get_or_create` (:189,:224) |
| `MAXA` | `MAX_ATTRIBUTES = u16::MAX` (attribute_store.rs:132) |
| `getOrCreate` | `AttrNameMap::get_or_create` — `get_or_create_node_attr_id` / `get_or_create_rel_attr_id` (graph.rs:1836,1740) |
| `nmInsert` | `AttrNameMap::insert` — `add_rel_attribute_name` (:4789); `addNodeAttrName` = `add_node_attribute_name` (:4777) |
| `u16c` | `as u16` |
| `getAttrIdU16` | `get_node_attr_id` / `get_rel_attr_id` (:1845,:1863) |
| `attrName` | `node_attr_name` / `rel_attr_name` (:1854,:1872) |
| `Store`, `byIdx` | `AttributeStore::get_attr_by_idx` (entity → [(attr id, value)]) |
| `attrByName`/`attrNames`/`attrPairs` | :3375 / :3386 / :3397 |
| `batchByIdx` | `get_node_attributes_by_idx` / `get_relationship_attributes_by_idx` (:2495,:3315) |
| `trackOfType` / `track` | `track_edge_index_updates_of_type` / `track_edge_index_updates` (:1927,:1952) |
| `rowsOfLabels` / `importResolvedTrack` | index bookkeeping of `set_nodes_attributes_rows_of_labels` (:1702) / `import_node_attrs_resolved` (:1808) |
-/
namespace GQ
variable {V : Type}

def MAXA : Nat := 65535
def u16c (n : Nat) : Nat := n % 65536

def getOrCreate (nm : List String) (s : String) : List String × Nat :=
  match idx nm s with
  | some i => (nm, i)
  | none => if nm.length ≥ MAXA then (nm, MAXA) else (nm ++ [s], u16c nm.length)

def nmInsert (nm : List String) (s : String) : List String :=
  if s ∈ nm then nm else if nm.length ≥ MAXA then nm else nm ++ [s]

def addNodeAttrName (nm : List String) (s : String) : List String :=
  if (idx nm s).isNone then nmInsert nm s else nm

/-- `add_node_attribute_name` and `add_rel_attribute_name` are the same operation. -/
theorem addNodeAttrName_eq_insert (nm : List String) (s : String) :
    addNodeAttrName nm s = nmInsert nm s := by
  unfold addNodeAttrName nmInsert
  cases h : idx nm s with
  | none => simp
  | some i => simp [List.mem_of_getElem? (idx_get _ _ _ h)]

theorem nmInsert_cap (nm : List String) (s : String) (h : nm.length ≤ MAXA) :
    (nmInsert nm s).length ≤ MAXA := by
  unfold nmInsert; split
  · exact h
  · split
    · exact h
    · simp; omega

theorem getOrCreate_cap (nm : List String) (s : String) (h : nm.length ≤ MAXA) :
    (getOrCreate nm s).1.length ≤ MAXA := by
  unfold getOrCreate; split
  · exact h
  · split
    · exact h
    · simp; unfold MAXA at *; omega

/-- `get_or_create` either returns an id that names `s`, or (table full and
`s` absent) the reserved `ATTRIBUTE_ID_NONE`, which names nothing. Under the
cap the `as u16` never wraps, so no new name can alias attribute 0. -/
theorem getOrCreate_spec (nm : List String) (s : String) (h : nm.length ≤ MAXA) :
    let r := getOrCreate nm s
    (r.1[r.2]? = some s) ∨ (r.2 = MAXA ∧ r.1 = nm ∧ s ∉ nm ∧ nm[MAXA]? = none) := by
  unfold getOrCreate
  cases hi : idx nm s with
  | some i => exact Or.inl (idx_get _ _ _ hi)
  | none =>
    simp only
    split
    · refine Or.inr ⟨rfl, rfl, (idx_none _ _).1 hi, ?_⟩
      simp; omega
    · left
      have : u16c nm.length = nm.length := by unfold u16c MAXA at *; omega
      simp [this]

/-- Without the cap, the 65 537th name would get id 0 and alias attribute 0. -/
example : u16c 65536 = 0 := by decide

def getAttrId (nm : List String) (s : String) : Option Nat := idx nm s
def getAttrIdU16 (nm : List String) (s : String) : Option Nat := (idx nm s).map u16c
def attrName (nm : List String) (i : Nat) : Option String := nm[i]?

theorem getAttrIdU16_eq (nm : List String) (s : String) (h : nm.length ≤ MAXA) :
    getAttrIdU16 nm s = getAttrId nm s := by
  unfold getAttrIdU16 getAttrId
  cases hi : idx nm s with
  | none => rfl
  | some i =>
    have := (List.getElem?_eq_some_iff.1 (idx_get _ _ _ hi)).1
    simp [u16c]; unfold MAXA at h; omega

theorem attrName_getAttrId (nm : List String) (s : String) (i : Nat)
    (h : getAttrId nm s = some i) : attrName nm i = some s := idx_get _ _ _ h

/-! ## Attribute stores -/

abbrev Store (V : Type) := Nat → List (Nat × V)

def byIdx (st : Store V) (key i : Nat) : Option V := (st key).lookup i
def attrByName (nm : List String) (st : Store V) (key : Nat) (s : String) : Option V :=
  (idx nm s).bind (fun i => byIdx st key (u16c i))
def attrNames (nm : List String) (st : Store V) (key : Nat) : List String :=
  (st key).filterMap (fun p => nm[p.1]?)
def attrPairs (nm : List String) (st : Store V) (key : Nat) : List (String × V) :=
  (st key).filterMap (fun p => nm[p.1]?.map (·, p.2))
def attrCount (st : Store V) (key : Nat) : Nat := (st key).length
def batchByIdx (st : Store V) (ids : List Nat) (i : Nat) (dflt : V) (out : List V) : List V :=
  out ++ ids.map (fun k => (byIdx st k i).getD dflt)

theorem attrByName_spec (nm : List String) (st : Store V) (key : Nat) (s : String)
    (h : nm.length ≤ MAXA) :
    attrByName nm st key s = (getAttrId nm s).bind (byIdx st key) := by
  unfold attrByName getAttrId
  cases hi : idx nm s with
  | none => rfl
  | some i =>
    have := (List.getElem?_eq_some_iff.1 (idx_get _ _ _ hi)).1
    have hu : u16c i = i := by unfold u16c MAXA at *; omega
    simp [hu]

theorem fm_len (nm : List String) : ∀ (l : List (Nat × V)), (∀ p ∈ l, p.1 < nm.length) →
    (l.filterMap (fun p => nm[p.1]?)).length = l.length
  | [], _ => rfl
  | p :: l, h => by
    have hp := h p (by simp)
    simp only [List.filterMap_cons, List.getElem?_eq_getElem hp, List.length_cons,
      fm_len nm l (fun q hq => h q (by simp [hq]))]

theorem attrNames_length (nm : List String) (st : Store V) (key : Nat)
    (h : ∀ p ∈ st key, p.1 < nm.length) : (attrNames nm st key).length = attrCount st key :=
  fm_len nm (st key) h

theorem attrPairs_names (nm : List String) (st : Store V) (key : Nat) :
    (attrPairs nm st key).map (·.1) = attrNames nm st key := by
  unfold attrPairs attrNames
  induction st key with
  | nil => rfl
  | cons p l ih =>
    simp only [List.filterMap_cons]
    cases nm[p.1]? <;> simp [ih]

theorem batchByIdx_get (st : Store V) (ids : List Nat) (i : Nat) (dflt : V) (out : List V)
    (k : Nat) (hk : k < ids.length) :
    (batchByIdx st ids i dflt out)[out.length + k]? = some ((byIdx st ids[k] i).getD dflt) := by
  unfold batchByIdx
  rw [List.getElem?_append_right (by omega)]
  simp [hk]

/-! ## Edge/node index-update tracking (membership semantics of `RoaringTreemap`) -/

/-- `track_edge_index_updates_of_type`: pairs `(type, edge)` marked. -/
def trackOfType (hasIdx : Bool) (indexed : String → String → Bool) (types nm : List String)
    (t : Nat) (attrs : List (Nat × List (Nat × V))) (docs : List (Nat × Nat)) : List (Nat × Nat) :=
  if !hasIdx then docs else
  docs ++ attrs.flatMap (fun (id, as) =>
    as.filterMap (fun (a, _) => match nm[a]? with
      | some k => if indexed (types[t]?.getD "") k then some (t, id) else none
      | none => none))

/-- `track_edge_index_updates`: the same, the type looked up per edge. -/
def track (hasIdx : Bool) (indexed : String → String → Bool) (types nm : List String)
    (typeOf : Nat → Nat) (attrs : List (Nat × List (Nat × V))) (docs : List (Nat × Nat)) :
    List (Nat × Nat) :=
  if !hasIdx then docs else
  docs ++ attrs.flatMap (fun (id, as) =>
    as.filterMap (fun (a, _) => match nm[a]? with
      | some k => if indexed (types[typeOf id]?.getD "") k then some (typeOf id, id) else none
      | none => none))

theorem flatMap_congr' {α β} (l : List α) (f g : α → List β) (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.flatMap_cons]
    rw [h a (by simp), ih (fun b hb => h b (by simp [hb]))]

/-- The type-known form (`set_relationships_attributes_of_type`,
`import_relationship_attrs_resolved`) marks exactly what the general form
(`set_relationships_attributes`, `import_relationship_attrs`) marks whenever
every edge really has that type. -/
theorem track_eq_ofType (hasIdx : Bool) (indexed : String → String → Bool) (types nm : List String)
    (typeOf : Nat → Nat) (t : Nat) (attrs : List (Nat × List (Nat × V))) (docs : List (Nat × Nat))
    (h : ∀ p ∈ attrs, typeOf p.1 = t) :
    track hasIdx indexed types nm typeOf attrs docs = trackOfType hasIdx indexed types nm t attrs docs := by
  unfold track trackOfType
  split
  · rfl
  · congr 1
    apply flatMap_congr'
    intro p hp
    obtain ⟨id, as⟩ := p
    have := h _ hp
    simp only at this ⊢
    rw [this]

theorem trackOfType_noindex (indexed : String → String → Bool) (types nm : List String) (t : Nat)
    (attrs : List (Nat × List (Nat × V))) (docs : List (Nat × Nat)) :
    trackOfType false indexed types nm t attrs docs = docs := rfl

/-- Node side, per-label once (`set_nodes_attributes_rows_of_labels`):
every id gets label `l` iff some attr of the record is indexed under `l`;
out-of-range label ids are skipped (`node_labels.get(..) else continue`). -/
def rowsOfLabels (hasIdx : Bool) (indexed : String → String → Bool) (labels nm : List String)
    (ids labelIds attrIds : List Nat) (docs : List (Nat × Nat)) : List (Nat × Nat) :=
  if !hasIdx then docs else
  docs ++ labelIds.flatMap (fun l => match labels[l]? with
    | none => []
    | some lab =>
      if attrIds.any (fun a => match nm[a]? with | some k => indexed lab k | none => false)
      then ids.map (l, ·) else [])

/-- Node side, per node (`import_node_attrs_resolved`; `node_labels[l]`
panics out of range — `none` here). -/
def importResolvedTrack (hasIdx : Bool) (indexed : String → String → Bool) (labels nm : List String)
    (rows : List (Nat × List Nat)) (labelIds : List Nat) (docs : List (Nat × Nat)) :
    List (Nat × Nat) :=
  if !hasIdx then docs else
  docs ++ rows.flatMap (fun (id, as) => labelIds.flatMap (fun l => as.filterMap (fun a =>
    match labels[l]?, nm[a]? with
    | some lab, some k => if indexed lab k then some (l, id) else none
    | _, _ => none)))

/-- The once-per-label decision marks the same `(label, node)` pairs as the
per-node walk when every row carries the record's attribute list. -/
theorem rowsOfLabels_mem (hasIdx : Bool) (indexed : String → String → Bool) (labels nm : List String)
    (ids labelIds attrIds : List Nat) (docs : List (Nat × Nat)) (l n : Nat) :
    (l, n) ∈ rowsOfLabels hasIdx indexed labels nm ids labelIds attrIds docs ↔
    (l, n) ∈ docs ∨ (hasIdx = true ∧ n ∈ ids ∧ l ∈ labelIds ∧ ∃ lab, labels[l]? = some lab ∧
      ∃ a ∈ attrIds, ∃ k, nm[a]? = some k ∧ indexed lab k = true) := by
  unfold rowsOfLabels
  cases hasIdx with
  | false => simp
  | true =>
    simp only [Bool.not_true, Bool.false_eq_true, ite_false, List.mem_append, List.mem_flatMap,
      true_and]
    apply or_congr Iff.rfl
    constructor
    · rintro ⟨l', hl', hm⟩
      cases hlab : labels[l']? with
      | none => simp [hlab] at hm
      | some lab =>
        simp only [hlab] at hm
        split at hm
        · rename_i hany
          simp only [List.mem_map] at hm
          obtain ⟨n', hn', he⟩ := hm
          cases he
          refine ⟨hn', hl', lab, hlab, ?_⟩
          obtain ⟨a, ha, hk⟩ := List.any_eq_true.1 hany
          cases hka : nm[a]? with
          | none => simp [hka] at hk
          | some k => exact ⟨a, ha, k, hka, by simpa [hka] using hk⟩
        · simp at hm
    · rintro ⟨hn, hl, lab, hlab, a, ha, k, hka, hk⟩
      refine ⟨l, hl, ?_⟩
      simp only [hlab]
      rw [ite_cond_eq_true _ _ (eq_true (List.any_eq_true.2 ⟨a, ha, by simp [hka, hk]⟩))]
      exact List.mem_map.2 ⟨n, hn, rfl⟩

end GQ
