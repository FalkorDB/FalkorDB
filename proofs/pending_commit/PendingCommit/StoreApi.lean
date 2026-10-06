import PendingCommit.StoreFold
/-
# `AttributeStore` API (attribute_store.rs:1150-1400)

The store is a `DataBlock` (`D`). Every read below is stated in terms of
`dread` (the entity's span) and every write in terms of the
`Store`/`StoreFold` theorems.

| Lean | Rust |
| --- | --- |
| `emptyD`, `new_spec` | `AttributeStore::new`/`default`, `DataBlock::default`, `DirPage::default` (:1152, :914) |
| `newVersion` | `new_version` (:1159), a value copy (`Arc` clone + `make_mut` on write) |
| `getAttr`, `getAttr_eq` | `get_attr_by_idx` (:1171) |
| `getBatch` | `get_attrs_by_idx_batch_into` (:1182) |
| `hasAttrs`, `attrCount`, `attrIds`, `allAttrs` | :1197, :1206, :1226, :1236 |
| `entityMem` | `entity_memory_usage` (:1215) |
| `removeAll` | `remove_all` (:1246) |
| `insertAttrs`, `insertRows` | `insert_attrs` (:1315), `insert_attrs_rows` (:1266) |
| `importAttrs`, `importResolved` | `import_attrs` (:1351), `import_attrs_resolved` (:1379) |
-/
namespace PendingCommit.Content
open Arena (Slot)

def emptyD : D := ⟨[]⟩

theorem new_spec : SInv emptyD ∧ ∀ id, dread emptyD id = [] := by
  refine ⟨⟨fun _ h => by simp [emptyD] at h, fun bi b e => by simp [blockAt, emptyD] at e⟩, fun id => ?_⟩
  simp [dread, blockOf, blockAt, emptyD, span, emptyB, slotAt, spanOf, Slot.empty]

def newVersion (d : D) : D := d
theorem newVersion_eq (d : D) : newVersion d = d := rfl

/-- Every span strictly sorted and null-free. -/
def SOK (d : D) : Prop := ∀ id, Sorted (dread d id) ∧ NoNull (dread d id)

def getAttr (d : D) (key k : Nat) : Option Val := (dget d key).bind (fun es => spanGet es k)

/-- **`getAttr_eq`**: `get_attr_by_idx` is first-match lookup in the span. -/
theorem getAttr_eq (d : D) (h : SInv d) (hs : SOK d) (key k : Nat) : getAttr d key k = get (dread d key) k := by
  unfold getAttr; rw [dget_eq d h]
  split
  · rename_i he; rw [he]; rfl
  · simp only [Option.bind_some]; exact spanGet_eq _ (hs key).1 k

def getBatch (d : D) (keys : List Nat) (k : Nat) (dflt : Val) (out : List Val) : List Val :=
  out ++ keys.map (fun key => (getAttr d key k).getD dflt)

theorem getBatch_spec (d : D) (h : SInv d) (hs : SOK d) (keys : List Nat) (k : Nat) (dflt : Val) (out : List Val) :
    getBatch d keys k dflt out = out ++ keys.map (fun key => (get (dread d key) k).getD dflt) := by
  simp only [getBatch, getAttr_eq d h hs]

def hasAttrs (d : D) (key : Nat) : Bool := (dget d key).isSome
def attrCount (d : D) (key : Nat) : Nat := ((dget d key).map List.length).getD 0
def attrIds (d : D) (key : Nat) : List Nat := ((dget d key).getD []).map (·.1)
def allAttrs (d : D) (key : Nat) : List Entry := (dget d key).getD []
/-- `entity_memory_usage`: `SpanRef::heap_bytes` (`hb`, `Pack.heapBytes_eq`) or 0. -/
def entityMem (hb : List Entry → Nat) (d : D) (key : Nat) : Nat := ((dget d key).map hb).getD 0

theorem reads_spec (d : D) (h : SInv d) (key : Nat) (hb : List Entry → Nat) :
    (hasAttrs d key = true ↔ dread d key ≠ []) ∧ attrCount d key = (dread d key).length ∧
      attrIds d key = (dread d key).map (·.1) ∧ allAttrs d key = dread d key ∧
      entityMem hb d key = (if dread d key = [] then 0 else hb (dread d key)) := by
  simp only [hasAttrs, attrCount, attrIds, allAttrs, entityMem, dget_eq d h]
  by_cases he : dread d key = []
  · simp [he]
  · simp [he]

/-! ## Batched writes -/

def removeAll (d : D) (keys : List Nat) : D := (runOps (fun d id (_ : Unit) => (dremove d id, 0, 0)) d (keys.map (·, ()))).1

theorem lk_unit (keys : List Nat) (id : Nat) : lk (keys.map (·, ())) id = if id ∈ keys then some () else none := by
  induction keys with
  | nil => rfl
  | cons k ks ih => simp only [List.map_cons, lk, ih, List.mem_cons]; by_cases h : id = k <;> simp [h]

/-- **`removeAll_spec`**: every key reads empty, everything else is unchanged. -/
theorem removeAll_spec (d : D) (h : SInv d) (keys : List Nat) (hk : keys.Nodup) :
    SInv (removeAll d keys) ∧ ∀ id, dread (removeAll d keys) id = if id ∈ keys then [] else dread d id := by
  have e : (keys.map (·, ())).map (·.1) = keys := by induction keys <;> simp_all
  have hnd : ((keys.map (·, ())).map (·.1)).Nodup := by rw [e]; exact hk
  obtain ⟨r1, r2, -⟩ := runOps_spec _ _ _ _ remove_spec _ d h hnd (fun _ _ => trivial)
  refine ⟨r1, fun id => ?_⟩
  simp only [removeAll]; rw [r2 id, lk_unit]
  by_cases hi : id ∈ keys <;> simp [hi, after]

/-- `insert_attrs`: skip empty update lists, visit in id order, merge, sum. -/
def insertAttrs (d : D) (items : List (Nat × List Entry)) : D × Nat × Nat :=
  runOps (fun d id ps => dmerge d id ps) d items

/-- **`insertAttrs_spec`**: for any visiting order of distinct ids (Rust sorts
by id), each entity reads its own merge and the counts are the per-entity
`Span.mergeSpan` counts summed — which `Span.mergeSpan_correct` identifies
with `insert_attrs`' documented `(nremoved, nset)`. -/
theorem insertAttrs_spec (d : D) (h : SInv d) (items : List (Nat × List Entry))
    (hnd : (items.map (·.1)).Nodup) (hfit : ∀ p ∈ items, (mergeSpan (dread d p.1) p.2).1.length ≤ 65535) :
    let r := insertAttrs d items
    SInv r.1 ∧ (∀ id, dread r.1 id = after (fun old ps => (mergeSpan old ps).1) (dread d id) (lk items id)) ∧
      r.2.1 = (items.map (fun p => (mergeSpan (dread d p.1) p.2).2.1)).sum ∧
      r.2.2 = (items.map (fun p => (mergeSpan (dread d p.1) p.2).2.2)).sum := by
  unfold insertAttrs
  obtain ⟨a, b, c, e⟩ := runOps_spec _ _ _ _ merge_spec items d h hnd hfit
  exact ⟨a, b, c, e⟩

/-- `insert_attrs_rows`: entity `ids[r]` gets `zip attr_ids rows[r]`
(`RowUpdates`, nulls passed through as removals). -/
def insertRows (d : D) (ids : List Nat) (attrIds : List Nat) (rows : List (List Val)) : D × Nat × Nat :=
  insertAttrs d (ids.zip (rows.map (attrIds.zip ·)))

/-- `AttrUpdates` (:363-421): the slice impl reads `self[i]`; `RowUpdates`
reads `ids[i]` / `values[i]` from two parallel slices. Both present the merge
with the same indexed list, which for rows is `zip ids values`. -/
def sliceLen (l : List Entry) : Nat := l.length
def sliceId (l : List Entry) (i : Nat) : Option Nat := (l[i]?).map (·.1)
def sliceValue (l : List Entry) (i : Nat) : Option Val := (l[i]?).map (·.2)
def rowLen (ids : List Nat) (_vals : List Val) : Nat := ids.length
def rowId (ids : List Nat) (_vals : List Val) (i : Nat) : Option Nat := ids[i]?
def rowValue (_ids : List Nat) (vals : List Val) (i : Nat) : Option Val := vals[i]?

theorem attrUpdates_view (ids : List Nat) (vals : List Val) (h : vals.length = ids.length) (i : Nat) :
    rowLen ids vals = sliceLen (ids.zip vals) ∧ rowId ids vals i = sliceId (ids.zip vals) i ∧
      rowValue ids vals i = sliceValue (ids.zip vals) i := by
  have hl : (ids.zip vals).length = ids.length := by simp [h]
  refine ⟨by simp [rowLen, sliceLen, h], ?_, ?_⟩
  · simp only [rowId, sliceId]
    by_cases hi : i < ids.length
    · have h1 : (ids.zip vals)[i]? = some (ids.zip vals)[i] := List.getElem?_eq_getElem (by omega)
      have h2 : ids[i]? = some ids[i] := List.getElem?_eq_getElem hi
      rw [h1, h2]; simp
    · have h1 : (ids.zip vals)[i]? = none := List.getElem?_eq_none (by omega)
      rw [h1, List.getElem?_eq_none (by omega)]; rfl
  · simp only [rowValue, sliceValue]
    by_cases hi : i < ids.length
    · have h1 : (ids.zip vals)[i]? = some (ids.zip vals)[i] := List.getElem?_eq_getElem (by omega)
      have h2 : vals[i]? = some vals[i] := List.getElem?_eq_getElem (by omega)
      rw [h1, h2]; simp
    · have h1 : (ids.zip vals)[i]? = none := List.getElem?_eq_none (by omega)
      rw [h1, List.getElem?_eq_none (by omega)]; rfl

/-- `import_attrs`: drop nulls, skip empty, `set_span`, count what was stored. -/
def importAttrs (d : D) (m : List (Nat × List Entry)) : D × Nat :=
  let items := (m.map (fun p => (p.1, p.2.filter (fun e => !e.2.isNull)))).filter (fun p => !p.2.isEmpty)
  let r := runOps (fun d id ps => (dsetSpan d id ps, 0, ps.length)) d items
  (r.1, r.2.2)

theorem lk_filter_map (m : List (Nat × List Entry)) (f : List Entry → List Entry) (id : Nat)
    (hnd : (m.map (·.1)).Nodup) :
    lk ((m.map (fun p => (p.1, f p.2))).filter (fun p => !p.2.isEmpty)) id =
      match lk m id with | some ps => (if f ps = [] then none else some (f ps)) | none => none := by
  induction m with
  | nil => rfl
  | cons p m ih =>
    obtain ⟨k, a⟩ := p
    simp only [List.map_cons, List.nodup_cons] at hnd
    have ih := ih hnd.2
    simp only [List.map_cons, List.filter_cons, lk]
    by_cases hid : id = k
    · subst hid
      have hn : lk m id = none := lk_none m id hnd.1
      rw [hn] at ih
      by_cases hf : f a = []
      · simp [hf, ih]
      · simp [hf, lk]
    · by_cases hf : f a = []
      · simp [hf, ih, hid]
      · simp [hf, lk, hid, ih]

theorem nodup_filter_map (m : List (Nat × List Entry)) (f : List Entry → List Entry)
    (hnd : (m.map (·.1)).Nodup) :
    (((m.map (fun p => (p.1, f p.2))).filter (fun p => !p.2.isEmpty)).map (·.1)).Nodup := by
  have hs : (((m.map (fun p => (p.1, f p.2))).filter (fun p => !p.2.isEmpty)).map (·.1)).Sublist (m.map (·.1)) := by
    have := (List.filter_sublist (p := fun p : Nat × List Entry => !p.2.isEmpty)
      (l := m.map (fun p => (p.1, f p.2)))).map (·.1)
    simpa [List.map_map, Function.comp_def] using this
  exact hnd.sublist hs

/-- **`importAttrs_spec`**: each new entity reads its non-null attributes
(nothing if all were null) and `nset` counts exactly those. -/
theorem importAttrs_spec (d : D) (h : SInv d) (m : List (Nat × List Entry)) (hnd : (m.map (·.1)).Nodup)
    (hfit : ∀ p ∈ m, p.2.length ≤ 65535) :
    let r := importAttrs d m
    SInv r.1 ∧ ∀ id, dread r.1 id = match lk m id with
      | some ps => (if ps.filter (fun e => !e.2.isNull) = [] then dread d id else ps.filter (fun e => !e.2.isNull))
      | none => dread d id := by
  have hnd' := nodup_filter_map m (fun ps => ps.filter (fun e => !e.2.isNull)) hnd
  obtain ⟨r1, r2, -⟩ := runOps_spec _ _ _ _ set_spec _ d h hnd' (by
    intro p hp
    simp only [List.mem_filter, List.mem_map] at hp
    obtain ⟨⟨q, hq, rfl⟩, -⟩ := hp
    exact Nat.le_trans (List.length_filter_le _ _) (hfit q hq))
  refine ⟨r1, fun id => ?_⟩
  simp only [importAttrs]; rw [r2 id, lk_filter_map m _ id hnd]
  cases lk m id with
  | none => simp [after]
  | some ps => by_cases hf : ps.filter (fun e => !e.2.isNull) = [] <;> simp [hf, after]

/-- `import_attrs_resolved` (:1379): no null filter, stable sort by id, `set_span`. -/
def sortById (es : List Entry) : List Entry := es.mergeSort (fun a b => a.1 ≤ b.1)

def importResolved (d : D) (data : List (Nat × List Entry)) : D × Nat :=
  let items := (data.map (fun p => (p.1, sortById p.2))).filter (fun p => !p.2.isEmpty)
  let r := runOps (fun d id ps => (dsetSpan d id ps, 0, ps.length)) d items
  (r.1, r.2.2)

/-- **`importResolved_spec`**: each entity reads its entries sorted by id —
duplicates and nulls included, since neither is filtered. -/
theorem importResolved_spec (d : D) (h : SInv d) (m : List (Nat × List Entry)) (hnd : (m.map (·.1)).Nodup)
    (hfit : ∀ p ∈ m, p.2.length ≤ 65535) :
    let r := importResolved d m
    SInv r.1 ∧ ∀ id, dread r.1 id = match lk m id with
      | some ps => (if ps = [] then dread d id else sortById ps)
      | none => dread d id := by
  have hnd' := nodup_filter_map m sortById hnd
  have hlen : ∀ ps, (sortById ps).length = ps.length := fun ps => (List.mergeSort_perm _ _).length_eq
  obtain ⟨r1, r2, -⟩ := runOps_spec _ _ _ _ set_spec _ d h hnd' (by
    intro p hp
    simp only [List.mem_filter, List.mem_map] at hp
    obtain ⟨⟨q, hq, rfl⟩, -⟩ := hp
    simp only [hlen]; exact hfit q hq)
  refine ⟨r1, fun id => ?_⟩
  simp only [importResolved]; rw [r2 id, lk_filter_map m _ id hnd]
  cases lk m id with
  | none => simp [after]
  | some ps =>
    have : sortById ps = [] ↔ ps = [] := by
      constructor
      · intro e; have := hlen ps; rw [e] at this; exact List.eq_nil_of_length_eq_zero this.symm
      · intro e; subst e; exact List.eq_nil_of_length_eq_zero (hlen [])
    by_cases hf : ps = []
    · subst hf; have h0 := this.2 rfl; simp [h0, after]
    · simp [hf, this, after]

/-- The duplicate-id case (`GRAPH.BULK` with a repeated header column, or a
malformed RDB): both entries are stored, the binary search answers with the
second, and `attr_count` says 2. -/
theorem resolved_duplicate :
    let d := (importResolved emptyD [(0, [(0, .inl 1), (0, .inl 2)])]).1
    dread d 0 = [(0, .inl 1), (0, .inl 2)] ∧ spanGet (dread d 0) 0 = some (.inl 2) := by
  have hs := (importResolved_spec emptyD new_spec.1 [(0, [(0, .inl 1), (0, .inl 2)])] (by simp)
    (by simp)).2 0
  simp only [lk, ite_true] at hs
  have e : dread emptyD 0 = [] := new_spec.2 0
  have hsort : sortById [(0, .inl 1), (0, .inl 2)] = [(0, .inl 1), (0, .inl 2)] :=
    List.mergeSort_of_pairwise (by simp)
  rw [hsort] at hs
  simp at hs
  refine ⟨hs, ?_⟩
  rw [hs]
  have hl : BinSearch.loop [0, 0] 0 0 2 = 1 := by
    rw [BinSearch.loop]; simp; rw [BinSearch.loop]; simp
  simp [spanGet, BinSearch.bsearch, hl]

end PendingCommit.Content
