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

/-- Stable sort by attribute id (`sort_by_key`, a stable merge sort). -/
def sortById (es : List Entry) : List Entry := es.mergeSort (fun a b => a.1 ≤ b.1)

/-- `import_attrs_resolved` before #3063 (cab3cf192 :1379): no null filter, stable sort
by id, `set_span` — no dedup, no `ATTRIBUTE_ID_NONE` drop. -/
def pre3063_importResolved (d : D) (data : List (Nat × List Entry)) : D × Nat :=
  let items := (data.map (fun p => (p.1, sortById p.2))).filter (fun p => !p.2.isEmpty)
  let r := runOps (fun d id ps => (dsetSpan d id ps, 0, ps.length)) d items
  (r.1, r.2.2)

/-- **`pre3063_importResolved_spec`**: each entity reads its entries sorted by id —
duplicates and nulls included, since neither is filtered. -/
theorem pre3063_importResolved_spec (d : D) (h : SInv d) (m : List (Nat × List Entry)) (hnd : (m.map (·.1)).Nodup)
    (hfit : ∀ p ∈ m, p.2.length ≤ 65535) :
    let r := pre3063_importResolved d m
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
  simp only [pre3063_importResolved]; rw [r2 id, lk_filter_map m _ id hnd]
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

/-- **History (before #3063).** The duplicate-id case (`GRAPH.BULK` with a repeated header column, or a
malformed RDB): both entries are stored, the binary search answers with the
second, and `attr_count` says 2. -/
theorem pre3063_resolved_duplicate :
    let d := (pre3063_importResolved emptyD [(0, [(0, .inl 1), (0, .inl 2)])]).1
    dread d 0 = [(0, .inl 1), (0, .inl 2)] ∧ spanGet (dread d 0) 0 = some (.inl 2) := by
  have hs := (pre3063_importResolved_spec emptyD new_spec.1 [(0, [(0, .inl 1), (0, .inl 2)])] (by simp)
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


/-! ## `import_attrs_resolved` since #3063 (0f45a8685, :1388)

`sort_by_key` (stable), `dedup_by_key` (of each run of one id keep the first), pop a
trailing `ATTRIBUTE_ID_NONE` (= 65535), skip if empty, `set_span`. -/

/-- `Vec::dedup_by_key(|(idx, _)| *idx)`: of each run of equal ids keep the first. -/
def dedupId : List Entry → List Entry
  | a :: b :: l => if a.1 = b.1 then dedupId (a :: l) else a :: dedupId (b :: l)
  | l => l
termination_by l => l.length

/-- `if entries.last().is_some_and(|(idx, _)| *idx == ATTRIBUTE_ID_NONE) { entries.pop(); }` -/
def dropNone (l : List Entry) : List Entry :=
  match l.getLast? with
  | some e => if e.1 = 65535 then l.dropLast else l
  | none => l

def normResolved (es : List Entry) : List Entry := dropNone (dedupId (sortById es))

/-- `import_attrs_resolved` (:1388). -/
def importResolved (d : D) (data : List (Nat × List Entry)) : D × Nat :=
  let items := (data.map (fun p => (p.1, normResolved p.2))).filter (fun p => !p.2.isEmpty)
  let r := runOps (fun d id ps => (dsetSpan d id ps, 0, ps.length)) d items
  (r.1, r.2.2)

theorem lk_mem {α : Type} : ∀ (m : List (Nat × α)) (id : Nat) (a : α), lk m id = some a → (id, a) ∈ m
  | [], _, _, h => by simp [lk] at h
  | (k, b) :: m, id, a, h => by
    simp only [lk] at h
    split at h
    · rename_i hk; cases h; subst hk; simp
    · exact List.mem_cons_of_mem _ (lk_mem m id a h)

theorem get_eq_head_filter : ∀ (l : List Entry) (k : Nat),
    get l k = ((l.filter (fun e => e.1 == k)).head?).map (·.2)
  | [], _ => rfl
  | e :: es, k => by
    simp only [get, List.filter_cons]
    by_cases h : k = e.1
    · subst h; simp
    · simp only [h, ↓reduceIte]; rw [get_eq_head_filter es k]
      have : (e.1 == k) = false := by simp; exact Ne.symm h
      simp [this]

private def leId (a b : Entry) : Bool := decide (a.1 ≤ b.1)

private theorem leId_trans : ∀ (a b c : Entry), leId a b → leId b c → leId a c := by
  intro a b c h1 h2; simp [leId] at *; omega
private theorem leId_total : ∀ (a b : Entry), leId a b || leId b a := by
  intro a b; simp [leId]; omega

theorem sortById_pairwise (es : List Entry) : (sortById es).Pairwise (fun a b => a.1 ≤ b.1) := by
  have := List.pairwise_mergeSort leId_trans leId_total es
  exact this.imp (fun h => by simpa [leId] using h)

/-- Stability: the entries of one id keep their relative order. -/
theorem sortById_filter (es : List Entry) (k : Nat) :
    (sortById es).filter (fun e => e.1 == k) = es.filter (fun e => e.1 == k) := by
  have hc : (es.filter (fun e => e.1 == k)).Pairwise (fun a b => leId a b = true) := by
    refine List.Pairwise.imp_of_mem ?_ (List.pairwise_of_forall (R := fun _ _ => True) (fun _ _ => trivial))
    intro a b ha hb _
    simp only [List.mem_filter, beq_iff_eq] at ha hb
    simp [leId, ha.2, hb.2]
  have hs : List.Sublist (es.filter (fun e => e.1 == k)) (sortById es) :=
    List.sublist_mergeSort leId_trans leId_total hc List.filter_sublist
  have hs2 := hs.filter (fun e => e.1 == k)
  rw [List.filter_filter] at hs2
  simp only [Bool.and_self] at hs2
  have hp : ((sortById es).filter (fun e => e.1 == k)).Perm (es.filter (fun e => e.1 == k)) :=
    (List.mergeSort_perm _ _).filter _
  exact (hs2.eq_of_length hp.length_eq.symm).symm

theorem get_sortById (es : List Entry) (k : Nat) : get (sortById es) k = get es k := by
  rw [get_eq_head_filter, get_eq_head_filter, sortById_filter]

theorem mem_dedupId : ∀ (l : List Entry) (x : Entry), x ∈ dedupId l → x ∈ l
  | a :: b :: l, x, h => by
    rw [dedupId] at h
    split at h
    · have := mem_dedupId (a :: l) x h
      simp only [List.mem_cons] at this ⊢
      rcases this with h | h
      · exact .inl h
      · exact .inr (.inr h)
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · simp [h]
      · have := mem_dedupId (b :: l) x h
        simp only [List.mem_cons] at this ⊢
        exact .inr this
  | [], _, h => by simp [dedupId] at h
  | [_], _, h => by simpa [dedupId] using h
termination_by l => l.length

theorem get_dedupId : ∀ (l : List Entry) (k : Nat), get (dedupId l) k = get l k
  | a :: b :: l, k => by
    rw [dedupId]
    split
    · rename_i hab
      rw [get_dedupId (a :: l) k]
      simp only [get]
      by_cases hk : k = a.1
      · simp [hk]
      · have hb : ¬ k = b.1 := by rw [← hab]; exact hk
        simp only [hk, hb, ↓reduceIte]
    · simp only [get]; rw [get_dedupId (b :: l) k]; rfl
  | [], _ => by simp [dedupId]
  | [_], _ => by simp [dedupId]
termination_by l => l.length

theorem dedupId_sorted : ∀ (l : List Entry), l.Pairwise (fun a b => a.1 ≤ b.1) → Sorted (dedupId l)
  | a :: b :: l, h => by
    rw [dedupId]
    split
    · exact dedupId_sorted (a :: l) (h.sublist (List.Sublist.cons_cons a (List.sublist_cons_self b l)))
    · rename_i hab
      have ht := dedupId_sorted (b :: l) (List.Pairwise.of_cons h)
      refine List.Pairwise.cons (fun x hx => ?_) ht
      have hx' := mem_dedupId _ _ hx
      have hab' : a.1 ≤ b.1 := List.rel_of_pairwise_cons h (by simp)
      simp only [List.mem_cons] at hx'
      rcases hx' with rfl | hx'
      · omega
      · have := List.rel_of_pairwise_cons (List.Pairwise.of_cons h) hx'; omega
  | [], _ => by simp [dedupId, Sorted]
  | [_], _ => by simp [dedupId, Sorted]
termination_by l => l.length

theorem get_none_of_ne : ∀ (l : List Entry) (k : Nat), (∀ y ∈ l, y.1 ≠ k) → get l k = none
  | [], _, _ => rfl
  | e :: es, k, h => by
    simp only [get]
    have hk : ¬ k = e.1 := fun hk => h e (by simp) hk.symm
    simp only [hk, ↓reduceIte]
    exact get_none_of_ne es k (fun y hy => h y (by simp [hy]))

theorem get_dropLast : ∀ (l : List Entry) (k : Nat), (∀ x, l.getLast? = some x → x.1 ≠ k) →
    get l.dropLast k = get l k
  | [], _, _ => rfl
  | [a], k, h => by
    have := h a (by simp)
    simp [get]; intro hk; exact this hk.symm
  | a :: b :: l, k, h => by
    have h' : ∀ x, (b :: l).getLast? = some x → x.1 ≠ k := fun x hx => h x (by
      rw [List.getLast?_cons_cons]; exact hx)
    rw [List.dropLast_cons_cons]
    simp only [get]
    rw [get_dropLast (b :: l) k h']; rfl

theorem sorted_last_bound (l : List Entry) (hs : Sorted l) (x : Entry) (hx : l.getLast? = some x) :
    l = l.dropLast ++ [x] ∧ ∀ y ∈ l.dropLast, y.1 < x.1 := by
  have hne : l ≠ [] := by intro h; subst h; simp at hx
  have he : l = l.dropLast ++ [x] := by
    have := List.dropLast_concat_getLast hne
    rw [List.getLast?_eq_some_getLast hne] at hx; cases hx; exact this.symm
  refine ⟨he, fun y hy => ?_⟩
  have hs' : Sorted (l.dropLast ++ [x]) := by rw [← he]; exact hs
  unfold Sorted at hs'
  rw [List.pairwise_append] at hs'
  exact hs'.2.2 y hy x (by simp)

/-- Every id `normResolved` keeps is below `ATTRIBUTE_ID_NONE`. -/
theorem dropNone_lt (l : List Entry) (hs : Sorted l) (hb : ∀ y ∈ l, y.1 ≤ 65535) :
    ∀ y ∈ dropNone l, y.1 < 65535 := by
  unfold dropNone
  cases hx : l.getLast? with
  | none =>
    have : l = [] := List.getLast?_eq_none_iff.1 hx
    subst this; simp
  | some x =>
    obtain ⟨he, hlt⟩ := sorted_last_bound l hs x hx
    have hxm : x ∈ l := List.mem_of_getLast? hx
    have hxb := hb x hxm
    simp only
    split
    · rename_i h; intro y hy; have := hlt y hy; omega
    · rename_i h
      intro y hy
      rw [he] at hy
      simp only [List.mem_append, List.mem_singleton] at hy
      rcases hy with hy | rfl
      · have := hlt y hy; omega
      · omega

theorem dropNone_sorted (l : List Entry) (hs : Sorted l) : Sorted (dropNone l) := by
  unfold dropNone
  split
  · split
    · exact hs.sublist (List.dropLast_sublist l)
    · exact hs
  · exact hs

theorem get_dropNone (l : List Entry) (k : Nat) (hk : k ≠ 65535) : get (dropNone l) k = get l k := by
  unfold dropNone
  split
  · rename_i x hx
    split
    · rename_i h
      exact get_dropLast l k (fun y hy => by rw [hx] at hy; cases hy; rw [h]; exact Ne.symm hk)
    · rfl
  · rfl

/-- **`normResolved_sorted`**: the span `set_span` receives is strictly ascending. -/
theorem normResolved_sorted (es : List Entry) : Sorted (normResolved es) :=
  dropNone_sorted _ (dedupId_sorted _ (sortById_pairwise es))

/-- **`normResolved_lt`**: with `u16` ids, every kept id is `< ATTRIBUTE_ID_NONE`. -/
theorem normResolved_lt (es : List Entry) (hb : ∀ e ∈ es, e.1 ≤ 65535) :
    ∀ y ∈ normResolved es, y.1 < 65535 :=
  dropNone_lt _ (dedupId_sorted _ (sortById_pairwise es)) (fun y hy =>
    hb y ((List.mem_mergeSort).1 (mem_dedupId _ _ hy)))

/-- **`normResolved_fits`** (#2539 closed): with `u16` ids the span has at most 65,535
entries, so `set_span`'s `n as u16` is exact (`setSpan_cast_exact`) — however many
columns the bulk header names. -/
theorem normResolved_fits (es : List Entry) (hb : ∀ e ∈ es, e.1 ≤ 65535) :
    (normResolved es).length ≤ 65535 ∧ u16 (normResolved es).length = (normResolved es).length :=
  setSpan_cast_exact _ (normResolved_sorted es) (normResolved_lt es hb)

/-- **`normResolved_get`**: an id other than `ATTRIBUTE_ID_NONE` reads the value of its
*first* column (`get es k` is first-match in header order); `ATTRIBUTE_ID_NONE` columns are
gone. -/
theorem normResolved_get (es : List Entry) (hb : ∀ e ∈ es, e.1 ≤ 65535) (k : Nat) :
    get (normResolved es) k = if k = 65535 then none else get es k := by
  split
  · rename_i hk; subst hk
    exact get_none_of_ne _ _ (fun y hy => Nat.ne_of_lt (normResolved_lt es hb y hy))
  · rename_i hk
    rw [normResolved, get_dropNone _ _ hk, get_dedupId, get_sortById]

/-- **`importResolved_spec`**: each entity reads its normalised entries (nothing new if
none survive); no length precondition any more — `u16` ids suffice. -/
theorem importResolved_spec (d : D) (h : SInv d) (m : List (Nat × List Entry)) (hnd : (m.map (·.1)).Nodup)
    (hb : ∀ p ∈ m, ∀ e ∈ p.2, e.1 ≤ 65535) :
    let r := importResolved d m
    SInv r.1 ∧ ∀ id, dread r.1 id = match lk m id with
      | some ps => (if normResolved ps = [] then dread d id else normResolved ps)
      | none => dread d id := by
  have hnd' := nodup_filter_map m normResolved hnd
  obtain ⟨r1, r2, -⟩ := runOps_spec _ _ _ _ set_spec _ d h hnd' (by
    intro p hp
    simp only [List.mem_filter, List.mem_map] at hp
    obtain ⟨⟨q, hq, rfl⟩, -⟩ := hp
    exact (normResolved_fits q.2 (hb q hq)).1)
  refine ⟨r1, fun id => ?_⟩
  simp only [importResolved]; rw [r2 id, lk_filter_map m _ id hnd]
  cases lk m id with
  | none => simp [after]
  | some ps => by_cases hf : normResolved ps = [] <;> simp [hf, after]

/-- **`importResolved_reads`**: after the import, a bulk entity answers every lookup
(`spanGet`, the binary search) with its first column's value for that id. -/
theorem importResolved_reads (d : D) (h : SInv d) (m : List (Nat × List Entry))
    (hnd : (m.map (·.1)).Nodup) (hb : ∀ p ∈ m, ∀ e ∈ p.2, e.1 ≤ 65535)
    (id : Nat) (ps : List Entry) (hps : lk m id = some ps) (hne : normResolved ps ≠ []) (k : Nat) :
    spanGet (dread (importResolved d m).1 id) k = if k = 65535 then none else get ps k := by
  have := (importResolved_spec d h m hnd hb).2 id
  simp only [hps, hne, ite_false] at this
  rw [this, spanGet_eq _ (normResolved_sorted ps), normResolved_get ps (hb _ (lk_mem m id ps hps)) k]

/-- **`resolved_duplicate_fixed`** (#3063): the header `p, p` of `pre3063_resolved_duplicate`
now stores one entry, the first column's. -/
theorem resolved_duplicate_fixed :
    let d := (importResolved emptyD [(0, [(0, .inl 1), (0, .inl 2)])]).1
    dread d 0 = [(0, .inl 1)] := by
  have hs := (importResolved_spec emptyD new_spec.1 [(0, [(0, .inl 1), (0, .inl 2)])] (by simp)
    (by decide)).2 0
  have hn : normResolved [(0, .inl 1), (0, .inl 2)] = [(0, .inl 1)] := by
    have hsort : sortById [(0, .inl 1), (0, .inl 2)] = [(0, .inl 1), (0, .inl 2)] :=
      List.mergeSort_of_pairwise (by simp)
    simp only [normResolved, hsort]
    rw [dedupId]; simp only [↓reduceIte]
    simp [dedupId, dropNone]
  simp only [lk, ite_true, hn] at hs
  simpa using hs

end PendingCommit.Content
