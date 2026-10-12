import FalkorNativeIndex.Numeric
set_option linter.unusedSimpArgs false

/-!
# `IndexColumn`, `FalkorDbIndex` (`graph/src/index/falkordb/falkordb_index.rs:38-432`)

A `HashMap<K, V>` is an association list with unique keys (`hmGet`/`hmInsert`/`hmRemove`); its
iteration order (`drain`) is arbitrary, so every theorem about a whole map is stated per key.
-/

namespace NI

/-! ## `HashMap` -/

def hmGet {K V} [DecidableEq K] (m : List (K × V)) (k : K) : Option V := (m.find? (fun p => p.1 == k)).map (·.2)
def hmInsert {K V} [DecidableEq K] (m : List (K × V)) (k : K) (v : V) : List (K × V) :=
  (k, v) :: m.filter (fun p => !(p.1 == k))
def hmRemove {K V} [DecidableEq K] (m : List (K × V)) (k : K) : List (K × V) := m.filter (fun p => !(p.1 == k))
def hmUpdate {K V} [DecidableEq K] (m : List (K × V)) (k : K) (f : V → V) : List (K × V) :=
  m.map (fun p => if p.1 == k then (p.1, f p.2) else p)

theorem find_filter_ne {K V} [DecidableEq K] (m : List (K × V)) (k k' : K) (h : k ≠ k') :
    (m.filter (fun p => !(p.1 == k))).find? (fun p => p.1 == k') = m.find? (fun p => p.1 == k') := by
  induction m with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨pk, pv⟩ := p
    by_cases h1 : pk = k
    · subst h1; simp [List.filter_cons, h, ih]
    · by_cases h2 : pk = k'
      · subst h2; simp [List.filter_cons, h1]
      · simp [List.filter_cons, h1, h2, ih]

theorem find_filter_eq {K V} [DecidableEq K] (m : List (K × V)) (k : K) :
    (m.filter (fun p => !(p.1 == k))).find? (fun p => p.1 == k) = none := by
  induction m with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨pk, pv⟩ := p
    by_cases h1 : pk = k
    · subst h1; simp [List.filter_cons, ih]
    · simp [List.filter_cons, h1, ih]

theorem hmGet_insert {K V} [DecidableEq K] (m : List (K × V)) (k k' : K) (v : V) :
    hmGet (hmInsert m k v) k' = if k = k' then some v else hmGet m k' := by
  unfold hmGet hmInsert
  by_cases h : k = k'
  · subst h; simp
  · simp [h, find_filter_ne m k k' h]

theorem hmGet_remove {K V} [DecidableEq K] (m : List (K × V)) (k k' : K) :
    hmGet (hmRemove m k) k' = if k = k' then none else hmGet m k' := by
  unfold hmGet hmRemove
  by_cases h : k = k'
  · subst h; simp [find_filter_eq]
  · simp [h, find_filter_ne m k k' h]

theorem hmGet_update {K V} [DecidableEq K] (m : List (K × V)) (k k' : K) (f : V → V) :
    hmGet (hmUpdate m k f) k' = if k = k' then (hmGet m k').map f else hmGet m k' := by
  unfold hmGet hmUpdate
  induction m with
  | nil => simp
  | cons p ps ih =>
    obtain ⟨pk, pv⟩ := p
    by_cases h1 : pk = k <;> by_cases h2 : pk = k'
    · subst h1 h2; simp
    · subst h1; simp [h2] at *; exact ih
    · subst h2; simp [h1, Ne.symm h1]
    · simp [h1, h2] at *; exact ih

/-! ## Columns -/

inductive EntityType where
  | node | relationship
deriving DecidableEq, Repr

/-- `IndexKey = (label, attribute)`. -/
abbrev IndexKey := String × String

/-- `IndexColumn` (`falkordb_index.rs:38-40`): one kind today. -/
inductive IndexColumn where
  | numeric (ix : NumericIndex)
deriving Repr

def IndexColumn.ix : IndexColumn → NumericIndex
  | .numeric ix => ix

/-- `IndexColumn::add` (`:45-53`). -/
def IndexColumn.add (c : IndexColumn) (v : Value) (id : Nat) : IndexColumn :=
  match c with | .numeric ix => .numeric (ix.add v id)
/-- `IndexColumn::remove` (`:56-64`). -/
def IndexColumn.remove (c : IndexColumn) (v : Value) (id : Nat) : IndexColumn :=
  match c with | .numeric ix => .numeric (ix.remove v id)
/-- `IndexColumn::add_batch` (`:67-75`). -/
def IndexColumn.addBatch (c : IndexColumn) (es : List (Value × Nat)) (scratch : List Nat) : IndexColumn × List Nat :=
  match c with | .numeric ix => let r := ix.addBatch es scratch; (.numeric r.1, r.2)
/-- `IndexColumn::remove_batch` (`:78-86`). -/
def IndexColumn.removeBatch (c : IndexColumn) (es : List (Value × Nat)) (scratch : List Nat) : IndexColumn × List Nat :=
  match c with | .numeric ix => let r := ix.removeBatch es scratch; (.numeric r.1, r.2)
/-- `IndexColumn::encode_value` (`:91-98`). -/
def IndexColumn.encodeValue (c : IndexColumn) (v : Value) : Option Nat :=
  match c with | .numeric _ => encodeNumeric v
/-- `IndexColumn::add_encoded` (`:101-108`). -/
def IndexColumn.addEncoded (c : IndexColumn) (pairs : List Nat) : IndexColumn × List Nat :=
  match c with | .numeric ix => let r := ix.addEncoded pairs; (.numeric r.1, r.2)
/-- `IndexColumn::remove_encoded` (`:111-118`). -/
def IndexColumn.removeEncoded (c : IndexColumn) (pairs : List Nat) : IndexColumn × List Nat :=
  match c with | .numeric ix => let r := ix.removeEncoded pairs; (.numeric r.1, r.2)
/-- `IndexColumn::is_empty` (`:122-126`). -/
def IndexColumn.isEmpty (c : IndexColumn) : Bool :=
  match c with | .numeric ix => ix.isEmpty

/-- `ColumnEntry` (`:131-133`). -/
structure ColumnEntry where
  column : IndexColumn
deriving Repr

/-- `ColumnEntry::ready` (`:138-140`). -/
def ColumnEntry.ready (c : IndexColumn) : ColumnEntry := ⟨c⟩

/-- `FalkorDbIndex` (`:156-159`). -/
structure FalkorDbIndex where
  nodeColumns : List (IndexKey × ColumnEntry)
  edgeColumns : List (IndexKey × ColumnEntry)
deriving Repr

namespace FalkorDbIndex

/-- `new` (`:164-166`). -/
def new : FalkorDbIndex := ⟨[], []⟩
/-- `is_empty` (`:170-172`). -/
def isEmpty (x : FalkorDbIndex) : Bool := x.nodeColumns.isEmpty && x.edgeColumns.isEmpty
/-- `columns` (`:192-200`). -/
def columns (x : FalkorDbIndex) : EntityType → List (IndexKey × ColumnEntry)
  | .node => x.nodeColumns
  | .relationship => x.edgeColumns
/-- `columns_mut` (`:202-210`), as a functional update. -/
def setColumns (x : FalkorDbIndex) : EntityType → List (IndexKey × ColumnEntry) → FalkorDbIndex
  | .node, m => { x with nodeColumns := m }
  | .relationship, m => { x with edgeColumns := m }
def columnsMut (x : FalkorDbIndex) (e : EntityType) (f : List (IndexKey × ColumnEntry) → List (IndexKey × ColumnEntry)) :
    FalkorDbIndex := x.setColumns e (f (x.columns e))
/-- `has_columns` (`:177-182`). -/
def hasColumns (x : FalkorDbIndex) (e : EntityType) : Bool := !(x.columns e).isEmpty
/-- `len` (`:186-188`). -/
def len (x : FalkorDbIndex) : Nat := x.nodeColumns.length + x.edgeColumns.length
/-- `create_numeric` (`:216-226`). -/
def createNumeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) : FalkorDbIndex :=
  x.columnsMut e (fun m => hmInsert m (label, attr) (ColumnEntry.ready (.numeric NumericIndex.new)))
/-- `build_numeric` (`:234-245`). -/
def buildNumeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) (entries : List (Value × Nat)) :
    FalkorDbIndex :=
  x.columnsMut e (fun m => hmInsert m (label, attr) (ColumnEntry.ready (.numeric (NumericIndex.fromEntries entries))))
/-- `drop_column` (`:248-256`). -/
def dropColumn (x : FalkorDbIndex) (e : EntityType) (label attr : String) : FalkorDbIndex :=
  x.columnsMut e (fun m => hmRemove m (label, attr))

/-- `IndexType` — only `Range` has native columns. -/
inductive IndexType where
  | range | fulltext | vector
deriving DecidableEq, Repr

/-- `create_unpopulated` (`:261-273`). -/
def createUnpopulated (x : FalkorDbIndex) (t : IndexType) (e : EntityType) (label : String) (attrs : List String) :
    FalkorDbIndex :=
  if t = .range then attrs.foldl (fun x a => x.createNumeric e label a) x else x
/-- `drop_index` (`:277-289`). -/
def dropIndex (x : FalkorDbIndex) (t : IndexType) (e : EntityType) (label : String) (attrs : List String) :
    FalkorDbIndex :=
  if t = .range then attrs.foldl (fun x a => x.dropColumn e label a) x else x
/-- `numeric` (`:295-308`). -/
def numeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Option NumericIndex :=
  (hmGet (x.columns e) (label, attr)).map (fun c => c.column.ix)
/-- `numeric_mut` (`:311-327`): the column to mutate, if any; mutation is `setNumeric`. -/
def numericMut (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Option NumericIndex :=
  x.numeric e label attr
def setNumeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) (f : NumericIndex → NumericIndex) :
    FalkorDbIndex :=
  x.columnsMut e (fun m => hmUpdate m (label, attr) (fun c => ⟨.numeric (f c.column.ix)⟩))
/-- `has_column` (`:331-339`). -/
def hasColumn (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Bool :=
  (hmGet (x.columns e) (label, attr)).isSome
/-- `encode_value` (`:344-355`). -/
def encodeValue (x : FalkorDbIndex) (e : EntityType) (label attr : String) (v : Value) : Option Nat :=
  (hmGet (x.columns e) (label, attr)).bind (fun c => c.column.encodeValue v)

/-- `query_numeric` (`:409-432`): only a numeric `Equal`/`Range` leaf, on an existing column. -/
def queryNumeric (x : FalkorDbIndex) (e : EntityType) (label : String) (q : IndexQuery) : Option (List Nat) :=
  let r : Option (String × Bool) := match q with
    | .equal key v => some (key, (encodeNumeric v).isSome)
    | .range key min max _ _ =>
      some (key, (min.all (fun v => (encodeNumeric v).isSome)) && (max.all (fun v => (encodeNumeric v).isSome)))
    | .other => none
  match r with
  | none => none
  | some (key, numeric) =>
    if !numeric then none
    else match hmGet (x.columns e) (label, key) with
      | none => none
      | some entry => match entry.column with
        | .numeric ix => ix.query q

end FalkorDbIndex

/-! ## Accessor theorems -/

open FalkorDbIndex

theorem columns_setColumns (x : FalkorDbIndex) (e e' : EntityType) (m) :
    (x.setColumns e m).columns e' = if e = e' then m else x.columns e' := by
  cases e <;> cases e' <;> rfl

theorem numeric_buildNumeric (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) (es) :
    (x.buildNumeric e l a es).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then some (NumericIndex.fromEntries es) else x.numeric e' l' a' := by
  unfold buildNumeric columnsMut numeric
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_insert]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2, ColumnEntry.ready, IndexColumn.ix]
  · simp [h]

theorem numeric_createNumeric (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) :
    (x.createNumeric e l a).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then some NumericIndex.new else x.numeric e' l' a' := by
  unfold createNumeric columnsMut numeric
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_insert]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2, ColumnEntry.ready, IndexColumn.ix]
  · simp [h]

theorem numeric_dropColumn (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) :
    (x.dropColumn e l a).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then none else x.numeric e' l' a' := by
  unfold dropColumn columnsMut numeric
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_remove]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2]
  · simp [h]

/-- Node and relationship columns never collide (`falkordb_index.rs:525-557`, #51). -/
theorem node_edge_independent (x : FalkorDbIndex) (l a l' a' : String) (es) :
    (x.buildNumeric .relationship l a es).numeric .node l' a' = x.numeric .node l' a' := by
  rw [numeric_buildNumeric]; simp

theorem setNumeric_numeric (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) (f) :
    (x.setNumeric e l a f).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then (x.numeric e' l' a').map f else x.numeric e' l' a' := by
  unfold setNumeric columnsMut numeric
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_update]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2, Option.map_map, IndexColumn.ix]
    · rfl
  · simp [h]

theorem encodeValue_spec (x : FalkorDbIndex) (e : EntityType) (l a : String) (v : Value) :
    x.encodeValue e l a v = if x.hasColumn e l a then encodeNumeric v else none := by
  unfold encodeValue hasColumn
  cases h : hmGet (x.columns e) (l, a) with
  | none => simp
  | some c => obtain ⟨⟨_⟩⟩ := c; simp [IndexColumn.encodeValue]

theorem hasColumns_spec (x : FalkorDbIndex) (e : EntityType) : x.hasColumns e = !(x.columns e).isEmpty := rfl
theorem isEmpty_spec' (x : FalkorDbIndex) : x.isEmpty = (x.nodeColumns.isEmpty && x.edgeColumns.isEmpty) := rfl
theorem len_spec (x : FalkorDbIndex) : x.len = x.nodeColumns.length + x.edgeColumns.length := rfl
theorem new_empty : FalkorDbIndex.new.isEmpty = true ∧ FalkorDbIndex.new.len = 0 := ⟨rfl, rfl⟩
theorem ready_spec (c : IndexColumn) : (ColumnEntry.ready c).column = c := rfl
theorem numericMut_spec (x : FalkorDbIndex) (e : EntityType) (l a : String) : x.numericMut e l a = x.numeric e l a := rfl
theorem hasColumn_spec (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    x.hasColumn e l a = (x.numeric e l a).isSome := by
  simp [hasColumn, numeric]

theorem createUnpopulated_nonrange (x : FalkorDbIndex) (t : IndexType) (h : t ≠ .range) (e l attrs) :
    x.createUnpopulated t e l attrs = x := by simp [createUnpopulated, h]
theorem dropIndex_nonrange (x : FalkorDbIndex) (t : IndexType) (h : t ≠ .range) (e l attrs) :
    x.dropIndex t e l attrs = x := by simp [dropIndex, h]

/-- `drop_index` of a Range index removes each listed column. -/
theorem dropIndex_removes (x : FalkorDbIndex) (e : EntityType) (l : String) :
    ∀ (attrs : List String) (a : String), a ∈ attrs → (x.dropIndex .range e l attrs).numeric e l a = none := by
  intro attrs
  unfold dropIndex; simp only [if_true]
  induction attrs generalizing x with
  | nil => intro a h; simp at h
  | cons b bs ih =>
    intro a h
    simp only [List.foldl_cons]
    by_cases hb : a ∈ bs
    · exact ih _ a hb
    · have : a = b := by simp at h; exact h.resolve_right hb
      subst this
      -- later drops do not resurrect it
      have keep : ∀ (ys : List String) (y : FalkorDbIndex), y.numeric e l a = none →
          (ys.foldl (fun x a => x.dropColumn e l a) y).numeric e l a = none := by
        intro ys; induction ys with
        | nil => intro y h; exact h
        | cons c cs ih2 => intro y h; simp only [List.foldl_cons]; apply ih2; rw [numeric_dropColumn]; split <;> simp_all
      apply keep; rw [numeric_dropColumn]; simp

/-- `create_unpopulated` of a Range index leaves each listed column present and empty. -/
theorem createUnpopulated_creates (x : FalkorDbIndex) (e : EntityType) (l : String) :
    ∀ (attrs : List String) (a : String), a ∈ attrs →
      (x.createUnpopulated .range e l attrs).numeric e l a = some NumericIndex.new := by
  intro attrs
  unfold createUnpopulated; simp only [if_true]
  induction attrs generalizing x with
  | nil => intro a h; simp at h
  | cons b bs ih =>
    intro a h
    simp only [List.foldl_cons]
    by_cases hb : a ∈ bs
    · exact ih _ a hb
    · have : a = b := by simp at h; exact h.resolve_right hb
      subst this
      have keep : ∀ (ys : List String) (y : FalkorDbIndex), y.numeric e l a = some NumericIndex.new →
          (ys.foldl (fun x a => x.createNumeric e l a) y).numeric e l a = some NumericIndex.new := by
        intro ys; induction ys with
        | nil => intro y h; exact h
        | cons c cs ih2 => intro y h; simp only [List.foldl_cons]; apply ih2; rw [numeric_createNumeric]; split <;> simp_all
      apply keep; rw [numeric_createNumeric]; simp

/-- **`query_numeric` routing**: it answers exactly a numeric `Equal`/`Range` on an existing column,
    from that column; everything else falls through (`None`). -/
theorem queryNumeric_spec (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) :
    x.queryNumeric e l q =
      match q with
      | .equal key v => if (encodeNumeric v).isSome then (x.numeric e l key).map (fun ix => ix.point v) else none
      | .range key mn mx a b =>
        if (mn.all (fun v => (encodeNumeric v).isSome)) && (mx.all (fun v => (encodeNumeric v).isSome))
        then (x.numeric e l key).map (fun ix => ix.range mn mx a b) else none
      | .other => none := by
  cases q with
  | equal key v =>
    simp only [queryNumeric, numeric]
    cases (encodeNumeric v).isSome <;> simp
    cases hmGet (x.columns e) (l, key) with
    | none => rfl
    | some c => obtain ⟨⟨ix⟩⟩ := c; rfl
  | range key mn mx a b =>
    simp only [queryNumeric, numeric]
    cases ((mn.all (fun v => (encodeNumeric v).isSome)) && (mx.all (fun v => (encodeNumeric v).isSome))) <;> simp
    cases hmGet (x.columns e) (l, key) with
    | none => rfl
    | some c => obtain ⟨⟨ix⟩⟩ := c; rfl
  | other => rfl

end NI
