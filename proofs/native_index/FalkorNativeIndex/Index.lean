import FalkorNativeIndex.Numeric
set_option linter.unusedSimpArgs false

/-!
# `IndexColumn`, `BuildCursor`, `IndexEntry`, `FalkorDbIndex`
# (`graph/src/index/falkordb/falkordb_index.rs:41-596`, `versioned.rs:36-185`)

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

/-- `IndexColumn` (`falkordb_index.rs:41-43`): one kind today. -/
inductive IndexColumn where
  | numeric (ix : NumericIndex)
deriving Repr

def IndexColumn.ix : IndexColumn → NumericIndex
  | .numeric ix => ix

/-- `IndexColumn::add` (`:48-56`). -/
def IndexColumn.add (c : IndexColumn) (v : Value) (id : Nat) : IndexColumn :=
  match c with | .numeric ix => .numeric (ix.add v id)
/-- `IndexColumn::remove` (`:59-67`). -/
def IndexColumn.remove (c : IndexColumn) (v : Value) (id : Nat) : IndexColumn :=
  match c with | .numeric ix => .numeric (ix.remove v id)
/-- `IndexColumn::add_batch` (`:70-78`). -/
def IndexColumn.addBatch (c : IndexColumn) (es : List (Value × Nat)) (scratch : List Nat) : IndexColumn × List Nat :=
  match c with | .numeric ix => let r := ix.addBatch es scratch; (.numeric r.1, r.2)
/-- `IndexColumn::remove_batch` (`:81-89`). -/
def IndexColumn.removeBatch (c : IndexColumn) (es : List (Value × Nat)) (scratch : List Nat) : IndexColumn × List Nat :=
  match c with | .numeric ix => let r := ix.removeBatch es scratch; (.numeric r.1, r.2)
/-- `IndexColumn::encode_value` (`:94-101`). -/
def IndexColumn.encodeValue (c : IndexColumn) (v : Value) : Option Nat :=
  match c with | .numeric _ => encodeNumeric v
/-- `IndexColumn::push_encoded` (`:106-115`): append the tuple `(key, id)` to `out` if the column
    indexes `v`. A tuple is its tree entry `enc key id`. -/
def IndexColumn.pushEncoded (c : IndexColumn) (v : Value) (id : Nat) (out : EncodedTuples) : EncodedTuples :=
  match c.encodeValue v with
  | some key => ⟨out.scalar ++ [enc key id]⟩
  | none => out
/-- `IndexColumn::add_encoded` (`:118-125`). -/
def IndexColumn.addEncoded (c : IndexColumn) (tuples : EncodedTuples) : IndexColumn × EncodedTuples :=
  match c with | .numeric ix => let r := ix.addEncoded tuples; (.numeric r.1, r.2)
/-- `IndexColumn::remove_encoded` (`:128-135`). -/
def IndexColumn.removeEncoded (c : IndexColumn) (tuples : EncodedTuples) : IndexColumn × EncodedTuples :=
  match c with | .numeric ix => let r := ix.removeEncoded tuples; (.numeric r.1, r.2)
/-- `IndexColumn::is_empty` (`:139-143`). -/
def IndexColumn.isEmpty (c : IndexColumn) : Bool :=
  match c with | .numeric ix => ix.isEmpty

/-! ## Build state (`versioned.rs`, `BuildCursor` in `falkordb_index.rs`) -/

/-- `BuildCursor` (`falkordb_index.rs:218-225`): where a building column's next batch resumes, and an
    entity's place in that order. -/
inductive BuildCursor where
  | node (id : Nat)
  | edge (src dst : Nat)
deriving DecidableEq, Repr

namespace BuildCursor
/-- `BuildCursor::precedes` (`:231-246`): a node by its id, an edge by its `(src, dst)` pair
    (Rust's tuple `<`, lexicographic). The `unreachable!` on a node/edge mix is modelled as `false`;
    `kind_new`/`kind_place` (`Build.lean`) show it never fires. -/
def precedes : BuildCursor → BuildCursor → Bool
  | node a, node n => decide (a < n)
  | edge s d, edge ns nd => decide (s < ns ∨ (s = ns ∧ d < nd))
  | _, _ => false
/-- `BuildCursor::new` (`:249-257`): the first node, or the first pair. -/
def new : EntityType → BuildCursor
  | .node => node 0
  | .relationship => edge 0 0
/-- Whether the cursor is of `entity`'s kind. -/
def OfKind : EntityType → BuildCursor → Prop
  | .node, node _ => True
  | .relationship, edge _ _ => True
  | _, _ => False
end BuildCursor

/-- `BuildState<BuildCursor>` (`versioned.rs:36-42`). -/
inductive BuildState where
  | building (next : BuildCursor)
  | ready
deriving DecidableEq, Repr

/-- `IndexEntry<IndexColumn>` (`versioned.rs:54-57`): the column and its build state. -/
structure IndexEntry where
  index : IndexColumn
  state : BuildState
deriving Repr

namespace IndexEntry
/-- `IndexEntry::new` (`versioned.rs:126-131`). -/
def new (index : IndexColumn) (state : BuildState) : IndexEntry := ⟨index, state⟩
/-- `is_building` (`versioned.rs:134-136`). -/
def isBuilding (en : IndexEntry) : Bool := match en.state with | .building _ => true | .ready => false
/-- `cursor` (`versioned.rs:139-144`). -/
def cursor (en : IndexEntry) : Option BuildCursor :=
  match en.state with | .building next => some next | .ready => none
/-- `indexer_apply` (`versioned.rs:161-172`): add a batch (`indexer_add` = `add_encoded`), then move the
    cursor to `next` or flip `Ready`; `false`, changing nothing, if not `Building`. -/
def indexerApply (en : IndexEntry) (work : EncodedTuples) (next : Option BuildCursor) :
    IndexEntry × EncodedTuples × Bool :=
  if en.isBuilding then
    let r := en.index.addEncoded work
    (⟨r.1, match next with | none => .ready | some n => .building n⟩, r.2, true)
  else (en, work, false)
end IndexEntry

/-- `FalkorDbIndex` (`:273-276`). -/
structure FalkorDbIndex where
  nodeColumns : List (IndexKey × IndexEntry)
  edgeColumns : List (IndexKey × IndexEntry)
deriving Repr

namespace FalkorDbIndex

/-- `new` (`:281-283`). -/
def new : FalkorDbIndex := ⟨[], []⟩
/-- `is_empty` (`:287-289`). -/
def isEmpty (x : FalkorDbIndex) : Bool := x.nodeColumns.isEmpty && x.edgeColumns.isEmpty
/-- `columns` (`:309-317`). -/
def columns (x : FalkorDbIndex) : EntityType → List (IndexKey × IndexEntry)
  | .node => x.nodeColumns
  | .relationship => x.edgeColumns
/-- `columns_mut` (`:319-327`), as a functional update. -/
def setColumns (x : FalkorDbIndex) : EntityType → List (IndexKey × IndexEntry) → FalkorDbIndex
  | .node, m => { x with nodeColumns := m }
  | .relationship, m => { x with edgeColumns := m }
def columnsMut (x : FalkorDbIndex) (e : EntityType) (f : List (IndexKey × IndexEntry) → List (IndexKey × IndexEntry)) :
    FalkorDbIndex := x.setColumns e (f (x.columns e))
/-- `has_columns` (`:294-299`). -/
def hasColumns (x : FalkorDbIndex) (e : EntityType) : Bool := !(x.columns e).isEmpty
/-- `len` (`:303-305`). -/
def len (x : FalkorDbIndex) : Nat := x.nodeColumns.length + x.edgeColumns.length
/-- `entry` (`:330-337`); also the trait's `VersionedIndex::entry` (`:157-162`, `entry_mut` `:164-169`
    names the same slot to change: `setEntry`). -/
def entry (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Option IndexEntry :=
  hmGet (x.columns e) (label, attr)
/-- The slot `entry_mut` hands out, written back. -/
def setEntry (x : FalkorDbIndex) (e : EntityType) (k : IndexKey) (en : IndexEntry) : FalkorDbIndex :=
  x.columnsMut e (fun m => hmUpdate m k (fun _ => en))
/-- `insert_column` (`:380-392`): put `column` at `(entity, label, attr)` in `state`, replacing. -/
def insertColumn (x : FalkorDbIndex) (e : EntityType) (label attr : String) (column : IndexColumn)
    (state : BuildState) : FalkorDbIndex :=
  x.columnsMut e (fun m => hmInsert m (label, attr) (IndexEntry.new column state))
/-- `create_numeric` (`:343-356`): an empty numeric column, `Ready`. -/
def createNumeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) : FalkorDbIndex :=
  x.insertColumn e label attr (.numeric NumericIndex.new) .ready
/-- `create_building` (`:362-377`): an empty numeric column, `Building` from the start. -/
def createBuilding (x : FalkorDbIndex) (e : EntityType) (label attr : String) : FalkorDbIndex :=
  x.insertColumn e label attr (.numeric NumericIndex.new) (.building (BuildCursor.new e))
/-- `drop_column` (`:395-403`). -/
def dropColumn (x : FalkorDbIndex) (e : EntityType) (label attr : String) : FalkorDbIndex :=
  x.columnsMut e (fun m => hmRemove m (label, attr))

/-- `IndexType` — only `Range` has native columns. -/
inductive IndexType where
  | range | fulltext | vector
deriving DecidableEq, Repr

/-- `create_unpopulated` (`:408-420`). -/
def createUnpopulated (x : FalkorDbIndex) (t : IndexType) (e : EntityType) (label : String) (attrs : List String) :
    FalkorDbIndex :=
  if t = .range then attrs.foldl (fun x a => x.createNumeric e label a) x else x
/-- `drop_index` (`:424-436`). -/
def dropIndex (x : FalkorDbIndex) (t : IndexType) (e : EntityType) (label : String) (attrs : List String) :
    FalkorDbIndex :=
  if t = .range then attrs.foldl (fun x a => x.dropColumn e label a) x else x
/-- `numeric` (`:442-455`). -/
def numeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Option NumericIndex :=
  (x.entry e label attr).map (fun en => en.index.ix)
/-- `numeric_mut` (`:458-474`): the column to mutate, if any; mutation is `setNumeric` (the state is
    kept). -/
def numericMut (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Option NumericIndex :=
  x.numeric e label attr
def setNumeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) (f : NumericIndex → NumericIndex) :
    FalkorDbIndex :=
  x.columnsMut e (fun m => hmUpdate m (label, attr) (fun en => ⟨.numeric (f en.index.ix), en.state⟩))
/-- `has_column` (`:478-486`). -/
def hasColumn (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Bool :=
  (x.entry e label attr).isSome
/-- `encode_value` (`:491-502`). -/
def encodeValue (x : FalkorDbIndex) (e : EntityType) (label attr : String) (v : Value) : Option Nat :=
  (x.entry e label attr).bind (fun en => en.index.encodeValue v)
/-- The `Ready` numeric column, the only one a read is served from. -/
def readyNumeric (x : FalkorDbIndex) (e : EntityType) (label attr : String) : Option NumericIndex :=
  (x.entry e label attr).bind (fun en => if en.isBuilding then none else some en.index.ix)
/-- `apply_indexer_batch` (`versioned.rs:178-185`) for the column `(e, k)`: `false`, changing nothing, if
    the column is gone or no longer `Building`. -/
def applyIndexerBatch (x : FalkorDbIndex) (e : EntityType) (k : IndexKey) (work : EncodedTuples)
    (next : Option BuildCursor) : FalkorDbIndex × Bool :=
  match x.entry e k.1 k.2 with
  | none => (x, false)
  | some en =>
    let r := en.indexerApply work next
    (x.setEntry e k r.1, r.2.2)

/-- The column a query is routed to (`query_numeric`'s `let key = match query`, `:569-588`): an
    `Equal`/`Range` only when its bounds encode (the column answers an unencodable bound with "no
    rows", not `None`), an `IN` list always (the column declines), an `OR` by its first member. -/
def queryKey : IndexQuery → Option String
  | .equal key v => if (encodeNumeric v).isSome then some key else none
  | .range key min max _ _ =>
    if (min.all (fun v => (encodeNumeric v).isSome)) && (max.all (fun v => (encodeNumeric v).isSome))
    then some key else none
  | .inList key _ => some key
  | .or cs => match cs with
    | .equal key _ :: _ => some key
    | .inList key _ :: _ => some key
    | _ => none
  | .other => none

/-- `query_numeric` (`:560-596`): a numeric `Equal`/`Range`, an `IN` list whose values are all numeric,
    or an `OR` of equalities and `IN` lists on one attribute (#2383), on an existing column that is
    not `Building` (#2351: not every entity is indexed yet, so RediSearch answers). -/
def queryNumeric (x : FalkorDbIndex) (e : EntityType) (label : String) (q : IndexQuery) : Option (List Nat) :=
  match queryKey q with
  | none => none
  | some key =>
    match x.entry e label key with
    | none => none
    | some entry =>
      if entry.isBuilding then none
      else match entry.index with
        | .numeric ix => ix.query q

end FalkorDbIndex

/-! ## Accessor theorems -/

open FalkorDbIndex

theorem columns_setColumns (x : FalkorDbIndex) (e e' : EntityType) (m) :
    (x.setColumns e m).columns e' = if e = e' then m else x.columns e' := by
  cases e <;> cases e' <;> rfl

theorem entry_setColumns (x : FalkorDbIndex) (e e' : EntityType) (m) (k : IndexKey) :
    (x.setColumns e m).entry e' k.1 k.2 = if e = e' then hmGet m k else x.entry e' k.1 k.2 := by
  unfold entry; rw [columns_setColumns]; split <;> rfl

theorem entry_insertColumn (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) (c st) :
    (x.insertColumn e l a c st).entry e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then some ⟨c, st⟩ else x.entry e' l' a' := by
  unfold insertColumn columnsMut entry
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_insert]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2, IndexEntry.new]
  · simp [h]

theorem numeric_createBuilding (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) :
    (x.createBuilding e l a).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then some NumericIndex.new else x.numeric e' l' a' := by
  unfold createBuilding numeric
  rw [entry_insertColumn]
  by_cases h : e = e' ∧ (l, a) = (l', a')
  · simp only [if_pos h, Option.map_some]; rfl
  · simp only [if_neg h]

/-- A fresh column is `Building` from the start of its entity's build order. -/
theorem entry_createBuilding (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    (x.createBuilding e l a).entry e l a = some ⟨.numeric NumericIndex.new, .building (BuildCursor.new e)⟩ := by
  unfold createBuilding; rw [entry_insertColumn]; simp

/-- `create_numeric` (RDB load) is `Ready`, `create_building` is not, so a read of it is declined. -/
theorem entry_createNumeric (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    (x.createNumeric e l a).entry e l a = some ⟨.numeric NumericIndex.new, .ready⟩ := by
  unfold createNumeric; rw [entry_insertColumn]; simp

theorem numeric_createNumeric (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) :
    (x.createNumeric e l a).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then some NumericIndex.new else x.numeric e' l' a' := by
  unfold createNumeric numeric
  rw [entry_insertColumn]
  by_cases h : e = e' ∧ (l, a) = (l', a')
  · simp only [if_pos h, Option.map_some]; rfl
  · simp only [if_neg h]

theorem numeric_dropColumn (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) :
    (x.dropColumn e l a).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then none else x.numeric e' l' a' := by
  unfold dropColumn columnsMut numeric entry
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_remove]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2]
  · simp [h]

/-- Node and relationship columns never collide (`falkordb_index.rs:273-276`, #51). -/
theorem node_edge_independent (x : FalkorDbIndex) (l a l' a' : String) :
    (x.createBuilding .relationship l a).numeric .node l' a' = x.numeric .node l' a' := by
  rw [numeric_createBuilding]; simp

theorem setNumeric_numeric (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) (f) :
    (x.setNumeric e l a f).numeric e' l' a' =
      if e = e' ∧ (l, a) = (l', a') then (x.numeric e' l' a').map f else x.numeric e' l' a' := by
  unfold setNumeric columnsMut numeric entry
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, true_and, hmGet_update]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2, Option.map_map, IndexColumn.ix]
    · rfl
  · simp [h]

/-- `numeric_mut`'s write keeps the build state. -/
theorem setNumeric_state (x : FalkorDbIndex) (e e' : EntityType) (l a l' a' : String) (f) :
    ((x.setNumeric e l a f).entry e' l' a').map (·.state) = (x.entry e' l' a').map (·.state) := by
  unfold setNumeric columnsMut entry
  rw [columns_setColumns]
  by_cases h : e = e'
  · subst h; simp only [if_true, hmGet_update]
    by_cases h2 : (l, a) = (l', a') <;> simp [h2, Option.map_map, Function.comp_def]
  · simp [h]

theorem encodeValue_spec (x : FalkorDbIndex) (e : EntityType) (l a : String) (v : Value) :
    x.encodeValue e l a v = if x.hasColumn e l a then encodeNumeric v else none := by
  unfold encodeValue hasColumn
  cases h : x.entry e l a with
  | none => simp
  | some c => obtain ⟨⟨_⟩, _⟩ := c; simp [IndexColumn.encodeValue]

theorem hasColumns_spec (x : FalkorDbIndex) (e : EntityType) : x.hasColumns e = !(x.columns e).isEmpty := rfl
theorem isEmpty_spec' (x : FalkorDbIndex) : x.isEmpty = (x.nodeColumns.isEmpty && x.edgeColumns.isEmpty) := rfl
theorem len_spec (x : FalkorDbIndex) : x.len = x.nodeColumns.length + x.edgeColumns.length := rfl
theorem new_empty : FalkorDbIndex.new.isEmpty = true ∧ FalkorDbIndex.new.len = 0 := ⟨rfl, rfl⟩
theorem isBuilding_spec (c : IndexColumn) (st : BuildState) :
    (IndexEntry.new c st).isBuilding = true ↔ ∃ n, st = .building n := by
  cases st <;> simp [IndexEntry.new, IndexEntry.isBuilding]
theorem cursor_spec (c : IndexColumn) (st : BuildState) :
    (IndexEntry.new c st).cursor = match st with | .building n => some n | .ready => none := rfl
theorem precedes_spec (a b : Nat) (s d ns nd : Nat) :
    (BuildCursor.node a).precedes (.node b) = decide (a < b) ∧
    (BuildCursor.edge s d).precedes (.edge ns nd) = decide (s < ns ∨ (s = ns ∧ d < nd)) := ⟨rfl, rfl⟩
theorem entry_spec (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    x.entry e l a = hmGet (x.columns e) (l, a) := rfl

/-- `indexer_apply` on a column that is not `Building` changes nothing and reports `false`. -/
theorem indexerApply_idle (en : IndexEntry) (work : EncodedTuples) (next : Option BuildCursor)
    (h : en.isBuilding = false) : en.indexerApply work next = (en, work, false) := by
  simp [IndexEntry.indexerApply, h]

/-- `indexer_apply` on a `Building` column adds the batch and moves the cursor to `next`, or flips the
    column `Ready` when there is none. -/
theorem indexerApply_building (en : IndexEntry) (work : EncodedTuples) (next : Option BuildCursor)
    (h : en.isBuilding = true) :
    (en.indexerApply work next).2.2 = true ∧
    (en.indexerApply work next).1.index = (en.index.addEncoded work).1 ∧
    (en.indexerApply work next).1.state = (match next with | none => .ready | some n => .building n) := by
  simp [IndexEntry.indexerApply, h]

/-- `apply_indexer_batch` on a missing column changes nothing. -/
theorem applyIndexerBatch_missing (x : FalkorDbIndex) (e : EntityType) (k : IndexKey) (work : EncodedTuples)
    (next : Option BuildCursor) (h : x.entry e k.1 k.2 = none) :
    x.applyIndexerBatch e k work next = (x, false) := by
  simp [FalkorDbIndex.applyIndexerBatch, h]

theorem new_cursor : BuildCursor.new .node = .node 0 ∧ BuildCursor.new .relationship = .edge 0 0 := ⟨rfl, rfl⟩
theorem numericMut_spec (x : FalkorDbIndex) (e : EntityType) (l a : String) : x.numericMut e l a = x.numeric e l a := rfl
theorem hasColumn_spec (x : FalkorDbIndex) (e : EntityType) (l a : String) :
    x.hasColumn e l a = (x.numeric e l a).isSome := by
  simp [hasColumn, numeric, Option.isSome_map]

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

/-- **`query_numeric` routing**: it answers exactly a numeric `Equal`/`Range`, an `IN` list or a
    same-attribute `OR` (the column declines what it cannot answer whole) on an existing column that
    is not `Building`, from that column; everything else, including every `Building` column, falls
    through (`None`) to RediSearch. -/
theorem queryNumeric_spec (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) :
    x.queryNumeric e l q =
      match q with
      | .equal key v => if (encodeNumeric v).isSome then (x.readyNumeric e l key).map (fun ix => ix.point v) else none
      | .range key mn mx a b =>
        if (mn.all (fun v => (encodeNumeric v).isSome)) && (mx.all (fun v => (encodeNumeric v).isSome))
        then (x.readyNumeric e l key).map (fun ix => ix.range mn mx a b) else none
      | .inList key (some items) => (x.readyNumeric e l key).bind (fun ix => ix.inList items)
      | .inList _ none => none
      | .or cs => match cs with
        | .equal key _ :: _ => (x.readyNumeric e l key).bind (fun ix => ix.union cs)
        | .inList key _ :: _ => (x.readyNumeric e l key).bind (fun ix => ix.union cs)
        | _ => none
      | .other => none := by
  have routed : ∀ key, queryKey q = some key →
      x.queryNumeric e l q = (x.readyNumeric e l key).bind (fun ix => ix.query q) := by
    intro key hk
    simp only [queryNumeric, hk, readyNumeric]
    cases x.entry e l key with
    | none => rfl
    | some c =>
      obtain ⟨⟨ix⟩, st⟩ := c
      by_cases hb : IndexEntry.isBuilding ⟨.numeric ix, st⟩ = true <;> simp [hb, IndexColumn.ix]
  have declined : queryKey q = none → x.queryNumeric e l q = none := by
    intro hk; simp only [queryNumeric, hk]
  cases q with
  | equal key v =>
    by_cases h : (encodeNumeric v).isSome = true
    · simp only [h, if_true]; rw [routed key (by simp [queryKey, h])]
      generalize x.readyNumeric e l key = o; cases o <;> rfl
    · simp only [h, if_false]; exact declined (by simp [queryKey, h])
  | range key mn mx a b =>
    by_cases h : ((mn.all (fun v => (encodeNumeric v).isSome)) && (mx.all (fun v => (encodeNumeric v).isSome))) = true
    · simp only [h, if_true]; rw [routed key (by simp only [queryKey]; simp only [h, if_true])]
      generalize x.readyNumeric e l key = o; cases o <;> rfl
    · simp only [h, if_false]; exact declined (by simp only [queryKey]; simp [h])
  | inList key items =>
    rw [routed key rfl]
    cases items <;> (dsimp only [NumericIndex.query]; try (generalize x.readyNumeric e l key = o; cases o <;> rfl))
  | or cs =>
    rcases cs with _ | ⟨c, cs'⟩
    · exact declined rfl
    · cases c with
      | equal key v => rw [routed key rfl]; rfl
      | inList key it => rw [routed key rfl]; rfl
      | range _ _ _ _ _ => exact declined rfl
      | or _ => exact declined rfl
      | other => exact declined rfl
  | other => exact declined rfl

/-- **A `Building` column is never read from** (#2351): `query_numeric` declines every query routed to it,
    so RediSearch answers until the last batch flips it `Ready`. -/
theorem queryNumeric_building (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (key : String)
    (hk : queryKey q = some key) (en : IndexEntry) (hen : x.entry e l key = some en)
    (hb : en.isBuilding = true) : x.queryNumeric e l q = none := by
  simp [queryNumeric, hk, hen, hb]

/-- ... and a `Ready` column answers as before. -/
theorem queryNumeric_ready (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (key : String)
    (hk : queryKey q = some key) (c : NumericIndex) (st : BuildState) (hen : x.entry e l key = some ⟨.numeric c, st⟩)
    (hb : st = .ready) : x.queryNumeric e l q = c.query q := by
  subst hb
  simp [queryNumeric, hk, hen, IndexEntry.isBuilding]

end NI
