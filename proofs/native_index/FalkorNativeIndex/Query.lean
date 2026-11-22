import FalkorNativeIndex.Index
set_option linter.unusedSimpArgs false

/-!
# `query_column` (`falkordb_index.rs:978-1130`), `combine_columns`, `ScanFilter` (`:442-536`),
# `equal_candidates` (`:1159-1177`)

Since #3547 / #2386 the native index is the ONLY Range index under `index-falkordb` (there is no
RediSearch to fall through to), and since #3736 / #3737 the layer asks the column only for a `Selection`
(`Selection::of`, `Numeric.lean`). A read has **three** outcomes (`query_column`'s doc):

* `Some(Ok(rows))`: served from `Ready` columns;
* `Some(Err(scan_filter))`: a shape the columns serve, on a column still `Building`: the caller scans
  and keeps what the `ScanFilter` matches (a timing state, never an error);
* `None`: any other shape, or a missing column. The caller turns it into the error
  `unsupported_by_native_index`.

Modelled as `Option (Except ScanFilter (List Nat))` (`Except.ok` = `Ok`, `Except.error` = `Err`). A query
over several attributes is `combine_columns`: one column per attribute, intersected for an `And`, a
`DistinctIter` union for an `Or`.
-/

namespace NI

/-- `ScanFilter<String>` (`falkordb_index.rs:446-451`): every attribute the query constrains with what the
    built column for it selects; `all` = a conjunction, `any` = a union across attributes. -/
inductive ScanFilter where
  | all (conditions : List (String × Selection))
  | any (conditions : List (String × Selection))

/-- `ScanFilter::one` (`:459-465`): the filter of a query on one attribute. -/
def ScanFilter.one (attr : String) (sel : Selection) : ScanFilter := .all [(attr, sel)]

/-- `ScanFilter::matches` (`:503-515`): whether the built columns answer the query with an entity whose
    attributes `valueOf` reads. An entity missing an attribute does not match its condition. -/
def ScanFilter.matches (f : ScanFilter) (valueOf : String → Option Value) : Bool :=
  let holds : String × Selection → Bool := fun c =>
    match valueOf c.1 with
    | some v => c.2.matches v
    | none => false
  match f with
  | .all cs => cs.all holds
  | .any cs => cs.any holds

/-- The loop of `ScanFilter::resolve` (`:471-493`): a conjunction needs every attribute, a union only one. -/
def resolveGo (isAll : Bool) (r : String → Option String) : List (String × Selection) →
    Option (List (String × Selection))
  | [] => some []
  | (a, sel) :: rest =>
    match r a with
    | some b => (resolveGo isAll r rest).map (fun l => (b, sel) :: l)
    | none => if isAll then none else resolveGo isAll r rest

/-- `ScanFilter::resolve` (`:471-493`): the same filter with each attribute resolved; `None` when the whole
    filter therefore matches nothing. -/
def ScanFilter.resolve (f : ScanFilter) (r : String → Option String) : Option ScanFilter :=
  match f with
  | .all cs => match resolveGo true r cs with
    | none => none
    | some [] => none
    | some l => some (.all l)
  | .any cs => match resolveGo false r cs with
    | none => none
    | some [] => none
    | some l => some (.any l)

/-- `DistinctIter` (`numeric.rs:97-113`): the streams read one after another, each doc yielded once, at
    its first appearance. As a list of docs, the concatenation with later repeats dropped. -/
def distinctDocs (streams : List (List Nat)) : List Nat := streams.flatten.eraseDups

/-- The intersection of `combine_columns` (`:1100-1115`): the first column's docs, each later column
    keeping those it also holds (`None` when there is no column). -/
def interDocs : List (List Nat) → Option (List Nat)
  | [] => none
  | r :: rs => some (rs.foldl (fun seen rows =>
      if seen.isEmpty then seen else rows.filter (fun d => seen.contains d)) r)

/-- A column as `combine_columns` holds it: attribute, entry, selection. -/
abbrev Col := String × IndexEntry × Selection

/-- Add `child` to the members of attribute `k` in `acc`, in order of first appearance. -/
def pushGroup (acc : List (String × List IndexQuery)) (k : String) (child : IndexQuery) :
    List (String × List IndexQuery) :=
  if acc.any (fun p => p.1 == k) then acc.map (fun p => if p.1 == k then (p.1, p.2 ++ [child]) else p)
  else acc ++ [(k, [child])]

/-- The leaf of `group` (`:1049-1066`): a child naming one attribute joins that attribute's members; one
    naming none, or spanning attributes, is not groupable. -/
def groupLeaf (acc : List (String × List IndexQuery)) (child : IndexQuery) :
    Option (List (String × List IndexQuery)) :=
  match attrsOf child with
  | none => none
  | some [] => none
  | some (k :: ks) => if ks.any (fun k' => k' != k) then none else some (pushGroup acc k child)

mutual
/-- One child of `group`: a nested combinator of the kind being combined is opened into its own
    members (`And(And(a, b), c)`), anything else is a leaf. -/
def groupOne (any : Bool) (acc : List (String × List IndexQuery)) :
    IndexQuery → Option (List (String × List IndexQuery))
  | .and nested => if any then groupLeaf acc (.and nested) else groupList any acc nested
  | .or nested => if any then groupList any acc nested else groupLeaf acc (.or nested)
  | q => groupLeaf acc q
/-- The loop of `group`. -/
def groupList (any : Bool) (acc : List (String × List IndexQuery)) :
    List IndexQuery → Option (List (String × List IndexQuery))
  | [] => some acc
  | c :: cs => (groupOne any acc c).bind (fun acc' => groupList any acc' cs)
end

/-- `combine_columns` (`:1041-1148`): a conjunction (`any = false`) or union (`any = true`) spanning
    several attributes: one column per attribute. Every column must exist and every share must translate
    before any is read; one `Building` column makes the whole query a scan. -/
def FalkorDbIndex.combineColumns (x : FalkorDbIndex) (e : EntityType) (label : String)
    (children : List IndexQuery) (any : Bool) : Option (Except ScanFilter (List Nat)) :=
  match groupList any [] children with
  | none => none
  | some perAttr =>
    match perAttr.mapM (fun p => (x.entry e label p.1).bind (fun en =>
        (if any then Selection.disjunction p.2 else Selection.conjunction p.2).map
          (fun sel => ((p.1, en, sel) : Col)))) with
    | none => none
    | some cols =>
      if cols.any (fun c => c.2.1.isBuilding) then
        let conds := cols.map (fun c => (c.1, c.2.2))
        some (.error (if any then .any conds else .all conds))
      else if any then
        some (.ok (distinctDocs (cols.map (fun c => c.2.1.index.lookup c.2.2))))
      else (interDocs (cols.map (fun c => c.2.1.index.lookup c.2.2))).map .ok

/-- `query_column` (`:978-1019`): `Some(Ok(rows))` for a `Ready` column, `Some(Err(scan_filter))` for a
    `Building` one, `None` otherwise. The selection is translated even while the column builds, so a
    shape no kind serves is `None` whatever its state. -/
def FalkorDbIndex.queryColumn (x : FalkorDbIndex) (e : EntityType) (label : String) (q : IndexQuery) :
    Option (Except ScanFilter (List Nat)) :=
  match attrsOf q with
  | none => none
  | some [] => none
  | some (first :: rest) =>
    if rest.any (fun k => k != first) then
      match q with
      | .and cs => x.combineColumns e label cs false
      | .or cs => x.combineColumns e label cs true
      | _ => none
    else
      match x.entry e label first with
      | none => none
      | some en =>
        match Selection.of q with
        | none => none
        | some sel =>
          if en.isBuilding then some (.error (ScanFilter.one first sel))
          else some (.ok (en.index.lookup sel))

open FalkorDbIndex

/-- **A query on one attribute, in one line**: on an existing column, for a shape a kind serves
    (`Selection::of`); then rows from a `Ready` column and a `ScanFilter` from a `Building` one. -/
theorem queryColumn_single (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) :
    x.queryColumn e l q = (x.entry e l a).bind (fun en => (Selection.of q).map (fun sel =>
      if en.isBuilding then .error (ScanFilter.one a sel) else .ok (en.index.lookup sel))) := by
  unfold queryColumn
  simp only [hat, List.any_nil, Bool.false_eq_true, ite_false]
  cases x.entry e l a with
  | none => rfl
  | some en =>
    cases Selection.of q with
    | none => rfl
    | some sel => by_cases hb : en.isBuilding = true <;> simp [hb]

/-- **A `Building` column is a timing state, not a gap** (#2386): a shape it serves is answered with a
    `ScanFilter` over the very selection a built column would look up, never with rows and never `None`. -/
theorem queryColumn_building (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) (en : IndexEntry) (hen : x.entry e l a = some en)
    (hb : en.isBuilding = true) :
    x.queryColumn e l q = (Selection.of q).map (fun sel => .error (ScanFilter.one a sel)) := by
  rw [queryColumn_single x e l q a hat, hen]
  simp only [Option.bind_some, hb, ite_true]

/-- ... and a `Ready` column answers the shape from its tree. -/
theorem queryColumn_ready (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) (c : NumericIndex) (st : BuildState)
    (hen : x.entry e l a = some ⟨.numeric c, st⟩) (hb : st = .ready) :
    x.queryColumn e l q = (Selection.of q).map (fun sel => .ok (c.lookupSel sel)) := by
  subst hb
  rw [queryColumn_single x e l q a hat, hen]
  simp only [Option.bind_some, IndexEntry.isBuilding, IndexColumn.lookup]
  cases Selection.of q <;> simp

/-- **What is declined does not depend on the build state**: for a query on one attribute with an existing
    column, the answer exists exactly when a kind serves the shape. A `Building` column therefore never
    turns an unservable shape into a quiet scan, and never turns a servable one into an error. -/
theorem queryColumn_isSome_iff (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) (en : IndexEntry) (hen : x.entry e l a = some en) :
    (x.queryColumn e l q).isSome = (Selection.of q).isSome := by
  rw [queryColumn_single x e l q a hat, hen]
  simp only [Option.bind_some]
  cases Selection.of q <;> simp

/-- A missing column declines a query on one attribute. -/
theorem queryColumn_no_column (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) (hen : x.entry e l a = none) : x.queryColumn e l q = none := by
  rw [queryColumn_single x e l q a hat, hen]; rfl

/-- The three outcomes are exact: `Ok(rows)` only from a `Ready` column, from the query's selection;
    `Err(f)` only from a `Building` one, with the query's attribute and selection. -/
theorem queryColumn_ok_single (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) (r : List Nat) (h : x.queryColumn e l q = some (.ok r)) :
    ∃ en sel, x.entry e l a = some en ∧ en.isBuilding = false ∧ Selection.of q = some sel ∧
      r = en.index.lookup sel := by
  rw [queryColumn_single x e l q a hat] at h
  cases hen : x.entry e l a with
  | none => simp [hen] at h
  | some en =>
    cases hs : Selection.of q with
    | none => simp [hen, hs] at h
    | some sel =>
      by_cases hb : en.isBuilding = true
      · simp [hen, hs, hb] at h
      · have hb' : en.isBuilding = false := by simpa using hb
        simp only [hen, hs, hb', Option.bind_some, Option.map_some, Bool.false_eq_true, ite_false,
          Option.some.injEq, Except.ok.injEq] at h
        exact ⟨en, sel, rfl, hb', rfl, h.symm⟩

theorem queryColumn_error_single (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (a : String)
    (hat : attrsOf q = some [a]) (sel : Selection) (hs : Selection.of q = some sel) (f : ScanFilter)
    (h : x.queryColumn e l q = some (.error f)) :
    ∃ en, x.entry e l a = some en ∧ en.isBuilding = true ∧ f = ScanFilter.one a sel := by
  rw [queryColumn_single x e l q a hat] at h
  cases hen : x.entry e l a with
  | none => simp [hen] at h
  | some en =>
    by_cases hb : en.isBuilding = true
    · simp only [hen, hs, hb, Option.bind_some, Option.map_some, ite_true, Option.some.injEq,
        Except.error.injEq] at h
      exact ⟨en, rfl, hb, h.symm⟩
    · have hb' : en.isBuilding = false := by simpa using hb
      simp [hen, hs, hb'] at h

/-- A `Range` on one attribute whose bounds are each numeric or `NULL` is served by an existing column,
    whatever its build state: no `unsupported_by_native_index` error, only rows or a scan. -/
theorem queryColumn_range_served (x : FalkorDbIndex) (e : EntityType) (l key : String) (mn mx : Option Value)
    (a b : Bool) (en : IndexEntry) (hen : x.entry e l key = some en)
    (hmn : ∀ v, mn = some v → isNumericKind v = true ∨ v = .null)
    (hmx : ∀ v, mx = some v → isNumericKind v = true ∨ v = .null) :
    (x.queryColumn e l (.range key mn mx a b)).isSome = true := by
  rw [queryColumn_isSome_iff x e l _ key (by simp [attrsOf]) en hen]
  have hnb : (mn.toList ++ mx.toList).any (fun v => !isNumericKind v && v != Value.null) = false := by
    simp only [List.any_eq_false, List.mem_append, Option.mem_toList, Option.mem_def]
    rintro v (hv | hv)
    · rcases hmn v hv with h | h <;> simp [h]
    · rcases hmx v hv with h | h <;> simp [h]
  simp only [Selection.of, Selection.ofLeaf, Selection.range, hnb, Bool.false_eq_true, ite_false]
  split <;> simp

/-- **`= null` and a null range bound hold for nothing**: a `Ready` column answers them with no rows
    (#2386), rather than declining a value no column indexes. -/
theorem queryColumn_null_equal (x : FalkorDbIndex) (e : EntityType) (l key : String) (c : NumericIndex)
    (hen : x.entry e l key = some ⟨.numeric c, .ready⟩) :
    x.queryColumn e l (.equal key .null) = some (.ok []) := by
  rw [queryColumn_ready x e l _ key (by simp [attrsOf]) c .ready hen rfl]
  simp [Selection.of, Selection.ofLeaf, equal_null, Selection.empty, NumericIndex.lookupSel, NumericIndex.lookup,
    NumericIndex.empty, tRange_empty]

/-- Shapes no kind serves are declined, whatever the state: a non-numeric `Equal` value (a string, a
    temporal, a list), a `Point`, and an empty composite. -/
theorem queryColumn_declined (x : FalkorDbIndex) (e : EntityType) (l : String) :
    (∀ key, x.queryColumn e l (.point key) = none) ∧
    x.queryColumn e l (.or []) = none ∧ x.queryColumn e l (.and []) = none ∧
    (∀ key v, v ≠ Value.null → isNumericKind v = false → x.queryColumn e l (.equal key v) = none) := by
  refine ⟨fun key => ?_, by simp [queryColumn, attrsOf], by simp [queryColumn, attrsOf], fun key v hn he => ?_⟩
  · rw [queryColumn_single x e l _ key (by simp [attrsOf])]
    cases x.entry e l key <;> simp [Selection.of, Selection.ofLeaf]
  · rw [queryColumn_single x e l _ key (by simp [attrsOf])]
    cases x.entry e l key <;> simp [Selection.of, Selection.ofLeaf, equal_other v he hn]

/-! ### Several attributes: `combine_columns` -/

/-- Every column must exist and every share translate: otherwise `combine_columns` declines, whatever the
    other columns' state. (A missing first attribute, for the first group.) -/
theorem combine_declines_of_group (x : FalkorDbIndex) (e : EntityType) (l : String) (cs : List IndexQuery)
    (any : Bool) (h : groupList any [] cs = none) : x.combineColumns e l cs any = none := by
  simp [FalkorDbIndex.combineColumns, h]

theorem distinctDocs_mem (streams : List (List Nat)) (d : Nat) :
    d ∈ distinctDocs streams ↔ ∃ s ∈ streams, d ∈ s := by
  simp [distinctDocs, List.mem_eraseDups, List.mem_flatten]

theorem interDocs_foldl_mem (rs : List (List Nat)) : ∀ (r : List Nat) (d : Nat),
    d ∈ rs.foldl (fun seen rows => if seen.isEmpty then seen else rows.filter (fun d => seen.contains d)) r ↔
      d ∈ r ∧ ∀ s ∈ rs, d ∈ s := by
  induction rs with
  | nil => intro r d; simp
  | cons s rs ih =>
    intro r d
    simp only [List.foldl_cons]
    rw [ih]
    by_cases he : r.isEmpty = true
    · have : r = [] := List.isEmpty_iff.1 he
      subst this; simp
    · simp only [he, Bool.false_eq_true, ite_false, List.mem_filter, List.contains_eq_any_beq,
        List.any_eq_true, beq_iff_eq, List.mem_cons, forall_eq_or_imp]
      constructor
      · rintro ⟨⟨h1, ⟨x, hx, rfl⟩⟩, h3⟩; exact ⟨hx, h1, h3⟩
      · rintro ⟨h1, h2, h3⟩; exact ⟨⟨h2, ⟨d, h1, rfl⟩⟩, h3⟩

theorem interDocs_mem (rows : List (List Nat)) (res : List Nat) (h : interDocs rows = some res) (d : Nat) :
    d ∈ res ↔ rows ≠ [] ∧ ∀ s ∈ rows, d ∈ s := by
  cases rows with
  | nil => simp [interDocs] at h
  | cons r rs =>
    simp only [interDocs, Option.some.injEq] at h
    subst h
    rw [interDocs_foldl_mem]
    simp [List.forall_mem_cons, and_assoc]

/-- **An `Or` over several attributes, every column `Ready`, returns the docs some column's share
    selects**, each once (`DistinctIter`). -/
theorem combine_or_ok (x : FalkorDbIndex) (e : EntityType) (l : String) (cs : List IndexQuery)
    (perAttr : List (String × List IndexQuery)) (hg : groupList true [] cs = some perAttr)
    (cols : List Col)
    (hc : perAttr.mapM (fun p => (x.entry e l p.1).bind (fun en =>
        (Selection.disjunction p.2).map (fun sel => ((p.1, en, sel) : Col)))) = some cols)
    (hready : cols.any (fun c => c.2.1.isBuilding) = false) (r : List Nat)
    (h : x.combineColumns e l cs true = some (.ok r)) (d : Nat) :
    d ∈ r ↔ ∃ c ∈ cols, d ∈ c.2.1.index.lookup c.2.2 := by
  simp only [FalkorDbIndex.combineColumns, hg, hc, hready, Bool.false_eq_true, ite_false, ite_true,
    Option.some.injEq, Except.ok.injEq] at h
  subst h
  rw [distinctDocs_mem]
  constructor
  · rintro ⟨s, hs, hd⟩
    obtain ⟨c, hc', rfl⟩ := List.mem_map.1 hs
    exact ⟨c, hc', hd⟩
  · rintro ⟨c, hc', hd⟩
    exact ⟨_, List.mem_map.2 ⟨c, hc', rfl⟩, hd⟩

/-- An `And` over several attributes, every column `Ready`, returns the docs every column's share selects. -/
theorem combine_and_ok (x : FalkorDbIndex) (e : EntityType) (l : String) (cs : List IndexQuery)
    (perAttr : List (String × List IndexQuery)) (hg : groupList false [] cs = some perAttr)
    (cols : List Col)
    (hc : perAttr.mapM (fun p => (x.entry e l p.1).bind (fun en =>
        (Selection.conjunction p.2).map (fun sel => ((p.1, en, sel) : Col)))) = some cols)
    (hready : cols.any (fun c => c.2.1.isBuilding) = false) (r : List Nat)
    (h : x.combineColumns e l cs false = some (.ok r)) (d : Nat) :
    d ∈ r ↔ cols ≠ [] ∧ ∀ c ∈ cols, d ∈ c.2.1.index.lookup c.2.2 := by
  simp only [FalkorDbIndex.combineColumns, hg, hc, hready, Bool.false_eq_true, ite_false,
    Bool.false_eq_true] at h
  cases hi : interDocs (cols.map (fun c => c.2.1.index.lookup c.2.2)) with
  | none => simp [hi] at h
  | some res =>
    simp only [hi, Option.map_some, Option.some.injEq, Except.ok.injEq] at h
    subst h
    rw [interDocs_mem _ _ hi]
    constructor
    · rintro ⟨hne, hall⟩
      exact ⟨fun h => hne (by simp [h]), fun c hc' => hall _ (List.mem_map.2 ⟨c, hc', rfl⟩)⟩
    · rintro ⟨hne, hall⟩
      refine ⟨fun h => hne (List.map_eq_nil_iff.1 h), fun s hs => ?_⟩
      obtain ⟨c, hc', rfl⟩ := List.mem_map.1 hs
      exact hall c hc'


/-- **One `Building` column makes the whole multi-attribute query a scan** (`Err`, a `ScanFilter` holding
    every attribute's selection), never a half-built answer. -/
theorem combine_building (x : FalkorDbIndex) (e : EntityType) (l : String) (cs : List IndexQuery) (any : Bool)
    (perAttr : List (String × List IndexQuery)) (hg : groupList any [] cs = some perAttr)
    (cols : List Col)
    (hc : perAttr.mapM (fun p => (x.entry e l p.1).bind (fun en =>
        (if any then Selection.disjunction p.2 else Selection.conjunction p.2).map
          (fun sel => ((p.1, en, sel) : Col)))) = some cols)
    (hb : cols.any (fun c => c.2.1.isBuilding) = true) :
    x.combineColumns e l cs any =
      some (.error (let conds := cols.map (fun c => (c.1, c.2.2)); if any then .any conds else .all conds)) := by
  simp [FalkorDbIndex.combineColumns, hg, hc, hb]

/-! ## `equal_candidates` (`falkordb_index.rs:1159-1177`) -/

/-- `EqualCandidates` (`:290-294`): the column a lookup was made against, the tree and the key it reads.
    The Rust holds `&'g IndexColumn`; its address (`id`, `:303-305`) is determined by the column's slot
    `(entity, label, attr)` in one index (distinct entries are distinct objects). -/
structure EqualCandidates where
  entity : EntityType
  label : String
  attr : String
  column : IndexColumn
  kind : KeyKind
  key : Nat

/-- `EqualCandidates::rows` (`:299-301`): every entity holding the value, and possibly others. -/
def EqualCandidates.rows (c : EqualCandidates) : List Nat := c.column.rowsAt c.kind c.key

/-- `EqualCandidates::id` (`:306-308`): the column's address, the tree and the key. -/
def EqualCandidates.id (c : EqualCandidates) : (EntityType × String × String) × KeyKind × Nat :=
  ((c.entity, c.label, c.attr), c.kind, c.key)

/-- `FalkorDbIndex::equal_candidates` (`:1159-1177`). -/
def FalkorDbIndex.equalCandidates (x : FalkorDbIndex) (e : EntityType) (label attr : String) (v : Value) :
    Option EqualCandidates :=
  match x.entry e label attr with
  | none => none
  | some en =>
    if en.isBuilding then none
    else (en.index.encodeValue v).map (fun kk => ⟨e, label, attr, en.index, kk.1, kk.2⟩)

/-- **A lookup is made exactly when its answer covers every entity holding the value**: there is a
    column, it is not `Building` (it has not seen every entity), and its kind indexes the value as a
    scalar (a string, a list, NaN: no). `None` is never "no entities". -/
theorem equalCandidates_some (x : FalkorDbIndex) (e : EntityType) (l a : String) (v : Value) (c : EqualCandidates) :
    x.equalCandidates e l a v = some c ↔
      ∃ en, x.entry e l a = some en ∧ en.isBuilding = false ∧ en.index.encodeValue v = some (c.kind, c.key) ∧
        c = ⟨e, l, a, en.index, c.kind, c.key⟩ := by
  unfold FalkorDbIndex.equalCandidates
  cases hen : x.entry e l a with
  | none => simp
  | some en =>
    by_cases hb : en.isBuilding = true
    · simp [hb]
    · have hb' : en.isBuilding = false := by simpa using hb
      simp only [hb', Bool.false_eq_true, ite_false, Option.map_eq_some_iff, Option.some.injEq]
      constructor
      · rintro ⟨k, hk, rfl⟩; exact ⟨en, rfl, hb', hk, rfl⟩
      · rintro ⟨en', he', _, hk, hc⟩
        subst he'
        exact ⟨(c.kind, c.key), hk, hc.symm⟩

theorem equalCandidates_none (x : FalkorDbIndex) (e : EntityType) (l a : String) (v : Value) :
    x.equalCandidates e l a v = none ↔
      x.entry e l a = none ∨ (∃ en, x.entry e l a = some en ∧ en.isBuilding = true) ∨
        (∃ en, x.entry e l a = some en ∧ en.index.encodeValue v = none) := by
  unfold FalkorDbIndex.equalCandidates
  cases hen : x.entry e l a with
  | none => simp
  | some en =>
    by_cases hb : en.isBuilding = true
    · simp [hb]
    · have hb' : en.isBuilding = false := by simpa using hb
      simp [hb', hb]

/-- **Complete**: on a `Ready` column built from `entries`, every entity whose value equals a guarded
    numeric `b` is among the candidates. (The candidates may include others, `rows_mem`.) -/
theorem equalCandidates_complete (x : FalkorDbIndex) (e : EntityType) (l a : String) (entries : List (Value × Nat))
    (hcol : x.readyNumeric e l a = some (NumericIndex.fromEntries entries))
    (hwf : ∀ p ∈ entries, p.1.WF ∧ NotBool p.1 ∧ p.2 < W) (b : Value) (hb : GoodBound b)
    (id : Nat) (hid : id < W) (v : Value) (hv : (v, id) ∈ entries) (heq : cmpPred .eq v b = some true) :
    ∃ c, x.equalCandidates e l a b = some c ∧ id ∈ c.rows := by
  obtain ⟨kb, hkb, _⟩ := goodBound_enc b hb
  have hnum : isNumericKind b = true := by
    cases b <;> simp [GoodBound] at hb <;> simp [isNumericKind]
  unfold FalkorDbIndex.readyNumeric at hcol
  cases hen : x.entry e l a with
  | none => simp [hen] at hcol
  | some en =>
    obtain ⟨⟨col⟩, st⟩ := en
    simp only [hen, Option.bind_some] at hcol
    by_cases hbld : IndexEntry.isBuilding ⟨.numeric col, st⟩ = true
    · simp [hbld] at hcol
    · have hb' : IndexEntry.isBuilding ⟨.numeric col, st⟩ = false := by simpa using hbld
      simp only [hb', Bool.false_eq_true, ite_false, Option.some.injEq, IndexColumn.ix] at hcol
      subst hcol
      refine ⟨⟨e, l, a, .numeric (NumericIndex.fromEntries entries), .scalar, kb⟩, ?_, ?_⟩
      · simp [FalkorDbIndex.equalCandidates, hen, hb', IndexColumn.encodeValue, hnum, hkb]
      · have hp := (point_agrees entries hwf b hb id hid).2 ⟨v, hv, heq⟩
        simp only [point_eq, hkb] at hp
        exact hp

/-- **Precise**: a candidate lookup's rows are exactly the ids whose value shares the key (so a column
    keying `1`, `1.0` and `true` alike returns all three, and the caller re-checks each). -/
theorem rows_mem (entries : List (Value × Nat)) (hids : ∀ p ∈ entries, p.2 < W) (key id : Nat) (hid : id < W) :
    id ∈ (EqualCandidates.rows ⟨.node, "", "", .numeric (NumericIndex.fromEntries entries), .scalar, key⟩) ↔
      ∃ v, (v, id) ∈ entries ∧ encodeNumeric v = some key := by
  have := lookup_fromEntries entries hids (.point key) id hid
  simp only [EqualCandidates.rows, IndexColumn.rowsAt]
  rw [this]
  constructor
  · rintro ⟨v, hm, hf⟩
    cases he : encodeNumeric v with
    | none => simp [KeySet.matchesValue, he] at hf
    | some k => simp [KeySet.matchesValue, he, KeySet.contains] at hf; exact ⟨v, hm, by rw [hf]; exact he⟩
  · rintro ⟨v, hm, he⟩; exact ⟨v, hm, by simp [KeySet.matchesValue, he, KeySet.contains]⟩

/-- An array-tree candidate (`rows_at(Array, key)`) is the lists holding `key`: it reads the array tree and
    never the scalar one. -/
theorem rowsAt_array (ix : NumericIndex) (key : Nat) :
    (IndexColumn.numeric ix).rowsAt .array key = tRange ix.arrayTree key key := rfl

end NI
