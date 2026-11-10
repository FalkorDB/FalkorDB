import FalkorNativeIndex.Index
set_option linter.unusedSimpArgs false

/-!
# `query_numeric` and `equal_candidates` (`falkordb_index.rs:633-717`), `ScanFilter` (`:266-277`)

Since #3547 / #2386 the native index is the ONLY Range index under `index-falkordb` (there is no
RediSearch to fall through to), and a read has **three** outcomes (`query_numeric`'s doc):

* `Some(Ok(rows))`: served from a `Ready` column;
* `Some(Err(scan_filter))`: a shape the column serves, on a column still `Building`: the caller scans
  and keeps what the `ScanFilter` matches (a timing state, never an error);
* `None`: any other shape, or no such column. The caller turns it into the error
  `unsupported_by_native_index`.

Modelled as `Option (Except ScanFilter (List Nat))` (`Except.ok` = `Ok`, `Except.error` = `Err`).
-/

namespace NI

/-- `ScanFilter` (`falkordb_index.rs:266-269`): the attribute and the keys the built column would
    look up. -/
structure ScanFilter where
  attr : String
  keys : KeySet

/-- `ScanFilter::matches` (`:272-277`): whether the built column answers the query with an entity
    whose attribute is `value`. -/
def ScanFilter.matches (f : ScanFilter) (v : Value) : Bool := f.keys.matchesValue v

namespace FalkorDbIndex

/-- The column a query is routed to (`query_numeric`'s `let key = match query`, `:645-670`): an
    `Equal`/`Range` only when its value/bounds are `Null` or encode (`= null` holds for nothing, so
    it is answered with no rows; a bound that is neither is a value no column indexes), an `IN` list
    always (the column declines), an `OR` by its first member. -/
def queryKey : IndexQuery → Option String
  | .equal key v => if v == Value.null || (encodeNumeric v).isSome then some key else none
  | .range key min max _ _ =>
    if (min.all (fun v => v == Value.null || (encodeNumeric v).isSome)) &&
        (max.all (fun v => v == Value.null || (encodeNumeric v).isSome))
    then some key else none
  | .inList key _ => some key
  | .or cs => match cs with
    | .equal key _ :: _ => some key
    | .inList key _ :: _ => some key
    | _ => none
  | .other => none

/-- `query_numeric` (`:633-695`): `Some(Ok(rows))` for a `Ready` column, `Some(Err(scan_filter))` for
    a `Building` one, `None` otherwise. The keys are translated even while the column builds, so a
    shape the column declines is `None` whatever its state. -/
def queryNumeric (x : FalkorDbIndex) (e : EntityType) (label : String) (q : IndexQuery) :
    Option (Except ScanFilter (List Nat)) :=
  match queryKey q with
  | none => none
  | some key =>
    match x.entry e label key with
    | none => none
    | some entry =>
      match NumericIndex.keysOf q with
      | none => none
      | some keys =>
        if entry.isBuilding then some (.error ⟨key, keys⟩)
        else some (.ok (entry.index.ix.lookup keys))

end FalkorDbIndex

open FalkorDbIndex

/-- **`query_numeric`, in one line**: routed by `queryKey`, on an existing column, for a shape the
    column serves (`keysOf`); then rows from a `Ready` column and a `ScanFilter` from a `Building`
    one. -/
theorem queryNumeric_spec (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) :
    x.queryNumeric e l q =
      (queryKey q).bind (fun key => (x.entry e l key).bind (fun en =>
        (NumericIndex.keysOf q).map (fun ks =>
          if en.isBuilding then .error ⟨key, ks⟩ else .ok (en.index.ix.lookup ks)))) := by
  unfold queryNumeric
  cases queryKey q with
  | none => rfl
  | some key =>
    simp only [Option.bind_some]
    cases x.entry e l key with
    | none => rfl
    | some en =>
      cases NumericIndex.keysOf q with
      | none => rfl
      | some ks => by_cases hb : en.isBuilding = true <;> simp [hb]

/-- **A `Building` column is a timing state, not a gap** (#2386): a shape it serves is answered with a
    `ScanFilter` over the very keys a built column would look up, never with rows and never `None`. -/
theorem queryNumeric_building (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (key : String)
    (hk : queryKey q = some key) (en : IndexEntry) (hen : x.entry e l key = some en)
    (hb : en.isBuilding = true) :
    x.queryNumeric e l q = (NumericIndex.keysOf q).map (fun ks => .error ⟨key, ks⟩) := by
  rw [queryNumeric_spec, hk]
  simp only [hen, Option.bind_some, hb, if_true]
  try (cases NumericIndex.keysOf q <;> rfl)

/-- ... and a `Ready` column answers the shape from its tree. -/
theorem queryNumeric_ready (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (key : String)
    (hk : queryKey q = some key) (c : NumericIndex) (st : BuildState) (hen : x.entry e l key = some ⟨.numeric c, st⟩)
    (hb : st = .ready) : x.queryNumeric e l q = (c.query q).map .ok := by
  subst hb
  rw [queryNumeric_spec, hk]
  simp only [hen, Option.bind_some, IndexEntry.isBuilding, IndexColumn.ix, NumericIndex.query]
  cases NumericIndex.keysOf q <;> rfl

/-- **What is declined does not depend on the build state**: for a routed query on an existing column,
    the answer exists exactly when the column serves the shape. A `Building` column therefore never
    turns an unservable shape into a quiet scan (the doc's "must stay `None`"), and never turns a
    servable one into an error. -/
theorem queryNumeric_isSome_iff (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (key : String)
    (hk : queryKey q = some key) (en : IndexEntry) (hen : x.entry e l key = some en) :
    (x.queryNumeric e l q).isSome = (NumericIndex.keysOf q).isSome := by
  rw [queryNumeric_spec, hk]
  simp only [hen, Option.bind_some]
  cases NumericIndex.keysOf q <;> simp

/-- The three outcomes are exhaustive and exact: `Ok(rows)` only from a `Ready` column, from the keys
    of the query; `Err(f)` only from a `Building` one, with the keys and attribute of the query. -/
theorem queryNumeric_ok (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (r : List Nat)
    (h : x.queryNumeric e l q = some (.ok r)) :
    ∃ key en ks, queryKey q = some key ∧ x.entry e l key = some en ∧ en.isBuilding = false ∧
      NumericIndex.keysOf q = some ks ∧ r = en.index.ix.lookup ks := by
  rw [queryNumeric_spec] at h
  cases hk : queryKey q with
  | none => simp [hk] at h
  | some key =>
    cases hen : x.entry e l key with
    | none => simp [hk, hen] at h
    | some en =>
      cases hks : NumericIndex.keysOf q with
      | none => simp [hk, hen, hks] at h
      | some ks =>
        by_cases hb : en.isBuilding = true
        · simp [hk, hen, hks, hb] at h
        · have hb' : en.isBuilding = false := by simpa using hb
          simp only [hk, hen, hks, hb', Option.bind_some, Option.map_some, Bool.false_eq_true, if_false,
            Option.some.injEq, Except.ok.injEq] at h
          exact ⟨key, en, ks, rfl, hen, hb', rfl, h.symm⟩

theorem queryNumeric_error (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (f : ScanFilter)
    (h : x.queryNumeric e l q = some (.error f)) :
    ∃ key en, queryKey q = some key ∧ x.entry e l key = some en ∧ en.isBuilding = true ∧
      f.attr = key ∧ NumericIndex.keysOf q = some f.keys := by
  rw [queryNumeric_spec] at h
  cases hk : queryKey q with
  | none => simp [hk] at h
  | some key =>
    cases hen : x.entry e l key with
    | none => simp [hk, hen] at h
    | some en =>
      cases hks : NumericIndex.keysOf q with
      | none => simp [hk, hen, hks] at h
      | some ks =>
        by_cases hb : en.isBuilding = true
        · simp only [hk, hen, hks, hb, Option.bind_some, Option.map_some, if_true, Option.some.injEq,
            Except.error.injEq] at h
          subst h
          exact ⟨key, en, rfl, hen, hb, rfl, rfl⟩
        · have hb' : en.isBuilding = false := by simpa using hb
          simp [hk, hen, hks, hb'] at h

/-- The routing key of a `Range` whose bounds are each `Null` or encodable, and of an `Equal`. -/
theorem queryKey_range (key : String) (mn mx : Option Value) (a b : Bool)
    (hmn : ∀ v, mn = some v → v = .null ∨ (encodeNumeric v).isSome = true)
    (hmx : ∀ v, mx = some v → v = .null ∨ (encodeNumeric v).isSome = true) :
    queryKey (.range key mn mx a b) = some key := by
  have h1 : mn.all (fun v => v == Value.null || (encodeNumeric v).isSome) = true := by
    cases mn with
    | none => rfl
    | some v => simp only [Option.all_some]; rcases hmn v rfl with h | h <;> simp [h]
  have h2 : mx.all (fun v => v == Value.null || (encodeNumeric v).isSome) = true := by
    cases mx with
    | none => rfl
    | some v => simp only [Option.all_some]; rcases hmx v rfl with h | h <;> simp [h]
  simp [queryKey, h1, h2]

/-- **A `Range` is never declined** on an existing column, whatever its build state, when every bound
    is `Null` or encodes: no `unsupported_by_native_index` error and no fallthrough, only rows or a
    scan. -/
theorem queryNumeric_range_served (x : FalkorDbIndex) (e : EntityType) (l key : String) (mn mx : Option Value)
    (a b : Bool) (en : IndexEntry) (hen : x.entry e l key = some en)
    (hmn : ∀ v, mn = some v → v = .null ∨ (encodeNumeric v).isSome = true)
    (hmx : ∀ v, mx = some v → v = .null ∨ (encodeNumeric v).isSome = true) :
    (x.queryNumeric e l (.range key mn mx a b)).isSome = true := by
  have hk := queryKey_range key mn mx a b hmn hmx
  rw [queryNumeric_isSome_iff x e l _ key hk en hen]
  rfl

/-- A `Ready` column found by `readyNumeric` answers a routed query with its tree's rows. -/
theorem queryNumeric_of_ready (x : FalkorDbIndex) (e : EntityType) (l : String) (q : IndexQuery) (key : String)
    (hk : queryKey q = some key) (c : NumericIndex) (hc : x.readyNumeric e l key = some c) :
    x.queryNumeric e l q = (c.query q).map .ok := by
  unfold FalkorDbIndex.readyNumeric at hc
  cases hen : x.entry e l key with
  | none => simp [hen] at hc
  | some en =>
    simp only [hen, Option.bind_some] at hc
    by_cases hb : en.isBuilding = true
    · simp [hb] at hc
    · have hb' : en.isBuilding = false := by simpa using hb
      simp only [hb', Bool.false_eq_true, if_false, Option.some.injEq] at hc
      rw [queryNumeric_spec, hk]
      simp only [hen, Option.bind_some, hb', Bool.false_eq_true, if_false, hc, NumericIndex.query]
      cases NumericIndex.keysOf q <;> rfl

/-- **`= null` and a null range bound hold for nothing**: a `Ready` column answers them with no rows
    (#2386), rather than declining a value no column indexes. -/
theorem queryNumeric_null_equal (x : FalkorDbIndex) (e : EntityType) (l key : String) (c : NumericIndex)
    (hen : x.entry e l key = some ⟨.numeric c, .ready⟩) :
    x.queryNumeric e l (.equal key .null) = some (.ok []) := by
  rw [queryNumeric_ready x e l _ key (by simp [queryKey]) c .ready hen rfl]
  simp [NumericIndex.query, NumericIndex.keysOf, pointKeys, encodeNumeric, NumericIndex.lookup,
    NumericIndex.empty, tRange_empty]

/-- Shapes the numeric column does not index are declined (`None`), whatever the state: a
    non-numeric `Equal` value (a string, a temporal), and anything that is no leaf/`IN`/`OR`. -/
theorem queryNumeric_declined (x : FalkorDbIndex) (e : EntityType) (l : String) :
    x.queryNumeric e l .other = none ∧
    (∀ key v, v ≠ Value.null → encodeNumeric v = none → x.queryNumeric e l (.equal key v) = none) := by
  refine ⟨by simp [queryNumeric, queryKey], fun key v hn he => ?_⟩
  have : queryKey (.equal key v) = none := by simp [queryKey, he, hn]
  simp [queryNumeric, this]

/-! ## `equal_candidates` (`falkordb_index.rs:697-715`) -/

/-- `EqualCandidates` (`:155-158`): the column a lookup was made against and the key it reads. The
    Rust holds `&'g IndexColumn`; its address (`id`, `:179`) is determined by the column's slot
    `(entity, label, attr)` in one index (distinct entries are distinct objects). -/
structure EqualCandidates where
  entity : EntityType
  label : String
  attr : String
  column : IndexColumn
  key : Nat

/-- `EqualCandidates::rows` (`:172-174`): every entity holding the value, and possibly others. -/
def EqualCandidates.rows (c : EqualCandidates) : List Nat := c.column.rowsAt c.key

/-- `EqualCandidates::id` (`:179-181`): the column's address and the key. -/
def EqualCandidates.id (c : EqualCandidates) : (EntityType × String × String) × Nat :=
  ((c.entity, c.label, c.attr), c.key)

/-- `FalkorDbIndex::equal_candidates` (`:697-715`). -/
def FalkorDbIndex.equalCandidates (x : FalkorDbIndex) (e : EntityType) (label attr : String) (v : Value) :
    Option EqualCandidates :=
  match x.entry e label attr with
  | none => none
  | some en =>
    if en.isBuilding then none
    else (en.index.encodeValue v).map (fun key => ⟨e, label, attr, en.index, key⟩)

/-- **A lookup is made exactly when its answer covers every entity holding the value**: there is a
    column, it is not `Building` (it has not seen every entity), and its kind indexes the value (a
    string in a numeric column, NaN: no). `None` is never "no entities". -/
theorem equalCandidates_some (x : FalkorDbIndex) (e : EntityType) (l a : String) (v : Value) (c : EqualCandidates) :
    x.equalCandidates e l a v = some c ↔
      ∃ en, x.entry e l a = some en ∧ en.isBuilding = false ∧ en.index.encodeValue v = some c.key ∧
        c = ⟨e, l, a, en.index, c.key⟩ := by
  unfold FalkorDbIndex.equalCandidates
  cases hen : x.entry e l a with
  | none => simp
  | some en =>
    by_cases hb : en.isBuilding = true
    · simp [hb]
    · have hb' : en.isBuilding = false := by simpa using hb
      simp only [hb', Bool.false_eq_true, if_false, Option.map_eq_some_iff, Option.some.injEq]
      constructor
      · rintro ⟨k, hk, rfl⟩; exact ⟨en, rfl, hb', hk, rfl⟩
      · rintro ⟨en', he', _, hk, hc⟩
        subst he'
        exact ⟨c.key, hk, hc.symm⟩

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
  unfold FalkorDbIndex.readyNumeric at hcol
  cases hen : x.entry e l a with
  | none => simp [hen] at hcol
  | some en =>
    obtain ⟨⟨col⟩, st⟩ := en
    simp only [hen, Option.bind_some] at hcol
    by_cases hbld : IndexEntry.isBuilding ⟨.numeric col, st⟩ = true
    · simp [hbld] at hcol
    · have hb' : IndexEntry.isBuilding ⟨.numeric col, st⟩ = false := by simpa using hbld
      simp only [hb', Bool.false_eq_true, if_false, Option.some.injEq, IndexColumn.ix] at hcol
      subst hcol
      refine ⟨⟨e, l, a, .numeric (NumericIndex.fromEntries entries), kb⟩, ?_, ?_⟩
      · simp [FalkorDbIndex.equalCandidates, hen, hb', IndexColumn.encodeValue, hkb]
      · have hp := (point_agrees entries hwf b hb id hid).2 ⟨v, hv, heq⟩
        simp only [point_eq, hkb] at hp
        exact hp

/-- **Precise**: a candidate lookup's rows are exactly the ids whose value shares the key (so a column
    keying `1`, `1.0` and `true` alike returns all three, and the caller re-checks each). -/
theorem rows_mem (entries : List (Value × Nat)) (hids : ∀ p ∈ entries, p.2 < W) (key id : Nat) (hid : id < W) :
    id ∈ (EqualCandidates.rows ⟨.node, "", "", .numeric (NumericIndex.fromEntries entries), key⟩) ↔
      ∃ v, (v, id) ∈ entries ∧ encodeNumeric v = some key := by
  have := lookup_fromEntries entries hids (.point key) id hid
  simp only [EqualCandidates.rows, IndexColumn.rowsAt, NumericIndex.pointKey]
  have h2 : id ∈ tRange (NumericIndex.fromEntries entries).tree key key ↔
      id ∈ (NumericIndex.fromEntries entries).lookup (.point key) := Iff.rfl
  rw [h2, this]
  constructor
  · rintro ⟨v, hm, hf⟩
    cases he : encodeNumeric v with
    | none => simp [KeySet.matchesValue, he] at hf
    | some k => simp [KeySet.matchesValue, he, KeySet.contains] at hf; exact ⟨v, hm, by rw [hf]; exact he⟩
  · rintro ⟨v, hm, he⟩; exact ⟨v, hm, by simp [KeySet.matchesValue, he, KeySet.contains]⟩

end NI
