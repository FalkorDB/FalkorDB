import FalkorIndexLayer.Meta
import FalkorIndexLayer.Model
/-
# `Index` (`graph/src/index/mod.rs:876-2226`): metadata, RS spec lifecycle

The RediSearch spec is abstract: `spec : Option Nat` is the handle identity
(`None` = the former null), `rs` the list of fields registered on it. The FFI
calls that can fail are parameters (`langOk`, `tieredOk`). `Slots` and their
theorems come from `Model.lean`/`Maintenance.lean`.
-/
namespace IndexLayer.Meta
open IndexLayer (Slots)

/-- A field registered on the RS spec by `RediSearch_CreateField`. -/
inductive RsKind
  | noneTag                          -- `NONE_INDEXABLE_FIELDS` (TAG, sep `\x01`, case-sensitive)
  | rangeMain                        -- NUMERIC|GEO|TAG, sep `\x01`, case-sensitive
  | numArr                           -- NUMERIC
  | strArr                           -- TAG, sep `\x01`, case-sensitive
  | text (weight : Nat) (nostem phonetic : Bool)
  | vector (metric : Option String)  -- `none` = dimension 0, params not set
  deriving DecidableEq, Repr

structure RsReg where
  name : String
  kind : RsKind
  deriving DecidableEq, Repr

structure Idx where
  id : Nat
  spec : Option Nat
  rs : List RsReg
  fields : AMap (List Field)
  order : List String
  slots : Slots
  progress : Nat
  total : Nat
  language : Option String
  stopwords : Option (List String)

namespace Idx

/-- `Default for Index` (`mod.rs:1014`). -/
def default (id : Nat) : Idx :=
  { id, spec := none, rs := [], fields := [], order := [], slots := ⟨id, 0, 0⟩,
    progress := 0, total := 0, language := none, stopwords := none }

/-- `Index::id` (`mod.rs:1040`). -/
def idOf (x : Idx) : Nat := x.id
/-- `Index::bump_id` (`mod.rs:1049`), `g'` the next value of the static counter. -/
def bumpId (x : Idx) (g' : Nat) : Idx := { x with id := g', slots := x.slots.bump g' }
/-- `Index::clone_for_update` (`mod.rs:1068`): every component copied; `spec`
and `pending_slots` are the *same* `Arc`s (shared, not copied). -/
def cloneForUpdate (x : Idx) : Idx :=
  { id := x.id, spec := x.spec, rs := x.rs, fields := x.fields, order := x.order, slots := x.slots,
    progress := x.progress, total := x.total, language := x.language, stopwords := x.stopwords }
/-- `has_rs_index` (`mod.rs:1088`). -/
def hasRsIndex (x : Idx) : Bool := x.spec.isSome
/-- `rs_ptr` (`mod.rs:1095`): `none` is the null pointer. -/
def rsPtr (x : Idx) : Option Nat := x.spec

def hasNul (s : String) : Bool := s.toList.contains (Char.ofNat 0)

/-- `create_rs_index` (`mod.rs:1101`). Error order as in Rust: stopwords
`CString::new` (1108), language `CString::new`/unsupported (1129-1133), label
`CString::new` (1140). On success the spec is fresh and carries only
`NONE_INDEXABLE_FIELDS` (1155-1163). -/
def createRsIndex (x : Idx) (label : String) (stop : Option (List String)) (lang : Option String)
    (langOk : String → Bool) (fresh : Nat) : Except String Idx :=
  if (stop.getD []).any hasNul then .error "nul" else
  match lang with
  | some l => if hasNul l then .error "nul" else if !langOk l then .error s!"Language is not supported: {l}"
              else if hasNul label then .error "nul"
              else .ok { x with spec := some fresh, rs := [⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩] }
  | none => if hasNul label then .error "nul"
            else .ok { x with spec := some fresh, rs := [⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩] }

/-- The metric match of `register_fields` (`mod.rs:1274-1287`). -/
def metricOf (sim : Option String) : Except String String :=
  match sim.getD "euclidean" with
  | "euclidean" => .ok "L2"
  | "ip" => .ok "IP"
  | "cosine" => .ok "COSINE"
  | o => .error s!"Unknown similarity function '{o}', expected 'euclidean', 'ip', or 'cosine'"

/-- One iteration of `register_fields` (`mod.rs:1193-1345`). `tieredOk` is
the result of `VecSimTieredParams_Init` + `VectorFieldSetParams` (FFI). -/
def registerOne (fo : Option TextOpts) (tieredOk : Bool) (f : Field) : Except String (List RsReg) :=
  match f.ty with
  | .range =>
    .ok ([⟨f.name, .rangeMain⟩] ++ (f.numArr.map (fun n => [⟨n, .numArr⟩])).getD [] ++
         (f.strArr.map (fun n => [⟨n, .strArr⟩])).getD [])
  | .fulltext =>
    let eo := fo.or f.options
    let w := (eo.bind (·.weight)).getD 1
    let ns := (eo.bind (·.nostem)).getD false
    let ph := (eo.bind (·.phonetic)).any (· ≠ "")
    .ok [⟨f.name, .text w ns ph⟩]
  | .vector =>
    match f.vopts with
    | some v =>
      if v.dimension > 0 then
        match metricOf v.sim with
        | .ok m => if tieredOk then .ok [⟨f.name, .vector (some m)⟩] else .error "failed to configure"
        | .error e => .error e
      else .ok [⟨f.name, .vector none⟩]
    | none => .ok [⟨f.name, .vector none⟩]

def registerAll (fo : Option TextOpts) (tieredOk : Bool) : List Field → Except String (List RsReg)
  | [] => .ok []
  | f :: fs => match registerOne fo tieredOk f with
    | .ok r => match registerAll fo tieredOk fs with
      | .ok rs => .ok (r ++ rs)
      | .error e => .error e
    | .error e => .error e

def allFieldsOf (m : AMap (List Field)) : List Field := m.flatMap (·.2)

/-- `register_fields` (`mod.rs:1181`), applied to the spec. -/
def registerFields (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) (tieredOk : Bool) :
    Except String Idx :=
  match registerAll fo tieredOk (allFieldsOf m) with
  | .ok r => .ok { x with rs := x.rs ++ r }
  | .error e => .error e

/-! Field-map operations (`mod.rs:1955-2064`). -/
def getFields (x : Idx) (a : String) : Option (List Field) := x.fields.get a
def containsField (x : Idx) (a : String) : Bool := (x.fields.get a).isSome
def hasFieldWithType (x : Idx) (a : String) (t : IType) : Bool :=
  (x.fields.get a).any (·.any (·.ty == t))
def addFieldToExisting (x : Idx) (a : String) (f : Field) : Idx :=
  if (x.fields.get a).isSome then { x with fields := x.fields.modify a (· ++ [f]) } else x
def insertField (x : Idx) (a : String) (f : Field) : Idx :=
  { x with order := if (x.fields.get a).isSome then x.order else x.order ++ [a],
           fields := x.fields.upd a [f] }
def removeField (x : Idx) (a : String) : Bool × Idx :=
  if (x.fields.get a).isSome then (true, { x with fields := x.fields.erase a, order := x.order.filter (· != a) })
  else (false, x)
def retainFields (x : Idx) (a : String) (t : IType) : Idx :=
  { x with fields := x.fields.modify a (·.filter (·.ty != t)) }
def isEmpty (x : Idx) : Bool := x.fields.isEmpty
def fieldKeys (x : Idx) : List String := AMap.keys x.fields
def fieldsOf (x : Idx) : AMap (List Field) := x.fields
def fieldOrder (x : Idx) : List String := x.order
def allFields (x : Idx) : List Field := allFieldsOf x.fields
def hasFulltextField (x : Idx) : Bool := x.fields.any (·.2.any (·.ty == .fulltext))
def setProgress (x : Idx) (p t : Nat) : Idx := { x with progress := p, total := t }
def progressOf (x : Idx) : Nat × Nat := (x.progress, x.total)
def languageOf (x : Idx) : Option String := x.language
def setLanguage (x : Idx) (l : Option String) : Idx := { x with language := l }
def stopwordsOf (x : Idx) : Option (List String) := x.stopwords
def setStopwords (x : Idx) (s : Option (List String)) : Idx := { x with stopwords := s }
/-- `memory_usage` (`mod.rs:2191`); `mem` = `RediSearch_MemUsage` (FFI). -/
def memoryUsage (x : Idx) (mem : Nat → Nat) : Nat := match x.spec with | none => 0 | some s => mem s
/-- `index_count` (`mod.rs:2201`). -/
def indexCount (x : Idx) : Nat := (x.fields.map (·.2.length)).sum

/-- The field `build_query_node` targets for attribute `a`:
`self.fields.get(key).and_then(|f| f.first())` (`mod.rs:1398,1423,1471,1483,1541,1602`). -/
def queryField (x : Idx) (a : String) : Option Field := (x.fields.get a).bind (·.head?)

/-- `recreate_index` (`mod.rs:2205`). -/
def recreateIndex (x : Idx) (label : String) (langOk : String → Bool) (tieredOk : Bool)
    (fresh g' : Nat) : Except String Idx :=
  match createRsIndex { x with spec := none } label x.stopwords x.language langOk fresh with
  | .error e => .error e
  | .ok x2 => match registerFields x2 x2.fields none tieredOk with
    | .error e => .error e
    | .ok x3 => .ok (x3.bumpId g')

/-! ## Theorems -/

theorem default_spec (id : Nat) :
    (default id).spec = none ∧ (default id).fields = [] ∧ (default id).order = [] ∧
    (default id).slots = ⟨id, 0, 0⟩ ∧ (default id).slots.countFor id = 0 := by
  simp [default, Slots.countFor]

theorem idOf_spec (x : Idx) : x.idOf = x.id := rfl

theorem bumpId_spec (x : Idx) (g' : Nat) :
    (x.bumpId g').id = g' ∧ (x.bumpId g').slots = { gen := g', cur := 0, stale := x.slots.stale + x.slots.cur } ∧
    (x.bumpId g').fields = x.fields ∧ (x.bumpId g').spec = x.spec := ⟨rfl, rfl, rfl, rfl⟩

/-- `clone_for_update` is observationally the same index (same spec handle, same
ticket slots, equal metadata). -/
theorem cloneForUpdate_eq (x : Idx) : x.cloneForUpdate = x := by cases x; rfl

theorem hasRsIndex_spec (x : Idx) : x.hasRsIndex = true ↔ x.rsPtr ≠ none := by
  cases h : x.spec <;> simp [hasRsIndex, rsPtr, h]

theorem createRsIndex_ok (x : Idx) (label : String) (stop : Option (List String)) (lang : Option String)
    (langOk : String → Bool) (fresh : Nat) (y : Idx) (h : createRsIndex x label stop lang langOk fresh = .ok y) :
    y.spec = some fresh ∧ y.rs = [⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩] ∧ y.fields = x.fields ∧
    y.order = x.order ∧ y.slots = x.slots ∧ y.id = x.id ∧ (lang.all langOk) = true := by
  unfold createRsIndex at h
  split at h; · cases h
  split at h
  · split at h; · cases h
    split at h; · cases h
    split at h; · cases h
    next l _ _ h2 _ => cases h; simp_all
  · split at h; · cases h
    cases h; simp

theorem createRsIndex_lang_err (x : Idx) (label : String) (stop : Option (List String)) (l : String)
    (langOk : String → Bool) (fresh : Nat) (h1 : (stop.getD []).any hasNul = false)
    (h2 : hasNul l = false) (h3 : langOk l = false) :
    createRsIndex x label stop (some l) langOk fresh = .error s!"Language is not supported: {l}" := by
  simp [createRsIndex, h1, h2, h3]

theorem metricOf_ok (sim : Option String) (m : String) (h : metricOf sim = .ok m) :
    sim.getD "euclidean" ∈ ["euclidean", "ip", "cosine"] := by
  unfold metricOf at h; split at h <;> simp_all

/-- `register_fields` on a Range field registers the main field and both array
sub-fields — exactly the three RS fields `Model.Sub` (`main`, `numArr`, `strArr`)
reads and writes. -/
theorem registerOne_range (fo : Option TextOpts) (tk : Bool) (n : String) (o : Option TextOpts) :
    registerOne fo tk (Field.new n .range o) =
      .ok [⟨n, .rangeMain⟩, ⟨n ++ ":numeric:arr", .numArr⟩, ⟨n ++ ":string:arr", .strArr⟩] := by
  simp [registerOne, Field.new, makeArrNames]

/-- Fulltext: the per-call options win over the field's own, weight defaults to
1, phonetic is on iff a non-empty code is given. -/
theorem registerOne_text (fo : Option TextOpts) (tk : Bool) (f : Field) (h : f.ty = .fulltext) :
    registerOne fo tk f = .ok [⟨f.name, .text (((fo.or f.options).bind (·.weight)).getD 1)
      (((fo.or f.options).bind (·.nostem)).getD false)
      (((fo.or f.options).bind (·.phonetic)).any (· ≠ ""))⟩] := by
  simp [registerOne, h]

theorem registerOne_vector_err (fo : Option TextOpts) (tk : Bool) (f : Field) (v : VecOpts)
    (h : f.ty = .vector) (hv : f.vopts = some v) (hd : v.dimension > 0) (e : String)
    (he : metricOf v.sim = .error e) : registerOne fo tk f = .error e := by
  simp [registerOne, h, hv, hd, he]

theorem registerAll_ok (fo : Option TextOpts) (tk : Bool) :
    ∀ (fs : List Field) (r : List RsReg), registerAll fo tk fs = .ok r →
      ∀ f ∈ fs, ∃ rf, registerOne fo tk f = .ok rf ∧ ∀ g ∈ rf, g ∈ r
  | [], _, _ => by simp
  | f :: fs, r, h => by
    simp only [registerAll] at h
    split at h
    · next r1 h1 =>
      split at h
      · next rs h2 =>
        cases h
        intro g hg
        rcases List.mem_cons.mp hg with rfl | hg
        · exact ⟨r1, h1, fun x hx => List.mem_append_left _ hx⟩
        · obtain ⟨rf, h3, h4⟩ := registerAll_ok fo tk fs rs h2 g hg
          exact ⟨rf, h3, fun x hx => List.mem_append_right _ (h4 x hx)⟩
      · cases h
    · cases h

theorem registerFields_ok (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) (tk : Bool) (y : Idx)
    (h : registerFields x m fo tk = .ok y) :
    y.spec = x.spec ∧ y.fields = x.fields ∧ y.order = x.order ∧ y.slots = x.slots ∧
    (∀ r ∈ x.rs, r ∈ y.rs) ∧
    ∀ f ∈ allFieldsOf m, ∃ rf, registerOne fo tk f = .ok rf ∧ ∀ g ∈ rf, g ∈ y.rs := by
  unfold registerFields at h
  split at h
  · next r hr =>
    cases h
    refine ⟨rfl, rfl, rfl, rfl, fun r h => List.mem_append_left _ h, fun f hf => ?_⟩
    obtain ⟨rf, h1, h2⟩ := registerAll_ok fo tk _ r hr f hf
    exact ⟨rf, h1, fun g hg => List.mem_append_right _ (h2 g hg)⟩
  · cases h

/-! Field map -/

theorem getFields_spec (x : Idx) (a : String) : x.getFields a = x.fields.get a := rfl
theorem containsField_iff (x : Idx) (a : String) : x.containsField a = true ↔ a ∈ x.fieldKeys :=
  AMap.get_isSome _ _
theorem hasFieldWithType_iff (x : Idx) (a : String) (t : IType) :
    x.hasFieldWithType a t = true ↔ ∃ fs, x.fields.get a = some fs ∧ ∃ f ∈ fs, f.ty = t := by
  unfold hasFieldWithType; cases x.fields.get a <;> simp
theorem hasFulltextField_iff (x : Idx) :
    x.hasFulltextField = true ↔ ∃ p ∈ x.fields, ∃ f ∈ p.2, f.ty = .fulltext := by
  simp [hasFulltextField]
theorem allFields_mem (x : Idx) (f : Field) : f ∈ x.allFields ↔ ∃ p ∈ x.fields, f ∈ p.2 := by
  simp [allFields, allFieldsOf]
theorem sum_len (m : AMap (List Field)) : (m.map (·.2.length)).sum = (allFieldsOf m).length := by
  induction m with
  | nil => rfl
  | cons p m ih =>
    simp only [allFieldsOf] at *
    simp only [List.map_cons, List.sum_cons, List.flatMap_cons, List.length_append, ih]
theorem indexCount_eq (x : Idx) : x.indexCount = x.allFields.length := sum_len _
theorem isEmpty_iff (x : Idx) : x.isEmpty = true ↔ x.fieldKeys = [] := by
  simp [isEmpty, fieldKeys, AMap.keys]
theorem accessors (x : Idx) : x.fieldsOf = x.fields ∧ x.fieldOrder = x.order ∧
    x.progressOf = (x.progress, x.total) ∧ x.languageOf = x.language ∧ x.stopwordsOf = x.stopwords :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩
theorem setters (x : Idx) (p t : Nat) (l : Option String) (s : Option (List String)) :
    (x.setProgress p t).progressOf = (p, t) ∧ (x.setLanguage l).languageOf = l ∧
    (x.setStopwords s).stopwordsOf = s ∧ (x.setProgress p t).fields = x.fields ∧
    (x.setLanguage l).fields = x.fields ∧ (x.setStopwords s).fields = x.fields := by
  simp [setProgress, progressOf, setLanguage, languageOf, setStopwords, stopwordsOf]
theorem memoryUsage_spec (x : Idx) (mem : Nat → Nat) :
    x.memoryUsage mem = if h : x.spec.isSome then mem (x.spec.get h) else 0 := by
  cases hs : x.spec <;> simp [memoryUsage, hs]

theorem addFieldToExisting_get (x : Idx) (a b : String) (f : Field) :
    (x.addFieldToExisting a f).fields.get b = if b = a then (x.fields.get b).map (· ++ [f]) else x.fields.get b := by
  unfold addFieldToExisting; split
  · exact AMap.get_modify x.fields a b (· ++ [f])
  · next h =>
    by_cases hb : b = a
    · subst hb
      have : x.fields.get b = none := by simpa using h
      simp [this]
    · simp [hb]
theorem insertField_get (x : Idx) (a b : String) (f : Field) :
    (x.insertField a f).fields.get b = if b = a then some [f] else x.fields.get b := AMap.get_upd _ _ _ _
theorem removeField_spec (x : Idx) (a b : String) :
    (x.removeField a).1 = x.containsField a ∧
    (x.removeField a).2.fields.get b = if b = a then none else x.fields.get b := by
  unfold removeField containsField; split
  · next h => simp [h, AMap.get_erase]
  · next h =>
    refine ⟨by simp at h; simp [h], ?_⟩
    by_cases hb : b = a
    · subst hb; simp at h; simp [h]
    · simp [hb]
theorem retainFields_get (x : Idx) (a b : String) (t : IType) :
    (x.retainFields a t).fields.get b =
      if b = a then (x.fields.get b).map (·.filter (·.ty != t)) else x.fields.get b := AMap.get_modify _ _ _ _

/-! ## The `field_order` invariant: it lists exactly the keys, once each. -/

def Inv (x : Idx) : Prop :=
  (AMap.keys x.fields).Nodup ∧ x.order.Nodup ∧ ∀ a, a ∈ x.order ↔ a ∈ AMap.keys x.fields

theorem default_inv (id : Nat) : Inv (default id) := by simp [Inv, default, AMap.keys]

theorem insertField_inv (x : Idx) (a : String) (f : Field) (h : Inv x) : Inv (x.insertField a f) := by
  obtain ⟨h1, h2, h3⟩ := h
  have hk := AMap.get_isSome x.fields a
  by_cases hs : a ∈ AMap.keys x.fields
  · have e1 : (x.insertField a f).order = x.order := by simp [insertField, hk.mpr hs]
    have e2 : AMap.keys (x.insertField a f).fields = AMap.keys x.fields := by
      simp only [insertField]; rw [AMap.keys_upd, if_pos hs]
    refine ⟨?_, ?_, fun b => ?_⟩
    · rw [e2]; exact h1
    · rw [e1]; exact h2
    · rw [e1, e2]; exact h3 b
  · have hn : (x.fields.get a).isSome = false := by
      cases hh : (x.fields.get a).isSome
      · rfl
      · exact absurd (hk.mp hh) hs
    have e1 : (x.insertField a f).order = x.order ++ [a] := by simp [insertField, hn]
    have e2 : AMap.keys (x.insertField a f).fields = AMap.keys x.fields ++ [a] := by
      simp only [insertField]; rw [AMap.keys_upd, if_neg hs]
    refine ⟨?_, ?_, fun b => ?_⟩
    · rw [e2]; exact List.nodup_append.mpr ⟨h1, by simp, by simp; intro y hy hya; subst hya; exact hs hy⟩
    · rw [e1]; exact List.nodup_append.mpr ⟨h2, by simp, by simp; intro y hy hya; subst hya; exact hs ((h3 y).mp hy)⟩
    · rw [e1, e2]; simp [h3 b]

theorem removeField_inv (x : Idx) (a : String) (h : Inv x) : Inv (x.removeField a).2 := by
  obtain ⟨h1, h2, h3⟩ := h
  unfold removeField; split
  · refine ⟨AMap.nodup_keys_erase _ _ h1, h2.filter _, fun b => ?_⟩
    rw [AMap.keys_erase]; simp [h3 b]
  · exact ⟨h1, h2, h3⟩

theorem addFieldToExisting_inv (x : Idx) (a : String) (f : Field) (h : Inv x) :
    Inv (x.addFieldToExisting a f) := by
  unfold addFieldToExisting; split
  · obtain ⟨h1, h2, h3⟩ := h
    exact ⟨by rw [AMap.keys_modify]; exact h1, h2, fun b => by rw [AMap.keys_modify]; exact h3 b⟩
  · exact h

theorem retainFields_inv (x : Idx) (a : String) (t : IType) (h : Inv x) : Inv (x.retainFields a t) := by
  obtain ⟨h1, h2, h3⟩ := h
  exact ⟨by simp only [retainFields]; rw [AMap.keys_modify]; exact h1, h2,
    fun b => by simp only [retainFields]; rw [AMap.keys_modify]; exact h3 b⟩

/-- `recreate_index`: on success the spec is fresh, every current field is
registered again, the metadata is unchanged and the generation is bumped (so
stale populate workers bail: `Maintenance.bump_fresh`). -/
theorem recreateIndex_ok (x : Idx) (label : String) (langOk : String → Bool) (tk : Bool)
    (fresh g' : Nat) (y : Idx) (h : recreateIndex x label langOk tk fresh g' = .ok y) :
    y.spec = some fresh ∧ y.fields = x.fields ∧ y.order = x.order ∧ y.id = g' ∧
    y.slots = x.slots.bump g' ∧ ⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩ ∈ y.rs ∧
    ∀ f ∈ x.allFields, ∃ rf, registerOne none tk f = .ok rf ∧ ∀ g ∈ rf, g ∈ y.rs := by
  unfold recreateIndex at h
  split at h
  · cases h
  · next x2 h2 =>
    obtain ⟨s2, r2, f2, o2, sl2, -, -⟩ := createRsIndex_ok _ _ _ _ _ _ _ h2
    split at h
    · cases h
    · next x3 h3 =>
      cases h
      obtain ⟨s3, f3, o3, sl3, r3, a3⟩ := registerFields_ok _ _ _ _ _ h3
      simp only at f2 o2 sl2
      refine ⟨?_, ?_, ?_, rfl, ?_, ?_, ?_⟩
      · simp only [bumpId]; rw [s3, s2]
      · simp only [bumpId]; rw [f3, f2]
      · simp only [bumpId]; rw [o3, o2]
      · simp only [bumpId]; rw [sl3, sl2]
      · simp only [bumpId]; exact r3 _ (by rw [r2]; simp)
      · intro f hf
        have : f ∈ allFieldsOf x2.fields := by rw [f2]; exact hf
        obtain ⟨rf, h1, h2⟩ := a3 f this
        exact ⟨rf, h1, fun g hg => by simp only [bumpId]; exact h2 g hg⟩

end Idx
end IndexLayer.Meta
