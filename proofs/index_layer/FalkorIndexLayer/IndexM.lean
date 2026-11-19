import FalkorIndexLayer.Meta
import FalkorIndexLayer.Model
/-
# `Index` (`graph/src/index/mod.rs:875-2248`): metadata, RS spec lifecycle

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
  /-- Gap closed 2026-10-20 (#3166): the stopword list the live RediSearch spec was
  *created* with (`RediSearch_IndexOptionsSetStopwords`, `mod.rs:1119-1133`, read by
  `RediSearch_CreateIndex`). RediSearch fixes it at creation: `register_fields`
  (`mod.rs:1181`) adds fields to the spec and never touches it. `none` = the
  `SetStopwords(null, 0)` arm (`mod.rs:1132`), i.e. RediSearch's own default. -/
  rsStop : Option (List String) := none
  /-- `population_owed` (#3680, `mod.rs:907`): set by `owe_population` on an `Index` a DDL
  statement built, taken once by the publisher (`take_population_owed`). -/
  owed : Bool := false
  /-- `deferred_fields` (#3680, `mod.rs:913`): fields added to a spec this `Index` shares
  with the published one, registered in it only at publish. -/
  deferred : List (AMap (List Field) × Option TextOpts) := []

namespace Idx

/-- `Default for Index` (`mod.rs:1056`): no owed population, no deferred fields (#3680). -/
def default (id : Nat) : Idx :=
  { id, spec := none, rs := [], fields := [], order := [], slots := ⟨id, 0, 0⟩,
    progress := 0, total := 0, language := none, stopwords := none }

/-- `Index::id` (`mod.rs:1084`). -/
def idOf (x : Idx) : Nat := x.id
/-- `Index::bump_id` (`mod.rs:1093`), `g'` the next value of the static counter. -/
def bumpId (x : Idx) (g' : Nat) : Idx := { x with id := g', slots := x.slots.bump g' }
/-- `Index::fork_for_ddl` (#3680, `mod.rs:1118`; replaces `clone_for_update`): a copy with
*fresh* pending counters (`PendingSlots { current_generation: id, 0, 0 }`), then
`bump_id` to `g'` (the next `NEXT_INDEX_ID`). The RediSearch spec (and what is
registered in it, and the stopwords it was created with) is carried only when
`share`; so are the deferred fields. `population_owed` starts `false`. -/
def forkForDdl (x : Idx) (share : Bool) (g' : Nat) : Idx :=
  ({ x with spec := if share then x.spec else none, rs := if share then x.rs else [],
            rsStop := if share then x.rsStop else none, slots := ⟨x.id, 0, 0⟩, owed := false,
            deferred := if share then x.deferred else [] } : Idx).bumpId g'
/-- `Index::defer_fields` (`mod.rs:1149`). -/
def deferFields (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) : Idx :=
  { x with deferred := x.deferred ++ [(m, fo)] }
/-- `Index::owe_population` (`mod.rs:1172`): the creation ticket
(`increment_pending_for_generation(self.id)`) and the flag. -/
def owePopulation (x : Idx) : Idx := { x with slots := x.slots.inc x.id, owed := true }
/-- `Index::take_population_owed` (`mod.rs:1180`): the flag, and the index with it cleared
(`swap(false)`). -/
def takePopulationOwed (x : Idx) : Bool × Idx := (x.owed, { x with owed := false })
/-- `has_rs_index` (`mod.rs:1088`). -/
def hasRsIndex (x : Idx) : Bool := x.spec.isSome
/-- `rs_ptr` (`mod.rs:1095`): `none` is the null pointer. -/
def rsPtr (x : Idx) : Option Nat := x.spec

def hasNul (s : String) : Bool := s.toList.contains (Char.ofNat 0)

/-- `l.eq_ignore_ascii_case("english")` (`mod.rs:1232`): `Char.toLower` lowers ASCII only. -/
def isEnglish (l : String) : Bool := l.toList.map Char.toLower == "english".toList

/-- The FFI fact the English filter relies on: RediSearch's `RSLanguage_Find`
supports English (its own default), in any ASCII case. -/
def EnglishOk (langOk : String → Bool) : Prop := ∀ l, isEnglish l = true → langOk l = true

/-- `create_rs_index` (`mod.rs:1203`). #3680: an English language is not passed
(`language.filter(..)`, `mod.rs:1232`), so it is neither checked nor set on the spec.
Error order as in Rust (#3274): stopwords
`CString::new` (1113-1120), language `CString::new` (1127-1129), label
`CString::new` (1130), all before the options block is allocated; then
unsupported language (1156-1160). On success the spec is fresh and carries only
`NONE_INDEXABLE_FIELDS` (1175-1186), and (#3166 gap) is created with the stopword
list `stop` (`mod.rs:1144-1154`), recorded as `rsStop`. -/
def createRsIndex (x : Idx) (label : String) (stop : Option (List String)) (lang : Option String)
    (langOk : String → Bool) (fresh : Nat) : Except String Idx :=
  if (stop.getD []).any hasNul then .error "nul" else
  match lang.filter (fun l => !isEnglish l) with
  | some l => if hasNul l then .error "nul" else if hasNul label then .error "nul"
              else if !langOk l then .error s!"Language is not supported: {l}"
              else .ok { x with spec := some fresh, rs := [⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩], rsStop := stop }
  | none => if hasNul label then .error "nul"
            else .ok { x with spec := some fresh, rs := [⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩], rsStop := stop }

/-- The metric match of `register_fields` (`mod.rs:1273-1286`). -/
def metricOf (sim : Option String) : Except String String :=
  match sim.getD "euclidean" with
  | "euclidean" => .ok "L2"
  | "ip" => .ok "IP"
  | "cosine" => .ok "COSINE"
  | o => .error s!"Unknown similarity function '{o}', expected 'euclidean', 'ip', or 'cosine'"

/-- One iteration of `register_fields` (`mod.rs:1192-1344`). `tieredOk` is
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
    let ph := eo.any TextOpts.phoneticEnabled
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

/-- `register_fields` (`mod.rs:1311`), applied to the spec. -/
def registerFields (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) (tieredOk : Bool) :
    Except String Idx :=
  match registerAll fo tieredOk (allFieldsOf m) with
  | .ok r => .ok { x with rs := x.rs ++ r }
  | .error e => .error e

/-- The fields in declaration order: attribute by attribute along `field_order`. -/
def orderedFields (x : Idx) : List Field := x.order.flatMap (fun a => (x.fields.get a).getD [])

/-- `register_all_fields` (#3680, `mod.rs:1300`): one `register_fields(.., None)` per
attribute of `field_order`, stopping at the first error. -/
def registerAllFields (x : Idx) (tieredOk : Bool) : Except String Idx :=
  match registerAll none tieredOk x.orderedFields with
  | .ok r => .ok { x with rs := x.rs ++ r }
  | .error e => .error e

/-- `register_deferred_fields` (#3680, `mod.rs:1159`): every deferred batch, in order, with its
options; an error is ignored (`let _ =`) and the next batch still runs. The list is taken. -/
def registerDeferredFields (x : Idx) (tieredOk : Bool) : Idx :=
  { (x.deferred.foldl (fun y p => match y.registerFields p.1 p.2 tieredOk with
      | .ok z => z | .error _ => y) x) with deferred := [] }

/-! Field-map operations (`mod.rs:1968-2077`). -/
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
/-- `memory_usage` (`mod.rs:2205`); `mem` = `RediSearch_MemUsage` (FFI). -/
def memoryUsage (x : Idx) (mem : Nat → Nat) : Nat := match x.spec with | none => 0 | some s => mem s
/-- `index_count` (`mod.rs:2215`). -/
def indexCount (x : Idx) : Nat := (x.fields.map (·.2.length)).sum

/-- The field `build_query_node` targets for attribute `a`:
`self.fields.get(key).and_then(|f| f.first())` (`mod.rs:1397,1426,1484,1496,1554,1615`). -/
def queryField (x : Idx) (a : String) : Option Field := (x.fields.get a).bind (·.head?)

/-- `recreate_index` (`mod.rs:2349`): since #3680 the fields go in through
`register_all_fields` (declaration order). -/
def recreateIndex (x : Idx) (label : String) (langOk : String → Bool) (tieredOk : Bool)
    (fresh g' : Nat) : Except String Idx :=
  match createRsIndex { x with spec := none } label x.stopwords x.language langOk fresh with
  | .error e => .error e
  | .ok x2 => match registerAllFields x2 tieredOk with
    | .error e => .error e
    | .ok x3 => .ok (x3.bumpId g')

/-! ## Theorems -/

theorem default_spec (id : Nat) :
    (default id).spec = none ∧ (default id).fields = [] ∧ (default id).order = [] ∧
    (default id).slots = ⟨id, 0, 0⟩ ∧ (default id).slots.countFor id = 0 ∧
    (default id).owed = false ∧ (default id).deferred = [] := by
  simp [default, Slots.countFor]

theorem idOf_spec (x : Idx) : x.idOf = x.id := rfl

theorem bumpId_spec (x : Idx) (g' : Nat) :
    (x.bumpId g').id = g' ∧ (x.bumpId g').slots = { gen := g', cur := 0, stale := x.slots.stale + x.slots.cur } ∧
    (x.bumpId g').fields = x.fields ∧ (x.bumpId g').spec = x.spec := ⟨rfl, rfl, rfl, rfl⟩

/-- **`fork_for_ddl`** (#3680): a new generation `g'` with counters of its own and no
pending ticket (none of the original's tickets, current or stale, is carried), the same
fields, order and metadata, nothing owed; the spec is the original's exactly when `share`. -/
theorem forkForDdl_spec (x : Idx) (share : Bool) (g' : Nat) :
    (x.forkForDdl share g').id = g' ∧ (x.forkForDdl share g').slots = ⟨g', 0, 0⟩ ∧
    (x.forkForDdl share g').fields = x.fields ∧ (x.forkForDdl share g').order = x.order ∧
    (x.forkForDdl share g').spec = (if share then x.spec else none) ∧
    (x.forkForDdl share g').language = x.language ∧
    (x.forkForDdl share g').stopwords = x.stopwords ∧
    (x.forkForDdl share g').progressOf = x.progressOf ∧
    (x.forkForDdl share g').owed = false ∧
    (x.forkForDdl share g').deferred = (if share then x.deferred else []) := by
  simp [forkForDdl, bumpId, Slots.bump, progressOf]

/-- Unshared, the fork has no spec: the DDL statement must build one. -/
theorem forkForDdl_unshared (x : Idx) (g' : Nat) :
    (x.forkForDdl false g').hasRsIndex = false ∧ (x.forkForDdl false g').rs = [] := by
  simp [forkForDdl, bumpId, hasRsIndex]

/-- `owe_population` on a fresh generation: exactly one pending ticket of it, flagged owed. -/
theorem owePopulation_spec (x : Idx) :
    x.owePopulation.slots = x.slots.inc x.id ∧ x.owePopulation.owed = true ∧
    x.owePopulation.id = x.id ∧ x.owePopulation.fields = x.fields ∧
    x.owePopulation.spec = x.spec := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `take_population_owed` answers `true` once: a second take answers `false`. -/
theorem takePopulationOwed_once (x : Idx) :
    (x.takePopulationOwed).1 = x.owed ∧ ((x.takePopulationOwed).2.takePopulationOwed).1 = false ∧
    (x.takePopulationOwed).2.slots = x.slots := ⟨rfl, rfl, rfl⟩

/-- `defer_fields` registers nothing: the spec and what it holds are unchanged (the
published version sharing it sees no change), the batch is queued last. -/
theorem deferFields_spec (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) :
    (x.deferFields m fo).spec = x.spec ∧ (x.deferFields m fo).rs = x.rs ∧
    (x.deferFields m fo).fields = x.fields ∧ (x.deferFields m fo).slots = x.slots ∧
    (x.deferFields m fo).deferred = x.deferred ++ [(m, fo)] := ⟨rfl, rfl, rfl, rfl, rfl⟩


theorem hasRsIndex_spec (x : Idx) : x.hasRsIndex = true ↔ x.rsPtr ≠ none := by
  cases h : x.spec <;> simp [hasRsIndex, rsPtr, h]

/-- `create_rs_index` success. The language check: a language that is not English was
accepted by `RSLanguage_Find` (`langOk`); `createRsIndex_ok_langOk` gives the
pre-#3680 conclusion `lang.all langOk` from the FFI fact `EnglishOk`. -/
theorem createRsIndex_ok (x : Idx) (label : String) (stop : Option (List String)) (lang : Option String)
    (langOk : String → Bool) (fresh : Nat) (y : Idx) (h : createRsIndex x label stop lang langOk fresh = .ok y) :
    y.spec = some fresh ∧ y.rs = [⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩] ∧ y.fields = x.fields ∧
    y.order = x.order ∧ y.slots = x.slots ∧ y.id = x.id ∧
    (lang.all (fun l => isEnglish l || langOk l)) = true := by
  unfold createRsIndex at h
  split at h; · cases h
  split at h
  · next l hl =>
    split at h; · cases h
    split at h; · cases h
    split at h; · cases h
    next h3 =>
    cases h
    cases lang with
    | none => simp at hl
    | some l0 =>
      simp only [Option.filter] at hl
      split at hl
      · cases hl; simp_all
      · cases hl
  · next hl =>
    split at h; · cases h
    cases h
    cases lang with
    | none => simp
    | some l0 =>
      simp only [Option.filter] at hl
      split at hl
      · cases hl
      · next he => simp_all

theorem createRsIndex_ok_langOk (x : Idx) (label : String) (stop : Option (List String))
    (lang : Option String) (langOk : String → Bool) (hen : EnglishOk langOk) (fresh : Nat) (y : Idx)
    (h : createRsIndex x label stop lang langOk fresh = .ok y) : (lang.all langOk) = true := by
  have := (createRsIndex_ok x label stop lang langOk fresh y h).2.2.2.2.2.2
  cases lang with
  | none => rfl
  | some l =>
    simp only [Option.all] at this ⊢
    cases he : isEnglish l
    · simpa [he] using this
    · exact hen l he

/-- #3680: an English language (any ASCII case) is not passed to RediSearch: the spec is
created exactly as with no language (no schema rule, so a range field keeps the value 0). -/
theorem createRsIndex_english (x : Idx) (label : String) (stop : Option (List String)) (l : String)
    (langOk : String → Bool) (fresh : Nat) (he : isEnglish l = true) :
    createRsIndex x label stop (some l) langOk fresh = createRsIndex x label stop none langOk fresh := by
  simp [createRsIndex, Option.filter, he]

/-- The pre-#3274 statement of `createRsIndex_lang_err`: with NUL-free stopwords
and language, an unsupported language is always reported, whatever the label. -/
def LangErrSpec (x : Idx) : Prop :=
  ∀ (label : String) (stop : Option (List String)) (l : String)
    (langOk : String → Bool) (fresh : Nat), (stop.getD []).any hasNul = false →
    hasNul l = false → langOk l = false →
    createRsIndex x label stop (some l) langOk fresh = .error s!"Language is not supported: {l}"

/-- Counterexample (#3274): the label `CString::new` now runs before the language
check, so a label with a NUL byte and an unsupported language `x` reports the NUL
error, not `Language is not supported: x`. -/
theorem createRsIndex_lang_err_fails (x : Idx) : ¬ LangErrSpec x := by
  intro h
  have := h "L\x00" none "x" (fun _ => false) 0 rfl (by decide) rfl
  have hx : isEnglish "x" = false := by decide
  simp [createRsIndex, hasNul, Option.filter, hx] at this
  exact absurd this (by decide)

/-- What still holds on the new code: an unsupported language is reported when
the stopwords, language and label are all NUL-free. `hen` is the FFI fact that
RediSearch supports English (`EnglishOk`), which #3680's filter no longer asks it. -/
theorem createRsIndex_lang_err_cleanLabel (x : Idx) (label : String) (stop : Option (List String))
    (l : String) (langOk : String → Bool) (fresh : Nat) (h1 : (stop.getD []).any hasNul = false)
    (h2 : hasNul l = false) (h4 : hasNul label = false) (h3 : langOk l = false)
    (hen : EnglishOk langOk) :
    createRsIndex x label stop (some l) langOk fresh = .error s!"Language is not supported: {l}" := by
  have he : isEnglish l = false := by
    cases h : isEnglish l
    · rfl
    · rw [hen l h] at h3; cases h3
  simp [createRsIndex, h1, h2, h3, h4, Option.filter, he]

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
1, phonetic is on iff the code is not absent, empty or C's `"no"` (`phonetic_enabled`, #3514). -/
theorem registerOne_text (fo : Option TextOpts) (tk : Bool) (f : Field) (h : f.ty = .fulltext) :
    registerOne fo tk f = .ok [⟨f.name, .text (((fo.or f.options).bind (·.weight)).getD 1)
      (((fo.or f.options).bind (·.nostem)).getD false)
      ((fo.or f.options).any TextOpts.phoneticEnabled)⟩] := by
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

theorem registerFields_spec_rs (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) (tk : Bool)
    (y : Idx) (h : registerFields x m fo tk = .ok y) :
    y.spec = x.spec ∧ y.fields = x.fields ∧ y.slots = x.slots ∧ y.id = x.id ∧
    y.deferred = x.deferred ∧ y.owed = x.owed := by
  unfold registerFields at h; split at h <;> cases h; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem registerDeferred_fold (tk : Bool) :
    ∀ (ds : List (AMap (List Field) × Option TextOpts)) (x : Idx),
      let y := ds.foldl (fun y p => match y.registerFields p.1 p.2 tk with
        | .ok z => z | .error _ => y) x
      y.spec = x.spec ∧ y.fields = x.fields ∧ y.slots = x.slots ∧ y.id = x.id ∧
        (∀ r ∈ x.rs, r ∈ y.rs) ∧ y.owed = x.owed
  | [], x => by simp
  | p :: ds, x => by
    simp only [List.foldl_cons]
    split
    · next z hz =>
      obtain ⟨a, b, c, d, -, f⟩ := registerFields_spec_rs _ _ _ _ _ hz
      obtain ⟨-, -, -, -, r, -⟩ := registerFields_ok _ _ _ _ _ hz
      obtain ⟨a', b', c', d', r', f'⟩ := registerDeferred_fold tk ds z
      exact ⟨a'.trans a, b'.trans b, c'.trans c, d'.trans d, fun q hq => r' q (r q hq), f'.trans f⟩
    · exact registerDeferred_fold tk ds x

/-- `register_deferred_fields`: registers into the *same* spec, leaves the fields, generation
and tickets alone, keeps every registration already there, and empties the queue (so a second
call registers nothing). -/
theorem registerDeferredFields_spec (x : Idx) (tk : Bool) :
    (x.registerDeferredFields tk).spec = x.spec ∧ (x.registerDeferredFields tk).fields = x.fields ∧
    (x.registerDeferredFields tk).slots = x.slots ∧ (x.registerDeferredFields tk).id = x.id ∧
    (∀ r ∈ x.rs, r ∈ (x.registerDeferredFields tk).rs) ∧ (x.registerDeferredFields tk).deferred = [] ∧
    ((x.registerDeferredFields tk).registerDeferredFields tk).rs = (x.registerDeferredFields tk).rs := by
  obtain ⟨a, b, c, d, r, -⟩ := registerDeferred_fold tk x.deferred x
  refine ⟨a, b, c, d, r, rfl, ?_⟩
  simp [registerDeferredFields]

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
      simp only [insertField]; rw [AMap.keys_upd, ite_eq_left hs]
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
      simp only [insertField]; rw [AMap.keys_upd, ite_eq_right hs]
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

theorem get_of_mem_nodup {β : Type} :
    ∀ (m : AMap β) (p : String × β), (AMap.keys m).Nodup → p ∈ m → m.get p.1 = some p.2
  | [], _, _, h => by simp at h
  | q :: m, p, hn, h => by
    rw [AMap.get_cons]
    simp only [AMap.keys, List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.mp h with rfl | h
    · simp
    · have hne : q.1 ≠ p.1 := fun e => hn.1 (by rw [e]; exact List.mem_map.mpr ⟨p, h, rfl⟩)
      simp only [hne, ite_false]
      exact get_of_mem_nodup m p hn.2 h

/-- Under the `field_order` invariant, declaration order lists every field
(`register_all_fields` misses none). -/
theorem orderedFields_complete (x : Idx) (h : Inv x) (f : Field) (hf : f ∈ x.allFields) :
    f ∈ x.orderedFields := by
  obtain ⟨h1, -, h3⟩ := h
  obtain ⟨p, hp, hfp⟩ := (allFields_mem x f).mp hf
  have hk : p.1 ∈ AMap.keys x.fields := List.mem_map.mpr ⟨p, hp, rfl⟩
  simp only [orderedFields, List.mem_flatMap]
  exact ⟨p.1, (h3 _).mpr hk, by rw [get_of_mem_nodup _ _ h1 hp]; exact hfp⟩

theorem registerAllFields_ok (x : Idx) (tk : Bool) (y : Idx) (h : registerAllFields x tk = .ok y) :
    y.spec = x.spec ∧ y.fields = x.fields ∧ y.order = x.order ∧ y.slots = x.slots ∧
    (∀ r ∈ x.rs, r ∈ y.rs) ∧
    ∀ f ∈ x.orderedFields, ∃ rf, registerOne none tk f = .ok rf ∧ ∀ g ∈ rf, g ∈ y.rs := by
  unfold registerAllFields at h
  split at h
  · next r hr =>
    cases h
    refine ⟨rfl, rfl, rfl, rfl, fun r h => List.mem_append_left _ h, fun f hf => ?_⟩
    obtain ⟨rf, h1, h2⟩ := registerAll_ok none tk _ r hr f hf
    exact ⟨rf, h1, fun g hg => List.mem_append_right _ (h2 g hg)⟩
  · cases h

/-- `recreate_index`: on success the spec is fresh, every field in declaration
order is registered again (#3680: `register_all_fields`), the metadata is
unchanged and the generation is bumped (so stale populate workers bail:
`Maintenance.bump_fresh`). -/
theorem recreateIndex_ok (x : Idx) (label : String) (langOk : String → Bool) (tk : Bool)
    (fresh g' : Nat) (y : Idx) (h : recreateIndex x label langOk tk fresh g' = .ok y) :
    y.spec = some fresh ∧ y.fields = x.fields ∧ y.order = x.order ∧ y.id = g' ∧
    y.slots = x.slots.bump g' ∧ ⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩ ∈ y.rs ∧
    ∀ f ∈ x.orderedFields, ∃ rf, registerOne none tk f = .ok rf ∧ ∀ g ∈ rf, g ∈ y.rs := by
  unfold recreateIndex at h
  split at h
  · cases h
  · next x2 h2 =>
    obtain ⟨s2, r2, f2, o2, sl2, -, -⟩ := createRsIndex_ok _ _ _ _ _ _ _ h2
    split at h
    · cases h
    · next x3 h3 =>
      cases h
      obtain ⟨s3, f3, o3, sl3, r3, a3⟩ := registerAllFields_ok _ _ _ h3
      simp only at f2 o2 sl2
      refine ⟨?_, ?_, ?_, rfl, ?_, ?_, ?_⟩
      · simp only [bumpId]; rw [s3, s2]
      · simp only [bumpId]; rw [f3, f2]
      · simp only [bumpId]; rw [o3, o2]
      · simp only [bumpId]; rw [sl3, sl2]
      · simp only [bumpId]; exact r3 _ (by rw [r2]; simp)
      · intro f hf
        have : f ∈ x2.orderedFields := by simp only [orderedFields] at hf ⊢; rw [f2, o2]; exact hf
        obtain ⟨rf, h1, h2⟩ := a3 f this
        exact ⟨rf, h1, fun g hg => by simp only [bumpId]; exact h2 g hg⟩

/-- The pre-#3680 statement of `recreateIndex_ok` (every field registered again), on
every index satisfying the `field_order` invariant (all reachable ones: `Inv` is
preserved by every field-map operation above). -/
theorem recreateIndex_ok_all (x : Idx) (hi : Inv x) (label : String) (langOk : String → Bool)
    (tk : Bool) (fresh g' : Nat) (y : Idx) (h : recreateIndex x label langOk tk fresh g' = .ok y) :
    y.spec = some fresh ∧ y.fields = x.fields ∧ y.order = x.order ∧ y.id = g' ∧
    y.slots = x.slots.bump g' ∧ ⟨"NONE_INDEXABLE_FIELDS", .noneTag⟩ ∈ y.rs ∧
    ∀ f ∈ x.allFields, ∃ rf, registerOne none tk f = .ok rf ∧ ∀ g ∈ rf, g ∈ y.rs := by
  obtain ⟨a, b, c, d, e, f, g⟩ := recreateIndex_ok x label langOk tk fresh g' y h
  exact ⟨a, b, c, d, e, f, fun q hq => g q (orderedFields_complete x hi q hq)⟩

end Idx
end IndexLayer.Meta
