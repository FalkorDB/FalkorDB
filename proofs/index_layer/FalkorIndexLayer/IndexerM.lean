import FalkorIndexLayer.IndexM
import FalkorIndexLayer.Proofs
/-
# `Indexer` (`graph/src/index/indexer.rs`, origin/main 3fec7d7c9; `create_index` re-targeted to 73077bb4a, #3086)

The `ArcSwap<HashMap<label, Arc<Index>>>` is `map : AMap Idx`; a store is a
functional update (readers see either the old or the new map, which is what
`ArcSwap` guarantees). `write_lock` is an identity (`lock`), `cancelled` and
the graph slot are plain fields. FFI results are parameters, as in `IndexM`.
-/
namespace IndexLayer.Meta

inductive IdxOptions
  | text (o : TextOpts)
  | vector (v : VecOpts)

/-- `IndexOptions::language` / `stopwords` / `field_options` / `vector_options`
(`indexer.rs:102-145`). -/
def IdxOptions.language : IdxOptions → Option String
  | .text o => o.language | .vector _ => none
def IdxOptions.stopwords : IdxOptions → Option (List String)
  | .text o => o.stopwords | .vector _ => none
def IdxOptions.fieldOptions : IdxOptions → Option TextOpts
  | .text o => if o.weight.isSome || o.nostem.isSome || o.phonetic.isSome
               then some { weight := o.weight, nostem := o.nostem, phonetic := o.phonetic } else none
  | .vector _ => none
def IdxOptions.vectorOptions : IdxOptions → Option VecOpts
  | .vector v => some v | .text _ => none

theorem IdxOptions.spec (o : TextOpts) (v : VecOpts) :
    (IdxOptions.text o).language = o.language ∧ (IdxOptions.text o).stopwords = o.stopwords ∧
    (IdxOptions.vector v).language = none ∧ (IdxOptions.vector v).stopwords = none ∧
    (IdxOptions.vector v).fieldOptions = none ∧ (IdxOptions.vector v).vectorOptions = some v ∧
    (IdxOptions.text o).vectorOptions = none ∧
    ((IdxOptions.text o).fieldOptions.isSome ↔ (o.weight.isSome ∨ o.nostem.isSome ∨ o.phonetic.isSome)) ∧
    ∀ fo, (IdxOptions.text o).fieldOptions = some fo →
      fo.weight = o.weight ∧ fo.nostem = o.nostem ∧ fo.phonetic = o.phonetic ∧ fo.language = none := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_⟩
  · simp only [IdxOptions.fieldOptions]; split <;> simp_all [or_assoc]
  · intro fo h; simp only [IdxOptions.fieldOptions] at h; split at h <;> cases h; simp

/-- `PopulationTicket` (`indexer.rs:76-97`). -/
structure Ticket where
  label : String
  gen : Nat
def Ticket.generationId (t : Ticket) : Nat := t.gen
def Ticket.labelOf (t : Ticket) : String := t.label
theorem Ticket.accessors (l : String) (g : Nat) :
    (Ticket.mk l g).generationId = g ∧ (Ticket.mk l g).labelOf = l := ⟨rfl, rfl⟩

structure Ixr where
  map : AMap Idx
  cancelled : Bool
  graph : Option Nat
  lock : Nat

/-- Environment of nondeterministic / FFI choices for one call. -/
structure Env where
  langOk : String → Bool
  tieredOk : Bool
  fresh : Nat        -- new RS spec handle
  freshId : Nat      -- `NEXT_INDEX_ID` for `Index::default`
  bumpTo : Nat       -- `NEXT_INDEX_ID` for `bump_id`
  regenTo : Nat      -- `NEXT_INDEX_ID` for `create_index`'s own `bump_id` (#3086)
  lower : String → String

namespace Ixr

def hasIndices (I : Ixr) : Bool := !I.map.isEmpty
def memoryUsage (I : Ixr) (mem : Nat → Nat) : Nat := (I.map.map (·.2.memoryUsage mem)).sum

def fieldName (t : IType) (a : String) : String :=
  match t with | .range => "range:" ++ a | .fulltext => a | .vector => "vector:" ++ a

/-- The field built for one attribute (`indexer.rs:269-287`). -/
def mkField (t : IType) (fo : Option TextOpts) (vo : Option VecOpts) (a : String) : Field :=
  match vo with | some v => Field.newVec (fieldName t a) t v | none => Field.new (fieldName t a) t fo

/-- The attribute loop of `create_index` (`indexer.rs:268-299`). -/
def addAttrs (t : IType) (fo : Option TextOpts) (vo : Option VecOpts) :
    Idx → AMap (List Field) → List String → Except String (Idx × AMap (List Field))
  | x, nf, [] => .ok (x, nf)
  | x, nf, a :: as =>
    let n := fieldName t a
    if Idx.hasNul n then .error "nul" else
    let f := mkField t fo vo a
    let nf' := nf.upd a ((nf.get a).getD [] ++ [f])
    let x' := if x.containsField a then x.addFieldToExisting a f else x.insertField a f
    addAttrs t fo vo x' nf' as

/-- The validation loop (`indexer.rs:247-259`). -/
def validate (existing : Option Idx) (t : IType) : List String → List String → Except String Unit
  | _, [] => .ok ()
  | seen, a :: as =>
    if a ∈ seen then .error s!"Attribute '{a}' is duplicated in the same request"
    else if existing.any (·.hasFieldWithType a t) then .error s!"Attribute '{a}' is already indexed"
    else validate existing t (seen ++ [a]) as

def optParts (lower : String → String) (opts : Option IdxOptions) :
    Option String × Option (List String) × Option TextOpts × Option VecOpts :=
  match opts with
  | some (.text o) => (o.language, o.stopwords.map (·.map lower), some o, none)
  | some (.vector v) => (none, none, none, some v)
  | none => (none, none, none, none)

/-- The pre-validation of `create_index` (`indexer.rs:222-241`). -/
def preCheck (existing : Option Idx) (t : IType) (label : String)
    (p : Option String × Option (List String) × Option TextOpts × Option VecOpts) : Except String Unit :=
  if existing.any (·.hasFulltextField) && p.1.isSome then
    .error s!"Can not override index configuration: Language is already set for label '{label}'"
  else if existing.any (·.hasFulltextField) && p.2.1.isSome then
    .error s!"Can not override index configuration: Stopwords are already set for label '{label}'"
  else if p.2.2.1.isSome && t != .fulltext then .error "Text index options are only valid for fulltext indexes"
  else .ok ()

/-- `share_spec` (#3680, `indexer.rs:357-361`): a statement that only adds fields to a
label whose index has a spec shares that spec (the fields are registered into it at
publish); a vector field, a language or stopword list, or a synchronous caller
(`owes = false`) gets a fresh spec. -/
def shareSpec (owes : Bool) (existing : Option Idx) (t : IType) (language : Option String)
    (stopwords : Option (List String)) : Bool :=
  owes && t != .vector && language.isNone && stopwords.isNone && existing.any (·.hasRsIndex)

/-- The private copy `create_index` mutates (`indexer.rs:362-363`): `fork_for_ddl` of the
existing entry (new generation `regenTo`), or `Index::default` for a new label. -/
def x0Of (E : Env) (existing : Option Idx) (share : Bool) : Idx :=
  match existing with
  | some x => x.forkForDdl share E.regenTo
  | none => Idx.default E.freshId

/-- The RS spec step of `create_index` (#3680, `indexer.rs:397-412`): when shared, the new
fields are deferred to publish; otherwise a fresh spec (the statement's stopwords and
language, else the entry's) with every field registered in declaration order. -/
def specStep (E : Env) (label : String) (share : Bool) (language : Option String)
    (stopwords : Option (List String)) (fo : Option TextOpts) (x1 : Idx) (nf : AMap (List Field)) :
    Except String Idx :=
  if share then .ok (x1.deferFields nf fo)
  else match x1.createRsIndex label (stopwords.or x1.stopwords) (language.or x1.language) E.langOk E.fresh with
    | .error e => .error e
    | .ok x' => x'.registerAllFields E.tieredOk

/-- Language/stopwords metadata and progress (#3680, `indexer.rs:420-429`): an explicit
language or stopword list is recorded (the spec above was built with it); a fulltext
index with no language records "english". -/
def finish (t : IType) (language : Option String) (stopwords : Option (List String)) (total : Nat)
    (x2 : Idx) : Idx :=
  let x3 := match language with
    | some l => x2.setLanguage (some l)
    | none => if x2.language.isNone && t = IType.fulltext then x2.setLanguage (some "english") else x2
  let x4 := if stopwords.isSome then x3.setStopwords stopwords else x3
  x4.setProgress 0 total

/-- `if owes_population { owe_population() }` (`indexer.rs:435-437`). -/
def owe (owes : Bool) (x : Idx) : Idx := if owes then x.owePopulation else x

/-- Everything after validation (`indexer.rs:362-444`). -/
def build (E : Env) (I : Ixr) (owes : Bool) (t : IType) (label : String) (attrs : List String) (total : Nat)
    (p : Option String × Option (List String) × Option TextOpts × Option VecOpts) : Except String Ixr :=
  let share := shareSpec owes (I.map.get label) t p.1 p.2.1
  match addAttrs t p.2.2.1 p.2.2.2 (x0Of E (I.map.get label) share) [] attrs with
  | .error e => .error e
  | .ok (x1, nf) => match specStep E label share p.1 p.2.1 p.2.2.1 x1 nf with
    | .error e => .error e
    | .ok x2 => .ok { I with map := I.map.upd label (owe owes (finish t p.1 p.2.1 total x2)) }

theorem finish_fields (t : IType) (l : Option String) (s : Option (List String)) (n : Nat) (x : Idx) :
    (finish t l s n x).fields = x.fields ∧ (finish t l s n x).spec = x.spec ∧
    (finish t l s n x).progressOf = (0, n) := by
  simp only [finish]; repeat' split
  all_goals simp [Idx.setLanguage, Idx.setStopwords, Idx.setProgress, Idx.progressOf]

theorem specStep_ok (E : Env) (label : String) (share : Bool) (la : Option String) (st : Option (List String))
    (fo : Option TextOpts) (x1 : Idx) (nf : AMap (List Field)) (x2 : Idx)
    (hs : share = true → x1.hasRsIndex = true)
    (h : specStep E label share la st fo x1 nf = .ok x2) : x2.fields = x1.fields ∧ x2.hasRsIndex = true := by
  unfold specStep at h
  split at h
  · next hsh => cases h; exact ⟨rfl, hs hsh⟩
  · split at h
    · cases h
    · next x' hx' =>
      obtain ⟨s1, -, f1, -⟩ := Idx.createRsIndex_ok _ _ _ _ _ _ _ hx'
      obtain ⟨s2, f2, -⟩ := Idx.registerAllFields_ok _ _ _ h
      exact ⟨f2.trans f1, by simp [Idx.hasRsIndex, s2, s1]⟩

/-- `Indexer::create_index` (`indexer.rs:262`); `owes` is `owes_population` (#3680). -/
def createIndex (E : Env) (I : Ixr) (t : IType) (label : String) (attrs : List String) (total : Nat)
    (opts : Option IdxOptions) (owes : Bool) : Except String Ixr :=
  let p := optParts E.lower opts
  match preCheck (I.map.get label) t label p with
  | .error e => .error e
  | .ok () => match validate (I.map.get label) t [] attrs with
    | .error e => .error e
    | .ok () => build E I owes t label attrs total p

/-- The drop loop (`indexer.rs:487-501`). -/
def dropAttrs (t : IType) : Idx → Bool → List String → Idx × Bool
  | x, r, [] => (x, r)
  | x, r, a :: as =>
    match x.getFields a with
    | none => dropAttrs t x r as
    | some fs =>
      if fs.any (·.ty == t) then
        dropAttrs t (if fs.length = 1 then (x.removeField a).2 else x.retainFields a t) true as
      else dropAttrs t x r as

/-- `Indexer::drop_index` (#3680, `indexer.rs:458`): on a private `fork_for_ddl(false)`
copy (generation `regenTo`). Nothing removed: the map is unchanged. No field left: the
label leaves the map. Otherwise `recreate_index` (fresh spec over the remaining fields,
generation `bumpTo`), progress reset, population owed. A RediSearch error leaves the map
unchanged. -/
def dropIndex (E : Env) (I : Ixr) (label : String) (attrs : List String) (t : IType) (total : Nat) :
    Except String (Option (Nat × Nat) × Ixr) :=
  match I.map.get label with
  | none => .ok (none, I)
  | some x0 =>
    let x := x0.forkForDdl false E.regenTo
    let before := x.indexCount
    let targets := if attrs.isEmpty then (x.fields.filter (·.2.any (·.ty == t))).map (·.1) else attrs
    let (x', removed) := dropAttrs t x false targets
    if !removed then .ok (some (0, before), I)
    else if x'.indexCount = 0 then .ok (some (before, 0), { I with map := I.map.erase label })
    else match x'.recreateIndex label E.langOk E.tieredOk E.fresh E.bumpTo with
      | .error e => .error e
      | .ok y => .ok (some (before - x'.indexCount, x'.indexCount),
                      { I with map := I.map.upd label (y.setProgress 0 total).owePopulation })

def hasFieldForLabel (I : Ixr) (l a : String) (t : IType) : Bool :=
  match I.map.get l with | some x => x.hasFieldWithType a t | none => false
def operational (x : Idx) : Bool := x.slots.cur == 0
def isLabelIndexed (I : Ixr) (l a : String) (t : IType) : Bool :=
  match I.map.get l with | some x => operational x && x.hasFieldWithType a t | none => false
def isAttrIndexed (I : Ixr) (l a : String) : Bool :=
  match I.map.get l with | some x => operational x && x.containsField a | none => false
/-- Routing shared by `query`, `query_edges`, `fulltext_query(_edges)`,
`vector_query(_edges)` (`indexer.rs:467-551`): the per-label index, or the
empty iterator when the label has none. -/
def route (I : Ixr) (l : String) : Option Idx := I.map.get l
def vectorField (I : Ixr) (l a : String) : Option Field := do
  let x ← I.map.get l
  let fs ← x.getFields a
  fs.find? (fun f => f.ty == .vector && f.vopts.isSome)
/-- `get_vector_metric` (`indexer.rs:562`). -/
def getVectorMetric (I : Ixr) (l a : String) : Option String := do
  let x ← I.map.get l
  let fs ← x.getFields a
  (fs.find? (fun f => f.ty == .vector && (f.vopts.bind (·.sim)).isSome)).bind (fun f => f.vopts.bind (·.sim))
/-- `get_vector_dimension` (`indexer.rs:585`). -/
def getVectorDimension (I : Ixr) (l a : String) : Option Nat :=
  (vectorField I l a).bind (fun f => f.vopts.map (·.dimension))
def getFields (I : Ixr) (l : String) : AMap (List Field) := ((I.map.get l).map (·.fields)).getD []
def getAllFields (I : Ixr) : List (String × AMap (List Field)) :=
  (I.map.filter (fun p => !p.2.isEmpty)).map (fun p => (p.1, p.2.fields))
/-- `index_info` (`indexer.rs:793`): non-empty indexes, sorted by label (`le`
is `String`'s `Ord`, passed in so that only its order laws are used). -/
def indexInfo (I : Ixr) (le : String → String → Bool) : List (String × Idx) :=
  (I.map.filter (fun p => !p.2.isEmpty)).mergeSort (fun a b => le a.1 b.1)
def hasIndex (I : Ixr) (l : String) : Bool := (I.map.get l).isSome
def hasIndexedAttr (I : Ixr) (l a : String) : Bool :=
  match I.map.get l with | some x => x.containsField a | none => false
/-- `update_progress` (`indexer.rs:840`): the atomics are interior-mutable, so
the published index itself changes. -/
def updateProgress (I : Ixr) (l : String) (p : Nat) : Ixr :=
  match I.map.get l with
  | some x => { I with map := I.map.upd l (x.setProgress p x.total) }
  | none => I
def writeLock (I : Ixr) : Nat := I.lock
def cancel (I : Ixr) : Ixr := { I with cancelled := true, graph := none }
def isCancelled (I : Ixr) : Bool := I.cancelled
def setGraph (I : Ixr) (g : Nat) : Ixr := { I with graph := some g }
def getGraph (I : Ixr) : Option Nat := I.graph

/-! ## Theorems -/

theorem hasIndices_iff (I : Ixr) : I.hasIndices = true ↔ I.map ≠ [] := by
  cases h : I.map <;> simp [hasIndices, h]

theorem memoryUsage_spec (I : Ixr) (mem : Nat → Nat) :
    I.memoryUsage mem = (I.map.map (fun p => p.2.memoryUsage mem)).sum := rfl

theorem validate_ok (ex : Option Idx) (t : IType) :
    ∀ (seen as : List String), validate ex t seen as = .ok () →
      (∀ a ∈ as, a ∉ seen ∧ ex.all (fun x => !x.hasFieldWithType a t) = true) ∧ as.Nodup
  | _, [], _ => by simp
  | seen, a :: as, h => by
    simp only [validate] at h
    split at h; · cases h
    split at h; · cases h
    next h1 h2 =>
    obtain ⟨ih1, ih2⟩ := validate_ok ex t (seen ++ [a]) as h
    refine ⟨fun b hb => ?_, List.nodup_cons.mpr ⟨fun ha => (ih1 a ha).1 (by simp), ih2⟩⟩
    rcases List.mem_cons.mp hb with rfl | hb
    · refine ⟨h1, ?_⟩; cases ex <;> simp_all
    · exact ⟨fun hs => (ih1 b hb).1 (by simp [hs]), (ih1 b hb).2⟩

/-- **Validation**: a successful `create_index` lists no attribute twice and never
re-adds a type an attribute already has (so each attribute keeps at most one
field per type), and `create_index` with text options on a non-fulltext index
fails. -/
theorem createIndex_validates (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat)
    (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (h : createIndex E I t l as tot o owes = .ok I') :
    as.Nodup ∧ ∀ a ∈ as, (I.map.get l).all (fun x => !x.hasFieldWithType a t) = true := by
  unfold createIndex at h
  simp only at h
  split at h; · cases h
  split at h; · cases h
  next hv =>
  obtain ⟨h1, h2⟩ := validate_ok _ t [] as hv
  exact ⟨h2, fun a ha => (h1 a ha).2⟩

theorem createIndex_textopts_err (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat)
    (o : TextOpts) (owes : Bool) (ht : t ≠ .fulltext) (hf : (I.map.get l).any (·.hasFulltextField) = false) :
    createIndex E I t l as tot (some (.text o)) owes = .error "Text index options are only valid for fulltext indexes" := by
  have : (t != .fulltext) = true := by cases t <;> simp_all
  simp [createIndex, preCheck, optParts, hf, this]

theorem hft_addFieldToExisting (x : Idx) (a b : String) (f : Field) (t : IType)
    (h : x.hasFieldWithType a t = true) : (x.addFieldToExisting b f).hasFieldWithType a t = true := by
  rw [Idx.hasFieldWithType_iff] at *
  obtain ⟨fs, h1, g, hg, hgt⟩ := h
  rw [Idx.addFieldToExisting_get]
  by_cases hb : a = b
  · subst hb; exact ⟨fs ++ [f], by simp [h1], g, by simp [hg], hgt⟩
  · exact ⟨fs, by simp [hb, h1], g, hg, hgt⟩

theorem hft_insertField (x : Idx) (a b : String) (f : Field) (t : IType)
    (h : x.hasFieldWithType a t = true) (hb : x.containsField b = false) :
    (x.insertField b f).hasFieldWithType a t = true := by
  rw [Idx.hasFieldWithType_iff] at *
  obtain ⟨fs, h1, g, hg, hgt⟩ := h
  rw [Idx.insertField_get]
  by_cases hab : a = b
  · subst hab; simp [Idx.containsField, h1] at hb
  · exact ⟨fs, by simp [hab, h1], g, hg, hgt⟩

theorem fieldName_ty (t : IType) (fo : Option TextOpts) (vo : Option VecOpts) (a : String) :
    (mkField t fo vo a).ty = t := by
  cases vo <;> rfl

/-- The attribute loop leaves every requested attribute with a field of the
requested type, and keeps every field of type `t` already present. -/
theorem addAttrs_spec (t : IType) (fo : Option TextOpts) (vo : Option VecOpts) :
    ∀ (as : List String) (x : Idx) (nf : AMap (List Field)) (y : Idx) (nf' : AMap (List Field)),
      addAttrs t fo vo x nf as = .ok (y, nf') →
      (∀ a, x.hasFieldWithType a t = true → y.hasFieldWithType a t = true) ∧
      (∀ a ∈ as, y.hasFieldWithType a t = true)
  | [], x, nf, y, nf', h => by simp [addAttrs] at h; obtain ⟨rfl, -⟩ := h; simp
  | a :: as, x, nf, y, nf', h => by
    simp only [addAttrs] at h
    split at h; · cases h
    obtain ⟨ih1, ih2⟩ := addAttrs_spec t fo vo as _ _ y nf' h
    have step : ∀ b, x.hasFieldWithType b t = true →
        (if x.containsField a then x.addFieldToExisting a (mkField t fo vo a)
         else x.insertField a (mkField t fo vo a)).hasFieldWithType b t = true := by
      intro b hb
      split
      · exact hft_addFieldToExisting _ _ _ _ _ hb
      · next hc => exact hft_insertField _ _ _ _ _ hb (by simpa using hc)
    refine ⟨fun b hb => ih1 b (step b hb), fun b hb => ?_⟩
    rcases List.mem_cons.mp hb with rfl | hb
    · apply ih1
      rw [Idx.hasFieldWithType_iff]
      split
      · next hc =>
        obtain ⟨fs, hfs⟩ := Option.isSome_iff_exists.mp hc
        refine ⟨fs ++ [mkField t fo vo b], by rw [Idx.addFieldToExisting_get]; simp [hfs], mkField t fo vo b, by simp, fieldName_ty t fo vo b⟩
      · exact ⟨[mkField t fo vo b], by rw [Idx.insertField_get]; simp, mkField t fo vo b, by simp, fieldName_ty t fo vo b⟩
    · exact ih2 b hb

theorem hft_fields (x y : Idx) (h : y.fields = x.fields) (a : String) (t : IType) :
    y.hasFieldWithType a t = x.hasFieldWithType a t := by simp [Idx.hasFieldWithType, h]

theorem addAttrs_slots (t : IType) (fo : Option TextOpts) (vo : Option VecOpts) :
    ∀ (as : List String) (x : Idx) (nf : AMap (List Field)) (y : Idx) (nf' : AMap (List Field)),
      addAttrs t fo vo x nf as = .ok (y, nf') → y.slots = x.slots ∧ y.id = x.id ∧ y.spec = x.spec
  | [], x, nf, y, nf', h => by simp [addAttrs] at h; obtain ⟨rfl, -⟩ := h; exact ⟨rfl, rfl, rfl⟩
  | a :: as, x, nf, y, nf', h => by
    simp only [addAttrs] at h
    split at h; · cases h
    have ih := addAttrs_slots t fo vo as _ _ y nf' h
    refine ⟨ih.1.trans ?_, ih.2.1.trans ?_, ih.2.2.trans ?_⟩ <;>
      (split <;> simp [Idx.addFieldToExisting, Idx.insertField] <;> split <;> rfl)

theorem owe_fields (owes : Bool) (x : Idx) :
    (owe owes x).fields = x.fields ∧ (owe owes x).spec = x.spec ∧ (owe owes x).progressOf = x.progressOf ∧
    (owe owes x).id = x.id ∧ (owe owes x).slots = (if owes then x.slots.inc x.id else x.slots) ∧
    (owe owes x).stopwords = x.stopwords ∧ (owe owes x).rsStop = x.rsStop := by
  unfold owe; cases owes <;> simp [Idx.owePopulation, Idx.progressOf]

/-- **The copy `create_index` starts from** (#3680): it carries the existing entry's spec
exactly when the spec is shared, and a shared spec exists. -/
theorem x0Of_share (E : Env) (ex : Option Idx) (share : Bool) :
    (x0Of E ex share).spec = (if share then ex.bind (·.spec) else none) ∧
    (x0Of E ex share).fields = ((ex.map (·.fields)).getD []) := by
  cases ex with
  | none => cases share <;> simp [x0Of, Idx.default]
  | some x => cases share <;> simp [x0Of, Idx.forkForDdl, Idx.bumpId]

theorem shareSpec_rs (owes : Bool) (ex : Option Idx) (t : IType) (la : Option String)
    (st : Option (List String)) (h : shareSpec owes ex t la st = true) :
    owes = true ∧ t ≠ .vector ∧ la = none ∧ st = none ∧ (ex.bind (·.spec)).isSome := by
  unfold shareSpec at h
  cases ex with
  | none => simp at h
  | some x => cases la <;> cases st <;> cases t <;> simp_all [Idx.hasRsIndex]

/-- **`create_index` success**: only the target label's entry changes; it now
has a field of the requested type for every requested attribute, its progress
is `(0, total)`, and it has a RediSearch spec. -/
theorem createIndex_ok (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat)
    (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (h : createIndex E I t l as tot o owes = .ok I') :
    (∀ l', l' ≠ l → I'.map.get l' = I.map.get l') ∧
    ∃ x, I'.map.get l = some x ∧ (∀ a ∈ as, x.hasFieldWithType a t = true) ∧
      x.progressOf = (0, tot) ∧ x.hasRsIndex = true := by
  unfold createIndex at h
  simp only at h
  split at h; · cases h
  split at h; · cases h
  unfold build at h
  simp only at h
  split at h; · cases h
  next x1 nf hr1 =>
  obtain ⟨-, hadd⟩ := addAttrs_spec t _ _ as _ [] x1 nf hr1
  obtain ⟨-, -, sp1⟩ := addAttrs_slots t _ _ as _ [] x1 nf hr1
  split at h; · cases h
  next x2 hx2 =>
  have hs : shareSpec owes (I.map.get l) t (optParts E.lower o).1 (optParts E.lower o).2.1 = true →
      x1.hasRsIndex = true := by
    intro hsh
    obtain ⟨-, -, -, -, hr⟩ := shareSpec_rs _ _ _ _ _ hsh
    rw [Idx.hasRsIndex, sp1, (x0Of_share E _ _).1, hsh]; simpa using hr
  obtain ⟨hf2, hs2⟩ := specStep_ok _ _ _ _ _ _ _ _ _ hs hx2
  cases h
  obtain ⟨ff, fs, fp⟩ := finish_fields t (optParts E.lower o).1 (optParts E.lower o).2.1 tot x2
  obtain ⟨of, os, op, -⟩ := owe_fields owes (finish t (optParts E.lower o).1 (optParts E.lower o).2.1 tot x2)
  refine ⟨fun l' hl' => by simp [AMap.get_upd, hl'], _, by simp [AMap.get_upd], fun a ha => ?_, op.trans fp, ?_⟩
  · rw [hft_fields x1 _ (of.trans (ff.trans hf2))]; exact hadd a ha
  · simp only [Idx.hasRsIndex, os, fs]; exact hs2

/-! ## `drop_index`: never leaves an attribute with an empty field list -/

/-- Well-formed field map: every attribute has a non-empty list with at most one
field per type (what `create_index`'s validation enforces). -/
def WF (x : Idx) : Prop := ∀ a fs, x.fields.get a = some fs → fs ≠ [] ∧ (fs.map (·.ty)).Nodup

theorem retain_nonempty (fs : List Field) (t : IType) (hn : (fs.map (·.ty)).Nodup)
    (hl : fs.length ≠ 1) (hne : fs ≠ []) : fs.filter (·.ty != t) ≠ [] := by
  intro h
  rw [List.filter_eq_nil_iff] at h
  match fs, hl, hne with
  | f :: g :: r, _, _ =>
    have hf : f.ty = t := by simpa using h f (by simp)
    have hg : g.ty = t := by simpa using h g (by simp)
    simp only [List.map_cons, List.nodup_cons, List.mem_cons] at hn
    exact hn.1 (Or.inl (hf.trans hg.symm))

theorem dropAttrs_wf (t : IType) :
    ∀ (as : List String) (x : Idx) (r : Bool), WF x → WF (dropAttrs t x r as).1
  | [], x, r, h => h
  | a :: as, x, r, h => by
    simp only [dropAttrs]
    split
    · exact dropAttrs_wf t as x r h
    · next fs hfs =>
      split
      · apply dropAttrs_wf
        split
        · intro b gs hb
          rw [(Idx.removeField_spec x a b).2] at hb
          split at hb; · cases hb
          exact h b gs hb
        · next hl =>
          intro b gs hb
          rw [Idx.retainFields_get] at hb
          split at hb
          · next hba =>
            subst hba
            simp only [Idx.getFields] at hfs
            rw [hfs] at hb; simp at hb; subst hb
            obtain ⟨h1, h2⟩ := h _ fs hfs
            exact ⟨retain_nonempty fs t h2 hl h1,
              List.Sublist.nodup ((List.filter_sublist).map _) h2⟩
          · exact h b gs hb
      · exact dropAttrs_wf t as x r h

theorem dropAttrs_slots (t : IType) :
    ∀ (as : List String) (x : Idx) (r : Bool),
      (dropAttrs t x r as).1.slots = x.slots ∧ (dropAttrs t x r as).1.id = x.id ∧
      (dropAttrs t x r as).1.spec = x.spec ∧ (dropAttrs t x r as).1.stopwords = x.stopwords ∧
      (dropAttrs t x r as).1.language = x.language
  | [], x, r => by simp [dropAttrs]
  | a :: as, x, r => by
    simp only [dropAttrs]
    split
    · exact dropAttrs_slots t as x r
    · next fs _ =>
      split
      · have hstep : ∀ y : Idx, (y = (x.removeField a).2 ∨ y = x.retainFields a t) →
            y.slots = x.slots ∧ y.id = x.id ∧ y.spec = x.spec ∧ y.stopwords = x.stopwords ∧
            y.language = x.language := by
          rintro y (rfl | rfl)
          · unfold Idx.removeField; split <;> simp
          · simp [Idx.retainFields]
        have hy := hstep (if fs.length = 1 then (x.removeField a).2 else x.retainFields a t)
          (by split <;> simp)
        obtain ⟨h1, h2, h3, h4, h5⟩ := dropAttrs_slots t as
          (if fs.length = 1 then (x.removeField a).2 else x.retainFields a t) true
        exact ⟨h1.trans hy.1, h2.trans hy.2.1, h3.trans hy.2.2.1, h4.trans hy.2.2.2.1, h5.trans hy.2.2.2.2⟩
      · exact dropAttrs_slots t as x r

theorem indexCount_fields (x y : Idx) (h : y.fields = x.fields) : y.indexCount = x.indexCount := by
  rw [Idx.indexCount_eq, Idx.indexCount_eq]; simp [Idx.allFields, h]

/-- **`drop_index`**: `None` exactly when the label has no index; only the label's
entry changes; and from a well-formed entry, either the label leaves the map (every
field dropped, reported `(before, 0)`, #3680: as the pre-#3680 `drop_index_bg` did) or
it holds a well-formed entry (no attribute is left with an empty field list) and the
reported counts are `(before - after, after)`. -/
theorem dropIndex_spec (E : Env) (I : Ixr) (l : String) (as : List String) (t : IType) (tot : Nat)
    (r : Option (Nat × Nat) × Ixr) (h : I.dropIndex E l as t tot = .ok r) :
    (r.1 = none ↔ I.map.get l = none) ∧
    (∀ l', l' ≠ l → r.2.map.get l' = I.map.get l') ∧
    (∀ x, I.map.get l = some x → WF x →
      (r.2.map.get l = none ∧ r.1 = some (x.indexCount, 0)) ∨
      ∃ y, r.2.map.get l = some y ∧ WF y ∧ r.1 = some (x.indexCount - y.indexCount, y.indexCount)) := by
  unfold dropIndex at h
  split at h
  · next h0 => cases h; simp [h0]
  · next x0 h0 =>
    simp only at h
    have hfk : (x0.forkForDdl false E.regenTo).fields = x0.fields := (Idx.forkForDdl_spec _ _ _).2.2.1
    have hb : (x0.forkForDdl false E.regenTo).indexCount = x0.indexCount := indexCount_fields _ _ hfk
    generalize hd : dropAttrs t (x0.forkForDdl false E.regenTo) false _ = d at h
    obtain ⟨x', rm⟩ := d
    have hw : ∀ x, I.map.get l = some x → WF x → WF x' := by
      intro x hx hwx; rw [h0] at hx; cases hx
      have e := congrArg Prod.fst hd; simp only at e; rw [← e]
      exact dropAttrs_wf t _ (x0.forkForDdl false E.regenTo) false
        (fun a fs hf => hwx a fs (by rw [hfk] at hf; exact hf))
    simp only at h
    split at h
    · cases h
      refine ⟨by simp [h0], fun _ _ => rfl, fun x hx hwx => Or.inr ⟨x, hx, hwx, ?_⟩⟩
      rw [h0] at hx; cases hx; simp [hb]
    · split at h
      · next hz =>
        cases h
        refine ⟨by simp [h0], fun l' hl' => by simp [AMap.get_erase, hl'], fun x hx _ => Or.inl ?_⟩
        rw [h0] at hx; cases hx
        simp [AMap.get_erase, hb]
      · split at h
        · cases h
        · next y hy =>
          cases h
          obtain ⟨-, fy, -⟩ := Idx.recreateIndex_ok _ _ _ _ _ _ _ hy
          refine ⟨by simp [h0], fun l' hl' => by simp [AMap.get_upd, hl'], fun x hx hwx => Or.inr ⟨(y.setProgress 0 tot).owePopulation, by simp [AMap.get_upd], ?_, ?_⟩⟩
          · intro a gs hg
            exact hw x hx hwx a gs (by
              simpa [Idx.owePopulation, Idx.setProgress, fy] using hg)
          · rw [h0] at hx; cases hx
            have : ((y.setProgress 0 tot).owePopulation).indexCount = x'.indexCount :=
              indexCount_fields _ _ (by simp [Idx.owePopulation, Idx.setProgress, fy])
            simp [this, hb]

/-- **`drop_index` rebuilds** (#3680): when the label keeps fields, its new entry has a
fresh spec (built over the remaining fields), generation `bumpTo`, exactly one pending
ticket of it (the owed population), progress `(0, total)`, and owes its population; the
published entry (its spec and counters) is not touched (the map is a new value). -/
theorem dropIndex_rebuilt (E : Env) (I : Ixr) (l : String) (as : List String) (t : IType) (tot : Nat)
    (r : Option (Nat × Nat) × Ixr) (h : I.dropIndex E l as t tot = .ok r) (y : Idx)
    (hy : r.2.map.get l = some y) (hne : r.2.map.get l ≠ I.map.get l) :
    y.spec = some E.fresh ∧ y.id = E.bumpTo ∧ y.slots = ⟨E.bumpTo, 1, 0⟩ ∧ y.owed = true ∧
      y.progressOf = (0, tot) := by
  unfold dropIndex at h
  split at h
  · cases h; exact absurd rfl hne
  · next x0 h0 =>
    simp only at h
    generalize hd : dropAttrs t (x0.forkForDdl false E.regenTo) false _ = d at h
    obtain ⟨x', rm⟩ := d
    have hsl : x'.slots = ⟨E.regenTo, 0, 0⟩ := by
      have e := congrArg Prod.fst hd; simp only at e; rw [← e, (dropAttrs_slots t _ _ false).1]
      exact (Idx.forkForDdl_spec _ _ _).2.1
    simp only at h
    split at h
    · cases h; exact absurd rfl hne
    · split at h
      · cases h; simp [AMap.get_erase] at hy
      · split at h
        · cases h
        · next y0 hy0 =>
          cases h
          simp [AMap.get_upd] at hy; subst hy
          obtain ⟨s0, -, -, i0, sl0, -⟩ := Idx.recreateIndex_ok _ _ _ _ _ _ _ hy0
          simp [Idx.owePopulation, Idx.setProgress, Idx.progressOf, s0, i0, sl0, hsl, Slots.bump, Slots.inc]

theorem lookups (I : Ixr) (l a : String) (t : IType) :
    I.hasFieldForLabel l a t = (I.map.get l).any (·.hasFieldWithType a t) ∧
    I.isLabelIndexed l a t = (I.map.get l).any (fun x => operational x && x.hasFieldWithType a t) ∧
    I.isAttrIndexed l a = (I.map.get l).any (fun x => operational x && x.containsField a) ∧
    I.hasIndexedAttr l a = (I.map.get l).any (·.containsField a) ∧
    I.hasIndex l = (I.map.get l).isSome ∧ I.route l = I.map.get l ∧
    I.getFields l = ((I.map.get l).map (·.fields)).getD [] := by
  simp only [hasFieldForLabel, isLabelIndexed, isAttrIndexed, hasIndexedAttr, hasIndex, route, getFields]
  cases I.map.get l <;> simp

/-- `is_label_indexed` implies `has_field_for_label`: a pending (populating)
index is never reported as usable. -/
theorem isLabelIndexed_imp (I : Ixr) (l a : String) (t : IType) (h : I.isLabelIndexed l a t = true) :
    I.hasFieldForLabel l a t = true ∧ ∃ x, I.map.get l = some x ∧ x.slots.cur = 0 := by
  simp only [isLabelIndexed, hasFieldForLabel] at *
  split at h
  · next x hx => simp [hx, operational] at h ⊢; exact ⟨h.2, h.1⟩
  · cases h

theorem getVectorDimension_spec (I : Ixr) (l a : String) (d : Nat) (h : I.getVectorDimension l a = some d) :
    ∃ x fs f, I.map.get l = some x ∧ x.getFields a = some fs ∧ f ∈ fs ∧ f.ty = .vector ∧
      (f.vopts.map (·.dimension)) = some d := by
  unfold getVectorDimension vectorField at h
  cases hx : I.map.get l with
  | none => simp [hx] at h
  | some x =>
    cases hfs : x.getFields a with
    | none => simp [hx, hfs] at h
    | some fs =>
      cases hf : fs.find? (fun f => f.ty == .vector && f.vopts.isSome) with
      | none => simp [hx, hfs, hf] at h
      | some f =>
        simp [hx, hfs, hf] at h
        have := List.find?_some hf
        simp at this
        exact ⟨x, fs, f, rfl, hfs, List.mem_of_find?_eq_some hf, this.1, by simpa using h⟩

theorem getVectorMetric_spec (I : Ixr) (l a : String) (m : String) (h : I.getVectorMetric l a = some m) :
    ∃ x fs f, I.map.get l = some x ∧ x.getFields a = some fs ∧ f ∈ fs ∧ f.ty = .vector ∧
      f.vopts.bind (·.sim) = some m := by
  unfold getVectorMetric at h
  cases hx : I.map.get l with
  | none => simp [hx] at h
  | some x =>
    cases hfs : x.getFields a with
    | none => simp [hx, hfs] at h
    | some fs =>
      cases hf : fs.find? (fun f => f.ty == .vector && (f.vopts.bind (·.sim)).isSome) with
      | none => simp [hx, hfs, hf] at h
      | some f =>
        simp [hx, hfs, hf] at h
        have := List.find?_some hf
        simp at this
        exact ⟨x, fs, f, rfl, hfs, List.mem_of_find?_eq_some hf, this.1, by simpa using h⟩

theorem getAllFields_spec (I : Ixr) (l : String) (m : AMap (List Field)) :
    (l, m) ∈ I.getAllFields ↔ ∃ x, (l, x) ∈ I.map ∧ x.isEmpty = false ∧ x.fields = m := by
  simp [getAllFields]
  constructor
  · rintro ⟨a, b, ⟨h1, h2⟩, rfl, rfl⟩; exact ⟨b, h1, h2, rfl⟩
  · rintro ⟨x, h1, h2, rfl⟩; exact ⟨l, x, ⟨h1, h2⟩, rfl, rfl⟩

/-- `index_info`: a permutation of the non-empty indexes, sorted by label
(given only that `le` is a total preorder, which `String::cmp` is). -/
theorem indexInfo_spec (I : Ixr) (le : String → String → Bool)
    (trans : ∀ a b c, le a b → le b c → le a c) (total : ∀ a b, le a b || le b a) :
    (I.indexInfo le).Perm (I.map.filter (fun p => !p.2.isEmpty)) ∧
    (I.indexInfo le).Pairwise (fun a b => le a.1 b.1) :=
  ⟨List.mergeSort_perm _ _, List.pairwise_mergeSort (le := fun (a b : String × Idx) => le a.1 b.1)
    (fun (a b c : String × Idx) h1 h2 => trans a.1 b.1 c.1 h1 h2) (fun (a b : String × Idx) => total a.1 b.1) _⟩

theorem updateProgress_spec (I : Ixr) (l l' : String) (p : Nat) :
    (I.updateProgress l p).map.get l' =
      if l' = l then (I.map.get l).map (fun x => x.setProgress p x.total) else I.map.get l' := by
  unfold updateProgress; split
  · next x hx => rw [AMap.get_upd]; by_cases h : l' = l <;> simp [h, hx]
  · next hx =>
    by_cases h : l' = l
    · subst h; simp [hx]
    · simp [h]

theorem misc (I : Ixr) (g : Nat) :
    I.writeLock = I.lock ∧ (I.cancel).isCancelled = true ∧ (I.cancel).getGraph = none ∧
    (I.setGraph g).getGraph = some g ∧ I.isCancelled = I.cancelled ∧ (I.cancel).map = I.map ∧
    (I.setGraph g).map = I.map := by
  simp [writeLock, cancel, isCancelled, getGraph, setGraph]

/-! ## #3086 (73077bb4a): adding a field to an existing entry starts a new generation -/

/-- `is_ticket_current` (`indexer.rs:698`). -/
def isTicketCurrent (I : Ixr) (tk : Ticket) : Bool :=
  match I.map.get tk.label with | some x => x.id == tk.gen | none => false

theorem registerFields_id (x : Idx) (m : AMap (List Field)) (fo : Option TextOpts) (tk : Bool) (y : Idx)
    (h : x.registerFields m fo tk = .ok y) : y.id = x.id := by
  unfold Idx.registerFields at h; split at h <;> cases h; rfl

theorem registerAllFields_id (x : Idx) (tk : Bool) (y : Idx)
    (h : x.registerAllFields tk = .ok y) : y.id = x.id := by
  unfold Idx.registerAllFields at h; split at h <;> cases h; rfl

/-- The RS step keeps the generation and its counters (#3680: a vector field no longer
recreates an existing spec here: the fork has none, `fork_for_ddl(false)`). -/
theorem specStep_slots (E : Env) (label : String) (share : Bool) (la : Option String)
    (st : Option (List String)) (fo : Option TextOpts) (x1 : Idx) (nf : AMap (List Field)) (x2 : Idx)
    (h : specStep E label share la st fo x1 nf = .ok x2) :
    x2.slots = x1.slots ∧ x2.id = x1.id := by
  unfold specStep at h
  split at h
  · cases h; exact ⟨rfl, rfl⟩
  · split at h
    · cases h
    · next x' hx' =>
      obtain ⟨-, -, -, -, sl1, i1, -⟩ := Idx.createRsIndex_ok _ _ _ _ _ _ _ hx'
      obtain ⟨-, -, -, sl2, -⟩ := Idx.registerAllFields_ok _ _ _ h
      exact ⟨sl2.trans sl1, (registerAllFields_id _ _ _ h).trans i1⟩

theorem finish_slots (t : IType) (l : Option String) (s : Option (List String)) (n : Nat) (x : Idx) :
    (finish t l s n x).slots = x.slots ∧ (finish t l s n x).id = x.id := by
  simp only [finish]; repeat' split
  all_goals simp [Idx.setLanguage, Idx.setStopwords, Idx.setProgress]

/-- The entry `create_index` stores, from the copy it started from. -/
theorem createIndex_entry (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat)
    (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (h : createIndex E I t l as tot o owes = .ok I') :
    ∃ x, I'.map.get l = some x ∧
      x.id = (x0Of E (I.map.get l) (shareSpec owes (I.map.get l) t (optParts E.lower o).1 (optParts E.lower o).2.1)).id ∧
      x.slots = (owe owes (x0Of E (I.map.get l) (shareSpec owes (I.map.get l) t (optParts E.lower o).1 (optParts E.lower o).2.1))).slots := by
  unfold createIndex at h
  simp only at h
  split at h; · cases h
  split at h; · cases h
  unfold build at h
  simp only at h
  split at h; · cases h
  next x1 nf hr1 =>
  obtain ⟨sl1, i1, -⟩ := addAttrs_slots t _ _ as _ [] x1 nf hr1
  split at h; · cases h
  next x2 hx2 =>
  cases h
  obtain ⟨sl2, i2⟩ := specStep_slots _ _ _ _ _ _ _ _ _ hx2
  obtain ⟨sl3, i3⟩ := finish_slots t (optParts E.lower o).1 (optParts E.lower o).2.1 tot x2
  refine ⟨owe owes (finish t (optParts E.lower o).1 (optParts E.lower o).2.1 tot x2), by simp [AMap.get_upd], ?_, ?_⟩
  · rw [(owe_fields _ _).2.2.2.1, i3, i2, i1]
  · rw [(owe_fields _ _).2.2.2.2.1, (owe_fields _ _).2.2.2.2.1, sl3, i3, sl2, i2, sl1, i1]

/-- **#3086, #3680**: `create_index` on a label that already has an entry stores it
under the fresh generation `regenTo` with counters of its own: one pending ticket of
that generation exactly when it owes its population (the publisher's job takes it
over), none otherwise, and no stale ticket (the old job's tickets stay on the published
entry it acquired them from: `createIndex_old_release_noop`). -/
theorem createIndex_regen (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat)
    (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (x0 : Idx) (hx0 : I.map.get l = some x0)
    (h : createIndex E I t l as tot o owes = .ok I') :
    ∃ x, I'.map.get l = some x ∧ x.id = E.regenTo ∧
      x.slots = ⟨E.regenTo, if owes then 1 else 0, 0⟩ := by
  obtain ⟨x, hx, hi, hs⟩ := createIndex_entry E I t l as tot o owes I' h
  refine ⟨x, hx, ?_, ?_⟩
  · rw [hi]; simp only [hx0, x0Of]; exact (Idx.forkForDdl_spec _ _ _).1
  · rw [hs]; simp only [hx0, x0Of]
    cases owes <;> simp [owe, Idx.owePopulation, Idx.forkForDdl, Idx.bumpId, Slots.bump, Slots.inc]

/-- **#3680**: a ticket of any other generation (a job of the published entry, still
running) released into the stored entry changes nothing: its counters only count the
new generation's ticket, so the old job can never mark the new entry OPERATIONAL. -/
theorem createIndex_old_release_noop (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String)
    (tot : Nat) (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (x0 : Idx) (hx0 : I.map.get l = some x0)
    (h : createIndex E I t l as tot o owes = .ok I') (g : Nat) (hg : g ≠ E.regenTo) :
    ∃ x, I'.map.get l = some x ∧ x.slots.dec g = x.slots ∧
      x.slots.countFor E.regenTo = (if owes then 1 else 0) := by
  obtain ⟨x, hx, -, hs⟩ := createIndex_regen E I t l as tot o owes I' x0 hx0 h
  refine ⟨x, hx, ?_, ?_⟩
  · rw [hs]; simp [Slots.dec, hg]
  · rw [hs]; simp [Slots.countFor]

/-- Hence the job the publisher starts for `regenTo` passes `is_ticket_current`, and a populate job
of any generation other than `regenTo` fails it at its next batch (`createIndex_old_ticket_stale_ne`;
an earlier generation is stale exactly when it is not `regenTo`). -/
theorem createIndex_old_ticket_stale (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String)
    (tot : Nat) (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (x0 : Idx) (hx0 : I.map.get l = some x0)
    (h : createIndex E I t l as tot o owes = .ok I') (g : Nat) :
    I'.isTicketCurrent ⟨l, g⟩ = (g == E.regenTo) := by
  obtain ⟨x, hx, hid, -⟩ := createIndex_regen E I t l as tot o owes I' x0 hx0 h
  simp only [isTicketCurrent, hx, hid]
  by_cases hg : g = E.regenTo
  · subst hg; simp only [beq_self_eq_true]
  · have : E.regenTo ≠ g := fun e => hg e.symm
    rw [beq_false_of_ne hg, beq_false_of_ne this]

/-- A populate job of a generation other than `regenTo` (in particular any earlier one) is stale. -/
theorem createIndex_old_ticket_stale_ne (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String)
    (tot : Nat) (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (x0 : Idx) (hx0 : I.map.get l = some x0)
    (h : createIndex E I t l as tot o owes = .ok I') (g : Nat) (hg : g ≠ E.regenTo) :
    I'.isTicketCurrent ⟨l, g⟩ = false := by
  rw [createIndex_old_ticket_stale E I t l as tot o owes I' x0 hx0 h g]
  exact beq_false_of_ne hg

/-- A brand-new entry is not bumped: it keeps `Index::default`'s generation, with the
owed ticket when it owes its population. -/
theorem createIndex_new_entry (E : Env) (I : Ixr) (t : IType) (l : String) (as : List String) (tot : Nat)
    (o : Option IdxOptions) (owes : Bool) (I' : Ixr) (hx0 : I.map.get l = none)
    (h : createIndex E I t l as tot o owes = .ok I') :
    ∃ x, I'.map.get l = some x ∧ x.id = E.freshId ∧
      x.slots = ⟨E.freshId, if owes then 1 else 0, 0⟩ := by
  obtain ⟨x, hx, hi, hs⟩ := createIndex_entry E I t l as tot o owes I' h
  refine ⟨x, hx, ?_, ?_⟩
  · rw [hi]; simp [hx0, x0Of, Idx.default]
  · rw [hs]; simp only [hx0, x0Of]
    cases owes <;> simp [owe, Idx.owePopulation, Idx.default, Slots.inc]

/-! ## BUG (new): a range index on an attribute that already has a fulltext
index is never used correctly. `build_query_node` takes the attribute's *first*
field (`queryField`), which is the fulltext field named `s`, not `range:s`. -/

def E0 : Env := { langOk := fun _ => true, tieredOk := true, fresh := 7, freshId := 1, bumpTo := 100,
                  regenTo := 101, lower := id }
def I0 : Ixr := { map := [], cancelled := false, graph := none, lock := 0 }

def scenario : Option (Option Field) :=
  match createIndex E0 I0 .fulltext "L" ["s"] 0 none true with
  | .ok I1 => match createIndex E0 I1 .range "L" ["s"] 0 none true with
    | .ok I2 => some ((I2.route "L").bind (·.queryField "s"))
    | .error _ => none
  | .error _ => none

/-- Both `CREATE` calls succeed, the label reports a RANGE index on `s`, yet the
field every range query is built against is the FULLTEXT one. Live:
`CREATE FULLTEXT INDEX FOR (n:L) ON (n.s)` then `CREATE INDEX FOR (n:L) ON (n.s)`;
`MATCH (n:L) WHERE n.s = 'foo'` → Rust `[]`, C `[2]` (also `>`, `IN`, inline `{s:..}`). -/
theorem bug_range_after_fulltext :
    scenario.map (·.map (fun f => (f.name, f.ty))) = some (some ("s", .fulltext)) := by
  decide

end Ixr
end IndexLayer.Meta
