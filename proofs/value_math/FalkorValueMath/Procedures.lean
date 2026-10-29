import FalkorValueMath.Value
import FalkorValueMath.Registry
/-!
# `graph/src/runtime/functions/procedures.rs`, `functions/path.rs`, `entity_type.rs`
(origin/main 1c0e6c8de)

A procedure returns a `Batch` built by `Batch::from_columns`; here a batch is its column
list (`List (List (V F))`, `Column::Ints(xs)` read back as `V.int`). Graph reads
(`get_labels`, `label_node_count_by_idx`, `index_info`, `constraints`, …) are fields of
`GraphView`; the function registry is a list of `Ent` (the `GraphFn` fields the
procedures read). `sort_by(|a, b| a.name.cmp(&b.name))` is Lean's stable `mergeSort` on
`name ≤` (Rust's `sort_by` is a stable sort, and a stable sort's output is unique).

**YIELD contract.** The binder resolves `YIELD x` against the name list registered in
`procedure: [...]` and the runtime reads the batch *positionally* (`procedure_call.rs`).
`yields_match` proves that for every registered procedure with a body, the batch has
exactly one column per declared yield, all of the same length — so `YIELD` by name can
never read a missing or misaligned column.

| Lean | Rust |
| --- | --- |
| `procRegs` | `register` :41 (name, yields, `write procedure:`) |
| `dbLabels`, `dbTypes`, `dbProps` | `db_labels` :47, `db_types` :64, `db_properties` :81 |
| `dbIndexes`, `typesOf`, `optionsOf`, `statusOf`, `infoNames` | `db_indexes` :97-284 |
| `dbMetaStats` | `db_meta_stats` :292 |
| `ftCreate`, `ftQueryNodes`, `ftQueryRels`, `ftDrop`, `vecQueryNodes`, `vecQueryRels` | :332 :342 :356 :366 :383 :393 |
| `dbConstraints` | `db_constraints` :403 |
| `dbmsProcedures` | `dbms_procedures` :437 |
| `dbmsFunctions`, `mergePicks` | `dbms_functions` :469 (+ `BUILTIN_ENTRIES` :531, `BUILTIN_LOWER_NAMES` :543) |
| `buildColumns` | `build_columns` :557 |
| `tyValue`, `tyString`, `collectTys`/`collectTysL` + `canonTypes` (`DBMS_TYPE_ORDER`), `formatUnion`, `formatUnionStrings` | :613, :646, :705, :717, :729 |
| `pathNodes`, `pathRels`, `pathSigs` | path.rs `nodes` :29, `relationships` :52, `register` :25 |
| `EntityType.fmt` | entity_type.rs:26 |
-/

namespace ValueMath.Proc
open FalkorCore

open ValueMath

variable {F : Type}

/-- A batch = its columns (`Batch::from_columns`). -/
abbrev Cols (F : Type) := List (List (V F))

/-- All columns have length `n`. -/
def Rect (n : Nat) (b : Cols F) : Prop := ∀ c ∈ b, c.length = n

/-! ## Graph-reading procedures -/

inductive IdxTy where
  | range | vector | fulltext
  deriving DecidableEq

structure VecOpts where
  dimension : Nat              -- `u64`
  similarity : Option String
  m : Option Nat
  efConstruction : Option Nat
  efRuntime : Option Nat

/-- `Field`: `name`/`numeric_arr_name`/`string_arr_name` are `CString`s; `none` inside
models a non-UTF-8 name (`to_str()` fails). -/
structure Field where
  ty : IdxTy
  name : Option String
  numArr : Option (Option String)
  strArr : Option (Option String)
  vec : Option VecOpts

structure IndexInfo where
  label : String
  pending : Int
  progress : Nat
  total : Nat
  fields : List (String × List Field)     -- HashMap
  fieldOrder : List String
  language : Option String
  stopwords : Option (List String)
  entityType : String

inductive CType where | unique | mandatory
inductive CStatus where | underConstruction | operational | failed
inductive EntityType where | node | relationship
  deriving DecidableEq

/-- `Display for EntityType` (entity_type.rs:26). -/
def EntityType.fmt : EntityType → String
  | .node => "NODE"
  | .relationship => "RELATIONSHIP"

def CType.fmt : CType → String
  | .unique => "UNIQUE" | .mandatory => "MANDATORY"
def CStatus.fmt : CStatus → String
  | .underConstruction => "UNDER CONSTRUCTION" | .operational => "OPERATIONAL" | .failed => "FAILED"

structure Constraint where
  ct : CType
  entityType : EntityType
  label : String
  properties : List String
  status : CStatus

/-- What the procedures read from the graph. -/
structure GraphView where
  labels : List String
  types : List String
  attrs : List String
  labelCount : Nat → Nat
  typeCount : Nat → Nat
  nodeCount : Nat
  relCount : Nat
  propKeyCount : Nat
  indexes : List IndexInfo
  constraints : List Constraint

def strs (l : List String) : List (V F) := l.map .str

/-- `db_labels` (:47), `db_types` (:64), `db_properties` (:81). -/
def dbLabels (g : GraphView) : Cols F := [strs g.labels]
def dbTypes (g : GraphView) : Cols F := [strs g.types]
def dbProps (g : GraphView) : Cols F := [strs g.attrs]

/-- `HashMap` index `fields[attr]` (:127): panics (`none`) on a missing key. -/
def flookup (m : List (String × List Field)) (k : String) : Option (List Field) :=
  (m.find? (·.1 = k)).map (·.2)

def slot : IdxTy → Nat
  | .range => 0 | .vector => 1 | .fulltext => 2

/-- The `types` list of one attribute (:126-142): a `seen[3]` bitmap, read back in the
fixed order RANGE, VECTOR, FULLTEXT (the `for (slot, name)` loop, unrolled). -/
def typesOfFields (fs : List Field) : List String :=
  (if fs.any (fun f => f.ty = .range) then ["RANGE"] else []) ++
  (if fs.any (fun f => f.ty = .vector) then ["VECTOR"] else []) ++
  (if fs.any (fun f => f.ty = .fulltext) then ["FULLTEXT"] else [])

/-- The `types` map (:124-145); `none` = the `fields[attr]` panic. -/
def typesOf (i : IndexInfo) : Option (List (String × V F)) :=
  i.fieldOrder.foldlM (fun m a => (flookup i.fields a).map fun fs =>
    omInsert m a (.list (strs (typesOfFields fs)))) []

/-- `i64::try_from(u64).unwrap_or(i64::MAX)` (:170). -/
def i64Max : Nat := 2 ^ 63 - 1
def dimToI64 (d : Nat) : Int := if d ≤ i64Max then d else i64Max

/-- One attribute's options map (:150-197). -/
def attrOpts (fs : Option (List Field)) : List (String × V F) :=
  match fs.bind (fun fs => fs.findSome? (·.vec)) with
  | none => []
  | some v =>
    omInsert (omInsert (omInsert (omInsert (omInsert []
      "dimension" (.int (dimToI64 v.dimension)))
      "similarityFunction" (.str (v.similarity.getD "euclidean")))
      "M" (.int (v.m.getD 16)))
      "efConstruction" (.int (v.efConstruction.getD 200)))
      "efRuntime" (.int (v.efRuntime.getD 10))

def optionsOf (i : IndexInfo) : List (String × V F) :=
  i.fieldOrder.foldl (fun m a => omInsert m a (.map (attrOpts (flookup i.fields a)))) []

/-- `status` (:211-217). -/
def statusOf (i : IndexInfo) : String :=
  if i.pending > 0 then s!"[Indexing] {i.progress}/{i.total}: UNDER CONSTRUCTION" else "OPERATIONAL"

/-- RediSearch field names of one `Field` (:237-249): name if non-empty, then the two
array companions when present and valid UTF-8. -/
def fieldNames (f : Field) : List String :=
  (let n := f.name.getD ""; if n.isEmpty then [] else [n]) ++
  (match f.numArr with | some (some s) => [s] | _ => []) ++
  (match f.strArr with | some (some s) => [s] | _ => [])

/-- The `info.fields` names (:232-252): walked in `field_order`, missing keys skipped. -/
def infoNames (i : IndexInfo) : List String :=
  (i.fieldOrder.flatMap fun a => match flookup i.fields a with
    | none => []
    | some fs => fs.flatMap fieldNames) ++ ["NONE_INDEXABLE_FIELDS"]

def infoOf (i : IndexInfo) : V F :=
  .map [("fields", .list ((infoNames i).map fun n => .map [("name", .str n)]))]

/-- One `db.indexes` row, nine values in yield order; `none` = panic. -/
def indexRow (i : IndexInfo) : Option (List (V F)) :=
  (typesOf i).map fun t =>
    [.str i.label, .list (strs i.fieldOrder), .map t, .map (optionsOf i),
     .str (i.language.getD "english"), .list (strs (i.stopwords.getD [])),
     .str i.entityType, .str (statusOf i), infoOf i]

/-- `db_indexes` (:97): nine columns, column `k` = the `k`-th value of every row. -/
def dbIndexes (g : GraphView) : Option (Cols F) :=
  (g.indexes.mapM indexRow).map fun rows => (List.range 9).map fun k => rows.map (·.getD k .null)

/-- `db_meta_stats` (:292): one row; maps built by `OrderMap::insert` in label/type order. -/
def dbMetaStats (g : GraphView) : Cols F :=
  let lm := (g.labels.zipIdx).foldl (fun m (n, i) => omInsert m n (.int (g.labelCount i))) []
  let tm := (g.types.zipIdx).foldl (fun m (n, i) => omInsert m n (.int (g.typeCount i))) []
  [[.map lm], [.map tm], [.int g.relCount], [.int g.nodeCount], [.int g.labels.length],
   [.int g.types.length], [.int g.propKeyCount]]

/-- `db_constraints` (:403): one row per constraint, in `g.constraints()` order. -/
def dbConstraints (g : GraphView) : Cols F :=
  [g.constraints.map (fun c => .str c.ct.fmt), g.constraints.map (fun c => .str c.label),
   g.constraints.map (fun c => .list (strs c.properties)),
   g.constraints.map (fun c => .str c.entityType.fmt), g.constraints.map (fun c => .str c.status.fmt)]

/-- The stubs. `empty_procedure_batch` = a batch with no rows (zero columns here). -/
def ftCreate : Except String (Cols F) := .ok []
def ftQueryNodes : Except String (Cols F) := .error "db.idx.fulltext.queryNodes() is not supported in this version"
def ftQueryRels : Except String (Cols F) := .error "db.idx.fulltext.queryRelationships() is not supported in this version"
def ftDrop : Except String (Cols F) := .error "db.idx.fulltext.drop() is not supported in this version"
def vecQueryNodes : Except String (Cols F) := .error "db.idx.vector.queryNodes() is rewritten by the planner"
def vecQueryRels : Except String (Cols F) := .error "db.idx.vector.queryRelationships() is rewritten by the planner"

/-! ## Registry procedures -/

open ValueMath.Reg (Ty FnType FnArgs)

/-- The `GraphFn` fields `dbms.*` read. -/
structure Ent where
  name : String
  write : Bool
  nonDet : Bool
  args : FnArgs
  fnType : FnType
  ret : Ty

def isProc (e : Ent) : Bool := match e.fnType with | .procedure _ => true | _ => false
def isAgg (e : Ent) : Bool := match e.fnType with | .aggregation => true | _ => false
def isInternal (e : Ent) : Bool := match e.fnType with | .internal => true | _ => false
def isUdf (e : Ent) : Bool := match e.fnType with | .udf => true | _ => false

def nameLe (a b : Ent) : Bool := decide (a.name ≤ b.name)

/-- `sort_by(|a, b| a.name.cmp(&b.name))`. -/
def sortByName (l : List Ent) : List Ent := l.mergeSort nameLe

/-- `dbms_procedures` (:437). -/
def dbmsProcedures (fns : List Ent) : Cols F :=
  let ps := sortByName (fns.filter isProc)
  [ps.map (fun f => .str f.name), ps.map (fun f => .str (if f.write then "WRITE" else "READ"))]

/-- `format_union` (:717) / `format_union_strings` (:729): the same text. -/
def formatUnion : List String → String
  | [] => ""
  | [a] => a
  | [a, b] => a ++ " or " ++ b
  | l => String.intercalate ", " l.dropLast ++ ", or " ++ l.getLast!

def formatUnionStrings (l : List String) : String := formatUnion l

def anyNames : List String :=
  ["Map", "Node", "Edge", "List", "Path", "Datetime", "Date", "Time", "Duration", "String",
   "Boolean", "Integer", "Float", "Null", "Pointer", "Point", "Vectorf32"]

/-- The single-type spelling of a non-composite type (the leaf arms of `type_to_dbms_string`). -/
def leafString : Ty → String
  | .null => "Null" | .bool => "Boolean" | .int => "Integer" | .float => "Float"
  | .string => "String" | .list _ => "List" | .map => "Map" | .node => "Node"
  | .rel => "Edge" | .path => "Path" | .vecf32 => "Vectorf32" | .point => "Point"
  | .datetime => "Datetime" | .date => "Date" | .time => "Time" | .duration => "Duration"
  | .any | .union _ | .optional _ => ""

mutual
/-- `collect_dbms_types` (:705): the spellings a type stands for, nested unions / `Optional` /
`Any` flattened (`Any` = all of `DBMS_TYPE_ORDER`). -/
def collectTys : Ty → List String
  | .union ts => collectTysL ts
  | .optional t => collectTys t
  | .any => anyNames
  | t => [leafString t]
def collectTysL : List Ty → List String
  | [] => []
  | t :: ts => collectTys t ++ collectTysL ts
end

/-- Position in `DBMS_TYPE_ORDER` (C's `SIType` bit order); unknown = its length. -/
def typePos (s : String) : Nat := anyNames.findIdx (· == s)

/-- `strs.sort_by_key(position in DBMS_TYPE_ORDER)` (stable). -/
def sortByPos (l : List String) : List String := l.mergeSort (fun a b => decide (typePos a ≤ typePos b))

/-- `Vec::dedup`: drop a member equal to its predecessor. -/
def dedupAdj : List String → List String
  | a :: b :: t => if a = b then dedupAdj (b :: t) else a :: dedupAdj (b :: t)
  | l => l

/-- The union branch of `type_to_dbms_string` (:662-675): collect, sort by C order, dedup. -/
def canonTypes (l : List String) : List String := dedupAdj (sortByPos l)

/-- `type_to_dbms_string` (:646). -/
def tyString : Ty → String
  | .any => formatUnion anyNames
  | .union ts => formatUnionStrings (canonTypes (collectTysL ts))
  | .optional t => tyString t
  | t => leafString t

/-- `type_to_dbms_value` (:613): interned spellings; `Union` built; `Optional` unwraps. -/
def tyValue : Ty → V F
  | .any => .str (formatUnion anyNames)          -- ANY_UNION (:607)
  | .null => .str "Null" | .bool => .str "Boolean" | .int => .str "Integer"
  | .float => .str "Float" | .string => .str "String" | .list _ => .str "List"
  | .map => .str "Map" | .node => .str "Node" | .rel => .str "Edge" | .path => .str "Path"
  | .vecf32 => .str "Vectorf32" | .point => .str "Point" | .datetime => .str "Datetime"
  | .date => .str "Date" | .time => .str "Time" | .duration => .str "Duration"
  | .union ts => .str (tyString (.union ts))
  | .optional t => tyValue t

/-- `build_columns` (:557): eight columns per entry. -/
def buildRow (f : Ent) : List (V F) :=
  [.str f.name, tyValue f.ret,
   .list (match f.args with | .fixed ts => ts.map tyValue | .varLength t => [tyValue t]),
   .bool (isInternal f), .bool (!f.nonDet && !isAgg f && !isProc f), .bool (isAgg f),
   .bool (match f.args with | .varLength _ => true | _ => false), .bool (isUdf f)]

def buildColumns (es : List Ent) : Cols F :=
  (List.range 8).map fun k => es.map fun e => (buildRow e).getD k .null

/-- The merge loop (:494-507) over the cursors `i`, `j`, as suffixes: take the UDF iff the
builtins are exhausted or `udfs[j].name < builtins[i].name`. -/
def mergePicks : List Ent → List Ent → List Ent
  | [], us => us
  | bs, [] => bs
  | b :: bs, u :: us => if u.name < b.name then u :: mergePicks (b :: bs) us else b :: mergePicks bs (u :: us)
termination_by bs us => bs.length + us.length

/-- `BUILTIN_LOWER_NAMES.binary_search(&f.name.to_lowercase()).is_err()` (:479): the
binary search on the sorted name list is membership. -/
def keepUdf (lower : String → String) (bs : List Ent) (u : Ent) : Bool :=
  !((bs.map fun b => lower b.name).contains (lower u.name))

/-- `dbms_functions` (:469): built-ins = non-procedures sorted; UDFs whose lowercase name is
not a built-in's lowercase name (binary search in `BUILTIN_LOWER_NAMES`), sorted; merged.
The early return (:485) is the `udfs = []` case of the merge. -/
def dbmsFunctions (lower : String → String) (fns udfReg : List Ent) : Cols F :=
  let bs := sortByName (fns.filter (fun f => !isProc f))
  let us := udfReg.filter (keepUdf lower bs)
  if us.isEmpty then buildColumns bs else buildColumns (mergePicks bs (sortByName us))

/-! ## `register` (:41): name, yields, write -/

def procRegs : List (String × List String × Bool) :=
  [("db.labels", ["label"], false), ("db.relationshipTypes", ["relationshipType"], false),
   ("db.propertyKeys", ["propertyKey"], false),
   ("db.indexes", ["label", "properties", "types", "options", "language", "stopwords",
     "entitytype", "status", "info"], false),
   ("db.meta.stats", ["labels", "relTypes", "relCount", "nodeCount", "labelCount",
     "relTypeCount", "propertyKeyCount"], false),
   ("db.idx.fulltext.createNodeIndex", [], true), ("db.idx.fulltext.queryNodes", ["node", "score"], false),
   ("db.idx.fulltext.queryRelationships", ["relationship", "score"], false),
   ("db.idx.fulltext.drop", [], true), ("db.idx.vector.queryNodes", ["node", "score"], false),
   ("db.idx.vector.queryRelationships", ["relationship", "score"], false),
   ("db.constraints", ["type", "label", "properties", "entitytype", "status"], false),
   ("dbms.procedures", ["name", "mode"], false),
   ("dbms.functions", ["name", "return_type", "arguments", "internal", "reducible",
     "aggregation", "variable_len", "udf"], false)]

def yieldsOf (n : String) : List String := ((procRegs.find? (·.1 = n)).map (·.2.1)).getD []

/-! ## Theorems -/

theorem procRegs_nodup : (procRegs.map (·.1)).Nodup := by decide +kernel

theorem mapM_length {α β} {f : α → Option β} : ∀ {l : List α} {r : List β}, l.mapM f = some r → r.length = l.length
  | [], r, h => by simp at h; subst h; rfl
  | a :: l, r, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨b, _, rs, hrs, rfl⟩ := h
    simp [mapM_length hrs]

theorem rect_map_len {α} (n : Nat) (l : List α) (fs : List (α → V F)) (hn : l.length = n) :
    Rect n (fs.map fun f => l.map f) := by
  intro c hc; simp only [List.mem_map] at hc; obtain ⟨f, _, rfl⟩ := hc; simp [hn]

/-- **YIELD alignment** for the graph procedures: one column per declared yield, all of
equal length (rows = labels / types / keys / constraints / 1). -/
theorem yields_labels (g : GraphView) :
    (dbLabels (F := F) g).length = (yieldsOf "db.labels").length ∧ Rect g.labels.length (dbLabels (F := F) g) := by
  refine ⟨rfl, ?_⟩; intro c hc; simp [dbLabels] at hc; subst hc; simp [strs]
theorem yields_types (g : GraphView) :
    (dbTypes (F := F) g).length = (yieldsOf "db.relationshipTypes").length ∧ Rect g.types.length (dbTypes (F := F) g) := by
  refine ⟨rfl, ?_⟩; intro c hc; simp [dbTypes] at hc; subst hc; simp [strs]
theorem yields_props (g : GraphView) :
    (dbProps (F := F) g).length = (yieldsOf "db.propertyKeys").length ∧ Rect g.attrs.length (dbProps (F := F) g) := by
  refine ⟨rfl, ?_⟩; intro c hc; simp [dbProps] at hc; subst hc; simp [strs]
theorem yields_meta (g : GraphView) :
    (dbMetaStats (F := F) g).length = (yieldsOf "db.meta.stats").length ∧ Rect 1 (dbMetaStats (F := F) g) := by
  refine ⟨rfl, ?_⟩; intro c hc; simp [dbMetaStats] at hc; rcases hc with h | h | h | h | h | h | h <;> subst h <;> rfl
theorem yields_constraints (g : GraphView) :
    (dbConstraints (F := F) g).length = (yieldsOf "db.constraints").length ∧
    Rect g.constraints.length (dbConstraints (F := F) g) := by
  refine ⟨rfl, ?_⟩; intro c hc; simp [dbConstraints] at hc
  rcases hc with h | h | h | h | h <;> subst h <;> simp
theorem yields_procedures (fns : List Ent) :
    (dbmsProcedures (F := F) fns).length = (yieldsOf "dbms.procedures").length ∧
    Rect (fns.filter isProc).length (dbmsProcedures (F := F) fns) := by
  refine ⟨rfl, ?_⟩; intro c hc; simp [dbmsProcedures] at hc
  rcases hc with h | h <;> subst h <;> simp [sortByName, List.length_mergeSort]
theorem yields_buildColumns (es : List Ent) :
    (buildColumns (F := F) es).length = (yieldsOf "dbms.functions").length ∧ Rect es.length (buildColumns (F := F) es) := by
  refine ⟨rfl, ?_⟩; intro c hc; simp [buildColumns] at hc; obtain ⟨k, _, rfl⟩ := hc; simp
theorem yields_indexes (g : GraphView) (b : Cols F) (h : dbIndexes g = some b) :
    b.length = (yieldsOf "db.indexes").length ∧ Rect g.indexes.length b := by
  simp only [dbIndexes, Option.map_eq_some_iff] at h
  obtain ⟨rows, hr, rfl⟩ := h
  refine ⟨by simp [yieldsOf, procRegs], ?_⟩
  intro c hc; simp at hc; obtain ⟨k, _, rfl⟩ := hc
  rw [List.length_map]; exact mapM_length hr

/-- The stubs: `createNodeIndex` yields no rows (so clauses after it never run — pinned by
test_effects_shapes.py:670), the others are errors whose bodies the planner bypasses. -/
theorem stubs : (ftCreate (F := F) = .ok []) ∧ (ftDrop (F := F)).toOption = none ∧
    (ftQueryNodes (F := F)).toOption = none ∧ (vecQueryNodes (F := F)).toOption = none ∧
    (vecQueryRels (F := F)).toOption = none ∧ (ftQueryRels (F := F)).toOption = none :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `db.labels` lists every label once, in label-id order. -/
theorem dbLabels_rows (g : GraphView) : dbLabels (F := F) g = [g.labels.map .str] := rfl

/-- The `types` list: exactly the index types present on the attribute, each once, in the
fixed RANGE < VECTOR < FULLTEXT order. -/
theorem typesOfFields_spec (fs : List Field) (t : IdxTy) :
    (match t with | .range => "RANGE" | .vector => "VECTOR" | .fulltext => "FULLTEXT") ∈ typesOfFields fs
      ↔ ∃ f ∈ fs, f.ty = t := by
  cases t <;> simp only [typesOfFields] <;> split <;> split <;> split <;> simp_all

theorem typesOfFields_sublist (fs : List Field) :
    (typesOfFields fs).Sublist ["RANGE", "VECTOR", "FULLTEXT"] := by
  simp only [typesOfFields]
  split <;> split <;> split <;> simp

/-- `fields[attr]` never panics iff every `field_order` key is in `fields` (the
`IndexInfo` invariant "field_order mirrors fields.keys()", index/mod.rs:217). -/
theorem typesOf_fold_isSome (fs : List (String × List Field)) (l : List String) (m0 : List (String × V F))
    (h : ∀ a ∈ l, (flookup fs a).isSome) :
    (l.foldlM (fun m a => (flookup fs a).map fun fl =>
      omInsert m a (.list (strs (typesOfFields fl)))) m0).isSome := by
  induction l generalizing m0 with
  | nil => rfl
  | cons a as ih =>
    obtain ⟨fl, hfl⟩ := Option.isSome_iff_exists.1 (h a (by simp))
    simp only [List.foldlM_cons, hfl, Option.map_some]
    exact ih _ (fun b hb => h b (List.mem_cons_of_mem _ hb))

theorem typesOf_isSome (i : IndexInfo) (h : ∀ a ∈ i.fieldOrder, (flookup i.fields a).isSome) :
    (typesOf (F := F) i).isSome := typesOf_fold_isSome _ _ _ h

/-- Vector options defaults (`similarityFunction` euclidean, `M` 16, `efConstruction`
200, `efRuntime` 10) and the `u64 → i64` dimension saturation. -/
theorem attrOpts_defaults (fs : List Field) (v : VecOpts) (hv : fs.findSome? (·.vec) = some v)
    (h : v.similarity = none ∧ v.m = none ∧ v.efConstruction = none ∧ v.efRuntime = none) :
    attrOpts (F := F) (some fs) = [("dimension", .int (dimToI64 v.dimension)),
      ("similarityFunction", .str "euclidean"), ("M", .int 16), ("efConstruction", .int 200),
      ("efRuntime", .int 10)] := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  simp [attrOpts, hv, h1, h2, h3, h4, omInsert]

theorem dimToI64_le (d : Nat) : dimToI64 d ≤ i64Max ∧ (d ≤ i64Max → dimToI64 d = d) := by
  unfold dimToI64; split <;> omega

/-- A non-vector attribute's options are `{}` (parity with C, live). -/
theorem attrOpts_novec (fs : List Field) (h : ∀ f ∈ fs, f.vec = none) : attrOpts (F := F) (some fs) = [] := by
  have : fs.findSome? (·.vec) = none := by
    rw [List.findSome?_eq_none_iff]; exact h
  simp [attrOpts, this]

theorem statusOf_spec (i : IndexInfo) :
    (i.pending ≤ 0 → statusOf i = "OPERATIONAL") ∧
    (0 < i.pending → statusOf i = s!"[Indexing] {i.progress}/{i.total}: UNDER CONSTRUCTION") := by
  unfold statusOf; constructor <;> intro h <;> split <;> first | rfl | omega

/-- `info.fields` always ends with `NONE_INDEXABLE_FIELDS`. -/
theorem infoNames_last (i : IndexInfo) : (infoNames i).getLast? = some "NONE_INDEXABLE_FIELDS" := by
  simp [infoNames]

/-- `db.meta.stats`: with distinct label names (the label table is a set), the `labels`
map is exactly `label ↦ count` in label-id order — no key is overwritten. -/
theorem omInsert_fresh (m : List (String × V F)) (k : String) (v : V F) (h : k ∉ m.map (·.1)) :
    omInsert m k v = m ++ [(k, v)] := by
  induction m with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k', v'⟩ := p
    simp only [List.map_cons, List.mem_cons, not_or] at h
    simp [omInsert, Ne.symm h.1, ih h.2]

theorem fold_fresh (cnt : Nat → Nat) (l : List (String × Nat)) (m : List (String × V F))
    (hd : (m.map (·.1) ++ l.map (·.1)).Nodup) :
    l.foldl (fun m (n, i) => omInsert m n (.int (cnt i))) m = m ++ l.map (fun (n, i) => (n, .int (cnt i))) := by
  induction l generalizing m with
  | nil => simp
  | cons p rest ih =>
    obtain ⟨n, i⟩ := p
    simp only [List.foldl_cons, List.map_cons]
    have hn : n ∉ m.map (·.1) := by
      intro hm; simp only [List.map_cons] at hd
      exact (List.nodup_append.1 hd).2.2 _ hm _ (by simp) rfl
    rw [omInsert_fresh _ _ _ hn, ih]
    · simp
    · simpa [List.map_append] using hd

theorem metaStats_labels (g : GraphView) (hd : g.labels.Nodup) :
    (dbMetaStats (F := F) g).head? = some [.map (g.labels.zipIdx.map fun (n, i) => (n, .int (g.labelCount i)))] := by
  simp only [dbMetaStats, List.head?_cons]
  rw [fold_fresh]
  · simp
  · simpa [List.zipIdx_map_fst] using hd

/-- `dbms.procedures`: names sorted, exactly the registered procedures (a permutation),
mode = WRITE iff registered as `write procedure:`. -/
theorem dbmsProcedures_sorted (fns : List Ent) :
    (sortByName (fns.filter isProc)).Pairwise (fun a b => nameLe a b = true) ∧
    (sortByName (fns.filter isProc)).Perm (fns.filter isProc) := by
  refine ⟨List.pairwise_mergeSort ?_ ?_ _, List.mergeSort_perm _ _⟩
  · intro a b c h1 h2; simp only [nameLe, decide_eq_true_eq] at *; exact String.le_trans h1 h2
  · intro a b; simp only [nameLe, Bool.or_eq_true, decide_eq_true_eq]; exact String.le_total _ _

/-- The hand-written merge is `List.merge` with "builtin first unless the UDF is strictly
smaller" … -/
theorem mergePicks_eq (bs us : List Ent) : mergePicks bs us = bs.merge us nameLe := by
  induction bs, us using mergePicks.induct with
  | case1 us => cases us <;> simp [mergePicks]
  | case2 bs h => cases bs <;> simp_all [mergePicks]
  | case3 b bs u us h ih =>
    have hn : ¬ (b.name ≤ u.name) := String.not_le.2 h
    rw [mergePicks, if_pos h, ih, List.cons_merge_cons]; simp [nameLe, hn]
  | case4 b bs u us h ih =>
    have hn : b.name ≤ u.name := String.not_lt.1 h
    rw [mergePicks, if_neg h, ih, List.cons_merge_cons]; simp [nameLe, hn]

/-- … so `dbms.functions` rows are sorted by name and are exactly built-ins ⊎ surviving UDFs. -/
theorem dbmsFunctions_merge_sorted (bs us : List Ent)
    (hb : bs.Pairwise (fun a b => nameLe a b = true)) (hu : us.Pairwise (fun a b => nameLe a b = true)) :
    (mergePicks bs us).Pairwise (fun a b => nameLe a b = true) ∧ (mergePicks bs us).Perm (bs ++ us) := by
  rw [mergePicks_eq]
  refine ⟨List.pairwise_merge ?_ ?_ _ _ hb hu, List.merge_perm_append _⟩
  · intro a b c h1 h2; simp only [nameLe, decide_eq_true_eq] at *; exact String.le_trans h1 h2
  · intro a b; simp only [nameLe, Bool.or_eq_true, decide_eq_true_eq]; exact String.le_total _ _

/-- The early return (:485) is the merge with no UDFs. -/
theorem mergePicks_nil (bs : List Ent) : mergePicks bs [] = bs := by cases bs <;> simp [mergePicks]

/-- A UDF shadowed (case-insensitively) by a built-in is dropped; others are kept. -/
theorem keepUdf_spec (lower : String → String) (bs : List Ent) (u : Ent) :
    keepUdf lower bs u = false ↔ ∃ b ∈ bs, lower b.name = lower u.name := by
  simp only [keepUdf, Bool.not_eq_false', List.contains_iff_mem, List.mem_map]

/-- `build_columns` flags: reducible = deterministic and neither aggregate nor procedure. -/
theorem buildRow_reducible (f : Ent) :
    (buildRow (F := F) f).getD 4 .null = .bool (!f.nonDet && !isAgg f && !isProc f) := rfl

/-- **Interning is invisible**: `type_to_dbms_value t` is `type_to_dbms_string t` as a
`Value::String` for every type (the `static`s only share the allocation). -/
theorem tyValue_eq : (t : Ty) → tyValue (F := F) t = .str (tyString t)
  | .optional t => by rw [tyValue, tyString]; exact tyValue_eq t
  | .union _ => rfl
  | .any | .null | .bool | .int | .float | .string | .list _ | .map | .node | .rel | .path
  | .vecf32 | .point | .datetime | .date | .time | .duration => rfl

/-- `format_union_strings` and `format_union` agree (two copies of one function). -/
theorem formatUnion_strings_eq (l : List String) : formatUnionStrings l = formatUnion l := rfl

set_option maxRecDepth 100000 in
/-- The `Any` spelling — matches C live, byte for byte. -/
theorem any_string : tyString .any =
    "Map, Node, Edge, List, Path, Datetime, Date, Time, Duration, String, Boolean, Integer, Float, Null, Pointer, Point, or Vectorf32" := by
  rfl

example : formatUnion ["Integer", "Float", "Null"] = "Integer, Float, or Null" := rfl

-- Every spelling a type collects is one of C's 17 type names.
mutual
theorem collectTys_mem : (t : Ty) → ∀ x ∈ collectTys t, x ∈ anyNames
  | .union ts => by simpa [collectTys] using collectTysL_mem ts
  | .optional t => by simpa [collectTys] using collectTys_mem t
  | .any => by simp [collectTys]
  | .null | .bool | .int | .float | .string | .list _ | .map | .node | .rel | .path | .vecf32
  | .point | .datetime | .date | .time | .duration => by
      intro x hx; simp [collectTys, leafString] at hx; subst hx; decide
theorem collectTysL_mem : (ts : List Ty) → ∀ x ∈ collectTysL ts, x ∈ anyNames
  | [] => by simp [collectTysL]
  | t :: ts => by
      intro x hx
      simp only [collectTysL, List.mem_append] at hx
      exact hx.elim (collectTys_mem t x) (collectTysL_mem ts x)
end

/-- `typePos` is injective on C's type names. -/
theorem typePos_inj : ∀ a ∈ anyNames, ∀ b ∈ anyNames, typePos a = typePos b → a = b := by
  unfold anyNames typePos; decide

theorem mem_dedupAdj : ∀ (l : List String) (x : String), x ∈ dedupAdj l ↔ x ∈ l
  | [], x => by simp [dedupAdj]
  | [a], x => by simp [dedupAdj]
  | a :: b :: t, x => by
      unfold dedupAdj
      by_cases h : a = b
      · subst h; simp only [if_true, mem_dedupAdj (a :: t) x]; simp
      · simp only [h, if_false, List.mem_cons, mem_dedupAdj (b :: t) x]

theorem dedupAdj_strict : ∀ (l : List String), (∀ x ∈ l, x ∈ anyNames) →
    l.Pairwise (fun a b => typePos a ≤ typePos b) →
    (dedupAdj l).Pairwise (fun a b => typePos a < typePos b)
  | [], _, _ => by simp [dedupAdj]
  | [a], _, _ => by simp [dedupAdj]
  | a :: b :: t, hm, hp => by
      unfold dedupAdj
      have hp' := List.pairwise_cons.1 hp
      have hbt : (b :: t).Pairwise (fun a b => typePos a ≤ typePos b) := hp'.2
      have hm' : ∀ x ∈ b :: t, x ∈ anyNames := fun x hx => hm x (List.mem_cons_of_mem _ hx)
      have ih := dedupAdj_strict (b :: t) hm' hbt
      by_cases h : a = b
      · simp [h]; exact ih
      · simp only [h, if_false]
        refine List.pairwise_cons.2 ⟨?_, ih⟩
        intro x hx
        rw [mem_dedupAdj] at hx
        have hax : typePos a ≤ typePos x := hp'.1 x hx
        have hbx : typePos b ≤ typePos x ∨ x = b := by
          rcases List.mem_cons.1 hx with rfl | ht
          · exact Or.inr rfl
          · exact Or.inl ((List.pairwise_cons.1 hbt).1 x ht)
        have hab : typePos a ≤ typePos b := hp'.1 b (List.mem_cons_self ..)
        refine Nat.lt_of_le_of_ne hax fun heq => h ?_
        have hbe : typePos b = typePos a := by
          rcases hbx with hb | rfl
          · omega
          · exact heq.symm
        exact typePos_inj a (hm a (List.mem_cons_self ..)) b (hm b (List.mem_cons_of_mem _ (List.mem_cons_self ..)))
          hbe.symm

/-- The canonical member list of any list of C type names: strictly increasing in C's
fixed order, with exactly the same members. -/
theorem canon_props (l : List String) (hm : ∀ x ∈ l, x ∈ anyNames) :
    (canonTypes l).Pairwise (fun a b => typePos a < typePos b) ∧
    ∀ x, x ∈ canonTypes l ↔ x ∈ l := by
  have hmem : ∀ x ∈ sortByPos l, x ∈ anyNames := fun x hx => hm x ((List.mem_mergeSort).1 hx)
  refine ⟨dedupAdj_strict _ hmem ?_, fun x => ?_⟩
  · have := List.pairwise_mergeSort (le := fun a b => decide (typePos a ≤ typePos b))
      (fun a b c h1 h2 => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) l
    exact this.imp (fun h => by simpa using h)
  · unfold canonTypes
    rw [mem_dedupAdj]; exact List.mem_mergeSort

/-- A strictly C-ordered list with the right members IS the canonical list. -/
theorem canon_eq_of (l c : List String) (hm : ∀ x ∈ l, x ∈ anyNames)
    (hc : c.Pairwise (fun a b => typePos a < typePos b)) (hiff : ∀ x, x ∈ c ↔ x ∈ l) :
    canonTypes l = c := by
  have h := canon_props l hm
  have nd1 : (canonTypes l).Nodup := h.1.imp (fun h e => by subst e; omega)
  have nd2 : c.Nodup := hc.imp (fun h e => by subst e; omega)
  have hp : (canonTypes l).Perm c := by
    rw [List.perm_ext_iff_of_nodup nd1 nd2]
    intro x; rw [h.2, hiff]
  exact List.Perm.eq_of_pairwise (fun a b _ _ h1 h2 => by omega) h.1 hc hp

/-- **#3280 (fixed): the union spelling is C's.** For any union, the member list that
`type_to_dbms_string` formats is strictly increasing in C's fixed type order
(`DBMS_TYPE_ORDER`, so no duplicates), whatever order or nesting the signature declares,
and it contains exactly the single types the union stands for. -/
theorem union_order_is_c (ts : List Ty) :
    (canonTypes (collectTysL ts)).Pairwise (fun a b => typePos a < typePos b) ∧
    ∀ x, x ∈ canonTypes (collectTysL ts) ↔ x ∈ collectTysL ts :=
  canon_props _ (collectTysL_mem ts)

/-- Declaration order is irrelevant: the spelling of a union depends only on the set of
types it stands for (the former divergence `Point or Null` vs C `Null or Point` is gone). -/
theorem union_perm_invariant (ts us : List Ty) (h : ∀ x, x ∈ collectTysL ts ↔ x ∈ collectTysL us) :
    tyString (.union ts) = tyString (.union us) := by
  have h2 := union_order_is_c us
  have : canonTypes (collectTysL ts) = canonTypes (collectTysL us) :=
    canon_eq_of _ _ (collectTysL_mem ts) h2.1 (fun x => by rw [h2.2, h])
  simp only [tyString, this]

/-- Spelling of a union whose collected members are exactly `c` (strictly C-ordered). -/
theorem union_spelling (ts : List Ty) (c : List String)
    (hc : c.Pairwise (fun a b => typePos a < typePos b)) (h : ∀ x, x ∈ c ↔ x ∈ collectTysL ts) :
    tyString (.union ts) = formatUnion c := by
  simp only [tyString, formatUnionStrings, canon_eq_of _ c (collectTysL_mem ts) hc h]

/-- The `dbms.functions()` rows that used to differ from C now match it:
`distance`/`point` (`Point or Null` -> `Null or Point`) ... -/
theorem union_point_null :
    tyString (.union [.point, .null]) = "Null or Point" ∧
    tyString (.union [.null, .point]) = "Null or Point" :=
  ⟨union_spelling _ ["Null", "Point"] (by decide)
      (fun x => by simp [collectTysL, collectTys, leafString]; grind),
   union_spelling _ ["Null", "Point"] (by decide)
      (fun x => by simp [collectTysL, collectTys, leafString])⟩

/-- ... and `tofloat` (`String, Float, Integer, or Null` -> `String, Integer, Float, or Null`). -/
theorem union_tofloat :
    tyString (.union [.string, .float, .int, .null]) = "String, Integer, Float, or Null" :=
  union_spelling _ ["String", "Integer", "Float", "Null"] (by decide)
    (fun x => by simp [collectTysL, collectTys, leafString]; grind)

/-- Nested unions and `Optional` flatten; duplicates collapse. -/
theorem union_nested :
    tyString (.union [.union [.int, .null], .optional .int, .float]) = "Integer, Float, or Null" :=
  union_spelling _ ["Integer", "Float", "Null"] (by decide)
    (fun x => by simp [collectTysL, collectTys, leafString]; grind)

/-- `Any` inside a union expands to all 17 types: the same text as `Any` itself. -/
theorem union_any : tyString (.union [.int, .any]) = tyString .any :=
  union_spelling _ anyNames (by decide)
    (fun x => by
      simp only [collectTysL, collectTys, leafString, List.append_nil, List.mem_append,
        List.mem_singleton]
      constructor
      · intro h; exact Or.inr h
      · rintro (rfl | h)
        · decide
        · exact h)

/-- Historical (pre-#3280): Rust spelled a union in declaration order (`Point or Null`).
Kept as the old model. -/
def pre3280TyString : Ty → String
  | .union ts => formatUnionStrings (ts.map leafString)
  | t => tyString t

theorem pre3280_declaration_order :
    pre3280TyString (.union [.point, .null]) = "Point or Null" ∧
    pre3280TyString (.union [.null, .point]) = "Null or Point" := ⟨rfl, rfl⟩

/-! ## path.rs -/

/-- `nodes` (:29) / `relationships` (:52): filter the flat path; `none` = `unreachable!()`. -/
def pathNodes : List (V F) → Option (V F)
  | .path vs :: _ => some (.list (vs.filter fun v => match v with | .node _ => true | _ => false))
  | .null :: _ => some .null
  | _ => none
def pathRels : List (V F) → Option (V F)
  | .path vs :: _ => some (.list (vs.filter fun v => match v with | .rel _ => true | _ => false))
  | .null :: _ => some .null
  | _ => none

/-- `register` (path.rs:25). -/
def pathSigs : List (String × List Ty × Ty) :=
  [("nodes", [.union [.path, .null]], .union [.list .node, .null]),
   ("relationships", [.union [.path, .null]], .union [.list .rel, .null])]

/-- A well-formed path `n₀ r₁ n₁ … rₖ nₖ`. -/
def wfPath : List Nat → List Nat → List (V F)
  | [n], [] => [.node n]
  | n :: ns, r :: rs => .node n :: .rel r :: wfPath ns rs
  | _, _ => []

theorem path_split (ns rs : List Nat) (h : ns.length = rs.length + 1) :
    pathNodes [.path (wfPath (F := F) ns rs)] = some (.list (ns.map .node)) ∧
    pathRels [.path (wfPath (F := F) ns rs)] = some (.list (rs.map .rel)) := by
  induction rs generalizing ns with
  | nil =>
    match ns, h with
    | [n], _ => simp [pathNodes, pathRels, wfPath]
  | cons r rs ih =>
    match ns, h with
    | n :: ns, h =>
      have := ih ns (by simpa using h)
      simp only [pathNodes, pathRels, Option.some.injEq, V.list.injEq] at this
      simp [pathNodes, pathRels, wfPath, this.1, this.2]

/-- So `size(nodes(p)) = size(relationships(p)) + 1 = length(p) + 1`. -/
theorem path_lengths (ns rs : List Nat) (h : ns.length = rs.length + 1) :
    ∃ a b, pathNodes [.path (wfPath (F := F) ns rs)] = some (.list a) ∧
      pathRels [.path (wfPath (F := F) ns rs)] = some (.list b) ∧ a.length = b.length + 1 :=
  ⟨_, _, (path_split ns rs h).1, (path_split ns rs h).2, by simp [h]⟩

theorem path_null : pathNodes [(.null : V F)] = some .null ∧ pathRels [(.null : V F)] = some .null :=
  ⟨rfl, rfl⟩

/-- The `unreachable!()` arms are dead after `validate_args_type` with `Path | Null`. -/
theorem path_no_panic (v : V F) (h : v = .null ∨ ∃ vs, v = .path vs) :
    (pathNodes [v]).isSome ∧ (pathRels [v]).isSome := by
  rcases h with rfl | ⟨vs, rfl⟩ <;> simp [pathNodes, pathRels]

/-! ## entity_type.rs -/

theorem entityType_fmt_inj (a b : EntityType) (h : a.fmt = b.fmt) : a = b := by
  cases a <;> cases b <;> simp_all [EntityType.fmt]

/-- The spellings C uses in `db.indexes`/`db.constraints` (live: `NODE`, `RELATIONSHIP`). -/
theorem entityType_fmt_values : EntityType.node.fmt = "NODE" ∧ EntityType.relationship.fmt = "RELATIONSHIP" :=
  ⟨rfl, rfl⟩

end ValueMath.Proc
