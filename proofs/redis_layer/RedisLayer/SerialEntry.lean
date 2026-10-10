import RedisLayer.SerialSchema
/-!
# Schema entries, constraint blocks and the whole schema (`src/serializers/mod.rs`)

`decSchema_encSchema`: decoding what `Schema::encode` wrote gives back the schema up to an
explicit normalisation `normSchema` — the attribute table, label and type lists exactly;
per label/type, its indexes merged into one `IndexInfo` (language and stop words of the
first, fields grouped by attribute in first-seen order, field options as in `normField`);
only `Operational` constraints, grouped under their label/type, all loading as
`Operational`; indexes and constraints of a label/type not in the lists are dropped.
-/
namespace RedisLayer.Serial
open RedisLayer.BufferedIO (W Bytes)

variable (U : Utf8) (one : Nat) (simCode : Option Bytes → Nat)

structure Info where
  label : Bytes
  isRel : Bool
  fields : List (Bytes × List Fld)   -- `field_order` with `fields[attr]`
  language : Option Bytes
  stopwords : Option (List Bytes)
  deriving DecidableEq, Repr

structure Cons where
  unique : Bool
  isRel : Bool
  label : Bytes
  props : List Bytes
  operational : Bool
  deriving DecidableEq, Repr

def ENGLISH : Bytes := "english".toUTF8.toList.map (·.toNat)

def allFields (is : List Info) : List (Bytes × Fld) :=
  (is.map fun i => (i.fields.map fun (a, fs) => fs.map fun f => (a, f)).flatten).flatten

/-- `encode_schema_index_block` (`:337-422`). -/
def encIdx (is : List Info) : List W :=
  match is with
  | [] => [.u 0]
  | i :: _ =>
    let sw := i.stopwords.getD []
    [.u 1, .buf (nullTerm (i.language.getD ENGLISH)), .u sw.length]
      ++ (sw.map fun s => [W.buf (nullTerm s)]).flatten
      ++ [.u (allFields is).length]
      ++ ((allFields is).map fun (a, f) => encField one simCode a f).flatten

/-- `position(..).unwrap_or(0)` (`:457-460`). -/
def pos (l : List Bytes) (p : Bytes) : Option Nat :=
  match l with
  | [] => none
  | a :: as => if a = p then some 0 else (pos as p).map (· + 1)

def attrId (attrs : List Bytes) (p : Bytes) : Nat := (pos attrs p).getD 0

/-- `encode_constraint_block` (`:439-464`). -/
def encCons (attrs : List Bytes) (cs : List Cons) : List W :=
  let act := cs.filter (·.operational)
  [.u act.length] ++ (act.map fun c =>
    [W.u (if c.unique then 0 else 1), .u c.props.length] ++ c.props.map fun p => W.u (attrId attrs p)).flatten

/-- Grouping by attribute, as the decoder's `fields` / `field_order` pair does (`:551-559`). -/
def addField (acc : List (Bytes × List Fld)) (af : Bytes × Fld) : List (Bytes × List Fld) :=
  if acc.any (·.1 = af.1) then acc.map fun (a, fs) => if a = af.1 then (a, fs ++ [af.2]) else (a, fs)
  else acc ++ [(af.1, [af.2])]

def group (l : List (Bytes × Fld)) : List (Bytes × List Fld) := l.foldl addField []

def attrName (i : Nat) : Bytes := "attr_".toUTF8.toList.map (·.toNat) ++ (toString i).toUTF8.toList.map (·.toNat)

def lookupAttr (attrs : List Bytes) (i : Nat) : Bytes :=
  match attrs[i]? with
  | some a => a
  | none => attrName i

def decCons1 (attrs : List Bytes) (name : Bytes) : Dec Cons :=
  Dec.bind rdU fun ct => Dec.bind rdU fun n => Dec.bind (rdMany rdU n) fun ids =>
  pure' ⟨decide (ct = 0), false, name, ids.map (lookupAttr attrs), true⟩

/-- The index half of `decode_schema_entry` (`:537-582`). -/
def decIdx : Dec (Option Info) :=
  Dec.bind rdU fun has =>
  if has ≠ 0 then
    Dec.bind rdB fun lb => Dec.bind rdU fun swc => Dec.bind (rdMany rdB swc) fun sws =>
    Dec.bind rdU fun fc => Dec.bind (rdMany (decField U) fc) fun fs =>
    pure' (some (⟨[], false, group fs, some (strip U lb),
      if sws = [] then none else some (sws.map (strip U))⟩ : Info))
  else pure' none

/-- `decode_schema_entry` (`:529-619`); `label`/`isRel` are stamped by the caller. -/
def decEntry (attrs : List Bytes) : Dec (Bytes × Option Info × List Cons) :=
  Dec.bind rdU fun _id => Dec.bind rdB fun nb =>
  let name := strip U nb
  Dec.bind (decIdx U) fun info =>
  Dec.bind rdU fun cc => Dec.bind (rdMany (decCons1 attrs name) cc) fun cs =>
  pure' (name, info, cs)

structure Schema where
  attrs : List Bytes
  labels : List Bytes
  types : List Bytes
  infos : List Info
  cons : List Cons
  deriving Repr

def encEntry (s : Schema) (rel : Bool) (il : Nat × Bytes) : List W :=
  [.u il.1, .buf (nullTerm il.2)]
    ++ encIdx one simCode (s.infos.filter fun i => i.label = il.2 && i.isRel = rel)
    ++ encCons s.attrs (s.cons.filter fun c => c.isRel = rel && c.label = il.2)

def enumFrom (n : Nat) : List Bytes → List (Nat × Bytes)
  | [] => []
  | l :: ls => (n, l) :: enumFrom (n+1) ls

/-- `Schema::encode` (`:260-331`). -/
def encSchema (s : Schema) : List W :=
  [.u s.attrs.length] ++ (s.attrs.map fun a => [W.buf (nullTerm a)]).flatten
  ++ [.u s.labels.length] ++ ((enumFrom 0 s.labels).map (encEntry one simCode s false)).flatten
  ++ [.u s.types.length] ++ ((enumFrom 0 s.types).map (encEntry one simCode s true)).flatten

def stamp (rel : Bool) (e : Bytes × Option Info × List Cons) : List Info × List Cons :=
  ((e.2.1.map fun i => { i with label := e.1, isRel := rel }).toList,
   e.2.2.map fun c => { c with isRel := rel })

/-- `Schema::decode` (`:466-527`). -/
def decSchema : Dec Schema :=
  Dec.bind rdU fun ac => Dec.bind (rdMany rdB ac) fun ab =>
  let attrs := ab.map (strip U)
  Dec.bind rdU fun nl => Dec.bind (rdMany (decEntry U attrs) nl) fun ns =>
  Dec.bind rdU fun nt => Dec.bind (rdMany (decEntry U attrs) nt) fun rs =>
  pure' ⟨attrs, ns.map (·.1), rs.map (·.1),
    (ns.map fun e => (stamp false e).1).flatten ++ (rs.map fun e => (stamp true e).1).flatten,
    (ns.map fun e => (stamp false e).2).flatten ++ (rs.map fun e => (stamp true e).2).flatten⟩

/-! ## Normal form -/

def mergeInfo (l : Bytes) (rel : Bool) (is : List Info) : Option Info :=
  match is with
  | [] => none
  | i :: _ =>
    let sw := i.stopwords.getD []
    some ⟨l, rel, group ((allFields is).map fun (a, f) => (attrBack a f.ty, normField one simCode f)),
      some (i.language.getD ENGLISH), if sw = [] then none else some sw⟩

def normCons (c : Cons) : Cons := { c with operational := true }

def normSchema (s : Schema) : Schema :=
  ⟨s.attrs, s.labels, s.types,
   (s.labels.map fun l => (mergeInfo one simCode l false
      (s.infos.filter fun i => i.label = l && i.isRel = false)).toList).flatten
   ++ (s.types.map fun l => (mergeInfo one simCode l true
      (s.infos.filter fun i => i.label = l && i.isRel = true)).toList).flatten,
   (s.labels.map fun l => ((s.cons.filter fun c => c.isRel = false && c.label = l).filter
      (·.operational)).map normCons).flatten
   ++ (s.types.map fun l => ((s.cons.filter fun c => c.isRel = true && c.label = l).filter
      (·.operational)).map normCons).flatten⟩

end RedisLayer.Serial
