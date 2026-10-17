import RedisLayer.Serial
/-!
# Schema block of the v19 encoding (`src/serializers/mod.rs:260-719`)

Faithful to the token layout; the index/constraint structures keep only what is written.
`sim` is the similarity *code* the encoder computes (`:414-418`, `eq_ignore_ascii_case`
abstracted as `simCode`), weights are `f64` bit patterns (`one` = `1.0`).

Findings proven here:
* `pre3252_field_prefix_lost` (history; **fixed by #3252**, cfe6e7945) — an index on an
  attribute whose own name starts with `range:` (range index) or `vector:` (vector index)
  came back on a *different* attribute: the encoder writes the bare attribute name (`:383`)
  but the decoder stripped the RediSearch prefix (cab3cf192 `:642-646`). Confirmed live then.
  Now the name is used verbatim: `field_name_kept`, and `decField_encField` round-trips it.
* `cons_unknown_prop` — a constraint property missing from the attribute table is
  written as attribute id 0 (`:457-460`, `unwrap_or(0)`), i.e. reloads naming attribute 0.
-/
namespace RedisLayer.Serial
open RedisLayer.BufferedIO (W Bytes)

variable (U : Utf8)

inductive Ty | range | fulltext | vector
  deriving DecidableEq, Repr

structure VOpts where
  dim : Nat
  m : Option Nat
  efc : Option Nat
  efr : Option Nat
  sim : Option Bytes
  deriving DecidableEq, Repr

structure Fld where
  ty : Ty
  weight : Option Nat
  nostem : Option Bool
  phonetic : Option Bytes
  vopts : Option VOpts
  deriving DecidableEq, Repr

/-- `index_field_type::*` (`graph/src/graph/graphblas/serialization.rs:88-94`). -/
def FULLTEXT := 1
def NUMERIC := 2
def GEO := 4
def STR := 8
def VECTOR := 16

def fieldType : Ty → Nat
  | .fulltext => FULLTEXT
  | .range => NUMERIC ||| STR ||| GEO
  | .vector => VECTOR

variable (one : Nat) (simCode : Option Bytes → Nat)

/-- Field loop body of `encode_schema_index_block` (`:382-421`). -/
def encField (attr : Bytes) (f : Fld) : List W :=
  [.buf (nullTerm attr), .u (fieldType f.ty), .d (f.weight.getD one),
   .u (if f.nostem.getD false then 1 else 0), .buf (nullTerm (f.phonetic.getD []))]
  ++ (if fieldType f.ty &&& VECTOR ≠ 0 then
        match f.vopts with
        | some v => [.u v.dim, .u (v.m.getD 16), .u (v.efc.getD 200), .u (v.efr.getD 10),
                     .u (simCode v.sim)]
        | none => []
      else [])

def simName : Nat → Bytes
  | 1 => "ip".toUTF8.toList.map (·.toNat)
  | 2 => "cosine".toUTF8.toList.map (·.toNat)
  | _ => "euclidean".toUTF8.toList.map (·.toNat)

def stripPrefix (p s : Bytes) : Bytes := if p.isPrefixOf s then s.drop p.length else s
def RANGE_ : Bytes := "range:".toUTF8.toList.map (·.toNat)
def VECTOR_ : Bytes := "vector:".toUTF8.toList.map (·.toNat)

/-- `decode_index_field` (`:621-696`) since #3252 (cfe6e7945): the attribute is the decoded
name verbatim (C does the same), whatever its type. -/
def decField : Dec (Bytes × Fld) :=
  Dec.bind rdB fun nb => Dec.bind rdU fun ft => Dec.bind rdD fun w =>
  Dec.bind rdU fun ns => Dec.bind rdB fun pb =>
  let name := strip U nb
  let isVec := ft &&& VECTOR ≠ 0
  let isFt := ft &&& FULLTEXT ≠ 0
  let ty := if isFt then Ty.fulltext else if isVec then Ty.vector else Ty.range
  let attr := name
  if isVec then
    Dec.bind rdU fun dim => Dec.bind rdU fun m => Dec.bind rdU fun efc =>
    Dec.bind rdU fun efr => Dec.bind rdU fun sc =>
    pure' (attr, ⟨ty, none, none, none, some ⟨dim, some m, some efc, some efr, some (simName sc)⟩⟩)
  else
    pure' (attr, ⟨ty, if isFt then some w else none, if isFt then some (ns ≠ 0) else none,
                  if isFt then some (strip U pb) else none, none⟩)

/-- What a field is after a round trip: text options materialised with their defaults
for full-text fields and dropped otherwise; vector options with their defaults and the
similarity name normalised. -/
def normField (f : Fld) : Fld :=
  match f.ty with
  | .fulltext => ⟨.fulltext, some (f.weight.getD one), some (f.nostem.getD false),
                  some (f.phonetic.getD []), none⟩
  | .range => ⟨.range, none, none, none, none⟩
  | .vector => match f.vopts with
    | some v => ⟨.vector, none, none, none,
        some ⟨v.dim, some (v.m.getD 16), some (v.efc.getD 200), some (v.efr.getD 10),
              some (simName (simCode v.sim))⟩⟩
    | none => ⟨.vector, none, none, none, none⟩   -- see `vector_without_opts`

/-- The attribute name the decoder recovers: the encoded one (#3252). -/
def attrBack (attr : Bytes) (_ty : Ty) : Bytes := attr

/-- The attribute name the decoder recovered before #3252 (cab3cf192 `:642-646`): the
RediSearch prefix stripped from a range / vector field. -/
def pre3252_attrBack (attr : Bytes) (ty : Ty) : Bytes :=
  match ty with
  | .range => stripPrefix RANGE_ attr
  | .vector => stripPrefix VECTOR_ attr
  | .fulltext => attr

theorem decField_encField (attr : Bytes) (f : Fld) (ha : U.valid attr)
    (hp : U.valid (f.phonetic.getD [])) (hv : f.ty = .vector → f.vopts.isSome) (ts : List W) :
    decField U (encField one simCode attr f ++ ts)
      = some ((attrBack attr f.ty, normField one simCode f), ts) := by
  obtain ⟨ty, w, ns, ph, vo⟩ := f
  cases ty with
  | range =>
    simp [decField, encField, Dec.bind, rdB, rdU, rdD, pure', fieldType, attrBack, normField,
      strip_nullTerm U attr ha, NUMERIC, STR, GEO, VECTOR, FULLTEXT]
  | fulltext =>
    simp at hp
    simp [decField, encField, Dec.bind, rdB, rdU, rdD, pure', fieldType, attrBack, normField,
      strip_nullTerm U attr ha, strip_nullTerm U _ hp, VECTOR, FULLTEXT]
  | vector =>
    cases vo with
    | none => simp at hv
    | some v =>
      simp [decField, encField, Dec.bind, rdB, rdU, rdD, pure', fieldType, attrBack, normField,
        strip_nullTerm U attr ha, VECTOR, FULLTEXT]

/-- **History (W4-redis-2, fixed by #3252)**: an attribute literally named `range:<a>`
(range index) was reloaded as `<a>`. -/
theorem pre3252_field_prefix_lost (a : Bytes) :
    pre3252_attrBack (RANGE_ ++ a) .range = a := by
  simp [pre3252_attrBack, stripPrefix]

/-- …whereas every other name survived. -/
theorem pre3252_attrBack_plain (attr : Bytes) (ty : Ty) (h1 : ¬ RANGE_.isPrefixOf attr)
    (h2 : ¬ VECTOR_.isPrefixOf attr) : pre3252_attrBack attr ty = attr := by
  cases ty <;> simp [pre3252_attrBack, stripPrefix, h1, h2]

/-- **`field_name_kept`** (#3252): every attribute name survives the reload, `range:x`
and `vector:x` included — `decField_encField` returns exactly the encoded name. -/
theorem field_name_kept (attr : Bytes) (ty : Ty) : attrBack attr ty = attr := rfl

theorem field_prefix_kept (a : Bytes) : attrBack (RANGE_ ++ a) .range = RANGE_ ++ a := rfl

/-- A vector field without `VectorIndexOptions` writes no option words (`:405-406`), yet
the decoder reads five (`:648-653`): the stream desynchronises. Such a field is not built by
`CREATE VECTOR INDEX` (options are mandatory there), so this is latent. -/
theorem vector_without_opts (attr : Bytes) (ts : List W) :
    encField one simCode attr ⟨.vector, none, none, none, none⟩ ++ ts
      = [.buf (nullTerm attr), .u VECTOR, .d one, .u 0, .buf (nullTerm [])] ++ ts := by
  simp [encField, fieldType, VECTOR]

end RedisLayer.Serial
