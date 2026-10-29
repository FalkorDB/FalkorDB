import RedisLayer.Serial
/-!
# Schema block of the v19 encoding (`src/serializers/mod.rs:262-743`)

Faithful to the token layout; the index/constraint structures keep only what is written.
`sim` is the similarity *code* the encoder computes (`:416-424`, `eq_ignore_ascii_case`
abstracted as `simCode`), weights are `f64` bit patterns (`one` = `1.0`).

Findings proven here:
* Field-prefix loss (counterexample removed; in 17ccc5412; **fixed by #3252**, cfe6e7945) — an index on an
  attribute whose own name starts with `range:` (range index) or `vector:` (vector index)
  came back on a *different* attribute: the encoder writes the bare attribute name (`:384`)
  but the decoder stripped the RediSearch prefix (cab3cf192 `:642-646`). Confirmed live then.
  Now the name is used verbatim: `field_name_kept`, and `decField_encField` round-trips it.
* Multi-type index loss (counterexamples removed; in 17ccc5412; **fixed by #3514**, a0ba12950, closes #3484) — see below and
  `SerialCLayout`: C writes ONE field entry per attribute with every type in the bitmask, but
  `decode_index_field` kept one type, so an attribute with a RANGE and a FULLTEXT or VECTOR index
  lost its RANGE index on loading a C-written RDB. Main now decodes every set type into its own
  field (`decField` returns a list), refuses a word with no type or an unknown bit, reads C's
  phonetic `"no"` as off, and writes off as `"no"` (an empty string read as ON by C).
  `decField_encField` (Rust round trip), `SerialCLayout.decField_encFieldC` (every C entry).
* `cons_unknown_prop` — a constraint property missing from the attribute table is
  written as attribute id 0 (`:462-466`, `unwrap_or(0)`), i.e. reloads naming attribute 0.
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

/-- The phonetic string written (`:402-406`, #3514): an absent or empty code is written as
C's `"no"` (`PHONETIC_OFF`), because C enables phonetic matching for any other string, an empty
one included. Before #3514 it wrote the empty string (`pre3514_encField`). -/
def encPh (f : Fld) : Bytes := if f.phonetic.getD [] = [] then PHON_OFF else f.phonetic.getD []

/-- Field loop body of `encode_schema_index_block` (`:383-427`). -/
def encField (attr : Bytes) (f : Fld) : List W :=
  [.buf (nullTerm attr), .u (fieldType f.ty), .d (f.weight.getD one),
   .u (if f.nostem.getD false then 1 else 0), .buf (nullTerm (encPh f))]
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

/-- `RANGE` bits of the type word (`NUMERIC | GEO | STR`). -/
def RANGEW : Nat := NUMERIC ||| STR ||| GEO

/-- Every bit the loader knows (`RANGE | FULLTEXT | VECTOR`). -/
def ANYW : Nat := RANGEW ||| FULLTEXT ||| VECTOR

def failD {α} : Dec α := fun _ => none

/-- `phonetic == PHONETIC_OFF → clear` (`:662-665`). -/
def phonNorm (p : Bytes) : Bytes := if p = PHON_OFF then [] else p

/-- **`decode_index_field` on main** (`:637-725`, #3514): the type word is a bitmask and every
set type becomes its own field, in `db.indexes()` order (range, fulltext, vector); a word with
no type or an unknown bit is an error (`:668-672`); C's phonetic `"no"` comes back as our
explicit off (`:662-665`). The attribute is the name verbatim (#3252). -/
def decField : Dec (Bytes × List Fld) :=
  Dec.bind rdB fun nb => Dec.bind rdU fun ft => Dec.bind rdD fun w =>
  Dec.bind rdU fun ns => Dec.bind rdB fun pb =>
  if ft = 0 ∨ ft &&& (ANYW ^^^ 0xFFFFFFFFFFFFFFFF) ≠ 0 then failD else
  let isR := ft &&& RANGEW ≠ 0
  let isFt := ft &&& FULLTEXT ≠ 0
  let isV := ft &&& VECTOR ≠ 0
  let rest : List Fld :=
    (if isR then [⟨.range, none, none, none, none⟩] else [])
    ++ (if isFt then [⟨.fulltext, some w, some (ns ≠ 0), some (phonNorm (strip U pb)), none⟩] else [])
  if isV then
    Dec.bind rdU fun dim => Dec.bind rdU fun m => Dec.bind rdU fun efc =>
    Dec.bind rdU fun efr => Dec.bind rdU fun sc =>
    pure' (strip U nb, rest ++ [⟨.vector, none, none, none,
      some ⟨dim, some m, some efc, some efr, some (simName sc)⟩⟩])
  else pure' (strip U nb, rest)

/-- What a field is after a round trip: text options materialised with their defaults
for full-text fields and dropped otherwise; vector options with their defaults and the
similarity name normalised. -/
def normField (f : Fld) : Fld :=
  match f.ty with
  | .fulltext => ⟨.fulltext, some (f.weight.getD one), some (f.nostem.getD false),
                  some (phonNorm (f.phonetic.getD [])), none⟩
  | .range => ⟨.range, none, none, none, none⟩
  | .vector => match f.vopts with
    | some v => ⟨.vector, none, none, none,
        some ⟨v.dim, some (v.m.getD 16), some (v.efc.getD 200), some (v.efr.getD 10),
              some (simName (simCode v.sim))⟩⟩
    | none => ⟨.vector, none, none, none, none⟩   -- see `vector_without_opts`

/-- The attribute name the decoder recovers: the encoded one (#3252). -/
def attrBack (attr : Bytes) (_ty : Ty) : Bytes := attr

theorem phonNorm_encPh (f : Fld) : phonNorm (encPh f) = phonNorm (f.phonetic.getD []) := by
  unfold encPh
  by_cases h : f.phonetic.getD [] = []
  · simp [h, phonNorm]
  · simp [h]

theorem encPh_valid (f : Fld) (hp : U.valid (f.phonetic.getD [])) : U.valid (encPh f) := by
  unfold encPh
  by_cases h : f.phonetic.getD [] = []
  · simp [h, U.valid_no]
  · simp [h, hp]

/-- **Round trip on main** (#3252 name verbatim, #3514 phonetic `"no"`): the decoder returns
exactly the one field Rust's encoder wrote (Rust writes one entry per type). -/
theorem decField_encField (attr : Bytes) (f : Fld) (ha : U.valid attr)
    (hp : U.valid (f.phonetic.getD [])) (hv : f.ty = .vector → f.vopts.isSome) (ts : List W) :
    decField U (encField one simCode attr f ++ ts)
      = some ((attrBack attr f.ty, [normField one simCode f]), ts) := by
  obtain ⟨ty, w, ns, ph, vo⟩ := f
  have hq := encPh_valid U ⟨ty, w, ns, ph, vo⟩ hp
  have hn := phonNorm_encPh ⟨ty, w, ns, ph, vo⟩
  cases ty with
  | range =>
    simp [decField, encField, Dec.bind, rdB, rdU, rdD, pure', fieldType, attrBack, normField,
      strip_nullTerm U attr ha, ANYW, RANGEW, NUMERIC, STR, GEO, VECTOR, FULLTEXT]
  | fulltext =>
    have e1 := strip_nullTerm U attr ha
    have e2 := strip_nullTerm U _ hq
    simp [decField, encField, Dec.bind, rdB, rdU, rdD, pure', fieldType, attrBack, normField,
      e1, e2, hn, ANYW, RANGEW, NUMERIC, STR, GEO, VECTOR, FULLTEXT, failD]
  | vector =>
    cases vo with
    | none => simp at hv
    | some v =>
      simp [decField, encField, Dec.bind, rdB, rdU, rdD, pure', fieldType, attrBack, normField,
        strip_nullTerm U attr ha, ANYW, RANGEW, NUMERIC, STR, GEO, VECTOR, FULLTEXT]

/-- **`field_name_kept`** (#3252): every attribute name survives the reload, `range:x`
and `vector:x` included — `decField_encField` returns exactly the encoded name. -/
theorem field_name_kept (attr : Bytes) (ty : Ty) : attrBack attr ty = attr := rfl

theorem field_prefix_kept (a : Bytes) : attrBack (RANGE_ ++ a) .range = RANGE_ ++ a := rfl

/-- A vector field without `VectorIndexOptions` writes no option words (`:408-420`), yet
the decoder reads five (`:677-683`): the stream desynchronises. Such a field is not built by
`CREATE VECTOR INDEX` (options are mandatory there), so this is latent. -/
theorem vector_without_opts (attr : Bytes) (ts : List W) :
    encField one simCode attr ⟨.vector, none, none, none, none⟩ ++ ts
      = [.buf (nullTerm attr), .u VECTOR, .d one, .u 0, .buf (nullTerm PHON_OFF)] ++ ts := by
  simp [encField, encPh, fieldType, VECTOR]

end RedisLayer.Serial
