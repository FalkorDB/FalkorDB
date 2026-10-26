import RedisLayer.SerialSchema
/-!
# C's index-field layout (#3484)

`decField_encField` proves `decode ∘ encode = id` against **Rust's** encoder, which writes
one field entry per index type, so its type word always has exactly one type. C writes
one entry per *attribute* with every type in the bitmask (`IndexField_Init`,
`_RdbSaveIndexField` in C's `src/serializers/encoder/v20/encode_schema.c`; the loader
accepts any non-empty subset, `decode_graph_schema.c:127`). A C word with several types
lies outside that theorem, and `decField` keeps one: an attribute with a RANGE and a
FULLTEXT (or VECTOR) index loses its RANGE index when a C-written RDB loads (#3484,
confirmed live: v4.22.2 → 6.0.3 and C master → Rust main).

* `encFieldC` — C's entry for an attribute with the types `r`, `ft`, `v`.
* `bug3484_range_fulltext_loses_range`, `bug3484_range_vector_loses_range` — the current
  decoder returns one field.
* `decFields` — the decoder of FalkorDB#3514 (one field per set type; no type or an
  unknown bit is refused). `decFields_encFieldC`: every C entry decodes to exactly its
  types; `decFields_empty_refused`.
* `decFields_encField` — Rust's own single-type entries still decode as before.
-/
namespace RedisLayer.Serial
open RedisLayer.BufferedIO (W Bytes)

variable (U : Utf8)

def RANGEW : Nat := NUMERIC ||| STR ||| GEO

/-- C's type word for an attribute with the given index types. -/
def maskOf (r ft v : Bool) : Nat :=
  (if r then RANGEW else 0) ||| (if ft then FULLTEXT else 0) ||| (if v then VECTOR else 0)

/-- C's vector option words: dimension, M, efConstruction, efRuntime, metric code. -/
structure VWords where
  dim : Nat
  m : Nat
  efc : Nat
  efr : Nat
  sc : Nat
  deriving DecidableEq, Repr

/-- `_RdbSaveIndexField`: name, type mask, weight, nostem, phonetic, then the vector
options iff the VECTOR bit is set. -/
def encFieldC (attr : Bytes) (r ft v : Bool) (w ns : Nat) (ph : Bytes) (vw : VWords) :
    List W :=
  [.buf (nullTerm attr), .u (maskOf r ft v), .d w, .u ns, .buf (nullTerm ph)]
  ++ (if v then [.u vw.dim, .u vw.m, .u vw.efc, .u vw.efr, .u vw.sc] else [])

/-- The fields an entry stands for, in `db.indexes()` order. -/
def cFields (r ft v : Bool) (w ns : Nat) (ph : Bytes) (vw : VWords) : List Fld :=
  (if r then [⟨.range, none, none, none, none⟩] else [])
  ++ (if ft then [⟨.fulltext, some w, some (ns ≠ 0), some ph, none⟩] else [])
  ++ (if v then [⟨.vector, none, none, none,
        some ⟨vw.dim, some vw.m, some vw.efc, some vw.efr, some (simName vw.sc)⟩⟩] else [])

/-! ## The current decoder keeps one type -/

theorem bug3484_range_fulltext_loses_range (attr ph : Bytes) (ha : U.valid attr)
    (hp : U.valid ph) (w ns : Nat) (vw : VWords) (ts : List W) :
    decField U (encFieldC attr true true false w ns ph vw ++ ts)
      = some ((attr, ⟨.fulltext, some w, some (ns ≠ 0), some ph, none⟩), ts) := by
  simp [decField, encFieldC, maskOf, Dec.bind, rdB, rdU, rdD, strip_nullTerm U attr ha,
    strip_nullTerm U ph hp, RANGEW, NUMERIC, STR, GEO, VECTOR, FULLTEXT]

theorem bug3484_range_vector_loses_range (attr ph : Bytes) (ha : U.valid attr)
    (w ns : Nat) (vw : VWords) (ts : List W) :
    decField U (encFieldC attr true false true w ns ph vw ++ ts)
      = some ((attr, ⟨.vector, none, none, none,
          some ⟨vw.dim, some vw.m, some vw.efc, some vw.efr, some (simName vw.sc)⟩⟩), ts) := by
  simp [decField, encFieldC, maskOf, Dec.bind, rdB, rdU, rdD, pure', strip_nullTerm U attr ha,
    RANGEW, NUMERIC, STR, GEO, VECTOR, FULLTEXT]

/-! ## The fixed decoder (FalkorDB#3514) -/

def ANYW : Nat := RANGEW ||| FULLTEXT ||| VECTOR

def failD {α} : Dec α := fun _ => none

/-- `decode_index_field` after #3514. -/
def decFields : Dec (Bytes × List Fld) :=
  Dec.bind rdB fun nb => Dec.bind rdU fun ft => Dec.bind rdD fun w =>
  Dec.bind rdU fun ns => Dec.bind rdB fun pb =>
  if ft = 0 ∨ ft &&& (ANYW ^^^ 0xFFFFFFFFFFFFFFFF) ≠ 0 then failD else
  let isR := ft &&& RANGEW ≠ 0
  let isFt := ft &&& FULLTEXT ≠ 0
  let isV := ft &&& VECTOR ≠ 0
  let rest : List Fld :=
    (if isR then [⟨.range, none, none, none, none⟩] else [])
    ++ (if isFt then [⟨.fulltext, some w, some (ns ≠ 0), some (strip U pb), none⟩] else [])
  if isV then
    Dec.bind rdU fun dim => Dec.bind rdU fun m => Dec.bind rdU fun efc =>
    Dec.bind rdU fun efr => Dec.bind rdU fun sc =>
    pure' (strip U nb, rest ++ [⟨.vector, none, none, none,
      some ⟨dim, some m, some efc, some efr, some (simName sc)⟩⟩])
  else pure' (strip U nb, rest)

/-- **Fixed**: every C entry with at least one type decodes to exactly its types. -/
theorem decFields_encFieldC (attr ph : Bytes) (ha : U.valid attr) (hp : U.valid ph)
    (r ft v : Bool) (hne : r ∨ ft ∨ v) (w ns : Nat) (vw : VWords) (ts : List W) :
    decFields U (encFieldC attr r ft v w ns ph vw ++ ts)
      = some ((attr, cFields r ft v w ns ph vw), ts) := by
  cases r <;> cases ft <;> cases v <;> simp at hne <;>
  simp [decFields, encFieldC, cFields, maskOf, Dec.bind, rdB, rdU, rdD, pure',
    strip_nullTerm U attr ha, strip_nullTerm U ph hp, ANYW, RANGEW, NUMERIC, STR, GEO,
    VECTOR, FULLTEXT]

/-- A word with no type is refused, as C's `_RdbLoadIndex` does. -/
theorem decFields_empty_refused (attr ph : Bytes) (w ns : Nat) (vw : VWords) (ts : List W) :
    decFields U (encFieldC attr false false false w ns ph vw ++ ts) = none := by
  simp [decFields, encFieldC, maskOf, Dec.bind, rdB, rdU, rdD, failD]

/-- Rust's single-type entries: the fix decodes them to the one field `decField` gave. -/
theorem decFields_encField (one : Nat) (simCode : Option Bytes → Nat) (attr : Bytes)
    (f : Fld) (ha : U.valid attr) (hp : U.valid (f.phonetic.getD []))
    (hv : f.ty = .vector → f.vopts.isSome) (ts : List W) :
    decFields U (encField one simCode attr f ++ ts)
      = some ((attr, [normField one simCode f]), ts) := by
  obtain ⟨ty, w, ns, ph, vo⟩ := f
  cases ty with
  | range =>
    simp [decFields, encField, Dec.bind, rdB, rdU, rdD, fieldType, normField,
      strip_nullTerm U attr ha, ANYW, RANGEW, NUMERIC, STR, GEO, VECTOR, FULLTEXT]
  | fulltext =>
    simp at hp
    simp [decFields, encField, Dec.bind, rdB, rdU, rdD, fieldType, normField,
      strip_nullTerm U attr ha, strip_nullTerm U _ hp, ANYW, RANGEW, NUMERIC, STR, GEO,
      VECTOR, FULLTEXT]
  | vector =>
    cases vo with
    | none => simp at hv
    | some v =>
      simp [decFields, encField, Dec.bind, rdB, rdU, rdD, pure', fieldType, normField,
        strip_nullTerm U attr ha, ANYW, RANGEW, NUMERIC, STR, GEO, VECTOR, FULLTEXT]

end RedisLayer.Serial
