import RedisLayer.SerialSchema
/-!
# C's index-field layout (#3484, fixed by #3514 `a0ba12950`)

`decField_encField` proves `decode ∘ encode = id` against **Rust's** encoder, which writes
one field entry per index type, so its type word always has exactly one type. C writes
one entry per *attribute* with every type in the bitmask (`IndexField_Init`,
`_RdbSaveIndexField` in C's `src/serializers/encoder/v20/encode_schema.c`; the loader
accepts any non-empty subset, `decode_graph_schema.c:127`). Before #3514 the decoder
kept one type: an attribute with a RANGE and a FULLTEXT (or VECTOR) index
lost its RANGE index when a C-written RDB loads (#3484, confirmed live: v4.22.2 → 6.0.3 and C
master → Rust main). **Main now satisfies the property**: `SerialSchema.decField` is the merged
`decode_index_field` (`src/serializers/mod.rs:637-725`), and the three theorems below hold on it.

* `encFieldC` — C's entry for an attribute with the types `r`, `ft`, `v`.
* `decField_encFieldC` — **main**: every C entry with at least one type decodes to exactly its
  types (range, fulltext, vector order), C's phonetic `"no"` read as off.
* `decField_empty_refused` — a word with no type is refused, as C's `_RdbLoadIndex` does
  (an unknown bit too: `decField_unknown_bit_refused`).
* `decField_encField` (in `SerialSchema`) — Rust's own single-type entries still decode as before.
* History (the pre-#3514 decoder and its counterexamples, removed; in 17ccc5412): C's `"no"` came back
  as a code, which `phonetic_enabled` read as ON; Rust wrote an empty string for off, and C enables
  phonetic for any other string than `"no"`; the fix is `encPh_c_on_iff`.
-/
namespace RedisLayer.Serial
open RedisLayer.BufferedIO (W Bytes)

variable (U : Utf8)

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
  ++ (if ft then [⟨.fulltext, some w, some (ns ≠ 0), some (phonNorm ph), none⟩] else [])
  ++ (if v then [⟨.vector, none, none, none,
        some ⟨vw.dim, some vw.m, some vw.efc, some vw.efr, some (simName vw.sc)⟩⟩] else [])

/-! ## History: the pre-#3514 decoder keeps one type -/

/-! ## Main: the merged decoder (`SerialSchema.decField`, FalkorDB#3514) -/

/-- **Fixed on main**: every C entry with at least one type decodes to exactly its types. -/
theorem decField_encFieldC (attr ph : Bytes) (ha : U.valid attr) (hp : U.valid ph)
    (r ft v : Bool) (hne : r ∨ ft ∨ v) (w ns : Nat) (vw : VWords) (ts : List W) :
    decField U (encFieldC attr r ft v w ns ph vw ++ ts)
      = some ((attr, cFields r ft v w ns ph vw), ts) := by
  cases r <;> cases ft <;> cases v <;> simp at hne <;>
  simp [decField, encFieldC, cFields, maskOf, Dec.bind, rdB, rdU, rdD, pure',
    strip_nullTerm U attr ha, strip_nullTerm U ph hp, ANYW, RANGEW, NUMERIC, STR, GEO,
    VECTOR, FULLTEXT]

/-- A word with no type is refused, as C's `_RdbLoadIndex` does. -/
theorem decField_empty_refused (attr ph : Bytes) (w ns : Nat) (vw : VWords) (ts : List W) :
    decField U (encFieldC attr false false false w ns ph vw ++ ts) = none := by
  simp [decField, encFieldC, maskOf, Dec.bind, rdB, rdU, rdD, failD]

/-- An unknown bit (here `0x20`) is refused too (`:668-672`). -/
theorem decField_unknown_bit_refused (attr ph : Bytes) (w ns : Nat) (ts : List W) :
    decField U ([.buf (nullTerm attr), .u 0x20, .d w, .u ns, .buf (nullTerm ph)] ++ ts) = none := by
  simp [decField, Dec.bind, rdB, rdU, rdD, failD, ANYW, RANGEW, NUMERIC, STR, GEO, VECTOR,
    FULLTEXT]

/-- C's phonetic `"no"` (its default on every field it did not enable) decodes to the explicit
off, never to a code. -/
theorem decField_c_phonetic_no_off (attr : Bytes) (ha : U.valid attr) (w ns : Nat) (ts : List W) :
    decField U (encFieldC attr false true false w ns PHON_OFF ⟨0, 0, 0, 0, 0⟩ ++ ts)
      = some ((attr, [⟨.fulltext, some w, some (ns ≠ 0), some [], none⟩]), ts) := by
  have := decField_encFieldC U attr PHON_OFF ha U.valid_no false true false (by simp) w ns
    ⟨0, 0, 0, 0, 0⟩ ts
  simpa [cFields, phonNorm] using this

/-! ## The phonetic string we write, as C reads it -/

/-- C enables phonetic matching for every phonetic string except `"no"`. -/
def cPhoneticOn (b : Bytes) : Prop := b ≠ PHON_OFF

/-- **Fixed (#3514)**: what Rust writes is read by C as ON exactly when the field's phonetic
is on (absent, empty and `"no"` are off). -/
theorem encPh_c_on_iff (f : Fld) :
    cPhoneticOn (encPh f) ↔ phonNorm (f.phonetic.getD []) ≠ [] := by
  unfold cPhoneticOn encPh phonNorm
  by_cases h : f.phonetic.getD [] = [] <;> by_cases h2 : f.phonetic.getD [] = PHON_OFF <;>
    simp [h, h2]
  all_goals first | (exact absurd (h.symm.trans h2) (by decide)) | skip

end RedisLayer.Serial
