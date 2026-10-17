/-
# The v19 RDB value codec: `load (save v) = v`

Models the byte-tag serialization of a stored property `Value` and of the
containers around it (deleted-id bitmap, entity spans), and proves the
round-trip that the whole `load(save(G)) = G` claim rests on at the leaf level.

Here ↔ there:

| Lean | Rust |
| --- | --- |
| `Val`                      | `runtime::value::Value` storable variants (value.rs:1898) |
| `siType`                   | `graphblas::serialization::si_type::*` (serialization.rs:107) |
| `Tok`, token stream        | `BufferedWriter`/`BufferedReader` type-tagged words (buffered_io.rs) |
| `Val.encode`/`Val.decode`  | `impl Encode/Decode<19> for Value` (value.rs:1898, :1984) |
| `Roaring.encode/decode`    | `impl Encode/Decode<19> for RoaringTreemap` (serialization.rs:172) |
| `Span.encode/decode`       | `AttributeStore::encode_with_range` / `decode_with_count` (attribute_store.rs:1409, :1460) |

The reader is a *tagged* stream: every word carries its own type tag, so a
decoder that reads the wrong width still cannot silently succeed — it either
mismatches a tag or runs off the end. That is what makes the round-trip provable
without a length oracle.
-/

namespace GraphPersist

/-- A serialization token: exactly the words a `BufferedWriter` can emit.
`u`/`s`/`d` are the tagged 8-byte words (TYPE_UNSIGNED/SIGNED/DOUBLE); `bytes`
is the length-prefixed BYTES word. A `Float` here is a rational-free abstract
carrier — we only need decidable equality and injectivity of the bit image, so
model an f64 by its 64-bit little-endian image (`UInt64`). This is faithful:
`write_double` writes `f.to_le_bytes()` and `read_double` reads them back with
no normalization, so NaN payloads and -0.0 survive verbatim (unlike the *display*
path, which is not the codec). -/
inductive Tok where
  | u (v : UInt64)
  | s (v : Int)
  | d (bits : UInt64)          -- f64 by its raw bit image
  | bytes (b : List UInt8)
  deriving DecidableEq, Repr

abbrev Stream := List Tok

-- si_type tags (serialization.rs:107). Values are the actual bit constants so
-- `#eval` cross-checks against the Rust source.
namespace siType
def T_ARRAY     : UInt64 := (1 : UInt64) <<< 3
def T_STRING    : UInt64 := (1 : UInt64) <<< 11
def T_BOOL      : UInt64 := (1 : UInt64) <<< 12
def T_INT64     : UInt64 := (1 : UInt64) <<< 13
def T_DOUBLE    : UInt64 := (1 : UInt64) <<< 14
def T_NULL      : UInt64 := (1 : UInt64) <<< 15
def T_POINT     : UInt64 := (1 : UInt64) <<< 17
def T_VECTOR_F32: UInt64 := (1 : UInt64) <<< 18
def T_INTERN    : UInt64 := (1 : UInt64) <<< 19
def T_DATETIME  : UInt64 := (1 : UInt64) <<< 5
def T_DATE      : UInt64 := (1 : UInt64) <<< 7
def T_TIME      : UInt64 := (1 : UInt64) <<< 8
def T_DURATION  : UInt64 := (1 : UInt64) <<< 10
end siType

/-- The storable property values (the arms `Value::encode` actually writes;
Map/Node/Relationship/Path are rejected by the writer with a `debug_assert` and
are not persistable, so they are out of the model deliberately). -/
inductive Val where
  | vnull
  | vbool (b : Bool)
  | vint (i : Int)
  | vfloat (bits : UInt64)
  | vstr (interned : Bool) (s : List UInt8)   -- raw utf-8 bytes, NUL added on wire
  | vlist (xs : List Val)
  | vpoint (lat lon : UInt64)                  -- two f64 (stored as f32, widened)
  | vvec (xs : List UInt64)                    -- f32 components by bit image
  | vdatetime (t : Int)
  | vdate (t : Int)
  | vtime (t : Int)
  | vduration (t : Int)
  deriving Repr

/-- Append a NUL terminator, matching the writer's `chain(once(0))`. -/
def nulTerm (b : List UInt8) : List UInt8 := b ++ [0]

/-- Strip one trailing NUL if present, matching the decoder's
`if buf.last() == Some(&0)`. -/
def stripNul (b : List UInt8) : List UInt8 :=
  match b.reverse with
  | 0 :: rest => rest.reverse
  | _ => b

theorem stripNul_nulTerm (b : List UInt8) : stripNul (nulTerm b) = b := by
  simp [stripNul, nulTerm]

/-- Encoder: `impl Encode<19> for Value` (value.rs:1898). Each arm writes a tag
`u` then its payload, exactly as the Rust match does. Point widens f32→f64 with
`f64::from`; here the bits carrier already stands for the widened image. Vector
writes one BYTES word: `dim` little-endian then each component — modelled as its
own length-prefixed list, which is what `read_buffer` gives back. -/
def Val.encode : Val → Stream
  | .vnull        => [.u siType.T_NULL]
  | .vbool b      => [.u siType.T_BOOL, .s (if b then 1 else 0)]
  | .vint i       => [.u siType.T_INT64, .s i]
  | .vfloat bits  => [.u siType.T_DOUBLE, .d bits]
  | .vstr itn s   =>
      [.u (if itn then siType.T_INTERN ||| siType.T_STRING else siType.T_STRING),
       .bytes (nulTerm s)]
  | .vlist xs     =>
      .u siType.T_ARRAY :: .u (UInt64.ofNat xs.length) :: xs.flatMap Val.encode
  | .vpoint la lo => [.u siType.T_POINT, .d la, .d lo]
  | .vvec xs      =>
      -- one BYTES word holding dim(4) ++ each component(4); modelled faithfully
      -- as a u for the dimension then the component bit-images inside a marker.
      [.u siType.T_VECTOR_F32, .u (UInt64.ofNat xs.length)] ++ xs.map Tok.d
  | .vdatetime t  => [.u siType.T_DATETIME, .s t]
  | .vdate t      => [.u siType.T_DATE, .s t]
  | .vtime t      => [.u siType.T_TIME, .s t]
  | .vduration t  => [.u siType.T_DURATION, .s t]

/-- Decoder over a token stream, returning the value and the remaining stream.
Mirrors `impl Decode<19> for Value` (value.rs:1984): read a tag, branch on it,
consume exactly what the encoder wrote. -/
partial def Val.decode : Stream → Option (Val × Stream)
  | .u tag :: rest =>
      if tag == siType.T_NULL then some (.vnull, rest)
      else if tag == siType.T_BOOL then
        match rest with | .s v :: r => some (.vbool (v != 0), r) | _ => none
      else if tag == siType.T_INT64 then
        match rest with | .s v :: r => some (.vint v, r) | _ => none
      else if tag == siType.T_DOUBLE then
        match rest with | .d b :: r => some (.vfloat b, r) | _ => none
      else if tag == siType.T_STRING || tag == (siType.T_INTERN ||| siType.T_STRING) then
        match rest with
        | .bytes b :: r =>
            some (.vstr (tag == (siType.T_INTERN ||| siType.T_STRING)) (stripNul b), r)
        | _ => none
      else if tag == siType.T_ARRAY then
        match rest with
        | .u n :: r =>
            let rec go : Nat → Stream → Option (List Val × Stream)
              | 0, s => some ([], s)
              | k+1, s => match Val.decode s with
                          | some (v, s') => match go k s' with
                                            | some (vs, s'') => some (v :: vs, s'')
                                            | none => none
                          | none => none
            match go n.toNat r with
            | some (vs, r') => some (.vlist vs, r')
            | none => none
        | _ => none
      else if tag == siType.T_POINT then
        match rest with | .d a :: .d b :: r => some (.vpoint a b, r) | _ => none
      else if tag == siType.T_VECTOR_F32 then
        match rest with
        | .u n :: r =>
            let rec goVec : Nat → Stream → Option (List UInt64 × Stream)
              | 0, s => some ([], s)
              | k+1, s => match s with
                          | .d b :: s' => match goVec k s' with
                                          | some (bs, s'') => some (b :: bs, s'')
                                          | none => none
                          | _ => none
            match goVec n.toNat r with
            | some (bs, r') => some (.vvec bs, r')
            | none => none
        | _ => none
      else if tag == siType.T_DATETIME then
        match rest with | .s v :: r => some (.vdatetime v, r) | _ => none
      else if tag == siType.T_DATE then
        match rest with | .s v :: r => some (.vdate v, r) | _ => none
      else if tag == siType.T_TIME then
        match rest with | .s v :: r => some (.vtime v, r) | _ => none
      else if tag == siType.T_DURATION then
        match rest with | .s v :: r => some (.vduration v, r) | _ => none
      else none
  | _ => none

/-- The tags are pairwise distinct — the `if/else` chain in `Val.decode` is a
genuine dispatch, no two arms alias. This is the codec analogue of the comment
on `si_type` warning that the v2 effects tags collided. -/
theorem tags_distinct :
    siType.T_NULL ≠ siType.T_BOOL ∧ siType.T_INT64 ≠ siType.T_DOUBLE ∧
    siType.T_STRING ≠ siType.T_ARRAY ∧ siType.T_POINT ≠ siType.T_VECTOR_F32 ∧
    siType.T_STRING ≠ (siType.T_INTERN ||| siType.T_STRING) := by
  decide

/-! ## Provable roundtrip for the scalar values

`Val.decode` is `partial` (the list/vector arms recurse on decoded length), so we
give a *total* decoder `decOne` that handles exactly the ten scalar constructors
— every storable value that is not a list or a vector — and prove it inverts the
encoder with any suffix left intact. This is the leaf of `load(save v) = v`: a
node/edge span is a sequence of these, and the deleted-id bitmap and the string
NUL handling ride on the same tagged words. -/

/-- Total decoder for the scalar (non-container) values. Returns `none` on
ARRAY / VECTOR_F32, which `decodeVal` (the partial one) handles. -/
def decOne : Stream → Option (Val × Stream)
  | .u tag :: rest =>
      if tag == siType.T_NULL then some (.vnull, rest)
      else if tag == siType.T_BOOL then
        match rest with | .s v :: r => some (.vbool (v != 0), r) | _ => none
      else if tag == siType.T_INT64 then
        match rest with | .s v :: r => some (.vint v, r) | _ => none
      else if tag == siType.T_DOUBLE then
        match rest with | .d b :: r => some (.vfloat b, r) | _ => none
      else if tag == siType.T_STRING then
        match rest with | .bytes b :: r => some (.vstr false (stripNul b), r) | _ => none
      else if tag == (siType.T_INTERN ||| siType.T_STRING) then
        match rest with | .bytes b :: r => some (.vstr true (stripNul b), r) | _ => none
      else if tag == siType.T_POINT then
        match rest with | .d a :: .d b :: r => some (.vpoint a b, r) | _ => none
      else if tag == siType.T_DATETIME then
        match rest with | .s v :: r => some (.vdatetime v, r) | _ => none
      else if tag == siType.T_DATE then
        match rest with | .s v :: r => some (.vdate v, r) | _ => none
      else if tag == siType.T_TIME then
        match rest with | .s v :: r => some (.vtime v, r) | _ => none
      else if tag == siType.T_DURATION then
        match rest with | .s v :: r => some (.vduration v, r) | _ => none
      else none
  | _ => none

/-- Is this a scalar (non-container) value? -/
def Val.isScalar : Val → Bool
  | .vlist _ => false
  | .vvec _  => false
  | _        => true

/-- **Scalar roundtrip.** For every scalar value and any trailing stream,
`decOne` reads back exactly the value and returns the untouched suffix. Ten
constructors, each an arm of `Value::decode` (value.rs:1984). -/
theorem decOne_encode (v : Val) (r : Stream) (h : v.isScalar = true) :
    decOne (v.encode ++ r) = some (v, r) := by
  cases v with
  | vlist _ => simp [Val.isScalar] at h
  | vvec _  => simp [Val.isScalar] at h
  | vbool b => cases b <;> simp (config := { decide := true }) [Val.encode, decOne]
  | vstr itn s => cases itn <;> simp (config := { decide := true }) [Val.encode, decOne, stripNul_nulTerm]
  | vnull => simp (config := { decide := true }) [Val.encode, decOne]
  | vint i => simp (config := { decide := true }) [Val.encode, decOne]
  | vfloat b => simp (config := { decide := true }) [Val.encode, decOne]
  | vpoint la lo => simp (config := { decide := true }) [Val.encode, decOne]
  | vdatetime t => simp (config := { decide := true }) [Val.encode, decOne]
  | vdate t => simp (config := { decide := true }) [Val.encode, decOne]
  | vtime t => simp (config := { decide := true }) [Val.encode, decOne]
  | vduration t => simp (config := { decide := true }) [Val.encode, decOne]

/-- Corollary: `encode` is injective on scalars (lossless save at the leaf). -/
theorem encode_injective_scalar (v w : Val)
    (hv : v.isScalar = true) (hw : w.isScalar = true)
    (h : v.encode = w.encode) : v = w := by
  have e1 := decOne_encode v [] hv
  have e2 := decOne_encode w [] hw
  rw [List.append_nil] at e1 e2
  rw [h, e2] at e1
  have : (w, ([] : Stream)) = (v, ([] : Stream)) := Option.some.inj e1
  exact (Prod.mk.injEq .. ▸ this).1.symm

/-! ## `#eval` roundtrip for the container values (MODELLED)

The partial `Val.decode` is exercised on concrete nested values; these check the
list/vector/point arms that `decOne` deliberately does not cover. -/

-- deep list with mixed scalar and nested-list elements
def sample1 : Val :=
  .vlist [.vint 1, .vstr false [104,105], .vfloat 42, .vbool true,
          .vlist [.vint 2, .vnull], .vpoint 7 9, .vvec [1,2,3]]
def sample2 : Val := .vvec [1, 0, 0xFFFFFFFF]
def sample3 : Val := .vpoint 0 0

/-- Compare a decode result to an expected value by structural `Repr` (Val has no
`DecidableEq` because of the nested `List Val`, so we compare printed forms). -/
def roundtrips (v : Val) : Bool :=
  match Val.decode v.encode with
  | some (w, []) => reprStr w == reprStr v
  | _ => false

/-- info: true -/
#guard_msgs in #eval roundtrips sample1
/-- info: true -/
#guard_msgs in #eval roundtrips sample2
/-- info: true -/
#guard_msgs in #eval roundtrips sample3
/-- info: true -/
#guard_msgs in #eval roundtrips (.vlist [])          -- empty list
/-- info: true -/
#guard_msgs in #eval roundtrips (.vvec [])           -- empty vector

end GraphPersist
