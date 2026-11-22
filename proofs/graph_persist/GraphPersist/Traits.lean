import GraphPersist.Codec
/-!
# `Encode` / `Decode`: the trait declarations and their defaults (serialization.rs:32-92)

```rust
pub trait Encode<const VERSION: u64> {
    fn encode(&self, w: &mut dyn Writer);                                   // :34, no body
    fn encode_with_range(&self, w: &mut dyn Writer, count: u64, offset: u64) {  // :40
        let _ = (w, count, offset);
        unimplemented!()
    }
}
pub trait Decode<const VERSION: u64>: Sized {
    fn decode(r: &mut dyn Reader) -> Result<Self, String>;                  // :64, no body
    fn decode_with_count(&mut self, r: &mut dyn Reader, count: u64, attr_limit: usize)
        -> Result<(), String> {                                             // :76
        let _ = (r, count, attr_limit);
        unimplemented!()
    }
}
```

An implementor supplies `encode` / `decode` and may override the two `*_with_range/_count`
defaults. A call through the trait runs the implementor's method; a call to a defaulted method
the implementor did not override panics before reading or writing anything (`unimplemented!()`).
A panic is `none` below.

| here | there |
| --- | --- |
| `EncodeImpl.encode` / `DecodeImpl.decode` | the required methods (:34, :64) |
| `EncodeImpl.withRange` (`none` default) | `encode_with_range` (:40) |
| `DecodeImpl.withCount` (`none` default) | `decode_with_count` (:76) |

* `encode_call`, `decode_call` — the required methods dispatch to the implementor.
* `encode_with_range_default_panics` / `decode_with_count_default_panics` — a default call
  panics, whatever its arguments.
* `*_override` — an override is what the call runs.
The implementors that override (`RoaringTreemap`, `AttributeStore`) are rows 16/18 of this
project and proofs/pending_commit; the ones that do not (`Value`, `Tensor`, `Matrix`, …) never
have `encode_with_range` called on them (`encode_with_range` is reached only from the
`RoaringTreemap`/`AttributeStore` payload encoders, `decode_with_count` only from the
`AttributeStore`/`RoaringTreemap` payload decoders).
-/
namespace GraphPersist

/-- An `Encode<V>` implementor for values of `A`: the stream a value encodes to, and the optional
override of `encode_with_range`. `none` = not overridden (the trait default). -/
structure EncodeImpl (A : Type) where
  encode : A → Stream
  withRange : Option (A → Nat → Nat → Stream) := none

/-- `Decode<V>`: `decode` returns `Err` (`none`) or the value and what is left; the optional
override of `decode_with_count`. -/
structure DecodeImpl (A : Type) where
  decode : Stream → Option (A × Stream)
  withCount : Option (A → Stream → Nat → Nat → Option (A × Stream)) := none

/-- `T::encode(&self, w)` (:34): the required method. -/
def EncodeImpl.encodeCall {A : Type} (I : EncodeImpl A) (a : A) : Stream := I.encode a

/-- `T::encode_with_range(&self, w, count, offset)` (:40): the override, else the trait default
(`unimplemented!()`, a panic: `none`). -/
def EncodeImpl.rangeCall {A : Type} (I : EncodeImpl A) (a : A) (count offset : Nat) : Option Stream :=
  match I.withRange with
  | some f => some (f a count offset)
  | none => none

/-- `T::decode(r)` (:64): the required method. -/
def DecodeImpl.decodeCall {A : Type} (I : DecodeImpl A) (r : Stream) : Option (A × Stream) := I.decode r

/-- `self.decode_with_count(r, count, attr_limit)` (:76): the override, else the trait default
(`unimplemented!()`: `none`). -/
def DecodeImpl.countCall {A : Type} (I : DecodeImpl A) (a : A) (r : Stream) (count limit : Nat) :
    Option (A × Stream) :=
  match I.withCount with
  | some f => f a r count limit
  | none => none

theorem encode_call {A : Type} (I : EncodeImpl A) (a : A) : I.encodeCall a = I.encode a := rfl
theorem decode_call {A : Type} (I : DecodeImpl A) (r : Stream) : I.decodeCall r = I.decode r := rfl

/-- The default `encode_with_range` panics, whatever the arguments, writing nothing. -/
theorem encode_with_range_default_panics {A : Type} (e : A → Stream) (a : A) (count offset : Nat) :
    ({ encode := e } : EncodeImpl A).rangeCall a count offset = none := rfl

/-- The default `decode_with_count` panics, whatever the arguments, reading nothing. -/
theorem decode_with_count_default_panics {A : Type} (d : Stream → Option (A × Stream)) (a : A)
    (r : Stream) (count limit : Nat) :
    ({ decode := d } : DecodeImpl A).countCall a r count limit = none := rfl

theorem encode_with_range_override {A : Type} (I : EncodeImpl A) (f : A → Nat → Nat → Stream)
    (h : I.withRange = some f) (a : A) (count offset : Nat) :
    I.rangeCall a count offset = some (f a count offset) := by simp [EncodeImpl.rangeCall, h]

theorem decode_with_count_override {A : Type} (I : DecodeImpl A)
    (f : A → Stream → Nat → Nat → Option (A × Stream)) (h : I.withCount = some f) (a : A)
    (r : Stream) (count limit : Nat) : I.countCall a r count limit = f a r count limit := by
  simp [DecodeImpl.countCall, h]

end GraphPersist
