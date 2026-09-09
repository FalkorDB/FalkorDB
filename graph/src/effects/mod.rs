//! Replication effects: the `GRAPH.EFFECT` payload a write produces and a
//! replica applies.
//!
//! One wire version so far, [`v3`] — the format both engines can read. Batched,
//! C-compatible widths. [`EffectsPayload`] is what the rest of the codebase
//! names; the version lives here and in the `impl EffectsFormat<N>` under each
//! version's module, nowhere else.
//!
//! [`Reader`] and [`DecodeError`] sit here rather than under a version because
//! a bounds-checked cursor over a byte slice is not version-specific, and a v4
//! would want the same one.
//!
//! ## Why this is not the RDB codec
//!
//! `serialization::{Encode, Decode}` and `BufferedReader` look like they should
//! serve, and cannot. RDB is **self-describing**: `read_unsigned` reads a
//! `TYPE_UNSIGNED` tag and then eight bytes, so every primitive carries its own
//! type. The effects wire is **positional and untagged**, because that is what C
//! writes — injecting a tag per primitive would produce bytes no C reader
//! accepts. `BufferedReader` also lives in the root crate, which this one cannot
//! depend on.
//!
//! What *is* shared is shared: the `SIValue` tags and index-field flags come
//! from `serialization::{si_type, index_field_type}` rather than being restated.

pub mod error;
pub mod reader;
pub mod v3;
pub mod writer;

pub use error::DecodeError;
pub use reader::Reader;
pub use writer::EffectWrite;

/// Types that can write themselves into an effects payload of a given version.
///
/// Mirrors `serialization::Encode<VERSION>` on the RDB side, and for the same
/// reason: the version belongs in the type system rather than in a module name,
/// so a v4 lands as `impl EffectEncode<4> for Record` beside the v3 impl and the
/// compiler picks between them.
///
/// Blocks implement it too, so the version is stated once per block rather than
/// left to the module path. A v4 impl beside a v3 one makes every unannotated
/// `.encode(buf)` ambiguous, and the compiler then names each site that has to
/// choose — which a free function in a `v3/` directory could never do.
pub trait EffectEncode<const VERSION: u8> {
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    );
}

/// Types that can read themselves out of an effects payload of a given version.
pub trait EffectDecode<const VERSION: u8>: Sized {
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are malformed, truncated, or
    /// describe a shape this version does not have.
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError>;
}

/// Blocks whose length is not on the wire, but stated by the record around them.
///
/// `AttrValues` is `count × attrs_per_row` values and `IdList` is `count` ids.
/// Neither repeats a length the record header already gave — that is the whole
/// point of the header — so [`EffectDecode::decode`] has nowhere to put it and
/// these implement this instead.
///
/// A separate trait rather than a second method with an `unimplemented!()`
/// default, which is what the RDB `Decode` does for `decode_with_count`. That
/// default turns "this type is not sized-decodable" into a runtime panic, and
/// having the compiler answer that kind of question is the reason these traits
/// carry a version at all.
pub trait EffectDecodeSized<const VERSION: u8>: Sized {
    /// What the surrounding record already knows about this block's length.
    type Size;

    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are malformed, truncated, or
    /// describe a shape this version does not have.
    fn decode_sized(
        r: &mut Reader<'_>,
        size: Self::Size,
    ) -> Result<Self, DecodeError>;
}

/// The one thing left that names a version, and it is configuration rather
/// than format: the compression threshold is read by `seal` itself, so the
/// setting stays behind the format boundary.
pub use v3::EFFECTS_COMPRESSION;
