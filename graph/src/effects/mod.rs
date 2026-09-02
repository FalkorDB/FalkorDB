//! Replication effects: the `GRAPH.EFFECT` payload a write produces and a
//! replica applies.
//!
//! One wire version, [`v3`] — the format both engines can read. Batched,
//! C-compatible widths.
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
pub mod payload;
pub mod reader;
pub mod v3;
pub mod writer;

pub use error::DecodeError;
pub use reader::Reader;

/// Types that can write themselves into an effects payload of a given version.
///
/// Mirrors `serialization::Encode<VERSION>` on the RDB side, and for the same
/// reason: the version belongs in the type system rather than in a module name,
/// so a v4 lands as `impl EffectEncode<4> for Record` beside the v3 impl and the
/// compiler picks between them.
///
/// The unit is a whole record. Blocks stay free functions because they are sized
/// by the record's `count`, which a `&self` encoder has no way to carry — the
/// same reason the RDB traits grow `encode_with_range` / `decode_with_count`
/// rather than folding counts into `encode`.
pub trait EffectEncode<const VERSION: u8> {
    fn encode(
        &self,
        buf: &mut Vec<u8>,
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

/// Helpers the byte-pinning tests share.
///
/// They live here rather than in one submodule's `mod tests` because the pins
/// are spread across `id_list`, `blocks`, `records` and `value` — each beside
/// the bytes it pins — and a `#[cfg(test)]` item is not importable from a
/// sibling's test module unless it sits above them.
#[cfg(test)]
pub(crate) mod testing {
    /// Render bytes the way the pins are written: lowercase, space-separated.
    pub(crate) fn hex(buf: &[u8]) -> String {
        buf.iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}
