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

use crate::graph::graph::Graph;
use v3::apply::ApplyError;

/// A whole `GRAPH.EFFECT` payload, end to end.
///
/// [`EffectEncode`] and [`EffectDecode`] describe one *record*; this describes
/// the buffer that carries them, which is where the version actually lives —
/// the header names it, and whether the bytes are compressed is a property of
/// the payload rather than of any record in it.
///
/// It exists so that nothing outside this module has to name a version. The
/// host module handles transport: it decides when a payload is sent, to which
/// key, and over which context. It does not decide what the bytes are. Before
/// this trait it did, in two places — `graph_core` called `v3::seal` to
/// compress, and `payload` carried its own copy of the header length to tell an
/// empty payload from a full one — so "which version" was spelled out at call
/// sites that have no business knowing.
///
/// A v4 lands as a second impl and one line changed in [`Current`].
pub trait EffectsFormat {
    /// The version byte a payload written by this format opens with.
    const VERSION: u8;

    /// True for a payload that carries no records — a bare header.
    ///
    /// What "bare" means is the format's: v3's header is two bytes, and a
    /// caller counting them itself is a caller that has to be revisited when
    /// the header grows.
    fn is_empty(buf: &[u8]) -> bool;

    /// Finish a payload: whatever has to happen to the bytes after the last
    /// record is written and before they go on the wire.
    ///
    /// Call once, on a complete buffer, at the last possible moment. For v3
    /// that is compression, which rewrites everything after the header — so
    /// running it per commit produced a payload compressed twice that could not
    /// be read at all.
    fn finish(buf: &mut Vec<u8>);

    /// Apply a payload to `graph`.
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError`] if the payload cannot be decoded, or describes a
    /// mutation this graph cannot perform — either of which means the sender
    /// and this node have diverged.
    fn apply(
        graph: &mut Graph,
        buf: &[u8],
    ) -> Result<(), ApplyError>;
}

/// The v3 wire format.
pub struct V3;

impl EffectsFormat for V3 {
    const VERSION: u8 = v3::EFFECTS_VERSION;

    fn is_empty(buf: &[u8]) -> bool {
        // `u8 version` + `u8 flags`, and nothing after them.
        buf.len() <= 2
    }

    fn finish(buf: &mut Vec<u8>) {
        v3::seal(buf);
    }

    fn apply(
        graph: &mut Graph,
        buf: &[u8],
    ) -> Result<(), ApplyError> {
        v3::apply::apply_effects(graph, buf)
    }
}

/// The format this build writes, and the only one it reads.
///
/// A payload announcing any other version came from a peer speaking a language
/// this build does not, which is divergence rather than a compatibility case —
/// [`EffectsFormat::apply`] rejects it and the host forces a resync.
pub type Current = V3;

/// The emitter's surface, re-exported so no caller names a version to reach it.
///
/// These are the current format's types by definition — an `AnnouncedIndex` is
/// how *this* wire describes an index — so the version belongs in one line here
/// rather than in every import that needs one.
pub use v3::EFFECTS_COMPRESSION;
pub use v3::emit::{
    AnnouncedConstraint, AnnouncedIndex, SchemaBaseline, build_constraint_buffer,
    build_effects_buffer, build_index_buffer,
};

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
