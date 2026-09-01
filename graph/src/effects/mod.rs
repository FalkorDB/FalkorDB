//! Replication effects: the `GRAPH.EFFECT` payload a write produces and a
//! replica applies.
//!
//! Split by wire version, because two of them coexist during a migration:
//!
//! - [`v3`] — the format both engines can read. Batched, C-compatible widths.
//! - [`v2`] — Rust's own, predating C compatibility. On its way out; deleting
//!   the directory is what retires it.
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
pub mod v2;
pub mod v3;
pub mod writer;

pub use error::DecodeError;
pub use reader::Reader;

use std::sync::atomic::{AtomicI64, Ordering};

/// Which effects wire format this node **emits**.
///
/// Reading is version-dispatched, so a node always reads both. This only picks
/// what it writes, and it defaults to 2: a master emitting v3 at a peer that
/// cannot read it is silent data loss, not a degraded mode. The ordering is the
/// ordinary one — upgrade readers, then flip writers.
///
/// Lives here rather than in the module's config because choosing *between*
/// the versions is this module's job; `GRAPH.CONFIG SET EFFECTS_VERSION`
/// forwards to it.
pub static EFFECTS_EMIT_VERSION: AtomicI64 = AtomicI64::new(2);

/// How long a mutation must have taken, on average, before its effects are
/// worth sending instead of replaying the query verbatim. Microseconds; 0 sends
/// them always.
///
/// Lives here rather than in the module's config for the same reason
/// [`EFFECTS_EMIT_VERSION`] does: the code that weighs it is the code that built
/// the payload. `GRAPH.CONFIG SET EFFECTS_THRESHOLD` forwards to it.
pub static EFFECTS_THRESHOLD: AtomicI64 = AtomicI64::new(300);

/// The buffer builder matching [`EFFECTS_EMIT_VERSION`].
///
/// Only correct for a buffer that does not exist yet — use [`emit_v3_for`] to
/// add to one that does.
#[must_use]
pub fn emit_v3() -> bool {
    EFFECTS_EMIT_VERSION.load(Ordering::Relaxed) >= 3
}

/// Which wire version `buf` is being built in: its own header if it has one,
/// otherwise the configured version.
///
/// The config decides only the *first* record. `EFFECTS_EMIT_VERSION` is a
/// `Relaxed` load and `GRAPH.CONFIG SET EFFECTS_VERSION` runs on the main
/// thread, so it can change part-way through a query — and a query commits more
/// than once into one buffer (`FOREACH`, `MERGE`, `UNION`, a `WITH` after a
/// write). Re-reading the config per commit therefore let one buffer be stamped
/// `[3, flags]` in commit 1 and carry v2 records appended in commit 2. Nothing
/// downstream catches that: both builders write a header only when the buffer is
/// empty and neither inspects the version already there, `header_len` reads the
/// v3 byte and accepts the length, so the replica routes the whole payload to
/// `v3::apply` and loses the mutation to a `BadOpcode`.
///
/// Keyed off the buffer's own header for the same reason `replicate_effects`
/// keys compression off it: it cannot then disagree with what was actually
/// built.
#[must_use]
pub fn emit_v3_for(buf: &[u8]) -> bool {
    match buf.first() {
        Some(&version) => version >= v3::EFFECTS_VERSION,
        None => emit_v3(),
    }
}

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
