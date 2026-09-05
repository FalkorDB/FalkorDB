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

pub mod announce;
pub mod error;
pub mod payload;
pub mod reader;
pub mod v3;
pub mod writer;

pub use error::DecodeError;
pub use reader::Reader;

use crate::graph::graph::Graph;
use announce::{AnnouncedConstraint, AnnouncedIndex, SchemaBaseline};
use atomic_refcell::AtomicRefCell;
use error::ApplyError;

use crate::runtime::pending::Pending;

/// Where a finished payload goes.
///
/// The seam, and the reason there is one: this crate has no Redis dependency
/// and must not grow one, but the format is what should be sending the payload
/// — the host has no business holding a half-finished buffer, and while it did,
/// it also ended up deciding what finishing meant.
///
/// So the format says *that* the bytes are sent and what they are; the host
/// says what sending means. One impl, over `redis_module::Context`, in the
/// crate that owns it.
pub trait ReplicationSink {
    fn replicate(
        &self,
        cmd: &str,
        args: &[&[u8]],
    );
}

/// A whole `GRAPH.EFFECT` payload, end to end, in one wire version.
///
/// [`EffectEncode`] and [`EffectDecode`] describe one *record*; this describes
/// the buffer that carries them, which is where the version actually lives —
/// the header names it, and whether the bytes are compressed is a property of
/// the payload rather than of any record in it. The version is a parameter for
/// the same reason it is on those two: so that a v4 is `impl EffectsFormat<4>
/// for EffectsPayload` beside the v3 impl and the compiler picks between them,
/// rather than a module name a caller has to spell.
///
/// Nothing outside this module implements or names it. The entry points are
/// [`EffectsPayload`]'s inherent methods, which decide *which* version applies
/// — and that decision is the whole reason the parameter is not a marker type.
///
/// Note what is **not** here: a way to finish a payload without sending it.
/// Finishing has to happen exactly once and last — running it per commit
/// produced a payload compressed twice that could not be read at all — and the
/// way to make that unrepeatable is to give no caller the option.
/// [`Self::replicate`] takes the buffer by value and is the only exit.
pub trait EffectsFormat<const VERSION: u8> {
    /// True for a payload that carries no records — a bare header.
    ///
    /// What "bare" means is the format's, because how long a header is belongs
    /// to it. A caller counting the bytes itself is a caller that has to be
    /// revisited when a version grows one.
    fn is_empty(buf: &[u8]) -> bool;

    /// Finish `buf` and send it as one `GRAPH.EFFECT` under `key`.
    ///
    /// By value, because a finished payload has no second use.
    fn replicate(
        sink: &dyn ReplicationSink,
        key: &[u8],
        buf: Vec<u8>,
    );

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

    // ── the write side ──
    //
    // Here for the same reason `apply` is: without it, reading dispatched
    // through this trait and writing did not dispatch at all — it went through
    // free functions re-exported out of one version's `emit`, which is how the
    // host came to import `build_constraint_buffer` directly. The input types
    // are [`announce`]'s, which describe a mutation rather than a wire, so a
    // second version encodes the same three types rather than defining its own.

    /// Digest a committed write into `buf`, appending to whatever is already
    /// there.
    ///
    /// Returns how many records were written.
    fn build(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        buf: &mut Vec<u8>,
    ) -> u64;

    /// Append one index DDL statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement names a label or property this graph
    /// has not registered, which would mean the DDL did not run.
    fn build_index(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        create: bool,
        index: &AnnouncedIndex<'_>,
        buf: &mut Vec<u8>,
    ) -> Result<(), String>;

    /// Append one constraint statement, with the status this node reached.
    ///
    /// # Errors
    ///
    /// Returns an error if a property is not registered, which would mean
    /// `create_constraint` did not run or did not register it.
    fn build_constraint(
        graph: &Graph,
        create: bool,
        constraint: &AnnouncedConstraint<'_>,
        baseline: &SchemaBaseline,
        buf: &mut Vec<u8>,
    ) -> Result<(), String>;
}

/// The version this build **writes**.
///
/// What it *reads* is every version [`EffectsPayload::apply`] can dispatch to,
/// which is not the same set and is not meant to be: a rolling upgrade runs a
/// new primary against old replicas and an old primary against new ones, so a
/// build that could only read its own output could never be deployed without
/// downtime. Today the two coincide because v3 is the only version that exists.
pub const WIRE_VERSION: u8 = v3::EFFECTS_VERSION;

/// A `GRAPH.EFFECT` payload, whatever version it is in.
///
/// The one type the rest of the codebase names. Its methods pick the format:
/// writing is always [`WIRE_VERSION`], and reading is whatever the buffer
/// declares in its first byte.
pub struct EffectsPayload;

impl EffectsPayload {
    /// True for a payload this build would have written that carries no
    /// records.
    ///
    /// About a buffer *being built here*, so it asks the version being written
    /// rather than reading a header that is not there yet.
    #[must_use]
    pub fn is_empty(buf: &[u8]) -> bool {
        <Self as EffectsFormat<WIRE_VERSION>>::is_empty(buf)
    }

    /// Digest a committed write, in the version this build writes.
    pub fn build(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        buf: &mut Vec<u8>,
    ) -> u64 {
        <Self as EffectsFormat<WIRE_VERSION>>::build(pending, graph, buf)
    }

    /// Append one index DDL statement, in the version this build writes.
    ///
    /// # Errors
    ///
    /// Whatever the format reports — see [`EffectsFormat::build_index`].
    pub fn build_index(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        create: bool,
        index: &AnnouncedIndex<'_>,
        buf: &mut Vec<u8>,
    ) -> Result<(), String> {
        <Self as EffectsFormat<WIRE_VERSION>>::build_index(pending, graph, create, index, buf)
    }

    /// Append one constraint statement, in the version this build writes.
    ///
    /// # Errors
    ///
    /// Whatever the format reports — see [`EffectsFormat::build_constraint`].
    pub fn build_constraint(
        graph: &Graph,
        create: bool,
        constraint: &AnnouncedConstraint<'_>,
        baseline: &SchemaBaseline,
        buf: &mut Vec<u8>,
    ) -> Result<(), String> {
        <Self as EffectsFormat<WIRE_VERSION>>::build_constraint(
            graph, create, constraint, baseline, buf,
        )
    }

    /// Finish a payload and send it, in the version this build writes.
    pub fn replicate(
        sink: &dyn ReplicationSink,
        key: &[u8],
        buf: Vec<u8>,
    ) {
        <Self as EffectsFormat<WIRE_VERSION>>::replicate(sink, key, buf);
    }

    /// Apply a payload, in the version **it** declares.
    ///
    /// Dispatch on the header rather than on `WIRE_VERSION`, because the sender
    /// is a different process on a different build: during a rolling upgrade a
    /// replica is routinely older or newer than its primary. A build reads
    /// every version it has an impl for.
    ///
    /// # Errors
    ///
    /// [`DecodeError::UnsupportedVersion`] for a version with no impl here —
    /// which is not a compatibility case but divergence, since it means a peer
    /// is speaking a language this build has never had. Also any
    /// [`ApplyError`] the chosen format returns.
    pub fn apply(
        graph: &mut Graph,
        buf: &[u8],
    ) -> Result<(), ApplyError> {
        // An empty buffer has no version byte to read. Nothing to apply either,
        // so this is not an error.
        let Some(&version) = buf.first() else {
            return Ok(());
        };
        match version {
            v3::EFFECTS_VERSION => {
                <Self as EffectsFormat<{ v3::EFFECTS_VERSION }>>::apply(graph, buf)
            }
            other => Err(DecodeError::UnsupportedVersion(other).into()),
        }
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

/// The one thing left that names a version, and it is configuration rather
/// than format: the compression threshold is read by `seal` itself, so the
/// setting stays behind the format boundary.
pub use v3::EFFECTS_COMPRESSION;
