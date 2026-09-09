//! Replication effects: the `GRAPH.EFFECT` payload a write produces and a
//! replica applies.
//!
//! One wire version so far, [`v3`] — the format both engines can read. Batched,
//! C-compatible widths. The version lives here and in the `impl
//! EffectsFormat<N>` under each version's module, nowhere else — the payload
//! and format types that the rest of the codebase names sit in the layer
//! above this one, which is where a caller meets them.
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

pub use error::{DecodeError, EncodeError};
pub use reader::Reader;
pub use writer::EffectWrite;

use std::fmt::Write as _;

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
///
/// The seam is a payload and the key it belongs to — deliberately not a command
/// name and an argv. That a payload travels as a `GRAPH.EFFECT` command is a
/// fact about Redis, and this crate does not know it is running inside Redis;
/// handing the format a command to name would have made it the format's fact,
/// and every future transport would have had to be spelled as a Redis command
/// to fit through.
pub trait ReplicationSink {
    /// Send one finished payload to the replicas, under `key`.
    fn replicate(
        &self,
        key: &[u8],
        payload: &[u8],
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
/// Nothing outside this crate implements or names it, and `pub(crate)` is
/// what says so — the sentence used to be a claim the compiler did not check. The entry points are
/// [`EffectsBuffer`]'s methods for writing and [`EffectsPayload`]'s for
/// reading, and it is the reading side that decides *which* version applies —
/// that decision is the whole reason the parameter is not a marker type.
///
/// Note what is **not** here: a way to finish a payload without sending it.
/// Finishing has to happen exactly once and last — running it per commit
/// produced a payload compressed twice that could not be read at all — and the
/// way to make that unrepeatable is to give no caller the option.
/// [`Self::replicate`] takes the buffer by value and is the only exit.
pub(crate) trait EffectsFormat<const VERSION: u8> {
    /// A new payload: framed, and holding no records.
    ///
    /// The buffer originates here rather than at a caller. It used to arrive
    /// as a bare `Vec::new()` from whoever wanted to write into it, which left
    /// every builder below opening with `if buf.is_empty() { …header… }` — the
    /// framing decided three times over, from a `Vec`'s length, by code that
    /// had no other business knowing a header existed. A buffer with junk in
    /// it would have been appended to and shipped unframed.
    fn new_buffer() -> Vec<u8>;

    /// True for a payload that carries no records — a bare header.
    ///
    /// What "bare" means is the format's, because how long a header is belongs
    /// to it. A caller counting the bytes itself is a caller that has to be
    /// revisited when a version grows one.
    fn is_empty(buf: &[u8]) -> bool;

    /// Finish `buf` and hand it to the sink as one payload under `key`.
    ///
    /// What sending means — and what command it travels as, if it travels as a
    /// command at all — belongs to the sink. By value, because a finished
    /// payload has no second use.
    fn replicate(
        sink: &dyn ReplicationSink,
        key: &[u8],
        buf: Vec<u8>,
    );

    /// Describe what this payload *says*, one line per record, for a
    /// divergence report.
    ///
    /// The format's half of [`EffectsPayload::describe`]: the caller renders
    /// the bytes, which are just bytes, and this renders their meaning, which
    /// is not. Best-effort by construction — it is called precisely because
    /// something in the payload was rejected, so it describes as far as it
    /// gets and then says where it stopped, rather than returning an error
    /// nobody could act on.
    fn describe(
        buf: &[u8],
        out: &mut Vec<String>,
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
    /// # Errors
    ///
    /// If a record cannot be encoded. That means the emitter built one wrong,
    /// which fails the write rather than shipping a payload the replica would
    /// read differently.
    fn build<W: EffectWrite + ?Sized>(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        buf: &mut W,
    ) -> Result<u64, String>;

    /// Append one index DDL statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement names a label or property this graph
    /// has not registered, which would mean the DDL did not run.
    fn build_index<W: EffectWrite + ?Sized>(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        create: bool,
        index: &AnnouncedIndex<'_>,
        buf: &mut W,
    ) -> Result<(), String>;

    /// Append one constraint statement, with the status this node reached.
    ///
    /// # Errors
    ///
    /// Returns an error if a property is not registered, which would mean
    /// `create_constraint` did not run or did not register it.
    fn build_constraint<W: EffectWrite + ?Sized>(
        graph: &Graph,
        create: bool,
        constraint: &AnnouncedConstraint<'_>,
        baseline: &SchemaBaseline,
        buf: &mut W,
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

/// How much of a diverged payload the log reproduces in hex.
///
/// A cap rather than the whole buffer because this runs on the path where a
/// replica is already about to resync the entire graph, and an unbounded dump
/// would put megabytes into the server log to explain a failure the lines above
/// it have usually already named. 2 KiB covers a typical effects payload whole
/// — the primary only takes this path when the payload is *smaller* than
/// shipping the query — and the total length is always reported, so a reader
/// can see when there was more.
const DESCRIBE_BYTE_LIMIT: usize = 2048;

/// Bytes per hex line. Two hex digits each, so 64 characters of payload per
/// line, which leaves room for the offset and the host's own prefix inside one
/// log message.
const DESCRIBE_BYTES_PER_LINE: usize = 32;

/// A `GRAPH.EFFECT` payload that arrived, whatever version it is in.
///
/// The read side. Which format applies is whatever the buffer declares in its
/// first byte, because the sender is a different process on a different build.
/// The write side is [`EffectsBuffer`], where the version is not a question:
/// this build writes [`WIRE_VERSION`].
pub struct EffectsPayload;

impl EffectsPayload {
    /// Describe a payload for a divergence report: what it says, then the
    /// bytes it arrived as.
    ///
    /// One line per element, because a Redis log message is truncated at a
    /// little over a kilobyte and a payload is not — the host logs these one
    /// at a time. The bytes come last and unconditionally, including for a
    /// version this build cannot read at all: they are the only part that is
    /// still true when the rendering above is empty or wrong, and they are
    /// what a reader diffs against the primary's own record of what it sent.
    #[must_use]
    pub fn describe(buf: &[u8]) -> Vec<String> {
        let mut out = vec![format!("{} bytes on the wire", buf.len())];
        match buf.first() {
            // Not a payload at all. The line above already said so.
            None => return out,
            Some(&v3::EFFECTS_VERSION) => {
                <Self as EffectsFormat<{ v3::EFFECTS_VERSION }>>::describe(buf, &mut out);
            }
            Some(&other) => out.push(format!(
                "version {other}: no format in this build reads it, so nothing below is decoded"
            )),
        }
        for (i, chunk) in buf
            .chunks(DESCRIBE_BYTES_PER_LINE)
            .take(DESCRIBE_BYTE_LIMIT / DESCRIBE_BYTES_PER_LINE)
            .enumerate()
        {
            let offset = i * DESCRIBE_BYTES_PER_LINE;
            let mut hex = String::with_capacity(chunk.len() * 2);
            for b in chunk {
                let _ = write!(hex, "{b:02x}");
            }
            out.push(format!("bytes[{offset:#06x}] {hex}"));
        }
        if buf.len() > DESCRIBE_BYTE_LIMIT {
            out.push(format!(
                "… {} more bytes not shown",
                buf.len() - DESCRIBE_BYTE_LIMIT
            ));
        }
        out
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
/// Blocks implement it too, so the version is stated once per block rather than
/// left to the module path. A v4 impl beside a v3 one makes every unannotated
/// `.encode(buf)` ambiguous, and the compiler then names each site that has to
/// choose — which a free function in a `v3/` directory could never do.
pub trait EffectEncode<const VERSION: u8> {
    /// # Errors
    ///
    /// Returns [`EncodeError`] when the value is internally inconsistent — an
    /// edge record whose endpoint columns do not match its ids, say. These were
    /// `debug_assert`s, which guarded the builds where a malformed record
    /// cannot reach a replica and vanished from the builds where it can.
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) -> Result<(), EncodeError>;
}

/// Types that can read themselves out of an effects payload of a given version.
pub trait EffectDecode<const VERSION: u8>: Sized {
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are malformed, truncated, or
    /// describe a shape this version does not have.
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError>;
}

/// Blocks the record around them gates, rather than sizes.
///
/// The mirror of [`EffectDecodeSized`] for the writing direction, and it exists
/// for the one case where the two are not symmetric. A *length* never needs to
/// reach an encoder — the writer holds the whole block, so a length argument
/// could only disagree with it, which is why [`EffectEncode`] takes none. But
/// `CREATE_INDEX`'s options are gated by `field_type`, which is not a length,
/// is not derivable from the data, and is what the *reader* gates on. Gating
/// the two directions on different things desynchronises the stream: the writer
/// emits `u64`s the reader never consumes and the next record is parsed from
/// the middle of them.
///
/// So both directions take the same context and state the same version, and a
/// v4 whose options layout differs lands as `impl EffectEncodeSized<4>` beside
/// this one with the compiler choosing between them.
pub trait EffectEncodeSized<const VERSION: u8> {
    /// What the surrounding record states, which this block is shaped by.
    type Size;

    /// # Errors
    ///
    /// Returns [`EncodeError`] when the block and the context it is written
    /// against disagree.
    fn encode_sized<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
        size: Self::Size,
    ) -> Result<(), EncodeError>;
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

/// A payload being built, in the version this build writes.
///
/// The write-side counterpart to [`EffectsPayload`], which is about a payload
/// that *arrived*. Both are needed and they are not the same thing: one is a
/// buffer this node is appending to and will eventually send, the other is a
/// slice some other node sent, whose version is whatever it says it is.
///
/// A `Vec<u8>` was standing in for this. That made the framing something each
/// builder had to notice was missing, let the host hand over a buffer it had
/// made itself, and gave anything holding one the ability to append arbitrary
/// bytes to a payload. There is no constructor here that does not frame it,
/// and no way to reach the bytes except by sending them.
///
/// Accumulating rather than one-shot on purpose: a query can commit more than
/// once — `Optional`, `Union`, `Apply`, `Merge` and `ForEach` all re-enter
/// `run_batch` — and every commit's records belong to the single
/// `GRAPH.EFFECT` that query replicates. So the buffer outlives any one build
/// call and lives on the runtime; what it must not do is originate there.
pub struct EffectsBuffer(Vec<u8>);

/// The sink half. Writing goes through [`EffectWrite`] like any other, which
/// is what lets the emitter be handed *this* rather than the `Vec` inside it.
impl std::io::Write for EffectsBuffer {
    fn write(
        &mut self,
        b: &[u8],
    ) -> std::io::Result<usize> {
        self.0.extend_from_slice(b);
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl EffectWrite for EffectsBuffer {
    fn bytes(
        &mut self,
        b: &[u8],
    ) {
        self.0.extend_from_slice(b);
    }

    fn written(&self) -> usize {
        self.0.len()
    }

    fn reserve(
        &mut self,
        n: usize,
    ) {
        self.0.reserve(n);
    }
}

impl Default for EffectsBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectsBuffer {
    /// A new payload, framed by the format and holding no records.
    #[must_use]
    pub fn new() -> Self {
        Self(<EffectsPayload as EffectsFormat<WIRE_VERSION>>::new_buffer())
    }

    /// True while this payload carries no records — a bare header.
    ///
    /// Asks the version being *written*, not one read off the bytes: the
    /// header here is one this build just put there.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        <EffectsPayload as EffectsFormat<WIRE_VERSION>>::is_empty(&self.0)
    }

    /// Digest a committed write into this payload.
    ///
    /// Returns how many records it added.
    ///
    /// # Errors
    ///
    /// Whatever the format reports — see [`EffectsFormat::build`].
    pub fn build(
        &mut self,
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
    ) -> Result<u64, String> {
        <EffectsPayload as EffectsFormat<WIRE_VERSION>>::build(pending, graph, self)
    }

    /// Append one index DDL statement.
    ///
    /// # Errors
    ///
    /// Whatever the format reports — see [`EffectsFormat::build_index`].
    pub fn build_index(
        &mut self,
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        create: bool,
        index: &AnnouncedIndex<'_>,
    ) -> Result<(), String> {
        <EffectsPayload as EffectsFormat<WIRE_VERSION>>::build_index(
            pending, graph, create, index, self,
        )
    }

    /// Append one constraint statement.
    ///
    /// # Errors
    ///
    /// Whatever the format reports — see [`EffectsFormat::build_constraint`].
    pub fn build_constraint(
        &mut self,
        graph: &Graph,
        create: bool,
        constraint: &AnnouncedConstraint<'_>,
        baseline: &SchemaBaseline,
    ) -> Result<(), String> {
        <EffectsPayload as EffectsFormat<WIRE_VERSION>>::build_constraint(
            graph, create, constraint, baseline, self,
        )
    }

    /// Finish this payload and hand it to the sink as one payload under `key`.
    ///
    /// By value: finishing happens once and last, and a finished payload has
    /// no second use. This is the only way the bytes leave.
    pub fn replicate(
        self,
        sink: &dyn ReplicationSink,
        key: &[u8],
    ) {
        <EffectsPayload as EffectsFormat<WIRE_VERSION>>::replicate(sink, key, self.0);
    }
}

/// The one thing left that names a version, and it is configuration rather
/// than format: the compression threshold is read by `seal` itself, so the
/// setting stays behind the format boundary.
pub use v3::EFFECTS_COMPRESSION;
