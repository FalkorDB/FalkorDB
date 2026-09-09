//! Why an effects buffer could not be written, decoded, or applied.

use thiserror::Error;

use super::v3::EFFECTS_VERSION;

// ── encode errors ──

/// What an encoder can refuse to write.
///
/// Every variant is a record the emitter built inconsistently — not bad input
/// off a wire, which is [`DecodeError`]'s job, but a payload this engine was
/// about to *produce* wrong. They were `debug_assert`s, which is the worst
/// place for them: they fired in the builds where a malformed record cannot
/// reach a replica, and were compiled out of the builds where it can. In
/// release, each one shipped corrupt bytes to a peer and let the divergence
/// guard sort it out.
///
/// Returning instead means one write fails loudly on the primary rather than
/// one replica silently diverging.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum EncodeError {
    /// An edge record whose endpoint columns are not the same length as its
    /// ids.
    ///
    /// `src` and `dst` are columns, read positionally against `ids` — the k-th
    /// entry of each belongs to the k-th edge. A short column does not
    /// truncate the record, it shifts every later value by one and the reader
    /// has no way to notice.
    #[error("{column} column has {got} entries for {expected} edges")]
    EndpointColumnMisaligned {
        column: &'static str,
        expected: usize,
        got: usize,
    },

    /// A `CREATE_INDEX` whose options do not match the field type they are
    /// gated by.
    ///
    /// The reader decides whether a vector block follows from `field_type`
    /// alone. If the writer decides from its own `Option` instead, the two
    /// disagree: the writer emits `u64`s the reader never consumes, and the
    /// next record is parsed from the middle of them.
    #[error("field type {field_type:#06x} and the options block disagree about the vector half")]
    OptionsFieldTypeMismatch { field_type: u32 },

    /// A record header whose count does not match what its opcode allows.
    ///
    /// A batchable opcode needs a count and a singular one must not have it —
    /// the reader takes the next four bytes as a count for the first kind and
    /// as the record's first field for the second, so getting it wrong shifts
    /// everything after.
    #[error("opcode {opcode} was given the wrong kind of header")]
    HeaderShapeMismatch { opcode: u32 },

    /// A block whose item count does not fit the fixed-width field that
    /// carries it.
    ///
    /// `LabelSet` and `AttrIds` state their length in a `u16`. `as u16` would
    /// truncate — 65,536 labels would write a count of 0 and the reader would
    /// take the whole block as the next field. Neither is reachable from any
    /// statement this engine accepts, which is why it must be loud: if it ever
    /// fires, something upstream stopped bounding what it builds.
    #[error("{block} has {len} entries, more than its u16 count can carry")]
    BlockCountTooLarge { block: &'static str, len: usize },

    /// A roaring bitmap that did not serialize to the length it predicted.
    ///
    /// The length prefix is written from `serialized_size` before the bitmap
    /// is asked to serialize. If they disagree the prefix lies, and every
    /// reader takes the wrong number of bytes for it.
    #[error("bitmap predicted {predicted} bytes and wrote {written}")]
    BitmapLengthLied { predicted: usize, written: usize },
}

// ── decode errors ──

/// Why a buffer could not be decoded.
///
/// Every variant is fatal for the whole buffer: effects are applied
/// transactionally, so a record that does not decode means the replica has
/// diverged and must not apply a partial prefix.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum DecodeError {
    /// Fewer bytes remain than the next field needs.
    #[error("truncated effects buffer: want {want} bytes, have {have}")]
    UnexpectedEof { want: usize, have: usize },

    /// A count would require more bytes than the buffer can possibly hold.
    /// Checked *before* allocating, so a corrupt length cannot OOM us.
    #[error("effects record claims {count} entries but only {remaining} bytes remain")]
    ImplausibleCount { count: u64, remaining: usize },

    /// A segment header this build cannot read — an unknown kind, or one of
    /// the reserved bits set. Refused rather than masked off, so a future
    /// segment shape is not misread as a range by a build that predates it.
    #[error("unknown segment header {0:#04x}")]
    BadEncoding(u8),

    /// The roaring blob did not deserialize.
    #[error("malformed roaring bitmap: {0}")]
    BadRoaring(String),

    /// The decoded bitmap holds a different number of ids than the record's
    /// count — the guard that keeps row *k* bound to the right entity.
    #[error("effects record declares {expected} ids but its bitmap holds {actual}")]
    CardinalityMismatch { expected: u64, actual: u64 },

    /// An `IdList` rank that does not address the dictionary.
    #[error("dictionary rank {rank} out of range for {cardinality} entries")]
    RankOutOfRange { rank: u64, cardinality: u64 },

    /// An `idx_width` other than 1, 2, 4 or 8.
    #[error("unsupported index width {0}")]
    BadIndexWidth(u8),

    /// An `SIType` bit pattern with no corresponding
    /// [`crate::runtime::value::Value`].
    #[error("unknown SIValue type {0:#x}")]
    BadValueType(u32),

    /// A string field that is not valid UTF-8, or is missing its NUL.
    #[error("malformed string in effects buffer")]
    BadString,

    /// A buffer whose version byte this build cannot read.
    #[error("effects buffer is version {0}, this build reads {EFFECTS_VERSION}")]
    UnsupportedVersion(u8),

    /// An opcode with no record shape.
    #[error("unknown effect opcode {0}")]
    BadOpcode(u32),

    /// A payload flag this build does not understand. Rejected rather than
    /// ignored: decoding the records anyway would apply a prefix of something
    /// whose shape is unknown.
    #[error("effects payload sets unknown flags {0:#04x}")]
    UnknownFlags(u8),

    #[error("malformed compressed effects payload: {0}")]
    BadCompression(String),

    #[error("compressed payload declares {declared} bytes but expands to {actual}")]
    CompressedLengthMismatch { declared: usize, actual: usize },

    #[error("{after_frame} bytes follow the compressed frame")]
    TrailingBytes { after_frame: usize },

    #[error("record with opcode {opcode} covers no entities")]
    EmptyRecord { opcode: u32 },

    #[error("a boolean field holds {value}, which is neither 0 nor 1")]
    BadBool { value: u64 },

    /// A `simFunc` discriminant that is not a `VecSimMetric`.
    ///
    /// Refused rather than read as L2: silently choosing a metric would build
    /// an index that answers different queries from the primary's and never
    /// say so. The set is closed at 0, 1 and 2 — this engine's parser rejects
    /// any other name at creation, so a fourth value on the wire came from
    /// something this build does not understand.
    #[error("unknown vector similarity function {value}")]
    BadSimilarityFunction { value: u64 },

    #[error("unknown schema type {0}")]
    BadSchemaType(u32),

    #[error("unknown constraint type {0}")]
    BadConstraintType(u32),

    /// C numbers `GraphEntityType` from 1, so 0 reaches here as often as a
    /// genuinely corrupt value does.
    #[error("unknown graph entity type {0}")]
    BadEntityType(u32),

    /// The compressed payload inflated to bytes that do not match the checksum
    /// the writer recorded.
    #[error(
        "compressed payload checksum mismatch: header says {declared:#010x}, payload is {actual:#010x}"
    )]
    ChecksumMismatch { declared: u32, actual: u32 },

    #[error("unknown constraint status {0}")]
    BadConstraintStatus(u32),

    /// A `Range` block whose `base + count` would wrap past `u64`.
    #[error("id range starting at {base} cannot hold {count} ids")]
    BadRange { base: u64, count: u64 },
}
