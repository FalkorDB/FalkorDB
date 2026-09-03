//! Why a v3 buffer could not be decoded.

use thiserror::Error;

use super::v3::EFFECTS_VERSION;

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

    /// An encoding discriminator that is neither [`ENC_PLAIN`] nor
    /// [`ENC_COMPRESSED`].
    #[error("unknown block encoding {0}")]
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

    /// An `SIType` bit pattern with no corresponding [`Value`].
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

    /// A list or map nested deeper than the decoder will recurse.
    ///
    /// Width is bounded by `guard_count`; depth costs one stack frame and about
    /// eight wire bytes per level, so it needs its own bound or a small buffer
    /// overflows the stack — a crash rather than a `DecodeError`.
    #[error("value nested deeper than the maximum of {max}")]
    ValueTooDeep { max: usize },
}
