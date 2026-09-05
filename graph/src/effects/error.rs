//! Why an effects buffer could not be decoded, or could not be applied.

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
}

// ── apply errors ──

/// Why an effects buffer could not be applied.
///
/// Divergence is the interesting half. `Decode` means the bytes were malformed;
/// everything below it means the bytes were *well formed* and described a graph
/// this replica does not have — which is the failure this format exists to make loud.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ApplyError {
    #[error(transparent)]
    Decode(#[from] DecodeError),

    /// The replica would have assigned a different id to a new schema entry or
    /// attribute. C's reader cannot see this case: it only refuses a name that
    /// already exists locally, which misses a replica whose dictionary is a
    /// different length, where appending the same new name yields a different
    /// id. That is the case that silently put a property value on the wrong
    /// attribute.
    #[error(
        "effects buffer assigns {kind} '{name}' id {expected}, but this replica would \
         assign {assigned}{local}. The two engines have diverged; the buffer was not applied."
    )]
    IdMismatch {
        kind: &'static str,
        name: String,
        expected: i64,
        assigned: i64,
        /// What the wire's id names locally, when it names anything.
        local: LocalName,
    },

    /// An id resolved, but to a different name. Mirrors C's `VerifySchema` /
    /// `VerifyAttribute`: the id is authoritative, the name is the cross-check.
    #[error("effects buffer references {kind} '{name}' (id {id}), which is '{local}' here")]
    NameMismatch {
        kind: &'static str,
        name: String,
        id: i64,
        local: String,
    },

    #[error("effects buffer references {kind} '{name}' (id {id}), which does not exist here")]
    Unresolved {
        kind: &'static str,
        name: String,
        id: i64,
    },

    /// A `CREATE_NODE` names an id that is neither in this replica's recycle
    /// bin nor past the first id it has never allocated — so it is already live
    /// here.
    ///
    /// Node ids are not carried by any record that could report a disagreement
    /// about them: the next fresh id is derived from `node_count` and the bin,
    /// and `create_nodes` removes ids from the bin whether or not they were in
    /// it. Left unchecked, a drift stays invisible until the replica is
    /// promoted and hands out an id that is already in use.
    #[error(
        "effects buffer creates node {id}, which is already live on this replica          (recycle bin holds {bin} ids, first unallocated id {first_unallocated}). The two engines          have diverged; the buffer was not applied."
    )]
    NodeAlreadyLive {
        id: u64,
        bin: u64,
        first_unallocated: u64,
    },

    /// A `DELETE_NODE` names an id this replica does not hold live — either it
    /// is already in the recycle bin, or it was never allocated.
    #[error(
        "effects buffer deletes node {id}, which is not live on this replica          ({reason}). The two engines have diverged; the buffer was not applied."
    )]
    NodeNotLive { id: u64, reason: &'static str },

    #[error("{kind} id {id} out of range")]
    IdOutOfRange { kind: &'static str, id: i64 },

    #[error("unknown {kind}: {value}")]
    UnknownDiscriminant { kind: &'static str, value: u32 },

    #[error("record declares {entities} entities x {width} attributes but carries {values} values")]
    ShapeMismatch {
        entities: usize,
        width: usize,
        values: usize,
    },

    /// An `AttrSet` that is not strictly ascending.
    ///
    /// The attribute stores take the record's ids as a *span* and merge it into
    /// a sorted one, so wire order is load-bearing rather than cosmetic.
    #[error("attribute ids must be strictly ascending, got {first} before {second}")]
    AttrIdsNotAscending { first: u16, second: u16 },

    /// A create-constraint record with no status field.
    ///
    /// Unreachable through the decoder, which derives the presence from the
    /// opcode — an error rather than a panic because the apply path must never
    /// take a replica down over a malformed buffer.
    #[error("create-constraint record carries no status")]
    MissingConstraintStatus,

    #[error("index options must be a map, got {0}")]
    OptionsNotAMap(String),

    /// The graph rejected the mutation. Still a `String` because that is what
    /// every `Graph` method returns; wrapping it keeps the apply path's own
    /// failures distinguishable from the graph's.
    #[error("{0}")]
    Graph(String),
}

/// What an id names locally, rendered for the `IdMismatch` message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalName(pub Option<String>);

impl std::fmt::Display for LocalName {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match &self.0 {
            Some(n) => write!(f, " (that id is '{n}' here)"),
            None => Ok(()),
        }
    }
}
