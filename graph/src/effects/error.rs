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
    /// attribute. C's reader cannot see this case: its `ADD_SCHEMA` and
    /// `ADD_ATTRIBUTE` records carry no id at all — only a name — and
    /// `ApplyAddSchema`/`ApplyAddAttribute` refuse just the name that already
    /// exists locally (`src/effects/effects_apply.c`). That misses a replica
    /// whose dictionary is a different length, where appending the same new name
    /// yields a different id. That is the case that silently put a property
    /// value on the wrong attribute.
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

    /// A create names an id that is neither in this replica's recycle bin nor
    /// past the first id it has never allocated — so it is already live here.
    /// `kind` says which entity: nodes and relationships are checked the same
    /// way, against the same [`crate::graph::id_space::IdSpace`].
    ///
    /// Ids are not carried by any record that could report a disagreement about
    /// them: the next fresh id is derived from the entity's count and its bin,
    /// and the create removes ids from the bin whether or not they were in it.
    /// Left unchecked, a drift stays invisible until the replica is promoted and
    /// hands out an id that is already in use.
    #[error(
        "effects buffer creates {kind} {id}, which is already live on this replica (the \
         boundary it was judged against is {first_unallocated}). The two engines have \
         diverged; the buffer was not applied."
    )]
    AlreadyLive {
        kind: &'static str,
        id: u64,
        first_unallocated: u64,
    },

    /// A delete names an id this replica does not hold live — either it is
    /// already in the recycle bin, or it was never allocated. `kind` says which
    /// entity: nodes and relationships are checked the same way, against the same
    /// [`crate::graph::id_space::IdSpace`].
    #[error(
        "effects buffer deletes {kind} {id}, which is not live on this replica ({reason}). \
         The two engines have diverged; the buffer was not applied."
    )]
    NotLive {
        kind: &'static str,
        id: u64,
        reason: &'static str,
    },

    /// A record names an id with nothing past it.
    ///
    /// `u64::MAX` cannot be created: there is no boundary above it, and a master
    /// that had genuinely handed out 2^64 ids would have exhausted memory long
    /// before reaching the top.
    #[error(
        "effects buffer creates {kind} {id}, which is past the end of the id space. \
         The two engines have diverged; the buffer was not applied."
    )]
    IdPastEndOfSpace { kind: &'static str, id: u64 },

    /// The batch left ids allocated between the boundary it started from and the
    /// highest id it created, without creating them.
    ///
    /// An allocator hands out the lowest free id, so it cannot reach an id
    /// without having handed out everything below it. A buffer that leaves a
    /// hole was not produced by one — the likeliest cause is a replica that has
    /// missed a buffer, and accepting it would leave an id space the master does
    /// not have.
    #[error(
        "effects buffer allocated {kind} ids {entry_bound}..={highest} but created only \
         {created} of them. The two engines have diverged; the buffer was not applied."
    )]
    IdsHaveAHole {
        kind: &'static str,
        entry_bound: u64,
        highest: u64,
        created: u64,
    },

    /// The graph's own id boundary for `kind` is not where the ids it was given
    /// put it.
    ///
    /// The entity's count is an independent counter, so the same id applied twice
    /// moves it twice while the set of ids does not change. This is the only
    /// place anything checks that counter against a value not derived from it.
    #[error(
        "effects buffer left this replica's {kind} id boundary at {graph_bound}, but the \
         ids it carried put it at {expected}. The two engines have diverged; the buffer \
         was not applied."
    )]
    CountMiscounted {
        kind: &'static str,
        graph_bound: u64,
        expected: u64,
    },

    /// A schema id the local dictionary does not hold.
    ///
    /// The field is unsigned on the wire, so C's sentinels cannot arrive as
    /// themselves — `GRAPH_NO_LABEL` (-1) reads as 4294967295 and lands here.
    /// That is the right outcome and the number is the honest one: those values
    /// are not schema ids, and a payload naming one has diverged whichever way
    /// it is spelled.
    #[error("{kind} id {id} out of range")]
    IdOutOfRange { kind: &'static str, id: i64 },

    /// An `UPDATE_EDGE` that carried no relationship type.
    ///
    /// Its own variant rather than an `IdOutOfRange` with a made-up id: nothing
    /// was out of range, the field was absent, and reporting it as "id -1 out
    /// of range" invented exactly the sentinel this format does not use.
    #[error(
        "effects buffer updates an edge without naming its relationship type. \
         The two engines have diverged; the buffer was not applied."
    )]
    MissingRelType,

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

    #[error("index option is not supported by this engine: {0}")]
    UnsupportedIndexOption(String),

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
