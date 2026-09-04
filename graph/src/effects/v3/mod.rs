//! Wire format **v3** — see `docs/effects-v3.md`.
//!
//! Every record is `u32 opcode · u32 count · blocks…`, built from five shared
//! blocks. There is no separate "batch" record type: a record with `count == 1`
//! and one with `count == 10_000` are the same record, so no record type is left
//! un-batchable and the decoder has one shape per opcode.
//!
//! This module is the codec only. It is deliberately free of `Pending` and
//! `Graph` so the format can be reviewed, and tested byte-for-byte, on its own.
//!
//! ## The blocks
//!
//! | block | layout |
//! | --- | --- |
//! | `IdList` | `u32 n_segments` · `Segment × n_segments` |
//! | `RelType` | `i32` |
//! | `LabelSet` | `u16 n` · `i32 × n` |
//! | `AttrIds` | `u16 n` · `u16 attr_id × n` |
//! | `AttrValues` | `SIValue × (count × n)`, row-major |
//!
//! An `IdList` is a sequence of self-describing segments, each a consecutive
//! `Range`, a `Repeat` of one id, or an `Ascending` roaring bitmap — see
//! [`IdList`]. There is no plain form and no dictionary: a `Range` of one
//! describes a single id, so duplicates and disorder are the ordinary case
//! rather than encodings of their own. A supernode's endpoint column, which is
//! one id repeated, is a `Repeat` and cannot become a bitmap — a bitmap holds a
//! value once, and the column's whole content is that it does not.
//!
//! ## Row order is id order
//!
//! Row *k* belongs to the k-th id in the record's `IdList`, **as written** — not
//! the k-th smallest. No per-row id is sent, so the segment list must total the
//! record's count and an `Ascending` segment's cardinality must match what the
//! record still owes: one id short would land every later row on the wrong
//! entity, silently.
//!
//! Nothing may reorder a list to make an encoding eligible. A bitmap sorts and
//! deduplicates, so it is only ever reached by collapsing ranges that were
//! already ascending.
//!
//! ## Why the widths matter more than usual
//!
//! A wrong width does not produce a decode error on the far side. C reads the
//! misaligned bytes as a type tag and writes through the resulting pointer — the
//! `AttributeSet_Update` segfault observed when feeding C a Rust buffer. Every
//! width here came from C source (`src/effects/effects.c`, `effects.h`), not from
//! inference, and the tests below pin the bytes rather than round-tripping only
//! against ourselves.
//!
//! ## Records
//!
//! Batchable records are `u32 opcode · u32 count · blocks…`. `ADD_SCHEMA` and
//! `ADD_ATTRIBUTE` are inherently singular — one schema, one name — so they
//! carry an opcode but no count.
//!
//! Where a record names the entity's schema membership, node and edge forms
//! fill the same slot with different blocks: `UPDATE_NODE` a `LabelSet`,
//! `UPDATE_EDGE` a `RelType`. Both are the group's partition key and the
//! identity the replica can check the record against — see [`write_update`].

use crate::{
    entity_type::EntityType,
    graph::{
        constraint::{ConstraintStatus, ConstraintType},
        graphblas::serialization::index_field_type,
    },
};

/// Buffer header. C accepts any version `<= EFFECTS_VERSION` and branches per
/// version, so raising this is its established mechanism rather than a break —
/// but C must be raised to 3 as well before it can read what we write.
pub const EFFECTS_VERSION: u8 = 3;

pub mod apply;
pub mod blocks;
pub mod emit;
pub mod format;
mod id_list;
pub mod records;
#[cfg(test)]
mod staging;
pub mod value;

use num_enum::TryFromPrimitive;
use std::sync::atomic::{AtomicI64, Ordering};

/// Smallest v3 payload worth compressing, in bytes. **0 disables it.**
///
/// Off by default because compression is a bandwidth trade, not a CPU one:
/// measured at 3.245 cycles/byte to compress against 0.067 to copy into the
/// replica output buffer, so on a fast link it spends far more than the bytes
/// are worth. Worth turning on when the replication link, not the write thread,
/// is the constraint.
///
/// v2 ignores this — only v3 reserves a flags byte to say a payload is
/// compressed.
pub static EFFECTS_COMPRESSION: AtomicI64 = AtomicI64::new(0);

/// The size threshold, as a `usize`. Negative or absurd values read as off.
fn compression_min_bytes() -> usize {
    usize::try_from(EFFECTS_COMPRESSION.load(Ordering::Relaxed)).unwrap_or(0)
}

/// Finish a payload: compress it if the configuration says it is worth it.
///
/// The one entry point the replication layer needs. It used to reach in for
/// `maybe_compress` and `compression_min_bytes` separately, which put the
/// decision of *whether* to compress outside the format — and made the
/// threshold `pub` for no other reason.
///
/// Call once, on a complete buffer. `maybe_compress` refuses a payload already
/// marked compressed, so a second call is a no-op rather than corruption, but
/// the contract is once.
pub fn seal(buf: &mut Vec<u8>) {
    maybe_compress(buf, compression_min_bytes());
}

pub use blocks::*;
pub use id_list::*;
pub use records::*;

// The cursor, the primitive writers and the codec traits are shared across
// versions, so they live a level up. Re-exported here so a record module can
// say `use super::*` and get everything the wire needs.
pub use super::writer::*;
pub use super::{DecodeError, EffectDecode, EffectEncode, Reader};

// ── effect types ──

// What a record is. C's `EffectType`, so 4 bytes on the wire, and
// `EFFECT_UNKNOWN = 0` shifts every discriminant down by one relative to a
// naive 1-based list.
//
// An enum rather than constants because the set is closed and the decoder
// matches it exhaustively: adding a record type stops compiling until every
// match handles it, which is the failure mode worth having. The bit-flag sets
// below stay constants for the opposite reason — several of those OR together,
// which no enum can express.
/// The discriminants, the repr and the parse from one declaration.
///
/// `TryFromPrimitive` rather than a hand-rolled macro: `num_enum` is already a
/// dependency and already derived this way on three enums in
/// `index::redisearch`, and its `error_type` yields
/// `TryFrom<u32, Error = DecodeError>` directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, TryFromPrimitive)]
#[num_enum(error_type(name = DecodeError, constructor = DecodeError::BadOpcode))]
#[repr(u32)]
pub enum Opcode {
    UpdateNode = 1,
    UpdateEdge = 2,
    CreateNode = 3,
    CreateEdge = 4,
    DeleteNode = 5,
    DeleteEdge = 6,
    SetLabels = 7,
    RemoveLabels = 8,
    AddSchema = 9,
    AddAttribute = 10,
    CreateIndex = 11,
    DropIndex = 12,
    CreateConstraint = 13,
    DropConstraint = 14,
}

impl Opcode {
    /// Whether the record carries a `count` and blocks sized by it.
    ///
    /// The schema and DDL records are inherently singular — one schema, one
    /// name, one index field — so they carry an opcode and no count.
    #[must_use]
    pub const fn is_batchable(self) -> bool {
        matches!(
            self,
            Self::UpdateNode
                | Self::UpdateEdge
                | Self::CreateNode
                | Self::CreateEdge
                | Self::DeleteNode
                | Self::DeleteEdge
                | Self::SetLabels
                | Self::RemoveLabels
        )
    }
}

/// Which dictionary a schema record names: C's `EntityType`.
///
/// **Zero-based**, unlike [`EntityType`] on the constraint records, which C
/// numbers from 1. The two are the same distinction with different wire
/// encodings, and getting them the same way round is not optional.
/// The schema-dictionary tag: **0-based**, node then edge.
///
/// `EntityType` is the crate's one node-or-edge enum; the two numberings C uses
/// for it belong to the wire, not to the type, so they live here as
/// conversions. This one is `ADD_SCHEMA`'s.
#[must_use]
pub const fn schema_tag(entity: EntityType) -> u32 {
    match entity {
        EntityType::Node => 0,
        EntityType::Relationship => 1,
    }
}

/// Inverse of [`schema_tag`].
pub const fn entity_from_schema_tag(v: u32) -> Result<EntityType, DecodeError> {
    match v {
        0 => Ok(EntityType::Node),
        1 => Ok(EntityType::Relationship),
        other => Err(DecodeError::BadSchemaType(other)),
    }
}

/// C's `GraphEntityType` tag: **1-based**, because `GETYPE_UNKNOWN` takes 0.
///
/// Not the same numbering as [`schema_tag`], and the difference is not
/// cosmetic — a 0-based encoding here would make every node record read as
/// "unknown" to C.
#[must_use]
pub const fn entity_tag(entity: EntityType) -> u32 {
    match entity {
        EntityType::Node => 1,
        EntityType::Relationship => 2,
    }
}

/// Inverse of [`entity_tag`].
pub const fn entity_from_tag(v: u32) -> Result<EntityType, DecodeError> {
    match v {
        1 => Ok(EntityType::Node),
        2 => Ok(EntityType::Relationship),
        other => Err(DecodeError::BadEntityType(other)),
    }
}

/// Which `UPDATE_*` opcode targets this entity kind.
#[must_use]
pub const fn update_opcode(entity: EntityType) -> Opcode {
    match entity {
        EntityType::Node => Opcode::UpdateNode,
        EntityType::Relationship => Opcode::UpdateEdge,
    }
}

// ── index field types ──
//
// Also derived from the RDB module's copy of C's `index_field.h`. This is a
// **bit flag set**, not an ordinal — several can be OR'd — which is why
// `INDEX_FLD_RANGE` is the union of the three scalar kinds and reads as `0x0E`
// rather than having a discriminant of its own. A bit set is the one shape an
// enum genuinely cannot model, so these stay constants.

pub const INDEX_FLD_FULLTEXT: u32 = narrow_flag(index_field_type::INDEX_FLD_FULLTEXT);
pub const INDEX_FLD_NUMERIC: u32 = narrow_flag(index_field_type::INDEX_FLD_NUMERIC);
pub const INDEX_FLD_GEO: u32 = narrow_flag(index_field_type::INDEX_FLD_GEO);
pub const INDEX_FLD_STR: u32 = narrow_flag(index_field_type::INDEX_FLD_STR);
pub const INDEX_FLD_VECTOR: u32 = narrow_flag(index_field_type::INDEX_FLD_VECTOR);

pub const INDEX_FLD_UNKNOWN: u32 = 0x00;
/// `INDEX_FLD_NUMERIC | INDEX_FLD_GEO | INDEX_FLD_STR` = `0x0E`.
pub const INDEX_FLD_RANGE: u32 = INDEX_FLD_NUMERIC | INDEX_FLD_GEO | INDEX_FLD_STR;

// ── constraint types ──

/// C's `ConstraintType` tag. The enum itself is `graph::constraint::
/// ConstraintType`; only the numbering is the wire's business.
#[must_use]
pub const fn constraint_tag(kind: ConstraintType) -> u32 {
    match kind {
        ConstraintType::Unique => 0,
        ConstraintType::Mandatory => 1,
    }
}

/// C's `ConstraintStatus` tag.
///
/// **Not `as u32`.** C is `CT_ACTIVE = 0, CT_PENDING = 1, CT_FAILED = 2`
/// (`src/constraint/constraint.h:39` on master) while Rust's enum reads
/// `UnderConstruction, Operational, Failed` — so the first two are the other
/// way round, and casting the discriminant would send an active constraint as
/// pending and a pending one as active. Nothing would fail; the replica would
/// simply believe the wrong thing about whether the constraint is enforcing.
#[must_use]
pub const fn constraint_status_tag(status: ConstraintStatus) -> u32 {
    match status {
        ConstraintStatus::Operational => 0,
        ConstraintStatus::UnderConstruction => 1,
        ConstraintStatus::Failed => 2,
    }
}

/// Inverse of [`constraint_status_tag`].
pub const fn constraint_status_from_tag(v: u32) -> Result<ConstraintStatus, DecodeError> {
    match v {
        0 => Ok(ConstraintStatus::Operational),
        1 => Ok(ConstraintStatus::UnderConstruction),
        2 => Ok(ConstraintStatus::Failed),
        other => Err(DecodeError::BadConstraintStatus(other)),
    }
}

/// Inverse of [`constraint_tag`].
pub const fn constraint_from_tag(v: u32) -> Result<ConstraintType, DecodeError> {
    match v {
        0 => Ok(ConstraintType::Unique),
        1 => Ok(ConstraintType::Mandatory),
        other => Err(DecodeError::BadConstraintType(other)),
    }
}

// ── SIValue type tags ──
//
// Derived from `serialization::si_type` rather than restated: the RDB encoder
// already carries C's tags, and two copies of a bitmask would drift silently —
// nothing fails when a tag is wrong, the far side just reads the next field as
// a type. They are `u64` there because RDB writes them through
// `write_unsigned`, which emits a type byte plus a fixed 8-byte LE value; the
// effects wire is a bare 4 bytes, so these narrow.
//
// C's `SIType` is a **bitmask**, not an ordinal: each type is a distinct bit.
// Rust's own v2 codec used sequential 0..12 tags, which collide with these
// almost everywhere and are the single most dangerous divergence in the format.

/// Narrow a C flag constant to the `u32` the wire and the index API use.
const fn narrow_flag(v: u64) -> u32 {
    assert!(
        v <= u32::MAX as u64,
        "a C constant must fit the 4 bytes the wire reads"
    );
    v as u32
}

/// Write a C type constant as the four bytes the effects wire carries it in.
///
/// `serialization` holds these as `u64` because RDB writes them through
/// `write_unsigned`; the effects wire is a bare `u32`. Going through one
/// narrowing writer means the tags are never aliased into this module at all —
/// each one is named once, at its definition in `si_type`, so there is no second
/// copy to drift. The assertion is what stands between a wider C constant and a
/// silently truncated tag.
pub fn write_tag(
    buf: &mut Vec<u8>,
    v: u64,
) {
    write_u32(
        buf,
        u32::try_from(v).expect("a C constant must fit the 4 bytes the wire reads"),
    );
}

/// `T_MAP` has no RDB counterpart — the RDB path never stores a bare map — so
/// unlike the rest it is stated here.
///
/// `u64` like `si_type`'s own constants, so encode and decode treat it exactly
/// as they treat the derived ones — and so it can be a `match` pattern beside
/// them without a second, differently-typed copy.
pub const T_MAP: u64 = 1 << 0;

// ── payload header ──

/// The payload is a zstd frame, not records.
///
/// When set, a `u32` uncompressed length follows the flags byte and the frame
/// follows that. The header itself is never compressed, so a reader always
/// knows what it is holding before it commits to decoding anything.
pub const FLAG_COMPRESSED: u8 = 1 << 0;

/// Every flag this build understands. A reader that meets a bit outside this
/// mask **rejects the buffer** rather than guessing: an old node reading a
/// future payload must fail loudly, not decode the first record it recognises
/// and corrupt itself with the rest.
pub const KNOWN_FLAGS: u8 = FLAG_COMPRESSED;

/// Start a fresh buffer: `u8 version · u8 flags`.
///
/// The flags byte is reserved from day one even though nothing sets it by
/// default. Adding it later would have meant another version bump for a single
/// byte, which is the whole reason it is here now rather than when compression
/// is first switched on.
#[must_use]
pub fn new_buffer() -> Vec<u8> {
    vec![EFFECTS_VERSION, 0]
}
