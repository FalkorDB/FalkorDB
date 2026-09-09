//! The blocks a record composes from, other than its ids.
//!
//! Each is a type with an `EffectEncode<3>` / `EffectDecode<3>` impl rather than
//! a pair of free functions, so the wire version is stated on the block itself.
//! A v4 that shares a block re-implements it for 4; one that does not, cannot
//! reach this one by accident.
//!
//! The types are thin wrappers, generic over their storage so writing borrows
//! and reading owns — the same shape `AttrRef<S>` already uses in `records`.
//!
//! `IdList` — the id block — lives in its own module: it carries an encoding
//! ladder and the measurements behind it, which is more than the rest of these
//! put together.

use crate::runtime::value::Value;

use super::value::MIN_VALUE_BYTES;
use super::{
    DecodeError, EffectDecode, EffectDecodeSized, EffectEncode, EffectWrite, EncodeError, Reader,
};

// ── RelType ──

/// `RelType` — one relationship id per record, part of the partition key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelType(pub u32);

impl EffectEncode<3> for RelType {
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) -> Result<(), EncodeError> {
        buf.schema_id(self.0);
        Ok(())
    }
}

impl EffectDecode<3> for RelType {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(Self(r.u32()?))
    }
}

// ── LabelSet ──

/// `LabelSet` — `u16 n` · `i32 × n`, stated once per record.
///
/// Always plain, never roaring: roaring's 27–30 byte floor exceeds the whole
/// block until roughly seven labels, and the block is already amortised over
/// every row in the record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelSet<S = Vec<u32>>(pub S);

impl<S: AsRef<[u32]>> EffectEncode<3> for LabelSet<S> {
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) -> Result<(), EncodeError> {
        let labels = self.0.as_ref();
        // Count and payload together, before either is written: the exact size
        // is known here, so the block costs at most one growth however long it
        // is.
        buf.reserve(2 + labels.len() * 4);
        buf.u16(labels.len() as u16);
        for &l in labels {
            buf.u32(l);
        }
        Ok(())
    }
}

impl EffectDecode<3> for LabelSet {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let n = r.u16()?;
        Ok(Self(r.take_n(u64::from(n), u32::from_le_bytes)?))
    }
}

// ── AttrSet ──

/// `AttrIds` — `u16 n` · `u16 attr_id × n`. The schema half of the old
/// `AttrSet`.
///
/// The attribute ids are stated **once**, not per row: that removes exactly
/// `2 × count × n` bytes, which is provable arithmetic rather than a measured
/// effect.
///
/// Split from the values so that every record states its whole schema before
/// any of its data — labels, then attribute ids, then ids, then values. A
/// reader can then resolve and verify the schema against its own dictionaries
/// before it touches a single row, instead of discovering a divergence halfway
/// through applying one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttrIds<S = Vec<u16>>(pub S);

impl<S: AsRef<[u16]>> EffectEncode<3> for AttrIds<S> {
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) -> Result<(), EncodeError> {
        let attr_ids = self.0.as_ref();
        buf.reserve(2 + attr_ids.len() * 2);
        buf.u16(attr_ids.len() as u16);
        for &id in attr_ids {
            buf.u16(id);
        }
        Ok(())
    }
}

impl EffectDecode<3> for AttrIds {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let n = r.u16()?;
        Ok(Self(r.take_n(u64::from(n), u16::from_le_bytes)?))
    }
}

/// `AttrValues` — `SIValue × (count × n)`, row-major. The data half.
///
/// Two numbers describe the block and neither is on the wire, because both are
/// already stated: `count` is the record's header — how many entities it covers
/// — and `n` is the length of its `AttrIds`, how many attributes each of them
/// carries. So `rows` is `count × n` values, entity by entity, and the k-th
/// entity's j-th attribute is at `k * n + j`.
///
/// That is why decoding goes through [`EffectDecodeSized`] and encoding does
/// not: the writer holds the whole block already, and a *length* argument there
/// could only disagree with it. The reader has to size a `Vec` before it has
/// read anything.
///
/// That reasoning is about lengths and does not generalise. `EffectEncodeSized`
/// exists for the case it does not cover — `CREATE_INDEX`'s options, which the
/// record *gates* rather than sizes, on a `field_type` the writer cannot derive
/// from the block and the reader has to be given.
///
/// `T_NULL` in a slot means "remove this attribute": FalkorDB never stores a null
/// property, so `SET n.x = NULL` is a removal and this is how it replicates.
/// Shapes are therefore exact — see `emit::gather_rows` for why no row is padded.
///
/// Rows stay row-major. Grouping values by attribute instead saves zero bytes
/// uncompressed and changes sign with the data once compressed, which does not
/// justify a second layout two engines must match byte-for-byte.
#[derive(Clone, Debug, PartialEq)]
pub struct AttrValues<S = Vec<Value>>(pub S);

impl<S: AsRef<[Value]>> EffectEncode<3> for AttrValues<S> {
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) -> Result<(), EncodeError> {
        let rows = self.0.as_ref();
        // A floor, not the size: a value is at least its 4-byte type tag, and
        // most carry a payload after it. Still worth reserving — this is the
        // largest block in a record by far.
        buf.reserve(rows.len() * MIN_VALUE_BYTES);
        for value in rows {
            value.encode(buf)?;
        }
        Ok(())
    }
}

impl EffectDecodeSized<3> for AttrValues {
    /// `(count, attrs_per_row)`. The second is a *number of attributes*, not a
    /// byte width — the two sit next to each other here and used to share the
    /// name `width`.
    type Size = (u32, usize);

    fn decode_sized(
        r: &mut Reader<'_>,
        (count, attrs_per_row): Self::Size,
    ) -> Result<Self, DecodeError> {
        let total = u64::from(count).saturating_mul(attrs_per_row as u64);
        let total = r.guard_count(total, MIN_VALUE_BYTES)?;
        let mut rows = Vec::with_capacity(total);
        for _ in 0..total {
            rows.push(Value::decode(r)?);
        }
        Ok(Self(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A count off the wire reaches `Vec::with_capacity` before a single item is
    /// read. Without the guard these allocate from the claim rather than from
    /// the bytes: four bytes of input ask for gigabytes.
    ///
    /// Sized against `remaining()`, so the assertion is not "some limit" but
    /// "more than this buffer could possibly hold".
    #[test]
    fn a_count_larger_than_the_buffer_is_refused_before_allocating() {
        // Two values' worth of payload, claiming u32::MAX rows of one attribute.
        let mut buf = Vec::new();
        Value::Int(1).encode(&mut buf).unwrap();
        Value::Int(2).encode(&mut buf).unwrap();
        let mut r = Reader::new(&buf);
        assert!(
            matches!(
                AttrValues::decode_sized(&mut r, (u32::MAX, 1)),
                Err(DecodeError::ImplausibleCount { .. })
            ),
            "a row count the buffer cannot hold must be refused"
        );

        // And the product of the two must not wrap: count x attrs_per_row
        // overflows u64 unless the multiply saturates.
        let mut r = Reader::new(&buf);
        assert!(
            matches!(
                AttrValues::decode_sized(&mut r, (u32::MAX, usize::MAX)),
                Err(DecodeError::ImplausibleCount { .. })
            ),
            "count x attrs_per_row must saturate, not wrap to a small read"
        );
    }

    // ── IdList, enc 2: the sorted bitmap ──

    #[test]
    fn rel_type_is_four_signed_bytes() {
        let mut buf = Vec::new();
        RelType(7).encode(&mut buf).unwrap();
        assert_eq!(format!("{buf:02x?}"), "[07, 00, 00, 00]");
        let mut r = Reader::new(&buf);
        assert_eq!(RelType::decode(&mut r).unwrap(), RelType(7));
    }

    #[test]
    fn label_set_pins_its_bytes() {
        let mut buf = Vec::new();
        LabelSet(&[7, 9][..]).encode(&mut buf).unwrap();
        assert_eq!(
            format!("{buf:02x?}"),
            "[02, 00, 07, 00, 00, 00, 09, 00, 00, 00]"
        );
        let mut r = Reader::new(&buf);
        assert_eq!(LabelSet::decode(&mut r).unwrap().0, vec![7, 9]);
    }

    #[test]
    fn attr_ids_and_values_are_separate_halves() {
        // Two rows, one attribute: the id appears once, the values twice — and
        // the two halves are written apart so a record can state its whole
        // schema before any of its data.
        let mut ids = Vec::new();
        AttrIds(&[0][..]).encode(&mut ids).unwrap();
        assert_eq!(format!("{ids:02x?}"), "[01, 00, 00, 00]");

        let mut vals = Vec::new();
        AttrValues(&[Value::Int(1), Value::Int(2)][..])
            .encode(&mut vals)
            .unwrap();
        assert_eq!(
            format!("{vals:02x?}"),
            concat!(
                "[00, 20, 00, 00, 01, 00, 00, 00, 00, 00, 00, 00, ",
                "00, 20, 00, 00, 02, 00, 00, 00, 00, 00, 00, 00]"
            )
        );

        let mut r = Reader::new(&ids);
        assert_eq!(AttrIds::decode(&mut r).unwrap().0, vec![0]);
        assert!(r.is_empty());

        let mut r = Reader::new(&vals);
        assert_eq!(
            AttrValues::decode_sized(&mut r, (2, 1)).unwrap().0,
            vec![Value::Int(1), Value::Int(2)]
        );
        assert!(r.is_empty());
    }

    #[test]
    fn attr_values_null_marks_an_absent_property() {
        // T_NULL is unambiguous: FalkorDB never stores a null property value.
        let mut buf = Vec::new();
        AttrValues(&[Value::Int(1), Value::Null, Value::Null, Value::Int(2)][..])
            .encode(&mut buf)
            .unwrap();
        let mut r = Reader::new(&buf);
        let rows = AttrValues::decode_sized(&mut r, (2, 2)).unwrap().0;
        assert_eq!(rows[1], Value::Null);
        assert_eq!(rows[3], Value::Int(2));
    }
}
