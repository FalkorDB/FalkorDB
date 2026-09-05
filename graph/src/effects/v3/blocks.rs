//! The blocks a record composes from, other than its ids.
//!
//! `IdList` — the id block — lives in its own module: it carries an encoding
//! ladder and the measurements behind it, which is more than the rest of these
//! put together.

use crate::runtime::value::Value;

use super::{DecodeError, EffectDecode, EffectEncode, EffectWrite, Reader};

// ── RelType ──

/// `RelType` — one relationship id per record, part of the partition key.
pub fn write_rel_type(
    buf: &mut Vec<u8>,
    relation_id: i32,
) {
    buf.label_id(relation_id);
}

/// Inverse of [`write_rel_type`].
pub fn read_rel_type(r: &mut Reader<'_>) -> Result<i32, DecodeError> {
    r.i32()
}

// ── LabelSet ──

/// `LabelSet` — `u16 n` · `i32 × n`, stated once per record.
///
/// Always plain, never roaring: roaring's 27–30 byte floor exceeds the whole
/// block until roughly seven labels, and the block is already amortised over
/// every row in the record.
pub fn write_label_set(
    buf: &mut Vec<u8>,
    labels: &[i32],
) {
    // Count and payload together, before either is written: the exact size is
    // known here, so the block costs at most one growth however long it is.
    buf.reserve(2 + labels.len() * 4);
    buf.u16(labels.len() as u16);
    buf.extend(labels.iter().flat_map(|&l| l.to_le_bytes()));
}

/// Inverse of [`write_label_set`].
pub fn read_label_set(r: &mut Reader<'_>) -> Result<Vec<i32>, DecodeError> {
    let n = r.u16()?;
    r.take_n(u64::from(n), i32::from_le_bytes)
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
pub fn write_attr_ids(
    buf: &mut Vec<u8>,
    attr_ids: &[u16],
) {
    buf.reserve(2 + attr_ids.len() * 2);
    buf.u16(attr_ids.len() as u16);
    buf.extend(attr_ids.iter().flat_map(|&id| id.to_le_bytes()));
}

/// Inverse of [`write_attr_ids`].
pub fn read_attr_ids(r: &mut Reader<'_>) -> Result<Vec<u16>, DecodeError> {
    let n = r.u16()?;
    r.take_n(u64::from(n), u16::from_le_bytes)
}

/// `AttrValues` — `SIValue × (count × n)`, row-major. The data half.
///
/// Two numbers describe the block and neither is on the wire, because both are
/// already stated: `count` is the record's header — how many entities it covers
/// — and `n` is the length of its `AttrIds`, how many attributes each of them
/// carries. So `rows` is `count × n` values, entity by entity, and the k-th
/// entity's j-th attribute is at `k * n + j`.
///
/// This side takes neither: `rows` is the whole block and the record already
/// knows its own shape, so a `width` argument here could only disagree with it.
/// The reader takes both, because it has to size a `Vec` before it has read
/// anything.
///
/// `T_NULL` in a slot means "remove this attribute": FalkorDB never stores a null
/// property, so `SET n.x = NULL` is a removal and this is how it replicates.
/// Shapes are therefore exact — see `emit::gather_rows` for why no row is padded.
///
/// Rows stay row-major. Grouping values by attribute instead saves zero bytes
/// uncompressed and changes sign with the data once compressed, which does not
/// justify a second layout two engines must match byte-for-byte.
pub fn write_attr_values(
    buf: &mut Vec<u8>,
    rows: &[Value],
) {
    // A floor, not the size: a value is at least its 4-byte type tag, and most
    // carry a payload after it. Still worth reserving — this is the largest
    // block in a record by far.
    buf.reserve(rows.len() * 4);
    for value in rows {
        value.encode(buf);
    }
}

/// Inverse of [`write_attr_values`], reading `count × attrs_per_row` values.
///
/// `attrs_per_row` is a *number of attributes*, not a byte width — the two sit
/// next to each other in this module and used to share the name `width`.
pub fn read_attr_values(
    r: &mut Reader<'_>,
    count: u32,
    attrs_per_row: usize,
) -> Result<Vec<Value>, DecodeError> {
    let total = u64::from(count).saturating_mul(attrs_per_row as u64);
    // Every value is at least its 4-byte type tag.
    let total = r.guard_count(total, 4)?;
    let mut rows = Vec::with_capacity(total);
    for _ in 0..total {
        rows.push(Value::decode(r)?);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── IdList, enc 2: the sorted bitmap ──

    #[test]
    fn rel_type_is_four_signed_bytes() {
        let mut buf = Vec::new();
        write_rel_type(&mut buf, 7);
        assert_eq!(format!("{buf:02x?}"), "[07, 00, 00, 00]");
        let mut r = Reader::new(&buf);
        assert_eq!(read_rel_type(&mut r).unwrap(), 7);
    }

    #[test]
    fn label_set_pins_its_bytes() {
        let mut buf = Vec::new();
        write_label_set(&mut buf, &[7, 9]);
        assert_eq!(
            format!("{buf:02x?}"),
            "[02, 00, 07, 00, 00, 00, 09, 00, 00, 00]"
        );
        let mut r = Reader::new(&buf);
        assert_eq!(read_label_set(&mut r).unwrap(), vec![7, 9]);
    }

    #[test]
    fn attr_ids_and_values_are_separate_halves() {
        // Two rows, one attribute: the id appears once, the values twice — and
        // the two halves are written apart so a record can state its whole
        // schema before any of its data.
        let mut ids = Vec::new();
        write_attr_ids(&mut ids, &[0]);
        assert_eq!(format!("{ids:02x?}"), "[01, 00, 00, 00]");

        let mut vals = Vec::new();
        write_attr_values(&mut vals, &[Value::Int(1), Value::Int(2)]);
        assert_eq!(
            format!("{vals:02x?}"),
            concat!(
                "[00, 20, 00, 00, 01, 00, 00, 00, 00, 00, 00, 00, ",
                "00, 20, 00, 00, 02, 00, 00, 00, 00, 00, 00, 00]"
            )
        );

        let mut r = Reader::new(&ids);
        assert_eq!(read_attr_ids(&mut r).unwrap(), vec![0]);
        assert!(r.is_empty());

        let mut r = Reader::new(&vals);
        assert_eq!(
            read_attr_values(&mut r, 2, 1).unwrap(),
            vec![Value::Int(1), Value::Int(2)]
        );
        assert!(r.is_empty());
    }

    #[test]
    fn attr_values_null_marks_an_absent_property() {
        // T_NULL is unambiguous: FalkorDB never stores a null property value.
        let mut buf = Vec::new();
        write_attr_values(
            &mut buf,
            &[Value::Int(1), Value::Null, Value::Null, Value::Int(2)],
        );
        let mut r = Reader::new(&buf);
        let rows = read_attr_values(&mut r, 2, 2).unwrap();
        assert_eq!(rows[1], Value::Null);
        assert_eq!(rows[3], Value::Int(2));
    }
}
