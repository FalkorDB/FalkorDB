//! Effects wire format v3 — see `docs/effects-v3.md`.
//!
//! v3 is v2 with two record families redefined: entity creation carries an
//! explicit id, and label effects carry `(node, label)` pairs instead of a
//! serialized GraphBLAS vector. Everything else — primitive widths, the string
//! encoding, `SIValue` tagging — is C's v2, so the two engines can read each
//! other once C also implements v3.
//!
//! This module is the codec only. It is deliberately free of `Pending` and
//! `Graph` so the format can be reviewed, and tested byte-for-byte, on its own.
//!
//! ## Why the widths matter more than usual
//!
//! A wrong width does not produce a decode error on the far side. C reads the
//! misaligned bytes as a type tag and writes through the resulting pointer — the
//! `AttributeSet_Update` segfault observed when feeding C a Rust buffer. Every
//! width here came from C source (`src/effects/effects.c`, `effects.h`), not from
//! inference, and the tests below pin the bytes rather than round-tripping only
//! against ourselves.

use crate::runtime::value::Value;

/// Buffer header. C accepts any version `<= EFFECTS_VERSION` and branches per
/// version, so raising this is its established mechanism rather than a break —
/// but C must be raised to 3 as well before it can read what we write.
pub const EFFECTS_VERSION: u8 = 3;

// ── effect types ──
//
// C's `EffectType` is an enum, so 4 bytes on the wire, and `EFFECT_UNKNOWN = 0`
// shifts every discriminant down by one relative to a naive 1-based list.

pub const EFFECT_UPDATE_NODE: u32 = 1;
pub const EFFECT_UPDATE_EDGE: u32 = 2;
pub const EFFECT_CREATE_NODE: u32 = 3;
pub const EFFECT_CREATE_EDGE: u32 = 4;
pub const EFFECT_DELETE_NODE: u32 = 5;
pub const EFFECT_DELETE_EDGE: u32 = 6;
pub const EFFECT_SET_LABELS: u32 = 7;
pub const EFFECT_REMOVE_LABELS: u32 = 8;
pub const EFFECT_ADD_SCHEMA: u32 = 9;
pub const EFFECT_ADD_ATTRIBUTE: u32 = 10;
pub const EFFECT_CREATE_INDEX: u32 = 11;
pub const EFFECT_DROP_INDEX: u32 = 12;
pub const EFFECT_CREATE_CONSTRAINT: u32 = 13;
pub const EFFECT_DROP_CONSTRAINT: u32 = 14;

/// C's `SchemaType`.
pub const SCHEMA_NODE: u32 = 0;
pub const SCHEMA_EDGE: u32 = 1;

// ── SIValue type tags ──
//
// C's `SIType` is a **bitmask**, not an ordinal: each type is a distinct bit.
// Rust's own effects codec used sequential 0..12 tags, which collide with these
// almost everywhere and are the single most dangerous divergence in the format.

pub const T_MAP: u32 = 1 << 0;
pub const T_ARRAY: u32 = 1 << 3;
pub const T_DATETIME: u32 = 1 << 5;
pub const T_DATE: u32 = 1 << 7;
pub const T_TIME: u32 = 1 << 8;
pub const T_DURATION: u32 = 1 << 10;
pub const T_STRING: u32 = 1 << 11;
pub const T_BOOL: u32 = 1 << 12;
pub const T_INT64: u32 = 1 << 13;
pub const T_DOUBLE: u32 = 1 << 14;
pub const T_NULL: u32 = 1 << 15;
pub const T_POINT: u32 = 1 << 17;
pub const T_VECTOR_F32: u32 = 1 << 18;

// ── primitive writers ──

pub fn write_u8(
    buf: &mut Vec<u8>,
    v: u8,
) {
    buf.push(v);
}

pub fn write_u16(
    buf: &mut Vec<u8>,
    v: u16,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u32(
    buf: &mut Vec<u8>,
    v: u32,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u64(
    buf: &mut Vec<u8>,
    v: u64,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_i64(
    buf: &mut Vec<u8>,
    v: i64,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_f64(
    buf: &mut Vec<u8>,
    v: f64,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// C's `LabelID` and `RelationID` are `int`, so signed and 4 bytes.
pub fn write_label_id(
    buf: &mut Vec<u8>,
    v: i32,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// A C string: length **including** the NUL terminator, then the bytes and the
/// terminator. C reads these straight into a `char[]` and treats them as C
/// strings, so omitting the NUL makes it read the name plus whatever follows.
pub fn write_string(
    buf: &mut Vec<u8>,
    s: &str,
) {
    debug_assert!(
        !s.as_bytes().contains(&0),
        "an interior NUL would truncate this string on the C side"
    );
    write_u64(buf, s.len() as u64 + 1);
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);
}

/// A `SIValue`: 4-byte type bitmask, then the payload.
///
/// `T_NULL` has no payload at all — not a zero byte.
pub fn write_value(
    buf: &mut Vec<u8>,
    value: &Value,
) {
    match value {
        Value::Null => write_u32(buf, T_NULL),
        Value::Bool(b) => {
            write_u32(buf, T_BOOL);
            write_u8(buf, u8::from(*b));
        }
        Value::Int(i) => {
            write_u32(buf, T_INT64);
            write_i64(buf, *i);
        }
        Value::Float(f) => {
            write_u32(buf, T_DOUBLE);
            write_f64(buf, *f);
        }
        Value::String(s) => {
            write_u32(buf, T_STRING);
            write_string(buf, s);
        }
        Value::List(items) => {
            write_u32(buf, T_ARRAY);
            // u32, not u64: C reads the count as `uint32`.
            write_u32(buf, items.len() as u32);
            for item in items.iter() {
                write_value(buf, item);
            }
        }
        Value::Map(m) => {
            write_u32(buf, T_MAP);
            write_u32(buf, m.len() as u32);
            for (k, v) in m.iter() {
                // Keys are strings on both sides; C writes the key as a bare
                // string, not as a nested SIValue.
                write_string(buf, k);
                write_value(buf, v);
            }
        }
        Value::Point(p) => {
            write_u32(buf, T_POINT);
            // 2 x f32. Rust's own format used f64 here, which silently doubles
            // the payload and desyncs everything after it.
            buf.extend_from_slice(&p.latitude.to_le_bytes());
            buf.extend_from_slice(&p.longitude.to_le_bytes());
        }
        Value::VecF32(v) => {
            write_u32(buf, T_VECTOR_F32);
            write_u32(buf, v.len() as u32);
            for f in v.iter() {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        Value::Datetime(ts) => {
            write_u32(buf, T_DATETIME);
            write_i64(buf, *ts);
        }
        Value::Date(ts) => {
            write_u32(buf, T_DATE);
            write_i64(buf, *ts);
        }
        Value::Time(ts) => {
            write_u32(buf, T_TIME);
            write_i64(buf, *ts);
        }
        Value::Duration(d) => {
            write_u32(buf, T_DURATION);
            write_i64(buf, *d);
        }
        // Nodes, edges and paths are never property values, so they cannot reach
        // an effect. Encoding one as NULL would corrupt the stream silently.
        other => panic!("value cannot appear in an effect: {other:?}"),
    }
}

/// `ushort count`, then `count` x (`AttributeID`, `SIValue`).
pub fn write_attribute_set(
    buf: &mut Vec<u8>,
    attrs: &[(u16, Value)],
) {
    write_u16(buf, attrs.len() as u16);
    for (attr_id, value) in attrs {
        write_u16(buf, *attr_id);
        write_value(buf, value);
    }
}

// ── records ──

/// Start a fresh buffer, stamped with the version header.
#[must_use]
pub fn new_buffer() -> Vec<u8> {
    vec![EFFECTS_VERSION]
}

/// `1 UPDATE_NODE` / `2 UPDATE_EDGE` — **one record per (entity, attribute)**.
pub fn write_update(
    buf: &mut Vec<u8>,
    is_node: bool,
    entity_id: u64,
    attr_id: u16,
    value: &Value,
) {
    write_u32(
        buf,
        if is_node {
            EFFECT_UPDATE_NODE
        } else {
            EFFECT_UPDATE_EDGE
        },
    );
    write_u64(buf, entity_id);
    write_u16(buf, attr_id);
    write_value(buf, value);
}

/// `3 CREATE_NODE` — **v3 adds the explicit id**.
pub fn write_create_node(
    buf: &mut Vec<u8>,
    node_id: u64,
    labels: &[i32],
    attrs: &[(u16, Value)],
) {
    write_u32(buf, EFFECT_CREATE_NODE);
    write_u64(buf, node_id);
    write_u16(buf, labels.len() as u16);
    for label in labels {
        write_label_id(buf, *label);
    }
    write_attribute_set(buf, attrs);
}

/// `4 CREATE_EDGE` — **v3 adds the explicit id**.
pub fn write_create_edge(
    buf: &mut Vec<u8>,
    edge_id: u64,
    relation_id: i32,
    src: u64,
    dest: u64,
    attrs: &[(u16, Value)],
) {
    write_u32(buf, EFFECT_CREATE_EDGE);
    write_u64(buf, edge_id);
    // C carries a rel_count here; a single edge per record is count 1.
    write_u16(buf, 1);
    write_label_id(buf, relation_id);
    write_u64(buf, src);
    write_u64(buf, dest);
    write_attribute_set(buf, attrs);
}

/// `5 DELETE_NODE`.
pub fn write_delete_node(
    buf: &mut Vec<u8>,
    node_id: u64,
) {
    write_u32(buf, EFFECT_DELETE_NODE);
    write_u64(buf, node_id);
}

/// `6 DELETE_EDGE`.
pub fn write_delete_edge(
    buf: &mut Vec<u8>,
    edge_id: u64,
    relation_id: i32,
    src: u64,
    dest: u64,
) {
    write_u32(buf, EFFECT_DELETE_EDGE);
    write_u64(buf, edge_id);
    write_label_id(buf, relation_id);
    write_u64(buf, src);
    write_u64(buf, dest);
}

/// `7 SET_LABELS` / `8 REMOVE_LABELS` — **v3 replaces C's `GxB_Vector_serialize`
/// blob with explicit pairs**, so the format carries no dependency on either
/// engine's GraphBLAS build.
pub fn write_labels(
    buf: &mut Vec<u8>,
    set: bool,
    pairs: &[(u64, i32)],
) {
    write_u32(
        buf,
        if set {
            EFFECT_SET_LABELS
        } else {
            EFFECT_REMOVE_LABELS
        },
    );
    write_u64(buf, pairs.len() as u64);
    for (node_id, label_id) in pairs {
        write_u64(buf, *node_id);
        write_label_id(buf, *label_id);
    }
}

/// `9 ADD_SCHEMA`.
pub fn write_add_schema(
    buf: &mut Vec<u8>,
    schema_type: u32,
    name: &str,
) {
    write_u32(buf, EFFECT_ADD_SCHEMA);
    write_u32(buf, schema_type);
    write_string(buf, name);
}

/// `10 ADD_ATTRIBUTE` — no node/relationship discriminator. Rust used to write
/// one because it had two attribute dictionaries; #2459 unified them.
pub fn write_add_attribute(
    buf: &mut Vec<u8>,
    name: &str,
) {
    write_u32(buf, EFFECT_ADD_ATTRIBUTE);
    write_string(buf, name);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::value::Point;
    use std::sync::Arc;

    fn s(v: &str) -> Arc<String> {
        Arc::new(v.to_string())
    }

    #[test]
    fn version_header_is_three() {
        assert_eq!(new_buffer(), vec![3u8]);
    }

    #[test]
    fn strings_carry_their_nul_and_count_it() {
        let mut b = Vec::new();
        write_string(&mut b, "ab");
        // len = 3 (two bytes plus terminator), then "ab\0"
        assert_eq!(b, vec![3, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', 0]);
    }

    #[test]
    fn empty_string_is_length_one() {
        let mut b = Vec::new();
        write_string(&mut b, "");
        assert_eq!(b, vec![1, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn null_has_a_tag_and_no_payload() {
        let mut b = Vec::new();
        write_value(&mut b, &Value::Null);
        // T_NULL = 1<<15 = 0x8000, and nothing after it.
        assert_eq!(b, vec![0x00, 0x80, 0x00, 0x00]);
    }

    #[test]
    fn value_tags_are_the_bitmask_not_an_ordinal() {
        // The divergence most likely to be reintroduced: sequential tags happen
        // to "work" against ourselves and desync against C.
        let cases: &[(Value, u32)] = &[
            (Value::Bool(true), T_BOOL),
            (Value::Int(1), T_INT64),
            (Value::Float(1.0), T_DOUBLE),
            (Value::Datetime(0), T_DATETIME),
            (Value::Date(0), T_DATE),
            (Value::Time(0), T_TIME),
            (Value::Duration(0), T_DURATION),
        ];
        for (value, tag) in cases {
            let mut b = Vec::new();
            write_value(&mut b, value);
            assert_eq!(
                u32::from_le_bytes(b[0..4].try_into().unwrap()),
                *tag,
                "wrong tag for {value:?}"
            );
            assert!(*tag > 12, "a bitmask tag must not look like an ordinal");
        }
    }

    #[test]
    fn point_is_two_f32_not_two_f64() {
        let mut b = Vec::new();
        write_value(
            &mut b,
            &Value::Point(Point {
                latitude: 1.5,
                longitude: -2.5,
            }),
        );
        assert_eq!(b.len(), 4 + 8, "tag plus 2 x f32");
        assert_eq!(f32::from_le_bytes(b[4..8].try_into().unwrap()), 1.5);
        assert_eq!(f32::from_le_bytes(b[8..12].try_into().unwrap()), -2.5);
    }

    #[test]
    fn collection_counts_are_u32() {
        let mut b = Vec::new();
        write_value(&mut b, &Value::List(Arc::new(thin_vec::thin_vec![])));
        assert_eq!(b.len(), 4 + 4, "tag plus a u32 count, not a u64");

        let mut b = Vec::new();
        write_value(&mut b, &Value::VecF32(Arc::new(vec![1.0f32].into())));
        assert_eq!(b.len(), 4 + 4 + 4, "tag, u32 count, one f32");
    }

    #[test]
    fn create_node_carries_its_id() {
        let mut b = new_buffer();
        write_create_node(&mut b, 0x0102_0304_0506_0708, &[7], &[]);

        let mut want = vec![3u8]; // version
        want.extend_from_slice(&EFFECT_CREATE_NODE.to_le_bytes());
        want.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes()); // the v3 addition
        want.extend_from_slice(&1u16.to_le_bytes()); // label count
        want.extend_from_slice(&7i32.to_le_bytes()); // LabelID is a 4-byte int
        want.extend_from_slice(&0u16.to_le_bytes()); // empty attribute set
        assert_eq!(b, want);
    }

    #[test]
    fn labels_are_pairs_not_a_serialized_vector() {
        let mut b = Vec::new();
        write_labels(&mut b, true, &[(1, 10), (2, 11)]);

        let mut want = Vec::new();
        want.extend_from_slice(&EFFECT_SET_LABELS.to_le_bytes());
        want.extend_from_slice(&2u64.to_le_bytes());
        want.extend_from_slice(&1u64.to_le_bytes());
        want.extend_from_slice(&10i32.to_le_bytes());
        want.extend_from_slice(&2u64.to_le_bytes());
        want.extend_from_slice(&11i32.to_le_bytes());
        assert_eq!(b, want);
    }

    #[test]
    fn update_is_one_record_per_attribute() {
        // Two attributes on one node produce two records, each repeating the id.
        let mut b = Vec::new();
        write_update(&mut b, true, 5, 0, &Value::Int(1));
        let one = b.len();
        write_update(&mut b, true, 5, 1, &Value::Int(2));
        assert_eq!(b.len(), one * 2, "records are independent and equal-sized");
        assert_eq!(
            u32::from_le_bytes(b[one..one + 4].try_into().unwrap()),
            EFFECT_UPDATE_NODE
        );
    }

    #[test]
    fn delete_records_are_packed() {
        let mut b = Vec::new();
        write_delete_node(&mut b, 9);
        assert_eq!(
            b.len(),
            4 + 8,
            "no alignment padding: C packs these structs"
        );

        let mut b = Vec::new();
        write_delete_edge(&mut b, 9, 1, 2, 3);
        assert_eq!(b.len(), 4 + 8 + 4 + 8 + 8);
    }

    #[test]
    fn add_attribute_has_no_entity_discriminator() {
        let mut b = Vec::new();
        write_add_attribute(&mut b, "a");
        // type, then straight to the string — 4 + (8 + 1 + 1)
        assert_eq!(b.len(), 4 + 8 + 2);
        assert_eq!(
            u32::from_le_bytes(b[0..4].try_into().unwrap()),
            EFFECT_ADD_ATTRIBUTE
        );
        assert_eq!(u64::from_le_bytes(b[4..12].try_into().unwrap()), 2);
    }

    #[test]
    fn map_keys_are_bare_strings() {
        // Index OPTIONS ride as a T_MAP; C writes each key as a string, not as a
        // nested SIValue, so a key must not carry a T_STRING tag.
        let mut m = crate::runtime::ordermap::OrderMap::with_capacity(1);
        m.insert(s("language"), Value::String(s("english")));
        let mut b = Vec::new();
        write_value(&mut b, &Value::Map(Arc::new(m)));

        assert_eq!(u32::from_le_bytes(b[0..4].try_into().unwrap()), T_MAP);
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 1);
        // Immediately a length prefix, not another 4-byte tag.
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 9);
    }
}
