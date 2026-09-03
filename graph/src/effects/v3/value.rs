//! The `SIValue` codec: a 4-byte type bitmask, then the payload.

use std::sync::Arc;

use thin_vec::ThinVec;

use crate::runtime::{
    ordermap::OrderMap,
    value::{Point, Value},
};

use crate::graph::graphblas::serialization::si_type;

use super::{
    DecodeError, EffectDecode, EffectEncode, Reader, T_MAP, write_f64, write_i64, write_string,
    write_tag, write_u8, write_u32,
};

// ── SIValue ──

/// The effects encoding of a `SIValue`.
///
/// `Value` already carries `Encode<19>` for the RDB path; this is the same type
/// on a different wire, and the two disagree in every way that matters. The RDB
/// stream is self-describing: `write_unsigned` emits a `TYPE_UNSIGNED` byte and
/// a fixed 8-byte LE value, so a tag costs 9 bytes there and 4 here, and every
/// field behind it is framed differently too — bool as a tagged i64 vs one byte,
/// point as two f64 vs two f32, list counts tagged vs a bare `u32`. RDB also
/// cannot represent a map at all. Keeping them as separate trait impls is what
/// stops one being mistaken for the other.
///
/// `T_NULL` has no payload at all — not a zero byte.
impl EffectEncode<3> for Value {
    fn encode(
        &self,
        buf: &mut Vec<u8>,
    ) {
        match self {
            Value::Null => write_tag(buf, si_type::T_NULL),
            Value::Bool(b) => {
                write_tag(buf, si_type::T_BOOL);
                write_u8(buf, u8::from(*b));
            }
            Value::Int(i) => {
                write_tag(buf, si_type::T_INT64);
                write_i64(buf, *i);
            }
            Value::Float(f) => {
                write_tag(buf, si_type::T_DOUBLE);
                write_f64(buf, *f);
            }
            Value::String(s) => {
                // The intern bit rides along, as it does in the RDB encoding.
                // Without it a replica rebuilds every string as a fresh `Arc`
                // and its pool stays empty, so repeated strings cost it what
                // interning exists to avoid — `test_intern_string`'s
                // replication case measured an empty pool where the primary
                // held one entry.
                let tag = if crate::runtime::string_pool::global().is_interned(s) {
                    si_type::T_INTERN | si_type::T_STRING
                } else {
                    si_type::T_STRING
                };
                write_tag(buf, tag);
                write_string(buf, s);
            }
            Value::List(items) => {
                write_tag(buf, si_type::T_ARRAY);
                // u32, not u64: C reads the count as `uint32`.
                write_u32(buf, items.len() as u32);
                // Floor: every element is at least its own type tag.
                buf.reserve(items.len() * 4);
                for item in items.iter() {
                    item.encode(buf);
                }
            }
            Value::Map(m) => {
                write_tag(buf, T_MAP);
                write_u32(buf, m.len() as u32);
                // Floor: an 8-byte key length plus the value's type tag.
                buf.reserve(m.len() * 12);
                for (k, v) in m.iter() {
                    // Keys are strings on both sides; C writes the key as a bare
                    // string, not as a nested SIValue.
                    write_string(buf, k);
                    v.encode(buf);
                }
            }
            Value::Point(p) => {
                write_tag(buf, si_type::T_POINT);
                // 2 x f32. Rust's own format used f64 here, which silently doubles
                // the payload and desyncs everything after it.
                buf.extend_from_slice(&p.latitude.to_le_bytes());
                buf.extend_from_slice(&p.longitude.to_le_bytes());
            }
            Value::VecF32(v) => {
                write_tag(buf, si_type::T_VECTOR_F32);
                // Exact: count then a fixed 4 bytes per element.
                buf.reserve(4 + v.len() * 4);
                write_u32(buf, v.len() as u32);
                buf.extend(v.iter().flat_map(|f| f.to_le_bytes()));
            }
            Value::Datetime(ts) => {
                write_tag(buf, si_type::T_DATETIME);
                write_i64(buf, *ts);
            }
            Value::Date(ts) => {
                write_tag(buf, si_type::T_DATE);
                write_i64(buf, *ts);
            }
            Value::Time(ts) => {
                write_tag(buf, si_type::T_TIME);
                write_i64(buf, *ts);
            }
            Value::Duration(d) => {
                write_tag(buf, si_type::T_DURATION);
                write_i64(buf, *d);
            }
            // Nodes, edges and paths are never property values, so they cannot reach
            // an effect. Encoding one as NULL would corrupt the stream silently.
            other => panic!("value cannot appear in an effect: {other:?}"),
        }
    }
}

/// How deep a list or map may nest before the decoder refuses it.
///
/// `Reader::guard_count` bounds how *wide* a container is, because every element
/// costs bytes. Depth does not work that way: one `T_ARRAY` level is a 4-byte tag
/// and a 4-byte count, so ~350 KB of nested tags is ~44,000 recursive calls and
/// overflows the stack — a SIGSEGV, which is exactly what `Reader` exists to
/// prevent.
///
/// 256 mirrors `Parser::MAX_TREE_DEPTH`. A value only reaches an effect by being
/// parsed and evaluated first, so nothing legitimate can arrive nested deeper
/// than the parser would accept, and that limit already sits well above the 100
/// nested lists the parser's own tests pin.
const MAX_VALUE_DEPTH: usize = 256;

impl EffectDecode<3> for Value {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        decode_at(r, 0)
    }
}

fn decode_at(
    r: &mut Reader<'_>,
    depth: usize,
) -> Result<Value, DecodeError> {
    if depth > MAX_VALUE_DEPTH {
        return Err(DecodeError::ValueTooDeep {
            max: MAX_VALUE_DEPTH,
        });
    }
    // Widened so the arms can name `si_type`'s own constants: they are
    // `u64` there, and a pattern has to be a named constant, not a
    // narrowing expression.
    let t = u64::from(r.u32()?);
    let v = match t {
        si_type::T_NULL => Value::Null,
        si_type::T_BOOL => Value::Bool(r.u8()? != 0),
        si_type::T_INT64 => Value::Int(r.i64()?),
        si_type::T_DOUBLE => Value::Float(r.f64()?),
        // Both spellings, and the bit decides whether the string joins this
        // node's pool. A pattern cannot name `T_INTERN | T_STRING`, so this is a
        // guard — the same shape the RDB decoder uses.
        t if t == si_type::T_STRING || t == (si_type::T_INTERN | si_type::T_STRING) => {
            let s = r.string()?;
            if t == (si_type::T_INTERN | si_type::T_STRING) {
                Value::String(crate::runtime::string_pool::global().intern(Arc::new(s)))
            } else {
                Value::String(Arc::new(s))
            }
        }
        si_type::T_ARRAY => {
            let n = r.u32()?;
            // A list entry is at least a 4-byte type tag.
            let n = r.guard_count(u64::from(n), 4)?;
            let mut items = ThinVec::with_capacity(n);
            for _ in 0..n {
                items.push(decode_at(r, depth + 1)?);
            }
            Value::List(Arc::new(items))
        }
        T_MAP => {
            let n = r.u32()?;
            // Each pair is at least an 8-byte length plus a 4-byte type tag.
            let n = r.guard_count(u64::from(n), 12)?;
            let mut m = OrderMap::default();
            for _ in 0..n {
                let k = Arc::new(r.string()?);
                m.insert(k, decode_at(r, depth + 1)?);
            }
            Value::Map(Arc::new(m))
        }
        si_type::T_POINT => {
            let latitude = r.f32()?;
            let longitude = r.f32()?;
            Value::Point(Point {
                latitude,
                longitude,
            })
        }
        si_type::T_VECTOR_F32 => {
            let n = r.u32()?;
            let n = r.guard_count(u64::from(n), 4)?;
            let mut v = ThinVec::with_capacity(n);
            for _ in 0..n {
                v.push(r.f32()?);
            }
            Value::VecF32(Arc::new(v))
        }
        si_type::T_DATETIME => Value::Datetime(r.i64()?),
        si_type::T_DATE => Value::Date(r.i64()?),
        si_type::T_TIME => Value::Time(r.i64()?),
        si_type::T_DURATION => Value::Duration(r.i64()?),
        other => return Err(DecodeError::BadValueType(other as u32)),
    };
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::testing::hex;

    // ── SIValue ──

    #[test]
    fn value_tags_are_c_bitmasks_not_ordinals() {
        // The single most dangerous divergence: Rust's own codec used sequential
        // 0..12 tags, which collide with C's bitmask almost everywhere.
        let mut buf = Vec::new();
        Value::Int(1).encode(&mut buf);
        assert_eq!(&buf[..4], &[0x00, 0x20, 0x00, 0x00], "T_INT64 = 1 << 13");

        buf.clear();
        Value::Null.encode(&mut buf);
        assert_eq!(hex(&buf), "00 80 00 00", "T_NULL = 1 << 15, no payload");
    }

    #[test]
    fn value_point_is_two_f32() {
        let mut buf = Vec::new();
        Value::Point(Point {
            latitude: 1.0,
            longitude: 2.0,
        })
        .encode(&mut buf);
        assert_eq!(buf.len(), 4 + 8, "type tag plus 2 x f32, not 2 x f64");
    }

    #[test]
    fn value_string_carries_its_nul() {
        let mut buf = Vec::new();
        Value::String(Arc::new("ab".into())).encode(&mut buf);
        // type tag, then len = 3 (including NUL), then "ab\0"
        assert_eq!(hex(&buf), "00 08 00 00 03 00 00 00 00 00 00 00 61 62 00");
    }

    #[test]
    fn value_roundtrips_every_encodable_variant() {
        let cases = vec![
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(-9_007_199_254_740_993),
            Value::Float(0.1),
            Value::String(Arc::new("hello".into())),
            Value::List(Arc::new([Value::Int(1), Value::Null].into_iter().collect())),
            Value::Point(Point {
                latitude: 32.07,
                longitude: 34.79,
            }),
            Value::VecF32(Arc::new([1.5_f32, -2.5].into_iter().collect())),
            Value::Datetime(1_700_000_000),
            Value::Date(19_000),
            Value::Time(3_600),
            Value::Duration(90),
        ];
        for case in cases {
            let mut buf = Vec::new();
            case.encode(&mut buf);
            let mut r = Reader::new(&buf);
            assert_eq!(Value::decode(&mut r).unwrap(), case);
            assert!(r.is_empty(), "{case:?} left {} bytes", r.remaining());
        }
    }

    #[test]
    fn deep_nesting_is_an_error_not_a_stack_overflow() {
        // One `T_ARRAY` level is eight wire bytes, so a tiny payload buys tens of
        // thousands of stack frames. `guard_count` bounds width only, and the
        // whole point of `Reader` is that a malformed buffer is an error rather
        // than a segfault.
        let mut buf = Vec::new();
        let levels = MAX_VALUE_DEPTH + 50;
        for _ in 0..levels {
            buf.extend_from_slice(&(si_type::T_ARRAY as u32).to_le_bytes());
            buf.extend_from_slice(&1_u32.to_le_bytes());
        }
        buf.extend_from_slice(&(si_type::T_NULL as u32).to_le_bytes());

        let mut r = Reader::new(&buf);
        assert_eq!(
            Value::decode(&mut r),
            Err(DecodeError::ValueTooDeep {
                max: MAX_VALUE_DEPTH
            })
        );

        // And a value nested right up to the limit still decodes, so the bound
        // is not quietly rejecting legitimate data.
        let mut buf = Vec::new();
        for _ in 0..MAX_VALUE_DEPTH {
            buf.extend_from_slice(&(si_type::T_ARRAY as u32).to_le_bytes());
            buf.extend_from_slice(&1_u32.to_le_bytes());
        }
        buf.extend_from_slice(&(si_type::T_NULL as u32).to_le_bytes());
        let mut r = Reader::new(&buf);
        Value::decode(&mut r).expect("at the limit must still decode");
    }

    #[test]
    fn an_unknown_value_tag_is_rejected() {
        let buf = 0x0000_0002_u32.to_le_bytes();
        let mut r = Reader::new(&buf);
        assert_eq!(Value::decode(&mut r), Err(DecodeError::BadValueType(2)));
    }
}
