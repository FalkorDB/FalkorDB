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

/// How deep a list or map may nest.
///
/// **This bounds the type, not the decoder.** `Value` holds its children behind
/// `Arc<ThinVec<Value>>`, so every operation on a nested one recurses — `Drop`,
/// `PartialEq`, `Debug`, `encode`. Making the decoder iterative removed its own
/// recursion and nothing else's: measured, 50,000 levels now decode without
/// touching the stack and then abort in `Drop`, because dropping the chain
/// recurses once per level.
///
/// So a depth bound is not a decoder shortcut standing in for trusting the
/// encoder. It is the only thing between a nested value and a SIGSEGV in code
/// that never looks at the wire at all.
///
/// 256 mirrors `Parser::MAX_TREE_DEPTH`. A value only reaches an effect by being
/// parsed and evaluated first, so nothing legitimate can arrive nested deeper
/// than the parser would accept, and that limit already sits well above the 100
/// nested lists the parser's own tests pin.
const MAX_VALUE_DEPTH: usize = 256;

/// One unfinished container, while its children are being read.
///
/// A map keeps the key it is waiting on: an entry is a key then a value, and the
/// key is read as soon as the previous entry closes, so the loop below always
/// has exactly one value to produce next.
enum Frame {
    List {
        items: ThinVec<Value>,
        left: usize,
    },
    Map {
        map: OrderMap<Arc<String>, Value>,
        left: usize,
        key: Arc<String>,
    },
}

impl EffectDecode<3> for Value {
    /// Iterative, not recursive, and that is a requirement rather than a style.
    ///
    /// `Reader::guard_count` bounds how *wide* a container is, because every
    /// element costs bytes. Depth does not work that way: one nesting level is a
    /// 4-byte tag and a 4-byte count, so a few hundred KB of nested tags was
    /// tens of thousands of stack frames and a SIGSEGV — exactly what `Reader`
    /// exists to prevent. The previous answer was a depth ceiling, which cannot
    /// be right for the same reason a ceiling was wrong for ids: it refuses
    /// legitimate input to bound a hostile one.
    ///
    /// With an explicit stack, depth costs heap instead — one `Frame` per open
    /// container. That removes the decoder's own recursion, which is worth
    /// having, but it does **not** remove the need for [`MAX_VALUE_DEPTH`]:
    /// `Value` is recursive, so dropping a deeply nested one overflows the stack
    /// whatever built it. The bound is enforced here because this is where the
    /// depth becomes known, not because decoding is the fragile part.
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let mut stack: Vec<Frame> = Vec::new();

        loop {
            // Produce one value. A container opens a frame and yields nothing
            // yet, so the next turn reads its first child.
            let Some(mut value) = read_one(r, &mut stack)? else {
                continue;
            };

            // Hand it to whatever is waiting, closing frames as they fill. A
            // closed container is itself a value, so this repeats.
            loop {
                match stack.last_mut() {
                    None => return Ok(value),
                    Some(Frame::List { items, left }) => {
                        items.push(value);
                        *left -= 1;
                        if *left > 0 {
                            break;
                        }
                        let Some(Frame::List { items, .. }) = stack.pop() else {
                            unreachable!("just matched a list frame")
                        };
                        value = Value::List(Arc::new(items));
                    }
                    Some(Frame::Map { map, left, key }) => {
                        map.insert(Arc::clone(key), value);
                        *left -= 1;
                        if *left > 0 {
                            // The next entry's key, read now so the next turn
                            // produces its value.
                            *key = Arc::new(r.string()?);
                            break;
                        }
                        let Some(Frame::Map { map, .. }) = stack.pop() else {
                            unreachable!("just matched a map frame")
                        };
                        value = Value::Map(Arc::new(map));
                    }
                }
            }
        }
    }
}

/// Refuse another level before one is opened.
fn check_depth(stack: &[Frame]) -> Result<(), DecodeError> {
    if stack.len() >= MAX_VALUE_DEPTH {
        return Err(DecodeError::ValueTooDeep {
            max: MAX_VALUE_DEPTH,
        });
    }
    Ok(())
}

/// Read one tag and either return a finished value or open a container.
///
/// `Ok(None)` means a frame was pushed: the value is not known until its
/// children are.
fn read_one(
    r: &mut Reader<'_>,
    stack: &mut Vec<Frame>,
) -> Result<Option<Value>, DecodeError> {
    // Widened so the arms can name `si_type`'s own constants: they are `u64`
    // there, and a pattern has to be a named constant, not a narrowing
    // expression.
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
            if n == 0 {
                return Ok(Some(Value::List(Arc::new(ThinVec::new()))));
            }
            check_depth(stack)?;
            stack.push(Frame::List {
                items: ThinVec::with_capacity(n),
                left: n,
            });
            return Ok(None);
        }
        T_MAP => {
            let n = r.u32()?;
            // Each pair is at least an 8-byte length plus a 4-byte type tag.
            let n = r.guard_count(u64::from(n), 12)?;
            if n == 0 {
                return Ok(Some(Value::Map(Arc::new(OrderMap::default()))));
            }
            check_depth(stack)?;
            let key = Arc::new(r.string()?);
            stack.push(Frame::Map {
                map: OrderMap::default(),
                left: n,
                key,
            });
            return Ok(None);
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
    Ok(Some(v))
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
    fn deep_nesting_is_refused_because_the_value_cannot_be_dropped() {
        // The bound is not about decoding. With an iterative decoder, a payload
        // 50,000 levels deep decodes without touching the stack — and then
        // aborts in `Drop`, because `Value` holds its children behind an `Arc`
        // and dropping the chain recurses once per level. `std::mem::forget` on
        // the decoded value is what proves it: forget it and nothing crashes.
        //
        // So the refusal happens where the depth first becomes known, and it
        // protects code that never looks at the wire.
        let mut buf = Vec::new();
        for _ in 0..MAX_VALUE_DEPTH + 50 {
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

        // And right up to the limit still decodes, so the bound is not quietly
        // rejecting legitimate data.
        let mut buf = Vec::new();
        for _ in 0..MAX_VALUE_DEPTH {
            buf.extend_from_slice(&(si_type::T_ARRAY as u32).to_le_bytes());
            buf.extend_from_slice(&1_u32.to_le_bytes());
        }
        buf.extend_from_slice(&(si_type::T_NULL as u32).to_le_bytes());
        let mut r = Reader::new(&buf);
        let v = Value::decode(&mut r).expect("at the limit must still decode");

        // Walked iteratively, and dropped normally at the end of scope — which
        // is the operation the bound exists for.
        let mut depth = 0_usize;
        let mut cur = &v;
        while let Value::List(items) = cur {
            assert_eq!(items.len(), 1);
            cur = &items[0];
            depth += 1;
        }
        assert_eq!(depth, MAX_VALUE_DEPTH);
        assert_eq!(*cur, Value::Null);
    }

    #[test]
    fn a_nested_container_that_runs_out_of_bytes_is_an_error() {
        // Depth is no longer bounded, so the guard that has to hold is the
        // ordinary one: a payload promising children it does not carry fails on
        // the read rather than producing a partial value.
        let mut buf = Vec::new();
        for _ in 0..MAX_VALUE_DEPTH - 1 {
            buf.extend_from_slice(&(si_type::T_ARRAY as u32).to_le_bytes());
            buf.extend_from_slice(&1_u32.to_le_bytes());
        }
        // ...and nothing at the bottom.
        //
        // `ImplausibleCount` rather than `UnexpectedEof`: the innermost level
        // promises one child, and the width guard notices there are not four
        // bytes left to hold even a tag before the read is attempted. Either way
        // it is an error and not a partial value, which is the property here.
        let mut r = Reader::new(&buf);
        assert!(matches!(
            Value::decode(&mut r),
            Err(DecodeError::ImplausibleCount { .. } | DecodeError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn a_map_nested_in_a_list_round_trips() {
        // The iterative decoder has to interleave two frame kinds, and a map
        // reads its key before its value — the case a recursive decoder got for
        // free.
        let mut m = OrderMap::default();
        m.insert(Arc::new("a".to_string()), Value::Int(1));
        m.insert(
            Arc::new("b".to_string()),
            Value::List(Arc::new(thin_vec::thin_vec![
                Value::Null,
                Value::Map(Arc::new(OrderMap::default())),
            ])),
        );
        let case = Value::List(Arc::new(thin_vec::thin_vec![
            Value::Map(Arc::new(m)),
            Value::List(Arc::new(ThinVec::new())),
        ]));

        let mut buf = Vec::new();
        case.encode(&mut buf);
        let mut r = Reader::new(&buf);
        assert_eq!(Value::decode(&mut r).unwrap(), case);
        assert!(r.is_empty());
    }

    #[test]
    fn an_unknown_value_tag_is_rejected() {
        let buf = 0x0000_0002_u32.to_le_bytes();
        let mut r = Reader::new(&buf);
        assert_eq!(Value::decode(&mut r), Err(DecodeError::BadValueType(2)));
    }
}
