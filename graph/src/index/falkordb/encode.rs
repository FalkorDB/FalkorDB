//! Order-preserving key encoding for the index (PR2 · P1).
//!
//! The numeric index stores `(key, doc)` tuples in a
//! [`CowBTree`](super::data_structures::cow_btree), sorted by `(key, doc)`:
//! `key` is the *encoded* numeric value, `doc` the entity id. For range scans to
//! be correct the encoding must be **monotone** — for any two non-NaN numeric
//! values `a`, `b`:
//!
//! ```text
//!     a < b   (as numbers)   <=>   encode(a) < encode(b)   (as u64)
//! ```
//!
//! Every value is coerced to `f64` first, matching the RediSearch NUMERIC field
//! this replaces, so a mixed Int/Float column shares a single order. That
//! coercion inherits RediSearch's precision limit for integer magnitudes
//! `>= 2^53`; it is documented here and a wider key is a future refinement.

use crate::runtime::value::Value;

/// IEEE-754 sign bit of an `f64`.
const SIGN: u64 = 0x8000_0000_0000_0000;

/// Encode a numeric [`Value`] into the monotone `u64` key half of a
/// `(key, doc)` index tuple.
///
/// Returns `None` for non-numeric values and for `NaN`, which has no position
/// in a sorted index.
#[must_use]
pub fn encode_numeric(value: &Value) -> Option<u64> {
    let x = match value {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        Value::Bool(b) => f64::from(*b),
        Value::Datetime(t) | Value::Date(t) | Value::Time(t) | Value::Duration(t) => *t as f64,
        _ => return None,
    };
    (!x.is_nan()).then(|| encode_f64(x))
}

/// Append the keys a **stored** value contributes, and say which tree they belong in.
///
/// A scalar contributes at most one key to the *scalar* tree. A list contributes one key per
/// numeric element to the *array* tree — so `1 IN n.samples` is an ordinary point lookup.
///
/// The two key spaces must stay separate. If a list's elements landed in the scalar tree, a node
/// with `v = [1, 2]` would carry the key for `1`, and `WHERE n.v = 1` — which is **false** in
/// Cypher — would match it. `Equal` retains no post-filter (the plan is a bare `Node By Index
/// Scan`), so that false positive would be returned to the user. RediSearch avoids the same
/// collision with a separate `numeric:arr` sub-field; this is the native equivalent.
///
/// Keys are deduplicated, so `[1, 1.0, 2]` contributes two tuples rather than three: `(key, doc)`
/// is a set member in the tree, and a removal must not depend on how many times the value
/// happened to list the same number. Nested lists are not flattened — Cypher's `IN` does not look
/// inside them.
///
/// Deliberately separate from [`encode_numeric`], which answers a different question: the key of
/// a **query** value. A query value is a single scalar; one that is a list is declined rather
/// than expanded, because `n.v = [1,2]` asks about the list itself.
#[must_use]
pub fn encode_stored(
    value: &Value,
    out: &mut Vec<u64>,
) -> StoredKeys {
    out.clear();
    match value {
        Value::List(items) => {
            out.extend(items.iter().filter_map(encode_numeric));
            out.sort_unstable();
            out.dedup();
            StoredKeys::Array
        }
        _ => {
            out.extend(encode_numeric(value));
            StoredKeys::Scalar
        }
    }
}

/// Which tree [`encode_stored`] filled `out` for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoredKeys {
    /// The value was a scalar: at most one key, for the scalar tree.
    Scalar,
    /// The value was a list: one key per numeric element, for the array tree. `out` may be empty
    /// (a list with no numeric elements) — that is still an array-kind value, not a scalar.
    Array,
}

/// Monotone total order over non-`NaN` `f64`, as a `u64`: for non-`NaN` `a`, `b`,
/// `a < b`  ⇔  `encode_f64(a) < encode_f64(b)`.
///
/// Non-negatives get the sign bit set (sorting them above every negative);
/// negatives are bit-inverted so more-negative sorts lower. `-0.0` collapses
/// into `+0.0` so the two numerically-equal zeros share a key.
#[must_use]
pub fn encode_f64(x: f64) -> u64 {
    // Collapse -0.0 into +0.0 so `n.x = 0` matches both signs of zero.
    let x = if x == 0.0 { 0.0 } else { x };
    let bits = x.to_bits();
    if bits & SIGN == 0 { bits | SIGN } else { !bits }
}

/// Inverse of [`encode_f64`], for tests and debugging. Note `-0.0` does not
/// round-trip: it was normalized to `+0.0` on encode.
#[cfg(test)]
#[must_use]
fn decode_f64(key: u64) -> f64 {
    let bits = if key & SIGN != 0 { key & !SIGN } else { !key };
    f64::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strictly increasing keys across a representative sweep of the f64 domain
    /// (a single zero — the two-zero collision is checked separately).
    #[test]
    fn strictly_monotone_across_the_domain() {
        let vals = [
            f64::NEG_INFINITY,
            -1e308,
            -1.0,
            -f64::MIN_POSITIVE,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            1e308,
            f64::INFINITY,
        ];
        let mut prev = encode_f64(vals[0]);
        for &v in &vals[1..] {
            let k = encode_f64(v);
            assert!(
                k > prev,
                "encode not strictly monotone at {v}: {k} <= {prev}"
            );
            prev = k;
        }
    }

    /// Both zeros — and the integer/bool zeros — map to one key.
    #[test]
    fn zeros_collide() {
        let z = encode_f64(0.0);
        assert_eq!(encode_f64(-0.0), z);
        assert_eq!(encode_numeric(&Value::Int(0)), Some(z));
        assert_eq!(encode_numeric(&Value::Float(-0.0)), Some(z));
        assert_eq!(encode_numeric(&Value::Bool(false)), Some(z));
    }

    /// Int and Float values interleave in a single order.
    #[test]
    fn int_float_interleave() {
        let three = encode_numeric(&Value::Int(3)).unwrap();
        let three_five = encode_numeric(&Value::Float(3.5)).unwrap();
        let four = encode_numeric(&Value::Int(4)).unwrap();
        assert!(three < three_five && three_five < four);

        // Negatives order by numeric value, not magnitude.
        let neg5 = encode_numeric(&Value::Int(-5)).unwrap();
        let neg4 = encode_numeric(&Value::Int(-4)).unwrap();
        assert!(neg5 < neg4 && neg4 < three);
    }

    /// Bool and temporal values coerce like their f64 equivalents (parity).
    #[test]
    fn bool_and_temporal_parity() {
        assert_eq!(encode_numeric(&Value::Bool(true)), Some(encode_f64(1.0)));
        assert_eq!(
            encode_numeric(&Value::Datetime(1000)),
            Some(encode_f64(1000.0))
        );
        assert_eq!(encode_numeric(&Value::Duration(-7)), Some(encode_f64(-7.0)));
    }

    /// NaN and non-numeric values are not indexable.
    #[test]
    fn nan_and_non_numeric_reject() {
        assert_eq!(encode_numeric(&Value::Float(f64::NAN)), None);
        assert_eq!(
            encode_numeric(&Value::String(std::sync::Arc::new("x".to_string()))),
            None
        );
    }

    /// `decode_f64` inverts `encode_f64` across the sweep.
    #[test]
    fn roundtrip() {
        for &v in &[
            f64::NEG_INFINITY,
            -1e308,
            -42.5,
            -1.0,
            0.0,
            1.0,
            42.5,
            1e308,
            f64::INFINITY,
        ] {
            assert_eq!(decode_f64(encode_f64(v)), v, "roundtrip failed for {v}");
        }
    }
}
