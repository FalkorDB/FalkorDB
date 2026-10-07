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

    /// The encoded order must agree with the evaluator's own `compare_value`, because the index
    /// answers `Equal` / `Range` with **no post-filter** — a pair the encoder orders differently
    /// from the evaluator is a wrong row, not a slow one.
    ///
    /// This is the contract that makes the `f64` coercion sound rather than merely convenient.
    /// `compare_value` promotes for *mixed* numerics — `(Int(i), Float(f)) => compare_floats(i as
    /// f64, f)` — which is exactly what `encode_numeric` does, so the two agree by construction and
    /// a `Float` bound against `Int` data needs no precision guard.
    ///
    /// The one pair deliberately excluded is **Int vs Int past 2^53**: there `compare_value` is
    /// exact (`a.cmp(b)`) while the encoder rounds both through `f64`, so they genuinely disagree.
    /// That gap is closed upstream — `can_utilize_index` rejects such a query via
    /// `Index::int_loses_f64_precision` and routes it to a label scan. If this test is ever
    /// extended to cover that pair, the guard is what has to change with it.
    #[test]
    fn encoded_order_matches_value_comparison() {
        use crate::index::Index;
        use crate::runtime::value::CompareValue;
        use std::cmp::Ordering;

        const P: i64 = 1 << 53; // first integer f64 cannot represent exactly alongside its successor

        // Int and Float only. `Bool` is deliberately absent: the evaluator treats it as a type
        // disjoint from the numerics, while the encoder folds it into the same key space — see
        // `bool_shares_a_numeric_key_but_not_an_equality` for that divergence.
        let vals = [
            Value::Int(i64::MIN),
            Value::Int(-P - 1),
            Value::Int(-P),
            Value::Int(-5),
            Value::Int(0),
            Value::Float(-0.0),
            Value::Int(1),
            Value::Float(1.5),
            Value::Int(P),
            Value::Int(P + 1),
            Value::Float(P as f64),
            Value::Int(i64::MAX),
            Value::Float(f64::INFINITY),
            Value::Float(f64::NEG_INFINITY),
        ];

        // Int-vs-Int is the only pair the evaluator compares exactly, so it is the only pair the
        // encoder can disagree with — and only once a magnitude stops fitting in f64's mantissa.
        let exempt = |a: &Value, b: &Value| match (a, b) {
            (Value::Int(x), Value::Int(y)) => {
                Index::int_loses_f64_precision(*x) || Index::int_loses_f64_precision(*y)
            }
            _ => false,
        };

        let mut checked = 0;
        for a in &vals {
            for b in &vals {
                if exempt(a, b) {
                    continue;
                }
                let (ka, kb) = (
                    encode_numeric(a).expect("all sweep values are numeric"),
                    encode_numeric(b).expect("all sweep values are numeric"),
                );
                let encoded = ka.cmp(&kb);
                let evaluated = a.compare_value(b).0;
                assert_eq!(
                    encoded, evaluated,
                    "encoder and evaluator disagree on {a:?} vs {b:?}: \
                     encoded {encoded:?} (keys {ka} vs {kb}), evaluator {evaluated:?}"
                );
                checked += 1;
            }
        }
        // Guard against the sweep silently collapsing (an over-broad exemption, or a `vals` edit
        // that drops the interesting cases). Most of the n² pairs should survive the exemption.
        assert!(
            checked > vals.len() * vals.len() / 2,
            "sweep collapsed to {checked} pairs out of {}",
            vals.len() * vals.len()
        );

        // And the exempted pair really does disagree — otherwise the guard, and this exemption,
        // would be dead weight nobody would notice removing.
        // `2^53` is the last integer f64 holds exactly with its successor unrepresentable: `P + 1`
        // rounds back to `P`, while `P + 2` is exact again. So `P` and `P + 1` are the collision.
        let (big, bigger) = (Value::Int(P), Value::Int(P + 1));
        assert_eq!(big.compare_value(&bigger).0, Ordering::Less, "exact i64");
        assert_eq!(
            encode_numeric(&big),
            encode_numeric(&bigger),
            "f64 rounding should collapse these to one key"
        );
        assert!(
            Index::int_loses_f64_precision(P),
            "the upstream guard must reject the values this test exempts"
        );
    }

    /// **Inherited divergence, pinned deliberately — do not "fix" this here.**
    ///
    /// `encode_numeric` folds `Bool` into the numeric key space (`false`/`true` → `0.0`/`1.0`), so a
    /// bool and a number share a key. The evaluator disagrees: `compare_value` has no `(Int, Bool)`
    /// arm, so the pair falls to `self.order().cmp(&b.order())` with `DisjointOrNull::Disjoint` —
    /// different types never compare equal.
    ///
    /// That is **not** this index's invention. The RediSearch path does the same thing: bools go
    /// into the NUMERIC field as `f64::from(b)` on write (`index/mod.rs`, `DocumentAddFieldNumber`)
    /// and are queried as a numeric node with the same bounds. Measured on `MATCH (n:F) WHERE
    /// n.flag = 1` over one `flag: true` node and one `flag: 1` node, with an index on `F.flag`:
    ///
    /// | engine | unindexed | indexed |
    /// |---|---|---|
    /// | C (`:edge-c`) | `int_one` | both |
    /// | Rust + RediSearch | `int_one` | both |
    /// | Rust + native index | — | both |
    ///
    /// Every engine is correct without an index and wrong with one, because `Node By Index Scan`
    /// drops the `Filter` for `Equal`. So the defect is the missing post-filter, engine-wide, not
    /// the encoding — and making the native index alone disagree would mean the same query
    /// returning different rows depending on a build flag.
    ///
    /// Pinned so the behaviour is visible, and so a real fix has a failing assertion to flip.
    #[test]
    fn bool_shares_a_numeric_key_but_not_an_equality() {
        use crate::runtime::value::CompareValue;
        use std::cmp::Ordering;

        for (b, n) in [
            (Value::Bool(true), Value::Int(1)),
            (Value::Bool(false), Value::Int(0)),
            (Value::Bool(true), Value::Float(1.0)),
        ] {
            assert_eq!(
                encode_numeric(&b),
                encode_numeric(&n),
                "the index gives {b:?} and {n:?} one key"
            );
            assert_ne!(
                b.compare_value(&n).0,
                Ordering::Equal,
                "but the evaluator does not consider {b:?} equal to {n:?}"
            );
            assert!(b != n, "so `Value::eq` disagrees with the index for {b:?}");
        }
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
