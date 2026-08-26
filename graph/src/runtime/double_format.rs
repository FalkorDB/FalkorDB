//! `%.*g` double formatting, byte-compatible with C's `printf`.
//!
//! FalkorDB renders doubles the way the C implementation does — `%.15g` for
//! query results, `%.5g` for slowlog latencies — so client output stays
//! identical across engines. Going through libc `snprintf` for every value is
//! expensive: glibc's `printf_fp` runs a multi-precision conversion, which
//! measured ~1,800 instructions per value and dominated the cost of a query
//! returning a float column.
//!
//! This module keeps that exact output while skipping `snprintf` for the
//! values that actually occur in graph data. See [`format_g_fast`] for why the
//! shortest round-trip digits are also the `%.*g` digits, and where that stops
//! being true.

use std::os::raw::c_char;

/// Scratch buffer size for [`format_g_into`].
///
/// The longest `%.*g` rendering a finite `f64` can produce at the precisions
/// used here is well under this: 17 significant digits plus sign, point,
/// `e`, exponent sign and three exponent digits.
pub const G_BUF_LEN: usize = 64;

/// Parse the exponent ryu appended after `e`, e.g. `-9` in `1.5e-9`.
fn parse_exponent(s: &[u8]) -> Option<i32> {
    let (neg, digits) = match s.split_first()? {
        (b'-', rest) => (true, rest),
        (b'+', rest) => (false, rest),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: i32 = 0;
    for &c in digits {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(i32::from(c - b'0'))?;
    }
    Some(if neg { -v } else { v })
}

/// Allocation-free `%.*g` for the common case, via ryu's shortest round-trip
/// digits. Returns the number of bytes written, or `None` when the caller must
/// fall back to snprintf.
///
/// # Why the shortest digits are also the `%.*g` digits
///
/// `%.*g` rounds the *exact* value of `d` to `precision` significant digits and
/// strips trailing zeros. Ryu yields the shortest decimal `D` that round-trips
/// to `d`. When `D` has no more than `precision` digits, `D` (zero-padded to
/// `precision`) is exactly what that rounding produces, because `D` is then the
/// *nearest* `precision`-digit decimal to `d`:
///
/// - round-tripping bounds the gap by half a binary ulp, so
///   `|d - D| / |d| <= 2^-53 ~= 1.11e-16`;
/// - half a step of the `precision`-digit decimal grid is
///   `> 0.5 * 10^-precision` relative to `|d|`, i.e. `>= 5e-16` at
///   `precision = 15`.
///
/// The first is smaller than the second, so no other `precision`-digit decimal
/// is closer. The margin disappears at `precision >= 16`, and the binary-ulp
/// bound does not hold for subnormals, so both are excluded and take the
/// snprintf path.
fn format_g_fast(
    d: f64,
    precision: i32,
    out: &mut [u8; G_BUF_LEN],
) -> Option<usize> {
    // `%g` treats precision 0 as 1. Above 15 the equivalence argument above no
    // longer holds; subnormals (and non-finite values) break it too.
    let p = precision.max(1) as usize;
    if p > 15 || !(d.is_normal() || d == 0.0) {
        return None;
    }

    let mut ryu_buf = ryu::Buffer::new();
    let s = ryu_buf.format_finite(d).as_bytes();
    let (neg, s) = match s.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, s),
    };

    // Split ryu's output into mantissa digits and a power-of-ten exponent.
    let (mantissa, mut exp) = match s.iter().position(|&c| c == b'e' || c == b'E') {
        Some(i) => (&s[..i], parse_exponent(&s[i + 1..])?),
        None => (s, 0),
    };

    let mut digits = [0u8; 24];
    let mut n = 0usize;
    let mut int_len = mantissa.len();
    for (i, &c) in mantissa.iter().enumerate() {
        if c == b'.' {
            int_len = i;
            continue;
        }
        if !c.is_ascii_digit() || n == digits.len() {
            return None;
        }
        digits[n] = c;
        n += 1;
    }
    if n == 0 {
        return None;
    }

    // Exponent of the leading digit, i.e. `x` in `d.ddd * 10^x`.
    exp += i32::try_from(int_len).ok()? - 1;

    let leading_zeros = digits[..n].iter().take_while(|&&c| c == b'0').count();
    if leading_zeros == n {
        // The value is zero; `%g` renders it as a bare "0".
        digits[0] = b'0';
        n = 1;
        exp = 0;
    } else {
        digits.copy_within(leading_zeros..n, 0);
        n -= leading_zeros;
        exp -= i32::try_from(leading_zeros).ok()?;
    }
    while n > 1 && digits[n - 1] == b'0' {
        n -= 1;
    }

    if n > p {
        return None;
    }

    let mut w = 0usize;
    if neg {
        out[w] = b'-';
        w += 1;
    }

    if exp < -4 || exp >= i32::try_from(p).ok()? {
        // Scientific style: one digit, optional fraction, `e`, signed exponent
        // padded to at least two digits.
        out[w] = digits[0];
        w += 1;
        if n > 1 {
            out[w] = b'.';
            w += 1;
            out[w..w + n - 1].copy_from_slice(&digits[1..n]);
            w += n - 1;
        }
        out[w] = b'e';
        w += 1;
        out[w] = if exp < 0 { b'-' } else { b'+' };
        w += 1;
        let mag = exp.unsigned_abs();
        if mag > 999 {
            return None;
        }
        if mag >= 100 {
            out[w] = b'0' + u8::try_from(mag / 100).ok()?;
            w += 1;
        }
        out[w] = b'0' + u8::try_from((mag / 10) % 10).ok()?;
        w += 1;
        out[w] = b'0' + u8::try_from(mag % 10).ok()?;
        w += 1;
    } else if exp >= 0 {
        let int_digits = exp as usize + 1;
        if n <= int_digits {
            out[w..w + n].copy_from_slice(&digits[..n]);
            w += n;
            for _ in n..int_digits {
                out[w] = b'0';
                w += 1;
            }
        } else {
            out[w..w + int_digits].copy_from_slice(&digits[..int_digits]);
            w += int_digits;
            out[w] = b'.';
            w += 1;
            out[w..w + n - int_digits].copy_from_slice(&digits[int_digits..n]);
            w += n - int_digits;
        }
    } else {
        out[w] = b'0';
        w += 1;
        out[w] = b'.';
        w += 1;
        for _ in 0..(-exp - 1) {
            out[w] = b'0';
            w += 1;
        }
        out[w..w + n].copy_from_slice(&digits[..n]);
        w += n;
    }

    Some(w)
}

/// `%.*g` via libc, for the cases [`format_g_fast`] declines.
fn format_g_snprintf(
    d: f64,
    precision: i32,
    out: &mut [u8; G_BUF_LEN],
) -> usize {
    let fmt = c"%.*g";
    // SAFETY: `out` is a live, uniquely borrowed array and we pass its true
    // length as the size bound, so snprintf writes at most `G_BUF_LEN` bytes
    // (always NUL-terminating) and cannot run past the end. The format string
    // is a literal whose `%.*g` conversions match the `c_int`/`c_double`
    // varargs supplied, so the call is well-typed.
    let n = unsafe {
        libc::snprintf(
            out.as_mut_ptr().cast::<c_char>(),
            out.len(),
            fmt.as_ptr(),
            precision,
            d,
        )
    };
    // snprintf returns the length it *would* have written; clamp to what fits.
    (n.max(0) as usize).min(out.len() - 1)
}

/// Format a double using C's `%.*g` (`precision` significant digits, shortest
/// of %e/%f with trailing zeros stripped), byte-for-byte identical to
/// snprintf for parity with FalkorDB C output.
///
/// Writes into `buf` and borrows from it, so formatting a result column costs
/// no allocation.
pub fn format_g_into(
    d: f64,
    precision: i32,
    buf: &mut [u8; G_BUF_LEN],
) -> &str {
    let n =
        format_g_fast(d, precision, buf).unwrap_or_else(|| format_g_snprintf(d, precision, buf));
    std::str::from_utf8(&buf[..n]).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::{G_BUF_LEN, format_g_into, format_g_snprintf};

    /// The contract: byte-for-byte equality with libc's `%.*g`.
    fn assert_matches_libc(
        d: f64,
        precision: i32,
    ) {
        let mut fast = [0u8; G_BUF_LEN];
        let mut libc_buf = [0u8; G_BUF_LEN];
        let got = format_g_into(d, precision, &mut fast);
        let n = format_g_snprintf(d, precision, &mut libc_buf);
        let want = std::str::from_utf8(&libc_buf[..n]).unwrap();
        assert_eq!(
            got,
            want,
            "%.{precision}g of {d:?} (bits {:#x})",
            d.to_bits()
        );
    }

    #[test]
    fn matches_libc_on_edge_cases() {
        let cases = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1.5,
            -1.5,
            0.1,
            100.0,
            1e14,
            1e15,
            1e16,
            1e-4,
            1e-5,
            1.0 / 3.0,
            2.0 / 3.0,
            1e100,
            1e-100,
            1e308,
            5e-324,
            2.2250738585072014e-308,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            123_456_789.123_456_79,
            9_007_199_254_740_993.0,
            32.0,
            34.8,
            -0.000_123_45,
        ];
        for d in cases {
            for p in [1, 2, 5, 6, 14, 15, 16, 17] {
                assert_matches_libc(d, p);
            }
        }
    }

    /// Values shaped like real graph properties: integers, halves, and
    /// short decimals — the population the fast path exists for.
    #[test]
    fn matches_libc_on_property_like_values() {
        for i in -20_000i64..20_000 {
            let f = i as f64;
            assert_matches_libc(f, 15);
            assert_matches_libc(f * 1.5, 15);
            assert_matches_libc(f / 100.0, 15);
            assert_matches_libc(f / 3.0, 15);
            assert_matches_libc(f * 1e-7, 15);
        }
    }

    /// Random bit patterns, including subnormals and huge/tiny exponents,
    /// so the snprintf fallback boundary is exercised too.
    #[test]
    fn matches_libc_on_random_bit_patterns() {
        let mut s: u64 = 0x243F_6A88_85A3_08D3;
        for _ in 0..300_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let d = f64::from_bits(s);
            if d.is_nan() {
                continue;
            }
            assert_matches_libc(d, 15);
            assert_matches_libc(d, 5);
        }
    }

    /// Random values in ranges a query is actually likely to return.
    #[test]
    fn matches_libc_on_random_realistic_magnitudes() {
        let mut s: u64 = 0x1357_9BDF_2468_ACE0;
        for _ in 0..300_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let unit = (s >> 11) as f64 / (1u64 << 53) as f64;
            for scale in [1e-9, 1e-3, 1.0, 1e3, 1e9, 1e18] {
                let d = unit * scale;
                assert_matches_libc(d, 15);
                assert_matches_libc(-d, 15);
                assert_matches_libc(d, 5);
            }
        }
    }
}
