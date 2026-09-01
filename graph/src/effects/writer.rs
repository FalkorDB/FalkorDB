//! The fixed-width primitive writers.
//!
//! Beside [`super::reader`] rather than inside a version module for the same
//! reason the cursor is: little-endian primitives and a NUL-terminated C string
//! are not version-specific, and a v4 would want the same ones. v2 is not a
//! counter-example — it predates C compatibility and carries its own
//! `write_u16`/`write_string` in a different shape.
//!
//! Deliberately one function per width rather than one generic writer over
//! `to_le_bytes`. The bodies are identical, but the names are what make a call
//! site's width readable, and byte-for-byte agreement with C is the point of
//! this format: `write_u32(buf, x)` next to C's `sizeof(int)` is checkable,
//! `write_le(buf, x)` is not.

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
