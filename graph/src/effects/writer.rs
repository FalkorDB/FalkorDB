//! The sink an effects payload is written into, and its fixed-width writers.
//!
//! Beside [`super::reader`] rather than inside a version module for the same
//! reason the cursor is: little-endian primitives and a NUL-terminated C string
//! are not version-specific, and a v4 would want the same ones. v2 is not a
//! counter-example — it predates C compatibility and carries its own
//! `write_u16`/`write_string` in a different shape.
//!

/// The sink an effects payload is written into.
///
/// The same shape as the RDB side's `serialization::Writer`, and for one of the
/// same two reasons. There the sink is genuinely foreign — Redis's module IO,
/// which this crate cannot name — so `Encode` takes `&mut dyn Writer` and the
/// root crate implements it for `BufferedWriter`. Here there is one sink and it
/// belongs to this crate: a payload has to be complete and contiguous before
/// `seal` compresses it, so it is a `Vec<u8>` and nothing else can be.
///
/// What carries over is the *other* reason: a caller should write through the
/// sink rather than at it. `buf.u32(x)` says what the buffer does; `write_u32(buf, x)`
/// said what a free function does to it, and there is no way to give a second
/// sink the same vocabulary.
///
/// **Static dispatch, unlike the RDB trait** — but not for the reason it looks
/// like. A first draft of this comment claimed `&mut dyn` would cost per
/// primitive on the encode path; measured, it does not: 6M writes came out at
/// 57.32M instructions through a trait object against 57.35M through a
/// generic, which is 0.04% and almost certainly LLVM devirtualizing a call it
/// can see the concrete type of anyway.
///
/// The actual reason is that nothing needs `dyn`. There is one sink, no
/// `dyn EffectEncode` anywhere, and no object-safety requirement — so the
/// generic is free to keep and leaves the boundary in place for the day a
/// second sink appears.
///
/// Deliberately one method per width rather than one generic over
/// `to_le_bytes`. The bodies are identical, but the names are what make a call
/// site's width readable, and byte-for-byte agreement with C is the point of
/// this format: `buf.u32(x)` next to C's `sizeof(int)` is checkable,
/// `buf.le(x)` is not.
pub trait EffectWrite {
    fn u8(
        &mut self,
        v: u8,
    );
    fn u16(
        &mut self,
        v: u16,
    );
    fn u32(
        &mut self,
        v: u32,
    );
    fn u64(
        &mut self,
        v: u64,
    );
    fn i64(
        &mut self,
        v: i64,
    );
    fn f64(
        &mut self,
        v: f64,
    );

    /// C's `LabelID` and `RelationID` are `int`, so signed and 4 bytes.
    fn label_id(
        &mut self,
        v: i32,
    );

    /// A C string: length **including** the NUL terminator, then the bytes and
    /// the terminator. C reads these straight into a `char[]` and treats them
    /// as C strings, so omitting the NUL makes it read the name plus whatever
    /// follows.
    fn string(
        &mut self,
        s: &str,
    );
}

impl EffectWrite for Vec<u8> {
    fn u8(
        &mut self,
        v: u8,
    ) {
        self.push(v);
    }

    fn u16(
        &mut self,
        v: u16,
    ) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn u32(
        &mut self,
        v: u32,
    ) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(
        &mut self,
        v: u64,
    ) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn i64(
        &mut self,
        v: i64,
    ) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn f64(
        &mut self,
        v: f64,
    ) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn label_id(
        &mut self,
        v: i32,
    ) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn string(
        &mut self,
        s: &str,
    ) {
        debug_assert!(
            !s.as_bytes().contains(&0),
            "an interior NUL would truncate this string on the C side"
        );
        self.u64(s.len() as u64 + 1);
        self.extend_from_slice(s.as_bytes());
        self.push(0);
    }
}
