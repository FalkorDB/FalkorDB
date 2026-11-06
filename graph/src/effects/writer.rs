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
/// The same shape as the RDB side's `serialization::Writer`, and for the same
/// reason: a caller should write through the sink rather than at it.
/// `buf.u32(x)` says what the buffer does; `write_u32(buf, x)` said what a free
/// function does to it.
///
/// The bytes underneath are still one contiguous `Vec<u8>` — `seal` compresses
/// a whole payload and `records()` reads it back as a slice, so they have to
/// be. That is a fact about the representation, and it is why `Vec<u8>`
/// implements this. It is not a reason for the encoder to say `Vec<u8>`.
///
/// **Dynamic dispatch**, and it is free: 6M writes came out at 57.32M
/// instructions through a trait object against 57.35M through a generic, which
/// is 0.04% and almost certainly LLVM devirtualizing a call it can see the
/// concrete type of anyway. What `dyn` buys is that the encode path names the
/// *sink* rather than infecting fifteen signatures with a type parameter.
///
/// This trait had one implementor and every signature below it said
/// `&mut Vec<u8>` anyway, which made it vocabulary rather than a boundary —
/// `buf.u32(x)` reading better than `write_u32(buf, x)`, and nothing more.
/// The payload buffer in the layer above is the second implementor and the
/// reason it is a boundary now: it owns a payload, it is what the emitter is
/// handed, and a `Vec<u8>` is its representation rather than its identity.
///
/// [`std::io::Write`] is a supertrait because one encoder needs it —
/// `roaring`'s `serialize_into` takes any `io::Write` — and because getting it
/// for free is what keeps that path from needing a scratch buffer and a copy.
///
/// Deliberately one method per width rather than one generic over
/// `to_le_bytes`. The bodies are identical, but the names are what make a call
/// site's width readable, and byte-for-byte agreement with C is the point of
/// this format: `buf.u32(x)` next to C's `sizeof(int)` is checkable,
/// `buf.le(x)` is not.
pub trait EffectWrite: std::io::Write {
    fn u8(
        &mut self,
        v: u8,
    ) {
        self.bytes(&v.to_le_bytes());
    }
    fn u16(
        &mut self,
        v: u16,
    ) {
        self.bytes(&v.to_le_bytes());
    }
    fn u32(
        &mut self,
        v: u32,
    ) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(
        &mut self,
        v: u64,
    ) {
        self.bytes(&v.to_le_bytes());
    }
    fn i64(
        &mut self,
        v: i64,
    ) {
        self.bytes(&v.to_le_bytes());
    }
    fn f64(
        &mut self,
        v: f64,
    ) {
        self.bytes(&v.to_le_bytes());
    }

    /// A schema id — C's `LabelID` or `RelationID`. Four bytes, and
    /// **unsigned**.
    ///
    /// C declares both as `int` and reserves the negatives for sentinels:
    /// `GRAPH_NO_LABEL` and `GRAPH_NO_RELATION` are -1, `GRAPH_UNKNOWN_LABEL`
    /// and `GRAPH_UNKNOWN_RELATION` are -2. None of them is a schema id, and
    /// none of them belongs in a payload — an effect names a label that exists.
    /// A `u32` says that in the type rather than in a runtime check, which is
    /// the difference between "we do not send those" and "we cannot".
    ///
    /// The bytes are unchanged: every id we send is far below `i32::MAX`, and
    /// `u32::to_le_bytes` and `i32::to_le_bytes` agree there, so C reads the
    /// same four bytes into its `int` as before.
    fn schema_id(
        &mut self,
        v: u32,
    ) {
        self.bytes(&v.to_le_bytes());
    }

    /// Append raw bytes.
    ///
    /// The one method an implementor must write; every width above is defaulted
    /// in terms of it, so a new sink owes three methods rather than nine.
    ///
    /// `io::Write::write_all` would do, minus the `Result` a `Vec` can never
    /// produce and every caller would have to discard.
    fn bytes(
        &mut self,
        b: &[u8],
    );

    /// How many bytes have been written so far.
    ///
    /// For the one encoder that checks its own work: an `Ascending` segment
    /// writes a length prefix from `serialized_size` and then asks `roaring` to
    /// serialize, and the two disagreeing would mean the prefix lies about the
    /// bytes that follow.
    fn written(&self) -> usize;

    /// Hint that `n` more bytes are coming.
    ///
    /// Defaulted to nothing: it is an allocation strategy, not part of the
    /// format, and a sink that has no allocator to hint is still a valid sink.
    fn reserve(
        &mut self,
        _n: usize,
    ) {
    }

    /// A length, then that many bytes, the last of which is a NUL.
    ///
    /// **The length is a byte count, not `strlen`.** The two differ by exactly
    /// one case and it is reachable: this engine stores strings containing
    /// interior NUL bytes, because openCypher's `\uXXXX` escape can produce
    /// one — `RETURN size('a\u0000b')` is 3, and the value survives storage
    /// and an RDB reload intact. `strlen` on such a value reports 1, so a
    /// writer using `strlen + 1` emits `a` where this emits `a\0b`, and two
    /// implementations disagree on the bytes for one value. The spec said
    /// `strlen + 1` and was wrong; a reader must honour the length and must
    /// never call `strlen` on the payload.
    ///
    /// The terminator is still written, because C reads these into a `char[]`
    /// and treats them as C strings; omitting it makes C read the value plus
    /// whatever follows. It is a convenience for the common case, not the
    /// authority on where the value ends — the length is.
    ///
    /// There is deliberately no check for interior NULs here. Refusing to
    /// encode would turn a value the engine has already stored into a
    /// replication failure, and the divergence guard would then force a resync
    /// on every retry for as long as the graph holds that property. If interior
    /// NULs are to be disallowed, it has to happen where the value is created.
    fn string(
        &mut self,
        s: &str,
    ) {
        self.u64(s.len() as u64 + 1);
        self.bytes(s.as_bytes());
        self.u8(0);
    }
}

impl EffectWrite for Vec<u8> {
    fn bytes(
        &mut self,
        b: &[u8],
    ) {
        self.extend_from_slice(b);
    }

    fn written(&self) -> usize {
        self.len()
    }

    fn reserve(
        &mut self,
        n: usize,
    ) {
        Vec::reserve(self, n);
    }
}
