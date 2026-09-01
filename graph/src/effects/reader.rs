//! The bounds-checked cursor. The writers are in [`super::writer`].

use super::DecodeError;

// ── reader ──

/// A bounds-checked cursor over an effects buffer.
///
/// Every read is checked. Trusting a length off the wire is how a malformed
/// buffer becomes a segfault rather than an error, which is what C does today.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    #[must_use]
    pub const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes not yet consumed.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// True once every byte has been consumed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Everything not yet consumed, without consuming it.
    #[must_use]
    pub fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    pub(super) fn take(
        &mut self,
        n: usize,
    ) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < n {
            return Err(DecodeError::UnexpectedEof {
                want: n,
                have: self.remaining(),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    /// Reject a count the remaining bytes could not possibly satisfy, *before*
    /// it reaches `Vec::with_capacity`.
    ///
    /// `pub(super)` rather than public: the blocks need it, but nothing outside
    /// the codec should be inventing its own bounds checks.
    ///
    /// `min_bytes_each` is the smallest number of bytes one entry can occupy —
    /// for a variable-width entry that is its fixed prefix, which bounds the
    /// allocation without having to predict the payload.
    pub(super) fn guard_count(
        &self,
        count: u64,
        min_bytes_each: usize,
    ) -> Result<usize, DecodeError> {
        let need = count.saturating_mul(min_bytes_each as u64);
        if need > self.remaining() as u64 {
            return Err(DecodeError::ImplausibleCount {
                count,
                remaining: self.remaining(),
            });
        }
        Ok(count as usize)
    }

    /// The next `N` bytes as a fixed-size array.
    ///
    /// Every primitive below is `from_le_bytes` over this. Written as a copy into
    /// an array rather than `take(N)?.try_into().unwrap()`, which is what these
    /// used to be: the slice is `N` long by construction, so that `unwrap` could
    /// not fire, but it read as a fallible conversion the code was ignoring — and
    /// `?` is no better an answer, since it would mean inventing a `DecodeError`
    /// variant for a state that cannot occur. This has no `Result` to discard.
    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut out = [0_u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take_array::<1>()?[0])
    }

    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take_array()?))
    }

    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take_array()?))
    }

    pub fn i32(&mut self) -> Result<i32, DecodeError> {
        Ok(i32::from_le_bytes(self.take_array()?))
    }

    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.take_array()?))
    }

    pub fn i64(&mut self) -> Result<i64, DecodeError> {
        Ok(i64::from_le_bytes(self.take_array()?))
    }

    pub fn f32(&mut self) -> Result<f32, DecodeError> {
        Ok(f32::from_le_bytes(self.take_array()?))
    }

    pub fn f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_le_bytes(self.take_array()?))
    }

    /// A C string: `u64` length **including** the NUL, then the bytes.
    /// `n` fixed-width values in one go.
    ///
    /// One bounds check and one allocation for the whole run, where a `for`
    /// loop over `u16()` or `i32()` pays a check per element and cannot be
    /// vectorized. Callers still `guard_count` first, so `n` is already known
    /// to be plausible against the bytes left.
    pub fn read_n<const W: usize, T>(
        &mut self,
        n: usize,
        from_le: fn([u8; W]) -> T,
    ) -> Result<Vec<T>, DecodeError> {
        // Saturating, so an `n` that would overflow the product asks for more
        // bytes than exist and fails the bounds check rather than wrapping to a
        // small read. Callers `guard_count` first, so this is belt and braces.
        let bytes = self.take(n.saturating_mul(W))?;
        Ok(bytes
            .chunks_exact(W)
            .map(|c| {
                let mut a = [0_u8; W];
                a.copy_from_slice(c);
                from_le(a)
            })
            .collect())
    }

    pub fn string(&mut self) -> Result<String, DecodeError> {
        let len = self.u64()?;
        if len == 0 {
            return Err(DecodeError::BadString);
        }
        let n = self.guard_count(len, 1)?;
        let raw = self.take(n)?;
        // The last byte is the terminator C requires; it is not part of the value.
        let (body, nul) = raw.split_at(n - 1);
        if nul != [0] {
            return Err(DecodeError::BadString);
        }
        String::from_utf8(body.to_vec()).map_err(|_| DecodeError::BadString)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::v3::{BlockEncoding, IdList, MAX_RECORD_IDS, read_ids};
    use crate::effects::writer::write_u64;

    // ── malformed input ──

    #[test]
    fn truncation_is_an_error_not_a_panic() {
        let ids: Vec<u64> = (0..10).collect();
        let mut buf = Vec::new();
        IdList::from(ids.as_slice()).encode(&mut buf);
        for cut in 0..buf.len() {
            let mut r = Reader::new(&buf[..cut]);
            // Must not panic, and must not claim success.
            assert!(read_ids(&mut r, 10).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn an_absurd_count_is_rejected_before_allocating() {
        // Two independent guards, and the tiny buffers here are the point: none
        // of these may reserve for the count they claim.
        //
        // The absolute cap is checked first, because it is the only one that can
        // bound `Range` and `Sorted` — both describe any count in a handful of
        // bytes, so weighing count against remaining bytes sees nothing wrong.
        for enc in [
            BlockEncoding::Plain,
            BlockEncoding::Compressed,
            BlockEncoding::Sorted,
            BlockEncoding::Range,
        ] {
            let buf = [enc as u8, 8, 0, 0];
            let mut r = Reader::new(&buf);
            assert_eq!(
                read_ids(&mut r, u32::MAX),
                Err(DecodeError::TooManyIds {
                    count: u64::from(u32::MAX),
                    max: MAX_RECORD_IDS,
                }),
                "{enc:?} accepted u32::MAX ids"
            );
        }

        // Under the cap, the length guard is what catches a count the buffer
        // cannot possibly hold — this is the one `Range` and `Sorted` escape.
        let under_cap = (MAX_RECORD_IDS - 1) as u32;
        let buf = [BlockEncoding::Plain as u8, 8, 0, 0];
        let mut r = Reader::new(&buf);
        assert!(matches!(
            read_ids(&mut r, under_cap),
            Err(DecodeError::ImplausibleCount { .. })
        ));
    }

    #[test]
    fn a_bad_encoding_byte_is_rejected() {
        let buf = [0xEE_u8];
        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 1), Err(DecodeError::BadEncoding(0xEE)));
    }

    #[test]
    fn a_string_without_its_nul_is_rejected() {
        let mut buf = Vec::new();
        write_u64(&mut buf, 3);
        buf.extend_from_slice(b"abc"); // no terminator
        let mut r = Reader::new(&buf);
        assert_eq!(r.string(), Err(DecodeError::BadString));
    }
}
