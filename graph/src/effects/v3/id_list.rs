//! `IdList` — a record's entity ids, held as the segments they form.
//!
//! Row *k* of a record belongs to the k-th id here, so this order is the
//! record's contract and nothing may reorder it.
//!
//! A list is a **sequence of segments**. A push that continues the current
//! segment extends it; one that does not opens a new segment. Nothing collapses
//! the whole list, and nothing is rebuilt at encode time to discover its shape —
//! the segments *are* the encoding, so `encode` writes what is already there.
//!
//! There are two kinds of segment and each writes itself:
//!
//! * [`Segment::Range`] — `len` consecutive ids from `base`. A single id is a
//!   range of one, which is why nothing else is needed to describe an arbitrary
//!   list.
//! * [`Segment::Ascending`] — a roaring bitmap, which several consecutive ranges
//!   collapse into when they ascend and the bitmap is genuinely cheaper.
//!
//! That second rule is decided by `serialized_size()`, which roaring reports
//! without serializing, so the comparison is exact rather than estimated — and
//! therefore reproducible by a second implementation, which is what makes it
//! safe to put in the spec.

use itertools::Either;
use roaring::RoaringTreemap;
use smallvec::SmallVec;

use super::{DecodeError, Reader, write_u8, write_u32};

/// The narrowest unsigned width that holds `max`.
///
/// One function for both jobs it has: sizing an id and sizing a run length.
//
// NOTE: `cow_btree/leaf/mod.rs` has `pow2_bytes_for` with identical arms. They
// are being folded into one shared helper — see review thread on `blocks.rs:115`.
#[must_use]
pub const fn width_for(max: u64) -> u8 {
    match max {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFFFF_FFFF => 4,
        _ => 8,
    }
}

/// A width as the two bits that go on the wire: 1, 2, 4, 8 → 0, 1, 2, 3.
const fn width_code(width: u8) -> u8 {
    match width {
        1 => 0,
        2 => 1,
        4 => 2,
        _ => 3,
    }
}

/// Inverse of [`width_code`]. Only the low two bits are meaningful, so every
/// input maps to a legal width and there is no invalid case to report.
const fn width_of_code(code: u8) -> u8 {
    match code & 0b11 {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => 8,
    }
}

/// Segments in an ascending run before a collapse is first attempted.
///
/// The trigger is the run's own segment count, which is a function of the ids —
/// so every implementation crosses it at the same push, and the bytes do not
/// depend on when an encoder chose to look. A counter over the whole list would
/// not have that property.
///
/// Eight because a bitmap cannot possibly win below it: roaring's smallest
/// serialization is around 27 bytes and a one-id range costs about three, so
/// there is nothing to weigh until the run is at least that long.
const COLLAPSE_ATTEMPT_AT: usize = 8;

/// One run of ids, and the only two shapes the wire knows.
///
/// Each variant writes itself — [`Segment::encode`] — so there is no second
/// enum mapping a representation onto an encoding, and no way for the two to
/// disagree.
#[derive(Clone, Debug)]
pub enum Segment {
    /// `len` consecutive ids ascending from `base`. `len` is never 0.
    ///
    /// A single id is `len == 1`, so this variant alone can describe any list,
    /// however unordered — which is why there is no plain or dictionary form.
    ///
    /// `len` is a `u32` rather than a `u64` because [`MAX_RECORD_IDS`] caps a
    /// record at 2^27 ids. That is not cosmetic: a list that cannot collapse
    /// holds one segment per id, so the eight bytes saved here are eight per
    /// id.
    Range { base: u64, len: u32 },
    /// Strictly ascending ids with gaps, as a run-optimized roaring bitmap.
    ///
    /// Only produced by [`IdList::collapse`], never by a push: a bitmap holds
    /// neither a repeat nor a step backwards, so it can only ever describe
    /// ranges that were already ascending.
    ///
    /// `len` and `max` are cached rather than asked of the bitmap.
    /// `RoaringTreemap::len()` sums every partition and `max()` reverse-walks
    /// them — neither is O(1) at any published version of the crate — and both
    /// are read on the encode path.
    Ascending {
        bitmap: RoaringTreemap,
        len: u32,
        max: u64,
    },
}

/// The segment header byte.
///
/// One byte carries everything but the payload, which is what keeps a
/// singleton-heavy list from paying for its own structure: a lone id is this
/// byte plus its narrowed base, two bytes in all.
///
/// ```text
/// bit 0    kind: 0 = Range, 1 = Ascending
/// bits 1-2 base width code   (Range)
/// bit 3    length is implied (Range) - see `LEN_IMPLIED`
/// bits 4-5 length width code (Range, when bit 3 is clear)
/// bits 6-7 reserved, must be zero
/// ```
mod hdr {
    pub const ASCENDING: u8 = 0b0000_0001;
    pub const BASE_W_SHIFT: u8 = 1;
    /// The segment runs to the end of the record, so its length is whatever
    /// the record's count has left over and is not written.
    ///
    /// Only ever set on the final segment. It is what makes the common shapes
    /// free: one consecutive run of any size is a header and a base, and so is
    /// a record carrying a single id.
    pub const LEN_IMPLIED: u8 = 0b0000_1000;
    pub const LEN_W_SHIFT: u8 = 4;
    pub const RESERVED: u8 = 0b1100_0000;
}

impl Segment {
    /// The largest id in the segment, which is also the last one — both shapes
    /// are ascending internally.
    #[must_use]
    pub const fn max(&self) -> u64 {
        match self {
            Self::Range { base, len } => *base + *len as u64 - 1,
            Self::Ascending { max, .. } => *max,
        }
    }

    /// Bytes this segment costs on the wire, without writing it.
    ///
    /// Exact, not an estimate: `serialized_size()` is what roaring itself will
    /// write, so the collapse decision below compares like with like.
    #[must_use]
    pub fn encoded_len(
        &self,
        len_implied: bool,
    ) -> usize {
        match self {
            Self::Range { base, len } => {
                1 + width_for(*base) as usize
                    + if len_implied {
                        0
                    } else {
                        width_for(u64::from(*len)) as usize
                    }
            }
            Self::Ascending { bitmap, .. } => 1 + 4 + bitmap.serialized_size(),
        }
    }

    /// The header byte, then whatever the variant needs.
    ///
    /// `len_implied` says this is the record's final segment, so its length is
    /// whatever the count has left and does not go on the wire. The caller owns
    /// that fact; the segment owns everything else about its own bytes.
    pub fn encode(
        &self,
        buf: &mut Vec<u8>,
        len_implied: bool,
    ) {
        match self {
            Self::Range { base, len } => {
                let bw = width_for(*base);
                let mut h = width_code(bw) << hdr::BASE_W_SHIFT;
                if len_implied {
                    h |= hdr::LEN_IMPLIED;
                    buf.reserve(1 + bw as usize);
                    write_u8(buf, h);
                    write_narrow(buf, *base, bw);
                } else {
                    let lw = width_for(u64::from(*len));
                    h |= width_code(lw) << hdr::LEN_W_SHIFT;
                    buf.reserve(1 + bw as usize + lw as usize);
                    write_u8(buf, h);
                    write_narrow(buf, *base, bw);
                    write_narrow(buf, u64::from(*len), lw);
                }
            }
            Self::Ascending { bitmap, .. } => {
                let n = bitmap.serialized_size();
                write_u8(buf, hdr::ASCENDING);
                write_u32(buf, n as u32);
                // Reserve first: roaring writes itself in many small pieces and
                // would otherwise grow the payload buffer under itself, each
                // realloc copying every byte written so far.
                buf.reserve(n);
                let before = buf.len();
                bitmap
                    .serialize_into(&mut *buf)
                    .expect("writing to a Vec cannot fail");
                debug_assert_eq!(
                    buf.len() - before,
                    n,
                    "serialized_size disagreed with serialize_into, so the length prefix lies"
                );
            }
        }
    }

    /// Read one segment back. `remaining` is how many ids the record still
    /// owes, which is both the bound on this segment and the value an implied
    /// length takes.
    fn decode(
        r: &mut Reader<'_>,
        remaining: u64,
    ) -> Result<Self, DecodeError> {
        let h = r.u8()?;
        if h & hdr::RESERVED != 0 {
            return Err(DecodeError::BadEncoding(h));
        }
        if h & hdr::ASCENDING != 0 {
            let blob_len = r.u32()? as usize;
            let blob = r.take(blob_len)?;
            let bitmap = RoaringTreemap::deserialize_from(blob)
                .map_err(|e| DecodeError::BadRoaring(e.to_string()))?;
            let len = bitmap.len();
            if len == 0 || len > remaining {
                return Err(DecodeError::CardinalityMismatch {
                    expected: remaining,
                    actual: len,
                });
            }
            let max = bitmap.max().unwrap_or(0);
            // `remaining` is bounded by `MAX_RECORD_IDS`, which is under
            // `u32::MAX`, so the narrowing above it cannot lose bits.
            let len = len as u32;
            return Ok(Self::Ascending { bitmap, len, max });
        }

        let base = read_narrow(r, width_of_code(h >> hdr::BASE_W_SHIFT))?;
        let len = if h & hdr::LEN_IMPLIED != 0 {
            remaining
        } else {
            read_narrow(r, width_of_code(h >> hdr::LEN_W_SHIFT))?
        };
        if len == 0 || len > remaining {
            return Err(DecodeError::BadRange { base, count: len });
        }
        // A run that would wrap past u64 describes ids that cannot exist, and
        // silently truncating it would bind rows to the wrong entities rather
        // than fail.
        base.checked_add(len - 1)
            .ok_or(DecodeError::BadRange { base, count: len })?;
        Ok(Self::Range {
            base,
            len: len as u32,
        })
    }

    /// The ids of this segment, in order.
    ///
    /// `Either` rather than a boxed trait object: a list is iterated once per
    /// row on the apply path, and a `Box` there is an allocation per segment.
    pub fn iter(&self) -> Either<std::ops::Range<u64>, roaring::treemap::Iter<'_>> {
        match self {
            Self::Range { base, len } => Either::Left(*base..*base + u64::from(*len)),
            Self::Ascending { bitmap, .. } => Either::Right(bitmap.iter()),
        }
    }
}

/// A record's entity ids, in row order.
///
/// Most lists are one segment: every id allocator hands out consecutive ids, so
/// a bulk create or a delete-by-label is a single [`Segment::Range`] from first
/// push to last and never allocates.
#[derive(Clone, Debug, Default)]
pub struct IdList {
    /// Inline for the common shapes — one range, or a range and a straggler —
    /// so the list itself costs no allocation either.
    segments: SmallVec<[Segment; 2]>,
    len: usize,
    /// Index of the first segment of the ascending run currently being built.
    ///
    /// A run ends the moment an id arrives that does not ascend past the last
    /// segment; the next one starts there. Only a run can become a bitmap.
    run_start: usize,
    /// Segments in that run, and the count at which the next collapse is tried.
    ///
    /// Both are functions of the ids alone, which is what keeps the encoding
    /// reproducible: the same sequence crosses the same thresholds at the same
    /// pushes on any implementation. `attempt_at` is `usize::MAX` once the run
    /// has collapsed, because further ascending ids then go straight into the
    /// bitmap and there is nothing left to weigh.
    run_segments: usize,
    attempt_at: usize,
}

/// By the ids, not by how they are segmented: the same sequence reached by
/// different push orders is the same list.
impl PartialEq for IdList {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for IdList {}

impl IdList {
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: SmallVec::new(),
            len: 0,
            run_start: 0,
            run_segments: 0,
            attempt_at: COLLAPSE_ATTEMPT_AT,
        }
    }

    /// A list expected to hold `n` ids.
    ///
    /// A hint and deliberately not an allocation: the segment count has no
    /// relationship to the id count — a million consecutive ids are one segment
    /// — so reserving by `n` would be wrong in the common case and wasteful in
    /// the rest.
    #[must_use]
    pub fn with_capacity(_n: usize) -> Self {
        Self::new()
    }

    /// Add an id, extending the current segment or opening a new one.
    ///
    /// The collapse decision is made here, in flight, rather than deferred to
    /// encode: the segments are the encoding, so what is held is what is
    /// written.
    pub fn push(
        &mut self,
        id: u64,
    ) {
        self.len += 1;

        // The two hot paths, in order of how often they are taken. Both extend
        // the segment already there and neither touches the run bookkeeping,
        // because neither changes how many segments the run has.
        match self.segments.last_mut() {
            // One more consecutive id — every bulk create, every
            // delete-by-label, from first push to last.
            Some(Segment::Range { base, len }) if id == *base + u64::from(*len) => {
                *len += 1;
                return;
            }
            // The run already collapsed, and this id continues it. Straight into
            // the bitmap, no new segment and nothing to re-weigh.
            Some(Segment::Ascending { bitmap, len, max }) if id > *max => {
                bitmap.insert(id);
                *len += 1;
                *max = id;
                return;
            }
            _ => {}
        }

        // A new segment. Whether it extends the current ascending run or starts
        // a fresh one is what decides if a bitmap is still reachable.
        let continues_run = self.segments.last().is_some_and(|last| id > last.max());
        self.segments.push(Segment::Range { base: id, len: 1 });

        if continues_run {
            self.run_segments += 1;
            if self.run_segments >= self.attempt_at {
                self.try_collapse_run();
            }
        } else {
            // A repeat or a step backwards ends the run: a bitmap can hold
            // neither, so everything before this is settled.
            self.run_start = self.segments.len() - 1;
            self.run_segments = 1;
            self.attempt_at = COLLAPSE_ATTEMPT_AT;
        }
    }

    /// Weigh the current run against the bitmap that would replace it, exactly.
    ///
    /// Called only at the thresholds above, so the bitmap is built O(log n)
    /// times over a list rather than per push — and each time the comparison is
    /// against `serialized_size()`, which is what roaring will actually write,
    /// not an estimate. That is what lets the rule be stated in the spec and
    /// reproduced.
    ///
    /// A decline doubles the threshold rather than retiring the run. The
    /// comparison is not monotone — crossing a 2^16 boundary adds a container
    /// and steps the bitmap's size up — so a run that loses now can win later.
    fn try_collapse_run(&mut self) {
        let run = &self.segments[self.run_start..];
        debug_assert!(run.len() >= 2, "a collapse needs a run to fold");

        let range_bytes: usize = run.iter().map(|s| s.encoded_len(false)).sum();

        let mut bitmap = RoaringTreemap::new();
        let mut len = 0_u32;
        for seg in run {
            let Segment::Range { base, len: n } = seg else {
                unreachable!("a run under consideration holds only ranges")
            };
            // One container operation per range, however many ids it spans.
            bitmap.insert_range(*base..=*base + u64::from(*n) - 1);
            len += *n;
        }
        // `optimize()` is **normative**, not a tuning knob: an unoptimized
        // bitmap serializes to different bytes, so two engines that disagree
        // about calling it produce different buffers for the same set.
        bitmap.optimize();
        let max = run[run.len() - 1].max();
        let candidate = Segment::Ascending { bitmap, len, max };

        if candidate.encoded_len(false) < range_bytes {
            self.segments.truncate(self.run_start);
            self.segments.push(candidate);
            self.run_segments = 1;
            // Nothing further to weigh: ascending ids now go straight into the
            // bitmap on the hot path above.
            self.attempt_at = usize::MAX;
        } else {
            self.attempt_at *= 2;
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many segments the list currently holds. The shape, for tests and
    /// benchmarks.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// The ids in order, without materializing anything.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.segments.iter().flat_map(Segment::iter)
    }

    /// The row count as it goes on the wire.
    ///
    /// Checked rather than cast: a list longer than `u32::MAX` cannot be
    /// encoded, and truncating silently would produce a record whose blocks
    /// disagree with its own count.
    #[must_use]
    pub fn count(&self) -> u32 {
        u32::try_from(self.len).expect("a record cannot carry more than u32::MAX entities")
    }

    /// Whether the whole list is one consecutive run.
    #[must_use]
    pub fn is_consecutive(&self) -> bool {
        self.len > 0
            && self.segments.len() == 1
            && matches!(self.segments[0], Segment::Range { .. })
    }

    /// The ids as a slice, materializing to produce one.
    ///
    /// Prefer [`Self::iter`]: this exists for the graph's bulk APIs, which take
    /// `&[u64]`, and every call undoes the allocation the segments avoid.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u64> {
        self.iter().collect()
    }

    /// Write the block: `u32 n_segments`, then each segment.
    ///
    /// Nothing is decided here. The segments already are the encoding, so this
    /// is a straight write of what the pushes built.
    pub fn encode(
        &self,
        buf: &mut Vec<u8>,
    ) {
        // Stated, not inferred from the record's id count. A reader could stop
        // once the ids are accounted for and save these four bytes, but then a
        // segment list would only be well-formed in the context of its record —
        // a truncated list and a complete one would be indistinguishable
        // without it.
        write_u32(buf, self.segments.len() as u32);
        let last = self.segments.len().saturating_sub(1);
        for (i, seg) in self.segments.iter().enumerate() {
            // The final segment leaves its length implied: the record's count
            // already fixes it, which is what keeps one consecutive run — the
            // commonest shape there is — at a header byte and a base.
            seg.encode(buf, i == last);
        }
    }

    /// Read `count` ids back.
    pub fn decode(
        r: &mut Reader<'_>,
        count: u32,
    ) -> Result<Self, DecodeError> {
        Ok(read_ids(r, count)?.into_iter().collect())
    }
}

/// Compare against a plain slice, so assertions do not have to build a vector.
///
/// No `Deref<Target = [u64]>`: segments have no slice to hand out, and offering
/// one would mean materializing behind the caller's back.
impl PartialEq<[u64]> for IdList {
    fn eq(
        &self,
        other: &[u64],
    ) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter().copied())
    }
}

impl<const N: usize> PartialEq<[u64; N]> for IdList {
    fn eq(
        &self,
        other: &[u64; N],
    ) -> bool {
        self == other.as_slice()
    }
}

impl FromIterator<u64> for IdList {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let mut out = Self::new();
        for id in iter {
            out.push(id);
        }
        out
    }
}

impl<const N: usize> From<[u64; N]> for IdList {
    fn from(ids: [u64; N]) -> Self {
        ids.into_iter().collect()
    }
}

impl From<&[u64]> for IdList {
    fn from(ids: &[u64]) -> Self {
        ids.iter().copied().collect()
    }
}

/// Write `value` at a fixed width, little-endian.
fn write_narrow(
    buf: &mut Vec<u8>,
    value: u64,
    width: u8,
) {
    match width {
        1 => buf.push(value as u8),
        2 => buf.extend_from_slice(&(value as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(value as u32).to_le_bytes()),
        _ => buf.extend_from_slice(&value.to_le_bytes()),
    }
}

/// Read one fixed-width little-endian value, zero-extended.
fn read_narrow(
    r: &mut Reader<'_>,
    width: u8,
) -> Result<u64, DecodeError> {
    Ok(match width {
        1 => u64::from(r.u8()?),
        2 => u64::from(r.u16()?),
        4 => u64::from(r.u32()?),
        _ => r.u64()?,
    })
}

/// The most ids one record may carry, and so the largest `Vec<u64>` a single
/// block can be talked into allocating.
///
/// `Reader::guard_count` bounds a count by weighing it against the bytes left to
/// read, which works whenever each item costs bytes on the wire. A segment does
/// not: one `Range` describes any count in a handful of bytes, and a roaring run
/// container holds 2^32 ids in about half a megabyte. For those the only
/// available bound is an absolute one.
///
/// `1 << 27` ids is a 1 GiB `Vec<u64>`. The largest record the benchmarks
/// produce is 10^6, so this leaves three orders of magnitude of headroom.
pub const MAX_RECORD_IDS: u64 = 1 << 27;

/// The ids of one block, whatever segments carried them.
pub fn read_ids(
    r: &mut Reader<'_>,
    count: u32,
) -> Result<Vec<u64>, DecodeError> {
    // Before anything is read: a segment can describe an enormous count from a
    // few bytes, so `guard_count` alone would see nothing wrong.
    if u64::from(count) > MAX_RECORD_IDS {
        return Err(DecodeError::TooManyIds {
            count: u64::from(count),
            max: MAX_RECORD_IDS,
        });
    }
    let n_segments = u64::from(r.u32()?);
    // No list can hold more segments than ids, since every segment carries at
    // least one. A tighter bound than the byte-length one, and it does not
    // depend on which shapes the segments turn out to be.
    if n_segments > u64::from(count) {
        return Err(DecodeError::ImplausibleCount {
            count: n_segments,
            remaining: count as usize,
        });
    }
    // The smallest segment is a header byte and a one-byte base.
    let n_segments = r.guard_count(n_segments, 2)?;

    let mut out: Vec<u64> = Vec::with_capacity(count as usize);
    for _ in 0..n_segments {
        let remaining = u64::from(count) - out.len() as u64;
        let seg = Segment::decode(r, remaining)?;
        out.extend(seg.iter());
    }
    // The guard that keeps row *k* bound to the k-th id: a segment list that
    // does not total the record's count would shift every later row onto the
    // wrong entity rather than fail.
    if out.len() as u64 != u64::from(count) {
        return Err(DecodeError::CardinalityMismatch {
            expected: u64::from(count),
            actual: out.len() as u64,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::testing::hex;

    /// 10,000 ids in two runs: two segments, and cheaper than any bitmap.
    fn run_structured() -> Vec<u64> {
        (0..5_000).chain(100_000..105_000).collect()
    }

    fn roundtrip(ids: &[u64]) -> Vec<u8> {
        let list = IdList::from(ids);
        let mut buf = Vec::new();
        list.encode(&mut buf);
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, ids.len() as u32).unwrap(),
            ids,
            "round-trip"
        );
        assert!(r.is_empty(), "{} bytes left over", r.remaining());
        buf
    }

    #[test]
    fn a_consecutive_list_is_one_segment_at_any_count() {
        // The property the whole arrangement exists for: flat in the count, and
        // no allocation — a million consecutive ids cost the same as ten.
        for n in [1_u64, 2, 10, 1_000, 1_000_000] {
            let ids: Vec<u64> = (0..n).collect();
            let list = IdList::from(ids.as_slice());
            assert_eq!(list.segment_count(), 1, "n = {n}");
            assert!(list.is_consecutive(), "n = {n}");
            roundtrip(&ids);
        }
    }

    #[test]
    fn a_singleton_is_a_range_of_one() {
        // n_segments=1, tag=0, widths=0, base=0, len=1.
        // Header alone plus the narrowed base: the length is implied by the
        // record's count, and there is no segment count. Two bytes, where the
        // encoding this replaces charged three.
        let buf = roundtrip(&[0]);
        assert_eq!(hex(&buf), "01 00 00 00 08 00");
    }

    #[test]
    fn a_gap_opens_a_new_segment_rather_than_collapsing_the_list() {
        let mut list = IdList::new();
        for id in [10, 11, 12] {
            list.push(id);
        }
        assert_eq!(list.segment_count(), 1);
        list.push(20);
        assert_eq!(list.segment_count(), 2, "a gap opens a segment");
        assert_eq!(list.len(), 4);
        assert_eq!(list, [10, 11, 12, 20]);
    }

    #[test]
    fn a_repeat_and_a_step_backwards_are_both_just_segments() {
        // The case the old ladder had to spill for. A segment describes a
        // single id, so neither needs a representation of its own.
        let mut list = IdList::new();
        for id in [10, 11, 20, 20, 5] {
            list.push(id);
        }
        assert_eq!(list, [10, 11, 20, 20, 5]);
        assert_eq!(list.segment_count(), 4);
        roundtrip(&[10, 11, 20, 20, 5]);
    }

    #[test]
    fn two_runs_stay_two_ranges_because_a_bitmap_is_dearer() {
        // The shape the old encoder sent as a bitmap. Two ranges are ~12 bytes
        // against roaring's floor, so the collapse rule correctly declines.
        let ids = run_structured();
        let list = IdList::from(ids.as_slice());
        assert_eq!(list.segment_count(), 2);
        let buf = roundtrip(&ids);
        assert!(
            buf.len() < 32,
            "two ranges should be tiny, got {}",
            buf.len()
        );
    }

    #[test]
    fn an_ascending_gapped_list_collapses_to_a_bitmap() {
        // Every id its own segment until the collapse fires: 10,000 ranges cost
        // far more than the bitmap that describes the same ids.
        let ids: Vec<u64> = (0..10_000).map(|i| i * 2).collect();
        let list = IdList::from(ids.as_slice());
        assert!(
            list.segment_count() < 100,
            "expected a collapse, got {} segments",
            list.segment_count()
        );
        roundtrip(&ids);
    }

    #[test]
    fn a_non_ascending_list_never_collapses() {
        // A bitmap holds neither a repeat nor a step backwards, so this stays
        // as ranges however many there are.
        let mut ids: Vec<u64> = (0..1_000).map(|i| i * 2).collect();
        ids.reverse();
        let list = IdList::from(ids.as_slice());
        assert_eq!(list.segment_count(), 1_000);
        roundtrip(&ids);
    }

    #[test]
    fn equality_is_about_the_ids_not_the_segments() {
        let a: IdList = (0..4).collect();
        let mut b = IdList::new();
        for id in [0, 1, 3, 2] {
            b.push(id);
        }
        assert_eq!(a.segment_count(), 1);
        assert!(b.segment_count() > 1);
        assert_ne!(a, b, "different order, different list");

        let rebuilt: IdList = a.iter().collect();
        assert_eq!(a, rebuilt);
    }

    // ── malformed input ──

    #[test]
    fn truncation_is_an_error_not_a_panic() {
        let ids: Vec<u64> = (0..10).map(|i| i * 3).collect();
        let mut buf = Vec::new();
        IdList::from(ids.as_slice()).encode(&mut buf);
        for cut in 0..buf.len() {
            let mut r = Reader::new(&buf[..cut]);
            assert!(read_ids(&mut r, 10).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn an_absurd_count_is_rejected_before_allocating() {
        let buf = [1_u8, 0, 0, 0, 0x08, 0];
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, u32::MAX),
            Err(DecodeError::TooManyIds {
                count: u64::from(u32::MAX),
                max: MAX_RECORD_IDS,
            })
        );
    }

    #[test]
    fn a_segment_longer_than_the_record_is_rejected() {
        // The guard that keeps row k bound to the k-th id: a run claiming more
        // ids than the record holds would shift every later row.
        let mut buf = Vec::new();
        write_u32(&mut buf, 1);
        buf.extend_from_slice(&[0x00, 0, 200]);
        let mut r = Reader::new(&buf);
        assert!(matches!(
            read_ids(&mut r, 4),
            Err(DecodeError::BadRange { .. })
        ));
    }

    #[test]
    fn a_header_with_reserved_bits_set_is_rejected() {
        // Forward compatibility: the two spare bits are refused rather than
        // masked off, so a future segment shape cannot be silently misread as a
        // range by a build that predates it.
        let mut buf = Vec::new();
        write_u32(&mut buf, 1);
        buf.extend_from_slice(&[0xC8, 0]);
        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 1), Err(DecodeError::BadEncoding(0xC8)));
    }

    #[test]
    fn a_range_that_would_wrap_is_rejected() {
        let mut buf = Vec::new();
        write_u32(&mut buf, 1);
        write_u8(
            &mut buf,
            (width_code(8) << hdr::BASE_W_SHIFT) | (width_code(8) << hdr::LEN_W_SHIFT),
        );
        buf.extend_from_slice(&(u64::MAX - 1).to_le_bytes());
        buf.extend_from_slice(&100_u64.to_le_bytes());
        let mut r = Reader::new(&buf);
        assert!(matches!(
            read_ids(&mut r, 100),
            Err(DecodeError::BadRange { .. })
        ));
    }

    #[test]
    fn a_segment_list_that_does_not_total_the_count_is_rejected() {
        // The stated segment count and the record's id count have to agree.
        // One explicit-length range of two ids, declared as the whole list,
        // against a record claiming four: the ids fall short and the record is
        // refused rather than applied as a prefix.
        let mut buf = Vec::new();
        write_u32(&mut buf, 1);
        write_u8(
            &mut buf,
            (width_code(1) << hdr::BASE_W_SHIFT) | (width_code(1) << hdr::LEN_W_SHIFT),
        );
        buf.extend_from_slice(&[1, 2]);
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, 4),
            Err(DecodeError::CardinalityMismatch {
                expected: 4,
                actual: 2,
            })
        );
    }

    #[test]
    fn width_boundaries() {
        assert_eq!(width_for(0xFF), 1);
        assert_eq!(width_for(0x100), 2);
        assert_eq!(width_for(0xFFFF), 2);
        assert_eq!(width_for(0x1_0000), 4);
        assert_eq!(width_for(0xFFFF_FFFF), 4);
        assert_eq!(width_for(0x1_0000_0000), 8);
        for w in [1_u8, 2, 4, 8] {
            assert_eq!(width_of_code(width_code(w)), w);
        }
    }
}
