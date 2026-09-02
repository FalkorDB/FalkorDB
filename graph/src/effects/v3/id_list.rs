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

/// Segment count at which a collapse is attempted, and the factor it grows by.
///
/// Collapsing is O(n) in the segments and builds a bitmap, so it is amortised
/// rather than run per push: it fires when the count reaches
/// `COLLAPSE_FLOOR << k`. Below the floor there is nothing to win — roaring has
/// a floor of its own around 27 bytes, which a handful of ranges never exceed.
const COLLAPSE_FLOOR: usize = 16;

/// Smallest range-bytes total worth building a bitmap to try to beat.
///
/// Roaring cannot serialize smaller than this, so a run costing less is not
/// worth the allocation to measure.
const ROARING_FLOOR_BYTES: usize = 32;

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
    Range { base: u64, len: u64 },
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
        len: u64,
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
    /// How many ids this segment carries.
    #[must_use]
    pub const fn len(&self) -> u64 {
        match self {
            Self::Range { len, .. } | Self::Ascending { len, .. } => *len,
        }
    }

    /// The largest id in the segment, which is also the last one — both shapes
    /// are ascending internally.
    #[must_use]
    pub const fn max(&self) -> u64 {
        match self {
            Self::Range { base, len } => *base + *len - 1,
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
                        width_for(*len) as usize
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
                    let lw = width_for(*len);
                    h |= width_code(lw) << hdr::LEN_W_SHIFT;
                    buf.reserve(1 + bw as usize + lw as usize);
                    write_u8(buf, h);
                    write_narrow(buf, *base, bw);
                    write_narrow(buf, *len, lw);
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
        Ok(Self::Range { base, len })
    }

    /// The ids of this segment, in order.
    ///
    /// `Either` rather than a boxed trait object: a list is iterated once per
    /// row on the apply path, and a `Box` there is an allocation per segment.
    pub fn iter(&self) -> Either<std::ops::Range<u64>, roaring::treemap::Iter<'_>> {
        match self {
            Self::Range { base, len } => Either::Left(*base..*base + *len),
            Self::Ascending { bitmap, .. } => Either::Right(bitmap.iter()),
        }
    }

    /// The first id in the segment. Both shapes are ascending internally.
    fn min_id(&self) -> u64 {
        match self {
            Self::Range { base, .. } => *base,
            // Only ever built by `collapse` from a non-empty ascending run, so
            // the minimum always exists.
            Self::Ascending { bitmap, .. } => bitmap.min().unwrap_or(0),
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
    /// Segment count at which the next collapse is attempted.
    next_collapse: usize,
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
            next_collapse: COLLAPSE_FLOOR,
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
    pub fn push(
        &mut self,
        id: u64,
    ) {
        self.len += 1;
        match self.segments.last_mut() {
            // The overwhelmingly common path: one more consecutive id.
            Some(Segment::Range { base, len }) if id == *base + *len => {
                *len += 1;
                return;
            }
            Some(Segment::Ascending { bitmap, len, max }) if id > *max => {
                bitmap.insert(id);
                *len += 1;
                *max = id;
                return;
            }
            _ => {}
        }
        self.segments.push(Segment::Range { base: id, len: 1 });
        if self.segments.len() >= self.next_collapse {
            self.collapse();
            // Grow the trigger whether or not anything collapsed: a list that
            // cannot collapse — anything not ascending — must not rescan on
            // every push.
            self.next_collapse = self.segments.len().max(COLLAPSE_FLOOR) * 2;
        }
    }

    /// Fold maximal ascending runs of ranges into bitmaps, where that is
    /// cheaper on the wire.
    ///
    /// Two conditions, both from the design: the run has to ascend — a bitmap
    /// holds neither a repeat nor a step backwards — and the bitmap has to be
    /// smaller than the ranges it replaces. The second is measured with
    /// `serialized_size()`, so it is the same answer on any implementation.
    fn collapse(&mut self) {
        if self.segments.len() < 2 {
            return;
        }
        let mut out: SmallVec<[Segment; 2]> = SmallVec::new();
        let mut i = 0;
        while i < self.segments.len() {
            // How far the ascending run starting at `i` reaches, and what the
            // ranges in it cost.
            let mut j = i;
            let mut range_bytes = 0_usize;
            while j < self.segments.len() {
                if !matches!(self.segments[j], Segment::Range { .. }) {
                    break;
                }
                if j > i && self.segments[j].min_id() <= self.segments[j - 1].max() {
                    break;
                }
                // Sized as a non-final segment: the run being weighed is compared
                // against one bitmap, and only the list's very last segment can leave
                // its length implied. Costing them all as explicit is the
                // conservative side of the comparison.
                range_bytes += self.segments[j].encoded_len(false);
                j += 1;
            }

            // One segment is never worth a bitmap, and neither is a run whose
            // ranges already cost less than roaring's floor.
            if j - i < 2 || range_bytes < ROARING_FLOOR_BYTES {
                if j == i {
                    out.push(self.segments[i].clone());
                    i += 1;
                } else {
                    out.extend(self.segments[i..j].iter().cloned());
                    i = j;
                }
                continue;
            }

            let mut bitmap = RoaringTreemap::new();
            let mut len = 0_u64;
            for seg in &self.segments[i..j] {
                let Segment::Range { base, len: n } = seg else {
                    unreachable!("the run was built from ranges only")
                };
                // One container operation per run, however many ids it spans.
                bitmap.insert_range(*base..=*base + *n - 1);
                len += *n;
            }
            // `optimize()` is **normative**, not a tuning knob: an unoptimized
            // bitmap serializes to different bytes, so two engines that
            // disagree about calling it produce different buffers for the same
            // set.
            bitmap.optimize();
            let max = self.segments[j - 1].max();
            let candidate = Segment::Ascending { bitmap, len, max };
            if candidate.encoded_len(false) < range_bytes {
                out.push(candidate);
            } else {
                out.extend(self.segments[i..j].iter().cloned());
            }
            i = j;
        }
        self.segments = out;
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
        // No segment count on the wire. The record already carries how many ids
        // it holds, so the reader takes segments until that is satisfied — which
        // also lets the last one leave its length implied. Writing the count
        // instead cost four bytes per record and showed up as +14.6% on a
        // payload of ten thousand single-id records.
        let last = self.segments.len().saturating_sub(1);
        for (i, seg) in self.segments.iter().enumerate() {
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
    let mut out: Vec<u64> = Vec::with_capacity(count as usize);
    // Bounded by the count, not by a segment count off the wire: every segment
    // carries at least one id, so this cannot run longer than `count` turns,
    // and `count` is already capped above.
    while (out.len() as u64) < u64::from(count) {
        let remaining = u64::from(count) - out.len() as u64;
        let seg = Segment::decode(r, remaining)?;
        out.extend(seg.iter());
    }
    // `Segment::decode` refuses a segment longer than what remains, so an
    // overshoot cannot happen — but the invariant is what keeps row *k* bound
    // to the k-th id, so it is asserted rather than assumed.
    debug_assert_eq!(out.len() as u64, u64::from(count));
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
        assert_eq!(hex(&buf), "08 00");
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
        assert!(buf.len() < 32, "two ranges should be tiny, got {}", buf.len());
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
        let buf = [0x08_u8, 0];
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
        let buf = [0x00_u8, 0, 200];
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
        let buf = [0xC8_u8, 0];
        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 1), Err(DecodeError::BadEncoding(0xC8)));
    }

    #[test]
    fn a_range_that_would_wrap_is_rejected() {
        let mut buf = Vec::new();
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
    fn a_segment_list_that_falls_short_of_the_count_is_rejected() {
        // Segments are read until the record's count is satisfied, so a list
        // that does not reach it runs out of bytes rather than applying a
        // prefix. The final segment is the one exception by design: it leaves
        // its length implied and takes whatever the count has left, exactly as
        // the range encoding this replaces did.
        let mut buf = Vec::new();
        // One explicit-length range of two, and nothing after it.
        write_u8(
            &mut buf,
            (width_code(1) << hdr::BASE_W_SHIFT) | (width_code(1) << hdr::LEN_W_SHIFT),
        );
        buf.extend_from_slice(&[1, 2]);
        let mut r = Reader::new(&buf);
        assert!(matches!(
            read_ids(&mut r, 4),
            Err(DecodeError::UnexpectedEof { .. })
        ));
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
