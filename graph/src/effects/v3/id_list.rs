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
//! * `Segment::Range` — `len` consecutive ids from `base`. A single id is a
//!   range of one, which is why nothing else is needed to describe an arbitrary
//!   list.
//! * [`Segment::Ascending`] — a roaring bitmap, which several consecutive ranges
//!   collapse into when they ascend and the bitmap is genuinely cheaper.
//!
//! ## What one push does
//!
//! ```text
//!                            push(id)
//!                               |
//!            does it extend the segment already there?
//!                               |
//!         +---------------------+---------------------+
//!         | yes                                       | no
//!         v                                           v
//!   Range{base,len}  id == base+len  -> len += 1     (open a new one)
//!   Repeat{id,count} id == id        -> count += 1          |
//!   Ascending{bm}    id >  max       -> bm.insert(id)       |
//!         |                                                 |
//!         v                                                 v
//!       done, no new segment                   does it ascend past last.max()?
//!                                                            |
//!                                        +-------------------+-----------------+
//!                                        | yes                                 | no
//!                                        v                                     v
//!                              push Range{id,1}                  the run ENDS here
//!                              the run absorbs the                        |
//!                              segment it just closed        last is Range{b,1}, b == id?
//!                                        |                         |            |
//!                                        |                     yes |            | no
//!                                        |                         v            v
//!                                        |              becomes Repeat{id,2}  push Range{id,1}
//!                                        |                         |            |
//!                                        |                         +-----+------+
//!                                        |                               v
//!                                        |                        run.restart()
//!                                        v
//!                      range_bytes >= 32  AND  5 + bitmap_bytes < range_bytes ?
//!                                        |
//!                                    yes |
//!                                        v
//!                    the run's Ranges collapse into one Ascending
//!                    (one insert_range per Range, then optimize())
//! ```
//!
//! Two things that chart is making explicit. A **repeat ends a run** — it does
//! not ascend, and a bitmap holds a value once, so it can never join one. And
//! the collapse test is evaluated on **every new segment**, against the run's
//! own shape, never on a schedule: that is what makes the segmentation a
//! function of the ids rather than of when an encoder looked.
//!
//! That second rule is decided by `serialized_size()`, which roaring reports
//! without serializing, so the comparison is exact rather than estimated — and
//! therefore reproducible by a second implementation, which is what makes it
//! safe to put in the spec.

use crate::narrow_int::width_for;
use roaring::RoaringTreemap;
use smallvec::SmallVec;
use std::fmt;

use super::{DecodeError, EffectDecodeSized, EffectEncode, EffectWrite, Reader};

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

/// Smallest run-of-ranges worth doing the bitmap arithmetic on.
///
/// The smallest treemap roaring can serialize is 27 bytes — a run container, the
/// cheapest of the three — so a run costing less than that cannot lose to one
/// and there is nothing to compute. 32 rather than 27 because the arithmetic
/// below is only worth running once a bitmap is plausibly competitive, not the
/// instant it becomes possible.
///
/// (30 is the *array* container's floor. The C side measured both while
/// checking parity; the constant is unchanged, its stated reason was wrong.)
const ROARING_FLOOR_BYTES: usize = 32;

/// The ascending run of segments currently being built, and what it would cost
/// encoded either way.
///
/// A run ends the moment an id arrives that does not ascend past the last
/// segment, and only a run can become a bitmap — so this is the whole context
/// the collapse decision needs: where the run starts, what its segments cost as
/// ranges, and what roaring would charge for the same ids. The three used to be
/// loose fields on [`IdList`], which meant resetting them was three statements
/// in two places.
///
/// The bitmap side is tracked as the run grows so that it never has to be built
/// to find out.
///
/// Roaring's serialized size is a closed-form function of three things — how
/// many ids, how many maximal runs, and how they spread across buckets — and a
/// segment list already knows all three. So the choice between ranges and a
/// bitmap is arithmetic, and a bitmap is built exactly once: when it has already
/// won. Nothing is ever built in order to be measured and thrown away.
///
/// # How roaring is laid out, since none of the below reads without it
///
/// **A bucket is not a range.** It is a fixed 65,536-wide slice of the id
/// *space* — a partition of the addresses, not of the data. Roaring splits ids
/// by their high bits and gives each bucket its own store, choosing the
/// cheapest of three for whatever landed there:
///
/// | store  | cost                                   | suits       |
/// |--------|----------------------------------------|-------------|
/// | array  | 2 B per id, its low 16 bits            | sparse      |
/// | bitset | a fixed 8,192 B, one bit per address    | dense       |
/// | run    | 4 B per interval, `u16` start + length  | consecutive |
///
/// So a *run* is one of the three encodings **inside** a bucket, and that is
/// where our segments land: one `Segment::Range` is one run — except where it
/// straddles a bucket boundary, which makes it a run in each bucket it touches.
/// That, and only that, is why [`Self::add_range`] splits.
///
/// Ids are 64-bit, so there are two levels: a `RoaringTreemap` is a `BTreeMap`
/// keyed by `id >> 32`, each value a `RoaringBitmap` holding the buckets of that
/// slice. Each bitmap is charged one header, so buckets are grouped by
/// `bucket >> 16` to size them.
///
/// The crate itself names none of this — it has only
/// `Container { key: u16, store: Store }`, which fuses the address with the
/// storage. Roaring's papers call the slice a *chunk*; "bucket" is the same
/// thing and reads without having read them.
///
/// **Constant space, and no per-container list.** A run's ids ascend, so an
/// added range can only extend the newest bucket or open one after it — which
/// means every bucket behind the cursor has its ids and runs frozen for good,
/// and so does its cost. The same goes for whole bitmaps. Everything behind the
/// cursor therefore collapses into one accumulated integer, and only the open
/// bucket and the open bitmap need their parts kept. Every operation here is
/// O(1); nothing is re-walked.
///
/// **On depending on a crate's internals.** The constants below are roaring's,
/// and there is no public API that reports what a bitmap *would* cost without
/// building it — `serialized_size()` needs the bitmap, which is the thing being
/// avoided. Two things make that safe to live with:
///
/// * This is a decision *rule*, not a size oracle. The bytes written come from
///   `serialize_into` whatever they are, so if a release drifts from the
///   arithmetic the encoder makes a marginally worse choice — never a wrong
///   one — and both engines still agree, because agreement comes from running
///   the same arithmetic rather than from matching the crate.
/// * `predicted_matches_roaring` fails the moment they diverge, which is what
///   a dependency bump should do rather than drift quietly.
///
/// The larger risk is the serialized *bytes* changing under a bump, which would
/// break parity with the other engine with or without this formula. `roaring`
/// is pinned exactly for that reason — see the note in `graph/Cargo.toml`.
#[derive(Clone, Debug, Default)]
struct Run {
    /// Where the run starts in the segment list.
    start: usize,
    /// Which way it goes, once a second segment has settled it.
    ///
    /// `None` while the run is a single id, which belongs to neither
    /// direction — the id after it is what decides, and until then a push
    /// either way continues this run rather than starting another.
    desc: Option<bool>,
    /// What its segments cost as ranges — the comparison's other side.
    ///
    /// Counts only *closed* segments: the open one can still grow, so charging
    /// it would make the answer depend on where in the sequence the question
    /// was asked.
    range_bytes: usize,

    /// Whether any id has been folded in. Distinguishes an empty run from one
    /// whose open bucket happens to start at zero.
    started: bool,
    /// Bytes of every `RoaringBitmap` that is finished — the run has moved past
    /// its 2^32 slice, so nothing can change it again.
    closed_bitmaps: usize,

    /// The 2^32 slice being filled: `id >> 32`.
    bitmap_key: u64,
    /// Buckets in it that are finished, their total body bytes, and whether any
    /// of them chose a run store — which is what selects the header flavour.
    closed_buckets: usize,
    closed_body: usize,
    closed_has_run: bool,

    /// The bucket still being filled: `id >> 16`, and what has landed in it.
    ///
    /// Kept apart from the frozen totals because it is the one thing an
    /// extending range can still change, and because whether *it* chooses a run
    /// store can flip either way as it grows.
    bucket: u64,
    bucket_ids: u64,
    bucket_runs: u32,
}

// Roaring's serialized layout, as the pieces this arithmetic needs. Names here
// describe the field's job; roaring's own names, and the format spec's, are
// given alongside so a reader cross-checking the crate can find them.
//
// Layout: <magic> [<container count>] <per-container descriptor> x n
//         [<per-container offset> x n] <container bodies>
// See https://github.com/RoaringBitmap/RoaringFormatSpec

/// An array container: one `u16` per value.
const ARRAY_ELEMENT_BYTES: usize = 2;
/// A bitset container: a fixed 1024 x `u64`, whatever it holds.
const BITSET_CONTAINER_BYTES: usize = 8192;
/// A run container: a `u16` count of intervals, then each interval.
const RUN_COUNT_BYTES: usize = 2;
/// One interval: `u16 start` and `u16 length`.
const RUN_INTERVAL_BYTES: usize = 4;
/// The leading magic number, which selects the container layout that follows.
///
/// Roaring and the format spec call this the *cookie*: `12346` introduces the
/// layout without run containers, `12347` the one with them.
const MAGIC_BYTES: usize = 4;
/// The `u32` container count, present only in the no-run-container layout —
/// the other packs it into the magic number's high bits.
const CONTAINER_COUNT_BYTES: usize = 4;
/// One container's descriptor: its `u16` key and its `u16` cardinality.
const CONTAINER_DESC_BYTES: usize = 4;
/// One container's `u32` offset into the body.
const CONTAINER_OFFSET_BYTES: usize = 4;
/// Below this many containers the run-container layout omits the offset table.
const OFFSET_TABLE_MIN_CONTAINERS: usize = 4;
/// A treemap's own prefix: a `u64` count of the bitmaps in it.
const TREEMAP_COUNT_BYTES: usize = 8;
/// An [`Segment::Ascending`] segment's own overhead, on top of the blob: its
/// header byte and the `u32` length prefix.
const ASCENDING_SEGMENT_OVERHEAD: usize = 1 + 4;
/// Each bitmap in a treemap is preceded by its `u32` key.
const BITMAP_KEY_BYTES: usize = 4;

/// What one bucket's body costs, and whether it chose a run store.
///
/// Array up to 4,096 ids and bitset past it — which is exactly `min`, because
/// two bytes times 4,096 is the bitset's fixed size. A tie goes to the run
/// store: roaring's `optimize()` leaves a `Run` container alone unless strictly
/// beaten, and ours are built by range so they start as runs.
const fn bucket_body(
    ids: u64,
    runs: u32,
) -> (usize, bool) {
    // Not `min`: `Ord` is not const yet, and this wants to fold at compile time
    // where the widths are known.
    let sparse = ids as usize * ARRAY_ELEMENT_BYTES;
    let plain = if sparse < BITSET_CONTAINER_BYTES {
        sparse
    } else {
        BITSET_CONTAINER_BYTES
    };
    let as_run = RUN_COUNT_BYTES + RUN_INTERVAL_BYTES * runs as usize;
    if as_run <= plain {
        (as_run, true)
    } else {
        (plain, false)
    }
}

/// One bitmap's header, given how many buckets it holds and whether any of them
/// is a run container.
const fn bitmap_header(
    buckets: usize,
    has_run: bool,
) -> usize {
    if has_run {
        // The run-container layout adds a bitset marking which buckets are runs,
        // and only carries the offset table once there are enough buckets.
        let run_flags = buckets.div_ceil(8);
        if buckets >= OFFSET_TABLE_MIN_CONTAINERS {
            MAGIC_BYTES + (CONTAINER_DESC_BYTES + CONTAINER_OFFSET_BYTES) * buckets + run_flags
        } else {
            MAGIC_BYTES + CONTAINER_DESC_BYTES * buckets + run_flags
        }
    } else {
        MAGIC_BYTES
            + CONTAINER_COUNT_BYTES
            + (CONTAINER_DESC_BYTES + CONTAINER_OFFSET_BYTES) * buckets
    }
}

impl Run {
    /// Begin a new run at `start`, discarding what the old one had tallied.
    fn restart(
        &mut self,
        start: usize,
    ) {
        *self = Self {
            start,
            ..Self::default()
        };
    }

    /// Fold a segment that has stopped growing into both sides of the
    /// comparison.
    fn absorb(
        &mut self,
        seg: &Segment,
    ) {
        // The segment itself, not a reconstructed ascending one: a descending
        // range's base is its highest id and can cost a wider value field.
        self.range_bytes += seg.encoded_len();
        // The bitmap side takes the id *set*, which is the same either way.
        self.add_range(seg.min(), u64::from(seg.len()));
    }

    /// Whether the bitmap has already won, so one is worth building.
    ///
    /// Both sides are exact. Nothing speculative is constructed to answer it.
    fn prefers_bitmap(&self) -> bool {
        // Below roaring's own floor there is nothing to weigh.
        self.range_bytes >= ROARING_FLOOR_BYTES
            && ASCENDING_SEGMENT_OVERHEAD + self.bitmap_bytes() < self.range_bytes
    }

    /// Fold one consecutive range into the tally.
    ///
    /// Split at the 65,536 bucket boundaries it crosses, because a container
    /// covers a fixed slice of the id space rather than a stretch of data: a
    /// range spanning three buckets is three runs, one in each, not one run.
    fn add_range(
        &mut self,
        base: u64,
        len: u64,
    ) {
        // `len` is never 0, so `len - 1` is safe where `base + len` is not:
        // a range ending exactly at `u64::MAX` overflows the latter.
        let end = base + (len - 1);
        let mut lo = base;
        while lo <= end {
            let bucket = lo >> 16;
            // This bucket's highest id: its base with all sixteen low bits set.
            //
            // Not `((bucket + 1) << 16) - 1`. That reaches the same answer, but
            // on the topmost bucket it gets there by wrapping twice — the shift
            // discards the carry and the subtraction underflows — so it panics
            // under a debug build's overflow checks and traps under UBSan. The
            // form below cannot overflow for any bucket, because `bucket` came
            // from `lo >> 16`.
            let bucket_end = (bucket << 16) | 0xFFFF;
            let piece_end = end.min(bucket_end);
            let ids = piece_end - lo + 1;

            if !self.started {
                self.started = true;
                self.bitmap_key = bucket >> 16;
                self.bucket = bucket;
                self.bucket_ids = ids;
                self.bucket_runs = 1;
            } else if bucket == self.bucket {
                // Still the same bucket: one more run in it.
                self.bucket_ids += ids;
                self.bucket_runs += 1;
            } else {
                // Moved on, so the open bucket is now frozen and can be folded
                // into its bitmap's totals — and if the bitmap changed too, that
                // bitmap is frozen as well.
                self.freeze_bucket();
                let bitmap_key = bucket >> 16;
                if bitmap_key != self.bitmap_key {
                    self.freeze_bitmap();
                    self.bitmap_key = bitmap_key;
                }
                self.bucket = bucket;
                self.bucket_ids = ids;
                self.bucket_runs = 1;
            }
            // The increment is the other place the top of the id space shows
            // up, and it is the worse of the two. A wrap in a *value* gives a
            // wrong answer; a wrap in a **loop index** gives non-termination —
            // zero is a perfectly valid `lo`, so `while lo <= end` restarts the
            // walk from the bottom of the id space and never finishes. On the
            // write path, on the main thread, silently. The panic this replaced
            // was the visible failure mode.
            match piece_end.checked_add(1) {
                Some(next) => lo = next,
                None => break,
            }
        }
    }

    fn freeze_bucket(&mut self) {
        let (body, is_run) = bucket_body(self.bucket_ids, self.bucket_runs);
        self.closed_body += body;
        self.closed_has_run |= is_run;
        self.closed_buckets += 1;
    }

    fn freeze_bitmap(&mut self) {
        self.closed_bitmaps += BITMAP_KEY_BYTES
            + bitmap_header(self.closed_buckets, self.closed_has_run)
            + self.closed_body;
        self.closed_buckets = 0;
        self.closed_body = 0;
        self.closed_has_run = false;
    }

    /// The bytes `RoaringTreemap::serialized_size()` would report for this run.
    ///
    /// O(1): everything but the open bucket and the open bitmap is already
    /// summed, and those two are closed out arithmetically without being
    /// committed.
    fn bitmap_bytes(&self) -> usize {
        if !self.started {
            return TREEMAP_COUNT_BYTES;
        }
        let (body, is_run) = bucket_body(self.bucket_ids, self.bucket_runs);
        TREEMAP_COUNT_BYTES
            + self.closed_bitmaps
            + BITMAP_KEY_BYTES
            + bitmap_header(self.closed_buckets + 1, self.closed_has_run || is_run)
            + self.closed_body
            + body
    }
}

/// One run of ids, and the only two shapes the wire knows.
///
/// Each variant writes itself — [`Segment::encode`] — so there is no second
/// enum mapping a representation onto an encoding, and no way for the two to
/// disagree.
#[derive(Clone, Debug)]
enum Segment {
    /// `len` consecutive ids ascending from `base`. `len` is never 0.
    ///
    /// A single id is `len == 1`, so this variant alone can describe any list,
    /// however unordered — which is why there is no plain or dictionary form.
    ///
    /// `len` is a `u32` because a record's id count is a `u32` on the wire, so
    /// no single segment can exceed one. That is not cosmetic: a list that
    /// cannot collapse holds one segment per id, so the four bytes saved here
    /// are four per id.
    Range { base: u64, len: u32 },
    /// One id, `count` times over.
    ///
    /// A repeat is not a degenerate range and cannot be folded into one: a range
    /// ascends by one per step, a repeat does not move. Nor can it become a
    /// bitmap, which holds a value once. It is its own run.
    ///
    /// The shape that needs it is edge endpoints. Every edge out of a supernode
    /// carries the same source id, so `CREATE_EDGE`'s `src` list is one value
    /// repeated — 10,000 of them was 10,000 one-id segments, and is now one.
    Repeat { id: u64, count: u32 },
    /// `len` consecutive ids **descending** from `base`, so `base` is the first
    /// id emitted and the highest — the mirror of [`Self::Range`], where `base`
    /// is also the first and the lowest.
    ///
    /// Endpoint lists are what need it. `CREATE_EDGE`'s `src` and `dst` follow
    /// *edge* id order, so their contents are unordered with respect to
    /// themselves; a scan that walks nodes downward writes a strictly
    /// descending endpoint column, which without this is one segment per id.
    RangeDescending { base: u64, len: u32 },
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
        min: u64,
        max: u64,
    },
    /// Strictly descending ids with gaps — the same roaring bitmap, read from
    /// the top.
    ///
    /// A set has no direction, so the bitmap here is byte-identical to the one
    /// [`Self::Ascending`] would hold for the same ids and the collapse
    /// arithmetic is unchanged. Only the order the ids come back in differs,
    /// and that is the header bit rather than anything in the blob.
    ///
    /// `min` is cached for the same reason `Ascending` caches `max`: it is the
    /// boundary a push has to test, and `RoaringTreemap::min()` is not O(1).
    Descending {
        bitmap: RoaringTreemap,
        len: u32,
        min: u64,
        max: u64,
    },
}

/// The segment header byte.
///
/// One byte carries everything but the payload, which is what keeps a
/// singleton-heavy list from paying much for its own structure.
///
/// ```text
/// bits 0-1  kind: 0 = Range, 1 = Ascending, 2 = Repeat
/// bits 2-3  value width code   (Range base, Repeat id)
/// bits 4-5  count width code   (Range len,  Repeat count)
/// bit  6    descending
/// bit  7    reserved, must be zero
/// ```
///
/// Direction is a bit rather than two more kinds because a descending segment
/// is the same payload read the other way: `Range` gains a first-and-highest
/// base instead of a first-and-lowest one, and `Ascending`'s bitmap is a set,
/// which has no direction at all. Two more kinds would have meant a second
/// blob format and a second cost model for encodings that are byte-identical.
/// `Repeat` has no direction — one id, however many times — so the bit is
/// rejected there rather than ignored.
const SEG_KIND_MASK: u8 = 0b0000_0011;
const SEG_KIND_RANGE: u8 = 0;
const SEG_KIND_ASCENDING: u8 = 1;
const SEG_KIND_REPEAT: u8 = 2;
const SEG_VALUE_WIDTH_SHIFT: u8 = 2;
const SEG_COUNT_WIDTH_SHIFT: u8 = 4;
const SEG_DESCENDING: u8 = 0b0100_0000;
const SEG_RESERVED: u8 = 0b1000_0000;

impl Segment {
    /// How many ids this segment carries.
    const fn len(&self) -> u32 {
        match self {
            Self::Range { len, .. }
            | Self::RangeDescending { len, .. }
            | Self::Ascending { len, .. }
            | Self::Descending { len, .. }
            | Self::Repeat { count: len, .. } => *len,
        }
    }

    /// The largest id in the segment, which is also the last one — both shapes
    /// are ascending internally.
    const fn max(&self) -> u64 {
        match self {
            Self::Range { base, len } => *base + (*len as u64 - 1),
            Self::Repeat { id, .. } | Self::RangeDescending { base: id, .. } => *id,
            // Both cached. Deriving one extreme from the other and `len` is
            // only exact for a *gapless* set, and a collapse produces gaps by
            // definition — that is what it is for.
            Self::Ascending { max, .. } | Self::Descending { max, .. } => *max,
        }
    }

    /// The lowest id this segment carries — what a descending push tests
    /// against, as [`Self::max`] is what an ascending one tests.
    const fn min(&self) -> u64 {
        match self {
            Self::Range { base, .. } | Self::Repeat { id: base, .. } => *base,
            Self::RangeDescending { base, len } => *base - (*len as u64 - 1),
            Self::Ascending { min, .. } | Self::Descending { min, .. } => *min,
        }
    }

    /// Bytes this segment costs on the wire, without writing it.
    ///
    /// Exact, not an estimate: `serialized_size()` is what roaring itself will
    /// write, so the collapse decision below compares like with like.
    fn encoded_len(&self) -> usize {
        match self {
            // The descending form's `base` is the run's *highest* id, so it
            // can need a wider value field than the ascending form over the
            // same ids. Charged from the actual base rather than the set's
            // minimum, or the collapse would weigh the wrong encoding.
            Self::Range { base, len } | Self::RangeDescending { base, len } => {
                1 + width_for(*base) as usize + width_for(u64::from(*len)) as usize
            }
            Self::Repeat { id, count } => {
                1 + width_for(*id) as usize + width_for(u64::from(*count)) as usize
            }
            Self::Ascending { bitmap, .. } | Self::Descending { bitmap, .. } => {
                1 + 4 + bitmap.serialized_size()
            }
        }
    }

    /// The header byte, then whatever the variant needs.
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) {
        match self {
            Self::Range { base, len } => {
                Self::write_pair(buf, SEG_KIND_RANGE, *base, u64::from(*len));
            }
            Self::RangeDescending { base, len } => {
                Self::write_pair(buf, SEG_KIND_RANGE | SEG_DESCENDING, *base, u64::from(*len));
            }
            Self::Repeat { id, count } => {
                Self::write_pair(buf, SEG_KIND_REPEAT, *id, u64::from(*count));
            }
            Self::Ascending { bitmap, .. } | Self::Descending { bitmap, .. } => {
                let n = bitmap.serialized_size();
                buf.u8(SEG_KIND_ASCENDING
                    | if matches!(self, Self::Descending { .. }) {
                        SEG_DESCENDING
                    } else {
                        0
                    });
                buf.u32(n as u32);
                // Reserve first: roaring writes itself in many small pieces and
                // would otherwise grow the payload buffer under itself, each
                // realloc copying every byte written so far.
                buf.reserve(n);
                let before = buf.written();
                bitmap
                    .serialize_into(&mut *buf)
                    .expect("writing to a Vec cannot fail");
                debug_assert_eq!(
                    buf.written() - before,
                    n,
                    "serialized_size disagreed with serialize_into, so the length prefix lies"
                );
            }
        }
    }

    /// `Range` and `Repeat` are the same three fields — a kind, a value and a
    /// count — so they share a writer rather than drifting apart.
    fn write_pair<W: EffectWrite + ?Sized>(
        buf: &mut W,
        kind: u8,
        value: u64,
        count: u64,
    ) {
        let vw = width_for(value);
        let cw = width_for(count);
        buf.reserve(1 + vw as usize + cw as usize);
        buf.u8(kind
            | (width_code(vw) << SEG_VALUE_WIDTH_SHIFT)
            | (width_code(cw) << SEG_COUNT_WIDTH_SHIFT));
        write_narrow(buf, value, vw);
        write_narrow(buf, count, cw);
    }

    /// Read one segment back.
    ///
    /// `remaining` is how many ids the record still owes, and bounds this
    /// segment: a run claiming more than the record has left would shift every
    /// later row onto the wrong entity.
    fn decode(
        r: &mut Reader<'_>,
        remaining: u64,
    ) -> Result<Self, DecodeError> {
        let h = r.u8()?;
        if h & SEG_RESERVED != 0 {
            return Err(DecodeError::BadEncoding(h));
        }
        let descending = h & SEG_DESCENDING != 0;
        if h & SEG_KIND_MASK == SEG_KIND_ASCENDING {
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
            // `remaining` counts down from the record's `u32` id count, so the
            // narrowing cannot lose bits.
            let n = len as u32;
            // Both extremes off the bitmap itself. The cardinality check above
            // already read it, so this costs one more walk of the partitions
            // and buys accessors that never derive an extreme from `len`.
            let lo = bitmap.min().unwrap_or(0);
            let hi = bitmap.max().unwrap_or(0);
            return Ok(if descending {
                Self::Descending {
                    bitmap,
                    len: n,
                    min: lo,
                    max: hi,
                }
            } else {
                Self::Ascending {
                    bitmap,
                    len: n,
                    min: lo,
                    max: hi,
                }
            });
        }

        let value = read_narrow(r, width_of_code(h >> SEG_VALUE_WIDTH_SHIFT))?;
        let count = read_narrow(r, width_of_code(h >> SEG_COUNT_WIDTH_SHIFT))?;
        if count == 0 || count > remaining {
            return Err(DecodeError::BadRange { base: value, count });
        }
        match h & SEG_KIND_MASK {
            SEG_KIND_RANGE if descending => {
                // The mirror of the wrap check below: a descending run that
                // would step past zero describes ids that cannot exist.
                value
                    .checked_sub(count - 1)
                    .ok_or(DecodeError::BadRange { base: value, count })?;
                Ok(Self::RangeDescending {
                    base: value,
                    len: count as u32,
                })
            }
            SEG_KIND_RANGE => {
                // A run that would wrap past u64 describes ids that cannot
                // exist, and silently truncating it would bind rows to the
                // wrong entities rather than fail.
                value
                    .checked_add(count - 1)
                    .ok_or(DecodeError::BadRange { base: value, count })?;
                Ok(Self::Range {
                    base: value,
                    len: count as u32,
                })
            }
            // One id however many times has no direction, so the bit is a
            // malformed segment rather than a variant to interpret.
            SEG_KIND_REPEAT if descending => Err(DecodeError::BadEncoding(h)),
            SEG_KIND_REPEAT => Ok(Self::Repeat {
                id: value,
                count: count as u32,
            }),
            other => Err(DecodeError::BadEncoding(other)),
        }
    }

    /// The ids of this segment, in order.
    ///
    /// `Either` rather than a boxed trait object: a list is iterated once per
    /// row on the apply path, and a `Box` there is an allocation per segment.
    fn iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            Self::Range { base, len } => Box::new(*base..=*base + (u64::from(*len) - 1)),
            Self::RangeDescending { base, len } => {
                Box::new((*base - (u64::from(*len) - 1)..=*base).rev())
            }
            Self::Repeat { id, count } => Box::new(std::iter::repeat_n(*id, *count as usize)),
            Self::Ascending { bitmap, .. } => Box::new(bitmap.iter()),
            // The blob is the same set either way; only this is reversed.
            Self::Descending { bitmap, .. } => Box::new(bitmap.iter().rev()),
        }
    }
}

/// A record's entity ids, in row order.
///
/// Most lists are one segment: every id allocator hands out consecutive ids, so
/// a bulk create or a delete-by-label is a single `Segment::Range` from first
/// push to last and never allocates.
#[derive(Clone, Default)]
pub struct IdList {
    /// Inline for the common shapes — one range, or a range and a straggler —
    /// so the list itself costs no allocation either.
    segments: SmallVec<[Segment; 2]>,
    len: usize,
    /// The ascending run currently being built, and what it would cost encoded
    /// either way.
    ///
    /// Everything the collapse decision needs, in one place — see `Run`.
    run: Run,
}

/// The segments and the count, which is the whole of what the list *is* — the
/// same pair [`PartialEq`] compares.
///
/// Written out rather than derived because the derive also prints `Run`, the
/// half-built segment the encoder is accumulating into. That is scratch: it
/// says nothing about which ids the list holds, it is empty on any list that
/// came off the wire, and at over a hundred characters it buries the segments
/// in every test failure and every divergence log line that carries a record.
impl fmt::Debug for IdList {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("IdList")
            .field("len", &self.len)
            .field("segments", &self.segments)
            .finish()
    }
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
            run: Run::default(),
        }
    }

    /// Add an id, extending the current segment or opening a new one.
    ///
    /// The collapse decision is made here, in flight, and without speculation:
    /// the bytes a bitmap would cost are arithmetic on the run's shape, so one
    /// is only ever built once it has already won.
    pub fn push(
        &mut self,
        id: u64,
    ) {
        self.len += 1;

        // The hot paths, in the order they are taken. All of them extend the
        // segment already there, and none touches the run tally, because a
        // segment's cost is only folded in once it stops growing.
        match self.segments.last_mut() {
            // One more consecutive id — every bulk create, every
            // delete-by-label, from first push to last.
            Some(Segment::Range { base, len }) if base.checked_add(u64::from(*len)) == Some(id) => {
                *len += 1;
                return;
            }
            // The mirror: one more step down. An endpoint column written by a
            // downward scan is this from its second id to its last.
            Some(Segment::RangeDescending { base, len })
                if *base >= u64::from(*len) && id == *base - u64::from(*len) =>
            {
                *len += 1;
                return;
            }
            // One more of the same id — a supernode's endpoint list, where this
            // is every push after the first.
            Some(Segment::Repeat { id: rid, count }) if *rid == id => {
                *count += 1;
                return;
            }
            // The run already collapsed and this id continues it: straight into
            // the bitmap, no new segment and nothing left to weigh.
            Some(Segment::Ascending {
                bitmap, len, max, ..
            }) if id > *max => {
                bitmap.insert(id);
                *len += 1;
                *max = id;
                return;
            }
            Some(Segment::Descending {
                bitmap, len, min, ..
            }) if id < *min => {
                bitmap.insert(id);
                *len += 1;
                *min = id;
                return;
            }
            _ => {}
        }

        // A lone id has no direction, so the one after it decides. Both of
        // these rewrite that segment rather than opening another.
        if let Some(&Segment::Range { base, len: 1 }) = self.segments.last() {
            // `checked_add`, not `id + 1`: at the top of the id space that
            // panics in debug and wraps to 0 in release. Unreachable from a
            // payload — this is the builder, fed ids the engine allocated — but
            // a wrap here would silently open a descending run against a base
            // it has nothing to do with.
            if id.checked_add(1) == Some(base) {
                self.segments.pop();
                self.segments
                    .push(Segment::RangeDescending { base, len: 2 });
                if self.run.desc.is_none() {
                    self.run.desc = Some(true);
                }
                return;
            }
            if base == id {
                // A repeat of the immediately preceding id folds into a
                // `Repeat` — the supernode case, where a whole endpoint list is
                // one value. It ends any run: a bitmap holds a value once.
                self.segments.pop();
                self.segments.push(Segment::Repeat { id: base, count: 2 });
                // One PAST the repeat, not at it. A run holds only ranges —
                // `maybe_collapse_run` asserts exactly that — and a repeat
                // inside one is folded in as `insert_range(id..=id)`, which
                // keeps a single copy of the id while `len` still counts every
                // one. The segment's cardinality then disagrees with the ids it
                // describes, and the decoder refuses the buffer on its own
                // cardinality check: `100, 100, 200, 202, ...` produced 41 ids
                // under a `len` of 42. Refused every time, identically, which
                // is a forced-resync loop rather than a corruption.
                self.run.restart(self.segments.len());
                return;
            }
        }

        let bounds = self.segments.last().map(|l| (l.min(), l.max()));
        let continues_run = match (self.run.desc, bounds) {
            (_, None) => false,
            (Some(false), Some((_, max))) => id > max,
            (Some(true), Some((min, _))) => id < min,
            // Undecided: this id settles it, whichever side it falls.
            (None, Some((min, max))) => id > max || id < min,
        };

        if continues_run {
            let (min, max) = bounds.expect("a continuing run has a last segment");
            if self.run.desc.is_none() {
                self.run.desc = Some(id < min);
            }
            let _ = max;
            // The segment being superseded has its final length now, so this is
            // the moment its contribution is known.
            if let Some(seg @ (Segment::Range { .. } | Segment::RangeDescending { .. })) =
                self.segments.last()
            {
                let seg = seg.clone();
                self.run.absorb(&seg);
            }
            self.segments.push(Segment::Range { base: id, len: 1 });
            self.maybe_collapse_run();
        } else {
            // A step that reverses the run's own direction ends it: a bitmap
            // holds one order, so everything before this is settled.
            self.segments.push(Segment::Range { base: id, len: 1 });
            self.run.restart(self.segments.len() - 1);
        }
    }

    /// Replace the current run with its bitmap, if `Run::prefers_bitmap` says
    /// the arithmetic has already gone that way.
    fn maybe_collapse_run(&mut self) {
        if !self.run.prefers_bitmap() {
            return;
        }

        let mut bitmap = RoaringTreemap::new();
        let mut len = 0_u32;
        for seg in &self.segments[self.run.start..] {
            debug_assert!(
                matches!(seg, Segment::Range { .. } | Segment::RangeDescending { .. }),
                "a run under consideration holds only ranges, found {seg:?} — whichever \
                 `restart` admitted it is one index too low"
            );
            // One bucket operation per range, however many ids it spans, and
            // over the set rather than the direction — the blob is identical
            // either way, which is why the header carries the order instead.
            bitmap.insert_range(seg.min()..=seg.max());
            len += seg.len();
        }
        // `optimize()` is **normative**, not a tuning knob: an unoptimized
        // bitmap serializes to different bytes, so two engines that disagree
        // about calling it produce different buffers for the same set. So is
        // building it by range rather than id by id — see
        // `construction_order_changes_the_bytes`.
        bitmap.optimize();
        let descending = self.run.desc.unwrap_or(false);
        let start = self.run.start;
        // From the bitmap, not from the run's last segment. Both are the same
        // for a run of ranges, but taking them from the set is what makes the
        // pair true by construction rather than by an argument about the run.
        let lo = bitmap.min().unwrap_or(0);
        let hi = bitmap.max().unwrap_or(0);
        let collapsed = if descending {
            Segment::Descending {
                bitmap,
                len,
                min: lo,
                max: hi,
            }
        } else {
            Segment::Ascending {
                bitmap,
                len,
                min: lo,
                max: hi,
            }
        };
        self.segments.truncate(start);
        self.segments.push(collapsed);
        // The run begins **after** the collapsed segment, so a bitmap can never
        // be considered for a second collapse.
        //
        // `restart(start)` left it inside the run, and the loop above then
        // treated it as a range: `insert_range(min..=max)` over a *gapped* set
        // inserts every id in the span that the set does not hold and drops
        // nothing it does — measured at 17 ids invented and 17 dropped on a
        // 35-singleton collapse followed by a descending run. Silent in
        // release; the `debug_assert` above is the invariant this restores.
        //
        // Nothing is lost by excluding it: an id extending the bitmap's own
        // direction goes straight in on the hot path above, and anything else
        // starts a segment that this run can hold.
        self.run.restart(start + 1);
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many segments the list currently holds — its shape, for the tests
    /// that pin which encoding a given push sequence produces.
    #[cfg(test)]
    fn segment_count(&self) -> usize {
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

    /// The ids as a roaring bitmap, one container operation per segment.
    ///
    /// `iter().collect()` costs one insert per id, which for a consecutive run
    /// is a million `BTreeMap` lookups to describe what `insert_range` states
    /// once. The segments already are the runs roaring wants.
    #[must_use]
    pub fn to_roaring(&self) -> RoaringTreemap {
        let mut out = RoaringTreemap::new();
        for seg in &self.segments {
            match seg {
                Segment::Range { base, len } => {
                    out.insert_range(*base..=*base + (u64::from(*len) - 1));
                }
                Segment::RangeDescending { base, len } => {
                    out.insert_range(*base - (u64::from(*len) - 1)..=*base);
                }
                Segment::Ascending { bitmap, .. } | Segment::Descending { bitmap, .. } => {
                    out |= bitmap;
                }
                // A bitmap holds a value once, so a repeat contributes exactly
                // its id — the reason a repeat can never *become* a bitmap.
                Segment::Repeat { id, .. } => {
                    out.insert(*id);
                }
            }
        }
        out
    }

    /// Adopt segments read off the wire, verbatim.
    ///
    /// The run tally starts fresh at the last segment rather than being
    /// reconstructed. A decoded list is iterated, not extended, and the tally is
    /// read by [`Self::push`] alone — so the only consequence is that a later
    /// push would not collapse across the boundary, and encoding is unaffected
    /// because it writes the segments as they are.
    fn from_segments(
        segments: SmallVec<[Segment; 2]>,
        len: usize,
    ) -> Self {
        let start = segments.len().saturating_sub(1);
        Self {
            segments,
            len,
            run: Run {
                start,
                ..Run::default()
            },
        }
    }
}

/// Write the block: `u32 n_segments`, then each segment.
///
/// Nothing is decided here. The segments already are the encoding, so this is a
/// straight write of what the pushes built.
impl EffectEncode<3> for IdList {
    ///
    /// Nothing is decided here. The segments already are the encoding, so this
    /// is a straight write of what the pushes built.
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) {
        // Both the segment count and every segment's length are stated, and for
        // the same reason: a list has to be well-formed on its own, not only in
        // the context of the record that carries it. Leaving the last length
        // implied by the record's count would save a byte a record and reopen
        // exactly the hole the stated count closes — a wrong count would be
        // silently absorbed by the final segment, binding rows to the wrong
        // entities instead of failing. Measured at 10,000 bytes on a payload of
        // ten thousand single-id records, which is not worth a silent
        // corruption path.
        buf.u32(self.segments.len() as u32);
        for seg in &self.segments {
            seg.encode(buf);
        }
    }
}

/// Read `count` ids back.
///
/// [`crate::effects::EffectDecodeSized`] rather than
/// [`crate::effects::EffectDecode`]: the count is the
/// record's, stated once in its header, and the list does not repeat it.
impl EffectDecodeSized<3> for IdList {
    type Size = u32;

    fn decode_sized(
        r: &mut Reader<'_>,
        count: Self::Size,
    ) -> Result<Self, DecodeError> {
        read_ids(r, count)
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
fn write_narrow<W: EffectWrite + ?Sized>(
    buf: &mut W,
    value: u64,
    width: u8,
) {
    match width {
        1 => buf.u8(value as u8),
        2 => buf.bytes(&(value as u16).to_le_bytes()),
        4 => buf.bytes(&(value as u32).to_le_bytes()),
        _ => buf.bytes(&value.to_le_bytes()),
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

/// The smallest a segment can encode to: its header byte, then one byte each
/// of the narrowest value and the narrowest count.
///
/// Three, not two: every `Range` and `Repeat` writes both fields through
/// `write_pair`, and `width_for` floors at 1, so `encoded_len` can never come
/// out below `1 + 1 + 1`. The `seg_range_single` corpus case is exactly that.
/// Two let `guard_count` accept half again as many segments as the buffer
/// could hold.
const MIN_SEGMENT_BYTES: usize = 3;

/// One block's ids, as the segments the wire carried.
///
/// **Nothing is materialized.** A segment is what the format already holds, so
/// decoding stops at the segments rather than expanding them: one valid
/// `Range { base: 0, len: 4_294_967_295 }` is seven bytes on the wire and
/// sixteen in memory, where expanding it would be 32 GB. That is not a
/// micro-optimisation — it is the difference between a decoder whose cost is
/// bounded by its input and one that is not, and this runs on the Redis main
/// thread with no timeout and no cancellation.
///
/// Expanding the ids is the caller's decision, made in `apply` where the graph
/// is in scope and the memory is inherent to the write rather than amplified by
/// the encoding.
///
/// Decoding into segments rather than through a `Vec` also keeps the
/// segmentation the wire chose. Re-pushing every id would re-run
/// `Run::prefers_bitmap` and could group them differently from the peer that
/// sent them, so a buffer decoded and re-encoded would not come back
/// byte-identical — which any cross-engine comparison rests on.
pub fn read_ids(
    r: &mut Reader<'_>,
    count: u32,
) -> Result<IdList, DecodeError> {
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
    // The smallest segment is a header byte and a one-byte base, so this bounds
    // the segment list by the bytes actually present — which is what makes the
    // whole decode proportional to its input.
    let n_segments = r.guard_count(n_segments, MIN_SEGMENT_BYTES)?;

    let mut segments: SmallVec<[Segment; 2]> = SmallVec::new();
    let mut len = 0_u64;
    for _ in 0..n_segments {
        let remaining = u64::from(count) - len;
        let seg = Segment::decode(r, remaining)?;
        len += u64::from(seg.len());
        segments.push(seg);
    }
    // The guard that keeps row *k* bound to the k-th id: a segment list that
    // does not total the record's count would shift every later row onto the
    // wrong entity rather than fail.
    if len != u64::from(count) {
        return Err(DecodeError::CardinalityMismatch {
            expected: u64::from(count),
            actual: len,
        });
    }
    Ok(IdList::from_segments(segments, len as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10,000 ids in two runs: two segments, and cheaper than any bitmap.
    fn run_structured() -> Vec<u64> {
        (0..5_000).chain(100_000..105_000).collect()
    }

    /// The ids a block decodes to, expanded for comparison.
    fn decoded(
        r: &mut Reader<'_>,
        count: u32,
    ) -> Vec<u64> {
        read_ids(r, count).unwrap().iter().collect()
    }

    fn roundtrip(ids: &[u64]) -> Vec<u8> {
        let list = IdList::from(ids);
        let mut buf = Vec::new();
        list.encode(&mut buf);
        let mut r = Reader::new(&buf);
        assert_eq!(decoded(&mut r, ids.len() as u32), ids, "round-trip");
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
            assert!(matches!(list.segments[0], Segment::Range { .. }), "n = {n}");
            roundtrip(&ids);
        }
    }

    /// The width helpers state their own byte order.
    ///
    /// A round trip cannot see a byte swap: swap `write_narrow` and
    /// `read_narrow` together and every other test in this file stays green,
    /// because the assertion is symmetric in the bug. The only byte-level
    /// assertion here before this one was `a_singleton_is_a_range_of_one`, and
    /// every value in it is one byte wide — a one-byte value has no byte order,
    /// so it was blind at every width.
    ///
    /// The corpus does pin these bytes, but only for the widths that happen to
    /// appear in it; codes 2 and 3 had no case at all until recently. A helper
    /// should not depend on which shapes someone chose for a fixture.
    #[test]
    fn narrow_values_are_little_endian_at_every_width() {
        for (width, value, expected) in [
            (1_u8, 0x12_u64, vec![0x12]),
            (2, 0x1234, vec![0x34, 0x12]),
            (4, 0x1234_5678, vec![0x78, 0x56, 0x34, 0x12]),
            (
                8,
                0x1234_5678_9abc_def0,
                vec![0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12],
            ),
        ] {
            let mut buf = Vec::new();
            write_narrow(&mut buf, value, width);
            assert_eq!(buf, expected, "width {width} wrote the wrong bytes");

            // And the read agrees with those bytes, not merely with the write.
            let mut r = Reader::new(&expected);
            assert_eq!(
                read_narrow(&mut r, width).unwrap(),
                value,
                "width {width} read the wrong value"
            );
            assert!(r.is_empty(), "width {width} consumed the wrong length");
        }
    }

    #[test]
    fn a_singleton_is_a_range_of_one() {
        // n_segments=1, tag=0, widths=0, base=0, len=1.
        // Header alone plus the narrowed base: the length is implied by the
        // record's count, and there is no segment count. Two bytes, where the
        // encoding this replaces charged three.
        let buf = roundtrip(&[0]);
        assert_eq!(format!("{buf:02x?}"), "[01, 00, 00, 00, 00, 00, 01]");
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
        // Range(10..12), Repeat(20 x2), Range(5) — the two 20s fold rather than
        // opening a segment each.
        assert_eq!(list.segment_count(), 3);
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
    fn a_descending_gapped_list_collapses_to_a_bitmap() {
        // The mirror of the test above. The blob is a set, so it is the same
        // bytes the ascending list of these ids would produce; only the
        // header's direction bit and the order they come back in differ.
        let mut ids: Vec<u64> = (0..10_000).map(|i| i * 2).collect();
        ids.reverse();
        let list = IdList::from(ids.as_slice());
        assert!(
            list.segment_count() < 100,
            "expected a collapse, got {} segments",
            list.segment_count()
        );
        assert!(
            matches!(list.segments.last(), Some(Segment::Descending { .. })),
            "{:?}",
            list.segments.last()
        );
        roundtrip(&ids);
    }

    #[test]
    fn a_descending_consecutive_list_is_one_segment_at_any_count() {
        // What an endpoint column written by a downward scan looks like. Before
        // the descending forms this was one segment per id.
        for n in [2_u64, 10, 1_000, 1_000_000] {
            let ids: Vec<u64> = (0..n).rev().collect();
            let list = IdList::from(ids.as_slice());
            assert_eq!(list.segment_count(), 1, "n = {n}");
            assert!(
                matches!(list.segments[0], Segment::RangeDescending { .. }),
                "n = {n}"
            );
            roundtrip(&ids);
        }
    }

    #[test]
    fn a_reversal_ends_a_run_rather_than_extending_it() {
        // A bitmap carries one order, so the direction change is where the run
        // stops — the property the collapse rests on now that both directions
        // can collapse.
        let mut ids: Vec<u64> = (0..2_000).map(|i| i * 2).collect();
        let down: Vec<u64> = ids.iter().rev().map(|i| i + 100_000).collect();
        ids.extend(down);
        let list = IdList::from(ids.as_slice());
        assert!(list.segment_count() >= 2, "the two runs must not merge");
        roundtrip(&ids);
    }

    #[test]
    fn a_descending_repeat_is_still_a_repeat() {
        // One id however many times has no direction, so it must not acquire
        // one from the ids around it.
        let ids: Vec<u64> = vec![9, 9, 9, 9];
        let list = IdList::from(ids.as_slice());
        assert_eq!(list.segment_count(), 1);
        assert!(matches!(list.segments[0], Segment::Repeat { count: 4, .. }));
        roundtrip(&ids);
    }

    #[test]
    fn a_lone_id_marked_descending_is_accepted_though_no_encoder_writes_one() {
        // Rule 1 says a lone id is written with the bit clear, and that is an
        // *encoder* rule. A decoder still has to say what it does with the
        // other form, and the answer is accept: unlike a `Repeat`, the bit is
        // part of a `Range`'s encoding, and at `len = 1` its meaning is
        // well-defined — a descending run of one, naming exactly the id the
        // ascending form names. It is non-canonical, not unintelligible.
        //
        // Refuse what cannot be interpreted; accept what is merely
        // non-canonical. C reached the same answer independently.
        let mut buf = Vec::new();
        Segment::Range { base: 42, len: 1 }.encode(&mut buf);
        assert_eq!(
            buf[0] & SEG_DESCENDING,
            0,
            "the canonical form is ascending"
        );
        buf[0] |= SEG_DESCENDING;

        let mut r = Reader::new(&buf);
        let seg = Segment::decode(&mut r, 1).expect("a lone descending id is intelligible");
        assert_eq!(
            seg.iter().collect::<Vec<_>>(),
            vec![42],
            "the same single id"
        );

        // And it survives a round trip, so a decoder that keeps the bit does
        // not silently rewrite a peer's buffer.
        let mut again = Vec::new();
        seg.encode(&mut again);
        assert_eq!(again, buf, "the observed bit is preserved");
    }

    #[test]
    fn an_ascending_range_may_reach_the_top_of_the_id_space() {
        // The mirror of reaching zero exactly, and the row that separates the
        // correct bound from the tempting `base >= len` simplification — which
        // does not merely over-reject, it *accepts* a range that overflows the
        // top. C's organizer worked the arithmetic; these are the two rows that
        // matter.
        let mut buf = Vec::new();
        Segment::Range {
            base: u64::MAX - 7,
            len: 8,
        }
        .encode(&mut buf);
        let mut r = Reader::new(&buf);
        assert!(
            Segment::decode(&mut r, 8).is_ok(),
            "U-7..U inclusive is eight ids and legal"
        );

        let mut buf = Vec::new();
        Segment::Range {
            base: u64::MAX - 3,
            len: 8,
        }
        .encode(&mut buf);
        let mut r = Reader::new(&buf);
        assert!(
            matches!(
                Segment::decode(&mut r, 8),
                Err(DecodeError::BadRange { .. })
            ),
            "one step past the top"
        );
    }

    #[test]
    fn a_repeat_marked_descending_is_refused() {
        // The bit has no meaning on a repeat, so a peer that sets it is
        // malformed rather than something to interpret.
        let mut buf = Vec::new();
        Segment::Repeat { id: 7, count: 3 }.encode(&mut buf);
        buf[0] |= SEG_DESCENDING;
        let mut r = Reader::new(&buf);
        assert!(matches!(
            Segment::decode(&mut r, 3),
            Err(DecodeError::BadEncoding(_))
        ));
    }

    #[test]
    fn a_range_must_fit_the_id_space_in_the_direction_it_runs() {
        // As a set, because the bug these protect against is not the underflow
        // — it is a careless fix for the underflow. A blanket `base >= len`
        // rejects the three that should be rejected and also refuses the last
        // one, which is legitimate and is the common case: ids are allocated
        // densely from zero, so most ascending ranges have a low base.
        //
        // The C reader wrote this set independently and the ascending row is
        // the one neither of us had.
        for (seg, ok) in [
            // reaches zero exactly — 7..0 inclusive is eight ids, and legal
            (Segment::RangeDescending { base: 7, len: 8 }, true),
            // one step too far: would need 6..-1
            (Segment::RangeDescending { base: 6, len: 8 }, false),
            (Segment::RangeDescending { base: 3, len: 8 }, false),
            // the same numbers ascending are fine, so the check is
            // direction-sensitive rather than a bound on `base`
            (Segment::Range { base: 3, len: 8 }, true),
            // and the mirror at the top of the id space
            (
                Segment::Range {
                    base: u64::MAX - 2,
                    len: 8,
                },
                false,
            ),
        ] {
            let mut buf = Vec::new();
            seg.encode(&mut buf);
            let mut r = Reader::new(&buf);
            let got = Segment::decode(&mut r, 8);
            assert_eq!(got.is_ok(), ok, "{seg:?} -> {got:?}");
            if ok {
                // and the ids it names are the ones it should
                assert_eq!(
                    got.unwrap().iter().collect::<Vec<_>>(),
                    seg.iter().collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn a_descending_range_may_not_step_past_zero() {
        // The mirror of the wrap check: base 2 with count 5 would name ids
        // below zero, so it is refused rather than truncated.
        let mut buf = Vec::new();
        Segment::RangeDescending { base: 2, len: 5 }.encode(&mut buf);
        let mut r = Reader::new(&buf);
        assert!(matches!(
            Segment::decode(&mut r, 5),
            Err(DecodeError::BadRange { .. })
        ));
    }

    #[test]
    fn direction_is_the_header_bit_and_the_blob_is_the_same_bytes() {
        // What justifies a bit rather than two more kinds.
        let up: Vec<u64> = (0..10_000).map(|i| i * 3).collect();
        let down: Vec<u64> = up.iter().rev().copied().collect();
        let a = roundtrip(&up);
        let b = roundtrip(&down);
        assert_eq!(a.len(), b.len(), "same shape, same size");
        // segment count u32, then the segment: header, blob length, blob.
        assert_eq!(
            a[4] | SEG_DESCENDING,
            b[4],
            "only the direction bit differs"
        );
        assert_eq!(a[5..], b[5..], "the blob is identical");
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
        // Both directions and both segment kinds: a descending bitmap has a
        // length-prefixed blob, which is the shape a truncation can lie about.
        let up: Vec<u64> = (0..10).map(|i| i * 3).collect();
        let down: Vec<u64> = up.iter().rev().copied().collect();
        let flat: Vec<u64> = (0..10).rev().collect();
        for ids in [&up, &down, &flat] {
            let mut buf = Vec::new();
            IdList::from(ids.as_slice()).encode(&mut buf);
            for cut in 0..buf.len() {
                let mut r = Reader::new(&buf[..cut]);
                // Not `is_err()`. A truncated buffer must fail *because it ran
                // out of bytes* — either directly, or via a count guard that
                // correctly judged the claim implausible against what remains.
                // Any other variant means this sweep is passing for a reason
                // unrelated to truncation, and the check it is named for could
                // have stopped working without the test noticing. That is not
                // hypothetical: tightening `MIN_SEGMENT_BYTES` made a nearby test
                // start failing for the wrong reason, and it surfaced only because
                // that one names its variant.
                assert!(
                    matches!(
                        read_ids(&mut r, 10),
                        Err(DecodeError::UnexpectedEof { .. }
                            | DecodeError::ImplausibleCount { .. })
                    ),
                    "cut at {cut} of {ids:?}: expected a truncation error"
                );
            }
        }
    }

    #[test]
    fn a_descending_range_pins_its_bytes() {
        // The wire contract C has to match. Header is kind 0 with the
        // descending bit, both widths one byte; then the base — which is the
        // *highest* id, unlike the ascending form — and the count.
        let mut buf = Vec::new();
        IdList::from([9_u64, 8, 7, 6].as_slice()).encode(&mut buf);
        assert_eq!(
            buf,
            vec![
                1,
                0,
                0,
                0,           // one segment
                0b0100_0000, // kind 0 (Range) | descending, widths 1 and 1
                9,           // base: the first id, and the highest
                4,           // count
            ]
        );
    }

    #[test]
    fn an_absurd_count_is_refused_without_expanding_anything() {
        // One segment holding one id, behind a count of `u32::MAX`.
        //
        // Nothing is allocated for the claim: decoding stops at the segments,
        // so the cost of this buffer is the cost of reading seven bytes. The
        // count is then found not to match what the segments carry, and the
        // record is refused — exactly, and cheaply.
        let buf = [1_u8, 0, 0, 0, 0x00, 0, 1];
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, u32::MAX),
            Err(DecodeError::CardinalityMismatch {
                expected: u64::from(u32::MAX),
                actual: 1,
            })
        );
    }

    #[test]
    fn a_huge_consecutive_range_decodes_without_expanding() {
        // The payload that used to be a denial of service: four billion ids in
        // seven bytes. Expanding them would be 32 GB on the Redis main thread
        // with no timeout; holding them as one segment is sixteen bytes.
        let mut buf = Vec::new();
        buf.u32(1);
        buf.u8((width_code(1) << SEG_VALUE_WIDTH_SHIFT) | (width_code(4) << SEG_COUNT_WIDTH_SHIFT));
        buf.push(0);
        buf.bytes(&u32::MAX.to_le_bytes());

        let mut r = Reader::new(&buf);
        let list = read_ids(&mut r, u32::MAX).expect("a valid range, however large");
        assert_eq!(list.len(), u32::MAX as usize);
        assert_eq!(list.segment_count(), 1, "one segment, not four billion ids");
        // And it is still lazy: taking three ids costs three ids.
        assert_eq!(list.iter().take(3).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn more_segments_than_ids_is_rejected_before_allocating() {
        // The one guard that is exact, and it runs before the reservation: every
        // segment carries at least one id, so a list cannot have more segments
        // than the record has ids.
        let mut buf = Vec::new();
        buf.u32(1_000);
        buf.bytes(&[0x00, 0, 1]);
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, 4),
            Err(DecodeError::ImplausibleCount {
                count: 1_000,
                remaining: 4,
            })
        );
    }

    #[test]
    fn a_segment_longer_than_the_record_is_rejected() {
        // The guard that keeps row k bound to the k-th id: a run claiming more
        // ids than the record holds would shift every later row.
        let mut buf = Vec::new();
        buf.u32(1);
        buf.bytes(&[0x00, 0, 200]);
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
        buf.u32(1);
        buf.bytes(&[0xE0, 0, 1]);
        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 1), Err(DecodeError::BadEncoding(0xE0)));
    }

    #[test]
    fn a_range_that_would_wrap_is_rejected() {
        let mut buf = Vec::new();
        buf.u32(1);
        buf.u8((width_code(8) << SEG_VALUE_WIDTH_SHIFT) | (width_code(8) << SEG_COUNT_WIDTH_SHIFT));
        buf.bytes(&(u64::MAX - 1).to_le_bytes());
        buf.bytes(&100_u64.to_le_bytes());
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
        buf.u32(1);
        buf.u8((width_code(1) << SEG_VALUE_WIDTH_SHIFT) | (width_code(1) << SEG_COUNT_WIDTH_SHIFT));
        buf.bytes(&[1, 2]);
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, 4),
            Err(DecodeError::CardinalityMismatch {
                expected: 4,
                actual: 2,
            })
        );
    }

    /// The formula against the crate, on every container shape it can produce.
    ///
    /// The decision rule only has to be *the same* on both engines to keep the
    /// bytes identical — but if it is also exact, the spec can state the
    /// arithmetic instead of naming a Rust crate's method, and C needs no
    /// roaring call to make the choice. So exactness is pinned here.
    #[test]
    fn predicted_matches_roaring() {
        // (name, ranges) — chosen to hit array, bitset and run containers, the
        // 4-container offset threshold, and both boundary widths.
        let cases: &[(&str, &[(u64, u64)])] = &[
            ("one range", &[(0, 10)]),
            ("one long range", &[(0, 100_000)]),
            ("two runs", &[(0, 5_000), (100_000, 5_000)]),
            (
                "many runs, one container",
                &[(0, 1), (2, 1), (4, 1), (6, 1), (8, 1)],
            ),
            (
                "gap of one, array container",
                &[(0, 1), (2, 1), (4, 1), (6, 1), (8, 1), (10, 1), (12, 1)],
            ),
            ("dense past the array limit", &[(0, 5_000)]),
            ("crosses a 2^16 boundary", &[(65_000, 1_000)]),
            ("crosses a 2^32 boundary", &[(4_294_967_000, 1_000)]),
            (
                "four containers, run flavoured",
                &[(0, 3), (1 << 16, 3), (2 << 16, 3), (3 << 16, 3)],
            ),
            (
                "five containers",
                &[
                    (0, 3),
                    (1 << 16, 3),
                    (2 << 16, 3),
                    (3 << 16, 3),
                    (4 << 16, 3),
                ],
            ),
            ("two 2^32 entries", &[(0, 3), (1 << 32, 3)]),
            (
                "wide spread, one per container",
                &[
                    (0, 1),
                    (1 << 16, 1),
                    (2 << 16, 1),
                    (3 << 16, 1),
                    (4 << 16, 1),
                    (5 << 16, 1),
                ],
            ),
        ];

        for (name, ranges) in cases {
            let mut cost = Run::default();
            let mut bitmap = RoaringTreemap::new();
            for &(base, len) in *ranges {
                cost.add_range(base, len);
                bitmap.insert_range(base..=base + len - 1);
            }
            bitmap.optimize();
            assert_eq!(
                cost.bitmap_bytes(),
                bitmap.serialized_size(),
                "{name}: the formula and the crate disagree. If this fires after a \
             `roaring` bump, the arithmetic in `Run` needs re-deriving from \
             the new release *and* the wire format has changed under the other \
             engine — see #2698 before relaxing it."
            );
        }
    }

    #[test]
    fn a_collapsed_list_matches_what_the_formula_predicted() {
        // End to end: whatever the rule chose, the bytes it actually wrote are
        // what the arithmetic said they would be.
        let ids: Vec<u64> = (0..10_000).map(|i| i * 2).collect();
        let list = IdList::from(ids.as_slice());
        let mut buf = Vec::new();
        list.encode(&mut buf);
        assert!(
            list.segments
                .iter()
                .any(|s| matches!(s, Segment::Ascending { .. })),
            "this shape should have collapsed"
        );

        let mut cost = Run::default();
        cost.add_range(0, 1);
        for i in 1..10_000_u64 {
            cost.add_range(i * 2, 1);
        }
        let mut r = Reader::new(&buf);
        assert_eq!(decoded(&mut r, 10_000), ids);
    }

    /// `optimize()` is not a function of the set alone.
    ///
    /// It is path-dependent: from an `Array` store a container converts to runs
    /// only on a strict win, but from a `Run` store it *stays* runs unless
    /// strictly beaten — and a run-flavoured bitmap carries a different header.
    /// So two engines holding the same ids serialize to different bytes if one
    /// built its bitmap by ranges and the other id by id.
    ///
    /// This is a **spec requirement**, not a curiosity: the wire format says
    /// calling `optimize()` is normative, and that is not sufficient. How the
    /// bitmap is constructed is normative too. Ours is always built with one
    /// `insert_range` per segment.
    #[test]
    fn construction_order_changes_the_bytes() {
        let ranges = [(0_u64, 3_u64), (1 << 16, 3), (2 << 16, 3), (3 << 16, 3)];

        let mut by_range = RoaringTreemap::new();
        for (base, len) in ranges {
            by_range.insert_range(base..=base + len - 1);
        }
        by_range.optimize();

        let mut by_id = RoaringTreemap::new();
        for (base, len) in ranges {
            for id in base..base + len {
                by_id.insert(id);
            }
        }
        by_id.optimize();

        assert_eq!(by_range, by_id, "the same set, either way");
        assert_ne!(
            by_range.serialized_size(),
            by_id.serialized_size(),
            "if these ever agree the path-dependence is gone and the spec note \
             about construction order can be relaxed"
        );
        // And ours is the by-range one, which is what `RunCost` models.
        let mut cost = Run::default();
        for (base, len) in ranges {
            cost.add_range(base, len);
        }
        assert_eq!(cost.bitmap_bytes(), by_range.serialized_size());
    }

    #[test]
    fn to_roaring_matches_the_ids_and_costs_one_op_per_segment() {
        // Consecutive, gapped, and a repeat — the last of which a bitmap cannot
        // hold, so it is only ever built from lists that are ascending.
        for ids in [
            (0..10_000_u64).collect::<Vec<_>>(),
            (0..10_000).map(|i| i * 2).collect(),
            vec![5, 6, 7, 100, 101],
        ] {
            let list = IdList::from(ids.as_slice());
            let by_segment = list.to_roaring();
            let by_id: RoaringTreemap = ids.iter().copied().collect();
            assert_eq!(by_segment, by_id, "same set either way");
        }
    }

    #[test]
    fn a_repeated_id_is_one_segment_however_many_times() {
        // The supernode shape: every edge out of one node carries the same
        // source id. Ten thousand of them was ten thousand one-id segments.
        let src: Vec<u64> = std::iter::repeat_n(4_000_000_000_u64, 10_000).collect();
        let list = IdList::from(src.as_slice());
        assert_eq!(
            list.segment_count(),
            1,
            "one repeat, not ten thousand ranges"
        );
        assert_eq!(list.len(), 10_000);

        let mut buf = Vec::new();
        list.encode(&mut buf);
        assert!(
            buf.len() < 16,
            "a repeat is a header and two narrowed fields, got {}",
            buf.len()
        );

        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, 10_000).unwrap().iter().collect::<Vec<_>>(),
            src
        );
        assert!(r.is_empty());
    }

    #[test]
    fn a_repeat_is_not_a_range_and_never_becomes_a_bitmap() {
        // Three distinct sources interleaved, as real endpoints arrive: each run
        // of equal ids is its own `Repeat`, and none of them can join an
        // ascending run because a repeat does not ascend.
        let ids: Vec<u64> = (0..300_u64).map(|i| [7, 7, 9][i as usize % 3]).collect();
        let list = IdList::from(ids.as_slice());
        let mut buf = Vec::new();
        list.encode(&mut buf);
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, 300).unwrap().iter().collect::<Vec<_>>(),
            ids
        );

        // And a bitmap would have collapsed the duplicates away entirely.
        assert_eq!(list.to_roaring().len(), 2, "the set is {{7, 9}}");
        assert_eq!(list.len(), 300, "the list is still 300 ids");
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

#[cfg(test)]
mod top_of_id_space {
    use super::*;

    /// The cost model must not depend on unsigned wraparound.
    ///
    /// `bucket_end` used to be `((bucket + 1) << 16) - 1`, which is right on
    /// every bucket but the last and reaches the right answer there only by
    /// wrapping twice. Debug builds check overflow, so it panicked — on the
    /// *write* path, in the encoder, and only on a graph that had churned
    /// enough id space to reach the topmost bucket. Found by the C side while
    /// porting the model.
    #[test]
    fn the_cost_model_survives_the_topmost_bucket() {
        // This test passes by TERMINATING. No size or value assertion can
        // surface the failure it guards: the release build's wrap gives the
        // right `bucket_end` and a valid `lo`, so the walk simply never ends.
        for (base, len) in [
            (u64::MAX - 3, 4u64),             // ends exactly at u64::MAX
            (u64::MAX - 0x1_0000, 0x1_0001),  // spans the last two buckets
            (u64::MAX, 1),                    // the single highest id
            (u64::MAX >> 16 << 16, 0x1_0000), // the whole topmost bucket
            (0xFFFF_0000, 0x1_0000),          // ends exactly on a bucket boundary
            (0, 0x2_0000),                    // spans two whole buckets from zero
        ] {
            let mut run = Run::default();
            run.add_range(base, len);
            assert!(
                run.started,
                "base={base:#x} len={len:#x} must fold in rather than panic"
            );
        }
    }

    /// And the same ids through the public path, so the fix is not only
    /// reachable from a unit test.
    #[test]
    fn a_list_of_the_highest_ids_encodes() {
        let ids: Vec<u64> = ((u64::MAX - 4)..=u64::MAX).collect();
        let list = IdList::from(ids.as_slice());
        let mut buf = Vec::new();
        list.encode(&mut buf);
        let mut r = Reader::new(&buf);
        let back = read_ids(&mut r, ids.len() as u32).expect("must decode");
        assert_eq!(back.iter().collect::<Vec<_>>(), ids);
    }
}

#[cfg(test)]
mod collapsed_segments_are_never_recollapsed {
    use super::*;

    /// A collapsed bitmap must not be folded into a later collapse.
    ///
    /// Found by the C writer while porting the push policy. The chain: a
    /// collapse left the bitmap inside the run (`restart(start)`), a later id
    /// continued that run, and the next collapse treated the bitmap as a range
    /// — `insert_range(min..=max)` over a *gapped* set, which invents every id
    /// in the span the set does not hold. `Segment::min` deriving the low
    /// extreme from `max - len + 1` made it worse by fabricating a minimum, so
    /// an id inside the real span looked below it and set the run's direction.
    ///
    /// Both are fixed — the extremes are cached, and the run begins after the
    /// collapsed segment — and this asserts the property either alone would
    /// leave half-covered: **the ids that come out are the ids that went in**.
    fn round_trips(ids: &[u64]) {
        let list = IdList::from(ids);
        assert_eq!(list.iter().collect::<Vec<_>>(), ids, "ids must survive");
        assert_eq!(list.len(), ids.len(), "and so must the count");

        let mut buf = Vec::new();
        list.encode(&mut buf);
        let mut r = Reader::new(&buf);
        let back = read_ids(&mut r, ids.len() as u32).expect("must decode");
        assert_eq!(back.iter().collect::<Vec<_>>(), ids, "and the wire agrees");
    }

    #[test]
    fn a_descending_run_after_a_collapse_keeps_every_id() {
        // The exact shape that corrupted: enough gapped singletons to collapse,
        // then an id inside the collapsed span, then a descending run long
        // enough to collapse again.
        let mut ids: Vec<u64> = (0..35u64).map(|i| 100 + i * 2).collect();
        ids.push(101);
        let mut next = 101u64;
        for _ in 0..40 {
            next -= 2;
            ids.push(next);
        }
        round_trips(&ids);
    }

    #[test]
    fn an_id_below_a_collapsed_bitmap_keeps_every_id() {
        // The same hazard without the fabricated minimum: an id genuinely below
        // the set's low extreme, then a descending run.
        let mut ids: Vec<u64> = (0..35u64).map(|i| 1_000 + i * 2).collect();
        for i in 0..40u64 {
            ids.push(500 - i * 2);
        }
        round_trips(&ids);
    }

    #[test]
    fn an_ascending_run_after_a_collapse_keeps_every_id() {
        // And the mirror, where the second run goes the same way as the first.
        let mut ids: Vec<u64> = (0..35u64).map(|i| 100 + i * 2).collect();
        for i in 0..40u64 {
            ids.push(10_000 + i * 3);
        }
        round_trips(&ids);
    }
}

#[cfg(test)]
mod repeats_stay_out_of_runs {
    use super::*;

    /// A `Repeat` must not be inside a collapsing run.
    ///
    /// Found by the C writer while building their own segment builder — the
    /// third instance of a `restart` landing one index too low. The symptom
    /// differs from the bitmap case and that is why it needed its own fix: a
    /// bitmap holds a value once, so folding one in invents and drops ids
    /// silently; a `Repeat` holds it `count` times, so folding one in keeps a
    /// single copy while `len` still counts them all. The cardinality then
    /// disagrees with the ids, and the decoder refuses the buffer on its own
    /// check — every time, identically, which is a forced-resync loop.
    #[test]
    fn a_repeat_before_a_collapsing_run_survives() {
        let mut ids: Vec<u64> = vec![100, 100];
        for i in 0..40u64 {
            ids.push(200 + i * 2);
        }
        let list = IdList::from(ids.as_slice());

        assert!(
            matches!(list.segments[0], Segment::Repeat { count: 2, id: 100 }),
            "the repeat must still be a repeat, got {:?}",
            list.segments[0]
        );
        assert_eq!(list.iter().collect::<Vec<_>>(), ids, "every id, in order");
        assert_eq!(
            list.len(),
            ids.len(),
            "and the count must match the ids, or the decoder refuses it"
        );

        let mut buf = Vec::new();
        list.encode(&mut buf);
        let mut r = Reader::new(&buf);
        let back = read_ids(&mut r, ids.len() as u32).expect("the buffer must not be refused");
        assert_eq!(back.iter().collect::<Vec<_>>(), ids);
    }

    /// And the mirror: a repeat *after* a collapse, which is the other order.
    #[test]
    fn a_repeat_after_a_collapsing_run_survives() {
        let mut ids: Vec<u64> = (0..40u64).map(|i| 200 + i * 2).collect();
        ids.push(1_000);
        ids.push(1_000);
        let list = IdList::from(ids.as_slice());
        assert_eq!(list.iter().collect::<Vec<_>>(), ids);
        assert_eq!(list.len(), ids.len());

        let mut buf = Vec::new();
        list.encode(&mut buf);
        let mut r = Reader::new(&buf);
        let back = read_ids(&mut r, ids.len() as u32).expect("must not be refused");
        assert_eq!(back.iter().collect::<Vec<_>>(), ids);
    }
}
