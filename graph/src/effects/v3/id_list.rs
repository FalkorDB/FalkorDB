//! `IdList` — a record's entity ids, and the encoding that suits them.
//!
//! Row *k* of a record belongs to the k-th id here, so this order is the
//! record's contract and nothing may reorder it. Everything the encoder needs
//! to know about the ids — strictly ascending, one consecutive run, the largest
//! of them — is maintained as they arrive, so choosing an encoding costs no
//! extra pass over the list.

use itertools::Either;
use roaring::RoaringTreemap;

use super::{BlockEncoding, DecodeError, Reader, write_u8, write_u32};

/// The narrowest unsigned width that holds `max`.
///
/// One function for both jobs it used to have: sizing an id, and sizing a rank
/// into a dictionary. A dictionary of `n` entries addresses `0..n-1`, so its
/// ranks are `width_for(n - 1)`.
#[must_use]
pub const fn width_for(max: u64) -> u8 {
    match max {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFFFF_FFFF => 4,
        _ => 8,
    }
}

/// How much smaller the bitmap must be before it is worth its CPU.
///
/// Smaller-wins is the right rule for [`BlockEncoding::Range`], which is
/// cheaper on every axis, and the wrong one here. Measured against the plain
/// form on the same input (`benches/effects_blocks.rs`, 10,000 gapped ids), the
/// bitmap costs about **7.4 ns per id** more across encode and decode together
/// — 43.1 µs against 4.8 to write and 36.7 against 0.8 to read. What it can
/// save is bounded by the id width, at most 8 bytes and usually 2, so a
/// marginal win is always a bad trade: the delete-by-half payload observed on
/// the wire saved 1,777 bytes for roughly 37 µs, which only repays itself on a
/// link slower than ~48 MB/s.
///
/// A large win is a different thing. The bitmap only gets far below the plain
/// form when the ids form runs, and then its cost is amortised over a handful
/// of containers rather than paid per id. So the gate asks whether the bitmap
/// found structure, not whether it saved a byte. Four is the point where the
/// two regimes separate cleanly: a gap-of-one set lands near 2.4x and a
/// run-structured one lands in the hundreds.
///
/// This mirrors the duplication probe that guards the dictionary, and for the
/// same reason — both encodings are size wins that can be CPU losses.
const SORTED_MIN_RATIO: usize = 4;

/// How many ids the duplication probe samples.
const DUP_PROBE_SAMPLE: usize = 128;

/// How many ids per run make the range-at-a-time bitmap build worthwhile.
///
/// Below this the list is mostly runs of one, where `insert_range` pays its
/// container-splitting logic per id; above it each call covers enough ids to
/// pay for itself. Measured at 10,000 ids: one run builds 17x faster by range,
/// runs of one build 28x faster in bulk.
const RUN_BUILD_RATIO: usize = 8;

/// A record's entity ids, in row order.
///
/// Built by pushing ids in the order the query produced them. The shape facts
/// are maintained on the way in rather than rediscovered by scanning, which is
/// what lets [`Self::encode`] pick an encoding without a pass of its own.
#[derive(Clone, Debug, Default)]
pub struct IdList {
    ids: Ids,
}

/// By the ids, not by how they are held: a range and a spilled vector holding
/// the same sequence are the same list, and only one of them is reachable
/// depending on the order the pushes arrived in.
impl PartialEq for IdList {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for IdList {}

/// How the ids are currently held.
///
/// The representation *is* the encoding decision, so it degrades as pushes
/// violate it rather than being re-derived from a vector at encode time. Every
/// id allocator hands out consecutive ids, so a bulk create or a delete-by-label
/// stays in [`Ids::Range`] from first push to last and never allocates — where
/// before it built a `Vec<u64>` (8 MB at a million ids) whose only purpose was
/// to prove, at encode time, that it held one run.
#[derive(Clone, Debug)]
enum Ids {
    /// `len` consecutive ids ascending from `base`. `len == 0` is the empty
    /// list, which is why `is_consecutive` has to ask.
    Range { base: u64, len: usize },
    /// Strictly ascending, with gaps a range cannot describe.
    ///
    /// A bitmap round-trips an ascending unique sequence exactly, so both forms
    /// this shape can take — the bare bitmap, or plain ids — stay writable from
    /// it, and the `SORTED_MIN_RATIO` comparison is unchanged: `len` and `max`
    /// give `plain_bytes` without a vector.
    ///
    /// Boxed so the hot `Range` arm does not carry a `RoaringTreemap`'s width
    /// through every push.
    ///
    /// **What this rung costs**, measured by `id_list/construct` in
    /// `benches/effects_blocks.rs` against a ladder without it:
    ///
    /// | shape       | without | with   |
    /// |-------------|---------|--------|
    /// | consecutive | 1.181ms | 1.179ms |
    /// | shuffled    | 2.982ms | 3.003ms |
    /// | gapped      | 2.029ms | 7.639ms |
    ///
    /// (one million ids each.) Consecutive pays nothing — it never reaches
    /// here, and keeping it that way is why [`IdList::widen`] is out of line.
    /// Gapped pays ~3.8x, and that part is inherent: a per-id `insert` is a
    /// `BTreeMap` lookup for the high 32 bits plus a container insert, against
    /// a bounds check and a store for `Vec::push`. What it buys is the vector
    /// never existing for a shape that is ascending but not contiguous.
    ///
    /// Shuffled pays a little because it now passes *through* here before
    /// spilling, expanding a bitmap it turned out not to need.
    Ascending {
        bitmap: Box<RoaringTreemap>,
        len: usize,
        max: u64,
    },
    /// Whatever neither of the above could describe, from that push onward.
    ///
    /// `ascending`, `runs` and `max` ride along because `encode` needs all
    /// three and none of them wants a pass of its own.
    Spilled {
        ids: Vec<u64>,
        /// Strictly ascending so far. Strict rather than merely non-decreasing:
        /// a bitmap cannot hold a repeat, so an equal pair would be dropped and
        /// shift every later row by one.
        ascending: bool,
        /// How many maximal consecutive runs the ids form. `len()` runs means
        /// no two ids are adjacent, which is what decides how the bitmap is
        /// built.
        runs: usize,
        max: u64,
    },
}

impl Default for Ids {
    fn default() -> Self {
        Self::Range { base: 0, len: 0 }
    }
}

impl IdList {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ids: Ids::Range { base: 0, len: 0 },
        }
    }

    #[must_use]
    /// A list expected to hold `n` ids.
    ///
    /// Only a hint, and deliberately not an allocation: a list that stays a
    /// range never needs one, and that is the common shape. The capacity is
    /// taken at the moment it spills instead, where the final length is still
    /// known to be at least what has been pushed.
    pub fn with_capacity(_n: usize) -> Self {
        Self::new()
    }

    /// Add an id, degrading the representation if it no longer fits.
    ///
    /// A range only survives an id that continues it. The first that does not
    /// spills the run into a vector — once, and only for lists that are not
    /// consecutive.
    pub fn push(
        &mut self,
        id: u64,
    ) {
        match &mut self.ids {
            Ids::Range { base, len } => {
                // In place while the range holds — reassigning the enum here
                // instead cost 79% on the consecutive path.
                if *len == 0 {
                    *base = id;
                    *len = 1;
                    return;
                }
                if id == *base + *len as u64 {
                    *len += 1;
                    return;
                }
                let (base, len) = (*base, *len);
                if id > base + len as u64 {
                    self.ids = Self::widen(base, len, id);
                } else {
                    self.ids = Self::spill(base, len, id);
                }
            }
            Ids::Ascending { bitmap, len, max } => {
                if id > *max {
                    bitmap.insert(id);
                    *len += 1;
                    *max = id;
                } else {
                    // A repeat or a step backwards: no bitmap holds either, so
                    // this is the one place the bitmap is expanded.
                    let mut ids: Vec<u64> = Vec::with_capacity(*len + 1);
                    ids.extend(bitmap.iter());
                    let runs = 1 + ids.windows(2).filter(|w| w[0] + 1 != w[1]).count();
                    let last = *max;
                    ids.push(id);
                    self.ids = Ids::Spilled {
                        ids,
                        ascending: false,
                        runs: runs + 1,
                        max: last,
                    };
                }
            }
            Ids::Spilled {
                ids,
                ascending,
                runs,
                max,
            } => {
                let last = *ids.last().expect("a spilled list is never empty");
                if id != last + 1 {
                    *runs += 1;
                    if id <= last {
                        *ascending = false;
                    }
                }
                if id > *max {
                    *max = id;
                }
                ids.push(id);
            }
        }
    }

    /// Turn the range into a bitmap and add the id that outgrew it.
    ///
    /// Out of line, like [`Self::spill`]. Inlining this into `push` put a
    /// `RoaringTreemap` construction in the hot loop's function body and cost
    /// 20% on the consecutive path, which never reaches it.
    #[inline(never)]
    fn widen(
        base: u64,
        len: usize,
        id: u64,
    ) -> Ids {
        // Still ascending, just no longer contiguous: the run becomes one
        // `insert_range` and the bitmap takes over.
        let mut bitmap = Box::new(RoaringTreemap::new());
        bitmap.insert_range(base..=base + len as u64 - 1);
        bitmap.insert(id);
        Ids::Ascending {
            bitmap,
            len: len + 1,
            max: id,
        }
    }

    /// Materialize `base..base + len` and append the id that broke it.
    ///
    /// The run being expanded is consecutive by construction, so it is exactly
    /// one run and `id` starts a second.
    fn spill(
        base: u64,
        len: usize,
        id: u64,
    ) -> Ids {
        let last = base + len as u64 - 1;
        let mut ids = Vec::with_capacity(len + 1);
        ids.extend(base..=last);
        ids.push(id);
        Ids::Spilled {
            ids,
            ascending: id > last,
            runs: 2,
            max: if id > last { id } else { last },
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match &self.ids {
            Ids::Range { len, .. } | Ids::Ascending { len, .. } => *len,
            Ids::Spilled { ids, .. } => ids.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The ids in order, without materializing a range.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        match &self.ids {
            Ids::Range { base, len } => Either::Left(*base..*base + *len as u64),
            Ids::Ascending { bitmap, .. } => Either::Right(Either::Left(bitmap.iter())),
            Ids::Spilled { ids, .. } => Either::Right(Either::Right(ids.iter().copied())),
        }
    }

    /// The largest id present, or 0 for an empty list — which is what
    /// `width_for` wants anyway.
    ///
    /// Only [`write_forced`] needs this: the encode path reads `max` straight
    /// off the variant that carries it.
    #[cfg(any(test, feature = "test-util"))]
    fn max(&self) -> u64 {
        match &self.ids {
            Ids::Range { base, len } => base.saturating_add(*len as u64).saturating_sub(1),
            Ids::Ascending { max, .. } | Ids::Spilled { max, .. } => *max,
        }
    }

    /// The row count as it goes on the wire.
    ///
    /// Checked rather than cast: a list longer than `u32::MAX` cannot be
    /// encoded, and truncating silently would produce a record whose blocks
    /// disagree with its own count — which the far side reads as corruption
    /// somewhere else entirely.
    #[must_use]
    pub fn count(&self) -> u32 {
        u32::try_from(self.len()).expect("a record cannot carry more than u32::MAX entities")
    }

    /// Whether the whole list is one consecutive run.
    #[must_use]
    pub const fn is_consecutive(&self) -> bool {
        // A range by construction, except that the empty list is also one and
        // has no run to encode.
        matches!(self.ids, Ids::Range { len, .. } if len > 0)
    }

    /// The ids as a slice, materializing a range to produce one.
    ///
    /// Prefer [`Self::iter`]: this exists for the graph's bulk APIs, which take
    /// `&[u64]`, and every call to it undoes the allocation a range avoids.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u64> {
        self.iter().collect()
    }

    /// The ids as a bitmap, built one [`RoaringTreemap::insert_range`] per
    /// **run** rather than one insert per id.
    ///
    /// Only meaningful for an ascending list, which is the only kind the bitmap
    /// encodings are offered to. A run is a single container operation however
    /// many ids it spans, so the consecutive case — every bulk create, every
    /// delete-by-label — costs one call rather than a million.
    #[must_use]
    pub fn to_bitmap(&self) -> RoaringTreemap {
        let (ids, runs) = match &self.ids {
            // One run, so one container operation for the whole list — and no
            // vector to walk to discover that.
            Ids::Range { base, len } => {
                let mut bitmap = RoaringTreemap::new();
                if *len > 0 {
                    bitmap.insert_range(*base..=*base + *len as u64 - 1);
                }
                bitmap.optimize();
                return bitmap;
            }
            // Already one, and built as the ids arrived.
            Ids::Ascending { bitmap, .. } => {
                let mut bitmap = (**bitmap).clone();
                bitmap.optimize();
                return bitmap;
            }
            Ids::Spilled {
                ids,
                ascending,
                runs,
                ..
            } => {
                debug_assert!(*ascending, "a bitmap cannot represent this list");
                (ids, *runs)
            }
        };
        let mut bitmap = RoaringTreemap::new();
        if runs * RUN_BUILD_RATIO <= ids.len() {
            // Few runs, so each `insert_range` covers many ids and the whole
            // bitmap is a handful of container operations. One consecutive run
            // of 10,000 builds in 2.5 us this way against 43 us id-at-a-time.
            let mut i = 0;
            while i < ids.len() {
                let start = ids[i];
                let mut end = start;
                i += 1;
                while i < ids.len() && ids[i] == end + 1 {
                    end = ids[i];
                    i += 1;
                }
                bitmap.insert_range(start..=end);
            }
        } else {
            // Runs all the way down — gap-of-one ids are the worst case, where
            // every id is its own run. `insert_range` then pays its
            // container-splitting logic per id and costs 28x what the bulk
            // sorted build does, so hand the whole list over instead.
            bitmap = RoaringTreemap::from_sorted_iter(ids.iter().copied())
                .expect("the caller has already proved the ids ascending");
        }
        // `optimize()` is **normative**, not a tuning knob: an unoptimized
        // bitmap serializes to different bytes, so two engines that disagree
        // about calling it produce different buffers for the same set.
        bitmap.optimize();
        bitmap
    }

    /// Whether the dictionary is worth building at all.
    ///
    /// It can only beat raw ids when its ranks are strictly narrower than they
    /// are, which takes real duplication. Deciding by size alone cannot see
    /// that: it has to build the dictionary to measure it, and building it is
    /// the whole cost. Measured at 10,000 rows, choosing the dictionary for an
    /// 8% size win cost 2,455x what those bytes are worth on the write thread.
    fn duplication_likely(ids: &[u64]) -> bool {
        let sampled = ids.len().min(DUP_PROBE_SAMPLE);
        if sampled == 0 {
            return false;
        }

        // Stride across the whole list rather than taking a prefix: endpoints
        // often arrive grouped by source, and a prefix would see one group.
        let step = (ids.len() / sampled).max(1);
        let mut sample = [0_u64; DUP_PROBE_SAMPLE];
        for (slot, &id) in sample[..sampled].iter_mut().zip(ids.iter().step_by(step)) {
            *slot = id;
        }

        // Sort and count runs: 128 elements, so this beats the quadratic scan a
        // linear "have I seen it" search would do.
        let sample = &mut sample[..sampled];
        sample.sort_unstable();
        let distinct = 1 + sample.windows(2).filter(|w| w[0] != w[1]).count();

        distinct * 4 < sampled * 3
    }

    /// Write the block: `u8 encoding`, then whatever that encoding needs.
    ///
    /// The representation already answered the first question. A range writes
    /// its base and stops — no vector was ever built and none is scanned. Only
    /// a spilled list reaches the size comparisons below, and it is exactly the
    /// shape that needs them.
    pub fn encode(
        &self,
        buf: &mut Vec<u8>,
    ) {
        // `Range` — the base alone, because the record already carries the
        // count. Nothing per row and, unlike roaring's ~27 byte floor, no
        // floor: this wins from two ids upward and ties plain at one.
        // Sequential id allocation produces exactly this shape.
        if let Ids::Range { base, len } = self.ids
            && len > 0
        {
            let base_width = width_for(base);
            write_u8(buf, BlockEncoding::Range as u8);
            write_u8(buf, base_width);
            write_widths(buf, &[base], base_width);
            return;
        }

        // Ascending with gaps: the bitmap already exists, so the choice is the
        // same one a spilled ascending list faces, made without ever holding the
        // ids. `len` and `max` give `plain_bytes`; the bitmap gives the other
        // side; and either form is writable from it, because an ascending unique
        // sequence round-trips a bitmap exactly.
        //
        // NOTE: this branch has to come before the `let-else` below. Without it
        // an `Ascending` list falls into the empty-list arm and writes a
        // zero-length block — which compiles clean and no test catches, because
        // it only shows up as a replica applying nothing.
        if let Ids::Ascending { bitmap, len, max } = &self.ids {
            let mut bitmap = (**bitmap).clone();
            bitmap.optimize();
            let width = width_for(*max);
            let plain_bytes = 1 + len * width as usize;
            let sorted_bytes = 4 + bitmap.serialized_size();
            if sorted_bytes * SORTED_MIN_RATIO <= plain_bytes {
                write_bitmap(buf, BlockEncoding::Sorted, &bitmap);
            } else {
                write_u8(buf, BlockEncoding::Plain as u8);
                write_u8(buf, width);
                write_widths_iter(buf, bitmap.iter(), *len, width);
            }
            return;
        }

        let Ids::Spilled {
            ids,
            ascending,
            max,
            ..
        } = &self.ids
        else {
            // The empty list: no run to describe, so it takes the plain form
            // with nothing after the width.
            write_plain(buf, &[], 1);
            return;
        };

        let width = width_for(*max);
        let plain_bytes = 1 + ids.len() * width as usize;

        // `Sorted` — a bare bitmap, nothing per row. A unique ascending list
        // gains nothing from the dictionary either — its dictionary is this
        // same bitmap plus a rank per row — so the choice here is between the
        // bitmap and the plain form, and nothing below runs.
        if *ascending {
            let bitmap = self.to_bitmap();
            let sorted_bytes = 4 + bitmap.serialized_size();
            if sorted_bytes * SORTED_MIN_RATIO <= plain_bytes {
                write_bitmap(buf, BlockEncoding::Sorted, &bitmap);
            } else {
                write_plain(buf, ids, width);
            }
            return;
        }

        // `Dictionary` — the distinct ids once, then a rank per row. Probe for
        // duplication before paying for the bitmap, then let the bitmap itself
        // answer whether there was any: a cardinality equal to the row count
        // means every id was distinct, so the ranks cannot be narrower than the
        // ids and the dictionary has already lost.
        if width > 1 && Self::duplication_likely(ids) {
            let mut dict = RoaringTreemap::new();
            dict.extend(ids.iter().copied());
            dict.optimize();
            let cardinality = dict.len();
            if cardinality < ids.len() as u64 {
                let rank_width = width_for(cardinality.saturating_sub(1));
                let dict_bytes = 5 + dict.serialized_size() + ids.len() * rank_width as usize;
                if dict_bytes < plain_bytes {
                    write_bitmap(buf, BlockEncoding::Compressed, &dict);
                    write_u8(buf, rank_width);
                    write_ranks(buf, ids, &dict, rank_width);
                    return;
                }
            }
        }

        write_plain(buf, ids, width);
    }

    /// Read `count` ids back.
    pub fn decode(
        r: &mut Reader<'_>,
        count: u32,
    ) -> Result<Self, DecodeError> {
        Ok(read_ids(r, count)?.into_iter().collect())
    }
}

/// Compare against a plain slice, so assertions and equality checks do not
/// have to build a vector every time.
///
/// No `Deref<Target = [u64]>`: a range has no slice to hand out, and offering
/// one would mean materializing it behind the caller's back — which is the
/// allocation the range exists to avoid. Consumers use [`IdList::iter`], or
/// [`IdList::to_vec`] where a bulk graph API genuinely needs `&[u64]`.
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

/// Yields `u64` by value, not `&u64`: a range has nothing to borrow from.
impl<'a> IntoIterator for &'a IdList {
    type Item = u64;
    type IntoIter = Either<
        std::ops::Range<u64>,
        Either<roaring::treemap::Iter<'a>, std::iter::Copied<std::slice::Iter<'a, u64>>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        match &self.ids {
            Ids::Range { base, len } => Either::Left(*base..*base + *len as u64),
            Ids::Ascending { bitmap, .. } => Either::Right(Either::Left(bitmap.iter())),
            Ids::Spilled { ids, .. } => Either::Right(Either::Right(ids.iter().copied())),
        }
    }
}

impl FromIterator<u64> for IdList {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let it = iter.into_iter();
        let mut out = Self::with_capacity(it.size_hint().0);
        for id in it {
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

/// `u8 encoding`, `u32 length`, then the bitmap serialized **straight into the
/// payload**.
///
/// The length prefix comes from `serialized_size()`, which roaring reports
/// without serializing, so the blob never needs a `Vec` of its own to be
/// measured in. `Vec<u8>` is an `io::Write`, so the bitmap writes itself the
/// rest of the way — one allocation and one copy fewer per block than staging
/// it first.
fn write_bitmap(
    buf: &mut Vec<u8>,
    encoding: BlockEncoding,
    bitmap: &RoaringTreemap,
) {
    let len = bitmap.serialized_size();
    write_u8(buf, encoding as u8);
    write_u32(buf, len as u32);
    // Reserve first. Roaring writes itself in many small pieces, and without
    // this the payload buffer grows underneath it — each realloc copying every
    // byte written so far, not just the blob. Staging into a sized `Vec` used
    // to hide that; measured at 10,000 gapped ids, leaving it out cost 243%.
    buf.reserve(len);
    let before = buf.len();
    bitmap
        .serialize_into(&mut *buf)
        .expect("writing to a Vec cannot fail");
    debug_assert_eq!(
        buf.len() - before,
        len,
        "serialized_size disagreed with serialize_into, so the length prefix lies"
    );
}

fn write_plain(
    buf: &mut Vec<u8>,
    ids: &[u64],
    width: u8,
) {
    write_u8(buf, BlockEncoding::Plain as u8);
    write_u8(buf, width);
    write_widths(buf, ids, width);
}

/// One rank per row, taken from the dictionary itself.
///
/// `rank(v)` counts the elements at or below `v`, so the 0-based index is one
/// less. Reading it off the bitmap means the distinct ids never have to be
/// materialized into a sorted `Vec` just to be searched.
fn write_ranks(
    buf: &mut Vec<u8>,
    ids: &[u64],
    dict: &RoaringTreemap,
    width: u8,
) {
    let ranks: Vec<u64> = ids.iter().map(|&id| dict.rank(id) - 1).collect();
    write_widths(buf, &ranks, width);
}

/// Write `values` at a fixed width.
///
/// The width is matched **once**, not per value: a branch inside the loop stops
/// the compiler vectorizing what is otherwise a straight truncate-and-store.
/// [`write_widths`] over an iterator, for a source that has no slice — so the
/// plain branch of an `Ascending` list does not materialize what the
/// representation exists to avoid.
fn write_widths_iter(
    buf: &mut Vec<u8>,
    values: impl Iterator<Item = u64>,
    len: usize,
    width: u8,
) {
    buf.reserve(len * width as usize);
    match width {
        1 => buf.extend(values.map(|v| v as u8)),
        2 => buf.extend(values.flat_map(|v| (v as u16).to_le_bytes())),
        4 => buf.extend(values.flat_map(|v| (v as u32).to_le_bytes())),
        _ => buf.extend(values.flat_map(u64::to_le_bytes)),
    }
}

fn write_widths(
    buf: &mut Vec<u8>,
    values: &[u64],
    width: u8,
) {
    buf.reserve(values.len() * width as usize);
    match width {
        1 => buf.extend(values.iter().map(|&v| v as u8)),
        2 => buf.extend(values.iter().flat_map(|&v| (v as u16).to_le_bytes())),
        4 => buf.extend(values.iter().flat_map(|&v| (v as u32).to_le_bytes())),
        _ => buf.extend(values.iter().flat_map(|&v| v.to_le_bytes())),
    }
}

/// Read `count` values at a fixed width.
///
/// One bounds check for the whole run and one width match for the whole loop,
/// so each arm is a tight widening map the compiler can vectorize. Doing it per
/// value instead measured 3.5x slower than reading raw `u64`s.
fn read_widths(
    r: &mut Reader<'_>,
    count: usize,
    width: u8,
) -> Result<Vec<u64>, DecodeError> {
    let bytes = r.take(count * width as usize)?;
    let mut out = Vec::with_capacity(count);
    match width {
        1 => out.extend(bytes.iter().map(|&b| u64::from(b))),
        2 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|c| u64::from(u16::from_le_bytes(c.try_into().unwrap()))),
        ),
        4 => out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| u64::from(u32::from_le_bytes(c.try_into().unwrap()))),
        ),
        _ => out.extend(
            bytes
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
        ),
    }
    Ok(out)
}

fn checked_width(r: &mut Reader<'_>) -> Result<u8, DecodeError> {
    let width = r.u8()?;
    if matches!(width, 1 | 2 | 4 | 8) {
        Ok(width)
    } else {
        Err(DecodeError::BadIndexWidth(width))
    }
}

fn read_bitmap(r: &mut Reader<'_>) -> Result<RoaringTreemap, DecodeError> {
    let blob_len = r.u32()?;
    let blob = r.take(blob_len as usize)?;
    RoaringTreemap::deserialize_from(blob).map_err(|e| DecodeError::BadRoaring(e.to_string()))
}

/// The most ids one record may carry, and so the largest `Vec<u64>` a single
/// block can be talked into allocating.
///
/// `Reader::guard_count` bounds a count by weighing it against the bytes left to
/// read, which works whenever each item costs bytes on the wire. Two of the four
/// encodings describe any count in a handful of bytes — `Range` is a base and an
/// implied count, `Sorted` a roaring blob whose run containers hold 2^32 ids in
/// about half a megabyte — so for those the only available bound is an absolute
/// one. Without it a 15-byte buffer claiming `u32::MAX` ids costs ~34 GB to
/// materialize, twice, before anything looks at it.
///
/// `1 << 27` ids is a 1 GiB `Vec<u64>`. The largest record the benchmarks
/// produce is 10^6, so this leaves three orders of magnitude of headroom; a
/// query that genuinely touched 134M entities in one shape would be refused,
/// which is the trade being made deliberately.
pub const MAX_RECORD_IDS: u64 = 1 << 27;

/// The ids of one block, whatever encoding carried them.
pub fn read_ids(
    r: &mut Reader<'_>,
    count: u32,
) -> Result<Vec<u64>, DecodeError> {
    // Before the encoding is even known: every arm below allocates `count`
    // values, and two of them can do it from a payload small enough that
    // `guard_count` sees nothing wrong.
    if u64::from(count) > MAX_RECORD_IDS {
        return Err(DecodeError::TooManyIds {
            count: u64::from(count),
            max: MAX_RECORD_IDS,
        });
    }
    match BlockEncoding::try_from(r.u8()?)? {
        BlockEncoding::Range => {
            let width = checked_width(r)?;
            let base = read_widths(r, 1, width)?[0];
            // `guard_count` weighs a claimed count against the bytes left in
            // the buffer, and a range block is a handful of bytes however large
            // the count — so the bound that matters here is arithmetic, not
            // length.
            let end = base
                .checked_add(u64::from(count))
                .ok_or(DecodeError::BadRange { base, count })?;
            Ok((base..end).collect())
        }
        // The cardinality assertion is the guard that keeps row k bound to the
        // k-th id: a bitmap silently deduplicates, so one id short would shift
        // every later row onto the wrong entity rather than fail.
        BlockEncoding::Sorted => {
            let bitmap = read_bitmap(r)?;
            if bitmap.len() != u64::from(count) {
                return Err(DecodeError::CardinalityMismatch {
                    expected: u64::from(count),
                    actual: bitmap.len(),
                });
            }
            Ok(bitmap.iter().collect())
        }
        BlockEncoding::Compressed => {
            let dict = read_bitmap(r)?;
            let width = checked_width(r)?;
            // Materialize the ascending order once, then every row is one index.
            let ascending: Vec<u64> = dict.iter().collect();
            let cardinality = ascending.len() as u64;
            let n = r.guard_count(u64::from(count), width as usize)?;
            read_widths(r, n, width)?
                .into_iter()
                .map(|rank| {
                    ascending
                        .get(rank as usize)
                        .copied()
                        .ok_or(DecodeError::RankOutOfRange { rank, cardinality })
                })
                .collect()
        }
        BlockEncoding::Plain => {
            let width = checked_width(r)?;
            let n = r.guard_count(u64::from(count), width as usize)?;
            read_widths(r, n, width)
        }
    }
}

/// Force one encoding, for tests and sizing comparisons.
///
/// Exhaustive on purpose: an `else` here once meant that asking for an encoding
/// this function did not know about silently produced a plain block, which in a
/// sizing helper reads as "plain won".
#[cfg(any(test, feature = "test-util"))]
pub fn write_forced(
    buf: &mut Vec<u8>,
    ids: &IdList,
    encoding: BlockEncoding,
) {
    match encoding {
        BlockEncoding::Compressed => {
            // A sizing helper, so it materializes without apology: forcing an
            // encoding is a benchmark's job, not the emitter's.
            let raw = ids.to_vec();
            let mut dict = RoaringTreemap::new();
            dict.extend(raw.iter().copied());
            dict.optimize();
            let width = width_for(dict.len().saturating_sub(1));
            write_bitmap(buf, BlockEncoding::Compressed, &dict);
            write_u8(buf, width);
            write_ranks(buf, &raw, &dict, width);
        }
        BlockEncoding::Sorted => write_bitmap(buf, BlockEncoding::Sorted, &ids.to_bitmap()),
        BlockEncoding::Range => {
            debug_assert!(ids.is_consecutive(), "Range needs a consecutive run");
            let base = ids.iter().next().unwrap_or(0);
            let width = width_for(base);
            write_u8(buf, BlockEncoding::Range as u8);
            write_u8(buf, width);
            write_widths(buf, &[base], width);
        }
        BlockEncoding::Plain => write_plain(buf, &ids.to_vec(), width_for(ids.max())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::testing::hex;

    /// 10,000 ids in two runs: the shape the bitmap is actually good at.
    fn run_structured() -> Vec<u64> {
        (0..5_000).chain(100_000..105_000).collect()
    }

    #[test]
    fn a_singleton_is_a_range_of_one() {
        // A single id is trivially consecutive, and Range costs exactly what the
        // plain form would — encoding, width, value — so nothing is lost by
        // taking it. Roaring, at a ~27 byte floor, was never in the running.
        let mut buf = Vec::new();
        IdList::from([0]).encode(&mut buf);
        assert_eq!(hex(&buf), "03 01 00");

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 1).unwrap(), vec![0]);
        assert!(r.is_empty());
    }

    #[test]
    fn a_range_is_three_bytes_at_any_count() {
        // The property that makes Range worth having: no floor, and flat in the
        // count. A million consecutive ids cost the same as ten.
        for n in [2_u32, 10, 1_000, 1_000_000] {
            let ids: Vec<u64> = (0..u64::from(n)).collect();
            let mut buf = Vec::new();
            IdList::from(ids.as_slice()).encode(&mut buf);
            assert_eq!(hex(&buf), "03 01 00", "n = {n}");

            let mut r = Reader::new(&buf);
            assert_eq!(read_ids(&mut r, n).unwrap(), ids, "n = {n}");
        }
    }

    #[test]
    fn a_range_that_would_wrap_is_rejected() {
        let mut buf = Vec::new();
        write_u8(&mut buf, BlockEncoding::Range as u8);
        write_u8(&mut buf, 8);
        buf.extend_from_slice(&(u64::MAX - 1).to_le_bytes());
        let mut r = Reader::new(&buf);
        assert!(matches!(
            read_ids(&mut r, 100),
            Err(DecodeError::BadRange { .. })
        ));
    }

    #[test]
    fn the_ladder_prefers_range_then_bitmap_then_plain() {
        // Four shapes, four encodings, and the sizes are why. A gap of one
        // drops out of Range; a transposition drops out of Sorted; and a merely
        // *denser* bitmap does not clear the ratio gate, so gapped-by-two lands
        // on Plain despite the bitmap being smaller.
        let consecutive: Vec<u64> = (0..10_000).collect();
        let runs = run_structured();
        let gapped: Vec<u64> = (0..10_000).map(|i| i * 2).collect();
        let mut shuffled = gapped.clone();
        shuffled.swap(0, 9_999);

        let mut a = Vec::new();
        IdList::from(consecutive.as_slice()).encode(&mut a);
        let mut b = Vec::new();
        IdList::from(runs.as_slice()).encode(&mut b);
        let mut c = Vec::new();
        IdList::from(gapped.as_slice()).encode(&mut c);
        let mut d = Vec::new();
        IdList::from(shuffled.as_slice()).encode(&mut d);

        assert_eq!(a[0], BlockEncoding::Range as u8, "consecutive");
        assert_eq!(b[0], BlockEncoding::Sorted as u8, "two runs");
        assert_eq!(
            c[0],
            BlockEncoding::Plain as u8,
            "gapped: bitmap too marginal"
        );
        assert_eq!(d[0], BlockEncoding::Plain as u8, "unsorted");

        assert_eq!(a.len(), 3);
        assert!(
            b.len() < 100,
            "two runs should be tens of bytes, got {}",
            b.len()
        );
        assert_eq!(c.len(), 20_002);
        assert_eq!(d.len(), 20_002);

        for (buf, want) in [
            (&a, &consecutive),
            (&b, &runs),
            (&c, &gapped),
            (&d, &shuffled),
        ] {
            let mut r = Reader::new(buf);
            assert_eq!(&read_ids(&mut r, 10_000).unwrap(), want);
        }
    }

    #[test]
    fn a_marginally_smaller_bitmap_is_refused() {
        // The gate. Gapped-by-two makes the bitmap genuinely smaller — 8,225
        // against 20,002 — and it is still refused, because 2.4x does not repay
        // ~7.4 ns per id. Below the gate the bitmap is a denser array, not
        // structure.
        let gapped: Vec<u64> = (0..10_000).map(|i| i * 2).collect();
        let mut forced = Vec::new();
        write_forced(
            &mut forced,
            &IdList::from(gapped.as_slice()),
            BlockEncoding::Sorted,
        );
        let mut chosen = Vec::new();
        IdList::from(gapped.as_slice()).encode(&mut chosen);

        assert!(forced.len() < chosen.len(), "the bitmap really is smaller");
        assert!(
            chosen.len() < forced.len() * SORTED_MIN_RATIO,
            "and it is refused for being under {SORTED_MIN_RATIO}x"
        );
        assert_eq!(chosen[0], BlockEncoding::Plain as u8);
    }

    #[test]
    fn sorted_is_refused_when_the_ids_arrive_unsorted() {
        // The rule that keeps enc 2 safe: the same ids, permuted, must not be
        // silently re-sorted — row k is bound to the k-th id *as written*.
        let ids: Vec<u64> = (0..10_000).collect();
        let mut shuffled = ids.clone();
        shuffled.swap(0, 9_999);

        let mut buf = Vec::new();
        IdList::from(shuffled.as_slice()).encode(&mut buf);
        assert_ne!(
            buf[0],
            BlockEncoding::Sorted as u8,
            "one transposition must disqualify the bitmap"
        );

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 10_000).unwrap(), shuffled);
    }

    #[test]
    fn sorted_is_refused_when_an_id_repeats() {
        // Ascending but not unique: a bitmap would drop the duplicate and shift
        // every later row by one.
        let ids: Vec<u64> = vec![1, 2, 2, 3];
        let mut buf = Vec::new();
        IdList::from(ids.as_slice()).encode(&mut buf);
        assert_ne!(buf[0], BlockEncoding::Sorted as u8);

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 4).unwrap(), ids);
    }

    #[test]
    fn sorted_wide_spread_stays_plain() {
        // The case that makes a count threshold unsound: at stride 2^32 roaring
        // costs ~22 B per id, so the plain form wins at *every* count.
        let ids: Vec<u64> = (0..1_024).map(|i| i << 32).collect();
        let mut buf = Vec::new();
        IdList::from(ids.as_slice()).encode(&mut buf);
        assert_eq!(buf[0], BlockEncoding::Plain as u8);
        assert_eq!(buf[1], 8, "ids past 2^32 need the full width");
        assert_eq!(buf.len(), 2 + 1_024 * 8);

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 1_024).unwrap(), ids);
    }

    #[test]
    fn sorted_roundtrips_across_the_crossover() {
        for n in [1_u64, 2, 4, 8, 63, 64, 65, 1_000, 4_096] {
            let ids: Vec<u64> = (0..n).collect();
            let mut buf = Vec::new();
            IdList::from(ids.as_slice()).encode(&mut buf);
            let mut r = Reader::new(&buf);
            assert_eq!(read_ids(&mut r, n as u32).unwrap(), ids, "n = {n}");
            assert!(r.is_empty(), "n = {n} left {} bytes", r.remaining());
        }
    }

    #[test]
    fn sorted_rejects_a_cardinality_lie() {
        // The guard that stops a short bitmap from shifting every row onto the
        // wrong entity. Run-structured, so the bitmap clears the ratio gate and
        // there is a cardinality to lie about.
        let ids = run_structured();
        let mut buf = Vec::new();
        IdList::from(ids.as_slice()).encode(&mut buf);
        assert_eq!(buf[0], BlockEncoding::Sorted as u8);

        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, 9_999),
            Err(DecodeError::CardinalityMismatch {
                expected: 9_999,
                actual: 10_000,
            })
        );
    }

    // ── IdList ──

    #[test]
    fn id_list_supernode_uses_the_dictionary() {
        // 10,000 edges from 3 sources in a graph big enough that raw ids need 4
        // bytes each. Heavy duplication plus wide ids is the dictionary's case:
        // the ranks collapse to one byte and the dictionary holds three values.
        let sources = [4_000_000_000_u64, 7, 2_100_000_000];
        let src: Vec<u64> = (0..10_000).map(|i| sources[i % 3]).collect();

        let mut buf = Vec::new();
        IdList::from(src.as_slice()).encode(&mut buf);
        assert_eq!(
            buf[0],
            BlockEncoding::Compressed as u8,
            "the dictionary should win here"
        );

        let mut narrow = Vec::new();
        write_forced(
            &mut narrow,
            &IdList::from(src.as_slice()),
            BlockEncoding::Plain,
        );
        assert!(
            buf.len() < narrow.len(),
            "dict {} should beat narrow {}",
            buf.len(),
            narrow.len()
        );

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 10_000).unwrap(), src);
        assert!(r.is_empty());
    }

    #[test]
    fn id_list_dictionary_loses_without_duplication() {
        // The case the dictionary was assumed to own and does not: 10,000
        // distinct dense ids. It costs a sort, a bitmap and a rank per row and
        // is still the largest of the three. These ids are ascending but only
        // gap-of-six dense, so the bitmap does not clear the ratio gate either
        // — the narrowed plain form wins outright.
        let src: Vec<u64> = (0..10_000).map(|i| (i * 6) % 60_000).collect();
        let mut buf = Vec::new();
        IdList::from(src.as_slice()).encode(&mut buf);
        assert_eq!(buf[0], BlockEncoding::Plain as u8);

        let mut dict = Vec::new();
        write_forced(
            &mut dict,
            &IdList::from(src.as_slice()),
            BlockEncoding::Compressed,
        );
        assert!(buf.len() < dict.len(), "{} vs {}", buf.len(), dict.len());

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 10_000).unwrap(), src);
    }

    #[test]
    fn id_list_unsorted_dense_ids_stay_plain() {
        // The same ids in a different order cannot use the bitmap, and with no
        // duplication the dictionary is not worth its sort either — so the
        // probe sends this straight to the narrowed plain form. Narrowing to 2
        // bytes costs one pass for the maximum and nothing else.
        let mut src: Vec<u64> = (0..10_000).map(|i| (i * 6) % 60_000).collect();
        src.reverse();
        let mut buf = Vec::new();
        IdList::from(src.as_slice()).encode(&mut buf);
        assert_eq!(buf[0], BlockEncoding::Plain as u8);
        assert_eq!(buf[1], 2, "59,994 fits two bytes");

        let mut dict = Vec::new();
        write_forced(
            &mut dict,
            &IdList::from(src.as_slice()),
            BlockEncoding::Compressed,
        );
        assert!(buf.len() < dict.len(), "{} vs {}", buf.len(), dict.len());

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 10_000).unwrap(), src);
    }

    #[test]
    fn id_list_wide_spread_stays_plain() {
        // Where the dictionary does lose: ids spread a bucket apart make the
        // bitmap itself larger than the ids it replaces, exactly as for IdSet.
        let src: Vec<u64> = (0..1_000).map(|i| i << 32).collect();
        let mut buf = Vec::new();
        IdList::from(src.as_slice()).encode(&mut buf);
        assert_eq!(buf[0], BlockEncoding::Plain as u8);
        assert_eq!(buf[1], 8, "ids past 2^32 need the full width");
        assert_eq!(buf.len(), 2 + 1_000 * 8);

        let mut r = Reader::new(&buf);
        assert_eq!(read_ids(&mut r, 1_000).unwrap(), src);
    }

    #[test]
    fn id_list_preserves_duplicates_and_order() {
        // The property that separates IdList from IdSet.
        let src = [9_u64, 1, 9, 9, 1, 5];
        let mut buf = Vec::new();
        IdList::from(src.as_slice()).encode(&mut buf);
        let mut r = Reader::new(&buf);
        let out = read_ids(&mut r, 6).unwrap();
        assert_eq!(out, src);
        assert_ne!(out, vec![1, 5, 9], "an IdSet would have produced this");
    }

    /// `rows` endpoints drawn from `distinct` sources inside a graph of
    /// `nodes` nodes. FalkorDB allocates node ids densely from zero and reuses
    /// them from a free list, so the largest id tracks the graph's size — which
    /// is what decides whether narrowing alone already captures the saving.
    fn endpoints(
        rows: usize,
        distinct: usize,
        nodes: u64,
    ) -> Vec<u64> {
        (0..rows)
            .map(|i| {
                let h = (i as u64).wrapping_mul(0x9E37_7997_9F4A_7C15);
                (h % distinct as u64) * (nodes / distinct as u64).max(1)
            })
            .collect()
    }

    #[test]
    fn id_list_selection_crossover() {
        // Which of the three encodings the size rule picks, and why. The
        // dictionary only wins where duplication is heavy *and* the ids are
        // wide enough that narrowing has not already taken the saving.
        println!("\n   nodes   rows distinct      raw     dict  picked");
        for (nodes, distinct) in [
            (10_000_u64, 1_usize),
            (10_000, 100),
            (10_000, 1_000),
            (10_000, 10_000),
            (5_000_000, 100),
            (5_000_000, 10_000),
        ] {
            let rows = 10_000;
            let ids = endpoints(rows, distinct, nodes);
            let mut p = Vec::new();
            write_forced(&mut p, &IdList::from(ids.as_slice()), BlockEncoding::Plain);
            let mut d = Vec::new();
            write_forced(
                &mut d,
                &IdList::from(ids.as_slice()),
                BlockEncoding::Compressed,
            );
            let mut auto = Vec::new();
            IdList::from(ids.as_slice()).encode(&mut auto);

            let picked = if auto[0] == BlockEncoding::Compressed as u8 {
                "dict"
            } else {
                "raw"
            };
            let given_up = auto.len() as i64 - d.len() as i64;
            println!(
                "{nodes:8} {rows:6} {distinct:8} {:8} {:8}  {picked:<5}{}",
                p.len(),
                d.len(),
                if given_up > 0 {
                    format!("  (+{given_up} B, dictionary declined by the probe)")
                } else {
                    String::new()
                }
            );

            // The rule is *not* "smallest of the three wins": the duplication
            // probe declines to even build a dictionary that would be only
            // marginally smaller, because building it is the expensive part.
            // What must hold is that the two cheap encodings are always beaten.
            assert!(auto.len() <= p.len());
            // And when the dictionary is chosen, it is genuinely the smallest.
            if auto[0] == BlockEncoding::Compressed as u8 {
                assert_eq!(auto.len(), d.len().min(p.len()));
            }
            // and whatever was picked must round-trip
            let mut r = Reader::new(&auto);
            assert_eq!(read_ids(&mut r, rows as u32).unwrap(), ids);
        }
    }

    #[test]
    fn id_list_index_width_boundaries() {
        // One width function for both jobs (#13): a dictionary of n entries
        // addresses 0..n-1, so its ranks are `width_for(n - 1)`.
        assert_eq!(width_for(1 - 1), 1);
        assert_eq!(width_for(0x100 - 1), 1);
        assert_eq!(width_for(0x101 - 1), 2);
        assert_eq!(width_for(0x1_0000 - 1), 2);
        assert_eq!(width_for(0x1_0001 - 1), 4);
        assert_eq!(width_for(0x1_0000_0000 - 1), 4);
        assert_eq!(width_for(0x1_0000_0001 - 1), 8);
    }

    #[test]
    fn a_consecutive_list_never_allocates() {
        // The representation *is* the encoding decision, so a list that stays a
        // range holds no vector at all — the point of the whole arrangement.
        // Sequential id allocation makes this the common shape: at a million
        // ids the vector this replaces was 8 MB, built only to prove at encode
        // time that it held one run.
        let mut list = IdList::new();
        for id in 100..1_100 {
            list.push(id);
        }
        assert!(
            matches!(
                list.ids,
                Ids::Range {
                    base: 100,
                    len: 1000
                }
            ),
            "a consecutive push sequence must stay a range"
        );
        assert!(list.is_consecutive());
        assert_eq!(list.len(), 1000);
        assert_eq!(list.iter().next(), Some(100));
        assert_eq!(list.iter().last(), Some(1099));

        // A gap that still ascends degrades one rung, to the bitmap — the run
        // becomes a single `insert_range`, and no vector appears here either.
        list.push(2_000);
        let Ids::Ascending { len, max, .. } = list.ids else {
            panic!("an ascending gap must reach the bitmap, not a vector");
        };
        assert_eq!((len, max), (1001, 2_000));
        assert_eq!(list.iter().last(), Some(2_000));
        assert_eq!(list.len(), 1001);
    }

    #[test]
    fn the_representation_degrades_one_rung_at_a_time() {
        // Range while consecutive, bitmap once gapped, vector once neither —
        // and never back up, because a push cannot un-break an invariant.
        let mut list = IdList::new();
        list.push(10);
        list.push(11);
        assert!(matches!(list.ids, Ids::Range { .. }));
        list.push(20);
        assert!(matches!(list.ids, Ids::Ascending { .. }));
        list.push(20); // a repeat: no bitmap can hold it
        assert!(matches!(
            list.ids,
            Ids::Spilled {
                ascending: false,
                ..
            }
        ));
        assert_eq!(list, [10, 11, 20, 20]);
    }

    #[test]
    fn an_out_of_order_push_spills_and_stops_ascending() {
        let mut list = IdList::new();
        for id in [5, 6, 7] {
            list.push(id);
        }
        assert!(matches!(list.ids, Ids::Range { .. }));
        list.push(2); // backwards: no bitmap can hold this order
        assert!(matches!(
            list.ids,
            Ids::Spilled {
                ascending: false,
                ..
            }
        ));
        assert_eq!(list, [5, 6, 7, 2]);
    }

    #[test]
    fn a_range_and_a_spilled_list_holding_the_same_ids_are_equal() {
        // Equality is about the sequence, not the representation — which of the
        // two you get depends only on the order the pushes arrived in.
        let range: IdList = (0..4).collect();
        let mut spilled = IdList::new();
        for id in [3, 0, 1, 2] {
            spilled.push(id);
        }
        assert!(matches!(range.ids, Ids::Range { .. }));
        assert!(matches!(spilled.ids, Ids::Spilled { .. }));
        assert_ne!(range, spilled, "different order, different list");

        let rebuilt: IdList = range.iter().collect();
        assert_eq!(range, rebuilt);
    }

    #[test]
    fn probe_sizes() {
        println!(
            "  size_of Ids={} IdList={} Roaring={} Box={}",
            std::mem::size_of::<Ids>(),
            std::mem::size_of::<IdList>(),
            std::mem::size_of::<RoaringTreemap>(),
            std::mem::size_of::<Box<RoaringTreemap>>(),
        );
    }

    #[test]
    fn id_list_rejects_an_out_of_range_rank() {
        let src = [5_u64; 10];
        let mut buf = Vec::new();
        write_forced(
            &mut buf,
            &IdList::from(src.as_slice()),
            BlockEncoding::Compressed,
        );
        assert_eq!(buf[0], BlockEncoding::Compressed as u8);
        // The dictionary holds one entry, so rank 1 is out of range.
        *buf.last_mut().unwrap() = 1;
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_ids(&mut r, 10),
            Err(DecodeError::RankOutOfRange {
                rank: 1,
                cardinality: 1,
            })
        );
    }

    // ── the other three blocks ──
}
