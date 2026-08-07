//! The leaf page: a tagged, self-describing byte blob in one of three in-RAM encodings (see [`Leaf`]).

use std::sync::Arc;

use super::{FIELD, STRIDE};

mod aos;
mod compact;
mod compact_indexed;

pub(super) use aos::AosLeaf;
use compact::{BODY_OFFSET, CompactLeaf, packing_fits};
use compact_indexed::CompactIndexedLeaf;

/// The encoding of a leaf page, carried **out of band**.
///
/// The tag lives in the [`Leaf`] enum discriminant in RAM, and the pairing tag
/// in [`CowBTree::leaves`]. The byte buffers themselves are **tag-free** — pure
/// data — so a tree may hold a mix of both formats, each read by its own
/// variant type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LeafFormat {
    /// `[(key:8, doc:8) × n]` — contiguous array-of-structs tuples, 16 B/entry (see [`AosLeaf`]).
    Aos,
    /// `[header][values][index?][docs]` — see [`CompactLeaf::build`].
    Compact,
}

/// Compact is chosen only when it saves at least this many bytes **per entry** vs the AoS form's 16. At 8
/// (half an entry) it captures the large wins — low-cardinality / narrow data compacts to ~5 B/entry (a
/// ~3× memory cut) — while leaving near-incompressible data as AoS. Reads are format-independent (the
/// cursor caches the layout and dispatches widths to fixed loads), so this is purely a *build-cost vs size*
/// trade: a swept micro-benchmark showed compact's build is ~2.5× AoS on a big win but ~5× on a marginal
/// one, so this floor keeps the cheap-to-build big wins and skips the expensive-to-build marginal ones.
const COMPACT_MIN_SAVING_BPE: usize = 8;

/// Fewest **power-of-two** bytes (1, 2, 4, or 8) that hold `x`.
const fn pow2_bytes_for(x: u64) -> usize {
    match x {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0xFFFF_FFFF => 4,
        _ => 8,
    }
}

/// Merge two **sorted** `(key, doc)` sequences into one new `Vec`, dropping exact duplicates (so a tuple
/// present in both — or repeated within either — appears once). One allocation, `O(a.len() + b.len())`, no
/// sort: the callers always hold two already-sorted runs (a leaf's entries and a sorted batch), which a
/// two-pointer merge handles far faster than concatenate-and-sort.
fn merge_sorted(
    lhs: &[(u64, u64)],
    rhs: &[(u64, u64)],
) -> Vec<(u64, u64)> {
    let mut out = Vec::with_capacity(lhs.len() + rhs.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < lhs.len() || j < rhs.len() {
        let take_lhs = i < lhs.len() && (j >= rhs.len() || lhs[i] <= rhs[j]);
        let next = if take_lhs {
            let v = lhs[i];
            i += 1;
            v
        } else {
            let v = rhs[j];
            j += 1;
            v
        };
        if out.last() == Some(&next) {
            continue; // collapse an exact `(key, doc)` duplicate
        }
        out.push(next);
    }
    out
}

/// Index of the first position in `range` for which `predicate` is false (the standard binary-search
/// lower bound). `predicate` must be monotone over `range` (true… then false…).
fn partition_point(
    range: std::ops::Range<usize>,
    mut predicate: impl FnMut(usize) -> bool,
) -> usize {
    let (mut lo, mut hi) = (range.start, range.end);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if predicate(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Two-pointer merge of a leaf's `count` entries (read via `leaf_key`/`leaf_doc`, ascending) with the
/// sorted `batch`, dropping exact `(key, doc)` duplicates, calling `emit(key, doc, leaf_index)` once per
/// output entry in order. `leaf_index` is `Some(i)` for a leaf entry (so the caller can copy its packed
/// cell verbatim) or `None` for a batch entry.
fn merge_walk(
    count: usize,
    leaf_key: impl Fn(usize) -> u64,
    leaf_doc: impl Fn(usize) -> u64,
    batch: &[(u64, u64)],
    mut emit: impl FnMut(u64, u64, Option<usize>),
) {
    let (mut leaf_i, mut batch_j) = (0usize, 0usize);
    let mut last: Option<(u64, u64)> = None;
    while leaf_i < count || batch_j < batch.len() {
        let leaf_entry = (leaf_i < count).then(|| (leaf_key(leaf_i), leaf_doc(leaf_i)));
        let take_leaf = match (leaf_entry, batch.get(batch_j)) {
            (Some(entry), Some(batch_entry)) => entry <= *batch_entry,
            (Some(_), None) => true,
            (None, _) => false,
        };
        let (key, doc, leaf_index) = if take_leaf {
            let index = leaf_i;
            leaf_i += 1;
            let entry = leaf_entry.unwrap();
            (entry.0, entry.1, Some(index))
        } else {
            let entry = batch[batch_j];
            batch_j += 1;
            (entry.0, entry.1, None)
        };
        if last == Some((key, doc)) {
            continue;
        }
        last = Some((key, doc));
        emit(key, doc, leaf_index);
    }
}

/// A leaf page, one of three in-RAM encodings — each variant owns its own **tag-free** `Arc<[u8]>` page and
/// reads it directly (the compact scalar fields are cheap byte loads, read from the buffer where needed
/// rather than cached in a parsed header). The format is carried out of band (this enum's discriminant in
/// RAM, see [`LeafFormat`]):
///
/// - [`AosLeaf`] — `[(key, doc) × n]`, each field a little-endian `u64` (16 B/entry). Key beside its doc
///   (array-of-structs) so a range scan reads one sequential stream.
/// - [`CompactLeaf`] — compact **without** a dedup index (`distinct_count == entry_count`, so every value is
///   distinct): values delta-encoded from the leaf minimum at a per-leaf power-of-two width, one delta per
///   entry, then docs at their own minimal width.
/// - [`CompactIndexedLeaf`] — compact **with** a dedup index (`distinct_count < entry_count`): a distinct
///   value column plus a 1-byte-per-entry index into it, then docs. Chosen when de-duplicating pays.
///
/// Compact (either flavour) is chosen per leaf only when it saves ≥ 8 B/entry (half an AoS entry).
///
/// Either way the blob *is* the leaf's serialized form: cloning is an `Arc::clone` and re-adopting bytes
/// is [`Leaf::from_parts`] — a copy, never a (de)serialization. These newtypes are the one place that knows
/// the byte layout; the enum dispatches to them, so a tree may freely mix the formats.
#[derive(Clone)]
pub(super) enum Leaf<const LEAF_MAX: usize> {
    Aos(AosLeaf),
    Compact(CompactLeaf),
    CompactIndexed(CompactIndexedLeaf),
}

/// Outcome of inserting into a single leaf: the replacement leaf if it still fits, or the two halves plus
/// their separator if it overflowed and split. (`None` from [`Leaf::insert`] means the tuple was present.)
pub(super) enum LeafInsert<const LEAF_MAX: usize> {
    Fit(Leaf<LEAF_MAX>),
    Split {
        left: Leaf<LEAF_MAX>,
        sep: (u64, u64),
        right: Leaf<LEAF_MAX>,
    },
}

impl<const LEAF_MAX: usize> Leaf<LEAF_MAX> {
    /// This leaf's serialized encoding (carried out of band, see [`LeafFormat`]). The two compact in-RAM
    /// types share one on-disk format — index presence lives in the buffer — so both map to `Compact`.
    /// Test-only for now (paired with [`CowBTree::leaves`]); the durable-write path re-exposes it on disk.
    #[cfg(test)]
    pub(super) const fn format(&self) -> LeafFormat {
        match self {
            Self::Aos(_) => LeafFormat::Aos,
            Self::Compact(_) | Self::CompactIndexed(_) => LeafFormat::Compact,
        }
    }

    /// Number of `(key, doc)` entries.
    pub(super) fn count(&self) -> usize {
        match self {
            Self::Aos(l) => l.count(),
            Self::Compact(l) => l.count(),
            Self::CompactIndexed(l) => l.count(),
        }
    }

    /// Key of entry `i`.
    pub(super) fn key(
        &self,
        i: usize,
    ) -> u64 {
        match self {
            Self::Aos(l) => l.key(i),
            Self::Compact(l) => l.key(i),
            Self::CompactIndexed(l) => l.key(i),
        }
    }

    /// Doc of entry `i`.
    pub(super) fn doc(
        &self,
        i: usize,
    ) -> u64 {
        match self {
            Self::Aos(l) => l.doc(i),
            Self::Compact(l) => l.doc(i),
            Self::CompactIndexed(l) => l.doc(i),
        }
    }

    /// The doc array's `(base, stride, width)` — read once per leaf by the cursor's per-entry doc reads.
    pub(super) fn doc_layout(&self) -> (usize, usize, usize) {
        match self {
            Self::Aos(_) => (FIELD, STRIDE, FIELD),
            Self::Compact(l) => l.doc_layout(),
            Self::CompactIndexed(l) => l.doc_layout(),
        }
    }

    /// First entry index whose key is `>= key` (binary search via [`Leaf::key`]). Used by the cursor to
    /// seek a leaf's range start and by the byte round-trip test.
    pub(super) fn lower_bound(
        &self,
        key: u64,
    ) -> usize {
        partition_point(0..self.count(), |i| self.key(i) < key)
    }

    /// Iterate the leaf's `(key, doc)` pairs in stored order. Used by `to_pairs` (mutation rebuilds) and
    /// tests — a cold path; the range cursor has its own decode. Re-derives offsets per entry (no cached
    /// scan state), which is fine off the hot path and lets one method serve all three formats.
    fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.count()).map(move |i| (self.key(i), self.doc(i)))
    }

    /// Decode to the owned `(key, doc)` pairs the mutation paths (insert/remove/merge) work in.
    pub(super) fn to_pairs(&self) -> Vec<(u64, u64)> {
        self.iter().collect()
    }

    /// Build a leaf from sorted `(key, doc)` pairs, choosing [`LeafFormat::Aos`] or [`LeafFormat::Compact`]
    /// per the data.
    ///
    /// The choice is a closed-form size comparison; the widths are always one of {1, 2, 4, 8}, so the only
    /// free variables are `count` and the data:
    /// - AoS:                `count·STRIDE`
    /// - compact, no-dedup:  `BODY_OFFSET + count·value_width + count·doc_width`
    /// - compact, dedup:     `BODY_OFFSET + distinct·value_width + count (index) + count·doc_width`
    ///
    /// Compact is used only when it saves at least [`COMPACT_MIN_SAVING_BPE`] bytes/entry, so AoS — simpler
    /// and cheaper to build — keeps the near-incompressible cases. An empty slice yields an empty AoS leaf;
    /// re-selecting on every rebuild is what migrates a leaf's format as its data changes.
    ///
    /// `value_width` comes from the value range, which is O(1): entries are sorted by `(key, doc)`, so the
    /// min/max values are the first/last keys. `doc_width` needs `max_doc`, which is *not* O(1) — docs are
    /// ordered only within a value, so the global max can sit under any key — so the single pass below (which
    /// also counts distinct values for the dedup decision) is unavoidable. The distinct table itself is built
    /// lazily in [`CompactLeaf::build`], only when compact is chosen.
    pub(super) fn from_pairs(pairs: &[(u64, u64)]) -> Self {
        let count = pairs.len();
        if count == 0 {
            return Self::Aos(AosLeaf::build(pairs));
        }
        // One pass for the two data-dependent inputs — the distinct-value count (runs, since sorted) and the
        // max doc. (The value range needs no scan; it's read O(1) from the sorted ends just below.)
        // Docs are sorted within a value, so a value's max doc is the last entry of its run: fold it in at
        // each run boundary (`window[0]`, the entry just before the key changes), then once more for the
        // final run, which has no boundary after it. Cheaper than `max`-ing every entry on low-card data.
        let mut distinct_count = 1usize;
        let mut max_doc = 0;
        for window in pairs.windows(2) {
            if window[0].0 != window[1].0 {
                distinct_count += 1;
                max_doc = max_doc.max(window[0].1); // window[0] = the ending value's last (= max) doc
            }
        }
        max_doc = max_doc.max(pairs[count - 1].1); // the final run has no boundary; fold in its last entry
        let min_value = pairs[0].0;
        let value_width = pow2_bytes_for(pairs[count - 1].0 - min_value);
        let doc_width = pow2_bytes_for(max_doc);
        let deduplicated = distinct_count < count;
        let compact_size = BODY_OFFSET
            + distinct_count * value_width
            + if deduplicated { count } else { 0 }
            + count * doc_width;
        let aos_size = count * STRIDE;
        // Pick compact only when it saves at least COMPACT_MIN_SAVING_BPE bytes *per entry* over AoS — the
        // floor (see the const) skips marginal wins whose extra build cost isn't worth the memory saved.
        if compact_size + COMPACT_MIN_SAVING_BPE * count <= aos_size {
            if deduplicated {
                Self::CompactIndexed(CompactIndexedLeaf::build(
                    pairs,
                    count,
                    distinct_count,
                    min_value,
                    value_width,
                    doc_width,
                ))
            } else {
                Self::Compact(CompactLeaf::build(
                    pairs,
                    count,
                    min_value,
                    value_width,
                    doc_width,
                ))
            }
        } else {
            Self::Aos(AosLeaf::build(pairs))
        }
    }

    /// Re-adopt a leaf from its format and serialized bytes — the counterpart of [`Leaf::format`] /
    /// [`Leaf::bytes`] and the read side of [`CowBTree::leaves`]. The bytes are the page verbatim, so this
    /// is a copy, never a decode. It **trusts** its input (validating arbitrary bytes is the caller's
    /// responsibility); currently exercised only by the byte round-trip test.
    #[allow(dead_code)]
    pub(super) fn from_parts(
        format: LeafFormat,
        bytes: Arc<[u8]>,
    ) -> Self {
        match format {
            LeafFormat::Aos => Self::Aos(AosLeaf(bytes)),
            // The compact format carries index presence in the buffer (it is not part of [`LeafFormat`]):
            // a dedup index is present iff there are fewer distinct values than entries (see
            // [`compact::is_indexed`]), so pick the in-RAM type from the buffer alone.
            LeafFormat::Compact => {
                if compact::is_indexed(&bytes) {
                    Self::CompactIndexed(CompactIndexedLeaf(bytes))
                } else {
                    Self::Compact(CompactLeaf(bytes))
                }
            }
        }
    }

    /// The leaf's serialized (tag-free) bytes — an `Arc` bump, no serialization step. Test-only for now
    /// (paired with [`CowBTree::leaves`]); the durable-write path re-exposes it when disk lands.
    #[cfg(test)]
    pub(super) fn bytes(&self) -> Arc<[u8]> {
        match self {
            Self::Aos(l) => Arc::clone(&l.0),
            Self::Compact(l) => Arc::clone(&l.0),
            Self::CompactIndexed(l) => Arc::clone(&l.0),
        }
    }

    /// The raw page bytes (for the cursor's cached doc read).
    pub(super) fn raw(&self) -> &[u8] {
        match self {
            Self::Aos(l) => &l.0,
            Self::Compact(l) => &l.0,
            Self::CompactIndexed(l) => &l.0,
        }
    }

    /// First entry whose `(key, doc)` is `>= (key, doc)` — the slot a single insert/remove targets.
    fn lower_bound_entry(
        &self,
        key: u64,
        doc: u64,
    ) -> usize {
        partition_point(0..self.count(), |i| (self.key(i), self.doc(i)) < (key, doc))
    }

    /// Insert one `(key, doc)`: the replacement leaf if it still fits, a [`LeafInsert::Split`] if it
    /// overflowed, or `None` if the tuple is already present. An [`AosLeaf`] that still fits is **spliced
    /// directly in its bytes** — no decode/re-encode. The overflow case and every [`CompactLeaf`] go
    /// through [`Leaf::to_pairs`] + [`Leaf::from_pairs`], which also re-selects the format (so an
    /// AoS→compact migration happens at the next split, not on every insert).
    pub(super) fn insert(
        &self,
        key: u64,
        doc: u64,
    ) -> Option<LeafInsert<LEAF_MAX>> {
        let count = self.count();
        let pos = self.lower_bound_entry(key, doc);
        if pos < count && self.key(pos) == key && self.doc(pos) == doc {
            return None; // already present
        }
        if count < LEAF_MAX {
            // In-place splice (no decode), each variant keeping its own format. `count < LEAF_MAX` ⇒
            // `count + 1 <= LEAF_MAX`, so the result still fits one page.
            match self {
                Self::Aos(aos) => {
                    let cut = pos * STRIDE; // memcpy prefix + tuple + suffix; data at byte 0 (tag-free)
                    let mut buf = Vec::with_capacity(aos.0.len() + STRIDE);
                    buf.extend_from_slice(&aos.0[..cut]);
                    buf.extend_from_slice(&key.to_le_bytes());
                    buf.extend_from_slice(&doc.to_le_bytes());
                    buf.extend_from_slice(&aos.0[cut..]);
                    return Some(LeafInsert::Fit(Self::Aos(AosLeaf(Arc::from(
                        buf.as_slice(),
                    )))));
                }
                Self::Compact(l) if packing_fits(&l.0, key, doc) => {
                    return Some(LeafInsert::Fit(Self::Compact(
                        l.splice_insert(key, doc, pos),
                    )));
                }
                Self::CompactIndexed(l) if packing_fits(&l.0, key, doc) => {
                    return Some(LeafInsert::Fit(Self::CompactIndexed(
                        l.splice_insert(key, doc, pos),
                    )));
                }
                // A compact insert that would widen a packing parameter (new min / wider delta or doc)
                // falls through to the rebuild, which re-picks the widths and format.
                _ => {}
            }
        }
        // Overflow, or a compact insert that widens a parameter: decode, insert, rebuild (re-selecting the
        // format and splitting if it no longer fits one page). `pos` is the same index in the sorted decode.
        let mut pairs = self.to_pairs();
        pairs.insert(pos, (key, doc));
        if pairs.len() <= LEAF_MAX {
            Some(LeafInsert::Fit(Self::from_pairs(&pairs)))
        } else {
            let mid = pairs.len() / 2;
            Some(LeafInsert::Split {
                left: Self::from_pairs(&pairs[..mid]),
                sep: pairs[mid],
                right: Self::from_pairs(&pairs[mid..]),
            })
        }
    }

    /// Remove one `(key, doc)`: the replacement leaf plus whether it underflowed (`< LEAF_MAX / 2`), or `None`
    /// if the tuple is absent. An [`AosLeaf`] is **spliced directly** (the 16-byte tuple is cut out); a
    /// [`CompactLeaf`] is rebuilt via [`Leaf::from_pairs`]. Re-compacting an AoS leaf happens at its next
    /// merge (which rebuilds through [`Leaf::from_pairs`]).
    pub(super) fn remove(
        &self,
        key: u64,
        doc: u64,
    ) -> Option<(Self, bool)> {
        let count = self.count();
        let pos = self.lower_bound_entry(key, doc);
        if pos >= count || self.key(pos) != key || self.doc(pos) != doc {
            return None; // absent
        }
        let new_count = count - 1;
        let leaf = match self {
            Self::Aos(aos) => {
                let cut = pos * STRIDE; // data at byte 0 (tag-free)
                let mut buf = Vec::with_capacity(aos.0.len() - STRIDE);
                buf.extend_from_slice(&aos.0[..cut]);
                buf.extend_from_slice(&aos.0[cut + STRIDE..]);
                Self::Aos(AosLeaf(Arc::from(buf.as_slice())))
            }
            // Cut the entry in place (no decode). Indexed leaves only when the result stays validly indexed
            // (`distinct_count < new_count`); a now-emptied leaf or an indexed leaf that would no longer
            // satisfy that falls to the rebuild below (canonical empty page / re-selects the format).
            Self::Compact(l) if new_count > 0 => Self::Compact(l.splice_remove(pos)),
            Self::CompactIndexed(l) if l.distinct_count() < new_count => {
                Self::CompactIndexed(l.splice_remove(pos))
            }
            _ => {
                let mut pairs = self.to_pairs();
                pairs.remove(pos);
                Self::from_pairs(&pairs)
            }
        };
        Some((leaf, new_count < LEAF_MAX / 2))
    }

    /// Apply a sorted `batch` (all of it routing into this leaf) and return the replacement leaf page(s).
    /// An [`AosLeaf`] whose result still fits one page is **byte-merged** (see [`AosLeaf::merge_batch`]);
    /// a [`CompactLeaf`], or one whose result overflows into several pages, decodes, merges, de-dups, and
    /// re-chunks through [`Leaf::from_pairs`] (re-selecting the format per chunk).
    pub(super) fn merge_batch(
        &self,
        batch: &[(u64, u64)],
    ) -> Vec<Self> {
        // Fast path: the whole batch fits one page and never widens a packing parameter — digest it into the
        // packed bytes (existing cells copied verbatim, only batch entries encoded), keeping the format.
        if self.count() + batch.len() <= LEAF_MAX {
            match self {
                Self::Aos(aos) => return vec![Self::Aos(aos.merge_batch(batch))],
                Self::Compact(l) if batch.iter().all(|&(k, d)| packing_fits(&l.0, k, d)) => {
                    return vec![Self::Compact(l.merge(batch))];
                }
                Self::CompactIndexed(l) if batch.iter().all(|&(k, d)| packing_fits(&l.0, k, d)) => {
                    // If every batch key is *already* a distinct value, block-copy the leaf's index + doc
                    // cells between the (galloped) insert positions — no index remap, and the gallop makes
                    // this never worse than the merge-walk at any batch size (so no size guard). A batch that
                    // introduces a new distinct value needs an index remap, so it takes the merge-walk.
                    if batch.iter().all(|&(k, _)| l.distinct_slot(k).is_ok()) {
                        return vec![Self::CompactIndexed(l.block_copy_merge(batch))];
                    }
                    return vec![Self::CompactIndexed(l.merge(batch))];
                }
                _ => {}
            }
        }
        // Overflow, or a batch that widens a parameter: decode, merge, re-chunk through `from_pairs`. The
        // leaf's entries and the batch are each already sorted, so two-pointer **merge** them (with exact-dup
        // removal) rather than concat + sort — a swept micro-benchmark put this ~3.5x ahead of `sort()` and
        // ~5x ahead of `sort_unstable()` (pdqsort can't exploit pre-sorted runs). Then re-chunk.
        let merged = merge_sorted(&self.to_pairs(), batch);
        merged.chunks(LEAF_MAX).map(Self::from_pairs).collect()
    }
}
