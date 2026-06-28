//! A copy-on-write B⁺-tree of compact `(key, doc)` tuple pages.
//!
//! This is the in-RAM core structure for the native numeric index.
//! It is a persistent, copy-on-write B⁺-tree (in the immutable-snapshot sense), specialised for the index's
//! needs:
//!
//! - **Compact tuples.** Each entry is a `(key: u64, doc: u64)` pair — a sortable encoding of the
//!   indexed value plus the entity (node or edge) id. Entries are kept sorted by `(key, doc)`.
//! - **Page-level copy-on-write MVCC.** A *snapshot* is [`CowBTree::clone`] — an `O(1)` `Arc` bump
//!   on the root that shares every page. A write **path-copies** only the nodes from the root to the
//!   touched leaf (sharing all untouched pages), so a reader holding an older root is never disturbed.
//!   This gives cheap snapshots with **one `Arc` per page** (not per node), which is both lean in
//!   memory (≤ 16 B/entry — compact pages are smaller) and a constant-time clone.
//! - **Leaf pages are plain byte blobs.** A leaf is a *tagged* little-endian byte blob in an `Arc<[u8]>`
//!   — either contiguous `(key, doc)` tuples (16 B/entry) or a compact form (per-leaf value deltas +
//!   de-duplication + minimal doc width), chosen per leaf by its data (see [`Leaf`]). The bytes are read
//!   in place and copy to / from a byte buffer verbatim, so the in-memory form *is* the serialized form —
//!   there is no separate serialize/deserialize step.
//! - **No bloat under churn.** Deletes are applied in place within a copied leaf; an underflowing leaf
//!   is merged with a neighbour (combine-and-re-split, with hysteresis). There is no tombstone backlog
//!   and no flush threshold, so a delete+reinsert workload stays compact without a background thread.
//!
//! Reads return a **lazy cursor [`RangeIter`]** that owns a clone of the root (a self-contained
//! snapshot) and yields matching **doc ids** one at a time — droppable mid-scan, matching the
//! pull-based query runtime.
//!
//! Bulk loads use [`CowBTree::from_sorted`] / [`CowBTree::insert_batch`] (pack pages bottom-up, one
//! page copy per touched node); single [`CowBTree::insert`] / [`CowBTree::remove`] are the
//! steady-state path.

use std::sync::Arc;

/// Max **entries** (`(key, doc)` tuples) per leaf page; split on overflow, merge below `LEAF_MIN`. Leaf
/// balance is by entry *count*, not byte size — the same bound for both leaf encodings (see [`Leaf`]), so a
/// page holds ≤ 256 entries whether they encode to ≈ 4 KiB ([`AosLeaf`], 16 B/tuple) or less ([`CompactLeaf`]).
///
/// `256` is also load-bearing for the compact format: its dedup index stores one slot per entry as a `u8`,
/// which only fits because `count ≤ LEAF_MAX ≤ 256` (see [`CompactLeaf::build`]). **Raising this past 256
/// would overflow that index** — the compact `as u8` cast would have to widen first.
const LEAF_MAX: usize = 256;
const LEAF_MIN: usize = LEAF_MAX / 2;
/// Max children per branch page (fan-out). Split on overflow, merge below `BRANCH_MIN`.
const BRANCH_MAX: usize = 256;
const BRANCH_MIN: usize = BRANCH_MAX / 2;

/// A tree node: either a leaf (a byte blob of sorted tuples) or an internal branch.
///
/// Both variants are `Arc`-wrapped, so cloning a node — and therefore a whole tree version — is an
/// `O(1)` reference-count bump that shares all underlying pages.
#[derive(Clone)]
enum Node {
    /// A leaf page of sorted `(key, doc)` tuples — see [`Leaf`].
    Leaf(Leaf),
    Branch(Arc<Branch>),
}

/// An internal node: separator keys plus child pointers. `seps[i]` is the minimum `(key, doc)` of
/// `children[i + 1]`, so `seps.len() == children.len() - 1`.
#[derive(Clone)]
struct Branch {
    seps: Vec<(u64, u64)>,
    children: Vec<Node>,
}

// ---- the leaf page: a tagged, self-describing byte blob (one of two encodings) ----------------

/// Byte width of one `u64` field (a key or a doc).
const FIELD: usize = std::mem::size_of::<u64>();
/// Byte stride of one `(key, doc)` entry in the [`AosLeaf`] layout.
const STRIDE: usize = 2 * FIELD;

/// The encoding of a leaf page, carried **out of band** (the [`Leaf`] enum discriminant in RAM, and the
/// pairing tag in [`CowBTree::leaves`]). The byte buffers themselves are **tag-free** — pure data — so a
/// tree may hold a mix of both formats, each read by its own variant type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LeafFormat {
    /// `[(key:8, doc:8) × n]` — contiguous array-of-structs tuples, 16 B/entry (see [`AosLeaf`]).
    Aos,
    /// `[header][values][index?][docs]` — see [`CompactLeaf::build`].
    Compact,
}

// Byte offsets of each field within the (tag-free) compact header.
const ENTRY_COUNT_OFFSET: usize = 0; // u16: number of entries
const MIN_VALUE_OFFSET: usize = 2; // u64: minimum value in the leaf (values are stored as deltas from it)
const VALUE_WIDTH_OFFSET: usize = 10; // u8: bytes per value delta (a power of two)
const DISTINCT_COUNT_OFFSET: usize = 11; // u16: number of distinct values
const DOC_WIDTH_OFFSET: usize = 13; // u8: bytes per doc (a power of two)
const BODY_OFFSET: usize = 14; // first byte of the value / index / doc bodies

/// Compact is chosen only when it saves at least this many bytes **per entry** vs the AoS form's 16. At 8
/// (half an entry) it captures the large wins — low-cardinality / narrow data compacts to ~5 B/entry (a
/// ~3× memory cut) — while leaving near-incompressible data as AoS. Reads are format-independent (the
/// cursor caches the layout and dispatches widths to fixed loads), so this is purely a *build-cost vs size*
/// trade: a swept micro-benchmark showed compact's build is ~2.5× AoS on a big win but ~5× on a marginal
/// one, so this floor keeps the cheap-to-build big wins and skips the expensive-to-build marginal ones.
const COMPACT_MIN_SAVING_BPE: usize = 8;

/// Read the little-endian `u64` at byte offset `off`. The `unwrap` is infallible: `b[off..off + FIELD]`
/// is always exactly `FIELD` bytes; an out-of-bounds `off` means a malformed page — a build bug, since the
/// build path always produces well-formed pages, so a panic is correct.
fn read_u64(
    b: &[u8],
    off: usize,
) -> u64 {
    u64::from_le_bytes(b[off..off + FIELD].try_into().unwrap())
}

/// Read the little-endian `u16` at byte offset `off`.
fn read_u16(
    b: &[u8],
    off: usize,
) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

/// Read a little-endian unsigned integer of `width` ∈ {1, 2, 4, 8} bytes at `off`, zero-extended to a
/// `u64`. Widths are powers of two so every read is a single fixed-size load, never a variable-length copy.
fn read_width(
    b: &[u8],
    off: usize,
    width: usize,
) -> u64 {
    match width {
        1 => b[off] as u64,
        2 => u16::from_le_bytes(b[off..off + 2].try_into().unwrap()) as u64,
        4 => u32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as u64,
        _ => {
            debug_assert_eq!(width, 8);
            u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
        }
    }
}

/// Fewest **power-of-two** bytes (1, 2, 4, or 8) that hold `x`.
fn pow2_bytes_for(x: u64) -> usize {
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
    a: &[(u64, u64)],
    b: &[(u64, u64)],
) -> Vec<(u64, u64)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() || j < b.len() {
        let take_a = i < a.len() && (j >= b.len() || a[i] <= b[j]);
        let next = if take_a {
            let v = a[i];
            i += 1;
            v
        } else {
            let v = b[j];
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

/// Galloping (exponential) lower bound: the first index in `[start, count)` where `predicate` is false,
/// found by probing outward from `start` (1, 2, 4, … entries) until it overshoots, then binary-searching
/// the final interval. Cost is `O(log(gap))` in the distance to the answer, so merging a sorted run by
/// calling this once per run entry (each resuming from the previous answer) is `O(B·log(N/B))` — it adapts:
/// cheap when run entries are sparse, and degrading to a linear scan (no worse than a merge-walk) when they
/// are dense. `predicate` must be monotone over `[start, count)` (true… then false…).
fn gallop_lower_bound(
    start: usize,
    count: usize,
    mut predicate: impl FnMut(usize) -> bool,
) -> usize {
    let mut step = 1;
    while start + step < count && predicate(start + step) {
        step *= 2;
    }
    // The answer is in `[start + step/2, min(start + step, count))`: `predicate` held at the previous probe
    // (`start + step/2`, or `start` itself) and fails at `start + step` (or that probe ran past `count`).
    partition_point(
        (start + step / 2)..(start + step).min(count),
        &mut predicate,
    )
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
enum Leaf {
    Aos(AosLeaf),
    Compact(CompactLeaf),
    CompactIndexed(CompactIndexedLeaf),
}

/// An array-of-structs leaf page: a tag-free `Arc<[u8]>` of `[(key:8, doc:8) × n]`, data at byte 0.
#[derive(Clone)]
struct AosLeaf(Arc<[u8]>);

/// A compact leaf page **without** a dedup index: a tag-free `Arc<[u8]>` of `[header][values][docs]`, with
/// one value delta per entry (`distinct_count == entry_count`). See [`CompactLeaf::build`].
#[derive(Clone)]
struct CompactLeaf(Arc<[u8]>);

/// A compact leaf page **with** a dedup index: a tag-free `Arc<[u8]>` of `[header][distinct values][index][docs]`,
/// where `index` is one `u8` per entry pointing into the distinct value column (`distinct_count < entry_count`).
/// See [`CompactIndexedLeaf::build`].
#[derive(Clone)]
struct CompactIndexedLeaf(Arc<[u8]>);

impl AosLeaf {
    /// Number of `(key, doc)` entries — the buffer is tag-free `[(key, doc) × n]`, so `len / STRIDE`.
    fn count(&self) -> usize {
        self.0.len() / STRIDE
    }

    /// Read the little-endian `u64` field at entry `i`'s offset `off_in_entry` (0 for the key, `FIELD`
    /// for the doc). Data is at byte 0 (the buffer is tag-free), so the absolute offset is
    /// `STRIDE * i + off_in_entry`.
    fn read(
        &self,
        i: usize,
        off_in_entry: usize,
    ) -> u64 {
        read_u64(&self.0, STRIDE * i + off_in_entry)
    }

    /// Key of entry `i`.
    fn key(
        &self,
        i: usize,
    ) -> u64 {
        self.read(i, 0)
    }

    /// Doc of entry `i`.
    fn doc(
        &self,
        i: usize,
    ) -> u64 {
        self.read(i, FIELD)
    }

    /// Encode sorted `pairs` into the tag-free `[(key, doc) × n]` layout.
    fn build(pairs: &[(u64, u64)]) -> Self {
        let mut buf = Vec::with_capacity(pairs.len() * STRIDE);
        for &(key, doc) in pairs {
            buf.extend_from_slice(&key.to_le_bytes());
            buf.extend_from_slice(&doc.to_le_bytes());
        }
        Self(Arc::from(buf.as_slice()))
    }
}

/// Write the shared compact header — `count`, `min`, `value_width`, `distinct_count`, `doc_width` — at the
/// offsets above, then assert the body starts at [`BODY_OFFSET`]. Both compact builders open with this, so
/// the read offsets (read straight from the buffer by each variant) and the write order stay in lock-step.
fn write_compact_header(
    buf: &mut Vec<u8>,
    count: usize,
    min_value: u64,
    value_width: usize,
    distinct_count: usize,
    doc_width: usize,
) {
    buf.extend_from_slice(&(count as u16).to_le_bytes());
    buf.extend_from_slice(&min_value.to_le_bytes());
    buf.push(value_width as u8);
    buf.extend_from_slice(&(distinct_count as u16).to_le_bytes());
    buf.push(doc_width as u8);
    debug_assert_eq!(buf.len(), BODY_OFFSET);
}

impl CompactLeaf {
    /// Number of `(key, doc)` entries — read from the header.
    fn count(&self) -> usize {
        read_u16(&self.0, ENTRY_COUNT_OFFSET) as usize
    }

    /// Key of entry `i` — `min + delta[i]` (one delta per entry; no index).
    fn key(
        &self,
        i: usize,
    ) -> u64 {
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        read_u64(&self.0, MIN_VALUE_OFFSET) + read_width(&self.0, BODY_OFFSET + i * vw, vw)
    }

    /// Doc of entry `i`.
    fn doc(
        &self,
        i: usize,
    ) -> u64 {
        let (base, stride, width) = self.doc_layout();
        read_width(&self.0, base + i * stride, width)
    }

    /// The doc array's `(base, stride, width)` — read once per leaf by the cursor. The docs start right
    /// after the `count` value deltas. Docs are stored contiguously, so the stride equals the width.
    fn doc_layout(&self) -> (usize, usize, usize) {
        let count = self.count();
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        let dw = self.0[DOC_WIDTH_OFFSET] as usize;
        let docs_offset = BODY_OFFSET + count * vw;
        (docs_offset, dw, dw)
    }

    /// Encode the no-index compact layout `[header][values][docs]` (the caller has already chosen it and
    /// computed the widths): one value delta per entry — no distinct table, no index. `distinct_count`
    /// equals `count` here and is still written to the header (so a reader can tell index-vs-no-index).
    fn build(
        pairs: &[(u64, u64)],
        count: usize,
        min_value: u64,
        value_width: usize,
        doc_width: usize,
    ) -> Self {
        let mut buf = Vec::with_capacity(BODY_OFFSET + count * value_width + count * doc_width);
        // No dedup ⇒ distinct_count == count.
        write_compact_header(&mut buf, count, min_value, value_width, count, doc_width);
        for &(key, _) in pairs {
            buf.extend_from_slice(&(key - min_value).to_le_bytes()[..value_width]);
        }
        for &(_, doc) in pairs {
            buf.extend_from_slice(&doc.to_le_bytes()[..doc_width]);
        }
        Self(Arc::from(buf.as_slice()))
    }
}

impl CompactIndexedLeaf {
    /// Number of `(key, doc)` entries — read from the header.
    fn count(&self) -> usize {
        read_u16(&self.0, ENTRY_COUNT_OFFSET) as usize
    }

    /// Byte offset of the per-entry distinct-value index (right after the `distinct` value column).
    fn index_offset(&self) -> usize {
        let distinct = read_u16(&self.0, DISTINCT_COUNT_OFFSET) as usize;
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        BODY_OFFSET + distinct * vw
    }

    /// Key of entry `i` — `min + distinct[index[i]]`.
    fn key(
        &self,
        i: usize,
    ) -> u64 {
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        let slot = self.0[self.index_offset() + i] as usize;
        read_u64(&self.0, MIN_VALUE_OFFSET) + read_width(&self.0, BODY_OFFSET + slot * vw, vw)
    }

    /// Doc of entry `i`.
    fn doc(
        &self,
        i: usize,
    ) -> u64 {
        let (base, stride, width) = self.doc_layout();
        read_width(&self.0, base + i * stride, width)
    }

    /// The doc array's `(base, stride, width)` — read once per leaf by the cursor. The docs start right
    /// after the `count`-byte index. Docs are stored contiguously, so the stride equals the width.
    fn doc_layout(&self) -> (usize, usize, usize) {
        let dw = self.0[DOC_WIDTH_OFFSET] as usize;
        let docs_offset = self.index_offset() + self.count();
        (docs_offset, dw, dw)
    }

    /// Encode the de-duplicated compact layout `[header][distinct values][index][docs]` (the caller has
    /// already chosen it and computed the widths). The distinct table and per-entry index are (re)built
    /// here in one in-cache pass over the ≤ `LEAF_MAX` entries — deliberately *not* carried over from
    /// [`Leaf::from_pairs`], since materialising them there measurably slowed the far more common AoS path
    /// while saving nothing here (the byte writes dominate).
    fn build(
        pairs: &[(u64, u64)],
        count: usize,
        distinct_count: usize,
        min_value: u64,
        value_width: usize,
        doc_width: usize,
    ) -> Self {
        let mut distinct: Vec<u64> = Vec::with_capacity(distinct_count);
        let mut index: Vec<u8> = Vec::with_capacity(count);
        for &(key, _) in pairs {
            if distinct.last() != Some(&key) {
                distinct.push(key);
            }
            // The index slot fits a `u8` only because dedup ⇒ distinct_count < count, and count ≤
            // `LEAF_MAX` (= 256) ⇒ slot ≤ 254. `LEAF_MAX`'s doc notes this ceiling must not exceed 256.
            index.push((distinct.len() - 1) as u8);
        }
        debug_assert_eq!(distinct.len(), distinct_count);
        debug_assert!(distinct_count < count);
        let mut buf = Vec::with_capacity(
            BODY_OFFSET + distinct_count * value_width + count + count * doc_width,
        );
        write_compact_header(
            &mut buf,
            count,
            min_value,
            value_width,
            distinct_count,
            doc_width,
        );
        for &value in &distinct {
            buf.extend_from_slice(&(value - min_value).to_le_bytes()[..value_width]);
        }
        buf.extend_from_slice(&index);
        for &(_, doc) in pairs {
            buf.extend_from_slice(&doc.to_le_bytes()[..doc_width]);
        }
        Self(Arc::from(buf.as_slice()))
    }
}

/// Parsed layout of a compact page (either format), read once from the header for a mutation. A transient
/// write-path helper — offsets only, not cached, and separate from the read-path decode (the cursor and
/// iterators read what they need inline). It removes the repeated header arithmetic from the splice/merge
/// methods below.
struct CompactLayout {
    count: usize,
    min: u64,
    value_width: usize,
    distinct_count: usize,
    doc_width: usize,
    /// First byte of the per-entry index; equals `docs_offset` for a no-index page (which has no index).
    index_offset: usize,
    /// First byte of the doc column.
    docs_offset: usize,
}

impl CompactLayout {
    fn read(bytes: &[u8]) -> Self {
        let count = read_u16(bytes, ENTRY_COUNT_OFFSET) as usize;
        let distinct_count = read_u16(bytes, DISTINCT_COUNT_OFFSET) as usize;
        let value_width = bytes[VALUE_WIDTH_OFFSET] as usize;
        let index_offset = BODY_OFFSET + distinct_count * value_width;
        // No-index pages (distinct_count == count) store no index, so docs follow the values directly;
        // indexed pages have one index byte per entry between the distinct values and the docs.
        let docs_offset = index_offset + if distinct_count < count { count } else { 0 };
        Self {
            count,
            min: read_u64(bytes, MIN_VALUE_OFFSET),
            value_width,
            distinct_count,
            doc_width: bytes[DOC_WIDTH_OFFSET] as usize,
            index_offset,
            docs_offset,
        }
    }
}

/// True when `(key, doc)` fits the page's current packing without widening — `key >= min`, the value delta
/// fits `value_width`, and `doc` fits `doc_width`. The precondition for an in-place splice/merge (below)
/// instead of a rebuild that would have to re-pick the widths. Shared by both compact formats (their
/// packing parameters live at the same header offsets).
fn packing_fits(
    bytes: &[u8],
    key: u64,
    doc: u64,
) -> bool {
    let min = read_u64(bytes, MIN_VALUE_OFFSET);
    key >= min
        && pow2_bytes_for(key - min) <= bytes[VALUE_WIDTH_OFFSET] as usize
        && pow2_bytes_for(doc) <= bytes[DOC_WIDTH_OFFSET] as usize
}

/// Append the doc column for an insert that adds one entry at `pos`: `docs[0..pos] | doc | docs[pos..count]`,
/// each existing cell copied verbatim from `bytes` at `layout.docs_offset` / `doc_width`. Shared by all three
/// insert arms (no-index, indexed existing-distinct, indexed new-distinct), which differ only above the docs.
fn append_spliced_docs(
    buf: &mut Vec<u8>,
    bytes: &[u8],
    layout: &CompactLayout,
    pos: usize,
    doc: u64,
) {
    let (docs_offset, doc_width, count) = (layout.docs_offset, layout.doc_width, layout.count);
    buf.extend_from_slice(&bytes[docs_offset..docs_offset + pos * doc_width]);
    buf.extend_from_slice(&doc.to_le_bytes()[..doc_width]);
    buf.extend_from_slice(&bytes[docs_offset + pos * doc_width..docs_offset + count * doc_width]);
}

impl CompactLeaf {
    /// Splice one `(key, doc)` into the packed page at entry `pos` — no decode. The caller guarantees
    /// [`packing_fits`], `count < LEAF_MAX`, and that the tuple is absent. A no-index leaf stores one
    /// value per entry, so the value and doc cells are simply inserted (`memcpy` of the surrounding cells);
    /// the leaf stays no-index (a value that collides with an existing one is stored verbatim, and is
    /// de-duplicated only at the next rebuild).
    fn splice_insert(
        &self,
        key: u64,
        doc: u64,
        pos: usize,
    ) -> CompactLeaf {
        let bytes = &self.0;
        let layout = CompactLayout::read(bytes);
        let count = layout.count;
        let min = layout.min;
        let value_width = layout.value_width;
        let doc_width = layout.doc_width;
        // no-index ⇒ docs_offset == BODY_OFFSET + count * value_width
        let docs_offset = layout.docs_offset;
        let new_count = count + 1;
        let mut buf =
            Vec::with_capacity(BODY_OFFSET + new_count * value_width + new_count * doc_width);
        // no-index ⇒ distinct_count == entry count
        write_compact_header(&mut buf, new_count, min, value_width, new_count, doc_width);
        // value column: prefix | new delta | suffix
        buf.extend_from_slice(&bytes[BODY_OFFSET..BODY_OFFSET + pos * value_width]);
        buf.extend_from_slice(&(key - min).to_le_bytes()[..value_width]);
        buf.extend_from_slice(&bytes[BODY_OFFSET + pos * value_width..docs_offset]);
        // doc column: prefix | new doc | suffix
        append_spliced_docs(&mut buf, bytes, &layout, pos, doc);
        CompactLeaf(Arc::from(buf.as_slice()))
    }

    /// Cut entry `pos` out of the packed page — no decode. The caller guarantees the tuple is present at
    /// `pos` and that `count > 1` (an emptied leaf is rebuilt by the caller for a canonical empty page).
    fn splice_remove(
        &self,
        pos: usize,
    ) -> CompactLeaf {
        let bytes = &self.0;
        let layout = CompactLayout::read(bytes);
        let count = layout.count;
        let min = layout.min;
        let value_width = layout.value_width;
        let doc_width = layout.doc_width;
        // no-index ⇒ docs_offset == BODY_OFFSET + count * value_width
        let docs_offset = layout.docs_offset;
        let new_count = count - 1;
        let mut buf =
            Vec::with_capacity(BODY_OFFSET + new_count * value_width + new_count * doc_width);
        write_compact_header(&mut buf, new_count, min, value_width, new_count, doc_width);
        // value column: prefix | (drop pos) | suffix
        buf.extend_from_slice(&bytes[BODY_OFFSET..BODY_OFFSET + pos * value_width]);
        buf.extend_from_slice(&bytes[BODY_OFFSET + (pos + 1) * value_width..docs_offset]);
        // doc column: prefix | (drop pos) | suffix
        buf.extend_from_slice(&bytes[docs_offset..docs_offset + pos * doc_width]);
        buf.extend_from_slice(
            &bytes[docs_offset + (pos + 1) * doc_width..docs_offset + count * doc_width],
        );
        CompactLeaf(Arc::from(buf.as_slice()))
    }

    /// Merge a sorted `batch` into this no-index page — no `Vec<(u64, u64)>`, no sort. The caller guarantees
    /// every batch entry [`packing_fits`] and `count + batch.len() <= LEAF_MAX`. Existing value/doc
    /// cells are copied verbatim; only batch entries are encoded; exact `(key, doc)` duplicates are dropped.
    fn merge(
        &self,
        batch: &[(u64, u64)],
    ) -> CompactLeaf {
        let bytes = &self.0;
        let layout = CompactLayout::read(bytes);
        let count = layout.count;
        let min = layout.min;
        let value_width = layout.value_width;
        let doc_width = layout.doc_width;
        let docs_offset = layout.docs_offset;
        let leaf_key =
            |i: usize| min + read_width(bytes, BODY_OFFSET + i * value_width, value_width);
        let leaf_doc = |i: usize| read_width(bytes, docs_offset + i * doc_width, doc_width);
        let cap = count + batch.len();
        let mut values: Vec<u8> = Vec::with_capacity(cap * value_width);
        let mut docs: Vec<u8> = Vec::with_capacity(cap * doc_width);
        // merge-walk: two-pointer over leaf entries + batch, dropping exact (key, doc) dups
        merge_walk(
            count,
            leaf_key,
            leaf_doc,
            batch,
            |key, doc, leaf_i| match leaf_i {
                // leaf entry: copy its value/doc cells verbatim
                Some(vi) => {
                    let value_off = BODY_OFFSET + vi * value_width;
                    let doc_off = docs_offset + vi * doc_width;
                    values.extend_from_slice(&bytes[value_off..value_off + value_width]);
                    docs.extend_from_slice(&bytes[doc_off..doc_off + doc_width]);
                }
                // batch entry: encode the new value delta and doc
                None => {
                    values.extend_from_slice(&(key - min).to_le_bytes()[..value_width]);
                    docs.extend_from_slice(&doc.to_le_bytes()[..doc_width]);
                }
            },
        );
        let new_count = values.len() / value_width;
        let mut buf = Vec::with_capacity(BODY_OFFSET + values.len() + docs.len());
        write_compact_header(&mut buf, new_count, min, value_width, new_count, doc_width);
        buf.extend_from_slice(&values);
        buf.extend_from_slice(&docs);
        CompactLeaf(Arc::from(buf.as_slice()))
    }
}

impl CompactIndexedLeaf {
    fn distinct_count(&self) -> usize {
        read_u16(&self.0, DISTINCT_COUNT_OFFSET) as usize
    }

    /// The slot of `key` in the (sorted) distinct value column: `Ok(slot)` if it is already a distinct
    /// value, `Err(insertion_slot)` if it is new.
    fn distinct_slot(
        &self,
        key: u64,
    ) -> Result<usize, usize> {
        let bytes = &self.0;
        let min = read_u64(bytes, MIN_VALUE_OFFSET);
        let value_width = bytes[VALUE_WIDTH_OFFSET] as usize;
        let distinct_count = read_u16(bytes, DISTINCT_COUNT_OFFSET) as usize;
        let distinct_value =
            |s: usize| min + read_width(bytes, BODY_OFFSET + s * value_width, value_width);
        let slot = partition_point(0..distinct_count, |s| distinct_value(s) < key);
        if slot < distinct_count && distinct_value(slot) == key {
            Ok(slot)
        } else {
            Err(slot)
        }
    }

    /// Splice one `(key, doc)` into the packed page at entry `pos` — no decode. The caller guarantees
    /// [`packing_fits`], `count < LEAF_MAX`, and that the tuple is absent. If `key` is already a
    /// distinct value the distinct table is copied verbatim and only an index byte + doc cell are inserted;
    /// otherwise the delta is inserted into the (sorted) distinct table at slot `slot` and every existing index
    /// byte `>= slot` is bumped by one. The leaf stays indexed (old distinct < old count guarantees it).
    fn splice_insert(
        &self,
        key: u64,
        doc: u64,
        pos: usize,
    ) -> CompactIndexedLeaf {
        let bytes = &self.0;
        let layout = CompactLayout::read(bytes);
        let count = layout.count;
        let min = layout.min;
        let value_width = layout.value_width;
        let distinct_count = layout.distinct_count;
        let doc_width = layout.doc_width;
        let index_offset = layout.index_offset;
        let new_count = count + 1;
        match self.distinct_slot(key) {
            Ok(slot) => {
                // existing distinct value: distinct table unchanged; only an index byte + doc cell are added
                let mut buf = Vec::with_capacity(
                    BODY_OFFSET + distinct_count * value_width + new_count + new_count * doc_width,
                );
                write_compact_header(
                    &mut buf,
                    new_count,
                    min,
                    value_width,
                    distinct_count,
                    doc_width,
                );
                // distinct table verbatim
                buf.extend_from_slice(&bytes[BODY_OFFSET..index_offset]);
                // index: prefix | new slot at pos | suffix
                buf.extend_from_slice(&bytes[index_offset..index_offset + pos]);
                buf.push(slot as u8);
                buf.extend_from_slice(&bytes[index_offset + pos..index_offset + count]);
                // doc column: prefix | new doc | suffix
                append_spliced_docs(&mut buf, bytes, &layout, pos, doc);
                CompactIndexedLeaf(Arc::from(buf.as_slice()))
            }
            Err(slot) => {
                // new distinct value at `slot`: grow the distinct table and remap the index
                let new_distinct_count = distinct_count + 1;
                let mut buf = Vec::with_capacity(
                    BODY_OFFSET
                        + new_distinct_count * value_width
                        + new_count
                        + new_count * doc_width,
                );
                write_compact_header(
                    &mut buf,
                    new_count,
                    min,
                    value_width,
                    new_distinct_count,
                    doc_width,
                );
                // distinct table: prefix | new delta at slot | suffix
                buf.extend_from_slice(&bytes[BODY_OFFSET..BODY_OFFSET + slot * value_width]);
                buf.extend_from_slice(&(key - min).to_le_bytes()[..value_width]);
                buf.extend_from_slice(&bytes[BODY_OFFSET + slot * value_width..index_offset]);
                // remap: a new distinct value at slot `slot` shifts every existing slot >= it up by one;
                // the new entry's index byte `slot` is inserted at `pos`
                let remap = |x: u8| if (x as usize) >= slot { x + 1 } else { x };
                for k in 0..pos {
                    buf.push(remap(bytes[index_offset + k]));
                }
                buf.push(slot as u8);
                for k in pos..count {
                    buf.push(remap(bytes[index_offset + k]));
                }
                // doc column: prefix | new doc | suffix
                append_spliced_docs(&mut buf, bytes, &layout, pos, doc);
                CompactIndexedLeaf(Arc::from(buf.as_slice()))
            }
        }
    }

    /// Cut entry `pos` out of the packed page — no decode. The caller guarantees the tuple is present at
    /// `pos` and that the leaf stays validly indexed afterward (`distinct_count < count - 1`). The distinct
    /// table is left untouched (a value whose last entry is removed becomes an unreferenced slot, reclaimed
    /// at the next rebuild, and reused if that value is re-inserted before then).
    fn splice_remove(
        &self,
        pos: usize,
    ) -> CompactIndexedLeaf {
        let bytes = &self.0;
        let layout = CompactLayout::read(bytes);
        let count = layout.count;
        let min = layout.min;
        let value_width = layout.value_width;
        let distinct_count = layout.distinct_count;
        let doc_width = layout.doc_width;
        let index_offset = layout.index_offset;
        let docs_offset = layout.docs_offset;
        let new_count = count - 1;
        let mut buf = Vec::with_capacity(
            BODY_OFFSET + distinct_count * value_width + new_count + new_count * doc_width,
        );
        write_compact_header(
            &mut buf,
            new_count,
            min,
            value_width,
            distinct_count,
            doc_width,
        );
        // distinct table verbatim
        buf.extend_from_slice(&bytes[BODY_OFFSET..index_offset]);
        // index: prefix | (drop pos) | suffix
        buf.extend_from_slice(&bytes[index_offset..index_offset + pos]);
        buf.extend_from_slice(&bytes[index_offset + pos + 1..index_offset + count]);
        // doc column: prefix | (drop pos) | suffix
        buf.extend_from_slice(&bytes[docs_offset..docs_offset + pos * doc_width]);
        buf.extend_from_slice(
            &bytes[docs_offset + (pos + 1) * doc_width..docs_offset + count * doc_width],
        );
        CompactIndexedLeaf(Arc::from(buf.as_slice()))
    }

    /// Merge a sorted `batch` into this indexed page — no full decode. The caller guarantees every batch
    /// entry [`packing_fits`] and `count + batch.len() <= LEAF_MAX`. Existing doc cells are copied
    /// verbatim; the distinct table is rebuilt as the sorted union of old + batch values and index bytes are
    /// re-pointed; exact `(key, doc)` duplicates are dropped. Stays indexed (old distinct < old count
    /// guarantees merged distinct < merged count).
    fn merge(
        &self,
        batch: &[(u64, u64)],
    ) -> CompactIndexedLeaf {
        let bytes = &self.0;
        let layout = CompactLayout::read(bytes);
        let count = layout.count;
        let min = layout.min;
        let value_width = layout.value_width;
        let distinct_count = layout.distinct_count;
        let doc_width = layout.doc_width;
        let index_offset = layout.index_offset;
        let docs_offset = layout.docs_offset;
        // distinct table = sorted union of old distinct values and the batch's keys
        let old_distinct: Vec<u64> = (0..distinct_count)
            .map(|slot| min + read_width(bytes, BODY_OFFSET + slot * value_width, value_width))
            .collect();
        let mut merged: Vec<u64> = Vec::with_capacity(distinct_count + batch.len());
        let (mut a, mut c) = (0usize, 0usize);
        while a < old_distinct.len() || c < batch.len() {
            let v = if a < old_distinct.len() && (c >= batch.len() || old_distinct[a] <= batch[c].0)
            {
                let x = old_distinct[a];
                a += 1;
                x
            } else {
                let x = batch[c].0;
                c += 1;
                x
            };
            if merged.last() != Some(&v) {
                merged.push(v);
            }
        }
        let new_distinct_count = merged.len();
        let slot_of = |key: u64| merged.partition_point(|&v| v < key);
        // merge-walk: two-pointer over leaf entries + batch, dropping exact (key, doc) dups; leaf doc cells
        // copied verbatim, every entry's index byte re-pointed into the rebuilt distinct table
        let leaf_key = |i: usize| {
            let slot = bytes[index_offset + i] as usize;
            min + read_width(bytes, BODY_OFFSET + slot * value_width, value_width)
        };
        let leaf_doc = |i: usize| read_width(bytes, docs_offset + i * doc_width, doc_width);
        let cap = count + batch.len();
        let mut index: Vec<u8> = Vec::with_capacity(cap);
        let mut docs: Vec<u8> = Vec::with_capacity(cap * doc_width);
        merge_walk(count, leaf_key, leaf_doc, batch, |key, doc, leaf_i| {
            index.push(slot_of(key) as u8);
            match leaf_i {
                // leaf entry: copy its doc cell verbatim
                Some(vi) => {
                    let doc_off = docs_offset + vi * doc_width;
                    docs.extend_from_slice(&bytes[doc_off..doc_off + doc_width]);
                }
                // batch entry: encode the new doc
                None => docs.extend_from_slice(&doc.to_le_bytes()[..doc_width]),
            }
        });
        let new_count = index.len();
        let mut buf = Vec::with_capacity(
            BODY_OFFSET + new_distinct_count * value_width + new_count + new_count * doc_width,
        );
        write_compact_header(
            &mut buf,
            new_count,
            min,
            value_width,
            new_distinct_count,
            doc_width,
        );
        // distinct value column
        for &v in &merged {
            buf.extend_from_slice(&(v - min).to_le_bytes()[..value_width]);
        }
        buf.extend_from_slice(&index);
        buf.extend_from_slice(&docs);
        CompactIndexedLeaf(Arc::from(buf.as_slice()))
    }

    /// Merge a sorted `batch` whose keys are all *existing* distinct values, by galloping to each entry's
    /// position and block-copying the leaf's index + doc cells between positions (verbatim — no per-entry key
    /// decode for the copied spans, and no distinct-table change so no index remap). The caller guarantees
    /// every entry [`packing_fits`], every key is already a distinct value, and
    /// `count + batch.len() <= LEAF_MAX`. Exact `(key, doc)` duplicates (within the batch or against the leaf)
    /// are dropped. Galloping (see [`gallop_lower_bound`]) makes this never worse than the full
    /// [`Self::merge`] walk at any batch size — `O(B·log(N/B))` — so the caller routes *all* all-existing-
    /// distinct batches here and only new-distinct ones (which need an index remap) to `merge`.
    fn block_copy_merge(
        &self,
        batch: &[(u64, u64)],
    ) -> CompactIndexedLeaf {
        let bytes = &self.0;
        let layout = CompactLayout::read(bytes);
        let count = layout.count;
        let value_width = layout.value_width;
        let doc_width = layout.doc_width;
        let index_offset = layout.index_offset;
        let docs_offset = layout.docs_offset;
        let distinct_value = |slot: usize| {
            layout.min + read_width(bytes, BODY_OFFSET + slot * value_width, value_width)
        };
        // slot of an (existing) distinct value, by binary search of the sorted distinct column
        let slot_of =
            |key: u64| partition_point(0..layout.distinct_count, |s| distinct_value(s) < key);
        let leaf_key = |i: usize| {
            layout.min
                + read_width(
                    bytes,
                    BODY_OFFSET + (bytes[index_offset + i] as usize) * value_width,
                    value_width,
                )
        };
        let leaf_doc = |i: usize| read_width(bytes, docs_offset + i * doc_width, doc_width);

        let capacity = count + batch.len();
        let mut new_index: Vec<u8> = Vec::with_capacity(capacity);
        let mut new_docs: Vec<u8> = Vec::with_capacity(capacity * doc_width);
        let mut leaf_pos = 0usize;
        let mut previous: Option<(u64, u64)> = None;
        for &(key, doc) in batch {
            // gallop to this entry's insert position within the not-yet-copied tail of the leaf — cheap
            // for a sparse batch, never worse than a linear scan for a dense one
            let position =
                gallop_lower_bound(leaf_pos, count, |i| (leaf_key(i), leaf_doc(i)) < (key, doc));
            // block-copy the leaf entries before it, verbatim (index byte + doc cell, no key decode)
            new_index.extend_from_slice(&bytes[index_offset + leaf_pos..index_offset + position]);
            new_docs.extend_from_slice(
                &bytes[docs_offset + leaf_pos * doc_width..docs_offset + position * doc_width],
            );
            leaf_pos = position;
            // drop exact `(key, doc)` duplicates: same as the previous batch entry, or already present in the
            // leaf at `position`
            let is_duplicate = previous == Some((key, doc))
                || (position < count && leaf_key(position) == key && leaf_doc(position) == doc);
            if !is_duplicate {
                new_index.push(slot_of(key) as u8);
                new_docs.extend_from_slice(&doc.to_le_bytes()[..doc_width]);
            }
            previous = Some((key, doc));
        }
        // block-copy the remaining leaf tail
        new_index.extend_from_slice(&bytes[index_offset + leaf_pos..index_offset + count]);
        new_docs.extend_from_slice(
            &bytes[docs_offset + leaf_pos * doc_width..docs_offset + count * doc_width],
        );

        let new_count = new_index.len();
        let mut buf = Vec::with_capacity(
            BODY_OFFSET + layout.distinct_count * value_width + new_count + new_count * doc_width,
        );
        write_compact_header(
            &mut buf,
            new_count,
            layout.min,
            value_width,
            layout.distinct_count,
            doc_width,
        );
        buf.extend_from_slice(&bytes[BODY_OFFSET..index_offset]); // distinct table unchanged (no new values)
        buf.extend_from_slice(&new_index);
        buf.extend_from_slice(&new_docs);
        CompactIndexedLeaf(Arc::from(buf.as_slice()))
    }
}

impl Leaf {
    /// This leaf's serialized encoding (carried out of band, see [`LeafFormat`]). The two compact in-RAM
    /// types share one on-disk format — index presence lives in the buffer — so both map to `Compact`.
    fn format(&self) -> LeafFormat {
        match self {
            Leaf::Aos(_) => LeafFormat::Aos,
            Leaf::Compact(_) | Leaf::CompactIndexed(_) => LeafFormat::Compact,
        }
    }

    /// Number of `(key, doc)` entries.
    fn count(&self) -> usize {
        match self {
            Leaf::Aos(l) => l.count(),
            Leaf::Compact(l) => l.count(),
            Leaf::CompactIndexed(l) => l.count(),
        }
    }

    /// Key of entry `i`.
    fn key(
        &self,
        i: usize,
    ) -> u64 {
        match self {
            Leaf::Aos(l) => l.key(i),
            Leaf::Compact(l) => l.key(i),
            Leaf::CompactIndexed(l) => l.key(i),
        }
    }

    /// Doc of entry `i`.
    fn doc(
        &self,
        i: usize,
    ) -> u64 {
        match self {
            Leaf::Aos(l) => l.doc(i),
            Leaf::Compact(l) => l.doc(i),
            Leaf::CompactIndexed(l) => l.doc(i),
        }
    }

    /// The doc array's `(base, stride, width)` — read once per leaf by the cursor's per-entry doc reads.
    fn doc_layout(&self) -> (usize, usize, usize) {
        match self {
            Leaf::Aos(_) => (FIELD, STRIDE, FIELD),
            Leaf::Compact(l) => l.doc_layout(),
            Leaf::CompactIndexed(l) => l.doc_layout(),
        }
    }

    /// First entry index whose key is `>= key` (binary search via [`Leaf::key`]). Used by the cursor to
    /// seek a leaf's range start and by the byte round-trip test.
    fn lower_bound(
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
    fn to_pairs(&self) -> Vec<(u64, u64)> {
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
    fn from_pairs(pairs: &[(u64, u64)]) -> Self {
        let count = pairs.len();
        if count == 0 {
            return Leaf::Aos(AosLeaf::build(pairs));
        }
        // One pass for the two data-dependent inputs — the distinct-value count (runs, since sorted) and the
        // max doc. (The value range needs no scan; it's read O(1) from the sorted ends just below.)
        let mut distinct_count = 1usize;
        let mut max_doc = pairs[0].1;
        for window in pairs.windows(2) {
            if window[0].0 != window[1].0 {
                distinct_count += 1;
            }
            max_doc = max_doc.max(window[1].1);
        }
        let min_value = pairs[0].0;
        let value_width = pow2_bytes_for(pairs[count - 1].0 - min_value);
        let doc_width = pow2_bytes_for(max_doc);
        let deduplicated = distinct_count < count;
        let compact_size = BODY_OFFSET
            + distinct_count * value_width
            + if deduplicated { count } else { 0 }
            + count * doc_width;
        let aos_size = count * STRIDE;
        if compact_size + COMPACT_MIN_SAVING_BPE * count <= aos_size {
            if deduplicated {
                Leaf::CompactIndexed(CompactIndexedLeaf::build(
                    pairs,
                    count,
                    distinct_count,
                    min_value,
                    value_width,
                    doc_width,
                ))
            } else {
                Leaf::Compact(CompactLeaf::build(
                    pairs,
                    count,
                    min_value,
                    value_width,
                    doc_width,
                ))
            }
        } else {
            Leaf::Aos(AosLeaf::build(pairs))
        }
    }

    /// Re-adopt a leaf from its format and serialized bytes — the counterpart of [`Leaf::format`] /
    /// [`Leaf::bytes`] and the read side of [`CowBTree::leaves`]. The bytes are the page verbatim, so this
    /// is a copy, never a decode. It **trusts** its input (validating arbitrary bytes is the caller's
    /// responsibility); currently exercised only by the byte round-trip test.
    #[allow(dead_code)]
    fn from_parts(
        format: LeafFormat,
        bytes: Arc<[u8]>,
    ) -> Self {
        match format {
            LeafFormat::Aos => Leaf::Aos(AosLeaf(bytes)),
            // The compact format carries index presence in the buffer (it is not part of [`LeafFormat`]):
            // a dedup index is present iff there are fewer distinct values than entries, so pick the in-RAM
            // type by comparing the two header counts.
            LeafFormat::Compact => {
                if read_u16(&bytes, DISTINCT_COUNT_OFFSET) < read_u16(&bytes, ENTRY_COUNT_OFFSET) {
                    Leaf::CompactIndexed(CompactIndexedLeaf(bytes))
                } else {
                    Leaf::Compact(CompactLeaf(bytes))
                }
            }
        }
    }

    /// The leaf's serialized (tag-free) bytes — an `Arc` bump, no serialization step.
    fn bytes(&self) -> Arc<[u8]> {
        match self {
            Leaf::Aos(l) => Arc::clone(&l.0),
            Leaf::Compact(l) => Arc::clone(&l.0),
            Leaf::CompactIndexed(l) => Arc::clone(&l.0),
        }
    }

    /// The raw page bytes (for the cursor's cached doc read).
    fn raw(&self) -> &[u8] {
        match self {
            Leaf::Aos(l) => &l.0,
            Leaf::Compact(l) => &l.0,
            Leaf::CompactIndexed(l) => &l.0,
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
    fn insert(
        &self,
        key: u64,
        doc: u64,
    ) -> Option<LeafInsert> {
        let count = self.count();
        let pos = self.lower_bound_entry(key, doc);
        if pos < count && self.key(pos) == key && self.doc(pos) == doc {
            return None; // already present
        }
        if count < LEAF_MAX {
            // In-place splice (no decode), each variant keeping its own format. `count < LEAF_MAX` ⇒
            // `count + 1 <= LEAF_MAX`, so the result still fits one page.
            match self {
                Leaf::Aos(aos) => {
                    let cut = pos * STRIDE; // memcpy prefix + tuple + suffix; data at byte 0 (tag-free)
                    let mut buf = Vec::with_capacity(aos.0.len() + STRIDE);
                    buf.extend_from_slice(&aos.0[..cut]);
                    buf.extend_from_slice(&key.to_le_bytes());
                    buf.extend_from_slice(&doc.to_le_bytes());
                    buf.extend_from_slice(&aos.0[cut..]);
                    return Some(LeafInsert::Fit(Leaf::Aos(AosLeaf(Arc::from(
                        buf.as_slice(),
                    )))));
                }
                Leaf::Compact(l) if packing_fits(&l.0, key, doc) => {
                    return Some(LeafInsert::Fit(Leaf::Compact(
                        l.splice_insert(key, doc, pos),
                    )));
                }
                Leaf::CompactIndexed(l) if packing_fits(&l.0, key, doc) => {
                    return Some(LeafInsert::Fit(Leaf::CompactIndexed(
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

    /// Remove one `(key, doc)`: the replacement leaf plus whether it underflowed (`< LEAF_MIN`), or `None`
    /// if the tuple is absent. An [`AosLeaf`] is **spliced directly** (the 16-byte tuple is cut out); a
    /// [`CompactLeaf`] is rebuilt via [`Leaf::from_pairs`]. Re-compacting an AoS leaf happens at its next
    /// merge (which rebuilds through [`Leaf::from_pairs`]).
    fn remove(
        &self,
        key: u64,
        doc: u64,
    ) -> Option<(Leaf, bool)> {
        let count = self.count();
        let pos = self.lower_bound_entry(key, doc);
        if pos >= count || self.key(pos) != key || self.doc(pos) != doc {
            return None; // absent
        }
        let new_count = count - 1;
        let leaf = match self {
            Leaf::Aos(aos) => {
                let cut = pos * STRIDE; // data at byte 0 (tag-free)
                let mut buf = Vec::with_capacity(aos.0.len() - STRIDE);
                buf.extend_from_slice(&aos.0[..cut]);
                buf.extend_from_slice(&aos.0[cut + STRIDE..]);
                Leaf::Aos(AosLeaf(Arc::from(buf.as_slice())))
            }
            // Cut the entry in place (no decode). Indexed leaves only when the result stays validly indexed
            // (`distinct_count < new_count`); a now-emptied leaf or an indexed leaf that would no longer
            // satisfy that falls to the rebuild below (canonical empty page / re-selects the format).
            Leaf::Compact(l) if new_count > 0 => Leaf::Compact(l.splice_remove(pos)),
            Leaf::CompactIndexed(l) if l.distinct_count() < new_count => {
                Leaf::CompactIndexed(l.splice_remove(pos))
            }
            _ => {
                let mut pairs = self.to_pairs();
                pairs.remove(pos);
                Self::from_pairs(&pairs)
            }
        };
        Some((leaf, new_count < LEAF_MIN))
    }

    /// Apply a sorted `batch` (all of it routing into this leaf) and return the replacement leaf page(s).
    /// An [`AosLeaf`] whose result still fits one page is **byte-merged** (see [`AosLeaf::merge_batch`]);
    /// a [`CompactLeaf`], or one whose result overflows into several pages, decodes, merges, de-dups, and
    /// re-chunks through [`Leaf::from_pairs`] (re-selecting the format per chunk).
    fn merge_batch(
        &self,
        batch: &[(u64, u64)],
    ) -> Vec<Leaf> {
        // Fast path: the whole batch fits one page and never widens a packing parameter — digest it into the
        // packed bytes (existing cells copied verbatim, only batch entries encoded), keeping the format.
        if self.count() + batch.len() <= LEAF_MAX {
            match self {
                Leaf::Aos(aos) => return vec![Leaf::Aos(aos.merge_batch(batch))],
                Leaf::Compact(l) if batch.iter().all(|&(k, d)| packing_fits(&l.0, k, d)) => {
                    return vec![Leaf::Compact(l.merge(batch))];
                }
                Leaf::CompactIndexed(l) if batch.iter().all(|&(k, d)| packing_fits(&l.0, k, d)) => {
                    // If every batch key is *already* a distinct value, block-copy the leaf's index + doc
                    // cells between the (galloped) insert positions — no index remap, and the gallop makes
                    // this never worse than the merge-walk at any batch size (so no size guard). A batch that
                    // introduces a new distinct value needs an index remap, so it takes the merge-walk.
                    if batch.iter().all(|&(k, _)| l.distinct_slot(k).is_ok()) {
                        return vec![Leaf::CompactIndexed(l.block_copy_merge(batch))];
                    }
                    return vec![Leaf::CompactIndexed(l.merge(batch))];
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

impl AosLeaf {
    /// Two-pointer merge of `self`'s entries with a sorted `batch` into a single new tag-free AoS page — no
    /// `Vec<(u64, u64)>`, no sort, one allocation. Exact `(key, doc)` duplicates are dropped (re-adding an
    /// existing tuple is a no-op). The caller guarantees the result fits (`count + batch.len() <= LEAF_MAX`).
    /// (A galloping block-copy was tried here but only helped tiny batches and regressed large ones — AoS has
    /// no per-entry decode for it to amortize, unlike the compact merges — so the simple walk stays.)
    fn merge_batch(
        &self,
        batch: &[(u64, u64)],
    ) -> AosLeaf {
        let count = self.count();
        let mut buf = Vec::with_capacity(self.0.len() + batch.len() * STRIDE);
        merge_walk(
            count,
            |i| self.key(i),
            |i| self.doc(i),
            batch,
            |key, doc, _| {
                buf.extend_from_slice(&key.to_le_bytes());
                buf.extend_from_slice(&doc.to_le_bytes());
            },
        );
        AosLeaf(Arc::from(buf.as_slice()))
    }
}

// ---- bottom-up builders (used by bulk build and batched insert) ------------------------------

/// Group `children` into branch pages of at most `BRANCH_MAX`, one separator per adjacent pair
/// (the separator is the minimum `(key, doc)` of the right child).
///
/// Every emitted branch gets **at least two children**: a lone single-child branch has no sibling, so a
/// later delete that underflowed its only child could not rebalance it (see [`Branch::rebalance`]). The
/// only way `chunks(BRANCH_MAX)` would leave a singleton is a trailing remainder of exactly 1, i.e. an
/// input length of `BRANCH_MAX + 1`; that case is split as `BRANCH_MAX - 1` + `2` instead.
fn pack_branches(children: Vec<Node>) -> Vec<Node> {
    // One branch per chunk of up to `BRANCH_MAX` children, so the count is known up front.
    let mut packed = Vec::with_capacity(children.len().div_ceil(BRANCH_MAX));
    let mut rest = &children[..];
    while !rest.is_empty() {
        let take = if rest.len() == BRANCH_MAX + 1 {
            BRANCH_MAX - 1
        } else {
            rest.len().min(BRANCH_MAX)
        };
        let (chunk, remaining) = rest.split_at(take);
        // Each separator is the right child's minimum `(key, doc)` — a single B+-tree boundary point,
        // NOT a `(min, max)` range: everything in `children[i]` is `< seps[i] == min(children[i + 1])`.
        let seps: Vec<(u64, u64)> = chunk[1..].iter().map(Node::min).collect();
        packed.push(Node::Branch(Arc::new(Branch {
            seps,
            children: chunk.to_vec(),
        })));
        rest = remaining;
    }
    packed
}

/// Stitch a flat list of node fragments into a single root, packing branch levels bottom-up until
/// one node remains. An empty list becomes an empty leaf.
fn build_root(mut fragments: Vec<Node>) -> Node {
    while fragments.len() > 1 {
        fragments = pack_branches(fragments);
    }
    fragments
        .pop()
        .unwrap_or_else(|| Node::Leaf(Leaf::from_pairs(&[])))
}

// ---- node-operation results ------------------------------------------------------------------

/// A node that split under an insert: the new right sibling plus the separator promoted to the parent.
struct Split {
    sep: (u64, u64),
    right: Node,
}
/// Outcome of inserting into a single leaf: the replacement leaf if it still fits, or the two halves plus
/// their separator if it overflowed and split. (`None` from [`Leaf::insert`] means the tuple was present.)
enum LeafInsert {
    Fit(Leaf),
    Split {
        left: Leaf,
        sep: (u64, u64),
        right: Leaf,
    },
}
/// Outcome of combining two siblings: a single merged node, or two re-balanced nodes plus their new
/// separator.
enum Combined {
    One(Node),
    Two(Node, (u64, u64), Node),
}

impl Branch {
    /// Index of the child that an entry `(key, doc)` routes into — the number of separators `<=` it.
    fn child_index(
        &self,
        key: u64,
        doc: u64,
    ) -> usize {
        self.seps.partition_point(|&sep| sep <= (key, doc))
    }

    /// Repair this branch **in place** after its child `child_idx` underflowed, by combining that
    /// child with an adjacent sibling — either merging the pair into one node (dropping a separator)
    /// or re-balancing them into two (updating the separator).
    fn rebalance(
        &mut self,
        child_idx: usize,
    ) {
        // A branch always has ≥ 2 children — [`pack_branches`] never emits a single-child branch and
        // splits/merges preserve this — so the underflowed child always has a sibling to pair with.
        // Guard anyway: with no sibling there is nothing to combine, so let the underflow propagate to
        // the parent (which will merge this branch) rather than indexing `child_idx - 1` out of bounds.
        if self.children.len() < 2 {
            return;
        }
        // Pair the underflowed child with its right neighbour when there is one, otherwise its left.
        let (left_idx, right_idx) = if child_idx + 1 < self.children.len() {
            (child_idx, child_idx + 1)
        } else {
            (child_idx - 1, child_idx)
        };
        // `combine` re-splits the pair in sorted order, so the merged / new-left node keeps the pair's
        // minimum (`min(new left) == min(old left)`). That is why only the **inter-pair** separator
        // `seps[left_idx]` is ever touched below: the separators bracketing the pair from outside
        // (`seps[left_idx - 1] == min(left)` and `seps[right_idx]`, untouched) still point at unchanged
        // minima, so they stay valid without being rewritten.
        match self.children[left_idx].combine(self.seps[left_idx], &self.children[right_idx]) {
            Combined::One(merged) => {
                // Merge: the inter-pair separator vanishes along with the absorbed right child.
                self.children[left_idx] = merged;
                self.children.remove(right_idx);
                self.seps.remove(left_idx);
            }
            Combined::Two(left, sep, right) => {
                // Borrow: only the inter-pair separator moves — to the rebalanced right node's new min.
                self.children[left_idx] = left;
                self.children[right_idx] = right;
                self.seps[left_idx] = sep;
            }
        }
    }
}

/// Test-only synchronization seam (compiled out of release builds). It lets the concurrency test
/// [`tests::make_mut_locks_out_a_concurrent_reader`] freeze a writer in the middle of `make_mut` —
/// while it still holds the `&mut Branch` — so the test can prove that a reader cannot clone the tree
/// during a mutation. The hook fires only for one sentinel key that no other test inserts, so it stays
/// inert even though the test binary runs tests in parallel.
#[cfg(test)]
mod make_mut_gate {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

    /// A key no other test inserts — the park hook keys off it so only the concurrency test triggers it.
    pub(super) const KEY: u64 = 0xFFFF_FFFF_FFFF_FF00;
    /// Writer → test: "I am parked inside `make_mut`, holding the `&mut`."
    pub(super) static PARKED: AtomicBool = AtomicBool::new(false);
    /// Test → writer: "you may proceed."
    pub(super) static RELEASE: AtomicBool = AtomicBool::new(false);

    /// Called right after `make_mut` in `insert_one`. For the sentinel key only, announce that we've
    /// parked (still holding the `&mut Branch`) and spin until the test releases us.
    pub(super) fn park_if(key: u64) {
        if key == KEY {
            PARKED.store(true, SeqCst);
            while !RELEASE.load(SeqCst) {
                std::hint::spin_loop();
            }
        }
    }
}

impl Node {
    /// The minimum `(key, doc)` in this subtree — walk the left spine down to its first leaf.
    fn min(&self) -> (u64, u64) {
        let mut node = self;
        loop {
            match node {
                Node::Leaf(leaf) => return (leaf.key(0), leaf.doc(0)),
                Node::Branch(branch) => node = &branch.children[0],
            }
        }
    }

    /// Apply a sorted `batch` (every entry of which routes into this subtree) by **copying only the
    /// nodes the batch touches**, returning the replacement node(s). A node that receives more
    /// entries than fit in one page splits, so this returns a *list* of fragments; the caller
    /// (the parent branch, or [`build_root`] at the top) stitches them back together.
    fn apply_batch(
        &self,
        batch: &[(u64, u64)],
    ) -> Vec<Node> {
        match self {
            // Leaf: merge the batch into this page (the leaf owns the encoding-specific work — an AoS page
            // that still fits is byte-merged, no decode; see [`Leaf::merge_batch`]). It may split into several.
            Node::Leaf(leaf) => leaf
                .merge_batch(batch)
                .into_iter()
                .map(Node::Leaf)
                .collect(),
            // Branch: hand each child the slice of `batch` that routes into it, recursing only into
            // children that actually receive entries — every other child is shared by `Arc` clone,
            // never copied. Re-pack the resulting child list (which may have grown, since a touched
            // child can split into several) back into branch pages.
            Node::Branch(branch) => {
                // At least one replacement child per existing child (a touched one may split into more).
                let mut new_children: Vec<Node> = Vec::with_capacity(branch.children.len());
                // Split the sorted `batch` across the children with a single linear sweep — a merge of the
                // sorted batch against the sorted separators. `cursor` only ever advances, so a child that
                // receives nothing costs one comparison, not a binary search over the whole remaining batch:
                // routing a localized batch over a wide branch is O(children + batch), not
                // O(children * log batch). (A merge-pass, not galloping — every child must be emitted
                // (touched ones recurse, untouched share by Arc), so the O(children) walk is unavoidable.)
                let mut cursor = 0usize;
                for (child_idx, child) in branch.children.iter().enumerate() {
                    // Each child owns keys strictly below its right separator; the last child (no separator)
                    // owns everything remaining.
                    let child_upper = branch
                        .seps
                        .get(child_idx)
                        .copied()
                        .unwrap_or((u64::MAX, u64::MAX));
                    let start = cursor;
                    while cursor < batch.len() && batch[cursor] < child_upper {
                        cursor += 1;
                    }
                    let for_child = &batch[start..cursor];
                    if for_child.is_empty() {
                        new_children.push(child.clone()); // nothing routes here ⇒ share the page
                    } else {
                        new_children.extend(child.apply_batch(for_child));
                    }
                }
                pack_branches(new_children)
            }
        }
    }

    /// Insert a single `(key, doc)` **in place**, copy-on-write down the touched path. `Arc::make_mut`
    /// mutates a branch in place when this version is its sole owner (refcount 1) and copies it only
    /// when a snapshot still shares it — so a held snapshot is never disturbed, yet a burst of inserts
    /// into an unshared tree pays no per-level branch copy after the first touch. Returns `Some(Split)`
    /// when the node split (the parent must take in the new right sibling), else `None`. Idempotent:
    /// inserting an already-present tuple is a no-op.
    fn insert_one(
        &mut self,
        key: u64,
        doc: u64,
    ) -> Option<Split> {
        match self {
            // The leaf owns the encoding-specific work (an AoS leaf splices its bytes; see [`Leaf::insert`]).
            Node::Leaf(leaf) => match leaf.insert(key, doc)? {
                LeafInsert::Fit(new) => {
                    *leaf = new;
                    None
                }
                LeafInsert::Split { left, sep, right } => {
                    *leaf = left;
                    Some(Split {
                        sep,
                        right: Node::Leaf(right),
                    })
                }
            },
            Node::Branch(branch_arc) => {
                let branch = Arc::make_mut(branch_arc); // in place if unshared, else copy-on-write
                #[cfg(test)]
                make_mut_gate::park_if(key); // test-only; no-op in release builds
                let child_idx = branch.child_index(key, doc);
                let Some(Split { sep, right }) = branch.children[child_idx].insert_one(key, doc)
                else {
                    return None; // absorbed below — nothing to insert here
                };
                // The child split — take in the promoted separator and the new right sibling beside it.
                branch.seps.insert(child_idx, sep);
                branch.children.insert(child_idx + 1, right);
                if branch.children.len() <= BRANCH_MAX {
                    None
                } else {
                    // This branch overflowed in turn: keep the left half, promote the middle
                    // separator (it moves up, into neither side), hand the right half up.
                    let mid = branch.children.len() / 2;
                    let right_children = branch.children.split_off(mid);
                    let right_seps = branch.seps.split_off(mid);
                    let promoted = branch.seps.pop().unwrap(); // the separator between the two halves
                    Some(Split {
                        sep: promoted,
                        right: Node::Branch(Arc::new(Branch {
                            seps: right_seps,
                            children: right_children,
                        })),
                    })
                }
            }
        }
    }

    /// Combine this node with its right sibling `right` (joined in the parent by separator `sep`),
    /// then re-split: into one node when they fit together (a merge), or two balanced nodes when they
    /// overflow (a borrow). Used to repair an underflow. Both nodes are always the same kind.
    fn combine(
        &self,
        sep: (u64, u64),
        right: &Node,
    ) -> Combined {
        match (self, right) {
            (Node::Leaf(left_leaf), Node::Leaf(right_leaf)) => {
                let mut pairs = left_leaf.to_pairs();
                pairs.extend(right_leaf.to_pairs());
                if pairs.len() <= LEAF_MAX {
                    Combined::One(Node::Leaf(Leaf::from_pairs(&pairs)))
                } else {
                    let mid = pairs.len() / 2;
                    Combined::Two(
                        Node::Leaf(Leaf::from_pairs(&pairs[..mid])),
                        pairs[mid],
                        Node::Leaf(Leaf::from_pairs(&pairs[mid..])),
                    )
                }
            }
            (Node::Branch(left_branch), Node::Branch(right_branch)) => {
                let mut children = left_branch.children.clone();
                children.extend(right_branch.children.iter().cloned());
                let mut seps = left_branch.seps.clone();
                seps.push(sep); // the parent separator becomes an internal one in the joined branch
                seps.extend(right_branch.seps.iter().cloned());
                if children.len() <= BRANCH_MAX {
                    Combined::One(Node::Branch(Arc::new(Branch { seps, children })))
                } else {
                    let mid = children.len() / 2;
                    let left_children = children[..mid].to_vec();
                    let right_children = children[mid..].to_vec();
                    let left_seps = seps[..mid - 1].to_vec();
                    let promoted = seps[mid - 1];
                    let right_seps = seps[mid..].to_vec();
                    Combined::Two(
                        Node::Branch(Arc::new(Branch {
                            seps: left_seps,
                            children: left_children,
                        })),
                        promoted,
                        Node::Branch(Arc::new(Branch {
                            seps: right_seps,
                            children: right_children,
                        })),
                    )
                }
            }
            _ => unreachable!("siblings are always the same node type"),
        }
    }

    /// Remove a single `(key, doc)` **in place**, copy-on-write down the touched path (see
    /// [`Node::insert_one`] on `make_mut`). Returns `None` if the tuple is absent; otherwise `Some`
    /// carrying `true` when the touched page dropped below its minimum fill, so the parent re-balances it.
    fn remove_one(
        &mut self,
        key: u64,
        doc: u64,
    ) -> Option<bool> {
        match self {
            Node::Leaf(leaf) => {
                let (new, underflow) = leaf.remove(key, doc)?; // `None` ⇒ not present
                *leaf = new;
                Some(underflow)
            }
            Node::Branch(branch_arc) => {
                let branch = Arc::make_mut(branch_arc); // in place if unshared, else copy-on-write
                let child_idx = branch.child_index(key, doc);
                if branch.children[child_idx].remove_one(key, doc)? {
                    branch.rebalance(child_idx);
                }
                Some(branch.children.len() < BRANCH_MIN)
            }
        }
    }
}

// ---- the tree --------------------------------------------------------------------------------

/// A copy-on-write B⁺-tree mapping sorted `(key, doc)` tuples; see the module docs.
#[derive(Clone)]
pub struct CowBTree {
    root: Node,
}

impl Default for CowBTree {
    fn default() -> Self {
        Self {
            root: Node::Leaf(Leaf::from_pairs(&[])),
        }
    }
}

impl CowBTree {
    /// An empty tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bulk-build from `pairs` **sorted ascending and unique** by `(key, doc)`. Packs full leaf pages
    /// directly from the slice and builds the branch levels bottom-up — no per-item sort, dedup, or
    /// insert traversal, so it is far cheaper than inserting one at a time.
    pub fn from_sorted(pairs: &[(u64, u64)]) -> Self {
        // Enforced in all builds (not just `debug`): a violation silently corrupts the tree — the
        // bottom-up build derives separators assuming this order — and the O(n) scan is dwarfed by the
        // O(n) page encoding the build does anyway.
        assert!(
            pairs.windows(2).all(|w| w[0] < w[1]),
            "from_sorted input must be sorted and unique"
        );
        if pairs.is_empty() {
            return Self::default();
        }
        let leaves: Vec<Node> = pairs
            .chunks(LEAF_MAX)
            .map(|chunk| Node::Leaf(Leaf::from_pairs(chunk)))
            .collect();
        Self {
            root: build_root(leaves),
        }
    }

    /// Insert a single `(key, doc)`. Idempotent: inserting an existing tuple is a no-op.
    pub fn insert(
        &mut self,
        key: u64,
        doc: u64,
    ) {
        if let Some(Split { sep, right }) = self.root.insert_one(key, doc) {
            // The root split — grow a fresh level above the two halves.
            let left = std::mem::replace(&mut self.root, Node::Leaf(Leaf::from_pairs(&[])));
            self.root = Node::Branch(Arc::new(Branch {
                seps: vec![sep],
                children: vec![left, right],
            }));
        }
    }

    /// Apply a batch of `(key, doc)` adds **sorted ascending**, copying each touched page once.
    pub fn insert_batch(
        &mut self,
        sorted_adds: &[(u64, u64)],
    ) {
        if sorted_adds.is_empty() {
            return;
        }
        // Enforced in all builds: out-of-order input mis-routes tuples (the branch path binary-searches
        // the remaining slice), permanently breaking ordering. The O(n) check is cheap next to the batch.
        assert!(
            sorted_adds.windows(2).all(|w| w[0] <= w[1]),
            "insert_batch input must be sorted"
        );
        self.root = build_root(self.root.apply_batch(sorted_adds));
    }

    /// Remove a single `(key, doc)`; merges underflowing pages so the tree stays compact. A missing
    /// tuple is a no-op.
    pub fn remove(
        &mut self,
        key: u64,
        doc: u64,
    ) {
        if self.root.remove_one(key, doc).is_some() {
            // A branch root that shrank to a single child loses a level.
            while let Node::Branch(branch) = &self.root {
                if branch.children.len() == 1 {
                    let only_child = branch.children[0].clone();
                    self.root = only_child;
                } else {
                    break;
                }
            }
        }
    }

    /// Lazily iterate the doc ids whose key lies in `[lo, hi]`, in `(key, doc)` order. The returned
    /// iterator owns a snapshot of the tree (an `O(1)` root clone) and can be dropped mid-scan.
    pub fn range(
        &self,
        lo: u64,
        hi: u64,
    ) -> RangeIter {
        RangeIter::new(&self.root, (lo, 0), hi)
    }

    /// Lazily iterate the doc ids whose key equals `key` (a degenerate range).
    pub fn point(
        &self,
        key: u64,
    ) -> RangeIter {
        self.range(key, key)
    }

    /// Total number of live tuples.
    pub fn len(&self) -> usize {
        fn count(node: &Node) -> usize {
            match node {
                Node::Leaf(leaf) => leaf.count(),
                Node::Branch(branch) => branch.children.iter().map(count).sum(),
            }
        }
        count(&self.root)
    }

    /// Whether the tree holds no tuples. `O(1)`: the tree is empty iff its root is an empty leaf — a
    /// non-empty tree's root is a branch or a leaf with entries, and underflowing leaves are merged, so an
    /// empty leaf only ever exists as the root of an empty tree. (Avoids walking every page like [`len`].)
    pub fn is_empty(&self) -> bool {
        matches!(&self.root, Node::Leaf(leaf) if leaf.count() == 0)
    }

    /// All leaf pages in key order, each paired with its [`LeafFormat`]. The bytes are the (tag-free)
    /// serialized form and the format is carried alongside, so writing the tree out to a byte store needs
    /// no further encoding — re-adopt a page with [`Leaf::from_parts`].
    pub fn leaves(&self) -> Vec<(LeafFormat, Arc<[u8]>)> {
        fn walk(
            node: &Node,
            out: &mut Vec<(LeafFormat, Arc<[u8]>)>,
        ) {
            match node {
                Node::Leaf(leaf) => out.push((leaf.format(), leaf.bytes())),
                Node::Branch(branch) => branch.children.iter().for_each(|child| walk(child, out)),
            }
        }
        let mut v = Vec::new();
        walk(&self.root, &mut v);
        v
    }
}

/// A lazy, droppable cursor over the doc ids in a key range. Owns a snapshot of the tree.
pub struct RangeIter {
    _root: Node, // keeps the snapshot (and all its pages) alive for the cursor's lifetime
    stack: Vec<(Arc<Branch>, usize)>, // (branch, next child index to descend)
    leaf: Option<Leaf>,
    leaf_count: usize, // entry count of `leaf`, read once per leaf rather than per entry
    whole: bool,       // every entry of `leaf` is `<= hi_key` ⇒ skip the per-entry bound check
    // Cached doc decode layout `(base, stride, width)` for the current leaf — read once in `set_leaf`
    // (from [`Leaf::doc_layout`]) so `next` decodes docs without recomputing offsets per entry (matters
    // for the compact formats). Keys, by contrast, are read straight through [`Leaf::key`]: that happens
    // only at the range start and on a boundary leaf, never per yielded entry, so it needs no cache.
    doc_base: usize,
    doc_stride: usize,
    doc_width: usize,
    pos: usize,
    hi_key: u64, // inclusive upper key bound (the doc half of the bound is always `u64::MAX`)
}

impl RangeIter {
    fn new(
        root: &Node,
        lo: (u64, u64),
        hi_key: u64,
    ) -> Self {
        let mut it = RangeIter {
            _root: root.clone(),
            stack: Vec::new(),
            leaf: None,
            leaf_count: 0,
            whole: false,
            doc_base: 0,
            doc_stride: 0,
            doc_width: 0,
            pos: 0,
            hi_key,
        };
        let mut node = root.clone();
        loop {
            match node {
                Node::Leaf(leaf) => {
                    let pos = leaf.lower_bound(lo.0);
                    it.set_leaf(leaf, pos);
                    break;
                }
                Node::Branch(branch) => {
                    let child_idx = branch.child_index(lo.0, lo.1);
                    let child = branch.children[child_idx].clone();
                    it.stack.push((branch, child_idx + 1)); // resume at the next sibling after this subtree
                    node = child;
                }
            }
        }
        it
    }

    /// Position the cursor on `leaf` at entry `pos`, caching its count, doc layout, and whether the leaf's
    /// last key is within the upper bound. When it is, every entry from `pos` on qualifies, so
    /// [`Iterator::next`] can emit docs without reading the key or comparing — only a *boundary* leaf needs
    /// per-entry checks.
    fn set_leaf(
        &mut self,
        leaf: Leaf,
        pos: usize,
    ) {
        let count = leaf.count();
        self.whole = count == 0 || leaf.key(count - 1) <= self.hi_key;
        self.leaf_count = count;
        let (base, stride, width) = leaf.doc_layout();
        self.doc_base = base;
        self.doc_stride = stride;
        self.doc_width = width;
        self.pos = pos;
        self.leaf = Some(leaf);
    }

    /// Descend a node's left spine to its first leaf, pushing branch frames.
    fn descend_left(
        &mut self,
        mut node: Node,
    ) {
        loop {
            match node {
                Node::Leaf(leaf) => {
                    self.set_leaf(leaf, 0);
                    return;
                }
                Node::Branch(branch) => {
                    let first_child = branch.children[0].clone();
                    self.stack.push((branch, 1));
                    node = first_child;
                }
            }
        }
    }

    /// Advance to the next leaf in key order (or leave `self.leaf` `None` if exhausted).
    fn advance_leaf(&mut self) {
        self.leaf = None;
        // Walk back up the saved path until a frame still has an unvisited sibling subtree.
        while let Some((branch, next_child)) = self.stack.pop() {
            if next_child < branch.children.len() {
                let child = branch.children[next_child].clone();
                self.stack.push((branch, next_child + 1)); // restore the frame, cursor advanced
                self.descend_left(child); // into this subtree's leftmost leaf
                return;
            }
            // else: this branch is fully consumed — leave it popped and keep going up.
        }
        // Stack emptied ⇒ no more leaves; `self.leaf` stays `None`.
    }
}

impl Iterator for RangeIter {
    type Item = u64; // doc id

    fn next(&mut self) -> Option<u64> {
        loop {
            let leaf = self.leaf.as_ref()?; // no current leaf ⇒ the scan is exhausted
            if self.pos < self.leaf_count {
                // Cached layout ⇒ no per-entry offset recompute (matters for the compact format).
                let doc = read_width(
                    leaf.raw(),
                    self.doc_base + self.pos * self.doc_stride,
                    self.doc_width,
                );
                // Interior leaves are wholly in range (`self.whole`); only a boundary leaf needs the
                // per-entry check. The doc half of the bound is always `MAX`, so the lexicographic
                // `(key, doc) > hi` reduces to `key > hi_key`.
                if !self.whole && leaf.key(self.pos) > self.hi_key {
                    self.leaf = None;
                    return None; // walked past the range
                }
                self.pos += 1;
                return Some(doc);
            }
            self.advance_leaf(); // consumed this leaf ⇒ move on; the loop re-checks at the top
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Sorted doc ids the tree yields for `[lo, hi]`.
    fn tree_range(
        t: &CowBTree,
        lo: u64,
        hi: u64,
    ) -> Vec<u64> {
        let mut v: Vec<u64> = t.range(lo, hi).collect();
        v.sort_unstable();
        v
    }
    /// The same, computed from a reference set of `(key, doc)`.
    fn ref_range(
        s: &BTreeSet<(u64, u64)>,
        lo: u64,
        hi: u64,
    ) -> Vec<u64> {
        let mut v: Vec<u64> = s.range((lo, 0)..=(hi, u64::MAX)).map(|&(_, d)| d).collect();
        v.sort_unstable();
        v
    }

    /// The full `(key, doc)` multiset the tree stores, decoded straight from the leaf bytes (so it verifies
    /// the *value* column, not just docs — a full-range cursor scan skips key reads on interior leaves).
    fn tree_pairs(t: &CowBTree) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = t
            .leaves()
            .into_iter()
            .flat_map(|(fmt, bytes)| Leaf::from_parts(fmt, bytes).to_pairs())
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn empty_and_single() {
        let mut t = CowBTree::new();
        assert!(t.is_empty());
        assert_eq!(t.point(5).count(), 0);
        t.insert(5, 50);
        assert_eq!(t.len(), 1);
        assert_eq!(t.point(5).collect::<Vec<_>>(), vec![50]);
        assert_eq!(t.point(6).count(), 0);
    }

    #[test]
    fn insert_split_parity() {
        // enough inserts to force several levels of splits
        let n = 5_000u64;
        let mut t = CowBTree::new();
        let mut reference = BTreeSet::new();
        for q in 0..n {
            let k = splitmix(q) % 1000; // duplicate keys (multiple docs per key)
            t.insert(k, q);
            reference.insert((k, q));
        }
        assert_eq!(t.len(), reference.len());
        for k in 0..1000u64 {
            assert_eq!(
                tree_range(&t, k, k),
                ref_range(&reference, k, k),
                "point {k}"
            );
        }
        for q in 0..200u64 {
            let lo = splitmix(q ^ 1) % 1000;
            let hi = (lo + 50).min(999);
            assert_eq!(
                tree_range(&t, lo, hi),
                ref_range(&reference, lo, hi),
                "range [{lo},{hi}]"
            );
        }
    }

    #[test]
    fn bulk_build_matches_incremental() {
        let pairs: Vec<(u64, u64)> = (0..3_000u64).map(|i| (i, i)).collect();
        let bulk = CowBTree::from_sorted(&pairs);
        let mut inc = CowBTree::new();
        for &(k, d) in &pairs {
            inc.insert(k, d);
        }
        assert_eq!(bulk.len(), inc.len());
        for k in (0..3_000u64).step_by(7) {
            assert_eq!(tree_range(&bulk, k, k), tree_range(&inc, k, k));
        }
        assert_eq!(tree_range(&bulk, 100, 2_900), tree_range(&inc, 100, 2_900));
    }

    #[test]
    fn remove_merge_parity() {
        let n = 4_000u64;
        let mut t = CowBTree::from_sorted(&(0..n).map(|i| (i, i)).collect::<Vec<_>>());
        let mut reference: BTreeSet<(u64, u64)> = (0..n).map(|i| (i, i)).collect();
        // remove ~75% in scattered order, forcing many merges
        for q in 0..n {
            if !splitmix(q).is_multiple_of(4) {
                let id = splitmix(q ^ 9) % n;
                t.remove(id, id);
                reference.remove(&(id, id));
            }
        }
        assert_eq!(t.len(), reference.len());
        assert_eq!(tree_range(&t, 0, n - 1), ref_range(&reference, 0, n - 1));
    }

    #[test]
    fn churn_stays_compact() {
        // build, then ~100% delete+reinsert turnover; assert no memory bloat (bytes/entry stays tight).
        let w = 20_000u64;
        let mut t = CowBTree::from_sorted(&(0..w).map(|i| (i, i)).collect::<Vec<_>>());
        let bytes_per = |t: &CowBTree| -> f64 {
            let b: usize = t.leaves().iter().map(|(_, l)| l.len()).sum();
            b as f64 / t.len().max(1) as f64
        };
        let build_bpc = bytes_per(&t);

        let mut live: Vec<u64> = (0..w).collect();
        let mut next = w;
        let rounds = 10u64;
        let per = w / rounds;
        for r in 0..rounds {
            for k in 0..per {
                let idx = (splitmix(r * 1009 + k) % live.len() as u64) as usize;
                let id = live.swap_remove(idx);
                t.remove(id, id);
            }
            let adds: Vec<(u64, u64)> = (0..per)
                .map(|_| {
                    let id = next;
                    next += 1;
                    live.push(id);
                    (id, id)
                })
                .collect();
            t.insert_batch(&adds);
        }
        let churn_bpc = bytes_per(&t);
        assert_eq!(t.len(), live.len());
        // tight build ≈ 16 B + header; no-bloat ⇒ stays well under 1.5× after full turnover
        assert!(
            churn_bpc < build_bpc * 1.5,
            "bloat under churn: {build_bpc:.1} → {churn_bpc:.1} B/entry"
        );
        // and reads are still correct
        let reference: BTreeSet<(u64, u64)> = live.iter().map(|&id| (id, id)).collect();
        assert_eq!(
            tree_range(&t, 0, u64::MAX),
            ref_range(&reference, 0, u64::MAX)
        );
    }

    #[test]
    fn clone_is_an_isolated_snapshot() {
        // a clone (snapshot) is unaffected by later mutations of the original — the MVCC property.
        let mut t = CowBTree::from_sorted(&(0..1_000u64).map(|i| (i, i)).collect::<Vec<_>>());
        let snapshot = t.clone();
        for i in 0..1_000u64 {
            t.remove(i, i);
            t.insert(10_000 + i, 10_000 + i);
        }
        // snapshot still sees the original contents; the live tree sees the new ones
        assert_eq!(snapshot.len(), 1_000);
        assert_eq!(snapshot.point(500).collect::<Vec<_>>(), vec![500]);
        assert_eq!(t.point(500).count(), 0);
        assert_eq!(t.point(10_500).collect::<Vec<_>>(), vec![10_500]);
    }

    #[test]
    fn snapshot_is_isolated_from_a_concurrent_writer() {
        // The MVCC property under real concurrency: a reader iterating a snapshot on one thread must keep
        // seeing the snapshot's contents while a writer churns the *shared* tree on another. Every page the
        // two share has `Arc` strong-count >= 2, so the writer's `Arc::make_mut` copies it rather than
        // mutating in place — the reader can never observe a write it shouldn't, and the writer only ever
        // *reads* a shared page (to clone it), so there is no write-while-read race. A bug that mutated a
        // shared page in place would surface here as a changed/garbled read.
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let n = 4_000u64;
        let base = CowBTree::from_sorted(&(0..n).map(|i| (i, i)).collect::<Vec<_>>());
        let snapshot = base.clone(); // shares every page with `base`
        let expected: Vec<u64> = (0..n).collect();

        let done = Arc::new(AtomicBool::new(false));
        let done_reader = Arc::clone(&done);
        let reader = thread::spawn(move || {
            // Re-read the whole snapshot for as long as the writer is mutating; it must never change.
            let mut reads = 0u64;
            while !done_reader.load(Ordering::Relaxed) {
                assert_eq!(
                    snapshot.range(0, u64::MAX).collect::<Vec<_>>(),
                    expected,
                    "snapshot observed a concurrent writer's mutation"
                );
                reads += 1;
            }
            assert_eq!(snapshot.range(0, u64::MAX).collect::<Vec<_>>(), expected);
            reads
        });

        // Churn the shared tree hard while the reader runs: delete every original entry, then insert a
        // disjoint range. Each step path-copies through pages the snapshot still holds — exercising
        // `make_mut`'s copy-on-write against a live reader.
        let mut writer = base;
        for i in 0..n {
            writer.remove(i, i);
        }
        for i in n..2 * n {
            writer.insert(i, i);
        }
        done.store(true, Ordering::Relaxed);

        let reads = reader.join().unwrap();
        assert!(reads > 0, "reader thread never completed a read");
        assert_eq!(
            writer.range(0, u64::MAX).collect::<Vec<_>>(),
            (n..2 * n).collect::<Vec<_>>(),
            "the writer's own tree should reflect all the churn"
        );
    }

    #[test]
    fn leaf_bytes_round_trip_through_a_byte_store() {
        // a leaf blob round-trips through a byte store verbatim: the in-memory form is the serialized form.
        let t = CowBTree::from_sorted(&(0..2_000u64).map(|i| (i, i)).collect::<Vec<_>>());
        // store the format tag + raw bytes
        let store: Vec<(LeafFormat, Vec<u8>)> =
            t.leaves().iter().map(|(f, b)| (*f, b.to_vec())).collect();
        // re-read the docs for a key directly from a stored leaf — proving the bytes are usable as-is.
        // find the leaf containing key 1234 by its min key, then scan it.
        let want = 1234u64;
        let mut found = None;
        for (fmt, blob) in &store {
            let leaf = Leaf::from_parts(*fmt, Arc::from(blob.as_slice())); // re-wrap — proving they're usable as-is
            if leaf.count() > 0 && leaf.key(0) <= want && leaf.key(leaf.count() - 1) >= want {
                let i = leaf.lower_bound(want);
                if i < leaf.count() && leaf.key(i) == want {
                    found = Some(leaf.doc(i));
                }
            }
        }
        assert_eq!(found, Some(1234));
    }

    #[test]
    fn leaf_format_roundtrip() {
        // For each data shape: build a leaf, assert it decodes back to the exact input, reads each entry
        // consistently, round-trips through bytes()/from_parts, and selected the expected in-RAM type.
        // Covers AoS, compact-without-dedup, and compact-with-dedup (the three encode paths) — now three
        // distinct in-RAM types, though only two serialized formats (index presence lives in the buffer).

        /// The expected in-RAM leaf type for a data shape (`None` ⇒ don't check, e.g. the empty leaf).
        enum Want {
            Aos,
            Compact,
            CompactIndexed,
        }

        fn check(
            pairs: Vec<(u64, u64)>,
            want: Option<Want>,
        ) {
            let leaf = Leaf::from_pairs(&pairs);
            assert_eq!(leaf.count(), pairs.len(), "count for {pairs:?}");
            assert_eq!(leaf.to_pairs(), pairs, "to_pairs for {pairs:?}");
            for (i, &(k, d)) in pairs.iter().enumerate() {
                assert_eq!(leaf.key(i), k, "key[{i}] of {pairs:?}");
                assert_eq!(leaf.doc(i), d, "doc[{i}] of {pairs:?}");
            }
            let blob = leaf.bytes();
            assert_eq!(
                Leaf::from_parts(leaf.format(), blob).to_pairs(),
                pairs,
                "bytes round-trip for {pairs:?}"
            );
            match want {
                Some(Want::Aos) => {
                    assert!(matches!(leaf, Leaf::Aos(_)), "expected AoS for {pairs:?}")
                }
                Some(Want::Compact) => assert!(
                    matches!(leaf, Leaf::Compact(_)),
                    "expected compact (no index) for {pairs:?}"
                ),
                Some(Want::CompactIndexed) => assert!(
                    matches!(leaf, Leaf::CompactIndexed(_)),
                    "expected compact-indexed for {pairs:?}"
                ),
                None => {}
            }
        }
        check(vec![], None); // empty ⇒ AoS
        check(vec![(5, 50)], Some(Want::Aos)); // single entry ⇒ AoS (header doesn't amortise)
        // wide, all-distinct values AND docs (both need 8-byte width) ⇒ no compression ⇒ AoS
        check(
            (0..200u64).map(|i| (i << 40, i << 40)).collect(),
            Some(Want::Aos),
        );
        // narrow consecutive values + small ids, all distinct ⇒ compact WITHOUT dedup
        check((0..256u64).map(|i| (i, i)).collect(), Some(Want::Compact));
        // low cardinality (4 distinct wide values × 64 docs) ⇒ compact WITH dedup
        let mut low_card: Vec<(u64, u64)> = Vec::new();
        for v in 0..4u64 {
            for d in 0..64u64 {
                low_card.push((v * 1_000_000, v * 64 + d));
            }
        }
        low_card.sort_unstable();
        check(low_card, Some(Want::CompactIndexed));
        // single value, many docs ⇒ compact (dedup, n_distinct == 1)
        check(
            (0..200u64).map(|d| (42, d)).collect(),
            Some(Want::CompactIndexed),
        );
    }

    #[test]
    fn low_cardinality_parity() {
        // Low-cardinality data forces compact (dedup) leaves; point + range reads must still match a
        // reference, and at least one leaf must actually have chosen the compact format.
        let (n_values, docs_per) = (60u64, 70u64);
        let mut t = CowBTree::new();
        let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
        let mut doc = 0u64;
        for v in 0..n_values {
            let key = v * 1_000;
            for _ in 0..docs_per {
                t.insert(key, doc);
                reference.insert((key, doc));
                doc += 1;
            }
        }
        assert_eq!(t.len(), reference.len());
        assert!(
            t.leaves().iter().any(|(f, _)| *f == LeafFormat::Compact),
            "expected a compact leaf"
        );
        for v in 0..n_values {
            let k = v * 1_000;
            assert_eq!(
                tree_range(&t, k, k),
                ref_range(&reference, k, k),
                "point {k}"
            );
        }
        assert_eq!(
            tree_range(&t, 0, u64::MAX),
            ref_range(&reference, 0, u64::MAX),
            "full range"
        );
        assert_eq!(
            tree_range(&t, 5_000, 25_000),
            ref_range(&reference, 5_000, 25_000),
            "sub range"
        );
    }

    #[test]
    fn format_migration() {
        // A leaf starts AoS (wide, all-distinct) and migrates to compact as low-cardinality data arrives;
        // reads stay correct across the mix. Exercises the cursor on a tree holding both formats.
        let mut t = CowBTree::new();
        let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
        for i in 0..120u64 {
            let (k, d) = (i << 40, i << 40); // wide value + wide doc, all distinct ⇒ AoS
            t.insert(k, d);
            reference.insert((k, d));
        }
        assert!(
            t.leaves().iter().any(|(f, _)| *f == LeafFormat::Aos),
            "expected AoS leaves initially"
        );
        for j in 0..500u64 {
            let k = (j % 3) << 40; // pile many docs onto 3 existing keys ⇒ low cardinality
            let d = 10_000 + j; // narrow docs so the now-low-card leaves clear the compaction margin
            t.insert(k, d);
            reference.insert((k, d));
        }
        assert!(
            t.leaves().iter().any(|(f, _)| *f == LeafFormat::Compact),
            "expected compact leaves after low-card inserts"
        );
        assert_eq!(t.len(), reference.len());
        assert_eq!(
            tree_range(&t, 0, u64::MAX),
            ref_range(&reference, 0, u64::MAX),
            "parity across mixed formats"
        );
    }

    #[test]
    fn bulk_built_thin_tail_deletes_without_panicking() {
        // Regression: a bulk build whose leaf count is `BRANCH_MAX + 1` (= 257) once packed a trailing
        // single-child branch; deleting from its lone child then panicked in `rebalance` (no sibling, so
        // `child_idx - 1` underflowed). Build exactly that shape, delete the tail, and check it stays sane.
        let n = (BRANCH_MAX * LEAF_MAX + 1) as u64; // 257 leaves: 256 full + 1 one-entry leaf
        let pairs: Vec<(u64, u64)> = (0..n).map(|i| (i, i)).collect();
        let mut t = CowBTree::from_sorted(&pairs);
        let mut reference: BTreeSet<(u64, u64)> = pairs.iter().copied().collect();
        assert_eq!(t.len(), reference.len());
        // Delete the tail — the first removal underflows the formerly-singleton branch's child.
        for k in (n - 400..n).rev() {
            t.remove(k, k);
            reference.remove(&(k, k));
        }
        assert_eq!(t.len(), reference.len());
        let got: Vec<u64> = t.range(0, u64::MAX).collect();
        let expected: Vec<u64> = reference.iter().map(|&(_, d)| d).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn aos_splice_insert_remove_edges() {
        // The AoS byte-splice path: insert/remove at the front, back, and middle of a leaf, plus the
        // already-present / absent no-ops. Wide, all-distinct data keeps the leaf in `LeafFormat::Aos`.
        let mut t = CowBTree::new();
        let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
        for i in 1..100u64 {
            let (k, d) = (i << 40, i << 40);
            t.insert(k, d);
            reference.insert((k, d));
        }
        assert!(
            t.leaves().iter().all(|(f, _)| *f == LeafFormat::Aos),
            "wide data should encode as AoS"
        );
        // insert below all, above all, and into the middle
        for (k, d) in [
            (0u64, 0u64),
            (1_000u64 << 40, 1_000 << 40),
            ((50 << 40) + 1, 7),
        ] {
            t.insert(k, d);
            reference.insert((k, d));
        }
        // a duplicate (key, doc) insert is a no-op
        let before = t.len();
        t.insert(0, 0);
        assert_eq!(t.len(), before, "duplicate insert must not grow the leaf");
        // remove front / back / middle, then an absent tuple (no-op)
        for (k, d) in [
            (0u64, 0u64),
            (1_000u64 << 40, 1_000 << 40),
            ((50 << 40) + 1, 7),
        ] {
            t.remove(k, d);
            reference.remove(&(k, d));
        }
        t.remove(99_999 << 40, 1); // absent
        assert_eq!(t.len(), reference.len());
        assert_eq!(
            t.range(0, u64::MAX).collect::<Vec<_>>(),
            reference.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn insert_batch_merge_parity() {
        // `insert_batch` must match a `BTreeSet` reference across the merge paths: small batches (AoS
        // byte-merge), large batches that overflow into several pages (fallback), tuples already present
        // (dedup), and compact source leaves (fallback).
        fn check(
            initial: &[(u64, u64)],
            batch: &[(u64, u64)],
        ) {
            let mut t = CowBTree::from_sorted(initial);
            let mut reference: BTreeSet<(u64, u64)> = initial.iter().copied().collect();
            let mut sorted = batch.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            t.insert_batch(&sorted);
            reference.extend(sorted.iter().copied());
            assert_eq!(t.len(), reference.len(), "len after batch");
            assert_eq!(
                t.range(0, u64::MAX).collect::<Vec<_>>(),
                reference.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
            );
        }
        // wide, all-distinct ⇒ AoS leaves
        let aos: Vec<(u64, u64)> = (0..500u64).map(|i| (i << 40, i << 40)).collect();
        // small batch (byte-merge fast path), including a tuple already present (must de-dup)
        check(
            &aos,
            &[
                (7 << 40, 7 << 40),
                ((3 << 40) + 1, 99),
                ((250 << 40) + 5, 7),
            ],
        );
        // large batch ⇒ overflows leaves into several pages (fallback path)
        let big: Vec<(u64, u64)> = (0..2_000u64).map(|i| ((i << 40) + 1, i)).collect();
        check(&aos, &big);
        // low cardinality ⇒ compact source leaves (fallback path)
        let mut low_card: Vec<(u64, u64)> = (0..2_000u64).map(|i| ((i % 8) * 1_000, i)).collect();
        low_card.sort_unstable();
        low_card.dedup();
        let add: Vec<(u64, u64)> = (0..60u64)
            .map(|i| ((i % 8) * 1_000, 1_000_000 + i))
            .collect();
        check(&low_card, &add);
    }

    #[test]
    fn is_empty_tracks_emptiness() {
        // `is_empty` is O(1) (root-only) and must agree with `len() == 0` through inserts, single
        // removes, and a bulk-built tree drained to nothing (leaves merge back to one empty root).
        let mut t = CowBTree::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        t.insert(5, 50);
        assert!(!t.is_empty());
        assert_eq!(t.len(), 1);
        t.remove(5, 50);
        assert!(t.is_empty(), "removing the last entry empties the tree");

        let pairs: Vec<(u64, u64)> = (0..1_000u64).map(|i| (i, i)).collect();
        let mut big = CowBTree::from_sorted(&pairs);
        assert!(!big.is_empty());
        for &(k, d) in &pairs {
            big.remove(k, d);
        }
        assert!(big.is_empty(), "draining every entry empties the tree");
        assert_eq!(big.len(), 0);
    }

    #[test]
    fn a_reader_reads_the_committed_version_lock_free_during_a_write() {
        // The MVCC question, modeled the way the graph actually does it — NO mutex. The committed version
        // is an immutable `Arc<CowBTree>` (cf. `MvccGraph.graph`); a writer takes a copy-on-write clone (cf.
        // `new_version()`), mutates *that*, and would publish by swapping the pointer (cf. `commit`). Readers
        // never lock — they just clone the committed `Arc` (cf. `read()`).
        //
        // We freeze the writer mid-`make_mut` on its working clone, then have the main thread read the
        // committed version concurrently. The point: the reader is NOT locked out (no mutex), and it still
        // sees the committed version intact — because the working clone shares the committed nodes, so
        // `make_mut` sees refcount ≥ 2 and COPIES rather than mutating in place. The in-place (refcount==1)
        // path is therefore only ever taken on the writer's own private, not-yet-published nodes, which no
        // reader can reach. That's why MVCC needs no lock between a reader's clone and a writer's mutation.
        use super::make_mut_gate;
        use std::sync::Arc;
        use std::sync::atomic::Ordering::SeqCst;
        use std::thread;

        // 1000 entries ⇒ depth-2 tree, so the insert hits exactly one branch `make_mut` and parks once.
        let pairs: Vec<(u64, u64)> = (0..1000u64).map(|i| (i, i)).collect();
        let committed = Arc::new(CowBTree::from_sorted(&pairs)); // the immutable committed version (V1)
        let reader_view = Arc::clone(&committed);

        make_mut_gate::PARKED.store(false, SeqCst);
        make_mut_gate::RELEASE.store(false, SeqCst);

        let writer = thread::spawn({
            let base = Arc::clone(&committed);
            move || {
                let mut working = (*base).clone(); // new_version(): a CoW clone sharing V1's nodes
                working.insert(make_mut_gate::KEY, 1); // make_mut on the SHARED root copies, then parks
                working // the new version that `commit` would publish
            }
        });

        while !make_mut_gate::PARKED.load(SeqCst) {
            std::hint::spin_loop(); // wait until the writer is frozen mid-`make_mut`
        }

        // Writer is frozen mid-mutation on its working clone. With no lock, the reader clones the committed
        // version and reads it — and must see V1 intact: 1000 entries, no sentinel leaked in.
        let snapshot = (*reader_view).clone();
        let docs: Vec<u64> = snapshot.range(0, u64::MAX).collect();
        assert_eq!(
            docs.len(),
            1000,
            "reader observed a writer's in-flight mutation"
        );
        assert!(
            snapshot.point(make_mut_gate::KEY).next().is_none(),
            "the writer's uncommitted insert leaked into the committed snapshot",
        );

        make_mut_gate::RELEASE.store(true, SeqCst); // let the writer finish
        let published = writer.join().unwrap();

        assert_eq!(
            published.len(),
            1001,
            "the new version must hold the insert"
        );
        assert_eq!(
            committed.len(),
            1000,
            "the committed version must be untouched"
        );
    }

    #[test]
    fn compact_splice_arms_parity() {
        // Exercise every packed-mutation arm explicitly on a compact leaf and check parity vs a reference,
        // including that the leaf *stays* compact (proving the in-place splice fired, not a rebuild to AoS).
        // Build a low-cardinality indexed leaf directly (5 distinct values, narrow docs ⇒ small widths).
        // `from_sorted` builds it via `from_pairs`, which selects the compact format; incremental inserts
        // would instead stay AoS until a split, so we seed the compact page up front.
        let mut doc = 0u64;
        let mut pairs: Vec<(u64, u64)> = Vec::new();
        for v in 0..5u64 {
            for _ in 0..30 {
                pairs.push((v * 100, doc));
                doc += 1;
            }
        }
        let mut t = CowBTree::from_sorted(&pairs);
        let mut reference: BTreeSet<(u64, u64)> = pairs.iter().copied().collect();
        let is_compact = |t: &CowBTree| t.leaves().iter().any(|(f, _)| *f == LeafFormat::Compact);
        assert!(is_compact(&t), "setup should be compact");
        let check = |t: &CowBTree, r: &BTreeSet<(u64, u64)>| {
            assert_eq!(t.len(), r.len());
            // doc parity via the cursor, *and* full (key, doc) parity decoded from the leaf bytes
            assert_eq!(tree_range(t, 0, u64::MAX), ref_range(r, 0, u64::MAX));
            assert_eq!(tree_pairs(t), r.iter().copied().collect::<Vec<_>>());
        };

        // insert: existing distinct value (no distinct-table change), new doc within width.
        t.insert(200, doc);
        reference.insert((200, doc));
        doc += 1;
        assert!(
            is_compact(&t),
            "existing-distinct insert must stay compact (spliced)"
        );
        check(&t, &reference);

        // insert: new distinct value within the current widths (distinct-table grow + index remap).
        t.insert(250, doc);
        reference.insert((250, doc));
        doc += 1;
        assert!(
            is_compact(&t),
            "new-distinct insert must stay compact (spliced)"
        );
        check(&t, &reference);

        // insert: a present tuple ((0, 0) was inserted in setup) is a no-op; a genuinely-new one grows by one.
        let before = t.len();
        t.insert(0, 0);
        assert_eq!(t.len(), before, "present insert must be a no-op");
        t.insert(300, doc); // new doc on an existing distinct value
        reference.insert((300, doc));
        check(&t, &reference);

        // remove: an existing entry (index byte + doc cell cut; orphan distinct slot left behind).
        t.remove(100, 30); // (v=1 first doc)
        reference.remove(&(100, 30));
        check(&t, &reference);

        // merge_batch through the packed indexed merge: existing + brand-new distinct values, plus a present
        // tuple (dedup). Docs stay < 256 so the doc width does not widen ⇒ the packed path fires (not rebuild).
        let mut batch: Vec<(u64, u64)> = vec![
            (200, 200), // existing distinct value, fresh doc
            (250, 201), // existing distinct value (added above)
            (175, 202), // new distinct value, between existing
            (425, 203), // new distinct value, above all
            (0, 0),     // already present ⇒ must dedup
        ];
        batch.sort_unstable();
        t.insert_batch(&batch);
        for &p in &batch {
            reference.insert(p);
        }
        check(&t, &reference);

        // param-grow fallback: a key below the current min forces a rebuild (re-picks min/widths). Result
        // must still be correct (the format may change here — we only assert contents).
        t.insert(7, 12_345_678);
        reference.insert((7, 12_345_678));
        check(&t, &reference);
    }

    #[test]
    fn no_index_compact_splice_arms() {
        // The no-index compact page (distinct == count): narrow, all-distinct data selects it via
        // `from_sorted`. Exercise splice insert (new value + value collision), remove, and merge, asserting
        // full (key, doc) parity decoded from the bytes. (The indexed arms are covered separately.)
        let pairs: Vec<(u64, u64)> = (0..200u64).map(|i| (i, i)).collect(); // narrow, all-distinct
        let mut t = CowBTree::from_sorted(&pairs);
        let mut reference: BTreeSet<(u64, u64)> = pairs.iter().copied().collect();
        let no_index = |t: &CowBTree| {
            t.leaves()
                .into_iter()
                .all(|(f, b)| matches!(Leaf::from_parts(f, b), Leaf::Compact(_)))
        };
        assert!(
            no_index(&t),
            "narrow all-distinct data should be no-index compact"
        );
        let check = |t: &CowBTree, r: &BTreeSet<(u64, u64)>| {
            assert_eq!(t.len(), r.len());
            assert_eq!(tree_pairs(t), r.iter().copied().collect::<Vec<_>>());
        };
        // new value (fits widths) ⇒ stays no-index.
        t.insert(250, 250);
        reference.insert((250, 250));
        assert!(
            no_index(&t),
            "new narrow value must stay no-index (spliced)"
        );
        check(&t, &reference);
        // value collision (key 50 already present, fresh narrow doc) ⇒ stored per-entry, still no-index.
        t.insert(50, 251);
        reference.insert((50, 251));
        assert!(
            no_index(&t),
            "value collision must stay no-index (stored per-entry)"
        );
        check(&t, &reference);
        // remove front / middle / back.
        for kd in [(0u64, 0u64), (100, 100), (250, 250)] {
            t.remove(kd.0, kd.1);
            reference.remove(&kd);
            check(&t, &reference);
        }
        // merge a narrow batch: new value, value collision, and a present tuple (dedup). All fit widths ⇒
        // packed no-index merge fires.
        let mut batch = vec![(10, 252), (50, 50), (190, 253), (175, 254)];
        batch.sort_unstable();
        t.insert_batch(&batch);
        for &p in &batch {
            reference.insert(p);
        }
        check(&t, &reference);
        // a widening doc forces the rebuild fallback (contents still correct; format may change).
        t.insert(5, 9_000_000);
        reference.insert((5, 9_000_000));
        check(&t, &reference);
    }

    #[test]
    fn block_copy_merge_parity() {
        // Batches whose keys are all existing distinct values, into a large indexed leaf, take the block-copy
        // merge path (gallop + memcpy, any size). Must match a reference, with a within-batch dup and a
        // leaf-present tuple both dropped.
        let mut seed: Vec<(u64, u64)> = Vec::new();
        for v in 0..4u64 {
            for d in 0..50u64 {
                seed.push((v * 1_000, d)); // 200 entries, 4 distinct, narrow docs
            }
        }
        let mut t = CowBTree::from_sorted(&seed);
        let mut reference: BTreeSet<(u64, u64)> = seed.iter().copied().collect();
        assert!(
            t.leaves()
                .into_iter()
                .all(|(f, b)| matches!(Leaf::from_parts(f, b), Leaf::CompactIndexed(_))),
            "setup should be a single indexed leaf"
        );
        // B = 8 (< 200/4), keys all existing distinct (0/1000/2000/3000), docs fit. (2000,102) is a
        // within-batch dup; (1000,0) and (0,0) are already in the leaf — all three must be dropped.
        let mut batch = vec![
            (0, 100),
            (1_000, 101),
            (2_000, 102),
            (2_000, 102),
            (3_000, 103),
            (1_000, 0),
            (0, 0),
            (3_000, 104),
        ];
        batch.sort_unstable();
        t.insert_batch(&batch);
        for &p in &batch {
            reference.insert(p);
        }
        assert_eq!(t.len(), reference.len());
        assert_eq!(
            tree_pairs(&t),
            reference.iter().copied().collect::<Vec<_>>()
        );
        assert!(
            t.leaves()
                .into_iter()
                .all(|(f, b)| matches!(Leaf::from_parts(f, b), Leaf::CompactIndexed(_))),
            "block-copy merge must keep the leaf indexed"
        );
        // A *large* all-existing-distinct batch (no size guard now — galloping keeps block-copy ≥ the walk).
        // 40 entries into the ~205-entry leaf stays under LEAF_MAX and on the block-copy path.
        let mut big: Vec<(u64, u64)> = (0..40u64).map(|i| ((i % 4) * 1_000, 500 + i)).collect();
        big.sort_unstable();
        t.insert_batch(&big);
        for &p in &big {
            reference.insert(p);
        }
        assert_eq!(t.len(), reference.len());
        assert_eq!(
            tree_pairs(&t),
            reference.iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn packed_compact_mutation_fuzz() {
        // Randomized insert / remove / insert_batch over mostly-low-cardinality data so the packed splice +
        // merge fast paths fire, with occasional wide keys (AoS) and widening docs (rebuild fallback). The
        // tree must equal a BTreeSet reference throughout — the ground-truth check for every packed arm and
        // their interaction with splits, merges, and format transitions.
        // Seed with a low-cardinality dataset built via `from_sorted` so leaves start *compact* (incremental
        // inserts would stay AoS until a split, leaving the packed splice arms barely exercised). The random
        // ops below then hit `splice_insert` / `merge` / `splice_remove` on compact pages from step 0.
        let mut seed: Vec<(u64, u64)> = Vec::new();
        for v in 0..40u64 {
            for d in 0..120u64 {
                seed.push((v * 1_000, d));
            }
        }
        let mut t = CowBTree::from_sorted(&seed);
        let mut reference: BTreeSet<(u64, u64)> = seed.iter().copied().collect();
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            s = splitmix(s);
            s
        };
        for step in 0..40_000u64 {
            // Low-card key most of the time (⇒ compact leaves); an occasional wide key (⇒ AoS). The 500
            // step interleaves with the 1000-step seed, so new distinct values land in the *middle* of a
            // leaf's distinct table (exercising the index remap), not only at the end.
            let gen_key = |n: u64| {
                if n % 16 == 0 { n >> 8 } else { (n % 100) * 500 }
            };
            let gen_doc = |n: u64| if n % 8 == 0 { n >> 20 } else { n % 800 }; // narrow + widening docs
            match next() % 10 {
                0..=5 => {
                    let (k, d) = (gen_key(next()), gen_doc(next()));
                    t.insert(k, d);
                    reference.insert((k, d));
                }
                6..=7 => {
                    let (k, d) = (gen_key(next()), gen_doc(next()));
                    t.remove(k, d);
                    reference.remove(&(k, d));
                }
                _ => {
                    let bn = (next() % 12) as usize;
                    let mut batch: Vec<(u64, u64)> = Vec::with_capacity(bn);
                    for _ in 0..bn {
                        batch.push((gen_key(next()), gen_doc(next())));
                    }
                    batch.sort_unstable();
                    batch.dedup();
                    t.insert_batch(&batch);
                    reference.extend(batch.iter().copied());
                }
            }
            if step % 53 == 0 {
                assert_eq!(t.len(), reference.len(), "len mismatch at step {step}");
                // doc parity via the cursor, *and* full (key, doc) parity decoded straight from the leaf
                // bytes — the latter verifies the value column the packed splice/merge code maintains (a
                // full-range cursor scan sets `whole` and skips key reads on interior leaves).
                assert_eq!(
                    t.range(0, u64::MAX).collect::<Vec<_>>(),
                    reference.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
                    "doc contents mismatch at step {step}"
                );
                assert_eq!(
                    tree_pairs(&t),
                    reference.iter().copied().collect::<Vec<_>>(),
                    "(key, doc) contents mismatch at step {step}"
                );
            }
        }
        assert_eq!(
            tree_pairs(&t),
            reference.iter().copied().collect::<Vec<_>>(),
            "final (key, doc) contents"
        );
    }
}
