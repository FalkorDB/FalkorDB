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
/// page holds ≤ 256 entries whether they encode to ≈ 4 KiB ([`FMT_AOS`], 16 B/tuple) or less ([`FMT_COMPACT`]).
///
/// `256` is also load-bearing for the compact format: its dedup index stores one slot per entry as a `u8`,
/// which only fits because `count ≤ LEAF_MAX ≤ 256` (see [`Leaf::build_compact`]). **Raising this past 256
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
/// Byte stride of one `(key, doc)` entry in the [`FMT_AOS`] layout.
const STRIDE: usize = 2 * FIELD;

/// Leaf format, stored as the **first byte** of every page so a leaf is fully self-describing — a tree
/// may hold a mix of both, each read by dispatching on its own tag.
const FMT_AOS: u8 = 0; // `[0][(key:8, doc:8) × n]` — contiguous array-of-structs tuples, 16 B/entry.
const FMT_COMPACT: u8 = 1; // `[1][header][values][index?][docs]` — see [`Leaf::build_compact`].

// Byte offsets of each field within the `FMT_COMPACT` header (the format tag occupies byte 0).
const ENTRY_COUNT_OFFSET: usize = 1; // u16: number of entries
const MIN_VALUE_OFFSET: usize = 3; // u64: minimum value in the leaf (values are stored as deltas from it)
const VALUE_WIDTH_OFFSET: usize = 11; // u8: bytes per value delta (a power of two)
const DISTINCT_COUNT_OFFSET: usize = 12; // u16: number of distinct values
const DOC_WIDTH_OFFSET: usize = 14; // u8: bytes per doc (a power of two)
const BODY_OFFSET: usize = 15; // first byte of the value / index / doc bodies

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

/// A leaf page held in an `Arc<[u8]>`, **self-describing** via its first (tag) byte:
///
/// - [`FMT_AOS`] — `[0][(key, doc) × n]`, each field a little-endian `u64` (16 B/entry). Key beside its
///   doc (array-of-structs) so a range scan reads one sequential stream.
/// - [`FMT_COMPACT`] — values delta-encoded from the leaf minimum at a per-leaf power-of-two width,
///   distinct values de-duplicated when that pays (an index, skipped when every value is distinct), and
///   docs stored at their own minimal width (decoded via [`CompactView`]). Chosen per leaf only when it
///   saves ≥ 8 B/entry (half an AoS entry).
///
/// Either way the blob *is* the leaf's serialized form: cloning is an `Arc::clone` and re-adopting bytes
/// is [`Leaf::from_bytes`] — a copy, never a (de)serialization. This newtype is the one place that knows
/// the byte layout; every reader dispatches on the tag, so a tree may freely mix the two formats.
#[derive(Clone)]
struct Leaf(Arc<[u8]>);

/// A decoded view over a [`FMT_COMPACT`] page: the fixed-size header parsed **once** so the per-entry
/// accessors don't re-read offsets on every call. Borrows the page bytes; constructing it is a handful of
/// field loads and no allocation. Keeping the compact decode here is why [`Leaf`]'s own methods stay a
/// thin two-arm dispatch on the tag rather than repeating the header arithmetic.
struct CompactView<'a> {
    bytes: &'a [u8],
    entry_count: usize,
    distinct_count: usize,
    min_value: u64,
    value_width: usize,
    doc_width: usize,
    /// `true` when distinct values are de-duplicated (a per-entry index is present), i.e.
    /// `distinct_count < entry_count`; when `false`, values are stored one-per-entry with no index.
    deduplicated: bool,
    /// Byte offset of the per-entry distinct-value index (only present when `deduplicated`).
    index_offset: usize,
    /// Byte offset of the docs array.
    docs_offset: usize,
}

impl<'a> CompactView<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        let entry_count = read_u16(bytes, ENTRY_COUNT_OFFSET) as usize;
        let distinct_count = read_u16(bytes, DISTINCT_COUNT_OFFSET) as usize;
        let value_width = bytes[VALUE_WIDTH_OFFSET] as usize;
        let doc_width = bytes[DOC_WIDTH_OFFSET] as usize;
        let deduplicated = distinct_count < entry_count;
        // The values occupy `distinct_count * value_width` bytes (which equals `entry_count * value_width`
        // when not de-duplicated, since then `distinct_count == entry_count`); a `entry_count`-byte index
        // follows only when de-duplicated, then the docs.
        let index_offset = BODY_OFFSET + distinct_count * value_width;
        let docs_offset = index_offset + if deduplicated { entry_count } else { 0 };
        Self {
            bytes,
            entry_count,
            distinct_count,
            min_value: read_u64(bytes, MIN_VALUE_OFFSET),
            value_width,
            doc_width,
            deduplicated,
            index_offset,
            docs_offset,
        }
    }

    /// Value (key) of entry `i`.
    fn key(
        &self,
        i: usize,
    ) -> u64 {
        // Entry `i`'s value is `distinct[index[i]]` when de-duplicated, else stored per-entry (slot == i).
        let slot = if self.deduplicated {
            self.bytes[self.index_offset + i] as usize
        } else {
            i
        };
        self.min_value
            + read_width(
                self.bytes,
                BODY_OFFSET + slot * self.value_width,
                self.value_width,
            )
    }

    /// `(base, stride, width)` of the docs array — the cursor caches this to scan docs without re-reading
    /// the header per entry. Docs are stored contiguously, so the stride equals the width.
    fn doc_layout(&self) -> (usize, usize, usize) {
        (self.docs_offset, self.doc_width, self.doc_width)
    }

    /// Decode every entry to owned `(key, doc)` pairs.
    fn to_pairs(&self) -> Vec<(u64, u64)> {
        let distinct: Vec<u64> = (0..self.distinct_count)
            .map(|j| {
                self.min_value
                    + read_width(
                        self.bytes,
                        BODY_OFFSET + j * self.value_width,
                        self.value_width,
                    )
            })
            .collect();
        (0..self.entry_count)
            .map(|i| {
                let slot = if self.deduplicated {
                    self.bytes[self.index_offset + i] as usize
                } else {
                    i
                };
                let doc = read_width(
                    self.bytes,
                    self.docs_offset + i * self.doc_width,
                    self.doc_width,
                );
                (distinct[slot], doc)
            })
            .collect()
    }
}

impl Leaf {
    /// Number of `(key, doc)` entries.
    fn count(&self) -> usize {
        match self.0[0] {
            FMT_AOS => (self.0.len() - 1) / STRIDE,
            _ => read_u16(&self.0, ENTRY_COUNT_OFFSET) as usize,
        }
    }

    /// Key of entry `i`.
    fn key(
        &self,
        i: usize,
    ) -> u64 {
        match self.0[0] {
            FMT_AOS => read_u64(&self.0, 1 + STRIDE * i),
            _ => CompactView::new(&self.0).key(i),
        }
    }

    /// Doc of entry `i`.
    fn doc(
        &self,
        i: usize,
    ) -> u64 {
        let (base, stride, width) = self.doc_layout();
        read_width(&self.0, base + i * stride, width)
    }

    /// The doc array's `(base, stride, width)` — read once per leaf by the cursor so a scan decodes
    /// docs with no per-entry header re-read.
    fn doc_layout(&self) -> (usize, usize, usize) {
        match self.0[0] {
            FMT_AOS => (1 + FIELD, STRIDE, FIELD),
            _ => CompactView::new(&self.0).doc_layout(),
        }
    }

    /// First entry index whose key is `>= key`. Entries are sorted by `(key, doc)`, so this lands on
    /// the first doc of `key` — what range/point reads need (they scan forward from there). Single
    /// insert/remove instead `binary_search` the decoded pairs, since they need the exact slot.
    fn lower_bound(
        &self,
        key: u64,
    ) -> usize {
        let (mut lo, mut hi) = (0usize, self.count());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.key(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Decode to the owned `(key, doc)` pairs the mutation paths (insert/remove/merge) work in.
    fn to_pairs(&self) -> Vec<(u64, u64)> {
        match self.0[0] {
            FMT_AOS => (0..self.count())
                .map(|i| {
                    (
                        read_u64(&self.0, 1 + STRIDE * i),
                        read_u64(&self.0, 1 + STRIDE * i + FIELD),
                    )
                })
                .collect(),
            _ => CompactView::new(&self.0).to_pairs(),
        }
    }

    /// Build a leaf from sorted `(key, doc)` pairs, choosing [`FMT_AOS`] or [`FMT_COMPACT`] per the data.
    ///
    /// The choice is a closed-form size comparison; the widths are always one of {1, 2, 4, 8}, so the only
    /// free variables are `count` and the data:
    /// - AoS:                `1 + count·STRIDE`
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
    /// lazily in [`Leaf::build_compact`], only when compact is chosen.
    fn from_pairs(pairs: &[(u64, u64)]) -> Self {
        let count = pairs.len();
        if count == 0 {
            return Self::build_aos(pairs);
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
        let aos_size = 1 + count * STRIDE;
        if compact_size + COMPACT_MIN_SAVING_BPE * count <= aos_size {
            Self::build_compact(
                pairs,
                count,
                distinct_count,
                min_value,
                value_width,
                doc_width,
                deduplicated,
            )
        } else {
            Self::build_aos(pairs)
        }
    }

    /// Encode the `[FMT_AOS][(key, doc) × n]` layout.
    fn build_aos(pairs: &[(u64, u64)]) -> Self {
        let mut buf = Vec::with_capacity(1 + pairs.len() * STRIDE);
        buf.push(FMT_AOS);
        for &(key, doc) in pairs {
            buf.extend_from_slice(&key.to_le_bytes());
            buf.extend_from_slice(&doc.to_le_bytes());
        }
        Self(Arc::from(buf.as_slice()))
    }

    /// Encode the `FMT_COMPACT` layout (the caller has already chosen it and computed the widths). The
    /// distinct table and per-entry index are (re)built here in one in-cache pass over the ≤ `LEAF_MAX`
    /// entries — deliberately *not* carried over from [`Leaf::from_pairs`], since materialising them there
    /// measurably slowed the far more common AoS path while saving nothing here (the byte writes dominate).
    fn build_compact(
        pairs: &[(u64, u64)],
        count: usize,
        distinct_count: usize,
        min_value: u64,
        value_width: usize,
        doc_width: usize,
        deduplicated: bool,
    ) -> Self {
        let mut distinct: Vec<u64> = Vec::with_capacity(distinct_count);
        let mut index: Vec<u8> = Vec::with_capacity(if deduplicated { count } else { 0 });
        for &(key, _) in pairs {
            if distinct.last() != Some(&key) {
                distinct.push(key);
            }
            if deduplicated {
                index.push((distinct.len() - 1) as u8);
            }
        }
        debug_assert_eq!(distinct.len(), distinct_count);
        let mut buf = Vec::with_capacity(
            BODY_OFFSET + distinct_count * value_width + index.len() + count * doc_width,
        );
        buf.push(FMT_COMPACT);
        buf.extend_from_slice(&(count as u16).to_le_bytes());
        buf.extend_from_slice(&min_value.to_le_bytes());
        buf.push(value_width as u8);
        buf.extend_from_slice(&(distinct_count as u16).to_le_bytes());
        buf.push(doc_width as u8);
        for &value in &distinct {
            buf.extend_from_slice(&(value - min_value).to_le_bytes()[..value_width]);
        }
        if deduplicated {
            // The index slot fits a `u8` only because dedup ⇒ distinct_count < count, and count ≤
            // `LEAF_MAX` (= 256) ⇒ slot ≤ 254. `LEAF_MAX`'s doc notes this ceiling must not exceed 256.
            debug_assert!(distinct_count < count);
            buf.extend_from_slice(&index);
        }
        for &(_, doc) in pairs {
            buf.extend_from_slice(&doc.to_le_bytes()[..doc_width]);
        }
        Self(Arc::from(buf.as_slice()))
    }

    /// Re-adopt a leaf from its serialized bytes — the counterpart of [`Leaf::bytes`] and the read side
    /// of [`CowBTree::leaves`]. The bytes are the page verbatim, so this is a copy, never a decode. It
    /// **trusts** its input (validating arbitrary bytes is the caller's responsibility); currently
    /// exercised only by the byte round-trip test.
    #[allow(dead_code)]
    fn from_bytes(b: &[u8]) -> Self {
        Self(Arc::from(b))
    }

    /// The leaf's serialized bytes — an `Arc` bump, no serialization step.
    fn bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.0)
    }

    /// The raw page bytes (for the cursor's cached doc read).
    fn raw(&self) -> &[u8] {
        &self.0
    }

    /// First entry whose `(key, doc)` is `>= (key, doc)` — the slot a single insert/remove targets.
    fn lower_bound_entry(
        &self,
        key: u64,
        doc: u64,
    ) -> usize {
        let (mut lo, mut hi) = (0usize, self.count());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (self.key(mid), self.doc(mid)) < (key, doc) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Insert one `(key, doc)`: the replacement leaf if it still fits, a [`LeafInsert::Split`] if it
    /// overflowed, or `None` if the tuple is already present. An [`FMT_AOS`] leaf that still fits is
    /// **spliced directly in its bytes** — no decode/re-encode. The overflow case and every [`FMT_COMPACT`]
    /// leaf go through [`Leaf::to_pairs`] + [`Leaf::from_pairs`], which also re-selects the format (so an
    /// AoS→compact migration happens at the next split, not on every insert).
    fn insert(
        &self,
        key: u64,
        doc: u64,
    ) -> Option<LeafInsert> {
        if self.0[0] == FMT_AOS && self.count() < LEAF_MAX {
            // AoS fast path: it still fits (count < LEAF_MAX ⇒ count + 1 ≤ LEAF_MAX), so splice the 16-byte
            // tuple into the page — memcpy of prefix + tuple + suffix, one allocation, no full decode.
            let pos = self.lower_bound_entry(key, doc);
            if pos < self.count() && self.key(pos) == key && self.doc(pos) == doc {
                return None; // already present
            }
            let cut = 1 + pos * STRIDE;
            let mut buf = Vec::with_capacity(self.0.len() + STRIDE);
            buf.extend_from_slice(&self.0[..cut]);
            buf.extend_from_slice(&key.to_le_bytes());
            buf.extend_from_slice(&doc.to_le_bytes());
            buf.extend_from_slice(&self.0[cut..]);
            return Some(LeafInsert::Fit(Leaf(Arc::from(buf.as_slice()))));
        }
        // Compact leaf, or an AoS leaf that would overflow: decode, insert, rebuild (re-selecting the format).
        let mut pairs = self.to_pairs();
        let Err(pos) = pairs.binary_search(&(key, doc)) else {
            return None; // already present
        };
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
    /// if the tuple is absent. An [`FMT_AOS`] leaf is **spliced directly** (the 16-byte tuple is cut out); a
    /// [`FMT_COMPACT`] leaf is rebuilt via [`Leaf::from_pairs`]. Re-compacting an AoS leaf happens at its
    /// next merge (which rebuilds through [`Leaf::from_pairs`]).
    fn remove(
        &self,
        key: u64,
        doc: u64,
    ) -> Option<(Leaf, bool)> {
        if self.0[0] == FMT_AOS {
            let count = self.count();
            let pos = self.lower_bound_entry(key, doc);
            if pos >= count || self.key(pos) != key || self.doc(pos) != doc {
                return None; // absent
            }
            let cut = 1 + pos * STRIDE;
            let mut buf = Vec::with_capacity(self.0.len() - STRIDE);
            buf.extend_from_slice(&self.0[..cut]);
            buf.extend_from_slice(&self.0[cut + STRIDE..]);
            return Some((Leaf(Arc::from(buf.as_slice())), count - 1 < LEAF_MIN));
        }
        let mut pairs = self.to_pairs();
        let pos = pairs.binary_search(&(key, doc)).ok()?; // `None` ⇒ absent
        pairs.remove(pos);
        Some((Self::from_pairs(&pairs), pairs.len() < LEAF_MIN))
    }

    /// Apply a sorted `batch` (all of it routing into this leaf) and return the replacement leaf page(s).
    /// An [`FMT_AOS`] leaf whose result still fits one page is **byte-merged** (see [`Leaf::merge_batch_aos`]);
    /// a [`FMT_COMPACT`] leaf, or one whose result overflows into several pages, decodes, merges, de-dups,
    /// and re-chunks through [`Leaf::from_pairs`] (re-selecting the format per chunk).
    fn merge_batch(
        &self,
        batch: &[(u64, u64)],
    ) -> Vec<Leaf> {
        if self.0[0] == FMT_AOS && self.count() + batch.len() <= LEAF_MAX {
            return vec![self.merge_batch_aos(batch)];
        }
        // The leaf's entries and the batch are each already sorted, so two-pointer **merge** them (with
        // exact-dup removal) rather than concat + sort — a swept micro-benchmark put this ~3.5x ahead of
        // `sort()` and ~5x ahead of `sort_unstable()` (pdqsort can't exploit pre-sorted runs). Then re-chunk.
        let merged = merge_sorted(&self.to_pairs(), batch);
        merged.chunks(LEAF_MAX).map(Self::from_pairs).collect()
    }

    /// Two-pointer merge of `self`'s entries with a sorted `batch` into a single new [`FMT_AOS`] page — no
    /// `Vec<(u64, u64)>`, no sort, one allocation. Exact `(key, doc)` duplicates are dropped (re-adding an
    /// existing tuple is a no-op). The caller guarantees the result fits (`count + batch.len() <= LEAF_MAX`).
    fn merge_batch_aos(
        &self,
        batch: &[(u64, u64)],
    ) -> Leaf {
        let count = self.count();
        let mut buf = Vec::with_capacity(self.0.len() + batch.len() * STRIDE);
        buf.push(FMT_AOS);
        let (mut i, mut j) = (0usize, 0usize);
        let mut last: Option<(u64, u64)> = None;
        while i < count || j < batch.len() {
            let take_leaf =
                i < count && (j >= batch.len() || (self.key(i), self.doc(i)) <= batch[j]);
            let next = if take_leaf {
                let entry = (self.key(i), self.doc(i));
                i += 1;
                entry
            } else {
                let entry = batch[j];
                j += 1;
                entry
            };
            if last == Some(next) {
                continue; // drop an exact `(key, doc)` duplicate
            }
            buf.extend_from_slice(&next.0.to_le_bytes());
            buf.extend_from_slice(&next.1.to_le_bytes());
            last = Some(next);
        }
        Leaf(Arc::from(buf.as_slice()))
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
        match self.children[left_idx].combine(self.seps[left_idx], &self.children[right_idx]) {
            Combined::One(merged) => {
                self.children[left_idx] = merged;
                self.children.remove(right_idx);
                self.seps.remove(left_idx);
            }
            Combined::Two(left, sep, right) => {
                self.children[left_idx] = left;
                self.children[right_idx] = right;
                self.seps[left_idx] = sep;
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
                // `rest` is the suffix of `batch` not yet assigned to an earlier child; it shrinks
                // from the front as we walk the children left to right.
                let mut rest = batch;
                for (child_idx, child) in branch.children.iter().enumerate() {
                    // Each child owns keys strictly below its right separator; the last child (which
                    // has no separator) owns everything remaining. `rest` is sorted, so a binary
                    // `partition_point` finds how many leading entries fall into this child.
                    let child_upper = branch
                        .seps
                        .get(child_idx)
                        .copied()
                        .unwrap_or((u64::MAX, u64::MAX));
                    let take = rest.partition_point(|&entry| entry < child_upper);
                    let (for_child, remaining) = rest.split_at(take);
                    rest = remaining;
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

    /// Whether the tree holds no tuples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All leaf page blobs in key order. Each is a self-contained byte page — the bytes are the
    /// serialized form, so writing the tree out to a byte store needs no further encoding.
    pub fn leaves(&self) -> Vec<Arc<[u8]>> {
        fn walk(
            node: &Node,
            out: &mut Vec<Arc<[u8]>>,
        ) {
            match node {
                Node::Leaf(leaf) => out.push(leaf.bytes()),
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
    // so `next` decodes docs without recomputing offsets per entry (matters for the compact format).
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

    /// Position the cursor on `leaf` at entry `pos`, caching its count and whether the leaf's last key
    /// is within the upper bound. When it is, every entry from `pos` on qualifies, so [`Iterator::next`]
    /// can emit docs without reading the key or comparing — only a *boundary* leaf needs per-entry checks.
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
            let b: usize = t.leaves().iter().map(|l| l.len()).sum();
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
        let store: Vec<Vec<u8>> = t.leaves().iter().map(|l| l.to_vec()).collect(); // store the raw bytes
        // re-read the docs for a key directly from a stored leaf — proving the bytes are usable as-is.
        // find the leaf containing key 1234 by its min key, then scan it.
        let want = 1234u64;
        let mut found = None;
        for blob in &store {
            let leaf = Leaf::from_bytes(blob); // re-wrap the raw bytes — proving they're usable as-is
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
        // consistently, round-trips through bytes()/from_bytes, and selected the expected format. Covers
        // AoS, compact-without-dedup, and compact-with-dedup (the three encode paths).
        fn check(
            pairs: Vec<(u64, u64)>,
            expect_compact: Option<bool>,
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
                Leaf::from_bytes(&blob).to_pairs(),
                pairs,
                "bytes round-trip for {pairs:?}"
            );
            if let Some(c) = expect_compact {
                assert_eq!(leaf.0[0] == FMT_COMPACT, c, "format tag for {pairs:?}");
            }
        }
        check(vec![], None); // empty ⇒ AoS (just the tag byte)
        check(vec![(5, 50)], Some(false)); // single entry ⇒ AoS (header doesn't amortise)
        // wide, all-distinct values AND docs (both need 8-byte width) ⇒ no compression ⇒ AoS
        check(
            (0..200u64).map(|i| (i << 40, i << 40)).collect(),
            Some(false),
        );
        // narrow consecutive values + small ids, all distinct ⇒ compact WITHOUT dedup
        check((0..256u64).map(|i| (i, i)).collect(), Some(true));
        // low cardinality (4 distinct wide values × 64 docs) ⇒ compact WITH dedup
        let mut low_card: Vec<(u64, u64)> = Vec::new();
        for v in 0..4u64 {
            for d in 0..64u64 {
                low_card.push((v * 1_000_000, v * 64 + d));
            }
        }
        low_card.sort_unstable();
        check(low_card, Some(true));
        // single value, many docs ⇒ compact (dedup, n_distinct == 1)
        check((0..200u64).map(|d| (42, d)).collect(), Some(true));
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
            t.leaves().iter().any(|b| b[0] == FMT_COMPACT),
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
            t.leaves().iter().any(|b| b[0] == FMT_AOS),
            "expected AoS leaves initially"
        );
        for j in 0..500u64 {
            let k = (j % 3) << 40; // pile many docs onto 3 existing keys ⇒ low cardinality
            let d = 10_000 + j; // narrow docs so the now-low-card leaves clear the compaction margin
            t.insert(k, d);
            reference.insert((k, d));
        }
        assert!(
            t.leaves().iter().any(|b| b[0] == FMT_COMPACT),
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
        // already-present / absent no-ops. Wide, all-distinct data keeps the leaf in `FMT_AOS`.
        let mut t = CowBTree::new();
        let mut reference: BTreeSet<(u64, u64)> = BTreeSet::new();
        for i in 1..100u64 {
            let (k, d) = (i << 40, i << 40);
            t.insert(k, d);
            reference.insert((k, d));
        }
        assert!(
            t.leaves().iter().all(|b| b[0] == FMT_AOS),
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
}
