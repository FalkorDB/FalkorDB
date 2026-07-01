//! The compact leaf encoding without a dedup index, plus the shared compact-header machinery
//! (offsets, layout reader, and splice/merge helpers) used by the indexed variant too.

use std::sync::Arc;

use super::super::{read_u64, read_width};
use super::{merge_walk, pow2_bytes_for};

// Byte offsets of each field within the (tag-free) compact header.
pub(super) const ENTRY_COUNT_OFFSET: usize = 0; // u16: number of entries
pub(super) const MIN_VALUE_OFFSET: usize = 2; // u64: minimum value in the leaf (values are stored as deltas from it)
pub(super) const VALUE_WIDTH_OFFSET: usize = 10; // u8: bytes per value delta (a power of two)
pub(super) const DISTINCT_COUNT_OFFSET: usize = 11; // u16: number of distinct values
pub(super) const DOC_WIDTH_OFFSET: usize = 13; // u8: bytes per doc (a power of two)
pub(super) const BODY_OFFSET: usize = 14; // first byte of the value / index / doc bodies

/// Read the little-endian `u16` at byte offset `off`.
pub(super) fn read_u16(
    b: &[u8],
    off: usize,
) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

/// True when the compact page carries a dedup index — fewer distinct values than entries. Lets
/// [`super::Leaf::from_parts`] pick the indexed vs no-index in-RAM type from the buffer alone,
/// keeping the header offsets private to this module.
pub(super) fn is_indexed(bytes: &[u8]) -> bool {
    read_u16(bytes, DISTINCT_COUNT_OFFSET) < read_u16(bytes, ENTRY_COUNT_OFFSET)
}

/// Parsed layout of a compact page (either format), read once from the header for a mutation. A transient
/// write-path helper — offsets only, not cached, and separate from the read-path decode (the cursor and
/// iterators read what they need inline). It removes the repeated header arithmetic from the splice/merge
/// methods below.
pub(super) struct CompactLayout {
    pub(super) count: usize,
    pub(super) min: u64,
    pub(super) value_width: usize,
    pub(super) distinct_count: usize,
    pub(super) doc_width: usize,
    /// First byte of the per-entry index; equals `docs_offset` for a no-index page (which has no index).
    pub(super) index_offset: usize,
    /// First byte of the doc column.
    pub(super) docs_offset: usize,
}

impl CompactLayout {
    pub(super) fn read(bytes: &[u8]) -> Self {
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

/// Write the shared compact header — `count`, `min`, `value_width`, `distinct_count`, `doc_width` — at the
/// offsets above, then assert the body starts at [`BODY_OFFSET`]. Both compact builders open with this, so
/// the read offsets (read straight from the buffer by each variant) and the write order stay in lock-step.
pub(super) fn write_compact_header(
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

/// Append the doc column for an insert that adds one entry at `pos`: `docs[0..pos] | doc | docs[pos..count]`,
/// each existing cell copied verbatim from `bytes` at `layout.docs_offset` / `doc_width`. Shared by all three
/// insert arms (no-index, indexed existing-distinct, indexed new-distinct), which differ only above the docs.
pub(super) fn append_spliced_docs(
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

/// True when `(key, doc)` fits the page's current packing without widening — `key >= min`, the value delta
/// fits `value_width`, and `doc` fits `doc_width`. The precondition for an in-place splice/merge (below)
/// instead of a rebuild that would have to re-pick the widths. Shared by both compact formats (their
/// packing parameters live at the same header offsets).
pub(super) fn packing_fits(
    bytes: &[u8],
    key: u64,
    doc: u64,
) -> bool {
    let min = read_u64(bytes, MIN_VALUE_OFFSET);
    key >= min
        && pow2_bytes_for(key - min) <= bytes[VALUE_WIDTH_OFFSET] as usize
        && pow2_bytes_for(doc) <= bytes[DOC_WIDTH_OFFSET] as usize
}

/// A compact leaf page **without** a dedup index: a tag-free `Arc<[u8]>` of `[header][values][docs]`, with
/// one value delta per entry (`distinct_count == entry_count`). See [`CompactLeaf::build`].
#[derive(Clone)]
pub(crate) struct CompactLeaf(pub(super) Arc<[u8]>);

impl CompactLeaf {
    /// Number of `(key, doc)` entries — read from the header.
    pub(super) fn count(&self) -> usize {
        read_u16(&self.0, ENTRY_COUNT_OFFSET) as usize
    }

    /// Key of entry `i` — `min + delta[i]` (one delta per entry; no index).
    pub(super) fn key(
        &self,
        i: usize,
    ) -> u64 {
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        read_u64(&self.0, MIN_VALUE_OFFSET) + read_width(&self.0, BODY_OFFSET + i * vw, vw)
    }

    /// Doc of entry `i`.
    pub(super) fn doc(
        &self,
        i: usize,
    ) -> u64 {
        let (base, stride, width) = self.doc_layout();
        read_width(&self.0, base + i * stride, width)
    }

    /// The doc array's `(base, stride, width)` — read once per leaf by the cursor. The docs start right
    /// after the `count` value deltas. Docs are stored contiguously, so the stride equals the width.
    pub(super) fn doc_layout(&self) -> (usize, usize, usize) {
        let count = self.count();
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        let dw = self.0[DOC_WIDTH_OFFSET] as usize;
        let docs_offset = BODY_OFFSET + count * vw;
        (docs_offset, dw, dw)
    }

    /// Encode the no-index compact layout `[header][values][docs]` (the caller has already chosen it and
    /// computed the widths): one value delta per entry — no distinct table, no index. `distinct_count`
    /// equals `count` here and is still written to the header (so a reader can tell index-vs-no-index).
    pub(super) fn build(
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

impl CompactLeaf {
    /// Splice one `(key, doc)` into the packed page at entry `pos` — no decode. The caller guarantees
    /// [`packing_fits`], `count < LEAF_MAX`, and that the tuple is absent. A no-index leaf stores one
    /// value per entry, so the value and doc cells are simply inserted (`memcpy` of the surrounding cells);
    /// the leaf stays no-index (a value that collides with an existing one is stored verbatim, and is
    /// de-duplicated only at the next rebuild).
    pub(super) fn splice_insert(
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
    pub(super) fn splice_remove(
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
    pub(super) fn merge(
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
