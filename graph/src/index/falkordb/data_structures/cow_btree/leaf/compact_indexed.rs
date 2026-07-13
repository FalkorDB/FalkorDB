//! The compact leaf encoding with a dedup index (a distinct-value column plus one index byte per
//! entry), reusing the shared compact-header machinery from the sibling [`super::compact`] module.

use std::sync::Arc;

use super::super::{read_u64, read_width};
use super::compact::{
    BODY_OFFSET, CompactLayout, DISTINCT_COUNT_OFFSET, DOC_WIDTH_OFFSET, ENTRY_COUNT_OFFSET,
    MIN_VALUE_OFFSET, VALUE_WIDTH_OFFSET, append_spliced_docs, read_u16, write_compact_header,
};
use super::{merge_walk, partition_point};

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

/// A compact leaf page **with** a dedup index: a tag-free `Arc<[u8]>` of `[header][distinct values][index][docs]`,
/// where `index` is one `u8` per entry pointing into the distinct value column (`distinct_count < entry_count`).
/// See [`CompactIndexedLeaf::build`].
#[derive(Clone)]
pub struct CompactIndexedLeaf(pub(super) Arc<[u8]>);

impl CompactIndexedLeaf {
    /// Number of `(key, doc)` entries — read from the header.
    pub(super) fn count(&self) -> usize {
        read_u16(&self.0, ENTRY_COUNT_OFFSET) as usize
    }

    /// Byte offset of the per-entry distinct-value index (right after the `distinct` value column).
    fn index_offset(&self) -> usize {
        let distinct = read_u16(&self.0, DISTINCT_COUNT_OFFSET) as usize;
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        BODY_OFFSET + distinct * vw
    }

    /// Key of entry `i` — `min + distinct[index[i]]`.
    pub(super) fn key(
        &self,
        i: usize,
    ) -> u64 {
        let vw = self.0[VALUE_WIDTH_OFFSET] as usize;
        let slot = self.0[self.index_offset() + i] as usize;
        read_u64(&self.0, MIN_VALUE_OFFSET) + read_width(&self.0, BODY_OFFSET + slot * vw, vw)
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
    /// after the `count`-byte index. Docs are stored contiguously, so the stride equals the width.
    pub(super) fn doc_layout(&self) -> (usize, usize, usize) {
        let dw = self.0[DOC_WIDTH_OFFSET] as usize;
        let docs_offset = self.index_offset() + self.count();
        (docs_offset, dw, dw)
    }

    /// Encode the de-duplicated compact layout `[header][distinct values][index][docs]` (the caller has
    /// already chosen it and computed the widths). The distinct table and per-entry index are (re)built
    /// here in one in-cache pass over the ≤ `LEAF_MAX` entries — deliberately *not* carried over from
    /// [`Leaf::from_pairs`], since materialising them there measurably slowed the far more common AoS path
    /// while saving nothing here (the byte writes dominate).
    pub(super) fn build(
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

impl CompactIndexedLeaf {
    pub(super) fn distinct_count(&self) -> usize {
        read_u16(&self.0, DISTINCT_COUNT_OFFSET) as usize
    }

    /// The slot of `key` in the (sorted) distinct value column: `Ok(slot)` if it is already a distinct
    /// value, `Err(insertion_slot)` if it is new.
    pub(super) fn distinct_slot(
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
    pub(super) fn splice_insert(
        &self,
        key: u64,
        doc: u64,
        pos: usize,
    ) -> Self {
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
                Self(Arc::from(buf.as_slice()))
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
                Self(Arc::from(buf.as_slice()))
            }
        }
    }

    /// Cut entry `pos` out of the packed page — no decode. The caller guarantees the tuple is present at
    /// `pos` and that the leaf stays validly indexed afterward (`distinct_count < count - 1`). The distinct
    /// table is left untouched (a value whose last entry is removed becomes an unreferenced slot, reclaimed
    /// at the next rebuild, and reused if that value is re-inserted before then).
    pub(super) fn splice_remove(
        &self,
        pos: usize,
    ) -> Self {
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
        Self(Arc::from(buf.as_slice()))
    }

    /// Merge a sorted `batch` into this indexed page — no full decode. The caller guarantees every batch
    /// entry [`packing_fits`] and `count + batch.len() <= LEAF_MAX`. Existing doc cells are copied
    /// verbatim; the distinct table is rebuilt as the sorted union of old + batch values and index bytes are
    /// re-pointed; exact `(key, doc)` duplicates are dropped. Stays indexed (old distinct < old count
    /// guarantees merged distinct < merged count).
    pub(super) fn merge(
        &self,
        batch: &[(u64, u64)],
    ) -> Self {
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
        Self(Arc::from(buf.as_slice()))
    }

    /// Merge a sorted `batch` whose keys are all *existing* distinct values, by galloping to each entry's
    /// position and block-copying the leaf's index + doc cells between positions (verbatim — no per-entry key
    /// decode for the copied spans, and no distinct-table change so no index remap). The caller guarantees
    /// every entry [`packing_fits`], every key is already a distinct value, and
    /// `count + batch.len() <= LEAF_MAX`. Exact `(key, doc)` duplicates (within the batch or against the leaf)
    /// are dropped. Galloping (see [`gallop_lower_bound`]) makes this never worse than the full
    /// [`Self::merge`] walk at any batch size — `O(B·log(N/B))` — so the caller routes *all* all-existing-
    /// distinct batches here and only new-distinct ones (which need an index remap) to `merge`.
    pub(super) fn block_copy_merge(
        &self,
        batch: &[(u64, u64)],
    ) -> Self {
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
        Self(Arc::from(buf.as_slice()))
    }
}
