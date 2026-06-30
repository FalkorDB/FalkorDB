//! The array-of-structs (AoS) leaf encoding: contiguous `(key, doc)` tuples, 16 B/entry.

use std::sync::Arc;

use super::super::{FIELD, STRIDE, read_u64};
use super::merge_walk;

/// An array-of-structs leaf page: a tag-free `Arc<[u8]>` of `[(key:8, doc:8) × n]`, data at byte 0.
#[derive(Clone)]
pub(crate) struct AosLeaf(pub(crate) Arc<[u8]>);

impl AosLeaf {
    /// Number of `(key, doc)` entries — the buffer is tag-free `[(key, doc) × n]`, so `len / STRIDE`.
    pub(super) fn count(&self) -> usize {
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
    pub(super) fn key(
        &self,
        i: usize,
    ) -> u64 {
        self.read(i, 0)
    }

    /// Doc of entry `i`.
    pub(super) fn doc(
        &self,
        i: usize,
    ) -> u64 {
        self.read(i, FIELD)
    }

    /// Encode sorted `pairs` into the tag-free `[(key, doc) × n]` layout.
    pub(super) fn build(pairs: &[(u64, u64)]) -> Self {
        let mut buf = Vec::with_capacity(pairs.len() * STRIDE);
        for &(key, doc) in pairs {
            buf.extend_from_slice(&key.to_le_bytes());
            buf.extend_from_slice(&doc.to_le_bytes());
        }
        Self(Arc::from(buf.as_slice()))
    }

    /// Two-pointer merge of `self`'s entries with a sorted `batch` into a single new tag-free AoS page — no
    /// `Vec<(u64, u64)>`, no sort, one allocation. Exact `(key, doc)` duplicates are dropped (re-adding an
    /// existing tuple is a no-op). The caller guarantees the result fits (`count + batch.len() <= LEAF_MAX`).
    /// (A galloping block-copy was tried here but only helped tiny batches and regressed large ones — AoS has
    /// no per-entry decode for it to amortize, unlike the compact merges — so the simple walk stays.)
    pub(super) fn merge_batch(
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
