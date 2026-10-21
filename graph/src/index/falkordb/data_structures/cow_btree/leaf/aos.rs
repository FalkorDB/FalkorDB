//! The array-of-structs (AoS) leaf encoding: contiguous `(key, doc)` tuples,
//! `FIELD + DOC_BYTES` B/entry (12 B for a 4-byte doc, 16 B for the full u64).

use std::sync::Arc;

use super::super::{FIELD, doc_le_bytes, read_u64, read_width};
use super::merge_walk;

/// An array-of-structs leaf page: a tag-free `Arc<[u8]>` of `[(key:8, doc:DOC_BYTES) × n]`, data at byte 0.
#[derive(Clone)]
pub struct AosLeaf<const DOC_BYTES: usize>(pub(crate) Arc<[u8]>);

impl<const DOC_BYTES: usize> AosLeaf<DOC_BYTES> {
    /// Byte stride of one `(key, doc)` entry: an 8-byte key beside a `DOC_BYTES`-byte doc.
    const STRIDE: usize = FIELD + DOC_BYTES;

    /// Number of `(key, doc)` entries — the buffer is tag-free `[(key, doc) × n]`, so `len / STRIDE`.
    pub(super) fn count(&self) -> usize {
        self.0.len() / Self::STRIDE
    }

    /// Key of entry `i` (a full little-endian `u64` at the entry's start).
    pub(super) fn key(
        &self,
        i: usize,
    ) -> u64 {
        read_u64(&self.0, Self::STRIDE * i)
    }

    /// Doc of entry `i` (a `DOC_BYTES`-wide little-endian int at offset `FIELD`, zero-extended to `u64`).
    pub(super) fn doc(
        &self,
        i: usize,
    ) -> u64 {
        read_width(&self.0, Self::STRIDE * i + FIELD, DOC_BYTES)
    }

    /// Encode sorted `pairs` into the tag-free `[(key, doc) × n]` layout.
    pub(super) fn build(pairs: &[(u64, u64)]) -> Self {
        let mut buf = Vec::with_capacity(pairs.len() * Self::STRIDE);
        for &(key, doc) in pairs {
            buf.extend_from_slice(&key.to_le_bytes());
            buf.extend_from_slice(&doc_le_bytes::<DOC_BYTES>(doc));
        }
        Self(Arc::from(buf.as_slice()))
    }

    /// Two-pointer merge of `self`'s entries with a sorted `batch` into a single new tag-free AoS page — no
    /// `Vec<(u64, u64)>`, no sort, one allocation. Exact `(key, doc)` duplicates are dropped (re-adding an
    /// existing tuple is a no-op). The caller guarantees the result fits (`count + batch.len() <= LEAF_MAX`).
    pub(super) fn merge_batch(
        &self,
        batch: &[(u64, u64)],
    ) -> Self {
        let count = self.count();
        let mut buf = Vec::with_capacity(self.0.len() + batch.len() * Self::STRIDE);
        merge_walk(
            count,
            |i| self.key(i),
            |i| self.doc(i),
            batch,
            |key, doc, _| {
                buf.extend_from_slice(&key.to_le_bytes());
                buf.extend_from_slice(&doc_le_bytes::<DOC_BYTES>(doc));
            },
        );
        Self(Arc::from(buf.as_slice()))
    }
}
