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

mod cursor;
mod leaf;
mod node;
#[cfg(test)]
mod tests;

pub use cursor::RangeIter;
use leaf::Leaf;
pub use leaf::LeafFormat;
use node::{Branch, Node, Split, build_root};

// Max **entries** (`(key, doc)` tuples) per leaf page is the `LEAF_MAX` const generic on [`CowBTree`]
// (default 256); split on overflow, merge below `LEAF_MAX / 2`. Leaf balance is by entry *count*, not byte
// size — the same bound for both leaf encodings (see [`Leaf`]), so a page holds ≤ `LEAF_MAX` entries whether
// they encode to ≈ 4 KiB ([`AosLeaf`], 16 B/tuple) or less ([`CompactLeaf`]).
//
// The 256 ceiling is load-bearing for the compact format: its dedup index stores one slot per entry as a
// `u8`, which only fits because `count ≤ LEAF_MAX ≤ 256` (see [`CompactLeaf::build`]). **Raising `LEAF_MAX`
// past 256 would overflow that index** — the compact `as u8` cast would have to widen first — which is why
// [`CowBTree::new`] / [`CowBTree::from_sorted`] / `Default` assert `LEAF_MAX <= 256`.
// Max children per branch page (fan-out) is the `BRANCH_MAX` const generic on [`CowBTree`] (default 256):
// a branch splits on overflow and merges below `BRANCH_MAX / 2`.

/// Byte width of one `u64` field (a key or a doc).
const FIELD: usize = std::mem::size_of::<u64>();
/// Byte stride of one `(key, doc)` entry in the [`AosLeaf`] layout.
const STRIDE: usize = 2 * FIELD;

/// Read the little-endian `u64` at byte offset `off`. The `unwrap` is infallible: `b[off..off + FIELD]`
/// is always exactly `FIELD` bytes; an out-of-bounds `off` means a malformed page — a build bug, since the
/// build path always produces well-formed pages, so a panic is correct.
fn read_u64(
    b: &[u8],
    off: usize,
) -> u64 {
    u64::from_le_bytes(b[off..off + FIELD].try_into().unwrap())
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

// ---- the tree --------------------------------------------------------------------------------

/// A copy-on-write B⁺-tree mapping sorted `(key, doc)` tuples; see the module docs.
///
/// The `LEAF_MAX` const generic is the maximum number of `(key, doc)` entries per leaf page (a leaf splits
/// on overflow and merges below `LEAF_MAX / 2`). It must be in `2..=256`: at least 2 so a split yields two
/// non-empty halves, and at most 256 because the compact leaf's dedup index stores one slot per entry as a
/// `u8` (see [`CompactLeaf::build`]).
///
/// The `BRANCH_MAX` const generic is the maximum number of children per branch page / fan-out (a branch
/// splits on overflow and merges below `BRANCH_MAX / 2`). It must be `>= 3`: the [`pack_branches`]
/// `BRANCH_MAX + 1` special case (which rewrites a trailing single-child remainder into a `BRANCH_MAX - 1`
/// + `2` split, so no branch is ever born with a single child) needs `BRANCH_MAX - 1 >= 2`.
///
/// Both default to 256, which keeps behaviour identical to the original fixed bounds; the assert in
/// [`CowBTree::new`] / [`CowBTree::from_sorted`] / `Default` trips any out-of-range monomorphization that
/// builds a tree.
#[derive(Clone)]
pub struct CowBTree<const LEAF_MAX: usize = 256, const BRANCH_MAX: usize = 256> {
    root: Node<LEAF_MAX, BRANCH_MAX>,
}

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize> Default for CowBTree<LEAF_MAX, BRANCH_MAX> {
    fn default() -> Self {
        const {
            assert!(
                LEAF_MAX >= 2 && LEAF_MAX <= 256 && BRANCH_MAX >= 3,
                "LEAF_MAX must be 2..=256 (compact index is u8); BRANCH_MAX >= 3 (the pack_branches BRANCH_MAX+1 split needs >= 3)"
            );
        }
        Self {
            root: Node::Leaf(Leaf::from_pairs(&[])),
        }
    }
}

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize> CowBTree<LEAF_MAX, BRANCH_MAX> {
    /// An empty tree.
    pub fn new() -> Self {
        const {
            assert!(
                LEAF_MAX >= 2 && LEAF_MAX <= 256 && BRANCH_MAX >= 3,
                "LEAF_MAX must be 2..=256 (compact index is u8); BRANCH_MAX >= 3 (the pack_branches BRANCH_MAX+1 split needs >= 3)"
            );
        }
        Self::default()
    }

    /// Bulk-build from `pairs` **sorted ascending and unique** by `(key, doc)`. Packs full leaf pages
    /// directly from the slice and builds the branch levels bottom-up — no per-item sort, dedup, or
    /// insert traversal, so it is far cheaper than inserting one at a time.
    pub fn from_sorted(pairs: &[(u64, u64)]) -> Self {
        const {
            assert!(
                LEAF_MAX >= 2 && LEAF_MAX <= 256 && BRANCH_MAX >= 3,
                "LEAF_MAX must be 2..=256 (compact index is u8); BRANCH_MAX >= 3 (the pack_branches BRANCH_MAX+1 split needs >= 3)"
            );
        }
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
        let leaves: Vec<Node<LEAF_MAX, BRANCH_MAX>> = pairs
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
    ) -> RangeIter<LEAF_MAX, BRANCH_MAX> {
        RangeIter::new(&self.root, (lo, 0), hi)
    }

    /// Lazily iterate the doc ids whose key equals `key` (a degenerate range).
    pub fn point(
        &self,
        key: u64,
    ) -> RangeIter<LEAF_MAX, BRANCH_MAX> {
        self.range(key, key)
    }

    /// Total number of live tuples. Test-only (`O(n)` page walk) — prefer [`is_empty`] in non-test code.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        fn count<const LEAF_MAX: usize, const BRANCH_MAX: usize>(
            node: &Node<LEAF_MAX, BRANCH_MAX>
        ) -> usize {
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
    /// no further encoding — re-adopt a page with [`Leaf::from_parts`]. Test-only for now (the byte-store
    /// round-trip + format assertions); the durable-write path will re-expose it when disk lands.
    #[cfg(test)]
    pub fn leaves(&self) -> Vec<(LeafFormat, Arc<[u8]>)> {
        fn walk<const LEAF_MAX: usize, const BRANCH_MAX: usize>(
            node: &Node<LEAF_MAX, BRANCH_MAX>,
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
