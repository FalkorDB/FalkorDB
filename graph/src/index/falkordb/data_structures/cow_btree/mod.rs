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

pub use cursor::{DocExtract, Extract, RangeIter, TupleExtract};
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

/// Byte width of the `u64` key field.
const FIELD: usize = std::mem::size_of::<u64>();

/// Little-endian `DOC_BYTES` bytes of `doc` for the [`AosLeaf`] layout. The AoS
/// entry is `key:8 + doc:DOC_BYTES`, so a tree built with `DOC_BYTES = 4` stores
/// 12 B/entry (docs — node/edge ids — are u32-ranged) while `DOC_BYTES = 8`
/// keeps the full-width 16 B/entry. Panics if `doc` does not fit the configured
/// width: ids are width-bounded by construction, so a loud failure beats
/// silently truncating an id.
fn doc_le_bytes<const DOC_BYTES: usize>(doc: u64) -> [u8; DOC_BYTES] {
    let le = doc.to_le_bytes();
    assert!(
        le[DOC_BYTES..].iter().all(|&b| b == 0),
        "cow_btree doc exceeds the configured doc width (DOC_BYTES)"
    );
    let mut out = [0u8; DOC_BYTES];
    out.copy_from_slice(&le[..DOC_BYTES]);
    out
}

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
pub struct CowBTree<
    const LEAF_MAX: usize = 256,
    const BRANCH_MAX: usize = 256,
    const DOC_BYTES: usize = 8,
> {
    root: Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>,
}

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize> Default
    for CowBTree<LEAF_MAX, BRANCH_MAX, DOC_BYTES>
{
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

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>
    CowBTree<LEAF_MAX, BRANCH_MAX, DOC_BYTES>
{
    /// An empty tree.
    #[must_use]
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
    #[must_use]
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
        let leaves: Vec<Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>> = pairs
            .chunks(LEAF_MAX)
            .map(|chunk| Node::Leaf(Leaf::from_pairs(chunk)))
            .collect();
        Self {
            root: build_root(leaves),
        }
    }

    /// Insert a single `(key, doc)`. Idempotent: inserting an existing tuple is a no-op. Returns
    /// `true` iff the tuple was newly inserted (so callers can maintain an exact live count).
    #[must_use]
    pub fn insert(
        &mut self,
        key: u64,
        doc: u64,
    ) -> bool {
        let (inserted, split) = self.root.insert_one(key, doc);
        if let Some(Split { sep, right }) = split {
            // The root split — grow a fresh level above the two halves.
            let left = std::mem::replace(&mut self.root, Node::Leaf(Leaf::from_pairs(&[])));
            self.root = Node::Branch(Arc::new(Branch {
                seps: vec![sep],
                children: vec![left, right],
            }));
        }
        inserted
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
    /// tuple is a no-op. Returns `true` iff a tuple was actually removed (for exact-count callers).
    #[must_use]
    pub fn remove(
        &mut self,
        key: u64,
        doc: u64,
    ) -> bool {
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
            true
        } else {
            false
        }
    }

    /// Lazily iterate the doc ids whose key lies in `[lo, hi]`, in `(key, doc)` order. The returned
    /// iterator owns a snapshot of the tree (an `O(1)` root clone) and can be dropped mid-scan.
    #[must_use]
    pub fn range(
        &self,
        lo: u64,
        hi: u64,
    ) -> RangeIter<LEAF_MAX, BRANCH_MAX, DOC_BYTES> {
        RangeIter::new(&self.root, (lo, 0), hi)
    }

    /// Lazily iterate the doc ids whose key equals `key` (a degenerate range).
    #[must_use]
    pub fn point(
        &self,
        key: u64,
    ) -> RangeIter<LEAF_MAX, BRANCH_MAX, DOC_BYTES> {
        self.range(key, key)
    }

    /// Whether any doc is stored under `key`. Descends to the first matching entry
    /// and stops — no full point scan.
    #[must_use]
    pub fn contains_key(
        &self,
        key: u64,
    ) -> bool {
        self.first_doc(key).is_some()
    }

    /// The smallest doc stored under `key`, or `None`. A direct reference-descent
    /// to the leaf — unlike [`point`](Self::point) it clones no page `Arc`s and
    /// allocates no cursor stack, so it is the cheap way to ask "is there one, and
    /// which" without building a cursor.
    ///
    /// Reached in production only through [`contains_key`](Self::contains_key).
    #[must_use]
    pub fn first_doc(
        &self,
        key: u64,
    ) -> Option<u64> {
        let mut node = &self.root;
        // The subtree immediately to the right of the descent path. When a key's
        // entries begin exactly at a child boundary (the key is the min of
        // `children[ci + 1]`), `child_index(key, 0)` steers into the *previous*
        // child, whose leaf then has no matching entry — the answer is that
        // subtree's minimum. (The cursor instead advances leaves via its stack.)
        //
        // It has to be the subtree, not the separator standing in for it. A
        // separator is only guaranteed to be a valid *routing* boundary, not a
        // live `min(children[ci + 1])`: removing a child's minimum without
        // triggering a rebalance leaves the separator naming a tuple that is no
        // longer in the tree, and reading a doc straight out of it hands back a
        // deleted entry. Re-deriving the minimum is one extra walk down a left
        // spine, and only on the miss path.
        let mut right_subtree: Option<&Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>> = None;
        loop {
            match node {
                Node::Leaf(leaf) => {
                    let pos = leaf.lower_bound(key);
                    if pos < leaf.count() && leaf.key(pos) == key {
                        return Some(leaf.doc(pos));
                    }
                    // Overshot this leaf: the next entry in key order is that minimum.
                    return right_subtree
                        .map(Node::min)
                        .filter(|&(k, _)| k == key)
                        .map(|(_, doc)| doc);
                }
                Node::Branch(branch) => {
                    let ci = branch.child_index(key, 0);
                    if ci + 1 < branch.children.len() {
                        // Deeper levels overwrite shallower ones, which is what we want:
                        // the nearest right neighbour is the deepest one on the path.
                        right_subtree = Some(&branch.children[ci + 1]);
                    }
                    node = &branch.children[ci];
                }
            }
        }
    }

    /// Lazily iterate the full `(key, doc)` tuples whose key lies in `[lo, hi]`, in `(key, doc)` order.
    /// Like [`range`](Self::range) but yields the key too, for a consumer that must recover the
    /// indexed value and not just the entity. Owns a snapshot.
    ///
    /// No caller today — the numeric index reads docs only.
    #[must_use]
    pub fn range_tuples(
        &self,
        lo: u64,
        hi: u64,
    ) -> RangeIter<LEAF_MAX, BRANCH_MAX, DOC_BYTES, TupleExtract> {
        RangeIter::new(&self.root, (lo, 0), hi)
    }

    /// Call `f(key, doc)` for every tuple in the tree, in `(key, doc)` order. A bulk full-scan that
    /// matches each leaf's format once and runs a tight inner loop — faster than collecting the lazy
    /// [`range_tuples`](Self::range_tuples) cursor when the whole tree is consumed (`iter_edges`, the MSF
    /// rebuild). No allocation, no per-entry `Iterator::next` dispatch.
    pub fn for_each_tuple<F: FnMut(u64, u64)>(
        &self,
        mut f: F,
    ) {
        fn walk<
            F: FnMut(u64, u64),
            const LEAF_MAX: usize,
            const BRANCH_MAX: usize,
            const DOC_BYTES: usize,
        >(
            node: &Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>,
            f: &mut F,
        ) {
            match node {
                Node::Leaf(leaf) => leaf.for_each_tuple(&mut *f),
                Node::Branch(branch) => branch.children.iter().for_each(|c| walk(c, f)),
            }
        }
        walk(&self.root, &mut f);
    }

    /// Approximate resident heap bytes: the sum of every leaf's byte blob plus branch child/separator
    /// vectors. Walks all pages (`O(pages)`), so call it off hot paths (memory reporting).
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        fn walk<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>(
            node: &Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>,
            acc: &mut usize,
        ) {
            match node {
                Node::Leaf(leaf) => *acc += leaf.raw().len(),
                Node::Branch(branch) => {
                    // `seps` is `Vec<(u64, u64)>` — each separator is a full
                    // `(key, doc)` pair, not a bare key.
                    *acc += branch.seps.len() * std::mem::size_of::<(u64, u64)>()
                        + branch.children.len()
                            * std::mem::size_of::<Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>>();
                    branch.children.iter().for_each(|c| walk(c, acc));
                }
            }
        }
        let mut acc = 0;
        walk(&self.root, &mut acc);
        acc
    }

    /// Total number of live tuples. Test-only (`O(n)` page walk) — prefer [`is_empty`] in non-test code.
    #[cfg(test)]
    #[must_use]
    pub fn len(&self) -> usize {
        fn count<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>(
            node: &Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>
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
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(&self.root, Node::Leaf(leaf) if leaf.count() == 0)
    }

    /// All leaf pages in key order, each paired with its [`LeafFormat`]. The bytes are the (tag-free)
    /// serialized form and the format is carried alongside, so writing the tree out to a byte store needs
    /// no further encoding — re-adopt a page with [`Leaf::from_parts`]. Test-only for now (the byte-store
    /// round-trip + format assertions); the durable-write path will re-expose it when disk lands.
    #[cfg(test)]
    #[must_use]
    pub fn leaves(&self) -> Vec<(LeafFormat, Arc<[u8]>)> {
        fn walk<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>(
            node: &Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>,
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
