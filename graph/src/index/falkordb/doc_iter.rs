//! The substrate every index kind shares: the tuple tree, the doc iterators over it, and the
//! encoded-tuple artifact the online build passes around.
//!
//! A kind (numeric, tag, geo) differs only in how it turns a [`Value`](crate::runtime::value::Value)
//! into the `u64` key half of a `(key, doc)` tuple. Everything downstream of that — the tree, the
//! cursors, the union machinery, the BASE/DELTA/TOMB artifacts — is identical, so it lives here
//! rather than being reinvented per kind.

use super::data_structures::cow_btree::{CowBTree, RangeIter};

/// Tree fan-out. Fixed at the tuned default for now; a future step can make the
/// index generic over these if a workload wants a different page size.
const LEAF_MAX: usize = 256;
const BRANCH_MAX: usize = 256;
/// Full-width doc ids. The tree can store them narrower (fewer bytes per entry, so more entries
/// per page), but that caps the representable id and the index must hold any node or edge id the
/// graph can mint. Narrowing is a memory optimization to make deliberately, once there is a bound
/// to justify it — not a default to inherit.
const DOC_BYTES: usize = 8;

pub type Tree = CowBTree<LEAF_MAX, BRANCH_MAX, DOC_BYTES>;

/// Lazy iterator of matching entity ids, yielded in `(value, id)` order.
pub type TreeIter = RangeIter<LEAF_MAX, BRANCH_MAX, DOC_BYTES>;

/// Lazy iterator of matching entity ids.
///
/// `One` is a single cursor over a contiguous key range. `Many` chains several — the shape an
/// `IN [...]` union, a mixed-type union, or a geo cell cover takes. An enum rather than a boxed
/// trait object because the scan op already boxes the result once; a second layer of dynamic
/// dispatch per id would be pure overhead.
///
/// The chain is *not* merged into id order. Order is deliberately unspecified here: Cypher
/// guarantees none without `ORDER BY`, and a merge would cost a comparison per id and force every
/// branch to stay live. Nothing downstream may assume sortedness.
pub enum DocIter {
    One(TreeIter),
    Many(UnionIter),
    /// An already-materialised doc set — an intersection, which cannot be produced lazily from
    /// value-ordered streams. Distinct by construction, and a `Vec` rather than a set iterator so
    /// producers can use whichever hasher they like.
    Set(std::vec::IntoIter<u64>),
}

impl Iterator for DocIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match self {
            Self::One(it) => it.next(),
            Self::Many(it) => it.next(),
            Self::Set(it) => it.next(),
        }
    }
}

/// The cursor chain behind a union: the key windows still to visit, and a cursor over the current
/// one. A window names its tree by index, so one chain can span the kinds a mixed union touches
/// (`n.v IN [1, 'a']` reads the numeric tree and the tag tree) without a tree clone per window.
///
/// **One cursor exists at a time, and none until the first `next()`.** `RangeIter::new` performs a
/// full root-to-leaf descent in its constructor, so building a cursor per member up front made
/// `IN $list` with 100k members do 100k descents — and hold 100k live snapshots — before yielding
/// a single row. Under a `LIMIT`, almost all of that work is thrown away.
///
/// The trees are **owned**, not borrowed. That is load-bearing rather than a lifetime convenience:
/// the eager version happened to share one root because every `point()` call sat inside a single
/// `&self` borrow. Deferring those calls past the borrow means the snapshot has to be pinned here,
/// or a member visited after a concurrent write would descend into a newer root and the union
/// would be a torn read. Each clone is an `O(1)` root-`Arc` bump, and there is one per *tree*, not
/// one per window.
///
/// Windows must be **disjoint**, or a doc lying in two of them is yielded twice. Every producer
/// upholds that: union members are deduplicated by key, and a geo cover is a set of disjoint
/// quadtree cells.
pub struct UnionIter {
    trees: Vec<Tree>,
    /// `(tree index, lo, hi)` — the inclusive key window to scan in that tree.
    windows: std::vec::IntoIter<(u8, u64, u64)>,
    current: Option<TreeIter>,
}

impl UnionIter {
    /// A chain over `windows`, each naming its tree by position in `trees`.
    #[must_use]
    pub fn new(
        trees: Vec<Tree>,
        windows: Vec<(u8, u64, u64)>,
    ) -> Self {
        Self {
            trees,
            windows: windows.into_iter(),
            current: None,
        }
    }

    /// Windows not yet visited — the laziness assertion in the tests.
    #[cfg(test)]
    #[must_use]
    pub fn pending(&self) -> usize {
        self.windows.len()
    }

    /// Whether a cursor has been built yet.
    #[cfg(test)]
    #[must_use]
    pub fn has_cursor(&self) -> bool {
        self.current.is_some()
    }
}

impl Iterator for UnionIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        loop {
            if let Some(doc) = self.current.as_mut().and_then(Iterator::next) {
                return Some(doc);
            }
            // Current window exhausted (or not started): descend for the next one. The window
            // list is finite and each step consumes one, so this cannot spin.
            let (tree, lo, hi) = self.windows.next()?;
            self.current = Some(self.trees[tree as usize].range(lo, hi));
        }
    }
}

/// An iterator over no entries (`lo > hi` yields nothing).
#[must_use]
pub fn empty_docs(tree: &Tree) -> DocIter {
    DocIter::One(tree.range(1, 0))
}

/// Encoded `(key, doc)` tuples for one kind within a column, split by key space.
///
/// The online build moves BASE, DELTA and TOMB around as raw tuples; each artifact has to keep the
/// scalar and array key spaces apart for the same reason the trees do — a list element must never
/// be reachable through a scalar predicate.
#[derive(Default, Debug, Clone)]
pub struct KeyTuples {
    pub scalar: Vec<(u64, u64)>,
    pub array: Vec<(u64, u64)>,
}

impl KeyTuples {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scalar.is_empty() && self.array.is_empty()
    }

    /// Total tuples across both key spaces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scalar.len() + self.array.len()
    }

    /// Scalar-only tuples — the shape a kind with no list values produces.
    #[must_use]
    pub fn scalars(scalar: Vec<(u64, u64)>) -> Self {
        Self {
            scalar,
            array: Vec::new(),
        }
    }

    /// Drop every tuple whose doc fails `keep` — the install's deleted-entity backstop, which
    /// must sweep both key spaces or a deleted node survives in its array half.
    pub fn retain_docs(
        &mut self,
        keep: &mut impl FnMut(u64) -> bool,
    ) {
        self.scalar.retain(|&(_, doc)| keep(doc));
        self.array.retain(|&(_, doc)| keep(doc));
    }

    /// Sort both halves, as `insert_batch` / `remove_batch` require.
    pub fn sort(&mut self) {
        self.scalar.sort_unstable();
        self.array.sort_unstable();
    }
}
