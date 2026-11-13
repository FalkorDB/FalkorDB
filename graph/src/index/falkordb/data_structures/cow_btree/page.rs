//! What the tree asks of its leaf pages: [`Page`].
//!
//! The tree above the leaves — branches, separators, copy-on-write path copying, splits, merges,
//! batched writes — needs only an order on keys and a handful of page operations. Each key type
//! brings its own page format: `u64` keys use [`Leaf`](super::leaf::Leaf) (array-of-structs or
//! compact), string keys use [`StrLeaf`](super::str_leaf::StrLeaf) (each string once, prefix
//! compressed). A page is always a self-contained byte blob, its in-memory form also its
//! serialized form.

/// One `(key, doc)` tuple of a page whose keys are `K`.
pub type Tuple<K> = (K, u64);

/// A leaf page of sorted, distinct `(key, doc)` tuples.
pub trait Page: Clone + Sized {
    /// The key: `Ord` gives the tree's order, `(key, doc)` the order of its tuples.
    type Key: Ord + Clone + 'static;

    /// A page holding nothing: the root of an empty tree.
    fn empty() -> Self;

    /// Pages holding `tuples` (sorted, distinct) in order, each within the format's limits: a
    /// tuple count for a fixed-width format, and also a byte budget for one whose keys vary in width.
    fn build(tuples: &[Tuple<Self::Key>]) -> Vec<Self>;

    /// How many tuples the page holds.
    fn count(&self) -> usize;

    /// The page's first tuple. Panics on an empty page: only an empty tree's root is one.
    fn min(&self) -> Tuple<Self::Key>;

    /// Whether the page fell below its minimum fill, so a parent must merge it.
    fn is_underfull(&self) -> bool;

    /// The page with `(key, doc)` added, or `None` if it is already there.
    fn insert(
        &self,
        key: &Self::Key,
        doc: u64,
    ) -> Option<PageInsert<Self>>;

    /// The page without `(key, doc)`, and whether it fell below its minimum fill; `None` if the
    /// tuple is not there.
    fn remove(
        &self,
        key: &Self::Key,
        doc: u64,
    ) -> Option<(Self, bool)>;

    /// The page(s) holding this page's tuples and `batch` (sorted, all routing here).
    fn merge_batch(
        &self,
        batch: &[Tuple<Self::Key>],
    ) -> Vec<Self>;

    /// The page without `batch` (sorted, all routing here). Removal only shrinks a page, so the
    /// result is one page, possibly empty.
    fn remove_batch(
        &self,
        batch: &[Tuple<Self::Key>],
    ) -> Self;

    /// This page and its right sibling `right` as one page, or as two balanced pages and the
    /// separator between them (the second page's first tuple).
    fn combine(
        &self,
        right: &Self,
    ) -> Combined<Self, Self::Key>;

    /// Heap bytes the page holds.
    fn heap_bytes(&self) -> usize;
}

/// A page after one insert: it still fits, or it split into itself and pages to its right, each
/// with the separator that precedes it (its first tuple).
pub enum PageInsert<P: Page> {
    Fit(P),
    Split {
        left: P,
        rights: Vec<(Tuple<P::Key>, P)>,
    },
}

/// Two siblings, pages or nodes, after they are combined: one, or two rebalanced ones and the
/// separator between them (the second's first tuple).
pub enum Combined<T, K> {
    One(T),
    Two(T, Tuple<K>, T),
}

impl<T, K> Combined<T, K> {
    /// The same outcome, each sibling wrapped by `f`.
    pub fn map<U>(
        self,
        f: impl Fn(T) -> U,
    ) -> Combined<U, K> {
        match self {
            Self::One(one) => Combined::One(f(one)),
            Self::Two(left, sep, right) => Combined::Two(f(left), sep, f(right)),
        }
    }
}
