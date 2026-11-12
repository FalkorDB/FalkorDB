//! The lazy cursor over a string-keyed tree: the docs of the tuples whose string lies in a range.
//!
//! It owns a snapshot (a root clone), descends the branches by their separators as the `u64`
//! cursor does, and inside a page binary-searches the restart points before decoding forward, so a
//! seek decodes at most [`RESTART_EVERY`](super::str_leaf::RESTART_EVERY) strings of its page.

use std::ops::Bound;
use std::sync::Arc;

use super::node::{Branch, Node};
use super::str_leaf::{RESTART_EVERY, StrLeaf, get_varint, header, restart_key, restart_offset};
use super::{PageTree, StrTree};

/// A key range's ends.
pub type StrBounds = (Bound<Arc<String>>, Bound<Arc<String>>);

/// Where a cursor is inside one page.
#[derive(Default)]
struct PagePos {
    pos: usize,
    groups_left: usize,
    docs_left: usize,
    prev_doc: u64,
    /// The current group's string.
    key: Vec<u8>,
    /// The current group's first doc, not yet yielded.
    pending: Option<u64>,
}

/// The docs of the tuples whose string lies between two bounds, lazily, over the snapshot the
/// cursor was made on.
pub struct StrRange<const BRANCH_MAX: usize = 256> {
    /// The branches from the root to the current page, with the child taken in each.
    path: Vec<(Arc<Branch<StrLeaf, BRANCH_MAX>>, usize)>,
    page: Option<StrLeaf>,
    at: PagePos,
    hi: Bound<Arc<String>>,
    done: bool,
}

impl<const BRANCH_MAX: usize> StrRange<BRANCH_MAX> {
    fn new(
        root: &Node<StrLeaf, BRANCH_MAX>,
        lo: Bound<Arc<String>>,
        hi: Bound<Arc<String>>,
    ) -> Self {
        let mut it = Self {
            path: Vec::new(),
            page: None,
            at: PagePos::default(),
            hi,
            done: false,
        };
        // Descend to the page holding the first tuple at or after `lo`.
        let mut node = root.clone();
        loop {
            match node {
                Node::Leaf(page) => {
                    it.enter(page);
                    break;
                }
                Node::Branch(branch) => {
                    let c = match &lo {
                        Bound::Included(k) => branch.child_index(k, 0),
                        Bound::Excluded(k) => branch.child_index(k, u64::MAX),
                        Bound::Unbounded => 0,
                    };
                    let child = branch.children[c].clone();
                    it.path.push((branch, c));
                    node = child;
                }
            }
        }
        let before_lo = |key: &[u8]| match &lo {
            Bound::Included(k) => key < k.as_bytes(),
            Bound::Excluded(k) => key <= k.as_bytes(),
            Bound::Unbounded => false,
        };
        if !matches!(lo, Bound::Unbounded) {
            it.seek_restart(before_lo);
        }
        if !it.load_group() {
            it.done = true;
            return it;
        }
        while before_lo(&it.at.key) {
            if !it.skip_group() {
                it.done = true;
                return it;
            }
        }
        if it.past_hi() {
            it.done = true;
        }
        it
    }

    fn enter(
        &mut self,
        page: StrLeaf,
    ) {
        let h = header(&page.blob);
        self.at = PagePos {
            pos: h.data,
            groups_left: h.groups,
            ..PagePos::default()
        };
        self.page = Some(page);
    }

    /// Move the cursor, just entered into a page, to the last restart whose string comes before
    /// `lo`: the strings from there to the first one at or after `lo` are fewer than a restart
    /// interval.
    fn seek_restart(
        &mut self,
        before_lo: impl Fn(&[u8]) -> bool,
    ) {
        let page = self.page.clone().expect("a cursor in a page");
        let blob = &page.blob;
        let h = header(blob);
        // The restarts whose string comes before `lo` are a prefix of them.
        let (mut lo, mut hi) = (0, h.restarts);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if before_lo(restart_key(blob, &h, mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > 1 {
            let r = lo - 1;
            self.at.pos = h.data + restart_offset(blob, &h, r);
            self.at.groups_left = h.groups - r * RESTART_EVERY;
            self.at.key.clear();
        }
    }

    /// Move to the next page, if any.
    fn next_page(&mut self) -> bool {
        while let Some((branch, c)) = self.path.pop() {
            if c + 1 < branch.children.len() {
                let mut node = branch.children[c + 1].clone();
                self.path.push((branch, c + 1));
                loop {
                    match node {
                        Node::Leaf(page) => {
                            self.enter(page);
                            return true;
                        }
                        Node::Branch(b) => {
                            let first = b.children[0].clone();
                            self.path.push((b, 0));
                            node = first;
                        }
                    }
                }
            }
        }
        false
    }

    /// Read the next group's header and first doc, moving to the next page if this one is spent.
    fn load_group(&mut self) -> bool {
        loop {
            if self.at.groups_left > 0 {
                let page = self.page.clone().expect("a cursor in a page");
                let blob = &page.blob;
                let at = &mut self.at;
                let shared = get_varint(blob, &mut at.pos) as usize;
                let suffix = get_varint(blob, &mut at.pos) as usize;
                at.key.truncate(shared);
                at.key.extend_from_slice(&blob[at.pos..at.pos + suffix]);
                at.pos += suffix;
                let docs = get_varint(blob, &mut at.pos) as usize;
                let first = get_varint(blob, &mut at.pos);
                at.prev_doc = first;
                at.docs_left = docs - 1;
                at.pending = Some(first);
                at.groups_left -= 1;
                return true;
            }
            if !self.next_page() {
                return false;
            }
        }
    }

    /// Skip the rest of the current group and load the next one.
    fn skip_group(&mut self) -> bool {
        if self.at.docs_left > 0 {
            let page = self.page.clone().expect("a cursor in a page");
            for _ in 0..self.at.docs_left {
                get_varint(&page.blob, &mut self.at.pos);
            }
            self.at.docs_left = 0;
        }
        self.at.pending = None;
        self.load_group()
    }

    fn past_hi(&self) -> bool {
        let key = &self.at.key[..];
        match &self.hi {
            Bound::Included(k) => key > k.as_bytes(),
            Bound::Excluded(k) => key >= k.as_bytes(),
            Bound::Unbounded => false,
        }
    }
}

impl<const BRANCH_MAX: usize> Iterator for StrRange<BRANCH_MAX> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.done {
            return None;
        }
        if let Some(d) = self.at.pending.take() {
            return Some(d);
        }
        if self.at.docs_left > 0 {
            let page = self.page.as_ref().expect("a cursor in a page");
            self.at.prev_doc += get_varint(&page.blob, &mut self.at.pos);
            self.at.docs_left -= 1;
            return Some(self.at.prev_doc);
        }
        if !self.load_group() || self.past_hi() {
            self.done = true;
            return None;
        }
        self.at.pending.take()
    }
}

/// The docs of several disjoint key ranges, one cursor after another, each opened only once the
/// one before it is spent, all over one snapshot taken when this was made.
pub struct StrRanges<const BRANCH_MAX: usize = 256> {
    tree: StrTree<BRANCH_MAX>,
    ranges: std::vec::IntoIter<StrBounds>,
    current: Option<StrRange<BRANCH_MAX>>,
}

impl<const BRANCH_MAX: usize> StrRanges<BRANCH_MAX> {
    /// The docs of `ranges`, which must not overlap, so no doc is yielded twice.
    #[must_use]
    pub fn new(
        tree: &StrTree<BRANCH_MAX>,
        ranges: Vec<StrBounds>,
    ) -> Self {
        Self {
            tree: tree.clone(),
            ranges: ranges.into_iter(),
            current: None,
        }
    }
}

impl<const BRANCH_MAX: usize> Iterator for StrRanges<BRANCH_MAX> {
    type Item = u64;

    // Kept out of line: inlined, the string cursor bloats `DocIter::next` enough that the numeric
    // cursor's `next` stops being inlined into it, and every numeric row pays for it (measured:
    // +3.8% instructions on a 10k-row numeric range scan).
    #[inline(never)]
    fn next(&mut self) -> Option<u64> {
        loop {
            if let Some(d) = self.current.as_mut().and_then(Iterator::next) {
                return Some(d);
            }
            let (lo, hi) = self.ranges.next()?;
            self.current = Some(self.tree.range(lo, hi));
        }
    }
}

/// The string tree's reads: a cursor over a range of strings, and the tuples for tests.
impl<const BRANCH_MAX: usize> PageTree<StrLeaf, BRANCH_MAX> {
    /// The docs under every string from `lo` to `hi`, in string then doc order, over a snapshot.
    #[must_use]
    pub fn range(
        &self,
        lo: Bound<Arc<String>>,
        hi: Bound<Arc<String>>,
    ) -> StrRange<BRANCH_MAX> {
        StrRange::new(&self.root, lo, hi)
    }

    /// The docs under `key`, in doc order.
    #[must_use]
    pub fn point(
        &self,
        key: &Arc<String>,
    ) -> StrRange<BRANCH_MAX> {
        self.range(Bound::Included(key.clone()), Bound::Included(key.clone()))
    }

    /// Every tuple, in order. For tests.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(Vec<u8>, u64)> {
        fn walk<const B: usize>(
            n: &Node<StrLeaf, B>,
            out: &mut Vec<(Vec<u8>, u64)>,
        ) {
            match n {
                Node::Leaf(page) => out.extend(page.tuples()),
                Node::Branch(b) => b.children.iter().for_each(|c| walk(c, out)),
            }
        }
        let mut out = Vec::new();
        walk(&self.root, &mut out);
        out
    }

    /// Every page's size in tuples and bytes, and the tree's height. Asserts every page sits at
    /// one depth. For tests.
    #[cfg(test)]
    #[must_use]
    pub fn shape(&self) -> (Vec<(usize, usize)>, usize) {
        fn walk<const B: usize>(
            n: &Node<StrLeaf, B>,
            depth: usize,
            out: &mut Vec<(usize, usize)>,
            depths: &mut Vec<usize>,
        ) {
            match n {
                Node::Leaf(page) => {
                    out.push(page.shape());
                    depths.push(depth);
                }
                Node::Branch(b) => b
                    .children
                    .iter()
                    .for_each(|c| walk(c, depth + 1, out, depths)),
            }
        }
        let (mut pages, mut depths) = (Vec::new(), Vec::new());
        walk(&self.root, 0, &mut pages, &mut depths);
        depths.dedup();
        assert!(depths.len() <= 1, "every page is at one depth");
        (pages, depths.first().copied().unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::super::str_leaf::{LEAF_BYTES, LEAF_MAX};
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::BTreeSet;

    fn key(i: u32) -> String {
        // Shared prefixes, lengths 0..40, an occasional NUL: what prefix compression must survive.
        let mut k = format!("{:x}", i % 997);
        if i.is_multiple_of(7) {
            k.push('\0');
        }
        if i.is_multiple_of(11) {
            k.extend(std::iter::repeat_n('z', (i % 40) as usize));
        }
        k
    }

    fn as_tuples(v: &[(String, u64)]) -> Vec<(Arc<String>, u64)> {
        v.iter().map(|(k, d)| (Arc::new(k.clone()), *d)).collect()
    }

    fn bytes(set: &BTreeSet<(String, u64)>) -> Vec<(Vec<u8>, u64)> {
        set.iter()
            .map(|(k, d)| (k.as_bytes().to_vec(), *d))
            .collect()
    }

    fn model_range(
        set: &BTreeSet<(String, u64)>,
        lo: &Bound<Arc<String>>,
        hi: &Bound<Arc<String>>,
    ) -> Vec<u64> {
        set.iter()
            .filter(|(k, _)| match lo {
                Bound::Included(l) => k.as_str() >= l.as_str(),
                Bound::Excluded(l) => k.as_str() > l.as_str(),
                Bound::Unbounded => true,
            })
            .filter(|(k, _)| match hi {
                Bound::Included(h) => k.as_str() <= h.as_str(),
                Bound::Excluded(h) => k.as_str() < h.as_str(),
                Bound::Unbounded => true,
            })
            .map(|(_, d)| *d)
            .collect()
    }

    fn bound(
        rng: &mut StdRng,
        k: String,
    ) -> Bound<Arc<String>> {
        match rng.random_range(0..3) {
            0 => Bound::Included(Arc::new(k)),
            1 => Bound::Excluded(Arc::new(k)),
            _ => Bound::Unbounded,
        }
    }

    /// Random batches and single inserts and removes against a sorted set: the tuples, every kind
    /// of range bound, point lookups, the shape (one depth, no page over its limits), and that a
    /// snapshot taken before a write still reads as it did.
    #[test]
    fn matches_a_sorted_set_under_random_writes() {
        let mut rng = StdRng::seed_from_u64(0x5eed);
        let mut tree = StrTree::<8>::default();
        let mut model: BTreeSet<(String, u64)> = BTreeSet::new();
        for round in 0..300 {
            let snapshot = (tree.clone(), bytes(&model));
            let add = rng.random_range(0..3) != 0;
            let n = rng.random_range(1..400);
            let mut batch: BTreeSet<(String, u64)> = BTreeSet::new();
            for _ in 0..n {
                let t = if !add && !model.is_empty() && rng.random_range(0..4) != 0 {
                    let i = rng.random_range(0..model.len());
                    model.iter().nth(i).cloned().expect("in range")
                } else {
                    // Hot strings get many docs, so a string spans pages.
                    let k = if rng.random_range(0..5) == 0 {
                        key(3)
                    } else {
                        key(rng.random_range(0..5000))
                    };
                    (k, rng.random_range(0..50_000))
                };
                batch.insert(t);
            }
            let batch: Vec<_> = batch.into_iter().collect();
            if round % 10 == 9 {
                // Every tenth round goes one tuple at a time, through the single-tuple paths.
                for (k, d) in &batch {
                    if add {
                        assert_eq!(
                            tree.insert(Arc::new(k.clone()), *d),
                            model.insert((k.clone(), *d))
                        );
                    } else {
                        assert_eq!(
                            tree.remove(Arc::new(k.clone()), *d),
                            model.remove(&(k.clone(), *d))
                        );
                    }
                }
            } else if add {
                tree.insert_batch(&as_tuples(&batch));
                model.extend(batch.iter().cloned());
            } else {
                tree.remove_batch(&as_tuples(&batch));
                for t in &batch {
                    model.remove(t);
                }
            }
            assert_eq!(tree.tuples(), bytes(&model), "round {round}");
            assert_eq!(
                snapshot.0.tuples(),
                snapshot.1,
                "a snapshot is not disturbed"
            );
            let (pages, _) = tree.shape();
            for (t, b) in &pages {
                assert!(*t <= LEAF_MAX, "a page of {t} tuples");
                assert!(*b <= LEAF_BYTES + 64, "a page of {b} bytes");
            }
            for _ in 0..20 {
                let k = key(rng.random_range(0..5000));
                let lo = bound(&mut rng, k);
                let k = key(rng.random_range(0..5000));
                let hi = bound(&mut rng, k);
                let got: Vec<u64> = tree.range(lo.clone(), hi.clone()).collect();
                assert_eq!(got, model_range(&model, &lo, &hi), "round {round}");
            }
            let k = key(rng.random_range(0..5000));
            let want: Vec<u64> = model
                .iter()
                .filter(|(mk, _)| *mk == k)
                .map(|(_, d)| *d)
                .collect();
            assert_eq!(tree.point(&Arc::new(k)).collect::<Vec<_>>(), want);
        }
        let all = as_tuples(&model.iter().cloned().collect::<Vec<_>>());
        tree.remove_batch(&all);
        assert!(
            tree.is_empty(),
            "the tree is empty, not a tree of empty pages"
        );
    }

    /// One string held by many entities spans pages, and is still read whole by a point lookup.
    #[test]
    fn a_string_held_by_many_entities_spans_pages() {
        let k = Arc::new("status:pending".to_string());
        let tuples: Vec<(Arc<String>, u64)> = (0..10_000u64).map(|d| (k.clone(), d * 3)).collect();
        let tree = StrTree::<256>::from_sorted(&tuples);
        let (pages, height) = tree.shape();
        assert!(pages.len() > 1 && height >= 1);
        assert_eq!(
            tree.point(&k).collect::<Vec<_>>(),
            (0..10_000u64).map(|d| d * 3).collect::<Vec<_>>()
        );
        assert_eq!(
            tree.point(&Arc::new("status:pendin".to_string())).count(),
            0
        );
        assert_eq!(
            tree.point(&Arc::new("status:pending!".to_string())).count(),
            0
        );
    }

    /// A string longer than a page's byte budget gets a page of its own, through a bulk build and
    /// through a single insert, and is found.
    #[test]
    fn a_string_longer_than_a_page() {
        let long = Arc::new("q".repeat(3 * LEAF_BYTES));
        let tuples = vec![
            (Arc::new("a".to_string()), 1),
            (long.clone(), 2),
            (Arc::new("r".to_string()), 3),
        ];
        let tree = StrTree::<256>::from_sorted(&tuples);
        assert_eq!(tree.point(&long).collect::<Vec<_>>(), vec![2]);
        let mut grown = StrTree::<256>::default();
        for (k, d) in &tuples {
            assert!(grown.insert(k.clone(), *d));
        }
        assert_eq!(grown.tuples(), tree.tuples());
        assert_eq!(grown.point(&long).collect::<Vec<_>>(), vec![2]);
    }

    /// Each string is stored once per page, so a repeated string costs about a doc per entity.
    #[test]
    fn a_repeated_string_costs_about_a_doc_per_entity() {
        let k = Arc::new("a-fairly-long-repeated-tag-value".to_string());
        let tuples: Vec<(Arc<String>, u64)> = (0..100_000u64).map(|d| (k.clone(), d)).collect();
        let tree = StrTree::<256>::from_sorted(&tuples);
        let per = tree.heap_bytes() as f64 / 100_000.0;
        assert!(per < 2.0, "{per} bytes per tuple");
    }
}
