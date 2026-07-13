//! The lazy range cursor: a droppable iterator over the doc ids in a key range, owning a snapshot.

use std::sync::Arc;

use super::leaf::Leaf;
use super::node::{Branch, Node};
use super::read_width;

/// A lazy, droppable cursor over the doc ids in a key range. Owns a snapshot of the tree.
pub struct RangeIter<const LEAF_MAX: usize, const BRANCH_MAX: usize> {
    _root: Node<LEAF_MAX, BRANCH_MAX>, // keeps the snapshot (and all its pages) alive for the cursor's lifetime
    stack: Vec<(Arc<Branch<LEAF_MAX, BRANCH_MAX>>, usize)>, // (branch, next child index to descend)
    leaf: Option<Leaf<LEAF_MAX>>,
    leaf_count: usize, // entry count of `leaf`, read once per leaf rather than per entry
    whole: bool,       // every entry of `leaf` is `<= hi_key` ⇒ skip the per-entry bound check
    // Cached doc decode layout `(base, stride, width)` for the current leaf — read once in `set_leaf`
    // (from [`Leaf::doc_layout`]) so `next` decodes docs without recomputing offsets per entry (matters
    // for the compact formats). Keys, by contrast, are read straight through [`Leaf::key`]: that happens
    // only at the range start and on a boundary leaf, never per yielded entry, so it needs no cache.
    doc_base: usize,
    doc_stride: usize,
    doc_width: usize,
    pos: usize,
    hi_key: u64, // inclusive upper key bound (the doc half of the bound is always `u64::MAX`)
}

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize> RangeIter<LEAF_MAX, BRANCH_MAX> {
    pub(super) fn new(
        root: &Node<LEAF_MAX, BRANCH_MAX>,
        lo: (u64, u64),
        hi_key: u64,
    ) -> Self {
        let mut it = Self {
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

    /// Position the cursor on `leaf` at entry `pos`, caching its count, doc layout, and whether the leaf's
    /// last key is within the upper bound. When it is, every entry from `pos` on qualifies, so
    /// [`Iterator::next`] can emit docs without reading the key or comparing — only a *boundary* leaf needs
    /// per-entry checks.
    fn set_leaf(
        &mut self,
        leaf: Leaf<LEAF_MAX>,
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
        mut node: Node<LEAF_MAX, BRANCH_MAX>,
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

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize> Iterator for RangeIter<LEAF_MAX, BRANCH_MAX> {
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
