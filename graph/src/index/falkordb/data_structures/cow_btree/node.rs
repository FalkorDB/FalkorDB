//! Internal tree nodes: branches and the leaf/branch [`Node`] enum, with the single + batched
//! insert/remove/combine machinery and the bottom-up builders.

use std::sync::Arc;

use super::leaf::{AosLeaf, Leaf, LeafInsert};
use super::{FIELD, read_u64, read_width};

/// A tree node: either a leaf (a byte blob of sorted tuples) or an internal branch.
///
/// Both variants are `Arc`-wrapped, so cloning a node — and therefore a whole tree version — is an
/// `O(1)` reference-count bump that shares all underlying pages.
#[derive(Clone)]
pub(super) enum Node<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize> {
    /// A leaf page of sorted `(key, doc)` tuples — see [`Leaf`].
    Leaf(Leaf<LEAF_MAX, DOC_BYTES>),
    Branch(Arc<Branch<LEAF_MAX, BRANCH_MAX, DOC_BYTES>>),
}

/// An internal node: separator keys plus child pointers. `seps[i]` is the minimum `(key, doc)` of
/// `children[i + 1]`, so `seps.len() == children.len() - 1`.
#[derive(Clone)]
pub(super) struct Branch<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize> {
    pub(super) seps: Vec<(u64, u64)>,
    pub(super) children: Vec<Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>>,
}

// ---- node-operation results ------------------------------------------------------------------

/// A node that split under an insert: the new right sibling plus the separator promoted to the parent.
pub(super) struct Split<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize> {
    pub(super) sep: (u64, u64),
    pub(super) right: Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>,
}

/// Outcome of combining two siblings: a single merged node, or two re-balanced nodes plus their new
/// separator.
enum Combined<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize> {
    One(Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>),
    Two(
        Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>,
        (u64, u64),
        Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>,
    ),
}

// ---- bottom-up builders (used by bulk build and batched insert) ------------------------------

/// Group `children` into branch pages of at most `BRANCH_MAX`, one separator per adjacent pair
/// (the separator is the minimum `(key, doc)` of the right child).
///
/// Every emitted branch gets **at least two children**: a lone single-child branch has no sibling, so a
/// later delete that underflowed its only child could not rebalance it (see [`Branch::rebalance`]). The
/// only way `chunks(BRANCH_MAX)` would leave a singleton is a trailing remainder of exactly 1, i.e. an
/// input length of `BRANCH_MAX + 1`; that case is split as `BRANCH_MAX - 1` + `2` instead.
fn pack_branches<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>(
    children: &[Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>]
) -> Vec<Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>> {
    // One branch per chunk of up to `BRANCH_MAX` children, so the count is known up front.
    let mut packed = Vec::with_capacity(children.len().div_ceil(BRANCH_MAX));
    let mut rest = children;
    while !rest.is_empty() {
        let take = if rest.len() == BRANCH_MAX + 1 {
            BRANCH_MAX - 1
        } else {
            rest.len().min(BRANCH_MAX)
        };
        // Never emit a single-child branch: a lone child can't rebalance against a sibling and would panic on
        // a later underflow. `build_root` only calls this with >= 2 children, and the BRANCH_MAX+1 special
        // case keeps the remainder from ever being exactly 1, so `take` is always >= 2.
        debug_assert!(
            take >= 2,
            "pack_branches must not emit a single-child branch"
        );
        let (chunk, remaining) = rest.split_at(take);
        // Each separator is the right child's minimum `(key, doc)` — a single B+-tree boundary point,
        // NOT a `(min, max)` range: everything in `children[i]` is `< seps[i] == min(children[i + 1])`.
        let seps: Vec<(u64, u64)> = chunk[1..].iter().map(Node::min).collect();
        packed.push(Node::Branch(Arc::new(Branch {
            seps,
            children: chunk.to_vec(),
        })));
        rest = remaining;
    }
    packed
}

/// Stitch a flat list of node fragments into a single root, packing branch levels bottom-up until
/// one node remains. An empty list becomes an empty leaf.
pub(super) fn build_root<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>(
    mut fragments: Vec<Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>>
) -> Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES> {
    while fragments.len() > 1 {
        fragments = pack_branches(&fragments);
    }
    fragments
        .pop()
        .unwrap_or_else(|| Node::Leaf(Leaf::from_pairs(&[])))
}

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>
    Branch<LEAF_MAX, BRANCH_MAX, DOC_BYTES>
{
    /// Index of the child that an entry `(key, doc)` routes into — the number of separators `<=` it.
    pub(super) fn child_index(
        &self,
        key: u64,
        doc: u64,
    ) -> usize {
        self.seps.partition_point(|&sep| sep <= (key, doc))
    }

    /// Repair this branch **in place** after its child `child_idx` underflowed, by combining that
    /// child with an adjacent sibling — either merging the pair into one node (dropping a separator)
    /// or re-balancing them into two (updating the separator).
    fn rebalance(
        &mut self,
        child_idx: usize,
    ) {
        // A branch always has ≥ 2 children — [`pack_branches`] never emits a single-child branch and
        // splits/merges preserve this — so the underflowed child always has a sibling to pair with.
        // Guard anyway: with no sibling there is nothing to combine, so let the underflow propagate to
        // the parent (which will merge this branch) rather than indexing `child_idx - 1` out of bounds.
        if self.children.len() < 2 {
            return;
        }
        // Pair the underflowed child with its right neighbour when there is one, otherwise its left.
        let (left_idx, right_idx) = if child_idx + 1 < self.children.len() {
            (child_idx, child_idx + 1)
        } else {
            (child_idx - 1, child_idx)
        };
        // `combine` re-splits the pair in sorted order, so the merged / new-left node keeps the pair's
        // minimum (`min(new left) == min(old left)`). That is why only the **inter-pair** separator
        // `seps[left_idx]` is ever touched below: the separators bracketing the pair from outside
        // (`seps[left_idx - 1] == min(left)` and `seps[right_idx]`, untouched) still point at unchanged
        // minima, so they stay valid without being rewritten.
        match self.children[left_idx].combine(self.seps[left_idx], &self.children[right_idx]) {
            Combined::One(merged) => {
                // Merge: the inter-pair separator vanishes along with the absorbed right child.
                self.children[left_idx] = merged;
                self.children.remove(right_idx);
                self.seps.remove(left_idx);
            }
            Combined::Two(left, sep, right) => {
                // Borrow: only the inter-pair separator moves — to the rebalanced right node's new min.
                self.children[left_idx] = left;
                self.children[right_idx] = right;
                self.seps[left_idx] = sep;
            }
        }
    }
}

impl<const LEAF_MAX: usize, const BRANCH_MAX: usize, const DOC_BYTES: usize>
    Node<LEAF_MAX, BRANCH_MAX, DOC_BYTES>
{
    /// The minimum `(key, doc)` in this subtree — walk the left spine down to its first leaf.
    ///
    /// Panics on an empty subtree. That is the tree's invariant, not an assumption: `strip_empty`
    /// drops emptied children along with their separator, so only a whole-tree root leaf is ever
    /// empty, and a root leaf is never reached through a branch.
    pub(super) fn min(&self) -> (u64, u64) {
        let mut node = self;
        loop {
            match node {
                Self::Leaf(leaf) => return (leaf.key(0), leaf.doc(0)),
                Self::Branch(branch) => node = &branch.children[0],
            }
        }
    }

    /// Apply a sorted `batch` (every entry of which routes into this subtree) by **copying only the
    /// nodes the batch touches**, returning the replacement node(s). A node that receives more
    /// entries than fit in one page splits, so this returns a *list* of fragments; the caller
    /// (the parent branch, or [`build_root`] at the top) stitches them back together.
    pub(super) fn apply_batch(
        &self,
        batch: &[(u64, u64)],
    ) -> Vec<Self> {
        match self {
            // Leaf: merge the batch into this page (the leaf owns the encoding-specific work — an AoS page
            // that still fits is byte-merged, no decode; see [`Leaf::merge_batch`]). It may split into several.
            Self::Leaf(leaf) => leaf
                .merge_batch(batch)
                .into_iter()
                .map(Node::Leaf)
                .collect(),
            // Branch: hand each child the slice of `batch` that routes into it, recursing only into
            // children that actually receive entries — every other child is shared by `Arc` clone,
            // never copied. Re-pack the resulting child list (which may have grown, since a touched
            // child can split into several) back into branch pages.
            Self::Branch(branch) => {
                // At least one replacement child per existing child (a touched one may split into more).
                let mut new_children = Vec::with_capacity(branch.children.len());
                // Split the sorted `batch` across the children with a single linear sweep — a merge of the
                // sorted batch against the sorted separators. `cursor` only ever advances, so a child that
                // receives nothing costs one comparison, not a binary search over the whole remaining batch:
                // routing a localized batch over a wide branch is O(children + batch), not
                // O(children * log batch). (A merge-pass, not galloping — every child must be emitted
                // (touched ones recurse, untouched share by Arc), so the O(children) walk is unavoidable.)
                let mut cursor = 0usize;
                for (child_idx, child) in branch.children.iter().enumerate() {
                    // Each child owns keys strictly below its right separator; the last child (no separator)
                    // owns everything remaining.
                    let child_upper = branch
                        .seps
                        .get(child_idx)
                        .copied()
                        .unwrap_or((u64::MAX, u64::MAX));
                    let start = cursor;
                    while cursor < batch.len() && batch[cursor] < child_upper {
                        cursor += 1;
                    }
                    let for_child = &batch[start..cursor];
                    if for_child.is_empty() {
                        new_children.push(child.clone()); // nothing routes here ⇒ share the page
                    } else {
                        new_children.extend(child.apply_batch(for_child));
                    }
                }
                pack_branches(&new_children)
            }
        }
    }

    /// Insert a single `(key, doc)`, copy-on-write down the touched path: each branch on the path is
    /// cloned into a private copy before it is mutated (see [`make_private`]), so the committed version a
    /// reader may hold is never disturbed. Returns `Some(Split)` when the node split (the parent must take
    /// in the new right sibling), else `None`. Idempotent: inserting an already-present tuple is a no-op.
    ///
    /// Returns `(inserted, split)`: `inserted` is `false` when the tuple was already present (a no-op
    /// on content), so callers can maintain an exact live count; `split` is `Some` when the node split.
    pub(super) fn insert_one(
        &mut self,
        key: u64,
        doc: u64,
    ) -> (bool, Option<Split<LEAF_MAX, BRANCH_MAX, DOC_BYTES>>) {
        match self {
            // The leaf owns the encoding-specific work (an AoS leaf splices its bytes; see [`Leaf::insert`]).
            Self::Leaf(leaf) => match leaf.insert(key, doc) {
                None => (false, None), // already present
                Some(LeafInsert::Fit(new)) => {
                    *leaf = new;
                    (true, None)
                }
                Some(LeafInsert::Split { left, sep, right }) => {
                    *leaf = left;
                    (
                        true,
                        Some(Split {
                            sep,
                            right: Self::Leaf(right),
                        }),
                    )
                }
            },
            Self::Branch(branch_arc) => {
                let branch = make_private(branch_arc); // CoW: clone the shared branch, mutate the copy
                let child_idx = branch.child_index(key, doc);
                let (inserted, child_split) = branch.children[child_idx].insert_one(key, doc);
                let Some(Split { sep, right }) = child_split else {
                    return (inserted, None); // absorbed below — nothing to insert here
                };
                // The child split — take in the promoted separator and the new right sibling beside it.
                branch.seps.insert(child_idx, sep);
                branch.children.insert(child_idx + 1, right);
                #[cfg(test)]
                cow_gate::park_if(key); // test-only: parks AFTER the working copy is mutated
                if branch.children.len() <= BRANCH_MAX {
                    (inserted, None)
                } else {
                    // This branch overflowed in turn: keep the left half, promote the middle
                    // separator (it moves up, into neither side), hand the right half up.
                    let mid = branch.children.len() / 2;
                    let right_children = branch.children.split_off(mid);
                    let right_seps = branch.seps.split_off(mid);
                    let promoted = branch.seps.pop().unwrap(); // the separator between the two halves
                    (
                        inserted,
                        Some(Split {
                            sep: promoted,
                            right: Self::Branch(Arc::new(Branch {
                                seps: right_seps,
                                children: right_children,
                            })),
                        }),
                    )
                }
            }
        }
    }

    /// Combine this node with its right sibling `right` (joined in the parent by separator `sep`),
    /// then re-split: into one node when they fit together (a merge), or two balanced nodes when they
    /// overflow (a borrow). Used to repair an underflow. Both nodes are always the same kind.
    fn combine(
        &self,
        sep: (u64, u64),
        right: &Self,
    ) -> Combined<LEAF_MAX, BRANCH_MAX, DOC_BYTES> {
        match (self, right) {
            (Self::Leaf(left_leaf), Self::Leaf(right_leaf)) => {
                // Two AoS leaves are tag-free `[(key, doc) × n]` and the siblings are ordered, so the join
                // is a byte concat (see [`Node::aos_combine`]). Compact/mixed leaves can't share a
                // `min_value`/width, so they fall back to the decode + rebuild below.
                if let (Leaf::Aos(left_aos), Leaf::Aos(right_aos)) = (left_leaf, right_leaf) {
                    Self::aos_combine(left_aos, right_aos)
                } else {
                    let mut pairs = left_leaf.to_pairs();
                    pairs.extend(right_leaf.to_pairs());
                    if pairs.len() <= LEAF_MAX {
                        Combined::One(Self::Leaf(Leaf::from_pairs(&pairs)))
                    } else {
                        let mid = pairs.len() / 2;
                        Combined::Two(
                            Self::Leaf(Leaf::from_pairs(&pairs[..mid])),
                            pairs[mid],
                            Self::Leaf(Leaf::from_pairs(&pairs[mid..])),
                        )
                    }
                }
            }
            (Self::Branch(left_branch), Self::Branch(right_branch)) => {
                let mut children = left_branch.children.clone();
                children.extend(right_branch.children.iter().cloned());
                let mut seps = left_branch.seps.clone();
                seps.push(sep); // the parent separator becomes an internal one in the joined branch
                seps.extend(right_branch.seps.iter().copied());
                if children.len() <= BRANCH_MAX {
                    Combined::One(Self::Branch(Arc::new(Branch { seps, children })))
                } else {
                    // Reuse the joined `children`/`seps` allocations for the left node — `split_off` hands
                    // back the right halves — instead of three more `to_vec` copies. After the splits, `seps`
                    // still holds the boundary separator at its end; pop it as the promoted one.
                    let mid = children.len() / 2;
                    let right_children = children.split_off(mid);
                    let right_seps = seps.split_off(mid);
                    let promoted = seps
                        .pop()
                        .expect("a split branch always has an internal separator to promote");
                    Combined::Two(
                        Self::Branch(Arc::new(Branch { seps, children })),
                        promoted,
                        Self::Branch(Arc::new(Branch {
                            seps: right_seps,
                            children: right_children,
                        })),
                    )
                }
            }
            _ => unreachable!("siblings are always the same node type"),
        }
    }

    /// Join two ordered AoS sibling leaves. Their buffers are tag-free `[(key, doc) × n]` and `left`
    /// precedes `right`, so the merge is a plain byte concat — no decode. If the result overflows
    /// `LEAF_MAX` we split at the midpoint entry; the fixed `(FIELD + DOC_BYTES)` makes that a balanced split (both
    /// halves `>= LEAF_MAX / 2`), and `mid_sep` (the first entry of the right half) is the new separator.
    fn aos_combine(
        left: &AosLeaf<DOC_BYTES>,
        right: &AosLeaf<DOC_BYTES>,
    ) -> Combined<LEAF_MAX, BRANCH_MAX, DOC_BYTES> {
        let mut buf = Vec::with_capacity(left.0.len() + right.0.len());
        buf.extend_from_slice(&left.0);
        buf.extend_from_slice(&right.0);
        let leaf = |bytes: Vec<u8>| Self::Leaf(Leaf::Aos(AosLeaf(bytes.into())));
        if buf.len() / (FIELD + DOC_BYTES) <= LEAF_MAX {
            Combined::One(leaf(buf))
        } else {
            let split = buf.len() / (FIELD + DOC_BYTES) / 2 * (FIELD + DOC_BYTES);
            let mid_sep = (
                read_u64(&buf, split),
                read_width(&buf, split + FIELD, DOC_BYTES),
            );
            let right_buf = buf.split_off(split);
            Combined::Two(leaf(buf), mid_sep, leaf(right_buf))
        }
    }

    /// Remove a single `(key, doc)`, copy-on-write down the touched path (each branch cloned before
    /// mutation, see [`Node::insert_one`]). Returns `None` if the tuple is absent; otherwise `Some`
    /// carrying `true` when the touched page dropped below its minimum fill, so the parent re-balances it.
    pub(super) fn remove_one(
        &mut self,
        key: u64,
        doc: u64,
    ) -> Option<bool> {
        match self {
            Self::Leaf(leaf) => {
                let (new, underflow) = leaf.remove(key, doc)?; // `None` ⇒ not present
                *leaf = new;
                Some(underflow)
            }
            Self::Branch(branch_arc) => {
                let branch = make_private(branch_arc); // CoW: clone the shared branch, mutate the copy
                let child_idx = branch.child_index(key, doc);
                if branch.children[child_idx].remove_one(key, doc)? {
                    branch.rebalance(child_idx);
                }
                Some(branch.children.len() < BRANCH_MAX / 2)
            }
        }
    }
}

/// Copy-on-write: replace `arc` with a private clone and hand back a mutable reference into it. Unlike
/// `Arc::make_mut` this ALWAYS clones — the writer's working tree shares the committed version's nodes, so
/// every branch it touches must be copied to keep the committed version immutable for any concurrent
/// reader. (We deliberately drop `make_mut`'s in-place path: it only ever fires on a re-touch of a node
/// already privatized within one transaction, and the batch digest path never re-touches.)
fn make_private<T: Clone>(arc: &mut Arc<T>) -> &mut T {
    *arc = Arc::new((**arc).clone());
    Arc::get_mut(arc).expect("uniquely owned right after the clone")
}

/// Test-only synchronization seam (compiled out of release builds). It lets the concurrency test
/// [`tests::a_reader_reads_the_committed_version_lock_free_during_a_write`] freeze a writer in the middle
/// of a copy-on-write mutation — while it still holds the `&mut Branch` of a private copy — so the test
/// can show a reader still sees the committed version intact. The hook fires only for one sentinel key no
/// other test inserts, so it stays inert even though the test binary runs tests in parallel.
#[cfg(test)]
pub(super) mod cow_gate {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

    /// A key no other test inserts — the park hook keys off it so only the concurrency test triggers it.
    pub const KEY: u64 = 0xFFFF_FFFF_FFFF_FF00;
    /// Writer → test: "I am parked mid-mutation, holding the `&mut Branch` after updating it."
    pub static PARKED: AtomicBool = AtomicBool::new(false);
    /// Test → writer: "you may proceed."
    pub static RELEASE: AtomicBool = AtomicBool::new(false);

    /// Called in `insert_one` *after* the working copy's branch has been mutated (its children/seps
    /// updated from a child split). For the sentinel key only, announce that we've parked mid-mutation
    /// (still holding the `&mut Branch`) and spin until released — so the test observes a genuinely
    /// in-flight write, not a freshly-cloned-but-unmodified copy.
    pub(super) fn park_if(key: u64) {
        if key == KEY {
            PARKED.store(true, SeqCst);
            while !RELEASE.load(SeqCst) {
                std::hint::spin_loop();
            }
        }
    }
}
