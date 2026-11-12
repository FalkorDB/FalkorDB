//! A copy-on-write B⁺-tree of `(string, doc)` tuples: the tag kind's trees.
//!
//! The sibling of [`CowBTree`](super::cow_btree::CowBTree) for keys that are byte strings rather
//! than `u64`s. It keeps the same contract — sorted `(key, doc)` tuples, `O(1)` snapshots by root
//! clone, path-copying writes, lazy cursors that own their snapshot — so a column's string trees
//! version exactly as its numeric ones do, and no dictionary is needed to turn strings into keys.
//!
//! - **Ordered by bytes.** Tuples sort by the key's bytes, then the doc. For UTF-8 that is
//!   codepoint order, the order Cypher compares strings in, so a string range is one cursor over
//!   one run of the tree.
//! - **Leaf pages are byte blobs.** A leaf is a self-contained blob, its in-memory form also its
//!   serialized form, as a `CowBTree` leaf is. It stores each distinct string **once**, followed by
//!   its docs, and only the bytes a string does not share with the one before it:
//!
//!   ```text
//!   leaf    := groups:varint restarts:varint offset:u32le{restarts} group*
//!   group   := shared:varint suffix_len:varint suffix:[u8] docs:varint first_doc:varint delta:varint*
//!   ```
//!
//!   So a string held by many entities costs its bytes once per leaf, and a doc a varint delta.
//!   Every [`RESTART_EVERY`]th string is stored whole (`shared = 0`) and its offset kept in the
//!   header, so a seek binary-searches those and decodes at most that many strings, rather than
//!   the leaf from its start.
//! - **Branches are rebuilt from leaves.** A branch holds its children and each child's first
//!   tuple, nothing a leaf does not also say.
//! - **No bloat under churn.** A delete removes the tuple from a copied leaf; an emptied leaf goes,
//!   and a leaf or branch left under a quarter full is merged with a neighbour.
//!
//! A leaf closes at [`LEAF_MAX`] tuples or [`LEAF_BYTES`] bytes, whichever comes first; one string
//! longer than that gets a leaf of its own. A string held by more entities than a leaf takes is
//! split across leaves at a doc boundary, so the tuples in each leaf stay sorted and unique.

use std::cmp::Ordering;
use std::mem::size_of;
use std::ops::Bound;
use std::sync::Arc;

/// Most tuples one leaf holds.
pub const LEAF_MAX: usize = 256;
/// Bytes a leaf stays under, unless one string alone is longer.
pub const LEAF_BYTES: usize = 4096;
/// A leaf under both of these is merged with a neighbour.
const LEAF_MIN: usize = LEAF_MAX / 4;
const LEAF_MIN_BYTES: usize = LEAF_BYTES / 4;
/// Every this many strings, a leaf stores one whole and records where: the points a seek
/// binary-searches.
const RESTART_EVERY: usize = 16;
/// Most children one branch holds; a branch with fewer than [`BRANCH_MIN`] is merged.
const BRANCH_MAX: usize = 128;
const BRANCH_MIN: usize = BRANCH_MAX / 4;

/// A key held by a branch: shared, since the first tuple of a child is copied into every branch
/// rebuild that keeps the child.
type Low = (Arc<[u8]>, u64);

/// A copy-on-write B⁺-tree of `(string, doc)` tuples. `Clone` is an `O(1)` root clone, and the
/// clone is a snapshot no later write to either tree disturbs.
#[derive(Clone, Default)]
pub struct StrBTree {
    root: Option<Arc<Node>>,
}

enum Node {
    Leaf {
        blob: Box<[u8]>,
        len: usize,
    },
    Branch {
        children: Vec<Arc<Node>>,
        /// The first tuple of each child.
        lows: Vec<Low>,
        len: usize,
    },
}

/// One string and its docs, sorted and distinct: a leaf's group, decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Group {
    key: Vec<u8>,
    docs: Vec<u64>,
}

impl Group {
    /// Bytes this group encodes to at most, before its key shares a prefix with the previous one.
    fn max_bytes(&self) -> usize {
        3 * 10 + self.key.len() + 10 * self.docs.len()
    }
}

// ---- varints ------------------------------------------------------------------------------------

fn put_varint(
    out: &mut Vec<u8>,
    mut v: u64,
) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(
    blob: &[u8],
    pos: &mut usize,
) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let b = blob[*pos];
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

// ---- leaves -------------------------------------------------------------------------------------

fn encode_leaf(groups: &[Group]) -> Node {
    let mut data = Vec::new();
    let mut restarts: Vec<u32> = Vec::with_capacity(groups.len().div_ceil(RESTART_EVERY));
    let mut prev: &[u8] = &[];
    let mut len = 0;
    for (i, g) in groups.iter().enumerate() {
        let shared = if i % RESTART_EVERY == 0 {
            restarts.push(u32::try_from(data.len()).expect("a leaf under 4 GiB"));
            0
        } else {
            prev.iter().zip(&g.key).take_while(|(a, b)| a == b).count()
        };
        put_varint(&mut data, shared as u64);
        put_varint(&mut data, (g.key.len() - shared) as u64);
        data.extend_from_slice(&g.key[shared..]);
        put_varint(&mut data, g.docs.len() as u64);
        let mut last = 0;
        for (j, &d) in g.docs.iter().enumerate() {
            put_varint(&mut data, if j == 0 { d } else { d - last });
            last = d;
        }
        len += g.docs.len();
        prev = &g.key;
    }
    let mut out = Vec::with_capacity(data.len() + 4 * restarts.len() + 10);
    put_varint(&mut out, groups.len() as u64);
    put_varint(&mut out, restarts.len() as u64);
    for r in &restarts {
        out.extend_from_slice(&r.to_le_bytes());
    }
    out.extend_from_slice(&data);
    Node::Leaf {
        blob: out.into_boxed_slice(),
        len,
    }
}

/// A leaf's header: its group count, its restart offsets (relative to its first group), and where
/// its first group starts.
struct Header {
    groups: usize,
    restarts: usize,
    offsets: usize,
    data: usize,
}

fn header(blob: &[u8]) -> Header {
    let mut pos = 0;
    let groups = get_varint(blob, &mut pos) as usize;
    let restarts = get_varint(blob, &mut pos) as usize;
    Header {
        groups,
        restarts,
        offsets: pos,
        data: pos + 4 * restarts,
    }
}

/// The offset of restart `r`, from the leaf's first group.
fn restart_offset(
    blob: &[u8],
    h: &Header,
    r: usize,
) -> usize {
    let at = h.offsets + 4 * r;
    u32::from_le_bytes(blob[at..at + 4].try_into().expect("four bytes")) as usize
}

/// The whole key stored at restart `r`.
fn restart_key<'b>(
    blob: &'b [u8],
    h: &Header,
    r: usize,
) -> &'b [u8] {
    let mut pos = h.data + restart_offset(blob, h, r);
    let _shared = get_varint(blob, &mut pos);
    let n = get_varint(blob, &mut pos) as usize;
    &blob[pos..pos + n]
}

fn decode_leaf(blob: &[u8]) -> Vec<Group> {
    let h = header(blob);
    let mut pos = h.data;
    let n = h.groups;
    let mut groups: Vec<Group> = Vec::with_capacity(n);
    let mut key: Vec<u8> = Vec::new();
    for _ in 0..n {
        let shared = get_varint(blob, &mut pos) as usize;
        let suffix = get_varint(blob, &mut pos) as usize;
        key.truncate(shared);
        key.extend_from_slice(&blob[pos..pos + suffix]);
        pos += suffix;
        let nd = get_varint(blob, &mut pos) as usize;
        let mut docs = Vec::with_capacity(nd);
        let mut last = 0;
        for i in 0..nd {
            let v = get_varint(blob, &mut pos);
            last = if i == 0 { v } else { last + v };
            docs.push(last);
        }
        groups.push(Group {
            key: key.clone(),
            docs,
        });
    }
    groups
}

/// Pack `groups` (sorted, distinct keys) into leaves of at most [`LEAF_MAX`] tuples and about
/// [`LEAF_BYTES`] bytes. A trailing leaf under the merge threshold shares the tuples of the one
/// before it evenly, so no build leaves an underfull leaf behind.
fn build_leaves(groups: Vec<Group>) -> Vec<Arc<Node>> {
    let mut chunks: Vec<Vec<Group>> = Vec::new();
    let mut cur: Vec<Group> = Vec::new();
    let (mut tuples, mut bytes) = (0usize, 0usize);
    for g in groups {
        let mut docs = &g.docs[..];
        while !docs.is_empty() {
            let key_cost = 30 + g.key.len();
            if tuples > 0 && (tuples == LEAF_MAX || bytes + key_cost + 10 > LEAF_BYTES) {
                chunks.push(std::mem::take(&mut cur));
                (tuples, bytes) = (0, 0);
            }
            let by_bytes = LEAF_BYTES.saturating_sub(bytes + key_cost) / 10;
            let n = docs.len().min(LEAF_MAX - tuples).min(by_bytes.max(1));
            cur.push(Group {
                key: g.key.clone(),
                docs: docs[..n].to_vec(),
            });
            tuples += n;
            bytes += key_cost + 10 * n;
            docs = &docs[n..];
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    if chunks.len() >= 2 {
        let last = chunks.last().expect("two chunks");
        if underfull_groups(last) {
            let b = chunks.pop().expect("two chunks");
            let a = chunks.pop().expect("two chunks");
            let both = concat_groups(a, b);
            let total: usize = both.iter().map(|g| g.docs.len()).sum();
            let (l, r) = split_groups(both, total / 2);
            chunks.push(l);
            if !r.is_empty() {
                chunks.push(r);
            }
        }
    }
    chunks.iter().map(|c| Arc::new(encode_leaf(c))).collect()
}

fn underfull_groups(groups: &[Group]) -> bool {
    let tuples: usize = groups.iter().map(|g| g.docs.len()).sum();
    let bytes: usize = groups.iter().map(Group::max_bytes).sum();
    tuples < LEAF_MIN && bytes < LEAF_MIN_BYTES
}

/// `a` then `b`, joining a string that ends `a` and starts `b`.
fn concat_groups(
    mut a: Vec<Group>,
    b: Vec<Group>,
) -> Vec<Group> {
    for g in b {
        match a.last_mut() {
            Some(last) if last.key == g.key => last.docs.extend(g.docs),
            _ => a.push(g),
        }
    }
    a
}

/// `groups` split after `at` tuples.
fn split_groups(
    groups: Vec<Group>,
    at: usize,
) -> (Vec<Group>, Vec<Group>) {
    let (mut l, mut r) = (Vec::new(), Vec::new());
    let mut seen = 0;
    for mut g in groups {
        if seen >= at {
            r.push(g);
        } else if seen + g.docs.len() <= at {
            seen += g.docs.len();
            l.push(g);
        } else {
            let tail = g.docs.split_off(at - seen);
            seen = at;
            r.push(Group {
                key: g.key.clone(),
                docs: tail,
            });
            l.push(g);
        }
    }
    (l, r)
}

/// `ops` (sorted, distinct tuples) as groups.
fn group_ops(ops: &[(&[u8], u64)]) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    for &(k, d) in ops {
        match out.last_mut() {
            Some(g) if g.key == k => g.docs.push(d),
            _ => out.push(Group {
                key: k.to_vec(),
                docs: vec![d],
            }),
        }
    }
    out
}

/// `old` with `ops` added (`add`) or removed, both sorted by key.
fn merge_groups(
    old: Vec<Group>,
    ops: Vec<Group>,
    add: bool,
) -> Vec<Group> {
    let mut out = Vec::with_capacity(old.len() + if add { ops.len() } else { 0 });
    let mut ops = ops.into_iter().peekable();
    for g in old {
        while let Some(op) = ops.next_if(|op| op.key < g.key) {
            if add {
                out.push(op);
            }
        }
        match ops.next_if(|op| op.key == g.key) {
            Some(op) => {
                let docs = if add {
                    union(&g.docs, &op.docs)
                } else {
                    difference(&g.docs, &op.docs)
                };
                if !docs.is_empty() {
                    out.push(Group { key: g.key, docs });
                }
            }
            None => out.push(g),
        }
    }
    if add {
        out.extend(ops);
    }
    out
}

fn union(
    a: &[u64],
    b: &[u64],
) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn difference(
    a: &[u64],
    b: &[u64],
) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len());
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j < b.len() && b[j] == x {
            continue;
        }
        out.push(x);
    }
    out
}

// ---- nodes --------------------------------------------------------------------------------------

impl Node {
    fn len(&self) -> usize {
        match self {
            Self::Leaf { len, .. } | Self::Branch { len, .. } => *len,
        }
    }

    /// The node's first tuple. Every node in a tree holds at least one.
    fn first(&self) -> Low {
        match self {
            Self::Leaf { blob, .. } => {
                let mut pos = header(blob).data;
                let _shared = get_varint(blob, &mut pos);
                let n = get_varint(blob, &mut pos) as usize;
                let key: Arc<[u8]> = Arc::from(&blob[pos..pos + n]);
                pos += n;
                let _docs = get_varint(blob, &mut pos);
                (key, get_varint(blob, &mut pos))
            }
            Self::Branch { lows, .. } => lows[0].clone(),
        }
    }

    fn underfull(&self) -> bool {
        match self {
            Self::Leaf { blob, len } => *len < LEAF_MIN && blob.len() < LEAF_MIN_BYTES,
            Self::Branch { children, .. } => children.len() < BRANCH_MIN,
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }
}

/// `children` (all one height) under as few branches as hold them, each with at most
/// [`BRANCH_MAX`] and the children spread evenly.
fn build_branches(children: Vec<(Arc<Node>, Low)>) -> Vec<(Arc<Node>, Low)> {
    if children.is_empty() {
        return Vec::new();
    }
    let n = children.len();
    let parts = n.div_ceil(BRANCH_MAX);
    let mut out = Vec::with_capacity(parts);
    let mut it = children.into_iter();
    for p in 0..parts {
        let take = n / parts + usize::from(p < n % parts);
        let (kids, lows): (Vec<_>, Vec<_>) = it.by_ref().take(take).unzip();
        let len = kids.iter().map(|k| k.len()).sum();
        let low = lows[0].clone();
        out.push((
            Arc::new(Node::Branch {
                children: kids,
                lows,
                len,
            }),
            low,
        ));
    }
    out
}

/// Merge each underfull node of `nodes` (one height, in order) with a neighbour.
fn fix_underflow(nodes: Vec<(Arc<Node>, Low)>) -> Vec<(Arc<Node>, Low)> {
    if nodes.len() < 2 || !nodes.iter().any(|(n, _)| n.underfull()) {
        return nodes;
    }
    let leaves = nodes[0].0.is_leaf();
    let mut out: Vec<(Arc<Node>, Low)> = Vec::with_capacity(nodes.len());
    let mut it = nodes.into_iter().peekable();
    while let Some((node, low)) = it.next() {
        if node.underfull() || it.peek().is_some_and(|(next, _)| next.underfull()) {
            if let Some((partner, _)) = it.next() {
                out.extend(combine(leaves, &node, &partner));
                continue;
            }
            // The last node, underfull: merge it into the one before.
            if let Some((prev, _)) = out.pop() {
                out.extend(combine(leaves, &prev, &node));
                continue;
            }
        }
        out.push((node, low));
    }
    out
}

/// Two neighbours of one height as one or two balanced nodes.
fn combine(
    leaves: bool,
    a: &Node,
    b: &Node,
) -> Vec<(Arc<Node>, Low)> {
    if leaves {
        let (Node::Leaf { blob: ba, .. }, Node::Leaf { blob: bb, .. }) = (a, b) else {
            unreachable!("neighbours are one height")
        };
        with_lows(build_leaves(concat_groups(
            decode_leaf(ba),
            decode_leaf(bb),
        )))
    } else {
        let (
            Node::Branch {
                children: ca,
                lows: la,
                ..
            },
            Node::Branch {
                children: cb,
                lows: lb,
                ..
            },
        ) = (a, b)
        else {
            unreachable!("neighbours are one height")
        };
        let kids = ca
            .iter()
            .cloned()
            .zip(la.iter().cloned())
            .chain(cb.iter().cloned().zip(lb.iter().cloned()))
            .collect();
        build_branches(kids)
    }
}

fn with_lows(nodes: Vec<Arc<Node>>) -> Vec<(Arc<Node>, Low)> {
    nodes
        .into_iter()
        .map(|n| {
            let low = n.first();
            (n, low)
        })
        .collect()
}

fn cmp_tuple(
    (k, d): (&[u8], u64),
    (lk, ld): (&[u8], u64),
) -> Ordering {
    k.cmp(lk).then(d.cmp(&ld))
}

/// `node` with `ops` (sorted, distinct) applied: the nodes of its height that replace it, in
/// order, possibly none.
fn apply(
    node: &Arc<Node>,
    ops: &[(&[u8], u64)],
    add: bool,
) -> Vec<(Arc<Node>, Low)> {
    match &**node {
        Node::Leaf { blob, .. } => with_lows(build_leaves(merge_groups(
            decode_leaf(blob),
            group_ops(ops),
            add,
        ))),
        Node::Branch { children, lows, .. } => {
            let mut out = Vec::with_capacity(children.len() + 2);
            let mut i = 0;
            for (c, child) in children.iter().enumerate() {
                let end = match lows.get(c + 1) {
                    Some((lk, ld)) => {
                        i + ops[i..].partition_point(|&(k, d)| {
                            cmp_tuple((k, d), (lk, *ld)) == Ordering::Less
                        })
                    }
                    None => ops.len(),
                };
                if end > i {
                    out.extend(apply(child, &ops[i..end], add));
                } else {
                    out.push((child.clone(), lows[c].clone()));
                }
                i = end;
            }
            build_branches(fix_underflow(out))
        }
    }
}

// ---- the tree -----------------------------------------------------------------------------------

impl StrBTree {
    /// Whether the tree holds no tuples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// How many tuples the tree holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.len())
    }

    /// A tree of `tuples`, which are sorted and distinct, built bottom-up.
    #[must_use]
    pub fn from_sorted(tuples: &[(&[u8], u64)]) -> Self {
        let mut tree = Self::default();
        tree.insert_batch(tuples);
        tree
    }

    /// Add `tuples`, sorted and distinct. A tuple already held is left as it is.
    pub fn insert_batch(
        &mut self,
        tuples: &[(&[u8], u64)],
    ) {
        if tuples.is_empty() {
            return;
        }
        let level = match &self.root {
            Some(root) => apply(root, tuples, true),
            None => with_lows(build_leaves(group_ops(tuples))),
        };
        self.set_root(level);
    }

    /// Remove `tuples`, sorted and distinct. A tuple not held is skipped.
    pub fn remove_batch(
        &mut self,
        tuples: &[(&[u8], u64)],
    ) {
        let Some(root) = &self.root else {
            return;
        };
        if tuples.is_empty() {
            return;
        }
        let level = apply(root, tuples, false);
        self.set_root(level);
    }

    /// Stack `level` (one height, in order) under branches until one node is left, then drop any
    /// branch with a single child on top.
    fn set_root(
        &mut self,
        mut level: Vec<(Arc<Node>, Low)>,
    ) {
        while level.len() > 1 {
            level = build_branches(level);
        }
        let Some((mut root, _)) = level.pop() else {
            self.root = None;
            return;
        };
        while let Node::Branch { children, .. } = &*root
            && children.len() == 1
        {
            root = children[0].clone();
        }
        self.root = Some(root);
    }

    /// The docs under `key`, in doc order.
    #[must_use]
    pub fn point(
        &self,
        key: &[u8],
    ) -> StrRange {
        let k: Arc<[u8]> = Arc::from(key);
        self.range(Bound::Included(k.clone()), Bound::Included(k))
    }

    /// The docs under every key from `lo` to `hi`, in key then doc order.
    #[must_use]
    pub fn range(
        &self,
        lo: Bound<Arc<[u8]>>,
        hi: Bound<Arc<[u8]>>,
    ) -> StrRange {
        StrRange::new(self.root.clone(), lo, hi)
    }

    /// Heap bytes the tree's nodes hold: every node it reaches, counted once, whether or not
    /// another version shares it.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        fn node(n: &Node) -> usize {
            size_of::<Node>()
                + match n {
                    Node::Leaf { blob, .. } => blob.len(),
                    Node::Branch { children, lows, .. } => {
                        children.capacity() * size_of::<Arc<Node>>()
                            + lows.capacity() * size_of::<Low>()
                            + lows.iter().map(|(k, _)| k.len() + 16).sum::<usize>()
                            + children.iter().map(|c| node(c)).sum::<usize>()
                    }
                }
        }
        self.root.as_ref().map_or(0, |r| node(r))
    }

    /// Every tuple, in order. For tests.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(Vec<u8>, u64)> {
        fn walk(
            n: &Node,
            out: &mut Vec<(Vec<u8>, u64)>,
        ) {
            match n {
                Node::Leaf { blob, .. } => {
                    for g in decode_leaf(blob) {
                        out.extend(g.docs.iter().map(|&d| (g.key.clone(), d)));
                    }
                }
                Node::Branch { children, .. } => children.iter().for_each(|c| walk(c, out)),
            }
        }
        let mut out = Vec::new();
        if let Some(r) = &self.root {
            walk(r, &mut out);
        }
        out
    }

    /// Every leaf's size in tuples and bytes, and the tree's height. For tests.
    #[cfg(test)]
    fn shape(&self) -> (Vec<(usize, usize)>, usize) {
        fn walk(
            n: &Node,
            depth: usize,
            out: &mut Vec<(usize, usize)>,
            heights: &mut Vec<usize>,
        ) {
            match n {
                Node::Leaf { blob, len } => {
                    out.push((*len, blob.len()));
                    heights.push(depth);
                }
                Node::Branch { children, lows, .. } => {
                    for (c, low) in children.iter().zip(lows) {
                        assert_eq!(&c.first(), low, "a branch's low is its child's first tuple");
                        walk(c, depth + 1, out, heights);
                    }
                }
            }
        }
        let (mut leaves, mut heights) = (Vec::new(), Vec::new());
        if let Some(r) = &self.root {
            walk(r, 0, &mut leaves, &mut heights);
        }
        heights.dedup();
        assert!(heights.len() <= 1, "every leaf is at one depth");
        (leaves, heights.first().copied().unwrap_or(0))
    }
}

// ---- cursors ------------------------------------------------------------------------------------

/// Where a cursor is inside one leaf.
#[derive(Default)]
struct LeafPos {
    pos: usize,
    groups_left: usize,
    docs_left: usize,
    prev_doc: u64,
    /// The current group's key.
    key: Vec<u8>,
    /// The current group's first doc, not yet yielded.
    pending: Option<u64>,
}

/// The docs of the tuples from `lo` to `hi`, lazily, over the snapshot the cursor was made on.
pub struct StrRange {
    /// The branches from the root to the current leaf, with the child taken in each.
    path: Vec<(Arc<Node>, usize)>,
    leaf: Option<Arc<Node>>,
    at: LeafPos,
    hi: Bound<Arc<[u8]>>,
    done: bool,
}

impl StrRange {
    fn new(
        root: Option<Arc<Node>>,
        lo: Bound<Arc<[u8]>>,
        hi: Bound<Arc<[u8]>>,
    ) -> Self {
        let mut it = Self {
            path: Vec::new(),
            leaf: None,
            at: LeafPos::default(),
            hi,
            done: true,
        };
        let Some(mut node) = root else {
            return it;
        };
        // Descend to the leaf holding the first tuple at or after `lo`.
        let target: Option<(&[u8], u64)> = match &lo {
            Bound::Included(k) => Some((k, 0)),
            Bound::Excluded(k) => Some((k, u64::MAX)),
            Bound::Unbounded => None,
        };
        loop {
            let next = match &*node {
                Node::Leaf { .. } => break,
                Node::Branch { children, lows, .. } => {
                    let c = target.map_or(0, |t| {
                        lows.partition_point(|(k, d)| cmp_tuple((k, *d), t) != Ordering::Greater)
                            .saturating_sub(1)
                    });
                    it.path.push((node.clone(), c));
                    children[c].clone()
                }
            };
            node = next;
        }
        it.enter(node);
        it.done = false;
        let before_lo = |key: &[u8]| match &lo {
            Bound::Included(k) => key < &k[..],
            Bound::Excluded(k) => key <= &k[..],
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
        leaf: Arc<Node>,
    ) {
        let Node::Leaf { blob, .. } = &*leaf else {
            unreachable!("a cursor enters leaves")
        };
        let h = header(blob);
        self.at = LeafPos {
            pos: h.data,
            groups_left: h.groups,
            ..LeafPos::default()
        };
        self.leaf = Some(leaf);
    }

    /// Move the cursor, just entered into a leaf, to the last restart whose string comes before
    /// `lo`: the strings from there to the first one at or after `lo` are fewer than a restart
    /// interval.
    fn seek_restart(
        &mut self,
        before_lo: impl Fn(&[u8]) -> bool,
    ) {
        let leaf = self.leaf.clone().expect("a cursor in a leaf");
        let Node::Leaf { blob, .. } = &*leaf else {
            unreachable!("a cursor enters leaves")
        };
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
        let r = lo;
        if r > 1 {
            let r = r - 1;
            self.at.pos = h.data + restart_offset(blob, &h, r);
            self.at.groups_left = h.groups - r * RESTART_EVERY;
            self.at.key.clear();
        }
    }

    /// Move to the next leaf, if any.
    fn next_leaf(&mut self) -> bool {
        while let Some((branch, c)) = self.path.pop() {
            let Node::Branch { children, .. } = &*branch else {
                unreachable!("the path holds branches")
            };
            if c + 1 < children.len() {
                let mut node = children[c + 1].clone();
                self.path.push((branch.clone(), c + 1));
                while let Node::Branch { children, .. } = &*node {
                    let first = children[0].clone();
                    self.path.push((node.clone(), 0));
                    node = first;
                }
                self.enter(node);
                return true;
            }
        }
        false
    }

    /// Read the next group's header and first doc, moving to the next leaf if this one is spent.
    fn load_group(&mut self) -> bool {
        loop {
            if self.at.groups_left > 0 {
                let leaf = self.leaf.clone().expect("a cursor in a leaf");
                let Node::Leaf { blob, .. } = &*leaf else {
                    unreachable!("a cursor enters leaves")
                };
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
            if !self.next_leaf() {
                return false;
            }
        }
    }

    /// Skip the rest of the current group and load the next one.
    fn skip_group(&mut self) -> bool {
        if self.at.docs_left > 0 {
            let leaf = self.leaf.clone().expect("a cursor in a leaf");
            let Node::Leaf { blob, .. } = &*leaf else {
                unreachable!("a cursor enters leaves")
            };
            for _ in 0..self.at.docs_left {
                get_varint(blob, &mut self.at.pos);
            }
            self.at.docs_left = 0;
        }
        self.at.pending = None;
        self.load_group()
    }

    fn past_hi(&self) -> bool {
        let key = &self.at.key[..];
        match &self.hi {
            Bound::Included(k) => key > &k[..],
            Bound::Excluded(k) => key >= &k[..],
            Bound::Unbounded => false,
        }
    }
}

impl Iterator for StrRange {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.done {
            return None;
        }
        if let Some(d) = self.at.pending.take() {
            return Some(d);
        }
        if self.at.docs_left > 0 {
            let leaf = self.leaf.clone().expect("a cursor in a leaf");
            let Node::Leaf { blob, .. } = &*leaf else {
                unreachable!("a cursor enters leaves")
            };
            self.at.prev_doc += get_varint(blob, &mut self.at.pos);
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
pub struct StrRanges {
    tree: StrBTree,
    ranges: std::vec::IntoIter<(Bound<Arc<[u8]>>, Bound<Arc<[u8]>>)>,
    current: Option<StrRange>,
}

impl StrRanges {
    /// The docs of `ranges`, which must not overlap, so no doc is yielded twice.
    #[must_use]
    pub fn new(
        tree: &StrBTree,
        ranges: Vec<(Bound<Arc<[u8]>>, Bound<Arc<[u8]>>)>,
    ) -> Self {
        Self {
            tree: tree.clone(),
            ranges: ranges.into_iter(),
            current: None,
        }
    }
}

impl Iterator for StrRanges {
    type Item = u64;

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::BTreeSet;

    fn key(i: u32) -> Vec<u8> {
        // Shared prefixes, lengths 0..40, an occasional NUL: what prefix compression must survive.
        let mut k = format!("{:x}", i % 997).into_bytes();
        if i.is_multiple_of(7) {
            k.push(0);
        }
        if i.is_multiple_of(11) {
            k.extend(std::iter::repeat_n(b'z', (i % 40) as usize));
        }
        k
    }

    fn sorted(set: &BTreeSet<(Vec<u8>, u64)>) -> Vec<(Vec<u8>, u64)> {
        set.iter().cloned().collect()
    }

    fn as_ops(v: &[(Vec<u8>, u64)]) -> Vec<(&[u8], u64)> {
        v.iter().map(|(k, d)| (&k[..], *d)).collect()
    }

    fn model_range(
        set: &BTreeSet<(Vec<u8>, u64)>,
        lo: &Bound<Arc<[u8]>>,
        hi: &Bound<Arc<[u8]>>,
    ) -> Vec<u64> {
        set.iter()
            .filter(|(k, _)| match lo {
                Bound::Included(l) => k[..] >= l[..],
                Bound::Excluded(l) => k[..] > l[..],
                Bound::Unbounded => true,
            })
            .filter(|(k, _)| match hi {
                Bound::Included(h) => k[..] <= h[..],
                Bound::Excluded(h) => k[..] < h[..],
                Bound::Unbounded => true,
            })
            .map(|(_, d)| *d)
            .collect()
    }

    fn bound(
        rng: &mut StdRng,
        k: Vec<u8>,
    ) -> Bound<Arc<[u8]>> {
        match rng.random_range(0..3) {
            0 => Bound::Included(Arc::from(k)),
            1 => Bound::Excluded(Arc::from(k)),
            _ => Bound::Unbounded,
        }
    }

    /// Random batches of inserts and removes against a set: the tuples, every kind of range, the
    /// shape (one depth, lows that match, no leaf over its limits), and that a snapshot taken
    /// before a batch still reads as it did.
    #[test]
    fn matches_a_sorted_set_under_random_batches() {
        let mut rng = StdRng::seed_from_u64(0x5eed);
        let mut tree = StrBTree::default();
        let mut model: BTreeSet<(Vec<u8>, u64)> = BTreeSet::new();
        for round in 0..300 {
            let snapshot = (tree.clone(), sorted(&model));
            let add = rng.random_range(0..3) != 0;
            let n = rng.random_range(1..400);
            let mut batch: BTreeSet<(Vec<u8>, u64)> = BTreeSet::new();
            for _ in 0..n {
                let t = if !add && !model.is_empty() && rng.random_range(0..4) != 0 {
                    let i = rng.random_range(0..model.len());
                    model.iter().nth(i).cloned().expect("in range")
                } else {
                    // Hot strings get many docs, so a string spans leaves.
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
            if add {
                tree.insert_batch(&as_ops(&batch));
                model.extend(batch.iter().cloned());
            } else {
                tree.remove_batch(&as_ops(&batch));
                for t in &batch {
                    model.remove(t);
                }
            }
            assert_eq!(tree.tuples(), sorted(&model), "round {round}");
            assert_eq!(tree.len(), model.len());
            assert_eq!(
                snapshot.0.tuples(),
                snapshot.1,
                "a snapshot is not disturbed"
            );
            let (leaves, _) = tree.shape();
            for (t, b) in &leaves {
                assert!(*t <= LEAF_MAX && *t > 0);
                assert!(*b <= LEAF_BYTES + 64, "a leaf of {b} bytes");
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
            assert_eq!(tree.point(&k).collect::<Vec<_>>(), want);
        }
        // Remove everything: the tree is empty, not a tree of empty leaves.
        let all = sorted(&model);
        tree.remove_batch(&as_ops(&all));
        assert!(tree.is_empty());
    }

    /// One string held by many entities spans leaves, and is still read whole by a point lookup.
    #[test]
    fn a_string_held_by_many_entities_spans_leaves() {
        let k = b"status:pending".to_vec();
        let tuples: Vec<(Vec<u8>, u64)> = (0..10_000u64).map(|d| (k.clone(), d * 3)).collect();
        let tree = StrBTree::from_sorted(&as_ops(&tuples));
        let (leaves, height) = tree.shape();
        assert!(leaves.len() > 1 && height >= 1);
        assert_eq!(
            tree.point(&k).collect::<Vec<_>>(),
            (0..10_000u64).map(|d| d * 3).collect::<Vec<_>>()
        );
        assert_eq!(tree.point(b"status:pendin").count(), 0);
        assert_eq!(tree.point(b"status:pending!").count(), 0);
    }

    /// A string longer than a leaf's byte budget gets a leaf of its own, and is found.
    #[test]
    fn a_string_longer_than_a_leaf() {
        let long = vec![b'q'; 3 * LEAF_BYTES];
        let tuples = vec![(b"a".to_vec(), 1), (long.clone(), 2), (b"r".to_vec(), 3)];
        let tree = StrBTree::from_sorted(&as_ops(&tuples));
        assert_eq!(tree.point(&long).collect::<Vec<_>>(), vec![2]);
        assert_eq!(tree.tuples(), tuples);
    }

    /// Each string is stored once per leaf, so a repeated string costs about a doc per entity.
    #[test]
    fn a_repeated_string_costs_about_a_doc_per_entity() {
        let k = b"a-fairly-long-repeated-tag-value".to_vec();
        let tuples: Vec<(Vec<u8>, u64)> = (0..100_000u64).map(|d| (k.clone(), d)).collect();
        let tree = StrBTree::from_sorted(&as_ops(&tuples));
        let per = tree.heap_bytes() as f64 / 100_000.0;
        assert!(per < 2.0, "{per} bytes per tuple");
    }
}
