//! The string-keyed page: `(string, doc)` tuples, each string stored once.
//!
//! ```text
//! leaf    := groups:varint restarts:varint offset:u32le{restarts} group*
//! group   := shared:varint suffix_len:varint suffix:[u8] docs:varint first_doc:varint delta:varint*
//! ```
//!
//! A *group* is one distinct string and the docs that hold it, in order. A string is stored as the
//! bytes it does not share with the string before it, so a run of similar strings costs little;
//! its docs are varint deltas. Every [`RESTART_EVERY`]th string is stored whole and its offset kept
//! in the header, so a seek binary-searches those and decodes at most that many strings.
//!
//! Strings sort by their bytes, which for UTF-8 is codepoint order, the order Cypher compares
//! strings in. A page closes at [`LEAF_MAX`] tuples or [`LEAF_BYTES`] bytes, whichever comes
//! first; one string longer than that gets a page of its own. A string held by more entities than a
//! page takes is split across pages at a doc boundary.

use std::cmp::Ordering;
use std::sync::Arc;

use super::page::{Page, PageCombined, PageInsert, Tuple};

/// Most tuples one page holds.
pub const LEAF_MAX: usize = 256;
/// Bytes a page stays under, unless one string alone is longer.
pub const LEAF_BYTES: usize = 4096;
/// A page under both of these is merged with a neighbour.
const LEAF_MIN: usize = LEAF_MAX / 4;
const LEAF_MIN_BYTES: usize = LEAF_BYTES / 4;
/// Every this many strings, a page stores one whole and records where.
pub(super) const RESTART_EVERY: usize = 16;

/// A page of `(string, doc)` tuples. `Clone` is an `Arc` bump.
#[derive(Clone)]
pub struct StrLeaf {
    pub(super) blob: Arc<[u8]>,
    len: usize,
}

/// One string and its docs, sorted and distinct: a page's group, decoded.
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

pub(super) fn get_varint(
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

// ---- the byte format ----------------------------------------------------------------------------

/// A page's header: its group count, its restart offsets (relative to its first group), and where
/// its first group starts.
pub(super) struct Header {
    pub(super) groups: usize,
    pub(super) restarts: usize,
    offsets: usize,
    pub(super) data: usize,
}

pub(super) fn header(blob: &[u8]) -> Header {
    // An empty page is an empty blob: nothing to store, not even a header.
    if blob.is_empty() {
        return Header {
            groups: 0,
            restarts: 0,
            offsets: 0,
            data: 0,
        };
    }
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

/// The offset of restart `r`, from the page's first group.
pub(super) fn restart_offset(
    blob: &[u8],
    h: &Header,
    r: usize,
) -> usize {
    let at = h.offsets + 4 * r;
    u32::from_le_bytes(blob[at..at + 4].try_into().expect("four bytes")) as usize
}

/// The whole key stored at restart `r`.
pub(super) fn restart_key<'b>(
    blob: &'b [u8],
    h: &Header,
    r: usize,
) -> &'b [u8] {
    let mut pos = h.data + restart_offset(blob, h, r);
    let _shared = get_varint(blob, &mut pos);
    let n = get_varint(blob, &mut pos) as usize;
    &blob[pos..pos + n]
}

fn encode(groups: &[Group]) -> StrLeaf {
    if groups.is_empty() {
        return StrLeaf {
            blob: Arc::from([]),
            len: 0,
        };
    }
    let mut data = Vec::new();
    let mut restarts: Vec<u32> = Vec::with_capacity(groups.len().div_ceil(RESTART_EVERY));
    let mut prev: &[u8] = &[];
    let mut len = 0;
    for (i, g) in groups.iter().enumerate() {
        let shared = if i % RESTART_EVERY == 0 {
            restarts.push(u32::try_from(data.len()).expect("a page under 4 GiB"));
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
    StrLeaf {
        blob: Arc::from(out),
        len,
    }
}

fn decode(blob: &[u8]) -> Vec<Group> {
    let h = header(blob);
    let mut pos = h.data;
    let mut groups: Vec<Group> = Vec::with_capacity(h.groups);
    let mut key: Vec<u8> = Vec::new();
    for _ in 0..h.groups {
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

/// Pack `groups` (sorted, distinct keys) into pages of at most [`LEAF_MAX`] tuples and about
/// [`LEAF_BYTES`] bytes. A trailing page under the merge threshold shares the tuples of the one
/// before it evenly, so no build leaves an underfull page behind.
fn pack(groups: Vec<Group>) -> Vec<StrLeaf> {
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
    if chunks.len() >= 2 && underfull_groups(chunks.last().expect("two chunks")) {
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
    chunks.iter().map(|c| encode(c)).collect()
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

/// `tuples` (sorted, distinct) as groups.
fn group_tuples(tuples: &[Tuple<Arc<String>>]) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    for (k, d) in tuples {
        match out.last_mut() {
            Some(g) if g.key == k.as_bytes() => g.docs.push(*d),
            _ => out.push(Group {
                key: k.as_bytes().to_vec(),
                docs: vec![*d],
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

fn as_key(bytes: &[u8]) -> Arc<String> {
    Arc::new(String::from_utf8(bytes.to_vec()).expect("a page holds UTF-8 strings"))
}

impl StrLeaf {
    /// Whether `groups` fit one page.
    fn fits(page: &Self) -> bool {
        page.len <= LEAF_MAX && page.blob.len() <= LEAF_BYTES
    }

    /// Every tuple, in order. For tests.
    #[cfg(test)]
    pub(super) fn tuples(&self) -> Vec<(Vec<u8>, u64)> {
        decode(&self.blob)
            .into_iter()
            .flat_map(|g| g.docs.into_iter().map(move |d| (g.key.clone(), d)))
            .collect()
    }

    /// The page's size in tuples and bytes. For tests.
    #[cfg(test)]
    pub(super) fn shape(&self) -> (usize, usize) {
        (self.len, self.blob.len())
    }
}

impl Page for StrLeaf {
    type Key = Arc<String>;

    fn empty() -> Self {
        encode(&[])
    }

    fn build(tuples: &[Tuple<Arc<String>>]) -> Vec<Self> {
        pack(group_tuples(tuples))
    }

    fn count(&self) -> usize {
        self.len
    }

    fn min(&self) -> Tuple<Arc<String>> {
        let h = header(&self.blob);
        let mut pos = h.data;
        let _shared = get_varint(&self.blob, &mut pos);
        let n = get_varint(&self.blob, &mut pos) as usize;
        let key = as_key(&self.blob[pos..pos + n]);
        pos += n;
        let _docs = get_varint(&self.blob, &mut pos);
        (key, get_varint(&self.blob, &mut pos))
    }

    fn is_underfull(&self) -> bool {
        self.len < LEAF_MIN && self.blob.len() < LEAF_MIN_BYTES
    }

    fn insert(
        &self,
        key: &Arc<String>,
        doc: u64,
    ) -> Option<PageInsert<Self>> {
        let groups = decode(&self.blob);
        let held = groups
            .iter()
            .find(|g| g.key == key.as_bytes())
            .is_some_and(|g| g.docs.binary_search(&doc).is_ok());
        if held {
            return None;
        }
        let mut pages = pack(merge_groups(
            groups,
            vec![Group {
                key: key.as_bytes().to_vec(),
                docs: vec![doc],
            }],
            true,
        ))
        .into_iter();
        let left = pages.next().expect("a page holding a tuple");
        let rights: Vec<_> = pages.map(|p| (p.min(), p)).collect();
        Some(if rights.is_empty() {
            PageInsert::Fit(left)
        } else {
            PageInsert::Split { left, rights }
        })
    }

    fn remove(
        &self,
        key: &Arc<String>,
        doc: u64,
    ) -> Option<(Self, bool)> {
        let groups = decode(&self.blob);
        let held = groups
            .iter()
            .find(|g| g.key == key.as_bytes())
            .is_some_and(|g| g.docs.binary_search(&doc).is_ok());
        if !held {
            return None;
        }
        let page = encode(&merge_groups(
            groups,
            vec![Group {
                key: key.as_bytes().to_vec(),
                docs: vec![doc],
            }],
            false,
        ));
        let underfull = page.is_underfull();
        Some((page, underfull))
    }

    fn merge_batch(
        &self,
        batch: &[Tuple<Arc<String>>],
    ) -> Vec<Self> {
        pack(merge_groups(decode(&self.blob), group_tuples(batch), true))
    }

    fn remove_batch(
        &self,
        batch: &[Tuple<Arc<String>>],
    ) -> Self {
        encode(&merge_groups(
            decode(&self.blob),
            group_tuples(batch),
            false,
        ))
    }

    /// The two pages' groups joined, as one page if they fit, or split in two at half the tuples.
    fn combine(
        &self,
        right: &Self,
    ) -> PageCombined<Self> {
        let both = concat_groups(decode(&self.blob), decode(&right.blob));
        let one = encode(&both);
        if Self::fits(&one) {
            return PageCombined::One(one);
        }
        let total = one.len;
        let (l, r) = split_groups(both, total / 2);
        let right = encode(&r);
        PageCombined::Two(encode(&l), right.min(), right)
    }

    fn heap_bytes(&self) -> usize {
        self.blob.len()
    }
}
