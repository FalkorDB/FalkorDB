//! The numeric index: a CoW B⁺-tree of `(key, entity_id)` tuples, where `key` is the
//! [`encode_numeric`] image of the indexed value.
//!
//! One concrete type, no trait: index kinds are a closed set, dispatched through the
//! [`IndexColumn`](super::falkordb_index::IndexColumn) enum. MVCC is intrinsic: a query snapshots
//! the tree in `O(1)` (root-`Arc` clone) and writes are copy-on-write, so readers never see a torn
//! write. The tree's root lives on the graph version itself — see
//! [`FalkorDbIndex`](super::falkordb_index::FalkorDbIndex).

use std::ops::{Index, IndexMut};
use std::sync::Arc;

use super::data_structures::cow_btree::{CowBTree, RangeIter};
use super::encode::{KeyKind, encode_numeric, for_each_stored_key};
use crate::index::IndexQuery;
use crate::runtime::value::Value;

/// Tree fan-out. Fixed at the tuned default for now; a future step can make the
/// index generic over these if a workload wants a different page size.
const LEAF_MAX: usize = 256;
const BRANCH_MAX: usize = 256;
/// Full-width doc ids. The tree can store them narrower (fewer bytes per entry, so more entries
/// per page), but that caps the representable id and the index must hold any node or edge id the
/// graph can mint. Narrowing is a memory optimization to make deliberately, once there is a bound
/// to justify it — not a default to inherit.
const DOC_BYTES: usize = 8;

type Tree = CowBTree<LEAF_MAX, BRANCH_MAX, DOC_BYTES>;

/// Lazy iterator of matching entity ids, yielded in `(value, id)` order.
type TreeIter = RangeIter<LEAF_MAX, BRANCH_MAX, DOC_BYTES>;

/// Lazy iterator of matching entity ids.
///
/// `One` is a single cursor over a contiguous key range. `Many` chains several — the shape an
/// `IN [...]` union takes, one cursor per distinct member. An enum rather than a boxed trait
/// object because the scan op already boxes the result once; a second layer of dynamic dispatch
/// per id would be pure overhead.
///
/// The chain is *not* merged into id order. Order is deliberately unspecified here: Cypher
/// guarantees none without `ORDER BY`, and a merge would cost a comparison per id and force every
/// branch to stay live. Nothing downstream may assume sortedness.
pub enum DocIter {
    One(TreeIter),
    Many(UnionIter),
}

impl Iterator for DocIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match self {
            Self::One(it) => it.next(),
            Self::Many(it) => it.next(),
        }
    }
}

/// The keys a numeric predicate selects: what a column looks up to answer it, and what a scan
/// standing in for a column still building checks each entity's key against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeySet {
    /// None: a value or bound that does not encode, or bounds that cross.
    Empty,
    /// One key: an `Equal`.
    Point(u64),
    /// An inclusive range of keys: a `Range`.
    Range(u64, u64),
    /// Distinct keys, ascending: an `IN` list or a union of equalities.
    Keys(Vec<u64>),
}

impl KeySet {
    /// Keys are deduplicated, so `IN [1,1,2]` yields each entity once. That is sufficient for
    /// uniqueness overall: a doc appears under at most one key in the scalar tree, which is the
    /// only tree an `IN` list or a union reads (a list value's elements live in the array tree), so
    /// distinct keys cannot both hold the same doc.
    fn distinct(mut keys: Vec<u64>) -> Self {
        keys.sort_unstable();
        keys.dedup();
        Self::Keys(keys)
    }

    /// Whether `key` is one of these.
    #[must_use]
    pub fn contains(
        &self,
        key: u64,
    ) -> bool {
        match self {
            Self::Empty => false,
            Self::Point(k) => *k == key,
            Self::Range(lo, hi) => (*lo..=*hi).contains(&key),
            Self::Keys(keys) => keys.binary_search(&key).is_ok(),
        }
    }
}

/// The keys a numeric predicate selects, and which of a column's two trees they are keys of: the
/// scalar tree for `Equal`, `Range`, `IN` and unions, the array tree for `value IN n.prop`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub kind: KeyKind,
    pub keys: KeySet,
}

impl Selection {
    /// Keys of the scalar tree.
    #[must_use]
    pub const fn scalar(keys: KeySet) -> Self {
        Self {
            kind: KeyKind::Scalar,
            keys,
        }
    }

    /// Whether a column answering this selection returns an entity whose value is `value`:
    /// whether `value` stores a key of this kind among these keys. A scalar stores one key of
    /// [`KeyKind::Scalar`] and a list one of [`KeyKind::Array`] per numeric element, exactly as
    /// the column indexes them ([`for_each_stored_key`]), so `n.v = 1` admits the scalar `1` and
    /// not `[1, 2]`, and `1 IN n.v` admits `[1, 2]` and not the scalar `1`.
    #[must_use]
    pub fn admits(
        &self,
        value: &Value,
    ) -> bool {
        let mut hit = false;
        for_each_stored_key(value, |kind, key| {
            hit |= kind == self.kind && self.keys.contains(key);
        });
        hit
    }
}

/// The cursor chain behind a union: the keys still to visit, and a cursor over the current one.
///
/// **One cursor exists at a time, and none until the first `next()`.** `RangeIter::new` performs a
/// full root-to-leaf descent in its constructor, so building a cursor per member up front made
/// `IN $list` with 100k members do 100k descents — and hold 100k live snapshots — before yielding
/// a single row. Under a `LIMIT`, almost all of that work is thrown away.
///
/// The tree is **owned**, not borrowed. That is load-bearing rather than a lifetime convenience:
/// the eager version happened to share one root because every `point()` call sat inside a single
/// `&self` borrow. Deferring those calls past the borrow means the snapshot has to be pinned here,
/// or a member visited after a concurrent write would descend into a newer root and the union
/// would be a torn read. The clone is an `O(1)` root-`Arc` bump.
pub struct UnionIter {
    tree: Tree,
    keys: std::vec::IntoIter<u64>,
    current: Option<TreeIter>,
}

impl Iterator for UnionIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        loop {
            if let Some(doc) = self.current.as_mut().and_then(Iterator::next) {
                return Some(doc);
            }
            // Current member exhausted (or not started): descend for the next one. Keys are
            // distinct, so this advances — it cannot spin.
            self.current = Some(self.tree.point(self.keys.next()?));
        }
    }
}

/// Encoded `(key, doc)` tuples for one column, kept apart by [`KeyKind`]: what everything that
/// moves a column's tuples in encoded form carries (a unit of work's collapsed adds and removes,
/// an indexer batch).
///
/// Indexed by the kind (`tuples[KeyKind::Array]`), so a tuple always goes back into the tree it
/// came from. A flat list would lose which tree each tuple belongs in, and a kind stored on every
/// tuple would cost 8 more bytes per tuple for what the position already says.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct EncodedTuples {
    by_kind: [Vec<(u64, u64)>; 2],
}

impl Index<KeyKind> for EncodedTuples {
    type Output = Vec<(u64, u64)>;

    fn index(
        &self,
        kind: KeyKind,
    ) -> &Self::Output {
        &self.by_kind[kind as usize]
    }
}

impl IndexMut<KeyKind> for EncodedTuples {
    fn index_mut(
        &mut self,
        kind: KeyKind,
    ) -> &mut Self::Output {
        &mut self.by_kind[kind as usize]
    }
}

impl EncodedTuples {
    /// Tuples of scalar values only: the shape a column holding no list values produces.
    #[must_use]
    pub fn scalars(scalar: Vec<(u64, u64)>) -> Self {
        Self {
            by_kind: [scalar, Vec::new()],
        }
    }

    /// Whether there are no tuples of either kind.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_kind.iter().all(Vec::is_empty)
    }

    /// How many tuples there are, of both kinds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_kind.iter().map(Vec::len).sum()
    }

    /// Empty the tuples, keeping their capacity for reuse.
    pub fn clear(&mut self) {
        self.by_kind.iter_mut().for_each(Vec::clear);
    }

    /// Keep only the tuples, of either kind, that `keep` accepts.
    pub fn retain(
        &mut self,
        mut keep: impl FnMut(&(u64, u64)) -> bool,
    ) {
        for tuples in &mut self.by_kind {
            tuples.retain(&mut keep);
        }
    }

    /// Append the numeric tuples `(value, id)` contributes, each with the tuples of its kind.
    /// Unsorted, and a list repeating a number appends its key more than once: see
    /// [`sort_dedup`](Self::sort_dedup).
    pub fn push_numeric(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        for_each_stored_key(value, |kind, key| self[kind].push((key, id)));
    }

    /// Sort each kind's tuples ascending and drop duplicates: the **sorted and unique** shape a
    /// tree's bulk build requires.
    pub fn sort_dedup(&mut self) {
        for tuples in &mut self.by_kind {
            tuples.sort_unstable();
            tuples.dedup();
        }
    }
}

/// A numeric property index over one `(label, attribute)`: entity ids keyed by
/// the order-preserving encoding of their value, so `n.x <predicate> v` is a
/// tree range scan. Non-numeric and `NaN` values are not indexed (parity with
/// the RediSearch NUMERIC field it replaces).
///
/// `Clone` is `O(1)` — the underlying `CowBTree` clone is a root-`Arc` bump — so
/// a graph version can fork its index snapshot cheaply (see
/// [`FalkorDbIndex`](super::falkordb_index::FalkorDbIndex)).
#[derive(Clone, Default)]
pub struct NumericIndex {
    /// Scalar values: at most one tuple per indexed entity.
    tree: Tree,
    /// Elements of **list**-valued properties: one tuple per distinct numeric element.
    ///
    /// A separate tree, not a convenience: see [`KeyKind`]. Sharing one tree would make a node
    /// with `v = [1, 2]` carry the key for `1`, so `WHERE n.v = 1` would match it.
    ///
    /// Read only by `ArrayContains` (`value IN n.prop`), which is a point lookup, so a doc is
    /// reached through exactly one key per query and cannot be yielded twice.
    array_tree: Tree,
}

impl NumericIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Index `id` under `value`: under its key if it is a scalar, under each numeric element's
    /// key if it is a list. A no-op for non-numeric / `NaN` values and idempotent for an
    /// already-present `(value, id)`.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        for_each_stored_key(value, |kind, k| {
            // The bool reports whether the tuple was newly inserted — for callers keeping an
            // exact live count. This index derives its counts from the tree, so it is discarded
            // deliberately rather than ignored.
            let _newly_inserted = self.tree_mut(kind).insert(k, id);
        });
    }

    /// Remove `id` from under `value`. A no-op if it was never indexed.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        for_each_stored_key(value, |kind, k| {
            let _was_present = self.tree_mut(kind).remove(k, id);
        });
    }

    /// Add a batch of `(value, id)` entries, consuming any iterator (the runtime's columnar batch).
    /// Non-numeric / `NaN` are dropped, and the encoded tuples are sorted here because the tree's
    /// `insert_batch` needs a sorted slice. Cheaper than repeated [`add`](Self::add).
    ///
    /// `scratch` is the caller's encode buffer and its contents are replaced. Taking it rather than
    /// allocating internally lets a commit that touches many columns allocate once and reuse the
    /// capacity, instead of a fresh buffer per column.
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        Self::encode_pairs_into(entries, scratch);
        for kind in KeyKind::ALL {
            self.tree_mut(kind).insert_batch(&scratch[kind]);
        }
    }

    /// Remove a batch of `(value, id)` entries, consuming any iterator — the write path's
    /// mass-delete / mass-update column. Non-numeric / `NaN` dropped, encoded tuples sorted here.
    ///
    /// `scratch` is the caller's encode buffer — see [`add_batch`](Self::add_batch).
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        Self::encode_pairs_into(entries, scratch);
        for kind in KeyKind::ALL {
            self.tree_mut(kind).remove_batch(&scratch[kind]);
        }
    }

    /// Encode `(value, id)` entries to `(key, id)` tree tuples in `out` (cleared first), each
    /// with the tuples of its [`KeyKind`], sorted and unique. Non-numeric / `NaN` values are
    /// dropped.
    ///
    /// Takes the buffer rather than returning one so a caller applying many columns in a row
    /// allocates once and reuses the capacity, instead of one buffer per column per commit.
    fn encode_pairs_into(
        entries: impl IntoIterator<Item = (Value, u64)>,
        out: &mut EncodedTuples,
    ) {
        out.clear();
        let entries = entries.into_iter();
        out[KeyKind::Scalar].reserve(entries.size_hint().0);
        for (v, id) in entries {
            out.push_numeric(&v, id);
        }
        out.sort_dedup();
    }

    /// Add already-encoded tuples: the adds a unit of work's staged writes collapsed to, or an
    /// indexer batch. Tuples are never re-encoded: re-encoding a decoded key would be a second trip
    /// through a many-to-one map. Into an empty tree the tuples are bulk-built, bottom-up from
    /// the sorted tuples, rather than merged in.
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        for kind in KeyKind::ALL {
            let tree = self.tree_mut(kind);
            if tree.is_empty() {
                *tree = Tree::from_sorted(&tuples[kind]);
            } else {
                tree.insert_batch(&tuples[kind]);
            }
        }
    }

    /// Remove already-encoded tuples: the removes a unit of work's staged writes collapsed to.
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        for kind in KeyKind::ALL {
            self.tree_mut(kind).remove_batch(&tuples[kind]);
        }
    }

    /// Whether the index holds no tuples of either kind.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty() && self.array_tree.is_empty()
    }

    /// The tree holding keys of `kind`.
    fn tree_ref(
        &self,
        kind: KeyKind,
    ) -> &Tree {
        match kind {
            KeyKind::Scalar => &self.tree,
            KeyKind::Array => &self.array_tree,
        }
    }

    /// The tree holding keys of `kind`, to change.
    fn tree_mut(
        &mut self,
        kind: KeyKind,
    ) -> &mut Tree {
        match kind {
            KeyKind::Scalar => &mut self.tree,
            KeyKind::Array => &mut self.array_tree,
        }
    }

    /// Heap bytes the two trees hold ([`CowBTree::heap_bytes`]): every page this version reaches,
    /// whether or not another version shares it. Walks every page.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.tree.heap_bytes() + self.array_tree.heap_bytes()
    }

    /// Entity ids whose value equals `value`. Empty for a non-numeric value.
    #[must_use]
    pub fn point(
        &self,
        value: &Value,
    ) -> DocIter {
        self.lookup(Selection::scalar(Self::point_keys(value)))
    }

    /// Entity ids stored under the encoded key `key`.
    #[must_use]
    pub fn point_key(
        &self,
        key: u64,
    ) -> DocIter {
        DocIter::One(self.tree.point(key))
    }

    /// Entity ids whose value lies in the (optionally half-open) numeric range.
    /// A `None` bound is unbounded on that side; a non-numeric bound yields no
    /// matches. Results are in `(value, id)` order.
    #[must_use]
    pub fn range(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> DocIter {
        self.lookup(Selection::scalar(Self::range_keys(
            min,
            max,
            include_min,
            include_max,
        )))
    }

    /// Entity ids whose **list**-valued property contains `value`: `value IN n.prop`.
    ///
    /// A point lookup into the array tree, so each matching doc is reached through one key and
    /// appears once. The scalar tree is deliberately not consulted: `1 IN n.v` must not match a
    /// node whose `v` is the scalar `1`.
    ///
    /// May over-return relative to Cypher: the column holds no type tag, so `1 IN n.v` also
    /// matches `v = [true]` (the encoder maps `true` to `1.0`). `utilize_index` keeps the original
    /// filter on this predicate so the runtime rechecks it, which is why the array tree may be
    /// approximate where the scalar tree may not.
    #[must_use]
    pub fn array_contains(
        &self,
        value: &Value,
    ) -> DocIter {
        self.lookup(Selection {
            kind: KeyKind::Array,
            keys: Self::point_keys(value),
        })
    }

    /// Dispatch the numeric predicates. `Equal` → [`point`](Self::point), `Range` →
    /// [`range`](Self::range), an `IN` list → [`in_list`](Self::in_list), an `OR` of
    /// equalities and `IN` lists → [`union`](Self::union), and `value IN n.prop` →
    /// [`array_contains`](Self::array_contains). `And` and non-numeric variants return `None` —
    /// the query router composes or rejects those.
    #[must_use]
    pub fn query(
        &self,
        q: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        Self::keys(q).map(|keys| self.lookup(keys))
    }

    /// The keys `q` selects, or `None` where [`query`](Self::query) declines it. A column answers
    /// from these, and a scan standing in for a column still building checks each entity against
    /// them, so the scan keeps exactly what the column will answer.
    #[must_use]
    pub fn keys(q: &IndexQuery<Value>) -> Option<Selection> {
        let keys = match q {
            IndexQuery::Equal { value, .. } => Self::point_keys(value),
            IndexQuery::Range {
                min,
                max,
                include_min,
                include_max,
                ..
            } => Self::range_keys(min.as_ref(), max.as_ref(), *include_min, *include_max),
            IndexQuery::InList {
                list: Value::List(items),
                ..
            } => Self::in_list(items)?,
            IndexQuery::Or(children) => Self::union(children)?,
            IndexQuery::ArrayContains { value, .. } => {
                return Some(Selection {
                    kind: KeyKind::Array,
                    keys: Self::point_keys(value),
                });
            }
            _ => return None,
        };
        Some(Selection::scalar(keys))
    }

    /// The entity ids under `selection`'s keys, in the tree of its kind.
    #[must_use]
    pub fn lookup(
        &self,
        selection: Selection,
    ) -> DocIter {
        let tree = self.tree_ref(selection.kind);
        match selection.keys {
            KeySet::Empty => self.empty(),
            KeySet::Point(k) => DocIter::One(tree.point(k)),
            KeySet::Range(lo, hi) => DocIter::One(tree.range(lo, hi)),
            // No cursor is built here — see [`UnionIter`]. The tree snapshot is taken now, so
            // every member is answered from the same version however late its cursor is created.
            KeySet::Keys(keys) => DocIter::Many(UnionIter {
                tree: tree.clone(),
                keys: keys.into_iter(),
                current: None,
            }),
        }
    }

    fn point_keys(value: &Value) -> KeySet {
        encode_numeric(value).map_or(KeySet::Empty, KeySet::Point)
    }

    fn range_keys(
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> KeySet {
        // Map value bounds to inclusive key bounds. Exclusive bounds step one key
        // inward: encodings of distinct f64 are adjacent u64, so `k+1` / `k-1` is
        // exactly the next / previous representable value. The `checked_*` guards
        // catch the ±∞ edges (`x > +inf`, `x < -inf`) as empty.
        let lo = match min {
            None => 0,
            Some(v) => {
                let Some(k) = encode_numeric(v) else {
                    return KeySet::Empty;
                };
                if include_min {
                    k
                } else {
                    match k.checked_add(1) {
                        Some(k1) => k1,
                        None => return KeySet::Empty,
                    }
                }
            }
        };
        let hi = match max {
            None => u64::MAX,
            Some(v) => {
                let Some(k) = encode_numeric(v) else {
                    return KeySet::Empty;
                };
                if include_max {
                    k
                } else {
                    match k.checked_sub(1) {
                        Some(k1) => k1,
                        None => return KeySet::Empty,
                    }
                }
            }
        };
        if lo > hi {
            return KeySet::Empty;
        }
        KeySet::Range(lo, hi)
    }

    /// `n.a IN [...]`. The list names one attribute by construction, so only its values are
    /// checked.
    ///
    /// Returns `None` unless **every** value encodes to a numeric key. Serving only the numeric
    /// subset would silently drop rows: a value the numeric column cannot represent (a string,
    /// say) may still match entities held by another kind, and answering without them is a wrong
    /// answer rather than a slower one. Declining hands the whole query back to the caller intact.
    ///
    /// An empty list is declined too, rather than answered as "no rows": the scan op drops the
    /// members it cannot index before the list gets here, so `IN [date(..)]` arrives empty, and
    /// "no rows" would miss every entity holding that date.
    fn in_list(items: &[Value]) -> Option<KeySet> {
        if items.is_empty() {
            return None;
        }
        let keys = items
            .iter()
            .map(encode_numeric)
            .collect::<Option<Vec<u64>>>()?;
        Some(KeySet::distinct(keys))
    }

    /// An `OR` of `Equal`s and `IN` lists — `a = 1 OR a IN [2, 3]`.
    ///
    /// Unlike an `IN` list, an `OR` can name several attributes, and this index is one column:
    /// answering `a = 1 OR b = 2` from it would look up both `1` and `2` in `a` and return
    /// `{a=1} ∪ {a=2}` — wrong rows, not missing ones. So every member must target the same
    /// attribute, and every value must encode, as in [`in_list`](Self::in_list); otherwise `None`.
    /// The exception is `a = null`, which holds for nothing, so it adds no key.
    fn union(children: &[IndexQuery<Value>]) -> Option<KeySet> {
        if children.is_empty() {
            return None;
        }
        let mut attr: Option<&Arc<String>> = None;
        let mut keys = Vec::with_capacity(children.len());
        for child in children {
            let (key, values) = match child {
                IndexQuery::Equal { key, value } => (key, std::slice::from_ref(value)),
                IndexQuery::InList {
                    key,
                    list: Value::List(items),
                } if !items.is_empty() => (key, items.as_slice()),
                // A range, an empty list, a nested combinator — not this shape.
                _ => return None,
            };
            if *attr.get_or_insert(key) != key {
                return None;
            }
            for value in values.iter().filter(|v| !matches!(v, Value::Null)) {
                keys.push(encode_numeric(value)?);
            }
        }
        Some(KeySet::distinct(keys))
    }

    /// An iterator over no entries (`lo > hi` yields nothing).
    fn empty(&self) -> DocIter {
        DocIter::One(self.tree.range(1, 0))
    }

    /// Bulk-build from `(value, id)` pairs in any order. Non-numeric / `NaN`
    /// values are dropped. Far cheaper than repeated [`add`](Self::add) for
    /// initial population — one sort + bottom-up page build, no per-item
    /// traversal. The tree's tests compare it with incremental adds.
    #[cfg(test)]
    #[must_use]
    pub fn from_entries<'a>(entries: impl IntoIterator<Item = (&'a Value, u64)>) -> Self {
        let mut tuples = EncodedTuples::default();
        for (v, id) in entries {
            tuples.push_numeric(v, id);
        }
        tuples.sort_dedup();
        Self {
            tree: Tree::from_sorted(&tuples[KeyKind::Scalar]),
            array_tree: Tree::from_sorted(&tuples[KeyKind::Array]),
        }
    }

    /// Every `(kind, key, doc)` tuple the index holds, in order: the scalar tree's, then the array
    /// tree's. For tests that compare a column with what it must hold.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(KeyKind, u64, u64)> {
        KeyKind::ALL
            .into_iter()
            .flat_map(|kind| {
                self.tree_ref(kind)
                    .range_tuples(0, u64::MAX)
                    .map(move |(key, doc)| (kind, key, doc))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn ids(it: DocIter) -> Vec<u64> {
        it.collect()
    }

    /// Values: 10→{1,2}, 20→{3}, -5→{4}, 3.5→{5}.
    fn sample() -> NumericIndex {
        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(10), 1);
        idx.add(&Value::Int(10), 2);
        idx.add(&Value::Int(20), 3);
        idx.add(&Value::Int(-5), 4);
        idx.add(&Value::Float(3.5), 5);
        idx
    }

    #[test]
    fn point_matches_all_ids_for_a_value() {
        let idx = sample();
        assert_eq!(ids(idx.point(&Value::Int(10))), vec![1, 2]); // doc-ascending
        assert_eq!(ids(idx.point(&Value::Int(20))), vec![3]);
        assert_eq!(ids(idx.point(&Value::Float(10.0))), vec![1, 2]); // 10 == 10.0
        assert!(ids(idx.point(&Value::Int(99))).is_empty());
    }

    #[test]
    fn range_inclusive_exclusive_unbounded() {
        let idx = sample();
        let i = |x| Value::Int(x);
        // inclusive [-5, 20] → ordered by (value, id): -5,3.5,10,10,20
        assert_eq!(
            ids(idx.range(Some(&i(-5)), Some(&i(20)), true, true)),
            vec![4, 5, 1, 2, 3]
        );
        // (10, 20] — exclusive min drops value 10
        assert_eq!(
            ids(idx.range(Some(&i(10)), Some(&i(20)), false, true)),
            vec![3]
        );
        // [10, 20) — exclusive max drops value 20
        assert_eq!(
            ids(idx.range(Some(&i(10)), Some(&i(20)), true, false)),
            vec![1, 2]
        );
        // unbounded below, <= 3.5
        assert_eq!(
            ids(idx.range(None, Some(&Value::Float(3.5)), true, true)),
            vec![4, 5]
        );
        // >= 10, unbounded above
        assert_eq!(
            ids(idx.range(Some(&i(10)), None, true, true)),
            vec![1, 2, 3]
        );
        // fully unbounded → everything, in order
        assert_eq!(ids(idx.range(None, None, true, true)), vec![4, 5, 1, 2, 3]);
    }

    #[test]
    fn empty_and_degenerate_ranges() {
        let idx = sample();
        let i = |x| Value::Int(x);
        // min > max
        assert!(ids(idx.range(Some(&i(20)), Some(&i(10)), true, true)).is_empty());
        // (10, 10) — exclusive both sides of one value
        assert!(ids(idx.range(Some(&i(10)), Some(&i(10)), false, false)).is_empty());
    }

    #[test]
    fn remove_deletes_one_tuple() {
        let mut idx = sample();
        idx.remove(&Value::Int(10), 1);
        assert_eq!(ids(idx.point(&Value::Int(10))), vec![2]);
        idx.remove(&Value::Int(10), 2);
        assert!(ids(idx.point(&Value::Int(10))).is_empty());
        // the rest are untouched
        assert_eq!(ids(idx.point(&Value::Int(20))), vec![3]);
    }

    /// A temporal value must not reach the tree. Stored as its raw number it lands among the
    /// numbers, so a numeric range scan walks straight over it and returns the entity — measured
    /// on a server as `WHERE n.v > 0` returning date-valued nodes. The RediSearch path skips
    /// temporals for exactly this reason (`index/mod.rs`), and no index query is ever built for a
    /// temporal value, so an entry here could only ever surface as a wrong row.
    ///
    /// Covers all four variants through both write paths — the bulk build and the incremental
    /// add — with raw numbers above zero, which is where they collided.
    #[test]
    fn temporal_values_are_not_returned_by_a_numeric_range() {
        let temporals = [
            Value::Date(18_262),
            Value::Datetime(1_577_836_800),
            Value::Time(3_600),
            Value::Duration(86_400),
        ];
        let int_id = 99;

        let entries: Vec<(&Value, u64)> = temporals
            .iter()
            .zip(1u64..)
            .chain(std::iter::once((&Value::Int(5), int_id)))
            .collect();
        let bulk = NumericIndex::from_entries(entries.iter().map(|&(v, id)| (v, id)));

        let mut incremental = NumericIndex::new();
        for &(v, id) in &entries {
            incremental.add(v, id);
        }

        for (how, idx) in [("bulk build", &bulk), ("incremental add", &incremental)] {
            assert_eq!(
                ids(idx.range(Some(&Value::Int(0)), None, false, true)),
                vec![int_id],
                "{how}: `> 0` must return only the integer, not the temporals"
            );
        }
    }

    #[test]
    fn non_numeric_and_nan_are_not_indexed() {
        let mut idx = NumericIndex::new();
        idx.add(&Value::String(Arc::new("hi".to_string())), 1); // skipped
        idx.add(&Value::Float(f64::NAN), 2); // skipped
        idx.add(&Value::Int(7), 3);
        assert!(!idx.is_empty());
        assert_eq!(ids(idx.point(&Value::Int(7))), vec![3]);
        // a non-numeric bound yields no matches
        assert!(
            ids(idx.range(
                Some(&Value::String(Arc::new("a".to_string()))),
                None,
                true,
                true
            ))
            .is_empty()
        );
    }

    #[test]
    fn bulk_build_matches_incremental() {
        let entries = [
            (Value::Int(10), 1),
            (Value::Int(10), 2),
            (Value::Int(20), 3),
            (Value::Int(-5), 4),
            (Value::Float(3.5), 5),
        ];
        let bulk = NumericIndex::from_entries(entries.iter().map(|(v, id)| (v, *id)));
        let inc = sample();
        let all = |ix: &NumericIndex| ids(ix.range(None, None, true, true));
        assert_eq!(all(&bulk), all(&inc));
    }

    #[test]
    fn duplicate_insert_is_idempotent() {
        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(5), 1);
        idx.add(&Value::Int(5), 1);
        assert_eq!(ids(idx.point(&Value::Int(5))), vec![1]);
    }

    #[test]
    fn query_dispatches_leaf_predicates() {
        let idx = sample();
        let key = Arc::new("x".to_string());
        let eq = IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(20),
        };
        assert_eq!(ids(idx.query(&eq).unwrap()), vec![3]);
        let rg = IndexQuery::Range {
            key: key.clone(),
            min: Some(Value::Int(10)),
            max: None,
            include_min: true,
            include_max: true,
        };
        assert_eq!(ids(idx.query(&rg).unwrap()), vec![1, 2, 3]);
        // composite predicates are the router's job, not a single index's
        assert!(idx.query(&IndexQuery::And(vec![eq])).is_none());
    }

    #[test]
    fn batch_ops_match_single_ops() {
        let entries = vec![
            (Value::Int(30), 1),
            (Value::Int(10), 2),
            (Value::Int(20), 3),
            (Value::Float(15.5), 4),
            (Value::Int(10), 5),
            (Value::String(Arc::new("skip".to_string())), 6), // non-numeric → dropped
        ];
        let all = |ix: &NumericIndex| ids(ix.range(None, None, true, true));

        // add_batch == repeated add
        let mut batched = NumericIndex::new();
        batched.add_batch(entries.iter().cloned(), &mut EncodedTuples::default());
        let mut singly = NumericIndex::new();
        for (v, id) in &entries {
            singly.add(v, *id);
        }
        assert_eq!(all(&batched), all(&singly));

        // remove_batch == repeated remove (incl. an absent entry, which is a no-op)
        let removes = vec![
            (Value::Int(10), 2),
            (Value::Int(30), 1),
            (Value::Int(99), 7),
        ];
        batched.remove_batch(removes.iter().cloned(), &mut EncodedTuples::default());
        for (v, id) in &removes {
            singly.remove(v, *id);
        }
        assert_eq!(all(&batched), all(&singly));
    }

    /// `n.a IN [...]` as the scan op hands it over: one attribute and a list of values.
    fn in_list(
        attr: &Arc<String>,
        values: impl IntoIterator<Item = Value>,
    ) -> IndexQuery<Value> {
        IndexQuery::InList {
            key: attr.clone(),
            list: Value::List(Arc::new(values.into_iter().collect())),
        }
    }

    /// A union's ids, sorted: its order is unspecified.
    fn sorted(it: DocIter) -> Vec<u64> {
        let mut got = ids(it);
        got.sort_unstable();
        got
    }

    fn ten_twenty_thirty() -> NumericIndex {
        NumericIndex::from_entries(
            [
                (&Value::Int(10), 0u64),
                (&Value::Int(20), 1),
                (&Value::Int(30), 2),
            ]
            .into_iter(),
        )
    }

    /// An `IN` list is served natively only when every value is numeric — a mixed list must be
    /// declined whole, not answered from the numeric values alone, or rows another kind holds would
    /// silently vanish. An empty list is declined too: the scan op empties a list by dropping the
    /// members it cannot index (`IN [date(..)]`), so "no rows" would be a wrong answer.
    #[test]
    fn in_list_serves_all_numeric_and_declines_mixed_or_empty() {
        let attr = Arc::new("v".to_string());
        let idx = ten_twenty_thirty();

        assert_eq!(
            sorted(
                idx.query(&in_list(&attr, [Value::Int(10), Value::Int(30)]))
                    .expect("an all-numeric list is servable")
            ),
            vec![0, 2]
        );
        // Duplicate values — including an Int and a Float that share a key — yield an id once.
        assert_eq!(
            sorted(
                idx.query(&in_list(
                    &attr,
                    [Value::Int(20), Value::Int(20), Value::Float(20.0)]
                ))
                .expect("servable")
            ),
            vec![1],
            "duplicate values must dedup"
        );
        assert!(
            idx.query(&in_list(
                &attr,
                [Value::Int(10), Value::String(Arc::new("a".to_string()))]
            ))
            .is_none(),
            "a mixed list must fall back rather than drop the non-numeric value"
        );
        assert!(
            idx.query(&in_list(&attr, Vec::new())).is_none(),
            "an empty list must fall back, not answer \"no rows\""
        );
    }

    /// An `OR` of equalities and `IN` lists on one attribute — `a = 10 OR a IN [20, 30]`, or
    /// `a IN [..] OR a IN [..]` — is one union. Any member that is not that shape declines the
    /// whole `OR`: a range, an empty list, a list holding a non-numeric value.
    #[test]
    fn or_serves_equalities_and_lists_on_one_attribute() {
        let attr = Arc::new("v".to_string());
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };
        let idx = ten_twenty_thirty();

        assert_eq!(
            sorted(
                idx.query(&IndexQuery::Or(vec![
                    eq(10),
                    in_list(&attr, [Value::Int(20), Value::Int(30)]),
                ]))
                .expect("an equality and a list on one attribute are servable")
            ),
            vec![0, 1, 2]
        );
        assert_eq!(
            sorted(
                idx.query(&IndexQuery::Or(vec![
                    in_list(&attr, [Value::Int(10)]),
                    in_list(&attr, [Value::Int(10), Value::Int(30)]),
                ]))
                .expect("two lists on one attribute are servable")
            ),
            vec![0, 2],
            "a value in both lists yields its id once"
        );

        assert!(
            idx.query(&IndexQuery::Or(vec![
                eq(10),
                in_list(
                    &attr,
                    [Value::Int(20), Value::String(Arc::new("a".to_string()))]
                ),
            ]))
            .is_none(),
            "a non-numeric value in any member declines the whole OR"
        );
        assert!(
            idx.query(&IndexQuery::Or(vec![eq(10), in_list(&attr, Vec::new())]))
                .is_none(),
            "an empty list member declines, as an empty list does on its own"
        );
        assert!(
            idx.query(&IndexQuery::Or(vec![
                eq(10),
                IndexQuery::Range {
                    key: attr.clone(),
                    min: Some(Value::Int(0)),
                    max: None,
                    include_min: true,
                    include_max: true,
                },
            ]))
            .is_none(),
            "a range member is not this shape"
        );
        assert!(
            idx.query(&IndexQuery::Or(vec![])).is_none(),
            "an empty OR is not match-all"
        );
    }

    /// One `NumericIndex` is one column, so an `OR` spanning two attributes cannot be answered
    /// here: looking both values up in this column returns `{a=1} ∪ {a=2}` — wrong rows, not
    /// missing ones. An `IN` list cannot span attributes, but an `OR` can, so the check lives in
    /// `union` itself and this test reaches `query` directly.
    #[test]
    fn or_declines_members_targeting_different_attributes() {
        let idx =
            NumericIndex::from_entries([(&Value::Int(1), 10u64), (&Value::Int(2), 20)].into_iter());
        let (a, b) = (Arc::new("a".to_string()), Arc::new("b".to_string()));
        let eq = |attr: &Arc<String>, v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };

        assert!(
            idx.query(&IndexQuery::Or(vec![eq(&a, 1), eq(&b, 2)]))
                .is_none(),
            "a cross-attribute OR must be declined, not answered from one column"
        );
        assert!(
            idx.query(&IndexQuery::Or(vec![
                in_list(&a, [Value::Int(1)]),
                in_list(&b, [Value::Int(2)]),
            ]))
            .is_none(),
            "`a IN [..] OR b IN [..]` must be declined the same way"
        );
        assert_eq!(
            sorted(
                idx.query(&IndexQuery::Or(vec![eq(&a, 1), eq(&a, 2)]))
                    .expect("one attribute is servable")
            ),
            vec![10, 20]
        );
    }

    /// A union must not descend for a member until it is reached. Building every cursor up front
    /// makes `IN $list` pay one root-to-leaf descent per member — 100k of them, plus 100k live
    /// snapshots — before the first row, which a `LIMIT` then throws away.
    #[test]
    fn union_builds_no_cursor_before_the_first_row() {
        let attr = Arc::new("v".to_string());
        let values: Vec<(Value, u64)> = (0..100u64).map(|i| (Value::Int(i as i64), i)).collect();
        let idx = NumericIndex::from_entries(values.iter().map(|(v, id)| (v, *id)));

        let q = in_list(&attr, (0..100).map(Value::Int));
        let DocIter::Many(mut u) = idx.query(&q).expect("an all-numeric list is servable") else {
            panic!("an IN list must produce DocIter::Many");
        };
        assert!(
            u.current.is_none(),
            "no cursor may exist before the first next()"
        );
        assert_eq!(u.keys.len(), 100, "and every member is still pending");

        assert_eq!(u.next(), Some(0));
        assert!(u.current.is_some(), "the first row descends for its member");
        assert_eq!(u.keys.len(), 99, "exactly one member consumed");
    }

    /// Deferring the descents means later members are visited after the caller has moved on, so
    /// the iterator has to pin the snapshot itself. Otherwise a write landing mid-scan would be
    /// half-visible: invisible to members already started, visible to the rest.
    #[test]
    fn union_pins_one_snapshot_across_lazily_built_cursors() {
        let attr = Arc::new("v".to_string());
        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(1), 1);
        idx.add(&Value::Int(2), 2);

        let mut it = idx
            .query(&in_list(&attr, [Value::Int(1), Value::Int(2)]))
            .expect("servable");
        assert_eq!(it.next(), Some(1), "first member started");

        // A write between the first member and the second — whose cursor does not exist yet.
        idx.add(&Value::Int(2), 99);

        let rest: Vec<u64> = it.collect();
        assert_eq!(
            rest,
            vec![2],
            "the second member must be answered from the snapshot the union started on"
        );
    }

    fn list(xs: impl IntoIterator<Item = Value>) -> Value {
        Value::List(Arc::new(xs.into_iter().collect()))
    }

    fn ints(xs: &[i64]) -> Value {
        list(xs.iter().map(|&x| Value::Int(x)))
    }

    fn contains(
        attr: &Arc<String>,
        v: i64,
    ) -> IndexQuery<Value> {
        IndexQuery::ArrayContains {
            key: attr.clone(),
            value: Value::Int(v),
        }
    }

    /// The reason the array elements live in their own tree.
    ///
    /// A node whose `v` is the list `[1, 2]` must NOT be returned by `WHERE n.v = 1` — that is
    /// false in Cypher — and a node whose `v` is the scalar `1` must NOT be returned by
    /// `WHERE 1 IN n.v`. One shared tree cannot tell those apart, and `Equal` carries no
    /// post-filter to clean up after it (its plan is a bare `Node By Index Scan`), so the wrong
    /// row would reach the caller.
    #[test]
    fn a_scalar_and_a_list_element_do_not_share_a_tree() {
        let attr = Arc::new("v".to_string());
        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(1), 10); // scalar 1
        idx.add(&ints(&[1, 2]), 20); // list [1, 2]

        let eq = IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(1),
        };
        assert_eq!(
            ids(idx.query(&eq).expect("servable")),
            vec![10],
            "`n.v = 1` must match the scalar only, never the list that contains 1"
        );
        assert_eq!(
            ids(idx.query(&contains(&attr, 1)).expect("servable")),
            vec![20],
            "`1 IN n.v` must match the list only, never the scalar 1"
        );

        // A range or an IN list over the scalar tree must not see list elements either.
        let rng = IndexQuery::Range {
            key: attr.clone(),
            min: Some(Value::Int(0)),
            max: Some(Value::Int(5)),
            include_min: true,
            include_max: true,
        };
        assert_eq!(ids(idx.query(&rng).expect("servable")), vec![10]);
        assert_eq!(
            sorted(
                idx.query(&in_list(&attr, [Value::Int(1), Value::Int(2)]))
                    .expect("servable")
            ),
            vec![10]
        );
    }

    /// Every numeric element is reachable, and a repeated element contributes one tuple: the
    /// tree stores `(key, doc)` as a set, so a removal must not depend on how many times the
    /// value happened to list the same number.
    #[test]
    fn every_element_is_indexed_once() {
        let attr = Arc::new("v".to_string());
        let mut idx = NumericIndex::new();
        idx.add(&ints(&[1, 1, 2]), 7);
        assert_eq!(
            ids(idx.query(&contains(&attr, 1)).expect("servable")),
            vec![7]
        );
        assert_eq!(
            ids(idx.query(&contains(&attr, 2)).expect("servable")),
            vec![7]
        );
        assert!(ids(idx.query(&contains(&attr, 3)).expect("servable")).is_empty());
        assert_eq!(idx.array_tree.len(), 2);

        // Removing the same value takes every element with it, duplicates included.
        idx.remove(&ints(&[1, 1, 2]), 7);
        assert!(idx.is_empty(), "both trees must be empty again");

        // The batch paths dedup too: one tuple per distinct element, removable in one go.
        let mut scratch = EncodedTuples::default();
        idx.add_batch([(ints(&[3, 3, 4]), 8)], &mut scratch);
        assert_eq!(idx.array_tree.len(), 2);
        idx.remove_batch([(ints(&[3, 3, 4]), 8)], &mut scratch);
        assert!(idx.is_empty());
    }

    /// Elements the encoder rejects are skipped one by one rather than rejecting the whole list:
    /// strings belong to a text kind, and temporals are not indexed at all, exactly as for a
    /// scalar.
    #[test]
    fn a_list_indexes_only_its_numeric_elements() {
        let attr = Arc::new("v".to_string());
        let mixed = list([
            Value::Int(4),
            Value::String(Arc::new("x".to_string())),
            Value::Date(0),
            Value::Float(f64::NAN),
            ints(&[5]),
        ]);
        let mut idx = NumericIndex::new();
        idx.add(&mixed, 3);
        assert_eq!(
            ids(idx.query(&contains(&attr, 4)).expect("servable")),
            vec![3]
        );
        assert!(
            ids(idx.query(&contains(&attr, 0)).expect("servable")).is_empty(),
            "a date element's raw number is not a key"
        );
        assert!(
            ids(idx.query(&contains(&attr, 5)).expect("servable")).is_empty(),
            "a nested list is not looked inside"
        );
        assert_eq!(idx.array_tree.len(), 1);
        assert!(idx.tree.is_empty());
    }

    /// An indexer batch carries both kinds, and each goes into its own tree: bulk-built into an
    /// empty one, merged into one that already holds tuples. A batch that dropped its array half
    /// would build a column that answers `IN` with nothing.
    #[test]
    fn an_encoded_batch_fills_both_trees() {
        let attr = Arc::new("v".to_string());
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };
        let mut batch = EncodedTuples::default();
        batch.push_numeric(&Value::Int(1), 10);
        batch.push_numeric(&ints(&[1, 2]), 20);
        let mut idx = NumericIndex::new();
        idx.add_encoded(&mut batch);
        assert_eq!(ids(idx.query(&eq(1)).expect("servable")), vec![10]);
        assert_eq!(
            ids(idx.query(&contains(&attr, 1)).expect("servable")),
            vec![20]
        );

        // Into trees that already hold tuples.
        let mut more = EncodedTuples::default();
        more.push_numeric(&Value::Int(2), 30);
        more.push_numeric(&ints(&[2]), 40);
        idx.add_encoded(&mut more);
        assert_eq!(ids(idx.query(&eq(2)).expect("servable")), vec![30]);
        assert_eq!(
            sorted(idx.query(&contains(&attr, 2)).expect("servable")),
            vec![20, 40]
        );

        // And out again, per kind.
        idx.remove_encoded(&mut more);
        assert!(ids(idx.query(&eq(2)).expect("servable")).is_empty());
        assert_eq!(
            ids(idx.query(&contains(&attr, 2)).expect("servable")),
            vec![20]
        );
    }

    /// A value changing kind must move its tuples between the two trees, not leave a copy behind.
    ///
    /// The dangerous case is the same key crossing over: scalar `1` becoming `[1]` and back. A
    /// remove that routed on the *new* value's kind instead of the old one would delete from the
    /// wrong tree, stranding a tuple that no later write can reach — and on the scalar side
    /// nothing re-checks the index's answer, so the stale tuple becomes a wrong row.
    #[test]
    fn a_value_changing_kind_moves_between_trees() {
        let attr = Arc::new("v".to_string());
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };

        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(1), 1);
        assert_eq!(ids(idx.query(&eq(1)).unwrap()), vec![1]);
        assert!(ids(idx.query(&contains(&attr, 1)).unwrap()).is_empty());

        // scalar 1 -> array [1]: the SAME key crosses into the other tree.
        idx.remove(&Value::Int(1), 1);
        idx.add(&ints(&[1]), 1);
        assert!(
            ids(idx.query(&eq(1)).unwrap()).is_empty(),
            "the scalar tuple must be gone, not shadowed"
        );
        assert_eq!(ids(idx.query(&contains(&attr, 1)).unwrap()), vec![1]);

        // array [1] -> scalar 1: and back again.
        idx.remove(&ints(&[1]), 1);
        idx.add(&Value::Int(1), 1);
        assert_eq!(ids(idx.query(&eq(1)).unwrap()), vec![1]);
        assert!(ids(idx.query(&contains(&attr, 1)).unwrap()).is_empty());

        // Removing whatever it currently is leaves nothing on either side.
        idx.remove(&Value::Int(1), 1);
        assert!(idx.is_empty(), "no tuple stranded in either tree");
    }
}
