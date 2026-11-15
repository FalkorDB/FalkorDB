//! The numeric kind of a Range column: CoW B⁺-trees of `(key, entity_id)` tuples, where `key` is
//! the [`encode_numeric`] image of the indexed value — one tree for scalar values and one for the
//! elements of list values ([`KeyKind`]).
//!
//! The trees are private. The column ([`RangeIndex`](super::range::RangeIndex)) reaches them only
//! through [`RangeKind`], the contract it holds every kind to. MVCC is intrinsic: a query
//! snapshots a tree in `O(1)` (root-`Arc` clone) and writes are copy-on-write, so readers never
//! see a torn write. The trees' roots live on the graph version itself — see
//! [`FalkorDbIndex`](super::falkordb_index::FalkorDbIndex).

use std::ops::{Index, IndexMut};
#[cfg(test)]
use std::sync::Arc;

use rustc_hash::FxHashSet;

use super::data_structures::cow_btree::{CowBTree, RangeIter};
use super::encode::{self, KeyKind, encode_numeric};
use super::range::RangeKind;
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
    /// An already-materialised doc set — an intersection, which cannot be produced lazily from
    /// value-ordered streams. Distinct by construction.
    Set(std::collections::hash_set::IntoIter<u64>),
    /// Several streams chained, each doc yielded once — a union across columns, where a doc may
    /// satisfy several members.
    Distinct(DistinctIter),
}

impl Iterator for DocIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match self {
            Self::One(it) => it.next(),
            Self::Many(it) => it.next(),
            Self::Set(it) => it.next(),
            Self::Distinct(it) => it.next(),
        }
    }
}

/// Doc streams read one after another, each doc yielded once.
///
/// **Lazy**: a stream is read only once the ones before it are exhausted, so a `LIMIT` stops the
/// union early. The set holds the docs yielded so far, which is what deduplication needs — a doc
/// matching several members is one row in Cypher — and nothing it has not yielded.
pub struct DistinctIter {
    streams: std::vec::IntoIter<DocIter>,
    current: Option<Box<DocIter>>,
    seen: FxHashSet<u64>,
}

impl DistinctIter {
    /// A union of `streams`, read in order.
    #[must_use]
    pub(super) fn new(streams: Vec<DocIter>) -> Self {
        Self {
            streams: streams.into_iter(),
            current: None,
            seen: FxHashSet::default(),
        }
    }
}

impl Iterator for DistinctIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        loop {
            if let Some(current) = self.current.as_mut() {
                for doc in current.by_ref() {
                    if self.seen.insert(doc) {
                        return Some(doc);
                    }
                }
            }
            // Current stream exhausted (or not started): move to the next one.
            self.current = Some(Box::new(self.streams.next()?));
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

    /// The set of `keys`, given in any order and possibly repeated. Spelled one way per size: none
    /// is [`Empty`](Self::Empty), so a set selecting nothing is recognised as such, and one is a
    /// [`Point`](Self::Point), so `n.v = x` and `IN [x]` read one cursor rather than a chain of one.
    #[must_use]
    pub fn of_keys(keys: Vec<u64>) -> Self {
        match Self::distinct(keys) {
            Self::Keys(keys) if keys.is_empty() => Self::Empty,
            Self::Keys(keys) if keys.len() == 1 => Self::Point(keys[0]),
            keys => keys,
        }
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

    /// The keys in both: two predicates on one scalar under `AND`. Two windows meet in one window
    /// and a point set keeps the points the other side holds, so the answer stays one lookup of the
    /// tree rather than an intersection of rows.
    ///
    /// `WHERE n.v = 5 AND n.v > 2` needs this: the planner's `merge_range_queries` gives up whenever
    /// two conjuncts bound the same side, because at plan time the bounds may still be expressions
    /// it cannot compare. By the time the column sees them they are keys.
    #[must_use]
    pub fn and(
        self,
        other: Self,
    ) -> Self {
        match (self, other) {
            (Self::Range(a, b), Self::Range(c, d)) => {
                let (lo, hi) = (a.max(c), b.min(d));
                if lo <= hi {
                    Self::Range(lo, hi)
                } else {
                    Self::Empty
                }
            }
            (Self::Point(key), other) | (other, Self::Point(key)) => {
                if other.contains(key) {
                    Self::Point(key)
                } else {
                    Self::Empty
                }
            }
            (Self::Keys(keys), other) | (other, Self::Keys(keys)) => Self::of_keys(
                keys.into_iter()
                    .filter(|&key| other.contains(key))
                    .collect(),
            ),
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
        }
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
    by_kind: [Vec<(u64, u64)>; KeyKind::ALL.len()],
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
        let mut tuples = Self::default();
        tuples[KeyKind::Scalar] = scalar;
        tuples
    }

    /// Whether there are no tuples of either kind.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_kind.iter().all(Vec::is_empty)
    }

    /// How many tuples there are, of every kind.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_kind.iter().map(Vec::len).sum()
    }

    /// Empty the tuples, keeping their capacity for reuse.
    pub fn clear(&mut self) {
        self.by_kind.iter_mut().for_each(Vec::clear);
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

/// The numeric kind of one `(label, attribute)` column: entity ids keyed by the order-preserving
/// encoding of their value, so `n.x <predicate> v` is a tree range scan. Non-numeric and `NaN`
/// values are not indexed (parity with the RediSearch NUMERIC field it replaces).
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
    /// Read only by `ArrayContains` (`value IN n.prop`): each probe is a point lookup, so a doc is
    /// reached through one key per probe and cannot be yielded twice. Several probes in a
    /// conjunction are intersected as a set ([`RangeIndex::lookup`](super::range::RangeIndex::lookup)).
    array_tree: Tree,
}

impl NumericIndex {
    /// The kinds of key this kind's trees hold, one tree each.
    const TREES: [KeyKind; 2] = [KeyKind::Scalar, KeyKind::Array];

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

    /// The keys from `min` to `max`, each bound inclusive or not, and `None` unbounded on its side.
    /// A bound that does not encode selects nothing.
    pub(super) fn range_keys(
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

    /// Every `(kind, key, doc)` tuple the index holds, in order: the scalar tree's, then the array
    /// tree's. For tests that compare a column with what it must hold.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(KeyKind, u64, u64)> {
        [
            (KeyKind::Scalar, &self.tree),
            (KeyKind::Array, &self.array_tree),
        ]
        .into_iter()
        .flat_map(|(kind, tree)| {
            tree.range_tuples(0, u64::MAX)
                .map(move |(key, doc)| (kind, key, doc))
        })
        .collect()
    }

    /// The column's leaf pages, shared with every version that holds them: for a test of when they
    /// are freed.
    #[cfg(test)]
    pub(crate) fn pages(&self) -> Vec<Arc<[u8]>> {
        self.tree
            .leaves()
            .into_iter()
            .map(|(_, page)| page)
            .collect()
    }
}

impl RangeKind for NumericIndex {
    type Keys = KeySet;

    /// A scalar is stored under its key in the scalar tree, and a list under each numeric
    /// element's key in the array tree ([`encode::for_each_stored_key`]).
    fn for_each_stored_key(
        &self,
        value: &Value,
        emit: impl FnMut(KeyKind, u64),
    ) {
        encode::for_each_stored_key(value, emit);
    }

    /// Tuples are never re-encoded: re-encoding a decoded key would be a second trip through a
    /// many-to-one map. Into an empty tree the tuples are bulk-built, bottom-up from the sorted
    /// tuples, rather than merged in.
    fn add_encoded(
        &mut self,
        tuples: &EncodedTuples,
    ) {
        for kind in Self::TREES {
            let tuples = &tuples[kind];
            if tuples.is_empty() {
                continue;
            }
            let tree = self.tree_mut(kind);
            if tree.is_empty() {
                *tree = Tree::from_sorted(tuples);
            } else {
                tree.insert_batch(tuples);
            }
        }
    }

    fn remove_encoded(
        &mut self,
        tuples: &EncodedTuples,
    ) {
        for kind in Self::TREES {
            if !tuples[kind].is_empty() {
                self.tree_mut(kind).remove_batch(&tuples[kind]);
            }
        }
    }

    /// Whether the index holds no tuples of either kind.
    fn is_empty(&self) -> bool {
        self.tree.is_empty() && self.array_tree.is_empty()
    }

    /// Heap bytes the two trees hold ([`CowBTree::heap_bytes`]): every page this version reaches,
    /// whether or not another version shares it. Walks every page.
    fn memory_usage(&self) -> usize {
        self.tree.heap_bytes() + self.array_tree.heap_bytes()
    }

    fn scalars(
        &self,
        keys: KeySet,
    ) -> DocIter {
        match keys {
            // `lo > hi` yields nothing.
            KeySet::Empty => DocIter::One(self.tree.range(1, 0)),
            KeySet::Point(key) => DocIter::One(self.tree.point(key)),
            KeySet::Range(lo, hi) => DocIter::One(self.tree.range(lo, hi)),
            // No cursor is built here — see [`UnionIter`]. The tree snapshot is taken now, so
            // every member is answered from the same version however late its cursor is created.
            KeySet::Keys(keys) => DocIter::Many(UnionIter {
                tree: self.tree.clone(),
                keys: keys.into_iter(),
                current: None,
            }),
        }
    }

    /// A point lookup into the array tree, so each matching doc is reached through one key and
    /// appears once. The scalar tree is deliberately not consulted: `1 IN n.v` must not match a
    /// node whose `v` is the scalar `1`.
    ///
    /// May over-return relative to Cypher: the column holds no type tag, so `1 IN n.v` also
    /// matches `v = [true]` (the encoder maps `true` to `1.0`). `utilize_index` keeps the original
    /// filter on this predicate so the runtime rechecks it, which is why the array tree may be
    /// approximate where the scalar tree may not.
    fn lists_holding(
        &self,
        key: u64,
    ) -> DocIter {
        DocIter::One(self.array_tree.point(key))
    }

    fn is_scalar_in(
        keys: &KeySet,
        value: &Value,
    ) -> bool {
        stores(value, KeyKind::Scalar, |key| keys.contains(key))
    }

    fn is_list_holding(
        key: u64,
        value: &Value,
    ) -> bool {
        stores(value, KeyKind::Array, |stored| stored == key)
    }
}

/// Whether `value` is stored under a key of `kind` that `wanted` accepts. Asked of the encoder that
/// decides what the trees hold ([`encode::for_each_stored_key`]), so a scan making this check keeps
/// exactly what a lookup returns: `n.v = 1` matches the scalar `1` and not `[1, 2]`, and `1 IN n.v`
/// matches `[1, 2]` and not the scalar `1`.
fn stores(
    value: &Value,
    kind: KeyKind,
    wanted: impl Fn(u64) -> bool,
) -> bool {
    let mut hit = false;
    encode::for_each_stored_key(value, |stored_kind, key| {
        hit |= stored_kind == kind && wanted(key);
    });
    hit
}

/// The numeric kind, tested through the column that routes to it ([`RangeIndex`]): what these
/// tests exercise is what a query runs.
#[cfg(test)]
mod tests {
    use super::super::range::RangeIndex;
    use super::*;
    use crate::index::IndexQuery;

    /// A union across columns reads its streams one after another and yields each doc once,
    /// holding only what it has yielded: `LIMIT 1` reads one row of the first stream.
    #[test]
    fn a_distinct_union_is_lazy_and_dedups() {
        let set = |ids: &[u64]| {
            DocIter::Set(
                ids.iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter(),
            )
        };
        let mut u = DistinctIter::new(vec![set(&[1]), set(&[1, 2]), set(&[3])]);
        assert_eq!(u.next(), Some(1));
        assert_eq!(u.streams.len(), 2, "the later streams are untouched");
        let mut rest: Vec<u64> = u.collect();
        rest.sort_unstable();
        assert_eq!(rest, vec![2, 3], "1 is not yielded twice");
    }

    fn ids(it: DocIter) -> Vec<u64> {
        it.collect()
    }

    fn attr() -> Arc<String> {
        Arc::new("v".to_string())
    }

    fn eq(value: Value) -> IndexQuery<Value> {
        IndexQuery::Equal { key: attr(), value }
    }

    fn range(
        min: Option<Value>,
        max: Option<Value>,
        include_min: bool,
        include_max: bool,
    ) -> IndexQuery<Value> {
        IndexQuery::Range {
            key: attr(),
            min,
            max,
            include_min,
            include_max,
        }
    }

    /// The rows the column answers `query` with, in the order it yields them.
    fn rows(
        idx: &RangeIndex,
        query: &IndexQuery<Value>,
    ) -> Vec<u64> {
        ids(idx.query(query).expect("servable"))
    }

    /// Every row the column holds a scalar for, in `(value, id)` order.
    fn all(idx: &RangeIndex) -> Vec<u64> {
        rows(idx, &range(None, None, true, true))
    }

    /// A column holding `entries`, bulk-built bottom-up as an indexer batch builds one.
    fn column(entries: &[(Value, u64)]) -> RangeIndex {
        let mut idx = RangeIndex::new();
        idx.add_batch(entries.iter().cloned(), &mut EncodedTuples::default());
        idx
    }

    /// Values: 10→{1,2}, 20→{3}, -5→{4}, 3.5→{5}.
    fn sample() -> RangeIndex {
        let mut idx = RangeIndex::new();
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
        assert_eq!(rows(&idx, &eq(Value::Int(10))), vec![1, 2]); // doc-ascending
        assert_eq!(rows(&idx, &eq(Value::Int(20))), vec![3]);
        assert_eq!(rows(&idx, &eq(Value::Float(10.0))), vec![1, 2]); // 10 == 10.0
        assert!(rows(&idx, &eq(Value::Int(99))).is_empty());
    }

    #[test]
    fn range_inclusive_exclusive_unbounded() {
        let idx = sample();
        let i = |x| Some(Value::Int(x));
        // inclusive [-5, 20] → ordered by (value, id): -5,3.5,10,10,20
        assert_eq!(
            rows(&idx, &range(i(-5), i(20), true, true)),
            vec![4, 5, 1, 2, 3]
        );
        // (10, 20] — exclusive min drops value 10
        assert_eq!(rows(&idx, &range(i(10), i(20), false, true)), vec![3]);
        // [10, 20) — exclusive max drops value 20
        assert_eq!(rows(&idx, &range(i(10), i(20), true, false)), vec![1, 2]);
        // unbounded below, <= 3.5
        assert_eq!(
            rows(&idx, &range(None, Some(Value::Float(3.5)), true, true)),
            vec![4, 5]
        );
        // >= 10, unbounded above
        assert_eq!(rows(&idx, &range(i(10), None, true, true)), vec![1, 2, 3]);
        // fully unbounded → everything, in order
        assert_eq!(all(&idx), vec![4, 5, 1, 2, 3]);
    }

    #[test]
    fn empty_and_degenerate_ranges() {
        let idx = sample();
        let i = |x| Some(Value::Int(x));
        // min > max
        assert!(rows(&idx, &range(i(20), i(10), true, true)).is_empty());
        // (10, 10) — exclusive both sides of one value
        assert!(rows(&idx, &range(i(10), i(10), false, false)).is_empty());
    }

    #[test]
    fn remove_deletes_one_tuple() {
        let mut idx = sample();
        idx.remove(&Value::Int(10), 1);
        assert_eq!(rows(&idx, &eq(Value::Int(10))), vec![2]);
        idx.remove(&Value::Int(10), 2);
        assert!(rows(&idx, &eq(Value::Int(10))).is_empty());
        // the rest are untouched
        assert_eq!(rows(&idx, &eq(Value::Int(20))), vec![3]);
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

        let entries: Vec<(Value, u64)> = temporals
            .into_iter()
            .zip(1u64..)
            .chain(std::iter::once((Value::Int(5), int_id)))
            .collect();
        let bulk = column(&entries);

        let mut incremental = RangeIndex::new();
        for (v, id) in &entries {
            incremental.add(v, *id);
        }

        for (how, idx) in [("bulk build", &bulk), ("incremental add", &incremental)] {
            assert_eq!(
                rows(idx, &range(Some(Value::Int(0)), None, false, true)),
                vec![int_id],
                "{how}: `> 0` must return only the integer, not the temporals"
            );
        }
    }

    #[test]
    fn non_numeric_and_nan_are_not_indexed() {
        let mut idx = RangeIndex::new();
        idx.add(&Value::String(Arc::new("hi".to_string())), 1); // skipped
        idx.add(&Value::Float(f64::NAN), 2); // skipped
        idx.add(&Value::Int(7), 3);
        assert!(!idx.is_empty());
        assert_eq!(all(&idx), vec![3]);
        assert_eq!(rows(&idx, &eq(Value::Int(7))), vec![3]);
        // A bound of a type no kind holds is declined, not answered "no rows".
        assert!(
            idx.query(&range(
                Some(Value::String(Arc::new("a".to_string()))),
                None,
                true,
                true
            ))
            .is_none()
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
        assert_eq!(all(&column(&entries)), all(&sample()));
    }

    #[test]
    fn duplicate_insert_is_idempotent() {
        let mut idx = RangeIndex::new();
        idx.add(&Value::Int(5), 1);
        idx.add(&Value::Int(5), 1);
        assert_eq!(rows(&idx, &eq(Value::Int(5))), vec![1]);
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
        // A conjunction on this column IS answerable — it folds to one window. What a single
        // column still cannot do is a conjunction spanning two attributes.
        assert_eq!(ids(idx.query(&IndexQuery::And(vec![eq])).unwrap()), vec![3]);
        assert!(
            idx.query(&IndexQuery::And(vec![
                IndexQuery::Equal {
                    key: Arc::new("a".to_string()),
                    value: Value::Int(1),
                },
                IndexQuery::Equal {
                    key: Arc::new("b".to_string()),
                    value: Value::Int(2),
                },
            ]))
            .is_none(),
            "two attributes need a cross-column intersection, not this column"
        );
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

        // add_batch == repeated add
        let mut batched = RangeIndex::new();
        batched.add_batch(entries.iter().cloned(), &mut EncodedTuples::default());
        let mut singly = RangeIndex::new();
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

    fn ten_twenty_thirty() -> RangeIndex {
        column(&[
            (Value::Int(10), 0u64),
            (Value::Int(20), 1),
            (Value::Int(30), 2),
        ])
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

    /// One column cannot answer an `OR` spanning two attributes: looking both values up in it
    /// returns `{a=1} ∪ {a=2}` — wrong rows, not missing ones. An `IN` list cannot span
    /// attributes, but an `OR` can, so the column's union checks it itself.
    #[test]
    fn or_declines_members_targeting_different_attributes() {
        let idx = column(&[(Value::Int(1), 10u64), (Value::Int(2), 20)]);
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
        let idx = column(&values);

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
        let mut idx = RangeIndex::new();
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
        let mut idx = RangeIndex::new();
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
        let mut idx = RangeIndex::new();
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
        assert_eq!(idx.numeric().array_tree.len(), 2);

        // Removing the same value takes every element with it, duplicates included.
        idx.remove(&ints(&[1, 1, 2]), 7);
        assert!(idx.is_empty(), "both trees must be empty again");

        // The batch paths dedup too: one tuple per distinct element, removable in one go.
        let mut scratch = EncodedTuples::default();
        idx.add_batch([(ints(&[3, 3, 4]), 8)], &mut scratch);
        assert_eq!(idx.numeric().array_tree.len(), 2);
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
        let mut idx = RangeIndex::new();
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
        assert_eq!(idx.numeric().array_tree.len(), 1);
        assert!(idx.numeric().tree.is_empty());
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
        let mut idx = RangeIndex::new();
        let mut batch = EncodedTuples::default();
        idx.push_encoded(&Value::Int(1), 10, &mut batch);
        idx.push_encoded(&ints(&[1, 2]), 20, &mut batch);
        idx.add_encoded(&mut batch);
        assert_eq!(ids(idx.query(&eq(1)).expect("servable")), vec![10]);
        assert_eq!(
            ids(idx.query(&contains(&attr, 1)).expect("servable")),
            vec![20]
        );

        // Into trees that already hold tuples.
        let mut more = EncodedTuples::default();
        idx.push_encoded(&Value::Int(2), 30, &mut more);
        idx.push_encoded(&ints(&[2]), 40, &mut more);
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

        let mut idx = RangeIndex::new();
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

    /// `KeySet::and` keeps a conjunction one lookup: windows meet in a window, a point set keeps
    /// the points the other side holds, and nothing left is spelled `Empty`.
    #[test]
    fn key_sets_intersect_into_one_lookup() {
        use KeySet::{Empty, Keys, Point, Range};
        assert_eq!(Range(1, 9).and(Range(5, 20)), Range(5, 9));
        assert_eq!(
            Range(1, 4).and(Range(5, 20)),
            Empty,
            "windows that do not meet"
        );
        assert_eq!(Point(5).and(Range(1, 9)), Point(5));
        assert_eq!(Range(1, 9).and(Point(10)), Empty);
        assert_eq!(Keys(vec![1, 5, 9]).and(Range(2, 9)), Keys(vec![5, 9]));
        assert_eq!(Keys(vec![1, 5]).and(Keys(vec![5, 9])), Point(5));
        assert_eq!(Keys(vec![1, 5]).and(Range(6, 9)), Empty);
        assert_eq!(Point(5).and(Keys(vec![1, 5])), Point(5));
        assert_eq!(Empty.and(Range(0, u64::MAX)), Empty);
        assert_eq!(KeySet::of_keys(vec![3, 1, 3]), Keys(vec![1, 3]));
        assert_eq!(
            KeySet::of_keys(vec![5, 5]),
            Point(5),
            "one key is one cursor"
        );
        assert_eq!(KeySet::of_keys(Vec::new()), Empty);
    }

    /// Two bounds on one attribute collapse to a single window rather than needing a doc-stream
    /// intersection. The planner cannot do this itself — `merge_range_queries` bails whenever two
    /// conjuncts constrain the same side, because at plan time the bounds may still be
    /// expressions it cannot compare. Here they are concrete.
    #[test]
    fn same_key_conjunctions_fold_to_one_window() {
        let attr = Arc::new("v".to_string());
        let idx = column(&[
            (Value::Int(1), 1u64),
            (Value::Int(5), 5),
            (Value::Int(9), 9),
            (Value::Int(20), 20),
        ]);
        let range =
            |min: Option<i64>, max: Option<i64>, inc_min: bool, inc_max: bool| IndexQuery::Range {
                key: attr.clone(),
                min: min.map(Value::Int),
                max: max.map(Value::Int),
                include_min: inc_min,
                include_max: inc_max,
            };
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };
        let got = |q: IndexQuery<Value>| sorted(idx.query(&q).expect("servable"));

        // Two lower bounds: the tighter one wins.
        assert_eq!(
            got(IndexQuery::And(vec![
                range(Some(1), None, false, true),
                range(Some(9), None, true, true),
            ])),
            vec![9, 20]
        );
        // Lower and upper from different conjuncts.
        assert_eq!(
            got(IndexQuery::And(vec![
                range(Some(5), None, true, true),
                range(None, Some(9), true, true),
            ])),
            vec![5, 9]
        );
        // `Equal` is a degenerate window, and an equality inside a satisfied range survives.
        assert_eq!(
            got(IndexQuery::And(vec![
                eq(5),
                range(Some(2), None, true, true)
            ])),
            vec![5]
        );
        // The same predicate twice — legal Cypher (`WHERE a.p = 1 AND a.p = 1`), which the planner
        // passes through as written — is one window, not a failure.
        assert_eq!(got(IndexQuery::And(vec![eq(5), eq(5)])), vec![5]);
        // `= null` holds for nothing, so a conjunction with it is answered with no rows, as
        // `= null` alone is, rather than declined.
        assert!(
            got(IndexQuery::And(vec![
                IndexQuery::Equal {
                    key: attr.clone(),
                    value: Value::Null,
                },
                range(Some(1), None, true, true),
            ]))
            .is_empty()
        );
        // Likewise a range with a null bound, as that range alone is.
        assert!(
            got(IndexQuery::And(vec![
                IndexQuery::Range {
                    key: attr.clone(),
                    min: Some(Value::Null),
                    max: Some(Value::Int(9)),
                    include_min: true,
                    include_max: true,
                },
                range(Some(1), None, true, true),
            ]))
            .is_empty()
        );
        // An equality outside the range yields nothing — answerable, and the answer is no rows.
        assert!(
            got(IndexQuery::And(vec![
                eq(1),
                range(Some(8), None, true, true)
            ]))
            .is_empty()
        );
        // Contradictory bounds likewise.
        assert!(
            got(IndexQuery::And(vec![
                range(Some(9), None, true, true),
                range(None, Some(2), true, true),
            ]))
            .is_empty()
        );
        // A scalar conjunct and a list conjunct: no value is both, so no rows — answered, not
        // declined.
        assert!(got(IndexQuery::And(vec![eq(1), contains(&attr, 1)])).is_empty());
        // Nesting is flattened — it still constrains one column.
        assert_eq!(
            got(IndexQuery::And(vec![
                IndexQuery::And(vec![range(Some(5), None, true, true)]),
                range(None, Some(9), true, true),
            ])),
            vec![5, 9]
        );
    }

    /// A conjunction the column cannot answer whole is declined, rather than answered from a
    /// subset of the conjuncts — that would return rows the query excludes.
    #[test]
    fn unservable_conjunctions_are_declined() {
        let idx = column(&[(Value::Int(1), 1u64)]);
        let eq = |attr: &str, v: i64| IndexQuery::Equal {
            key: Arc::new(attr.to_string()),
            value: Value::Int(v),
        };
        assert!(
            idx.query(&IndexQuery::And(vec![eq("a", 1), eq("b", 2)]))
                .is_none(),
            "two attributes need a cross-column intersection"
        );
        assert!(
            idx.query(&IndexQuery::And(vec![
                eq("a", 1),
                IndexQuery::Equal {
                    key: Arc::new("a".to_string()),
                    value: Value::String(Arc::new("x".to_string())),
                },
            ]))
            .is_none(),
            "a conjunct no kind serves makes the whole conjunction unservable"
        );
        assert!(
            idx.query(&IndexQuery::And(vec![])).is_none(),
            "an empty conjunction names no column"
        );
    }
}
