//! The numeric index: a CoW B⁺-tree of `(key, entity_id)` tuples, where `key` is the
//! [`encode_numeric`] image of the indexed value.
//!
//! One concrete type, no trait: index kinds are a closed set, dispatched through the
//! [`IndexColumn`](super::falkordb_index::IndexColumn) enum. MVCC is intrinsic: a query snapshots
//! the tree in `O(1)` (root-`Arc` clone) and writes are copy-on-write, so readers never see a torn
//! write. The tree's root lives on the graph version itself — see
//! [`FalkorDbIndex`](super::falkordb_index::FalkorDbIndex).

use std::sync::Arc;

use super::data_structures::cow_btree::{CowBTree, RangeIter};
use super::encode::encode_numeric;
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

/// Encoded `(key, doc)` tuples for one column: what everything that moves a column's tuples in
/// encoded form carries (a unit of work's collapsed adds and removes, an indexer batch).
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct EncodedTuples {
    pub scalar: Vec<(u64, u64)>,
}

impl EncodedTuples {
    /// Tuples of scalar values.
    #[must_use]
    pub fn scalars(scalar: Vec<(u64, u64)>) -> Self {
        Self { scalar }
    }

    /// Whether there are no tuples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scalar.is_empty()
    }

    /// How many tuples there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scalar.len()
    }

    /// Empty the tuples, keeping their capacity for reuse.
    pub fn clear(&mut self) {
        self.scalar.clear();
    }

    /// Sort the tuples ascending and drop duplicates: the **sorted and unique** shape a tree's
    /// bulk build requires.
    pub fn sort_dedup(&mut self) {
        self.scalar.sort_unstable();
        self.scalar.dedup();
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
    tree: Tree,
}

impl NumericIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bulk-build from `(value, id)` pairs in any order. Non-numeric / `NaN`
    /// values are dropped. Far cheaper than repeated [`add`](Self::add) for
    /// initial population — one sort + bottom-up page build, no per-item
    /// traversal. The tree's tests compare it with incremental adds.
    #[cfg(test)]
    #[must_use]
    pub fn from_entries<'a>(entries: impl IntoIterator<Item = (&'a Value, u64)>) -> Self {
        let mut pairs: Vec<(u64, u64)> = entries
            .into_iter()
            .filter_map(|(v, id)| encode_numeric(v).map(|k| (k, id)))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        Self {
            tree: Tree::from_sorted(&pairs),
        }
    }

    /// Index `id` under `value`. A no-op for non-numeric / `NaN` values and
    /// idempotent for an already-present `(value, id)`.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        if let Some(k) = encode_numeric(value) {
            // The bool reports whether the tuple was newly inserted — for callers keeping an
            // exact live count. This index derives its counts from the tree, so it is discarded
            // deliberately rather than ignored.
            let _newly_inserted = self.tree.insert(k, id);
        }
    }

    /// Remove `id` from under `value`. A no-op if it was never indexed.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        if let Some(k) = encode_numeric(value) {
            let _was_present = self.tree.remove(k, id);
        }
    }

    /// Add a batch of `(value, id)` entries, consuming any iterator (the runtime's columnar batch).
    /// Non-numeric / `NaN` are dropped, and the encoded tuples are sorted here because the tree's
    /// `insert_batch` needs a sorted slice. Cheaper than repeated [`add`](Self::add).
    ///
    /// `scratch` is the caller's encode buffer and its contents are replaced. Taking it rather than
    /// allocating internally lets a commit that touches many columns allocate once and reuse the
    /// capacity, instead of a fresh `Vec` per column.
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut Vec<(u64, u64)>,
    ) {
        Self::encode_pairs_into(entries, scratch);
        scratch.sort_unstable();
        self.tree.insert_batch(scratch);
    }

    /// Remove a batch of `(value, id)` entries, consuming any iterator — the write path's
    /// mass-delete / mass-update column. Non-numeric / `NaN` dropped, encoded tuples sorted here.
    ///
    /// `scratch` is the caller's encode buffer — see [`add_batch`](Self::add_batch).
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut Vec<(u64, u64)>,
    ) {
        Self::encode_pairs_into(entries, scratch);
        scratch.sort_unstable();
        self.tree.remove_batch(scratch);
    }

    /// Encode `(value, id)` entries to `(key, id)` tree tuples in `out` (cleared first), dropping
    /// non-numeric / `NaN` values.
    ///
    /// Takes the buffer rather than returning one so a caller applying many columns in a row
    /// allocates once and reuses the capacity, instead of one `Vec` per column per commit.
    fn encode_pairs_into(
        entries: impl IntoIterator<Item = (Value, u64)>,
        out: &mut Vec<(u64, u64)>,
    ) {
        out.clear();
        let entries = entries.into_iter();
        out.reserve(entries.size_hint().0);
        out.extend(entries.filter_map(|(v, id)| encode_numeric(&v).map(|k| (k, id))));
    }

    /// Add already-encoded tuples: the adds a unit of work's staged writes collapsed to, or an
    /// indexer batch. Tuples are never re-encoded: re-encoding a decoded key would be a second trip
    /// through a many-to-one map. Into an empty index the tuples are bulk-built, bottom-up from
    /// the sorted tuples, rather than merged in.
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        if self.tree.is_empty() {
            self.tree = Tree::from_sorted(&tuples.scalar);
        } else {
            self.tree.insert_batch(&tuples.scalar);
        }
    }

    /// Remove already-encoded tuples: the removes a unit of work's staged writes collapsed to.
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        self.tree.remove_batch(&tuples.scalar);
    }

    /// Whether the index holds no tuples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Entity ids whose value equals `value`. Empty for a non-numeric value.
    #[must_use]
    pub fn point(
        &self,
        value: &Value,
    ) -> DocIter {
        match encode_numeric(value) {
            Some(k) => DocIter::One(self.tree.point(k)),
            None => self.empty(),
        }
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
        // Map value bounds to inclusive key bounds. Exclusive bounds step one key
        // inward: encodings of distinct f64 are adjacent u64, so `k+1` / `k-1` is
        // exactly the next / previous representable value. The `checked_*` guards
        // catch the ±∞ edges (`x > +inf`, `x < -inf`) as empty.
        let lo = match min {
            None => 0,
            Some(v) => {
                let Some(k) = encode_numeric(v) else {
                    return self.empty();
                };
                if include_min {
                    k
                } else {
                    match k.checked_add(1) {
                        Some(k1) => k1,
                        None => return self.empty(),
                    }
                }
            }
        };
        let hi = match max {
            None => u64::MAX,
            Some(v) => {
                let Some(k) = encode_numeric(v) else {
                    return self.empty();
                };
                if include_max {
                    k
                } else {
                    match k.checked_sub(1) {
                        Some(k1) => k1,
                        None => return self.empty(),
                    }
                }
            }
        };
        if lo > hi {
            return self.empty();
        }
        DocIter::One(self.tree.range(lo, hi))
    }

    /// Dispatch the numeric predicates. `Equal` → [`point`](Self::point), `Range` →
    /// [`range`](Self::range), an `IN` list → [`in_list`](Self::in_list), and an `OR` of
    /// equalities and `IN` lists → [`union`](Self::union). `And` and non-numeric variants return
    /// `None` — the query router composes or rejects those.
    #[must_use]
    pub fn query(
        &self,
        q: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        match q {
            IndexQuery::Equal { value, .. } => Some(self.point(value)),
            IndexQuery::Range {
                min,
                max,
                include_min,
                include_max,
                ..
            } => Some(self.range(min.as_ref(), max.as_ref(), *include_min, *include_max)),
            IndexQuery::InList {
                list: Value::List(items),
                ..
            } => self.in_list(items),
            IndexQuery::Or(children) => self.union(children),
            _ => None,
        }
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
    fn in_list(
        &self,
        items: &[Value],
    ) -> Option<DocIter> {
        if items.is_empty() {
            return None;
        }
        let keys = items
            .iter()
            .map(encode_numeric)
            .collect::<Option<Vec<u64>>>()?;
        Some(self.union_of(keys))
    }

    /// An `OR` of `Equal`s and `IN` lists — `a = 1 OR a IN [2, 3]`.
    ///
    /// Unlike an `IN` list, an `OR` can name several attributes, and this index is one column:
    /// answering `a = 1 OR b = 2` from it would look up both `1` and `2` in `a` and return
    /// `{a=1} ∪ {a=2}` — wrong rows, not missing ones. So every member must target the same
    /// attribute, and every value must encode, as in [`in_list`](Self::in_list); otherwise `None`.
    fn union(
        &self,
        children: &[IndexQuery<Value>],
    ) -> Option<DocIter> {
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
            for value in values {
                keys.push(encode_numeric(value)?);
            }
        }
        (!keys.is_empty()).then(|| self.union_of(keys))
    }

    /// A lazy chain of point cursors over `keys` — see [`UnionIter`].
    ///
    /// Keys are deduplicated, so `IN [1,1,2]` yields each entity once. That is sufficient for
    /// uniqueness overall: a doc appears under one key per column (the encoder rejects lists, so
    /// an array-valued property is not in this column at all), so distinct keys cannot both hold
    /// the same doc.
    fn union_of(
        &self,
        mut keys: Vec<u64>,
    ) -> DocIter {
        keys.sort_unstable();
        keys.dedup();
        // No cursor is built here — see [`UnionIter`]. The tree snapshot is taken now, so every
        // member is answered from the same version however late its cursor is created.
        DocIter::Many(UnionIter {
            tree: self.tree.clone(),
            keys: keys.into_iter(),
            current: None,
        })
    }

    /// An iterator over no entries (`lo > hi` yields nothing).
    fn empty(&self) -> DocIter {
        DocIter::One(self.tree.range(1, 0))
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
        batched.add_batch(entries.iter().cloned(), &mut Vec::new());
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
        batched.remove_batch(removes.iter().cloned(), &mut Vec::new());
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
}
