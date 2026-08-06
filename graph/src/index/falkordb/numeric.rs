//! The index (PR2 · P2): a CoW B⁺-tree of `(key, entity_id)`
//! tuples, where `key` is the [`encode_numeric`] image of the indexed value.
//!
//! Concrete-first — no trait yet. The generic `Index` trait is extracted at P4,
//! when there is an actual dispatch caller (the indexer). MVCC is intrinsic: a
//! query snapshots the tree in `O(1)` (root-`Arc` clone) and writes are
//! copy-on-write, so readers never see a torn write. Folding the root into the
//! graph's committed version is P3.

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
    Many(std::iter::Flatten<std::vec::IntoIter<TreeIter>>),
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
    /// traversal.
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
    /// Non-numeric / `NaN` are dropped. Collects + sorts internally — the tree's `insert_batch` needs
    /// a sorted slice — so callers never have to pre-materialize; cheaper than repeated
    /// [`add`](Self::add).
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
    ) {
        let mut pairs = Self::encode_pairs(entries);
        pairs.sort_unstable();
        self.tree.insert_batch(&pairs);
    }

    /// Remove a batch of `(value, id)` entries, consuming any iterator. Non-numeric / `NaN` dropped;
    /// collect + sort internally — for the write path's mass-delete / mass-update column.
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
    ) {
        let mut pairs = Self::encode_pairs(entries);
        pairs.sort_unstable();
        self.tree.remove_batch(&pairs);
    }

    /// Encode `(value, id)` entries to `(key, id)` tree tuples, dropping non-numeric / `NaN` values.
    fn encode_pairs(entries: impl IntoIterator<Item = (Value, u64)>) -> Vec<(u64, u64)> {
        entries
            .into_iter()
            .filter_map(|(v, id)| encode_numeric(&v).map(|k| (k, id)))
            .collect()
    }

    /// Build directly from already-encoded `(key, doc)` tuples, in any order — how the
    /// install adopts a background-built BASE without re-encoding or re-sorting it per row.
    #[must_use]
    pub fn from_encoded(mut pairs: Vec<(u64, u64)>) -> Self {
        pairs.sort_unstable();
        pairs.dedup();
        Self {
            tree: Tree::from_sorted(&pairs),
        }
    }

    /// Encode `(value, id)` entries to tree tuples, dropping non-numeric / `NaN`.
    /// Public so a background build can encode BASE off the write thread.
    #[must_use]
    pub fn encode_entries(entries: Vec<(Value, u64)>) -> Vec<(u64, u64)> {
        Self::encode_pairs(entries)
    }

    /// Every `(key, doc)` tuple, in key order — the install's DELTA/TOMB enumeration.
    /// `O(n)`; intended for build-sized artifacts, not a populated column.
    ///
    /// Uses the tree's bulk `for_each_tuple` walk rather than a cursor: it reads each page's
    /// tuples in a tight loop, skipping the per-entry enum dispatch and `Iterator::next` a lazy
    /// cursor pays, and this call always consumes the whole artifact anyway.
    #[must_use]
    pub fn encoded_tuples(&self) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        self.tree.for_each_tuple(|k, d| out.push((k, d)));
        out
    }

    /// Add already-encoded tuples. Used by the install to replay DELTA onto BASE, where the
    /// tuples come straight out of another tree and must not be re-encoded — re-encoding a
    /// decoded key would be a second trip through a many-to-one map.
    pub fn add_encoded(
        &mut self,
        pairs: &mut Vec<(u64, u64)>,
    ) {
        pairs.sort_unstable();
        self.tree.insert_batch(pairs);
    }

    /// Remove already-encoded tuples — the install subtracting TOMB from BASE.
    pub fn remove_encoded(
        &mut self,
        pairs: &mut Vec<(u64, u64)>,
    ) {
        pairs.sort_unstable();
        self.tree.remove_batch(pairs);
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

    /// Dispatch the numeric *leaf* predicates. `Equal` → [`point`](Self::point),
    /// `Range` → [`range`](Self::range). Composite (`And`/`Or`) and non-numeric
    /// variants return `None` — the query router composes or rejects those (P5).
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
            IndexQuery::Or(children) => self.union(children),
            _ => None,
        }
    }

    /// A union of `Equal` leaves — what `n.a IN [...]` desugars to before it reaches the index.
    ///
    /// Returns `None` unless **every** member encodes to a numeric key. Serving only the numeric
    /// subset would silently drop rows: a member the numeric column cannot represent (a string,
    /// say) may still match entities held by another kind, and answering without them is a wrong
    /// answer rather than a slower one. Declining hands the whole query back to the caller intact.
    ///
    /// Members are deduplicated by encoded key, so `IN [1,1,2]` yields each entity once. That is
    /// sufficient for uniqueness overall: a doc appears under one key per column (the encoder
    /// rejects lists, so an array-valued property is not in this column at all), so distinct keys
    /// cannot both hold the same doc.
    fn union(
        &self,
        children: &[IndexQuery<Value>],
    ) -> Option<DocIter> {
        // Flatten nested unions. `a IN [..] OR a IN [..]` arrives as an `Or` of `Or`s, and each
        // level is still a union of the same column, so the nesting carries no meaning here.
        fn collect(
            children: &[IndexQuery<Value>],
            keys: &mut Vec<u64>,
        ) -> Option<()> {
            for child in children {
                match child {
                    IndexQuery::Equal { value, .. } => keys.push(encode_numeric(value)?),
                    IndexQuery::Or(nested) => collect(nested, keys)?,
                    // A range or other combinator inside the union — not this shape.
                    _ => return None,
                }
            }
            Some(())
        }
        let mut keys = Vec::with_capacity(children.len());
        collect(children, &mut keys)?;
        keys.sort_unstable();
        keys.dedup();
        let cursors: Vec<TreeIter> = keys.into_iter().map(|k| self.tree.point(k)).collect();
        Some(DocIter::Many(cursors.into_iter().flatten()))
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
        batched.add_batch(entries.iter().cloned());
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
        batched.remove_batch(removes.iter().cloned());
        for (v, id) in &removes {
            singly.remove(v, *id);
        }
        assert_eq!(all(&batched), all(&singly));
    }

    /// `IN [...]` reaches the index as a union of `Equal`s. It is served natively only when every
    /// member is numeric — a mixed list must be declined whole, not answered from the numeric
    /// members alone, or rows another kind holds would silently vanish.
    #[test]
    fn union_serves_all_numeric_and_declines_mixed() {
        let attr = Arc::new("v".to_string());
        let eq = |v: Value| IndexQuery::Equal {
            key: attr.clone(),
            value: v,
        };
        let idx = NumericIndex::from_entries(
            [
                (&Value::Int(10), 0u64),
                (&Value::Int(20), 1),
                (&Value::Int(30), 2),
            ]
            .into_iter(),
        );

        // all-numeric union -> every matching id, once each
        let mut got = ids(idx
            .query(&IndexQuery::Or(vec![
                eq(Value::Int(10)),
                eq(Value::Int(30)),
            ]))
            .expect("all-numeric union is servable"));
        got.sort_unstable();
        assert_eq!(got, vec![0, 2]);

        // duplicate members must not yield an id twice
        let mut dup = ids(idx
            .query(&IndexQuery::Or(vec![
                eq(Value::Int(20)),
                eq(Value::Int(20)),
            ]))
            .expect("servable"));
        dup.sort_unstable();
        assert_eq!(dup, vec![1], "duplicate members must dedup");

        // a non-numeric member declines the WHOLE union — not "just the numeric part"
        assert!(
            idx.query(&IndexQuery::Or(vec![
                eq(Value::Int(10)),
                eq(Value::String(Arc::new("a".to_string()))),
            ]))
            .is_none(),
            "a mixed list must fall back rather than drop the non-numeric member"
        );

        // `a IN [..] OR a IN [..]` reaches the index as an Or of Ors — the nesting carries no
        // meaning for a single column, so it must flatten rather than decline.
        let mut nested = ids(idx
            .query(&IndexQuery::Or(vec![
                IndexQuery::Or(vec![eq(Value::Int(10)), eq(Value::Int(20))]),
                IndexQuery::Or(vec![eq(Value::Int(30))]),
            ]))
            .expect("nested unions flatten"));
        nested.sort_unstable();
        assert_eq!(nested, vec![0, 1, 2]);

        // a non-numeric member nested one level down still declines the whole union
        assert!(
            idx.query(&IndexQuery::Or(vec![
                eq(Value::Int(10)),
                IndexQuery::Or(vec![eq(Value::String(Arc::new("a".to_string())))]),
            ]))
            .is_none(),
            "a nested non-numeric member must decline the whole union"
        );

        // a member that is not a plain Equal (a nested range) is likewise declined
        assert!(
            idx.query(&IndexQuery::Or(vec![
                eq(Value::Int(10)),
                IndexQuery::Range {
                    key: attr.clone(),
                    min: Some(Value::Int(0)),
                    max: None,
                    include_min: true,
                    include_max: true,
                },
            ]))
            .is_none()
        );
    }
}
