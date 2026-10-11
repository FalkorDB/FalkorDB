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
use super::encode::{KeySpace, encode_numeric, for_each_stored_key};
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
    /// A separate key space, not a convenience — see [`KeySpace`]. Sharing one tree would make a
    /// node with `v = [1, 2]` carry the key for `1`, so `WHERE n.v = 1` would match it.
    ///
    /// Read only by `ArrayContains` (`value IN n.prop`), which is a point lookup, so a doc is
    /// reached through exactly one key per query and cannot be yielded twice.
    array_tree: Tree,
}

/// Encoded `(key, doc)` tuples for one column, split by the [`KeySpace`] they belong to.
///
/// Everything that moves a column's tuples around in encoded form — the online build's BASE, the
/// collapsed adds and removes of a unit of work, the RDB-load populate — has to keep the two key
/// spaces apart for the same reason the trees do. A flat list of tuples would lose which tree
/// each one goes back into.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct EncodedTuples {
    pub scalar: Vec<(u64, u64)>,
    pub array: Vec<(u64, u64)>,
}

impl EncodedTuples {
    /// Scalar-only tuples — the shape a column holding no list values produces.
    #[must_use]
    pub fn scalars(scalar: Vec<(u64, u64)>) -> Self {
        Self {
            scalar,
            array: Vec::new(),
        }
    }

    /// Whether there are no tuples in either key space.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scalar.is_empty() && self.array.is_empty()
    }

    /// Total tuples across both key spaces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scalar.len() + self.array.len()
    }

    /// Empty both halves, keeping their capacity for reuse.
    pub fn clear(&mut self) {
        self.scalar.clear();
        self.array.clear();
    }

    /// The half for `space`.
    #[must_use]
    pub fn half(
        &self,
        space: KeySpace,
    ) -> &Vec<(u64, u64)> {
        match space {
            KeySpace::Scalar => &self.scalar,
            KeySpace::Array => &self.array,
        }
    }

    /// The half for `space`, mutably.
    pub fn half_mut(
        &mut self,
        space: KeySpace,
    ) -> &mut Vec<(u64, u64)> {
        match space {
            KeySpace::Scalar => &mut self.scalar,
            KeySpace::Array => &mut self.array,
        }
    }

    /// Append the numeric tuples `(value, id)` contributes, each to the half it belongs in.
    /// Unsorted, and a list repeating a number appends its key more than once — see
    /// [`sort_dedup`](Self::sort_dedup).
    pub fn push_numeric(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        for_each_stored_key(value, |space, key| self.half_mut(space).push((key, id)));
    }

    /// Sort both halves ascending and drop duplicate tuples: the **sorted and unique** shape the
    /// tree's bulk builds require.
    pub fn sort_dedup(&mut self) {
        for half in [&mut self.scalar, &mut self.array] {
            half.sort_unstable();
            half.dedup();
        }
    }

    /// Drop every tuple whose doc fails `keep` — the install's deleted-entity backstop, which must
    /// sweep both key spaces or a deleted node survives in its array half.
    pub fn retain_docs(
        &mut self,
        mut keep: impl FnMut(u64) -> bool,
    ) {
        self.scalar.retain(|&(_, doc)| keep(doc));
        self.array.retain(|&(_, doc)| keep(doc));
    }
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
        let mut tuples = EncodedTuples::default();
        for (v, id) in entries {
            tuples.push_numeric(v, id);
        }
        tuples.sort_dedup();
        Self {
            tree: Tree::from_sorted(&tuples.scalar),
            array_tree: Tree::from_sorted(&tuples.array),
        }
    }

    /// Index `id` under `value`. A no-op for non-numeric / `NaN` values and
    /// idempotent for an already-present `(value, id)`.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        for_each_stored_key(value, |space, k| {
            // The bool reports whether the tuple was newly inserted — for callers keeping an
            // exact live count. This index derives its counts from the tree, so it is discarded
            // deliberately rather than ignored.
            let _newly_inserted = self.tree_mut(space).insert(k, id);
        });
    }

    /// Remove `id` from under `value`. A no-op if it was never indexed.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        for_each_stored_key(value, |space, k| {
            let _was_present = self.tree_mut(space).remove(k, id);
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
        self.tree.insert_batch(&scratch.scalar);
        self.array_tree.insert_batch(&scratch.array);
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
        self.tree.remove_batch(&scratch.scalar);
        self.array_tree.remove_batch(&scratch.array);
    }

    /// Encode `(value, id)` entries to `(key, id)` tree tuples in `out` (cleared first), each in
    /// the key space it belongs to, sorted and unique. Non-numeric / `NaN` values are dropped.
    ///
    /// Takes the buffer rather than returning one so a caller applying many columns in a row
    /// allocates once and reuses the capacity, instead of one buffer per column per commit.
    fn encode_pairs_into(
        entries: impl IntoIterator<Item = (Value, u64)>,
        out: &mut EncodedTuples,
    ) {
        out.clear();
        let entries = entries.into_iter();
        out.scalar.reserve(entries.size_hint().0);
        for (v, id) in entries {
            out.push_numeric(&v, id);
        }
        out.sort_dedup();
    }

    /// Build from already-encoded `(key, doc)` tuples, one stream per key space, each arriving
    /// **sorted ascending and unique** — how the install builds the finished column from BASE,
    /// TOMB and DELTA in one pass. Only one page's worth of each input is buffered.
    #[must_use]
    pub fn from_sorted_tuples(
        scalar: impl IntoIterator<Item = (u64, u64)>,
        array: impl IntoIterator<Item = (u64, u64)>,
    ) -> Self {
        Self {
            tree: Tree::from_sorted_iter(scalar),
            array_tree: Tree::from_sorted_iter(array),
        }
    }

    /// Every `(key, doc)` tuple in key space `space`, in `(key, doc)` order, read lazily from a
    /// snapshot of the tree: the install streams TOMB and DELTA through this without collecting
    /// them.
    pub fn tuples(
        &self,
        space: KeySpace,
    ) -> impl Iterator<Item = (u64, u64)> + use<> {
        self.tree_ref(space).range_tuples(0, u64::MAX)
    }

    /// Add already-encoded tuples — the adds a unit of work's staged writes collapsed to. Tuples
    /// are never re-encoded: re-encoding a decoded key would be a second trip through a
    /// many-to-one map.
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.scalar.sort_unstable();
        tuples.array.sort_unstable();
        self.tree.insert_batch(&tuples.scalar);
        self.array_tree.insert_batch(&tuples.array);
    }

    /// Remove already-encoded tuples — the removes a unit of work's staged writes collapsed to.
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.scalar.sort_unstable();
        tuples.array.sort_unstable();
        self.tree.remove_batch(&tuples.scalar);
        self.array_tree.remove_batch(&tuples.array);
    }

    /// Whether the index holds no tuples in either key space.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty() && self.array_tree.is_empty()
    }

    /// The tree holding key space `space`.
    fn tree_ref(
        &self,
        space: KeySpace,
    ) -> &Tree {
        match space {
            KeySpace::Scalar => &self.tree,
            KeySpace::Array => &self.array_tree,
        }
    }

    /// The tree holding key space `space`, mutably.
    fn tree_mut(
        &mut self,
        space: KeySpace,
    ) -> &mut Tree {
        match space {
            KeySpace::Scalar => &mut self.tree,
            KeySpace::Array => &mut self.array_tree,
        }
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

    /// Entity ids whose **list**-valued property contains `value` — `value IN n.prop`.
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
        match encode_numeric(value) {
            Some(k) => DocIter::One(self.array_tree.point(k)),
            None => self.empty(),
        }
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
            IndexQuery::ArrayContains { value, .. } => Some(self.array_contains(value)),
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
    /// Also returns `None` unless every member targets the **same attribute**. This index is one
    /// column: answering `a = 1 OR b = 2` from it would look up both `1` and `2` in `a` and return
    /// `{a=1} ∪ {a=2}` — wrong rows, not missing ones. The facade checks this too before routing
    /// here, but the check belongs on this side of the boundary as well: it is a precondition of
    /// the operation, not of one caller, and it was previously possible to reach `union` directly
    /// and bypass it.
    ///
    /// Members are deduplicated by encoded key, so `IN [1,1,2]` yields each entity once. That is
    /// sufficient for uniqueness overall: a doc appears under at most one key in the scalar tree,
    /// which is the only tree a union reads (a list value's elements live in the array tree), so
    /// distinct keys cannot both hold the same doc.
    fn union(
        &self,
        children: &[IndexQuery<Value>],
    ) -> Option<DocIter> {
        // Flatten nested unions. `a IN [..] OR a IN [..]` arrives as an `Or` of `Or`s, and each
        // level is still a union of the same column, so the nesting carries no meaning here.
        fn collect<'a>(
            children: &'a [IndexQuery<Value>],
            keys: &mut Vec<u64>,
            attr: &mut Option<&'a Arc<String>>,
        ) -> Option<()> {
            for child in children {
                match child {
                    IndexQuery::Equal { key, value } => {
                        match attr {
                            Some(first) if *first != key => return None,
                            Some(_) => {}
                            None => *attr = Some(key),
                        }
                        keys.push(encode_numeric(value)?);
                    }
                    IndexQuery::Or(nested) => collect(nested, keys, attr)?,
                    // A range or other combinator inside the union — not this shape.
                    _ => return None,
                }
            }
            Some(())
        }
        let mut keys = Vec::with_capacity(children.len());
        collect(children, &mut keys, &mut None)?;
        keys.sort_unstable();
        keys.dedup();
        // No cursor is built here — see [`UnionIter`]. The tree snapshot is taken now, so every
        // member is answered from the same version however late its cursor is created.
        Some(DocIter::Many(UnionIter {
            tree: self.tree.clone(),
            keys: keys.into_iter(),
            current: None,
        }))
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
    use thin_vec::ThinVec;

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

    /// One `NumericIndex` is one column, so a union spanning two attributes cannot be answered
    /// here: looking both values up in this column returns `{a=1} ∪ {a=2}` — wrong rows, not
    /// missing ones. The facade declines this before routing, but the guard has to hold on this
    /// side too, and this test reaches `query` directly the way a future caller would.
    #[test]
    fn union_declines_members_targeting_different_attributes() {
        let idx =
            NumericIndex::from_entries([(&Value::Int(1), 10u64), (&Value::Int(2), 20)].into_iter());
        let eq = |attr: &str, v: i64| IndexQuery::Equal {
            key: Arc::new(attr.to_string()),
            value: Value::Int(v),
        };

        assert!(
            idx.query(&IndexQuery::Or(vec![eq("a", 1), eq("b", 2)]))
                .is_none(),
            "a cross-attribute union must be declined, not answered from one column"
        );
        // Nested the same way `a IN [..] OR b IN [..]` arrives.
        assert!(
            idx.query(&IndexQuery::Or(vec![
                IndexQuery::Or(vec![eq("a", 1)]),
                IndexQuery::Or(vec![eq("b", 2)]),
            ]))
            .is_none(),
            "nesting must not smuggle a second attribute past the check"
        );
        // The same-attribute case still works.
        let mut same = ids(idx
            .query(&IndexQuery::Or(vec![eq("a", 1), eq("a", 2)]))
            .expect("one attribute is servable"));
        same.sort_unstable();
        assert_eq!(same, vec![10, 20]);
    }

    /// A union must not descend for a member until it is reached. Building every cursor up front
    /// makes `IN $list` pay one root-to-leaf descent per member — 100k of them, plus 100k live
    /// snapshots — before the first row, which a `LIMIT` then throws away.
    #[test]
    fn union_builds_no_cursor_before_the_first_row() {
        let attr = Arc::new("v".to_string());
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };
        let values: Vec<(Value, u64)> = (0..100u64).map(|i| (Value::Int(i as i64), i)).collect();
        let idx = NumericIndex::from_entries(values.iter().map(|(v, id)| (v, *id)));

        let q = IndexQuery::Or((0..100).map(eq).collect());
        let DocIter::Many(mut u) = idx.query(&q).expect("all-numeric union is servable") else {
            panic!("a union must produce DocIter::Many");
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
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };
        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(1), 1);
        idx.add(&Value::Int(2), 2);

        let mut it = idx
            .query(&IndexQuery::Or(vec![eq(1), eq(2)]))
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

    fn list(xs: &[Value]) -> Value {
        Value::List(Arc::new(xs.iter().cloned().collect::<ThinVec<_>>()))
    }

    /// The reason the array elements live in their own tree.
    ///
    /// A node whose `v` is the list `[1, 2]` must NOT be returned by `WHERE n.v = 1` — that is
    /// false in Cypher — and a node whose `v` is the scalar `1` must NOT be returned by
    /// `WHERE 1 IN n.v`. One shared key space cannot tell those apart, and `Equal` carries no
    /// post-filter to clean up after it (its plan is a bare `Node By Index Scan`), so the wrong
    /// row would reach the caller.
    #[test]
    fn a_scalar_and_a_list_element_do_not_share_a_key_space() {
        let attr = Arc::new("v".to_string());
        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(1), 10); // scalar 1
        idx.add(&list(&[Value::Int(1), Value::Int(2)]), 20); // list [1, 2]

        let eq = IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(1),
        };
        assert_eq!(
            ids(idx.query(&eq).expect("servable")),
            vec![10],
            "`n.v = 1` must match the scalar only, never the list that contains 1"
        );

        let contains = IndexQuery::ArrayContains {
            key: attr.clone(),
            value: Value::Int(1),
        };
        assert_eq!(
            ids(idx.query(&contains).expect("servable")),
            vec![20],
            "`1 IN n.v` must match the list only, never the scalar 1"
        );

        // A range over the scalar tree must not see list elements either.
        let rng = IndexQuery::Range {
            key: attr,
            min: Some(Value::Int(0)),
            max: Some(Value::Int(5)),
            include_min: true,
            include_max: true,
        };
        assert_eq!(ids(idx.query(&rng).expect("servable")), vec![10]);
    }

    /// Every numeric element is reachable, and a repeated element contributes one tuple — the
    /// tree stores `(key, doc)` as a set, so a removal must not depend on how many times the
    /// value happened to list the same number.
    #[test]
    fn every_element_is_indexed_once() {
        let attr = Arc::new("v".to_string());
        let ints = |xs: &[i64]| list(&xs.iter().map(|&x| Value::Int(x)).collect::<Vec<_>>());
        let mut idx = NumericIndex::new();
        idx.add(&ints(&[1, 1, 2]), 7);

        let contains = |v: i64| IndexQuery::ArrayContains {
            key: attr.clone(),
            value: Value::Int(v),
        };
        assert_eq!(ids(idx.query(&contains(1)).expect("servable")), vec![7]);
        assert_eq!(ids(idx.query(&contains(2)).expect("servable")), vec![7]);
        assert!(ids(idx.query(&contains(3)).expect("servable")).is_empty());

        // Removing the same value takes every element with it, duplicates included.
        idx.remove(&ints(&[1, 1, 2]), 7);
        assert!(idx.is_empty(), "both trees must be empty again");

        // The batch paths dedup too: one tuple per distinct element, removable in one go.
        let mut scratch = EncodedTuples::default();
        idx.add_batch([(ints(&[3, 3, 4]), 8)], &mut scratch);
        assert_eq!(idx.tuples(KeySpace::Array).count(), 2);
        idx.remove_batch([(ints(&[3, 3, 4]), 8)], &mut scratch);
        assert!(idx.is_empty());
    }

    /// Elements the encoder rejects are skipped one by one rather than rejecting the whole list:
    /// strings belong to a text kind, and temporals are not indexed at all, exactly as for a
    /// scalar.
    #[test]
    fn a_list_indexes_only_its_numeric_elements() {
        let attr = Arc::new("v".to_string());
        let mixed = list(&[
            Value::Int(4),
            Value::String(Arc::new("x".to_string())),
            Value::Date(0),
            Value::Float(f64::NAN),
            list(&[Value::Int(5)]),
        ]);
        let mut idx = NumericIndex::new();
        idx.add(&mixed, 3);
        let contains = |v: i64| IndexQuery::ArrayContains {
            key: attr.clone(),
            value: Value::Int(v),
        };
        assert_eq!(ids(idx.query(&contains(4)).expect("servable")), vec![3]);
        assert!(
            ids(idx.query(&contains(0)).expect("servable")).is_empty(),
            "a date element's raw number is not a key"
        );
        assert!(
            ids(idx.query(&contains(5)).expect("servable")).is_empty(),
            "a nested list is not looked inside"
        );
        assert_eq!(idx.tuples(KeySpace::Array).count(), 1);
    }

    /// The install rebuilds a column from its per-space tuple streams, so both trees must
    /// survive the trip — or a background-built column loses its array half.
    #[test]
    fn the_tuple_streams_round_trip_both_trees() {
        let attr = Arc::new("v".to_string());
        let mut idx = NumericIndex::new();
        idx.add(&Value::Int(1), 10);
        idx.add(&list(&[Value::Int(1)]), 20);

        let rebuilt = NumericIndex::from_sorted_tuples(
            idx.tuples(KeySpace::Scalar),
            idx.tuples(KeySpace::Array),
        );
        let eq = IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(1),
        };
        assert_eq!(ids(rebuilt.query(&eq).expect("servable")), vec![10]);
        let contains = IndexQuery::ArrayContains {
            key: attr,
            value: Value::Int(1),
        };
        assert_eq!(
            ids(rebuilt.query(&contains).expect("servable")),
            vec![20],
            "the array half must survive the round trip"
        );
    }
}
