//! The Range column: one `(label, attribute)` index, made of the kinds a RediSearch Range field is
//! made of — NUMERIC, TAG and GEO. Only the numeric kind exists so far; the others land as further
//! fields of [`RangeIndex`] and further [`KeyKind`]s.
//!
//! A Cypher property is not typed, so one column has to hold whatever the entities put in it. The
//! kinds are therefore not alternatives but **co-resident**, and a predicate is routed by the type
//! of its operand.
//!
//! Every kind's keys are `u64`, each in a tree of its own [`KeyKind`], so a column's tuples move between threads and versions as
//! [`EncodedTuples`] whatever their kind.
//!
//! Routing is exhaustive and total. Every predicate either reaches a kind, is answered empty
//! because Cypher says it cannot match (a `NULL` operand, or a comparison across types), or is
//! declined — and a decline is a hard error under `index-falkordb`, because there is no other
//! index to try. The three outcomes are kept apart deliberately: an empty answer that should have
//! been a decline is exactly how an unimplemented kind hides.

use std::sync::Arc;

use rustc_hash::FxHashSet;

use super::encode::{KeyKind, encode_numeric, for_each_stored_key};
use super::numeric::{DocIter, EncodedTuples, KeySet, NumericIndex, Tree, UnionIter, empty_docs};
use crate::index::IndexQuery;
use crate::runtime::value::Value;

/// Which kind holds a scalar value, and therefore which kind answers a predicate over it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Numeric,
}

impl Kind {
    /// The kind that indexes `value`, or `None` for a value no kind holds (`NULL`, a string, a
    /// point, a temporal, a list, a map, a vector). A list is deliberately not a kind: its
    /// *elements* are indexed by the numeric kind's array tree, but the list itself is not a value
    /// any predicate can be routed by. Temporals are not indexed, as in the RediSearch path ([`encode_numeric`]).
    const fn of(value: &Value) -> Option<Self> {
        match value {
            Value::Int(_) | Value::Float(_) | Value::Bool(_) => Some(Self::Numeric),
            _ => None,
        }
    }
}

/// What a column answers a predicate with: the entities whose value it selects.
///
/// Built from a query without reading the column, so it serves twice: a `Ready` column looks it up
/// ([`RangeIndex::lookup`]), and a column that is still `Building` hands it to a scan that keeps
/// the entities whose value it [`matches`](Self::matches).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Keys of a tree whose key a value determines on its own: the numeric trees.
    Keys { kind: KeyKind, keys: KeySet },
    /// Every entity any member selects.
    Any(Vec<Selection>),
    /// Every entity all members select.
    All(Vec<Selection>),
}

impl Selection {
    /// Selects nothing.
    #[must_use]
    pub const fn empty() -> Self {
        Self::Keys {
            kind: KeyKind::Scalar,
            keys: KeySet::Empty,
        }
    }

    /// Whether this selects nothing by construction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(
            self,
            Self::Keys {
                keys: KeySet::Empty,
                ..
            }
        )
    }

    /// Whether the column selects an entity holding `value`: the scan's check while the column
    /// is building, which must keep exactly the entities a lookup of the built column returns.
    #[must_use]
    pub fn matches(
        &self,
        value: &Value,
    ) -> bool {
        match self {
            Self::Keys { kind, keys } => {
                let mut hit = false;
                for_each_stored_key(value, |k, key| hit |= k == *kind && keys.contains(key));
                hit
            }
            Self::Any(members) => members.iter().any(|m| m.matches(value)),
            Self::All(members) => members.iter().all(|m| m.matches(value)),
        }
    }

    /// The selection of `n.v = value`, routed by the value's type.
    ///
    /// A `NULL` operand is answered **empty**, not declined: `n.v = null` is unknown in Cypher and
    /// so matches nothing, whatever the column holds. A value no kind holds is declined (`None`).
    fn point(value: &Value) -> Option<Self> {
        Some(match Kind::of(value) {
            Some(Kind::Numeric) => Self::Keys {
                kind: KeyKind::Scalar,
                keys: NumericIndex::point_keys(value),
            },
            None if matches!(value, Value::Null) => Self::empty(),
            None => return None,
        })
    }

    /// The selection of `n.v <op> bound`, routed by the bounds' type.
    ///
    /// Cypher orders numbers with numbers; every other comparison is unknown and matches nothing.
    /// So a range whose bounds disagree on kind — or whose bound is a `NULL` — is answered **empty** rather than declined. That is not a gap in the
    /// index; it is the language. A bound of a type no kind holds is declined.
    fn range(
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> Option<Self> {
        // `Some(None)` for a `NULL` bound, `None` for a type no kind holds.
        let kind_of = |v: &Value| match v {
            Value::Null => Some(None),
            v => Kind::of(v).map(Some),
        };
        let kinds = [min, max]
            .into_iter()
            .flatten()
            .map(kind_of)
            .collect::<Option<Vec<_>>>()?;
        let kind = match kinds.as_slice() {
            // Unbounded on both sides: no operand to route by. The numeric tree answers it, as it
            // did before the other kinds existed.
            [] => Kind::Numeric,
            [Some(k)] => *k,
            [Some(a), Some(b)] if a == b => *a,
            // A `NULL` bound, or bounds of different kinds: nothing can hold.
            _ => return Some(Self::empty()),
        };
        Some(match kind {
            Kind::Numeric => Self::Keys {
                kind: KeyKind::Scalar,
                keys: NumericIndex::range_keys(min, max, include_min, include_max),
            },
        })
    }

    /// The selection of `value IN n.v`, where `n.v` is a list — a point in a kind's array tree.
    ///
    /// Only the numeric kind has one so far: a probe of any other type is declined and stays
    /// visible as a gap.
    fn array_contains(value: &Value) -> Option<Self> {
        match (Kind::of(value), value) {
            (Some(Kind::Numeric), _) => Some(Self::Keys {
                kind: KeyKind::Array,
                keys: NumericIndex::point_keys(value),
            }),
            _ => None,
        }
    }

    /// The selection of `n.v = v₁ OR n.v = v₂ OR ...`, one member per kind its values belong to.
    ///
    /// A `NULL` member contributes nothing (`x IN [1, null]` matches only `1`). A member whose type
    /// no kind holds declines the **whole** union: answering from the rest would silently drop the
    /// rows that member would have matched.
    fn values<'a>(values: impl IntoIterator<Item = &'a Value>) -> Option<Self> {
        let mut numeric = Vec::new();
        for value in values {
            match (Kind::of(value), value) {
                (Some(Kind::Numeric), _) => numeric.extend(encode_numeric(value)),
                (None, Value::Null) => {}
                _ => return None,
            }
        }
        let mut members = Vec::new();
        if !numeric.is_empty() {
            members.push(Self::Keys {
                kind: KeyKind::Scalar,
                keys: KeySet::distinct(numeric),
            });
        }
        Some(match members.len() {
            0 => Self::empty(),
            1 => members.pop().expect("one member"),
            _ => Self::Any(members),
        })
    }

    /// The selection of a predicate on **one** attribute, or `None` when no kind serves it.
    ///
    /// # Panics
    /// Never: a malformed `IN` (a non-list operand) is declined.
    #[must_use]
    pub fn of(query: &IndexQuery<Value>) -> Option<Self> {
        match query {
            IndexQuery::Equal { value, .. } => Self::point(value),
            IndexQuery::Range {
                min,
                max,
                include_min,
                include_max,
                ..
            } => Self::range(min.as_ref(), max.as_ref(), *include_min, *include_max),
            // An empty list is declined rather than answered as "no rows": the scan op drops the
            // members it cannot index before the list gets here, so `IN [date(..)]` arrives empty,
            // and "no rows" would miss every entity holding that date.
            IndexQuery::InList {
                list: Value::List(items),
                ..
            } if !items.is_empty() => Self::values(items.iter()),
            IndexQuery::InList { .. } => None,
            IndexQuery::Or(children) => Self::disjunction(&children.iter().collect::<Vec<_>>()),
            IndexQuery::And(children) => Self::conjunction(&children.iter().collect::<Vec<_>>()),
            IndexQuery::ArrayContains { value, .. } => Self::array_contains(value),
            // No geo kind yet.
            IndexQuery::Point { .. } => None,
        }
    }

    /// An `OR` of `Equal`s and `IN` lists, all on one attribute — `a = 1 OR a IN [2, 'x']`.
    ///
    /// An `OR` can name several attributes, and this is one column: answering `a = 1 OR b = 2`
    /// from it would look up both `1` and `2` in `a` and return wrong rows, not missing ones. So
    /// every member must target the same attribute; otherwise `None`, and the caller combines one
    /// column per attribute instead. Nested unions are flattened: the nesting carries no meaning
    /// for one column.
    pub(super) fn disjunction(members: &[&IndexQuery<Value>]) -> Option<Self> {
        fn collect<'a>(
            members: &[&'a IndexQuery<Value>],
            attr: &mut Option<&'a Arc<String>>,
            out: &mut Vec<&'a Value>,
        ) -> Option<()> {
            for member in members {
                let (key, values) = match member {
                    IndexQuery::Equal { key, value } => (key, std::slice::from_ref(value)),
                    IndexQuery::InList {
                        key,
                        list: Value::List(items),
                    } if !items.is_empty() => (key, items.as_slice()),
                    IndexQuery::Or(nested) if !nested.is_empty() => {
                        collect(&nested.iter().collect::<Vec<_>>(), attr, out)?;
                        continue;
                    }
                    // A range, an empty list, a conjunction — not this shape.
                    _ => return None,
                };
                if *attr.get_or_insert(key) != key {
                    return None;
                }
                out.extend(values);
            }
            Some(())
        }
        if members.is_empty() {
            return None;
        }
        let mut values = Vec::with_capacity(members.len());
        collect(members, &mut None, &mut values)?;
        Self::values(values)
    }

    /// A conjunction of predicates that all constrain **one** attribute, collapsed as far as it
    /// can be.
    ///
    /// Numeric conjuncts fold arithmetically into a single window: `WHERE n.v = 5 AND n.v > 2`
    /// arrives here because the planner's `merge_range_queries` gives up whenever two conjuncts
    /// specify the same side of the window — at plan time the bounds may still be expressions it
    /// cannot compare. A conjunction with a `NULL` operand is unsatisfiable, and is answered empty,
    /// not declined.
    ///
    /// A conjunct that does not fold but is servable on its own — a union, an array-contains — is
    /// kept as a member of an [`All`](Self::All).
    ///
    /// Returns `None` if a member targets another attribute or is a shape no kind can serve.
    pub(super) fn conjunction(children: &[&IndexQuery<Value>]) -> Option<Self> {
        #[derive(Default)]
        struct Fold<'a> {
            attr: Option<&'a Arc<String>>,
            numeric: Option<(u64, u64)>,
            /// The numeric keys an `IN` list (or a union of equalities) allows, intersected: a
            /// point set the window then narrows, so `v >= 0 AND v IN [5, 7]` reads two keys
            /// rather than every row of the window.
            points: Option<Vec<u64>>,
            unsatisfiable: bool,
            /// A conjunct no kind serves. It declines the conjunction unless another conjunct
            /// already makes it unsatisfiable: `n.v = null AND <anything>` holds for nothing.
            declined: bool,
            others: Vec<Selection>,
        }

        fn fold<'a>(
            children: &[&'a IndexQuery<Value>],
            acc: &mut Fold<'a>,
        ) -> Option<()> {
            for child in children {
                if let IndexQuery::And(nested) = child {
                    if nested.is_empty() {
                        return None;
                    }
                    fold(&nested.iter().collect::<Vec<_>>(), acc)?;
                    continue;
                }
                let mut keys = Vec::new();
                if !attributes_of(child, &mut keys) {
                    return None;
                }
                for key in keys {
                    if *acc.attr.get_or_insert(key) != key {
                        return None;
                    }
                }
                let Some(selection) = Selection::of(child) else {
                    acc.declined = true;
                    continue;
                };
                match selection {
                    Selection::Keys {
                        kind: KeyKind::Scalar,
                        keys,
                    } => {
                        let (lo, hi) = match keys {
                            KeySet::Point(k) => (k, k),
                            KeySet::Range(lo, hi) => (lo, hi),
                            KeySet::Empty => {
                                acc.unsatisfiable = true;
                                continue;
                            }
                            KeySet::Keys(keys) => {
                                acc.points = Some(match acc.points.take() {
                                    None => keys,
                                    Some(prev) => prev
                                        .into_iter()
                                        .filter(|k| keys.binary_search(k).is_ok())
                                        .collect(),
                                });
                                continue;
                            }
                        };
                        let (alo, ahi) = acc.numeric.unwrap_or((0, u64::MAX));
                        acc.numeric = Some((alo.max(lo), ahi.min(hi)));
                    }
                    other if other.is_empty() => acc.unsatisfiable = true,
                    other => acc.others.push(other),
                }
            }
            Some(())
        }

        if children.is_empty() {
            return None;
        }
        let mut acc = Fold::default();
        fold(children, &mut acc)?;
        acc.attr?; // an `And` of nothing but empty nested `And`s names no column

        if acc.unsatisfiable {
            return Some(Self::empty());
        }
        if acc.declined {
            return None;
        }
        // The folded keys first: they are read first, and the rest probe what they found. A point
        // set is narrowed by the window, not intersected with it as a doc set.
        let mut members = Vec::with_capacity(acc.others.len() + 1);
        let (lo, hi) = acc.numeric.unwrap_or((0, u64::MAX));
        match acc.points {
            Some(points) => members.push(Self::Keys {
                kind: KeyKind::Scalar,
                keys: match points
                    .into_iter()
                    .filter(|k| (lo..=hi).contains(k))
                    .collect::<Vec<_>>()
                {
                    keys if keys.is_empty() => KeySet::Empty,
                    keys => KeySet::Keys(keys),
                },
            }),
            None if acc.numeric.is_some() => members.push(Self::Keys {
                kind: KeyKind::Scalar,
                keys: if lo <= hi {
                    KeySet::Range(lo, hi)
                } else {
                    KeySet::Empty
                },
            }),
            None => {}
        }
        members.extend(acc.others);
        if members.iter().any(Self::is_empty) {
            return Some(Self::empty());
        }
        Some(match members.len() {
            // Every conjunct named the attribute, so each left a member or was unsatisfiable.
            0 => unreachable!("a conjunction with an attribute constrains it"),
            1 => members.pop().expect("one member"),
            _ => Self::All(members),
        })
    }
}

/// Collect the attributes a predicate constrains, flattening nested composites. `false` when the
/// shape names no attribute at all — an empty `And`/`Or`.
///
/// Deliberately says nothing about *values*: which types a column can serve is decided inside the
/// column, where the kinds live.
pub(super) fn attributes_of<'a>(
    query: &'a IndexQuery<Value>,
    out: &mut Vec<&'a Arc<String>>,
) -> bool {
    match query {
        IndexQuery::Equal { key, .. }
        | IndexQuery::Range { key, .. }
        | IndexQuery::ArrayContains { key, .. }
        | IndexQuery::Point { key, .. }
        | IndexQuery::InList { key, .. } => {
            out.push(key);
            true
        }
        IndexQuery::And(children) | IndexQuery::Or(children) => {
            !children.is_empty() && children.iter().all(|c| attributes_of(c, out))
        }
    }
}

/// One indexed `(label, attribute)`: the kinds of the same column.
///
/// `Clone` is `O(1)` — a root-`Arc` bump per tree — so a graph version forks its index snapshot
/// cheaply.
#[derive(Clone, Default)]
pub struct RangeIndex {
    numeric: NumericIndex,
}

impl RangeIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The numeric kind, for the tests that predate the other two.
    #[must_use]
    pub fn numeric(&self) -> &NumericIndex {
        &self.numeric
    }

    /// The numeric kind, to change.
    pub fn numeric_mut(&mut self) -> &mut NumericIndex {
        &mut self.numeric
    }

    /// The tree of `kind`.
    fn tree(
        &self,
        kind: KeyKind,
    ) -> &Tree {
        match kind {
            KeyKind::Scalar | KeyKind::Array => self.numeric.tree_ref(kind),
        }
    }

    /// The tree of `kind`, to change.
    fn tree_mut(
        &mut self,
        kind: KeyKind,
    ) -> &mut Tree {
        match kind {
            KeyKind::Scalar | KeyKind::Array => self.numeric.tree_mut(kind),
        }
    }

    /// Call `emit` with each key a stored `value` takes in this column, and its [`KeyKind`]: a
    /// scalar lands in at most one kind, and a list's elements in the kinds' array trees.
    pub fn for_each_stored_key(
        &self,
        value: &Value,
        emit: impl FnMut(KeyKind, u64),
    ) {
        for_each_stored_key(value, emit);
    }

    /// Append the tuples `(value, id)` takes in this column to `out`, each with the tuples of its
    /// kind. Unsorted and possibly repeated: see [`EncodedTuples::sort_dedup`].
    pub fn push_encoded(
        &self,
        value: &Value,
        id: u64,
        out: &mut EncodedTuples,
    ) {
        self.for_each_stored_key(value, |kind, key| out[kind].push((key, id)));
    }

    /// Index `id` under `value`.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        let mut tuples = EncodedTuples::default();
        self.push_encoded(value, id, &mut tuples);
        self.add_encoded(&mut tuples);
    }

    /// Remove `id` from under `value` — the old value on delete/update. Routes on the **old**
    /// value's kind, or the tuple is stranded in the kind nothing will look at again.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        let mut tuples = EncodedTuples::default();
        self.push_encoded(value, id, &mut tuples);
        self.remove_encoded(&mut tuples);
    }

    /// Add a batch of `(value, id)` entries — one batched tree op per kind.
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        scratch.clear();
        for (value, id) in entries {
            self.push_encoded(&value, id, scratch);
        }
        self.add_encoded(scratch);
    }

    /// Remove a batch of `(value, id)` entries.
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        scratch.clear();
        for (value, id) in entries {
            self.push_encoded(&value, id, scratch);
        }
        self.remove_encoded(scratch);
    }

    /// Add already-encoded tuples (a unit of work's collapsed adds, or an indexer batch), each
    /// kind into its own tree.
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        for kind in KeyKind::ALL {
            if tuples[kind].is_empty() {
                continue;
            }
            let tree = self.tree_mut(kind);
            if tree.is_empty() {
                *tree = Tree::from_sorted(&tuples[kind]);
            } else {
                tree.insert_batch(&tuples[kind]);
            }
        }
    }

    /// Remove already-encoded tuples (a unit of work's collapsed removes).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        for kind in KeyKind::ALL {
            if !tuples[kind].is_empty() {
                self.tree_mut(kind).remove_batch(&tuples[kind]);
            }
        }
    }

    /// Whether the column holds no tuples in any kind.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.numeric.is_empty()
    }

    /// Heap bytes every kind's trees hold.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.numeric.memory_usage()
    }

    /// The scalar tree and key `value` takes, or `None` when no kind indexes it as a scalar (a
    /// list: its elements are in the array trees, which say which lists contain a value, not
    /// which equal a list).
    #[must_use]
    pub fn encode_value(
        &self,
        value: &Value,
    ) -> Option<(KeyKind, u64)> {
        match Kind::of(value)? {
            Kind::Numeric => encode_numeric(value).map(|k| (KeyKind::Scalar, k)),
        }
    }

    /// The ids stored under `key` in the tree of `kind`.
    #[must_use]
    pub fn rows_at(
        &self,
        kind: KeyKind,
        key: u64,
    ) -> DocIter {
        DocIter::One(self.tree(kind).point(key))
    }

    /// The windows `selection` reads in this column, each naming its tree by position in
    /// `trees`, or `None` when it is not a plain union of key windows (an intersection).
    fn windows(
        &self,
        selection: &Selection,
        trees: &mut Vec<KeyKind>,
        out: &mut Vec<(usize, u64, u64)>,
    ) -> Option<()> {
        let mut tree_of = |kind: KeyKind| match trees.iter().position(|&k| k == kind) {
            Some(i) => i,
            None => {
                trees.push(kind);
                trees.len() - 1
            }
        };
        match selection {
            Selection::Keys { kind, keys } => {
                let t = tree_of(*kind);
                match keys {
                    KeySet::Empty => {}
                    KeySet::Point(k) => out.push((t, *k, *k)),
                    KeySet::Range(lo, hi) => out.push((t, *lo, *hi)),
                    KeySet::Keys(keys) => out.extend(keys.iter().map(|&k| (t, k, k))),
                }
            }
            Selection::Any(members) => {
                for member in members {
                    self.windows(member, trees, out)?;
                }
            }
            Selection::All(_) => return None,
        }
        Some(())
    }

    /// The entities `selection` selects in this column.
    ///
    /// A union of key windows is one lazy chain over the trees it touches. An intersection is a
    /// set probe rather than a sorted merge: each member yields docs in *value* order, not id
    /// order, so the first member is materialised and the rest probe it.
    #[must_use]
    pub fn lookup(
        &self,
        selection: &Selection,
    ) -> DocIter {
        match selection {
            Selection::Keys { kind, keys } => {
                let tree = self.tree(*kind);
                match keys {
                    KeySet::Empty => empty_docs(tree),
                    KeySet::Point(k) => DocIter::One(tree.point(*k)),
                    KeySet::Range(lo, hi) => DocIter::One(tree.range(*lo, *hi)),
                    _ => self.chain(selection),
                }
            }
            Selection::Any(_) => self.chain(selection),
            Selection::All(members) => {
                let mut members = members.iter();
                let mut docs: FxHashSet<u64> = members
                    .next()
                    .map_or_else(FxHashSet::default, |m| self.lookup(m).collect());
                for member in members {
                    if docs.is_empty() {
                        break; // a later conjunct can only remove rows
                    }
                    // Probe rather than materialise: the set never grows past what the first
                    // member found, however many rows a later member matches.
                    docs = self
                        .lookup(member)
                        .filter(|doc| docs.contains(doc))
                        .collect();
                }
                DocIter::Set(docs.into_iter())
            }
        }
    }

    /// A union: one lazy chain when every member is a set of key windows, else a materialised
    /// set, since a doc an intersection member selects may also be selected by another member.
    fn chain(
        &self,
        selection: &Selection,
    ) -> DocIter {
        let (mut kinds, mut windows) = (Vec::new(), Vec::new());
        if self.windows(selection, &mut kinds, &mut windows).is_some() {
            let trees = kinds.into_iter().map(|k| self.tree(k).clone()).collect();
            return DocIter::Many(UnionIter::new(trees, windows));
        }
        let Selection::Any(members) = selection else {
            unreachable!("only a union holds an intersection")
        };
        let docs: FxHashSet<u64> = members.iter().flat_map(|m| self.lookup(m)).collect();
        DocIter::Set(docs.into_iter())
    }

    /// Answer a predicate on this column, or decline it (`None`).
    #[must_use]
    pub fn query(
        &self,
        query: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        Selection::of(query).map(|selection| self.lookup(&selection))
    }

    /// Every `(kind, key, doc)` tuple the column holds, in kind order. For tests that compare a
    /// column with what it must hold.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(KeyKind, u64, u64)> {
        KeyKind::ALL
            .into_iter()
            .flat_map(|kind| {
                self.tree(kind)
                    .range_tuples(0, u64::MAX)
                    .map(move |(key, doc)| (kind, key, doc))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thin_vec::ThinVec;

    fn attr() -> Arc<String> {
        Arc::new("v".to_string())
    }

    fn eq(value: Value) -> IndexQuery<Value> {
        IndexQuery::Equal { key: attr(), value }
    }

    fn range(
        min: Option<Value>,
        max: Option<Value>,
    ) -> IndexQuery<Value> {
        IndexQuery::Range {
            key: attr(),
            min,
            max,
            include_min: true,
            include_max: true,
        }
    }

    fn list(values: Vec<Value>) -> Value {
        Value::List(Arc::new(values.into_iter().collect::<ThinVec<_>>()))
    }

    /// The rows of a servable predicate, sorted — checked against the scan's answer, so a
    /// selection that a lookup and a scan disagree on fails here.
    fn ids(
        idx: &RangeIndex,
        q: &IndexQuery<Value>,
        values: &[(Value, u64)],
    ) -> Vec<u64> {
        let selection = Selection::of(q).expect("servable");
        let mut looked_up: Vec<u64> = idx.lookup(&selection).collect();
        looked_up.sort_unstable();
        let scanned: Vec<u64> = values
            .iter()
            .filter(|(v, _)| selection.matches(v))
            .map(|&(_, id)| id)
            .collect();
        assert_eq!(
            looked_up, scanned,
            "a lookup and a scan must agree on {q:?}"
        );
        looked_up
    }

    fn sample() -> (RangeIndex, Vec<(Value, u64)>) {
        let values = vec![
            (Value::Int(1), 1),
            (Value::Float(2.5), 2),
            (Value::Int(7), 3),
            (list(vec![Value::Int(1), Value::Int(9)]), 4),
            (Value::String(Arc::new("x".to_string())), 5),
        ];
        let mut idx = RangeIndex::new();
        for (v, id) in &values {
            idx.add(v, *id);
        }
        (idx, values)
    }

    #[test]
    fn points_ranges_and_array_probes_agree_with_the_scan() {
        let (idx, vs) = sample();
        assert_eq!(
            ids(&idx, &eq(Value::Int(1)), &vs),
            vec![1],
            "the scalar only"
        );
        assert_eq!(ids(&idx, &eq(Value::Float(2.5)), &vs), vec![2]);
        assert_eq!(
            ids(&idx, &range(Some(Value::Int(0)), Some(Value::Int(5))), &vs),
            vec![1, 2]
        );
        let contains = IndexQuery::ArrayContains {
            key: attr(),
            value: Value::Int(9),
        };
        assert_eq!(ids(&idx, &contains, &vs), vec![4]);
    }

    /// `NULL` operands are answered empty — `n.v = null` is unknown in Cypher. A value no kind
    /// holds is declined, which keeps a missing kind visible instead of masquerading as "nothing
    /// matched".
    #[test]
    fn null_answers_empty_and_an_unheld_type_declines() {
        let (idx, vs) = sample();
        assert!(ids(&idx, &eq(Value::Null), &vs).is_empty());
        assert!(ids(&idx, &range(Some(Value::Null), None), &vs).is_empty());
        assert!(
            idx.query(&eq(Value::String(Arc::new("x".to_string()))))
                .is_none()
        );
        assert!(idx.query(&eq(list(vec![Value::Int(1)]))).is_none());
    }

    /// A union drops `NULL` members, dedups repeated ones, and declines whole if a member's type
    /// has no kind.
    #[test]
    fn a_union_of_values() {
        let (idx, vs) = sample();
        let q = IndexQuery::Or(vec![eq(Value::Int(1)), eq(Value::Null), eq(Value::Int(7))]);
        assert_eq!(ids(&idx, &q, &vs), vec![1, 3]);
        let in_list = IndexQuery::InList {
            key: attr(),
            list: list(vec![Value::Int(7), Value::Int(7)]),
        };
        assert_eq!(ids(&idx, &in_list, &vs), vec![3]);
        assert!(
            idx.query(&IndexQuery::Or(vec![
                eq(Value::Int(1)),
                eq(Value::String(Arc::new("x".to_string())))
            ]))
            .is_none()
        );
    }

    /// Bounds on one attribute fold into one window; a union inside a conjunction is intersected
    /// as a doc set.
    #[test]
    fn conjunctions_fold_and_intersect() {
        let (idx, vs) = sample();
        let q = IndexQuery::And(vec![
            range(Some(Value::Int(0)), None),
            range(None, Some(Value::Int(2))),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![1]);
        let q = IndexQuery::And(vec![
            IndexQuery::Or(vec![eq(Value::Int(1)), eq(Value::Int(7))]),
            range(Some(Value::Int(5)), None),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![3]);
        let q = IndexQuery::And(vec![eq(Value::Int(1)), eq(Value::Null)]);
        assert!(ids(&idx, &q, &vs).is_empty());
        // A `NULL` conjunct empties the conjunction even beside one no kind serves.
        let q = IndexQuery::And(vec![eq(Value::Null), eq(list(vec![Value::Int(1)]))]);
        assert!(ids(&idx, &q, &vs).is_empty());
        let q = IndexQuery::And(vec![eq(Value::Int(1)), eq(list(vec![Value::Int(1)]))]);
        assert!(
            idx.query(&q).is_none(),
            "without one, that conjunct declines it"
        );
        // An `IN` list folds into the window as a point set: it reads its keys, not the window.
        let in_list = |vals: Vec<Value>| IndexQuery::InList {
            key: attr(),
            list: list(vals),
        };
        let q = IndexQuery::And(vec![
            range(Some(Value::Int(2)), None),
            in_list(vec![Value::Int(1), Value::Int(7)]),
        ]);
        assert!(
            matches!(
                Selection::of(&q),
                Some(Selection::Keys {
                    kind: KeyKind::Scalar,
                    keys: KeySet::Keys(_)
                })
            ),
            "{:?}",
            Selection::of(&q)
        );
        assert_eq!(ids(&idx, &q, &vs), vec![3]);
        let q = IndexQuery::And(vec![
            in_list(vec![Value::Int(1), Value::Int(7)]),
            in_list(vec![Value::Int(7), Value::Int(9)]),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![3], "two lists intersect");
        let q = IndexQuery::And(vec![
            range(Some(Value::Int(8)), None),
            in_list(vec![Value::Int(1), Value::Int(7)]),
        ]);
        assert!(
            ids(&idx, &q, &vs).is_empty(),
            "the window excludes every key"
        );
    }

    /// Batch maintenance must agree with the one-at-a-time path.
    #[test]
    fn batch_ops_match_single_ops() {
        let entries = vec![
            (Value::Int(3), 1u64),
            (list(vec![Value::Int(4), Value::Int(5)]), 2),
            (Value::Null, 3),
        ];
        let mut scratch = EncodedTuples::default();
        let mut batched = RangeIndex::new();
        batched.add_batch(entries.iter().cloned(), &mut scratch);
        let mut singly = RangeIndex::new();
        for (v, id) in &entries {
            singly.add(v, *id);
        }
        assert_eq!(batched.tuples(), singly.tuples());
        assert_eq!(batched.tuples().len(), 3);
        batched.remove_batch(entries.iter().cloned(), &mut scratch);
        assert!(batched.is_empty());
    }
}
