//! The Range column: one `(label, attribute)` index, made of the kinds a RediSearch Range field is
//! made of — NUMERIC, TAG and GEO. Only the numeric kind exists so far; the others land as further
//! fields of [`RangeIndex`], each implementing [`RangeKind`].
//!
//! A Cypher property is not typed, so one column has to hold whatever the entities put in it. The
//! kinds are therefore not alternatives but **co-resident**, and a predicate is routed by the type
//! of its operand.
//!
//! Every kind's keys are `u64`, each in a tree of its own [`KeyKind`], so a column's tuples move
//! between threads and versions as [`EncodedTuples`] whatever their kind.
//!
//! Routing is exhaustive and total. Every predicate either reaches a kind, is answered empty
//! because it cannot match — a `NULL` operand, say, or a scalar predicate and a list predicate on
//! the same attribute — or is declined; and a decline is a hard error under `index-falkordb`,
//! because there is no other index to try. The three outcomes are kept apart deliberately: an empty
//! answer that should have been a decline is exactly how an unimplemented kind hides.

use std::iter;
use std::sync::Arc;

use rustc_hash::FxHashSet;

use super::encode::{KeyKind, encode_numeric};
use super::numeric::{DocIter, EncodedTuples, KeySet, NumericIndex};
use crate::index::IndexQuery;
use crate::runtime::value::Value;

/// What a Range column needs from each of its kinds to store values and look them up.
///
/// A kind owns its trees: the tuples a value is stored under, how a sorted batch of them goes in
/// and out, which entities a set of its keys reaches, and whether a value is one of them. The
/// column reaches the trees only through this trait. What it does itself is route a value or a
/// predicate to a kind ([`Kind::of`]) and build a [`Selection`] of the kind's keys from the query.
///
/// The scan checks ([`is_scalar_in`](Self::is_scalar_in), [`is_list_holding`](Self::is_list_holding))
/// take no kind: a scan standing in for a building column holds the [`Selection`], not the column.
pub trait RangeKind {
    /// A set of this kind's keys: what a predicate over its scalars selects.
    type Keys;

    /// Call `emit` with each tuple `value` is stored under in this kind, as the tree it goes in and
    /// its key. Nothing when the kind holds no part of `value`.
    fn for_each_stored_key(
        &self,
        value: &Value,
        emit: impl FnMut(KeyKind, u64),
    );

    /// Add this kind's share of `tuples`, which are sorted and unique
    /// ([`EncodedTuples::sort_dedup`]).
    fn add_encoded(
        &mut self,
        tuples: &EncodedTuples,
    );

    /// Remove this kind's share of `tuples`, which are sorted and unique.
    fn remove_encoded(
        &mut self,
        tuples: &EncodedTuples,
    );

    /// Whether this kind holds no tuples.
    fn is_empty(&self) -> bool;

    /// Heap bytes this kind's trees hold.
    fn memory_usage(&self) -> usize;

    /// The entities whose value is a scalar with a key in `keys`.
    fn scalars(
        &self,
        keys: Self::Keys,
    ) -> DocIter;

    /// The entities whose value is a list holding an element stored under `key`.
    fn lists_holding(
        &self,
        key: u64,
    ) -> DocIter;

    /// Whether `value` is one [`scalars`](Self::scalars) returns for `keys`: what a scan standing
    /// in for a column still building checks, so it keeps exactly what the built column returns.
    fn is_scalar_in(
        keys: &Self::Keys,
        value: &Value,
    ) -> bool;

    /// Whether `value` is one [`lists_holding`](Self::lists_holding) returns for `key`.
    fn is_list_holding(
        key: u64,
        value: &Value,
    ) -> bool;
}

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

/// What a column answers a predicate on one attribute with: the entities whose value it selects.
///
/// Built from a query without reading the column, so it serves twice: a `Ready` column looks it up
/// ([`RangeIndex::lookup`]), and a column that is still `Building` hands it to a scan that keeps
/// the entities whose value it [`matches`](Self::matches).
///
/// A stored value is a scalar or a list, never both, so a selection is of one or the other, and a
/// conjunction of a scalar predicate with a list predicate selects nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Scalar values with a key in the set: `=`, a range, `IN`, and conjunctions and unions of them.
    Scalars(KeySet),
    /// List values holding an element under `first` and under each of `rest`: `x IN n.v`, and
    /// conjunctions of it.
    ListsHolding { first: u64, rest: Vec<u64> },
}

impl Selection {
    /// Selects nothing.
    #[must_use]
    pub const fn empty() -> Self {
        Self::Scalars(KeySet::Empty)
    }

    /// Whether this selects nothing by construction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Scalars(KeySet::Empty))
    }

    /// Whether the column selects an entity holding `value`: the scan's check while the column
    /// is building, which must keep exactly the entities a lookup of the built column returns.
    #[must_use]
    pub fn matches(
        &self,
        value: &Value,
    ) -> bool {
        match self {
            Self::Scalars(keys) => NumericIndex::is_scalar_in(keys, value),
            Self::ListsHolding { first, rest } => iter::once(first)
                .chain(rest)
                .all(|&key| NumericIndex::is_list_holding(key, value)),
        }
    }

    /// The selection of a predicate on **one** attribute, or `None` when no kind serves it.
    #[must_use]
    pub fn of(query: &IndexQuery<Value>) -> Option<Self> {
        match query {
            IndexQuery::Equal { value, .. } => Self::equal(value),
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
            } if !items.is_empty() => Self::equal_to_any(items.iter()),
            IndexQuery::InList { .. } => None,
            IndexQuery::ArrayContains { value, .. } => Self::lists_holding(value),
            IndexQuery::Or(members) => Self::disjunction(members),
            IndexQuery::And(members) => Self::conjunction(members),
            // No geo kind yet.
            IndexQuery::Point { .. } => None,
        }
    }

    /// The scalars equal to `value`: `n.v = value`. A `NULL` matches nothing — `n.v = null` is
    /// unknown in Cypher — and is answered empty; a value no kind holds is declined.
    fn equal(value: &Value) -> Option<Self> {
        match Kind::of(value) {
            Some(Kind::Numeric) => Some(Self::Scalars(
                encode_numeric(value).map_or(KeySet::Empty, KeySet::Point),
            )),
            None if matches!(value, Value::Null) => Some(Self::empty()),
            None => None,
        }
    }

    /// The scalars equal to any of `values`: `n.v IN [..]`, and unions of equalities.
    ///
    /// Each value is routed as [`equal`](Self::equal) routes it, so `x IN [1, null]` matches only
    /// `1`. A value no kind holds declines the **whole** union: answering from the rest would
    /// silently drop the rows that value would have matched.
    fn equal_to_any<'a>(values: impl IntoIterator<Item = &'a Value>) -> Option<Self> {
        let mut keys = Vec::new();
        for value in values {
            match Self::equal(value)? {
                Self::Scalars(KeySet::Point(key)) => keys.push(key),
                // `NULL`, or a number with no key (`NaN`): it matches nothing, so adds no key.
                Self::Scalars(KeySet::Empty) => {}
                other => unreachable!("an equality selects one key or none, not {other:?}"),
            }
        }
        Some(Self::Scalars(KeySet::of_keys(keys)))
    }

    /// The selection of `n.v <op> bound`.
    ///
    /// Cypher orders numbers only with numbers, so a range with a `NULL` bound matches nothing and
    /// is answered **empty** rather than declined: that is not a gap in the index, it is the
    /// language. A bound of a type no kind holds is declined.
    fn range(
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> Option<Self> {
        let mut null_bound = false;
        for bound in [min, max].into_iter().flatten() {
            match Kind::of(bound) {
                Some(Kind::Numeric) => {}
                None if matches!(bound, Value::Null) => null_bound = true,
                None => return None,
            }
        }
        Some(if null_bound {
            Self::empty()
        } else {
            Self::Scalars(NumericIndex::range_keys(min, max, include_min, include_max))
        })
    }

    /// The selection of `value IN n.v`, where `n.v` is a list: the lists holding `value`.
    ///
    /// Only the numeric kind has an array tree so far: a probe of any other type (a `NULL`
    /// included) is declined and stays visible as a gap.
    fn lists_holding(value: &Value) -> Option<Self> {
        match Kind::of(value)? {
            Kind::Numeric => Some(match encode_numeric(value) {
                Some(first) => Self::ListsHolding {
                    first,
                    rest: Vec::new(),
                },
                None => Self::empty(),
            }),
        }
    }

    /// An `OR` of `=`s and `IN` lists, all on one attribute — `a = 1 OR a IN [2, 3]`. Nested `OR`s
    /// are flattened: the nesting carries no meaning for one column.
    ///
    /// Every member must target the same attribute: answering `a = 1 OR b = 2` from one column
    /// would look up both values in `a` and return wrong rows, not missing ones. Any other member —
    /// a range, an array-contains, an empty list, a conjunction — declines the union.
    pub(super) fn disjunction<'a>(
        members: impl IntoIterator<Item = &'a IndexQuery<Value>>
    ) -> Option<Self> {
        let members = leaves(members, |query| match query {
            IndexQuery::Or(nested) => Some(nested.as_slice()),
            _ => None,
        })?;
        if !on_one_attribute(&members) {
            return None;
        }
        let mut values = Vec::with_capacity(members.len());
        for member in members {
            match member {
                IndexQuery::Equal { value, .. } => values.push(value),
                IndexQuery::InList {
                    list: Value::List(items),
                    ..
                } if !items.is_empty() => values.extend(items.iter()),
                _ => return None,
            }
        }
        Self::equal_to_any(values)
    }

    /// A conjunction of predicates on **one** attribute, nested `AND`s flattened: what every
    /// conjunct selects ([`and`](Self::and)). On scalars that is arithmetic on keys
    /// ([`KeySet::and`]), so `n.v = 5 AND n.v > 2` reads one key and `n.v >= 0 AND n.v IN [5, 7]`
    /// reads two, not the window.
    ///
    /// A conjunct that selects nothing (a `NULL` operand) empties the conjunction, even beside one
    /// no kind serves: `n.v = null AND <anything>` holds for nothing. Otherwise a conjunct no kind
    /// serves declines the conjunction, and so does one on another attribute.
    pub(super) fn conjunction<'a>(
        members: impl IntoIterator<Item = &'a IndexQuery<Value>>
    ) -> Option<Self> {
        let conjuncts = leaves(members, |query| match query {
            IndexQuery::And(nested) => Some(nested.as_slice()),
            _ => None,
        })?;
        if !on_one_attribute(&conjuncts) {
            return None;
        }
        let selections: Vec<Option<Self>> = conjuncts.into_iter().map(Self::of).collect();
        if selections.iter().flatten().any(Self::is_empty) {
            return Some(Self::empty());
        }
        selections
            .into_iter()
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .reduce(Self::and)
    }

    /// What both select. A value is a scalar or a list, never both, so a scalar selection and a list
    /// selection select nothing together.
    fn and(
        self,
        other: Self,
    ) -> Self {
        match (self, other) {
            (Self::Scalars(a), Self::Scalars(b)) => Self::Scalars(a.and(b)),
            (
                Self::ListsHolding { first, mut rest },
                Self::ListsHolding {
                    first: other_first,
                    rest: other_rest,
                },
            ) => {
                rest.push(other_first);
                rest.extend(other_rest);
                Self::ListsHolding { first, rest }
            }
            _ => Self::empty(),
        }
    }
}

/// The leaves of `members`, with any combinator `nested` opens flattened into them:
/// `And(And(a, b), c)` constrains what `And(a, b, c)` does. `None` when `members`, or a combinator
/// nested in it, is empty: an empty `AND` or `OR` names no column.
fn leaves<'a>(
    members: impl IntoIterator<Item = &'a IndexQuery<Value>>,
    nested: fn(&IndexQuery<Value>) -> Option<&[IndexQuery<Value>]>,
) -> Option<Vec<&'a IndexQuery<Value>>> {
    fn push<'a>(
        members: impl IntoIterator<Item = &'a IndexQuery<Value>>,
        nested: fn(&IndexQuery<Value>) -> Option<&[IndexQuery<Value>]>,
        out: &mut Vec<&'a IndexQuery<Value>>,
    ) -> Option<()> {
        let before = out.len();
        for member in members {
            match nested(member) {
                Some(inner) => push(inner, nested, out)?,
                None => out.push(member),
            }
        }
        (out.len() > before).then_some(())
    }
    let mut out = Vec::new();
    push(members, nested, &mut out)?;
    Some(out)
}

/// Whether every one of `queries` constrains the same single attribute: what one column can answer.
fn on_one_attribute(queries: &[&IndexQuery<Value>]) -> bool {
    let mut attributes = Vec::new();
    queries
        .iter()
        .all(|query| attributes_of(query, &mut attributes))
        && attributes.windows(2).all(|pair| pair[0] == pair[1])
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

    /// The numeric kind, for tests that look at its trees.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn numeric(&self) -> &NumericIndex {
        &self.numeric
    }

    /// Call `emit` with each key a stored `value` takes in this column, and its [`KeyKind`]: a
    /// scalar lands in at most one kind, and a list's elements in the kinds' array trees.
    pub fn for_each_stored_key(
        &self,
        value: &Value,
        emit: impl FnMut(KeyKind, u64),
    ) {
        self.numeric.for_each_stored_key(value, emit);
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
    ///
    /// `scratch` is the caller's encode buffer and its contents are replaced. Taking it rather than
    /// allocating internally lets a commit that touches many columns allocate once and reuse the
    /// capacity, instead of a fresh buffer per column.
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

    /// Remove a batch of `(value, id)` entries. `scratch` as in [`add_batch`](Self::add_batch).
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
    /// kind into its own trees.
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        self.numeric.add_encoded(tuples);
    }

    /// Remove already-encoded tuples (a unit of work's collapsed removes).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        self.numeric.remove_encoded(tuples);
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
        match kind {
            KeyKind::Scalar => self.numeric.scalars(KeySet::Point(key)),
            KeyKind::Array => self.numeric.lists_holding(key),
        }
    }

    /// The entities `selection` selects in this column.
    #[must_use]
    pub fn lookup(
        &self,
        selection: Selection,
    ) -> DocIter {
        match selection {
            Selection::Scalars(keys) => self.numeric.scalars(keys),
            Selection::ListsHolding { first, rest } => {
                let lists = self.numeric.lists_holding(first);
                if rest.is_empty() {
                    return lists;
                }
                // Every element must be held. A probe yields docs in key order, not id order, so
                // this is a set probe rather than a merge: the first probe's lists are
                // materialised and each later probe can only narrow them.
                let mut found: FxHashSet<u64> = lists.collect();
                for element in rest {
                    if found.is_empty() {
                        break;
                    }
                    found = self
                        .numeric
                        .lists_holding(element)
                        .filter(|doc| found.contains(doc))
                        .collect();
                }
                DocIter::Set(found.into_iter())
            }
        }
    }

    /// Answer a predicate on this column, or decline it (`None`).
    #[must_use]
    pub fn query(
        &self,
        query: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        Selection::of(query).map(|selection| self.lookup(selection))
    }

    /// Every `(kind, key, doc)` tuple the column holds, in kind order. For tests that compare a
    /// column with what it must hold.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(KeyKind, u64, u64)> {
        self.numeric.tuples()
    }

    /// The scalars equal to the number `value`, reached as a query reaches them. For tests.
    #[cfg(test)]
    pub(crate) fn point(
        &self,
        value: &Value,
    ) -> DocIter {
        self.lookup(Selection::equal(value).expect("a numeric value"))
    }

    /// The scalars from `min` to `max`, numbers or unbounded, reached as a query reaches them. For
    /// tests.
    #[cfg(test)]
    pub(crate) fn range(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> DocIter {
        self.lookup(Selection::range(min, max, include_min, include_max).expect("numeric bounds"))
    }

    /// The lists holding the number `value`, reached as a query reaches them. For tests.
    #[cfg(test)]
    pub(crate) fn array_contains(
        &self,
        value: &Value,
    ) -> DocIter {
        self.lookup(Selection::lists_holding(value).expect("a numeric value"))
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

    fn contains(value: Value) -> IndexQuery<Value> {
        IndexQuery::ArrayContains { key: attr(), value }
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
        let scanned: Vec<u64> = values
            .iter()
            .filter(|(v, _)| selection.matches(v))
            .map(|&(_, id)| id)
            .collect();
        let mut looked_up: Vec<u64> = idx.lookup(selection).collect();
        looked_up.sort_unstable();
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
            (list(vec![Value::Int(1), Value::Int(7)]), 6),
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
        assert_eq!(ids(&idx, &contains(Value::Int(9)), &vs), vec![4]);
        assert_eq!(ids(&idx, &contains(Value::Int(1)), &vs), vec![4, 6]);
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
        assert!(
            idx.query(&range(
                Some(Value::Null),
                Some(Value::String(Arc::new("x".to_string())))
            ))
            .is_none(),
            "a bound no kind holds declines, even beside a NULL one"
        );
        assert!(idx.query(&contains(Value::Null)).is_none());
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
        assert!(
            idx.query(&IndexQuery::Or(vec![
                contains(Value::Int(1)),
                contains(Value::Int(9))
            ]))
            .is_none(),
            "a union of array probes is not a union of keys: a list holding both would come twice"
        );
    }

    /// Bounds on one attribute fold into one window and an `IN` list into a point set, so a
    /// conjunction on scalars stays one lookup; array probes intersect by entity.
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
        assert_eq!(
            Selection::of(&q),
            Some(Selection::Scalars(KeySet::Point(
                encode_numeric(&Value::Int(7)).unwrap()
            ))),
            "one key to read"
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
        // Array probes: a list must hold every probed element.
        let q = IndexQuery::And(vec![contains(Value::Int(1)), contains(Value::Int(9))]);
        assert_eq!(ids(&idx, &q, &vs), vec![4]);
        let q = IndexQuery::And(vec![contains(Value::Int(9)), contains(Value::Int(7))]);
        assert!(ids(&idx, &q, &vs).is_empty(), "no list holds both");
        // No value is both a scalar and a list.
        let q = IndexQuery::And(vec![eq(Value::Int(1)), contains(Value::Int(1))]);
        assert_eq!(Selection::of(&q), Some(Selection::empty()));
        assert!(ids(&idx, &q, &vs).is_empty());
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
